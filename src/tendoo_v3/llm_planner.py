"""
src/tendoo_v3/llm_planner.py

LLM Creative Director & Template Router cho Tendoo v3:
======================================================
- Kết nối tới OpenAI-compatible API endpoint (vLLM / sidecar server nội bộ).
- Mô hình mặc định mạnh nhất: `Qwen/Qwen3.8-27B` (hoặc `google/gemma-4-31B-it`, `Qwen/Qwen3.6-35B-A3B`).
- LLM đóng vai trò Giám đốc Sáng tạo (Creative Director):
  + Tiếp nhận form người dùng đã điền (có thể thiếu hoặc thừa).
  + Tiếp nhận câu lệnh Prompt tự do (ý đồ sáng tạo, phong cách, mong muốn bổ sung).
  + Lựa chọn template tối ưu nhất từ danh mục Catalog (`split_left`, `split_right`, `sandwich_top_heavy`,
    `lifestyle_corner_pod`, `diagonal_slash`, `step_process_roadmap`, `customer_feedback_card`...).
  + Xuất ra JSON có cấu trúc `TendooCreativePlan` chuẩn xác.
- Phòng vệ nâng cao:
  + Tắt thinking mode của Qwen3 (`chat_template_kwargs: {"enable_thinking": false}`).
  + Bộ quét ngoặc cân bằng `{...}` triệt tiêu text rác bên ngoài JSON.
  + Cơ chế Fallback Heuristic tự động khi mất kết nối mạng / chưa điền token.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

from tendoo_v3.catalog import (
    build_llm_catalog_prompt,
    build_llm_effect_and_vfx_prompt,
    build_llm_font_prompt,
    TEMPLATE_CATALOG,
)
from tendoo_v3.schema import StyleConfig, TendooCreativePlan

logger = logging.getLogger("TendooV3.LLMPlanner")

# Nạp .env từ cả thư mục src/tendoo_v3/.env lẫn thư mục gốc của repo
_v3_dir = Path(__file__).resolve().parent
_root_dir = _v3_dir.parent.parent
load_dotenv(_v3_dir / ".env", override=False)
load_dotenv(_root_dir / ".env", override=False)

# Cấu hình môi trường OpenAI-compatible
LLM_BASE_URL = os.environ.get("TENDOO_V3_LLM_BASE_URL", "http://10.221.155.3:8004/v1")
LLM_API_KEY = os.environ.get("TENDOO_V3_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("TENDOO_V3_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")
LLM_TIMEOUT_S = float(os.environ.get("TENDOO_V3_LLM_TIMEOUT_S", "60"))

SYSTEM_PROMPT = f"""Bạn là Giám đốc Nghệ thuật & Sáng tạo (Creative Director) hàng đầu của Tendoo AI Studio.
Nhiệm vụ của bạn là tiếp nhận thông tin từ form người dùng + câu lệnh tự do (freeform prompt) để tạo nên một kế hoạch sáng tạo poster hoàn hảo.

1. QUY TẮC BẮT BUỘC CHỌN TEMPLATE & ĐỊNH VỊ KHÔNG GIAN (Template & Spatial Routing):
   - Khi prompt yêu cầu chữ/nội dung gom ở góc, capsule: BẮT BUỘC CHỌN `template = "lifestyle_corner_pod"`.
     Xác định trường `orientation`:
     + Góc trên bên trái, trên trái, top left -> `"top_left"`
     + Góc trên bên phải, trên phải, top right -> `"top_right"`
     + Góc dưới bên trái, dưới trái, bottom left -> `"bottom_left"`
     + Góc dưới bên phải, dưới phải, bottom right -> `"bottom_right"`
   - Khi prompt yêu cầu thông tin cửa hàng / hotline đặt trên cùng, hoặc dồn chữ xuống dưới đáy:
     BẮT BUỘC CHỌN `template = "sandwich_bottom_heavy"` (thanh thông tin cửa hàng kẹp trên đỉnh, tiêu đề lớn và CTA dồn xuống đáy).
   - Khi prompt yêu cầu tiêu đề lớn ở trên cùng, thông tin cửa hàng ở dưới đáy:
     CHỌN `template = "sandwich_top_heavy"`.
   - Khi prompt yêu cầu chia đôi bên trái (chữ cột trái, ảnh bên phải): `template = "split_left"`.
   - Khi prompt yêu cầu chia đôi bên phải (ảnh bên trái, chữ cột phải): `template = "split_right"`.
   - Khi prompt yêu cầu cắt chéo, phong cách thể thao, fintech, năng động: `template = "diagonal_slash"` (orientation: 'left' hoặc 'right').
   - Khi nội dung là quy trình, các bước, lộ trình: `template = "step_process_roadmap"`.
   - Khi nội dung là review, đánh giá khách hàng: `template = "customer_feedback_card"`.
   - Khi nội dung là tuyển dụng: `template = "recruitment_board"`.
   - Khi so sánh trước/sau: `template = "before_after_split"`.

2. PHÂN BỔ CÁC THÀNH PHẦN CHỮ (Hero, Subhead, Badge, Pills tính năng, CTA, Hotline/Địa chỉ...):
   - Mọi câu chữ người dùng muốn hiển thị (ví dụ: dòng chữ trong ngoặc kép "Mì số 1 việt nam") PHẢI ĐƯỢC ĐƯA VÀO TRƯỜNG `hero`, `subhead`, `badge`...
   - Đúng chính tả tiếng Việt, có dấu đầy đủ, viết hoa tiêu đề Hero chuẩn mực thương mại.

3. QUY TẮC BẤT DI BẤT DỊCH VỀ ZERO-TEXT BACKGROUND (TUYỆT ĐỐI KHÔNG CHỮ TRONG SCENE_PROMPT & CORRIDOR_PROMPT):
   - Mô hình khuếch tán (DiT Base 4B) CHỈ DÙNG ĐỂ SINH ẢNH NỀN SẢN PHẨM KHÔNG CHỮ (100% Zero-Text Background). Toàn bộ chữ sẽ do hệ thống VAE/HTML vẽ đồ họa đè lên sau đó.
   - NẾU trong `scene_prompt` xuất hiện chữ (ví dụ: "dòng chữ Mì số 1", "tiêu đề Trà đào", "thông tin cửa hàng đặt trên cùng"), DiT Base 4B sẽ bị nhiễu và vẽ chữ rác vỡ nát làm hỏng toàn bộ poster!
   - QUY TẮC BẮT BUỘC CHO `scene_prompt`:
     + CHỈ mô tả vật thể thực tế, món ăn, thức uống, bối cảnh studio, ánh sáng, góc máy quay macro/85mm, độ sâu trường ảnh (shallow depth of field), màu sắc điện ảnh.
     + TUYỆT ĐỐI KHÔNG ghi nội dung chữ cần vẽ, KHÔNG nhắc đến "dòng chữ", "slogan", "tiêu đề", "chữ nằm ở...", "thông tin cửa hàng".
     + KHÔNG BAO GIỜ chép nguyên văn User Prompt vào `scene_prompt` nếu prompt đó chứa yêu cầu chữ hoặc bố cục!
     + Luôn kết thúc bằng: ", commercial studio photography, professional lighting, shallow depth of field, zero text, clean background".
   - CÔNG THỨC KHOẢNG TRỐNG TỰ NHIÊN CHO `corridor_prompt` (Contextual Negative Space):
     + Mô tả corridor như bề mặt mờ ảo tự nhiên của bối cảnh chính trong trường độ sâu (extreme bokeh, shallow depth of field, smooth clean table/wall texture, zero objects or text, matching scene colors).

4. VÍ DỤ MINH HỌA CỤ THỂ (Few-shot Ground Truth Examples):
   * Ví dụ 1:
     User Prompt: "tạo ảnh quảng cáo mỳ hảo hảo, dòng chữ: 'Mì số 1 việt nam nằm góc trên bên trái'"
     -> `template`: "lifestyle_corner_pod"
     -> `orientation`: "top_left"
     -> `hero`: "MÌ SỐ 1 VIỆT NAM"
     -> `subhead`: "Hương vị tôm chua cay đậm đà thơm ngon nức tiếng"
     -> `scene_prompt`: "Commercial food photography of an appetizing steaming bowl of Vietnamese sour and spicy noodles with fresh juicy prawns, sliced red chili, fragrant herbs, rich golden broth splashing subtly, soft studio rim lighting, 85mm macro lens f/2.0, cinematic food styling, zero text, clean background"
     -> `corridor_prompt`: "Clean rustic textured tabletop in extreme creamy bokeh, warm atmospheric studio light, completely empty negative space without any props or text, matching scene colors"
   * Ví dụ 2:
     User Prompt: "Tạo ảnh quảng cáo trà đào cam sả, thông tin cửa hàng đặt trên cùng"
     -> `template`: "sandwich_bottom_heavy"
     -> `orientation`: null
     -> `hero`: "TRÀ ĐÀO CAM SẢ"
     -> `subhead`: "Thanh mát tự nhiên đậm vị đào giòn sảng khoái"
     -> `scene_prompt`: "High-end commercial beverage photography of a tall glass of refreshing iced peach lemongrass tea with sliced orange wheels and fresh peach slices, condensation glistening on the glass, soft golden natural light, blurred cafe background, 85mm lens, commercial aesthetic, zero text, no letters"
     -> `corridor_prompt`: "Smooth light wooden table in deep soft bokeh, gentle diffused sunbeams, clean negative space without any objects or text, matching beverage warmth"

5. QUY TẮC QUYỀN UY & TRƯỜNG THÔNG TIN (Prompt Authority & Strict Suppression Rules):
   - QUY TẮC THỨ BẬC: Câu lệnh tự do của người dùng (User Prompt) LUÔN CÓ QUYỀN ƯU TIÊN CAO NHẤT (User Prompt > Form fields).
   - Nếu trong prompt người dùng yêu cầu KHÔNG vẽ hoặc BỎ một thông tin nào (ví dụ: "đừng vẽ thông tin cửa hàng", "không cần hotline", "không ghi giá", "bỏ nút CTA"), bạn BẮT BUỘC gán trường đó thành `null` (hoặc `[]`), KỂ CẢ KHI TRONG FORM CÓ ĐIỀN THÔNG TIN ĐÓ!
   - Nếu CẢ TRONG PROMPT LẪN FORM ĐỀU KHÔNG CÓ thông tin đó: BẮT BUỘC gán `null` (hoặc `[]`), TUYỆT ĐỐI KHÔNG tự bịa hoặc hallucinate các câu slogan/hotline/giá mẫu (như không tự bịa "HOT DEAL", "1900 xxxx", "Tendoo Audio Lab" nếu người dùng làm poster về đồ ăn/thức uống).
   - MÃ QR (`qr_code`): CHỈ điền khi người dùng thực sự yêu cầu hiển thị mã QR. Nếu prompt/form không nhắc gì đến QR, để `qr_code: null`, `qr_label: null`.

6. ĐIỀU PHỐI PHONG CÁCH (Style, Font, Text Effect, Cinematic VFX):
   - Chọn phông chữ (`style.font`) phù hợp với ngành hàng.
   - Chọn hiệu ứng chữ (`style.text_effect`) phù hợp với chất liệu và ánh sáng.
   - Chọn hiệu ứng điện ảnh quang học (`style.vfx`): 'film_grain', 'anamorphic_flare', 'gold_dust', 'light_leak', 'cinematic_haze', 'none'.

7. BẢNG ÁNH XẠ TRƯỜNG DỮ LIỆU TỪ FORM (Form Fields Semantic Mapping):
   - Khuyến Mại (promo): `title` -> `hero`, `discount` -> `badge`, `applied_product` -> `subhead`, ngày tháng -> `extra_texts`.
   - Giới Thiệu Sản Phẩm (product_intro): `title` hoặc `product_name` -> `hero`, `price` -> `badge`, `product_desc` -> `subhead`, `highlights` -> `extra_texts`.
   - Khai Trương (opening): `title` -> `hero`, `opening_promo` -> `badge`, `opening_date` + `booking_contact` -> `subhead` hoặc `extra_texts`.
   - Đánh Giá (feedback): `feedback_target` hoặc `title` -> `hero`, `feedback_quote` -> `testimonial`, `customer_name` -> `reviewer_name`, `feedback_rating` -> `rating`, `special_offer` -> `badge`. Template: 'customer_feedback_card'.
   - Tuyển Dụng (recruitment): `title` hoặc `job_position` -> `hero`, `job_desc` -> `subhead`, `apply_deadline` + `apply_method` -> `extra_texts`. Template: 'recruitment_board'.
   - Quy Trình / Hướng Dẫn (guide): `title` -> `hero`, `guide_steps` -> `steps`. Template: 'step_process_roadmap'.

{build_llm_catalog_prompt()}

{build_llm_font_prompt()}

{build_llm_effect_and_vfx_prompt()}

QUY TẮC BẮT BUỘC VỀ ĐẦU RA:
- Chỉ trả về DUY NHẤT một đối tượng JSON hợp lệ, KHÔNG bọc trong markdown ```json, KHÔNG kèm lời chào, KHÔNG có thẻ <think>.
- Bắt buộc tuân thủ đúng định dạng JSON mẫu sau:
{{
  "template": "split_right",
  "hero": "CÀ PHÊ PHA PHIN ĐẬM VỊ",
  "subhead": "Hạt cà phê Robusta rang mộc truyền thống Buôn Ma Thuột",
  "badge": "NGUYÊN CHẤT 100%",
  "tag_left": null,
  "tag_right": null,
  "rating": null,
  "extra_texts": ["Hương thơm nồng nàn", "Rang củi thủ công"],
  "cta": "THƯỞNG THỨC NGAY",
  "store_info": null,
  "qr_code": null,
  "qr_label": null,
  "testimonial": null,
  "reviewer_name": null,
  "steps": [],
  "orientation": null,
  "scene_prompt": "Cinematic close-up shot of a traditional Vietnamese metal coffee filter dripping rich dark espresso on a dark rustic wooden table, morning golden hour sunlight cutting through soft steam, high-end food photography, rich textures, 85mm lens f/1.8, zero text, clean background",
  "corridor_prompt": "Smooth rustic dark wooden table surface in extreme bokeh, soft diffused golden morning light, warm atmospheric blur, completely clean negative space without any cups or objects, identical color temperature and mood",
  "style": {{
    "font": "playfair",
    "theme_color": "#C88A35",
    "text_effect": "embossed",
    "background_tone": "warm_rustic",
    "vfx": "film_grain"
  }}
}}
"""


def extract_balanced_json(text: str) -> Optional[Dict[str, Any]]:
    """Trích xuất JSON an toàn bằng thuật toán quét ngoặc cân bằng (Balanced Braces)."""
    if not text:
        return None
    # Loại bỏ thẻ think nếu có
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    start_idx = cleaned.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    json_str = cleaned[start_idx : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
    return None


def sanitize_scene_prompt(raw_prompt: str, subject_hint: str = "") -> str:
    """Loại bỏ 100% các từ ngữ liên quan đến text, chữ, slogan, bố cục và quotes khỏi scene_prompt.
    Đảm bảo đầu ra cho DiT Base 4B là ZERO-TEXT BACKGROUND (Tuyệt đối không có chữ trên ảnh nền).
    """
    if not raw_prompt:
        raw_prompt = subject_hint or "Commercial studio product photography"

    text = raw_prompt.strip()

    # 1. Bóc tách và xóa các cụm text trong ngoặc kép hoặc sau 'dòng chữ'
    text = re.sub(r'(?:dòng\s+)?chữ\s*[:=]?\s*["\'“](.*?)["\'”]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'["\'“].*?["\'”]', '', text)

    # 2. Xóa các chỉ dẫn bố cục & vai trò
    patterns_to_remove = [
        r'tạo\s+ảnh\s*(?:quảng\s+cáo|poster|banner)?\s*',
        r'ảnh\s*(?:quảng\s+cáo|poster|banner)\s*',
        r'(?:dòng\s+)?chữ\s*[:=]?\s*.*',
        r'slogan\s*[:=]?\s*.*',
        r'tiêu\s+đề\s*[:=]?\s*.*',
        r'nằm\s+(?:ở\s+)?góc\s+.*',
        r'ở\s+góc\s+.*',
        r'thông\s+tin\s+cửa\s+hàng\s+.*',
        r'cửa\s+hàng\s+(?:đặt\s+)?(?:lên\s+|ở\s+)?(?:trên|dưới|đỉnh|đáy).*',
        r'đặt\s+(?:ở\s+|lên\s+|xuống\s+)?(?:trên|dưới|giữa|đáy|đỉnh)\s*(?:cùng)?.*',
        r'chữ\s+(?:ở\s+|lên\s+|xuống\s+)?(?:trên|dưới|giữa|đáy).*',
        r'(?:9:16|16:9|1:1|4:5|2:3|8k|4k|1080p)',
    ]
    for pat in patterns_to_remove:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)

    # Xóa các ký tự phân cách còn thừa
    text = re.sub(r'[,;:\-–—\s]+', ' ', text).strip()

    clean_sub = text or subject_hint or "commercial product"
    sub_lower = clean_sub.lower()

    if any(k in sub_lower for k in ["mì", "mỳ", "noodle", "ramen", "phở", "bún", "hảo hảo"]):
        prompt_en = f"Commercial food photography of an appetizing steaming bowl of Vietnamese noodles ({clean_sub}) with fresh ingredients and fragrant herbs, rich savory broth, professional studio rim lighting, 85mm macro lens f/2.0, cinematic food styling"
    elif any(k in sub_lower for k in ["trà", "tea", "cà phê", "coffee", "nước", "drink", "juice"]):
        prompt_en = f"High-end commercial drink photography of refreshing {clean_sub} in a clear glass with ice and fresh fruit slices, glistening condensation droplets, soft warm lighting, blurred atmospheric cafe background, 85mm lens"
    elif any(k in sub_lower for k in ["spa", "mỹ phẩm", "cosmetic", "kem", "serum", "lotion"]):
        prompt_en = f"Luxury beauty and cosmetic product photography of {clean_sub}, elegant podium, soft ambient studio lighting, gentle water reflections, minimalist aesthetic, 85mm lens"
    elif any(k in sub_lower for k in ["sofa", "bàn", "ghế", "nội thất", "furniture"]):
        prompt_en = f"High-end architectural interior photography of {clean_sub}, modern elegant living space, natural sunlight through large windows, minimalist aesthetic"
    elif any(k in sub_lower for k in ["gym", "thể thao", "fitness", "yoga"]):
        prompt_en = f"Dynamic commercial fitness photography of {clean_sub}, athletic environment, dramatic contrast rim lighting, cinematic energy"
    else:
        prompt_en = f"Commercial studio product photography of {clean_sub}, clean background, elegant composition, professional studio lighting, shallow depth of field, 85mm lens"

    return f"{prompt_en}, absolutely zero text, no letters, no words, clean commercial background"


def sanitize_corridor_prompt(raw_corridor: str, scene_prompt: str = "") -> str:
    """Đảm bảo corridor_prompt là vùng không gian âm ngữ cảnh (Contextual Negative Space) thuần túy không chữ."""
    if raw_corridor and "negative space" in raw_corridor.lower() and not any(k in raw_corridor.lower() for k in ["chữ", "text", "dòng chữ"]):
        return raw_corridor.strip()
    return "Smooth textured surface in extreme creamy bokeh, soft diffused lighting matching the scene ambiance, completely clean negative space, absolutely zero objects or text, perfect shallow depth of field"


def parse_user_prompt_intents(prompt: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Phân tích câu lệnh tự do của người dùng để rút trích:
    - template: template đề xuất
    - orientation: hướng bố cục
    - extracted_hero: nội dung chữ Hero (nếu người dùng chỉ định rõ)
    """
    if not prompt:
        return None, None, None

    prompt_lower = prompt.lower()
    template: Optional[str] = None
    orientation: Optional[str] = None
    extracted_hero: Optional[str] = None

    # 1. Trích xuất text chỉ định rõ (ví dụ: dòng chữ: "Mì số 1 việt nam")
    m_quote = re.search(r'(?:dòng\s+)?chữ\s*[:=]?\s*["\'“](.*?)["\'”]', prompt, re.IGNORECASE)
    if m_quote:
        raw_val = m_quote.group(1).strip()
        for p in ["nằm góc", "ở góc", "đặt trên", "đặt dưới", "bên trái", "bên phải", "góc trên", "góc dưới"]:
            raw_val = re.split(p, raw_val, flags=re.IGNORECASE)[0].strip()
        if raw_val:
            extracted_hero = raw_val
    elif 'dòng chữ' in prompt_lower:
        m_dc = re.search(r'dòng\s+chữ\s*[:=]?\s*([^,\.\n]+)', prompt, re.IGNORECASE)
        if m_dc:
            raw_c = m_dc.group(1).strip()
            for p in ["nằm góc", "ở góc", "đặt trên", "đặt dưới", "bên trái", "bên phải", "góc trên", "góc dưới"]:
                raw_c = re.split(p, raw_c, flags=re.IGNORECASE)[0].strip()
            if raw_c:
                extracted_hero = raw_c

    # 2. Định vị Template & Không gian
    if any(k in prompt_lower for k in ["góc", "capsule", "corner", "bo góc"]):
        template = "lifestyle_corner_pod"
        is_top = any(w in prompt_lower for w in ["trên", "top", "đỉnh"])
        is_bottom = any(w in prompt_lower for w in ["dưới", "bottom", "đáy"])
        is_right = any(w in prompt_lower for w in ["phải", "right"])
        is_left = any(w in prompt_lower for w in ["trái", "left"])
        if is_top and is_left:
            orientation = "top_left"
        elif is_top and is_right:
            orientation = "top_right"
        elif is_bottom and is_right:
            orientation = "bottom_right"
        elif is_bottom and is_left:
            orientation = "bottom_left"
        elif is_top:
            orientation = "top_left"
        elif is_right:
            orientation = "bottom_right"
        else:
            orientation = "bottom_left"

    elif (any(w in prompt_lower for w in ["cửa hàng", "store", "hotline", "địa chỉ"]) and any(w in prompt_lower for w in ["trên", "đỉnh", "top"])) or any(w in prompt_lower for w in ["chữ xuống dưới", "chữ ở dưới", "nội dung ở dưới", "tiêu đề ở dưới"]):
        template = "sandwich_bottom_heavy"

    elif any(k in prompt_lower for k in ["bước", "quy trình"]) or "guide" in prompt_lower:
        template = "step_process_roadmap"

    elif any(k in prompt_lower for k in ["feedback", "đánh giá", "review"]):
        template = "customer_feedback_card"

    elif any(k in prompt_lower for k in ["recruitment", "tuyển dụng"]):
        template = "recruitment_board"

    elif any(k in prompt_lower for k in ["before", "trước và sau", "trước sau"]):
        template = "before_after_split"

    elif any(k in prompt_lower for k in ["chéo", "cắt chéo", "diagonal"]):
        template = "diagonal_slash"
        orientation = "right" if any(w in prompt_lower for w in ["phải", "right"]) else "left"

    elif any(k in prompt_lower for k in ["chia đôi trái", "cột trái", "bên trái", "split left"]):
        template = "split_left"

    elif any(k in prompt_lower for k in ["chia đôi phải", "cột phải", "bên phải", "split right"]):
        template = "split_right"

    elif any(k in prompt_lower for k in ["chữ ở trên", "tiêu đề trên đỉnh"]):
        template = "sandwich_top_heavy"

    return template, orientation, extracted_hero


def fallback_heuristic_planner(
    form_data: Dict[str, Any],
    prompt: str = "",
    aspect_ratio: str = "1:1",
) -> TendooCreativePlan:
    """Bộ lập kế hoạch tất định dự phòng (Deterministic Heuristic Fallback) khi LLM offline."""
    category = form_data.get("category", "promo")
    prompt_lower = (prompt or "").lower()

    # 1. Phân tích ý đồ tự do từ Prompt (Template, Orientation, Hero)
    t_tpl, t_ori, t_hero = parse_user_prompt_intents(prompt)
    if t_tpl:
        template = t_tpl
        orientation = t_ori
    elif category == "guide":
        template = "step_process_roadmap"
        orientation = None
    elif category == "feedback":
        template = "customer_feedback_card"
        orientation = None
    elif category == "recruitment":
        template = "recruitment_board"
        orientation = None
    else:
        template = "sandwich_top_heavy" if aspect_ratio in ["9:16", "2:3", "4:5", "1:1"] else "split_left"
        orientation = None

    # Tránh rò rỉ dữ liệu placeholder không liên quan (ví dụ Audio Lab cho đồ ăn / đồ uống)
    prompt_or_title = f"{prompt} {form_data.get('title', '')} {form_data.get('product_name', '')}".lower()
    is_food_drink = any(k in prompt_or_title for k in ["mì", "mỳ", "trà", "cà phê", "nước", "thịt", "bánh", "quán", "food", "tea", "coffee", "noodle", "hảo hảo"])
    if is_food_drink and form_data.get("store_name") == "Tendoo Audio Lab":
        form_data = dict(form_data)
        form_data["store_name"] = None
        form_data["phone"] = None
        form_data["address"] = None
    if is_food_drink and form_data.get("website_link") == "https://tendoo.ai/headphones":
        form_data = dict(form_data)
        form_data["website_link"] = None

    # 2. Rút trích các trường văn bản theo Category và quyền uy Prompt > Form
    testimonial: Optional[str] = None
    reviewer_name: Optional[str] = None
    rating: Optional[int] = None
    steps: List[str] = []
    extra_texts: List[str] = []

    if category == "feedback":
        hero = t_hero or form_data.get("feedback_target") or form_data.get("title") or "KHÁCH HÀNG NÓI GÌ VỀ TENDOO"
        subhead = form_data.get("subhead")
        badge = form_data.get("special_offer") or form_data.get("discount")
        testimonial = form_data.get("feedback_quote") or "Trải nghiệm dịch vụ tuyệt vời, chất lượng vượt trội ngoài mong đợi!"
        reviewer_name = form_data.get("customer_name") or "Khách hàng thân thiết"
        raw_r = str(form_data.get("feedback_rating", "5"))
        rating = 5
        for ch in raw_r:
            if ch.isdigit():
                rating = max(1, min(5, int(ch)))
                break
        cta = form_data.get("cta") or "ĐẶT LỊCH NGAY"

    elif category == "guide":
        hero = t_hero or form_data.get("title") or "QUY TRÌNH HƯỚNG DẪN"
        subhead = form_data.get("subhead")
        badge = form_data.get("discount") or form_data.get("badge")
        raw_steps = form_data.get("guide_steps") or []
        if isinstance(raw_steps, list) and raw_steps:
            steps = [str(s).strip() for s in raw_steps if str(s).strip()]
        if not steps:
            steps = ["Bước 1: Chọn dịch vụ", "Bước 2: Xác nhận thông tin", "Bước 3: Hoàn tất đơn hàng"]
        cta = form_data.get("cta") or "BẮT ĐẦU NGAY"

    elif category == "recruitment":
        hero = t_hero or form_data.get("job_position") or form_data.get("title") or "TENDOO TÌM ĐỒNG ĐỘI"
        subhead = form_data.get("job_desc") or form_data.get("subhead")
        badge = form_data.get("discount")
        if form_data.get("apply_deadline"):
            extra_texts.append(f"Hạn nộp: {form_data['apply_deadline']}")
        if form_data.get("apply_method"):
            extra_texts.append(str(form_data["apply_method"]))
        cta = form_data.get("cta") or "ỨNG TUYỂN NGAY"

    elif category == "opening":
        hero = t_hero or form_data.get("title") or "TƯNG BỪNG KHAI TRƯƠNG"
        subhead = form_data.get("booking_contact") or form_data.get("subhead")
        badge = form_data.get("opening_promo") or form_data.get("discount")
        if form_data.get("opening_date"):
            extra_texts.append(f"Ngày mở bán: {form_data['opening_date']}")
        cta = form_data.get("cta") or "ĐẾN NGAY"

    elif category == "product_intro":
        hero = t_hero or form_data.get("title") or form_data.get("product_name") or "GIỚI THIỆU SẢN PHẨM"
        subhead = form_data.get("product_desc") or form_data.get("subhead")
        badge = form_data.get("price") or form_data.get("discount")
        if form_data.get("highlights"):
            hl = str(form_data["highlights"])
            extra_texts.extend([p.strip() for p in hl.split(",") if p.strip()])
        cta = form_data.get("cta") or "XEM CHI TIẾT"

    else:  # promo
        hero = t_hero or form_data.get("title") or form_data.get("product_name") or "ƯU ĐÃI ĐẶC BIỆT"
        subhead = form_data.get("applied_product") or form_data.get("product_desc") or form_data.get("subhead")
        badge = form_data.get("discount")
        dates = []
        if form_data.get("date_start"):
            dates.append(f"Từ {form_data['date_start']}")
        if form_data.get("date_end"):
            dates.append(f"Đến {form_data['date_end']}")
        if dates:
            extra_texts.append(" - ".join(dates))
        if form_data.get("highlights"):
            extra_texts.append(str(form_data["highlights"]))
        cta = form_data.get("cta") or "XEM CHI TIẾT"

    # CTA suppression nếu prompt cấm
    if any(k in prompt_lower for k in ["bỏ cta", "không cần cta", "đừng vẽ cta", "không có nút"]):
        cta = None

    # Store info: kiểm tra quyền uy Prompt > Form
    suppress_store = any(k in prompt_lower for k in [
        "đừng vẽ thông tin cửa hàng", "không vẽ thông tin cửa hàng", "bỏ thông tin cửa hàng",
        "không cần hotline", "đừng ghi địa chỉ", "không ghi địa chỉ", "bỏ địa chỉ"
    ])
    if suppress_store:
        store_info = None
    else:
        parts = []
        if form_data.get("store_name"):
            parts.append(form_data["store_name"])
        if form_data.get("phone"):
            parts.append(f"Hotline: {form_data['phone']}")
        if form_data.get("address"):
            parts.append(form_data["address"])
        store_info = " | ".join(parts) if parts else None

    # QR Code
    if form_data.get("website_link") and form_data.get("enable_qr"):
        qr_code = form_data["website_link"]
        qr_label = "QUÉT MÃ NGAY"
    elif form_data.get("qr_code"):
        qr_code = form_data["qr_code"]
        qr_label = form_data.get("qr_label", "QUÉT MÃ NGAY")
    elif "qr" in prompt_lower or "quét mã" in prompt_lower:
        qr_code = "https://tendoo.ai"
        qr_label = "QUÉT MÃ NGAY"
    else:
        qr_code = None
        qr_label = None

    # VFX khí quyển
    if any(k in prompt_lower for k in ["film", "grain", "retro", "vintage", "tạp chí", "cổ điển", "analog"]):
        vfx = "film_grain"
    elif any(k in prompt_lower for k in ["flare", "quang học", "lóa", "sci-fi", "tương lai", "xe", "ô tô", "automotive"]):
        vfx = "anamorphic_flare"
    elif any(k in prompt_lower for k in ["vàng", "gold", "bụi vàng", "luxury", "kim hoàn", "trang sức", "spa"]):
        vfx = "gold_dust"
    elif any(k in prompt_lower for k in ["nắng", "leak", "ánh sáng", "hoàng hôn", "ấm", "morning"]):
        vfx = "light_leak"
    elif any(k in prompt_lower for k in ["sương", "haze", "khói", "mù", "núi", "moody"]):
        vfx = "cinematic_haze"
    else:
        vfx = "none"

    # Font & Text Effect từ prompt hoặc style_pref
    style_pref = str(form_data.get("style_pref") or "auto").lower()
    if style_pref == "sang_trong":
        font = "playfair"
        text_effect = "3d_gold"
        background_tone = "dark_luxury"
    elif style_pref == "nang_dong":
        font = "anton"
        text_effect = "embossed"
        background_tone = "vibrant"
    elif style_pref == "le_hoi":
        font = "playfair"
        text_effect = "fire"
        background_tone = "dark_luxury"
        vfx = "gold_dust"
    elif style_pref == "am_cung":
        font = "lobster"
        text_effect = "plain_elegant"
        background_tone = "warm_rustic"
        vfx = "light_leak"
    elif style_pref == "hien_dai":
        font = "bevietnam"
        text_effect = "plain_elegant"
        background_tone = "dark_luxury"
    else:
        if any(k in prompt_lower for k in ["serif", "cổ điển", "sang trọng", "luxury", "bìa"]):
            font = "playfair"
        elif any(k in prompt_lower for k in ["mạnh mẽ", "sale", "khỏe", "gym", "thể thao"]):
            font = "anton"
        elif any(k in prompt_lower for k in ["tech", "công nghệ", "robot", "hiện đại"]):
            font = "days"
        elif any(k in prompt_lower for k in ["cafe", "mềm mại", "handwriting"]):
            font = "lobster"
        else:
            font = "bevietnam"

        if "neon" in prompt_lower:
            text_effect = "neon_bloom" if "bloom" in prompt_lower else "neon"
        elif "dập nổi" in prompt_lower or "emboss" in prompt_lower:
            text_effect = "embossed"
        elif "chromatic" in prompt_lower or "sai sắc" in prompt_lower:
            text_effect = "chromatic"
        elif "hologram" in prompt_lower:
            text_effect = "hologram"
        elif "vàng" in prompt_lower or "gold" in prompt_lower:
            text_effect = "3d_gold"
        else:
            text_effect = "plain_elegant"
        background_tone = "warm_rustic" if vfx in ["film_grain", "light_leak"] else "dark_luxury"

    # Theme color: ưu tiên primary_color từ form
    theme_color = form_data.get("primary_color") or ("#06B6D4" if "neon" in prompt_lower else "#D4AF37")

    style = StyleConfig(
        font=font,
        theme_color=theme_color,
        text_effect=text_effect,
        background_tone=background_tone,
        vfx=vfx,
    )

    # 3. Tạo Zero-Text Prompts chuẩn mực cho DiT Base 4B
    clean_scene = sanitize_scene_prompt(prompt, subject_hint=hero or form_data.get("title", ""))
    clean_corridor = sanitize_corridor_prompt("", clean_scene)

    return TendooCreativePlan(
        template=template,
        hero=hero,
        subhead=subhead,
        badge=badge,
        extra_texts=extra_texts,
        cta=cta,
        store_info=store_info,
        qr_code=qr_code,
        qr_label=qr_label,
        orientation=orientation,
        testimonial=testimonial,
        reviewer_name=reviewer_name,
        rating=rating,
        steps=steps,
        scene_prompt=clean_scene,
        corridor_prompt=clean_corridor,
        style=style,
    )


def _save_debug_trace(debug_trace: Dict[str, Any], custom_path: Optional[Path | str] = None) -> None:
    """Lưu trace thông tin input/output của LLM phục vụ debug."""
    try:
        # 1. Lưu tại custom_path nếu được truyền (thường là run_dir / 'llm_debug.json')
        if custom_path:
            p = Path(custom_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(debug_trace, ensure_ascii=False, indent=2), encoding="utf-8")

            # Tách riêng input.json và output.json cho tiện tra cứu nhanh
            p_in = p.parent / "llm_input.json"
            p_out = p.parent / "llm_output.json"
            p_in.write_text(json.dumps(debug_trace.get("input", {}), ensure_ascii=False, indent=2), encoding="utf-8")
            p_out.write_text(json.dumps(debug_trace.get("output", {}), ensure_ascii=False, indent=2), encoding="utf-8")

        # 2. Luôn ghi đè vào thư mục output_tendoo_v3/llm_debug để xem ngay trong terminal
        repo_root = Path(__file__).resolve().parent.parent.parent
        global_trace_dir = repo_root / "output_tendoo_v3" / "llm_debug"
        global_trace_dir.mkdir(parents=True, exist_ok=True)
        latest_file = global_trace_dir / "latest_llm_trace.json"
        latest_file.write_text(json.dumps(debug_trace, ensure_ascii=False, indent=2), encoding="utf-8")
        (global_trace_dir / "latest_llm_input.json").write_text(json.dumps(debug_trace.get("input", {}), ensure_ascii=False, indent=2), encoding="utf-8")
        (global_trace_dir / "latest_llm_output.json").write_text(json.dumps(debug_trace.get("output", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[LLM Planner] Không thể ghi file debug LLM: {e}")


def generate_creative_plan(
    form_data: Dict[str, Any],
    prompt: str = "",
    aspect_ratio: str = "1:1",
    model_override: Optional[str] = None,
    debug_save_path: Optional[Path | str] = None,
    return_debug: bool = False,
) -> Union[TendooCreativePlan, Tuple[TendooCreativePlan, Dict[str, Any]]]:
    """Gọi LLM (Qwen3.8-27B) để phân tích form + prompt và sinh TendooCreativePlan.
    Nếu không có API key hoặc lỗi mạng -> tự động kích hoạt Fallback Heuristic.
    Toàn bộ input, output thô và kết quả parse đều được lưu lại để phục vụ debug.
    """
    t_start = time.time()
    model_name = model_override or LLM_MODEL
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    user_payload = {
        "form_fields": form_data,
        "user_prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }
    user_message = f"Dữ liệu người dùng nhập:\n```json\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n```"

    request_body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
        # Tắt thinking mode để không nuốt token (hỗ trợ cả root kwarg và OpenAI extra_body)
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    debug_trace: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "llm_api",
        "status": "pending",
        "input": {
            "form_data": form_data,
            "user_prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "model": model_name,
            "api_endpoint": url,
            "has_api_key": bool(LLM_API_KEY),
            "timeout_seconds": LLM_TIMEOUT_S,
            "system_prompt": SYSTEM_PROMPT,
            "user_message": user_message,
            "request_payload": request_body,
        },
        "output": {
            "http_status_code": None,
            "raw_text_response": None,
            "raw_api_response": None,
            "extracted_json": None,
            "final_plan": None,
        },
        "latency_seconds": None,
        "error": None,
    }

    # Lấy API Key từ biến môi trường
    api_key = os.environ.get("TENDOO_V3_LLM_API_KEY") or LLM_API_KEY

    # Nếu không có API KEY hoặc chưa cho phép chế độ không key -> chuyển sang Fallback Heuristic
    if not api_key and not os.environ.get("TENDOO_V3_LLM_ALLOW_NO_KEY"):
        logger.info("[LLM Planner] Chưa cấu hình TENDOO_V3_LLM_API_KEY -> Sử dụng Fallback Heuristic.")
        plan = fallback_heuristic_planner(form_data, prompt, aspect_ratio)
        debug_trace["mode"] = "fallback_heuristic"
        debug_trace["status"] = "fallback_no_api_key"
        debug_trace["error"] = "TENDOO_V3_LLM_API_KEY chưa được cấu hình. Hệ thống tự động chuyển sang Fallback Heuristic."
        debug_trace["output"]["final_plan"] = plan.to_dict()
        debug_trace["latency_seconds"] = round(time.time() - t_start, 4)
        _save_debug_trace(debug_trace, debug_save_path)
        if return_debug:
            return plan, debug_trace
        return plan

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        logger.info(f"[LLM Planner] Gọi {model_name} tại {url}...")
        resp = requests.post(url, json=request_body, headers=headers, timeout=LLM_TIMEOUT_S)
        debug_trace["output"]["http_status_code"] = resp.status_code
        resp.raise_for_status()
        resp_data = resp.json()
        debug_trace["output"]["raw_api_response"] = resp_data

        content = resp_data["choices"][0]["message"]["content"]
        debug_trace["output"]["raw_text_response"] = content

        extracted_dict = extract_balanced_json(content)
        debug_trace["output"]["extracted_json"] = extracted_dict

        if not extracted_dict:
            logger.warning("[LLM Planner] Không parse được JSON từ phản hồi LLM -> Kích hoạt Fallback.")
            plan = fallback_heuristic_planner(form_data, prompt, aspect_ratio)
            debug_trace["mode"] = "fallback_heuristic"
            debug_trace["status"] = "fallback_unparseable_json"
            debug_trace["error"] = f"Phản hồi từ LLM không chứa JSON hợp lệ. Raw content: {content[:500]}"
            debug_trace["output"]["final_plan"] = plan.to_dict()
            debug_trace["latency_seconds"] = round(time.time() - t_start, 4)
            _save_debug_trace(debug_trace, debug_save_path)
            if return_debug:
                return plan, debug_trace
            return plan

        # Validate template -- chuẩn hoá trước khi so khớp
        tpl_raw = str(extracted_dict.get("template") or "")
        tpl_norm = tpl_raw.strip().lower().replace(" ", "_").replace("-", "_")
        if tpl_norm in TEMPLATE_CATALOG:
            extracted_dict["template"] = tpl_norm
        else:
            logger.warning(f"[LLM Planner] Template lạ '{tpl_raw}' không có trong catalog -> dùng mặc định sandwich_top_heavy.")
            extracted_dict["template"] = "sandwich_top_heavy"

        # Bảo đảm tuyệt đối 100% Zero-Text Background cho DiT Base 4B
        raw_sc = str(extracted_dict.get("scene_prompt") or "")
        extracted_dict["scene_prompt"] = sanitize_scene_prompt(
            raw_sc,
            subject_hint=str(extracted_dict.get("hero") or form_data.get("title") or "")
        )
        raw_cor = str(extracted_dict.get("corridor_prompt") or "")
        extracted_dict["corridor_prompt"] = sanitize_corridor_prompt(
            raw_cor,
            scene_prompt=extracted_dict["scene_prompt"]
        )

        plan = TendooCreativePlan.from_dict(extracted_dict)
        debug_trace["status"] = "success"
        debug_trace["output"]["final_plan"] = plan.to_dict()
        debug_trace["latency_seconds"] = round(time.time() - t_start, 4)
        _save_debug_trace(debug_trace, debug_save_path)
        logger.info(f"[LLM Planner] ✅ Kế hoạch sáng tạo thành công (Template: {plan.template}, Latency: {debug_trace['latency_seconds']}s)")
        if return_debug:
            return plan, debug_trace
        return plan

    except Exception as e:
        logger.warning(f"[LLM Planner] Lỗi khi kết nối tới LLM ({e}) -> Kích hoạt Fallback Heuristic.")
        plan = fallback_heuristic_planner(form_data, prompt, aspect_ratio)
        debug_trace["mode"] = "fallback_heuristic"
        debug_trace["status"] = "fallback_exception"
        debug_trace["error"] = f"Exception: {type(e).__name__}: {str(e)}"
        debug_trace["output"]["final_plan"] = plan.to_dict()
        debug_trace["latency_seconds"] = round(time.time() - t_start, 4)
        _save_debug_trace(debug_trace, debug_save_path)
        if return_debug:
            return plan, debug_trace
        return plan


__all__ = [
    "_save_debug_trace",
    "extract_balanced_json",
    "fallback_heuristic_planner",
    "generate_creative_plan",
    "parse_user_prompt_intents",
    "sanitize_corridor_prompt",
    "sanitize_scene_prompt",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 70)
    print("🔍 TENDOO v3 LLM PLANNER - CONNECTION & INFERENCE TEST")
    print(f"   Base URL: {LLM_BASE_URL}")
    print(f"   Model:    {LLM_MODEL}")
    key_disp = f"CONFIGURED (***{LLM_API_KEY[-4:]})" if LLM_API_KEY else "NOT SET"
    print(f"   API Key:  {key_disp}")
    print("=" * 70)

    test_form = {
        "category": "promo",
        "title": "Mì Hảo Hảo Tôm Chua Cay",
        "discount": "GIẢM 50%",
        "applied_product": "Thùng 30 gói",
        "store_name": "Acecook Mart",
    }
    test_prompt = "Poster phong cách điện ảnh ấm cúng, chữ đặt ở góc trên"
    print("\n👉 Gửi request thử nghiệm tới LLM Planner...")
    plan, debug_info = generate_creative_plan(test_form, prompt=test_prompt, return_debug=True)

    print(f"\n📊 KẾT QUẢ KIỂM TRA:")
    print(f"   Mode:    {debug_info.get('mode')}")
    print(f"   Status:  {debug_info.get('status')}")
    print(f"   Latency: {debug_info.get('latency_seconds')}s")
    if debug_info.get("error"):
        print(f"   Error:   {debug_info.get('error')}")
    print(f"   Template được chọn: {plan.template}")
    print(f"   Hero text:          {plan.hero}")
    print(f"   Scene Prompt:       {plan.scene_prompt[:120]}...")
    print("=" * 70)

