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
from pathlib import Path
from typing import Any, Dict, Optional

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

# Nạp src/tendoo_v3/.env NGAY TẠI ĐÂY (không chỉ ở demo_server.py) -- các hằng số
# LLM_BASE_URL/LLM_API_KEY/LLM_MODEL bên dưới đọc os.environ NGAY LÚC IMPORT MODULE
# này, nên nếu chỉ load .env ở demo_server.py sau khi module này đã được import (hoặc
# khi 1 script khác import thẳng llm_planner mà không qua demo_server.py), các giá trị
# thật trong .env sẽ không bao giờ có hiệu lực. override=False: biến môi trường đã
# export sẵn ngoài shell luôn thắng .env, không bị ghi đè âm thầm.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# Cấu hình môi trường OpenAI-compatible
LLM_BASE_URL = os.environ.get("TENDOO_V3_LLM_BASE_URL", "http://10.221.155.3:8004/v1")
LLM_API_KEY = os.environ.get("TENDOO_V3_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("TENDOO_V3_LLM_MODEL", "Qwen/Qwen3.8-27B")
LLM_TIMEOUT_S = float(os.environ.get("TENDOO_V3_LLM_TIMEOUT_S", "60"))

SYSTEM_PROMPT = f"""Bạn là Giám đốc Nghệ thuật & Sáng tạo (Creative Director) hàng đầu của Tendoo AI Studio.
Nhiệm vụ của bạn là tiếp nhận thông tin từ form người dùng + câu lệnh tự do (freeform prompt) để:
1. CHỌN 1 TEMPLATE TỐI ƯU NHẤT từ danh mục có sẵn để tạo nên một poster thương mại tuyệt đẹp.
2. PHÂN BỔ CÁC THÀNH PHẦN CHỮ (Hero, Subhead, Badge, Pills tính năng, CTA, Hotline/Địa chỉ, QR Code, Steps, Review...) sao cho đúng chính tả tiếng Việt, ngắn gọn, súc tích và có tính chuyển đổi cao.
3. VIẾT PROMPT MÔ HÌNH HÌNH ẢNH (DiT Base 4B):
   - `scene_prompt`: Mô tả bối cảnh và sản phẩm chân thực, ánh sáng studio, chất liệu vật lý. TUYỆT ĐỐI KHÔNG ghi chữ cần vẽ, KHÔNG ghi tỷ lệ 16:9 hay 8k.
   - `corridor_prompt`: CÔNG THỨC KHOẢNG TRỐNG TỰ NHIÊN (Contextual Negative Space) — Không dùng phông trắng studio đơn điệu gây ra vệt trắng cắt vụn poster. Hãy mô tả corridor như một phần tiếp nối mờ ảo tự nhiên của bối cảnh chính trong trường độ sâu (extreme bokeh, shallow depth of field, soft diffused light, smooth texture of table/wall/surface, zero objects or props, keeping the exact same color temperature and ambiance).
4. QUY TẮC QUYỀN UY & TRƯỜNG THÔNG TIN (Prompt Authority & Strict Suppression Rules):
   - QUY TẮC THỨ BẬC: Câu lệnh tự do của người dùng (User Prompt) LUÔN CÓ QUYỀN ƯU TIÊN CAO NHẤT (User Prompt > Form fields).
   - Nếu trong prompt người dùng yêu cầu KHÔNG vẽ hoặc BỎ một thông tin nào (ví dụ: "đừng vẽ thông tin cửa hàng", "không cần hotline", "không ghi giá", "bỏ nút CTA"), bạn BẮT BUỘC gán trường đó thành `null` (hoặc `[]`), KỂ CẢ KHI TRONG FORM CÓ ĐIỀN THÔNG TIN ĐÓ!
   - Nếu trong prompt không cấm và trong form có điền: Giữ nguyên thông tin chuẩn xác từ form.
   - Nếu CẢ TRONG PROMPT LẪN FORM ĐỀU KHÔNG CÓ thông tin đó: BẮT BUỘC gán `null` (hoặc `[]`), TUYỆT ĐỐI KHÔNG tự bịa hoặc hallucinate các câu slogan/hotline/giá mẫu (như không tự bịa "HOT DEAL", "1900 xxxx", "www.tendoo.ai" nếu người dùng không yêu cầu).
   - MÃ QR (`qr_code`): CHỈ điền khi người dùng thực sự yêu cầu hiển thị mã QR (link đặt hàng, quét mã...). Nếu prompt/form không nhắc gì đến QR, để `qr_code: null`, `qr_label: null`.
5. ĐIỀU PHỐI PHONG CÁCH (Style, Font, Text Effect, Cinematic VFX):
   - Chọn phông chữ (`style.font`) phù hợp với ngành hàng và tinh thần poster.
   - Chọn hiệu ứng chữ (`style.text_effect`) phù hợp với chất liệu và ánh sáng.
   - Chọn hiệu ứng điện ảnh quang học (`style.vfx`):
     + Chủ động chọn 'film_grain' khi prompt mang hơi hướng hoài cổ, retro, vintage, bìa tạp chí thời trang, cafe mộc, phim analog 35mm.
     + Chọn 'anamorphic_flare' cho ô tô, đồng hồ, công nghệ hiện đại, gaming, sci-fi.
     + Chọn 'gold_dust' cho trang sức, tiệc tùng sang trọng, spa luxury, mỹ phẩm cao cấp.
     + Chọn 'light_leak' cho ảnh chụp đời sống ấm áp, dã ngoại mùa hè, gia đình.
     + Chọn 'cinematic_haze' cho du lịch hùng vĩ, núi non, điện ảnh sử thi moody.
     + Chọn 'none' nếu poster phẳng tối giản hiện đại.

6. BẢNG ÁNH XẠ TRƯỜNG DỮ LIỆU TỪ FORM (Form Fields Semantic Mapping):
   - Khuyến Mại (promo): `title` -> `hero`, `discount` -> `badge`, `applied_product` -> `subhead`, ngày tháng -> `extra_texts`.
   - Giới Thiệu Sản Phẩm (product_intro): `title` hoặc `product_name` -> `hero`, `price` -> `badge`, `product_desc` -> `subhead`, `highlights` -> `extra_texts` (tách thành các pills ngắn gọn).
   - Khai Trương (opening): `title` -> `hero`, `opening_promo` -> `badge`, `opening_date` + `booking_contact` -> `subhead` hoặc `extra_texts`.
   - Đánh Giá (feedback): `feedback_target` hoặc `title` -> `hero`, `feedback_quote` -> `testimonial`, `customer_name` -> `reviewer_name`, `feedback_rating` (vd 5 sao) -> `rating` (số nguyên 1-5), `special_offer` -> `badge`. Template tối ưu: 'customer_feedback_card'.
   - Tuyển Dụng (recruitment): `title` hoặc `job_position` -> `hero`, `job_desc` -> `subhead`, `apply_deadline` + `apply_method` -> `extra_texts`. Template tối ưu: 'recruitment_board'.
   - Quy Trình / Hướng Dẫn (guide): `title` -> `hero`, `guide_steps` -> `steps`. Template tối ưu: 'step_process_roadmap'.
   - Màu sắc & Phong cách từ Form:
     + Nếu có `primary_color` (hex color): gán `style.theme_color = primary_color`.
     + Nếu có `style_pref`:
       * 'sang_trong': ưu tiên font 'playfair', text_effect '3d_gold', background_tone 'dark_luxury'.
       * 'hien_dai': ưu tiên font 'bevietnam' hoặc 'days', text_effect 'plain_elegant'.
       * 'nang_dong': ưu tiên font 'anton', text_effect 'embossed'.
       * 'le_hoi': ưu tiên font 'playfair', text_effect 'fire', vfx 'gold_dust'.
       * 'am_cung': ưu tiên font 'lobster' hoặc 'bevietnam', background_tone 'warm_rustic', vfx 'light_leak'.

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
  "scene_prompt": "Cinematic close-up shot of a traditional Vietnamese metal coffee filter dripping rich dark espresso on a dark rustic wooden table, morning golden hour sunlight cutting through soft steam, high-end food photography, rich textures, 85mm lens f/1.8",
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


def fallback_heuristic_planner(
    form_data: Dict[str, Any],
    prompt: str = "",
    aspect_ratio: str = "1:1",
) -> TendooCreativePlan:
    """Bộ lập kế hoạch tất định dự phòng (Deterministic Heuristic Fallback) khi LLM offline."""
    category = form_data.get("category", "promo")
    prompt_lower = (prompt or "").lower()
    orientation: Optional[str] = None
    
    # 1. Chọn template dựa trên intent & định vị không gian (Spatial Orientation Priority)
    if "bước" in prompt_lower or "quy trình" in prompt_lower or category == "guide":
        template = "step_process_roadmap"
    elif "feedback" in prompt_lower or "đánh giá" in prompt_lower or category == "feedback":
        template = "customer_feedback_card"
    elif "recruitment" in prompt_lower or "tuyển dụng" in prompt_lower or category == "recruitment":
        template = "recruitment_board"
    elif "before" in prompt_lower or ("trước" in prompt_lower and "sau" in prompt_lower):
        template = "before_after_split"
    elif "góc" in prompt_lower or "capsule" in prompt_lower:
        # Gom ở góc -> Hộp bo góc Lifestyle Corner Pod
        template = "lifestyle_corner_pod"
        if "trên cùng bên trái" in prompt_lower or "góc trên trái" in prompt_lower or "top left" in prompt_lower:
            orientation = "top_left"
        elif "trên cùng bên phải" in prompt_lower or "góc trên phải" in prompt_lower or "top right" in prompt_lower:
            orientation = "top_right"
        elif "dưới cùng bên phải" in prompt_lower or "góc dưới phải" in prompt_lower or "bottom right" in prompt_lower:
            orientation = "bottom_right"
        else:
            orientation = "bottom_left"
    elif "thông tin cửa hàng lên trên" in prompt_lower or "chữ xuống dưới" in prompt_lower:
        template = "sandwich_bottom_heavy"
    elif ("trên" in prompt_lower or "đỉnh" in prompt_lower) and ("dưới" in prompt_lower or "đáy" in prompt_lower):
        # Có cả trên lẫn dưới -> Băng kẹp Sandwich Top-Heavy
        template = "sandwich_top_heavy"
    elif "giữa" in prompt_lower or "chính giữa" in prompt_lower or "center" in prompt_lower:
        template = "sandwich_top_heavy"
    elif "chéo" in prompt_lower or "cắt chéo" in prompt_lower:
        template = "diagonal_slash"
        if "phải" in prompt_lower or "bên phải" in prompt_lower or "right" in prompt_lower:
            orientation = "right"
        else:
            orientation = "left"
    elif "trái" in prompt_lower or "bên trái" in prompt_lower:
        template = "split_left"
    elif "phải" in prompt_lower or "bên phải" in prompt_lower:
        template = "split_right"
    elif "thể thao" in prompt_lower or "app" in prompt_lower or "fintech" in prompt_lower:
        template = "diagonal_slash"
    elif "trên" in prompt_lower:
        template = "sandwich_top_heavy"
    elif "dưới" in prompt_lower:
        template = "sandwich_bottom_heavy"
    else:
        # Mặc định thông minh theo aspect ratio:
        # Khung dọc (9:16, 2:3, 4:5, 1:1) -> sandwich_top_heavy (kẹp 2 đầu chuẩn tỷ lệ vàng poster)
        # Khung ngang (16:9, 4:3) -> split_left (cột bên trái thuận hướng quét mắt)
        template = "sandwich_top_heavy" if aspect_ratio in ["9:16", "2:3", "4:5", "1:1"] else "split_left"

    # 2. Rút trích các trường văn bản theo Category và quyền uy Prompt > Form
    testimonial: Optional[str] = None
    reviewer_name: Optional[str] = None
    rating: Optional[int] = None
    steps: List[str] = []
    extra_texts: List[str] = []

    if category == "feedback":
        hero = form_data.get("feedback_target") or form_data.get("title") or "KHÁCH HÀNG NÓI GÌ VỀ TENDOO"
        subhead = form_data.get("subhead")
        badge = form_data.get("special_offer") or form_data.get("discount")
        testimonial = form_data.get("feedback_quote") or "Trải nghiệm dịch vụ tuyệt vời, chất lượng vượt trội ngoài mong đợi!"
        reviewer_name = form_data.get("customer_name") or "Khách hàng thân thiết"
        # Parse rating
        raw_r = str(form_data.get("feedback_rating", "5"))
        rating = 5
        for ch in raw_r:
            if ch.isdigit():
                rating = max(1, min(5, int(ch)))
                break
        cta = form_data.get("cta") or "ĐẶT LỊCH NGAY"

    elif category == "guide":
        hero = form_data.get("title") or "QUY TRÌNH HƯỚNG DẪN"
        subhead = form_data.get("subhead")
        badge = form_data.get("discount") or form_data.get("badge")
        raw_steps = form_data.get("guide_steps") or []
        if isinstance(raw_steps, list) and raw_steps:
            steps = [str(s).strip() for s in raw_steps if str(s).strip()]
        if not steps:
            steps = ["Bước 1: Chọn dịch vụ", "Bước 2: Xác nhận thông tin", "Bước 3: Hoàn tất đơn hàng"]
        cta = form_data.get("cta") or "BẮT ĐẦU NGAY"

    elif category == "recruitment":
        hero = form_data.get("job_position") or form_data.get("title") or "TENDOO TÌM ĐỒNG ĐỘI"
        subhead = form_data.get("job_desc") or form_data.get("subhead")
        badge = form_data.get("discount")
        if form_data.get("apply_deadline"):
            extra_texts.append(f"Hạn nộp: {form_data['apply_deadline']}")
        if form_data.get("apply_method"):
            extra_texts.append(str(form_data["apply_method"]))
        cta = form_data.get("cta") or "ỨNG TUYỂN NGAY"

    elif category == "opening":
        hero = form_data.get("title") or "TƯNG BỪNG KHAI TRƯƠNG"
        subhead = form_data.get("booking_contact") or form_data.get("subhead")
        badge = form_data.get("opening_promo") or form_data.get("discount")
        if form_data.get("opening_date"):
            extra_texts.append(f"Ngày mở bán: {form_data['opening_date']}")
        cta = form_data.get("cta") or "ĐẾN NGAY"

    elif category == "product_intro":
        hero = form_data.get("title") or form_data.get("product_name") or "GIỚI THIỆU SẢN PHẨM"
        subhead = form_data.get("product_desc") or form_data.get("subhead")
        badge = form_data.get("price") or form_data.get("discount")
        if form_data.get("highlights"):
            hl = str(form_data["highlights"])
            extra_texts.extend([p.strip() for p in hl.split(",") if p.strip()])
        cta = form_data.get("cta") or "XEM CHI TIẾT"

    else:  # promo
        hero = form_data.get("title") or form_data.get("product_name") or "ƯU ĐÃI ĐẶC BIỆT"
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
        scene_prompt=prompt or "Studio product photography, high-end commercial aesthetic, beautiful lighting",
        corridor_prompt="Smooth background surface in soft focus, gentle bokeh, clean negative space without objects, matching scene color and lighting",
        style=style,
    )


def generate_creative_plan(
    form_data: Dict[str, Any],
    prompt: str = "",
    aspect_ratio: str = "1:1",
    model_override: Optional[str] = None,
) -> TendooCreativePlan:
    """Gọi LLM (Qwen3.8-27B) để phân tích form + prompt và sinh TendooCreativePlan.
    Nếu không có API key hoặc lỗi mạng -> tự động kích hoạt Fallback Heuristic.
    """
    model_name = model_override or LLM_MODEL

    # Nếu không có API KEY hoặc LLM_BASE_URL rỗng -> chạy ngay fallback
    if not LLM_API_KEY and not os.environ.get("TENDOO_V3_LLM_ALLOW_NO_KEY"):
        logger.info("[LLM Planner] Chưa cấu hình TENDOO_V3_LLM_API_KEY -> Sử dụng Fallback Heuristic.")
        return fallback_heuristic_planner(form_data, prompt, aspect_ratio)

    user_payload = {
        "form_fields": form_data,
        "user_prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }
    user_message = f"Dữ liệu người dùng nhập:\n```json\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n```"

    headers = {
        "Content-Type": "application/json",
    }
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    request_body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
        # Tắt thinking mode để không bị nuốt token
        "chat_template_kwargs": {"enable_thinking": False},
    }

    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    try:
        logger.info(f"[LLM Planner] Gọi {model_name} tại {url}...")
        resp = requests.post(url, json=request_body, headers=headers, timeout=LLM_TIMEOUT_S)
        resp.raise_for_status()
        resp_data = resp.json()

        content = resp_data["choices"][0]["message"]["content"]
        extracted_dict = extract_balanced_json(content)

        if not extracted_dict:
            logger.warning("[LLM Planner] Không parse được JSON từ phản hồi LLM -> Kích hoạt Fallback.")
            return fallback_heuristic_planner(form_data, prompt, aspect_ratio)

        # Validate template -- chuẩn hoá trước khi so khớp (model đôi khi trả về sai
        # hoa/thường hoặc dư khoảng trắng, vd "Split_Left " thay vì "split_left").
        tpl_raw = str(extracted_dict.get("template") or "")
        tpl_norm = tpl_raw.strip().lower().replace(" ", "_").replace("-", "_")
        if tpl_norm in TEMPLATE_CATALOG:
            extracted_dict["template"] = tpl_norm
        else:
            logger.warning(f"[LLM Planner] Template lạ '{tpl_raw}' không có trong catalog -> dùng mặc định sandwich_top_heavy.")
            extracted_dict["template"] = "sandwich_top_heavy"

        return TendooCreativePlan.from_dict(extracted_dict)

    except Exception as e:
        logger.warning(f"[LLM Planner] Lỗi khi kết nối tới LLM ({e}) -> Kích hoạt Fallback Heuristic.")
        return fallback_heuristic_planner(form_data, prompt, aspect_ratio)


__all__ = [
    "extract_balanced_json",
    "fallback_heuristic_planner",
    "generate_creative_plan",
]
