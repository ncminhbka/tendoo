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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.hero_markup import suggest_hero_parts
from tendoo_v3.routing import route_template
from tendoo_v3.llm_prompts import MULTI_VARIANT_INSTRUCTION_TEMPLATE, SYSTEM_PROMPT
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.validators import dedupe_plan, log_plan_issues

logger = logging.getLogger("TendooV3.LLMPlanner")

# Nạp .env từ cả thư mục src/tendoo_v3/.env lẫn thư mục gốc của repo
_v3_dir = Path(__file__).resolve().parent
_root_dir = _v3_dir.parent.parent
load_dotenv(_v3_dir / ".env", override=False)
load_dotenv(_root_dir / ".env", override=False)

# Cấu hình Backend & Môi trường OpenAI-compatible / Local
LLM_BACKEND = os.environ.get("TENDOO_V3_LLM_BACKEND", "auto").lower()  # "auto", "local", "api"
LLM_BASE_URL = os.environ.get("TENDOO_V3_LLM_BASE_URL", "http://10.221.155.3:8004/v1")
LLM_API_KEY = os.environ.get("TENDOO_V3_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("TENDOO_V3_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")
LLM_TIMEOUT_S = float(os.environ.get("TENDOO_V3_LLM_TIMEOUT_S", "60"))
LLM_CONNECT_TIMEOUT_S = float(os.environ.get("TENDOO_V3_LLM_CONNECT_TIMEOUT_S", "3.0"))

# ==============================================================================
# Local Qwen3-4B Checkpoint Resolution & In-Process Generation
# (Học tập từ Phase C src/tendoo/llm_render_plan_server.py)
# ==============================================================================

LOCAL_QWEN_MODEL: Any = None
LOCAL_QWEN_TOKENIZER: Any = None

QWEN_CHATML_FALLBACK_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def resolve_qwen3_checkpoint_path(variant: str = "4B") -> str:
    """Tự động tìm đường dẫn checkpoint Qwen3 có sẵn trên máy chủ:
    1. Biến môi trường QWEN3_4B_MODEL_PATH hoặc TEXT_ENCODER_PATH
    2. Thư mục persistent-data/FLUX.2-klein-base-4B/text_encoder
    3. Hugging Face hub id 'Qwen/Qwen3-{variant}-FP8'
    """
    env_key = f"QWEN3_{variant.upper()}_MODEL_PATH"
    if env_key in os.environ and os.path.exists(os.environ[env_key]):
        return os.environ[env_key]
    if "TEXT_ENCODER_PATH" in os.environ and os.path.exists(os.environ["TEXT_ENCODER_PATH"]):
        return os.environ["TEXT_ENCODER_PATH"]

    try:
        from flux2.util import find_persistent_data_root
        p_root = find_persistent_data_root()
        if p_root:
            for cp in (os.path.join(p_root, "text_encoder"), p_root):
                if os.path.exists(cp) and (
                    os.path.exists(os.path.join(cp, "config.json"))
                    or os.path.exists(os.path.join(cp, "model.safetensors.index.json"))
                ):
                    return cp
    except Exception:
        pass

    candidates = [
        Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B")),
        Path("/home/jovyan/persistent-data/FLUX.2-klein-base-4B"),
        Path("/persistent-data/FLUX.2-klein-base-4B"),
        Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-4B")),
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4B"),
        Path("/persistent-data/FLUX.2-klein-4B"),
    ]
    for c in candidates:
        te = c / "text_encoder"
        if te.exists() and ((te / "config.json").exists() or (te / "model.safetensors.index.json").exists()):
            return str(te)

    return f"Qwen/Qwen3-{variant}-FP8"


def resolve_tokenizer_path(model_path: str) -> str:
    """Trên máy chủ FLUX.2, weights nằm ở 'text_encoder' nhưng tokenizer đầy đủ
    thường nằm ở thư mục ngang hàng 'tokenizer'.
    """
    if os.path.exists(model_path):
        parent_dir = os.path.dirname(os.path.abspath(model_path))
        sibling_tokenizer = os.path.join(parent_dir, "tokenizer")
        if os.path.exists(sibling_tokenizer):
            return sibling_tokenizer
    return model_path


def load_local_qwen3(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Nạp Qwen3-4B CausalLM + Tokenizer cục bộ vào GPU để suy luận in-process."""
    global LOCAL_QWEN_MODEL, LOCAL_QWEN_TOKENIZER
    if LOCAL_QWEN_MODEL is not None and LOCAL_QWEN_TOKENIZER is not None:
        return LOCAL_QWEN_MODEL, LOCAL_QWEN_TOKENIZER

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine. Local Qwen3 requires an NVIDIA GPU.")

    m_path = model_path or resolve_qwen3_checkpoint_path("4B")
    if not m_path or not os.path.exists(m_path):
        raise FileNotFoundError(f"Local Qwen3 weights not found on disk: '{m_path}'. Bypassing remote download.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_device = device or os.environ.get("TENDOO_V3_DEVICE_AUX", "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    t_path = resolve_tokenizer_path(m_path)

    logger.info(f"[Local Qwen3] Khởi tạo mô hình Qwen3-4B cục bộ:")
    logger.info(f"   Model Weights: {m_path}")
    logger.info(f"   Tokenizer:     {t_path}")
    logger.info(f"   Target Device: {target_device}")

    LOCAL_QWEN_TOKENIZER = AutoTokenizer.from_pretrained(t_path, trust_remote_code=True)
    if getattr(LOCAL_QWEN_TOKENIZER, "chat_template", None) is None:
        logger.warning("[Local Qwen3] Tokenizer thiếu chat_template -> Cài đặt ChatML fallback.")
        LOCAL_QWEN_TOKENIZER.chat_template = QWEN_CHATML_FALLBACK_TEMPLATE

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    LOCAL_QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
        m_path,
        torch_dtype=dtype,
        device_map=target_device,
        trust_remote_code=True,
    ).eval()

    logger.info(f"[Local Qwen3] ✅ Đã nạp thành công Qwen3-4B CausalLM lên {target_device}!")
    return LOCAL_QWEN_MODEL, LOCAL_QWEN_TOKENIZER


def free_local_qwen3():
    """Giải phóng bộ nhớ VRAM của Qwen3 local."""
    global LOCAL_QWEN_MODEL, LOCAL_QWEN_TOKENIZER
    import gc
    try:
        import torch
        if LOCAL_QWEN_MODEL is not None:
            del LOCAL_QWEN_MODEL
            LOCAL_QWEN_MODEL = None
        if LOCAL_QWEN_TOKENIZER is not None:
            del LOCAL_QWEN_TOKENIZER
            LOCAL_QWEN_TOKENIZER = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[Local Qwen3] Đã giải phóng bộ nhớ VRAM của Qwen3 local.")
    except Exception as e:
        logger.warning(f"[Local Qwen3] Notice freeing VRAM: {e}")


def generate_local_qwen_response(
    model: Any,
    tokenizer: Any,
    messages: list,
    max_new_tokens: int = 1536,
) -> str:
    """Thực hiện suy luận .generate() cục bộ với enable_thinking=False."""
    import torch

    device = next(model.parameters()).device
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        if getattr(tokenizer, "chat_template", None) is None:
            tokenizer.chat_template = QWEN_CHATML_FALLBACK_TEMPLATE
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return raw


def extract_balanced_json(text: str) -> Optional[Dict[str, Any]]:
    """Trích xuất JSON an toàn bằng thuật toán quét ngoặc cân bằng (Balanced Braces).

    Khi thất bại, LOG rõ 1 trong 2 nguyên nhân khác nhau (trước đây gộp chung thành
    1 lỗi generic "không chứa JSON hợp lệ", không phân biệt được -- user báo cáo thật:
    gemma-4-E4B-it (model nhỏ) hay lỗi ở case num_variants > 1, nhưng log cũ chỉ in
    400 ký tự đầu của raw content nên KHÔNG BAO GIỜ thấy được điểm lỗi thật (thường
    nằm xa hơn 400 ký tự với JSON nhiều phương án), cũng không biết là do (A) response
    bị CẮT NGANG (hết token budget giữa chừng, không bao giờ đóng ngoặc) hay (B) response
    đã đóng ngoặc đủ nhưng cú pháp JSON bên trong sai (escape lỗi, dấu ngoặc kép thông
    minh “ ” lẫn vào chuỗi tiếng Việt...) -- 2 nguyên nhân cần hướng debug khác hẳn nhau
    (A -> tăng max_tokens hoặc giảm num_variants; B -> lỗi model tạo JSON sai cú pháp,
    tăng token không giúp được gì)."""
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
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"[extract_balanced_json] (B) Ngoặc CÂN BẰNG nhưng JSON sai cú "
                            f"pháp tại dòng {e.lineno} cột {e.colno} (ký tự {e.pos}): {e.msg}. "
                            f"Ngữ cảnh quanh lỗi: ...{json_str[max(0, e.pos - 80):e.pos]}"
                            f"[LỖI TẠI ĐÂY>>>]{json_str[e.pos:e.pos + 80]}..."
                        )
                        return None
    logger.warning(
        f"[extract_balanced_json] (A) Response bị CẮT NGANG -- quét hết {len(cleaned) - start_idx} "
        f"ký tự từ dấu '{{' đầu tiên mà ngoặc không bao giờ cân bằng lại (depth kết thúc = {depth}), "
        f"nghĩa là response dừng giữa chừng trước khi model kịp đóng JSON (hết token budget hoặc "
        f"model dừng sớm) -- không phải lỗi cú pháp. 200 ký tự cuối cùng nhận được: "
        f"...{cleaned[-200:]}"
    )
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


# Tín hiệu tích cực: corridor phải đọc như 1 vùng quang học có thật (bokeh/mất nét),
# không phải chỉ cần đúng 1 cụm từ "negative space" là đủ -- 1 câu nhắc "blurred",
# "out of focus", "shallow depth of field" mà thiếu đúng cụm đó vẫn hợp lệ.
_CORRIDOR_OPTICAL_SIGNALS = (
    "negative space", "bokeh", "out of focus", "out-of-focus", "blurred",
    "shallow depth of field", "defocused", "soft blur",
)
# Tín hiệu cấm: không chỉ "chữ/text" (đã có) mà cả danh từ vật thể salient phổ biến --
# 1 corridor_prompt có thể lọt qua check "negative space" cũ trong khi vẫn mô tả 1 vật
# thể cụ thể (vd "negative space with a small wooden chair"), khiến diffusion vẫn có
# đủ tín hiệu để vẽ 1 chủ thể cạnh tranh vào đúng vùng dành cho chữ.
_CORRIDOR_BANNED_WORDS = (
    "chữ", "text", "dòng chữ", "slogan", "word", "letter",
    "person", "human", "hand", "face", "product", "object", "subject",
    "chair", "table setting", "prop", "item", "figure",
)


def _extract_scene_material_hint(scene_prompt: str) -> Optional[str]:
    """Trích 1 từ khoá CHẤT LIỆU THẬT đã có sẵn trong scene_prompt (vd 'wooden', 'marble',
    'glass'...) để corridor fallback có thể "tiếp nối" đúng vật liệu đó -- CHỈ trả về khi
    THỰC SỰ tìm thấy tín hiệu rõ ràng, không bao giờ tự đoán/bịa chất liệu.

    Trước đây khi không tìm thấy từ khoá nào, hàm này lấy tạm 6 từ đầu của scene_prompt
    làm "chất liệu" -- gây lỗi thật: corridor tự khẳng định 1 chất liệu không có căn cứ gì
    (vd 1 poster trà đào cam sả có thể bị gán nhầm sang mô tả như đang có tấm lụa/vải bên
    cạnh nếu 6 từ đầu tình cờ khớp 1 từ khoá không liên quan). Giờ trả `None` khi không
    chắc, để `sanitize_corridor_prompt()` chọn phương án AN TOÀN HƠN: ánh sáng/màu trung
    tính không tuyên bố chất liệu nào, thay vì đoán bừa.
    """
    if not scene_prompt:
        return None
    lower = scene_prompt.lower()
    material_keywords = [
        "wooden", "wood", "marble", "concrete", "fabric", "velvet", "leather",
        "metal", "glass", "ceramic", "stone", "paper", "linen", "silk",
        "gỗ", "đá", "vải", "kim loại", "kính",
    ]
    found = [kw for kw in material_keywords if kw in lower]
    return f"{found[0]} surface" if found else None


def sanitize_corridor_prompt(raw_corridor: str, scene_prompt: str = "") -> str:
    """Đảm bảo corridor_prompt là vùng không gian âm ngữ cảnh (Contextual Negative Space)
    thuần túy không chữ, KHÔNG chứa vật thể salient cụ thể nào. Khi phải rơi vào fallback,
    CHỈ tiếp nối 1 chất liệu cụ thể (bàn gỗ, kính, đá...) nếu `scene_prompt` THẬT SỰ có
    tín hiệu chất liệu đó -- nếu không, dùng 1 lớp ánh sáng/màu trung tính KHÔNG tuyên bố
    chất liệu nào, để tránh áp đặt 1 vật liệu không có căn cứ (vd tấm lụa cạnh ly trà đào)
    chỉ vì cần 1 cụm từ để điền vào công thức "Extension of the same X"."""
    lower = (raw_corridor or "").lower()
    has_optical_signal = any(sig in lower for sig in _CORRIDOR_OPTICAL_SIGNALS)
    # Word-boundary, KHÔNG phải substring: "object" (số ít, bị cấm) không được khớp nhầm
    # vào bên trong "objects" (số nhiều) -- chính là cụm "zero objects or text" mà hệ
    # thống đang DẠY LLM viết (xem mục 3 SYSTEM_PROMPT + template fallback ngay dưới) --
    # substring match cũ sẽ bác bỏ oan đúng câu mà LLM được dặn phải viết.
    has_banned_word = any(re.search(rf"\b{re.escape(w)}\b", lower) for w in _CORRIDOR_BANNED_WORDS)
    if raw_corridor and has_optical_signal and not has_banned_word:
        return raw_corridor.strip()

    material_hint = _extract_scene_material_hint(scene_prompt)
    if material_hint:
        return (
            f"Extension of the same {material_hint} from the main scene, extreme creamy bokeh, "
            "soft diffused lighting matching the scene ambiance and color temperature, "
            "completely clean negative space, absolutely zero objects, people, or text, "
            "perfect shallow depth of field"
        )
    return (
        "Soft ambient light and color gradient wash extending naturally from the main scene, "
        "zero specific texture or material, blending seamlessly with the scene's color temperature "
        "and mood, extreme creamy bokeh, completely clean negative space, absolutely zero objects, "
        "people, or text, perfect shallow depth of field"
    )


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
    # (l_frame_showcase kiểm tra TRƯỚC lifestyle_corner_pod: "khung góc chữ L" chứa
    # substring "góc" nên nếu đặt sau sẽ luôn bị nhánh "góc" ở dưới chặn mất, không
    # bao giờ chạy tới được -- xác nhận qua kiểm thử thực tế 2026-09-19)
    if any(k in prompt_lower for k in ["l-frame", "khung chữ l", "showroom", "khung góc chữ l"]):
        template = "l_frame_showcase"
        orientation = "right" if any(w in prompt_lower for w in ["phải", "right"]) else "left"

    elif any(k in prompt_lower for k in ["góc", "capsule", "corner", "bo góc"]):
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

    elif any(k in prompt_lower for k in ["khai trương", "grand opening", "lễ hội", "festive"]):
        template = "grand_opening_banner"

    elif any(k in prompt_lower for k in ["thẻ kính", "thiệp mời", "tri ân khách hàng", "luxury card", "thẻ sang trọng"]):
        template = "luxury_centered_card"
        orientation = "center"

    elif any(k in prompt_lower for k in ["menu", "thực đơn", "bảng giá", "bảng menu"]):
        template = "menu_price_board"
        orientation = "right" if any(w in prompt_lower for w in ["phải", "right"]) else "left"

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


# Chế độ đa phương án (num_variants > 1) khi KHÔNG có LLM (offline) -- không thể tự
# suy luận "N phương án khác biệt có chủ đích" như LLM thật, nên chỉ xoay vòng qua
# vài preset style cố định + đảo orientation nếu template hỗ trợ, giữ NGUYÊN template
# đã chọn theo category/keyword (lý do chọn nó vẫn đúng, không nên đổi tuỳ tiện).
_FALLBACK_STYLE_ROTATION = [
    StyleConfig(font="anton", theme_color="#06B6D4", text_effect="embossed", background_tone="vibrant"),
    StyleConfig(font="playfair", theme_color="#D4AF37", text_effect="3d_gold", background_tone="dark_luxury"),
    StyleConfig(font="lobster", theme_color="#F97316", text_effect="plain_elegant", background_tone="warm_rustic"),
    StyleConfig(font="days", theme_color="#38BDF8", text_effect="chrome", background_tone="cyber_neon"),
]

_ORIENTATION_MIRROR = {
    "left": "right", "right": "left",
    "bottom_left": "bottom_right", "bottom_right": "bottom_left",
    "top_left": "top_right", "top_right": "top_left",
}


def _expand_fallback_variants(primary: TendooCreativePlan, num_variants: int) -> List[TendooCreativePlan]:
    """Nhân bản `primary` thành `num_variants` phương án -- nội dung giữ nguyên 100%,
    chỉ đổi style (xoay vòng preset) và đảo orientation (nếu template hỗ trợ mirror)
    để vẫn có khác biệt thị giác thật dù không có LLM để tự suy luận."""
    if num_variants <= 1:
        return [primary]
    variants = [primary]
    orientation = primary.orientation
    for i in range(1, num_variants):
        style = _FALLBACK_STYLE_ROTATION[(i - 1) % len(_FALLBACK_STYLE_ROTATION)]
        orientation = _ORIENTATION_MIRROR.get(orientation, orientation)
        variants.append(replace(primary, style=style, orientation=orientation))
    return variants


def fallback_heuristic_planner(
    form_data: Dict[str, Any],
    prompt: str = "",
    aspect_ratio: str = "1:1",
    num_variants: int = 1,
) -> Union[TendooCreativePlan, List[TendooCreativePlan]]:
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
        raw_hl = form_data.get("feedback_highlights") or form_data.get("highlights")
        if raw_hl:
            for item in str(raw_hl).split(","):
                if item.strip():
                    extra_texts.append(item.strip())
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
    elif style_pref == "am_cung":
        font = "lobster"
        text_effect = "plain_elegant"
        background_tone = "warm_rustic"
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
        # Tông ấm cho prompt gợi bối cảnh vintage/retro/ánh nắng, còn lại mặc định tối sang trọng.
        background_tone = "warm_rustic" if any(
            k in prompt_lower for k in ["film", "grain", "retro", "vintage", "tạp chí", "cổ điển", "analog", "nắng", "ánh sáng", "hoàng hôn", "ấm", "morning"]
        ) else "dark_luxury"

    # Theme color: ưu tiên primary_color từ form
    theme_color = form_data.get("primary_color") or ("#06B6D4" if "neon" in prompt_lower else "#D4AF37")

    style = StyleConfig(
        font=font,
        theme_color=theme_color,
        text_effect=text_effect,
        background_tone=background_tone,
    )

    # 3. Tạo Zero-Text Prompts chuẩn mực cho DiT Base 4B
    clean_scene = sanitize_scene_prompt(prompt, subject_hint=hero or form_data.get("title", ""))
    clean_corridor = sanitize_corridor_prompt("", clean_scene)

    primary = TendooCreativePlan(
        template=template,
        hero=hero,
        # Không có LLM vẫn có tiêu đề nhiều cỡ khi hero có con số/cụm từ móc (GĐ 3).
        hero_parts=suggest_hero_parts(hero),
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
    log_plan_issues(primary)
    num_variants = max(1, min(5, int(num_variants or 1)))
    if num_variants <= 1:
        return primary
    return _expand_fallback_variants(primary, num_variants)


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



def _finalize_plan_from_dict(extracted_dict: Dict[str, Any], form_data: Dict[str, Any]) -> TendooCreativePlan:
    """Chuẩn hóa template và làm sạch 100% Zero-Text Background cho DiT Base 4B."""
    tpl_raw = str(extracted_dict.get("template") or "")
    tpl_norm = tpl_raw.strip().lower().replace(" ", "_").replace("-", "_")
    if tpl_norm in TEMPLATE_CATALOG:
        extracted_dict["template"] = tpl_norm
    else:
        logger.warning(f"[LLM Planner] Template lạ '{tpl_raw}' không có trong catalog -> dùng mặc định sandwich_top_heavy.")
        extracted_dict["template"] = "sandwich_top_heavy"

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
    log_plan_issues(plan)
    return plan


# Field nội dung PHẢI giống hệt nhau giữa mọi phương án trong chế độ đa ảnh (num_variants
# > 1) -- chỉ template/orientation/style được phép khác nhau. Ép buộc bằng code thay vì
# chỉ tin tưởng LLM tuân thủ đúng chỉ dẫn (đúng nguyên tắc zero-text sanitize đã áp dụng
# cho scene_prompt/corridor_prompt -- validate/enforce, không tin tưởng mù).
_CONTENT_FIELDS_TO_UNIFY = (
    "hero", "hero_parts", "subhead", "badge", "tag_left", "tag_right", "rating",
    "extra_texts", "cta", "store_info", "qr_code", "qr_label",
    "testimonial", "reviewer_name", "steps", "scene_prompt", "corridor_prompt",
)


def _finalize_plans_from_response(
    extracted_dict: Dict[str, Any],
    form_data: Dict[str, Any],
    num_variants: int,
) -> list:
    """Chuẩn hóa phản hồi LLM thành 1-N TendooCreativePlan.
    - num_variants <= 1: hành vi y hệt `_finalize_plan_from_dict` cũ (1 object phẳng).
    - num_variants > 1: kỳ vọng `{"plans": [...]}` (N phương án). Nếu LLM không tuân
      thủ đúng định dạng mảng (rủi ro thật với model nhỏ/local), coi `extracted_dict`
      là 1 phương án duy nhất và nhân bản -- không bao giờ crash vì sai định dạng.
      Sau khi parse, LUÔN ép các field nội dung + scene/corridor prompt giống hệt
      phương án đầu tiên, bất kể LLM có tuân thủ yêu cầu "giữ nguyên nội dung" hay không.
    """
    if num_variants <= 1:
        return [_finalize_plan_from_dict(dict(extracted_dict), form_data)]

    raw_plans = extracted_dict.get("plans")
    if not isinstance(raw_plans, list) or not raw_plans:
        logger.warning("[LLM Planner] Thiếu mảng 'plans' hợp lệ cho chế độ đa phương án -> nhân bản 1 phương án duy nhất.")
        raw_plans = [extracted_dict]

    plans = [_finalize_plan_from_dict(dict(p), form_data) for p in raw_plans[:num_variants] if isinstance(p, dict)]
    if not plans:
        plans = [_finalize_plan_from_dict(dict(extracted_dict), form_data)]
    while len(plans) < num_variants:
        plans.append(replace(plans[0]))

    canonical = plans[0]
    for p in plans[1:]:
        for field in _CONTENT_FIELDS_TO_UNIFY:
            setattr(p, field, getattr(canonical, field))
    return plans


def _try_run_local_qwen(
    messages: list,
    form_data: Dict[str, Any],
    debug_trace: Dict[str, Any],
    t_start: float,
    num_variants: int = 1,
) -> Optional[list]:
    """Thực thi suy luận thông qua mô hình Qwen3-4B cục bộ trên GPU (in-process)."""
    try:
        logger.info("[LLM Planner] 🚀 Khởi chạy suy luận qua Local Qwen3-4B trên GPU...")
        model, tokenizer = load_local_qwen3()
        raw_content = generate_local_qwen_response(model, tokenizer, messages)
        debug_trace["mode"] = "local_qwen3_4b"
        debug_trace["output"]["raw_text_response"] = raw_content

        extracted_dict = extract_balanced_json(raw_content)
        debug_trace["output"]["extracted_json"] = extracted_dict
        if not extracted_dict:
            logger.warning("[LLM Planner] Local Qwen3 không sinh được JSON hợp lệ.")
            return None

        plans = _finalize_plans_from_response(extracted_dict, form_data, num_variants)
        debug_trace["status"] = "success"
        debug_trace["output"]["final_plan"] = plans[0].to_dict()
        debug_trace["output"]["final_plans"] = [p.to_dict() for p in plans]
        debug_trace["latency_seconds"] = round(time.time() - t_start, 4)
        logger.info(f"[LLM Planner] ✅ Local Qwen3-4B thành công ({len(plans)} phương án, Template chính: {plans[0].template}, Latency: {debug_trace['latency_seconds']}s)")
        return plans
    except Exception as e:
        logger.warning(f"[LLM Planner] Không thể chạy Local Qwen3 ({e})")
        return None



def generate_creative_plan(
    form_data: Dict[str, Any],
    prompt: str = "",
    aspect_ratio: str = "1:1",
    model_override: Optional[str] = None,
    debug_save_path: Optional[Path | str] = None,
    return_debug: bool = False,
    num_variants: int = 1,
) -> Union[
    TendooCreativePlan,
    List[TendooCreativePlan],
    Tuple[TendooCreativePlan, Dict[str, Any]],
    Tuple[List[TendooCreativePlan], Dict[str, Any]],
]:
    """Tạo TendooCreativePlan từ form + prompt:
    Hỗ trợ 3 tầng kiến trúc:
      1. API (OpenAI-compatible vLLM endpoint) nếu cấu hình và khả dụng.
      2. Local Qwen3-4B CausalLM (chạy trực tiếp trên GPU in-process) không cần mạng ngoài.
      3. Fallback Heuristic (quy tắc xác định thông minh) bảo đảm 100% không bao giờ crash.

    `num_variants`: khi > 1 (tối đa 5, khớp giới hạn của UI), LLM được yêu cầu đề xuất
    N phương án sáng tạo KHÁC NHAU (template/style) cho CÙNG nội dung trong 1 lần gọi
    duy nhất -- xem `MULTI_VARIANT_INSTRUCTION_TEMPLATE` trong `llm_prompts.py`. Trả về `TendooCreativePlan`
    đơn lẻ như trước khi `num_variants <= 1` (mặc định, tương thích ngược 100%), hoặc
    `List[TendooCreativePlan]` khi > 1.
    """
    t_start = time.time()
    num_variants = max(1, min(5, int(num_variants or 1)))
    backend = os.environ.get("TENDOO_V3_LLM_BACKEND", LLM_BACKEND).lower()
    model_name = model_override or LLM_MODEL
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    user_payload = {
        "form_fields": form_data,
        "user_prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }
    user_message = f"Dữ liệu người dùng nhập:\n```json\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n```"
    if num_variants > 1:
        user_message += "\n" + MULTI_VARIANT_INSTRUCTION_TEMPLATE.format(n=num_variants)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Mỗi phương án JSON tốn ~500-900 token thực đo (bao gồm scene/corridor_prompt lặp
    # lại y hệt nhau giữa các phương án) -- trần cũ 2048 chỉ đủ cho 1 phương án, từng
    # là nguyên nhân Qwen3 bị cắt cụt JSON giữa chừng khi ép sinh nhiều object cùng lúc.
    max_tokens = 2048 if num_variants <= 1 else min(8192, 1200 + 1100 * num_variants)

    request_body = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    # `chat_template_kwargs`/`extra_body` (tắt "thinking mode") chỉ có ý nghĩa với dòng
    # Qwen3 -- các model khác (vd Gemma, Llama) không có khái niệm này, và tùy độ nghiêm
    # ngặt của server vLLM/OpenAI-compatible đích, field lạ có thể bị từ chối (400) thay
    # vì được âm thầm bỏ qua. Chỉ gửi khi model_name thực sự thuộc họ Qwen.
    # OpenAI họ suy luận (gpt-5*, o1/o3/o4): KHÔNG nhận `max_tokens` + temperature khác mặc định --
    # dùng `max_completion_tokens` (gồm cả token suy luận nên cấp rộng hơn) và reasoning thấp
    # (lập plan là việc ngữ nghĩa ngắn; GĐ 3R thử bằng OpenAI API).
    m_low = model_name.lower()
    if m_low.startswith(("gpt-5", "o1", "o3", "o4")):
        request_body.pop("temperature", None)
        request_body.pop("max_tokens", None)
        request_body["max_completion_tokens"] = max_tokens * 4
        request_body["reasoning_effort"] = os.environ.get("TENDOO_V3_LLM_REASONING_EFFORT", "low")
    if "qwen" in model_name.lower():
        request_body["chat_template_kwargs"] = {"enable_thinking": False}
        request_body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    debug_trace: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "llm_api" if backend != "local" else "local_qwen3_4b",
        "status": "pending",
        "input": {
            "backend": backend,
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
            "num_variants": num_variants,
        },
        "output": {
            "http_status_code": None,
            "raw_text_response": None,
            "raw_api_response": None,
            "extracted_json": None,
            "final_plan": None,
            "final_plans": None,
        },
        "latency_seconds": None,
        "error": None,
    }

    def _pack(plans: list):
        """Ghi debug_trace + trả kết quả -- gói lại pattern lặp 7 lần trong hàm này
        thành 1 chỗ duy nhất, giảm rủi ro quên cập nhật 1 nhánh khi sửa sau này."""
        # Cổng 3 (GĐ 3): mọi nhánh (API/local/dự phòng) đều qua phủ quyết dung lượng ở đây.
        routed = [route_template(dedupe_plan(p), aspect_ratio) for p in plans]
        plans = [p for p, _ in routed]
        debug_trace["gate3"] = [why for _, why in routed if why]
        debug_trace["output"]["final_plan"] = plans[0].to_dict()
        debug_trace["output"]["final_plans"] = [p.to_dict() for p in plans]
        debug_trace["latency_seconds"] = round(time.time() - t_start, 4)
        _save_debug_trace(debug_trace, debug_save_path)
        result = plans if num_variants > 1 else plans[0]
        if return_debug:
            return result, debug_trace
        return result

    # =========================================================================
    # NHÁNH 1: Chế độ THUẦN LOCAL (chạy trực tiếp Qwen 4B có sẵn trên GPU)
    # =========================================================================
    if backend == "local":
        plans = _try_run_local_qwen(messages, form_data, debug_trace, t_start, num_variants)
        if plans:
            return _pack(plans)

        logger.warning("[LLM Planner] Local Qwen3 không khả dụng -> Chuyển sang Fallback Heuristic.")
        fb = fallback_heuristic_planner(form_data, prompt, aspect_ratio, num_variants)
        plans = fb if isinstance(fb, list) else [fb]
        debug_trace["mode"] = "fallback_heuristic"
        debug_trace["status"] = "fallback_local_unavailable"
        return _pack(plans)

    # =========================================================================
    # NHÁNH 2: Chế độ API hoặc AUTO (ưu tiên API, tự động failover sang Local)
    # =========================================================================
    api_key = os.environ.get("TENDOO_V3_LLM_API_KEY") or LLM_API_KEY

    # Nếu chưa có API key và không cho phép chế độ no-key:
    if not api_key and not os.environ.get("TENDOO_V3_LLM_ALLOW_NO_KEY"):
        if backend == "auto":
            # Tự động chuyển ngay sang Local Qwen3
            plans = _try_run_local_qwen(messages, form_data, debug_trace, t_start, num_variants)
            if plans:
                return _pack(plans)

        logger.info("[LLM Planner] Chưa cấu hình TENDOO_V3_LLM_API_KEY -> Sử dụng Fallback Heuristic.")
        fb = fallback_heuristic_planner(form_data, prompt, aspect_ratio, num_variants)
        plans = fb if isinstance(fb, list) else [fb]
        debug_trace["mode"] = "fallback_heuristic"
        debug_trace["status"] = "fallback_no_api_key"
        debug_trace["error"] = "TENDOO_V3_LLM_API_KEY chưa được cấu hình. Hệ thống tự động chuyển sang Fallback Heuristic."
        return _pack(plans)

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        logger.info(f"[LLM Planner] Gọi {model_name} tại {url}...")
        # Thử lại 1 lần với lỗi TẠM THỜI (timeout, mất kết nối, 429/5xx) -- GĐ 3R: 2/16 brief rơi về
        # dự phòng vì lỗi thoáng qua, gọi lại ngay thì thành công.
        for attempt in range(2):
            try:
                resp = requests.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=(LLM_CONNECT_TIMEOUT_S, LLM_TIMEOUT_S),
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                    logger.warning(f"[LLM Planner] HTTP {resp.status_code} -> thử lại 1 lần")
                    time.sleep(2)
                    continue
                break
            except (requests.Timeout, requests.ConnectionError) as net_err:
                if attempt == 1:
                    raise
                logger.warning(f"[LLM Planner] Lỗi mạng tạm thời ({net_err}) -> thử lại 1 lần")
        debug_trace["output"]["http_status_code"] = resp.status_code
        resp.raise_for_status()
        resp_data = resp.json()
        debug_trace["output"]["raw_api_response"] = resp_data

        content = resp_data["choices"][0]["message"]["content"]
        debug_trace["output"]["raw_text_response"] = content

        extracted_dict = extract_balanced_json(content)
        debug_trace["output"]["extracted_json"] = extracted_dict

        if not extracted_dict:
            # Chi tiết lỗi thật (cắt ngang vs sai cú pháp, kèm vị trí) đã được log ở
            # WARNING ngay phía trên bởi extract_balanced_json() -- preview 400 ký tự ở
            # đây chỉ để xem nhanh, KHÔNG phải nguồn chẩn đoán chính (raw content đầy đủ
            # cũng đã lưu trong debug_trace["output"]["raw_text_response"]).
            raise ValueError(f"Phản hồi từ LLM không chứa JSON hợp lệ (xem log WARNING phía trên để biết chi tiết cắt ngang/sai cú pháp). Raw content preview: {content[:400]}")

        plans = _finalize_plans_from_response(extracted_dict, form_data, num_variants)
        debug_trace["status"] = "success"
        logger.info(f"[LLM Planner] ✅ Kế hoạch sáng tạo API thành công ({len(plans)} phương án, Template chính: {plans[0].template}, Latency: {round(time.time() - t_start, 4)}s)")
        return _pack(plans)

    except Exception as e:
        logger.warning(f"[LLM Planner] Lỗi kết nối tới LLM API ({e}).")
        # Failover tự động sang Local Qwen3 nếu ở chế độ auto
        if backend == "auto":
            logger.info("[LLM Planner] 🔄 Tự động chuyển vùng sang Local Qwen3-4B trên GPU...")
            plans = _try_run_local_qwen(messages, form_data, debug_trace, t_start, num_variants)
            if plans:
                return _pack(plans)

        # Fallback Heuristic nếu cả API lẫn Local đều không chạy được
        logger.warning("[LLM Planner] Kích hoạt Fallback Heuristic.")
        fb = fallback_heuristic_planner(form_data, prompt, aspect_ratio, num_variants)
        plans = fb if isinstance(fb, list) else [fb]
        debug_trace["mode"] = "fallback_heuristic"
        debug_trace["status"] = "fallback_exception"
        debug_trace["error"] = f"Exception: {type(e).__name__}: {str(e)}"
        return _pack(plans)


__all__ = [
    "_save_debug_trace",
    "extract_balanced_json",
    "fallback_heuristic_planner",
    "free_local_qwen3",
    "generate_creative_plan",
    "generate_local_qwen_response",
    "load_local_qwen3",
    "parse_user_prompt_intents",
    "resolve_qwen3_checkpoint_path",
    "resolve_tokenizer_path",
    "sanitize_corridor_prompt",
    "sanitize_scene_prompt",
    "LLM_BACKEND",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tendoo v3 LLM Planner Test")
    parser.add_argument("--backend", type=str, default=None, choices=["local", "api", "auto"], help="Override LLM backend")
    args = parser.parse_args()

    if args.backend:
        os.environ["TENDOO_V3_LLM_BACKEND"] = args.backend

    active_backend = os.environ.get("TENDOO_V3_LLM_BACKEND", LLM_BACKEND).lower()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 70)
    print("🔍 TENDOO v3 LLM PLANNER - CONNECTION & INFERENCE TEST")
    print(f"   Active Backend: {active_backend.upper()}")
    if active_backend in ["api", "auto"]:
        print(f"   API Base URL:   {LLM_BASE_URL}")
        print(f"   API Model:      {LLM_MODEL}")
        key_disp = f"CONFIGURED (***{LLM_API_KEY[-4:]})" if LLM_API_KEY else "NOT SET"
        print(f"   API Key:        {key_disp}")
    if active_backend in ["local", "auto"]:
        resolved_cp = resolve_qwen3_checkpoint_path("4B")
        print(f"   Local Qwen3:    {resolved_cp}")
    print("=" * 70)

    test_form = {
        "category": "promo",
        "title": "Mì Hảo Hảo Tôm Chua Cay",
        "discount": "GIẢM 50%",
        "applied_product": "Thùng 30 gói",
        "store_name": "Acecook Mart",
    }
    test_prompt = "Poster phong cách điện ảnh ấm cúng, chữ đặt ở góc trên bên trái"
    print("\n👉 Gửi request thử nghiệm tới LLM Planner...")
    plan, debug_info = generate_creative_plan(test_form, prompt=test_prompt, return_debug=True)

    print(f"\n📊 KẾT QUẢ KIỂM TRA:")
    print(f"   Mode:               {debug_info.get('mode')}")
    print(f"   Status:             {debug_info.get('status')}")
    print(f"   Latency:            {debug_info.get('latency_seconds')}s")
    if debug_info.get("error"):
        print(f"   Notice/Error:       {debug_info.get('error')}")
    print(f"   Template được chọn: {plan.template}")
    print(f"   Orientation:        {plan.orientation}")
    print(f"   Hero text:          {plan.hero}")
    print(f"   Scene Prompt:       {plan.scene_prompt[:120]}...")
    print("=" * 70)


