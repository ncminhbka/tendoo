"""
src/tendoo/layouts/style_matcher.py

Pure OmniBlock Style Auto-Matching Engine:
==========================================
- Triết lý cốt lõi: "Everything is an Adaptive Block" trên nền tảng OmniBlockLayout (`omni`).
- Bố cục poster là BẤT BIẾN (Invariant: DEFAULT_LAYOUT = "omni") cho toàn bộ 6 ngành hàng.
- Thứ thay đổi theo yêu cầu người dùng là Phong cách Thị giác (Style Hint) và Font chữ,
  được ánh xạ tất định 1-1 từ tâm trạng thương mại (style_pref) sang phong cách thẩm mỹ OmniBlock.
- Tuyệt đối loại bỏ mọi tàn dư của 7 layout template cũ.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Hằng số bố cục duy nhất của toàn bộ hệ thống Tendoo AI
DEFAULT_LAYOUT = "omni"

# Danh mục 11 phong cách thẩm mỹ thương mại cao cấp của Omni-Block
OMNI_STYLES: List[str] = [
    "luxury_gold",
    "cyberpunk_grid",
    "warm_wood",
    "festive_light",
    "minimal_wall",
    "champagne_silk",
    "studio_spotlight",
    "moonbeam",
    "daylight_clean",
    "tech_minimal",
    "cinematic_asphalt",
]

# Cấu hình phong cách tương thích cho Omni-Block
LAYOUT_COMPATIBLE_STYLES: Dict[str, Dict[str, object]] = {
    "omni": {
        "styles": OMNI_STYLES,
        "default": "luxury_gold",
    },
}

# 6 tâm trạng kinh doanh người dùng lựa chọn trên giao diện
STYLE_PREFS = ("auto", "hien_dai", "sang_trong", "nang_dong", "am_cung", "le_hoi")

# Ánh xạ trực tiếp 1-1 từ tâm trạng thương mại sang Style Hint của OmniBlock
STYLE_PREF_TO_STYLE_HINT: Dict[str, str] = {
    "auto": "auto",
    "sang_trong": "luxury_gold",
    "nang_dong": "cyberpunk_grid",
    "hien_dai": "minimal_wall",
    "am_cung": "warm_wood",
    "le_hoi": "festive_light",
}

# Danh mục các trường dữ liệu theo 6 ngành hàng thương mại
CATEGORY_FIELD_SLOTS: Dict[str, List[str]] = {
    "promo": ["discount", "applied_product", "date_start", "date_end"],
    "product_intro": ["product_name", "price", "product_desc", "highlights"],
    "opening": ["opening_date", "opening_promo", "booking_contact"],
    "feedback": ["feedback_target", "feedback_quote", "feedback_rating", "special_offer"],
    "recruitment": ["job_position", "job_desc", "apply_deadline", "apply_method"],
    "guide": ["guide_steps"],
}

# Vai trò thị giác mặc định (AdaptiveBlock role) cho từng trường
DEFAULT_FIELD_ROLE: Dict[str, str] = {
    "discount": "badge",
    "applied_product": "body",
    "date_start": "caption",
    "date_end": "caption",
    "product_name": "subtitle",
    "price": "badge",
    "product_desc": "body",
    "highlights": "body",
    "opening_date": "badge",
    "opening_promo": "body",
    "booking_contact": "caption",
    "feedback_target": "subtitle",
    "feedback_quote": "body",
    "feedback_rating": "badge",
    "special_offer": "body",
    "job_position": "subtitle",
    "job_desc": "body",
    "apply_deadline": "caption",
    "apply_method": "caption",
    "guide_steps": "body",
}


def resolve_style_preset(
    category: str,
    style_pref: str = "auto",
    scene_prompt: str = "",
) -> Dict[str, str]:
    """
    Điều phối phong cách thẩm mỹ và font chữ tất định cho poster Omni-Block.
    
    TẠI SAO CẦN LÀM:
    - Bố cục poster luôn là DEFAULT_LAYOUT ('omni') cho mọi ngành hàng.
    - Ánh xạ tâm trạng kinh doanh (style_pref) sang phong cách thẩm mỹ tương ứng.
    - Không gọi API ngoại bộ, tốc độ < 0.1ms, bảo đảm tính tất định 100%.
    """
    clean_cat = (category or "promo").lower().strip()
    clean_pref = (style_pref or "auto").lower().strip()
    if clean_pref not in STYLE_PREFS:
        clean_pref = "auto"

    style_hint = STYLE_PREF_TO_STYLE_HINT.get(clean_pref, "auto")

    return {
        "layout": DEFAULT_LAYOUT,
        "style_hint": style_hint,
        "font_key": "auto",
    }


def inject_color_guidance(prompt: str, hex_color: Optional[str]) -> str:
    """
    Cổng re-export hàm dẫn dắt màu sắc từ core.style cho các module trong layouts/.
    """
    from tendoo.core.style import inject_color_guidance as _core_inject
    return _core_inject(prompt, hex_color)


__all__ = [
    "CATEGORY_FIELD_SLOTS",
    "DEFAULT_FIELD_ROLE",
    "DEFAULT_LAYOUT",
    "LAYOUT_COMPATIBLE_STYLES",
    "OMNI_STYLES",
    "STYLE_PREFS",
    "STYLE_PREF_TO_STYLE_HINT",
    "inject_color_guidance",
    "resolve_style_preset",
]
