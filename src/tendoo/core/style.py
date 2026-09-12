"""
src/tendoo/core/style.py

Tầng cốt lõi quản lý Kiểu dáng, Chủ đề Thẩm mỹ & Dẫn dắt Màu sắc (Style Core):
=============================================================================
- Cổng kết nối trung tâm cho logic ánh xạ phong cách tự động (Deterministic Style Auto-Matching).
- Quản lý các slot trường theo ngành hàng (`CATEGORY_FIELD_SLOTS`) cho cả Form UI, LLM sidecar và Server.
- Cung cấp hàm dẫn dắt màu sắc chủ đạo (`inject_color_guidance`) vào câu lệnh prompt của AI Diffusion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Re-export các hằng số và hàm cốt lõi từ layouts/style_matcher (Single Source of Truth)
from tendoo.layouts.style_matcher import (
    DEFAULT_LAYOUT,
    CATEGORY_FIELD_SLOTS,
    DEFAULT_FIELD_ROLE,
    LAYOUT_COMPATIBLE_STYLES,
    OMNI_STYLES,
    STYLE_PREFS,
    STYLE_PREF_TO_STYLE_HINT,
    resolve_style_preset,
)

# Bảng phân nhóm góc Hue (0 - 360 độ) sang tên gọi màu sắc tiếng Anh tự nhiên.
# Dùng để bổ sung chỉ dẫn màu sắc vào Prompt cho DiT (FLUX.2 Base 4B).
_HUE_NAME_BUCKETS: List[Tuple[int, str]] = [
    (15, "warm red"),
    (45, "warm amber orange"),
    (70, "golden yellow"),
    (170, "fresh green"),
    (200, "cyan teal"),
    (250, "cool blue"),
    (290, "deep purple violet"),
    (330, "vivid magenta pink"),
    (360, "warm red"),
]


def _hue_to_color_name(hue: int) -> str:
    """Ánh xạ góc màu Hue sang chuỗi mô tả màu sắc tiếng Anh chuẩn."""
    for upper_bound, name in _HUE_NAME_BUCKETS:
        if hue <= upper_bound:
            return name
    return "warm red"


def inject_color_guidance(prompt: str, hex_color: Optional[str]) -> str:
    """Bổ sung mệnh đề chỉ dẫn tông màu chủ đạo vào câu prompt của cảnh nền hoặc hành lang.
    
    NGUYÊN LÝ HOẠT ĐỘNG:
    1. Chuyển đổi mã Hex color do người dùng chọn trên Web UI thành góc màu Hue (0 - 360).
    2. Ánh xạ Hue sang cụm từ màu sắc tự nhiên (ví dụ `#06B6D4` -> Hue 188 -> "cyan teal accent tones").
    3. Nối cụm từ này vào câu prompt nếu chưa tồn tại, tránh lặp từ.
    4. Trả về nguyên bản nếu không có mã màu hex hợp lệ.
    
    TẠI SAO CẦN LÀM:
    - Khi người dùng chọn "Màu chủ đạo" trên giao diện, nếu chỉ đổi màu text CSS thì bức ảnh nền do AI
      sinh ra vẫn có thể mang tông màu hoàn toàn lệch pha (ví dụ text xanh neon trên nền ảnh vàng rực).
    - Việc "tiêm" nhẹ nhàng cụm từ chỉ dẫn màu vào Prompt giúp mô hình DiT hướng sự chú ý và ánh sáng
      của bức ảnh theo đúng dải màu người dùng mong muốn, tạo nên sự đồng bộ thẩm mỹ hoàn hảo.
    """
    from tendoo.layouts.color_engine import hex_to_hue

    hue = hex_to_hue(hex_color) if hex_color else None
    if hue is None:
        return prompt
    color_name = _hue_to_color_name(hue)
    clause = f"{color_name} accent tones"
    if clause in (prompt or ""):
        return prompt
    return f"{prompt}, {clause}" if prompt else clause


__all__ = [
    "DEFAULT_LAYOUT",
    "CATEGORY_FIELD_SLOTS",
    "DEFAULT_FIELD_ROLE",
    "LAYOUT_COMPATIBLE_STYLES",
    "OMNI_STYLES",
    "STYLE_PREFS",
    "STYLE_PREF_TO_STYLE_HINT",
    "inject_color_guidance",
    "resolve_style_preset",
]
