"""
src/tendoo/core/style.py

Tầng cốt lõi quản lý Kiểu dáng, Chủ đề Thẩm mỹ & Dẫn dắt Màu sắc (Style Core):
=============================================================================
- Triết lý cốt lõi: "Everything is an Adaptive Block" trên nền tảng OmniBlockLayout (`omni`).
- Bố cục poster là BẤT BIẾN (Invariant: DEFAULT_LAYOUT = "omni") cho toàn bộ 6 ngành hàng.
- Thứ thay đổi theo yêu cầu người dùng là Phong cách Thị giác (Style Hint) và Font chữ,
  được ánh xạ tất định 1-1 từ tâm trạng thương mại (style_pref) sang phong cách thẩm mỹ OmniBlock.
- Cung cấp hàm dẫn dắt màu sắc chủ đạo (`inject_color_guidance`) vào câu lệnh prompt của AI Diffusion.

(2026-09-13: gộp từ `layouts/style_matcher.py` -- trước đó `core/style.py` chỉ re-export từ
`layouts/style_matcher.py` (ngoại trừ `inject_color_guidance`, vốn đã là bản thật ở đây từ
trước). `core/` giờ là tầng nền tảng DUY NHẤT. Bảng category/field slot chuyển hẳn sang
`core/category_schema.py` -- xem file đó để biết lý do (gộp với `engine/blocks.py`'s
`FIELD_BLOCK_SPECS`, từng là 2 bảng song song không đồng bộ gây bug thật).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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

# Hiệu ứng chữ mặc định cho tiêu đề chính theo từng style_hint, khi không ai chỉ định effect
# tường minh (2026-09-13 fix). Trước đây `layout.py` chỉ áp effect mặc định khi
# `default_effect_name` là 1 trong các tên effect thật -- giá trị mặc định thực tế của
# `content.text_effect` luôn là "auto" (không nằm trong danh sách đó), nên MỌI tiêu đề không set
# effect tường minh render chữ phẳng, không neon/3D/chrome dù CSS các hiệu ứng đó đã có sẵn và
# đúng. Bảng này ánh xạ 11 `OMNI_STYLES` sang 1 trong các effect có CSS thật trong
# `engine/templates/master.html` (neon_cyan/neon_pink/neon_amber/neon_green/3d_gold/chrome/fire/
# shadow) -- lựa chọn theo tâm trạng thẩm mỹ của từng style, có thể tinh chỉnh sau nếu cần.
STYLE_HINT_DEFAULT_EFFECT: Dict[str, str] = {
    "luxury_gold": "3d_gold",
    "cyberpunk_grid": "neon_cyan",
    "warm_wood": "fire",
    "festive_light": "neon_amber",
    "minimal_wall": "shadow",
    "champagne_silk": "chrome",
    "studio_spotlight": "shadow",
    "moonbeam": "neon_cyan",
    "daylight_clean": "shadow",
    "tech_minimal": "chrome",
    "cinematic_asphalt": "shadow",
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


def resolve_style_preset(
    category: str,
    style_pref: str = "auto",
) -> Dict[str, str]:
    """
    Điều phối phong cách thẩm mỹ và font chữ tất định cho poster Omni-Block.

    TẠI SAO CẦN LÀM:
    - Bố cục poster luôn là DEFAULT_LAYOUT ('omni') cho mọi ngành hàng.
    - Ánh xạ tâm trạng kinh doanh (style_pref) sang phong cách thẩm mỹ tương ứng.
    - Không gọi API ngoại bộ, tốc độ < 0.1ms, bảo đảm tính tất định 100%.
    """
    clean_pref = (style_pref or "auto").lower().strip()
    if clean_pref not in STYLE_PREFS:
        clean_pref = "auto"

    style_hint = STYLE_PREF_TO_STYLE_HINT.get(clean_pref, "auto")

    return {
        "layout": DEFAULT_LAYOUT,
        "style_hint": style_hint,
        "font_key": "auto",
    }


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
    from tendoo.core.colors import hex_to_hue

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
    "LAYOUT_COMPATIBLE_STYLES",
    "OMNI_STYLES",
    "STYLE_HINT_DEFAULT_EFFECT",
    "STYLE_PREFS",
    "STYLE_PREF_TO_STYLE_HINT",
    "inject_color_guidance",
    "resolve_style_preset",
]
