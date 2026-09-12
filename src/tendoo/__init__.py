"""
Tendoo AI Core Package.
=======================
Hệ thống AI sinh Poster Thương mại Tự động cao cấp kết hợp mô hình Diffusion
(FLUX.2-klein-base-4B) và Vector HTML/CSS Typography đa khối tiếng Việt.

KIẾN TRÚC 5 PHÂN TẦNG (5-LAYER ARCHITECTURE):
--------------------------------------------
1. tendoo.core:
   - Các thực thể nền tảng (AdaptiveBlock, ColorPalette, PosterContent).
   - Quản lý font chữ chuẩn tiếng Việt (19 fonts), phân tích màu WCAG, SVG icons.
   - Thước đo hình học chữ (wrap_and_measure, balance_vietnamese_headline).

2. tendoo.engine:
   - Hệ tọa độ ma trận 9 ô không gian (3x3 Spatial Grid) & Hành lang an toàn (Product Sanctuary).
   - Thuật toán sinh mặt nạ corridor mask giải tích và trình duyệt (generate_omni_corridor_mask).
   - Layout đa khối toàn năng: OmniBlockLayout.

3. tendoo.layouts:
   - Bộ điều phối phong cách thông minh: style_matcher (resolve_style_preset).
   - Bộ cân bằng ngữ pháp tiếng Việt: text_engine (37 từ ghép, line balancing).
   - Cung cấp backward-compatibility facade cho các layout thế hệ trước.

4. tendoo.velocity_blending:
   - Thuật toán khuếch tán không gian Regional Velocity Blending cho FLUX.2 DiT 4B Base.
   - Mã hóa sản phẩm thật In-Context RoPE tại t=10.0 (load_and_encode_ref_image).

5. tendoo.poster_renderer:
   - Trình render vector HTML/CSS sang ảnh PNG pixel-perfect qua Playwright Headless Chromium.
"""

from tendoo.core.base import ColorPalette, PosterContent
from tendoo.engine.blocks import AdaptiveBlock
from tendoo.engine import OmniBlockLayout
from tendoo.layouts import (
    BaseLayout,
    analyze_color_harmony,
    balance_vietnamese_headline,
    get_layout,
    list_layouts,
    normalize_text,
)
from tendoo.poster_renderer import PosterRenderer

__all__ = [
    "AdaptiveBlock",
    "BaseLayout",
    "ColorPalette",
    "OmniBlockLayout",
    "PosterContent",
    "PosterRenderer",
    "analyze_color_harmony",
    "balance_vietnamese_headline",
    "get_layout",
    "list_layouts",
    "normalize_text",
]
