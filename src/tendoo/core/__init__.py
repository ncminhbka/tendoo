"""
src/tendoo/core

Tầng cốt lõi đồ họa & typography cho hệ thống Tendoo AI:
========================================================
- base: BaseLayout, PosterContent, ColorPalette, SVG icons vector
- fonts: Quản lý Font Catalog 19 họ font thương mại tiếng Việt & font-face CSS
- colors: Color harmony (đơn sắc, tương phản, tam giác màu) & WCAG contrast ratio
- typography: Bounding box math, greedy word-wrapping, PIL font fitting & Vietnamese line balancing
- components: SVG icons vector & visual building blocks (O(1) semantic icon lookup)
- style: Category slot definitions, style presets & color guidance injection

TẠI SAO CẦN TẦNG CORE ĐỘC LẬP:
- Tách biệt rõ ràng giữa Business Logic / Data Models (core) và Spatial Layout / Rendering (engine & renderer).
- Cho phép bất kỳ module nào (LLM planner, API server, layout generator, visual test suite)
  dễ dàng import các hằng số, kiểu dữ liệu, và công cụ typography mà không bị dính vòng lặp phụ thuộc (circular imports).
"""

from tendoo.core.base import (
    BaseLayout,
    PosterContent,
    ColorPalette,
    CALENDAR_ICON_SVG,
    PHONE_ICON_SVG,
    LOCATION_ICON_SVG,
    GLOBE_ICON_SVG,
    GIFT_ICON_SVG,
    TAG_ICON_SVG,
    STAR_ICON_SVG,
    CLOCK_ICON_SVG,
    CHECK_ICON_SVG,
    ARROW_RIGHT_ICON_SVG,
)
from tendoo.core.fonts import (
    FONT_CATALOG,
    FONTS_DIR,
    resolve_font,
    recommend_font,
    list_font_options,
)
from tendoo.core.colors import (
    analyze_color_harmony,
    hex_to_hue,
)
from tendoo.core.typography import (
    LINE_HEIGHT_MULT,
    balance_vietnamese_headline,
    fit_font_size_px,
    normalize_text,
    resolve_headline_effect,
    strip_emojis,
    wrap_and_measure,
)
from tendoo.core.components import (
    ICON_SVG_BY_NAME,
    render_category_body,
    get_component_css,
)
from tendoo.core.style import (
    CATEGORY_FIELD_SLOTS,
    DEFAULT_FIELD_ROLE,
    LAYOUT_COMPATIBLE_STYLES,
    resolve_style_preset,
    inject_color_guidance,
)

__all__ = [
    "BaseLayout",
    "PosterContent",
    "ColorPalette",
    "CALENDAR_ICON_SVG",
    "PHONE_ICON_SVG",
    "LOCATION_ICON_SVG",
    "GLOBE_ICON_SVG",
    "GIFT_ICON_SVG",
    "TAG_ICON_SVG",
    "STAR_ICON_SVG",
    "CLOCK_ICON_SVG",
    "CHECK_ICON_SVG",
    "ARROW_RIGHT_ICON_SVG",
    "FONT_CATALOG",
    "FONTS_DIR",
    "LINE_HEIGHT_MULT",
    "resolve_font",
    "recommend_font",
    "list_font_options",
    "analyze_color_harmony",
    "hex_to_hue",
    "balance_vietnamese_headline",
    "fit_font_size_px",
    "wrap_and_measure",
    "normalize_text",
    "resolve_headline_effect",
    "strip_emojis",
    "ICON_SVG_BY_NAME",
    "render_category_body",
    "get_component_css",
    "CATEGORY_FIELD_SLOTS",
    "DEFAULT_FIELD_ROLE",
    "LAYOUT_COMPATIBLE_STYLES",
    "resolve_style_preset",
    "inject_color_guidance",
]
