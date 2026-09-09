"""
Tendoo Layouts Architecture.

Modular, decoupled architecture pairing visual topology masks with adaptive sub-pixel HTML5 templates.
"""

from tendoo.layouts.base import BaseLayout, CALENDAR_ICON_SVG, ColorPalette, PosterContent
from tendoo.layouts.color_engine import analyze_color_harmony, hex_to_hue
from tendoo.layouts.component_engine import get_component_css, render_category_body
from tendoo.layouts.font_engine import FONT_CATALOG, list_font_options, recommend_font, resolve_font
from tendoo.layouts.registry import get_layout, list_layouts, register_layout
from tendoo.layouts.style_matcher import LAYOUT_COMPATIBLE_STYLES, resolve_style_preset
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text

__all__ = [
    "BaseLayout",
    "CALENDAR_ICON_SVG",
    "ColorPalette",
    "PosterContent",
    "get_layout",
    "list_layouts",
    "register_layout",
    "analyze_color_harmony",
    "hex_to_hue",
    "balance_vietnamese_headline",
    "normalize_text",
    "render_category_body",
    "get_component_css",
    "resolve_font",
    "recommend_font",
    "list_font_options",
    "FONT_CATALOG",
    "resolve_style_preset",
    "LAYOUT_COMPATIBLE_STYLES",
]

