"""
Tendoo Layouts Architecture.

Modular, decoupled architecture pairing visual topology masks with adaptive sub-pixel HTML5 templates.
"""

from tendoo.layouts.base import BaseLayout, CALENDAR_ICON_SVG, ColorPalette, PosterContent
from tendoo.layouts.color_engine import analyze_color_harmony
from tendoo.layouts.component_engine import get_component_css, render_category_body
from tendoo.layouts.registry import get_layout, list_layouts, register_layout
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
    "balance_vietnamese_headline",
    "normalize_text",
    "render_category_body",
    "get_component_css",
]

