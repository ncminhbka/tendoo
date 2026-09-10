"""
src/tendoo/core

Tầng cốt lõi đồ họa & typography cho hệ thống Tendoo AI:
- fonts: Quản lý font catalog & font-face CSS
- colors: Color harmony & WCAG contrast ratio
- typography: Bounding box math, measurement & line balancing
- components: SVG icons & visual building blocks
"""

from tendoo.core.fonts import FONT_CATALOG, FONTS_DIR, resolve_font, recommend_font, list_font_options
from tendoo.core.colors import ColorPalette, analyze_color_harmony, hex_to_hue
from tendoo.core.typography import (
    balance_vietnamese_headline,
    fit_font_size_px,
    normalize_text,
    resolve_headline_effect,
    strip_emojis,
)
from tendoo.core.components import ICON_SVG_BY_NAME, render_category_body

__all__ = [
    "FONT_CATALOG",
    "FONTS_DIR",
    "resolve_font",
    "recommend_font",
    "list_font_options",
    "ColorPalette",
    "analyze_color_harmony",
    "hex_to_hue",
    "balance_vietnamese_headline",
    "fit_font_size_px",
    "normalize_text",
    "resolve_headline_effect",
    "strip_emojis",
    "ICON_SVG_BY_NAME",
    "render_category_body",
]
