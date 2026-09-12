"""
src/tendoo/layouts

Pure OmniBlock Layouts Architecture:
====================================
- Cổng tích hợp các động cơ tạo hình (Typography, Color Engine, Font Engine, Style Matcher).
- Thống nhất 100% toàn bộ hệ thống xoay quanh OmniBlockLayout (`omni`).
- 100% tự chứa, loại bỏ toàn bộ cơ chế ủy quyền legacy và các phụ thuộc ngoại lai.
"""

from __future__ import annotations

from typing import Any

from tendoo.core.base import (
    ARROW_RIGHT_ICON_SVG,
    BaseLayout,
    CALENDAR_ICON_SVG,
    CHECK_ICON_SVG,
    CLOCK_ICON_SVG,
    ColorPalette,
    GIFT_ICON_SVG,
    GLOBE_ICON_SVG,
    LOCATION_ICON_SVG,
    PHONE_ICON_SVG,
    PosterContent,
    STAR_ICON_SVG,
    TAG_ICON_SVG,
)
from tendoo.layouts.color_engine import analyze_color_harmony, hex_to_hue
from tendoo.layouts.font_engine import (
    FONT_CATALOG,
    list_font_options,
    recommend_font,
    resolve_font,
)
from tendoo.layouts.registry import get_layout, list_layouts, register_layout
from tendoo.layouts.style_matcher import (
    CATEGORY_FIELD_SLOTS,
    DEFAULT_FIELD_ROLE,
    DEFAULT_LAYOUT,
    LAYOUT_COMPATIBLE_STYLES,
    OMNI_STYLES,
    STYLE_PREFS,
    STYLE_PREF_TO_STYLE_HINT,
    inject_color_guidance,
    resolve_style_preset,
)
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text


def __getattr__(name: str) -> Any:
    """Nạp lười OmniBlockLayout để phá vỡ vòng lặp import giữa engine và layouts."""
    if name == "OmniBlockLayout":
        from tendoo.engine.layout import OmniBlockLayout
        return OmniBlockLayout
    raise AttributeError(f"module 'tendoo.layouts' has no attribute '{name}'")


__all__ = [
    "ARROW_RIGHT_ICON_SVG",
    "BaseLayout",
    "CALENDAR_ICON_SVG",
    "CATEGORY_FIELD_SLOTS",
    "CHECK_ICON_SVG",
    "CLOCK_ICON_SVG",
    "ColorPalette",
    "DEFAULT_FIELD_ROLE",
    "DEFAULT_LAYOUT",
    "FONT_CATALOG",
    "GIFT_ICON_SVG",
    "GLOBE_ICON_SVG",
    "LAYOUT_COMPATIBLE_STYLES",
    "LOCATION_ICON_SVG",
    "OMNI_STYLES",
    "OmniBlockLayout",
    "PHONE_ICON_SVG",
    "PosterContent",
    "STAR_ICON_SVG",
    "STYLE_PREFS",
    "STYLE_PREF_TO_STYLE_HINT",
    "TAG_ICON_SVG",
    "analyze_color_harmony",
    "balance_vietnamese_headline",
    "get_layout",
    "hex_to_hue",
    "inject_color_guidance",
    "list_font_options",
    "list_layouts",
    "normalize_text",
    "recommend_font",
    "register_layout",
    "resolve_font",
    "resolve_style_preset",
]
