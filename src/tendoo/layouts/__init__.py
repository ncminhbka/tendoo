"""
Tendoo Layouts Architecture & Facade.

Modular, decoupled architecture pairing visual topology masks with adaptive sub-pixel HTML5 templates.
Unified around the Tendoo Omni-Block Engine (OmniBlockLayout).
"""

from __future__ import annotations

import sys
from typing import Any

from tendoo.core.base import (
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
    LAYOUT_COMPATIBLE_STYLES,
    resolve_style_preset,
)
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text
from tendoo.engine.layout import OmniBlockLayout

_LEGACY_MODULES = {
    "top_dome",
    "bottom_platform",
    "center_hourglass",
    "split_column",
    "diagonal_slash",
    "l_frame",
    "freeform",
    "component_engine",
}


def __getattr__(name: str) -> Any:
    """Dynamically load legacy submodules from tendoo_legacy.layouts when requested."""
    if name in _LEGACY_MODULES:
        import importlib
        from pathlib import Path

        src_dir = Path(__file__).resolve().parent.parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        module = importlib.import_module(f"tendoo_legacy.layouts.{name}")
        sys.modules[f"tendoo.layouts.{name}"] = module
        return module

    if name == "get_component_css":
        try:
            from tendoo_legacy.layouts.component_engine import get_component_css
            return get_component_css
        except Exception:
            return lambda *args, **kwargs: ""
    if name == "render_category_body":
        try:
            from tendoo_legacy.layouts.component_engine import render_category_body
            return render_category_body
        except Exception:
            return lambda *args, **kwargs: ""

    raise AttributeError(f"module 'tendoo.layouts' has no attribute '{name}'")


__all__ = [
    "BaseLayout",
    "CALENDAR_ICON_SVG",
    "CHECK_ICON_SVG",
    "CLOCK_ICON_SVG",
    "ColorPalette",
    "GIFT_ICON_SVG",
    "GLOBE_ICON_SVG",
    "LOCATION_ICON_SVG",
    "OmniBlockLayout",
    "PHONE_ICON_SVG",
    "PosterContent",
    "STAR_ICON_SVG",
    "TAG_ICON_SVG",
    "get_layout",
    "list_layouts",
    "register_layout",
    "analyze_color_harmony",
    "hex_to_hue",
    "balance_vietnamese_headline",
    "normalize_text",
    "resolve_font",
    "recommend_font",
    "list_font_options",
    "FONT_CATALOG",
    "resolve_style_preset",
    "LAYOUT_COMPATIBLE_STYLES",
    "CATEGORY_FIELD_SLOTS",
    "DEFAULT_FIELD_ROLE",
]
