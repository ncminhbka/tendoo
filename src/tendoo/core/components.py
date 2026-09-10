"""
src/tendoo/core/components.py

Tầng cốt lõi quản lý Component đồ họa & SVG Icons:
- SVG Icons vector chuẩn (phone, location, calendar, star, check, gift, tag, clock, globe...).
- Component HTML renderers cho các danh mục thương mại.
"""

from __future__ import annotations

from typing import Any, Dict

from tendoo.core.base import (
    ARROW_RIGHT_ICON_SVG,
    CALENDAR_ICON_SVG,
    CHECK_ICON_SVG,
    CLOCK_ICON_SVG,
    GIFT_ICON_SVG,
    GLOBE_ICON_SVG,
    LOCATION_ICON_SVG,
    PHONE_ICON_SVG,
    STAR_ICON_SVG,
    TAG_ICON_SVG,
)

ICON_SVG_BY_NAME: Dict[str, str] = {
    "phone": PHONE_ICON_SVG,
    "location": LOCATION_ICON_SVG,
    "globe": GLOBE_ICON_SVG,
    "calendar": CALENDAR_ICON_SVG,
    "gift": GIFT_ICON_SVG,
    "tag": TAG_ICON_SVG,
    "clock": CLOCK_ICON_SVG,
    "check": CHECK_ICON_SVG,
    "star": STAR_ICON_SVG,
    "arrow_right": ARROW_RIGHT_ICON_SVG,
}


def get_component_css() -> str:
    """Fallback / legacy CSS for category components."""
    try:
        from tendoo_legacy.layouts.component_engine import get_component_css as _legacy_css
        return _legacy_css()
    except Exception:
        return ""


def render_category_body(category: str, content: Any, *args, **kwargs) -> str:
    """Fallback / legacy helper to render category components if needed."""
    try:
        from tendoo_legacy.layouts.component_engine import render_category_body as _legacy_render
        return _legacy_render(category, content, *args, **kwargs)
    except Exception:
        return ""


__all__ = [
    "ARROW_RIGHT_ICON_SVG",
    "CALENDAR_ICON_SVG",
    "CHECK_ICON_SVG",
    "CLOCK_ICON_SVG",
    "GIFT_ICON_SVG",
    "GLOBE_ICON_SVG",
    "ICON_SVG_BY_NAME",
    "LOCATION_ICON_SVG",
    "PHONE_ICON_SVG",
    "STAR_ICON_SVG",
    "TAG_ICON_SVG",
    "get_component_css",
    "render_category_body",
]
