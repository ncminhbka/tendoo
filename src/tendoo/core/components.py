"""
src/tendoo/core/components.py

Tầng cốt lõi quản lý Component đồ họa & SVG Icons:
- SVG Icons vector chuẩn (phone, location, calendar, star, check, gift, tag, clock, globe...).
- Component HTML renderers cho 6 danh mục thương mại.
"""

from __future__ import annotations

from tendoo.layouts.base import (
    CALENDAR_ICON_SVG,
    CHECK_ICON_SVG,
    CLOCK_ICON_SVG,
    GIFT_ICON_SVG,
    PHONE_ICON_SVG,
    STAR_ICON_SVG,
)
from tendoo.layouts.freeform.layout import ICON_SVG_BY_NAME
from tendoo.layouts.component_engine import render_category_body

__all__ = [
    "CALENDAR_ICON_SVG",
    "CHECK_ICON_SVG",
    "CLOCK_ICON_SVG",
    "GIFT_ICON_SVG",
    "ICON_SVG_BY_NAME",
    "PHONE_ICON_SVG",
    "STAR_ICON_SVG",
    "render_category_body",
]
