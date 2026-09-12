"""
src/tendoo/layouts/base.py

Cổng tương thích ngược cho Base Definitions (Backward Compatibility Facade):
=============================================================================
- Re-export toàn bộ định nghĩa nền tảng từ `tendoo.core.base`:
  * BaseLayout, PosterContent, ColorPalette.
  * 10 Vector SVG Icons chuẩn (phone, location, globe, calendar, gift, tag, clock, check, star, arrow_right).

TẠI SAO CẦN FILE NÀY:
1. Duy trì tính tương thích ngược (Zero Breaking Changes):
   - Trước khi refactor cấu trúc kiến trúc sang `tendoo.core.base`, tất cả layout templates
     và unit test đều import từ `tendoo.layouts.base`.
   - File facade này đảm bảo 100% các script cũ, import statements và notebook thử nghiệm
     vẫn tiếp tục hoạt động trơn tru mà không cần sửa đổi mã nguồn người dùng.
"""

from __future__ import annotations

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
    BaseLayout,
    ColorPalette,
    PosterContent,
)

__all__ = [
    "ARROW_RIGHT_ICON_SVG",
    "BaseLayout",
    "CALENDAR_ICON_SVG",
    "CHECK_ICON_SVG",
    "CLOCK_ICON_SVG",
    "ColorPalette",
    "GIFT_ICON_SVG",
    "GLOBE_ICON_SVG",
    "LOCATION_ICON_SVG",
    "PHONE_ICON_SVG",
    "PosterContent",
    "STAR_ICON_SVG",
    "TAG_ICON_SVG",
]

