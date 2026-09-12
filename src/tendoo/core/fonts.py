"""
src/tendoo/core/fonts.py

Tầng cốt lõi quản lý Typography Fonts (Typography Core Gateway):
================================================================
- Cung cấp cổng truy cập chuẩn cho Font Catalog 19 họ font thương mại tiếng Việt.
- Re-export các hàm hạt nhân từ `font_engine`: `resolve_font`, `recommend_font`, `list_font_options`.
"""

from __future__ import annotations

from tendoo.layouts.font_engine import (
    FONT_CATALOG,
    FONTS_DIR,
    list_font_options,
    recommend_font,
    resolve_font,
)

__all__ = [
    "FONT_CATALOG",
    "FONTS_DIR",
    "list_font_options",
    "recommend_font",
    "resolve_font",
]
