"""
src/tendoo/core/fonts.py

Tầng cốt lõi quản lý Typography Fonts:
Re-export và mở rộng Font Catalog, Google Fonts mapping, @font-face generator.
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
