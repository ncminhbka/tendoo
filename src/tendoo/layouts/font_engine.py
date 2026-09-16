"""
src/tendoo/layouts/font_engine.py

Cổng tương thích ngược cho Font Engine (Backward Compatibility Facade):
=============================================================================
- Re-export toàn bộ nội dung font-catalog/resolve từ `tendoo.core.fonts`
  (nơi logic thật đã chuyển hẳn sang trong đợt tái cấu trúc 2026-09-13).

TẠI SAO CẦN FILE NÀY:
- `src/tendoo_legacy/` (8 layout module + không chỗ nào khác trong `tendoo/` hiện hành)
  vẫn `from tendoo.layouts.font_engine import resolve_font` / `FONT_CATALOG` / `FONTS_DIR`.
  Vì `tendoo_legacy` nằm ngoài phạm vi tái cấu trúc (không được sửa), file facade mỏng
  này đảm bảo import chain của nó không vỡ mà không cần nhân đôi logic thật.
"""

from __future__ import annotations

from tendoo.core.fonts import (
    FONT_ALIASES,
    FONT_CATALOG,
    FONTS_DIR,
    list_font_options,
    recommend_font,
    resolve_font,
)

__all__ = [
    "FONT_ALIASES",
    "FONT_CATALOG",
    "FONTS_DIR",
    "list_font_options",
    "recommend_font",
    "resolve_font",
]
