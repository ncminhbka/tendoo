"""
src/tendoo/layouts/text_engine.py

Cổng tương thích ngược cho Text Engine (Backward Compatibility Facade):
=============================================================================
- Re-export toàn bộ hàm xử lý text tiếng Việt/emoji từ `tendoo.core.typography`
  (nơi logic thật đã chuyển hẳn sang trong đợt tái cấu trúc 2026-09-13).

TẠI SAO CẦN FILE NÀY:
- `src/tendoo_legacy/` (8 layout module) vẫn `from tendoo.layouts.text_engine import
  balance_vietnamese_headline, normalize_text, resolve_headline_effect`. Vì
  `tendoo_legacy` nằm ngoài phạm vi tái cấu trúc (không được sửa), file facade mỏng này
  đảm bảo import chain của nó không vỡ mà không cần nhân đôi logic thật.
"""

from __future__ import annotations

from tendoo.core.typography import (
    EMOJI_PATTERN,
    VIETNAMESE_COMPOUND_WORDS,
    balance_vietnamese_headline,
    compute_font_ladder,
    is_compound_pair,
    normalize_text,
    resolve_headline_effect,
    strip_emojis,
)

__all__ = [
    "EMOJI_PATTERN",
    "VIETNAMESE_COMPOUND_WORDS",
    "balance_vietnamese_headline",
    "compute_font_ladder",
    "is_compound_pair",
    "normalize_text",
    "resolve_headline_effect",
    "strip_emojis",
]
