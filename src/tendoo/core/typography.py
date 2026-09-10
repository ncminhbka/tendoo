"""
src/tendoo/core/typography.py

Tầng cốt lõi xử lý Typography & Đo đạc kích thước chữ:
- Cân bằng dòng tiếng Việt (balance_vietnamese_headline): tránh mồ côi từ, chia đều cụm từ 2 dòng.
- Chuẩn hóa văn bản (normalize_text, strip_emojis).
- Đo đạc font metrics bằng PIL chính xác 100% (fit_font_size_px):
  * Tự động áp dụng đo đạc trên `text.upper()` khi `is_uppercase=True` để chống cắt mép chữ in hoa (ví dụ: TRANSFORMATION).
  * Chống tràn cả chiều rộng và chiều cao (Width & Height constraints).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import ImageFont

from tendoo.layouts.text_engine import (
    balance_vietnamese_headline,
    normalize_text,
    resolve_headline_effect,
    strip_emojis,
)

LINE_HEIGHT_MULT = 1.25


def wrap_and_measure(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width_px: float,
) -> Tuple[float, float, List[str]]:
    """Greedy word-wrap `text` to fit within max_width_px.
    Returns (widest_line_width_px, total_block_height_px, lines).
    """
    words = text.split()
    if not words:
        return 0.0, 0.0, []

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = font.getbbox(candidate)
        candidate_w = bbox[2] - bbox[0]
        if candidate_w <= max_width_px or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    max_line_w = max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in lines)
    ascent, descent = font.getmetrics()
    line_h = (ascent + descent) * LINE_HEIGHT_MULT
    return float(max_line_w), float(len(lines) * line_h), lines


def fit_font_size_px(
    text: str,
    font_path: str,
    base_font_size: int,
    max_width_px: float,
    max_height_px: float,
    min_font_size: int = 14,
    step: int = 2,
    is_uppercase: bool = False,
) -> Tuple[int, float, float]:
    """
    Shrinks `base_font_size` (px) just enough that `text`, greedily word-wrapped at
    `max_width_px`, fits within BOTH `max_width_px` and `max_height_px`.

    CRITICAL FIX FOR CLIPPING (e.g. 'TRANSFORMATION'):
    If `is_uppercase` is True, `text` is measured in uppercase (`text.upper()`)
    so the advance widths of capital letters are accurately accounted for before
    rendering in CSS with `text-transform: uppercase`.
    """
    measure_text = text.upper() if is_uppercase else text

    font_size = base_font_size
    while font_size > min_font_size:
        font = ImageFont.truetype(font_path, font_size)
        w, h, _ = wrap_and_measure(measure_text, font, max_width_px)
        if h <= max_height_px and w <= max_width_px:
            return font_size, w, h
        font_size -= step

    font = ImageFont.truetype(font_path, min_font_size)
    w, h, _ = wrap_and_measure(measure_text, font, max_width_px)
    return min_font_size, w, h


__all__ = [
    "balance_vietnamese_headline",
    "fit_font_size_px",
    "normalize_text",
    "resolve_headline_effect",
    "strip_emojis",
    "wrap_and_measure",
]
