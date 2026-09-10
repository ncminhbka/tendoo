"""
src/tendoo/core/colors.py

Tầng cốt lõi quản lý Màu sắc & Độ tương phản:
Color Harmony, WCAG Contrast Ratio, User Hue Override.
"""

from __future__ import annotations

from tendoo.layouts.color_engine import (
    analyze_color_harmony,
    compute_relative_luminance,
    extract_dominant_hsv,
    hex_to_hue,
    hsl_to_rgb,
)
from tendoo.layouts.base import ColorPalette

__all__ = [
    "ColorPalette",
    "analyze_color_harmony",
    "compute_relative_luminance",
    "extract_dominant_hsv",
    "hex_to_hue",
    "hsl_to_rgb",
]
