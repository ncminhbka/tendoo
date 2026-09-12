"""
src/tendoo/core/colors.py

Tầng cốt lõi quản lý Màu sắc & Độ tương phản (Color & Contrast Core):
====================================================================
- Re-export các hàm hạt nhân từ color_engine để các module tầng cao sử dụng.
- Đảm bảo điểm kết nối chuẩn xác tới ColorPalette từ `tendoo.core.base`,
  loại bỏ hoàn toàn phụ thuộc vòng (circular dependency) qua tầng layouts.
"""

from __future__ import annotations

from tendoo.core.base import ColorPalette
from tendoo.layouts.color_engine import (
    analyze_color_harmony,
    compute_relative_luminance,
    extract_dominant_hsv,
    hex_to_hue,
    hsl_to_rgb,
)

__all__ = [
    "ColorPalette",
    "analyze_color_harmony",
    "compute_relative_luminance",
    "extract_dominant_hsv",
    "hex_to_hue",
    "hsl_to_rgb",
]
