"""
Tendoo AI Core Package.
Live demo package: the layout engine (`tendoo.layouts`), its HTML->PNG renderer
(`tendoo.poster_renderer`), and the FastAPI demo server (`tendoo.demo_server`).

The older "100%-overlay" pipeline (PosterTemplateEngine et al.), the glyph-rasterization
dataset-synthesis engine, LoRA injection helpers, and the multi-region layout experiment
moved to `tendoo_legacy` on 2026-09-08 -- they aren't imported by anything in this package.
"""

from tendoo.poster_renderer import PosterRenderer
from tendoo.layouts import (
    BaseLayout,
    ColorPalette,
    PosterContent,
    get_layout,
    list_layouts,
    analyze_color_harmony,
    balance_vietnamese_headline,
    normalize_text,
)

__all__ = [
    "PosterRenderer",
    "BaseLayout",
    "ColorPalette",
    "PosterContent",
    "get_layout",
    "list_layouts",
    "analyze_color_harmony",
    "balance_vietnamese_headline",
    "normalize_text",
]
