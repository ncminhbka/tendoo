"""
Tendoo AI Core Package.
Contains domain logic, typography engines, and dataset synthesis utilities for FLUX.2 DiT typography fine-tuning.
"""

from src.tendoo.glyph_engine import (
    GlyphEngine,
    GlyphInfo,
    FONT_REGISTRY,
    FONT_TIERS,
    render_glyph,
    compute_optimal_glyph_box,
)

__all__ = [
    "GlyphEngine",
    "GlyphInfo",
    "FONT_REGISTRY",
    "FONT_TIERS",
    "render_glyph",
    "compute_optimal_glyph_box",
]
