"""
Tendoo AI Core Package.
Contains domain logic, typography engines, and dataset synthesis utilities for FLUX.2 DiT typography fine-tuning.
"""

from tendoo.glyph_engine import (
    GlyphEngine,
    GlyphInfo,
    FONT_REGISTRY,
    FONT_TIERS,
    render_glyph,
    compute_optimal_glyph_box,
)
from tendoo.typography_engine import (
    BackgroundAnalysis,
    ZoneMetrics,
    PosterBackgroundAnalyzer,
    TypographyPromptBuilder,
    PosterTemplateEngine,
    PosterRenderer,
)

__all__ = [
    "GlyphEngine",
    "GlyphInfo",
    "FONT_REGISTRY",
    "FONT_TIERS",
    "render_glyph",
    "compute_optimal_glyph_box",
    "BackgroundAnalysis",
    "ZoneMetrics",
    "PosterBackgroundAnalyzer",
    "TypographyPromptBuilder",
    "PosterTemplateEngine",
    "PosterRenderer",
]
