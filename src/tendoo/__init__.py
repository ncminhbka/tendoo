"""
================================================================================
TENDOO AI - CORE SDK PACKAGE
High-performance Vietnamese typography rendering, In-Context 4D RoPE encoding,
and multi-slot banner generation engine.
================================================================================
"""

from tendoo.glyph import (
    FONT_REGISTRY,
    create_glyph_image,
    encode_glyph_to_incontext_tokens,
    encode_product_to_incontext_tokens,
    resolve_font_path,
)

__all__ = [
    "FONT_REGISTRY",
    "resolve_font_path",
    "create_glyph_image",
    "encode_glyph_to_incontext_tokens",
    "encode_product_to_incontext_tokens",
]
