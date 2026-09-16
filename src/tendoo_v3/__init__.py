"""
src/tendoo_v3/__init__.py

Tendoo v3: Next-Generation Pluggable Creative Template & Continuous Mask Engine
"""

from __future__ import annotations

from tendoo_v3.catalog import TEMPLATE_CATALOG, build_llm_catalog_prompt
from tendoo_v3.mask_engine import generate_template_mask, save_mask_preview
from tendoo_v3.renderer import build_template_html, pil_to_base64_data_uri, render_plan_to_poster
from tendoo_v3.schema import StyleConfig, TendooCreativePlan

__all__ = [
    "TEMPLATE_CATALOG",
    "StyleConfig",
    "TendooCreativePlan",
    "build_llm_catalog_prompt",
    "build_template_html",
    "generate_template_mask",
    "pil_to_base64_data_uri",
    "render_plan_to_poster",
    "save_mask_preview",
]
