"""
Tendoo AI Legacy / Research Package.

Holds the earlier "100%-overlay" poster pipeline (PosterTemplateEngine,
TypographyPromptBuilder, PosterBackgroundAnalyzer), the glyph-rasterization
dataset-synthesis engine used by the FLUX.2 DiT fine-tuning scripts, LoRA
injection helpers, and the multi-region layout experiment. None of this is
imported by the live demo (`tendoo.demo_server`, `tendoo.layouts`) -- it's
kept for the research scripts under `scripts/` that still depend on it.

Deliberately has NO eager submodule imports (unlike `tendoo/__init__.py`,
which does and was flagged for it): importing e.g. `tendoo_legacy.lora`
should not require `playwright`/`torch` to be installed just because some
other submodule in this package happens to need them.
"""
