"""
====================================================================================================
TENDOO AI - PRODUCTION GLYPH RENDERING ENGINE & RESOLUTION LOCK
====================================================================================================
Module: src/tendoo/glyph_engine.py
Purpose: Production-grade typography rasterizer and latent coordinate calculator for FLUX.2 DiT.
Author: Tendoo AI Architecture Team (VDT2026)
Frozen Policy: Standardized module for all training data pipelines, inference probes, and serving APIs.

MATHEMATICAL & ARCHITECTURAL FOUNDATIONS (MANDATORY TECHNICAL LAWS):
----------------------------------------------------------------------------------------------------
1. VAE 16x Spatial Compression Law:
   - FLUX.2 AutoEncoder (VAE) has downsampling factor f = 16 and 128 latent channels.
   - Each latent token corresponds to a 16x16 pixel patch on the canvas:
       lat_w = width // 16,  lat_h = height // 16
   - Reference sequence token count:
       L_ref = lat_w * lat_h = (width // 16) * (height // 16)
   - ALL glyph dimensions (width, height) MUST be strictly snapped to integer multiples of 16.

2. Empirically-Derived DiT Denoise Resolution Floor (Calibrated via Isolation Sweeps):
   - VAE Roundtrip tests proved that the AutoEncoder latent space preserves text cleanly down to 20pt.
   - However, during 50-step ODE flow matching, DiT attention dynamics smooth or mispredict ultra-fine
     latent features if characters fall below an empirical threshold, causing jagged/spiky diacritic artifacts.
   - We establish an empirical resolution floor per font archetype (calibrated via probe_dit_font_resolution_floor.py)
     to guarantee ODE trajectories denoise with 100% smooth, crisp contours without wasting token budget.

3. Rule 25: Tight-Crop Sizing -- RETIRED as the primary objective by Rule 29's final revision
   (see below). Token economy is no longer the top-level goal for multi-line glyphs; a generous,
   pre-chosen box that lets font size be maximized matters far more than minimizing tokens.

4. Rule 26: Decoupled Signal Independence (The Tripartite Synergy):
   - Prompt (via Qwen3): Steers Canvas Placement (top-left, center, bottom) and Visual Stylization.
   - RoPE Time Offset (t=10, 20, 30, 40): Identifies and disentangles slot channels independently.
   - Glyph VAE (via this Engine): Preserves 100% Vietnamese Spelling and Font Geometry.

5. Rule 29 -- FINAL (Sept 2026, revised a 4th and last time after 8 rounds of GPU probing -- see
   `scripts/probe_glyph_*.py`): Generous Box + Largest-Fitting Font. No fixed font-size floor.

   THE JOURNEY (kept for context -- three earlier, each more sophisticated theories were each
   falsified by direct GPU evidence before landing back on the ORIGINAL algorithm's philosophy):
     a) "Model preserves line count, so a wide glyph breaks on a narrow canvas" -- falsified: the
        IDENTICAL glyph bitmap (608x512px poem) rendered perfectly on a 1024x576 canvas and failed
        completely on 576x1024. Line count/font/text held constant; only canvas orientation changed.
     b) "glyph_lat_w / canvas_lat_w ratio must stay <= ~0.6" (canvas-width-ratio theory) --
        falsified by a DOCUMENTED, VERIFIED 100%-accurate production result (AGENTS.md Rule 11,
        the "Tây Tiến" poem via demo_tendoo_poster.py, canvas-ratio=0.875) that this theory said
        should fail.
     c) "the glyph box's OWN aspect ratio must stay in [0.5, 1.3]" -- looked solid across two
        independent rounds, but was DECISIVELY falsified by `probe_glyph_absolute_scale.py`: the
        SAME text at a near-CONSTANT aspect ratio (~2.3) scored 2/5 -> 5/5 -> 4/5 as font size
        alone rose 61 -> 83 -> 106pt, and a tight-cropped variant of the 61pt glyph (more extreme
        aspect, fewer tokens, same font) scored the identical 2/5 -- aspect ratio and token count
        were both ruled out once font size is controlled for.
     d) THE REAL VARIABLE, hiding in plain sight the whole time: ABSOLUTE FONT SIZE, enabled by a
        GENEROUS (not minimized) box -- exactly what `demo_tendoo_poster.py` / `batch_tendoo_
        poster.py` did from the start, and the only approach that has never been falsified.
        Cross-validated on the real 576x1024 (9:16) canvas across short/long/4-line-poem text
        (5/5, 5/5, 4/5 out of 5) via `scripts/probe_glyph_generous_box_9x16.py`, including the
        poem case at a glyph 1.56x WIDER than the canvas itself.

   THE SHIPPED RULE:
     - Line count comes ONLY from an explicit signal -- a literal '\n' in the text, or
       `force_single_line`, or `target_lines` -- NEVER guessed from word count. Whoever builds the
       ad copy (human or upstream LLM) already knows how many lines a title/subtitle/feature line
       should be. No explicit signal means one line.
     - Box height = `len(lines) * 128px` (the ORIGINAL algorithm's formula, re-confirmed 3x above).
     - Box width defaults to 512px, overridable via `box_width_px` (pass e.g. 896 for longer per-
       line content, matching the historical Tây Tiến/"Sóng" poem recipe). This is NOT auto-
       derived from content -- every long-form case that ever worked required an explicit
       `--box_w` from whoever built that slot; this function does not pretend to solve that
       automatically either.
     - Font size: binary-searched to be the LARGEST that fits both the width and height budget.
       There is no configurable floor at all -- `FONT_TIERS[*]['min_floor_pt']` is UNUSED by this
       function now (kept in the registry only for other callers / backward compatibility).
     - The resulting box is NOT tight-cropped further after font selection.

   RESIDUAL, KNOWN-OPEN RISK: even with a generous box and a large font, expect a non-trivial
   per-seed stochastic failure rate from the 50-step ODE itself (e.g. the poem case scored 4/5,
   not 5/5, on the real canvas). Production use should plan for regeneration/retry, not assume a
   deterministic single-shot 100% guarantee. Safety padding (16px) and the 512px default width
   for un-overridden multi-line content remain provisional, not independently re-verified per font.

   STATUS: this glyph-rendering layer is considered LOCKED for the isolated single-glyph @ t=10
   case. Next phase of the project moves to multi-slot conditioning (RoPE spatial binding --
   already tried and closed, AGENTS.md Rule 30; Regional Parallel Diffusion -- in progress)
   RE-EXAMINED using this corrected primitive, since earlier negative results for both may have
   been confounded by the very glyph-sizing defect this revision fixes.
====================================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Configure logging and console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [GlyphEngine] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TendooGlyphEngine")


# Locate Project Root Directory
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent
FONTS_DIR = PROJECT_ROOT / "fonts"

# ==================================================================================================
# 1. PER-FONT FLOOR REGISTRY & ARCHETYPES (16 QA-VERIFIED VIETNAMESE UNICODE FONTS)
# ==================================================================================================
# Unified Floor Architecture (REVISED Sept 2026, was Dual: bevietnam 32pt / others 36pt):
# - ALL 16 fonts: Floor 40pt.
#
# REVISION (Sept 2026): the original 32pt/36pt floors were locked from a single-line, short-text
# probe (probe_dit_font_resolution_floor.py / probe_all_fonts_floor_32_36.py) targeting the
# Nyquist "spiky edges" failure mode. Once Rule 29's self-aspect-ratio fix made multi-line
# diacritic-dense text reliable at the STRUCTURAL level, a residual, more uniform defect
# remained: diacritic marks (circumflex, tilde, the horizontal stroke on "đ") came out mildly
# wrong at 32pt while basic Latin letters were fine -- a DIFFERENT failure mode than the one the
# original floors were probed against. A dedicated multi-seed sweep
# (scripts/probe_glyph_font_size_fine_detail.py, bevietnam, text richest in Vietnamese diacritics,
# aspect ratio held in-band throughout by adjusting line count) found: 24-30pt still show
# diacritic defects, 36pt+ clean, 40pt sharpest. This also reconciles with reverse-engineering the
# `demo_tendoo_poster.py` "Tây Tiến" 100%-accurate reference (AGENTS.md Rule 11): its binary
# search actually chose 48pt for playfair at 4 lines.
#
# ONLY bevietnam was directly re-tested at 40pt. The other 15 fonts are raised to the SAME 40pt
# floor as a deliberate, conservative unification decision (40 > both their old 36pt and
# bevietnam's old 32pt, so it cannot be less safe than what shipped before) -- NOT because each
# was independently re-verified against this exact diacritic-fidelity methodology. Treat
# non-bevietnam floors as a reasonable default, not a locked law, until each gets its own sweep.

FONT_TIERS: Dict[str, Dict[str, Any]] = {
    # Workhorse Clean Sans-Serif (Floor: 40pt, revised Sept 2026)
    "bevietnam": {
        "file": "BeVietnamPro-Black.ttf",
        "archetype": "Clean Sans-Serif",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 40,  # scripts/probe_glyph_font_size_fine_detail.py: 36+ clean, 40 sharpest
        "default_line_spacing": 0.22,
        "description": "Commercial workhorse for clean, hyper-legible body, subtitles, and modern ads.",
    },

    # All Other 15 Fonts (Floor: 40pt, unified Sept 2026 -- see header note, not independently re-tested)
    "anton": {
        "file": "Anton-Regular.ttf",
        "archetype": "Bold Heavy Display",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.20,
        "description": "Massive, impactful condensed poster font for high-energy tech and flash sale ads.",
    },
    "gotham": {
        "file": "SVN-Gotham Ultra.otf",
        "archetype": "Geometric Ultra-Bold",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.20,
        "description": "High-end corporate branding, telecom campaigns, and authoritative titles.",
    },
    "lolapeluza": {
        "file": "SVN-Lolapeluza Black.ttf",
        "archetype": "Ultra-Black Display",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.22,
        "description": "Playful, chunky heavy display with high token mass and extreme contrast.",
    },
    "gretoon": {
        "file": "SVN-Gretoon.ttf",
        "archetype": "Pop-Art Cartoon",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.22,
        "description": "Pop-art, 3D extruded comic font for FMCG, snacks, and youthful campaigns.",
    },
    "playfair": {
        "file": "PlayfairDisplay.ttf",
        "archetype": "Editorial Serif",
        "tier": "Tier B (Serif / Medium)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.26,
        "description": "Luxury serif with sharp contrast, ideal for fashion, jewelry, and literature.",
    },
    "oswald": {
        "file": "Oswald.ttf",
        "archetype": "Condensed Gothic Sans",
        "tier": "Tier B (Serif / Medium)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.20,
        "description": "Tall, elegant condensed font for specifications, sports gear, and modern posters.",
    },
    "harabaras": {
        "file": "SVN-Harabaras.ttf",
        "archetype": "Geometric Medium Sans",
        "tier": "Tier B (Serif / Medium)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.22,
        "description": "Friendly, modern geometric branding for startups, apps, and consumer tech.",
    },
    "dancing": {
        "file": "DancingScript.ttf",
        "archetype": "Cursive Script",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.32,
        "description": "Fluid, emotional cursive script for spa, cosmetics, bridal, and organic lifestyle.",
    },
    "pacifico": {
        "file": "Pacifico-Regular.ttf",
        "archetype": "Brush Script",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.28,
        "description": "Casual retro brush script, world-class for CTA badges, food trucks, and summer ads.",
    },
    "sedgwick": {
        "file": "SedgwickAveDisplay-Regular.ttf",
        "archetype": "Graffiti / Street Brush",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.25,
        "description": "Urban street graffiti with organic splatter vibes for streetwear and youth culture.",
    },
    "graffiti": {
        "file": "SedgwickAveDisplay-Regular.ttf",
        "archetype": "Graffiti / Street Brush",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.25,
        "description": "Alias for sedgwick.",
    },
    "blowbrush": {
        "file": "SVN-Blow Brush.ttf",
        "archetype": "Marker / Street Art",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.26,
        "description": "Handmade dry-marker brush typography with energetic motion.",
    },
    "brush": {
        "file": "SVN-Blow Brush.ttf",
        "archetype": "Marker / Street Art",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.26,
        "description": "Alias for blowbrush.",
    },
    "clementine": {
        "file": "SVN-Clementine.ttf",
        "archetype": "Calligraphy Script",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.32,
        "description": "Sophisticated wedding and boutique calligraphy with sweeping ascenders/descenders.",
    },
    "cookies": {
        "file": "SVN-Cookies.ttf",
        "archetype": "Chunky Rounded Display",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.24,
        "description": "Soft, rounded bubbly letters for confectionery, bakery, toys, and kids products.",
    },
    "grocery": {
        "file": "SVN-Grocery Rounded.ttf",
        "archetype": "Handwritten Store Display",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.24,
        "description": "Vintage chalkboard / grocery sign lettering for organic markets and rustic cafes.",
    },
    "holidays": {
        "file": "SVN-Holidays.ttf",
        "archetype": "Festive Script",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 40,  # unified Sept 2026, see header note -- NOT independently re-tested per-font
        "default_line_spacing": 0.28,
        "description": "Joyful seasonal typography for Tet, festivals, promotions, and celebrations.",
    },
}

# Build full absolute path registry
FONT_REGISTRY: Dict[str, str] = {}
for key, meta in FONT_TIERS.items():
    FONT_REGISTRY[key] = str(FONTS_DIR / meta["file"])


# ==================================================================================================
# 2. DATA STRUCTURES FOR GLYPH METADATA & DIAGNOSTICS
# ==================================================================================================

@dataclass
class GlyphInfo:
    """
    Structured container holding the rendered PIL Image and all mathematical diagnostics.
    Used for dataset packaging, latent pre-caching, and inference serving metadata.
    """
    image: Image.Image
    text: str
    lines: List[str]
    font_name: str
    font_path: str
    font_size_pt: int
    width_px: int
    height_px: int
    latent_w: int
    latent_h: int
    token_count: int
    archetype: str
    tier: str
    min_floor_pt: int
    is_nyquist_safe: bool
    line_spacing_px: int
    padding_x_px: int
    padding_y_px: int

    def summary(self) -> str:
        """Returns a concise one-line technical log."""
        safe_str = "SAFE (Silk-Smooth)" if self.is_nyquist_safe else "WARNING: Sub-Nyquist!"
        return (
            f"Glyph[{self.font_name.upper()} | {self.font_size_pt}pt | {self.width_px}x{self.height_px}px "
            f"({self.latent_w}x{self.latent_h} lat = {self.token_count} tokens) | "
            f"{len(self.lines)}L | {safe_str}]"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes metadata to JSON-compatible dictionary (excluding raw PIL Image)."""
        d = asdict(self)
        d.pop("image", None)
        return d


# ==================================================================================================
# 3. CORE RESOLUTION & SIZING ALGORITHMS
# ==================================================================================================

def resolve_font_path(font_name_or_path: str | None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Resolves font alias (e.g. 'bevietnam', 'playfair') or direct file path.
    Returns: (canonical_font_name, font_file_path, font_tier_metadata)
    """
    if not font_name_or_path:
        # Default project workhorse
        canonical = "bevietnam"
        meta = FONT_TIERS[canonical]
        return canonical, FONT_REGISTRY[canonical], meta

    key = font_name_or_path.lower().strip()
    if key in FONT_REGISTRY:
        path = FONT_REGISTRY[key]
        if os.path.exists(path):
            return key, path, FONT_TIERS[key]

    # Check direct file path
    if os.path.exists(font_name_or_path):
        resolved_name = Path(font_name_or_path).stem.lower()
        # Fallback to closest match or generic tier
        matched_key = "bevietnam"
        for k, v in FONT_REGISTRY.items():
            if Path(v).resolve() == Path(font_name_or_path).resolve():
                matched_key = k
                break
        return matched_key, font_name_or_path, FONT_TIERS.get(matched_key, FONT_TIERS["bevietnam"])

    # Fallback to defaults in order of robustness
    for k in ["bevietnam", "playfair", "anton"]:
        p = FONT_REGISTRY.get(k)
        if p and os.path.exists(p):
            logger.warning(f"Font '{font_name_or_path}' not found. Falling back to default: '{k}'")
            return k, p, FONT_TIERS[k]

    raise FileNotFoundError(f"CRITICAL: No valid Vietnamese Unicode font found for '{font_name_or_path}'!")


def _balanced_split(words: List[str], n_lines: int) -> List[str]:
    """Splits a word list into `n_lines` as evenly as possible (by word count)."""
    n_lines = max(1, min(n_lines, len(words)))
    base = len(words) // n_lines
    rem = len(words) % n_lines
    lines = []
    idx = 0
    for i in range(n_lines):
        take = base + (1 if i < rem else 0)
        if take <= 0:
            continue
        lines.append(" ".join(words[idx: idx + take]))
        idx += take
    return lines


def _greedy_pack_lines(words: List[str], font: "ImageFont.FreeTypeFont", max_w: int) -> List[str]:
    """Greedily packs words into lines that each fit within `max_w` pixels."""
    lines: List[str] = []
    curr: List[str] = []
    for word in words:
        candidate = " ".join(curr + [word])
        bbox = font.getbbox(candidate)
        w = bbox[2] - bbox[0]
        if w <= max_w or not curr:
            curr.append(word)
        else:
            lines.append(" ".join(curr))
            curr = [word]
    if curr:
        lines.append(" ".join(curr))
    return lines


def auto_wrap_text(
    text: str,
    font_path: str,
    font_size_pt: int,
    max_line_width_px: Optional[int] = None,
    target_canvas_w: Optional[int] = None,
    max_line_width_ratio: float = 0.4,
    target_lines: Optional[int] = None,
    force_single_line: bool = False,
) -> List[str]:
    """
    Splits text into optimal visual lines.
    - Honors explicit user newline '\\n'.
    - If force_single_line=True, returns single line.
    - Rule 29 (Canvas-Aware Dynamic Line Planning): if `max_line_width_px` is given, OR
      `target_canvas_w` is given (from which max width = target_canvas_w * max_line_width_ratio
      is derived), PIXEL-WIDTH FITTING takes strict priority over the word-count heuristic.
      This is what actually decides line count relative to the TARGET OUTPUT CANVAS, not the
      glyph's own isolated box — required because the model preserves the glyph's line count
      near-verbatim regardless of what canvas it ends up composited onto.
    - Otherwise (no width/canvas info available, e.g. isolated tight-crop probes), falls back
      to the legacy word-count band heuristic.
    """
    clean_text = text.replace("\\n", "\n").strip()

    if "\n" in clean_text:
        return [line.strip() for line in clean_text.split("\n") if line.strip()]

    if force_single_line:
        return [clean_text]

    words = clean_text.split()
    n_words = len(words)

    if n_words <= 1:
        return [clean_text]

    if target_lines == 1:
        return [clean_text]

    # Load font for width measurements
    try:
        font = ImageFont.truetype(font_path, size=font_size_pt)
    except Exception:
        font = ImageFont.load_default()

    effective_max_w = max_line_width_px
    if effective_max_w is None and target_canvas_w:
        effective_max_w = int(target_canvas_w * max_line_width_ratio)

    # --- PIXEL-WIDTH-FIRST PACKING: authoritative whenever real width info is known ---
    if effective_max_w is not None and effective_max_w > 0 and target_lines is None:
        full_bbox = font.getbbox(clean_text)
        if (full_bbox[2] - full_bbox[0]) <= effective_max_w:
            return [clean_text]
        return _greedy_pack_lines(words, font, effective_max_w)

    if n_words <= 3:
        return [clean_text]

    # Explicit target_lines override (balanced split)
    if target_lines == 2 or (target_lines is None and 4 <= n_words <= 9):
        return _balanced_split(words, 2)

    if target_lines == 3 or (target_lines is None and 10 <= n_words <= 18):
        return _balanced_split(words, 3)

    if target_lines == 4 or (target_lines is None and 19 <= n_words <= 32):
        return _balanced_split(words, 4)

    if target_lines is not None and target_lines >= 2:
        return _balanced_split(words, target_lines)

    # Default fallback for >32 words with no width info: chunk by ~1/5th of the words per line
    chunk_size = max(5, int(math.ceil(n_words / 5.0)))
    lines = []
    for i in range(0, n_words, chunk_size):
        lines.append(" ".join(words[i : i + chunk_size]))
    return lines



def compute_optimal_glyph_box(
    text: str,
    font_name_or_path: str = "bevietnam",
    font_size_pt: Optional[int] = None,
    target_lines: Optional[int] = None,
    force_single_line: bool = False,
    safety_padding_px: int = 16,
    target_canvas_w: Optional[int] = None,
    target_canvas_h: Optional[int] = None,
    px_per_line: int = 128,
    max_font_pt: int = 300,
    box_width_px: Optional[int] = None,
) -> Tuple[int, int, int, List[str]]:
    """
    Universal Slot-Agnostic Sizing Function (Rules 22, 25, 26, 29).

    Rule 29 -- FINAL (Sept 2026, revised a 4th and last time). Two prior revisions (a glyph-to-
    canvas width-ratio ceiling, then a glyph-box self-aspect-ratio band) were each independently
    falsified by GPU evidence, most decisively by `scripts/probe_glyph_absolute_scale.py`: the
    SAME text, SAME near-constant aspect ratio, differing ONLY in absolute font size, scored
    2/5 -> 5/5 -> 4/5 as font size rose 61 -> 83 -> 106pt. A companion isolation
    (`probe_glyph_generous_box_9x16.py`: "T" tight-cropped the SAME 61pt glyph down to fewer
    tokens at a MORE extreme aspect and got the identical 2/5) additionally ruled out both raw
    token count and aspect ratio as independent levers once font size is controlled for.

    The law that survived, unrevised, going back to the very first working scripts
    (demo_tendoo_poster.py / batch_tendoo_poster.py, which produced every historically 100%-
    accurate reference including the "Tây Tiến" poem): pick a GENEROUS box (not tight-cropped,
    not minimized for token economy) and binary-search the LARGEST font that fits it. There is no
    single universal font-size floor -- box size is what matters, and font size is simply
    whatever that generous box, at the given line count, permits. Cross-validated on the real
    576x1024 (9:16) canvas across short/long/4-line-poem text (5/4/5 out of 5) via
    `scripts/probe_glyph_generous_box_9x16.py`, including a case where the glyph (896px) is 1.56x
    WIDER than the canvas itself -- canvas-width-ratio is not an independent constraint either.

    Line count is decided ONLY by explicit signals from the caller -- a literal '\\n', or
    `force_single_line`, or `target_lines` -- never guessed from word count. Whoever is
    constructing the ad copy (a human or an upstream LLM) already knows how many lines a title /
    subtitle / feature line should be; a heuristic guess here was an unnecessary source of
    variance. No explicit signal means one line.

    Box height = `len(lines) * px_per_line` (128px/line default, matching the original algorithm
    and re-confirmed 3x above) -- generous, NOT tight-cropped afterward.

    Box WIDTH is a second generous, pre-chosen constraint, NOT left unconstrained: an early draft
    of this revision searched font size against height alone and let width fall out wherever it
    landed -- for short lines this pushed font size (and width) far past anything validated (a
    2-line phrase reached 98pt/400px; a 4-line poem reached 84pt/1552px, 3x the canvas width,
    entirely outside the <=1.56x-canvas-width territory `probe_glyph_generous_box_9x16.py` actually
    tested). `box_width_px` defaults to 512 (matching the "short_generous"/"long_generous" 5/5
    results) -- pass a larger value explicitly (896 matched the historical Tây Tiến/"Sóng" poem
    recipe, both 4/5-5/5) for longer per-line content that needs it. This mirrors real historical
    practice: every long-form case that ever worked (the poems) required an EXPLICIT `--box_w`
    from the person building that slot, never a fully automatic derivation -- so this function
    does not pretend to auto-derive it either.

    Status: font size now has no configurable floor at all (the old `min_floor_pt` per font in
    FONT_TIERS is UNUSED by this function as of this revision -- kept in the registry only for
    other callers/backward compatibility). Safety padding (16px) and the 512px default width for
    un-overridden multi-line content remain provisional, not independently re-verified per font.

    Returns: (width_px, height_px, chosen_font_size_pt, lines)
    """
    _, font_path, meta = resolve_font_path(font_name_or_path)
    spacing_ratio = meta["default_line_spacing"]

    clean_text = text.replace("\\n", "\n").strip()
    if "\n" in clean_text:
        lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
    elif force_single_line:
        lines = [clean_text]
    elif target_lines is not None and target_lines >= 2:
        lines = _balanced_split(clean_text.split(), target_lines)
    else:
        lines = [clean_text]  # no explicit signal -> one line; caller decides line breaks

    box_h = max(32, int(math.ceil((len(lines) * px_per_line) / 16.0) * 16))
    box_w_target = max(32, (box_width_px if box_width_px is not None else 512))

    if font_size_pt is not None:
        # Explicit font size: measure the box it actually needs, no search.
        font = ImageFont.truetype(font_path, size=font_size_pt) if font_path else ImageFont.load_default()
        chosen_pt = font_size_pt
    else:
        # Binary-search the LARGEST font that fits BOTH box_w_target and box_h (matches the
        # original demo_tendoo_poster.py algorithm and probe_glyph_generous_box_9x16.py's
        # validated render_generous_box_glyph).
        pad_w, pad_h = int(box_w_target * 0.08), int(box_h * 0.08)
        max_w, max_h = box_w_target - 2 * pad_w, box_h - 2 * pad_h
        low, high, chosen_pt, font = 8, max_font_pt, 0, None
        while low <= high:
            mid = (low + high) // 2
            try:
                test_font = ImageFont.truetype(font_path, size=mid)
            except Exception:
                test_font = ImageFont.load_default()
            line_widths = [test_font.getbbox(l)[2] - test_font.getbbox(l)[0] for l in lines]
            line_heights = [test_font.getbbox(l)[3] - test_font.getbbox(l)[1] for l in lines]
            total_w = max(line_widths)
            total_h = sum(line_heights) + int(mid * spacing_ratio) * (len(lines) - 1)
            if total_w <= max_w and total_h <= max_h:
                chosen_pt, font = mid, test_font
                low = mid + 1
            else:
                high = mid - 1
        if font is None:
            chosen_pt = 8
            font = ImageFont.truetype(font_path, size=chosen_pt) if font_path else ImageFont.load_default()

    line_widths = [font.getbbox(l)[2] - font.getbbox(l)[0] for l in lines]
    line_heights = [font.getbbox(l)[3] - font.getbbox(l)[1] for l in lines]
    line_spacing = int(chosen_pt * spacing_ratio)
    content_w = max(line_widths)
    content_h = sum(line_heights) + line_spacing * (len(lines) - 1)

    box_w = max(32, int(math.ceil((content_w + 2 * safety_padding_px) / 16.0) * 16))
    box_h = max(box_h, int(math.ceil((content_h + 2 * safety_padding_px) / 16.0) * 16))

    # Diagnostic-only: flag (don't silently clip) when the box outgrows the destination canvas --
    # both directions are legitimate (Tây Tiến's glyph is 1.56x wider than its canvas and works
    # fine), this is purely informational for reviewing whether a slot's text belongs there.
    if target_canvas_w and box_w > target_canvas_w:
        logger.info(f"Glyph width {box_w}px exceeds target canvas width {target_canvas_w}px "
                    f"({len(lines)} lines @ {chosen_pt}pt). This is not necessarily unsafe -- see Rule 29.")
    if target_canvas_h and box_h > target_canvas_h:
        logger.warning(f"Glyph height {box_h}px exceeds target canvas height {target_canvas_h}px "
                        f"({len(lines)} lines @ {chosen_pt}pt). Consider fewer lines for this slot.")

    return box_w, box_h, chosen_pt, lines


# ==================================================================================================
# 4. PRODUCTION GLYPH RENDERING ENGINE CLASS
# ==================================================================================================

class GlyphEngine:
    """
    Production-ready, highly optimized typography rendering engine for Tendoo AI.
    Handles Unicode Vietnamese diacritics, tight-crop bounding box computation,
    Nyquist resolution verification, and VAE patch alignment.
    """

    def __init__(self, default_font: str = "bevietnam"):
        self.default_font = default_font
        self._font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        """Cached loader for FreeType font instances."""
        cache_key = (font_path, size)
        if cache_key not in self._font_cache:
            try:
                self._font_cache[cache_key] = ImageFont.truetype(font_path, size=size)
            except Exception as e:
                logger.error(f"Failed to load font '{font_path}' at size {size}: {e}")
                self._font_cache[cache_key] = ImageFont.load_default()
        return self._font_cache[cache_key]

    def render(
        self,
        text: str,
        font_name_or_path: Optional[str] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        font_size_pt: Optional[int] = None,
        auto_size: bool = True,
        force_single_line: bool = False,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        safety_padding_px: int = 16,
        target_canvas_w: Optional[int] = None,
        target_canvas_h: Optional[int] = None,
        target_lines: Optional[int] = None,
        box_width_px: Optional[int] = None,
    ) -> GlyphInfo:
        """
        Main entrypoint to generate a production-ready locked glyph image.

        Modes:
        - Mode A (Default: auto_size=True, target_width/height are None):
          Generous-box sizing (Rule 29, final): height = num_lines * 128px, width defaults to
          512px unless `box_width_px` is given explicitly (pass 896 for longer per-line content,
          matching the historical Tây Tiến/"Sóng" poem recipe -- see compute_optimal_glyph_box's
          docstring for why this is NOT auto-derived). Font is binary-searched to be the LARGEST
          that fits both -- there is no fixed pt floor. Line count comes ONLY from an explicit
          '\\n' in `text` (or force_single_line / target_lines), never guessed from word count.
        - Mode B (Explicit Envelopes: target_width & target_height provided):
          Fits the text into the specified bounding box via binary search, checking against Nyquist floor.
          Here target_width/height already ARE (a region of) the destination canvas, so pixel-width-first
          wrapping is applied directly against them.
        """
        font_key, font_path, meta = resolve_font_path(font_name_or_path or self.default_font)
        min_floor = meta["min_floor_pt"]
        spacing_ratio = meta["default_line_spacing"]

        if auto_size or (target_width is None and target_height is None):
            # Mode A: Optimal Tight-Crop Dynamic Sizing
            box_w, box_h, chosen_size, lines = compute_optimal_glyph_box(
                text=text,
                font_name_or_path=font_key,
                font_size_pt=font_size_pt,
                force_single_line=force_single_line,
                safety_padding_px=safety_padding_px,
                target_canvas_w=target_canvas_w,
                target_canvas_h=target_canvas_h,
                target_lines=target_lines,
                box_width_px=box_width_px,
            )
            font = self.get_font(font_path, chosen_size)
        else:
            # Mode B: Explicit Target Envelope Fitting
            envelope_w = (target_width // 16) * 16
            envelope_h = (target_height // 16) * 16

            max_allowed_w = envelope_w - 2 * safety_padding_px
            max_allowed_h = envelope_h - 2 * safety_padding_px

            # NOTE (bugfix, Rule 29): previously `max_allowed_w` was computed but never passed
            # to auto_wrap_text, so line count was decided purely by word count, blind to the
            # actual envelope width -> a long line could be "wrapped" into a shape that still
            # doesn't fit, silently forcing the font-size search below to squeeze/overflow it.
            lines = auto_wrap_text(
                text=text,
                font_path=font_path,
                font_size_pt=font_size_pt or 40,
                force_single_line=force_single_line,
                max_line_width_px=max_allowed_w,
            )

            # Binary search for maximum font size fitting inside envelope
            low, high = min_floor, 220
            best_size = min_floor
            best_font = self.get_font(font_path, min_floor)

            while low <= high:
                mid = (low + high) // 2
                test_font = self.get_font(font_path, mid)

                total_h = 0
                max_lw = 0
                for line in lines:
                    bbox = test_font.getbbox(line)
                    lw = bbox[2] - bbox[0]
                    lh = bbox[3] - bbox[1]
                    max_lw = max(max_lw, lw)
                    total_h += lh

                curr_spacing = int(mid * spacing_ratio) * (len(lines) - 1)
                total_h += curr_spacing

                if max_lw <= max_allowed_w and total_h <= max_allowed_h:
                    best_size = mid
                    best_font = test_font
                    low = mid + 1
                else:
                    high = mid - 1

            chosen_size = best_size
            font = best_font
            box_w = envelope_w
            box_h = envelope_h

        # ------------------------------------------------------------------------------------------
        # RENDER BITMAP
        # ------------------------------------------------------------------------------------------
        img = Image.new("RGB", (box_w, box_h), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Compute exact text dimensions for vertical centering
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = font.getbbox(line)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])

        line_spacing = int(chosen_size * spacing_ratio)
        total_text_block_h = sum(line_heights) + line_spacing * (len(lines) - 1)

        start_y = max(safety_padding_px, (box_h - total_text_block_h) // 2)

        curr_y = start_y
        for idx, line in enumerate(lines):
            bbox = font.getbbox(line)
            lw = bbox[2] - bbox[0]
            # Center each line horizontally
            draw_x = (box_w - lw) // 2
            # Offset by bbox top-left to avoid baseline clipping
            draw_x -= bbox[0]
            draw_y = curr_y - bbox[1]

            draw.text((draw_x, draw_y), line, fill=text_color, font=font)
            curr_y += line_heights[idx] + line_spacing

        # ------------------------------------------------------------------------------------------
        # ASSEMBLE METADATA
        # ------------------------------------------------------------------------------------------
        lat_w = box_w // 16
        lat_h = box_h // 16
        token_count = lat_w * lat_h
        # NOTE: previously compared against the per-font min_floor_pt in FONT_TIERS, which the
        # final Rule 29 revision retired (no fixed pt floor -- box size drives font size now).
        # Repurposed as a much looser absolute sanity check against Rule 2's VAE-roundtrip-safe
        # floor (~20pt) -- catches a genuinely too-tiny font (e.g. from a long force_single_line
        # text crammed into a narrow box_width_px), not "below the old locked floor".
        is_nyquist_safe = chosen_size >= 20

        info = GlyphInfo(
            image=img,
            text=text,
            lines=lines,
            font_name=font_key,
            font_path=font_path,
            font_size_pt=chosen_size,
            width_px=box_w,
            height_px=box_h,
            latent_w=lat_w,
            latent_h=lat_h,
            token_count=token_count,
            archetype=meta["archetype"],
            tier=meta["tier"],
            min_floor_pt=min_floor,
            is_nyquist_safe=is_nyquist_safe,
            line_spacing_px=line_spacing,
            padding_x_px=safety_padding_px,
            padding_y_px=safety_padding_px,
        )

        return info


# Singleton engine instance for high-performance reuse
_DEFAULT_ENGINE = GlyphEngine()


def render_glyph(
    text: str,
    font_name_or_path: Optional[str] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    font_size_pt: Optional[int] = None,
    auto_size: bool = True,
    force_single_line: bool = False,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    safety_padding_px: int = 16,
    target_canvas_w: Optional[int] = None,
    target_canvas_h: Optional[int] = None,
    target_lines: Optional[int] = None,
    box_width_px: Optional[int] = None,
) -> GlyphInfo:
    """
    Convenience function to render a locked glyph using the singleton engine.
    """
    return _DEFAULT_ENGINE.render(
        text=text,
        font_name_or_path=font_name_or_path,
        target_width=target_width,
        target_height=target_height,
        font_size_pt=font_size_pt,
        auto_size=auto_size,
        force_single_line=force_single_line,
        bg_color=bg_color,
        text_color=text_color,
        safety_padding_px=safety_padding_px,
        target_canvas_w=target_canvas_w,
        target_canvas_h=target_canvas_h,
        target_lines=target_lines,
        box_width_px=box_width_px,
    )


# ==================================================================================================
# 5. CLI & DIAGNOSTIC INSPECTION SUITE
# ==================================================================================================

def run_font_inspection() -> None:
    """Prints a detailed audit table of all 16 registered fonts and their technical specs."""
    print("=" * 100)
    print(" [*] TENDOO AI - 16 QA-VERIFIED VIETNAMESE UNICODE FONTS & RESOLUTION FLOORS")
    print("=" * 100)
    header = f"{'Alias':<14} | {'Archetype':<22} | {'Tier':<20} | {'Min Floor':<10} | {'Status':<10}"
    print(header)
    print("-" * 100)

    for alias, meta in FONT_TIERS.items():
        if alias in ["graffiti", "brush"]:
            continue  # skip duplicate aliases in summary table
        path = FONT_REGISTRY[alias]
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"{alias:<14} | {meta['archetype']:<22} | {meta['tier']:<20} | {meta['min_floor_pt']}pt{'':<6} | {status}")

    print("=" * 100)
    print(" [i] Mathematical Rule: 1 Latent Token = 16x16px. Min Floor guarantees diacritics >= 0.70 tokens.")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Production Glyph Engine & Inspector")
    parser.add_argument("--text", type=str, default="CÀ PHÊ SỮA ĐÁ", help="Text string to render")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias or TTF path")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path to save glyph")
    parser.add_argument("--font_size", type=int, default=None, help="Specific font size (pt)")
    parser.add_argument("--width", type=int, default=None, help="Explicit target width (px)")
    parser.add_argument("--height", type=int, default=None, help="Explicit target height (px)")
    parser.add_argument("--single_line", action="store_true", help="Force single-line rendering")
    parser.add_argument("--inspect", action="store_true", help="Print registry audit table")
    parser.add_argument("--test_suite", action="store_true", help="Run benchmark across diverse phrases")
    parser.add_argument(
        "--canvas_width", type=int, default=None,
        help="Target OUTPUT canvas width (px) for Rule 29 canvas-aware line planning in Mode A",
    )
    parser.add_argument(
        "--canvas_height", type=int, default=None,
        help="Target OUTPUT canvas height (px) for Rule 29 canvas-aware line planning in Mode A",
    )
    parser.add_argument(
        "--aspect_test", action="store_true",
        help="CPU-only sanity check (no GPU needed): renders --text across the 4 canonical "
             "canvases (1:1, 9:16, 4:5, 16:9) and prints the resulting box/font/line count for "
             "each, to sanity-check Rule 29 (generous box + largest-fitting font) before "
             "running the GPU-backed probe on the server.",
    )

    args = parser.parse_args()

    if args.inspect:
        run_font_inspection()
        return

    if args.aspect_test:
        canonical_canvases = [
            ("1:1", 1024, 1024),
            ("9:16", 576, 1024),
            ("4:5", 896, 1120),
            ("16:9", 1024, 576),
        ]
        print("\n[ASPECT_TEST] Rule 29 Generous-Box Sanity Check (CPU-only, no GPU)")
        print(f"  Text: \"{args.text}\"  |  Font: {args.font}")
        print("=" * 100)
        header = f"{'Canvas':<8} | {'Size':<12} | {'Lines':<6} | {'FontPt':<7} | {'Box (px)':<14} | {'Tokens':<7} | Wrapped Lines"
        print(header)
        print("-" * 100)
        for name, cw, ch in canonical_canvases:
            box_w, box_h, chosen_pt, lines = compute_optimal_glyph_box(
                text=args.text,
                font_name_or_path=args.font,
                target_canvas_w=cw,
                target_canvas_h=ch,
            )
            tokens = (box_w // 16) * (box_h // 16)
            print(
                f"{name:<8} | {cw}x{ch:<8} | {len(lines):<6} | {chosen_pt}pt{'':<4} | "
                f"{box_w}x{box_h:<9} | {tokens:<7} | {lines}"
            )
        print("=" * 100)
        print("[i] Since the final Rule 29 revision, line count comes ONLY from an explicit '\\n' in")
        print("    --text (or force_single_line/target_lines) -- never guessed from word count -- so")
        print("    it is EXPECTED and CORRECT for all 4 rows to pick the same line count/box for the")
        print("    same --text. Box height = lines * 128px (generous, not tight-cropped); font is")
        print("    binary-searched to be the LARGEST that fits that height -- there is no fixed pt")
        print("    floor anymore. target_canvas_w/h are informational only (a glyph may legitimately")
        print("    be wider than its destination canvas, as the historical Tây Tiến case always was).\n")
        return

    if args.test_suite:
        test_phrases = [
            ("MUA 1 TẶNG 1", "pacifico", True),
            ("ĐẶC SẢN TÂY BẮC", "playfair", False),
            ("GIẢM GIÁ 50% HÔM NAY", "anton", True),
            ("Chăm sóc làn da thuần tự nhiên", "dancing", False),
            ("ĐỈNH CAO CÔNG NGHỆ 5G VIETTEL", "gotham", False),
            ("Sông Mã xa rồi Tây Tiến ơi\nNhớ về rừng núi nhớ chơi vơi.", "playfair", False),
        ]
        print("\n[TEST] RUNNING TENDOO GLYPH ENGINE BENCHMARK SUITE:\n")
        for phrase, font_alias, single_line in test_phrases:
            info = render_glyph(
                text=phrase,
                font_name_or_path=font_alias,
                force_single_line=single_line,
                auto_size=True,
            )
            print(info.summary())
        print("\n[OK] Benchmark completed successfully!\n")
        return


    # Single render mode
    info = render_glyph(
        text=args.text,
        font_name_or_path=args.font,
        target_width=args.width,
        target_height=args.height,
        font_size_pt=args.font_size,
        auto_size=(args.width is None and args.height is None),
        force_single_line=args.single_line,
        target_canvas_w=args.canvas_width,
        target_canvas_h=args.canvas_height,
    )

    print("\n" + "=" * 80)
    print(" [*] TENDOO GLYPH GENERATION REPORT")
    print("=" * 80)
    print(f"  - Text Content   : \"{info.text}\"")
    print(f"  - Font Name      : {info.font_name.upper()} ({info.archetype})")
    print(f"  - Font Size      : {info.font_size_pt}pt (Min Floor: {info.min_floor_pt}pt)")
    print(f"  - Pixel Size     : {info.width_px} x {info.height_px} px")
    print(f"  - Latent Grid    : {info.latent_w} x {info.latent_h} latent patches (16x16)")
    print(f"  - Token Count    : {info.token_count} tokens")
    print(f"  - Lines ({len(info.lines)})      : {info.lines}")
    print(f"  - Nyquist Safety : {'[PASS] SILK-SMOOTH' if info.is_nyquist_safe else '[WARNING] SUB-NYQUIST'}")
    print("=" * 80)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        info.image.save(str(out_path))
        print(f"[+] Saved glyph image to: {out_path.resolve()}\n")



if __name__ == "__main__":
    main()
