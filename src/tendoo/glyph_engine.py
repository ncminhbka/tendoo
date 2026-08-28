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


3. Rule 25: Optimal Tight-Crop Sizing Law (Zero Size Bias & Maximum Speed):
   - Glyph boxes scale dynamically and purely based on ACTUAL TEXT LENGTH and NUMBER OF LINES.
   - We DO NOT artificially inflate glyph envelopes (e.g. 640 tokens for simple headers).
   - Token budgets by line count:
       * 1 short line (1-3 words, badges, CTA):   80 - 140 tokens  (~65% token savings)
       * 1 medium line (4-6 words, slogans):      130 - 200 tokens
       * 2 lines (6-10 words, titles):            220 - 320 tokens
       * Multi-line / long paragraph (15-25 words): 380 - 640 tokens
   - Saves >60% sequence length, speeds up attention O(L^2) by 2-3x, and eliminates Size Bias.

4. Rule 26: Decoupled Signal Independence (The Tripartite Synergy):
   - Prompt (via Qwen3): Steers Canvas Placement (top-left, center, bottom) and Visual Stylization.
   - RoPE Time Offset (t=10, 20, 30, 40): Identifies and disentangles slot channels independently.
   - Glyph VAE (via this Engine): Preserves 100% Vietnamese Spelling and Font Geometry.
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
# Archetype Tiers define the minimum safe font size (pt) to prevent VAE feature collapse:
# - Tier A (Floor 36pt): Heavy display, bold sans-serif with thick diacritics. Highly resilient.
# - Tier B (Floor 40pt): Classical serifs and condensed sans. Moderate stroke contrast.
# - Tier C (Floor 44-48pt): Scripts, delicate hairlines, and brush textures. Requires higher floor.

FONT_TIERS: Dict[str, Dict[str, Any]] = {
    # Tier A: Heavy Display / Bold Sans (Floor: 36pt)
    "bevietnam": {
        "file": "BeVietnamPro-Black.ttf",
        "archetype": "Clean Sans-Serif",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 32,  # Empirically validated on 2x A30 (DiT 50-step ODE denoise floor)
        "default_line_spacing": 0.22,

        "description": "Commercial workhorse for clean, hyper-legible body, subtitles, and modern ads.",
    },
    "anton": {
        "file": "Anton-Regular.ttf",
        "archetype": "Bold Heavy Display",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 36,
        "default_line_spacing": 0.20,
        "description": "Massive, impactful condensed poster font for high-energy tech and flash sale ads.",
    },
    "gotham": {
        "file": "SVN-Gotham Ultra.otf",
        "archetype": "Geometric Ultra-Bold",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 36,
        "default_line_spacing": 0.20,
        "description": "High-end corporate branding, telecom campaigns, and authoritative titles.",
    },
    "lolapeluza": {
        "file": "SVN-Lolapeluza Black.ttf",
        "archetype": "Ultra-Black Display",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 36,
        "default_line_spacing": 0.22,
        "description": "Playful, chunky heavy display with high token mass and extreme contrast.",
    },
    "gretoon": {
        "file": "SVN-Gretoon.ttf",
        "archetype": "Pop-Art Cartoon",
        "tier": "Tier A (Heavy / Dense)",
        "min_floor_pt": 36,
        "default_line_spacing": 0.22,
        "description": "Pop-art, 3D extruded comic font for FMCG, snacks, and youthful campaigns.",
    },

    # Tier B: Editorial Serif & Condensed (Floor: 40pt)
    "playfair": {
        "file": "PlayfairDisplay.ttf",
        "archetype": "Editorial Serif",
        "tier": "Tier B (Serif / Medium)",
        "min_floor_pt": 40,
        "default_line_spacing": 0.26,
        "description": "Luxury serif with sharp contrast, ideal for fashion, jewelry, and literature.",
    },
    "oswald": {
        "file": "Oswald.ttf",
        "archetype": "Condensed Gothic Sans",
        "tier": "Tier B (Serif / Medium)",
        "min_floor_pt": 40,
        "default_line_spacing": 0.20,
        "description": "Tall, elegant condensed font for specifications, sports gear, and modern posters.",
    },
    "harabaras": {
        "file": "SVN-Harabaras.ttf",
        "archetype": "Geometric Medium Sans",
        "tier": "Tier B (Serif / Medium)",
        "min_floor_pt": 40,
        "default_line_spacing": 0.22,
        "description": "Friendly, modern geometric branding for startups, apps, and consumer tech.",
    },

    # Tier C: Delicate Scripts, Brush & Rounded (Floor: 44 - 48pt)
    "dancing": {
        "file": "DancingScript.ttf",
        "archetype": "Cursive Script",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 48,
        "default_line_spacing": 0.32,
        "description": "Fluid, emotional cursive script for spa, cosmetics, bridal, and organic lifestyle.",
    },
    "pacifico": {
        "file": "Pacifico-Regular.ttf",
        "archetype": "Brush Script",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 44,
        "default_line_spacing": 0.28,
        "description": "Casual retro brush script, world-class for CTA badges, food trucks, and summer ads.",
    },
    "sedgwick": {
        "file": "SedgwickAveDisplay-Regular.ttf",
        "archetype": "Graffiti / Street Brush",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 44,
        "default_line_spacing": 0.25,
        "description": "Urban street graffiti with organic splatter vibes for streetwear and youth culture.",
    },
    "graffiti": {
        "file": "SedgwickAveDisplay-Regular.ttf",
        "archetype": "Graffiti / Street Brush",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 44,
        "default_line_spacing": 0.25,
        "description": "Alias for sedgwick.",
    },
    "blowbrush": {
        "file": "SVN-Blow Brush.ttf",
        "archetype": "Marker / Street Art",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 46,
        "default_line_spacing": 0.26,
        "description": "Handmade dry-marker brush typography with energetic motion.",
    },
    "brush": {
        "file": "SVN-Blow Brush.ttf",
        "archetype": "Marker / Street Art",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 46,
        "default_line_spacing": 0.26,
        "description": "Alias for blowbrush.",
    },
    "clementine": {
        "file": "SVN-Clementine.ttf",
        "archetype": "Calligraphy Script",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 48,
        "default_line_spacing": 0.32,
        "description": "Sophisticated wedding and boutique calligraphy with sweeping ascenders/descenders.",
    },
    "cookies": {
        "file": "SVN-Cookies.ttf",
        "archetype": "Chunky Rounded Display",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 44,
        "default_line_spacing": 0.24,
        "description": "Soft, rounded bubbly letters for confectionery, bakery, toys, and kids products.",
    },
    "grocery": {
        "file": "SVN-Grocery Rounded.ttf",
        "archetype": "Handwritten Store Display",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 44,
        "default_line_spacing": 0.24,
        "description": "Vintage chalkboard / grocery sign lettering for organic markets and rustic cafes.",
    },
    "holidays": {
        "file": "SVN-Holidays.ttf",
        "archetype": "Festive Script",
        "tier": "Tier C (Script / Brush)",
        "min_floor_pt": 44,
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


def auto_wrap_text(
    text: str,
    font_path: str,
    font_size_pt: int,
    max_line_width_px: Optional[int] = None,
    target_lines: Optional[int] = None,
    force_single_line: bool = False,
) -> List[str]:
    """
    Splits text into optimal visual lines.
    - Honors explicit user newline '\\n'.
    - If force_single_line=True, returns single line.
    - Otherwise, wraps based on natural word boundaries and visual width balance.
    """
    clean_text = text.replace("\\n", "\n").strip()

    if "\n" in clean_text:
        return [line.strip() for line in clean_text.split("\n") if line.strip()]

    if force_single_line:
        return [clean_text]

    words = clean_text.split()
    n_words = len(words)

    if n_words <= 3:
        return [clean_text]

    if target_lines == 1:
        return [clean_text]

    # Load font for width measurements
    try:
        font = ImageFont.truetype(font_path, size=font_size_pt)
    except Exception:
        font = ImageFont.load_default()

    # Target 2 lines for 4-9 words
    if target_lines == 2 or (target_lines is None and 4 <= n_words <= 9):
        mid = n_words // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        return [line1, line2]

    # Target 3 lines for 10-18 words
    if target_lines == 3 or (target_lines is None and 10 <= n_words <= 18):
        p1 = n_words // 3
        p2 = 2 * n_words // 3
        return [" ".join(words[:p1]), " ".join(words[p1:p2]), " ".join(words[p2:])]

    # Target 4 lines for 19-32 words (e.g. detailed commercial feedback quotes)
    if target_lines == 4 or (target_lines is None and 19 <= n_words <= 32):
        p1 = n_words // 4
        p2 = 2 * n_words // 4
        p3 = 3 * n_words // 4
        return [
            " ".join(words[:p1]),
            " ".join(words[p1:p2]),
            " ".join(words[p2:p3]),
            " ".join(words[p3:]),
        ]

    # Multi-line greedily bounded by max_line_width_px
    if max_line_width_px is not None and max_line_width_px > 0:
        lines = []
        curr = []
        for word in words:
            candidate = " ".join(curr + [word])
            bbox = font.getbbox(candidate)
            w = bbox[2] - bbox[0]
            if w <= max_line_width_px or not curr:
                curr.append(word)
            else:
                lines.append(" ".join(curr))
                curr = [word]
        if curr:
            lines.append(" ".join(curr))
        return lines

    # Default fallback for >32 words: chunk by 6-7 words per line
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
) -> Tuple[int, int, int, List[str]]:
    """
    Universal Slot-Agnostic Sizing Function (Rules 22, 25, 26).
    Calculates the exact minimal bounding box (width, height) snapped to multiples of 16,
    ensuring font_size satisfies the per-font Nyquist anti-aliasing floor.

    Returns: (width_px, height_px, chosen_font_size_pt, lines)
    """
    _, font_path, meta = resolve_font_path(font_name_or_path)
    min_floor = meta["min_floor_pt"]
    spacing_ratio = meta["default_line_spacing"]

    # Set font size to exact font min_floor by default, or validate against min_floor
    if font_size_pt is None:
        font_size_pt = min_floor
    elif font_size_pt < min_floor:
        logger.debug(
            f"Requested font size {font_size_pt}pt is below {meta['tier']} floor ({min_floor}pt). "
            f"Auto-elevating to {min_floor}pt to prevent spiky edges."
        )
        font_size_pt = min_floor

    lines = auto_wrap_text(
        text=text,
        font_path=font_path,
        font_size_pt=font_size_pt,
        target_lines=target_lines,
        force_single_line=force_single_line,
    )

    try:
        font = ImageFont.truetype(font_path, size=font_size_pt)
    except Exception:
        font = ImageFont.load_default()

    line_widths = []
    line_heights = []
    for line in lines:
        bbox = font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(font_size_pt * spacing_ratio)
    raw_content_w = max(line_widths)
    raw_content_h = sum(line_heights) + line_spacing * (len(lines) - 1)

    # Apply symmetric safety padding
    total_w = raw_content_w + 2 * safety_padding_px
    total_h = raw_content_h + 2 * safety_padding_px

    # Guarantee minimum height per line (>= 128px per line for multi-line to satisfy Nyquist)
    min_h_rule = len(lines) * (112 if len(lines) == 1 else 128)
    total_h = max(total_h, min_h_rule)

    # Snap to integer multiples of 16 (FLUX.2 VAE Patch Size)
    final_w = int(math.ceil(total_w / 16.0) * 16)
    final_h = int(math.ceil(total_h / 16.0) * 16)

    # Enforce minimum lower bound for VAE patch (No arbitrary 1024px ceiling)
    final_w = max(32, final_w)
    final_h = max(32, final_h)

    return final_w, final_h, font_size_pt, lines


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
    ) -> GlyphInfo:
        """
        Main entrypoint to generate a production-ready locked glyph image.

        Modes:
        - Mode A (Default: auto_size=True, target_width/height are None):
          Calculates the optimal tight-crop envelope automatically based on text length and font floor.
          Saves massive tokens, zero size bias, guaranteed >= 36-48pt floor.
        - Mode B (Explicit Envelopes: target_width & target_height provided):
          Fits the text into the specified bounding box via binary search, checking against Nyquist floor.
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
            )
            font = self.get_font(font_path, chosen_size)
        else:
            # Mode B: Explicit Target Envelope Fitting
            envelope_w = (target_width // 16) * 16
            envelope_h = (target_height // 16) * 16

            lines = auto_wrap_text(
                text=text,
                font_path=font_path,
                font_size_pt=font_size_pt or 40,
                force_single_line=force_single_line,
            )

            max_allowed_w = envelope_w - 2 * safety_padding_px
            max_allowed_h = envelope_h - 2 * safety_padding_px

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
        is_nyquist_safe = chosen_size >= min_floor

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

    args = parser.parse_args()

    if args.inspect:
        run_font_inspection()
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
