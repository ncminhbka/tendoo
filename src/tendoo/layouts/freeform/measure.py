"""
src/tendoo/layouts/freeform/measure.py

Measures how much pixel space a render plan's text blocks will actually need, per
zone, using real PIL font metrics -- so the mask reserved for the diffusion
background can be SIZED TO FIT the actual content instead of a fixed guess.

Why this matters: the earlier "MER-vs-content-height" bug (documented in project
memory) affected the 6 fixed layouts precisely because their reserved corridor size
was a static/tuned guess disconnected from actual content length. Freeform needs this
fixed MORE, not less -- its content shape varies far more request-to-request (a
1-line badge vs. a 3-line hero + subtitle stack) than a fixed layout's headline+
slogan+offer ever does, so a fixed per-zone size is either too small (text overflows
past the mask into product territory) or too large (needlessly starves the product
of space) almost every time.

Deliberately measured with PIL, not a real browser: the actual line-wrapping/height
only depends on font + size + text + max-width, none of which need the diffusion
background to exist yet -- so this can run entirely offline, before the diffusion
step, without paying for a second Playwright/Chromium launch just to measure text
(the existing PosterRenderer already launches one per output image; doubling that for
measurement alone would compound a real perf issue already flagged elsewhere).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import ImageFont

from tendoo.layouts.font_engine import FONT_CATALOG, FONTS_DIR
from tendoo.layouts.freeform.zones import ZONE_NAMES, ZONE_SELF_ALIGN

# Must mirror layout.py's ROLE_SCALE font-size ratios exactly -- these two tables
# describe the SAME rendering decision (how big does each role render), just consumed
# by two different code paths (mask sizing vs. actual HTML/CSS).
ROLE_FONT_RATIO: Dict[str, float] = {
    "hero": 0.072,
    "subtitle": 0.030,
    "body": 0.021,
    "caption": 0.015,
    "badge": 0.023,
}

LINE_HEIGHT_MULT = 1.25
BLOCK_GAP_PX = 10
ZONE_MARGIN_PCT = 0.04   # matches template.html's .freeform-grid { inset: 4%; }
ZONE_MAX_W_PCT = 0.42    # matches layout.py's per-zone max-width cap
ZONE_MAX_H_PCT = 0.42
QR_SIZE_PX = 56          # matches template.html's .freeform-qr-img
QR_CARD_PADDING_PX = 24  # padding + hint label allowance


def _font_path_for(font_key: str) -> str:
    meta = FONT_CATALOG.get(font_key) or FONT_CATALOG["bevietnam"]
    return str(FONTS_DIR / meta["file"])


def _wrap_and_measure(text: str, font: "ImageFont.FreeTypeFont", max_width_px: float) -> Tuple[float, float]:
    """Greedy word-wrap `text` to fit within max_width_px (mirrors the browser's own
    word-break: break-word behavior closely enough for sizing purposes). Returns
    (widest_line_width_px, total_block_height_px)."""
    words = text.split()
    if not words:
        return 0.0, 0.0

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = font.getbbox(candidate)
        candidate_w = bbox[2] - bbox[0]
        if candidate_w <= max_width_px or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    max_line_w = max((font.getbbox(l)[2] - font.getbbox(l)[0]) for l in lines)
    ascent, descent = font.getmetrics()
    line_h = (ascent + descent) * LINE_HEIGHT_MULT
    return float(max_line_w), float(len(lines) * line_h)


def fit_font_size_px(
    text: str,
    font_path: str,
    base_font_size: int,
    max_width_px: float,
    max_height_px: float,
    min_font_size: int = 14,
    step: int = 2,
) -> Tuple[int, float, float]:
    """
    Shrinks `base_font_size` (px) just enough that `text`, greedily word-wrapped at
    `max_width_px`, fits within BOTH `max_width_px` and `max_height_px` -- shared by
    compute_zone_rects() (mask sizing) and FreeformLayout.render_html() (actual CSS),
    so both agree on how big a block can render before it must shrink.

    Checks width too, not just height: _wrap_and_measure() lets a single word that
    still doesn't fit `max_width_px` overflow it anyway (rather than splitting the
    word) -- for a narrow zone (e.g. a 9:16 canvas's 42%-width budget, ~240px) a long
    unbroken English/French word ("TRANSFORMATION") can still exceed the width budget
    even after the height constraint alone is satisfied, if the loop stopped too
    early. Confirmed on a real measurement (2026-09-10): height-only fitting converged
    at a font size whose widest wrapped line was still ~40% over the width budget.

    Without this, a block whose ROLE implies a large flat font (e.g. "hero" at 7.2% of
    canvas width) but whose actual TEXT is long (a full sentence, not a short title --
    real render-plan LLM output isn't guaranteed to keep hero/subtitle text short)
    overflows both the diffusion-reserved mask AND its own HTML box with no safety
    net -- confirmed on a real render-plan LLM run (2026-09-10): a full-sentence
    "hero"/"subtitle" pair flooded the entire canvas, each rendered at its flat
    ROLE_SCALE size regardless of length. This is the freeform-specific instance of
    the same "MER-vs-content-height" bug class the 6 fixed layouts already had fixed
    (l_frame/diagonal_slash) -- freeform predates that fix and was never exercised
    against long text until a real LLM produced some.

    Returns (chosen_font_size, wrapped_width_px, wrapped_height_px) so the caller gets
    the final measurement back without a second pass.
    """
    font_size = base_font_size
    while font_size > min_font_size:
        font = ImageFont.truetype(font_path, font_size)
        w, h = _wrap_and_measure(text, font, max_width_px)
        if h <= max_height_px and w <= max_width_px:
            return font_size, w, h
        font_size -= step
    font = ImageFont.truetype(font_path, min_font_size)
    w, h = _wrap_and_measure(text, font, max_width_px)
    return min_font_size, w, h


def compute_zone_rects(
    blocks: List[Dict[str, Any]],
    width: int,
    height: int,
    font_key: str = "bevietnam",
    qr_zone: Optional[str] = None,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Returns normalized (y0, x0, y1, x1) rects for every zone that actually has
    content assigned (text blocks and/or a QR code), sized to fit that zone's
    stacked content (+ a safety margin absorbing PIL-vs-browser metric drift) and
    anchored against the zone's named corner/edge/center -- mirroring
    layout.py/ZONE_SELF_ALIGN's own CSS anchoring, so the mask and the HTML that
    later overlays it always agree on where each zone actually sits.

    Zones with no content (and not the qr_zone) are simply absent from the result --
    callers should union whatever rects come back, same as the static ZONE_RECTS path.
    """
    font_path = _font_path_for(font_key)

    by_zone: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ZONE_NAMES}
    for b in blocks:
        zone = b.get("zone")
        if zone in by_zone:
            by_zone[zone].append(b)

    margin = ZONE_MARGIN_PCT
    max_w_px = width * ZONE_MAX_W_PCT
    max_h_px = height * ZONE_MAX_H_PCT

    rects: Dict[str, Tuple[float, float, float, float]] = {}
    for zone_name in ZONE_NAMES:
        zone_blocks = by_zone[zone_name]
        has_qr = qr_zone == zone_name
        if not zone_blocks and not has_qr:
            continue

        # Each block sharing this zone gets an equal slice of the zone's total height
        # budget (a simple, bounded heuristic -- not "hero deserves more room than
        # caption", just "no single block can silently claim the whole zone and push
        # its neighbors out"); fit_font_size_px() then shrinks that block's font until
        # its wrapped text actually fits its slice, rather than rendering at a flat
        # ROLE_FONT_RATIO size regardless of how long the text turns out to be.
        share_h_px = max_h_px / max(1, len(zone_blocks))
        wrap_w_px = min(max_w_px, width * 0.9)
        total_h_px = 0.0
        max_w_needed_px = 0.0
        for i, b in enumerate(zone_blocks):
            role = b.get("role") if b.get("role") in ROLE_FONT_RATIO else "body"
            base_font_size = max(14, int(width * ROLE_FONT_RATIO[role]))
            _, w_px, h_px = fit_font_size_px(
                b.get("text", ""), font_path, base_font_size, wrap_w_px, share_h_px,
            )
            total_h_px += h_px + (BLOCK_GAP_PX if i > 0 else 0.0)
            max_w_needed_px = max(max_w_needed_px, w_px)

        if has_qr:
            qr_footprint = QR_SIZE_PX + QR_CARD_PADDING_PX
            total_h_px += qr_footprint + (BLOCK_GAP_PX if zone_blocks else 0.0)
            max_w_needed_px = max(max_w_needed_px, qr_footprint)

        # Safety margin: PIL's font metrics can differ slightly from the browser's
        # actual text shaping/kerning -- pad generously rather than risk the real
        # HTML overflowing past what the mask reserved as product-free.
        zone_w_px = min(max_w_px, max_w_needed_px * 1.20 + 12)
        zone_h_px = min(max_h_px, total_h_px * 1.15 + 12)
        zone_w_norm = zone_w_px / width
        zone_h_norm = zone_h_px / height

        justify_self, align_self = ZONE_SELF_ALIGN[zone_name]
        if justify_self == "start":
            x0, x1 = margin, margin + zone_w_norm
        elif justify_self == "end":
            x0, x1 = (1.0 - margin) - zone_w_norm, 1.0 - margin
        else:
            x0, x1 = 0.5 - zone_w_norm / 2.0, 0.5 + zone_w_norm / 2.0

        if align_self == "start":
            y0, y1 = margin, margin + zone_h_norm
        elif align_self == "end":
            y0, y1 = (1.0 - margin) - zone_h_norm, 1.0 - margin
        else:
            y0, y1 = 0.5 - zone_h_norm / 2.0, 0.5 + zone_h_norm / 2.0

        rects[zone_name] = (max(0.0, y0), max(0.0, x0), min(1.0, y1), min(1.0, x1))

    return rects
