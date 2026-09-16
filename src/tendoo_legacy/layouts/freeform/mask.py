"""
src/tendoo/layouts/freeform/mask.py

Freeform Multi-Zone Corridor Mask Generator.
=============================================
Unlike the other 6 layouts (each a fixed 1-2 region topology tuned for one content
shape), this one reserves an ARBITRARY SUBSET of the named 3x3 anchor grid (see
zones.py) -- which zones get reserved comes from the per-request render plan (an LLM
decision, upstream of this function), not fixed layout geometry.

All requested zones share ONE flat corridor prompt/tone (see
FreeformLayout.get_corridor_prompt) -- generalizing to a different tone per zone would
mean forward-passing K+1 branches through denoise_regional_velocity_blended instead of
2, which isn't implemented (see the audit notes in src/tendoo/velocity_blending.py).
Keep it to 2 branches (scene + one shared corridor) unless a real need for
heterogeneous per-zone backgrounds emerges.

Reuses the same smooth-union technique already proven in
tendoo/layouts/l_frame/mask.py (M = 1 - product(1 - m_i) over all requested zones) so
multiple simultaneously-requested zones blend into one soft corridor with no hard
internal seams, even when the zones are adjacent or diagonal from each other.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from tendoo_legacy.layouts.freeform.zones import ZONE_RECTS


def _soft_interval(coord: np.ndarray, lo: float, hi: float, delta: float) -> np.ndarray:
    """1D soft indicator: ~1.0 inside [lo, hi], cosine-feathered to 0.0 over `delta`,
    centered on each edge (matches the feathering convention used by every other
    layout's mask.py -- delta/2 on each side of the nominal boundary)."""
    half = delta / 2.0
    rise = np.clip((coord - (lo - half)) / delta, 0.0, 1.0)
    rise = 0.5 * (1.0 - np.cos(rise * np.pi))
    fall = np.clip(((hi + half) - coord) / delta, 0.0, 1.0)
    fall = 0.5 * (1.0 - np.cos(fall * np.pi))
    return np.clip(np.minimum(rise, fall), 0.0, 1.0)


def _zone_rect_mask(height: int, width: int, rect, delta: float) -> np.ndarray:
    """Soft rectangle mask (cosine-feathered on all 4 edges) for one normalized
    (y0, x0, y1, x1) rect."""
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y0, x0, y1, x1 = rect
    m_y = _soft_interval(y, y0, y1, delta)
    m_x = _soft_interval(x, x0, x1, delta)
    return (m_y * m_x).astype(np.float32)


def generate_freeform_mask(
    height: int,
    width: int,
    zones: Optional[List[str]] = None,
    zone_rects: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    delta: float = 0.05,
    int_max: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Unions the soft rectangle masks of every requested named zone.

    Two ways to specify which zones to reserve, in priority order:
      1. `zone_rects`: a dict of zone_name -> (y0, x0, y1, x1), typically the output
         of measure.compute_zone_rects() -- rects SIZED TO FIT the actual content of
         each zone, not a fixed guess. This is the path that should be used whenever
         the real render plan (blocks) is known, i.e. essentially always in production.
      2. `zones`: a plain list of zone names, falling back to the STATIC ZONE_RECTS
         footprint for each (the original v1 behavior, kept for simple callers/tests
         that don't need content-aware sizing, e.g. a quick mask preview).

    zones=None/[] and zone_rects=None/{} both fall back to a single generic
    "top_center" corridor (roughly matching top_dome's footprint) so this layout still
    behaves sanely if ever called with no render plan at all.
    """
    rects_to_union: Dict[str, Tuple[float, float, float, float]]
    if zone_rects:
        rects_to_union = zone_rects
    elif zones:
        rects_to_union = {name: ZONE_RECTS[name] for name in zones if name in ZONE_RECTS}
    else:
        rects_to_union = {"top_center": ZONE_RECTS["top_center"]}

    one_minus = np.ones((height, width), dtype=np.float32)
    for rect in rects_to_union.values():
        m = _zone_rect_mask(height, width, rect, delta)
        one_minus *= (1.0 - m)

    mask = (1.0 - one_minus) * int_max
    return mask.astype(np.float32)
