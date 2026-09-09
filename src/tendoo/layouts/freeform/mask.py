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

from typing import List, Optional

import numpy as np

from tendoo.layouts.freeform.zones import ZONE_RECTS


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
    delta: float = 0.05,
    int_max: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Unions the soft rectangle masks of every requested named zone.

    zones=None or [] falls back to a single generic "top_center" corridor (roughly
    matching top_dome's footprint) so this layout still behaves sanely if ever called
    with no render plan at all -- but the real per-request zone list should always be
    passed in by the caller (computed from the LLM render plan before this is called).
    """
    if not zones:
        zones = ["top_center"]

    one_minus = np.ones((height, width), dtype=np.float32)
    for zone_name in zones:
        rect = ZONE_RECTS.get(zone_name)
        if rect is None:
            continue  # unknown zone name -- silently skip rather than crash on a bad LLM output
        m = _zone_rect_mask(height, width, rect, delta)
        one_minus *= (1.0 - m)

    mask = (1.0 - one_minus) * int_max
    return mask.astype(np.float32)
