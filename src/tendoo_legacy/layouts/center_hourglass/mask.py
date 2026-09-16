"""
Center Hourglass Corridor Mask Generator.

Produces a 3-tier parametric light funnel:
1. Upper cone (y < y_top_max): Broad light beam holding headline & promotional badges.
2. Central waist (y_top_max <= y < y_floor_start): Tapers smoothly to frame the hero product with ambient rim-light.
3. Perspective floor (y >= y_floor_start): Expands outward to provide a clean perspective floor for footer typography.
"""

from __future__ import annotations

import numpy as np


def generate_hourglass_mask(
    height: int,
    width: int,
    y_waist: float = 0.56,
    w_half_top: float = 0.485,
    w_half_waist: float = 0.26,
    w_half_floor: float = 0.485,
    int_top: float = 1.0,
    int_waist: float = 0.72,
    int_floor: float = 0.96,
    delta: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """
    Constructs a C1-continuous, smooth Center Hourglass mask:
    - Upper zone (y <= y_waist): Smooth cosine narrowing from w_half_top to w_half_waist.
    - Floor zone (y > y_waist): Smooth cosine expansion from w_half_waist to w_half_floor.
    - Zero derivative at y=0, y=y_waist, and y=1.0 guarantees zero creases or seams.
    - Vectorized numpy execution for sub-millisecond generation.

    Returns:
        np.ndarray [H, W] float32 with values in [0.0, 1.0].
    """
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # [H, 1]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]   # [1, W]

    # C1 continuous width profile
    w_half = np.where(
        y <= y_waist,
        w_half_waist + (w_half_top - w_half_waist) * 0.5 * (1.0 + np.cos(np.pi * y / y_waist)),
        w_half_waist + (w_half_floor - w_half_waist) * 0.5 * (1.0 - np.cos(np.pi * (y - y_waist) / (1.0 - y_waist))),
    )

    # C1 continuous intensity falloff
    intensity = np.where(
        y <= y_waist,
        int_waist + (int_top - int_waist) * 0.5 * (1.0 + np.cos(np.pi * y / y_waist)),
        int_waist + (int_floor - int_waist) * 0.5 * (1.0 - np.cos(np.pi * (y - y_waist) / (1.0 - y_waist))),
    )

    dist = np.abs(x - 0.5)
    inner = w_half - delta
    outer = w_half
    ratio = np.clip((dist - inner) / delta, 0.0, 1.0)
    v = np.where(
        dist <= inner,
        1.0,
        np.where(dist >= outer, 0.0, 0.5 * (1.0 + np.cos(ratio * np.pi))),
    )

    return (v * intensity).astype(np.float32)

