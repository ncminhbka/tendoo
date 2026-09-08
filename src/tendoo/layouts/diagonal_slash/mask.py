"""
Diagonal Slash (Sport Dynamic) Corridor Mask Generator.
======================================================
Produces a dynamic diagonal slash mask angled at ~35°-45° across the canvas:
- Upper-left corridor area (35-45% area) allocated for impactful typography.
- Diagonal boundary with uniform perpendicular cosine feathering.
- Preserves 55-65% lower-right open space for dynamic hero subjects
  (sneakers in mid-air, athletic gym models, energy drinks, gaming setups).
- 100% vectorized NumPy implementation.
"""

from __future__ import annotations

import numpy as np


def generate_diagonal_slash_mask(
    height: int,
    width: int,
    x_top: float = 0.65,
    x_bottom: float = 0.20,
    delta: float = 0.10,
    side: str = "top_left",
    wave_amp: float = 0.0,
    wave_freq: float = 1.0,
    int_max: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Constructs an energetic diagonal slash corridor mask:
    - height, width: Canvas dimensions in pixels.
    - x_top: Normalized x-coordinate at top edge y=0 (default 0.65).
    - x_bottom: Normalized x-coordinate at bottom edge y=1 (default 0.20).
    - delta: Perpendicular cosine feathering transition width (default 0.10).
    - side: 'top_left' (default) or 'top_right'.
    - wave_amp: Optional subtle organic wave modulation (default 0.0 for straight crisp line).
    - int_max: Maximum mask intensity (default 1.0).

    Returns:
        np.ndarray [H, W] float32 with values in [0.0, 1.0].
    """
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # [H, 1]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]   # [1, W]

    if side == "top_right":
        x = 1.0 - x

    delta_x = x_bottom - x_top
    norm_factor = np.sqrt(1.0 + delta_x * delta_x)

    # Optional wave along boundary
    if wave_amp > 0.0:
        wave = wave_amp * np.sin(2.0 * np.pi * wave_freq * y)
    else:
        wave = 0.0

    # Signed perpendicular distance from the line (negative = inside corridor, positive = outside)
    d_perp = ((x - x_top - wave) - delta_x * y) / norm_factor

    half_delta = delta / 2.0
    d_inner = -half_delta
    d_outer = half_delta

    ratio = np.clip((d_perp - d_inner) / delta, 0.0, 1.0)
    v = np.where(
        d_perp <= d_inner,
        1.0,
        np.where(d_perp >= d_outer, 0.0, 0.5 * (1.0 + np.cos(ratio * np.pi))),
    )

    mask = (v * int_max).astype(np.float32)
    return mask
