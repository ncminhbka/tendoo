"""
L-Frame Corner Anchor Corridor Mask Generator.
==============================================
Produces an L-shaped corridor framing the canvas along the top edge and left column:
- Top bar (y in [0.0, y_bar]): reserved for bold horizontal headline & subtitle.
- Left column (x in [0.0, x_col]): reserved for feature specs, bullet points, or vertical steps.
- Lower-right open quadrant (x > x_col, y > y_bar): strictly 0.0, preserving 45-55%
  pristine space for hero tech devices, home appliances, or workshop keynote figures.
- Exact Euclidean Signed Distance Field (SDF) with smooth cosine feathering.
- 100% vectorized NumPy implementation.
"""

from __future__ import annotations

import numpy as np


def generate_l_frame_mask(
    height: int,
    width: int,
    y_bar: float = 0.30,
    x_col: float = 0.38,
    delta: float = 0.10,
    side: str = "top_left",
    int_max: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Constructs an L-shaped corridor mask:
    - height, width: Canvas dimensions in pixels.
    - y_bar: Normalized height of the top horizontal bar (default 0.30).
    - x_col: Normalized width of the left vertical column (default 0.38).
    - delta: Cosine feathering transition width (default 0.10).
    - side: 'top_left' (default) or 'top_right'.
    - int_max: Maximum mask intensity (default 1.0).

    Returns:
        np.ndarray [H, W] float32 with values in [0.0, 1.0].
    """
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # [H, 1]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]   # [1, W]

    if side == "top_right":
        x = 1.0 - x

    # Relative coordinates from the inner corner (x_col, y_bar)
    u = x - x_col  # >0 outside left column, <=0 inside
    v = y - y_bar  # >0 outside top bar, <=0 inside

    # Exact Signed Distance Function to the L-shape boundary:
    # d <= 0 inside corridor (top bar or left column)
    # d > 0 outside corridor (lower-right subject area)
    d_outside = np.sqrt(np.maximum(u, 0.0) ** 2 + np.maximum(v, 0.0) ** 2)
    d_inside = np.where(
        u > 0.0,
        v,
        np.where(v > 0.0, u, np.maximum(u, v)),
    )
    d = np.where((u > 0.0) & (v > 0.0), d_outside, d_inside)

    half_delta = delta / 2.0
    d_inner = -half_delta
    d_outer = half_delta

    ratio = np.clip((d - d_inner) / delta, 0.0, 1.0)
    val = np.where(
        d <= d_inner,
        1.0,
        np.where(d >= d_outer, 0.0, 0.5 * (1.0 + np.cos(ratio * np.pi))),
    )

    mask = (val * int_max).astype(np.float32)
    return mask
