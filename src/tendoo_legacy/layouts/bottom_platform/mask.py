"""
Bottom Platform (Cinematic Base) Corridor Mask Generator.

Produces a smooth parametric ground platform mask:
- Upper zone (y < y_start): Completely transparent (0.0), preserving the full-canvas
  cinematic landscape, sky, architecture, or hero vehicle/product.
- Transition zone (y_start <= y < y_full): C1-continuous Cosine Ramp creating a natural
  horizon horizon/stage floor transition without harsh edges.
- Lower platform zone (y >= y_full): High-density clean platform surface (~1.0)
  providing solid contrast for theatrical headline and specs typography.
- Supports optional subtle convex stage curvature to frame subjects naturally.
"""

from __future__ import annotations

import numpy as np


def generate_bottom_platform_mask(
    height: int,
    width: int,
    y_start: float = 0.58,
    y_full: float = 0.72,
    curvature: float = 0.025,
    w_half: float = 0.49,
    delta_x: float = 0.04,
    int_max: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Constructs a C1-continuous, smooth Bottom Platform mask:
    - Upper 58% of canvas is strictly 0.0 (zero interference with hero visual).
    - Lower 35-40% has smooth Cosine Ramp climbing from 0.0 to int_max.
    - Curvature provides a subtle organic stage podium arch across the horizon line.
    - Lateral edges have soft feathering to eliminate rectangular window borders.
    - Fully vectorized numpy implementation for sub-millisecond execution.

    Returns:
        np.ndarray [H, W] float32 with values in [0.0, 1.0].
    """
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # [H, 1]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]   # [1, W]

    # Organic stage curve: center arches slightly upward if curvature > 0
    # Parabola peak at x = 0.5: 1 - 4*(x - 0.5)^2 ranges from 1.0 at center to 0.0 at edges
    arch_offset = curvature * (1.0 - 4.0 * (x - 0.5) ** 2)
    y_start_x = y_start - arch_offset
    y_full_x = y_full - arch_offset

    # C1 continuous vertical cosine ramp
    span = np.maximum(y_full_x - y_start_x, 1e-5)
    t_y = np.clip((y - y_start_x) / span, 0.0, 1.0)
    v_y = 0.5 * (1.0 - np.cos(np.pi * t_y))

    # Lateral horizontal boundary feathering
    dist_x = np.abs(x - 0.5)
    inner_x = w_half - delta_x
    ratio_x = np.clip((dist_x - inner_x) / delta_x, 0.0, 1.0)
    v_x = np.where(
        dist_x <= inner_x,
        1.0,
        np.where(dist_x >= w_half, 0.0, 0.5 * (1.0 + np.cos(ratio_x * np.pi))),
    )

    mask = (v_y * v_x * int_max).astype(np.float32)
    return mask
