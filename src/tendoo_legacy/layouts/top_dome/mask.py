"""
Top Arch Dome Corridor Mask Generator.

Produces a smooth parametric convex arch dome occupying the upper 35-40% of the canvas.
Leaves the lower 60-65% completely unconstrained (M = 0.0) for the hero product and scene.
"""

from __future__ import annotations

import numpy as np


def generate_top_dome_mask(
    height: int,
    width: int,
    y_max: float = 0.40,
    w_half_top: float = 0.49,
    w_half_bottom: float = 0.38,
    delta: float = 0.04,
) -> np.ndarray:
    """
    Constructs a smooth, seamless Arch Dome mask:
    - Upper zone (y < 0.22): Broad negative space corridor, intensity = 1.0.
    - Arch transition (0.22 <= y < y_max): Smooth quadratic/cosine taper.
    - Lower zone (y >= y_max): Strictly 0.0 -> zero constraint on product and scene physics.

    Args:
        height: Canvas height in pixels (e.g. 1024).
        width: Canvas width in pixels (e.g. 1024).
        y_max: Normalized cutoff where the corridor reaches zero (default 0.40).
        w_half_top: Maximum half-width at the top boundary (0.49 = nearly full width).
        w_half_bottom: Tapered half-width near the arch transition base.
        delta: Horizontal feather margin for smooth cosine transition.

    Returns:
        np.ndarray [H, W] float32 with values in [0.0, 1.0].
    """
    mask = np.zeros((height, width), dtype=np.float32)
    y_core = max(0.18, y_max - 0.14)

    for i in range(height):
        y = i / float(height - 1)
        if y >= y_max:
            continue

        if y < y_core:
            # Flat top arch ceiling with subtle curve
            t = y / y_core
            s = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep
            w_half = w_half_top * (1.0 - s) + (w_half_top - 0.02) * s
            intensity = 1.0
        else:
            # Gentle parabolic arch drop
            t = (y - y_core) / (y_max - y_core)
            s = t * t * (3.0 - 2.0 * t)
            w_half = (w_half_top - 0.02) * (1.0 - s) + w_half_bottom * s
            intensity = 0.5 * (1.0 + np.cos(t * np.pi))  # Smooth cosine decay to 0.0

        for j in range(width):
            x = j / float(width - 1)
            dist = abs(x - 0.5)

            if dist <= (w_half - delta):
                v = 1.0
            elif dist >= w_half:
                v = 0.0
            else:
                ratio = (dist - (w_half - delta)) / delta
                v = 0.5 * (1.0 + np.cos(ratio * np.pi))

            mask[i, j] = v * intensity

    return mask
