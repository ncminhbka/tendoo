"""
Split Column (Flat Silk Ribbon Panel) Corridor Mask Generator.

Produces a clean, straight vertical column mask creating a flat, unwrinkled silk banner:
- Occupies approximately 36-38% width along the left (or right) side.
- Features a straight vertical boundary along the y-axis to ensure the silk ribbon
  remains completely flat and uncurled (no 3D twisting, no wavy creases).
- Smooth cosine feathering seamlessly blends the edge into the background.
- Preserves 62-64% pristine space for full-length models, lookbooks, or hero subjects.
"""

from __future__ import annotations

import numpy as np


def generate_split_column_mask(
    height: int,
    width: int,
    side: str = "left",
    col_width: float = 0.38,
    delta: float = 0.040,
    wave_amp: float = 0.0,
    wave_freq: float = 1.0,
    int_max: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Constructs a straight vertical column mask for a flat, unwrinkled silk ribbon:
    - col_width: Base width ratio of the column (default 0.38).
    - delta: Cosine feathering width (default 0.040).
    - wave_amp: Wave amplitude (default 0.0 for perfectly straight, flat silk).
    - side: 'left' (default) or 'right'.
    - Fully vectorized numpy execution.

    Returns:
        np.ndarray [H, W] float32 with values in [0.0, 1.0].
    """
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # [H, 1]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]   # [1, W]

    # Straight vertical boundary (wave_amp=0.0 produces zero twisting/curling)
    if wave_amp > 0.0:
        wave = (
            wave_amp * np.sin(2.0 * np.pi * wave_freq * y + 0.3)
            + 0.010 * np.cos(4.0 * np.pi * wave_freq * y)
        )
        boundary = col_width + wave
    else:
        boundary = np.full_like(y, col_width)

    if side == "left":
        dist = x
    else:
        dist = 1.0 - x

    inner = boundary - delta
    outer = boundary
    ratio = np.clip((dist - inner) / delta, 0.0, 1.0)
    v = np.where(
        dist <= inner,
        1.0,
        np.where(dist >= outer, 0.0, 0.5 * (1.0 + np.cos(ratio * np.pi))),
    )

    mask = (v * int_max).astype(np.float32)
    return mask
