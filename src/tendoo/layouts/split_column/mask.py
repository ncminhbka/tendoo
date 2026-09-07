"""
Split Column (Cascading Ribbon Sash) Corridor Mask Generator.

Produces a flowing vertical column mask mimicking an organic silk satin sash:
- Occupies approximately 36-40% width along the left (or right) side.
- Features a gentle undulating wave boundary along the vertical axis (y)
  to look like genuine fabric drapery rather than a rigid geometric box.
- Cosine feathering smoothly blends the silk edge into the background.
- Preserves 60-64% pristine space for full-length models, lookbooks, or hero subjects.
"""

from __future__ import annotations

import numpy as np


def generate_split_column_mask(
    height: int,
    width: int,
    side: str = "left",
    col_width: float = 0.38,
    delta: float = 0.045,
    wave_amp: float = 0.022,
    wave_freq: float = 1.6,
    int_max: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Constructs an organic cascading ribbon sash column mask:
    - col_width: Base width ratio of the column (~0.38).
    - wave_amp: Undulation amplitude of the silk fabric wave.
    - wave_freq: Vertical frequency of the fabric folds.
    - side: 'left' (default) or 'right'.
    - Fully vectorized numpy execution.

    Returns:
        np.ndarray [H, W] float32 with values in [0.0, 1.0].
    """
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # [H, 1]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]   # [1, W]

    # Organic fabric boundary with dual harmonics for realistic drapery flow
    wave = (
        wave_amp * np.sin(2.0 * np.pi * wave_freq * y + 0.3)
        + 0.010 * np.cos(4.0 * np.pi * wave_freq * y)
    )
    boundary = col_width + wave

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
