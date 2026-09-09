"""
src/tendoo/layouts/freeform/zones.py

9-cell named anchor grid: the single source of truth shared by the freeform mask
(generate_freeform_mask) and the HTML renderer (FreeformLayout.render_html), so the
diffusion-reserved corridor and the CSS grid that overlays text on top of it always
agree on where each named zone actually is.

Chosen over free (x, y) coordinates deliberately: an LLM producing "put this near the
top-left" maps naturally and reliably onto one of these 9 names (matches how people
describe poster layouts in plain language -- see prompt_test.txt), while true floating
coordinates would need the mask geometry to be reconstructed from scratch per request
with no fixed vocabulary to validate against or reason about.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Normalized (y0, x0, y1, x1) rectangles. ~4% canvas margin, ~3.5% gutter between cells.
ZONE_RECTS: Dict[str, Tuple[float, float, float, float]] = {
    "top_left":      (0.04, 0.04,  0.32, 0.465),
    "top_center":    (0.04, 0.285, 0.32, 0.715),
    "top_right":     (0.04, 0.535, 0.32, 0.96),
    "middle_left":   (0.36, 0.04,  0.64, 0.465),
    "center":        (0.36, 0.285, 0.64, 0.715),
    "middle_right":  (0.36, 0.535, 0.64, 0.96),
    "bottom_left":   (0.68, 0.04,  0.96, 0.465),
    "bottom_center": (0.68, 0.285, 0.96, 0.715),
    "bottom_right":  (0.68, 0.535, 0.96, 0.96),
}

ZONE_NAMES = list(ZONE_RECTS.keys())

# CSS Grid line coordinates (1-indexed, "row-start / col-start / row-end / col-end")
# matching ZONE_RECTS's 3x3 structure exactly.
ZONE_GRID_AREA: Dict[str, str] = {
    "top_left": "1 / 1 / 2 / 2",
    "top_center": "1 / 2 / 2 / 3",
    "top_right": "1 / 3 / 2 / 4",
    "middle_left": "2 / 1 / 3 / 2",
    "center": "2 / 2 / 3 / 3",
    "middle_right": "2 / 3 / 3 / 4",
    "bottom_left": "3 / 1 / 4 / 2",
    "bottom_center": "3 / 2 / 4 / 3",
    "bottom_right": "3 / 3 / 4 / 4",
}

# Default text-align per zone column (left column -> left, center column -> center, ...).
ZONE_DEFAULT_ALIGN: Dict[str, str] = {
    "top_left": "left", "middle_left": "left", "bottom_left": "left",
    "top_center": "center", "center": "center", "bottom_center": "center",
    "top_right": "right", "middle_right": "right", "bottom_right": "right",
}


def is_valid_zone(name: str) -> bool:
    return name in ZONE_RECTS
