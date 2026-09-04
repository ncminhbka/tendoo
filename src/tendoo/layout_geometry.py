"""
==================================================================================================
TENDOO AI - DETERMINISTIC LAYOUT GEOMETRY (Maximal Empty Rectangle)
==================================================================================================
Module: src/tendoo/layout_geometry.py
Purpose: Pure computational-geometry helper for Cấp độ 2 (Object Detection + Maximal Empty
Rectangle) of the occlusion-avoidance discussion in AGENTS.md -- finds the largest axis-aligned
empty rectangle available for an HTML secondary-content card WITHIN a given search region,
avoiding a set of "forbidden" boxes (detected hero-title / product bounding boxes).

Deliberately NOT a VLM call: this is the "find where the empty space actually is" sub-problem,
which has an exact geometric solution -- asking a language/vision model to output pixel
coordinates directly reproduces the same "numerical coordinate blindness" failure mode already
identified for free-form VLM HTML generation. No model inference happens in this module at all;
it only consumes bounding boxes that some upstream detector already produced.

Zero GPU / zero external dependency (pure Python) -- fully unit-testable offline.
==================================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class EmptyRect:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_css_percent(self, canvas_w: float, canvas_h: float) -> dict:
        """Converts to CSS-ready percentages relative to the full canvas -- drop-in replacement
        for a template's fixed `top/left/right/bottom` percentages."""
        return {
            "left_pct": round(100.0 * self.x1 / canvas_w, 2),
            "top_pct": round(100.0 * self.y1 / canvas_h, 2),
            "right_pct": round(100.0 * (canvas_w - self.x2) / canvas_w, 2),
            "bottom_pct": round(100.0 * (canvas_h - self.y2) / canvas_h, 2),
            "width_pct": round(100.0 * self.width / canvas_w, 2),
            "height_pct": round(100.0 * self.height / canvas_h, 2),
        }


def _clip_box(box: BBox, region: BBox) -> BBox | None:
    """Clips `box` to `region`; returns None if there's no overlap at all."""
    x1 = max(box[0], region[0])
    y1 = max(box[1], region[1])
    x2 = min(box[2], region[2])
    y2 = min(box[3], region[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _overlaps(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def find_largest_empty_rect(
    search_region: BBox,
    forbidden_boxes: Sequence[BBox],
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> EmptyRect:
    """
    Finds the largest axis-aligned empty rectangle within `search_region` that does not
    intersect any box in `forbidden_boxes`.

    Algorithm: classic "candidate coordinates" approach for the largest-empty-rectangle problem.
    Forbidden boxes are first clipped to the search region (obstacles outside the region we're
    even willing to consider don't matter). All obstacle edge x/y coordinates plus the region's
    own boundary become candidate grid lines; every candidate sub-rectangle formed by pairs of
    those lines is tested against the (small, typically <10) obstacle set for overlap, and the
    largest valid one wins. With N obstacles this is O(N^4) candidate checks in the worst case --
    trivial for the N ~ 1-5 (hero + product + maybe a couple more) this is designed for; this is
    NOT the asymptotically optimal O(N log N) sweep-line algorithm, but is simple, easy to verify
    correct by inspection, and fast enough at this scale (sub-millisecond).

    Falls back to returning `search_region` itself (zero obstacles effectively) if no obstacle
    actually intersects it, and returns the best-effort largest rectangle found (which may be
    smaller than min_width/min_height) if no candidate meets the minimum -- callers should check
    `.width`/`.height` against their own minimums rather than assume success.
    """
    rx1, ry1, rx2, ry2 = search_region
    clipped = [c for c in (_clip_box(b, search_region) for b in forbidden_boxes) if c is not None]

    if not clipped:
        return EmptyRect(rx1, ry1, rx2, ry2)

    xs = sorted({rx1, rx2} | {b[0] for b in clipped} | {b[2] for b in clipped})
    ys = sorted({ry1, ry2} | {b[1] for b in clipped} | {b[3] for b in clipped})

    best = EmptyRect(rx1, ry1, rx1, ry1)  # zero-area sentinel
    for i, x1 in enumerate(xs):
        for x2 in xs[i + 1:]:
            for j, y1 in enumerate(ys):
                for y2 in ys[j + 1:]:
                    candidate = (x1, y1, x2, y2)
                    if any(_overlaps(candidate, b) for b in clipped):
                        continue
                    area = (x2 - x1) * (y2 - y1)
                    if area > best.area:
                        best = EmptyRect(x1, y1, x2, y2)
    return best


def compute_safe_rect_for_category(
    canvas_w: float,
    canvas_h: float,
    orientation: str,
    forbidden_boxes: Sequence[BBox],
) -> EmptyRect:
    """
    Convenience wrapper: picks the same general "bottom stack" search region the hand-designed
    templates already assume (bottom ~78-82% of canvas, ~6-11% side margins -- matching the
    `bottom-stack`/`frosted-box`/`menu-stack` containers in typography_engine.py), then finds the
    largest empty sub-rectangle within it avoiding the detected hero/product boxes. This is meant
    to REPLACE that container's fixed top/bottom/left/right percentages with the actual detected
    safe area for this specific generated image, not to replace the template's overall design.
    """
    if orientation == "portrait":
        region = (canvas_w * 0.06, canvas_h * 0.18, canvas_w * 0.94, canvas_h * 0.96)
    else:
        region = (canvas_w * 0.055, canvas_h * 0.20, canvas_w * 0.945, canvas_h * 0.96)
    return find_largest_empty_rect(region, forbidden_boxes)
