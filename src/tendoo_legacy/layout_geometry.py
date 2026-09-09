"""
==================================================================================================
TENDOO AI - DETERMINISTIC LAYOUT GEOMETRY (Maximal Empty Rectangle)
==================================================================================================
Module: src/tendoo_legacy/layout_geometry.py (moved out of src/tendoo/ on 2026-09-08)
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

    @property
    def aspect_ratio(self) -> float:
        """width / height. 0.0 for a degenerate (zero-height) rect instead of raising --
        callers doing a `wide-vs-tall` comparison across a list that may include a sentinel
        empty result shouldn't have to special-case a ZeroDivisionError."""
        return self.width / self.height if self.height > 0 else 0.0

    def quadrant(self, canvas_w: float, canvas_h: float) -> str:
        """Classifies this rect's CENTER into one of the 9 canonical composition positions
        ("top-left" .. "bottom-right", "middle-center" for the true center cell) relative to
        the full canvas. Pure metadata for a downstream layout-assignment step to match a
        content block's canonical position preference (e.g. "title prefers top-*", "CTA/QR
        prefer bottom-*/a corner") against a candidate region without re-deriving position
        from raw coordinates at every call site."""
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        col = "left" if cx < canvas_w / 3 else ("right" if cx > 2 * canvas_w / 3 else "center")
        row = "top" if cy < canvas_h / 3 else ("bottom" if cy > 2 * canvas_h / 3 else "middle")
        return f"{row}-{col}"

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


def _iter_candidate_rects(search_region: BBox, forbidden_boxes: Sequence[BBox]):
    """
    Yields every axis-aligned candidate empty rectangle within `search_region`, using the
    classic "candidate coordinates" construction for the largest-empty-rectangle problem:
    forbidden boxes are first clipped to the search region (obstacles outside the region we're
    even willing to consider don't matter), all obstacle edge x/y coordinates plus the region's
    own boundary become candidate grid lines, and every sub-rectangle formed by pairs of those
    lines that does NOT intersect any (clipped) obstacle is yielded. With N obstacles this is
    O(N^4) candidates in the worst case -- trivial for the N ~ 1-5 this is designed for.

    Shared by both `find_largest_empty_rect` (keeps only the single biggest) and
    `find_top_empty_rects` (keeps the best K, spatially distinct) so the two can never silently
    disagree on what counts as "empty" -- one candidate universe, two different selection
    policies on top of it.

    Yields nothing but the untouched `search_region` itself when no obstacle actually intersects
    it (the original single-rect implementation's fast path, preserved here).
    """
    rx1, ry1, rx2, ry2 = search_region
    clipped = [c for c in (_clip_box(b, search_region) for b in forbidden_boxes) if c is not None]

    if not clipped:
        yield EmptyRect(rx1, ry1, rx2, ry2)
        return

    xs = sorted({rx1, rx2} | {b[0] for b in clipped} | {b[2] for b in clipped})
    ys = sorted({ry1, ry2} | {b[1] for b in clipped} | {b[3] for b in clipped})

    for i, x1 in enumerate(xs):
        for x2 in xs[i + 1:]:
            for j, y1 in enumerate(ys):
                for y2 in ys[j + 1:]:
                    candidate = (x1, y1, x2, y2)
                    if any(_overlaps(candidate, b) for b in clipped):
                        continue
                    yield EmptyRect(x1, y1, x2, y2)


def find_largest_empty_rect(
    search_region: BBox,
    forbidden_boxes: Sequence[BBox],
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> EmptyRect:
    """
    Finds the largest axis-aligned empty rectangle within `search_region` that does not
    intersect any box in `forbidden_boxes`. See `_iter_candidate_rects` for the candidate
    construction this scans over (shared with `find_top_empty_rects`) -- this function's own
    job is purely "keep the max-area one", unchanged in behavior from before this module grew a
    multi-region sibling (regression-checked: same candidate universe, same argmax).

    NOT the asymptotically optimal O(N log N) sweep-line algorithm, but simple, easy to verify
    correct by inspection, and fast enough at this scale (sub-millisecond for N ~ 1-5 obstacles).

    Falls back to returning `search_region` itself (zero obstacles effectively) if no obstacle
    actually intersects it, and returns the best-effort largest rectangle found (which may be
    smaller than min_width/min_height) if no candidate meets the minimum -- callers should check
    `.width`/`.height` against their own minimums rather than assume success.
    """
    rx1, ry1, _, _ = search_region
    best = EmptyRect(rx1, ry1, rx1, ry1)  # zero-area sentinel
    for candidate in _iter_candidate_rects(search_region, forbidden_boxes):
        if candidate.area > best.area:
            best = candidate
    return best


def _coverage_ratio(candidate: EmptyRect, already_chosen: EmptyRect) -> float:
    """Fraction of `candidate`'s OWN area that overlaps `already_chosen`. Deliberately
    asymmetric (not classic IoU): a small candidate fully nested inside an already-picked large
    region must be suppressed even though its IoU against that large region is tiny (IoU is
    dominated by the big region's area) -- what matters for top-K diversity is "how much of
    THIS candidate is redundant with what we already picked", not "how similar in size are the
    two boxes"."""
    ix1, iy1 = max(candidate.x1, already_chosen.x1), max(candidate.y1, already_chosen.y1)
    ix2, iy2 = min(candidate.x2, already_chosen.x2), min(candidate.y2, already_chosen.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / candidate.area if candidate.area > 0 else 0.0


def find_top_empty_rects(
    search_region: BBox,
    forbidden_boxes: Sequence[BBox],
    k: int = 5,
    min_width: float = 0.0,
    min_height: float = 0.0,
    max_coverage_by_prior: float = 0.6,
) -> List[EmptyRect]:
    """
    Multi-region generalization of `find_largest_empty_rect`: returns up to `k` DISTINCT empty
    rectangles within `search_region` (largest first) instead of only the single biggest one --
    the geometric primitive the "multiple content blocks -> multiple different empty spots"
    layout direction needs (a real poster puts title, feature list, price badge, logo/QR into
    DIFFERENT empty regions of the frame, not all forced into one shared safe zone).

    Shares `_iter_candidate_rects` with `find_largest_empty_rect` -- same candidate universe,
    so the two can never disagree on what counts as empty. Selection is greedy-by-area with
    suppression: take the largest remaining candidate, then discard every candidate that is
    already mostly (`> max_coverage_by_prior` fraction of ITS OWN area, see `_coverage_ratio`)
    covered by a region already picked -- otherwise the naive top-K by area would just return K
    near-identical nested crops of the single biggest empty area, not K genuinely different
    spots on the canvas.

    `min_width`/`min_height` filter candidates BEFORE ranking (unlike `find_largest_empty_rect`,
    which ignores these and expects the caller to check the single result afterwards) -- here it
    matters more, since an unfiltered tiny sliver candidate could otherwise consume an early slot
    in the top-K list ahead of a smaller-but-still-useful region.

    Returns fewer than `k` results if the region doesn't contain that many sufficiently distinct
    candidates (a single obstacle-free region yields exactly 1 result, matching
    `find_largest_empty_rect`'s behavior in that same case); returns `[]` if nothing meets
    `min_width`/`min_height` at all. Never raises.
    """
    candidates = [
        c for c in _iter_candidate_rects(search_region, forbidden_boxes)
        if c.area > 0 and c.width >= min_width and c.height >= min_height
    ]
    candidates.sort(key=lambda r: r.area, reverse=True)

    selected: List[EmptyRect] = []
    for candidate in candidates:
        if len(selected) >= k:
            break
        if any(_coverage_ratio(candidate, chosen) > max_coverage_by_prior for chosen in selected):
            continue
        selected.append(candidate)
    return selected


def compute_safe_rect_for_category(
    canvas_w: float,
    canvas_h: float,
    orientation: str,
    forbidden_boxes: Sequence[BBox],
    vertical_anchor: str = "bottom",
) -> EmptyRect:
    """
    Convenience wrapper: picks a search region matching where the caller's title/card is actually
    anchored, then finds the largest empty sub-rectangle within it avoiding the detected
    hero/product boxes. This is meant to REPLACE a container's fixed top/bottom/left/right
    percentages with the actual detected safe area for this specific generated image, not to
    replace the template's overall design.

    `vertical_anchor` picks which horizontal band of the canvas to search within:
      - "bottom" (default, unchanged from the original implementation): the
        `bottom-stack`/`frosted-box`/`menu-stack` convention used by feedback/recruitment/menu/
        grand_opening, and `product_ad` titles with `title_position="bottom-*"`. Region: bottom
        ~78-82% of canvas, ~6-11% side margins.
      - "top": `product_ad`'s default `title_position="top-center"` (and any other top-anchored
        title). Region tuned empirically against real generated images in
        scripts/probe_mer_representative_cases.py (Case A) and the prompt7 end-to-end test in
        output_hero_selector_test/ -- both previously had to bypass this wrapper with an ad-hoc
        region tuple because no "top" branch existed yet.
      - "middle": `product_ad`'s `title_position="middle-*"`. Region tuned against Case C
        (scripts/probe_mer_representative_cases.py) -- a middle band, not the full canvas height,
        since a tall centered subject can intersect a naive "middle third" band even when most of
        the canvas is otherwise clear.
      - "bottom-compact": feedback/before_after's small corner-card (redesigned 2026-09-05 -- see
        memory css-hero-title-overlay-direction.md). A real bug found on this exact redesign: using
        plain "bottom" (its ~80%-tall region, sized for the OLD full-width card stack) here made
        `top_pct` land wherever the search region's own arbitrary upper boundary was (often ~18-20%
        of canvas), NOT where an actual obstacle was -- the small card then anchored its `top` way
        up near that boundary instead of hugging the bottom the way a compact corner card should.
        This region is deliberately much shorter (bottom ~40-45% of canvas only), matching the
        compact card's real footprint, so `top_pct` only moves up when an obstacle genuinely
        intrudes into that smaller zone.
    Unrecognized values fall back to "bottom" -- existing callers that don't pass this argument at
    all are completely unaffected (this is the same region as before this parameter existed).
    """
    if orientation == "portrait":
        left, right = canvas_w * 0.06, canvas_w * 0.94
    else:
        left, right = canvas_w * 0.055, canvas_w * 0.945

    if vertical_anchor == "top":
        top, bottom = canvas_h * 0.01, canvas_h * 0.35
    elif vertical_anchor == "middle":
        top, bottom = canvas_h * 0.30, canvas_h * 0.65
    elif vertical_anchor == "bottom-compact":
        # Widened from 0.55/0.60 after a real render still overflowed: a 4-line quote + 3
        # possibly-2-line-wrapped feature tags + customer row + CTA genuinely needs more than a
        # 40-45%-tall region gives on a real, content-heavy brief (see memory
        # css-hero-title-overlay-direction.md) -- 0.45/0.50 gives a real, tested margin without
        # going back to plain "bottom"'s much taller region (which defeats the whole point of a
        # compact card, see that anchor's own docstring above).
        top = canvas_h * (0.45 if orientation == "portrait" else 0.50)
        bottom = canvas_h * 0.96
    else:
        top = canvas_h * (0.18 if orientation == "portrait" else 0.20)
        bottom = canvas_h * 0.96

    region = (left, top, right, bottom)
    return find_largest_empty_rect(region, forbidden_boxes)
