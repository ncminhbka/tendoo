"""
src/tendoo_v2/solver.py

The deterministic layout solver -- this is where "where does each block go, and
what size" gets decided, NOT the LLM (see schema.py's docstring for the full
reasoning, and the approved plan robust-moseying-tower.md). Implements the
6-step algorithm agreed on this session:

  1. Group resolution    -- cluster Blocks sharing `group` into one Unit.
  2. Pinned/floating split -- a Unit with an explicit `zone` on any member is
     a hard constraint; everything else needs solving.
  3. Zone assignment for floating Units -- role-ranked preference list,
     highest-importance Unit first, never lands on "center"/"middle_center"
     (not present in GRID_ZONE_NAMES -- only reachable via an explicit pin,
     same rule as demo_server.py's _auto_assign_zones()).
  4. Real font-size fitting -- `role` sets a MAX CEILING, not a fixed value;
     the actual size is whatever `fit_font_size_px()` finds fits the Unit's
     REAL zone box. Multi-block Units split the zone's height proportionally
     to each member's role ceiling (hero gets more of the shared column than
     meta), not an equal share.
  5. Collision safety net -- NOT implemented here; reused from
     engine/templates/master.html's existing runtime JS in templates/master_v2.html.
  6. Negative-space check -- logged only this round (see plan's scope).

Reuses, does not reinvent: tendoo.engine.geometry's zone-box math (the exact
same functions demo_server.py's fallback path uses) and tendoo.core.typography's
font-fitting binary search -- both already solid, tested code; this module only
adds the NEW part (role/group-driven zone choice), which does not exist
anywhere in production yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from tendoo.core.fonts import FONT_CATALOG, FONTS_DIR
from tendoo.core.typography import fit_font_size_px
from tendoo.engine.geometry import GRID_ZONE_NAMES, get_zone_bounding_box

from tendoo_v2.schema import Block, ROLE_ORDER

# --- Step 3 table: role -> zone preference, in priority order. Deliberately
# excludes "center"/"middle_center" entirely (GRID_ZONE_NAMES already excludes
# it) -- a centered placement is ONLY ever reachable via an explicit `zone` on
# a Block (a real user request the prompt stated), never auto-chosen here,
# because it structurally conflicts with the Product Sanctuary (see the
# session's "coffee beans" discussion: text auto-placed in the visual center
# fights whatever hero subject the diffusion model wants there).
ROLE_ZONE_PREFERENCE: Dict[str, List[str]] = {
    "hero": ["top_center", "top_left", "top_right", "bottom_center"],
    "subhead": ["top_center", "middle_left", "middle_right", "bottom_center"],
    "body": ["middle_left", "middle_right", "bottom_center"],
    "cta": ["bottom_right", "bottom_center", "bottom_left"],
    "meta": ["bottom_left", "bottom_right", "bottom_center"],
}

# --- Step 4 table: role -> font-size-to-canvas-width CEILING ratio (a max, not
# a fixed target -- fit_font_size_px() below always finds the largest size that
# still fits the real zone box, up to this ceiling). Mirrors the current
# SIZE_SCALE ratios in engine/blocks.py post-2026-09-14 bump (xlarge=0.084,
# large=0.052, medium=0.038, small=0.024) -- hero/subhead map 1:1; body/cta
# share medium's ceiling (a CTA phrase is content-important, not tiny, but
# rendered via a button/ribbon shape, not a giant headline); meta maps to small.
ROLE_SIZE_CEILING: Dict[str, float] = {
    "hero": 0.084,
    "subhead": 0.052,
    "body": 0.038,
    "cta": 0.038,
    "meta": 0.024,
}

# hero/subhead read as commanding headline-style typography (uppercase, per the
# same convention SIZE_SCALE's xlarge/large used); body/cta/meta keep natural
# casing -- long sentences and action phrases read faster in mixed case.
ROLE_UPPERCASE: Dict[str, bool] = {
    "hero": True,
    "subhead": True,
    "body": False,
    "cta": False,
    "meta": False,
}

DEFAULT_FONT_KEY = "bevietnam"


@dataclass
class Unit:
    """One placement unit -- either a single Block (ungrouped) or several
    Blocks sharing a `group`, ordered by role so the highest-importance member
    renders first/on top within the stack."""

    blocks: List[Block]
    zone: Optional[str] = None
    pinned: bool = False

    @property
    def dominant_role(self) -> str:
        """The Unit's own importance rank for zone-preference purposes -- the
        highest-importance member's role (lowest ROLE_ORDER value)."""
        return min(self.blocks, key=lambda b: ROLE_ORDER.get(b.role, 99)).role


@dataclass
class PlacedBlock:
    block: Block
    font_size: int
    display_text: str  # may contain \n if fit_font_size_px() wrapped it
    is_uppercase: bool


@dataclass
class PlacedUnit:
    zone: str
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in canvas pixels
    placed_blocks: List[PlacedBlock]
    pinned: bool


# ==================================================================================
# Step 1: Group resolution
# ==================================================================================

def _group_into_units(blocks: List[Block]) -> List[Unit]:
    groups: Dict[str, List[Block]] = {}
    units: List[Unit] = []
    for b in blocks:
        if b.group:
            groups.setdefault(b.group, []).append(b)
        else:
            units.append(Unit(blocks=[b]))
    for members in groups.values():
        ordered = sorted(members, key=lambda b: ROLE_ORDER.get(b.role, 99))
        units.append(Unit(blocks=ordered))
    return units


# ==================================================================================
# Step 2: Pinned vs floating split
# ==================================================================================

def _split_pinned_floating(units: List[Unit]) -> Tuple[List[Unit], List[Unit]]:
    pinned: List[Unit] = []
    floating: List[Unit] = []
    for u in units:
        explicit_zone = next((b.zone for b in u.blocks if b.zone), None)
        if explicit_zone:
            u.zone = explicit_zone
            u.pinned = True
            pinned.append(u)
        else:
            floating.append(u)
    return pinned, floating


# ==================================================================================
# Step 3: Zone assignment for floating units
# ==================================================================================

def _assign_floating_zones(floating: List[Unit], occupied: Set[str]) -> None:
    """Mutates each Unit's `.zone` in place. Processes highest-importance Units
    first (hero before meta) so they get first pick of their preferred zones --
    mirrors demo_server.py's _auto_assign_zones() ordering rationale exactly."""
    ordered = sorted(floating, key=lambda u: ROLE_ORDER.get(u.dominant_role, 99))
    for u in ordered:
        candidates = ROLE_ZONE_PREFERENCE.get(u.dominant_role, GRID_ZONE_NAMES)
        chosen = next((z for z in candidates if z not in occupied), None)
        if chosen is None:
            chosen = next((z for z in GRID_ZONE_NAMES if z not in occupied), None)
        if chosen is None:
            # Graceful degradation past 8 zones (more units than zones): reuse
            # the first candidate rather than crash.
            chosen = candidates[0] if candidates else GRID_ZONE_NAMES[0]
        u.zone = chosen
        occupied.add(chosen)


# ==================================================================================
# Step 4: real font-size fitting (not a lookup table)
# ==================================================================================

def _resolve_font_path(font_key: str) -> str:
    spec = FONT_CATALOG.get(font_key) or FONT_CATALOG[DEFAULT_FONT_KEY]
    return str(FONTS_DIR / spec["file"])


def _fit_unit(unit: Unit, width: int, height: int, font_path: str) -> List[PlacedBlock]:
    x1, y1, x2, y2 = get_zone_bounding_box(unit.zone, width, height)
    # Small safety margin (mirrors the 0.95 margin geometry.py's own
    # compute_block_metrics() applies after subtracting container padding) --
    # keeps the fitted text from touching the zone's hard edge.
    max_w = max(40.0, (x2 - x1) * 0.92)
    total_h = max(40.0, float(y2 - y1))

    weights = [ROLE_SIZE_CEILING.get(b.role, 0.038) for b in unit.blocks]
    weight_sum = sum(weights) or 1.0

    placed: List[PlacedBlock] = []
    for b, w in zip(unit.blocks, weights):
        ceiling_ratio = ROLE_SIZE_CEILING.get(b.role, 0.038)
        base_font_size = max(14, int(width * ceiling_ratio))
        # Multi-block Units split the zone's height proportionally to each
        # member's role ceiling -- a hero+subhead pair gives hero more of the
        # shared column than subhead, instead of an arbitrary equal split.
        height_share = max(30.0, total_h * (w / weight_sum))
        is_uppercase = ROLE_UPPERCASE.get(b.role, False)

        font_size, _measured_w, _measured_h, lines = fit_font_size_px(
            text=b.text,
            font_path=font_path,
            base_font_size=base_font_size,
            max_width_px=max_w,
            max_height_px=height_share,
            is_uppercase=is_uppercase,
            allow_wrap=True,
        )
        display_text = "\n".join(lines) if lines else b.text
        placed.append(PlacedBlock(
            block=b, font_size=font_size, display_text=display_text, is_uppercase=is_uppercase,
        ))
    return placed


# ==================================================================================
# Top-level entry point
# ==================================================================================

def solve(
    blocks: List[Block],
    width: int,
    height: int,
    font_key: str = DEFAULT_FONT_KEY,
) -> List[PlacedUnit]:
    """Runs the full 6-step pipeline (steps 5-6 are partial -- see module
    docstring) and returns one PlacedUnit per group/singleton, each with a
    resolved zone box and per-block fitted font sizes. Never raises on
    ordinary content shapes; degrades (reuses a zone) rather than crashing
    once more Units exist than free zones."""
    font_path = _resolve_font_path(font_key)

    units = _group_into_units(blocks)
    pinned, floating = _split_pinned_floating(units)
    occupied = {u.zone for u in pinned if u.zone}
    _assign_floating_zones(floating, occupied)

    all_units = pinned + floating
    result: List[PlacedUnit] = []
    used_area = 0
    for u in all_units:
        x1, y1, x2, y2 = get_zone_bounding_box(u.zone, width, height)
        used_area += (x2 - x1) * (y2 - y1)
        placed_blocks = _fit_unit(u, width, height, font_path)
        result.append(PlacedUnit(zone=u.zone, box=(x1, y1, x2, y2), placed_blocks=placed_blocks, pinned=u.pinned))

    # Step 6: negative-space check -- log-only this round (see plan's scope;
    # no automatic content-dropping/re-solve loop implemented yet).
    canvas_area = width * height
    coverage = (used_area / canvas_area) if canvas_area else 0.0
    print(f"[tendoo_v2 solver] {len(all_units)} unit(s) placed, zone boxes cover ~{coverage * 100:.1f}% of canvas area")

    return result


__all__ = [
    "Block", "Unit", "PlacedBlock", "PlacedUnit",
    "ROLE_ZONE_PREFERENCE", "ROLE_SIZE_CEILING", "ROLE_UPPERCASE",
    "solve",
]
