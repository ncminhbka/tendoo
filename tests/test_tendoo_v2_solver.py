"""
tests/test_tendoo_v2_solver.py

Real pytest coverage for src/tendoo_v2/solver.py's zone-pinning mechanism -- this
was the ONE thing production needs to get right for "user prompt asks for text at
a specific position" (see the 2026-09-15 conversation: paused the diffusion-side
salience/attention-suppression work specifically to verify this foundation first),
and until this file existed there was ZERO automated coverage of it anywhere --
only a manual Playwright script (scripts/test_tendoo_v2_solver.py, needs a browser,
not wired into pytest) and a live-LLM script (scripts/run_tendoo_v2_llm.py, needs a
real GPU server + hosted endpoint, only 1 of 8 cases even touches an explicit zone).

These tests exercise solver.solve() directly with hand-written Blocks -- pure
Python geometry, no LLM/GPU/browser needed, so they run in this dev sandbox.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

from tendoo.engine.geometry import GRID_ZONE_NAMES
from tendoo_v2.schema import Block
from tendoo_v2.solver import solve

WIDTH, HEIGHT = 1024, 1024

# The 9 positions a real user prompt can name (see llm_client.py's SYSTEM_PROMPT) --
# the 8 perimeter grid cells plus the true dead-center spot.
ALL_NAMEABLE_ZONES = list(GRID_ZONE_NAMES) + ["middle_center"]


@pytest.mark.parametrize("zone", ALL_NAMEABLE_ZONES)
def test_explicit_zone_is_honored_exactly(zone: str) -> None:
    """An explicit Block.zone must be a HARD constraint: the resulting PlacedUnit
    lands at exactly that zone, for every one of the 9 nameable positions --
    including "middle_center" (chính giữa), the one position solver.py's own
    auto-assignment (ROLE_ZONE_PREFERENCE) deliberately never picks on its own."""
    blocks = [Block(text="Vị trí yêu cầu", role="hero", zone=zone)]
    placed = solve(blocks, width=WIDTH, height=HEIGHT)

    assert len(placed) == 1
    assert placed[0].zone == zone
    assert placed[0].pinned is True
    x1, y1, x2, y2 = placed[0].box
    assert x2 > x1 and y2 > y1, f"zone={zone!r} produced a degenerate/empty box {placed[0].box}"


def test_middle_center_never_auto_assigned_without_explicit_request() -> None:
    """A block with NO explicit zone must never land on middle_center -- that spot
    is reserved for the product/subject unless the user's prompt explicitly asked
    for center placement (mirrors demo_server.py's _auto_assign_zones() rule and
    solver.py's ROLE_ZONE_PREFERENCE, which excludes it from every role's candidate
    list on purpose)."""
    # Enough floating (unzoned) blocks to fill all 8 perimeter zones and force a
    # 9th -- if middle_center were ever a fallback candidate, it would show up here.
    blocks = [Block(text=f"Khối {i}", role="body") for i in range(9)]
    placed = solve(blocks, width=WIDTH, height=HEIGHT)

    assert all(p.zone != "middle_center" for p in placed), (
        "an unzoned (floating) block landed on middle_center -- auto-assignment "
        "must never pick this zone, only an explicit user-requested pin may"
    )


def test_explicit_middle_center_request_is_still_honored_alongside_others() -> None:
    """The flip side of the previous test: middle_center IS reachable, but only via
    an explicit ask -- and doing so must not disturb other floating blocks' own
    zone assignment."""
    blocks = [
        Block(text="Chữ ở chính giữa", role="hero", zone="middle_center"),
        Block(text="Chữ tự do 1", role="body"),
        Block(text="Chữ tự do 2", role="meta"),
    ]
    placed = solve(blocks, width=WIDTH, height=HEIGHT)
    zones = {p.zone: p for p in placed}

    assert "middle_center" in zones
    assert zones["middle_center"].pinned is True
    # The 2 floating blocks must still resolve to real, non-center zones.
    floating_zones = [p.zone for p in placed if not p.pinned]
    assert len(floating_zones) == 2
    assert "middle_center" not in floating_zones


def test_group_with_one_pinned_member_pins_the_whole_unit() -> None:
    """A `group` ties blocks into one Unit (schema.py) -- if ANY member names an
    explicit zone, the whole Unit is pinned there (solver.py's
    _split_pinned_floating() takes the first zone found among the group's
    members), matching the real "GRAND OPENING"/"MUA 1 TẶNG 1" case from
    prompt_test.txt line 43 (hero+subhead that must render as one placed block)."""
    blocks = [
        Block(text="GRAND OPENING", role="hero", group="opening", zone="top_center"),
        Block(text="MUA 1 TẶNG 1", role="subhead", group="opening"),
    ]
    placed = solve(blocks, width=WIDTH, height=HEIGHT)

    assert len(placed) == 1, "grouped blocks must resolve into exactly one PlacedUnit"
    unit = placed[0]
    assert unit.zone == "top_center"
    assert unit.pinned is True
    assert len(unit.placed_blocks) == 2
    # hero (higher importance) must be ordered before subhead within the unit.
    assert unit.placed_blocks[0].block.role == "hero"


def test_no_zone_still_produces_a_valid_placement() -> None:
    """A block with zone=None (the common case -- most prompts don't name a
    position) must still resolve to SOME real, non-degenerate zone box, not crash
    or silently disappear."""
    placed = solve([Block(text="Không nêu vị trí", role="body")], width=WIDTH, height=HEIGHT)
    assert len(placed) == 1
    assert placed[0].zone in GRID_ZONE_NAMES
    assert placed[0].pinned is False


def test_more_units_than_free_zones_degrades_without_crashing() -> None:
    """Step 3's documented degradation path (solver.py::_assign_floating_zones):
    once every candidate zone is occupied, it must reuse one rather than raise."""
    blocks = [Block(text=f"Khối {i}", role="body") for i in range(12)]
    placed = solve(blocks, width=WIDTH, height=HEIGHT)
    assert len(placed) == 12
    assert all(p.zone for p in placed)
