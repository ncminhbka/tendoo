"""
tests/test_tendoo_v2_llm_client.py

Covers src/tendoo_v2/llm_client.py's pure functions (JSON extraction, Block
conversion/validation) -- NO real network call, so this runs anywhere. Added
2026-09-15 alongside the fix for a real gap: `_to_blocks()` used to pass an
LLM-returned `zone` string straight through with zero validation, and
`tendoo.engine.geometry.get_zone_bounding_box()` silently falls back to a
top_left-shaped rect for any unrecognized zone name -- so a hallucinated/
misspelled zone (e.g. "center" instead of "middle_center") would silently place
content at the WRONG position with no visible error at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tendoo_v2.llm_client import VALID_LLM_ZONES, _to_blocks


def test_valid_zone_passes_through_unchanged() -> None:
    blocks, errors = _to_blocks([{"text": "Ưu đãi", "role": "hero", "zone": "top_left"}])
    assert errors == []
    assert blocks[0].zone == "top_left"


def test_middle_center_is_a_valid_zone() -> None:
    """The exact gap this test guards: middle_center must be accepted, not
    treated as unrecognized (it was missing from the system prompt's vocabulary
    before this fix, and must not now ALSO be rejected by validation)."""
    assert "middle_center" in VALID_LLM_ZONES
    blocks, errors = _to_blocks([{"text": "Chính giữa", "role": "hero", "zone": "middle_center"}])
    assert errors == []
    assert blocks[0].zone == "middle_center"


def test_hallucinated_zone_is_dropped_not_silently_trusted() -> None:
    """A plausible-looking but WRONG model output ("center" is not a real zone
    name -- the real one is "middle_center") must be dropped to None (falls back
    to the solver's own placement) instead of reaching get_zone_bounding_box()'s
    silent top_left fallback, which would place it at the wrong spot with zero
    visible error."""
    blocks, errors = _to_blocks([{"text": "Chính giữa", "role": "hero", "zone": "center"}])
    assert blocks[0].zone is None
    assert any("invalid zone" in e for e in errors)


def test_misspelled_zone_is_dropped() -> None:
    blocks, errors = _to_blocks([{"text": "Góc trái", "role": "body", "zone": "top-left"}])
    assert blocks[0].zone is None
    assert any("invalid zone" in e for e in errors)


def test_no_zone_is_still_fine() -> None:
    blocks, errors = _to_blocks([{"text": "Không nêu vị trí", "role": "body"}])
    assert errors == []
    assert blocks[0].zone is None


def test_all_9_nameable_zones_individually_valid() -> None:
    for zone in sorted(VALID_LLM_ZONES):
        blocks, errors = _to_blocks([{"text": "x", "role": "body", "zone": zone}])
        assert errors == [], f"zone={zone!r} unexpectedly rejected: {errors}"
        assert blocks[0].zone == zone
    assert len(VALID_LLM_ZONES) == 9
