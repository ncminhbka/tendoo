"""
src/tendoo_v2/bridge.py

Thin translation layer wiring tendoo_v2 into REAL production (src/tendoo/demo_server.py),
per explicit instruction (2026-09-14): "Wire thẳng v2 vào v1, push lên server để tôi test
qua giao diện". Turns tendoo_v2's solved `PlacedUnit` list (solver.py -- role/group/zone
already resolved into a real zone + font ceiling per block) into the EXACT `plan_blocks`
dict shape demo_server.py's `PosterContent.free_text_blocks` already expects
(text/zone/size/container/effect/icon/color).

WHY THIS SHAPE, NOT A SECOND RENDERER: demo_server.py's OmniBlockLayout (the real,
already-hardened text engine -- padding/floor fixes, style_variant visuals, mask
generation via actual Chromium DOM measurement) stays completely untouched. Only "which
blocks, what role/group, what zone" is decided differently (llm_client.py's hosted-LLM
call + solver.py's group/zone resolution) than the legacy render_plan flow -- matches
yeu_cau.txt's stated flow exactly: LLM curates -> text engine renders (unchanged) ->
diffusion (unchanged). tendoo_v2's OWN font-fitting (solver.py Step 4) and renderer.py's
VFX treatments are NOT used in this wiring -- v1's SIZE_SCALE-driven autofit already
does that job, hardened by this session's own bug fixes; running two independent sizing
engines on the same text would be redundant and could disagree with each other.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tendoo_v2.solver import PlacedUnit

# role -> (size, container) in v1's SIZE_SCALE/VALID_CONTAINERS vocabulary (see
# src/tendoo/render_plan.py's VALID_SIZES/VALID_CONTAINERS) -- mirrors renderer.py's
# own _ROLE_BASE_CLASS mapping (block-plain/card/pill/button), just translated into
# v1's naming instead of a v2 CSS class, since v1's engine does the actual rendering
# here.
ROLE_TO_V1_SIZE_CONTAINER: Dict[str, Tuple[str, str]] = {
    "hero": ("xlarge", "none"),
    "subhead": ("large", "none"),
    "body": ("medium", "card"),
    "cta": ("medium", "button"),
    "meta": ("small", "pill"),
}


def placed_units_to_plan_blocks(placed_units: List[PlacedUnit]) -> List[Dict[str, Any]]:
    """Flattens every PlacedUnit's blocks into v1 plan_blocks dicts, in solver-resolved
    zone order. Uses the ORIGINAL Block.text (not solver.py's own pre-wrapped
    display_text/font_size) -- v1's OmniBlockLayout re-fits font size and wrapping
    itself against the real zone box, so a second, independent sizing pass here would
    be redundant and could disagree with it."""
    plan_blocks: List[Dict[str, Any]] = []
    for unit in placed_units:
        for pb in unit.placed_blocks:
            b = pb.block
            size, container = ROLE_TO_V1_SIZE_CONTAINER.get(b.role, ("medium", "card"))
            block: Dict[str, Any] = {"text": b.text, "zone": unit.zone, "size": size, "container": container}
            if b.effect:
                block["effect"] = b.effect
            if b.icon:
                block["icon"] = b.icon
            if b.color:
                block["color"] = b.color
            plan_blocks.append(block)
    return plan_blocks


__all__ = ["ROLE_TO_V1_SIZE_CONTAINER", "placed_units_to_plan_blocks"]
