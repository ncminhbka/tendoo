"""
src/tendoo_v2/schema.py

Prototype content schema for the "role/group" render-plan redesign (see the
approved plan, robust-moseying-tower.md, 2026-09-14).

WHY THIS SCHEMA EXISTS (recap of the session's reasoning, not just a design
choice made in isolation): the CURRENT production schema
(src/tendoo/render_plan.py's extra_blocks) has the LLM choose `zone` + `size` +
`container` per block -- i.e. it makes the model do 2D spatial/geometric
planning. Real research (TextLap: arxiv.org/html/2410.12844v1 -- GPT-4 hit a
98.6% failure rate on spatial tasks without explicit coordinates; LaySPA:
arxiv.org/abs/2509.16891 -- needed specialized RL training just to get a
general-purpose LLM to do layout reliably) and this project's OWN capability
probe (scripts/verify_render_plan_llm.py, run against a live Qwen3-4B-FP8
sidecar: 61% of real cases violated a much simpler length rule) both confirm
this is asking the model to do the one thing it's structurally weakest at.

This schema narrows the model's job to what it's actually good at (content
curation, importance ranking -- confirmed strong on EQ-Bench/copywriting
benchmarks) and extracting an EXPLICIT position when the user's own prompt
states one (a reading-comprehension task, not spatial invention):

- `role`: an ORDINAL importance rank, NOT a pixel size. The deterministic
  solver (see solver.py) turns this into an actual font-size ceiling and a
  zone-preference list -- the model never picks either directly.
- `group`: ties blocks that must stay physically together (e.g. a hero line +
  its supporting subhead, like the real "GRAND OPENING" / "MUA 1 TẶNG 1" case
  from prompt_test.txt line 43) into ONE placement unit, without needing a
  hand-added schema case for every possible pairing pattern.
- `zone`: OPTIONAL, and only ever meant to carry a position the user's free
  prompt stated explicitly (e.g. "ở góc trên bên trái" -> "top_left", see
  prompt_test.txt lines 1-19) -- extraction, never model-invented placement.
  When set, the solver treats it as a hard constraint.
- `field`: ties a block back to a real form field for the fidelity/dedup
  mechanism. Multiple blocks may share the same `field` -- the model is
  allowed to split one field's long value into several shorter, better-looking
  blocks (e.g. "Chống ồn chủ động, âm thanh Hi-Res chuẩn phòng thu" -> 2 pills)
  as long as the real content still appears somewhere across them (not
  dropped, just re-presented).

No LLM is wired up to produce this schema yet (explicitly out of scope this
round, per the approved plan) -- every Block in this prototype is hand-written
mock data, consumed directly by solver.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# Ordinal importance ranks (NOT pixel sizes) + the one function-based tag
# ("cta") that exists for CONTENT TYPE, not importance -- see module docstring.
ROLE_VALUES: Tuple[str, ...] = ("hero", "subhead", "body", "meta", "cta")

# Lower number = higher importance / processed first by the solver. `cta` sits
# alongside "body" in raw importance (a call-to-action isn't usually the single
# most important thing on the poster) but always gets button/ribbon treatment
# regardless of this rank -- see solver.py's ROLE_CONTAINER mapping.
ROLE_ORDER = {"hero": 0, "subhead": 1, "body": 2, "cta": 2, "meta": 3}


@dataclass
class Block:
    """One piece of curated content. Mirrors what an LLM stage would emit in the
    real design (see plan file), but every instance in this prototype is
    hand-written -- there is no parser/validator here, just the shape."""

    text: str
    role: str = "body"
    field: Optional[str] = None
    group: Optional[str] = None
    # Only ever set to simulate "the free prompt named an explicit position" --
    # never invented by this prototype's mock data as a stand-in for "the
    # solver should figure out a good spot" (that's what leaving it None means).
    zone: Optional[str] = None
    effect: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None
    # VFX treatment name (arc | tilted_impact | ribbon | script_accent | seal) --
    # optional, orthogonal to role/zone/effect. See templates/master_v2.html.
    treatment: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ROLE_VALUES:
            raise ValueError(f"Block.role={self.role!r} must be one of {ROLE_VALUES}")


__all__ = ["Block", "ROLE_VALUES", "ROLE_ORDER"]
