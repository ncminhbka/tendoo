"""
src/tendoo_v2/

Prototype package (2026-09-14) for the "role/group" render-plan redesign discussed
at length this session -- see the approved plan at
C:\\Users\\Admin\\.claude\\plans\\robust-moseying-tower.md for full context.

Completely ISOLATED from production `tendoo`/`tendoo_legacy` -- imports select,
already-verified primitives from `tendoo` (font fitting, zone geometry, color
harmony, the Chromium renderer) but defines its own schema/solver/template. Nothing
in `src/tendoo/` or `demo_server.py` imports from here; this package exists purely
to validate the solver/group direction with hand-written mock data before any LLM
integration is attempted (explicitly out of scope this round).
"""
