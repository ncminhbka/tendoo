"""
src/tendoo_v2/renderer.py

Turns solver.py's PlacedUnit list into a real HTML string and (optionally)
screenshots it via the exact same Chromium engine production uses
(tendoo.poster_renderer.PosterRenderer -- reused, not reimplemented).

VFX treatments (`Block.treatment`) are the 5 techniques prototyped and
verified earlier this session (rendered through the real PosterRenderer,
visually confirmed to look good before this module was written -- see the
conversation history for the original scratchpad prototype this ports from):
  - "arc": SVG <textPath>, curved headline.
  - "tilted_impact": rotated 3-layer duotone stack (retro poster look).
  - "ribbon": clip-path polygon banner instead of a plain rounded pill.
  - "script_accent": script/cursive font paired against the bold sans body.
  - "seal": circular badge with curved text around the rim.
No treatment (None) falls back to a plain role-appropriate look (hero/subhead
as bare text, body as a glass card, meta as a small pill, cta as a button) --
simple by design, since this prototype's purpose is validating the SOLVER
(placement/sizing), not re-building the full production style_variant system.
"""

from __future__ import annotations

import base64
import html as html_lib
from pathlib import Path
from typing import List

from tendoo.core.fonts import FONT_CATALOG, FONTS_DIR

from tendoo_v2.schema import Block
from tendoo_v2.solver import PlacedBlock, PlacedUnit, solve

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "master_v2.html"


def _zone_alignment(zone: str) -> str:
    if "left" in zone:
        return "left"
    if "right" in zone:
        return "right"
    return "center"


def _font_data_uri(font_key: str = "bevietnam") -> str:
    spec = FONT_CATALOG.get(font_key) or FONT_CATALOG["bevietnam"]
    font_path = FONTS_DIR / spec["file"]
    data = base64.b64encode(font_path.read_bytes()).decode("ascii")
    return f"data:font/truetype;base64,{data}"


_ROLE_BASE_CLASS = {
    "hero": "block-plain",
    "subhead": "block-plain",
    "body": "block-card",
    "cta": "block-button",
    "meta": "block-pill",
}


def _render_plain_block(pb: PlacedBlock) -> str:
    style = f"font-size:{pb.font_size}px;"
    if pb.block.color:
        style += f"color:{html_lib.escape(pb.block.color)};"
    cls = _ROLE_BASE_CLASS.get(pb.block.role, "block-card")
    text = html_lib.escape(pb.display_text).replace("\n", "<br>")
    return f'<div class="{cls}" style="{style}">{text}</div>'


def _render_tilted_impact(pb: PlacedBlock) -> str:
    text = html_lib.escape(pb.display_text).replace("\n", "<br>")
    fs = pb.font_size
    return (
        '<div class="impact-wrap"><div class="impact-stack">'
        f'<div class="impact-layer l1" style="font-size:{fs}px;">{text}</div>'
        f'<div class="impact-layer l2" style="font-size:{fs}px;">{text}</div>'
        f'<div class="impact-layer l3" style="font-size:{fs}px;">{text}</div>'
        "</div></div>"
    )


def _render_ribbon(pb: PlacedBlock) -> str:
    text = html_lib.escape(pb.display_text)
    return f'<div class="treatment-ribbon" style="font-size:{pb.font_size}px;">{text}</div>'


def _render_script_accent(pb: PlacedBlock) -> str:
    text = html_lib.escape(pb.display_text)
    return f'<div class="treatment-script-accent" style="font-size:{pb.font_size}px;">{text}</div>'


def _render_arc(pb: PlacedBlock, unit_width: int) -> str:
    text = html_lib.escape(pb.display_text)
    fs = pb.font_size
    svg_h = max(80, int(fs * 2.2))
    path_d = f"M 10 {svg_h - 30} Q {unit_width // 2} 10 {unit_width - 10} {svg_h - 30}"
    grad_id = f"arcGoldGrad-{id(pb)}"
    return (
        f'<svg class="treatment-arc-svg" width="{unit_width}" height="{svg_h}" '
        f'viewBox="0 0 {unit_width} {svg_h}">'
        f'<defs><path id="arcpath-{id(pb)}" d="{path_d}" fill="none"/>'
        f'<linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#fff3d6"/><stop offset="50%" stop-color="#e6b954"/>'
        '<stop offset="100%" stop-color="#fff3d6"/></linearGradient></defs>'
        f'<text font-size="{fs}" font-weight="900" fill="url(#{grad_id})" '
        'letter-spacing="2" stroke="rgba(0,0,0,0.35)" stroke-width="1.5">'
        f'<textPath href="#arcpath-{id(pb)}" startOffset="50%" text-anchor="middle">{text}</textPath>'
        "</text></svg>"
    )


def _render_seal(pb: PlacedBlock) -> str:
    text = html_lib.escape(pb.display_text)
    size = max(120, pb.font_size * 4)
    r_outer = size / 2 - 6
    r_inner = r_outer * 0.78
    cx = cy = size / 2
    path_d = f"M {cx - r_outer},{cy} a {r_outer},{r_outer} 0 1,1 {2 * r_outer},0 a {r_outer},{r_outer} 0 1,1 -{2 * r_outer},0"
    return (
        f'<svg class="treatment-seal-svg" width="{size:.0f}" height="{size:.0f}" '
        f'viewBox="0 0 {size:.0f} {size:.0f}">'
        f'<defs><path id="sealpath-{id(pb)}" d="{path_d}"/></defs>'
        f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r_outer:.0f}" fill="none" stroke="#ffd77a" stroke-width="2"/>'
        f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r_inner:.0f}" fill="rgba(0,0,0,0.35)" stroke="#ffd77a" stroke-width="1.5"/>'
        f'<text font-size="14" font-weight="800" fill="#ffe9c4" letter-spacing="2">'
        f'<textPath href="#sealpath-{id(pb)}" startOffset="2%">{text}</textPath></text>'
        f'</svg>'
    )


def _render_block(pb: PlacedBlock, unit_width: int) -> str:
    t = pb.block.treatment
    if t == "tilted_impact":
        return _render_tilted_impact(pb)
    if t == "ribbon":
        return _render_ribbon(pb)
    if t == "script_accent":
        return _render_script_accent(pb)
    if t == "arc":
        return _render_arc(pb, unit_width)
    if t == "seal":
        return _render_seal(pb)
    return _render_plain_block(pb)


def _render_unit(u: PlacedUnit) -> str:
    x1, y1, x2, y2 = u.box
    w, h = x2 - x1, y2 - y1
    align = _zone_alignment(u.zone)
    align_class = f"align-{align}"
    blocks_html = "".join(_render_block(pb, w) for pb in u.placed_blocks)
    return (
        f'<div class="unit {align_class}" data-zone="{u.zone}" '
        f'style="left:{x1}px;top:{y1}px;width:{w}px;height:{h}px;">'
        f"{blocks_html}</div>"
    )


def render_blocks_to_html(
    blocks: List[Block],
    width: int,
    height: int,
    bg_data_uri: str,
    font_key: str = "bevietnam",
) -> str:
    """Solves the layout (solver.solve) and renders it into a full HTML
    document string, ready for PosterRenderer.render()."""
    placed_units = solve(blocks, width=width, height=height, font_key=font_key)
    units_html = "\n".join(_render_unit(u) for u in placed_units)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template
        .replace("__WIDTH__", str(width))
        .replace("__HEIGHT__", str(height))
        .replace("__FONT_DATA_URI__", _font_data_uri(font_key))
        .replace("__BG_DATA_URI__", bg_data_uri)
        .replace("__UNITS_HTML__", units_html)
    )


__all__ = ["render_blocks_to_html"]
