"""
src/tendoo/layouts/freeform/layout.py

FreeformLayout implementation.
===============================
The 7th layout topology, distinct from the other 6: instead of a fixed 1-2 region
geometry + a fixed PosterContent field-to-CSS-slot mapping, it renders whatever
`content.free_text_blocks` a render plan supplies, each placed into one cell of a
named 3x3 anchor grid (see zones.py).

Exists for the "user's free-form prompt asks for specific text at a specific spot"
case (see prompt_test.txt lines 1-19, and yeu_cau.txt) that the other 6 fixed
templates cannot represent -- they stay unchanged as the default/fallback path when no
custom prompt is given. An upstream LLM stage (not implemented in this file) is
responsible for turning "form fields + free prompt text" into the render plan this
layout consumes; this module only knows how to RENDER a plan, not how to produce one.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import (
    ARROW_RIGHT_ICON_SVG,
    BaseLayout,
    CALENDAR_ICON_SVG,
    CHECK_ICON_SVG,
    ColorPalette,
    CLOCK_ICON_SVG,
    GIFT_ICON_SVG,
    GLOBE_ICON_SVG,
    LOCATION_ICON_SVG,
    PHONE_ICON_SVG,
    PosterContent,
    TAG_ICON_SVG,
)
from tendoo.layouts.font_engine import resolve_font
from tendoo.layouts.freeform.mask import generate_freeform_mask
from tendoo.layouts.freeform.measure import _font_path_for, compute_zone_rects, fit_font_size_px
from tendoo.layouts.freeform.zones import ZONE_DEFAULT_ALIGN, ZONE_GRID_AREA, ZONE_NAMES, ZONE_SELF_ALIGN, is_valid_zone
from tendoo.layouts.text_engine import normalize_text, resolve_headline_effect

TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"

# Optional per-block "icon" name -> the shared cross-platform SVG icon constants
# every other layout's footer already uses (base.py) -- reused here so a block like
# {"text": "0334842155", "zone": "bottom_left", "role": "caption", "icon": "phone"}
# renders with the same visual language as the other 6 layouts' contact footers.
ICON_SVG_BY_NAME: Dict[str, str] = {
    "phone": PHONE_ICON_SVG,
    "location": LOCATION_ICON_SVG,
    "globe": GLOBE_ICON_SVG,
    "calendar": CALENDAR_ICON_SVG,
    "gift": GIFT_ICON_SVG,
    "tag": TAG_ICON_SVG,
    "clock": CLOCK_ICON_SVG,
    "check": CHECK_ICON_SVG,
    "arrow_right": ARROW_RIGHT_ICON_SVG,
}

# Static role -> (font-size ratio of canvas width, font-weight, uppercase?) scale.
# Deliberately simple (no per-line-count adaptive ladder like the other 6 layouts use
# for their single headline) -- freeform posters can hold many independent short text
# blocks at once, so a per-role fixed scale is a more predictable starting point than
# trying to reuse balance_vietnamese_headline()/compute_font_ladder() per block.
ROLE_SCALE: Dict[str, Tuple[float, str, bool]] = {
    "hero": (0.072, "900", True),
    "subtitle": (0.030, "700", False),
    "body": (0.021, "600", False),
    "caption": (0.015, "600", False),
    "badge": (0.023, "800", True),
}


class FreeformLayout(BaseLayout):
    """
    Freeform Multi-Zone Layout:
      - Renders an arbitrary render plan (list of {text, zone, role}) instead of a
        fixed PosterContent field mapping.
      - Reserves whichever subset of the named 3x3 anchor grid the plan requests.
      - Use when the user's own prompt specifies particular text/positions; the other
        6 layouts remain the default when no custom prompt is given.
    """

    @property
    def name(self) -> str:
        return "freeform"

    @property
    def display_name(self) -> str:
        return "Tự Do Đa Vùng (Freeform Multi-Zone)"

    @property
    def description(self) -> str:
        return (
            "Bố cục linh hoạt theo yêu cầu tự do của người dùng: đặt nhiều đoạn text "
            "độc lập vào bất kỳ vùng nào trong lưới 3x3, theo đúng chỉ dẫn trong prompt."
        )

    def generate_mask(
        self,
        width: int,
        height: int,
        zones: Optional[List[str]] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        font_key: str = "bevietnam",
        qr_zone: Optional[str] = None,
        delta: float = 0.05,
        int_max: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Prefer passing `blocks` (the render plan's text blocks, same shape as
        PosterContent.free_text_blocks) + `qr_zone`: sizes each reserved zone to fit
        that zone's ACTUAL measured content (see measure.compute_zone_rects) rather
        than a fixed guess -- this is the path production should use. `zones` (a bare
        list of names, using the static ZONE_RECTS footprint) is kept for simple
        callers that don't have real content yet (e.g. a quick preview).
        """
        zone_rects = None
        if blocks:
            zone_rects = compute_zone_rects(
                blocks=blocks, width=width, height=height, font_key=font_key, qr_zone=qr_zone,
            )
        return generate_freeform_mask(
            height=height,
            width=width,
            zones=zones,
            zone_rects=zone_rects,
            delta=delta,
            int_max=int_max,
            **kwargs,
        )

    def get_corridor_prompt(self, style_hint: str = "daylight") -> str:
        # Generic fallback: real usage should supply its own scene/corridor prompt
        # pair from the render plan (via req.prompt_corridor), since the actual mood
        # of a freeform poster is whatever the user's own prompt describes, not a
        # fixed per-layout aesthetic like the other 6 layouts have.
        #
        # DELIBERATE DESIGN DECISION (see yeu_cau.txt's own note asking about this):
        # ONE shared corridor prompt is used for every reserved zone, however many/
        # scattered they are -- NOT generalized to K+1 branches (1 scene + K per-zone
        # corridors) through denoise_regional_velocity_blended. Two reasons:
        #   1. Cost scales linearly with K -- a 4-5 zone freeform poster (see
        #      prompt_test.txt's coffee-banner case) would mean 4-5x the forward
        #      passes per step, on a pipeline already ~6s/image with just 2 branches.
        #   2. It's less necessary than it first looks: denoise_regional_velocity_
        #      blended still runs ONE shared token sequence -- the "corridor" branch's
        #      prediction at any position still attends (self-attention) to the
        #      surrounding CANVAS tokens' evolving state, not just this prompt's text.
        #      So as long as this prompt describes "declutter/simplify, harmonize
        #      with whatever is locally around you" rather than one fixed absolute
        #      look, scattered zones over a naturally-varied scene (e.g. bright
        #      top-left, shadowed bottom-right) should still pick up locally coherent
        #      tone for free, without paying for extra branches.
        # If a real GPU test later shows a genuinely different look is needed per
        # region (not just tone drift), prefer encoding that variation in the SCENE
        # prompt instead (e.g. "warm window light grazing the left side, cool falloff
        # to the right") -- every reserved zone's corridor will then inherit local
        # coherence from whatever the scene branch is producing nearby, still at zero
        # extra branch cost. Only reach for true K+1 branches if that also proves
        # insufficient on real renders.
        return (
            "A clean, softly lit uncluttered background, gently simplified and "
            "decluttered wherever it appears in the frame -- never a hard-edged flat "
            "block of one fixed color, always blending into and harmonizing with "
            "whatever lighting and tone is locally around it, "
            "pristine space for typography, "
            "clean photographic background, text-free area, no floating graphic text, no poster typography"
        )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        # No single fixed safe zone makes sense here (the reserved region varies per
        # request) -- fall back to a generic top-band for color-harmony sampling,
        # roughly matching top_dome's footprint.
        return (0.04, 0.04, 0.32, 0.96)

    @staticmethod
    def _default_blocks_from_content(content: PosterContent) -> List[Dict[str, Any]]:
        """Fallback render plan synthesized from the fixed PosterContent fields, used
        only when this layout is selected WITHOUT a proper free_text_blocks render plan
        (e.g. called directly, bypassing the LLM planning stage) -- keeps this layout
        from silently rendering a blank poster (no text at all) in that case, matching
        the other 6 layouts' contract of always rendering content.headline."""
        blocks: List[Dict[str, Any]] = []
        headline = normalize_text(content.headline or "")
        if headline:
            blocks.append({"text": headline, "zone": "top_center", "role": "hero"})
        slogan = normalize_text(content.slogan or "")
        if slogan:
            blocks.append({"text": slogan, "zone": "top_center", "role": "subtitle"})
        offer_main = normalize_text(content.offer_main or "")
        if offer_main:
            blocks.append({"text": offer_main, "zone": "bottom_center", "role": "badge"})
        return blocks

    def render_html(
        self,
        content: PosterContent,
        palette: ColorPalette,
        bg_data_uri: str = "",
        width: int = 1024,
        height: int = 1024,
        headline_effect: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Renders HTML for the Freeform Multi-Zone layout."""
        if isinstance(bg_data_uri, int):
            actual_width = bg_data_uri
            actual_height = width
            actual_bg_data_uri = height if isinstance(height, str) else kwargs.get("bg_data_uri", "")
            width, height, bg_data_uri = actual_width, actual_height, actual_bg_data_uri
        blocks = list(content.free_text_blocks or []) or self._default_blocks_from_content(content)

        # Pick one representative block to drive font selection for the whole poster
        # (same "one font per poster" convention the other 6 layouts follow) -- prefer
        # the first "hero" role block, else just the first block, else category alone.
        hero_block = next((b for b in blocks if b.get("role") == "hero"), blocks[0] if blocks else None)
        hero_text = normalize_text((hero_block or {}).get("text", ""))

        font_key = getattr(content, "font_family", "auto")
        resolved_font_key, font_face_css, headline_font_css = resolve_font(
            font_key=font_key,
            category=content.category,
            style_hint=kwargs.get("style_hint", ""),
            text_content=hero_text,
        )
        font_path = _font_path_for(resolved_font_key)

        effect_name = headline_effect or content.text_effect or "auto"
        _, headline_fill_css, wrap_filter_css = resolve_headline_effect(
            effect=effect_name,
            headline_text=hero_text,
            category=content.category,
            layout_name=self.name,
            palette_is_dark=palette.is_dark,
            accent_color=palette.accent_color,
            headline_color=palette.headline_color,
        )

        # Group blocks by zone, preserving request order within each zone (a zone can
        # legitimately hold more than one block, e.g. a "hero" + "subtitle" stacked).
        by_zone: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ZONE_NAMES}
        for b in blocks:
            zone = b.get("zone", "")
            if not is_valid_zone(zone):
                continue  # drop silently rather than crash on a bad LLM output
            by_zone[zone].append(b)

        zone_cells_html = []
        for zone_name in ZONE_NAMES:
            zone_blocks = by_zone[zone_name]
            zone_has_qr = bool(content.qr_data_uri) and getattr(content, "qr_zone", None) == zone_name
            if not zone_blocks and not zone_has_qr:
                continue
            align = ZONE_DEFAULT_ALIGN[zone_name]
            # Cap each zone's footprint (in px, not %, to sidestep grid/percentage
            # sizing ambiguity) so a long/large block placed in a side zone (e.g. a
            # "hero" in middle_left, per prompt_test.txt's own examples) still wraps
            # within a sane width instead of forcing its "auto" track to balloon --
            # the middle row/column (the actual product/scene area) must stay intact.
            max_w_px, max_h_px = int(width * 0.42), int(height * 0.42)
            # Equal-share height budget per block sharing this zone (same heuristic
            # compute_zone_rects() uses for mask sizing -- kept identical so the HTML
            # and the diffusion-reserved mask agree on how much room each block gets).
            num_shares = len(zone_blocks) + (1 if zone_has_qr else 0)
            share_h_px = max_h_px / max(1, num_shares)
            wrap_w_px = min(max_w_px, width * 0.9)
            items_html = []
            for b in zone_blocks:
                text = normalize_text(b.get("text", ""))
                if not text:
                    continue
                role = b.get("role") if b.get("role") in ROLE_SCALE else "body"
                size_ratio, weight, uppercase = ROLE_SCALE[role]
                base_font_size = max(14, int(width * size_ratio))
                # Shrinks the flat ROLE_SCALE size only if this block's actual text is
                # too long to fit its share of the zone at that size (e.g. the
                # render-plan LLM assigned "hero"/"subtitle" to a full sentence, not a
                # short title) -- see fit_font_size_px()'s docstring for the real bug
                # this fixes (confirmed on a live run, 2026-09-10: unchecked overflow
                # flooded the whole canvas).
                font_size, _, _ = fit_font_size_px(text, font_path, base_font_size, wrap_w_px, share_h_px)
                is_hero = b is hero_block
                fill_css = headline_fill_css if is_hero else f"color: {html.escape(b.get('color') or palette.sub_color)};"
                text_transform = "uppercase" if uppercase else "none"
                icon_svg = ICON_SVG_BY_NAME.get(b.get("icon", ""), "")
                items_html.append(
                    f'          <div class="freeform-block freeform-role-{role}" '
                    f'style="font-size:{font_size}px; font-weight:{weight}; text-transform:{text_transform}; '
                    f'text-align:{align}; {fill_css}">{icon_svg}{html.escape(text)}</div>'
                )
            # QR code: a special image element (not a text block) placed into whichever
            # zone the render plan (or the caller) designated via content.qr_zone.
            if zone_has_qr:
                items_html.append(
                    '          <div class="freeform-qr-card">'
                    f'<img class="freeform-qr-img" src="{content.qr_data_uri}" alt="QR Code">'
                    '<div class="freeform-qr-hint">QUÉT MÃ QR</div></div>'
                )
            if not items_html:
                continue
            justify_self, align_self = ZONE_SELF_ALIGN[zone_name]
            zone_cells_html.append(
                f'        <div class="freeform-zone" style="grid-area: {ZONE_GRID_AREA[zone_name]}; '
                f'justify-self: {justify_self}; align-self: {align_self}; '
                f'max-width: {max_w_px}px; max-height: {max_h_px}px; '
                f'align-items: {"flex-start" if align == "left" else ("flex-end" if align == "right" else "center")}; '
                f'text-align: {align};">\n' + "\n".join(items_html) + "\n        </div>"
            )

        zones_html = "\n".join(zone_cells_html)

        template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
        replacements = {
            "{{width}}": str(width),
            "{{height}}": str(height),
            "{{headline_plain}}": html.escape(hero_text),
            "{{bg_data_uri}}": bg_data_uri,
            "{{font_face_css}}": font_face_css,
            "{{headline_font_css}}": headline_font_css,
            "{{css_vars}}": palette.to_css_vars(),
            "{{wrap_filter_css}}": wrap_filter_css,
            "{{zones_html}}": zones_html,
            "{{custom_css}}": content.custom_css or "",
        }

        rendered = template_text
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)

        return rendered
