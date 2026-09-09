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

from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo.layouts.font_engine import resolve_font
from tendoo.layouts.freeform.mask import generate_freeform_mask
from tendoo.layouts.freeform.zones import ZONE_DEFAULT_ALIGN, ZONE_GRID_AREA, ZONE_NAMES, is_valid_zone
from tendoo.layouts.text_engine import normalize_text, resolve_headline_effect

TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"

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
        delta: float = 0.05,
        int_max: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        return generate_freeform_mask(
            height=height,
            width=width,
            zones=zones,
            delta=delta,
            int_max=int_max,
            **kwargs,
        )

    def get_corridor_prompt(self, style_hint: str = "daylight") -> str:
        # Generic fallback: real usage should supply its own scene/corridor prompt
        # pair from the render plan (via req.prompt_corridor), since the actual mood
        # of a freeform poster is whatever the user's own prompt describes, not a
        # fixed per-layout aesthetic like the other 6 layouts have.
        return (
            "A clean, softly lit uncluttered background with a subtle gentle gradient "
            "(never a hard-edged flat block), diffuse tone harmonizing with the scene, "
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
        _, font_face_css, headline_font_css = resolve_font(
            font_key=font_key,
            category=content.category,
            style_hint=kwargs.get("style_hint", ""),
            text_content=hero_text,
        )

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
            if not zone_blocks:
                continue
            align = ZONE_DEFAULT_ALIGN[zone_name]
            items_html = []
            for b in zone_blocks:
                text = normalize_text(b.get("text", ""))
                if not text:
                    continue
                role = b.get("role") if b.get("role") in ROLE_SCALE else "body"
                size_ratio, weight, uppercase = ROLE_SCALE[role]
                font_size = max(14, int(width * size_ratio))
                is_hero = b is hero_block
                fill_css = headline_fill_css if is_hero else f"color: {html.escape(b.get('color') or palette.sub_color)};"
                text_transform = "uppercase" if uppercase else "none"
                items_html.append(
                    f'          <div class="freeform-block freeform-role-{role}" '
                    f'style="font-size:{font_size}px; font-weight:{weight}; text-transform:{text_transform}; '
                    f'text-align:{align}; {fill_css}">{html.escape(text)}</div>'
                )
            if not items_html:
                continue
            zone_cells_html.append(
                f'        <div class="freeform-zone" style="grid-area: {ZONE_GRID_AREA[zone_name]}; '
                f'justify-items: {"flex-start" if align == "left" else ("flex-end" if align == "right" else "center")}; '
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
