"""
LFrameLayout implementation.
============================
L-Frame Corner Anchor topology:
- Top horizontal bar: bold headline & kicker across the upper canvas (y in [0.0, 0.30]).
- Left vertical column: spec badges, feature pills, steps, or requirements (x in [0.0, 0.38]).
- Preserves 45-55% lower-right open quadrant for hero tech gadgets, appliances, or keynote subjects.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo_legacy.layouts.component_engine import get_component_css, render_category_body
from tendoo.layouts.font_engine import resolve_font
from tendoo_legacy.layouts.l_frame.mask import generate_l_frame_mask
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text, resolve_headline_effect


TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


class LFrameLayout(BaseLayout):
    """
    L-Frame Corner Anchor Layout:
      - Framing along top edge and left column.
      - Lower-right open quadrant reserved for hero subjects.
      - Ideal for technology devices, smart home appliances, workshops, and courses.
    """

    @property
    def name(self) -> str:
        return "l_frame"

    @property
    def display_name(self) -> str:
        return "Khung Góc L (L-Frame Anchor)"

    @property
    def description(self) -> str:
        return (
            "Bố cục khung góc chữ L bám cạnh trên và cột trái, dành 55% góc dưới-phải "
            "cho sản phẩm công nghệ, thiết bị gia dụng, hội thảo, khóa học."
        )

    def generate_mask(
        self,
        width: int,
        height: int,
        y_bar: float = 0.28,
        x_col: float = 0.34,
        delta: float = 0.16,
        side: str = "top_left",
        int_max: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        return generate_l_frame_mask(
            height=height,
            width=width,
            y_bar=y_bar,
            x_col=x_col,
            delta=delta,
            side=side,
            int_max=int_max,
            **kwargs,
        )

    def get_corridor_prompt(self, style_hint: str = "tech_minimal") -> str:
        if style_hint in ("tech_minimal", "minimal_studio"):
            return (
                "A soft dark monochrome studio canvas tone framing the top margin (y < 0.28) and left column (x < 0.34), "
                "uniform negative copy space with a subtle gentle gradient (never a hard-edged flat block), "
                "diffuse gradient gradation harmonizing with the scene tones, soft seamless atmospheric transition blending "
                "naturally into the lower-right product stage with no visible seam or boundary line, "
                "pristine uncluttered space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("cyber_tech", "cyan_circuit"):
            return (
                "A sleek deep navy monochrome canvas tone framing the top margin (y < 0.28) and left column (x < 0.34), "
                "uniform dark negative space with a subtle gentle gradient (never a hard-edged flat block), "
                "diffuse gradient gradation harmonizing with the scene tones, soft seamless atmospheric transition blending "
                "naturally into the lower-right product stage with no visible seam or boundary line, "
                "pristine uncluttered space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("luxury_gold", "warm_editorial"):
            return (
                "A warm charcoal-toned studio canvas background framing the top margin (y < 0.28) and left column (x < 0.34), "
                "uniform negative copy space with a subtle gentle gradient (never a hard-edged flat block), "
                "diffuse gradient gradation harmonizing with the scene tones, soft seamless atmospheric transition blending "
                "naturally into the lower-right product stage with no visible seam or boundary line, "
                "pristine uncluttered space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("daylight_clean", "sunlight_airy"):
            return (
                "A bright, crisp, light studio canvas tone framing the top margin (y < 0.28) and left column (x < 0.34), "
                "uniform light negative space with a subtle gentle gradient (never a hard-edged flat block), "
                "diffuse gradient gradation harmonizing with the scene tones, soft seamless atmospheric transition blending "
                "naturally into the lower-right product stage with no visible seam or boundary line, "
                "pristine uncluttered space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )
        else:
            return (
                "A clean monochrome studio canvas tone framing the top margin (y < 0.28) and left column (x < 0.34), "
                "uniform negative copy space with a subtle gentle gradient (never a hard-edged flat block), "
                "diffuse gradient gradation harmonizing with the scene tones, soft seamless atmospheric transition blending "
                "naturally into the lower-right product stage with no visible seam or boundary line, "
                "pristine uncluttered space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """
        Safe zone in normalized coordinates (y0, x0, y1, x1):
        Upper-left corner safe zone for color sampling & typography framing.
        """
        return (0.04, 0.04, 0.28, 0.34)

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
        """
        Renders HTML for the L-Frame Corner layout.
        """
        if isinstance(bg_data_uri, int):
            actual_width = bg_data_uri
            actual_height = width
            actual_bg_data_uri = height if isinstance(height, str) else kwargs.get("bg_data_uri", "")
            width, height, bg_data_uri = actual_width, actual_height, actual_bg_data_uri
        headline_plain = normalize_text(content.headline or "")
        lines, metrics = balance_vietnamese_headline(headline_plain, max_one_line_chars=16)

        # NOTE: balance_vietnamese_headline() deliberately does NOT re-wrap a line that came
        # from an explicit user "\n" (it "respects user intent" verbatim, however long that
        # line is) -- so `len(lines)` is a LOGICAL line count, not the number of lines the
        # browser will actually render. `.headline-line`'s `word-break: break-word` in the
        # template is what prevents horizontal overflow for an overlong logical line, but it
        # does so by wrapping into extra *visual* lines that this function has no way to know
        # about from `len(lines)` alone. Estimate the real visual line count from character
        # width so the sizing/collision-guard math below isn't silently wrong for exactly the
        # kind of long single-\n-segment headline that triggered the original overlap bug.
        top_bar_width_px = width * 0.65  # matches .lframe-top-bar's max-width: 65%

        def _visual_line_count(font_px: float) -> int:
            if not lines:
                return 1
            avg_char_w = font_px * 0.62  # uppercase, bold condensed-ish average glyph width
            total = 0
            for ln in lines:
                total += max(1, math.ceil((len(ln) * avg_char_w) / max(1.0, top_bar_width_px)))
            return total

        # Dynamic font sizing ladder (char-length AND visual-line-count aware -- a headline
        # that wraps into 3-4 visual lines needs a real size cut too, not just longest-line
        # length, otherwise the top-bar's total block height grows unchecked; see the
        # collision guard below which reacts to whatever height this still produces).
        longest_line = max(len(l) for l in lines) if lines else 10
        base_size = int(width * 0.068)
        if longest_line > 16:
            base_size = int(base_size * 0.85)
        elif longest_line > 12:
            base_size = int(base_size * 0.92)
        visual_lines_estimate = _visual_line_count(base_size)
        if visual_lines_estimate >= 4:
            base_size = int(base_size * 0.80)
        elif visual_lines_estimate == 3:
            base_size = int(base_size * 0.90)
        headline_font_size = max(26, min(base_size, 84))

        # Headline HTML
        headline_line_divs = []
        for line in lines:
            headline_line_divs.append(f'          <div class="headline-line">{html.escape(line)}</div>')
        headline_html = "\n".join(headline_line_divs)

        # Headline effect styling
        effect_name = headline_effect or content.text_effect or "auto"
        _, headline_fill_css, wrap_filter_css = resolve_headline_effect(
            effect=effect_name,
            headline_text=headline_plain,
            category=content.category,
            layout_name=self.name,
            palette_is_dark=palette.is_dark,
            accent_color=palette.accent_color,
            headline_color=palette.headline_color,
        )

        # Font resolution
        font_key = getattr(content, "font_family", "auto")
        _, font_face_css, headline_font_css = resolve_font(
            font_key=font_key,
            category=content.category,
            style_hint=kwargs.get("style_hint", ""),
            text_content=headline_plain,
        )

        # Normalization
        pre_header = normalize_text(content.pre_header)
        slogan = normalize_text(content.slogan)
        brand = normalize_text(content.brand)
        hotline = normalize_text(content.hotline)
        address = normalize_text(content.address)
        website_link = normalize_text(content.website_link)
        qr_data_uri = content.qr_data_uri or ""

        # --- Dynamic top-bar / left-column collision guard ---
        # .lframe-top-bar (pre-header + headline + slogan) and .lframe-left-column (category
        # body) are two independently-positioned absolute blocks. Previously the left column's
        # "top" was a hardcoded percentage (matching the mask's y_bar) with no awareness of how
        # tall the top bar's actual content grew -- a 3-4 line headline + slogan would overflow
        # past that fixed line and visually overlap the category body below it. Since y_bar is
        # also what the diffusion mask reserves as "text territory", an under-sized budget here
        # is the same reason text could spill onto the product region in the generated photo.
        # Compute the real pixel budget needed from the actual content and reposition the left
        # column to clear it, with a max-height + overflow:hidden safety net on the top bar
        # itself in case content is still unusually long.
        top_bar_start_pct = 0.055 if (width / max(1, height)) >= 1.6 else 0.048
        top_bar_start_px = height * top_bar_start_pct
        # top_bar_width_px already computed above, reused here for the slogan-width estimate.

        pre_header_h_px = (13 * 1.25 + 8) if pre_header else 0.0
        line_height_px = headline_font_size * 1.16
        final_visual_lines = _visual_line_count(headline_font_size)
        # Gaps (8px) are a flex `gap` between .headline-line DIVS, not between wrapped rows
        # within the same div, so they scale with the logical (len(lines)), not visual, count.
        headline_block_h_px = final_visual_lines * line_height_px + max(0, len(lines) - 1) * 8

        slogan_h_px = 0.0
        if slogan:
            slogan_width_px = top_bar_width_px * 0.90  # .slogan-text max-width: 90%
            chars_per_line = max(18, int(slogan_width_px / 7.3))
            slogan_lines = max(1, -(-len(slogan) // chars_per_line))  # ceil division
            slogan_h_px = slogan_lines * (14.5 * 1.45) + 4

        content_h_px = pre_header_h_px + headline_block_h_px + 6 + slogan_h_px  # 6 = title-wrap margin
        min_gap_px = 16

        y_bar_baseline_px = height * 0.28  # matches generate_mask()'s default y_bar
        left_col_top_px = max(y_bar_baseline_px, top_bar_start_px + content_h_px + min_gap_px)
        # Never push the left column past ~62% of the canvas -- keep the lower-right product
        # stage genuinely open; the max-height safety net below absorbs anything past this.
        left_col_top_px = min(left_col_top_px, height * 0.62)

        top_bar_max_height_px = max(60.0, left_col_top_px - top_bar_start_px - min_gap_px)

        # Category Body Component
        category_body_html = render_category_body(
            content=content,
            palette=palette,
            layout_name=self.name,
        )
        component_css = get_component_css()

        # Read template
        template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

        # Footer item visibility
        has_contacts = bool(brand or hotline or address or website_link)
        has_qr = bool(qr_data_uri)
        has_footer = has_contacts or has_qr

        replacements = {
            "{{width}}": str(width),
            "{{height}}": str(height),
            "{{headline_plain}}": html.escape(headline_plain),
            "{{bg_data_uri}}": bg_data_uri,
            "{{font_face_css}}": font_face_css,
            "{{headline_font_css}}": headline_font_css,
            "{{css_vars}}": palette.to_css_vars(),
            "{{component_css}}": component_css,
            "{{headline_font_size}}": str(headline_font_size),
            "{{wrap_filter_css}}": wrap_filter_css,
            "{{headline_fill_css}}": headline_fill_css,
            "{{headline_html}}": headline_html,
            "{{left_col_top_px}}": f"{left_col_top_px:.0f}",
            "{{top_bar_max_height_px}}": f"{top_bar_max_height_px:.0f}",
            "{{pre_header}}": html.escape(pre_header),
            "{{pre_header_display}}": "flex" if pre_header else "none",
            "{{slogan}}": html.escape(slogan),
            "{{slogan_display}}": "block" if slogan else "none",
            "{{category_body_html}}": category_body_html,
            "{{footer_display}}": "flex" if has_footer else "none",
            "{{brand_wrap_display}}": "flex" if has_contacts else "none",
            "{{brand}}": html.escape(brand),
            "{{brand_display}}": "block" if brand else "none",
            "{{hotline}}": f"HOTLINE: {html.escape(hotline)}" if hotline else "",
            "{{hotline_display}}": "inline-flex" if hotline else "none",
            "{{address}}": html.escape(address),
            "{{address_display}}": "inline-flex" if address else "none",
            "{{website_link}}": html.escape(website_link),
            "{{website_display}}": "inline-flex" if website_link else "none",
            "{{qr_data_uri}}": qr_data_uri,
            "{{qr_display}}": "flex" if has_qr else "none",
            "{{custom_css}}": content.custom_css or "",
        }

        rendered = template_text
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)

        return rendered
