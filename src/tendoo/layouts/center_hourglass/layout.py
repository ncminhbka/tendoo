"""
CenterHourglassLayout implementation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo.layouts.center_hourglass.mask import generate_hourglass_mask
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text


TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


class CenterHourglassLayout(BaseLayout):
    """
    Center Hourglass Layout:
    2-tier flowing stream:
      - Upper light funnel (y < 0.42) holds the main headline, subtitle, and primary promotional badges.
      - Central waist (y in [0.42, 0.72]) tapers to frame the central hero product with rim-light.
      - Perspective floor (y >= 0.72) expands outward to hold clean floating footer typography.
    """

    @property
    def name(self) -> str:
        return "center_hourglass"

    @property
    def display_name(self) -> str:
        return "Đồng Hồ Cát (Center Hourglass)"

    @property
    def description(self) -> str:
        return (
            "Bố cục phễu sáng đa tầng ở đỉnh, thắt eo ôm lấy sản phẩm trung tâm "
            "và mở rộng chân sàn cho footer."
        )

    def generate_mask(
        self,
        width: int,
        height: int,
        y_waist: float = 0.56,
        w_half_top: float = 0.485,
        w_half_waist: float = 0.26,
        w_half_floor: float = 0.485,
        int_top: float = 1.0,
        int_waist: float = 0.72,
        int_floor: float = 0.96,
        delta: float = 0.05,
        **kwargs,
    ) -> np.ndarray:
        return generate_hourglass_mask(
            height=height,
            width=width,
            y_waist=y_waist,
            w_half_top=w_half_top,
            w_half_waist=w_half_waist,
            w_half_floor=w_half_floor,
            int_top=int_top,
            int_waist=int_waist,
            int_floor=int_floor,
            delta=delta,
            **kwargs,
        )


    def get_corridor_prompt(self, style_hint: str = "moonbeam") -> str:
        if style_hint == "studio_spotlight":
            return (
                "A focused theatrical vertical studio spotlight beam streaming down "
                "from the top ceiling onto a clean dark polished floor walkway, "
                "pristine copy space, zero clutter, zero text, no words, no letters"
            )
        elif style_hint == "festive_light":
            return (
                "Festive golden volumetric light rays descending from the top, "
                "gentle atmospheric haze, clean floorboards, pristine empty negative space, "
                "zero clutter, zero text, no words, no letters"
            )
        else:
            # Default volumetric moonbeam
            return (
                "Volumetric golden moonbeam, ethereal mist, wooden floor, empty space, "
                "zero clutter, zero text, no words, no letters"
            )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """Upper funnel safe zone: (y1, x1, y2, x2)."""
        return (0.03, 0.06, 0.42, 0.94)

    def render_html(
        self,
        content: PosterContent,
        palette: ColorPalette,
        bg_data_uri: str,
        width: int,
        height: int,
    ) -> str:
        """Assembles and returns the full HTML document for Playwright rendering."""
        # 1. Headline balancing & font sizing ladder
        raw_hl = content.headline
        hl_lines, metrics = balance_vietnamese_headline(raw_hl, max_one_line_chars=16)

        if hl_lines:
            headline_html = "\n".join(
                f'          <div class="headline-line">{html.escape(line)}</div>'
                for line in hl_lines
            )
            headline_plain = " - ".join(hl_lines)
        else:
            headline_html = ""
            headline_plain = "Poster"

        # Headline fill CSS & drop shadow handling
        if palette.headline_is_gradient:
            headline_fill_css = (
                f"background: {palette.headline_color}; "
                "-webkit-background-clip: text; "
                "-webkit-text-fill-color: transparent; "
                "text-shadow: none;"
            )
            wrap_filter_css = (
                "filter: drop-shadow(0 3px 10px rgba(0, 0, 0, 0.90)) "
                "drop-shadow(0 1px 2px rgba(0, 0, 0, 0.70));"
            )
        else:
            headline_fill_css = f"color: {palette.headline_color}; text-shadow: var(--text-shadow);"
            wrap_filter_css = ""

        # Badge border contrast
        if palette.badge_text == "#0D0D14":
            badge_border = "rgba(0, 0, 0, 0.20)"
        else:
            badge_border = "rgba(255, 255, 255, 0.38)"

        # 2. Field presence toggles
        pre_header = normalize_text(content.pre_header)
        slogan = normalize_text(content.slogan)
        offer_main = normalize_text(content.offer_main)
        offer_sub = normalize_text(content.offer_sub)
        dates = normalize_text(content.dates)
        applicable = normalize_text(content.applicable)
        brand = normalize_text(content.brand)
        hotline = normalize_text(content.hotline)

        # 3. Read template
        template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

        # 4. Perform substitutions
        replacements = {
            "{{width}}": str(width),
            "{{height}}": str(height),
            "{{headline_plain}}": html.escape(headline_plain),
            "{{bg_data_uri}}": bg_data_uri,
            "{{css_vars}}": palette.to_css_vars(),
            "{{font_size}}": str(metrics["font_size"]),
            "{{line_height}}": f"{metrics['line_height']:.2f}",
            "{{letter_spacing}}": f"{metrics['letter_spacing']:.1f}",
            "{{wrap_filter_css}}": wrap_filter_css,
            "{{headline_fill_css}}": headline_fill_css,
            "{{headline_html}}": headline_html,
            "{{pre_header}}": html.escape(pre_header),
            "{{pre_header_display}}": "block" if pre_header else "none",
            "{{slogan}}": html.escape(slogan),
            "{{slogan_display}}": "block" if slogan else "none",
            "{{offer_main}}": html.escape(offer_main),
            "{{badge_display}}": "inline-flex" if offer_main else "none",
            "{{badge_border}}": badge_border,
            "{{offer_sub}}": html.escape(offer_sub),
            "{{offer_sub_display}}": "block" if offer_sub else "none",
            "{{dates}}": f"📅 {html.escape(dates)}" if dates else "",
            "{{dates_display}}": "inline-flex" if dates else "none",
            "{{applicable}}": html.escape(applicable),
            "{{applicable_display}}": "block" if applicable else "none",
            "{{brand}}": html.escape(brand),
            "{{brand_display}}": "block" if brand else "none",
            "{{hotline}}": f"HOTLINE: {html.escape(hotline)}" if hotline else "",
            "{{hotline_display}}": "inline-flex" if hotline else "none",
            "{{custom_css}}": content.custom_css or "",
        }

        rendered = template_text
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)

        return rendered
