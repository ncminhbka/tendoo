"""
TopDomeLayout implementation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text
from tendoo.layouts.top_dome.mask import generate_top_dome_mask


TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


class TopDomeLayout(BaseLayout):
    """
    Top Arch Dome Layout:
    Focuses all typography inside the upper 35-38% safe zone.
    Leaves the bottom 62-65% completely unconstrained for the hero product, reflections, and splashes.
    """

    @property
    def name(self) -> str:
        return "top_dome"

    @property
    def display_name(self) -> str:
        return "Vòm Đỉnh (Top Arch Dome)"

    @property
    def description(self) -> str:
        return "Bố cục vòm sáng trên cao (y < 0.38), giải phóng 62% không gian bên dưới cho sản phẩm chính."

    def generate_mask(
        self,
        width: int,
        height: int,
        y_max: float = 0.40,
        w_half_top: float = 0.49,
        w_half_bottom: float = 0.38,
        delta: float = 0.04,
        **kwargs,
    ) -> np.ndarray:
        return generate_top_dome_mask(
            height=height,
            width=width,
            y_max=y_max,
            w_half_top=w_half_top,
            w_half_bottom=w_half_bottom,
            delta=delta,
        )

    def get_corridor_prompt(self, style_hint: str = "daylight") -> str:
        if style_hint == "studio_dark":
            return (
                "A clean smooth charcoal dark studio ceiling with soft focused spotlight glow, "
                "elegant pristine empty space, flat gradient vignette, zero clutter, "
                "zero text, no words, no letters"
            )
        elif style_hint == "ribbon":
            return (
                "A luxurious flowing 3D silk ribbon fluttering gracefully across the upper dome, "
                "smooth satin fabric texture, soft studio rim lighting, clean flat ribbon surface "
                "with no pattern, elegant festive banner, zero text, no words, no letters"
            )
        elif style_hint == "gold_bevel":
            return (
                "An elegant arched architectural alcove with soft golden atmospheric rim light, "
                "pristine empty surface, smooth gradient, zero clutter, zero text, no words, no letters"
            )
        else:
            # Default daylight arch
            return (
                "A clean smooth elegant architectural arch dome in the upper sky, "
                "soft glowing daylight ambiance, flat pristine negative space with no clutter, "
                "subtle soft atmospheric gradient, perfectly smooth surface for typography, "
                "zero text, no words, no letters"
            )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """Normalized (y1, x1, y2, x2) defining the primary dome safe zone."""
        return (0.03, 0.08, 0.38, 0.92)

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

        # Build headline HTML lines
        if hl_lines:
            headline_html = "\n".join(
                f'        <div class="headline-line">{html.escape(line)}</div>'
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
        brand = normalize_text(content.brand)
        hotline = normalize_text(content.hotline)
        address = normalize_text(content.address)
        website_link = normalize_text(content.website_link)
        qr_data_uri = content.qr_data_uri or ""

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
            "{{badge_display}}": "flex" if offer_main else "none",
            "{{badge_border}}": badge_border,
            "{{offer_sub}}": html.escape(offer_sub),
            "{{offer_sub_display}}": "block" if offer_sub else "none",
            "{{dates}}": html.escape(dates),
            "{{dates_display}}": "inline-flex" if dates else "none",
            "{{brand}}": html.escape(brand),
            "{{brand_display}}": "block" if brand else "none",
            "{{address}}": html.escape(address),
            "{{address_display}}": "block" if address else "none",
            "{{hotline}}": f"HOTLINE: {html.escape(hotline)}" if hotline else "",
            "{{hotline_display}}": "inline-flex" if hotline else "none",
            "{{website_link}}": html.escape(website_link),
            "{{website_display}}": "block" if website_link else "none",
            "{{qr_data_uri}}": qr_data_uri,
            "{{qr_display}}": "flex" if qr_data_uri else "none",
            "{{custom_css}}": content.custom_css or "",
        }

        rendered = template_text
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)

        return rendered
