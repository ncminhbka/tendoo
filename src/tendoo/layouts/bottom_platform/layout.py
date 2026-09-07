"""
BottomPlatformLayout implementation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo.layouts.bottom_platform.mask import generate_bottom_platform_mask
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text


TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


class BottomPlatformLayout(BaseLayout):
    """
    Bottom Platform (Cinematic Base) Layout:
      - Upper 60% is strictly 0.0 copy space, preserving unobstructed scenic photography,
        sky, architecture, luxury vehicles, or grand real estate.
      - Lower 35-40% forms a solid, elegant platform/pedestal holding commanding theatrical typography.
    """

    @property
    def name(self) -> str:
        return "bottom_platform"

    @property
    def display_name(self) -> str:
        return "Bệ Đáy Điện Ảnh (Bottom Platform)"

    @property
    def description(self) -> str:
        return (
            "Bố cục bệ sàn điện ảnh ở đáy, giải phóng 60% không gian phía trên cho cảnh quan "
            "hùng vĩ, kiến trúc biệt thự, xe hơi hoặc sản phẩm cao cấp."
        )

    def generate_mask(
        self,
        width: int,
        height: int,
        y_start: float = 0.58,
        y_full: float = 0.72,
        curvature: float = 0.025,
        w_half: float = 0.49,
        delta_x: float = 0.04,
        int_max: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        return generate_bottom_platform_mask(
            height=height,
            width=width,
            y_start=y_start,
            y_full=y_full,
            curvature=curvature,
            w_half=w_half,
            delta_x=delta_x,
            int_max=int_max,
            **kwargs,
        )

    def get_corridor_prompt(self, style_hint: str = "cinematic_asphalt") -> str:
        if style_hint == "luxury_marble":
            return (
                "A grand dark polished obsidian marble platform terrace, dramatic subtle ambient rim light, "
                "pristine copy space, zero clutter, zero text, no words, no letters"
            )
        elif style_hint == "cyberpunk_grid":
            return (
                "A sleek dark metallic stage floor with subtle neon edge glow, minimal futuristic studio reflection, "
                "pristine copy space, zero clutter, zero text, no words, no letters"
            )
        elif style_hint == "nature_stone":
            return (
                "A natural dark slate stone terrace ground, subtle mist, atmospheric rim light, "
                "pristine copy space, zero clutter, zero text, no words, no letters"
            )
        else:
            # Default cinematic asphalt / wet polished floor
            return (
                "A clean sleek dark wet asphalt road surface reflecting atmospheric lights, cinematic mist, "
                "pristine copy space, zero clutter, zero text, no words, no letters"
            )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """Lower platform safe zone: (y1, x1, y2, x2)."""
        return (0.60, 0.05, 0.97, 0.95)

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
        hl_lines, metrics = balance_vietnamese_headline(raw_hl, max_one_line_chars=18)

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
                "filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.95)) "
                "drop-shadow(0 1px 3px rgba(0, 0, 0, 0.85));"
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
            "{{pre_header_display}}": "inline-flex" if pre_header else "none",
            "{{slogan}}": html.escape(slogan),
            "{{slogan_display}}": "block" if slogan else "none",
            "{{offer_main}}": html.escape(offer_main),
            "{{badge_display}}": "inline-flex" if offer_main else "none",
            "{{badge_border}}": badge_border,
            "{{offer_sub}}": html.escape(offer_sub),
            "{{offer_sub_display}}": "block" if offer_sub else "none",
            "{{dates}}": f"📅 {html.escape(dates)}" if dates else "",
            "{{dates_display}}": "block" if dates else "none",
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
