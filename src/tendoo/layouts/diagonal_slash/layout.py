"""
DiagonalSlashLayout implementation.
===================================
Energetic diagonal slash topology (~35°-45°):
- Upper-left corridor (35-48% width) reserved for high-impact italic / athletic typography.
- Diagonal boundary with smooth perpendicular cosine feathering.
- Preserves 55-65% lower-right open space for dynamic hero products
  (sneakers, sports equipment, gym nutrition, gaming gear, energy drinks).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo.layouts.component_engine import get_component_css, render_category_body
from tendoo.layouts.diagonal_slash.mask import generate_diagonal_slash_mask
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text, resolve_headline_effect


TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


class DiagonalSlashLayout(BaseLayout):
    """
    Diagonal Slash Layout:
      - 35°-45° dynamic diagonal corridor across the upper-left quadrant.
      - 55-65% lower-right quadrant preserved for high-energy hero subjects.
      - Bold athletic typography with italic velocity styling.
    """

    @property
    def name(self) -> str:
        return "diagonal_slash"

    @property
    def display_name(self) -> str:
        return "Vát Chéo Năng Động (Diagonal Slash)"

    @property
    def description(self) -> str:
        return (
            "Bố cục đường cắt chéo năng động (~35°-45°), dành 45% góc trên-trái cho tiêu đề mạnh mẽ, "
            "55% góc dưới-phải cho sản phẩm thể thao, sneaker, gym, gaming, nước tăng lực."
        )

    def generate_mask(
        self,
        width: int,
        height: int,
        x_top: float = 0.48,
        x_bottom: float = 0.10,
        delta: float = 0.14,
        side: str = "top_left",
        wave_amp: float = 0.0,
        wave_freq: float = 1.0,
        int_max: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        return generate_diagonal_slash_mask(
            height=height,
            width=width,
            x_top=x_top,
            x_bottom=x_bottom,
            delta=delta,
            side=side,
            wave_amp=wave_amp,
            wave_freq=wave_freq,
            int_max=int_max,
            **kwargs,
        )

    def get_corridor_prompt(self, style_hint: str = "sport_speed") -> str:
        if style_hint in ("sport_speed", "dynamic_velocity"):
            return (
                "A clean, flat, smooth solid dark carbon studio canvas tone across the upper-left diagonal corridor, "
                "completely uniform flat negative space with zero texture, soft seamless diagonal atmospheric transition blending into the active motion scene on the lower-right, "
                "pristine uncluttered copy space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("cyber_neon", "neon_edge"):
            return (
                "A sleek, flat, smooth solid deep navy studio canvas tone across the upper-left diagonal corridor, "
                "completely uniform flat dark negative space with zero texture, soft seamless diagonal atmospheric transition harmonizing with the vibrant tech scene on the lower-right, "
                "pristine uncluttered copy space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("carbon_mesh", "dark_carbon"):
            return (
                "A sleek, flat, smooth solid dark matte studio canvas tone across the upper-left diagonal corridor, "
                "completely uniform flat negative space with zero texture, soft seamless diagonal transition blending into the scene on the lower-right, "
                "pristine uncluttered copy space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("daylight_motion", "sunlight_streak"):
            return (
                "A bright, crisp, flat, smooth solid light studio canvas tone across the upper-left diagonal corridor, "
                "completely uniform flat negative space with zero texture, soft seamless diagonal atmospheric transition blending into the outdoor motion scene on the lower-right, "
                "pristine uncluttered copy space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )
        else:
            return (
                "A clean, flat, smooth solid studio canvas tone across the upper-left diagonal corridor, "
                "completely uniform flat negative space with zero texture, soft seamless diagonal atmospheric transition blending into the scene on the lower-right, "
                "pristine uncluttered copy space for typography, "
                "clean photographic background, text-free corridor area, no floating graphic text, no poster typography"
            )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """
        Safe zone in normalized coordinates (y0, x0, y1, x1):
        Upper-left diagonal corridor occupies (y: 0.04 -> 0.40, x: 0.04 -> 0.40).
        """
        return (0.04, 0.04, 0.40, 0.40)

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
        Renders HTML for the Diagonal Slash layout.
        """
        if isinstance(bg_data_uri, int):
            actual_width = bg_data_uri
            actual_height = width
            actual_bg_data_uri = height if isinstance(height, str) else kwargs.get("bg_data_uri", "")
            width, height, bg_data_uri = actual_width, actual_height, actual_bg_data_uri
        headline_plain = normalize_text(content.headline or "")
        lines, metrics = balance_vietnamese_headline(headline_plain, max_one_line_chars=13)

        # Dynamic font size scaling based on length and canvas
        longest_line = max(len(l) for l in lines) if lines else 10
        base_size = int(width * 0.072)
        if longest_line > 14:
            base_size = int(base_size * 0.85)
        elif longest_line > 10:
            base_size = int(base_size * 0.92)
        headline_font_size = max(28, min(base_size, 88))

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

        # Normalization
        pre_header = normalize_text(content.pre_header)
        slogan = normalize_text(content.slogan)
        brand = normalize_text(content.brand)
        hotline = normalize_text(content.hotline)
        address = normalize_text(content.address)
        website_link = normalize_text(content.website_link)
        qr_data_uri = content.qr_data_uri or ""

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
            "{{css_vars}}": palette.to_css_vars(),
            "{{component_css}}": component_css,
            "{{headline_font_size}}": str(headline_font_size),
            "{{wrap_filter_css}}": wrap_filter_css,
            "{{headline_fill_css}}": headline_fill_css,
            "{{headline_html}}": headline_html,
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
