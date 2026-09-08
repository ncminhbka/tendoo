"""
SplitColumnLayout implementation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, CALENDAR_ICON_SVG, ColorPalette, PosterContent
from tendoo.layouts.component_engine import get_component_css, render_category_body
from tendoo.layouts.split_column.mask import generate_split_column_mask
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text, resolve_headline_effect


TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


class SplitColumnLayout(BaseLayout):
    """
    Split Column (Cascading Ribbon Sash) Layout:
      - Occupies approximately 36-40% along the left side with an organic, undulating silk ribbon wave.
      - Preserves 60-64% pristine canvas on the right for full-length fashion models, lookbooks, or cosmetics.
      - Uses Vogue / Harper's Bazaar high-fashion editorial typography with left-aligned vertical rhythm.
    """

    @property
    def name(self) -> str:
        return "split_column"

    @property
    def display_name(self) -> str:
        return "Dải Lụa Phân Cột (Split Column / Silk Sash)"

    @property
    def description(self) -> str:
        return (
            "Bố cục dải lụa phẳng phân cột dọc bên trái (x < 0.38), phẳng mịn hoàn toàn không cuộn xoắn, "
            "dành 62% cho người mẫu toàn thân, ảnh lookbook thời trang hoặc mỹ phẩm cao cấp."
        )

    def generate_mask(
        self,
        width: int,
        height: int,
        side: str = "left",
        col_width: float = 0.36,
        delta: float = 0.10,
        wave_amp: float = 0.0,
        wave_freq: float = 1.0,
        int_max: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        return generate_split_column_mask(
            height=height,
            width=width,
            side=side,
            col_width=col_width,
            delta=delta,
            wave_amp=wave_amp,
            wave_freq=wave_freq,
            int_max=int_max,
            **kwargs,
        )

    def get_corridor_prompt(self, style_hint: str = "champagne_silk") -> str:
        if style_hint in ("velvet_drape", "studio_light_pillar"):
            return (
                "A clean, flat, smooth studio ambient shadow wash and soft vertical light column cascading down the left column (x < 0.36), "
                "uniform flat negative copy space with zero texture, diffuse gradient gradation harmonizing with scene tones, seamless transition into the right product stage, "
                "smooth unwrinkled surface, pristine copy space for text, zero clutter, "
                "clean photographic background, text-free column area, no floating graphic text, no poster typography"
            )
        elif style_hint == "minimal_wall":
            return (
                "A clean, flat, diffused vertical studio ambient light column running down the left column (x < 0.36), "
                "uniform flat negative copy space with zero texture, subtle airy gradient softly blending into the scene atmosphere, luminous clean copy space, "
                "zero clutter, clean photographic background, text-free column area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("champagne_silk", "silk_sash"):
            return (
                "A clean, soft, flat sheer silk wash with warm champagne ambient studio glow down the left column (x < 0.36), "
                "uniform flat negative copy space with zero texture, harmonious pastel environmental tones, smooth airy gradient, pristine clean space for typography, "
                "subtle edge feathering blending smoothly into the right-side product stage, zero clutter, "
                "clean photographic background, text-free column area, no floating graphic text, no poster typography"
            )
        else:
            # Default: Soft translucent ambient light veil & sheer silk wash
            return (
                "A clean, soft, flat vertical ambient light veil and sheer silk wash gently flowing down the left column (x < 0.36), "
                "uniform flat negative copy space with zero texture, naturally catching and blending with the environmental colors and warm lighting of the scene, "
                "smooth gradient falloff, clean uncluttered space for typography, "
                "subtle edge feathering seamlessly merging into the right-side product stage, "
                "clean photographic background, text-free column area, no floating graphic text, no poster typography"
            )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """Left silk column safe zone: (y1, x1, y2, x2)."""
        return (0.04, 0.04, 0.96, 0.30)

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
        """Assembles and returns the full HTML document for Playwright rendering."""
        if isinstance(bg_data_uri, int):
            actual_width = bg_data_uri
            actual_height = width
            actual_bg_data_uri = height if isinstance(height, str) else kwargs.get("bg_data_uri", "")
            width, height, bg_data_uri = actual_width, actual_height, actual_bg_data_uri

        # 1. Headline balancing & font sizing ladder (tuned for narrower column width: max_one_line_chars=12)
        raw_hl = content.headline
        hl_lines, metrics = balance_vietnamese_headline(raw_hl, max_one_line_chars=12)

        if hl_lines:
            headline_html = "\n".join(
                f'          <div class="headline-line">{html.escape(line)}</div>'
                for line in hl_lines
            )
            headline_plain = " - ".join(hl_lines)
        else:
            headline_html = ""
            headline_plain = "Poster"

        # Headline visual styling & effects (Embossed, Shadow, LED, Neon, Auto)
        effect_to_use = headline_effect or content.text_effect or "auto"
        _, headline_fill_css, wrap_filter_css = resolve_headline_effect(
            effect=effect_to_use,
            headline_text=raw_hl,
            category=content.category,
            layout_name=self.name,
            palette_is_dark=palette.is_dark,
            accent_color=palette.accent_color,
            headline_color=palette.headline_color,
        )

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
        address = normalize_text(content.address)
        website_link = normalize_text(content.website_link)
        qr_data_uri = content.qr_data_uri or ""

        # Adaptive Category Body Component
        category_body_html = render_category_body(
            content=content,
            palette=palette,
            layout_name=self.name,
        )
        component_css = get_component_css()

        # 3. Read template
        template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

        # 4. Perform substitutions
        replacements = {
            "{{width}}": str(width),
            "{{height}}": str(height),
            "{{headline_plain}}": html.escape(headline_plain),
            "{{bg_data_uri}}": bg_data_uri,
            "{{css_vars}}": palette.to_css_vars(),
            "{{component_css}}": component_css,
            "{{font_size}}": str(metrics["font_size"]),
            "{{line_height}}": f"{metrics['line_height']:.2f}",
            "{{letter_spacing}}": f"{metrics['letter_spacing']:.1f}",
            "{{wrap_filter_css}}": wrap_filter_css,
            "{{headline_fill_css}}": headline_fill_css,
            "{{headline_html}}": headline_html,
            "{{pre_header}}": html.escape(pre_header),
            "{{pre_header_display}}": "flex" if pre_header else "none",
            "{{slogan}}": html.escape(slogan),
            "{{slogan_display}}": "block" if slogan else "none",
            "{{category_body_html}}": category_body_html,
            "{{offer_main}}": html.escape(offer_main),
            "{{badge_display}}": "inline-flex" if offer_main else "none",
            "{{badge_border}}": badge_border,
            "{{offer_sub}}": html.escape(offer_sub),
            "{{offer_sub_display}}": "block" if offer_sub else "none",
            "{{dates}}": f"{CALENDAR_ICON_SVG}{html.escape(dates)}" if dates else "",
            "{{dates_display}}": "block" if dates else "none",
            "{{applicable}}": html.escape(applicable),
            "{{applicable_display}}": "block" if applicable else "none",
            "{{address}}": html.escape(address),
            "{{address_display}}": "block" if address else "none",
            "{{brand}}": html.escape(brand),
            "{{brand_display}}": "block" if brand else "none",
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
