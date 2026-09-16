"""
TopDomeLayout implementation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, CALENDAR_ICON_SVG, ColorPalette, PosterContent
from tendoo_legacy.layouts.component_engine import get_component_css, render_category_body
from tendoo.layouts.font_engine import resolve_font
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text, resolve_headline_effect
from tendoo_legacy.layouts.top_dome.mask import generate_top_dome_mask


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
                "Soft diffuse downward studio spotlight illumination, ethereal ambient atmospheric haze, "
                "clean smooth dark gradient falloff, luminous negative space for text, "
                "pure diffuse lighting with no ceiling, no walls, no architecture, zero clutter, "
                "clean photographic background, text-free background area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("festive_moon", "ribbon"):
            return (
                "A gentle, ethereal sweep of translucent luminous ambient light across the upper area, "
                "soft golden atmospheric particles, radiant festive glow, smooth luminous gradient, "
                "pure atmospheric lighting with no heavy physical structures, zero clutter, "
                "clean photographic background, text-free background area, no floating graphic text, no poster typography"
            )
        elif style_hint in ("golden_hour", "gold_bevel"):
            return (
                "Warm golden hour atmospheric light descending from above, soft ethereal sunbeams, "
                "luminous golden haze, smooth radiant gradient negative space, "
                "pure light and atmospheric glow with no architectural arches, no alcove, no walls, zero clutter, "
                "clean photographic background, text-free background area, no floating graphic text, no poster typography"
            )
        else:
            # Default daylight: Pure luminous sky light (no architecture)
            return (
                "Soft glowing natural daylight radiating from above, bright airy ambient sky illumination, "
                "clean ethereal atmospheric gradient, luminous pristine copy space, "
                "pure diffuse lighting with no buildings, no arches, no ceiling, no architecture, zero clutter, "
                "clean photographic background, text-free background area, no floating graphic text, no poster typography"
            )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """Normalized (y1, x1, y2, x2) defining the primary dome safe zone."""
        return (0.03, 0.08, 0.38, 0.92)

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

        # Font resolution
        font_key = getattr(content, "font_family", "auto")
        _, font_face_css, headline_font_css = resolve_font(
            font_key=font_key,
            category=content.category,
            style_hint=kwargs.get("style_hint", ""),
            text_content=raw_hl,
        )

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
            "{{font_face_css}}": font_face_css,
            "{{headline_font_css}}": headline_font_css,
            "{{css_vars}}": palette.to_css_vars(),
            "{{component_css}}": component_css,
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
            "{{category_body_html}}": category_body_html,
            "{{offer_main}}": html.escape(offer_main),
            "{{badge_display}}": "flex" if offer_main else "none",
            "{{badge_border}}": badge_border,
            "{{offer_sub}}": html.escape(offer_sub),
            "{{offer_sub_display}}": "block" if offer_sub else "none",
            "{{dates}}": f"{CALENDAR_ICON_SVG}{html.escape(dates)}" if dates else "",
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
