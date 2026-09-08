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
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo.layouts.component_engine import get_component_css, render_category_body
from tendoo.layouts.diagonal_slash.mask import generate_diagonal_slash_mask
from tendoo.layouts.font_engine import resolve_font
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

        # NOTE: balance_vietnamese_headline() deliberately does NOT re-wrap a line that came
        # from an explicit user "\n" ("respects user intent" verbatim, however long) -- so
        # `len(lines)` is a LOGICAL line count, not what the browser actually renders.
        # `.headline-line`'s `word-break: break-word` prevents horizontal overflow for an
        # overlong logical line by wrapping it into extra *visual* lines this function has no
        # way to know about from `len(lines)` alone. Estimate the real visual line count from
        # character width so the size/collision decisions below aren't silently wrong for
        # exactly this kind of long single-\n-segment headline.
        content_width_px = width * 0.40  # .diagonal-content-stack's effective max-width: 40%

        def _visual_line_count(font_px: float) -> int:
            if not lines:
                return 1
            avg_char_w = font_px * 0.60  # uppercase italic, slightly narrower average glyph
            total = 0
            for ln in lines:
                total += max(1, math.ceil((len(ln) * avg_char_w) / max(1.0, content_width_px)))
            return total

        # Dynamic font size scaling based on length, visual-line-count, and canvas (line count
        # matters independently of longest-line length: a headline that wraps into 4 visual
        # lines needs a real size cut too, or the content stack overflows into the
        # bottom-anchored footer -- see the max-height safety net computed below, which reacts
        # to whatever height this still produces).
        longest_line = max(len(l) for l in lines) if lines else 10
        base_size = int(width * 0.072)
        if longest_line > 14:
            base_size = int(base_size * 0.85)
        elif longest_line > 10:
            base_size = int(base_size * 0.92)
        visual_lines_estimate = _visual_line_count(base_size)
        if visual_lines_estimate >= 4:
            base_size = int(base_size * 0.80)
        elif visual_lines_estimate == 3:
            base_size = int(base_size * 0.90)
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

        # --- Content-stack / footer collision guard ---
        # .diagonal-content-stack (kicker + headline + slogan + category body) grows top-down
        # unbounded, while .diagonal-footer is independently anchored to the bottom. At short
        # canvases (16:9 = 576px tall) a rich category body (e.g. product_intro spec chips)
        # easily overflows into the footer -- give the stack a real max-height budget computed
        # from the actual footer size, with overflow:hidden as a safety net so content clips
        # instead of visually overlapping the brand/QR footer.
        ratio = width / max(1, height)
        content_top_pct = 0.065 if ratio >= 1.6 else 0.055
        content_top_px = height * content_top_pct

        qr_img_px = 60 if ratio >= 1.6 else 72
        qr_card_h_px = (qr_img_px + 16 + 18) if has_qr else 0.0  # padding + hint label
        brand_card_h_px = 84.0 if has_contacts else 0.0  # padding + brand-name + wrapped contacts
        footer_content_h_px = max(qr_card_h_px, brand_card_h_px) if has_footer else 0.0
        footer_bottom_px = height * 0.038
        footer_reserved_px = (footer_content_h_px + footer_bottom_px) if has_footer else 0.0

        safety_gap_px = 16
        content_stack_max_height_px = max(80.0, height - content_top_px - footer_reserved_px - safety_gap_px)

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
            "{{content_stack_max_height_px}}": f"{content_stack_max_height_px:.0f}",
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
