"""
SplitColumnLayout implementation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo.layouts.split_column.mask import generate_split_column_mask
from tendoo.layouts.text_engine import balance_vietnamese_headline, normalize_text


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
            "Bố cục phân cột dọc dạng dải lụa satin uốn lượn chiếm ~38% bên sườn, "
            "dành 62% cho người mẫu toàn thân, ảnh lookbook thời trang hoặc mỹ phẩm cao cấp."
        )

    def generate_mask(
        self,
        width: int,
        height: int,
        side: str = "left",
        col_width: float = 0.38,
        delta: float = 0.045,
        wave_amp: float = 0.022,
        wave_freq: float = 1.6,
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

    def get_corridor_prompt(self, style_hint: str = "silk_sash") -> str:
        if style_hint == "velvet_drape":
            return (
                "A regal dark velvet drapery curtain panel cascading down the left side, "
                "soft deep shadows, smooth clean fabric space for text, pristine copy space, "
                "zero clutter, zero text, no words, no letters"
            )
        elif style_hint == "minimal_wall":
            return (
                "A sleek vertical architectural wall panel with soft ambient studio side shadow "
                "on the left side, pristine copy space, zero clutter, zero text, no words, no letters"
            )
        elif style_hint == "champagne_silk":
            return (
                "An ethereal shimmering champagne golden silk ribbon sash cascading down the left, "
                "gentle organic satin folds, clean smooth flat surface for text, studio lighting, "
                "pristine copy space, zero clutter, zero text, no words, no letters"
            )
        else:
            # Default luxurious dark silk sash
            return (
                "An elegant luxurious vertical cascading dark silk ribbon sash flowing from top to bottom "
                "on the left side, soft organic satin folds, delicate fabric ripples, clean smooth flat surface for text, "
                "studio lighting, pristine copy space, zero clutter, zero text, no words, no letters"
            )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """Left silk column safe zone: (y1, x1, y2, x2)."""
        return (0.04, 0.045, 0.96, 0.35)

    def render_html(
        self,
        content: PosterContent,
        palette: ColorPalette,
        bg_data_uri: str,
        width: int,
        height: int,
    ) -> str:
        """Assembles and returns the full HTML document for Playwright rendering."""
        # 1. Headline balancing & font sizing ladder (tuned for narrower column width: max_one_line_chars=14)
        raw_hl = content.headline
        hl_lines, metrics = balance_vietnamese_headline(raw_hl, max_one_line_chars=14)

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
            "{{pre_header_display}}": "flex" if pre_header else "none",
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
