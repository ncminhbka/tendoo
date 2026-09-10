"""
src/tendoo/engine/layout.py

OmniBlockLayout -- Bộ Dàn Trang Khối Đa Năng Thích Ứng (Tendoo Omni-Block Engine):
- Kế thừa BaseLayout, đóng vai trò Engine thống nhất thay thế cho 7 layouts cũ.
- Tự động chuyển đổi các trường dữ liệu form thành các AdaptiveBlock ngữ nghĩa.
- Khử trùng lặp 100% bằng Content Fingerprint.
- Sinh Parametric Corridor Mask duy nhất cho 2-branch velocity blending O(1).
- Render HTML/CSS bằng Master Template với các visual component macro cao cấp.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jinja2
import numpy as np

from tendoo.core.fonts import FONT_CATALOG, FONTS_DIR, resolve_font
from tendoo.core.typography import normalize_text, resolve_headline_effect
from tendoo.engine.blocks import (
    AdaptiveBlock,
    deduplicate_blocks,
    map_category_to_default_blocks,
)
from tendoo.engine.geometry import compute_block_metrics
from tendoo.engine.mask import generate_omni_corridor_mask
from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent
from tendoo.layouts.freeform.layout import ICON_SVG_BY_NAME
from tendoo.layouts.freeform.zones import (
    ZONE_DEFAULT_ALIGN,
    ZONE_GRID_AREA,
    ZONE_NAMES,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
)


class OmniBlockLayout(BaseLayout):
    """Layout động dựa trên khối ngữ nghĩa, không phụ thuộc template tĩnh."""

    @property
    def name(self) -> str:
        return "omni"

    @property
    def display_name(self) -> str:
        return "Bố cục Thích ứng Đa Năng (Omni-Block)"

    @property
    def description(self) -> str:
        return "Tự động phân bổ khối chữ theo ngữ nghĩa ngành nghề, linh hoạt di chuyển theo prompt, bảo vệ vùng sản phẩm."

    def _resolve_font_path(self, font_key: str) -> str:
        meta = FONT_CATALOG.get(font_key) or FONT_CATALOG["bevietnam"]
        return str(FONTS_DIR / meta["file"])

    def _extract_blocks(
        self,
        content: PosterContent,
        blocks_input: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AdaptiveBlock]:
        """Trích xuất và chuẩn hóa danh sách AdaptiveBlock từ PosterContent."""
        if blocks_input:
            blocks = [AdaptiveBlock.from_dict(b) for b in blocks_input]
        elif content.free_text_blocks:
            blocks = [AdaptiveBlock.from_dict(b) for b in content.free_text_blocks]
        else:
            # Fallback từ form fields khi không có plan bên ngoài
            fields = {
                "discount": content.offer_main,
                "offer_main": content.offer_main,
                "applied_product": content.offer_sub,
                "offer_sub": content.offer_sub,
                "dates": content.dates,
                "product_name": content.product_name,
                "product_desc": content.product_desc,
                "highlights": content.highlights,
                "price": content.price,
                "opening_date": content.opening_date,
                "opening_promo": content.opening_promo,
                "booking_contact": content.booking_contact,
                "feedback_target": content.feedback_target,
                "feedback_quote": content.feedback_quote,
                "feedback_rating": content.feedback_rating,
                "special_offer": content.special_offer,
                "job_position": content.job_position,
                "job_desc": content.job_desc,
                "apply_deadline": content.apply_deadline,
                "apply_method": content.apply_method,
                "guide_steps": " | ".join(content.steps) if content.steps else "",
            }
            store_info = {
                "store_name": content.brand,
                "phone": content.hotline,
                "address": content.address,
            }
            blocks = map_category_to_default_blocks(
                category=content.category,
                title=content.headline,
                fields=fields,
                store_info=store_info,
            )

        # Khử trùng lặp nội dung 100%
        return deduplicate_blocks(blocks, hero_title=content.headline)

    def generate_mask(
        self,
        width: int,
        height: int,
        blocks: Optional[List[Dict[str, Any]]] = None,
        font_key: str = "bevietnam",
        **kwargs,
    ) -> np.ndarray:
        """Sinh ma trận mask float32 [height, width] với biên mềm Gaussian."""
        font_path = self._resolve_font_path(font_key)
        adaptive_blocks: List[AdaptiveBlock] = []
        if blocks:
            adaptive_blocks = [AdaptiveBlock.from_dict(b) for b in blocks]

        return generate_omni_corridor_mask(
            width=width,
            height=height,
            blocks=adaptive_blocks,
            font_path=font_path,
        )

    def get_corridor_prompt(self, style_hint: str = "daylight") -> str:
        """Prompt corridor dùng chung 1 nhánh cho toàn bộ các vùng chữ."""
        return (
            "A clean, softly lit uncluttered background, gently simplified and "
            "decluttered wherever it appears in the frame, "
            "harmonizing naturally with local surroundings, "
            "pristine space for typography, clean photographic background, "
            "text-free area, no floating graphic text, no poster typography"
        )

    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """Safe zone cho color harmony sampling (dải trên)."""
        return (0.04, 0.04, 0.32, 0.96)

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
        """Render HTML hoàn chỉnh từ Jinja2 Master Template."""
        if isinstance(bg_data_uri, int):
            actual_w = bg_data_uri
            actual_h = width
            actual_bg = height if isinstance(height, str) else kwargs.get("bg_data_uri", "")
            width, height, bg_data_uri = actual_w, actual_h, actual_bg

        blocks = self._extract_blocks(content)

        # Chọn font
        font_key = getattr(content, "font_family", "auto")
        hero_block = next((b for b in blocks if b.role == "hero"), blocks[0] if blocks else None)
        hero_text = normalize_text(hero_block.text if hero_block else (content.headline or ""))

        resolved_font_key, font_face_css, headline_font_css = resolve_font(
            font_key=font_key,
            category=content.category,
            style_hint=kwargs.get("style_hint", ""),
            text_content=hero_text,
        )
        font_path = self._resolve_font_path(resolved_font_key)

        # Headline effect
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

        # Phân chia blocks vào các khu vực
        top_bar_blocks: List[AdaptiveBlock] = []
        bottom_bar_blocks: List[AdaptiveBlock] = []
        zone_blocks: Dict[str, List[AdaptiveBlock]] = {z: [] for z in ZONE_NAMES}

        for b in blocks:
            z = b.zone or "top_left"
            if z == "top_bar":
                top_bar_blocks.append(b)
            elif z == "bottom_bar":
                bottom_bar_blocks.append(b)
            elif z in zone_blocks:
                zone_blocks[z].append(b)
            else:
                zone_blocks["top_left"].append(b)

        # Render Top Bar HTML
        top_bar_html = ""
        if top_bar_blocks:
            top_parts = []
            for b in top_bar_blocks:
                icon_svg = ICON_SVG_BY_NAME.get(b.icon or "", "")
                top_parts.append(f"<span>{icon_svg}{html.escape(b.text)}</span>")
            top_bar_html = "  •  ".join(top_parts)

        # Render Bottom Bar HTML
        bottom_bar_html = ""
        if bottom_bar_blocks:
            bot_parts = []
            for b in bottom_bar_blocks:
                icon_svg = ICON_SVG_BY_NAME.get(b.icon or "", "")
                bot_parts.append(f"<span>{icon_svg}{html.escape(b.text)}</span>")
            # Nếu có QR code và không chỉ định zone riêng, gắn vào góc phải bottom bar
            if content.qr_data_uri and getattr(content, "qr_zone", None) in (None, "bottom_bar"):
                bot_parts.append(f'<div class="qr-badge"><img src="{content.qr_data_uri}" alt="QR" /></div>')
            bottom_bar_html = "".join(bot_parts)

        # Render 3x3 Zone Cells HTML
        zone_cells: Dict[str, str] = {}
        for z_name, z_blist in zone_blocks.items():
            z_has_qr = bool(content.qr_data_uri) and getattr(content, "qr_zone", None) == z_name
            if not z_blist and not z_has_qr:
                continue

            cell_items = []
            for b in z_blist:
                metrics = compute_block_metrics(b, font_path, width, height)
                f_size = metrics["font_size"]
                f_weight = metrics["font_weight"]
                disp_text = metrics["display_text"]
                text_trans = "uppercase" if metrics["is_uppercase"] else "none"

                icon_svg = ICON_SVG_BY_NAME.get(b.icon or "", "")
                escaped_text = html.escape(disp_text).replace("\n", "<br>")

                if b.role == "hero":
                    cell_items.append(
                        f'<div class="block-hero" style="font-size:{f_size}px; font-weight:{f_weight}; '
                        f'text-transform:{text_trans};">{escaped_text}</div>'
                    )
                elif b.role == "badge":
                    cell_items.append(
                        f'<div class="block-badge" style="font-size:{f_size}px;">'
                        f'{icon_svg}<span>{escaped_text}</span></div>'
                    )
                elif b.role == "step_list":
                    step_num = f"{b.step_index:02d}" if b.step_index is not None else "•"
                    cell_items.append(
                        f'<div class="block-step" style="font-size:{f_size}px;">'
                        f'<div class="step-badge">{step_num}</div><span>{escaped_text}</span></div>'
                    )
                elif b.style_variant == "quote":
                    cell_items.append(
                        f'<div class="block-quote" style="font-size:{f_size}px;">'
                        f'{escaped_text}</div>'
                    )
                elif b.role == "subtitle":
                    cell_items.append(
                        f'<div class="block-subtitle" style="font-size:{f_size}px;">'
                        f'{escaped_text}</div>'
                    )
                elif b.role == "meta":
                    cell_items.append(
                        f'<div class="block-meta" style="font-size:{f_size}px;">'
                        f'{icon_svg}<span>{escaped_text}</span></div>'
                    )
                else:
                    # body / caption
                    cell_items.append(
                        f'<div class="block-body" style="font-size:{f_size}px;">'
                        f'{icon_svg}<span>{escaped_text}</span></div>'
                    )

            if z_has_qr:
                cell_items.append(f'<div class="qr-badge"><img src="{content.qr_data_uri}" alt="QR" /></div>')

            zone_cells[z_name] = "".join(cell_items)

        template = _JINJA_ENV.get_template("master.html")
        return template.render(
            width=width,
            height=height,
            bg_data_uri=bg_data_uri,
            headline_plain=hero_text,
            font_face_css=font_face_css,
            headline_font_css=headline_font_css,
            headline_fill_css=headline_fill_css,
            wrap_filter_css=wrap_filter_css,
            css_vars=palette.to_css_vars(),
            palette=palette,
            top_bar_html=top_bar_html,
            bottom_bar_html=bottom_bar_html,
            zone_cells=zone_cells,
            zone_grid_areas=ZONE_GRID_AREA,
            zone_aligns=ZONE_DEFAULT_ALIGN,
        )
