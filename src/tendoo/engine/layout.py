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

from tendoo.layouts.font_engine import FONT_CATALOG, FONTS_DIR, resolve_font
from tendoo.core.typography import normalize_text, resolve_headline_effect
from tendoo.engine.blocks import (
    AdaptiveBlock,
    deduplicate_blocks,
    map_category_to_default_blocks,
    normalize_zone,
)
from tendoo.core.base import BaseLayout, ColorPalette, PosterContent
from tendoo.core.components import ICON_SVG_BY_NAME, infer_icon_from_text
from tendoo.engine.geometry import (
    compute_block_metrics,
    get_zone_bounding_box,
    GRID_ZONE_NAMES,
    ZONE_DEFAULT_ALIGN,
    ZONE_GRID_AREA,
    ZONE_NAMES,
    ZONE_SELF_ALIGN,
)
from tendoo.engine.mask import (
    generate_omni_corridor_mask,
    generate_omni_corridor_mask_from_rects,
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
        """
        Trích xuất và chuẩn hóa danh sách AdaptiveBlock từ PosterContent.
        
        TẠI SAO CẦN QUY TRÌNH NÀY:
        1. Tính đa hình nguồn dữ liệu: Chấp nhận cả 3 hình thức đầu vào:
           - blocks_input từ LLM Layout Planner (danh sách dict).
           - content.free_text_blocks từ API sinh ảnh trực tiếp.
           - Form fields rời rạc (headline, offer_main, hotline, brand) thông qua map_category_to_default_blocks.
        2. Phòng thủ sâu Store Info: Đảm bảo thương hiệu và hotline luôn xuất hiện ở bottom_bar.
        3. Hạng nhất hóa QR Code: Nhúng QR code vào hệ thống lưới 3x3 như một block chính quy.
        4. Khử trùng lặp Content Fingerprint: Loại bỏ hoàn toàn lỗi lặp lại thông tin giữa các ô.
        """
        if blocks_input:
            blocks = [b if isinstance(b, AdaptiveBlock) else AdaptiveBlock.from_dict(b) for b in blocks_input]
        elif content.free_text_blocks:
            blocks = [b if isinstance(b, AdaptiveBlock) else AdaptiveBlock.from_dict(b) for b in content.free_text_blocks]
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

        # Bảo vệ Tiêu đề chính (Hero Headline Guard):
        # Nếu danh sách blocks truyền từ ngoài vào thiếu block role='hero' nhưng content.headline
        # có dữ liệu, tự động chèn hero block vào đầu danh sách để đảm bảo poster không bị mất tiêu đề.
        if content.headline and not any(b.role == "hero" for b in blocks):
            blocks.insert(0, AdaptiveBlock(
                text=content.headline,
                role="hero",
                zone="top_left",
                field="headline",
            ))

        # Phòng thủ sâu cho Store Info:
        # Nếu danh sách blocks chưa có block nào thuộc top_bar hoặc bottom_bar (hoặc role='brand_bar')
        # mà content có brand/hotline/address -> tự động bổ sung block Store Info ở bottom_bar
        has_store_bar = any(b.zone in ("top_bar", "bottom_bar") or b.role == "brand_bar" for b in blocks)
        if not has_store_bar and (content.brand or content.hotline or content.address):
            store_parts = []
            if content.brand:
                store_parts.append(content.brand)
            if content.hotline:
                store_parts.append(f"Hotline: {content.hotline}")
            if content.address:
                store_parts.append(content.address)
            if store_parts:
                blocks.append(AdaptiveBlock(
                    text="  •  ".join(store_parts),
                    role="brand_bar",
                    zone="bottom_bar",
                    field="store_info",
                    icon="phone" if content.hotline else "globe",
                ))

        # Chuẩn hóa zone cho toàn bộ blocks
        for b in blocks:
            b.zone = normalize_zone(b.zone)

        # Xử lý QR Code thành một AdaptiveBlock hạng nhất:
        # Nếu content có qr_data_uri và chưa có block QR nào trong danh sách -> tự động thêm block QR
        if content.qr_data_uri:
            has_qr_block = any(b.role == "qr" for b in blocks)
            if not has_qr_block:
                qr_z = normalize_zone(getattr(content, "qr_zone", None) or "bottom_right")
                blocks.append(AdaptiveBlock(
                    text=getattr(content, "qr_label", "") or "",
                    role="qr",
                    zone=qr_z,
                    field="qr_code",
                    data_uri=content.qr_data_uri,
                ))
            else:
                for b in blocks:
                    if b.role == "qr" and not b.data_uri:
                        b.data_uri = content.qr_data_uri

        # Tự động suy luận icon vector thông minh theo ngữ nghĩa câu nếu block chưa có icon
        for b in blocks:
            if not b.icon and b.text and b.role not in ("hero", "qr"):
                inferred = infer_icon_from_text(b.text)
                if inferred:
                    b.icon = inferred

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

    def generate_mask_from_render(
        self,
        html_str: str,
        width: int,
        height: int,
        padding_px: int = 14,
        blur_radius_px: int = 16,
    ) -> Optional[np.ndarray]:
        """
        Sinh corridor mask từ hình học THẬT của chính `html_str` sẽ (hoặc đã) được
        render bằng Chromium -- thay vì suy luận offline bằng PIL như `generate_mask()`.

        Đây là điểm sửa gốc cho lỗi "mask không khớp text HTML": trước đây mask được vẽ
        từ 1 bounding-box do PIL/FreeType ước lượng (word-wrap tự viết, không có
        padding/icon/gap/letter-spacing của CSS thật), hoàn toàn độc lập với việc
        Chromium sẽ layout text đó ra sao -- 2 hệ đo khác nhau, không có cách nào đảm
        bảo khớp 100%. Hàm này gọi PosterRenderer.measure_zone_rects(), tức là chạy
        đúng `html_str` này qua chính Chromium sẽ vẽ pixel cuối cùng, để nó tự báo lại
        rect thật (`getBoundingClientRect()`) sau khi script autofit trong master.html
        đã thu nhỏ font vừa khít khung an toàn. Vì mask dùng ĐÚNG rect đó -- không suy
        luận -- nó khớp 100% ở MỌI kích thước đầu ra theo đúng định nghĩa.

        Gọi hàm này với `html_str` được render bằng `render_html(..., bg_data_uri=<nền
        placeholder bất kỳ>)`: màu nền/palette không ảnh hưởng hình học (chỉ đổi màu
        CSS var), nên rect đo được ở bước này giữ nguyên khi `render_html` được gọi lại
        lần 2 với nền thật + palette thật để render ảnh cuối cùng -- cùng width/height/
        text/font/CSS thì Chromium luôn layout ra y hệt.

        Trả về None nếu việc đo thất bại (Playwright lỗi, trang không set được flag...)
        để caller (`demo_server.py`) rơi về `generate_mask()` (PIL) như một lưới an toàn,
        thay vì crash toàn bộ pipeline.
        """
        from tendoo.poster_renderer import PosterRenderer

        try:
            zone_rects = PosterRenderer.measure_zone_rects(html_str, width=width, height=height)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[OmniBlockLayout] Browser mask measurement failed: {e}")
            return None

        if not zone_rects:
            return None

        return generate_omni_corridor_mask_from_rects(
            width=width,
            height=height,
            rects=list(zone_rects.values()),
            padding_px=padding_px,
            blur_radius_px=blur_radius_px,
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
        """
        Render HTML hoàn chỉnh từ Jinja2 Master Template.

        TẠI SAO CẦN KIẾN TRÚC RENDER NÀY:
        1. Xử lý đa hình tham số (Polymorphic Positional Arguments):
           - Trong các layout legacy, chữ ký hàm có thể là (content, palette, width, height, bg_data_uri)
             hoặc (content, palette, bg_data_uri, width, height). Khối kiểm tra bg_data_uri tự động hoán đổi
             giá trị nếu nhận int, đảm bảo không bao giờ bị TypeError khi gọi từ các pipeline cũ.
        2. Tự động hóa Typography & Effect:
           - Tự động tra cứu font tối ưu cho ngành nghề bằng resolve_font() và nạp @font-face CSS.
           - Áp dụng hiệu ứng chữ headline_effect (nhũ vàng, neon phát quang, 3D chrome) hài hòa với bảng màu.
        3. Phân luồng không gian 3x3 Grid & Bar Containers:
           - Top/Bottom bar: Đóng gói hotline, địa chỉ, brand vào viên nang capsule bo tròn với backdrop blur.
           - 8 Spatial Cells: Đo đạc font metrics từng block và gom nhóm theo vị trí không gian.
        4. Bảo vệ Vùng Sản phẩm (Product Sanctuary Enforcement):
           - Tính toán zone_max_w và zone_max_h dựa trên tỷ lệ khung hình và sự hiện diện của Top/Bottom bar,
             ngăn chặn tuyệt đối tình trạng văn bản dài che khuất vật thể trung tâm của FLUX.2 DiT.
        """
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

        # Phân chia blocks vào các khu vực (8 ô ngoại vi + ô tâm middle_center)
        top_bar_blocks: List[AdaptiveBlock] = []
        bottom_bar_blocks: List[AdaptiveBlock] = []
        zone_blocks: Dict[str, List[AdaptiveBlock]] = {z: [] for z in (GRID_ZONE_NAMES + ["middle_center"])}

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

        # Render Bottom Bar HTML (chỉ chứa store info dạng pill vừa vặn, không đính kèm QR)
        bottom_bar_html = ""
        if bottom_bar_blocks:
            bot_parts = []
            for b in bottom_bar_blocks:
                icon_svg = ICON_SVG_BY_NAME.get(b.icon or "", "")
                bot_parts.append(f"<span>{icon_svg}{html.escape(b.text)}</span>")
            bottom_bar_html = "  •  ".join(bot_parts)

        # Render 3x3 Zone Cells HTML
        zone_cells: Dict[str, str] = {}
        for z_name, z_blist in zone_blocks.items():
            if not z_blist:
                continue

            cell_items = []
            for b in z_blist:
                if b.role == "qr":
                    qr_uri = b.data_uri or content.qr_data_uri or ""
                    if qr_uri:
                        caption_html = f'<div class="qr-caption">{html.escape(b.text)}</div>' if b.text else ""
                        cell_items.append(
                            f'<div class="block-qr" data-tendoo-role="qr">'
                            f'<div class="qr-canvas"><img src="{qr_uri}" alt="QR" /></div>'
                            f'{caption_html}</div>'
                        )
                    continue

                metrics = compute_block_metrics(b, font_path, width, height)
                f_size = metrics["font_size"]
                f_weight = metrics["font_weight"]
                disp_text = metrics["display_text"]
                text_trans = "uppercase" if metrics["is_uppercase"] else "none"

                icon_name = b.icon or ""
                # Chống nhân đôi icon star: nếu text đã chứa ký tự sao unicode (★, ⭐) thì không inject thêm SVG star
                if icon_name == "star" and ("★" in b.text or "⭐" in b.text):
                    icon_svg = ""
                else:
                    icon_svg = ICON_SVG_BY_NAME.get(icon_name, "")
                escaped_text = html.escape(disp_text).replace("\n", "<br>")

                variant_suffix = f" style-{b.style_variant.replace('_', '-')}" if b.style_variant and b.style_variant != "default" else ""

                if b.role == "hero":
                    cell_items.append(
                        f'<div class="hero-effect-wrap"><div class="block-hero{variant_suffix}" style="font-size:{f_size}px; font-weight:{f_weight}; '
                        f'text-transform:{text_trans};">{escaped_text}</div></div>'
                    )
                elif b.role == "badge":
                    cell_items.append(
                        f'<div class="block-badge{variant_suffix}" style="font-size:{f_size}px;">'
                        f'{icon_svg}<span>{escaped_text}</span></div>'
                    )
                elif b.role == "step_list":
                    step_num = f"{b.step_index:02d}" if b.step_index is not None else "•"
                    cell_items.append(
                        f'<div class="block-step{variant_suffix}" style="font-size:{f_size}px;">'
                        f'<div class="step-badge">{step_num}</div><span>{escaped_text}</span></div>'
                    )
                elif b.style_variant in ("quote", "quote_bubble"):
                    cell_items.append(
                        f'<div class="block-quote{variant_suffix}" style="font-size:{f_size}px;">'
                        f'{escaped_text}</div>'
                    )
                elif b.role == "subtitle":
                    cell_items.append(
                        f'<div class="block-subtitle{variant_suffix}" style="font-size:{f_size}px;">'
                        f'{escaped_text}</div>'
                    )
                elif b.role == "meta":
                    cell_items.append(
                        f'<div class="block-meta{variant_suffix}" style="font-size:{f_size}px;">'
                        f'{icon_svg}<span>{escaped_text}</span></div>'
                    )
                else:
                    # body / caption
                    cell_items.append(
                        f'<div class="block-body{variant_suffix}" style="font-size:{f_size}px;">'
                        f'{icon_svg}<span>{escaped_text}</span></div>'
                    )

            zone_cells[z_name] = "".join(cell_items)

        # Ràng buộc CSS an toàn theo safe-zone rect (px) cho từng ô có nội dung:
        # - Multiline Hero / Slogan dài được cấp bounds rộng rãi (88-90% width, 45% height)
        #   và trải dài toàn bộ hàng 1 (grid-column: 1 / -1) để tránh bị rectangle cắt mép.
        # - Khi top_center không có block, top_left mở rộng lên tới 52% canvas width.
        # - Vùng Step list trong guide được mở rộng chiều cao (tới 58% canvas height)
        #   để gom toàn bộ các bước thành 1 card dọc chuyên nghiệp, không bị tách rời hay cắt đuôi.
        # - Tính toán zone_max_h chính xác bảo vệ Product Sanctuary kể cả khi có top_bar/bottom_bar.
        zone_max_w: Dict[str, int] = {}
        zone_max_h: Dict[str, int] = {}
        resolved_grid_areas = dict(ZONE_GRID_AREA)

        for z_name in zone_cells:
            zx1, zy1, zx2, zy2 = get_zone_bounding_box(z_name, width, height)
            z_blocks = [b for b in blocks if (b.zone or "top_left") == z_name]
            has_multiline = any(b.role == "hero" and ("\n" in b.text or len(b.text.split()) >= 6) for b in z_blocks)

            # --- HÀNG 1: TOP ROW (top_left, top_center, top_right) ---
            if z_name == "top_center":
                has_tl = "top_left" in zone_cells
                has_tr = "top_right" in zone_cells
                if not has_tl and not has_tr:
                    zone_max_w[z_name] = max(1, int(width * 0.88))
                    if has_multiline:
                        resolved_grid_areas["top_center"] = "1 / 1 / 2 / 4"
                elif has_tl and has_tr:
                    zone_max_w[z_name] = max(1, int(width * 0.36))
                else:
                    zone_max_w[z_name] = max(1, int(width * 0.52))
            elif z_name == "top_left":
                if "top_center" not in zone_cells:
                    zone_max_w[z_name] = max(int(width * 0.48), zx2 - zx1)
                else:
                    has_tr = "top_right" in zone_cells
                    zone_max_w[z_name] = min(zx2 - zx1, int(width * (0.28 if has_tr else 0.35)))
            elif z_name == "top_right":
                if "top_center" not in zone_cells:
                    zone_max_w[z_name] = max(int(width * 0.38), zx2 - zx1)
                else:
                    has_tl = "top_left" in zone_cells
                    zone_max_w[z_name] = min(zx2 - zx1, int(width * (0.28 if has_tl else 0.35)))

            # --- HÀNG 2: MIDDLE ROW (middle_left, middle_center, middle_right) ---
            elif z_name == "middle_center":
                has_ml = "middle_left" in zone_cells
                has_mr = "middle_right" in zone_cells
                if not has_ml and not has_mr:
                    resolved_grid_areas["middle_center"] = "2 / 1 / 3 / 4"
                    zone_max_w[z_name] = max(1, int(width * 0.88))
                elif has_ml and has_mr:
                    zone_max_w[z_name] = max(1, int(width * 0.36))
                else:
                    zone_max_w[z_name] = max(1, int(width * 0.50))
            elif z_name in ("middle_left", "middle_right"):
                zone_max_w[z_name] = max(1, zx2 - zx1)

            # --- HÀNG 3: BOTTOM ROW (bottom_left, bottom_center, bottom_right) ---
            elif z_name == "bottom_center":
                has_bl = "bottom_left" in zone_cells
                has_br = "bottom_right" in zone_cells
                if not has_bl and not has_br:
                    zone_max_w[z_name] = max(1, int(width * 0.88))
                elif has_bl and has_br:
                    zone_max_w[z_name] = max(1, int(width * 0.34))
                else:
                    zone_max_w[z_name] = max(1, int(width * 0.36))
            elif z_name == "bottom_left":
                has_bc = "bottom_center" in zone_cells
                if has_bc:
                    zone_max_w[z_name] = min(zx2 - zx1, int(width * 0.34))
                else:
                    zone_max_w[z_name] = max(1, zx2 - zx1)
            elif z_name == "bottom_right":
                has_bc = "bottom_center" in zone_cells
                if has_bc:
                    zone_max_w[z_name] = min(zx2 - zx1, int(width * 0.36))
                else:
                    zone_max_w[z_name] = max(1, zx2 - zx1)
            else:
                zone_max_w[z_name] = max(1, zx2 - zx1)

            # Giới hạn chiều cao AN TOÀN bảo vệ Product Sanctuary:
            # - Khi có top_bar, lưới omni-grid bị đẩy xuống 13% chiều cao, nên các ô hàng 1
            #   (top_left, top_center, top_right) chỉ còn chiều cao khả dụng từ 13% đến s_y1 - pad.
            # - Khi có bottom_bar, lưới kết thúc ở 87% chiều cao, nên các ô hàng 3
            #   (bottom_left, bottom_center, bottom_right) chỉ còn từ s_y2 + pad đến 87%.
            base_h = zy2 - zy1
            if top_bar_html and z_name in ("top_left", "top_center", "top_right"):
                grid_top_y = int(height * 0.13)
                safe_h = max(30, zy2 - grid_top_y)
                zone_max_h[z_name] = min(base_h, safe_h)
            elif bottom_bar_html and z_name in ("bottom_left", "bottom_center", "bottom_right"):
                grid_bot_y = int(height * 0.87)
                safe_h = max(30, grid_bot_y - zy1)
                zone_max_h[z_name] = min(base_h, safe_h)
            else:
                zone_max_h[z_name] = max(1, base_h)

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
            zone_grid_areas=resolved_grid_areas,
            zone_aligns=ZONE_DEFAULT_ALIGN,
            zone_self_aligns=ZONE_SELF_ALIGN,
            zone_max_w=zone_max_w,
            zone_max_h=zone_max_h,
        )
