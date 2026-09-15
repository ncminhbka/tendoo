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
from tendoo.core.style import STYLE_HINT_DEFAULT_EFFECT
from tendoo.core.typography import normalize_text
from tendoo.engine.blocks import (
    AdaptiveBlock,
    deduplicate_blocks,
    map_category_to_default_blocks,
    normalize_fingerprint,
    normalize_zone,
)
from tendoo.core.base import BaseLayout, ColorPalette, PosterContent
from tendoo.core.components import ICON_SVG_BY_NAME, infer_icon_from_text
from tendoo.engine.geometry import (
    compute_block_metrics,
    estimate_zone_available_box,
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
        raw_blocks = blocks_input or getattr(content, "free_text_blocks", None) or getattr(content, "extra_blocks", None)
        if raw_blocks:
            blocks = [b if isinstance(b, AdaptiveBlock) else AdaptiveBlock.from_dict(b) for b in raw_blocks]
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

        # Bảo vệ Tiêu đề chính (Primary Headline Guard):
        # Nếu KHÔNG có block xlarge nào thực sự mang đúng nội dung content.headline (mandatory,
        # nguồn từ form) thì tự động chèn thêm 1 block tiêu đề xlarge vào đầu danh sách.
        # (2026-09-12 fix: trước đây chỉ kiểm tra "có block xlarge NÀO đó chưa" -- nếu 1 block
        # không liên quan (vd 1 badge khuyến mãi) tình cờ là size=xlarge, content.headline bị
        # bỏ qua hoàn toàn dù là field bắt buộc. Dùng normalize_fingerprint() để so khớp đúng
        # NỘI DUNG, không chỉ sự tồn tại của MỘT block xlarge bất kỳ.)
        headline_fp = normalize_fingerprint(content.headline) if content.headline else ""
        has_headline_block = bool(headline_fp) and any(
            b.size == "xlarge" and normalize_fingerprint(b.text) == headline_fp for b in blocks
        )
        if content.headline and not has_headline_block:
            blocks.insert(0, AdaptiveBlock(
                text=content.headline,
                size="xlarge",
                container="none",
                effect=content.text_effect,
                zone="top_left",
                field="headline",
            ))

        # Chuẩn hóa zone cho toàn bộ blocks TRƯỚC khi kiểm tra has_store_bar bên dưới --
        # (2026-09-12 fix: thứ tự cũ chuẩn hóa SAU has_store_bar khiến 1 block dùng alias zone
        # (vd "footer", bí danh hợp lệ của "bottom_bar") không được nhận diện đúng, khiến hệ
        # thống tưởng chưa có store bar và tự thêm 1 block Store Info thứ 2 -> trùng lặp.)
        for b in blocks:
            b.zone = normalize_zone(b.zone)

        # Phòng thủ sâu cho Store Info:
        # Nếu danh sách blocks chưa có block nào thuộc top_bar hoặc bottom_bar
        # mà content có brand/hotline/address -> tự động bổ sung block Store Info ở bottom_bar
        has_store_bar = any(b.zone in ("top_bar", "bottom_bar") or b.field == "store_info" for b in blocks)
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
                    size="small",
                    container="pill",
                    zone="bottom_bar",
                    field="store_info",
                    icon="phone" if content.hotline else "globe",
                ))

        # Xử lý QR Code thành một AdaptiveBlock hạng nhất:
        if content.qr_data_uri:
            has_qr_block = any(b.container == "qr" or b.field == "qr_code" or b.data_uri for b in blocks)
            if not has_qr_block:
                qr_z = normalize_zone(getattr(content, "qr_zone", None) or "bottom_right")
                blocks.append(AdaptiveBlock(
                    text=getattr(content, "qr_label", "") or "",
                    size="small",
                    container="qr",
                    zone=qr_z,
                    field="qr_code",
                    data_uri=content.qr_data_uri,
                ))
            else:
                for b in blocks:
                    if (b.container == "qr" or b.field == "qr_code") and not b.data_uri:
                        b.data_uri = content.qr_data_uri

        # Tự động suy luận icon vector thông minh theo ngữ nghĩa câu nếu block chưa có icon
        # CHỈ tự động suy luận icon cho các khối có container (button, pill, card), KHÔNG áp dụng cho pure text
        for b in blocks:
            if not b.icon and b.text and b.container not in ("none", "qr"):
                inferred = infer_icon_from_text(b.text)
                if inferred:
                    b.icon = inferred

        # Khử trùng lặp nội dung 100%
        return deduplicate_blocks(blocks)

    def generate_mask(
        self,
        width: int,
        height: int,
        blocks: Optional[List[Dict[str, Any]]] = None,
        font_key: str = "bevietnam",
        qr_data_uri: Optional[str] = None,
        qr_zone: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Sinh ma trận mask float32 [height, width] với biên mềm Gaussian.

        `qr_data_uri`/`qr_zone`: trước đây 2 tham số này bị nuốt im lặng bởi `**kwargs`
        (không hề ảnh hưởng gì) -- nghĩa là mã QR không bao giờ được bảo vệ trong đường
        fallback offline PIL này (khác với `_extract_blocks()`, nơi QR luôn được tổng
        hợp thành 1 AdaptiveBlock hạng nhất trước khi render HTML). Giờ đây, nếu có
        `qr_data_uri` và `blocks` chưa chứa sẵn 1 block QR, tự tổng hợp thêm 1
        AdaptiveBlock QR vào danh sách trước khi sinh mask -- cùng logic với
        `_extract_blocks()` để đường online (Chromium-measured) và đường offline
        (PIL-estimated) luôn bảo vệ vùng QR nhất quán với nhau.
        """
        font_path = self._resolve_font_path(font_key)
        adaptive_blocks: List[AdaptiveBlock] = []
        if blocks:
            adaptive_blocks = [AdaptiveBlock.from_dict(b) for b in blocks]

        if qr_data_uri:
            has_qr_block = any(
                b.container == "qr" or b.field == "qr_code" or b.data_uri
                for b in adaptive_blocks
            )
            if not has_qr_block:
                adaptive_blocks.append(AdaptiveBlock(
                    text="",
                    size="small",
                    container="qr",
                    zone=normalize_zone(qr_zone or "bottom_right"),
                    field="qr_code",
                    data_uri=qr_data_uri,
                ))

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
        primary_block = next((b for b in blocks if b.size == "xlarge"), blocks[0] if blocks else None)
        primary_text = normalize_text(primary_block.text if primary_block else (content.headline or ""))

        resolved_font_key, font_face_css, headline_font_css = resolve_font(
            font_key=font_key,
            category=content.category,
            style_hint=kwargs.get("style_hint", ""),
            text_content=primary_text,
        )
        font_path = self._resolve_font_path(resolved_font_key)
        # (2026-09-13 fix: `.block-card`/`.block-pill`/`.block-cta`/`.block-badge` trong
        # master.html CÓ font-family HARDCODE 'Be Vietnam Pro'/'Plus Jakarta Sans' !important --
        # bất kể headline dùng font gì (vd Playfair Display cho style luxury). Trước đây
        # `compute_block_metrics()` luôn đo TẤT CẢ block (kể cả card/pill/cta/badge) bằng
        # `font_path` của HEADLINE, khiến font_size/wrap tính offline lệch hẳn so với chữ Be
        # Vietnam Pro thật sự render trong Chromium (2 font có bề rộng ký tự khác nhau đáng kể)
        # -- nguyên nhân chính khiến card mô tả dài bị browser tự bẻ dòng thêm 1 lần nữa sau khi
        # đã "vừa" theo phép đo offline, tạo cột chữ lởm chởm. `body_font_path` luôn trỏ đúng file
        # Be Vietnam Pro (giống hệt file được nhúng @font-face bắt buộc trong resolve_font()).
        body_font_path = self._resolve_font_path("bevietnam")

        # Headline effect mặc định nếu block chưa khai báo effect riêng.
        # (2026-09-12: đã bỏ lệnh gọi resolve_headline_effect() ở đây -- kết quả của nó
        # (headline_fill_css/wrap_filter_css) chưa từng được master.html tiêu thụ, tính
        # toán xong rồi bỏ đi mỗi lần render. Hiệu ứng chữ headline thực tế được áp dụng
        # qua cơ chế effect-*/eff_suffix ngay bên dưới, dựa trên default_effect_name này.)
        default_effect_name = headline_effect or content.text_effect or "auto"

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

        # Render Bottom Bar HTML (chứa store info dạng pill vừa vặn, không đính kèm QR)
        bottom_bar_html = ""
        if bottom_bar_blocks:
            bot_parts = []
            for b in bottom_bar_blocks:
                icon_svg = ICON_SVG_BY_NAME.get(b.icon or "", "")
                top_parts_b = f"<span>{icon_svg}{html.escape(b.text)}</span>"
                bot_parts.append(top_parts_b)
            bottom_bar_html = "  •  ".join(bot_parts)

        # Render 3x3 Zone Cells HTML theo Kiến trúc 3 Trục Trực quan
        zone_cells: Dict[str, str] = {}
        for z_name, z_blist in zone_blocks.items():
            if not z_blist:
                continue

            cell_items = []
            for b in z_blist:
                if b.container == "qr" or b.field == "qr_code" or b.data_uri:
                    qr_uri = b.data_uri or content.qr_data_uri or ""
                    if qr_uri:
                        caption_html = f'<div class="qr-caption">{html.escape(b.text)}</div>' if b.text else ""
                        cell_items.append(
                            f'<div class="block-qr" data-tendoo-role="qr">'
                            f'<div class="qr-canvas"><img src="{qr_uri}" alt="QR" /></div>'
                            f'{caption_html}</div>'
                        )
                    continue

                # Đo bằng đúng font sẽ thực sự render: pure-text (container="none") dùng
                # font_path của tiêu đề, mọi container khác (card/pill/cta/badge) dùng
                # body_font_path (Be Vietnam Pro, khớp CSS hardcode ở trên).
                metrics_font_path = font_path if b.container == "none" else body_font_path
                metrics = compute_block_metrics(b, metrics_font_path, width, height)
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

                # Xác định hiệu ứng ánh sáng (effect) cho block
                eff = b.effect
                if not eff and b.container == "none" and b.size in ("xlarge", "large"):
                    if default_effect_name in ("neon", "neon_cyan", "neon_pink", "neon_green", "neon_amber", "3d_gold", "chrome", "fire", "shadow"):
                        eff = "neon_cyan" if default_effect_name == "neon" else default_effect_name
                    else:
                        # (2026-09-13 fix: `default_effect_name` mặc định luôn là "auto" -- giá
                        # trị này không khớp bất kỳ effect thật nào ở trên nên trước đây `eff`
                        # luôn ở lại None, khiến MỌI tiêu đề không set effect tường minh render
                        # chữ phẳng dù CSS effect đã có sẵn đầy đủ. Giờ "auto"/rỗng tra bảng
                        # `STYLE_HINT_DEFAULT_EFFECT` theo style_hint hiện tại để luôn có 1 hiệu
                        # ứng thị giác mặc định hợp lý, "shadow" làm fallback cuối an toàn.)
                        eff = STYLE_HINT_DEFAULT_EFFECT.get(kwargs.get("style_hint", ""), "shadow")
                eff_suffix = f" effect-{eff.replace('_', '-')}" if eff and eff != "none" else ""

                if b.container == "none":
                    # 1. PURE TYPOGRAPHY (Chữ nổi trực tiếp, không hộp, không viền)
                    icon_part = f'<span style="display:inline-block; margin-right:8px; vertical-align:middle;">{icon_svg}</span>' if icon_svg else ""
                    cell_items.append(
                        f'<div class="block-pure-text size-{b.size}{eff_suffix}{variant_suffix}" '
                        f'style="font-size:{f_size}px; font-weight:{f_weight}; text-transform:{text_trans};">'
                        f'{icon_part}<span>{escaped_text}</span></div>'
                    )
                elif b.container == "button":
                    # 2. CTA BUTTON (Nút bấm tương tác sang trọng, chống dãn cách icon)
                    cell_items.append(
                        f'<div class="block-cta size-{b.size}{eff_suffix}{variant_suffix}" '
                        f'style="font-size:{f_size}px; font-weight:{f_weight};">'
                        f'{icon_svg}<span>{escaped_text}</span></div>'
                    )
                elif b.container == "pill":
                    # 3. PILL TAG / BADGE (Viên thuốc bo tròn)
                    cell_items.append(
                        f'<div class="block-pill size-{b.size}{variant_suffix}" style="font-size:{f_size}px;">'
                        f'{icon_svg}<span>{escaped_text}</span></div>'
                    )
                else:
                    # 4. CARD (Thẻ kính mờ / Feature card / Quote / Step list)
                    if b.step_index is not None:
                        step_num = f"{b.step_index:02d}"
                        content_html = f'<div class="step-badge">{step_num}</div><span>{escaped_text}</span>'
                    elif b.style_variant in ("quote", "quote_bubble"):
                        content_html = f'<span>{escaped_text}</span>'
                    else:
                        content_html = f'{icon_svg}<span>{escaped_text}</span>'
                    cell_items.append(
                        f'<div class="block-card size-{b.size}{variant_suffix}" style="font-size:{f_size}px;">'
                        f'{content_html}</div>'
                    )

            zone_cells[z_name] = "".join(cell_items)

        # Ràng buộc CSS an toàn theo safe-zone rect (px) cho từng ô có nội dung.
        # (2026-09-12: logic thu hẹp theo zone lân cận được rút ra thành
        # `geometry.estimate_zone_available_box()` -- nguồn chân lý DUY NHẤT dùng chung
        # với `compute_block_metrics()`'s offline-mask fallback path, xem mask.py.)
        zone_max_w: Dict[str, int] = {}
        zone_max_h: Dict[str, int] = {}
        resolved_grid_areas = dict(ZONE_GRID_AREA)
        occupied_zones = set(zone_cells.keys())

        for z_name in zone_cells:
            zx1, zy1, zx2, zy2 = estimate_zone_available_box(
                z_name, occupied_zones, width, height,
                has_top_bar=bool(top_bar_html), has_bottom_bar=bool(bottom_bar_html),
            )
            zone_max_w[z_name] = max(1, zx2 - zx1)
            zone_max_h[z_name] = max(1, zy2 - zy1)

            # Zone chiếm trọn 1 hàng khi không có zone lân cận nào cùng hàng -- ảnh hưởng
            # tới CSS grid-area (span rộng ra), tách biệt khỏi max-width/max-height nên
            # vẫn cần xử lý riêng ở đây.
            if z_name == "top_center" and "top_left" not in occupied_zones and "top_right" not in occupied_zones:
                resolved_grid_areas["top_center"] = "1 / 1 / 2 / 4"
            elif z_name == "middle_center" and "middle_left" not in occupied_zones and "middle_right" not in occupied_zones:
                resolved_grid_areas["middle_center"] = "2 / 1 / 3 / 4"

        template = _JINJA_ENV.get_template("master.html")
        return template.render(
            width=width,
            height=height,
            bg_data_uri=bg_data_uri,
            font_face_css=font_face_css,
            headline_font_css=headline_font_css,
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
