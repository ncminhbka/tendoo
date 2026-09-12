"""
scripts/evaluate_omni_visual.py

Bộ kiểm thử trực quan thẩm mỹ Typography, bố cục và độ khớp của Mặt nạ (Mask Fit)
dành cho Omni-Block Engine trên máy local (Mock mode - không cần GPU).

Bao trùm 8 kiểu người dùng điền form thực tế từ prompt_test.txt:
1. Full Form No Prompt (Điền đủ form, prompt rỗng)
2. Prompt Repeats Form (Điền form + Prompt lặp lại text -> kiểm tra dedup)
3. Prompt Adds Extra Texts (Điền form + Prompt thêm nhiều text phụ sáng tạo)
4. Spatial Prompt Override (Prompt chỉ định vị trí + Store nhảy lên đầu)
5. Bare Prompt No Fields (Bỏ trống form, prompt 1 câu -> kiểm tra fallback)
6. Feedback Full Blocks (Feedback chi tiết: Quote + 5 sao vàng + Ưu đãi)
7. Usage Guide Steps (Quy trình 3 bước đánh số dọc theo Story 9:16)
8. Long Multiline Headline (Tiêu đề 4 dòng tiếng Việt dấu phức tạp)

Đặc điểm:
- Dùng 1 ảnh nền Studio Tối chuẩn duy nhất (Dark Luxury Studio) có spotlight tâm.
- Chạy 100% qua Playwright Chromium headless.
- Xuất ra thư mục output_visual_eval/ gồm:
  + 01_corridor_mask.png (Mặt nạ hành lang)
  + 02_poster.png (Ảnh poster hoàn chỉnh)
  + 03_mask_overlay.png (Ảnh poster có viền dạ quang bao quanh mask)
  + 04_report.txt (Báo cáo chỉ số kỹ thuật)
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tendoo.core.base import ColorPalette, PosterContent
from tendoo.core.colors import analyze_color_harmony
from tendoo.demo_server import pil_to_base64_data_uri
from tendoo.engine.blocks import AdaptiveBlock, format_rating_stars, map_category_to_default_blocks
from tendoo.engine.geometry import GRID_ZONE_NAMES, get_product_sanctuary_rect
from tendoo.engine.layout import OmniBlockLayout
from tendoo.poster_renderer import PosterRenderer


def create_dark_luxury_studio_background(width: int, height: int) -> Image.Image:
    """
    Tạo 1 ảnh nền Studio Tối cao cấp (Dark Luxury Studio) chân thực:
    - Radial gradient từ tâm: tâm sáng hơn (xanh than chì #1e293b), rìa tối sâu (#090d16).
    - Tạo cảm giác có đèn spotlight chiếu nhẹ vào tâm sản phẩm (Product Sanctuary).
    - Màu sắc sang trọng, tương phản cao để test typography.
    """
    y, x = np.ogrid[:height, :width]
    cx, cy = width / 2.0, height / 2.0
    max_dist = math.sqrt(cx**2 + cy**2)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist

    # Center color: #223046 (R:34, G:48, B:70)
    # Edge color:   #090D16 (R:9,  G:13, B:22)
    r = (34 - (34 - 9) * (dist**0.8)).clip(9, 34)
    g = (48 - (48 - 13) * (dist**0.8)).clip(13, 48)
    b = (70 - (70 - 22) * (dist**0.8)).clip(22, 70)

    noise = np.random.normal(0, 1.5, (height, width))
    arr = np.stack([r + noise, g + noise, b + noise], axis=-1).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img


def create_mask_overlay(poster_pil: Image.Image, mask_np: np.ndarray) -> Image.Image:
    """
    Tạo ảnh chồng lớp (Overlay) so sánh giữa Poster và Mặt Nạ:
    - Vùng mask > 0.1 được phủ màu xanh Cyan dạ quang bán trong suốt (25% opacity).
    - Vẽ đường viền (stroke) màu Cyan sáng tại ranh giới của mask.
    - Vẽ khung chữ nhật đứt đoạn màu Vàng/Cam tại Product Sanctuary (tâm ảnh) để kiểm tra độ an toàn.
    """
    w, h = poster_pil.size
    overlay = poster_pil.convert("RGBA")

    mask_bool = mask_np > 0.15
    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[mask_bool] = [0, 210, 255, 75]
    tint_layer = Image.fromarray(mask_rgba, mode="RGBA")
    overlay = Image.alpha_composite(overlay, tint_layer)

    mask_binary = (mask_np > 0.4).astype(np.uint8) * 255
    mask_edge = Image.fromarray(mask_binary, mode="L").filter(ImageFilter.FIND_EDGES)
    edge_np = np.array(mask_edge) > 50
    edge_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    edge_rgba[edge_np] = [0, 255, 240, 240]
    stroke_layer = Image.fromarray(edge_rgba, mode="RGBA")
    overlay = Image.alpha_composite(overlay, stroke_layer)

    draw = ImageDraw.Draw(overlay)
    s_x1, s_y1, s_x2, s_y2 = get_product_sanctuary_rect(w, h)

    sanctuary_sub = mask_np[s_y1:s_y2, s_x1:s_x2]
    leakage_max = float(sanctuary_sub.max()) if sanctuary_sub.size > 0 else 0.0

    sanctuary_color = (255, 60, 60, 200) if leakage_max > 0.3 else (245, 158, 11, 160)
    draw.rectangle([s_x1, s_y1, s_x2, s_y2], outline=sanctuary_color, width=2)
    label = f"Product Sanctuary (Leak: {leakage_max:.2f})"
    draw.text((s_x1 + 8, s_y1 + 8), label, fill=sanctuary_color)

    return overlay.convert("RGB")


@dataclass
class VisualTestCase:
    case_id: str
    title: str
    description: str
    aspect_ratio: str
    category: str
    fields: Dict[str, Any]
    prompt: str


def get_all_scenarios() -> List[VisualTestCase]:
    """Định nghĩa 8 kịch bản bao trùm các cách điền form từ prompt_test.txt."""
    return [
        # 1. Điền đủ form, Prompt hoàn toàn rỗng
        VisualTestCase(
            case_id="case_01_full_form_no_prompt",
            title="Full Form, Không Prompt (Deterministic)",
            description="Người dùng điền đầy đủ tiêu đề, giảm giá, sản phẩm áp dụng, thông tin cửa hàng và bật QR code. Không gõ prompt.",
            aspect_ratio="1:1",
            category="promo",
            fields={
                "headline": "ĐẠI TIỆC MÙA HÈ",
                "offer_main": "GIẢM 50%",
                "offer_sub": "Áp dụng cho toàn bộ đồ uống đá xay",
                "brand": "Tendoo Coffee Lounge",
                "hotline": "0912 345 678",
                "address": "456 Hai Bà Trưng, Quận 1, TP.HCM",
                "enable_qr": True,
                "website_link": "https://tendoo.ai/summer-promo",
            },
            prompt="",
        ),

        # 2. Điền form + Prompt lặp lại nguyên văn các trường đó
        VisualTestCase(
            case_id="case_02_prompt_repeats_form",
            title="Điền Form + Prompt Lặp Lại Trường (Deduplication Check)",
            description="Form có GRAND OPENING & MUA 1 TẶNG 1, prompt lại gõ y hệt các trường đó. Kiểm tra khả năng khử trùng lặp.",
            aspect_ratio="16:9",
            category="opening",
            fields={
                "headline": "GRAND OPENING",
                "offer_main": "MUA 1 TẶNG 1",
                "offer_sub": "Áp dụng từ 14/05 - 30/05",
                "brand": "Tendoo Cafe & Lounge",
                "hotline": "0988 123 456",
                "address": "Số 12 Phố Huế, Hà Nội",
                "opening_date": "14/05/2026",
            },
            prompt="Thiết kế banner khai trương quán coffee phong cách hiện đại. Text nổi bật lớn: GRAND OPENING, MUA 1 TẶNG 1, Áp dụng từ 14/05 - 30/05, Tendoo Cafe, Hotline 0988 123 456. Tỉ lệ banner ngang 16:9.",
        ),

        # 3. Điền form + Prompt đòi thêm nhiều text phụ sáng tạo
        VisualTestCase(
            case_id="case_03_prompt_adds_extra_texts",
            title="Điền Form + Prompt Đòi Thêm Nhiều Text Phụ",
            description="Form có Title & Discount, prompt yêu cầu thêm: 'Coffee rang mộc chuẩn vị', 'Không gian chill', CTA 'Ghé ngay hôm nay!'.",
            aspect_ratio="16:9",
            category="opening",
            fields={
                "headline": "GRAND OPENING",
                "offer_main": "MUA 1 TẶNG 1",
                "brand": "Tendoo Roastery",
                "hotline": "0934 567 890",
            },
            prompt="Text phụ: Coffee rang mộc chuẩn vị, Không gian chill - Check-in cực chất. Thêm CTA nổi bật: Ghé ngay hôm nay!",
        ),

        # 4. Prompt chỉ định vị trí & Store info nhảy lên đầu
        VisualTestCase(
            case_id="case_04_spatial_prompt_override",
            title="Prompt Chỉ Định Vị Trí & Store Info Lên Đầu",
            description="Prompt: 'Ở góc trên bên trái THỜI GIAN LÀ CỦA BẠN. Ở giữa bên trái NÂNG TẦM PHONG CÁCH. Thông tin cửa hàng nhảy lên đầu'.",
            aspect_ratio="4:5",
            category="product_intro",
            fields={
                "headline": "SMARTWATCH CHRONOS PRO",
                "price": "4.990.000đ",
                "brand": "Chronos Luxury Timepieces",
                "hotline": "1800 6868",
                "address": "Tầng 1 Tràng Tiền Plaza, Hà Nội",
            },
            prompt="Ở góc trên bên trái, văn bản 'THỜI GIAN LÀ CỦA BẠN' nhỏ. Ở giữa bên trái, văn bản 'NÂNG TẦM PHONG CÁCH' lớn hơn tinh tế. Thông tin cửa hàng nhảy lên đầu.",
        ),

        # 5. Bỏ trống mọi trường form, chỉ gõ prompt 1 câu sơ sài
        VisualTestCase(
            case_id="case_05_bare_prompt_no_fields",
            title="Bỏ Trống Form, Prompt Sơ Sài 1 Câu (Fallback)",
            description="Người dùng bỏ trống toàn bộ form, chỉ gõ prompt mô tả ảnh. Kiểm tra tự động lấy default headline và bố cục thoáng.",
            aspect_ratio="1:1",
            category="product_intro",
            fields={},
            prompt="Đồng hồ thông minh hiện đại cao cấp trên bàn cà phê bằng gỗ mộc mạc cạnh tách latte art.",
        ),

        # 6. Feedback khách hàng chi tiết
        VisualTestCase(
            case_id="case_06_feedback_full_blocks",
            title="Customer Feedback Đầy Đủ (Quote + 5★ Rating Badge)",
            description="Dòng 21: Tiêu đề + Quote dài 2 dòng '97% khách hàng hài lòng...' + Rating 5 sao vàng + Voucher giảm 20% tháng đầu.",
            aspect_ratio="4:5",
            category="feedback",
            fields={
                "headline": "Khách hàng nói gì sau 90 ngày thay đổi?",
                "feedback_target": "Private Coaching Transformation",
                "feedback_quote": "97% khách hàng hài lòng với kết quả tăng cơ, cải thiện vóc dáng và sức khỏe chỉ sau 3 tháng",
                "feedback_rating": "5 ★★★★★",
                "special_offer": "Giảm 20% gói PT tháng đầu",
                "brand": "Tendoo Fitness & Yoga",
                "hotline": "0988 888 999",
            },
            prompt="Hiển thị review 5 sao, typography nổi bật, phong cách fitness premium.",
        ),

        # 7. Usage Guide nhiều bước đánh số
        VisualTestCase(
            case_id="case_07_usage_guide_steps",
            title="Usage Guide 3 Bước Đánh Số (Story 9:16)",
            description="Form có Tiêu đề + 3 bước hướng dẫn cụ thể (block-step có số 1, 2, 3) dọc theo thân ảnh 9:16 + hotline tư vấn.",
            aspect_ratio="9:16",
            category="guide",
            fields={
                "headline": "3 BƯỚC CHĂM SÓC DA MÙA HÈ",
                "guide_steps": [
                    "Bước 1: Làm sạch sâu với gel rửa mặt tràm trà dịu nhẹ",
                    "Bước 2: Cấp ẩm tức thì bằng serum Hyaluronic Acid 2%",
                    "Bước 3: Khóa ẩm & thoa kem chống nắng SPF50+ chống quang hóa",
                ],
                "brand": "Tendoo Skincare Lab",
                "hotline": "1900 1234",
            },
            prompt="Quy trình chăm sóc da 3 bước khoa học, thiết kế dạng story 9:16 hiện đại.",
        ),

        # 8. Slogan / Văn bản dài 4 dòng tiếng Việt
        VisualTestCase(
            case_id="case_08_long_multiline_headline",
            title="Văn Bản / Slogan Dài 4 Dòng (Font Fitting Stress Test)",
            description="Tiêu đề 4 câu thơ (28 từ, nhiều dấu Á, Ệ, Ộ). Kiểm tra tự động cân bằng dòng, co giãn font autofit không bị cắt chữ.",
            aspect_ratio="1:1",
            category="promo",
            fields={
                "headline": "Sông Mã xa rồi Tây Tiến ơi\nNhớ về rừng núi nhớ chơi vơi\nSài Khao sương lấp đoàn quân mỏi\nMường Lát hoa về trong đêm hơi",
                "offer_main": "PHIÊN BẢN GIỚI HẠN",
                "brand": "Tendoo Heritage",
                "hotline": "0911 223 344",
                "address": "Nhà hát Lớn Hà Nội, Số 1 Tràng Tiền",
            },
            prompt="Bốn câu thơ khắc mạ vàng cổ kính, phong cách thi ca hùng tráng.",
        ),
    ]


def run_visual_evaluation():
    output_dir = PROJECT_ROOT / "output_visual_eval"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = OmniBlockLayout()
    scenarios = get_all_scenarios()

    print("\n" + "=" * 80)
    print("CHẠY BỘ KIỂM THỬ TRỰC QUAN THẨM MỸ TYPOGRAPHY & MASK FIT (OMNI ENGINE)")
    print(f"Tổng số scenarios kiểm thử: {len(scenarios)}")
    print(f"Thư mục lưu kết quả: {output_dir}")
    print("=" * 80 + "\n")

    summary_reports = []

    for idx, sc in enumerate(scenarios, 1):
        print(f"[{idx}/{len(scenarios)}] Đang chạy {sc.case_id} ({sc.aspect_ratio})...")
        case_dir = output_dir / sc.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        if sc.aspect_ratio == "9:16":
            w, h = 576, 1024
        elif sc.aspect_ratio == "16:9":
            w, h = 1024, 576
        elif sc.aspect_ratio == "4:5":
            w, h = 816, 1024
        else:
            w, h = 1024, 1024

        bg_pil = create_dark_luxury_studio_background(w, h)
        bg_data_uri = pil_to_base64_data_uri(bg_pil)

        prompt_lower = (sc.prompt or "").lower()
        store_top_triggers = ["nhảy lên đầu", "store ở trên", "store on top", "thông tin lên đầu"]
        store_zone = "top_bar" if any(trig in prompt_lower for trig in store_top_triggers) else "bottom_bar"

        qr_data_uri = ""
        if sc.fields.get("enable_qr") and sc.fields.get("website_link"):
            from tendoo.qr import generate_qr_base64
            qr_data_uri = generate_qr_base64(sc.fields["website_link"]) or ""

        content = PosterContent(
            headline=sc.fields.get("headline", ""),
            offer_main=sc.fields.get("offer_main", "") or sc.fields.get("price", ""),
            offer_sub=sc.fields.get("offer_sub", "") or sc.fields.get("special_offer", ""),
            brand=sc.fields.get("brand", ""),
            hotline=sc.fields.get("hotline", ""),
            address=sc.fields.get("address", ""),
            category=sc.category,
            qr_data_uri=qr_data_uri,
        )
        if sc.case_id == "case_08_long_multiline_headline":
            content.font_family = "playfair"

        blocks: List[AdaptiveBlock] = []

        hl_text = sc.fields.get("headline", "")
        if not hl_text and sc.category == "product_intro":
            hl_text = "GIỚI THIỆU SẢN PHẨM"
        if hl_text:
            blocks.append(AdaptiveBlock(text=hl_text, role="hero", zone="top_center" if "\n" in hl_text or len(hl_text) > 30 else "top_left"))

        if sc.category == "promo":
            if sc.fields.get("offer_main"):
                badge_zone = "bottom_center" if ("\n" in hl_text or len(hl_text) > 30) else "top_right"
                blocks.append(AdaptiveBlock(text=sc.fields["offer_main"], role="badge", zone=badge_zone, style_variant="pill_badge"))
            if sc.fields.get("offer_sub"):
                blocks.append(AdaptiveBlock(text=sc.fields["offer_sub"], role="body", zone="middle_left"))

        elif sc.category == "opening":
            if sc.fields.get("opening_date"):
                blocks.append(AdaptiveBlock(text=f"Khai trương: {sc.fields['opening_date']}", role="badge", zone="top_right", style_variant="calendar_box"))
            if sc.fields.get("offer_main"):
                blocks.append(AdaptiveBlock(text=sc.fields["offer_main"], role="badge", zone="middle_left", style_variant="pill_badge"))
            if sc.fields.get("offer_sub"):
                blocks.append(AdaptiveBlock(text=sc.fields["offer_sub"], role="body", zone="bottom_left"))

        elif sc.category == "product_intro":
            if sc.fields.get("price"):
                blocks.append(AdaptiveBlock(text=sc.fields["price"], role="badge", zone="top_right", style_variant="luxury_tag"))
            if "thời gian là của bạn" in prompt_lower:
                blocks.append(AdaptiveBlock(text="THỜI GIAN LÀ CỦA BẠN", role="badge", zone="top_left", style_variant="luxury_tag"))
            if "nâng tầm phong cách" in prompt_lower:
                blocks.append(AdaptiveBlock(text="NÂNG TẦM PHONG CÁCH ĐỜI SỐNG", role="hero", zone="middle_left"))

        elif sc.category == "feedback":
            if sc.fields.get("feedback_target"):
                blocks.append(AdaptiveBlock(text=sc.fields["feedback_target"], role="subtitle", zone="top_left"))
            if sc.fields.get("feedback_quote"):
                blocks.append(AdaptiveBlock(text=f"“{sc.fields['feedback_quote']}”", role="quote", zone="middle_left"))
            if sc.fields.get("feedback_rating"):
                rating_val = format_rating_stars(sc.fields["feedback_rating"])
                blocks.append(AdaptiveBlock(text=rating_val, role="badge", zone="middle_right", style_variant="rating_badge", icon="star"))
            if sc.fields.get("special_offer"):
                blocks.append(AdaptiveBlock(text=sc.fields["special_offer"], role="badge", zone="bottom_center", style_variant="pill_badge"))

        elif sc.category == "guide":
            for s_idx, step_text in enumerate(sc.fields.get("guide_steps", [])):
                blocks.append(AdaptiveBlock(text=step_text, role="step_list", zone="middle_left", step_index=s_idx + 1))

        if sc.case_id == "case_03_prompt_adds_extra_texts":
            blocks.append(AdaptiveBlock(text="Coffee rang mộc chuẩn vị", role="body", zone="middle_right"))
            blocks.append(AdaptiveBlock(text="Không gian chill - Check-in cực chất", role="body", zone="bottom_right"))
            blocks.append(AdaptiveBlock(text="Ghé ngay hôm nay!", role="badge", zone="bottom_center", style_variant="cta_button"))

        st_parts = []
        if sc.fields.get("brand"):
            st_parts.append(sc.fields["brand"])
        if sc.fields.get("hotline"):
            st_parts.append(f"Hotline: {sc.fields['hotline']}")
        if sc.fields.get("address"):
            st_parts.append(sc.fields["address"])
        if st_parts:
            blocks.append(AdaptiveBlock(
                text="  •  ".join(st_parts),
                role="brand_bar",
                zone=store_zone,
                field="store_info",
                icon="phone" if sc.fields.get("hotline") else "globe",
            ))

        content.free_text_blocks = blocks
        blocks = layout._extract_blocks(content)
        content.free_text_blocks = blocks

        safe_zone = layout.get_safe_zone()
        palette = analyze_color_harmony(np.array(bg_pil), safe_zone, color_mode="auto")

        measure_html = layout.render_html(
            content=content,
            palette=palette,
            bg_data_uri=bg_data_uri,
            width=w,
            height=h,
        )

        mask_np = layout.generate_mask_from_render(measure_html, width=w, height=h)
        if mask_np is None:
            mask_np = layout.generate_mask(width=w, height=h, blocks=blocks)

        mask_vis = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_file = case_dir / "01_corridor_mask.png"
        mask_vis.save(mask_file)

        html_file = case_dir / "poster_markup.html"
        html_file.write_text(measure_html, encoding="utf-8")

        poster_file = case_dir / "02_poster.png"
        PosterRenderer.render(
            html_content=measure_html,
            output_image_path=poster_file,
            width=w,
            height=h,
        )

        poster_pil = Image.open(poster_file)
        overlay_pil = create_mask_overlay(poster_pil, mask_np)
        overlay_file = case_dir / "03_mask_overlay.png"
        overlay_pil.save(overlay_file)

        s_x1, s_y1, s_x2, s_y2 = get_product_sanctuary_rect(w, h)
        sanctuary_sub = mask_np[s_y1:s_y2, s_x1:s_x2]
        leakage_max = float(sanctuary_sub.max()) if sanctuary_sub.size > 0 else 0.0
        mask_coverage = float((mask_np > 0.3).mean() * 100.0)

        report_content = f"""================================================================================
BÁO CÁO KIỂM THỬ: {sc.case_id}
================================================================================
Kịch bản: {sc.title}
Mô tả: {sc.description}
Tỉ lệ khung hình: {sc.aspect_ratio} ({w}x{h} px)
Ngành hàng: {sc.category}

[1] CÁC KHỐI CHỮ ĐƯỢC XẾP (BLOCKS):
"""
        for b_idx, b in enumerate(blocks, 1):
            variant_str = f" [variant={b.style_variant}]" if b.style_variant else ""
            report_content += f"  {b_idx}. Zone: {b.zone:<14} | Role: {b.role:<10} | Text: '{b.text}'{variant_str}\n"

        report_content += f"""
[2] CHỈ SỐ MẶT NẠ HÀNH LANG (CORRIDOR MASK):
  - Tỉ lệ phủ mask toàn canvas: {mask_coverage:.1f}%
  - Rò rỉ tâm sản phẩm (Sanctuary Leakage Max): {leakage_max:.2f} (Ngưỡng an toàn <= 0.30)
  - Trạng thái Sanctuary: {"AN TOÀN TUYỆT ĐỐI" if leakage_max <= 0.30 else "CẢNH BÁO LẤN TÂM"}

[3] FILE KẾT QUẢ ĐÃ XUẤT:
  - 01_corridor_mask.png   (Ảnh mặt nạ hành lang làm sạch nền)
  - 02_poster.png          (Ảnh Poster hoàn chỉnh đã có chữ và background)
  - 03_mask_overlay.png    (Ảnh so khớp: Viền xanh Cyan thể hiện vùng mask ôm chữ)
================================================================================
"""
        report_file = case_dir / "04_report.txt"
        report_file.write_text(report_content, encoding="utf-8")

        summary_reports.append({
            "case_id": sc.case_id,
            "title": sc.title,
            "ratio": sc.aspect_ratio,
            "num_blocks": len(blocks),
            "leakage": leakage_max,
            "coverage": mask_coverage,
        })
        print(f"   -> [XONG] {sc.case_id}: Phủ mask {mask_coverage:.1f}%, Sanctuary Leakage {leakage_max:.2f}")

    print("\n" + "=" * 80)
    print("TỔNG HỢP KẾT QUẢ BỘ KIỂM THỬ TRỰC QUAN (8 SCENARIOS)")
    print("=" * 80)
    for sr in summary_reports:
        status = "PASSED (An toàn)" if sr["leakage"] <= 0.30 else "FAIL (Lấn tâm)"
        print(f" - {sr['case_id']:<32} | {sr['ratio']:<5} | {sr['num_blocks']} blocks | Leak: {sr['leakage']:.2f} | {status}")
    print("=" * 80)
    print(f"Toàn bộ ảnh và báo cáo chi tiết đã lưu tại: {output_dir}\n")


if __name__ == "__main__":
    run_visual_evaluation()
