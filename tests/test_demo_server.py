"""
tests/test_demo_server.py

Comprehensive tests for Tendoo AI Demo Server:
- Health check endpoint
- UI HTML template serving
- Unified GenerateRequest schema validation
- Automatic QR code generation and embedding
- All 4 layout topologies under mock inference
- Custom aspect ratios (1:1, 9:16, 16:9, 4:5)
"""

import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tendoo import demo_server
demo_server.IS_MOCK_MODE = True

from tendoo.demo_server import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_fonts_endpoint(client):
    response = client.get("/api/fonts")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["groups"], list) and len(data["groups"]) > 0

    all_keys = []
    for group in data["groups"]:
        assert "group_name" in group and "fonts" in group
        for font in group["fonts"]:
            assert {"key", "display_name", "css_family", "description"} <= font.keys()
            all_keys.append(font["key"])

    assert len(all_keys) == 19
    assert "bevietnam" in all_keys
    assert "anton" in all_keys


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["mock_mode"] is True


def test_serve_ui_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    # Verify presence of the 6 intent buttons
    assert "btn-cat-promo" in html
    assert "btn-cat-product_intro" in html
    assert "btn-cat-opening" in html
    assert "btn-cat-feedback" in html
    assert "btn-cat-recruitment" in html
    assert "btn-cat-guide" in html
    # Verify presence of the 3 form sections
    assert "PHẦN 1: THÔNG TIN" in html
    assert "PHẦN 2: THÔNG TIN CỬA HÀNG" in html
    assert "PHẦN 3: HIỂN THỊ & THIẾT KẾ" in html
    # Verify QR code toggle
    assert "inp-enable-qr" in html
    # Verify file upload
    assert "upload-dropzone" in html
    # Verify multi-image selector & gallery
    assert "sel-num-images" in html
    assert "gallery-container" in html


def test_generate_promo_poster_with_qr(client, tmp_path):
    payload = {
        "category": "promo",
        "title": "TRÀ ĐÀO CAM SẢ\nTHANH MÁT TỰ NHIÊN",
        "discount": "MUA 2 TẶNG 1",
        "applied_product": "Áp dụng cho toàn bộ trà trái cây",
        "date_start": "01/09/2026",
        "date_end": "15/09/2026",
        "store_name": "Tendoo Beverage",
        "phone": "1900 8888",
        "address": "123 Hoàng Hoa Thám, Ba Đình, Hà Nội",
        "website_link": "https://tendoo.ai/summer-sale",
        "enable_qr": True,
        "layout": "top_dome",
        "style_hint": "daylight",
        "aspect_ratio": "1:1",
        "image_description": "A glass of iced peach tea with citrus slices and mint leaves",
    }

    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["category"] == "promo"
    assert data["qr_url"] is not None
    assert "00_qr_code.png" in data["qr_url"]
    assert "04_final_poster.png" in data["final_poster_url"]

    # Verify files created on disk in OUTPUT_DIR
    run_folder_name = Path(data["final_poster_url"]).parent.name
    poster_path = demo_server.OUTPUT_DIR / run_folder_name / "04_final_poster.png"
    assert poster_path.exists()
    assert poster_path.stat().st_size > 0

    qr_path = demo_server.OUTPUT_DIR / run_folder_name / "00_qr_code.png"
    assert qr_path.exists()
    assert qr_path.stat().st_size > 0


def test_generate_bottom_platform_vertical_ratio(client):
    payload = {
        "category": "promo",
        "title": "SUV THẾ HỆ MỚI\nCHINH PHỤC MỌI ĐỊA HÌNH",
        "discount": "ƯU ĐÃI 100 TRIỆU",
        "applied_product": "Tặng gói bảo hiểm thân vỏ",
        "date_start": "01/09/2026",
        "date_end": "30/09/2026",
        "store_name": "Tendoo Motors",
        "phone": "1900 6868",
        "address": "Showroom Phú Mỹ Hưng, Quận 7, TP. HCM",
        "website_link": "https://tendoo.ai/suv-2026",
        "enable_qr": True,
        "layout": "bottom_platform",
        "style_hint": "cinematic_asphalt",
        "aspect_ratio": "9:16",
        "image_description": "Luxury SUV on winding twilight mountain road",
    }

    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["qr_url"] is not None

    run_folder_name = Path(data["final_poster_url"]).parent.name
    poster_path = demo_server.OUTPUT_DIR / run_folder_name / "04_final_poster.png"
    assert poster_path.exists()
    assert poster_path.stat().st_size > 0


def test_generate_with_uploaded_product_image(client):
    import base64
    import io
    from PIL import Image

    # Create dummy 64x64 PNG image
    img = Image.new("RGB", (64, 64), color=(255, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "category": "product_intro",
        "title": "TAI NGHE KHÔNG DÂY",
        "image_base64": f"data:image/png;base64,{b64}",
        "product_name": "Tai Nghe Không Dây Sonic",
        "product_desc": "Chống ồn chủ động Hybrid",
        "layout": "top_dome",
        "aspect_ratio": "1:1",
        "num_images": 1,
    }

    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True

    run_folder_name = Path(data["final_poster_url"]).parent.name
    uploaded_ref = demo_server.OUTPUT_DIR / run_folder_name / "00_uploaded_product.png"
    assert uploaded_ref.exists()
    assert uploaded_ref.stat().st_size > 0


def test_generate_multi_images(client):
    payload = {
        "category": "promo",
        "title": "KHUYẾN MẠI MÙA HÈ",
        "discount": "GIẢM 50%",
        "layout": "split_column",
        "aspect_ratio": "1:1",
        "num_images": 2,
    }

    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["num_images"] == 2
    assert "posters" in data
    assert len(data["posters"]) == 2

    run_folder_name = Path(data["final_poster_url"]).parent.name
    poster_dir = demo_server.OUTPUT_DIR / run_folder_name

    # Check both indexed posters and default poster exist
    assert (poster_dir / "04_final_poster_0.png").exists()
    assert (poster_dir / "04_final_poster_1.png").exists()
    assert (poster_dir / "04_final_poster.png").exists()
    assert (poster_dir / "02_blended_background_0.png").exists()
    assert (poster_dir / "02_blended_background_1.png").exists()


def test_generate_with_explicit_font_family(client):
    """Verifies an explicit font_family request is threaded through and embedded as @font-face in the rendered HTML."""
    payload = {
        "category": "opening",
        "title": "TENDOO BAKERY\nTƯNG BỪNG KHAI TRƯƠNG",
        "opening_date": "01/10/2026",
        "store_name": "Tendoo Bakery",
        "layout": "top_dome",
        "aspect_ratio": "1:1",
        "font_family": "cookies",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "@font-face" in html
    assert "SVN-Cookies" in html
    assert "--headline-font:" in html


def test_generate_with_auto_font_family_and_style_hint(client):
    """Verifies font_family='auto' (the default) resolves through style_hint without crashing,
    and that a resolved font is actually embedded (regression guard for style_hint not reaching
    the layout's font resolution)."""
    payload = {
        "category": "promo",
        "title": "SIÊU SALE MÙA HÈ\nGIẢM SỐC 50%",
        "discount": "GIẢM 50%",
        "layout": "diagonal_slash",
        "style_hint": "sport_speed",
        "aspect_ratio": "1:1",
        "font_family": "auto",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "@font-face" in html
    assert "--headline-font:" in html


def test_sanitize_and_inject_zero_text():
    from tendoo.demo_server import sanitize_and_inject_zero_text

    # Case 1: Empty prompt gets background copy space negative constraints (WITHOUT erasing brand/logos)
    p1 = sanitize_and_inject_zero_text("")
    assert "text-free background" in p1
    assert "no floating graphic text" in p1
    # Must NOT suppress brand logos or labels on authentic products
    assert "unbranded" not in p1
    assert "no logos" not in p1
    assert "no labels" not in p1

    # Case 2: Dimensions / resolution pollution stripped
    p2 = sanitize_and_inject_zero_text("Gói mỳ tôm Hảo Hảo 8k 16:9 4k")
    assert "8k" not in p2
    assert "16:9" not in p2
    assert "Hảo Hảo" in p2
    assert "text-free background" in p2

    # Case 3: Text intent directives stripped from background
    p3 = sanitize_and_inject_zero_text("Ảnh quảng cáo trà đào có chữ MUA 1 TẶNG 1 trên ly")
    assert "có chữ MUA 1 TẶNG 1" not in p3
    assert "text-free background" in p3

    # Case 4: Uploaded product reference protects authentic branding and packaging
    p4 = sanitize_and_inject_zero_text("Gói mỳ tôm chua cay", has_ref_image=True)
    assert "preserve authentic product packaging" in p4
    assert "original brand details" in p4
    assert "unbranded" not in p4


def test_spatial_layout_guidance():
    """Verifies that spatial layout steering tokens are injected accurately for asymmetrical & center layouts."""
    from tendoo.demo_server import inject_spatial_layout_guidance

    # 1. split_column steers product to the right half and clears the left vertical third
    sc_prompt = inject_spatial_layout_guidance("Người mẫu áo dài lụa tơ tằm", layout_name="split_column")
    assert "right half of the frame" in sc_prompt
    assert "left vertical third" in sc_prompt

    # 2. diagonal_slash steers product to lower-right diagonal and clears upper-left
    ds_prompt = inject_spatial_layout_guidance("Đôi giày sneaker chạy bộ thể thao", layout_name="diagonal_slash")
    assert "lower-right diagonal half" in ds_prompt
    assert "upper-left diagonal quadrant" in ds_prompt

    # 3. l_frame steers product to lower-right quadrant and clears top header and left column
    lf_prompt = inject_spatial_layout_guidance("Laptop gaming bàn phím cơ RGB", layout_name="l_frame")
    assert "lower-right quadrant" in lf_prompt
    assert "top horizontal header and left vertical column" in lf_prompt

    # 4. Center layouts reinforce centered composition
    td_prompt = inject_spatial_layout_guidance("Ly cà phê sữa đá", layout_name="top_dome")
    assert "centered in the lower two-thirds" in td_prompt

    bp_prompt = inject_spatial_layout_guidance("Chai nước hoa Chanel", layout_name="bottom_platform")
    assert "standing centered on the bottom platform stage" in bp_prompt

    ch_prompt = inject_spatial_layout_guidance("Hộp bánh trung thu", layout_name="center_hourglass")
    assert "center aperture" in ch_prompt

    # 5. With uploaded reference image, retains reference guidance
    sc_ref = inject_spatial_layout_guidance("Chai serum dưỡng da", layout_name="split_column", has_ref_image=True)
    assert "preserve authentic product placed on the right side" in sc_ref


def test_cross_layout_style_defense():
    """Verifies backend defense prevents spatial semantic clash between layout and incompatible styles."""
    from tendoo.demo_server import detect_scene_lighting_tone

    # 1. Incompatible style for top_dome (e.g. cinematic_asphalt) falls back to daylight
    style_top = detect_scene_lighting_tone("Ly trà đào cam sả ban ngày", user_hint="cinematic_asphalt", layout_name="top_dome")
    assert style_top == "daylight"

    # 2. Incompatible style for bottom_platform (e.g. daylight sky) falls back to asphalt/scene
    style_bottom = detect_scene_lighting_tone("Chiếc xe SUV sang trọng", user_hint="daylight", layout_name="bottom_platform")
    assert style_bottom == "cinematic_asphalt"

    # 3. Compatible style is preserved
    style_valid = detect_scene_lighting_tone("Chiếc xe SUV", user_hint="luxury_marble", layout_name="bottom_platform")
    assert style_valid == "luxury_marble"

    # 4. Smart auto detects night/neon mood
    style_auto_dark = detect_scene_lighting_tone("Quán bar đêm ánh neon rực rỡ", user_hint="auto", layout_name="top_dome")
    assert style_auto_dark == "studio_dark"



def test_zero_emoji_calendar_svg_in_html(client):
    """Verifies no raw calendar emojis exist in rendered HTML (preventing tofu boxes on headless Linux)."""
    payload = {
        "category": "promo",
        "title": "TRÀ HOA CÚC MẬT ONG",
        "discount": "MUA 1 TẶNG 1",
        "date_start": "01/09/2026",
        "date_end": "15/09/2026",
        "layout": "split_column",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html_content = (run_folder / "03_poster.html").read_text(encoding="utf-8")

    # Raw emoji calendar 📅 must NOT be present
    assert "📅" not in html_content
    # Cross-platform SVG calendar icon MUST be present
    assert "icon-calendar" in html_content
    assert "<svg" in html_content


def test_generate_product_intro_adaptive_components(client):
    """Verifies product_intro category renders price badge, spec chips, and product name across layouts."""
    payload = {
        "category": "product_intro",
        "title": "TAI NGHE WIRELESS SONIC PRO\nCHỐNG ỒN CHỦ ĐỘNG HYBRID",
        "product_name": "Sonic Pro Wireless",
        "price": "1.290.000đ",
        "product_desc": "Chất âm Hi-Res chuẩn phòng thu âm thanh vòm 360",
        "highlights": "ANC 45dB, Pin 40 Giờ, Bluetooth 5.3",
        "store_name": "Tendoo Audio Store",
        "phone": "1900 8888",
        "layout": "top_dome",
        "aspect_ratio": "1:1",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["category"] == "product_intro"

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "category-product-container" in html
    assert "Sonic Pro Wireless" in html
    assert "1.290.000đ" in html
    assert "cat-comp-price-pill" in html
    assert "ANC 45dB" in html
    assert "spec-chip" in html


def test_generate_grand_opening_adaptive_components(client):
    """Verifies opening category renders opening date chip, promo ribbon, and reservation hotline."""
    payload = {
        "category": "opening",
        "title": "TENDOO COFFEE ROASTERY\nTƯNG BỪNG KHAI TRƯƠNG",
        "opening_date": "15/10/2026",
        "opening_promo": "TẶNG 100 LY CÀ PHÊ MIỄN PHÍ",
        "booking_contact": "0988 123 456",
        "store_name": "Tendoo Coffee",
        "address": "45 Lê Lợi, Quận 1, TP. HCM",
        "layout": "split_column",
        "aspect_ratio": "4:5",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["category"] == "opening"

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "category-opening-container" in html
    assert "15/10/2026" in html
    assert "TẶNG 100 LY CÀ PHÊ MIỄN PHÍ" in html
    assert "cat-comp-opening-promo" in html
    assert "0988 123 456" in html
    assert "cat-comp-booking-chip" in html


def test_generate_customer_feedback_adaptive_components(client):
    """Verifies feedback category renders liquid glass quote card, 5-star rating, and loyalty offer."""
    payload = {
        "category": "feedback",
        "title": "TRẢI NGHIỆM THƯ GIÃN ĐẲNG CẤP\nLIỆU TRÌNH SPA TRẺ HÓA",
        "feedback_target": "Liệu trình Trẻ hóa da Chuyên sâu",
        "feedback_rating": "5.0 / 5.0",
        "feedback_quote": "Không gian cực kỳ thư thái, nhân viên tận tâm, da mình sáng mịn rõ rệt!",
        "special_offer": "VOUCHER 20% CHO KHÁCH HÀNG MỚI",
        "store_name": "Tendoo Spa & Wellness",
        "layout": "center_hourglass",
        "aspect_ratio": "1:1",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["category"] == "feedback"

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "category-feedback-container" in html
    assert "cat-comp-feedback-card" in html
    assert "icon-star" in html
    assert "5.0 / 5.0" in html
    assert "Không gian cực kỳ thư thái" in html
    assert "Liệu trình Trẻ hóa" in html
    assert "VOUCHER 20%" in html
    assert "cat-comp-special-offer" in html


def test_generate_recruitment_adaptive_components(client):
    """Verifies recruitment category renders role badge, benefits box, deadline, and apply method."""
    payload = {
        "category": "recruitment",
        "title": "GIA NHẬP ĐỘI NGŨ CÔNG NGHỆ\nCÙNG TENDOO AI CHINH PHỤC ĐỈNH CAO",
        "job_position": "SENIOR AI RESEARCH ENGINEER",
        "job_desc": "Thu nhập up to $4,000 / Thưởng hiệu suất / Làm việc Hybrid linh hoạt",
        "apply_deadline": "31/10/2026",
        "apply_method": "hr@tendoo.ai",
        "store_name": "Tendoo AI Labs",
        "layout": "bottom_platform",
        "aspect_ratio": "9:16",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["category"] == "recruitment"

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "category-recruitment-container" in html
    assert "cat-comp-job-role" in html
    assert "SENIOR AI RESEARCH ENGINEER" in html
    assert "up to $4,000" in html
    assert "31/10/2026" in html
    assert "hr@tendoo.ai" in html


def test_generate_usage_guide_adaptive_components(client):
    """Verifies guide category renders vertical stepper timeline with step numbers and descriptions."""
    payload = {
        "category": "guide",
        "title": "QUY TRÌNH MUA HÀNG TIỆN LỢI\nCHỈ VỚI 3 BƯỚC ĐƠN GIẢN",
        "guide_steps": [
            "Quét mã QR hoặc truy cập website",
            "Chọn sản phẩm và nhập mã ưu đãi",
            "Nhận hàng hỏa tốc trong 2 giờ"
        ],
        "store_name": "Tendoo Express",
        "layout": "split_column",
        "aspect_ratio": "16:9",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["category"] == "guide"

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "category-guide-container" in html
    assert "cat-comp-stepper-timeline" in html
    assert "timeline-step" in html
    assert "step-num-node" in html
    assert "Quét mã QR" in html
    assert "Nhận hàng hỏa tốc" in html


def test_generate_diagonal_slash_mock(client):
    """Verifies end-to-end FastAPI poster generation with diagonal_slash layout."""
    payload = {
        "category": "product_intro",
        "title": "GIÀY CHẠY BỘ CARBON\nSIÊU TỐC ĐỘ 2026",
        "product_name": "Tendoo Speed Elite",
        "product_desc": "Đế carbon siêu nhẹ, bật nảy tối ưu tốc độ",
        "price": "3.290.000đ",
        "highlights": "Đế Carbon, Siêu Nhẹ 150g, Bật Nảy 90%",
        "store_name": "Tendoo Athletics",
        "layout": "diagonal_slash",
        "style_hint": "sport_speed",
        "aspect_ratio": "1:1",
        "image_description": "A pair of high-performance running sneakers floating in dynamic athletic studio",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["category"] == "product_intro"

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "diagonal-content-stack" in html
    assert "headline-wrap" in html
    assert "3.290.000đ" in html


def test_generate_l_frame_mock(client):
    """Verifies end-to-end FastAPI poster generation with l_frame layout."""
    payload = {
        "category": "product_intro",
        "title": "ROBOT HÚT BỤI LAU NHÀ\nECOVACS X1 OMNI PRO",
        "product_name": "Ecovacs Deebot X1",
        "product_desc": "Robot hút bụi lau nhà cao cấp tự động",
        "price": "18.990.000đ",
        "highlights": "Lực hút 8000Pa, Giặt sấy giẻ khí nóng, Camera AI",
        "store_name": "Tendoo Smart Home",
        "layout": "l_frame",
        "style_hint": "tech_minimal",
        "aspect_ratio": "1:1",
        "image_description": "A luxury robot vacuum cleaner on polished hardwood living room floor",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["category"] == "product_intro"

    run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
    html = (run_folder / "03_poster.html").read_text(encoding="utf-8")
    assert "lframe-top-bar" in html
    assert "lframe-left-column" in html
    assert "18.990.000đ" in html


def test_all_categories_no_raw_emojis_and_valid_svg_icons(client):
    """
    Verifies that generated HTML templates across all commercial categories
    contain ZERO raw unicode emojis (which cause tofu square boxes □ on Linux)
    and use inline vector SVGs instead.
    """
    from tendoo.layouts.text_engine import EMOJI_PATTERN

    categories_payloads = [
        {
            "category": "promo",
            "title": "🔥 SIÊU SALE 50% 🔥\nĐẶT NGAY ➔",
            "discount": "GIẢM 50%",
            "date_start": "01/09/2026",
            "date_end": "15/09/2026",
            "store_name": "Tendoo Fashion",
            "phone": "📞 0988 123 456",
            "address": "📍 128 Trần Duy Hưng",
            "layout": "top_dome",
            "aspect_ratio": "1:1",
            "image_description": "Fashion clothing on display",
        },
        {
            "category": "opening",
            "title": "🎉 TƯNG BỪNG KHAI TRƯƠNG\nCƠ SỞ MỚI",
            "opening_date": "15/09/2026",
            "discount": "GIẢM 20% TOÀN BỘ MENU",
            "date_start": "15/09/2026",
            "store_name": "Tendoo Coffee",
            "phone": "0912 345 678",
            "layout": "split_column",
            "aspect_ratio": "1:1",
            "image_description": "Modern aesthetic cafe interior",
        },
        {
            "category": "feedback",
            "title": "CẢM NHẬN KHÁCH HÀNG\nTRẢI NGHIỆM ĐỈNH CAO",
            "feedback_target": "Liệu trình chăm sóc da chuyên sâu",
            "customer_name": "Nguyễn Văn A",
            "feedback_quote": "Dịch vụ tuyệt vời, sản phẩm rất tốt!",
            "rating": 5,
            "store_name": "Tendoo Spa",
            "layout": "center_hourglass",
            "aspect_ratio": "1:1",
            "image_description": "Spa wellness relaxing atmosphere",
        },
        {
            "category": "recruitment",
            "title": "TUYỂN DỤNG NHÂN TÀI\nCHUYÊN VIÊN AI",
            "job_position": "Chuyên Viên AI",
            "apply_deadline": "30/09/2026",
            "apply_method": "hr@tendoo.ai",
            "salary": "25 - 40 Triệu",
            "date_end": "30/09/2026",
            "store_name": "Tendoo Tech",
            "phone": "0988 999 888",
            "layout": "bottom_platform",
            "aspect_ratio": "1:1",
            "image_description": "Modern tech office team working",
        },
    ]

    for payload in categories_payloads:
        response = client.post("/api/generate", json=payload)
        assert response.status_code == 200, f"Failed for {payload['category']}: {response.text}"
        data = response.json()
        assert data["success"] is True

        run_folder = demo_server.OUTPUT_DIR / Path(data["final_poster_url"]).parent.name
        html_path = run_folder / "03_poster.html"
        assert html_path.exists()
        html_content = html_path.read_text(encoding="utf-8")

        # Must not contain raw emojis that trigger square boxes
        found_emojis = EMOJI_PATTERN.findall(html_content)
        assert not found_emojis, f"Found raw emojis {found_emojis} in category {payload['category']}"

        # Must contain vector SVGs
        assert "<svg" in html_content
        assert "</svg>" in html_content


def test_required_fields_endpoint(client):
    """/api/required-fields is the single source of truth the frontend fetches to
    render '*' markers -- must match yeu_cau.txt's per-category spec exactly."""
    response = client.get("/api/required-fields")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["guide_first_step"] is True
    fields = data["fields"]
    assert fields["promo"] == ["discount"]
    assert fields["product_intro"] == ["product_name", "product_desc"]
    assert fields["opening"] == ["opening_date"]
    assert fields["feedback"] == ["feedback_target", "feedback_quote"]
    assert fields["recruitment"] == ["job_position", "apply_deadline", "apply_method"]
    assert fields["guide"] == []


@pytest.mark.parametrize(
    "category,payload_extra,expected_missing",
    [
        ("promo", {}, ["discount"]),
        ("product_intro", {"product_name": "Sonic Pro"}, ["product_desc"]),
        ("product_intro", {}, ["product_name", "product_desc"]),
        ("opening", {}, ["opening_date"]),
        ("feedback", {"feedback_target": "Spa"}, ["feedback_quote"]),
        ("recruitment", {"job_position": "AI Engineer"}, ["apply_deadline", "apply_method"]),
    ],
)
def test_generate_rejects_missing_required_fields(client, category, payload_extra, expected_missing):
    """Omitting a category's required field(s) must 422 with the exact missing-field list --
    never silently fall through to a 200 (or a 500) as it did before Phase A."""
    payload = {"category": category, "title": "Tiêu đề bất kỳ", **payload_extra}
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "missing_required_fields"
    assert detail["category"] == category
    assert sorted(detail["fields"]) == sorted(expected_missing)


@pytest.mark.parametrize(
    "category,payload_extra",
    [
        ("promo", {"discount": "GIẢM 50%"}),
        ("product_intro", {"product_name": "Sonic Pro", "product_desc": "Chống ồn Hybrid"}),
        ("opening", {"opening_date": "15/10/2026"}),
        ("feedback", {"feedback_target": "Spa trẻ hóa da", "feedback_quote": "Rất hài lòng!"}),
        (
            "recruitment",
            {
                "job_position": "AI Engineer",
                "apply_deadline": "30/10/2026",
                "apply_method": "hr@tendoo.ai",
            },
        ),
    ],
)
def test_generate_succeeds_with_only_required_fields_filled(client, category, payload_extra):
    """Happy path: every other field left blank, only the category's required field(s) filled."""
    payload = {"category": category, **payload_extra}
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True


def test_generate_with_style_pref_auto_matches_layout_and_style(client):
    """Phase B: no `layout`/`style_hint`/`font_family` sent at all -- the server must
    auto-resolve them from category + style_pref (style_matcher.resolve_style_preset)
    rather than silently defaulting to top_dome/daylight as if the caller had chosen it."""
    payload = {
        "category": "promo",
        "title": "SIÊU SALE CUỐI TUẦN",
        "discount": "GIẢM 30%",
        "style_pref": "nang_dong",
        "aspect_ratio": "1:1",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    # nang_dong (dynamic/sport mood) nudges promo away from its plain top_dome default.
    assert data["resolved_layout"] == "diagonal_slash"
    assert data["resolved_style_hint"]


def test_generate_with_explicit_layout_overrides_style_pref_auto_match(client):
    """An explicit `layout` must still win over style_pref -- back-compat for direct API
    callers, per the Phase B design (auto-match only fires when layout is left unset)."""
    payload = {
        "category": "promo",
        "title": "SIÊU SALE CUỐI TUẦN",
        "discount": "GIẢM 30%",
        "style_pref": "nang_dong",
        "layout": "split_column",
        "aspect_ratio": "1:1",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["resolved_layout"] == "split_column"


def test_generate_with_primary_color_shifts_palette_hue(client):
    """A user-picked "màu chủ đạo" (primary_color) must actually reach the rendered
    poster's CSS accent tokens, not just be accepted and ignored."""
    import re

    def headline_hue(html: str):
        m = re.search(r"--headline-color:\s*hsl\((\d+),", html)
        return int(m.group(1)) if m else None

    payload_base = {
        "category": "promo",
        "title": "SIÊU SALE CUỐI TUẦN",
        "discount": "GIẢM 30%",
        "aspect_ratio": "1:1",
        "layout": "top_dome",
        "style_hint": "studio_dark",  # force a dark-background palette branch deterministically
    }

    resp_red = client.post("/api/generate", json={**payload_base, "primary_color": "#FF0000", "seed": 1})
    resp_blue = client.post("/api/generate", json={**payload_base, "primary_color": "#0000FF", "seed": 1})
    assert resp_red.status_code == 200, resp_red.text
    assert resp_blue.status_code == 200, resp_blue.text

    html_red = (demo_server.OUTPUT_DIR / Path(resp_red.json()["final_poster_url"]).parent.name / "03_poster.html").read_text(encoding="utf-8")
    html_blue = (demo_server.OUTPUT_DIR / Path(resp_blue.json()["final_poster_url"]).parent.name / "03_poster.html").read_text(encoding="utf-8")

    hue_red = headline_hue(html_red)
    hue_blue = headline_hue(html_blue)
    assert hue_red is not None and hue_blue is not None
    assert hue_red != hue_blue
    assert hue_red == 0
    assert hue_blue == 240


def test_generate_guide_requires_non_empty_first_step(client):
    """guide has no named required field -- it's guide_steps[0] that must be non-blank."""
    # No steps at all -> 422
    resp_empty = client.post("/api/generate", json={"category": "guide", "guide_steps": []})
    assert resp_empty.status_code == 422, resp_empty.text
    assert resp_empty.json()["detail"]["fields"] == ["guide_steps[0]"]

    # Step 1 present but blank/whitespace-only -> 422
    resp_blank = client.post("/api/generate", json={"category": "guide", "guide_steps": ["   "]})
    assert resp_blank.status_code == 422, resp_blank.text
    assert resp_blank.json()["detail"]["fields"] == ["guide_steps[0]"]

    # Step 1 non-empty -> 200
    resp_ok = client.post(
        "/api/generate",
        json={"category": "guide", "guide_steps": ["Quét mã QR để bắt đầu"]},
    )
    assert resp_ok.status_code == 200, resp_ok.text
    assert resp_ok.json()["success"] is True






