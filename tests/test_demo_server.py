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
        "discount": "CHỐNG ỒN HYBRID",
        "applied_product": "Bảo hành 2 năm",
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
    assert "diagonal-gradient-shim" in html
    assert "3.290.000đ" in html


def test_generate_l_frame_mock(client):
    """Verifies end-to-end FastAPI poster generation with l_frame layout."""
    payload = {
        "category": "product_intro",
        "title": "ROBOT HÚT BỤI LAU NHÀ\nECOVACS X1 OMNI PRO",
        "product_name": "Ecovacs Deebot X1",
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





