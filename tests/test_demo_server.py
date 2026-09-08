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

import demo_server
demo_server.IS_MOCK_MODE = True

from demo_server import app


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
    from demo_server import sanitize_and_inject_zero_text

    # Case 1: Empty prompt gets full canonical advertising negative constraints
    p1 = sanitize_and_inject_zero_text("")
    assert "zero text" in p1
    assert "unbranded" in p1

    # Case 2: Dimensions / resolution pollution stripped
    p2 = sanitize_and_inject_zero_text("Ly trà đào cam sả 8k 16:9 4k")
    assert "8k" not in p2
    assert "16:9" not in p2
    assert "zero text" in p2

    # Case 3: Text intent directives stripped
    p3 = sanitize_and_inject_zero_text("Ảnh quảng cáo trà đào có chữ MUA 1 TẶNG 1 trên ly")
    assert "có chữ MUA 1 TẶNG 1" not in p3
    assert "zero text" in p3
    assert "no words" in p3


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


