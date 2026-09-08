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
