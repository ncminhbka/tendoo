#!/usr/bin/env python3
"""
tests/test_demo_server_endpoints.py

Tests FastAPI server endpoints directly using starlette/fastapi TestClient.
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi.testclient import TestClient
from tendoo_v3.demo_server import app

client = TestClient(app)

def test_endpoints():
    print("--> Testing GET / (UI HTML)...")
    res = client.get("/")
    assert res.status_code == 200
    assert "Tendoo AI v3" in res.text
    print("    ✓ UI served successfully!")

    print("--> Testing GET /api/v3/health...")
    res = client.get("/api/v3/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "active_llm" in data
    print(f"    ✓ Health check OK! Active LLM: {data['active_llm']}, Mock: {data['mock_mode']}")

    print("--> Testing GET /api/v3/templates...")
    res = client.get("/api/v3/templates")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["templates"]) == 11
    print(f"    ✓ Templates catalog OK! Total: {len(data['templates'])} templates")

    print("--> Testing POST /api/v3/plan-only (Fast Preview)...")
    req_body = {
        "category": "promo",
        "product_name": "Trà Đào Cam Sả Tươi Mát",
        "discount": "MUA 1 TẶNG 1",
        "price": "45.000đ",
        "highlights": "Trái cây tươi 100% • Giảm 50% topping",
        "prompt": "Cốc trà đào thơm mát trên nền gỗ mộc, phong cách retro film grain, không vẽ hotline",
        "aspect_ratio": "1:1",
        "template": "auto"
    }
    res = client.post("/api/v3/plan-only", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "plan" in data
    assert "palette" in data
    assert "html" in data
    assert "bg_data_uri" in data
    
    plan = data["plan"]
    print(f"    ✓ Fast preview OK! Chosen template: {plan['template']}, VFX: {plan['style'].get('vfx')}")
    print(f"    ✓ Palette Primary: {data['palette'].get('text_primary') if data['palette'] else 'N/A'}")
    
    print("\n=======================================================")
    print("🎉 ALL DEMO SERVER API ENDPOINTS VERIFIED SUCCESSFULLY!")
    print("=======================================================\n")

if __name__ == "__main__":
    test_endpoints()
