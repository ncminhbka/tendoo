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
    assert "Tendoo" in res.text
    print("    ✓ UI served successfully!")

    print("--> Testing GET /api/health...")
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "gpus_available" in data
    print(f"    ✓ Health check OK! Status: {data['status']}, GPUs: {data['gpus_available']}")

    print("--> Testing GET /api/required-fields...")
    res = client.get("/api/required-fields")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "promo" in data["fields"]
    print(f"    ✓ Required fields OK! Categories: {list(data['fields'].keys())}")

    print("--> Testing GET /api/v3/templates...")
    res = client.get("/api/v3/templates")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["templates"]) == 11
    print(f"    ✓ Templates catalog OK! Total: {len(data['templates'])} templates")

    print("--> Testing POST /api/generate (Fast Preview / Mock)...")
    req_body = {
        "category": "promo",
        "title": "TRÀ ĐÀO CAM SẢ",
        "discount": "MUA 1 TẶNG 1",
        "applied_product": "Áp dụng toàn bộ menu",
        "store_name": "Tendoo Tea",
        "phone": "0988 123 456",
        "image_description": "Ly trà đào thơm mát trên quầy bar gỗ tự nhiên, ánh sáng studio",
        "aspect_ratio": "1:1",
        "fast_preview": True,
    }
    res = client.post("/api/generate", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "final_poster_url" in data
    assert "blended_bg_url" in data
    assert "mask_url" in data
    assert "resolved_layout" in data
    assert "llm_debug" in data
    assert "debug_url" in data["llm_debug"]
    run_id = data["run_id"]
    print(f"    ✓ Generate API OK! Resolved layout: {data['resolved_layout']}, LLM Mode: {data['llm_debug']['mode']}")

    print(f"--> Testing GET /api/v3/runs/{run_id}/llm-debug...")
    res_debug = client.get(f"/api/v3/runs/{run_id}/llm-debug")
    assert res_debug.status_code == 200
    debug_json = res_debug.json()
    assert "input" in debug_json
    assert "output" in debug_json
    assert "timestamp" in debug_json
    print(f"    ✓ Run LLM debug OK! Status: {debug_json['status']}, Mode: {debug_json['mode']}")

    print("--> Testing GET /api/v3/llm-debug/latest...")
    res_latest = client.get("/api/v3/llm-debug/latest")
    assert res_latest.status_code == 200
    latest_json = res_latest.json()
    assert "input" in latest_json
    assert "output" in latest_json
    print(f"    ✓ Latest LLM debug OK! Status: {latest_json['status']}")

    print("--> Testing POST /api/v3/plan-only...")
    res_plan = client.post("/api/v3/plan-only", json=req_body)
    assert res_plan.status_code == 200
    plan_data = res_plan.json()
    assert "llm_debug" in plan_data
    assert "html" in plan_data
    print("    ✓ Plan-only API OK with LLM debug!")

    print("--> Testing POST /api/generate (Pre-flight 422 validation on missing required field)...")
    req_missing = {
        "category": "promo",
        "title": "TRÀ ĐÀO CAM SẢ",
        "discount": "",  # missing required field
        "fast_preview": True,
    }
    res = client.post("/api/generate", json=req_missing)
    assert res.status_code == 422
    err = res.json()
    assert err["detail"]["error"] == "missing_required_fields"
    assert "discount" in err["detail"]["fields"]
    print(f"    ✓ Pre-flight 422 validation OK! Missing fields caught: {err['detail']['fields']}")

    print("\n=======================================================")
    print("🎉 ALL DEMO SERVER API ENDPOINTS VERIFIED SUCCESSFULLY!")
    print("=======================================================\n")

if __name__ == "__main__":
    test_endpoints()
