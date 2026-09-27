"""/api/v3/retext (GĐ 10): sửa chữ / đổi kiểu chữ trên nền đã sinh, không chạy lại diffusion; từ chối thay đổi làm
đổi vùng chữ (mask nền đã sinh theo vùng đó)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from PIL import Image

import tendoo_v3.demo_server as ds
from tendoo_v3.schema import StyleConfig, TendooCreativePlan


def _run(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "OUTPUT_RUNS_DIR", tmp_path)
    vdir = tmp_path / "run_20270101_000000" / "variant_0"
    vdir.mkdir(parents=True)
    plan = TendooCreativePlan(template="sandwich_top_heavy", hero="GIẢM 30% TOÀN BỘ MENU", subhead="Áp dụng đến hết chủ nhật",
                              cta="ĐẶT NGAY", visual_intent="big_number_deal",
                              hero_parts=[{"t": "GIẢM", "role": "prefix"}, {"t": "30%", "role": "stat"}, {"t": "TOÀN BỘ MENU", "role": "suffix"}],
                              style=StyleConfig(font="anton", theme_color="#E11D48"))
    (vdir / "plan.json").write_text(json.dumps(plan.to_dict(), ensure_ascii=False), encoding="utf-8")
    Image.new("RGB", (1024, 1024), (30, 20, 20)).save(vdir / "background.png")
    return TestClient(ds.app), vdir


def test_retext_edits_text_and_returns_typography_variants(tmp_path, monkeypatch):
    client, vdir = _run(tmp_path, monkeypatch)
    r = client.post("/api/v3/retext", json={"run_id": "run_20270101_000000", "edits": {"hero": "GIẢM 40% TOÀN BỘ MENU", "cta": "MUA NGAY"},
                                            "typography_variants": 3})
    assert r.status_code == 200, r.text
    posters = r.json()["posters"]
    assert len(posters) == 3
    assert all(p["plan"]["hero"] == "GIẢM 40% TOÀN BỘ MENU" and p["plan"]["cta"] == "MUA NGAY" for p in posters)
    assert posters[0]["plan"]["hero_parts"] == []  # markup cũ không khớp tiêu đề mới -> bỏ
    assert len(list(vdir.glob("retext_*.png"))) == 3


def test_retext_rejects_zone_changes(tmp_path, monkeypatch):
    client, _ = _run(tmp_path, monkeypatch)
    r = client.post("/api/v3/retext", json={"run_id": "run_20270101_000000", "edits": {"template": "split_left"}})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "field_not_editable"
    r = client.post("/api/v3/retext", json={"run_id": "../../etc", "edits": {}})
    assert r.status_code == 400
