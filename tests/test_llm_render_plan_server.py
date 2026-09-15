"""
tests/test_llm_render_plan_server.py

Phase C v2: offline, GPU-free tests for src/tendoo/llm_render_plan_server.py -- model
loading helpers (tokenizer/checkpoint path resolution), generate_render_plan()'s
orchestration (with the actual model call monkeypatched), and the FastAPI endpoint.
Real generation on a live Qwen3-4B-FP8 model needs the user's own GPU server (see
project memory: tendoo-3phase-plan-2026-09-09.md) and is explicitly out of scope here.

Pure-function coverage (JSON parsing, schema validation, prompt construction -- none of
which need MODEL/TOKENIZER at all) lives in test_render_plan.py (2026-09-13, split out
alongside the src/ module split -- see render_plan.py's docstring).
"""

import json

import pytest
from fastapi.testclient import TestClient

from tendoo import llm_render_plan_server as srv

PROMO_FIELDS_FILLED = {"discount": "GIẢM 50%", "applied_product": "", "date_start": "", "date_end": ""}


# ==================================================================================
# _resolve_tokenizer_path -- confirmed bug found on a real server run (2026-09-10):
# a local ".../text_encoder" checkpoint dir can have valid model weights but an
# incomplete tokenizer (missing chat_template.json), crashing apply_chat_template()
# with "chat_template is not set" even though the checkpoint is otherwise fine. Mirrors
# src/flux2/text_encoder.py::Qwen3Embedder's own sibling-tokenizer-directory lookup.
# ==================================================================================

def test_resolve_tokenizer_path_uses_sibling_tokenizer_dir_when_present(tmp_path):
    root = tmp_path / "FLUX.2-klein-base-4B"
    text_encoder_dir = root / "text_encoder"
    tokenizer_dir = root / "tokenizer"
    text_encoder_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)

    result = srv._resolve_tokenizer_path(str(text_encoder_dir))
    assert result == str(tokenizer_dir)


def test_resolve_tokenizer_path_falls_back_to_model_path_without_sibling(tmp_path):
    text_encoder_dir = tmp_path / "some_checkpoint" / "text_encoder"
    text_encoder_dir.mkdir(parents=True)

    result = srv._resolve_tokenizer_path(str(text_encoder_dir))
    assert result == str(text_encoder_dir)


def test_resolve_tokenizer_path_passes_through_hub_id_unchanged():
    # A bare HF hub id (no local sibling to check) must be returned as-is.
    assert srv._resolve_tokenizer_path("Qwen/Qwen3-4B-FP8") == "Qwen/Qwen3-4B-FP8"


def test_qwen_chatml_fallback_template_renders_valid_chatml():
    from jinja2 import Template

    rendered = Template(srv.QWEN_CHATML_FALLBACK_TEMPLATE).render(
        messages=[{"role": "system", "content": "sys msg"}, {"role": "user", "content": "user msg"}],
        add_generation_prompt=True,
    )
    assert rendered == "<|im_start|>system\nsys msg<|im_end|>\n<|im_start|>user\nuser msg<|im_end|>\n<|im_start|>assistant\n"


# ==================================================================================
# generate_render_plan -- monkeypatch the actual model call
# ==================================================================================

def test_generate_render_plan_end_to_end_with_mocked_model(monkeypatch):
    monkeypatch.setattr(srv, "MODEL", object())
    monkeypatch.setattr(srv, "TOKENIZER", object())
    monkeypatch.setattr(
        srv, "generate_raw_response",
        lambda model, tokenizer, device, messages, max_new_tokens=700: (
            '{"extra_blocks": '
            '[{"field": null, "text": "OK", "zone": "top_left"}], '
            '"scene_prompt": "a clean studio background", "style_hint": "auto", "font_key": "auto"}'
        ),
    )
    plan, errors = srv.generate_render_plan("promo", PROMO_FIELDS_FILLED, "test prompt", "auto")
    assert errors == []
    assert plan["scene_prompt"] == "a clean studio background"
    assert srv.is_render_plan_usable(plan) is True


def test_generate_render_plan_without_loaded_model_returns_none():
    prev_model = srv.MODEL
    try:
        srv.MODEL = None
        plan, errors = srv.generate_render_plan("promo", {}, "x", "auto")
        assert plan is None
        assert "model not loaded" in errors
    finally:
        srv.MODEL = prev_model


# ==================================================================================
# Request/response logging (2026-09-14) -- RENDER_PLAN_LOG_PATH is redirected to a
# per-test tmp_path by tests/conftest.py's autouse redirect_render_plan_log_path
# fixture; requesting `tmp_path` here resolves to that SAME directory (pytest caches
# fixture instances per test), so we can read back exactly what got written.
# ==================================================================================

def test_generate_render_plan_writes_full_input_and_output_to_log(monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "MODEL", object())
    monkeypatch.setattr(srv, "TOKENIZER", object())
    monkeypatch.setattr(
        srv, "generate_raw_response",
        lambda model, tokenizer, device, messages, max_new_tokens=700: (
            '{"extra_blocks": [], "scene_prompt": "a clean background", "style_hint": "auto", "font_key": "auto"}'
        ),
    )
    srv.generate_render_plan("promo", PROMO_FIELDS_FILLED, "test prompt", "auto", primary_color="#FF0000")

    log_path = srv.RENDER_PLAN_LOG_PATH
    assert log_path.exists() and log_path.parent == tmp_path
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["input"] == {
        "category": "promo", "category_fields": PROMO_FIELDS_FILLED,
        "prompt": "test prompt", "style_pref": "auto", "primary_color": "#FF0000",
    }
    assert entry["output"]["raw_response"] is not None and "scene_prompt" in entry["output"]["raw_response"]
    assert entry["output"]["plan"]["scene_prompt"] == "a clean background"
    assert entry["output"]["usable"] is True
    assert entry["output"]["errors"] == []
    assert "timestamp" in entry and "latency_s" in entry


def test_generate_render_plan_logs_even_when_model_not_loaded(tmp_path):
    prev_model = srv.MODEL
    try:
        srv.MODEL = None
        srv.generate_render_plan("promo", {}, "x", "auto")
        entry = json.loads(srv.RENDER_PLAN_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()[0])
        assert entry["output"]["plan"] is None
        assert entry["output"]["errors"] == ["model not loaded"]
        assert entry["output"]["raw_response"] is None
    finally:
        srv.MODEL = prev_model


def test_generate_render_plan_logs_raw_response_even_on_parse_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "MODEL", object())
    monkeypatch.setattr(srv, "TOKENIZER", object())
    monkeypatch.setattr(
        srv, "generate_raw_response",
        lambda model, tokenizer, device, messages, max_new_tokens=700: "not json at all, model rambled",
    )
    srv.generate_render_plan("promo", {}, "x", "auto")
    entry = json.loads(srv.RENDER_PLAN_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()[0])
    assert entry["output"]["plan"] is None
    assert entry["output"]["raw_response"] == "not json at all, model rambled"
    assert any("No JSON object found" in e for e in entry["output"]["errors"])


def test_log_render_plan_io_failure_is_swallowed_not_raised(monkeypatch):
    """Best-effort logging: a real I/O failure (disk full, permissions, ...) must
    never break/slow the actual render-plan request."""
    def _raise_open(*args, **kwargs):
        raise OSError("disk full (simulated)")
    monkeypatch.setattr(srv, "open", _raise_open, raising=False)
    srv._log_render_plan_io("promo", {}, "x", "auto", None, "raw", {"scene_prompt": "x", "extra_blocks": []}, [], 0.1)


# ==================================================================================
# FastAPI endpoint
# ==================================================================================

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(srv, "MODEL", object())
    monkeypatch.setattr(srv, "TOKENIZER", object())
    with TestClient(srv.app) as c:
        yield c


def test_health_endpoint_reports_model_loaded(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_render_plan_endpoint_returns_usable_plan(client, monkeypatch):
    monkeypatch.setattr(
        srv, "generate_raw_response",
        lambda model, tokenizer, device, messages, max_new_tokens=700: (
            # No more dedicated "title" concept -- a prompt-requested headline is just
            # another extra_blocks entry (field=null, size="xlarge", container="none").
            '{"extra_blocks": [{"field": null, "text": "SIÊU SALE", "zone": null, '
            '"size": "xlarge", "container": "none"}], '
            '"scene_prompt": "a clean sunlit studio background", "style_hint": "daylight", "font_key": "auto"}'
        ),
    )
    resp = client.post("/api/render-plan", json={
        "category": "promo", "category_fields": {}, "prompt": "x",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["usable"] is True
    assert data["plan"]["extra_blocks"][0]["text"] == "SIÊU SALE"
    assert data["errors"] == []


def test_render_plan_endpoint_reports_unusable_on_malformed_model_output(monkeypatch, client):
    monkeypatch.setattr(
        srv, "generate_raw_response",
        lambda model, tokenizer, device, messages, max_new_tokens=700: "not json at all, model rambled",
    )
    resp = client.post("/api/render-plan", json={
        "category": "promo", "category_fields": {}, "prompt": "x",
    })
    assert resp.status_code == 200  # never a 500 -- the caller (demo_server.py) decides how to react
    data = resp.json()
    assert data["usable"] is False
    assert data["plan"] is None
    assert data["errors"]


def test_render_plan_endpoint_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(srv, "MODEL", None)
    monkeypatch.setattr(srv, "TOKENIZER", None)
    with TestClient(srv.app) as c:
        resp = c.post("/api/render-plan", json={
            "category": "promo", "category_fields": {}, "prompt": "x",
        })
    assert resp.status_code == 503
