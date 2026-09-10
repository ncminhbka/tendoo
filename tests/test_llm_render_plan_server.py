"""
tests/test_llm_render_plan_server.py

Phase C v2: offline, GPU-free tests for src/tendoo/llm_render_plan_server.py. Everything
here exercises pure functions (JSON parsing, schema validation, prompt construction,
the FastAPI endpoint with the actual model call monkeypatched) -- real generation on a
live Qwen3-4B-FP8 model needs the user's own GPU server (see project memory:
tendoo-3phase-plan-2026-09-09.md) and is explicitly out of scope here.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tendoo import llm_render_plan_server as srv

VALID_PLAN = {
    "title": {"action": "none", "text": None},
    "extra_blocks": [
        {"field": None, "text": "MUA 2 TẶNG 1", "zone": "top_left", "role": "badge", "icon": "tag"},
        {"field": None, "text": "0334 842 155", "zone": "bottom_left", "role": "caption", "icon": "phone"},
    ],
    "scene_prompt": "commercial beverage advertisement photo, clean studio lighting, unbranded",
    "style_hint": "auto",
    "font_key": "auto",
}

PROMO_FIELDS_BLANK = {"discount": "", "applied_product": "", "date_start": "", "date_end": ""}
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
# parse_render_plan_json
# ==================================================================================

def test_parse_render_plan_json_extracts_valid_json():
    raw = 'Sure, here is the plan:\n{"scene_prompt": "x"}\nDone.'
    plan, errors = srv.parse_render_plan_json(raw)
    assert plan == {"scene_prompt": "x"}
    assert errors == []


def test_parse_render_plan_json_handles_no_json():
    plan, errors = srv.parse_render_plan_json("no json here at all")
    assert plan is None
    assert errors


def test_parse_render_plan_json_handles_malformed_json():
    plan, errors = srv.parse_render_plan_json('{"scene_prompt": }')
    assert plan is None
    assert errors


def test_parse_render_plan_json_ignores_trailing_prose_with_braces():
    """Regression for the greedy-regex bug (audit PHẦN 3.1, confirmed real): the old
    `re.search(r"\\{.*\\}", ..., re.DOTALL)` matched from the FIRST '{' to the LAST '}'
    in the whole response, so trailing commentary that happens to mention a brace
    corrupted the parse. A balanced-brace scanner must stop at the real JSON object's
    own closing brace and ignore everything after it."""
    raw = '{"scene_prompt": "a clean background"} Lưu ý: hãy nhớ dùng ký hiệu {ok} khi cần.'
    plan, errors = srv.parse_render_plan_json(raw)
    assert plan == {"scene_prompt": "a clean background"}
    assert errors == []


def test_parse_render_plan_json_handles_nested_braces_in_string_values():
    """A '}' inside a quoted string value must not be mistaken for the object's closing
    brace (the balanced scanner tracks string-literal state, not just raw brace count)."""
    raw = '{"scene_prompt": "a sign reading {SALE} in the background"}'
    plan, errors = srv.parse_render_plan_json(raw)
    assert plan == {"scene_prompt": "a sign reading {SALE} in the background"}
    assert errors == []


# ==================================================================================
# validate_render_plan
# ==================================================================================

def test_validate_render_plan_accepts_fully_valid_plan():
    cleaned, errors = srv.validate_render_plan(VALID_PLAN, "promo", PROMO_FIELDS_BLANK)
    assert errors == []
    assert cleaned["title"] == {"action": "none", "text": None}
    assert len(cleaned["extra_blocks"]) == 2
    assert cleaned["font_key"] == "auto"


def test_validate_render_plan_rejects_non_dict():
    cleaned, errors = srv.validate_render_plan("not a dict", "promo", PROMO_FIELDS_BLANK)
    assert cleaned is None
    assert errors


def test_validate_render_plan_title_use_prompt():
    plan = dict(VALID_PLAN, title={"action": "use_prompt", "text": "SIÊU SALE"})
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["title"] == {"action": "use_prompt", "text": "SIÊU SALE"}
    assert errors == []


def test_validate_render_plan_title_unknown_action_defaults_to_none():
    plan = dict(VALID_PLAN, title={"action": "made_up_action", "text": "X"})
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["title"]["action"] == "none"
    assert any("unknown title.action" in e for e in errors)


def test_validate_render_plan_title_non_dict_ignored():
    plan = dict(VALID_PLAN, title="not an object")
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["title"] == {"action": "none", "text": None}
    assert any("'title' is not an object" in e for e in errors)


def test_validate_render_plan_drops_unknown_zone_but_keeps_block():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "BAD ZONE", "zone": "nonexistent_zone", "role": "body"},
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert len(cleaned["extra_blocks"]) == 1
    assert cleaned["extra_blocks"][0]["zone"] is None
    assert any("unknown zone" in e for e in errors)


def test_validate_render_plan_extra_block_without_zone_is_valid():
    """v2: zone is OPTIONAL -- a field-tagged block with no zone (blank-field
    extraction, no position requested) must survive validation intact."""
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": "discount", "text": "GIẢM 50%", "zone": None, "role": "badge"},
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert len(cleaned["extra_blocks"]) == 1
    assert cleaned["extra_blocks"][0]["field"] == "discount"
    assert cleaned["extra_blocks"][0]["zone"] is None
    assert errors == []


def test_validate_render_plan_drops_block_with_no_text():
    plan = dict(VALID_PLAN, extra_blocks=[{"field": None, "zone": "top_left", "role": "body"}])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"] == []
    assert any("no non-empty" in e for e in errors)


def test_validate_render_plan_unknown_field_for_category_treated_as_null():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": "job_position", "text": "X", "zone": None, "role": "body"},  # not a promo field
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["field"] is None
    assert any("unknown field" in e for e in errors)


def test_validate_render_plan_qr_field_always_valid():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": "qr", "text": "(ignored)", "zone": "bottom_right", "role": "body"},
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["field"] == "qr"
    assert not any("unknown field" in e for e in errors)


def test_validate_render_plan_defaults_unknown_role_to_body():
    plan = dict(VALID_PLAN, extra_blocks=[{"field": None, "text": "X", "zone": "center", "role": "made_up_role"}])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["role"] == "body"
    assert any("unknown role" in e for e in errors)


def test_validate_render_plan_drops_unknown_icon_keeps_block():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "X", "zone": "center", "role": "body", "icon": "not_a_real_icon"}
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert "icon" not in cleaned["extra_blocks"][0]
    assert any("unknown icon" in e for e in errors)


def test_validate_render_plan_drops_invalid_color():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "X", "zone": "center", "role": "body", "color": "not-a-hex-color"}
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert "color" not in cleaned["extra_blocks"][0]
    assert any("invalid color" in e for e in errors)


def test_validate_render_plan_keeps_valid_hex_color():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "X", "zone": "center", "role": "body", "color": "#FF00AA"}
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["color"] == "#FF00AA"


def test_validate_render_plan_defaults_unknown_font_key_to_auto():
    plan = dict(VALID_PLAN, font_key="not_a_real_font")
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["font_key"] == "auto"
    assert any("unknown font_key" in e for e in errors)


def test_validate_render_plan_flags_missing_scene_prompt():
    plan = dict(VALID_PLAN)
    del plan["scene_prompt"]
    cleaned, errors = srv.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["scene_prompt"] == ""
    assert any("scene_prompt" in e for e in errors)


# --- duplicate-content filter (the exact gap the user flagged) ---

def test_validate_render_plan_drops_near_duplicate_of_filled_field():
    """A field:null block that closely restates an already-filled field's real value
    must be dropped -- this is the concrete duplication case (form has date_start/
    date_end filled, prompt re-states an equivalent date range as free text)."""
    fields = dict(PROMO_FIELDS_BLANK, date_start="01/09/2026", date_end="15/09/2026")
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "Áp dụng từ 01/09/2026 đến 15/09/2026", "zone": None, "role": "caption"},
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", fields)
    assert cleaned["extra_blocks"] == []
    assert any("near-duplicate" in e for e in errors)


def test_validate_render_plan_keeps_genuinely_new_ad_hoc_block():
    """A field:null block with content that does NOT match any filled field survives --
    the duplicate filter must not be overzealous."""
    fields = dict(PROMO_FIELDS_BLANK, discount="GIẢM 50%")
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "Số lượng có hạn, nhanh tay!", "zone": "bottom_right", "role": "caption"},
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", fields)
    assert len(cleaned["extra_blocks"]) == 1
    assert not any("near-duplicate" in e for e in errors)


def test_validate_render_plan_field_tagged_block_never_filtered_as_duplicate():
    """A field-tagged block is never subject to the duplicate filter (only field:null
    blocks are) -- tagging IS the correct way to reference existing content."""
    fields = dict(PROMO_FIELDS_BLANK, discount="GIẢM 50%")
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": "discount", "text": "GIẢM 50%", "zone": "top_right", "role": "badge"},
    ])
    cleaned, errors = srv.validate_render_plan(plan, "promo", fields)
    assert len(cleaned["extra_blocks"]) == 1
    assert cleaned["extra_blocks"][0]["field"] == "discount"


# ==================================================================================
# title vs. filled-field duplicate filter (audit PHẦN 2.1, confirmed real: a
# "generate" title could restate an already-filled field's real value verbatim,
# rendering it twice -- once as the field, once as the invented "title").
# ==================================================================================

def test_validate_render_plan_drops_generated_title_duplicating_filled_field():
    fields = dict(PROMO_FIELDS_BLANK, discount="GIẢM 50%")
    plan = dict(VALID_PLAN, title={"action": "generate", "text": "GIẢM 50%"})
    cleaned, errors = srv.validate_render_plan(plan, "promo", fields)
    assert cleaned["title"] == {"action": "none", "text": None}
    assert any("near-duplicate" in e for e in errors)


def test_validate_render_plan_keeps_generated_title_not_matching_any_field():
    fields = dict(PROMO_FIELDS_BLANK, discount="GIẢM 50%")
    plan = dict(VALID_PLAN, title={"action": "generate", "text": "Ưu Đãi Cuối Tuần"})
    cleaned, errors = srv.validate_render_plan(plan, "promo", fields)
    assert cleaned["title"] == {"action": "generate", "text": "Ưu Đãi Cuối Tuần"}
    assert not any("near-duplicate" in e for e in errors)


def test_validate_render_plan_use_prompt_title_never_filtered_as_duplicate():
    """action == 'use_prompt' is an explicit user request (branch (a) of the title
    precedence rule) and must be honored verbatim even if it happens to overlap a
    filled field -- only 'generate' (the model's own invention) is second-guessed."""
    fields = dict(PROMO_FIELDS_BLANK, discount="GIẢM 50%")
    plan = dict(VALID_PLAN, title={"action": "use_prompt", "text": "GIẢM 50%"})
    cleaned, errors = srv.validate_render_plan(plan, "promo", fields)
    assert cleaned["title"] == {"action": "use_prompt", "text": "GIẢM 50%"}
    assert not any("near-duplicate" in e for e in errors)


# ==================================================================================
# is_render_plan_usable
# ==================================================================================

def test_is_render_plan_usable_true_with_scene_prompt_only():
    plan = {"title": {"action": "none", "text": None}, "extra_blocks": [], "scene_prompt": "a clean background"}
    assert srv.is_render_plan_usable(plan) is True


def test_is_render_plan_usable_true_with_title_decision_only():
    plan = {"title": {"action": "generate", "text": "SIÊU SALE"}, "extra_blocks": [], "scene_prompt": ""}
    assert srv.is_render_plan_usable(plan) is True


def test_is_render_plan_usable_true_with_extra_blocks_only():
    plan = {"title": {"action": "none", "text": None}, "extra_blocks": [{"text": "x"}], "scene_prompt": ""}
    assert srv.is_render_plan_usable(plan) is True


def test_is_render_plan_usable_false_when_nothing_present():
    plan = {"title": {"action": "none", "text": None}, "extra_blocks": [], "scene_prompt": ""}
    assert srv.is_render_plan_usable(plan) is False


def test_is_render_plan_usable_handles_none():
    assert srv.is_render_plan_usable(None) is False


# ==================================================================================
# build_messages -- prompt construction sanity
# ==================================================================================

def test_build_messages_includes_category_fields_and_prompt():
    messages = srv.build_messages(
        category="promo",
        category_fields={"discount": "GIẢM 50%", "applied_product": ""},
        title_value="",
        prompt="thêm dòng ABC ở góc trên trái",
        style_pref="nang_dong",
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "GIẢM 50%" in messages[1]["content"]
    assert "thêm dòng ABC ở góc trên trái" in messages[1]["content"]
    assert "nang_dong" in messages[1]["content"]
    # every valid zone name must be enumerated in the system prompt so the model has a
    # closed vocabulary to pick from
    for zone in ["top_left", "center", "bottom_right"]:
        assert zone in messages[0]["content"]


def test_build_messages_handles_no_category_fields():
    messages = srv.build_messages(category="guide", category_fields={}, title_value="", prompt="", style_pref="auto")
    assert "không có trường nào" in messages[1]["content"]


def test_build_messages_shows_blank_fields_explicitly():
    messages = srv.build_messages(
        category="promo", category_fields={"discount": "", "applied_product": "X"},
        title_value="", prompt="", style_pref="auto",
    )
    assert "(trống)" in messages[1]["content"]


# ==================================================================================
# generate_render_plan -- monkeypatch the actual model call
# ==================================================================================

def test_generate_render_plan_end_to_end_with_mocked_model(monkeypatch):
    monkeypatch.setattr(srv, "MODEL", object())
    monkeypatch.setattr(srv, "TOKENIZER", object())
    monkeypatch.setattr(
        srv, "generate_raw_response",
        lambda model, tokenizer, device, messages, max_new_tokens=700: (
            '{"title": {"action": "none", "text": null}, "extra_blocks": '
            '[{"field": null, "text": "OK", "zone": "top_left", "role": "body"}], '
            '"scene_prompt": "a clean studio background", "style_hint": "auto", "font_key": "auto"}'
        ),
    )
    plan, errors = srv.generate_render_plan("promo", PROMO_FIELDS_FILLED, "", "test prompt", "auto")
    assert errors == []
    assert plan["scene_prompt"] == "a clean studio background"
    assert srv.is_render_plan_usable(plan) is True


def test_generate_render_plan_without_loaded_model_returns_none():
    prev_model = srv.MODEL
    try:
        srv.MODEL = None
        plan, errors = srv.generate_render_plan("promo", {}, "", "x", "auto")
        assert plan is None
        assert "model not loaded" in errors
    finally:
        srv.MODEL = prev_model


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
            '{"title": {"action": "generate", "text": "SIÊU SALE"}, "extra_blocks": [], '
            '"scene_prompt": "a clean sunlit studio background", "style_hint": "daylight", "font_key": "auto"}'
        ),
    )
    resp = client.post("/api/render-plan", json={
        "category": "promo", "title": "", "category_fields": {}, "prompt": "x",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["usable"] is True
    assert data["plan"]["title"]["text"] == "SIÊU SALE"
    assert data["errors"] == []


def test_render_plan_endpoint_reports_unusable_on_malformed_model_output(monkeypatch, client):
    monkeypatch.setattr(
        srv, "generate_raw_response",
        lambda model, tokenizer, device, messages, max_new_tokens=700: "not json at all, model rambled",
    )
    resp = client.post("/api/render-plan", json={
        "category": "promo", "title": "", "category_fields": {}, "prompt": "x",
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
            "category": "promo", "title": "", "category_fields": {}, "prompt": "x",
        })
    assert resp.status_code == 503
