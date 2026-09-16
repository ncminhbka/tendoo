"""
tests/test_render_plan.py

Offline, GPU-free tests for src/tendoo/render_plan.py (extracted 2026-09-13 from
llm_render_plan_server.py -- pure functions: JSON parsing, schema validation, prompt
construction. Zero GPU/model/FastAPI dependency, no monkeypatching needed at all).
Model-loading + FastAPI endpoint tests live in test_llm_render_plan_server.py.
"""

from tendoo import render_plan

VALID_PLAN = {
    "extra_blocks": [
        {"field": None, "text": "MUA 2 TẶNG 1", "zone": "top_left", "icon": "tag"},
        {"field": None, "text": "0334 842 155", "zone": "bottom_left", "icon": "phone"},
    ],
    "scene_prompt": "commercial beverage advertisement photo, clean studio lighting, unbranded",
    "style_hint": "auto",
    "font_key": "auto",
}

PROMO_FIELDS_BLANK = {"discount": "", "applied_product": "", "date_start": "", "date_end": ""}


# ==================================================================================
# parse_render_plan_json
# ==================================================================================

def test_parse_render_plan_json_extracts_valid_json():
    raw = 'Sure, here is the plan:\n{"scene_prompt": "x"}\nDone.'
    plan, errors = render_plan.parse_render_plan_json(raw)
    assert plan == {"scene_prompt": "x"}
    assert errors == []


def test_parse_render_plan_json_handles_no_json():
    plan, errors = render_plan.parse_render_plan_json("no json here at all")
    assert plan is None
    assert errors


def test_parse_render_plan_json_handles_malformed_json():
    plan, errors = render_plan.parse_render_plan_json('{"scene_prompt": }')
    assert plan is None
    assert errors


def test_parse_render_plan_json_ignores_trailing_prose_with_braces():
    """Regression for the greedy-regex bug (audit PHẦN 3.1, confirmed real): the old
    `re.search(r"\\{.*\\}", ..., re.DOTALL)` matched from the FIRST '{' to the LAST '}'
    in the whole response, so trailing commentary that happens to mention a brace
    corrupted the parse. A balanced-brace scanner must stop at the real JSON object's
    own closing brace and ignore everything after it."""
    raw = '{"scene_prompt": "a clean background"} Lưu ý: hãy nhớ dùng ký hiệu {ok} khi cần.'
    plan, errors = render_plan.parse_render_plan_json(raw)
    assert plan == {"scene_prompt": "a clean background"}
    assert errors == []


def test_parse_render_plan_json_handles_nested_braces_in_string_values():
    """A '}' inside a quoted string value must not be mistaken for the object's closing
    brace (the balanced scanner tracks string-literal state, not just raw brace count)."""
    raw = '{"scene_prompt": "a sign reading {SALE} in the background"}'
    plan, errors = render_plan.parse_render_plan_json(raw)
    assert plan == {"scene_prompt": "a sign reading {SALE} in the background"}
    assert errors == []


# ==================================================================================
# validate_render_plan
# ==================================================================================

def test_validate_render_plan_accepts_fully_valid_plan():
    cleaned, errors = render_plan.validate_render_plan(VALID_PLAN, "promo", PROMO_FIELDS_BLANK)
    assert errors == []
    assert "title" not in cleaned  # 2026-09-13: no more dedicated title concept
    assert len(cleaned["extra_blocks"]) == 2
    assert cleaned["font_key"] == "auto"


def test_validate_render_plan_rejects_non_dict():
    cleaned, errors = render_plan.validate_render_plan("not a dict", "promo", PROMO_FIELDS_BLANK)
    assert cleaned is None
    assert errors


def test_validate_render_plan_drops_unknown_zone_but_keeps_block():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "BAD ZONE", "zone": "nonexistent_zone"},
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert len(cleaned["extra_blocks"]) == 1
    assert cleaned["extra_blocks"][0]["zone"] is None
    assert any("unknown zone" in e for e in errors)


def test_validate_render_plan_extra_block_without_zone_is_valid():
    """v2: zone is OPTIONAL -- a field-tagged block with no zone (blank-field
    extraction, no position requested) must survive validation intact."""
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": "discount", "text": "GIẢM 50%", "zone": None},
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert len(cleaned["extra_blocks"]) == 1
    assert cleaned["extra_blocks"][0]["field"] == "discount"
    assert cleaned["extra_blocks"][0]["zone"] is None
    assert errors == []


def test_validate_render_plan_drops_block_with_no_text():
    plan = dict(VALID_PLAN, extra_blocks=[{"field": None, "zone": "top_left"}])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"] == []
    assert any("no non-empty" in e for e in errors)


def test_validate_render_plan_unknown_field_for_category_treated_as_null():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": "job_position", "text": "X", "zone": None},  # not a promo field
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["field"] is None
    assert any("unknown field" in e for e in errors)


def test_validate_render_plan_qr_field_always_valid():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": "qr", "text": "(ignored)", "zone": "bottom_right"},
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["field"] == "qr"
    assert not any("unknown field" in e for e in errors)


def test_validate_render_plan_defaults_unknown_size_to_medium():
    """The current schema has no "role" concept at all (extra_blocks[] carry
    field/text/zone/size/container/effect/icon/color -- see engine/blocks.py's
    AdaptiveBlock) -- this used to assert an unknown 'role' defaults to 'body', which
    validate_render_plan has never produced. 'size' is the closest real analog: an
    unrecognized value silently defaults to 'medium' with a recorded error."""
    plan = dict(VALID_PLAN, extra_blocks=[{"field": None, "text": "X", "zone": "center", "size": "made_up_size"}])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["size"] == "medium"
    assert any("unknown size" in e for e in errors)


def test_validate_render_plan_drops_unknown_icon_keeps_block():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "X", "zone": "center", "icon": "not_a_real_icon"}
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert "icon" not in cleaned["extra_blocks"][0]
    assert any("unknown icon" in e for e in errors)


def test_validate_render_plan_downgrades_long_text_in_button_to_card():
    """(2026-09-14) A real-GPU capability probe (scripts/verify_render_plan_llm.py)
    found the live Qwen3-4B-FP8 model ignores the "keep button/pill text short"
    instruction in SYSTEM_PROMPT_TEMPLATE in a large fraction of real cases (17/28) --
    container="button"/"pill" render as a single-line CSS chip that never wraps, so
    long text there either gets crushed to a near-invisible font or CSS-ellipsis-
    truncated (the exact defect found and fixed earlier the same session). Prompt
    wording alone can't be trusted -- this safety net downgrades to container="card"
    (the one container that actually wraps) whenever the model doesn't comply."""
    long_text = "Giảm ngay 20% cho khách hàng đặt lịch dọn dẹp tổng vệ sinh hôm nay"
    assert len(long_text.split()) > 5
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": long_text, "zone": None, "size": "medium", "container": "pill"}
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["container"] == "card"
    assert any("never wraps" in e for e in errors)


def test_validate_render_plan_keeps_short_text_in_button_unchanged():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "Nhanh tay!", "zone": None, "size": "medium", "container": "button"}
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["container"] == "button"
    assert not any("never wraps" in e for e in errors)


def test_validate_render_plan_downgrades_long_text_in_xlarge_to_medium_card():
    """Mirrors the adversarial 'hero_must_not_giant_ify_long_quote' capability-probe
    case, which the live model actually failed (assigned a 29-word quote to
    size='xlarge') -- the whole point of this safety net."""
    long_quote = (
        "Tôi đã dùng rất nhiều liệu trình khác nhau nhưng đây là nơi mang lại kết quả "
        "rõ rệt nhất, đội ngũ tư vấn tận tâm và chuyên nghiệp"
    )
    assert len(long_quote.split()) > 6
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": long_quote, "zone": None, "size": "xlarge", "container": "none"}
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "feedback", PROMO_FIELDS_BLANK)
    block = cleaned["extra_blocks"][0]
    assert block["size"] == "medium"
    assert block["container"] == "card"
    assert any("short hero phrase" in e for e in errors)


def test_validate_render_plan_keeps_short_text_in_xlarge_unchanged():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "SIÊU SALE", "zone": None, "size": "xlarge", "container": "none"}
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    block = cleaned["extra_blocks"][0]
    assert block["size"] == "xlarge"
    assert block["container"] == "none"
    assert not any("short hero phrase" in e for e in errors)


def test_validate_render_plan_long_xlarge_with_explicit_container_keeps_that_container():
    """If the model already picked a non-'none' container for an overly long
    xlarge/large block, respect that choice (it already opted into something other
    than pure hero text) rather than overriding it to 'card' -- only 'none' (the
    conventional xlarge/large pairing) gets upgraded to 'card' automatically."""
    long_text = "Một câu khá dài được gán nhầm cho kích thước xlarge không hợp lý chút nào"
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": long_text, "zone": None, "size": "xlarge", "container": "button"}
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    block = cleaned["extra_blocks"][0]
    assert block["size"] == "medium"
    # container="button" is itself downgraded to "card" by the button/pill length
    # check (independent of the size check) since it's also too long for a chip.
    assert block["container"] == "card"


def test_validate_render_plan_drops_invalid_color():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "X", "zone": "center", "color": "not-a-hex-color"}
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert "color" not in cleaned["extra_blocks"][0]
    assert any("invalid color" in e for e in errors)


def test_validate_render_plan_keeps_valid_hex_color():
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "X", "zone": "center", "color": "#FF00AA"}
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["extra_blocks"][0]["color"] == "#FF00AA"


def test_validate_render_plan_defaults_unknown_font_key_to_auto():
    plan = dict(VALID_PLAN, font_key="not_a_real_font")
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["font_key"] == "auto"
    assert any("unknown font_key" in e for e in errors)


def test_validate_render_plan_flags_missing_scene_prompt():
    plan = dict(VALID_PLAN)
    del plan["scene_prompt"]
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", PROMO_FIELDS_BLANK)
    assert cleaned["scene_prompt"] == ""
    assert any("scene_prompt" in e for e in errors)


# --- duplicate-content filter (the exact gap the user flagged) ---

def test_validate_render_plan_drops_near_duplicate_of_filled_field():
    """A field:null block that closely restates an already-filled field's real value
    must be dropped -- this is the concrete duplication case (form has date_start/
    date_end filled, prompt re-states an equivalent date range as free text)."""
    fields = dict(PROMO_FIELDS_BLANK, date_start="01/09/2026", date_end="15/09/2026")
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "Áp dụng từ 01/09/2026 đến 15/09/2026", "zone": None},
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", fields)
    assert cleaned["extra_blocks"] == []
    assert any("near-duplicate" in e for e in errors)


def test_validate_render_plan_keeps_genuinely_new_ad_hoc_block():
    """A field:null block with content that does NOT match any filled field survives --
    the duplicate filter must not be overzealous."""
    fields = dict(PROMO_FIELDS_BLANK, discount="GIẢM 50%")
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": None, "text": "Số lượng có hạn, nhanh tay!", "zone": "bottom_right"},
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", fields)
    assert len(cleaned["extra_blocks"]) == 1
    assert not any("near-duplicate" in e for e in errors)


def test_validate_render_plan_field_tagged_block_never_filtered_as_duplicate():
    """A field-tagged block is never subject to the duplicate filter (only field:null
    blocks are) -- tagging IS the correct way to reference existing content."""
    fields = dict(PROMO_FIELDS_BLANK, discount="GIẢM 50%")
    plan = dict(VALID_PLAN, extra_blocks=[
        {"field": "discount", "text": "GIẢM 50%", "zone": "top_right"},
    ])
    cleaned, errors = render_plan.validate_render_plan(plan, "promo", fields)
    assert len(cleaned["extra_blocks"]) == 1
    assert cleaned["extra_blocks"][0]["field"] == "discount"


# ==================================================================================
# is_render_plan_usable
# ==================================================================================

def test_is_render_plan_usable_true_with_scene_prompt_only():
    plan = {"extra_blocks": [], "scene_prompt": "a clean background"}
    assert render_plan.is_render_plan_usable(plan) is True


def test_is_render_plan_usable_true_with_extra_blocks_only():
    plan = {"extra_blocks": [{"text": "x"}], "scene_prompt": ""}
    assert render_plan.is_render_plan_usable(plan) is True


def test_is_render_plan_usable_false_when_nothing_present():
    plan = {"extra_blocks": [], "scene_prompt": ""}
    assert render_plan.is_render_plan_usable(plan) is False


def test_is_render_plan_usable_handles_none():
    assert render_plan.is_render_plan_usable(None) is False


# ==================================================================================
# build_messages -- prompt construction sanity
# ==================================================================================

def test_build_messages_includes_category_fields_and_prompt():
    messages = render_plan.build_messages(
        category="promo",
        category_fields={"discount": "GIẢM 50%", "applied_product": ""},
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
    messages = render_plan.build_messages(category="guide", category_fields={}, prompt="", style_pref="auto")
    assert "không có trường nào" in messages[1]["content"]


def test_build_messages_shows_blank_fields_explicitly():
    messages = render_plan.build_messages(
        category="promo", category_fields={"discount": "", "applied_product": "X"},
        prompt="", style_pref="auto",
    )
    assert "(trống)" in messages[1]["content"]
