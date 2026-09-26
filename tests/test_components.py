"""Linh kiện đồ hoạ GĐ 2 (ROADMAP §4.4): components.py + Cổng 2 cho linh kiện + A/B squint.

Nghiệm thu A/B (e2e): mỗi mẫu trong tests/components_showcase.json render CÓ và KHÔNG có linh
kiện -- linh kiện không được làm TỆ ĐI điều kiện squint nào (§4.5) và phải thật sự lên ảnh.
Chưa đòi đạt cả 4 điều kiện tuyệt đối: C1 của đa số mẫu bị khoá bởi chính sách cỡ chữ, không
phải bởi linh kiện (số đo GĐ 1, §3.4 -- quyết định ở GĐ 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tendoo_v3.catalog import COMPONENT_STYLES, INTENT_PROFILES, TEMPLATE_CATALOG
from tendoo_v3.components import (
    STAMP_FONT_MIN,
    build_components,
    enrich_hero_parts,
    find_stamp_box,
    split_stat,
    stamp_ring,
)
from tendoo_v3.geometry import get_zones
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.validators import check_plan

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _plan(**kw) -> TendooCreativePlan:
    base = dict(template="sandwich_top_heavy", hero="SIÊU SALE 50%", style=StyleConfig(font="bevietnam", theme_color="#F59E0B"),
                hero_parts=[{"t": "SIÊU SALE", "role": "prefix"}, {"t": "50%", "role": "stat", "emphasis": "accent"}])
    return TendooCreativePlan(**{**base, **kw})


def _zones(template, w=1024, h=1024):
    return {k: dict(zip(("x1", "y1", "x2", "y2"), r)) for k, r in get_zones(template, w, h).items()}


@pytest.mark.parametrize("text,expected", [
    ("50%", ("50", "%")), ("-30%", ("-30", "%")), ("99K", ("99", "K")), ("12 TRIỆU", ("12", " TRIỆU")),
    ("99.000Đ", ("99.000", "Đ")), ("10+", ("10", "+")), ("MIỄN PHÍ", None), ("TẶNG 1", None), ("50", None),
])
def test_split_stat_keeps_text_verbatim(text, expected):
    assert split_stat(text) == expected
    if expected:
        assert "".join(expected) == text  # Cổng 1: không thêm/bớt ký tự nào


def test_enrich_marks_word_units_and_never_changes_text():
    parts = [{"t": "CHỈ TỪ", "role": "prefix"}, {"t": "12 TRIỆU", "role": "stat"}, {"t": "50%", "role": "stat"}]
    out = enrich_hero_parts(parts, "unit")
    assert [p.get("unit_word") for p in out] == [None, True, False]
    assert all(p.get("num", "") + p.get("unit", "") in ("", p["t"]) for p in out)
    assert enrich_hero_parts(parts, "plain") is parts


def test_default_plan_builds_no_components():
    """Mặc định -> None -> template render y hệt trước GĐ 2 (visual_diff 379 case giống từng điểm ảnh)."""
    assert build_components(_plan(), _zones("sandwich_top_heavy"), 1024, 1024) is None
    assert build_components(_plan(badge_style="ribbon"), _zones("sandwich_top_heavy"), 1024, 1024) is None  # không có badge


@pytest.mark.parametrize("template", sorted(TEMPLATE_CATALOG))
@pytest.mark.parametrize("w,h", [(1024, 1024), (576, 1024), (1024, 576), (816, 1024)])
def test_stamp_never_covers_a_text_zone(template, w, h):
    zones = _zones(template, w, h)
    box = find_stamp_box(list((z["x1"], z["y1"], z["x2"], z["y2"]) for z in zones.values()), w, h)
    if box is None:
        return  # rơi về pill -- chấp nhận được (vd luxury_centered_card)
    assert 0 <= box[0] and box[2] <= w and 0 <= box[1] and box[3] <= h
    for z in zones.values():
        assert not (box[0] < z["x2"] and z["x1"] < box[2] and box[1] < z["y2"] and z["y1"] < box[3]), (template, w, h)


def test_stamp_falls_back_to_pill_without_free_corner():
    plan = _plan(template="luxury_centered_card", badge="ƯU ĐÃI", badge_style="stamp")
    comp = build_components(plan, _zones("luxury_centered_card"), 1024, 1024)
    assert comp["badge_style"] == "pill" and "tk-stamp" not in comp["decor_html"]


def test_stamp_ring_fills_circle_and_shrinks_long_badges():
    ring, font = stamp_ring("ƯU ĐÃI CÓ HẠN")
    assert ring.count("ƯU ĐÃI CÓ HẠN") >= 2 and font > STAMP_FONT_MIN
    assert stamp_ring("MIỄN PHÍ GIAO HÀNG TOÀN QUỐC CHO MỌI ĐƠN HÀNG TỪ 200K TRONG THÁNG CHÍN")[1] < STAMP_FONT_MIN


def test_capsule_without_separator_falls_back_to_pill():
    comp = build_components(_plan(badge="HOT DEAL", badge_style="capsule"), _zones("sandwich_top_heavy"), 1024, 1024)
    assert comp["badge_style"] == "pill"
    comp = build_components(_plan(badge="MỚI | 3 NGÀY", badge_style="capsule"), _zones("sandwich_top_heavy"), 1024, 1024)
    assert comp["badge_style"] == "capsule" and comp["capsule"] == ("MỚI", "3 NGÀY")


def test_sparkles_are_seeded_by_content():
    z = _zones("sandwich_top_heavy")
    a = build_components(_plan(decor="sparkles"), z, 1024, 1024)["decor_html"]
    assert a == build_components(_plan(decor="sparkles"), z, 1024, 1024)["decor_html"]
    assert a != build_components(_plan(decor="sparkles", hero="GIẢM 50%", hero_parts=[]), z, 1024, 1024)["decor_html"]


def test_every_component_option_names_known_intents():
    for field, options in COMPONENT_STYLES.items():
        assert field in TendooCreativePlan.__dataclass_fields__
        assert set(next(iter(options.values()))) == set(INTENT_PROFILES), f"{field}: lựa chọn mặc định phải hợp mọi intent"
        for name, intents in options.items():
            assert intents and set(intents) <= set(INTENT_PROFILES), (field, name)


@pytest.mark.parametrize("kw,needle", [
    (dict(badge="X", badge_style="rainbow"), "không có trong danh mục"),
    (dict(badge="X", badge_style="ribbon", template="customer_feedback_card"), "không hợp intent"),
    (dict(badge_style="ribbon"), "không có badge"),
    (dict(badge="HOT DEAL", badge_style="capsule"), "TRÁI | PHẢI"),
    (dict(stat_style="unit", hero_parts=[]), "role=stat"),
    (dict(stat_style="burst", hero="MUA 1 TẶNG 1", hero_parts=[{"t": "MUA 1", "role": "prefix"}, {"t": "TẶNG 1", "role": "stat"}]), "stat ngắn"),
])
def test_gate2_flags_component_misuse(kw, needle):
    issues = check_plan(_plan(**kw))
    assert any(needle in i for i in issues), issues


def test_from_dict_reads_component_fields():
    plan = TendooCreativePlan.from_dict({"hero": "SALE", "badge_style": "ribbon", "stat_style": "unit", "decor": "sparkles"})
    assert (plan.badge_style, plan.stat_style, plan.decor) == ("ribbon", "unit", "sparkles")


# ---------------------------------------------------------------------------- A/B squint (e2e)

@pytest.fixture(scope="module")
def showcase_ab():
    from playwright.sync_api import sync_playwright

    from probe_type_hierarchy import measure_case
    from render_components_showcase import load_showcase, without_components

    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            for case in load_showcase():
                tpl = case["plan"]["template"]
                on = measure_case(page, case, tpl, with_bg=True)
                dom = page.evaluate("""() => ({
                    ribbon: [...document.querySelectorAll('.badge--ribbon')].map(e => getComputedStyle(e).clipPath !== 'none'),
                    capsule: document.querySelectorAll('.badge--capsule .cap-l, .badge--capsule .cap-r').length,
                    unit: document.querySelectorAll('.stat-unit').length,
                    burst: document.querySelectorAll('.hero-seg--burst').length,
                    stamp: document.querySelectorAll('.tk-stamp textPath').length,
                    sparkles: document.querySelectorAll('.tk-sparkles path').length,
                })""")
                off = measure_case(page, without_components(case), tpl, with_bg=True)
                rows.append((case, on, off, dom))
        finally:
            browser.close()
    return rows



@pytest.mark.e2e
def test_components_never_make_squint_worse(showcase_ab):
    worse = [f"{c['id']}: {k}" for c, on, off, _ in showcase_ab
             for k in ("c1_anchor", "c2_no_wall", "c3_bg", "c4_no_loss") if off.get(k) and not on.get(k)]
    assert not worse, "Linh kiện làm tệ đi squint: " + "; ".join(worse)
    lost = [c["id"] for c, on, _, _ in showcase_ab if on["overflow"]]
    assert not lost, f"Linh kiện gây tràn (Cổng 4): {lost}"


@pytest.mark.e2e
def test_components_actually_render(showcase_ab):
    missing = []
    for case, _, _, dom in showcase_ab:
        plan = case["plan"]
        want = {
            "ribbon": plan.get("badge_style") == "ribbon", "capsule": plan.get("badge_style") == "capsule",
            "stamp": plan.get("badge_style") == "stamp", "burst": plan.get("stat_style") == "burst",
            "unit": plan.get("stat_style") in ("unit", "burst"), "sparkles": plan.get("decor") == "sparkles",
        }
        for k, needed in want.items():
            got = dom[k]
            ok = (all(got) and got) if k == "ribbon" else got > 0
            if needed and not ok:
                missing.append(f"{case['id']}: {k}")
    assert not missing, "Linh kiện yêu cầu mà không lên ảnh: " + "; ".join(missing)


def test_material_effects_emit_scaled_svg_filter():
    from tendoo_v3.styles import TEXT_EFFECT_INTENTS, TEXT_EFFECTS, get_effect_css, svg_filter_defs

    assert svg_filter_defs("3d_gold", 80) == "" and svg_filter_defs(None, 80) == ""
    for eff in ("metal_emboss", "glossy_gel"):
        assert eff in TEXT_EFFECTS and eff in TEXT_EFFECT_INTENTS
        assert "url(#tk-material)" in get_effect_css(eff, "#F59E0B", is_dark=True)
        small, big = svg_filter_defs(eff, 40), svg_filter_defs(eff, 120)
        assert 'id="tk-material"' in small and small != big  # độ nhoè co giãn theo cỡ hero


def test_gate2_flags_material_effect_on_wrong_intent():
    plan = _plan(template="customer_feedback_card", style=StyleConfig(font="bevietnam", text_effect="glossy_gel"))
    assert any("không hợp intent" in i for i in check_plan(plan))
