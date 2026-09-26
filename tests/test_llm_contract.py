"""Hợp đồng LLM GĐ 3 (ROADMAP §2.2-2.3) -- kiểm OFFLINE, cô lập khỏi chất lượng LLM thật:
(1) system prompt nói đủ danh mục đóng mà Cổng 2 kiểm (sinh từ code, không lệch được);
(2) một phản hồi của "LLM lý tưởng" đi qua đúng đường xử lý thật mà không mất markup/linh kiện."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tendoo_core.fonts import FONT_CATALOG
from tendoo_v3.catalog import COMPONENT_STYLES, INTENT_PROFILES, TEMPLATE_CATALOG, build_llm_catalog_prompt
from tendoo_v3.llm_planner import _finalize_plan_from_dict, extract_balanced_json
from tendoo_v3.llm_prompts import SYSTEM_PROMPT
from tendoo_v3.styles import BACKGROUND_TONES, TEXT_EFFECTS
from tendoo_v3.validators import check_plan

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_prompt_lists_every_closed_catalog():
    missing = [t for t in TEMPLATE_CATALOG if f"'{t}'" not in SYSTEM_PROMPT]
    missing += [i for i in INTENT_PROFILES if f"'{i}'" not in SYSTEM_PROMPT]
    missing += [t for t in BACKGROUND_TONES if f"'{t}'" not in SYSTEM_PROMPT]
    missing += [e for e in TEXT_EFFECTS if f"'{e}'" not in SYSTEM_PROMPT]
    missing += [f for f in FONT_CATALOG if f"'{f}'" not in SYSTEM_PROMPT]
    missing += [f"{f}.{o}" for f, opts in COMPONENT_STYLES.items() for o in opts if f"'{o}'" not in SYSTEM_PROMPT]
    assert not missing, f"Prompt thiếu lựa chọn mà Cổng 2 chấp nhận: {missing}"
    for key in ("hero_parts", "visual_intent", "badge_style", "stat_style", "decor"):
        assert f"`{key}`" in SYSTEM_PROMPT


def test_template_catalog_is_one_line_per_template():
    """§2.2: hint dài (~100 từ/template, phần lớn để LLM đoán sức chứa) không còn gửi LLM."""
    lines = build_llm_catalog_prompt().splitlines()
    assert len(lines) == len(TEMPLATE_CATALOG) + 1
    assert all(len(line) < 260 for line in lines)


def test_prompt_example_passes_gates():
    start = SYSTEM_PROMPT.index('{\n  "template": "split_right"')
    example = extract_balanced_json(SYSTEM_PROMPT[start:])
    plan = _finalize_plan_from_dict(dict(example), {})
    assert plan.hero_parts, "ví dụ trong prompt phải qua Cổng 1 nguyên văn"
    assert check_plan(plan) == []


# Phản hồi của "LLM lý tưởng": markup đúng nguyên văn, intent hợp template, linh kiện hợp intent.
IDEAL_RESPONSES = [
    {"template": "sandwich_top_heavy", "visual_intent": "big_number_deal", "hero": "SIÊU SALE 50%",
     "hero_parts": [{"t": "SIÊU SALE", "role": "prefix"}, {"t": "50%", "role": "stat", "emphasis": "accent"}],
     "subhead": "Áp dụng toàn hệ thống tới hết chủ nhật", "badge": "ƯU ĐÃI CÓ HẠN", "badge_style": "ribbon",
     "stat_style": "burst", "cta": "MUA NGAY", "scene_prompt": "Luxury sneakers on marble podium, zero text",
     "corridor_prompt": "soft bokeh", "style": {"font": "bevietnam", "theme_color": "#F59E0B", "text_effect": "metal_emboss", "background_tone": "dark_luxury"}},
    {"template": "grand_opening_banner", "visual_intent": "festive_event", "hero": "TƯNG BỪNG KHAI TRƯƠNG TENDOO COFFEE",
     "hero_parts": [{"t": "TƯNG BỪNG", "role": "prefix"}, {"t": "KHAI TRƯƠNG", "role": "stat", "emphasis": "accent"}, {"t": "TENDOO COFFEE", "role": "suffix"}],
     "subhead": "Tặng quà cho 100 khách đầu tiên", "badge": "KHAI TRƯƠNG", "badge_style": "stamp", "decor": "sparkles",
     "cta": "GHÉ NGAY", "scene_prompt": "Warm coffee shop interior, zero text", "corridor_prompt": "soft bokeh",
     "style": {"font": "holidays", "theme_color": "#E11D48", "text_effect": "3d_gold", "background_tone": "cinema_red"}},
    {"template": "split_right", "visual_intent": "product_showcase", "hero": "CHỈ TỪ 12 TRIỆU",
     "hero_parts": [{"t": "CHỈ TỪ", "role": "prefix"}, {"t": "12 TRIỆU", "role": "stat", "emphasis": "accent"}],
     "badge": "TRẢ GÓP | 0%", "badge_style": "capsule", "stat_style": "unit", "subhead": "Laptop mỏng nhẹ",
     "scene_prompt": "Slim laptop on desk, zero text", "corridor_prompt": "soft bokeh",
     "style": {"font": "bevietnam", "theme_color": "#3B82F6", "text_effect": "plain_elegant", "background_tone": "light_clean"}},
]


@pytest.mark.parametrize("resp", IDEAL_RESPONSES, ids=lambda r: r["template"])
def test_ideal_llm_response_survives_pipeline(resp):
    raw = json.dumps(resp, ensure_ascii=False)
    plan = _finalize_plan_from_dict(extract_balanced_json(raw), {})
    assert plan.hero_parts == [dict(p, emphasis=p.get("emphasis")) for p in resp["hero_parts"]]
    assert (plan.visual_intent, plan.badge_style, plan.stat_style, plan.decor) == (
        resp["visual_intent"], resp.get("badge_style"), resp.get("stat_style"), resp.get("decor"))
    assert check_plan(plan) == []


@pytest.mark.e2e
@pytest.mark.parametrize("resp", IDEAL_RESPONSES, ids=lambda r: r["template"])
def test_ideal_llm_response_renders_components(resp):
    from playwright.sync_api import sync_playwright

    from run_template_test import generate_mock_backdrop_data_uri
    from tendoo_v3.renderer import build_template_html

    plan = _finalize_plan_from_dict(extract_balanced_json(json.dumps(resp, ensure_ascii=False)), {})
    w, h = 1024, 1024
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.set_content(build_template_html(plan, generate_mock_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone), w, h), wait_until="load")
        pg.wait_for_function("window.__tendooAutofitDone === true", timeout=5000)
        dom = pg.evaluate("""() => ({stat: document.querySelectorAll('.hero-seg--stat').length,
            overflow: (window.__tendooOverflow || []).filter(o => o.verdict !== 'spill').length})""")
        b.close()
    assert dom["stat"] == 1 and dom["overflow"] == 0
