"""Style pack + brand kit + hoạ tiết (GĐ 8, ROADMAP §10.5-10.6): pack ghi đè font/màu/hiệu ứng, chỉ điền linh kiện
LLM bỏ trống, brand kit ghi đè pack; hoạ tiết TẤT ĐỊNH và KHÔNG BAO GIỜ chạm chữ / khối đồ hoạ."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tendoo_v3.renderer import build_template_html
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.style_packs import STYLE_PACKS, apply_style_pack
from tendoo_v3.validators import check_plan

SUITE = json.loads((Path(__file__).resolve().parent / "test_style_pack_suite.json").read_text(encoding="utf-8"))


def _plan(**kw):
    return TendooCreativePlan(template="sandwich_top_heavy", hero="GIẢM 30% TOÀN BỘ MENU",
                              style=StyleConfig(font="bevietnam", theme_color="#64748B", text_effect="neon"), **kw)


def test_pack_overrides_style_and_fills_empty_components():
    p = apply_style_pack(_plan(style_pack="fnb_sale", visual_intent="big_number_deal", badge_style="pill"))
    assert (p.style.font, p.style.theme_color) == ("anton", STYLE_PACKS["fnb_sale"]["style"]["theme_color"])
    assert p.badge_style == "pill"  # LLM đã chọn -> giữ
    assert p.stat_style == "unit"  # LLM bỏ trống -> pack điền


def test_brand_kit_overrides_pack():
    p = apply_style_pack(_plan(style_pack="tet", brand_color="#0057B8", brand_font="oswald"))
    assert (p.style.theme_color, p.style.font) == ("#0057B8", "oswald")


def test_gate2_checks_pack():
    assert any("style_pack 'khong_co'" in i for i in check_plan(_plan(style_pack="khong_co")))
    assert any("không hợp intent" in i for i in check_plan(_plan(style_pack="recruit", visual_intent="big_number_deal")))


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser.new_page()
        browser.close()


BG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
OVERLAP_JS = """() => {
  const orn = Array.from(document.querySelectorAll('.tk-ornaments > *')).map(e => e.getBoundingClientRect());
  const boxes = [];
  for (const el of document.querySelectorAll('.badge-pill, .badge-capsule, .kicker-tag, .notice-label, .cta-btn, .store-item, .extra-pill, .notice-facts'))
    boxes.push(el.getBoundingClientRect());
  for (const el of document.querySelectorAll('[data-autofit]')) {
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    while (w.nextNode()) { if (!w.currentNode.textContent.trim()) continue;
      const rg = document.createRange(); rg.selectNodeContents(w.currentNode);
      for (const b of rg.getClientRects()) if (b.width > 1) boxes.push(b); }
  }
  const hit = orn.filter(o => boxes.some(b => o.right > b.left + 1 && o.left < b.right - 1 && o.bottom > b.top + 1 && o.top < b.bottom - 1));
  return [orn.length, hit.length, document.querySelector('.tk-ornaments') ? document.querySelector('.tk-ornaments').innerHTML.length : 0];
}"""


@pytest.mark.parametrize("case", SUITE, ids=[c["id"] for c in SUITE])
def test_ornaments_never_touch_text_and_are_deterministic(page, case):
    plan = TendooCreativePlan.from_dict({**case["plan"]})
    w, h = case["width"], case["height"]
    runs = []
    for _ in range(2):
        page.set_viewport_size({"width": w, "height": h})
        page.set_content(build_template_html(plan, BG, w, h), wait_until="load")
        page.wait_for_function("window.__tendooAutofitDone === true", timeout=8000)
        runs.append(page.evaluate(OVERLAP_JS))
    n, hit, size = runs[0]
    assert hit == 0, f"{case['id']}: {hit}/{n} hoạ tiết chạm chữ/khối"
    assert runs[0] == runs[1], f"{case['id']}: hoạ tiết không tất định {runs}"


def test_pack_script_font_is_a_script_font_and_reaches_lockup():
    from tendoo_v3.fonts import SCRIPT_FONTS, script_font
    from tendoo_v3.style_packs import pack_script_font
    for name, pack in STYLE_PACKS.items():
        assert pack.get("script_font") in SCRIPT_FONTS | {None}, name
    extra, family = script_font("cormorant", pack_script_font(_plan(style_pack="luxury")))
    assert "Great Vibes" in family and "Great Vibes" in extra
    assert script_font("lobster", "greatvibes")[0] == ""  # tiêu đề đã viết tay -> dùng chính nó, không nhúng thêm
