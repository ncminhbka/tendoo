"""Lockup GĐ 7c (ROADMAP §10.4, R4): cụm hero_parts sắp đặt như designer -- stat_stack / script_over_caps / band.
Kiểm: điều kiện áp (thiếu vai trò -> dòng ngang), Cổng 2 báo, HTML mang đúng lớp + font viết tay được nhúng,
và autofit ĐO rồi quay về dòng ngang khi lockup làm điểm neo nhỏ đi (squint không bao giờ tệ hơn)."""

from __future__ import annotations

import pytest

from tendoo_v3.components import resolve_lockup, suggest_lockup
from tendoo_v3.renderer import build_template_html
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.validators import check_plan

BG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def _plan(lockup, parts, hero, **kw):
    return TendooCreativePlan(template=kw.pop("template", "sandwich_top_heavy"), hero=hero, hero_parts=parts, lockup=lockup,
                              visual_intent=kw.pop("visual_intent", "big_number_deal"), style=StyleConfig(font=kw.pop("font", "anton")), **kw)


STACK = [{"t": "GIẢM GIÁ LÊN ĐẾN", "role": "prefix"}, {"t": "70%", "role": "stat"}]
SCRIPT = [{"t": "Tháng của Nàng", "role": "prefix"}, {"t": "NGÀN ƯU ĐÃI", "role": "stat"}]
BAND = [{"t": "TUYỂN DỤNG", "role": "stat"}, {"t": "NHÂN VIÊN KINH DOANH", "role": "suffix"}]


def test_lockup_needs_its_roles():
    assert resolve_lockup(_plan("stat_stack", STACK, "GIẢM GIÁ LÊN ĐẾN 70%")) == "stat_stack"
    assert resolve_lockup(_plan("band", STACK, "GIẢM GIÁ LÊN ĐẾN 70%")) == "none"  # không có suffix
    assert resolve_lockup(_plan("script_over_caps", BAND, "TUYỂN DỤNG NHÂN VIÊN KINH DOANH")) == "none"  # không có prefix
    long = [{"t": "Chào mừng quý khách đến với", "role": "prefix"}, {"t": "TENDOO", "role": "stat"}]
    assert resolve_lockup(_plan("script_over_caps", long, "Chào mừng quý khách đến với TENDOO")) == "none"
    assert resolve_lockup(_plan("khong_co", STACK, "GIẢM GIÁ LÊN ĐẾN 70%")) == "none"


def test_gate2_reports_unusable_lockup():
    issues = check_plan(_plan("band", STACK, "GIẢM GIÁ LÊN ĐẾN 70%"))
    assert any("lockup 'band'" in i for i in issues)
    assert not any("lockup" in i for i in check_plan(_plan("stat_stack", STACK, "GIẢM GIÁ LÊN ĐẾN 70%")))


def test_lockup_markup_and_fonts():
    html = build_template_html(_plan("script_over_caps", SCRIPT, "Tháng của Nàng NGÀN ƯU ĐÃI", visual_intent="hook_headline"), BG, 1024, 1024)
    assert "hero-phrase lockup lockup--script_over_caps" in html
    assert "font-family: 'Dancing Script'" in html  # font viết tay được nhúng
    # tiêu đề đã là font viết tay -> dòng in hoa dùng Be Vietnam Pro, không nhúng Dancing thêm
    html2 = build_template_html(_plan("script_over_caps", SCRIPT, "Tháng của Nàng NGÀN ƯU ĐÃI", font="pacifico", visual_intent="hook_headline"), BG, 1024, 1024)
    assert "font-family: 'Dancing Script'" not in html2
    stack = build_template_html(_plan("stat_stack", STACK, "GIẢM GIÁ LÊN ĐẾN 70%"), BG, 1024, 1024)
    assert "hero-phrase lockup lockup--stat_stack" in stack and 'class="stat-num"' in stack  # con số tự tách đơn vị
    flat = build_template_html(_plan("none", STACK, "GIẢM GIÁ LÊN ĐẾN 70%"), BG, 1024, 1024)
    assert "hero-phrase lockup" not in flat


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser.new_page()
        browser.close()


def test_autofit_keeps_or_drops_lockup_by_measurement(page):
    """Vùng tiêu đề cao (1:1) -> giữ lockup; vùng thấp 16:9 sandwich -> con số nhỏ đi > 10% -> quay về ngang."""
    for (w, h), expect in (((1024, 1024), "kept"), ((1024, 576), "flat")):
        plan = _plan("stat_stack", STACK, "GIẢM GIÁ LÊN ĐẾN 70%", subhead="Toàn bộ tai nghe chính hãng", badge="SIÊU SALE", cta="MUA NGAY")
        page.set_viewport_size({"width": w, "height": h})
        page.set_content(build_template_html(plan, BG, w, h), wait_until="load")
        page.wait_for_function("window.__tendooAutofitDone === true", timeout=8000)
        assert page.evaluate("document.querySelector('.hero-title').dataset.tendooLockup") == expect


def test_python_suggests_lockup_when_llm_leaves_it_empty():
    # GĐ 3R v7: LLM thật bỏ trống lockup 18/18 -> Python đề xuất; 'none' tường minh được tôn trọng.
    assert suggest_lockup(STACK) == "stat_stack"
    assert suggest_lockup(BAND) == "band"
    assert suggest_lockup(SCRIPT) == "none"  # script_over_caps cần phán đoán -- để LLM
    assert suggest_lockup([{"t": "TUYỂN DỤNG", "role": "stat"}, {"t": "KỸ SƯ AI CAO CẤP CHO DỰ ÁN MỚI", "role": "suffix"}]) == "none"
    assert resolve_lockup(_plan(None, STACK, "GIẢM GIÁ LÊN ĐẾN 70%")) == "stat_stack"
    assert resolve_lockup(_plan("none", STACK, "GIẢM GIÁ LÊN ĐẾN 70%")) == "none"
