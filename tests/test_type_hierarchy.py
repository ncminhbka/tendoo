"""Nghiệm thu GĐ 0A+ (ROADMAP §4.1, §7): đo cỡ chữ THẬT sau autofit trên Chromium cho toàn
bộ suite JSON -- không phần tử Cấp 3 nào được to hơn subhead.

Đây là test đầu tiên trong pytest thực sự RENDER 14 template (MIGRATION §4: trước đây các
suite JSON chỉ được script trong scripts/ đọc). Baseline 26/09: 53/379 case đảo bậc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from probe_type_hierarchy import SQUINT_KEYS, load_cases, measure_case, squint_baseline, summarize, with_oracle_hero_parts  # noqa: E402

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def page():
    """Một trình duyệt cho cả module (hai `sync_playwright()` lồng nhau thì Playwright báo lỗi)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser.new_page()
        finally:
            browser.close()


@pytest.fixture(scope="module")
def measured(page):
    return [{**measure_case(page, case, tpl, with_bg=True), "suite": suite} for suite, tpl, case in load_cases(None)]


def test_all_suite_cases_measured(measured):
    assert len(measured) >= 379


# Luật 6: 3 case suite dày nhất ở 16:9 -- mọi chữ bị co đều để không tràn khung, subhead về sàn cứu 12px (§8).
KNOWN_INVERSION = {"sbh_11_16x9_heavy", "sbh_noqr_11_16x9_heavy", "sth_noqr_12_16x9_heavy"}


def test_no_tier3_larger_than_subhead(measured):
    bad = [f"{r['id']}: subhead {r['subhead']} < {r['t3_max_cls']} {r['t3_max']}" for r in measured if r["inversion"] and r["id"] not in KNOWN_INVERSION]
    assert not bad, "Cấp 3 to hơn Cấp 2:\n" + "\n".join(bad)


def test_every_autofit_class_has_a_tier(measured):
    unknown = sorted({c for r in measured for c in r["unknown"]})
    assert not unknown, f"Class autofit chưa phân cấp trong styles.py::TIER*_CLASSES: {unknown}"


def _by_id(measured, case_id, suite):
    return next(r for r in measured if r["id"] == case_id and r["suite"] == suite)


# Cổng 4 (GĐ 0C). Phân loại dưới đây đã đối chiếu bằng MẮT trên ảnh render ngày 26/09:
# sbh_11 cắt mất nửa dòng email ở dải đỉnh; lframe_06 (và ba_02 trước GĐ 3) vượt ngân sách nhưng hiển thị đủ.
def test_gate4_classifies_verified_cases(measured):
    """Cổng 4 phân loại đúng: 2 case vượt ngân sách nhưng hiển thị đủ (đối chiếu bằng mắt) vẫn là spill."""
    assert _by_id(measured, "lframe_06_9x16_light_left", "test_l_frame_showcase_suite.json")["overflow"] == ["store-details-row:spill"]
    ba = _by_id(measured, "ba_02_9x16_left_dental", "test_before_after_split_suite.json")["overflow"]
    assert not any(o.endswith((":clipped", ":overlap")) for o in ba), ba


# Mất chữ ĐÃ BIẾT, chưa sửa (ROADMAP §8). Case mới xuất hiện -> hồi quy thật. Case biến mất
# khỏi đây -> đã sửa được, cập nhật danh sách.
# Luật 6: suite viết tay dày nhất (chủ yếu 16:9) -- còn mất chữ dù đã cứu tới 12px + co cả khối (§8).
# Nội dung cỡ này phải được Cổng 3 đổi template / LLM viết ngắn; poster LLM thật: 0 mất chữ.
# 27/09: lframe_10 hết mất chữ (co chữ phụ trước hero ở Bước 3d); sbh_noqr_15 (4:5, nội dung dày nhất) vào
# danh sách -- mọi chữ đã về sàn cứu 12.5px mà dòng cửa hàng vẫn không vừa.
KNOWN_TEXT_LOSS = {"lframe_12_16x9_heavy_right", "master_03_grand_opening_cafe", "sbh_11_16x9_heavy",
                   "sbh_noqr_11_16x9_heavy", "sbh_noqr_15_4x5_heavy", "sth_noqr_12_16x9_heavy"}


def test_no_new_text_loss(measured):
    lost = {r["id"] for r in measured if r["text_lost"]}
    assert lost == KNOWN_TEXT_LOSS, f"mới mất chữ: {sorted(lost - KNOWN_TEXT_LOSS)}; đã hết: {sorted(KNOWN_TEXT_LOSS - lost)}"


def test_render_plan_to_poster_returns_gate4_report(tmp_path):
    """Đường render THẬT (demo_server dùng) phải mang báo cáo Cổng 4 ra ngoài."""
    import json

    from run_template_test import generate_mock_backdrop_data_uri, parse_case_to_plan
    from tendoo_v3.renderer import render_plan_to_poster

    suite = Path(__file__).resolve().parent / "test_sandwich_bottom_heavy_suite.json"
    case = next(c for c in json.loads(suite.read_text(encoding="utf-8")) if c["id"] == "sbh_11_16x9_heavy")
    plan, w, h = parse_case_to_plan(case, "sandwich_bottom_heavy")
    report: list = []
    render_plan_to_poster(plan, generate_mock_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone), tmp_path / "p.png", w, h, overflow_report=report)
    assert report and all({"cls", "verdict"} <= set(o) for o in report), report  # đường render thật mang báo cáo ra


def test_hero_is_largest_text(measured):
    bad = [r["id"] for r in measured if r["hero"] and any(e["tier"] in (2, 3) and e["size"] > r["hero"] for e in r["elements"])]
    assert not bad, f"Có chữ phụ to hơn hero: {bad}"


# SQUINT TEST §4.5 -- BÁNH CÓC (GĐ 1). Chưa thể đòi mọi poster đạt 4 điều kiện (mốc 26/09: 51/379),
# nên CI chặn mọi thay đổi làm GIẢM số case đạt bất kỳ điều kiện nào ở bất kỳ template nào.
# Cải thiện được thì chạy lại: probe_type_hierarchy.py --bg --write-baseline tests/squint_baseline.json
BASELINE = Path(__file__).resolve().parent / "squint_baseline.json"


def test_squint_does_not_regress(measured):
    import json

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    now = squint_baseline(summarize(measured))
    worse = [
        f"{tpl}.{k}: {now[tpl][k]} < mốc {b[k]}"
        for tpl, b in base.items() for k in SQUINT_KEYS if now.get(tpl, {}).get(k, 0) < b[k]
    ]
    assert not worse, "Squint test tệ đi so với mốc: " + "; ".join(worse)


# GĐ 3 -- BÁNH CÓC CHO TIÊU ĐỀ NHIỀU CỠ: các case mà "LLM lý tưởng" (hero_markup.suggest_hero_parts)
# tách được hero_parts, đo với trần subhead theo intent. Mốc 26/09: 80 case, đạt cả 4 = 41
# (tiêu đề phẳng cùng các case đó: 17). Cập nhật:
#   probe_type_hierarchy.py --bg --oracle --write-baseline tests/squint_baseline_oracle.json
BASELINE_ORACLE = Path(__file__).resolve().parent / "squint_baseline_oracle.json"


@pytest.fixture(scope="module")
def measured_oracle(page):
    return [{**measure_case(page, case, tpl, with_bg=True), "suite": suite}
            for suite, tpl, case in with_oracle_hero_parts(load_cases(None), only_changed=True)]


def test_squint_with_hero_parts_does_not_regress(measured_oracle):
    import json

    base = json.loads(BASELINE_ORACLE.read_text(encoding="utf-8"))
    now = squint_baseline(summarize(measured_oracle))
    worse = [
        f"{tpl}.{k}: {now[tpl][k]} < mốc {b[k]}"
        for tpl, b in base.items() for k in SQUINT_KEYS if now.get(tpl, {}).get(k, 0) < b[k]
    ]
    assert not worse, "Squint (hero_parts) tệ đi so với mốc: " + "; ".join(worse)
    # Luật 6: master_03 -- dải đáy grand_opening, CTA to đè dòng cửa hàng; cứu chữ chỉ co phần tử bị đè (§8).
    # sbh_noqr_15: đã mất chữ cả khi tiêu đề phẳng (KNOWN_TEXT_LOSS) -- không do hero_parts.
    lost = {r["id"] for r in measured_oracle if r["text_lost"]} - {"master_03_grand_opening_cafe", "sbh_noqr_15_4x5_heavy"}
    assert not lost, f"hero_parts gây mất chữ: {sorted(lost)}"



# GĐ 4 -- BÁNH CÓC NỀN KHẮC NGHIỆT: nền giả có vệt sáng / mảng tối cục bộ (run_template_test.
# generate_harsh_backdrop_data_uri), đo 1/4 suite (lấy đều). Mốc 26/09: C3 trước quầng thích ứng
# 159/379 toàn suite -> 317/379. Cập nhật:
#   probe_type_hierarchy.py --bg --harsh --every 4 --write-baseline tests/squint_baseline_harsh.json
BASELINE_HARSH = Path(__file__).resolve().parent / "squint_baseline_harsh.json"


@pytest.fixture(scope="module")
def measured_harsh(page):
    return [{**measure_case(page, case, tpl, with_bg=True, harsh=True), "suite": suite}
            for suite, tpl, case in load_cases(None)[::4]]


def test_squint_on_harsh_background_does_not_regress(measured_harsh):
    import json

    base = json.loads(BASELINE_HARSH.read_text(encoding="utf-8"))
    now = squint_baseline(summarize(measured_harsh))
    worse = [
        f"{tpl}.{k}: {now[tpl][k]} < mốc {b[k]}"
        for tpl, b in base.items() for k in SQUINT_KEYS if now.get(tpl, {}).get(k, 0) < b[k]
    ]
    assert not worse, "Squint (nền khắc nghiệt) tệ đi so với mốc: " + "; ".join(worse)


def test_adaptive_halo_only_where_needed(page):
    """Quầng chỉ bật ở dòng thiếu tương phản: nền giả thường -> hầu như không; nền khắc nghiệt -> có."""
    from run_template_test import generate_harsh_backdrop_data_uri, generate_mock_backdrop_data_uri, parse_case_to_plan
    from tendoo_v3.renderer import build_template_html

    case = json.loads((Path(__file__).resolve().parent / "test_lifestyle_corner_pod_suite.json").read_text(encoding="utf-8"))[0]
    plan, w, h = parse_case_to_plan(case, "lifestyle_corner_pod")
    halos = []
    for bg in (generate_mock_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone),
               generate_harsh_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone, seed=case["id"])):
        page.set_viewport_size({"width": w, "height": h})
        page.set_content(build_template_html(plan, bg, w, h), wait_until="load")
        page.wait_for_function("window.__tendooAutofitDone === true", timeout=5000)
        halos.append(page.evaluate("window.__tendooHalo"))
    # Nền khắc nghiệt -> nhiều dòng cần xử lý tương phản (quầng / đảo màu / lớp mờ nhẹ) hơn nền thường.
    assert len(halos[1]) > len(halos[0])
    assert any(x.get("scrim") or x.get("flip") or x.get("bg") for x in halos[1])



# GĐ 5 -- BÁNH CÓC MASKLESS: case có intent cho phép maskless (catalog.MASKLESS_INTENTS), hero_parts
# của LLM lý tưởng, trên nền ít chi tiết phủ CẢ khung (không vùng tĩnh dưới chữ), 1/2 số case.
#   probe_type_hierarchy.py --bg --maskless --oracle --every 2 --write-baseline tests/squint_baseline_maskless.json
BASELINE_MASKLESS = Path(__file__).resolve().parent / "squint_baseline_maskless.json"


@pytest.fixture(scope="module")
def measured_maskless(page):
    from tendoo_v3.catalog import MASKLESS_INTENTS, resolve_intent

    cases = [(s, t, c) for s, t, c in with_oracle_hero_parts(load_cases(None), only_changed=True)
             if resolve_intent(t, c.get("plan", c).get("visual_intent")) in MASKLESS_INTENTS][::2]
    return [{**measure_case(page, case, tpl, with_bg=True, maskless=True), "suite": suite} for suite, tpl, case in cases]


def test_squint_maskless_does_not_regress(measured_maskless):
    base = json.loads(BASELINE_MASKLESS.read_text(encoding="utf-8"))
    now = squint_baseline(summarize(measured_maskless))
    worse = [
        f"{tpl}.{k}: {now[tpl][k]} < mốc {b[k]}"
        for tpl, b in base.items() for k in SQUINT_KEYS if now.get(tpl, {}).get(k, 0) < b[k]
    ]
    assert not worse, "Squint (maskless) tệ đi so với mốc: " + "; ".join(worse)
