"""Nghiệm thu GĐ 0A+ (ROADMAP §4.1, §7): đo cỡ chữ THẬT sau autofit trên Chromium cho toàn
bộ suite JSON -- không phần tử Cấp 3 nào được to hơn subhead.

Đây là test đầu tiên trong pytest thực sự RENDER 14 template (MIGRATION §4: trước đây các
suite JSON chỉ được script trong scripts/ đọc). Baseline 26/09: 53/379 case đảo bậc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from probe_type_hierarchy import load_cases, measure_case  # noqa: E402

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def measured():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            yield [measure_case(page, case, tpl) for _, tpl, case in load_cases(None)]
        finally:
            browser.close()


def test_all_suite_cases_measured(measured):
    assert len(measured) >= 379


def test_no_tier3_larger_than_subhead(measured):
    bad = [f"{r['id']}: subhead {r['subhead']} < {r['t3_max_cls']} {r['t3_max']}" for r in measured if r["inversion"]]
    assert not bad, "Cấp 3 to hơn Cấp 2:\n" + "\n".join(bad)


def test_every_autofit_class_has_a_tier(measured):
    unknown = sorted({c for r in measured for c in r["unknown"]})
    assert not unknown, f"Class autofit chưa phân cấp trong styles.py::TIER*_CLASSES: {unknown}"


def test_hero_is_largest_text(measured):
    bad = [r["id"] for r in measured if r["hero"] and any(e["tier"] in (2, 3) and e["size"] > r["hero"] for e in r["elements"])]
    assert not bad, f"Có chữ phụ to hơn hero: {bad}"
