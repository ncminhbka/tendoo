#!/usr/bin/env python3
"""
scripts/render_components_showcase.py

Bộ mẫu linh kiện GĐ 2 (ROADMAP §4.4): render tests/components_showcase.json ra ảnh và in squint
test (§4.5) của từng mẫu CÓ và KHÔNG có linh kiện -- để nhìn bằng mắt và so điều kiện nào đổi.

  PYTHONPATH=src python scripts/render_components_showcase.py --out output_probe/components_showcase
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SHOWCASE = PROJECT_ROOT / "tests" / "components_showcase.json"
COMPONENT_KEYS = ("badge_style", "stat_style", "decor")
CONDITIONS = ("c1_anchor", "c2_no_wall", "c3_bg", "c4_no_loss")


def without_components(case: dict) -> dict:
    return {**case, "plan": {k: v for k, v in case["plan"].items() if k not in COMPONENT_KEYS}}


def load_showcase() -> list:
    return json.loads(SHOWCASE.read_text(encoding="utf-8"))


def main() -> None:
    from playwright.sync_api import sync_playwright

    from probe_type_hierarchy import measure_case
    from run_template_test import generate_mock_backdrop_data_uri, parse_case_to_plan
    from tendoo_v3.renderer import build_template_html
    from tendoo_v3.validators import check_plan

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output_probe" / "components_showcase"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for case in load_showcase():
            tpl = case["plan"]["template"]
            on, off = measure_case(page, case, tpl, with_bg=True), measure_case(page, without_components(case), tpl, with_bg=True)
            plan, w, h = parse_case_to_plan(case, tpl)
            flags = "  ".join(f"{k[:2]} {'✓' if on.get(k) else '✗'}{'' if on.get(k) == off.get(k) else ' (không LK: ' + ('✓' if off.get(k) else '✗') + ')'}" for k in CONDITIONS)
            print(f"{case['id']:<38} tương phản {on['contrast']} (không LK {off['contrast']}, cần {on['contrast_target']})  {flags}")
            for issue in check_plan(plan):
                print(f"    [Cổng 2] {issue}")
            page.set_viewport_size({"width": w, "height": h})
            page.set_content(build_template_html(plan, generate_mock_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone), w, h), wait_until="load")
            page.wait_for_function("window.__tendooAutofitDone === true", timeout=3000)
            page.screenshot(path=str(out / f"{case['id']}.png"))
        browser.close()
    print(f"\nẢnh: {out}")


if __name__ == "__main__":
    main()
