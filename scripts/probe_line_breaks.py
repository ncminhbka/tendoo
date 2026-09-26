#!/usr/bin/env python3
"""
scripts/probe_line_breaks.py

GĐ 7a (ROADMAP §10.11, R2): đo chất lượng NGẮT DÒNG tiêu đề/subhead/câu trích dẫn trên toàn suite.
Đếm, cho từng dòng chữ sau render (Range.getClientRects theo từng từ):
  orphan        dòng cuối chỉ có 1 âm tiết (chữ mồ côi) -- trong khối >= 2 dòng
  split_pair    cặp âm tiết của TỪ GHÉP thông dụng (hero_markup.COMPOUNDS) bị tách qua 2 dòng
  split_unit    con số và đơn vị ("12 | TRIỆU", "3 | BƯỚC") bị tách qua 2 dòng
  ragged        độ chênh bề rộng dòng dài nhất / ngắn nhất > 2.5 (khối lởm chởm)

  PYTHONPATH=src python scripts/probe_line_breaks.py [--template X]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LINES_JS = r"""
(sel) => {
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    const words = [];
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode, text = node.textContent;
      const re = /\S+/g; let m;
      while ((m = re.exec(text))) {
        const rg = document.createRange(); rg.setStart(node, m.index); rg.setEnd(node, m.index + m[0].length);
        const r = rg.getClientRects()[0];
        if (r) words.push({w: m[0], top: Math.round(r.top), left: r.left, right: r.right});
      }
    }
    if (!words.length) continue;
    // Gom từ theo dòng (cùng top ±4px, theo thứ tự đọc)
    const lines = []; let cur = null;
    for (const w of words) {
      if (!cur || Math.abs(w.top - cur.top) > 4) { cur = {top: w.top, words: [], left: w.left, right: w.right}; lines.push(cur); }
      cur.words.push(w.w); cur.left = Math.min(cur.left, w.left); cur.right = Math.max(cur.right, w.right);
    }
    out.push({cls: el.className.toString().split(' ')[0], lines: lines.map(l => ({words: l.words, width: l.right - l.left}))});
  }
  return out;
}
"""
SELECTOR = ".hero-title, .hero-top-title, .quote-hero, .subhead-title"


def analyze(blocks, compounds, units):
    stats = collections.Counter()
    for b in blocks:
        lines = b["lines"]
        stats["blocks"] += 1
        if len(lines) < 2:
            continue
        stats["multiline"] += 1
        if len(lines[-1]["words"]) == 1:
            stats["orphan"] += 1
        widths = [l["width"] for l in lines if l["width"] > 0]
        if widths and max(widths) / max(1.0, min(widths)) > 2.5:
            stats["ragged"] += 1
        for a, c in zip(lines, lines[1:]):
            last, first = a["words"][-1].upper().strip(",.:;!?"), c["words"][0].upper().strip(",.:;!?")
            if f"{last} {first}" in compounds:
                stats["split_pair"] += 1
            if last[:1].isdigit() and first in units:
                stats["split_unit"] += 1
    return stats


def main() -> None:
    from playwright.sync_api import sync_playwright

    from probe_type_hierarchy import load_cases
    from run_template_test import generate_mock_backdrop_data_uri, parse_case_to_plan
    from tendoo_v3.hero_markup import COMPOUNDS, UNIT_WORDS
    from tendoo_v3.renderer import build_template_html

    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output_probe" / "line_breaks.json"))
    args = ap.parse_args()
    total = collections.Counter()
    per_tpl = collections.defaultdict(collections.Counter)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for suite, tpl, case in load_cases(args.template):
            plan, w, h = parse_case_to_plan(case, tpl)
            page.set_viewport_size({"width": w, "height": h})
            page.set_content(build_template_html(plan, generate_mock_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone), w, h), wait_until="load")
            page.wait_for_function("window.__tendooAutofitDone === true", timeout=5000)
            st = analyze(page.evaluate(LINES_JS, SELECTOR), COMPOUNDS, UNIT_WORDS)
            total.update(st); per_tpl[tpl].update(st)
        browser.close()
    keys = ("blocks", "multiline", "orphan", "split_pair", "split_unit", "ragged")
    print(f"{'template':<24}" + "".join(f"{k:>12}" for k in keys))
    for tpl, st in sorted(per_tpl.items()):
        print(f"{tpl:<24}" + "".join(f"{st[k]:>12}" for k in keys))
    print(f"{'TỔNG':<24}" + "".join(f"{total[k]:>12}" for k in keys))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"total": total, "per_template": per_tpl}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
