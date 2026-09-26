#!/usr/bin/env python3
"""
scripts/probe_type_hierarchy.py

Đo THỨ BẬC CỠ CHỮ THẬT sau autofit trên Chromium cho toàn bộ suite JSON của 14 template
(ROADMAP §4.1 Luật 1 + §4.5 Luật 5). Đây là công cụ nghiệm thu của giai đoạn 0A+:
"không phần tử Cấp 3 nào to hơn Cấp 2".

Khác `run_template_test.py` (kiểm không-crash, mỗi case khởi động 1 Chromium mới ~2s):
  - dùng CHUNG 1 trình duyệt, chỉ dựng HTML qua `build_template_html` (không ghi PNG);
  - đo cỡ chữ HIỆU DỤNG của từng phần tử `[data-autofit]` = cỡ lớn nhất trong các
    phần tử con thực sự mang chữ (pill con có thể dùng `em` khác cha);
  - phân cấp theo class (bảng `TIER_BY_CLASS` bên dưới), class lạ báo `?` chứ không đoán.

Chỉ số mỗi case:
  contrast   = cỡ hero / cỡ phần tử phụ to nhất (Cấp 2 ∪ Cấp 3)       -- §4.5 điều kiện 1
  inversion  = có phần tử Cấp 3 to hơn subhead (chỉ tính khi có subhead) -- DoD 0A+
  wall       = có >=3 phần tử (mọi cấp) nằm trong dải ±20% của nhau     -- §4.5 điều kiện 2
  overflow   = phần tử kẹt ở sàn min_font mà scrollHeight > max_height   -- §4.5 điều kiện 4
  pinned     = phần tử chạm trần max_font (cho biết ràng buộc nào thắng)

Chạy:
  PYTHONPATH=src python scripts/probe_type_hierarchy.py --out output_probe/baseline
  PYTHONPATH=src python scripts/probe_type_hierarchy.py --template split_left --compare output_probe/baseline
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

from run_template_test import generate_mock_backdrop_data_uri, parse_case_to_plan
from tendoo_v3.renderer import build_template_html
from tendoo_v3.styles import CONTENT_CLASSES, TIER1_CLASSES, TIER2_CLASSES, TIER3_CLASSES

TESTS_DIR = PROJECT_ROOT / "tests"

# Cấp thị giác lấy từ CÙNG nguồn mà bước giữ thứ bậc trong autofit JS dùng (styles.py).
# "content" = nội dung CHÍNH của template dạng bảng (menu, các bước, lời chứng thực) --
# không kiểm đảo thứ bậc (ROADMAP §8: áp nhầm ngưỡng cho matrix_board là rủi ro Cao).
TIER_BY_CLASS = {
    **{c: 1 for c in TIER1_CLASSES},
    **{c: 2 for c in TIER2_CLASSES},
    **{c: 3 for c in TIER3_CLASSES},
    **{c: "content" for c in CONTENT_CLASSES},
}

MEASURE_JS = r"""
() => {
  const out = [];
  for (const el of document.querySelectorAll('[data-autofit]')) {
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    if (cs.display === 'none' || cs.visibility === 'hidden' || r.width <= 0 || r.height <= 0) continue;
    // Autofit lồng nhau (vd store-item bên trong store-info-row): chỉ đo phần tử ngoài cùng.
    if (el.parentElement && el.parentElement.closest('[data-autofit]')) continue;
    let eff = 0;
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const t = walker.currentNode;
      if (!t.textContent.trim()) continue;
      const host = t.parentElement;
      if (host.closest('svg') || host.classList.contains('deco-bullet') || host.classList.contains('freetext-bullet')) continue;
      const hr = host.getBoundingClientRect();
      if (hr.width <= 0 || hr.height <= 0) continue;
      eff = Math.max(eff, parseFloat(getComputedStyle(host).fontSize) || 0);
    }
    out.push({
      cls: (el.className || '').toString().trim().split(/\s+/),
      font: parseFloat(cs.fontSize) || 0,
      eff: eff,
      minFont: parseFloat(el.dataset.minFont || 'NaN'),
      maxFont: parseFloat(el.dataset.maxFont || 'NaN'),
      maxH: parseFloat(el.dataset.maxHeight || 'NaN'),
      scrollH: el.scrollHeight,
      text: (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 50),
    });
  }
  return out;
}
"""


def classify(classes: List[str]) -> tuple:
    for c in classes:
        if c in TIER_BY_CLASS:
            return c, TIER_BY_CLASS[c]
    return (classes[0] if classes else "?"), "?"


def has_wall(sizes: List[float]) -> bool:
    s = sorted(x for x in sizes if x > 0)
    for i in range(len(s) - 2):
        if s[i + 2] <= s[i] * 1.2:
            return True
    return False


def measure_case(page, case: Dict[str, Any], template: str) -> Dict[str, Any]:
    plan, w, h = parse_case_to_plan(case, default_template=template)
    html = build_template_html(plan=plan, bg_data_uri=generate_mock_backdrop_data_uri(w, h, plan.style.theme_color), width=w, height=h)
    page.set_viewport_size({"width": w, "height": h})
    page.set_content(html, wait_until="load")
    page.evaluate("document.fonts.ready")
    try:
        page.wait_for_function("window.__tendooAutofitDone === true", timeout=3000)
    except Exception:
        pass
    raw = page.evaluate(MEASURE_JS)

    elements = []
    for e in raw:
        name, tier = classify(e["cls"])
        size = e["eff"] or e["font"]
        at_floor = e["font"] <= e["minFont"] + 0.25 if e["minFont"] == e["minFont"] else False
        elements.append({
            "cls": name, "tier": tier, "size": size, "font": e["font"],
            "min_font": e["minFont"], "max_font": e["maxFont"], "max_h": e["maxH"], "scroll_h": e["scrollH"],
            "pinned": e["font"] >= e["maxFont"] - 0.5 if e["maxFont"] == e["maxFont"] else False,
            "overflow": bool(at_floor and e["maxH"] == e["maxH"] and e["scrollH"] > e["maxH"] + 1.5),
            "text": e["text"],
        })

    hero = max((e["size"] for e in elements if e["tier"] == 1), default=0.0)
    t2 = [e for e in elements if e["tier"] == 2]
    t3 = [e for e in elements if e["tier"] == 3]
    sub = max((e["size"] for e in t2), default=0.0)
    t3_top = max(t3, key=lambda e: e["size"], default=None)
    secondary = [e for e in t2 + t3]
    sec_top = max(secondary, key=lambda e: e["size"], default=None)
    return {
        "id": case["id"], "template": plan.template, "w": w, "h": h,
        "hero": hero, "subhead": sub,
        "t3_max": t3_top["size"] if t3_top else 0.0, "t3_max_cls": t3_top["cls"] if t3_top else "",
        "contrast": round(hero / sec_top["size"], 2) if (sec_top and hero) else None,
        "top_secondary": sec_top["cls"] if sec_top else "",
        "inversion": bool(sub and t3_top and t3_top["size"] > sub),
        "wall": has_wall([e["size"] for e in elements if e["tier"] in (1, 2, 3)]),
        "overflow": [e["cls"] for e in elements if e["overflow"]],
        "unknown": [e["cls"] for e in elements if e["tier"] == "?"],
        "elements": elements,
    }


def load_cases(template: Optional[str]) -> List[tuple]:
    out = []
    for f in sorted(glob.glob(str(TESTS_DIR / "test_*_suite.json"))):
        suite = Path(f).name
        for c in json.load(open(f, encoding="utf-8")):
            src = c.get("plan", c)
            tpl = src.get("template") or suite[len("test_"):-len("_suite.json")].replace("_no_content", "")
            if template and tpl != template:
                continue
            out.append((suite, tpl, c))
    return out


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by = defaultdict(list)
    for r in results:
        by[r["template"]].append(r)
    summ = {}
    for tpl, rs in sorted(by.items()):
        cs = [r["contrast"] for r in rs if r["contrast"]]
        tops = defaultdict(int)
        for r in rs:
            if r["top_secondary"]:
                tops[r["top_secondary"]] += 1
        summ[tpl] = {
            "n": len(rs),
            "contrast_mean": round(sum(cs) / len(cs), 2) if cs else None,
            "contrast_max": max(cs) if cs else None,
            "ge4x": sum(1 for c in cs if c >= 4.0),
            "inversions": sum(r["inversion"] for r in rs),
            "walls": sum(r["wall"] for r in rs),
            "overflows": sum(bool(r["overflow"]) for r in rs),
            "top_secondary": ", ".join(f"{k}:{v}" for k, v in sorted(tops.items(), key=lambda kv: -kv[1])[:2]),
        }
    return summ


def print_table(summ, base=None):
    def d(key, tpl, fmt="{:+}"):
        if not base or tpl not in base or base[tpl].get(key) is None or summ[tpl].get(key) is None:
            return ""
        delta = summ[tpl][key] - base[tpl][key]
        return f" ({fmt.format(round(delta, 2))})" if delta else ""

    hdr = f"{'template':<24}{'n':>4} {'contrast TB':>16} {'max':>6} {'>=4x':>9} {'đảo bậc':>12} {'tường':>10} {'tràn':>9}  phụ to nhất"
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(int)
    for tpl, s in summ.items():
        for k in ("n", "ge4x", "inversions", "walls", "overflows"):
            tot[k] += s[k]
        print(
            f"{tpl:<24}{s['n']:>4} {str(s['contrast_mean']) + d('contrast_mean', tpl):>16} {str(s['contrast_max']):>6}"
            f" {str(s['ge4x']) + d('ge4x', tpl):>9} {str(s['inversions']) + d('inversions', tpl):>12}"
            f" {str(s['walls']) + d('walls', tpl):>10} {str(s['overflows']) + d('overflows', tpl):>9}  {s['top_secondary']}"
        )
    print("-" * len(hdr))
    print(f"{'TỔNG':<24}{tot['n']:>4} {'':>16} {'':>6} {tot['ge4x']:>9} {tot['inversions']:>12} {tot['walls']:>10} {tot['overflows']:>9}")


def main():
    ap = argparse.ArgumentParser(description="Đo thứ bậc cỡ chữ thật (ROADMAP §4.1/§4.5)")
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output_probe" / "latest"))
    ap.add_argument("--compare", default=None, help="Thư mục kết quả cũ để so sánh (chứa summary.json)")
    ap.add_argument("--show", choices=["inversions", "overflows", "unknown"], default=None, help="In chi tiết case vi phạm")
    args = ap.parse_args()

    cases = load_cases(args.template)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for i, (suite, tpl, case) in enumerate(cases, 1):
            try:
                r = measure_case(page, case, tpl)
                r["suite"] = suite
                results.append(r)
            except Exception as ex:
                print(f"[LỖI] {case.get('id')}: {ex}")
            if i % 50 == 0:
                print(f"  ... {i}/{len(cases)}", flush=True)
        browser.close()
    print(f"Đo {len(results)}/{len(cases)} case trong {time.perf_counter() - t0:.1f}s\n")

    summ = summarize(results)
    base = None
    if args.compare:
        base_file = Path(args.compare) / "summary.json"
        base = json.loads(base_file.read_text(encoding="utf-8")) if base_file.exists() else None
    print_table(summ, base)

    if args.show:
        print()
        for r in results:
            if args.show == "inversions" and r["inversion"]:
                print(f"  {r['id']:<34} subhead {r['subhead']:>5} < {r['t3_max_cls']} {r['t3_max']}")
            elif args.show == "overflows" and r["overflow"]:
                print(f"  {r['id']:<34} tràn: {r['overflow']}")
            elif args.show == "unknown" and r["unknown"]:
                print(f"  {r['id']:<34} class chưa phân cấp: {r['unknown']}")

    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "cases.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "elements.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["suite", "case", "template", "w", "h", "cls", "tier", "size", "font", "min_font", "max_font", "pinned", "overflow", "text"])
        for r in results:
            for e in r["elements"]:
                wr.writerow([r["suite"], r["id"], r["template"], r["w"], r["h"], e["cls"], e["tier"], e["size"], e["font"],
                             e["min_font"], e["max_font"], e["pinned"], e["overflow"], e["text"]])
    print(f"\nKết quả: {out_dir}")


if __name__ == "__main__":
    main()
