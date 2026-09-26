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
  overflow   = Cổng 4 (window.__tendooOverflow, styles.py Bước 4)        -- §4.5 điều kiện 4
               text_lost = verdict clipped/overlap (mất chữ thật); spill = báo động giả
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
import re
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

from run_template_test import generate_harsh_backdrop_data_uri, generate_lowdetail_backdrop_data_uri, generate_mock_backdrop_data_uri, parse_case_to_plan
from tendoo_v3.catalog import INTENT_PROFILES, resolve_intent
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


def has_cross_tier_wall(elements: List[Dict[str, Any]]) -> bool:
    """§4.5 điều kiện 2, ĐỊNH NGHĨA LẠI (GĐ 1, theo §8): >= 3 phần tử nằm trong dải ±20% cỡ của
    nhau VÀ thuộc >= 2 cấp khác nhau. Nhiều phần tử CÙNG Cấp 3 (hotline + địa chỉ + CTA) cùng cỡ
    là một nhóm hợp lệ, không phải "chữ nhòe ngang nhau" (DESIGN_PRINCIPLES §1.2)."""
    items = sorted((e["size"], e["tier"]) for e in elements if e["tier"] in (1, 2, 3) and e["size"] > 0)
    for i in range(len(items)):
        window = [t for sz, t in items[i:] if sz <= items[i][0] * 1.2]
        if len(window) >= 3 and len(set(window)) >= 2:
            return True
    return False


TEXT_BOXES_JS = r"""
() => {
  const out = [];
  for (const el of document.querySelectorAll('[data-autofit]')) {
    if (el.parentElement && el.parentElement.closest('[data-autofit]')) continue;
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode, host = node.parentElement;
      if (!node.textContent.trim() || host.closest('svg')) continue;
      const cs = getComputedStyle(host);
      const range = document.createRange(); range.selectNodeContents(node);
      for (const r of range.getClientRects()) {
        if (r.width < 2 || r.height < 2) continue;
        out.push({cls: (el.className || '').toString().trim().split(/\s+/)[0], x: r.left, y: r.top, w: r.width, h: r.height,
                  color: cs.webkitTextFillColor || cs.color, size: parseFloat(cs.fontSize) || 0, weight: parseInt(cs.fontWeight) || 400});
      }
    }
  }
  return out;
}
"""
# Ẩn RUỘT chữ nhưng GIỮ text-shadow (GĐ 4): quầng/bóng quanh nét là một phần nền cục bộ thật mà
# mắt thấy sau chữ -- trước đây bỏ cả shadow nên quầng thích ứng không được tính.
HIDE_TEXT_CSS = "* { color: transparent !important; -webkit-text-fill-color: transparent !important; }"
HIDE_SHADOW_CSS = "* { text-shadow: none !important; }"


def _parse_rgba(css):
    m = re.match(r"rgba?\(([^)]+)\)", css or "")
    if not m:
        return None
    parts = [float(x) for x in m.group(1).replace("/", ",").split(",") if x.strip()]
    return parts[:3], (parts[3] if len(parts) > 3 else 1.0)


def _luminance(rgb) -> float:
    def ch(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def measure_bg_contrast(page) -> List[Dict[str, Any]]:
    """§4.5 điều kiện 3: WCAG màu chữ vs màu nền ĐO THẬT ngay dưới từng dòng chữ. Chụp lại trang
    với chữ ẩn rồi lấy màu trung bình vùng dưới chữ. Giới hạn: bỏ qua text-shadow/glow (WCAG không
    tính); chữ tô gradient (fill trong suốt, vd 3d_gold) đánh dấu `gradient`, không chấm."""
    import io

    import numpy as np
    from PIL import Image

    boxes = page.evaluate(TEXT_BOXES_JS)
    page.add_style_tag(content=HIDE_TEXT_CSS)
    # Style inline !important (quầng/đổi màu nhấn của autofit Bước 5) thắng stylesheet !important ->
    # phải ẩn ruột chữ bằng CHÍNH inline style, nếu không chữ còn hiện và bị tính như nền.
    page.evaluate("""() => document.querySelectorAll('body *').forEach(n => {
        n.style.setProperty('color', 'transparent', 'important');
        n.style.setProperty('-webkit-text-fill-color', 'transparent', 'important'); })""")
    img_sh = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB")).astype(float)
    page.evaluate("() => document.querySelectorAll('body *').forEach(n => n.style.setProperty('text-shadow', 'none', 'important'))")
    page.add_style_tag(content=HIDE_SHADOW_CSS)
    img_plain = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB")).astype(float)
    a = _box_ratios(boxes, img_sh)
    b = _box_ratios(boxes, img_plain)
    # Lấy lần đo TỐT HƠN cho từng dòng: quầng tối (GĐ 4) chỉ hiện ở lần giữ shadow; quầng phát
    # sáng CÙNG màu chữ (neon) làm lần đó tụt oan nhưng lần bỏ shadow vẫn đúng như trước GĐ 4.
    return [x if (x.get("ratio") or 0) >= (y.get("ratio") or 0) else y for x, y in zip(a, b)]


def _box_ratios(boxes, img) -> List[Dict[str, Any]]:
    out = []
    for b in boxes:
        parsed = _parse_rgba(b["color"])
        x0, y0 = max(0, int(b["x"])), max(0, int(b["y"]))
        x1, y1 = min(img.shape[1], int(b["x"] + b["w"])), min(img.shape[0], int(b["y"] + b["h"]))
        if parsed is None or x1 <= x0 or y1 <= y0:
            out.append({"cls": b["cls"], "ratio": None, "gradient": False, "skip": True})
            continue
        rgb, alpha = parsed
        if alpha < 0.5:
            out.append({"cls": b["cls"], "ratio": None, "gradient": True})
            continue
        # GĐ 4: chấm theo MẢNG XẤU NHẤT, không theo trung bình cả dòng -- nền trung bình "tối"
        # nhưng có vệt chói ngay dưới vài chữ vẫn làm chìm đúng những chữ đó (§4.3). Cắt dòng
        # thành các ô vuông cạnh = chiều cao dòng, lấy tỉ lệ WCAG thấp nhất.
        l1 = _luminance(rgb)
        side = max(4, y1 - y0)
        ratios = []
        for xs in range(x0, x1, side):
            patch = img[y0:y1, xs:min(x1, xs + side)].reshape(-1, 3)
            if len(patch) < side:  # mẩu cuối quá hẹp
                continue
            l2 = _luminance(patch.mean(axis=0))
            ratios.append((max(l1, l2) + 0.05) / (min(l1, l2) + 0.05))
        if not ratios:
            out.append({"cls": b["cls"], "ratio": None, "gradient": False, "skip": True})
            continue
        l2 = _luminance(img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0))
        large = b["size"] >= 24 or (b["size"] >= 18.66 and b["weight"] >= 700)
        out.append({"cls": b["cls"], "ratio": round(min(ratios), 2), "ratio_mean": round((max(l1, l2) + 0.05) / (min(l1, l2) + 0.05), 2),
                    "need": 3.0 if large else 4.5, "gradient": False})
    return out


def measure_plan(page, plan, w: int, h: int, with_bg: bool = False, harsh_seed: Optional[str] = None,
                 lowdetail_seed: Optional[str] = None, bg_override: Optional[str] = None) -> Dict[str, Any]:
    if bg_override is not None:  # GĐ 3R: ảnh nền THẬT (sinh bằng mô hình ảnh) thay nền giả
        bg = bg_override
    elif lowdetail_seed is not None:  # GĐ 5: nền maskless ít chi tiết phủ cả khung
        bg = generate_lowdetail_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone, seed=lowdetail_seed)
    elif harsh_seed is not None:  # GĐ 4: nền có vệt sáng / mảng tối cục bộ
        bg = generate_harsh_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone, seed=harsh_seed)
    else:
        bg = generate_mock_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone)
    html = build_template_html(plan=plan, bg_data_uri=bg, width=w, height=h)
    page.set_viewport_size({"width": w, "height": h})
    page.set_content(html, wait_until="load")
    page.evaluate("document.fonts.ready")
    try:
        page.wait_for_function("window.__tendooAutofitDone === true", timeout=3000)
    except Exception:
        pass
    raw = page.evaluate(MEASURE_JS)
    gate4 = page.evaluate("window.__tendooOverflow || []")
    gate4_cls = {o["cls"] for o in gate4}

    elements = []
    for e in raw:
        name, tier = classify(e["cls"])
        size = e["eff"] or e["font"]
        elements.append({
            "cls": name, "tier": tier, "size": size, "font": e["font"],
            "min_font": e["minFont"], "max_font": e["maxFont"], "max_h": e["maxH"], "scroll_h": e["scrollH"],
            "pinned": e["font"] >= e["maxFont"] - 0.5 if e["maxFont"] == e["maxFont"] else False,
            "overflow": name in gate4_cls,
            "text": e["text"],
        })

    hero = max((e["size"] for e in elements if e["tier"] == 1), default=0.0)
    t2 = [e for e in elements if e["tier"] == 2]
    t3 = [e for e in elements if e["tier"] == 3]
    sub = max((e["size"] for e in t2), default=0.0)
    t3_top = max(t3, key=lambda e: e["size"], default=None)
    intent = resolve_intent(plan.template, getattr(plan, "visual_intent", None))
    target = INTENT_PROFILES[intent]["contrast_target"]
    # matrix_board: điểm neo so với cả khối nội dung chính (menu/steps), không chỉ Cấp 2/3.
    pool = t2 + t3 + ([e for e in elements if e["tier"] == "content"] if intent == "matrix_board" else [])
    sec_top = max(pool, key=lambda e: e["size"], default=None)
    contrast = round(hero / sec_top["size"], 2) if (sec_top and hero) else None
    text_lost = [o["cls"] for o in gate4 if o["verdict"] in ("clipped", "overlap")]
    wall = has_cross_tier_wall(elements)
    result = {
        "template": plan.template, "w": w, "h": h, "intent": intent, "contrast_target": target,
        "hero": hero, "subhead": sub,
        "t3_max": t3_top["size"] if t3_top else 0.0, "t3_max_cls": t3_top["cls"] if t3_top else "",
        "contrast": contrast,
        "top_secondary": sec_top["cls"] if sec_top else "",
        "inversion": bool(sub and t3_top and t3_top["size"] > sub),
        "wall": wall,
        "overflow": [f"{o['cls']}:{o['verdict']}" for o in gate4],
        "text_lost": text_lost,
        "unknown": [e["cls"] for e in elements if e["tier"] == "?"],
        "elements": elements,
        # 4 điều kiện squint test §4.5 (điều kiện 3 chỉ khi with_bg)
        "c1_anchor": contrast is None or contrast >= target,
        "c2_no_wall": not wall,
        "c4_no_loss": not text_lost,
    }
    if with_bg:
        bgc = measure_bg_contrast(page)
        fails = [b for b in bgc if b.get("ratio") is not None and b["ratio"] < b["need"]]
        result.update({
            "bg_contrast": bgc, "c3_bg": not fails, "bg_fail": sorted({b["cls"] for b in fails}),
            "bg_min": min((b["ratio"] for b in bgc if b.get("ratio")), default=None),
        })
    result["squint_pass"] = result["c1_anchor"] and result["c2_no_wall"] and result["c4_no_loss"] and result.get("c3_bg", True)
    return result


def measure_case(page, case: Dict[str, Any], template: str, with_bg: bool = False, harsh: bool = False, maskless: bool = False) -> Dict[str, Any]:
    plan, w, h = parse_case_to_plan(case, default_template=template)
    return {"id": case["id"], **measure_plan(page, plan, w, h, with_bg=with_bg, harsh_seed=case["id"] if harsh else None,
                                             lowdetail_seed=case["id"] if maskless else None)}


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
            "text_lost": sum(bool(r["text_lost"]) for r in rs),
            "intent": rs[0]["intent"], "target": rs[0]["contrast_target"],
            "c1": sum(r["c1_anchor"] for r in rs), "c2": sum(r["c2_no_wall"] for r in rs),
            "c3": sum(r["c3_bg"] for r in rs) if "c3_bg" in rs[0] else None, "c4": sum(r["c4_no_loss"] for r in rs),
            "squint": sum(r["squint_pass"] for r in rs),
            "top_secondary": ", ".join(f"{k}:{v}" for k, v in sorted(tops.items(), key=lambda kv: -kv[1])[:2]),
        }
    return summ


SQUINT_KEYS = ("c1", "c2", "c3", "c4", "squint")


def squint_baseline(summ: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Mốc bánh cóc cho CI: số case ĐẠT từng điều kiện squint, theo template."""
    return {tpl: {"n": s["n"], **{k: s[k] for k in SQUINT_KEYS}} for tpl, s in summ.items()}


def print_table(summ, base=None):
    def d(key, tpl, fmt="{:+}"):
        if not base or tpl not in base or base[tpl].get(key) is None or summ[tpl].get(key) is None:
            return ""
        delta = summ[tpl][key] - base[tpl][key]
        return f" ({fmt.format(round(delta, 2))})" if delta else ""

    hdr = (f"{'template':<24}{'n':>4} {'intent':<18}{'ngưỡng':>6} {'contrast TB':>13} {'đảo':>5}"
           f" {'C1 neo':>9} {'C2 tường':>9} {'C3 nền':>8} {'C4 chữ':>8} {'ĐẠT CẢ 4':>12}")
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(int)
    for tpl, s in summ.items():
        for k in ("n", "inversions", "c1", "c2", "c3", "c4", "squint"):
            tot[k] += s.get(k) or 0
        c3 = "-" if s.get("c3") is None else str(s["c3"])
        print(
            f"{tpl:<24}{s['n']:>4} {s.get('intent', ''):<18}{s.get('target', ''):>6} {str(s['contrast_mean']) + d('contrast_mean', tpl):>13}"
            f" {s['inversions']:>5} {str(s.get('c1')) + d('c1', tpl):>9} {str(s.get('c2')) + d('c2', tpl):>9} {c3:>8}"
            f" {str(s.get('c4')) + d('c4', tpl):>8} {str(s.get('squint')) + d('squint', tpl):>12}"
        )
    print("-" * len(hdr))
    print(f"{'TỔNG':<24}{tot['n']:>4} {'':<18}{'':>6} {'':>13} {tot['inversions']:>5} {tot['c1']:>9} {tot['c2']:>9}"
          f" {tot['c3']:>8} {tot['c4']:>8} {tot['squint']:>12}")


def with_oracle_hero_parts(cases: List[tuple], only_changed: bool = False) -> List[tuple]:
    """Cô lập LLM khỏi phép đo (GĐ 3): gắn hero_parts mà một LLM giỏi sẽ trả -- bộ tách tất định
    `hero_markup.suggest_hero_parts` (chỉ tách khi có con số/cụm từ móc; còn lại giữ phẳng)."""
    import copy

    from tendoo_v3.hero_markup import suggest_hero_parts

    out = []
    for suite, tpl, c in cases:
        src = c.get("plan", c)
        parts = [] if src.get("hero_parts") else suggest_hero_parts(src.get("hero", ""))
        if parts:
            c = copy.deepcopy(c)
            (c["plan"] if "plan" in c else c)["hero_parts"] = parts
        if parts or not only_changed:
            out.append((suite, tpl, c))
    return out


def main():
    ap = argparse.ArgumentParser(description="Đo thứ bậc cỡ chữ thật (ROADMAP §4.1/§4.5)")
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output_probe" / "latest"))
    ap.add_argument("--compare", default=None, help="Thư mục kết quả cũ để so sánh (chứa summary.json)")
    ap.add_argument("--show", choices=["inversions", "overflows", "unknown", "squint"], default=None, help="In chi tiết case vi phạm")
    ap.add_argument("--bg", action="store_true", help="Đo thêm §4.5 điều kiện 3 (tương phản nền thật, chậm ~2x)")
    ap.add_argument("--harsh", action="store_true", help="GĐ 4: nền giả có vệt sáng / mảng tối cục bộ (cần --bg)")
    ap.add_argument("--maskless", action="store_true", help="GĐ 5: chỉ case có intent cho phép maskless, trên nền ít chi tiết phủ cả khung")
    ap.add_argument("--every", type=int, default=1, help="Chỉ đo 1/N case (lấy đều, tất định) -- mốc nền khắc nghiệt dùng 4")
    ap.add_argument("--oracle", action="store_true", help="GĐ 3: chỉ đo các case mà 'LLM lý tưởng' (hero_markup.suggest_hero_parts) tách được hero_parts")
    ap.add_argument("--write-baseline", default=None, help="Ghi mốc squint (bánh cóc CI), vd tests/squint_baseline.json -- cần --bg")
    args = ap.parse_args()

    cases = load_cases(args.template)
    if args.oracle:
        cases = with_oracle_hero_parts(cases, only_changed=True)
    if args.maskless:
        from tendoo_v3.catalog import MASKLESS_INTENTS, resolve_intent

        cases = [(s, tp, c) for s, tp, c in cases if resolve_intent(tp, c.get("plan", c).get("visual_intent")) in MASKLESS_INTENTS]
    cases = cases[:: max(1, args.every)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for i, (suite, tpl, case) in enumerate(cases, 1):
            try:
                r = measure_case(page, case, tpl, with_bg=args.bg, harsh=args.harsh, maskless=args.maskless)
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
            elif args.show == "squint" and not r["squint_pass"]:
                why = [k for k in ("c1_anchor", "c2_no_wall", "c3_bg", "c4_no_loss") if r.get(k) is False]
                print(f"  {r['id']:<34} trượt {why}  contrast {r['contrast']} / {r['contrast_target']}  nền yếu {r.get('bg_fail', [])}")
            elif args.show == "unknown" and r["unknown"]:
                print(f"  {r['id']:<34} class chưa phân cấp: {r['unknown']}")

    if args.write_baseline:
        if not args.bg or args.template:
            raise SystemExit("--write-baseline cần --bg và toàn bộ template (không dùng --template)")
        # --oracle: mốc riêng tests/squint_baseline_oracle.json cho các case có hero_parts
        Path(args.write_baseline).write_text(json.dumps(squint_baseline(summ), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Đã ghi mốc squint: {args.write_baseline}")
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
