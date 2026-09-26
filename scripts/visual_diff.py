#!/usr/bin/env python3
"""
scripts/visual_diff.py

So ảnh render THẬT (Chromium) của mọi case suite giữa 2 phiên bản code -- lưới an toàn khi
viết lại HTML template (GĐ 0B-3). Dùng chung 1 trình duyệt, nền giả tất định.

  # 1. Chụp ảnh gốc (trước khi sửa)
  PYTHONPATH=src python scripts/visual_diff.py snap --out output_probe/vd_base
  # 2. Sửa template, chụp lại
  PYTHONPATH=src python scripts/visual_diff.py snap --out output_probe/vd_new --template sandwich_bottom_heavy
  # 3. So sánh: in % điểm ảnh khác, ghi ảnh diff (khung đỏ quanh vùng khác) cho case khác
  PYTHONPATH=src python scripts/visual_diff.py diff output_probe/vd_base output_probe/vd_new
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageChops, ImageDraw

# Ngưỡng coi 1 điểm ảnh là "khác": chênh > 24/255 trên kênh bất kỳ (bỏ qua nhiễu khử răng cưa).
PIXEL_TOL = 24


def snap(out: Path, template: str | None) -> None:
    from playwright.sync_api import sync_playwright

    from probe_type_hierarchy import load_cases
    from run_template_test import generate_mock_backdrop_data_uri, parse_case_to_plan
    from tendoo_v3.renderer import build_template_html

    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(template)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for suite, tpl, case in cases:
            plan, w, h = parse_case_to_plan(case, tpl)
            html = build_template_html(plan, generate_mock_backdrop_data_uri(w, h, plan.style.theme_color, plan.style.background_tone), w, h)
            page.set_viewport_size({"width": w, "height": h})
            page.set_content(html, wait_until="load")
            page.evaluate("document.fonts.ready")
            try:
                page.wait_for_function("window.__tendooAutofitDone === true", timeout=3000)
            except Exception:
                pass
            page.wait_for_timeout(200)  # lần fitElements() dự phòng ở setTimeout(150)
            page.screenshot(path=str(out / f"{Path(suite).stem}__{case['id']}.png"))
        browser.close()
    print(f"Chụp {len(cases)} case -> {out}")


def diff(base: Path, new: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fb in sorted(base.glob("*.png")):
        fn = new / fb.name
        if not fn.exists():
            continue
        a, b = Image.open(fb).convert("RGB"), Image.open(fn).convert("RGB")
        if a.size != b.size:
            rows.append((fb.stem, 100.0, "khác kích thước"))
            continue
        d = np.asarray(ImageChops.difference(a, b)).max(axis=2) > PIXEL_TOL
        pct = 100.0 * d.mean()
        if pct > 0:
            ys, xs = np.nonzero(d)
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            vis = Image.blend(a, b, 0.5)
            ImageDraw.Draw(vis).rectangle(box, outline=(255, 0, 0), width=3)
            side = Image.new("RGB", (a.width * 3, a.height))
            side.paste(a, (0, 0)); side.paste(b, (a.width, 0)); side.paste(vis, (a.width * 2, 0))
            side.save(out / fb.name)
            rows.append((fb.stem, pct, f"vùng khác {box}"))
        else:
            rows.append((fb.stem, 0.0, ""))
    changed = [r for r in rows if r[1] > 0]
    for name, pct, note in sorted(changed, key=lambda r: -r[1]):
        print(f"  {pct:6.2f}%  {name}  {note}")
    print(f"\n{len(rows)} case so sánh: {len(rows) - len(changed)} giống hệt, {len(changed)} khác. Ảnh diff (trước | sau | chồng): {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snap"); s.add_argument("--out", required=True); s.add_argument("--template")
    d = sub.add_parser("diff"); d.add_argument("base"); d.add_argument("new"); d.add_argument("--out")
    args = ap.parse_args()
    if args.cmd == "snap":
        snap(Path(args.out), args.template)
    else:
        diff(Path(args.base), Path(args.new), Path(args.out or f"{args.new}_diff"))


if __name__ == "__main__":
    main()
