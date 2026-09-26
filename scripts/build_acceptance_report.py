#!/usr/bin/env python3
"""
scripts/build_acceptance_report.py -- trang NGHIỆM THU (HTML tĩnh) từ một lượt run_real_llm --images.

Gom poster theo 6 NHU CẦU người dùng (6 loại form: khuyến mãi, giới thiệu sản phẩm, khai trương, đánh giá,
tuyển dụng, hướng dẫn/quy trình), kèm điểm squint 5 điều kiện của từng poster và tổng hợp.

  PYTHONPATH=src python scripts/build_acceptance_report.py --tag gpt-5.4-mini_v5
  -> output_probe/nghiem_thu/index.html (mở bằng trình duyệt; ảnh là đường dẫn tương đối)
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEEDS = [
    ("promo", "Khuyến mãi / giảm giá"),
    ("product_intro", "Giới thiệu sản phẩm"),
    ("opening", "Khai trương / sự kiện"),
    ("feedback", "Đánh giá khách hàng"),
    ("recruitment", "Tuyển dụng"),
    ("guide", "Hướng dẫn / quy trình"),
]
CONDS = [("c1_anchor", "Tiêu đề nổi bật"), ("c2_no_wall", "Không tường chữ"), ("c3_bg", "Tương phản nền"),
         ("c4_no_loss", "Không mất chữ"), ("c5_phone", "Đọc được trên điện thoại")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="thư mục trong output_probe/real_llm/")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output_probe" / "nghiem_thu"))
    args = ap.parse_args()
    src = PROJECT_ROOT / "output_probe" / "real_llm" / args.tag
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    report = {r["id"]: r for r in json.loads((src / "report.json").read_text(encoding="utf-8"))}
    briefs = json.loads((PROJECT_ROOT / "tests" / "real_briefs.json").read_text(encoding="utf-8"))

    total = {k: 0 for k, _ in CONDS}
    n_all = n_pass = 0
    sections = []
    for need, label in NEEDS:
        cards = []
        for b in briefs:
            if b["form"].get("category") != need or b["id"] not in report:
                continue
            r = report[b["id"]]
            img = src / f"poster_{b['id']}.png"
            if not img.exists():
                continue
            shutil.copy(img, out / "img" / img.name)
            marks = "".join(f'<li class="{"ok" if r.get(k) else "no"}">{"✓" if r.get(k) else "✗"} {html.escape(t)}</li>' for k, t in CONDS)
            ok_all = all(r.get(k) for k, _ in CONDS)
            n_all += 1
            n_pass += ok_all
            for k, _ in CONDS:
                total[k] += bool(r.get(k))
            comps = ", ".join(f"{k}={v}" for k, v in (r.get("components") or {}).items())
            cards.append(f'''<figure class="card{' pass' if ok_all else ''}">
  <img src="img/{img.name}" alt="{html.escape(b['id'])}" loading="lazy">
  <figcaption><b>{html.escape(b['id'])}</b> · {html.escape(b['aspect'])} · <code>{html.escape(r.get('template', ''))}</code>
  <div class="brief">{html.escape(b['prompt'])}</div>
  <ul>{marks}</ul>
  <div class="meta">intent: {html.escape(str(r.get('intent')))} · tương phản điểm neo {r.get('contrast')} (cần {r.get('contrast_target')}){' · ' + html.escape(comps) if comps else ''}{' · maskless' if r.get('maskless') else ''}</div>
  </figcaption></figure>''')
        sections.append(f'<section><h2>{html.escape(label)} <small>({len(cards)} poster)</small></h2><div class="grid">{"".join(cards) or "<p>Chưa có poster.</p>"}</div></section>')

    summary = "".join(f"<li>{html.escape(t)}: <b>{total[k]}/{n_all}</b></li>" for k, t in CONDS)
    page = f'''<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nghiệm thu Tendoo v3</title><style>
:root {{ --bg:#f6f7f9; --fg:#111827; --muted:#6b7280; --card:#fff; --ok:#15803d; --no:#b91c1c; --line:#e5e7eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0b0f19; --fg:#e5e7eb; --muted:#9ca3af; --card:#111827; --line:#1f2937; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px; }}
h1 {{ margin:0 0 4px; }} h2 {{ margin:32px 0 12px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
small, .meta, .brief {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:16px; }}
.card {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
.card.pass {{ border-color:var(--ok); }}
.card img {{ width:100%; display:block; background:#000; }}
figcaption {{ padding:10px 12px; font-size:13px; }}
ul {{ list-style:none; padding:0; margin:6px 0; }} li.ok {{ color:var(--ok); }} li.no {{ color:var(--no); }}
.summary {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; }}
</style></head><body><main>
<h1>Nghiệm thu Tendoo v3 — 6 nhu cầu người dùng</h1>
<p class="brief">LLM lập plan: {html.escape(args.tag)} · ảnh nền: mô hình ảnh GPT (thay FLUX để kiểm tầng chữ) · chữ: HTML/CSS overlay.</p>
<div class="summary"><b>Đạt cả 5 điều kiện: {n_pass}/{n_all} poster</b><ul>{summary}</ul></div>
{"".join(sections)}
</main></body></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{n_pass}/{n_all} đạt cả 5 -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
