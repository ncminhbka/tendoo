#!/usr/bin/env python3
"""
scripts/run_e2e_server.py -- TEST END-TO-END máy chủ demo đang chạy, thay cho bấm tay trên giao diện.

Gửi lần lượt các mẫu trong tests/e2e_server_cases.json tới POST /api/generate (đúng JSON giao diện gửi), rồi với mỗi
mẫu: tải poster + ảnh nền, đọc trạng thái LLM của lượt đó (/api/v3/runs/<run_id>/llm-debug), chấm squint 6 điều kiện
trên nền THẬT (probe_type_hierarchy.measure_plan, như GĐ 3R), thử SỬA CHỮ trên nền cũ (/api/v3/retext, như nút "Cập nhật
chữ" của giao diện), và gom tất cả vào 1 trang HTML xem nhanh.

  # máy chủ demo đang chạy (docs/HUONG_DAN_SERVER.md bước 5), ở terminal khác:
  cd ~/work/tendoo-v3 && python scripts/run_e2e_server.py                     # 20 mẫu, ~2-5 phút với bản distill
  python scripts/run_e2e_server.py --only e02,e18 --url http://127.0.0.1:8088
  python scripts/run_e2e_server.py --no-squint                                # chỉ gọi API, không chấm

Kết quả: output_probe/e2e_<thời gian>/index.html (+ results.json, poster_*.png, bg_*.png). Mã thoát 0 = mọi mẫu sinh
được poster bằng GPU thật + LLM thật; 1 = có mẫu lỗi / mock / plan dự phòng (bảng cuối ghi rõ).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    sys.path.insert(0, str(p))

COND = [("c1_anchor", "C1 tiêu đề áp đảo"), ("c2_no_wall", "C2 không tường chữ"), ("c3_bg", "C3 tương phản nền"),
        ("c4_no_loss", "C4 không mất chữ"), ("c5_phone", "C5 đọc được trên điện thoại"), ("c6_room", "C6 không phí chỗ")]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--cases", default=str(PROJECT_ROOT / "tests" / "e2e_server_cases.json"))
    ap.add_argument("--only", default=None, help="danh sách tiền tố id, phẩy ngăn cách (vd e02,e18)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=600, help="giây chờ mỗi mẫu")
    ap.add_argument("--no-squint", action="store_true")
    ap.add_argument("--no-retext", action="store_true", help="bỏ bước sửa chữ trên nền cũ (/api/v3/retext)")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    try:
        health = requests.get(f"{base}/api/health", timeout=10).json()
    except Exception as e:
        print(f"Không kết nối được {base} ({type(e).__name__}) -- máy chủ demo đã chạy chưa? (docs/HUONG_DAN_SERVER.md bước 5)")
        return 1
    mock = bool(health.get("mock_mode"))
    print(f"Máy chủ: model={health.get('model_name')} mock_mode={mock} gpu={health.get('gpu_names')} llm={health.get('active_llm')}")
    if mock:
        print("  CHÚ Ý: mock_mode=true -> ảnh nền là ảnh GIẢ (xem demo.log, mục Sự cố trong hướng dẫn). Vẫn chạy để thử luồng.")

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    if args.only:
        pre = [x.strip() for x in args.only.split(",") if x.strip()]
        cases = [c for c in cases if any(c["id"].startswith(p) for p in pre)]
    out = Path(args.out or PROJECT_ROOT / "output_probe" / f"e2e_{datetime.now().strftime('%m%d_%H%M')}")
    out.mkdir(parents=True, exist_ok=True)

    page = pw = browser = None
    if not args.no_squint:
        from playwright.sync_api import sync_playwright
        from probe_type_hierarchy import measure_plan
        from tendoo_v3.schema import TendooCreativePlan
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page()

    rows = []
    for i, c in enumerate(cases, 1):
        row = {"id": c["id"], "need": c.get("need", ""), "ui": c.get("ui", True), "aspect": c["request"].get("aspect_ratio", "1:1")}
        print(f"[{i}/{len(cases)}] {c['id']} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = requests.post(f"{base}/api/generate", json=c["request"], timeout=args.timeout)
            row["http"] = r.status_code
            d = r.json()
            if r.status_code != 200 or d.get("status") != "success":
                raise RuntimeError(json.dumps(d.get("detail", d), ensure_ascii=False)[:300])
        except Exception as e:
            row.update(ok=False, error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1))
            print(f"LỖI {row['error'][:120]}")
            rows.append(row)
            continue
        row.update(ok=True, seconds=d.get("elapsed_seconds", round(time.time() - t0, 1)), run_id=d.get("run_id"),
                   template=d.get("resolved_layout"), n_posters=len(d.get("posters") or []))
        plan = d.get("plan") or {}
        row["hero"] = plan.get("hero", "")
        try:
            tr = requests.get(f"{base}/api/v3/runs/{row['run_id']}/llm-debug", timeout=10).json()
            row.update(llm_status=tr.get("status"), llm_mode=tr.get("mode"), llm_latency=tr.get("latency_seconds"),
                       llm_error=str(tr.get("error") or "")[:200], revision=(tr.get("revision") or {}).get("adopted"))
        except Exception:
            row["llm_status"] = "?"
        for kind, key in (("poster", "final_poster_url"), ("bg", "blended_bg_url")):
            img = requests.get(f"{base}/{d[key]}", timeout=30)
            (out / f"{kind}_{c['id']}.png").write_bytes(img.content)
        if page is not None:
            try:
                bg = "data:image/png;base64," + base64.b64encode((out / f"bg_{c['id']}.png").read_bytes()).decode()
                p = TendooCreativePlan.from_dict(plan)
                m = measure_plan(page, p, int(d.get("width")), int(d.get("height")), with_bg=True, bg_override=bg)
                row.update({k: bool(m.get(k)) for k, _ in COND}, contrast=m.get("contrast"), target=m.get("contrast_target"),
                           pass6=bool(m.get("pass6")), text_lost=m.get("text_lost"))
            except Exception as e:
                row["squint_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        if not args.no_retext:
            _retext(base, row, plan, out, c["id"])
        flags = "" if page is None else " squint " + "".join("✓" if row.get(k) else "✗" for k, _ in COND)
        rt = "" if args.no_retext else f"  sửa chữ={'OK' if row.get('retext_ok') else 'LỖI'}"
        print(f"{row['seconds']}s  {row['template']}  LLM={row.get('llm_status')}/{row.get('llm_mode')}{flags}{rt}")
        rows.append(row)

    if browser:
        browser.close()
        pw.stop()

    ok = [r for r in rows if r.get("ok")]
    llm_ok = [r for r in ok if r.get("llm_status") == "success"]
    summary = {"server": health, "n": len(rows), "ok": len(ok), "llm_success": len(llm_ok), "mock_mode": mock,
               "avg_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 1) if ok else None,
               "pass6": sum(bool(r.get("pass6")) for r in ok) if page is not None else None,
               "retext_ok": None if args.no_retext else sum(bool(r.get("retext_ok")) for r in ok)}
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "index.html").write_text(_page(summary, rows, page is not None), encoding="utf-8")

    print(f"\n{len(ok)}/{len(rows)} mẫu ra poster | LLM thật {len(llm_ok)}/{len(ok)} | trung bình {summary['avg_seconds']}s/mẫu"
          + (f" | đạt cả 6 điều kiện {summary['pass6']}/{len(ok)}" if page is not None else "")
          + ("" if args.no_retext else f" | sửa chữ trên nền cũ {summary['retext_ok']}/{len(ok)}") + (" | MOCK (nền giả)" if mock else ""))
    for r in rows:
        if not r.get("ok"):
            print(f"  ✗ {r['id']}: {r.get('error')}")
        elif not args.no_retext and not r.get("retext_ok"):
            print(f"  ✗ {r['id']}: sửa chữ lỗi {r.get('retext_error', '')[:160]}")
        elif r.get("llm_status") != "success":
            print(f"  ! {r['id']}: LLM {r.get('llm_status')} ({r.get('llm_mode')}) {r.get('llm_error', '')[:120]}")
    print(f"Xem: {out / 'index.html'}")
    retext_bad = 0 if args.no_retext else len(ok) - summary["retext_ok"]
    return 0 if (len(ok) == len(rows) and len(llm_ok) == len(ok) and not mock and not retext_bad) else 1


RETEXT_TRY = (("cta", "XEM NGAY"), ("badge", "ƯU ĐÃI"), ("subhead", "Ưu đãi có hạn, đặt ngay hôm nay"))


def _retext(base: str, row: dict, plan: dict, out: Path, cid: str) -> None:
    """Sửa 1 dòng chữ có sẵn + xin thêm 1 kiểu chữ trên nền cũ, như nút "Cập nhật chữ" / "Thêm kiểu chữ" của giao diện."""
    key, val = next(((k, v) for k, v in RETEXT_TRY if plan.get(k)), ("hero", (plan.get("hero") or "") + " MỚI"))
    t0 = time.time()
    try:
        r = requests.post(f"{base}/api/v3/retext", timeout=120,
                          json={"run_id": row["run_id"], "variant_idx": 0, "edits": {key: val}, "typography_variants": 2})
        d = r.json()
        if r.status_code != 200:
            raise RuntimeError(json.dumps(d.get("detail", d), ensure_ascii=False)[:200])
        ps = d.get("posters") or []
        ok = len(ps) == 2 and (ps[0].get("plan") or {}).get(key) == val
        (out / f"retext_{cid}.png").write_bytes(requests.get(f"{base}/{ps[0]['url']}", timeout=30).content)
        row.update(retext_ok=ok, retext_field=key, retext_seconds=round(time.time() - t0, 1),
                   retext_text_lost=any(p.get("text_lost") for p in ps))
    except Exception as e:
        row.update(retext_ok=False, retext_error=f"{type(e).__name__}: {e}")


def _page(s: dict, rows: list, squint: bool) -> str:
    cards = []
    for r in rows:
        e = html.escape
        if not r.get("ok"):
            cards.append(f'<div class="card bad"><h3>{e(r["id"])}</h3><p>{e(r.get("error", ""))}</p></div>')
            continue
        marks = "".join(f'<span class="{"y" if r.get(k) else "n"}" title="{e(label)}">{k[:2].upper()}</span>' for k, label in COND) if squint else ""
        llm = r.get("llm_status")
        cards.append(f'''<div class="card"><h3>{e(r["id"])}</h3>
<div class="imgs"><img src="poster_{e(r["id"])}.png" alt="poster"><div class="side"><img src="bg_{e(r["id"])}.png" alt="nền" title="ảnh nền">{f'<img src="retext_{e(r["id"])}.png" alt="đã sửa chữ" title="sửa chữ trên nền cũ">' if r.get("retext_ok") else ""}</div></div>
<p><b>{e(r.get("need", ""))}</b> · {e(r["aspect"])} · {e(str(r.get("template")))} · {r["seconds"]}s{"" if r.get("ui") else " · qua API prompt"}</p>
<p class="{"" if llm == "success" else "warn"}">LLM: {e(str(llm))} / {e(str(r.get("llm_mode")))}{" · đã rút gọn" if r.get("revision") else ""}{(" · " + e(r.get("llm_error", ""))) if llm != "success" else ""}</p>
<p class="{"" if r.get("retext_ok", True) else "warn"}">{("Sửa chữ (" + e(str(r.get("retext_field"))) + "): " + ("OK " + str(r.get("retext_seconds")) + "s" + (" · CẮT CHỮ" if r.get("retext_text_lost") else "") if r.get("retext_ok") else "LỖI " + e(r.get("retext_error", "")))) if "retext_ok" in r else ""}</p>
<p class="marks">{marks}{(" tỉ lệ " + str(r.get("contrast")) + "/" + str(r.get("target"))) if squint else ""}</p></div>''')
    head = (f'{s["ok"]}/{s["n"]} mẫu ra poster · LLM thật {s["llm_success"]}/{s["ok"]} · trung bình {s["avg_seconds"]}s'
            + (f' · đạt cả 6 điều kiện {s["pass6"]}/{s["ok"]}' if squint else "") + (" · <b>MOCK: nền giả</b>" if s["mock_mode"] else ""))
    return f'''<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E2E máy chủ demo</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--text:#1d2129;--muted:#667;--bad:#c62828;--ok:#2e7d32}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181d;--card:#22252c;--text:#e8eaed;--muted:#9aa0a6}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.card{{background:var(--card);border-radius:10px;padding:10px}} .card.bad{{border:2px solid var(--bad)}}
.card h3{{margin:0 0 6px;font-size:14px}} .imgs{{display:flex;gap:6px;align-items:flex-start}}
.imgs>img{{width:66%;border-radius:6px}} .side{{width:32%;display:flex;flex-direction:column;gap:6px}} .side img{{width:100%;border-radius:6px}}
p{{margin:4px 0;color:var(--muted)}} .warn{{color:var(--bad)}} .marks span{{display:inline-block;margin-right:3px;padding:0 4px;border-radius:4px;font-size:11px;color:#fff}}
.y{{background:var(--ok)}} .n{{background:var(--bad)}}
</style></head><body><h2>E2E máy chủ demo · {html.escape(str(s["server"].get("model_name")))}</h2><p>{head}</p><div class="grid">{"".join(cards)}</div></body></html>'''


if __name__ == "__main__":
    sys.exit(main())
