#!/usr/bin/env python3
"""
scripts/build_blind_review.py -- trang ĐÁNH GIÁ MÙ THEO CẶP (ROADMAP §10.10: thước đo chính của GĐ 7–10).

Mỗi cặp: 1 poster của hệ thống (lượt GPT thật, `run_real_llm.py --images`) và 1 poster tham chiếu do designer làm
(bộ `references/posters`, cùng NHU CẦU, khung gần nhất). Trái/phải xáo ngẫu nhiên có seed; tên ảnh là mã băm; cả hai
nén cùng kiểu JPEG, cùng chiều cao -> người chấm không đoán nguồn qua tên file / định dạng / độ nét.

Người chấm mở trang, bấm "Trái đẹp hơn / Ngang nhau / Phải đẹp hơn" (phím ← ↓ →). Phiếu lưu trong trình duyệt và
tải về bằng nút "Tải kết quả" -> đặt file JSON vào references/blind_review/results/ -> scripts/score_blind_review.py.

CHỈ DÙNG NỘI BỘ: thư mục references/ nằm trong .gitignore (ảnh tham chiếu thuộc bản quyền Canva/tác giả).

  python scripts/build_blind_review.py                                  # lượt mới nhất, 1 tham chiếu / poster
  python scripts/build_blind_review.py --run output_probe/real_llm/gpt-5.4-mini_v9 --refs-per-poster 2

GIỚI HẠN ĐÃ BIẾT (ghi vào kết quả, không giấu): poster tham chiếu KHÔNG cùng brief (khác nội dung, nhiều mẫu tiếng
Anh) -- người chấm được dặn chấm THIẾT KẾ, nhưng ngôn ngữ vẫn có thể lộ nguồn. Tỉ lệ đo được là cận dưới thô.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = PROJECT_ROOT / "references" / "posters"
OUT = PROJECT_ROOT / "references" / "blind_review"
MAX_H = 1000

# brief (tests/real_briefs.json) -> nhu cầu của bộ tham chiếu (fetch_reference_posters.SOURCES)
NEED_BY_KEYWORD = [
    ("textonly_tet", "festive_tet"), ("notice_tet", "festive_tet"), ("greeting", "festive_tet"),
    ("textonly", "typographic"), ("flash_sale", "promo"), ("promo", "promo"), ("combo", "promo"),
    ("minigame", "promo"), ("before_after", "promo"), ("notice", "opening"), ("opening", "opening"),
    ("vip_invite", "opening"), ("workshop", "opening"), ("feedback", "feedback"), ("recruit", "recruitment"),
    ("menu", "menu"), ("guide", "guide"), ("luxury", "product_intro"), ("product", "product_intro"),
    ("pod_", "product_intro"),
]


def need_of(brief_id: str) -> str:
    return next((n for k, n in NEED_BY_KEYWORD if k in brief_id), "promo")


def latest_run() -> Path:
    runs = [p for p in (PROJECT_ROOT / "output_probe" / "real_llm").glob("gpt-*") if list(p.glob("poster_*.png"))]
    return max(runs, key=lambda p: max(f.stat().st_mtime for f in p.glob("poster_*.png")))


def to_jpeg(src: Path, dst_dir: Path, salt: str) -> str:
    im = Image.open(src).convert("RGB")
    if im.height > MAX_H:
        im = im.resize((round(im.width * MAX_H / im.height), MAX_H), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    name = hashlib.sha1((salt + src.name).encode()).hexdigest()[:12] + ".jpg"
    (dst_dir / name).write_bytes(buf.getvalue())
    return name


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None, help="thư mục lượt GPT thật có poster_*.png (mặc định: mới nhất)")
    ap.add_argument("--refs-per-poster", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    run = (args.run or latest_run()).resolve()
    ours = sorted(run.glob("poster_*.png"))
    refs = json.loads((REF_DIR / "manifest.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    img_dir = OUT / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.jpg"):
        old.unlink()
    (OUT / "results").mkdir(exist_ok=True)

    used: dict[str, int] = {}
    pairs, key = [], {}
    for f in ours:
        bid = f.stem.removeprefix("poster_")
        need = need_of(bid)
        w, h = Image.open(f).size
        pool = [r for r in refs if r["need"] == need and (PROJECT_ROOT / r["file"]).exists()]
        # ưu tiên tham chiếu ít dùng nhất, rồi khung gần nhất (tỉ lệ khung lộ nguồn nếu lệch nhiều)
        pool.sort(key=lambda r: (used.get(r["id"], 0), abs(r["size"][0] / r["size"][1] - w / h), rng.random()))
        for r in pool[: args.refs_per_poster]:
            used[r["id"]] = used.get(r["id"], 0) + 1
            pid = f"p{len(pairs) + 1:03d}"
            a = to_jpeg(f, img_dir, pid)
            b = to_jpeg(PROJECT_ROOT / r["file"], img_dir, pid)
            ours_left = rng.random() < 0.5
            pairs.append({"id": pid, "left": a if ours_left else b, "right": b if ours_left else a})
            key[pid] = {"ours": "left" if ours_left else "right", "brief": bid, "need": need, "ref": r["id"],
                        "ref_source": r.get("image_url")}
    rng.shuffle(pairs)
    (OUT / "key.json").write_text(json.dumps({"run": str(run.relative_to(PROJECT_ROOT)), "seed": args.seed, "pairs": key},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    page = PAGE.replace("__PAIRS__", json.dumps(pairs)).replace("__ROUND__", hashlib.sha1(json.dumps(pairs).encode()).hexdigest()[:8])
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(pairs)} cặp từ {run.name} -> {OUT / 'index.html'}")
    print("Mở file index.html bằng trình duyệt; chấm xong bấm 'Tải kết quả' và đặt file vào references/blind_review/results/")
    return 0


PAGE = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chấm poster mù</title>
<style>
  :root { --bg:#1d1f24; --panel:#272a31; --text:#eceef2; --muted:#9aa1ad; --accent:#e9eef7; --line:#3a3e47; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.5 system-ui, "Segoe UI", sans-serif; }
  header { max-width:1200px; margin:0 auto; padding:16px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; }
  h1 { font-size:18px; margin:0; font-weight:650; }
  .muted { color:var(--muted); font-size:14px; }
  .bar { height:4px; background:var(--line); max-width:1200px; margin:0 auto; }
  .bar > i { display:block; height:100%; background:var(--accent); width:0; transition:width .2s; }
  main { max-width:1200px; margin:0 auto; padding:16px; }
  .pair { display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:center; }
  .pair figure { margin:0; background:var(--panel); border-radius:8px; padding:8px; display:flex; justify-content:center; align-items:center; height:min(72vh, 760px); }
  .pair img { max-width:100%; max-height:100%; object-fit:contain; display:block; border-radius:4px; }
  .choices { display:grid; grid-template-columns:1fr auto 1fr; gap:12px; margin-top:16px; }
  button { font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 18px; cursor:pointer; }
  button:hover, button:focus-visible { border-color:var(--accent); outline:none; }
  button.primary { background:var(--accent); color:#16181d; border-color:var(--accent); font-weight:650; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
  input { font:inherit; background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px 12px; }
  .done { text-align:center; padding:48px 16px; }
  @media (max-width:700px) { .pair { grid-template-columns:1fr; } .pair figure { height:auto; } .choices { grid-template-columns:1fr; } }
</style></head>
<body>
<header>
  <div><h1>Poster nào đẹp hơn?</h1>
    <div class="muted">Chấm THIẾT KẾ (bố cục, chữ, màu, độ bắt mắt), bỏ qua nội dung và ngôn ngữ. Phím: ← trái · ↓ ngang nhau · → phải · Backspace quay lại.</div></div>
  <div class="row"><input id="rater" placeholder="Tên người chấm" aria-label="Tên người chấm"><span id="count" class="muted"></span></div>
</header>
<div class="bar"><i id="prog"></i></div>
<main id="app"></main>
<script>
const PAIRS = __PAIRS__, ROUND = "__ROUND__", KEY = "tendoo_blind_" + ROUND;
let st = { votes: {}, rater: "" };
try { st = JSON.parse(localStorage.getItem(KEY)) || st; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const rater = document.getElementById("rater");
rater.value = st.rater || ""; rater.oninput = () => { st.rater = rater.value.trim(); save(); };
const next = () => PAIRS.findIndex(p => !(p.id in st.votes));
function vote(p, v) { st.votes[p.id] = { v, t: Date.now() }; save(); render(); }
function undo() { const ids = Object.keys(st.votes).sort((a, b) => st.votes[b].t - st.votes[a].t); if (ids.length) { delete st.votes[ids[0]]; save(); render(); } }
function download() {
  const blob = new Blob([JSON.stringify({ round: ROUND, rater: st.rater || "an_danh", votes: st.votes }, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `blind_${ROUND}_${(st.rater || "an_danh").replace(/\W+/g, "_")}.json`; a.click();
}
function render() {
  const n = Object.keys(st.votes).length, i = next(), app = document.getElementById("app");
  document.getElementById("count").textContent = `${n}/${PAIRS.length}`;
  document.getElementById("prog").style.width = (100 * n / PAIRS.length) + "%";
  if (i < 0) {
    app.innerHTML = `<div class="done"><h2>Xong ${PAIRS.length} cặp. Cảm ơn bạn!</h2>
      <p class="muted">Bấm để tải file kết quả, rồi gửi file đó cho người phụ trách.</p>
      <div class="row" style="justify-content:center"><button class="primary" id="dl">Tải kết quả</button><button id="back">Quay lại cặp trước</button></div></div>`;
    document.getElementById("dl").onclick = download; document.getElementById("back").onclick = undo; return;
  }
  const p = PAIRS[i];
  app.innerHTML = `<div class="pair"><figure><img src="img/${p.left}" alt="Poster trái"></figure><figure><img src="img/${p.right}" alt="Poster phải"></figure></div>
    <div class="choices"><button id="l">← Trái đẹp hơn</button><button id="t">Ngang nhau</button><button id="r">Phải đẹp hơn →</button></div>
    <div class="row" style="margin-top:12px;justify-content:space-between"><button id="u">Quay lại</button><button id="dl">Tải kết quả (${n} phiếu)</button></div>`;
  document.getElementById("l").onclick = () => vote(p, "left");
  document.getElementById("t").onclick = () => vote(p, "tie");
  document.getElementById("r").onclick = () => vote(p, "right");
  document.getElementById("u").onclick = undo; document.getElementById("dl").onclick = download;
  const nx = PAIRS[next() + 1]; if (nx) { new Image().src = "img/" + nx.left; new Image().src = "img/" + nx.right; }
}
document.addEventListener("keydown", e => {
  if (e.target === rater) return;
  const i = next(); if (e.key === "Backspace") { e.preventDefault(); undo(); return; }
  if (i < 0) return;
  const m = { ArrowLeft: "left", ArrowDown: "tie", ArrowRight: "right" }[e.key];
  if (m) { e.preventDefault(); vote(PAIRS[i], m); }
});
render();
</script>
</body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
