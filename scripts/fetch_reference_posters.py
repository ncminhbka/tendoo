#!/usr/bin/env python3
"""
scripts/fetch_reference_posters.py -- bộ POSTER THAM CHIẾU (ROADMAP §10.13 mục 3, H1).

Tải ảnh xem trước mẫu thiết kế công khai trên Canva (thư viện mẫu do designer làm) theo 6 nhu cầu
người dùng + menu + Tết, để làm thước đo so sánh GĐ 7–10 (đánh giá mù, lockup, style pack).

CHỈ DÙNG NỘI BỘ để tham chiếu thiết kế: ảnh thuộc bản quyền Canva/tác giả. Thư mục ảnh nằm trong
.gitignore -- KHÔNG đẩy lên GitHub, không dùng lại trong sản phẩm. manifest.json giữ link nguồn.

  python scripts/fetch_reference_posters.py --per-need 15
  -> references/posters/<nhu_cau>/<template_id>.jpg + references/posters/manifest.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "references" / "posters"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36"}

# nhu cầu -> trang thư viện mẫu (ưu tiên trang tiếng Việt; thiếu thì trang tiếng Anh cùng chủ đề)
SOURCES = {
    "promo": ["https://www.canva.com/vi_vn/mau/s/mau-chuong-trinh-khuyen-mai/", "https://www.canva.com/posters/templates/sale/", "https://www.canva.com/templates/s/sale/"],
    "product_intro": ["https://www.canva.com/vi_vn/mau/s/san-pham/", "https://www.canva.com/posters/templates/advertising/"],
    "opening": ["https://www.canva.com/flyers/templates/grand-opening/", "https://www.canva.com/banners/templates/grand-opening/"],
    "feedback": ["https://www.canva.com/templates/s/testimonial/"],
    "recruitment": ["https://www.canva.com/vi_vn/mau/s/tuyen-dung/"],
    "guide": ["https://www.canva.com/templates/s/how-to/", "https://www.canva.com/templates/s/tips/", "https://www.canva.com/vi_vn/mau/s/cham-soc-da/"],
    "menu": ["https://www.canva.com/vi_vn/menu/mau/", "https://www.canva.com/posters/templates/cafe/"],
    "festive_tet": ["https://www.canva.com/vi_vn/mau/s/poster-tet/"],
    "typographic": ["https://www.canva.com/posters/templates/typographic/"],
}
IMG_RE = re.compile(r"https://template\.canva\.com/([A-Za-z0-9_-]+)/\d+/\d+/(\d+)w-[A-Za-z0-9_-]+\.jpg")


def candidates(page_url: str) -> dict:
    """{template_id: url ảnh rộng nhất <= 1200px} theo thứ tự xuất hiện trên trang."""
    html = requests.get(page_url, headers=UA, timeout=60).text
    best: dict = {}
    for m in IMG_RE.finditer(html):
        tid, w = m.group(1), int(m.group(2))
        if w > 1200:
            continue
        if tid not in best or w > best[tid][0]:
            best[tid] = (w, m.group(0))
    return {k: v[1] for k, v in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-need", type=int, default=15)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mf = OUT / "manifest.json"
    manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else []
    # Duyệt bằng mắt: mẫu KHÔNG phải poster (lịch trình trống, phiếu bài tập, slide) -> blocklist, không tải lại.
    bl = PROJECT_ROOT / "references" / "poster_blocklist.json"
    blocked = set(json.loads(bl.read_text(encoding="utf-8"))) if bl.exists() else set()
    for m in [m for m in manifest if m["id"] in blocked]:
        (PROJECT_ROOT / m["file"]).unlink(missing_ok=True)
    manifest = [m for m in manifest if m["id"] not in blocked]
    have = {m["id"] for m in manifest} | blocked
    for need, pages in SOURCES.items():
        (OUT / need).mkdir(exist_ok=True)
        got = sum(1 for m in manifest if m["need"] == need)
        for page in pages:
            if got >= args.per_need:
                break
            try:
                cands = candidates(page)
            except requests.RequestException as e:
                print(f"  lỗi trang {page}: {e}")
                continue
            for tid, url in cands.items():
                if got >= args.per_need:
                    break
                if tid in have:
                    continue
                try:
                    data = requests.get(url, headers=UA, timeout=60).content
                    im = Image.open(io.BytesIO(data))
                    w, h = im.size
                except Exception as e:  # ảnh hỏng / không phải ảnh
                    print(f"  bỏ {tid}: {e}")
                    continue
                # Chỉ giữ tỉ lệ poster/social/banner (1:2 .. 2:1); bỏ trang tài liệu dài, dải quá mảnh.
                if not (0.5 <= w / h <= 2.0):
                    continue
                path = OUT / need / f"{tid}.jpg"
                path.write_bytes(data)
                manifest.append({"id": tid, "need": need, "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                                 "size": [w, h], "image_url": url, "source_page": page, "source": "Canva template preview",
                                 "license_note": "bản quyền Canva/tác giả -- chỉ tham chiếu nội bộ"})
                have.add(tid)
                got += 1
                time.sleep(0.3)
        print(f"{need}: {got}")
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"tổng {len(manifest)} -> {OUT}")


if __name__ == "__main__":
    main()
