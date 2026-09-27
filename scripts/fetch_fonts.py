#!/usr/bin/env python3
"""
scripts/fetch_fonts.py -- tải bộ font MỞ RỘNG (ROADMAP §10.7, GĐ 7: "4–6 font thư pháp/viết tay VN" + font nhiều độ đậm).

Chỉ lấy font giấy phép SIL OFL 1.1 từ kho chính thức google/fonts (được dùng thương mại, được nhúng vào ảnh/HTML;
khác bộ SVN-* chưa rõ giấy phép -- docs/FONT_LICENSES.md). Mỗi font:
  1. tải TTF gốc + file OFL.txt đi kèm (giữ bản quyền đúng điều kiện OFL),
  2. KIỂM đủ 134 ký tự dấu tiếng Việt bằng fontTools -- thiếu 1 ký tự là LOẠI (đã loại khi chọn: Caveat, Kaushan
     Script, Lobster Two, Bebas Neue, Bodoni Moda, DM Serif Display, Oleo Script, Archivo Black thiếu 68–76 ký tự),
  3. nén WOFF2 (không mất dữ liệu) để nhúng Base64 nhẹ hơn; TTF giữ lại cho kiểm thử/fontTools.

  python scripts/fetch_fonts.py            # tải thiếu
  python scripts/fetch_fonts.py --force    # tải lại tất cả
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import requests
from fontTools.ttLib import TTFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = PROJECT_ROOT / "fonts"
RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl/{dir}/{name}"

VN_CHARS = ("ĐđĂăÂâÊêÔôƠơƯư"
            "ẤẦẨẪẬẮẰẲẴẶẾỀỂỄỆỐỒỔỖỘỚỜỞỠỢỨỪỬỮỰỲỸỶỴÝ"
            "ấầẩẫậắằẳẵặếềểễệốồổỗộớờởỡợứừửữựỳỹỷỵý"
            "ÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụ")

# (thư mục google/fonts/ofl, file TTF nguồn, tên file lưu local không có đuôi)
FONTS = [
    ("greatvibes", "GreatVibes-Regular.ttf", "GreatVibes-Regular"),
    ("alexbrush", "AlexBrush-Regular.ttf", "AlexBrush-Regular"),
    ("lobster", "Lobster-Regular.ttf", "Lobster-Regular"),
    ("pattaya", "Pattaya-Regular.ttf", "Pattaya-Regular"),
    ("charm", "Charm-Bold.ttf", "Charm-Bold"),
    ("sriracha", "Sriracha-Regular.ttf", "Sriracha-Regular"),
    ("montserrat", "Montserrat[wght].ttf", "Montserrat"),
    ("cormorant", "Cormorant[wght].ttf", "Cormorant"),
    ("barlowcondensed", "BarlowCondensed-Black.ttf", "BarlowCondensed-Black"),
]


def missing_vn(font: TTFont) -> list[str]:
    cmap = font.getBestCmap()
    return [ch for ch in VN_CHARS if ord(ch) not in cmap]


def fetch(dir_: str, src: str, stem: str, force: bool) -> bool:
    ttf, woff2 = FONTS_DIR / f"{stem}.ttf", FONTS_DIR / f"{stem}.woff2"
    if ttf.exists() and woff2.exists() and not force:
        print(f"  = {stem} (đã có)")
        return True
    data = requests.get(RAW.format(dir=dir_, name=src.replace("[", "%5B").replace("]", "%5D")), timeout=60)
    data.raise_for_status()
    font = TTFont(io.BytesIO(data.content))
    miss = missing_vn(font)
    if miss:
        print(f"  ✗ {stem}: thiếu {len(miss)} ký tự tiếng Việt ({''.join(miss[:12])}...) -- LOẠI")
        return False
    lic = requests.get(RAW.format(dir=dir_, name="OFL.txt"), timeout=60)
    lic.raise_for_status()
    ttf.write_bytes(data.content)
    (FONTS_DIR / f"{stem.split('-')[0]}-OFL.txt").write_text(lic.text, encoding="utf-8")
    font.flavor = "woff2"
    font.save(str(woff2))
    print(f"  ✓ {stem}: {len(data.content) // 1024}KB TTF -> {woff2.stat().st_size // 1024}KB WOFF2, đủ dấu tiếng Việt")
    return True


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    FONTS_DIR.mkdir(exist_ok=True)
    ok = [fetch(d, s, n, args.force) for d, s, n in FONTS]
    print(f"{sum(ok)}/{len(ok)} font sẵn sàng trong {FONTS_DIR}")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
