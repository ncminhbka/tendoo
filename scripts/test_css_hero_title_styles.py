#!/usr/bin/env python3
"""
scripts/test_css_hero_title_styles.py

==================================================================================================
TENDOO AI - HƯỚNG "100% OVERLAY" (KHÔNG DIFFUSION VẼ CHỮ) — CSS HERO TITLE STYLE BENCH
==================================================================================================

WHY THIS SCRIPT?
  Nhánh song song với "hero text bằng diffusion" (đang được nghiên cứu riêng): thử nghiệm liệu
  Title chính (vốn luôn do FLUX.2 glyph injection @ t=10 vẽ với hiệu ứng 3D/kim loại/neon tích
  hợp ánh sáng thật) có thể thay hoàn toàn bằng HTML/CSS + Playwright hay không -- đúng thiết kế
  triệt để của PosterVerse (Stage 3 của họ KHÔNG BAO GIỜ để diffusion vẽ bất kỳ chữ nào, kể cả
  title -- xem lại phần đọc paper/code trước đó).

  Đây là phép thử quyết định: nếu CSS thuần không tạo được hiệu ứng đủ thuyết phục, hướng "100%
  overlay" thua ngay từ vòng gửi xe về mặt thẩm mỹ, bất kể latency/độ tin cậy tốt đến đâu.

3 STYLE PRESET (đều PURE CSS, không cần ảnh render sẵn nào khác):
  1. metallic_3d : Kim loại dập nổi 3D -- gradient fill + nhiều lớp text-shadow tạo độ đùn/extrude.
  2. neon_glow    : Neon phát sáng -- nhiều lớp text-shadow blur tăng dần tạo hiệu ứng bloom.
  3. gold_foil    : Nhũ vàng sang trọng -- gradient ấm + highlight mỏng + đổ bóng sâu.

Dùng ĐÚNG text "ÂM THANH ĐỈNH CAO" để so sánh trực tiếp, ngang hàng với sample diffusion đã có
sẵn: images/commercial_steps8_g1.5_seed123_576x1024.png

USAGE:
  python scripts/test_css_hero_title_styles.py --background images/commercial_steps8_g1.5_seed123_576x1024.png
  python scripts/test_css_hero_title_styles.py  # không có --background -> dùng nền giả lập tối tương tự
==================================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path
from string import Template

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


HERO_TEXT = "ÂM THANH<br>ĐỈNH CAO"
CANVAS_W, CANVAS_H = 576, 1024

# ==================================================================================================
# 3 CSS HERO TITLE STYLES -- pure text-shadow/gradient tricks, zero images/filters needed
# ==================================================================================================

STYLE_CSS = {
    "metallic_3d": """
        font-family: 'Montserrat', sans-serif;
        font-weight: 900;
        font-size: 64px;
        line-height: 1.15;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(180deg, #FFFFFF 0%, #D8D8D8 35%, #8A8A8A 55%, #C8C8C8 70%, #FFFFFF 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        /* Stacked text-shadow layers simulate a 3D extruded/embossed metal edge -- each layer
           offset by 1px darker than the last, building a "depth" illusion pure CSS can't get
           from a single shadow. */
        filter:
            drop-shadow(0px 1px 0px #b0b0b0)
            drop-shadow(0px 2px 0px #999999)
            drop-shadow(0px 3px 0px #808080)
            drop-shadow(0px 4px 0px #666666)
            drop-shadow(0px 5px 0px #4d4d4d)
            drop-shadow(0px 6px 2px rgba(0,0,0,0.5))
            drop-shadow(0px 10px 16px rgba(0,0,0,0.6));
    """,
    "neon_glow": """
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 64px;
        line-height: 1.15;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #FFFFFF;
        /* Bloom effect: tight bright core shadows + progressively larger/dimmer blurred layers
           in the neon hue, matching how real neon tube glow falls off with distance. */
        text-shadow:
            0 0 4px #FFFFFF,
            0 0 10px #FFFFFF,
            0 0 18px #00F0FF,
            0 0 34px #00F0FF,
            0 0 60px #00B8FF,
            0 0 90px #0080FF,
            0 2px 2px rgba(0,0,0,0.4);
    """,
    "gold_foil": """
        font-family: 'Playfair Display', serif;
        font-weight: 900;
        font-size: 60px;
        line-height: 1.2;
        text-align: center;
        letter-spacing: 1px;
        background: linear-gradient(180deg, #FFF6D8 0%, #F5D485 20%, #C9971F 45%, #FFE9A8 55%, #B8860B 75%, #FFF2C4 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        filter:
            drop-shadow(0px 1px 0px rgba(255,255,255,0.5))
            drop-shadow(0px 3px 4px rgba(0,0,0,0.5))
            drop-shadow(0px 8px 18px rgba(0,0,0,0.55));
    """,
}

PAGE_TPL = Template("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Playfair+Display:wght@800;900&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { width: 100vw; height: 100vh; }
  .poster {
    position: relative; width: ${w}px; height: ${h}px; overflow: hidden;
    $bg_css
  }
  .hero-zone {
    position: absolute; top: 6%; left: 5%; right: 5%;
  }
  .hero-title { $style_css }
  .style-tag {
    position: absolute; bottom: 3%; left: 5%;
    color: rgba(255,255,255,0.55); font-family: monospace; font-size: 13px;
  }
</style></head>
<body>
  <div class="poster">
    <div class="hero-zone"><div class="hero-title">${text}</div></div>
    <div class="style-tag">style: ${style_name} (pure CSS, no diffusion)</div>
  </div>
</body></html>""")


def build_bg_css(background_path: str | None) -> str:
    if background_path and Path(background_path).exists():
        data = base64.b64encode(Path(background_path).read_bytes()).decode("utf-8")
        ext = Path(background_path).suffix.lower().replace(".", "")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        return f"background-image: url('data:{mime};base64,{data}'); background-size: cover; background-position: center;"
    # Fallback: approximate the existing diffusion sample's dark studio-neon scene so the
    # comparison is at least tonally fair even without the real background.
    return (
        "background: radial-gradient(circle at 50% 55%, #163028 0%, #0a1512 55%, #050908 100%);"
    )


async def render_all(background_path: str | None, output_dir: str) -> None:
    from playwright.async_api import async_playwright

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bg_css = build_bg_css(background_path)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for style_name, style_css in STYLE_CSS.items():
            html = PAGE_TPL.substitute(
                w=CANVAS_W, h=CANVAS_H, bg_css=bg_css, style_css=style_css,
                text=HERO_TEXT, style_name=style_name,
            )
            page = await browser.new_page(viewport={"width": CANVAS_W, "height": CANVAS_H})
            await page.set_content(html, wait_until="networkidle")
            await page.evaluate("document.fonts.ready")
            out_file = out_path / f"hero_style_{style_name}.png"
            await page.screenshot(path=str(out_file))
            await page.close()
            print(f"  [✓] {style_name} -> {out_file}")
        await browser.close()


def main():
    parser = argparse.ArgumentParser(description="CSS-only Hero Title style bench (100%-overlay direction)")
    parser.add_argument("--background", type=str, default=None,
                         help="Path to an existing (with-or-without-text) background image to overlay on; "
                              "falls back to an approximated dark studio gradient if omitted")
    parser.add_argument("--output_dir", type=str, default="output_css_hero_title_styles")
    args = parser.parse_args()

    print("=" * 90)
    print(" [*] TENDOO AI - CSS HERO TITLE STYLE BENCH (100%-OVERLAY DIRECTION, NO DIFFUSION TEXT)")
    print("=" * 90)
    print(f"  Background : {args.background or '(approximated dark studio gradient)'}")
    print(f"  Text       : {HERO_TEXT.replace('<br>', ' / ')}")
    print(f"  Styles     : {list(STYLE_CSS.keys())}")
    asyncio.run(render_all(args.background, args.output_dir))
    print(f"\n[✓] Done. Compare visually against images/commercial_steps8_g1.5_seed123_576x1024.png\n")


if __name__ == "__main__":
    main()
