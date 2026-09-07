#!/usr/bin/env python3
"""
scripts/render_html_posters.py

==================================================================================================
TENDOO AI - STANDALONE HTML-TO-POSTER VECTOR RENDERER (PLAYWRIGHT CHROMIUM)
==================================================================================================

OBJECTIVE:
  Renders any `03_poster_template.html` files into pixel-perfect PNG posters using Playwright Chromium.
  Can be run on ANY machine with Playwright (such as local Windows PC or server with Chromium).
  Also regenerates the 3-panel comparison strip `04_comparison_strip.png`.

USAGE:
  # Render all cases inside output_demo_promo_posters:
  python scripts/render_html_posters.py --dir output_demo_promo_posters

  # Render a specific case or HTML file:
  python scripts/render_html_posters.py --html output_demo_promo_posters/case1_mid_autumn/03_poster_template.html
==================================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont
from tendoo.typography_engine import PosterRenderer


def get_fallback_font(size: int = 24, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_single_html(html_file: Path, width: int = 576, height: int = 1024) -> Path:
    case_dir = html_file.parent
    final_poster_path = case_dir / "03_final_poster.png"

    print(f"\n🎨 Rendering: {html_file.resolve()}...")
    html_content = html_file.read_text(encoding="utf-8")

    PosterRenderer.render(
        html_content=html_content,
        output_image_path=final_poster_path,
        width=width,
        height=height,
    )
    print(f"  [✓] Vector poster rendered -> {final_poster_path.name}")

    # Update comparison strip if baseline and cleaned background exist
    baseline_path = case_dir / "01_baseline_raw.png"
    cleaned_bg_path = case_dir / "02_cleaned_background.png"

    if baseline_path.exists() and cleaned_bg_path.exists():
        baseline_pil = Image.open(baseline_path).convert("RGB")
        cleaned_bg_pil = Image.open(cleaned_bg_path).convert("RGB")
        final_poster_pil = Image.open(final_poster_path).convert("RGB")

        w, h = baseline_pil.size
        comp = Image.new("RGB", (w * 3, h), (18, 18, 18))
        comp.paste(baseline_pil, (0, 0))
        comp.paste(cleaned_bg_pil, (w, 0))
        comp.paste(final_poster_pil, (w * 2, 0))

        draw = ImageDraw.Draw(comp)
        font = get_fallback_font(size=26, bold=True)

        draw.rectangle([(10, 10), (250, 48)], fill=(0, 0, 0, 200))
        draw.text((20, 14), "1. RAW BASELINE", fill=(255, 255, 255), font=font)

        draw.rectangle([(w + 10, 10), (w + 290, 48)], fill=(0, 0, 0, 200))
        draw.text((w + 20, 14), "2. CLEANED COPY SPACE", fill=(255, 200, 50), font=font)

        draw.rectangle([(w * 2 + 10, 10), (w * 2 + 300, 48)], fill=(0, 0, 0, 200))
        draw.text((w * 2 + 20, 14), "3. PLAYWRIGHT VECTOR", fill=(80, 255, 80), font=font)

        comp_path = case_dir / "04_comparison_strip.png"
        comp.save(comp_path)
        print(f"  [✓] Updated comparison strip with real vector typography -> {comp_path.name}")

    return final_poster_path


def main():
    parser = argparse.ArgumentParser(description="Render poster HTML templates using Playwright Chromium")
    parser.add_argument("--dir", type=str, default=None, help="Root directory containing case folders with 03_poster_template.html")
    parser.add_argument("--html", type=str, default=None, help="Path to a single 03_poster_template.html file")
    parser.add_argument("--width", type=int, default=576, help="Viewport width")
    parser.add_argument("--height", type=int, default=1024, help="Viewport height")
    args = parser.parse_args()

    if args.html:
        html_files = [Path(args.html)]
    elif args.dir:
        root = Path(args.dir)
        html_files = sorted(list(root.rglob("*poster_template.html")))
    else:
        # Default search in common output directories
        candidates = [
            Path("output_demo_promo_posters"),
            Path("output_custom_poster"),
            Path("output_custom_coffee"),
        ]
        html_files = []
        for c in candidates:
            if c.exists():
                html_files.extend(c.rglob("*poster_template.html"))

    if not html_files:
        print("[!] No *poster_template.html files found!")
        print("    Specify --dir <path> or --html <path_to_file.html>")
        sys.exit(1)

    print("=" * 80)
    print(f"🚀 TENDOO AI - PLAYWRIGHT VECTOR POSTER RENDERER ({len(html_files)} files)")
    print("=" * 80)

    for hf in html_files:
        render_single_html(hf, width=args.width, height=args.height)

    print("\n" + "=" * 80)
    print("🎉 TẤT CẢ POSTER ĐÃ ĐƯỢC RENDER HOÀN TẤT VỚI PLAYWRIGHT CHROMIUM!")
    print("=" * 80)


if __name__ == "__main__":
    main()
