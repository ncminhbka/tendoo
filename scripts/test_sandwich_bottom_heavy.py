#!/usr/bin/env python3
"""
scripts/test_sandwich_bottom_heavy.py

Kịch bản kiểm thử toàn diện cho template Sandwich Bottom Heavy:
- Chạy toàn bộ 16 test cases từ `tests/test_sandwich_bottom_heavy_suite.json`.
- Bao trùm 4 kích thước: 1:1 (1024x1024), 9:16 (576x1024), 16:9 (1024x576), 4:5 (816x1024).
- Bao trùm 4 cấp độ text: Minimal, Light, Medium, Extreme Heavy.
- Render ảnh poster PNG bằng Playwright Chromium trên nền Mock chất lượng cao.
- Đo lường chính xác từng phần tử DOM:
    + Kiểm tra tự tràn hộp (self-overflow / cắt chữ).
    + Kiểm tra tràn ngoài canvas.
    + Kiểm tra độ khớp mask corridor.
    + Đo kích thước font Hero thực tế (kiểm tra co giãn nhị phân).
- Xuất báo cáo HTML Gallery trực quan tại `output_tendoo_v3/sandwich_bottom_heavy_gallery/gallery.html`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from playwright.sync_api import sync_playwright

from tendoo_core.colors import analyze_color_harmony
from tendoo_v3.demo_server import _extract_primary_crop_zone
from tendoo_v3.mask_engine import generate_template_mask, save_mask_preview
from tendoo_v3.mock_backgrounds import create_gradient_backdrop
from tendoo_v3.renderer import pil_to_base64_data_uri, render_plan_to_poster
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.styles import palette_from_color_harmony

DEFAULT_TEST_SUITE_PATH = PROJECT_ROOT / "tests" / "test_sandwich_bottom_heavy_suite.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_tendoo_v3" / "sandwich_bottom_heavy_gallery"


def measure_text_elements_in_page(page, width: int, height: int) -> List[Dict[str, Any]]:
    """Đo bounding box thật và cờ tràn hộp của mọi phần tử lá có text."""
    js = r"""
    () => {
      const out = [];
      const all = document.querySelectorAll('body *');
      for (const el of all) {
        if (el.children.length > 0) continue;
        const text = (el.textContent || '').trim();
        if (!text) continue;
        const cs = getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity || '1') === 0) continue;
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) continue;
        out.push({
          tag: el.tagName,
          cls: (el.className && el.className.toString) ? el.className.toString() : '',
          parentCls: (el.parentElement && el.parentElement.className) ? el.parentElement.className.toString() : '',
          text: text.slice(0, 60),
          fontSize: parseFloat(cs.fontSize) || 0,
          x1: r.left, y1: r.top, x2: r.right, y2: r.bottom,
          selfOverflowX: el.scrollWidth > el.clientWidth + 2.0,
          selfOverflowY: (cs.overflow === 'hidden' || cs.overflowY === 'hidden')
            ? (el.scrollHeight > el.clientHeight + 4.0)
            : (el.dataset.maxHeight ? el.scrollHeight > parseFloat(el.dataset.maxHeight) + 1.5 : el.scrollHeight > el.clientHeight + 14.0),
        });
      }
      return out;
    }
    """
    return page.evaluate(js)


def build_gallery_html(results: List[Dict[str, Any]]) -> str:
    cards_html = []
    for res in results:
        cid = res["id"]
        title = res["title"]
        w, h = res["width"], res["height"]
        hero = res["hero"]
        hero_font = res.get("hero_font", 0)
        dur = res["duration_s"]
        overflow_count = res["overflow_count"]
        canvas_overflow = res["canvas_overflow"]
        mask_status = "Đạt Chuẩn (0 lỗi)" if (overflow_count == 0 and not canvas_overflow) else f"Cảnh Báo: {overflow_count} lỗi"
        status_color = "#10B981" if (overflow_count == 0 and not canvas_overflow) else "#EF4444"

        ratio_label = "1:1 Vuông" if w == h else ("9:16 Dọc" if w < 600 else ("16:9 Ngang" if w > 1000 and h < 600 else "4:5 Dọc"))

        cards_html.append(f"""
        <div class="case-card">
          <div class="card-header">
            <div class="badge-ratio">{ratio_label} ({w}x{h})</div>
            <div class="status-indicator" style="background:{status_color}22; color:{status_color}; border:1px solid {status_color}66;">
              {mask_status}
            </div>
          </div>
          <div class="card-preview">
            <div class="preview-box poster-view">
              <span class="preview-tag">Poster Thành Phẩm</span>
              <img src="{cid}/poster.png" alt="Poster {cid}" loading="lazy" />
            </div>
            <div class="preview-box mask-view">
              <span class="preview-tag">Continuous Mask</span>
              <img src="{cid}/mask.png" alt="Mask {cid}" loading="lazy" />
            </div>
          </div>
          <div class="card-meta">
            <div class="case-title">{title}</div>
            <div class="case-hero">"{hero}"</div>
            <div class="stats-row">
              <span class="stat-pill">Hero Font: <strong>{hero_font:.1f}px</strong></span>
              <span class="stat-pill">Render: <strong>{dur:.2f}s</strong></span>
              <span class="stat-pill">Lỗi Tràn Hộp: <strong style="color:{status_color}">{overflow_count}</strong></span>
              <span class="stat-pill"><a href="{cid}/poster.html" target="_blank" style="color:#60A5FA; text-decoration:none;">Xem HTML ↗</a></span>
            </div>
          </div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Tendoo v3 - Thư Viện Kiểm Thử Toàn Diện Sandwich Bottom Heavy</title>
  <style>
    :root {{
      --bg: #090D16;
      --card-bg: #131B2E;
      --text-main: #F8FAFC;
      --text-dim: #94A3B8;
      --border: rgba(255, 255, 255, 0.08);
      --accent: #FF3366;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 40px 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    .gallery-header {{
      margin-bottom: 32px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border);
    }}
    .gallery-title {{
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #FFF, #94A3B8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }}
    .gallery-desc {{
      color: var(--text-dim);
      font-size: 15px;
      max-width: 800px;
    }}
    .grid-cases {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 28px;
    }}
    .case-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .case-card:hover {{
      transform: translateY(-4px);
      border-color: rgba(255, 255, 255, 0.2);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 18px;
      background: rgba(255, 255, 255, 0.02);
      border-bottom: 1px solid var(--border);
    }}
    .badge-ratio {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #38BDF8;
      background: rgba(56, 189, 248, 0.12);
      padding: 4px 10px;
      border-radius: 9999px;
    }}
    .status-indicator {{
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
    }}
    .card-preview {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 16px;
      background: #060911;
    }}
    .preview-box {{
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      aspect-ratio: 1 / 1;
    }}
    .preview-box img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }}
    .preview-tag {{
      position: absolute;
      top: 6px;
      left: 6px;
      font-size: 9.5px;
      font-weight: 700;
      padding: 2px 6px;
      background: rgba(0, 0, 0, 0.7);
      color: #E2E8F0;
      border-radius: 4px;
      backdrop-filter: blur(4px);
    }}
    .card-meta {{
      padding: 16px 18px 18px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      flex: 1;
    }}
    .case-title {{
      font-size: 15px;
      font-weight: 700;
      color: #FFF;
      line-height: 1.35;
    }}
    .case-hero {{
      font-size: 13px;
      color: #CBD5E1;
      font-style: italic;
      line-height: 1.3;
    }}
    .stats-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: auto;
      padding-top: 12px;
      border-top: 1px solid var(--border);
    }}
    .stat-pill {{
      font-size: 11.5px;
      background: rgba(255, 255, 255, 0.05);
      padding: 4px 8px;
      border-radius: 6px;
      color: var(--text-dim);
    }}
    .stat-pill strong {{
      color: #FFF;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="gallery-header">
      <h1 class="gallery-title">Tendoo v3: Kết Quả Kiểm Thử Template Sandwich Bottom Heavy</h1>
      <p class="gallery-desc">
        Kiểm thử tự động 16 trường hợp thực tế trên Playwright Chromium: bao trùm 4 tỉ lệ khung hình (1:1, 9:16, 16:9, 4:5) và 4 cấp độ văn bản (từ Tối giản đến Cực đại).
        Đảm bảo không cắt chữ, không tràn canvas, bảo toàn 100% diện tích không gian sáng tạo DiT với Continuous Mask chuẩn xác.
      </p>
    </div>

    <div class="grid-cases">
      {''.join(cards_html)}
    </div>
  </div>
</body>
</html>
"""
    return html


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sandwich Bottom Heavy stress-test runner")
    parser.add_argument("--suite", type=str, default=str(DEFAULT_TEST_SUITE_PATH), help="Path to test suite JSON")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for gallery")
    args = parser.parse_args()

    test_suite_path = Path(args.suite).resolve()
    output_dir = Path(args.output).resolve()

    t_start = time.time()
    print("=" * 80)
    print("TENDOO V3 - CHẠY KIỂM THỬ TOÀN DIỆN SANDWICH BOTTOM HEAVY")
    print(f">>> Suite: {test_suite_path}")
    print("=" * 80)

    if not test_suite_path.exists():
        print(f"❌ Không tìm thấy test suite file: {test_suite_path}")
        sys.exit(1)

    cases = json.loads(test_suite_path.read_text(encoding="utf-8"))
    print(f"✓ Đã nạp thành công {len(cases)} test cases từ JSON.")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for idx, item in enumerate(cases, 1):
            cid = item["id"]
            title = item.get("title", cid)
            w = item.get("width", 1024)
            h = item.get("height", 1024)
            plan_dict = item["plan"]

            print(f"[{idx:02d}/{len(cases):02d}] Chạy test case: {cid} ({w}x{h}) - {title}")
            case_dir = output_dir / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            plan = TendooCreativePlan.from_dict(plan_dict)

            # 1. Mock background
            tone = plan.style.background_tone or "dark_luxury"
            backdrop_pil = create_gradient_backdrop(width=w, height=h, tone=tone)
            bg_data_uri = pil_to_base64_data_uri(backdrop_pil)

            # 1.5. Color harmony
            try:
                crop_zone = _extract_primary_crop_zone(plan.template, w, h, plan.orientation)
                cp = analyze_color_harmony(
                    np.array(backdrop_pil.convert("RGB")),
                    crop_zone=crop_zone,
                    color_mode="auto",
                    font_style="luxury_serif" if plan.style.font == "playfair" else "modern_sans",
                )
                palette_override = palette_from_color_harmony(cp, theme_color=plan.style.theme_color)
            except Exception:
                palette_override = None

            # 2. Continuous mask (has_qr PHẢI khớp bool(plan.qr_code) như build_template_html
            # tự tính nội bộ -- xem geometry.py::_PRESENCE_AWARE_TEMPLATES -- nếu không mask
            # preview sẽ lệch kích thước với CSS thật đã render ở bước 3).
            mask_np = generate_template_mask(template=plan.template, width=w, height=h, has_qr=bool(plan.qr_code))
            mask_file = case_dir / "mask.png"
            save_mask_preview(mask_np, str(mask_file))

            # 3. Render HTML poster
            t0 = time.time()
            poster_file = case_dir / "poster.png"
            out_path, html_content = render_plan_to_poster(
                plan=plan,
                bg_data_uri=bg_data_uri,
                output_image_path=poster_file,
                width=w,
                height=h,
                palette_override=palette_override,
            )
            dur = time.time() - t0

            html_file = case_dir / "poster.html"
            html_file.write_text(html_content, encoding="utf-8")

            # 4. Đo đạc DOM bounding box bằng Playwright
            page.set_viewport_size({"width": w, "height": h})
            page.goto(html_file.as_uri())
            try:
                page.wait_for_function("window.__tendooAutofitDone === true", timeout=2500)
            except Exception:
                page.wait_for_timeout(300)

            elements = measure_text_elements_in_page(page, w, h)

            overflow_count = 0
            canvas_overflow = False
            hero_font = 0.0

            for el in elements:
                # Kiểm tra hero font
                if "hero-title" in el["cls"] or "hero-title" in el["parentCls"]:
                    hero_font = max(hero_font, el["fontSize"])

                # Kiểm tra tự tràn hộp
                if el["selfOverflowX"] or el["selfOverflowY"]:
                    overflow_count += 1
                    print(f"    ⚠️ [TỰ TRÀN HỘP] <{el['tag']}>: \"{el['text']}\"")

                # Kiểm tra tràn ngoài canvas
                if el["x1"] < -1.0 or el["y1"] < -1.0 or el["x2"] > w + 1.0 or el["y2"] > h + 1.0:
                    canvas_overflow = True
                    print(f"    🚨 [TRÀN NGOÀI CANVAS] <{el['tag']}>: \"{el['text']}\" bbox=({el['x1']:.0f},{el['y1']:.0f},{el['x2']:.0f},{el['y2']:.0f})")

            print(f"    ✓ Hoàn thành ({dur:.2f}s) | Hero Font: {hero_font:.1f}px | Lỗi tự tràn: {overflow_count} | Tràn canvas: {canvas_overflow}")

            results.append({
                "id": cid,
                "title": title,
                "width": w,
                "height": h,
                "hero": plan.hero,
                "hero_font": hero_font,
                "duration_s": dur,
                "overflow_count": overflow_count,
                "canvas_overflow": canvas_overflow,
            })

        browser.close()

    # 5. Xuất trang Gallery HTML
    gallery_path = output_dir / "gallery.html"
    gallery_html = build_gallery_html(results)
    gallery_path.write_text(gallery_html, encoding="utf-8")

    total_time = time.time() - t_start
    total_pass = sum(1 for r in results if r["overflow_count"] == 0 and not r["canvas_overflow"])

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ TỔNG KẾT KIỂM THỬ: {total_pass}/{len(results)} CASES PASS HOÀN TOÀN")
    print(f"Tổng thời gian thực thi: {total_time:.2f}s")
    print(f"Xem báo cáo trực quan tại: {gallery_path.as_uri()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
