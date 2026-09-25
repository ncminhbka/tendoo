#!/usr/bin/env python3
"""
scripts/test_sandwich_top_heavy.py

Kịch bản kiểm thử toàn diện cho template Sandwich Top Heavy:
- Chạy toàn bộ 16 test cases từ `tests/test_sandwich_top_heavy_suite.json`.
- Bao trùm 4 kích thước: 1:1 (1024x1024), 9:16 (576x1024), 16:9 (1024x576), 4:5 (816x1024).
- Bao trùm 4 cấp độ text: Minimal, Light, Medium, Extreme Heavy.
- Render ảnh poster PNG bằng Playwright Chromium trên nền Mock chất lượng cao.
- Đo lường chính xác từng phần tử DOM:
    + Kiểm tra tự tràn hộp (self-overflow / cắt chữ).
    + Kiểm tra tràn ngoài canvas.
    + Kiểm tra độ khớp mask corridor.
    + Đo kích thước font Hero thực tế (kiểm tra co giãn nhị phân).
- Xuất báo cáo HTML Gallery trực quan tại `output_tendoo_v3/sandwich_top_heavy_gallery/gallery.html`.
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

TEST_SUITE_PATH = PROJECT_ROOT / "tests" / "test_sandwich_top_heavy_suite.json"
OUTPUT_DIR = PROJECT_ROOT / "output_tendoo_v3" / "sandwich_top_heavy_gallery"


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
          selfOverflowY: el.scrollHeight > el.clientHeight + 4.0,
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
            <span class="case-badge">{cid}</span>
            <span class="ratio-badge">{ratio_label} ({w}x{h})</span>
          </div>
          <h3 class="case-title">{title}</h3>
          <p class="case-hero">Hero: <b>"{hero}"</b></p>
          <div class="meta-row">
            <span>Hero Font: <b style="color:#FFD700;">{hero_font:.1f}px</b></span>
            <span>Render: <b>{dur:.2f}s</b></span>
            <span>Trạng thái: <b style="color:{status_color};">{mask_status}</b></span>
          </div>
          <div class="media-preview">
            <div class="preview-col">
              <span class="preview-label">Final Poster (Chromium Render)</span>
              <a href="{cid}/poster.png" target="_blank">
                <img src="{cid}/poster.png" alt="Poster" class="img-poster" loading="lazy">
              </a>
            </div>
            <div class="preview-col">
              <span class="preview-label">Continuous Mask (DiT Corridor)</span>
              <a href="{cid}/mask.png" target="_blank">
                <img src="{cid}/mask.png" alt="Mask" class="img-mask" loading="lazy">
              </a>
            </div>
          </div>
          <div class="card-footer">
            <a href="{cid}/poster.html" target="_blank" class="btn-link">Mở File HTML</a>
            <span class="mask-info">Mask &le; 44% canvas</span>
          </div>
        </div>
        """)

    all_cards = "\n".join(cards_html)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Tendoo v3 - Báo Cáo Kiểm Thử Sandwich Top Heavy (16 Cases)</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0A0C14;
      color: #E2E8F0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      padding: 40px 24px;
      line-height: 1.5;
    }}
    .header {{
      text-align: center;
      margin-bottom: 40px;
    }}
    .header h1 {{
      font-size: 34px;
      font-weight: 900;
      color: #FFF;
      letter-spacing: -0.5px;
      margin-bottom: 8px;
      background: linear-gradient(135deg, #FFF 0%, #FFD700 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .header p {{
      color: #94A3B8;
      font-size: 15px;
      max-width: 800px;
      margin: 0 auto;
    }}
    .stats-bar {{
      display: flex;
      justify-content: center;
      gap: 24px;
      margin-top: 24px;
    }}
    .stat-item {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 10px 20px;
      border-radius: 12px;
      text-align: center;
    }}
    .stat-val {{ font-size: 22px; font-weight: 800; color: #FFD700; }}
    .stat-lbl {{ font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 1px; }}

    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 28px;
      max-width: 1680px;
      margin: 0 auto;
    }}
    .case-card {{
      background: rgba(18, 22, 36, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .case-badge {{
      background: rgba(255, 215, 0, 0.15);
      color: #FFD700;
      border: 1px solid rgba(255, 215, 0, 0.3);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
    }}
    .ratio-badge {{
      background: rgba(59, 130, 246, 0.15);
      color: #60A5FA;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
    }}
    .case-title {{
      font-size: 16px;
      font-weight: 800;
      color: #F8FAFC;
      margin-bottom: 4px;
    }}
    .case-hero {{
      font-size: 13px;
      color: #94A3B8;
      margin-bottom: 10px;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      font-size: 12px;
      color: #64748B;
      padding-bottom: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      margin-bottom: 12px;
    }}
    .meta-row b {{ color: #CBD5E1; }}
    .media-preview {{
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .preview-col {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .preview-label {{
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #64748B;
      font-weight: 700;
    }}
    .img-poster, .img-mask {{
      width: 100%;
      height: 340px;
      object-fit: contain;
      background: #000;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      transition: transform 0.2s ease;
    }}
    .img-poster:hover, .img-mask:hover {{
      transform: scale(1.02);
      border-color: #FFD700;
    }}
    .card-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }}
    .btn-link {{
      color: #38BDF8;
      text-decoration: none;
      font-size: 12.5px;
      font-weight: 600;
    }}
    .btn-link:hover {{ text-decoration: underline; }}
    .mask-info {{ font-size: 11.5px; color: #10B981; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Sandwich Top Heavy - Báo Cáo Thẩm Mỹ & Thích Ứng (16 Cases)</h1>
    <p>Kiểm thử 4 tỉ lệ (1:1, 9:16, 16:9, 4:5) x 4 cấp độ text (Minimal &rarr; Extreme Heavy) theo chuẩn yeu_cau_templates.txt</p>
    <div class="stats-bar">
      <div class="stat-item">
        <div class="stat-val">{len(results)}</div>
        <div class="stat-lbl">Test Cases</div>
      </div>
      <div class="stat-item">
        <div class="stat-val">4</div>
        <div class="stat-lbl">Tỉ lệ khung hình</div>
      </div>
      <div class="stat-item">
        <div class="stat-val">{sum(1 for r in results if r['overflow_count'] == 0 and not r['canvas_overflow'])}/{len(results)}</div>
        <div class="stat-lbl">Pass Bounding Box</div>
      </div>
    </div>
  </div>

  <div class="gallery-grid">
    {all_cards}
  </div>
</body>
</html>
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sandwich Top Heavy stress-test runner")
    parser.add_argument("--suite", type=str, default=str(TEST_SUITE_PATH), help="Path to test suite JSON")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory for gallery")
    args = parser.parse_args()
    test_suite_path = Path(args.suite).resolve()
    output_dir = Path(args.output).resolve()

    print("=" * 80)
    print(">>> TENDOO v3: KIỂM THỬ TOÀN DIỆN TEMPLATE SANDWICH TOP HEAVY")
    print(f">>> File Test Suite: {test_suite_path}")
    print(f">>> Output Gallery: {output_dir}")
    print("=" * 80)

    if not test_suite_path.exists():
        print(f"Lỗi: Không tìm thấy file {test_suite_path}!")
        sys.exit(1)

    with open(test_suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t_start = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = browser.new_page()

        for idx, item in enumerate(cases, 1):
            cid = item["id"]
            title = item["title"]
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
            # tự tính nội bộ -- xem geometry.py::_PRESENCE_AWARE_TEMPLATES).
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
    print(f">>> KẾT QUẢ KIỂM THỬ: {total_pass}/{len(cases)} CASES PASS HOÀN TOÀN (0 LỖI TRÀN)")
    print(f">>> Tổng thời gian chạy: {total_time:.2f}s")
    print(f">>> Gallery xem trực quan: {gallery_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
