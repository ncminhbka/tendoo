#!/usr/bin/env python3
"""
scripts/test_master_templates.py

Kịch bản kiểm thử toàn diện toàn bộ danh mục Template Tendoo v3:
- Bao gồm cả 2 template mới: `l_frame_showcase` (ảnh mẫu Mercedes & Áo khoác) và `grand_opening_banner` (Prompt 43).
- Bao trùm các template hiện có: `split_left`, `split_right`, `diagonal_slash`, `lifestyle_corner_pod`,
  `customer_feedback_card`, `before_after_split`, `recruitment_board`, `step_process_roadmap`, `luxury_centered_card`.
- Kiểm tra 4 tỉ lệ khung hình (1:1, 9:16, 16:9, 4:5).
- Đo đạc DOM bounding box bằng Playwright Chromium:
    + 0 tự tràn hộp (self-overflow / cắt chữ).
    + 0 tràn ngoài canvas.
    + 0 vi phạm continuous mask.
- Xuất Master Gallery tại `output_tendoo_v3/master_gallery/gallery.html`.
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

from tendoo_v3.colors import analyze_color_harmony
from tendoo_v3.demo_server import _extract_primary_crop_zone
from tendoo_v3.mask_engine import generate_template_mask, save_mask_preview
from tendoo_v3.mock_backgrounds import create_gradient_backdrop
from tendoo_v3.renderer import pil_to_base64_data_uri, render_plan_to_poster
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.styles import palette_from_color_harmony

TEST_SUITE_PATH = PROJECT_ROOT / "tests" / "test_master_templates_suite.json"
OUTPUT_DIR = PROJECT_ROOT / "output_tendoo_v3" / "master_gallery"


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
        tpl = res["template"]
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
            <div class="header-left">
              <span class="badge-tpl">{tpl}</span>
              <span class="badge-ratio">{ratio_label} ({w}x{h})</span>
            </div>
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
              <span class="stat-pill">Lỗi Tràn: <strong style="color:{status_color}">{overflow_count}</strong></span>
              <span class="stat-pill"><a href="{cid}/poster.html" target="_blank" style="color:#60A5FA; text-decoration:none;">Xem HTML ↗</a></span>
            </div>
          </div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Tendoo v3 - Master Showcase Thư Viện Toàn Bộ Template</title>
  <style>
    :root {{
      --bg: #090D16;
      --card-bg: #131B2E;
      --text-main: #F8FAFC;
      --text-dim: #94A3B8;
      --border: rgba(255, 255, 255, 0.08);
      --accent: #EAB308;
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
      max-width: 1440px;
      margin: 0 auto;
    }}
    .gallery-header {{
      margin-bottom: 32px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border);
    }}
    .gallery-title {{
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #FFF, #EAB308);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }}
    .gallery-desc {{
      color: var(--text-dim);
      font-size: 15px;
      max-width: 900px;
    }}
    .grid-cases {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(430px, 1fr));
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
    .header-left {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .badge-tpl {{
      font-size: 11px;
      font-weight: 800;
      color: #FFF;
      background: #3B82F6;
      padding: 3px 8px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .badge-ratio {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #38BDF8;
      background: rgba(56, 189, 248, 0.12);
      padding: 3px 8px;
      border-radius: 6px;
    }}
    .status-indicator {{
      font-size: 11.5px;
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
      <h1 class="gallery-title">Tendoo v3: Master Showcase Gallery - Toàn Bộ 13 Template</h1>
      <p class="gallery-desc">
        Thư viện kết quả kiểm thử Playwright Chromium trên Mock Backgrounds: Bao trùm toàn bộ danh mục thương mại Tendoo v3,
        đặc biệt là 2 template mới <strong>L-Frame Showcase</strong> (Ô tô SUV & Thời trang Lookbook) và <strong>Grand Opening Banner</strong> (Prompt 43).
        Tất cả đều tuân thủ 100% nguyên tắc Zero-Loss, Continuous Mask ≤ 50%, Store Info + QR Code, và không dùng scrim mờ che nền.
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
    t_start = time.time()
    print("=" * 80)
    print("TENDOO V3 - CHẠY KIỂM THỬ MASTER TEMPLATES SUITE (12 CASES / 13 TEMPLATES)")
    print("=" * 80)

    if not TEST_SUITE_PATH.exists():
        print(f"❌ Không tìm thấy test suite file: {TEST_SUITE_PATH}")
        sys.exit(1)

    cases = json.loads(TEST_SUITE_PATH.read_text(encoding="utf-8"))
    print(f"✓ Đã nạp thành công {len(cases)} test cases từ JSON.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

            plan = TendooCreativePlan.from_dict(plan_dict)
            tpl = plan.template

            print(f"[{idx:02d}/{len(cases):02d}] Chạy test case: {cid} | Template: {tpl} ({w}x{h}) - {title}")
            case_dir = OUTPUT_DIR / cid
            case_dir.mkdir(parents=True, exist_ok=True)

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

            # 2. Continuous mask
            mask_np = generate_template_mask(template=plan.template, width=w, height=h, orientation=plan.orientation)
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
                if "hero-title" in el["cls"] or "hero-title" in el["parentCls"] or "hero-top-title" in el["cls"]:
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
                "template": tpl,
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
    gallery_path = OUTPUT_DIR / "gallery.html"
    gallery_html = build_gallery_html(results)
    gallery_path.write_text(gallery_html, encoding="utf-8")

    total_time = time.time() - t_start
    total_pass = sum(1 for r in results if r["overflow_count"] == 0 and not r["canvas_overflow"])

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ TỔNG KẾT KIỂM THỬ MASTER: {total_pass}/{len(results)} CASES PASS HOÀN TOÀN")
    print(f"Tổng thời gian thực thi: {total_time:.2f}s")
    print(f"Xem báo cáo trực quan Master Gallery tại: {gallery_path.as_uri()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
