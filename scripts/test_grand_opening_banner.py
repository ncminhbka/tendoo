#!/usr/bin/env python3
"""
scripts/test_grand_opening_banner.py

Script kiểm thử toàn diện cho template: grand_opening_banner (Banner Khai Trương Trọng Thể)
- Kiểm tra 16 test cases qua 4 tỉ lệ (1:1, 9:16, 16:9, 4:5) x 4 mức độ dài text.
- Đo lường kích thước font, chống mất chữ, chống tràn bounding box, chống tràn canvas.
- Tạo báo cáo visual gallery tại: output_tendoo_v3/grand_opening_banner_gallery/gallery.html
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

# Thêm src vào sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tendoo_v3.renderer import render_plan_to_poster
from tendoo_v3.schema import StyleConfig, TendooCreativePlan


def generate_mock_background(width: int, height: int, theme_hex: str = "#D4AF37") -> str:
    """Tạo ảnh nền lễ hội sang trọng với ánh sáng trung tâm."""
    img = Image.new("RGB", (width, height), color=(10, 14, 24))
    draw = ImageDraw.Draw(img)

    r = int(theme_hex[1:3], 16)
    g = int(theme_hex[3:5], 16)
    b = int(theme_hex[5:7], 16)

    # Điểm sáng trung tâm poster cho chủ thể khai trương (ly cà phê, dải băng khánh thành)
    center_x = int(width * 0.50)
    center_y = int(height * 0.56)
    radius = int(min(width, height) * 0.42)

    for i in range(radius, 0, -20):
        alpha = int(40 * (1.0 - i / radius))
        glow_col = (
            min(255, 10 + int(r * alpha / 255)),
            min(255, 14 + int(g * alpha / 255)),
            min(255, 24 + int(b * alpha / 255)),
        )
        draw.ellipse(
            [center_x - i, center_y - i, center_x + i, center_y + i],
            fill=glow_col,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"


def inspect_page_metrics(page, width: int, height: int) -> List[Dict[str, Any]]:
    """Đo lường chính xác các phần tử DOM trên trang bằng Chromium."""
    js = """
    () => {
      const out = [];
      const els = document.querySelectorAll(
        '.kicker-capsule, .hero-title, .subhead-date, .extra-pill, .freetext-block, .cta-btn, .store-brand-pill, .store-detail-item, .qr-card-pod'
      );
      for (const el of els) {
        const r = el.getBoundingClientRect();
        const cs = window.getComputedStyle(el);
        const text = (el.innerText || '').trim();
        if (r.width <= 0 || r.height <= 0) continue;

        let selfOverX = false;
        let selfOverY = false;
        if (el.dataset && el.dataset.maxWidth) {
          const maxW = parseFloat(el.dataset.maxWidth);
          if (el.scrollWidth > maxW + 4.0) selfOverX = true;
        } else {
          if (el.scrollWidth > el.clientWidth + 4.0) selfOverX = true;
        }

        if (el.dataset && el.dataset.maxHeight) {
          const maxH = parseFloat(el.dataset.maxHeight);
          if (el.scrollHeight > maxH + 4.0) selfOverY = true;
        } else {
          if (el.scrollHeight > el.clientHeight + 6.0) selfOverY = true;
        }

        out.push({
          tag: el.tagName,
          cls: (el.className && el.className.toString) ? el.className.toString() : '',
          text: text.slice(0, 60),
          fontSize: parseFloat(cs.fontSize) || 0,
          x1: r.left, y1: r.top, x2: r.right, y2: r.bottom,
          selfOverflowX: selfOverX,
          selfOverflowY: selfOverY,
        });
      }
      return out;
    }
    """
    return page.evaluate(js)


def build_gallery_html(title_name: str, results: List[Dict[str, Any]]) -> str:
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
        status_color = "#10B981" if (overflow_count == 0 and not canvas_overflow) else "#EF4444"
        status_text = "PASS HOÀN TOÀN" if (overflow_count == 0 and not canvas_overflow) else f"CÓ LỖI: {overflow_count} lỗi"
        ratio_label = "1:1 Vuông" if w == h else ("9:16 Dọc" if w < 600 else ("16:9 Ngang" if w > 1000 and h < 600 else "4:5 Dọc"))

        cards_html.append(f"""
        <div class="case-card">
          <div class="card-header">
            <div class="card-badge" style="background:{status_color};">{status_text}</div>
            <div class="card-meta">
              <span class="ratio-pill">{ratio_label} ({w}x{h})</span>
              <span class="time-pill">⚡ {dur:.2f}s</span>
            </div>
          </div>
          <div class="card-body">
            <div class="poster-preview">
              <a href="{cid}/poster.png" target="_blank">
                <img src="{cid}/poster.png" alt="{title}" loading="lazy" />
              </a>
            </div>
            <div class="metrics-panel">
              <h3 class="case-title">{title}</h3>
              <div class="metric-row">
                <span class="metric-label">Hero Font:</span>
                <span class="metric-val" style="color:#38BDF8; font-weight:700;">{hero_font:.1f}px</span>
              </div>
              <div class="metric-row">
                <span class="metric-label">Lỗi tự tràn:</span>
                <span class="metric-val" style="color:{status_color}; font-weight:700;">{overflow_count}</span>
              </div>
              <div class="metric-row">
                <span class="metric-label">Tràn canvas:</span>
                <span class="metric-val">{'Có' if canvas_overflow else 'Không'}</span>
              </div>
              <div class="action-links">
                <a class="btn-link" href="{cid}/poster.html" target="_blank">📄 Xem HTML</a>
                <a class="btn-link" href="{cid}/poster.png" target="_blank">🖼️ Xem PNG</a>
              </div>
            </div>
          </div>
        </div>
        """)

    cards_joined = "\n".join(cards_html)
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Gallery Kiểm Thử: {title_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0B0F19;
      color: #E2E8F0;
      padding: 32px 24px;
    }}
    .header {{
      text-align: center;
      margin-bottom: 36px;
    }}
    .header h1 {{
      font-size: 28px;
      font-weight: 800;
      color: #FFFFFF;
      margin-bottom: 8px;
    }}
    .header p {{
      color: #94A3B8;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
      gap: 24px;
      max-width: 1600px;
      margin: 0 auto;
    }}
    .case-card {{
      background: #151C2C;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: #1A2337;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .card-badge {{
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
      color: #fff;
    }}
    .card-meta {{
      display: flex;
      gap: 8px;
    }}
    .ratio-pill, .time-pill {{
      font-size: 11.5px;
      background: rgba(255, 255, 255, 0.08);
      padding: 3px 8px;
      border-radius: 6px;
      color: #CBD5E1;
    }}
    .card-body {{
      display: flex;
      gap: 16px;
      padding: 16px;
    }}
    .poster-preview {{
      flex: 0 0 200px;
      background: #0B0F19;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .poster-preview img {{
      max-width: 100%;
      max-height: 240px;
      object-fit: contain;
      display: block;
    }}
    .metrics-panel {{
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .case-title {{
      font-size: 14.5px;
      font-weight: 700;
      color: #F8FAFC;
      line-height: 1.35;
      margin-bottom: 8px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      padding: 4px 0;
      border-bottom: 1px dashed rgba(255, 255, 255, 0.06);
    }}
    .metric-label {{
      color: #94A3B8;
    }}
    .metric-val {{
      color: #F1F5F9;
    }}
    .action-links {{
      display: flex;
      gap: 8px;
      margin-top: 12px;
    }}
    .btn-link {{
      font-size: 12px;
      text-decoration: none;
      color: #38BDF8;
      background: rgba(56, 189, 248, 0.1);
      padding: 5px 10px;
      border-radius: 6px;
      transition: all 0.15s ease;
    }}
    .btn-link:hover {{
      background: rgba(56, 189, 248, 0.25);
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Gallery Kiểm Thử: {title_name}</h1>
    <p>Kiểm tra tự động 16 ca thử nghiệm across 4 tỉ lệ & đa dạng độ dài văn bản</p>
  </div>
  <div class="grid">
    {cards_joined}
  </div>
</body>
</html>"""


def run_grand_opening_test_suite():
    import argparse

    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="grand_opening_banner stress-test runner")
    parser.add_argument("--suite", type=str, default=str(root_dir / "tests" / "test_grand_opening_banner_suite.json"))
    parser.add_argument("--output", type=str, default=str(root_dir / "output_tendoo_v3" / "grand_opening_banner_gallery"))
    args, _ = parser.parse_known_args()
    suite_path = Path(args.suite).resolve()
    with open(suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    gallery_dir = Path(args.output).resolve()
    gallery_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"TENDOO V3 - CHẠY KIỂM THỬ TEMPLATE: GRAND_OPENING_BANNER (16 CASES)")
    print("=" * 80)

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for idx, case in enumerate(cases, 1):
            cid = case["id"]
            title = case["title"]
            w, h = case["width"], case["height"]

            case_dir = gallery_dir / cid
            case_dir.mkdir(parents=True, exist_ok=True)
            output_png = case_dir / "poster.png"
            output_html = case_dir / "poster.html"

            print(f"[{idx:02d}/{len(cases)}] Chạy: {cid} | ({w}x{h}) - {title}")
            t0 = time.perf_counter()

            # 1. Tạo plan
            style_dict = case.get("style", {})
            style_obj = StyleConfig(
                theme_color=style_dict.get("theme_color", "#D4AF37"),
                text_effect=style_dict.get("text_effect", "3d_gold"),
                background_tone=style_dict.get("background_tone", "dark_luxury"),
            )

            plan = TendooCreativePlan(
                template="grand_opening_banner",
                hero=case.get("hero", ""),
                subhead=case.get("subhead"),
                badge=case.get("badge"),
                extra_texts=case.get("extra_texts", []),
                cta=case.get("cta"),
                store_info=case.get("store_info"),
                qr_code=case.get("qr_code"),
                qr_label=case.get("qr_label"),
                style=style_obj,
            )

            # 2. Sinh mock background
            bg_data_uri = generate_mock_background(w, h, theme_hex=style_obj.theme_color)

            # 3. Render HTML & PNG
            out_path, html_content = render_plan_to_poster(
                plan=plan,
                bg_data_uri=bg_data_uri,
                output_image_path=output_png,
                width=w,
                height=h,
            )

            with open(output_html, "w", encoding="utf-8") as f:
                f.write(html_content)

            # 4. Kiểm tra DOM metrics bằng Playwright
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(f"file:///{output_html.resolve().as_posix()}")
            page.wait_for_timeout(350)

            dom_metrics = inspect_page_metrics(page, w, h)
            page.close()

            hero_font = 0.0
            overflow_count = 0
            canvas_overflow = False

            for m in dom_metrics:
                if "hero-title" in m["cls"]:
                    hero_font = m["fontSize"]
                if m["selfOverflowX"] or m["selfOverflowY"]:
                    overflow_count += 1
                # Kiểm tra lọt ra ngoài canvas
                if m["x1"] < -4 or m["y1"] < -4 or m["x2"] > w + 4 or m["y2"] > h + 4:
                    canvas_overflow = True

            dur = time.perf_counter() - t0
            status_symbol = "✓" if (overflow_count == 0 and not canvas_overflow) else "🚨"
            print(f"    {status_symbol} Hoàn thành ({dur:.2f}s) | Hero Font: {hero_font:.1f}px | Lỗi tự tràn: {overflow_count} | Tràn canvas: {canvas_overflow}")

            results.append({
                "id": cid,
                "title": title,
                "template": "grand_opening_banner",
                "width": w,
                "height": h,
                "hero": case.get("hero", ""),
                "hero_font": hero_font,
                "overflow_count": overflow_count,
                "canvas_overflow": canvas_overflow,
                "duration_s": dur,
            })

    # Xây dựng trang Gallery HTML tổng hợp
    gallery_html = build_gallery_html("Banner Khai Trương (Grand Opening)", results)
    gallery_file = gallery_dir / "gallery.html"
    with open(gallery_file, "w", encoding="utf-8") as f:
        f.write(gallery_html)

    print("\n" + "=" * 80)
    print("TỔNG KẾT KIỂM THỬ GRAND OPENING BANNER:")
    print(f"Gallery Report: file:///{gallery_file.resolve().as_posix()}")
    print("=" * 80)


if __name__ == "__main__":
    run_grand_opening_test_suite()
