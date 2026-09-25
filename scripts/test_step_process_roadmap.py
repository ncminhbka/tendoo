#!/usr/bin/env python3
"""
scripts/test_step_process_roadmap.py

Script kiểm thử toàn diện cho template: step_process_roadmap (Bệ Quy Trình Lộ Trình)
- Kiểm tra 16 test cases qua 4 tỉ lệ (1:1, 9:16, 16:9, 4:5) x 2 hướng (left, right mirror) x nhiều mật độ text.
- Đo lường kích thước font, chống mất chữ, chống tràn bounding box, chống tràn canvas.
- Tạo báo cáo visual gallery tại: output_tendoo_v3/step_process_roadmap_gallery/gallery.html
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


def generate_mock_background(width: int, height: int, orientation: str = "left", theme_hex: str = "#D4AF37") -> str:
    """Tạo ảnh nền studio/workflow với ánh sáng tập trung ở vùng giữa poster."""
    img = Image.new("RGB", (width, height), color=(8, 12, 22))
    draw = ImageDraw.Draw(img)

    r = int(theme_hex[1:3], 16)
    g = int(theme_hex[3:5], 16)
    b = int(theme_hex[5:7], 16)

    # Đặt vùng sáng chủ thể ở đối diện của Header
    if orientation in ("right", "top_right", "bottom_right"):
        center_x = int(width * 0.35)
        center_y = int(height * 0.42)
    else:
        center_x = int(width * 0.65)
        center_y = int(height * 0.42)

    radius = int(min(width, height) * 0.42)

    for i in range(radius, 0, -18):
        alpha = int(48 * (1.0 - i / radius))
        glow_col = (
            min(255, 8 + int(r * alpha / 255)),
            min(255, 12 + int(g * alpha / 255)),
            min(255, 22 + int(b * alpha / 255)),
        )
        draw.ellipse([center_x - i, center_y - i, center_x + i, center_y + i], fill=glow_col)

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
        '.top-header, .badge-pill, .hero-title, .subhead-title, .roadmap-platform, .step-card, .step-num, .step-text, .cta-btn, .store-item, .qr-col'
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
        orient = res.get("orientation", "left")
        status_color = "#10B981" if (overflow_count == 0 and not canvas_overflow) else "#EF4444"
        status_text = "PASS HOÀN TOÀN" if (overflow_count == 0 and not canvas_overflow) else f"CÓ LỖI: {overflow_count} lỗi"
        ratio_label = "1:1 Vuông" if w == h else ("9:16 Dọc" if w < 600 else ("16:9 Ngang" if w > 1000 and h < 600 else "4:5 Dọc"))

        orient_labels = {
            "left": "Bố Cục Trái",
            "right": "Bố Cục Phải (Mirror)",
        }
        orient_str = orient_labels.get(orient, orient)

        cards_html.append(f"""
        <div class="case-card">
          <div class="card-header">
            <div class="card-badge" style="background:{status_color};">{status_text}</div>
            <div class="card-meta">
              <span class="orient-pill">{orient_str}</span>
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
    .ratio-pill, .orient-pill, .time-pill {{
      font-size: 11.5px;
      background: rgba(255, 255, 255, 0.08);
      padding: 3px 8px;
      border-radius: 6px;
      color: #CBD5E1;
    }}
    .card-body {{
      display: flex;
      padding: 16px;
      gap: 16px;
    }}
    .poster-preview {{
      flex: 0 0 200px;
      height: 240px;
      background: #0D131F;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .poster-preview img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }}
    .metrics-panel {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .case-title {{
      font-size: 15px;
      font-weight: 700;
      color: #F8FAFC;
      line-height: 1.3;
      margin-bottom: 4px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      padding-bottom: 4px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .metric-label {{
      color: #94A3B8;
    }}
    .metric-val {{
      color: #E2E8F0;
    }}
    .action-links {{
      margin-top: auto;
      display: flex;
      gap: 8px;
      padding-top: 10px;
    }}
    .btn-link {{
      font-size: 12px;
      padding: 6px 12px;
      border-radius: 6px;
      text-decoration: none;
      background: #1E293B;
      color: #38BDF8;
      border: 1px solid rgba(56, 189, 248, 0.3);
      font-weight: 600;
      transition: all 0.2s;
    }}
    .btn-link:hover {{
      background: #0284C7;
      color: #fff;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Báo Cáo Kiểm Thử Tự Động: {title_name}</h1>
    <p>Tổng cộng {len(results)} kịch bản kiểm thử (4 tỉ lệ x 2 hướng Mirror x Đa dạng quy trình)</p>
  </div>
  <div class="grid">
    {cards_joined}
  </div>
</body>
</html>
"""


def main():
    import argparse

    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="step_process_roadmap stress-test runner")
    parser.add_argument("--suite", type=str, default=str(root_dir / "tests" / "test_step_process_roadmap_suite.json"))
    parser.add_argument("--output", type=str, default=str(root_dir / "output_tendoo_v3" / "step_process_roadmap_gallery"))
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"================================================================================")
    print(f"🚀 BẮT ĐẦU KIỂM THỬ: step_process_roadmap (Tổng số: {len(cases)} test cases)")
    print(f"================================================================================")

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for idx, case in enumerate(cases, 1):
            cid = case["id"]
            title = case["title"]
            w = case["width"]
            h = case["height"]
            hero = case["hero"]
            subhead = case.get("subhead", "")
            badge = case.get("badge", "")
            steps = case.get("steps", [])
            cta = case.get("cta", "")
            store_info = case.get("store_info", "")
            orientation = case.get("orientation", "left")

            case_dir = out_dir / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            style_dict = case.get("style", {})
            theme_hex = style_dict.get("theme_color", "#D4AF37")
            bg_data_uri = generate_mock_background(w, h, orientation=orientation, theme_hex=theme_hex)

            style_obj = StyleConfig(
                font=style_dict.get("font", "bevietnam"),
                theme_color=theme_hex,
                text_effect=style_dict.get("text_effect", "plain_elegant"),
                background_tone=style_dict.get("background_tone", "dark_luxury"),
            )

            plan = TendooCreativePlan(
                template="step_process_roadmap",
                hero=hero,
                subhead=subhead,
                badge=badge,
                steps=steps,
                cta=cta,
                store_info=store_info,
                qr_code=case.get("qr_code"),
                qr_label=case.get("qr_label", "XEM TIẾP"),
                orientation=orientation,
                style=style_obj,
            )

            t0 = time.time()
            output_png = case_dir / "poster.png"
            output_html = case_dir / "poster.html"

            out_path, html_content = render_plan_to_poster(
                plan=plan,
                bg_data_uri=bg_data_uri,
                output_image_path=output_png,
                width=w,
                height=h,
            )

            with open(output_html, "w", encoding="utf-8") as f:
                f.write(html_content)

            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(f"file:///{output_html.resolve().as_posix()}")
            page.wait_for_timeout(350)

            # Chụp ảnh chất lượng cao
            page.screenshot(path=str(output_png))

            metrics = inspect_page_metrics(page, w, h)
            page.close()
            dur = time.time() - t0

            # Phân tích lỗi tràn
            overflow_count = 0
            canvas_overflow = False
            hero_font = 0

            for m in metrics:
                if "hero-title" in m["cls"]:
                    hero_font = m["fontSize"]
                if m["selfOverflowX"] or m["selfOverflowY"]:
                    overflow_count += 1
                if m["x1"] < -2 or m["y1"] < -2 or m["x2"] > w + 2 or m["y2"] > h + 2:
                    canvas_overflow = True

            status_sym = "✅" if (overflow_count == 0 and not canvas_overflow) else "❌"
            print(f"[{idx:02d}/{len(cases):02d}] {status_sym} {cid} ({w}x{h}, orient={orientation:5s}) - HeroFont: {hero_font:.1f}px - Overflow: {overflow_count} - Dur: {dur:.2f}s")

            results.append({
                "id": cid,
                "title": title,
                "width": w,
                "height": h,
                "hero": hero,
                "hero_font": hero_font,
                "duration_s": dur,
                "overflow_count": overflow_count,
                "canvas_overflow": canvas_overflow,
                "orientation": orientation
            })

        browser.close()

    # Ghi gallery
    gallery_html = build_gallery_html("step_process_roadmap (Bệ Quy Trình Lộ Trình)", results)
    gallery_path = out_dir / "gallery.html"
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write(gallery_html)

    total_overflow = sum(r["overflow_count"] for r in results)
    total_canvas_over = sum(1 for r in results if r["canvas_overflow"])

    print(f"\n================================================================================")
    print(f"🏁 HOÀN TẤT KIỂM THỬ step_process_roadmap:")
    print(f"   - Tổng test cases: {len(results)}")
    print(f"   - Tổng lỗi tự tràn (Self Overflow): {total_overflow}")
    print(f"   - Tổng lỗi tràn Canvas (Canvas Overflow): {total_canvas_over}")
    print(f"   - Báo cáo Gallery: {gallery_path.resolve().as_posix()}")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()
