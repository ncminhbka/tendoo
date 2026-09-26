#!/usr/bin/env python3
"""
scripts/test_l_frame_showcase.py

Kịch bản kiểm thử toàn diện cho template L-Frame Showcase (chuẩn theo 2 ảnh mẫu SUV Mercedes & Áo khoác dạ):
- Chạy toàn bộ 16 test cases từ `tests/test_l_frame_showcase_suite.json`.
- Bao trùm 4 tỉ lệ: 1:1 (1024x1024), 9:16 (576x1024), 16:9 (1024x576), 4:5 (816x1024).
- Bao trùm 4 cấp độ text: Minimal, Light, Medium, Heavy.
- Bao trùm 2 chiều Mirror: Left Anchor (mặc định) và Right Mirror.
- Render ảnh poster PNG bằng Playwright Chromium trên nền Mock chất lượng cao.
- Đo lường chính xác từng phần tử DOM:
    + Kiểm tra tự tràn hộp (self-overflow / cắt chữ).
    + Kiểm tra tràn ngoài canvas.
    + Đo kích thước font Hero thực tế (kiểm tra co giãn nhị phân).
- Xuất báo cáo HTML Gallery trực quan tại `output_tendoo_v3/l_frame_showcase_gallery/gallery.html`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

from tendoo_core.colors import analyze_color_harmony
from tendoo_v3.demo_server import _extract_primary_crop_zone
from tendoo_v3.mask_engine import generate_template_mask, save_mask_preview
from tendoo_v3.mock_backgrounds import create_gradient_backdrop
from tendoo_v3.renderer import pil_to_base64_data_uri, render_plan_to_poster
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.styles import palette_from_color_harmony

TEST_SUITE_PATH = PROJECT_ROOT / "tests" / "test_l_frame_showcase_suite.json"
OUTPUT_DIR = PROJECT_ROOT / "output_tendoo_v3" / "l_frame_showcase_gallery"


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
          parentCls: (el.parentElement && el.parentElement.className) ? el.parentElement.className.toString() : '',
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
        orient = res.get("orientation", "left")
        status_color = "#10B981" if (overflow_count == 0 and not canvas_overflow) else "#EF4444"
        status_text = "PASS HOÀN TOÀN" if (overflow_count == 0 and not canvas_overflow) else f"CÓ LỖI: {overflow_count} lỗi"

        ratio_label = "1:1 Vuông" if w == h else ("9:16 Dọc" if w < 600 else ("16:9 Ngang" if w > 1000 and h < 600 else "4:5 Dọc"))

        cards_html.append(f"""
        <div class="case-card">
          <div class="card-header">
            <div class="card-badge" style="background:{status_color};">{status_text}</div>
            <div class="card-meta">
              <span class="ratio-pill">{ratio_label} ({w}x{h})</span>
              <span class="orient-pill">Hướng: {orient.upper()}</span>
              <span class="time-pill">⚡ {dur:.2f}s</span>
            </div>
          </div>
          <div class="card-body">
            <div class="poster-preview">
              <a href="{cid}/poster.png" target="_blank">
                <img src="{cid}/poster.png" alt="{cid}" loading="lazy" />
              </a>
            </div>
            <div class="mask-preview">
              <a href="{cid}/mask.png" target="_blank">
                <img src="{cid}/mask.png" alt="{cid} mask" loading="lazy" />
              </a>
            </div>
          </div>
          <div class="card-footer">
            <h3 class="case-title">{title}</h3>
            <p class="hero-line"><strong>Hero:</strong> "{hero}"</p>
            <div class="metrics-row">
              <span>Cỡ Font Hero: <strong>{hero_font:.1f}px</strong></span>
              <span>Lỗi Tự Tràn: <strong style="color:{status_color}">{overflow_count}</strong></span>
              <span>Tràn Canvas: <strong style="color:{status_color}">{'Có' if canvas_overflow else 'Không'}</strong></span>
            </div>
            <div class="view-links">
              <a href="{cid}/poster.html" target="_blank">Xem File HTML Trực Tiếp</a>
            </div>
          </div>
        </div>
        """)

    total_pass = sum(1 for r in results if r["overflow_count"] == 0 and not r["canvas_overflow"])
    total_cases = len(results)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Gallery Kiểm Thử: L-Frame Showcase (16 Cases)</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #090D16;
      color: #F1F5F9;
      padding: 32px 40px;
      line-height: 1.5;
    }}
    .gallery-header {{
      text-align: center;
      margin-bottom: 40px;
      padding-bottom: 24px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .gallery-header h1 {{
      font-size: 32px;
      font-weight: 800;
      color: #FFF;
      letter-spacing: -0.5px;
      margin-bottom: 8px;
    }}
    .stat-bar {{
      display: inline-flex;
      align-items: center;
      gap: 16px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 9999px;
      padding: 8px 24px;
      font-size: 15px;
      margin-top: 12px;
    }}
    .grid-container {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 28px;
    }}
    .case-card {{
      background: #131B2E;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }}
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 18px;
      background: rgba(0, 0, 0, 0.3);
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .card-badge {{
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 9999px;
      color: #FFF;
    }}
    .card-meta {{
      display: flex;
      gap: 8px;
      font-size: 12px;
    }}
    .ratio-pill, .orient-pill, .time-pill {{
      background: rgba(255, 255, 255, 0.1);
      padding: 3px 8px;
      border-radius: 6px;
      color: #94A3B8;
    }}
    .orient-pill {{
      color: #38BDF8;
      font-weight: 700;
    }}
    .card-body {{
      display: flex;
      background: #0B0F19;
      padding: 12px;
      gap: 12px;
      align-items: center;
      justify-content: center;
    }}
    .poster-preview {{
      flex: 2.2;
      text-align: center;
    }}
    .poster-preview img {{
      max-width: 100%;
      max-height: 380px;
      border-radius: 8px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.6);
      display: block;
      margin: 0 auto;
    }}
    .mask-preview {{
      flex: 1;
      text-align: center;
    }}
    .mask-preview img {{
      max-width: 100%;
      max-height: 160px;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      display: block;
      margin: 0 auto;
    }}
    .card-footer {{
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .case-title {{
      font-size: 15px;
      font-weight: 700;
      color: #F8FAFC;
    }}
    .hero-line {{
      font-size: 12.5px;
      color: #94A3B8;
    }}
    .metrics-row {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: #CBD5E1;
      padding-top: 6px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .view-links {{
      margin-top: 4px;
      text-align: right;
    }}
    .view-links a {{
      color: #38BDF8;
      font-size: 12px;
      text-decoration: none;
      font-weight: 600;
    }}
    .view-links a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="gallery-header">
    <h1>BÁO CÁO KIỂM THỬ: TEMPLATE L-FRAME SHOWCASE</h1>
    <p style="color: #94A3B8;">Thiết kế chuẩn 2 poster mẫu SUV Mercedes & Áo khoác dạ nữ • Hỗ trợ Mirror Trái/Phải 100% • Quét mã QR thực tế</p>
    <div class="stat-bar">
      <span>Tổng số ca kiểm thử: <strong>{total_cases}</strong></span>
      <span>Kết quả: <strong style="color: {'#10B981' if total_pass == total_cases else '#EF4444'};">{total_pass}/{total_cases} PASS ({total_pass/total_cases*100:.1f}%)</strong></span>
    </div>
  </div>

  <div class="grid-container">
    {''.join(cards_html)}
  </div>
</body>
</html>
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="l_frame_showcase stress-test runner")
    parser.add_argument("--suite", type=str, default=str(TEST_SUITE_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args, _ = parser.parse_known_args()
    suite_path = Path(args.suite).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(suite_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print("=" * 80)
    print("TENDOO V3 - CHẠY KIỂM THỬ L-FRAME SHOWCASE SUITE (16 CASES)")
    print(f"Bao trùm: 4 tỉ lệ (1:1, 9:16, 16:9, 4:5) x 4 cấp độ text x 2 hướng Mirror")
    print("=" * 80)

    results = []
    t0_total = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for idx, tc in enumerate(test_cases, 1):
            cid = tc["id"]
            title = tc["title"]
            w = tc["width"]
            h = tc["height"]
            plan_dict = tc["plan"]
            orient = plan_dict.get("orientation", "left")

            case_dir = output_dir / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{idx:02d}/{len(test_cases)}] Chạy: {cid} | ({w}x{h}) {orient.upper()} - {title}")
            t_start = time.time()

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

            # 2. Continuous mask (has_freetext PHẢI khớp bool(plan.extra_texts) như
            # build_template_html tự tính nội bộ -- xem catalog.py slots[...]["drives_geometry"]).
            mask_np = generate_template_mask(
                template=plan.template,
                width=w,
                height=h,
                orientation=orient,
                blur_radius_px=18,
                has_freetext=bool(plan.extra_texts),
            )
            mask_path = case_dir / "mask.png"
            save_mask_preview(mask_np, str(mask_path))

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

            # 4. Đo lường phần tử DOM bằng Playwright
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            page.set_content(html_content, wait_until="networkidle")

            # Đo lường phần tử
            elements = measure_text_elements_in_page(page, w, h)

            # Lấy cỡ font của Hero
            hero_el = next((e for e in elements if plan.hero[:15].lower() in e["text"].lower()), None)
            hero_font = hero_el["fontSize"] if hero_el else 0.0

            # `measure_text_elements_in_page` CHỈ đo phần tử LÁ (bỏ qua phần tử có
            # children) -- không bao giờ bắt được lỗi "tổng các con vượt quá hộp cha
            # cố định" (yeu_cau_templates.txt mục kinh nghiệm #3), đã từng để lọt lỗi
            # cắt mất chữ thật ở `.lframe-top-cluster` (overflow:hidden + max-height
            # cố định) khi badge+subhead+freetext cùng có mặt. Đo bổ sung riêng các
            # container cố định chiều cao để bắt đúng lớp lỗi này.
            container_overflow = page.evaluate("""
            () => {
              const out = [];
              document.querySelectorAll('.lframe-top-cluster').forEach(el => {
                const over = el.scrollHeight > el.clientHeight + 6.0;
                out.push({cls: el.className, over, scrollH: el.scrollHeight, clientH: el.clientHeight});
              });
              return out;
            }
            """)
            container_overflow_count = sum(1 for c in container_overflow if c["over"])

            # Đếm lỗi tràn
            overflow_count = sum(1 for e in elements if e["selfOverflowX"] or e["selfOverflowY"]) + container_overflow_count
            canvas_overflow = any(
                e["x1"] < -2.0 or e["y1"] < -2.0 or e["x2"] > (w + 2.0) or e["y2"] > (h + 2.0)
                for e in elements
            )

            dur = time.time() - t_start
            page.close()

            status_icon = "✓" if (overflow_count == 0 and not canvas_overflow) else "🚨"
            print(f"    {status_icon} Hoàn thành ({dur:.2f}s) | Hero Font: {hero_font:.1f}px | Lỗi tự tràn: {overflow_count} | Tràn canvas: {canvas_overflow}")

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
                "orientation": orient,
            })

        browser.close()

    # Tạo file gallery.html
    gallery_path = output_dir / "gallery.html"
    gallery_path.write_text(build_gallery_html(results), encoding="utf-8")

    total_pass = sum(1 for r in results if r["overflow_count"] == 0 and not r["canvas_overflow"])
    total_dur = time.time() - t0_total

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ TỔNG KẾT L-FRAME SHOWCASE: {total_pass}/{len(test_cases)} CASES PASS HOÀN TOÀN")
    print(f"Tổng thời gian thực thi: {total_dur:.2f}s")
    print(f"Xem báo cáo trực quan Gallery tại: file:///{gallery_path.as_posix()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
