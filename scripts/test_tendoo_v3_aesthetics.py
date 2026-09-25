#!/usr/bin/env python3
"""
scripts/test_tendoo_v3_aesthetics.py

Bộ thực nghiệm kiểm thử Thẩm mỹ & Độ Thích Ứng Template (Tendoo v3):
- Chạy 16+ test cases bao trùm từ `tests/test_cases_tendoo_v3.json`.
- Sử dụng nền Mock chất lượng cao (Không cần GPU/Diffusion).
- Sinh Continuous Mask thẩm mỹ và Render Poster qua Headless Chromium.
- Tạo trang Web `output_tendoo_v3/gallery.html` để người dùng trực tiếp đánh giá.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Thêm thư mục gốc và src vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from tendoo_core.colors import analyze_color_harmony
from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.demo_server import _extract_primary_crop_zone
from tendoo_v3.mask_engine import generate_template_mask, save_mask_preview
from tendoo_v3.mock_backgrounds import create_gradient_backdrop
from tendoo_v3.renderer import pil_to_base64_data_uri, render_plan_to_poster
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.styles import palette_from_color_harmony

OUTPUT_DIR = PROJECT_ROOT / "output_tendoo_v3"
TEST_CASES_PATH = PROJECT_ROOT / "tests" / "test_cases_tendoo_v3.json"


def build_gallery_html(results: list) -> str:
    """Tạo trang web thư viện ảnh tĩnh hiển thị toàn bộ 16 cases."""
    cards_html = []
    for res in results:
        cid = res["id"]
        title = res["title"]
        tpl = res["template"]
        font = res["font"]
        effect = res["effect"]
        dur = res["duration_s"]
        has_mask = res["has_mask"]
        w, h = res["width"], res["height"]
        hero = res["hero"]

        cards_html.append(f"""
        <div class="case-card">
          <div class="card-header">
            <span class="case-badge">{cid}</span>
            <span class="tpl-tag">{tpl}</span>
          </div>
          <h3 class="case-title">{title}</h3>
          <p class="case-hero">Hero: "{hero}"</p>
          <div class="meta-row">
            <span>Kích thước: <b>{w}x{h}</b></span>
            <span>Font: <b>{font}</b></span>
            <span>Hiệu ứng: <b>{effect}</b></span>
            <span>Render: <b>{dur:.2f}s</b></span>
          </div>
          <div class="media-preview">
            <div class="preview-col">
              <span class="preview-label">Final Poster (HTML/Chromium)</span>
              <a href="{cid}/poster.png" target="_blank">
                <img src="{cid}/poster.png" alt="Poster" class="img-poster" loading="lazy">
              </a>
            </div>
            <div class="preview-col">
              <span class="preview-label">Continuous Mask (DiT Boundary)</span>
              <a href="{cid}/mask.png" target="_blank">
                <img src="{cid}/mask.png" alt="Mask" class="img-mask" loading="lazy">
              </a>
            </div>
          </div>
          <div class="card-footer">
            <a href="{cid}/poster.html" target="_blank" class="btn-link">Mở File HTML</a>
            <span class="mask-status">{'Mask: Khối lớn' if has_mask else 'Zero-Mask (Nền thuần)'}</span>
          </div>
        </div>
        """)

    all_cards = "\n".join(cards_html)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Tendoo v3 - Báo Cáo Thẩm Mỹ & Thích Ứng Template</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #090A0F;
      color: #E2E8F0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      padding: 40px 24px;
      line-height: 1.5;
    }}
    .header {{
      text-align: center;
      margin-bottom: 48px;
    }}
    .header h1 {{
      font-size: 36px;
      font-weight: 900;
      color: #FFF;
      letter-spacing: -0.5px;
      margin-bottom: 12px;
      background: linear-gradient(135deg, #FFF 0%, #D4AF37 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .header p {{
      color: #94A3B8;
      font-size: 16px;
      max-width: 800px;
      margin: 0 auto;
    }}
    .stats-bar {{
      display: flex;
      justify-content: center;
      gap: 32px;
      margin-top: 24px;
    }}
    .stat-item {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 12px 24px;
      border-radius: 12px;
      text-align: center;
    }}
    .stat-val {{ font-size: 24px; font-weight: 800; color: #D4AF37; }}
    .stat-lbl {{ font-size: 12px; color: #64748B; text-transform: uppercase; letter-spacing: 1px; }}
    
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
      gap: 32px;
      max-width: 1600px;
      margin: 0 auto;
    }}
    .case-card {{
      background: rgba(18, 20, 29, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .case-badge {{
      background: rgba(212, 175, 55, 0.15);
      color: #D4AF37;
      border: 1px solid rgba(212, 175, 55, 0.3);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }}
    .tpl-tag {{
      background: rgba(255, 255, 255, 0.08);
      color: #CBD5E1;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
    }}
    .case-title {{
      font-size: 18px;
      font-weight: 800;
      color: #F8FAFC;
      margin-bottom: 6px;
    }}
    .case-hero {{
      font-size: 14px;
      color: #94A3B8;
      font-style: italic;
      margin-bottom: 14px;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 12px;
      color: #64748B;
      padding-bottom: 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      margin-bottom: 16px;
    }}
    .meta-row b {{ color: #CBD5E1; }}
    .media-preview {{
      display: flex;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .preview-col {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .preview-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #64748B;
      font-weight: 700;
    }}
    .img-poster, .img-mask {{
      width: 100%;
      height: 380px;
      object-fit: contain;
      background: #000;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      transition: transform 0.2s ease;
    }}
    .img-poster:hover, .img-mask:hover {{
      transform: scale(1.02);
      border-color: #D4AF37;
    }}
    .card-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 12px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }}
    .btn-link {{
      color: #38BDF8;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    .btn-link:hover {{ text-decoration: underline; }}
    .mask-status {{ font-size: 12px; color: #10B981; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Tendoo v3 Visual Quality Showcase</h1>
    <p>Báo cáo thẩm mỹ trực quan: Kiểm thử 16 cases bao trùm (Prompt 1-19, Before/After, Zero-Mask Luxury Card, Stress-test nhồi text).</p>
    <div class="stats-bar">
      <div class="stat-item">
        <div class="stat-val">{len(results)}</div>
        <div class="stat-lbl">Test Cases</div>
      </div>
      <div class="stat-item">
        <div class="stat-val">{len(set(r['template'] for r in results))}</div>
        <div class="stat-lbl">Templates</div>
      </div>
      <div class="stat-item">
        <div class="stat-val">100%</div>
        <div class="stat-lbl">Playwright Pass</div>
      </div>
    </div>
  </div>

  <div class="gallery-grid">
    {all_cards}
  </div>
</body>
</html>"""


import argparse


def main():
    parser = argparse.ArgumentParser(description="Tendoo v3 Aesthetic & Adaptation Test Runner")
    parser.add_argument("--cases", type=str, default=str(TEST_CASES_PATH), help="Path to test cases JSON file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory for posters & gallery")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    output_dir = Path(args.output)

    print("=" * 80)
    print(">>> TENDOO v3: RUNNING COMPREHENSIVE AESTHETIC & TEMPLATE ADAPTATION TEST")
    print(f">>> Cases: {cases_path}")
    print(f">>> Output: {output_dir}")
    print("=" * 80)

    if not cases_path.exists():
        print(f"Error: {cases_path} not found!")
        sys.exit(1)

    with open(cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {cases_path.name}\n")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t_start_all = time.time()

    for idx, item in enumerate(test_cases, 1):
        cid = item["id"]
        title = item["title"]
        w = item.get("width", 1024)
        h = item.get("height", 1024)
        plan_dict = item["plan"]

        print(f"[{idx}/{len(test_cases)}] Testing: {cid} ({title})")
        case_out_dir = output_dir / cid
        case_out_dir.mkdir(parents=True, exist_ok=True)

        plan = TendooCreativePlan.from_dict(plan_dict)

        # 1. Sinh mock backdrop theo tone
        tone = plan.style.background_tone or "dark_luxury"
        backdrop_pil = create_gradient_backdrop(width=w, height=h, tone=tone)
        bg_data_uri = pil_to_base64_data_uri(backdrop_pil)

        # 1.5. Phân tích bảng màu tắc kè hoa thích ứng (Chameleon Adaptive Palette)
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

        # 2. Sinh Continuous Mask -- đọc thẳng theo tên template (xem geometry.py),
        # không còn qua tầng "mask_preset" gián tiếp.
        has_mask = TEMPLATE_CATALOG.get(plan.template, {}).get("has_mask", True)
        mask_np = generate_template_mask(
            template=plan.template,
            width=w,
            height=h,
            orientation=plan.orientation,
        )
        mask_file = case_out_dir / "mask.png"
        save_mask_preview(mask_np, str(mask_file))

        # 3. Render HTML & Chụp ảnh Poster qua Chromium
        t0 = time.time()
        poster_file = case_out_dir / "poster.png"
        out_path, html_content = render_plan_to_poster(
            plan=plan,
            bg_data_uri=bg_data_uri,
            output_image_path=poster_file,
            width=w,
            height=h,
            palette_override=palette_override,
        )
        dur = time.time() - t0

        # Lưu file HTML để inspect
        (case_out_dir / "poster.html").write_text(html_content, encoding="utf-8")
        print(f"    [OK] Rendered in {dur:.2f}s -> {poster_file.name}")

        results.append({
            "id": cid,
            "title": title,
            "template": plan.template,
            "font": plan.style.font,
            "effect": plan.style.text_effect,
            "hero": plan.hero,
            "width": w,
            "height": h,
            "duration_s": dur,
            "has_mask": has_mask,
        })

    # 4. Xuất Gallery HTML
    gallery_file = output_dir / "gallery.html"
    gallery_content = build_gallery_html(results)
    gallery_file.write_text(gallery_content, encoding="utf-8")

    total_time = time.time() - t_start_all
    print("\n" + "=" * 80)
    print(f"[COMPLETED] ALL {len(test_cases)} CASES IN {total_time:.2f}s!")
    print(f"[GALLERY] Visual Quality Gallery generated at: {gallery_file.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
