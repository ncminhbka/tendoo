#!/usr/bin/env python3
"""
scripts/render_all_14_templates_gallery.py

Render 1 case đại diện cho ĐỦ CẢ 14 template trong TEMPLATE_CATALOG lên nền mock
(gradient backdrop, không cần GPU) kèm mask preview, để xem nhanh toàn bộ catalog
trong 1 gallery HTML duy nhất -- không phải test suite CI, chỉ phục vụ xem trực quan
1 lần theo yêu cầu user (2026-09-20).

Tái dùng machinery render+mask+gallery đã có sẵn, đã chạy tốt trong
`scripts/test_master_templates.py` (measure_text_elements_in_page/build_gallery_html)
thay vì viết lại -- suite gốc của nó (`tests/test_master_templates_suite.json`) chỉ có
12/14 template (thiếu sandwich_top_heavy, sandwich_bottom_heavy, menu_price_board),
nên script này bổ sung đúng 3 case còn thiếu (lấy case đầu tiên của suite JSON riêng
từng template đó, vốn đã kiểm chứng dùng tốt) rồi chạy full 14 case vào 1 output dir
mới, không đụng tới suite JSON gốc dùng cho CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
from playwright.sync_api import sync_playwright

import test_master_templates as mt  # tái dùng measure_text_elements_in_page/build_gallery_html
from tendoo_v3.colors import analyze_color_harmony
from tendoo_v3.demo_server import _extract_primary_crop_zone
from tendoo_v3.mask_engine import generate_template_mask, save_mask_preview
from tendoo_v3.mock_backgrounds import create_gradient_backdrop
from tendoo_v3.renderer import pil_to_base64_data_uri, render_plan_to_poster
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.styles import palette_from_color_harmony

OUTPUT_DIR = PROJECT_ROOT / "output_tendoo_v3" / "all_14_templates_gallery"

# 3 case bổ sung cho đúng 3 template thiếu trong tests/test_master_templates_suite.json
# -- lấy nguyên case đầu tiên (đã kiểm chứng dùng tốt) từ suite JSON riêng của từng template.
EXTRA_CASES = [
    {
        "id": "extra_01_sandwich_top_heavy",
        "title": "Sandwich Top Heavy - Siêu Sale Mỹ Phẩm",
        "width": 1024,
        "height": 1024,
        "plan": {
            "template": "sandwich_top_heavy",
            "style": {
                "font": "montserrat",
                "theme_color": "#FFD700",
                "text_effect": "3d_gold",
                "background_tone": "dark_luxury",
            },
            "hero": "SIÊU SALE 50%",
            "store_info": "Hotline: 1900 8888",
            "qr_code": "https://tendoo.ai/sale",
            "qr_label": "QUÉT MÃ MUA",
            "scene_prompt": "Luxury golden cosmetic bottle standing on dark marble podium, cinematic lighting, 8k",
            "corridor_prompt": "Dark luxury studio gradient background, soft gold ambient lighting, clean copy space",
        },
    },
    {
        "id": "extra_02_sandwich_bottom_heavy",
        "title": "Sandwich Bottom Heavy - Siêu Sale Mỹ Phẩm",
        "width": 1024,
        "height": 1024,
        "plan": {
            "template": "sandwich_bottom_heavy",
            "style": {
                "font": "montserrat",
                "theme_color": "#FFD700",
                "text_effect": "3d_gold",
                "background_tone": "dark_luxury",
            },
            "hero": "SIÊU SALE 50%",
            "store_info": "Hotline: 1900 8888",
            "qr_code": "https://tendoo.ai/sale",
            "qr_label": "QUÉT MÃ MUA",
            "scene_prompt": "Luxury golden cosmetic bottle standing on dark marble podium, cinematic lighting, 8k",
            "corridor_prompt": "Dark luxury studio gradient background, soft gold ambient lighting, clean copy space",
        },
    },
    {
        "id": "extra_03_menu_price_board",
        "title": "Menu Price Board - Quán Cà Phê Specialty",
        "width": 1024,
        "height": 1024,
        "plan": {
            "template": "menu_price_board",
            "orientation": "left",
            "hero": "THỰC ĐƠN QUÁN",
            "subhead": "Cà phê rang mộc chuẩn vị Việt",
            "badge": "GIẢM 10% CUỐI TUẦN",
            "extra_texts": [
                "Cà Phê Sữa Đá - 25.000đ",
                "Bạc Xỉu - 29.000đ",
                "Trà Đào Cam Sả - 35.000đ",
                "Trà Vải Hoa Hồng - 38.000đ",
                "Cold Brew Nguyên Chất - 45.000đ",
            ],
            "cta": "ĐẶT NGAY HÔM NAY",
            "store_info": "Hotline: 1900 6868 - 88 Phố Huế, Hà Nội",
            "qr_code": "https://tendoo.ai/menu-cafe",
            "qr_label": "QUÉT MÃ ĐẶT BÀN",
            "style": {
                "theme_color": "#D4AF37",
                "text_effect": "plain_elegant",
                "background_tone": "dark_luxury",
            },
            "scene_prompt": "Warm cozy artisan coffee shop counter, roasted beans, soft ambient lighting, 8k",
            "corridor_prompt": "Warm wooden counter surface extending into soft bokeh, matching cafe ambient tones",
        },
    },
]


def main():
    t_start = time.time()
    print("=" * 80)
    print("TENDOO V3 - RENDER ĐỦ 14 TEMPLATE (MOCK BACKDROP + MASK) VÀO 1 GALLERY")
    print("=" * 80)

    base_cases = json.loads(mt.TEST_SUITE_PATH.read_text(encoding="utf-8"))
    cases = base_cases + EXTRA_CASES
    templates_covered = sorted({c["plan"]["template"] for c in cases})
    print(f"✓ Tổng {len(cases)} case, phủ {len(templates_covered)} template: {templates_covered}")
    assert len(templates_covered) == 14, f"Thiếu template, chỉ có {len(templates_covered)}/14"

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
            plan = TendooCreativePlan.from_dict(item["plan"])
            tpl = plan.template

            print(f"[{idx:02d}/{len(cases):02d}] {cid} | Template: {tpl} ({w}x{h})")
            case_dir = OUTPUT_DIR / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            tone = plan.style.background_tone or "dark_luxury"
            backdrop_pil = create_gradient_backdrop(width=w, height=h, tone=tone)
            bg_data_uri = pil_to_base64_data_uri(backdrop_pil)

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

            mask_np = generate_template_mask(template=plan.template, width=w, height=h, orientation=plan.orientation)
            save_mask_preview(mask_np, str(case_dir / "mask.png"))

            t0 = time.time()
            poster_file = case_dir / "poster.png"
            out_path, html_content = render_plan_to_poster(
                plan=plan, bg_data_uri=bg_data_uri, output_image_path=poster_file,
                width=w, height=h, palette_override=palette_override,
            )
            dur = time.time() - t0

            html_file = case_dir / "poster.html"
            html_file.write_text(html_content, encoding="utf-8")

            page.set_viewport_size({"width": w, "height": h})
            page.goto(html_file.as_uri())
            try:
                page.wait_for_function("window.__tendooAutofitDone === true", timeout=2500)
            except Exception:
                page.wait_for_timeout(300)

            elements = mt.measure_text_elements_in_page(page, w, h)
            overflow_count = 0
            canvas_overflow = False
            hero_font = 0.0
            for el in elements:
                if "hero-title" in el["cls"] or "hero-title" in el["parentCls"] or "hero-top-title" in el["cls"]:
                    hero_font = max(hero_font, el["fontSize"])
                if el["selfOverflowX"] or el["selfOverflowY"]:
                    overflow_count += 1
                    print(f"    ⚠️ [TỰ TRÀN HỘP] <{el['tag']}>: \"{el['text']}\"")
                if el["x1"] < -1.0 or el["y1"] < -1.0 or el["x2"] > w + 1.0 or el["y2"] > h + 1.0:
                    canvas_overflow = True
                    print(f"    🚨 [TRÀN NGOÀI CANVAS] <{el['tag']}>: \"{el['text']}\"")

            print(f"    ✓ ({dur:.2f}s) Hero Font: {hero_font:.1f}px | Lỗi tự tràn: {overflow_count} | Tràn canvas: {canvas_overflow}")

            results.append({
                "id": cid, "title": title, "template": tpl, "width": w, "height": h,
                "hero": plan.hero, "hero_font": hero_font, "duration_s": dur,
                "overflow_count": overflow_count, "canvas_overflow": canvas_overflow,
            })

        browser.close()

    gallery_path = OUTPUT_DIR / "gallery.html"
    gallery_html = mt.build_gallery_html(results).replace(
        "Toàn Bộ 13 Template", "Toàn Bộ 14 Template"
    ).replace(
        "(12 CASES / 13 TEMPLATES)", "(14 CASES / 14 TEMPLATES)"
    )
    gallery_path.write_text(gallery_html, encoding="utf-8")

    total_time = time.time() - t_start
    total_pass = sum(1 for r in results if r["overflow_count"] == 0 and not r["canvas_overflow"])
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ: {total_pass}/{len(results)} case pass hoàn toàn | {total_time:.2f}s")
    print(f"Gallery: {gallery_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
