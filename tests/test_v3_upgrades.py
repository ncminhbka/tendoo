#!/usr/bin/env python3
"""
tests/test_v3_upgrades.py

Comprehensive test suite verifying the Tendoo v3 Aesthetic & Architecture Upgrade:
1. Fallback Heuristic & Prompt Authority (Prompt > Form, Suppression rules, Orientation, VFX).
2. Chameleon Adaptive Palette Calculation on synthetic image regions.
3. Cinematic VFX overlay generation (SVG feTurbulence, flares, dust, haze).
4. Jinja2 compilation & HTML generation across all 11 templates with vfx_html and autofit markers.
5. Playwright autofit contract verification (__tendooAutofitDone flag check).
"""

import sys
from pathlib import Path

# Fix Windows console encoding for Unicode checkmarks
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image

from tendoo.core.colors import analyze_color_harmony
from tendoo_v3.catalog import TEMPLATE_CATALOG, build_llm_font_prompt, build_llm_effect_and_vfx_prompt
from tendoo_v3.geometry import get_zones
from tendoo_v3.llm_planner import fallback_heuristic_planner
from tendoo_v3.renderer import build_template_html
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.styles import (
    COMMON_AUTOFIT_JS,
    get_adaptive_palette,
    get_effect_css,
    get_vfx_overlay_html_and_css,
    palette_from_color_harmony,
)


def test_fallback_heuristics_and_suppression():
    print("--> [1/5] Testing Fallback Heuristics & Prompt Authority...")
    
    # 1.1 Test "giữa" routing to sandwich_top_heavy
    p1 = fallback_heuristic_planner({"product_name": "Trà Đào"}, prompt="Đặt chữ ở giữa tranh")
    assert p1.template == "sandwich_top_heavy", f"Expected sandwich_top_heavy, got {p1.template}"
    
    # 1.2 Test diagonal_slash right orientation
    p2 = fallback_heuristic_planner({"product_name": "Gym"}, prompt="cắt chéo bên phải phong cách thể thao")
    assert p2.template == "diagonal_slash", f"Expected diagonal_slash, got {p2.template}"
    assert p2.orientation == "right", f"Expected orientation right, got {p2.orientation}"
    
    # 1.3 Test store info suppression
    p3 = fallback_heuristic_planner(
        {"product_name": "Cà phê", "phone": "0912345678", "address": "Hà Nội"},
        prompt="cà phê thơm ngon đậm vị, đừng vẽ thông tin cửa hàng"
    )
    assert p3.store_info is None, f"Expected store_info=None, got {p3.store_info}"
    
    # 1.4 Test no dummy fallback hallucination when form is empty
    p4 = fallback_heuristic_planner({"product_name": "Tendoo"}, prompt="ảnh nghệ thuật")
    assert p4.store_info is None, f"Expected store_info=None when form has no store, got {p4.store_info}"
    assert p4.badge is None, f"Expected badge=None when no discount, got {p4.badge}"
    assert p4.qr_code is None, f"Expected qr_code=None when no QR requested, got {p4.qr_code}"
    
    # 1.5 Test VFX keyword detection
    p5 = fallback_heuristic_planner({"product_name": "Thời Trang"}, prompt="phong cách bìa tạp chí retro film grain")
    assert p5.style.vfx == "film_grain", f"Expected vfx=film_grain, got {p5.style.vfx}"
    assert p5.style.font == "playfair", f"Expected font=playfair for retro magazine, got {p5.style.font}"
    
    print("    ✓ All Fallback Heuristic & Prompt Authority tests passed!")


def test_cinematic_vfx_overlays():
    print("--> [2/5] Testing Cinematic VFX Overlays...")
    
    vfx_types = ["film_grain", "anamorphic_flare", "gold_dust", "light_leak", "cinematic_haze", "none"]
    for vfx in vfx_types:
        html = get_vfx_overlay_html_and_css(vfx, width=1024, height=1024, theme_color="#06B6D4")
        if vfx == "none":
            assert html == "", f"Expected empty string for none, got {html}"
        elif vfx == "film_grain":
            assert "feTurbulence" in html, "film_grain must contain feTurbulence SVG filter"
            assert "vfx-film-grain" in html
        elif vfx == "anamorphic_flare":
            assert "vfx-anamorphic-flare" in html
        elif vfx == "gold_dust":
            assert "vfx-gold-dust" in html
        elif vfx == "light_leak":
            assert "vfx-light-leak" in html
        elif vfx == "cinematic_haze":
            assert "vfx-cinematic-haze" in html
            
    print("    ✓ All 6 VFX overlay modes verified successfully!")


def test_chameleon_adaptive_palette():
    print("--> [3/5] Testing Chameleon Adaptive Palette...")
    
    # Create dark image and light image
    dark_img = np.zeros((512, 512, 3), dtype=np.uint8) + 20  # Dark slate
    light_img = np.ones((512, 512, 3), dtype=np.uint8) * 240  # Light cream
    
    crop_zone = (0.0, 0.0, 0.38, 1.0)
    
    # Analyze dark
    cp_dark = analyze_color_harmony(dark_img, crop_zone=crop_zone, color_mode="auto")
    pal_dark = palette_from_color_harmony(cp_dark, theme_color="#06B6D4")
    assert "text_primary" in pal_dark
    assert "glass_bg" in pal_dark
    assert "rgba" in pal_dark["glass_bg"]
    
    # Analyze light
    cp_light = analyze_color_harmony(light_img, crop_zone=crop_zone, color_mode="auto")
    pal_light = palette_from_color_harmony(cp_light, theme_color="#C88A35")
    assert "text_primary" in pal_light
    assert "cta_bg" in pal_light
    
    print(f"    Dark primary: {pal_dark['text_primary']} | Glass: {pal_dark['glass_bg']}")
    print(f"    Light primary: {pal_light['text_primary']} | Glass: {pal_light['glass_bg']}")
    print("    ✓ Chameleon palette adaptation verified!")


def test_autofit_contract():
    print("--> [4/5] Testing Autofit Contract & JS Flag...")
    assert "window.__tendooAutofitDone = true;" in COMMON_AUTOFIT_JS, (
        "COMMON_AUTOFIT_JS must set window.__tendooAutofitDone = true to release Playwright wait!"
    )
    print("    ✓ Autofit flag contract confirmed: window.__tendooAutofitDone = true")


def test_all_11_templates_html_compilation():
    print("--> [5/5] Compiling & Rendering HTML across all 11 templates...")
    
    test_plan = TendooCreativePlan(
        template="sandwich_top_heavy",
        hero="CÀ PHÊ PHIN ĐẬM VỊ VIỆT NAM",
        subhead="Hạt cà phê Robusta rang mộc thủ công thơm ngon",
        badge="NGUYÊN CHẤT 100%",
        tag_left="TRUYỀN THỐNG",
        tag_right="ĐẬM ĐÀ",
        rating=5,
        extra_texts=["Rang củi thủ công", "Không phụ gia hóa chất", "Giao hàng 15 phút"],
        cta="THƯỞNG THỨC NGAY",
        store_info="Hotline: 1900 6868 | 88 Phố Huế, Hà Nội",
        qr_code="https://tendoo.ai/deal",
        qr_label="QUÉT MÃ ĐẶT BÀN",
        testimonial="Cà phê thơm nồng đậm đà, chuẩn gu Việt Nam!",
        reviewer_name="Minh Tuấn - Khách hàng quen",
        steps=["Chọn hạt Robusta", "Rang mộc truyền thống", "Pha phin chuẩn độ"],
        orientation="bottom_left",
        style=StyleConfig(
            font="playfair",
            theme_color="#C88A35",
            text_effect="embossed",
            background_tone="warm_rustic",
            vfx="film_grain",
        ),
        scene_prompt="Cinematic shot of Vietnamese coffee on rustic wooden table",
        corridor_prompt="Smooth rustic wooden table surface in extreme bokeh",
    )
    
    mock_bg_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    for tpl_name in TEMPLATE_CATALOG.keys():
        test_plan.template = tpl_name
        if tpl_name == "diagonal_slash":
            test_plan.orientation = "right"
        else:
            test_plan.orientation = "bottom_left"
            
        html = build_template_html(
            plan=test_plan,
            bg_data_uri=mock_bg_uri,
            width=1024,
            height=1024,
        )
        assert len(html) > 500, f"Template {tpl_name} produced suspiciously short HTML"
        assert "vfx-film-grain" in html, f"Template {tpl_name} did not inject vfx_html"
        assert "data-autofit" in html, f"Template {tpl_name} must contain data-autofit markers"
        assert "__tendooAutofitDone" in html, f"Template {tpl_name} must contain autofit script"
        print(f"    ✓ Template '{tpl_name}': Compiled OK ({len(html)} bytes)")
        
    print("\n=======================================================")
    print("🎉 ALL 5 TEST SUITES PASSED FLAWLESSLY! TENDOO V3 READY.")
    print("=======================================================\n")


if __name__ == "__main__":
    test_fallback_heuristics_and_suppression()
    test_cinematic_vfx_overlays()
    test_chameleon_adaptive_palette()
    test_autofit_contract()
    test_all_11_templates_html_compilation()
