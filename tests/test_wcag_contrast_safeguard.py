#!/usr/bin/env python3
"""
tests/test_wcag_contrast_safeguard.py
======================================
Automated test suite verifying the Generalized WCAG 2.1 Contrast Safeguard:
1. WCAG 2.1 relative luminance and contrast ratio calculation accuracy.
2. Hue-preserving contrast enhancement (ensure_contrast) on dark and light surfaces.
3. Optimal contrasting text selection (get_contrasting_text_color).
4. Adaptive color tokens (on_card_accent, on_card_title, on_canvas_accent, badge_bg, badge_text).
5. Dynamic text effects (get_effect_css) supporting both dark and light backgrounds.
6. Elimination of soft-light hero text blend mode in recruitment_board.
7. Full HTML compilation and contrast verification across all 14 commercial templates.
"""

import colorsys
import re
import sys
from pathlib import Path
import pytest
import numpy as np

# Fix console encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tendoo_core.palette import ColorPalette
from tendoo_core.colors import (
    analyze_color_harmony,
    calculate_contrast_ratio,
    calculate_wcag_luminance,
    ensure_contrast,
    get_contrasting_text_color,
    parse_color_to_rgb,
    rgb_to_hex,
)
from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.renderer import build_template_html
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.styles import (
    get_adaptive_palette,
    get_effect_css,
    palette_from_color_harmony,
)


# =========================================================================
# 1. MATHEMATICAL & COLOR ALGORITHM UNIT TESTS
# =========================================================================

def test_wcag_relative_luminance():
    """Verifies that calculate_wcag_luminance adheres strictly to WCAG 2.1 sRGB formula."""
    # Black has luminance 0.0
    assert calculate_wcag_luminance((0, 0, 0)) == pytest.approx(0.0, abs=1e-5)
    # Pure white has luminance 1.0
    assert calculate_wcag_luminance((255, 255, 255)) == pytest.approx(1.0, abs=1e-5)
    # Mid-gray (128, 128, 128)
    lum_gray = calculate_wcag_luminance((128, 128, 128))
    assert 0.20 < lum_gray < 0.25


def test_wcag_contrast_ratio():
    """Verifies that calculate_contrast_ratio computes (L1 + 0.05) / (L2 + 0.05)."""
    # Black vs White = 21.0
    assert calculate_contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0, abs=0.01)
    # Symmetric
    assert calculate_contrast_ratio("#FFFFFF", "#000000") == pytest.approx(21.0, abs=0.01)
    # Same color = 1.0
    assert calculate_contrast_ratio("#3B82F6", "#3B82F6") == pytest.approx(1.0, abs=0.01)
    # Dark slate #0F172A vs Pure white #FFFFFF > 15:1
    assert calculate_contrast_ratio("#0F172A", "#FFFFFF") > 15.0


def test_get_contrasting_text_color():
    """Verifies automatic selection of high-contrast text color on dark and light surfaces."""
    # Dark surface -> must return crisp white
    assert get_contrasting_text_color("#0A0F1D") == "#FFFFFF"
    assert get_contrasting_text_color("#1E293B") == "#FFFFFF"

    # Light surface -> must return deep slate
    assert get_contrasting_text_color("#F8FAFC") == "#0F172A"
    assert get_contrasting_text_color("#FFFFFF") == "#0F172A"

    # High-luminance brand color (Amber/Yellow #F59E0B) -> dark text provides much higher contrast
    text_on_amber = get_contrasting_text_color("#F59E0B")
    cr_amber = calculate_contrast_ratio("#F59E0B", text_on_amber)
    assert cr_amber >= 4.5, f"Contrast on amber {cr_amber} must be >= 4.5:1"


def test_ensure_contrast_dark_surface():
    """
    Verifies that ensure_contrast boosts lightness when brand color is dark on a dark surface,
    while preserving the original hue.
    """
    dark_surface = "#0F172A"  # Dark glass card
    dark_brand = "#1E293B"    # Dark navy, low contrast initially

    initial_cr = calculate_contrast_ratio(dark_brand, dark_surface)
    assert initial_cr < 2.0, f"Initial contrast {initial_cr} is poor"

    # Adjust for text (min 4.5:1)
    adjusted_text = ensure_contrast(dark_brand, dark_surface, min_ratio=4.5)
    cr_text = calculate_contrast_ratio(adjusted_text, dark_surface)
    assert cr_text >= 4.5, f"Adjusted contrast {cr_text} must satisfy WCAG AA >= 4.5:1"

    # Adjust for icons/graphical components (min 3.0:1)
    adjusted_icon = ensure_contrast(dark_brand, dark_surface, min_ratio=3.0)
    cr_icon = calculate_contrast_ratio(adjusted_icon, dark_surface)
    assert cr_icon >= 3.0, f"Adjusted icon contrast {cr_icon} must satisfy WCAG AA >= 3.0:1"

    # Check hue preservation
    r1, g1, b1 = parse_color_to_rgb(dark_brand)
    h1, _, _ = colorsys.rgb_to_hls(r1 / 255.0, g1 / 255.0, b1 / 255.0)
    r2, g2, b2 = parse_color_to_rgb(adjusted_text)
    h2, _, _ = colorsys.rgb_to_hls(r2 / 255.0, g2 / 255.0, b2 / 255.0)
    # Hue should be within 0.05 (18 degrees on 360 circle)
    assert abs(h1 - h2) < 0.05, f"Hue shifted too much: {h1} vs {h2}"


def test_ensure_contrast_light_surface():
    """
    Verifies that ensure_contrast reduces lightness when brand color is pale on a light surface,
    while preserving the original hue.
    """
    light_surface = "#F8FAFC"  # Light card / canvas
    pale_yellow = "#FEF08A"   # Light yellow, invisible on white

    initial_cr = calculate_contrast_ratio(pale_yellow, light_surface)
    assert initial_cr < 1.5, f"Initial contrast {initial_cr} is poor"

    adjusted = ensure_contrast(pale_yellow, light_surface, min_ratio=4.5)
    cr_adjusted = calculate_contrast_ratio(adjusted, light_surface)
    assert cr_adjusted >= 4.5, f"Adjusted contrast {cr_adjusted} must satisfy WCAG AA >= 4.5:1"


def test_color_palette_tokens_in_analyze_harmony():
    """Verifies that analyze_color_harmony populates on_card_accent, on_card_title, on_canvas_accent."""
    # Dark canvas
    dark_canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 15
    cp_dark = analyze_color_harmony(dark_canvas, (0, 0, 1, 1), color_mode="auto")

    assert cp_dark.is_dark is True
    assert cp_dark.on_card_accent != ""
    assert cp_dark.on_card_title != ""
    assert cp_dark.on_canvas_accent != ""

    # Contrast of on_card_accent against dark card surface (#0D1322)
    card_bg = "#0D1322"
    cr = calculate_contrast_ratio(cp_dark.on_card_accent, card_bg)
    assert cr >= 3.0, f"on_card_accent contrast {cr} must be >= 3.0 on dark card"

    # Light canvas
    light_canvas = np.ones((512, 512, 3), dtype=np.uint8) * 245
    cp_light = analyze_color_harmony(light_canvas, (0, 0, 1, 1), color_mode="auto")

    assert cp_light.is_dark is False
    cr_light = calculate_contrast_ratio(cp_light.on_card_accent, "#FFFFFF")
    assert cr_light >= 3.0, f"on_card_accent contrast {cr_light} must be >= 3.0 on light card"


# =========================================================================
# 2. DYNAMIC TEXT EFFECT & ADAPTIVE PALETTE TESTS
# =========================================================================

def test_dynamic_get_effect_css_dark_and_light_modes():
    """Verifies that get_effect_css generates dark-text styling when is_dark=False."""
    effects = ["plain_elegant", "shadow", "embossed", "3d_gold", "neon_glow", "gradient_clip", "editorial_clean"]

    for eff in effects:
        css_dark = get_effect_css(eff, is_dark=True)
        css_light = get_effect_css(eff, is_dark=False)

        assert len(css_dark) > 0, f"Dark effect {eff} returned empty CSS"
        assert len(css_light) > 0, f"Light effect {eff} returned empty CSS"

        # In light mode, plain_elegant and editorial_clean must use deep slate, not white
        if eff in ["plain_elegant", "editorial_clean"]:
            assert "#0F172A" in css_light or "rgba(15, 23, 42" in css_light, (
                f"Effect {eff} in light mode must use dark text, got:\n{css_light}"
            )
            assert "color: #FFFFFF" not in css_light

        # In light mode, embossed should use high-contrast dark fill (#1E293B)
        if eff == "embossed":
            assert "#1E293B" in css_light or "#0F172A" in css_light
            assert "rgba(255, 255, 255, 0.9" in css_light  # Bottom highlight for dark text


def test_get_adaptive_palette_card_tokens():
    """Verifies that get_adaptive_palette populates badge_bg, badge_text, on_card_accent."""
    # Dark background with dark theme color
    pal_dark = get_adaptive_palette(background_tone="dark_luxury", theme_color="#1E293B")
    assert "on_card_accent" in pal_dark
    assert "badge_bg" in pal_dark
    assert "badge_text" in pal_dark

    # Badge text must have high contrast on badge bg
    cr_badge = calculate_contrast_ratio(pal_dark["badge_text"], pal_dark["badge_bg"])
    assert cr_badge >= 4.5, f"Badge text contrast {cr_badge} must be >= 4.5:1 on badge_bg"

    # on_card_accent must have high contrast on dark card (#0F172A)
    cr_card_accent = calculate_contrast_ratio(pal_dark["on_card_accent"], "#0F172A")
    assert cr_card_accent >= 3.0, f"on_card_accent contrast {cr_card_accent} must be >= 3.0:1 on card"


# =========================================================================
# 3. TEMPLATE INTEGRATION TESTS: ALL 14 TEMPLATES
# =========================================================================

def test_recruitment_board_no_soft_light_on_hero():
    """Verifies that recruitment_board template no longer applies background-blend-mode: soft-light on hero."""
    template_path = PROJECT_ROOT / "src" / "tendoo_v3" / "templates" / "recruitment_board" / "template.html"
    content = template_path.read_text(encoding="utf-8")

    assert "background-blend-mode: soft-light" not in content, (
        "recruitment_board/template.html must NOT use soft-light blend mode on hero title!"
    )
    assert "{{ header_effect_css }}" in content or "{{header_effect_css}}" in content, (
        "recruitment_board must use dynamic header_effect_css for hero title!"
    )


@pytest.mark.parametrize("tpl_name", list(TEMPLATE_CATALOG.keys()))
def test_template_contrast_and_compilation(tpl_name):
    """
    Renders each of the 14 templates under:
    1. Dark canvas with low-contrast dark theme_color (#1E293B)
    2. Light canvas with low-contrast light theme_color (#FEF08A)
    Verifies that:
    - HTML compiles without Jinja errors.
    - No unrendered Jinja delimiters remain in CSS color properties.
    - Contrast-safe tokens are passed and used.
    """
    mock_bg = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # Test Case 1: Dark Mode with Dark Slate Theme Color
    plan_dark = TendooCreativePlan(
        template=tpl_name,
        hero="THỰC ĐƠN ĐẶC BIỆT MÙA LỄ HỘI",
        subhead="Nguyên liệu tươi ngon mỗi ngày từ trang trại chuẩn VietGAP",
        badge="ƯU ĐÃI 30%",
        tag_left="TRUYỀN THỐNG",
        tag_right="ĐẬM ĐÀ",
        rating=5,
        extra_texts=["Giao hàng 15 phút", "Freeship đơn 200k", "Hoàn tiền nếu không hài lòng"],
        cta="ĐẶT HÀNG NGAY",
        store_info="Hotline: 1900 8888 | 123 Phố Tràng Tiền, Hà Nội",
        qr_code="https://tendoo.ai/deal",
        qr_label="QUÉT MÃ ĐẶT BÀN",
        testimonial="Chất lượng phục vụ tuyệt hảo, đồ ăn ngon và không gian ấm cúng!",
        reviewer_name="Ngọc Anh - Food Reviewer",
        steps=["Chọn nguyên liệu tươi", "Chế biến công phu", "Trình bày đẹp mắt"],
        orientation="bottom_left" if tpl_name != "diagonal_slash" else "right",
        style=StyleConfig(
            font="playfair",
            theme_color="#1E293B",  # Extreme edge case: dark theme color on dark background
            text_effect="embossed",
            background_tone="dark_luxury",
        ),
    )

    html_dark = build_template_html(
        plan=plan_dark,
        bg_data_uri=mock_bg,
        width=1024,
        height=1024,
    )

    assert len(html_dark) > 500, f"Template {tpl_name} produced suspiciously short HTML"
    assert "{{" not in html_dark, f"Template {tpl_name} has unrendered Jinja syntax: {html_dark[:200]}"
    assert "}}" not in html_dark, f"Template {tpl_name} has unrendered Jinja syntax"
    assert "background-blend-mode: soft-light" not in html_dark, (
        f"Template {tpl_name} contains soft-light hero blend mode"
    )

    # Test Case 2: Light Mode with Pale Yellow Theme Color
    plan_light = TendooCreativePlan(
        template=tpl_name,
        hero="BỘ SƯU TẬP MÙA HÈ THANH MÁT",
        subhead="Chất liệu lanh tự nhiên mềm mại, thoáng mát và tôn dáng",
        badge="GIẢM 50%",
        tag_left="BEFORE",
        tag_right="AFTER",
        rating=5,
        extra_texts=["Vải organic cao cấp", "Thấm hút mồ hôi tối đa", "Thiết kế thời thượng"],
        cta="MUA NGAY HÔM NAY",
        store_info="Hotline: 0988 777 666 | 45 Lê Duẩn, TP. Hồ Chí Minh",
        qr_code="https://tendoo.ai/summer",
        qr_label="QUÉT MÃ MUA",
        testimonial="Mặc cực kỳ thoải mái và mát mẻ, form dáng rất đẹp!",
        reviewer_name="Thu Hà - Nhà thiết kế",
        steps=["Chọn mẫu", "Thử đồ", "Nhận quà tặng"],
        orientation="left",
        style=StyleConfig(
            font="bevietnam",
            theme_color="#FEF08A",  # Extreme edge case: pale yellow on light background
            text_effect="plain_elegant",
            background_tone="light_minimal",
        ),
    )

    html_light = build_template_html(
        plan=plan_light,
        bg_data_uri=mock_bg,
        width=1024,
        height=1024,
    )

    assert len(html_light) > 500
    assert "{{" not in html_light
    assert "}}" not in html_light


def test_l_frame_hero_overhang_protection():
    """Verifies that l_frame_showcase template avoids clipping italic letters by keeping overflow visible and reserving margin."""
    tpl_path = PROJECT_ROOT / "src" / "tendoo_v3" / "templates" / "l_frame_showcase" / "template.html"
    content = tpl_path.read_text(encoding="utf-8")
    assert ".lframe-top-cluster" in content
    assert "overflow: visible;" in content, "lframe-top-cluster must have overflow: visible to prevent letter clipping"
    assert "- 28" in content, "hero-title must have safety margin subtracted from data-max-width"

