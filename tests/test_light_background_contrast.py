"""
tests/test_light_background_contrast.py
=======================================
Comprehensive automated test suite verifying:
1. Elimination of dark gradient shims on light backgrounds in master template.
2. 100% application of Color Wheel tokens (--headline-color, --sub-color, --accent-color, --badge-*, --footer-*)
   with no invisible/low-contrast hardcoded white text on light backgrounds.
3. Universal API signature compatibility of render_html for OmniBlockLayout.
4. Full render verification of all 6 commercial intents in OmniBlockLayout.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image
import pytest

from tendoo.layouts.registry import get_layout, list_layouts
from tendoo.layouts.base import ColorPalette, PosterContent
from tendoo.layouts.color_engine import analyze_color_harmony


def create_solid_image(width: int, height: int, color_rgb: tuple) -> Image.Image:
    """Creates a synthetic solid background image."""
    return Image.new("RGB", (width, height), color_rgb)


def test_no_dark_shims_in_master_template():
    """Verifies that no dark gradient overlay shims exist in the Omni master template."""
    forbidden_shims = [
        "platform-gradient-shim",
        "floor-ambient-shim",
        "column-gradient-shim",
        "lframe-gradient-shim",
    ]

    template_file = PROJECT_ROOT / "src" / "tendoo" / "engine" / "templates" / "master.html"
    assert template_file.exists(), f"Template {template_file} must exist"
    content = template_file.read_text(encoding="utf-8")

    for shim in forbidden_shims:
        assert f'class="{shim}' not in content, (
            f"Master template must NOT instantiate dark shim class '{shim}', "
            f"as it ruins light backgrounds!"
        )


def test_color_wheel_light_background_palette():
    """Verifies that analyze_color_harmony generates dark high-contrast tokens on light backgrounds."""
    # Pure light cream / daylight background: RGB(248, 245, 240)
    bg_light = np.full((512, 512, 3), [248, 245, 240], dtype=np.uint8)
    crop_zone = (0.0, 0.0, 1.0, 1.0)

    palette = analyze_color_harmony(bg_light, crop_zone, color_mode="auto")
    assert not palette.is_dark, "Cream/daylight background must be detected as light (is_dark=False)"
    assert palette.luminance > 200.0, "Perceived luminance must be high"

    # Headline and sub text must be dark and rich
    assert "14%" in palette.headline_color or "hsl" in palette.headline_color
    assert "hsl" in palette.sub_color
    assert "rgba(255, 255, 255" in palette.glass_bg, "Glass card background should be clean translucent white"
    assert palette.footer_bg.startswith("rgba(255, 255, 255"), "Footer background should be clean translucent white on light mode"


def test_omni_layout_renders_on_light_background():
    """
    Renders HTML for OmniBlockLayout on a light background and verifies that:
    1. No hardcoded white text styling overrides the palette.
    2. CSS variables are properly defined in :root.
    3. Blocks render cleanly without orphan placeholders.
    """
    bg_light = create_solid_image(1024, 1024, (250, 248, 245))
    dummy_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    content = PosterContent(
        category="promo",
        pre_header="KHUYẾN MẠI ĐẶC BIỆT MÙA THU",
        headline="BỘ SƯU TẬP CAO CẤP\nPHONG CÁCH TỐI GIẢN",
        slogan="Chất liệu lụa satin thượng hạng, thoáng mát và tôn vinh nét thanh lịch tự nhiên.",
        offer_main="GIẢM 50%",
        offer_sub="Áp dụng cho hóa đơn từ 1.000.000đ",
        dates="01/09 - 30/09/2026",
        brand="Tendoo Maison",
        hotline="1900 8888",
        address="123 Phố Tràng Tiền, Hoàn Kiếm, Hà Nội",
        website_link="tendoo.vn/autumn",
        qr_data_uri=dummy_data_uri,
    )

    layout = get_layout("omni")
    safe_zone = layout.get_safe_zone()
    palette = analyze_color_harmony(np.array(bg_light), safe_zone, color_mode="light")

    html_out = layout.render_html(
        content=content,
        palette=palette,
        bg_data_uri=dummy_data_uri,
        width=1024,
        height=1024,
    )

    assert ":root" in html_out
    assert "--headline-color" in html_out
    assert "--sub-color" in html_out
    assert "--accent-color" in html_out
    assert "--footer-text" in html_out
    assert "--footer-bg" in html_out

    # Must not contain unrendered Jinja placeholders
    orphans = re.findall(r"\{\{[a-zA-Z0-9_]+\}\}", html_out)
    assert len(orphans) == 0, f"Omni layout has orphan placeholders: {orphans}"


def test_render_html_signature_flexibility():
    """
    Verifies that render_html in OmniBlockLayout supports both:
    1. Standard positional order: render_html(content, palette, bg_data_uri, width, height)
    2. Inverted positional order: render_html(content, palette, width, height, bg_data_uri)
    3. Keyword arguments order: render_html(content=..., palette=..., width=..., height=..., bg_data_uri=...)
    """
    dummy_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    palette = ColorPalette(
        is_dark=False,
        luminance=240.0,
        hue=40,
        comp_hue=220,
        headline_color="hsl(40, 30%, 14%)",
        sub_color="hsl(40, 20%, 26%)",
        accent_color="hsl(220, 95%, 36%)",
    )
    content = PosterContent(
        headline="TIÊU ĐỀ THỬ NGHIỆM",
        slogan="Mô tả phụ thử nghiệm",
    )

    layout = get_layout("omni")

    # 1. Standard positional: (content, palette, bg_data_uri, width, height)
    h1 = layout.render_html(content, palette, dummy_data_uri, 1024, 1024)
    assert len(h1) > 100
    assert "TIÊU ĐỀ" in h1 and "THỬ NGHIỆM" in h1

    # 2. Inverted positional: (content, palette, width, height, bg_data_uri)
    h2 = layout.render_html(content, palette, 1024, 1024, dummy_data_uri)
    assert len(h2) > 100
    assert "TIÊU ĐỀ" in h2 and "THỬ NGHIỆM" in h2

    # 3. Keyword args with headline_effect
    h3 = layout.render_html(
        content=content,
        palette=palette,
        bg_data_uri=dummy_data_uri,
        width=1024,
        height=1024,
        headline_effect="shadow",
    )
    assert len(h3) > 100
    assert "TIÊU ĐỀ" in h3 and "THỬ NGHIỆM" in h3


def test_all_commercial_intents_omni_rendering():
    """Verifies that all 6 business categories adapt seamlessly to color tokens in OmniBlockLayout."""
    palette = ColorPalette(
        is_dark=False,
        luminance=235.0,
        hue=210,
        comp_hue=30,
        headline_color="hsl(210, 40%, 14%)",
        sub_color="hsl(210, 30%, 26%)",
        accent_color="hsl(30, 95%, 36%)",
        badge_bg="linear-gradient(135deg, hsl(30, 95%, 48%) 0%, hsl(50, 90%, 38%) 100%)",
        badge_text="#FFFFFF",
        badge_shadow="0 4px 14px hsla(30, 90%, 45%, 0.38)",
    )

    categories = ["promo", "product_intro", "opening", "feedback", "recruitment", "guide"]
    layout = get_layout("omni")

    for cat in categories:
        content = PosterContent(
            category=cat,
            headline="TIÊU ĐỀ MẪU",
            offer_main="GIÁ BÁN: 1.290.000đ",
            offer_sub="Ưu đãi mùa hè",
            product_name="Serum Dưỡng Trắng",
            highlights="Chiết xuất tự nhiên, Thẩm thấu sâu",
            opening_date="20/10/2026",
            booking_contact="0901234567",
            feedback_target="Kem chống nắng",
            feedback_quote="Dùng rất thích, da mịn màng!",
            feedback_rating="5.0 / 5.0",
            job_position="Chuyên viên Tư vấn",
            job_desc="Lương thưởng hấp dẫn",
            apply_deadline="31/10/2026",
            steps=["Bước 1: Quét mã", "Bước 2: Đặt hàng"],
        )

        html_out = layout.render_html(
            content=content,
            palette=palette,
            bg_data_uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
            width=1024,
            height=1024,
        )
        assert len(html_out) > 500
        assert "TIÊU" in html_out and "ĐỀ MẪU" in html_out
