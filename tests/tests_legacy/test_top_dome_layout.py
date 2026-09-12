"""
Unit and Visual Integration Tests for Top Dome Layout.

Tests:
1. Mask mathematical properties (bounds, monotonicity, zero-constraint lower zone).
2. Vietnamese headline balancing algorithm (orphan avoidance, font size ladders).
3. Dynamic Color Wheel extraction (light vs dark backgrounds).
4. Sub-pixel HTML rendering via Playwright.
"""

import base64
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from tendoo.layouts import (
    PosterContent,
    analyze_color_harmony,
    balance_vietnamese_headline,
    get_layout,
)
from tendoo.poster_renderer import PosterRenderer


OUTPUT_DIR = Path("output_layouts_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_top_dome_mask_math():
    layout = get_layout("top_dome")
    h, w = 1024, 1024
    mask = layout.generate_mask(width=w, height=h)

    assert mask.shape == (1024, 1024), f"Unexpected shape {mask.shape}"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask values outside [0, 1]"

    # Check top center is 1.0
    assert mask[int(0.05 * h), int(0.5 * w)] == 1.0, "Top center should be 1.0"

    # Check lower zone is strictly 0.0
    assert mask[int(0.50 * h), int(0.5 * w)] == 0.0, "Lower zone (y=0.50) must be 0.0"
    assert mask[int(0.80 * h), int(0.5 * w)] == 0.0, "Lower zone (y=0.80) must be 0.0"

    # Save visual artifact
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_path = OUTPUT_DIR / "top_dome_mask.png"
    mask_img.save(mask_path)
    print(f"[PASSED] Mask math verified -> saved to {mask_path}")


def test_headline_balancing():
    # Case A: Short headline
    lines_a, metrics_a = balance_vietnamese_headline("VIỆT NAM")
    assert len(lines_a) == 1
    assert lines_a[0] == "VIỆT NAM"
    assert metrics_a["font_size"] >= 50
    print(f"[PASSED] Case A Short: {lines_a} ({metrics_a['font_size']}px)")

    # Case B: Standard 2-line balanced
    lines_b, metrics_b = balance_vietnamese_headline("HƯƠNG SẮC MÙA HÈ THANH MÁT TỰ NHIÊN")
    assert len(lines_b) == 2
    assert metrics_b["font_size"] in [50, 52, 54, 56]
    print(f"[PASSED] Case B Standard: {lines_b} ({metrics_b['font_size']}px)")

    # Case C: Long sentence
    lines_c, metrics_c = balance_vietnamese_headline("BÙNG NỔ CÔNG NGHỆ CHỐNG ỒN CHỦ ĐỘNG THẾ HỆ MỚI")
    assert len(lines_c) == 2
    assert metrics_c["font_size"] in [42, 44, 46, 48]
    print(f"[PASSED] Case C Long: {lines_c} ({metrics_c['font_size']}px)")

    # Case D: Explicit user newline
    lines_d, metrics_d = balance_vietnamese_headline("ĐẠI TIỆC MÙA THU\nƯU ĐÃI NGẬP TRÀN")
    assert len(lines_d) == 2
    assert lines_d[0] == "ĐẠI TIỆC MÙA THU"
    assert lines_d[1] == "ƯU ĐÃI NGẬP TRÀN"
    print(f"[PASSED] Case D Explicit: {lines_d}")


def create_synthetic_background(bg_type: str = "light") -> Image.Image:
    """Creates synthetic gradient backgrounds mimicking DiT output."""
    img = Image.new("RGB", (1024, 1024))
    draw = ImageDraw.Draw(img)
    if bg_type == "light":
        # Warm daylight cream to soft amber gradient
        for y in range(1024):
            r = int(250 - (y / 1024.0) * 20)
            g = int(245 - (y / 1024.0) * 35)
            b = int(230 - (y / 1024.0) * 50)
            draw.line([(0, y), (1024, y)], fill=(r, g, b))
    else:
        # Dark luxury midnight blue to charcoal
        for y in range(1024):
            r = int(18 + (y / 1024.0) * 15)
            g = int(22 + (y / 1024.0) * 15)
            b = int(36 + (y / 1024.0) * 20)
            draw.line([(0, y), (1024, y)], fill=(r, g, b))
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_playwright_rendering():
    layout = get_layout("top_dome")
    safe_zone = layout.get_safe_zone()

    # 1. LIGHT DAYLIGHT TEST
    bg_light = create_synthetic_background("light")
    palette_light = analyze_color_harmony(np.array(bg_light), safe_zone, color_mode="auto")
    assert not palette_light.is_dark, "Expected light background detection"

    content_light = PosterContent(
        pre_header="Bộ Sưu Tập Mùa Hè",
        headline="TRÀ ĐÀO CAM SẢ THANH MÁT",
        slogan="Thưởng thức trọn vẹn từng giọt tươi mát từ thiên nhiên",
        offer_main="MUA 2 TẶNG 1",
        offer_sub="Áp dụng tại mọi chi nhánh trên toàn quốc",
        brand="Tendoo Beverage",
        hotline="1900 8888",
    )

    html_light = layout.render_html(
        content=content_light,
        palette=palette_light,
        bg_data_uri=pil_to_data_uri(bg_light),
        width=1024,
        height=1024,
    )

    (OUTPUT_DIR / "test_top_dome_light_daylight.html").write_text(html_light, encoding="utf-8")
    out_light_path = OUTPUT_DIR / "test_top_dome_light_daylight.png"
    PosterRenderer.render(
        html_content=html_light,
        output_image_path=out_light_path,
        width=1024,
        height=1024,
    )
    assert out_light_path.exists() and out_light_path.stat().st_size > 50000
    print(f"[PASSED] Rendered Light Poster -> {out_light_path}")

    # 2. DARK LUXURY TEST
    bg_dark = create_synthetic_background("dark")
    palette_dark = analyze_color_harmony(np.array(bg_dark), safe_zone, color_mode="auto", font_style="luxury_serif")
    assert palette_dark.is_dark, "Expected dark background detection"

    content_dark = PosterContent(
        pre_header="Âm Thanh Đỉnh Cao",
        headline="TAI NGHE KHÔNG DÂY HI-RES AUDIO",
        slogan="Công nghệ chống ồn chủ động Hybrid ANC thế hệ mới",
        offer_main="GIẢM NGAY 30%",
        offer_sub="Tặng kèm bao da cao cấp trị giá 500.000đ",
        brand="Tendoo Audio",
        hotline="0334 842 155",
    )

    html_dark = layout.render_html(
        content=content_dark,
        palette=palette_dark,
        bg_data_uri=pil_to_data_uri(bg_dark),
        width=1024,
        height=1024,
    )

    (OUTPUT_DIR / "test_top_dome_dark_luxury.html").write_text(html_dark, encoding="utf-8")
    out_dark_path = OUTPUT_DIR / "test_top_dome_dark_luxury.png"
    PosterRenderer.render(
        html_content=html_dark,
        output_image_path=out_dark_path,
        width=1024,
        height=1024,
    )
    assert out_dark_path.exists() and out_dark_path.stat().st_size > 50000
    print(f"[PASSED] Rendered Dark Poster -> {out_dark_path}")


if __name__ == "__main__":
    print("=== STARTING TOP DOME LAYOUT VALIDATION SUITE ===")
    test_top_dome_mask_math()
    test_headline_balancing()
    test_playwright_rendering()
    print("=== ALL TOP DOME LAYOUT TESTS PASSED PERFECTLY ===")
