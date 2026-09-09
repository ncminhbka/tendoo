"""
Unit and Visual Integration Tests for Center Hourglass Layout.

Tests:
1. Hourglass mathematical mask properties (upper funnel, waist dip, floor perspective expansion).
2. 2-tier typography stream and floor footer integration.
3. Playwright sub-pixel rendering across festive and tech commercial campaigns.
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
    get_layout,
)
from tendoo.poster_renderer import PosterRenderer


OUTPUT_DIR = Path("output_layouts_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_hourglass_mask_math():
    layout = get_layout("center_hourglass")
    h, w = 1024, 1024
    mask = layout.generate_mask(width=w, height=h)

    assert mask.shape == (1024, 1024), f"Unexpected shape {mask.shape}"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask values outside [0, 1]"

    # Check upper light funnel center
    assert mask[int(0.05 * h), int(0.5 * w)] >= 0.98, "Top center should be ~1.0"

    # Check waist dip
    waist_val = mask[int(0.55 * h), int(0.5 * w)]
    assert 0.60 <= waist_val <= 0.85, f"Waist center should dip (got {waist_val})"

    # Check floor perspective expansion
    floor_val = mask[int(0.92 * h), int(0.5 * w)]
    assert floor_val >= 0.90, f"Floor center should expand (got {floor_val})"

    # Check sides at waist are zero
    assert mask[int(0.55 * h), int(0.08 * w)] == 0.0, "Waist outer flank must be 0.0"
    assert mask[int(0.55 * h), int(0.92 * w)] == 0.0, "Waist outer flank must be 0.0"

    # Save visual mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_path = OUTPUT_DIR / "center_hourglass_mask.png"
    mask_img.save(mask_path)
    print(f"[PASSED] Hourglass mask math verified -> saved to {mask_path}")


def create_warm_wood_background() -> Image.Image:
    """Creates synthetic moonlit festival background with warm wood floor."""
    img = Image.new("RGB", (1024, 1024))
    draw = ImageDraw.Draw(img)
    for y in range(1024):
        # Deep indigo/night blue upper sky fading into warm mahogany wood floor at bottom
        ratio = y / 1024.0
        if ratio < 0.70:
            r = int(22 + ratio * 20)
            g = int(24 + ratio * 15)
            b = int(45 - ratio * 10)
        else:
            t = (ratio - 0.70) / 0.30
            r = int(50 + t * 45)
            g = int(30 + t * 25)
            b = int(20 + t * 15)
        draw.line([(0, y), (1024, y)], fill=(r, g, b))
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_hourglass_playwright_rendering():
    layout = get_layout("center_hourglass")
    safe_zone = layout.get_safe_zone()

    bg = create_warm_wood_background()
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto", font_style="luxury_serif")

    # 1. Festive Mid-Autumn Campaign Test
    content_festive = PosterContent(
        pre_header="Tết Trông Trăng Đoàn Viên",
        headline="HỘP BÁNH TRUNG THU HOÀNG GIA THƯỢNG HẠNG",
        slogan="Món quà trọn vẹn ân tình gửi gắm gia đình",
        offer_main="CHIẾT KHẤU ĐẾN 20%",
        offer_sub="TẶNG KÈM TRÀ SEN TÂY HỒ CAO CẤP",
        dates="01/09 - 15/09/2026",
        applicable="Áp dụng cho mọi đơn hàng đặt trước trên toàn hệ thống.",
        brand="Tendoo Mooncake",
        hotline="0988 123 456",
    )

    html_festive = layout.render_html(
        content=content_festive,
        palette=palette,
        bg_data_uri=pil_to_data_uri(bg),
        width=1024,
        height=1024,
    )

    out_festive_path = OUTPUT_DIR / "test_hourglass_festive.png"
    PosterRenderer.render(
        html_content=html_festive,
        output_image_path=out_festive_path,
        width=1024,
        height=1024,
    )
    assert out_festive_path.exists() and out_festive_path.stat().st_size > 50000
    print(f"[PASSED] Rendered Hourglass Festive Poster -> {out_festive_path}")

    # 2. Modern Tech Campaign Test
    content_tech = PosterContent(
        pre_header="Siêu Đại Tiệc Công Nghệ",
        headline="BÙNG NỔ CÔNG NGHỆ CHỐNG ỒN THẾ HỆ MỚI",
        slogan="Trải nghiệm âm thanh vòm không gian sống động chuẩn rạp chiếu",
        offer_main="GIẢM SỐC 40%",
        offer_sub="FREESHIP TOÀN QUỐC CHO ĐƠN HÀNG HÔM NAY",
        dates="20/09 - 30/09/2026",
        applicable="Áp dụng khi thanh toán qua ví điện tử.",
        brand="Tendoo Audio Lab",
        hotline="1900 6868",
    )

    html_tech = layout.render_html(
        content=content_tech,
        palette=palette,
        bg_data_uri=pil_to_data_uri(bg),
        width=1024,
        height=1024,
    )

    out_tech_path = OUTPUT_DIR / "test_hourglass_tech.png"
    PosterRenderer.render(
        html_content=html_tech,
        output_image_path=out_tech_path,
        width=1024,
        height=1024,
    )
    assert out_tech_path.exists() and out_tech_path.stat().st_size > 50000
    print(f"[PASSED] Rendered Hourglass Tech Poster -> {out_tech_path}")


if __name__ == "__main__":
    print("=== STARTING CENTER HOURGLASS VALIDATION SUITE ===")
    test_hourglass_mask_math()
    test_hourglass_playwright_rendering()
    print("=== ALL CENTER HOURGLASS TESTS PASSED PERFECTLY ===")
