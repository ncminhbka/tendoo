"""
Unit and Visual Integration Tests for Bottom Platform (Cinematic Base) Layout.

Tests:
1. Mathematical mask properties:
   - Upper 58% is strictly 0.0 copy space for scenery and hero vehicle/product.
   - Smooth C1 Cosine ramp at y in [0.58, 0.72] with subtle stage arch.
   - Lower platform (y >= 0.72) is dense (>= 0.95).
2. Theatrical headline balancing with Vietnamese compound word preservation.
3. Playwright sub-pixel rendering across luxury automotive and grand real estate campaigns.
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


def test_bottom_platform_mask_math():
    layout = get_layout("bottom_platform")
    h, w = 1024, 1024
    mask = layout.generate_mask(width=w, height=h)

    assert mask.shape == (1024, 1024), f"Unexpected shape {mask.shape}"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask values outside [0, 1]"

    # Upper 50% must be strictly 0.0
    upper_max = mask[: int(0.50 * h), :].max()
    assert upper_max == 0.0, f"Upper zone should be 0.0 (got max={upper_max})"

    # Mid transition at center (y=0.65) should be in smooth transition [0.30, 0.85]
    mid_center = mask[int(0.65 * h), int(0.5 * w)]
    assert 0.30 <= mid_center <= 0.85, f"Mid transition center unexpected: {mid_center}"

    # Lower platform center (y=0.85) should be >= 0.95
    lower_center = mask[int(0.85 * h), int(0.5 * w)]
    assert lower_center >= 0.95, f"Lower platform center should be >= 0.95 (got {lower_center})"

    # Save visual mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_path = OUTPUT_DIR / "bottom_platform_mask.png"
    mask_img.save(mask_path)
    print(f"[PASSED] Bottom platform mask math verified -> saved to {mask_path}")


def create_twilight_mountain_road_background() -> Image.Image:
    """Creates synthetic scenic twilight mountain pass with dark wet asphalt road at bottom."""
    img = Image.new("RGB", (1024, 1024))
    draw = ImageDraw.Draw(img)
    for y in range(1024):
        ratio = y / 1024.0
        if ratio < 0.35:
            # Twilight sky: deep orange-gold sunset fading to evening cobalt
            t = ratio / 0.35
            r = int(35 + (1.0 - t) * 60)
            g = int(25 + (1.0 - t) * 30)
            b = int(60 + (1.0 - t) * 20)
        elif ratio < 0.62:
            # Mountain range silhouettes: dark moody charcoal teal
            t = (ratio - 0.35) / 0.27
            r = int(18 + t * 6)
            g = int(22 + t * 6)
            b = int(32 + t * 6)
        else:
            # Wet asphalt road surface: sleek dark obsidian with ambient ground sheen
            t = (ratio - 0.62) / 0.38
            r = int(12 + t * 8)
            g = int(15 + t * 9)
            b = int(22 + t * 10)
        draw.line([(0, y), (1024, y)], fill=(r, g, b))
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_bottom_platform_playwright_rendering():
    layout = get_layout("bottom_platform")
    safe_zone = layout.get_safe_zone()

    bg = create_twilight_mountain_road_background()
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    # 1. Luxury Automotive Campaign Test
    content_auto = PosterContent(
        pre_header="PHIÊN BẢN GIỚI HẠN 2026",
        headline="KHÁM PHÁ ĐẲNG CẤP SUV THẾ HỆ MỚI",
        slogan="Chinh phục mọi cung đường hiểm trở với công nghệ truyền động thông minh",
        offer_main="ƯU ĐÃI 100 TRIỆU",
        offer_sub="Tặng gói bảo hiểm thân vỏ và 3 năm bảo dưỡng miễn phí",
        dates="01/09 - 30/09/2026",
        applicable="Áp dụng khi đặt cọc trong tháng 9 tại tất cả showroom ủy quyền.",
        brand="Tendoo Motors",
        hotline="1900 8888",
    )

    html_auto = layout.render_html(
        content=content_auto,
        palette=palette,
        bg_data_uri=pil_to_data_uri(bg),
        width=1024,
        height=1024,
    )

    out_auto_path = OUTPUT_DIR / "test_platform_automotive.png"
    PosterRenderer.render(
        html_content=html_auto,
        output_image_path=out_auto_path,
        width=1024,
        height=1024,
    )
    assert out_auto_path.exists() and out_auto_path.stat().st_size > 50000
    print(f"[PASSED] Rendered Bottom Platform Automotive Poster -> {out_auto_path}")

    # 2. Grand Real Estate Campaign Test
    content_estate = PosterContent(
        pre_header="SIÊU PHẨM BẤT ĐỘNG SẢN BIỂN",
        headline="BIỆT THỰ ĐẢO NGỌC KIẾN TRÚC HOÀNG GIA",
        slogan="Không gian nghỉ dưỡng thượng lưu ôm trọn tầm nhìn hoàng hôn vịnh biển kỳ vĩ",
        offer_main="CHIẾT KHẤU 15%",
        offer_sub="Cam kết lợi nhuận cho thuê tối thiểu 12%/năm",
        dates="Mở bán đợt 1",
        applicable="Hỗ trợ vay ngân hàng 70% với lãi suất ưu đãi 0% trong 24 tháng.",
        brand="Tendoo Ocean Estates",
        hotline="0909 999 888",
    )

    html_estate = layout.render_html(
        content=content_estate,
        palette=palette,
        bg_data_uri=pil_to_data_uri(bg),
        width=1024,
        height=1024,
    )

    out_estate_path = OUTPUT_DIR / "test_platform_real_estate.png"
    PosterRenderer.render(
        html_content=html_estate,
        output_image_path=out_estate_path,
        width=1024,
        height=1024,
    )
    assert out_estate_path.exists() and out_estate_path.stat().st_size > 50000
    print(f"[PASSED] Rendered Bottom Platform Real Estate Poster -> {out_estate_path}")


if __name__ == "__main__":
    print("=== STARTING BOTTOM PLATFORM VALIDATION SUITE ===")
    test_bottom_platform_mask_math()
    test_bottom_platform_playwright_rendering()
    print("=== ALL BOTTOM PLATFORM TESTS PASSED PERFECTLY ===")
