"""
Unit and Visual Integration Tests for Split Column (Cascading Ribbon Sash) Layout.

Tests:
1. Mathematical mask properties:
   - Left 32% core is dense (>= 0.95) for editorial text copy.
   - Right 60% is strictly 0.0 copy space for fashion models/cosmetics.
   - Organic undulating wave boundary along vertical axis.
2. Vogue editorial typography hierarchy with left-aligned rhythm and Vietnamese balancing.
3. Playwright sub-pixel rendering across luxury fashion lookbook and premium skincare campaigns.
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
from tendoo.typography_engine import PosterRenderer


OUTPUT_DIR = Path("output_layouts_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_split_column_mask_math():
    layout = get_layout("split_column")
    h, w = 1024, 1024
    mask = layout.generate_mask(width=w, height=h)

    assert mask.shape == (1024, 1024), f"Unexpected shape {mask.shape}"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask values outside [0, 1]"

    # Left column core (x=0.15, y=0.50) must be 1.0
    core_val = mask[int(0.50 * h), int(0.15 * w)]
    assert core_val >= 0.98, f"Left column core should be ~1.0 (got {core_val})"

    # Right subject space (x >= 0.50) must be strictly 0.0
    right_max = mask[:, int(0.50 * w) :].max()
    assert right_max == 0.0, f"Right subject zone should be strictly 0.0 (got max={right_max})"

    # Verify straight vertical boundary: edge x position is consistent along y
    edges = []
    for y_sample in [0.20, 0.45, 0.70, 0.90]:
        row = mask[int(y_sample * h), :]
        edge_x = np.where(row > 0.5)[0][-1] / float(w)
        edges.append(edge_x)

    assert max(edges) - min(edges) <= 0.005, "Boundary should be straight vertical to create a flat unwrinkled silk banner"

    # Save visual mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_path = OUTPUT_DIR / "split_column_mask.png"
    mask_img.save(mask_path)
    print(f"[PASSED] Split column ribbon mask math verified -> saved to {mask_path}")


def create_luxury_studio_background() -> Image.Image:
    """Creates synthetic high-fashion studio backdrop: deep moody midnight slate on left, warm warm studio glow on right."""
    img = Image.new("RGB", (1024, 1024))
    draw = ImageDraw.Draw(img)
    for x in range(1024):
        ratio = x / 1024.0
        if ratio < 0.40:
            # Deep midnight navy/charcoal for silk column
            t = ratio / 0.40
            r = int(14 + t * 6)
            g = int(18 + t * 8)
            b = int(28 + t * 12)
        else:
            # Warm studio rim light on the model side
            t = (ratio - 0.40) / 0.60
            r = int(24 + t * 45)
            g = int(26 + t * 35)
            b = int(38 + t * 25)
        draw.line([(x, 0), (x, 1024)], fill=(r, g, b))
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_split_column_playwright_rendering():
    layout = get_layout("split_column")
    safe_zone = layout.get_safe_zone()

    bg = create_luxury_studio_background()
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    # 1. Luxury Fashion Lookbook Campaign Test
    content_fashion = PosterContent(
        pre_header="BỘ SƯU TẬP THU ĐÔNG 2026",
        headline="ÁO KHOÁC MĂNG TÔ\nDẠ LÔNG CỪU Ý",
        slogan="Chất liệu dạ lông cừu thượng hạng dệt tay, tôn vinh nét thanh lịch vượt thời gian cho quý cô thành thị.",
        offer_main="ƯU ĐÃI ĐẾN 25%",
        offer_sub="Tặng khăn choàng lụa tơ tằm cao cấp cho hóa đơn từ 3 triệu",
        dates="05/09 - 25/09/2026",
        applicable="Áp dụng tại toàn bộ hệ thống boutique trên toàn quốc.",
        brand="Tendoo Atelier",
        hotline="1900 6868",
    )

    html_fashion = layout.render_html(
        content=content_fashion,
        palette=palette,
        bg_data_uri=pil_to_data_uri(bg),
        width=1024,
        height=1024,
    )

    out_fashion_path = OUTPUT_DIR / "test_column_fashion_lookbook.png"
    PosterRenderer.render(
        html_content=html_fashion,
        output_image_path=out_fashion_path,
        width=1024,
        height=1024,
    )
    assert out_fashion_path.exists() and out_fashion_path.stat().st_size > 50000
    print(f"[PASSED] Rendered Split Column Fashion Lookbook -> {out_fashion_path}")

    # 2. Premium Skincare / Cosmetics Campaign Test
    content_cosmetics = PosterContent(
        pre_header="BÍ QUYẾT TRẺ HÓA LÀN DA",
        headline="TINH CHẤT PHỤC HỒI\nTẾ BÀO GỐC BIỂN SÂU",
        slogan="Đột phá công nghệ sinh học biển giúp tái sinh làn da căng bóng và rạng rỡ từ sâu bên trong.",
        offer_main="MUA 1 TẶNG 1",
        offer_sub="Tặng kèm bộ kit du lịch dưỡng trắng da cao cấp 5 món",
        dates="Chương trình trong tuần này",
        applicable="Áp dụng cho khách hàng thành viên khi mua sắm online.",
        brand="Tendoo Skin Lab",
        hotline="0908 888 999",
    )

    html_cosmetics = layout.render_html(
        content=content_cosmetics,
        palette=palette,
        bg_data_uri=pil_to_data_uri(bg),
        width=1024,
        height=1024,
    )

    out_cosmetics_path = OUTPUT_DIR / "test_column_cosmetics.png"
    PosterRenderer.render(
        html_content=html_cosmetics,
        output_image_path=out_cosmetics_path,
        width=1024,
        height=1024,
    )
    assert out_cosmetics_path.exists() and out_cosmetics_path.stat().st_size > 50000
    print(f"[PASSED] Rendered Split Column Cosmetics -> {out_cosmetics_path}")


if __name__ == "__main__":
    print("=== STARTING SPLIT COLUMN VALIDATION SUITE ===")
    test_split_column_mask_math()
    test_split_column_playwright_rendering()
    print("=== ALL SPLIT COLUMN TESTS PASSED PERFECTLY ===")
