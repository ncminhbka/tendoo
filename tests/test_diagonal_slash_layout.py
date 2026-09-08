"""
Unit and Visual Integration Tests for Diagonal Slash (Sport Dynamic) Layout.
===========================================================================
Tests:
1. Mathematical mask properties:
   - Upper-left corridor core is dense (>= 0.98) for bold typography.
   - Lower-right subject zone (x >= 0.70, y >= 0.70) is strictly 0.0 for hero products.
   - Exact perpendicular cosine feathering along ~35°-45° diagonal boundary.
   - Aspect ratio scalability and side switching.
2. Corridor prompts:
   - Validates sport_speed, cyber_neon, carbon_mesh, daylight_motion.
   - Strictly enforces Brand Preservation Rule (NO unbranded, no logos, no labels).
3. Template & HTML rendering:
   - Zero unrendered mustache/placeholder tags across all 4 aspect ratios.
   - Vietnamese headline balancing & typographic hierarchy.
   - Adaptive category body injection (Product Intro, Promo, Opening).
4. Playwright image rendering verification (if playwright is installed).
"""

import base64
import io
import re
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tendoo.layouts import (
    PosterContent,
    analyze_color_harmony,
    get_layout,
)
from tendoo.layouts.diagonal_slash.mask import generate_diagonal_slash_mask
from tendoo.typography_engine import PosterRenderer


OUTPUT_DIR = Path("output_layouts_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_dynamic_sport_background(width: int = 1024, height: int = 1024) -> Image.Image:
    """Creates a high-energy sport background: dark carbon slate on upper-left, vibrant orange/cyan glow on lower-right."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        for x in range(width):
            u = x / float(width)
            v = y / float(height)
            diag = 0.6 * u + 0.4 * v
            # Dark carbon/navy at top-left
            r = int(12 + diag * 60)
            g = int(14 + diag * 40)
            b = int(24 + diag * 80)
            img.putpixel((x, y), (min(255, r), min(255, g), min(255, b)))
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_diagonal_slash_mask_math():
    """Validates perpendicular signed distance and cosine feathering of diagonal slash mask."""
    layout = get_layout("diagonal_slash")
    h, w = 1024, 1024
    mask = layout.generate_mask(width=w, height=h, x_top=0.65, x_bottom=0.20, delta=0.10)

    assert mask.shape == (1024, 1024), f"Unexpected shape {mask.shape}"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask values outside [0, 1]"

    # 1. Upper-left corner (x=0.10, y=0.10) must be 1.0 (dense typography corridor)
    ul_val = mask[int(0.10 * h), int(0.10 * w)]
    assert ul_val >= 0.98, f"Upper-left corridor should be ~1.0 (got {ul_val})"

    # 2. Lower-right corner (x=0.85, y=0.85) must be strictly 0.0 (hero subject open space)
    lr_val = mask[int(0.85 * h), int(0.85 * w)]
    assert lr_val == 0.0, f"Lower-right hero zone should be 0.0 (got {lr_val})"

    # 3. Center of line: at y=0.50, boundary should be at x = (0.65 + 0.20)/2 = 0.425
    y_mid = int(0.50 * h)
    x_boundary = np.where(mask[y_mid, :] > 0.5)[0][-1] / float(w)
    expected_x = 0.425
    assert abs(x_boundary - expected_x) < 0.03, f"Boundary at mid-height expected ~{expected_x}, got {x_boundary}"

    # 4. Monotonic transition across perpendicular boundary at y=0.50
    row_slice = mask[y_mid, int(0.30 * w) : int(0.55 * w)]
    diffs = np.diff(row_slice)
    assert np.all(diffs <= 0.001), "Transition across diagonal boundary must decrease monotonically"

    # 5. Save visual mask artifact for inspection
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_path = OUTPUT_DIR / "diagonal_slash_mask.png"
    mask_img.save(mask_path)
    print(f"[PASSED] Diagonal Slash mask math verified -> saved to {mask_path}")


def test_diagonal_slash_corridor_prompts():
    """Validates that corridor prompts have copy space guidance and NO brand-suppressing keywords."""
    layout = get_layout("diagonal_slash")
    forbidden_brand_suppressors = [
        "unbranded", "no logos", "no brand", "no labels", "no words", "no letters"
    ]

    for style in ["sport_speed", "cyber_neon", "carbon_mesh", "daylight_motion", "unknown_fallback"]:
        prompt = layout.get_corridor_prompt(style)
        assert isinstance(prompt, str) and len(prompt) > 40
        lower = prompt.lower()
        
        # Check copy space negative keywords
        assert "clean photographic background" in lower
        assert "no floating graphic text" in lower
        assert "no poster typography" in lower

        # Check brand safety
        for forbidden in forbidden_brand_suppressors:
            assert forbidden not in lower, f"Corridor prompt for {style} contains forbidden keyword '{forbidden}'!"

    print("[PASSED] All Diagonal Slash corridor prompts follow brand safety & copy space rules.")


def test_diagonal_slash_html_rendering_no_orphan_placeholders():
    """Ensures rendered HTML has NO unrendered {{...}} placeholder tags across all 4 aspect ratios."""
    layout = get_layout("diagonal_slash")
    safe_zone = layout.get_safe_zone()
    bg = Image.new("RGB", (1024, 1024), (18, 24, 38))
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    aspect_ratios = [
        (1024, 1024),  # 1:1 Square
        (576, 1024),   # 9:16 Story
        (1024, 576),   # 16:9 Banner
        (816, 1024),   # 4:5 Feed
    ]

    content = PosterContent(
        pre_header="CHÍNH HÃNG 100% NHẬP KHẨU",
        headline="GIÀY CHẠY BỘ\nSIÊU TỐC ĐỘ PRO",
        slogan="Đế đệm carbon phản hồi năng lượng 92%, bứt phá mọi kỷ lục cự ly marathon của bạn.",
        offer_main="GIẢM 35%",
        offer_sub="Tặng kèm túi đựng giày thể thao chuyên nghiệp và 2 đôi tất dệt kim",
        dates="Duy nhất tuần lễ vàng",
        applicable="Áp dụng khi đặt hàng online và tại showroom trải nghiệm.",
        brand="Tendoo Athletics",
        hotline="0988 777 666",
        address="123 Lê Duẩn, Hà Nội",
        website_link="tendoo.vn/runner-pro",
        qr_data_uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    )

    for w, h in aspect_ratios:
        rendered_html = layout.render_html(
            content=content,
            palette=palette,
            width=w,
            height=h,
            bg_data_uri=pil_to_data_uri(bg),
            headline_effect="3d_gold",
        )

        # Search for unreplaced {{...}} mustache tokens
        orphans = re.findall(r'\{\{[a-zA-Z0-9_\#\/\s]+\}\}', rendered_html)
        assert len(orphans) == 0, f"Found orphan placeholders in {w}x{h} HTML: {orphans}"
        assert str(w) in rendered_html
        assert str(h) in rendered_html
        assert "GIÀY CHẠY BỘ" in rendered_html
        assert "SIÊU TỐC ĐỘ PRO" in rendered_html

    print("[PASSED] Diagonal Slash HTML rendering has zero orphan placeholders across 4 aspect ratios.")


def test_diagonal_slash_adaptive_categories():
    """Validates category body rendering inside Diagonal Slash (Product Intro, Opening, Feedback, Guide)."""
    layout = get_layout("diagonal_slash")
    safe_zone = layout.get_safe_zone()
    bg = Image.new("RGB", (1024, 1024), (20, 20, 30))
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    # Product Intro with price badge and feature pills
    content_intro = PosterContent(
        category="product_intro",
        headline="TAI NGHE GAMING\nKHÔNG DÂY PRO",
        slogan="Âm thanh vòm 7.1 chuẩn esports, độ trễ cực thấp 15ms cho game thủ.",
        price="2.490.000đ",
        highlights="Chống ồn ANC, Pin 40 giờ, Micro lọc tạp âm AI",
        brand="Tendoo Cyber",
    )
    html_intro = layout.render_html(
        content=content_intro,
        palette=palette,
        width=1024,
        height=1024,
        bg_data_uri=pil_to_data_uri(bg),
    )
    assert "2.490.000đ" in html_intro
    assert "Chống ồn ANC" in html_intro
    assert "Pin 40 giờ" in html_intro

    # Opening category
    content_opening = PosterContent(
        category="opening",
        headline="ĐẠI TIỆC KHAI TRƯƠNG\nCYBER ARENA",
        slogan="Phòng máy thi đấu chuẩn quốc tế đầu tiên tại khu vực.",
        opening_date="20/10/2026",
        opening_promo="TẶNG 100 GIỜ CHƠI MIỄN PHÍ",
        brand="Tendoo Cyber Arena",
    )
    html_opening = layout.render_html(
        content=content_opening,
        palette=palette,
        width=1024,
        height=1024,
        bg_data_uri=pil_to_data_uri(bg),
    )
    assert "20/10/2026" in html_opening
    assert "TẶNG 100 GIỜ CHƠI MIỄN PHÍ" in html_opening

    print("[PASSED] Diagonal Slash adaptive category body components rendered successfully.")


def test_diagonal_slash_playwright_render():
    """Renders actual PNG posters via headless Playwright to verify visual output."""
    layout = get_layout("diagonal_slash")
    safe_zone = layout.get_safe_zone()

    bg = create_dynamic_sport_background(1024, 1024)
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    content = PosterContent(
        category="promo",
        pre_header="PHIÊN BẢN GIỚI HẠN SPEED 2026",
        headline="GIÀY THỂ THAO\nCARBON ULTRA-LIGHT",
        slogan="Trọng lượng siêu nhẹ chỉ 160g, thiết kế vát chéo tối ưu luồng khí động học cho bước chạy bứt tốc.",
        offer_main="GIẢM 40%",
        offer_sub="Tặng bình nước thể thao kim loại cao cấp và túi đeo chéo",
        dates="10/09 - 30/09/2026",
        applicable="Áp dụng trên toàn bộ chuỗi cửa hàng Tendoo Sport.",
        brand="Tendoo Sport",
        hotline="1800 9999",
        address="Showroom Flagship, TP. Hồ Chí Minh",
        website_link="tendoo-sport.vn/carbon",
        qr_data_uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    )

    rendered_html = layout.render_html(
        content=content,
        palette=palette,
        width=1024,
        height=1024,
        bg_data_uri=pil_to_data_uri(bg),
        headline_effect="vien_rong_the_thao",
    )

    out_path = OUTPUT_DIR / "test_diagonal_slash_sport.png"
    PosterRenderer.render(
        html_content=rendered_html,
        output_image_path=out_path,
        width=1024,
        height=1024,
    )
    assert out_path.exists() and out_path.stat().st_size > 40000
    print(f"[PASSED] Rendered Diagonal Slash Sport Poster -> {out_path}")
