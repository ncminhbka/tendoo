"""
Unit and Visual Integration Tests for L-Frame Corner Anchor Layout.
===================================================================
Tests:
1. Mathematical mask properties:
   - Top bar (y <= 0.25) across all x is dense (>= 0.98).
   - Left column (x <= 0.30) across all y is dense (>= 0.98).
   - Lower-right open quadrant (x >= 0.60, y >= 0.50) is strictly 0.0 for hero subjects.
   - Inner corner transition is smooth and continuous via Euclidean SDF.
2. Corridor prompts:
   - Validates tech_minimal, cyber_tech, luxury_gold, daylight_clean.
   - Strictly enforces Brand Preservation Rule (NO unbranded, no logos, no labels).
3. Template & HTML rendering:
   - Zero unrendered mustache/placeholder tags across all 4 aspect ratios.
   - Typographic hierarchy & Vietnamese headline balancing.
   - Adaptive category body injection (Product Intro, Recruitment, Guide).
4. Playwright image rendering verification.
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
from tendoo_legacy.layouts.l_frame.mask import generate_l_frame_mask
from tendoo.poster_renderer import PosterRenderer


OUTPUT_DIR = Path("output_layouts_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_tech_minimal_background(width: int = 1024, height: int = 1024) -> Image.Image:
    """Creates a sleek tech studio backdrop: dark charcoal slate with subtle electric cyan rim glow."""
    img = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            u = x / float(width)
            v = y / float(height)
            # Ambient gradient: darker at top-left, subtle glow at bottom-right for device
            r = int(10 + u * 25 + v * 15)
            g = int(14 + u * 35 + v * 25)
            b = int(24 + u * 60 + v * 40)
            img.putpixel((x, y), (min(255, r), min(255, g), min(255, b)))
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_l_frame_mask_math():
    """Validates Euclidean signed distance field and cosine feathering of L-Frame mask."""
    layout = get_layout("l_frame")
    h, w = 1024, 1024
    mask = layout.generate_mask(width=w, height=h, y_bar=0.30, x_col=0.38, delta=0.10)

    assert mask.shape == (1024, 1024), f"Unexpected shape {mask.shape}"
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "Mask values outside [0, 1]"

    # 1. Top horizontal bar: deep inside at y=0.10 across x in [0.10, 0.85] must be >= 0.98
    for x_test in [0.15, 0.45, 0.75]:
        val = mask[int(0.10 * h), int(x_test * w)]
        assert val >= 0.98, f"Top bar at x={x_test}, y=0.10 should be ~1.0 (got {val})"

    # 2. Left vertical column: deep inside at x=0.15 across y in [0.15, 0.85] must be >= 0.98
    for y_test in [0.20, 0.50, 0.80]:
        val = mask[int(y_test * h), int(0.15 * w)]
        assert val >= 0.98, f"Left column at x=0.15, y={y_test} should be ~1.0 (got {val})"

    # 3. Lower-right open quadrant: x=0.75, y=0.75 must be strictly 0.0
    val_lr = mask[int(0.75 * h), int(0.75 * w)]
    assert val_lr == 0.0, f"Lower-right hero space should be strictly 0.0 (got {val_lr})"

    # 4. Inner concave corner: at (x_col=0.38, y_bar=0.30), smooth union coverage is 1 - (1-0.5)*(1-0.5) = 0.75
    val_corner = mask[int(0.30 * h), int(0.38 * w)]
    assert 0.50 <= val_corner <= 0.85, f"Inner corner at (0.38, 0.30) expected smooth union coverage ~0.75, got {val_corner}"
    # Verify outer decay
    val_decay = mask[int((0.30 + 0.16/2 + 0.02) * h), int((0.38 + 0.16/2 + 0.02) * w)]
    assert val_decay <= 0.05, f"Outer zone should decay softly to 0.0, got {val_decay}"

    # 5. Save visual mask artifact for inspection
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_path = OUTPUT_DIR / "l_frame_mask.png"
    mask_img.save(mask_path)
    print(f"[PASSED] L-Frame mask math verified -> saved to {mask_path}")


def test_l_frame_corridor_prompts():
    """Validates that corridor prompts have copy space guidance and NO brand-suppressing keywords."""
    layout = get_layout("l_frame")
    forbidden_brand_suppressors = [
        "unbranded", "no logos", "no brand", "no labels", "no words", "no letters"
    ]

    for style in ["tech_minimal", "cyber_tech", "luxury_gold", "daylight_clean", "unknown_fallback"]:
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

    print("[PASSED] All L-Frame corridor prompts follow brand safety & copy space rules.")


def test_l_frame_html_rendering_no_orphan_placeholders():
    """Ensures rendered HTML has NO unrendered {{...}} placeholder tags across all 4 aspect ratios."""
    layout = get_layout("l_frame")
    safe_zone = layout.get_safe_zone()
    bg = Image.new("RGB", (1024, 1024), (16, 20, 32))
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    aspect_ratios = [
        (1024, 1024),  # 1:1 Square
        (576, 1024),   # 9:16 Story
        (1024, 576),   # 16:9 Banner
        (816, 1024),   # 4:5 Feed
    ]

    content = PosterContent(
        pre_header="CÔNG NGHỆ THÔNG MINH 2026",
        headline="ROBOT HÚT BỤI LAU NHÀ\nECOVACS X1 OMNI PRO",
        slogan="Lực hút 8000Pa siêu mạnh, tự động giặt sấy giẻ bằng khí nóng và đổ rác thông minh.",
        offer_main="GIẢM 5 TRIỆU ĐỒNG",
        offer_sub="Tặng kèm bộ phụ kiện tiêu hao 2 năm trị giá 2.500.000đ",
        dates="Duy nhất tháng này",
        applicable="Bảo hành chính hãng 24 tháng tận nhà trên toàn quốc.",
        brand="Tendoo Smart Home",
        hotline="1900 8989",
        address="Trung tâm Trải nghiệm Công nghệ, Cầu Giấy, Hà Nội",
        website_link="tendoo.ai/robot-omni",
        qr_data_uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    )

    for w, h in aspect_ratios:
        rendered_html = layout.render_html(
            content=content,
            palette=palette,
            width=w,
            height=h,
            bg_data_uri=pil_to_data_uri(bg),
            headline_effect="led",
        )

        orphans = re.findall(r'\{\{[a-zA-Z0-9_\#\/\s]+\}\}', rendered_html)
        assert len(orphans) == 0, f"Found orphan placeholders in {w}x{h} HTML: {orphans}"
        assert str(w) in rendered_html
        assert str(h) in rendered_html
        assert "ROBOT HÚT BỤI LAU NHÀ" in rendered_html

    print("[PASSED] L-Frame HTML rendering has zero orphan placeholders across 4 aspect ratios.")


def test_l_frame_adaptive_categories():
    """Validates adaptive category components inside L-Frame layout (Product Intro, Recruitment, Guide)."""
    layout = get_layout("l_frame")
    safe_zone = layout.get_safe_zone()
    bg = Image.new("RGB", (1024, 1024), (18, 22, 36))
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    # 1. Product Intro category
    content_intro = PosterContent(
        category="product_intro",
        headline="CAMERA AN NINH 360°\nAI VISION 4K HDR",
        slogan="Nhận diện khuôn mặt chuẩn xác, cảnh báo đột nhập tức thì qua ứng dụng di động.",
        price="1.890.000đ",
        highlights="Độ phân giải 4K, AI phát hiện người, Đàm thoại 2 chiều",
        brand="Tendoo Security",
    )
    html_intro = layout.render_html(
        content=content_intro,
        palette=palette,
        width=1024,
        height=1024,
        bg_data_uri=pil_to_data_uri(bg),
    )
    assert "1.890.000đ" in html_intro
    assert "AI phát hiện người" in html_intro

    # 2. Recruitment category
    content_recruitment = PosterContent(
        category="recruitment",
        headline="TUYỂN DỤNG NHÂN TÀI\nKỸ SƯ TRÍ TUỆ NHÂN TẠO",
        slogan="Cơ hội nghiên cứu và phát triển các mô hình DiT thế hệ mới tại Viettel Telecom.",
        job_position="SENIOR AI RESEARCH SCIENTIST",
        job_desc="Mức lương đến $4,500 + Thưởng dự án",
        apply_deadline="30/10/2026",
        apply_method="Gửi CV về: hr@tendoo.ai",
        brand="Tendoo AI Lab",
    )
    html_recruitment = layout.render_html(
        content=content_recruitment,
        palette=palette,
        width=1024,
        height=1024,
        bg_data_uri=pil_to_data_uri(bg),
    )
    assert "SENIOR AI RESEARCH SCIENTIST" in html_recruitment
    assert "30/10/2026" in html_recruitment

    # 3. Guide category with Stepper Timeline
    content_guide = PosterContent(
        category="guide",
        headline="HƯỚNG DẪN KẾT NỐI\nTHIẾT BỊ TRONG 3 BƯỚC",
        steps=[
            "Cắm nguồn và bật bluetooth trên điện thoại",
            "Mở app Tendoo và quét mã QR trên thân máy",
            "Hoàn tất cấu hình wifi và bắt đầu sử dụng"
        ],
        brand="Tendoo Support",
    )
    html_guide = layout.render_html(
        content=content_guide,
        palette=palette,
        width=1024,
        height=1024,
        bg_data_uri=pil_to_data_uri(bg),
    )
    assert "Cắm nguồn và bật bluetooth" in html_guide
    assert "Hoàn tất cấu hình wifi" in html_guide

    print("[PASSED] L-Frame adaptive category components rendered successfully.")


def test_l_frame_playwright_render():
    """Renders actual PNG poster via headless Playwright to verify visual output."""
    layout = get_layout("l_frame")
    safe_zone = layout.get_safe_zone()

    bg = create_tech_minimal_background(1024, 1024)
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    content = PosterContent(
        category="product_intro",
        pre_header="FLAGSHIP CÔNG NGHỆ 2026",
        headline="LOA THÔNG MINH AI\nSPATIAL SOUND PRO",
        slogan="Âm thanh vòm đa hướng 360 độ, điều khiển nhà thông minh bằng giọng nói tiếng Việt mượt mà.",
        price="4.590.000đ",
        highlights="Công suất 120W, Trợ lý ảo Tendoo AI, Vỏ hợp kim nhôm",
        brand="Tendoo Audio Lab",
        hotline="1800 6868",
        address="Tòa nhà Công nghệ Tendoo, Hà Nội",
        website_link="tendoo.ai/spatial-speaker",
        qr_data_uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    )

    rendered_html = layout.render_html(
        content=content,
        palette=palette,
        width=1024,
        height=1024,
        bg_data_uri=pil_to_data_uri(bg),
        headline_effect="led",
    )

    out_path = OUTPUT_DIR / "test_l_frame_tech.png"
    PosterRenderer.render(
        html_content=rendered_html,
        output_image_path=out_path,
        width=1024,
        height=1024,
    )
    assert out_path.exists() and out_path.stat().st_size > 40000
    print(f"[PASSED] Rendered L-Frame Tech Poster -> {out_path}")
