"""
tests/test_omni_end_to_end.py

End-to-End Integration Tests for Tendoo Omni-Block Engine:
1. Verifies Store Info is 100% preserved in HTML when passing free_text_blocks (Bug #2 fix).
2. Verifies Store Info spatial override to top_bar.
3. Verifies Product Sanctuary: center is strictly excluded from GRID_ZONE_NAMES (Bug #6 fix).
4. Verifies feedback_rating renders as a badge with style-rating-badge (Bug #5 fix).
5. Verifies Category Visual Style Variants (pill-badge, luxury-tag, calendar-box, cta-button) (Bug #4).
6. Verifies style_matcher routes 100% of categories to 'omni' by default (Bug #1).
7. Verifies demo_server pipeline inference preserves store info and routes to omni in mock mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tendoo.engine.blocks import AdaptiveBlock, VALID_ZONES, map_category_to_default_blocks
from tendoo.engine.geometry import GRID_ZONE_NAMES, ZONE_NAMES
from tendoo.engine.layout import OmniBlockLayout
from tendoo.layouts.base import ColorPalette, PosterContent
from tendoo.layouts.style_matcher import DEFAULT_LAYOUT, resolve_style_preset


def test_zone_names_consistency_no_center():
    """Bug #6 verification: ZONE_NAMES must match VALID_ZONES and exclude center."""
    assert "center" not in ZONE_NAMES, "center must not be in ZONE_NAMES (sanctuary protection)"
    assert "center" not in GRID_ZONE_NAMES, "center must not be in GRID_ZONE_NAMES"
    assert "top_bar" in ZONE_NAMES
    assert "bottom_bar" in ZONE_NAMES
    assert len(GRID_ZONE_NAMES) == 8, "Grid must have exactly 8 perimeter zones"
    assert len(ZONE_NAMES) == 10, "Total valid zones must be 10"


def test_store_info_preserved_with_free_text_blocks():
    """Bug #2 verification: OmniBlockLayout must retain store info even when free_text_blocks is provided."""
    layout = OmniBlockLayout()
    # Simulate demo_server output: plan_blocks passed via free_text_blocks
    custom_plan_blocks = [
        {"text": "FLASH SALE 50%", "zone": "top_center", "role": "hero"},
        {"text": "MUA 2 TẶNG 1", "zone": "middle_right", "role": "badge"},
    ]
    content = PosterContent(
        headline="FLASH SALE 50%",
        offer_main="MUA 2 TẶNG 1",
        brand="Tendoo Coffee Store",
        hotline="0909 123 456",
        address="123 Lê Lợi, Quận 1",
        category="promo",
        free_text_blocks=custom_plan_blocks,
    )
    palette = ColorPalette(is_dark=True, luminance=0.15, hue=30, comp_hue=210, headline_color="#FFFFFF")
    html_output = layout.render_html(content, palette, bg_data_uri="", width=1024, height=1024)

    # Must contain omni-bottom-bar container
    assert "omni-bottom-bar" in html_output, "HTML must contain .omni-bottom-bar container"
    assert "Tendoo Coffee Store" in html_output, "Store brand must be rendered in HTML"
    assert "0909 123 456" in html_output, "Hotline must be rendered in HTML"
    assert "123 Lê Lợi" in html_output, "Address must be rendered in HTML"


def test_store_info_top_bar_override():
    """Verifies that store info properly moves to top_bar when requested."""
    layout = OmniBlockLayout()
    top_plan_blocks = [
        {"text": "TIÊU ĐỀ NỔI BẬT", "zone": "top_center", "role": "hero"},
        {
            "text": "Tendoo Luxury  •  Hotline: 1900 8888",
            "zone": "top_bar",
            "role": "brand_bar",
            "field": "store_info",
        },
    ]
    content = PosterContent(
        headline="TIÊU ĐỀ NỔI BẬT",
        brand="Tendoo Luxury",
        hotline="1900 8888",
        category="product_intro",
        free_text_blocks=top_plan_blocks,
    )
    palette = ColorPalette(is_dark=True, luminance=0.15, hue=45, comp_hue=225, headline_color="#FFFFFF")
    html_output = layout.render_html(content, palette, bg_data_uri="", width=1024, height=1024)

    assert "omni-top-bar" in html_output, "HTML must contain .omni-top-bar container when store info is at top"
    assert "Tendoo Luxury" in html_output
    assert "1900 8888" in html_output


def test_rating_badge_visual_style():
    """Bug #5 verification: feedback_rating must render as badge with style-rating-badge."""
    layout = OmniBlockLayout()
    fields = {
        "feedback_target": "Trẻ Hóa Làn Da",
        "feedback_quote": "Rất hài lòng về dịch vụ",
        "feedback_rating": "5.0",
        "special_offer": "VOUCHER 20%",
    }
    blocks = map_category_to_default_blocks("feedback", title="FEEDBACK KHÁCH HÀNG", fields=fields)
    rating_block = next((b for b in blocks if b.field == "feedback_rating"), None)
    assert rating_block is not None
    assert rating_block.role == "badge", f"feedback_rating role must be 'badge', got {rating_block.role}"
    assert rating_block.style_variant == "rating_badge"

    content = PosterContent(
        headline="FEEDBACK KHÁCH HÀNG",
        category="feedback",
        feedback_target="Trẻ Hóa Làn Da",
        feedback_quote="Rất hài lòng về dịch vụ",
        feedback_rating="5.0",
    )
    palette = ColorPalette(is_dark=True, luminance=0.15, hue=150, comp_hue=330, headline_color="#FFFFFF")
    html_output = layout.render_html(content, palette, bg_data_uri="", width=1024, height=1024)
    assert "style-rating-badge" in html_output


def test_category_visual_variants_rendering():
    """Bug #4 verification: category-specific visual styles must be reflected in HTML."""
    layout = OmniBlockLayout()
    palette = ColorPalette(is_dark=True, luminance=0.15, hue=45, comp_hue=225, headline_color="#FFFFFF")

    # 1. Promo: pill_badge for discount
    promo_blocks = map_category_to_default_blocks("promo", title="SALE", fields={"discount": "GIẢM 50%"})
    d_block = next(b for b in promo_blocks if b.field == "discount")
    assert d_block.style_variant == "pill_badge"

    # 2. Product intro: luxury_tag for price
    prod_blocks = map_category_to_default_blocks("product_intro", title="WATCH", fields={"price": "5.990.000đ"})
    p_block = next(b for b in prod_blocks if b.field == "price")
    assert p_block.style_variant == "luxury_tag"

    # 3. Opening: calendar_box for opening_date
    open_blocks = map_category_to_default_blocks("opening", title="OPEN", fields={"opening_date": "15/10/2026"})
    o_block = next(b for b in open_blocks if b.field == "opening_date")
    assert o_block.style_variant == "calendar_box"

    # 4. Render HTML and verify CSS classes
    content = PosterContent(headline="SALE", offer_main="GIẢM 50%", category="promo")
    html_out = layout.render_html(content, palette, bg_data_uri="", width=1024, height=1024)
    assert "style-pill-badge" in html_out


def test_style_matcher_routes_to_omni():
    """Bug #1 verification: DEFAULT_LAYOUT is unconditionally 'omni' for 100% of categories."""
    assert DEFAULT_LAYOUT == "omni"
    categories = ["promo", "product_intro", "opening", "feedback", "recruitment", "guide"]
    for cat in categories:
        preset = resolve_style_preset(cat, "auto", "")
        assert preset["layout"] == "omni", f"resolve_style_preset for {cat} must resolve layout to 'omni'"


def test_demo_server_pipeline_mock_run():
    """End-to-End test running demo_server.run_pipeline_inference in MOCK mode."""
    from tendoo.demo_server import GenerateRequest, OUTPUT_DIR, run_pipeline_inference

    req = GenerateRequest(
        category="promo",
        title="ĐẠI TIỆC MÙA HÈ",
        discount="GIẢM 30%",
        applied_product="Toàn bộ đồ uống",
        store_name="Tendoo Coffee Lounge",
        phone="0912 345 678",
        address="456 Hai Bà Trưng, Hà Nội",
        image_description="Ly cà phê kem béo mát lạnh trên bàn gỗ sang trọng",
        style_pref="auto",
        aspect_ratio="1:1",
        num_images=1,
    )

    result = run_pipeline_inference(req)
    assert result["success"] is True
    assert result["resolved_layout"] == "omni", f"Resolved layout must be 'omni', got {result['resolved_layout']}"

    # Verify generated HTML file on physical disk
    case_name = Path(result["posters"][0]["html_url"]).parts[1]
    html_path = OUTPUT_DIR / case_name / "03_poster_0.html"
    assert html_path.exists(), f"Generated HTML file must exist at {html_path}"
    html_content = html_path.read_text(encoding="utf-8")

    # Verify Store Info is present in the final HTML
    assert "omni-bottom-bar" in html_content, "Generated HTML must contain .omni-bottom-bar"
    assert "Tendoo Coffee Lounge" in html_content, "Generated HTML must contain store name"
    assert "0912 345 678" in html_content, "Generated HTML must contain phone number"
    assert "456 Hai Bà Trưng" in html_content, "Generated HTML must contain address"
    assert "ĐẠI TIỆC MÙA HÈ" in html_content, "Generated HTML must contain headline"
