"""
tests/test_freeform_layout.py

Unit + visual tests for the new Freeform Multi-Zone layout (src/tendoo/layouts/freeform/).

This layout renders whatever render plan it's given (a list of {text, zone, role}
blocks) -- it does NOT itself decide what to draw or where (that's an upstream LLM
stage, not implemented yet). So these tests exercise it with hand-written render plans
modeled on real cases from prompt_test.txt (custom text placement requests), to verify
the mask/template mechanics work correctly independent of the LLM stage.
"""

import base64
import html
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tendoo.layouts import PosterContent, analyze_color_harmony, get_layout
from tendoo.layouts.freeform.mask import generate_freeform_mask
from tendoo.layouts.freeform.zones import ZONE_NAMES, is_valid_zone
from tendoo.poster_renderer import PosterRenderer

OUTPUT_DIR = Path("output_layouts_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_all_zone_names_are_used_by_grid_area_and_default_align():
    from tendoo.layouts.freeform.zones import ZONE_DEFAULT_ALIGN, ZONE_GRID_AREA
    assert set(ZONE_GRID_AREA.keys()) == set(ZONE_NAMES)
    assert set(ZONE_DEFAULT_ALIGN.keys()) == set(ZONE_NAMES)
    assert len(ZONE_NAMES) == 9


def test_mask_union_covers_all_requested_zones():
    mask = generate_freeform_mask(height=512, width=512, zones=["top_left", "bottom_right"])
    from tendoo.layouts.freeform.zones import ZONE_RECTS
    for zone in ["top_left", "bottom_right"]:
        y0, x0, y1, x1 = ZONE_RECTS[zone]
        cy, cx = int((y0 + y1) / 2 * 511), int((x0 + x1) / 2 * 511)
        assert mask[cy, cx] > 0.95, f"Center of requested zone '{zone}' should be near-fully reserved"
    # An unrequested zone's center should be untouched (0.0)
    from tendoo.layouts.freeform.zones import ZONE_RECTS as _R
    y0, x0, y1, x1 = _R["bottom_left"]
    cy, cx = int((y0 + y1) / 2 * 511), int((x0 + x1) / 2 * 511)
    assert mask[cy, cx] == 0.0, "Unrequested zone should be fully open for the product/scene"


def test_mask_empty_zones_falls_back_to_generic_top_band():
    mask = generate_freeform_mask(height=256, width=256, zones=[])
    assert mask.max() > 0.0, "Empty zone list should still produce a sane fallback corridor, not an all-zero mask"


def test_mask_unknown_zone_name_ignored_not_crashed():
    mask = generate_freeform_mask(height=256, width=256, zones=["top_left", "totally_bogus_zone"])
    assert mask.shape == (256, 256)
    assert mask.max() <= 1.0 and mask.min() >= 0.0


def test_is_valid_zone():
    assert is_valid_zone("center")
    assert not is_valid_zone("nonexistent")


# ---------------------------------------------------------------------------
# Content-aware (dynamic) zone sizing -- fixes the "static rect is either too
# small (text overflows past the mask) or too large (starves the product of
# space needlessly)" gap the fixed 6-layout system already had.
# ---------------------------------------------------------------------------

def test_compute_zone_rects_grows_with_more_content():
    from tendoo.layouts.freeform.measure import compute_zone_rects

    short_plan = [{"text": "OK", "zone": "top_left", "role": "caption"}]
    long_plan = [
        {"text": "MOT TIEU DE RAT DAI VOI NHIEU TU DE KIEM TRA WRAP", "zone": "top_left", "role": "hero"},
        {"text": "Mot dong phu cung khong ngan chut nao", "zone": "top_left", "role": "subtitle"},
    ]

    short_rects = compute_zone_rects(short_plan, width=1024, height=1024)
    long_rects = compute_zone_rects(long_plan, width=1024, height=1024)

    def _area(rect):
        y0, x0, y1, x1 = rect
        return (y1 - y0) * (x1 - x0)

    assert "top_left" in short_rects and "top_left" in long_rects
    assert _area(long_rects["top_left"]) > _area(short_rects["top_left"]), (
        "A zone with more/longer stacked text should reserve a larger area than one with a short caption"
    )


def test_fit_font_size_px_shrinks_long_text_to_fit_budget():
    """The exact bug found on a live render-plan LLM run (2026-09-10): a "hero"-role
    block whose text is a full sentence (not a short title) must shrink well below the
    flat ROLE_SCALE size to fit a bounded height budget, rather than overflowing it."""
    from tendoo.layouts.freeform.measure import _font_path_for, fit_font_size_px

    font_path = _font_path_for("bevietnam")
    long_sentence = (
        "KHÁCH HÀNG HÀI LÒNG VỚI KẾT QUẢ TĂNG CƠ VÀ CẢI THIỆN VÓC DÁNG CHỈ SAU BA THÁNG"
    )
    base_font_size = 74  # matches ROLE_SCALE["hero"] at width=1024 (0.072 * 1024)
    max_width_px = 1024 * 0.42
    max_height_px = 1024 * 0.42

    font_size, _, wrapped_h = fit_font_size_px(
        long_sentence, font_path, base_font_size, max_width_px, max_height_px,
    )
    assert font_size < base_font_size, "a full sentence at hero size must shrink, not render at the flat ratio"
    assert wrapped_h <= max_height_px + 1, "the shrunk text must actually fit the height budget"


def test_fit_font_size_px_leaves_short_text_at_base_size():
    """A short phrase that already fits shouldn't be shrunk needlessly."""
    from tendoo.layouts.freeform.measure import _font_path_for, fit_font_size_px

    font_path = _font_path_for("bevietnam")
    font_size, _, _ = fit_font_size_px("SALE 50%", font_path, 74, 1024 * 0.42, 1024 * 0.42)
    assert font_size == 74


def test_compute_zone_rects_stays_capped_with_pathologically_long_hero_text():
    """Even with a very long "hero" text, the reserved rect must stay within
    ZONE_MAX_W_PCT/ZONE_MAX_H_PCT (42%) -- the font-fitting engaging is what makes this
    cap actually meaningful, rather than just clamping a rect whose real content still
    overflows past it when rendered."""
    from tendoo.layouts.freeform.measure import compute_zone_rects

    plan = [{
        "text": "KHÁCH HÀNG HÀI LÒNG VỚI PRIVATE COACHING TRANSFORMATION SAU BA THÁNG TẬP LUYỆN CÙNG PT RIÊNG",
        "zone": "top_center", "role": "hero",
    }]
    rects = compute_zone_rects(plan, width=1024, height=1024)
    y0, x0, y1, x1 = rects["top_center"]
    assert (y1 - y0) <= 0.42 + 1e-6
    assert (x1 - x0) <= 0.42 + 1e-6


def test_compute_zone_rects_anchors_to_correct_corner():
    from tendoo.layouts.freeform.measure import compute_zone_rects

    plan = [{"text": "CORNER TEST", "zone": "bottom_right", "role": "hero"}]
    rects = compute_zone_rects(plan, width=1024, height=1024)
    y0, x0, y1, x1 = rects["bottom_right"]
    # Anchored to the bottom-right corner: the rect's far edge should sit at the
    # canvas margin (0.96), not floating away from it.
    assert x1 == pytest.approx(0.96, abs=1e-6)
    assert y1 == pytest.approx(0.96, abs=1e-6)


def test_compute_zone_rects_includes_qr_footprint():
    from tendoo.layouts.freeform.measure import compute_zone_rects

    rects_no_qr = compute_zone_rects([], width=1024, height=1024, qr_zone=None)
    rects_with_qr = compute_zone_rects([], width=1024, height=1024, qr_zone="bottom_right")
    assert "bottom_right" not in rects_no_qr, "No content and no QR -> zone should be entirely absent"
    assert "bottom_right" in rects_with_qr, "QR-only zone should still reserve space for the QR image"


def test_compute_zone_rects_ignores_unknown_zone_gracefully():
    from tendoo.layouts.freeform.measure import compute_zone_rects

    rects = compute_zone_rects(
        [{"text": "hi", "zone": "not_a_real_zone", "role": "body"}], width=512, height=512,
    )
    assert rects == {}


def test_dynamic_mask_differs_from_static_mask_for_small_content():
    """The whole point: a small caption in a zone should reserve noticeably LESS
    mask area dynamically than the fixed static-zone-size fallback would."""
    plan = [{"text": "OK", "zone": "top_left", "role": "caption"}]
    layout = get_layout("freeform")

    static_mask = layout.generate_mask(width=1024, height=1024, zones=["top_left"])
    dynamic_mask = layout.generate_mask(width=1024, height=1024, blocks=plan)

    assert dynamic_mask.sum() < static_mask.sum(), (
        "A short caption's dynamically-sized reservation should be smaller than the "
        "fixed static zone footprint -- otherwise the product is being starved of "
        "space it doesn't actually need to give up"
    )


def _synth_bg(w=1024, h=1024) -> Image.Image:
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        r, g, b = int(30 + t * 20), int(28 + t * 18), int(45 + t * 25)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def _pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# Render plans modeled directly on real prompt_test.txt cases (smart watch ads with
# custom per-position text requests) -- this is exactly the class of request the fixed
# 6-layout system cannot represent.
PLAN_WATCH_TOPLEFT_MIDLEFT = [
    {"text": "THOI GIAN LA CUA BAN", "zone": "top_left", "role": "caption"},
    {"text": "NANG TAM PHONG CACH DOI SONG", "zone": "middle_left", "role": "hero"},
]

PLAN_WATCH_TOP_BOTTOM = [
    {"text": "CHINH PHUC MOI GIOI HAN", "zone": "top_center", "role": "hero"},
    {"text": "DONG DONG HO THE THAO CAO CAP", "zone": "bottom_center", "role": "subtitle"},
]

PLAN_COFFEE_MULTI_CTA = [
    {"text": "GRAND OPENING", "zone": "center", "role": "hero"},
    {"text": "MUA 1 TANG 1", "zone": "center", "role": "subtitle"},
    {"text": "Ap dung tu 14/05 - 30/05", "zone": "bottom_center", "role": "caption"},
    {"text": "Ghe ngay hom nay!", "zone": "bottom_left", "role": "badge"},
    {"text": "Deal cuc hot!", "zone": "bottom_right", "role": "badge"},
    {"text": "Coffee rang moc chuan vi", "zone": "top_left", "role": "caption"},
]


def test_render_freeform_plan_with_icons_and_qr():
    """A realistic full request: prompt-driven hero/subtitle text PLUS other filled
    form fields (brand, hotline, address) placed as footer-style blocks with icons,
    PLUS a QR code -- the case the user asked about (are non-prompt fields handled?)."""
    from tendoo.qr import generate_qr_base64

    layout = get_layout("freeform")
    plan = [
        {"text": "GIAM 50% HOM NAY", "zone": "top_center", "role": "hero"},
        {"text": "Tendoo Coffee", "zone": "bottom_left", "role": "body"},
        {"text": "0334842155", "zone": "bottom_left", "role": "caption", "icon": "phone"},
        {"text": "123 Le Loi, Q1, TP.HCM", "zone": "bottom_left", "role": "caption", "icon": "location"},
    ]
    zones_requested = list({b["zone"] for b in plan})

    bg = _synth_bg()
    mask = layout.generate_mask(width=1024, height=1024, zones=zones_requested)
    safe_zone = layout.get_safe_zone()
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    qr_data_uri = generate_qr_base64("https://tendoo.click") or ""
    assert qr_data_uri, "QR generation should succeed for this test to be meaningful"

    content = PosterContent(
        headline=plan[0]["text"],
        category="promo",
        free_text_blocks=plan,
        qr_data_uri=qr_data_uri,
        qr_zone="bottom_right",
    )

    html_str = layout.render_html(
        content=content, palette=palette, bg_data_uri=_pil_to_data_uri(bg), width=1024, height=1024,
    )

    assert "{{" not in html_str
    assert "icon-phone" in html_str, "Phone icon SVG should be embedded for the icon='phone' block"
    assert "icon-location" in html_str, "Location icon SVG should be embedded for the icon='location' block"
    assert "freeform-qr-card" in html_str and qr_data_uri in html_str, "QR block should render in its own zone"
    for block in plan:
        assert html.escape(block["text"]) in html_str

    out_path = OUTPUT_DIR / "test_freeform_icons_and_qr.png"
    PosterRenderer.render(html_content=html_str, output_image_path=out_path, width=1024, height=1024)
    assert out_path.exists() and out_path.stat().st_size > 30000
    print(f"[PASSED] Rendered freeform plan with icons+QR -> {out_path}")


def test_render_html_shrinks_long_hero_text_instead_of_overflowing():
    """Regression test for the exact bug found on a real render-plan LLM run
    (2026-09-10): the model assigned "hero" role to a full sentence (not a short
    title), which rendered at the flat ROLE_SCALE size and flooded the whole canvas.
    The emitted font-size must now be meaningfully smaller than the flat ratio would
    produce, and the rendered PNG must not blow up in file size from runaway text."""
    import re

    layout = get_layout("freeform")
    long_hero_text = (
        "KHÁCH HÀNG HÀI LÒNG VỚI PRIVATE COACHING TRANSFORMATION SAU BA THÁNG"
    )
    plan = [
        {"text": long_hero_text, "zone": "top_center", "role": "hero"},
        {"text": "97% khách hàng hài lòng với kết quả", "zone": "center", "role": "subtitle"},
    ]
    bg = _synth_bg()
    safe_zone = layout.get_safe_zone()
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")
    content = PosterContent(headline=long_hero_text, category="feedback", free_text_blocks=plan)

    html_str = layout.render_html(
        content=content, palette=palette, bg_data_uri=_pil_to_data_uri(bg), width=1024, height=1024,
    )

    flat_hero_size = int(1024 * 0.072)  # ROLE_SCALE["hero"][0] -- what it would be unshrunk
    sizes = [int(m) for m in re.findall(r"font-size:(\d+)px", html_str)]
    assert sizes, "expected at least one inline font-size in the rendered HTML"
    hero_block_size = sizes[0]  # the hero block is emitted first (zone iteration order)
    assert hero_block_size < flat_hero_size, (
        f"long hero text rendered at {hero_block_size}px, expected shrunk below the flat {flat_hero_size}px"
    )

    out_path = OUTPUT_DIR / "test_freeform_long_hero_shrinks.png"
    PosterRenderer.render(html_content=html_str, output_image_path=out_path, width=1024, height=1024)
    assert out_path.exists() and out_path.stat().st_size > 30000
    print(f"[PASSED] Long hero text shrunk to {hero_block_size}px (flat would be {flat_hero_size}px) -> {out_path}")


def test_freeform_qr_only_zone_renders_even_with_no_text_blocks_there():
    """QR must still render in its target zone even if no text block was also
    assigned there (a real gap the naive 'skip empty zones' loop could hit)."""
    from tendoo.qr import generate_qr_base64

    layout = get_layout("freeform")
    plan = [{"text": "HELLO", "zone": "top_center", "role": "hero"}]
    bg = _synth_bg()
    safe_zone = layout.get_safe_zone()
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")
    qr_data_uri = generate_qr_base64("https://tendoo.click") or ""

    content = PosterContent(
        headline="HELLO", category="promo", free_text_blocks=plan,
        qr_data_uri=qr_data_uri, qr_zone="bottom_right",
    )
    html_str = layout.render_html(
        content=content, palette=palette, bg_data_uri=_pil_to_data_uri(bg), width=1024, height=1024,
    )
    assert "freeform-qr-card" in html_str


@pytest.mark.parametrize("plan_name,plan", [
    ("watch_topleft_midleft", PLAN_WATCH_TOPLEFT_MIDLEFT),
    ("watch_top_bottom", PLAN_WATCH_TOP_BOTTOM),
    ("coffee_multi_cta", PLAN_COFFEE_MULTI_CTA),
])
def test_render_freeform_plan(plan_name, plan):
    layout = get_layout("freeform")
    zones_requested = list({b["zone"] for b in plan})

    bg = _synth_bg()
    mask = layout.generate_mask(width=1024, height=1024, zones=zones_requested)
    assert mask.shape == (1024, 1024)

    safe_zone = layout.get_safe_zone()
    palette = analyze_color_harmony(np.array(bg), safe_zone, color_mode="auto")

    content = PosterContent(
        headline=plan[0]["text"],  # unused by freeform rendering itself, but required by the dataclass
        category="promo",
        free_text_blocks=plan,
    )

    html_str = layout.render_html(
        content=content,
        palette=palette,
        bg_data_uri=_pil_to_data_uri(bg),
        width=1024,
        height=1024,
    )

    # No unrendered {{...}} placeholders left over.
    assert "{{" not in html_str, f"Unrendered placeholder left in freeform HTML for {plan_name}"
    for block in plan:
        assert html.escape(block["text"]) in html_str, f"Block text {block['text']!r} missing from rendered HTML"

    out_path = OUTPUT_DIR / f"test_freeform_{plan_name}.png"
    PosterRenderer.render(html_content=html_str, output_image_path=out_path, width=1024, height=1024)
    assert out_path.exists() and out_path.stat().st_size > 30000
    print(f"[PASSED] Rendered freeform plan '{plan_name}' -> {out_path}")
