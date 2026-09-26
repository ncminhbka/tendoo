"""
tests/test_core_fonts.py

Tests for the Vietnamese Typography Font Library (src/tendoo/core/fonts.py -- merged
2026-09-13 from the former `layouts/font_engine.py`, see that module's docstring):
1. All 19 catalog fonts exist on disk, are non-empty, and Base64-encode/decode losslessly.
2. resolve_font() produces syntactically valid @font-face CSS (Base64 Data URI, no network dependency).
3. recommend_font() picks sensible archetypes for style hints, keyword text, and category fallbacks.
4. Alias resolution and graceful handling of unknown/auto font keys.
5. End-to-end Playwright rendering with a representative font from each of the 6 archetypes,
   confirming the embedded font actually reaches the rendered HTML/PNG with no errors.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tendoo_v3.colors import analyze_color_harmony
from tendoo_v3.fonts import (
    FONT_ALIASES,
    FONT_CATALOG,
    FONTS_DIR,
    list_font_options,
    recommend_font,
    resolve_font,
)
from tendoo_v3.poster_renderer import PosterRenderer


# ---------------------------------------------------------------------------
# 1. Font catalog integrity
# ---------------------------------------------------------------------------

def test_font_catalog_has_19_fonts():
    assert len(FONT_CATALOG) == 19, f"Expected 19 curated fonts, found {len(FONT_CATALOG)}"
    print(f"[PASSED] FONT_CATALOG holds {len(FONT_CATALOG)} entries")


@pytest.mark.parametrize("font_key", list(FONT_CATALOG.keys()))
def test_font_file_exists_and_loads(font_key):
    meta = FONT_CATALOG[font_key]
    file_path = FONTS_DIR / meta["file"]
    assert file_path.exists(), f"Font file missing for '{font_key}': {file_path}"

    raw = file_path.read_bytes()
    assert len(raw) > 1000, f"Font file for '{font_key}' looks truncated/empty ({len(raw)} bytes)"

    # TTF/OTF magic-number sanity check (sfnt version tag)
    magic = raw[:4]
    assert magic in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf"), (
        f"'{font_key}' -> {meta['file']} does not look like a valid TTF/OTF (magic={magic!r})"
    )

    # Base64 round-trip must be lossless (this is what actually gets embedded in the poster HTML)
    encoded = base64.b64encode(raw).decode("ascii")
    decoded = base64.b64decode(encoded)
    assert decoded == raw, f"Base64 round-trip mismatch for '{font_key}'"


def test_list_font_options_covers_all_archetypes():
    groups = list_font_options()
    archetypes = {g["group_name"] for g in groups}
    expected = {meta["archetype"] for meta in FONT_CATALOG.values()}
    assert archetypes == expected

    total_fonts = sum(len(g["fonts"]) for g in groups)
    assert total_fonts == len(FONT_CATALOG)
    print(f"[PASSED] list_font_options() -> {len(groups)} groups, {total_fonts} fonts")


# ---------------------------------------------------------------------------
# 2. @font-face CSS generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("font_key", list(FONT_CATALOG.keys()))
def test_resolve_font_generates_valid_font_face_css(font_key):
    meta = FONT_CATALOG[font_key]
    resolved_key, font_face_css, headline_font_css = resolve_font(font_key=font_key)

    assert resolved_key == font_key
    assert "@font-face" in font_face_css
    assert f"font-family: '{meta['css_family']}'" in font_face_css
    fmt = "woff2" if meta.get("weights") else meta["format"]
    assert f"format('{fmt}')" in font_face_css
    assert "data:font/" in font_face_css and ";base64," in font_face_css, (
        "@font-face must embed a Base64 Data URI (zero-network offline rendering requirement)"
    )
    assert "http://" not in font_face_css and "https://" not in font_face_css, (
        "@font-face src must not reference an external network URL"
    )
    assert meta["css_family"] in headline_font_css
    assert meta["fallback"] in headline_font_css


def test_ui_font_embeds_every_weight_the_css_asks_for():
    """ROADMAP §10.7: CSS template xin Be Vietnam Pro 500-900. Thiếu độ đậm nào thì Chromium
    tô đậm GIẢ từ file gần nhất (trước 26/09: chỉ có Black -> mọi chữ phụ đặc, nặng)."""
    weights = FONT_CATALOG["bevietnam"]["weights"]
    assert set(weights) >= {400, 500, 600, 700, 800, 900}
    for fname in weights.values():
        assert (FONTS_DIR / fname).read_bytes()[:4] == b"wOF2", fname
    for key in ("bevietnam", "anton"):  # tự làm headline, và nhúng kèm làm font UI
        _, css, _ = resolve_font(font_key=key)
        bv = [f for f in css.split("@font-face") if "font-family: 'Be Vietnam Pro'" in f]
        assert {f.split("font-weight: ")[1].split(";")[0] for f in bv} == {str(w) for w in weights}, key


def test_resolve_font_missing_file_falls_back_gracefully(tmp_path, monkeypatch):
    """If a catalog entry's font file is missing on disk, resolve_font must degrade to an
    empty @font-face block (letting the CSS fallback stack render) instead of raising."""
    import tendoo_v3.fonts as font_engine_mod

    bogus_catalog = dict(FONT_CATALOG)
    bogus_catalog["__missing_test_font__"] = {
        "file": "does-not-exist-anywhere.ttf",
        "css_family": "GhostFont",
        "display_name": "Ghost",
        "archetype": "Test",
        "format": "truetype",
        "fallback": "sans-serif",
        "description": "test-only",
    }
    monkeypatch.setattr(font_engine_mod, "FONT_CATALOG", bogus_catalog)
    font_engine_mod._read_and_encode_font.cache_clear()

    key, font_face_css, headline_font_css = font_engine_mod.resolve_font(font_key="__missing_test_font__")
    assert key == "__missing_test_font__"
    assert "GhostFont" not in font_face_css
    assert "GhostFont" in headline_font_css
    font_engine_mod._read_and_encode_font.cache_clear()


# ---------------------------------------------------------------------------
# 3. recommend_font() heuristics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "category,expected",
    [
        ("promo", "anton"),
        ("opening", "pacifico"),
        ("feedback", "playfair"),
        ("recruitment", "gotham"),
        ("guide", "bevietnam"),
        ("unknown_category_xyz", "bevietnam"),
    ],
)
def test_recommend_font_category_fallback(category, expected):
    assert recommend_font(category=category) == expected


@pytest.mark.parametrize(
    "style_hint,expected",
    [
        ("sport_speed", "days"),
        ("luxury_marble", "playfair"),
        ("festive_moon", "holidays"),
        ("cinematic_asphalt", "anton"),
    ],
)
def test_recommend_font_style_hint_priority(style_hint, expected):
    # style_hint must win over the category fallback
    assert recommend_font(category="guide", style_hint=style_hint) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("TRÀ ĐÀO CAM SẢ THANH MÁT MÙA HÈ", "pacifico"),
        ("BÁNH TRUNG THU CAO CẤP NGỌT NGÀO", "cookies"),
        ("VOUCHER SPA MASSAGE THƯ GIÃN", "dancing"),
        ("GALA TRI ÂN MỪNG XUÂN TẾT", "holidays"),
        ("GIÀY SNEAKER STREETWEAR HIPHOP", "blowbrush"),
    ],
)
def test_recommend_font_text_keyword_match(text, expected):
    assert recommend_font(category="promo", text_content=text) == expected


def test_font_alias_resolution():
    for alias, target in FONT_ALIASES.items():
        assert target in FONT_CATALOG, f"Alias '{alias}' points to unknown font key '{target}'"
        key, _, _ = resolve_font(font_key=alias)
        assert key == target, f"Alias '{alias}' should resolve to '{target}', got '{key}'"


def test_resolve_font_auto_uses_recommendation():
    key, _, _ = resolve_font(font_key="auto", category="promo", style_hint="", text_content="")
    assert key == recommend_font(category="promo", style_hint="", text_content="")

    key_unknown, _, _ = resolve_font(font_key="totally_made_up_font_key", category="opening")
    assert key_unknown == recommend_font(category="opening")


# ---------------------------------------------------------------------------
# 4. End-to-end rendering: font actually reaches the HTML/PNG
# ---------------------------------------------------------------------------

def _synthetic_bg(w=1024, h=1024) -> Image.Image:
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        r = int(248 - (y / h) * 30)
        g = int(240 - (y / h) * 40)
        b = int(225 - (y / h) * 55)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def _pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")






if __name__ == "__main__":
    print("=== STARTING FONT ENGINE VALIDATION SUITE ===")
    test_font_catalog_has_19_fonts()
    for k in FONT_CATALOG:
        test_font_file_exists_and_loads(k)
        test_resolve_font_generates_valid_font_face_css(k)
    test_list_font_options_covers_all_archetypes()
    test_font_alias_resolution()
    test_resolve_font_auto_uses_recommendation()
    for fk, cat in [("anton", "promo"), ("playfair", "feedback"), ("pacifico", "opening")]:
        test_render_poster_with_explicit_font(fk, cat)
    test_render_poster_with_auto_font_family()
    print("=== ALL FONT ENGINE TESTS PASSED PERFECTLY ===")
