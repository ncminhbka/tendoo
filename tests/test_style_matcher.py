"""
tests/test_style_matcher.py

Phase B / Pure OmniBlock: verifies the layout/style auto-match engine in
src/tendoo/layouts/style_matcher.py.
Under Pure OmniBlock Architecture:
  - Layout is unconditionally DEFAULT_LAYOUT = "omni".
  - Style preferences map cleanly via STYLE_PREF_TO_STYLE_HINT to OMNI_STYLES.
  - Zero legacy layouts or fallback matrices.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tendoo.layouts.registry import list_layouts
from tendoo.layouts.style_matcher import (
    DEFAULT_LAYOUT,
    LAYOUT_COMPATIBLE_STYLES,
    OMNI_STYLES,
    STYLE_PREFS,
    STYLE_PREF_TO_STYLE_HINT,
    resolve_style_preset,
)

CATEGORIES = ["promo", "product_intro", "opening", "feedback", "recruitment", "guide"]


def test_default_layout_is_omni():
    assert DEFAULT_LAYOUT == "omni"
    registered = {l["name"] for l in list_layouts()}
    assert "omni" in registered


def test_omni_compatible_styles():
    assert "omni" in LAYOUT_COMPATIBLE_STYLES
    assert set(LAYOUT_COMPATIBLE_STYLES["omni"]["styles"]) == set(OMNI_STYLES)
    assert len(OMNI_STYLES) == 11


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("style_pref", STYLE_PREFS)
def test_resolve_style_preset_returns_omni_and_valid_style(category, style_pref):
    """Every (category, style_pref) combination must resolve to layout='omni'
    and a style_hint that's either 'auto' or a genuinely compatible style for OmniBlock."""
    result = resolve_style_preset(category, style_pref, "")
    assert result["layout"] == "omni"
    assert result["font_key"] == "auto"

    if style_pref == "auto":
        assert result["style_hint"] == "auto"
    else:
        assert result["style_hint"] in OMNI_STYLES
        assert result["style_hint"] == STYLE_PREF_TO_STYLE_HINT[style_pref]


def test_unknown_style_pref_falls_back_to_auto():
    result = resolve_style_preset("promo", "not_a_real_mood", "")
    assert result["style_hint"] == "auto"
    assert result["layout"] == "omni"


def test_unknown_category_resolves_to_omni():
    result = resolve_style_preset("not_a_real_category", "auto", "")
    assert result["layout"] == "omni"


def test_style_pref_mapping():
    """Verify specific 1D mappings from user style_pref to rich commercial Omni styles."""
    assert resolve_style_preset("promo", "sang_trong", "")["style_hint"] == "luxury_gold"
    assert resolve_style_preset("promo", "nang_dong", "")["style_hint"] == "cyberpunk_grid"
    assert resolve_style_preset("promo", "hien_dai", "")["style_hint"] == "minimal_wall"
    assert resolve_style_preset("promo", "am_cung", "")["style_hint"] == "warm_wood"
    assert resolve_style_preset("promo", "le_hoi", "")["style_hint"] == "festive_light"
