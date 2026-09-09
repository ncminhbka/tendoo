"""
tests/test_style_matcher.py

Phase B: verifies the deterministic layout/style auto-match engine that replaced the
raw layout/font/corridor-material pickers in the UI (see src/tendoo/layouts/style_matcher.py
and yeu_cau.txt's "Hiển thị/thiết kế" redesign requirement).
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tendoo.layouts.registry import list_layouts
from tendoo.layouts.style_matcher import (
    CATEGORY_DEFAULT_LAYOUT,
    LAYOUT_COMPATIBLE_STYLES,
    STYLE_PREFS,
    resolve_style_preset,
)

CATEGORIES = ["promo", "product_intro", "opening", "feedback", "recruitment", "guide"]


def test_category_defaults_are_real_registered_layouts():
    registered = {l["name"] for l in list_layouts()}
    for category, layout_name in CATEGORY_DEFAULT_LAYOUT.items():
        assert layout_name in registered, f"{category}'s default layout '{layout_name}' isn't registered"


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("style_pref", STYLE_PREFS)
def test_resolve_style_preset_returns_valid_layout_and_style(category, style_pref):
    """Every (category, style_pref) combination must resolve to a real registered layout
    and a style_hint that's either 'auto' or a genuinely compatible style for that layout
    (per LAYOUT_COMPATIBLE_STYLES) -- never a style_hint the layout would reject."""
    registered = {l["name"] for l in list_layouts()}
    result = resolve_style_preset(category, style_pref, "")
    assert result["layout"] in registered
    assert result["font_key"] == "auto"

    if style_pref == "auto":
        assert result["style_hint"] == "auto"
    else:
        valid_styles = LAYOUT_COMPATIBLE_STYLES[result["layout"]]["styles"]
        assert result["style_hint"] in valid_styles


def test_unknown_style_pref_falls_back_to_auto():
    result = resolve_style_preset("promo", "not_a_real_mood", "")
    assert result["style_hint"] == "auto"
    assert result["layout"] == CATEGORY_DEFAULT_LAYOUT["promo"]


def test_unknown_category_falls_back_to_top_dome():
    result = resolve_style_preset("not_a_real_category", "auto", "")
    assert result["layout"] == "top_dome"


def test_style_pref_can_override_category_default_layout():
    """nang_dong (dynamic/sport mood) should nudge promo away from its plain top_dome
    default toward the more energetic diagonal_slash layout."""
    default = resolve_style_preset("promo", "auto", "")
    nudged = resolve_style_preset("promo", "nang_dong", "")
    assert default["layout"] == "top_dome"
    assert nudged["layout"] == "diagonal_slash"
