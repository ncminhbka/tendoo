"""
tests/test_color_engine.py

Phase B: verifies hex_to_hue() and analyze_color_harmony()'s new `user_hue` override
(the "màu chủ đạo" primary-color picker's backend plumbing), and that it doesn't
compromise the WCAG contrast guarantee (is_dark/luminance still come from real pixels).
"""

import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tendoo.layouts.color_engine import analyze_color_harmony, hex_to_hue

FULL_ZONE = (0.0, 0.0, 1.0, 1.0)


def _solid_image(rgb):
    return np.full((64, 64, 3), rgb, dtype=np.uint8)


def test_hex_to_hue_parses_common_formats():
    assert hex_to_hue("#FF0000") == 0
    assert hex_to_hue("00FF00") == 120  # '#' optional
    assert hex_to_hue("#00f") == 240  # 3-digit shorthand


def test_hex_to_hue_rejects_malformed_input():
    assert hex_to_hue("") is None
    assert hex_to_hue(None) is None
    assert hex_to_hue("not-a-color") is None
    assert hex_to_hue("#ZZZZZZ") is None


def _hue_of(color_str: str) -> int:
    m = re.search(r"hsl\((\d+),", color_str)
    assert m, f"no hsl(...) hue found in {color_str!r}"
    return int(m.group(1))


def test_user_hue_overrides_palette_hue_on_dark_background():
    bg = _solid_image((20, 20, 20))  # near-black -> is_dark path
    palette_default = analyze_color_harmony(bg, FULL_ZONE, color_mode="auto")
    palette_override = analyze_color_harmony(bg, FULL_ZONE, color_mode="auto", user_hue=210)

    # Override actually changes the resulting hue used in the headline color token.
    assert _hue_of(palette_override.headline_color) == 210
    assert _hue_of(palette_override.headline_color) != _hue_of(palette_default.headline_color)


def test_user_hue_does_not_affect_dark_vs_light_contrast_decision():
    """The override must only change *hue* -- is_dark (and therefore which of the two
    WCAG-safe palettes gets used) must still come from the real pixels, unaffected."""
    dark_bg = _solid_image((15, 15, 15))
    light_bg = _solid_image((240, 240, 240))

    dark_default = analyze_color_harmony(dark_bg, FULL_ZONE, color_mode="auto")
    dark_override = analyze_color_harmony(dark_bg, FULL_ZONE, color_mode="auto", user_hue=45)
    light_default = analyze_color_harmony(light_bg, FULL_ZONE, color_mode="auto")
    light_override = analyze_color_harmony(light_bg, FULL_ZONE, color_mode="auto", user_hue=45)

    # Dark-background palette uses a luminous near-white headline (gradient or high-lightness);
    # light-background palette uses a deep near-black headline -- that split must survive the override.
    assert _hue_of(dark_override.headline_color) == 45 or dark_override.headline_is_gradient
    assert dark_default.headline_is_gradient == dark_override.headline_is_gradient
    assert light_default.headline_color != dark_default.headline_color
    assert light_override.headline_color != dark_override.headline_color


def test_no_user_hue_matches_prior_pixel_derived_behavior():
    """Regression guard: omitting user_hue (the default) must be pixel-identical to
    before this parameter existed."""
    bg = _solid_image((180, 90, 40))
    p1 = analyze_color_harmony(bg, FULL_ZONE, color_mode="auto")
    p2 = analyze_color_harmony(bg, FULL_ZONE, color_mode="auto", user_hue=None)
    assert p1.headline_color == p2.headline_color
    assert p1.badge_bg == p2.badge_bg
