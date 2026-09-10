"""
src/tendoo/layouts/style_matcher.py

Smart Layout/Corridor-Style Auto-Matching Engine.
==================================================
Per yeu_cau.txt's explicit "Hiển thị/thiết kế" redesign requirement: layout, font, and
corridor "chất liệu" (material/lighting) must no longer be raw user-facing pickers --
the UI now exposes only a business-mood *suggestion* ("phong cách") plus màu chủ đạo,
tỷ lệ, số lượng. This module is the deterministic, rule-based auto-match this hint
resolves to (category + style_pref + scene keywords -> layout + style_hint + font_key).

No LLM involved here -- this is Phase B of the project's 3-phase plan (required-field
validation -> this auto-match -> LLM render-plan hosting for the freeform layout). It
also becomes the fallback path the future LLM stage uses whenever it's unavailable or
returns an invalid render plan, so keeping it deterministic and dependency-free matters.

Font is intentionally left as "auto" always here: font_engine.recommend_font() (driven
by category + style_hint + text content) is already intelligent and doesn't need a
style_pref input duplicated into it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Corridor lighting/material styles each of the 6 fixed layouts' masks are tuned for.
# (freeform has no corridor-style vocabulary of its own yet -- it's LLM/render-plan
# territory, see project memory on the freeform layout direction.) Moved here from
# demo_server.py so this module (not the FastAPI orchestration layer) owns "what style
# fits what layout" -- demo_server.py imports this dict rather than defining its own.
LAYOUT_COMPATIBLE_STYLES: Dict[str, Dict[str, object]] = {
    "top_dome": {
        "styles": ["daylight", "studio_dark", "golden_hour", "gold_bevel", "festive_moon", "ribbon"],
        "default": "daylight",
    },
    "bottom_platform": {
        "styles": ["cinematic_asphalt", "luxury_marble", "warm_wood", "nature_stone", "water_mirror", "cyberpunk_grid"],
        "default": "cinematic_asphalt",
    },
    "split_column": {
        "styles": ["champagne_silk", "silk_sash", "minimal_wall", "studio_light_pillar", "velvet_drape"],
        "default": "champagne_silk",
    },
    "center_hourglass": {
        "styles": ["moonbeam", "studio_spotlight", "festive_light"],
        "default": "moonbeam",
    },
    "diagonal_slash": {
        "styles": ["sport_speed", "cyber_neon", "carbon_mesh", "daylight_motion"],
        "default": "sport_speed",
    },
    "l_frame": {
        "styles": ["tech_minimal", "cyber_tech", "luxury_gold", "daylight_clean"],
        "default": "tech_minimal",
    },
}

# Each category's base recommended layout -- lifted verbatim from the frontend's old
# CATEGORIES[cat].recommendedLayout table (demo_ui.html), now the server-side source of
# truth since layout is no longer a user-facing choice.
CATEGORY_DEFAULT_LAYOUT: Dict[str, str] = {
    "promo": "top_dome",
    "product_intro": "top_dome",
    "opening": "center_hourglass",
    "feedback": "split_column",
    "recruitment": "split_column",
    "guide": "split_column",
}

# Every non-title field a category's form / fixed layout holds -- mirrors
# demo_ui.html's cat-fields-* blocks exactly. Phase C (LLM render-plan) uses this as
# the single source of truth for: (1) which fields to show the render-plan LLM as
# context, (2) which fields are "mandatory if filled" (see llm_render_plan_server.py's
# module docstring for the full rule), (3) which field names extra_blocks[].field is
# allowed to reference. This is the generalization of Phase A's
# CATEGORY_REQUIRED_FIELDS to every field, not just the required ones.
CATEGORY_FIELD_SLOTS: Dict[str, list] = {
    "promo": ["discount", "applied_product", "date_start", "date_end"],
    "product_intro": ["product_name", "price", "product_desc", "highlights"],
    "opening": ["opening_date", "opening_promo", "booking_contact"],
    "feedback": ["feedback_target", "feedback_quote", "feedback_rating", "special_offer"],
    "recruitment": ["job_position", "job_desc", "apply_deadline", "apply_method"],
    "guide": ["guide_steps"],  # every non-empty step is mandatory-if-filled, not just step 1
}

# Default visual role for each CATEGORY_FIELD_SLOTS field, used only when a field's
# content needs to be placed into a freeform zone (i.e. when the plan escalates to
# freeform for an unrelated reason and these mandatory fields need auto-placement too).
DEFAULT_FIELD_ROLE: Dict[str, str] = {
    "discount": "badge",
    "applied_product": "body",
    "date_start": "caption",
    "date_end": "caption",
    "product_name": "subtitle",
    "price": "badge",
    "product_desc": "body",
    "highlights": "body",
    "opening_date": "badge",
    "opening_promo": "body",
    "booking_contact": "caption",
    "feedback_target": "subtitle",
    "feedback_quote": "body",
    "feedback_rating": "badge",
    "special_offer": "body",
    "job_position": "subtitle",
    "job_desc": "body",
    "apply_deadline": "caption",
    "apply_method": "caption",
    "guide_steps": "body",
}

# The 6 "phong cách" (style preference) values the redesigned UI exposes -- a business
# mood, deliberately decoupled from the technical layout/style_hint vocabulary it
# replaces. "auto" means: don't bias anything, let the category default and the
# existing scene-keyword auto-harmonize logic (detect_scene_lighting_tone) decide alone.
STYLE_PREFS = ("auto", "hien_dai", "sang_trong", "nang_dong", "am_cung", "le_hoi")

# (style_pref, category) -> layout override, for the moods that meaningfully change the
# best-fit layout away from the category's plain default. Absent entries simply fall
# back to CATEGORY_DEFAULT_LAYOUT[category] -- most (style_pref, category) pairs do.
STYLE_PREF_LAYOUT_OVERRIDE: Dict[Tuple[str, str], str] = {
    ("nang_dong", "promo"): "diagonal_slash",
    ("nang_dong", "product_intro"): "diagonal_slash",
    ("nang_dong", "recruitment"): "diagonal_slash",
    ("sang_trong", "product_intro"): "bottom_platform",
    ("sang_trong", "promo"): "split_column",
    ("hien_dai", "product_intro"): "l_frame",
    ("hien_dai", "recruitment"): "l_frame",
    ("le_hoi", "promo"): "center_hourglass",
    ("le_hoi", "opening"): "center_hourglass",
}

# style_pref -> {layout: style_hint}, one fitting entry per layout drawn straight from
# that layout's own LAYOUT_COMPATIBLE_STYLES list (never an invalid style_hint). "auto"
# has no entries -- resolve_style_preset() returns style_hint="auto" for it, so the
# existing keyword-based detect_scene_lighting_tone() auto-harmonize logic keeps running
# exactly as it does today (this module is purely additive, not a replacement of that).
STYLE_PREF_HINT_BIAS: Dict[str, Dict[str, str]] = {
    "hien_dai": {
        "top_dome": "daylight",
        "bottom_platform": "cinematic_asphalt",
        "split_column": "minimal_wall",
        "center_hourglass": "moonbeam",
        "diagonal_slash": "daylight_motion",
        "l_frame": "daylight_clean",
    },
    "sang_trong": {
        "top_dome": "gold_bevel",
        "bottom_platform": "luxury_marble",
        "split_column": "champagne_silk",
        "center_hourglass": "moonbeam",
        "diagonal_slash": "carbon_mesh",
        "l_frame": "luxury_gold",
    },
    "nang_dong": {
        "top_dome": "daylight",
        "bottom_platform": "cyberpunk_grid",
        "split_column": "studio_light_pillar",
        "center_hourglass": "studio_spotlight",
        "diagonal_slash": "sport_speed",
        "l_frame": "cyber_tech",
    },
    "am_cung": {
        "top_dome": "golden_hour",
        "bottom_platform": "warm_wood",
        "split_column": "champagne_silk",
        "center_hourglass": "festive_light",
        "diagonal_slash": "daylight_motion",
        "l_frame": "daylight_clean",
    },
    "le_hoi": {
        "top_dome": "festive_moon",
        "bottom_platform": "warm_wood",
        "split_column": "silk_sash",
        "center_hourglass": "festive_light",
        "diagonal_slash": "daylight_motion",
        "l_frame": "luxury_gold",
    },
}


def resolve_style_preset(category: str, style_pref: str, scene_prompt: str = "") -> Dict[str, str]:
    """
    Deterministic auto-match: (category, style_pref) -> {"layout", "style_hint", "font_key"}.

    `scene_prompt` is accepted for a stable call signature (a future version could use it
    to break ties) but isn't consulted today -- the existing detect_scene_lighting_tone()
    already does scene-keyword-based style refinement downstream in run_pipeline_inference,
    and duplicating that logic here would just create two sources of truth for the same
    decision. This function only picks the *starting point* (layout + a mood-informed
    default style_hint); detect_scene_lighting_tone() still runs afterward exactly as before.

    font_key is always "auto" -- font_engine.recommend_font() is already intelligent
    (driven by category + style_hint + text content) and needs no separate input here.
    """
    category = (category or "promo").lower().strip()
    style_pref = (style_pref or "auto").lower().strip()
    if style_pref not in STYLE_PREFS:
        style_pref = "auto"

    layout_name = STYLE_PREF_LAYOUT_OVERRIDE.get(
        (style_pref, category), CATEGORY_DEFAULT_LAYOUT.get(category, "top_dome")
    )

    if style_pref == "auto":
        style_hint = "auto"
    else:
        layout_cfg = LAYOUT_COMPATIBLE_STYLES.get(layout_name, LAYOUT_COMPATIBLE_STYLES["top_dome"])
        style_hint = STYLE_PREF_HINT_BIAS.get(style_pref, {}).get(layout_name, layout_cfg["default"])

    return {"layout": layout_name, "style_hint": style_hint, "font_key": "auto"}


_HUE_NAME_BUCKETS: List[Tuple[int, str]] = [
    (15, "warm red"),
    (45, "warm amber orange"),
    (70, "golden yellow"),
    (170, "fresh green"),
    (200, "cyan teal"),
    (250, "cool blue"),
    (290, "deep purple violet"),
    (330, "vivid magenta pink"),
    (360, "warm red"),
]


def _hue_to_color_name(hue: int) -> str:
    for upper_bound, name in _HUE_NAME_BUCKETS:
        if hue <= upper_bound:
            return name
    return "warm red"


def inject_color_guidance(prompt: str, hex_color: Optional[str]) -> str:
    """
    Appends a short color-accent clause derived from a user-picked hex color to a
    scene/corridor prompt. No-op when hex_color is blank/unparseable.
    """
    from tendoo.layouts.color_engine import hex_to_hue
    hue = hex_to_hue(hex_color) if hex_color else None
    if hue is None:
        return prompt
    color_name = _hue_to_color_name(hue)
    clause = f"{color_name} accent tones"
    if clause in (prompt or ""):
        return prompt
    return f"{prompt}, {clause}" if prompt else clause

