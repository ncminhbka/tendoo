"""
tests/test_core_category_schema.py

Covers src/tendoo/core/category_schema.py -- the single merged category/field schema
(2026-09-13, gộp `layouts/style_matcher.py`'s CATEGORY_FIELD_SLOTS/DEFAULT_FIELD_ROLE với
`engine/blocks.py`'s FIELD_BLOCK_SPECS cũ, xem module docstring của category_schema.py) --
and its consumption by engine/blocks.py::map_category_to_default_blocks().

(Moved 2026-09-13 from tests/test_omni_engine.py::test_category_mappings_all_six, which
tested this exact concern before the 2 schemas were merged.)
"""

from __future__ import annotations

from tendoo.core.category_schema import (
    CATEGORY_FIELD_SLOTS,
    CATEGORY_SCHEMA,
    DEFAULT_FIELD_ROLE,
    get_field_block_spec,
)
from tendoo.engine.blocks import map_category_to_default_blocks

CATEGORIES = ["promo", "product_intro", "opening", "feedback", "recruitment", "guide"]


def test_category_mappings_all_six():
    fields = {
        'discount': 'GIAM 50%',
        'product_name': 'Tai Nghe Sonic Pro',
        'opening_date': '14/05/2026',
        'feedback_target': 'Dich vu PT 1:1',
        'job_position': 'Senior AI Engineer',
        'guide_steps': 'Buoc 1: Cai dat | Buoc 2: Khoi chay',
    }
    for cat in CATEGORIES:
        blocks = map_category_to_default_blocks(
            category=cat,
            title='TIEU DE CHINH',
            fields=fields,
            store_info={'brand': 'Tendoo Studio', 'phone': '0123456789'},
        )
        assert len(blocks) >= 2
        # A headline ('xlarge') block is the current equivalent of the old role='hero'
        # tag -- AdaptiveBlock has no 'role' field, headline status is expressed via
        # size + field="title" instead (see engine/blocks.py::map_category_to_default_blocks).
        sizes = [b.size for b in blocks]
        assert 'xlarge' in sizes, f"{cat}: expected a headline-sized (xlarge) block"
        assert any(b.zone == 'bottom_bar' for b in blocks)


def test_category_field_slots_covers_all_six_categories():
    assert set(CATEGORY_FIELD_SLOTS.keys()) == set(CATEGORIES)
    for cat in CATEGORIES:
        assert isinstance(CATEGORY_FIELD_SLOTS[cat], list) and len(CATEGORY_FIELD_SLOTS[cat]) >= 1


def test_category_field_slots_excludes_internal_pseudo_fields():
    """'dates' is a spec-only internal pseudo-field (date_start+date_end merge, used only
    by map_category_to_default_blocks()) -- it must never leak into CATEGORY_FIELD_SLOTS,
    which demo_server.py iterates expecting every entry to be a real GenerateRequest field."""
    assert "dates" not in CATEGORY_FIELD_SLOTS["promo"]
    assert "dates" in CATEGORY_SCHEMA["promo"]


def test_default_field_role_has_no_duplicate_field_names_across_categories():
    """Field names must be unique across all 6 categories -- DEFAULT_FIELD_ROLE is a flat
    dict keyed by field name only (not per-category), so a collision would silently drop one
    category's role."""
    seen = set()
    for cat, fields in CATEGORY_SCHEMA.items():
        for field in fields:
            assert field not in seen, f"duplicate field name '{field}' across categories"
            seen.add(field)
    assert seen == set(DEFAULT_FIELD_ROLE.keys())


def test_get_field_block_spec_matches_deterministic_path_style_variant():
    """The exact bug this schema merge fixes: get_field_block_spec() (used by the
    render_plan path) must agree with what map_category_to_default_blocks() (the
    deterministic path) actually renders for the same field -- both now derive from the
    same CATEGORY_SCHEMA, so this can no longer drift."""
    spec = get_field_block_spec("product_intro", "price")
    assert spec["style_variant"] == "luxury_tag"

    blocks = map_category_to_default_blocks(
        category="product_intro",
        title=None,
        fields={"price": "1.990.000đ"},
    )
    price_block = next(b for b in blocks if b.field == "price")
    assert price_block.style_variant == spec["style_variant"]
    assert price_block.size == spec["size"]
    assert price_block.container == spec["container"]


def test_get_field_block_spec_unknown_field_returns_empty_dict():
    assert get_field_block_spec("promo", "not_a_real_field") == {}
    assert get_field_block_spec("not_a_real_category", "discount") == {}
