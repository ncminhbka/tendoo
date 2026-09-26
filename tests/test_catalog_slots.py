"""`catalog.py` slots là nguồn sự thật về field nào template hiển thị (ROADMAP §6.2) -- test
này giữ nó khớp với chính template.html, để khai báo không trôi dạt khỏi code thật."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.geometry import geometry_drivers

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "src" / "tendoo_v3" / "templates"

# Biến Jinja -> field của TendooCreativePlan mà nó hiển thị.
VAR_TO_FIELD = {
    "hero": "hero", "hero_parts": "hero", "subhead": "subhead", "badge": "badge",
    "extra_texts": "extra_texts", "extra_pills": "extra_texts",
    "cta": "cta", "store_info": "store_info", "store_items": "store_info",
    "qr_svg": "qr_code", "stars_svg": "rating",
    "testimonial": "testimonial", "reviewer_name": "reviewer_name", "steps": "steps",
    "tag_left": "tag_left", "tag_right": "tag_right",
}
KNOWN_FLAGS = {"has_qr", "has_footer", "has_message", "has_freetext"}


def fields_rendered_by(template: str) -> set:
    src = (TEMPLATES_DIR / template / "template.html").read_text(encoding="utf-8")
    return {f for v, f in VAR_TO_FIELD.items() if re.search(r"(\{\{|\{%)[^}]*\b" + v + r"\b", src)}


@pytest.mark.parametrize("template", sorted(TEMPLATE_CATALOG))
def test_slots_match_template_html(template):
    declared = set(TEMPLATE_CATALOG[template]["slots"])
    rendered = fields_rendered_by(template)
    assert declared == rendered, (
        f"{template}: khai báo thừa {sorted(declared - rendered)}, template hiển thị mà chưa khai báo {sorted(rendered - declared)}"
    )


@pytest.mark.parametrize("template", sorted(TEMPLATE_CATALOG))
def test_geometry_drivers_are_known_flags(template):
    assert set(geometry_drivers(template)) <= KNOWN_FLAGS
