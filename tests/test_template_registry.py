"""Thêm template mới phải chạm đủ 4 nơi (GĐ 0B): template.html, catalog.py, hàm zone trong
geometry.py, hàm ngân sách trong renderer.py. Test này báo ĐÚNG chỗ còn thiếu, thay vì để
renderer/geometry âm thầm rơi về giá trị mặc định."""

from __future__ import annotations

import inspect

import pytest

from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.geometry import _GEOMETRY_FUNCS, geometry_drivers
from tendoo_v3.renderer import _TEMPLATE_BUDGETS

TEMPLATES = sorted(TEMPLATE_CATALOG)


def test_registries_match_catalog():
    assert set(_GEOMETRY_FUNCS) == set(TEMPLATE_CATALOG), "geometry._GEOMETRY_FUNCS lệch catalog"
    assert set(_TEMPLATE_BUDGETS) == set(TEMPLATE_CATALOG), "renderer._TEMPLATE_BUDGETS lệch catalog"


@pytest.mark.parametrize("template", TEMPLATES)
def test_zone_fn_accepts_declared_params(template):
    params = inspect.signature(_GEOMETRY_FUNCS[template]).parameters
    has_orient = "default_orientation" in TEMPLATE_CATALOG[template]
    assert ("orientation" in params) == has_orient, (
        f"{template}: catalog default_orientation={has_orient} nhưng hàm zone "
        f"{'có' if 'orientation' in params else 'không có'} tham số orientation"
    )
    for flag in geometry_drivers(template):
        assert flag in params, f"{template}: catalog khai báo drives_geometry {flag} nhưng hàm zone không nhận"


@pytest.mark.parametrize("template", TEMPLATES)
def test_budget_fn_returns_hero(template):
    from tendoo_v3.geometry import get_zones
    from tendoo_v3.renderer import _zone_to_ctx, compute_template_budget
    from tendoo_v3.schema import TendooCreativePlan

    plan = TendooCreativePlan(template=template, hero="GIẢM 50%", subhead="Toàn bộ menu")
    zones = {k: _zone_to_ctx(v) for k, v in get_zones(template, 1024, 1024).items()}
    budget = compute_template_budget(template, plan, zones, 1024, 1024)
    assert budget.get("hero", {}).get("max_font"), f"{template}: ngân sách không có hero.max_font"
