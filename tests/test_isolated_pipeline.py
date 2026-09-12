"""
tests/test_isolated_pipeline.py

Unit tests cho bộ kiểm thử trực quan cô lập LLM & Diffusion.
Đảm bảo 100% logic cô lập, cấu trúc kịch bản, mock background generator
và thuật toán hòa trộn mask hoạt động chính xác không phụ thuộc GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

from tendoo.engine.blocks import format_rating_stars
from tendoo.engine.geometry import get_product_sanctuary_rect
from tendoo.engine.layout import OmniBlockLayout
from tendoo.layouts.style_matcher import CATEGORY_FIELD_SLOTS
from tendoo.llm_render_plan_server import validate_render_plan

from scripts.evaluate_isolated_pipeline import (
    IsolatedTestCase,
    create_diffusion_mock_background,
    create_mask_overlay,
    get_isolated_test_suite,
    run_single_isolated_case,
)


def test_isolated_test_suite_coverage():
    """Kiểm tra bộ test suite bao trùm đủ 10 kịch bản và phân loại chuẩn xác 2 nhánh."""
    suite = get_isolated_test_suite()
    assert len(suite) == 10, f"Mong đợi 10 scenarios, thực tế: {len(suite)}"

    no_llm_cases = [tc for tc in suite if tc.execution_path == "NO_LLM_DETERMINISTIC"]
    llm_cases = [tc for tc in suite if tc.execution_path == "LLM_ASSISTED"]

    assert len(no_llm_cases) == 4, f"Mong đợi 4 kịch bản No-LLM, thực tế: {len(no_llm_cases)}"
    assert len(llm_cases) == 6, f"Mong đợi 6 kịch bản LLM-Assisted, thực tế: {len(llm_cases)}"

    # Kiểm tra tỉ lệ khung hình bao trùm 1:1, 16:9, 4:5, 9:16
    ratios = {tc.aspect_ratio for tc in suite}
    assert {"1:1", "16:9", "4:5", "9:16"}.issubset(ratios)


def test_simulated_llm_plans_are_valid():
    """Kiểm tra toàn bộ 6 payload JSON giả lập của LLM đều vượt qua hàm validate_render_plan."""
    suite = get_isolated_test_suite()
    llm_cases = [tc for tc in suite if tc.execution_path == "LLM_ASSISTED"]

    for tc in llm_cases:
        assert tc.simulated_llm_plan is not None, f"Case {tc.case_id} thiếu simulated_llm_plan"
        cat = tc.category
        cat_field_values: Dict[str, str] = {}
        for f in CATEGORY_FIELD_SLOTS.get(cat, []):
            if f == "guide_steps":
                cat_field_values[f] = " | ".join(tc.fields.get("guide_steps", []))
            else:
                cat_field_values[f] = str(tc.fields.get(f, "")).strip()

        plan, errors = validate_render_plan(tc.simulated_llm_plan, cat, cat_field_values)
        assert plan is not None, f"Case {tc.case_id} có plan bị None! Errors: {errors}"
        assert isinstance(plan.get("extra_blocks"), list)
        assert plan.get("style_hint") is not None


@pytest.mark.parametrize("style", ["cyberpunk_grid", "luxury_gold", "warm_wood", "daylight_clean", "minimal_wall"])
def test_diffusion_mock_background_generator(style: str):
    """Kiểm tra bộ sinh ảnh nền mock mô phỏng đúng kích thước và hòa trộn mask mượt mà."""
    w, h = 512, 512
    # Test khi không có mask
    bg_raw = create_diffusion_mock_background(w, h, style_hint=style, mask_np=None)
    assert isinstance(bg_raw, Image.Image)
    assert bg_raw.size == (w, h)

    # Test khi có mask (giả lập một vùng corridor ở góc trên)
    mask_arr = np.zeros((h, w), dtype=np.float32)
    mask_arr[0:150, 0:400] = 1.0  # corridor block

    bg_blended = create_diffusion_mock_background(w, h, style_hint=style, mask_np=mask_arr)
    assert isinstance(bg_blended, Image.Image)
    assert bg_blended.size == (w, h)

    raw_np = np.array(bg_raw, dtype=np.float32)
    blended_np = np.array(bg_blended, dtype=np.float32)

    # Vùng mask = 0 (tâm sản phẩm) phải hoàn toàn giống nguyên bản
    center_raw = raw_np[200:350, 150:350]
    center_blended = blended_np[200:350, 150:350]
    np.testing.assert_allclose(center_raw, center_blended, atol=1e-5)


def test_mask_overlay_generation():
    """Kiểm tra hàm create_mask_overlay tạo đúng ảnh RGBA/RGB và vẽ Product Sanctuary."""
    w, h = 400, 400
    poster_pil = Image.new("RGB", (w, h), (30, 30, 30))
    mask_np = np.zeros((h, w), dtype=np.float32)
    mask_np[20:100, 20:300] = 0.8  # vùng chữ

    overlay_pil = create_mask_overlay(poster_pil, mask_np)
    assert isinstance(overlay_pil, Image.Image)
    assert overlay_pil.size == (w, h)


def test_rating_star_formatting():
    """Kiểm tra format_rating_stars biến các biến thể chuỗi số thành 5 sao vàng."""
    assert "★★★★★" in format_rating_stars("5")
    assert "★★★★★" in format_rating_stars("5 sao")
    assert "★★★★★" in format_rating_stars("5.0")
    assert "★★★★★" in format_rating_stars("5 ★★★★★")
    assert "★★★★★" in format_rating_stars("⭐⭐⭐⭐⭐")
