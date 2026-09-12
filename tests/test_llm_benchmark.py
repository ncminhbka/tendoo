"""
tests/test_llm_benchmark.py

Unit tests cho bộ kiểm thử và chấm điểm LLM Sidecar (scripts/benchmark_llm_sidecar.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_llm_sidecar import (
    BenchmarkCase,
    build_benchmark_suite,
    evaluate_case_output,
    run_benchmark,
)


def test_build_benchmark_suite_has_12_cases():
    cases = build_benchmark_suite()
    assert len(cases) == 12
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "Tất cả test case ID phải là duy nhất"


def test_evaluate_case_output_pass():
    case = BenchmarkCase(
        id="test_case",
        name="Test",
        category="opening",
        form_fields={},
        title_value="",
        prompt="Khai trương quán cafe",
        expected_title_action="use_prompt",
        expected_title_keywords=["GRAND OPENING"],
        expected_zones=["middle_center"],
        forbidden_scene_strings=["GRAND OPENING", "16:9"],
    )
    raw_json = json.dumps({
        "title": {"action": "use_prompt", "text": "GRAND OPENING"},
        "extra_blocks": [{"field": None, "text": "Menu mới", "zone": "middle_center", "role": "badge"}],
        "scene_prompt": "A modern coffee shop with warm lighting and wooden furniture, commercial photography",
        "style_hint": "warm_cafe",
        "font_key": "playfair",
        "headline_effect": "neon"
    })

    res = evaluate_case_output(case, raw_json, latency_sec=0.15)
    assert res.is_json_valid is True
    assert res.json_score == 20.0
    assert res.schema_score == 20.0
    assert res.spatial_score == 20.0
    assert res.scene_cleanliness_score == 20.0
    assert res.total_score >= 90.0
    assert res.passed is True


def test_evaluate_case_output_catches_leakage():
    case = BenchmarkCase(
        id="test_leak",
        name="Test Leakage",
        category="promo",
        form_fields={},
        title_value="",
        prompt="Sale 50%",
        forbidden_scene_strings=["Sale 50%", "16:9"],
    )
    # Rò rỉ chữ "Sale 50%" và "16:9" vào scene_prompt
    raw_json = json.dumps({
        "title": {"action": "none", "text": None},
        "extra_blocks": [],
        "scene_prompt": "A promotional banner showing Sale 50% in 16:9 aspect ratio",
        "style_hint": "auto",
        "font_key": "auto"
    })

    res = evaluate_case_output(case, raw_json, latency_sec=0.1)
    # Phải bị trừ điểm rò rỉ
    assert res.scene_cleanliness_score < 20.0
    assert any("[CẢNH BÁO RÒ RỈ]" in a for a in res.audit_notes)


def test_run_benchmark_mock_mode(tmp_path):
    summary = run_benchmark(mode="mock", out_dir_path=str(tmp_path))
    assert summary["total_cases"] == 12
    assert summary["pass_rate_percent"] >= 80.0
    assert (tmp_path / "benchmark_report.json").exists()
    assert (tmp_path / "benchmark_report.md").exists()
