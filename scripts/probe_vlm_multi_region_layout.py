#!/usr/bin/env python3
"""
scripts/probe_vlm_multi_region_layout.py

==================================================================================================
TENDOO AI -- SPIKE: VLM-driven multi-region layout vs. current template + single-safe_rect
==================================================================================================
Evaluation spike, NOT a production pipeline change (see plan
C:\\Users\\Admin\\.claude\\plans\\ok-t-m-b-qua-vivid-goose.md / memory
css-hero-title-overlay-direction.md 2026-09-06 update). For each test case, produces:

  <out_dir>/<case_id>/baseline.png                  -- CURRENT production pipeline (unchanged
                                                        PosterTemplateEngine + ONE Cấp độ 2
                                                        safe_rect), for a fair side-by-side.
  <out_dir>/<case_id>/experiment.png                 -- NEW multi-region VLM-planner pipeline
                                                        (src/tendoo/multi_region_layout.py).
  <out_dir>/<case_id>/experiment_regions_debug.png   -- background + every candidate region
                                                        outlined, for understanding what the VLM
                                                        was offered/chose.
  <out_dir>/<case_id>/assignment_log.json            -- full VLM input/output + validator result.
  <out_dir>/<case_id>/case_summary.json              -- timings, warnings, fallback_used, etc.
  <out_dir>/run_summary.json                         -- one row per case.

Test cases (see plan for why these 4):
  - product_ad line7, line19  -- REUSE the real blueprint+background+detection already saved in
                                  output_full_pipeline/ (zero incremental API cost for Stage 1/2/
                                  detection; these are real known-tricky compositions).
  - recruitment, menu         -- NO real saved data exists for these categories anywhere in the
                                  repo (confirmed by exploration) -- runs real Stage 1 (gpt-4o-mini,
                                  WITH an explicit category_hint this time, since this spike is
                                  evaluating layout quality, not re-testing the already-documented
                                  zero-hint misclassification risk) + real Stage 2 (gpt-image-1).

Usage:
  python scripts/probe_vlm_multi_region_layout.py --cases product_ad_line7,product_ad_line19,recruitment,menu
==================================================================================================
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("probe_vlm_multi_region_layout")

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402
from PIL import Image  # noqa: E402

from tendoo.layout_geometry import compute_safe_rect_for_category  # noqa: E402
from tendoo.typography_engine import PosterBackgroundAnalyzer, PosterRenderer, PosterTemplateEngine  # noqa: E402
from tendoo.multi_region_layout import (  # noqa: E402
    build_content_manifest, build_region_candidates, call_vlm_planner,
    render_debug_overlay, render_experiment_poster, resolve_assignments,
)

from test_hero_selector import HERO_SELECTOR_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, validate_blueprint  # noqa: E402
from probe_mer_representative_cases import detect_faces_opencv, detect_product_yoloworld  # noqa: E402
from run_full_pipeline import _pick_image_size, _vertical_anchor_from_title_position  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OUT_ROOT = PROJECT_ROOT / "output_vlm_multi_region_spike"
VLM_PLANNER_MODEL = "gpt-4o-mini"
STAGE1_MODEL = "gpt-4o-mini"

# Categories out of scope for this spike keep working exactly as in run_full_pipeline.py -- only
# these 3 have a `build_content_manifest` in multi_region_layout.py.
_MULTI_REGION_CATEGORIES = {"product_ad", "recruitment", "menu"}

# Real request text authored for this spike -- prompt_test.txt has NO recruitment/menu lines at
# all (confirmed by grep before writing this), so these mimic its own style/length/register.
REQUEST_TEXTS = {
    "recruitment": (
        'Tạo poster tuyển dụng công nghệ cho công ty "Tendoo AI Lab". Vị trí: Chuyên viên AI/Data '
        "Science. Mô tả ngắn: tìm 20 nhân tài công nghệ cho dự án AI sáng tạo poster quảng cáo tự "
        "động. Yêu cầu: tốt nghiệp CNTT/Khoa học dữ liệu, có kinh nghiệm Python/Machine Learning, "
        "tư duy sáng tạo, khả năng làm việc độc lập tốt. Quyền lợi: lương thưởng cạnh tranh lên tới "
        "40 triệu/tháng, môi trường làm việc trẻ trung năng động, được đào tạo bởi chuyên gia AI "
        "hàng đầu, du lịch công ty hàng năm, bảo hiểm sức khỏe cao cấp. Hạn nộp hồ sơ: 30/09/2026. "
        "Liên hệ: hotline 1900 6789, email tuyendung@tendoo.ai. Background: văn phòng công nghệ "
        "hiện đại, tông màu xanh dương và cam, ánh sáng neon, phong cách tương lai, có khoảng trống "
        "rõ ràng phía trên cho tiêu đề lớn."
    ),
    "menu": (
        'Tạo poster thực đơn cho quán "Bếp Cô Ba - Bánh Xèo Miền Tây". Có 2 nhóm món: "MÓN ĐẶC '
        'TRƯNG" gồm Bánh xèo tôm thịt (55.000đ), Bánh khọt (45.000đ), Gỏi cuốn tôm thịt (35.000đ); '
        '"THỨC UỐNG" gồm Nước sâm (15.000đ), Trà tắc (12.000đ). Tagline: "Hương vị miền Tây trong '
        'từng miếng bánh". Ghi chú: Mở cửa 8h-22h hàng ngày. Hotline: 0909 888 777. Background: '
        "không gian quán ăn miền Tây mộc mạc, bàn gỗ, lá chuối trang trí, ánh sáng vàng ấm, có "
        "khoảng trống rõ ràng cho tiêu đề và các cột thực đơn."
    ),
}
# Matches the exact hint strings test_hero_selector.py's own suggestion-chip test cases already
# use (lines 103/113) -- reusing established vocabulary, not inventing new hint phrasing.
CATEGORY_HINTS = {"recruitment": "tin tuyển dụng", "menu": "menu món ăn"}


# ==================================================================================================
# Stage 1/2/detection -- fresh real calls (recruitment/menu), matching run_full_pipeline.py's
# calling conventions but WITH an explicit category_hint (see module docstring for why).
# ==================================================================================================

def call_stage1_hinted(category: str, request: str) -> Tuple[Dict[str, Any], List[str]]:
    user_msg = USER_PROMPT_TEMPLATE.format(category_hint=CATEGORY_HINTS[category], request=request)
    resp = _client.chat.completions.create(
        model=STAGE1_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": HERO_SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    blueprint = json.loads(resp.choices[0].message.content)
    errors = validate_blueprint(blueprint)
    if blueprint.get("category") != category:
        errors.append(f"expected category={category!r} but Stage 1 returned {blueprint.get('category')!r}")
    return blueprint, errors


def call_stage2(background_prompt: str, canvas: Dict[str, Any], out_path: Path) -> Tuple[Path, int, int]:
    size, w, h = _pick_image_size(canvas)
    resp = _client.images.generate(model="gpt-image-1", prompt=background_prompt, size=size, quality="low", n=1)
    b64 = resp.data[0].b64_json
    out_path.write_bytes(base64.b64decode(b64))
    return out_path, w, h


def run_detection(image_path: Path, product_keywords: Optional[List[str]]) -> Tuple[List[Tuple[float, float, float, float]], Dict[str, Any]]:
    """Same detector pair/filter (conf >= 0.20 for YOLO-World) as run_full_pipeline.py's
    call_stage3 -- reused, not reimplemented, so the baseline branch below (which ALSO calls this)
    and the experiment branch see the identical forbidden-box set."""
    log: Dict[str, Any] = {}
    forbidden: List[Tuple[float, float, float, float]] = []
    try:
        faces, _dt = detect_faces_opencv(image_path)
        forbidden.extend(faces)
        log["faces_detected"] = list(faces)
    except Exception as e:
        logger.warning(f"face detection failed: {e}")
        log["faces_error"] = str(e)

    yolo_boxes = []
    if product_keywords:
        try:
            boxes, _dt = detect_product_yoloworld(image_path, product_keywords)
            yolo_boxes = boxes
            forbidden.extend(b[2] for b in boxes if b[1] >= 0.20)
        except Exception as e:
            logger.warning(f"YOLO-World detection failed: {e}")
            log["yolo_error"] = str(e)
    log["yolo_boxes"] = [(name, conf, list(xyxy)) for name, conf, xyxy in yolo_boxes]
    log["product_keywords_used"] = product_keywords or []
    return forbidden, log


# ==================================================================================================
# Baseline branch -- unchanged production path (PosterTemplateEngine + ONE Cấp độ 2 safe_rect).
# ==================================================================================================

def render_baseline(
    category: str, blueprint: Dict[str, Any], image_path: Path, canvas_w: int, canvas_h: int,
    orientation: str, forbidden: List[Tuple[float, float, float, float]], out_path: Path,
) -> Path:
    brief = dict(blueprint.get("template_brief") or {})
    brief.setdefault("style_theme", blueprint.get("style_theme"))
    anchor = _vertical_anchor_from_title_position(category, brief.get("title_position"))
    rect = compute_safe_rect_for_category(canvas_w, canvas_h, orientation, forbidden, vertical_anchor=anchor)
    brief["safe_rect"] = rect.as_css_percent(canvas_w, canvas_h)
    analysis = PosterBackgroundAnalyzer.analyze(image_path)
    html = PosterTemplateEngine.generate_html(
        analysis=analysis, brief=brief, background_image_path=str(image_path), category=category
    )
    PosterRenderer.render(html_content=html, output_image_path=out_path, width=canvas_w, height=canvas_h)
    return out_path


# ==================================================================================================
# Experiment branch -- new multi-region + VLM planner pipeline.
# ==================================================================================================

def render_experiment(
    category: str, blueprint: Dict[str, Any], image_path: Path, canvas_w: int, canvas_h: int,
    forbidden: List[Tuple[float, float, float, float]], case_dir: Path,
) -> Dict[str, Any]:
    brief = dict(blueprint.get("template_brief") or {})
    style_hint = blueprint.get("style_theme")

    blocks, corner_block = build_content_manifest(category, brief)
    regions, corner_box = build_region_candidates(canvas_w, canvas_h, forbidden, image_path, k=8)

    vlm_raw: Optional[Dict[str, Any]] = None
    vlm_error: Optional[str] = None
    try:
        vlm_raw = call_vlm_planner(_client, VLM_PLANNER_MODEL, category, canvas_w, canvas_h, blocks, regions, image_path)
    except Exception as e:
        vlm_error = str(e)
        logger.warning(f"[{category}] VLM planner call failed: {e}")

    resolved = resolve_assignments(blocks, regions, vlm_raw)
    for w in resolved["warnings"]:
        logger.warning(f"[{category}] {w}")

    experiment_png = case_dir / "experiment.png"
    render_experiment_poster(
        category, canvas_w, canvas_h, image_path, blocks, corner_block, regions, corner_box,
        resolved, style_hint, experiment_png,
    )
    debug_png = case_dir / "experiment_regions_debug.png"
    render_debug_overlay(image_path, regions, corner_box, debug_png)

    return {
        "blocks": blocks, "corner_block": corner_block, "regions": regions,
        "vlm_raw": vlm_raw, "vlm_error": vlm_error, "resolved": resolved,
        "experiment_png": str(experiment_png), "debug_png": str(debug_png),
    }


# ==================================================================================================
# Orchestration per case
# ==================================================================================================

def run_reused_product_ad_case(line_no: int, case_dir: Path) -> Dict[str, Any]:
    """Reuses the exact blueprint/background/detection already saved in output_full_pipeline/
    (zero incremental API cost) -- these are real, previously-flagged tricky compositions
    (line7: narrow-MER/hairline-touch case; line19: hiker-head-touch case)."""
    src_dir = PROJECT_ROOT / "output_full_pipeline" / f"line{line_no}"
    blueprint = json.loads((src_dir / "01_blueprint.json").read_text(encoding="utf-8"))["blueprint"]
    image_path = src_dir / "02_background.png"
    detection_log = json.loads((src_dir / "03_detection_log.json").read_text(encoding="utf-8"))

    canvas_w, canvas_h = Image.open(image_path).size
    orientation = "portrait" if canvas_h > canvas_w else "landscape"

    forbidden: List[Tuple[float, float, float, float]] = [tuple(b) for b in detection_log.get("faces_detected", [])]
    forbidden.extend(
        tuple(xyxy) for (_name, conf, xyxy) in detection_log.get("yolo_boxes", []) if conf >= 0.20
    )

    return _finish_case(
        case_id=f"product_ad_line{line_no}", category="product_ad", blueprint=blueprint,
        validation_errors=[], image_path=image_path, canvas_w=canvas_w, canvas_h=canvas_h,
        orientation=orientation, forbidden=forbidden, case_dir=case_dir,
        stage1_stage2_note=f"REUSED from output_full_pipeline/line{line_no}/ (no new API cost)",
    )


def run_fresh_case(category: str, case_dir: Path) -> Dict[str, Any]:
    request = REQUEST_TEXTS[category]
    blueprint, val_errors = call_stage1_hinted(category, request)
    (case_dir / "01_blueprint.json").write_text(
        json.dumps({"blueprint": blueprint, "validation_errors": val_errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if val_errors:
        logger.warning(f"[{category}] validate_blueprint found {len(val_errors)} issue(s): {val_errors}")

    image_path, canvas_w, canvas_h = call_stage2(
        blueprint.get("background_prompt", ""), blueprint.get("canvas", {}), case_dir / "02_background.png"
    )
    orientation = "portrait" if canvas_h > canvas_w else "landscape"

    forbidden, detection_log = run_detection(image_path, blueprint.get("product_keywords") or [])
    (case_dir / "03_detection_log.json").write_text(json.dumps(detection_log, ensure_ascii=False, indent=2), encoding="utf-8")

    return _finish_case(
        case_id=category, category=category, blueprint=blueprint, validation_errors=val_errors,
        image_path=image_path, canvas_w=canvas_w, canvas_h=canvas_h, orientation=orientation,
        forbidden=forbidden, case_dir=case_dir, stage1_stage2_note="fresh real Stage 1 + Stage 2 call",
    )


def _finish_case(
    case_id: str, category: str, blueprint: Dict[str, Any], validation_errors: List[str],
    image_path: Path, canvas_w: int, canvas_h: int, orientation: str,
    forbidden: List[Tuple[float, float, float, float]], case_dir: Path, stage1_stage2_note: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case_id": case_id, "category": category, "stage1_stage2_note": stage1_stage2_note,
        "canvas": {"width": canvas_w, "height": canvas_h, "orientation": orientation},
        "validation_errors": validation_errors, "forbidden_box_count": len(forbidden),
    }
    t0 = time.time()
    baseline_png = render_baseline(category, blueprint, image_path, canvas_w, canvas_h, orientation, forbidden, case_dir / "baseline.png")
    row["baseline_elapsed_s"] = round(time.time() - t0, 1)
    row["baseline_png"] = str(baseline_png)

    t1 = time.time()
    exp = render_experiment(category, blueprint, image_path, canvas_w, canvas_h, forbidden, case_dir)
    row["experiment_elapsed_s"] = round(time.time() - t1, 1)
    row["experiment_png"] = exp["experiment_png"]
    row["debug_png"] = exp["debug_png"]
    row["vlm_error"] = exp["vlm_error"]
    row["fallback_used"] = exp["resolved"]["fallback_used"]
    row["warnings"] = exp["resolved"]["warnings"]
    row["n_regions_found"] = len(exp["regions"])
    row["n_content_blocks"] = len(exp["blocks"])
    row["region_ids_used"] = sorted({a["region_id"] for a in exp["resolved"]["assignments"]})
    row["n_distinct_regions_used"] = len(row["region_ids_used"])

    (case_dir / "assignment_log.json").write_text(
        json.dumps({
            "blocks": exp["blocks"], "corner_block": exp["corner_block"], "regions": exp["regions"],
            "vlm_raw": exp["vlm_raw"], "vlm_error": exp["vlm_error"], "resolved": exp["resolved"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (case_dir / "case_summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


_CASE_RUNNERS = {
    "product_ad_line7": lambda case_dir: run_reused_product_ad_case(7, case_dir),
    "product_ad_line19": lambda case_dir: run_reused_product_ad_case(19, case_dir),
    "recruitment": lambda case_dir: run_fresh_case("recruitment", case_dir),
    "menu": lambda case_dir: run_fresh_case("menu", case_dir),
}


def main():
    parser = argparse.ArgumentParser(description="Spike: VLM multi-region layout vs. template baseline")
    parser.add_argument("--cases", type=str, default=",".join(_CASE_RUNNERS),
                         help=f"Comma-separated subset of {list(_CASE_RUNNERS)}")
    args = parser.parse_args()
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    for c in case_ids:
        if c not in _CASE_RUNNERS:
            raise ValueError(f"Unknown case '{c}', valid: {list(_CASE_RUNNERS)}")

    OUT_ROOT.mkdir(exist_ok=True)
    summary = []
    for case_id in case_ids:
        logger.info(f"=== case: {case_id} ===")
        case_dir = OUT_ROOT / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            row = _CASE_RUNNERS[case_id](case_dir)
            row["error"] = None
        except Exception as e:
            logger.exception(f"[{case_id}] case failed")
            row = {"case_id": case_id, "error": str(e)}
        summary.append(row)
        logger.info(f"[{case_id}] -> {json.dumps({k: v for k, v in row.items() if k not in ('warnings',)}, ensure_ascii=False)}")

    (OUT_ROOT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Done. Summary written to {OUT_ROOT / 'run_summary.json'}")


if __name__ == "__main__":
    main()
