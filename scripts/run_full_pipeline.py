#!/usr/bin/env python3
"""
scripts/run_full_pipeline.py

==================================================================================================
TENDOO AI -- FULL 4-STAGE PIPELINE ORCHESTRATOR (first real one; none existed before this)
==================================================================================================
Stage 1 (Blueprint):  OpenAI gpt-4o (see STAGE1_MODEL -- switched from gpt-4o-mini 2026-09-06, a
                       real reliability gap found+fixed, see memory
                       css-hero-title-overlay-direction.md), JSON mode, reusing
                       HERO_SELECTOR_SYSTEM_PROMPT + validate_blueprint() from
                       scripts/test_hero_selector.py. Called WITHOUT a category_hint -- this is the
                       first real test of whether the model can infer category from free-form text
                       alone (the "Khac" chat flow in the app, as opposed to a suggestion-chip flow
                       where category is already known).
Stage 2 (Background):  OpenAI gpt-image-1, quality="low", same calling pattern as
                       scripts/gen_representative_test_images.py.
Stage 3 (Safe-zone):    Wraps detect_faces_opencv/detect_product_yoloworld from
                       scripts/probe_mer_representative_cases.py + compute_safe_rect_for_category
                       from src/tendoo/layout_geometry.py. Runs only for categories that actually
                       have a safe_rect hook wired in typography_engine.py (menu/generic don't).
Stage 4 (Render):       PosterTemplateEngine.generate_html + PosterRenderer.render, unchanged.

Usage:
  python scripts/run_full_pipeline.py --lines 1,7,13,19,21,33,39
  python scripts/run_full_pipeline.py --lines 1,7,13,19,21,33,39 --prompt-file prompt_test.txt

Cost: ~0.3-0.5 USD for 7 lines (gpt-4o Stage 1 calls a few cents total, still small next to
gpt-image-1 quality=low ~0.02-0.04/image, which dominates the per-line cost either way).
==================================================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_full_pipeline")

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from tendoo.typography_engine import PosterBackgroundAnalyzer, PosterRenderer, PosterTemplateEngine  # noqa: E402
from tendoo.layout_geometry import compute_safe_rect_for_category  # noqa: E402

from test_hero_selector import HERO_SELECTOR_SYSTEM_PROMPT, validate_blueprint  # noqa: E402
from probe_mer_representative_cases import detect_faces_opencv, detect_product_yoloworld  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OUT_ROOT = PROJECT_ROOT / "output_full_pipeline"

# Switched from gpt-4o-mini to gpt-4o (2026-09-06) after a real reliability gap was found + fixed:
# gpt-4o-mini never picked up the new feedback/before_after brand_color/secondary_color fields
# (see memory css-hero-title-overlay-direction.md) even with explicit instructions/examples in
# HERO_SELECTOR_SYSTEM_PROMPT, while gpt-4o did -- verified on a real held-out set (all 11
# feedback/before_after lines in prompt_test.txt, not just the 1 line that first surfaced the gap):
# 10/11 correctly extracted a real 2-tone hex pair matching the request's described colors, the
# 11th correctly left secondary_color empty when the request only named ONE tone, `stars` came back
# as a proper "★★★★★" string in all 11 (gpt-4o-mini had returned a bare int at least once), and
# font_mood picks matched the requested typography mood well (vd "to rõ dễ đọc"->clean_readable,
# "chữ ký bay bổng"->elegant_script). Kept as ONE constant (not conditional per category) since
# Stage 1 must pick a model before it even knows the category it's about to choose.
STAGE1_MODEL = "gpt-4o"

# Only these categories have a `safe_rect` hook wired in typography_engine.py today (see (b)/(c)
# in memory css-hero-title-overlay-direction.md) -- menu/generic don't, so Stage 3 would be wasted
# work there.
_STAGE3_ENABLED_CATEGORIES = {"product_ad", "grand_opening", "feedback", "recruitment", "before_after"}


# ==================================================================================================
# Stage 1
# ==================================================================================================

def call_stage1(request: str) -> Tuple[Dict[str, Any], List[str]]:
    """Calls STAGE1_MODEL with NO category_hint -- the model must infer category from the free-form
    request text alone, exactly like the app's "Khac" (free chat) flow. Returns (blueprint, errors)
    -- errors from validate_blueprint() are logged, not hard-failed (matches the existing script's
    own tolerant spirit)."""
    resp = _client.chat.completions.create(
        model=STAGE1_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": HERO_SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Yêu cầu người dùng:\n{request}"},
        ],
    )
    blueprint = json.loads(resp.choices[0].message.content)
    errors = validate_blueprint(blueprint)
    return blueprint, errors


# ==================================================================================================
# Stage 2
# ==================================================================================================

def _pick_image_size(canvas: Dict[str, Any]) -> Tuple[str, int, int]:
    """Maps Stage 1's own chosen canvas dimensions to the nearest gpt-image-1 supported size.
    Returns (api_size_str, actual_w, actual_h) -- `actual_w`/`actual_h` are what the REST of the
    pipeline (Stage 3 MER math, Stage 4 render) must use, since they must exactly match the real
    generated photo's pixel size (see typography_engine.py's Type Collage crop-math invariant)."""
    w = int(canvas.get("width", 1024) or 1024)
    h = int(canvas.get("height", 1024) or 1024)
    if abs(w - h) < 0.15 * max(w, h):
        return "1024x1024", 1024, 1024
    if h > w:
        return "1024x1536", 1024, 1536
    return "1536x1024", 1536, 1024


def call_stage2(background_prompt: str, canvas: Dict[str, Any], out_path: Path) -> Tuple[Path, int, int]:
    import base64
    size, w, h = _pick_image_size(canvas)
    resp = _client.images.generate(model="gpt-image-1", prompt=background_prompt, size=size, quality="low", n=1)
    b64 = resp.data[0].b64_json
    out_path.write_bytes(base64.b64decode(b64))
    return out_path, w, h


# ==================================================================================================
# Stage 3
# ==================================================================================================

# feedback/before_after now use a small corner-card (redesigned 2026-09-05), not the full-width
# bottom-stack grand_opening/recruitment still use -- needs the shorter "bottom-compact" search
# region (see layout_geometry.py's docstring for the real bug this fixes: plain "bottom"'s ~80%-
# tall region put the compact card's `top` wherever that big region's own arbitrary boundary was,
# not where an obstacle actually was).
_COMPACT_CARD_CATEGORIES = {"feedback", "before_after"}


def _vertical_anchor_from_title_position(category: str, title_position: Optional[str]) -> str:
    if category in _COMPACT_CARD_CATEGORIES:
        return "bottom-compact"
    if category != "product_ad":
        # bottom-stack/frosted-box convention -- see layout_geometry.py's own docstring.
        # grand_opening/recruitment have no title_position field at all.
        return "bottom"
    v = (title_position or "top-center").replace("_", "-").split("-")[0]
    return v if v in ("top", "middle", "bottom") else "bottom"


def call_stage3(
    image_path: Path, canvas_w: int, canvas_h: int, orientation: str, category: str,
    title_position: Optional[str], product_keywords: Optional[List[str]] = None,
    subtitle_position: Optional[str] = None,
) -> Tuple[Dict[str, Optional[Dict[str, float]]], Dict[str, Any]]:
    """Returns ({"safe_rect": ..., "subtitle_safe_rect": ...}, detection_log_dict). Each rect is an
    EmptyRect.as_css_percent()-shaped dict, or None -- meaning the caller falls back to Cấp độ 1's
    fixed zone for that zone, exactly like brief.get("safe_rect") being absent today. Never raises
    -- a detector hiccup degrades to None, not a pipeline failure.

    `subtitle_safe_rect` is only computed for `product_ad` when `subtitle_position` is given and
    differs from `title_position` ("independent mode" -- see typography_engine.py's
    _generate_product_ad) -- REAL BUG found+fixed after the first real-pipeline run (2026-09-05,
    see memory css-hero-title-overlay-direction.md): the subtitle's own independent zone was never
    protected by Cấp độ 2 at all, only the title's was -- a real product/face sitting in the
    subtitle's fixed zone (prompt_test.txt line 1's own case) got covered. Detection (the expensive
    part) still runs only ONCE -- the same forbidden boxes are reused for a second, cheap MER call
    with the subtitle's own anchor band."""
    log: Dict[str, Any] = {"category": category, "stage3_ran": False}
    if category not in _STAGE3_ENABLED_CATEGORIES:
        log["skip_reason"] = f"category '{category}' has no safe_rect hook in typography_engine.py"
        return {"safe_rect": None, "subtitle_safe_rect": None}, log

    log["stage3_ran"] = True
    forbidden: List[Tuple[float, float, float, float]] = []
    faces: List[Tuple[float, float, float, float]] = []
    try:
        faces, _dt = detect_faces_opencv(image_path)
        forbidden.extend(faces)
    except Exception as e:
        logger.warning(f"[Stage3] face detection failed: {e}")
    log["faces_detected"] = list(faces)

    yolo_boxes = []
    if product_keywords:
        try:
            boxes, _dt = detect_product_yoloworld(image_path, product_keywords)
            yolo_boxes = boxes
            forbidden.extend(b[2] for b in boxes if b[1] >= 0.20)
        except Exception as e:
            logger.warning(f"[Stage3] YOLO-World detection failed: {e}")
    else:
        log["yolo_skip_reason"] = "no product_keywords from Stage 1"
    log["yolo_boxes"] = [(name, conf, list(xyxy)) for name, conf, xyxy in yolo_boxes]
    log["product_keywords_used"] = product_keywords or []

    def _mer(anchor: str) -> Optional[Dict[str, float]]:
        try:
            rect = compute_safe_rect_for_category(canvas_w, canvas_h, orientation, forbidden, vertical_anchor=anchor)
            return rect.as_css_percent(canvas_w, canvas_h)
        except Exception as e:
            logger.warning(f"[Stage3] MER computation failed (anchor={anchor}): {e}")
            return None

    vertical_anchor = _vertical_anchor_from_title_position(category, title_position)
    log["vertical_anchor"] = vertical_anchor
    safe_rect = _mer(vertical_anchor)
    log["safe_rect"] = safe_rect

    subtitle_safe_rect = None
    independent_mode = category == "product_ad" and subtitle_position and subtitle_position != title_position
    if independent_mode:
        subtitle_anchor = _vertical_anchor_from_title_position(category, subtitle_position)
        log["subtitle_vertical_anchor"] = subtitle_anchor
        subtitle_safe_rect = _mer(subtitle_anchor)
        log["subtitle_safe_rect"] = subtitle_safe_rect

    return {"safe_rect": safe_rect, "subtitle_safe_rect": subtitle_safe_rect}, log


# ==================================================================================================
# Stage 4
# ==================================================================================================

def call_stage4(blueprint: Dict[str, Any], safe_rects: Dict[str, Optional[Dict[str, float]]], image_path: Path, canvas_w: int, canvas_h: int, out_path: Path) -> Path:
    analysis = PosterBackgroundAnalyzer.analyze(image_path)
    category = blueprint.get("category", "generic")
    brief = dict(blueprint.get("template_brief") or {})  # fresh dict -- doesn't mutate blueprint,
    brief.setdefault("style_theme", blueprint.get("style_theme"))  # matches render_preview()'s pattern
    if safe_rects.get("safe_rect") is not None:
        brief["safe_rect"] = safe_rects["safe_rect"]
    if safe_rects.get("subtitle_safe_rect") is not None:
        brief["subtitle_safe_rect"] = safe_rects["subtitle_safe_rect"]
    html = PosterTemplateEngine.generate_html(analysis=analysis, brief=brief, background_image_path=str(image_path), category=category)
    PosterRenderer.render(html_content=html, output_image_path=out_path, width=canvas_w, height=canvas_h)
    return out_path


# ==================================================================================================
# Orchestration
# ==================================================================================================

def run_one_line(line_no: int, request: str, out_root: Path) -> Dict[str, Any]:
    line_dir = out_root / f"line{line_no}"
    line_dir.mkdir(parents=True, exist_ok=True)
    (line_dir / "00_request.txt").write_text(request, encoding="utf-8")

    row: Dict[str, Any] = {"line_id": line_no}
    t0 = time.time()
    try:
        blueprint, val_errors = call_stage1(request)
        (line_dir / "01_blueprint.json").write_text(
            json.dumps({"blueprint": blueprint, "validation_errors": val_errors}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        category = blueprint.get("category", "generic")
        row.update({"category": category, "validation_errors_count": len(val_errors)})
        if val_errors:
            logger.warning(f"[line {line_no}] validate_blueprint found {len(val_errors)} issue(s): {val_errors}")

        bg_path, canvas_w, canvas_h = call_stage2(
            blueprint.get("background_prompt", ""), blueprint.get("canvas", {}), line_dir / "02_background.png"
        )

        orientation = "portrait" if canvas_h > canvas_w else "landscape"
        brief = blueprint.get("template_brief") or {}
        safe_rects, detection_log = call_stage3(
            bg_path, canvas_w, canvas_h, orientation, category,
            title_position=brief.get("title_position"),
            product_keywords=blueprint.get("product_keywords") or [],
            subtitle_position=brief.get("subtitle_position"),
        )
        (line_dir / "03_detection_log.json").write_text(json.dumps(detection_log, ensure_ascii=False, indent=2), encoding="utf-8")
        row["stage3_ran"] = detection_log.get("stage3_ran", False)
        row["safe_rect_applied"] = safe_rects.get("safe_rect") is not None
        row["subtitle_safe_rect_applied"] = safe_rects.get("subtitle_safe_rect") is not None

        call_stage4(blueprint, safe_rects, bg_path, canvas_w, canvas_h, line_dir / "04_poster.png")
        row["error"] = None
    except Exception as e:
        logger.exception(f"[line {line_no}] pipeline failed")
        row["error"] = str(e)
    row["elapsed_seconds"] = round(time.time() - t0, 1)
    return row


def load_prompt_lines(prompt_file: Path, line_nos: List[int]) -> Dict[int, str]:
    raw_lines = prompt_file.read_text(encoding="utf-8").splitlines()
    result = {}
    for n in line_nos:
        if n < 1 or n > len(raw_lines):
            raise ValueError(f"line {n} out of range for {prompt_file} ({len(raw_lines)} lines)")
        text = raw_lines[n - 1].strip()
        if not text:
            raise ValueError(f"line {n} in {prompt_file} is blank -- check the line number")
        result[n] = text
    return result


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI full 4-stage pipeline orchestrator")
    parser.add_argument("--lines", type=str, required=True, help="Comma-separated 1-indexed line numbers in --prompt-file, e.g. 1,7,13,19,21,33,39")
    parser.add_argument("--prompt-file", type=str, default=str(PROJECT_ROOT / "prompt_test.txt"))
    args = parser.parse_args()

    line_nos = [int(x) for x in args.lines.split(",") if x.strip()]
    prompt_file = Path(args.prompt_file)
    prompts = load_prompt_lines(prompt_file, line_nos)

    OUT_ROOT.mkdir(exist_ok=True)
    summary = []
    for n in line_nos:
        logger.info(f"=== line {n} ===")
        row = run_one_line(n, prompts[n], OUT_ROOT)
        summary.append(row)
        logger.info(f"[line {n}] -> {row}")

    (OUT_ROOT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Done. Summary written to {OUT_ROOT / 'run_summary.json'}")


if __name__ == "__main__":
    main()
