#!/usr/bin/env python3
"""
scripts/probe_reserve_and_edit.py

==================================================================================================
TENDOO AI -- SPIKE: "Reserve-and-Edit" (chừa chỗ chữ TRƯỚC khi sinh ảnh, không dò-sau)
==================================================================================================
User's proposal (2026-09-06): thay vì sinh ảnh mù rồi mới dò tìm chỗ trống đặt chữ (cách hiện tại,
dễ vỡ -- xem bug thật line5: face detector bỏ sót mặt người cha, chữ đè lên mặt), hãy CHỪA CHỖ chữ
TRƯỚC bằng 1 mask, rồi để gpt-image-1's `images.edit()` sinh ảnh xung quanh vùng đó.

Cơ chế đã kiểm chứng bằng 2 test có kiểm soát (xem memory css-hero-title-overlay-direction.md):
`images.edit()` KHÔNG giữ nguyên pixel trong vùng mask như inpaint cổ điển -- nó vẽ lại TOÀN BỘ
ảnh, dùng mask + nội dung ảnh gốc trong vùng đó như 1 gợi ý bố cục MẠNH (không phải khoá cứng).
Vùng mask có sẵn 1 placeholder rõ ràng (hình dạng thật của chữ) được né tốt hơn hẳn vùng mask để
trống trơn -- nên bước "seed" (vẽ placeholder trước khi gọi edit) là bắt buộc, không phải tuỳ chọn.

Pipeline từng dòng:
  1. Đọc blueprint đã có (tái dùng output_full_pipeline/lineN/01_blueprint.json nếu có, tiết kiệm
     chi phí Stage 1) -- lấy title_position/subtitle_position/canvas.
  2. Render CHỈ riêng khối chữ (dùng đúng PosterTemplateEngine._generate_product_ad, background=
     None) lên nền trắng, ĐO chính xác bounding box thật của .title-text/.subtitle-text bằng
     Playwright (không đoán) -- đây là "placeholder" thật (đúng font/size/wrap), không phải hình
     chữ nhật vẽ tay.
  3. Dựng mask (PIL, alpha=255 tại đúng 2 bounding box đã đo + đệm nhẹ, alpha=0 chỗ còn lại).
  4. Gọi images.edit(image=seed có chữ, mask, prompt=background_prompt gốc + "không vẽ chữ") --
     sinh ảnh thật.
  5. Overlay chữ thật (Stage 4, KHÔNG đổi code) vào ĐÚNG vị trí đã đo ở bước 2 -- không cần detect
     gì nữa vì vị trí đã biết trước bằng chính minh (không phải suy ra sau khi ảnh đã tồn tại).

Usage:
  python scripts/probe_reserve_and_edit.py --lines 1,5,7
Output: output_reserve_edit_experiment/lineN/{00_request.txt, 01_blueprint.json, 02_seed_base.png,
        02_seed_mask.png, 03_edit_result.png, 04_final_poster.png}
==================================================================================================
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("probe_reserve_and_edit")

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from tendoo.typography_engine import PosterBackgroundAnalyzer, PosterRenderer, PosterTemplateEngine  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OUT_ROOT = PROJECT_ROOT / "output_reserve_edit_experiment"

# Extra breathing room around each measured text bounding box, as a fraction of canvas
# width/height -- gives the model a slightly bigger "keep clear" hint than the bare glyph pixels,
# and leaves room for the real Stage 4 scrim/padding that isn't present in this bare-text seed.
PAD_FRAC = 0.025


def render_text_only_and_measure(brief: Dict[str, Any], canvas_w: int, canvas_h: int) -> Tuple[bytes, List[Dict[str, float]]]:
    """Renders ONLY the title/subtitle text (no background photo) via the REAL
    `PosterTemplateEngine._generate_product_ad`, then measures the ACTUAL rendered bounding boxes
    of `.title-text`/`.subtitle-text` via Playwright's `getBoundingClientRect()` -- not guessed,
    not hand-drawn, the exact pixels the real typography engine would occupy for this exact
    text/style/font/canvas size. Returns (full-canvas PNG bytes with the text rendered on a plain
    white background, list of {x,y,w,h} boxes in DOM order [title, subtitle-if-any])."""
    blank = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    blank_path = OUT_ROOT / "_blank_ref.png"
    OUT_ROOT.mkdir(exist_ok=True)
    blank.save(blank_path)
    analysis = PosterBackgroundAnalyzer.analyze(blank_path)
    analysis.width, analysis.height = canvas_w, canvas_h

    html = PosterTemplateEngine._generate_product_ad(analysis, brief, background_image_path=None)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": canvas_w, "height": canvas_h})
        page.set_content(html, wait_until="networkidle")
        boxes = page.eval_on_selector_all(
            ".title-text, .subtitle-text",
            "els => els.map(el => { const r = el.getBoundingClientRect(); "
            "return {x:r.x, y:r.y, w:r.width, h:r.height}; })",
        )
        png_bytes = page.screenshot(full_page=True)
        browser.close()
    return png_bytes, boxes


def build_mask(canvas_w: int, canvas_h: int, boxes: List[Dict[str, float]]) -> Image.Image:
    """Alpha=255 (opaque) at each measured text box (+ padding) -- everywhere else alpha=0."""
    mask = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    pad_x, pad_y = canvas_w * PAD_FRAC, canvas_h * PAD_FRAC
    for b in boxes:
        x1, y1 = max(0, b["x"] - pad_x), max(0, b["y"] - pad_y)
        x2, y2 = min(canvas_w, b["x"] + b["w"] + pad_x), min(canvas_h, b["y"] + b["h"] + pad_y)
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 255))
    return mask


def box_to_rect_pct(b: Dict[str, float], canvas_w: int, canvas_h: int) -> Dict[str, float]:
    """Same padded box -> the `EmptyRect.as_css_percent()`-shaped dict Stage 4's `safe_rect`/
    `subtitle_safe_rect` already expect -- lets us feed our OWN measured (not detected) position
    straight into the existing, unmodified Stage 4 rendering path."""
    pad_x, pad_y = canvas_w * PAD_FRAC, canvas_h * PAD_FRAC
    x1, y1 = max(0, b["x"] - pad_x), max(0, b["y"] - pad_y)
    x2, y2 = min(canvas_w, b["x"] + b["w"] + pad_x), min(canvas_h, b["y"] + b["h"] + pad_y)
    return {
        "left_pct": round(100 * x1 / canvas_w, 2), "top_pct": round(100 * y1 / canvas_h, 2),
        "right_pct": round(100 * (canvas_w - x2) / canvas_w, 2), "bottom_pct": round(100 * (canvas_h - y2) / canvas_h, 2),
        "width_pct": round(100 * (x2 - x1) / canvas_w, 2), "height_pct": round(100 * (y2 - y1) / canvas_h, 2),
    }


def run_one_line(line_no: int, out_root: Path) -> Dict[str, Any]:
    src_dir = PROJECT_ROOT / "output_full_pipeline" / f"line{line_no}"
    blueprint = json.loads((src_dir / "01_blueprint.json").read_text(encoding="utf-8"))["blueprint"]
    request = (src_dir / "00_request.txt").read_text(encoding="utf-8")
    canvas_w, canvas_h = blueprint["canvas"]["width"], blueprint["canvas"]["height"]
    # `_pick_image_size`-equivalent rounding to a gpt-image-1-supported size (must match exactly).
    if abs(canvas_w - canvas_h) < 0.15 * max(canvas_w, canvas_h):
        size_str, canvas_w, canvas_h = "1024x1024", 1024, 1024
    elif canvas_h > canvas_w:
        size_str, canvas_w, canvas_h = "1024x1536", 1024, 1536
    else:
        size_str, canvas_w, canvas_h = "1536x1024", 1536, 1024

    brief = dict(blueprint["template_brief"])
    background_prompt = blueprint["background_prompt"]

    out_dir = out_root / f"line{line_no}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "00_request.txt").write_text(request, encoding="utf-8")
    (out_dir / "01_blueprint.json").write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"[line {line_no}] measuring real text boxes...")
    seed_png, boxes = render_text_only_and_measure(brief, canvas_w, canvas_h)
    (out_dir / "02_seed_base.png").write_bytes(seed_png)
    mask = build_mask(canvas_w, canvas_h, boxes)
    mask.save(out_dir / "02_seed_mask.png")
    logger.info(f"[line {line_no}] measured {len(boxes)} text box(es): {boxes}")

    logger.info(f"[line {line_no}] calling images.edit ...")
    resp = _client.images.edit(
        model="gpt-image-1",
        image=open(out_dir / "02_seed_base.png", "rb"),
        mask=open(out_dir / "02_seed_mask.png", "rb"),
        prompt=background_prompt + " Do not add any text, letters, or writing anywhere in the image.",
        size=size_str,
        quality="low",
    )
    edit_png = base64.b64decode(resp.data[0].b64_json)
    (out_dir / "03_edit_result.png").write_bytes(edit_png)

    # Stage 4, unmodified: feed our OWN measured boxes back in as safe_rect/subtitle_safe_rect --
    # no detection needed, we already know exactly where the text goes (we decided it ourselves).
    independent_mode = bool(brief.get("subtitle_position")) and brief.get("subtitle_position") != brief.get("title_position")
    if independent_mode and len(boxes) >= 2:
        brief["safe_rect"] = box_to_rect_pct(boxes[0], canvas_w, canvas_h)
        brief["subtitle_safe_rect"] = box_to_rect_pct(boxes[1], canvas_w, canvas_h)
    elif boxes:
        # Stacked mode: title+subtitle share one zone -- union of all measured boxes.
        x1 = min(b["x"] for b in boxes); y1 = min(b["y"] for b in boxes)
        x2 = max(b["x"] + b["w"] for b in boxes); y2 = max(b["y"] + b["h"] for b in boxes)
        union_box = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        brief["safe_rect"] = box_to_rect_pct(union_box, canvas_w, canvas_h)

    analysis_final = PosterBackgroundAnalyzer.analyze(out_dir / "03_edit_result.png")
    html_final = PosterTemplateEngine._generate_product_ad(analysis_final, brief, background_image_path=str(out_dir / "03_edit_result.png"))
    PosterRenderer.render(html_content=html_final, output_image_path=out_dir / "04_final_poster.png", width=canvas_w, height=canvas_h)
    logger.info(f"[line {line_no}] done -> {out_dir / '04_final_poster.png'}")

    return {"line_id": line_no, "canvas": f"{canvas_w}x{canvas_h}", "n_text_boxes": len(boxes), "independent_mode": independent_mode}


def main():
    parser = argparse.ArgumentParser(description="Spike: reserve-and-edit (seed text BEFORE image generation)")
    parser.add_argument("--lines", type=str, required=True, help="Comma-separated line numbers that already have output_full_pipeline/lineN/01_blueprint.json")
    args = parser.parse_args()
    line_nos = [int(x) for x in args.lines.split(",") if x.strip()]

    OUT_ROOT.mkdir(exist_ok=True)
    summary = []
    for n in line_nos:
        logger.info(f"=== line {n} ===")
        try:
            row = run_one_line(n, OUT_ROOT)
            row["error"] = None
        except Exception as e:
            logger.exception(f"[line {n}] failed")
            row = {"line_id": n, "error": str(e)}
        summary.append(row)

    (OUT_ROOT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Done. Summary -> {OUT_ROOT / 'run_summary.json'}")


if __name__ == "__main__":
    main()
