#!/usr/bin/env python3
"""
scripts/check_tendoo_v3_mask_overflow.py

Kiểm tra thật (không đoán): với mọi case đã render sẵn trong output_tendoo_v3/
(bao gồm cả 2 thư mục con prompts_1_19/ và prompts_21_43/), đo bounding box THẬT
của từng phần tử chữ qua chính Chromium đã render ra poster.html (không đo lại
bằng PIL, không đoán offline), rồi so với mask hình học mà tendoo_v3.mask_engine
khai báo cho template đó ở đúng width/height. Báo cáo phần tử nào rơi vào vùng
"scene" (mask thấp) thay vì vùng "corridor" (mask cao) đã được thiết kế riêng cho
chữ, và phần tử nào tràn hẳn ra ngoài khung canvas hoặc tự tràn trong hộp của nó
(scrollWidth/Height > clientWidth/Height, dấu hiệu ellipsis/cắt chữ).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.mask_engine import generate_template_mask

OUTPUT_DIR = PROJECT_ROOT / "output_tendoo_v3"

MANIFESTS = [
    PROJECT_ROOT / "tests" / "test_cases_tendoo_v3.json",
    PROJECT_ROOT / "tests" / "test_prompts_1_19.json",
    PROJECT_ROOT / "tests" / "test_prompts_21_43.json",
]

# Ngưỡng phân loại mức độ tràn (mean mask value trong bbox: 0=nằm hết trong vùng
# "scene", 1=nằm hết trong vùng "corridor" đã dọn cho chữ).
SEVERE_MEAN_THRESHOLD = 0.5   # >=50% diện tích bbox nằm ngoài corridor -> nghiêm trọng
MILD_MEAN_THRESHOLD = 0.75    # 50-75% trong corridor -> ở rìa (do Gaussian blur), đáng chú ý
CANVAS_EPS = 1.0              # dung sai px khi so bbox với biên canvas


def load_case_index() -> Dict[str, Dict[str, Any]]:
    """id -> {template, width, height, dir} gộp từ cả 3 manifest, khớp với thư mục
    output tương ứng (case_* nằm phẳng, prompt_* nằm trong prompts_1_19/prompts_21_43)."""
    index: Dict[str, Dict[str, Any]] = {}
    for manifest_path in MANIFESTS:
        if not manifest_path.exists():
            continue
        cases = json.loads(manifest_path.read_text(encoding="utf-8"))
        for c in cases:
            cid = c["id"]
            template = c["plan"]["template"]
            width, height = c["width"], c["height"]
            for candidate_dir in (OUTPUT_DIR / cid, OUTPUT_DIR / "prompts_1_19" / cid, OUTPUT_DIR / "prompts_21_43" / cid):
                if (candidate_dir / "poster.html").exists():
                    index[cid] = {"template": template, "width": width, "height": height, "dir": candidate_dir}
                    break
    return index


def measure_text_elements(page, width: int, height: int) -> List[Dict[str, Any]]:
    """Đo bounding box THẬT của mọi phần tử lá (không có con) có chữ hiển thị,
    cộng với cờ tự-tràn-trong-hộp (scrollWidth/Height > clientWidth/Height)."""
    js = r"""
    () => {
      const out = [];
      const all = document.querySelectorAll('body *');
      for (const el of all) {
        if (el.children.length > 0) continue;
        const text = (el.textContent || '').trim();
        if (!text) continue;
        const cs = getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity || '1') === 0) continue;
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) continue;
        out.push({
          tag: el.tagName,
          cls: (el.className && el.className.toString) ? el.className.toString() : '',
          text: text.slice(0, 60),
          x1: r.left, y1: r.top, x2: r.right, y2: r.bottom,
          selfOverflowX: el.scrollWidth > el.clientWidth + 1,
          selfOverflowY: el.scrollHeight > el.clientHeight + 1,
        });
      }
      return out;
    }
    """
    return page.evaluate(js)


def classify_mask_overlap(mask: np.ndarray, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, bool]:
    """Trả về (mean_mask_value_trong_bbox, co_tran_ngoai_canvas)."""
    canvas_overflow = x1 < -CANVAS_EPS or y1 < -CANVAS_EPS or x2 > width + CANVAS_EPS or y2 > height + CANVAS_EPS
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, canvas_overflow
    region = mask[iy1:iy2, ix1:ix2]
    return float(region.mean()) if region.size else 0.0, canvas_overflow


def main() -> None:
    index = load_case_index()
    print(f"Tổng số case tìm thấy poster.html đã render: {len(index)}\n")

    findings_by_template: Dict[str, List[str]] = {}
    total_severe = 0
    total_mild = 0
    total_canvas = 0
    total_self = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = browser.new_page()

        for cid, meta in sorted(index.items()):
            template = meta["template"]
            width, height = meta["width"], meta["height"]
            html_path = meta["dir"] / "poster.html"

            page.goto(html_path.as_uri())
            try:
                page.wait_for_function("window.__tendooDone === true", timeout=3000)
            except Exception:
                page.wait_for_timeout(400)

            elements = measure_text_elements(page, width, height)

            has_mask = TEMPLATE_CATALOG.get(template, {}).get("has_mask", False)
            mask = generate_template_mask(template, width, height) if has_mask else None

            case_lines = []
            for el in elements:
                x1, y1, x2, y2 = el["x1"], el["y1"], el["x2"], el["y2"]
                tag_desc = f'<{el["tag"].lower()} class="{el["cls"]}"> "{el["text"]}"'

                if el["selfOverflowX"] or el["selfOverflowY"]:
                    total_self += 1
                    case_lines.append(f'  [TỰ TRÀN HỘP] {tag_desc} (scrollW/H > clientW/H -- chữ bị cắt/ellipsis trong chính khối của nó)')

                if mask is not None:
                    mean_val, canvas_overflow = classify_mask_overlap(mask, x1, y1, x2, y2, width, height)
                    if canvas_overflow:
                        total_canvas += 1
                        case_lines.append(f'  [TRÀN NGOÀI CANVAS] {tag_desc} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) canvas={width}x{height}')
                    elif mean_val < SEVERE_MEAN_THRESHOLD:
                        total_severe += 1
                        case_lines.append(f'  [NGHIÊM TRỌNG] {tag_desc} -- {(1-mean_val)*100:.0f}% diện tích nằm ngoài vùng mask (mean_mask={mean_val:.2f})')
                    elif mean_val < MILD_MEAN_THRESHOLD:
                        total_mild += 1
                        case_lines.append(f'  [RÌA MASK] {tag_desc} -- mean_mask={mean_val:.2f} (một phần nằm ở dải blur biên)')

            if case_lines:
                findings_by_template.setdefault(template, []).append(f"{cid} ({width}x{height}):\n" + "\n".join(case_lines))

        browser.close()

    print("=" * 90)
    print(f"TỔNG KẾT: {total_canvas} tràn ngoài canvas | {total_severe} nghiêm trọng (đè lên vùng scene) | "
          f"{total_mild} ở rìa mask | {total_self} tự tràn trong hộp riêng\n")

    if not findings_by_template:
        print("Không phát hiện text nào tràn quá mask ở bất kỳ template nào trong bộ case hiện có.")
        return

    print("CHI TIẾT THEO TEMPLATE:")
    for template, cases in findings_by_template.items():
        print(f"\n### Template: {template} ({len(cases)} case bị) ###")
        for c in cases:
            print(c)
            print("-" * 60)


if __name__ == "__main__":
    main()
