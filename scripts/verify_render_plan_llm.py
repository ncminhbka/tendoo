#!/usr/bin/env python3
"""
scripts/verify_render_plan_llm.py

Phase C real-GPU verification script -- run this on the actual server (this dev
environment has no GPU, cannot run it here). Exercises the LLM render-plan sidecar
(src/tendoo/llm_render_plan_server.py) against a battery of real test cases, with NO
manual UI interaction needed. Two modes:

  --mode llm-only (default, fast): POSTs straight to the sidecar's /api/render-plan
    for each case and inspects the raw JSON -- no diffusion, no demo_server.py needed,
    good for quickly iterating on the system prompt's wording.
  --mode full (slower, needs demo_server.py running too): POSTs to demo_server.py's
    /api/generate for each case (which internally calls the sidecar over HTTP exactly
    like real traffic would), saves the final poster PNG + rendered HTML, and runs
    structural sanity checks (resolved_layout, duplicate-content counts, title
    presence/absence) against the ones with a known-correct expected outcome.

Test cases, three groups:
  1. prompt_test.txt lines 1-19 (10 entries) -- pure positional/freeform prompts, form
     left entirely blank, image_description = the full line (scene + positioned text
     requests inline), aspect ratio parsed from the line's own "--ar W:H" suffix.
  2. prompt_test.txt lines 21-43 (12 entries) -- "whole brief typed into the free
     prompt instead of the form" cases, tested BLANK-form (tests blank-field
     extraction) plus 2 hand-picked DUPLICATE variants (same brief, but this time the
     matching form fields are ALSO pre-filled -- tests the dedup rule: the real form
     value must win and render exactly once, never escalating to freeform just because
     of the restatement).
  3. Hand-written synthetic cases for the title precedence rule specifically (skip /
     override / auto-generate) -- these exact scenarios aren't naturally present in
     prompt_test.txt, so they're authored directly here with a concretely known-correct
     expected outcome, checkable by the script itself (not just human visual review).

For groups 1 and 2 (open-ended real content), there is no single "correct" answer to
assert against -- this script just runs them, saves everything, and prints a short
summary; JUDGING WHETHER THE OUTPUT IS ACTUALLY GOOD (sensible title, sensible
positions, no weirdness) still needs a human looking at the saved PNGs/JSON. Group 3
(and the 2 duplicate variants) DO have a known-correct outcome and get real
PASS/FAIL assertions.

Usage:
  1. Start the render-plan sidecar (loads real Qwen3-4B-FP8):
       python -m tendoo.llm_render_plan_server --port 7861
  2. (--mode full only) Start demo_server.py normally, real GPU or --mock:
       python -m tendoo.demo_server            # real inference
       python -m tendoo.demo_server --mock      # fast HTML/mask sanity pass first
  3. python scripts/verify_render_plan_llm.py --mode llm-only
     python scripts/verify_render_plan_llm.py --mode full --demo-url http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import requests  # not a src/tendoo/ production dependency -- fine here, scripts/ only

from tendoo.layouts.style_matcher import CATEGORY_FIELD_SLOTS

PROMPT_TEST_PATH = PROJECT_ROOT / "prompt_test.txt"


# ==================================================================================
# 1. prompt_test.txt parsing
# ==================================================================================

def load_prompt_test_paragraphs(path: Path) -> List[str]:
    content = path.read_text(encoding="utf-8")
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def extract_aspect_ratio(text: str) -> str:
    m = re.search(r"--ar\s+(\d+:\d+)", text)
    return m.group(1) if m else "1:1"


def strip_aspect_ratio(text: str) -> str:
    return re.sub(r"--ar\s+\d+:\d+", "", text).strip()


def guess_category(text: str) -> str:
    lower = text.lower()
    if "khai trương" in lower or "grand opening" in lower:
        return "opening"
    if "feedback" in lower or "đánh giá" in lower:
        return "feedback"
    if "tuyển dụng" in lower:
        return "recruitment"
    if "hướng dẫn" in lower or "quy trình" in lower:
        return "guide"
    if "giới thiệu sản phẩm" in lower or "tính năng" in lower:
        return "product_intro"
    return "promo"


# ==================================================================================
# 2. Test case model + builders
# ==================================================================================

def extract_visible_text(html: str) -> str:
    """Strips HTML tags and collapses whitespace, joining what were separate elements
    with a single space. balance_vietnamese_headline() wraps long titles across
    multiple <span class="headline-line"> elements at WORD boundaries (never mid-word)
    -- so a raw substring search against the full HTML source can miss a genuinely
    correct render just because of where the wrap landed. Checking against this
    extracted, space-joined text is robust to that."""
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TestCase:
    id: str
    category: str
    form_fields: Dict[str, str]
    image_description: str
    aspect_ratio: str = "1:1"
    note: str = ""
    # Optional: (visible_text) -> list of failure-description strings, empty = pass.
    # Only set for cases with a concretely known-correct expected outcome. Runs
    # against extract_visible_text(html), not the raw HTML.
    checks: Optional[Callable[[str], List[str]]] = None
    # Optional: assert response["resolved_layout"] == this value.
    expected_layout: Optional[str] = None


def build_positional_cases(paragraphs: List[str]) -> List[TestCase]:
    cases = []
    for i, p in enumerate(paragraphs[:10]):
        cases.append(TestCase(
            id=f"positional_{i + 1:02d}",
            category="promo",
            form_fields={},
            image_description=strip_aspect_ratio(p),
            aspect_ratio=extract_aspect_ratio(p),
            note="prompt_test.txt lines 1-19: positional/freeform, form left blank",
        ))
    return cases


def build_whole_brief_blank_cases(paragraphs: List[str]) -> List[TestCase]:
    cases = []
    for i, p in enumerate(paragraphs[10:22]):
        cat = guess_category(p)
        cases.append(TestCase(
            id=f"whole_brief_blank_{i + 1:02d}",
            category=cat,
            form_fields={},
            image_description=p,
            aspect_ratio=extract_aspect_ratio(p) if "--ar" in p else "1:1",
            note=f"prompt_test.txt lines 21-43, form BLANK -- tests blank-field extraction (category guessed: '{cat}')",
        ))
    return cases


def _contains_at_most_once(text: str, needle: str) -> Optional[str]:
    count = text.count(needle)
    if count == 0:
        return f"expected '{needle}' to appear at least once, found 0"
    if count > 1:
        return f"expected '{needle}' to appear exactly once (duplicate-content check), found {count}"
    return None


def build_duplicate_cases() -> List[TestCase]:
    """Hand-picked: content matching a real prompt_test.txt brief, but this time the
    corresponding form fields are ALSO pre-filled -- the exact scenario the product
    owner flagged. Known-correct outcome: the real form value wins, renders exactly
    once, and this alone does not escalate to freeform."""

    def check_promo_dates(text: str) -> List[str]:
        errs = []
        for needle in ["2026-09-01", "2026-09-15"]:
            err = _contains_at_most_once(text, needle)
            if err:
                errs.append(err)
        return errs

    def check_feedback_fields(text: str) -> List[str]:
        errs = []
        for needle in ["Private Coaching Transformation"]:
            err = _contains_at_most_once(text, needle)
            if err:
                errs.append(err)
        return errs

    return [
        TestCase(
            id="duplicate_promo_dates",
            category="promo",
            form_fields={
                "title": "SIÊU SALE CUỐI TUẦN",
                "discount": "GIẢM 30%",
                "date_start": "2026-09-01",
                "date_end": "2026-09-15",
            },
            image_description=(
                "Banner khuyến mại, tone màu cam ấm áp. Áp dụng từ 01/09/2026 đến "
                "15/09/2026, giảm 30% toàn bộ sản phẩm."
            ),
            note="discount + dates filled in form AND restated in the free prompt -- must render once each",
            checks=check_promo_dates,
        ),
        TestCase(
            id="duplicate_feedback_target",
            category="feedback",
            form_fields={
                "feedback_target": "Private Coaching Transformation",
                "feedback_quote": "97% khách hàng hài lòng với kết quả tăng cơ sau 3 tháng",
            },
            image_description=(
                "Tạo ảnh feedback khách hàng cho dịch vụ gym & PT cao cấp. "
                "Tên sản phẩm/dịch vụ nhận feedback: \"Private Coaching Transformation\". "
                "Mô tả ngắn feedback: \"97% khách hàng hài lòng với kết quả tăng cơ sau 3 tháng\". "
                "Ưu đãi đặc biệt: \"Giảm 20% gói PT tháng đầu\". Background phòng gym sang trọng, tone đen đỏ."
            ),
            note="prompt_test.txt line 21's real wording, but feedback_target/feedback_quote ALSO pre-filled",
            checks=check_feedback_fields,
        ),
    ]


def build_title_precedence_cases() -> List[TestCase]:
    def check_title_skipped(text: str) -> List[str]:
        return ["title should have been DROPPED but is present"] if "TIÊU ĐỀ GỐC SẼ BỊ BỎ" in text else []

    def check_title_overridden(text: str) -> List[str]:
        errs = []
        if "ƯU ĐÃI CỰC SỐC" not in text:
            errs.append("expected the prompt-sourced title to appear, not found")
        if "TIÊU ĐỀ FORM CŨ" in text:
            errs.append("form's original title should have been overridden but is still present")
        return errs

    return [
        TestCase(
            id="title_skip_no_hero_mention",
            category="promo",
            form_fields={"title": "TIÊU ĐỀ GỐC SẼ BỊ BỎ", "discount": "GIẢM 50%"},
            image_description=(
                "Thêm dòng chữ 'SỐ LƯỢNG CÓ HẠN' ở góc dưới bên phải, và số điện thoại "
                "'0334842155' ở góc dưới bên trái."
            ),
            note="filled title + positioned request with NO hero mention -> title must be dropped entirely",
            checks=check_title_skipped,
        ),
        TestCase(
            id="title_override_from_prompt",
            category="promo",
            form_fields={"title": "TIÊU ĐỀ FORM CŨ", "discount": "GIẢM 50%"},
            image_description="Đổi tiêu đề chính thành 'ƯU ĐÃI CỰC SỐC HÔM NAY', chữ lớn nổi bật ở giữa.",
            note="filled title + prompt explicitly asks for a different hero text -> prompt's text must win",
            checks=check_title_overridden,
        ),
        TestCase(
            id="title_auto_generate",
            category="product_intro",
            form_fields={
                "product_name": "Tai Nghe Sonic Pro",
                "product_desc": "Chống ồn chủ động, âm thanh Hi-Res",
            },
            image_description=(
                "Tai nghe không dây đặt trên bệ đá cẩm thạch, ánh sáng studio xanh dương "
                "lạnh, phong cách cao cấp."
            ),
            note="title field BLANK, prompt doesn't mention a hero either -> model must generate a fitting title (no automatic check -- eyeball the saved output)",
        ),
    ]


def build_multi_cta_case(paragraphs: List[str]) -> Optional[TestCase]:
    """prompt_test.txt line 43 -- 4 independent CTA lines, more than the fixed
    'opening' layout's slots (opening_date/opening_promo/booking_contact = 3) can
    natively hold -- should escalate to freeform on its own."""
    if len(paragraphs) < 22:
        return None
    line43 = paragraphs[21]
    return TestCase(
        id="multi_cta_overflow_line43",
        category="opening",
        form_fields={},
        image_description=line43,
        aspect_ratio=extract_aspect_ratio(line43),
        note="prompt_test.txt line 43: 4 independent CTA lines -- expect resolved_layout == freeform",
        expected_layout="freeform",
    )


# ==================================================================================
# 3. Execution
# ==================================================================================

def collect_category_fields(case: TestCase) -> Dict[str, str]:
    """Projects case.form_fields onto the full CATEGORY_FIELD_SLOTS[category] set
    (blank string for anything not set) -- exactly what demo_server.py's
    _collect_category_field_values() sends to the sidecar in production."""
    slots = CATEGORY_FIELD_SLOTS.get(case.category, [])
    return {f: case.form_fields.get(f, "") for f in slots}


def call_sidecar(sidecar_url: str, case: TestCase, style_pref: str, timeout: float) -> Dict[str, Any]:
    payload = {
        "category": case.category,
        "title": case.form_fields.get("title", ""),
        "category_fields": collect_category_fields(case),
        "prompt": case.image_description,
        "style_pref": style_pref,
    }
    t0 = time.time()
    resp = requests.post(f"{sidecar_url.rstrip('/')}/api/render-plan", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    data["_latency_s"] = round(time.time() - t0, 2)
    return data


def call_demo_generate(demo_url: str, case: TestCase, timeout: float) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "category": case.category,
        "image_description": case.image_description,
        "aspect_ratio": case.aspect_ratio,
        "enable_qr": False,
    }
    payload.update(case.form_fields)
    resp = requests.post(f"{demo_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_html(demo_url: str, html_url: str) -> str:
    resp = requests.get(f"{demo_url.rstrip('/')}/outputs/{html_url.split('outputs/', 1)[-1]}", timeout=30)
    resp.raise_for_status()
    return resp.text


def run_llm_only(cases: List[TestCase], sidecar_url: str, style_pref: str, timeout: float, out_dir: Path) -> None:
    print(f"\n{'=' * 90}\nMODE: llm-only ({len(cases)} cases against {sidecar_url})\n{'=' * 90}")
    for case in cases:
        try:
            data = call_sidecar(sidecar_url, case, style_pref, timeout)
        except Exception as e:
            print(f"[FAIL] {case.id}: sidecar call raised {e!r}")
            continue

        out_file = out_dir / f"{case.id}.json"
        out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        plan = data.get("plan")
        usable = data.get("usable")
        errors = data.get("errors") or []
        status = "OK" if usable else "UNUSABLE"
        print(f"[{status:8}] {case.id:32} latency={data.get('_latency_s')}s  note={case.note}")
        if plan:
            title = plan.get("title") or {}
            n_blocks = len(plan.get("extra_blocks") or [])
            print(f"           title.action={title.get('action')!r} title.text={title.get('text')!r} extra_blocks={n_blocks}")
        if errors:
            for e in errors:
                print(f"           - {e}")
        print(f"           saved: {out_file}")


def run_full(cases: List[TestCase], demo_url: str, out_dir: Path, timeout: float) -> None:
    print(f"\n{'=' * 90}\nMODE: full ({len(cases)} cases against {demo_url})\n{'=' * 90}")
    n_pass, n_fail, n_unchecked = 0, 0, 0
    for case in cases:
        try:
            data = call_demo_generate(demo_url, case, timeout)
        except Exception as e:
            print(f"[ERROR] {case.id}: /api/generate raised {e!r}")
            n_fail += 1
            continue

        resolved_layout = data.get("resolved_layout")
        case_dir = out_dir / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "response.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        html = ""
        try:
            html = fetch_html(demo_url, data.get("html_url", ""))
            (case_dir / "poster.html").write_text(html, encoding="utf-8")
        except Exception as e:
            print(f"           (warning: could not fetch HTML for manual review: {e})")

        try:
            png_resp = requests.get(f"{demo_url.rstrip('/')}/{data['final_poster_url']}", timeout=30)
            (case_dir / "poster.png").write_bytes(png_resp.content)
        except Exception as e:
            print(f"           (warning: could not fetch PNG for manual review: {e})")

        failures: List[str] = []
        if case.expected_layout is not None and resolved_layout != case.expected_layout:
            failures.append(f"expected resolved_layout='{case.expected_layout}', got '{resolved_layout}'")
        if case.checks is not None and html:
            failures += case.checks(extract_visible_text(html))

        has_any_check = case.expected_layout is not None or case.checks is not None
        if not has_any_check:
            n_unchecked += 1
            print(f"[SAVED   ] {case.id:32} resolved_layout={resolved_layout}  (open-ended content -- review {case_dir}/poster.png manually)")
        elif failures:
            n_fail += 1
            print(f"[FAIL    ] {case.id:32} resolved_layout={resolved_layout}")
            for f_ in failures:
                print(f"           - {f_}")
        else:
            n_pass += 1
            print(f"[PASS    ] {case.id:32} resolved_layout={resolved_layout}")

    print(f"\n{'-' * 90}\n{n_pass} passed, {n_fail} failed, {n_unchecked} saved for manual review (of {len(cases)} total)\n{'-' * 90}")


# ==================================================================================
# main
# ==================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["llm-only", "full"], default="llm-only")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:7861")
    parser.add_argument("--demo-url", default="http://127.0.0.1:7860")
    parser.add_argument("--style-pref", default="auto")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "output_render_plan_verify"))
    parser.add_argument("--filter", default=None, help="Only run cases whose id contains this substring")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds (full mode needs more for real diffusion)")
    parser.add_argument(
        "--groups", default="positional,whole_brief,duplicate,title,multi_cta",
        help="Comma-separated subset of: positional,whole_brief,duplicate,title,multi_cta",
    )
    args = parser.parse_args()

    if not PROMPT_TEST_PATH.exists():
        print(f"ERROR: {PROMPT_TEST_PATH} not found.")
        sys.exit(1)
    paragraphs = load_prompt_test_paragraphs(PROMPT_TEST_PATH)
    if len(paragraphs) < 22:
        print(f"WARNING: expected 22 paragraphs in prompt_test.txt, found {len(paragraphs)} -- some case groups may be short/empty.")

    requested_groups = set(g.strip() for g in args.groups.split(","))
    cases: List[TestCase] = []
    if "positional" in requested_groups:
        cases += build_positional_cases(paragraphs)
    if "whole_brief" in requested_groups:
        cases += build_whole_brief_blank_cases(paragraphs)
    if "duplicate" in requested_groups:
        cases += build_duplicate_cases()
    if "title" in requested_groups:
        cases += build_title_precedence_cases()
    if "multi_cta" in requested_groups:
        mc = build_multi_cta_case(paragraphs)
        if mc:
            cases.append(mc)

    if args.filter:
        cases = [c for c in cases if args.filter in c.id]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("No cases selected (check --groups/--filter/--limit).")
        sys.exit(0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(paragraphs)} prompt_test.txt paragraphs; running {len(cases)} case(s).")

    if args.mode == "llm-only":
        run_llm_only(cases, args.sidecar_url, args.style_pref, args.timeout, out_dir)
    else:
        run_full(cases, args.demo_url, out_dir, args.timeout)

    print(f"\nAll outputs saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
