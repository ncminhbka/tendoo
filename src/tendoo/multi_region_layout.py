"""
==================================================================================================
TENDOO AI -- MULTI-REGION VLM LAYOUT ASSIGNMENT (SPIKE, NOT WIRED INTO PRODUCTION)
==================================================================================================
Module: src/tendoo/multi_region_layout.py
Purpose: evaluation spike for "does a VLM-driven multi-region layout beat the current fixed-CSS-
template + single-safe_rect pipeline?" (see memory css-hero-title-overlay-direction.md, 2026-09-06
update, and the plan this implements: content-aware layout generation, layer C).

Pipeline this module implements, GIVEN a real background image + real blueprint + real detected
forbidden boxes (all produced elsewhere -- this module does none of that itself):

  1. `build_content_manifest`   -- blueprint dict -> list of generic content blocks
                                   (+ 1 optional corner-reserved block, handled by a fixed rule).
  2. `build_region_candidates`  -- wraps `find_top_empty_rects` (layout_geometry.py, already
                                   built/verified) to get K DISTINCT empty regions across the
                                   WHOLE canvas (not one fixed anchored band), each tagged with
                                   quadrant/aspect-ratio/local luminance.
  3. `call_vlm_planner`         -- ONE real gpt-4o-mini (vision) call: given the background image
                                   + the manifest + the region candidates, choose exactly one
                                   region_id per block (never freeform coordinates -- avoids the
                                   known "VLM numeric-coordinate blindness" failure mode).
  4. `resolve_assignments`      -- rule-based validator (NOT a learned scorer -- explicitly out of
                                   scope for this spike): rejects a structurally invalid VLM
                                   response wholesale and falls back to a deterministic
                                   priority-vs-area assignment, so this pipeline never crashes and
                                   never silently drops a block. Also logs (does not auto-fix)
                                   possible content-overflow warnings.
  5. `render_experiment_poster` -- builds ONE full HTML page from scratch (bypasses
                                   PosterTemplateEngine.generate_html entirely) and renders it via
                                   the existing `PosterRenderer.render`. Deliberately reuses
                                   typography_engine.py's existing, already-tuned building blocks
                                   (HERO_STYLE_CSS, _derive_theme_palette, _resolve_font,
                                   _modular_ratio/_scaled_clamp_str, _bg_image_css) instead of
                                   reinventing them, so a visual comparison against the baseline
                                   isn't confounded by "the new version has worse CSS".
  6. `render_debug_overlay`     -- draws every candidate region as a labeled outline on top of the
                                   real background, for understanding what the VLM was offered/
                                   chose (debugging aid, not part of the actual comparison).

Explicitly OUT of scope for this spike (see the approved plan): no learned layout-quality scorer,
no changes to `run_full_pipeline.py`/`typography_engine.py`/`layout_geometry.py`, no redesign of
feedback/before_after (already a deliberately compact single corner-card), no automatic render-
and-measure retry loop for content-fit (the fit check below is observational logging only).
==================================================================================================
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from tendoo.layout_geometry import BBox, EmptyRect, find_top_empty_rects
from tendoo.typography_engine import PosterBackgroundAnalyzer, PosterRenderer, PosterTemplateEngine, _bg_image_css

# ==================================================================================================
# Reserved corner (logo / QR / contact stand-in) -- handled by a FIXED RULE, never delegated to the
# VLM. Matches the reference posters (LPBank, Orange drink) always keeping one small corner for a
# QR/brand mark untouched by whatever else is happening in the frame.
# ==================================================================================================
RESERVED_CORNER_FRAC = {"left": 0.76, "top": 0.86, "right": 0.98, "bottom": 0.975}


def _reserved_corner_box(canvas_w: float, canvas_h: float) -> BBox:
    f = RESERVED_CORNER_FRAC
    return (canvas_w * f["left"], canvas_h * f["top"], canvas_w * f["right"], canvas_h * f["bottom"])


# ==================================================================================================
# 1. Content manifest -- ONLY the 3 categories in scope for this spike (product_ad, recruitment,
#    menu -- see plan's "ngoài phạm vi" section for why feedback/before_after are excluded).
#    Field names read from `blueprint`/`template_brief` are the REAL ones confirmed against
#    typography_engine.py/test_hero_selector.py -- no invented field names.
# ==================================================================================================

def _manifest_product_ad(brief: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    blocks = [
        {"id": "title", "type": "headline", "priority": "primary",
         "text": brief.get("title_text", "TIÊU ĐỀ SẢN PHẨM"), "position_hint": ["top-center", "top-left", "top-right"]},
        {"id": "subtitle", "type": "subhead", "priority": "secondary",
         "text": brief.get("subtitle_text", "Dòng mô tả phụ"), "position_hint": ["middle-center", "bottom-center"]},
    ]
    return blocks, None  # product_ad's schema has no logo/QR/contact field -- nothing to reserve.


def _manifest_recruitment(brief: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    company = brief.get("company", "TENDOO AI RESEARCH LAB")
    deadline = brief.get("deadline", "")
    pos_label = brief.get("pos_label", "WE ARE HIRING")
    headline_text = f"{company}\n{pos_label}" + (f"\nHạn nộp: {deadline}" if deadline else "")
    blocks = [
        {"id": "headline", "type": "headline", "priority": "primary", "text": headline_text,
         "position_hint": ["top-center", "top-left"]},
        {"id": "salary_badge", "type": "badge", "priority": "secondary", "text": brief.get("salary", ""),
         "position_hint": ["top-right", "middle-right"]},
        {"id": "requirements", "type": "list", "priority": "secondary", "label": "YÊU CẦU",
         "text": list(brief.get("requirements") or []), "position_hint": ["middle-left", "middle-center"]},
        {"id": "benefits", "type": "list", "priority": "secondary", "label": "QUYỀN LỢI",
         "text": list(brief.get("benefits") or []), "position_hint": ["middle-right", "middle-center"]},
        {"id": "cta", "type": "cta", "priority": "primary", "text": brief.get("cta_text", "ỨNG TUYỂN NGAY"),
         "position_hint": ["bottom-center", "bottom-left"]},
    ]
    corner = {"text": f"{brief.get('contact_line1', '')}\n{brief.get('contact_email', '')}".strip()}
    return blocks, corner


def _manifest_menu(brief: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    categories = brief.get("categories") or []
    blocks = [
        {"id": "headline", "type": "headline", "priority": "primary", "text": brief.get("sub_brand", "MENU"),
         "position_hint": ["top-center", "top-left"]},
        {"id": "tagline", "type": "subhead", "priority": "secondary", "text": brief.get("tagline", ""),
         "position_hint": ["top-center"]},
    ]
    for i, cat in enumerate(categories):
        items = cat.get("items") or []
        lines = [f"{it.get('name', '')} — {it.get('price', '')}" for it in items]
        blocks.append({
            "id": f"category_{i}", "type": "list", "priority": "secondary",
            "label": cat.get("title", f"Mục {i + 1}"), "text": lines,
            "position_hint": ["middle-left", "middle-center", "middle-right", "bottom-left", "bottom-right"],
        })
    corner = {"text": f"{brief.get('footer_note', '')}\n{brief.get('hotline', '')}".strip()}
    return blocks, corner


_MANIFEST_BUILDERS = {
    "product_ad": _manifest_product_ad,
    "recruitment": _manifest_recruitment,
    "menu": _manifest_menu,
}


def build_content_manifest(category: str, brief: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Returns (flexible_blocks, corner_block_or_None). Raises KeyError for any category outside
    this spike's scope (product_ad/recruitment/menu) -- deliberate, not a silent generic fallback,
    since a manifest for an unsupported category would just be guessing field names."""
    try:
        builder = _MANIFEST_BUILDERS[category]
    except KeyError:
        raise KeyError(f"multi_region_layout spike only supports {list(_MANIFEST_BUILDERS)}, got '{category}'")
    return builder(brief)


# ==================================================================================================
# 2. Region candidates -- generalizes Cap do 2's single safe_rect into K distinct regions across
#    the WHOLE canvas, each tagged with quadrant/aspect/luminance for the VLM (and the fallback
#    rule) to reason about.
# ==================================================================================================

def build_region_candidates(
    canvas_w: float, canvas_h: float, forbidden_boxes: Sequence[BBox], image_path: Path, k: int = 8,
) -> Tuple[List[Dict[str, Any]], BBox]:
    """Returns (region_candidates, reserved_corner_box). Each region dict has: id, the same
    left/top/right/bottom/width/height _pct keys `EmptyRect.as_css_percent()` already produces
    (so it drop-in matches what `_zone_css_from_rect`-style CSS building already expects),
    aspect_ratio, quadrant, is_dark, mean_luminance. Reuses
    `PosterBackgroundAnalyzer._build_zone_metrics` for luminance (same measurement the production
    pipeline already trusts) instead of re-deriving it -- called per-region instead of the fixed
    3-horizontal-band split `PosterBackgroundAnalyzer.analyze()` itself uses, which is a strictly
    more accurate reading for an arbitrary rect (see memory's "#7" finding this incidentally fixes
    for this spike's own renderer, though `analyze()` itself is untouched)."""
    corner_box = _reserved_corner_box(canvas_w, canvas_h)
    forbidden_with_corner = list(forbidden_boxes) + [corner_box]

    region = (0.0, 0.0, canvas_w, canvas_h)
    rects = find_top_empty_rects(
        region, forbidden_with_corner, k=k,
        min_width=canvas_w * 0.12, min_height=canvas_h * 0.05,
    )

    img_np = np.array(Image.open(image_path).convert("RGB"))
    candidates: List[Dict[str, Any]] = []
    for i, rect in enumerate(rects):
        pct = rect.as_css_percent(canvas_w, canvas_h)
        bbox_ratio = (rect.y1 / canvas_h, rect.x1 / canvas_w, rect.y2 / canvas_h, rect.x2 / canvas_w)
        zone = PosterBackgroundAnalyzer._build_zone_metrics(f"region_{i}", bbox_ratio, img_np)
        candidates.append({
            "id": f"r{i}", **pct,
            "aspect_ratio": round(rect.aspect_ratio, 2),
            "quadrant": rect.quadrant(canvas_w, canvas_h),
            "is_dark": zone.is_dark,
            "mean_luminance": zone.mean_luminance,
        })
    return candidates, corner_box


# ==================================================================================================
# 3. VLM planner -- ONE real vision call. Region choice is constrained to the given candidate IDs;
#    the model never outputs raw coordinates.
# ==================================================================================================

_PLANNER_SYSTEM_PROMPT = """Bạn là một AI thiết kế bố cục (layout designer) cho poster quảng cáo.

Bạn nhận được: (1) ảnh nền THẬT của poster, (2) danh sách các "vùng trống" (region) đã tính sẵn
bằng hình học -- toạ độ % chính xác, đảm bảo KHÔNG có vật thể/khuôn mặt nào chồng lên, (3) danh
sách các khối nội dung (content block) cần đặt vào poster.

NHIỆM VỤ DUY NHẤT: với MỖI content block, chọn ĐÚNG MỘT region_id có trong danh sách đã cho.
TUYỆT ĐỐI KHÔNG tự bịa toạ độ mới, KHÔNG tự nghĩ ra region_id không có trong danh sách. Một
region chỉ được gán cho tối đa 1 block.

Nguyên tắc khi chọn:
- Block có priority="primary" nên vào vùng dễ nhìn nhất (diện tích lớn, gần phía trên hoặc giữa
  khung hình).
- Block type="list" cần vùng đủ RỘNG và đủ CAO cho số dòng nội dung của nó.
- CỐ GẮNG rải các block ra NHIỀU vùng khác nhau trên khung hình thay vì dồn hết vào 1 vùng lớn
  nhất -- giống cách nhà thiết kế thật đặt tiêu đề lên khoảng trời trống, đặt danh sách vào
  khoảng trống cạnh sản phẩm, đặt badge vào 1 góc riêng. KHÔNG dồn mọi thứ vào 1 vùng duy nhất
  nếu có nhiều vùng trống tách biệt khả dụng.
- Nếu region có is_dark=true, chữ trên đó nên sáng màu -> gợi ý emphasis phù hợp; ngược lại nếu
  is_dark=false.

Trả lời DUY NHẤT bằng JSON đúng schema sau, không thêm chữ nào khác ngoài JSON:
{"assignments": [{"block_id": "...", "region_id": "...", "emphasis": "bold"|"normal"|"subtle"}],
 "reasoning": "1-2 câu ngắn giải thích lựa chọn"}
"""


def call_vlm_planner(
    client, model: str, category: str, canvas_w: int, canvas_h: int,
    blocks: List[Dict[str, Any]], regions: List[Dict[str, Any]], image_path: Path,
) -> Dict[str, Any]:
    """ONE real chat.completions call (vision, JSON mode). Raises on a hard API/transport error --
    caller (the orchestrator script) is responsible for catching and treating that as
    'VLM unavailable, use the fallback' the same way a structurally-invalid JSON response is
    handled by `resolve_assignments`."""
    import base64
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    payload = {
        "category": category, "canvas": {"width": canvas_w, "height": canvas_h},
        "content_blocks": [
            {k: v for k, v in b.items() if k != "text"} | {"text_preview": _text_preview(b.get("text"))}
            for b in blocks
        ],
        "region_candidates": regions,
    }
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _text_preview(text: Any, max_chars: int = 160) -> str:
    """Compact preview of a block's `text` field for the VLM prompt payload -- a `list` block's
    text is a list of lines; the model only needs to gauge roughly how much content it is, not
    read every line verbatim (keeps the prompt payload small)."""
    if isinstance(text, list):
        joined = " | ".join(text)
        return (joined[:max_chars] + "...") if len(joined) > max_chars else (joined or "(rỗng)")
    s = str(text or "")
    return (s[:max_chars] + "...") if len(s) > max_chars else s


# ==================================================================================================
# 4. Validator + deterministic fallback -- rule-based, explicitly NOT a learned scorer.
# ==================================================================================================

_PRIORITY_ORDER = {"primary": 0, "secondary": 1, "tertiary": 2}
_EST_LIST_LINE_HEIGHT_PCT = 9.0  # rough, tuned for this renderer's list font size -- see render_experiment_poster


def _fallback_assignment(blocks: List[Dict[str, Any]], regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic: sort blocks by priority (primary first), sort regions by area descending,
    zip. Not the "best possible" layout -- just a safe, crash-proof default so a VLM hiccup
    degrades gracefully instead of dropping content, mirroring this project's established
    'never silently drop content' principle.

    REAL BUG caught by this module's own smoke test (not just asserted correct): a plain
    `zip(blocks, regions)` silently DROPS every block past `len(regions)` when there are fewer
    distinct empty regions than content blocks (a real, common case -- e.g. 5 recruitment blocks
    but only 3 regions found on a simple background) -- exactly the failure mode this whole
    project has repeatedly treated as unacceptable elsewhere. Fixed by cycling through `regions`
    (via modulo) once they run out, so every block gets SOME region even if it means 2 blocks
    stack in the same one -- visually degraded (a real trade-off, not swept under the rug), but
    never silently missing."""
    if not regions:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: _PRIORITY_ORDER.get(b.get("priority", "secondary"), 1))
    sorted_regions = sorted(regions, key=lambda r: r["width_pct"] * r["height_pct"], reverse=True)
    return [
        {"block_id": b["id"], "region_id": sorted_regions[i % len(sorted_regions)]["id"], "emphasis": "normal"}
        for i, b in enumerate(sorted_blocks)
    ]


def resolve_assignments(
    blocks: List[Dict[str, Any]], regions: List[Dict[str, Any]], vlm_raw: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Returns {"assignments": [...], "warnings": [str, ...], "fallback_used": bool}.
    `vlm_raw` may be None (VLM call itself failed) -- treated identically to a structurally
    invalid response: hard fallback, never raises."""
    region_by_id = {r["id"]: r for r in regions}
    block_ids = {b["id"] for b in blocks}
    warnings: List[str] = []
    fallback_used = False
    assignments: Optional[List[Dict[str, Any]]] = None

    try:
        if vlm_raw is None:
            raise ValueError("VLM call failed / returned nothing")
        raw_assignments = vlm_raw.get("assignments")
        if not isinstance(raw_assignments, list):
            raise ValueError("'assignments' missing or not a list")
        seen_blocks, seen_regions, cleaned = set(), set(), []
        for a in raw_assignments:
            bid, rid = a.get("block_id"), a.get("region_id")
            if bid not in block_ids:
                raise ValueError(f"unknown block_id {bid!r}")
            if rid not in region_by_id:
                raise ValueError(f"unknown region_id {rid!r}")
            if bid in seen_blocks:
                raise ValueError(f"block {bid!r} assigned twice")
            if rid in seen_regions:
                raise ValueError(f"region {rid!r} used by more than one block")
            seen_blocks.add(bid)
            seen_regions.add(rid)
            cleaned.append({"block_id": bid, "region_id": rid, "emphasis": a.get("emphasis", "normal")})
        if seen_blocks != block_ids:
            raise ValueError(f"missing assignment(s) for {block_ids - seen_blocks}")
        assignments = cleaned
    except Exception as e:
        warnings.append(f"VLM output rejected ({e}); using deterministic fallback assignment")
        fallback_used = True
        assignments = _fallback_assignment(blocks, regions)
        region_usage: Dict[str, List[str]] = {}
        for a in assignments:
            region_usage.setdefault(a["region_id"], []).append(a["block_id"])
        for rid, bids in region_usage.items():
            if len(bids) > 1:
                warnings.append(
                    f"fallback ran out of distinct regions -- blocks {bids} stacked into the same "
                    f"region '{rid}' (fewer empty regions than content blocks); will visually overlap"
                )

    block_by_id = {b["id"]: b for b in blocks}
    for a in assignments:
        block = block_by_id[a["block_id"]]
        region = region_by_id[a["region_id"]]
        if block["type"] == "list":
            n_lines = len(block.get("text") or [])
            needed_pct = n_lines * _EST_LIST_LINE_HEIGHT_PCT
            if needed_pct > region["height_pct"] * 1.15:
                warnings.append(
                    f"possible overflow: block '{a['block_id']}' (~{n_lines} lines) into region "
                    f"'{a['region_id']}' ({region['height_pct']:.1f}% canvas height) -- observational "
                    f"only, not auto-repaired (see plan's scope cuts)"
                )
    return {"assignments": assignments, "warnings": warnings, "fallback_used": fallback_used}


# ==================================================================================================
# 5. Renderer -- builds one full HTML page from scratch, reusing typography_engine.py's existing
#    style building blocks so the visual comparison isn't confounded by worse CSS.
# ==================================================================================================

_PAGE_TPL = Template("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:100vw; height:100vh; font-family:'Montserrat', sans-serif; }
  .poster { position:relative; width:${w}px; height:${h}px; overflow:hidden; ${bg_css} isolation:isolate; }
  .block { position:absolute; }
  .block-headline { font-weight:900; text-transform:uppercase; line-height:1.12; letter-spacing:0.5px;
    font-size:clamp(22px, 5.2vw, 58px); white-space:pre-line; padding:1% 3%; ${headline_css} }
  .block-subhead { font-weight:600; line-height:1.35; font-size:clamp(13px, 2.1vw, 24px); white-space:pre-line;
    padding:0.5% 3%; ${subhead_css} }
  .block-list { background:${glass_bg}; backdrop-filter:blur(18px); border:1px solid ${glass_border};
    border-radius:16px; padding:5% 6%; overflow:hidden; }
  .block-list h4 { font-size:clamp(11px, 1.5vw, 18px); text-transform:uppercase; letter-spacing:0.6px;
    margin-bottom:0.6em; color:${accent}; font-weight:800; }
  .block-list ul { list-style:none; }
  .block-list li { font-size:clamp(10px, 1.3vw, 16px); line-height:1.5; margin-bottom:0.35em; color:${body_text_color}; }
  .block-list li::before { content:'✓  '; opacity:0.75; }
  .block-cta { background:linear-gradient(135deg, var(--accent), var(--accent-dark)); color:#fff;
    text-align:center; font-weight:800; border-radius:999px; padding:4% 6%;
    box-shadow:0 8px 24px rgba(0,0,0,0.28); font-size:clamp(12px, 1.7vw, 20px); text-transform:uppercase; }
  .block-badge { background:${glass_bg}; backdrop-filter:blur(14px); border:1px solid ${glass_border};
    border-radius:999px; padding:3% 5%; font-weight:700; text-align:center; color:${body_text_color};
    font-size:clamp(11px, 1.4vw, 17px); white-space:pre-line; }
  .block-corner { background:rgba(15,23,42,0.55); backdrop-filter:blur(12px); border-radius:12px;
    padding:2.5% 4%; font-size:clamp(9px, 1.0vw, 13px); color:#fff; text-align:right; white-space:pre-line; }
  .poster { --accent:${accent_var}; --accent-dark:${accent_dark_var}; }
</style></head>
<body><div class="poster">
${blocks_html}
</div></body></html>""")

_BLOCK_TPL = Template('<div class="block block-${css_type}" style="${style}">${inner}</div>')


def _region_style(region: Dict[str, Any]) -> str:
    return f"top:{region['top_pct']}%; left:{region['left_pct']}%; right:{region['right_pct']}%; bottom:{region['bottom_pct']}%;"


def _headline_style_css(is_dark: bool, style_hint: Optional[str]) -> str:
    """Mirrors `PosterTemplateEngine._auto_pick_style`'s family-pick + substring-hint logic, but
    applied to THIS region's own measured luminance rather than a fixed header/center/footer zone
    (that method's API is scoped to the 3-band abstraction, not an arbitrary candidate rect) --
    reuses the exact same `HERO_STYLE_CSS` registry, not a re-invented style set."""
    family = PosterTemplateEngine._LIGHT_BG_STYLE_ORDER if not is_dark else PosterTemplateEngine._DARK_BG_STYLE_ORDER
    key = family[0]
    if style_hint:
        hint = style_hint.lower()
        for candidate in family:
            if hint in candidate:
                key = candidate
                break
    return PosterTemplateEngine.HERO_STYLE_CSS[key]


def render_experiment_poster(
    category: str, canvas_w: int, canvas_h: int, image_path: Path,
    blocks: List[Dict[str, Any]], corner_block: Optional[Dict[str, Any]],
    regions: List[Dict[str, Any]], corner_box: BBox, resolved: Dict[str, Any],
    style_hint: Optional[str], out_path: Path,
) -> Path:
    region_by_id = {r["id"]: r for r in regions}
    block_by_id = {b["id"]: b for b in blocks}

    # Overall "is the poster mostly dark" reading drives the generic list/badge/cta text color --
    # reuses the same PosterBackgroundAnalyzer.analyze() the production pipeline already trusts,
    # NOT a re-derivation.
    analysis = PosterBackgroundAnalyzer.analyze(image_path)
    palette_defaults = {
        "product_ad": {"accent": "#38BDF8", "accent_dark": "#0284C7"},
        "recruitment": PosterTemplateEngine._RECRUITMENT_DEFAULT_PALETTE,
        "menu": PosterTemplateEngine._MENU_DEFAULT_PALETTE,
    }[category]
    body_text_color = "#F8FAFC" if analysis.overall_is_dark else "#0F172A"
    glass_bg = "rgba(255,255,255,0.14)" if analysis.overall_is_dark else "rgba(15,23,42,0.10)"
    glass_border = "rgba(255,255,255,0.28)" if analysis.overall_is_dark else "rgba(15,23,42,0.18)"

    blocks_html_parts: List[str] = []
    for a in resolved["assignments"]:
        block = block_by_id[a["block_id"]]
        region = region_by_id[a["region_id"]]
        style = _region_style(region)
        if block["type"] == "headline":
            headline_css = _headline_style_css(region["is_dark"], style_hint)
            inner = block["text"]
            blocks_html_parts.append(
                f'<div class="block block-headline" style="{style} {headline_css}">{inner}</div>'
            )
        elif block["type"] == "subhead":
            color = "#FFFFFF" if region["is_dark"] else "#0F172A"
            shadow = "0 2px 10px rgba(0,0,0,0.5)" if region["is_dark"] else "0 2px 8px rgba(255,255,255,0.6)"
            blocks_html_parts.append(
                f'<div class="block block-subhead" style="{style} color:{color}; text-shadow:{shadow};">{block["text"]}</div>'
            )
        elif block["type"] == "list":
            items_html = "".join(f"<li>{line}</li>" for line in (block.get("text") or []))
            label = block.get("label", "")
            blocks_html_parts.append(
                f'<div class="block block-list" style="{style}"><h4>{label}</h4><ul>{items_html}</ul></div>'
            )
        elif block["type"] == "cta":
            blocks_html_parts.append(f'<div class="block block-cta" style="{style}">{block["text"]}</div>')
        elif block["type"] == "badge":
            blocks_html_parts.append(f'<div class="block block-badge" style="{style}">{block["text"]}</div>')

    if corner_block and corner_block.get("text"):
        cx1, cy1, cx2, cy2 = corner_box
        corner_style = (
            f"left:{100 * cx1 / canvas_w:.2f}%; top:{100 * cy1 / canvas_h:.2f}%; "
            f"right:{100 * (canvas_w - cx2) / canvas_w:.2f}%; bottom:{100 * (canvas_h - cy2) / canvas_h:.2f}%;"
        )
        blocks_html_parts.append(f'<div class="block block-corner" style="{corner_style}">{corner_block["text"]}</div>')

    html = _PAGE_TPL.substitute(
        w=canvas_w, h=canvas_h, bg_css=_bg_image_css(str(image_path)),
        headline_css="", subhead_css="",
        glass_bg=glass_bg, glass_border=glass_border, body_text_color=body_text_color,
        accent=palette_defaults["accent"], accent_var=palette_defaults["accent"],
        accent_dark_var=palette_defaults.get("accent_dark", palette_defaults["accent"]),
        blocks_html="\n".join(blocks_html_parts),
    )
    PosterRenderer.render(html_content=html, output_image_path=out_path, width=canvas_w, height=canvas_h)
    return out_path


# ==================================================================================================
# 6. Debug overlay -- draws every candidate region (+ the reserved corner) as a labeled outline on
#    the real background, purely for understanding what the VLM was offered/chose. Not part of the
#    actual baseline-vs-experiment visual comparison.
# ==================================================================================================

_OVERLAY_COLORS = ["#00E5FF", "#FF3DAE", "#FFD166", "#7CFF6B", "#B98CFF", "#FF6B35", "#4DD0E1", "#F06292"]


def render_debug_overlay(image_path: Path, regions: List[Dict[str, Any]], corner_box: BBox, out_path: Path) -> Path:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i, r in enumerate(regions):
        x1 = r["left_pct"] / 100.0 * w
        y1 = r["top_pct"] / 100.0 * h
        x2 = w - r["right_pct"] / 100.0 * w
        y2 = h - r["bottom_pct"] / 100.0 * h
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 6, y1 + 4), r["id"], fill=color)
    cx1, cy1, cx2, cy2 = corner_box
    draw.rectangle([cx1, cy1, cx2, cy2], outline="#FFFFFF", width=3)
    draw.text((cx1 + 6, cy1 + 4), "corner(reserved)", fill="#FFFFFF")
    img.save(out_path)
    return out_path
