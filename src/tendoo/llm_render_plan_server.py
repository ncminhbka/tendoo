#!/usr/bin/env python3
"""
src/tendoo/llm_render_plan_server.py

Phase C (LLM Render-Plan Hosting Sidecar) -- v2, minimal-LLM-surface design.
=============================================================================
Standalone FastAPI service, deliberately a SEPARATE OS process from demo_server.py
(started/restarted/scaled independently -- a genuine hosting script, not a method
bolted onto the diffusion process). Turns a filled-in poster form + a user's free-form
"prompt ảnh" into a small render plan: how to resolve the headline/title, which extra
text (if any) the free prompt asks for beyond the form's own fields, and a clean
diffusion scene prompt. See project memory (tendoo-3phase-plan-2026-09-09.md) for the
full design rationale -- summarized here:

THE CORE RULE (decided directly by the product owner, not inferred -- do not
re-litigate without new direction):
  1. Store/contact fields (store_name/phone/address/website/QR) always render --
     unconditional, handled entirely outside this module, unchanged from before Phase C.
  2. Every OTHER filled "Thông tin chung" field (discount, product_name, feedback_quote,
     job_position, guide_steps, ...) -- filled means DRAWN, no negotiation. This model
     is never given authority to drop, edit, or paraphrase that content; its *content*
     never passes through generation, only its *placement* (via extra_blocks[].zone) is
     ever LLM/auto-decided. This is why CATEGORY_FIELD_SLOTS-listed fields sent in the
     request are context for reasoning, not a source of truth this service authors text
     for -- see resolution rule below.
  3. The one field with real creative/visual weight is the title/headline, resolved
     with a 4-branch precedence (this module only proposes an action+text; the actual
     precedence logic, which needs the real GenerateRequest, lives in demo_server.py):
       (a) prompt explicitly names/describes a hero/prominent headline -> use it
           regardless of the title field's own value.
       (b) prompt asks for other explicitly-positioned text without mentioning a hero
           -> title is skipped entirely, even if filled.
       (c) title field filled (and neither (a) nor (b) applies) -> use it verbatim.
       (d) neither -> this model must generate a contextually fitting title.

DUPLICATE/OMISSION PREVENTION (extra_blocks[].field): a block tagged with `field` never
by itself triggers `freeform` -- it only affects that ONE field's content, resolved in
Python (see demo_server.py::run_pipeline_inference): the real already-filled value wins
over anything this model writes (fidelity + de-duplication guarantee -- a form value can
never be silently corrupted/duplicated by generation), while a genuinely blank field
falls back to trusting this model's extraction (fills gaps when the user typed the whole
brief into the free prompt instead of the form, e.g. prompt_test.txt lines 21-41).
`freeform` triggers only when a block has `field: null` (content matching no known slot)
or an explicit `zone` (the user asked for a specific position for that content,
regardless of whether it's also field-tagged) -- computed in demo_server.py, not here.

Model: a plain Qwen3-4B-FP8 checkpoint (the same weights already used as FLUX.2's text
encoder in demo_server.py, but loaded HERE as a genuine AutoModelForCausalLM + .generate()
text-generation model -- Qwen3Embedder, src/flux2/text_encoder.py, is embedding-only and
never implements .generate()). Deliberately NOT vLLM/SGLang/TGI: no serving-library
dependency exists anywhere in this repo yet and this is one render-plan generation per
poster request, not high-concurrency chat traffic -- reconsider only if real throughput
demands it. Escalate to Qwen3-8B-FP8 (swap --model) if 4B proves unreliable at
structured JSON in practice; both are untested assumptions, not validated choices --
needs real GPU verification this environment cannot perform.

Contract with demo_server.py (see try_fetch_render_plan() there): this endpoint may
return a plan that's partially or wholly invalid (unparseable JSON, unknown zone/field
names, near-duplicate content) -- validate_render_plan() below never raises for bad
model output, it filters/drops/corrects what it can and reports the rest as `errors`.
NEVER hard-fail a poster request just because this sidecar is unreachable or returned
garbage -- always fall back to the deterministic style_matcher.resolve_style_preset()
path instead (demo_server.py's job, not this module's).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tendoo.layouts.font_engine import FONT_CATALOG
from tendoo.layouts.freeform.layout import ICON_SVG_BY_NAME, ROLE_SCALE
from tendoo.layouts.freeform.zones import ZONE_NAMES
from tendoo.layouts.style_matcher import CATEGORY_FIELD_SLOTS

VALID_ROLES = set(ROLE_SCALE.keys())
VALID_ICONS = set(ICON_SVG_BY_NAME.keys())
VALID_FONT_KEYS = set(FONT_CATALOG.keys()) | {"auto"}
VALID_TITLE_ACTIONS = {"use_prompt", "generate", "none"}
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
# How similar (0..1, difflib ratio) a `field: null` block's text must be to an
# already-filled mandatory field's real value before it's treated as an accidental
# restatement and dropped -- see validate_render_plan()'s duplicate filter.
DUPLICATE_SIMILARITY_THRESHOLD = 0.72

MODEL: Any = None
TOKENIZER: Any = None
DEVICE: str = "cuda:0"


# ==================================================================================
# Model loading
# ==================================================================================

def _resolve_qwen3_checkpoint_path(variant: str) -> str:
    """
    Mirrors src/flux2/text_encoder.py::load_qwen3_embedder's own checkpoint-resolution
    order (env var override -> generic TEXT_ENCODER_PATH -> local persistent-data
    auto-detect -> the public 'Qwen/Qwen3-{variant}-FP8' HF hub id) so this sidecar picks
    up the SAME local weights the main demo server already uses, without needing the
    embedding-only Qwen3Embedder wrapper class (which has no .generate() method).
    """
    env_key = f"QWEN3_{variant.upper()}_MODEL_PATH"
    if env_key in os.environ and os.path.exists(os.environ[env_key]):
        return os.environ[env_key]
    if "TEXT_ENCODER_PATH" in os.environ and os.path.exists(os.environ["TEXT_ENCODER_PATH"]):
        return os.environ["TEXT_ENCODER_PATH"]
    try:
        from flux2.util import find_persistent_data_root
        p_root = find_persistent_data_root()
        if p_root:
            for cp in (os.path.join(p_root, "text_encoder"), p_root):
                if os.path.exists(cp) and (
                    os.path.exists(os.path.join(cp, "config.json"))
                    or os.path.exists(os.path.join(cp, "model.safetensors.index.json"))
                ):
                    return cp
    except Exception:
        pass
    return f"Qwen/Qwen3-{variant}-FP8"


# Standard Qwen ChatML template (system/user/assistant turns), used ONLY as a last-
# resort fallback -- see load_model()'s defensive check below. Confirmed needed on a
# real server run (2026-09-10): a local '.../text_encoder' checkpoint directory can
# have valid model weights but an incomplete tokenizer (missing chat_template.json),
# which crashes apply_chat_template() with "chat_template is not set" even though the
# checkpoint is otherwise perfectly usable.
QWEN_CHATML_FALLBACK_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def _resolve_tokenizer_path(model_path: str) -> str:
    """
    Mirrors src/flux2/text_encoder.py::Qwen3Embedder's own tokenizer-path resolution
    EXACTLY: when the model weights live in a local '.../text_encoder' directory, the
    full tokenizer (incl. chat_template.json) often lives in a SIBLING '.../tokenizer'
    directory instead of alongside the model weights -- loading the tokenizer from
    `model_path` directly can silently pick up an incomplete tokenizer missing its
    chat template. Only applies to local paths; a bare HF hub id (e.g.
    'Qwen/Qwen3-4B-FP8') has no local sibling to check and is returned as-is.
    """
    if os.path.exists(model_path):
        parent_dir = os.path.dirname(os.path.abspath(model_path))
        sibling_tokenizer = os.path.join(parent_dir, "tokenizer")
        if os.path.exists(sibling_tokenizer):
            return sibling_tokenizer
    return model_path


def load_model(model_path: str, device: str) -> Tuple[Any, Any]:
    """Loads a plain AutoModelForCausalLM + AutoTokenizer for real text generation
    (unlike Qwen3Embedder, which only ever does an embedding forward pass)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = _resolve_tokenizer_path(model_path)
    if tokenizer_path != model_path:
        print(f"  [Tokenizer] Model dir has no full tokenizer -- using sibling: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    ).eval()

    # Defensive fallback: if even the resolved tokenizer has no chat_template (e.g. an
    # unexpected local layout, or a bare hub id whose repo genuinely lacks one), install
    # a standard Qwen ChatML template rather than crashing every single request.
    if getattr(tokenizer, "chat_template", None) is None:
        print("  [Tokenizer] No chat_template found on the resolved tokenizer -- falling back to a manual ChatML template.")
        tokenizer.chat_template = QWEN_CHATML_FALLBACK_TEMPLATE

    return model, tokenizer


# ==================================================================================
# Prompt construction
# ==================================================================================

SYSTEM_PROMPT_TEMPLATE = """Bạn là bộ não dàn trang (layout brain) cho một hệ thống sinh poster thương mại. Nhiệm vụ CHỈ \
gồm 3 việc: (1) quyết định tiêu đề cuối cùng, (2) liệt kê các đoạn chữ PHỤ THÊM mà prompt tự do yêu cầu (nếu có), \
(3) viết prompt mô tả nền cho mô hình diffusion. Bạn KHÔNG quyết định bố cục (layout) -- việc đó do code xử lý.

QUY TẮC TUYỆT ĐỐI (không tự suy diễn khác):
- CÁC TRƯỜNG THÔNG TIN đã liệt kê bên dưới (dù trống hay đã điền) là NGỮ CẢNH tham khảo, không phải để bạn viết lại
  nội dung của chúng -- nếu trường đã điền, nó SẼ được vẽ nguyên văn giá trị thật (không qua bạn). Bạn CHỈ nêu ý kiến
  về các trường này qua "extra_blocks" khi: (i) trường đang TRỐNG và prompt tự do có mô tả rõ nội dung cho nó (lúc đó
  gắn "field" đúng tên trường + "text" là nội dung bạn trích được), hoặc (ii) người dùng yêu cầu VỊ TRÍ cụ thể cho nội
  dung của 1 trường (gắn "field" + "zone").
- TUYỆT ĐỐI KHÔNG tạo "extra_blocks" mới (field: null) cho nội dung TRÙNG Ý NGHĨA với 1 trường đã liệt kê (dù trường
  đó đang trống hay đã điền) -- nếu prompt tự do lặp lại/diễn giải lại đúng nội dung 1 trường, hãy gắn "field" tương
  ứng thay vì tạo khối chữ mới trùng lặp. Lỗi thật cần tránh: form đã điền "Từ ngày...đến ngày..." NHƯNG prompt cũng
  viết lại y hệt -- TUYỆT ĐỐI không tạo thêm 1 khối chữ "Từ ngày...đến ngày..." nữa, vì nó sẽ bị vẽ 2 LẦN.
- Chỉ dùng "extra_blocks" với field: null cho nội dung THỰC SỰ MỚI, không khớp bất kỳ trường nào đã liệt kê (ví dụ:
  1 dòng CTA phụ, 1 câu khẩu hiệu thêm mà form không có trường tương ứng).
- "zone" CHỈ điền khi người dùng yêu cầu RÕ RÀNG 1 vị trí cụ thể ("ở góc trên bên trái", "ở giữa", "phía dưới cùng"...)
  -- đừng tự ý gán zone nếu người dùng không yêu cầu vị trí.
- Mã QR: nếu người dùng yêu cầu vị trí cụ thể cho mã QR, tạo 1 block field="qr" kèm "zone" tương ứng (bỏ qua "text").
  Nếu không có yêu cầu, đừng tạo block này -- mã QR (nếu bật) sẽ tự hiển thị ở vị trí mặc định.

TIÊU ĐỀ (title), quyết định theo đúng thứ tự sau:
(a) Nếu prompt tự do nêu rõ 1 dòng chữ lớn/hero/tiêu đề nổi bật (không nhất thiết dùng đúng từ "tiêu đề") -> action="use_prompt", text=nội dung đó.
(b) Nếu prompt tự do yêu cầu các đoạn chữ rải rác theo vị trí NHƯNG KHÔNG nhắc đến hero/tiêu đề -> action="none" (bỏ qua tiêu đề hoàn toàn).
(c) Nếu (a) và (b) đều không đúng và trường tiêu đề đã điền -> action="none" (dùng nguyên giá trị đã điền, không cần bạn nêu lại).
(d) Nếu không trường hợp nào đúng (tiêu đề trống, prompt không nhắc) -> action="generate", text=tiêu đề bạn tự nghĩ ra phù hợp ngữ cảnh.

VÙNG ĐẶT CHỮ HỢP LỆ (zone, đúng 9 tên sau): {zones}
VAI TRÒ CHỮ HỢP LỆ (role): {roles}
ICON TÙY CHỌN HỢP LỆ (icon): {icons}
FONT (font_key): "auto" là lựa chọn an toàn nhất trừ khi người dùng nêu rõ 1 font cụ thể trong danh sách: {fonts}.
STYLE_HINT: "auto" là lựa chọn an toàn nhất trừ khi người dùng mô tả rõ ràng 1 tông màu/ánh sáng cụ thể.

QUAN TRỌNG -- scene_prompt (mô tả nền cho mô hình diffusion) TUYỆT ĐỐI KHÔNG được chứa bất kỳ chuỗi chữ nội dung nào
(không trích dẫn tiêu đề, không viết chữ sẽ hiện trên poster) -- toàn bộ chữ được vẽ riêng bằng HTML/CSS, không phải
bởi mô hình diffusion. Viết scene_prompt bằng tiếng Anh, mô tả thuần cảnh/sản phẩm/ánh sáng, không nhắc đến khái niệm
"text"/"chữ"/"title" dưới bất kỳ hình thức nào.

CHỈ xuất ra DUY NHẤT một khối JSON hợp lệ, không kèm giải thích, đúng khuôn dạng sau:
{{"title": {{"action": "use_prompt|generate|none", "text": "..."}}, \
"extra_blocks": [{{"field": "ten_truong hoặc null", "text": "...", "zone": "ten_zone hoặc null", "role": "...", "icon": "..."}}], \
"scene_prompt": "...", "style_hint": "...", "font_key": "..."}}
"""

USER_PROMPT_TEMPLATE = """Danh mục (category): {category}

CÁC TRƯỜNG THÔNG TIN (tên trường: giá trị hiện tại, "(trống)" nếu chưa điền):
{category_fields_block}

Tiêu đề hiện tại (title): {title_value}

PROMPT TỰ DO (prompt ảnh): {prompt}

Gợi ý phong cách người dùng chọn (style_pref, chỉ là gợi ý, có thể điều chỉnh theo prompt tự do): {style_pref}
"""


def _format_category_fields(category_fields: Dict[str, str]) -> str:
    if not category_fields:
        return "(danh mục này không có trường nào)"
    return "\n".join(f"- {k}: {v if v else '(trống)'}" for k, v in category_fields.items())


def build_messages(
    category: str,
    category_fields: Dict[str, str],
    title_value: str,
    prompt: str,
    style_pref: str,
) -> List[Dict[str, str]]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        zones=", ".join(ZONE_NAMES),
        roles=", ".join(sorted(VALID_ROLES)),
        icons=", ".join(sorted(VALID_ICONS)),
        fonts=", ".join(sorted(FONT_CATALOG.keys())),
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        category=category or "promo",
        category_fields_block=_format_category_fields(category_fields),
        title_value=title_value if title_value else "(trống)",
        prompt=prompt or "(không có)",
        style_pref=style_pref or "auto",
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ==================================================================================
# Generation
# ==================================================================================

def generate_raw_response(
    model: Any,
    tokenizer: Any,
    device: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 1024,
) -> str:
    """Runs one .generate() call and returns the decoded continuation text (raw model
    output, not yet JSON-parsed). Separated from parsing so a mocked/monkeypatched
    version of this exact function is all tests need to exercise the rest of the
    pipeline without a GPU.

    enable_thinking=False mirrors src/flux2/text_encoder.py::Qwen3Embedder.forward()'s
    own apply_chat_template() call on this exact model family -- Qwen3 defaults to an
    internal <think>...</think> reasoning pass before its actual answer, which can eat
    the entire max_new_tokens budget before ever reaching the JSON output (confirmed on
    a real server run, 2026-09-10: every case timed out around ~28s with "No JSON
    object found in model response" until this flag was added).
    """
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_render_plan_json(raw_response: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Extracts the first {...} block from the model's raw text and JSON-decodes it.
    Never raises -- returns (None, [error]) on any parse failure."""
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        return None, ["No JSON object found in model response."]
    try:
        return json.loads(match.group(0)), []
    except json.JSONDecodeError as e:
        return None, [f"JSON parse failure: {e}"]


# ==================================================================================
# Validation -- pure functions, fully offline-testable (no GPU/model needed)
# ==================================================================================

def validate_render_plan(
    plan: Any,
    category: str,
    category_field_values: Dict[str, str],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Cleans and validates a raw (possibly malformed) render-plan dict from the model.

    `category_field_values`: the REAL current req values for CATEGORY_FIELD_SLOTS[category]
    (blank string if unfilled) -- used both to know which field names are valid for
    this category and to run the duplicate-content filter against already-filled ones.

    Never raises. Drops/corrects individual bad pieces rather than rejecting the whole
    plan outright (mirrors FreeformLayout.render_html's own silent-drop tolerance for
    unknown zone/icon) -- every drop/correction is recorded in `errors` but doesn't by
    itself invalidate the plan. Returns (None, errors) only when the input isn't a dict
    at all.
    """
    errors: List[str] = []
    if not isinstance(plan, dict):
        return None, ["render plan is not a JSON object"]

    valid_fields = set(CATEGORY_FIELD_SLOTS.get(category, [])) | {"qr"}

    # --- title ---
    raw_title = plan.get("title")
    title_action = "none"
    title_text: Optional[str] = None
    if isinstance(raw_title, dict):
        action = raw_title.get("action")
        title_action = action if action in VALID_TITLE_ACTIONS else "none"
        if action is not None and action not in VALID_TITLE_ACTIONS:
            errors.append(f"unknown title.action '{action}', defaulted to 'none'")
        text = raw_title.get("text")
        if isinstance(text, str) and text.strip():
            title_text = text.strip()
    elif raw_title is not None:
        errors.append("'title' is not an object, ignored")

    # --- extra_blocks ---
    raw_blocks = plan.get("extra_blocks")
    clean_blocks: List[Dict[str, Any]] = []
    if isinstance(raw_blocks, list):
        for i, b in enumerate(raw_blocks):
            if not isinstance(b, dict):
                errors.append(f"extra_blocks[{i}] is not an object, dropped")
                continue
            text = b.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"extra_blocks[{i}] has no non-empty 'text', dropped")
                continue
            field = b.get("field")
            if field is not None and field not in valid_fields:
                errors.append(f"extra_blocks[{i}] has unknown field '{field}' for category '{category}', treated as null")
                field = None
            zone = b.get("zone")
            if zone is not None and zone not in ZONE_NAMES:
                errors.append(f"extra_blocks[{i}] has unknown zone '{zone}', dropped (block kept)")
                zone = None
            role = b.get("role") if b.get("role") in VALID_ROLES else "body"
            if b.get("role") is not None and b.get("role") not in VALID_ROLES:
                errors.append(f"extra_blocks[{i}] has unknown role '{b.get('role')}', defaulted to 'body'")
            clean_block: Dict[str, Any] = {"field": field, "text": text.strip(), "zone": zone, "role": role}
            icon = b.get("icon")
            if icon:
                if icon in VALID_ICONS:
                    clean_block["icon"] = icon
                else:
                    errors.append(f"extra_blocks[{i}] has unknown icon '{icon}', dropped")
            color = b.get("color")
            if color:
                if isinstance(color, str) and _HEX_COLOR_RE.match(color):
                    clean_block["color"] = color
                else:
                    errors.append(f"extra_blocks[{i}] has invalid color '{color}', dropped")
            clean_blocks.append(clean_block)
    elif raw_blocks is not None:
        errors.append("'extra_blocks' is not a list, ignored")

    # --- duplicate filter: drop field:null blocks that closely restate an
    # already-filled mandatory field's real value (the model failing to tag `field`
    # correctly per the system prompt's explicit instruction). Two heuristics, since a
    # restatement is often a short value embedded in a longer natural-language
    # sentence (whole-string ratio alone misses that): (1) the field's exact value
    # appears verbatim inside the block's text (or vice versa), (2) the two strings are
    # a close whole-string match (catches near-identical short blocks). ---
    filled_values = [v for v in category_field_values.values() if v and v.strip()]

    def _is_near_duplicate(text: str) -> bool:
        norm_text = text.lower()
        for fv in filled_values:
            norm_fv = fv.lower()
            if norm_fv in norm_text or norm_text in norm_fv:
                return True
            if difflib.SequenceMatcher(None, norm_text, norm_fv).ratio() >= DUPLICATE_SIMILARITY_THRESHOLD:
                return True
        return False

    deduped_blocks: List[Dict[str, Any]] = []
    for b in clean_blocks:
        if b["field"] is None and _is_near_duplicate(b["text"]):
            errors.append(
                f"extra_block text '{b['text'][:40]}...' looks like a near-duplicate "
                f"of an already-filled field, dropped"
            )
            continue
        deduped_blocks.append(b)

    # --- scene_prompt / style_hint / font_key ---
    scene_prompt = plan.get("scene_prompt")
    if not isinstance(scene_prompt, str) or not scene_prompt.strip():
        errors.append("missing/empty 'scene_prompt'")
        scene_prompt = ""
    else:
        scene_prompt = scene_prompt.strip()

    font_key = plan.get("font_key")
    if font_key not in VALID_FONT_KEYS:
        if font_key is not None:
            errors.append(f"unknown font_key '{font_key}', defaulted to 'auto'")
        font_key = "auto"

    style_hint = plan.get("style_hint")
    if not isinstance(style_hint, str) or not style_hint.strip():
        style_hint = "auto"
    # Not validated against LAYOUT_COMPATIBLE_STYLES here -- the final layout isn't
    # known yet (decided downstream in demo_server.py from has_freeform_trigger); the
    # existing detect_scene_lighting_tone() re-validates it against whatever layout
    # actually gets chosen, same as every other style_hint source in this pipeline.

    cleaned = {
        "title": {"action": title_action, "text": title_text},
        "extra_blocks": deduped_blocks,
        "scene_prompt": scene_prompt,
        "style_hint": style_hint,
        "font_key": font_key,
    }
    return cleaned, errors


def is_render_plan_usable(plan: Optional[Dict[str, Any]]) -> bool:
    """A much lower bar than v1: a plan contributes value if it has a non-empty
    scene_prompt, a real title decision, or any extra_blocks -- any one of these alone
    is still worth using over the fully-deterministic fallback. A plan with NONE of
    these (e.g. total generation failure) is not worth threading through at all."""
    if not plan:
        return False
    title = plan.get("title") or {}
    has_title_decision = title.get("action") in ("use_prompt", "generate") and bool(title.get("text"))
    return bool(plan.get("scene_prompt")) or has_title_decision or bool(plan.get("extra_blocks"))


def generate_render_plan(
    category: str,
    category_fields: Dict[str, str],
    title_value: str,
    prompt: str,
    style_pref: str = "auto",
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """End-to-end: build prompt -> generate -> parse -> validate. Requires MODEL/TOKENIZER
    to already be loaded (see lifespan/main() below)."""
    if MODEL is None or TOKENIZER is None:
        return None, ["model not loaded"]
    messages = build_messages(category, category_fields, title_value, prompt, style_pref)
    raw_response = generate_raw_response(MODEL, TOKENIZER, DEVICE, messages)
    raw_plan, parse_errors = parse_render_plan_json(raw_response)
    if raw_plan is None:
        # Include a preview of what the model actually said -- otherwise a parse
        # failure is undiagnosable after the fact (confirmed needed on a real server
        # run, 2026-09-10: "No JSON object found" alone didn't say WHY).
        preview = raw_response[:400].replace("\n", " ")
        return None, parse_errors + [f"raw model output preview: {preview!r}"]
    cleaned, validation_errors = validate_render_plan(raw_plan, category, category_fields)
    return cleaned, parse_errors + validation_errors


# ==================================================================================
# FastAPI app
# ==================================================================================

class RenderPlanRequest(BaseModel):
    category: str = "promo"
    title: str = ""
    category_fields: Dict[str, str] = {}
    prompt: str = ""
    style_pref: str = "auto"
    primary_color: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER
    # MODEL/TOKENIZER are set by main() before uvicorn starts in production; tests
    # exercise the endpoint with them monkeypatched and never hit this branch.
    yield


app = FastAPI(title="Tendoo LLM Render-Plan Sidecar", lifespan=lifespan)


@app.get("/api/health")
async def health_check():
    return {"status": "online", "model_loaded": MODEL is not None}


@app.post("/api/render-plan")
async def api_render_plan(req: RenderPlanRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    plan, errors = generate_render_plan(
        category=req.category,
        category_fields=req.category_fields,
        title_value=req.title,
        prompt=req.prompt,
        style_pref=req.style_pref,
    )
    return {
        "plan": plan,
        "usable": is_render_plan_usable(plan),
        "errors": errors,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tendoo LLM Render-Plan Sidecar")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861, help="Distinct from demo_server.py's 7860")
    parser.add_argument("--model", type=str, default=None, help="HF hub id or local path; default: auto-resolve Qwen3-4B-FP8")
    parser.add_argument("--variant", type=str, default="4B", choices=["4B", "8B"], help="Used only when --model is omitted")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    global MODEL, TOKENIZER, DEVICE
    args = parse_args()
    DEVICE = args.device
    model_path = args.model or _resolve_qwen3_checkpoint_path(args.variant)

    print("=" * 80)
    print("Tendoo LLM Render-Plan Sidecar")
    print(f"  Model  : {model_path}")
    print(f"  Device : {DEVICE}")
    print("=" * 80)
    print("Loading model & tokenizer...")
    MODEL, TOKENIZER = load_model(model_path, DEVICE)
    print("Model loaded. Starting server...")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
