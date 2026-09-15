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
  3. There is no dedicated "title" concept in this model's output (2026-09-13
     simplification -- removed the earlier action/text title object + its 4-branch
     precedence). A form title field is just another mandatory field, resolved
     entirely in Python without this model's input: rendered verbatim if filled and no
     ad-hoc content exists, or skipped outright the moment ANY `extra_blocks` entry
     with `field: null` is present (demo_server.py::run_pipeline_inference). If the
     free prompt wants a specific headline instead of (or in addition to) the form
     title, this model expresses that the exact same way as any other ad-hoc content:
     one more `extra_blocks` entry with `field: null`, `size: "xlarge"` (or `"large"`),
     `container: "none"` -- no special-casing needed, since the ad-hoc-suppresses-title
     rule above already makes that block the one thing that renders in the title's
     place. When the form title is empty AND the prompt is sparse/asks for nothing new,
     Python's own per-category default headline + Primary Headline Guard
     (layout.py::OmniBlockLayout._extract_blocks) already guarantee a reasonable
     fallback headline with zero model involvement.

DUPLICATE/OMISSION PREVENTION (extra_blocks[].field): a block tagged with `field`
only affects that ONE field's content, resolved in
Python (see demo_server.py::run_pipeline_inference): the real already-filled value wins
over anything this model writes (fidelity + de-duplication guarantee -- a form value can
never be silently corrupted/duplicated by generation), while a genuinely blank field
falls back to trusting this model's extraction (fills gaps when the user typed the whole
brief into the free prompt instead of the form, e.g. prompt_test.txt lines 21-41).
In Pure OmniBlock architecture, every block is dynamically routed to its sanctuary zone
based on role and explicit zone request.

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
names, near-duplicate content) -- validate_render_plan() (src/tendoo/render_plan.py) never
raises for bad model output, it filters/drops/corrects what it can and reports the rest
as `errors`. NEVER hard-fail a poster request just because this sidecar is unreachable or
returned garbage -- always fall back to the deterministic style_matcher.resolve_style_preset()
path instead (demo_server.py's job, not this module's).

(2026-09-13: this file now holds ONLY model loading + FastAPI routing + CLI entrypoint --
prompt construction/JSON parsing/validation moved to `render_plan.py`, a pure-function
module with zero GPU/model dependency, easier to unit-test and to read in isolation.)

(2026-09-14: every generate_render_plan() call now appends 1 JSON line to
RENDER_PLAN_LOG_PATH (default logs/render_plan_requests.jsonl) -- full input
(category/category_fields/prompt/style_pref/primary_color) and output (raw model
text before parsing, the cleaned+validated plan, usable, errors, latency). Real
production traffic is exactly the kind of data scripts/verify_render_plan_llm.py's
hand-written cases can't fully substitute for -- see _log_render_plan_io() below.
Best-effort: a logging failure never breaks/slows the actual request.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tendoo.render_plan import (
    build_messages,
    is_render_plan_usable,
    parse_render_plan_json,
    validate_render_plan,
)

MODEL: Any = None
TOKENIZER: Any = None

# (2026-09-14) Every /api/render-plan call's full input + output, 1 JSON line each --
# for offline review (real-GPU debugging, prompt-tuning iteration; see
# scripts/verify_render_plan_llm.py's capability probe, which found real gaps this
# log would have made much faster to diagnose from actual production traffic instead
# of only from hand-written test cases). A plain module-level path (not computed
# inside the log function) so tests/conftest.py can redirect it to tmp_path the same
# way demo_server.py's OUTPUT_DIR is redirected -- never write real log files during
# a pytest run.
RENDER_PLAN_LOG_PATH = PROJECT_ROOT / "logs" / "render_plan_requests.jsonl"
DEVICE: str = "cuda:0"

__all__ = [
    "RenderPlanRequest",
    "app",
    "build_messages",
    "generate_raw_response",
    "generate_render_plan",
    "is_render_plan_usable",
    "load_model",
    "parse_render_plan_json",
    "validate_render_plan",
]


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
# Generation
# ==================================================================================

def generate_raw_response(
    model: Any,
    tokenizer: Any,
    device: str,
    messages: list,
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


def _log_render_plan_io(
    category: str,
    category_fields: Dict[str, str],
    prompt: str,
    style_pref: str,
    primary_color: Optional[str],
    raw_response: Optional[str],
    plan: Optional[Dict[str, Any]],
    errors: list,
    latency_s: float,
) -> None:
    """Appends ONE JSON line to RENDER_PLAN_LOG_PATH per generate_render_plan() call --
    full input (category/category_fields/prompt/style_pref/primary_color) and output
    (raw model text BEFORE parsing, the cleaned+validated plan, usable, errors,
    latency). Read this file back with `jq`/pandas/a text editor for offline review.

    Best-effort by design: a logging failure (disk full, permissions, whatever) must
    NEVER break or slow down the actual render-plan request -- mirrors
    try_fetch_render_plan()'s own "never let an auxiliary concern fail the main
    request" philosophy in demo_server.py. Every exception is swallowed, only printed.
    """
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {
                "category": category,
                "category_fields": category_fields,
                "prompt": prompt,
                "style_pref": style_pref,
                "primary_color": primary_color,
            },
            "output": {
                "raw_response": raw_response,
                "plan": plan,
                "usable": is_render_plan_usable(plan),
                "errors": errors,
            },
            "latency_s": round(latency_s, 2),
        }
        RENDER_PLAN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RENDER_PLAN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [Render Plan Log] Notice: failed to write log entry: {e}")


def generate_render_plan(
    category: str,
    category_fields: Dict[str, str],
    prompt: str,
    style_pref: str = "auto",
    primary_color: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], list]:
    """End-to-end: build prompt -> generate -> parse -> validate. Requires MODEL/TOKENIZER
    to already be loaded (see lifespan/main() below).

    `primary_color`: not used by any prompt/generation logic here (build_messages()
    never takes it) -- accepted purely so it can be included in the request log below
    (see RENDER_PLAN_LOG_PATH), since it's part of the real HTTP request's full input
    and callers may want it in the log even though it doesn't affect this function's
    own behavior.

    Every call is logged (see _log_render_plan_io) regardless of which branch it
    returns from -- "model not loaded" and "no JSON found" failures are just as
    valuable to have on record as a successful plan.
    """
    t0 = time.time()
    if MODEL is None or TOKENIZER is None:
        errors = ["model not loaded"]
        _log_render_plan_io(category, category_fields, prompt, style_pref, primary_color, None, None, errors, time.time() - t0)
        return None, errors
    messages = build_messages(category, category_fields, prompt, style_pref)
    raw_response = generate_raw_response(MODEL, TOKENIZER, DEVICE, messages)
    raw_plan, parse_errors = parse_render_plan_json(raw_response)
    if raw_plan is None:
        # Include a preview of what the model actually said -- otherwise a parse
        # failure is undiagnosable after the fact (confirmed needed on a real server
        # run, 2026-09-10: "No JSON object found" alone didn't say WHY). The FULL raw
        # response still goes into the log entry below regardless of this preview.
        preview = raw_response[:400].replace("\n", " ")
        errors = parse_errors + [f"raw model output preview: {preview!r}"]
        _log_render_plan_io(category, category_fields, prompt, style_pref, primary_color, raw_response, None, errors, time.time() - t0)
        return None, errors
    cleaned, validation_errors = validate_render_plan(raw_plan, category, category_fields)
    errors = parse_errors + validation_errors
    _log_render_plan_io(category, category_fields, prompt, style_pref, primary_color, raw_response, cleaned, errors, time.time() - t0)
    return cleaned, errors


# ==================================================================================
# FastAPI app
# ==================================================================================

class RenderPlanRequest(BaseModel):
    """
    Schema nhận diện yêu cầu tạo Render Plan từ demo_server.

    TẠI SAO CẦN LÀM:
    - Định nghĩa tường minh các trường dữ liệu cần thiết cho Qwen3 suy luận:
      danh mục (category), các trường danh mục (category_fields), prompt tự do của
      người dùng và style gợi ý. (2026-09-13: bỏ `title` -- không còn khái niệm tiêu
      đề riêng cho model quyết định, xem module docstring.)
    - Giúp FastAPI và Pydantic tự động validate kiểu dữ liệu, ngăn ngừa injection
      hoặc payload rỗng làm crash sidecar process.
    """
    category: str = "promo"
    category_fields: Dict[str, str] = {}
    prompt: str = ""
    style_pref: str = "auto"
    primary_color: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời FastAPI sidecar service.

    TẠI SAO CẦN LÀM:
    - MODEL và TOKENIZER được nạp trước khi server lắng nghe request.
    - Sử dụng mô hình lifespan hiện đại của FastAPI (thay thế @app.on_event deprecated)
      giúp giải phóng tài nguyên và tương thích chuẩn ASGI/Uvicorn mới nhất.
    """
    global MODEL, TOKENIZER
    # MODEL/TOKENIZER are set by main() before uvicorn starts in production; tests
    # exercise the endpoint with them monkeypatched and never hit this branch.
    yield


app = FastAPI(title="Tendoo LLM Render-Plan Sidecar", lifespan=lifespan)


@app.get("/api/health")
async def health_check():
    """Kiểm tra trạng thái sẵn sàng của LLM Sidecar."""
    return {"status": "online", "model_loaded": MODEL is not None}


@app.post("/api/render-plan")
async def api_render_plan(req: RenderPlanRequest):
    """
    Endpoint chính sinh Render Plan từ form + prompt tự do.

    TẠI SAO CẦN LÀM:
    - Tiếp nhận yêu cầu từ demo_server qua HTTP REST nội bộ.
    - Nếu sidecar chưa nạp weights (hoặc đang tải), trả mã 503 ngay lập tức để
      demo_server chủ động fallback sang quy tắc heuristic xác định (deterministic fallback),
      tuyệt đối không làm treo hoặc crash tiến trình sinh ảnh chính.
    """
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    plan, errors = generate_render_plan(
        category=req.category,
        category_fields=req.category_fields,
        prompt=req.prompt,
        style_pref=req.style_pref,
        primary_color=req.primary_color,
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
