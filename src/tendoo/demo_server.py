#!/usr/bin/env python3
"""
src/tendoo/demo_server.py

TENDOO AI - MULTI-USER DEMO SERVER
===================================
1. Long-running FastAPI daemon: Models remain warm in VRAM for ~3.5s instant inference.
2. Concurrency Control: Async Lock prevents multiple users from colliding on GPU VRAM.
3. Accessible over internal company network (0.0.0.0) or JupyterLab Proxy.
4. Clean Shutdown (Ctrl+C): Explicitly deletes model refs, runs gc.collect(),
   and flushes torch.cuda.empty_cache() across both A30 GPUs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import io
import json
import os
import re
import shutil
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tendoo.core.base import ColorPalette, PosterContent
from tendoo.core.colors import analyze_color_harmony, hex_to_hue
from tendoo.engine import get_layout
from tendoo.core.style import inject_color_guidance, resolve_style_preset
from tendoo.core.category_schema import CATEGORY_FIELD_SLOTS, DEFAULT_FIELD_ROLE
from tendoo.poster_renderer import PosterRenderer


# Global application state
DIT_MODEL: Any = None
AE_MODEL: Any = None
TEXT_ENCODER: Any = None
AE_DTYPE: Any = None
DEVICE_DIT: str = "cuda:0"
DEVICE_AUX: str = "cuda:1"
INFER_LOCK = asyncio.Lock()
IS_MOCK_MODE: bool = False

OUTPUT_DIR = PROJECT_ROOT / "output_demo_server"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UI_HTML_PATH = Path(__file__).resolve().parent / "demo_ui.html"

# Retention cap (2026-09-10): every generate request writes a brand-new run_<ts>/
# folder (mask/backgrounds/HTML/final poster) and nothing ever deleted the old ones --
# this directory grew to 1600+ run folders with no bound. Keep only the N most
# recently created run folders; the rest are pruned right after each successful
# generation (see _prune_old_output_runs()). Override via env var if needed.
MAX_OUTPUT_RUNS = int(os.environ.get("MAX_OUTPUT_RUNS", "100"))


__all__ = [
    "CATEGORY_REQUIRED_FIELDS",
    "GenerateRequest",
    "app",
    "build_poster_content",
    "detect_scene_lighting_tone",
    "free_all_gpu_memory",
    "inject_color_guidance",
    "inject_spatial_layout_guidance",
    "main",
    "run_pipeline_inference",
    "sanitize_and_inject_zero_text",
    "try_fetch_render_plan",
    "validate_required_fields",
]


def free_all_gpu_memory():
    """
    Giải phóng triệt để VRAM trên toàn bộ các GPU CUDA khi tắt server hoặc reload.
    
    TẠI SAO CẦN LÀM:
    - Trên môi trường multi-GPU (2x NVIDIA A30), PyTorch caching allocator thường giữ lại
      reserved memory ngay cả khi python process kết thúc hoặc reload.
    - Xóa tường minh tham chiếu model, ép gọi gc.collect(), rồi lặp qua từng device ID
      để gọi empty_cache() + ipc_collect() đảm bảo không rò rỉ VRAM (OOM) cho các phiên
      chạy tiếp theo.
    """
    global DIT_MODEL, AE_MODEL, TEXT_ENCODER
    print("\n🛑 SHUTTING DOWN TENDOO DEMO SERVER...")
    print("🧹 Releasing model weights and cleaning GPU memory...")
    
    del DIT_MODEL
    del AE_MODEL
    del TEXT_ENCODER
    DIT_MODEL = None
    AE_MODEL = None
    TEXT_ENCODER = None

    gc.collect()
    if torch.cuda.is_available():
        for dev_id in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(dev_id):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                alloc = torch.cuda.memory_allocated(dev_id) / (1024 ** 2)
                res = torch.cuda.memory_reserved(dev_id) / (1024 ** 2)
                print(f"  [GPU {dev_id}] Allocated: {alloc:.1f} MB | Reserved: {res:.1f} MB (FREED)")
            except Exception as e:
                print(f"  [GPU {dev_id}] Cleanup notice: {e}")
    print("✨ All GPU VRAM has been successfully released!\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models on startup; ensures 100% VRAM release on shutdown."""
    global DIT_MODEL, AE_MODEL, TEXT_ENCODER, AE_DTYPE, DEVICE_DIT, DEVICE_AUX

    if IS_MOCK_MODE or not torch.cuda.is_available():
        print("🟡 [SERVER] Running in MOCK / LOCAL mode (No GPU weights loaded).")
    else:
        print("=" * 80)
        print("🚀 INITIALIZING FLUX.2 KLEIN 4B DISTILL FOR DEMO SERVER")
        print("=" * 80)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            DEVICE_DIT = "cuda:0"
            DEVICE_AUX = "cuda:1"
            print(f"  [Device Setup] Multi-GPU: DiT on {DEVICE_DIT}, AE & Qwen3 on {DEVICE_AUX}")
        else:
            DEVICE_DIT = DEVICE_AUX = "cuda:0"
            print(f"  [Device Setup] Single GPU: All models on {DEVICE_DIT}")

        from flux2 import util

        model_name = "flux.2-klein-4b"
        pdata_candidates = [
            Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-4B")),
            Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B")),
        ]
        for pdata in pdata_candidates:
            distill_cand = pdata / "flux-2-klein-4b.safetensors"
            if distill_cand.exists() and "KLEIN_4B_MODEL_PATH" not in os.environ:
                os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_cand)
                break

        t0 = time.time()
        print("⏳ Loading DiT 4B weights into VRAM...")
        DIT_MODEL = util.load_flow_model(model_name, device=DEVICE_DIT)
        DIT_MODEL.eval()

        print("⏳ Loading AutoEncoder (VAE) weights...")
        AE_MODEL = util.load_ae(model_name, device=DEVICE_AUX)
        AE_MODEL.eval()
        AE_DTYPE = next(AE_MODEL.parameters()).dtype

        print("⏳ Loading Qwen3 TextEncoder weights...")
        TEXT_ENCODER = util.load_text_encoder(model_name, device=DEVICE_AUX)

        dur_init = time.time() - t0
        print(f"✅ All models loaded and warm in VRAM in {dur_init:.1f}s!")
        for dev_id in range(num_gpus):
            alloc = torch.cuda.memory_allocated(dev_id) / (1024 ** 2)
            res = torch.cuda.memory_reserved(dev_id) / (1024 ** 2)
            print(f"  [GPU {dev_id}] Occupied: {alloc:.1f} MB (Reserved: {res:.1f} MB)")
        print("=" * 80)

    yield

    # Teardown logic upon server exit
    free_all_gpu_memory()


app = FastAPI(title="Tendoo AI Demo Server", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


class GenerateRequest(BaseModel):
    # 1. Category selector
    category: str = "promo"

    # 2. General Store / Brand Information (Common to all categories)
    store_name: str = ""
    address: str = ""
    phone: str = ""
    website_link: str = ""
    enable_qr: bool = False

    # 3. Category: Promo Poster specific
    title: str = ""
    image_base64: Optional[str] = None
    discount: str = ""
    applied_product: str = ""
    date_start: str = ""
    date_end: str = ""
    image_description: str = ""

    # Category: Product Intro (Giới thiệu sản phẩm)
    price: str = ""
    product_name: str = ""
    product_desc: str = ""
    highlights: str = ""

    # Category: Opening Banner (Khai trương)
    opening_date: str = ""
    opening_promo: str = ""
    booking_contact: str = ""

    # Category: Customer Feedback (Feedback & Đánh giá)
    feedback_target: str = ""
    feedback_quote: str = ""
    feedback_rating: str = ""
    special_offer: str = ""

    # Category: Recruitment (Tuyển dụng)
    job_position: str = ""
    job_desc: str = ""
    apply_deadline: str = ""
    apply_method: str = ""

    # Category: Guide (Quy trình / Hướng dẫn)
    guide_steps: List[str] = []

    # Legacy / alias parameters for backwards compatibility
    headline: Optional[str] = None
    pre_header: str = ""
    slogan: str = ""
    offer_main: Optional[str] = None
    offer_sub: Optional[str] = None
    applicable: Optional[str] = None
    brand: Optional[str] = None
    hotline: Optional[str] = None
    rating: Optional[Any] = None
    customer_name: Optional[str] = None
    prompt_scene: Optional[str] = None
    prompt_corridor: Optional[str] = None

    # 4. Display / Design & Inference Controls (Common to all categories)
    # Phase B (per yeu_cau.txt): layout/style_hint/font_family are no longer raw
    # user-facing pickers -- the UI only sends style_pref + primary_color now, and
    # these three are auto-resolved by style_matcher.resolve_style_preset() in
    # run_pipeline_inference whenever left unset (None/""). Kept as plain Optional
    # fields (not removed) so a direct API caller / back-compat client can still force
    # an explicit value, same as before Phase B.
    layout: Optional[str] = None
    # (2026-09-14) "v1" (default, unchanged) or "v2" -- opts into the experimental
    # src/tendoo_v2 content-curation engine (role/group/zone via a hosted LLM +
    # deterministic solver) instead of the legacy render_plan sidecar, for THIS
    # request only. Falls back to TENDOO_DEFAULT_ENGINE env var when unset, so the
    # engine can be flipped server-side without any frontend change -- see
    # run_pipeline_inference()'s engine_version resolution. See src/tendoo_v2/README-
    # equivalent module docstrings (schema.py/solver.py/llm_client.py/bridge.py) for
    # the full design; this field is the ONLY production wiring point.
    engine_version: Optional[str] = None
    style_hint: Optional[str] = None
    text_effect: str = "auto"
    font_family: Optional[str] = None
    style_pref: str = "auto"
    primary_color: Optional[str] = None
    aspect_ratio: str = "1:1"
    num_images: int = 1
    seed: int = 42
    steps: int = 8
    guidance: float = 1.5
    width: Optional[int] = None
    height: Optional[int] = None


# Per-category required fields, per yeu_cau.txt (BA/tester spec). "Thông tin cửa hàng"
# and "Hiển thị/thiết kế" sections are explicitly NOT required for any category, and
# are therefore absent here entirely -- only fields inside a category's own "Thông tin
# chung" block that carry a `*` in the spec are listed. `guide` has no named scalar
# field to require -- it's special-cased in validate_required_fields() instead, since
# its requirement is "step 1 of guide_steps must be non-empty", not a fixed field name.
CATEGORY_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "promo": ["discount"],
    "product_intro": ["product_name", "product_desc"],
    "opening": ["opening_date"],
    "feedback": ["feedback_target", "feedback_quote"],
    "recruitment": ["job_position", "apply_deadline", "apply_method"],
    "guide": [],
}


def validate_required_fields(req: GenerateRequest) -> List[str]:
    """
    Returns the list of required field names that are missing/blank for req.category,
    per CATEGORY_REQUIRED_FIELDS. Empty list = valid. A field counts as missing when
    it's absent, None, or a blank/whitespace-only string -- matching how the frontend
    inputs behave (an untouched text input is just an empty string, never absent).

    `guide` is special-cased: yeu_cau.txt requires "mô tả bước 1", i.e. the first
    element of `guide_steps` must be non-blank (there's no fixed `guide_step_1` field --
    steps are a dynamic list, add/removed client-side).
    """
    missing: List[str] = []
    if req.category == "guide":
        if not req.guide_steps or not req.guide_steps[0].strip():
            missing.append("guide_steps[0]")
        return missing

    for field_name in CATEGORY_REQUIRED_FIELDS.get(req.category, []):
        value = getattr(req, field_name, None)
        if value is None or not str(value).strip():
            missing.append(field_name)
    return missing


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the Tendoo Studio web frontend."""
    if not UI_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="demo_ui.html not found")
    return HTMLResponse(content=UI_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Silences browser favicon 404 probes."""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/api/fonts")
async def get_fonts():
    """Returns all 19 curated Vietnamese typography fonts grouped by aesthetic archetype."""
    from tendoo.core.fonts import list_font_options
    return {
        "status": "success",
        "groups": list_font_options(),
    }


@app.get("/api/required-fields")
async def get_required_fields():
    """
    Single source of truth for per-category required fields (see
    CATEGORY_REQUIRED_FIELDS / validate_required_fields) -- the frontend fetches this
    once on load instead of hand-duplicating the list in JS, so the two can't drift.
    `guide_first_step: true` tells the frontend to special-case guide's first step
    input rather than looking for a fixed field name in the `fields` dict.
    """
    return {
        "status": "success",
        "fields": CATEGORY_REQUIRED_FIELDS,
        "guide_first_step": True,
    }


@app.get("/api/health")
async def health_check():
    """Health status endpoint."""
    return {
        "status": "online",
        "mock_mode": IS_MOCK_MODE,
        "gpus_available": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
def pil_to_base64_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _prune_old_output_runs(max_runs: int = MAX_OUTPUT_RUNS) -> None:
    """
    Keeps only the `max_runs` most recently created `run_*` folders under OUTPUT_DIR,
    deleting the rest. Called once per generate request (see run_pipeline_inference)
    so the directory self-limits instead of growing without bound. Sorts by folder
    name -- `run_<timestamp_ms>` is already lexicographically time-ordered -- rather
    than filesystem mtime, which Windows can report inconsistently for directories.
    Best-effort: a folder that fails to delete (e.g. still open elsewhere) is skipped,
    not fatal to the request that triggered the prune.
    """
    try:
        run_dirs = sorted(
            (p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")),
            key=lambda p: p.name,
        )
    except FileNotFoundError:
        return

    excess = len(run_dirs) - max_runs
    if excess <= 0:
        return

    for old_dir in run_dirs[:excess]:
        try:
            shutil.rmtree(old_dir)
        except Exception as e:
            print(f"  [!] Notice: failed to prune old output run {old_dir.name}: {e}")


def sanitize_and_inject_zero_text(user_prompt: str, has_ref_image: bool = False) -> str:
    """
    Sanitizes user scene prompt and injects targeted background ZERO-TEXT negative constraints.
    Protects product packaging and authentic brand labels (e.g. Hảo Hảo noodles, logo badges)
    while preventing DiT from hallucinating broken advertising slogans onto the background copy space.
    """
    if not user_prompt or not user_prompt.strip():
        base = (
            "Commercial advertising photography, professional studio lighting, "
            "clean photographic background, text-free background area, "
            "no floating graphic text, no headline or poster typography on background"
        )
        if has_ref_image:
            base = f"{base}, preserve authentic product packaging, original brand details and label typography on the product"
        return base

    prompt = user_prompt.strip()

    # 1. Strip resolution / aspect ratio pollution (Rule 6: 9:16, 16:9, 4:5, 1:1, 8k, 4k, 1080p, etc.)
    prompt = re.sub(
        r'\b(?:9:16|16:9|4:5|1:1|8k|4k|2k|1080p|720p|hd|uhd|full\s*hd)\b',
        '',
        prompt,
        flags=re.IGNORECASE
    )

    # 2. Filter out explicit user directives trying to draw text onto the background image
    # (Representation Clash prevention - Rule 3)
    text_intent_patterns = [
        r'\b(?:có|với|kèm|ghi|viết|in|thêu)\s+(?:dòng\s+)?(?:chữ|text|tiêu\s+đề|slogan|thông\s+tin|chữ\s+viết)\b[^\,\.]*',
        r'\bwith\s+(?:text|words|letters|typography|title|headline|caption|written\s+words)\b[^\,\.]*',
        r'\b(?:chữ|text)\s*:\s*["\'][^"\']*["\']',
        r'\b(?:chữ|text)\s*:\s*[^\,\.]*',
    ]
    for pat in text_intent_patterns:
        prompt = re.sub(pat, '', prompt, flags=re.IGNORECASE)

    # Clean redundant punctuation and whitespaces
    prompt = re.sub(r'[\,\;\s]+', ' ', prompt).strip(' ,;')

    # 3. Targeted background negative constraints (DO NOT use unbranded, no logos, no labels!)
    zero_text_clause = (
        "clean photographic background, text-free background area, "
        "no floating graphic text, no headline or poster typography on background"
    )

    lower_prompt = prompt.lower()
    if "text-free background" not in lower_prompt and "zero text" not in lower_prompt:
        if prompt:
            prompt = f"{prompt}, {zero_text_clause}"
        else:
            prompt = f"Commercial advertising photography, {zero_text_clause}"

    # 4. If an uploaded reference product is present, protect its authentic packaging & logo
    if has_ref_image and "preserve authentic product packaging" not in prompt.lower():
        prompt = f"{prompt}, preserve authentic product packaging, original brand details and label typography on the product"

    return prompt


def inject_spatial_layout_guidance(
    scene_prompt: str,
    layout_name: str = "omni",
    has_ref_image: bool = False,
) -> str:
    """
    Tiêm chỉ dẫn không gian bảo vệ Product Sanctuary của OmniBlock vào scene prompt.
    Giúp mô hình khuếch tán tập trung chủ thể thương mại vào tâm, chừa khoảng trống biên 9 ô cho chữ.
    """
    prompt = (scene_prompt or "").strip()
    lower_p = prompt.lower()

    spatial_guidance = (
        "commercial advertising composition, pristine central sanctuary protecting centered subject, "
        "harmonious balanced negative copy space on perimeter and corners reserved for typography"
    )
    if has_ref_image:
        spatial_guidance = f"{spatial_guidance}, preserve authentic product details and branding placed in the center sanctuary"

    needed = not any(k in lower_p for k in ["product sanctuary", "negative copy space"])
    if needed:
        prompt = f"{prompt}, {spatial_guidance}" if prompt else spatial_guidance

    return prompt


def detect_scene_lighting_tone(
    scene_prompt: str,
    user_hint: str = "auto",
    layout_name: str = "omni",
) -> str:
    """
    Tự động nhận diện phong cách ánh sáng và chất liệu quang học phù hợp nhất từ mô tả cảnh.
    Tương thích với danh mục 11 phong cách thẩm mỹ thương mại cao cấp của OmniBlock.
    """
    from tendoo.core.style import LAYOUT_COMPATIBLE_STYLES, OMNI_STYLES

    valid_styles = OMNI_STYLES
    default_style = str(LAYOUT_COMPATIBLE_STYLES["omni"]["default"])

    # Nếu người dùng chỉ định một phong cách hợp lệ trong danh mục Omni, tôn trọng lựa chọn đó
    if user_hint and user_hint.lower() not in ("auto", "", "none"):
        clean_hint = user_hint.lower().strip()
        if clean_hint in valid_styles:
            return clean_hint

    # Tự động nhận diện phong cách dựa trên từ khóa ngữ nghĩa trong prompt
    lower_p = (scene_prompt or "").lower()
    is_dark = any(k in lower_p for k in ["neon", "cyber", "night", "dark", "tối", "bóng đêm", "đen", "slate", "black", "moody", "gaming", "tech"])
    is_warm_gold = any(k in lower_p for k in ["gold", "vàng", "hoàng gia", "luxury", "sunset", "golden hour", "warm", "nến", "candle", "sang trọng"])
    is_stone_marble = any(k in lower_p for k in ["marble", "đá cẩm thạch", "bục đá", "sa thạch", "stone", "pedestal", "nhựa đường", "asphalt"])
    is_wood = any(k in lower_p for k in ["wood", "gỗ", "mặt bàn", "quán cafe", "tea table", "ấm cúng"])
    # (2026-09-12 fix, found via scripts/test_render_engine_isolated.py's diverse-case
    # sweep: bare "light" and bare "clean" are BOTH substrings that always appear in the
    # zero-text sanitize clause `sanitize_and_inject_zero_text()` unconditionally appends
    # to every single scene_prompt ("... professional studio lighting, clean photographic
    # background ..." -- see that function). That made every category's "no image
    # description given" default path silently resolve to "festive_light" (via "light" in
    # "lighting") instead of falling through to the real default_style, and would have
    # collapsed to "daylight_clean" (via bare "clean") for almost every OTHER prompt too
    # once the first collision was fixed, since that same clause is present everywhere.
    # Removed both overly generic single-word matches; the remaining, more specific
    # keywords for each style are unaffected.)
    is_festive = any(k in lower_p for k in ["lễ hội", "festival", "sparkle", "festive", "trăng", "moon", "tết", "trung thu", "pháo hoa"])
    is_minimal = any(k in lower_p for k in ["tường", "bê tông", "wall", "minimal", "phòng", "văn phòng", "office", "tối giản"])
    is_silk = any(k in lower_p for k in ["lụa", "silk", "thời trang", "fashion", "champagne", "mềm mại"])
    is_daylight = any(k in lower_p for k in ["daylight", "ban ngày", "ánh sáng tự nhiên", "sunlight", "sunny", "morning", "buổi sáng"])

    if is_dark:
        return "cyberpunk_grid"
    if is_warm_gold:
        return "luxury_gold"
    if is_wood:
        return "warm_wood"
    if is_festive:
        return "festive_light"
    if is_minimal:
        return "minimal_wall"
    if is_silk:
        return "champagne_silk"
    if is_stone_marble:
        return "cinematic_asphalt"
    if is_daylight:
        return "daylight_clean"

    return default_style


# (2026-09-12 cleanup: this used to carry its own independent copy of
# _HUE_NAME_BUCKETS/_hue_to_color_name/inject_color_guidance -- byte-for-byte identical
# to core/style.py's, which is the intended canonical "central style gateway" per that
# module's own docstring. core/style.py's copy and layouts/style_matcher.py's copy (a
# thin delegator to core/style.py) had zero real callers anywhere in the codebase; this
# module's own copy was the only one actually used (see call sites below). Consolidated
# to import the canonical implementation instead of maintaining a duplicate.)


# Phase C: where the render-plan sidecar (src/tendoo/llm_render_plan_server.py) listens.
# Distinct port from this server's own 7860 -- a genuinely separate OS process, started/
# scaled/restarted independently (see project memory: tendoo-3phase-plan-2026-09-09.md).
LLM_RENDER_PLAN_URL = os.environ.get("LLM_RENDER_PLAN_URL", "http://127.0.0.1:7861/api/render-plan")
# A real Qwen3-4B-FP8 .generate() call for this JSON-shaped output measured ~10-28s on
# a real server (2026-09-10) -- the original "5" default here was a placeholder picked
# before that was known and silently caused most requests to time out and fall back to
# the deterministic path (never an error, just the wrong resolved_layout/title -- only
# caught via scripts/verify_render_plan_llm.py's assertions). 45s leaves real margin.
LLM_RENDER_PLAN_TIMEOUT_S = float(os.environ.get("LLM_RENDER_PLAN_TIMEOUT_S", "45"))

def _collect_category_field_values(req: GenerateRequest) -> Dict[str, str]:
    """Every CATEGORY_FIELD_SLOTS[category] field's real current value, blank string if
    unfilled -- the single source of truth for both the render-plan LLM's context
    payload and the Python-side mandatory-field / de-duplication logic in
    run_pipeline_inference. Guide's steps are joined into one display string here (the
    LLM only ever reasons about them as context); the real per-step list is read
    straight from req.guide_steps wherever individual omni step blocks are built."""
    cat = (req.category or "promo").lower().strip()
    values: Dict[str, str] = {}
    for field in CATEGORY_FIELD_SLOTS.get(cat, []):
        if field == "guide_steps":
            steps = [s.strip() for s in (req.guide_steps or []) if s and s.strip()]
            values[field] = " | ".join(steps)
        elif field == "feedback_rating":
            val = req.feedback_rating or (str(req.rating) if req.rating is not None else "")
            values[field] = str(val or "").strip()
        else:
            values[field] = str(getattr(req, field, "") or "").strip()
    return values


def try_fetch_render_plan(
    req: GenerateRequest, category_field_values: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Calls the LLM render-plan sidecar (src/tendoo/llm_render_plan_server.py) with the
    whole category's field values (filled and blank) + free prompt. Returns the
    validated plan dict when usable, else None -- on ANY failure (sidecar unreachable,
    timeout, non-200, invalid/empty plan), logs a warning and returns None. NEVER
    raises: the caller must fall through to the deterministic
    style_matcher.resolve_style_preset() path unchanged, exactly as if this function
    didn't exist, so a sidecar hiccup never turns into a failed poster request.
    """
    import urllib.error
    import urllib.request

    payload = {
        "category": req.category,
        "category_fields": category_field_values,
        "prompt": req.image_description or req.prompt_scene or "",
        "style_pref": req.style_pref or "auto",
        "primary_color": req.primary_color,
    }
    try:
        body = json.dumps(payload).encode("utf-8")
        http_req = urllib.request.Request(
            LLM_RENDER_PLAN_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_req, timeout=LLM_RENDER_PLAN_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
        print(f"  [Render Plan] Sidecar unavailable/failed ({e}); falling back to deterministic auto-match.")
        return None

    if not data.get("usable"):
        print(f"  [Render Plan] Sidecar returned an unusable plan (errors: {data.get('errors')}); falling back.")
        return None
    return data.get("plan")


def try_fetch_tendoo_v2_blocks(
    req: GenerateRequest, category_field_values: Dict[str, str]
) -> Tuple[List[Any], List[str]]:
    """Experimental v2 engine (2026-09-14, "wire thẳng v2 vào v1 ... dùng model đã
    host"): calls src/tendoo_v2/llm_client.py's hosted-LLM content curation stage
    (role/group/zone, NOT size/container/zone -- see schema.py's docstring for why)
    instead of the legacy render-plan sidecar. Store/contact fields are excluded from
    the LLM's authority here (STORE_INFO_KEYS in llm_client.py) exactly like the v1
    path's unconditional store_parts block further down in run_pipeline_inference --
    both engines render store info verbatim, neither ever hands it to a model.

    NEVER raises (call_llm_for_blocks already never raises internally; this wrapper
    is defense-in-depth): any failure returns ([], [error]) so the caller renders a
    store-info-only poster rather than failing the whole request, mirroring
    try_fetch_render_plan()'s "a sidecar hiccup never breaks generation" contract.
    """
    from tendoo_v2.llm_client import call_llm_for_blocks

    fields = {
        **category_field_values,
        "store_name": (req.store_name or req.brand or "").strip(),
        "phone": (req.phone or req.hotline or "").strip(),
        "address": (req.address or "").strip(),
    }
    free_prompt = req.image_description or req.prompt_scene or ""
    try:
        return call_llm_for_blocks(fields, free_prompt)
    except Exception as e:
        print(f"  [tendoo_v2] Engine call failed unexpectedly ({e}); falling back to store-info-only content.")
        return [], [str(e)]


# Role-ranked zone preference lists used to auto-place any omni block that wasn't
# given an explicit zone (either because the model correctly left it blank -- no
# position was requested -- or because a mandatory field needs placing).
#
# "center" deliberately excluded from every role's preference list: it's the
# zone diffusion needs clearest for the actual product/subject, and an auto-GUESSED
# placement (the LLM left this block's zone unset) should never compete for it --
# only an EXPLICIT zone="center" from the render plan (a deliberate user/LLM request)
# ever lands text there. It's still technically reachable as the absolute last resort
# in _auto_assign_zones()'s "any remaining zone" fallback below, once every other
# named zone is already taken.
_ZONE_BLOCK_PREFERENCE: Dict[str, List[str]] = {
    "xlarge": ["top_center", "top_left", "top_right", "bottom_center"],
    "large": ["top_center", "middle_left", "middle_right", "bottom_center"],
    "button": ["bottom_right", "bottom_center", "bottom_left"],
    "pill": ["top_left", "top_right", "bottom_left"],
    "card": ["middle_left", "middle_right", "bottom_center"],
    "medium": ["middle_left", "middle_right", "bottom_center"],
    "small": ["bottom_left", "bottom_right", "bottom_center"],
}


def _auto_assign_zones(blocks: List[Dict[str, Any]]) -> None:
    """Mutates `blocks` in place, filling in a `zone` for every block that doesn't
    already have one, via a size/container-ranked preference list that skips zones already
    claimed by an earlier (explicit or auto-assigned) block. Degrades gracefully past 8
    blocks by reusing the earliest-assigned zone. Always stays in outer perimeter zones,
    never assigning into center sanctuary or top_bar/bottom_bar."""
    from tendoo.engine.geometry import GRID_ZONE_NAMES
    from tendoo.engine.blocks import normalize_zone

    for b in blocks:
        if b.get("zone"):
            b["zone"] = normalize_zone(b["zone"])

    used: List[str] = [b["zone"] for b in blocks if b.get("zone")]
    for b in blocks:
        if b.get("zone"):
            continue
        size = b.get("size")
        container = b.get("container")
        candidates = _ZONE_BLOCK_PREFERENCE.get(size) or _ZONE_BLOCK_PREFERENCE.get(container, GRID_ZONE_NAMES)
        chosen = next((z for z in candidates if z not in used), None)
        if chosen is None:
            chosen = next((z for z in GRID_ZONE_NAMES if z not in used), None)
        if chosen is None:
            chosen = used[0] if used else GRID_ZONE_NAMES[0]
        b["zone"] = chosen
        used.append(chosen)


def build_poster_content(
    req: GenerateRequest,
    final_headline: str,
    final_offer_main: str,
    final_offer_sub: str,
    final_dates: str,
    qr_data_uri: str,
    free_text_blocks: Optional[List[Dict[str, Any]]] = None,
    qr_zone: Optional[str] = None,
) -> PosterContent:
    """Instantiates a complete, well-typed PosterContent from GenerateRequest.
    free_text_blocks/qr_zone: fed directly into OmniBlockLayout for dynamic adaptive composition.
    """
    cat = (req.category or "promo").lower().strip()
    return PosterContent(
        free_text_blocks=free_text_blocks or [],
        qr_zone=qr_zone,
        headline=final_headline,
        pre_header=req.pre_header,
        slogan=req.slogan,
        offer_main=final_offer_main,
        offer_sub=final_offer_sub,
        dates=final_dates,
        brand=req.store_name or req.brand or "",
        hotline=req.phone or req.hotline or "",
        address=req.address,
        website_link=req.website_link,
        qr_data_uri=qr_data_uri,
        applicable=req.applied_product or req.applicable or "",
        category=cat,
        text_effect=req.text_effect,
        font_family=req.font_family,
        # Product intro
        price=req.price,
        product_name=req.product_name,
        product_desc=req.product_desc,
        highlights=req.highlights,
        # Opening
        opening_date=req.opening_date or (final_dates if cat == "opening" else ""),
        opening_promo=req.opening_promo or (final_offer_main if cat == "opening" else ""),
        booking_contact=req.booking_contact or (req.phone or req.hotline if cat == "opening" else ""),
        # Feedback
        feedback_target=req.feedback_target,
        feedback_quote=req.feedback_quote,
        feedback_rating=req.feedback_rating,
        special_offer=req.special_offer,
        # Recruitment
        job_position=req.job_position,
        job_desc=req.job_desc,
        apply_deadline=req.apply_deadline or (final_dates if cat == "recruitment" else ""),
        apply_method=req.apply_method or (req.applied_product if cat == "recruitment" else ""),
        # Guide
        steps=req.guide_steps if req.guide_steps else [],
    )


def run_pipeline_inference(
    req: GenerateRequest,
    render_plan: Optional[Dict[str, Any]] = None,
    category_field_values: Optional[Dict[str, str]] = None,
    _render_plan_prefetched: bool = False,
    tendoo_v2_blocks: Optional[List[Any]] = None,
    tendoo_v2_errors: Optional[List[str]] = None,
    _tendoo_v2_prefetched: bool = False,
) -> Dict[str, Any]:
    """Synchronous core inference worker executed under INFER_LOCK.

    `render_plan`/`category_field_values`: normally pre-computed by `api_generate()`
    BEFORE `INFER_LOCK` is acquired (2026-09-12 fix -- see below) and passed straight
    through here. `_render_plan_prefetched=True` means the caller already resolved the
    render-plan question one way or the other (found one, or deliberately skipped/failed
    to fetch one) -- so this function must NOT attempt its own fetch even if
    `render_plan` came back None. Only when a caller invokes this function directly
    without going through `api_generate()` (e.g. a test, or a future non-HTTP caller)
    does it fall back to fetching the render plan itself, exactly as this function used
    to do unconditionally.
    """
    t_start = time.time()
    _prune_old_output_runs()  # keep OUTPUT_DIR bounded to MAX_OUTPUT_RUNS folders
    timestamp_str = str(int(time.time() * 1000))
    case_dir = OUTPUT_DIR / f"run_{timestamp_str}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Dimensions from Aspect Ratio
    if req.width is not None and req.height is not None:
        w, h = req.width, req.height
    else:
        ar = req.aspect_ratio or "1:1"
        if ar == "9:16":
            w, h = 576, 1024
        elif ar == "16:9":
            w, h = 1024, 576
        elif ar == "4:5":
            w, h = 816, 1024
        else:
            w, h = 1024, 1024

    cat = (req.category or "promo").lower().strip()

    # 2. Category field values -- every CATEGORY_FIELD_SLOTS[cat] field's real current
    # value (blank if unfilled). Computed unconditionally (not just when a render plan
    # is fetched): the mandatory-field/omni-block-building logic below needs it
    # regardless of whether the LLM stage ran at all.
    if category_field_values is None:
        category_field_values = _collect_category_field_values(req)

    # (2026-09-14) Which content-curation engine this request uses -- "v1" (default,
    # unchanged legacy render_plan sidecar) or "v2" (experimental src/tendoo_v2:
    # hosted-LLM role/group/zone curation + deterministic solver). Per-request
    # override via req.engine_version, else server-wide via TENDOO_DEFAULT_ENGINE env
    # var, else "v1". See GenerateRequest.engine_version's docstring.
    engine_version = (req.engine_version or os.environ.get("TENDOO_DEFAULT_ENGINE") or "v1").strip().lower()

    # 3. LLM Render Plan (Phase C v2 -- see llm_render_plan_server.py's module
    # docstring for the full rule). Only attempted when engine_version != "v2" (the
    # v2 engine has its own, separate curation call below, never both) AND a free
    # "prompt ảnh" was given AND the caller hasn't already forced an explicit layout
    # (back-compat/direct-API callers) -- empty prompt keeps the deterministic path
    # entirely unchanged.
    #
    # (2026-09-12 fix: `api_generate()` now pre-fetches this BEFORE acquiring
    # INFER_LOCK, since this is a blocking HTTP call to a separate, GPU-independent
    # sidecar process with a timeout of up to LLM_RENDER_PLAN_TIMEOUT_S (45s) -- running
    # it under the lock meant a slow/unreachable sidecar serialized every OTHER user's
    # GPU request behind it for no reason. `_render_plan_prefetched` tells us not to
    # redo that fetch here; only a direct caller that never went through api_generate()
    # falls back to the old inline-fetch behavior.)
    if engine_version != "v2" and not _render_plan_prefetched and not req.layout and req.image_description and req.image_description.strip():
        render_plan = try_fetch_render_plan(req, category_field_values)

    # v2 engine's own curation call -- same prefetch-before-lock convenience as
    # render_plan above; a direct caller that skipped api_generate() (tests, a future
    # non-HTTP caller) gets it fetched here instead.
    if engine_version == "v2" and not _tendoo_v2_prefetched:
        tendoo_v2_blocks, tendoo_v2_errors = try_fetch_tendoo_v2_blocks(req, category_field_values)

    # Style/font hints from the plan (still re-validated downstream against whatever
    # layout actually gets chosen -- never trust the model's style_hint blindly) --
    # only applied when the caller hasn't already forced an explicit value.
    if render_plan is not None:
        if render_plan.get("style_hint") and render_plan["style_hint"] != "auto" and not req.style_hint:
            req.style_hint = render_plan["style_hint"]
        if render_plan.get("font_key") and render_plan["font_key"] != "auto" and not req.font_family:
            req.font_family = render_plan["font_key"]

    # Per-field dedup/extraction resolution -- the actual duplicate-prevention AND
    # blank-field-extraction mechanism. The real req value always wins when a field is
    # already filled (fidelity + de-dup guarantee); the model's text is trusted only
    # for genuinely-blank fields (fills the gap when the whole brief was typed into the
    # free prompt instead of the form).
    field_values: Dict[str, Dict[str, Any]] = {}
    ad_hoc_blocks: List[Dict[str, Any]] = []
    qr_zone_request: Optional[str] = None
    if render_plan is not None:
        for b in render_plan.get("extra_blocks", []):
            field = b.get("field")
            zone = b.get("zone")
            if field == "qr":
                qr_zone_request = zone
                continue
            if field is None:
                ad_hoc_blocks.append(b)
            else:
                real_value = category_field_values.get(field, "")
                final_text = real_value if real_value else (b.get("text") or "")
                if final_text:
                    field_values[field] = {
                        "text": final_text,
                        "zone": zone,
                        "role": b.get("role") or DEFAULT_FIELD_ROLE.get(field, "body"),
                        "icon": b.get("icon"),
                        "color": b.get("color"),
                        # (2026-09-13 fix: trước đây "effect" bị bỏ qua ở đây dù được đọc lại
                        # bằng fv.get("effect") ngay dưới -- luôn None, hiệu ứng model đề xuất
                        # cho 1 field bắt buộc không bao giờ tới được HTML. size/container KHÔNG
                        # được thêm ở đây có chủ đích: 2 thuộc tính này luôn do
                        # `get_field_block_spec()` quyết định cho field bắt buộc (không để model
                        # tự quyết định thẩm mỹ field bắt buộc, chỉ nội dung/zone/effect/icon/color).
                        "effect": b.get("effect"),
                    }

    # Title resolution (2026-09-13 simplification -- "title" is no longer its own
    # LLM-decided concept; see llm_render_plan_server.py's module docstring for the
    # updated rationale):
    #   - genuinely ad-hoc scattered content (field:null blocks in the render plan)
    #     skips the mandatory title field entirely -- deliberately NOT triggered by a
    #     mere reposition of an existing field (e.g. "move the discount badge"), which
    #     doesn't itself imply the user abandoned the headline concept. A prompt that
    #     wants a DIFFERENT headline instead of the form title is handled the exact
    #     same way: the model just emits that text as an ad-hoc xlarge/none block like
    #     any other ad-hoc content -- no special-casing needed, since any ad-hoc
    #     content already suppresses the mandatory field and the ad-hoc block itself
    #     still renders (just via the ad-hoc-blocks loop below, not the title slot).
    #   - otherwise the filled form title wins verbatim.
    #   - otherwise (no ad-hoc content, no form title) the existing per-category
    #     default_hl fallback + Primary Headline Guard (layout.py::_extract_blocks)
    #     further below already guarantee a reasonable headline with zero LLM input
    #     required -- no "generate" step needed here.
    title_skipped = False
    if engine_version == "v2":
        # v2 owns the headline entirely via its own role="hero" block(s) further down
        # (plan_blocks construction) -- always suppress the mandatory-field/default
        # headline path (Primary Headline Guard in layout.py) exactly like an ad-hoc
        # render_plan block does, so the guard never inserts a SECOND, redundant
        # headline the LLM never asked for.
        final_title = None
        title_skipped = True
    elif ad_hoc_blocks:
        final_title = None
        title_skipped = True
    elif req.title and req.title.strip():
        final_title = req.title
    else:
        final_title = req.title or None

    # Write any LLM-resolved content back onto req's own fields for consistency & logging
    if final_title and final_title != req.title:
        req.title = final_title
    for field_name, fv in field_values.items():
        if field_name == "guide_steps":
            if not req.guide_steps:
                req.guide_steps = [s.strip() for s in fv["text"].split("|") if s.strip()]
        elif not getattr(req, field_name, ""):
            setattr(req, field_name, fv["text"])

    # 4. Layout / Style / Font Auto-Match
    # Pure Omni architecture: every request runs through OmniBlockLayout.
    _preset = resolve_style_preset(req.category, req.style_pref)
    req.layout = "omni"
    if not req.style_hint:
        req.style_hint = _preset["style_hint"]
    if not req.font_family:
        req.font_family = _preset["font_key"]

    # 5. Retrieve Layout (Pure OmniBlock)
    layout = get_layout("omni")

    dates_parts = []
    if req.date_start:
        dates_parts.append(f"Từ {req.date_start}")
    if req.date_end:
        dates_parts.append(f"Đến {req.date_end}")
    final_dates = " - ".join(dates_parts) if dates_parts else ""

    # Store Info blocks (Brand / Hotline / Address) cho bottom_bar hoặc top_bar
    st_brand = (req.store_name or req.brand or "").strip()
    st_phone = (req.phone or req.hotline or "").strip()
    st_addr = (req.address or "").strip()
    store_parts = []
    if st_brand:
        store_parts.append(st_brand)
    if st_phone:
        store_parts.append(f"Hotline: {st_phone}")
    if st_addr:
        store_parts.append(st_addr)

    user_prompt_lower = (req.image_description or req.prompt_scene or "").lower()
    store_top_triggers = [
        "cửa hàng nhảy lên đầu", "thông tin ở trên", "store ở trên",
        "store on top", "header bar", "store info on top", "thông tin lên đầu"
    ]
    store_zone = "top_bar" if any(trig in user_prompt_lower for trig in store_top_triggers) else "bottom_bar"

    # Build the final block list: prioritize resolved title first, followed by
    # category fields and ad-hoc extra content -- then deduplicate via Content Fingerprint.
    plan_blocks: Optional[List[Dict[str, Any]]] = []
    if engine_version == "v2":
        # (2026-09-14) v2 branch: tendoo_v2_blocks (role/group/zone, already curated
        # by the hosted LLM in try_fetch_tendoo_v2_blocks() above) get their actual
        # zone assigned by tendoo_v2's own deterministic solver (group co-location +
        # role-priority zone preference -- see solver.py), then translated into the
        # EXACT plan_blocks shape the rest of this function (mask generation, Chromium
        # render, diffusion) already expects -- see bridge.py's module docstring for
        # why this reuses v1's OmniBlockLayout/font-fitting instead of re-rendering.
        from tendoo_v2.solver import solve as tendoo_v2_solve
        from tendoo_v2.bridge import placed_units_to_plan_blocks

        placed_units = tendoo_v2_solve(tendoo_v2_blocks or [], width=w, height=h)
        plan_blocks = placed_units_to_plan_blocks(placed_units)
        if tendoo_v2_errors:
            print(f"  [tendoo_v2] {len(tendoo_v2_errors)} curation error(s): {tendoo_v2_errors}")
        if store_parts:
            plan_blocks.append({
                "text": "  •  ".join(store_parts),
                "zone": store_zone,
                "role": "brand_bar",
                "field": "store_info",
                "icon": "phone" if st_phone else "globe",
            })
        plan_qr_zone = None
    elif render_plan is None:
        from tendoo.engine.blocks import map_category_to_default_blocks
        fields_for_map = {
            "discount": req.discount or req.offer_main,
            "offer_main": req.discount or req.offer_main,
            "applied_product": req.applied_product or req.offer_sub,
            "offer_sub": req.applied_product or req.offer_sub,
            "date_start": req.date_start,
            "date_end": req.date_end,
            "dates": final_dates,
            "product_name": req.product_name,
            "product_desc": req.product_desc,
            "highlights": req.highlights,
            "price": req.price,
            "opening_date": req.opening_date,
            "opening_promo": req.opening_promo,
            "booking_contact": req.booking_contact,
            "feedback_target": req.feedback_target,
            "feedback_quote": req.feedback_quote,
            "feedback_rating": req.feedback_rating,
            "special_offer": req.special_offer,
            "job_position": req.job_position,
            "job_desc": req.job_desc,
            "apply_deadline": req.apply_deadline,
            "apply_method": req.apply_method,
            "guide_steps": " | ".join(req.guide_steps) if req.guide_steps else "",
        }
        st_dict = {}
        if st_brand:
            st_dict["store_name"] = st_brand
        if st_phone:
            st_dict["phone"] = st_phone
        if st_addr:
            st_dict["address"] = st_addr

        default_blocks = map_category_to_default_blocks(
            category=cat,
            title=final_title,
            fields=fields_for_map,
            store_info=st_dict if st_dict else None,
            spatial_override_zone=store_zone if st_dict else None,
        )
        plan_blocks = [b.to_dict() for b in default_blocks]
        plan_qr_zone = None
    else:
        from tendoo.engine.blocks import get_field_block_spec

        if not title_skipped and final_title:
            plan_blocks.append({"text": final_title, "zone": None, "size": "xlarge", "container": "none"})

        # (2026-09-13 fix -- Finding A: trước đây nhánh này tự đoán size/container/icon rời rạc
        # ngay tại chỗ và KHÔNG BAO GIỜ gán style_variant, khiến path có render_plan (rất phổ biến
        # -- bất cứ khi nào người dùng nhập "prompt ảnh") mất hết phần thẩm mỹ style_variant mà
        # path tất định `map_category_to_default_blocks()` vẫn có (viền nhũ luxury_tag, khung lịch
        # calendar_box, khung quote, rating_badge...). Giờ cả 2 path tra cùng 1 bảng
        # `get_field_block_spec()` (blocks.py) nên 1 field luôn render cùng 1 kiểu thị giác bất kể
        # có render_plan hay không. size/container/style_variant/icon-mặc-định do bảng quyết định
        # (field bắt buộc không để model tự quyết định thẩm mỹ); model vẫn được quyết effect/color
        # và icon riêng khi bảng không có icon mặc định cho field đó.
        for cat_field in CATEGORY_FIELD_SLOTS.get(cat, []):
            spec = get_field_block_spec(cat, cat_field)
            if cat_field in field_values:
                fv = field_values[cat_field]
                block = {
                    "text": fv["text"],
                    "zone": fv.get("zone"),
                    "size": spec.get("size", "medium"),
                    "container": spec.get("container", "card"),
                }
                if spec.get("style_variant"):
                    block["style_variant"] = spec["style_variant"]
                if spec.get("icon"):
                    block["icon"] = spec["icon"]
                elif fv.get("icon"):
                    block["icon"] = fv["icon"]
                if fv.get("effect"):
                    block["effect"] = fv["effect"]
                if fv.get("color"):
                    block["color"] = fv["color"]
                plan_blocks.append(block)
            elif cat_field == "guide_steps":
                # (2026-09-13 fix: thêm step_index -- thiếu trước đây khiến path render_plan
                # không hiển thị số thứ tự 01/02/03 mà path tất định vẫn có, xem layout.py's
                # `if b.step_index is not None` numbering logic.)
                for idx, step in enumerate((s for s in (req.guide_steps or []) if s and s.strip()), 1):
                    plan_blocks.append({"text": step.strip(), "zone": None, "size": "medium", "container": "card", "style_variant": "step_list", "step_index": idx})
            else:
                real_value = category_field_values.get(cat_field, "")
                if real_value:
                    block = {
                        "text": real_value, "zone": None,
                        "size": spec.get("size", "medium"),
                        "container": spec.get("container", "card"),
                    }
                    if spec.get("style_variant"):
                        block["style_variant"] = spec["style_variant"]
                    if spec.get("icon"):
                        block["icon"] = spec["icon"]
                    plan_blocks.append(block)

        # Content Dedup: If final_title is rendered as xlarge headline, don't re-render an identical
        # category field block (e.g. feedback_target) as a subordinate card block.
        if not title_skipped and final_title:
            norm_title = " ".join(re.sub(r"[^\w\s]", "", final_title.lower()).split())
            plan_blocks = [
                b for b in plan_blocks
                if " ".join(re.sub(r"[^\w\s]", "", b["text"].lower()).split()) != norm_title
            ]
            plan_blocks.insert(0, {"text": final_title, "zone": None, "size": "xlarge", "container": "none"})
        for b in ad_hoc_blocks:
            block = {
                "text": b["text"],
                "zone": b.get("zone"),
                "size": b.get("size") or "medium",
                "container": b.get("container") or "card",
                "effect": b.get("effect"),
            }
            if b.get("icon"):
                block["icon"] = b["icon"]
            if b.get("color"):
                block["color"] = b["color"]
            plan_blocks.append(block)

        if store_parts:
            plan_blocks.append({
                "text": "  •  ".join(store_parts),
                "zone": store_zone,
                "role": "brand_bar",
                "field": "store_info",
                "icon": "phone" if st_phone else "globe",
            })

        _auto_assign_zones(plan_blocks)
        plan_qr_zone = qr_zone_request

    # 6. Harmonize Field Values (Unified schema mapping)
    if cat == "product_intro":
        default_hl = "GIỚI THIỆU SẢN PHẨM"
    elif cat == "opening":
        default_hl = "TƯNG BỪNG KHAI TRƯƠNG"
    elif cat == "feedback":
        default_hl = "KHÁCH HÀNG NÓI GÌ VỀ TENDOO"
    elif cat == "recruitment":
        default_hl = "TENDOO TÌM ĐỒNG ĐỘI"
    elif cat == "guide":
        default_hl = "QUY TRÌNH HƯỚNG DẪN"
    else:
        default_hl = "ƯU ĐÃI ĐẶC BIỆT"

    if title_skipped:
        final_headline = ""
        req.title = ""
    else:
        final_headline = req.title or req.headline or default_hl
    final_offer_main = (
        req.discount
        or req.offer_main
        or req.price
        or req.opening_promo
        or req.job_position
        or ""
    )
    final_offer_sub = (
        req.applied_product
        or req.offer_sub
        or req.product_desc
        or req.booking_contact
        or req.job_desc
        or req.feedback_quote
        or ""
    )
    # (2026-09-12 cleanup: removed dead `final_brand`/`final_hotline` locals here --
    # computed but never referenced again; the real brand/hotline values are correctly
    # recomputed inside build_poster_content() below.)
    # A usable render plan's scene_prompt (LLM-refined, already zero-text by
    # construction) takes precedence over the raw user prompt; still runs through the
    # same sanitize/inject chain below as a defense-in-depth net either way.
    raw_scene_prompt = (
        (render_plan.get("scene_prompt") if render_plan else "")
        or req.image_description or req.prompt_scene or ""
    )
    has_ref_image = bool(req.image_base64)
    zero_text_prompt = sanitize_and_inject_zero_text(raw_scene_prompt, has_ref_image=has_ref_image)
    final_scene_prompt = inject_spatial_layout_guidance(zero_text_prompt, layout_name=layout.name, has_ref_image=has_ref_image)
    final_scene_prompt = inject_color_guidance(final_scene_prompt, req.primary_color)
    if raw_scene_prompt != final_scene_prompt:
        print(f"  [Prompt Engine] Injected ZERO-TEXT & Spatial Guidance:\n    Raw: '{raw_scene_prompt}'\n    Final: '{final_scene_prompt}'")

    # User-picked "màu chủ đạo" hue override, threaded into analyze_color_harmony()
    # below (CSS accent palette) -- None when unset/unparseable, i.e. today's 100%
    # auto-detected-from-pixels behavior is unchanged.
    user_hue = hex_to_hue(req.primary_color) if req.primary_color else None

    style_hint = detect_scene_lighting_tone(final_scene_prompt, req.style_hint or "auto", layout_name=layout.name)

    # 7. Optional QR Code Generation (Backend QR Engine)
    qr_data_uri = ""
    qr_url = None
    if req.enable_qr and req.website_link and req.website_link.strip():
        from tendoo.qr import generate_qr_base64
        qr_data_uri = generate_qr_base64(req.website_link.strip()) or ""
        if qr_data_uri:
            try:
                qr_raw = base64.b64decode(qr_data_uri.split(",")[1])
                qr_file = case_dir / "00_qr_code.png"
                qr_file.write_bytes(qr_raw)
                qr_url = f"outputs/{case_dir.name}/00_qr_code.png"
            except Exception as e:
                print(f"Notice: Failed to save QR code image: {e}")

    # 8. Build PosterContent ONCE (identical across the mock/real branches below and
    # across every num_images iteration -- none of these fields depend on the
    # background/palette, only on req/plan_blocks/qr_data_uri, all resolved by now).
    content = build_poster_content(
        req=req,
        final_headline=final_headline,
        final_offer_main=final_offer_main,
        final_offer_sub=final_offer_sub,
        final_dates=final_dates,
        qr_data_uri=qr_data_uri,
        free_text_blocks=plan_blocks,
        qr_zone=plan_qr_zone,
    )

    # 9. Generate Parametric Corridor Mask
    # Preferred path: render this exact `content` through the SAME Chromium engine that
    # will later draw the final pixels (with a neutral placeholder background -- colors
    # don't affect text geometry) and read back its real getBoundingClientRect() rects
    # after the in-page autofit script settles. This is what makes the mask match the
    # HTML text exactly, at any output size -- see OmniBlockLayout.generate_mask_from_render.
    # Falls back to the legacy PIL-estimated mask (generate_mask) only if the browser
    # measurement pass fails outright (e.g. Playwright unavailable), so the pipeline
    # degrades gracefully instead of crashing.
    mask_np = None
    if hasattr(layout, "generate_mask_from_render") and hasattr(layout, "render_html"):
        try:
            _dummy_palette = ColorPalette(
                is_dark=True, luminance=0.5, hue=0, comp_hue=180, headline_color="#FFFFFF",
            )
            # Colors/pixels of this placeholder never influence text geometry (CSS
            # `background-size:cover` on a 1x1 image is purely a fill color); only
            # `content`/`width`/`height`/font/CSS drive layout, and those are identical
            # to the real render below -- so the rects measured against this
            # placeholder are exactly the rects the final render will occupy.
            _neutral_bg_data_uri = pil_to_base64_data_uri(Image.new("RGB", (1, 1), (60, 60, 60)))
            _measure_html = layout.render_html(
                content=content,
                palette=_dummy_palette,
                bg_data_uri=_neutral_bg_data_uri,
                width=w,
                height=h,
                headline_effect=req.text_effect,
                style_hint=style_hint,
            )
            mask_np = layout.generate_mask_from_render(_measure_html, width=w, height=h)
        except Exception as e:
            print(f"  [!] Warning: browser-measured mask generation failed, falling back to PIL estimate: {e}")
            mask_np = None

    if mask_np is None:
        if plan_blocks is not None:
            mask_np = layout.generate_mask(
                width=w, height=h, blocks=plan_blocks, qr_zone=plan_qr_zone,
                qr_data_uri=qr_data_uri or None,
                font_key=req.font_family if req.font_family != "auto" else "bevietnam",
            )
        else:
            mask_np = layout.generate_mask(width=w, height=h, qr_data_uri=qr_data_uri or None, qr_zone=plan_qr_zone)

    mask_vis = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    mask_file = case_dir / "01_corridor_mask.png"
    mask_vis.save(mask_file)

    # 10. Save Uploaded Product Image if provided
    ref_image_path = None
    if req.image_base64:
        try:
            b64_clean = req.image_base64
            if "," in b64_clean:
                b64_clean = b64_clean.split(",")[1]
            ref_raw = base64.b64decode(b64_clean)
            ref_file = case_dir / "00_uploaded_product.png"
            ref_file.write_bytes(ref_raw)
            ref_image_path = ref_file
            print(f"[✓] Saved uploaded product reference image: {ref_file.name}")
        except Exception as e:
            print(f"Notice: Failed to save uploaded product image: {e}")

    # 11. Generate Blended Background & Final Poster (Regional Velocity Blending + HTML Typography)
    num_images = max(1, min(4, req.num_images or 1))
    posters_list = []
    dur_dit_total = 0.0

    if IS_MOCK_MODE or DIT_MODEL is None:
        # Mock mode for testing without GPU
        for img_idx in range(num_images):
            seed_i = req.seed + img_idx * 1000
            np.random.seed(seed_i % (2**32))
            blended_pil = Image.fromarray((np.random.rand(h, w, 3) * 30 + 15).astype(np.uint8))

            bg_filename = f"02_blended_background_{img_idx}.png"
            blended_file = case_dir / bg_filename
            blended_pil.save(blended_file)
            if img_idx == 0:
                blended_pil.save(case_dir / "02_blended_background.png")

            safe_zone = layout.get_safe_zone()
            palette = analyze_color_harmony(np.array(blended_pil), safe_zone, color_mode="auto", user_hue=user_hue)

            bg_data_uri = pil_to_base64_data_uri(blended_pil)
            # `content` (step 8) is identical across every img_idx -- built once above.
            html_str = layout.render_html(
                content=content,
                palette=palette,
                bg_data_uri=bg_data_uri,
                width=w,
                height=h,
                style_hint=style_hint,
            )
            html_file = case_dir / f"03_poster_{img_idx}.html"
            html_file.write_text(html_str, encoding="utf-8")
            if img_idx == 0:
                (case_dir / "03_poster.html").write_text(html_str, encoding="utf-8")

            poster_filename = f"04_final_poster_{img_idx}.png"
            final_poster_file = case_dir / poster_filename
            PosterRenderer.render(
                html_content=html_str,
                output_image_path=final_poster_file,
                width=w,
                height=h,
            )
            if img_idx == 0:
                shutil.copyfile(final_poster_file, case_dir / "04_final_poster.png")

            posters_list.append({
                "index": img_idx,
                "seed": seed_i,
                "final_poster_url": f"outputs/{case_dir.name}/{poster_filename}",
                "blended_bg_url": f"outputs/{case_dir.name}/{bg_filename}",
                "html_url": f"outputs/{case_dir.name}/03_poster_{img_idx}.html",
            })
        dur_dit_total = 0.5 * num_images
    else:
        # Real DiT inference on 2x A30
        from flux2.sampling import get_schedule, prc_img, prc_txt
        from tendoo.velocity_blending import denoise_regional_velocity_blended, load_and_encode_ref_image

        prompt_corr = req.prompt_corridor or layout.get_corridor_prompt(style_hint)
        prompt_corr = inject_color_guidance(prompt_corr, req.primary_color)

        # Encode prompts with Qwen3 ONCE outside loop
        with torch.no_grad():
            ctx_scene = TEXT_ENCODER([final_scene_prompt]).to(torch.bfloat16)
            ctx_scene, ctx_scene_ids = prc_txt(ctx_scene[0])
            ctx_scene = ctx_scene.unsqueeze(0).to(DEVICE_DIT)
            ctx_scene_ids = ctx_scene_ids.unsqueeze(0).to(DEVICE_DIT)

            ctx_corridor = TEXT_ENCODER([prompt_corr]).to(torch.bfloat16)
            ctx_corridor, ctx_corridor_ids = prc_txt(ctx_corridor[0])
            ctx_corridor = ctx_corridor.unsqueeze(0).to(DEVICE_DIT)
            ctx_corridor_ids = ctx_corridor_ids.unsqueeze(0).to(DEVICE_DIT)

            # Spatial mask tensor
            mask_scaled = Image.fromarray(mask_np).resize((w // 16, h // 16), Image.Resampling.BICUBIC)
            mask_flat = torch.from_numpy(np.array(mask_scaled)).float().reshape(1, -1, 1).to(DEVICE_DIT)

            # Encode optional Reference Product Image (In-Context RoPE offset t=10.0) ONCE outside loop
            ref_toks = None
            ref_ids = None
            if ref_image_path and ref_image_path.exists():
                print("  [Ref Product] Encoding uploaded product into VAE latents...")
                try:
                    r_toks, r_ids = load_and_encode_ref_image(
                        ref_image_path=ref_image_path,
                        ae=AE_MODEL,
                        device=DEVICE_DIT,
                        ae_device=DEVICE_AUX,
                        target_dim=512,
                        time_offset=10.0,
                    )
                    ref_toks = r_toks.to(dtype=torch.bfloat16)
                    ref_ids = r_ids
                    print(f"  [✓] In-Context Product Reference attached at RoPE t=10.0 (shape: {ref_toks.shape})")
                except Exception as e:
                    print(f"  [!] Warning: Failed to encode reference product image: {e}")

            w_lat, h_lat = w // 16, h // 16

            for img_idx in range(num_images):
                seed_i = req.seed + img_idx * 1000
                torch.manual_seed(seed_i)
                z_init = torch.randn(1, 128, h_lat, w_lat, device=DEVICE_DIT, dtype=torch.bfloat16)
                img_tokens, img_ids = prc_img(z_init[0])
                img_tokens = img_tokens.unsqueeze(0).to(DEVICE_DIT)
                img_ids = img_ids.unsqueeze(0).to(DEVICE_DIT)

                # Concatenate canvas noise + product reference tokens
                if ref_toks is not None and ref_ids is not None:
                    img_total = torch.cat([img_tokens, ref_toks], dim=1)
                    img_ids_total = torch.cat([img_ids, ref_ids], dim=1)
                else:
                    img_total = img_tokens
                    img_ids_total = img_ids

                timesteps = get_schedule(num_steps=req.steps, image_seq_len=img_tokens.shape[1])

                t0_dit = time.time()
                out_blended = denoise_regional_velocity_blended(
                    model=DIT_MODEL,
                    img=img_total,
                    img_ids=img_ids_total,
                    txt_scene=ctx_scene,
                    txt_scene_ids=ctx_scene_ids,
                    txt_corridor=ctx_corridor,
                    txt_corridor_ids=ctx_corridor_ids,
                    spatial_mask=mask_flat,
                    timesteps=timesteps,
                    guidance=req.guidance,
                    num_canvas_tokens=img_tokens.shape[1],
                )
                dur_dit = time.time() - t0_dit
                dur_dit_total += dur_dit

                z_dec = out_blended[0].transpose(0, 1).reshape(1, 128, h_lat, w_lat).to(device=DEVICE_AUX, dtype=AE_DTYPE)
                x_dec = AE_MODEL.decode(z_dec).float()
                x_arr = ((x_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
                blended_pil = Image.fromarray(x_arr)

                bg_filename = f"02_blended_background_{img_idx}.png"
                blended_file = case_dir / bg_filename
                blended_pil.save(blended_file)
                if img_idx == 0:
                    blended_pil.save(case_dir / "02_blended_background.png")

                safe_zone = layout.get_safe_zone()
                palette = analyze_color_harmony(np.array(blended_pil), safe_zone, color_mode="auto", user_hue=user_hue)

                bg_data_uri = pil_to_base64_data_uri(blended_pil)
                # `content` (step 8) is identical across every img_idx -- built once above.
                html_str = layout.render_html(
                    content=content,
                    palette=palette,
                    bg_data_uri=bg_data_uri,
                    width=w,
                    height=h,
                    headline_effect=req.text_effect,
                    style_hint=style_hint,
                )
                html_file = case_dir / f"03_poster_{img_idx}.html"
                html_file.write_text(html_str, encoding="utf-8")
                if img_idx == 0:
                    (case_dir / "03_poster.html").write_text(html_str, encoding="utf-8")

                poster_filename = f"04_final_poster_{img_idx}.png"
                final_poster_file = case_dir / poster_filename
                PosterRenderer.render(
                    html_content=html_str,
                    output_image_path=final_poster_file,
                    width=w,
                    height=h,
                )
                if img_idx == 0:
                    shutil.copyfile(final_poster_file, case_dir / "04_final_poster.png")

                posters_list.append({
                    "index": img_idx,
                    "seed": seed_i,
                    "final_poster_url": f"outputs/{case_dir.name}/{poster_filename}",
                    "blended_bg_url": f"outputs/{case_dir.name}/{bg_filename}",
                    "html_url": f"outputs/{case_dir.name}/03_poster_{img_idx}.html",
                })

    total_latency = round(time.time() - t_start, 2)
    rel_case = f"outputs/{case_dir.name}"
    return {
        "success": True,
        "latency_s": total_latency,
        "dit_latency_s": round(dur_dit_total, 2),
        "final_poster_url": f"{rel_case}/04_final_poster.png",
        "blended_bg_url": f"{rel_case}/02_blended_background.png",
        "mask_url": f"{rel_case}/01_corridor_mask.png",
        "html_url": f"{rel_case}/03_poster.html",
        "qr_url": qr_url,
        "category": req.category,
        "posters": posters_list,
        "num_images": len(posters_list),
        # Exposes the Phase B auto-match's actual decision -- lets tests (and curious
        # callers) verify layout/style_hint without parsing the rendered HTML/CSS.
        "resolved_layout": req.layout or layout.name,
        "resolved_style_hint": style_hint,
        "engine": engine_version,
        "tendoo_v2_errors": tendoo_v2_errors if engine_version == "v2" else None,
    }


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """
    Thread-safe poster generation endpoint protected by INFER_LOCK.
    Queues simultaneous requests smoothly without GPU VRAM collision.
    """
    # Phase A's required-field gate only applies to the deterministic (no-prompt) form
    # flow. When a free "prompt ảnh" is given, Phase C's render-plan LLM is a valid
    # alternative way to supply that same required content (prompt_test.txt lines
    # 21-41's whole-brief-in-prompt shape depends on this) -- rejecting the request
    # before the LLM stage even runs would defeat that path's entire purpose. If the
    # LLM's extraction fails to produce the content, that surfaces as a poster missing
    # that field's slot, not a pre-flight validation error.
    if not (req.image_description and req.image_description.strip()):
        missing = validate_required_fields(req)
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "missing_required_fields",
                    "category": req.category,
                    "fields": missing,
                },
            )

    # Pre-fetch the LLM render-plan (if applicable) BEFORE acquiring INFER_LOCK.
    #
    # (2026-09-12 fix: `try_fetch_render_plan` is a blocking HTTP call to a separate,
    # GPU-independent sidecar process with a timeout of up to LLM_RENDER_PLAN_TIMEOUT_S
    # (45s). It used to run inside `run_pipeline_inference()`, which is only ever
    # invoked from here AFTER `INFER_LOCK` is held -- so a slow/unreachable sidecar
    # blocked every OTHER user's GPU request behind it for up to 45s, even though
    # nothing about this step touches the GPU. Fetching it here, before the lock, means
    # the lock is only ever held for the actual GPU-bound portion of the pipeline.)
    loop = asyncio.get_running_loop()
    category_field_values = _collect_category_field_values(req)
    engine_version = (req.engine_version or os.environ.get("TENDOO_DEFAULT_ENGINE") or "v1").strip().lower()

    render_plan: Optional[Dict[str, Any]] = None
    tendoo_v2_blocks: Optional[List[Any]] = None
    tendoo_v2_errors: Optional[List[str]] = None
    if engine_version == "v2":
        # (2026-09-14) Same "fetch the slow, GPU-independent LLM call BEFORE the lock"
        # rationale as the render_plan branch below -- see its comment for the full
        # story (2026-09-12 fix).
        tendoo_v2_blocks, tendoo_v2_errors = await loop.run_in_executor(
            None, try_fetch_tendoo_v2_blocks, req, category_field_values
        )
    elif not req.layout and req.image_description and req.image_description.strip():
        render_plan = await loop.run_in_executor(
            None, try_fetch_render_plan, req, category_field_values
        )

    async with INFER_LOCK:
        try:
            result = await loop.run_in_executor(
                None, run_pipeline_inference, req, render_plan, category_field_values, True,
                tendoo_v2_blocks, tendoo_v2_errors, True,
            )
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="Tendoo AI Demo Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default 7860)")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without loading GPU weights")
    return parser.parse_args()


def main():
    global IS_MOCK_MODE
    args = parse_args()
    if args.mock:
        IS_MOCK_MODE = True

    # Register exit hooks
    def sig_handler(sig, frame):
        free_all_gpu_memory()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print(f"\n🌐 Starting Tendoo Demo Server on http://{args.host}:{args.port}")
    print("💡 Colleagues can connect via http://<server-ip>:7860 or JupyterLab Proxy: /proxy/7860/\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
