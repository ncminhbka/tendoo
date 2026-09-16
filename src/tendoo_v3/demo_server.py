#!/usr/bin/env python3
"""
src/tendoo_v3/demo_server.py

TENDOO AI v3 - HIGH-PERFORMANCE MULTI-USER DEMO SERVER
======================================================
1. Long-running FastAPI daemon: Models remain warm in VRAM on 2x NVIDIA A30.
2. Concurrency Lock: `asyncio.Lock()` ngăn chặn va chạm bộ nhớ VRAM khi nhiều người dùng gửi request.
3. Tích hợp trọn vẹn quy trình Tendoo v3:
   - LLM Planner (Qwen3.8-27B) làm Giám đốc Sáng tạo chọn template & biên tập nội dung.
   - Continuous Mask Engine làm mờ Gaussian loang mượt bảo vệ không gian text.
   - Single-Pass Regional Velocity Blending ODE flow matching trên GPU (hoặc Mock Backdrop trên CPU/Local).
   - Playwright Chromium Renderer tổng hợp đồ họa, Font Unicode offline, Semantic SVG Icons và Scannable QR.
4. Clean Shutdown: Giải phóng triệt để VRAM trên cả 2 GPU khi tắt server.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import io
import json
import logging
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image
from pydantic import BaseModel
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from tendoo.core.colors import analyze_color_harmony
from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.geometry import get_zones
from tendoo_v3.mask_engine import generate_template_mask
from tendoo_v3.llm_planner import (
    generate_creative_plan,
    free_local_qwen3,
    load_local_qwen3,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_BACKEND,
)
from tendoo_v3.qr import generate_qr_base64
from tendoo_v3.renderer import build_template_html, pil_to_base64_data_uri, render_plan_to_poster
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.styles import palette_from_color_harmony
from tendoo_v3.velocity_blending import (
    denoise_regional_velocity_blended,
    generate_mock_backdrop,
    load_and_encode_ref_image,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TendooV3.Server")

# Global State
DIT_MODEL: Any = None
AE_MODEL: Any = None
TEXT_ENCODER: Any = None
AE_DTYPE: Any = None
DEVICE_DIT: str = os.environ.get("TENDOO_V3_DEVICE_DIT", "cuda:0")
DEVICE_AUX: str = os.environ.get("TENDOO_V3_DEVICE_AUX", "cuda:1")
INFER_LOCK = asyncio.Lock()
IS_MOCK_MODE: bool = not torch.cuda.is_available() or os.environ.get("MOCK_MODE", "").lower() in ["1", "true"]
ACTIVE_MODEL_NAME: str = "flux.2-klein-4b"

OUTPUT_RUNS_DIR = PROJECT_ROOT / "output_tendoo_v3" / "demo_runs"
OUTPUT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
UI_HTML_PATH = Path(__file__).resolve().parent / "demo_ui.html"

MAX_OUTPUT_RUNS = int(os.environ.get("TENDOO_V3_MAX_RUNS", "100"))

# Aspect ratio mapper
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "9:16": (576, 1024),
    "16:9": (1024, 576),
    "4:5": (816, 1024),
    "2:3": (680, 1024),
    "4:3": (1024, 768),
}


def free_all_gpu_memory():
    """Giải phóng triệt để VRAM trên toàn bộ GPU CUDA."""
    global DIT_MODEL, AE_MODEL, TEXT_ENCODER, AE_DTYPE
    logger.info("Releasing GPU memory...")
    del DIT_MODEL
    del AE_MODEL
    del TEXT_ENCODER
    DIT_MODEL = None
    AE_MODEL = None
    TEXT_ENCODER = None
    AE_DTYPE = None
    free_local_qwen3()

    gc.collect()
    if torch.cuda.is_available():
        for dev_id in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(dev_id):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                alloc = torch.cuda.memory_allocated(dev_id) / (1024 ** 2)
                res = torch.cuda.memory_reserved(dev_id) / (1024 ** 2)
                logger.info(f"  [GPU {dev_id}] Allocated: {alloc:.1f} MB | Reserved: {res:.1f} MB")
            except Exception as e:
                logger.warning(f"  [GPU {dev_id}] Cleanup notice: {e}")
    logger.info("GPU VRAM released.")


def _ensure_aspect_compatible(template: str, aspect_ratio: str) -> str:
    """Nếu template được chọn (bởi LLM, fallback heuristic, hoặc override thủ công)
    không khai báo hỗ trợ đúng tỉ lệ khung hình đang yêu cầu (`aspect_ratios` trong
    TEMPLATE_CATALOG), đổi sang template khác CÓ hỗ trợ -- thay vì render một bố cục
    chưa từng được thiết kế/đo cho tỉ lệ đó (đã bắt được thật: sandwich_top_heavy chỉ
    khai báo 1:1/9:16/4:5 nhưng vẫn bị chọn cho canvas 16:9, khiến QR label bị đẩy
    hẳn ra ngoài khung hình, vô hình 100% trong ảnh cuối). Nếu KHÔNG template nào
    trong catalog khai báo hỗ trợ tỉ lệ này (vd 2:3/4:3, hiện chưa có template nào
    liệt kê), giữ nguyên lựa chọn ban đầu thay vì đổi mà không có gì tốt hơn."""
    info = TEMPLATE_CATALOG.get(template)
    if info and aspect_ratio in info.get("aspect_ratios", []):
        return template
    for candidate, candidate_info in TEMPLATE_CATALOG.items():
        if aspect_ratio in candidate_info.get("aspect_ratios", []):
            logger.warning(
                f"Template '{template}' không khai báo hỗ trợ tỉ lệ {aspect_ratio} "
                f"-> đổi sang '{candidate}' để tránh tràn/cắt khung."
            )
            return candidate
    return template


def _extract_primary_crop_zone(
    template: str, width: int, height: int, orientation: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """Tìm normalized (y1, x1, y2, x2) của vùng chữ chính để phân tích Chameleon Palette."""
    zones = get_zones(template, width, height, orientation)
    if not zones:
        return (0.0, 0.0, 0.38, 1.0)

    # Ưu tiên zone có hero text
    priority_keys = ["content", "top", "main", "card", "board", "bottom"]
    selected_rect = None
    for k in priority_keys:
        if k in zones:
            selected_rect = zones[k]
            break
    if not selected_rect:
        # Lấy rect có diện tích lớn nhất
        selected_rect = max(zones.values(), key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))

    x1, y1, x2, y2 = selected_rect
    # Chuyển đổi px sang normalized (y1, x1, y2, x2) trong khoảng [0..1]
    norm_y1 = max(0.0, min(1.0, y1 / height))
    norm_x1 = max(0.0, min(1.0, x1 / width))
    norm_y2 = max(0.0, min(1.0, y2 / height))
    norm_x2 = max(0.0, min(1.0, x2 / width))
    return (norm_y1, norm_x1, norm_y2, norm_x2)


def _prune_old_runs():
    """Giới hạn số lượng run folders để không làm tràn ổ cứng."""
    try:
        run_dirs = sorted(
            [d for d in OUTPUT_RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda p: p.stat().st_mtime,
        )
        if len(run_dirs) > MAX_OUTPUT_RUNS:
            for old_dir in run_dirs[:-MAX_OUTPUT_RUNS]:
                shutil.rmtree(old_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Error pruning old runs: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời khởi động và dọn dẹp bộ nhớ server."""
    global DIT_MODEL, AE_MODEL, TEXT_ENCODER, AE_DTYPE, DEVICE_DIT, DEVICE_AUX, IS_MOCK_MODE, ACTIVE_MODEL_NAME

    active_backend = os.environ.get("TENDOO_V3_LLM_BACKEND", LLM_BACKEND).lower()
    logger.info("=" * 70)
    logger.info("🚀 STARTING TENDOO v3 STUDIO DEMO SERVER")
    logger.info(f"   Mock Mode: {IS_MOCK_MODE} | CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"   LLM Backend:      {active_backend.upper()}")
    if active_backend in ["api", "auto"]:
        logger.info(f"   Active LLM Model: {LLM_MODEL}")
        logger.info(f"   LLM Base URL:     {LLM_BASE_URL}")
        key_status = f"CONFIGURED (***{LLM_API_KEY[-4:]})" if LLM_API_KEY else "NOT SET (Failover to Local Qwen3 / Heuristic)"
        logger.info(f"   LLM API Key:      {key_status}")
    if active_backend in ["local", "auto"]:
        logger.info(f"   Local LLM:        Qwen3-4B on GPU (persistent-data)")
    logger.info("=" * 70)

    if not IS_MOCK_MODE and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            DEVICE_DIT = os.environ.get("TENDOO_V3_DEVICE_DIT", "cuda:0")
            DEVICE_AUX = os.environ.get("TENDOO_V3_DEVICE_AUX", "cuda:1")
            logger.info(f"  [Device Setup] Multi-GPU: DiT on {DEVICE_DIT}, AE & Qwen3 on {DEVICE_AUX}")
        else:
            DEVICE_DIT = DEVICE_AUX = os.environ.get("TENDOO_V3_DEVICE_DIT", "cuda:0")
            logger.info(f"  [Device Setup] Single GPU: All models on {DEVICE_DIT}")

        from flux2 import util

        pdata_candidates = [
            Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-4B")),
            Path("/home/jovyan/persistent-data/FLUX.2-klein-4B"),
            Path("/persistent-data/FLUX.2-klein-4B"),
            Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B")),
            Path("/home/jovyan/persistent-data/FLUX.2-klein-base-4B"),
            Path("/persistent-data/FLUX.2-klein-base-4B"),
        ]

        # Mặc định ưu tiên FLUX.2-klein-4B (bản Distilled, 4-8 bước, nhanh và đẹp cho Studio Demo)
        force_base = os.environ.get("TENDOO_USE_BASE_MODEL", "").lower() in ["1", "true"]
        model_name = None

        if not force_base:
            for pdata in pdata_candidates:
                distill_cand = pdata / "flux-2-klein-4b.safetensors"
                if distill_cand.exists():
                    model_name = "flux.2-klein-4b"
                    os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_cand)
                    logger.info(f"✅ Found local Distilled DiT weights (FLUX.2-klein-4B): {distill_cand}")
                    break

        if model_name is None:
            for pdata in pdata_candidates:
                base_cand = pdata / "flux-2-klein-base-4b.safetensors"
                if base_cand.exists():
                    model_name = "flux.2-klein-base-4b"
                    os.environ["KLEIN_4B_BASE_MODEL_PATH"] = str(base_cand)
                    logger.info(f"Found local Base DiT weights (FLUX.2-klein-base-4B): {base_cand}")
                    break

        if model_name is None:
            model_name = "flux.2-klein-4b"

        # Tự động tìm VAE và TextEncoder qua các thư mục persistent-data nếu chưa được gán
        if "AE_MODEL_PATH" not in os.environ:
            for pdata in pdata_candidates:
                vae_cand = pdata / "vae" / "diffusion_pytorch_model.safetensors"
                if vae_cand.exists():
                    os.environ["AE_MODEL_PATH"] = str(vae_cand)
                    logger.info(f"✅ Found local VAE weights: {vae_cand}")
                    break

        if "TEXT_ENCODER_PATH" not in os.environ:
            for pdata in pdata_candidates:
                te_cand = pdata / "text_encoder"
                if te_cand.exists() and ((te_cand / "config.json").exists() or (te_cand / "model.safetensors.index.json").exists()):
                    os.environ["TEXT_ENCODER_PATH"] = str(te_cand)
                    logger.info(f"✅ Found local Qwen3 weights: {te_cand}")
                    break

        ACTIVE_MODEL_NAME = model_name
        logger.info("=" * 80)
        logger.info(f"🚀 INITIALIZING {ACTIVE_MODEL_NAME.upper()} FOR TENDOO V3 DEMO SERVER")
        logger.info("=" * 80)

        t0 = time.time()
        try:
            logger.info(f"⏳ Loading DiT weights into VRAM for [{ACTIVE_MODEL_NAME}]...")
            DIT_MODEL = util.load_flow_model(ACTIVE_MODEL_NAME, device=DEVICE_DIT)
            DIT_MODEL.eval()

            logger.info(f"⏳ Loading AutoEncoder (VAE) weights for [{ACTIVE_MODEL_NAME}]...")
            AE_MODEL = util.load_ae(ACTIVE_MODEL_NAME, device=DEVICE_AUX)
            AE_MODEL.eval()
            AE_DTYPE = next(AE_MODEL.parameters()).dtype

            logger.info(f"⏳ Loading Qwen3 TextEncoder weights for [{ACTIVE_MODEL_NAME}]...")
            TEXT_ENCODER = util.load_text_encoder(ACTIVE_MODEL_NAME, device=DEVICE_AUX)

            dur_init = time.time() - t0
            logger.info(f"✅ All Diffusion models loaded and warm in VRAM in {dur_init:.1f}s!")

            if active_backend == "local":
                logger.info(f"⏳ Pre-loading Local Qwen3-4B CausalLM into VRAM on {DEVICE_AUX}...")
                try:
                    load_local_qwen3(device=DEVICE_AUX)
                except Exception as e:
                    logger.warning(f"Could not preload local Qwen3: {e}")
            for dev_id in range(num_gpus):
                alloc = torch.cuda.memory_allocated(dev_id) / (1024 ** 2)
                res = torch.cuda.memory_reserved(dev_id) / (1024 ** 2)
                logger.info(f"  [GPU {dev_id}] Occupied: {alloc:.1f} MB (Reserved: {res:.1f} MB)")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"❌ Failed to load GPU models ({e}). Falling back to Mock Mode.", exc_info=True)
            free_all_gpu_memory()
            IS_MOCK_MODE = True

    yield

    free_all_gpu_memory()


app = FastAPI(title="Tendoo AI v3 Studio", version="3.0.0", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_RUNS_DIR)), name="outputs")

# Per-category required fields (matching business specification)
CATEGORY_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "promo": ["discount"],
    "product_intro": ["product_name", "product_desc"],
    "opening": ["opening_date"],
    "feedback": ["feedback_target", "feedback_quote"],
    "recruitment": ["job_position", "apply_deadline", "apply_method"],
    "guide": [],
}


class GenerateRequest(BaseModel):
    category: str = "promo"

    # Common Store / Brand Info
    store_name: Optional[str] = ""
    phone: Optional[str] = ""
    address: Optional[str] = ""
    website_link: Optional[str] = ""
    enable_qr: bool = False

    # Promo Category Fields
    title: Optional[str] = ""
    image_base64: Optional[str] = None
    ref_image_b64: Optional[str] = None
    discount: Optional[str] = ""
    applied_product: Optional[str] = ""
    date_start: Optional[str] = ""
    date_end: Optional[str] = ""
    image_description: Optional[str] = ""

    # Product Intro
    product_name: Optional[str] = ""
    price: Optional[str] = ""
    product_desc: Optional[str] = ""
    highlights: Optional[str] = ""

    # Grand Opening
    opening_date: Optional[str] = ""
    opening_promo: Optional[str] = ""
    booking_contact: Optional[str] = ""

    # Customer Feedback
    feedback_target: Optional[str] = ""
    feedback_quote: Optional[str] = ""
    feedback_highlights: Optional[str] = ""
    feedback_rating: Optional[Any] = ""
    special_offer: Optional[str] = ""
    customer_name: Optional[str] = None

    # Recruitment
    job_position: Optional[str] = ""
    job_desc: Optional[str] = ""
    apply_deadline: Optional[str] = ""
    apply_method: Optional[str] = ""

    # Usage Guide
    guide_steps: Optional[List[str]] = None

    # Visual / Design Controls
    style_pref: Optional[str] = "auto"
    primary_color: Optional[str] = None
    aspect_ratio: str = "1:1"
    num_images: int = 1
    seed: int = 42
    steps: Optional[int] = None
    num_steps: Optional[int] = 50
    guidance: float = 4.0

    # Freeform Prompt
    prompt: Optional[str] = None

    # Template selection ("auto" - user does not pick template on UI; backend chooses best)
    template: str = "auto"
    qr_data: Optional[str] = None

    # Fast Preview mode (skip diffusion, generate plan & layout HTML only in 1s)
    fast_preview: bool = False


def validate_required_fields(req: GenerateRequest) -> List[str]:
    """Kiểm tra các trường bắt buộc theo nghiệp vụ của category.
    Trả về danh sách các trường còn thiếu (danh sách rỗng = hợp lệ).
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
    """Phục vụ giao diện người dùng Studio Web."""
    if not UI_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="demo_ui.html not found")
    return HTMLResponse(content=UI_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Silences browser favicon 404 probes."""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/api/health")
@app.get("/api/v3/health")
async def health_check():
    """Kiểm tra tình trạng hoạt động của hệ thống, GPU và LLM."""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_available else []
    display_model = "FLUX.2-klein-4B" if ACTIVE_MODEL_NAME == "flux.2-klein-4b" else "FLUX.2-klein-base-4B"
    return {
        "status": "online",
        "gpus_available": gpu_count,
        "mock_mode": IS_MOCK_MODE or DIT_MODEL is None,
        "models_loaded": DIT_MODEL is not None,
        "model": display_model,
        "model_name": ACTIVE_MODEL_NAME,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "device_dit": DEVICE_DIT,
        "device_aux": DEVICE_AUX,
        "active_llm": LLM_MODEL,
        "version": "3.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/required-fields")
async def get_required_fields():
    """Trả về danh sách các trường bắt buộc theo nghiệp vụ."""
    return {
        "status": "success",
        "fields": CATEGORY_REQUIRED_FIELDS,
        "guide_first_step": True,
    }


@app.get("/api/fonts")
async def get_fonts():
    """Trả về danh sách 19 phông chữ tiếng Việt chuẩn hóa."""
    from tendoo.core.fonts import list_font_options
    return {
        "status": "success",
        "groups": list_font_options(),
    }


@app.get("/api/v3/templates")
async def get_templates():
    """Lấy danh sách catalog toàn bộ templates khả dụng trong Tendoo v3."""
    items = []
    for k, v in TEMPLATE_CATALOG.items():
        items.append({
            "name": k,
            "label": v.get("name", k),
            "description": v.get("hint", ""),
            "best_for": v.get("hint", "").split(".")[0] if "." in v.get("hint", "") else v.get("hint", ""),
            "ratios": v.get("aspect_ratios", []),
            "has_qr": v.get("has_qr", True),
        })
    return {"status": "success", "templates": items}


@app.get("/api/v3/models")
async def get_models():
    """Lấy danh sách các mô hình LLM khả dụng."""
    return {
        "status": "success",
        "active_model": LLM_MODEL,
        "available_models": [
            {"id": "Qwen/Qwen3.8-27B", "type": "Dense 27B", "recommended": True, "description": "Mạnh nhất về JSON & tiếng Việt"},
            {"id": "google/gemma-4-31B-it", "type": "Dense 31B", "recommended": False, "description": "Suy luận logic vượt trội"},
            {"id": "Qwen/Qwen3.6-35B-A3B", "type": "MoE 35B", "recommended": False, "description": "Tốc độ xử lý siêu tốc"},
            {"id": "Qwen/Qwen3.5-4B", "type": "Dense 4B", "recommended": False, "description": "Bản nhẹ tiết kiệm VRAM"},
        ],
    }


@app.post("/api/v3/plan-only")
async def plan_only(req: GenerateRequest):
    """Endpoint duyệt nhanh kế hoạch sáng tạo và preview HTML trong 1s mà không chạy diffusion."""
    form_dict = req.model_dump(
        exclude={"prompt", "template", "aspect_ratio", "ref_image_b64", "image_base64", "fast_preview"}
    )
    form_dict["has_product_image"] = bool(req.ref_image_b64 or req.image_base64)
    width, height = ASPECT_RATIOS.get(req.aspect_ratio, (1024, 1024))

    plan, llm_trace = generate_creative_plan(
        form_data=form_dict,
        prompt=req.prompt or "",
        aspect_ratio=req.aspect_ratio,
        return_debug=True,
    )
    if req.template != "auto" and req.template in TEMPLATE_CATALOG:
        plan.template = req.template
    plan.template = _ensure_aspect_compatible(plan.template, req.aspect_ratio)

    mock_bg = generate_mock_backdrop(
        width=width,
        height=height,
        theme_color=plan.style.theme_color,
        background_tone=plan.style.background_tone,
        scene_prompt=plan.scene_prompt,
    )
    bg_uri = pil_to_base64_data_uri(mock_bg)

    # Phân tích bảng màu tắc kè hoa thích ứng theo pixel thực tế của vùng chữ chính
    try:
        crop_zone = _extract_primary_crop_zone(plan.template, width, height, plan.orientation)
        cp = analyze_color_harmony(
            np.array(mock_bg.convert("RGB")),
            crop_zone=crop_zone,
            color_mode="auto",
            font_style="luxury_serif" if plan.style.font == "playfair" else "modern_sans",
        )
        palette_override = palette_from_color_harmony(cp, theme_color=plan.style.theme_color)
    except Exception as e:
        logger.warning(f"Chameleon palette analysis notice: {e}")
        palette_override = None

    html_content = build_template_html(
        plan=plan,
        bg_data_uri=bg_uri,
        width=width,
        height=height,
        palette_override=palette_override,
    )

    return {
        "status": "success",
        "plan": plan.to_dict(),
        "palette": palette_override,
        "html": html_content,
        "bg_data_uri": bg_uri,
        "width": width,
        "height": height,
        "llm_debug": llm_trace,
    }


@app.post("/api/generate")
@app.post("/api/v3/generate")
async def generate_poster(req: GenerateRequest):
    """Sinh poster hoàn chỉnh qua toàn bộ pipeline Tendoo v3."""
    start_time = time.time()
    width, height = ASPECT_RATIOS.get(req.aspect_ratio, (1024, 1024))
    raw_prompt = (req.prompt or req.image_description or "").strip()
    # Pre-flight required-field gate: áp dụng khi không có prompt tự do
    if not raw_prompt:
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
    if ACTIVE_MODEL_NAME == "flux.2-klein-4b":
        num_steps = req.steps or 8
        guidance = req.guidance if req.guidance is not None else 1.5
    else:
        num_steps = req.steps if (req.steps and req.steps >= 20) else 50
        guidance = req.guidance if (req.guidance and req.guidance >= 3.0) else 4.0

    # 1. Thu thập dữ liệu form & Khởi tạo thư mục run
    form_dict = req.model_dump(
        exclude={"prompt", "template", "aspect_ratio", "ref_image_b64", "image_base64", "fast_preview"}
    )
    form_dict["has_product_image"] = bool(req.ref_image_b64 or req.image_base64)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:19]}"
    run_dir = OUTPUT_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Bước Lập Kế Hoạch Sáng Tạo (LLM Creative Director tự chọn Template tối ưu)
    logger.info(f"Generating creative plan for category: {req.category} (run_id: {run_id})...")
    plan, llm_trace = generate_creative_plan(
        form_data=form_dict,
        prompt=raw_prompt,
        aspect_ratio=req.aspect_ratio,
        debug_save_path=run_dir / "llm_debug.json",
        return_debug=True,
    )
    if req.template != "auto" and req.template in TEMPLATE_CATALOG:
        plan.template = req.template
    plan.template = _ensure_aspect_compatible(plan.template, req.aspect_ratio)

    qr_url = None
    if req.enable_qr and req.website_link and req.website_link.strip():
        from tendoo_v3.qr import generate_qr_base64
        qr_b64 = generate_qr_base64(req.website_link.strip())
        if qr_b64 and "," in qr_b64:
            try:
                qr_raw = base64.b64decode(qr_b64.split(",")[1])
                qr_file = run_dir / "00_qr_code.png"
                qr_file.write_bytes(qr_raw)
                qr_url = f"outputs/{run_id}/00_qr_code.png"
            except Exception as e:
                logger.warning(f"Failed to save QR file: {e}")
        plan.qr_code = req.website_link.strip()
    elif req.qr_data:
        plan.qr_code = req.qr_data

    # 3. Bước Tính Toán Continuous Mask -- đọc thẳng hình học theo TÊN TEMPLATE
    logger.info(f"Computing Continuous Mask for template: {plan.template} ({width}x{height})...")
    mask_np = generate_template_mask(
        template=plan.template,
        width=width,
        height=height,
        orientation=plan.orientation,
    )
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")

    # 4. Bước Sinh Nền Hình Ảnh (Diffusion hoặc Mock)

    if req.fast_preview or IS_MOCK_MODE or DIT_MODEL is None or not torch.cuda.is_available():
        logger.info("[Mock Mode] Generating aesthetic mock backdrop...")
        bg_img = generate_mock_backdrop(
            width=width,
            height=height,
            theme_color=plan.style.theme_color,
            background_tone=plan.style.background_tone,
            scene_prompt=plan.scene_prompt,
            spatial_mask=mask_np,
        )
    else:
        # Chạy suy luận DiT trên GPU với Khóa bảo vệ Async Lock
        async with INFER_LOCK:
            logger.info("[Diffusion Phase] Running Single-Pass Regional Velocity Blending on GPU...")
            from flux2.sampling import get_schedule, prc_img, prc_txt

            prompt_scene = plan.scene_prompt or req.prompt or "Studio product photography, high-end commercial aesthetic, beautiful lighting"
            prompt_corr = plan.corridor_prompt or (
                "Smooth background surface in soft focus, gentle bokeh, clean negative space without objects, "
                "matching scene color and lighting"
            )

            try:
                with torch.no_grad():
                    # 1. Mã hóa prompt bằng Qwen3 FP8 trên DEVICE_AUX, đẩy sang DEVICE_DIT
                    ctx_scene = TEXT_ENCODER([prompt_scene]).to(torch.bfloat16)
                    ctx_scene, ctx_scene_ids = prc_txt(ctx_scene[0])
                    ctx_scene = ctx_scene.unsqueeze(0).to(DEVICE_DIT)
                    ctx_scene_ids = ctx_scene_ids.unsqueeze(0).to(DEVICE_DIT)

                    ctx_corridor = TEXT_ENCODER([prompt_corr]).to(torch.bfloat16)
                    ctx_corridor, ctx_corridor_ids = prc_txt(ctx_corridor[0])
                    ctx_corridor = ctx_corridor.unsqueeze(0).to(DEVICE_DIT)
                    ctx_corridor_ids = ctx_corridor_ids.unsqueeze(0).to(DEVICE_DIT)

                    # 2. Downsample mặt nạ Gaussian sang latent space (1/16 độ phân giải)
                    w_lat, h_lat = width // 16, height // 16
                    mask_scaled = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                        (w_lat, h_lat), Image.Resampling.BICUBIC
                    )
                    mask_flat = torch.from_numpy(
                        np.array(mask_scaled, dtype=np.float32) / 255.0
                    ).reshape(1, -1, 1).to(DEVICE_DIT)

                    # 3. Mã hóa ảnh sản phẩm tham chiếu (nếu có tải lên) tại RoPE t=10.0
                    ref_toks = None
                    ref_ids = None
                    raw_b64 = req.ref_image_b64 or req.image_base64
                    if raw_b64:
                        try:
                            logger.info("  [Ref Product] Encoding uploaded product into VAE latents at RoPE t=10.0...")
                            if "," in raw_b64:
                                raw_b64 = raw_b64.split(",")[1]
                            ref_raw = base64.b64decode(raw_b64)
                            ref_file = run_dir / "00_uploaded_product.png"
                            ref_file.write_bytes(ref_raw)

                            r_toks, r_ids = load_and_encode_ref_image(
                                ref_image_path=ref_file,
                                ae=AE_MODEL,
                                device=DEVICE_DIT,
                                ae_device=DEVICE_AUX,
                                target_dim=512,
                                time_offset=10.0,
                            )
                            ref_toks = r_toks.to(dtype=torch.bfloat16)
                            ref_ids = r_ids
                            logger.info(f"  [✓] In-Context Product Reference attached: {ref_toks.shape[1]} tokens at RoPE t=10.0 (shape: {ref_toks.shape})")
                        except Exception as e:
                            logger.error(f"  [!] Failed to encode reference product image: {e}", exc_info=True)

                    # 4. Tạo nhiễu hạt khởi tạo canvas và lịch trình Euler ODE
                    torch.manual_seed(req.seed)
                    z_init = torch.randn(1, 128, h_lat, w_lat, device=DEVICE_DIT, dtype=torch.bfloat16)
                    img_tokens, img_ids = prc_img(z_init[0])
                    img_tokens = img_tokens.unsqueeze(0).to(DEVICE_DIT)
                    img_ids = img_ids.unsqueeze(0).to(DEVICE_DIT)

                    if ref_toks is not None and ref_ids is not None:
                        img_total = torch.cat([img_tokens, ref_toks], dim=1)
                        img_ids_total = torch.cat([img_ids, ref_ids], dim=1)
                        logger.info(f"  [Sequence] Total DiT tokens = {img_total.shape[1]} (Canvas: {img_tokens.shape[1]}, Ref: {ref_toks.shape[1]})")
                    else:
                        img_total = img_tokens
                        img_ids_total = img_ids
                        logger.info(f"  [Sequence] Canvas-only DiT tokens = {img_tokens.shape[1]}")

                    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_total.shape[1])

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
                        guidance=guidance,
                        num_canvas_tokens=img_tokens.shape[1],
                    )
                    dur_dit = time.time() - t0_dit
                    logger.info(f"  [DiT Finished] Euler ODE {num_steps} steps in {dur_dit:.2f}s")

                    # 5. Giải mã VAE Decoder trên DEVICE_AUX
                    target_ae_dtype = AE_DTYPE if AE_DTYPE is not None else torch.bfloat16
                    z_dec = out_blended[0].transpose(0, 1).reshape(1, 128, h_lat, w_lat).to(
                        device=DEVICE_AUX, dtype=target_ae_dtype
                    )
                    with torch.inference_mode():
                        x_dec = AE_MODEL.decode(z_dec).float()
                    x_arr = ((x_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
                    bg_img = Image.fromarray(x_arr)

            except Exception as e:
                logger.error(f"❌ Lỗi trong quá trình GPU Diffusion ({e}) -> Tự động chuyển sang Mock Backdrop.", exc_info=True)
                bg_img = generate_mock_backdrop(
                    width=width,
                    height=height,
                    theme_color=plan.style.theme_color,
                    background_tone=plan.style.background_tone,
                    scene_prompt=plan.scene_prompt,
                    spatial_mask=mask_np,
                )

    # Lưu ảnh nền và mask vào đĩa
    bg_path = run_dir / "background.png"
    mask_path = run_dir / "mask.png"
    poster_path = run_dir / "poster.png"
    html_path = run_dir / "poster.html"
    plan_path = run_dir / "plan.json"

    bg_img.save(bg_path, format="PNG")
    mask_img.save(mask_path, format="PNG")
    plan_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # 5. Bước Phân Tích Bảng Màu Tắc Kè Hoa (Chameleon Adaptive Palette)
    try:
        crop_zone = _extract_primary_crop_zone(plan.template, width, height, plan.orientation)
        cp = analyze_color_harmony(
            np.array(bg_img.convert("RGB")),
            crop_zone=crop_zone,
            color_mode="auto",
            font_style="luxury_serif" if plan.style.font == "playfair" else "modern_sans",
        )
        palette_override = palette_from_color_harmony(cp, theme_color=plan.style.theme_color)
    except Exception as e:
        logger.warning(f"Chameleon palette analysis notice: {e}")
        palette_override = None

    # 6. Bước Biên Dịch & Chụp Ảnh Render (Playwright Chromium)
    logger.info("Rendering composited HTML poster via Playwright...")
    bg_uri = pil_to_base64_data_uri(bg_img)
    rendered_path, html_content = render_plan_to_poster(
        plan=plan,
        bg_data_uri=bg_uri,
        output_image_path=poster_path,
        width=width,
        height=height,
        palette_override=palette_override,
    )
    html_path.write_text(html_content, encoding="utf-8")

    # Đọc kết quả Base64 trả về client
    poster_img = Image.open(poster_path)
    poster_uri = pil_to_base64_data_uri(poster_img)
    mask_uri = pil_to_base64_data_uri(mask_img)

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"✅ Finished request in {elapsed}s -> {poster_path.name}")
    final_poster_rel = f"outputs/{run_id}/poster.png"
    bg_rel = f"outputs/{run_id}/background.png"
    mask_rel = f"outputs/{run_id}/mask.png"

    posters_list = [
        {
            "index": 0,
            "seed": req.seed,
            "final_poster_url": final_poster_rel,
            "blended_bg_url": bg_rel,
        }
    ]

    _prune_old_runs()

    return {
        "status": "success",
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "latency_s": str(elapsed),
        "resolved_layout": plan.template,
        "final_poster_url": final_poster_rel,
        "blended_bg_url": bg_rel,
        "mask_url": mask_rel,
        "qr_url": qr_url,
        "posters": posters_list,
        "plan": plan.to_dict(),
        "creative_plan": plan.to_dict(),
        "palette": palette_override,
        "poster_uri": poster_uri,
        "bg_uri": bg_uri,
        "mask_uri": mask_uri,
        "html": html_content,
        "width": width,
        "height": height,
        "llm_debug": {
            "mode": llm_trace.get("mode"),
            "status": llm_trace.get("status"),
            "error": llm_trace.get("error"),
            "latency_seconds": llm_trace.get("latency_seconds"),
            "debug_url": f"outputs/{run_id}/llm_debug.json",
            "input_url": f"outputs/{run_id}/llm_input.json",
            "output_url": f"outputs/{run_id}/llm_output.json",
        },
    }


@app.get("/api/v3/llm-debug/latest")
async def get_latest_llm_debug():
    """Tra cứu bản ghi debug LLM gần nhất."""
    p = PROJECT_ROOT / "output_tendoo_v3" / "llm_debug" / "latest_llm_trace.json"
    if not p.exists():
        return {"status": "none", "message": "Chưa có lượt gọi LLM nào được ghi nhận."}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file debug: {e}")


@app.get("/api/v3/runs/{run_id}/llm-debug")
async def get_run_llm_debug(run_id: str):
    """Tra cứu bản ghi debug LLM theo từng run_id cụ thể."""
    p = OUTPUT_RUNS_DIR / run_id / "llm_debug.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy llm_debug.json cho run_id '{run_id}'.")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file debug: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI v3 Studio Demo Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on")
    parser.add_argument("--mock", action="store_true", help="Force mock mode without loading GPU weights")
    parser.add_argument("--device_dit", type=str, default=None, help="CUDA device for DiT model (default: cuda:0)")
    parser.add_argument("--device_aux", type=str, default=None, help="CUDA device for VAE & Qwen3 (default: cuda:1)")
    parser.add_argument("--model", type=str, default="distill", choices=["distill", "base"],
                        help="Model variant: 'distill' (default: FLUX.2-klein-4B, fast 4-8 steps) or 'base' (FLUX.2-klein-base-4B)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to FLUX.2-klein-4B or FLUX.2-klein-base-4B directory or checkpoint")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_MODE"] = "1"
    if args.device_dit:
        os.environ["TENDOO_V3_DEVICE_DIT"] = args.device_dit
    if args.device_aux:
        os.environ["TENDOO_V3_DEVICE_AUX"] = args.device_aux
    if args.model == "base":
        os.environ["TENDOO_USE_BASE_MODEL"] = "1"
    else:
        os.environ["TENDOO_USE_BASE_MODEL"] = "0"

    if args.model_path:
        os.environ["FLUX_CHECKPOINT_DIR"] = args.model_path
        p = Path(args.model_path)
        if p.is_dir():
            distill_cand = p / "flux-2-klein-4b.safetensors"
            if distill_cand.exists():
                os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_cand)
                os.environ["TENDOO_USE_BASE_MODEL"] = "0"
            base_cand = p / "flux-2-klein-base-4b.safetensors"
            if base_cand.exists() and not distill_cand.exists():
                os.environ["KLEIN_4B_BASE_MODEL_PATH"] = str(base_cand)
                os.environ["TENDOO_USE_BASE_MODEL"] = "1"
            cand_ae = p / "vae" / "diffusion_pytorch_model.safetensors"
            if cand_ae.exists():
                os.environ["AE_MODEL_PATH"] = str(cand_ae)
        elif p.is_file():
            if "base" in p.name.lower():
                os.environ["KLEIN_4B_BASE_MODEL_PATH"] = str(p)
                os.environ["TENDOO_USE_BASE_MODEL"] = "1"
            else:
                os.environ["KLEIN_4B_MODEL_PATH"] = str(p)
                os.environ["TENDOO_USE_BASE_MODEL"] = "0"

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
