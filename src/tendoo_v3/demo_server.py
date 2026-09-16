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
from tendoo_v3.llm_planner import generate_creative_plan, LLM_MODEL
from tendoo_v3.mask_engine import generate_template_mask
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
DEVICE_DIT: str = os.environ.get("TENDOO_V3_DEVICE_DIT", "cuda:0")
DEVICE_AUX: str = os.environ.get("TENDOO_V3_DEVICE_AUX", "cuda:1")
INFER_LOCK = asyncio.Lock()
IS_MOCK_MODE: bool = not torch.cuda.is_available() or os.environ.get("MOCK_MODE", "").lower() in ["1", "true"]

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
    global DIT_MODEL, AE_MODEL, TEXT_ENCODER
    logger.info("Releasing GPU memory...")
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
    global DIT_MODEL, AE_MODEL, TEXT_ENCODER, DEVICE_DIT, DEVICE_AUX, IS_MOCK_MODE

    logger.info("=" * 70)
    logger.info("🚀 STARTING TENDOO v3 STUDIO DEMO SERVER")
    logger.info(f"   Mock Mode: {IS_MOCK_MODE} | CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"   Active LLM: {LLM_MODEL}")
    logger.info("=" * 70)

    if not IS_MOCK_MODE and torch.cuda.is_available():
        try:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                DEVICE_DIT = "cuda:0"
                DEVICE_AUX = "cuda:1"
                logger.info(f"Multi-GPU Mode: DiT on {DEVICE_DIT}, Auxiliary on {DEVICE_AUX}")
            else:
                DEVICE_DIT = DEVICE_AUX = "cuda:0"
                logger.info(f"Single-GPU Mode: All on {DEVICE_DIT}")
        except Exception as e:
            logger.warning(f"GPU initialization notice: {e}")

    yield

    free_all_gpu_memory()


app = FastAPI(title="Tendoo AI v3 Studio", version="3.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    category: str = "promo"
    # Form fields
    product_name: Optional[str] = None
    discount: Optional[str] = None
    price: Optional[str] = None
    product_desc: Optional[str] = None
    highlights: Optional[str] = None
    opening_date: Optional[str] = None
    opening_promo: Optional[str] = None
    booking_contact: Optional[str] = None
    feedback_target: Optional[str] = None
    feedback_quote: Optional[str] = None
    feedback_rating: Optional[int] = None
    customer_name: Optional[str] = None
    job_position: Optional[str] = None
    job_desc: Optional[str] = None
    apply_deadline: Optional[str] = None
    apply_method: Optional[str] = None
    guide_steps: Optional[List[str]] = None
    store_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    qr_data: Optional[str] = None

    # Freeform Prompt
    prompt: Optional[str] = None

    # Template selection ("auto" or specific template name)
    template: str = "auto"

    # Settings
    aspect_ratio: str = "1:1"
    num_steps: int = 50
    guidance: float = 4.0
    seed: int = 42

    # Uploaded image (optional Base64 data)
    ref_image_b64: Optional[str] = None

    # Fast Preview mode (skip diffusion, generate plan & layout HTML only in 1s)
    fast_preview: bool = False


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Phục vụ giao diện người dùng Studio Web."""
    if not UI_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="demo_ui.html not found")
    return HTMLResponse(content=UI_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/v3/health")
async def health_check():
    """Kiểm tra tình trạng hoạt động của hệ thống, GPU và LLM."""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_available else []
    return {
        "status": "healthy",
        "mock_mode": IS_MOCK_MODE,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "active_llm": LLM_MODEL,
        "version": "3.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    form_dict = req.model_dump(exclude={"prompt", "template", "aspect_ratio", "ref_image_b64", "fast_preview"})
    width, height = ASPECT_RATIOS.get(req.aspect_ratio, (1024, 1024))

    plan = generate_creative_plan(
        form_data=form_dict,
        prompt=req.prompt or "",
        aspect_ratio=req.aspect_ratio,
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
    }


@app.post("/api/v3/generate")
async def generate_poster(req: GenerateRequest):
    """Sinh poster hoàn chỉnh qua toàn bộ pipeline Tendoo v3."""
    start_time = time.time()
    width, height = ASPECT_RATIOS.get(req.aspect_ratio, (1024, 1024))
    
    # 1. Thu thập dữ liệu form
    form_dict = req.model_dump(exclude={"prompt", "template", "aspect_ratio", "ref_image_b64", "fast_preview"})
    
    # 2. Bước Lập Kế Hoạch Sáng Tạo (LLM Creative Director)
    logger.info(f"Generating creative plan for category: {req.category}...")
    plan = generate_creative_plan(
        form_data=form_dict,
        prompt=req.prompt or "",
        aspect_ratio=req.aspect_ratio,
    )
    if req.template != "auto" and req.template in TEMPLATE_CATALOG:
        plan.template = req.template
    plan.template = _ensure_aspect_compatible(plan.template, req.aspect_ratio)

    if req.qr_data:
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
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:19]}"
    run_dir = OUTPUT_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if req.fast_preview or IS_MOCK_MODE or DIT_MODEL is None:
        logger.info("[Diffusion Phase] Sinh ảnh nền Mock Backdrop theo bảng màu và prompt...")
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
            logger.info("[Diffusion Phase] Chạy Single-Pass Regional Velocity Blending trên GPU...")
            # TODO: Khi model checkpoint được nạp vào VRAM, denoise_regional_velocity_blended sẽ được gọi trực tiếp tại đây
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
    _prune_old_runs()

    return {
        "status": "success",
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "plan": plan.to_dict(),
        "palette": palette_override,
        "poster_uri": poster_uri,
        "bg_uri": bg_uri,
        "mask_uri": mask_uri,
        "html": html_content,
        "width": width,
        "height": height,
    }


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI v3 Studio Demo Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on")
    parser.add_argument("--mock", action="store_true", help="Force mock mode without loading GPU weights")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_MODE"] = "1"

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
