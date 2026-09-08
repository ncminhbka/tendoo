#!/usr/bin/env python3
"""
scripts/demo_server.py

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
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
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

from tendoo.layouts import (
    PosterContent,
    analyze_color_harmony,
    get_layout,
)
from tendoo.typography_engine import PosterRenderer


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
if not UI_HTML_PATH.exists():
    UI_HTML_PATH = PROJECT_ROOT / "scripts" / "demo_ui.html"


def free_all_gpu_memory():
    """Flushes VRAM across all CUDA devices cleanly."""
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
    prompt_scene: Optional[str] = None
    prompt_corridor: Optional[str] = None

    # 4. Display / Design & Inference Controls (Common to all categories)
    layout: str = "top_dome"
    style_hint: str = "daylight"
    text_effect: str = "auto"
    aspect_ratio: str = "1:1"
    num_images: int = 1
    seed: int = 42
    steps: int = 8
    guidance: float = 1.5
    width: Optional[int] = None
    height: Optional[int] = None


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


LAYOUT_COMPATIBLE_STYLES = {
    "top_dome": {
        "styles": ["daylight", "studio_dark", "golden_hour", "gold_bevel", "festive_moon", "ribbon"],
        "default": "daylight",
    },
    "bottom_platform": {
        "styles": ["cinematic_asphalt", "luxury_marble", "warm_wood", "nature_stone", "water_mirror", "cyberpunk_grid"],
        "default": "cinematic_asphalt",
    },
    "split_column": {
        "styles": ["champagne_silk", "silk_sash", "minimal_wall", "studio_light_pillar", "velvet_drape"],
        "default": "champagne_silk",
    },
    "center_hourglass": {
        "styles": ["moonbeam", "studio_spotlight", "festive_light"],
        "default": "moonbeam",
    },
    "diagonal_slash": {
        "styles": ["sport_speed", "cyber_neon", "carbon_mesh", "daylight_motion"],
        "default": "sport_speed",
    },
    "l_frame": {
        "styles": ["tech_minimal", "cyber_tech", "luxury_gold", "daylight_clean"],
        "default": "tech_minimal",
    },
}


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
    layout_name: str = "top_dome",
    has_ref_image: bool = False,
) -> str:
    """
    Injects layout-aware spatial composition steering tokens into the scene prompt to overcome
    the diffusion model's inherent center bias and prevent product/typography collision.

    Center layouts (top_dome, bottom_platform, center_hourglass):
      Reinforce clean copy space around the central aperture.

    Asymmetrical layouts (split_column, diagonal_slash, l_frame):
      Explicitly steer the hero subject/product away from the typography corridor and into the
      dedicated product zone (right side, lower-right diagonal, or lower-right quadrant).
    """
    prompt = (scene_prompt or "").strip()
    lower_p = prompt.lower()
    layout = (layout_name or "top_dome").lower().strip()

    if layout == "split_column":
        spatial_guidance = (
            "asymmetrical editorial composition, hero product placed prominently on the right half of the frame (x > 0.48, rule of thirds), "
            "clean open negative copy space across the left vertical third, no subject or foreground elements on the left side, "
            "balanced commercial fashion layout"
        )
        if has_ref_image:
            spatial_guidance = f"{spatial_guidance}, preserve authentic product placed on the right side"

    elif layout == "diagonal_slash":
        spatial_guidance = (
            "dynamic athletic diagonal composition, hero product positioned dynamically in the lower-right diagonal half of the frame (x > 0.45, y > 0.45), "
            "angled towards center, wide open clean negative copy space across the upper-left diagonal quadrant, "
            "no product or clutter in the upper-left area, high energy motion background"
        )
        if has_ref_image:
            spatial_guidance = f"{spatial_guidance}, preserve authentic product placed in the lower-right diagonal area"

    elif layout == "l_frame":
        spatial_guidance = (
            "architectural framing composition, hero product positioned strictly in the lower-right quadrant of the frame (x in [0.42, 0.94], y in [0.32, 0.94]), "
            "wide open clean negative copy space framing the entire top horizontal header and left vertical column, "
            "no product or foreground elements in upper-left corner or top bar, modern technology commercial setup"
        )
        if has_ref_image:
            spatial_guidance = f"{spatial_guidance}, preserve authentic product placed in the lower-right quadrant"

    elif layout == "bottom_platform":
        spatial_guidance = (
            "cinematic commercial composition, hero product standing centered on the bottom platform stage, "
            "clean negative copy space across the lower pedestal area, stable centered framing"
        )
    elif layout == "center_hourglass":
        spatial_guidance = (
            "studio commercial composition, hero product positioned in the center aperture of the frame, "
            "clean open negative copy space across top header and bottom footer, balanced hourglass framing"
        )
    else:  # top_dome
        spatial_guidance = (
            "commercial advertising composition, hero product centered in the lower two-thirds of the frame, "
            "clean open negative copy space in the upper dome area, no product in the top header"
        )

    # Prevent redundant token injection
    steering_check_keys = {
        "split_column": ["right half", "right side", "left vertical third"],
        "diagonal_slash": ["lower-right diagonal", "upper-left diagonal"],
        "l_frame": ["lower-right quadrant", "architectural framing"],
        "bottom_platform": ["bottom platform stage", "lower pedestal"],
        "center_hourglass": ["center aperture", "hourglass framing"],
        "top_dome": ["lower two-thirds", "upper dome"],
    }
    needed = not any(k in lower_p for k in steering_check_keys.get(layout, []))
    if needed:
        prompt = f"{prompt}, {spatial_guidance}" if prompt else spatial_guidance

    return prompt


def detect_scene_lighting_tone(
    scene_prompt: str,
    user_hint: str = "auto",
    layout_name: str = "top_dome",
) -> str:
    """
    Prevents Spatial Semantic Clash between Layout geometry, corridor lighting, and scene atmosphere.
    Validates compatibility against LAYOUT_COMPATIBLE_STYLES:
      - If user specifies an explicit valid style for this layout, respects user's choice.
      - If user chooses 'auto' or specifies an incompatible style (e.g. top_dome + cinematic_asphalt),
        intelligently analyzes scene keywords to select the most harmonious optical corridor style.
    """
    layout_cfg = LAYOUT_COMPATIBLE_STYLES.get(layout_name, LAYOUT_COMPATIBLE_STYLES["top_dome"])
    valid_styles = layout_cfg["styles"]
    default_style = layout_cfg["default"]

    # If user explicitly chose a compatible style, respect it
    if user_hint and user_hint.lower() not in ("auto", "", "none"):
        clean_hint = user_hint.lower().strip()
        if clean_hint in valid_styles:
            return clean_hint
        print(f"  [Notice] Incompatible style_hint '{user_hint}' for layout '{layout_name}'. Auto-harmonizing...")

    # Auto-detection based on scene keywords and layout geometry
    lower_p = (scene_prompt or "").lower()
    is_dark = any(k in lower_p for k in ["neon", "cyber", "night", "dark", "tối", "bóng đêm", "đen", "slate", "black", "moody", "asphalt"])
    is_warm_gold = any(k in lower_p for k in ["gold", "vàng", "hoàng gia", "luxury", "sunset", "golden hour", "warm", "trung thu", "lễ hội", "nến", "candle"])
    is_stone_marble = any(k in lower_p for k in ["marble", "đá cẩm thạch", "bục đá", "sa thạch", "stone", "pedestal"])
    is_wood = any(k in lower_p for k in ["wood", "gỗ", "mặt bàn", "quán cafe", "tea table"])
    is_water = any(k in lower_p for k in ["water", "nước", "hồ", "lake", "ocean", "river", "biển", "phản chiếu", "reflection"])

    if layout_name == "bottom_platform":
        if is_stone_marble:
            return "luxury_marble"
        if is_wood:
            return "warm_wood"
        if is_water:
            return "water_mirror"
        return "cinematic_asphalt"

    elif layout_name == "split_column":
        if any(k in lower_p for k in ["tường", "bê tông", "wall", "minimal", "phòng"]):
            return "minimal_wall"
        if any(k in lower_p for k in ["khói", "pillar", "beam", "smoke", "haze"]):
            return "studio_light_pillar"
        return "champagne_silk"

    elif layout_name == "center_hourglass":
        if is_warm_gold:
            return "festive_light"
        if is_dark:
            return "studio_spotlight"
        return "moonbeam"

    elif layout_name == "diagonal_slash":
        if is_dark or any(k in lower_p for k in ["cyber", "neon", "gaming", "tech"]):
            return "cyber_neon"
        if any(k in lower_p for k in ["carbon", "black", "matte", "stealth", "kim loại"]):
            return "carbon_mesh"
        if any(k in lower_p for k in ["daylight", "sun", "outdoor", "sáng", "nắng"]):
            return "daylight_motion"
        return "sport_speed"

    elif layout_name == "l_frame":
        if any(k in lower_p for k in ["cyber", "neon", "tech", "điện tử", "gaming"]):
            return "cyber_tech"
        if is_warm_gold:
            return "luxury_gold"
        if any(k in lower_p for k in ["daylight", "sun", "outdoor", "sáng", "nắng", "văn phòng", "office"]):
            return "daylight_clean"
        return "tech_minimal"

    else:  # top_dome
        if is_dark:
            return "studio_dark"
        if is_warm_gold:
            return "golden_hour"
        if any(k in lower_p for k in ["moon", "trăng", "lễ hội", "festival"]):
            return "festive_moon"
        return default_style


def build_poster_content(
    req: GenerateRequest,
    final_headline: str,
    final_offer_main: str,
    final_offer_sub: str,
    final_dates: str,
    qr_data_uri: str,
) -> PosterContent:
    """Instantiates a complete, well-typed PosterContent from GenerateRequest."""
    cat = (req.category or "promo").lower().strip()
    return PosterContent(
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


def run_pipeline_inference(req: GenerateRequest) -> Dict[str, Any]:
    """Synchronous core inference worker executed under INFER_LOCK."""
    t_start = time.time()
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

    # 2. Retrieve Layout
    layout = get_layout(req.layout)

    # 3. Generate Parametric Mask
    mask_np = layout.generate_mask(width=w, height=h)
    mask_vis = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    mask_file = case_dir / "01_corridor_mask.png"
    mask_vis.save(mask_file)

    # 4. Harmonize Field Values (Unified schema mapping)
    cat = (req.category or "promo").lower().strip()
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
    final_brand = req.store_name or req.brand or ""
    final_hotline = req.phone or req.hotline or ""
    raw_scene_prompt = req.image_description or req.prompt_scene or ""
    has_ref_image = bool(req.image_base64)
    zero_text_prompt = sanitize_and_inject_zero_text(raw_scene_prompt, has_ref_image=has_ref_image)
    final_scene_prompt = inject_spatial_layout_guidance(zero_text_prompt, layout_name=layout.name, has_ref_image=has_ref_image)
    if raw_scene_prompt != final_scene_prompt:
        print(f"  [Prompt Engine] Injected ZERO-TEXT & Spatial Guidance:\n    Raw: '{raw_scene_prompt}'\n    Final: '{final_scene_prompt}'")
    
    style_hint = detect_scene_lighting_tone(final_scene_prompt, req.style_hint or "auto", layout_name=layout.name)

    dates_parts = []
    if req.date_start:
        dates_parts.append(f"Từ {req.date_start}")
    if req.date_end:
        dates_parts.append(f"Đến {req.date_end}")
    final_dates = " - ".join(dates_parts) if dates_parts else ""

    # 5. Optional QR Code Generation (Backend QR Engine)
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

    # 6. Save Uploaded Product Image if provided
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

    # 7. Generate Blended Background & Final Poster (Regional Velocity Blending + HTML Typography)
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
            palette = analyze_color_harmony(np.array(blended_pil), safe_zone, color_mode="auto")

            bg_data_uri = pil_to_base64_data_uri(blended_pil)
            content = build_poster_content(
                req=req,
                final_headline=final_headline,
                final_offer_main=final_offer_main,
                final_offer_sub=final_offer_sub,
                final_dates=final_dates,
                qr_data_uri=qr_data_uri,
            )
            html_str = layout.render_html(
                content=content,
                palette=palette,
                bg_data_uri=bg_data_uri,
                width=w,
                height=h,
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
        from pipeline_e2e_poster import denoise_regional_velocity_blended, load_and_encode_ref_image

        prompt_corr = req.prompt_corridor or layout.get_corridor_prompt(style_hint)

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
                print(f"  [Ref Product] Encoding uploaded product into VAE latents...")
                try:
                    r_toks, r_ids = load_and_encode_ref_image(
                        ref_image_path=ref_image_path,
                        ae=AE_MODEL,
                        device=DEVICE_AUX,
                        target_dim=512,
                        time_offset=10.0,
                    )
                    ref_toks = r_toks.to(device=DEVICE_DIT, dtype=torch.bfloat16)
                    ref_ids = r_ids.to(device=DEVICE_DIT)
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
                palette = analyze_color_harmony(np.array(blended_pil), safe_zone, color_mode="auto")

                bg_data_uri = pil_to_base64_data_uri(blended_pil)
                content = build_poster_content(
                    req=req,
                    final_headline=final_headline,
                    final_offer_main=final_offer_main,
                    final_offer_sub=final_offer_sub,
                    final_dates=final_dates,
                    qr_data_uri=qr_data_uri,
                )
                html_str = layout.render_html(
                    content=content,
                    palette=palette,
                    bg_data_uri=bg_data_uri,
                    width=w,
                    height=h,
                    headline_effect=req.text_effect,
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
    }


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """
    Thread-safe poster generation endpoint protected by INFER_LOCK.
    Queues simultaneous requests smoothly without GPU VRAM collision.
    """
    async with INFER_LOCK:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_pipeline_inference, req)
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
