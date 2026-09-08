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
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

    # Legacy / alias parameters for backwards compatibility
    headline: Optional[str] = None
    pre_header: str = ""
    slogan: str = ""
    offer_main: Optional[str] = None
    offer_sub: Optional[str] = None
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


def sanitize_and_inject_zero_text(user_prompt: str) -> str:
    """
    Sanitizes user scene prompt and injects mandatory ZERO-TEXT negative constraints.
    Prevents DiT from hallucinating broken/deformed characters and text artifacts.
    """
    if not user_prompt or not user_prompt.strip():
        return (
            "Commercial advertising photography, professional studio lighting, "
            "clean photographic background, unbranded, zero text, no words, no letters, "
            "no typography, no logos, no watermarks, no labels, no signs"
        )

    prompt = user_prompt.strip()

    # 1. Strip resolution / aspect ratio pollution (Rule 6: 9:16, 16:9, 4:5, 1:1, 8k, 4k, 1080p, etc.)
    prompt = re.sub(
        r'\b(?:9:16|16:9|4:5|1:1|8k|4k|2k|1080p|720p|hd|uhd|full\s*hd)\b',
        '',
        prompt,
        flags=re.IGNORECASE
    )

    # 2. Filter out explicit user directives trying to draw text onto the image
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

    # 3. Canonical Zero-Text negative constraint clause
    zero_text_clause = (
        "clean photographic background, unbranded, zero text, "
        "no words, no letters, no typography, no logos, no watermarks, no labels, no signs"
    )

    lower_prompt = prompt.lower()
    if "zero text" not in lower_prompt and "no words" not in lower_prompt:
        if prompt:
            prompt = f"{prompt}, {zero_text_clause}"
        else:
            prompt = f"Commercial advertising photography, {zero_text_clause}"
    else:
        missing_parts = []
        for term in ["unbranded", "zero text", "no words", "no letters", "no typography", "no logos"]:
            if term not in lower_prompt:
                missing_parts.append(term)
        if missing_parts:
            prompt = f"{prompt}, {', '.join(missing_parts)}"

    return prompt


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
    final_headline = req.title or req.headline or "ƯU ĐÃI ĐẶC BIỆT"
    final_offer_main = req.discount or req.offer_main or ""
    final_offer_sub = req.applied_product or req.offer_sub or ""
    final_brand = req.store_name or req.brand or ""
    final_hotline = req.phone or req.hotline or ""
    raw_scene_prompt = req.image_description or req.prompt_scene or ""
    final_scene_prompt = sanitize_and_inject_zero_text(raw_scene_prompt)
    if raw_scene_prompt != final_scene_prompt:
        print(f"  [Prompt Engine] Injected ZERO-TEXT negative constraints:\n    Raw: '{raw_scene_prompt}'\n    Injected: '{final_scene_prompt}'")
    style_hint = req.style_hint or "daylight"

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
            content = PosterContent(
                headline=final_headline,
                pre_header=req.pre_header,
                slogan=req.slogan,
                offer_main=final_offer_main,
                offer_sub=final_offer_sub,
                dates=final_dates,
                brand=final_brand,
                hotline=final_hotline,
                address=req.address,
                website_link=req.website_link,
                qr_data_uri=qr_data_uri,
                applicable=req.applied_product,
                category=req.category,
                text_effect=req.text_effect,
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
                content = PosterContent(
                    headline=final_headline,
                    pre_header=req.pre_header,
                    slogan=req.slogan,
                    offer_main=final_offer_main,
                    offer_sub=final_offer_sub,
                    dates=final_dates,
                    brand=final_brand,
                    hotline=final_hotline,
                    address=req.address,
                    website_link=req.website_link,
                    qr_data_uri=qr_data_uri,
                    applicable=req.applied_product,
                    category=req.category,
                    text_effect=req.text_effect,
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


if __name__ == "__main__":
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
