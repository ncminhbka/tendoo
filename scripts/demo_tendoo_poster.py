"""
================================================================================
TENDOO AI - EXECUTIVE DEMO SCRIPT (PHASE 1 MILESTONE)
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Features:
- Single Vietnamese Typography with 100% Diacritic Accuracy & Auto-Wrapping
- Optional Real Product Reference Image Ingestion (t=60.0 default)
- Customizable Canvas Aspect Ratio & Resolution (9:16, 1:1, 16:9, etc.)
- Convenient Font Selector (Playfair, BeVietnam, Anton, Pacifico, Graffiti...)
- Fast 1-Pass In-Context ODE Flow Matching (~30s on 2x A30)
================================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Auto-configure PYTHONPATH to include src/
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Auto-configure Offline Mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from einops import rearrange
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import native FLUX.2 modules
from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import (
    batched_prc_txt,
    denoise_cfg,
    get_schedule,
    prc_img,
)
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import (
    load_ae,
    load_flow_model,
)
from tendoo import (
    FONT_REGISTRY,
    create_glyph_image,
    encode_glyph_to_incontext_tokens,
    encode_product_to_incontext_tokens,
    resolve_font_path,
)


def run_demo(
    text: str,
    prompt: str,
    t_text: float = 10.0,
    image_ref: str | None = None,
    t_product: float = 60.0,
    width: int = 576,
    height: int = 1024,
    box_w: int | None = None,
    box_h: int | None = None,
    font: str = "bevietnam",
    output_path: str = "demo_output.png",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
):
    """
    Executes end-to-end Tendoo AI pipeline on FLUX.2 Klein 4B Base.
    """
    start_time = time.time()
    width = (width // 16) * 16
    height = (height // 16) * 16

    resolved_font = resolve_font_path(font)

    print("=" * 80)
    print(" 🚀 TENDOO AI: EXECUTIVE POSTER GENERATOR")
    print("=" * 80)
    print(f"📝 Text Input     : '{text}'")
    print(f"🧭 Text Time (t)  : {t_text}")
    print(f"🔤 Font Selected  : {Path(resolved_font).name}")
    print(f"📐 Canvas Size    : {width}x{height} pixels (Divisible by 16)")
    if image_ref:
        print(f"📦 Product Ref    : {image_ref} (t={t_product})")
    print(f"🎨 Prompt         : {prompt}")
    print(f"⚙️ Denoise Config : {num_steps} steps, CFG Guidance = {guidance}, Seed = {seed}")
    print("=" * 80)

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print(f"🚀 Dual GPU Mode: DiT on GPU 0, VAE/Qwen3 on GPU 1")
    else:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:0")
        device_te = torch.device("cuda:0")
        print(f"🚀 Single GPU Mode: {device_dit}")

    torch.manual_seed(seed)

    # 1. Render Glyph Bitmap with Intelligent Auto-Scaling for Multi-line / Poems
    print("\n[1/5] Generating Vietnamese Orthographic Glyph...")
    raw_lines = [l.strip() for l in text.replace("\\n", "\n").split("\n") if l.strip()]
    num_lines = len(raw_lines) if len(raw_lines) > 1 else (2 if len(text.split()) >= 4 else 1)

    if box_w is None:
        calc_w = min(width - 64, 896 if num_lines >= 3 else 512)
    else:
        calc_w = min(width, box_w)
    calc_w = (calc_w // 16) * 16

    if box_h is None:
        calc_h = min(height - 64, max(160, num_lines * 128))
    else:
        calc_h = min(height, box_h)
    calc_h = (calc_h // 16) * 16

    glyph_img = create_glyph_image(
        text=text,
        target_width=calc_w,
        target_height=calc_h,
        font_path=resolved_font,
    )
    glyph_preview = Path(output_path).stem + "_glyph_preview.png"
    glyph_img.save(glyph_preview)
    print(f"  -> Glyph preview saved: {glyph_preview} ({calc_w}x{calc_h} - {num_lines} lines)")

    # 2. Load Models
    print("\n[2/5] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)

    # 3. Encode Text Prompt
    print("\n[3/5] Encoding Text Prompt via Qwen3-4B-FP8...")
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    if num_gpus < 2:
        del text_encoder
        torch.cuda.empty_cache()

    # 4. Encode In-Context Reference Tokens
    print("\n[4/5] Encoding 4D RoPE In-Context Tokens...")
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    ref_token_list = []
    ref_id_list = []

    # Add Product Reference if provided
    if image_ref and os.path.exists(image_ref):
        prod_tokens, prod_ids = encode_product_to_incontext_tokens(
            ae=ae, image_path=image_ref, t_offset=t_product, device=device_ae
        )
        ref_token_list.append(prod_tokens)
        ref_id_list.append(prod_ids)
        print(f"  -> Added Product Reference at t={t_product} ({prod_tokens.shape[1]} tokens)")

    # Add Text Glyph
    text_tokens, text_ids = encode_glyph_to_incontext_tokens(
        ae=ae, glyph_img=glyph_img, t_offset=t_text, device=device_ae
    )
    ref_token_list.append(text_tokens)
    ref_id_list.append(text_ids)
    print(f"  -> Added Text Glyph at t={t_text} ({text_tokens.shape[1]} tokens)")

    all_ref_tokens = torch.cat(ref_token_list, dim=1).to(device_dit)
    all_ref_ids = torch.cat(ref_id_list, dim=1).to(device_dit)

    print(f"  -> Total Canvas Tokens: {img_tokens.shape[1]} (Grid: {lat_h}x{lat_w})")
    print(f"  -> Total Ref Tokens   : {all_ref_tokens.shape[1]}")

    # 5. Denoise Euler ODE
    print(f"\n[5/5] Running ODE Denoise ({num_steps} steps, CFG {guidance})...")
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    with torch.no_grad():
        out_latent = denoise_cfg(
            model=model,
            img=img_tokens,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=all_ref_tokens,
            img_cond_seq_ids=all_ref_ids,
        )

        out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        print("  -> Decoding final image via VAE Decoder...")
        out_pixels = ae.decode(out_latent.to(device_ae))
        out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_img = Image.fromarray(out_pixels)
        result_img.save(output_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 DEMO POSTER GENERATED SUCCESSFULLY IN {elapsed:.2f}s!")
    print(f"📸 Final Result saved to : {output_path} ({width}x{height})")
    print(f"🖼️ Glyph Preview saved to: {glyph_preview}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tendoo AI - Executive Demo Script for Vietnamese Typography & Poster Generation")
    parser.add_argument("--text", type=str, required=True, help="Vietnamese Text String (e.g. 'TIỆM CÀ PHÊ ANH QUÂN')")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing background scene, material & lighting")
    parser.add_argument("--t_text", type=float, default=10.0, help="Time offset for Text Glyph (default: 10.0)")
    parser.add_argument("--image_ref", type=str, default=None, help="Optional path to Product Reference Image")
    parser.add_argument("--t_product", type=float, default=60.0, help="Time offset for Product Reference (default: 60.0)")
    parser.add_argument("--width", type=int, default=576, help="Width in pixels (default: 576 for 9:16)")
    parser.add_argument("--height", type=int, default=1024, help="Height in pixels (default: 1024 for 9:16)")
    parser.add_argument("--box_w", type=int, default=None, help="Optional custom Glyph Box width (default: auto)")
    parser.add_argument("--box_h", type=int, default=None, help="Optional custom Glyph Box height (default: auto)")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (playfair, bevietnam, anton, pacifico, graffiti, dancing, oswald) or path")
    parser.add_argument("--output", type=str, default="demo_output.png", help="Output image file path")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG Guidance scale (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    run_demo(
        text=args.text,
        prompt=args.prompt,
        t_text=args.t_text,
        image_ref=args.image_ref,
        t_product=args.t_product,
        width=args.width,
        height=args.height,
        box_w=args.box_w,
        box_h=args.box_h,
        font=args.font,
        output_path=args.output,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

