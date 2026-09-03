#!/usr/bin/env python3
"""
scripts/probe_chained_reference_conditioning.py

==================================================================================================
TENDOO AI - CHAINED REFERENCE CONDITIONING PROBE (DIRECTION 4)
==================================================================================================

WHY THIS EXPERIMENT?
  1. Base FLUX.2 Klein 4B Base struggles when TWO SPARSE GLYPHS (black/white text bitmaps) are passed
     simultaneously at t=10 and t=20, because both are small/sparse and fight in the joint attention space.
  2. HOWEVER, Rule 8 & exp45 proved that when given a FULL NATURAL REFERENCE IMAGE (2304 tokens, rich
     continuous manifold features at t=20 or t=60) + ONE TEXT GLYPH (at t=10), the model preserves
     the reference image with near-100% fidelity while drawing the 3D text cleanly!
  3. USER'S HYPOTHESIS:
     - Pass 1: Render Title text glyph at t=10.0 -> Generates high-quality base poster with Title.
     - Pass 2: Feed Pass 1's full generated poster as Reference 1 at t=20.0, and feed Subtitle glyph
       as Reference 2 at t=10.0. Canvas denoises from noise, conditioned on both!
     - Let FLUX's native image-reference preservation capability lock the upper visual scene and Title,
       while the t=10 glyph guides the synthesis of the 3D Subtitle at the bottom.

EXECUTION REQUIREMENTS:
  - 2x NVIDIA A30 (DiT on cuda:0, VAE/Qwen3 on cuda:1) or Single GPU.
  - Zero HTML output (AGENTS.md Rule 28 compliant: PNGs + ASCII summary only).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import load_ae, load_flow_model
from tendoo.glyph_engine import GlyphInfo, render_glyph

# ==================================================================================================
# 1. LAYOUT & PROMPTS
# ==================================================================================================

TITLE_TEXT = "TUYỂN DỤNG\nNHÂN TÀI"
SUBTITLE_TEXT = "BỨT PHÁ MỌI\nGIỚI HẠN"
CANVAS = (576, 1024)  # 9:16 target
DEFAULT_SEEDS = [42, 123, 777]

TITLE_PROMPT = (
    "Poster quảng cáo phong cách công nghệ hiện đại, nền gradient xanh dương đậm sang trọng, "
    "dòng chữ tiêu đề 3D dập nổi mạ vàng kim loại sắc nét ở phía trên, bố cục sạch sẽ chuyên "
    "nghiệp, không có chữ ký, không có watermark"
)

# Pass 2 prompt: instructs the model to preserve the reference poster style while adding the bottom 3D text
PASS2_PROMPT = (
    "Poster quảng cáo phong cách công nghệ hiện đại kế thừa toàn bộ phong cách và tiêu đề của poster "
    "tham chiếu, dòng chữ 3D dập nổi mạ vàng kim loại sắc nét ở phía dưới, bố cục sạch sẽ chuyên "
    "nghiệp, không có chữ ký, không có watermark"
)


# ==================================================================================================
# 2. ENCODING FUNCTIONS
# ==================================================================================================

def encode_image_to_ref_tokens(
    ae: AutoEncoder, img: Image.Image, t_offset: float, device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes any image (glyph or full natural poster) into canonical 4D RoPE tokens at local origin."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latent = ae.encode(tensor)
    ref_tokens, _ = prc_img(latent[0])
    ref_tokens = ref_tokens.unsqueeze(0)
    g_h, g_w = latent.shape[2], latent.shape[3]
    t_coords = torch.full((g_h, g_w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(g_h, dtype=torch.float32, device=device).unsqueeze(1).expand(g_h, g_w)
    w_coords = torch.arange(g_w, dtype=torch.float32, device=device).unsqueeze(0).expand(g_h, g_w)
    l_coords = torch.zeros((g_h, g_w), dtype=torch.float32, device=device)
    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0)
    return ref_tokens, ref_ids


# ==================================================================================================
# 3. RUNNER ORCHESTRATION
# ==================================================================================================

def run_chained_conditioning_probe(
    seeds: List[int],
    font: str = "bevietnam",
    output_dir: str = "output_chained_reference_conditioning",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    envelope_w: int = 512,
    envelope_h: int = 224,
    t_img: float = 20.0,
    t_text: float = 10.0,
    pass2_prompt: str = PASS2_PROMPT,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(" [*] TENDOO AI - CHAINED REFERENCE CONDITIONING PROBE (DIRECTION 4)")
    print("=" * 100)
    print(f"  Title      : \"{TITLE_TEXT.replace(chr(10), ' ')}\" (Pass 1 @ t=10.0)")
    print(f"  Subtitle   : \"{SUBTITLE_TEXT.replace(chr(10), ' ')}\" (Pass 2 @ t={t_text})")
    print(f"  Poster Ref : Pass 1 output image chained into Pass 2 @ t={t_img}")
    print(f"  Canvas     : {CANVAS[0]}x{CANVAS[1]} (9:16 target)")
    print(f"  Envelope   : {envelope_w}x{envelope_h}px (Mode B fixed envelope)")
    print(f"  Seeds      : {seeds}")
    print(f"  Steps / CFG: {num_steps} steps | CFG guidance = {guidance:.1f}")

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print(f"[*] Dual GPU Mode: DiT on {device_dit} | VAE & Qwen3 on {device_ae}")
    else:
        device_dit = device_ae = device_te = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[*] Single Device Mode: {device_dit}")

    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    print("\n[1/4] Loading FLUX.2 Klein 4B Base models (AE + DiT + Qwen3)...")
    ae = load_ae(model_name, device=device_ae)
    model = load_flow_model(model_name, device=device_dit)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    print("\n[2/4] Encoding prompts via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt_pass1, txt_ids_pass1 = batched_prc_txt(text_encoder(["", TITLE_PROMPT]))
        txt_pass1, txt_ids_pass1 = txt_pass1.to(device_dit), txt_ids_pass1.to(device_dit)

        txt_pass2, txt_ids_pass2 = batched_prc_txt(text_encoder(["", pass2_prompt]))
        txt_pass2, txt_ids_pass2 = txt_pass2.to(device_dit), txt_ids_pass2.to(device_dit)

    if num_gpus >= 2:
        try:
            text_encoder.model.to("cpu")
        except Exception:
            pass
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    canvas_w, canvas_h = CANVAS
    canvas_w = (canvas_w // 16) * 16
    canvas_h = (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    print("\n[3/4] Rendering Title & Subtitle Glyphs...")
    title_info = render_glyph(
        text=TITLE_TEXT, font_name_or_path=font, auto_size=False,
        target_width=envelope_w, target_height=envelope_h,
    )
    subtitle_info = render_glyph(
        text=SUBTITLE_TEXT, font_name_or_path=font, auto_size=False,
        target_width=envelope_w, target_height=envelope_h,
    )
    print(f"  [Title]    : {title_info.width_px}x{title_info.height_px}px {title_info.font_size_pt}pt {title_info.token_count}tok :: {title_info.lines}")
    print(f"  [Subtitle] : {subtitle_info.width_px}x{subtitle_info.height_px}px {subtitle_info.font_size_pt}pt {subtitle_info.token_count}tok :: {subtitle_info.lines}")
    title_info.image.save(out_path / "title_glyph.png")
    subtitle_info.image.save(out_path / "subtitle_glyph.png")

    # Encode Title at t=10.0 for Pass 1, and Subtitle at t_text (default 10.0) for Pass 2
    title_ref_tokens, title_ref_ids = encode_image_to_ref_tokens(ae, title_info.image, 10.0, device_ae)
    subtitle_ref_tokens, subtitle_ref_ids = encode_image_to_ref_tokens(ae, subtitle_info.image, t_text, device_ae)

    title_ref_tokens = title_ref_tokens.to(device_dit)
    title_ref_ids = title_ref_ids.to(device_dit)
    subtitle_ref_tokens = subtitle_ref_tokens.to(device_dit)
    subtitle_ref_ids = subtitle_ref_ids.to(device_dit)

    print(f"\n[4/4] Executing Chained Conditioning across {len(seeds)} seed(s)...\n")
    results_summary: List[Dict[str, Any]] = []

    for idx, seed in enumerate(seeds, 1):
        print("=" * 80)
        print(f"▶️  SEED [{idx}/{len(seeds)}]: {seed}")
        print("=" * 80)
        t_start = time.time()

        # ------------------------------------------------------------------------------------------
        # STEP 1: GENERATE BASE POSTER WITH TITLE AT t=10.0
        # ------------------------------------------------------------------------------------------
        print(f"  [Step 1] Generating Base Poster with Title (t=10.0)...")
        t_s1 = time.time()
        torch.manual_seed(seed)
        z_init_1 = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens_1, img_ids_1 = prc_img(z_init_1[0])
        img_tokens_1 = img_tokens_1.unsqueeze(0).to(device_dit)
        img_ids_1 = img_ids_1.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens_1.shape[1])

        with torch.no_grad():
            pass1_latent_tokens = denoise_cfg(
                model=model,
                img=img_tokens_1,
                img_ids=img_ids_1,
                txt=txt_pass1,
                txt_ids=txt_ids_pass1,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=title_ref_tokens,
                img_cond_seq_ids=title_ref_ids,
            )

            # Decode Step 1 Base Poster
            pass1_lat_2d = rearrange(pass1_latent_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
            pass1_img = ae.decode(pass1_lat_2d.to(device_ae))
            pass1_pil = Image.fromarray(((pass1_img[0].float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).byte().cpu().numpy())
            pass1_file = out_path / f"seed{seed}_step1_title_poster.png"
            pass1_pil.save(pass1_file)
            dur_s1 = time.time() - t_s1
            print(f"  [Step 1] Completed in {dur_s1:.1f}s -> Saved: {pass1_file.name}")

        # ------------------------------------------------------------------------------------------
        # STEP 2: CHAINED CONDITIONING (Pass 1 Poster @ t=20.0 + Subtitle Glyph @ t=10.0)
        # ------------------------------------------------------------------------------------------
        print(f"  [Step 2] Encoding Step 1 Poster as Reference Image at t={t_img}...")
        pass1_ref_tokens, pass1_ref_ids = encode_image_to_ref_tokens(ae, pass1_pil, t_img, device_ae)
        pass1_ref_tokens = pass1_ref_tokens.to(device_dit)
        pass1_ref_ids = pass1_ref_ids.to(device_dit)
        print(f"  [Step 2] Poster tokens: {pass1_ref_tokens.shape[1]}tok | Subtitle tokens: {subtitle_ref_tokens.shape[1]}tok")

        # Combine references: Subtitle Glyph (@ t_text) + Pass 1 Poster (@ t_img)
        all_ref_tokens = torch.cat([subtitle_ref_tokens, pass1_ref_tokens], dim=1)
        all_ref_ids = torch.cat([subtitle_ref_ids, pass1_ref_ids], dim=1)

        print(f"  [Step 2] Denoising canvas with Chained References ({num_steps} steps Euler ODE)...")
        t_s2 = time.time()
        torch.manual_seed(seed + 1000)  # fresh seed for Pass 2 canvas
        z_init_2 = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens_2, img_ids_2 = prc_img(z_init_2[0])
        img_tokens_2 = img_tokens_2.unsqueeze(0).to(device_dit)
        img_ids_2 = img_ids_2.unsqueeze(0).to(device_dit)

        with torch.no_grad():
            pass2_latent_tokens = denoise_cfg(
                model=model,
                img=img_tokens_2,
                img_ids=img_ids_2,
                txt=txt_pass2,
                txt_ids=txt_ids_pass2,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=all_ref_tokens,
                img_cond_seq_ids=all_ref_ids,
            )

            # Decode Step 2 Chained Completed Poster
            pass2_lat_2d = rearrange(pass2_latent_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
            pass2_img = ae.decode(pass2_lat_2d.to(device_ae))
            pass2_pil = Image.fromarray(((pass2_img[0].float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).byte().cpu().numpy())
            pass2_file = out_path / f"seed{seed}_step2_chained_poster.png"
            pass2_pil.save(pass2_file)
            dur_s2 = time.time() - t_s2
            print(f"  [Step 2] Completed in {dur_s2:.1f}s -> Saved: {pass2_file.name}")

        # ------------------------------------------------------------------------------------------
        # SIDE-BY-SIDE COMPARISON
        # ------------------------------------------------------------------------------------------
        comp_w = canvas_w * 2 + 30
        comp_h = canvas_h + 60
        comp_img = Image.new("RGB", (comp_w, comp_h), color=(25, 25, 25))
        draw = ImageDraw.Draw(comp_img)

        comp_img.paste(pass1_pil, (10, 50))
        comp_img.paste(pass2_pil, (canvas_w + 20, 50))

        draw.text((20, 15), f"STEP 1: Title Poster (Ref for Step 2)", fill=(255, 215, 0))
        draw.text((canvas_w + 30, 15), f"STEP 2: Chained (Poster @ t={t_img} + Subtitle @ t={t_text})", fill=(0, 255, 127))

        comp_file = out_path / f"seed{seed}_chained_comparison.png"
        comp_img.save(comp_file)
        dur_total = time.time() - t_start
        print(f"  [+] Comparison image: {comp_file.name} (Total time: {dur_total:.1f}s)\n")

        results_summary.append({
            "seed": seed,
            "step1_file": str(pass1_file),
            "step2_file": str(pass2_file),
            "comparison_file": str(comp_file),
            "dur_s1": round(dur_s1, 1),
            "dur_s2": round(dur_s2, 1),
            "dur_total": round(dur_total, 1),
        })

    # ==============================================================================================
    # ASCII SUMMARY TABLE (Rule 28 Compliant: NO HTML)
    # ==============================================================================================
    print("=" * 100)
    print(f"{'SEED':<8} | {'STEP 1 (TITLE POSTER)':<32} | {'STEP 2 (CHAINED POSTER)':<35} | {'TIME (S1 / S2 / TOT)':<20}")
    print("-" * 100)
    for res in results_summary:
        time_str = f"{res['dur_s1']}s / {res['dur_s2']}s / {res['dur_total']}s"
        print(f"{res['seed']:<8} | {Path(res['step1_file']).name:<32} | {Path(res['step2_file']).name:<35} | {time_str:<20}")
    print("=" * 100)
    print(f"\n[✓] Chained Reference Conditioning probe finished successfully. Outputs in: {out_path.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Chained Reference Conditioning Probe (Direction 4)")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to run")
    parser.add_argument("--output_dir", type=str, default="output_chained_reference_conditioning", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")
    parser.add_argument("--envelope_w", type=int, default=512, help="Glyph envelope width in px (default: 512)")
    parser.add_argument("--envelope_h", type=int, default=224, help="Glyph envelope height in px (default: 224)")
    parser.add_argument("--t_img", type=float, default=20.0, help="Time offset for Step 1 Poster reference (default: 20.0, can try 30.0 or 60.0)")
    parser.add_argument("--t_text", type=float, default=10.0, help="Time offset for Subtitle glyph reference (default: 10.0)")
    parser.add_argument("--pass2_prompt", type=str, default=PASS2_PROMPT, help="Custom prompt for Step 2")

    args = parser.parse_args()
    run_chained_conditioning_probe(
        seeds=args.seeds,
        font=args.font,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps,
        guidance=args.guidance,
        envelope_w=args.envelope_w,
        envelope_h=args.envelope_h,
        t_img=args.t_img,
        t_text=args.t_text,
        pass2_prompt=args.pass2_prompt,
    )


if __name__ == "__main__":
    main()
