#!/usr/bin/env python3
"""
scripts/probe_sequential_inpainting.py

==================================================================================================
TENDOO AI - TRAINING-FREE 2-STAGE SEQUENTIAL INPAINTING PROBE (DIRECTION 3)
==================================================================================================

WHY THIS EXPERIMENT?
  1. Base FLUX.2 Klein 4B Base has been proven 100% reliable when rendering a SINGLE glyph
     at canonical t=10.0 origin (Rule 10, Rule 31, Step 1). Both Title ("TUYỂN DỤNG NHÂN TÀI")
     and Subtitle ("BỨT PHÁ MỌI GIỚI HẠN") render with 100% Vietnamese diacritic fidelity in isolation.
  2. Crosstalk occurs ONLY when N >= 2 reference slots are placed in the same Attention matrix.
  3. Regional Parallel Diffusion (Direction 2) failed because an external spatial velocity mask
     at split_y=0.5 cuts through the body of the letters ("BỨT HÁ MI...").
  4. SEQUENTIAL INPAINTING solves this completely without training:
     - Pass 1: Generate canvas with Title at t=10.0 (50 steps) -> Title is 100% sharp and intact.
     - Pass 2: Freeze the Title region (top), and inpaint the Subtitle at t=10.0 into the bottom
       region using Flow Matching Known-Latent Replacement with a smooth cosine transition band.
     - At NO POINT do Title and Subtitle coexist in the same forward pass!
     - The bottom half attends to the top half at every step, naturally matching lighting, color,
       and aesthetic coherence.

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
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import batched_prc_txt, get_schedule, prc_img
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

# Clean, robust 3D metallic prompts without semantic subordination words
TITLE_PROMPT = (
    "Poster quảng cáo phong cách công nghệ hiện đại, nền gradient xanh dương đậm sang trọng, "
    "dòng chữ tiêu đề 3D dập nổi mạ vàng kim loại sắc nét ở phía trên, bố cục sạch sẽ chuyên "
    "nghiệp, không có chữ ký, không có watermark"
)

SUBTITLE_PROMPT = (
    "Poster quảng cáo phong cách công nghệ hiện đại, nền gradient xanh dương đậm sang trọng, "
    "dòng chữ 3D dập nổi mạ vàng kim loại sắc nét ở phía dưới, bố cục sạch sẽ chuyên nghiệp, không "
    "có chữ ký, không có watermark"
)


# ==================================================================================================
# 2. 4D ROPE ENCODING & SMOOTH INPAINTING MASK
# ==================================================================================================

def encode_glyph_to_ref_tokens(
    ae: AutoEncoder, glyph_img: Image.Image, t_offset: float, device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Canonical encoding: local (0,0) origin at canonical t_offset (Rule 30)."""
    arr = np.array(glyph_img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
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


def compute_inpainting_mask(
    lat_h: int, lat_w: int, keep_y_end: float = 0.38, inpaint_y_start: float = 0.52, device=None,
) -> torch.Tensor:
    """
    Computes a smooth 3-zone inpainting mask M(h, w) in [0, 1]:
      - y <= keep_y_end: M = 0.0 (100% frozen Pass 1 Title region).
      - keep_y_end < y < inpaint_y_start: Smooth Cosine transition (seamless blending, no sharp seam).
      - y >= inpaint_y_start: M = 1.0 (100% freshly inpainted Subtitle region).
    Returns tensor shaped [lat_h, lat_w].
    """
    h_idx = torch.arange(lat_h, device=device, dtype=torch.float32)
    y_grid = (h_idx + 0.5) / float(lat_h)

    mask_1d = torch.zeros_like(y_grid)
    in_transition = (y_grid > keep_y_end) & (y_grid < inpaint_y_start)
    in_bottom = y_grid >= inpaint_y_start

    # Cosine ramp from 0 to 1
    t = (y_grid[in_transition] - keep_y_end) / (inpaint_y_start - keep_y_end)
    mask_1d[in_transition] = 0.5 * (1.0 - torch.cos(torch.pi * t))
    mask_1d[in_bottom] = 1.0

    mask_2d = mask_1d.unsqueeze(1).expand(lat_h, lat_w)
    return mask_2d


# ==================================================================================================
# 3. DENOISING LOOPS: PASS 1 (FULL ODE) & PASS 2 (INPAINTING ODE)
# ==================================================================================================

def denoise_single_slot(
    model: Flux2,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: List[float],
    guidance: float,
    ref_tokens: torch.Tensor,
    ref_ids: torch.Tensor,
) -> torch.Tensor:
    """Standard Euler ODE flow matching for a single reference slot."""
    n_canvas = img.shape[1]
    orig_dtype = img.dtype

    ref_tokens_cfg = torch.cat([ref_tokens, ref_tokens], dim=0)
    ref_ids_cfg = torch.cat([ref_ids, ref_ids], dim=0)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((2,), t_curr, dtype=img.dtype, device=img.device)
        img_cfg = torch.cat([img, img], dim=0)
        img_ids_cfg = torch.cat([img_ids, img_ids], dim=0)

        img_input = torch.cat([img_cfg, ref_tokens_cfg], dim=1)
        img_input_ids = torch.cat([img_ids_cfg, ref_ids_cfg], dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=None,
        )
        pred = pred[:, :n_canvas]
        pred_uncond, pred_cond = pred.chunk(2)
        v_pred = pred_uncond + guidance * (pred_cond - pred_uncond)

        img = (img + (t_prev - t_curr) * v_pred).to(orig_dtype)

    return img


def denoise_flow_matching_inpaint(
    model: Flux2,
    z_init_tokens: torch.Tensor,   # [1, n_canvas, C] noise state z_1
    z_known_tokens: torch.Tensor,  # [1, n_canvas, C] clean ground truth state z_0 from Pass 1
    mask_tokens: torch.Tensor,     # [1, n_canvas, 1] smooth spatial mask (0 = keep, 1 = inpaint)
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: List[float],
    guidance: float,
    ref_tokens: torch.Tensor,
    ref_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Flow Matching Inpainting via Known-Latent Trajectory Replacement:
      At each step t_curr -> t_prev:
        1. Model predicts velocity v_theta on current canvas x_curr.
        2. x_model_next = x_curr + (t_prev - t_curr) * v_theta
        3. Known clean Title trajectory at t_prev:
           x_known_next = (1.0 - t_prev) * z_known_tokens + t_prev * z_init_tokens
        4. Blend:
           x_curr = (1.0 - mask_tokens) * x_known_next + mask_tokens * x_model_next
    """
    n_canvas = z_init_tokens.shape[1]
    orig_dtype = z_init_tokens.dtype

    # Start with initial noise
    img = z_init_tokens.clone()

    ref_tokens_cfg = torch.cat([ref_tokens, ref_tokens], dim=0)
    ref_ids_cfg = torch.cat([ref_ids, ref_ids], dim=0)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((2,), t_curr, dtype=img.dtype, device=img.device)
        img_cfg = torch.cat([img, img], dim=0)
        img_ids_cfg = torch.cat([img_ids, img_ids], dim=0)

        img_input = torch.cat([img_cfg, ref_tokens_cfg], dim=1)
        img_input_ids = torch.cat([img_ids_cfg, ref_ids_cfg], dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=None,
        )
        pred = pred[:, :n_canvas]
        pred_uncond, pred_cond = pred.chunk(2)
        v_pred = pred_uncond + guidance * (pred_cond - pred_uncond)

        # Model's step
        img_model_next = img + (t_prev - t_curr) * v_pred

        # Known trajectory of Pass 1 Title at t_prev
        img_known_next = (1.0 - t_prev) * z_known_tokens + t_prev * z_init_tokens

        # Seamless blended replacement
        img = ((1.0 - mask_tokens) * img_known_next + mask_tokens * img_model_next).to(orig_dtype)

    # At t=0, enforce clean known latent in the unmasked region
    img = ((1.0 - mask_tokens) * z_known_tokens + mask_tokens * img).to(orig_dtype)
    return img


# ==================================================================================================
# 4. MAIN PROBE ORCHESTRATION
# ==================================================================================================

def run_sequential_probe(
    seeds: List[int],
    font: str = "bevietnam",
    output_dir: str = "output_sequential_inpainting",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    envelope_w: int = 512,
    envelope_h: int = 224,
    keep_y_end: float = 0.38,
    inpaint_y_start: float = 0.52,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(" [*] TENDOO AI - SEQUENTIAL INPAINTING PROBE (DIRECTION 3, TRAINING-FREE)")
    print("=" * 100)
    print(f"  Title      : \"{TITLE_TEXT.replace(chr(10), ' ')}\" (top)")
    print(f"  Subtitle   : \"{SUBTITLE_TEXT.replace(chr(10), ' ')}\" (bottom)")
    print(f"  Canvas     : {CANVAS[0]}x{CANVAS[1]} (9:16 target)")
    print(f"  Envelope   : {envelope_w}x{envelope_h}px (Mode B fixed envelope, 448 tokens)")
    print(f"  Mask Zones : Keep Title [0.0 -> {keep_y_end:.2f}] | Cosine Transition [{keep_y_end:.2f} -> {inpaint_y_start:.2f}] | Inpaint Subtitle [{inpaint_y_start:.2f} -> 1.0]")
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
        txt_title, txt_ids_title = batched_prc_txt(text_encoder(["", TITLE_PROMPT]))
        txt_title, txt_ids_title = txt_title.to(device_dit), txt_ids_title.to(device_dit)

        txt_subtitle, txt_ids_subtitle = batched_prc_txt(text_encoder(["", SUBTITLE_PROMPT]))
        txt_subtitle, txt_ids_subtitle = txt_subtitle.to(device_dit), txt_ids_subtitle.to(device_dit)

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

    print("\n[3/4] Rendering Glyphs & Computing Inpainting Masks...")
    # Render glyphs via Mode B fixed envelope
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

    # Encode reference tokens at canonical t=10.0 (Rule 30)
    title_ref_tokens, title_ref_ids = encode_glyph_to_ref_tokens(ae, title_info.image, 10.0, device_ae)
    subtitle_ref_tokens, subtitle_ref_ids = encode_glyph_to_ref_tokens(ae, subtitle_info.image, 10.0, device_ae)

    title_ref_tokens = title_ref_tokens.to(device_dit)
    title_ref_ids = title_ref_ids.to(device_dit)
    subtitle_ref_tokens = subtitle_ref_tokens.to(device_dit)
    subtitle_ref_ids = subtitle_ref_ids.to(device_dit)

    # Compute inpainting mask
    mask_2d = compute_inpainting_mask(lat_h, lat_w, keep_y_end=keep_y_end, inpaint_y_start=inpaint_y_start, device=device_dit)
    mask_tokens = rearrange(mask_2d, "h w -> 1 (h w) 1")
    # Save visual representation of mask
    mask_vis = Image.fromarray((mask_2d.cpu().numpy() * 255).astype(np.uint8)).resize((canvas_w, canvas_h), resample=Image.NEAREST)
    mask_vis.save(out_path / "inpainting_mask_preview.png")
    print(f"  Saved inpainting mask visualization to: {out_path / 'inpainting_mask_preview.png'}")

    print(f"\n[4/4] Executing Sequential Inpainting across {len(seeds)} seed(s)...\n")
    results_summary: List[Dict[str, Any]] = []

    for idx, seed in enumerate(seeds, 1):
        print("=" * 80)
        print(f"▶️  SEED [{idx}/{len(seeds)}]: {seed}")
        print("=" * 80)
        t_start = time.time()

        torch.manual_seed(seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens_init, img_ids = prc_img(z_init[0])
        img_tokens_init = img_tokens_init.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens_init.shape[1])

        # ------------------------------------------------------------------------------------------
        # PASS 1: GENERATE CANVAS WITH TITLE ONLY (t=10.0)
        # ------------------------------------------------------------------------------------------
        print(f"  [Pass 1] Generating canvas with Title ({num_steps} steps Euler ODE)...")
        t_p1 = time.time()
        with torch.no_grad():
            z0_title_tokens = denoise_single_slot(
                model=model,
                img=img_tokens_init.clone(),
                img_ids=img_ids,
                txt=txt_title,
                txt_ids=txt_ids_title,
                timesteps=timesteps,
                guidance=guidance,
                ref_tokens=title_ref_tokens,
                ref_ids=title_ref_ids,
            )

            # Decode Pass 1 image
            pass1_lat_2d = rearrange(z0_title_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
            pass1_img = ae.decode(pass1_lat_2d.to(device_ae))
            pass1_pil = Image.fromarray(((pass1_img[0].float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).byte().cpu().numpy())
            pass1_file = out_path / f"seed{seed}_pass1_title.png"
            pass1_pil.save(pass1_file)
            dur_p1 = time.time() - t_p1
            print(f"  [Pass 1] Done in {dur_p1:.1f}s -> Saved: {pass1_file.name}")

        # ------------------------------------------------------------------------------------------
        # PASS 2: INPAINT SUBTITLE INTO BOTTOM REGION (t=10.0)
        # ------------------------------------------------------------------------------------------
        print(f"  [Pass 2] Inpainting Subtitle with known-latent trajectory replacement...")
        t_p2 = time.time()
        with torch.no_grad():
            z_completed_tokens = denoise_flow_matching_inpaint(
                model=model,
                z_init_tokens=img_tokens_init,
                z_known_tokens=z0_title_tokens,
                mask_tokens=mask_tokens,
                img_ids=img_ids,
                txt=txt_subtitle,
                txt_ids=txt_ids_subtitle,
                timesteps=timesteps,
                guidance=guidance,
                ref_tokens=subtitle_ref_tokens,
                ref_ids=subtitle_ref_ids,
            )

            # Decode Pass 2 completed image
            completed_lat_2d = rearrange(z_completed_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
            completed_img = ae.decode(completed_lat_2d.to(device_ae))
            completed_pil = Image.fromarray(((completed_img[0].float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).byte().cpu().numpy())
            completed_file = out_path / f"seed{seed}_pass2_completed_poster.png"
            completed_pil.save(completed_file)
            dur_p2 = time.time() - t_p2
            print(f"  [Pass 2] Done in {dur_p2:.1f}s -> Saved: {completed_file.name}")

        # ------------------------------------------------------------------------------------------
        # SIDE-BY-SIDE COMPARISON
        # ------------------------------------------------------------------------------------------
        comp_w = canvas_w * 2 + 30
        comp_h = canvas_h + 60
        comp_img = Image.new("RGB", (comp_w, comp_h), color=(25, 25, 25))
        draw = ImageDraw.Draw(comp_img)

        comp_img.paste(pass1_pil, (10, 50))
        comp_img.paste(completed_pil, (canvas_w + 20, 50))

        draw.text((20, 15), f"PASS 1: Title Only (Seed {seed})", fill=(255, 215, 0))
        draw.text((canvas_w + 30, 15), f"PASS 2: Title + Inpainted Subtitle (Seed {seed})", fill=(0, 255, 127))

        comp_file = out_path / f"seed{seed}_comparison.png"
        comp_img.save(comp_file)
        dur_total = time.time() - t_start
        print(f"  [+] Comparison image: {comp_file.name} (Total time: {dur_total:.1f}s)\n")

        results_summary.append({
            "seed": seed,
            "pass1_file": str(pass1_file),
            "completed_file": str(completed_file),
            "comparison_file": str(comp_file),
            "dur_pass1": round(dur_p1, 1),
            "dur_pass2": round(dur_p2, 1),
            "dur_total": round(dur_total, 1),
        })

    # ==============================================================================================
    # ASCII SUMMARY TABLE (Rule 28 Compliant: NO HTML)
    # ==============================================================================================
    print("=" * 100)
    print(f"{'SEED':<8} | {'PASS 1 TITLE':<30} | {'COMPLETED POSTER':<35} | {'TIME (P1 / P2 / TOT)':<20}")
    print("-" * 100)
    for res in results_summary:
        time_str = f"{res['dur_pass1']}s / {res['dur_pass2']}s / {res['dur_total']}s"
        print(f"{res['seed']:<8} | {Path(res['pass1_file']).name:<30} | {Path(res['completed_file']).name:<35} | {time_str:<20}")
    print("=" * 100)
    print(f"\n[✓] Sequential Inpainting probe finished successfully. Outputs in: {out_path.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI 2-Stage Sequential Inpainting Probe (Direction 3)")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to run")
    parser.add_argument("--output_dir", type=str, default="output_sequential_inpainting", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")
    parser.add_argument("--envelope_w", type=int, default=512, help="Glyph envelope width in px (default: 512)")
    parser.add_argument("--envelope_h", type=int, default=224, help="Glyph envelope height in px (default: 224)")
    parser.add_argument("--keep_y_end", type=float, default=0.38, help="Top fraction to freeze (default: 0.38)")
    parser.add_argument("--inpaint_y_start", type=float, default=0.52, help="Bottom fraction to inpaint (default: 0.52)")

    args = parser.parse_args()
    run_sequential_probe(
        seeds=args.seeds,
        font=args.font,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps,
        guidance=args.guidance,
        envelope_w=args.envelope_w,
        envelope_h=args.envelope_h,
        keep_y_end=args.keep_y_end,
        inpaint_y_start=args.inpaint_y_start,
    )


if __name__ == "__main__":
    main()
