#!/usr/bin/env python3
"""
scripts/probe_klein_distill_glyph.py

==================================================================================================
TENDOO AI - FLUX.2-KLEIN-4B DISTILLED IN-CONTEXT GLYPH PROBE
==================================================================================================

OBJECTIVE:
  Empirically verify whether the 4-step Guidance-Distilled model (FLUX.2-klein-4B Distill)
  preserves the 100% Vietnamese diacritic fidelity of the In-Context Glyph at t=10.0.

KEY ADVANTAGE IF SUCCESSFUL:
  - Base model: 50 steps * 2 (CFG) = 100 DiT forward passes (~18-25 seconds on 2x A30).
  - Distilled model: 4 steps * 1 (Single pass) = 4 DiT forward passes (~0.8-1.5 seconds).
  -> A 20x to 25x speedup, making poster generation near REAL-TIME (<2 seconds total)!

SAMPLING SPECIFICATIONS (per BFL FLUX.2 Docs):
  - Model: FLUX.2-klein-4B (Distilled)
  - Guidance mode: Guidance-distilled (guidance embedded in time-step modulation, NO CFG 2x batch)
  - Default steps: 4 steps (swept: 4, 6, 8 steps)
  - Default guidance: 1.0 (swept: 1.0, 1.5, 2.0)
  - Denoise function: src.flux2.sampling.denoise() (single batch forward with img_cond_seq)

EXECUTION CONSTRAINTS:
  - Runs on Remote Server (2x NVIDIA A30 or Single GPU).
  - Compliant with AGENTS.md Rule 28: Zero HTML, pure PNG + ASCII summary.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import batched_prc_txt, denoise, get_schedule, prc_img
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import load_ae, load_flow_model
from tendoo.glyph_engine import GlyphInfo, render_glyph

CANVAS_DEFAULT = (576, 1024)  # 9:16 target
DEFAULT_SEEDS = [42, 123, 777]


def encode_glyph_to_ref_tokens(
    ae: AutoEncoder, img: Image.Image, t_offset: float, device: str | torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes a glyph image into canonical 4D RoPE tokens at local origin (0, 0)."""
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32) / 127.5 - 1.0
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


def find_distill_checkpoint(base_checkpoint_dir: Path) -> Path | None:
    """Searches for the 4B distilled DiT checkpoint across standard paths."""
    candidates = [
        base_checkpoint_dir / "flux-2-klein-4b.safetensors",
        base_checkpoint_dir.parent / "FLUX.2-klein-4B" / "flux-2-klein-4b.safetensors",
        base_checkpoint_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4B/flux-2-klein-4b.safetensors"),
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4b.safetensors"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def run_distill_glyph_probe(
    text: str,
    prompt: str,
    font: str,
    steps_list: List[int],
    guidance_list: List[float],
    seeds: List[int],
    output_dir: str,
    distill_model_path: str | None,
    checkpoint_dir: str | None,
    canvas_w: int,
    canvas_h: int,
    box_w: int,
    box_h: int,
    t_offset: float = 10.0,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("🚀 TENDOO AI - FLUX.2-KLEIN-4B DISTILLED IN-CONTEXT GLYPH PROBE")
    print("=" * 100)
    print(f"  Text Payload      : {text}")
    print(f"  Prompt            : {prompt}")
    print(f"  Font              : {font}")
    print(f"  Glyph Box (WxH)   : {box_w}x{box_h}px")
    print(f"  Canvas (WxH)      : {canvas_w}x{canvas_h}px")
    print(f"  Steps Sweep       : {steps_list}")
    print(f"  Guidance Sweep    : {guidance_list}")
    print(f"  Seeds             : {seeds}")
    print(f"  Output Dir        : {out_path.resolve()}")

    # 1. Hardware setup
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = "cuda:0"
        device_ae = "cuda:1"
        print(f"  [GPU Setup] Dual GPU active: DiT on {device_dit}, VAE/Qwen3 on {device_ae}")
    elif num_gpus == 1:
        device_dit = "cuda:0"
        device_ae = "cuda:0"
        print(f"  [GPU Setup] Single GPU active: {device_dit}")
    else:
        device_dit = "cpu"
        device_ae = "cpu"
        print("  [!] WARNING: Running on CPU (testing syntax only)")

    # 2. Checkpoint resolution
    base_dir = Path(checkpoint_dir or "/home/jovyan/persistent-data/FLUX.2-klein-base-4B")
    if distill_model_path:
        model_file = Path(distill_model_path)
    else:
        model_file = find_distill_checkpoint(base_dir)

    if not model_file or not model_file.exists():
        print(f"\n[ERROR] Distilled DiT checkpoint not found!")
        print(f"  Searched locations around: {base_dir}")
        print("  Please specify explicitly with --distill_model_path <path_to_flux-2-klein-4b.safetensors>")
        sys.exit(1)

    print(f"\n[1/4] Loading Distilled DiT from: {model_file}")
    # Load Distilled DiT with Klein4BParams via environment variable
    os.environ["KLEIN_4B_MODEL_PATH"] = str(model_file)
    model = load_flow_model(
        model_name="flux.2-klein-4b",
        device=device_dit,
    )
    model.eval()

    print("\n[2/4] Loading VAE and Text Encoder (Qwen3-4B-FP8)...")
    if checkpoint_dir:
        os.environ["FLUX_CHECKPOINT_DIR"] = str(checkpoint_dir)
    ae = load_ae(model_name="flux.2-klein-base-4b", device=device_ae)
    ae.eval()

    text_encoder = load_qwen3_embedder(variant="4B", device=device_ae)

    # 3. Render Glyph Bitmap
    print("\n[3/4] Rendering Glyph Bitmap (Rule 29/31 Generous Envelope)...")
    glyph_info: GlyphInfo = render_glyph(
        text=text,
        font_name_or_path=font,
        target_width=box_w,
        target_height=box_h,
        auto_size=False,
    )
    glyph_file = out_path / "probe_glyph_input.png"
    glyph_info.image.save(glyph_file)
    print(f"  [Glyph Saved]   : {glyph_file.name} ({glyph_info.width_px}x{glyph_info.height_px}px, {glyph_info.font_size_pt}pt, {glyph_info.token_count}tok)")

    # Encode glyph to ref tokens at canonical t=10.0
    ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae, glyph_info.image, t_offset=t_offset, device=device_ae)
    ref_tokens = ref_tokens.to(device_dit)
    ref_ids = ref_ids.to(device_dit)

    # Encode prompt for Guidance-Distilled model:
    # IMPORTANT: Distilled models DO NOT USE CFG (no empty text, no 2x batch).
    # Only a single prompt embedding is passed.
    print(f"\n[Prompt Encoding] Single-batch distilled conditioning (No CFG doubling):")
    print(f"  Prompt: '{prompt}'")
    with torch.no_grad():
        txt_emb = text_encoder([prompt]).to(device=device_dit, dtype=torch.bfloat16)
    txt_tokens, txt_ids = batched_prc_txt(txt_emb)
    print(f"  Text Tokens     : {txt_tokens.shape} on {txt_tokens.device}")

    # Prepare Canvas Latent Dimensions
    canvas_w = (canvas_w // 16) * 16
    canvas_h = (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    # 4. Sweep Grid: Seeds x Steps x Guidance
    print("\n[4/4] Executing Distilled Inference Sweep...")
    results: List[Dict[str, Any]] = []

    for steps in steps_list:
        for guidance in guidance_list:
            for seed in seeds:
                tag = f"distill_steps{steps}_g{guidance}_seed{seed}"
                print(f"\n▶️ Running: {tag} (steps={steps}, guidance={guidance}, seed={seed})...")
                t_start = time.time()

                torch.manual_seed(seed)
                z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
                img_tokens, img_ids = prc_img(z_init[0])
                img_tokens = img_tokens.unsqueeze(0).to(device_dit)
                img_ids = img_ids.unsqueeze(0).to(device_dit)

                # Time Schedule for num_steps
                timesteps = get_schedule(num_steps=steps, image_seq_len=img_tokens.shape[1])

                with torch.no_grad():
                    # BFL Guidance-Distilled Denoise (Single batch, 1 forward pass per step!)
                    out_tokens = denoise(
                        model=model,
                        img=img_tokens,
                        img_ids=img_ids,
                        txt=txt_tokens,
                        txt_ids=txt_ids,
                        timesteps=timesteps,
                        guidance=guidance,
                        img_cond_seq=ref_tokens,
                        img_cond_seq_ids=ref_ids,
                    )

                    # Decode VAE
                    lat_2d = rearrange(out_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
                    img_out = ae.decode(lat_2d.to(device_ae))
                    dur_dit = time.time() - t_start

                    # Convert to PIL
                    pil_img = Image.fromarray(
                        ((img_out[0].float().clamp(-1, 1) + 1.0) * 127.5)
                        .permute(1, 2, 0)
                        .byte()
                        .cpu()
                        .numpy()
                    )

                    out_file = out_path / f"{tag}.png"
                    pil_img.save(out_file)
                    print(f"  [✓] Completed in {dur_dit:.2f}s! -> Saved: {out_file.name}")

                    results.append({
                        "steps": steps,
                        "guidance": guidance,
                        "seed": seed,
                        "duration_s": round(dur_dit, 2),
                        "file": out_file.name,
                    })

    # Summary Report
    print("\n" + "=" * 90)
    print("📊 FLUX.2-KLEIN-4B DISTILLED IN-CONTEXT GLYPH PROBE SUMMARY")
    print("=" * 90)
    print(f"{'STEPS':<8} | {'GUIDANCE':<10} | {'SEED':<8} | {'INFERENCE TIME':<18} | {'OUTPUT FILE':<35}")
    print("-" * 90)
    for r in results:
        print(f"{r['steps']:<8} | {r['guidance']:<10} | {r['seed']:<8} | {r['duration_s']}s{' ':<13} | {r['file']:<35}")
    print("=" * 90)
    print(f"\n[Done] All distilled outputs saved in: {out_path.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI FLUX.2-klein-4B Distill Glyph Probe")
    parser.add_argument("--text", type=str, default="CHINH PHỤC MỌI GIỚI HẠN",
                        help="Text payload for Hero Title glyph")
    parser.add_argument("--prompt", type=str,
                        default="Poster quảng cáo thể thao ngoài trời hiện đại, nền ánh sáng hoàng hôn điện ảnh kịch tính, dòng chữ tiêu đề lớn 3D dập nổi mạ vàng kim loại sắc nét ở phía trên, bố cục sạch sẽ chuyên nghiệp, không có watermark",
                        help="Text prompt for background & material")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 6, 8],
                        help="Steps to sweep (default: 4 6 8)")
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.0, 1.5],
                        help="Guidance values to sweep (default: 1.0 1.5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds to test")
    parser.add_argument("--box_w", type=int, default=512, help="Glyph box width (default: 512)")
    parser.add_argument("--box_h", type=int, default=224, help="Glyph box height (default: 224)")
    parser.add_argument("--canvas_w", type=int, default=CANVAS_DEFAULT[0], help="Canvas width")
    parser.add_argument("--canvas_h", type=int, default=CANVAS_DEFAULT[1], help="Canvas height")
    parser.add_argument("--distill_model_path", type=str, default=None,
                        help="Direct path to flux-2-klein-4b.safetensors")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
                        help="Base persistent data dir containing text_encoder and VAE")
    parser.add_argument("--output_dir", type=str, default="output_probe_distill_glyph",
                        help="Output directory")

    args = parser.parse_args()
    run_distill_glyph_probe(
        text=args.text,
        prompt=args.prompt,
        font=args.font,
        steps_list=args.steps,
        guidance_list=args.guidance,
        seeds=args.seeds,
        output_dir=args.output_dir,
        distill_model_path=args.distill_model_path,
        checkpoint_dir=args.checkpoint_dir,
        canvas_w=args.canvas_w,
        canvas_h=args.canvas_h,
        box_w=args.box_w,
        box_h=args.box_h,
    )


if __name__ == "__main__":
    main()
