#!/usr/bin/env python3
"""
scripts/diagnose_case3_tightcrop_spacing.py

==================================================================================================
TENDOO AI - DEEP DIAGNOSTIC ON CASE 3 ("SIÊU SALE 50% / DUY NHẤT HÔM NAY"):
TIGHT CROP, LINE SPACING & ASPECT RATIO BENCHMARK (FLUX.2-DISTILL 8 STEPS)
==================================================================================================

OBJECTIVE:
  Diagnose and resolve the failure of Case 3 ("SIÊU SALE 50%\nDUY NHẤT HÔM NAY") on 1:1 Square (1024x1024).
  Specifically evaluate:
    1. Dead Margin Void Elimination (Tight-Crop Y from 448px down to 288px):
       - Baseline 1152x448 has 211px of dead black margin (nearly 50% of the box is empty void).
       - Tight-crop Y (1152x288) eliminates 900 dead tokens and packs 100% active text tokens.
    2. Vertical Diacritic Breathing Room (Line Spacing Ratio):
       - Baseline 0.22 (only 22px gap, or 1.3 latent cells between '%' and the stacked accent on 'Ấ').
       - Generous 0.40 (41px gap, ~2.6 latent cells, preventing semantic bleed between lines).
    3. Dimension Expansion (1280x320):
       - Expands font size from 104pt to 115pt while keeping vertical envelope tight.
    4. Line Layout Rebalancing (3 lines):
       - "SIÊU SALE 50%\nDUY NHẤT\nHÔM NAY" (font size jumps to 119pt).

EXECUTION:
  - 100% FLUX.2-klein-4B Distilled (8 steps, guidance=1.5, t=10.0).
  - Evaluated primarily on 1:1 Square (1024x1024) + cross-checked on 9:16 Portrait (576x1024).
  - Automatically packages all results into `diagnose_case3_results.zip`.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
import zipfile
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
from flux2.sampling import batched_prc_txt, denoise, get_schedule, prc_img
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import load_ae, load_flow_model
from tendoo.glyph_engine import GlyphEngine, GlyphInfo, resolve_font_path

CASE3_PROMPT = (
    "Poster flash sale thương mại điện tử bùng nổ, các hộp quà tặng màu đỏ và dải ruy băng vàng "
    "bay lơ lửng xung quanh, dòng chữ tiêu đề lớn 3D mạ vàng kim loại đúc khối rực rỡ ở phía trên, "
    "ánh sáng studio tương phản cao phong cách lễ hội mua sắm, không có watermark"
)
CASE3_SEED = 777

DIAGNOSTIC_VARIANTS = [
    {
        "id": "var1_baseline_1152x448",
        "desc": "Baseline 1152x448 | Spacing 0.22 (22px) | Dead padding 211px",
        "text": "SIÊU SALE 50%\nDUY NHẤT HÔM NAY",
        "box_w": 1152,
        "box_h": 448,
        "spacing_ratio": 0.22,
        "tight_crop_y": False,
    },
    {
        "id": "var2_tightcrop_1152x288",
        "desc": "Tight-Crop Y 1152x288 | Spacing 0.22 | 0px dead void",
        "text": "SIÊU SALE 50%\nDUY NHẤT HÔM NAY",
        "box_w": 1152,
        "box_h": 448,
        "spacing_ratio": 0.22,
        "tight_crop_y": True,
    },
    {
        "id": "var3_spacing040_1152x448",
        "desc": "Generous Spacing 0.40 (41px gap) | 1152x448",
        "text": "SIÊU SALE 50%\nDUY NHẤT HÔM NAY",
        "box_w": 1152,
        "box_h": 448,
        "spacing_ratio": 0.40,
        "tight_crop_y": False,
    },
    {
        "id": "var4_tightcrop_spacing040",
        "desc": "Tight-Crop Y (1152x288) + Generous Spacing 0.40 (41px gap)",
        "text": "SIÊU SALE 50%\nDUY NHẤT HÔM NAY",
        "box_w": 1152,
        "box_h": 448,
        "spacing_ratio": 0.40,
        "tight_crop_y": True,
    },
    {
        "id": "var5_wider_1280x320_spacing040",
        "desc": "Wider Envelope 1280x320 (115pt) + Spacing 0.40",
        "text": "SIÊU SALE 50%\nDUY NHẤT HÔM NAY",
        "box_w": 1280,
        "box_h": 320,
        "spacing_ratio": 0.40,
        "tight_crop_y": False,
    },
    {
        "id": "var6_3lines_rebalanced",
        "desc": "3-line Rebalance (119pt) | 1152x448 | Spacing 0.35",
        "text": "SIÊU SALE 50%\nDUY NHẤT\nHÔM NAY",
        "box_w": 1152,
        "box_h": 448,
        "spacing_ratio": 0.35,
        "tight_crop_y": False,
    },
]


def render_diagnostic_glyph(
    text: str,
    font_name: str,
    box_w: int,
    box_h: int,
    spacing_ratio: float,
    tight_crop_y: bool,
    padding_px: int = 16,
) -> Tuple[Image.Image, int, int, int, int]:
    """Renders glyph with precise control over line spacing and adaptive tight cropping."""
    _, font_path, _ = resolve_font_path(font_name)
    ge = GlyphEngine()
    lines = text.split("\n")

    # Binary search largest font that fits width and height
    low, high, best_size, best_font = 8, 200, 8, None
    while low <= high:
        mid = (low + high) // 2
        f = ge.get_font(font_path, mid)
        lws = [f.getbbox(l)[2] - f.getbbox(l)[0] for l in lines]
        lhs = [f.getbbox(l)[3] - f.getbbox(l)[1] for l in lines]
        tot_w = max(lws)
        tot_h = sum(lhs) + int(mid * spacing_ratio) * (len(lines) - 1)
        if tot_w <= box_w - 2 * padding_px and tot_h <= box_h - 2 * padding_px:
            best_size, best_font = mid, f
            low = mid + 1
        else:
            high = mid - 1

    lws = [best_font.getbbox(l)[2] - best_font.getbbox(l)[0] for l in lines]
    lhs = [best_font.getbbox(l)[3] - best_font.getbbox(l)[1] for l in lines]
    line_spacing = int(best_size * spacing_ratio)
    tot_h = sum(lhs) + line_spacing * (len(lines) - 1)

    img = Image.new("RGB", (box_w, box_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = (box_h - tot_h) // 2
    for idx, l in enumerate(lines):
        bbox = best_font.getbbox(l)
        lw = bbox[2] - bbox[0]
        dx = (box_w - lw) // 2 - bbox[0]
        dy = curr_y - bbox[1]
        draw.text((dx, dy), l, fill=(255, 255, 255), font=best_font)
        curr_y += lhs[idx] + line_spacing

    final_w, final_h = box_w, box_h
    if tight_crop_y:
        arr = np.array(img.convert("L"))
        non_empty = np.where(arr > 0)
        min_y, max_y = non_empty[0].min(), non_empty[0].max()
        ch = max_y - min_y + 1
        h_aligned = int(math.ceil((ch + 2 * padding_px) / 16.0) * 16)
        # Crop Y strictly, keep width aligned
        cropped = img.crop((0, min_y, box_w, max_y + 1))
        new_img = Image.new("RGB", (box_w, h_aligned), color=(0, 0, 0))
        paste_y = (h_aligned - ch) // 2
        new_img.paste(cropped, (0, paste_y))
        img = new_img
        final_h = h_aligned

    return img, best_size, final_w, final_h, line_spacing


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
    """Searches for flux-2-klein-4b.safetensors across standard paths."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Deep diagnostic on Case 3: Tight-Crop Y, Line Spacing, and Layout Rebalancing on 1:1 Square"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
        help="Directory containing VAE and Text Encoder",
    )
    parser.add_argument(
        "--distill_model_path",
        type=str,
        default=None,
        help="Explicit path to flux-2-klein-4b.safetensors",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Distill ODE steps (default: 8)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.5,
        help="Guidance scale (default: 1.5)",
    )
    parser.add_argument(
        "--canvases",
        type=str,
        nargs="+",
        default=["square_1x1", "portrait_9x16"],
        help="Canvases to test ('square_1x1' [1024x1024], 'portrait_9x16' [576x1024])",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_diagnose_case3",
        help="Directory to save generated outputs and ZIP archive",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Device Allocation
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            device_dit = torch.device("cuda:0")
            device_ae = torch.device("cuda:1")
            print(f"[HW] Dual-GPU Setup: DiT on {device_dit} | VAE & TextEncoder on {device_ae}")
        else:
            device_dit = torch.device("cuda:0")
            device_ae = torch.device("cuda:0")
            print(f"[HW] Single-GPU Setup: All components on {device_dit}")
    else:
        print("[ERROR] CUDA GPU is required!")
        sys.exit(1)

    # 2. Checkpoint Resolution
    base_dir = Path(args.checkpoint_dir)
    distill_file = Path(args.distill_model_path) if args.distill_model_path else find_distill_checkpoint(base_dir)
    if not distill_file or not distill_file.exists():
        print(f"[ERROR] Distilled DiT checkpoint not found around: {base_dir}")
        print("  Please specify with --distill_model_path <path_to_flux-2-klein-4b.safetensors>")
        sys.exit(1)

    print("\n" + "=" * 105)
    print("🔬 TENDOO AI - CASE 3 TIGHT-CROP & SPACING DIAGNOSTIC SUITE")
    print("=" * 105)
    print(f"  Target Model       : {distill_file}")
    print(f"  Target Text        : 'SIÊU SALE 50% / DUY NHẤT HÔM NAY'")
    print(f"  ODE Steps          : {args.steps} (Guidance: {args.guidance})")
    print(f"  Canvases           : {args.canvases}")
    print(f"  Variants to test   : {len(DIAGNOSTIC_VARIANTS)}")
    print(f"  Output Directory   : {out_path.resolve()}")

    # 3. Load Models
    print("\n[1/3] Loading Distilled DiT (4B)...")
    os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_file)
    model = load_flow_model(model_name="flux.2-klein-4b", device=device_dit)
    model.eval()

    print("[2/3] Loading VAE and Text Encoder (Qwen3-4B-FP8)...")
    os.environ["FLUX_CHECKPOINT_DIR"] = str(base_dir)
    ae = load_ae(model_name="flux.2-klein-base-4b", device=device_ae)
    ae.eval()
    text_encoder = load_qwen3_embedder(variant="4B", device=device_ae)

    # Text prompt encoding (done once)
    with torch.no_grad():
        txt_prompt = text_encoder([CASE3_PROMPT]).to(device=device_dit, dtype=torch.bfloat16)
    txt_tokens, txt_ids = batched_prc_txt(txt_prompt)

    # Resolve Canvas Dimensions
    canvas_dict = {
        "square_1x1": (1024, 1024),
        "portrait_9x16": (576, 1024),
    }

    selected_canvases = [(k, canvas_dict[k][0], canvas_dict[k][1]) for k in args.canvases if k in canvas_dict]
    if not selected_canvases:
        print("[ERROR] No valid canvas specified!")
        sys.exit(1)

    total_runs = len(DIAGNOSTIC_VARIANTS) * len(selected_canvases)
    print(f"\n[3/3] Executing {total_runs} Diagnostic Runs...")

    results_table: List[Dict[str, Any]] = []
    generated_files: List[Path] = []
    current_run = 0

    for v in DIAGNOSTIC_VARIANTS:
        vid = v["id"]
        vdesc = v["desc"]
        vtext = v["text"]
        v_box_w = v["box_w"]
        v_box_h = v["box_h"]
        v_spacing = v["spacing_ratio"]
        v_tight = v["tight_crop_y"]

        # Render diagnostic glyph
        glyph_img, font_size, actual_w, actual_h, actual_gap = render_diagnostic_glyph(
            text=vtext,
            font_name="bevietnam",
            box_w=v_box_w,
            box_h=v_box_h,
            spacing_ratio=v_spacing,
            tight_crop_y=v_tight,
        )
        glyph_tokens = (actual_w // 16) * (actual_h // 16)

        glyph_name = f"glyph__{vid}.png"
        glyph_path = out_path / glyph_name
        glyph_img.save(glyph_path)

        # Encode glyph to ref tokens at t=10.0
        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae, glyph_img, t_offset=10.0, device=device_ae)
        ref_tokens = ref_tokens.to(device_dit)
        ref_ids = ref_ids.to(device_dit)

        print(f"\n▶ [{vid}] {vdesc}")
        print(f"   Glyph: {actual_w}x{actual_h}px ({font_size}pt, gap: {actual_gap}px, {glyph_tokens} tokens)")

        for c_label, c_w, c_h in selected_canvases:
            current_run += 1
            lat_w, lat_h = c_w // 16, c_h // 16
            tag = f"{vid}__{c_label}"

            print(f"   [{current_run:02d}/{total_runs:02d}] Canvas {c_w}x{c_h} ({c_label}) ... ", end="", flush=True)

            # Gaussian Noise from fixed seed
            torch.manual_seed(CASE3_SEED)
            z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
            img_tokens, img_ids = prc_img(z_init[0])
            img_tokens = img_tokens.unsqueeze(0).to(device_dit)
            img_ids = img_ids.unsqueeze(0).to(device_dit)

            timesteps = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])

            t0 = time.time()
            with torch.no_grad():
                out_tokens = denoise(
                    model=model,
                    img=img_tokens,
                    img_ids=img_ids,
                    txt=txt_tokens,
                    txt_ids=txt_ids,
                    timesteps=timesteps,
                    guidance=args.guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
                lat_2d = rearrange(out_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
                img_out = ae.decode(lat_2d.to(device_ae))
            dur = time.time() - t0

            pil_img = Image.fromarray(
                ((img_out[0].float().clamp(-1, 1) + 1.0) * 127.5)
                .permute(1, 2, 0)
                .byte()
                .cpu()
                .numpy()
            )

            out_file = out_path / f"{tag}.png"
            pil_img.save(out_file)
            generated_files.append(out_file)
            print(f"Done in {dur:.2f}s -> {out_file.name}")

            results_table.append({
                "variant": vid,
                "box": f"{actual_w}x{actual_h}",
                "font_pt": font_size,
                "gap_px": actual_gap,
                "tokens": glyph_tokens,
                "canvas": f"{c_w}x{c_h}",
                "ratio": c_label,
                "time_s": round(dur, 2),
                "file": out_file.name,
            })

    # 4. Packaging & Summary
    zip_filename = out_path / "diagnose_case3_results.zip"
    print(f"\n[Packaging] Creating ZIP: {zip_filename.name}...")
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in generated_files:
            zf.write(f, arcname=f.name)
        for g in out_path.glob("glyph__*.png"):
            zf.write(g, arcname=g.name)

    print("\n" + "=" * 115)
    print("📊 CASE 3 DIAGNOSTIC SUMMARY TABLE")
    print("=" * 115)
    print(f"{'Variant ID':<26} | {'Glyph Box':<11} | {'Font':<7} | {'Gap':<6} | {'Tokens':<7} | {'Canvas':<13} | {'Time (s)':<9} | {'File'}")
    print("-" * 115)
    for r in results_table:
        print(
            f"{r['variant']:<26} | {r['box']:<11} | {r['font_pt']}pt{' ':<3} | {r['gap_px']}px{' ':<2} | "
            f"{r['tokens']:<7} | {r['canvas']:<13} | {r['time_s']}s{' ':<5} | {r['file']}"
        )
    print("=" * 115)
    print(f"\n🎉 DIAGNOSTIC RUNS COMPLETE! Download archive directly from:")
    print(f"   👉 {zip_filename.resolve()}\n")


if __name__ == "__main__":
    main()
