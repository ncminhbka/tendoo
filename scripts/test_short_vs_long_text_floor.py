#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - EMPIRICAL COMPARISON: SHORT TEXT VS. LONG TEXT AT 32PT
====================================================================================================
Script: scripts/test_short_vs_long_text_floor.py
Purpose:
    Rigorously tests whether text length (1 line / 4 words vs. 4 lines / 28 words / 119 chars)
    affects DiT resolution stability at 32pt.

Runs exactly 3 targeted samples:
    [1/3] Short Text (1 line / 4 words)       @ 32pt: "TÔI YÊU VIỆT NAM"
    [2/3] Long Text  (4 lines / 28 words)     @ 32pt: Tây Tiến (4 câu thơ)
    [3/3] Long Text  (4 lines / 28 words)     @ 40pt: Tây Tiến (4 câu thơ - Control Benchmark)

Strict Rule Adherence:
    - Rule 28: ZERO HTML output. Clean Terminal table + direct PNG images.
    - VRAM safe: Text encoder offloaded to CPU immediately, gc.collect() called.
    - Canvas: 576 x 1024 (Runs in ~1.5 mins total on 2x A30).
====================================================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

from flux2.autoencoder import AutoEncoder
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.util import (
    load_ae,
    load_flow_model,
    load_qwen3_embedder,
)
from src.tendoo.glyph_engine import resolve_font_path


def render_multiline_glyph(
    lines: List[str],
    font_path: str,
    font_size_pt: int,
    padding_px: int = 16,
    line_spacing_ratio: float = 0.22,
) -> Tuple[Image.Image, int, int, int]:
    """Renders multi-line glyph bitmap at exact font size snapped to 16px."""
    try:
        font = ImageFont.truetype(font_path, size=font_size_pt)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        font = ImageFont.load_default()

    line_w_list = []
    line_h_list = []
    for l in lines:
        bbox = font.getbbox(l)
        line_w_list.append(bbox[2] - bbox[0])
        line_h_list.append(bbox[3] - bbox[1])

    max_w = max(line_w_list) if line_w_list else 100
    line_spacing = int(font_size_pt * line_spacing_ratio)
    total_h = sum(line_h_list) + line_spacing * (len(lines) - 1)

    raw_w = max_w + 2 * padding_px
    raw_h = total_h + 2 * padding_px

    snapped_w = int(np.ceil(raw_w / 16.0) * 16)
    snapped_h = int(np.ceil(max(raw_h, 80) / 16.0) * 16)

    img = Image.new("RGB", (snapped_w, snapped_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = (snapped_h - total_h) // 2
    for idx, l in enumerate(lines):
        bbox = font.getbbox(l)
        lw = bbox[2] - bbox[0]
        x = (snapped_w - lw) // 2 - bbox[0]
        y = curr_y - bbox[1]
        draw.text((x, y), l, font=font, fill=(255, 255, 255))
        curr_y += line_h_list[idx] + line_spacing

    tokens = (snapped_w // 16) * (snapped_h // 16)
    return img, snapped_w, snapped_h, tokens


def encode_glyph_to_ref_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes glyph image into 4D RoPE In-Context tokens."""
    arr = np.array(glyph_img).astype(np.float32) / 127.5 - 1.0
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


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI - Short vs Long Text at 32pt Floor Benchmark")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--output_dir", type=str, default="output_short_vs_long", help="Output directory")
    parser.add_argument("--width", type=int, default=576, help="Canvas width (default: 576)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height (default: 1024)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to FLUX.2 checkpoints")
    parser.add_argument("--steps", type=int, default=50, help="ODE denoise steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 3 Target Test Cases
    cases = [
        {
            "id": "1_short_1line_32pt",
            "name": "Short Text (1 dòng / 4 từ)",
            "lines": ["TÔI YÊU VIỆT NAM"],
            "pt": 32,
            "prompt": (
                "Một tấm bảng gỗ sồi phẳng phiu sang trọng đặt ở trung tâm, "
                "trên mặt gỗ có dòng chữ ngắn dập chìm mạ vàng sắc nét tinh xảo, "
                "ánh sáng studio tương phản cao, bề mặt gỗ bóng nhẹ"
            ),
        },
        {
            "id": "2_long_4lines_32pt",
            "name": "Long Text (4 câu thơ / 28 từ)",
            "lines": [
                "Sông Mã xa rồi Tây Tiến ơi",
                "Nhớ về rừng núi nhớ chơi vơi.",
                "Sài Khao sương lấp đoàn quân mỏi,",
                "Mường Lát hoa về trong đêm hơi.",
            ],
            "pt": 32,
            "prompt": (
                "Một tấm bảng gỗ sồi phẳng phiu sang trọng đặt ở trung tâm, "
                "trên mặt gỗ có bốn câu thơ chữ dập chìm mạ vàng sắc nét tinh xảo, "
                "ánh sáng studio tương phản cao, bề mặt gỗ bóng nhẹ"
            ),
        },
        {
            "id": "3_long_4lines_40pt",
            "name": "Long Text Control (4 câu thơ / 28 từ)",
            "lines": [
                "Sông Mã xa rồi Tây Tiến ơi",
                "Nhớ về rừng núi nhớ chơi vơi.",
                "Sài Khao sương lấp đoàn quân mỏi,",
                "Mường Lát hoa về trong đêm hơi.",
            ],
            "pt": 40,
            "prompt": (
                "Một tấm bảng gỗ sồi phẳng phiu sang trọng đặt ở trung tâm, "
                "trên mặt gỗ có bốn câu thơ chữ dập chìm mạ vàng sắc nét tinh xảo, "
                "ánh sáng studio tương phản cao, bề mặt gỗ bóng nhẹ"
            ),
        },
    ]

    # Device allocation
    num_gpus = torch.cuda.device_count()
    print(f"[*] Detected {num_gpus} CUDA device(s).")
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print(f"  -> Dual GPU: DiT on GPU 0, VAE on GPU 1")
    else:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:0")
        device_te = torch.device("cuda:0")
        print(f"  -> Single GPU: {device_dit}")

    if args.checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = args.checkpoint_dir

    print("\n[1/3] Loading AutoEncoder and DiT Base 4B...")
    ae = load_ae("flux.2-klein-base-4b", device=device_ae)
    model = load_flow_model("flux.2-klein-base-4b", device=device_dit)

    font_alias, font_path, meta = resolve_font_path(args.font)

    # Prepare Canvas Dimensions
    width = (args.width // 16) * 16
    height = (args.height // 16) * 16
    lat_w, lat_h = width // 16, height // 16

    results_manifest = []

    print(f"\n[2/3] Starting Comparative Probe ({len(cases)} runs total)...")

    for idx, c in enumerate(cases, 1):
        print(f"\n" + "=" * 80)
        print(f" >>> [{idx}/{len(cases)}] TESTING: {c['name']} @ {c['pt']}pt ({font_alias.upper()})")
        print("=" * 80)

        # 1. Render Glyph
        glyph_img, g_w, g_h, g_tokens = render_multiline_glyph(
            lines=c["lines"],
            font_path=font_path,
            font_size_pt=c["pt"],
            padding_px=16,
            line_spacing_ratio=meta["default_line_spacing"],
        )
        glyph_file = out_path / f"glyph_{c['id']}.png"
        glyph_img.save(glyph_file)
        print(f"  -> Glyph Bitmap: {g_w}x{g_h}px ({g_tokens} tokens) saved: {glyph_file.name}")

        # 2. Encode Text Prompt
        print(f"  -> Encoding Text Prompt...")
        text_encoder = load_qwen3_embedder(variant="4B", device=device_te)
        with torch.no_grad():
            txt = text_encoder(["", c["prompt"]])
            txt, txt_ids = batched_prc_txt(txt)
            txt = txt.to(device_dit)
            txt_ids = txt_ids.to(device_dit)

        # Offload text encoder immediately to preserve 100% free VRAM
        try:
            text_encoder.model.to("cpu")
        except Exception:
            pass
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Encode In-Context Reference Tokens at t=10.0
        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(
            ae=ae, glyph_img=glyph_img, t_offset=10.0, device=device_ae
        )
        ref_tokens = ref_tokens.to(device_dit)
        ref_ids = ref_ids.to(device_dit)

        # 4. Canvas Initial Noise
        torch.manual_seed(args.seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        # 5. Denoise 50 steps Euler ODE
        timesteps = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])
        t0 = time.time()
        with torch.no_grad():
            out_latent = denoise_cfg(
                model=model,
                img=img_tokens,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=args.guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
        denoise_time = time.time() - t0

        # 6. Decode Image
        torch.cuda.empty_cache()
        out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_latent = out_latent.to(device=device_ae, dtype=torch.bfloat16)
        with torch.no_grad():
            out_tensor = ae.decode(out_latent)

        out_tensor = torch.clamp((out_tensor[0] + 1.0) / 2.0, min=0.0, max=1.0)
        out_arr = (out_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        res_img = Image.fromarray(out_arr)

        out_file = out_path / f"output_{c['id']}.png"
        res_img.save(out_file)
        print(f"  -> Generated Image in {denoise_time:.2f}s saved: {out_file.name}")

        # Cleanup memory
        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids, txt, txt_ids
        gc.collect()
        torch.cuda.empty_cache()

        results_manifest.append({
            "id": c["id"],
            "name": c["name"],
            "lines": len(c["lines"]),
            "font_size_pt": c["pt"],
            "glyph_size": f"{g_w}x{g_h}",
            "tokens": g_tokens,
            "image_file": out_file.name,
            "glyph_file": glyph_file.name,
            "denoise_time_sec": round(denoise_time, 2),
        })

    # Save manifest JSON (Rule 28: Zero HTML)
    manifest_file = out_path / "results_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(results_manifest, f, ensure_ascii=False, indent=2)

    # Print Clean ASCII Summary Table
    print("\n" + "=" * 95)
    print(" [*] TENDOO AI - SHORT VS. LONG TEXT AT 32PT COMPARISON SUMMARY")
    print("=" * 95)
    header = f"{'Case ID':<22} | {'Description':<28} | {'Size':<6} | {'Tokens':<8} | {'Denoise':<8} | {'Image Output':<22}"
    print(header)
    print("-" * 95)
    for r in results_manifest:
        print(f"{r['id']:<22} | {r['name']:<28} | {r['font_size_pt']}pt{'':<2} | {r['tokens']:<8} | {r['denoise_time_sec']}s{'':<2} | {r['image_file']:<22}")
    print("=" * 95)
    print(f"\n[+] All {len(results_manifest)} generated images saved in: {out_path.resolve()}")
    print(f"[+] Detailed JSON manifest saved in: {manifest_file.resolve()}")
    print("[*] Open individual images directly in JupyterLab File Viewer to compare diacritics!\n")


if __name__ == "__main__":
    main()
