#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - LEAN DiT FONT RESOLUTION FLOOR PROBE (3 CRITICAL POINTS)
====================================================================================================
Script: scripts/probe_dit_font_resolution_floor.py
Purpose:
    Fast, focused probe testing ONLY 3 critical boundary points: [24pt, 32pt, 40pt]
    to immediately identify the empirical resolution threshold of FLUX.2 Klein 4B Base DiT
    in under 2 minutes (3 runs total).

Default Configuration:
    - Font   : BeVietnamPro-Black (Standard Workhorse Sans)
    - Sizes  : 24pt (Small), 32pt (Boundary), 40pt (Safe) -> Total: 3 runs!
    - Canvas : 576 x 1024 (Standard 9:16 poster, low VRAM, fast denoise)
    - ODE    : 50 steps, CFG = 4.0

Usage on Remote Server (2x A30):
    python scripts/probe_dit_font_resolution_floor.py
    # Optional: Test Playfair instead
    python scripts/probe_dit_font_resolution_floor.py --font playfair
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
from typing import Any, Dict, List, Optional, Tuple

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
from src.tendoo.glyph_engine import FONT_REGISTRY, FONT_TIERS, resolve_font_path


def render_single_line_glyph(
    text: str,
    font_path: str,
    font_size_pt: int,
    padding_px: int = 16,
) -> Tuple[Image.Image, int, int, int]:
    """Renders a single-line glyph bitmap at exact font size snapped to 16px."""
    try:
        font = ImageFont.truetype(font_path, size=font_size_pt)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        font = ImageFont.load_default()

    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    raw_w = tw + 2 * padding_px
    raw_h = th + 2 * padding_px

    # Snap dimensions to 16px multiples
    snapped_w = int(np.ceil(raw_w / 16.0) * 16)
    snapped_h = int(np.ceil(max(raw_h, 80) / 16.0) * 16)

    img = Image.new("RGB", (snapped_w, snapped_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    x = (snapped_w - tw) // 2 - bbox[0]
    y = (snapped_h - th) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

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


def run_di_isolation_sweep(
    fonts_to_test: List[str],
    sizes_to_test: List[int],
    text: str = "TÔI YÊU VIỆT NAM",
    prompt: str = (
        "Một cuốn sổ tay vintage bìa da màu nâu sẫm phẳng phiu đặt trên mặt bàn gỗ cổ, "
        "ở góc dưới bìa da có dòng chữ nhỏ dập chìm mạ vàng đồng cổ sắc nét tinh xảo, "
        "ánh sáng studio chụp cận cảnh macro tương phản cao, bề mặt da mịn màng"
    ),
    output_dir: str = "output_dit_floor",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: Optional[str] = None,
    width: int = 576,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
):
    """Executes lean DiT isolation sweep across 3 font sizes."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Device allocation
    num_gpus = torch.cuda.device_count()
    print(f"[*] Detected {num_gpus} CUDA device(s).")
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print(f"  -> Dual GPU: DiT on GPU 0, VAE/TextEncoder on GPU 1")
    else:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:0")
        device_te = torch.device("cuda:0")
        print(f"  -> Single GPU: All models on {device_dit}")

    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    print("\n[1/3] Loading AutoEncoder (VAE) and DiT Base 4B...")
    ae = load_ae(model_name, device=device_ae)
    model = load_flow_model(model_name, device=device_dit)

    print("\n[2/3] Encoding Text Prompt via Qwen3-4B-FP8...")
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    # Completely offload and purge text_encoder to guarantee 100% VRAM free on GPU 1
    try:
        text_encoder.model.to("cpu")
    except Exception:
        pass
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_ae)
        print(f"  -> Successfully purged Qwen3! Free VRAM on {device_ae}: {free_bytes / (1024**3):.2f} / {total_bytes / (1024**3):.2f} GiB")

    # Prepare Canvas Dimensions (576x1024 = 2304 tokens, 2x faster, minimal VRAM)
    width = (width // 16) * 16
    height = (height // 16) * 16
    lat_w, lat_h = width // 16, height // 16

    results_manifest = []

    total_runs = len(fonts_to_test) * len(sizes_to_test)
    print(f"\n[3/3] Running Lean DiT Sweep ({total_runs} run(s) total: sizes {sizes_to_test} pt)...")
    run_idx = 0

    for font_key in fonts_to_test:
        font_alias, font_path, meta = resolve_font_path(font_key)
        tier_name = meta["tier"]

        print(f"\n" + "=" * 80)
        print(f" [*] PROBING FONT: {font_alias.upper()} ({tier_name})")
        print("=" * 80)

        for pt in sizes_to_test:
            run_idx += 1
            print(f"\n>>> [{run_idx}/{total_runs}] Testing {font_alias.upper()} @ {pt}pt...")

            # 1. Render glyph bitmap
            glyph_img, g_w, g_h, g_tokens = render_single_line_glyph(
                text=text, font_path=font_path, font_size_pt=pt, padding_px=16
            )
            glyph_preview_path = out_path / f"glyph_{font_alias}_{pt}pt.png"
            glyph_img.save(glyph_preview_path)
            print(f"  -> Glyph: {g_w}x{g_h}px ({g_tokens} tokens) saved: {glyph_preview_path.name}")

            # 2. Encode to In-Context Ref Tokens at t=10.0
            ref_tokens, ref_ids = encode_glyph_to_ref_tokens(
                ae=ae, glyph_img=glyph_img, t_offset=10.0, device=device_ae
            )
            ref_tokens = ref_tokens.to(device_dit)
            ref_ids = ref_ids.to(device_dit)

            # 3. Canvas Initial Noise
            torch.manual_seed(seed)
            z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
            img_tokens, img_ids = prc_img(z_init[0])
            img_tokens = img_tokens.unsqueeze(0).to(device_dit)
            img_ids = img_ids.unsqueeze(0).to(device_dit)

            # 4. Schedule and Denoise 50 steps
            timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

            t0 = time.time()
            with torch.no_grad():
                out_latent = denoise_cfg(
                    model=model,
                    img=img_tokens,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    timesteps=timesteps,
                    guidance=guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
            denoise_time = time.time() - t0

            # 5. Decode to Pixel Space
            torch.cuda.empty_cache()
            out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
            out_latent = out_latent.to(device=device_ae, dtype=torch.bfloat16)
            with torch.no_grad():
                out_tensor = ae.decode(out_latent)

            out_tensor = torch.clamp((out_tensor[0] + 1.0) / 2.0, min=0.0, max=1.0)
            out_arr = (out_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
            res_img = Image.fromarray(out_arr)

            out_file = out_path / f"dit_denoise_{font_alias}_{pt}pt.png"
            res_img.save(out_file)
            print(f"  -> Generated Image in {denoise_time:.2f}s saved: {out_file.name}")

            # Clean memory
            del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
            gc.collect()
            torch.cuda.empty_cache()

            results_manifest.append({
                "font": font_alias,
                "tier": tier_name,
                "size_pt": pt,
                "glyph_px": f"{g_w}x{g_h}",
                "tokens": g_tokens,
                "image_file": out_file.name,
                "glyph_file": glyph_preview_path.name,
                "denoise_time": round(denoise_time, 2),
            })

    # Save manifest and HTML report
    manifest_path = out_path / "sweep_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results_manifest, f, ensure_ascii=False, indent=2)

    html_path = out_path / "comparison_report.html"
    generate_html_report(results_manifest, html_path, text)
    print("\n" + "=" * 80)
    print(f"[OK] Lean Sweep Completed! Manifest: {manifest_path}")
    print(f"[*] Visual HTML Report: {html_path.resolve()}")
    print("=" * 80 + "\n")


def generate_html_report(manifest: List[dict], html_path: Path, text: str):
    """Generates a clean HTML visual grid for side-by-side inspection in JupyterLab."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Tendoo AI - Lean DiT Resolution Floor Probe</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #121212; color: #eee; padding: 24px; }}
        h1 {{ color: #4CAF50; margin-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #333; padding: 12px; text-align: center; vertical-align: middle; }}
        th {{ background: #1e1e1e; color: #bbb; font-weight: 600; }}
        tr:nth-child(even) {{ background: #181818; }}
        img {{ max-width: 240px; height: auto; border-radius: 6px; border: 1px solid #444; transition: transform 0.2s; }}
        img:hover {{ transform: scale(1.05); }}
        .badge {{ padding: 6px 12px; border-radius: 4px; font-weight: bold; background: #2e7d32; color: #fff; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>🔬 Tendoo AI - Lean DiT Font Resolution Floor Probe</h1>
    <p><b>Target Text:</b> "{text}" | <b>Canvas:</b> 576x1024 | <b>ODE Steps:</b> 50 | <b>CFG:</b> 4.0</p>
    <p><i>Click any image to inspect full-resolution. Identify where diacritics transition from jagged to silk-smooth:</i></p>
    <table>
        <tr>
            <th>Font</th>
            <th>Tier</th>
            <th>Size (pt)</th>
            <th>Glyph Input</th>
            <th>Tokens</th>
            <th>DiT 50-step Output Image</th>
            <th>Denoise Time</th>
        </tr>
    """
    for r in manifest:
        html += f"""
        <tr>
            <td><b>{r['font'].upper()}</b></td>
            <td>{r['tier']}</td>
            <td><span class="badge">{r['size_pt']}pt</span></td>
            <td><img src="{r['glyph_file']}" alt="Glyph"></td>
            <td>{r['tokens']}</td>
            <td><a href="{r['image_file']}" target="_blank"><img src="{r['image_file']}" alt="DiT Output"></a></td>
            <td>{r['denoise_time']}s</td>
        </tr>
        """
    html += """
    </table>
</body>
</html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Lean DiT Font Resolution Floor Probe")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font to test (default: bevietnam)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[24, 32, 40], help="Critical test sizes in pt (default: 24 32 40)")
    parser.add_argument("--text", type=str, default="TÔI YÊU VIỆT NAM", help="Vietnamese test phrase")
    parser.add_argument("--output_dir", type=str, default="output_dit_floor", help="Output directory")
    parser.add_argument("--width", type=int, default=576, help="Canvas width (default: 576)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height (default: 1024)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to FLUX.2 checkpoints")
    parser.add_argument("--steps", type=int, default=50, help="ODE denoise steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_di_isolation_sweep(
        fonts_to_test=[args.font],
        sizes_to_test=args.sizes,
        text=args.text,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
