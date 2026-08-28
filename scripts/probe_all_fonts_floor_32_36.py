#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - EMPIRICAL RESOLUTION FLOOR PROBE: 32PT VS. 36PT ACROSS 15 FONTS
====================================================================================================
Script: scripts/probe_all_fonts_floor_32_36.py
Purpose:
    Empirically evaluates the denoise resolution floor for all 15 remaining fonts (excluding bevietnam)
    on FLUX.2-klein-base-4B at the two most critical candidate floors: 32pt and 36pt.
    Tests each size across 3 distinct typographic conditions:
        1. Short text    (1 line  / 4 words) : "CÀ PHÊ SỮA ĐÁ"
        2. Long text     (1 line  / 10 words): "Áp dụng từ 14/05 - 30/05 / Coffee rang mộc chuẩn vị"
        3. Multiline text(2 lines / 11 words): "Ghé ngay hôm nay!\nDeal cực hot - Số lượng có hạn!"

Strict Rule Adherence:
    - Rule 28: ZERO HTML output. Clean ASCII table in Terminal + JSON manifest + PNG images.
    - Hardware Target: 2x NVIDIA A30 (DiT on GPU 0, VAE on GPU 1).
    - VRAM Optimization: Text Encoder offloaded to CPU immediately, aggressive gc.collect().
    - Canvas: 9:16 Poster (576 x 1024 px).
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
from typing import Dict, List, Optional, Tuple

# Project root
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
from PIL import Image, ImageDraw, ImageFont

# 15 Target Fonts (All except bevietnam which is locked at 32pt)
ALL_TARGET_FONTS = [
    "anton",       # Anton-Regular.ttf (Tier A: Heavy Display)
    "gotham",      # SVN-Gotham Ultra.otf (Tier A: Ultra-Bold)
    "lolapeluza",  # SVN-Lolapeluza Black.ttf (Tier A: Ultra-Black)
    "gretoon",     # SVN-Gretoon.ttf (Tier A: Pop-Art Cartoon)
    "playfair",    # PlayfairDisplay.ttf (Tier B: Editorial Serif)
    "oswald",      # Oswald.ttf (Tier B: Condensed Gothic Sans)
    "harabaras",   # SVN-Harabaras.ttf (Tier B: Geometric Medium Sans)
    "dancing",     # DancingScript.ttf (Tier C: Cursive Script)
    "pacifico",    # Pacifico-Regular.ttf (Tier C: Retro Brush Script)
    "sedgwick",    # SedgwickAveDisplay-Regular.ttf (Tier C: Street Graffiti)
    "blowbrush",   # SVN-Blow Brush.ttf (Tier C: Marker Brush)
    "clementine",  # SVN-Clementine.ttf (Tier C: Calligraphy Script)
    "cookies",     # SVN-Cookies.ttf (Tier C: Chunky Rounded)
    "grocery",     # SVN-Grocery Rounded.ttf (Tier C: Chalkboard Rounded)
    "holidays",    # SVN-Holidays.ttf (Tier C: Festive Script)
]

TEXT_CASES = {
    "short": {
        "name": "Short (1 line / 4 words)",
        "lines": ["CÀ PHÊ SỮA ĐÁ"],
    },
    "long": {
        "name": "Long (1 line / 10 words)",
        "lines": ["Áp dụng từ 14/05 - 30/05 / Coffee rang mộc chuẩn vị"],
    },
    "multiline": {
        "name": "Multiline (2 lines / 11 words)",
        "lines": [
            "Ghé ngay hôm nay!",
            "Deal cực hot - Số lượng có hạn!",
        ],
    },
}

CONTROL_PROMPT = (
    "Một tấm bảng gỗ sồi phẳng phiu sang trọng đặt ở trung tâm poster dọc, "
    "trên mặt gỗ có dòng chữ dập chìm mạ vàng sắc nét tinh xảo, "
    "ánh sáng studio tương phản cao, bề mặt gỗ bóng nhẹ"
)


def render_probe_glyph(
    lines: List[str],
    font_path: str,
    font_size_pt: int,
    padding_px: int = 16,
    line_spacing_ratio: float = 0.22,
) -> Tuple[Image.Image, int, int, int]:
    """Renders multi-line glyph bitmap at exact font size snapped to 16px multiples, no 1024 clamp."""
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

    # Snap to integer multiples of 16 (No arbitrary 1024 clamp)
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


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI - Empirical Font Floor Probe (32pt vs 36pt)")
    parser.add_argument(
        "--fonts",
        type=str,
        default="all",
        help="Comma-separated font aliases, or 'all' for all 15 fonts (default: all)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="32,36",
        help="Comma-separated font sizes in pt to test (default: 32,36)",
    )
    parser.add_argument(
        "--types",
        type=str,
        default="short,long,multiline",
        help="Comma-separated text types to test: short, long, multiline (default: short,long,multiline)",
    )
    parser.add_argument("--output_dir", type=str, default="output_all_fonts_floor_32_36", help="Output directory")
    parser.add_argument("--width", type=int, default=576, help="Canvas width 9:16 (default: 576)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height 9:16 (default: 1024)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to FLUX.2 checkpoints")
    parser.add_argument("--steps", type=int, default=50, help="ODE denoise steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry_run", action="store_true", help="Dry run: render glyphs and print plan without GPU inference")

    args = parser.parse_args()

    # Parse fonts
    if args.fonts.lower().strip() == "all":
        selected_fonts = ALL_TARGET_FONTS
    else:
        selected_fonts = [f.strip().lower() for f in args.fonts.split(",") if f.strip()]

    # Parse sizes
    selected_sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    # Parse text types
    selected_types = [t.strip().lower() for t in args.types.split(",") if t.strip() and t.strip() in TEXT_CASES]

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from src.tendoo.glyph_engine import resolve_font_path

    # Build plan
    plan: List[Dict] = []
    for f_alias in selected_fonts:
        try:
            canonical, font_path, meta = resolve_font_path(f_alias)
        except Exception as e:
            print(f"[WARN] Skipping font '{f_alias}': {e}")
            continue

        for size_pt in selected_sizes:
            for t_key in selected_types:
                t_info = TEXT_CASES[t_key]
                case_id = f"{canonical}_{size_pt}pt_{t_key}"
                plan.append({
                    "case_id": case_id,
                    "font_alias": canonical,
                    "font_path": font_path,
                    "font_meta": meta,
                    "size_pt": size_pt,
                    "type_key": t_key,
                    "type_name": t_info["name"],
                    "lines": t_info["lines"],
                    "prompt": CONTROL_PROMPT,
                })

    print("=" * 95)
    print(f" [*] TENDOO AI - EMPIRICAL FONT RESOLUTION FLOOR PROBE (32pt vs 36pt)")
    print("=" * 95)
    print(f"  - Selected Fonts ({len(selected_fonts)}) : {', '.join(selected_fonts)}")
    print(f"  - Selected Sizes : {selected_sizes} pt")
    print(f"  - Text Types     : {selected_types}")
    print(f"  - Canvas Size    : {args.width}x{args.height} (9:16 TikTok Poster)")
    print(f"  - Total Runs     : {len(plan)} sample(s)")
    print(f"  - Output Dir     : {out_path.resolve()}")
    print("=" * 95)

    # Dry-run mode: Render all glyphs and verify fonts
    if args.dry_run:
        print("\n[DRY RUN] Rendering all glyph bitmaps to verify fonts and layout dimensions...")
        for idx, item in enumerate(plan, 1):
            g_img, gw, gh, gtok = render_probe_glyph(
                lines=item["lines"],
                font_path=item["font_path"],
                font_size_pt=item["size_pt"],
                line_spacing_ratio=item["font_meta"]["default_line_spacing"],
            )
            g_file = out_path / f"glyph_{item['case_id']}.png"
            g_img.save(g_file)
            print(f" [{idx:02d}/{len(plan)}] {item['case_id']:<32} | {gw}x{gh}px | {gtok:3d} tokens | OK")
        print("\n[OK] Dry-run completed successfully! All fonts load and render cleanly.")
        return

    # Full GPU Inference Mode
    import torch
    from einops import rearrange
    from flux2.autoencoder import AutoEncoder
    from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
    from flux2.util import load_ae, load_flow_model, load_qwen3_embedder

    num_gpus = torch.cuda.device_count()
    print(f"\n[*] Detected {num_gpus} CUDA device(s).")
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print(f"  -> Dual GPU: DiT on GPU 0, VAE/Qwen on GPU 1")
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

    # Canvas setup
    width = (args.width // 16) * 16
    height = (args.height // 16) * 16
    lat_w, lat_h = width // 16, height // 16

    results_manifest = []

    print(f"\n[2/3] Starting Probe Execution ({len(plan)} runs total)...")

    for idx, item in enumerate(plan, 1):
        print(f"\n" + "-" * 85)
        print(f" >>> [{idx}/{len(plan)}] {item['font_alias'].upper()} @ {item['size_pt']}pt | {item['type_name']}")
        print("-" * 85)

        # 1. Render Glyph
        glyph_img, g_w, g_h, g_tokens = render_probe_glyph(
            lines=item["lines"],
            font_path=item["font_path"],
            font_size_pt=item["size_pt"],
            padding_px=16,
            line_spacing_ratio=item["font_meta"]["default_line_spacing"],
        )
        glyph_file = out_path / f"glyph_{item['case_id']}.png"
        glyph_img.save(glyph_file)

        # 2. Text Encoder Prompt
        text_encoder = load_qwen3_embedder(variant="4B", device=device_te)
        with torch.no_grad():
            txt = text_encoder(["", item["prompt"]])
            txt, txt_ids = batched_prc_txt(txt)
            txt = txt.to(device_dit)
            txt_ids = txt_ids.to(device_dit)

        try:
            text_encoder.model.to("cpu")
        except Exception:
            pass
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Encode Reference Glyph at t=10.0
        arr = np.array(glyph_img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device_ae, dtype=torch.bfloat16)
        with torch.no_grad():
            latent = ae.encode(tensor)

        ref_tokens, _ = prc_img(latent[0])
        ref_tokens = ref_tokens.unsqueeze(0).to(device_dit)

        gh_l, gw_l = latent.shape[2], latent.shape[3]
        t_coords = torch.full((gh_l, gw_l), fill_value=10.0, dtype=torch.float32, device=device_ae)
        h_coords = torch.arange(gh_l, dtype=torch.float32, device=device_ae).unsqueeze(1).expand(gh_l, gw_l)
        w_coords = torch.arange(gw_l, dtype=torch.float32, device=device_ae).unsqueeze(0).expand(gh_l, gw_l)
        l_coords = torch.zeros((gh_l, gw_l), dtype=torch.float32, device=device_ae)
        ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
        ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0).to(device_dit)

        # 4. Canvas Init
        torch.manual_seed(args.seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        # 5. Denoise Euler ODE
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
        out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w).to(device=device_ae, dtype=torch.bfloat16)
        with torch.no_grad():
            out_tensor = ae.decode(out_latent)

        out_tensor = torch.clamp((out_tensor[0] + 1.0) / 2.0, min=0.0, max=1.0)
        out_arr = (out_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        res_img = Image.fromarray(out_arr)

        out_file = out_path / f"output_{item['case_id']}.png"
        res_img.save(out_file)
        print(f"  -> Generated: {out_file.name} ({g_w}x{g_h}px, {g_tokens} tokens, {denoise_time:.1f}s)")

        # Cleanup memory
        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids, txt, txt_ids, tensor, latent
        gc.collect()
        torch.cuda.empty_cache()

        results_manifest.append({
            "case_id": item["case_id"],
            "font": item["font_alias"],
            "font_size_pt": item["size_pt"],
            "text_type": item["type_key"],
            "glyph_size": f"{g_w}x{g_h}",
            "tokens": g_tokens,
            "image_file": out_file.name,
            "glyph_file": glyph_file.name,
            "denoise_sec": round(denoise_time, 2),
        })

    # Save JSON manifest
    manifest_file = out_path / "results_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(results_manifest, f, ensure_ascii=False, indent=2)

    # Print ASCII Summary Table
    print("\n" + "=" * 105)
    print(" [*] TENDOO AI - EMPIRICAL FONT RESOLUTION FLOOR PROBE SUMMARY")
    print("=" * 105)
    print(f"{'Case ID':<30} | {'Font':<12} | {'Size':<6} | {'Type':<10} | {'Tokens':<8} | {'Denoise':<8} | {'Image Output':<24}")
    print("-" * 105)
    for r in results_manifest:
        print(f"{r['case_id']:<30} | {r['font']:<12} | {r['font_size_pt']}pt{'':<2} | {r['text_type']:<10} | {r['tokens']:<8} | {r['denoise_sec']}s{'':<2} | {r['image_file']:<24}")
    print("=" * 105)
    print(f"\n[+] Total runs completed: {len(results_manifest)}")
    print(f"[+] Output directory    : {out_path.resolve()}")
    print(f"[+] Manifest JSON saved : {manifest_file.resolve()}\n")


if __name__ == "__main__":
    main()
