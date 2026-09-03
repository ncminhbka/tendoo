#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - GENEROUS-BOX PHILOSOPHY VALIDATION ON THE REAL 9:16 CANVAS
====================================================================================================
Script: scripts/probe_glyph_generous_box_9x16.py
Purpose:
    Consolidates everything found so far into one decisive test on the project's REAL primary
    target format (576x1024, 9:16) -- which the poem/896x512 success (Tây Tiến AND "Sóng", both
    4-line poems) has NOT yet been tested on (only verified on a 1024x1024 canvas so far).

    The emerging design philosophy (superseding Rule 29's "fix font at a low floor, tight-crop,
    enforce a self-aspect-ratio band"): size the box GENEROUSLY according to actual content
    (not minimized), then binary-search the LARGEST font that fits -- exactly the original
    demo_tendoo_poster.py algorithm's philosophy, now cross-checked against
    probe_glyph_absolute_scale.py's clean result (a 4-word phrase at 83pt/836 tokens hit 5/5,
    vs the SAME phrase at 61pt/448 tokens hitting only 2/5 -- font size, not aspect ratio or raw
    token count, is what moved the needle).

    Three configs, all on the real 576x1024 canvas, 5 seeds each:
      poem_9x16       : the exact Tây Tiến box (896x512px, NOT tight-cropped) -- glyph is 1.56x
                        WIDER than the whole canvas (896 vs 576). Untested on 9:16 until now.
      short_generous  : "MUA 1 TẶNG 1" in a generously-sized (not tiny) 384x256 box -> 86pt.
      long_generous   : the 14-word sentence, pre-wrapped to 4 reasonably-wide lines in a
                        512x512 box (not the old crude 2-tier auto-formula, which gave an
                        unreasonably small font for this length) -> 38pt.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 ONLY.

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_generous_box_9x16.py                # all 3 configs x 5 seeds = 15 runs
====================================================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from flux2.util import load_ae, load_flow_model, load_qwen3_embedder
from tendoo.glyph_engine import GlyphInfo, resolve_font_path


# ==================================================================================================
# 1. CONFIGS: pre-picked lines + generous (NOT tight-cropped) box, font auto-maximized within it
# ==================================================================================================

FONT = "bevietnam"
CANVAS = (576, 1024)  # real 9:16 primary target
DEFAULT_SEEDS = [42, 123, 777, 2024, 8888]

CONFIGS = {
    "poem_9x16": {
        "lines": [
            "Sông Mã xa rồi Tây Tiến ơi",
            "Nhớ về rừng núi nhớ chơi vơi.",
            "Sài Khao sương lấp đoàn quân mỏi,",
            "Mường Lát hoa về trong đêm hơi.",
        ],
        "box": (896, 512),  # the exact historical Tây Tiến box, untested on 9:16 until now
    },
    "short_generous": {
        "lines": ["MUA 1", "TẶNG 1"],
        "box": (384, 256),
    },
    "long_generous": {
        "lines": ["Đậm đà hương vị", "cà phê Việt Nam", "pha phin truyền thống", "mỗi sáng"],
        "box": (512, 512),
    },
}

DEFAULT_PROMPT = (
    "Poster quảng cáo phong cách hiện đại tối giản, ánh sáng studio ấm áp tương phản cao, "
    "dòng chữ 3D mạ vàng đồng cổ sắc nét nổi bật, bố cục sạch sẽ gọn gàng, "
    "không có chữ ký, không có watermark, không có chữ trang trí thừa"
)


def render_generous_box_glyph(lines: List[str], font_name: str, box_w: int, box_h: int, spacing_ratio: float = 0.18) -> GlyphInfo:
    """
    The original demo_tendoo_poster.py philosophy: a GENEROUS, pre-chosen (not tight-cropped,
    not minimized) box; binary-search the LARGEST font that fits both dimensions. Lines are
    passed in pre-wrapped (not re-split here) since the point is to test specific, deliberately-
    sized box/line combinations, not to auto-derive them from raw text.
    """
    _, font_path, meta = resolve_font_path(font_name)
    box_w, box_h = (box_w // 16) * 16, (box_h // 16) * 16
    pad_w, pad_h = int(box_w * 0.08), int(box_h * 0.08)
    max_w, max_h = box_w - 2 * pad_w, box_h - 2 * pad_h

    low, high, best_size, best_font = 14, 300, 0, None
    while low <= high:
        mid = (low + high) // 2
        try:
            test_font = ImageFont.truetype(font_path, size=mid)
        except Exception:
            test_font = ImageFont.load_default()
        total_h, max_line_w = 0, 0
        for line in lines:
            bbox = test_font.getbbox(line)
            max_line_w = max(max_line_w, bbox[2] - bbox[0])
            total_h += bbox[3] - bbox[1]
        total_h += int(mid * spacing_ratio) * (len(lines) - 1)
        if max_line_w <= max_w and total_h <= max_h:
            best_size, best_font = mid, test_font
            low = mid + 1
        else:
            high = mid - 1

    if best_font is None:
        best_font = ImageFont.truetype(font_path, size=20)
        best_size = 20

    line_spacing = int(best_size * spacing_ratio)
    line_widths = [best_font.getbbox(l)[2] - best_font.getbbox(l)[0] for l in lines]
    line_heights = [best_font.getbbox(l)[3] - best_font.getbbox(l)[1] for l in lines]
    total_block_h = sum(line_heights) + line_spacing * (len(lines) - 1)

    img = Image.new("RGB", (box_w, box_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    curr_y = (box_h - total_block_h) // 2
    for i, line in enumerate(lines):
        lw = line_widths[i]
        bbox = best_font.getbbox(line)
        draw.text(((box_w - lw) // 2 - bbox[0], curr_y - bbox[1]), line, fill=(255, 255, 255), font=best_font)
        curr_y += line_heights[i] + line_spacing

    text_joined = " ".join(lines)
    return GlyphInfo(
        image=img, text=text_joined, lines=lines, font_name=font_name, font_path=font_path,
        font_size_pt=best_size, width_px=box_w, height_px=box_h,
        latent_w=box_w // 16, latent_h=box_h // 16, token_count=(box_w // 16) * (box_h // 16),
        archetype=meta["archetype"], tier=meta["tier"], min_floor_pt=meta["min_floor_pt"],
        is_nyquist_safe=best_size >= meta["min_floor_pt"],
        line_spacing_px=line_spacing, padding_x_px=pad_w, padding_y_px=pad_h,
    )


# ==================================================================================================
# 2. 4D ROPE ENCODING (isolated single-glyph @ t=10.0)
# ==================================================================================================

def encode_glyph_to_ref_tokens(
    ae: AutoEncoder, glyph_img: Image.Image, t_offset: float, device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# ==================================================================================================
# 3. MAIN RUNNER
# ==================================================================================================

@dataclass
class RunConfig:
    config_key: str
    seed: int
    run_id: str


def build_run_matrix(config_keys: List[str], seeds: List[int]) -> List[RunConfig]:
    return [RunConfig(config_key=k, seed=s, run_id=f"{k}_seed{s}") for k in config_keys for s in seeds]


def run_probe(
    config_keys: List[str], seeds: List[int], prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_generous_box_9x16", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(config_keys, seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - GENEROUS-BOX PHILOSOPHY VALIDATION ON THE REAL 9:16 CANVAS")
    print("=" * 100)
    print(f"  Configs  : {config_keys}")
    print(f"  Seeds    : {seeds}")
    print(f"  Canvas   : {CANVAS[0]}x{CANVAS[1]} (real 9:16 primary target)")
    print(f"  Total runs: {len(all_runs)}")

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print("[*] Dual GPU Mode: DiT on GPU 0 | VAE & Qwen3 on GPU 1")
    else:
        device_dit = device_ae = device_te = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[*] Single Device Mode: {device_dit}")

    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    print("\n[1/3] Loading FLUX.2 Klein 4B Base (AE + DiT + Qwen3)...")
    ae = load_ae(model_name, device=device_ae)
    model = load_flow_model(model_name, device=device_dit)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    print("[2/3] Encoding shared prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt, txt_ids = batched_prc_txt(text_encoder(["", prompt]))
        txt, txt_ids = txt.to(device_dit), txt_ids.to(device_dit)

    if num_gpus >= 2:
        try:
            text_encoder.model.to("cpu")
        except Exception:
            pass
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    canvas_w, canvas_h = CANVAS
    canvas_w, canvas_h = (canvas_w // 16) * 16, (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    glyph_infos: Dict[str, GlyphInfo] = {}
    for key in config_keys:
        cfg = CONFIGS[key]
        box_w, box_h = cfg["box"]
        info = render_generous_box_glyph(cfg["lines"], FONT, box_w, box_h)
        glyph_infos[key] = info
        aspect = info.width_px / info.height_px
        canvas_ratio = info.latent_w / lat_w
        print(f"  [{key}] {info.width_px}x{info.height_px}px {info.font_size_pt}pt {len(info.lines)}L "
              f"{info.token_count}tok aspect={aspect:.2f} canvas_width_ratio={canvas_ratio:.2f} :: {info.lines}")
        info.image.save(out_path / f"{key}_glyph.png")

    print(f"\n[3/3] Executing {len(all_runs)} denoise run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        glyph_info = glyph_infos[run.config_key]

        print("-" * 100)
        print(f"▶️  [{idx}/{len(all_runs)}] {run.run_id}")
        t_run_start = time.time()

        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae=ae, glyph_img=glyph_info.image, t_offset=10.0, device=device_ae)
        ref_tokens, ref_ids = ref_tokens.to(device_dit), ref_ids.to(device_dit)

        torch.manual_seed(run.seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

        with torch.no_grad():
            out_latent = denoise_cfg(
                model=model, img=img_tokens, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                timesteps=timesteps, guidance=guidance, img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )

        torch.cuda.empty_cache()
        out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_latent = out_latent.to(device=device_ae, dtype=torch.bfloat16)
        with torch.no_grad():
            out_tensor = ae.decode(out_latent)

        out_tensor = torch.clamp((out_tensor[0] + 1.0) / 2.0, min=0.0, max=1.0)
        out_arr = (out_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        result_img = Image.fromarray(out_arr)
        result_path = out_path / f"{run.run_id}_result.png"
        result_img.save(result_path)

        elapsed = time.time() - t_run_start
        manifest.append({
            "run_id": run.run_id, "config_key": run.config_key, "seed": run.seed,
            "glyph_px": f"{glyph_info.width_px}x{glyph_info.height_px}", "font_pt": glyph_info.font_size_pt,
            "tokens": glyph_info.token_count, "elapsed_s": round(elapsed, 2),
            "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "generous_box_9x16_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] SUMMARY (fill in Verdict, then tally X/5 per config)")
    print("=" * 100)
    print(f"{'Run ID':<24} | {'Config':<16} | {'Box':<10} | {'Pt':<4} | {'Seed':<6} | Verdict")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<24} | {r['config_key']:<16} | {r['glyph_px']:<10} | {r['font_pt']:<4} | {r['seed']:<6} | ? PASS/FAIL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP]")
    print("  - poem_9x16 (896px glyph on 576px canvas, ratio 1.56) succeeds like it did on 1024x1024")
    print("    -> canvas orientation doesn't matter for the 'generous box + max font' philosophy;")
    print("    confidently rewrite compute_optimal_glyph_box around it.")
    print("  - poem_9x16 fails on 9:16 despite working on 1:1 -> canvas shape/orientation IS still a")
    print("    real factor for very wide-relative-to-canvas glyphs, independent of font size --")
    print("    the new design needs a canvas-aware width cap after all, just a much looser one")
    print("    than Rule 29's old max_line_width_ratio=0.4.")
    print("  - short_generous (86pt) and long_generous (38pt) results confirm whether font size")
    print("    should scale down gracefully with text length (both still beating the old 40pt?).\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Generous-Box Philosophy Validation (9:16)")
    parser.add_argument("--configs", type=str, nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", type=str, default="output_glyph_generous_box_9x16")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)

    args = parser.parse_args()
    run_probe(
        config_keys=args.configs, seeds=args.seeds, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
