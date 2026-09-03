#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - OLD VS NEW GLYPH SIZING RECIPE, HEAD-TO-HEAD (REGRESSION CHECK)
====================================================================================================
Script: scripts/probe_old_vs_new_glyph_recipe.py
Purpose:
    User reports "BỨT PHÁ MỌI GIỚI HẠN" (the subtitle text used across the Direction 1/2 probes)
    fails even in TRUE isolation (isolated_subtitle condition, probe_regional_parallel_diffusion.py)
    under the NEW glyph_engine.py (Rule 29: tight-crop, self-aspect-ratio band [0.5,1.3], 40pt
    floor) -- yet text like this NEVER failed under the OLDER scripts (demo_tendoo_poster.py /
    test_tiktok_poster.py) before this investigation's glyph_engine rewrite.

    Reverse-engineering test_tiktok_poster.py's create_glyph_image for this EXACT text
    (bevietnam) reveals THREE variables differ simultaneously from the new engine's choice:
        OLD: font=61pt, lines=["BỨT PHÁ","MỌI GIỚI HẠN"], box=512x224 FIXED (aspect 2.29,
             NOT tight-cropped -- binary-searches the largest font that fills this fixed envelope)
        NEW: font=40pt, lines=["BỨT PHÁ MỌI","GIỚI HẠN"], box=320x256 tight-cropped
             (aspect 1.25, inside the "safe" band)

    This is a real risk that Rounds 1-7's aspect-ratio-band conclusion (built almost entirely on
    repeated use of 2-3 "Đậm đà hương vị..." family texts for cross-round comparability) may be
    overfit to that specific text family rather than a general law -- and/or that 40pt is simply
    not enough font size for full reliability regardless of aspect ratio.

    This probe isolates the SIZING RECIPE as the only variable: renders "BỨT PHÁ MỌI GIỚI HẠN"
    (bevietnam) via BOTH recipes, runs BOTH through the exact same isolated single-glyph @ t=10
    pipeline (no multi-slot, no merging -- removes every other confound from the last two
    probes), multi-seed, to settle head-to-head whether the OLD recipe still works today and
    whether it's font size, aspect ratio, or the specific line split that matters.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 ONLY.

Usage on Remote Server (2x A30):
    python scripts/probe_old_vs_new_glyph_recipe.py                # both recipes x 5 seeds = 10 runs
====================================================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
from tendoo.glyph_engine import GlyphInfo, render_glyph, resolve_font_path


# ==================================================================================================
# 1. TEXT + THE OLD RECIPE (reproduced from test_tiktok_poster.py / demo_tendoo_poster.py's
#    create_glyph_image: fixed envelope, binary-search largest font, NOT tight-cropped)
# ==================================================================================================

TEXT = "BỨT PHÁ MỌI GIỚI HẠN"
FONT = "bevietnam"
CANVAS = (576, 1024)  # real 9:16 primary target
DEFAULT_SEEDS = [42, 123, 777, 2024, 8888]

DEFAULT_PROMPT = (
    "Poster tuyển dụng phong cách công ty công nghệ hiện đại, nền gradient xanh dương đậm sang "
    "trọng, dòng chữ phát sáng neon tinh tế, bố cục sạch sẽ chuyên nghiệp, không có chữ ký, "
    "không có watermark"
)


def render_old_recipe_glyph(text: str, font_name: str, envelope_w: int = 512, envelope_h: int = 224) -> GlyphInfo:
    """
    Reproduces test_tiktok_poster.py / demo_tendoo_poster.py's create_glyph_image EXACTLY:
    fixed envelope (NOT derived from text length or canvas), binary-search the LARGEST font that
    fits both dimensions, multi-line boxes are NOT tight-cropped (kept at the full envelope size).
    """
    _, font_path, meta = resolve_font_path(font_name)
    envelope_w = (envelope_w // 16) * 16
    envelope_h = (envelope_h // 16) * 16
    padding_ratio = 0.08
    pad_w, pad_h = int(envelope_w * padding_ratio), int(envelope_h * padding_ratio)
    max_w, max_h = envelope_w - 2 * pad_w, envelope_h - 2 * pad_h

    words = text.split()
    candidate_layouts = []
    if len(words) >= 4:
        mid = len(words) // 2
        candidate_layouts.append([" ".join(words[:mid]), " ".join(words[mid:])])
    if len(words) >= 6:
        p1, p2 = len(words) // 3, 2 * len(words) // 3
        candidate_layouts.append([" ".join(words[:p1]), " ".join(words[p1:p2]), " ".join(words[p2:])])
    candidate_layouts.append([text])

    best_size, best_lines, best_font = 0, None, None
    for lines in candidate_layouts:
        spacing_ratio = 0.32 if len(lines) >= 2 else 0.20
        low, high, opt_size, opt_font = 14, 200, 0, None
        while low <= high:
            mid_size = (low + high) // 2
            try:
                test_font = ImageFont.truetype(font_path, size=mid_size)
            except Exception:
                test_font = ImageFont.load_default()
            total_h, max_line_w = 0, 0
            for line in lines:
                bbox = test_font.getbbox(line)
                max_line_w = max(max_line_w, bbox[2] - bbox[0])
                total_h += bbox[3] - bbox[1]
            total_h += int(mid_size * spacing_ratio) * (len(lines) - 1)
            if max_line_w <= max_w and total_h <= max_h:
                opt_size, opt_font = mid_size, test_font
                low = mid_size + 1
            else:
                high = mid_size - 1
        if opt_size > best_size:
            best_size, best_lines, best_font = opt_size, lines, opt_font

    if best_font is None:
        best_font = ImageFont.truetype(font_path, size=20)
        best_lines, best_size = candidate_layouts[-1], 20

    spacing_ratio = 0.32 if len(best_lines) >= 2 else 0.20
    line_spacing = int(best_size * spacing_ratio)
    line_widths = [best_font.getbbox(l)[2] - best_font.getbbox(l)[0] for l in best_lines]
    line_heights = [best_font.getbbox(l)[3] - best_font.getbbox(l)[1] for l in best_lines]
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)

    img = Image.new("RGB", (envelope_w, envelope_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    curr_y = (envelope_h - total_block_h) // 2
    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        bbox = best_font.getbbox(line)
        draw.text(((envelope_w - lw) // 2 - bbox[0], curr_y - bbox[1]), line, fill=(255, 255, 255), font=best_font)
        curr_y += line_heights[i] + line_spacing

    return GlyphInfo(
        image=img, text=text, lines=best_lines, font_name=font_name, font_path=font_path,
        font_size_pt=best_size, width_px=envelope_w, height_px=envelope_h,
        latent_w=envelope_w // 16, latent_h=envelope_h // 16, token_count=(envelope_w // 16) * (envelope_h // 16),
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
    recipe: str
    seed: int
    run_id: str


def build_run_matrix(recipes: List[str], seeds: List[int]) -> List[RunConfig]:
    return [RunConfig(recipe=r, seed=s, run_id=f"{r}_seed{s}") for r in recipes for s in seeds]


def run_probe(
    recipes: List[str], seeds: List[int], prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_old_vs_new_recipe", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(recipes, seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - OLD VS NEW GLYPH SIZING RECIPE, HEAD-TO-HEAD")
    print("=" * 100)
    print(f"  Text     : \"{TEXT}\"  Font: {FONT}")
    print(f"  Recipes  : {recipes}")
    print(f"  Seeds    : {seeds}")
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
    canvas_w = (canvas_w // 16) * 16
    canvas_h = (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    glyph_infos: Dict[str, GlyphInfo] = {}
    if "old" in recipes:
        glyph_infos["old"] = render_old_recipe_glyph(TEXT, FONT)
    if "new" in recipes:
        glyph_infos["new"] = render_glyph(text=TEXT, font_name_or_path=FONT, auto_size=True, target_canvas_w=canvas_w, target_canvas_h=canvas_h)

    for recipe, info in glyph_infos.items():
        aspect = info.width_px / info.height_px
        print(f"  [{recipe:5s}] {info.width_px}x{info.height_px}px {info.font_size_pt}pt {len(info.lines)}L "
              f"{info.token_count}tok aspect={aspect:.2f} :: {info.lines}")
        info.image.save(out_path / f"{recipe}_glyph.png")

    print(f"\n[3/3] Executing {len(all_runs)} denoise run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        glyph_info = glyph_infos[run.recipe]

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
            "run_id": run.run_id, "recipe": run.recipe, "seed": run.seed,
            "glyph_px": f"{glyph_info.width_px}x{glyph_info.height_px}", "font_pt": glyph_info.font_size_pt,
            "lines": glyph_info.lines, "tokens": glyph_info.token_count,
            "elapsed_s": round(elapsed, 2), "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "old_vs_new_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] SUMMARY (fill in Verdict per run, then tally X/N per recipe)")
    print("=" * 100)
    print(f"{'Run ID':<16} | {'Recipe':<6} | {'Box':<10} | {'Pt':<4} | {'Seed':<6} | Verdict")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<16} | {r['recipe']:<6} | {r['glyph_px']:<10} | {r['font_pt']:<4} | {r['seed']:<6} | ? PASS/FAIL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP]")
    print("  - OLD wins clearly, NEW fails -> Rule 29's aspect-ratio band and/or 40pt floor is")
    print("    wrong or incomplete; font size (61pt here) and/or the specific line split matter")
    print("    more than self-aspect-ratio. Re-open glyph_engine.py sizing -- likely raise the")
    print("    floor further and/or stop tight-cropping multi-line boxes (reproduce OLD behavior).")
    print("  - Both fail similarly -> something else changed (model checkpoint, seed pool, prompt)")
    print("    -- not a glyph-sizing regression specifically.")
    print("  - Both work -> Rounds 1-7 conclusions hold for this text too; the isolated_subtitle")
    print("    failure was noise (only 3 seeds tested there) -- rerun it with more seeds.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Old vs New Glyph Sizing Recipe Head-to-Head")
    parser.add_argument("--recipes", type=str, nargs="+", default=["old", "new"], choices=["old", "new"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", type=str, default="output_old_vs_new_recipe")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)

    args = parser.parse_args()
    run_probe(
        recipes=args.recipes, seeds=args.seeds, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
