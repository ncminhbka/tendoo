#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - FONT-SIZE FINE-DETAIL PROBE (DIACRITIC FIDELITY, ASPECT RATIO HELD FIXED)
====================================================================================================
Script: scripts/probe_glyph_font_size_fine_detail.py
Purpose:
    probe_glyph_final_lock_validation.py confirmed the Rule 29 self-aspect-ratio fix resolves the
    BIG failure mode (broken layout / crosstalk-like garbling) -- basic Latin letters render
    correctly across the board on both canvases. What remains is a smaller, more uniform residual:
    fine details (diacritic marks, the horizontal stroke on "đ") come out mildly wrong/damaged,
    seemingly independent of aspect ratio or canvas.

    This matches a DIFFERENT, earlier-flagged axis that was deprioritized in favor of chasing
    aspect ratio: probe_glyph_engine_lock.py's Section A (single-line "MUA 1 TẶNG 1", floor-bypass
    sweep) already showed the CURRENTLY LOCKED floor (32pt for bevietnam) scored WORSE ("hơi thừa
    nét nhỏ") than 28pt ("rất tốt") in that one sample -- i.e. the font-size floor itself may not
    be optimal, independent of anything line-count/aspect-ratio related.

    This probe isolates font size as the ONLY variable: fixes the "long" text (richest in Vietnamese
    diacritics: â/ă/ơ/ư/ê + multiple "đ") at a CONSTANT 3-line layout (the same line count Rule 29
    already picks for it at the aspect-ratio-good default), and sweeps font size across
    [24, 26, 28, 30, 32, 34] pt -- BYPASSING the per-font floor auto-elevation (compute_optimal_
    glyph_box unconditionally bumps anything below the locked floor back up to it, so the floor
    itself cannot be probed through the normal API) -- each replicated over 3 seeds.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 ONLY.

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_font_size_fine_detail.py                # 6 sizes x 3 seeds = 18 runs
    python scripts/probe_glyph_font_size_fine_detail.py --sizes 28 32  # fewer sizes if GPU time is tight
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
from tendoo.glyph_engine import GlyphInfo, resolve_font_path


# ==================================================================================================
# 1. FIXED TEXT/LAYOUT + FLOOR-BYPASSING RAW RENDERER
# ==================================================================================================

TEXT = "Đậm đà hương vị cà phê Việt Nam pha phin truyền thống mỗi sáng"
N_LINES = 3  # matches what Rule 29's aspect-ratio search already picks for this text at 32pt
FONT_SIZES = [24, 26, 28, 30, 32, 34]
DEFAULT_SEEDS = [42, 123, 777]
CANVAS = (576, 1024)  # real 9:16 primary target

# Reference point (Sept 2026): reverse-engineering scripts/demo_tendoo_poster.py's binary search
# on the exact "Tây Tiến" recipe (playfair, box_w=896, box_h=512, 4 lines) shows it actually chose
# 48pt -- much higher than anything tested here so far. At N_LINES=3, pushing bevietnam past ~40pt
# drives the box's self-aspect-ratio OUT of the [0.5, 1.3] safe band (width grows with font size,
# height is pinned by line count); at N_LINES=4 for this text, 48pt stays comfortably in-band
# (aspect ~0.88). Use `--n_lines 4 --sizes 36 40 44 48 52` to directly test reproducing Tây Tiến's
# recipe (bigger font enabled by more lines, not by leaving the aspect-ratio band).


def _balanced_split(words: List[str], n_lines: int) -> List[str]:
    n_lines = max(1, min(n_lines, len(words)))
    base = len(words) // n_lines
    rem = len(words) % n_lines
    lines, idx = [], 0
    for i in range(n_lines):
        take = base + (1 if i < rem else 0)
        if take > 0:
            lines.append(" ".join(words[idx: idx + take]))
            idx += take
    return lines


def render_raw_multiline_glyph_ignoring_floor(
    text: str, font_name: str, n_lines: int, font_size_pt: int,
    padding_px: int = 16, min_line_height_px: int = 128,
) -> GlyphInfo:
    """
    Renders a fixed-line-count glyph at an EXACT font size, deliberately bypassing
    compute_optimal_glyph_box's floor auto-elevation -- needed because this probe's whole point is
    to test sizes BELOW the currently-locked floor (32pt bevietnam), which the production API
    correctly (by design) refuses to do.
    """
    _, font_path, meta = resolve_font_path(font_name)
    spacing_ratio = meta["default_line_spacing"]
    try:
        font = ImageFont.truetype(font_path, size=font_size_pt)
    except Exception:
        font = ImageFont.load_default()

    words = text.strip().split()
    lines = _balanced_split(words, n_lines)

    line_widths = [font.getbbox(l)[2] - font.getbbox(l)[0] for l in lines]
    line_heights = [font.getbbox(l)[3] - font.getbbox(l)[1] for l in lines]
    line_spacing = int(font_size_pt * spacing_ratio)
    raw_w = max(line_widths)
    raw_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    total_w = raw_w + 2 * padding_px
    total_h = max(raw_h + 2 * padding_px, len(lines) * min_line_height_px)
    box_w = max(32, int(math.ceil(total_w / 16.0) * 16))
    box_h = max(32, int(math.ceil(total_h / 16.0) * 16))

    img = Image.new("RGB", (box_w, box_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    total_block_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    curr_y = max(padding_px, (box_h - total_block_h) // 2)
    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        lw = bbox[2] - bbox[0]
        draw_x = (box_w - lw) // 2 - bbox[0]
        draw_y = curr_y - bbox[1]
        draw.text((draw_x, draw_y), line, fill=(255, 255, 255), font=font)
        curr_y += line_heights[i] + line_spacing

    return GlyphInfo(
        image=img, text=text, lines=lines, font_name=font_name, font_path=font_path,
        font_size_pt=font_size_pt, width_px=box_w, height_px=box_h,
        latent_w=box_w // 16, latent_h=box_h // 16, token_count=(box_w // 16) * (box_h // 16),
        archetype=meta["archetype"], tier=meta["tier"], min_floor_pt=meta["min_floor_pt"],
        is_nyquist_safe=font_size_pt >= meta["min_floor_pt"],
        line_spacing_px=line_spacing, padding_x_px=padding_px, padding_y_px=padding_px,
    )


@dataclass
class RunConfig:
    font_pt: int
    seed: int
    run_id: str


def build_run_matrix(sizes: List[int], seeds: List[int]) -> List[RunConfig]:
    return [RunConfig(font_pt=pt, seed=seed, run_id=f"pt{pt}_seed{seed}") for pt in sizes for seed in seeds]


# ==================================================================================================
# 2. 4D ROPE ENCODING (isolated single-glyph @ t=10.0)
# ==================================================================================================

def encode_glyph_to_ref_tokens(
    ae: AutoEncoder, glyph_img: Image.Image, t_offset: float = 10.0, device: str | torch.device = "cuda",
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

DEFAULT_PROMPT = (
    "Poster quảng cáo cà phê phong cách hiện đại tối giản, ánh sáng studio ấm áp tương phản cao, "
    "dòng chữ 3D mạ vàng đồng cổ sắc nét nổi bật, bố cục sạch sẽ gọn gàng, "
    "không có chữ ký, không có watermark, không có chữ trang trí thừa"
)


def run_probe(
    sizes: List[int], seeds: List[int], font: str = "bevietnam", prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_font_size_fine_detail", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
    n_lines: int = N_LINES,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(sizes, seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - FONT-SIZE FINE-DETAIL PROBE (DIACRITIC FIDELITY)")
    print("=" * 100)
    print(f"  Text     : \"{TEXT}\"")
    print(f"  Lines    : {n_lines} (pass --n_lines to override; more lines keeps aspect ratio")
    print(f"             in-band at higher font sizes -- see the Tây Tiến reference note above)")
    print(f"  Sizes    : {sizes}")
    print(f"  Seeds    : {seeds}")
    print(f"  Canvas   : {CANVAS[0]}x{CANVAS[1]} (real 9:16)")
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
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

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

    # Render each unique font size ONCE (seed only affects diffusion sampling, not the glyph bitmap).
    glyph_cache: Dict[int, GlyphInfo] = {}
    for pt in sizes:
        info = render_raw_multiline_glyph_ignoring_floor(TEXT, font, n_lines, pt)
        glyph_cache[pt] = info
        aspect = info.width_px / info.height_px
        print(f"  [{pt}pt] {info.width_px}x{info.height_px}px, {info.token_count} tokens, "
              f"aspect={aspect:.2f} :: {info.lines}")
        info.image.save(out_path / f"pt{pt}_glyph.png")

    print(f"\n[3/3] Executing {len(all_runs)} denoise run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        glyph_info = glyph_cache[run.font_pt]

        print("-" * 100)
        print(f"▶️  [{idx}/{len(all_runs)}] {run.run_id}")
        t_run_start = time.time()

        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae=ae, glyph_img=glyph_info.image, t_offset=10.0, device=device_ae)
        ref_tokens = ref_tokens.to(device_dit)
        ref_ids = ref_ids.to(device_dit)

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
            "run_id": run.run_id, "font_pt": run.font_pt, "seed": run.seed,
            "tokens": glyph_info.token_count, "elapsed_s": round(elapsed, 2),
            "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "font_size_fine_detail_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] FONT-SIZE FINE-DETAIL SUMMARY (fill in Verdict, focus on diacritic/đ-stroke quality)")
    print("=" * 100)
    print(f"{'Run ID':<16} | {'Pt':<5} | {'Seed':<6} | {'Verdict (note which diacritics are wrong)'}")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<16} | {r['font_pt']:<5} | {r['seed']:<6} | ?")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP] For each font size, tally how many of the 3 seeds show clean diacritics")
    print("            (đ horizontal stroke, â/ă/ê/ô circumflex+breve, ã/ẽ/õ tilde all correct).")
    print("            If a LOWER pt (e.g. 28) scores better than the current locked floor (32),")
    print("            that confirms the floor itself -- not aspect ratio -- as the residual issue,")
    print("            and glyph_engine.py's FONT_TIERS floor should be revised.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Font-Size Fine-Detail Probe")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--sizes", type=int, nargs="+", default=FONT_SIZES, help="Font sizes (pt) to sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to replicate")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt")
    parser.add_argument("--output_dir", type=str, default="output_glyph_font_size_fine_detail", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")
    parser.add_argument(
        "--n_lines", type=int, default=N_LINES,
        help="Fixed line count for the sweep (default: 3). Use a higher value (e.g. 4) to keep "
             "the self-aspect-ratio in-band [0.5, 1.3] when sweeping font sizes above ~40pt -- "
             "see the Tây Tiến reference note above (48pt needed 4 lines to stay in-band).",
    )

    args = parser.parse_args()
    run_probe(
        sizes=args.sizes, seeds=args.seeds, font=args.font, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance, n_lines=args.n_lines,
    )


if __name__ == "__main__":
    main()
