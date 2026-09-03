#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - TOKEN-MASS x CANVAS-SHAPE PROBE (2x2, MULTI-SEED)
====================================================================================================
Script: scripts/probe_glyph_token_mass_vs_canvas.py
Purpose:
    A documented, verified 100%-accurate result (AGENTS.md Rule 11, exp54, the "Tây Tiến" poem via
    `scripts/demo_tendoo_poster.py --box_w 896 --box_h 512` on a 1024x1024 canvas) directly
    contradicts the width-ratio theory from Rounds 1-3: glyph_lat_w=56, canvas_lat_w=64, ratio=0.875
    -- well inside what those rounds called the "failure zone" (>=0.86). Meanwhile, the current
    tight-crop `render_glyph()` path (Rounds 1-4) never reached that same 100% crispness (92-99%
    across 3 seeds), even at a conservative ratio of 0.4.

    The likely explanation: `scripts/demo_tendoo_poster.py`'s `create_glyph_image()` has a rule that
    was NEVER carried over into `src/tendoo/glyph_engine.py`:

        if tight_crop and len(best_lines) == 1:
            # Single-line: tight crop height and width
            ...
        else:
            # Multi-line (>= 2 lines): PRESERVE ENVELOPE HEIGHT to guarantee latent tokens per line
            final_w = envelope_w
            final_h = envelope_h

    i.e. for multi-line text, the OLD code deliberately does NOT tight-crop -- it keeps the full,
    generously-sized user-specified envelope (more absolute tokens, more padding) and binary-searches
    the LARGEST font that fits inside it. `glyph_engine.py`'s `compute_optimal_glyph_box` always
    tight-crops toward minimal tokens (Rule 25), regardless of line count -- optimizing for the
    OPPOSITE thing.

    This is a 2x2 factorial to isolate which factor (or both) explains the accuracy gap:
        Canvas shape : 1024x1024 (1:1, the proven format) vs 576x1024 (9:16, project's real target)
        Box strategy : TIGHT   (current shipped render_glyph(), minimal tokens, Rule 29 ratio=0.4)
                       GENEROUS (mimics demo_tendoo_poster.py: envelope width = 85% of canvas width,
                                 height = num_lines * 160px, binary-search font to fill it -- NOT
                                 tight-cropped)
    Each of the 4 cells run on 2 texts (medium, long -- the ones that showed <100% in Round 4) x 2
    seeds = 16 runs total.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 ONLY.

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_token_mass_vs_canvas.py                 # full 2x2, 16 runs
    python scripts/probe_glyph_token_mass_vs_canvas.py --seeds 42      # 1 seed only, 8 runs (faster)
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
from PIL import Image

from flux2.autoencoder import AutoEncoder
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.util import load_ae, load_flow_model, load_qwen3_embedder
from tendoo.glyph_engine import GlyphInfo, auto_wrap_text, render_glyph, resolve_font_path


# ==================================================================================================
# 1. TEXTS, CANVASES, BOX STRATEGIES
# ==================================================================================================

TEXTS = {
    "medium": "Đậm đà hương vị cà phê Việt",
    "long": "Đậm đà hương vị cà phê Việt Nam pha phin truyền thống mỗi sáng",
}
CANVASES = {"1x1": (1024, 1024), "9x16": (576, 1024)}
DEFAULT_SEEDS = [42, 123]

GENEROUS_WIDTH_FRAC = 0.85   # matches Tây Tiến's 896/1024 ~= 0.875, rounded down slightly for margin
GENEROUS_LINE_HEIGHT_PX = 160  # matches AGENTS.md rule 4's documented >=160px floor (not 112/128)
TIGHT_RATIO = 0.4              # current shipped max_line_width_ratio default


def render_generous_glyph(
    text: str, font_name: str, canvas_w: int, canvas_h: int,
    width_frac: float = GENEROUS_WIDTH_FRAC, line_height_px: int = GENEROUS_LINE_HEIGHT_PX,
    padding_px: int = 16,
) -> GlyphInfo:
    """
    Mimics scripts/demo_tendoo_poster.py's multi-line branch: a deliberately generous, NON-tight-crop
    envelope (large fixed fraction of canvas width, height scaled by line count at a taller-than-
    current-default per-line floor), with the largest font that fits binary-searched inside it --
    the opposite optimization direction from compute_optimal_glyph_box's minimal-token tight crop.
    """
    _, font_path, meta = resolve_font_path(font_name)
    floor_pt = meta["min_floor_pt"]

    box_w = max(32, round(canvas_w * width_frac / 16) * 16)
    max_line_w = box_w - 2 * padding_px

    # First pass at the font floor size to determine how many lines this generous width needs.
    lines_probe = auto_wrap_text(text=text, font_path=font_path, font_size_pt=floor_pt, max_line_width_px=max_line_w)
    n_lines = len(lines_probe)
    box_h = max(32, round(n_lines * line_height_px / 16) * 16)

    # Mode B (explicit envelope): binary-searches the LARGEST font filling (box_w, box_h),
    # re-wrapping internally against the same max_line_w -- consistent with lines_probe above.
    return render_glyph(text=text, font_name_or_path=font_name, target_width=box_w, target_height=box_h, auto_size=False)


def render_tight_glyph(text: str, font_name: str, canvas_w: int, canvas_h: int) -> GlyphInfo:
    """Current shipped production path: minimal tokens, Rule 29 canvas-aware wrapping at ratio=0.4."""
    return render_glyph(
        text=text, font_name_or_path=font_name, auto_size=True,
        target_canvas_w=canvas_w, target_canvas_h=canvas_h, max_line_width_ratio=TIGHT_RATIO,
    )


@dataclass
class RunConfig:
    text_key: str
    canvas_key: str
    strategy: str
    seed: int
    run_id: str


def build_run_matrix(seeds: List[int]) -> List[RunConfig]:
    runs = []
    for text_key in TEXTS:
        for canvas_key in CANVASES:
            for strategy in ["tight", "generous"]:
                for seed in seeds:
                    runs.append(RunConfig(
                        text_key=text_key, canvas_key=canvas_key, strategy=strategy, seed=seed,
                        run_id=f"{text_key}_{canvas_key}_{strategy}_seed{seed}",
                    ))
    return runs


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
    seeds: List[int], font: str = "bevietnam", prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_token_mass_vs_canvas", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - TOKEN-MASS x CANVAS-SHAPE PROBE (2x2 FACTORIAL, MULTI-SEED)")
    print("=" * 100)
    print(f"  Texts    : {list(TEXTS.keys())}")
    print(f"  Canvases : {CANVASES}")
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

    # Render each unique (text, canvas, strategy) glyph ONCE -- seed only affects diffusion sampling.
    glyph_cache: Dict[Tuple[str, str, str], GlyphInfo] = {}
    for text_key, text in TEXTS.items():
        for canvas_key, (cw, ch) in CANVASES.items():
            for strategy in ["tight", "generous"]:
                key = (text_key, canvas_key, strategy)
                if strategy == "tight":
                    info = render_tight_glyph(text, font, cw, ch)
                else:
                    info = render_generous_glyph(text, font, cw, ch)
                glyph_cache[key] = info
                cw16, lat_w = (cw // 16) * 16, (cw // 16)
                ratio = info.latent_w / lat_w
                print(f"  [{text_key:7s} | {canvas_key:5s} | {strategy:9s}] {info.width_px}x{info.height_px}px, "
                      f"{info.font_size_pt}pt, {len(info.lines)}L, {info.token_count} tokens, ratio={ratio:.3f} "
                      f":: {info.lines}")
                info.image.save(out_path / f"{text_key}_{canvas_key}_{strategy}_glyph.png")

    print(f"\n[3/3] Executing {len(all_runs)} denoise run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        glyph_info = glyph_cache[(run.text_key, run.canvas_key, run.strategy)]
        canvas_w, canvas_h = CANVASES[run.canvas_key]
        canvas_w = (canvas_w // 16) * 16
        canvas_h = (canvas_h // 16) * 16
        lat_w, lat_h = canvas_w // 16, canvas_h // 16

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
        ratio = glyph_info.latent_w / lat_w
        manifest.append({
            "run_id": run.run_id, "text_key": run.text_key, "canvas_key": run.canvas_key,
            "strategy": run.strategy, "seed": run.seed, "glyph_lines": glyph_info.lines,
            "glyph_px": f"{glyph_info.width_px}x{glyph_info.height_px}", "font_pt": glyph_info.font_size_pt,
            "tokens": glyph_info.token_count, "ratio": round(ratio, 3),
            "elapsed_s": round(elapsed, 2), "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "token_mass_vs_canvas_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] 2x2 SUMMARY (fill in Verdict, then compare tight vs generous per canvas per text)")
    print("=" * 100)
    print(f"{'Run ID':<34} | {'Tokens':<7} | {'Ratio':<6} | {'Verdict'}")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<34} | {r['tokens']:<7} | {r['ratio']:<6.3f} | ? PASS/FAIL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP] For each (text, canvas) pair, compare tight vs generous reliability:")
    print("  - generous >> tight on BOTH canvases -> token-mass/non-tight-crop is the dominant fix,")
    print("    independent of canvas shape -> port demo_tendoo_poster.py's 'preserve envelope for")
    print("    multi-line' rule into glyph_engine.py.")
    print("  - generous helps mainly on 1x1, not on 9x16 -> canvas SHAPE (not just token mass) is an")
    print("    independent constraint on the 9:16 target format -- may need a different mitigation")
    print("    for the project's actual primary format.")
    print("  - tight already close to generous on both -> token-mass theory overstated, look elsewhere.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Token-Mass x Canvas-Shape Probe (2x2)")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to replicate (default: 42 123)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt")
    parser.add_argument("--output_dir", type=str, default="output_glyph_token_mass_vs_canvas", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")

    args = parser.parse_args()
    run_probe(
        seeds=args.seeds, font=args.font, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
