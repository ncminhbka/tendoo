#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - GLYPH-TO-CANVAS LATENT WIDTH RATIO PROBE (RULE 29 THRESHOLD SWEEP)
====================================================================================================
Script: scripts/probe_glyph_width_ratio.py
Purpose:
    Directly and cleanly measure the failure threshold for `glyph_latent_width /
    canvas_latent_width` implicated by scripts/probe_glyph_engine_lock.py's round-1 results:

        ratio <= 0.59  -> every observed case rendered cleanly
        ratio >= 0.86  -> every observed case failed (garbled/broken text)

    Round 1 confounded width-ratio with text content, line count, and canvas height at the
    same time (e.g. the poem A/B varied canvas orientation; the C_minh sweep varied height at
    a FIXED, already-high 0.86 ratio; the long-1-line case failed on BOTH canvases because its
    ratio was >1.0 on both). This probe isolates width-ratio as the ONLY variable:

    - TWO glyph bitmaps are rendered ONCE each (fixed pixel content, fixed font, fixed size):
        Glyph A = "Đậm đà hương vị cà phê Việt" (diacritic-dense, 6 words)   -- lat_w ~31
        Glyph B = "MUA 1 TẶNG 1"                (diacritic-light, 3 tokens) -- lat_w ~17
      (Glyph B already rendered cleanly at ratio 0.47 in probe_glyph_engine_lock.py Section A/B
      -- included here again as a sanity anchor plus to extend its curve toward higher ratios.)
    - Each fixed glyph is then composited (isolated single-glyph @ t=10.0) onto a SERIES of
      canvases whose width is chosen to hit specific target ratios, holding canvas height
      constant at 1024px throughout (Round 1 showed HEIGHT ratio does not predict failure, so
      it is deliberately not controlled for -- only WIDTH ratio is swept).
    - Comparing Glyph A's curve vs Glyph B's curve at matched ratios tests whether diacritic
      density is an independent confound or whether width-ratio alone predicts the outcome.

    NOTE: the canvases used here are SYNTHETIC diagnostic shapes chosen to hit exact ratios --
    they do NOT represent real poster formats. Do not use this script's canvas dimensions for
    production; use scripts/probe_glyph_engine_lock.py / real formats for that.

Strict Rule Adherence (AGENTS.md Rule 28):
    - ZERO HTML output. Clean ASCII table in Terminal + JSON manifest + PNG images only.
    - Hardware Target: 2x NVIDIA A30. Model: FLUX.2-klein-base-4B, isolated single-glyph @
      t=10.0 ONLY (the one config already proven 100% reliable in isolation).

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_width_ratio.py                  # both glyphs, ~11 runs
    python scripts/probe_glyph_width_ratio.py --glyphs A        # only the diacritic-dense glyph
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
from tendoo.glyph_engine import GlyphInfo, render_glyph


# ==================================================================================================
# 1. FIXED GLYPH DEFINITIONS + RATIO-DERIVED CANVAS WIDTHS
# ==================================================================================================

GLYPH_DEFS = {
    "A": dict(text="Đậm đà hương vị cà phê Việt", label="A_heavy_diacritic",
              ratios=[1.5, 1.2, 1.0, 0.7, 0.55, 0.4]),
    "B": dict(text="MUA 1 TẶNG 1", label="B_light_diacritic",
              ratios=[1.3, 1.0, 0.86, 0.7, 0.55]),
}
CANVAS_HEIGHT_PX = 1024  # held constant for every synthetic row; Round 1 showed height ratio
                          # does not predict failure, so it is deliberately not swept here.


def canvas_width_for_ratio(glyph_lat_w: int, ratio: float) -> int:
    """Returns the canvas pixel width (snapped to 16) that gives glyph_lat_w/canvas_lat_w == ratio."""
    canvas_lat_w = max(16, round(glyph_lat_w / ratio))
    return canvas_lat_w * 16


@dataclass
class RunConfig:
    glyph_key: str
    run_id: str
    target_ratio: float
    canvas_wh: Tuple[int, int]


def build_run_matrix(glyph_keys: List[str]) -> Dict[str, List[RunConfig]]:
    matrix: Dict[str, List[RunConfig]] = {}
    for key in glyph_keys:
        gdef = GLYPH_DEFS[key]
        matrix[key] = []
    return matrix  # filled in run_probe once glyph_lat_w is known from the actual render


# ==================================================================================================
# 2. 4D ROPE ENCODING (isolated single-glyph @ t=10.0)
# ==================================================================================================

def encode_glyph_to_ref_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: str | torch.device = "cuda",
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
# 3. MAIN SWEEP RUNNER
# ==================================================================================================

DEFAULT_PROMPT = (
    "Poster quảng cáo cà phê phong cách hiện đại tối giản, ánh sáng studio ấm áp tương phản cao, "
    "dòng chữ 3D mạ vàng đồng cổ sắc nét nổi bật, bố cục sạch sẽ gọn gàng, "
    "không có chữ ký, không có watermark, không có chữ trang trí thừa"
)


def run_probe(
    glyph_keys: List[str],
    font: str = "bevietnam",
    prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_width_ratio",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(" [*] TENDOO AI - GLYPH-TO-CANVAS WIDTH RATIO PROBE (RULE 29 THRESHOLD SWEEP)")
    print("=" * 100)
    print(f"  Font    : {font}")
    print(f"  Glyphs  : {glyph_keys}")

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

    manifest: List[Dict[str, Any]] = []

    for glyph_key in glyph_keys:
        gdef = GLYPH_DEFS[glyph_key]
        text, label, ratios = gdef["text"], gdef["label"], gdef["ratios"]

        print(f"\n{'-' * 100}\n[GLYPH {glyph_key}] \"{text}\" ({label})\n{'-' * 100}")
        glyph_info = render_glyph(text=text, font_name_or_path=font, force_single_line=True, auto_size=True)
        glyph_path = out_path / f"{label}_glyph.png"
        glyph_info.image.save(glyph_path)
        print(f"  Rendered ONCE: {glyph_info.width_px}x{glyph_info.height_px}px, {glyph_info.font_size_pt}pt, "
              f"lat_w={glyph_info.latent_w}, {glyph_info.token_count} tokens -> {glyph_info.lines}")

        ref_tokens_base, ref_ids_base = encode_glyph_to_ref_tokens(
            ae=ae, glyph_img=glyph_info.image, t_offset=10.0, device=device_ae
        )

        runs = [
            RunConfig(glyph_key, f"{label}_r{r:.2f}".replace(".", "p"), r,
                      (canvas_width_for_ratio(glyph_info.latent_w, r), CANVAS_HEIGHT_PX))
            for r in ratios
        ]

        for idx, run in enumerate(runs, 1):
            canvas_w, canvas_h = run.canvas_wh
            canvas_w = (canvas_w // 16) * 16
            canvas_h = (canvas_h // 16) * 16
            lat_w, lat_h = canvas_w // 16, canvas_h // 16
            actual_ratio = glyph_info.latent_w / lat_w

            print(f"\n▶️  [{idx}/{len(runs)}] {run.run_id}: target_ratio={run.target_ratio:.2f} "
                  f"-> canvas={canvas_w}x{canvas_h} (actual_ratio={actual_ratio:.3f})")
            t_run_start = time.time()

            ref_tokens = ref_tokens_base.to(device_dit)
            ref_ids = ref_ids_base.to(device_dit)

            torch.manual_seed(seed)
            z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
            img_tokens, img_ids = prc_img(z_init[0])
            img_tokens = img_tokens.unsqueeze(0).to(device_dit)
            img_ids = img_ids.unsqueeze(0).to(device_dit)

            timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

            with torch.no_grad():
                out_latent = denoise_cfg(
                    model=model, img=img_tokens, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                    timesteps=timesteps, guidance=guidance,
                    img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
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
                "glyph_key": glyph_key,
                "glyph_label": label,
                "text": text,
                "run_id": run.run_id,
                "target_ratio": run.target_ratio,
                "actual_ratio": round(actual_ratio, 3),
                "glyph_lat_w": glyph_info.latent_w,
                "canvas": f"{canvas_w}x{canvas_h}",
                "canvas_lat_w": lat_w,
                "elapsed_s": round(elapsed, 2),
                "result_file": result_path.name,
                "verdict": "TBD",  # fill in manually after visual inspection
            })
            print(f"    -> {result_path.name} ({elapsed:.1f}s)")

            del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
            gc.collect()
            torch.cuda.empty_cache()

    manifest_path = out_path / "width_ratio_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] WIDTH RATIO SWEEP SUMMARY (fill in 'Verdict' after opening each PNG)")
    print("=" * 100)
    print(f"{'Run ID':<28} | {'Glyph':<18} | {'TargetR':<8} | {'ActualR':<8} | {'Canvas':<12} | {'Verdict':<10}")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<28} | {r['glyph_label']:<18} | {r['target_ratio']:<8.2f} | "
              f"{r['actual_ratio']:<8.3f} | {r['canvas']:<12} | {'? PASS/FAIL'}")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP] For each glyph (A and B), find the LOWEST target_ratio that still shows")
    print("            broken/garbled text, and the HIGHEST target_ratio that still renders clean.")
    print("            That brackets the true safe ceiling for max_line_width_ratio in glyph_engine.py.")
    print("            Compare A vs B at matched ratios: if both fail/pass together, diacritic")
    print("            density is NOT an independent confound -- width ratio alone predicts it.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Glyph-to-Canvas Width Ratio Probe")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--glyphs", type=str, nargs="+", default=["A", "B"], choices=["A", "B"],
                         help="Which fixed glyph(s) to sweep (default: both)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt")
    parser.add_argument("--output_dir", type=str, default="output_glyph_width_ratio", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    run_probe(
        glyph_keys=args.glyphs, font=args.font, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance, seed=args.seed,
    )


if __name__ == "__main__":
    main()
