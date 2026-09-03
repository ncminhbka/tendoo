#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - GLYPH ENGINE LOCK VALIDATION (MULTI-SEED, PRODUCTION CODE PATH)
====================================================================================================
Script: scripts/probe_glyph_lock_validation.py
Purpose:
    Rounds 1-3 (probe_glyph_engine_lock.py / probe_glyph_width_ratio.py / probe_glyph_aspect_ratio.py)
    each used n=1 sample (seed=42) per data point on SYNTHETIC diagnostic canvases -- enough to find
    the ballpark of a failure zone, NOT enough to certify a "locked" default with confidence: a
    canvas-ratio fine sweep showed 0.50 failing while 0.55 passed, a result indistinguishable from
    single-seed sampling noise near a fuzzy boundary.

    This script exists to answer a different, more trustworthy question: "if I ship
    max_line_width_ratio=0.4 today, how reliably does REAL production text render cleanly on the
    REAL primary target canvas (576x1024, 9:16)?" -- by:
      1. Calling the ACTUAL shipped API (`render_glyph(..., target_canvas_w=, target_canvas_h=,
         max_line_width_ratio=)`), not a hand-built synthetic box -- validates the code path users
         will actually call, not an isolated variable.
      2. Running each (text, ratio) config across MULTIPLE SEEDS (default 3: 42/123/777) and
         reporting a X/N reliability score per config, instead of a single anecdotal PASS/FAIL.
      3. Covering short / medium / long text on the SAME real canvas, so the reported reliability
         reflects the actual range of copy length the project needs, not one cherry-picked phrase.
      4. Including ONE bonus comparison at ratio=0.5 on the short text, to see whether 0.4 is
         leaving safety margin on the table that could be reclaimed later (informational only --
         0.4 remains the shipped default regardless of this row's outcome).

    A config is only worth calling "locked safe" if it scores N/N (all seeds clean). Anything less
    means real, reproducible unreliability -- not noise -- and should NOT be shipped as-is.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 (the one config already proven 100%
reliable in isolation -- so any unreliability found here is attributable to sizing, not crosstalk).

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_lock_validation.py                  # 3 texts x 3 seeds + bonus row = 12 runs
    python scripts/probe_glyph_lock_validation.py --seeds 42 123   # fewer seeds if GPU time is tight
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
from PIL import Image

from flux2.autoencoder import AutoEncoder
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.util import load_ae, load_flow_model, load_qwen3_embedder
from tendoo.glyph_engine import GlyphInfo, render_glyph


# ==================================================================================================
# 1. REAL PRODUCTION CANVAS + REALISTIC TEXT CORPUS (short / medium / long)
# ==================================================================================================

CANVAS_9x16 = (576, 1024)  # primary target format (AGENTS.md: "chuẩn 9:16 TikTok")

TEXT_CONFIGS = [
    dict(key="short", text="MUA 1 TẶNG 1", ratio=0.4),
    dict(key="medium", text="Đậm đà hương vị cà phê Việt", ratio=0.4),
    dict(key="long", text="Đậm đà hương vị cà phê Việt Nam pha phin truyền thống mỗi sáng", ratio=0.4),
    # Bonus/informational only -- does NOT change the shipped 0.4 default either way. NOTE: "short"
    # ("MUA 1 TẶNG 1") is NOT used here because its natural width (226px) already fits under BOTH
    # the 0.4 (230px) and 0.5 (288px) budgets -- ratio wouldn't change its wrapping at all, so it
    # can't tell us anything about 0.5's safety. "medium" genuinely wraps differently (3 lines at
    # 0.4 vs 2 wider lines at 0.5), so it actually exercises the looser threshold.
    dict(key="medium_bonus_r0.5", text="Đậm đà hương vị cà phê Việt", ratio=0.5),
]

DEFAULT_SEEDS = [42, 123, 777]


@dataclass
class RunConfig:
    config_key: str
    text: str
    ratio: float
    seed: int
    run_id: str


def build_run_matrix(seeds: List[int]) -> List[RunConfig]:
    runs: List[RunConfig] = []
    for cfg in TEXT_CONFIGS:
        for seed in seeds:
            runs.append(RunConfig(
                config_key=cfg["key"], text=cfg["text"], ratio=cfg["ratio"], seed=seed,
                run_id=f"{cfg['key']}_r{cfg['ratio']:.2f}_seed{seed}".replace(".", "p"),
            ))
    return runs


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
# 3. MAIN VALIDATION RUNNER
# ==================================================================================================

DEFAULT_PROMPT = (
    "Poster quảng cáo cà phê phong cách hiện đại tối giản, ánh sáng studio ấm áp tương phản cao, "
    "dòng chữ 3D mạ vàng đồng cổ sắc nét nổi bật, bố cục sạch sẽ gọn gàng, "
    "không có chữ ký, không có watermark, không có chữ trang trí thừa"
)


def run_validation(
    seeds: List[int],
    font: str = "bevietnam",
    prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_lock_validation",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - GLYPH ENGINE LOCK VALIDATION (MULTI-SEED, PRODUCTION CODE PATH)")
    print("=" * 100)
    print(f"  Canvas   : {CANVAS_9x16[0]}x{CANVAS_9x16[1]} (real 9:16 target format)")
    print(f"  Seeds    : {seeds}")
    print(f"  Configs  : {[c['key'] for c in TEXT_CONFIGS]}")
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

    canvas_w, canvas_h = CANVAS_9x16
    canvas_w = (canvas_w // 16) * 16
    canvas_h = (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    # Render each UNIQUE (text, ratio) config's glyph ONCE via the real production API -- the
    # bitmap does not depend on seed, only the diffusion sampling does.
    glyph_cache: Dict[Tuple[str, float], GlyphInfo] = {}
    for cfg in TEXT_CONFIGS:
        key = (cfg["text"], cfg["ratio"])
        if key not in glyph_cache:
            info = render_glyph(
                text=cfg["text"], font_name_or_path=font, auto_size=True,
                target_canvas_w=canvas_w, target_canvas_h=canvas_h,
                max_line_width_ratio=cfg["ratio"],
            )
            glyph_cache[key] = info
            ratio_actual = info.latent_w / lat_w
            print(f"\n[Glyph cache] \"{cfg['text']}\" @ ratio={cfg['ratio']} -> "
                  f"{info.width_px}x{info.height_px}px, {len(info.lines)}L, lat_w={info.latent_w} "
                  f"(actual canvas ratio={ratio_actual:.3f}) :: {info.lines}")
            info.image.save(out_path / f"{cfg['key']}_r{cfg['ratio']:.2f}_glyph.png".replace(".00", ""))

    print(f"\n[3/3] Executing {len(all_runs)} run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        glyph_info = glyph_cache[(run.text, run.ratio)]

        print("-" * 100)
        print(f"▶️  [{idx}/{len(all_runs)}] {run.run_id} (seed={run.seed})")
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
            "config_key": run.config_key, "text": run.text, "ratio": run.ratio, "seed": run.seed,
            "run_id": run.run_id, "glyph_lines": glyph_info.lines, "glyph_lat_w": glyph_info.latent_w,
            "elapsed_s": round(elapsed, 2), "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "lock_validation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] LOCK VALIDATION SUMMARY (fill in Verdict per run, then tally X/N per config below)")
    print("=" * 100)
    print(f"{'Run ID':<32} | {'Config':<18} | {'Seed':<6} | {'Lines':<6} | {'Verdict'}")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<32} | {r['config_key']:<18} | {r['seed']:<6} | {len(r['glyph_lines']):<6} | ? PASS/FAIL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP] For each config_key, count how many of its N seeds rendered CLEAN text:")
    print("  - N/N (e.g. 3/3)  -> genuinely reliable at that ratio, safe to lock.")
    print("  - <N/N            -> reproducible unreliability (not noise) -- ratio 0.4 needs to go")
    print("                       lower still, or that text length needs a different strategy.")
    print("  - Compare 'short' (ratio 0.4) vs 'short_bonus_r0.5' (ratio 0.5): if 0.5 is ALSO N/N,")
    print("    there is headroom to loosen 0.4 later for shorter copy without giving up safety.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Glyph Engine Lock Validation (multi-seed)")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to replicate (default: 42 123 777)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt")
    parser.add_argument("--output_dir", type=str, default="output_glyph_lock_validation", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")

    args = parser.parse_args()
    run_validation(
        seeds=args.seeds, font=args.font, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
