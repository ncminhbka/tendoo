#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - GLYPH ENGINE FINAL LOCK VALIDATION (SELF-ASPECT-RATIO RULE, MULTI-SEED)
====================================================================================================
Script: scripts/probe_glyph_final_lock_validation.py
Purpose:
    Validates the FINAL Rule 29 revision now shipped in glyph_engine.py's compute_optimal_glyph_box:
    line count is chosen to land the glyph box's OWN aspect ratio (width/height) inside [0.5, 1.3],
    replacing the earlier canvas-width-ratio theory that a documented production result (AGENTS.md
    Rule 11, canvas-ratio=0.875) and two independent GPU rounds (aspect-isolation sweep + 2x2
    token-mass x canvas-shape factorial) both showed did not hold.

    Two things this script exists to check with real confidence (not n=1-2 anecdote):
      1. Reliability at the new default: short/medium/long text on the REAL primary target canvas
         (576x1024, 9:16), each replicated over 5 FRESH seeds (42, 123, 777, 2024, 8888) -- not just
         seed 42, which multiple earlier rounds turned out to have been unusually lucky on (e.g. the
         "short" 1-line config in probe_glyph_lock_validation.py rendered clean ONLY on seed 42 of 3).
      2. The theory's central new claim -- canvas-INDEPENDENCE -- by also running the same 3 texts on
         a 1024x1024 (1:1) canvas with 2 cross-check seeds. If quality holds up similarly on both
         canvases, that corroborates self-aspect-ratio (not canvas shape) as the real driver.

    Calls the ACTUAL shipped API (`render_glyph(text=..., font_name_or_path=..., auto_size=True,
    target_canvas_w=, target_canvas_h=)` with NO ratio override -- the new default aspect-band logic
    is what's under test, exactly as production code will call it.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 ONLY.

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_final_lock_validation.py                # full: 21 runs (~15 min)
    python scripts/probe_glyph_final_lock_validation.py --seeds916 42 123   # fewer seeds if GPU time is tight
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
# 1. TEXT CORPUS + CANVASES + SEEDS
# ==================================================================================================

TEXTS = {
    "short": "MUA 1 TẶNG 1",
    "medium": "Đậm đà hương vị cà phê Việt",
    "long": "Đậm đà hương vị cà phê Việt Nam pha phin truyền thống mỗi sáng",
}
CANVAS_9x16 = (576, 1024)   # primary target format -- gets the full seed set
CANVAS_1x1 = (1024, 1024)  # cross-check for the theory's canvas-independence claim

DEFAULT_SEEDS_9x16 = [42, 123, 777, 2024, 8888]
DEFAULT_SEEDS_1x1 = [42, 555]


@dataclass
class RunConfig:
    text_key: str
    canvas_key: str
    seed: int
    run_id: str


def build_run_matrix(seeds_9x16: List[int], seeds_1x1: List[int]) -> List[RunConfig]:
    runs: List[RunConfig] = []
    for text_key in TEXTS:
        for seed in seeds_9x16:
            runs.append(RunConfig(text_key=text_key, canvas_key="9x16", seed=seed, run_id=f"{text_key}_9x16_seed{seed}"))
        for seed in seeds_1x1:
            runs.append(RunConfig(text_key=text_key, canvas_key="1x1", seed=seed, run_id=f"{text_key}_1x1_seed{seed}"))
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
# 3. MAIN VALIDATION RUNNER
# ==================================================================================================

DEFAULT_PROMPT = (
    "Poster quảng cáo cà phê phong cách hiện đại tối giản, ánh sáng studio ấm áp tương phản cao, "
    "dòng chữ 3D mạ vàng đồng cổ sắc nét nổi bật, bố cục sạch sẽ gọn gàng, "
    "không có chữ ký, không có watermark, không có chữ trang trí thừa"
)


def run_validation(
    seeds_9x16: List[int], seeds_1x1: List[int], font: str = "bevietnam", prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_final_lock_validation", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(seeds_9x16, seeds_1x1)

    print("=" * 100)
    print(" [*] TENDOO AI - GLYPH ENGINE FINAL LOCK VALIDATION (SELF-ASPECT-RATIO RULE)")
    print("=" * 100)
    print(f"  Texts       : {list(TEXTS.keys())}")
    print(f"  9:16 seeds  : {seeds_9x16}  (primary target format)")
    print(f"  1:1 seeds   : {seeds_1x1}  (canvas-independence cross-check)")
    print(f"  Total runs  : {len(all_runs)}")

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

    canvases = {"9x16": CANVAS_9x16, "1x1": CANVAS_1x1}

    # Render each UNIQUE (text, canvas) glyph ONCE via the real shipped API -- the bitmap does not
    # depend on seed, only the diffusion sampling does. NOTE: no ratio override passed -- this is
    # the new default self-aspect-ratio-band behavior exactly as production code will call it.
    glyph_cache: Dict[Tuple[str, str], GlyphInfo] = {}
    for text_key, text in TEXTS.items():
        for canvas_key, (cw, ch) in canvases.items():
            info = render_glyph(text=text, font_name_or_path=font, auto_size=True, target_canvas_w=cw, target_canvas_h=ch)
            glyph_cache[(text_key, canvas_key)] = info
            self_ar = info.width_px / info.height_px
            print(f"  [{text_key:7s}|{canvas_key:5s}] {info.width_px}x{info.height_px}px {info.font_size_pt}pt "
                  f"{len(info.lines)}L {info.token_count}tok selfAR={self_ar:.2f} :: {info.lines}")
            info.image.save(out_path / f"{text_key}_{canvas_key}_glyph.png")

    print(f"\n[3/3] Executing {len(all_runs)} denoise run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        glyph_info = glyph_cache[(run.text_key, run.canvas_key)]
        canvas_w, canvas_h = canvases[run.canvas_key]
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
        manifest.append({
            "run_id": run.run_id, "text_key": run.text_key, "canvas_key": run.canvas_key, "seed": run.seed,
            "glyph_lines": glyph_info.lines, "self_aspect": round(glyph_info.width_px / glyph_info.height_px, 3),
            "tokens": glyph_info.token_count, "elapsed_s": round(elapsed, 2),
            "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "final_lock_validation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] FINAL LOCK VALIDATION SUMMARY (fill in Verdict, then tally X/N per text x canvas)")
    print("=" * 100)
    print(f"{'Run ID':<24} | {'Canvas':<7} | {'SelfAR':<7} | {'Seed':<6} | {'Verdict'}")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<24} | {r['canvas_key']:<7} | {r['self_aspect']:<7.2f} | {r['seed']:<6} | ? PASS/FAIL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP]")
    print("  1. Per (text, canvas), count clean-render seeds out of N. This is the REAL reliability")
    print("     rate to plan production retries around -- do not expect single-shot 100%.")
    print("  2. Compare '<text>_9x16_*' vs '<text>_1x1_*' reliability: if similar, canvas-independence")
    print("     is corroborated -> self-aspect-ratio is confirmed as the operative rule to keep long-term.")
    print("  3. If any text/canvas scores notably worse than the others despite an in-band self-aspect,")
    print("     that is a genuine remaining gap (not explained by this rule) worth its own follow-up.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Glyph Engine Final Lock Validation")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--seeds916", type=int, nargs="+", default=DEFAULT_SEEDS_9x16, help="Seeds for the 9:16 primary format")
    parser.add_argument("--seeds1x1", type=int, nargs="+", default=DEFAULT_SEEDS_1x1, help="Seeds for the 1:1 cross-check")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt")
    parser.add_argument("--output_dir", type=str, default="output_glyph_final_lock_validation", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")

    args = parser.parse_args()
    run_validation(
        seeds_9x16=args.seeds916, seeds_1x1=args.seeds1x1, font=args.font, prompt=args.prompt,
        output_dir=args.output_dir, model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
