#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - REGIONAL PARALLEL DIFFUSION PROBE (DIRECTION 2, TRAINING-FREE, FIRST MINIMAL TEST)
====================================================================================================
Script: scripts/probe_regional_parallel_diffusion.py
Purpose:
    Direction 1 (RoPE spatial binding) was falsified and closed (AGENTS.md Rule 30): shifting a
    glyph reference's (h, w) coordinates away from the canonical local origin (0,0) -- even by a
    couple of latent units -- corrupted even the previously-bulletproof t=10 slot. The lesson:
    NEVER move a glyph reference away from its known-safe canonical anchor (t=10, local (0,0)).

    Direction 2 (Regional Parallel Diffusion, MultiDiffusion-style) is designed to respect that
    lesson by construction: instead of packing N glyph references into ONE joint forward pass
    (the actual source of Cross-Slot Attention Bleeding, AGENTS.md rules 10/12) or moving any
    glyph off its canonical anchor, run N SEPARATE branches -- each branch's forward pass contains
    EXACTLY ONE glyph reference, ALWAYS at the canonical (t=10, local (0,0)) anchor that has been
    proven bulletproof throughout this entire investigation. All branches share the SAME evolving
    canvas latent. After every Euler ODE step, each branch's predicted velocity is combined into
    ONE consensus update via a smooth per-region spatial mask (title's branch dominates the top
    rows, subtitle's branch dominates the bottom rows), and that single merged canvas is what all
    branches see as input to the next step. No two glyphs are EVER present in the same forward
    pass, and no glyph is ever moved off its canonical anchor -- crosstalk is avoided by
    construction rather than discouraged via attention bias (which is what the earlier, shelved
    `test_attention_steering.py` experiment tried, with mixed/inconclusive results and a
    ballooning hyperparameter surface the user explicitly did not want to tune at inference time).

    Two conditions, 3 seeds each, on the SAME 2-text layout used in the (now-closed) Direction 1
    probe for direct comparability:
      0) BASELINE           - standard method: both glyphs in ONE joint forward pass, t=10/t=20,
                              local origin (reproduces probe_rope_spatial_binding.py condition A).
      1) REGIONAL_PARALLEL  - the new mechanism described above.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B. Uses the LOCKED src/tendoo/glyph_engine.py render_glyph() API.

Usage on Remote Server (2x A30):
    python scripts/probe_regional_parallel_diffusion.py                 # both conditions x 3 seeds = 6 runs
    python scripts/probe_regional_parallel_diffusion.py --conditions regional_parallel  # new mechanism only
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
from flux2.model import Flux2
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.util import load_ae, load_flow_model, load_qwen3_embedder
from tendoo.glyph_engine import GlyphInfo, render_glyph


# ==================================================================================================
# 1. LAYOUT: 2-TEXT POSTER (title @ top, subtitle @ bottom), 9:16 CANVAS
#    (identical texts/canvas to probe_rope_spatial_binding.py for direct comparability)
# ==================================================================================================

TITLE_TEXT = "TUYỂN DỤNG NHÂN TÀI"
SUBTITLE_TEXT = "BỨT PHÁ MỌI GIỚI HẠN"
CANVAS = (576, 1024)  # real 9:16 primary target
DEFAULT_SEEDS = [42, 123, 777]
SPLIT_Y = 0.5        # canvas fraction: title branch dominates above this, subtitle below
MASK_SHARPNESS = 14.0  # sigmoid transition sharpness (matches the earlier test_attention_steering.py convention)

DEFAULT_PROMPT = (
    "Poster tuyển dụng phong cách công ty công nghệ hiện đại, nền gradient xanh dương đậm sang "
    "trọng, dòng chữ tiêu đề lớn dập nổi kim loại sắc nét ở phía trên, dòng chữ phụ phát sáng neon "
    "tinh tế ở phía dưới, bố cục sạch sẽ chuyên nghiệp, không có chữ ký, không có watermark"
)


# ==================================================================================================
# 2. 4D ROPE ENCODING -- ALWAYS canonical (local origin) per AGENTS.md Rule 30
# ==================================================================================================

def encode_glyph_to_ref_tokens(
    ae: AutoEncoder, glyph_img: Image.Image, t_offset: float, device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Canonical encoding ONLY: local (0,0) origin. Never shift -- see AGENTS.md Rule 30."""
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


def compute_regional_weights(lat_h: int, lat_w: int, split_y: float, sharpness: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smooth 2-way vertical partition of unity over canvas tokens: (w_top, w_bottom), each summing to 1 pointwise."""
    h_idx = torch.arange(lat_h, device=device, dtype=torch.float32)
    y_grid = (h_idx + 0.5) / float(lat_h)  # normalized row center, 0 (top) -> 1 (bottom)
    w_top_row = torch.sigmoid(sharpness * (split_y - y_grid))  # ~1 near top, ~0 near bottom
    w_bottom_row = 1.0 - w_top_row
    w_top = w_top_row.unsqueeze(1).expand(lat_h, lat_w).flatten()      # [num_canvas_tokens]
    w_bottom = w_bottom_row.unsqueeze(1).expand(lat_h, lat_w).flatten()
    return w_top, w_bottom


# ==================================================================================================
# 3. DENOISE LOOPS
# ==================================================================================================

def denoise_baseline_joint(
    model: Flux2, img: torch.Tensor, img_ids: torch.Tensor, txt: torch.Tensor, txt_ids: torch.Tensor,
    timesteps: List[float], guidance: float, ref_tokens: torch.Tensor, ref_ids: torch.Tensor,
) -> torch.Tensor:
    """Standard single joint forward pass per step (both glyphs concatenated) -- the current default method."""
    return denoise_cfg(
        model=model, img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, timesteps=timesteps,
        guidance=guidance, img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
    )


def denoise_regional_parallel(
    model: Flux2, img: torch.Tensor, img_ids: torch.Tensor, txt: torch.Tensor, txt_ids: torch.Tensor,
    timesteps: List[float], guidance: float,
    branch_refs: List[Tuple[torch.Tensor, torch.Tensor]], branch_weights: List[torch.Tensor],
) -> torch.Tensor:
    """
    N independent single-glyph branches sharing one canvas. Each branch runs its OWN full joint-
    attention forward pass (canvas + prompt + exactly ONE canonical glyph ref) -- crosstalk is
    impossible by construction since no two glyphs are ever in the same pass. Per-step, each
    branch's CFG-guided velocity prediction is combined via a smooth per-canvas-token spatial mask
    into ONE consensus update; that single merged canvas is what every branch sees next step.
    """
    n_canvas = img.shape[1]
    orig_dtype = img.dtype  # bfloat16 -- guarded against below, see the cast on `weight`
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((2,), t_curr, dtype=img.dtype, device=img.device)
        img_cfg = torch.cat([img, img], dim=0)          # [2, n_canvas, C] for CFG [uncond, cond]
        img_ids_cfg = torch.cat([img_ids, img_ids], dim=0)

        merged_pred = torch.zeros_like(img)  # [1, n_canvas, C]
        for (ref_tokens, ref_ids), weight in zip(branch_refs, branch_weights):
            ref_tokens_cfg = torch.cat([ref_tokens, ref_tokens], dim=0)
            ref_ids_cfg = torch.cat([ref_ids, ref_ids], dim=0)
            img_input = torch.cat([img_cfg, ref_tokens_cfg], dim=1)
            img_input_ids = torch.cat([img_ids_cfg, ref_ids_cfg], dim=1)

            pred = model(x=img_input, x_ids=img_input_ids, timesteps=t_vec, ctx=txt, ctx_ids=txt_ids, guidance=None)
            pred = pred[:, :n_canvas]
            pred_uncond, pred_cond = pred.chunk(2)
            pred_branch = pred_uncond + guidance * (pred_cond - pred_uncond)  # [1, n_canvas, C]

            # NOTE (bugfix): `weight` (from compute_regional_weights) is float32; multiplying it
            # directly against a bfloat16 `pred_branch` silently upcasts the product (and then
            # `img` itself, via the update below) to float32. On the NEXT loop iteration,
            # `t_vec = torch.full(..., dtype=img.dtype, ...)` then builds a float32 timestep
            # tensor, which crashes inside model.time_in's bfloat16 Linear layer ("expected mat1
            # and mat2 to have the same dtype"). Must cast weight to match pred_branch's dtype.
            merged_pred = merged_pred + weight.to(pred_branch.dtype).view(1, -1, 1) * pred_branch

        img = (img + (t_prev - t_curr) * merged_pred).to(orig_dtype)  # defensive: keep bfloat16

    return img


# ==================================================================================================
# 4. MAIN RUNNER
# ==================================================================================================

@dataclass
class RunConfig:
    condition: str
    seed: int
    run_id: str


def build_run_matrix(conditions: List[str], seeds: List[int]) -> List[RunConfig]:
    return [RunConfig(condition=c, seed=s, run_id=f"{c}_seed{s}") for c in conditions for s in seeds]


def run_probe(
    conditions: List[str], seeds: List[int], font: str = "bevietnam", prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_regional_parallel_diffusion", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(conditions, seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - REGIONAL PARALLEL DIFFUSION PROBE (DIRECTION 2, FIRST MINIMAL TEST)")
    print("=" * 100)
    print(f"  Title    : \"{TITLE_TEXT}\" (top)")
    print(f"  Subtitle : \"{SUBTITLE_TEXT}\" (bottom)")
    print(f"  Canvas   : {CANVAS[0]}x{CANVAS[1]} (9:16)")
    print(f"  Conditions: {conditions}")
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

    canvas_w, canvas_h = CANVAS
    canvas_w = (canvas_w // 16) * 16
    canvas_h = (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    # Render title + subtitle glyphs ONCE via the LOCKED production glyph_engine API.
    glyph_infos: Dict[str, GlyphInfo] = {}
    for role, text in [("title", TITLE_TEXT), ("subtitle", SUBTITLE_TEXT)]:
        info = render_glyph(text=text, font_name_or_path=font, auto_size=True, target_canvas_w=canvas_w, target_canvas_h=canvas_h)
        glyph_infos[role] = info
        print(f"  [{role:9s}] {info.width_px}x{info.height_px}px {info.font_size_pt}pt {len(info.lines)}L "
              f"{info.token_count}tok :: {info.lines}")
        info.image.save(out_path / f"{role}_glyph.png")

    # Both glyphs ALWAYS canonical: local (0,0) origin. Title @ t=10 (bulletproof anchor); subtitle
    # also encoded @ t=10 in its OWN branch (regional_parallel never puts 2 refs in one pass, so
    # there is no t=10/t=10 collision to worry about) and @ t=20 for the baseline joint pass (the
    # only place 2 refs ever coexist in one forward call, matching the pre-existing convention).
    title_ref_t10 = encode_glyph_to_ref_tokens(ae, glyph_infos["title"].image, 10.0, device_ae)
    subtitle_ref_t10 = encode_glyph_to_ref_tokens(ae, glyph_infos["subtitle"].image, 10.0, device_ae)
    subtitle_ref_t20 = encode_glyph_to_ref_tokens(ae, glyph_infos["subtitle"].image, 20.0, device_ae)

    w_top, w_bottom = compute_regional_weights(lat_h, lat_w, SPLIT_Y, MASK_SHARPNESS, device_dit)
    print(f"  Regional mask: split_y={SPLIT_Y}, sharpness={MASK_SHARPNESS} "
          f"(title dominates rows 0-{int(SPLIT_Y*lat_h)}, subtitle rows {int(SPLIT_Y*lat_h)}-{lat_h})")

    print(f"\n[3/3] Executing {len(all_runs)} run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        print("-" * 100)
        print(f"▶️  [{idx}/{len(all_runs)}] {run.run_id}")
        t_run_start = time.time()

        torch.manual_seed(run.seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

        with torch.no_grad():
            if run.condition == "baseline":
                ref_tokens = torch.cat([title_ref_t10[0], subtitle_ref_t20[0]], dim=1).to(device_dit)
                ref_ids = torch.cat([title_ref_t10[1], subtitle_ref_t20[1]], dim=1).to(device_dit)
                out_latent = denoise_baseline_joint(
                    model=model, img=img_tokens, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                    timesteps=timesteps, guidance=guidance, ref_tokens=ref_tokens, ref_ids=ref_ids,
                )
            else:  # "regional_parallel"
                branch_refs = [
                    (title_ref_t10[0].to(device_dit), title_ref_t10[1].to(device_dit)),
                    (subtitle_ref_t10[0].to(device_dit), subtitle_ref_t10[1].to(device_dit)),
                ]
                branch_weights = [w_top, w_bottom]
                out_latent = denoise_regional_parallel(
                    model=model, img=img_tokens, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                    timesteps=timesteps, guidance=guidance, branch_refs=branch_refs, branch_weights=branch_weights,
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
            "run_id": run.run_id, "condition": run.condition, "seed": run.seed,
            "elapsed_s": round(elapsed, 2), "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "regional_parallel_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] SUMMARY (fill in Verdict: are BOTH title AND subtitle correct/sharp/non-overlapping?)")
    print("=" * 100)
    print(f"{'Run ID':<28} | {'Condition':<20} | {'Seed':<6} | {'Time':<8} | Verdict")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<28} | {r['condition']:<20} | {r['seed']:<6} | {r['elapsed_s']}s{'':<4} | ? PASS/FAIL/PARTIAL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP] Tally X/3 seeds per condition, compare regional_parallel vs baseline:")
    print("  - regional_parallel clearly beats baseline (both title AND subtitle sharp/correct) ->")
    print("    the mechanism works -- worth extending to 3-4 slots, adding a product reference,")
    print("    and checking global style/lighting coherence across the merged regions.")
    print("  - regional_parallel ~= baseline (subtitle still bad) -> per-branch generation ok in")
    print("    isolation but the per-step latent merge itself is reintroducing interference --")
    print("    would need to inspect whether the merge boundary itself is the problem (try a")
    print("    harder split, or fewer merge steps late in the schedule).")
    print("  - Look also for a visible SEAM at the split_y boundary (indicates the regions aren't")
    print("    blending stylistically even if each half's text is individually correct).\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Regional Parallel Diffusion Probe (Direction 2)")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--conditions", type=str, nargs="+", default=["baseline", "regional_parallel"],
                         choices=["baseline", "regional_parallel"], help="Which conditions to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to replicate")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt")
    parser.add_argument("--output_dir", type=str, default="output_regional_parallel_diffusion", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")

    args = parser.parse_args()
    run_probe(
        conditions=args.conditions, seeds=args.seeds, font=args.font, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
