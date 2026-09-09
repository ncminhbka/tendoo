#!/usr/bin/env python3
"""
scripts/verify_ref_kv_cache_hypothesis.py

Empirically verifies (or refutes) the hypothesis raised while auditing
denoise_regional_velocity_blended() in scripts/pipeline_e2e_poster.py:

  "Reusing model.forward_kv_extract()/forward_kv_cached() to skip recomputing the
   reference-image tokens' K/V at every denoise step is NOT a value-preserving
   optimization, because forward_kv_extract gives reference tokens a separate FIXED
   modulation vector (ref_fixed_timestep, default 0.0), whereas the current plain
   model() calls modulate ref tokens uniformly with the same evolving t_vec as the
   canvas tokens every step."

This script runs BOTH formulations from the exact same initial noise / prompts /
reference image, and reports:
  1. How far the two sampled latents diverge, step by step (L2 / cosine distance).
  2. How different the two FINAL decoded images are (pixel MAE, saved diff heatmap).
  3. Wall-clock time for each path, to quantify whether the KV-cache path is even
     worth the tradeoff if the hypothesis turns out to be wrong (i.e. if the two
     paths turn out visually indistinguishable, the fixed-modulation KV-cache path
     is a straightforward win and should be adopted).

Requires a real FLUX.2 Klein 4B checkpoint + a real product reference image -- this
only matters when a reference image is used at all; without one, both formulations
are already identical (no ref tokens to disagree about) and this script will refuse
to run (pass --ref-image).

USAGE (on the GPU server):
  python scripts/verify_ref_kv_cache_hypothesis.py \\
    --ref-image path/to/product.png \\
    --preset fresh_beverage \\
    --steps 8 --seed 42 --width 576 --height 1024 \\
    --out-dir output_verify_kv_cache

Reads PRESETS/prompt_scene/prompt_corridor/layout straight from pipeline_e2e_poster.py
so the scene/corridor prompts and mask geometry exactly match production.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from flux2 import util
from flux2.sampling import get_schedule, prc_img, prc_txt

import pipeline_e2e_poster as pep  # noqa: E402  (PRESETS, mask builders, ref-image loader, Path A itself)


# ==================================================================================================
# Path B: hypothesis under test -- KV-cached reference tokens with a FIXED modulation
# (ref_fixed_timestep), batched the same way Path A now is (see the commit that added
# the batching optimization: pred_scene/pred_corridor share x/x_ids/timesteps/guidance,
# only ctx differs, so both are one batch-2 forward call instead of two sequential ones).
# ==================================================================================================

def denoise_regional_velocity_blended_kv_cached(
    model: Any,
    canvas_init: torch.Tensor,      # (1, L_canvas, C) initial canvas noise
    canvas_ids: torch.Tensor,       # (1, L_canvas, 4)
    ref_tokens: torch.Tensor,       # (1, L_ref, C) encoded reference-image tokens
    ref_ids: torch.Tensor,          # (1, L_ref, 4)
    txt_scene: torch.Tensor,
    txt_scene_ids: torch.Tensor,
    txt_corridor: torch.Tensor,
    txt_corridor_ids: torch.Tensor,
    spatial_mask: torch.Tensor,     # (1, L_canvas, 1)
    timesteps: List[float],
    guidance: float = 1.5,
    ref_fixed_timestep: float = 0.0,
    record_trajectory: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Same regional velocity blending, but the reference-image tokens' K/V are extracted
    once (step 0, via model.forward_kv_extract) and reused every subsequent step (via
    model.forward_kv_cached) instead of being recomputed from a plain model() call.
    forward_kv_extract/forward_kv_cached modulate reference tokens with a FIXED
    `ref_fixed_timestep` rather than the evolving per-step t_vec the canvas tokens (and
    the current production path) use -- that's the exact semantic difference under test.

    Returns (final_canvas_latent, trajectory) where trajectory is a list of the blended
    canvas latent after every step (only populated if record_trajectory=True), so the
    caller can measure how far the two paths' intermediate states diverge, not just the
    endpoint.
    """
    orig_dtype = canvas_init.dtype
    device = canvas_init.device
    mask = spatial_mask.to(device=device, dtype=orig_dtype)

    canvas_b = torch.cat([canvas_init, canvas_init], dim=0)
    canvas_ids_b = torch.cat([canvas_ids, canvas_ids], dim=0)
    ref_b = torch.cat([ref_tokens, ref_tokens], dim=0)
    ref_ids_b = torch.cat([ref_ids, ref_ids], dim=0)
    txt_b = torch.cat([txt_scene, txt_corridor], dim=0)
    txt_ids_b = torch.cat([txt_scene_ids, txt_corridor_ids], dim=0)

    trajectory: List[torch.Tensor] = []
    kv_cache = None

    for step_idx in range(len(timesteps) - 1):
        t_curr = timesteps[step_idx]
        t_prev = timesteps[step_idx + 1]
        t_vec = torch.full((canvas_b.shape[0],), t_curr, dtype=orig_dtype, device=device)
        guidance_vec = torch.full((canvas_b.shape[0],), guidance, dtype=orig_dtype, device=device)

        if step_idx == 0:
            pred_b, kv_cache = model.forward_kv_extract(
                x=canvas_b,
                x_ids=canvas_ids_b,
                timesteps=t_vec,
                ctx=txt_b,
                ctx_ids=txt_ids_b,
                guidance=guidance_vec,
                x_seq_concat=ref_b,
                x_seq_concat_ids=ref_ids_b,
                ref_fixed_timestep=ref_fixed_timestep,
            )
        else:
            pred_b = model.forward_kv_cached(
                x=canvas_b,
                x_ids=canvas_ids_b,
                timesteps=t_vec,
                ctx=txt_b,
                ctx_ids=txt_ids_b,
                guidance=guidance_vec,
                kv_cache=kv_cache,
            )

        pred_scene, pred_corridor = pred_b[0:1], pred_b[1:2]
        v_blend = (1.0 - mask) * pred_scene + mask * pred_corridor

        new_canvas = canvas_b[0:1, ...] + (t_prev - t_curr) * v_blend
        new_canvas = new_canvas.to(orig_dtype)
        if record_trajectory:
            trajectory.append(new_canvas.detach().clone())
        canvas_b = torch.cat([new_canvas, new_canvas], dim=0)

    return canvas_b[0:1], trajectory


def _run_path_a_with_trajectory(
    model, canvas_init, canvas_ids, ref_tokens, ref_ids,
    txt_scene, txt_scene_ids, txt_corridor, txt_corridor_ids,
    spatial_mask, timesteps, guidance,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Re-runs the CURRENT production denoise_regional_velocity_blended() step-by-step
    (instead of calling it as one opaque function) purely so we can record the same
    per-step trajectory Path B records, for a fair step-by-step divergence comparison.
    Numerically identical to calling pep.denoise_regional_velocity_blended() directly
    (same batching, same math) -- see tests/test_pipeline_velocity_blending.py."""
    orig_dtype = canvas_init.dtype
    device = canvas_init.device
    mask = spatial_mask.to(device=device, dtype=orig_dtype)
    L_canvas = canvas_init.shape[1]

    img = torch.cat([canvas_init, ref_tokens], dim=1)
    img_ids = torch.cat([canvas_ids, ref_ids], dim=1)

    img_b = torch.cat([img, img], dim=0)
    img_ids_b = torch.cat([img_ids, img_ids], dim=0)
    txt_b = torch.cat([txt_scene, txt_corridor], dim=0)
    txt_ids_b = torch.cat([txt_scene_ids, txt_corridor_ids], dim=0)

    trajectory: List[torch.Tensor] = []

    for step_idx in range(len(timesteps) - 1):
        t_curr = timesteps[step_idx]
        t_prev = timesteps[step_idx + 1]
        t_vec = torch.full((img_b.shape[0],), t_curr, dtype=orig_dtype, device=device)
        guidance_vec = torch.full((img_b.shape[0],), guidance, dtype=orig_dtype, device=device)

        pred_b = model(x=img_b, x_ids=img_ids_b, timesteps=t_vec, ctx=txt_b, ctx_ids=txt_ids_b, guidance=guidance_vec)
        pred_scene, pred_corridor = pred_b[0:1], pred_b[1:2]

        v_scene_canvas = pred_scene[:, :L_canvas, :]
        v_corridor_canvas = pred_corridor[:, :L_canvas, :]
        v_blend_canvas = (1.0 - mask) * v_scene_canvas + mask * v_corridor_canvas

        canvas_tokens = img_b[0:1, :L_canvas, :] + (t_prev - t_curr) * v_blend_canvas
        canvas_tokens = canvas_tokens.to(orig_dtype)
        trajectory.append(canvas_tokens.detach().clone())

        new_img = torch.cat([canvas_tokens, img_b[0:1, L_canvas:, :]], dim=1)
        img_b = torch.cat([new_img, new_img], dim=0)

    return img_b[0:1, :L_canvas, :], trajectory


def _sync(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _decode(ae, ae_dtype, aux_device, canvas_latent, h_lat, w_lat, C) -> Image.Image:
    z = canvas_latent[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
    with torch.no_grad():
        x = ae.decode(z).float()
    x = ((x[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def _label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(str(PROJECT_ROOT / "fonts" / "BeVietnamPro-Black.ttf"), 22)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    draw.text((10, 6), text, fill=(255, 255, 255), font=font)
    return img


def main():
    parser = argparse.ArgumentParser(description="Verify: does the ref-token KV-cache path change sampled output?")
    parser.add_argument("--ref-image", type=str, required=True, help="Required: a real product reference image path.")
    parser.add_argument("--preset", type=str, default="fresh_beverage", choices=list(pep.PRESETS.keys()),
                        help="Which PRESETS entry's scene/corridor prompts + mask layout to use.")
    parser.add_argument("--ref-fixed-timestep", type=float, default=0.0,
                        help="ref_fixed_timestep to pass to forward_kv_extract (default matches its own default, 0.0).")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=str, default="output_verify_kv_cache")
    args = parser.parse_args()

    ref_path = Path(args.ref_image)
    if not ref_path.exists():
        print(f"[Error] --ref-image not found: {ref_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 VERIFY: does caching reference-image KV (fixed modulation) change the output?")
    print("=" * 80)

    # ---- Device / model loading (mirrors pipeline_e2e_poster.py::main()) ----
    device = args.device
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        aux_device = "cuda:1"
    elif num_gpus == 1:
        aux_device = "cuda:0"
    else:
        device = aux_device = "cpu"
        print("  [Warning] No CUDA device detected -- this WILL be extremely slow / may not "
              "be practical for a 4B model, but the comparison logic itself is device-agnostic.")

    model_name = "flux.2-klein-4b"
    pdata_candidates = [
        Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-4B")),
        Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B")),
    ]
    for pdata in pdata_candidates:
        distill_cand = pdata / "flux-2-klein-4b.safetensors"
        if distill_cand.exists() and "KLEIN_4B_MODEL_PATH" not in os.environ:
            os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_cand)
            break

    print("⏳ Loading FLUX.2 Klein 4B Distill...")
    model = util.load_flow_model(model_name, device=device)
    model.eval()
    ae = util.load_ae(model_name, device=aux_device)
    ae.eval()
    ae_dtype = next(ae.parameters()).dtype
    text_encoder = util.load_text_encoder(model_name, device=aux_device)

    # ---- Build the exact same inputs both paths will consume ----
    case_info = pep.PRESETS[args.preset]
    w, h = args.width, args.height
    h_lat, w_lat, C = h // 16, w // 16, 128

    mask_type = case_info.get("layout", "hourglass")
    envelope = pep.compute_text_envelope(content=case_info.get("content", {}), width=w, height=h, layout_type=mask_type)
    y_top_max, w_half_req = envelope["y_top_max"], envelope["w_half_req"]

    # Mirrors pipeline_e2e_poster.py::run_e2e_case's own dispatch verbatim (kwarg names
    # genuinely differ per mask function -- don't "simplify" this into one generic call).
    if mask_type == "adaptive" or "text_zones" in case_info:
        mask_np = pep.build_adaptive_text_corridor_mask(h_lat, w_lat, text_zones=envelope.get("text_zones"), delta=0.04)
    elif mask_type == "dome":
        mask_np = pep.build_beverage_dome_mask(h_lat, w_lat, y_max=y_top_max, w_half_top=w_half_req, delta=0.04)
    elif mask_type == "hero_top":
        mask_np = pep.build_hero_top_corridor_mask(h_lat, w_lat, y_max=y_top_max, w_half_top=w_half_req, delta=0.04)
    else:
        mask_np = pep.build_hourglass_corridor_mask(h_lat, w_lat, y_top_max=y_top_max, w_half_top=w_half_req, delta=0.04)
    mask_tensor = torch.from_numpy(mask_np).view(1, -1, 1).to(device=device)

    print("🔤 Encoding scene/corridor prompts (Qwen3)...")
    with torch.no_grad():
        ctx_scene = text_encoder([case_info["prompt_scene"]]).to(torch.bfloat16)
        ctx_scene, ctx_scene_ids = prc_txt(ctx_scene[0])
        ctx_scene, ctx_scene_ids = ctx_scene.unsqueeze(0).to(device), ctx_scene_ids.unsqueeze(0).to(device)

        ctx_corridor = text_encoder([case_info["prompt_corridor"]]).to(torch.bfloat16)
        ctx_corridor, ctx_corridor_ids = prc_txt(ctx_corridor[0])
        ctx_corridor, ctx_corridor_ids = ctx_corridor.unsqueeze(0).to(device), ctx_corridor_ids.unsqueeze(0).to(device)

    torch.manual_seed(args.seed)
    z_init = torch.randn(1, C, h_lat, w_lat, device=device, dtype=torch.bfloat16)
    canvas_init, canvas_ids = prc_img(z_init[0])
    canvas_init, canvas_ids = canvas_init.unsqueeze(0).to(device), canvas_ids.unsqueeze(0).to(device)
    L_canvas = canvas_init.shape[1]

    print(f"📎 Encoding reference image: {ref_path}")
    ref_tokens, ref_ids = pep.load_and_encode_ref_image(ref_image_path=ref_path, ae=ae, device=device, target_dim=512, time_offset=10.0)

    timesteps = get_schedule(num_steps=args.steps, image_seq_len=canvas_init.shape[1] + ref_tokens.shape[1])

    # ---- Path A: current production path (uniform t_vec modulation for ref tokens) ----
    print("\n▶ Path A -- production denoise_regional_velocity_blended (uniform t_vec modulation)")
    _sync(device)
    t0 = time.time()
    with torch.no_grad():
        final_a, traj_a = _run_path_a_with_trajectory(
            model, canvas_init.clone(), canvas_ids.clone(), ref_tokens.clone(), ref_ids.clone(),
            ctx_scene, ctx_scene_ids, ctx_corridor, ctx_corridor_ids,
            mask_tensor, timesteps, args.guidance,
        )
    _sync(device)
    dur_a = time.time() - t0
    print(f"  [✓] Path A done in {dur_a:.3f}s")

    # ---- Path B: hypothesis under test (KV-cached, fixed ref_fixed_timestep modulation) ----
    print(f"\n▶ Path B -- forward_kv_extract/forward_kv_cached (ref_fixed_timestep={args.ref_fixed_timestep})")
    _sync(device)
    t0 = time.time()
    with torch.no_grad():
        final_b, traj_b = denoise_regional_velocity_blended_kv_cached(
            model, canvas_init.clone(), canvas_ids.clone(), ref_tokens.clone(), ref_ids.clone(),
            ctx_scene, ctx_scene_ids, ctx_corridor, ctx_corridor_ids,
            mask_tensor, timesteps, args.guidance,
            ref_fixed_timestep=args.ref_fixed_timestep, record_trajectory=True,
        )
    _sync(device)
    dur_b = time.time() - t0
    print(f"  [✓] Path B done in {dur_b:.3f}s")

    # ---- Numeric comparison: latent-space divergence, step by step ----
    print("\n📊 Per-step latent divergence (Path A vs Path B), L2 norm of (a - b):")
    for i, (a_step, b_step) in enumerate(zip(traj_a, traj_b)):
        diff = (a_step.float() - b_step.float())
        l2 = diff.norm().item()
        rel = l2 / (a_step.float().norm().item() + 1e-8)
        cos = torch.nn.functional.cosine_similarity(a_step.float().flatten(), b_step.float().flatten(), dim=0).item()
        print(f"  step {i+1}/{len(traj_a)}: L2={l2:8.4f}  relative={rel:7.4%}  cos_sim={cos:.6f}")

    final_diff = (final_a.float() - final_b.float())
    print(f"\n📊 Final latent: L2={final_diff.norm().item():.4f}  "
          f"relative={final_diff.norm().item() / (final_a.float().norm().item()+1e-8):.4%}  "
          f"max_abs={final_diff.abs().max().item():.6f}")

    # ---- Decode both to real images and save for visual inspection ----
    print("\n🖼  Decoding both paths to pixel space...")
    img_a = _decode(ae, ae_dtype, aux_device, final_a, h_lat, w_lat, C)
    img_b = _decode(ae, ae_dtype, aux_device, final_b, h_lat, w_lat, C)

    arr_a = np.asarray(img_a).astype(np.float32)
    arr_b = np.asarray(img_b).astype(np.float32)
    pixel_mae = float(np.abs(arr_a - arr_b).mean())
    pixel_max = float(np.abs(arr_a - arr_b).max())
    print(f"📊 Decoded pixel MAE (0-255 scale): {pixel_mae:.3f}   max abs diff: {pixel_max:.1f}")

    diff_vis = np.clip(np.abs(arr_a - arr_b) * 4, 0, 255).astype(np.uint8)  # 4x amplified for visibility
    diff_img = Image.fromarray(diff_vis)

    path_a_file = out_dir / "path_a_production_uniform_modulation.png"
    path_b_file = out_dir / "path_b_kv_cached_fixed_modulation.png"
    diff_file = out_dir / "diff_amplified_4x.png"
    img_a.save(path_a_file)
    img_b.save(path_b_file)
    diff_img.save(diff_file)

    side_by_side = Image.new("RGB", (w * 2 + 20, h + 40), (30, 30, 30))
    side_by_side.paste(_label(img_a, f"A: production ({dur_a:.2f}s)"), (0, 40))
    side_by_side.paste(_label(img_b, f"B: kv-cached ({dur_b:.2f}s)"), (w + 20, 40))
    side_by_side_file = out_dir / "side_by_side.png"
    side_by_side.save(side_by_side_file)

    print(f"\n💾 Saved: {path_a_file.name}, {path_b_file.name}, {diff_file.name}, {side_by_side_file.name} -> {out_dir}/")

    speedup = dur_a / dur_b if dur_b > 0 else float("nan")
    print("\n" + "=" * 80)
    print("VERDICT INPUTS (read the numbers + look at side_by_side.png / diff_amplified_4x.png "
          "yourself -- 'is this different enough to matter' is a visual judgment call):")
    print(f"  Speed:   Path A {dur_a:.3f}s  vs  Path B {dur_b:.3f}s  ->  {speedup:.2f}x "
          f"({'B faster' if speedup > 1 else 'A faster'})")
    print(f"  Latent:  final relative L2 divergence = {final_diff.norm().item() / (final_a.float().norm().item()+1e-8):.4%}")
    print(f"  Pixels:  mean abs diff = {pixel_mae:.3f} / 255, max = {pixel_max:.1f} / 255")
    print("  If pixel MAE is small (~single digits) and the two images look the same to your eye,")
    print("  the KV-cache path (Path B) is a safe, straightforward speed win -- adopt it.")
    print("  If the images visibly differ (e.g. the product itself renders differently), the")
    print("  hypothesis is CONFIRMED: the two formulations are not interchangeable, and the")
    print("  speed gain would come at the cost of silently changing which sampling recipe is")
    print("  actually used for reference-image conditioning.")
    print("=" * 80)


if __name__ == "__main__":
    main()
