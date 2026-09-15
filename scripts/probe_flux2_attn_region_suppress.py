#!/usr/bin/env python3
"""
scripts/probe_flux2_attn_region_suppress.py

Isolated real-GPU probe comparing 2 different mechanisms for "keep the corridor-mask
region clean/simple" against each other, on the SAME category/mask/seed/steps, so the
comparison is apples-to-apples with scripts/probe_diffusion_mask_adherence.py's own
busyness-ratio metric:

  (A) BASELINE -- the current production mechanism, unchanged: 2-branch Regional
      Velocity Blending (src/tendoo/velocity_blending.py::denoise_regional_velocity_blended).
      Runs a full extra forward pass every step (scene branch + corridor branch,
      batched as one Batch-2 call) and blends the two predicted velocities by mask.

  (B) PROBE -- single-branch cross-attention region suppression, inspired by Desigen
      (CVPR 2024, arxiv 2403.09093)'s inference-time technique:
        A'(x_t,t) = beta * A(x_t,t) * M_r + A(x_t,t) * (1 - M_r)
      Desigen assumes a classic U-Net with a separately-addressable, already-materialized
      cross-attention probability map -- FLUX.2 is a joint-attention DiT (model.py's
      causal_attn_fn) computed entirely inside a single fused
      F.scaled_dot_product_attention call, which never materializes attention
      probabilities as a readable/writable tensor. So (B) reimplements the SAME
      qualitative effect a different way: an ADDITIVE bias of log(beta) on the
      attention LOGITS (added before softmax, via SDPA's native `attn_mask` argument --
      see flux2/model.py's `causal_attn_fn`/`Flux2.forward` `attn_bias` param, added
      2026-09-15 specifically to enable this probe, default None everywhere = zero
      change to existing production behavior) restricted to (query = an image token
      whose spatial position falls inside the corridor mask) x (key = any TEXT token).
      This uses only ONE forward pass per step (no corridor branch, no Batch-2), and
      needs only the scene prompt -- no separate corridor prompt at all.

KNOWN LIMITATION, DO NOT SKIP READING THIS BEFORE INTERPRETING RESULTS:
  Desigen's technique suppresses the ONLY channel a classic U-Net's image tokens have
  to "salient object" conditioning: cross-attention to text. FLUX.2's joint
  self-attention gives image tokens a SECOND channel -- image-to-image attention among
  themselves -- that (B) does NOT suppress here (that would need a live saliency
  signal for "which other image tokens are salient", which is exactly the extra
  saliency-detector machinery Desigen also relies on and which this probe deliberately
  does not attempt). So (B) may under-perform its theoretical ceiling: if the busyness
  ratio for (B) is still noticeably above (A)'s, that is evidence of this specific
  image-to-image leak, not proof the whole approach is wrong.

WHAT THIS SCRIPT MEASURES: identical metric to probe_diffusion_mask_adherence.py --
Laplacian-variance "busyness" INSIDE the corridor mask vs OUTSIDE it (lower ratio =
cleaner/simpler corridor, which is what makes an overlaid title/CTA readable). Also
records wall-clock latency per run, since the whole point of (B) is that it's
1 branch instead of 2 -- if the ratio is comparable, latency is the tie-breaker.

Run this on the actual GPU server (this dev environment has no GPU):
  python scripts/probe_flux2_attn_region_suppress.py --category promo
  python scripts/probe_flux2_attn_region_suppress.py --category feedback --beta 1.0,0.1,0.01,0.001

Output:
  output_flux2_attn_suppress_probe/baseline/background.png, mask_overlay.png
  output_flux2_attn_suppress_probe/beta<B>/background.png, mask_overlay.png
  output_flux2_attn_suppress_probe/SUMMARY.md
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tendoo.core.style import resolve_style_preset
from tendoo.demo_server import (
    detect_scene_lighting_tone,
    inject_spatial_layout_guidance,
    sanitize_and_inject_zero_text,
)
from tendoo.engine import get_layout
from tendoo.engine.blocks import map_category_to_default_blocks
from tendoo.velocity_blending import denoise_regional_velocity_blended

OUTPUT_DIR = PROJECT_ROOT / "output_flux2_attn_suppress_probe"

ASPECT_DIMS: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "9:16": (576, 1024),
    "16:9": (1024, 576),
    "4:5": (816, 1024),
}

# Same representative content set as probe_diffusion_mask_adherence.py, kept in sync
# on purpose -- if you add a category there, add it here too so both probes stay
# comparable against each other.
SAMPLE_FIELDS: Dict[str, Dict[str, Any]] = {
    "promo": {
        "title": "ƯU ĐÃI ĐẶC BIỆT",
        "fields": {"discount": "GIẢM 40%", "applied_product": "Toàn bộ sản phẩm", "dates": "Từ 01/07 - Đến 31/07"},
    },
    "product_intro": {
        "title": "SẢN PHẨM MỚI",
        "fields": {"price": "1.990.000đ", "product_desc": "Chống ồn chủ động, âm thanh Hi-Res", "highlights": "Pin 40 giờ"},
    },
    "opening": {
        "title": "KHAI TRƯƠNG",
        "fields": {"opening_date": "15/10/2026", "opening_promo": "Tặng 100 phần quà"},
    },
    "feedback": {
        "title": None,
        "fields": {"feedback_target": "Dịch Vụ Spa", "feedback_quote": "Rất hài lòng với kết quả", "feedback_rating": "5"},
    },
    "recruitment": {
        "title": "TUYỂN DỤNG",
        "fields": {"job_position": "AI Engineer", "apply_deadline": "31/10/2026"},
    },
    "guide": {
        "title": "HƯỚNG DẪN",
        "fields": {"guide_steps": "Bước 1 | Bước 2 | Bước 3"},
    },
}


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def corridor_busyness_ratio(image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """Identical metric to probe_diffusion_mask_adherence.py -- kept as a verbatim
    copy (not a cross-script import, matching this repo's existing convention of
    self-contained probe scripts) so results from both scripts are directly comparable
    without depending on the other script's internals staying stable."""
    gray = (0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2])
    g = gray.astype(np.float32)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1)
    )
    lap_sq = lap ** 2
    inside = mask > 0.5
    outside = ~inside
    inside_score = float(lap_sq[inside].mean()) if inside.any() else 0.0
    outside_score = float(lap_sq[outside].mean()) if outside.any() else 0.0
    ratio = (inside_score / outside_score) if outside_score > 1e-6 else float("nan")
    return ratio, inside_score, outside_score


def save_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    overlay = image_rgb.astype(np.float32).copy()
    red_tint = np.zeros_like(overlay)
    red_tint[..., 0] = 255
    alpha = (mask[..., None] * 0.35)
    blended = overlay * (1 - alpha) + red_tint * alpha
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(out_path)


def resolve_devices() -> Tuple[str, str]:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "cuda:0", "cuda:1"
    if torch.cuda.is_available():
        return "cuda:0", "cuda:0"
    raise RuntimeError(
        "No CUDA device available -- this script exercises the real FLUX.2 diffusion "
        "model and cannot run in mock/CPU mode. Run it on the real GPU server."
    )


def build_region_suppress_attn_bias(
    mask_flat: torch.Tensor,  # (1, L_img, 1) float in [0,1], one value per image token
    num_txt_tokens: int,
    beta: float,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """
    Additive attention-logit bias for flux2.model.causal_attn_fn's `attn_bias` param.
    Sequence order for a plain (no ref-image) Flux2.forward() call is [txt, img], so
    the joint attention matrix is (num_txt_tokens + L_img) x (same) -- see this
    module's docstring for why this is a pre-softmax additive bias (log(beta)) rather
    than Desigen's literal post-softmax rescale.

    beta=1.0 returns an all-zero bias (a true no-op -- use this as a sanity check that
    the single-branch path reproduces plain unmasked generation before trusting the
    suppressed runs).
    """
    L_img = mask_flat.shape[1]
    L = num_txt_tokens + L_img
    bias = torch.zeros((L, L), dtype=dtype, device=device)
    if beta >= 1.0:
        return bias
    log_beta = math.log(max(beta, 1e-6))
    region = mask_flat[0, :, 0] > 0.5
    img_rows = num_txt_tokens + torch.nonzero(region, as_tuple=True)[0].to(device)
    if img_rows.numel() == 0:
        return bias
    txt_cols = torch.arange(num_txt_tokens, device=device)
    bias[img_rows.unsqueeze(1), txt_cols.unsqueeze(0)] = log_beta
    return bias


def denoise_single_branch_attn_suppressed(
    model: Any,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: List[float],
    guidance: float,
    attn_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Plain single-branch Euler denoise (mirrors flux2.sampling.denoise) but threads
    `attn_bias` into every model() call -- this is mechanism (B), see module docstring."""
    device = img.device
    orig_dtype = img.dtype
    guidance_vec = torch.full((img.shape[0],), guidance, dtype=orig_dtype, device=device)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=orig_dtype, device=device)
        pred = model(
            x=img,
            x_ids=img_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
            attn_bias=attn_bias,
        )
        img = img + (t_prev - t_curr) * pred
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="promo", choices=list(SAMPLE_FIELDS.keys()))
    parser.add_argument("--style-hint", default="auto")
    parser.add_argument("--aspect-ratio", default="1:1", choices=list(ASPECT_DIMS.keys()))
    parser.add_argument("--scene-prompt", default="")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--beta", default="1.0,0.1,0.01,0.001",
        help="Comma-separated attention-suppression strengths to sweep for mechanism (B). "
             "1.0 = no suppression (sanity check). Lower = stronger.",
    )
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    betas = _parse_float_list(args.beta)
    w, h = ASPECT_DIMS[args.aspect_ratio]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FLUX.2 ATTENTION REGION-SUPPRESSION PROBE -- (A) 2-branch velocity blend  vs  (B) single-branch attn-bias")
    print(f"category={args.category} style_hint={args.style_hint} aspect_ratio={args.aspect_ratio} "
          f"steps={args.steps} guidance={args.guidance} seed={args.seed}")
    print(f"sweeping beta={betas} for mechanism (B)")
    print("=" * 90)

    device_dit, device_aux = resolve_devices()
    print(f"[Device] DiT on {device_dit}, AE + Qwen3 on {device_aux}")

    from flux2 import util
    from flux2.sampling import get_schedule, prc_img, prc_txt

    t0 = time.time()
    print("Loading DiT 4B weights...")
    dit_model = util.load_flow_model("flux.2-klein-4b", device=device_dit)
    dit_model.eval()
    print("Loading AutoEncoder...")
    ae_model = util.load_ae("flux.2-klein-4b", device=device_aux)
    ae_model.eval()
    ae_dtype = next(ae_model.parameters()).dtype
    print("Loading Qwen3 text encoder...")
    text_encoder = util.load_text_encoder("flux.2-klein-4b", device=device_aux)
    print(f"Models loaded in {time.time() - t0:.1f}s")

    # --- Build the REAL corridor mask + prompts, exactly as demo_server.py would
    # (identical construction to probe_diffusion_mask_adherence.py, kept in sync) ---
    layout = get_layout("omni")
    sample = SAMPLE_FIELDS[args.category]
    default_blocks = map_category_to_default_blocks(
        category=args.category, title=sample["title"], fields=sample["fields"],
    )
    plan_blocks = [b.to_dict() for b in default_blocks]
    mask_np = layout.generate_mask(width=w, height=h, blocks=plan_blocks, font_key="bevietnam")

    _preset = resolve_style_preset(args.category, "auto")
    style_hint = args.style_hint if args.style_hint != "auto" else _preset["style_hint"]
    raw_scene_prompt = args.scene_prompt or ""
    zero_text_prompt = sanitize_and_inject_zero_text(raw_scene_prompt, has_ref_image=False)
    final_scene_prompt = inject_spatial_layout_guidance(zero_text_prompt, layout_name=layout.name)
    style_hint = detect_scene_lighting_tone(final_scene_prompt, style_hint, layout_name=layout.name)
    corridor_prompt = layout.get_corridor_prompt(style_hint)  # only mechanism (A) uses this

    print(f"\nscene_prompt:    {final_scene_prompt}")
    print(f"corridor_prompt: {corridor_prompt}  (baseline-only, mechanism B uses scene_prompt alone)")
    print(f"mask coverage:   {(mask_np > 0.5).mean() * 100:.1f}% of frame\n")

    results: List[Dict[str, Any]] = []

    with torch.no_grad():
        ctx_scene = text_encoder([final_scene_prompt]).to(torch.bfloat16)
        ctx_scene, ctx_scene_ids = prc_txt(ctx_scene[0])
        ctx_scene = ctx_scene.unsqueeze(0).to(device_dit)
        ctx_scene_ids = ctx_scene_ids.unsqueeze(0).to(device_dit)
        num_txt_tokens = ctx_scene.shape[1]

        ctx_corridor = text_encoder([corridor_prompt]).to(torch.bfloat16)
        ctx_corridor, ctx_corridor_ids = prc_txt(ctx_corridor[0])
        ctx_corridor = ctx_corridor.unsqueeze(0).to(device_dit)
        ctx_corridor_ids = ctx_corridor_ids.unsqueeze(0).to(device_dit)

        w_lat, h_lat = w // 16, h // 16
        mask_scaled = Image.fromarray(mask_np).resize((w_lat, h_lat), Image.Resampling.BICUBIC)
        mask_flat = torch.from_numpy(np.array(mask_scaled)).float().reshape(1, -1, 1).to(device_dit)

        def fresh_noise() -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
            torch.manual_seed(args.seed)
            z_init = torch.randn(1, 128, h_lat, w_lat, device=device_dit, dtype=torch.bfloat16)
            img_tokens, img_ids = prc_img(z_init[0])
            img_tokens = img_tokens.unsqueeze(0).to(device_dit)
            img_ids = img_ids.unsqueeze(0).to(device_dit)
            timesteps = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])
            return img_tokens, img_ids, timesteps

        def decode_and_score(latent_tokens: torch.Tensor, name: str) -> Dict[str, Any]:
            z_dec = latent_tokens[0].transpose(0, 1).reshape(1, 128, h_lat, w_lat).to(device=device_aux, dtype=ae_dtype)
            x_dec = ae_model.decode(z_dec).float()
            x_arr = ((x_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            run_dir = out_root / name
            run_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(x_arr).save(run_dir / "background.png")
            save_mask_overlay(x_arr, mask_np, run_dir / "mask_overlay.png")
            ratio, inside_score, outside_score = corridor_busyness_ratio(x_arr, mask_np)
            return {"ratio": ratio, "inside_score": inside_score, "outside_score": outside_score}

        # --- (A) BASELINE: current production 2-branch velocity blend, unchanged ---
        print("[baseline] mechanism (A) 2-branch velocity blend...")
        t_run = time.time()
        img_tokens, img_ids, timesteps = fresh_noise()
        out_baseline = denoise_regional_velocity_blended(
            model=dit_model,
            img=img_tokens,
            img_ids=img_ids,
            txt_scene=ctx_scene,
            txt_scene_ids=ctx_scene_ids,
            txt_corridor=ctx_corridor,
            txt_corridor_ids=ctx_corridor_ids,
            spatial_mask=mask_flat,
            timesteps=timesteps,
            guidance=args.guidance,
            num_canvas_tokens=img_tokens.shape[1],
        )
        dur = time.time() - t_run
        scores = decode_and_score(out_baseline, "baseline")
        print(f"           busyness ratio = {scores['ratio']:.3f}  [{dur:.1f}s]")
        results.append({"run_id": "baseline (A) 2-branch", "beta": None, "latency_s": round(dur, 2), **scores})

        # --- (B) PROBE: single-branch attention-region-suppression, swept over beta ---
        for beta in betas:
            run_id = f"beta{beta}".replace(".", "p")
            print(f"[{run_id}] mechanism (B) single-branch attn-suppress (beta={beta})...")
            t_run = time.time()
            img_tokens, img_ids, timesteps = fresh_noise()
            attn_bias = build_region_suppress_attn_bias(
                mask_flat=mask_flat,
                num_txt_tokens=num_txt_tokens,
                beta=beta,
                dtype=ctx_scene.dtype,
                device=device_dit,
            )
            out_suppressed = denoise_single_branch_attn_suppressed(
                model=dit_model,
                img=img_tokens,
                img_ids=img_ids,
                txt=ctx_scene,
                txt_ids=ctx_scene_ids,
                timesteps=timesteps,
                guidance=args.guidance,
                attn_bias=attn_bias,
            )
            dur = time.time() - t_run
            scores = decode_and_score(out_suppressed, run_id)
            print(f"           busyness ratio = {scores['ratio']:.3f}  [{dur:.1f}s]")
            results.append({"run_id": f"beta={beta} (B) single-branch", "beta": beta, "latency_s": round(dur, 2), **scores})

    md = ["# FLUX.2 attention region-suppression probe\n\n"]
    md.append(f"- category: `{args.category}` | style_hint: `{style_hint}` | aspect_ratio: `{args.aspect_ratio}`\n")
    md.append(f"- scene_prompt: {final_scene_prompt}\n")
    md.append(f"- corridor_prompt (baseline only): {corridor_prompt}\n")
    md.append(f"- mask coverage: {(mask_np > 0.5).mean() * 100:.1f}% of frame\n\n")
    md.append(
        "Busyness ratio = Laplacian-variance INSIDE the corridor mask / OUTSIDE it. "
        "**Lower is better**. Compare each `beta=...` row against the `baseline (A)` "
        "row: if a beta value reaches a similar-or-lower ratio at meaningfully lower "
        "latency, mechanism (B) is a viable cheaper replacement for that beta. If every "
        "beta value plateaus well above the baseline's ratio, that is evidence of the "
        "image-to-image attention leak described in this script's module docstring, "
        "not proof the whole idea failed -- worth a follow-up probe before discarding.\n\n"
    )
    md.append("| run | beta | busyness ratio | inside score | outside score | latency (s) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for r in sorted(results, key=lambda r: (r["beta"] is None, r["ratio"])):
        beta_str = "-" if r["beta"] is None else str(r["beta"])
        md.append(
            f"| {r['run_id']} | {beta_str} | {r['ratio']:.3f} | {r['inside_score']:.1f} | "
            f"{r['outside_score']:.1f} | {r['latency_s']} |\n"
        )
    (out_root / "SUMMARY.md").write_text("".join(md), encoding="utf-8")

    print("\n" + "=" * 90)
    baseline_ratio = next(r["ratio"] for r in results if r["beta"] is None)
    best_b = min((r for r in results if r["beta"] is not None), key=lambda r: r["ratio"])
    print(f"Baseline (A) ratio: {baseline_ratio:.3f}")
    print(f"Best (B) result: {best_b['run_id']} -> ratio {best_b['ratio']:.3f}, latency {best_b['latency_s']}s")
    print(f"Full report: {out_root / 'SUMMARY.md'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
