#!/usr/bin/env python3
"""
scripts/probe_diffusion_mask_adherence.py

Isolated real-GPU probe for JUST the diffusion stage (FLUX.2 klein 4B DiT + AE +
Qwen3 text encoder + Regional Velocity Blending, src/tendoo/velocity_blending.py) --
run this on the actual GPU server (this dev environment has no GPU, cannot run/verify
it here). No LLM sidecar, no HTML/CSS render, no Playwright/Chromium involved: this
answers ONE question in isolation -- "does the diffusion model actually keep the
corridor-mask region clean/simple the way the corridor prompt asks it to, or does the
scene bleed through regardless of the mask?" -- and lets you experiment with 2
knobs to make the model comply more strongly if it currently doesn't, without
touching any production code path (both default to the exact current behavior):

  --mask-gamma G       M' = M ** G applied to the corridor mask before blending
                       (see velocity_blending.py's updated docstring). G < 1.0
                       pushes the mask's soft transition zone (values strictly
                       between 0 and 1) TOWARD 1.0 -- i.e. widens/strengthens how
                       far the corridor prompt's influence reaches beyond the
                       mask's hard core, without changing the mask's actual drawn
                       shape. G > 1.0 does the opposite (narrows influence to the
                       mask's core only). G = 1.0 (default) is the current,
                       unmodified production behavior.
  --corridor-guidance-boost B
                       Added ONLY to the corridor branch's CFG guidance scale (the
                       scene branch keeps the base --guidance unchanged) -- higher
                       guidance makes the model follow its text prompt more
                       literally, so boosting just the corridor branch is a direct
                       way to make it obey "clean, text-free, decluttered" more
                       strongly without affecting scene/product fidelity. B = 0.0
                       (default) is the current, unmodified production behavior.

Both accept a comma-separated LIST (e.g. --mask-gamma 1.0,0.7,0.5) to sweep
multiple values in one run and compare them side by side.

WHAT THIS SCRIPT MEASURES (so "does it adhere to the mask" isn't just eyeballed):
for each generated background, computes a Laplacian-variance "detail busyness"
score (a standard blur/detail proxy: smoother/simpler regions score LOW, detailed/
textured regions score HIGH) separately INSIDE the corridor-mask region (mask > 0.5)
and OUTSIDE it (mask <= 0.5), then reports the ratio (inside / outside). A properly
"decluttered, text-friendly" corridor should score LOWER than the surrounding scene
-- ratio well below 1.0. A ratio near or above 1.0 means the corridor is just as (or
more) visually busy as the rest of the frame -- exactly the "not respecting the
mask" symptom this script exists to detect and let you tune away via the 2 knobs
above.

The REAL corridor mask + prompts used here are the exact ones production uses
(reuses tendoo.engine.blocks.map_category_to_default_blocks + OmniBlockLayout.
generate_mask for the mask, and tendoo.demo_server's own prompt-construction
helpers for scene_prompt/corridor_prompt) -- not a synthetic stand-in, so results
here are directly representative of what a real /api/generate call would produce.

Usage (single or dual A30, mirrors demo_server.py's own device selection):
  python scripts/probe_diffusion_mask_adherence.py --category promo --style-hint auto
  python scripts/probe_diffusion_mask_adherence.py --mask-gamma 1.0,0.7,0.5 \\
      --corridor-guidance-boost 0.0,1.0,2.0 --category product_intro
Output:
  output_diffusion_mask_probe/<combo_id>/background.png       (the raw blended background)
  output_diffusion_mask_probe/<combo_id>/mask_overlay.png     (background + corridor mask tinted red, for visual review)
  output_diffusion_mask_probe/SUMMARY.md                       (busyness ratio table across all combos)
"""

from __future__ import annotations

import argparse
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

OUTPUT_DIR = PROJECT_ROOT / "output_diffusion_mask_probe"

ASPECT_DIMS: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "9:16": (576, 1024),
    "16:9": (1024, 576),
    "4:5": (816, 1024),
}

# A representative, non-trivial content set per category -- enough real blocks to
# produce a realistic (not near-empty) corridor mask, mirroring what a filled-in
# form would actually send through map_category_to_default_blocks().
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


def laplacian_variance(gray: np.ndarray) -> float:
    """3x3 Laplacian kernel convolution (hand-rolled -- no cv2/scipy dependency, so
    this script doesn't need an extra install on the GPU server). A standard detail/
    sharpness proxy: smooth, simplified regions score LOW after this filter; busy,
    detailed regions score HIGH. Used to compare the corridor-mask region against
    the rest of the frame -- see the module docstring for how to read the result."""
    g = gray.astype(np.float32)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1)
    )
    lap = lap[1:-1, 1:-1]  # drop 1px border to avoid np.roll wrap-around artifacts
    return float(lap.var())


def corridor_busyness_ratio(image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """Returns (ratio, inside_score, outside_score). `mask` is the float [0,1]
    corridor mask at full image resolution (same H,W as image_rgb)."""
    gray = (0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2])
    inside = gray[mask > 0.5]
    outside = gray[mask <= 0.5]
    # Compute the Laplacian on the full image once (edge effects only matter at the
    # image border, not at the mask boundary), then split the per-pixel |laplacian|^2
    # by region -- equivalent to variance-per-region without re-filtering twice.
    g = gray.astype(np.float32)
    lap = (
        -4.0 * g
        + np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1)
    )
    lap_sq = lap ** 2
    inside_score = float(lap_sq[mask > 0.5].mean()) if inside.size else 0.0
    outside_score = float(lap_sq[mask <= 0.5].mean()) if outside.size else 0.0
    ratio = (inside_score / outside_score) if outside_score > 1e-6 else float("nan")
    return ratio, inside_score, outside_score


def save_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    """Background image with the corridor mask tinted translucent red on top --
    lets a human directly see whether the model actually kept that region simple."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="promo", choices=list(SAMPLE_FIELDS.keys()))
    parser.add_argument("--style-hint", default="auto")
    parser.add_argument("--aspect-ratio", default="1:1", choices=list(ASPECT_DIMS.keys()))
    parser.add_argument("--scene-prompt", default="", help="Free-text scene description (optional -- falls back to a generic commercial studio scene, same as demo_server.py does)")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-gamma", default="1.0", help="Comma-separated list, e.g. '1.0,0.7,0.5'")
    parser.add_argument("--corridor-guidance-boost", default="0.0", help="Comma-separated list, e.g. '0.0,1.0,2.0'")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    gammas = _parse_float_list(args.mask_gamma)
    boosts = _parse_float_list(args.corridor_guidance_boost)
    w, h = ASPECT_DIMS[args.aspect_ratio]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("DIFFUSION-ONLY MASK ADHERENCE PROBE (real Qwen3 text encoder + FLUX.2 klein 4B DiT + AE)")
    print(f"category={args.category} style_hint={args.style_hint} aspect_ratio={args.aspect_ratio} "
          f"steps={args.steps} guidance={args.guidance} seed={args.seed}")
    print(f"sweeping mask_gamma={gammas} x corridor_guidance_boost={boosts} ({len(gammas) * len(boosts)} combos)")
    print("=" * 90)

    device_dit, device_aux = resolve_devices()
    print(f"[Device] DiT on {device_dit}, AE + Qwen3 on {device_aux}")

    from flux2 import util
    from flux2.sampling import get_schedule, prc_img, prc_txt

    t0 = time.time()
    print("Loading DiT 4B weights...")
    # "flux.2-klein-base-4b" (NOT "flux.2-klein-4b", the distilled variant) --
    # matches the real checkpoint's directory/filename on the server
    # (persistent-data/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors, per
    # AGENTS.md's documented layout). Using the wrong name here made
    # find_persistent_data_root()'s primary candidate path
    # (persistent-data/FLUX.2-klein-base-4B/flux-2-klein-4b.safetensors) not exist,
    # silently falling through to the mismatched transformer/diffusion_pytorch_model
    # .safetensors (Diffusers-format DiT) candidate instead -- confirmed 2026-09-15
    # via a real-GPU run that produced garbage/streaked output with this bug.
    dit_model = util.load_flow_model("flux.2-klein-base-4b", device=device_dit)
    dit_model.eval()
    print("Loading AutoEncoder...")
    ae_model = util.load_ae("flux.2-klein-base-4b", device=device_aux)
    ae_model.eval()
    ae_dtype = next(ae_model.parameters()).dtype
    print("Loading Qwen3 text encoder...")
    text_encoder = util.load_text_encoder("flux.2-klein-base-4b", device=device_aux)
    print(f"Models loaded in {time.time() - t0:.1f}s")

    # --- Build the REAL corridor mask + prompts, exactly as demo_server.py would ---
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
    corridor_prompt = layout.get_corridor_prompt(style_hint)

    print(f"\nscene_prompt:    {final_scene_prompt}")
    print(f"corridor_prompt: {corridor_prompt}")
    print(f"mask coverage:   {(mask_np > 0.5).mean() * 100:.1f}% of frame\n")

    with torch.no_grad():
        ctx_scene = text_encoder([final_scene_prompt]).to(torch.bfloat16)
        ctx_scene, ctx_scene_ids = prc_txt(ctx_scene[0])
        ctx_scene = ctx_scene.unsqueeze(0).to(device_dit)
        ctx_scene_ids = ctx_scene_ids.unsqueeze(0).to(device_dit)

        ctx_corridor = text_encoder([corridor_prompt]).to(torch.bfloat16)
        ctx_corridor, ctx_corridor_ids = prc_txt(ctx_corridor[0])
        ctx_corridor = ctx_corridor.unsqueeze(0).to(device_dit)
        ctx_corridor_ids = ctx_corridor_ids.unsqueeze(0).to(device_dit)

        w_lat, h_lat = w // 16, h // 16
        mask_scaled = Image.fromarray(mask_np).resize((w_lat, h_lat), Image.Resampling.BICUBIC)
        mask_flat = torch.from_numpy(np.array(mask_scaled)).float().reshape(1, -1, 1).to(device_dit)

        results: List[Dict[str, Any]] = []
        for gamma in gammas:
            for boost in boosts:
                combo_id = f"gamma{gamma}_boost{boost}".replace(".", "p")
                print(f"[{combo_id}] running {args.steps}-step denoise...")
                t_combo = time.time()

                torch.manual_seed(args.seed)
                z_init = torch.randn(1, 128, h_lat, w_lat, device=device_dit, dtype=torch.bfloat16)
                img_tokens, img_ids = prc_img(z_init[0])
                img_tokens = img_tokens.unsqueeze(0).to(device_dit)
                img_ids = img_ids.unsqueeze(0).to(device_dit)
                timesteps = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])

                out_blended = denoise_regional_velocity_blended(
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
                    mask_gamma=gamma,
                    corridor_guidance_boost=boost,
                )

                z_dec = out_blended[0].transpose(0, 1).reshape(1, 128, h_lat, w_lat).to(device=device_aux, dtype=ae_dtype)
                x_dec = ae_model.decode(z_dec).float()
                x_arr = ((x_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()

                combo_dir = out_root / combo_id
                combo_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(x_arr).save(combo_dir / "background.png")
                save_mask_overlay(x_arr, mask_np, combo_dir / "mask_overlay.png")

                ratio, inside_score, outside_score = corridor_busyness_ratio(x_arr, mask_np)
                dur = time.time() - t_combo
                print(f"           busyness ratio (inside/outside) = {ratio:.3f}  "
                      f"(inside={inside_score:.1f}, outside={outside_score:.1f})  [{dur:.1f}s]")
                results.append({
                    "combo_id": combo_id, "mask_gamma": gamma, "corridor_guidance_boost": boost,
                    "ratio": ratio, "inside_score": inside_score, "outside_score": outside_score,
                    "latency_s": round(dur, 2),
                })

    md = ["# Diffusion mask-adherence probe\n\n"]
    md.append(f"- category: `{args.category}` | style_hint: `{style_hint}` | aspect_ratio: `{args.aspect_ratio}`\n")
    md.append(f"- scene_prompt: {final_scene_prompt}\n")
    md.append(f"- corridor_prompt: {corridor_prompt}\n")
    md.append(f"- mask coverage: {(mask_np > 0.5).mean() * 100:.1f}% of frame\n\n")
    md.append(
        "Busyness ratio = Laplacian-variance INSIDE the corridor mask / OUTSIDE it. "
        "**Lower is better** (the corridor should look simpler/cleaner than the rest "
        "of the scene); a ratio near or above 1.0 means the mask is not being "
        "respected -- try a lower --mask-gamma or a higher --corridor-guidance-boost.\n\n"
    )
    md.append("| combo | mask_gamma | corridor_guidance_boost | busyness ratio | latency (s) |\n")
    md.append("|---|---|---|---|---|\n")
    for r in sorted(results, key=lambda r: r["ratio"]):
        md.append(f"| `{r['combo_id']}` | {r['mask_gamma']} | {r['corridor_guidance_boost']} | {r['ratio']:.3f} | {r['latency_s']} |\n")
    (out_root / "SUMMARY.md").write_text("".join(md), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"Done. Lowest (best) busyness ratio: {min(results, key=lambda r: r['ratio'])['combo_id']}")
    print(f"Full report: {out_root / 'SUMMARY.md'}")
    print(f"Images: {out_root}/<combo_id>/background.png, mask_overlay.png")
    print("=" * 90)


if __name__ == "__main__":
    main()
