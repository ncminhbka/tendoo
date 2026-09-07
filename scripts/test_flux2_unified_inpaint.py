#!/usr/bin/env python3
"""
scripts/test_flux2_unified_inpaint.py

==================================================================================================
TENDOO AI -- HƯỚNG A: FLUX.2 KLEIN UNIFIED INPAINT (MASKED-LATENT + REFERENCE-LATENT)
==================================================================================================

BẢN CHẤT KỸ THUẬT:
  Tái hiện workflow "FLUX Klein Unified Image Editing" (ComfyUI / runcomfy.com) trực tiếp trên
  mã nguồn Python thuần của FLUX.2 klein 4B (Apache 2.0).

  Khắc phục triệt để các hạn chế của các thử nghiệm trước:
  1. KHÔNG đưa ảnh mask đen trắng vào Reference (tránh lỗi Attention bỏ qua reference).
  2. Kênh Reference (t=10.0): Chứa TOÀN BỘ ẢNH CẢNH GỐC ĐẦY ĐỦ (đèn lồng, lều quán, ánh sáng)
     để Attention Heads làm mốc tham chiếu phong cách và ánh sáng môi trường.
  3. Kênh Canvas (t=0.0): Khởi tạo x_1 theo Flow Matching. Tại mỗi bước t_prev, vùng ngoài mask
     được đồng bộ hóa nghiêm ngặt về đúng trạng thái Flow Matching của ảnh gốc:
     z_known(t_prev) = (1 - t_prev) * z_orig + t_prev * noise_fixed.
  4. Vùng trong mask: Tự do khử nhiễu theo inpaint prompt (bầu trời thanh bình, không dây đèn).
  5. Vùng biên giới: Được làm mềm qua `soften_region_mask` (average pool 3x3) để triệt tiêu seam.

TỐC ĐỘ:
  - Mặc định chạy bản DISTILL 4B (8 steps, guidance=1.5) -> ~2.0 - 2.5s trên 2x NVIDIA A30.
  - Tùy chọn bản BASE 4B (True CFG, 20-50 steps) qua `--model-name flux.2-klein-base-4b`.

USAGE TRÊN SERVER:
  # 1. Chạy với ảnh có sẵn (ví dụ ảnh chợ đêm Trung Thu đã sinh trước đó):
  python scripts/test_flux2_unified_inpaint.py \
      --input-image "output_flux2_reserve_test/baseline.png" \
      --steps 8 \
      --guidance 1.5

  # 2. Tự sinh ảnh baseline trước nếu chưa có, rồi inpaint ngay lập tức:
  python scripts/test_flux2_unified_inpaint.py \
      --generate-first \
      --steps 8 \
      --guidance 1.5

OUTPUT:
  - <out-dir>/01_original.png
  - <out-dir>/02_mask_preview.png
  - <out-dir>/03_inpainted.png
  - <out-dir>/04_side_by_side.png
  - <out-dir>/run_info.json
==================================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from flux2 import util
from flux2.masked_generation import (
    build_reference_tokens_from_latent,
    build_region_token_mask,
    build_region_token_mask_multi,
    denoise_unified_inpaint,
    soften_region_mask,
)
from flux2.sampling import (
    batched_prc_txt,
    default_images_prep,
    denoise as denoise_baseline,
    get_schedule,
    prc_img,
    prc_txt,
)

DEFAULT_SCENE_PROMPT = (
    "A bustling nighttime Mid-Autumn Festival street market, rows of traditional wooden stalls "
    "decorated with paper lanterns and string lights hanging overhead across the street, tall "
    "trees strung with glowing lanterns reaching up toward the sky, a large full moon glowing "
    "above, warm golden lighting, richly detailed festive decorations filling the whole scene, "
    "people walking, cinematic lighting"
)

DEFAULT_INPAINT_PROMPT = (
    "A quiet peaceful night sky with a soft glowing full moon, gentle wisps of cloud, warm golden "
    "amber gradient from dark indigo to warm amber, clean empty open copy space, no lanterns, "
    "no hanging strings, no decorations, no stalls, no people"
)


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein Unified Inpainting Test")
    parser.add_argument("--input-image", type=str, default=None,
                        help="Path to an existing image to inpaint. If missing or None, can generate first.")
    parser.add_argument("--generate-first", action="store_true",
                        help="Generate a baseline image first before inpainting.")
    parser.add_argument("--scene-prompt", type=str, default=DEFAULT_SCENE_PROMPT,
                        help="Prompt for generating the baseline scene (if --generate-first is used).")
    parser.add_argument("--inpaint-prompt", type=str, default=DEFAULT_INPAINT_PROMPT,
                        help="Prompt for the inpaint region (clearing lanterns/adding quiet sky).")
    parser.add_argument("--model-name", type=str, default="flux.2-klein-4b",
                        choices=["flux.2-klein-4b", "flux.2-klein-base-4b"],
                        help="Model variant: 'flux.2-klein-4b' (distill, fast ~2s) or 'flux.2-klein-base-4b' (base, CFG).")
    parser.add_argument("--distill-model-path", type=str, default=None,
                        help="Direct path to flux-2-klein-4b.safetensors if located outside persistent-data.")
    parser.add_argument("--width", type=int, default=576, help="Canvas width (must be multiple of 16)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height (must be multiple of 16)")
    parser.add_argument("--region", type=str, default="top_sky",
                        choices=["top_sky", "title_two_line", "custom"],
                        help="Inpaint region layout ('top_sky' for Mid-Autumn lantern clearing, 'title_two_line', 'custom').")
    parser.add_argument("--row-frac", type=float, nargs=2, default=[0.05, 0.52],
                        help="Custom row fraction range [y_start, y_end] (only if --region custom).")
    parser.add_argument("--col-frac", type=float, nargs=2, default=[0.08, 0.92],
                        help="Custom col fraction range [x_start, x_end] (only if --region custom).")
    parser.add_argument("--steps", type=int, default=8, help="ODE steps (default 8 for distill, 50 for base).")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale (default 1.5 for distill, 4.0 for base).")
    parser.add_argument("--feather-iters", type=int, default=3, help="Feathering iterations for soft mask border (3 ~ 48px).")
    parser.add_argument("--ref-t-offset", type=float, default=10.0, help="Time offset for ReferenceLatent (default 10.0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Primary GPU for DiT.")
    parser.add_argument("--out-dir", type=str, default="output_flux2_unified_inpaint", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 FLUX.2 KLEIN UNIFIED INPAINTING BENCHMARK (HƯỚNG A)")
    print("=" * 80)
    print(f"  Model variant     : {args.model_name}")
    print(f"  Canvas resolution : {args.width}x{args.height}")
    print(f"  Inpaint region    : {args.region}")
    print(f"  ODE steps         : {args.steps}")
    print(f"  Guidance          : {args.guidance}")
    print(f"  Seed              : {args.seed}")
    print(f"  Output directory  : {out_dir}")
    print("-" * 80)

    # 1. Device mapping (Multi-GPU split if available)
    device = args.device
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        aux_device = "cuda:1"
        print(f"  [Device] Multi-GPU detected: DiT on {device}, AE & TextEncoder on {aux_device}")
    elif num_gpus == 1:
        aux_device = "cuda:0"
        print(f"  [Device] Single GPU detected: All components on {device}")
    else:
        device = aux_device = "cpu"
        print("  [Device] Running on CPU (Warning: Very slow)")

    # 2. Checkpoint resolution
    if args.distill_model_path:
        os.environ["KLEIN_4B_MODEL_PATH"] = str(args.distill_model_path)
    elif args.model_name == "flux.2-klein-4b" and "KLEIN_4B_MODEL_PATH" not in os.environ:
        # Check standard persistent-data locations
        pdata = Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B"))
        distill_cand = pdata / "flux-2-klein-4b.safetensors"
        if distill_cand.exists():
            os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_cand)
            print(f"  [Model] Found Distill weights: {distill_cand}")

    # 3. Load Models
    print("\n⏳ [1/4] Loading models...")
    is_base = (args.model_name == "flux.2-klein-base-4b")
    model = util.load_flow_model(args.model_name, device=device)
    model.eval()

    ae = util.load_ae(args.model_name, device=aux_device)
    ae.eval()
    ae_dtype = next(ae.parameters()).dtype

    text_encoder = util.load_text_encoder(args.model_name, device=aux_device)

    # 4. Latent dimensions & Mask construction
    h_lat = args.height // 16
    w_lat = args.width // 16
    C = 128

    if args.region == "top_sky":
        rects = [((0.04, 0.52), (0.08, 0.92))]
        region_mask_flat = build_region_token_mask(h_lat, w_lat, (0.04, 0.52), (0.08, 0.92))
    elif args.region == "title_two_line":
        rects = [((0.05, 0.28), (0.08, 0.92)), ((0.28, 0.52), (0.16, 0.84))]
        region_mask_flat = build_region_token_mask_multi(h_lat, w_lat, rects)
    else:  # custom
        rects = [(tuple(args.row_frac), tuple(args.col_frac))]
        region_mask_flat = build_region_token_mask(h_lat, w_lat, tuple(args.row_frac), tuple(args.col_frac))

    soft_mask = soften_region_mask(region_mask_flat, h_lat, w_lat, feather_iters=args.feather_iters)

    # 5. Obtain or Generate Baseline Image
    input_img_path = None
    if args.input_image and Path(args.input_image).exists():
        input_img_path = Path(args.input_image)
        print(f"\n🖼️ [2/4] Using existing input image: {input_img_path}")
        orig_pil = Image.open(input_img_path).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    elif args.generate_first or not input_img_path:
        print(f"\n🎨 [2/4] Generating baseline scene image first...")
        torch.manual_seed(args.seed)
        z_init = torch.randn(1, C, h_lat, w_lat, device=device, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device)
        img_ids = img_ids.unsqueeze(0).to(device)

        timesteps_base = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])
        with torch.no_grad():
            if is_base:
                ctx_raw = text_encoder(["", args.scene_prompt]).to(torch.bfloat16)
                ctx, ctx_ids = batched_prc_txt(ctx_raw)
                ctx, ctx_ids = ctx.to(device), ctx_ids.to(device)
                out_base = denoise_baseline(model, img_tokens, img_ids, ctx, ctx_ids, timesteps_base, guidance=args.guidance)
            else:
                ctx_raw = text_encoder([args.scene_prompt]).to(torch.bfloat16)
                ctx, ctx_ids = prc_txt(ctx_raw[0])
                ctx = ctx.unsqueeze(0).to(device)
                ctx_ids = ctx_ids.unsqueeze(0).to(device)
                out_base = denoise_baseline(model, img_tokens, img_ids, ctx, ctx_ids, timesteps_base, guidance=args.guidance)

            # Decode baseline
            z_dec = out_base[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
            x_dec = ae.decode(z_dec).float()
            x_dec = ((x_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            orig_pil = Image.fromarray(x_dec)
            print("  [✓] Baseline scene generated successfully.")

    # Save 01_original.png
    orig_save_path = out_dir / "01_original.png"
    orig_pil.save(orig_save_path)

    # Generate and save 02_mask_preview.png (Visual overlay)
    mask_2d_np = soft_mask.reshape(h_lat, w_lat).cpu().numpy()
    mask_full = Image.fromarray((mask_2d_np * 255).astype(np.uint8)).resize((args.width, args.height), Image.Resampling.BILINEAR)
    
    preview_img = orig_pil.copy().convert("RGBA")
    red_overlay = Image.new("RGBA", (args.width, args.height), (255, 50, 50, 120))
    preview_img.paste(red_overlay, (0, 0), mask_full)
    preview_img.convert("RGB").save(out_dir / "02_mask_preview.png")

    # 6. Encode Original Image into z_orig and ReferenceLatent
    print("\n🧬 [3/4] Encoding original image into Reference & Canvas latents...")
    with torch.no_grad():
        prep_tensor = default_images_prep(orig_pil)
        if isinstance(prep_tensor, list):
            prep_tensor = prep_tensor[0]
        prep_tensor = prep_tensor.unsqueeze(0).to(device=aux_device, dtype=ae_dtype)
        z_orig = ae.encode(prep_tensor)[0].to(device=device, dtype=torch.bfloat16)  # (C, h_lat, w_lat)

        # Reference tokens at t=10.0 (The full unmasked scene)
        ref_tokens, ref_ids = build_reference_tokens_from_latent(z_orig, t_offset=args.ref_t_offset, device=device)

        # Canvas tokens at t=0.0
        z_orig_tokens, img_ids = prc_img(z_orig)
        z_orig_tokens = z_orig_tokens.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        img_ids = img_ids.unsqueeze(0).to(device=device)

        # Encode Inpaint Prompt
        print(f"  Inpaint Prompt: \"{args.inpaint_prompt}\"")
        if is_base:
            ctx_raw = text_encoder(["", args.inpaint_prompt]).to(torch.bfloat16)
            ctx_inp, ctx_ids_inp = batched_prc_txt(ctx_raw)
            ctx_inp, ctx_ids_inp = ctx_inp.to(device), ctx_ids_inp.to(device)
        else:
            ctx_raw = text_encoder([args.inpaint_prompt]).to(torch.bfloat16)
            ctx_inp, ctx_ids_inp = prc_txt(ctx_raw[0])
            ctx_inp = ctx_inp.unsqueeze(0).to(device)
            ctx_ids_inp = ctx_ids_inp.unsqueeze(0).to(device)

    # 7. Run Unified Inpaint Denoise Loop
    print(f"\n⚡ [4/4] Running Unified Inpaint (Steps={args.steps}, Guidance={args.guidance})...")
    torch.manual_seed(args.seed)
    noise_fixed = torch.randn_like(z_orig_tokens)
    timesteps = get_schedule(num_steps=args.steps, image_seq_len=z_orig_tokens.shape[1])

    t0 = time.time()
    with torch.no_grad():
        out_tokens = denoise_unified_inpaint(
            model=model,
            z_orig_tokens=z_orig_tokens,
            img_ids=img_ids,
            ref_tokens=ref_tokens,
            ref_ids=ref_ids,
            soft_mask=soft_mask,
            txt=ctx_inp,
            txt_ids=ctx_ids_inp,
            timesteps=timesteps,
            guidance=args.guidance,
            noise_fixed=noise_fixed,
            is_cfg=is_base,
        )
        dur = time.time() - t0

        # Decode Result
        z_res = out_tokens[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
        x_res = ae.decode(z_res).float()
        x_res = ((x_res[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        inpainted_pil = Image.fromarray(x_res)

    inpaint_save_path = out_dir / "03_inpainted.png"
    inpainted_pil.save(inpaint_save_path)
    print(f"  [✓] Inpainting finished in {dur:.2f}s! Saved: {inpaint_save_path}")

    # 8. Create Side-by-Side Comparison
    w, h = orig_pil.size
    side_by_side = Image.new("RGB", (w * 3, h), (20, 20, 20))
    side_by_side.paste(orig_pil, (0, 0))
    side_by_side.paste(preview_img.convert("RGB"), (w, 0))
    side_by_side.paste(inpainted_pil, (w * 2, 0))

    # Add header labels
    draw = ImageDraw.Draw(side_by_side)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([(10, 10), (220, 50)], fill=(0, 0, 0, 180))
    draw.text((20, 15), "1. ORIGINAL", fill=(255, 255, 255), font=font)

    draw.rectangle([(w + 10, 10), (w + 260, 50)], fill=(0, 0, 0, 180))
    draw.text((w + 20, 15), "2. MASK REGION", fill=(255, 100, 100), font=font)

    draw.rectangle([(w * 2 + 10, 10), (w * 2 + 320, 50)], fill=(0, 0, 0, 180))
    draw.text((w * 2 + 20, 15), f"3. INPAINTED ({dur:.1f}s)", fill=(100, 255, 100), font=font)

    comparison_path = out_dir / "04_side_by_side.png"
    side_by_side.save(comparison_path)
    print(f"  [✓] Comparison saved: {comparison_path}")

    # 9. Save Metadata JSON
    run_info = {
        "model_name": args.model_name,
        "is_base": is_base,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "resolution": f"{args.width}x{args.height}",
        "region": args.region,
        "rects": rects,
        "feather_iters": args.feather_iters,
        "ref_t_offset": args.ref_t_offset,
        "inpaint_prompt": args.inpaint_prompt,
        "time_seconds": round(dur, 3),
        "outputs": {
            "original": str(orig_save_path.name),
            "mask_preview": "02_mask_preview.png",
            "inpainted": str(inpaint_save_path.name),
            "comparison": str(comparison_path.name),
        }
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"🎉 HOÀN THÀNH KIỂM THỬ HƯỚNG A TRONG {dur:.2f}s!")
    print(f"   Ảnh so sánh trực quan: {comparison_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
