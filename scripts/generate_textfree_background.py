#!/usr/bin/env python3
"""
scripts/generate_textfree_background.py

==================================================================================================
TENDOO AI - PURE TEXT-FREE BACKGROUND GENERATOR (100%-OVERLAY DIRECTION)
==================================================================================================

WHY THIS SCRIPT?
  For the "100% overlay, zero diffusion text" direction: needs a background+product image with
  ABSOLUTELY NO text/glyph reference injected at all -- pure standard text-to-image, so the HTML/
  CSS layer can carry 100% of the typography (see scripts/test_css_hero_title_styles.py, which is
  waiting on exactly this kind of clean background for a fair side-by-side comparison against the
  existing diffusion-drawn-text sample images/commercial_steps8_g1.5_seed123_576x1024.png).

  Unlike every other probe in this repo, this one does NOT call render_glyph/encode_glyph_to_ref_
  tokens at all, and does NOT pass img_cond_seq to the model -- it's the simplest possible FLUX.2
  call (prompt -> image), which is why no new architecture concepts are needed here.

USAGE (mirrors the "commercial" preset scene from probe_klein_distill_glyph.py, minus the title
sentence, plus an explicit "no text" instruction and a reserved-space cue for where the CSS hero
title will later go -- same seed/steps/guidance for the closest possible comparison):
  python scripts/generate_textfree_background.py --seed 123 --steps 8 --guidance 1.5
==================================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flux2.sampling import batched_prc_txt, denoise, get_schedule, prc_img
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import load_ae, load_flow_model

DEFAULT_PROMPT = (
    "Poster quảng cáo tai nghe hi-res cao cấp trên bục trưng bày studio tối giản, ánh sáng đèn "
    "led neon xanh tím phản chiếu sang trọng, bố cục cân đối điện ảnh, chi tiết tinh xảo, để trống "
    "khoảng không gian sạch đồng nhất ở phía trên cho tiêu đề, tuyệt đối không có chữ, không có "
    "văn bản, không có watermark"
)


def find_distill_checkpoint(base_checkpoint_dir: Path) -> Path | None:
    candidates = [
        base_checkpoint_dir / "flux-2-klein-4b.safetensors",
        base_checkpoint_dir.parent / "FLUX.2-klein-4B" / "flux-2-klein-4b.safetensors",
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4B/flux-2-klein-4b.safetensors"),
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4b.safetensors"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI pure text-free background generator (no glyph, no ref)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--canvas_w", type=int, default=576)
    parser.add_argument("--canvas_h", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--distill_model_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="/home/jovyan/persistent-data/FLUX.2-klein-base-4B")
    parser.add_argument("--output_dir", type=str, default="output_textfree_background")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit, device_ae = "cuda:0", "cuda:1"
    elif num_gpus == 1:
        device_dit, device_ae = "cuda:0", "cuda:0"
    else:
        device_dit, device_ae = "cpu", "cpu"
        print("[!] WARNING: no GPU detected, this will be very slow / for syntax-check only.")

    base_dir = Path(args.checkpoint_dir)
    model_file = Path(args.distill_model_path) if args.distill_model_path else find_distill_checkpoint(base_dir)
    if not model_file or not model_file.exists():
        print(f"[ERROR] Distilled DiT checkpoint not found around {base_dir}. Pass --distill_model_path explicitly.")
        sys.exit(1)

    print("=" * 100)
    print(" [*] TENDOO AI - PURE TEXT-FREE BACKGROUND GENERATOR (no glyph reference at all)")
    print("=" * 100)
    print(f"  Prompt   : {args.prompt}")
    print(f"  Canvas   : {args.canvas_w}x{args.canvas_h}")
    print(f"  Steps/G  : {args.steps} / {args.guidance}")
    print(f"  Seed     : {args.seed}")

    os.environ["KLEIN_4B_MODEL_PATH"] = str(model_file)
    model = load_flow_model(model_name="flux.2-klein-4b", device=device_dit).eval()
    ae = load_ae(model_name="flux.2-klein-base-4b", device=device_ae).eval()
    if args.checkpoint_dir:
        os.environ["FLUX_CHECKPOINT_DIR"] = str(args.checkpoint_dir)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_ae)

    with torch.no_grad():
        txt_emb = text_encoder([args.prompt]).to(device=device_dit, dtype=torch.bfloat16)
    txt_tokens, txt_ids = batched_prc_txt(txt_emb)

    canvas_w = (args.canvas_w // 16) * 16
    canvas_h = (args.canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    torch.manual_seed(args.seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    timesteps = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])

    print("\n[*] Denoising (NO img_cond_seq -- pure prompt-to-image, no glyph/reference at all)...")
    t0 = time.time()
    with torch.no_grad():
        # No img_cond_seq / img_cond_seq_ids passed -- deliberately the simplest possible call.
        out_tokens = denoise(
            model=model, img=img_tokens, img_ids=img_ids,
            txt=txt_tokens, txt_ids=txt_ids, timesteps=timesteps, guidance=args.guidance,
        )
        lat_2d = rearrange(out_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
        img_out = ae.decode(lat_2d.to(device_ae))
    dur = time.time() - t0

    pil_img = Image.fromarray(
        ((img_out[0].float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
    )
    out_file = out_path / f"textfree_steps{args.steps}_g{args.guidance}_seed{args.seed}_{canvas_w}x{canvas_h}.png"
    pil_img.save(out_file)
    print(f"\n[✓] Done in {dur:.2f}s -> Saved: {out_file}\n")


if __name__ == "__main__":
    main()
