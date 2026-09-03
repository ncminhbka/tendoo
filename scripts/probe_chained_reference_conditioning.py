#!/usr/bin/env python3
"""
scripts/probe_chained_reference_conditioning.py

==================================================================================================
TENDOO AI - CHAINED MULTI-MODAL REFERENCE CONDITIONING PROBE (DIRECTION 4)
==================================================================================================

WHY THIS EXPERIMENT?
  1. The user verified that Chained Reference Conditioning WORKS WELL:
     - Pass 1 renders Title + Product into a coherent base poster.
     - Pass 2 feeds Pass 1's poster at t=20.0, and feeds Subtitle glyph at t=10.0.
  2. USER'S NEW SPECIFICATION:
     - Subtitle should NOT be large 3D embossed like the Title; it should have an elegant, refined,
       minimalist typography style (e.g. silver-white clean text, not giant extruded 3D blocks).
     - Step 1 (Pass 1):
       * Title glyph @ t=10.0
       * Reference Product Image (Headphone or Shoes) @ t=20.0
       -> Generates Step 1 Base Poster containing Title + Product.
     - Step 2 (Pass 2):
       * Long Subtitle glyph @ t=10.0
       * Step 1 Output Poster @ t=20.0
       * Adjusted prompt for refined, non-embossed commercial ad subtitle.
       -> Generates Final Chained Completed Poster with Product + Title + Long Subtitle!

EXECUTION REQUIREMENTS:
  - 2x NVIDIA A30 (DiT on cuda:0, VAE/Qwen3 on cuda:1) or Single GPU.
  - Zero HTML output (AGENTS.md Rule 28 compliant: PNGs + ASCII summary only).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import load_ae, load_flow_model
from tendoo.glyph_engine import GlyphInfo, render_glyph

# ==================================================================================================
# 1. PRESETS (HEADPHONES & SHOES)
# ==================================================================================================

PRESETS = {
    "headphones": {
        "prod_image": "images/ref_prod_02.png",
        "title_text": "ÂM THANH\nĐỈNH CAO",
        "subtitle_text": "Chống ồn chủ động đỉnh cao\nvà thời lượng pin vượt trội suốt 40 giờ",
        "step1_prompt": (
            "Poster quảng cáo thương mại cho chiếc tai nghe chụp tai cao cấp màu đen sang trọng đặt ở "
            "trung tâm, nền studio công nghệ ánh sáng tương phản cao, dòng chữ tiêu đề lớn 3D dập nổi mạ "
            "vàng kim loại sắc nét ở phía trên, bố cục sạch sẽ chuyên nghiệp, không có watermark"
        ),
        "step2_prompt": (
            "Poster quảng cáo thương mại kế thừa toàn bộ phong cách bố cục và chiếc tai nghe của poster "
            "tham chiếu, bổ sung dòng chữ phụ màu trắng bạc thanh lịch tinh tế sắc nét ở phía dưới, "
            "phong cách typography tối giản hiện đại, bố cục sang trọng, không có watermark"
        ),
    },
    "shoes": {
        "prod_image": "images/shoes.jpeg",
        "title_text": "BỨT PHÁ\nTỐC ĐỘ",
        "subtitle_text": "Thiết kế đệm khí êm ái linh hoạt\ncho mọi cung đường bứt phá",
        "step1_prompt": (
            "Poster quảng cáo thương mại cho đôi giày thể thao cao cấp đặt ở vị trí trung tâm, "
            "nền studio ánh sáng điện ảnh tương phản cao sang trọng, dòng chữ tiêu đề lớn 3D dập nổi "
            "mạ vàng kim loại sắc nét ở phía trên, bố cục sạch sẽ chuyên nghiệp, không có watermark"
        ),
        "step2_prompt": (
            "Poster quảng cáo thương mại kế thừa toàn bộ phong cách bố cục và đôi giày thể thao của poster "
            "tham chiếu, bổ sung dòng chữ phụ màu trắng bạc thanh lịch tinh tế sắc nét ở phía dưới, "
            "phong cách typography thể thao tối giản hiện đại, bố cục sang trọng, không có watermark"
        ),
    },
}

CANVAS = (576, 1024)  # 9:16 target
DEFAULT_SEEDS = [42, 123, 777]


# ==================================================================================================
# 2. ENCODING FUNCTIONS
# ==================================================================================================

def encode_image_to_ref_tokens(
    ae: AutoEncoder, img: Image.Image, t_offset: float, device: str | torch.device, target_size: Tuple[int, int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes any image (glyph, product, or full poster) into canonical 4D RoPE tokens at local origin."""
    img_rgb = img.convert("RGB")
    if target_size is not None:
        img_rgb = img_rgb.resize(target_size, Image.Resampling.LANCZOS)

    arr = np.array(img_rgb).astype(np.float32) / 127.5 - 1.0
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
# 3. RUNNER ORCHESTRATION
# ==================================================================================================

def run_chained_conditioning_probe(
    preset: str = "headphones",
    prod_image: str | None = None,
    title_text: str | None = None,
    subtitle_text: str | None = None,
    step1_prompt: str | None = None,
    step2_prompt: str | None = None,
    seeds: List[int] = DEFAULT_SEEDS,
    font: str = "bevietnam",
    output_dir: str = "output_chained_reference_conditioning",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    title_w: int = 512,
    title_h: int = 224,
    subtitle_w: int = 832,
    subtitle_h: int = 224,
    t_title: float = 10.0,
    t_prod: float = 20.0,
    t_subtitle: float = 10.0,
    t_poster: float = 20.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resolve preset configurations
    cfg = PRESETS.get(preset, PRESETS["headphones"])
    prod_img_path = prod_image or cfg["prod_image"]
    title = title_text or cfg["title_text"]
    subtitle = subtitle_text or cfg["subtitle_text"]
    p_step1 = step1_prompt or cfg["step1_prompt"]
    p_step2 = step2_prompt or cfg["step2_prompt"]

    print("=" * 100)
    print(" [*] TENDOO AI - CHAINED MULTI-MODAL REFERENCE CONDITIONING PROBE (DIRECTION 4)")
    print("=" * 100)
    print(f"  Preset     : {preset.upper()}")
    print(f"  Product Img: {prod_img_path} (Step 1 @ t={t_prod})")
    print(f"  Title      : \"{title.replace(chr(10), ' ')}\" (Step 1 @ t={t_title}, {title_w}x{title_h}px)")
    print(f"  Subtitle   : \"{subtitle.replace(chr(10), ' ')}\" (Step 2 @ t={t_subtitle}, {subtitle_w}x{subtitle_h}px)")
    print(f"  Poster Ref : Step 1 output poster chained into Step 2 @ t={t_poster}")
    print(f"  Canvas     : {CANVAS[0]}x{CANVAS[1]} (9:16 target)")
    print(f"  Seeds      : {seeds}")
    print(f"  Steps / CFG: {num_steps} steps | CFG guidance = {guidance:.1f}")

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print(f"[*] Dual GPU Mode: DiT on {device_dit} | VAE & Qwen3 on {device_ae}")
    else:
        device_dit = device_ae = device_te = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[*] Single Device Mode: {device_dit}")

    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    print("\n[1/4] Loading FLUX.2 Klein 4B Base models (AE + DiT + Qwen3)...")
    ae = load_ae(model_name, device=device_ae)
    model = load_flow_model(model_name, device=device_dit)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    print("\n[2/4] Encoding prompts via Qwen3-4B-FP8...")
    print(f"  [Step 1 Prompt]: {p_step1[:90]}...")
    print(f"  [Step 2 Prompt]: {p_step2[:90]}...")
    with torch.no_grad():
        txt_step1, txt_ids_step1 = batched_prc_txt(text_encoder(["", p_step1]))
        txt_step1, txt_ids_step1 = txt_step1.to(device_dit), txt_ids_step1.to(device_dit)

        txt_step2, txt_ids_step2 = batched_prc_txt(text_encoder(["", p_step2]))
        txt_step2, txt_ids_step2 = txt_step2.to(device_dit), txt_ids_step2.to(device_dit)

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

    print("\n[3/4] Preparing Glyph Bitmaps & Product Image...")
    # Render Title glyph (bold prominent header, default 512x224)
    title_info = render_glyph(
        text=title, font_name_or_path=font, auto_size=False,
        target_width=title_w, target_height=title_h,
    )
    # Render Subtitle glyph (generous width 832px to guarantee zero edge truncation)
    subtitle_info = render_glyph(
        text=subtitle, font_name_or_path=font, auto_size=False,
        target_width=subtitle_w, target_height=subtitle_h,
    )
    print(f"  [Title Glyph]   : {title_info.width_px}x{title_info.height_px}px {title_info.font_size_pt}pt {title_info.token_count}tok :: {title_info.lines}")
    print(f"  [Subtitle Glyph]: {subtitle_info.width_px}x{subtitle_info.height_px}px {subtitle_info.font_size_pt}pt {subtitle_info.token_count}tok :: {subtitle_info.lines}")
    title_info.image.save(out_path / "title_glyph.png")
    subtitle_info.image.save(out_path / "subtitle_glyph.png")

    # Anti-Truncation Verification: Ensure NO glyph pixels touch the left/right borders
    for name, info in [("Title", title_info), ("Subtitle", subtitle_info)]:
        arr = np.array(info.image)
        l_col = int(np.count_nonzero(arr[:, 0]))
        r_col = int(np.count_nonzero(arr[:, -1]))
        if l_col > 0 or r_col > 0:
            print(f"  [!] CRITICAL WARNING: {name} glyph text touches border! (col0={l_col}, col_end={r_col})")
        else:
            print(f"  [✓] {name} glyph: Verified 100% clean borders (col0=0, col_end=0, safe padding intact).")

    # Encode glyphs
    title_ref_tokens, title_ref_ids = encode_image_to_ref_tokens(ae, title_info.image, t_title, device_ae)
    subtitle_ref_tokens, subtitle_ref_ids = encode_image_to_ref_tokens(ae, subtitle_info.image, t_subtitle, device_ae)
    title_ref_tokens, title_ref_ids = title_ref_tokens.to(device_dit), title_ref_ids.to(device_dit)
    subtitle_ref_tokens, subtitle_ref_ids = subtitle_ref_tokens.to(device_dit), subtitle_ref_ids.to(device_dit)

    # Encode Product image (768x768 for optimal 2304 token density)
    prod_pil = Image.open(prod_img_path).convert("RGB")
    prod_ref_tokens, prod_ref_ids = encode_image_to_ref_tokens(ae, prod_pil, t_prod, device_ae, target_size=(768, 768))
    prod_ref_tokens, prod_ref_ids = prod_ref_tokens.to(device_dit), prod_ref_ids.to(device_dit)
    print(f"  [Product Image] : {prod_pil.size} -> 768x768 -> {prod_ref_tokens.shape[1]} tokens (t={t_prod})")

    # Step 1 Combined Reference: Title Glyph (@ t_title) + Product Image (@ t_prod)
    step1_ref_tokens = torch.cat([title_ref_tokens, prod_ref_tokens], dim=1)
    step1_ref_ids = torch.cat([title_ref_ids, prod_ref_ids], dim=1)
    print(f"  [Step 1 Input]  : {step1_ref_tokens.shape[1]} total ref tokens (Title {title_ref_tokens.shape[1]}tok + Product {prod_ref_tokens.shape[1]}tok)")

    print(f"\n[4/4] Executing Chained Multi-Modal Conditioning across {len(seeds)} seed(s)...\n")
    results_summary: List[Dict[str, Any]] = []

    for idx, seed in enumerate(seeds, 1):
        print("=" * 80)
        print(f"▶️  SEED [{idx}/{len(seeds)}]: {seed}")
        print("=" * 80)
        t_start = time.time()

        # ------------------------------------------------------------------------------------------
        # STEP 1: GENERATE BASE POSTER (Title @ t=10 + Product @ t=20)
        # ------------------------------------------------------------------------------------------
        print(f"  [Step 1] Generating Base Poster with Title + Product ({num_steps} steps Euler ODE)...")
        t_s1 = time.time()
        torch.manual_seed(seed)
        z_init_1 = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens_1, img_ids_1 = prc_img(z_init_1[0])
        img_tokens_1 = img_tokens_1.unsqueeze(0).to(device_dit)
        img_ids_1 = img_ids_1.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens_1.shape[1])

        with torch.no_grad():
            pass1_latent_tokens = denoise_cfg(
                model=model,
                img=img_tokens_1,
                img_ids=img_ids_1,
                txt=txt_step1,
                txt_ids=txt_ids_step1,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=step1_ref_tokens,
                img_cond_seq_ids=step1_ref_ids,
            )

            # Decode Step 1 Base Poster
            pass1_lat_2d = rearrange(pass1_latent_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
            pass1_img = ae.decode(pass1_lat_2d.to(device_ae))
            pass1_pil = Image.fromarray(((pass1_img[0].float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).byte().cpu().numpy())
            pass1_file = out_path / f"seed{seed}_step1_title_product_poster.png"
            pass1_pil.save(pass1_file)
            dur_s1 = time.time() - t_s1
            print(f"  [Step 1] Completed in {dur_s1:.1f}s -> Saved: {pass1_file.name}")

        # ------------------------------------------------------------------------------------------
        # STEP 2: CHAINED CONDITIONING (Step 1 Poster @ t_poster + Long Subtitle @ t_subtitle)
        # ------------------------------------------------------------------------------------------
        print(f"  [Step 2] Encoding Step 1 Poster as Reference Image at t={t_poster}...")
        pass1_ref_tokens, pass1_ref_ids = encode_image_to_ref_tokens(ae, pass1_pil, t_poster, device_ae)
        pass1_ref_tokens, pass1_ref_ids = pass1_ref_tokens.to(device_dit), pass1_ref_ids.to(device_dit)
        print(f"  [Step 2] Poster tokens: {pass1_ref_tokens.shape[1]}tok | Subtitle tokens: {subtitle_ref_tokens.shape[1]}tok")

        # Step 2 Combined Reference: Subtitle Glyph (@ t_subtitle) + Step 1 Poster (@ t_poster)
        step2_ref_tokens = torch.cat([subtitle_ref_tokens, pass1_ref_tokens], dim=1)
        step2_ref_ids = torch.cat([subtitle_ref_ids, pass1_ref_ids], dim=1)

        print(f"  [Step 2] Denoising canvas with Chained References ({num_steps} steps Euler ODE)...")
        t_s2 = time.time()
        torch.manual_seed(seed + 5000)  # independent seed for Step 2 canvas synthesis
        z_init_2 = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens_2, img_ids_2 = prc_img(z_init_2[0])
        img_tokens_2 = img_tokens_2.unsqueeze(0).to(device_dit)
        img_ids_2 = img_ids_2.unsqueeze(0).to(device_dit)

        with torch.no_grad():
            pass2_latent_tokens = denoise_cfg(
                model=model,
                img=img_tokens_2,
                img_ids=img_ids_2,
                txt=txt_step2,
                txt_ids=txt_ids_step2,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=step2_ref_tokens,
                img_cond_seq_ids=step2_ref_ids,
            )

            # Decode Step 2 Chained Completed Poster
            pass2_lat_2d = rearrange(pass2_latent_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
            pass2_img = ae.decode(pass2_lat_2d.to(device_ae))
            pass2_pil = Image.fromarray(((pass2_img[0].float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).byte().cpu().numpy())
            pass2_file = out_path / f"seed{seed}_step2_completed_chained_poster.png"
            pass2_pil.save(pass2_file)
            dur_s2 = time.time() - t_s2
            print(f"  [Step 2] Completed in {dur_s2:.1f}s -> Saved: {pass2_file.name}")

        # ------------------------------------------------------------------------------------------
        # SIDE-BY-SIDE COMPARISON
        # ------------------------------------------------------------------------------------------
        comp_w = canvas_w * 2 + 30
        comp_h = canvas_h + 60
        comp_img = Image.new("RGB", (comp_w, comp_h), color=(25, 25, 25))
        draw = ImageDraw.Draw(comp_img)

        comp_img.paste(pass1_pil, (10, 50))
        comp_img.paste(pass2_pil, (canvas_w + 20, 50))

        draw.text((20, 15), f"STEP 1: Title + Product (Ref for Step 2)", fill=(255, 215, 0))
        draw.text((canvas_w + 30, 15), f"STEP 2: Completed (Poster @ t={t_poster} + Subtitle @ t={t_subtitle})", fill=(0, 255, 127))

        comp_file = out_path / f"seed{seed}_chained_comparison.png"
        comp_img.save(comp_file)
        dur_total = time.time() - t_start
        print(f"  [+] Comparison image: {comp_file.name} (Total time: {dur_total:.1f}s)\n")

        results_summary.append({
            "seed": seed,
            "step1_file": str(pass1_file),
            "step2_file": str(pass2_file),
            "comparison_file": str(comp_file),
            "dur_s1": round(dur_s1, 1),
            "dur_s2": round(dur_s2, 1),
            "dur_total": round(dur_total, 1),
        })

    # ==============================================================================================
    # ASCII SUMMARY TABLE (Rule 28 Compliant: NO HTML)
    # ==============================================================================================
    print("=" * 100)
    print(f"{'SEED':<8} | {'STEP 1 (TITLE + PRODUCT)':<35} | {'STEP 2 (COMPLETED POSTER)':<35} | {'TIME (S1 / S2 / TOT)':<20}")
    print("-" * 100)
    for res in results_summary:
        time_str = f"{res['dur_s1']}s / {res['dur_s2']}s / {res['dur_total']}s"
        print(f"{res['seed']:<8} | {Path(res['step1_file']).name:<35} | {Path(res['step2_file']).name:<35} | {time_str:<20}")
    print("=" * 100)
    print(f"\n[✓] Chained Reference Conditioning probe finished successfully. Outputs in: {out_path.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Chained Multi-Modal Reference Conditioning Probe (Direction 4)")
    parser.add_argument("--preset", type=str, default="headphones", choices=["headphones", "shoes"],
                        help="Preset configuration (default: headphones)")
    parser.add_argument("--prod_image", type=str, default=None, help="Custom path to product image")
    parser.add_argument("--title_text", type=str, default=None, help="Custom Title text")
    parser.add_argument("--subtitle_text", type=str, default=None, help="Custom Subtitle text")
    parser.add_argument("--step1_prompt", type=str, default=None, help="Custom prompt for Step 1")
    parser.add_argument("--step2_prompt", type=str, default=None, help="Custom prompt for Step 2")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to run")
    parser.add_argument("--output_dir", type=str, default="output_chained_reference_conditioning", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")
    parser.add_argument("--title_w", type=int, default=512, help="Title envelope width in px (default: 512)")
    parser.add_argument("--title_h", type=int, default=224, help="Title envelope height in px (default: 224)")
    parser.add_argument("--subtitle_w", type=int, default=832, help="Subtitle envelope width in px (default: 832)")
    parser.add_argument("--subtitle_h", type=int, default=224, help="Subtitle envelope height in px (default: 224)")
    parser.add_argument("--t_title", type=float, default=10.0, help="Time offset for Title glyph in Step 1 (default: 10.0)")
    parser.add_argument("--t_prod", type=float, default=20.0, help="Time offset for Product image in Step 1 (default: 20.0)")
    parser.add_argument("--t_subtitle", type=float, default=10.0, help="Time offset for Subtitle glyph in Step 2 (default: 10.0)")
    parser.add_argument("--t_poster", type=float, default=20.0, help="Time offset for Step 1 Poster in Step 2 (default: 20.0)")

    args = parser.parse_args()
    run_chained_conditioning_probe(
        preset=args.preset,
        prod_image=args.prod_image,
        title_text=args.title_text,
        subtitle_text=args.subtitle_text,
        step1_prompt=args.step1_prompt,
        step2_prompt=args.step2_prompt,
        seeds=args.seeds,
        font=args.font,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps,
        guidance=args.guidance,
        title_w=args.title_w,
        title_h=args.title_h,
        subtitle_w=args.subtitle_w,
        subtitle_h=args.subtitle_h,
        t_title=args.t_title,
        t_prod=args.t_prod,
        t_subtitle=args.t_subtitle,
        t_poster=args.t_poster,
    )


if __name__ == "__main__":
    main()
