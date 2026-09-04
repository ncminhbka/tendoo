#!/usr/bin/env python3
"""
scripts/compare_distill_vs_base.py

==================================================================================================
TENDOO AI - EMPIRICAL COMPARISON SUITE: DISTILL (8-12s) VS BASE (15-25s) UNDER IDENTICAL PROMPTS
==================================================================================================

OBJECTIVE:
  Empirically test whether FLUX.2-klein-4B Distilled (8-12 steps, guidance=1.5) or
  FLUX.2-klein-base-4B (15-25 steps, CFG=4.0) renders Vietnamese 3D Hero Titles accurately (100%)
  under the EXACT SAME commercial prompts, seeds, glyph bitmaps, and resolutions.

KEY TEST CRITERIA:
  1. Identical prompt content across both strategies.
  2. Identical initial noise (same seed).
  3. Controlled step sweep:
     - Distill: 8 steps, 12 steps.
     - Base: 15 steps, 20 steps, 25 steps.
  4. Precise runtime measurement (seconds per image) on 2x NVIDIA A30.
  5. Automatic packaging of all results into `compare_distill_vs_base.zip`.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import batched_prc_txt, denoise, denoise_cfg, get_schedule, prc_img
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import load_ae, load_flow_model
from tendoo.glyph_engine import GlyphInfo, render_glyph

# Default 4 Benchmark Cases (Uncensored, realistic commercial prompts)
BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "id": "case1_burger",
        "title": "Burger Grand Opening",
        "hero_text": "TƯNG BỪNG\nKHAI TRƯƠNG",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo ẩm thực sang trọng, bánh burger bò đẫm phô mai vàng ươm bốc khói nghi ngút "
            "thơm ngon trên bàn gỗ mun, dòng chữ tiêu đề lớn 3D mạ vàng kim loại nổi bật ở phía trên, "
            "ánh sáng studio tương phản cao, phong cách điện ảnh ẩm thực sang trọng, không có watermark"
        ),
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "seed": 42,
    },
    {
        "id": "case2_pet_spa",
        "title": "Pet Grooming & Spa",
        "hero_text": "PET GROOMING\n& SPA",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo dịch vụ thú cưng cao cấp, chú chó Poodle trắng xù dễ thương đang tắm "
            "trong bồn với bọt xà phòng bồng bềnh sạch sẽ, dòng chữ tiêu đề lớn 3D phát sáng nổi bật ở "
            "phía trên, ánh sáng salon ấm áp lung linh, phong cách hiện đại, không có watermark"
        ),
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "seed": 123,
    },
    {
        "id": "case3_recruitment",
        "title": "Tech Talent Recruitment",
        "hero_text": "TUYỂN DỤNG\nAI DESIGNER",
        "font": "bevietnam",
        "prompt": (
            "Poster tuyển dụng công nghệ cao, văn phòng mở hiện đại với dàn màn hình máy tính phát sáng "
            "và các kỹ sư đang làm việc, dòng chữ tiêu đề lớn 3D kim loại chrome phản chiếu ánh sáng sắc nét "
            "ở phía trên, phong cách điện ảnh hiện đại tối giản, không có watermark"
        ),
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "seed": 777,
    },
    {
        "id": "case4_menu",
        "title": "Artisan Culinary Menu",
        "hero_text": "THỰC ĐƠN ĐẶC BIỆT",
        "font": "playfair",
        "prompt": (
            "Poster thực đơn nhà hàng ẩm thực sang trọng, bàn gỗ mun tối màu với đĩa bít tết bò nướng xèo xèo "
            "và ly cocktail cam sả mát lạnh, ánh nến lung linh mờ ảo, dòng chữ tiêu đề lớn 3D khắc gỗ mạ vàng đồng "
            "cổ sắc nét tinh xảo ở phía trên, phong cách điện ảnh nghệ thuật ẩm thực, không có watermark"
        ),
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "seed": 999,
    },
]


def encode_glyph_to_ref_tokens(
    ae: AutoEncoder, img: Image.Image, t_offset: float, device: str | torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes a glyph image into canonical 4D RoPE tokens at local origin (0, 0)."""
    img_rgb = img.convert("RGB")
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


def find_checkpoint(base_checkpoint_dir: Path, filename: str) -> Path | None:
    """Searches for a specific DiT safetensors file across common server paths."""
    candidates = [
        base_checkpoint_dir / filename,
        base_checkpoint_dir.parent / "FLUX.2-klein-4B" / filename,
        base_checkpoint_dir.parent / "FLUX.2-klein-base-4B" / filename,
        Path("/home/jovyan/persistent-data/FLUX.2-klein-base-4B") / filename,
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4B") / filename,
        Path("/home/jovyan/persistent-data") / filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Empirical comparison between FLUX.2-klein-4B Distill and Base under identical prompts"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
        help="Base directory containing text_encoder and VAE",
    )
    parser.add_argument(
        "--distill_model_path",
        type=str,
        default=None,
        help="Explicit path to flux-2-klein-4b.safetensors (Distilled)",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Explicit path to flux-2-klein-base-4b.safetensors (Base)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["both", "distill", "base"],
        default="both",
        help="Which model strategies to evaluate (both, distill, base)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        help="Which case to run ('all', 'case1_burger', 'case2_pet_spa', 'case3_recruitment', 'case4_menu')",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text to override test cases (if specified, runs 1 custom case)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt to override test cases",
    )
    parser.add_argument(
        "--font",
        type=str,
        default="bevietnam",
        help="Font family for custom text ('bevietnam', 'playfair', 'oswald', 'montserrat')",
    )
    parser.add_argument(
        "--distill_steps",
        type=int,
        nargs="+",
        default=[8, 12],
        help="ODE steps list for Distill strategy (default: 8 12)",
    )
    parser.add_argument(
        "--base_steps",
        type=int,
        nargs="+",
        default=[15, 20, 25],
        help="ODE steps list for Base strategy (default: 15 20 25)",
    )
    parser.add_argument(
        "--distill_guidance",
        type=float,
        default=1.5,
        help="Embedded guidance for Distilled model (default: 1.5)",
    )
    parser.add_argument(
        "--base_guidance",
        type=float,
        default=4.0,
        help="CFG guidance multiplier for Base model (default: 4.0)",
    )
    parser.add_argument(
        "--t_offset",
        type=float,
        default=10.0,
        help="Pretrained discrete time-offset for Glyph tokens (default: 10.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_compare_distill_vs_base",
        help="Directory to save generated comparison images and zip archive",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Device Configuration
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            device_dit = torch.device("cuda:0")
            device_ae = torch.device("cuda:1")
            print(f"[HW] Multi-GPU Mode: DiT on {device_dit} | VAE & TextEncoder on {device_ae}")
        else:
            device_dit = torch.device("cuda:0")
            device_ae = torch.device("cuda:0")
            print(f"[HW] Single-GPU Mode: All components on {device_dit}")
    else:
        print("[ERROR] CUDA is required to run FLUX.2 inference!")
        sys.exit(1)

    # 2. Select Test Cases
    if args.text and args.prompt:
        cases_to_run = [
            {
                "id": "custom_case",
                "title": "Custom Benchmark Case",
                "hero_text": args.text.replace("\\n", "\n"),
                "font": args.font,
                "prompt": args.prompt,
                "canvas_w": 1024,
                "canvas_h": 1024,
                "box_w": 768,
                "box_h": 224,
                "seed": 42,
            }
        ]
    elif args.case != "all":
        matched = [c for c in BENCHMARK_CASES if c["id"] == args.case]
        if not matched:
            print(f"[ERROR] Case '{args.case}' not found in benchmark suite!")
            sys.exit(1)
        cases_to_run = matched
    else:
        cases_to_run = BENCHMARK_CASES

    # 3. Load Shared Encoders: VAE and Text Encoder (Qwen3-4B-FP8)
    base_dir = Path(args.checkpoint_dir)
    os.environ["FLUX_CHECKPOINT_DIR"] = str(base_dir)

    print("\n" + "=" * 80)
    print("🚀 [1/3] LOADING SHARED ENCODERS: VAE & QWEN3-4B-FP8")
    print("=" * 80)
    ae = load_ae(model_name="flux.2-klein-base-4b", device=device_ae)
    ae.eval()
    text_encoder = load_qwen3_embedder(variant="4B", device=device_ae)

    # Pre-render Glyph bitmaps and extract prompt embeddings
    case_data: Dict[str, Dict[str, Any]] = {}
    print("\nPreparing Glyphs and Token representations for test cases:")
    for c in cases_to_run:
        cid = c["id"]
        hero_text = c["hero_text"]
        font = c["font"]
        prompt = c["prompt"]

        glyph_info: GlyphInfo = render_glyph(
            text=hero_text,
            font_name_or_path=font,
            target_width=c["box_w"],
            target_height=c["box_h"],
            auto_size=False,
        )
        glyph_file = out_path / f"glyph_{cid}.png"
        glyph_info.image.save(glyph_file)

        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae, glyph_info.image, t_offset=args.t_offset, device=device_ae)
        ref_tokens = ref_tokens.to(device_dit)
        ref_ids = ref_ids.to(device_dit)

        # Distill prompt context (1x prompt)
        with torch.no_grad():
            txt_prompt = text_encoder([prompt]).to(device=device_dit, dtype=torch.bfloat16)
        txt_tokens_distill, txt_ids_distill = batched_prc_txt(txt_prompt)

        # Base prompt context (2x batch: uncond + cond)
        with torch.no_grad():
            txt_empty = text_encoder([""]).to(device=device_dit, dtype=torch.bfloat16)
            txt_cfg = torch.cat([txt_empty, txt_prompt], dim=0)
        txt_tokens_base, txt_ids_base = batched_prc_txt(txt_cfg)

        case_data[cid] = {
            "config": c,
            "ref_tokens": ref_tokens,
            "ref_ids": ref_ids,
            "txt_tokens_distill": txt_tokens_distill,
            "txt_ids_distill": txt_ids_distill,
            "txt_tokens_base": txt_tokens_base,
            "txt_ids_base": txt_ids_base,
        }
        print(f"  [✓] Case '{cid}': Glyph rasterized -> {glyph_file.name} (Font size: {glyph_info.font_size}pt)")

    results_table: List[Dict[str, Any]] = []
    generated_files: List[Path] = []

    # ==============================================================================================
    # STRATEGY 1: FLUX.2-KLEIN-4B DISTILLED (Fast guidance-distilled, no CFG)
    # ==============================================================================================
    if args.strategy in ["both", "distill"]:
        print("\n" + "=" * 80)
        print("⚡ [2/3] EVALUATING STRATEGY 1: FLUX.2-KLEIN-4B DISTILLED")
        print("=" * 80)

        distill_file = Path(args.distill_model_path) if args.distill_model_path else find_checkpoint(base_dir, "flux-2-klein-4b.safetensors")
        if not distill_file or not distill_file.exists():
            print(f"[WARNING] Distilled DiT checkpoint not found at: {distill_file}. Skipping Strategy 1.")
        else:
            print(f"Loading Distilled DiT weights from: {distill_file}")
            os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_file)
            distill_model = load_flow_model(model_name="flux.2-klein-4b", device=device_dit)
            distill_model.eval()

            for c in cases_to_run:
                cid = c["id"]
                data = case_data[cid]
                w, h = (c["canvas_w"] // 16) * 16, (c["canvas_h"] // 16) * 16
                lat_w, lat_h = w // 16, h // 16
                seed = c["seed"]

                for steps in args.distill_steps:
                    tag = f"{cid}__distill__s{steps}__g{args.distill_guidance}"
                    print(f"\n  ▶ Running Distill: {c['title']} | Steps={steps} | Guidance={args.distill_guidance} | Seed={seed}...")

                    torch.manual_seed(seed)
                    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
                    img_tokens, img_ids = prc_img(z_init[0])
                    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
                    img_ids = img_ids.unsqueeze(0).to(device_dit)

                    timesteps = get_schedule(num_steps=steps, image_seq_len=img_tokens.shape[1])

                    t0 = time.time()
                    with torch.no_grad():
                        out_tokens = denoise(
                            model=distill_model,
                            img=img_tokens,
                            img_ids=img_ids,
                            txt=data["txt_tokens_distill"],
                            txt_ids=data["txt_ids_distill"],
                            timesteps=timesteps,
                            guidance=args.distill_guidance,
                            img_cond_seq=data["ref_tokens"],
                            img_cond_seq_ids=data["ref_ids"],
                        )
                        lat_2d = rearrange(out_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
                        img_out = ae.decode(lat_2d.to(device_ae))
                    dur = time.time() - t0

                    pil_img = Image.fromarray(
                        ((img_out[0].float().clamp(-1, 1) + 1.0) * 127.5)
                        .permute(1, 2, 0)
                        .byte()
                        .cpu()
                        .numpy()
                    )
                    out_file = out_path / f"{tag}.png"
                    pil_img.save(out_file)
                    generated_files.append(out_file)
                    print(f"     [✓] Done in {dur:.2f}s -> Saved: {out_file.name}")

                    results_table.append({
                        "case": cid,
                        "model": "Distill-4B",
                        "steps": steps,
                        "guidance": args.distill_guidance,
                        "time_s": round(dur, 2),
                        "file": out_file.name,
                    })

            # Cleanup Distill DiT to release VRAM
            print("\nUnloading Distill DiT from VRAM...")
            del distill_model
            gc.collect()
            torch.cuda.empty_cache()

    # ==============================================================================================
    # STRATEGY 2: FLUX.2-KLEIN-BASE-4B WITH REDUCED STEPS (15, 20, 25 steps, CFG=4.0)
    # ==============================================================================================
    if args.strategy in ["both", "base"]:
        print("\n" + "=" * 80)
        print("🛡️ [3/3] EVALUATING STRATEGY 2: FLUX.2-KLEIN-BASE-4B WITH REDUCED STEPS")
        print("=" * 80)

        base_file = Path(args.base_model_path) if args.base_model_path else find_checkpoint(base_dir, "flux-2-klein-base-4b.safetensors")
        if not base_file or not base_file.exists():
            print(f"[WARNING] Base DiT checkpoint not found at: {base_file}. Skipping Strategy 2.")
        else:
            print(f"Loading Base DiT weights from: {base_file}")
            os.environ["KLEIN_4B_BASE_MODEL_PATH"] = str(base_file)
            base_model = load_flow_model(model_name="flux.2-klein-base-4b", device=device_dit)
            base_model.eval()

            for c in cases_to_run:
                cid = c["id"]
                data = case_data[cid]
                w, h = (c["canvas_w"] // 16) * 16, (c["canvas_h"] // 16) * 16
                lat_w, lat_h = w // 16, h // 16
                seed = c["seed"]

                for steps in args.base_steps:
                    tag = f"{cid}__base__s{steps}__g{args.base_guidance}"
                    print(f"\n  ▶ Running Base: {c['title']} | Steps={steps} | CFG={args.base_guidance} | Seed={seed}...")

                    # EXACT SAME INITIAL NOISE FOR FAIR HEAD-TO-HEAD COMPARISON
                    torch.manual_seed(seed)
                    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
                    img_tokens, img_ids = prc_img(z_init[0])
                    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
                    img_ids = img_ids.unsqueeze(0).to(device_dit)

                    timesteps = get_schedule(num_steps=steps, image_seq_len=img_tokens.shape[1])

                    t0 = time.time()
                    with torch.no_grad():
                        out_tokens = denoise_cfg(
                            model=base_model,
                            img=img_tokens,
                            img_ids=img_ids,
                            txt=data["txt_tokens_base"],
                            txt_ids=data["txt_ids_base"],
                            timesteps=timesteps,
                            guidance=args.base_guidance,
                            img_cond_seq=data["ref_tokens"],
                            img_cond_seq_ids=data["ref_ids"],
                        )
                        lat_2d = rearrange(out_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
                        img_out = ae.decode(lat_2d.to(device_ae))
                    dur = time.time() - t0

                    pil_img = Image.fromarray(
                        ((img_out[0].float().clamp(-1, 1) + 1.0) * 127.5)
                        .permute(1, 2, 0)
                        .byte()
                        .cpu()
                        .numpy()
                    )
                    out_file = out_path / f"{tag}.png"
                    pil_img.save(out_file)
                    generated_files.append(out_file)
                    print(f"     [✓] Done in {dur:.2f}s -> Saved: {out_file.name}")

                    results_table.append({
                        "case": cid,
                        "model": "Base-4B",
                        "steps": steps,
                        "guidance": args.base_guidance,
                        "time_s": round(dur, 2),
                        "file": out_file.name,
                    })

            print("\nUnloading Base DiT from VRAM...")
            del base_model
            gc.collect()
            torch.cuda.empty_cache()

    # ==============================================================================================
    # 4. PACKAGE ARCHIVE AND PRINT COMPARISON TABLE
    # ==============================================================================================
    zip_filename = out_path / "compare_distill_vs_base.zip"
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in generated_files:
            zf.write(f, arcname=f.name)
        for c in cases_to_run:
            g_file = out_path / f"glyph_{c['id']}.png"
            if g_file.exists():
                zf.write(g_file, arcname=g_file.name)

    print("\n" + "=" * 90)
    print(f"📊 EMPIRICAL COMPARISON SUMMARY: DISTILL VS BASE (HEAD-TO-HEAD)")
    print("=" * 90)
    print(f"{'Case ID':<18} | {'Model':<12} | {'Steps':<6} | {'Guidance':<8} | {'Latency (s)':<11} | {'Output Image'}")
    print("-" * 90)
    for r in results_table:
        print(f"{r['case']:<18} | {r['model']:<12} | {r['steps']:<6} | {r['guidance']:<8} | {r['time_s']:<11} | {r['file']}")
    print("=" * 90)
    print(f"\n📦 All {len(generated_files)} comparison images packaged into: {zip_filename.resolve()}")
    print("   Download this zip file to your local machine for side-by-side visual inspection!\n")


if __name__ == "__main__":
    main()
