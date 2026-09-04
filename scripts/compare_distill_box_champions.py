#!/usr/bin/env python3
"""
scripts/compare_distill_box_champions.py

==================================================================================================
TENDOO AI - CHAMPIONSHIP MATRIX: 1152x448 VS 1280x512 GLYPH ENVELOPES ON FLUX.2-DISTILL (8 STEPS)
==================================================================================================

OBJECTIVE:
  Decide the single winning Glyph Envelope for Tendoo AI production by testing:
    - Champion A: 1152x448 (2016 tokens, ~151pt, 21% faster & lighter)
    - Champion B: 1280x512 (2560 tokens, ~168pt, maximum resolution ceiling)
  across:
    1. Diverse commercial ad copy (Hooks 'Ư/Ơ', 4 consecutive accents, % numbers, luxury terms)
    2. Multiple real-world output canvas aspect ratios:
       - 9:16 Portrait   (576 x 1024) - Mobile Story / TikTok / Standee
       - 1:1  Square     (1024 x 1024) - E-commerce / Social Feed
       - 16:9 Landscape  (1024 x 576)  - Web Banner / Billboard Cover

EXECUTION:
  - 100% FLUX.2-klein-4B Distilled (8 steps, guidance=1.5, t=10.0).
  - Takes ~3s to ~6s per image on 2x NVIDIA A30.
  - Automatically packages all results into `box_champions_compare.zip`.
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
from flux2.sampling import batched_prc_txt, denoise, get_schedule, prc_img
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import load_ae, load_flow_model
from tendoo.glyph_engine import GlyphInfo, render_glyph

CHAMPION_CASES = [
    {
        "id": "case1_burger",
        "title": "Burger Opening (Hooks: Ư, Ừ)",
        "hero_text": "TƯNG BỪNG\nKHAI TRƯƠNG",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo ẩm thực sang trọng, bánh burger bò đẫm phô mai vàng ươm bốc khói nghi ngút "
            "thơm ngon trên bàn gỗ mun, dòng chữ tiêu đề lớn 3D mạ vàng kim loại nổi bật ở phía trên, "
            "ánh sáng studio tương phản cao, phong cách điện ảnh ẩm thực sang trọng, không có watermark"
        ),
        "seed": 42,
    },
    {
        "id": "case2_anc",
        "title": "ANC Headphones (4 Accents: Ố, Ồ, Ủ, Ộ)",
        "hero_text": "CHỐNG ỒN\nCHỦ ĐỘNG",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo tai nghe hi-end thương mại đỉnh cao, tai nghe chụp tai không dây màu bạc kim loại "
            "đặt trên bục đen phẳng ở trung tâm, dòng chữ tiêu đề lớn 3D mạ vàng kim loại đúc nổi khối sắc nét "
            "ở phía trên, ánh sáng studio tương phản mềm mại, phong cách hiện đại tối giản, không có watermark"
        ),
        "seed": 123,
    },
    {
        "id": "case3_sale",
        "title": "Flash Sale (Numbers, %, Clustered Vowels)",
        "hero_text": "SIÊU SALE 50%\nDUY NHẤT HÔM NAY",
        "font": "bevietnam",
        "prompt": (
            "Poster flash sale thương mại điện tử bùng nổ, các hộp quà tặng màu đỏ và dải ruy băng vàng "
            "bay lơ lửng xung quanh, dòng chữ tiêu đề lớn 3D mạ vàng kim loại đúc khối rực rỡ ở phía trên, "
            "ánh sáng studio tương phản cao phong cách lễ hội mua sắm, không có watermark"
        ),
        "seed": 777,
    },
    {
        "id": "case4_luxury",
        "title": "Fine Dining (Rare Accents: Ư, Ợ, Nặng)",
        "hero_text": "THỰC ĐƠN\nTHƯỢNG HẠNG",
        "font": "bevietnam",
        "prompt": (
            "Poster ẩm thực nhà hàng khách sạn 5 sao, đĩa bò wagyu nướng xèo xèo kèm ly rượu vang đỏ "
            "lung linh dưới ánh nến trên bàn đá cẩm thạch tối màu, dòng chữ tiêu đề lớn 3D khắc kim loại "
            "mạ vàng đồng cổ sắc nét ở phía trên, phong cách điện ảnh ẩm thực cao cấp, không có watermark"
        ),
        "seed": 999,
    },
]

BOX_CANDIDATES = [
    {"label": "box1152x448", "w": 1152, "h": 448},
    {"label": "box1280x512", "w": 1280, "h": 512},
]

ASPECT_RATIOS = [
    {"label": "portrait_9x16", "w": 576, "h": 1024},
    {"label": "square_1x1", "w": 1024, "h": 1024},
    {"label": "landscape_16x9", "w": 1024, "h": 576},
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


def find_distill_checkpoint(base_checkpoint_dir: Path) -> Path | None:
    """Searches for flux-2-klein-4b.safetensors across standard paths."""
    candidates = [
        base_checkpoint_dir / "flux-2-klein-4b.safetensors",
        base_checkpoint_dir.parent / "FLUX.2-klein-4B" / "flux-2-klein-4b.safetensors",
        base_checkpoint_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4B/flux-2-klein-4b.safetensors"),
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4b.safetensors"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: 1152x448 vs 1280x512 Glyph Boxes across diverse ad formats on Distill"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
        help="Directory containing VAE and Text Encoder",
    )
    parser.add_argument(
        "--distill_model_path",
        type=str,
        default=None,
        help="Explicit path to flux-2-klein-4b.safetensors",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        help="Which case to evaluate ('all', 'case1_burger', 'case2_anc', 'case3_sale', 'case4_luxury')",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        nargs="+",
        default=["portrait_9x16", "square_1x1", "landscape_16x9"],
        help="Aspect ratios to evaluate ('portrait_9x16', 'square_1x1', 'landscape_16x9')",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of Distill ODE steps (default: 8)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.5,
        help="Embedded guidance value (default: 1.5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_box_champions",
        help="Directory to save generated outputs and ZIP",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Device Allocation
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            device_dit = torch.device("cuda:0")
            device_ae = torch.device("cuda:1")
            print(f"[HW] Dual-GPU Setup: DiT on {device_dit} | VAE & TextEncoder on {device_ae}")
        else:
            device_dit = torch.device("cuda:0")
            device_ae = torch.device("cuda:0")
            print(f"[HW] Single-GPU Setup: All components on {device_dit}")
    else:
        print("[ERROR] CUDA GPU is required!")
        sys.exit(1)

    # 2. Checkpoint Resolution
    base_dir = Path(args.checkpoint_dir)
    distill_file = Path(args.distill_model_path) if args.distill_model_path else find_distill_checkpoint(base_dir)
    if not distill_file or not distill_file.exists():
        print(f"[ERROR] Distilled DiT checkpoint not found around: {base_dir}")
        print("  Please specify with --distill_model_path <path_to_flux-2-klein-4b.safetensors>")
        sys.exit(1)

    print("\n" + "=" * 105)
    print("🏆 TENDOO AI - 1152x448 VS 1280x512 GLYPH BOX CHAMPIONSHIP")
    print("=" * 105)
    print(f"  Target DiT Model   : {distill_file}")
    print(f"  ODE Steps          : {args.steps} (Guidance: {args.guidance})")
    print(f"  Output Canvas List : {args.ratios}")
    print(f"  Output Directory   : {out_path.resolve()}")

    # 3. Load Models
    print("\n[1/3] Loading Distilled DiT (4B)...")
    os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_file)
    model = load_flow_model(model_name="flux.2-klein-4b", device=device_dit)
    model.eval()

    print("[2/3] Loading VAE and Text Encoder (Qwen3-4B-FP8)...")
    os.environ["FLUX_CHECKPOINT_DIR"] = str(base_dir)
    ae = load_ae(model_name="flux.2-klein-base-4b", device=device_ae)
    ae.eval()
    text_encoder = load_qwen3_embedder(variant="4B", device=device_ae)

    # Filter Cases and Aspect Ratios
    if args.case == "all":
        selected_cases = CHAMPION_CASES
    else:
        selected_cases = [c for c in CHAMPION_CASES if c["id"] == args.case]
        if not selected_cases:
            print(f"[ERROR] Case '{args.case}' not found!")
            sys.exit(1)

    selected_ratios = [r for r in ASPECT_RATIOS if r["label"] in args.ratios]
    if not selected_ratios:
        print(f"[ERROR] None of the specified ratios {args.ratios} match known aspect ratios!")
        sys.exit(1)

    total_runs = len(selected_cases) * len(BOX_CANDIDATES) * len(selected_ratios)
    print(f"\n[3/3] Executing Championship Matrix ({len(selected_cases)} cases x 2 boxes x {len(selected_ratios)} ratios = {total_runs} images)...")

    results_table: List[Dict[str, Any]] = []
    generated_files: List[Path] = []
    current_run = 0

    for c in selected_cases:
        cid = c["id"]
        hero_text = c["hero_text"]
        font = c["font"]
        prompt = c["prompt"]
        seed = c["seed"]

        print(f"\n" + "-" * 90)
        print(f"▶ [{cid.upper()}] '{hero_text.replace(chr(10), ' / ')}' | Font: {font}")

        # Text prompt encoding (done once per case)
        with torch.no_grad():
            txt_prompt = text_encoder([prompt]).to(device=device_dit, dtype=torch.bfloat16)
        txt_tokens, txt_ids = batched_prc_txt(txt_prompt)

        for b in BOX_CANDIDATES:
            box_label = b["label"]
            box_w, box_h = b["w"], b["h"]

            # Render Glyph Bitmap
            glyph_info: GlyphInfo = render_glyph(
                text=hero_text,
                font_name_or_path=font,
                target_width=box_w,
                target_height=box_h,
                auto_size=False,
            )
            glyph_name = f"glyph__{cid}__{box_label}.png"
            glyph_path = out_path / glyph_name
            if not glyph_path.exists():
                glyph_info.image.save(glyph_path)

            # Encode Glyph to Reference tokens at t=10.0
            ref_tokens, ref_ids = encode_glyph_to_ref_tokens(
                ae, glyph_info.image, t_offset=10.0, device=device_ae
            )
            ref_tokens = ref_tokens.to(device_dit)
            ref_ids = ref_ids.to(device_dit)

            for r in selected_ratios:
                current_run += 1
                ratio_label = r["label"]
                canvas_w, canvas_h = r["w"], r["h"]
                lat_w, lat_h = canvas_w // 16, canvas_h // 16

                tag = f"{cid}__{box_label}__{ratio_label}"
                print(
                    f"   [{current_run:02d}/{total_runs:02d}] {box_label} ({glyph_info.font_size_pt}pt, {glyph_info.token_count}tok) "
                    f"-> Canvas {canvas_w}x{canvas_h} ({ratio_label}) ... ",
                    end="",
                    flush=True,
                )

                # Initialize Gaussian Noise from same seed
                torch.manual_seed(seed)
                z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
                img_tokens, img_ids = prc_img(z_init[0])
                img_tokens = img_tokens.unsqueeze(0).to(device_dit)
                img_ids = img_ids.unsqueeze(0).to(device_dit)

                timesteps = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])

                t0 = time.time()
                with torch.no_grad():
                    out_tokens = denoise(
                        model=model,
                        img=img_tokens,
                        img_ids=img_ids,
                        txt=txt_tokens,
                        txt_ids=txt_ids,
                        timesteps=timesteps,
                        guidance=args.guidance,
                        img_cond_seq=ref_tokens,
                        img_cond_seq_ids=ref_ids,
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
                print(f"Done in {dur:.2f}s -> {out_file.name}")

                results_table.append({
                    "case": cid,
                    "box": box_label,
                    "font_pt": glyph_info.font_size_pt,
                    "tokens": glyph_info.token_count,
                    "canvas": f"{canvas_w}x{canvas_h}",
                    "ratio": ratio_label,
                    "time_s": round(dur, 2),
                    "file": out_file.name,
                })

    # 4. Packaging & Summary Table
    zip_filename = out_path / "box_champions_compare.zip"
    print(f"\n[Packaging] Compressing all outputs into: {zip_filename.name}...")
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in generated_files:
            zf.write(f, arcname=f.name)
        for g in out_path.glob("glyph__*.png"):
            zf.write(g, arcname=g.name)

    print("\n" + "=" * 115)
    print("📊 1152x448 VS 1280x512 CHAMPIONSHIP COMPARISON SUMMARY")
    print("=" * 115)
    print(f"{'Case ID':<15} | {'Box Size':<12} | {'Font':<7} | {'Tokens':<7} | {'Canvas':<11} | {'Ratio':<16} | {'Time (s)':<9} | {'File'}")
    print("-" * 115)
    for r in results_table:
        print(
            f"{r['case']:<15} | {r['box']:<12} | {r['font_pt']}pt{' ':<3} | {r['tokens']:<7} | "
            f"{r['canvas']:<11} | {r['ratio']:<16} | {r['time_s']}s{' ':<5} | {r['file']}"
        )
    print("=" * 115)
    print(f"\n🎉 ALL RUNS FINISHED! Download archive directly from:")
    print(f"   👉 {zip_filename.resolve()}\n")


if __name__ == "__main__":
    main()
