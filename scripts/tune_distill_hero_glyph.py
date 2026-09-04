#!/usr/bin/env python3
"""
scripts/tune_distill_hero_glyph.py

==================================================================================================
TENDOO AI - DISTILL-FOCUSED HERO GLYPH OPTIMIZATION BENCHMARK
==================================================================================================

OBJECTIVE:
  Target FLUX.2-klein-4B Distilled (8 steps, ~6s) and empirically determine the exact
  Glyph Box dimensions, Font archetype, and Guidance scale needed to achieve 100% accurate
  Vietnamese Hero Title generation (specifically resolving complex diacritics like 'Ư', 'Ừ').

EXPERIMENTAL AXES:
  1. Box Envelope Scaling:
     - 768x224 (Baseline, 91pt, 672 tokens - previously missed hook strokes on 'TƯNG BỪNG')
     - 896x256 (108pt, 896 tokens, rule: 2 lines * 128px)
     - 896x288 (116pt, 1008 tokens, +30% vertical space)
     - 896x320 (116pt, 1120 tokens, maximum diacritic breathing room)
  2. Font Archetype:
     - 'bevietnam' (BeVietnamPro-Black: clean geometric sans)
     - 'gotham'    (SVN-Gotham Ultra: ultra-dense heavy strokes, bold diacritics)
  3. Embedded Guidance Scale:
     - 1.5 (Standard Distill baseline)
     - 1.8 (Tighter reference anchor without CFG edge burn)
     - 2.0 (High conditioning adherence)

EXECUTION:
  - 100% Distill-only (No Base model loaded, zero CFG overhead).
  - Takes only ~6s per run on 2x NVIDIA A30.
  - Automatically packages all results into `tune_distill_hero_glyph.zip`.
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

TEST_CASES = [
    {
        "id": "case_burger_tung_bung",
        "title": "Burger Opening (Heavy Diacritics: Ư, Ừ)",
        "hero_text": "TƯNG BỪNG\nKHAI TRƯƠNG",
        "prompt": (
            "Poster quảng cáo ẩm thực sang trọng, bánh burger bò đẫm phô mai vàng ươm bốc khói nghi ngút "
            "thơm ngon trên bàn gỗ mun, dòng chữ tiêu đề lớn 3D mạ vàng kim loại nổi bật ở phía trên, "
            "ánh sáng studio tương phản cao, phong cách điện ảnh ẩm thực sang trọng, không có watermark"
        ),
        "seed": 42,
    },
    {
        "id": "case_chong_on",
        "title": "Headphones Noise Cancelling (4 Consecutive Accents: Ố, Ồ, Ủ, Ộ)",
        "hero_text": "CHỐNG ỒN\nCHỦ ĐỘNG",
        "prompt": (
            "Poster quảng cáo tai nghe hi-end thương mại đỉnh cao, tai nghe chụp tai không dây màu bạc kim loại "
            "đặt trên bục đen phẳng ở trung tâm, dòng chữ tiêu đề lớn 3D mạ vàng kim loại đúc nổi khối sắc nét "
            "ở phía trên, ánh sáng studio mềm mại, phong cách hiện đại tối giản, không có watermark"
        ),
        "seed": 123,
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


def parse_box_size(size_str: str) -> Tuple[int, int]:
    """Parses '896x288' into (896, 288)."""
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid box size format '{size_str}'. Expected e.g. '896x288'")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Glyph Box and Guidance for FLUX.2-klein-4B Distill on complex Vietnamese text"
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
        "--boxes",
        type=str,
        nargs="+",
        default=["1024x384", "1152x448", "1280x512"],
        help="List of Glyph box sizes to test (e.g. 1024x384 1152x448 1280x512)",
    )
    parser.add_argument(
        "--fonts",
        type=str,
        nargs="+",
        default=["bevietnam", "gotham"],
        help="Font archetypes to evaluate ('bevietnam', 'gotham')",
    )
    parser.add_argument(
        "--guidance_list",
        type=float,
        nargs="+",
        default=[1.5],
        help="Embedded guidance values to test (default: 1.5)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of Distill ODE steps (default: 8)",
    )
    parser.add_argument(
        "--t_offset",
        type=float,
        default=10.0,
        help="Discrete time offset for In-Context Glyph (default: 10.0)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="case_burger_tung_bung",
        help="Which case to evaluate ('case_burger_tung_bung', 'case_chong_on', or 'all')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_tune_distill",
        help="Output directory for results",
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
            print(f"[HW] Single-GPU Setup: All models on {device_dit}")
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

    print("\n" + "=" * 95)
    print("🎯 TENDOO AI - FLUX.2-KLEIN-4B DISTILLED HERO GLYPH TUNING SUITE")
    print("=" * 95)
    print(f"  Target DiT Model   : {distill_file}")
    print(f"  ODE Steps          : {args.steps} (Targeting ~6s per image)")
    print(f"  Box Variations     : {args.boxes}")
    print(f"  Font Variations    : {args.fonts}")
    print(f"  Guidance Values    : {args.guidance_list}")
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

    # Filter Cases
    if args.case == "all":
        cases_to_run = TEST_CASES
    else:
        cases_to_run = [c for c in TEST_CASES if c["id"] == args.case]
        if not cases_to_run:
            print(f"[ERROR] Case '{args.case}' not found!")
            sys.exit(1)

    # 4. Run Grid Sweep
    parsed_boxes = [parse_box_size(b) for b in args.boxes]
    results_table: List[Dict[str, Any]] = []
    generated_files: List[Path] = []

    print("\n" + "=" * 95)
    print(f"⚡ [3/3] EXECUTING DISTILL GLYPH SCALING EXPERIMENTS ({len(cases_to_run)} Cases)")
    print("=" * 95)

    for c in cases_to_run:
        cid = c["id"]
        hero_text = c["hero_text"]
        prompt = c["prompt"]
        seed = c["seed"]

        print(f"\n▶ Testing Case: {c['title']} ('{hero_text.replace(chr(10), ' / ')}')")

        # Encode prompt once per case
        with torch.no_grad():
            txt_prompt = text_encoder([prompt]).to(device=device_dit, dtype=torch.bfloat16)
        txt_tokens, txt_ids = batched_prc_txt(txt_prompt)

        # Canvas is always 1024x1024
        canvas_w, canvas_h = 1024, 1024
        lat_w, lat_h = canvas_w // 16, canvas_h // 16

        for box_w, box_h in parsed_boxes:
            for font in args.fonts:
                # Render Glyph Bitmap
                glyph_info: GlyphInfo = render_glyph(
                    text=hero_text,
                    font_name_or_path=font,
                    target_width=box_w,
                    target_height=box_h,
                    auto_size=False,
                )
                glyph_name = f"glyph__{cid}__{box_w}x{box_h}__{font}.png"
                glyph_path = out_path / glyph_name
                glyph_info.image.save(glyph_path)

                # Encode Glyph to Reference tokens at t=10.0
                ref_tokens, ref_ids = encode_glyph_to_ref_tokens(
                    ae, glyph_info.image, t_offset=args.t_offset, device=device_ae
                )
                ref_tokens = ref_tokens.to(device_dit)
                ref_ids = ref_ids.to(device_dit)

                for guidance in args.guidance_list:
                    tag = f"{cid}__box{box_w}x{box_h}__{font}__g{guidance}__s{args.steps}"
                    print(
                        f"   • Box {box_w}x{box_h} ({glyph_info.font_size_pt}pt, {glyph_info.token_count}tok) | "
                        f"Font: {font:10s} | Guidance: {guidance:.1f} ... ",
                        end="",
                        flush=True,
                    )

                    # Exact same initial Gaussian noise
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
                            guidance=guidance,
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
                        "box": f"{box_w}x{box_h}",
                        "font": font,
                        "font_pt": glyph_info.font_size_pt,
                        "tokens": glyph_info.token_count,
                        "guidance": guidance,
                        "steps": args.steps,
                        "time_s": round(dur, 2),
                        "file": out_file.name,
                    })

    # 5. Packaging & Summary Table
    zip_filename = out_path / "tune_distill_hero_glyph.zip"
    print(f"\n[Packaging] Compressing all outputs into: {zip_filename.name}...")
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in generated_files:
            zf.write(f, arcname=f.name)
        for g in out_path.glob("glyph__*.png"):
            zf.write(g, arcname=g.name)

    print("\n" + "=" * 105)
    print("📊 DISTILL HERO GLYPH TUNING SUMMARY TABLE")
    print("=" * 105)
    print(f"{'Box Size':<11} | {'Font':<10} | {'Font Size':<9} | {'Tokens':<7} | {'Guidance':<8} | {'Time (s)':<9} | {'Output File'}")
    print("-" * 105)
    for r in results_table:
        print(
            f"{r['box']:<11} | {r['font']:<10} | {r['font_pt']}pt{' ':<5} | {r['tokens']:<7} | "
            f"{r['guidance']:<8} | {r['time_s']}s{' ':<5} | {r['file']}"
        )
    print("=" * 105)
    print(f"\n🎉 TUNING SUITE COMPLETE! Download archive directly from:")
    print(f"   👉 {zip_filename.resolve()}\n")


if __name__ == "__main__":
    main()
