#!/usr/bin/env python3
"""
scripts/generate_distill_4cases.py

==================================================================================================
TENDOO AI - BATCH 4-CASE DIVERSE COMMERCIAL GENERATOR (FLUX.2-KLEIN-4B DISTILLED)
==================================================================================================

OBJECTIVE:
  Generate 4 photorealistic baseline commercial posters with integrated 3D Hero Titles
  using FLUX.2-klein-4B Distilled (8 steps, guidance=1.5, t=10.0 In-Context Glyph) on 2x A30 GPU:
    1. Case 1: Burger Grand Opening ("TƯNG BỪNG KHAI TRƯƠNG")
    2. Case 2: Pet Spa Customer Feedback ("PET GROOMING & SPA")
    3. Case 3: High-Density Tech Recruitment ("SENIOR AI DESIGNER")
    4. Case 4: Restaurant / Culinary Menu ("THỰC ĐƠN ĐẶC BIỆT")

EXECUTION ADVANTAGE:
  - Loads 4B Distilled DiT into GPU ONCE, then iterates through all 4 cases in sequence.
  - Takes only ~10 - 15 seconds total on 2x NVIDIA A30.
  - Automatically packages all 4 generated PNGs into `output_distill_4cases.zip`
    for convenient 1-click download from JupyterLab to the local development machine.
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

CASES_CONFIG: List[Dict[str, Any]] = [
    {
        "id": "case1_burger_opening",
        "title": "Burger Grand Opening",
        "hero_text": "TƯNG BỪNG\nKHAI TRƯƠNG",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo ẩm thực sang trọng, mảng tường gạch tối màu phẳng ở phía trên làm nền tĩnh lặng, "
            "dòng chữ tiêu đề lớn 3D mạ vàng kim loại đúc nổi sắc nét ở phía trên, bên dưới là bàn gỗ với chiếc "
            "bánh burger bò đẫm phô mai vàng ươm thơm ngon, ánh sáng studio tương phản cao, phong cách điện ảnh "
            "sang trọng, không có watermark"
        ),
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "steps": 8,
        "guidance": 1.5,
        "seed": 42,
    },
    {
        "id": "case2_pet_spa",
        "title": "Pet Grooming & Spa Feedback",
        "hero_text": "PET GROOMING\n& SPA",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo dịch vụ thú cưng cao cấp, mảng tường màu xanh teal sẫm tối ở phía trên làm nền tĩnh, "
            "dòng chữ tiêu đề lớn 3D mạ vàng phát sáng sắc nét ở phía trên, bên dưới là chú chó Poodle trắng xù sau "
            "khi tắm cắt tỉa lông bồng bềnh sạch sẽ mỉm cười trong tiệm spa, ánh sáng studio mềm mại, không có watermark"
        ),
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "steps": 8,
        "guidance": 1.5,
        "seed": 123,
    },
    {
        "id": "case3_recruitment",
        "title": "Tech Talent Recruitment",
        "hero_text": "TUYỂN DỤNG\nAI DESIGNER",
        "font": "bevietnam",
        "prompt": (
            "Poster tuyển dụng công nghệ cao, mảng tường kính tối màu phẳng ở phía trên làm nền tĩnh, "
            "dòng chữ tiêu đề lớn 3D kim loại chrome phản chiếu ánh sáng sắc nét ở phía trên, các kỹ sư "
            "làm việc mờ ảo có chiều sâu ở hậu cảnh phía dưới, phong cách điện ảnh hiện đại tối giản, không có watermark"
        ),
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "steps": 8,
        "guidance": 1.5,
        "seed": 777,
    },
    {
        "id": "case4_food_menu",
        "title": "Artisan Culinary Menu",
        "hero_text": "THỰC ĐƠN ĐẶC BIỆT",
        "font": "playfair",
        "prompt": (
            "Poster thực đơn nhà hàng ẩm thực sang trọng, bàn gỗ mun tối màu với đĩa bít tết bò nướng "
            "xèo xèo và ly cocktail cam sả mát lạnh, ánh nến lung linh mờ ảo, dòng chữ tiêu đề lớn 3D "
            "khắc gỗ mạ vàng đồng cổ sắc nét tinh xảo ở phía trên, phong cách điện ảnh nghệ thuật ẩm thực, "
            "không có watermark"
        ),
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "steps": 8,
        "guidance": 1.5,
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


def find_distill_checkpoint(base_checkpoint_dir: Path) -> Path | None:
    """Searches for the 4B distilled DiT checkpoint across standard paths."""
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
    parser = argparse.ArgumentParser(description="Generate 4 diverse commercial DiT poster baselines")
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
        help="Direct path to flux-2-klein-4b.safetensors",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_distill_4cases",
        help="Output directory for generated PNGs and ZIP",
    )
    parser.add_argument(
        "--t_offset",
        type=float,
        default=10.0,
        help="RoPE time offset for In-Context Glyph (default: 10.0)",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 105)
    print("🚀 TENDOO AI - FLUX.2-KLEIN-4B DISTILLED 4-CASE BATCH GENERATOR")
    print("=" * 105)
    print(f"  Target Cases      : {len(CASES_CONFIG)} (Burger Opening, Pet Spa, Recruitment, Food Menu)")
    print(f"  Output Directory  : {out_path.resolve()}")

    # 1. Device Allocation
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = "cuda:0"
        device_ae = "cuda:1"
        print(f"  [GPU Setup] Dual GPU active: DiT on {device_dit}, VAE/Qwen3 on {device_ae}")
    elif num_gpus == 1:
        device_dit = "cuda:0"
        device_ae = "cuda:0"
        print(f"  [GPU Setup] Single GPU active: {device_dit}")
    else:
        device_dit = "cpu"
        device_ae = "cpu"
        print("  [!] WARNING: Running on CPU (testing syntax only)")

    # 2. Checkpoint Resolution
    base_dir = Path(args.checkpoint_dir)
    if args.distill_model_path:
        model_file = Path(args.distill_model_path)
    else:
        model_file = find_distill_checkpoint(base_dir)

    if not model_file or not model_file.exists():
        print(f"\n[ERROR] Distilled DiT checkpoint not found!")
        print(f"  Searched locations around: {base_dir}")
        print("  Please specify explicitly with --distill_model_path <path_to_flux-2-klein-4b.safetensors>")
        sys.exit(1)

    print(f"\n[1/3] Loading Distilled DiT from: {model_file}")
    os.environ["KLEIN_4B_MODEL_PATH"] = str(model_file)
    model = load_flow_model(
        model_name="flux.2-klein-4b",
        device=device_dit,
    )
    model.eval()

    print("\n[2/3] Loading VAE and Text Encoder (Qwen3-4B-FP8)...")
    if args.checkpoint_dir:
        os.environ["FLUX_CHECKPOINT_DIR"] = str(args.checkpoint_dir)
    ae = load_ae(model_name="flux.2-klein-base-4b", device=device_ae)
    ae.eval()

    text_encoder = load_qwen3_embedder(variant="4B", device=device_ae)

    # 3. Iterate over the 4 cases
    print(f"\n[3/3] Generating {len(CASES_CONFIG)} Commercial Cases...")
    generated_files: List[Path] = []
    summary_records: List[Dict[str, Any]] = []

    for idx, c in enumerate(CASES_CONFIG, 1):
        cid = c["id"]
        hero_text = c["hero_text"]
        prompt = c["prompt"]
        font = c["font"]
        w = (c["canvas_w"] // 16) * 16
        h = (c["canvas_h"] // 16) * 16
        lat_w, lat_h = w // 16, h // 16
        steps = c["steps"]
        guidance = c["guidance"]
        seed = c["seed"]

        print("\n" + "-" * 90)
        print(f"▶️ [{idx}/{len(CASES_CONFIG)}] Case: {c['title']} ({cid})")
        print(f"   Hero Text: '{hero_text}' | Font: {font} | Canvas: {w}x{h}px | Steps: {steps} | g: {guidance}")
        t0 = time.time()

        # A. Render Glyph Bitmap
        glyph_info: GlyphInfo = render_glyph(
            text=hero_text,
            font_name_or_path=font,
            target_width=c["box_w"],
            target_height=c["box_h"],
            auto_size=False,
        )
        glyph_file = out_path / f"glyph_{cid}.png"
        glyph_info.image.save(glyph_file)

        # B. Encode glyph to ref tokens at t=10.0
        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae, glyph_info.image, t_offset=args.t_offset, device=device_ae)
        ref_tokens = ref_tokens.to(device_dit)
        ref_ids = ref_ids.to(device_dit)

        # C. Encode prompt
        with torch.no_grad():
            txt_emb = text_encoder([prompt]).to(device=device_dit, dtype=torch.bfloat16)
        txt_tokens, txt_ids = batched_prc_txt(txt_emb)

        # D. Run Distilled Euler ODE Sampling
        torch.manual_seed(seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=steps, image_seq_len=img_tokens.shape[1])

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

            # Decode VAE
            lat_2d = rearrange(out_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
            img_out = ae.decode(lat_2d.to(device_ae))

        dur = time.time() - t0

        # Save Image
        pil_img = Image.fromarray(
            ((img_out[0].float().clamp(-1, 1) + 1.0) * 127.5)
            .permute(1, 2, 0)
            .byte()
            .cpu()
            .numpy()
        )

        out_file = out_path / f"{cid}.png"
        pil_img.save(out_file)
        generated_files.append(out_file)
        print(f"   [✓] Completed in {dur:.2f}s! -> Saved: {out_file.name}")

        summary_records.append({
            "case": cid,
            "title": c["title"],
            "hero_text": hero_text,
            "duration_s": round(dur, 2),
            "file": out_file.name,
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Package all outputs into a single ZIP for 1-click download from JupyterLab
    zip_filename = out_path / "output_distill_4cases.zip"
    print(f"\n[Packaging] Creating ZIP archive: {zip_filename.name}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in generated_files:
            zf.write(f, arcname=f.name)
        for g in out_path.glob("glyph_*.png"):
            zf.write(g, arcname=g.name)
    print(f"  [✓] ZIP Archive Created ({zip_filename.stat().st_size / 1024 / 1024:.2f} MB)")

    # 5. Summary Table
    print("\n" + "=" * 105)
    print("📊 TENDOO AI - FLUX.2 DISTILLED 4-CASE BATCH GENERATION SUMMARY")
    print("=" * 105)
    print(f"{'CASE ID':<24} | {'HERO TITLE (DiT)':<26} | {'TIME (s)':<10} | {'OUTPUT FILE':<28}")
    print("-" * 105)
    for r in summary_records:
        print(f"{r['case']:<24} | {r['hero_text']:<26} | {r['duration_s']}s{' ':<5} | {r['file']:<28}")
    print("=" * 105)
    print(f"\n🎉 ALL 4 CASES READY! Download archive directly from:")
    print(f"   👉 {zip_filename.resolve()}\n")


if __name__ == "__main__":
    main()
