"""
====================================================================================================
TENDOO AI - LORA VISUAL EVALUATION BENCHMARK SUITE
====================================================================================================
Script: scripts/eval_lora_suite.py
Purpose:
    Performs visual qualitative inspection of trained LoRA checkpoints by generating
    images for 3 standardized benchmark probe cases:
    1. Probe 1 (T2I Multi-Block): Business Recruitment Poster (Title t=10 + Slogan t=20)
    2. Probe 2 (I2I Product Multi-Block): Luxury Perfume (Product t=30 + Title t=10 + Subtitle t=20)
    3. Probe 3 (Hard Crosstalk Stress Test): Tech Gala Neon (Headline t=10 + Thin CTA t=20)
====================================================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.flux2.autoencoder import AutoEncoder
from src.flux2.model import Flux2
from src.flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from src.flux2.text_encoder import load_qwen3_embedder
from src.flux2.util import load_ae, load_flow_model
from src.tendoo.glyph_engine import create_glyph_image, resolve_font_path
from src.tendoo.lora import inject_lora_to_flux2_klein, load_lora_weights


BENCHMARK_PROBES = [
    {
        "name": "probe1_t2i_recruitment",
        "title": "Probe 1: T2I Multi-Block Poster (Title + Slogan)",
        "prompt": "Poster tuyển dụng doanh nghiệp hiện đại sang trọng, tiêu đề dập nổi mạ vàng đồng cổ sắc nét ở tiền cảnh, slogan viền bạc tinh tế ngay bên dưới, hậu cảnh văn phòng cao ốc kính nhìn ra thành phố hoàng hôn, phong cách thiết kế corporate cao cấp, ánh sáng studio tương phản",
        "slots": [
            {"type": "glyph", "text": "TUYỂN DỤNG NHÂN TÀI", "font": "bevietnam", "t": 10.0, "w": 688, "h": 160},
            {"type": "glyph", "text": "BỨT PHÁ MỌI GIỚI HẠN", "font": "playfair", "t": 20.0, "w": 600, "h": 192},
        ],
        "width": 1024,
        "height": 1024,
        "product_path": None,
    },
    {
        "name": "probe2_i2i_luxury_perfume",
        "title": "Probe 2: I2I Product + 2 Text Slots",
        "prompt": "Chai nước hoa sang trọng đặt ngay ngắn trên bệ đá cẩm thạch đen bóng loáng, tiêu đề chữ vàng kim loại 3D phản chiếu studio ở trên cao, dòng chữ phụ khắc chìm mạ vàng tinh xảo trên đế đá, hậu cảnh sương đêm le lói ánh sáng kịch tính, phong cách chụp ảnh quảng cáo thương mại đỉnh cao",
        "slots": [
            {"type": "product", "path": "data/lifestyle_products/perfume_noir.png", "t": 30.0},
            {"type": "glyph", "text": "HƯƠNG SẮC QUÝ PHÁI", "font": "playfair", "t": 10.0, "w": 640, "h": 160},
            {"type": "glyph", "text": "ĐẲNG CẤP VƯỢT THỜI GIAN", "font": "bevietnam", "t": 20.0, "w": 720, "h": 192},
        ],
        "width": 1024,
        "height": 1024,
        "product_path": "data/lifestyle_products/perfume_noir.png",
    },
    {
        "name": "probe3_hard_crosstalk_stress",
        "title": "Probe 3: Hard Crosstalk Stress Test (Headline + Short CTA)",
        "prompt": "Sân khấu sự kiện công nghệ tương lai hoành tráng, tiêu đề chữ kim loại titan phản chiếu ánh đèn laser ở góc trên, khối huy hiệu nút bấm dập nổi dòng chữ khuyến mại màu vàng neon nổi bật ở góc đối diện, hậu cảnh khán phòng số hóa futuristic, ánh sáng volumetric tương phản cao",
        "slots": [
            {"type": "glyph", "text": "ĐẠI TIỆC CÔNG NGHỆ", "font": "bevietnam", "t": 10.0, "w": 640, "h": 160},
            {"type": "glyph", "text": "MUA 1 TẶNG 1", "font": "bevietnam", "t": 20.0, "w": 480, "h": 160},
        ],
        "width": 1024,
        "height": 1024,
        "product_path": None,
    },
]


def encode_image_slot(ae: AutoEncoder, img: Image.Image, t_offset: float, device: torch.device):
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latent = ae.encode(tensor)
    tokens, _ = prc_img(latent[0])
    tokens = tokens.unsqueeze(0).to(device)
    _, _, h, w = latent.shape
    t_c = torch.full((h, w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_c = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).expand(h, w)
    w_c = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).expand(h, w)
    l_c = torch.zeros((h, w), dtype=torch.float32, device=device)
    ids = torch.stack([t_c, h_c, w_c, l_c], dim=-1).reshape(-1, 4).unsqueeze(0)
    return tokens, ids


def generate_single_probe(
    model: Flux2,
    ae: AutoEncoder,
    text_encoder,
    probe_cfg: Dict,
    steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
    device: torch.device = torch.device("cuda:0"),
) -> Image.Image:
    torch.manual_seed(seed)

    width = probe_cfg["width"]
    height = probe_cfg["height"]
    prompt = probe_cfg["prompt"]

    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device)
        txt_ids = txt_ids.to(device)

    ref_tokens_list = []
    ref_ids_list = []

    for slot in probe_cfg["slots"]:
        t_val = slot["t"]
        if slot["type"] == "glyph":
            g_img = create_glyph_image(slot["text"], slot["w"], slot["h"], resolve_font_path(slot["font"]))
            toks, ids = encode_image_slot(ae, g_img, t_val, device)
            ref_tokens_list.append(toks)
            ref_ids_list.append(ids)
        elif slot["type"] == "product":
            p_path = PROJECT_ROOT / slot["path"]
            if p_path.exists():
                p_img = Image.open(p_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
                toks, ids = encode_image_slot(ae, p_img, t_val, device)
                ref_tokens_list.append(toks)
                ref_ids_list.append(ids)

    all_ref_tokens = torch.cat(ref_tokens_list, dim=1)
    all_ref_ids = torch.cat(ref_ids_list, dim=1)

    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device, dtype=torch.bfloat16)
    canvas_tokens, canvas_ids = prc_img(z_init[0])
    canvas_tokens = canvas_tokens.unsqueeze(0).to(device)
    canvas_ids = canvas_ids.unsqueeze(0).to(device)

    timesteps = get_schedule(num_steps=steps, image_seq_len=canvas_tokens.shape[1])
    with torch.no_grad():
        out = denoise_cfg(
            model=model,
            img=canvas_tokens,
            img_ids=canvas_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=all_ref_tokens,
            img_cond_seq_ids=all_ref_ids,
        )
        out = out.reshape(1, lat_h, lat_w, 128).permute(0, 3, 1, 2)
        pixels = ae.decode(out)
        pixels = ((pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI - LoRA Visual Benchmark Suite")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA safetensors checkpoint")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save output images")
    parser.add_argument("--model-name", type=str, default="flux.2-klein-base-4b", help="Base model name")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = Path(args.checkpoint).stem

    print("=" * 90)
    print(" 🎨 TENDOO AI - LORA VISUAL EVALUATION BENCHMARK SUITE")
    print(f" [*] Checkpoint : {args.checkpoint}")
    print(f" [*] Output Dir : {out_dir}")
    print(f" [*] Device     : {device}")
    print("=" * 90)

    print("\n[1/3] Loading VAE and Qwen3 Text Encoder...")
    ae = load_ae(args.model_name, device=device)
    te = load_qwen3_embedder(variant="4B", device=device)

    print("\n[2/3] Loading FLUX.2 Base 4B and Injecting LoRA Checkpoint...")
    model = load_flow_model(args.model_name, device=device)
    model.eval()

    model, _ = inject_lora_to_flux2_klein(model, r=32, lora_alpha=32.0)
    load_lora_weights(model, args.checkpoint)

    print("\n[3/3] Generating Visual Inspection Images...")
    for idx, probe in enumerate(BENCHMARK_PROBES, start=1):
        print(f"\n   -> Running {probe['title']}...")
        t0 = time.time()
        img = generate_single_probe(
            model=model,
            ae=ae,
            text_encoder=te,
            probe_cfg=probe,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            device=device,
        )
        elapsed = time.time() - t0

        out_file = out_dir / f"{ckpt_name}_{probe['name']}.png"
        img.save(out_file)
        print(f"      [SAVED] {out_file.name} ({elapsed:.1f}s)")

    print("\n" + "=" * 90)
    print(" 🎉 VISUAL BENCHMARK COMPLETED!")
    print(f" [*] Images saved to directory: {out_dir.resolve()}")
    print(" [*] Open the images directly in JupyterLab to inspect typography & anti-crosstalk fidelity.")
    print("=" * 90)


if __name__ == "__main__":
    main()
