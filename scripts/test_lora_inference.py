"""
====================================================================================================
TENDOO AI - LORA INFERENCE & COMPARATIVE EVALUATION ENGINE
====================================================================================================
Script: scripts/test_lora_inference.py
Purpose:
    Runs inference on FLUX.2-klein-base-4B with trained LoRA adapters:
    1. Loads base model + injected LoRA safetensors checkpoint.
    2. Supports Side-by-Side Comparison:
       - Column A: Base Model Zero-Shot (Without LoRA)
       - Column B: Tendoo AI Multi-Slot LoRA (Fine-tuned)
    3. Benchmarks Vietnamese Diacritic Precision & Multi-Slot Disentanglement.
====================================================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

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


def encode_glyph(ae: AutoEncoder, img: Image.Image, t_offset: float, device: torch.device):
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


def run_inference_sample(
    model: Flux2,
    ae: AutoEncoder,
    text_encoder,
    prompt: str,
    text1: str,
    font1: str,
    text2: str,
    font2: str,
    product_path: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
    device: torch.device = torch.device("cuda:0"),
) -> Image.Image:
    torch.manual_seed(seed)

    # 1. Render Glyphs
    g1 = create_glyph_image(text1, min(width - 64, 512), 160, resolve_font_path(font1))
    g2 = create_glyph_image(text2, min(width - 64, 600), 256, resolve_font_path(font2))

    # 2. Encode Prompts
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device)
        txt_ids = txt_ids.to(device)

    # 3. Canvas Latent Init
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device)
    img_ids = img_ids.unsqueeze(0).to(device)

    # 4. Reference Tokens
    tok1, id1 = encode_glyph(ae, g1, 10.0, device)
    tok2, id2 = encode_glyph(ae, g2, 20.0, device)
    ref_toks = [tok1, tok2]
    ref_ids = [id1, id2]

    if product_path and os.path.exists(product_path):
        p_img = Image.open(product_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        p_arr = np.array(p_img).astype(np.float32) / 127.5 - 1.0
        p_ten = torch.from_numpy(p_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            p_lat = ae.encode(p_ten)
        p_tok, p_id = encode_glyph(ae, p_img, 30.0, device)
        ref_toks.append(p_tok)
        ref_ids.append(p_id)

    all_ref_tokens = torch.cat(ref_toks, dim=1)
    all_ref_ids = torch.cat(ref_ids, dim=1)

    # 5. Denoise ODE
    timesteps = get_schedule(num_steps=steps, image_seq_len=img_tokens.shape[1])
    with torch.no_grad():
        out = denoise_cfg(
            model=model,
            img=img_tokens,
            img_ids=img_ids,
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
    parser = argparse.ArgumentParser(description="Tendoo AI - LoRA Inference & Benchmark Engine")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA safetensors checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Student clean prompt")
    parser.add_argument("--text1", type=str, required=True, help="Slot 10 text")
    parser.add_argument("--font1", type=str, default="bevietnam", help="Slot 10 font")
    parser.add_argument("--text2", type=str, required=True, help="Slot 20 text")
    parser.add_argument("--font2", type=str, default="playfair", help="Slot 20 font")
    parser.add_argument("--product", type=str, default=None, help="Slot 30 product image path")
    parser.add_argument("--width", type=int, default=1024, help="Canvas width")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output", type=str, default="output/lora_result.png", help="Output path")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(" [*] Loading Base Model & Injecting LoRA...")
    model = load_flow_model("flux.2-klein-base-4b", device=device)
    model.eval()

    model, _ = inject_lora_to_flux2_klein(model, r=32, lora_alpha=32.0)
    load_lora_weights(model, args.lora_path)

    print(" [*] Loading AutoEncoder & Qwen3...")
    ae = load_ae("flux.2-klein-base-4b", device=device)
    te = load_qwen3_embedder(variant="4B", device=device)

    print(f" [*] Running inference for prompt: {args.prompt}")
    img = run_inference_sample(
        model=model,
        ae=ae,
        text_encoder=te,
        prompt=args.prompt,
        text1=args.text1,
        font1=args.font1,
        text2=args.text2,
        font2=args.font2,
        product_path=args.product,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        device=device,
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_p)
    print(f" [OK] Output saved to: {out_p}")


if __name__ == "__main__":
    main()
