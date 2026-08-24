"""
Dynamic Multi-Text In-Context Generator
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE

Features:
- Supports arbitrary N Vietnamese text blocks with individual time offsets (e.g., t=10.0, 11.0, 12.0...)
- Automatic tight-crop glyph rendering with binary-search font sizing
- Multi-GPU auto-distribution (GPU 0 for DiT, GPU 1 for VAE & Qwen3)
- Fully aligned with Tendoo AI 3-Pillar & Dual Division of Labor principles
"""

import argparse
import os
import sys
from pathlib import Path

# Auto-configure PYTHONPATH to include src/
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Auto-configure Offline Mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from einops import rearrange
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import native FLUX.2 modules
from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import (
    batched_prc_txt,
    denoise_cfg,
    get_schedule,
    prc_img,
)
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import (
    load_ae,
    load_flow_model,
)


def create_glyph_image(
    text: str,
    target_width: int,
    target_height: int,
    font_path: str | None = None,
    font_size: int = 64,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    padding_ratio: float = 0.08,
) -> Image.Image:
    """
    Renders a tight-crop Vietnamese glyph image with exact text and diacritics.
    Handles auto-wrapping and binary-search font sizing.
    """
    assert target_width > 0 and target_height > 0
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16

    candidate_fonts = [
        font_path,
        str(ROOT_DIR / "fonts" / "BeVietnamPro-Black.ttf"),
        str(ROOT_DIR / "fonts" / "PlayfairDisplay.ttf"),
        str(ROOT_DIR / "fonts" / "Anton-Regular.ttf"),
        str(ROOT_DIR / "fonts" / "Pacifico-Regular.ttf"),
        str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
        str(ROOT_DIR / "fonts" / "DancingScript.ttf"),
        str(ROOT_DIR / "fonts" / "Oswald.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    selected_font_path = None
    for fp in candidate_fonts:
        if fp and os.path.exists(fp):
            selected_font_path = fp
            break

    if selected_font_path is None:
        raise RuntimeError("❌ No valid Unicode font found with Vietnamese support!")

    pad_w = int(target_width * padding_ratio)
    pad_h = int(target_height * padding_ratio)
    max_w = target_width - 2 * pad_w
    max_h = target_height - 2 * pad_h

    text = text.replace("\\n", "\n")

    if "\n" in text:
        candidate_layouts = [[line.strip() for line in text.split("\n") if line.strip()]]
    else:
        words = text.split()
        candidate_layouts = []
        if len(words) >= 4:
            mid = len(words) // 2
            candidate_layouts.append([" ".join(words[:mid]), " ".join(words[mid:])])
        if len(words) >= 6:
            p1 = len(words) // 3
            p2 = 2 * len(words) // 3
            candidate_layouts.append([" ".join(words[:p1]), " ".join(words[p1:p2]), " ".join(words[p2:])])
        candidate_layouts.append([text])

    best_font = None
    best_lines = None
    best_size = 0
    best_ascent_offset = 0

    for lines in candidate_layouts:
        low, high = 14, 160
        opt_font = None
        opt_size = 0
        opt_offset = 0

        while low <= high:
            mid_size = (low + high) // 2
            try:
                test_font = ImageFont.truetype(selected_font_path, size=mid_size)
            except Exception:
                test_font = ImageFont.load_default()

            total_h = 0
            max_line_w = 0
            ascent_offsets = []

            for line in lines:
                bbox = test_font.getbbox(line)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                ascent_offsets.append(bbox[1])
                max_line_w = max(max_line_w, lw)
                total_h += lh

            line_spacing = int(mid_size * 0.18) * (len(lines) - 1)
            total_h += line_spacing

            if max_line_w <= max_w and total_h <= max_h:
                opt_font = test_font
                opt_size = mid_size
                opt_offset = ascent_offsets[0] if ascent_offsets else 0
                low = mid_size + 1
            else:
                high = mid_size - 1

        if opt_size > best_size:
            best_size = opt_size
            best_font = opt_font
            best_lines = lines
            best_ascent_offset = opt_offset

    if best_font is None:
        best_font = ImageFont.truetype(selected_font_path, size=18)
        best_lines = candidate_layouts[-1]
        best_size = 18
        best_ascent_offset = 0

    img = Image.new("RGB", (target_width, target_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    line_heights = []
    line_widths = []
    for line in best_lines:
        bbox = best_font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(best_size * 0.18)
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)
    curr_y = (target_height - total_block_h) // 2

    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        curr_x = (target_width - lw) // 2
        bbox = best_font.getbbox(line)
        draw.text((curr_x - bbox[0], curr_y - bbox[1]), line, fill=text_color, font=best_font)
        curr_y += line_heights[i] + line_spacing

    return img


def encode_glyph_to_incontext_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a tight-crop glyph image with AutoEncoder and builds 4D RoPE coordinate IDs at (t_offset, 0, 0, 0).
    """
    g_arr = np.array(glyph_img).astype(np.float32) / 127.5 - 1.0
    g_tensor = torch.from_numpy(g_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        g_latent = ae.encode(g_tensor)

    ref_tokens, _ = prc_img(g_latent[0])
    ref_tokens = ref_tokens.unsqueeze(0)

    g_h, g_w = g_latent.shape[2], g_latent.shape[3]
    t_coords = torch.full((g_h, g_w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(g_h, dtype=torch.float32, device=device).unsqueeze(1).expand(g_h, g_w)
    w_coords = torch.arange(g_w, dtype=torch.float32, device=device).unsqueeze(0).expand(g_h, g_w)
    l_coords = torch.zeros((g_h, g_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0)

    return ref_tokens, ref_ids


def generate_multi_text_poster(
    prompt: str,
    texts: list[str],
    t_offsets: list[float],
    image_ref: str | None = None,
    t_product: float = 10.0,
    width: int = 576,
    height: int = 1024,
    output_path: str = "output_multi_text.png",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    font_path: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
    device: str = "cuda",
):
    print("=" * 80)
    print("🐾 TENDOO AI: DYNAMIC MULTI-TEXT POSTER GENERATION")
    print(f"🎯 Target Model : {model_name.upper()} (Sole Target Base 4B)")
    print(f"📐 Canvas Size  : {width}x{height} (Divisible by 16)")
    print(f"📝 Text Count   : {len(texts)} independent text blocks")
    for i, (txt, t_val) in enumerate(zip(texts, t_offsets)):
        print(f"   [{i+1}] '{txt}' -> t_offset = {t_val}")
    print(f"🎨 Guidance Scale: {guidance} | ODE Steps: {num_steps}")
    print("=" * 80)

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print(f"🚀 Running on Dual GPUs: DiT on {device_dit}, VAE/Qwen3 on {device_ae}")
    else:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:0")
        device_te = torch.device("cuda:0")
        print(f"⚠️ Running on Single GPU: {device_dit}")

    torch.manual_seed(seed)

    # 1. Render Tight-Crop Glyphs for all text blocks
    print(f"\n[Step 1/5] Rendering {len(texts)} Tight-Crop Typography Bitmaps...")
    glyph_imgs = []
    stem = Path(output_path).stem
    for i, (txt_content, t_val) in enumerate(zip(texts, t_offsets)):
        box_w = min(width - 64, 512)
        box_w = (box_w // 16) * 16
        # Adaptive height based on length
        num_words = len(txt_content.replace("\\n", " ").split())
        if "\n" in txt_content or num_words >= 6:
            box_h = 224
        elif num_words >= 4:
            box_h = 192
        else:
            box_h = 160

        g_img = create_glyph_image(text=txt_content, target_width=box_w, target_height=box_h, font_path=font_path)
        glyph_imgs.append(g_img)
        preview_file = f"{stem}_text{i+1}_t{int(t_val)}_preview.png"
        g_img.save(preview_file)
        print(f"  -> Saved Glyph [{i+1}] preview: {preview_file} ({box_w}x{box_h})")

    # 2. Load Models
    print("\n[Step 2/5] Loading FLUX.2 Base DiT & VAE...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)

    # 3. Encode Text Prompt
    print("\n[Step 3/5] Encoding Prompt via Qwen3-4B-FP8...")
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    if num_gpus < 2:
        del text_encoder
        torch.cuda.empty_cache()

    # 4. Prepare Canvas & Multi-Reference Tokens
    print("\n[Step 4/5] Encoding Multi-Reference In-Context Tokens...")
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    ref_token_list = []
    ref_id_list = []

    # Encode all Glyph blocks
    for i, (g_img, t_val) in enumerate(zip(glyph_imgs, t_offsets)):
        ref_tok, ref_id = encode_glyph_to_incontext_tokens(ae=ae, glyph_img=g_img, t_offset=t_val, device=device_ae)
        ref_token_list.append(ref_tok)
        ref_id_list.append(ref_id)
        print(f"  -> Added Text [{i+1}] at t={t_val} ({ref_tok.shape[1]} tokens)")

    all_ref_tokens = torch.cat(ref_token_list, dim=1).to(device_dit)
    all_ref_ids = torch.cat(ref_id_list, dim=1).to(device_dit)

    print(f"  -> Total Canvas Tokens: {img_tokens.shape[1]} (Grid: {lat_h}x{lat_w})")
    print(f"  -> Total Ref Tokens   : {all_ref_tokens.shape[1]} across {len(ref_token_list)} text blocks")

    # 5. Denoise
    print("\n[Step 5/5] Running Denoise ODE (50 Steps)...")
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    with torch.no_grad():
        out_latent = denoise_cfg(
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

        out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        print("  -> Decoding output poster via VAE...")
        out_pixels = ae.decode(out_latent.to(device_ae))
        out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_img = Image.fromarray(out_pixels)
        result_img.save(output_path)

    print("\n" + "=" * 80)
    print(f"🎉 Multi-Text Poster Generated Successfully!")
    print(f"📸 Result saved to: {output_path} ({width}x{height})")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Dynamic Multi-Text Poster with Arbitrary In-Context Offsets")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing background scene & materials")
    parser.add_argument("--texts", nargs="+", required=True, help="List of Vietnamese text strings (e.g. --texts 'Text 1' 'Text 2')")
    parser.add_argument("--t_offsets", nargs="+", type=float, required=True, help="List of time offsets (e.g. --t_offsets 10.0 11.0 12.0)")
    parser.add_argument("--width", type=int, default=576, help="Width in pixels (multiple of 16, default 576)")
    parser.add_argument("--height", type=int, default=1024, help="Height in pixels (multiple of 16, default 1024)")
    parser.add_argument("--output", type=str, default="output_dynamic_poster.png", help="Output path")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--font_path", type=str, default=None, help="Path to Unicode font")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()
    assert len(args.texts) == len(args.t_offsets), "Number of texts must match number of t_offsets!"

    generate_multi_text_poster(
        prompt=args.prompt,
        texts=args.texts,
        t_offsets=args.t_offsets,
        width=args.width,
        height=args.height,
        output_path=args.output,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        font_path=args.font_path,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        device=args.device,
    )
