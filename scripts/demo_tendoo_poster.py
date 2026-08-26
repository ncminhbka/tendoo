"""
================================================================================
TENDOO AI - EXECUTIVE DEMO SCRIPT (PHASE 1 MILESTONE)
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Features:
- Single Vietnamese Typography with 100% Diacritic Accuracy & Auto-Wrapping
- Optional Real Product Reference Image Ingestion (t=60.0 default)
- Customizable Canvas Aspect Ratio & Resolution (9:16, 1:1, 16:9, etc.)
- Convenient Font Selector (Playfair, BeVietnam, Anton, Pacifico, Graffiti...)
- Fast 1-Pass In-Context ODE Flow Matching (~30s on 2x A30)
================================================================================
"""

import argparse
import os
import sys
import time
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

# Built-in Font Registry with Friendly Aliases
FONT_REGISTRY = {
    "bevietnam": str(ROOT_DIR / "fonts" / "BeVietnamPro-Black.ttf"),
    "playfair": str(ROOT_DIR / "fonts" / "PlayfairDisplay.ttf"),
    "anton": str(ROOT_DIR / "fonts" / "Anton-Regular.ttf"),
    "pacifico": str(ROOT_DIR / "fonts" / "Pacifico-Regular.ttf"),
    "graffiti": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "sedgwick": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "dancing": str(ROOT_DIR / "fonts" / "DancingScript.ttf"),
    "oswald": str(ROOT_DIR / "fonts" / "Oswald.ttf"),
    # New Playful, Fresh, Summer SVN Fonts:
    "cookies": str(ROOT_DIR / "fonts" / "SVN-Cookies.ttf"),
    "grocery": str(ROOT_DIR / "fonts" / "SVN-Grocery Rounded.ttf"),
    "gretoon": str(ROOT_DIR / "fonts" / "SVN-Gretoon.ttf"),
    "blowbrush": str(ROOT_DIR / "fonts" / "SVN-Blow Brush.ttf"),
    "brush": str(ROOT_DIR / "fonts" / "SVN-Blow Brush.ttf"),
    "holidays": str(ROOT_DIR / "fonts" / "SVN-Holidays.ttf"),
    "clementine": str(ROOT_DIR / "fonts" / "SVN-Clementine.ttf"),
    "harabaras": str(ROOT_DIR / "fonts" / "SVN-Harabaras.ttf"),
    "lolapeluza": str(ROOT_DIR / "fonts" / "SVN-Lolapeluza Black.ttf"),
    "gotham": str(ROOT_DIR / "fonts" / "SVN-Gotham Ultra.otf"),
}



def resolve_font_path(font_name_or_path: str | None) -> str:
    """Resolves font alias (e.g. 'playfair', 'bevietnam') or validates file path."""
    if font_name_or_path:
        key = font_name_or_path.lower().strip()
        if key in FONT_REGISTRY and os.path.exists(FONT_REGISTRY[key]):
            return FONT_REGISTRY[key]
        if os.path.exists(font_name_or_path):
            return font_name_or_path

    # Fallback to defaults
    for p in [
        FONT_REGISTRY["bevietnam"],
        FONT_REGISTRY["playfair"],
        FONT_REGISTRY["anton"],
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]:
        if os.path.exists(p):
            return p

    raise RuntimeError("❌ No valid Vietnamese Unicode font found!")


def create_glyph_image(
    text: str,
    target_width: int = 512,
    target_height: int = 224,
    font_path: str | None = None,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    padding_ratio: float = 0.08,
    tight_crop: bool = True,
) -> Image.Image:
    """
    Renders TRUE TIGHT-CROP Vietnamese glyph bitmap with automatic line wrapping,
    binary-search font sizing, and exact bounding-box cropping snapped to multiples of 16.
    """
    assert target_width > 0 and target_height > 0
    envelope_w = (target_width // 16) * 16
    envelope_h = (target_height // 16) * 16

    font_path = resolve_font_path(font_path)

    pad_w = int(envelope_w * padding_ratio)
    pad_h = int(envelope_h * padding_ratio)
    max_w = envelope_w - 2 * pad_w
    max_h = envelope_h - 2 * pad_h

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

    for lines in candidate_layouts:
        low, high = 14, 200
        opt_font = None
        opt_size = 0

        while low <= high:
            mid_size = (low + high) // 2
            try:
                test_font = ImageFont.truetype(font_path, size=mid_size)
            except Exception:
                test_font = ImageFont.load_default()

            total_h = 0
            max_line_w = 0

            for line in lines:
                bbox = test_font.getbbox(line)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                max_line_w = max(max_line_w, lw)
                total_h += lh

            line_spacing = int(mid_size * 0.20) * (len(lines) - 1)
            total_h += line_spacing

            if max_line_w <= max_w and total_h <= max_h:
                opt_font = test_font
                opt_size = mid_size
                low = mid_size + 1
            else:
                high = mid_size - 1

        if opt_size > best_size:
            best_size = opt_size
            best_font = opt_font
            best_lines = lines

    if best_font is None:
        best_font = ImageFont.truetype(font_path, size=20)
        best_lines = candidate_layouts[-1]
        best_size = 20

    line_heights = []
    line_widths = []
    for line in best_lines:
        bbox = best_font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(best_size * 0.20)
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)
    total_block_w = max(line_widths)

    if tight_crop:
        # Tight-crop dynamically around the actual text bounding box
        pad_x = max(10, int(total_block_w * padding_ratio))
        pad_y = max(8, int(total_block_h * padding_ratio))
        final_w = total_block_w + 2 * pad_x
        final_h = total_block_h + 2 * pad_y
        final_w = max(32, ((final_w + 15) // 16) * 16)
        final_h = max(32, ((final_h + 15) // 16) * 16)
    else:
        final_w = envelope_w
        final_h = envelope_h

    img = Image.new("RGB", (final_w, final_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    curr_y = (final_h - total_block_h) // 2

    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        curr_x = (final_w - lw) // 2
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
    """Encodes a tight-crop glyph image to 4D RoPE coordinate tokens."""
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


def encode_product_to_incontext_tokens(
    ae: AutoEncoder,
    image_path: str,
    t_offset: float = 60.0,
    target_size: int = 1024,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes a natural product image into 4D RoPE tokens."""
    prod_img = Image.open(image_path).convert("RGB")
    prod_img = prod_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    p_arr = np.array(prod_img).astype(np.float32) / 127.5 - 1.0
    p_tensor = torch.from_numpy(p_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        p_latent = ae.encode(p_tensor)

    ref_tokens, _ = prc_img(p_latent[0])
    ref_tokens = ref_tokens.unsqueeze(0)

    p_h, p_w = p_latent.shape[2], p_latent.shape[3]
    t_coords = torch.full((p_h, p_w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(p_h, dtype=torch.float32, device=device).unsqueeze(1).expand(p_h, p_w)
    w_coords = torch.arange(p_w, dtype=torch.float32, device=device).unsqueeze(0).expand(p_h, p_w)
    l_coords = torch.zeros((p_h, p_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0)

    return ref_tokens, ref_ids


def run_demo(
    text: str,
    prompt: str,
    t_text: float = 10.0,
    image_ref: str | None = None,
    t_product: float = 60.0,
    width: int = 576,
    height: int = 1024,
    box_w: int | None = None,
    box_h: int | None = None,
    font: str = "bevietnam",
    output_path: str = "demo_output.png",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
):
    """
    Executes end-to-end Tendoo AI pipeline on FLUX.2 Klein 4B Base.
    """
    start_time = time.time()
    width = (width // 16) * 16
    height = (height // 16) * 16

    resolved_font = resolve_font_path(font)

    print("=" * 80)
    print(" 🚀 TENDOO AI: EXECUTIVE POSTER GENERATOR")
    print("=" * 80)
    print(f"📝 Text Input     : '{text}'")
    print(f"🧭 Text Time (t)  : {t_text}")
    print(f"🔤 Font Selected  : {Path(resolved_font).name}")
    print(f"📐 Canvas Size    : {width}x{height} pixels (Divisible by 16)")
    if image_ref:
        print(f"📦 Product Ref    : {image_ref} (t={t_product})")
    print(f"🎨 Prompt         : {prompt}")
    print(f"⚙️ Denoise Config : {num_steps} steps, CFG Guidance = {guidance}, Seed = {seed}")
    print("=" * 80)

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print(f"🚀 Dual GPU Mode: DiT on GPU 0, VAE/Qwen3 on GPU 1")
    else:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:0")
        device_te = torch.device("cuda:0")
        print(f"🚀 Single GPU Mode: {device_dit}")

    torch.manual_seed(seed)

    # 1. Render Glyph Bitmap with Intelligent Auto-Scaling for Multi-line / Poems
    print("\n[1/5] Generating Vietnamese Orthographic Glyph...")
    raw_lines = [l.strip() for l in text.replace("\\n", "\n").split("\n") if l.strip()]
    num_lines = len(raw_lines) if len(raw_lines) > 1 else (2 if len(text.split()) >= 4 else 1)

    if box_w is None:
        calc_w = min(width - 64, 896 if num_lines >= 3 else 512)
    else:
        calc_w = min(width, box_w)
    calc_w = (calc_w // 16) * 16

    if box_h is None:
        calc_h = min(height - 64, max(160, num_lines * 128))
    else:
        calc_h = min(height, box_h)
    calc_h = (calc_h // 16) * 16

    glyph_img = create_glyph_image(
        text=text,
        target_width=calc_w,
        target_height=calc_h,
        font_path=resolved_font,
    )
    glyph_preview = Path(output_path).stem + "_glyph_preview.png"
    glyph_img.save(glyph_preview)
    print(f"  -> Glyph preview saved: {glyph_preview} ({calc_w}x{calc_h} - {num_lines} lines)")

    # 2. Load Models
    print("\n[2/5] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)

    # 3. Encode Text Prompt
    print("\n[3/5] Encoding Text Prompt via Qwen3-4B-FP8...")
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    if num_gpus < 2:
        del text_encoder
        torch.cuda.empty_cache()

    # 4. Encode In-Context Reference Tokens
    print("\n[4/5] Encoding 4D RoPE In-Context Tokens...")
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    ref_token_list = []
    ref_id_list = []

    # Add Product Reference if provided
    if image_ref and os.path.exists(image_ref):
        prod_tokens, prod_ids = encode_product_to_incontext_tokens(
            ae=ae, image_path=image_ref, t_offset=t_product, device=device_ae
        )
        ref_token_list.append(prod_tokens)
        ref_id_list.append(prod_ids)
        print(f"  -> Added Product Reference at t={t_product} ({prod_tokens.shape[1]} tokens)")

    # Add Text Glyph
    text_tokens, text_ids = encode_glyph_to_incontext_tokens(
        ae=ae, glyph_img=glyph_img, t_offset=t_text, device=device_ae
    )
    ref_token_list.append(text_tokens)
    ref_id_list.append(text_ids)
    print(f"  -> Added Text Glyph at t={t_text} ({text_tokens.shape[1]} tokens)")

    all_ref_tokens = torch.cat(ref_token_list, dim=1).to(device_dit)
    all_ref_ids = torch.cat(ref_id_list, dim=1).to(device_dit)

    print(f"  -> Total Canvas Tokens: {img_tokens.shape[1]} (Grid: {lat_h}x{lat_w})")
    print(f"  -> Total Ref Tokens   : {all_ref_tokens.shape[1]}")

    # 5. Denoise Euler ODE
    print(f"\n[5/5] Running ODE Denoise ({num_steps} steps, CFG {guidance})...")
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
        print("  -> Decoding final image via VAE Decoder...")
        out_pixels = ae.decode(out_latent.to(device_ae))
        out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_img = Image.fromarray(out_pixels)
        result_img.save(output_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 DEMO POSTER GENERATED SUCCESSFULLY IN {elapsed:.2f}s!")
    print(f"📸 Final Result saved to : {output_path} ({width}x{height})")
    print(f"🖼️ Glyph Preview saved to: {glyph_preview}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tendoo AI - Executive Demo Script for Vietnamese Typography & Poster Generation")
    parser.add_argument("--text", type=str, required=True, help="Vietnamese Text String (e.g. 'TIỆM CÀ PHÊ ANH QUÂN')")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing background scene, material & lighting")
    parser.add_argument("--t_text", type=float, default=10.0, help="Time offset for Text Glyph (default: 10.0)")
    parser.add_argument("--image_ref", type=str, default=None, help="Optional path to Product Reference Image")
    parser.add_argument("--t_product", type=float, default=60.0, help="Time offset for Product Reference (default: 60.0)")
    parser.add_argument("--width", type=int, default=576, help="Width in pixels (default: 576 for 9:16)")
    parser.add_argument("--height", type=int, default=1024, help="Height in pixels (default: 1024 for 9:16)")
    parser.add_argument("--box_w", type=int, default=None, help="Optional custom Glyph Box width (default: auto)")
    parser.add_argument("--box_h", type=int, default=None, help="Optional custom Glyph Box height (default: auto)")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (playfair, bevietnam, anton, pacifico, graffiti, dancing, oswald) or path")
    parser.add_argument("--output", type=str, default="demo_output.png", help="Output image file path")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG Guidance scale (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    run_demo(
        text=args.text,
        prompt=args.prompt,
        t_text=args.t_text,
        image_ref=args.image_ref,
        t_product=args.t_product,
        width=args.width,
        height=args.height,
        box_w=args.box_w,
        box_h=args.box_h,
        font=args.font,
        output_path=args.output,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

