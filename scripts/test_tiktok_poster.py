"""
TikTok / Reels / Shorts Vertical 9:16 Poster Generator
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE

Features:
- Native 9:16 vertical aspect ratio (Default: 576x1024, divisible by 16)
- Dual Independent Vietnamese Typography Areas:
  1. Main Title (e.g. "BỘ SƯU TẬP MÙA THU")
  2. Sub-title / Slogan / Promotion (e.g. "GIẢM NGAY 50% HÔM NAY")
- Optional Product Reference Image (e.g. images/reference_prod.png)
- Standard In-Context Time-Offset Conditioning (t=10, 20, 30 at 0,0) — Fast 1-Pass Execution!
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
    default_prep,
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
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
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
        box_ar = target_width / target_height
        candidate_layouts = [[text]]

        if len(words) >= 2 and box_ar < 3.5:
            mid = (len(words) + 1) // 2
            candidate_layouts.append([" ".join(words[:mid]), " ".join(words[mid:])])

        if len(words) >= 4 and box_ar < 2.0:
            w_per_l = (len(words) + 2) // 3
            candidate_layouts.append([
                " ".join(words[:w_per_l]),
                " ".join(words[w_per_l : 2 * w_per_l]),
                " ".join(words[2 * w_per_l :]),
            ])

    def test_layout_fit(lines: list[str], size: int):
        f = ImageFont.truetype(selected_font_path, size)
        dummy = Image.new("RGB", (1, 1))
        d = ImageDraw.Draw(dummy)

        line_spacing = max(int(size * 0.2), 4)
        max_line_w = 0
        total_h = 0
        line_boxes = []

        for line in lines:
            bbox = d.textbbox((0, 0), line, font=f)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            line_boxes.append((lw, lh, bbox))
            max_line_w = max(max_line_w, lw)
            total_h += lh

        total_h += line_spacing * (len(lines) - 1)
        fits = (max_line_w <= max_w) and (total_h <= max_h)
        return fits, size, max_line_w, total_h, f, line_spacing, line_boxes

    best_layout_result = None
    max_score = -1

    for lines in candidate_layouts:
        low = 14
        high = 180
        best_for_this_layout = None
        while low <= high:
            mid = (low + high) // 2
            fits, size, lw, lh, f, l_spacing, l_boxes = test_layout_fit(lines, mid)
            if fits:
                best_for_this_layout = (lines, size, lw, lh, f, l_spacing, l_boxes)
                low = mid + 1
            else:
                high = mid - 1

        if best_for_this_layout is not None:
            lines, size, lw, lh, f, l_spacing, l_boxes = best_for_this_layout
            score = size * (lw * lh) ** 0.5
            if score > max_score:
                max_score = score
                best_layout_result = best_for_this_layout

    if best_layout_result is None:
        fallback_lines = candidate_layouts[-1]
        _, size, lw, lh, f, l_spacing, l_boxes = test_layout_fit(fallback_lines, 14)
        lines = fallback_lines
    else:
        lines, size, lw, lh, f, l_spacing, l_boxes = best_layout_result

    print(f"  -> [Glyph] Layout '{text.replace(chr(10), ' ')}': {len(lines)} line(s), font_size={size}px in box ({target_width}x{target_height})")

    img = Image.new("RGB", (target_width, target_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    total_block_h = sum(b[1] for b in l_boxes) + l_spacing * (len(lines) - 1)
    curr_y = (target_height - total_block_h) // 2

    for i, line in enumerate(lines):
        lw, lh, bbox = l_boxes[i]
        curr_x = (target_width - lw) // 2 - bbox[0]
        draw.text((curr_x, curr_y - bbox[1]), line, font=f, fill=text_color)
        curr_y += lh + l_spacing

    return img


def encode_glyph_to_incontext_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a tight-crop glyph into standard In-Context tokens starting at (0, 0).
    """
    w, h = glyph_img.size
    expected_h = h // 16
    expected_w = w // 16

    np_arr = np.array(glyph_img, dtype=np.float32) / 127.5 - 1.0
    glyph_dtype = next(ae.parameters()).dtype if hasattr(ae, "parameters") else torch.bfloat16
    glyph_tensor = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=glyph_dtype)

    with torch.no_grad():
        encoded = ae.encode(glyph_tensor)[0]

    _, lat_h, lat_w = encoded.shape
    assert lat_h == expected_h and lat_w == expected_w

    t_c = torch.tensor([t_offset], dtype=torch.float32, device=device)
    h_c = torch.arange(lat_h, dtype=torch.float32, device=device)
    w_c = torch.arange(lat_w, dtype=torch.float32, device=device)
    l_c = torch.arange(1, dtype=torch.float32, device=device)

    ref_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0)
    ref_tokens = rearrange(encoded, "c h w -> (h w) c").unsqueeze(0).to(torch.bfloat16)

    return ref_tokens, ref_ids


def encode_product_to_incontext_tokens(
    ae: AutoEncoder,
    image_path: str,
    t_offset: float = 10.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a product image reference into standard In-Context tokens.
    """
    img = Image.open(image_path).convert("RGB")
    img_prep = default_prep(img, limit_pixels=1024**2, ensure_multiple=16)

    glyph_dtype = next(ae.parameters()).dtype if hasattr(ae, "parameters") else torch.bfloat16
    img_tensor = img_prep.unsqueeze(0).to(device=device, dtype=glyph_dtype)

    with torch.no_grad():
        encoded = ae.encode(img_tensor)[0]

    _, lat_h, lat_w = encoded.shape
    t_c = torch.tensor([t_offset], dtype=torch.float32, device=device)
    h_c = torch.arange(lat_h, dtype=torch.float32, device=device)
    w_c = torch.arange(lat_w, dtype=torch.float32, device=device)
    l_c = torch.arange(1, dtype=torch.float32, device=device)

    prod_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0)
    prod_tokens = rearrange(encoded, "c h w -> (h w) c").unsqueeze(0).to(torch.bfloat16)

    return prod_tokens, prod_ids


def generate_tiktok_poster(
    prompt: str,
    title: str,
    slogan: str | None = None,
    image_ref: str | None = None,
    width: int = 576,
    height: int = 1024,
    output_path: str = "output_tiktok_poster.png",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    font_path: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
    device: str = "cuda",
):
    width = (width // 16) * 16
    height = (height // 16) * 16

    print("=" * 80)
    print("📱 Starting TikTok / Reels 9:16 Vertical Poster Generation")
    print(f"Canvas Size : {width}x{height} (Aspect Ratio 9:16)")
    print(f"Main Title  : '{title}'")
    if slogan:
        print(f"Sub-Slogan  : '{slogan}'")
    if image_ref:
        print(f"Product Ref : '{image_ref}'")
    print(f"Prompt      : {prompt}")
    print(f"ODE Steps   : {num_steps} | CFG: {guidance} | Seed: {seed}")
    print("=" * 80)

    # Multi-GPU Device Allocation
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = "cuda:0"
        device_ae = "cuda:1"
        device_te = "cuda:1"
        print(f"  -> Detected {num_gpus} GPUs. Running DiT on {device_dit}, VAE/Qwen3 on {device_ae}")
    else:
        device_dit = device
        device_ae = device
        device_te = device

    torch.manual_seed(seed)

    # 1. Render 2 Separate Tight-Crop Glyphs
    print("\n[Step 1/5] Rendering 2 Separate Tight-Crop Typography Bitmaps...")
    title_w = min(width - 64, 512)
    title_w = (title_w // 16) * 16
    title_h = 224 if "\n" in title or len(title.split()) >= 4 else 160
    glyph_title = create_glyph_image(text=title, target_width=title_w, target_height=title_h, font_path=font_path)
    title_preview = Path(output_path).stem + "_title_preview.png"
    glyph_title.save(title_preview)
    print(f"  -> Saved Title preview: {title_preview} ({title_w}x{title_h})")

    glyph_slogan = None
    if slogan:
        slogan_w = min(width - 64, 512)
        slogan_w = (slogan_w // 16) * 16
        slogan_h = 192 if len(slogan.split()) >= 4 else 160
        glyph_slogan = create_glyph_image(text=slogan, target_width=slogan_w, target_height=slogan_h, font_path=font_path)
        slogan_preview = Path(output_path).stem + "_slogan_preview.png"
        glyph_slogan.save(slogan_preview)
        print(f"  -> Saved Slogan preview: {slogan_preview} ({slogan_w}x{slogan_h})")

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

    # 4. Prepare Multi-Context Tokens (Sharing the SAME t_offset for all text glyphs)
    print("\n[Step 4/5] Preparing In-Context Multi-Reference Tokens...")
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    ref_token_list = []
    ref_id_list = []

    # Optional Product Reference Image (t=10.0)
    if image_ref and os.path.exists(image_ref):
        prod_tokens, prod_ids = encode_product_to_incontext_tokens(
            ae=ae, image_path=image_ref, t_offset=10.0, device=device_ae
        )
        ref_token_list.append(prod_tokens)
        ref_id_list.append(prod_ids)
        print(f"  -> Added Product Image at t=10.0 ({prod_tokens.shape[1]} tokens)")
        typo_time_offset = 20.0
    else:
        typo_time_offset = 10.0

    # Main Title Glyph (sharing typo_time_offset)
    title_tokens, title_ids = encode_glyph_to_incontext_tokens(
        ae=ae, glyph_img=glyph_title, t_offset=typo_time_offset, device=device_ae
    )
    ref_token_list.append(title_tokens)
    ref_id_list.append(title_ids)
    print(f"  -> Added Main Title at t={typo_time_offset} ({title_tokens.shape[1]} tokens)")

    # Sub-Slogan Glyph (sharing the EXACT SAME typo_time_offset as Title!)
    if glyph_slogan is not None:
        slogan_tokens, slogan_ids = encode_glyph_to_incontext_tokens(
            ae=ae, glyph_img=glyph_slogan, t_offset=typo_time_offset, device=device_ae
        )
        ref_token_list.append(slogan_tokens)
        ref_id_list.append(slogan_ids)
        print(f"  -> Added Sub-Slogan at t={typo_time_offset} (SAME t_offset as Title! {slogan_tokens.shape[1]} tokens)")

    # Combined Reference Tokens
    all_ref_tokens = torch.cat(ref_token_list, dim=1).to(device_dit)
    all_ref_ids = torch.cat(ref_id_list, dim=1).to(device_dit)

    print(f"  -> Total Canvas Tokens: {img_tokens.shape[1]} (Grid: {lat_h}x{lat_w})")
    print(f"  -> Total Ref Tokens   : {all_ref_tokens.shape[1]} across {len(ref_token_list)} reference blocks")

    # 5. Denoise (Single Ultra-Fast Pass)
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
        print("  -> Decoding 9:16 vertical poster image via VAE...")
        out_pixels = ae.decode(out_latent.to(device_ae))
        out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_img = Image.fromarray(out_pixels)
        result_img.save(output_path)

    print("\n" + "=" * 80)
    print(f"🎉 9:16 TikTok Vertical Poster Generated Successfully!")
    print(f"📸 Result saved to: {output_path} ({width}x{height})")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 9:16 Vertical TikTok Poster with Dual Vietnamese Typography")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing background scene")
    parser.add_argument("--title", type=str, required=True, help="Main Title Vietnamese Text")
    parser.add_argument("--slogan", type=str, default=None, help="Optional Sub-Slogan / Promo Text")
    parser.add_argument("--image_ref", type=str, default=None, help="Optional Path to Product Reference Image")
    parser.add_argument("--width", type=int, default=576, help="Width in pixels (multiple of 16, default 576)")
    parser.add_argument("--height", type=int, default=1024, help="Height in pixels (multiple of 16, default 1024)")
    parser.add_argument("--output", type=str, default="output_tiktok_poster.png", help="Output path")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--font_path", type=str, default=None, help="Path to Unicode font")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    generate_tiktok_poster(
        prompt=args.prompt,
        title=args.title,
        slogan=args.slogan,
        image_ref=args.image_ref,
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
