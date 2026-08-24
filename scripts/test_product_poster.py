"""
Test Script: Commercial Product Advertisement Poster Generation
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE

Inputs:
1. Product Reference Image: e.g. images/reference_prod.png (Lotion / Cosmetic bottle)
2. Vietnamese Typography Text: e.g. "KEM DƯỠNG THỂ THUẦN CHAY"

Conditioning Mechanism:
- Reference 1 (Product Image): t = 10.0 (Standard In-Context grid)
- Reference 2 (Vietnamese Glyph): t = 20.0 (Baseline [0,0] vs RoPE Bound Box)

Generates:
[ Panel 1: Baseline (Time-Offset In-Context [0,0]) | Panel 2: RoPE Bound Box ]
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

# Auto-configure Offline Mode (No internet / HuggingFace calls)
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


def encode_product_image_ref(
    ae: AutoEncoder,
    image_path: str,
    t_offset: float = 10.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes product reference image using native FLUX.2 preprocessing and VAE.
    Returns:
    - prod_tokens: (1, N_prod, 128)
    - prod_ids: (1, N_prod, 4) with standard [t, h, w, l] coordinates starting at (0, 0)
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


def encode_text_glyph_ref(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    box: tuple[int, int, int, int],
    t_offset: float = 20.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encodes text glyph image and returns:
    - glyph_tokens: (1, N_glyph, 128)
    - rope_ids: (1, N_glyph, 4) with explicit box coordinates
    - base_ids: (1, N_glyph, 4) with default (0, 0) coordinates
    """
    ymin, xmin, ymax, xmax = box
    h_start = ymin // 16
    h_end = ymax // 16
    w_start = xmin // 16
    w_end = xmax // 16

    expected_h = h_end - h_start
    expected_w = w_end - w_start

    np_arr = np.array(glyph_img, dtype=np.float32) / 127.5 - 1.0
    glyph_dtype = next(ae.parameters()).dtype if hasattr(ae, "parameters") else torch.bfloat16
    glyph_tensor = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=glyph_dtype)

    with torch.no_grad():
        encoded = ae.encode(glyph_tensor)[0]

    _, lat_h, lat_w = encoded.shape
    assert lat_h == expected_h and lat_w == expected_w

    t_c = torch.tensor([t_offset], dtype=torch.float32, device=device)
    h_c = torch.arange(h_start, h_end, dtype=torch.float32, device=device)
    w_c = torch.arange(w_start, w_end, dtype=torch.float32, device=device)
    l_c = torch.arange(1, dtype=torch.float32, device=device)
    rope_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0)

    h_base = torch.arange(lat_h, dtype=torch.float32, device=device)
    w_base = torch.arange(lat_w, dtype=torch.float32, device=device)
    base_ids = torch.cartesian_prod(t_c, h_base, w_base, l_c).unsqueeze(0)

    glyph_tokens = rearrange(encoded, "c h w -> (h w) c").unsqueeze(0).to(torch.bfloat16)

    return glyph_tokens, rope_ids, base_ids


def run_product_poster_experiment(
    image_ref: str,
    text: str,
    prompt: str,
    box: list[int] = [128, 128, 384, 896],
    output_path: str = "output_product_poster.png",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    font_path: str | None = None,
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
    device: str = "cuda",
):
    ymin = (box[0] // 16) * 16
    xmin = (box[1] // 16) * 16
    ymax = (box[2] // 16) * 16
    xmax = (box[3] // 16) * 16
    box_snapped = (ymin, xmin, ymax, xmax)
    box_h = ymax - ymin
    box_w = xmax - xmin

    print("=" * 80)
    print("🚀 Starting Commercial Product Poster Experiment")
    print(f"Product Image: {image_ref}")
    print(f"Slogan Text  : '{text}' (Box: {box_snapped}, Size: {box_w}x{box_h})")
    print(f"Prompt       : {prompt}")
    print(f"Canvas       : {width}x{height} | Steps: {num_steps} | Guidance: {guidance} | Seed: {seed}")
    print("=" * 80)

    # Multi-GPU Device Allocation
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = "cuda:0"
        device_ae = "cuda:1"
        device_te = "cuda:1"
        print(f"  -> Detected {num_gpus} GPUs. DiT on {device_dit}, VAE/Qwen3 on {device_ae}")
    else:
        device_dit = device
        device_ae = device
        device_te = device

    torch.manual_seed(seed)

    # 1. Render Glyph
    print("\n[Step 1/5] Rendering Vietnamese Slogan Glyph...")
    glyph_img = create_glyph_image(text=text, target_width=box_w, target_height=box_h, font_path=font_path)
    glyph_path = Path(output_path).stem + "_glyph_preview.png"
    glyph_img.save(glyph_path)
    print(f"  -> Saved Glyph preview: {glyph_path}")

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

    # 4. Prepare Multi-Reference Conditioning (Product + Slogan)
    print("\n[Step 4/5] Preparing Multi-Reference Conditioning...")
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    # Reference 1: Product Image (t = 10.0)
    prod_tokens, prod_ids = encode_product_image_ref(
        ae=ae, image_path=image_ref, t_offset=10.0, device=device_ae
    )

    # Reference 2: Vietnamese Glyph (t = 20.0)
    glyph_tokens, rope_glyph_ids, base_glyph_ids = encode_text_glyph_ref(
        ae=ae, glyph_img=glyph_img, box=box_snapped, t_offset=20.0, device=device_ae
    )

    # Combined Tokens & IDs
    rope_tokens = torch.cat([prod_tokens, glyph_tokens], dim=1).to(device_dit)
    rope_ids = torch.cat([prod_ids, rope_glyph_ids], dim=1).to(device_dit)

    base_tokens = torch.cat([prod_tokens, glyph_tokens], dim=1).to(device_dit)
    base_ids = torch.cat([prod_ids, base_glyph_ids], dim=1).to(device_dit)

    print(f"  -> Canvas Tokens  : {img_tokens.shape[1]} tokens (Grid: {lat_h}x{lat_w})")
    print(f"  -> Product Tokens : {prod_tokens.shape[1]} tokens (t=10.0)")
    print(f"  -> Glyph Tokens   : {glyph_tokens.shape[1]} tokens (t=20.0)")

    # 5. Denoise
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    with torch.no_grad():
        # RUN 1: Baseline (Time-Offset In-Context [0,0])
        print("\n[Step 5/5] [Run 1/2] Running Baseline (Standard In-Context [0,0])...")
        out_latent_base = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=base_tokens,
            img_cond_seq_ids=base_ids,
        )
        out_latent_base = rearrange(out_latent_base, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_pixels_base = ae.decode(out_latent_base.to(device_ae))
        out_pixels_base = ((out_pixels_base[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_base = Image.fromarray(out_pixels_base)
        baseline_path = Path(output_path).stem + "_baseline.png"
        result_base.save(baseline_path)
        print(f"  -> Saved Baseline result to: {baseline_path}")

        # RUN 2: RoPE Bound Box
        print("\n[Run 2/2] Running RoPE Bound Box...")
        out_latent_rope = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=rope_tokens,
            img_cond_seq_ids=rope_ids,
        )
        out_latent_rope = rearrange(out_latent_rope, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_pixels_rope = ae.decode(out_latent_rope.to(device_ae))
        out_pixels_rope = ((out_pixels_rope[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_rope = Image.fromarray(out_pixels_rope)
        result_rope.save(output_path)
        print(f"  -> Saved RoPE result to: {output_path}")

    # Build 2-Panel Comparison
    comparison_path = Path(output_path).stem + "_COMPARISON.png"
    comp_w = width * 2
    comp_h = height
    comp_img = Image.new("RGB", (comp_w, comp_h), color=(20, 20, 20))
    comp_img.paste(result_base, (0, 0))
    comp_img.paste(result_rope, (width, 0))

    draw_comp = ImageDraw.Draw(comp_img)
    lbl_font = ImageFont.load_default()
    draw_comp.text((20, 20), "Panel 1: Baseline (Time-Offset In-Context [0,0])", fill=(255, 255, 0), font=lbl_font)
    draw_comp.text((width + 20, 20), "Panel 2: RoPE Bound Box", fill=(0, 255, 255), font=lbl_font)

    comp_img.save(comparison_path)
    print(f"\n" + "=" * 80)
    print(f"✅ Commercial Product Poster Experiment Completed!")
    print(f"📸 2-Panel Comparison saved to: {comparison_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Commercial Product Advertisement Poster with Vietnamese Text")
    parser.add_argument("--image_ref", type=str, required=True, help="Path to product reference image")
    parser.add_argument("--text", type=str, required=True, help="Vietnamese slogan / typography text")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a luxury commercial cosmetics advertisement poster, the product standing elegantly on smooth white silk, soft morning sunlight, studio product photography, clean minimalist, 8k",
        help="Prompt describing commercial poster background/scene",
    )
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        default=[128, 128, 384, 896],
        help="Target bounding box for slogan text in pixels: ymin xmin ymax xmax",
    )
    parser.add_argument("--output", type=str, default="output_product_poster.png", help="Output path")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--font_path", type=str, default=None, help="Path to Unicode font")
    parser.add_argument("--height", type=int, default=1024, help="Output image height")
    parser.add_argument("--width", type=int, default=1024, help="Output image width")
    parser.add_argument("--steps", type=int, default=50, help="Denoise steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    run_product_poster_experiment(
        image_ref=args.image_ref,
        text=args.text,
        prompt=args.prompt,
        box=args.box,
        output_path=args.output,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        font_path=args.font_path,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        device=args.device,
    )
