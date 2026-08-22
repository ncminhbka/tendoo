"""
Test Script: Phase 1A - RoPE Spatial Coordinate Binding for Vietnamese Text Rendering
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE

Usage on Remote Server (2x NVIDIA A30 / JupyterLab):
    python scripts/test_rope_spatial_binding.py \
        --prompt "a rustic wooden signboard hanging in front of a cozy coffee shop, highly detailed, 8k" \
        --text "CÀ PHÊ SỮA ĐÁ" \
        --box 256 128 512 896 \
        --output "output_rope_test.png" \
        --model_name "flux.2-klein-base-4b"
"""

import argparse
import math
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
    denoise_cfg,
    get_schedule,
    prc_img,
)
from flux2.text_encoder import Qwen3Embedder
from flux2.util import (
    FLUX2_MODEL_INFO,
    load_ae,
    load_flow_model,
    load_text_encoder,
    find_persistent_data_root,
)


def create_glyph_image(
    text: str,
    target_width: int,
    target_height: int,
    font_path: str | None = None,
    font_size: int = 72,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Renders a tight-crop Vietnamese glyph image with exact text and diacritics.
    Dimensions must be divisible by 16 (for 16x VAE patchification).
    """
    assert target_width > 0 and target_height > 0 and target_width % 16 == 0 and target_height % 16 == 0

    img = Image.new("RGB", (target_width, target_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Load font
    font = None
    candidate_fonts = [
        font_path,
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for fp in candidate_fonts:
        if fp and os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                print(f"  -> Using Unicode font: {fp}")
                break
            except Exception:
                continue

    if font is None:
        raise RuntimeError(
            "❌ No valid Unicode font found with Vietnamese support! "
            "Please specify a valid font using --font_path /path/to/font.ttf "
            "(e.g. /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf)."
        )

    # Calculate text bounding box to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Auto scale font if text is larger than target box
    if text_w > target_width * 0.9 or text_h > target_height * 0.9:
        scale_factor = min((target_width * 0.9) / max(text_w, 1), (target_height * 0.9) / max(text_h, 1))
        new_font_size = max(int(font_size * scale_factor), 16)
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, new_font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x_pos = (target_width - text_w) // 2 - bbox[0]
    y_pos = (target_height - text_h) // 2 - bbox[1]

    draw.text((x_pos, y_pos), text, font=font, fill=text_color)
    return img


def encode_glyph_with_custom_rope(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    canvas_height: int,
    canvas_width: int,
    box: tuple[int, int, int, int],  # (ymin, xmin, ymax, xmax) in pixels, already snapped to 16
    t_offset: float = 10.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes glyph image via VAE and constructs 4D RoPE IDs matching target canvas coordinates.

    Math:
        Canvas Latent Grid: H_c = canvas_height // 16, W_c = canvas_width // 16
        Box in Latent Grid:
            h_start = ymin // 16, h_end = ymax // 16
            w_start = xmin // 16, w_end = xmax // 16
        Ref RoPE IDs:
            t = t_offset (10.0)
            h = arange(h_start, h_end)   <-- EXACT ALIGNMENT (Delta h = 0)
            w = arange(w_start, w_end)   <-- EXACT ALIGNMENT (Delta w = 0)
            l = 0
    """
    ymin, xmin, ymax, xmax = box
    h_start = ymin // 16
    h_end = ymax // 16
    w_start = xmin // 16
    w_end = xmax // 16

    expected_h = h_end - h_start
    expected_w = w_end - w_start
    assert expected_h > 0 and expected_w > 0, "Invalid target box dimensions"

    # Preprocess glyph image to tensor [-1, 1] in bfloat16
    np_arr = np.array(glyph_img, dtype=np.float32) / 127.5 - 1.0
    glyph_dtype = next(ae.parameters()).dtype if hasattr(ae, "parameters") else torch.bfloat16
    glyph_tensor = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=glyph_dtype)

    # Encode with VAE
    with torch.no_grad():
        encoded = ae.encode(glyph_tensor)[0]  # Shape: (128, H_lat, W_lat)

    _, lat_h, lat_w = encoded.shape
    assert lat_h == expected_h and lat_w == expected_w, (
        f"Encoded latent shape ({lat_h}, {lat_w}) does not match expected box latent shape ({expected_h}, {expected_w})"
    )

    # Construct Custom RoPE Coordinate IDs
    t_coord = torch.tensor([t_offset], dtype=torch.float32, device=device)
    h_coord = torch.arange(h_start, h_end, dtype=torch.float32, device=device)
    w_coord = torch.arange(w_start, w_end, dtype=torch.float32, device=device)
    l_coord = torch.arange(1, dtype=torch.float32, device=device)

    # Cartesian product for (t, h, w, l)
    ref_ids = torch.cartesian_prod(t_coord, h_coord, w_coord, l_coord)  # Shape: (lat_h * lat_w, 4)

    # Flatten tokens: (128, H, W) -> (H*W, 128)
    ref_tokens = rearrange(encoded, "c h w -> (h w) c")

    # Add batch dimension
    ref_tokens = ref_tokens.unsqueeze(0).to(torch.bfloat16)  # (1, num_ref_tokens, C)
    ref_ids = ref_ids.unsqueeze(0)  # (1, num_ref_tokens, 4)

    return ref_tokens, ref_ids


def resolve_model_paths(custom_dir: str | None = None):
    """
    Auto-detects model checkpoints in persistent-data directory structure:
    ~/persistent-data/FLUX.2-klein-base-4B/
    """
    p_root = custom_dir or find_persistent_data_root()
    if p_root:
        print(f"  -> Detected persistent checkpoint directory: {p_root}")
        dit_path = os.path.join(p_root, "flux-2-klein-base-4b.safetensors")
        if os.path.exists(dit_path):
            os.environ["KLEIN_4B_BASE_MODEL_PATH"] = dit_path
            print(f"     Found DiT weights: {dit_path}")

        ae_path = os.path.join(p_root, "vae", "diffusion_pytorch_model.safetensors")
        if not os.path.exists(ae_path):
            ae_path = os.path.join(p_root, "ae.safetensors")
        if os.path.exists(ae_path):
            os.environ["AE_MODEL_PATH"] = ae_path
            print(f"     Found AE weights : {ae_path}")

        te_dir = os.path.join(p_root, "text_encoder")
        if os.path.exists(te_dir):
            os.environ["QWEN3_4B_DIR"] = te_dir
            print(f"     Found Text Encoder dir: {te_dir}")


def run_experiment(
    prompt: str,
    text: str,
    box: list[int],
    output_path: str,
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
    # Snap box coordinates to multiples of 16 immediately (prevents any rounding mismatch)
    ymin = (box[0] // 16) * 16
    xmin = (box[1] // 16) * 16
    ymax = (box[2] // 16) * 16
    xmax = (box[3] // 16) * 16
    box_h = ymax - ymin
    box_w = xmax - xmin
    assert box_h > 0 and box_w > 0, f"Invalid bounding box: {[ymin, xmin, ymax, xmax]}"

    print("=" * 80)
    print(f"🚀 Starting RoPE Spatial Binding Experiment (Phase 1A)")
    print(f"Prompt        : {prompt}")
    print(f"Target Text   : {text}")
    print(f"Target Box    : [{ymin}, {xmin}, {ymax}, {xmax}] (Snapped to 16px grid)")
    print(f"Model         : {model_name}")
    print(f"Canvas Size   : {width}x{height}")
    print("=" * 80)

    # Auto-resolve checkpoint paths from persistent-data
    resolve_model_paths(checkpoint_dir)

    torch.manual_seed(seed)

    # 1. Render Glyph Image
    print("\n[Step 1/5] Rendering tight-crop Vietnamese glyph bitmap...")
    glyph_img = create_glyph_image(
        text=text,
        target_width=box_w,
        target_height=box_h,
        font_path=font_path,
        font_size=64,
    )
    glyph_preview_path = Path(output_path).stem + "_glyph_preview.png"
    glyph_img.save(glyph_preview_path)
    print(f"  -> Saved glyph reference preview: {glyph_preview_path}")

    # 2. Setup Multi-GPU or Single-GPU Devices
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        device_dit = "cuda:0"
        device_te = "cuda:1"
        device_ae = "cuda:1"
        print(f"\n[Step 2/5] Multi-GPU Mode Activated ({num_gpus}x GPUs detected):")
        print(f"  -> GPU 0 (cuda:0): DiT Base 4B ({model_name})")
        print(f"  -> GPU 1 (cuda:1): Qwen3-4B-FP8 Text Encoder & VAE AutoEncoder")
    else:
        device_dit = device
        device_te = device
        device_ae = device
        print(f"\n[Step 2/5] Loading models to {device}...")

    ae = load_ae(model_name, device=device_ae)
    model = load_flow_model(model_name, device=device_dit)
    text_encoder = load_text_encoder(model_name, device=device_te)

    # 3. Encode Text Prompt (Single synchronized batch for identical seq_len padding)
    print("\n[Step 3/5] Extracting context from Qwen3 text encoder...")
    with torch.no_grad():
        txt = text_encoder(["", prompt]).to(device_dit)  # Batch size 2: [uncond, cond]
        _, txt_ids = batched_prc_txt(txt)
        txt_ids = txt_ids.to(device_dit)

    # Optional memory cleanup if single GPU
    if num_gpus < 2:
        print("  -> Freeing Qwen3 Text Encoder from single GPU VRAM...")
        del text_encoder
        torch.cuda.empty_cache()

    # 4. Prepare Canvas & RoPE Bound Glyph
    print("\n[Step 4/5] Preparing Latent Canvas and RoPE Spatial Coordinates...")
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    # Encode Glyph with RoPE Coordinate Alignment
    ref_tokens, ref_ids = encode_glyph_with_custom_rope(
        ae=ae,
        glyph_img=glyph_img,
        canvas_height=height,
        canvas_width=width,
        box=(ymin, xmin, ymax, xmax),
        t_offset=10.0,
        device=device_ae,
    )
    ref_tokens = ref_tokens.to(device_dit)
    ref_ids = ref_ids.to(device_dit)

    print(f"  -> Canvas Tokens : {img_tokens.shape[1]} tokens (Grid: {lat_h}x{lat_w})")
    print(f"  -> Glyph Tokens  : {ref_tokens.shape[1]} tokens (Box: {box_h//16}x{box_w//16})")
    print(f"  -> RoPE h range  : [{ymin//16}, {ymax//16}), w range: [{xmin//16}, {xmax//16})")

    # 5. Denoise with Flow Matching ODE (RoPE Bound Experiment)
    print("\n[Step 5/5] Running Denoise with RoPE Bound Glyph Conditioning...")
    timesteps = get_schedule(
        num_steps=num_steps,
        image_seq_len=img_tokens.shape[1],
    )

    with torch.no_grad():
        out_latent = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )

        out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        print("  -> Decoding RoPE bound output image via VAE...")
        out_pixels = ae.decode(out_latent.to(device_ae))
        out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_rope = Image.fromarray(out_pixels)
        result_rope.save(output_path)
        print(f"  -> Saved RoPE-bound result to: {output_path}")

        # Baseline 1: Unbound coordinates (starts at 0, 0)
        baseline_path = Path(output_path).stem + "_baseline_default_coords.png"
        print(f"\n[Comparison] Running Baseline 1 (Default ref coordinates at 0,0)...")
        t_c = torch.tensor([10.0], dtype=torch.float32, device=device_dit)
        h_c = torch.arange(box_h // 16, dtype=torch.float32, device=device_dit)
        w_c = torch.arange(box_w // 16, dtype=torch.float32, device=device_dit)
        l_c = torch.arange(1, dtype=torch.float32, device=device_dit)
        baseline_ref_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0).to(device_dit)

        out_latent_base = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=baseline_ref_ids,
        )
        out_latent_base = rearrange(out_latent_base, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_pixels_base = ae.decode(out_latent_base.to(device_ae))
        out_pixels_base = ((out_pixels_base[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_base = Image.fromarray(out_pixels_base)
        result_base.save(baseline_path)
        print(f"  -> Saved Baseline 1 result to: {baseline_path}")

        # Baseline 2: Pure Prompt (No Glyph Conditioning)
        pure_prompt_path = Path(output_path).stem + "_pure_prompt_no_glyph.png"
        print(f"\n[Comparison] Running Baseline 2 (Pure Prompt, No Glyph Conditioning)...")
        out_latent_pure = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=None,
            img_cond_seq_ids=None,
        )
        out_latent_pure = rearrange(out_latent_pure, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_pixels_pure = ae.decode(out_latent_pure.to(device_ae))
        out_pixels_pure = ((out_pixels_pure[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_pure = Image.fromarray(out_pixels_pure)
        result_pure.save(pure_prompt_path)
        print(f"  -> Saved Baseline 2 (Pure Prompt) result to: {pure_prompt_path}")

        # Create 3-panel Side-by-Side Comparison: [Pure Prompt | Baseline (0,0) | RoPE Bound (target box)]
        comparison_path = Path(output_path).stem + "_COMPARISON.png"
        comp_img = Image.new("RGB", (width * 3, height), color=(30, 30, 30))
        comp_img.paste(result_pure, (0, 0))
        comp_img.paste(result_base, (width, 0))
        comp_img.paste(result_rope, (width * 2, 0))
        comp_img.save(comparison_path)
        print(f"  -> 🌟 Saved 3-Panel Side-by-Side Comparison to: {comparison_path}")

    print("\n" + "=" * 80)
    print(f"✅ EXPERIMENT COMPLETE!")
    print(f"1. Glyph Preview       : {glyph_preview_path}")
    print(f"2. Pure Prompt (No Ref): {pure_prompt_path}")
    print(f"3. Baseline (Coords 0,0): {baseline_path}")
    print(f"4. RoPE Bound (Target) : {output_path}")
    print(f"5. 3-Panel Comparison  : {comparison_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RoPE Spatial Binding for Vietnamese Text in FLUX.2")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--text", type=str, required=True, help="Vietnamese text to render into the image")
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        default=[256, 128, 448, 896],
        help="Target bounding box on canvas in pixels: ymin xmin ymax xmax",
    )
    parser.add_argument("--output", type=str, default="output_rope_vietnamese.png", help="Path to save output image")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Optional explicit path to pre-downloaded persistent-data directory",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default=None,
        help="Optional path to TTF/OTF Unicode font with Vietnamese support",
    )
    parser.add_argument("--height", type=int, default=1024, help="Output image height (multiple of 16)")
    parser.add_argument("--width", type=int, default=1024, help="Output image width (multiple of 16)")
    parser.add_argument("--steps", type=int, default=50, help="Number of Euler ODE denoise steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/cuda:0)")

    args = parser.parse_args()

    run_experiment(
        prompt=args.prompt,
        text=args.text,
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
