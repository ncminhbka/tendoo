"""
Test Script: Multi-Text Spatial Binding Experiment
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE

Tests rendering TWO independent Vietnamese text strings at TWO different spatial locations:
- Text 1 (e.g. Signboard on top): Box 1
- Text 2 (e.g. Sidewalk board on bottom): Box 2

Generates a 2-Panel Comparison Image:
[ Panel 1: Baseline (Both start at 0,0) | Panel 2: RoPE Bound (Box 1 & Box 2) ]
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
    denoise_cfg,
    get_schedule,
    prc_img,
)
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import (
    FLUX2_MODEL_INFO,
    find_persistent_data_root,
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


def encode_single_glyph(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    box: tuple[int, int, int, int],
    t_offset: float = 10.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encodes one glyph image and returns:
    1. ref_tokens: (1, N, 128)
    2. rope_ids: (1, N, 4) with exact box coordinates
    3. baseline_ids: (1, N, 4) with default (0, 0) coordinates
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

    # RoPE coordinates matching target box
    t_c = torch.tensor([t_offset], dtype=torch.float32, device=device)
    h_c = torch.arange(h_start, h_end, dtype=torch.float32, device=device)
    w_c = torch.arange(w_start, w_end, dtype=torch.float32, device=device)
    l_c = torch.arange(1, dtype=torch.float32, device=device)
    rope_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0)

    # Baseline coordinates starting at (0, 0)
    h_base = torch.arange(lat_h, dtype=torch.float32, device=device)
    w_base = torch.arange(lat_w, dtype=torch.float32, device=device)
    baseline_ids = torch.cartesian_prod(t_c, h_base, w_base, l_c).unsqueeze(0)

    ref_tokens = rearrange(encoded, "c h w -> (h w) c").unsqueeze(0).to(torch.bfloat16)

    return ref_tokens, rope_ids, baseline_ids


def run_multi_text_experiment(
    prompt: str,
    text1: str,
    box1: list[int],
    text2: str,
    box2: list[int],
    output_path: str = "output_multi_text.png",
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
    # Snap box coordinates to multiples of 16
    ymin1 = (box1[0] // 16) * 16
    xmin1 = (box1[1] // 16) * 16
    ymax1 = (box1[2] // 16) * 16
    xmax1 = (box1[3] // 16) * 16
    box1_snapped = (ymin1, xmin1, ymax1, xmax1)
    box1_h = ymax1 - ymin1
    box1_w = xmax1 - xmin1

    ymin2 = (box2[0] // 16) * 16
    xmin2 = (box2[1] // 16) * 16
    ymax2 = (box2[2] // 16) * 16
    xmax2 = (box2[3] // 16) * 16
    box2_snapped = (ymin2, xmin2, ymax2, xmax2)
    box2_h = ymax2 - ymin2
    box2_w = xmax2 - xmin2

    print("=" * 80)
    print("🚀 Starting Multi-Text Spatial Binding Experiment")
    print(f"Prompt   : {prompt}")
    print(f"Text 1   : '{text1}' in Box 1: {box1_snapped} (Size: {box1_w}x{box1_h})")
    print(f"Text 2   : '{text2}' in Box 2: {box2_snapped} (Size: {box2_w}x{box2_h})")
    print(f"Canvas   : {width}x{height} | Steps: {num_steps} | CFG: {guidance} | Seed: {seed}")
    print("=" * 80)

    # Multi-GPU Device Allocation
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = "cuda:0"
        device_ae = "cuda:1"
        device_te = "cuda:1"
        print(f"  -> Detected {num_gpus} GPUs. Running Pipeline Parallelism: DiT on {device_dit}, VAE/Qwen3 on {device_ae}")
    else:
        device_dit = device
        device_ae = device
        device_te = device
        print(f"  -> Running on single device: {device}")

    torch.manual_seed(seed)

    # 1. Render both Glyphs
    print("\n[Step 1/5] Rendering Glyphs for Text 1 and Text 2...")
    glyph1_img = create_glyph_image(text=text1, target_width=box1_w, target_height=box1_h, font_path=font_path)
    glyph2_img = create_glyph_image(text=text2, target_width=box2_w, target_height=box2_h, font_path=font_path)

    glyph1_path = Path(output_path).stem + "_glyph1_preview.png"
    glyph2_path = Path(output_path).stem + "_glyph2_preview.png"
    glyph1_img.save(glyph1_path)
    glyph2_img.save(glyph2_path)
    print(f"  -> Saved Glyph 1 preview: {glyph1_path}")
    print(f"  -> Saved Glyph 2 preview: {glyph2_path}")

    # 2. Load Models
    print("\n[Step 2/5] Loading FLUX.2 Base DiT & VAE...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)

    # 3. Encode Text Prompt via Qwen3
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

    # 4. Prepare Canvas & Multi-Glyph Tokens
    print("\n[Step 4/5] Encoding Glyphs and Constructing Spatial Coordinates...")
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    # Encode Glyph 1 (t_offset = 10.0)
    ref1_tokens, rope1_ids, base1_ids = encode_single_glyph(
        ae=ae, glyph_img=glyph1_img, box=box1_snapped, t_offset=10.0, device=device_ae
    )
    # Encode Glyph 2 (t_offset = 20.0)
    ref2_tokens, rope2_ids, base2_ids = encode_single_glyph(
        ae=ae, glyph_img=glyph2_img, box=box2_snapped, t_offset=20.0, device=device_ae
    )

    # Combined Tokens & IDs
    rope_tokens = torch.cat([ref1_tokens, ref2_tokens], dim=1).to(device_dit)
    rope_ids = torch.cat([rope1_ids, rope2_ids], dim=1).to(device_dit)

    base_tokens = torch.cat([ref1_tokens, ref2_tokens], dim=1).to(device_dit)
    base_ids = torch.cat([base1_ids, base2_ids], dim=1).to(device_dit)

    print(f"  -> Canvas Tokens  : {img_tokens.shape[1]} tokens (Grid: {lat_h}x{lat_w})")
    print(f"  -> Total Ref Tokens: {rope_tokens.shape[1]} (Glyph 1: {ref1_tokens.shape[1]}, Glyph 2: {ref2_tokens.shape[1]})")

    # 5. Run Denoise
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    with torch.no_grad():
        # RUN 1: RoPE Bound Multi-Text
        print("\n[Step 5/5] [Run 1/2] Running Denoise with RoPE Bound Multi-Text...")
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
        print(f"  -> Saved RoPE-bound result to: {output_path}")

        # RUN 2: Baseline (Default 0,0 for both)
        baseline_path = Path(output_path).stem + "_baseline_default_coords.png"
        print(f"\n[Run 2/2] Running Baseline (Default coordinates at 0,0 for both)...")
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
        result_base.save(baseline_path)
        print(f"  -> Saved Baseline result to: {baseline_path}")

    # Build 2-Panel Comparison
    comparison_path = Path(output_path).stem + "_COMPARISON.png"
    comp_w = width * 2
    comp_h = height
    comp_img = Image.new("RGB", (comp_w, comp_h), color=(20, 20, 20))
    comp_img.paste(result_base, (0, 0))
    comp_img.paste(result_rope, (width, 0))

    draw_comp = ImageDraw.Draw(comp_img)
    lbl_font = ImageFont.load_default()
    draw_comp.text((20, 20), "Panel 1: Baseline [Default (0,0)]", fill=(255, 255, 0), font=lbl_font)
    draw_comp.text((width + 20, 20), "Panel 2: RoPE Bound [Box 1 & Box 2]", fill=(0, 255, 255), font=lbl_font)

    comp_img.save(comparison_path)
    print(f"\n" + "=" * 80)
    print(f"✅ Multi-Text Experiment Completed Successfully!")
    print(f"📸 2-Panel Comparison saved to: {comparison_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Multi-Text Spatial Binding for Vietnamese Text in FLUX.2")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--text1", type=str, required=True, help="First Vietnamese text")
    parser.add_argument(
        "--box1",
        type=int,
        nargs=4,
        default=[128, 128, 384, 896],
        help="Target bounding box for Text 1 in pixels: ymin xmin ymax xmax",
    )
    parser.add_argument("--text2", type=str, required=True, help="Second Vietnamese text")
    parser.add_argument(
        "--box2",
        type=int,
        nargs=4,
        default=[640, 256, 896, 768],
        help="Target bounding box for Text 2 in pixels: ymin xmin ymax xmax",
    )
    parser.add_argument("--output", type=str, default="output_multi_text.png", help="Path to save output image")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Optional path to persistent-data")
    parser.add_argument("--font_path", type=str, default=None, help="Optional path to TTF/OTF font")
    parser.add_argument("--height", type=int, default=1024, help="Output image height")
    parser.add_argument("--width", type=int, default=1024, help="Output image width")
    parser.add_argument("--steps", type=int, default=50, help="Denoise steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    run_multi_text_experiment(
        prompt=args.prompt,
        text1=args.text1,
        box1=args.box1,
        text2=args.text2,
        box2=args.box2,
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
