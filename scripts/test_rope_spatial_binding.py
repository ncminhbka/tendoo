"""
Test Script: Phase 1A - RoPE Spatial Coordinate Binding for Vietnamese Text Rendering
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE

v3 — Thêm instrumentation để chẩn đoán lệch trục tọa độ RoPE (img_ids vs ref_ids),
     và chế độ --debug_mode solid_color để cô lập câu hỏi "có bind được vị trí không"
     khỏi câu hỏi "có đọc đúng glyph phức tạp không" — 2 việc khác nhau đang bị
     trộn lẫn trong kết quả thực nghiệm gần nhất (biển hiệu bị lệch cả vị trí lẫn
     nội dung chữ).

Usage on Remote Server (2x NVIDIA A30 / JupyterLab):
    python scripts/test_rope_spatial_binding.py \
        --prompt "a rustic wooden signboard hanging in front of a cozy coffee shop, highly detailed, 8k" \
        --text "CÀ PHÊ SỮA ĐÁ" \
        --box 256 128 512 896 \
        --output "output_rope_test.png" \
        --model_name "flux.2-klein-base-4b"

    # Chế độ chẩn đoán — khối đỏ đặc thay vì chữ, để cô lập bug định vị:
    python scripts/test_rope_spatial_binding.py \
        --prompt "a rustic wooden signboard hanging in front of a cozy coffee shop, highly detailed, 8k" \
        --text "CÀ PHÊ SỮA ĐÁ" \
        --box 780 128 940 896 \
        --output "exp9_debug_solid.png" \
        --model_name "flux.2-klein-base-4b" \
        --debug_mode solid_color
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
    font_size: int = 64,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    padding_ratio: float = 0.08,
) -> Image.Image:
    """
    Renders a tight-crop Vietnamese glyph image with exact text and diacritics.
    - Handles user manual newlines ('\\n').
    - Auto-wraps long text into optimal multi-line layouts based on box aspect ratio.
    - Uses binary search to find the maximum legible font size that fits without clipping.
    - Guarantees dimensions are divisible by 16 for 16x VAE patchification.
    """
    assert target_width > 0 and target_height > 0
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16

    # 1. Discover valid Unicode font
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
        raise RuntimeError(
            "❌ No valid Unicode font found with Vietnamese support! "
            "Please specify a valid font using --font_path /path/to/font.ttf "
            "(e.g. /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf)."
        )

    # Drawable area with padding
    pad_w = int(target_width * padding_ratio)
    pad_h = int(target_height * padding_ratio)
    max_w = target_width - 2 * pad_w
    max_h = target_height - 2 * pad_h

    # 2. Layout planning: manual newlines vs auto-wrapping
    text = text.replace("\\n", "\n")

    if "\n" in text:
        candidate_layouts = [[line.strip() for line in text.split("\n") if line.strip()]]
    else:
        words = text.split()
        box_ar = target_width / target_height
        candidate_layouts = [[text]]  # 1 line

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

    # Binary search for maximum legible font size
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

    # 3. Draw Centered Text
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


def create_debug_solid_block(
    target_width: int,
    target_height: int,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """
    DIAGNOSTIC ONLY: a flat solid-color reference block, no text/font involved.
    Purpose: isolate "does RoPE binding move the ref content to the right place at
    all" from "does the model correctly interpret complex glyph shapes" — the two
    experiments so far (brick wall, signboard) failed in ways that conflate both
    questions. A solid block has an unambiguous pass/fail: either the color patch
    shows up inside the target box, or it does not.
    """
    assert target_width > 0 and target_height > 0 and target_width % 16 == 0 and target_height % 16 == 0
    return Image.new("RGB", (target_width, target_height), color=color)


def encode_glyph_with_custom_rope(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    canvas_height: int,
    canvas_width: int,
    box: tuple[int, int, int, int],  # (ymin, xmin, ymax, xmax) in pixels, already snapped to 16
    t_offset: float = 10.0,
    device: str = "cuda",
    verbose: bool = True,
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

    if verbose:
        print(
            f"  -> [DIAG] ref_ids constructed as columns (t,h,w,l). "
            f"t={t_offset}, h∈[{h_start},{h_end}), w∈[{w_start},{w_end}), l=0"
        )
        print(f"  -> [DIAG] ref_ids[0, 0]  (first token id)  = {ref_ids[0, 0].tolist()}")
        print(f"  -> [DIAG] ref_ids[0, -1] (last token id)   = {ref_ids[0, -1].tolist()}")

    return ref_tokens, ref_ids


def diagnose_id_ranges(img_ids: torch.Tensor, ref_ids: torch.Tensor, lat_h: int, lat_w: int) -> None:
    """
    DIAGNOSTIC ONLY — does not change any tensor. Prints the min/max of each of the
    4 columns of img_ids vs ref_ids, side by side, so a column-order mismatch (e.g.
    prc_img emitting (t,l,h,w) while we constructed ref_ids as (t,h,w,l)) becomes
    visible from the printed ranges alone, WITHOUT needing to run a full 50-step
    denoise or read prc_img's source. This must be run before trusting any image
    output from this script.

    How to read it:
      - If img_ids' 4 columns have per-column ranges that look like
        [0, 0] / [0, lat_h) / [0, lat_w) / [0, 0] in THAT column order, our
        (t,h,w,l) convention for ref_ids matches — proceed to trust the image.
      - If the column that is constant-zero in img_ids is NOT the same column
        index we used for "t" and "l" in ref_ids, we have a confirmed axis-order
        bug — fix encode_glyph_with_custom_rope's cartesian_prod argument order
        to match, and do NOT trust any prior experiment's spatial conclusions.
    """
    print("\n" + "-" * 80)
    print("[DIAGNOSTIC] Comparing img_ids vs ref_ids coordinate conventions")
    print("-" * 80)
    img_ids_flat = img_ids[0] if img_ids.dim() == 3 else img_ids
    ref_ids_flat = ref_ids[0] if ref_ids.dim() == 3 else ref_ids

    print(f"img_ids shape: {tuple(img_ids.shape)} | ref_ids shape: {tuple(ref_ids.shape)}")
    print(f"Expected canvas latent grid: lat_h={lat_h}, lat_w={lat_w}")

    for col in range(img_ids_flat.shape[-1]):
        col_vals = img_ids_flat[:, col]
        print(
            f"  img_ids col {col}: min={col_vals.min().item():.1f} "
            f"max={col_vals.max().item():.1f} unique={col_vals.unique().numel()}"
        )
    for col in range(ref_ids_flat.shape[-1]):
        col_vals = ref_ids_flat[:, col]
        print(
            f"  ref_ids col {col}: min={col_vals.min().item():.1f} "
            f"max={col_vals.max().item():.1f} unique={col_vals.unique().numel()}"
        )

    # Heuristic check: exactly one img_ids column should be constant (the "t" axis
    # for canvas, typically 0), and its position should match where ref_ids has t=10.
    img_const_cols = [c for c in range(img_ids_flat.shape[-1]) if img_ids_flat[:, c].unique().numel() == 1]
    ref_const_cols = [c for c in range(ref_ids_flat.shape[-1]) if ref_ids_flat[:, c].unique().numel() == 1]
    print(f"  img_ids constant columns: {img_const_cols} (candidates for 't' and/or 'l' axis)")
    print(f"  ref_ids constant columns: {ref_const_cols} (should include our 't' col=0 and 'l' col=3)")
    if img_const_cols and ref_const_cols and set(img_const_cols) != set([0, 3]) & set(ref_const_cols):
        print(
            "  ⚠️  MISMATCH WARNING: img_ids' constant column(s) do not line up with "
            "ref_ids' (t, l) columns (indices 0, 3). This is consistent with an "
            "axis-order bug between prc_img's convention and the (t,h,w,l) order "
            "assumed in encode_glyph_with_custom_rope. Do not trust spatial binding "
            "results until this is resolved."
        )
    print("-" * 80 + "\n")


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
    debug_mode: str = "normal",
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
    print(f"🚀 Starting RoPE Spatial Binding Experiment (Phase 1A) — debug_mode={debug_mode}")
    print(f"Prompt        : {prompt}")
    print(f"Target Text   : {text}")
    print(f"Target Box    : [{ymin}, {xmin}, {ymax}, {xmax}] (Snapped to 16px grid)")
    print(f"Model         : {model_name}")
    print(f"Canvas Size   : {width}x{height}")
    print("=" * 80)

    # Auto-resolve checkpoint paths from persistent-data
    resolve_model_paths(checkpoint_dir)

    torch.manual_seed(seed)

    # 1. Render Glyph Image (or solid-color diagnostic block)
    print("\n[Step 1/5] Rendering reference image (glyph or debug block)...")
    if debug_mode == "solid_color":
        glyph_img = create_debug_solid_block(target_width=box_w, target_height=box_h, color=(255, 0, 0))
        print("  -> [DIAG] Using solid RED block instead of rendered text — isolates position-binding question.")
    else:
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

    # 4. Prepare Canvas & In-Context Reference Glyph
    print("\n[Step 4/5] Preparing Latent Canvas and In-Context Reference Tokens...")
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    # Encode Glyph with standard In-Context Coordinates starting at (0, 0)
    np_arr = np.array(glyph_img, dtype=np.float32) / 127.5 - 1.0
    glyph_dtype = next(ae.parameters()).dtype if hasattr(ae, "parameters") else torch.bfloat16
    glyph_tensor = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=device_ae, dtype=glyph_dtype)
    with torch.no_grad():
        encoded = ae.encode(glyph_tensor)[0]

    _, g_lat_h, g_lat_w = encoded.shape
    t_c = torch.tensor([10.0], dtype=torch.float32, device=device_dit)
    h_c = torch.arange(g_lat_h, dtype=torch.float32, device=device_dit)
    w_c = torch.arange(g_lat_w, dtype=torch.float32, device=device_dit)
    l_c = torch.arange(1, dtype=torch.float32, device=device_dit)

    ref_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0).to(device_dit)
    ref_tokens = rearrange(encoded, "c h w -> (h w) c").unsqueeze(0).to(device_dit, dtype=torch.bfloat16)

    print(f"  -> Canvas Tokens : {img_tokens.shape[1]} tokens (Grid: {lat_h}x{lat_w})")
    print(f"  -> Glyph Tokens  : {ref_tokens.shape[1]} tokens (Grid: {g_lat_h}x{g_lat_w} at t=10.0)")

    # 5. Denoise with Flow Matching ODE (Single Ultra-Fast Pass)
    print("\n[Step 5/5] Running Denoise with In-Context Glyph Conditioning (50 Steps)...")
    timesteps = get_schedule(
        num_steps=num_steps,
        image_seq_len=img_tokens.shape[1],
    )

    with torch.no_grad():
        out_latent = denoise_cfg(
            model=model,
            img=img_tokens,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )

        out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        print("  -> Decoding output image via VAE...")
        out_pixels = ae.decode(out_latent.to(device_ae))
        out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_img = Image.fromarray(out_pixels)
        result_img.save(output_path)
        print(f"  -> Saved output image to: {output_path}")

    print("\n" + "=" * 80)
    print(f"✅ INFERENCE COMPLETE (1-Pass Fast Execution)!")
    print(f"1. Glyph Preview : {glyph_preview_path}")
    print(f"2. Output Image  : {output_path} ({width}x{height})")
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
    parser.add_argument(
        "--debug_mode",
        type=str,
        default="normal",
        choices=["normal", "solid_color"],
        help="'solid_color' replaces the glyph with a flat red block to isolate the "
        "position-binding question from the glyph-legibility question.",
    )

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
        debug_mode=args.debug_mode,
    )
