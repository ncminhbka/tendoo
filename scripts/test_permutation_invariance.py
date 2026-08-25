"""
================================================================================
TENDOO AI - SEQUENCE CONCATENATION PERMUTATION INVARIANCE DIAGNOSTIC
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Objective:
- Verify that 4D RoPE coordinate binding is purely mathematical and invariant
  to the physical concatenation order of reference tokens in sequence (dim=1).
- Tests 3 orderings with identical (t, h, w, l) coordinate IDs:
    Pass A: Normal Order   -> [Canvas, Ref_t10, Ref_t20, Ref_t30]
    Pass B: Inverted Order -> [Canvas, Ref_t30, Ref_t20, Ref_t10]
    Pass C: Shuffled Order -> [Canvas, Ref_t20, Ref_t10, Ref_t30]
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

# Built-in Font Registry
FONT_REGISTRY = {
    "bevietnam": str(ROOT_DIR / "fonts" / "BeVietnamPro-Black.ttf"),
    "playfair": str(ROOT_DIR / "fonts" / "PlayfairDisplay.ttf"),
    "anton": str(ROOT_DIR / "fonts" / "Anton-Regular.ttf"),
    "pacifico": str(ROOT_DIR / "fonts" / "Pacifico-Regular.ttf"),
    "sedgwick": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
}


def resolve_font_path(font_name_or_path: str | None) -> str:
    """Resolves font alias or validates file path."""
    if font_name_or_path:
        key = font_name_or_path.lower().strip()
        if key in FONT_REGISTRY and os.path.exists(FONT_REGISTRY[key]):
            return FONT_REGISTRY[key]
        if os.path.exists(font_name_or_path):
            return font_name_or_path

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
    target_width: int,
    target_height: int,
    font_path: str,
    padding_ratio: float = 0.08,
) -> Image.Image:
    """Renders tight-crop Vietnamese glyph bitmap with automatic line wrapping."""
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16

    pad_w = int(target_width * padding_ratio)
    pad_h = int(target_height * padding_ratio)
    max_w = target_width - 2 * pad_w
    max_h = target_height - 2 * pad_h

    text = text.replace("\\n", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

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

        line_spacing = int(mid_size * 0.18) * (len(lines) - 1)
        total_h += line_spacing

        if max_line_w <= max_w and total_h <= max_h:
            opt_font = test_font
            opt_size = mid_size
            low = mid_size + 1
        else:
            high = mid_size - 1

    if opt_font is None:
        opt_font = ImageFont.truetype(font_path, size=24)
        opt_size = 24

    img = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    line_heights = []
    line_widths = []
    for line in lines:
        bbox = opt_font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(opt_size * 0.18)
    total_block_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    curr_y = (target_height - total_block_h) // 2

    for i, line in enumerate(lines):
        lw = line_widths[i]
        curr_x = (target_width - lw) // 2
        bbox = opt_font.getbbox(line)
        draw.text((curr_x - bbox[0], curr_y - bbox[1]), line, fill=(255, 255, 255), font=opt_font)
        curr_y += line_heights[i] + line_spacing

    return img


def encode_glyph_to_incontext_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes glyph image to 4D RoPE coordinate tokens."""
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


def stitch_horizontal_comparison(images: list[Image.Image], titles: list[str], output_path: str):
    """Stitches multiple images into a labeled horizontal side-by-side comparison panel."""
    w, h = images[0].size
    header_h = 60
    total_w = w * len(images)
    total_h = h + header_h

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 25))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(resolve_font_path("bevietnam"), size=22)
    except Exception:
        font = ImageFont.load_default()

    for idx, (img, title) in enumerate(zip(images, titles)):
        x_offset = idx * w
        canvas.paste(img, (x_offset, header_h))
        draw.rectangle([x_offset, 0, x_offset + w, header_h], fill=(35, 35, 45), outline=(60, 60, 75))
        bbox = font.getbbox(title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x_offset + (w - tw) // 2
        ty = (header_h - th) // 2 - bbox[1]
        draw.text((tx, ty), title, fill=(255, 220, 100), font=font)

    canvas.save(output_path)
    print(f"\n📊 [Stitched Comparison Saved] -> {output_path} ({total_w}x{total_h})")


def run_permutation_benchmark():
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print("🚀 Multi-GPU Mode: DiT on GPU 0, VAE & Qwen3 on GPU 1")
    else:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:0")
        device_te = torch.device("cuda:0")
        print(f"🚀 Single-GPU Mode: {device_dit}")

    print("⏳ Loading FLUX.2 Klein 4B Base Models once into VRAM...")
    t0 = time.time()
    model = load_flow_model("flux.2-klein-base-4b", device=device_dit)
    ae = load_ae("flux.2-klein-base-4b", device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)
    print(f"✅ Models loaded in {time.time() - t0:.2f}s!")

    output_dir = Path("permutation_test_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    prompt = (
        "Thiết kế banner quảng cáo khai trương quán cafe sang trọng hiện đại tông nâu ấm, "
        "biển hiệu lớn trên cao với dòng chữ vàng gold 3D dập nổi tinh xảo, "
        "thông tin mô tả mạ đồng cổ điển ở giữa, góc dưới có các huy hiệu chữ phát sáng nổi bật, "
        "ánh sáng studio điện ảnh, chất lượng cao"
    )

    # Prepare Glyphs
    # 1. Ref 1 (t=10)
    g1 = create_glyph_image("GRAND OPENING", 640, 192, resolve_font_path("anton"))
    tok1, ids1 = encode_glyph_to_incontext_tokens(ae, g1, t_offset=10.0, device=device_ae)

    # 2. Ref 2 (t=20)
    g2 = create_glyph_image("CÀ PHÊ RANG MỘC", 576, 160, resolve_font_path("bevietnam"))
    tok2, ids2 = encode_glyph_to_incontext_tokens(ae, g2, t_offset=20.0, device=device_ae)

    # 3. Ref 3 (t=30)
    g3 = create_glyph_image("MUA 1 TẶNG 1", 512, 160, resolve_font_path("pacifico"))
    tok3, ids3 = encode_glyph_to_incontext_tokens(ae, g3, t_offset=30.0, device=device_ae)

    ref_items = {
        "t10": (tok1, ids1),
        "t20": (tok2, ids2),
        "t30": (tok3, ids3),
    }

    # Text prompt encoding (shared)
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    permutations = [
        ("Pass A: Normal [t10 -> t20 -> t30]", ["t10", "t20", "t30"]),
        ("Pass B: Inverted [t30 -> t20 -> t10]", ["t30", "t20", "t10"]),
        ("Pass C: Shuffled [t20 -> t10 -> t30]", ["t20", "t10", "t30"]),
    ]

    results = []
    titles = []

    for name, order in permutations:
        print(f"\n▶ Running {name}...")
        t_start = time.time()
        torch.manual_seed(200)

        # Concatenate in specified order
        tokens_ordered = [ref_items[k][0] for k in order]
        ids_ordered = [ref_items[k][1] for k in order]
        all_ref_tokens = torch.cat(tokens_ordered, dim=1).to(device_dit)
        all_ref_ids = torch.cat(ids_ordered, dim=1).to(device_dit)

        # Init canvas
        lat_h, lat_w = 64, 64  # 1024x1024
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=50, image_seq_len=img_tokens.shape[1])

        with torch.no_grad():
            out_latent = denoise_cfg(
                model=model,
                img=img_tokens,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=4.0,
                img_cond_seq=all_ref_tokens,
                img_cond_seq_ids=all_ref_ids,
            )

            out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
            out_pixels = ae.decode(out_latent.to(device_ae))
            out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            img_res = Image.fromarray(out_pixels)
            out_file = output_dir / f"{name.split(':')[0].lower().replace(' ', '_')}.png"
            img_res.save(out_file)
            print(f"  -> ✅ Saved: {out_file.name} in {time.time() - t_start:.2f}s")
            results.append(img_res)
            titles.append(name)

    stitch_horizontal_comparison(
        images=results,
        titles=titles,
        output_path=str(output_dir / "PERMUTATION_INVARIANCE_COMPARISON.png"),
    )


if __name__ == "__main__":
    run_permutation_benchmark()
