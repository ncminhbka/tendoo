"""
================================================================================
TENDOO AI - SURFACE PRIOR & CARRIER WHITELIST BENCHMARK
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Objective:
- Test the Surface Prior Hypothesis:
  Compare Group A (Neutral Flat Carrier Surfaces: Ribbon, Glass Panel, Metal Plaque, Solid Frame)
  vs Group B (Dense Text Prior Objects: Feature Info Specs, Business Card, Menu).
- Validate if neutral carrier surfaces eliminate prompt-driven text hallucination zero-shot!

6 Comparative Passes (All at High Density: 768x224 = 672 tokens):
- Pass 1 (Group A): Dải ruy băng phẳng (Flat Ribbon Bar)
- Pass 2 (Group A): Tấm kính mờ phẳng (Frosted Glass Panel)
- Pass 3 (Group A): Tấm bảng kim loại phẳng (Flat Metal Plaque)
- Pass 4 (Group A): Khung viền đơn sắc phẳng (Solid Frame Bar)
- Pass 5 (Group B - Baseline): "Thông tin tính năng" (Feature Info Specs Prior)
- Pass 6 (Group B): "Tấm danh thiếp nhỏ" (Business Card Dense Prior)
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


def encode_product_to_incontext_tokens(
    ae: AutoEncoder,
    image_path: str,
    t_offset: float,
    target_size: int = 1024,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes natural product image into 4D RoPE tokens."""
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


def stitch_grid_comparison(images: list[Image.Image], titles: list[str], output_path: str, cols: int = 3):
    """Stitches images into a 2-row grid comparison panel."""
    w, h = images[0].size
    header_h = 50
    rows = (len(images) + cols - 1) // cols

    cell_w = w
    cell_h = h + header_h
    total_w = cell_w * cols
    total_h = cell_h * rows

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 25))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(resolve_font_path("bevietnam"), size=18)
    except Exception:
        font = ImageFont.load_default()

    for idx, (img, title) in enumerate(zip(images, titles)):
        r = idx // cols
        c = idx % cols
        x_offset = c * cell_w
        y_offset = r * cell_h

        canvas.paste(img, (x_offset, y_offset + header_h))
        draw.rectangle([x_offset, y_offset, x_offset + cell_w, y_offset + header_h], fill=(35, 35, 45), outline=(60, 60, 75))
        bbox = font.getbbox(title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x_offset + (cell_w - tw) // 2
        ty = y_offset + (header_h - th) // 2 - bbox[1]
        draw.text((tx, ty), title, fill=(255, 220, 100), font=font)

    canvas.save(output_path)
    print(f"\n📊 [Stitched Grid Comparison Saved] -> {output_path} ({total_w}x{total_h})")


def run_surface_benchmark():
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

    output_dir = Path("surface_prior_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    ref_img_path = str(ROOT_DIR / "images" / "ref_prod_02.png")

    # Encode Product Reference (4096 tokens at t=40.0)
    prod_tokens, prod_ids = encode_product_to_incontext_tokens(
        ae=ae,
        image_path=ref_img_path,
        t_offset=40.0,
        target_size=1024,
        device=device_ae,
    )

    # Encode 3 Text Glyphs (High Density 768x224 = 672 tokens)
    BW, BH = 768, 224
    g1 = create_glyph_image("ÂM THANH ĐỈNH CAO", BW, BH, resolve_font_path("anton"))
    tok1, ids1 = encode_glyph_to_incontext_tokens(ae, g1, t_offset=10.0, device=device_ae)

    g2 = create_glyph_image("CHỐNG ỒN CHỦ ĐỘNG", BW, BH, resolve_font_path("bevietnam"))
    tok2, ids2 = encode_glyph_to_incontext_tokens(ae, g2, t_offset=20.0, device=device_ae)

    g3 = create_glyph_image("MUA 1 TẶNG 1", BW, BH, resolve_font_path("pacifico"))
    tok3, ids3 = encode_glyph_to_incontext_tokens(ae, g3, t_offset=30.0, device=device_ae)

    ref_tokens_list = [prod_tokens, tok1, tok2, tok3]
    ref_ids_list = [prod_ids, ids1, ids2, ids3]

    all_ref_tokens = torch.cat(ref_tokens_list, dim=1).to(device_dit)
    all_ref_ids = torch.cat(ref_ids_list, dim=1).to(device_dit)

    # Candidate Prompts: Group A (Neutral Carrier Surfaces) vs Group B (Dense Text Priors)
    test_candidates = [
        (
            "Pass 1 [Group A: Dải Ruy Băng Phẳng]",
            "Ảnh chụp quảng cáo thương mại công nghệ cho chiếc tai nghe không dây ở trung tâm, "
            "phía trên có biển hiệu chữ vàng kim loại 3D dập nổi, "
            "ở giữa có một dải ruy băng mạ bạc phẳng in dòng chữ sắc nét, "
            "góc dưới có huy hiệu chữ phát sáng dạ quang rực rỡ, ánh sáng studio sang trọng",
        ),
        (
            "Pass 2 [Group A: Tấm Kính Mờ Phẳng]",
            "Ảnh chụp quảng cáo thương mại công nghệ cho chiếc tai nghe không dây ở trung tâm, "
            "phía trên có biển hiệu chữ vàng kim loại 3D dập nổi, "
            "ở giữa có một tấm kính mờ phẳng in dòng chữ mạ bạc sắc nét, "
            "góc dưới có huy hiệu chữ phát sáng dạ quang rực rỡ, ánh sáng studio sang trọng",
        ),
        (
            "Pass 3 [Group A: Tấm Bảng Kim Loại Phẳng]",
            "Ảnh chụp quảng cáo thương mại công nghệ cho chiếc tai nghe không dây ở trung tâm, "
            "phía trên có biển hiệu chữ vàng kim loại 3D dập nổi, "
            "ở giữa có một tấm bảng kim loại phẳng in dòng chữ mạ bạc sắc nét, "
            "góc dưới có huy hiệu chữ phát sáng dạ quang rực rỡ, ánh sáng studio sang trọng",
        ),
        (
            "Pass 4 [Group A: Khung Viền Đơn Sắc Phẳng]",
            "Ảnh chụp quảng cáo thương mại công nghệ cho chiếc tai nghe không dây ở trung tâm, "
            "phía trên có biển hiệu chữ vàng kim loại 3D dập nổi, "
            "ở giữa có một khung viền đơn sắc phẳng in dòng chữ mạ bạc sắc nét, "
            "góc dưới có huy hiệu chữ phát sáng dạ quang rực rỡ, ánh sáng studio sang trọng",
        ),
        (
            "Pass 5 [Group B: 'Thông tin tính năng' Specs Prior]",
            "Ảnh chụp quảng cáo thương mại công nghệ cho chiếc tai nghe không dây ở trung tâm, "
            "phía trên có biển hiệu chữ vàng kim loại 3D dập nổi, "
            "ở giữa có thông tin tính năng chữ mạ bạc, "
            "góc dưới có huy hiệu chữ phát sáng dạ quang rực rỡ, ánh sáng studio sang trọng",
        ),
        (
            "Pass 6 [Group B: 'Tấm Danh Thiếp' Dense Text Prior]",
            "Ảnh chụp quảng cáo thương mại công nghệ cho chiếc tai nghe không dây ở trung tâm, "
            "phía trên có biển hiệu chữ vàng kim loại 3D dập nổi, "
            "ở giữa có một tấm danh thiếp nhỏ chữ mạ bạc, "
            "góc dưới có huy hiệu chữ phát sáng dạ quang rực rỡ, ánh sáng studio sang trọng",
        ),
    ]

    results = []
    titles = []

    lat_h, lat_w = 64, 64  # 1024x1024
    timesteps = get_schedule(num_steps=50, image_seq_len=lat_h * lat_w)

    for idx, (title, prompt_str) in enumerate(test_candidates):
        print(f"\n" + "-" * 75)
        print(f"▶ [{idx+1}/6] Running {title}...")
        t_start = time.time()
        torch.manual_seed(300)

        # Encode Prompt
        with torch.no_grad():
            txt = text_encoder(["", prompt_str])
            txt, txt_ids = batched_prc_txt(txt)
            txt = txt.to(device_dit)
            txt_ids = txt_ids.to(device_dit)

        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

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
            out_file = output_dir / f"pass_{idx+1}.png"
            img_res.save(out_file)
            print(f"  -> ✅ Saved: {out_file.name} in {time.time() - t_start:.2f}s")
            results.append(img_res)
            titles.append(title)

    stitch_grid_comparison(
        images=results,
        titles=titles,
        output_path=str(output_dir / "SURFACE_PRIOR_WHITELIST_COMPARISON.png"),
        cols=3,
    )


if __name__ == "__main__":
    run_surface_benchmark()
