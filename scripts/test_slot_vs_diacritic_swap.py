"""
================================================================================
TENDOO AI - SLOT POSITION (t=20) VS DIACRITIC COMPLEXITY DISENTANGLEMENT BENCHMARK
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Objective:
- Decisive experiment to disentangle whether failure of "CHỐNG ỒN CHỦ ĐỘNG" is:
  1. Position-Specific Bottleneck at t=20 (RoPE attention routing/decay)
  2. OR Glyph Diacritic Complexity (4 heavy consecutive diacritics: Ố - Ồ - Ủ - Ộ)

Comparative Passes (All at High Density: 768x224 = 672 tokens):
- Pass 1 (Baseline): t10="ÂM THANH ĐỈNH CAO" | t20="CHỐNG ỒN CHỦ ĐỘNG" | t30="MUA 1 TẶNG 1"
- Pass 2 (Swap t20 & t30): t10="ÂM THANH ĐỈNH CAO" | t20="MUA 1 TẶNG 1" | t30="CHỐNG ỒN CHỦ ĐỘNG"
- Pass 3 (Swap t10 & t20): t10="CHỐNG ỒN CHỦ ĐỘNG" | t20="ÂM THANH ĐỈNH CAO" | t30="MUA 1 TẶNG 1"
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


def stitch_horizontal_comparison(images: list[Image.Image], titles: list[str], output_path: str):
    """Stitches multiple images into a labeled horizontal side-by-side comparison panel."""
    w, h = images[0].size
    header_h = 60
    total_w = w * len(images)
    total_h = h + header_h

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 25))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(resolve_font_path("bevietnam"), size=20)
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


def run_slot_swap_benchmark():
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

    output_dir = Path("slot_swap_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    ref_img_path = str(ROOT_DIR / "images" / "ref_prod_02.png")
    prompt = (
        "Ảnh chụp quảng cáo thương mại công nghệ cho chiếc tai nghe chụp tai không dây màu đen sang trọng đặt ở vị trí trung tâm, "
        "phía trên có biển hiệu lớn chữ vàng kim loại 3D dập nổi sắc nét, "
        "thông tin tính năng chữ mạ bạc ở giữa, góc dưới có các huy hiệu chữ phát sáng dạ quang rực rỡ, "
        "ánh sáng studio điện ảnh tương phản cao, siêu chi tiết"
    )

    # Encode Text Prompt (shared)
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    # Encode Product (4096 tokens at t=40.0)
    prod_tokens, prod_ids = encode_product_to_incontext_tokens(
        ae=ae,
        image_path=ref_img_path,
        t_offset=40.0,
        target_size=1024,
        device=device_ae,
    )

    # All text blocks use 768x224 (672 tokens)
    BW, BH = 768, 224

    test_passes = [
        (
            "Pass 1: Baseline [t10=Âm thanh | t20=Chống ồn | t30=Mua 1]",
            [
                {"text": "ÂM THANH ĐỈNH CAO", "t": 10.0, "font": "anton"},
                {"text": "CHỐNG ỒN CHỦ ĐỘNG", "t": 20.0, "font": "bevietnam"},
                {"text": "MUA 1 TẶNG 1", "t": 30.0, "font": "pacifico"},
            ],
        ),
        (
            "Pass 2: Swap t20 & t30 [t10=Âm thanh | t20=Mua 1 | t30=Chống ồn]",
            [
                {"text": "ÂM THANH ĐỈNH CAO", "t": 10.0, "font": "anton"},
                {"text": "MUA 1 TẶNG 1", "t": 20.0, "font": "pacifico"},  # CTA moved to t=20
                {"text": "CHỐNG ỒN CHỦ ĐỘNG", "t": 30.0, "font": "bevietnam"},  # Subtitle moved to t=30
            ],
        ),
        (
            "Pass 3: Swap t10 & t20 [t10=Chống ồn | t20=Âm thanh | t30=Mua 1]",
            [
                {"text": "CHỐNG ỒN CHỦ ĐỘNG", "t": 10.0, "font": "bevietnam"},  # Subtitle moved to t=10
                {"text": "ÂM THANH ĐỈNH CAO", "t": 20.0, "font": "anton"},  # Headline moved to t=20
                {"text": "MUA 1 TẶNG 1", "t": 30.0, "font": "pacifico"},
            ],
        ),
    ]

    results = []
    titles = []

    for name, text_configs in test_passes:
        print(f"\n" + "-" * 75)
        print(f"▶ Running {name}...")
        t_start = time.time()
        torch.manual_seed(300)

        ref_tokens_list = [prod_tokens]
        ref_ids_list = [prod_ids]

        for tc in text_configs:
            f_p = resolve_font_path(tc["font"])
            g_img = create_glyph_image(tc["text"], BW, BH, f_p)
            tok, ids = encode_glyph_to_incontext_tokens(ae, g_img, tc["t"], device=device_ae)
            ref_tokens_list.append(tok)
            ref_ids_list.append(ids)
            print(f"    🔤 [Slot t={tc['t']}] '{tc['text']}' -> {BW}x{BH} ({tok.shape[1]} tokens)")

        all_ref_tokens = torch.cat(ref_tokens_list, dim=1).to(device_dit)
        all_ref_ids = torch.cat(ref_ids_list, dim=1).to(device_dit)

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
            titles.append(name.split(":")[0])

    stitch_horizontal_comparison(
        images=results,
        titles=titles,
        output_path=str(output_dir / "SLOT_VS_DIACRITIC_SWAP_COMPARISON.png"),
    )


if __name__ == "__main__":
    run_slot_swap_benchmark()
