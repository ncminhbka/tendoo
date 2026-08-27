"""
Diagnostic Script: Character-Level Nyquist Resolution & Anti-Aliasing Threshold Probe
Model: FLUX.2-klein-base-4B + Qwen3-4B-FP8 + AE

Purpose:
    Quantitatively isolates the exact boundary between:
    - Sub-Nyquist Aliasing (Nét gai, dấu răng cưa do diacritic < 0.5 latent tokens)
    - The Golden Smoothness Threshold (Nét mịn lụa, dấu phụ sắc nét 100% khi diacritic >= 0.70 latent tokens)
    - Ultra-HD Display Tier (Chữ 3D dập nổi hoàn hảo)

Usage:
    # 1. Print Mathematical Metric Matrix across all primary Vietnamese Unicode fonts:
    python scripts/probe_glyph_resolution_threshold.py --mode analyze

    # 2. Run 3-Tier Resolution Comparison on 2x A30 Server (4-line Vietnamese text):
    python scripts/probe_glyph_resolution_threshold.py --mode generate --text_type 4lines
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Auto-configure PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Set UTF-8 encoding for stdout on all platforms
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Offline Mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


from PIL import Image, ImageDraw, ImageFont
import numpy as np



FONT_PRESETS = {
    "bevietnam": str(ROOT_DIR / "fonts" / "BeVietnamPro-Black.ttf"),
    "playfair": str(ROOT_DIR / "fonts" / "PlayfairDisplay.ttf"),
    "anton": str(ROOT_DIR / "fonts" / "Anton-Regular.ttf"),
    "oswald": str(ROOT_DIR / "fonts" / "Oswald.ttf"),
}


def analyze_font_resolution_matrix():
    """Calculates character, diacritic, and stroke metrics in pixels and latent tokens."""
    sizes = [24, 30, 36, 42, 48, 56, 64]
    
    print("=" * 100)
    print(" 📊 TENDOO AI: CHARACTER-LEVEL NYQUIST RESOLUTION MATRIX (FLUX.2 16x LATENT GRID)")
    print("=" * 100)
    print(f"{'Font Name':<20} | {'Size':<6} | {'Char H (px/lat)':<18} | {'Diacritic H (px/lat)':<22} | {'Dot Below (px/lat)':<20} | {'Status':<15}")
    print("-" * 100)

    for font_key, font_path in FONT_PRESETS.items():
        if not os.path.exists(font_path):
            continue
        for sz in sizes:
            try:
                f = ImageFont.truetype(font_path, sz)
            except Exception:
                continue

            bbox_O = f.getbbox("O")
            bbox_O_hat = f.getbbox("Ô")
            h_O = bbox_O[3] - bbox_O[1]
            h_hat = (bbox_O_hat[3] - bbox_O_hat[1]) - h_O

            bbox_E_hat = f.getbbox("Ê")
            bbox_E_dot = f.getbbox("Ệ")
            dot_h = bbox_E_dot[3] - bbox_E_hat[3]

            lat_char = h_O / 16.0
            lat_hat = h_hat / 16.0
            lat_dot = dot_h / 16.0

            # Nyquist Classification
            if lat_hat < 0.50:
                status = "🔴 Sub-Nyquist (Gai nét)"
            elif lat_hat < 0.70:
                status = "🟡 Marginal (Mép mờ nhẹ)"
            else:
                status = "🟢 Silk-Smooth (Mịn 100%)"

            print(
                f"{font_key:<20} | {sz:>3}pt  | "
                f"{h_O:>2}px ({lat_char:4.2f} lat)    | "
                f"{h_hat:>2}px ({lat_hat:4.2f} lat)       | "
                f"{dot_h:>2}px ({lat_dot:4.2f} lat)     | "
                f"{status}"
            )
        print("-" * 100)


def create_precise_glyph_box(
    text_lines: list[str],
    font_path: str,
    font_size: int,
    line_spacing_ratio: float = 0.28,
    pad_ratio: float = 0.15,
) -> Image.Image:
    """Renders a pixel-accurate multi-line glyph bitmap at exact target font size."""
    f = ImageFont.truetype(font_path, font_size)

    line_boxes = [f.getbbox(l) for l in text_lines]
    widths = [b[2] - b[0] for b in line_boxes]
    heights = [b[3] - b[1] for b in line_boxes]

    max_w = max(widths)
    spacing = int(font_size * line_spacing_ratio)
    total_h = sum(heights) + spacing * (len(text_lines) - 1)

    pad_x = int(font_size * pad_ratio)
    pad_y = int(font_size * pad_ratio)

    canvas_w = int(np.ceil((max_w + 2 * pad_x) / 16.0) * 16)
    canvas_h = int(np.ceil((total_h + 2 * pad_y) / 16.0) * 16)

    img = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = pad_y
    for i, line in enumerate(text_lines):
        w = widths[i]
        curr_x = (canvas_w - w) // 2
        bbox = line_boxes[i]
        draw.text((curr_x - bbox[0], curr_y - bbox[1]), line, font=f, fill=(255, 255, 255))
        curr_y += heights[i] + spacing

    return img


def run_3tier_resolution_probe(
    text_type: str = "4lines",
    output_dir: str = "output_resolution_probe",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    """Executes the 3-Tier Resolution Comparison on Remote Server."""
    import torch
    from einops import rearrange
    from flux2.autoencoder import AutoEncoder
    from flux2.model import Flux2
    from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
    from flux2.text_encoder import load_qwen3_embedder
    from flux2.util import load_ae, load_flow_model

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)


    if text_type == "4lines":
        lines = [
            "Sông Mã xa rồi Tây Tiến ơi",
            "Nhớ về rừng núi nhớ chơi vơi.",
            "Sài Khao sương lấp đoàn quân mỏi,",
            "Mường Lát hoa về trong đêm hơi.",
        ]
        prompt = (
            "Bức vách đá sa thạch phẳng cổ kính sừng sững ở tiền cảnh góc bên, "
            "bốn câu thơ chữ dập nổi mạ vàng đồng cổ sắc nét trên mặt đá phủ rêu phong, "
            "hậu cảnh núi non Tây Bắc hùng vĩ hoàng hôn le lói mây mù, ánh sáng ven cinematic studio"
        )
    else:
        lines = ["TÔI YÊU VIỆT NAM"]
        prompt = (
            "Một cuốn sổ tay vintage bìa da màu nâu ấm đặt trên bàn gỗ mộc mạc cạnh tách trà hoa cúc, "
            "góc dưới bìa da có dòng chữ nhỏ dập chìm mạ vàng đồng tinh tế thanh lịch, ánh sáng ven êm dịu, chụp cận cảnh 35mm"
        )

    # 3 Experimental Tiers:
    tiers = [
        {"id": "tier1_subnyquist_28pt", "name": "Tier 1: Sub-Nyquist (28pt - Dấu < 0.50 lat)", "font_size": 28},
        {"id": "tier2_gold_threshold_44pt", "name": "Tier 2: Gold Threshold (44pt - Dấu >= 0.70 lat)", "font_size": 44},
        {"id": "tier3_ultrahd_56pt", "name": "Tier 3: Ultra-HD Display (56pt - Dấu >= 0.85 lat)", "font_size": 56},
    ]

    print("=" * 80)
    print(" 🚀 TENDOO AI: 3-TIER RESOLUTION PROBE BENCHMARK")
    print(f"📝 Text: {len(lines)} line(s) | Prompt: {prompt[:60]}...")
    print("=" * 80)

    # Device
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
    else:
        device_dit = device_ae = device_te = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Models
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    # Encode prompt
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    if num_gpus < 2:
        del text_encoder
        torch.cuda.empty_cache()

    width, height = 576, 1024
    lat_h, lat_w = height // 16, width // 16
    num_canvas = lat_h * lat_w

    # Fixed seed for strictly identical scene geometry across all tiers
    torch.manual_seed(42)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    timesteps = get_schedule(num_steps=50, image_seq_len=num_canvas)

    results = []

    for t in tiers:
        print(f"\n▶️ Running {t['name']}...")
        glyph = create_precise_glyph_box(
            text_lines=lines,
            font_path=FONT_PRESETS["playfair"] if text_type == "4lines" else FONT_PRESETS["bevietnam"],
            font_size=t["font_size"],
        )
        glyph_path = out_path / f"{t['id']}_glyph_preview.png"
        glyph.save(glyph_path)

        gw, gh = glyph.size
        glat_w, glat_h = gw // 16, gh // 16
        tokens = glat_w * glat_h
        print(f"   • Glyph Dimensions: {gw}x{gh}px ({glat_w}x{glat_h} = {tokens} latent tokens)")

        np_arr = np.array(glyph, dtype=np.float32) / 127.5 - 1.0
        g_tensor = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=device_ae, dtype=torch.bfloat16)
        with torch.no_grad():
            g_latent = ae.encode(g_tensor)[0]

        ref_tokens = rearrange(g_latent, "c h w -> (h w) c").unsqueeze(0).to(device_dit, dtype=torch.bfloat16)
        t_c = torch.tensor([10.0], dtype=torch.float32, device=device_dit)
        h_c = torch.arange(glat_h, dtype=torch.float32, device=device_dit)
        w_c = torch.arange(glat_w, dtype=torch.float32, device=device_dit)
        l_c = torch.arange(1, dtype=torch.float32, device=device_dit)
        ref_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0).to(device_dit)

        t_start = time.time()
        with torch.no_grad():
            out_latent = denoise_cfg(
                model=model,
                img=img_tokens.clone(),
                img_ids=img_ids.clone(),
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=4.5,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
            out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
            out_pixels = ae.decode(out_latent.to(device_ae))
            out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            res_img = Image.fromarray(out_pixels)

        img_file = out_path / f"{t['id']}.png"
        res_img.save(img_file)
        elapsed = time.time() - t_start
        print(f"   ✅ Saved: {img_file.name} in {elapsed:.2f}s")
        results.append((t, res_img))

    print("\n" + "=" * 80)
    print(f"🎉 RESOLUTION PROBE COMPLETED! Images saved in: {out_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tendoo AI - Glyph Resolution & Anti-Aliasing Probe")
    parser.add_argument("--mode", type=str, default="analyze", choices=["analyze", "generate"], help="Execution mode")
    parser.add_argument("--text_type", type=str, default="4lines", choices=["4lines", "single"], help="Probe text type")
    parser.add_argument("--output_dir", type=str, default="output_resolution_probe", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="Model name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_font_resolution_matrix()
    else:
        run_3tier_resolution_probe(
            text_type=args.text_type,
            output_dir=args.output_dir,
            model_name=args.model_name,
            checkpoint_dir=args.checkpoint_dir,
        )
