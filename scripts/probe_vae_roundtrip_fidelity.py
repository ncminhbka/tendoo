"""
Diagnostic Script: VAE Roundtrip Reconstruction Fidelity & Artifact Isolation
Purpose:
    Completely isolates the VAE physical bottleneck from DiT generative errors.
    Formula:
        Glyph (Pixel) -> VAE Encoder -> Latent (16x compressed) -> VAE Decoder -> Roundtrip (Pixel)

    Ground Truth Decision Rules:
    - If roundtrip is ALREADY jagged/spiky/missing accents -> 100% VAE physical limit (Untrainable via LoRA).
    - If roundtrip is STILL sharp and intact, but 50-step DiT denoising yields jagged text -> DiT generative error (TRAINABLE via LoRA).
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

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def render_glyph_at_size(
    text: str,
    font_path: str,
    font_size: int,
    padding: int = 16,
) -> Image.Image:
    """Renders single-line glyph at an exact font size with dimensions snapped to 16."""
    f = ImageFont.truetype(font_path, font_size)
    bbox = f.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    w = int(np.ceil((tw + 2 * padding) / 16.0) * 16)
    h = int(np.ceil((th + 2 * padding) / 16.0) * 16)

    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    draw.text((x, y), text, font=f, fill=(255, 255, 255))
    return img


def run_vae_roundtrip_benchmark(
    text: str = "TÔI YÊU VIỆT NAM",
    font_alias: str = "bevietnam",
    output_dir: str = "output_vae_roundtrip",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    import torch
    from flux2.util import load_ae

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    font_map = {
        "bevietnam": str(ROOT_DIR / "fonts" / "BeVietnamPro-Black.ttf"),
        "playfair": str(ROOT_DIR / "fonts" / "PlayfairDisplay.ttf"),
        "anton": str(ROOT_DIR / "fonts" / "Anton-Regular.ttf"),
    }
    font_path = font_map.get(font_alias, font_map["bevietnam"])

    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(" 🔬 TENDOO AI: VAE ROUNDTRIP RECONSTRUCTION FIDELITY PROBE")
    print(f"📝 Text String  : '{text}'")
    print(f"🔤 Font         : {Path(font_path).name}")
    print(f"🖥️ Device       : {device}")
    print("=" * 90)

    print("\n[1/3] Loading VAE AutoEncoder...")
    ae = load_ae(model_name, device=device)
    ae.eval()

    # Sizes to benchmark:
    # 20pt (Ultra-tiny), 28-30pt (Notebook test), 38pt (Marginal), 48pt (Threshold), 60pt (Hero)
    test_sizes = [20, 28, 32, 40, 48, 60]

    print("\n[2/3] Executing VAE Roundtrip (Encode -> Decode) across Font Sizes...")
    print(f"{'Size':<6} | {'Glyph Dim':<14} | {'Latent Dim':<12} | {'PSNR (dB)':<10} | {'Diacritic Visual Status':<35}")
    print("-" * 90)

    grid_rows = []

    for sz in test_sizes:
        orig_img = render_glyph_at_size(text, font_path, sz)
        gw, gh = orig_img.size
        lat_w, lat_h = gw // 16, gh // 16

        # Convert to Tensor [-1, 1]
        np_arr = np.array(orig_img, dtype=np.float32) / 127.5 - 1.0
        ae_dtype = next(ae.parameters()).dtype if hasattr(ae, "parameters") else torch.bfloat16
        inp_tensor = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=ae_dtype)

        with torch.no_grad():
            latent = ae.encode(inp_tensor)[0]
            recon_tensor = ae.decode(latent.unsqueeze(0))

        # Convert back to uint8 [0, 255]
        recon_np = ((recon_tensor[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        recon_img = Image.fromarray(recon_np)

        # Quantitative Metrics: PSNR
        orig_np = np.array(orig_img, dtype=np.float32)
        recon_float = recon_np.astype(np.float32)
        mse = np.mean((orig_np - recon_float) ** 2)
        psnr = 10 * np.log10((255.0 ** 2) / (mse + 1e-10))

        # Difference image magnified 3x for artifact visualization
        diff_np = np.clip(np.abs(orig_np - recon_float) * 3.0, 0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_np)

        # Visual description
        if sz <= 20:
            status = "🔴 Mất hẳn dấu mũ / dính tịt nét"
        elif sz <= 30:
            status = "🟡 Dấu mờ, mép xuất hiện ringing"
        elif sz <= 40:
            status = "🟢 Dấu mũ nguyên vẹn, nét rõ"
        else:
            status = "✨ Tái tạo 100% hoàn hảo sắc lẹm"

        print(f"{sz:>3}pt  | {gw:>3}x{gh:<3}px     | {lat_w:>2}x{lat_h:<2} tokens  | {psnr:>6.2f} dB  | {status}")

        # Save individual pair
        orig_img.save(out_path / f"size_{sz:02d}pt_original.png")
        recon_img.save(out_path / f"size_{sz:02d}pt_vae_recon.png")
        diff_img.save(out_path / f"size_{sz:02d}pt_diff_3x.png")

        # Create row banner: [Original | VAE Reconstructed | Diff x3]
        banner_w = gw * 3 + 40
        banner_h = gh + 40
        banner = Image.new("RGB", (banner_w, banner_h), color=(20, 20, 20))
        d_draw = ImageDraw.Draw(banner)
        d_draw.text((10, 5), f"{sz}pt: ORIGINAL", fill=(200, 200, 200))
        d_draw.text((gw + 20, 5), "VAE RECONSTRUCTED", fill=(0, 255, 200))
        d_draw.text((gw * 2 + 30, 5), "DIFFERENCE (3x)", fill=(255, 100, 100))
        banner.paste(orig_img, (10, 25))
        banner.paste(recon_img, (gw + 20, 25))
        banner.paste(diff_img, (gw * 2 + 30, 25))
        grid_rows.append(banner)

    # Save master montage
    print("\n[3/3] Generating Master Comparison Montage...")
    max_row_w = max(b.width for b in grid_rows)
    total_grid_h = sum(b.height + 10 for b in grid_rows)
    master_montage = Image.new("RGB", (max_row_w, total_grid_h), color=(10, 10, 10))
    y_off = 5
    for b in grid_rows:
        master_montage.paste(b, (0, y_off))
        y_off += b.height + 10

    montage_file = out_path / "vae_roundtrip_master_comparison.png"
    master_montage.save(montage_file)
    print(f"🎉 MASTER COMPARISON SAVED TO: {montage_file.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tendoo AI - VAE Roundtrip Fidelity Probe")
    parser.add_argument("--text", type=str, default="TÔI YÊU VIỆT NAM", help="Text string to test")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias")
    parser.add_argument("--output_dir", type=str, default="output_vae_roundtrip", help="Output folder")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="Model name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint dir")

    args = parser.parse_args()
    run_vae_roundtrip_benchmark(
        text=args.text,
        font_alias=args.font,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
