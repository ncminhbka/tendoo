"""
================================================================================
TENDOO AI - ROPE 4D PHASE ALIASING & HARMONIC RESONANCE BENCHMARK
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Scientific Objective:
- Prove the "RoPE Fourier Phase Aliasing Hypothesis":
  At t ~= k * 2pi (e.g. t=43.98, 50.27, 56.55, 62.83), the fastest frequency
  omega_0 = 1.0 completes full revolutions, resulting in near-zero relative phase
  difference with Canvas (t=0), causing Attention Phase Collision / Aliasing.
- Conversely, at t ~= (k + 0.5) * 2pi (e.g. t=47.12, 53.41, 59.69 ~= 60.0),
  the phase is near 180 degrees (maximum distinctiveness against Canvas).
- Compare Alias Points vs Anti-Alias Points vs Canonical Integer Multiples of 10.
- Pure Object Reference Mode: 4096 tokens (No Text Tokens).
================================================================================
"""

import argparse
import math
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

    raise RuntimeError("❌ No valid font found for rendering comparison labels!")


def calculate_rope_phase(t_val: float) -> tuple[float, float, str]:
    """
    Calculates the relative phase angle and cosine similarity for omega_0 = 1.0
    relative to Canvas (t = 0.0).
    """
    phase_rad = t_val % (2 * math.pi)
    phase_deg = math.degrees(phase_rad)
    cos_sim = math.cos(phase_rad)

    # Distance to 0 degrees or 360 degrees
    dist_to_zero = min(phase_deg, 360.0 - phase_deg)

    if dist_to_zero <= 25.0:
        tag = "🔴 ALIAS PEAK (Nguy cơ Chập Pha)"
    elif 155.0 <= phase_deg <= 205.0:
        tag = "🟢 ANTI-ALIAS (Ngược Pha Tối Ưu)"
    else:
        tag = "🟡 TRANSITION (Pha Trung Gian)"

    return phase_deg, cos_sim, tag


def encode_product_image_to_latent(
    ae: AutoEncoder,
    image_path: str,
    target_size: int = 1024,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, int, int]:
    """Encodes natural product reference image into VAE latent space."""
    prod_img = Image.open(image_path).convert("RGB")
    prod_img = prod_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    p_arr = np.array(prod_img).astype(np.float32) / 127.5 - 1.0
    p_tensor = torch.from_numpy(p_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        p_latent = ae.encode(p_tensor)

    lat_h, lat_w = p_latent.shape[2], p_latent.shape[3]
    return p_latent, lat_h, lat_w


def build_incontext_tokens_for_t(
    p_latent: torch.Tensor,
    t_offset: float,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds 4D RoPE coordinate tokens (t, h, w, l) for time offset t."""
    ref_tokens, _ = prc_img(p_latent[0])
    ref_tokens = ref_tokens.unsqueeze(0).to(device)

    p_h, p_w = p_latent.shape[2], p_latent.shape[3]
    t_coords = torch.full((p_h, p_w), fill_value=float(t_offset), dtype=torch.float32, device=device)
    h_coords = torch.arange(p_h, dtype=torch.float32, device=device).unsqueeze(1).expand(p_h, p_w)
    w_coords = torch.arange(p_w, dtype=torch.float32, device=device).unsqueeze(0).expand(p_h, p_w)
    l_coords = torch.zeros((p_h, p_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0).to(device)

    return ref_tokens, ref_ids


def stitch_phase_grid_panel(
    images: list[Image.Image],
    t_values: list[float],
    ref_thumbnail: Image.Image,
    output_path: str,
    cols: int = 4,
):
    """Creates a high-density executive panel with detailed mathematical phase annotations."""
    w, h = images[0].size
    header_h = 135
    card_header_h = 62
    footer_h = 50

    rows = (len(images) + cols - 1) // cols
    cell_w = w
    cell_h = h + card_header_h

    total_w = cell_w * cols
    total_h = header_h + cell_h * rows + footer_h

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 17, 23))
    draw = ImageDraw.Draw(canvas)

    try:
        font_main = ImageFont.truetype(resolve_font_path("bevietnam"), size=26)
        font_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=16)
        font_card_title = ImageFont.truetype(resolve_font_path("bevietnam"), size=18)
        font_card_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=13)
        font_footer = ImageFont.truetype(resolve_font_path("bevietnam"), size=14)
    except Exception:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_card_title = ImageFont.load_default()
        font_card_sub = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # 1. Main Header
    draw.rectangle([0, 0, total_w, header_h], fill=(22, 26, 36), outline=(40, 46, 62), width=2)
    main_title = "KIỂM CHỨNG TOÁN HỌC: HIỆN TƯỢNG TRÙNG PHA ROPE (PHASE ALIASING) TẠI CÁC MỐC THỜI GIAN"
    bbox_m = font_main.getbbox(main_title)
    draw.text(((total_w - (bbox_m[2] - bbox_m[0])) // 2, 20), main_title, fill=(255, 215, 80), font=font_main)

    sub_title = (
        "So sánh trực tiếp giữa các Điểm Trùng Pha (Alias t ~= k*2π: Phase ~0°) "
        "vs Điểm Ngược Pha (Anti-Alias t ~= (k+0.5)*2π: Phase ~180°)"
    )
    bbox_s = font_sub.getbbox(sub_title)
    draw.text(((total_w - (bbox_s[2] - bbox_s[0])) // 2, 60), sub_title, fill=(200, 210, 230), font=font_sub)

    # Paste Thumbnail
    thumb_sz = 90
    thumb = ref_thumbnail.resize((thumb_sz, thumb_sz), Image.Resampling.LANCZOS)
    canvas.paste(thumb, (25, 22))
    draw.rectangle([25, 22, 25 + thumb_sz, 22 + thumb_sz], outline=(255, 215, 80), width=2)
    draw.text((125, 35), "ẢNH GỐC THAM CHIẾU", fill=(255, 215, 80), font=font_footer)
    draw.text((125, 58), "(4096 tokens VAE)", fill=(170, 180, 200), font=font_footer)

    # 2. Grid Cards
    for idx, (img, t_val) in enumerate(zip(images, t_values)):
        r = idx // cols
        c = idx % cols
        x_offset = c * cell_w
        y_offset = header_h + r * cell_h

        phase_deg, cos_sim, tag = calculate_rope_phase(t_val)

        if "ALIAS PEAK" in tag:
            header_bg = (48, 20, 20)
            header_border = (90, 35, 35)
            title_color = (255, 110, 110)
        elif "ANTI-ALIAS" in tag:
            header_bg = (18, 44, 26)
            header_border = (35, 85, 45)
            title_color = (110, 255, 150)
        else:
            header_bg = (32, 36, 48)
            header_border = (55, 62, 80)
            title_color = (255, 225, 120)

        # Draw card header
        draw.rectangle([x_offset, y_offset, x_offset + cell_w, y_offset + card_header_h], fill=header_bg, outline=header_border, width=1)

        card_title = f"t = {t_val:.1f}  |  {tag}"
        bbox_ct = font_card_title.getbbox(card_title)
        draw.text(
            (x_offset + (cell_w - (bbox_ct[2] - bbox_ct[0])) // 2, y_offset + 8),
            card_title,
            fill=title_color,
            font=font_card_title,
        )

        card_sub = f"Pha RoPE (ω=1.0): {phase_deg:.1f}°  |  Cosine Sim với Canvas (t=0): {cos_sim:+.3f}"
        bbox_cs = font_card_sub.getbbox(card_sub)
        draw.text(
            (x_offset + (cell_w - (bbox_cs[2] - bbox_cs[0])) // 2, y_offset + 34),
            card_sub,
            fill=(200, 210, 230),
            font=font_card_sub,
        )

        # Paste image
        canvas.paste(img, (x_offset, y_offset + card_header_h))

    # 3. Footer Bar
    footer_y = header_h + cell_h * rows
    draw.rectangle([0, footer_y, total_w, total_h], fill=(18, 20, 28), outline=(35, 40, 55), width=1)
    footer_text = (
        "Thực nghiệm Đối chứng: FLUX.2-klein-base-4B | Text Encoder: Qwen3-4B-FP8 | VAE: 128-ch | "
        "Steps: 50 | CFG: 4.5 | Seed: 42 | Pure In-Context Image Flow Matching"
    )
    bbox_f = font_footer.getbbox(footer_text)
    draw.text(
        ((total_w - (bbox_f[2] - bbox_f[0])) // 2, footer_y + (footer_h - (bbox_f[3] - bbox_f[1])) // 2),
        footer_text,
        fill=(130, 140, 165),
        font=font_footer,
    )

    canvas.save(output_path, quality=95)
    print(f"\n📊 [Phase Aliasing Executive Grid Saved] -> {output_path} ({total_w}x{total_h})")


def run_phase_aliasing_probe(
    image_ref: str,
    prompt: str,
    suite_mode: str = "comprehensive",
    custom_t_list: list[float] | None = None,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    cols: int = 4,
    output_dir: str = "output_phase_aliasing_probe",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(image_ref):
        raise FileNotFoundError(f"❌ Image reference not found at: {image_ref}")

    # Determine Target T Values
    if custom_t_list is not None and len(custom_t_list) > 0:
        eval_t_list = sorted(list(set(custom_t_list)))
    elif suite_mode == "alias_vs_antialias":
        # Targeted Harmonic Resonance Pairs:
        # [10 (Anchor), 44.0 (Alias k=7), 47.1 (Anti-Alias k=7.5), 50.0 (Alias k=8), 53.4 (Anti-Alias k=8.5), 56.5 (Alias k=9), 60.0 (Anti-Alias k=9.5), 63.0 (Alias k=10)]
        eval_t_list = [10.0, 44.0, 47.1, 50.0, 53.4, 56.5, 60.0, 63.0]
    elif suite_mode == "canonical_multiples":
        # Pure 10x Multiples
        eval_t_list = [10.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    else:  # comprehensive
        # Full Spectrum: Anchor + 40 + [44 vs 47.1] + [50 vs 53.4] + [56.5 vs 60] + 70 + 80
        eval_t_list = [10.0, 40.0, 44.0, 47.1, 50.0, 53.4, 56.5, 60.0, 70.0, 80.0]

    print("=" * 80)
    print("🔬 TENDOO AI - ROPE 4D PHASE ALIASING & HARMONIC RESONANCE BENCHMARK")
    print(f"📸 Image Reference : {image_ref}")
    print(f"📝 Scene Prompt    : {prompt}")
    print(f"⏱️  Evaluation Mode : {suite_mode} -> {eval_t_list}")
    print(f"📐 Canvas Size     : {width}x{height}")
    print(f"⚙️  Steps / CFG     : {num_steps} steps | CFG {guidance} | Seed: {seed}")
    print(f"📁 Output Directory: {out_path.resolve()}")
    print("=" * 80)

    # 1. Setup Hardware Allocation
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print("🚀 Multi-GPU Mode: DiT on GPU 0 (cuda:0), VAE & Qwen3 on GPU 1 (cuda:1)")
    elif num_gpus == 1:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:0")
        device_te = torch.device("cuda:0")
        print("🚀 Single-GPU Mode: All components on cuda:0")
    else:
        raise RuntimeError("❌ CUDA GPU is required to run this benchmark!")

    # 2. Load Models Once
    print("\n[1/4] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    # 3. Encode Scene Prompt Once
    print("\n[2/4] Encoding Scene Text Prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    # 4. Encode Reference Image Once into VAE Latent Space
    print("\n[3/4] Encoding Reference Image (1024x1024 -> 4096 latent tokens)...")
    p_latent, p_lat_h, p_lat_w = encode_product_image_to_latent(
        ae=ae,
        image_path=image_ref,
        target_size=1024,
        device=device_ae,
    )
    print(f"  -> Reference Latent: {p_latent.shape} ({p_lat_h * p_lat_w} tokens)")

    # Prepare Shared Initial Noise Latent
    lat_h = height // 16
    lat_w = width // 16
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    # 5. Sequential Sweep Across All Evaluated Time Offsets
    print("\n[4/4] Running Sequential ODE Denoise across Harmonic Phase Points...")
    generated_images = []

    for step_idx, t_val in enumerate(eval_t_list, 1):
        phase_deg, cos_sim, tag = calculate_rope_phase(t_val)
        print("\n" + "-" * 70)
        print(f"⚡ [PASS {step_idx}/{len(eval_t_list)}] Testing t = {t_val:.1f}")
        print(f"   -> Phase: {phase_deg:.1f}° | Cosine: {cos_sim:+.3f} | {tag}")
        print("-" * 70)

        start_t = time.time()

        ref_tokens, ref_ids = build_incontext_tokens_for_t(
            p_latent=p_latent,
            t_offset=t_val,
            device=device_dit,
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
            out_pixels = ae.decode(out_latent.to(device_ae))
            out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            img_result = Image.fromarray(out_pixels)

        elapsed = time.time() - start_t
        t_tag_str = f"t{t_val:.1f}".replace(".", "_")
        out_file = out_path / f"probe_{t_tag_str}.png"
        img_result.save(out_file)
        print(f"  -> Finished in {elapsed:.2f}s | Saved: {out_file.name}")

        generated_images.append(img_result)

    # 6. Build Stitched Executive Comparison Panel
    ref_orig_img = Image.open(image_ref).convert("RGB")
    grid_output_path = str(out_path / "ROPE_PHASE_ALIASING_GRID_COMPARISON.png")

    stitch_phase_grid_panel(
        images=generated_images,
        t_values=eval_t_list,
        ref_thumbnail=ref_orig_img,
        output_path=grid_output_path,
        cols=cols,
    )

    print("\n" + "=" * 80)
    print("🎉 HARMONIC ALIASING BENCHMARK COMPLETED!")
    print(f"📊 Stitched Panel Saved: {grid_output_path}")
    print(f"📁 Individual Passes   : {out_path.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tendoo AI - RoPE 4D Fourier Phase Aliasing Benchmark"
    )
    parser.add_argument(
        "--image_ref",
        type=str,
        default="images/reference_prod.png",
        help="Path to natural object / product reference image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Sản phẩm được đặt trang trọng ở chính giữa trên bục đá cẩm thạch sang trọng, "
            "hậu cảnh ánh sáng studio nghệ thuật tương phản cao, phản chiếu bóng đổ chân thực, "
            "phong cách chụp ảnh quảng cáo thương mại cao cấp, chi tiết sắc nét"
        ),
        help="Scene text prompt describing studio environment (NO text instructions)",
    )
    parser.add_argument(
        "--suite_mode",
        type=str,
        choices=["alias_vs_antialias", "canonical_multiples", "comprehensive"],
        default="alias_vs_antialias",
        help=(
            "Suite mode: "
            "'alias_vs_antialias' (t=10, 44, 47.1, 50, 53.4, 56.5, 60, 63) | "
            "'canonical_multiples' (t=10, 40, 50, 60, 70, 80, 90) | "
            "'comprehensive' (all key points)"
        ),
    )
    parser.add_argument(
        "--custom_t_list",
        type=float,
        nargs="+",
        default=None,
        help="Optional custom list of float t values (e.g. --custom_t_list 10 44 47.1 50 60)",
    )
    parser.add_argument("--width", type=int, default=1024, help="Canvas width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height in pixels (default: 1024)")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 1:1 identical initial state")
    parser.add_argument("--cols", type=int, default=4, help="Grid columns for stitched panel (default: 4)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_phase_aliasing_probe",
        help="Output directory for generated probe passes",
    )
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    resolved_img = args.image_ref
    if not os.path.exists(resolved_img):
        for candidate in ["images/hao_hao.jpg", "images/ref_prod_02.png", "images/shoes.jpeg"]:
            if os.path.exists(candidate):
                resolved_img = candidate
                break

    run_phase_aliasing_probe(
        image_ref=resolved_img,
        prompt=args.prompt,
        suite_mode=args.suite_mode,
        custom_t_list=args.custom_t_list,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        cols=args.cols,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
