"""
================================================================================
TENDOO AI - OBJECT REFERENCE TEMPORAL DECAY BENCHMARK (NO-TEXT SUITE)
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Scientific Objective:
- Measure Attention Signal Retention and Object Identity Preservation for Natural
  Objects / Products (4096 tokens) across large time offsets:
    t in [10.0 (Anchor), 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
- Identify the exact phase transition and collapse point (decay horizon) for 4096-token
  natural images vs sparse glyph tokens (which decayed at t >= 50.0).
- Pure Object Reference mode: ZERO text tokens used.
- Models loaded ONCE in memory with fixed seed z_init for 100% fair 1:1 comparison.
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

# Built-in Font Registry for labeling UI
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


def encode_product_image_to_latent(
    ae: AutoEncoder,
    image_path: str,
    target_size: int = 1024,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, int, int]:
    """
    Loads and encodes the natural product reference image into VAE latent space.
    Returns:
    - p_latent: (1, 128, lat_h, lat_w)
    - lat_h, lat_w
    """
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
    """
    Builds 4D RoPE coordinate tokens (t, h, w, l) for a specific time offset t.
    """
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


def stitch_grid_comparison_panel(
    images: list[Image.Image],
    titles: list[str],
    ref_thumbnail: Image.Image,
    output_path: str,
    cols: int = 3,
    main_title: str = "KHẢO SÁT SUY HAO TÍN HIỆU ẢNH THAM CHIẾU VẬT THỂ THEO THỜI GIAN (OBJECT ROPE TIME DECAY)",
    sub_title: str = "Đánh giá khả năng bảo toàn đặc trưng vật thể (4096 tokens) khi t tăng từ 40.0 đến 90.0 trong 4D RoPE",
):
    """
    Stitches generated images into an executive grid panel with reference thumbnail and labels.
    """
    w, h = images[0].size
    header_h = 130
    card_header_h = 48
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
        font_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=17)
        font_card = ImageFont.truetype(resolve_font_path("bevietnam"), size=19)
        font_footer = ImageFont.truetype(resolve_font_path("bevietnam"), size=14)
    except Exception:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_card = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # 1. Main Header
    draw.rectangle([0, 0, total_w, header_h], fill=(22, 26, 36), outline=(40, 46, 62), width=2)
    bbox_m = font_main.getbbox(main_title)
    draw.text(((total_w - (bbox_m[2] - bbox_m[0])) // 2, 20), main_title, fill=(255, 215, 80), font=font_main)

    bbox_s = font_sub.getbbox(sub_title)
    draw.text(((total_w - (bbox_s[2] - bbox_s[0])) // 2, 60), sub_title, fill=(200, 210, 230), font=font_sub)

    # Paste small reference thumbnail in header corner
    thumb_sz = 90
    thumb = ref_thumbnail.resize((thumb_sz, thumb_sz), Image.Resampling.LANCZOS)
    canvas.paste(thumb, (25, 20))
    draw.rectangle([25, 20, 25 + thumb_sz, 20 + thumb_sz], outline=(255, 215, 80), width=2)
    draw.text((125, 30), "ẢNH GỐC THAM CHIẾU", fill=(255, 215, 80), font=font_footer)
    draw.text((125, 52), "(4096 tokens VAE)", fill=(170, 180, 200), font=font_footer)

    # 2. Grid Cards
    for idx, (img, title) in enumerate(zip(images, titles)):
        r = idx // cols
        c = idx % cols
        x_offset = c * cell_w
        y_offset = header_h + r * cell_h

        # Card Header
        is_anchor = "10.0" in title
        header_bg = (24, 48, 32) if is_anchor else (32, 36, 48)
        header_border = (45, 90, 60) if is_anchor else (55, 62, 80)
        title_color = (120, 255, 160) if is_anchor else (255, 225, 120)

        draw.rectangle([x_offset, y_offset, x_offset + cell_w, y_offset + card_header_h], fill=header_bg, outline=header_border, width=1)
        bbox_t = font_card.getbbox(title)
        tx = x_offset + (cell_w - (bbox_t[2] - bbox_t[0])) // 2
        ty = y_offset + (card_header_h - (bbox_t[3] - bbox_t[1])) // 2 - bbox_t[1]
        draw.text((tx, ty), title, fill=title_color, font=font_card)

        # Card Image
        canvas.paste(img, (x_offset, y_offset + card_header_h))

    # 3. Footer Bar
    footer_y = header_h + cell_h * rows
    draw.rectangle([0, footer_y, total_w, total_h], fill=(18, 20, 28), outline=(35, 40, 55), width=1)
    footer_text = (
        "Thực nghiệm: FLUX.2-klein-base-4B | Text Encoder: Qwen3-4B-FP8 | VAE: 128-ch | "
        "Steps: 50 | CFG: 4.5 | Seed: 42 | Pure In-Context Image Flow Matching (No Text Tokens)"
    )
    bbox_f = font_footer.getbbox(footer_text)
    draw.text(
        ((total_w - (bbox_f[2] - bbox_f[0])) // 2, footer_y + (footer_h - (bbox_f[3] - bbox_f[1])) // 2),
        footer_text,
        fill=(130, 140, 165),
        font=font_footer,
    )

    canvas.save(output_path, quality=95)
    print(f"\n📊 [Executive Grid Comparison Saved] -> {output_path} ({total_w}x{total_h})")


def run_object_decay_probe(
    image_ref: str,
    prompt: str,
    t_offsets: list[float],
    include_anchor_t10: bool = True,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    cols: int = 3,
    output_dir: str = "output_object_decay_probe",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(image_ref):
        raise FileNotFoundError(f"❌ Product reference image not found at: {image_ref}")

    # Build evaluation list of t offsets
    eval_t_list = []
    if include_anchor_t10 and 10.0 not in t_offsets:
        eval_t_list.append(10.0)
    for t_val in t_offsets:
        if t_val not in eval_t_list:
            eval_t_list.append(t_val)
    eval_t_list.sort()

    print("=" * 80)
    print("🔬 TENDOO AI - OBJECT REFERENCE TEMPORAL DECAY PROBE (NO-TEXT SUITE)")
    print(f"📸 Image Reference : {image_ref}")
    print(f"📝 Scene Prompt    : {prompt}")
    print(f"⏱️  Time Offsets   : {eval_t_list}")
    print(f"📐 Resolution      : {width}x{height}")
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
    print("\n[2/4] Encoding Text Scene Prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    # 4. Encode Reference Image Once into VAE Latent Space
    print("\n[3/4] Encoding Product Reference Image (1024x1024 -> 4096 latent tokens)...")
    p_latent, p_lat_h, p_lat_w = encode_product_image_to_latent(
        ae=ae,
        image_path=image_ref,
        target_size=1024,
        device=device_ae,
    )
    print(f"  -> Product Latent Shape: {p_latent.shape} ({p_lat_h}x{p_lat_w} = {p_lat_h * p_lat_w} tokens)")

    # Prepare Canvas Latent Initial Noise (Same for all t for fair 1:1 comparison)
    lat_h = height // 16
    lat_w = width // 16
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    # 5. Sequential Sweep Across All Time Offsets
    print("\n[4/4] Running Sequential ODE Denoise across Time Offsets...")
    generated_images = []
    card_titles = []

    for step_idx, t_val in enumerate(eval_t_list, 1):
        print("\n" + "-" * 70)
        is_anchor = (t_val == 10.0)
        label_prefix = "ANCHOR BASELINE" if is_anchor else f"PROBE PASS {step_idx}/{len(eval_t_list)}"
        print(f"⚡ [{label_prefix}] Testing Time Offset: t = {t_val:.1f} (Tokens: {p_lat_h * p_lat_w})...")
        print("-" * 70)

        start_t = time.time()

        # Build RoPE 4D tokens with target time coordinate
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
        out_file = out_path / f"prod_slot_t{int(t_val):02d}.png"
        img_result.save(out_file)
        print(f"  -> Finished in {elapsed:.2f}s | Saved: {out_file.name}")

        generated_images.append(img_result)
        if is_anchor:
            card_titles.append(f"Anchor Baseline: t = {t_val:.1f} (Gold Standard)")
        else:
            card_titles.append(f"Time Offset Slot: t = {t_val:.1f}")

    # 6. Build Stitched Executive Comparison Panels
    ref_orig_img = Image.open(image_ref).convert("RGB")
    grid_output_path = str(out_path / "OBJECT_TIME_DECAY_GRID_COMPARISON.png")

    stitch_grid_comparison_panel(
        images=generated_images,
        titles=card_titles,
        ref_thumbnail=ref_orig_img,
        output_path=grid_output_path,
        cols=cols,
        main_title="KHẢO SÁT SUY HAO TÍN HIỆU ẢNH THAM CHIẾU VẬT THỂ THEO THỜI GIAN",
        sub_title=f"Đánh giá khả năng bảo toàn vật thể (4096 tokens) qua các mốc t = {[int(x) for x in eval_t_list]} (Seed {seed})",
    )

    print("\n" + "=" * 80)
    print("🎉 ALL PROBE PASSES COMPLETED SUCCESSFULLY!")
    print(f"📊 Executive Grid Panel Saved: {grid_output_path}")
    print(f"📁 Check all individual passes in: {out_path.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tendoo AI - Object Reference Temporal Decay Benchmark (No-Text Suite: t=40..90)"
    )
    parser.add_argument(
        "--image_ref",
        type=str,
        default="images/reference_prod.png",
        help="Path to natural object / product reference image (default: images/reference_prod.png)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Sản phẩm được đặt trang trọng ở chính giữa trên bục đá cẩm thạch sang trọng, "
            "hậu cảnh ánh sáng studio nghệ thuật tương phản cao, phản chiếu bóng đổ chân thực, "
            "phong cách chụp ảnh quảng cáo thương mại cao cấp, chi tiết sắc nét"
        ),
        help="Scene text prompt describing studio lighting & environment (NO text instructions)",
    )
    parser.add_argument(
        "--t_offsets",
        type=float,
        nargs="+",
        default=[40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        help="List of time offsets to test (default: 40 50 60 70 80 90)",
    )
    parser.add_argument(
        "--no_anchor_t10",
        action="store_true",
        help="Exclude t=10.0 anchor baseline from the test suite",
    )
    parser.add_argument("--width", type=int, default=1024, help="Canvas width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height in pixels (default: 1024)")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 1:1 identical initial state")
    parser.add_argument("--cols", type=int, default=3, help="Grid columns for stitched panel (default: 3 or 4)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_object_decay_probe",
        help="Output directory for generated probe passes",
    )
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    # Fallback to alternative sample images if default does not exist
    resolved_img = args.image_ref
    if not os.path.exists(resolved_img):
        for candidate in ["images/hao_hao.jpg", "images/ref_prod_02.png", "images/shoes.jpeg"]:
            if os.path.exists(candidate):
                resolved_img = candidate
                break

    run_object_decay_probe(
        image_ref=resolved_img,
        prompt=args.prompt,
        t_offsets=args.t_offsets,
        include_anchor_t10=not args.no_anchor_t10,
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
