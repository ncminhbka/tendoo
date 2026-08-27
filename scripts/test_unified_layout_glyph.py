"""
Test Script: Unified Spatial Layout Glyph Conditioning (All-in-One Canvas at t=10.0)
Model: FLUX.2-klein-base-4B + Qwen3-4B-FP8 + AE

Concept:
    Instead of fragmenting text into separate time slots (t=10, 20, 30) with tight crops
    that lose spatial context, we render ALL texts onto a SINGLE reference canvas
    matching the target poster's aspect ratio and resolution.
    
    This unified reference is fed at canonical t=10.0:
    1. Zero slot conflict (no t=20 or t=30 Softmax cannibalization).
    2. Exact 1:1 spatial correspondence between reference (h, w) and canvas (h, w).
    3. Natural visual hierarchy (Header large, CTA small, left/right/center positions preserved).

Usage on Remote Server (2x NVIDIA A30 / JupyterLab):
    # Test 1: Classic Coffee Poster (3 Texts: Top, Mid, Bottom)
    python scripts/test_unified_layout_glyph.py --preset coffee_poster

    # Test 2: Tester Prompt 1 (Smart Watch Ad: Top-Left + Mid-Left, 4:5 AR)
    python scripts/test_unified_layout_glyph.py --preset smartwatch_ad
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

# Offline Mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from einops import rearrange

from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.text_encoder import Qwen3Embedder
from flux2.util import load_ae, load_flow_model, load_text_encoder


def resolve_font_path(font_alias: str | None) -> str:
    """Finds a valid Unicode TTF font from alias or fallback system fonts."""
    font_dir = ROOT_DIR / "fonts"
    alias_map = {
        "playfair": font_dir / "PlayfairDisplay.ttf",
        "bevietnam": font_dir / "BeVietnamPro-Black.ttf",
        "anton": font_dir / "Anton-Regular.ttf",
        "oswald": font_dir / "Oswald.ttf",
        "pacifico": font_dir / "Pacifico-Regular.ttf",
        "dancing": font_dir / "DancingScript.ttf",
    }

    if font_alias and font_alias.lower() in alias_map:
        target = alias_map[font_alias.lower()]
        if target.exists():
            return str(target)

    candidates = [
        font_dir / "BeVietnamPro-Black.ttf",
        font_dir / "PlayfairDisplay.ttf",
        font_dir / "Anton-Regular.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    return "arial.ttf"


def create_unified_layout_glyph(
    canvas_w: int,
    canvas_h: int,
    text_specs: list[dict],
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Renders all text elements onto a single full-canvas image matching the target dimensions.
    
    Each item in text_specs can specify:
        text: str
        font: str (alias or path)
        font_size: int (optional, if omitted auto-calculated from size_ratio)
        size_ratio: float (font size relative to canvas_h, e.g. 0.06 = 6% of canvas height)
        pos: "center", "top_center", "mid_center", "btm_center", "top_left", "mid_left", "btm_left", etc.
        y_ratio: float (0.0 to 1.0, vertical center of text)
        x_ratio: float (0.0 to 1.0, horizontal anchor of text)
        color: tuple (RGB)
    """
    canvas_w = (canvas_w // 16) * 16
    canvas_h = (canvas_h // 16) * 16

    img = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    for spec in text_specs:
        text = spec.get("text", "")
        if not text.strip():
            continue

        font_path = resolve_font_path(spec.get("font", "bevietnam"))
        size_ratio = spec.get("size_ratio", 0.05)
        font_size = spec.get("font_size", max(18, int(canvas_h * size_ratio)))
        col = spec.get("color", text_color)

        font = ImageFont.truetype(font_path, size=font_size)
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Determine coordinates based on pos or explicit ratios
        pos = spec.get("pos", "center")
        y_ratio = spec.get("y_ratio", None)
        x_ratio = spec.get("x_ratio", None)

        if x_ratio is not None:
            curr_x = int(canvas_w * x_ratio)
        elif "left" in pos:
            curr_x = int(canvas_w * 0.08)
        elif "right" in pos:
            curr_x = int(canvas_w * 0.92) - text_w
        else:  # center
            curr_x = (canvas_w - text_w) // 2

        if y_ratio is not None:
            curr_y = int(canvas_h * y_ratio) - text_h // 2
        elif "top" in pos:
            curr_y = int(canvas_h * 0.12)
        elif "mid" in pos:
            curr_y = int(canvas_h * 0.52)
        elif "btm" in pos:
            curr_y = int(canvas_h * 0.88)
        else:
            curr_y = (canvas_h - text_h) // 2

        draw.text((curr_x - bbox[0], curr_y - bbox[1]), text, font=font, fill=col)

    return img


def run_unified_layout_experiment(
    prompt: str,
    text_specs: list[dict],
    width: int = 576,
    height: int = 1024,
    steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    output_dir: str = "output_unified_layout_test",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    """Executes the Unified Spatial Layout Glyph Experiment at t=10.0."""
    start_time = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    width = (width // 16) * 16
    height = (height // 16) * 16
    lat_h = height // 16
    lat_w = width // 16
    num_canvas = lat_h * lat_w

    print("=" * 80)
    print(" 🚀 TENDOO AI: UNIFIED SPATIAL LAYOUT GLYPH (ALL TEXT IN ONE CANVAS AT t=10.0)")
    print("=" * 80)
    print(f"📐 Canvas Resolution  : {width}x{height} pixels (Latent: {lat_h}x{lat_w} = {num_canvas} tokens)")
    print(f"📝 Text Elements ({len(text_specs)} blocks):")
    for idx, spec in enumerate(text_specs, 1):
        print(f"   [{idx}] '{spec['text']}' -> Pos: {spec.get('pos', 'custom')}, SizeRatio: {spec.get('size_ratio', 0.05):.3f}")
    print(f"🎨 Visual Prompt      : {prompt}")
    print(f"⚙️ Sampling Parameters: Steps={steps}, Guidance={guidance}, Seed={seed}")
    print("=" * 80)

    # 1. Device Allocation
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print("🚀 Dual GPU Mode: DiT on GPU 0 | VAE & Text Encoder on GPU 1")
    else:
        device_dit = device_ae = device_te = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Single Device Mode: {device_dit}")

    # 2. Render the Unified Layout Glyph Canvas
    print("\n[1/4] Generating Unified Spatial Layout Glyph Canvas...")
    unified_glyph = create_unified_layout_glyph(
        canvas_w=width,
        canvas_h=height,
        text_specs=text_specs,
        bg_color=(0, 0, 0),
        text_color=(255, 255, 255),
    )
    glyph_preview = out_path / "unified_glyph_preview.png"
    unified_glyph.save(glyph_preview)
    print(f"  -> Saved layout glyph preview: {glyph_preview.name} ({width}x{height})")

    # 3. Load Models
    print("\n[2/4] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_text_encoder(model_name, device=device_te)

    # Encode Visual Prompt
    print("  -> Encoding Prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    if num_gpus < 2:
        del text_encoder
        torch.cuda.empty_cache()

    # 4. Encode Unified Glyph via VAE to 4D RoPE at t=10.0
    print("\n[3/4] Encoding Unified Glyph into 4D RoPE Reference Tokens (t=10.0)...")
    np_arr = np.array(unified_glyph, dtype=np.float32) / 127.5 - 1.0
    glyph_dtype = next(ae.parameters()).dtype if hasattr(ae, "parameters") else torch.bfloat16
    glyph_tensor = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=device_ae, dtype=glyph_dtype)

    with torch.no_grad():
        glyph_latent = ae.encode(glyph_tensor)[0]

    _, g_lat_h, g_lat_w = glyph_latent.shape
    ref_tokens = rearrange(glyph_latent, "c h w -> (h w) c").unsqueeze(0).to(device_dit, dtype=torch.bfloat16)

    # Clean float32 RoPE coordinates for reference at t=10.0 (prevents cartesian_prod meshgrid dtype error)
    t_c = torch.tensor([10.0], dtype=torch.float32, device=device_dit)
    h_c = torch.arange(g_lat_h, dtype=torch.float32, device=device_dit)
    w_c = torch.arange(g_lat_w, dtype=torch.float32, device=device_dit)
    l_c = torch.arange(1, dtype=torch.float32, device=device_dit)
    ref_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0).to(device_dit)

    num_ref = ref_tokens.shape[1]

    print(f"  -> Reference Sequence: {num_ref} tokens (Latent: {lat_h}x{lat_w} at t=10.0)")
    print(f"  -> Canvas Sequence   : {num_canvas} tokens (Latent: {lat_h}x{lat_w})")
    print(f"  -> Total Combined    : {txt.shape[1] + num_canvas + num_ref} tokens")

    # Initial Noise for Canvas
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    timesteps = get_schedule(num_steps=steps, image_seq_len=num_canvas)

    # 5. Denoise with Native BFL denoise_cfg
    print(f"\n[4/4] Denoising with Native Full Bidirectional Joint Attention ({steps} steps, CFG={guidance})...")
    t_denoise_start = time.time()

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
        print("  -> Decoding output latent via VAE...")
        out_pixels = ae.decode(out_latent.to(device_ae))
        out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_img = Image.fromarray(out_pixels)

    denoise_time = time.time() - t_denoise_start
    total_time = time.time() - start_time

    out_file = out_path / "result_unified_layout.png"
    result_img.save(out_file)

    print("\n" + "=" * 80)
    print("🎉 UNIFIED LAYOUT EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"⏱️ Denoise Time: {denoise_time:.2f}s | Total Time: {total_time:.2f}s")
    print(f"📁 Layout Glyph Preview : {glyph_preview.resolve()}")
    print(f"🖼️ Generated Image      : {out_file.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tendoo AI - Unified Spatial Layout Glyph Test")
    parser.add_argument(
        "--preset",
        type=str,
        default="coffee_poster",
        choices=["coffee_poster", "smartwatch_ad", "custom"],
        help="Predefined test case",
    )
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="output_unified_layout", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    if args.preset == "coffee_poster":
        # Classic 3-Slot Coffee Poster (All in 1 Canvas at t=10.0)
        prompt = (
            "Poster quảng cáo quầy bar cà phê gỗ mộc mạc cổ điển, ánh sáng ven ấm áp tương phản cao, "
            "dòng chữ tiêu đề 3D mạ vàng đồng cổ sắc nét nổi bật ở phía trên, dòng chữ phụ giới thiệu phong vị tinh tế khắc trên gờ gỗ ở giữa quầy bar, "
            "và dòng chữ thông báo ưu đãi chữ trắng sắc nét trên bảng đế ở chân quầy bar, "
            "bố cục sạch sẽ gọn gàng, không có chữ ký, không có watermark, không có chữ trang trí thừa, độ sâu trường ảnh điện ảnh"
        )
        text_specs = [
            {"text": "CÀ PHÊ SỮA ĐÁ", "font": "playfair", "pos": "top_center", "size_ratio": 0.075, "y_ratio": 0.16},
            {"text": "ĐẬM ĐÀ HƯƠNG VỊ VIỆT", "font": "bevietnam", "pos": "mid_center", "size_ratio": 0.045, "y_ratio": 0.58},
            {"text": "MUA 1 TẶNG 1", "font": "bevietnam", "pos": "btm_center", "size_ratio": 0.055, "y_ratio": 0.90},
        ]
        w, h = 576, 1024

    elif args.preset == "smartwatch_ad":
        # Tester Prompt 1: Smartwatch on rustic wooden coffee table with Top-Left & Mid-Left texts (AR 4:5)
        prompt = (
            "Một chiếc đồng hồ thông minh hiện đại cao cấp với dây đeo kim loại màu bạc bóng bẩy, "
            "đặt trên chiếc bàn cà phê bằng gỗ mộc mạc cạnh một tách cà phê latte art và cặp kính râm thời trang. "
            "Ánh nắng ban mai nhẹ nhàng chiếu qua cửa sổ, bầu không khí ấm áp, chụp bằng ống kính 35mm, chân thực, độ chi tiết cao, "
            "bố cục tinh tế với khoảng không gian thanh lịch bên trái dành cho dòng chữ màu trắng sắc nét"
        )
        text_specs = [
            {"text": "THỜI GIAN LÀ CỦA BẠN", "font": "bevietnam", "pos": "top_left", "size_ratio": 0.038, "x_ratio": 0.08, "y_ratio": 0.12},
            {"text": "NÂNG TẦM PHONG CÁCH ĐỜI SỐNG", "font": "bevietnam", "pos": "mid_left", "size_ratio": 0.052, "x_ratio": 0.08, "y_ratio": 0.42},
        ]
        w, h = 816, 1024  # 4:5 Aspect Ratio snapped to 16

    else:
        # Default Custom template
        prompt = "Poster nghệ thuật hiện đại sang trọng, ánh sáng studio tương phản cao"
        text_specs = [
            {"text": "TIÊU ĐỀ CHÍNH", "font": "playfair", "pos": "top_center", "size_ratio": 0.08, "y_ratio": 0.20},
            {"text": "THÔNG ĐIỆP PHỤ", "font": "bevietnam", "pos": "btm_center", "size_ratio": 0.05, "y_ratio": 0.85},
        ]
        w, h = 576, 1024

    run_unified_layout_experiment(
        prompt=prompt,
        text_specs=text_specs,
        width=w,
        height=h,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
