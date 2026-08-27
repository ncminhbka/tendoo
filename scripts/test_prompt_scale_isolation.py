"""
================================================================================
TENDOO AI - PROMPT SCALE ISOLATION SCIENTIFIC BENCHMARK
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Scientific Objective:
Rigorously test whether the Text Prompt CAN control the physical rendering scale
of text in the generated image, when holding the Glyph Bitmap 100% IDENTICAL.

Setup (Strict Variable Isolation):
  - Fixed Glyph: "CÀ PHÊ SỮA ĐÁ" (Font: BeVietnamPro-Black, 448x160 px, pristine diacritics)
  - Fixed Random Seed: 42 (Identical initial noise latent z_init)
  - Fixed Canvas: 9:16 (576x1024) | Steps: 50 | CFG Guidance: 4.5 | t = 10.0

Prompt Variations (Scale & Object Carrier Modulation):
  1. Scale 1 (GIANT HEADLINE): Chữ khổng lồ choán ngợp toàn bộ poster.
  2. Scale 2 (MEDIUM SIGNBOARD): Chữ kích thước vừa vặn trên tấm biển hiệu gỗ trước quán.
  3. Scale 3 (SMALL CUP LABEL): Chữ nhỏ thanh mảnh in trên thân tách cà phê sứ.
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
}


def resolve_font_path(font_name_or_path: str | None) -> str:
    if font_name_or_path:
        key = font_name_or_path.lower().strip()
        if key in FONT_REGISTRY and os.path.exists(FONT_REGISTRY[key]):
            return FONT_REGISTRY[key]
        if os.path.exists(font_name_or_path):
            return font_name_or_path

    for p in [
        FONT_REGISTRY["bevietnam"],
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]:
        if os.path.exists(p):
            return p

    raise RuntimeError("❌ No valid Vietnamese Unicode font found!")


def create_glyph_image(
    text: str,
    target_width: int = 448,
    target_height: int = 160,
    font_path: str | None = None,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    padding_ratio: float = 0.08,
) -> Image.Image:
    """Renders clean, bold Vietnamese glyph bitmap with generous diacritic spacing."""
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16

    font_path = resolve_font_path(font_path)

    pad_w = int(target_width * padding_ratio)
    pad_h = int(target_height * padding_ratio)
    max_w = target_width - 2 * pad_w
    max_h = target_height - 2 * pad_h

    lines = [l.strip() for l in text.replace("\\n", "\n").split("\n") if l.strip()]

    low, high = 16, 200
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

        spacing = int(mid_size * 0.25) * (len(lines) - 1)
        total_h += spacing

        if max_line_w <= max_w and total_h <= max_h:
            opt_font = test_font
            opt_size = mid_size
            low = mid_size + 1
        else:
            high = mid_size - 1

    if opt_font is None:
        opt_font = ImageFont.truetype(font_path, size=28)
        opt_size = 28

    line_heights = []
    line_widths = []
    for line in lines:
        bbox = opt_font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(opt_size * 0.25)
    total_block_h = sum(line_heights) + line_spacing * (len(lines) - 1)

    img = Image.new("RGB", (target_width, target_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    curr_y = (target_height - total_block_h) // 2
    for i, line in enumerate(lines):
        lw = line_widths[i]
        curr_x = (target_width - lw) // 2
        bbox = opt_font.getbbox(line)
        draw.text((curr_x - bbox[0], curr_y - bbox[1]), line, fill=text_color, font=opt_font)
        curr_y += line_heights[i] + line_spacing

    return img


def encode_glyph_to_incontext_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes a tight-crop glyph image to 4D RoPE coordinate tokens."""
    ae_device = next(ae.parameters()).device if hasattr(ae, "parameters") else device

    g_arr = np.array(glyph_img).astype(np.float32) / 127.5 - 1.0
    g_tensor = torch.from_numpy(g_arr).permute(2, 0, 1).unsqueeze(0).to(device=ae_device, dtype=torch.bfloat16)

    with torch.no_grad():
        g_latent = ae.encode(g_tensor)

    ref_tokens, _ = prc_img(g_latent[0])
    ref_tokens = ref_tokens.unsqueeze(0).to(device)

    g_h, g_w = g_latent.shape[2], g_latent.shape[3]
    t_coords = torch.full((g_h, g_w), fill_value=float(t_offset), dtype=torch.float32, device=device)
    h_coords = torch.arange(g_h, dtype=torch.float32, device=device).unsqueeze(1).expand(g_h, g_w)
    w_coords = torch.arange(g_w, dtype=torch.float32, device=device).unsqueeze(0).expand(g_h, g_w)
    l_coords = torch.zeros((g_h, g_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0).to(device)

    return ref_tokens, ref_ids


# 3 Prompt Variations testing scale control across identical Glyph tokens
SCALE_TEST_CASES = [
    {
        "id": "scale_giant_headline",
        "title": "CASE 1: CHỮ KHỔNG LỒ (GIANT HEADLINE)",
        "subtitle": "Text is the Monumental Hero Subject",
        "prompt": (
            "Poster quảng cáo cà phê nghệ thuật hiện đại, dòng chữ tiêu đề 3D khổng lồ choán ngợp ở trung tâm poster "
            "mạ vàng ánh kim phát sáng rực rỡ sắc nét tinh xảo (giant monumental 3D golden headline typography filling the poster), "
            "hậu cảnh không gian quán cafe tối mờ phía sau, ánh sáng studio tương phản cao, đổ bóng chân thực, chi tiết sắc nét 8k"
        ),
        "tag_color": (255, 215, 80),
    },
    {
        "id": "scale_medium_signboard",
        "title": "CASE 2: CHỮ VỪA VẶN (STORE SIGNBOARD)",
        "subtitle": "Text is on a Physical Wooden Storefront Sign",
        "prompt": (
            "Góc phố cổ Hà Nội yên bình buổi sớm mai với ánh nắng le lói, trước hiên quán cafe có treo một tấm biển hiệu gỗ sồi hình chữ nhật phẳng cổ kính, "
            "trên mặt biển gỗ là dòng chữ 3D mạ đồng nổi bật sắc nét tinh xảo (medium-sized 3D bronze typography on rectangular wooden signboard), "
            "ánh sáng sớm chiếu rọi chân thực, phong cách điện ảnh ấm cúng"
        ),
        "tag_color": (255, 160, 80),
    },
    {
        "id": "scale_small_cup_label",
        "title": "CASE 3: CHỮ NHỎ (TINY CUP LABEL)",
        "subtitle": "Text is a Subtle Delicate Brand Label on a Cup",
        "prompt": (
            "Góc chụp cận cảnh nghệ thuật một tách cà phê gốm sứ trắng cao cấp bốc khói nghi ngút trên mặt bàn gỗ mộc, "
            "trên bề mặt thân tách cà phê có in dòng chữ thương hiệu kích thước nhỏ thanh mảnh tinh tế mạ vàng ánh kim "
            "(small delicate minimalist brand typography printed on the side of the ceramic coffee cup), "
            "ánh sáng studio tương phản cao, đổ bóng chân thực, độ sâu trường ảnh nông bokeh, chi tiết sắc nét"
        ),
        "tag_color": (100, 220, 255),
    },
]


def stitch_3_panel_comparison(
    results: list[tuple[dict, Image.Image]],
    glyph_img: Image.Image,
    output_path: str,
):
    """Stitches the 3 generated scale images + glyph preview into an executive panel."""
    sample_img = results[0][1]
    img_w, img_h = sample_img.size

    cols = 3
    margin = 24
    gap_x = 20

    card_header_h = 75
    header_bar_h = 170
    footer_bar_h = 50

    total_w = margin * 2 + cols * img_w + (cols - 1) * gap_x
    total_h = header_bar_h + margin + (img_h + card_header_h) + margin + footer_bar_h

    canvas = Image.new("RGB", (total_w, total_h), color=(14, 16, 22))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype(resolve_font_path("bevietnam"), size=26)
        font_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=15)
        font_card_t = ImageFont.truetype(resolve_font_path("bevietnam"), size=16)
        font_card_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=12)
        font_footer = ImageFont.truetype(resolve_font_path("bevietnam"), size=13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_card_t = ImageFont.load_default()
        font_card_sub = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # 1. Main Header
    draw.rectangle([0, 0, total_w, header_bar_h], fill=(20, 24, 34), outline=(38, 44, 60), width=2)
    title_text = "THỰC NGHIỆM CÔ LẬP: KIỂM CHỨNG KHẢ NĂNG ĐIỀU CHỈNH KÍCH THƯỚC CHỮ BẰNG PROMPT"
    bbox_t = font_title.getbbox(title_text)
    draw.text(((total_w - (bbox_t[2] - bbox_t[0])) // 2, 16), title_text, fill=(255, 215, 80), font=font_title)

    sub_1 = "Biến cố định (Cô lập 100%): Cùng 1 ảnh Glyph 'CÀ PHÊ SỮA ĐÁ' (448x160) + Cùng Seed 42 + Cùng Canvas 576x1024"
    bbox_s1 = font_sub.getbbox(sub_1)
    draw.text(((total_w - (bbox_s1[2] - bbox_s1[0])) // 2, 54), sub_1, fill=(195, 205, 225), font=font_sub)

    sub_2 = "Biến thay đổi duy nhất: Prompt điều khiển vật chứa (Khổng lồ choán ngợp vs Biển hiệu gỗ vừa vặn vs Nhãn nhỏ trên tách cafe)"
    bbox_s2 = font_sub.getbbox(sub_2)
    draw.text(((total_w - (bbox_s2[2] - bbox_s2[0])) // 2, 80), sub_2, fill=(150, 235, 175), font=font_sub)

    # Embed Glyph thumbnail in header
    gw, gh = glyph_img.size
    thumb_h = 44
    thumb_w = int(gw * (thumb_h / gh))
    glyph_thumb = glyph_img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    thumb_x = (total_w - thumb_w) // 2
    canvas.paste(glyph_thumb, (thumb_x, 110))
    draw.rectangle([thumb_x - 2, 108, thumb_x + thumb_w + 2, 110 + thumb_h + 2], outline=(255, 215, 80), width=1)

    # 2. Render 3 Panels
    start_y = header_bar_h + margin
    for idx, (item, img) in enumerate(results):
        x = margin + idx * (img_w + gap_x)
        y = start_y

        # Header card
        draw.rectangle([x, y, x + img_w, y + card_header_h], fill=(25, 30, 42), outline=(48, 56, 75), width=2)
        draw.text((x + 14, y + 12), item["title"], fill=(255, 255, 255), font=font_card_t)
        draw.text((x + 14, y + 42), item["subtitle"], fill=item["tag_color"], font=font_card_sub)

        # Image
        canvas.paste(img, (x, y + card_header_h))
        draw.rectangle([x, y + card_header_h, x + img_w, y + card_header_h + img_h], outline=(48, 56, 75), width=2)

    # 3. Footer Bar
    footer_y = total_h - footer_bar_h
    draw.rectangle([0, footer_y, total_w, total_h], fill=(18, 20, 28), outline=(32, 36, 48), width=1)
    footer_str = "Bản quyền R&D Tendoo AI | Nghiên cứu Tương tác Không gian Ngữ nghĩa (Spatial Cross-Attention Scale Binding) | 2026"
    bbox_ft = font_footer.getbbox(footer_str)
    draw.text(
        ((total_w - (bbox_ft[2] - bbox_ft[0])) // 2, footer_y + (footer_bar_h - (bbox_ft[3] - bbox_ft[1])) // 2),
        footer_str,
        fill=(130, 140, 165),
        font=font_footer,
    )

    canvas.save(output_path, quality=95)
    print(f"\n📊 [Scale Isolation 3-Panel Saved] -> {output_path} ({total_w}x{total_h})")


def run_benchmark(
    text: str = "CÀ PHÊ SỮA ĐÁ",
    width: int = 576,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    output_dir: str = "output_scale_isolation_benchmark",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 TENDOO AI - PROMPT SCALE ISOLATION SCIENTIFIC BENCHMARK")
    print(f"📝 Text Under Test   : '{text}' (Fixed across all 3 cases)")
    print(f"📐 Canvas Resolution : 9:16 ({width}x{height})")
    print(f"⚙️  Denoise Config    : {num_steps} steps | CFG {guidance} | Seed: {seed}")
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

    # 2. Render THE SINGLE SHARED GLYPH IMAGE (100% Constant)
    print("\n[1/4] Generating Single Master Glyph Bitmap...")
    glyph_img = create_glyph_image(
        text=text,
        target_width=448,
        target_height=160,
        font_path="bevietnam",
    )
    glyph_file = out_path / "master_glyph_caphesuada.png"
    glyph_img.save(glyph_file)
    print(f"  -> Master Glyph saved: {glyph_file} ({glyph_img.size[0]}x{glyph_img.size[1]})")

    # 3. Load Models Once
    print("\n[2/4] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    # 4. Encode Shared Latents (Strict 1:1 Fair Comparison)
    lat_h = height // 16
    lat_w = width // 16
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    # Encode Single Shared Glyph Tokens
    ref_tokens, ref_ids = encode_glyph_to_incontext_tokens(
        ae=ae, glyph_img=glyph_img, t_offset=10.0, device=device_dit
    )
    print(f"  -> Encoded Shared Glyph Tokens: {ref_tokens.shape[1]} tokens at t=10.0")

    # 5. Execute 3 Scale Benchmark Runs
    generated_results = []
    print("\n[3/4] Executing 3 Prompt Scale Benchmark Runs...")

    for idx, sc in enumerate(SCALE_TEST_CASES):
        print("-" * 75)
        print(f"⚡ [{idx + 1}/{len(SCALE_TEST_CASES)}] Running: {sc['title']}...")
        print(f"   Prompt: '{sc['prompt']}'")

        start_t = time.time()

        with torch.no_grad():
            txt = text_encoder(["", sc["prompt"]])
            txt, txt_ids = batched_prc_txt(txt)
            txt = txt.to(device_dit)
            txt_ids = txt_ids.to(device_dit)

            # Denoise Euler ODE
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
            res_img = Image.fromarray(out_pixels)

        elapsed = time.time() - start_t
        out_file = out_path / f"{sc['id']}.png"
        res_img.save(out_file)
        print(f"   -> Generated in {elapsed:.2f}s | Saved: {out_file.name}")

        generated_results.append((sc, res_img))

    # 6. Stitch Master 3-Panel Image
    print("\n[4/4] Stitching Master 3-Panel Comparison Image...")
    panel_file = out_path / "PROMPT_SCALE_ISOLATION_PANEL.png"
    stitch_3_panel_comparison(
        results=generated_results,
        glyph_img=glyph_img,
        output_path=str(panel_file),
    )

    print("\n" + "=" * 80)
    print("🎉 PROMPT SCALE ISOLATION BENCHMARK COMPLETED!")
    print(f"📁 Output Directory : {out_path.resolve()}")
    print(f"📊 Comparison Panel  : {panel_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tendoo AI - Prompt Scale Isolation Scientific Benchmark")
    parser.add_argument("--text", type=str, default="CÀ PHÊ SỮA ĐÁ", help="Text under test (default: 'CÀ PHÊ SỮA ĐÁ')")
    parser.add_argument("--width", type=int, default=576, help="Canvas width (default: 576)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height (default: 1024)")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="output_scale_isolation_benchmark", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="Model name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint dir")

    args = parser.parse_args()

    run_benchmark(
        text=args.text,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
