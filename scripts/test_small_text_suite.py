"""
================================================================================
TENDOO AI - SMALL & SUBTLE TEXT BENCHMARK SUITE (9:16)
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Scientific Objective:
Stress-test the model's ability to render "SMALL / TINY / SUBTLE" typography:
  1. Cosmetics Serum Label: "CHIẾT XUẤT THIÊN NHIÊN" (Chữ nhỏ in trên thân lọ serum)
  2. Art Gallery Bottom Corner: "BỘ SƯU TẬP MÙA HÈ" (Chữ siêu nhỏ khiêm tốn ở góc đáy)
  3. Tech Micro-Engraving: "CÔNG NGHỆ KHÔNG DÂY" (Chữ in laser siêu nhỏ trên vỏ tai nghe)
  4. Luxury Debossed Card: "KIẾN TRÚC SÁNG TẠO" (Chữ dập chìm mạ bạc nhỏ tinh tế trên danh thiếp)

Key Observation:
- Check if Prompt keywords ('tiny', 'small', 'delicate', 'discreet', 'micro')
  successfully shrink the text rendering size in the generated image.
- Check diacritic survival under VAE 16x compression when text is physically small.
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

# Built-in Font Registry with Friendly Aliases
FONT_REGISTRY = {
    "bevietnam": str(ROOT_DIR / "fonts" / "BeVietnamPro-Black.ttf"),
    "playfair": str(ROOT_DIR / "fonts" / "PlayfairDisplay.ttf"),
    "anton": str(ROOT_DIR / "fonts" / "Anton-Regular.ttf"),
    "pacifico": str(ROOT_DIR / "fonts" / "Pacifico-Regular.ttf"),
    "sedgwick": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "dancing": str(ROOT_DIR / "fonts" / "DancingScript.ttf"),
    "oswald": str(ROOT_DIR / "fonts" / "Oswald.ttf"),
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
    target_width: int = 512,
    target_height: int = 224,
    font_path: str | None = None,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    padding_ratio: float = 0.08,
    tight_crop: bool = True,
) -> Image.Image:
    """
    Renders TRUE TIGHT-CROP Vietnamese glyph bitmap with automatic line wrapping,
    binary-search font sizing, and exact bounding-box cropping snapped to multiples of 16.
    """
    assert target_width > 0 and target_height > 0
    envelope_w = (target_width // 16) * 16
    envelope_h = (target_height // 16) * 16

    font_path = resolve_font_path(font_path)

    pad_w = int(envelope_w * padding_ratio)
    pad_h = int(envelope_h * padding_ratio)
    max_w = envelope_w - 2 * pad_w
    max_h = envelope_h - 2 * pad_h

    text = text.replace("\\n", "\n")

    if "\n" in text:
        candidate_layouts = [[line.strip() for line in text.split("\n") if line.strip()]]
    else:
        words = text.split()
        candidate_layouts = []
        if len(words) >= 4:
            mid = len(words) // 2
            candidate_layouts.append([" ".join(words[:mid]), " ".join(words[mid:])])
        if len(words) >= 6:
            p1 = len(words) // 3
            p2 = 2 * len(words) // 3
            candidate_layouts.append([" ".join(words[:p1]), " ".join(words[p1:p2]), " ".join(words[p2:])])
        candidate_layouts.append([text])

    best_font = None
    best_lines = None
    best_size = 0

    spacing_ratio = 0.32 if len(candidate_layouts[0]) >= 2 else 0.20

    for lines in candidate_layouts:
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

            curr_spacing = int(mid_size * spacing_ratio) * (len(lines) - 1)
            total_h += curr_spacing

            if max_line_w <= max_w and total_h <= max_h:
                opt_font = test_font
                opt_size = mid_size
                low = mid_size + 1
            else:
                high = mid_size - 1

        if opt_size > best_size:
            best_size = opt_size
            best_font = opt_font
            best_lines = lines

    if best_font is None:
        best_font = ImageFont.truetype(font_path, size=20)
        best_lines = candidate_layouts[-1]
        best_size = 20

    line_heights = []
    line_widths = []
    for line in best_lines:
        bbox = best_font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(best_size * spacing_ratio)
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)
    total_block_w = max(line_widths)

    if tight_crop and len(best_lines) == 1:
        # Single-line: tight crop height and width
        pad_x = max(10, int(total_block_w * padding_ratio))
        pad_y = max(8, int(total_block_h * padding_ratio))
        final_w = total_block_w + 2 * pad_x
        final_h = total_block_h + 2 * pad_y
        final_w = max(32, ((final_w + 15) // 16) * 16)
        final_h = max(32, ((final_h + 15) // 16) * 16)
    else:
        # Multi-line (>= 2 lines): PRESERVE ENVELOPE HEIGHT to guarantee latent tokens per line
        final_w = envelope_w
        final_h = envelope_h

    img = Image.new("RGB", (final_w, final_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    curr_y = (final_h - total_block_h) // 2

    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        curr_x = (final_w - lw) // 2
        bbox = best_font.getbbox(line)
        draw.text((curr_x - bbox[0], curr_y - bbox[1]), line, fill=text_color, font=best_font)
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


# 4 Diverse Small & Subtle Text Test Cases
SMALL_TEXT_CASES = [
    {
        "id": "cosmetics_label",
        "title": "CASE 1: CHỮ NHỎ IN TRÊN LỌ SERUM",
        "subtitle": "Cosmetic Label / Delicate Minimalist",
        "text": "CHIẾT XUẤT THIÊN NHIÊN",
        "font": "playfair",
        "prompt": (
            "Lọ serum mỹ phẩm thủy tinh mờ cao cấp đặt trang trọng trên bệ đá travertine màu kem, "
            "trên thân lọ thủy tinh có dòng chữ thương hiệu kích thước nhỏ thanh mảnh tinh tế màu vàng ánh kim "
            "(small delicate minimalist gold foiled typography on bottle), "
            "ánh sáng studio dịu nhẹ tương phản cao, đổ bóng chân thực, chi tiết sắc nét 8k"
        ),
        "tag_color": (255, 215, 120),
    },
    {
        "id": "gallery_corner",
        "title": "CASE 2: CHỮ SIÊU NHỎ Ở GÓC ĐÁY",
        "subtitle": "Discreet Corner Text / Gallery Minimal",
        "text": "BỘ SƯU TẬP MÙA HÈ",
        "font": "bevietnam",
        "prompt": (
            "Bức tranh sơn mài nghệ thuật đương đại khổ lớn treo trên bức tường gallery trắng muốt phẳng lặng, "
            "dòng chữ tiêu đề kích thước siêu nhỏ khiêm tốn in chìm tinh xảo ở góc đáy phía dưới "
            "(tiny subtle discreet minimalist text in bottom corner), "
            "không gian tối giản thanh lịch, ánh đèn rọi spotlight nghệ thuật, chi tiết sắc nét"
        ),
        "tag_color": (160, 210, 255),
    },
    {
        "id": "tech_laser",
        "title": "CASE 3: CHỮ KHẮC LASER SIÊU NHỎ",
        "subtitle": "Micro Laser-Etched / Tech Gadget",
        "text": "CÔNG NGHỆ KHÔNG DÂY",
        "font": "bevietnam",
        "prompt": (
            "Hộp sạc tai nghe không dây cao cấp màu đen nhám đặt trên mặt kim loại xước mờ công nghiệp, "
            "trên bề mặt hộp sạc có dòng chữ in laser kích thước siêu nhỏ màu xám bạc sắc lẹm "
            "(micro laser-etched subtle metallic gray text), "
            "góc chụp cận cảnh macro, ánh sáng viền xanh neon công nghệ tương phản cao, chi tiết cực kỳ sắc nét"
        ),
        "tag_color": (100, 255, 220),
    },
    {
        "id": "luxury_card",
        "title": "CASE 4: CHỮ DẬP CHÌM NHỎ TRÊN DANH THIẾP",
        "subtitle": "Debossed Foil / Luxury Business Card",
        "text": "KIẾN TRÚC SÁNG TẠO",
        "font": "playfair",
        "prompt": (
            "Tấm danh thiếp giấy mỹ thuật dập gân màu đen nhung cao cấp đặt trên mặt bàn gỗ sồi, "
            "dòng chữ tiêu đề kích thước nhỏ dập chìm mạ bạc sắc lẹm tinh tế ăn sâu vào thớ giấy "
            "(small debossed silver foil typography), "
            "ánh sáng xiên đổ bóng vi mô chân thực, phong cách tối giản sang trọng đẳng cấp"
        ),
        "tag_color": (220, 225, 240),
    },
]


def stitch_2x2_comparison_panel(
    results: list[tuple[dict, Image.Image]],
    output_path: str,
):
    """Stitches the 4 generated images into an executive 2x2 comparison panel."""
    sample_img = results[0][1]
    img_w, img_h = sample_img.size

    cols = 2
    rows = 2

    card_header_h = 75
    margin = 24
    gap_x = 20
    gap_y = 22

    header_bar_h = 130
    footer_bar_h = 50

    cell_w = img_w
    cell_h = img_h + card_header_h

    total_w = margin * 2 + cols * cell_w + (cols - 1) * gap_x
    total_h = header_bar_h + margin * 2 + rows * cell_h + (rows - 1) * gap_y + footer_bar_h

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
    title_text = "KHẢO SÁT NĂNG LỰC SINH CHỮ NHỎ & TINH TẾ BẰNG PROMPT (SMALL TEXT SUITE)"
    bbox_t = font_title.getbbox(title_text)
    draw.text(((total_w - (bbox_t[2] - bbox_t[0])) // 2, 18), title_text, fill=(255, 215, 80), font=font_title)

    sub_text = "Thực nghiệm: Lọ mỹ phẩm Serum | Góc đáy Gallery | Khắc Laser tai nghe | Dập chìm Danh thiếp"
    bbox_s = font_sub.getbbox(sub_text)
    draw.text(((total_w - (bbox_s[2] - bbox_s[0])) // 2, 58), sub_text, fill=(195, 205, 225), font=font_sub)

    sub_text_2 = "Kiểm tra: DiT có thu nhỏ tỷ lệ chữ theo lệnh Prompt ('small', 'tiny', 'delicate', 'micro') hay không?"
    bbox_s2 = font_sub.getbbox(sub_text_2)
    draw.text(((total_w - (bbox_s2[2] - bbox_s2[0])) // 2, 85), sub_text_2, fill=(150, 165, 190), font=font_sub)

    # 2. Render 2x2 Grid
    start_y = header_bar_h + margin

    for idx, (item, img) in enumerate(results):
        r = idx // cols
        c = idx % cols

        x = margin + c * (cell_w + gap_x)
        y = start_y + r * (cell_h + gap_y)

        # Card header
        draw.rectangle([x, y, x + cell_w, y + card_header_h], fill=(25, 30, 42), outline=(48, 56, 75), width=2)
        draw.text((x + 14, y + 12), item["title"], fill=(255, 255, 255), font=font_card_t)
        draw.text((x + 14, y + 40), f"{item['subtitle']} | Text: \"{item['text']}\"", fill=item["tag_color"], font=font_card_sub)

        # Image
        canvas.paste(img, (x, y + card_header_h))
        draw.rectangle([x, y + card_header_h, x + cell_w, y + cell_h], outline=(48, 56, 75), width=2)

    # 3. Footer Bar
    footer_y = total_h - footer_bar_h
    draw.rectangle([0, footer_y, total_w, total_h], fill=(18, 20, 28), outline=(32, 36, 48), width=1)
    footer_str = "Bản quyền R&D Tendoo AI | Nghiên cứu Năng lực Biểu diễn Kiểu chữ Tinh xảo & Siêu nhỏ | 2026"
    bbox_ft = font_footer.getbbox(footer_str)
    draw.text(
        ((total_w - (bbox_ft[2] - bbox_ft[0])) // 2, footer_y + (footer_bar_h - (bbox_ft[3] - bbox_ft[1])) // 2),
        footer_str,
        fill=(130, 140, 165),
        font=font_footer,
    )

    canvas.save(output_path, quality=95)
    print(f"\n📊 [Small Text Master Panel Saved] -> {output_path} ({total_w}x{total_h})")


def run_benchmark(
    width: int = 576,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    output_dir: str = "output_small_text_suite",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 TENDOO AI - SMALL & SUBTLE TEXT BENCHMARK SUITE (9:16)")
    print(f"📐 Canvas Resolution : 9:16 ({width}x{height})")
    print(f"⚙️  Denoise Config    : {num_steps} steps | CFG {guidance} | Seed: {seed}")
    print(f"🎯 Test Cases        : {len(SMALL_TEXT_CASES)} subtle typography scenarios")
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
    print("\n[1/3] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    # 3. Prepare Shared Initial Noise Latent (Strict 1:1 Fair Comparison)
    lat_h = height // 16
    lat_w = width // 16
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    # 4. Execute 4 Benchmark Runs
    generated_results = []
    print("\n[2/3] Executing 4 Small Text Benchmark Runs...")

    for idx, sc in enumerate(SMALL_TEXT_CASES):
        print("-" * 75)
        print(f"⚡ [{idx + 1}/{len(SMALL_TEXT_CASES)}] Running: {sc['title']}...")
        print(f"   Text   : '{sc['text']}' (Font: {sc['font']})")
        print(f"   Prompt : '{sc['prompt']}'")

        start_t = time.time()

        # Render True Tight-Crop Glyph
        glyph_img = create_glyph_image(
            text=sc["text"],
            target_width=min(width - 64, 480),
            target_height=200,
            font_path=sc["font"],
            tight_crop=True,
        )
        glyph_file = out_path / f"glyph_{sc['id']}.png"
        glyph_img.save(glyph_file)

        # Encode Glyph Tokens to t = 10.0
        ref_tokens, ref_ids = encode_glyph_to_incontext_tokens(
            ae=ae, glyph_img=glyph_img, t_offset=10.0, device=device_dit
        )

        # Encode Text Prompt via Qwen3
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

    # 5. Stitch Master 2x2 Grid
    print("\n[3/3] Stitching Master 2x2 Grid Comparison Panel...")
    grid_file = out_path / "SMALL_TEXT_COMPARISON_PANEL.png"
    stitch_2x2_comparison_panel(
        results=generated_results,
        output_path=str(grid_file),
    )

    print("\n" + "=" * 80)
    print("🎉 SMALL TEXT BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"📁 Individual Results in: {out_path.resolve()}")
    print(f"📊 Master Panel Image   : {grid_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tendoo AI - Small & Subtle Text Benchmark Suite"
    )
    parser.add_argument("--width", type=int, default=576, help="Canvas width (default: 576 for 9:16)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height (default: 1024 for 9:16)")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_small_text_suite",
        help="Output directory for generated benchmark images",
    )
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    run_benchmark(
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
