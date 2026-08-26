"""
================================================================================
TENDOO AI - PROMPT TECHNIQUE SCIENTIFIC BENCHMARK (3x3 MATRIX - 9:16)
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Objective:
Compare 3 Prompt Engineering Levels across 3 Text Lengths (9:16 Canvas 576x1024):
  - Row 1 (Short Text): "CÀ PHÊ SỮA" (3 words)
  - Row 2 (Medium Text): "CHỐNG ỒN CHỦ ĐỘNG" (4 words, complex diacritics)
  - Row 3 (Long Text): Thơ Tây Tiến 4 câu (28 words, 119 chars)

Columns (Prompt Levels - All strictly omit verbatim text):
  - Col 1 (P1): Zero text mention, no container (Pure scene description)
  - Col 2 (P2): Text style & material mentioned, NO container
  - Col 3 (P3): Text style & material mentioned + Specific physical container
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

# Import Tendoo Core SDK
from tendoo import (
    create_glyph_image,
    encode_glyph_to_incontext_tokens,
    resolve_font_path,
)

# Define the 3x3 Test Matrix
TEST_MATRIX = [
    # Row 0: Short Text
    {
        "row_id": "short",
        "row_name": "TEXT NGẮN (3 Từ)",
        "text": "CÀ PHÊ SỮA",
        "font": "playfair",
        "box_w": 448,
        "box_h": 224,
        "prompts": {
            "p1": (
                "Quán cafe phong cách vintage ấm cúng cổ điển với bàn gỗ mộc, "
                "tách cafe bốc khói le lói ánh sáng hoàng hôn, phong cách điện ảnh, "
                "ánh sáng studio tương phản cao, đổ bóng chân thực"
            ),
            "p2": (
                "Quán cafe phong cách vintage ấm cúng cổ điển với bàn gỗ mộc, tách cafe bốc khói, "
                "dòng chữ tiêu đề 3D dập nổi mạ vàng đồng phát sáng sắc nét tinh xảo, "
                "phong cách điện ảnh, ánh sáng studio tương phản cao, đổ bóng chân thực"
            ),
            "p3": (
                "Quán cafe phong cách vintage ấm cúng, trên bức tường gạch mộc có tấm biển gỗ sồi phẳng cổ kính, "
                "dòng chữ tiêu đề 3D dập nổi mạ vàng đồng cổ sắc nét tinh xảo trên mặt gỗ, "
                "tách cafe bốc khói, ánh sáng studio tương phản cao, đổ bóng chân thực"
            ),
        },
    },
    # Row 1: Medium Text
    {
        "row_id": "medium",
        "row_name": "TEXT TRUNG BÌNH (4 Từ)",
        "text": "CHỐNG ỒN CHỦ ĐỘNG",
        "font": "bevietnam",
        "box_w": 512,
        "box_h": 224,
        "prompts": {
            "p1": (
                "Không gian công nghệ âm thanh hi-fi sci-fi hiện đại với ánh sáng neon xanh lam tím huyền ảo, "
                "góc nhìn điện ảnh, ánh sáng studio tương phản cao, chi tiết sắc nét, đổ bóng chân thực"
            ),
            "p2": (
                "Không gian công nghệ âm thanh hi-fi sci-fi hiện đại với ánh sáng huyền ảo, "
                "dòng chữ tiêu đề đèn neon phát quang màu xanh ngọc sắc nét với quầng sáng dạ quang rực rỡ, "
                "phong cách điện ảnh, ánh sáng tương phản cao, đổ bóng chân thực"
            ),
            "p3": (
                "Không gian công nghệ âm thanh sci-fi hiện đại, tấm bảng kim loại xước mờ công nghiệp ở tiền cảnh, "
                "dòng chữ tiêu đề đèn neon phát quang màu xanh ngọc sắc nét ăn sâu vào mặt kim loại với quầng sáng dạ quang rực rỡ, "
                "ánh sáng studio tương phản cao, đổ bóng chân thực"
            ),
        },
    },
    # Row 2: Long Text
    {
        "row_id": "long",
        "row_name": "TEXT DÀI (Thơ Tây Tiến - 28 Từ)",
        "text": (
            "Sông Mã xa rồi Tây Tiến ơi\n"
            "Nhớ về rừng núi nhớ chơi vơi\n"
            "Sài Khao sương lấp đoàn quân mỏi\n"
            "Mường Lát hoa về trong đêm hơi"
        ),
        "font": "playfair",
        "box_w": 512,
        "box_h": 576,
        "prompts": {
            "p1": (
                "Hậu cảnh núi non Tây Bắc hùng vĩ mây mù hoàng hôn le lói, "
                "rừng già đại ngàn Tây Tiến hùng tráng hoang sơ, "
                "phong cách điện ảnh sử thi cổ trang, ánh sáng studio tương phản cao, chi tiết sắc nét"
            ),
            "p2": (
                "Bốn câu thơ chữ khắc chìm mạ vàng đồng cổ sắc nét tinh xảo, "
                "hậu cảnh núi non Tây Bắc hùng vĩ mây mù hoàng hôn le lói, rừng già đại ngàn hùng tráng hoang sơ, "
                "phong cách điện ảnh sử thi cổ trang, ánh sáng studio tương phản cao"
            ),
            "p3": (
                "Bức vách đá sa thạch cổ kính phẳng sừng sững ở tiền cảnh góc bên, "
                "bốn câu thơ chữ khắc chìm sâu vào mặt đá phẳng mạ vàng đồng cổ sắc nét phủ rêu phong, "
                "hậu cảnh núi non Tây Bắc hùng vĩ mây mù hoàng hôn le lói, phong cách điện ảnh sử thi cổ trang, "
                "ánh sáng studio tương phản cao"
            ),
        },
    },
]

PROMPT_COLS = [
    {
        "col_id": "p1",
        "title": "PROMPT 1: KHÔNG NHẮC TEXT",
        "subtitle": "Chỉ tả bối cảnh / Không vật chứa",
        "header_bg": (45, 30, 40),
        "border_color": (90, 50, 70),
        "tag_color": (255, 150, 180),
    },
    {
        "col_id": "p2",
        "title": "PROMPT 2: NHẮC CHẤT LIỆU TEXT",
        "subtitle": "Chữ 3D/Neon/Khắc chìm / Không vật chứa",
        "header_bg": (25, 45, 55),
        "border_color": (45, 90, 110),
        "tag_color": (120, 230, 255),
    },
    {
        "col_id": "p3",
        "title": "PROMPT 3: CHẤT LIỆU + VẬT CHỨA",
        "subtitle": "Chữ 3D + Bảng gỗ/Bảng kim loại/Vách đá",
        "header_bg": (20, 50, 35),
        "border_color": (40, 110, 75),
        "tag_color": (100, 255, 160),
    },
]


def stitch_3x3_comparison_grid(
    results_matrix: dict,
    output_path: str,
):
    """Stitches the 9 generated images into an executive 3x3 comparison grid."""
    sample_img = results_matrix["short"]["p1"]
    img_w, img_h = sample_img.size

    cols = 3
    rows = 3

    col_header_h = 75
    row_header_w = 90
    margin = 24
    gap_x = 18
    gap_y = 20

    header_bar_h = 130
    footer_bar_h = 50

    cell_w = img_w
    cell_h = img_h + col_header_h

    total_w = margin * 2 + row_header_w + cols * cell_w + (cols - 1) * gap_x
    total_h = header_bar_h + margin * 2 + rows * cell_h + (rows - 1) * gap_y + footer_bar_h

    canvas = Image.new("RGB", (total_w, total_h), color=(14, 16, 22))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype(resolve_font_path("bevietnam"), size=26)
        font_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=15)
        font_col_t = ImageFont.truetype(resolve_font_path("bevietnam"), size=16)
        font_col_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=12)
        font_row = ImageFont.truetype(resolve_font_path("bevietnam"), size=15)
        font_footer = ImageFont.truetype(resolve_font_path("bevietnam"), size=13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_col_t = ImageFont.load_default()
        font_col_sub = ImageFont.load_default()
        font_row = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # 1. Main Header
    draw.rectangle([0, 0, total_w, header_bar_h], fill=(20, 24, 34), outline=(38, 44, 60), width=2)
    title_text = "ĐỐI CHỨNG KHOA HỌC 3 KỸ THUẬT PROMPTING TRÊN 3 ĐỘ DÀI VĂN BẢN (9:16)"
    bbox_t = font_title.getbbox(title_text)
    draw.text(((total_w - (bbox_t[2] - bbox_t[0])) // 2, 18), title_text, fill=(255, 215, 80), font=font_title)

    sub_text = "Thực nghiệm: DiT Base 4B | Seed: 42 (Đồng nhất 100% z_init) | CFG 4.5 | Steps 50 | In-Context t = 10.0"
    bbox_s = font_sub.getbbox(sub_text)
    draw.text(((total_w - (bbox_s[2] - bbox_s[0])) // 2, 58), sub_text, fill=(195, 205, 225), font=font_sub)

    sub_text_2 = "So sánh: P1 (Không nhắc text) vs P2 (Nhắc chất liệu 3D/Neon) vs P3 (Nhắc chất liệu + Vật chứa vật lý)"
    bbox_s2 = font_sub.getbbox(sub_text_2)
    draw.text(((total_w - (bbox_s2[2] - bbox_s2[0])) // 2, 85), sub_text_2, fill=(150, 165, 190), font=font_sub)

    # 2. Render 3x3 Grid
    start_x = margin + row_header_w
    start_y = header_bar_h + margin

    for r_idx, row_item in enumerate(TEST_MATRIX):
        r_id = row_item["row_id"]
        curr_y = start_y + r_idx * (cell_h + gap_y)

        # Draw Vertical Row Header
        row_rect = [margin, curr_y, margin + row_header_w - 12, curr_y + cell_h]
        draw.rectangle(row_rect, fill=(24, 28, 40), outline=(48, 56, 75), width=2)

        # Vertical text drawing
        row_lbl = row_item["row_name"]
        bbox_r = font_row.getbbox(row_lbl)
        rw = bbox_r[2] - bbox_r[0]
        rh = bbox_r[3] - bbox_r[1]

        # Draw row label centered
        draw.text((margin + 10, curr_y + (cell_h - rh) // 2 - 20), row_lbl, fill=(255, 215, 100), font=font_row)
        draw.text((margin + 10, curr_y + (cell_h - rh) // 2 + 10), f"Font: {row_item['font']}", fill=(160, 175, 200), font=font_col_sub)

        for c_idx, col_item in enumerate(PROMPT_COLS):
            c_id = col_item["col_id"]
            curr_x = start_x + c_idx * (cell_w + gap_x)

            img = results_matrix[r_id][c_id]

            # Draw Column/Cell Header
            draw.rectangle(
                [curr_x, curr_y, curr_x + cell_w, curr_y + col_header_h],
                fill=col_item["header_bg"],
                outline=col_item["border_color"],
                width=2,
            )

            draw.text((curr_x + 12, curr_y + 12), col_item["title"], fill=(255, 255, 255), font=font_col_t)
            draw.text((curr_x + 12, curr_y + 40), col_item["subtitle"], fill=col_item["tag_color"], font=font_col_sub)

            # Paste image
            canvas.paste(img, (curr_x, curr_y + col_header_h))
            draw.rectangle(
                [curr_x, curr_y + col_header_h, curr_x + cell_w, curr_y + cell_h],
                outline=col_item["border_color"],
                width=2,
            )

    # 3. Footer Bar
    footer_y = total_h - footer_bar_h
    draw.rectangle([0, footer_y, total_w, total_h], fill=(18, 20, 28), outline=(32, 36, 48), width=1)
    footer_str = (
        "Bản quyền R&D Tendoo AI | Nghiên cứu Động lực học Attention & Ngữ nghĩa Prompt trên FLUX.2 Base 4B | 2026"
    )
    bbox_ft = font_footer.getbbox(footer_str)
    draw.text(
        ((total_w - (bbox_ft[2] - bbox_ft[0])) // 2, footer_y + (footer_bar_h - (bbox_ft[3] - bbox_ft[1])) // 2),
        footer_str,
        fill=(130, 140, 165),
        font=font_footer,
    )

    canvas.save(output_path, quality=95)
    print(f"\n📊 [Master 3x3 Grid Saved] -> {output_path} ({total_w}x{total_h})")


def run_3x3_prompt_benchmark(
    width: int = 576,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    output_dir: str = "output_prompt_3x3_benchmark",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 TENDOO AI - PROMPT TECHNIQUE BENCHMARK (3x3 MATRIX - 9:16)")
    print(f"📐 Canvas Resolution : 9:16 ({width}x{height})")
    print(f"⚙️  Denoise Config    : {num_steps} steps | CFG {guidance} | Seed: {seed}")
    print(f"🎯 Total Generations : 3 Text Lengths x 3 Prompt Levels = 9 Runs")
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

    # 4. Execute 3x3 Matrix Runs
    results_matrix = {"short": {}, "medium": {}, "long": {}}
    run_count = 0

    print("\n[2/3] Executing 9 Benchmark Runs (3x3 Matrix)...")

    for row_item in TEST_MATRIX:
        r_id = row_item["row_id"]
        text_str = row_item["text"]
        font_key = row_item["font"]
        box_w = row_item["box_w"]
        box_h = row_item["box_h"]

        # Render Glyph Bitmap once per row
        glyph_img = create_glyph_image(
            text=text_str,
            target_width=box_w,
            target_height=box_h,
            font_path=font_key,
        )
        glyph_file = out_path / f"glyph_{r_id}.png"
        glyph_img.save(glyph_file)

        # Encode Glyph Tokens to t = 10.0
        ref_tokens, ref_ids = encode_glyph_to_incontext_tokens(
            ae=ae, glyph_img=glyph_img, t_offset=10.0, device=device_dit
        )

        for col_item in PROMPT_COLS:
            c_id = col_item["col_id"]
            prompt_text = row_item["prompts"][c_id]
            run_count += 1

            print("-" * 75)
            print(f"⚡ [{run_count}/9] Running {row_item['row_name']} x {col_item['title']}...")
            print(f"   Prompt: '{prompt_text}'")

            start_t = time.time()

            # Encode Text Prompt via Qwen3
            with torch.no_grad():
                txt = text_encoder(["", prompt_text])
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
            out_file = out_path / f"{r_id}_{c_id}.png"
            res_img.save(out_file)
            print(f"   -> Generated in {elapsed:.2f}s | Saved: {out_file.name}")

            results_matrix[r_id][c_id] = res_img

    # 5. Stitch Master 3x3 Grid
    print("\n[3/3] Stitching Master 3x3 Grid Comparison Panel...")
    grid_file = out_path / "PROMPT_TECHNIQUE_COMPARISON_3X3.png"
    stitch_3x3_comparison_grid(
        results_matrix=results_matrix,
        output_path=str(grid_file),
    )

    print("\n" + "=" * 80)
    print("🎉 3x3 PROMPT TECHNIQUE BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"📁 9 Individual Images in : {out_path.resolve()}")
    print(f"📊 Master 3x3 Grid Panel  : {grid_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tendoo AI - Prompt Technique Scientific Benchmark (3x3 Matrix - 9:16)"
    )
    parser.add_argument("--width", type=int, default=576, help="Canvas width (default: 576 for 9:16)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height (default: 1024 for 9:16)")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_prompt_3x3_benchmark",
        help="Output directory for generated benchmark images",
    )
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    run_3x3_prompt_benchmark(
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
