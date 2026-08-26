"""
================================================================================
TENDOO AI - TEXT ROPE ROTATION PHASE ALIASING & PERIODIC RECOVERY SWEEP
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Scientific Objective:
- Stress-test the "RoPE Fourier Phase Aliasing & Periodic Recovery Hypothesis"
  on a pure VIETNAMESE TEXT GLYPH (Rich Diacritics: "CHỐNG ỒN CHỦ ĐỘNG").
- Prior Knowledge:
  * In Single-Text mode, text is 100% sharp at t = 10, 20, 30, 40, and degrades at t = 50.
  * Question: Does text recover beyond t = 50 at mathematical Anti-Alias points (180 deg),
    or at BFL discrete multiples of 10 (t = 60, 70, 80), or does it stay collapsed?
- Test 12 Discrete & Continuous Time Offsets:
  1. t = 10.0  (Anchor Baseline - BFL Pretrained Step 1)
  2. t = 40.0  (Safe Zone Boundary - BFL Pretrained Step 4)
  3. t = 44.0  (Alias Peak: 7.0 * 2pi = 43.98, Phase = 1.0 deg)
  4. t = 47.1  (Anti-Alias Peak: 7.5 * 2pi = 47.12, Phase = 180.0 deg)
  5. t = 50.0  (BFL Step 5 / Near Alias: 8.0 * 2pi = 50.27, Phase = 344.8 deg)
  6. t = 53.4  (Anti-Alias Peak: 8.5 * 2pi = 53.41, Phase = 180.0 deg)
  7. t = 56.5  (Alias Peak: 9.0 * 2pi = 56.55, Phase = 2.8 deg)
  8. t = 60.0  (BFL Pretrained Step 6 - Phase = 197.8 deg)
  9. t = 62.8  (Exact 10-Period Alias Peak: 10.0 * 2pi = 62.83, Phase = 0.0 deg)
  10. t = 66.0 (Exact 10.5-Period Anti-Alias Peak: 10.5 * 2pi = 65.97, Phase = 180.0 deg)
  11. t = 70.0 (BFL Pretrained Step 7 - Phase = 15.6 deg)
  12. t = 80.0 (BFL Pretrained Step 8 - Phase = 193.4 deg)
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

    raise RuntimeError("❌ No valid font found for rendering comparison labels!")


def calculate_rope_phase(t_val: float) -> tuple[float, float, str]:
    """
    Calculates the relative phase angle and cosine similarity for omega_0 = 1.0
    relative to Canvas (t = 0.0).
    """
    phase_rad = t_val % (2 * math.pi)
    phase_deg = math.degrees(phase_rad)
    cos_sim = math.cos(phase_rad)

    dist_to_zero = min(phase_deg, 360.0 - phase_deg)

    if dist_to_zero <= 25.0:
        tag = "🔴 ALIAS (Chập Pha)"
    elif abs(phase_deg - 180.0) <= 25.0:
        tag = "🟢 ANTI-ALIAS (Ngược Pha 180°)"
    else:
        tag = "🟡 TRANSITION (Pha Trung Gian)"

    return phase_deg, cos_sim, tag


# 12 Target Test Points
TEST_TIME_OFFSETS = [
    {"t": 10.0, "label": "Anchor Baseline (BFL Step 1)", "type": "canonical"},
    {"t": 40.0, "label": "Safe Upper Bound (BFL Step 4)", "type": "canonical"},
    {"t": 44.0, "label": "Alias Peak (7.0 x 2pi)", "type": "alias"},
    {"t": 47.1, "label": "Anti-Alias 180° (7.5 x 2pi)", "type": "anti_alias"},
    {"t": 50.0, "label": "BFL Step 5 (8.0 x 2pi ≈ 50.27)", "type": "canonical"},
    {"t": 53.4, "label": "Anti-Alias 180° (8.5 x 2pi)", "type": "anti_alias"},
    {"t": 56.5, "label": "Alias Peak (9.0 x 2pi)", "type": "alias"},
    {"t": 60.0, "label": "BFL Step 6 (Discrete Step)", "type": "canonical"},
    {"t": 62.8, "label": "Exact 10-Period Alias (10.0 x 2pi)", "type": "alias"},
    {"t": 66.0, "label": "Exact Anti-Alias 180° (10.5 x 2pi)", "type": "anti_alias"},
    {"t": 70.0, "label": "BFL Step 7 (Discrete Step)", "type": "canonical"},
    {"t": 80.0, "label": "BFL Step 8 (Discrete Step)", "type": "canonical"},
]


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

            line_spacing = int(mid_size * 0.20) * (len(lines) - 1)
            total_h += line_spacing

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

    line_spacing = int(best_size * 0.20)
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)
    total_block_w = max(line_widths)

    if tight_crop:
        # Tight-crop dynamically around the actual text bounding box
        pad_x = max(10, int(total_block_w * padding_ratio))
        pad_y = max(8, int(total_block_h * padding_ratio))
        final_w = total_block_w + 2 * pad_x
        final_h = total_block_h + 2 * pad_y
        final_w = max(32, ((final_w + 15) // 16) * 16)
        final_h = max(32, ((final_h + 15) // 16) * 16)
    else:
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
    t_offset: float,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes a tight-crop glyph image to 4D RoPE coordinate tokens."""
    # Ensure ae device is used for VAE encoding
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


def stitch_text_rope_sweep_grid(
    generated_images: list[tuple[dict, Image.Image]],
    text_content: str,
    output_path: str,
    cols: int = 4,
):
    """Creates a scientific executive grid comparing all time offsets."""
    num_imgs = len(generated_images)
    rows = (num_imgs + cols - 1) // cols

    first_img = generated_images[0][1]
    img_w, img_h = first_img.size

    # Card header and footer height
    card_header_h = 75
    cell_w = img_w
    cell_h = img_h + card_header_h

    margin_x = 24
    margin_y = 24
    gap_x = 18
    gap_y = 22

    header_bar_h = 130
    footer_bar_h = 50

    total_w = margin_x * 2 + cols * cell_w + (cols - 1) * gap_x
    total_h = header_bar_h + margin_y * 2 + rows * cell_h + (rows - 1) * gap_y + footer_bar_h

    canvas = Image.new("RGB", (total_w, total_h), color=(14, 16, 22))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype(resolve_font_path("bevietnam"), size=26)
        font_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=15)
        font_card_t = ImageFont.truetype(resolve_font_path("bevietnam"), size=17)
        font_card_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=13)
        font_footer = ImageFont.truetype(resolve_font_path("bevietnam"), size=13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_card_t = ImageFont.load_default()
        font_card_sub = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # 1. Main Header
    draw.rectangle([0, 0, total_w, header_bar_h], fill=(20, 24, 34), outline=(38, 44, 60), width=2)
    title_text = "KHẢO SÁT CHU KỲ GÓC QUAY 4D RoPE TRÊN GLYPH CHỮ TIẾNG VIỆT"
    bbox_t = font_title.getbbox(title_text)
    draw.text(((total_w - (bbox_t[2] - bbox_t[0])) // 2, 18), title_text, fill=(255, 215, 80), font=font_title)

    sub_text = (
        f'Text Khảo Sát: "{text_content}" | Đối chứng: Chập Pha (Alias) vs Ngược Pha 180° (Anti-Alias) vs Bội Số 10 BFL'
    )
    bbox_s = font_sub.getbbox(sub_text)
    draw.text(((total_w - (bbox_s[2] - bbox_s[0])) // 2, 58), sub_text, fill=(195, 205, 225), font=font_sub)

    sub_text_2 = "Khám phá: Liệu ngoài t=50, chữ có xuất hiện trở lại ở các điểm ngược pha 180° hay các bội số 10 không?"
    bbox_s2 = font_sub.getbbox(sub_text_2)
    draw.text(((total_w - (bbox_s2[2] - bbox_s2[0])) // 2, 85), sub_text_2, fill=(150, 165, 190), font=font_sub)

    # 2. Render Cards
    start_y = header_bar_h + margin_y

    for idx, (item, img) in enumerate(generated_images):
        r = idx // cols
        c = idx % cols

        x = margin_x + c * (cell_w + gap_x)
        y = start_y + r * (cell_h + gap_y)

        t_val = item["t"]
        phase_deg, cos_sim, tag = calculate_rope_phase(t_val)

        # Card header background
        if item["type"] == "canonical":
            bg_color = (25, 45, 65)
            border_color = (45, 90, 130)
            tag_color = (120, 220, 255)
        elif item["type"] == "anti_alias":
            bg_color = (20, 50, 35)
            border_color = (40, 110, 75)
            tag_color = (100, 255, 160)
        else:  # alias
            bg_color = (55, 25, 30)
            border_color = (120, 45, 55)
            tag_color = (255, 120, 130)

        # Draw card header
        draw.rectangle([x, y, x + cell_w, y + card_header_h], fill=bg_color, outline=border_color, width=2)

        # Header Title
        h_title = f"t = {t_val:.1f} ({item['label']})"
        draw.text((x + 12, y + 10), h_title, fill=(255, 255, 255), font=font_card_t)

        # Header Subtitle (Angle & Tag)
        h_sub = f"Pha: {phase_deg:.1f}° | cos: {cos_sim:+.3f} | {tag}"
        draw.text((x + 12, y + 38), h_sub, fill=tag_color, font=font_card_sub)

        # Paste Image
        canvas.paste(img, (x, y + card_header_h))
        draw.rectangle([x, y + card_header_h, x + cell_w, y + cell_h], outline=border_color, width=2)

    # 3. Footer Bar
    footer_y = total_h - footer_bar_h
    draw.rectangle([0, footer_y, total_w, total_h], fill=(18, 20, 28), outline=(32, 36, 48), width=1)
    footer_str = (
        "Thực nghiệm: FLUX.2-klein-base-4B | T2I 9:16 (576x1024) | Steps: 50 | CFG: 4.5 | Seed: 42 | "
        "Strict 1:1 Initial Noise Latent"
    )
    bbox_ft = font_footer.getbbox(footer_str)
    draw.text(
        ((total_w - (bbox_ft[2] - bbox_ft[0])) // 2, footer_y + (footer_bar_h - (bbox_ft[3] - bbox_ft[1])) // 2),
        footer_str,
        fill=(130, 140, 165),
        font=font_footer,
    )

    canvas.save(output_path, quality=95)
    print(f"\n📊 [Master Sweep Grid Saved] -> {output_path} ({total_w}x{total_h})")


def run_text_rope_sweep(
    text: str = "CHỐNG ỒN CHỦ ĐỘNG",
    font: str = "bevietnam",
    prompt: str = (
        "Không gian công nghệ âm thanh hi-fi cao cấp với bề mặt kim loại xước mờ hiện đại, "
        "dòng chữ tiêu đề 3D dập nổi mạ vàng đồng cổ phát sáng sắc nét ăn sâu vào bề mặt kim loại, "
        "ánh sáng studio tương phản cao, đổ bóng chân thực, chi tiết sắc nét"
    ),
    width: int = 576,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    output_dir: str = "output_text_rope_sweep",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    font_path = resolve_font_path(font)

    print("=" * 80)
    print("🔬 TENDOO AI - TEXT ROPE ROTATION PHASE ALIASING BENCHMARK")
    print(f"📝 Target Text   : '{text}' (Font: {Path(font_path).name})")
    print(f"🎨 Clean Prompt  : '{prompt}'")
    print(f"📐 Canvas Ratio  : 9:16 ({width}x{height})")
    print(f"⚙️  Steps / CFG   : {num_steps} steps | CFG {guidance} | Seed: {seed}")
    print(f"🎯 Total Sweeps  : {len(TEST_TIME_OFFSETS)} time offsets to test")
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

    # 2. Render In-Context Glyph Bitmap
    box_w = min(512, int(width * 0.90))
    box_w = (box_w // 16) * 16

    print("\n[1/4] Rendering Glyph Bitmap (Glyph Scaling Law)...")
    glyph_img = create_glyph_image(text, target_width=box_w, target_height=224, font_path=font_path)
    glyph_file = out_path / "glyph_text_preview.png"
    glyph_img.save(glyph_file)
    print(f"  -> Saved Glyph Preview in {glyph_file.resolve()}")

    # 3. Load Models Once
    print("\n[2/4] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    # 4. Encode Text Prompt Once
    print("\n[3/4] Encoding Scene Text Prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    # Shared Initial Noise Latent (Strict 1:1 Fair Comparison)
    lat_h = height // 16
    lat_w = width // 16
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    # 5. Run Sweep Over All Time Offsets
    print("\n[4/4] Executing Sweep Across 12 Time Offsets...")
    generated_images = []

    for idx, item in enumerate(TEST_TIME_OFFSETS):
        t_val = item["t"]
        phase_deg, cos_sim, tag = calculate_rope_phase(t_val)

        print("-" * 75)
        print(
            f"⚡ [{idx + 1}/{len(TEST_TIME_OFFSETS)}] Running t = {t_val:.1f} ({item['label']}) | Phase: {phase_deg:.1f}° | {tag}..."
        )

        start_t = time.time()

        # Encode glyph with this specific t_offset
        ref_tokens, ref_ids = encode_glyph_to_incontext_tokens(
            ae=ae, glyph_img=glyph_img, t_offset=t_val, device=device_dit
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
            res_img = Image.fromarray(out_pixels)

        elapsed = time.time() - start_t
        out_file = out_path / f"t_{t_val:.1f}_{item['type']}.png"
        res_img.save(out_file)
        print(f"  -> Generated in {elapsed:.2f}s | Saved: {out_file.name}")

        generated_images.append((item, res_img))

    # 6. Stitch Master Grid Panel
    print("\n" + "=" * 80)
    print("📊 Stitching Scientific Master Grid Comparison Panel (Cols = 4, Rows = 3)...")
    grid_file = out_path / "TEXT_ROPE_ROTATION_SWEEP_PANEL.png"
    stitch_text_rope_sweep_grid(
        generated_images=generated_images,
        text_content=text,
        output_path=str(grid_file),
        cols=4,
    )

    print("\n" + "=" * 80)
    print("🎉 TEXT ROPE ROTATION BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"📁 Individual Results in: {out_path.resolve()}")
    print(f"📊 Master Panel Image   : {grid_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tendoo AI - Text RoPE Rotation Phase Aliasing & Periodic Recovery Sweep"
    )
    parser.add_argument("--text", type=str, default="CHỐNG ỒN CHỦ ĐỘNG", help="Vietnamese text to probe")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias or path")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Không gian công nghệ âm thanh hi-fi cao cấp với bề mặt kim loại xước mờ hiện đại, "
            "dòng chữ tiêu đề 3D dập nổi mạ vàng đồng cổ phát sáng sắc nét ăn sâu vào bề mặt kim loại, "
            "ánh sáng studio tương phản cao, đổ bóng chân thực, chi tiết sắc nét"
        ),
        help="Clean prompt describing material and lighting",
    )
    parser.add_argument("--width", type=int, default=576, help="Canvas width (default: 576 for 9:16)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height (default: 1024 for 9:16)")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_text_rope_sweep",
        help="Output directory for generated sweep images",
    )
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    run_text_rope_sweep(
        text=args.text,
        font=args.font,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
