"""
================================================================================
TENDOO AI - TEXT ATTENTION COMPETITION BENCHMARK (T2I 9:16)
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Scientific Objective:
- Stress-test Cross-Slot Softmax Attention Competition between multiple text blocks
  in pure Text-to-Image (T2I) mode at vertical 9:16 ratio (fast execution).
- STRICT CONDITION: Text Prompt is 100% natural, containing ZERO mentions of
  placement positions (top/bottom) and ZERO surface priors (ribbon, signboard, plaque, screen).
- Compare 2-Slot [t=10, t=20] vs 3-Slot [t=10, t=20, t=30] under identical initial noise.

Passes:
1. Pass 1 (2-Slot): Text 1 (t=10.0) + Text 2 (t=20.0)
2. Pass 2 (3-Slot): Text 1 (t=10.0) + Text 2 (t=20.0) + Text 3 (t=30.0)
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
    "graffiti": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "sedgwick": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "dancing": str(ROOT_DIR / "fonts" / "DancingScript.ttf"),
    "oswald": str(ROOT_DIR / "fonts" / "Oswald.ttf"),
    "cookies": str(ROOT_DIR / "fonts" / "SVN-Cookies.ttf"),
    "brush": str(ROOT_DIR / "fonts" / "SVN-Blow Brush.ttf"),
    "gotham": str(ROOT_DIR / "fonts" / "SVN-Gotham Ultra.otf"),
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
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    padding_ratio: float = 0.08,
) -> Image.Image:
    """
    Renders tight-crop Vietnamese glyph bitmap with automatic line wrapping
    and binary-search font sizing adhering to the Glyph Scaling Law.
    """
    assert target_width > 0 and target_height > 0
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16

    pad_w = int(target_width * padding_ratio)
    pad_h = int(target_height * padding_ratio)
    max_w = target_width - 2 * pad_w
    max_h = target_height - 2 * pad_h

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
        best_font = ImageFont.truetype(font_path, size=24)
        best_lines = candidate_layouts[-1]
        best_size = 24

    img = Image.new("RGB", (target_width, target_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    line_heights = []
    line_widths = []
    for line in best_lines:
        bbox = best_font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(best_size * 0.20)
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)
    curr_y = (target_height - total_block_h) // 2

    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        curr_x = (target_width - lw) // 2
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
    g_arr = np.array(glyph_img).astype(np.float32) / 127.5 - 1.0
    g_tensor = torch.from_numpy(g_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

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


def stitch_competition_comparison_panel(
    pass1_img: Image.Image,
    pass2_img: Image.Image,
    text1: str,
    text2: str,
    text3: str,
    prompt: str,
    output_path: str,
):
    """Creates a high-end side-by-side presentation panel for 9:16 comparison."""
    w, h = pass1_img.size
    header_h = 135
    footer_h = 55
    total_w = w * 2
    total_h = h + header_h + footer_h

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 17, 23))
    draw = ImageDraw.Draw(canvas)

    try:
        font_main = ImageFont.truetype(resolve_font_path("bevietnam"), size=24)
        font_sub = ImageFont.truetype(resolve_font_path("bevietnam"), size=15)
        font_card_title = ImageFont.truetype(resolve_font_path("bevietnam"), size=19)
        font_card_note = ImageFont.truetype(resolve_font_path("bevietnam"), size=14)
        font_footer = ImageFont.truetype(resolve_font_path("bevietnam"), size=13)
    except Exception:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_card_title = ImageFont.load_default()
        font_card_note = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # 1. Main Header
    draw.rectangle([0, 0, total_w, header_h], fill=(22, 26, 36), outline=(40, 46, 62), width=2)
    main_title = "KHẢO SÁT CẠNH TRANH ATTENTION ĐA KHỐI TEXT (T2I 9:16 - CLEAN PROMPT)"
    bbox_m = font_main.getbbox(main_title)
    draw.text(((total_w - (bbox_m[2] - bbox_m[0])) // 2, 16), main_title, fill=(255, 215, 80), font=font_main)

    sub_title = (
        f'Prompt Tự Nhiên: "{prompt[:95]}..." (KHÔNG gợi mở vị trí, KHÔNG gợi mở vật thể đỡ)'
    )
    bbox_s = font_sub.getbbox(sub_title)
    draw.text(((total_w - (bbox_s[2] - bbox_s[0])) // 2, 50), sub_title, fill=(200, 210, 230), font=font_sub)

    # Column 1 Header (2-Slot)
    draw.rectangle([0, 80, w, header_h], fill=(20, 36, 48), outline=(40, 75, 100), width=1)
    col1_title = "🔹 PASS 1: 2-SLOT TEXT (t = 10, 20)"
    bbox_c1 = font_card_title.getbbox(col1_title)
    draw.text(((w - (bbox_c1[2] - bbox_c1[0])) // 2, 85), col1_title, fill=(120, 220, 255), font=font_card_title)
    col1_sub = f'[t=10] "{text1}"  +  [t=20] "{text2}"'
    bbox_c1s = font_card_note.getbbox(col1_sub)
    draw.text(((w - (bbox_c1s[2] - bbox_c1s[0])) // 2, 109), col1_sub, fill=(170, 200, 230), font=font_card_note)

    # Column 2 Header (3-Slot)
    draw.rectangle([w, 80, total_w, header_h], fill=(42, 28, 48), outline=(85, 55, 100), width=1)
    col2_title = "⚡ PASS 2: 3-SLOT TEXT (t = 10, 20, 30)"
    bbox_c2 = font_card_title.getbbox(col2_title)
    draw.text((w + (w - (bbox_c2[2] - bbox_c2[0])) // 2, 85), col2_title, fill=(255, 160, 220), font=font_card_title)
    col2_sub = f'[t=10] "{text1}"  +  [t=20] "{text2}"  +  [t=30] "{text3}"'
    bbox_c2s = font_card_note.getbbox(col2_sub)
    draw.text((w + (w - (bbox_c2s[2] - bbox_c2s[0])) // 2, 109), col2_sub, fill=(230, 190, 215), font=font_card_note)

    # 2. Paste Images
    canvas.paste(pass1_img, (0, header_h))
    canvas.paste(pass2_img, (w, header_h))

    # Divider line
    draw.line([(w, header_h), (w, header_h + h)], fill=(60, 68, 88), width=3)

    # 3. Footer Bar
    footer_y = header_h + h
    draw.rectangle([0, footer_y, total_w, total_h], fill=(18, 20, 28), outline=(35, 40, 55), width=1)
    footer_text = (
        "Thực nghiệm: FLUX.2-klein-base-4B | T2I 9:16 (576x1024) | Steps: 50 | CFG: 4.5 | Seed: 42 | "
        "Strict Pure Prompting (Zero Surface Carrier Priors)"
    )
    bbox_ft = font_footer.getbbox(footer_text)
    draw.text(
        ((total_w - (bbox_ft[2] - bbox_ft[0])) // 2, footer_y + (footer_h - (bbox_ft[3] - bbox_ft[1])) // 2),
        footer_text,
        fill=(130, 140, 165),
        font=font_footer,
    )

    canvas.save(output_path, quality=95)
    print(f"\n📊 [Executive Side-by-Side Panel Saved] -> {output_path} ({total_w}x{total_h})")


def run_text_attention_competition(
    text1: str = "CÀ PHÊ SỮA ĐÁ",
    text2: str = "ĐẬM ĐÀ HƯƠNG VỊ VIỆT",
    text3: str = "MUA 1 TẶNG 1 HÔM NAY",
    font1: str = "sedgwick",
    font2: str = "bevietnam",
    font3: str = "pacifico",
    prompt: str = (
        "Quán cà phê phong cách vintage ấm cúng với quầy bar gỗ cổ điển, "
        "ánh đèn vàng dịu nhẹ, vài chậu cây xanh nhỏ, ly cà phê phin truyền thống bốc khói nhẹ trên mặt bàn, "
        "phong cách chụp ảnh điện ảnh nghệ thuật, ánh sáng studio tương phản cao, đổ bóng tự nhiên, chi tiết sắc nét"
    ),
    width: int = 576,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    output_dir: str = "output_text_competition_9_16",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    font_path1 = resolve_font_path(font1)
    font_path2 = resolve_font_path(font2)
    font_path3 = resolve_font_path(font3)

    print("=" * 80)
    print("⚡ TENDOO AI - TEXT ATTENTION COMPETITION EXPERIMENT (T2I 9:16)")
    print(f"📝 Text 1 [t=10.0] : '{text1}' (Font: {Path(font_path1).name})")
    print(f"📝 Text 2 [t=20.0] : '{text2}' (Font: {Path(font_path2).name})")
    print(f"📝 Text 3 [t=30.0] : '{text3}' (Font: {Path(font_path3).name})")
    print(f"🎨 Clean Prompt    : '{prompt}'")
    print(f"📐 Canvas Ratio    : 9:16 ({width}x{height})")
    print(f"⚙️  Steps / CFG     : {num_steps} steps | CFG {guidance} | Seed: {seed}")
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

    # 2. Render In-Context Glyphs
    box_w = min(512, int(width * 0.90))
    box_w = (box_w // 16) * 16

    print("\n[1/4] Rendering Glyph Bitmaps (Adhering to Glyph Scaling Law)...")
    glyph1 = create_glyph_image(text1, target_width=box_w, target_height=224, font_path=font_path1)
    glyph2 = create_glyph_image(text2, target_width=box_w, target_height=192, font_path=font_path2)
    glyph3 = create_glyph_image(text3, target_width=min(448, box_w), target_height=160, font_path=font_path3)

    glyph1.save(out_path / "glyph_text1_preview.png")
    glyph2.save(out_path / "glyph_text2_preview.png")
    glyph3.save(out_path / "glyph_text3_preview.png")
    print(f"  -> Saved Glyph Previews in {out_path.resolve()}")

    # 3. Load Models Once
    print("\n[2/4] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    # 4. Encode Clean Prompt Once
    print("\n[3/4] Encoding Clean Scene Text Prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    # 5. Encode Reference Glyphs into 4D RoPE Tokens
    tok1, ids1 = encode_glyph_to_incontext_tokens(ae=ae, glyph_img=glyph1, t_offset=10.0, device=device_dit)
    tok2, ids2 = encode_glyph_to_incontext_tokens(ae=ae, glyph_img=glyph2, t_offset=20.0, device=device_dit)
    tok3, ids3 = encode_glyph_to_incontext_tokens(ae=ae, glyph_img=glyph3, t_offset=30.0, device=device_dit)

    # Shared Initial Noise Latent
    lat_h = height // 16
    lat_w = width // 16
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    # =========================================================================
    # PASS 1: 2-SLOT TEXT (t=10, t=20)
    # =========================================================================
    print("\n" + "-" * 70)
    print("⚡ [PASS 1/2] Running 2-Slot Text Benchmark: [t=10 (Headline) + t=20 (Subtitle)]...")
    print("-" * 70)
    start_t1 = time.time()

    ref_tokens_2slot = torch.cat([tok1, tok2], dim=1)
    ref_ids_2slot = torch.cat([ids1, ids2], dim=1)

    with torch.no_grad():
        out_latent1 = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens_2slot,
            img_cond_seq_ids=ref_ids_2slot,
        )

        out_latent1 = rearrange(out_latent1, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_pixels1 = ae.decode(out_latent1.to(device_ae))
        out_pixels1 = ((out_pixels1[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        pass1_img = Image.fromarray(out_pixels1)

    elapsed1 = time.time() - start_t1
    pass1_file = out_path / "pass1_2slots_t10_t20.png"
    pass1_img.save(pass1_file)
    print(f"  -> Pass 1 completed in {elapsed1:.2f}s | Saved: {pass1_file.name}")

    # =========================================================================
    # PASS 2: 3-SLOT TEXT (t=10, t=20, t=30)
    # =========================================================================
    print("\n" + "-" * 70)
    print("⚡ [PASS 2/2] Running 3-Slot Text Benchmark: [t=10 + t=20 + t=30 (CTA Badge)]...")
    print("-" * 70)
    start_t2 = time.time()

    ref_tokens_3slot = torch.cat([tok1, tok2, tok3], dim=1)
    ref_ids_3slot = torch.cat([ids1, ids2, ids3], dim=1)

    with torch.no_grad():
        out_latent2 = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens_3slot,
            img_cond_seq_ids=ref_ids_3slot,
        )

        out_latent2 = rearrange(out_latent2, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_pixels2 = ae.decode(out_latent2.to(device_ae))
        out_pixels2 = ((out_pixels2[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        pass2_img = Image.fromarray(out_pixels2)

    elapsed2 = time.time() - start_t2
    pass2_file = out_path / "pass2_3slots_t10_t20_t30.png"
    pass2_img.save(pass2_file)
    print(f"  -> Pass 2 completed in {elapsed2:.2f}s | Saved: {pass2_file.name}")

    # =========================================================================
    # STITCH SIDE-BY-SIDE PRESENTATION PANEL
    # =========================================================================
    print("\n[4/4] Stitching Executive Side-by-Side Presentation Panel...")
    comparison_file = out_path / "TEXT_ATTENTION_COMPETITION_9_16_COMPARISON.png"
    stitch_competition_comparison_panel(
        pass1_img=pass1_img,
        pass2_img=pass2_img,
        text1=text1,
        text2=text2,
        text3=text3,
        prompt=prompt,
        output_path=str(comparison_file),
    )

    print("\n" + "=" * 80)
    print("🎉 TEXT ATTENTION COMPETITION BENCHMARK COMPLETED!")
    print(f"📸 1. Pass 1 (2-Slot: t=10, 20)      : {pass1_file}")
    print(f"📸 2. Pass 2 (3-Slot: t=10, 20, 30)  : {pass2_file}")
    print(f"📊 3. EXECUTIVE COMPARISON PANEL    : {comparison_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tendoo AI - Text Attention Competition Benchmark (T2I 9:16 Clean Prompt)"
    )
    parser.add_argument("--text1", type=str, default="CÀ PHÊ SỮA ĐÁ", help="Text 1 at t=10.0 (Headline)")
    parser.add_argument("--text2", type=str, default="ĐẬM ĐÀ HƯƠNG VỊ VIỆT", help="Text 2 at t=20.0 (Subtitle)")
    parser.add_argument("--text3", type=str, default="MUA 1 TẶNG 1 HÔM NAY", help="Text 3 at t=30.0 (CTA Badge)")
    parser.add_argument("--font1", type=str, default="sedgwick", help="Font for Text 1 (default: sedgwick)")
    parser.add_argument("--font2", type=str, default="bevietnam", help="Font for Text 2 (default: bevietnam)")
    parser.add_argument("--font3", type=str, default="pacifico", help="Font for Text 3 (default: pacifico)")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Quán cà phê phong cách vintage ấm cúng với quầy bar gỗ cổ điển, "
            "ánh đèn vàng dịu nhẹ, vài chậu cây xanh nhỏ, ly cà phê phin truyền thống bốc khói nhẹ trên mặt bàn, "
            "phong cách chụp ảnh điện ảnh nghệ thuật, ánh sáng studio tương phản cao, đổ bóng tự nhiên, chi tiết sắc nét"
        ),
        help="Clean scene text prompt (NO position/carrier instructions)",
    )
    parser.add_argument("--width", type=int, default=576, help="Canvas width in pixels (default: 576 for 9:16)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height in pixels (default: 1024 for 9:16)")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_text_competition_9_16",
        help="Output directory for generated benchmark passes",
    )
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    run_text_attention_competition(
        text1=args.text1,
        text2=args.text2,
        text3=args.text3,
        font1=args.font1,
        font2=args.font2,
        font3=args.font3,
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
