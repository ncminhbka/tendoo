"""
================================================================================
TENDOO AI - PROBLEM STATEMENT EXPERIMENT & SIDE-BY-SIDE COMPARISON
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Objective:
- Compare Direct Prompting (Baseline) vs Tendoo AI In-Context Glyph Pipeline.
- Target: Medium-length Vietnamese text seamlessly engraved/carved/embedded into materials.
- Produce high-resolution side-by-side comparison panel for Executive Report & Slides.

Methodology:
1. Pass A (Baseline): Direct Prompt to FLUX.2 Base 4B (literal text in prompt, no reference tokens).
2. Pass B (Tendoo AI): In-Context Glyph Conditioning at t=10.0 + Clean material prompt (no literal text).
3. Stitched Output: Side-by-side labeled presentation panel ready for Slide / Report.
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
    "playfair": str(ROOT_DIR / "fonts" / "PlayfairDisplay.ttf"),
    "bevietnam": str(ROOT_DIR / "fonts" / "BeVietnamPro-Black.ttf"),
    "anton": str(ROOT_DIR / "fonts" / "Anton-Regular.ttf"),
    "pacifico": str(ROOT_DIR / "fonts" / "Pacifico-Regular.ttf"),
    "graffiti": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "sedgwick": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "dancing": str(ROOT_DIR / "fonts" / "DancingScript.ttf"),
    "oswald": str(ROOT_DIR / "fonts" / "Oswald.ttf"),
    "gotham": str(ROOT_DIR / "fonts" / "SVN-Gotham Ultra.otf"),
    "cookies": str(ROOT_DIR / "fonts" / "SVN-Cookies.ttf"),
    "brush": str(ROOT_DIR / "fonts" / "SVN-Blow Brush.ttf"),
}

# Material Presets
MATERIAL_PRESETS = {
    "sandstone_carved": {
        "name": "Khắc chìm mạ vàng trên vách đá sa thạch cổ kính",
        "baseline_prompt_template": (
            "Bức vách đá sa thạch cổ kính phủ rêu phong sừng sững ở tiền cảnh, "
            "ở chính giữa khắc chìm sâu vào thớ đá dòng chữ tiếng Việt '{text}' "
            "mạ vàng đồng cổ sắc nét tinh xảo ăn sâu vào vân đá sa thạch thô ráp, "
            "hậu cảnh núi non mây mù hoàng hôn le lói, phong cách điện ảnh sử thi cổ trang, "
            "ánh sáng studio tương phản cao, đổ bóng chân thực, chi tiết sắc nét"
        ),
        "tendoo_prompt": (
            "Bức vách đá sa thạch cổ kính phủ rêu phong sừng sững ở tiền cảnh, "
            "dòng chữ tiêu đề khắc chìm sâu vào thớ đá mạ vàng đồng cổ sắc nét tinh xảo "
            "ăn sâu vào vân đá sa thạch thô ráp, "
            "hậu cảnh núi non mây mù hoàng hôn le lói, phong cách điện ảnh sử thi cổ trang, "
            "ánh sáng studio tương phản cao, đổ bóng chân thực, chi tiết sắc nét"
        ),
        "default_font": "playfair",
    },
    "wood_engraved": {
        "name": "Khắc laser chìm trên thớt gỗ sồi mộc mạc vintage",
        "baseline_prompt_template": (
            "Tấm thớt gỗ sồi mộc mạc nguyên khối vân gỗ tự nhiên đặt trên bàn cafe vintage, "
            "ở giữa khắc laser chìm sâu vào thớ gỗ dòng chữ tiếng Việt '{text}' "
            "cháy cạnh tinh xảo sắc nét ăn sâu vào từng thớ gỗ, xung quanh vài hạt cafe rang mộc, "
            "ánh sáng vàng ấm dịu studio, phong cách chụp cận cảnh macro tĩnh vật, chi tiết sắc nét"
        ),
        "tendoo_prompt": (
            "Tấm thớt gỗ sồi mộc mạc nguyên khối vân gỗ tự nhiên đặt trên bàn cafe vintage, "
            "dòng chữ tiêu đề khắc laser chìm sâu vào thớ gỗ cháy cạnh tinh xảo "
            "ăn sâu vào từng thớ gỗ, xung quanh vài hạt cafe rang mộc, "
            "ánh sáng vàng ấm dịu studio, phong cách chụp cận cảnh macro tĩnh vật, chi tiết sắc nét"
        ),
        "default_font": "bevietnam",
    },
    "metal_embossed": {
        "name": "Dập nổi kim loại đồng xước trên tấm thép công nghiệp",
        "baseline_prompt_template": (
            "Tấm thép không gỉ xước mờ công nghiệp hiện đại sừng sững ở tiền cảnh, "
            "ở chính giữa dập nổi kim loại nguyên khối dòng chữ tiếng Việt '{text}' "
            "vát cạnh sắc sảo mạ đồng xước bóng bẩy ăn sâu vào bề mặt kim loại, "
            "ánh sáng đèn LED xanh cyber tương phản mạnh, phong cách tương lai sci-fi, "
            "đổ bóng 3D chân thực, chi tiết kim khí sắc nét"
        ),
        "tendoo_prompt": (
            "Tấm thép không gỉ xước mờ công nghiệp hiện đại sừng sững ở tiền cảnh, "
            "dòng chữ tiêu đề dập nổi kim loại nguyên khối vát cạnh sắc sảo "
            "mạ đồng xước bóng bẩy ăn sâu vào bề mặt kim loại, "
            "ánh sáng đèn LED xanh cyber tương phản mạnh, phong cách tương lai sci-fi, "
            "đổ bóng 3D chân thực, chi tiết kim khí sắc nét"
        ),
        "default_font": "anton",
    },
    "leather_stamped": {
        "name": "Dập nhiệt chìm trên bề mặt da thuộc thủ công cao cấp",
        "baseline_prompt_template": (
            "Bề mặt sổ da thuộc thủ công màu nâu sáp cổ điển đặt phẳng ở góc nghiêng nhẹ, "
            "ở giữa dập nhiệt chìm sâu dòng chữ tiếng Việt '{text}' "
            "ép nhũ vàng gold tinh xảo chìm sâu vào bề mặt da thuộc sần sùi sang trọng, "
            "ánh sáng studio mềm mại, phong cách thương gia cao cấp, đổ bóng vi mô sắc nét"
        ),
        "tendoo_prompt": (
            "Bề mặt sổ da thuộc thủ công màu nâu sáp cổ điển đặt phẳng ở góc nghiêng nhẹ, "
            "dòng chữ tiêu đề dập nhiệt chìm sâu ép nhũ vàng gold tinh xảo "
            "chìm sâu vào bề mặt da thuộc sần sùi sang trọng, "
            "ánh sáng studio mềm mại, phong cách thương gia cao cấp, đổ bóng vi mô sắc nét"
        ),
        "default_font": "playfair",
    },
}


def resolve_font_path(font_name_or_path: str | None) -> str:
    """Resolves font alias or validates file path."""
    if font_name_or_path:
        key = font_name_or_path.lower().strip()
        if key in FONT_REGISTRY and os.path.exists(FONT_REGISTRY[key]):
            return FONT_REGISTRY[key]
        if os.path.exists(font_name_or_path):
            return font_name_or_path

    # Fallback to defaults
    for p in [
        FONT_REGISTRY["playfair"],
        FONT_REGISTRY["bevietnam"],
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

            line_spacing = int(mid_size * 0.22) * (len(lines) - 1)
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

    line_spacing = int(best_size * 0.22)
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
    t_offset: float = 10.0,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes a tight-crop glyph image to 4D RoPE coordinate tokens."""
    g_arr = np.array(glyph_img).astype(np.float32) / 127.5 - 1.0
    g_tensor = torch.from_numpy(g_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        g_latent = ae.encode(g_tensor)

    ref_tokens, _ = prc_img(g_latent[0])
    ref_tokens = ref_tokens.unsqueeze(0)

    g_h, g_w = g_latent.shape[2], g_latent.shape[3]
    t_coords = torch.full((g_h, g_w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(g_h, dtype=torch.float32, device=device).unsqueeze(1).expand(g_h, g_w)
    w_coords = torch.arange(g_w, dtype=torch.float32, device=device).unsqueeze(0).expand(g_h, g_w)
    l_coords = torch.zeros((g_h, g_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0)

    return ref_tokens, ref_ids


def stitch_side_by_side_comparison(
    baseline_img: Image.Image,
    tendoo_img: Image.Image,
    text_content: str,
    material_desc: str,
    font_name: str,
    output_path: str,
):
    """
    Creates an ultra high-quality executive side-by-side presentation panel.
    """
    w, h = baseline_img.size
    header_h = 130
    footer_h = 60
    total_w = w * 2
    total_h = h + header_h + footer_h

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 17, 23))
    draw = ImageDraw.Draw(canvas)

    # Load Fonts for UI
    try:
        font_main_title = ImageFont.truetype(resolve_font_path("bevietnam"), size=26)
        font_sub_title = ImageFont.truetype(resolve_font_path("bevietnam"), size=17)
        font_card_title = ImageFont.truetype(resolve_font_path("bevietnam"), size=21)
        font_card_note = ImageFont.truetype(resolve_font_path("bevietnam"), size=15)
        font_footer = ImageFont.truetype(resolve_font_path("bevietnam"), size=14)
    except Exception:
        font_main_title = ImageFont.load_default()
        font_sub_title = ImageFont.load_default()
        font_card_title = ImageFont.load_default()
        font_card_note = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # 1. Main Header Background
    draw.rectangle([0, 0, total_w, header_h], fill=(22, 26, 36), outline=(40, 46, 62), width=2)

    # Main Title
    main_title = "ĐỐI CHỨNG NĂNG LỰC RENDER CHỮ TIẾNG VIỆT HÒA QUYỆN VÀO CHẤT LIỆU"
    bbox_mt = font_main_title.getbbox(main_title)
    draw.text(
        ((total_w - (bbox_mt[2] - bbox_mt[0])) // 2, 16),
        main_title,
        fill=(255, 215, 80),
        font=font_main_title,
    )

    # Subtitle Context
    sub_title = f'Văn bản: "{text_content}"  |  Hiệu ứng: {material_desc}'
    bbox_st = font_sub_title.getbbox(sub_title)
    draw.text(
        ((total_w - (bbox_st[2] - bbox_st[0])) // 2, 52),
        sub_title,
        fill=(200, 210, 230),
        font=font_sub_title,
    )

    # Column 1 Header (Baseline)
    draw.rectangle([0, 80, w, header_h], fill=(38, 22, 22), outline=(75, 35, 35), width=1)
    col1_title = "❌ BASELINE: Gửi thẳng Prompt vào FLUX.2 Base 4B"
    bbox_c1 = font_card_title.getbbox(col1_title)
    draw.text(((w - (bbox_c1[2] - bbox_c1[0])) // 2, 85), col1_title, fill=(255, 110, 110), font=font_card_title)
    col1_sub = "(Text Encoder Qwen3-4B tự sinh chữ -> Mất dấu, sai chính tả, méo mó nét chữ)"
    bbox_c1s = font_card_note.getbbox(col1_sub)
    draw.text(((w - (bbox_c1s[2] - bbox_c1s[0])) // 2, 108), col1_sub, fill=(220, 150, 150), font=font_card_note)

    # Column 2 Header (Tendoo AI)
    draw.rectangle([w, 80, total_w, header_h], fill=(18, 38, 26), outline=(35, 75, 45), width=1)
    col2_title = "✅ TENDOO AI: In-Context Glyph Conditioning (t=10.0)"
    bbox_c2 = font_card_title.getbbox(col2_title)
    draw.text((w + (w - (bbox_c2[2] - bbox_c2[0])) // 2, 85), col2_title, fill=(110, 255, 150), font=font_card_title)
    col2_sub = "(Glyph VAE định hình 100% chính tả & dấu + DiT hòa trộn chất liệu khắc chìm tự nhiên)"
    bbox_c2s = font_card_note.getbbox(col2_sub)
    draw.text((w + (w - (bbox_c2s[2] - bbox_c2s[0])) // 2, 108), col2_sub, fill=(150, 220, 170), font=font_card_note)

    # 2. Paste Images
    canvas.paste(baseline_img, (0, header_h))
    canvas.paste(tendoo_img, (w, header_h))

    # Divider line between columns
    draw.line([(w, header_h), (w, header_h + h)], fill=(60, 68, 88), width=3)

    # 3. Footer Bar
    footer_y = header_h + h
    draw.rectangle([0, footer_y, total_w, total_h], fill=(18, 20, 28), outline=(35, 40, 55), width=1)
    footer_text = (
        f"Mô hình: FLUX.2-klein-base-4B | Text Encoder: Qwen3-4B-FP8 | VAE: 128-ch | "
        f"Font: {font_name} | Steps: 50 | CFG: 4.5 | Platform: 2x NVIDIA A30 (48GB VRAM)"
    )
    bbox_ft = font_footer.getbbox(footer_text)
    draw.text(
        ((total_w - (bbox_ft[2] - bbox_ft[0])) // 2, footer_y + (footer_h - (bbox_ft[3] - bbox_ft[1])) // 2),
        footer_text,
        fill=(130, 140, 165),
        font=font_footer,
    )

    canvas.save(output_path, quality=95)
    print(f"\n📊 [Executive Presentation Panel Saved] -> {output_path} ({total_w}x{total_h})")


def run_problem_statement_experiment(
    text: str,
    preset: str,
    custom_baseline_prompt: str | None = None,
    custom_tendoo_prompt: str | None = None,
    font_name: str | None = None,
    width: int = 1024,
    height: int = 1024,
    box_w: int | None = None,
    box_h: int | None = None,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    output_dir: str = "output_problem_statement",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 TENDOO AI - PROBLEM STATEMENT EXPERIMENT & SIDE-BY-SIDE BENCHMARK")
    print(f"📌 Vietnamese Text : '{text}'")
    print(f"🎨 Material Preset : {preset}")
    print(f"📐 Canvas Size     : {width}x{height}")
    print(f"⚙️  ODE Steps / CFG : {num_steps} steps | CFG {guidance} | Seed: {seed}")
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
        raise RuntimeError("❌ CUDA is required to run this experiment on GPU!")

    # 2. Setup Prompts and Material
    preset_data = MATERIAL_PRESETS.get(preset, MATERIAL_PRESETS["sandstone_carved"])
    material_desc = preset_data["name"]

    baseline_prompt = custom_baseline_prompt or preset_data["baseline_prompt_template"].format(text=text)
    tendoo_prompt = custom_tendoo_prompt or preset_data["tendoo_prompt"]
    selected_font = font_name or preset_data["default_font"]
    font_path = resolve_font_path(selected_font)

    print(f"\n[Configuration Details]")
    print(f"• Baseline Prompt (Direct) : {baseline_prompt}")
    print(f"• Tendoo Clean Prompt      : {tendoo_prompt}")
    print(f"• Font                     : {selected_font} ({Path(font_path).name})")

    # 3. Compute Dynamic Glyph Box
    lines = [l.strip() for l in text.replace("\\n", "\n").split("\n") if l.strip()]
    if len(lines) == 1 and len(text.split()) >= 4:
        words = text.split()
        if len(words) >= 6:
            p1 = len(words) // 3
            p2 = 2 * len(words) // 3
            num_lines = 3
        else:
            num_lines = 2
    else:
        num_lines = max(1, len(lines))

    calc_w = box_w or min(896, int(width * 0.88))
    calc_h = box_h or max(384, min(640, num_lines * 160))
    calc_w = (calc_w // 16) * 16
    calc_h = (calc_h // 16) * 16

    print(f"\n[1/4] Rendering In-Context Glyph Bitmap ({calc_w}x{calc_h}, {num_lines} lines)...")
    glyph_img = create_glyph_image(
        text=text,
        target_width=calc_w,
        target_height=calc_h,
        font_path=font_path,
        padding_ratio=0.08,
    )
    glyph_preview_path = out_path / "glyph_preview.png"
    glyph_img.save(glyph_preview_path)
    print(f"  -> Glyph preview saved: {glyph_preview_path}")

    # 4. Load Models
    print("\n[2/4] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    # 5. Shared Initial Noise (for 1:1 identical initial state)
    lat_h = height // 16
    lat_w = width // 16
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    # =========================================================================
    # PASS A: BASELINE (DIRECT PROMPT TO FLUX.2 BASE 4B)
    # =========================================================================
    print("\n" + "-" * 70)
    print("⚡ [PASS A/2] Running Baseline: Direct Prompting (No In-Context Glyph)...")
    print("-" * 70)
    start_t_base = time.time()

    with torch.no_grad():
        txt_base = text_encoder(["", baseline_prompt])
        txt_base, txt_ids_base = batched_prc_txt(txt_base)
        txt_base = txt_base.to(device_dit)
        txt_ids_base = txt_ids_base.to(device_dit)

        # Baseline: img_cond_seq is None
        out_latent_base = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt_base,
            txt_ids=txt_ids_base,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=None,
            img_cond_seq_ids=None,
        )

        out_latent_base = rearrange(out_latent_base, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_pixels_base = ae.decode(out_latent_base.to(device_ae))
        out_pixels_base = ((out_pixels_base[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        baseline_img = Image.fromarray(out_pixels_base)

        baseline_file = out_path / "baseline_direct_prompt.png"
        baseline_img.save(baseline_file)
        print(f"  -> Baseline image saved in {time.time() - start_t_base:.2f}s: {baseline_file}")

    # =========================================================================
    # PASS B: TENDOO AI (IN-CONTEXT GLYPH CONDITIONING AT t=10.0)
    # =========================================================================
    print("\n" + "-" * 70)
    print("⚡ [PASS B/2] Running Tendoo AI: In-Context Glyph Conditioning (t=10.0)...")
    print("-" * 70)
    start_t_tendoo = time.time()

    with torch.no_grad():
        txt_tendoo = text_encoder(["", tendoo_prompt])
        txt_tendoo, txt_ids_tendoo = batched_prc_txt(txt_tendoo)
        txt_tendoo = txt_tendoo.to(device_dit)
        txt_ids_tendoo = txt_ids_tendoo.to(device_dit)

        glyph_tokens, glyph_ids = encode_glyph_to_incontext_tokens(
            ae=ae, glyph_img=glyph_img, t_offset=10.0, device=device_ae
        )
        glyph_tokens = glyph_tokens.to(device_dit)
        glyph_ids = glyph_ids.to(device_dit)

        out_latent_tendoo = denoise_cfg(
            model=model,
            img=img_tokens.clone(),
            img_ids=img_ids.clone(),
            txt=txt_tendoo,
            txt_ids=txt_ids_tendoo,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=glyph_tokens,
            img_cond_seq_ids=glyph_ids,
        )

        out_latent_tendoo = rearrange(out_latent_tendoo, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_pixels_tendoo = ae.decode(out_latent_tendoo.to(device_ae))
        out_pixels_tendoo = ((out_pixels_tendoo[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        tendoo_img = Image.fromarray(out_pixels_tendoo)

        tendoo_file = out_path / "tendoo_incontext_glyph.png"
        tendoo_img.save(tendoo_file)
        print(f"  -> Tendoo AI image saved in {time.time() - start_t_tendoo:.2f}s: {tendoo_file}")

    # =========================================================================
    # STITCH SIDE-BY-SIDE PRESENTATION PANEL
    # =========================================================================
    print("\n[4/4] Stitching Executive Side-by-Side Presentation Panel...")
    comparison_file = out_path / "PROBLEM_STATEMENT_COMPARISON.png"
    stitch_side_by_side_comparison(
        baseline_img=baseline_img,
        tendoo_img=tendoo_img,
        text_content=text,
        material_desc=material_desc,
        font_name=Path(font_path).stem,
        output_path=str(comparison_file),
    )

    print("\n" + "=" * 80)
    print(f"🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"📸 1. Baseline Image         : {baseline_file}")
    print(f"📸 2. Tendoo AI Image        : {tendoo_file}")
    print(f"🖼️  3. Glyph Preview Bitmap  : {glyph_preview_path}")
    print(f"📊 4. EXECUTIVE SLIDE PANEL : {comparison_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tendoo AI - Problem Statement Side-by-Side Comparison Script (Baseline vs In-Context Glyph)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Cà phê sữa đá đậm đà hương vị truyền thống",
        help="Medium-length Vietnamese text string (e.g. 'Cà phê sữa đá đậm đà hương vị truyền thống')",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(MATERIAL_PRESETS.keys()),
        default="sandstone_carved",
        help="Material preset: sandstone_carved, wood_engraved, metal_embossed, leather_stamped",
    )
    parser.add_argument(
        "--custom_baseline_prompt",
        type=str,
        default=None,
        help="Optional custom prompt for Baseline (must contain literal text)",
    )
    parser.add_argument(
        "--custom_tendoo_prompt",
        type=str,
        default=None,
        help="Optional custom prompt for Tendoo (clean prompt without literal text)",
    )
    parser.add_argument(
        "--font",
        type=str,
        default="playfair",
        help="Font alias (playfair, bevietnam, anton, pacifico, graffiti, dancing, oswald) or path",
    )
    parser.add_argument("--width", type=int, default=1024, help="Canvas width in pixels (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height in pixels (default: 1024)")
    parser.add_argument("--box_w", type=int, default=896, help="Glyph box width in pixels (default: 896)")
    parser.add_argument("--box_h", type=int, default=448, help="Glyph box height in pixels (default: 448)")
    parser.add_argument("--steps", type=int, default=50, help="ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 1:1 fair comparison (default: 42)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_problem_statement",
        help="Output directory for generated images and comparison panel",
    )
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")

    args = parser.parse_args()

    run_problem_statement_experiment(
        text=args.text,
        preset=args.preset,
        custom_baseline_prompt=args.custom_baseline_prompt,
        custom_tendoo_prompt=args.custom_tendoo_prompt,
        font_name=args.font,
        width=args.width,
        height=args.height,
        box_w=args.box_w,
        box_h=args.box_h,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
