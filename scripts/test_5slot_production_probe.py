"""
================================================================================
TENDOO AI - 5-SLOT PRODUCTION STRESS TEST & ATTENTION RETENTION BENCHMARK
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Objective:
- Stress-test the full concurrent 5-slot production setup:
    [4 Text Slots (t=10, 20, 30, 40) + 1 Real Product Image Slot (t=50)]
- Test the comparative inverted architecture:
    [Product at t=10 + 3-4 Text Slots at t=20, 30, 40]
- Determine token mass interaction and cross-slot attention limits under heavy real-world load.
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
    """Renders tight-crop Vietnamese glyph bitmap with automatic line wrapping and binary-search sizing."""
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

            line_spacing = int(mid_size * 0.18) * (len(lines) - 1)
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

    line_spacing = int(best_size * 0.18)
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
    """Encodes glyph image to 4D RoPE coordinate tokens."""
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


def encode_product_to_incontext_tokens(
    ae: AutoEncoder,
    image_path: str,
    t_offset: float,
    target_size: int = 1024,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes a natural product image into 4D RoPE tokens."""
    prod_img = Image.open(image_path).convert("RGB")
    prod_img = prod_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    p_arr = np.array(prod_img).astype(np.float32) / 127.5 - 1.0
    p_tensor = torch.from_numpy(p_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        p_latent = ae.encode(p_tensor)

    ref_tokens, _ = prc_img(p_latent[0])
    ref_tokens = ref_tokens.unsqueeze(0)

    p_h, p_w = p_latent.shape[2], p_latent.shape[3]
    t_coords = torch.full((p_h, p_w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(p_h, dtype=torch.float32, device=device).unsqueeze(1).expand(p_h, p_w)
    w_coords = torch.arange(p_w, dtype=torch.float32, device=device).unsqueeze(0).expand(p_h, p_w)
    l_coords = torch.zeros((p_h, p_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0)

    return ref_tokens, ref_ids


def stitch_horizontal_comparison(images: list[Image.Image], titles: list[str], output_path: str):
    """Stitches multiple images into a labeled horizontal side-by-side comparison panel."""
    if not images:
        return

    w, h = images[0].size
    header_h = 60
    total_w = w * len(images)
    total_h = h + header_h

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 25))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(resolve_font_path("bevietnam"), size=24)
    except Exception:
        font = ImageFont.load_default()

    for idx, (img, title) in enumerate(zip(images, titles)):
        x_offset = idx * w
        canvas.paste(img, (x_offset, header_h))
        draw.rectangle([x_offset, 0, x_offset + w, header_h], fill=(35, 35, 45), outline=(60, 60, 75))
        bbox = font.getbbox(title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x_offset + (w - tw) // 2
        ty = (header_h - th) // 2 - bbox[1]
        draw.text((tx, ty), title, fill=(255, 220, 100), font=font)

    canvas.save(output_path)
    print(f"\n📊 [Stitched Comparison Saved] -> {output_path} ({total_w}x{total_h})")


class ProductionStressProber:
    """Manages loaded models in VRAM for fast benchmark execution."""

    def __init__(self, model_name: str = "flux.2-klein-base-4b", checkpoint_dir: str | None = None):
        if checkpoint_dir:
            os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            self.device_dit = torch.device("cuda:0")
            self.device_ae = torch.device("cuda:1")
            self.device_te = torch.device("cuda:1")
            print("🚀 Multi-GPU Mode: DiT on GPU 0, VAE & Qwen3 on GPU 1")
        else:
            self.device_dit = torch.device("cuda:0")
            self.device_ae = torch.device("cuda:0")
            self.device_te = torch.device("cuda:0")
            print(f"🚀 Single-GPU Mode: {self.device_dit}")

        print("⏳ Loading FLUX.2 Klein 4B Base Models once into VRAM...")
        t0 = time.time()
        self.model = load_flow_model(model_name, device=self.device_dit)
        self.ae = load_ae(model_name, device=self.device_ae)
        self.text_encoder = load_qwen3_embedder(variant="4B", device=self.device_te)
        print(f"✅ Models loaded in {time.time() - t0:.2f}s!")

    def generate_stress_pass(
        self,
        prompt: str,
        text_slots: list[dict],  # [{"text": str, "t": float, "font": str, "w": int, "h": int}]
        product_slots: list[dict],  # [{"image_path": str, "t": float}]
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 50,
        guidance: float = 4.0,
        seed: int = 42,
    ) -> Image.Image:
        """Executes a single denoise pass with combined text and product image slots."""
        width = (width // 16) * 16
        height = (height // 16) * 16
        lat_h = height // 16
        lat_w = width // 16

        torch.manual_seed(seed)

        # 1. Encode Text Prompt
        with torch.no_grad():
            txt = self.text_encoder(["", prompt])
            txt, txt_ids = batched_prc_txt(txt)
            txt = txt.to(self.device_dit)
            txt_ids = txt_ids.to(self.device_dit)

        # 2. Encode Canvas Init
        z_init = torch.randn(1, 128, lat_h, lat_w, device=self.device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(self.device_dit)
        img_ids = img_ids.unsqueeze(0).to(self.device_dit)

        ref_token_list = []
        ref_id_list = []

        # 3. Encode Product Images
        for ps in product_slots:
            img_p = ps["image_path"]
            t_val = ps["t"]
            if os.path.exists(img_p):
                p_tokens, p_ids = encode_product_to_incontext_tokens(
                    ae=self.ae,
                    image_path=img_p,
                    t_offset=t_val,
                    device=self.device_ae,
                )
                ref_token_list.append(p_tokens)
                ref_id_list.append(p_ids)
                print(f"    📦 [Product Slot] t={t_val}: '{Path(img_p).name}' ({p_tokens.shape[1]} tokens)")

        # 4. Encode Text Glyphs
        for ts in text_slots:
            txt_content = ts["text"]
            t_val = ts["t"]
            f_path = resolve_font_path(ts.get("font", "bevietnam"))
            bw = (ts.get("w", 512) // 16) * 16
            bh = (ts.get("h", 192) // 16) * 16

            glyph_img = create_glyph_image(
                text=txt_content,
                target_width=bw,
                target_height=bh,
                font_path=f_path,
            )
            r_tokens, r_ids = encode_glyph_to_incontext_tokens(
                ae=self.ae,
                glyph_img=glyph_img,
                t_offset=t_val,
                device=self.device_ae,
            )
            ref_token_list.append(r_tokens)
            ref_id_list.append(r_ids)
            print(f"    🔤 [Text Slot]    t={t_val}: '{txt_content}' ({r_tokens.shape[1]} tokens)")

        all_ref_tokens = torch.cat(ref_token_list, dim=1).to(self.device_dit)
        all_ref_ids = torch.cat(ref_id_list, dim=1).to(self.device_dit)

        print(f"  -> Total Ref Tokens: {all_ref_tokens.shape[1]} (Across {len(product_slots) + len(text_slots)} concurrent slots)")

        # 5. Denoise Euler ODE
        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

        with torch.no_grad():
            out_latent = denoise_cfg(
                model=self.model,
                img=img_tokens,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=all_ref_tokens,
                img_cond_seq_ids=all_ref_ids,
            )

            out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
            out_pixels = self.ae.decode(out_latent.to(self.device_ae))
            out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(out_pixels)


def run_5slot_production_test(prober: ProductionStressProber, output_dir: Path, ref_img_path: str):
    """
    Executes the 3-Way Comparative Production Stress Test:
    1. Case 1: The Requested 5-Slot Setup (4 Texts at t=10, 20, 30, 40 + Product at t=50)
    2. Case 2: Inverted Setup (Product Anchor at t=10 + 3 Texts at t=20, 30, 40)
    3. Case 3: Balanced 4-Slot Setup (3 Texts at t=10, 20, 30 + Product at t=40)
    """
    print("\n" + "=" * 80)
    print(" 🚀 5-SLOT PRODUCTION STRESS BENCHMARK EXECUTION")
    print(f" 📦 Target Product Reference: {ref_img_path}")
    print("=" * 80)

    prompt = (
        "Ảnh chụp thương mại sản phẩm thật đặt ở vị trí trung tâm nổi bật, "
        "phía trên có biển hiệu lớn chữ vàng 3D dập nổi sang trọng, "
        "thông tin chi tiết chữ mạ đồng ở giữa, góc dưới có các huy hiệu chữ phát sáng rực rỡ, "
        "ánh sáng studio điện ảnh, độ chi tiết cao"
    )

    results = []
    titles = []

    # --------------------------------------------------------------------------
    # CASE 1: 4 Texts (t=10, 20, 30, 40) + 1 Product (t=50)
    # --------------------------------------------------------------------------
    print("\n▶ [TEST 1/3] CASE 1: 4 Texts (t=10, 20, 30, 40) + Product at t=50.0")
    t_start = time.time()
    text_slots_c1 = [
        {"text": "GRAND OPENING", "t": 10.0, "font": "anton", "w": 640, "h": 192},
        {"text": "CÀ PHÊ RANG MỘC", "t": 20.0, "font": "bevietnam", "w": 576, "h": 160},
        {"text": "MUA 1 TẶNG 1", "t": 30.0, "font": "pacifico", "w": 512, "h": 160},
        {"text": "GHÉ NGAY HÔM NAY", "t": 40.0, "font": "sedgwick", "w": 512, "h": 160},
    ]
    prod_slots_c1 = [{"image_path": ref_img_path, "t": 50.0}]

    img_c1 = prober.generate_stress_pass(
        prompt=prompt,
        text_slots=text_slots_c1,
        product_slots=prod_slots_c1,
        width=1024,
        height=1024,
        seed=300,
    )
    out_c1 = output_dir / "case1_4texts_t10_40_prod_t50.png"
    img_c1.save(out_c1)
    print(f"  -> ✅ Case 1 Saved: {out_c1.name} in {time.time() - t_start:.2f}s")
    results.append(img_c1)
    titles.append("Case 1: 4 Texts (t=10..40) + Prod (t=50)")

    # --------------------------------------------------------------------------
    # CASE 2: Product Anchor at t=10.0 + 3 Texts (t=20, 30, 40)
    # --------------------------------------------------------------------------
    print("\n▶ [TEST 2/3] CASE 2: Product at t=10.0 + 3 Texts (t=20, 30, 40)")
    t_start = time.time()
    text_slots_c2 = [
        {"text": "GRAND OPENING", "t": 20.0, "font": "anton", "w": 640, "h": 192},
        {"text": "CÀ PHÊ RANG MỘC", "t": 30.0, "font": "bevietnam", "w": 576, "h": 160},
        {"text": "MUA 1 TẶNG 1", "t": 40.0, "font": "pacifico", "w": 512, "h": 160},
    ]
    prod_slots_c2 = [{"image_path": ref_img_path, "t": 10.0}]

    img_c2 = prober.generate_stress_pass(
        prompt=prompt,
        text_slots=text_slots_c2,
        product_slots=prod_slots_c2,
        width=1024,
        height=1024,
        seed=300,
    )
    out_c2 = output_dir / "case2_prod_t10_3texts_t20_40.png"
    img_c2.save(out_c2)
    print(f"  -> ✅ Case 2 Saved: {out_c2.name} in {time.time() - t_start:.2f}s")
    results.append(img_c2)
    titles.append("Case 2: Prod (t=10) + 3 Texts (t=20..40)")

    # --------------------------------------------------------------------------
    # CASE 3: 3 Texts (t=10, 20, 30) + Product at t=40.0
    # --------------------------------------------------------------------------
    print("\n▶ [TEST 3/3] CASE 3: 3 Texts (t=10, 20, 30) + Product at t=40.0")
    t_start = time.time()
    text_slots_c3 = [
        {"text": "GRAND OPENING", "t": 10.0, "font": "anton", "w": 640, "h": 192},
        {"text": "CÀ PHÊ RANG MỘC", "t": 20.0, "font": "bevietnam", "w": 576, "h": 160},
        {"text": "MUA 1 TẶNG 1", "t": 30.0, "font": "pacifico", "w": 512, "h": 160},
    ]
    prod_slots_c3 = [{"image_path": ref_img_path, "t": 40.0}]

    img_c3 = prober.generate_stress_pass(
        prompt=prompt,
        text_slots=text_slots_c3,
        product_slots=prod_slots_c3,
        width=1024,
        height=1024,
        seed=300,
    )
    out_c3 = output_dir / "case3_3texts_t10_30_prod_t40.png"
    img_c3.save(out_c3)
    print(f"  -> ✅ Case 3 Saved: {out_c3.name} in {time.time() - t_start:.2f}s")
    results.append(img_c3)
    titles.append("Case 3: 3 Texts (t=10..30) + Prod (t=40)")

    # Stitched 3-Way Production Comparison
    stitch_horizontal_comparison(
        images=results,
        titles=titles,
        output_path=str(output_dir / "PROBE_5SLOT_PRODUCTION_3WAY_COMPARISON.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tendoo AI - 5-Slot Production Stress Test")
    parser.add_argument("--image_ref", type=str, default="images/ref_prod_02.png", help="Path to product reference image")
    parser.add_argument("--output_dir", type=str, default="probe_production_results", help="Directory to save benchmark results")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="Model name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to checkpoint directory")

    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    # Validate image ref
    ref_img = args.image_ref
    if not os.path.exists(ref_img):
        # Fallback to any image in images/
        for cand in ["images/reference_prod.png", "images/hao_hao.jpg", "images/shoes.jpeg"]:
            if os.path.exists(str(ROOT_DIR / cand)):
                ref_img = str(ROOT_DIR / cand)
                break

    print("=" * 80)
    print(" 🔬 TENDOO AI: 5-SLOT PRODUCTION CONCURRENT STRESS BENCHMARK")
    print(f" 📂 Output Directory: {out_path.resolve()}")
    print("=" * 80)

    prober = ProductionStressProber(model_name=args.model_name, checkpoint_dir=args.checkpoint_dir)
    t_start_all = time.time()
    run_5slot_production_test(prober, out_path, ref_img)
    total_elapsed = time.time() - t_start_all

    print("\n" + "=" * 80)
    print(f"🎉 5-SLOT STRESS BENCHMARK FINISHED IN {total_elapsed:.2f}s (~{total_elapsed/60:.1f} minutes)!")
    print(f"📁 Check comparison panel in: {out_path.resolve() / 'PROBE_5SLOT_PRODUCTION_3WAY_COMPARISON.png'}")
    print("=" * 80 + "\n")
