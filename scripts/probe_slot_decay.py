"""
================================================================================
TENDOO AI - MULTI-SLOT TIME OFFSET SIGNAL PROBING BENCHMARK
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Objective:
- Measure zero-shot Attention signal retention across time offsets: t = 10, 20, 30, 40, 50.
- Determine the Pareto boundary for Phase 3 LoRA Fine-Tuning (2-Slot vs 3-Slot vs 4-Slot).
- Models are loaded ONCE in memory for ultra-fast sequential benchmark passes.
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
    t_offset: float = 10.0,
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
        font = ImageFont.truetype(resolve_font_path("bevietnam"), size=26)
    except Exception:
        font = ImageFont.load_default()

    for idx, (img, title) in enumerate(zip(images, titles)):
        x_offset = idx * w
        # Paste Image
        canvas.paste(img, (x_offset, header_h))

        # Draw Header Banner
        draw.rectangle([x_offset, 0, x_offset + w, header_h], fill=(35, 35, 45), outline=(60, 60, 75))
        bbox = font.getbbox(title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x_offset + (w - tw) // 2
        ty = (header_h - th) // 2 - bbox[1]
        draw.text((tx, ty), title, fill=(255, 220, 100), font=font)

    canvas.save(output_path)
    print(f"\n📊 [Stitched Comparison Saved] -> {output_path} ({total_w}x{total_h})")


class SlotProber:
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

    def generate_single_pass(
        self,
        prompt: str,
        text_blocks: list[dict],  # list of {"text": str, "t_offset": float, "font": str, "w": int, "h": int}
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 50,
        guidance: float = 4.0,
        seed: int = 42,
    ) -> Image.Image:
        """Executes a single denoise pass for given text blocks and prompt."""
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

        # 3. Encode In-Context Text Glyphs
        ref_token_list = []
        ref_id_list = []

        for tb in text_blocks:
            txt_content = tb["text"]
            t_val = tb["t_offset"]
            f_path = resolve_font_path(tb.get("font", "bevietnam"))
            bw = (tb.get("w", 512) // 16) * 16
            bh = (tb.get("h", 192) // 16) * 16

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

        all_ref_tokens = torch.cat(ref_token_list, dim=1).to(self.device_dit)
        all_ref_ids = torch.cat(ref_id_list, dim=1).to(self.device_dit)

        # 4. Denoise Euler ODE
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


def run_isolated_slot_decay_benchmark(prober: SlotProber, output_dir: Path):
    """
    Test Suite 1: Probes the decay of signal for the EXACT same text placed at t = 10, 20, 30, 40, 50.
    """
    print("\n" + "=" * 80)
    print(" 🧪 TEST SUITE 1: ISOLATED SINGLE-SLOT SIGNAL DECAY BENCHMARK")
    print("=" * 80)

    test_text = "MUA 1 TẶNG 1"
    prompt = (
        "Ảnh chụp banner quảng cáo hiện đại trên nền màu nâu cafe sang trọng, "
        "dòng chữ nổi 3D mạ vàng gold lấp lánh ở trung tâm phản chiếu ánh sáng studio, "
        "ánh sáng điện ảnh cao cấp, độ chi tiết cao"
    )
    t_offsets = [10.0, 20.0, 30.0, 40.0, 50.0]
    generated_images = []
    titles = []

    for t_val in t_offsets:
        print(f"\n▶ Running Probe for isolated slot: t = {t_val}...")
        t_start = time.time()
        blocks = [{"text": test_text, "t_offset": t_val, "font": "anton", "w": 576, "h": 224}]
        img = prober.generate_single_pass(
            prompt=prompt,
            text_blocks=blocks,
            width=1024,
            height=1024,
            seed=100,
        )
        elapsed = time.time() - t_start
        out_file = output_dir / f"probe_isolated_t{int(t_val)}.png"
        img.save(out_file)
        print(f"  -> ✅ Saved: {out_file.name} in {elapsed:.2f}s")
        generated_images.append(img)
        titles.append(f"Slot t = {t_val:.1f}")

    stitch_horizontal_comparison(
        images=generated_images,
        titles=titles,
        output_path=str(output_dir / "PROBE_SUITE_1_ISOLATED_DECAY_CURVE.png"),
    )


def run_concurrent_multi_slot_benchmark(prober: SlotProber, output_dir: Path):
    """
    Test Suite 2: Probes 3-Slot vs 4-Slot Concurrent Text Competition.
    """
    print("\n" + "=" * 80)
    print(" 🧪 TEST SUITE 2: CONCURRENT MULTI-SLOT CAPACITY BENCHMARK")
    print("=" * 80)

    prompt = (
        "Thiết kế banner quảng cáo khai trương quán cafe sang trọng hiện đại tông nâu ấm, "
        "biển hiệu lớn trên cao với dòng chữ vàng gold 3D dập nổi tinh xảo, "
        "thông tin mô tả mạ đồng cổ điển ở giữa, góc dưới có các huy hiệu chữ phát sáng nổi bật, "
        "ánh sáng studio điện ảnh, chất lượng cao"
    )

    # 1. Configuration A: 3-Slot Architecture (t=10, t=20, t=30)
    print("\n▶ Running Config A: 3-Slot Architecture (t=10, t=20, t=30)...")
    blocks_3slots = [
        {"text": "GRAND OPENING", "t_offset": 10.0, "font": "anton", "w": 640, "h": 192},
        {"text": "CÀ PHÊ RANG MỘC", "t_offset": 20.0, "font": "bevietnam", "w": 576, "h": 160},
        {"text": "MUA 1 TẶNG 1", "t_offset": 30.0, "font": "pacifico", "w": 512, "h": 160},
    ]
    t0 = time.time()
    img_3s = prober.generate_single_pass(
        prompt=prompt,
        text_blocks=blocks_3slots,
        width=1024,
        height=1024,
        seed=200,
    )
    out_3s = output_dir / "probe_concurrent_3slots_t10_20_30.png"
    img_3s.save(out_3s)
    print(f"  -> ✅ Saved 3-Slot Result: {out_3s.name} in {time.time() - t0:.2f}s")

    # 2. Configuration B: 4-Slot Architecture (t=10, t=20, t=30, t=40)
    print("\n▶ Running Config B: 4-Slot Architecture (t=10, t=20, t=30, t=40)...")
    blocks_4slots = [
        {"text": "GRAND OPENING", "t_offset": 10.0, "font": "anton", "w": 640, "h": 192},
        {"text": "CÀ PHÊ RANG MỘC", "t_offset": 20.0, "font": "bevietnam", "w": 576, "h": 160},
        {"text": "MUA 1 TẶNG 1", "t_offset": 30.0, "font": "pacifico", "w": 512, "h": 160},
        {"text": "GHÉ NGAY HÔM NAY", "t_offset": 40.0, "font": "sedgwick", "w": 512, "h": 160},
    ]
    t0 = time.time()
    img_4s = prober.generate_single_pass(
        prompt=prompt,
        text_blocks=blocks_4slots,
        width=1024,
        height=1024,
        seed=200,
    )
    out_4s = output_dir / "probe_concurrent_4slots_t10_20_30_40.png"
    img_4s.save(out_4s)
    print(f"  -> ✅ Saved 4-Slot Result: {out_4s.name} in {time.time() - t0:.2f}s")

    stitch_horizontal_comparison(
        images=[img_3s, img_4s],
        titles=["3-Slot (t=10, 20, 30)", "4-Slot (t=10, 20, 30, 40)"],
        output_path=str(output_dir / "PROBE_SUITE_2_CONCURRENT_CAPACITY_COMPARISON.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tendoo AI - Time Offset Signal Probing Benchmark")
    parser.add_argument("--suite", type=str, default="all", choices=["all", "isolated", "concurrent"], help="Benchmark suite to run")
    parser.add_argument("--output_dir", type=str, default="probe_results", help="Directory to save benchmark results")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="Model name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to checkpoint directory")

    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print(" 🔬 TENDOO AI: MULTI-SLOT TIME OFFSET SIGNAL PROBING BENCHMARK")
    print(f" 📂 Output Directory: {out_path.resolve()}")
    print("=" * 80)

    # Initialize prober (Loads weights once)
    prober = SlotProber(model_name=args.model_name, checkpoint_dir=args.checkpoint_dir)

    t_benchmark_start = time.time()

    if args.suite in ["all", "isolated"]:
        run_isolated_slot_decay_benchmark(prober, out_path)

    if args.suite in ["all", "concurrent"]:
        run_concurrent_multi_slot_benchmark(prober, out_path)

    total_time = time.time() - t_benchmark_start
    print("\n" + "=" * 80)
    print(f"🎉 BENCHMARK COMPLETED SUCCESSFULLY IN {total_time:.2f}s (~{total_time/60:.1f} minutes)!")
    print(f"📁 Check your comparison grids in: {out_path.resolve()}")
    print("=" * 80 + "\n")
