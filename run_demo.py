#!/usr/bin/env python3
"""
run_demo.py - TENDOO AI COMMERCIAL POSTER DEMO RUNNER
======================================================
Zero-port, zero-network-proxy standalone CLI runner for remote servers (2x NVIDIA A30)
and local testing.

USAGE ON JUPYTERLAB SERVER (2 dòng lệnh duy nhất):
  git pull origin main
  python run_demo.py --all

OR RUN SPECIFIC PRESETS:
  python run_demo.py --preset midautumn
  python run_demo.py --preset suv
  python run_demo.py --preset lookbook
  python run_demo.py --preset beverage
  python run_demo.py --interactive
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project paths
PROJECT_ROOT = Path(__file__).resolve().parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image
import torch

from tendoo.layouts import (
    PosterContent,
    analyze_color_harmony,
    get_layout,
    list_layouts,
)
from tendoo.typography_engine import PosterRenderer


# Catalog of 4 Canonical Commercial Campaigns
PRESETS: Dict[str, Dict[str, Any]] = {
    "beverage": {
        "id": "01_top_dome_beverage",
        "title": "🍵 Trà Đào Cam Sả (Vòm Đỉnh - Top Dome)",
        "layout": "top_dome",
        "pre_header": "BỘ SƯU TẬP MÙA HÈ",
        "headline": "TRÀ ĐÀO CAM SẢ\nTHANH MÁT TỰ NHIÊN",
        "slogan": "Thưởng thức trọn vẹn từng giọt tươi mát từ thiên nhiên",
        "offer_main": "MUA 2 TẶNG 1",
        "offer_sub": "Áp dụng tại mọi chi nhánh trên toàn quốc",
        "brand": "Tendoo Beverage",
        "hotline": "1900 8888",
        "prompt_scene": (
            "Commercial beverage advertisement photo of a tall glass of iced peach tea "
            "with fresh sliced oranges, lemongrass stalks, floating mint leaves, "
            "crystalline water splash, sunlight studio lighting, 8k crisp details, unbranded, zero text"
        ),
        "style_hint": "daylight",
    },
    "midautumn": {
        "id": "02_center_hourglass_midautumn",
        "title": "🥮 Bánh Trung Thu Hoàng Gia (Đồng Hồ Cát - Center Hourglass)",
        "layout": "center_hourglass",
        "pre_header": "TẾT TRÔNG TRĂNG ĐOÀN VIÊN",
        "headline": "HỘP BÁNH TRUNG THU\nHOÀNG GIA THƯỢNG HẠNG",
        "slogan": "Món quà trọn vẹn ân tình gửi gắm gia đình",
        "offer_main": "CHIẾT KHẤU ĐẾN 20%",
        "offer_sub": "Tặng kèm trà sen Tây Hồ hảo hạng",
        "brand": "Tendoo Mooncake",
        "hotline": "0988 123 456",
        "prompt_scene": (
            "Traditional Mid-Autumn festival scene with traditional wooden street stalls, "
            "glowing lanterns, golden full moon light beam shining down, wooden floorboards, "
            "festive atmosphere, unbranded, zero text"
        ),
        "style_hint": "festive_light",
    },
    "suv": {
        "id": "03_bottom_platform_suv",
        "title": "🏎️ Khám Phá SUV Thế Hệ Mới (Bệ Đáy Điện Ảnh - Bottom Platform)",
        "layout": "bottom_platform",
        "pre_header": "PHIÊN BẢN GIỚI HẠN 2026",
        "headline": "KHÁM PHÁ ĐẲNG CẤP\nSUV THẾ HỆ MỚI",
        "slogan": "Chinh phục mọi cung đường hiểm trở với công nghệ truyền động thông minh",
        "offer_main": "ƯU ĐÃI 100 TRIỆU",
        "offer_sub": "Tặng gói bảo hiểm thân vỏ và 3 năm bảo dưỡng miễn phí",
        "brand": "Tendoo Motors",
        "hotline": "1900 8888",
        "prompt_scene": (
            "Commercial automobile advertising photography of a sleek luxury metallic dark grey SUV "
            "driving on a winding mountain road at golden hour twilight, motion blur wheels, "
            "sharp car body reflection, dramatic sky, unbranded, zero text"
        ),
        "style_hint": "cinematic_asphalt",
    },
    "lookbook": {
        "id": "04_split_column_lookbook",
        "title": "👗 Lookbook Thời Trang Áo Măng Tô (Dải Lụa Phân Cột - Split Column)",
        "layout": "split_column",
        "pre_header": "BỘ SƯU TẬP THU ĐÔNG 2026",
        "headline": "ÁO KHOÁC MĂNG TÔ\nDẠ LÔNG CỪU Ý",
        "slogan": "Chất liệu dạ lông cừu thượng hạng dệt tay, tôn vinh nét thanh lịch vượt thời gian",
        "offer_main": "ƯU ĐÃI ĐẾN 25%",
        "offer_sub": "Tặng khăn choàng lụa tơ tằm cao cấp cho hóa đơn từ 3 triệu",
        "brand": "Tendoo Atelier",
        "hotline": "1900 6868",
        "prompt_scene": (
            "Editorial high fashion photography of an elegant Asian female model wearing a tailored luxury "
            "wool coat, studio portrait, soft warm rim lighting, dramatic pose, high-end lookbook magazine, "
            "unbranded, zero text"
        ),
        "style_hint": "silk_sash",
    },
}


def load_models_on_gpus(is_mock: bool = False):
    """Loads DiT on cuda:0, VAE & Qwen3 on cuda:1 for 2x A30 setup."""
    if is_mock or not torch.cuda.is_available():
        print("[INFO] Chay o che do MOCK / CPU (khong nap weights nang).")
        return None, None, None, None, "cpu", "cpu"

    print("\n" + "=" * 75)
    print("🚀 DANG NAP FLUX.2 KLEIN 4B VAO 2x NVIDIA A30...")
    print("=" * 75)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        dev_dit = "cuda:0"
        dev_aux = "cuda:1"
        print(f"  [Phan bo Multi-GPU]: DiT tren {dev_dit} | Qwen3 & VAE tren {dev_aux}")
    else:
        dev_dit = dev_aux = "cuda:0"
        print(f"  [Single-GPU]: Tat ca mo hinh tren {dev_dit}")

    from flux2 import util
    model_name = "flux.2-klein-4b"

    pdata_candidates = [
        Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-4B")),
        Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B")),
    ]
    for pdata in pdata_candidates:
        cand = pdata / "flux-2-klein-4b.safetensors"
        if cand.exists() and "KLEIN_4B_MODEL_PATH" not in os.environ:
            os.environ["KLEIN_4B_MODEL_PATH"] = str(cand)
            break

    t0 = time.time()
    print("⏳ Nap DiT 4B Flow Model vao GPU 0...")
    dit_model = util.load_flow_model(model_name, device=dev_dit)
    dit_model.eval()

    print("⏳ Nap AutoEncoder (VAE 16x) vao GPU 1...")
    ae_model = util.load_ae(model_name, device=dev_aux)
    ae_model.eval()
    ae_dtype = next(ae_model.parameters()).dtype

    print("⏳ Nap Qwen3 Text Encoder vao GPU 1...")
    text_encoder = util.load_text_encoder(model_name, device=dev_aux)

    dur = time.time() - t0
    print(f"\n✅ TAT CA MO HINH DA SAN SANG TRONG VRAM ({dur:.1f}s)!")
    for dev_id in range(num_gpus):
        alloc = torch.cuda.memory_allocated(dev_id) / (1024 ** 2)
        res = torch.cuda.memory_reserved(dev_id) / (1024 ** 2)
        print(f"  [GPU {dev_id}] Occupied: {alloc:.1f} MB (Reserved: {res:.1f} MB)")
    print("=" * 75 + "\n")

    return dit_model, ae_model, text_encoder, ae_dtype, dev_dit, dev_aux


def free_gpu_memory(dit_model, ae_model, text_encoder):
    """Cleanly flushes GPU memory across all devices."""
    print("\n🛑 Dang giai phong toan bo VRAM GPU...")
    del dit_model
    del ae_model
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        for dev_id in range(torch.cuda.device_count()):
            with torch.cuda.device(dev_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            alloc = torch.cuda.memory_allocated(dev_id) / (1024 ** 2)
            print(f"  [GPU {dev_id}] Da giai phong sach (Con lai: {alloc:.1f} MB)")
    print("✨ Da giai phong 100% VRAM!\n")


def generate_single_poster(
    preset_data: Dict[str, Any],
    dit_model: Any,
    ae_model: Any,
    text_encoder: Any,
    ae_dtype: Any,
    dev_dit: str,
    dev_aux: str,
    out_dir: Path,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    guidance: float = 1.5,
    seed: int = 42,
) -> Path:
    """Executes single poster generation pipeline."""
    title = preset_data.get("title", "Commercial Poster")
    layout_name = preset_data["layout"]
    layout = get_layout(layout_name)

    print("-" * 75)
    print(f"🎨 BAT DAU TAO: {title}")
    print(f"   Layout: {layout.display_name} | Kich thuoc: {width}x{height}")
    t0 = time.time()

    # 1. Mask
    mask_np = layout.generate_mask(width=width, height=height)
    case_id = preset_data.get("id", f"poster_{int(time.time()*1000)}")
    mask_path = out_dir / f"{case_id}_mask.png"
    Image.fromarray((mask_np * 255).astype(np.uint8), mode="L").save(mask_path)

    # 2. DiT Inference
    if dit_model is not None and torch.cuda.is_available():
        from flux2.sampling import get_schedule, prc_img, prc_txt
        from pipeline_e2e_poster import denoise_regional_velocity_blended

        prompt_scene = preset_data["prompt_scene"]
        style_hint = preset_data.get("style_hint", "daylight")
        prompt_corr = layout.get_corridor_prompt(style_hint)

        print(f"   Inference DiT ({steps} steps, guidance={guidance})...")
        with torch.no_grad():
            ctx_scene = text_encoder([prompt_scene]).to(torch.bfloat16)
            ctx_scene, ctx_scene_ids = prc_txt(ctx_scene[0])
            ctx_scene = ctx_scene.unsqueeze(0).to(dev_dit)
            ctx_scene_ids = ctx_scene_ids.unsqueeze(0).to(dev_dit)

            ctx_corridor = text_encoder([prompt_corr]).to(torch.bfloat16)
            ctx_corridor, ctx_corridor_ids = prc_txt(ctx_corridor[0])
            ctx_corridor = ctx_corridor.unsqueeze(0).to(dev_dit)
            ctx_corridor_ids = ctx_corridor_ids.unsqueeze(0).to(dev_dit)

            mask_scaled = Image.fromarray(mask_np).resize((width // 16, height // 16), Image.Resampling.BICUBIC)
            mask_flat = torch.from_numpy(np.array(mask_scaled)).float().reshape(1, -1, 1).to(dev_dit)

            torch.manual_seed(seed)
            z_init = torch.randn(1, 128, height // 16, width // 16, device=dev_dit, dtype=torch.bfloat16)
            img_tokens, img_ids = prc_img(z_init[0])
            img_tokens = img_tokens.unsqueeze(0).to(dev_dit)
            img_ids = img_ids.unsqueeze(0).to(dev_dit)

            timesteps = get_schedule(num_steps=steps, image_seq_len=img_tokens.shape[1])

            h_lat, w_lat = height // 16, width // 16
            z_clean = denoise_regional_velocity_blended(
                model=dit_model,
                img=img_tokens,
                img_ids=img_ids,
                txt_scene=ctx_scene,
                txt_scene_ids=ctx_scene_ids,
                txt_corridor=ctx_corridor,
                txt_corridor_ids=ctx_corridor_ids,
                spatial_mask=mask_flat,
                timesteps=timesteps,
                guidance=guidance,
                num_canvas_tokens=img_tokens.shape[1],
            )

            z_clean_dec = z_clean[0].transpose(0, 1).reshape(1, 128, h_lat, w_lat).to(device=dev_aux, dtype=ae_dtype)
            x_dec = ae_model.decode(z_clean_dec).float()
            x_arr = ((x_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            blended_pil = Image.fromarray(x_arr)

    else:
        # Mock background for testing without GPU
        blended_pil = Image.new("RGB", (width, height), (16, 22, 34))

    bg_path = out_dir / f"{case_id}_background.png"
    blended_pil.save(bg_path)

    # 3. Color Extraction & Responsive Typography
    safe_zone = layout.get_safe_zone()
    palette = analyze_color_harmony(np.array(blended_pil), safe_zone, color_mode="auto")

    buf = io.BytesIO()
    blended_pil.save(buf, format="PNG")
    b64_bg = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    content = PosterContent(
        headline=preset_data.get("headline", ""),
        pre_header=preset_data.get("pre_header", ""),
        slogan=preset_data.get("slogan", ""),
        offer_main=preset_data.get("offer_main", ""),
        offer_sub=preset_data.get("offer_sub", ""),
        brand=preset_data.get("brand", ""),
        hotline=preset_data.get("hotline", ""),
    )

    html_str = layout.render_html(content=content, palette=palette, bg_data_uri=b64_bg, width=width, height=height)
    html_path = out_dir / f"{case_id}_poster.html"
    html_path.write_text(html_str, encoding="utf-8")

    out_poster_path = out_dir / f"{case_id}_final_poster.png"
    print("   Rendering HTML5 Vector Typography (Playwright)...")
    PosterRenderer.render(html_content=html_str, output_image_path=out_poster_path, width=width, height=height)

    dur = time.time() - t0
    print(f"✨ HOAN THANH TRONG {dur:.2f}s -> {out_poster_path}")
    return out_poster_path


def interactive_menu() -> List[str]:
    """Displays terminal interactive choice menu."""
    print("\n" + "=" * 75)
    print("🎨 TENDOO AI STUDIO - RUNNER DEMO POSTER (2x NVIDIA A30)")
    print("=" * 75)
    print("Chon che do ban muon chay:")
    print("  [1] Tao tat ca 4 Layouts (Khuyen nghi cho Sep xem tron bo)")
    print("  [2] Layout 1: Top Dome - 🍵 Tra Dao Cam Sa")
    print("  [3] Layout 2: Center Hourglass - 🥮 Banh Trung Thu Hoang Gia")
    print("  [4] Layout 3: Bottom Platform - 🏎️ SUV Kham Pha Dang Cap")
    print("  [5] Layout 4: Split Column - 👗 Lookbook Thoi Trang Ao Mang To")
    
    try:
        choice = input("\nNhap lua chon (1-5) [Mac dinh: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    mapping = {
        "1": ["beverage", "midautumn", "suv", "lookbook"],
        "2": ["beverage"],
        "3": ["midautumn"],
        "4": ["suv"],
        "5": ["lookbook"],
    }
    return mapping.get(choice, ["beverage", "midautumn", "suv", "lookbook"])


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Commercial Poster Demo Runner")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["all", "beverage", "midautumn", "suv", "lookbook"],
        help="Run specific preset or 'all'",
    )
    parser.add_argument("--all", action="store_true", help="Generate all 4 layouts")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive terminal prompt")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no GPU required)")
    parser.add_argument("--out-dir", type=str, default="output_demo", help="Output directory")
    parser.add_argument("--steps", type=int, default=8, help="DiT denoising steps")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which presets to run
    if args.all or args.preset == "all":
        selected_keys = ["beverage", "midautumn", "suv", "lookbook"]
    elif args.preset:
        selected_keys = [args.preset]
    elif args.interactive or sys.stdin.isatty():
        selected_keys = interactive_menu()
    else:
        # Default non-interactive: run all
        selected_keys = ["beverage", "midautumn", "suv", "lookbook"]

    # 1. Warm up models
    dit_model, ae_model, text_encoder, ae_dtype, dev_dit, dev_aux = load_models_on_gpus(is_mock=args.mock)

    t_total_start = time.time()
    generated_files = []

    try:
        for key in selected_keys:
            pdata = PRESETS[key]
            out_file = generate_single_poster(
                preset_data=pdata,
                dit_model=dit_model,
                ae_model=ae_model,
                text_encoder=text_encoder,
                ae_dtype=ae_dtype,
                dev_dit=dev_dit,
                dev_aux=dev_aux,
                out_dir=out_dir,
                steps=args.steps,
                guidance=args.guidance,
                seed=args.seed,
            )
            generated_files.append(out_file)

        total_dur = time.time() - t_total_start
        print("\n" + "=" * 75)
        print(f"🎉 HOAN THANH TOAN BO {len(generated_files)} POSTER TRONG {total_dur:.2f} GIAY!")
        print("=" * 75)
        print("👉 MOI SEP VA DONG NGHIEP MO CAY THU MUC BEN TRAI JUPYTERLAB,")
        print(f"   VAO THU MUC '{args.out_dir}/' VA DOUBLE-CLICK CAC ANH:")
        for idx, f in enumerate(generated_files, start=1):
            rel_path = f.relative_to(PROJECT_ROOT) if PROJECT_ROOT in f.parents else f
            print(f"   {idx}. {rel_path}")
        print("=" * 75 + "\n")

    finally:
        # 2. Always free GPU memory cleanly
        if dit_model is not None:
            free_gpu_memory(dit_model, ae_model, text_encoder)


if __name__ == "__main__":
    main()
