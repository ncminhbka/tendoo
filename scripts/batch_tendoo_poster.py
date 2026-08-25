"""
================================================================================
TENDOO AI - MULTI-GPU BATCH PARALLEL POSTER GENERATOR
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (24GB x 2 = 48GB VRAM)

Features:
- True Multi-Process Dual-GPU Asynchronous Worker Pool (GPU 0 & GPU 1 run concurrently)
- 2x Throughput: Generates 2 high-res posters simultaneously in ~30 seconds (~15s/image!)
- Built-in Executive Demo Presets (Cafe, Headphone Gold, Cyberpunk Neon)
- Supports custom JSON task lists or CLI batch tasks
================================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch.multiprocessing as mp

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
    "graffiti": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "sedgwick": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "dancing": str(ROOT_DIR / "fonts" / "DancingScript.ttf"),
    "oswald": str(ROOT_DIR / "fonts" / "Oswald.ttf"),
    # New Playful, Fresh, Summer SVN Fonts:
    "cookies": str(ROOT_DIR / "fonts" / "SVN-Cookies.ttf"),
    "grocery": str(ROOT_DIR / "fonts" / "SVN-Grocery Rounded.ttf"),
    "gretoon": str(ROOT_DIR / "fonts" / "SVN-Gretoon.ttf"),
    "blowbrush": str(ROOT_DIR / "fonts" / "SVN-Blow Brush.ttf"),
    "brush": str(ROOT_DIR / "fonts" / "SVN-Blow Brush.ttf"),
    "holidays": str(ROOT_DIR / "fonts" / "SVN-Holidays.ttf"),
    "clementine": str(ROOT_DIR / "fonts" / "SVN-Clementine.ttf"),
    "harabaras": str(ROOT_DIR / "fonts" / "SVN-Harabaras.ttf"),
    "lolapeluza": str(ROOT_DIR / "fonts" / "SVN-Lolapeluza Black.ttf"),
    "gotham": str(ROOT_DIR / "fonts" / "SVN-Gotham Ultra.otf"),
}



def resolve_font_path(font_name_or_path: str | None) -> str:
    """Resolves font alias (e.g. 'playfair', 'bevietnam') or validates file path."""
    if font_name_or_path:
        key = font_name_or_path.lower().strip()
        if key in FONT_REGISTRY and os.path.exists(FONT_REGISTRY[key]):
            return FONT_REGISTRY[key]
        if os.path.exists(font_name_or_path):
            return font_name_or_path

    # Fallback to defaults
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
    best_ascent_offset = 0

    for lines in candidate_layouts:
        low, high = 14, 180
        opt_font = None
        opt_size = 0
        opt_offset = 0

        while low <= high:
            mid_size = (low + high) // 2
            try:
                test_font = ImageFont.truetype(font_path, size=mid_size)
            except Exception:
                test_font = ImageFont.load_default()

            total_h = 0
            max_line_w = 0
            ascent_offsets = []

            for line in lines:
                bbox = test_font.getbbox(line)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                ascent_offsets.append(bbox[1])
                max_line_w = max(max_line_w, lw)
                total_h += lh

            line_spacing = int(mid_size * 0.18) * (len(lines) - 1)
            total_h += line_spacing

            if max_line_w <= max_w and total_h <= max_h:
                opt_font = test_font
                opt_size = mid_size
                opt_offset = ascent_offsets[0] if ascent_offsets else 0
                low = mid_size + 1
            else:
                high = mid_size - 1

        if opt_size > best_size:
            best_size = opt_size
            best_font = opt_font
            best_lines = lines
            best_ascent_offset = opt_offset

    if best_font is None:
        best_font = ImageFont.truetype(font_path, size=20)
        best_lines = candidate_layouts[-1]
        best_size = 20
        best_ascent_offset = 0

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


def encode_product_to_incontext_tokens(
    ae: AutoEncoder,
    image_path: str,
    t_offset: float = 60.0,
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


def worker_generate_poster(task_dict: dict, gpu_id: int, result_queue: mp.Queue):
    """
    Dedicated Worker Process running on a specific GPU (e.g. cuda:0 or cuda:1).
    Loads model onto the target GPU and processes the assigned task.
    """
    device = torch.device(f"cuda:{gpu_id}")
    task_id = task_dict.get("id", 1)
    text = task_dict["text"]
    prompt = task_dict["prompt"]
    font = task_dict.get("font", "bevietnam")
    image_ref = task_dict.get("image_ref", None)
    t_text = task_dict.get("t_text", 10.0)
    t_product = task_dict.get("t_product", 60.0)
    width = (task_dict.get("width", 576) // 16) * 16
    height = (task_dict.get("height", 1024) // 16) * 16
    output_path = task_dict.get("output", f"batch_out_{task_id}.png")
    num_steps = task_dict.get("steps", 50)
    guidance = task_dict.get("guidance", 4.0)
    seed = task_dict.get("seed", 42 + task_id)
    model_name = task_dict.get("model_name", "flux.2-klein-base-4b")

    t_start = time.time()
    resolved_font = resolve_font_path(font)

    print(f"\n[Worker GPU {gpu_id} | Task {task_id}] 🚀 Starting: '{text}' ({width}x{height})")

    torch.manual_seed(seed)

    # 1. Render Glyph
    box_w = min(width - 64, 512)
    box_w = (box_w // 16) * 16
    num_words = len(text.replace("\\n", " ").split())
    box_h = 224 if ("\n" in text or num_words >= 4) else 160
    glyph_img = create_glyph_image(text=text, target_width=box_w, target_height=box_h, font_path=resolved_font)
    glyph_output_path = Path(output_path).stem + "_glyph.png"
    glyph_img.save(glyph_output_path)
    print(f"  -> [Worker GPU {gpu_id} | Task {task_id}] 🖼️ Saved Glyph: {glyph_output_path} ({box_w}x{box_h})")

    # 2. Load Models on this Worker's GPU
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)
    text_encoder = load_qwen3_embedder(variant="4B", device=device)


    # 3. Encode Prompt
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device)
        txt_ids = txt_ids.to(device)

    del text_encoder
    torch.cuda.empty_cache()

    # 4. In-Context References
    lat_h = height // 16
    lat_w = width // 16
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device)
    img_ids = img_ids.unsqueeze(0).to(device)

    ref_token_list = []
    ref_id_list = []

    if image_ref and os.path.exists(image_ref):
        prod_tokens, prod_ids = encode_product_to_incontext_tokens(
            ae=ae, image_path=image_ref, t_offset=t_product, device=device
        )
        ref_token_list.append(prod_tokens)
        ref_id_list.append(prod_ids)

    text_tokens, text_ids = encode_glyph_to_incontext_tokens(
        ae=ae, glyph_img=glyph_img, t_offset=t_text, device=device
    )
    ref_token_list.append(text_tokens)
    ref_id_list.append(text_ids)

    all_ref_tokens = torch.cat(ref_token_list, dim=1).to(device)
    all_ref_ids = torch.cat(ref_id_list, dim=1).to(device)

    # 5. Denoise Euler ODE
    timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

    with torch.no_grad():
        out_latent = denoise_cfg(
            model=model,
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
        out_pixels = ae.decode(out_latent.to(device))
        out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        result_img = Image.fromarray(out_pixels)
        result_img.save(output_path)

    elapsed = time.time() - t_start
    print(f"  -> [Worker GPU {gpu_id} | Task {task_id}] ✅ DONE in {elapsed:.2f}s! Saved to: {output_path}")
    result_queue.put({"task_id": task_id, "gpu_id": gpu_id, "output": output_path, "elapsed": elapsed})


def run_parallel_batch(tasks: list[dict]):
    """Dispatches a list of generation tasks in parallel across all available GPUs."""
    start_total = time.time()
    num_gpus = torch.cuda.device_count()
    print("=" * 80)
    print(" 🚀 TENDOO AI: MULTI-GPU PARALLEL BATCH GENERATOR")
    print("=" * 80)
    print(f"📊 Total Tasks to Process : {len(tasks)}")
    print(f"⚡ Available GPUs Detected: {num_gpus} GPU(s)")
    for i, t in enumerate(tasks):
        print(f"  [{i+1}] '{t['text']}' (Font: {t.get('font', 'bevietnam')}) -> {t.get('output', f'out_{i+1}.png')}")
    print("=" * 80)

    if num_gpus < 2:
        print("⚠️ Warning: Only 1 GPU detected. Running sequentially on cuda:0.")
        assigned_gpus = [0] * len(tasks)
    else:
        # Distribute across GPU 0 and GPU 1
        assigned_gpus = [i % num_gpus for i in range(len(tasks))]

    # Run in batches of size = num_gpus
    batch_size = max(num_gpus, 1)
    results = []

    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i : i + batch_size]
        processes = []
        result_queue = mp.Queue()

        print(f"\n[Batch Execution] Launching {len(batch_tasks)} task(s) concurrently...")

        for j, task in enumerate(batch_tasks):
            gpu_id = assigned_gpus[i + j]
            p = mp.Process(target=worker_generate_poster, args=(task, gpu_id, result_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        while not result_queue.empty():
            results.append(result_queue.get())

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"🎉 ALL {len(tasks)} POSTERS GENERATED SUCCESSFULLY IN {total_elapsed:.2f}s!")
    print(f"⚡ Average Speed: {total_elapsed / len(tasks):.2f}s per poster!")
    print("=" * 80 + "\n")


# Built-in Executive Demo Suite Presets
DEMO_PRESETS = [
    {
        "id": 1,
        "text": "TIỆM CÀ PHỄ ANH QUÂN",
        "font": "playfair",
        "prompt": "Ảnh chụp kiến trúc ngoại cảnh một quán cà phê phong cách vintage ấm cúng vào buổi chiều hoàng hôn, biển hiệu gỗ màu nâu trầm cổ điển treo phía trên cửa chính, dòng chữ chạm khắc mạ vàng đồng tinh xảo, ánh nắng vàng rọi qua tán cây, chất lượng điện ảnh",
        "width": 576,
        "height": 1024,
        "output": "batch_demo_1_tiem_ca_phe.png",
    },
    {
        "id": 2,
        "image_ref": "images/ref_prod_02.png",
        "text": "ÂM THANH ĐỈNH CAO",
        "font": "anton",
        "prompt": "Ảnh chụp quảng cáo thương mại cho tai nghe chụp tai không dây màu đen sang trọng đặt trên mặt bục tối giản, dòng chữ vàng gold kim loại dập nổi 3D phản chiếu ánh sáng studio ấm áp, đổ bóng sắc nét lên nền vàng, phong cách điện ảnh cao cấp",
        "width": 576,
        "height": 1024,
        "output": "batch_demo_2_tai_nghe_gold.png",
    },
    {
        "id": 3,
        "image_ref": "images/ref_prod_02.png",
        "text": "CHỐNG ỒN CHỦ ĐỘNG",
        "font": "pacifico",
        "prompt": "Ảnh chụp thương mại công nghệ cho chiếc tai nghe không dây chụp tai màu đen, dòng chữ đèn neon phát quang màu xanh ngọc bích rực rỡ lơ lửng, hắt ánh sáng dạ quang lên bề mặt tai nghe kim loại, không gian studio tối tương phản mạnh, ánh sáng điện ảnh",
        "width": 576,
        "height": 1024,
        "output": "batch_demo_3_tai_nghe_neon.png",
    },
]


if __name__ == "__main__":
    # Ensure multiprocessing works cleanly with PyTorch CUDA
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Tendoo AI - Multi-GPU Parallel Batch Poster Generator")
    parser.add_argument("--preset", action="store_true", help="Run the 3 built-in Executive Demo Suite tasks")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON file containing list of task dictionaries")

    args = parser.parse_args()

    if args.preset or (args.config is None):
        print("💡 Running Built-in Executive Demo Suite (3 Parallel Tasks)...")
        run_parallel_batch(DEMO_PRESETS)
    else:
        with open(args.config, "r", encoding="utf-8") as f:
            custom_tasks = json.load(f)
        run_parallel_batch(custom_tasks)
