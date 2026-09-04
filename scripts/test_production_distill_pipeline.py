#!/usr/bin/env python3
"""
scripts/test_production_distill_pipeline.py

==================================================================================================
TENDOO AI - FINAL PRODUCTION DISTILLED DiT PIPELINE BENCHMARK
==================================================================================================

FEATURES:
  1. Engine: FLUX.2-klein-4B Distilled (8 steps, guidance=1.5, discrete RoPE t=10.0).
  2. Smart Line Balancing: Automatically balances line lengths to eliminate bottleneck lines.
  3. Font-First Adaptive 2D Envelope:
     - Targets optimal font size (140-150pt for 1-2 lines, 120-130pt for 3 lines).
     - Dynamically computes tight-crop width and height with 24px safety margin, aligned to 16px.
     - Zero dead padding voids (100% active text tokens).
     - Line spacing ratio = 0.35 (generous vertical breathing room for stacked diacritics).
  4. Standard 9:16 Portrait Canvas (576 x 1024) for commercial advertising.
  5. Automatic packaging of all outputs into `production_distill_pipeline_results.zip`.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import batched_prc_txt, denoise, get_schedule, prc_img
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import load_ae, load_flow_model
from tendoo.glyph_engine import GlyphEngine, resolve_font_path

PRODUCTION_CASES: List[Dict[str, Any]] = [
    {
        "id": "case1_audio_headline",
        "title": "Hi-End Audio (Long 8-Word Headline, 34 Chars)",
        "hero_text": "ÂM THANH ĐỈNH CAO\ntuôn trào cảm xúc",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo tai nghe hi-end thương mại đỉnh cao, tai nghe chụp tai không dây bluetooth "
            "màu bạc kim loại sang trọng đặt ở trung tâm trên bục đen phẳng, dòng chữ tiêu đề lớn 3D kim loại "
            "mạ chrome sáng bóng kết hợp viền đèn neon xanh cyan phát sáng sắc nét ở phía trên, "
            "ánh sáng studio tương phản cao, phong cách hiện đại tối giản sang trọng, không có watermark"
        ),
        "canvas_w": 576,
        "canvas_h": 1024,
        "seed": 123,
    },
    {
        "id": "case2_burger_opening",
        "title": "Burger Opening (Complex Hooks: Ư, Ừ)",
        "hero_text": "TƯNG BỪNG\nKHAI TRƯƠNG",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo ẩm thực sang trọng, bánh burger bò đẫm phô mai vàng ươm bốc khói nghi ngút "
            "thơm ngon trên bàn gỗ mun, dòng chữ tiêu đề lớn 3D mạ vàng kim loại nổi bật ở phía trên, "
            "ánh sáng studio tương phản cao, phong cách điện ảnh ẩm thực sang trọng, không có watermark"
        ),
        "canvas_w": 576,
        "canvas_h": 1024,
        "seed": 42,
    },
    {
        "id": "case3_anc_headphones",
        "title": "ANC Audio (4 Consecutive Accents: Ố, Ồ, Ủ, Ộ)",
        "hero_text": "CHỐNG ỒN\nCHỦ ĐỘNG",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo tai nghe chống ồn cao cấp, tai nghe chụp tai không dây màu bạc kim loại "
            "đặt trên bục đen phẳng ở trung tâm, dòng chữ tiêu đề lớn 3D mạ vàng kim loại đúc nổi khối sắc nét "
            "ở phía trên, ánh sáng studio mềm mại, phong cách hiện đại tối giản, không có watermark"
        ),
        "canvas_w": 576,
        "canvas_h": 1024,
        "seed": 123,
    },
    {
        "id": "case4_flash_sale_3lines",
        "title": "Flash Sale (Numbers, %, Balanced 3-Lines)",
        "hero_text": "SIÊU SALE 50%\nDUY NHẤT\nHÔM NAY",
        "font": "bevietnam",
        "prompt": (
            "Poster flash sale thương mại điện tử bùng nổ, các hộp quà tặng màu đỏ và dải ruy băng vàng "
            "bay lơ lửng xung quanh, dòng chữ tiêu đề lớn 3D mạ vàng kim loại đúc khối rực rỡ ở phía trên, "
            "ánh sáng studio tương phản cao phong cách lễ hội mua sắm, không có watermark"
        ),
        "canvas_w": 576,
        "canvas_h": 1024,
        "seed": 777,
    },
    {
        "id": "case5_menu_single_line",
        "title": "Artisan Menu (Single Medium Line)",
        "hero_text": "THỰC ĐƠN ĐẶC BIỆT",
        "font": "playfair",
        "prompt": (
            "Poster thực đơn nhà hàng ẩm thực sang trọng, bàn gỗ mun tối màu với đĩa bít tết bò nướng "
            "xèo xèo và ly cocktail cam sả mát lạnh, ánh nến lung linh mờ ảo, dòng chữ tiêu đề lớn 3D "
            "khắc gỗ mạ vàng đồng cổ sắc nét tinh xảo ở phía trên, phong cách điện ảnh nghệ thuật ẩm thực, "
            "không có watermark"
        ),
        "canvas_w": 576,
        "canvas_h": 1024,
        "seed": 999,
    },
    {
        "id": "case6_burger_king_short",
        "title": "Burger Brand (Ultra-Short 2 Words)",
        "hero_text": "BURGER KING",
        "font": "bevietnam",
        "prompt": (
            "Poster quảng cáo ẩm thực burger hảo hạng, chiếc bánh burger bò đẫm phô mai tan chảy thơm ngon "
            "trên đĩa phẳng, dòng chữ tiêu đề lớn 3D mạ vàng kim loại đúc khối vuông vức nổi bật ở phía trên, "
            "ánh sáng studio tương phản cao sang trọng, không có watermark"
        ),
        "canvas_w": 576,
        "canvas_h": 1024,
        "seed": 888,
    },
]


def balance_text_lines(text: str) -> List[str]:
    """
    Intelligently balances text into 1, 2, or 3 lines to minimize length variance.
    Preserves user explicit line breaks if already reasonably balanced.
    """
    if "\n" in text:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # If user explicitly formatted multiple lines and none is excessively long (<= 4 words), preserve
        if len(lines) >= 2 and all(len(l.split()) <= 4 for l in lines):
            return lines

    words = text.replace("\n", " ").strip().split()
    if len(words) <= 3 or (len(words) <= 4 and len(text) <= 22):
        return [" ".join(words)]
    elif len(words) <= 6:
        # Split into 2 balanced lines by char length
        best_split, min_diff = 1, 999
        for i in range(1, len(words)):
            l1 = " ".join(words[:i])
            l2 = " ".join(words[i:])
            diff = abs(len(l1) - len(l2))
            if diff < min_diff:
                min_diff = diff
                best_split = i
        return [" ".join(words[:best_split]), " ".join(words[best_split:])]
    else:
        # Split into 3 balanced lines by minimizing max-min length variance
        best_p1, best_p2, min_var = 1, 2, 999999
        for i in range(1, len(words) - 1):
            for j in range(i + 1, len(words)):
                l1 = " ".join(words[:i])
                l2 = " ".join(words[i:j])
                l3 = " ".join(words[j:])
                lengths = [len(l1), len(l2), len(l3)]
                var = max(lengths) - min(lengths)
                if var < min_var:
                    min_var = var
                    best_p1, best_p2 = i, j
        return [" ".join(words[:best_p1]), " ".join(words[best_p1:best_p2]), " ".join(words[best_p2:])]


def render_adaptive_2d_glyph(
    lines: List[str],
    font_name: str,
    target_font_size_pt: int = 145,
    spacing_ratio: float = 0.35,
    padding_px: int = 24,
    max_box_w: int = 1152,
) -> Tuple[Image.Image, int, int, int, int]:
    """
    Renders an adaptive 2D tight-cropped glyph with zero dead margins and generous line spacing.
    Returns: (PIL Image, font_size_pt, box_w, box_h, token_count)
    """
    _, font_path, _ = resolve_font_path(font_name)
    ge = GlyphEngine()

    # Binary search font size capped at target_font_size_pt
    low, high, chosen_size, chosen_font = 8, target_font_size_pt, 8, None
    while low <= high:
        mid = (low + high) // 2
        f = ge.get_font(font_path, mid)
        lws = [f.getbbox(l)[2] - f.getbbox(l)[0] for l in lines]
        lhs = [f.getbbox(l)[3] - f.getbbox(l)[1] for l in lines]
        tot_w = max(lws)
        if tot_w + 2 * padding_px <= max_box_w:
            chosen_size, chosen_font = mid, f
            low = mid + 1
        else:
            high = mid - 1

    lws = [chosen_font.getbbox(l)[2] - chosen_font.getbbox(l)[0] for l in lines]
    lhs = [chosen_font.getbbox(l)[3] - chosen_font.getbbox(l)[1] for l in lines]
    line_gap = int(chosen_size * spacing_ratio)

    content_w = max(lws)
    content_h = sum(lhs) + line_gap * (len(lines) - 1)

    # 2D Tight Crop aligned to 16px multiples
    box_w = int(math.ceil((content_w + 2 * padding_px) / 16.0) * 16)
    box_h = int(math.ceil((content_h + 2 * padding_px) / 16.0) * 16)

    img = Image.new("RGB", (box_w, box_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = (box_h - content_h) // 2
    for idx, l in enumerate(lines):
        lw = lws[idx]
        curr_x = (box_w - lw) // 2
        bbox = chosen_font.getbbox(l)
        dx = curr_x - bbox[0]
        dy = curr_y - bbox[1]
        draw.text((dx, dy), l, fill=(255, 255, 255), font=chosen_font)
        curr_y += lhs[idx] + line_gap

    tokens = (box_w // 16) * (box_h // 16)
    return img, chosen_size, box_w, box_h, tokens


def encode_glyph_to_ref_tokens(
    ae: AutoEncoder, img: Image.Image, t_offset: float, device: str | torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes a glyph image into canonical 4D RoPE tokens at local origin (0, 0)."""
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latent = ae.encode(tensor)
    ref_tokens, _ = prc_img(latent[0])
    ref_tokens = ref_tokens.unsqueeze(0)
    g_h, g_w = latent.shape[2], latent.shape[3]
    t_coords = torch.full((g_h, g_w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(g_h, dtype=torch.float32, device=device).unsqueeze(1).expand(g_h, g_w)
    w_coords = torch.arange(g_w, dtype=torch.float32, device=device).unsqueeze(0).expand(g_h, g_w)
    l_coords = torch.zeros((g_h, g_w), dtype=torch.float32, device=device)
    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0)
    return ref_tokens, ref_ids


def find_distill_checkpoint(base_checkpoint_dir: Path) -> Path | None:
    """Searches for flux-2-klein-4b.safetensors across standard paths."""
    candidates = [
        base_checkpoint_dir / "flux-2-klein-4b.safetensors",
        base_checkpoint_dir.parent / "FLUX.2-klein-4B" / "flux-2-klein-4b.safetensors",
        base_checkpoint_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4B/flux-2-klein-4b.safetensors"),
        Path("/home/jovyan/persistent-data/FLUX.2-klein-4b.safetensors"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Final Production FLUX.2 Distilled DiT Typography Benchmark Suite"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
        help="Directory containing VAE and Text Encoder",
    )
    parser.add_argument(
        "--distill_model_path",
        type=str,
        default=None,
        help="Explicit path to flux-2-klein-4b.safetensors",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        help="Which case to run ('all' or case ID)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text (if set, runs single custom case)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for single custom case",
    )
    parser.add_argument(
        "--font",
        type=str,
        default="bevietnam",
        help="Font for custom case",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=576,
        help="Canvas width (default: 576)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Canvas height (default: 1024)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Distill ODE steps (default: 8)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.5,
        help="Distill guidance scale (default: 1.5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_production_distill_pipeline",
        help="Output directory for results",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Device Setup
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            device_dit = torch.device("cuda:0")
            device_ae = torch.device("cuda:1")
            print(f"[HW] Dual-GPU Setup: DiT on {device_dit} | VAE & TextEncoder on {device_ae}")
        else:
            device_dit = torch.device("cuda:0")
            device_ae = torch.device("cuda:0")
            print(f"[HW] Single-GPU Setup: All models on {device_dit}")
    else:
        print("[ERROR] CUDA GPU is required!")
        sys.exit(1)

    # 2. Checkpoint Resolution
    base_dir = Path(args.checkpoint_dir)
    distill_file = Path(args.distill_model_path) if args.distill_model_path else find_distill_checkpoint(base_dir)
    if not distill_file or not distill_file.exists():
        print(f"[ERROR] Distilled DiT checkpoint not found around: {base_dir}")
        print("  Please specify with --distill_model_path <path_to_flux-2-klein-4b.safetensors>")
        sys.exit(1)

    print("\n" + "=" * 105)
    print("🚀 TENDOO AI - FINAL PRODUCTION DISTILLED DiT PIPELINE TEST")
    print("=" * 105)
    print(f"  Target DiT Model   : {distill_file}")
    print(f"  Inference Defaults : Steps={args.steps}, Guidance={args.guidance}, t_offset=10.0")
    print(f"  Output Directory   : {out_path.resolve()}")

    # 3. Load Models
    print("\n[1/3] Loading Distilled DiT (4B)...")
    os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_file)
    model = load_flow_model(model_name="flux.2-klein-4b", device=device_dit)
    model.eval()

    print("[2/3] Loading VAE and Text Encoder (Qwen3-4B-FP8)...")
    os.environ["FLUX_CHECKPOINT_DIR"] = str(base_dir)
    ae = load_ae(model_name="flux.2-klein-base-4b", device=device_ae)
    ae.eval()
    text_encoder = load_qwen3_embedder(variant="4B", device=device_ae)

    # Filter Cases
    if args.text and args.prompt:
        cases_to_run = [
            {
                "id": "custom_case",
                "title": "Custom Production Case",
                "hero_text": args.text.replace("\\n", "\n"),
                "font": args.font,
                "prompt": args.prompt,
                "canvas_w": args.width,
                "canvas_h": args.height,
                "seed": 42,
            }
        ]
    elif args.case == "all":
        cases_to_run = PRODUCTION_CASES
    else:
        cases_to_run = [c for c in PRODUCTION_CASES if c["id"] == args.case]
        if not cases_to_run:
            print(f"[ERROR] Case '{args.case}' not found!")
            sys.exit(1)

    print(f"\n[3/3] Executing Production Pipeline ({len(cases_to_run)} Cases)...")
    results_table: List[Dict[str, Any]] = []
    generated_files: List[Path] = []

    for idx, c in enumerate(cases_to_run, 1):
        cid = c["id"]
        title = c["title"]
        raw_text = c["hero_text"]
        font = c["font"]
        prompt = c["prompt"]
        canvas_w = c["canvas_w"]
        canvas_h = c["canvas_h"]
        seed = c["seed"]

        # A. Smart Line Balancing
        balanced_lines = balance_text_lines(raw_text)

        # Target font size: 145pt for 1-2 lines, 125pt for 3 lines
        target_pt = 125 if len(balanced_lines) >= 3 else 145

        # B. Adaptive 2D Tight-Crop Glyph Rendering
        glyph_img, font_size, box_w, box_h, tokens = render_adaptive_2d_glyph(
            lines=balanced_lines,
            font_name=font,
            target_font_size_pt=target_pt,
            spacing_ratio=0.35,
            padding_px=24,
            max_box_w=1152,
        )

        glyph_file = out_path / f"glyph__{cid}.png"
        glyph_img.save(glyph_file)

        print("\n" + "-" * 95)
        print(f"▶ [{idx}/{len(cases_to_run)}] {title} ({cid})")
        print(f"   Lines ({len(balanced_lines)}): {' // '.join(balanced_lines)}")
        print(f"   Glyph Box: {box_w}x{box_h}px ({font_size}pt, {tokens} tokens, ZERO dead void)")
        print(f"   Canvas: {canvas_w}x{canvas_h} (Aspect: {canvas_w/canvas_h:.3f}) | Seed: {seed}")

        # C. Encode Glyph to Ref Tokens at t=10.0
        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae, glyph_img, t_offset=10.0, device=device_ae)
        ref_tokens = ref_tokens.to(device_dit)
        ref_ids = ref_ids.to(device_dit)

        # D. Encode Prompt
        with torch.no_grad():
            txt_prompt = text_encoder([prompt]).to(device=device_dit, dtype=torch.bfloat16)
        txt_tokens, txt_ids = batched_prc_txt(txt_prompt)

        # E. Sample Distilled ODE
        lat_w, lat_h = canvas_w // 16, canvas_h // 16
        torch.manual_seed(seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])

        t0 = time.time()
        with torch.no_grad():
            out_tokens = denoise(
                model=model,
                img=img_tokens,
                img_ids=img_ids,
                txt=txt_tokens,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=args.guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
            lat_2d = rearrange(out_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
            img_out = ae.decode(lat_2d.to(device_ae))
        dur = time.time() - t0

        pil_img = Image.fromarray(
            ((img_out[0].float().clamp(-1, 1) + 1.0) * 127.5)
            .permute(1, 2, 0)
            .byte()
            .cpu()
            .numpy()
        )

        out_file = out_path / f"{cid}__576x1024.png"
        pil_img.save(out_file)
        generated_files.append(out_file)
        print(f"   [✓] Completed in {dur:.2f}s! -> Saved: {out_file.name}")

        results_table.append({
            "case": cid,
            "title": title,
            "lines": len(balanced_lines),
            "lines_text": balanced_lines,
            "font_pt": font_size,
            "font_name": font,
            "box": f"{box_w}x{box_h}",
            "tokens": tokens,
            "canvas": f"{canvas_w}x{canvas_h}",
            "time_s": round(dur, 2),
            "glyph_file": glyph_file.name,
            "output_file": out_file.name,
            "prompt": prompt,
            "seed": seed,
        })

        # VRAM cleanup
        del ref_tokens, ref_ids, txt_tokens, txt_ids, z_init, img_tokens, img_ids, out_tokens, lat_2d, img_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 4. Generate HTML Visual Report
    html_report_path = out_path / "index.html"
    generate_html_report(results_table, html_report_path)
    print(f"\n[Dashboard] Generated visual report: {html_report_path.name}")

    # 5. Packaging Archive & Summary Table
    zip_filename = out_path / "production_distill_pipeline_results.zip"
    print(f"\n[Packaging] Compressing all outputs into: {zip_filename.name}...")
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if html_report_path.exists():
            zf.write(html_report_path, arcname="index.html")
        for f in generated_files:
            zf.write(f, arcname=f.name)
        for g in out_path.glob("glyph__*.png"):
            zf.write(g, arcname=g.name)

    print("\n" + "=" * 115)
    print("🏆 FINAL PRODUCTION DISTILLED DiT PIPELINE SUMMARY TABLE")
    print("=" * 115)
    print(f"{'Case ID':<25} | {'Lines':<5} | {'Font':<7} | {'Adaptive Box':<14} | {'Tokens':<7} | {'Canvas':<11} | {'Time (s)':<9} | {'File'}")
    print("-" * 115)
    for r in results_table:
        print(
            f"{r['case']:<25} | {r['lines']}L{' ':<3} | {r['font_pt']}pt{' ':<3} | {r['box']:<14} | "
            f"{r['tokens']:<7} | {r['canvas']:<11} | {r['time_s']}s{' ':<5} | {r['output_file']}"
        )
    print("=" * 115)
    print(f"\n🎉 ALL 6 PRODUCTION CASES FINISHED! Download archive directly from:")
    print(f"   👉 {zip_filename.resolve()}\n")


def generate_html_report(results: List[Dict[str, Any]], out_file: Path) -> None:
    """Builds a rich, self-contained HTML visual comparison dashboard."""
    avg_time = sum(r["time_s"] for r in results) / max(1, len(results))

    cards_html = []
    for r in results:
        lines_str = " &bull; ".join(r["lines_text"])
        card = f"""
        <div class="card">
          <div class="card-header">
            <div>
              <span class="badge badge-primary">{r["case"]}</span>
              <h3 class="case-title">{r["title"]}</h3>
            </div>
            <div class="timing-badge">⚡ {r["time_s"]}s</div>
          </div>

          <div class="meta-row">
            <span class="meta-item"><b>Lines:</b> {r["lines"]} ({lines_str})</span>
            <span class="meta-item"><b>Font:</b> {r["font_name"]} ({r["font_pt"]}pt)</span>
            <span class="meta-item"><b>Glyph Box:</b> {r["box"]} ({r["tokens"]} tokens)</span>
            <span class="meta-item"><b>Canvas:</b> {r["canvas"]}</span>
            <span class="meta-item"><b>Seed:</b> {r["seed"]}</span>
          </div>

          <div class="prompt-box">
            <b>Prompt:</b> {r["prompt"]}
          </div>

          <div class="image-row">
            <div class="image-col glyph-col">
              <div class="img-label">In-Context Glyph (t=10.0)</div>
              <div class="glyph-wrapper">
                <img src="{r["glyph_file"]}" alt="Glyph {r["case"]}" />
              </div>
            </div>
            <div class="image-col poster-col">
              <div class="img-label">FLUX.2 Distilled 8-Step Output (576x1024)</div>
              <img src="{r["output_file"]}" alt="Poster {r["case"]}" />
            </div>
          </div>
        </div>
        """
        cards_html.append(card)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Tendoo AI - Production Distilled DiT Benchmark</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #131b2e;
      --border: #23314f;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --primary: #38bdf8;
      --accent: #10b981;
      --gold: #fbbf24;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 32px 24px;
      line-height: 1.5;
    }}
    .header {{
      max-width: 1200px;
      margin: 0 auto 32px auto;
      text-align: center;
    }}
    h1 {{
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #fff;
      margin-bottom: 8px;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 15px;
      margin-bottom: 20px;
    }}
    .stats-bar {{
      display: flex;
      justify-content: center;
      gap: 24px;
      flex-wrap: wrap;
    }}
    .stat-pill {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      padding: 8px 16px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
    }}
    .stat-pill span {{ color: var(--primary); }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 32px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}
    .badge-primary {{ background: rgba(56, 189, 248, 0.15); color: var(--primary); }}
    .case-title {{
      font-size: 18px;
      font-weight: 700;
      color: #fff;
    }}
    .timing-badge {{
      font-size: 16px;
      font-weight: 800;
      color: var(--accent);
      background: rgba(16, 185, 129, 0.12);
      padding: 6px 14px;
      border-radius: 8px;
      border: 1px solid rgba(16, 185, 129, 0.25);
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .meta-item b {{ color: var(--text); }}
    .prompt-box {{
      background: #0d1322;
      border: 1px solid #1a253e;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 13px;
      color: #cbd5e1;
      margin-bottom: 20px;
    }}
    .prompt-box b {{ color: var(--gold); }}
    .image-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      align-items: start;
    }}
    @media (max-width: 768px) {{
      .image-row {{ grid-template-columns: 1fr; }}
    }}
    .image-col {{
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .img-label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--muted);
      margin-bottom: 8px;
      width: 100%;
      text-align: left;
    }}
    .glyph-wrapper {{
      background: #000;
      border: 1px dashed var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      justify-content: center;
      align-items: center;
      width: 100%;
      min-height: 200px;
    }}
    .glyph-wrapper img {{
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }}
    .poster-col img {{
      width: 100%;
      max-width: 480px;
      height: auto;
      border-radius: 12px;
      border: 1px solid var(--border);
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>🚀 TENDOO AI - FINAL PRODUCTION DISTILLED DiT BENCHMARK</h1>
    <div class="subtitle">FLUX.2-klein-4B Distilled (8 ODE Steps, Guidance=1.5, RoPE Discrete t=10.0, 9:16 Portrait Canvas)</div>
    <div class="stats-bar">
      <div class="stat-pill">Engine: <span>FLUX.2-klein-4B Distilled</span></div>
      <div class="stat-pill">Avg Latency: <span>{avg_time:.2f}s</span></div>
      <div class="stat-pill">Total Cases: <span>{len(results)}</span></div>
      <div class="stat-pill">Envelope: <span>2D Adaptive Tight Crop</span></div>
    </div>
  </div>

  <div class="container">
    {"".join(cards_html)}
  </div>
</body>
</html>
"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)



if __name__ == "__main__":
    main()
