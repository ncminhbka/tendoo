#!/usr/bin/env python3
"""
scripts/probe_klein_distill_glyph.py

==================================================================================================
TENDOO AI - FLUX.2-KLEIN-4B DISTILLED IN-CONTEXT GLYPH PROBE
==================================================================================================

OBJECTIVE:
  Empirically verify whether the fast Guidance-Distilled model (FLUX.2-klein-4B Distill)
  preserves the 100% Vietnamese diacritic fidelity of the In-Context Glyph at t=10.0 across:
    1. Different commercial text payloads and aspect ratios (e.g. Square 1024x1024).
    2. Dense multiline literary text: the 4-line "Tây Tiến" poem (28 words, 119 characters).

BUILT-IN PRESETS:
  - --preset commercial : "ÂM THANH ĐỈNH CAO" on 1024x1024 canvas (1:1 square, Box 768x224, font BeVietnam)
  - --preset tay_tien   : 4-line Tây Tiến poem on 1024x1024 canvas (Box 896x512, font Playfair)
  - --preset all        : Runs BOTH presets sequentially in 1 session (loads 4B model only once!)

SAMPLING SPECIFICATIONS (per BFL FLUX.2 Docs):
  - Model: FLUX.2-klein-4B (Distilled)
  - Guidance mode: Guidance-distilled (embedded scalar guidance via MLPEmbedder, NO CFG 2x batch)
  - Proven configuration: steps=8, guidance=1.5 (executes in ~2.3s - 5s on 2x A30)
  - Denoise function: src.flux2.sampling.denoise() (single batch forward with img_cond_seq)

EXECUTION CONSTRAINTS:
  - Runs on Remote Server (2x NVIDIA A30 or Single GPU).
  - Compliant with AGENTS.md Rule 28: Zero HTML, pure PNG + ASCII summary.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image

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
from tendoo.glyph_engine import GlyphInfo, render_glyph

CANVAS_DEFAULT = (576, 1024)
DEFAULT_SEEDS = [42, 123]

PRESETS: Dict[str, Dict[str, Any]] = {
    "commercial": {
        "name": "commercial",
        "description": "Commercial Slogan 'ÂM THANH ĐỈNH CAO' | 1:1 Square (1024x1024) | Box 768x224",
        "text": "ÂM THANH ĐỈNH CAO",
        "prompt": (
            "Poster quảng cáo tai nghe hi-res cao cấp trên bục trưng bày studio tối giản, "
            "ánh sáng đèn led neon xanh tím phản chiếu sang trọng, dòng chữ tiêu đề lớn 3D "
            "mạ bạc phát quang sắc nét ở phía trên, bố cục cân đối điện ảnh, chi tiết tinh xảo, "
            "không có watermark"
        ),
        "font": "bevietnam",
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 768,
        "box_h": 224,
        "steps": [8],
        "guidance": [1.5],
        "seeds": [42, 123],
        "output_dir": "output_distill_commercial_1024",
    },
    "tay_tien": {
        "name": "tay_tien",
        "description": "Poem 'Tây Tiến' 4 lines (28 words) | 1:1 Square (1024x1024) | Box 896x512",
        "text": (
            "Sông Mã xa rồi Tây Tiến ơi\n"
            "Nhớ về rừng núi nhớ chơi vơi.\n"
            "Sài Khao sương lấp đoàn quân mỏi,\n"
            "Mường Lát hoa về trong đêm hơi."
        ),
        "prompt": (
            "Bức vách đá sa thạch cổ kính phẳng sừng sững ở tiền cảnh góc bên, bốn câu thơ "
            "chữ khắc chìm mạ vàng đồng cổ sắc nét trên mặt đá phẳng phủ rêu phong, hậu cảnh "
            "núi non Tây Bắc hùng vĩ mây mù hoàng hôn le lói, phong cách điện ảnh sử thi cổ "
            "trang, ánh sáng studio tương phản cao"
        ),
        "font": "playfair",
        "canvas_w": 1024,
        "canvas_h": 1024,
        "box_w": 896,
        "box_h": 512,
        "steps": [8, 12],
        "guidance": [1.5],
        "seeds": [42],
        "output_dir": "output_distill_tay_tien_1024",
    },
}


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
    """Searches for the 4B distilled DiT checkpoint across standard paths."""
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


def run_single_case(
    case_name: str,
    text: str,
    prompt: str,
    font: str,
    steps_list: List[int],
    guidance_list: List[float],
    seeds: List[int],
    output_dir: str,
    canvas_w: int,
    canvas_h: int,
    box_w: int,
    box_h: int,
    t_offset: float,
    model: Flux2,
    ae: AutoEncoder,
    text_encoder: Any,
    device_dit: str | torch.device,
    device_ae: str | torch.device,
) -> List[Dict[str, Any]]:
    """Runs glyph rasterization, prompt embedding, and distilled ODE sweep for 1 test case."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    clean_text = text.replace("\\n", "\n").strip()

    print("\n" + "=" * 90)
    print(f"🎯 EXECUTING TEST CASE: [{case_name.upper()}]")
    print("=" * 90)
    print(f"  Canvas Dimensions : {canvas_w}x{canvas_h}px")
    print(f"  Glyph Box (WxH)   : {box_w}x{box_h}px")
    print(f"  Font Archetype    : {font}")
    print(f"  Steps Sweep       : {steps_list}")
    print(f"  Guidance Sweep    : {guidance_list}")
    print(f"  Seeds             : {seeds}")
    print(f"  Output Directory  : {out_path.resolve()}")
    print("  Text Payload:")
    for line in clean_text.split("\n"):
        print(f"    │ {line}")

    # 1. Render Glyph Bitmap
    print("\n  [Step 1/3] Rasterizing In-Context Glyph Bitmap...")
    glyph_info: GlyphInfo = render_glyph(
        text=clean_text,
        font_name_or_path=font,
        target_width=box_w,
        target_height=box_h,
        auto_size=False,
    )
    glyph_file = out_path / f"probe_glyph_{case_name}.png"
    glyph_info.image.save(glyph_file)
    print(
        f"    ✓ Glyph Saved   : {glyph_file.name} "
        f"({glyph_info.width_px}x{glyph_info.height_px}px, {glyph_info.font_size_pt}pt, {glyph_info.token_count}tok)"
    )

    # 2. Encode glyph to ref tokens at t_offset (canonical t=10.0)
    ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae, glyph_info.image, t_offset=t_offset, device=device_ae)
    ref_tokens = ref_tokens.to(device_dit)
    ref_ids = ref_ids.to(device_dit)

    # 3. Encode prompt for Guidance-Distilled model
    print(f"\n  [Step 2/3] Encoding Text Prompt via Qwen3-4B-FP8:")
    print(f"    Prompt: '{prompt}'")
    with torch.no_grad():
        txt_emb = text_encoder([prompt]).to(device=device_dit, dtype=torch.bfloat16)
    txt_tokens, txt_ids = batched_prc_txt(txt_emb)

    # Prepare Canvas Latent Dimensions
    canvas_w_snapped = (canvas_w // 16) * 16
    canvas_h_snapped = (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w_snapped // 16, canvas_h_snapped // 16

    # 4. Sweep Grid: Steps x Guidance x Seeds
    print(f"\n  [Step 3/3] Running ODE Integration Sweep (Canvas {canvas_w_snapped}x{canvas_h_snapped}px)...")
    case_results: List[Dict[str, Any]] = []

    for steps in steps_list:
        for guidance in guidance_list:
            for seed in seeds:
                tag = f"{case_name}_steps{steps}_g{guidance}_seed{seed}_{canvas_w_snapped}x{canvas_h_snapped}"
                print(f"    ▶️ Sampling: steps={steps}, guidance={guidance}, seed={seed}...", end="", flush=True)
                t_start = time.time()

                torch.manual_seed(seed)
                z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
                img_tokens, img_ids = prc_img(z_init[0])
                img_tokens = img_tokens.unsqueeze(0).to(device_dit)
                img_ids = img_ids.unsqueeze(0).to(device_dit)

                # Time Schedule for num_steps
                timesteps = get_schedule(num_steps=steps, image_seq_len=img_tokens.shape[1])

                with torch.no_grad():
                    # Single-batch denoise forward pass per step (No CFG doubling!)
                    out_tokens = denoise(
                        model=model,
                        img=img_tokens,
                        img_ids=img_ids,
                        txt=txt_tokens,
                        txt_ids=txt_ids,
                        timesteps=timesteps,
                        guidance=guidance,
                        img_cond_seq=ref_tokens,
                        img_cond_seq_ids=ref_ids,
                    )

                    # Decode VAE
                    lat_2d = rearrange(out_tokens[0], "(h w) c -> 1 c h w", h=lat_h, w=lat_w)
                    img_out = ae.decode(lat_2d.to(device_ae))
                    dur_dit = time.time() - t_start

                    # Convert to PIL
                    pil_img = Image.fromarray(
                        ((img_out[0].float().clamp(-1, 1) + 1.0) * 127.5)
                        .permute(1, 2, 0)
                        .byte()
                        .cpu()
                        .numpy()
                    )

                    out_file = out_path / f"{tag}.png"
                    pil_img.save(out_file)
                    print(f" Done in {dur_dit:.2f}s! -> Saved: {out_file.name}")

                    case_results.append({
                        "case": case_name,
                        "canvas": f"{canvas_w_snapped}x{canvas_h_snapped}",
                        "steps": steps,
                        "guidance": guidance,
                        "seed": seed,
                        "duration_s": round(dur_dit, 2),
                        "file": out_file.name,
                        "output_dir": str(out_path),
                    })

    # Clean up intermediate tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return case_results


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI FLUX.2-klein-4B Distill Glyph Probe")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["commercial", "tay_tien", "all", "custom"],
        default=None,
        help="Run built-in test preset ('commercial', 'tay_tien', 'all') or custom parameters",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="CHINH PHỤC MỌI GIỚI HẠN",
        help="Text payload for Hero Title glyph",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Poster quảng cáo thể thao ngoài trời hiện đại, nền ánh sáng hoàng hôn điện ảnh kịch tính, "
            "dòng chữ tiêu đề lớn 3D dập nổi mạ vàng kim loại sắc nét ở phía trên, bố cục sạch sẽ "
            "chuyên nghiệp, không có watermark"
        ),
        help="Text prompt for background & material",
    )
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[8],
        help="Steps to sweep (default: 8)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        nargs="+",
        default=[1.5],
        help="Guidance values to sweep (default: 1.5)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to test")
    parser.add_argument("--box_w", type=int, default=512, help="Glyph box width (default: 512)")
    parser.add_argument("--box_h", type=int, default=224, help="Glyph box height (default: 224)")
    parser.add_argument("--canvas_w", type=int, default=CANVAS_DEFAULT[0], help="Canvas width")
    parser.add_argument("--canvas_h", type=int, default=CANVAS_DEFAULT[1], help="Canvas height")
    parser.add_argument("--t_offset", type=float, default=10.0, help="RoPE time offset (default: 10.0)")
    parser.add_argument(
        "--distill_model_path",
        type=str,
        default=None,
        help="Direct path to flux-2-klein-4b.safetensors",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
        help="Base persistent data dir containing text_encoder and VAE",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_probe_distill_glyph",
        help="Output directory (for custom runs)",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("🚀 TENDOO AI - FLUX.2-KLEIN-4B DISTILLED IN-CONTEXT GLYPH PROBE SUITE")
    print("=" * 100)

    # 1. Hardware setup
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = "cuda:0"
        device_ae = "cuda:1"
        print(f"  [GPU Setup] Dual GPU active: DiT on {device_dit}, VAE/Qwen3 on {device_ae}")
    elif num_gpus == 1:
        device_dit = "cuda:0"
        device_ae = "cuda:0"
        print(f"  [GPU Setup] Single GPU active: {device_dit}")
    else:
        device_dit = "cpu"
        device_ae = "cpu"
        print("  [!] WARNING: Running on CPU (testing syntax only)")

    # 2. Checkpoint resolution
    base_dir = Path(args.checkpoint_dir or "/home/jovyan/persistent-data/FLUX.2-klein-base-4B")
    if args.distill_model_path:
        model_file = Path(args.distill_model_path)
    else:
        model_file = find_distill_checkpoint(base_dir)

    if not model_file or not model_file.exists():
        print(f"\n[ERROR] Distilled DiT checkpoint not found!")
        print(f"  Searched locations around: {base_dir}")
        print("  Please specify explicitly with --distill_model_path <path_to_flux-2-klein-4b.safetensors>")
        sys.exit(1)

    print(f"\n[1/3] Loading Distilled DiT from: {model_file}")
    os.environ["KLEIN_4B_MODEL_PATH"] = str(model_file)
    model = load_flow_model(
        model_name="flux.2-klein-4b",
        device=device_dit,
    )
    model.eval()

    print("\n[2/3] Loading VAE and Text Encoder (Qwen3-4B-FP8)...")
    if args.checkpoint_dir:
        os.environ["FLUX_CHECKPOINT_DIR"] = str(args.checkpoint_dir)
    ae = load_ae(model_name="flux.2-klein-base-4b", device=device_ae)
    ae.eval()

    text_encoder = load_qwen3_embedder(variant="4B", device=device_ae)

    # 3. Determine Execution Cases
    cases_to_run: List[Dict[str, Any]] = []

    if args.preset == "all":
        cases_to_run.append(PRESETS["commercial"])
        cases_to_run.append(PRESETS["tay_tien"])
    elif args.preset in PRESETS:
        preset_cfg = dict(PRESETS[args.preset])
        # Allow CLI overrides if passed
        if args.steps != [8]:
            preset_cfg["steps"] = args.steps
        if args.guidance != [1.5]:
            preset_cfg["guidance"] = args.guidance
        if args.seeds != DEFAULT_SEEDS:
            preset_cfg["seeds"] = args.seeds
        cases_to_run.append(preset_cfg)
    else:
        # Custom run from CLI args
        cases_to_run.append({
            "name": "custom",
            "description": "Custom CLI run",
            "text": args.text,
            "prompt": args.prompt,
            "font": args.font,
            "canvas_w": args.canvas_w,
            "canvas_h": args.canvas_h,
            "box_w": args.box_w,
            "box_h": args.box_h,
            "steps": args.steps,
            "guidance": args.guidance,
            "seeds": args.seeds,
            "output_dir": args.output_dir,
        })

    print(f"\n[3/3] Queued {len(cases_to_run)} test case(s) for execution.")

    all_results: List[Dict[str, Any]] = []
    for c in cases_to_run:
        results = run_single_case(
            case_name=c["name"],
            text=c["text"],
            prompt=c["prompt"],
            font=c["font"],
            steps_list=c["steps"],
            guidance_list=c["guidance"],
            seeds=c["seeds"],
            output_dir=c["output_dir"],
            canvas_w=c["canvas_w"],
            canvas_h=c["canvas_h"],
            box_w=c["box_w"],
            box_h=c["box_h"],
            t_offset=args.t_offset,
            model=model,
            ae=ae,
            text_encoder=text_encoder,
            device_dit=device_dit,
            device_ae=device_ae,
        )
        all_results.extend(results)

    # 4. Final Summary Report
    print("\n" + "=" * 105)
    print("📊 FLUX.2-KLEIN-4B DISTILLED IN-CONTEXT GLYPH PROBE OVERALL SUMMARY")
    print("=" * 105)
    header = f"{'CASE':<12} | {'CANVAS':<10} | {'STEPS':<6} | {'GUIDANCE':<8} | {'SEED':<6} | {'TIME':<8} | {'OUTPUT FILE':<40}"
    print(header)
    print("-" * 105)
    for r in all_results:
        print(
            f"{r['case']:<12} | {r['canvas']:<10} | {r['steps']:<6} | {r['guidance']:<8} | "
            f"{r['seed']:<6} | {r['duration_s']}s{' ':<2} | {r['file']:<40}"
        )
    print("=" * 105)
    print("\n[✓] All probe runs finished successfully!")


if __name__ == "__main__":
    main()
