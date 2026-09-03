#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - GLYPH SELF-ASPECT-RATIO PROBE (ROUND 3: DISENTANGLING WIDTH-RATIO FROM ASPECT-RATIO)
====================================================================================================
Script: scripts/probe_glyph_aspect_ratio.py
Purpose:
    Round 2 (probe_glyph_width_ratio.py) falsified the pure "glyph-to-canvas width ratio" theory:
    Glyph A ("Đậm đà hương vị cà phê Việt", single line, lat 31x7, aspect 4.43:1) failed at EVERY
    tested canvas ratio from 1.5 down to 0.4 -- including 0.4, well inside what Round 2 showed was
    the SAFE zone for Glyph B (lat 17x7, aspect 2.43:1, which passed at ratio 0.55). Since Glyph A
    failed even at a comfortably low canvas ratio, canvas-ratio alone cannot be the whole story.

    The one structural difference between the two glyphs: Glyph A is a much longer single line at
    the SAME height (both lat_h=7, since height depends on font metrics/line count, not word
    count) -- i.e. Glyph A has a more extreme SELF aspect ratio (lat_w / lat_h). This script tests
    directly whether the glyph's OWN aspect ratio (independent of canvas) is a second, separate
    failure mode:

    Section 1 (aspect-ratio isolation): renders Glyph A's EXACT text at target_lines in [1,2,3,4]
        (natural aspect ratios 4.43 / 1.19 / 0.71 / 0.34), all placed on the SAME oversized,
        deliberately-safe canvas (1424x1024, giving canvas-ratio <=0.35 for every variant) so
        canvas-ratio cannot be the confound -- if 1-line still fails while 2/3/4-line pass, the
        glyph's own aspect ratio (or, equivalently, absolute latent width at a fixed height) is
        an independent necessary condition, and the fix (force more line breaks for long single
        lines) is doubly justified.

    Section 2 (canvas-ratio fine sweep): Round 2 bracketed the canvas-ratio threshold between
        0.55 (PASS) and 0.7 (FAIL) using Glyph B. This section fills in 0.50 / 0.60 / 0.65 on the
        SAME fixed Glyph B bitmap to narrow that boundary before finalizing
        `max_line_width_ratio` in glyph_engine.py.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 ONLY.

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_aspect_ratio.py                     # both sections, 7 runs
    python scripts/probe_glyph_aspect_ratio.py --sections aspect   # only Section 1
====================================================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from flux2.autoencoder import AutoEncoder
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.util import load_ae, load_flow_model, load_qwen3_embedder
from tendoo.glyph_engine import GlyphInfo, render_glyph


# ==================================================================================================
# 1. FIXED TEXTS + RUN MATRIX
# ==================================================================================================

TEXT_A = "Đậm đà hương vị cà phê Việt"    # lat_w=31 @ 1 line, lat_h=7 fixed regardless of line count
TEXT_B = "MUA 1 TẶNG 1"                    # lat_w=17 @ 1 line (the already-proven-workable shape)

ASPECT_CANVAS = (1424, 1024)  # oversized synthetic canvas: canvas-ratio <=0.35 for ALL of TEXT_A's
                               # target_lines variants (1/2/3/4), so canvas-ratio cannot confound.

# Section 2: new points only. 0.55 (PASS) and 0.70 (FAIL) already confirmed in probe_glyph_width_ratio.py.
CANVAS_RATIO_FINE_SWEEP = [0.50, 0.60, 0.65]


@dataclass
class RunConfig:
    section: str
    run_id: str
    render_kwargs: Dict[str, Any]
    canvas_wh: Tuple[int, int]
    note: str = ""


def build_run_matrix() -> List[RunConfig]:
    runs: List[RunConfig] = []

    for n_lines in [1, 2, 3, 4]:
        runs.append(RunConfig(
            section="aspect",
            run_id=f"aspect_L{n_lines}",
            render_kwargs=dict(text=TEXT_A, font_name_or_path="bevietnam", target_lines=n_lines, auto_size=True),
            canvas_wh=ASPECT_CANVAS,
            note=f"{n_lines}-line forced wrap of Glyph A text, canvas-ratio held <=0.35 throughout",
        ))

    for ratio in CANVAS_RATIO_FINE_SWEEP:
        runs.append(RunConfig(
            section="canvas_ratio_fine",
            run_id=f"canvasratio_r{ratio:.2f}".replace(".", "p"),
            render_kwargs=dict(text=TEXT_B, font_name_or_path="bevietnam", force_single_line=True, auto_size=True),
            canvas_wh=None,  # filled in at runtime from the glyph's actual lat_w
            note=f"Glyph B fixed shape, target canvas-ratio={ratio:.2f}",
        ))

    return runs


def canvas_width_for_ratio(glyph_lat_w: int, ratio: float) -> int:
    canvas_lat_w = max(16, round(glyph_lat_w / ratio))
    return canvas_lat_w * 16


# ==================================================================================================
# 2. 4D ROPE ENCODING (isolated single-glyph @ t=10.0)
# ==================================================================================================

def encode_glyph_to_ref_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    arr = np.array(glyph_img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
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


# ==================================================================================================
# 3. MAIN SWEEP RUNNER
# ==================================================================================================

DEFAULT_PROMPT = (
    "Poster quảng cáo cà phê phong cách hiện đại tối giản, ánh sáng studio ấm áp tương phản cao, "
    "dòng chữ 3D mạ vàng đồng cổ sắc nét nổi bật, bố cục sạch sẽ gọn gàng, "
    "không có chữ ký, không có watermark, không có chữ trang trí thừa"
)


def run_probe(
    sections: List[str],
    prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_aspect_ratio",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = [r for r in build_run_matrix() if r.section in sections]
    if not all_runs:
        print(f"[ERROR] No runs match requested sections: {sections}")
        return

    print("=" * 100)
    print(" [*] TENDOO AI - GLYPH SELF-ASPECT-RATIO PROBE (ROUND 3)")
    print("=" * 100)
    print(f"  Sections   : {sections}")
    print(f"  Total runs : {len(all_runs)}")

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print("[*] Dual GPU Mode: DiT on GPU 0 | VAE & Qwen3 on GPU 1")
    else:
        device_dit = device_ae = device_te = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[*] Single Device Mode: {device_dit}")

    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    print("\n[1/3] Loading FLUX.2 Klein 4B Base (AE + DiT + Qwen3)...")
    ae = load_ae(model_name, device=device_ae)
    model = load_flow_model(model_name, device=device_dit)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    print("[2/3] Encoding shared prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    if num_gpus >= 2:
        try:
            text_encoder.model.to("cpu")
        except Exception:
            pass
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[3/3] Executing {len(all_runs)} run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        print("-" * 100)
        print(f"▶️  [{idx}/{len(all_runs)}] [{run.section}] {run.run_id}: {run.note}")
        t_run_start = time.time()

        glyph_info = render_glyph(**run.render_kwargs)
        glyph_path = out_path / f"{run.run_id}_glyph.png"
        glyph_info.image.save(glyph_path)
        aspect = glyph_info.latent_w / glyph_info.latent_h

        if run.canvas_wh is not None:
            canvas_w, canvas_h = run.canvas_wh
        else:
            # Section 2: derive canvas width from the glyph's actual rendered lat_w to hit the
            # exact target ratio embedded in the run_id (e.g. "canvasratio_r0p50" -> 0.50).
            target_ratio = float(run.run_id.split("_r")[-1].replace("p", "."))
            canvas_w = canvas_width_for_ratio(glyph_info.latent_w, target_ratio)
            canvas_h = 1024

        canvas_w = (canvas_w // 16) * 16
        canvas_h = (canvas_h // 16) * 16
        lat_w, lat_h = canvas_w // 16, canvas_h // 16
        canvas_ratio = glyph_info.latent_w / lat_w

        print(f"    Glyph  : {glyph_info.width_px}x{glyph_info.height_px}px lat={glyph_info.latent_w}x{glyph_info.latent_h} "
              f"aspect={aspect:.2f} -> {glyph_info.lines}")
        print(f"    Canvas : {canvas_w}x{canvas_h} (canvas_ratio={canvas_ratio:.3f})")

        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae=ae, glyph_img=glyph_info.image, t_offset=10.0, device=device_ae)
        ref_tokens = ref_tokens.to(device_dit)
        ref_ids = ref_ids.to(device_dit)

        torch.manual_seed(seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

        with torch.no_grad():
            out_latent = denoise_cfg(
                model=model, img=img_tokens, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                timesteps=timesteps, guidance=guidance,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )

        torch.cuda.empty_cache()
        out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
        out_latent = out_latent.to(device=device_ae, dtype=torch.bfloat16)
        with torch.no_grad():
            out_tensor = ae.decode(out_latent)

        out_tensor = torch.clamp((out_tensor[0] + 1.0) / 2.0, min=0.0, max=1.0)
        out_arr = (out_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        result_img = Image.fromarray(out_arr)
        result_path = out_path / f"{run.run_id}_result.png"
        result_img.save(result_path)

        elapsed = time.time() - t_run_start
        manifest.append({
            "section": run.section, "run_id": run.run_id, "note": run.note,
            "text": run.render_kwargs.get("text", ""),
            "glyph_lat_w": glyph_info.latent_w, "glyph_lat_h": glyph_info.latent_h,
            "glyph_aspect": round(aspect, 3), "glyph_lines": glyph_info.lines,
            "canvas": f"{canvas_w}x{canvas_h}", "canvas_ratio": round(canvas_ratio, 3),
            "elapsed_s": round(elapsed, 2), "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "aspect_ratio_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] SECTION 1 - GLYPH SELF-ASPECT-RATIO ISOLATION (canvas_ratio held <=0.35)")
    print("=" * 100)
    print(f"{'Run ID':<16} | {'Glyph lat':<10} | {'Aspect':<7} | {'CanvasRatio':<12} | {'Verdict'}")
    print("-" * 100)
    for r in manifest:
        if r["section"] == "aspect":
            print(f"{r['run_id']:<16} | {r['glyph_lat_w']}x{r['glyph_lat_h']:<7} | {r['glyph_aspect']:<7.2f} | "
                  f"{r['canvas_ratio']:<12.3f} | ? PASS/FAIL")
    print("=" * 100)

    print("\n" + "=" * 100)
    print(" [*] SECTION 2 - CANVAS-RATIO FINE SWEEP (Glyph B fixed shape, aspect~2.43 held constant)")
    print("=" * 100)
    print("  (already known from Round 2: ratio=0.55 -> PASS, ratio=0.70 -> FAIL)")
    print(f"{'Run ID':<20} | {'CanvasRatio':<12} | {'Verdict'}")
    print("-" * 100)
    for r in manifest:
        if r["section"] == "canvas_ratio_fine":
            print(f"{r['run_id']:<20} | {r['canvas_ratio']:<12.3f} | ? PASS/FAIL")
    print("=" * 100)

    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP]")
    print("  Section 1: if aspect_L1 (aspect 4.43, canvas_ratio~0.35) FAILS while aspect_L2/L3/L4")
    print("             (aspect 1.19/0.71/0.34) PASS, this confirms the glyph's OWN aspect ratio")
    print("             (not just canvas-ratio) is an independent failure mode -- an extremely")
    print("             elongated single-line glyph is unsafe regardless of destination canvas.")
    print("  Section 2: combined with 0.55(PASS)/0.70(FAIL) from Round 2, find the exact crossover")
    print("             among 0.50/0.55/0.60/0.65/0.70 to set the final max_line_width_ratio.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Glyph Self-Aspect-Ratio Probe (Round 3)")
    parser.add_argument("--sections", type=str, nargs="+", default=["aspect", "canvas_ratio_fine"],
                         choices=["aspect", "canvas_ratio_fine"], help="Which sections to run (default: both)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt")
    parser.add_argument("--output_dir", type=str, default="output_glyph_aspect_ratio", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    run_probe(
        sections=args.sections, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance, seed=args.seed,
    )


if __name__ == "__main__":
    main()
