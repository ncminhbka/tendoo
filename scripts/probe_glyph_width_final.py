#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - GLYPH BOX WIDTH: FINAL DECISIVE SWEEP (CLEAN PROMPT, NO CONFOUNDS)
====================================================================================================
Script: scripts/probe_glyph_width_final.py
Purpose:
    AGENTS.md Rule 31 (corrected): the ORIGINAL "isolated_subtitle fails" result was caused by
    TWO real confounds -- the tight-cropped 512x256 envelope (vs the historical 512x224) AND the
    prompt's "chữ phụ" (subordinate-role) wording confusing Qwen3 when no title co-exists. The
    neon-bloom-eats-fine-strokes theory from an earlier draft was WRONG (the model handles any
    material/effect fine) and has been retracted.

    With BOTH real confounds now removed (clean envelope, clean prompt with no role-subordination
    wording), THREE questions remain genuinely open and are answered here in one pass:
      1. Is box_width_px >= 512 really necessary, or did the "chữ phụ" prompt confound make
         everything below 512 look worse than it actually is? Sweep width 384 -> 1024 with the
         CLEAN prompt to find out.
      2. Should glyph width be allowed to exceed the destination canvas width (576px)? By how
         much? The sweep includes points both below and well above the 576px canvas.
      3. Can a genuinely LONG, single-line (un-wrapped) text still render correctly when forced
         onto the real 9:16 canvas -- where the resulting text will appear visually SMALL in the
         final composition? A separate config tests this directly (force_single_line=True on a
         14-word sentence, width left generous/unconstrained so only height bounds the font).

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 ONLY. Canvas: real 576x1024 (9:16).

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_width_final.py                       # width sweep + long-line, ~36 runs
    python scripts/probe_glyph_width_final.py --configs long_line    # only the long-line question
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
# 1. CONFIGS
# ==================================================================================================

FONT = "bevietnam"
CANVAS = (576, 1024)  # real 9:16 primary target
DEFAULT_SEEDS = [42, 123, 777]

# Width sweep: SAME text/height as the just-validated 512x224 ("BỨT PHÁ MỌI"/"GIỚI HẠN", 69pt,
# "hoàn hảo" per the user). Only width varies -- font is re-binary-searched at each width, height
# fixed at 224px (2 lines) throughout so it never confounds with the width question.
SWEEP_TEXT_LINES_HEIGHT = ("BỨT PHÁ MỌI\nGIỚI HẠN", 224)
WIDTH_SWEEP = [384, 448, 512, 640, 768, 896, 1024]  # 576 = canvas width, for reference

# Long single-line question: a genuinely long sentence, forced into ONE line (no wrapping), width
# left generous (2400px, far beyond what's needed) so ONLY height (128px, 1 line) bounds the font
# -- i.e. this is the glyph's natural, width-unconstrained single-line shape, then composited as-is
# onto the real 576px-wide canvas (so the glyph is far wider than the canvas and the resulting
# text will appear small in the final poster).
LONG_LINE_TEXT = "Đậm đà hương vị cà phê Việt Nam pha phin truyền thống mỗi sáng"

# Clean prompt: no role-subordination wording ("chữ phụ"), no assumption about material/effect
# (Rule 31 correction: material does not matter) -- kept simple and neutral throughout.
DEFAULT_PROMPT = (
    "Poster quảng cáo phong cách hiện đại tối giản, ánh sáng studio ấm áp tương phản cao, "
    "dòng chữ sắc nét nổi bật, bố cục sạch sẽ gọn gàng, không có chữ ký, không có watermark"
)


@dataclass
class RunConfig:
    config_key: str
    seed: int
    run_id: str


def build_run_matrix(config_keys: List[str], seeds: List[int]) -> List[RunConfig]:
    return [RunConfig(config_key=k, seed=s, run_id=f"{k}_seed{s}") for k in config_keys for s in seeds]


def render_configs(font: str) -> Dict[str, GlyphInfo]:
    infos: Dict[str, GlyphInfo] = {}
    text, height = SWEEP_TEXT_LINES_HEIGHT
    for w in WIDTH_SWEEP:
        info = render_glyph(text=text, font_name_or_path=font, auto_size=False, target_width=w, target_height=height)
        infos[f"width_{w}"] = info
    infos["long_line"] = render_glyph(
        text=LONG_LINE_TEXT, font_name_or_path=font, auto_size=False, target_width=2400, target_height=128,
    )
    return infos


# ==================================================================================================
# 2. 4D ROPE ENCODING (isolated single-glyph @ t=10.0)
# ==================================================================================================

def encode_glyph_to_ref_tokens(
    ae: AutoEncoder, glyph_img: Image.Image, t_offset: float, device: str | torch.device,
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
# 3. MAIN RUNNER
# ==================================================================================================

def run_probe(
    config_keys: List[str], seeds: List[int], prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_width_final", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(config_keys, seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - GLYPH BOX WIDTH: FINAL DECISIVE SWEEP")
    print("=" * 100)
    print(f"  Configs  : {config_keys}")
    print(f"  Seeds    : {seeds}")
    print(f"  Canvas   : {CANVAS[0]}x{CANVAS[1]} (real 9:16, canvas latent width = {CANVAS[0]//16})")
    print(f"  Total runs: {len(all_runs)}")

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

    print("[2/3] Encoding shared clean prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt, txt_ids = batched_prc_txt(text_encoder(["", prompt]))
        txt, txt_ids = txt.to(device_dit), txt_ids.to(device_dit)

    if num_gpus >= 2:
        try:
            text_encoder.model.to("cpu")
        except Exception:
            pass
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    canvas_w, canvas_h = CANVAS
    canvas_w, canvas_h = (canvas_w // 16) * 16, (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    all_glyphs = render_configs(FONT)
    glyph_infos = {k: v for k, v in all_glyphs.items() if k in config_keys}
    for key, info in glyph_infos.items():
        canvas_ratio = info.latent_w / lat_w
        print(f"  [{key:10s}] {info.width_px}x{info.height_px}px {info.font_size_pt}pt {len(info.lines)}L "
              f"{info.token_count}tok canvas_width_ratio={canvas_ratio:.2f} :: {info.lines}")
        info.image.save(out_path / f"{key}_glyph.png")

    print(f"\n[3/3] Executing {len(all_runs)} denoise run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        glyph_info = glyph_infos[run.config_key]

        print("-" * 100)
        print(f"▶️  [{idx}/{len(all_runs)}] {run.run_id}")
        t_run_start = time.time()

        ref_tokens, ref_ids = encode_glyph_to_ref_tokens(ae=ae, glyph_img=glyph_info.image, t_offset=10.0, device=device_ae)
        ref_tokens, ref_ids = ref_tokens.to(device_dit), ref_ids.to(device_dit)

        torch.manual_seed(run.seed)
        z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
        img_tokens, img_ids = prc_img(z_init[0])
        img_tokens = img_tokens.unsqueeze(0).to(device_dit)
        img_ids = img_ids.unsqueeze(0).to(device_dit)

        timesteps = get_schedule(num_steps=num_steps, image_seq_len=img_tokens.shape[1])

        with torch.no_grad():
            out_latent = denoise_cfg(
                model=model, img=img_tokens, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                timesteps=timesteps, guidance=guidance, img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
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
        ratio = glyph_info.latent_w / lat_w
        manifest.append({
            "run_id": run.run_id, "config_key": run.config_key, "seed": run.seed,
            "glyph_px": f"{glyph_info.width_px}x{glyph_info.height_px}", "font_pt": glyph_info.font_size_pt,
            "tokens": glyph_info.token_count, "canvas_ratio": round(ratio, 3),
            "elapsed_s": round(elapsed, 2), "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "width_final_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] SUMMARY (fill in Verdict, then tally X/N per config)")
    print("=" * 100)
    print(f"{'Run ID':<24} | {'Config':<12} | {'Box':<10} | {'Pt':<4} | {'CanvasRatio':<11} | {'Seed':<6} | Verdict")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<24} | {r['config_key']:<12} | {r['glyph_px']:<10} | {r['font_pt']:<4} | "
              f"{r['canvas_ratio']:<11.2f} | {r['seed']:<6} | ? PASS/FAIL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP]")
    print("  1. Width sweep: for each width_XXX config, tally X/3. Find the LOWEST width that's")
    print("     still reliable (tests whether 512 is a real threshold or was a prompt artifact)")
    print("     and check whether reliability holds, degrades, or improves as width keeps growing")
    print("     past the 576px canvas width (ratios >1.0).")
    print("  2. long_line: does this genuinely long, un-wrapped, visually-small-on-canvas text")
    print("     still render correctly? This directly answers whether forcing long text onto a")
    print("     narrow 9:16 output without line-wrapping is viable at all.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Glyph Box Width Final Decisive Sweep")
    parser.add_argument("--configs", type=str, nargs="+",
                         default=[f"width_{w}" for w in WIDTH_SWEEP] + ["long_line"],
                         choices=[f"width_{w}" for w in WIDTH_SWEEP] + ["long_line"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", type=str, default="output_glyph_width_final")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)

    args = parser.parse_args()
    run_probe(
        config_keys=args.configs, seeds=args.seeds, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
