#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - GLYPH ABSOLUTE SCALE PROBE (PRETRAINING-DISTRIBUTION-MATCH HYPOTHESIS)
====================================================================================================
Script: scripts/probe_glyph_absolute_scale.py
Purpose:
    probe_old_vs_new_glyph_recipe.py: the OLD sizing recipe (test_tiktok_poster.py's fixed
    512x224 envelope, binary-search-largest-font, NOT tight-cropped, 448 tokens) beat the NEW
    tight-crop/aspect-band recipe (320 tokens) 3/5 vs 1/5 on "BỨT PHÁ MỌI GIỚI HẠN". Notably, OLD's
    own glyph-to-canvas width ratio (512/576=0.89) EXCEEDS the "unsafe >=0.86" threshold Rounds
    1-2 established -- so that theory doesn't explain OLD's win either. And OLD's aspect ratio
    (2.29) sits well outside the [0.5, 1.3] "safe" self-aspect-ratio band Rounds 3+6 established.

    New hypothesis (raised by the user): if the model's t=10 "preserve the reference almost
    verbatim" behavior comes from heavy pretraining exposure to REAL PHOTOS as references, then a
    reference's ABSOLUTE SIZE / TOKEN COUNT matching that photo-like scale may matter more than any
    of the RATIOS (to canvas, or self-aspect) tested so far -- a very small, tightly-cropped glyph
    (Rule 25's original token-minimization goal) may simply be too far outside the size range the
    model has ever seen as a "reference image" at t=10, regardless of its shape. This is
    consistent with AGENTS.md's already-documented "Token Mass Dominance" law (a weak CTA badge
    auto-recovers once scaled to >=672 tokens) and with the Tây Tiến 100%-accurate reference using
    1792 tokens.

    This probe holds ASPECT RATIO roughly constant (~2.3, matching OLD's just-proven-better shape)
    and varies ONLY absolute scale (token count) across 3 steps, reusing the OLD non-tight-crop /
    binary-search-largest-font mechanism at each size, to test whether reliability increases
    monotonically with scale independent of any ratio:
        S (=OLD)  : 512x224px,  448 tokens  (glyph/canvas width ratio 0.89 -- already "unsafe"
                    under the old canvas-ratio theory, yet this is the one that just won 3/5)
        M         : 704x304px,  836 tokens  (ratio 1.22 -- glyph is WIDER than the whole canvas)
        L         : 896x384px, 1344 tokens  (ratio 1.56 -- even more so; approaching Tây Tiến's
                    1792-token scale)
    5 seeds each. Canvas stays the real 576x1024 (9:16) throughout -- only the glyph's own
    absolute size changes.

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B, isolated single-glyph @ t=10.0 ONLY.

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_absolute_scale.py                # S/M/L x 5 seeds = 15 runs
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
from PIL import Image, ImageDraw, ImageFont

from flux2.autoencoder import AutoEncoder
from flux2.sampling import batched_prc_txt, denoise_cfg, get_schedule, prc_img
from flux2.util import load_ae, load_flow_model, load_qwen3_embedder
from tendoo.glyph_engine import GlyphInfo, resolve_font_path


# ==================================================================================================
# 1. TEXT + SCALE STEPS (aspect ratio held ~constant at ~2.3, matching OLD's just-proven shape)
# ==================================================================================================

TEXT = "BỨT PHÁ MỌI GIỚI HẠN"
FONT = "bevietnam"
CANVAS = (576, 1024)  # real 9:16 primary target
DEFAULT_SEEDS = [42, 123, 777, 2024, 8888]

SCALE_STEPS = {
    "S": (512, 224),  # = the OLD recipe, 448 tokens, glyph/canvas width ratio 0.89
    "M": (704, 304),  # 836 tokens, ratio 1.22 (wider than the whole canvas)
    "L": (896, 384),  # 1344 tokens, ratio 1.56 (approaching Tây Tiến's 1792-token scale)
}

# NOTE (user correction): the historically-reliable recipe was actually "render chữ to + crop
# sát" (big font, THEN tight-crop) -- not "big font, keep the full fixed envelope" (what "S"
# above actually is). This third recipe reuses S's own font-maximizing candidate-layout search
# (61pt, ["BỨT PHÁ","MỌI GIỚI HẠN"]) but then tight-crops to content afterward instead of keeping
# the 512x224 envelope -- same font size as S, but fewer tokens (384 vs 448, at 8% padding).
TIGHT_BIGFONT_SEARCH_ENVELOPE = (512, 224)  # search envelope only -- final box is tight-cropped
TIGHT_BIGFONT_PADDING_RATIO = 0.08


def render_bigfont_tight_crop_glyph(
    text: str, font_name: str, search_envelope_w: int = 512, search_envelope_h: int = 224,
    padding_ratio: float = 0.08,
) -> GlyphInfo:
    """
    "Render chữ to + crop sát": reuses the OLD algorithm's own font-maximizing candidate-layout
    search (find the layout + largest font size that fits within a generous search envelope --
    this is what "render chữ to" means: pick whichever line split lets the font be as big as
    possible) but then TIGHT-CROPS the box down to the actual rendered content afterward, instead
    of keeping the full fixed search envelope. Same font size as the OLD recipe, fewer tokens.
    """
    _, font_path, meta = resolve_font_path(font_name)
    search_w, search_h = (search_envelope_w // 16) * 16, (search_envelope_h // 16) * 16
    pad_w0, pad_h0 = int(search_w * 0.08), int(search_h * 0.08)
    max_w, max_h = search_w - 2 * pad_w0, search_h - 2 * pad_h0

    words = text.split()
    candidate_layouts = []
    if len(words) >= 4:
        mid = len(words) // 2
        candidate_layouts.append([" ".join(words[:mid]), " ".join(words[mid:])])
    if len(words) >= 6:
        p1, p2 = len(words) // 3, 2 * len(words) // 3
        candidate_layouts.append([" ".join(words[:p1]), " ".join(words[p1:p2]), " ".join(words[p2:])])
    candidate_layouts.append([text])

    best_size, best_lines, best_font = 0, None, None
    for lines in candidate_layouts:
        spacing_ratio = 0.32 if len(lines) >= 2 else 0.20
        low, high, opt_size, opt_font = 14, 300, 0, None
        while low <= high:
            mid_size = (low + high) // 2
            try:
                test_font = ImageFont.truetype(font_path, size=mid_size)
            except Exception:
                test_font = ImageFont.load_default()
            total_h, max_line_w = 0, 0
            for line in lines:
                bbox = test_font.getbbox(line)
                max_line_w = max(max_line_w, bbox[2] - bbox[0])
                total_h += bbox[3] - bbox[1]
            total_h += int(mid_size * spacing_ratio) * (len(lines) - 1)
            if max_line_w <= max_w and total_h <= max_h:
                opt_size, opt_font = mid_size, test_font
                low = mid_size + 1
            else:
                high = mid_size - 1
        if opt_size > best_size:
            best_size, best_lines, best_font = opt_size, lines, opt_font

    if best_font is None:
        best_font = ImageFont.truetype(font_path, size=20)
        best_lines, best_size = candidate_layouts[-1], 20

    spacing_ratio = 0.32 if len(best_lines) >= 2 else 0.20
    line_spacing = int(best_size * spacing_ratio)
    line_widths = [best_font.getbbox(l)[2] - best_font.getbbox(l)[0] for l in best_lines]
    line_heights = [best_font.getbbox(l)[3] - best_font.getbbox(l)[1] for l in best_lines]
    content_w = max(line_widths)
    content_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)

    # TIGHT CROP (this is the only difference from render_fixed_envelope_glyph): box sized to
    # the actual content + a real (not wasted) padding margin, not the full search envelope.
    pad_x = max(10, int(content_w * padding_ratio))
    pad_y = max(8, int(content_h * padding_ratio))
    final_w = max(32, ((content_w + 2 * pad_x + 15) // 16) * 16)
    final_h = max(32, ((content_h + 2 * pad_y + 15) // 16) * 16)

    img = Image.new("RGB", (final_w, final_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    curr_y = (final_h - content_h) // 2
    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        bbox = best_font.getbbox(line)
        draw.text(((final_w - lw) // 2 - bbox[0], curr_y - bbox[1]), line, fill=(255, 255, 255), font=best_font)
        curr_y += line_heights[i] + line_spacing

    return GlyphInfo(
        image=img, text=text, lines=best_lines, font_name=font_name, font_path=font_path,
        font_size_pt=best_size, width_px=final_w, height_px=final_h,
        latent_w=final_w // 16, latent_h=final_h // 16, token_count=(final_w // 16) * (final_h // 16),
        archetype=meta["archetype"], tier=meta["tier"], min_floor_pt=meta["min_floor_pt"],
        is_nyquist_safe=best_size >= meta["min_floor_pt"],
        line_spacing_px=line_spacing, padding_x_px=pad_x, padding_y_px=pad_y,
    )

DEFAULT_PROMPT = (
    "Poster tuyển dụng phong cách công ty công nghệ hiện đại, nền gradient xanh dương đậm sang "
    "trọng, dòng chữ phát sáng neon tinh tế, bố cục sạch sẽ chuyên nghiệp, không có chữ ký, "
    "không có watermark"
)


def render_fixed_envelope_glyph(text: str, font_name: str, envelope_w: int, envelope_h: int) -> GlyphInfo:
    """
    Same mechanism as test_tiktok_poster.py / demo_tendoo_poster.py's create_glyph_image: fixed
    envelope (NOT tight-cropped), binary-search the LARGEST font that fits both dimensions.
    """
    _, font_path, meta = resolve_font_path(font_name)
    envelope_w, envelope_h = (envelope_w // 16) * 16, (envelope_h // 16) * 16
    padding_ratio = 0.08
    pad_w, pad_h = int(envelope_w * padding_ratio), int(envelope_h * padding_ratio)
    max_w, max_h = envelope_w - 2 * pad_w, envelope_h - 2 * pad_h

    words = text.split()
    candidate_layouts = []
    if len(words) >= 4:
        mid = len(words) // 2
        candidate_layouts.append([" ".join(words[:mid]), " ".join(words[mid:])])
    if len(words) >= 6:
        p1, p2 = len(words) // 3, 2 * len(words) // 3
        candidate_layouts.append([" ".join(words[:p1]), " ".join(words[p1:p2]), " ".join(words[p2:])])
    candidate_layouts.append([text])

    best_size, best_lines, best_font = 0, None, None
    for lines in candidate_layouts:
        spacing_ratio = 0.32 if len(lines) >= 2 else 0.20
        low, high, opt_size, opt_font = 14, 300, 0, None
        while low <= high:
            mid_size = (low + high) // 2
            try:
                test_font = ImageFont.truetype(font_path, size=mid_size)
            except Exception:
                test_font = ImageFont.load_default()
            total_h, max_line_w = 0, 0
            for line in lines:
                bbox = test_font.getbbox(line)
                max_line_w = max(max_line_w, bbox[2] - bbox[0])
                total_h += bbox[3] - bbox[1]
            total_h += int(mid_size * spacing_ratio) * (len(lines) - 1)
            if max_line_w <= max_w and total_h <= max_h:
                opt_size, opt_font = mid_size, test_font
                low = mid_size + 1
            else:
                high = mid_size - 1
        if opt_size > best_size:
            best_size, best_lines, best_font = opt_size, lines, opt_font

    if best_font is None:
        best_font = ImageFont.truetype(font_path, size=20)
        best_lines, best_size = candidate_layouts[-1], 20

    spacing_ratio = 0.32 if len(best_lines) >= 2 else 0.20
    line_spacing = int(best_size * spacing_ratio)
    line_widths = [best_font.getbbox(l)[2] - best_font.getbbox(l)[0] for l in best_lines]
    line_heights = [best_font.getbbox(l)[3] - best_font.getbbox(l)[1] for l in best_lines]
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)

    img = Image.new("RGB", (envelope_w, envelope_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    curr_y = (envelope_h - total_block_h) // 2
    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        bbox = best_font.getbbox(line)
        draw.text(((envelope_w - lw) // 2 - bbox[0], curr_y - bbox[1]), line, fill=(255, 255, 255), font=best_font)
        curr_y += line_heights[i] + line_spacing

    return GlyphInfo(
        image=img, text=text, lines=best_lines, font_name=font_name, font_path=font_path,
        font_size_pt=best_size, width_px=envelope_w, height_px=envelope_h,
        latent_w=envelope_w // 16, latent_h=envelope_h // 16, token_count=(envelope_w // 16) * (envelope_h // 16),
        archetype=meta["archetype"], tier=meta["tier"], min_floor_pt=meta["min_floor_pt"],
        is_nyquist_safe=best_size >= meta["min_floor_pt"],
        line_spacing_px=line_spacing, padding_x_px=pad_w, padding_y_px=pad_h,
    )


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

@dataclass
class RunConfig:
    scale_key: str
    seed: int
    run_id: str


def build_run_matrix(scale_keys: List[str], seeds: List[int]) -> List[RunConfig]:
    return [RunConfig(scale_key=k, seed=s, run_id=f"{k}_seed{s}") for k in scale_keys for s in seeds]


def run_probe(
    scale_keys: List[str], seeds: List[int], prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_absolute_scale", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(scale_keys, seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - GLYPH ABSOLUTE SCALE PROBE (PRETRAINING-DISTRIBUTION-MATCH HYPOTHESIS)")
    print("=" * 100)
    print(f"  Text     : \"{TEXT}\"  Font: {FONT}")
    print(f"  Scales   : {scale_keys}")
    print(f"  Seeds    : {seeds}")
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

    print("[2/3] Encoding shared prompt via Qwen3-4B-FP8...")
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

    glyph_infos: Dict[str, GlyphInfo] = {}
    for key in scale_keys:
        if key == "T":
            info = render_bigfont_tight_crop_glyph(TEXT, FONT, *TIGHT_BIGFONT_SEARCH_ENVELOPE, TIGHT_BIGFONT_PADDING_RATIO)
        else:
            ew, eh = SCALE_STEPS[key]
            info = render_fixed_envelope_glyph(TEXT, FONT, ew, eh)
        glyph_infos[key] = info
        aspect = info.width_px / info.height_px
        canvas_ratio = info.latent_w / lat_w
        print(f"  [{key}] {info.width_px}x{info.height_px}px {info.font_size_pt}pt {len(info.lines)}L "
              f"{info.token_count}tok aspect={aspect:.2f} canvas_width_ratio={canvas_ratio:.2f} :: {info.lines}")
        info.image.save(out_path / f"scale_{key}_glyph.png")

    print(f"\n[3/3] Executing {len(all_runs)} denoise run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        glyph_info = glyph_infos[run.scale_key]

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
        manifest.append({
            "run_id": run.run_id, "scale_key": run.scale_key, "seed": run.seed,
            "glyph_px": f"{glyph_info.width_px}x{glyph_info.height_px}", "font_pt": glyph_info.font_size_pt,
            "tokens": glyph_info.token_count, "elapsed_s": round(elapsed, 2),
            "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "absolute_scale_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] SUMMARY (fill in Verdict, then tally X/5 per scale step)")
    print("=" * 100)
    print(f"{'Run ID':<16} | {'Scale':<6} | {'Box':<10} | {'Pt':<4} | {'Tokens':<7} | {'Seed':<6} | Verdict")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<16} | {r['scale_key']:<6} | {r['glyph_px']:<10} | {r['font_pt']:<4} | "
              f"{r['tokens']:<7} | {r['seed']:<6} | ? PASS/FAIL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP]")
    print("  - Reliability increases monotonically S -> M -> L -> confirms the pretraining-scale-")
    print("    match hypothesis: absolute token count/size is the dominant lever, independent of")
    print("    (and possibly overriding) both the canvas-width-ratio and self-aspect-ratio")
    print("    theories from earlier rounds -- glyph_engine.py's tight-crop token-minimization")
    print("    (Rule 25) needs to be deprioritized in favor of a minimum absolute-size floor.")
    print("  - No clear trend / L not better than S -> scale alone doesn't explain OLD's win either;")
    print("    the specific font size (61pt) or line split may matter more than raw token count.\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Glyph Absolute Scale Probe")
    parser.add_argument("--scales", type=str, nargs="+", default=list(SCALE_STEPS.keys()) + ["T"],
                         choices=list(SCALE_STEPS.keys()) + ["T"],
                         help="S/M/L: fixed-envelope, not tight-cropped. T: 'render chữ to + crop sát' -- "
                              "same font-maximizing search as S, but tight-cropped afterward (384 tokens).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", type=str, default="output_glyph_absolute_scale")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)

    args = parser.parse_args()
    run_probe(
        scale_keys=args.scales, seeds=args.seeds, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
