#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - GLYPH ENGINE LOCK PROBE (MIN TOKEN / MIN FONT SIZE / CANVAS-AWARE WRAPPING)
====================================================================================================
Script: scripts/probe_glyph_engine_lock.py
Purpose:
    Empirically settle the open questions in `src/tendoo/glyph_engine.py` (Rule 29 block) BEFORE
    the module is declared "locked", using REAL GPU inference on FLUX.2 Klein 4B Base (isolated
    single-glyph @ t=10.0 -- the ONE configuration proven 100% reliable, so any artifact observed
    here is attributable to the glyph engine's sizing choices, not to multi-slot crosstalk).

    Four independent experiment sections (run via `--sections` to select a subset):

    A) font_floor   - Refines the per-font min font-size floor (currently locked at 32pt for
                       bevietnam) by sweeping sizes BELOW and AT the current floor on a short text.
    B) padding      - Tests whether `safety_padding_px` can be tightened below the current 16px
                       default without introducing artifacts (more token savings if safe).
    C) min_height   - Resolves the discrepancy between AGENTS.md rule 4 (>=160px / >=10 latent
                       tokens height floor) and the current glyph_engine.py code (112px for a
                       single line) by sweeping candidate line-height floors directly.
    D) aspect_ratio - THE CORE EXPERIMENT: validates the Rule 29 hypothesis that FLUX.2 Base DiT
                       preserves a glyph's LINE COUNT near-verbatim regardless of the destination
                       canvas, so a long line auto-wrapped WITHOUT knowledge of the target canvas
                       breaks when forced onto a narrow 9:16 poster. Compares OLD (isolated,
                       canvas-agnostic) vs NEW (Rule 29 canvas-aware) wrapping on the SAME long
                       text, rendered onto a narrow 9:16 canvas, plus a wide 16:9 control.

Strict Rule Adherence (AGENTS.md Rule 28):
    - ZERO HTML output. Clean ASCII tables in Terminal + JSON manifest + PNG images only.
    - Hardware Target: 2x NVIDIA A30 (DiT on GPU 0, VAE/Qwen3 on GPU 1). Falls back to single GPU.
    - Model: FLUX.2-klein-base-4B ONLY (AGENTS.md Sec. 4). Isolated single-glyph @ t=10.0 only --
      this probe is NOT testing multi-slot crosstalk, only the glyph engine's own sizing choices.

Usage on Remote Server (2x A30):
    python scripts/probe_glyph_engine_lock.py                       # all 4 sections, ~15 runs
    python scripts/probe_glyph_engine_lock.py --sections aspect_ratio   # only the core Rule 29 test
    python scripts/probe_glyph_engine_lock.py --font playfair        # re-run methodology on another font
====================================================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root wiring
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
from tendoo.glyph_engine import GlyphInfo, render_glyph, resolve_font_path


# ==================================================================================================
# 1. TEXT CORPUS (short / medium / long-single-line / long-multiline)
# ==================================================================================================

TEXT_SHORT = "MUA 1 TẶNG 1"                                                           # 3 words
TEXT_MEDIUM = "Đậm đà hương vị cà phê Việt"                                           # 6 words
TEXT_LONG_1LINE = "Đậm đà hương vị cà phê Việt Nam pha phin truyền thống mỗi sáng"    # 14 words
TEXT_LONG_MULTILINE = (
    "Sông Mã xa rồi Tây Tiến ơi\n"
    "Nhớ về rừng núi nhớ chơi vơi.\n"
    "Sài Khao sương lấp đoàn quân mỏi,\n"
    "Mường Lát hoa về trong đêm hơi."
)

CANVAS_1x1 = (1024, 1024)
CANVAS_9x16 = (576, 1024)
CANVAS_16x9 = (1024, 576)


@dataclass
class RunConfig:
    section: str
    run_id: str
    title: str
    text: str
    render_kwargs: Dict[str, Any] = field(default_factory=dict)
    canvas_wh: Tuple[int, int] = CANVAS_9x16
    prompt_hint: str = ""


def build_run_matrix(font: str) -> List[RunConfig]:
    runs: List[RunConfig] = []

    # --- Section A: Font-size floor refinement (below and at the current locked floor) ---
    for pt in [20, 24, 28, 32]:
        runs.append(RunConfig(
            section="font_floor",
            run_id=f"A_floor_{pt}pt",
            title=f"Font floor sweep @ {pt}pt",
            text=TEXT_SHORT,
            render_kwargs=dict(font_name_or_path=font, font_size_pt=pt, force_single_line=True, auto_size=True),
            canvas_wh=CANVAS_9x16,
        ))

    # --- Section B: Safety padding tightness (at the currently-locked floor) ---
    for pad in [8, 16]:
        runs.append(RunConfig(
            section="padding",
            run_id=f"B_pad_{pad}px",
            title=f"Safety padding sweep @ {pad}px",
            text=TEXT_SHORT,
            render_kwargs=dict(font_name_or_path=font, force_single_line=True, auto_size=True, safety_padding_px=pad),
            canvas_wh=CANVAS_9x16,
        ))

    # --- Section C: Minimum line-height floor (AGENTS.md rule 4 [>=160px] vs code [112px]) ---
    for min_h in [96, 112, 128, 160]:
        runs.append(RunConfig(
            section="min_height",
            run_id=f"C_minh_{min_h}px",
            title=f"Min line-height floor @ {min_h}px (1 line)",
            text=TEXT_MEDIUM,
            render_kwargs=dict(font_name_or_path=font, force_single_line=True, auto_size=True),
            canvas_wh=CANVAS_9x16,
        ))
        # Patch the height floor directly via compute kwargs (see run_single below for wiring)
        runs[-1].render_kwargs["_min_line_height_single_px_override"] = min_h

    # --- Section D: THE CORE TEST -- Rule 29 canvas-aware wrapping vs legacy isolated wrapping ---
    runs.append(RunConfig(
        section="aspect_ratio",
        run_id="D_long1line_OLD_narrow9x16",
        title="[OLD/isolated] long 1-line glyph forced onto narrow 9:16 canvas",
        text=TEXT_LONG_1LINE,
        render_kwargs=dict(font_name_or_path=font, force_single_line=True, auto_size=True),
        canvas_wh=CANVAS_9x16,
    ))
    runs.append(RunConfig(
        section="aspect_ratio",
        run_id="D_long1line_NEW_narrow9x16",
        title="[NEW/Rule29] same text, canvas-aware wrapping, narrow 9:16 canvas",
        text=TEXT_LONG_1LINE,
        render_kwargs=dict(font_name_or_path=font, auto_size=True, target_canvas_w=CANVAS_9x16[0], target_canvas_h=CANVAS_9x16[1]),
        canvas_wh=CANVAS_9x16,
    ))
    runs.append(RunConfig(
        section="aspect_ratio",
        run_id="D_long1line_OLD_wide16x9_control",
        title="[control] same text, isolated wrapping, WIDE 16:9 canvas (expected to already be fine)",
        text=TEXT_LONG_1LINE,
        render_kwargs=dict(font_name_or_path=font, force_single_line=True, auto_size=True),
        canvas_wh=CANVAS_16x9,
    ))
    runs.append(RunConfig(
        section="aspect_ratio",
        run_id="D_poem_narrow9x16",
        title="[explicit multiline] 4-line poem (user \\n, wrap-agnostic) on narrow 9:16 canvas",
        text=TEXT_LONG_MULTILINE,
        render_kwargs=dict(font_name_or_path=font, auto_size=True),
        canvas_wh=CANVAS_9x16,
    ))
    runs.append(RunConfig(
        section="aspect_ratio",
        run_id="D_poem_wide16x9",
        title="[explicit multiline] same poem on WIDE 16:9 canvas",
        text=TEXT_LONG_MULTILINE,
        render_kwargs=dict(font_name_or_path=font, auto_size=True),
        canvas_wh=CANVAS_16x9,
    ))

    return runs


# ==================================================================================================
# 2. GLYPH RENDER + 4D ROPE ENCODING (isolated single-glyph @ t=10.0)
# ==================================================================================================

def _render_raw_glyph_ignoring_floor(
    text: str,
    font_name_or_path: str,
    font_size_pt: int,
    padding_px: int = 16,
) -> GlyphInfo:
    """
    Renders a single-line glyph at an EXACT font size, deliberately bypassing
    `compute_optimal_glyph_box`'s floor auto-elevation. Needed ONLY by Section A: the whole
    point of that experiment is to test sizes BELOW the currently-locked floor, which the
    production `render_glyph()` correctly (by design) refuses to do.
    """
    from PIL import ImageDraw, ImageFont
    import math as _math

    _, font_path, meta = resolve_font_path(font_name_or_path)
    try:
        font = ImageFont.truetype(font_path, size=font_size_pt)
    except Exception:
        font = ImageFont.load_default()

    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    raw_w, raw_h = tw + 2 * padding_px, th + 2 * padding_px
    box_w = max(16, int(_math.ceil(raw_w / 16.0) * 16))
    box_h = max(16, int(_math.ceil(max(raw_h, 80) / 16.0) * 16))

    img = Image.new("RGB", (box_w, box_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = (box_w - tw) // 2 - bbox[0]
    y = (box_h - th) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return GlyphInfo(
        image=img, text=text, lines=[text], font_name=font_name_or_path, font_path=font_path,
        font_size_pt=font_size_pt, width_px=box_w, height_px=box_h,
        latent_w=box_w // 16, latent_h=box_h // 16, token_count=(box_w // 16) * (box_h // 16),
        archetype=meta["archetype"], tier=meta["tier"], min_floor_pt=meta["min_floor_pt"],
        is_nyquist_safe=font_size_pt >= meta["min_floor_pt"],
        line_spacing_px=0, padding_x_px=padding_px, padding_y_px=padding_px,
    )


def render_glyph_for_run(run: RunConfig) -> GlyphInfo:
    """Renders the glyph for a run, honoring the min-line-height override used by Section C."""
    kwargs = dict(run.render_kwargs)
    height_override = kwargs.pop("_min_line_height_single_px_override", None)

    if run.section == "font_floor":
        # Deliberately bypass the locked floor to probe below it (see docstring above).
        return _render_raw_glyph_ignoring_floor(
            text=run.text,
            font_name_or_path=kwargs.get("font_name_or_path", "bevietnam"),
            font_size_pt=kwargs.get("font_size_pt", 32),
        )

    if height_override is not None:
        # compute_optimal_glyph_box exposes this directly; render_glyph (Mode A) does not, so we
        # call the box computation once ourselves and force-render at that exact box via Mode B.
        from tendoo.glyph_engine import compute_optimal_glyph_box
        box_w, box_h, chosen_pt, _lines = compute_optimal_glyph_box(
            text=run.text,
            font_name_or_path=kwargs.get("font_name_or_path", "bevietnam"),
            force_single_line=kwargs.get("force_single_line", False),
            min_line_height_single_px=height_override,
            min_line_height_multi_px=height_override,
        )
        return render_glyph(
            text=run.text,
            font_name_or_path=kwargs.get("font_name_or_path", "bevietnam"),
            target_width=box_w,
            target_height=box_h,
            font_size_pt=chosen_pt,
            force_single_line=kwargs.get("force_single_line", False),
            auto_size=False,  # MUST be explicit: render_glyph defaults auto_size=True, which
                               # silently ignores target_width/height (Mode A) and would discard
                               # the min-line-height override this branch exists to apply.
        )

    return render_glyph(text=run.text, **kwargs)


def encode_glyph_to_ref_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encodes a glyph bitmap into 4D RoPE In-Context reference tokens (t, h, w, l)."""
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
    font: str = "bevietnam",
    prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_glyph_lock",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: Optional[str] = None,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = 42,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = [r for r in build_run_matrix(font) if r.section in sections]
    if not all_runs:
        print(f"[ERROR] No runs match requested sections: {sections}")
        return

    print("=" * 100)
    print(" [*] TENDOO AI - GLYPH ENGINE LOCK PROBE")
    print("=" * 100)
    print(f"  Font              : {font}")
    print(f"  Sections          : {sections}")
    print(f"  Total runs        : {len(all_runs)}")
    print(f"  Model             : {model_name} (isolated single-glyph @ t=10.0 ONLY)")
    print("=" * 100)

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

    if num_gpus < 2:
        del text_encoder
    else:
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
        print(f"▶️  [{idx}/{len(all_runs)}] [{run.section}] {run.run_id}: {run.title}")

        t_run_start = time.time()

        glyph_info = render_glyph_for_run(run)
        glyph_path = out_path / f"{run.run_id}_glyph.png"
        glyph_info.image.save(glyph_path)

        canvas_w, canvas_h = run.canvas_wh
        canvas_w = (canvas_w // 16) * 16
        canvas_h = (canvas_h // 16) * 16
        lat_w, lat_h = canvas_w // 16, canvas_h // 16

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
                model=model,
                img=img_tokens,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
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
        record = {
            "section": run.section,
            "run_id": run.run_id,
            "title": run.title,
            "text": run.text.replace("\n", "\\n"),
            "font": font,
            "canvas": f"{canvas_w}x{canvas_h}",
            "glyph_px": f"{glyph_info.width_px}x{glyph_info.height_px}",
            "glyph_font_size_pt": glyph_info.font_size_pt,
            "glyph_lines": glyph_info.lines,
            "glyph_num_lines": len(glyph_info.lines),
            "glyph_tokens": glyph_info.token_count,
            "elapsed_s": round(elapsed, 2),
            "glyph_file": glyph_path.name,
            "result_file": result_path.name,
        }
        manifest.append(record)

        print(f"    Glyph  : {glyph_info.width_px}x{glyph_info.height_px}px, {glyph_info.font_size_pt}pt, "
              f"{len(glyph_info.lines)}L, {glyph_info.token_count} tokens -> {glyph_info.lines}")
        print(f"    Result : {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "glyph_lock_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _print_summary_tables(manifest)
    print(f"\n[+] All images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in   : {manifest_path.resolve()}")
    print("\n[NEXT STEP] Open each *_result.png in JupyterLab's image viewer and judge PASS/FAIL:")
    print("  - Section A/B/C: does the text render clean (no spiky/jagged diacritics)?")
    print("  - Section D    : does D_long1line_NEW_narrow9x16 render CLEAN text while")
    print("                   D_long1line_OLD_narrow9x16 shows squeezed/broken text (Rule 29 hypothesis)?")
    print("                   D_long1line_OLD_wide16x9_control should look clean (sanity control).\n")


def _print_summary_tables(manifest: List[Dict[str, Any]]) -> None:
    by_section: Dict[str, List[Dict[str, Any]]] = {}
    for r in manifest:
        by_section.setdefault(r["section"], []).append(r)

    section_titles = {
        "font_floor": "SECTION A - FONT-SIZE FLOOR SWEEP",
        "padding": "SECTION B - SAFETY PADDING SWEEP",
        "min_height": "SECTION C - MIN LINE-HEIGHT FLOOR SWEEP",
        "aspect_ratio": "SECTION D - CANVAS-AWARE WRAPPING (RULE 29 CORE TEST)",
    }

    for section, rows in by_section.items():
        print("\n" + "=" * 100)
        print(f" [*] {section_titles.get(section, section.upper())}")
        print("=" * 100)
        header = f"{'Run ID':<32} | {'Canvas':<10} | {'Glyph px':<12} | {'Pt':<5} | {'Lines':<6} | {'Tokens':<7} | {'Time':<6}"
        print(header)
        print("-" * 100)
        for r in rows:
            print(
                f"{r['run_id']:<32} | {r['canvas']:<10} | {r['glyph_px']:<12} | {r['glyph_font_size_pt']}pt{'':<2} | "
                f"{r['glyph_num_lines']:<6} | {r['glyph_tokens']:<7} | {r['elapsed_s']}s"
            )
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Glyph Engine Lock Probe")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias to run the methodology on (default: bevietnam)")
    parser.add_argument(
        "--sections", type=str, nargs="+",
        default=["font_floor", "padding", "min_height", "aspect_ratio"],
        choices=["font_floor", "padding", "min_height", "aspect_ratio"],
        help="Which experiment sections to run (default: all 4)",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt (material/optics only, no literal text)")
    parser.add_argument("--output_dir", type=str, default="output_glyph_lock", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    run_probe(
        sections=args.sections,
        font=args.font,
        prompt=args.prompt,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
