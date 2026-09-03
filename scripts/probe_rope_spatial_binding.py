#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - ROPE SPATIAL BINDING PROBE (DIRECTION 1: MULTI-SLOT CROSSTALK, TRAINING-FREE)
====================================================================================================
Script: scripts/probe_rope_spatial_binding.py
Purpose:
    First probe of "Direction 1" (RoPE spatial binding), pursued now that the single-glyph
    rendering layer is LOCKED (glyph_engine.py Rule 29 -- see AGENTS.md). Multi-slot text
    (>=2 simultaneous glyph references) is known to suffer Cross-Slot Attention Bleeding on the
    Base 4B model (AGENTS.md rules 10/12) -- LoRA fine-tuning was the assumed-necessary fix, but
    it is expensive to prepare data for. This tests a training-free alternative first.

    Root motivation (AGENTS.md rule 26 / glyph_engine.py Rule 26): in the CURRENT encoding
    (`encode_glyph_to_incontext_tokens`, used identically across every prior probe in this repo),
    a glyph's RoPE (h, w) coordinates are LOCAL to its own bounding box (0 -> H_glyph, 0 -> W_glyph)
    -- they carry NO information about where on the canvas that glyph will actually be composited.
    Every reference block "starts" at the same local origin regardless of role or intended
    position; only the discrete time-offset (t=10, 20, 30...) currently differentiates slots.
    Hypothesis: giving each glyph ABSOLUTE canvas-matching (h, w) coordinates (i.e. the exact
    latent rows/columns it will occupy in the final image) gives the DiT's relative-position RoPE
    mechanics a second, more natural disambiguation signal -- canvas patches near a glyph's real
    position get a small relative offset (strong local attention), far patches get a large one
    (natural decay) -- something the current local-origin encoding cannot provide at all.

    3 conditions on the SAME 2-text layout (title @ top, subtitle @ bottom of a 9:16 canvas),
    each x 3 seeds:
      A) BASELINE       - current method: local (0,0) origin for both glyphs, t=10 (title) /
                           t=20 (subtitle). Expected: some crosstalk per AGENTS.md rules 10/12.
      B) SPATIAL_DIFF_T - glyphs get (h, w) shifted to their REAL canvas target position (title
                          near the top rows, subtitle near the bottom rows, both horizontally
                          centered), t=10/t=20 kept as well -- spatial binding ADDS to time-offset.
      C) SPATIAL_SAME_T - same spatial shift as (B), but BOTH glyphs at t=10 -- tests whether
                          spatial coordinates ALONE can substitute for the time-offset axis
                          entirely (a stronger, more surprising version of the hypothesis).

Strict Rule Adherence (AGENTS.md Rule 28): ASCII table + JSON manifest + PNGs only, no HTML.
Model: FLUX.2-klein-base-4B. Uses the NOW-LOCKED src/tendoo/glyph_engine.py render_glyph() API
for both glyphs (auto_size, canvas-aware) -- this probe assumes that foundation is reliable.

Usage on Remote Server (2x A30):
    python scripts/probe_rope_spatial_binding.py                # all 3 conditions x 3 seeds = 9 runs
    python scripts/probe_rope_spatial_binding.py --conditions A C   # baseline vs strongest treatment only
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
# 1. LAYOUT: 2-TEXT POSTER (title @ top, subtitle @ bottom), 9:16 CANVAS
# ==================================================================================================

TITLE_TEXT = "TUYỂN DỤNG NHÂN TÀI"
SUBTITLE_TEXT = "BỨT PHÁ MỌI GIỚI HẠN"
CANVAS = (576, 1024)  # real 9:16 primary target
DEFAULT_SEEDS = [42, 123, 777]

DEFAULT_PROMPT = (
    "Poster tuyển dụng phong cách công ty công nghệ hiện đại, nền gradient xanh dương đậm sang "
    "trọng, dòng chữ tiêu đề lớn dập nổi kim loại sắc nét ở phía trên, dòng chữ phụ phát sáng neon "
    "tinh tế ở phía dưới, bố cục sạch sẽ chuyên nghiệp, không có chữ ký, không có watermark"
)


@dataclass
class SlotSpec:
    role: str
    text: str
    t_offset: float
    v_anchor: str  # "top" or "bottom" -- which part of the canvas this slot targets


CONDITIONS: Dict[str, Dict[str, Any]] = {
    "A": {
        "label": "BASELINE (local origin, t=10/t=20)",
        "spatial_binding": False,
        "slots": [
            SlotSpec("title", TITLE_TEXT, 10.0, "top"),
            SlotSpec("subtitle", SUBTITLE_TEXT, 20.0, "bottom"),
        ],
    },
    "B": {
        "label": "SPATIAL_DIFF_T (canvas-matched h,w, t=10/t=20)",
        "spatial_binding": True,
        "slots": [
            SlotSpec("title", TITLE_TEXT, 10.0, "top"),
            SlotSpec("subtitle", SUBTITLE_TEXT, 20.0, "bottom"),
        ],
    },
    "C": {
        "label": "SPATIAL_SAME_T (canvas-matched h,w, BOTH t=10)",
        "spatial_binding": True,
        "slots": [
            SlotSpec("title", TITLE_TEXT, 10.0, "top"),
            SlotSpec("subtitle", SUBTITLE_TEXT, 10.0, "bottom"),
        ],
    },
}


# ==================================================================================================
# 2. 4D ROPE ENCODING -- LOCAL (baseline) vs CANVAS-MATCHED (spatial binding)
# ==================================================================================================

def encode_glyph_to_ref_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float,
    device: str | torch.device,
    h_offset: int = 0,
    w_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a glyph bitmap into 4D RoPE In-Context reference tokens.
    - h_offset/w_offset = 0 (default): reproduces the EXISTING local-origin convention used
      identically across every prior probe in this repo (h,w range from 0 -> glyph_H/W).
    - h_offset/w_offset != 0: shifts the glyph's (h, w) coordinate origin to an ABSOLUTE position
      in the canvas's own latent grid -- this is the "spatial binding" treatment under test.
    """
    arr = np.array(glyph_img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latent = ae.encode(tensor)
    ref_tokens, _ = prc_img(latent[0])
    ref_tokens = ref_tokens.unsqueeze(0)

    g_h, g_w = latent.shape[2], latent.shape[3]
    t_coords = torch.full((g_h, g_w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(g_h, dtype=torch.float32, device=device).unsqueeze(1).expand(g_h, g_w) + h_offset
    w_coords = torch.arange(g_w, dtype=torch.float32, device=device).unsqueeze(0).expand(g_h, g_w) + w_offset
    l_coords = torch.zeros((g_h, g_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0)
    return ref_tokens, ref_ids


def compute_canvas_offset(glyph_lat_h: int, glyph_lat_w: int, canvas_lat_h: int, canvas_lat_w: int, v_anchor: str) -> Tuple[int, int]:
    """
    Computes the (h_offset, w_offset) that places a glyph at its intended region of the canvas's
    OWN latent grid: horizontally centered always; vertically near the top (small margin) or
    near the bottom (small margin) depending on `v_anchor`.
    """
    margin = 2  # latent rows/cols of breathing room from the canvas edge
    w_offset = max(0, (canvas_lat_w - glyph_lat_w) // 2)
    if v_anchor == "top":
        h_offset = margin
    else:  # "bottom"
        h_offset = max(0, canvas_lat_h - glyph_lat_h - margin)
    return h_offset, w_offset


# ==================================================================================================
# 3. MAIN RUNNER
# ==================================================================================================

@dataclass
class RunConfig:
    condition: str
    seed: int
    run_id: str


def build_run_matrix(conditions: List[str], seeds: List[int]) -> List[RunConfig]:
    return [RunConfig(condition=c, seed=s, run_id=f"cond{c}_seed{s}") for c in conditions for s in seeds]


def run_probe(
    conditions: List[str], seeds: List[int], font: str = "bevietnam", prompt: str = DEFAULT_PROMPT,
    output_dir: str = "output_rope_spatial_binding", model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None, num_steps: int = 50, guidance: float = 4.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_runs = build_run_matrix(conditions, seeds)

    print("=" * 100)
    print(" [*] TENDOO AI - ROPE SPATIAL BINDING PROBE (DIRECTION 1)")
    print("=" * 100)
    print(f"  Title    : \"{TITLE_TEXT}\" (top)")
    print(f"  Subtitle : \"{SUBTITLE_TEXT}\" (bottom)")
    print(f"  Canvas   : {CANVAS[0]}x{CANVAS[1]} (9:16)")
    print(f"  Conditions: {conditions}")
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

    canvas_w, canvas_h = CANVAS
    canvas_w = (canvas_w // 16) * 16
    canvas_h = (canvas_h // 16) * 16
    lat_w, lat_h = canvas_w // 16, canvas_h // 16

    # Render title + subtitle glyphs ONCE via the LOCKED production glyph_engine API.
    glyph_infos: Dict[str, GlyphInfo] = {}
    for role, text in [("title", TITLE_TEXT), ("subtitle", SUBTITLE_TEXT)]:
        info = render_glyph(text=text, font_name_or_path=font, auto_size=True, target_canvas_w=canvas_w, target_canvas_h=canvas_h)
        glyph_infos[role] = info
        aspect = info.width_px / info.height_px
        print(f"  [{role:9s}] {info.width_px}x{info.height_px}px {info.font_size_pt}pt {len(info.lines)}L "
              f"{info.token_count}tok aspect={aspect:.2f} :: {info.lines}")
        info.image.save(out_path / f"{role}_glyph.png")

    # Pre-compute the 4D RoPE ref tokens+ids for every (condition, role) combination -- these do
    # not depend on seed, only the diffusion sampling does.
    ref_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
    for cond_key, cond in CONDITIONS.items():
        if cond_key not in conditions:
            continue
        for slot in cond["slots"]:
            info = glyph_infos[slot.role]
            if cond["spatial_binding"]:
                h_off, w_off = compute_canvas_offset(info.latent_h, info.latent_w, lat_h, lat_w, slot.v_anchor)
            else:
                h_off, w_off = 0, 0
            ref_tokens, ref_ids = encode_glyph_to_ref_tokens(
                ae=ae, glyph_img=info.image, t_offset=slot.t_offset, device=device_ae, h_offset=h_off, w_offset=w_off,
            )
            ref_cache[(cond_key, slot.role)] = (ref_tokens, ref_ids)
            print(f"  [{cond_key}:{slot.role:9s}] t={slot.t_offset} h_offset={h_off} w_offset={w_off} "
                  f"(canvas lat {lat_h}x{lat_w}, glyph lat {info.latent_h}x{info.latent_w})")

    print(f"\n[3/3] Executing {len(all_runs)} denoise run(s)...\n")
    manifest: List[Dict[str, Any]] = []

    for idx, run in enumerate(all_runs, 1):
        cond = CONDITIONS[run.condition]

        print("-" * 100)
        print(f"▶️  [{idx}/{len(all_runs)}] {run.run_id}: {cond['label']}")
        t_run_start = time.time()

        all_ref_tokens, all_ref_ids = [], []
        for slot in cond["slots"]:
            rt, ri = ref_cache[(run.condition, slot.role)]
            all_ref_tokens.append(rt.to(device_dit))
            all_ref_ids.append(ri.to(device_dit))
        ref_tokens = torch.cat(all_ref_tokens, dim=1)
        ref_ids = torch.cat(all_ref_ids, dim=1)

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
            "run_id": run.run_id, "condition": run.condition, "label": cond["label"], "seed": run.seed,
            "elapsed_s": round(elapsed, 2), "result_file": result_path.name, "verdict": "TBD",
        })
        print(f"    -> {result_path.name} ({elapsed:.1f}s)")

        del out_latent, out_tensor, z_init, img_tokens, img_ids, ref_tokens, ref_ids
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = out_path / "rope_spatial_binding_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print(" [*] SUMMARY (fill in Verdict: are BOTH title AND subtitle correct/sharp/non-overlapping?)")
    print("=" * 100)
    print(f"{'Run ID':<20} | {'Condition':<40} | {'Seed':<6} | Verdict")
    print("-" * 100)
    for r in manifest:
        print(f"{r['run_id']:<20} | {r['label']:<40} | {r['seed']:<6} | ? PASS/FAIL/PARTIAL")
    print("=" * 100)
    print(f"\n[+] Images saved in : {out_path.resolve()}")
    print(f"[+] Manifest saved in: {manifest_path.resolve()}")
    print("\n[NEXT STEP] Tally X/3 seeds per condition:")
    print("  - B or C clearly beats A -> spatial binding helps, worth building out further")
    print("    (more slots, product reference, more layouts).")
    print("  - C works about as well as B -> spatial coordinates alone can substitute for the")
    print("    canonical t-offset scheme -- a bigger, more surprising result worth its own writeup.")
    print("  - No condition beats A -> crosstalk is likely a distributional exposure gap (the model")
    print("    has just never seen >=2 simultaneous glyph-type tokens, regardless of coordinates),")
    print("    not a coordinate-collision problem -- back to considering LoRA or Regional Parallel")
    print("    Diffusion (Direction 2).\n")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI RoPE Spatial Binding Probe (Direction 1)")
    parser.add_argument("--font", type=str, default="bevietnam", help="Font alias (default: bevietnam)")
    parser.add_argument("--conditions", type=str, nargs="+", default=list(CONDITIONS.keys()), choices=list(CONDITIONS.keys()),
                         help="Which conditions to run (default: all A/B/C)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds to replicate")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Shared text prompt")
    parser.add_argument("--output_dir", type=str, default="output_rope_spatial_binding", help="Output directory")
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG guidance scale (default: 4.0)")

    args = parser.parse_args()
    run_probe(
        conditions=args.conditions, seeds=args.seeds, font=args.font, prompt=args.prompt, output_dir=args.output_dir,
        model_name=args.model_name, checkpoint_dir=args.checkpoint_dir,
        num_steps=args.steps, guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
