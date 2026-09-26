#!/usr/bin/env python3
"""
scripts/bench_maskless.py -- CHỈ CHẠY TRÊN MÁY CHỦ GPU (AGENTS.md §1: không chạy DiT trên máy local).

GĐ 5 (ROADMAP §5.3, ràng buộc 3): ĐO wall-clock thật của Maskless Mode so với Velocity Blending
(scene + corridor). Không ghi "nhanh gấp 2" vào tài liệu trước khi có số từ script này.

Mỗi seed đo từng pha, có torch.cuda.synchronize() trước mỗi mốc giờ:
  text_encode   mã hoá prompt (masked: scene + corridor; maskless: chỉ scene)
  dit           Euler ODE (masked: batch 2 blend; maskless: batch 1 scene-only)
  vae_decode    giải mã latent
  render        HTML/CSS overlay qua Playwright (giống nhau ở 2 chế độ -- đo để biết tỉ trọng)

  cd ~/work/tendoo-v3 && PYTHONPATH=src python scripts/bench_maskless.py --seeds 5 --model distill
  -> in bảng + ghi output_probe/bench_maskless.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

PDATA = [Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B")), Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-4B")),
         Path("/home/jovyan/persistent-data/FLUX.2-klein-base-4B")]
SCENE = "Soft golden bokeh and fine glitter dust drifting over a smooth deep red gradient, cinematic lighting, zero text"
CORRIDOR = "Smooth deep red gradient in extreme soft focus, clean negative space without objects, matching scene color"


def _setup_paths(model: str) -> str:
    """Tìm weights giống demo_server.lifespan -- trả tên model cho flux2.util."""
    fname, name, env = (("flux-2-klein-4b.safetensors", "flux.2-klein-4b", "KLEIN_4B_MODEL_PATH") if model == "distill"
                        else ("flux-2-klein-base-4b.safetensors", "flux.2-klein-base-4b", "KLEIN_4B_BASE_MODEL_PATH"))
    for p in PDATA:
        if (p / fname).exists():
            os.environ.setdefault(env, str(p / fname))
        if (p / "vae" / "diffusion_pytorch_model.safetensors").exists():
            os.environ.setdefault("AE_MODEL_PATH", str(p / "vae" / "diffusion_pytorch_model.safetensors"))
        if (p / "text_encoder").exists():
            os.environ.setdefault("TEXT_ENCODER_PATH", str(p / "text_encoder"))
    if env not in os.environ:
        raise SystemExit(f"Không tìm thấy {fname} trong {PDATA}")
    return name


def _sync():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--model", choices=["distill", "base"], default="distill")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--template", default="grand_opening_banner")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output_probe" / "bench_maskless.json"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Cần CUDA -- script này chỉ chạy trên máy chủ GPU (AGENTS.md §1).")

    from flux2 import util
    from flux2.sampling import get_schedule, prc_img, prc_txt
    from tendoo_v3.mask_engine import generate_template_mask
    from tendoo_v3.renderer import render_plan_to_poster
    from tendoo_v3.schema import StyleConfig, TendooCreativePlan
    from tendoo_v3.velocity_blending import denoise_regional_velocity_blended, denoise_scene_only

    name = _setup_paths(args.model)
    steps, guidance = (8, 1.5) if args.model == "distill" else (50, 4.0)
    dev_dit = "cuda:0"
    dev_aux = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    dit = util.load_flow_model(name, device=dev_dit).eval()
    ae = util.load_ae(name, device=dev_aux).eval()
    te = util.load_text_encoder(name, device=dev_aux)
    ae_dtype = next(ae.parameters()).dtype
    w, h = args.width, args.height
    plan = TendooCreativePlan(template=args.template, hero="MỪNG KHAI TRƯƠNG GIẢM 30%", subhead="Duy nhất 3 ngày đầu tiên",
                              cta="GHÉ NGAY", style=StyleConfig(font="bevietnam", theme_color="#FACC15", text_effect="3d_gold", background_tone="cinema_red"))
    mask_np = generate_template_mask(template=args.template, width=w, height=h)
    mask_flat = torch.from_numpy(np.array(Image.fromarray((mask_np * 255).astype(np.uint8)).resize((w // 16, h // 16), Image.Resampling.BICUBIC),
                                          dtype=np.float32) / 255.0).reshape(1, -1, 1).to(dev_dit)

    def encode(prompt):
        ctx = te([prompt]).to(torch.bfloat16)
        ctx, ids = prc_txt(ctx[0])
        return ctx.unsqueeze(0).to(dev_dit), ids.unsqueeze(0).to(dev_dit)

    def run(mode: str, seed: int, out_dir: Path) -> dict:
        t = {}
        with torch.no_grad():
            _sync(); t0 = time.perf_counter()
            cs, cs_ids = encode(SCENE)
            if mode == "masked":
                cc, cc_ids = encode(CORRIDOR)
            _sync(); t["text_encode"] = time.perf_counter() - t0

            torch.manual_seed(seed)
            z = torch.randn(1, 128, h // 16, w // 16, device=dev_dit, dtype=torch.bfloat16)
            tok, ids = prc_img(z[0])
            tok, ids = tok.unsqueeze(0).to(dev_dit), ids.unsqueeze(0).to(dev_dit)
            ts = get_schedule(num_steps=steps, image_seq_len=tok.shape[1])
            _sync(); t0 = time.perf_counter()
            if mode == "masked":
                out = denoise_regional_velocity_blended(dit, tok, ids, cs, cs_ids, cc, cc_ids, mask_flat, ts, guidance=guidance, num_canvas_tokens=tok.shape[1])
            else:
                out = denoise_scene_only(dit, tok, ids, cs, cs_ids, ts, guidance=guidance, num_canvas_tokens=tok.shape[1])
            _sync(); t["dit"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            zd = out[0].transpose(0, 1).reshape(1, 128, h // 16, w // 16).to(device=dev_aux, dtype=ae_dtype)
            x = ae.decode(zd).float()
            _sync(); t["vae_decode"] = time.perf_counter() - t0
        img = Image.fromarray(((x[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy())
        bg_path = out_dir / f"{mode}_{seed}_bg.png"
        img.save(bg_path)
        import base64, io
        buf = io.BytesIO(); img.save(buf, format="PNG")
        t0 = time.perf_counter()
        render_plan_to_poster(replace_maskless(plan, mode == "maskless"), "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                              out_dir / f"{mode}_{seed}_poster.png", w, h)
        t["render"] = time.perf_counter() - t0
        t["total"] = sum(t.values())
        return t

    def replace_maskless(p, flag):
        from dataclasses import replace
        return replace(p, maskless=flag)

    out_dir = Path(args.out).parent / "bench_maskless"
    out_dir.mkdir(parents=True, exist_ok=True)
    run("masked", 0, out_dir); run("maskless", 0, out_dir)  # khởi động (CUDA kernels, cache)
    res = {"model": name, "steps": steps, "size": [w, h], "gpus": torch.cuda.device_count(), "masked": [], "maskless": []}
    for s in range(1, args.seeds + 1):
        for mode in ("masked", "maskless"):
            res[mode].append(run(mode, s, out_dir))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{name} {steps} bước {w}x{h}, {args.seeds} seed (trung vị giây):")
    for k in ("text_encode", "dit", "vae_decode", "render", "total"):
        a, b = statistics.median(r[k] for r in res["masked"]), statistics.median(r[k] for r in res["maskless"])
        print(f"  {k:<12} có mask {a:7.3f}   maskless {b:7.3f}   tăng tốc x{a / b if b else float('nan'):.2f}")
    print(f"Ảnh + số liệu: {out_dir}, {args.out}")


if __name__ == "__main__":
    main()
