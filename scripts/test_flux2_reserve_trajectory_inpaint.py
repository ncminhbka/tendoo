#!/usr/bin/env python3
"""
scripts/test_flux2_reserve_trajectory_inpaint.py

==================================================================================================
SPIKE -- Co che D: sequential exact-trajectory latent injection + True CFG, tren
flux.2-klein-base-4b (50 buoc). Ke thua TRUC TIEP tu 1 file DA CO SAN, DA CHAY DUOC trong repo --
`scripts/probe_sequential_inpainting.py` (doc truc tiep, khong doan mu) -- von dung de ghep
tieu de+phu de (2 lan sinh anh rieng), o day doi muc dich sang "tranh vat the lon xon trong 1
vung reserve".

TAI SAO CO CHE NAY (sau khi 2 co che truoc deu khong an thua tren server that):
  - Co che A (`denoise_reserve`, ghi de latent tu 1 patch mau GIA) -- ra "o gia tao" (mang xam co
    van, lac tong) vi noi dung ep vao khong phai thu model tung sinh ra that.
  - Co che C (`denoise_cached`, reference-conditioning MEM qua KV-cache) -- 3 lan chay tren 2 canh
    khac nhau (macro dong ho, cho dem Trung Thu co/khong cau chi dan) deu cho ref_guided ~ baseline
    -- model co toan quyen lo anh tham chieu di vi khong co rang buoc cung nao.
  - Co che D (file nay): Pass 1 sinh THAT 1 canh don gian (vd chi troi/trang), ghi lai TOAN BO
    trajectory. Pass 2 sinh THAT canh day du, nhung MOI BUOC ep vung reserve ve DUNG latent that
    cua Pass 1 tai buoc tuong ung (khong phai xap xi/gia tri gia) -- vi day la RANG BUOC CUNG
    (khong phai "goi y"), va noi dung ghep vao la THAT (khong phai ngoai lai), ca 2 van de cua A
    va C deu duoc giai quyet dong thoi.

MODEL BAT BUOC la flux.2-klein-base-4b (hoac klein-base-9b) -- KHONG phai ban distill: base co 50
buoc (du "duong bang" de bien mem hoa tron bien), va can True CFG (batch doi ["", prompt]) vi
params.use_guidance_embed=False cho ca 2 bien the 4B (xac nhan tu util.py). Cham hon han cac spike
truoc (2-3 lan chay full 50-buoc CFG, tuc ~200-300 forward pass qua model 4B) nhung van chap nhan
duoc voi "GPU thoai mai" -- day la spike kiem chung co che, khong phai toi uu toc do production.

Usage:
  python scripts/test_flux2_reserve_trajectory_inpaint.py \
      --scene-prompt "A bustling nighttime Mid-Autumn Festival street market, rows of traditional wooden stalls decorated with paper lanterns and string lights hanging overhead across the entire scene, tall trees strung with glowing lanterns reaching up toward the sky, a large full moon glowing above, warm golden lighting, richly detailed festive decorations filling the whole scene from top to bottom" \
      --simple-prompt "A plain quiet night sky with a soft glowing full moon and gentle wisps of cloud, smooth gradient from dark indigo to warm amber near the horizon, empty open space, no lanterns, no buildings, no decorations, no people, no trees" \
      --region title_two_line --seed 42 --out-dir output_flux2_trajectory_inpaint_test

Output: <out-dir>/baseline.png (scene prompt, khong can thiep), <out-dir>/pass1_simple.png (Pass 1
rieng, de kiem tra), <out-dir>/pass2_completed.png (Co che D -- KET QUA CHINH can xem), <out-dir>/
mask_preview.png (mask mem da dung), <out-dir>/region_debug.png, <out-dir>/run_info.json.

CHUA verify that tren GPU (khong co GPU trong moi truong viet code nay) -- lan chay dau tien cua
co che nay, can user tu danh gia.
==================================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from flux2 import util  # noqa: E402
from flux2.sampling import batched_prc_txt, get_schedule, prc_img  # noqa: E402
from flux2.masked_generation import (  # noqa: E402
    build_region_token_mask_multi,
    denoise_trajectory_cfg,
    denoise_trajectory_inpaint_exact,
    soften_region_mask,
)

REGION_SPECS: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = {
    "top": [((0.0, 0.25), (0.0, 1.0))],
    "bottom": [((0.75, 1.0), (0.0, 1.0))],
    "left": [((0.0, 1.0), (0.0, 0.4))],
    "right": [((0.0, 1.0), (0.6, 1.0))],
    "middle_left": [((0.3, 0.7), (0.0, 0.45))],
    "middle_right": [((0.3, 0.7), (0.55, 1.0))],
    # Mo phong tieu de 2 dong hinh bac thang user ve ra (xem scripts/test_flux2_reserve_region.py).
    "title_two_line": [
        ((0.05, 0.20), (0.08, 0.92)),
        ((0.20, 0.38), (0.28, 0.72)),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-prompt", required=True, help="Prompt canh DAY DU (Pass 2 / baseline) -- TIENG ANH, khong nhac chu/text/title")
    ap.add_argument("--simple-prompt", required=True, help="Prompt canh DON GIAN cho Pass 1 -- noi dung se duoc ep vao vung reserve o Pass 2")
    ap.add_argument("--model-name", default="flux.2-klein-base-4b", choices=["flux.2-klein-base-4b", "flux.2-klein-base-9b"],
                     help="PHAI la ban base (KHONG phai distill) -- can 50 buoc + True CFG, xem docstring dau file.")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="output_flux2_trajectory_inpaint_test")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                     help="Device cho DiT (buoc nang nhat). Mac dinh cuda:0.")
    ap.add_argument("--aux-device", default=None,
                     help="Device cho text-encoder + AE. Mac dinh: cuda:1 neu may co >=2 GPU, khong thi dung chung --device.")
    ap.add_argument("--region", default="title_two_line", choices=list(REGION_SPECS.keys()))
    ap.add_argument("--feather-iters", type=int, default=3,
                     help="So lan average-pool 3x3 lam mem bien mask (soften_region_mask) -- cang lon bien cang rong/mem.")
    ap.add_argument("--num-steps", type=int, default=None, help="Mac dinh: lay tu FLUX2_MODEL_INFO defaults (50 cho base).")
    ap.add_argument("--guidance", type=float, default=None, help="Mac dinh: lay tu FLUX2_MODEL_INFO defaults (4.0 cho base).")
    args = ap.parse_args()

    model_name = args.model_name
    device = args.device
    if device == "cpu":
        print("[CANH BAO] Khong thay CUDA -- script se rat cham hoac khong chay noi tren CPU voi model that.")

    if args.aux_device:
        aux_device = args.aux_device
    elif device.startswith("cuda") and torch.cuda.device_count() >= 2:
        aux_device = "cuda:1" if device != "cuda:1" else "cuda:0"
    else:
        aux_device = device
    print(f"DiT device: {device} | text-encoder+AE device: {aux_device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name} model/text-encoder/AE...")
    model = util.load_flow_model(model_name, device=device)
    text_encoder = util.load_text_encoder(model_name, device=aux_device)
    ae = util.load_ae(model_name, device=aux_device)

    defaults = util.FLUX2_MODEL_INFO[model_name]["defaults"]
    num_steps = args.num_steps if args.num_steps is not None else defaults["num_steps"]
    guidance = args.guidance if args.guidance is not None else defaults["guidance"]
    print(f"num_steps={num_steps} guidance={guidance}")

    ae_dtype = next(ae.parameters()).dtype
    print(f"AE dtype phat hien: {ae_dtype}")

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- True CFG: moi prompt encode CUNG voi chuoi rong "" thanh batch=2 (uncond, cond) --
    #     dung Y HET pattern DA CO SAN va DA CHAY DUOC trong probe_sequential_inpainting.py. ---
    with torch.no_grad():
        txt_scene, txt_ids_scene = batched_prc_txt(text_encoder(["", args.scene_prompt]).to(torch.bfloat16))
        txt_simple, txt_ids_simple = batched_prc_txt(text_encoder(["", args.simple_prompt]).to(torch.bfloat16))
    txt_scene, txt_ids_scene = txt_scene.to(device), txt_ids_scene.to(device)
    txt_simple, txt_ids_simple = txt_simple.to(device), txt_ids_simple.to(device)

    # --- Latent kich thuoc canvas: suy ra tu 1 lan encode anh trang cung kich thuoc, KHONG doan he
    #     so downsample. ---
    blank = Image.new("RGB", (args.width, args.height), (255, 255, 255))
    x_probe = torch.from_numpy(np.array(blank)).permute(2, 0, 1).float().unsqueeze(0).to(device=aux_device, dtype=ae_dtype) / 127.5 - 1.0
    with torch.no_grad():
        z_probe = ae.encode(x_probe)
    _, C, h_lat, w_lat = z_probe.shape
    print(f"Latent size suy ra tu anh that: C={C} h_lat={h_lat} w_lat={w_lat}")

    # Noise cho DiT phai la bfloat16 ngay tu luc tao (khop dtype model) -- CUNG 1 noise cho ca 2
    # pass (dung y het probe_sequential_inpainting.py: z_init dung chung giup bien mask khop nhau
    # tu nhien hon vi ca 2 nhanh xuat phat tu cung 1 diem nhieu).
    noise = torch.randn((1, C, h_lat, w_lat), device=device, generator=generator, dtype=torch.bfloat16)
    img_init, img_ids = prc_img(noise[0])
    img_init, img_ids = img_init.unsqueeze(0).to(device), img_ids.unsqueeze(0).to(device)

    timesteps = get_schedule(num_steps, image_seq_len=img_init.shape[1])

    # --- Vung reserve + mask mem ---
    rects = REGION_SPECS[args.region]
    hard_mask = build_region_token_mask_multi(h_lat, w_lat, rects, device=device)
    soft_mask = soften_region_mask(hard_mask, h_lat, w_lat, feather_iters=args.feather_iters)
    print(f"Region '{args.region}' ({len(rects)} rect): {hard_mask.sum().item()}/{hard_mask.numel()} token (cung), "
          f"mask mem trung binh={soft_mask.mean().item():.3f}")

    def decode_and_save(latent_tokens: torch.Tensor, path: Path):
        z = latent_tokens[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
        with torch.no_grad():
            x = ae.decode(z).float()
        x = ((x[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(x).save(path)

    # QUAN TRONG: het thay ca 3 lan chay PHAI trong torch.no_grad() -- da hoc tu OOM that o
    # scripts/test_flux2_reserve_region.py (thieu no_grad khien autograd giu do thi tinh toan qua
    # het cac buoc, phong VRAM du model nho).
    with torch.no_grad():
        print("Chay BASELINE (scene prompt, True CFG, khong can thiep)...")
        out_baseline, _ = denoise_trajectory_cfg(model, img_init.clone(), img_ids, txt_scene, txt_ids_scene, timesteps, guidance=guidance)
        decode_and_save(out_baseline, out_dir / "baseline.png")
        del out_baseline
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        print(f"Chay PASS 1 (simple prompt, True CFG, ghi lai trajectory, {num_steps} buoc)...")
        out_pass1, pass1_trajectory = denoise_trajectory_cfg(model, img_init.clone(), img_ids, txt_simple, txt_ids_simple, timesteps, guidance=guidance)
        decode_and_save(out_pass1, out_dir / "pass1_simple.png")
        del out_pass1
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        print("Chay PASS 2 (Co che D -- scene prompt + ep vung reserve theo trajectory Pass 1)...")
        out_pass2 = denoise_trajectory_inpaint_exact(
            model, img_init.clone(), pass1_trajectory, soft_mask,
            img_ids, txt_scene, txt_ids_scene, timesteps, guidance=guidance,
        )
        decode_and_save(out_pass2, out_dir / "pass2_completed.png")

    # --- Mask preview (mask mem, phong to len kich thuoc canvas) ---
    mask_vis_2d = soft_mask.reshape(h_lat, w_lat).cpu().numpy()
    mask_vis = Image.fromarray((mask_vis_2d * 255).astype("uint8")).resize((args.width, args.height), resample=Image.NEAREST)
    mask_vis.save(out_dir / "mask_preview.png")

    # --- Debug overlay: to do (hop) vung reserve len anh baseline de doi chieu ---
    dbg = Image.open(out_dir / "baseline.png").convert("RGB")
    draw = ImageDraw.Draw(dbg, "RGBA")
    for row_frac, col_frac in rects:
        r0, r1 = row_frac[0] * dbg.height, row_frac[1] * dbg.height
        c0, c1 = col_frac[0] * dbg.width, col_frac[1] * dbg.width
        draw.rectangle([c0, r0, c1, r1], outline=(255, 0, 0, 255), width=4, fill=(255, 0, 0, 60))
    dbg.save(out_dir / "region_debug.png")

    info = {
        "model": model_name, "scene_prompt": args.scene_prompt, "simple_prompt": args.simple_prompt,
        "region": args.region, "num_rects": len(rects), "seed": args.seed, "num_steps": num_steps,
        "guidance": guidance, "latent_shape": [C, h_lat, w_lat], "feather_iters": args.feather_iters,
        "timesteps": timesteps,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"XONG. So sanh {out_dir}/baseline.png vs {out_dir}/pass2_completed.png (Co che D -- KET QUA "
        f"CHINH), doi chieu voi {out_dir}/pass1_simple.png (Pass 1 rieng), {out_dir}/mask_preview.png "
        f"va {out_dir}/region_debug.png."
    )


if __name__ == "__main__":
    main()
