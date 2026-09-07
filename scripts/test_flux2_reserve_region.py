#!/usr/bin/env python3
"""
scripts/test_flux2_reserve_region.py

==================================================================================================
SPIKE -- test THẬT trên server (2x A30 hoặc máy có GPU đã có weights FLUX.2): so sánh trực tiếp
baseline (`sampling.denoise()` gốc, không sửa) vs `masked_generation.denoise_reserve()` (Cơ chế A
-- ép vùng reserve giữ đơn giản/ít vật thể) TRÊN CÙNG 1 seed/prompt, để xem cơ chế A có thực sự
"chừa chỗ ít vật thể hơn" mà KHÔNG tạo ô chết/giả tạo hay không.

CHỈ chạy được trên máy có GPU + đã có sẵn weights FLUX.2 klein base 4B (theo đúng cách
`src/flux2/util.py` tự tìm -- xem `find_persistent_data_root()`, hoặc set biến môi trường
KLEIN_4B_BASE_MODEL_PATH trỏ thẳng tới file .safetensors). Model MẶC ĐỊNH dùng
"flux.2-klein-base-4b" (50 bước, guidance=4.0, True CFG) -- đúng bản Base mà
docs/PHASE_3_LORA_TRAINING_ROADMAP.md nhắm tới, KHÔNG phải bản distill 4-bước.

Usage:
  python scripts/test_flux2_reserve_region.py \
      --prompt "A luxurious rose gold smart watch on a wooden desk, pastel pink studio background" \
      --product-phrase "smart watch" \
      --out-dir output_flux2_reserve_test \
      --seed 42

Output: <out-dir>/baseline.png, <out-dir>/reserve.png, <out-dir>/region_debug.png (vùng reserve
tô đỏ để đối chiếu 2 ảnh), <out-dir>/run_info.json.
==================================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from flux2 import util  # noqa: E402
from flux2.sampling import denoise as denoise_baseline  # noqa: E402
from flux2.sampling import get_schedule, prc_img, prc_txt  # noqa: E402
from flux2.masked_generation import build_region_token_mask, default_lock_schedule, denoise_reserve  # noqa: E402

MODEL_NAME = "flux.2-klein-base-4b"


def encode_flat_patch(ae, device, color=(235, 225, 210), size=(64, 64)) -> torch.Tensor:
    """Ma hoa 1 patch mau phang don gian (dai dien 'noi dung don gian' cho vung reserve) qua VAE,
    tra ve latent shape (1, C, h_small, w_small) -- CHỈ dung de tao z_known nho roi broadcast, vi
    z_known chi can 1 vector kenh (C,), khong can dung nguyen ca patch lon."""
    img = Image.new("RGB", size, color)
    x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        z = ae.encode(x)
    return z  # (1, C, h_lat_small, w_lat_small)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Mo ta canh/san pham, TIENG ANH, KHONG nhac chu/text/title (dung quy tac da co)")
    ap.add_argument("--product-phrase", required=True, help="Cum tu san pham xuat hien NGUYEN VAN trong --prompt (dung cho Co che B neu ban tu bat -- Co che A khong can)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="output_flux2_reserve_test")
    ap.add_argument("--region", default="top", choices=["top", "bottom", "left", "right", "middle_left", "middle_right"],
                     help="Vung reserve don gian de test nhanh -- top/bottom = dai ngang tren/duoi 25%%, left/right = nua doc trai/phai, middle_left/middle_right = 1 nua doc nhung chi tam giua")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[CANH BAO] Khong thay CUDA -- script se rat cham hoac khong chay noi tren CPU voi model that.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} model/text-encoder/AE ({device})...")
    model = util.load_flow_model(MODEL_NAME, device=device)
    text_encoder = util.load_text_encoder(MODEL_NAME, device=device)
    ae = util.load_ae(MODEL_NAME, device=device)

    defaults = util.FLUX2_MODEL_INFO[MODEL_NAME]["defaults"]
    num_steps, guidance = defaults["num_steps"], defaults["guidance"]

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- Text conditioning (dung chung cho ca 2 nhanh) ---
    ctx = text_encoder([args.prompt])  # (1, L_txt, D)
    ctx, ctx_ids = prc_txt(ctx[0])
    ctx, ctx_ids = ctx.unsqueeze(0).to(device), ctx_ids.unsqueeze(0).to(device)

    # --- Latent kich thuoc canvas: suy ra tu 1 lan encode anh trang cung kich thuoc, KHONG doan he
    #     so downsample --> luon dung du bao nhieu VAE/model doi ---
    blank = Image.new("RGB", (args.width, args.height), (255, 255, 255))
    import numpy as np
    x_probe = torch.from_numpy(np.array(blank)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
    with torch.no_grad():
        z_probe = ae.encode(x_probe)
    _, C, h_lat, w_lat = z_probe.shape
    print(f"Latent size suy ra tu anh that: C={C} h_lat={h_lat} w_lat={w_lat}")

    noise = torch.randn((1, C, h_lat, w_lat), device=device, generator=generator)
    img, img_ids = prc_img(noise[0])
    img, img_ids = img.unsqueeze(0).to(device), img_ids.unsqueeze(0).to(device)

    timesteps = get_schedule(num_steps, image_seq_len=img.shape[1])

    # --- Vung reserve (vi du don gian de test nhanh -- thay bang safe_rect thuc te khi tich hop) ---
    region_specs = {
        "top": ((0.0, 0.25), (0.0, 1.0)),
        "bottom": ((0.75, 1.0), (0.0, 1.0)),
        "left": ((0.0, 1.0), (0.0, 0.4)),
        "right": ((0.0, 1.0), (0.6, 1.0)),
        "middle_left": ((0.3, 0.7), (0.0, 0.45)),
        "middle_right": ((0.3, 0.7), (0.55, 1.0)),
    }
    row_frac, col_frac = region_specs[args.region]
    region_mask = build_region_token_mask(h_lat, w_lat, row_frac, col_frac, device=device)
    print(f"Region '{args.region}': {region_mask.sum().item()}/{region_mask.numel()} token")

    z_known_patch = encode_flat_patch(ae, device)  # (1, C, h_small, w_small)
    z_known = z_known_patch.mean(dim=(0, 2, 3))  # (C,) -- gia tri kenh trung binh, broadcast cho ca vung

    def decode_and_save(latent_tokens: torch.Tensor, path: Path):
        # latent_tokens: (1, L, C) -> ve lai (1, C, h_lat, w_lat) truoc khi decode
        z = latent_tokens[0].transpose(0, 1).reshape(1, C, h_lat, w_lat)
        with torch.no_grad():
            x = ae.decode(z)
        x = ((x[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(x).save(path)

    print("Chay BASELINE (denoise goc, khong sua)...")
    out_baseline = denoise_baseline(model, img.clone(), img_ids, ctx, ctx_ids, timesteps, guidance=guidance)
    decode_and_save(out_baseline, out_dir / "baseline.png")

    print("Chay RESERVE (Co che A -- denoise_reserve)...")
    out_reserve = denoise_reserve(
        model, img.clone(), img_ids, ctx, ctx_ids, timesteps, guidance=guidance,
        region_mask=region_mask, z_known=z_known, lock_schedule=default_lock_schedule,
    )
    decode_and_save(out_reserve, out_dir / "reserve.png")

    # --- Debug overlay: to do vung reserve len anh baseline de doi chieu ---
    dbg = Image.open(out_dir / "baseline.png").convert("RGB")
    draw = ImageDraw.Draw(dbg, "RGBA")
    r0, r1 = row_frac[0] * dbg.height, row_frac[1] * dbg.height
    c0, c1 = col_frac[0] * dbg.width, col_frac[1] * dbg.width
    draw.rectangle([c0, r0, c1, r1], outline=(255, 0, 0, 255), width=4, fill=(255, 0, 0, 60))
    dbg.save(out_dir / "region_debug.png")

    info = {
        "model": MODEL_NAME, "prompt": args.prompt, "product_phrase": args.product_phrase,
        "region": args.region, "seed": args.seed, "num_steps": num_steps, "guidance": guidance,
        "latent_shape": [C, h_lat, w_lat],
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"XONG. Xem {out_dir}/baseline.png vs {out_dir}/reserve.png (doi chieu voi region_debug.png).")


if __name__ == "__main__":
    main()
