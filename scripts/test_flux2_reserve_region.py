#!/usr/bin/env python3
"""
scripts/test_flux2_reserve_region.py

==================================================================================================
SPIKE -- test THẬT trên server (2x A30 hoặc máy có GPU đã có weights FLUX.2): so sánh trực tiếp
baseline (`sampling.denoise()` gốc, không sửa) vs `masked_generation.denoise_reserve()` (Cơ chế A
-- ép vùng reserve giữ đơn giản/ít vật thể) TRÊN CÙNG 1 seed/prompt, để xem cơ chế A có thực sự
"chừa chỗ ít vật thể hơn" mà KHÔNG tạo ô chết/giả tạo hay không.

Model MẶC ĐỊNH: "flux.2-klein-4b" (bản DISTILL, 4 bước, guidance_distilled=True) -- KHÔNG PHẢI
bản base như bản đầu tiên của script này. Đổi lại theo đúng góp ý: spike này không train LoRA
(chỉ test cơ chế reserve/attention-bias, training-free), nên không có lý do bắt buộc dùng base --
base (guidance_distilled=False, 50 bước) cần TRUE CFG (2 forward pass/bước, xem
`scripts/cli.py:576-626` `denoise_cfg`) mà `denoise()`/`denoise_reserve()` ở đây KHÔNG cài -- tức
chạy base qua các hàm này trước đó không chỉ CHẬM HƠN mà còn SAI ngữ nghĩa guidance. Vẫn có thể
chọn base qua `--model-name flux.2-klein-base-4b` để so sánh, nhưng script sẽ in cảnh báo.

Weights tìm theo đúng cách `src/flux2/util.py` tự tìm (`find_persistent_data_root()`), hoặc set
biến môi trường riêng theo model (xem `FLUX2_MODEL_INFO[...]["model_path"]`):
  - distill 4B:  KLEIN_4B_MODEL_PATH
  - base 4B:     KLEIN_4B_BASE_MODEL_PATH
LƯU Ý: `find_persistent_data_root()` hiện CHỈ tự nhận diện thư mục "FLUX.2-klein-base-4B" -- nếu
server có CẢ 2 thư mục weights (base 4B và distill 4B) cạnh nhau, hàm này sẽ luôn trỏ về thư mục
base bất kể `--model-name` là gì. Vì vậy khi chạy distill, PHẢI set KLEIN_4B_MODEL_PATH trỏ thẳng
tới file .safetensors của bản distill (xem hướng dẫn chạy ở cuối file/README lệnh).

Dùng 2 GPU (mặc định tự phát hiện qua torch.cuda.device_count()): DiT đặt ở `--device` (mặc định
cuda:0), text-encoder + AE đặt ở GPU còn lại (tự chọn cuda:1 nếu có, không thì dùng chung/CPU) --
để tránh OOM khi tải cả 3 model cùng lúc lên 1 GPU 24GB (đã gặp thật khi cả 3 dồn vào cuda:0).

OOM LẦN 3 (đã sửa): sau khi tách GPU + đổi sang bản distill, vẫn OOM ngay ở block DiT đầu tiên --
nguyên nhân THẬT không phải thiếu VRAM (model 4B, ~25 block, seq len ~6656 token không thể tự
cần 23GB nếu chạy inference thuần), mà là 2 lệnh gọi `denoise_baseline()`/`denoise_reserve()`
CHƯA được bọc trong `torch.no_grad()` -- autograd giữ nguyên đồ thị tính toán suốt toàn bộ vòng
lặp (25 block x 4 bước) làm phồng VRAM. Đối chiếu `scripts/cli.py:453` xác nhận: production code
bọc `with torch.no_grad():` quanh TOÀN BỘ generation (text encode -> denoise -> ae.decode), không
chỉ quanh ae.encode/decode như bản trước của script này. Đã sửa.

CHẠY THẬT LẦN 1 (đã có ảnh, đã tìm ra bug lock_schedule): baseline.png đẹp, không dính sản phẩm/
face. Nhưng reserve.png vùng top bị 1 MẢNG XÁM CÓ VÂN lạc hẳn tông màu (đúng "ô giả tạo" đã lo từ
đầu) -- tính lại bằng tay (không cần GPU) xác nhận: `default_lock_schedule` dùng ngưỡng t TUYỆT
ĐỐI (0.85/0.55) hiệu chỉnh hình dung cho lịch trình SNR-shift ~50 bước của base, nhưng với model
4 bước distill, timesteps thực tế đo được là [1.0, 0.9306, 0.8171, 0.5982, 0.0] --> lam(t_prev)
ra [1.0, 0.9706, 0.0624, 0.0] -- khoá CỨNG 2/4 bước (50% tổng bước) thay vì ~15% như tính cho
50 bước, mô hình không còn đủ bước để "vẽ lại" texture tự nhiên. Đã sửa bằng
`build_index_based_lock_schedule()` (masked_generation.py) -- tính lam THEO VỊ TRÍ BƯỚC (index),
không phụ thuộc hình dạng đường cong t -- mặc định giờ chỉ khoá đúng 1/4 bước với model distill.
Có thể chỉnh qua `--lock-frac`/`--anneal-frac`.

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
from flux2.masked_generation import build_index_based_lock_schedule, build_region_token_mask, denoise_reserve  # noqa: E402


def encode_flat_patch(ae, device, ae_dtype, color=(235, 225, 210), size=(64, 64)) -> torch.Tensor:
    """Ma hoa 1 patch mau phang don gian (dai dien 'noi dung don gian' cho vung reserve) qua VAE,
    tra ve latent shape (1, C, h_small, w_small) -- CHỈ dung de tao z_known nho roi broadcast, vi
    z_known chi can 1 vector kenh (C,), khong can dung nguyen ca patch lon."""
    img = Image.new("RGB", size, color)
    x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    x = x.unsqueeze(0).to(device=device, dtype=ae_dtype)
    with torch.no_grad():
        z = ae.encode(x)
    return z  # (1, C, h_lat_small, w_lat_small)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Mo ta canh/san pham, TIENG ANH, KHONG nhac chu/text/title (dung quy tac da co)")
    ap.add_argument("--product-phrase", required=True, help="Cum tu san pham xuat hien NGUYEN VAN trong --prompt (dung cho Co che B neu ban tu bat -- Co che A khong can)")
    ap.add_argument("--model-name", default="flux.2-klein-4b", choices=["flux.2-klein-4b", "flux.2-klein-base-4b"],
                     help="Mac dinh ban DISTILL (4 buoc, khong can True CFG -- khop dung denoise()/denoise_reserve() da viet). "
                          "Chon 'flux.2-klein-base-4b' se in canh bao: ham nay chua cai True CFG cho base.")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="output_flux2_reserve_test")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                     help="Device cho DiT (buoc nang nhat). Mac dinh cuda:0.")
    ap.add_argument("--aux-device", default=None,
                     help="Device cho text-encoder + AE. Mac dinh: cuda:1 neu may co >=2 GPU (tranh OOM do "
                          "don ca 3 model vao 1 GPU 24GB), khong thi dung chung --device.")
    ap.add_argument("--region", default="top", choices=["top", "bottom", "left", "right", "middle_left", "middle_right"],
                     help="Vung reserve don gian de test nhanh -- top/bottom = dai ngang tren/duoi 25%%, left/right = nua doc trai/phai, middle_left/middle_right = 1 nua doc nhung chi tam giua")
    ap.add_argument("--lock-frac", type=float, default=0.25,
                     help="Ti le SO BUOC dau tien khoa cung (lam=1.0) -- theo VI TRI BUOC, khong theo t tuyet doi. Mac dinh 0.25 (vd 4 buoc -> khoa dung buoc 1).")
    ap.add_argument("--anneal-frac", type=float, default=0.25,
                     help="Ti le so buoc TIEP THEO dung anneal cosine 1.0->0.0 (can >=2 buoc thuc te moi anneal, khong thi bo qua va tha han luon).")
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

    if model_name == "flux.2-klein-base-4b":
        print(
            "[CANH BAO] --model-name flux.2-klein-base-4b: model nay guidance_distilled=False, can "
            "TRUE CFG (2 forward pass/buoc, xem scripts/cli.py denoise_cfg) de guidance hoat dong dung. "
            "denoise()/denoise_reserve() trong file nay KHONG cai True CFG (chi 1 forward pass/buoc) -- "
            "ket qua co the khac ky vong. Dung ban distill (mac dinh) neu chi test co che reserve/attention-bias."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name} model/text-encoder/AE...")
    model = util.load_flow_model(model_name, device=device)
    text_encoder = util.load_text_encoder(model_name, device=aux_device)
    ae = util.load_ae(model_name, device=aux_device)

    defaults = util.FLUX2_MODEL_INFO[model_name]["defaults"]
    num_steps, guidance = defaults["num_steps"], defaults["guidance"]

    # AE checkpoint tren server duoc luu san o bfloat16 (khong phai fp32) -- doc dtype THAT tu
    # chinh cac tham so da load, KHONG doan, roi ep moi tensor dua vao ae.encode/decode ve dung
    # dtype nay (day la nguyen nhan loi "Input type (float) and bias type (c10::BFloat16)").
    ae_dtype = next(ae.parameters()).dtype
    print(f"AE dtype phat hien: {ae_dtype}")

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- Text conditioning (dung chung cho ca 2 nhanh) -- text_encoder chay tren aux_device, sau do
    #     chuyen ctx sang `device` (noi DiT chay) -- model DiT la bfloat16 (util.load_flow_model ep
    #     cung), nen ctx cung phai bfloat16, dung y het pattern that trong scripts/cli.py ---
    with torch.no_grad():
        ctx = text_encoder([args.prompt]).to(torch.bfloat16)  # (1, L_txt, D), tren aux_device
    ctx, ctx_ids = prc_txt(ctx[0])
    ctx, ctx_ids = ctx.unsqueeze(0).to(device), ctx_ids.unsqueeze(0).to(device)

    # --- Latent kich thuoc canvas: suy ra tu 1 lan encode anh trang cung kich thuoc, KHONG doan he
    #     so downsample --> luon dung du bao nhieu VAE/model doi. AE o aux_device. ---
    blank = Image.new("RGB", (args.width, args.height), (255, 255, 255))
    import numpy as np
    x_probe = torch.from_numpy(np.array(blank)).permute(2, 0, 1).float().unsqueeze(0).to(device=aux_device, dtype=ae_dtype) / 127.5 - 1.0
    with torch.no_grad():
        z_probe = ae.encode(x_probe)
    _, C, h_lat, w_lat = z_probe.shape
    print(f"Latent size suy ra tu anh that: C={C} h_lat={h_lat} w_lat={w_lat}")

    # Noise cho DiT phai la bfloat16 ngay tu luc tao (khop dtype model), dung y het scripts/cli.py
    noise = torch.randn((1, C, h_lat, w_lat), device=device, generator=generator, dtype=torch.bfloat16)
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

    # Lock schedule THEO VI TRI BUOC (khong theo t tuyet doi) -- fix da xac nhan bang so lieu that:
    # voi model 4-buoc, nguong t=0.85/0.55 (default_lock_schedule) khoa cung 2/4 buoc (50%) thay vi
    # ~15% nhu tinh cho 50-buoc base, ra dung hien tuong "vung reserve thanh mang xam co van, lac
    # tong" da thay trong ket qua that. build_index_based_lock_schedule tu suy tu chinh `timesteps`
    # nen luon dung ti le bat ke so buoc/model nao.
    lock_schedule = build_index_based_lock_schedule(timesteps, lock_frac=args.lock_frac, anneal_frac=args.anneal_frac)

    z_known_patch = encode_flat_patch(ae, aux_device, ae_dtype)  # (1, C, h_small, w_small), tren aux_device
    z_known = z_known_patch.mean(dim=(0, 2, 3)).to(device=device, dtype=torch.bfloat16)  # (C,) -- gia tri
    # kenh trung binh, chuyen sang `device`+bfloat16 vi day la dung chung voi img latent (bfloat16,
    # tren device cua DiT) trong denoise_reserve, khong phai dua thang vao AE.

    def decode_and_save(latent_tokens: torch.Tensor, path: Path):
        # latent_tokens: (1, L, C) tren `device` (DiT) -> ve lai (1, C, h_lat, w_lat), chuyen sang
        # aux_device + dung dtype cua AE truoc khi goi ae.decode (AE nam o aux_device).
        z = latent_tokens[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
        with torch.no_grad():
            x = ae.decode(z).float()
        x = ((x[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(x).save(path)

    # QUAN TRONG: ca 2 nhanh denoise PHAI chay trong torch.no_grad() -- day chinh la nguyen nhan
    # OOM that tren server (model 4B, seq len ~6656 token, chi 5+20=25 block, KHONG the tu no can
    # 23GB neu inference thuan tuy; thieu no_grad khien autograd giu nguyen do thi tinh toan qua
    # het 25 block x 4 buoc, dung y het cai bay da gap va sua trong scripts/cli.py:453/`with
    # torch.no_grad():` bao quanh toan bo generation, khong chi ae.encode/decode).
    with torch.no_grad():
        print("Chay BASELINE (denoise goc, khong sua)...")
        out_baseline = denoise_baseline(model, img.clone(), img_ids, ctx, ctx_ids, timesteps, guidance=guidance)
        decode_and_save(out_baseline, out_dir / "baseline.png")
        del out_baseline
        if device.startswith("cuda"):
            torch.cuda.empty_cache()  # giai phong cache truoc khi chay nhanh thu 2, bien an toan them

        print("Chay RESERVE (Co che A -- denoise_reserve)...")
        out_reserve = denoise_reserve(
            model, img.clone(), img_ids, ctx, ctx_ids, timesteps, guidance=guidance,
            region_mask=region_mask, z_known=z_known, lock_schedule=lock_schedule,
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
        "model": model_name, "prompt": args.prompt, "product_phrase": args.product_phrase,
        "region": args.region, "seed": args.seed, "num_steps": num_steps, "guidance": guidance,
        "latent_shape": [C, h_lat, w_lat],
        "lock_frac": args.lock_frac, "anneal_frac": args.anneal_frac, "timesteps": timesteps,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"XONG. Xem {out_dir}/baseline.png vs {out_dir}/reserve.png (doi chieu voi region_debug.png).")


if __name__ == "__main__":
    main()
