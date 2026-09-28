#!/usr/bin/env python3
"""
scripts/check_server.py -- TỰ KIỂM TRA máy chủ GPU trước khi chạy demo (docs/HUONG_DAN_SERVER.md, bước 3).

Không tải model lên GPU, không sinh ảnh: chỉ kiểm tra những thứ hay làm demo hỏng âm thầm (rơi về Mock Mode):
  GPU, file trọng số (distill / VAE / text encoder / tokenizer), transformers, bộ mã hoá chữ FP8 trên A30,
  Chromium cho Playwright, font, LLM endpoint, cổng.

  cd ~/work/tendoo-v3 && python scripts/check_server.py [--model_path ~/persistent-data/FLUX.2-klein-4B] [--port 8088]
Mã thoát 0 = đủ để chạy GPU thật; 1 = có mục LỖI (đọc dòng "SỬA:").
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ROOTS = [Path(os.path.expanduser(f"~/persistent-data/{d}")) for d in ("FLUX.2-klein-4B", "FLUX.2-klein-base-4B")] + \
        [Path(f"{r}/persistent-data/{d}") for r in ("/home/jovyan", "") for d in ("FLUX.2-klein-4B", "FLUX.2-klein-base-4B")]
fails: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ĐẠT] {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"  [LỖI] {msg}\n        SỬA: {fix}")
    fails.append(msg)


def warn(msg: str) -> None:
    print(f"  [CHÚ Ý] {msg}")


def find(rel: str, extra: list[Path]) -> Path | None:
    for r in extra + ROOTS:
        if (r / rel).exists():
            return r / rel
    return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None, help="thư mục hoặc file flux-2-klein-4b.safetensors (nếu không ở chỗ mặc định)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("TENDOO_V3_SERVER_PORT", "8088")))
    args = ap.parse_args()
    extra = []
    if args.model_path:
        p = Path(os.path.expanduser(args.model_path))
        extra = [p if p.is_dir() else p.parent]

    print("1. GPU")
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n == 0:
            bad(f"torch {torch.__version__} không thấy GPU", "cài torch bản CUDA; kiểm tra nvidia-smi")
        else:
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                ok(f"cuda:{i} {p.name} {p.total_memory / 2**30:.0f}GB (compute {p.major}.{p.minor})")
            if n < 2:
                warn("chỉ 1 GPU -- mọi model lên cùng 1 card (24GB có thể chật khi dùng LLM cục bộ)")
    except Exception as e:
        bad(f"không import được torch ({e})", "pip install -r requirements.txt")

    print("2. File trọng số (bản DISTILL)")
    dit = Path(os.path.expanduser(args.model_path)) if args.model_path and Path(os.path.expanduser(args.model_path)).is_file() \
        else find("flux-2-klein-4b.safetensors", extra)
    if dit:
        ok(f"DiT distill: {dit} ({dit.stat().st_size / 2**30:.1f}GB)")
    else:
        base = find("flux-2-klein-base-4b.safetensors", extra)
        bad("KHÔNG thấy flux-2-klein-4b.safetensors (bản distill)" + (f"; chỉ có bản base {base}" if base else ""),
            "đặt file vào ~/persistent-data/FLUX.2-klein-4B/ hoặc chạy demo với --model_path <đường dẫn file>")
    vae = find("vae/diffusion_pytorch_model.safetensors", extra)
    ok(f"VAE: {vae}") if vae else bad("không thấy vae/diffusion_pytorch_model.safetensors", "chép thư mục vae/ của FLUX.2-klein")
    te = next((r / "text_encoder" for r in extra + ROOTS if (r / "text_encoder" / "config.json").exists()), None)
    if te:
        ok(f"Text encoder: {te}")
        tok = te.parent / "tokenizer"
        ok(f"Tokenizer: {tok}") if (tok / "tokenizer_config.json").exists() or (tok / "tokenizer.json").exists() else \
            bad(f"không thấy tokenizer cạnh {te}", "chép thư mục tokenizer/ cùng cấp text_encoder/")
        cfg = json.loads((te / "config.json").read_text(encoding="utf-8"))
        q = cfg.get("quantization_config") or {}
        if q:
            warn(f"text encoder lượng tử hoá {q.get('quant_method')} ({q.get('fmt') or q.get('weight_block_size') or ''}) -- "
                 "A30 (compute 8.0) không có FP8 phần cứng; nếu demo log 'Failed to load GPU models' ở bước TextEncoder -> xem mục Sự cố")
    else:
        bad("không thấy text_encoder/config.json", "chép thư mục text_encoder/ (Qwen3-4B) của FLUX.2-klein")

    print("3. Thư viện")
    try:
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        ok(f"transformers {transformers.__version__}") if v >= (4, 51) else \
            bad(f"transformers {transformers.__version__} quá cũ (cần >= 4.51 cho Qwen3/Mistral3)", "pip install 'transformers>=4.51'")
    except Exception as e:
        bad(f"không import được transformers ({e})", "pip install -r requirements.txt")
    for mod in ("fastapi", "uvicorn", "jinja2", "playwright", "qrcode", "einops", "safetensors", "dotenv"):
        try:
            __import__(mod)
        except Exception:
            bad(f"thiếu thư viện {mod}", "pip install -r requirements.txt")
    try:
        from flux2 import util  # noqa: F401
        ok("import flux2 được")
    except BaseException as e:
        bad(f"import flux2 lỗi ({e!r})", "xem thông báo; thường do transformers cũ")

    print("4. Chromium (dựng chữ poster)")
    chrome = os.environ.get("PLAYWRIGHT_CHROME_PATH") or next(
        (shutil.which(b) for b in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable") if shutil.which(b)), None)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            kw = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]}
            if chrome:
                kw["executable_path"] = chrome
            b = p.chromium.launch(**kw)
            pg = b.new_page()
            pg.set_content("<p style='font-size:40px'>Ệ Ợ Ữ</p>")
            b.close()
        ok(f"Chromium chạy được ({chrome or 'bản Playwright tải về'})")
    except Exception as e:
        bad(f"không mở được Chromium ({str(e).splitlines()[0][:160]})",
            "python -m playwright install chromium  (cần mạng; máy offline: chép thư mục ~/.cache/ms-playwright từ máy khác, "
            "hoặc đặt PLAYWRIGHT_CHROME_PATH=<đường dẫn chromium có sẵn>); thiếu thư viện hệ thống: python -m playwright install-deps")

    print("5. Font")
    n_fonts = len(list((PROJECT_ROOT / "fonts").glob("*.*")))
    ok(f"{n_fonts} file font trong fonts/") if n_fonts >= 40 else bad(f"chỉ {n_fonts} file font", "git pull lại (font nằm trong repo)")

    print("6. LLM lập kế hoạch")
    from tendoo_v3 import llm_planner as lp  # nạp .env
    backend = os.environ.get("TENDOO_V3_LLM_BACKEND", "auto")
    print(f"  backend={backend}  base_url={lp.LLM_BASE_URL}  model={lp.LLM_MODEL}  key={'có' if lp.LLM_API_KEY else 'không'}")
    if backend in ("api", "auto"):
        try:
            import requests
            r = requests.get(lp.LLM_BASE_URL.rstrip("/") + "/models", timeout=5,
                             headers={"Authorization": f"Bearer {lp.LLM_API_KEY}"} if lp.LLM_API_KEY else {})
            ok(f"endpoint trả HTTP {r.status_code}") if r.status_code < 500 else bad(f"endpoint lỗi HTTP {r.status_code}", "kiểm tra dịch vụ LLM")
            if not lp.LLM_API_KEY and not os.environ.get("TENDOO_V3_LLM_ALLOW_NO_KEY"):
                warn("chưa có API key -> hệ thống BỎ QUA gọi API; endpoint không cần key thì đặt TENDOO_V3_LLM_ALLOW_NO_KEY=1")
        except Exception as e:
            msg = f"không kết nối được {lp.LLM_BASE_URL} ({type(e).__name__})"
            if backend == "api":
                bad(msg, "đổi TENDOO_V3_LLM_BASE_URL trong .env, hoặc dùng TENDOO_V3_LLM_BACKEND=local")
            else:
                warn(msg + " -> auto sẽ dùng Qwen cục bộ")
    if backend in ("local", "auto"):
        path = lp.resolve_qwen3_checkpoint_path()
        if os.path.exists(path):
            ok(f"Qwen cục bộ: {path}")
        elif backend == "local":
            bad(f"không thấy Qwen cục bộ ({path})", "chép text_encoder/ + tokenizer/ của FLUX.2-klein")
        else:
            warn(f"không thấy Qwen cục bộ ({path}) -> plan dự phòng theo quy tắc")

    print("7. Cổng")
    with socket.socket() as s:
        busy = s.connect_ex(("127.0.0.1", args.port)) == 0
    warn(f"cổng {args.port} đang có tiến trình khác dùng") if busy else ok(f"cổng {args.port} còn trống")

    print("\n" + ("SẴN SÀNG chạy GPU thật." if not fails else f"CÒN {len(fails)} LỖI -- sửa theo dòng 'SỬA:' rồi chạy lại."))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
