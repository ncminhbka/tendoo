#!/usr/bin/env python3
"""
scripts/run_tendoo_v2_llm.py

Runs tendoo_v2's full pipeline (form fields + free prompt -> REAL hosted LLM ->
role/group/zone curation -> deterministic solver -> HTML -> PNG) end to end, per
"wire tạm v2 vào production, dùng LLM đã host ... để tôi lên server test, lỗi gì
thì fix sau" (2026-09-14). Unlike scripts/test_tendoo_v2_solver.py (hand-written
mock Blocks, no LLM), every case here is RAW form fields + a free-text prompt --
the LLM (src/tendoo_v2/llm_client.py) must do the actual curation.

Run this on the GPU server (needs real network access to the hosted LLM endpoint;
the poster screenshot itself only needs Playwright/Chromium, works anywhere):
  python scripts/run_tendoo_v2_llm.py
  python scripts/run_tendoo_v2_llm.py --model Qwen/Qwen3.8-27B
  python scripts/run_tendoo_v2_llm.py --case grand_opening_hero_subhead

Output:
  output_tendoo_v2_llm/<case_id>/poster.png, poster.html, llm_raw_response.txt
  output_tendoo_v2_llm/SUMMARY.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import tendoo_v2.llm_client as llm_client
from tendoo_v2.pipeline import run_pipeline

OUTPUT_DIR = PROJECT_ROOT / "output_tendoo_v2_llm"
WIDTH, HEIGHT = 1024, 1024

# Common store info reused across cases -- exercises the deterministic (non-LLM)
# store-info block on every case, mirroring how every real poster carries it.
_STORE = {
    "store_name": "Tendoo Coffee",
    "phone": "0988 123 456",
    "address": "45 Phố Huế, Hai Bà Trưng, Hà Nội",
}

# Each case: RAW fields (business form, may be empty/partial) + a free-text prompt --
# the LLM must decide what to draw, how to rank/group it, and whether the prompt
# names an explicit position. No role/group/zone here -- that's exactly what this
# script is testing.
CASES: List[Dict[str, Any]] = [
    {
        "id": "01_promo_dual_message",
        "desc": "2 thong diep quan trong ngang nhau, khong vi tri tuong minh -- LLM co tach 2 hero khong?",
        "fields": {**_STORE, "discount": "Giảm 60%", "special_offer": ""},
        "free_prompt": "Poster khuyến mãi, muốn 'SIÊU SALE' và 'DUY NHẤT HÔM NAY' đều thật nổi bật, ngang hàng nhau.",
    },
    {
        "id": "02_grand_opening_hero_subhead",
        "desc": "Case that tu prompt_test.txt line 43: GRAND OPENING + MUA 1 TANG 1 phai di cung nhau.",
        "fields": {**_STORE, "opening_date": "15/10/2026", "opening_promotion": "Mua 1 tặng 1"},
        "free_prompt": "Khai trương quán cà phê, chữ 'GRAND OPENING' và 'MUA 1 TẶNG 1' phải đứng liền kề nhau thành 1 khối, không tách rời.",
    },
    {
        "id": "03_product_intro_long_desc",
        "desc": "product_desc dai -- LLM co tu tach thanh nhieu block ngan (thay vi 1 card chu nho) khong?",
        "fields": {
            **_STORE, "product_name": "Tai Nghe Sonic Pro", "price": "1.990.000đ",
            "product_desc": "Chống ồn chủ động, âm thanh Hi-Res chuẩn phòng thu",
            "highlights": "ANC 45dB, Pin 40 giờ, Bluetooth 5.3",
        },
        "free_prompt": "Làm nổi bật tên sản phẩm và giá, phần mô tả nếu dài quá thì tách gọn cho dễ đọc, đừng nhồi hết vào 1 khối.",
    },
    {
        "id": "04_feedback_quote",
        "desc": "Feedback 1 cau dai -- LLM co tranh gan hero cho ca cau dai khong (nguyen tac tu render_plan.py).",
        "fields": {**_STORE, "feedback_product_name": "Serum Vitamin C", "feedback_content": "Tôi đã dùng rất nhiều loại serum nhưng chưa sản phẩm nào thấm nhanh và dịu da như thế này", "rating": "5.0/5.0"},
        "free_prompt": "Đăng feedback khách hàng, giữ nguyên văn câu feedback, có thể làm nổi bật điểm rating.",
    },
    {
        "id": "05_recruitment_explicit_zone",
        "desc": "Prompt neu ro vi tri -- kiem tra LLM trich xuat zone dung, khong tu bia them zone khac.",
        "fields": {**_STORE, "job_position": "Nhân viên Pha chế", "job_desc": "Làm việc ca linh hoạt, môi trường trẻ trung", "deadline": "30/09/2026"},
        "free_prompt": "Tuyển dụng vị trí pha chế, đặt tên vị trí tuyển dụng ở góc trên bên trái, hạn nộp hồ sơ ở góc dưới bên phải.",
    },
    {
        "id": "06_guide_steps_grouped",
        "desc": "3 buoc huong dan -- kiem tra LLM co gom group de solver xep thanh 1 cot khong.",
        "fields": {**_STORE, "steps": "Bước 1: Quét mã QR để bắt đầu\nBước 2: Chọn sản phẩm và nhập mã ưu đãi\nBước 3: Nhận hàng hoả tốc trong 2 giờ"},
        "free_prompt": "Hướng dẫn 3 bước sử dụng ưu đãi, giữ đúng thứ tự, các bước nên đứng thành 1 cụm.",
    },
    {
        "id": "07_sparse_prompt_only",
        "desc": "Form gan nhu trong, chi co prompt tu do mo ta y muon -- LLM co tu suy ra noi dung tu prompt khong.",
        "fields": {**_STORE},
        "free_prompt": "Poster khai trương quán, tôi muốn có dòng chữ 'ƯU ĐÃI ĐẶC BIỆT' thật to, không cần thêm gì khác.",
    },
    {
        "id": "08_empty_everything",
        "desc": "Khong field, khong prompt -- kiem tra pipeline khong crash, fallback ve chi store-info.",
        "fields": {**_STORE},
        "free_prompt": "",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Override TENDOO_V2_LLM_MODEL default")
    parser.add_argument("--case", type=str, default=None, help="Run only this case id (substring match)")
    args = parser.parse_args()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = args.model or llm_client.LLM_MODEL
    cases = [c for c in CASES if not args.case or args.case in c["id"]]

    print("=" * 90)
    print("TENDOO_V2 + HOSTED LLM END-TO-END RUN")
    print(f"  Endpoint : {llm_client.LLM_BASE_URL}")
    print(f"  Model    : {model}")
    print(f"  Log      : {llm_client.LLM_LOG_PATH}")
    print("=" * 90)

    md = ["# Bao cao tendoo_v2 + LLM da host\n\n", f"Model: `{model}`\n\n"]
    md.append("| Case | Mo ta | So block LLM | Loi LLM | Anh |\n|---|---|---|---|---|\n")

    n_ok, n_llm_errors = 0, 0
    for i, case in enumerate(cases, 1):
        print(f"\n[{i:02d}/{len(cases)}] {case['id']} -- {case['desc'][:80]}")
        case_dir = OUTPUT_DIR / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_pipeline(
                fields=case["fields"], free_prompt=case["free_prompt"],
                width=WIDTH, height=HEIGHT, model=model,
                output_image_path=case_dir / "poster.png",
            )
            (case_dir / "poster.html").write_text(result["html"], encoding="utf-8")
            blocks_repr = "\n".join(f"  - [{b.role}] group={b.group} zone={b.zone} field={b.field}: {b.text}" for b in result["blocks"])
            (case_dir / "blocks.txt").write_text(blocks_repr, encoding="utf-8")

            print(f"   -> {result['llm_block_count']} block(s) tu LLM, {len(result['blocks'])} block tong (kem store-info)")
            if result["llm_errors"]:
                n_llm_errors += 1
                print(f"   -> LOI LLM ({len(result['llm_errors'])}):")
                for e in result["llm_errors"]:
                    print(f"      - {e}")
            else:
                print("   -> Khong loi LLM")
            n_ok += 1
            md.append(f"| `{case['id']}` | {case['desc'][:70]} | {result['llm_block_count']} | {len(result['llm_errors'])} | {case_dir / 'poster.png'} |\n")
        except Exception as e:
            print(f"   -> LOI PIPELINE: {e}")
            import traceback
            traceback.print_exc()
            md.append(f"| `{case['id']}` | {case['desc'][:70]} | - | CRASH: {e} | - |\n")

    (OUTPUT_DIR / "SUMMARY.md").write_text("".join(md), encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"HOAN TAT: {n_ok}/{len(cases)} case chay xong, {n_llm_errors} case co loi LLM (xem log: {llm_client.LLM_LOG_PATH}).")
    print(f"Bao cao: {OUTPUT_DIR / 'SUMMARY.md'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
