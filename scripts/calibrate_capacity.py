#!/usr/bin/env python3
"""
scripts/calibrate_capacity.py

Đo SỨC CHỨA thật của từng template × tỉ lệ khung hình (ROADMAP §3.4, GĐ 1): tăng dần lượng
chữ theo một "thang nội dung" tất định, render qua Chromium ở mỗi bậc, tìm ĐIỂM GÃY.

  capacity_safe       = số ký tự ở bậc cuối cùng KHÔNG mất chữ (Cổng 4, §4.5 điều kiện 4)
  capacity_aesthetic  = số ký tự ở bậc cuối cùng vừa không mất chữ VỪA giữ tương phản điểm neo
                        >= ngưỡng intent mặc định của template (§4.5 điều kiện 1) -- đây là
                        `capacity_chars` điền vào catalog (Cổng 3 dùng để định tuyến)

Điểm gãy = bậc cuối trước lần TRƯỢT ĐẦU TIÊN (không nhảy cóc qua bậc trượt). 0 = trượt ngay
bậc 1 (chỉ có tiêu đề ngắn) -> template không đạt profile intent ở bất kỳ lượng chữ nào.

Giới hạn đã biết (ghi vào kết quả): đo với 1 style cố định (Be Vietnam Pro, plain_elegant,
dark_luxury) và hero PHẲNG (không hero_parts) -- font hẹp/rộng và markup Luật 2 đổi sức chứa.

  PYTHONPATH=src python scripts/calibrate_capacity.py --out output_probe/capacity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

from probe_type_hierarchy import measure_plan
from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.validators import CONTENT_FIELDS

ASPECTS = {"1:1": (1024, 1024), "9:16": (576, 1024), "16:9": (1024, 576), "4:5": (816, 1024)}
STYLE = StyleConfig(font="bevietnam", theme_color="#F59E0B", text_effect="plain_elegant", background_tone="dark_luxury")

HERO = ["SIÊU SALE 50%", "ƯU ĐÃI THÁNG 9 GIẢM TỚI 50%", "ĐẠI TIỆC ƯU ĐÃI MÙA THU GIẢM TỚI 50% TOÀN BỘ SẢN PHẨM"]
SUBHEAD = ["Áp dụng toàn hệ thống tới hết chủ nhật",
           "Áp dụng cho toàn bộ hệ thống cửa hàng trên toàn quốc từ nay tới hết ngày 30 tháng 9"]
EXTRA = ["Miễn phí giao hàng nội thành", "Tặng quà cho 100 khách đầu tiên", "Tích điểm đổi quà hấp dẫn",
         "Bảo hành chính hãng 12 tháng", "Trả góp lãi suất 0%", "Đổi trả miễn phí trong 7 ngày"]
MENU = ["Cà phê sữa đá - 29.000đ", "Bạc xỉu - 32.000đ", "Trà đào cam sả - 39.000đ",
        "Trà vải hoa hồng - 42.000đ", "Cold brew nguyên chất - 45.000đ", "Bánh sừng bò bơ - 35.000đ"]
STEPS = ["Tư vấn 1:1 miễn phí", "Lên phác đồ riêng", "Thực hiện liệu trình", "Theo dõi và bảo hành 12 tháng"]
STORE = ["Hotline: 1900 8888", "123 Nguyễn Huệ, Quận 1, TP.HCM", "Mở cửa: 8:00 - 22:00"]

# Thang nội dung: mỗi bậc là tập field -> giá trị; chỉ áp field template THỰC SỰ hiển thị (slots).
LADDER = [
    {"hero": HERO[0]},
    {"subhead": SUBHEAD[0]},
    {"cta": "ĐẶT HÀNG NGAY"},
    {"badge": "ƯU ĐÃI CÓ HẠN"},
    {"store_info": STORE[0], "qr_code": "https://tendoo.ai/promo"},
    {"extra_texts": EXTRA[:2]},
    {"hero": HERO[1]},
    {"store_info": " • ".join(STORE)},
    {"extra_texts": EXTRA[:4]},
    {"subhead": SUBHEAD[1]},
    {"extra_texts": EXTRA[:6]},
    {"hero": HERO[2]},
    # Bậc quá tải: tìm điểm gãy AN TOÀN (bản đầu dừng ở ~400 ký tự mà hầu hết chưa mất chữ).
    {"extra_texts": [e + " cho mọi đơn hàng trong tháng" for e in EXTRA]},
    {"store_info": " • ".join(STORE + ["Chi nhánh 2: 45 Lê Lợi, Quận 3, TP.HCM"])},
    {"subhead": SUBHEAD[1] + ", số lượng quà tặng có hạn, nhanh tay đặt hàng để nhận ưu đãi sớm nhất"},
    {"hero": HERO[2] + " VÀ NHIỀU QUÀ TẶNG"},
]
# Field chuyên biệt: nội dung chính của template, tăng dần cùng thang (menu/steps) hoặc cố định.
SPECIAL = {
    "menu_price_board": lambda lvl: {"extra_texts": MENU[: min(6, 2 + lvl // 2)]},
    "step_process_roadmap": lambda lvl: {"steps": STEPS[: min(4, 2 + lvl // 3)]},
    "customer_feedback_card": lambda lvl: {"testimonial": "Dịch vụ tuyệt vời, nhân viên tận tâm, tôi sẽ quay lại.",
                                           "reviewer_name": "Ngọc Lan - Khách hàng thân thiết", "rating": 5},
    "before_after_split": lambda lvl: {"tag_left": "TRƯỚC", "tag_right": "SAU", "rating": 5},
}


def plan_at(template: str, level: int) -> TendooCreativePlan:
    slots = TEMPLATE_CATALOG[template]["slots"]
    fields = {}
    for step in LADDER[: level + 1]:
        fields.update({k: v for k, v in step.items() if k in slots or k == "qr_code" and "qr_code" in slots})
    if template in SPECIAL:
        fields.update({k: v for k, v in SPECIAL[template](level).items() if k in slots})
    return replace(TendooCreativePlan(template=template, hero=fields.pop("hero"), style=STYLE), **fields)


def content_chars(plan: TendooCreativePlan) -> int:
    n = 0
    for f in CONTENT_FIELDS:
        v = getattr(plan, f, None)
        if f in ("qr_code", "rating") or not v:
            continue
        n += sum(len(x) for x in v) if isinstance(v, list) else len(str(v))
    return n


def calibrate(page, template: str, w: int, h: int) -> dict:
    rows, safe, aes, safe_broken, aes_broken = [], 0, 0, False, False
    for lvl in range(len(LADDER)):
        plan = plan_at(template, lvl)
        r = measure_plan(page, plan, w, h)
        chars = content_chars(plan)
        ok_safe = r["c4_no_loss"]
        ok_aes = ok_safe and r["c1_anchor"]
        rows.append({"level": lvl, "chars": chars, "contrast": r["contrast"], "text_lost": r["text_lost"], "safe": ok_safe, "aes": ok_aes})
        if not safe_broken:
            if ok_safe:
                safe = chars
            else:
                safe_broken = True
        if not aes_broken:
            if ok_aes:
                aes = chars
            else:
                aes_broken = True
    return {"capacity_safe": safe, "capacity_aesthetic": aes, "target": r["contrast_target"], "intent": r["intent"], "levels": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output_probe" / "capacity"))
    ap.add_argument("--template", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    templates = [args.template] if args.template else sorted(TEMPLATE_CATALOG)
    result, t0 = {}, time.perf_counter()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for tpl in templates:
            result[tpl] = {}
            for asp, (w, h) in ASPECTS.items():
                result[tpl][asp] = calibrate(page, tpl, w, h)
        browser.close()
    print(f"Đo xong trong {time.perf_counter() - t0:.0f}s\n")
    hdr = f"{'template':<24}{'intent':<18}{'ngưỡng':>6}  " + "  ".join(f"{a:>13}" for a in ASPECTS)
    print(hdr + "\n" + " " * 50 + "  ".join(f"{'thẩm mỹ/an toàn':>13}" for _ in ASPECTS))
    print("-" * len(hdr))
    for tpl, by in result.items():
        any_ = next(iter(by.values()))
        cells = "  ".join(f"{str(by[a]['capacity_aesthetic']) + '/' + str(by[a]['capacity_safe']):>13}" for a in ASPECTS)
        print(f"{tpl:<24}{any_['intent']:<18}{any_['target']:>6}  {cells}")
    (out / "capacity.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nKết quả: {out / 'capacity.json'}")


if __name__ == "__main__":
    main()
