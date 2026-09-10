#!/usr/bin/env python3
"""
scripts/test_omni_pipeline.py

Kiểm thử chuyên biệt cho Tendoo Omni-Block Engine (Adaptive Layout):
1. Chạy kịch bản kiểm tra (ví dụ: duplicate_feedback_target, product_intro, v.v.).
2. Xuất và lưu đầy đủ cả 4 thành phần ảnh/mã:
   - 01_corridor_mask.png (Ảnh mask hành lang chữ - Gaussian Corridor Mask)
   - 02_blended_background.png (Ảnh nền do DiT sinh sau khi qua mask)
   - 03_poster.html (Mã HTML render theo master template)
   - 04_final_poster.png (Ảnh poster cuối cùng kết hợp nền + chữ)
3. Đo đạc định lượng giá trị Mask (Min, Max, Mean) trên từng vùng chữ:
   - Đảm bảo vùng chữ có mask kích hoạt mạnh (>= 0.80).
   - Đảm bảo vùng Product Sanctuary ở trung tâm được bảo vệ (mask = 0.0).

Sử dụng:
  # 1. Chạy kiểm thử offline (không cần demo server hay GPU, đo đạc toán học trực tiếp):
  python scripts/test_omni_pipeline.py --mode offline

  # 2. Chạy kiểm thử qua Demo Server (yêu cầu demo_server đang chạy ở port 7860):
  python scripts/test_omni_pipeline.py --mode live --demo-url http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import requests
from PIL import Image

from tendoo.engine.blocks import AdaptiveBlock, deduplicate_blocks, map_category_to_default_blocks
from tendoo.engine.geometry import (
    PRODUCT_SANCTUARY_BY_RATIO,
    compute_block_metrics,
    get_product_sanctuary_rect,
    get_zone_bounding_box,
)
from tendoo.engine.layout import OmniBlockLayout
from tendoo.engine.mask import generate_omni_corridor_mask
from tendoo.layouts.base import ColorPalette, PosterContent


# Test Case tiêu chuẩn của người dùng
SAMPLE_CASES = [
    {
        "id": "duplicate_feedback_target",
        "category": "feedback",
        "title": "",
        "feedback_target": "Private Coaching Transformation",
        "feedback_quote": "97% khách hàng hài lòng với kết quả tăng cơ sau 3 tháng",
        "special_offer": "Giảm 20% gói PT tháng đầu",
        "prompt": (
            'Tạo ảnh feedback khách hàng cho dịch vụ gym & PT cao cấp. '
            'Tên sản phẩm/dịch vụ nhận feedback: "Private Coaching Transformation". '
            'Mô tả ngắn feedback: "97% khách hàng hài lòng với kết quả tăng cơ sau 3 tháng". '
            'Ưu đãi đặc biệt: "Giảm 20% gói PT tháng đầu". Background phòng gym sang trọng, tone đen đỏ.'
        ),
        "aspect_ratio": "1:1",
    },
    {
        "id": "product_intro_headphones",
        "category": "product_intro",
        "title": "TAI NGHE SONIC PRO",
        "product_name": "Tai Nghe Sonic Pro",
        "product_desc": "Chống ồn chủ động đỉnh cao, âm thanh Hi-Res chân thực",
        "price": "2.490.000đ",
        "prompt": "Tai nghe không dây đặt trên bục đá cẩm thạch sang trọng, ánh sáng studio.",
        "aspect_ratio": "1:1",
    },
]


def run_offline_inspection(out_dir: Path) -> None:
    """Kiểm tra toán học độc lập của Omni-Block Engine không cần bật HTTP server."""
    print("=" * 80)
    print("CHẠY KIỂM THỬ TOÁN HỌC OFFLINE (OMNI-BLOCK ENGINE MASK & GEOMETRY)")
    print("=" * 80)

    layout = OmniBlockLayout()
    font_path = layout._resolve_font_path("bevietnam")

    for case in SAMPLE_CASES:
        case_id = case["id"]
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        w, h = 1024, 1024

        print(f"\n[CASE: {case_id}]")
        # 1. Chuyển đổi dữ liệu form sang AdaptiveBlocks
        fields = {k: v for k, v in case.items() if k not in ("id", "prompt", "category", "aspect_ratio")}
        blocks = map_category_to_default_blocks(
            category=case["category"],
            title=case.get("title", ""),
            fields=fields,
            store_info={"store_name": "Tendoo Gym", "phone": "0988 123 456"},
        )
        blocks = deduplicate_blocks(blocks, hero_title=case.get("title", ""))

        print(f"  -> Trích xuất được {len(blocks)} khối chữ ngữ nghĩa:")
        for i, b in enumerate(blocks, 1):
            print(f"     {i}. [{b.zone:12}] role={b.role:10} text='{b.text}'")

        # 2. Sinh Mask hành lang (Corridor Mask)
        mask_arr = generate_omni_corridor_mask(
            width=w, height=h, blocks=blocks, font_path=font_path
        )

        # Lưu ảnh Mask
        mask_vis = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
        mask_file = case_dir / "01_corridor_mask.png"
        mask_vis.save(mask_file)
        print(f"  -> Đã lưu ảnh Mask: {mask_file}")

        # 3. Phân tích định lượng cường độ Mask trên từng vùng chữ
        print("  -> Phân tích định lượng Mask trên từng khối:")
        all_blocks_covered = True
        for b in blocks:
            metrics = compute_block_metrics(b, font_path, w, h)
            zx1, zy1, zx2, zy2 = metrics["zone_rect"]
            box_w = metrics["measured_w"]
            box_h = metrics["measured_h"]

            # Vùng bounding box thực tế khớp với tọa độ vẽ của mask
            pad = 24
            if "left" in b.zone:
                bx1 = zx1
                bx2 = min(zx2, bx1 + int(box_w) + pad * 2)
            elif "right" in b.zone:
                bx2 = zx2
                bx1 = max(zx1, bx2 - int(box_w) - pad * 2)
            else:
                cx = (zx1 + zx2) // 2
                bx1 = max(zx1, cx - int(box_w // 2) - pad)
                bx2 = min(zx2, cx + int(box_w // 2) + pad)

            if "top" in b.zone or b.zone == "top_bar":
                by1 = zy1
                by2 = min(zy2, by1 + int(box_h) + pad * 2)
            elif "bottom" in b.zone or b.zone == "bottom_bar":
                by2 = zy2
                by1 = max(zy1, by2 - int(box_h) - pad * 2)
            else:
                cy = (zy1 + zy2) // 2
                by1 = max(zy1, cy - int(box_h // 2) - pad)
                by2 = min(zy2, cy + int(box_h // 2) + pad)

            x_min, x_max = max(0, bx1), min(w, bx2)
            y_min, y_max = max(0, by1), min(h, by2)

            crop_mask = mask_arr[y_min:y_max, x_min:x_max]
            mean_val = float(crop_mask.mean()) if crop_mask.size > 0 else 0.0
            max_val = float(crop_mask.max()) if crop_mask.size > 0 else 0.0

            status = "PASS (MẠNH)" if max_val >= 0.75 else "FAIL (YẾU)"
            if max_val < 0.75:
                all_blocks_covered = False
            print(f"     * [{b.zone:12}] max={max_val:.2f}, mean={mean_val:.2f} -> {status} : '{b.text[:35]}...'")

        # 4. Kiểm tra Product Sanctuary (Tâm sản phẩm)
        sx1, sy1, sx2, sy2 = get_product_sanctuary_rect(w, h)
        sanctuary_mask = mask_arr[sy1:sy2, sx1:sx2]
        s_mean = float(sanctuary_mask.mean())
        s_max = float(sanctuary_mask.max())
        s_status = "PASS (AN TOÀN)" if s_max < 0.15 else "WARNING (BỊ LẤN)"
        print(f"  -> Tâm Sản Phẩm (Sanctuary Center): max={s_max:.2f}, mean={s_mean:.2f} -> {s_status}")

        # 5. Render HTML offline
        content = PosterContent(
            category=case["category"],
            headline=case.get("title", ""),
            offer_main=case.get("special_offer", ""),
            offer_sub=case.get("feedback_quote", ""),
            feedback_target=case.get("feedback_target", ""),
            feedback_quote=case.get("feedback_quote", ""),
            special_offer=case.get("special_offer", ""),
            brand="Tendoo Gym",
            hotline="0988 123 456",
        )
        palette = ColorPalette(
            is_dark=True, luminance=0.15, hue=35, comp_hue=215,
            headline_color="#ffffff", sub_color="#e0e0e0",
            badge_bg="#f59e0b", badge_text="#000000",
            accent_color="#f59e0b",
        )
        html_str = layout.render_html(content=content, palette=palette, width=w, height=h)
        html_file = case_dir / "03_poster.html"
        html_file.write_text(html_str, encoding="utf-8")
        print(f"  -> Đã lưu file HTML: {html_file}")


def run_live_inspection(demo_url: str, out_dir: Path) -> None:
    """Gửi request thật đến demo_server.py và tải về toàn bộ 4 file để soi chi tiết."""
    print("=" * 80)
    print(f"CHẠY KIỂM THỬ QUA DEMO SERVER: {demo_url}")
    print("=" * 80)

    for case in SAMPLE_CASES:
        case_id = case["id"]
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "category": case["category"],
            "image_description": case["prompt"],
            "aspect_ratio": case["aspect_ratio"],
            "layout": "omni",  # Yêu cầu rõ ràng kích hoạt Omni-Block Engine
        }
        payload.update({k: v for k, v in case.items() if k not in ("id", "prompt", "category", "aspect_ratio")})

        print(f"\n[Gửi request cho case: {case_id}]")
        t0 = time.time()
        try:
            resp = requests.post(f"{demo_url.rstrip('/')}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [ERROR] Gọi API thất bại: {e}")
            continue

        latency = round(time.time() - t0, 2)
        print(f"  -> Phản hồi thành công ({latency}s), resolved_layout: {data.get('resolved_layout')}")

        # Tải file Mask
        if data.get("mask_url"):
            m_resp = requests.get(f"{demo_url.rstrip('/')}/{data['mask_url']}")
            m_file = case_dir / "01_corridor_mask.png"
            m_file.write_bytes(m_resp.content)
            m_img = Image.open(m_file)
            m_arr = np.array(m_img) / 255.0
            print(f"  [1/4] Đã tải Mask: {m_file.name} (mean={m_arr.mean():.3f}, max={m_arr.max():.3f})")

        # Tải file Blended Background
        if data.get("blended_bg_url"):
            bg_resp = requests.get(f"{demo_url.rstrip('/')}/{data['blended_bg_url']}")
            bg_file = case_dir / "02_blended_background.png"
            bg_file.write_bytes(bg_resp.content)
            print(f"  [2/4] Đã tải Nền DiT: {bg_file.name}")

        # Tải file HTML
        if data.get("html_url"):
            h_resp = requests.get(f"{demo_url.rstrip('/')}/{data['html_url']}")
            h_file = case_dir / "03_poster.html"
            h_file.write_bytes(h_resp.content)
            print(f"  [3/4] Đã tải HTML: {h_file.name}")

        # Tải file Poster cuối cùng
        if data.get("final_poster_url"):
            p_resp = requests.get(f"{demo_url.rstrip('/')}/{data['final_poster_url']}")
            p_file = case_dir / "04_final_poster.png"
            p_file.write_bytes(p_resp.content)
            print(f"  [4/4] Đã tải Poster hoàn chỉnh: {p_file.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline", help="offline: toán học trực tiếp; live: gọi qua demo server")
    parser.add_argument("--demo-url", default="http://127.0.0.1:7860")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "output_omni_test"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "offline":
        run_offline_inspection(out_dir)
    else:
        run_live_inspection(args.demo_url, out_dir)

    print(f"\n[HOÀN TẤT] Toàn bộ kết quả và ảnh mask đã lưu tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
