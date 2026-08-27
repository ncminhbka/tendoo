#!/usr/bin/env python3
"""
Test script to render diverse text lengths using Tendoo Production GlyphEngine.
Saves rendered bitmaps both to repo tests/ and directly to the agent artifact directory for UI display.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tendoo.glyph_engine import render_glyph

ARTIFACT_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\7127dfb8-1b26-4b69-ad04-fa7b65c2dd63\test_glyphs")
LOCAL_TEST_DIR = PROJECT_ROOT / "tests" / "rendered_glyphs"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_TEST_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {
        "id": "1_ultra_short_badge",
        "label": "1. Ultra-Short Badge (CTA 2 từ)",
        "text": "GIẢM 50%",
        "font": "pacifico",
        "single_line": True,
        "desc": "Badge ưu đãi nhỏ xinh, font cọ uốn lượn, 1 dòng ngắn tối ưu token",
    },
    {
        "id": "2_short_headline",
        "label": "2. Short Headline (Tiêu đề 4 từ)",
        "text": "CÀ PHÊ SỮA ĐÁ",
        "font": "bevietnam",
        "single_line": True,
        "desc": "Tiêu đề F&B kinh điển, font Sans hiện đại đậm nét",
    },
    {
        "id": "3_medium_slogan",
        "label": "3. Medium Slogan (Slogan 6 từ / 2 dòng)",
        "text": "ĐỈNH CAO CÔNG NGHỆ 5G VIETTEL",
        "font": "gotham",
        "single_line": False,
        "desc": "Slogan công nghệ Telecom, font Geometric Ultra-Bold, chia 2 dòng cân đối",
    },
    {
        "id": "4_luxury_fashion",
        "label": "4. Luxury Title (Thời trang 7 từ / 2 dòng)",
        "text": "BỘ SƯU TẬP THỜI TRANG CAO CẤP",
        "font": "playfair",
        "single_line": False,
        "desc": "Tiêu đề thời trang sang trọng, font Serif cổ điển, dấu tiếng Việt tròn đầy",
    },
    {
        "id": "5_delicate_spa",
        "label": "5. Delicate Script (Spa/Mỹ phẩm 8 từ / 2 dòng)",
        "text": "Chăm sóc làn da thuần tự nhiên mỗi ngày",
        "font": "dancing",
        "single_line": False,
        "desc": "Thông điệp mỹ phẩm tự nhiên, font Cursive mềm mại (Floor 48pt)",
    },
    {
        "id": "6_tester_feedback_quote",
        "label": "6. Tester Feedback Quote (Đoạn trích dẫn dài 25 từ)",
        "text": "97% khách hàng hài lòng với kết quả tăng cơ, cải thiện vóc dáng và sức khỏe chỉ sau 3 tháng tập luyện cùng PT riêng",
        "font": "bevietnam",
        "single_line": False,
        "desc": "Trích từ Prompt 11 của tester: Card Feedback Gym PT, đoạn văn 4 dòng tự động wrap",
    },
    {
        "id": "7_poem_4lines",
        "label": "7. Classic Poem (Bài thơ 4 câu / 28 từ / 119 ký tự)",
        "text": "Sông Mã xa rồi Tây Tiến ơi\nNhớ về rừng núi nhớ chơi vơi.\nSài Khao sương lấp đoàn quân mỏi,\nMường Lát hoa về trong đêm hơi.",
        "font": "playfair",
        "single_line": False,
        "desc": "Bài thơ Tây Tiến, 4 dòng ngắt thủ công bằng '\\n', kiểm chứng trần dung lượng",
    },
]

def main():
    print("=" * 100)
    print(" [*] TENDOO GLYPH ENGINE - COMPREHENSIVE MULTI-LENGTH BENCHMARK")
    print("=" * 100)

    results = []

    for item in TEST_CASES:
        info = render_glyph(
            text=item["text"],
            font_name_or_path=item["font"],
            force_single_line=item["single_line"],
            auto_size=True,
        )

        artifact_file = ARTIFACT_DIR / f"{item['id']}.png"
        local_file = LOCAL_TEST_DIR / f"{item['id']}.png"

        info.image.save(str(artifact_file))
        info.image.save(str(local_file))

        res = {
            "id": item["id"],
            "label": item["label"],
            "text": item["text"],
            "font": info.font_name.upper(),
            "archetype": info.archetype,
            "font_size": info.font_size_pt,
            "min_floor": info.min_floor_pt,
            "width": info.width_px,
            "height": info.height_px,
            "latent": f"{info.latent_w}x{info.latent_h}",
            "tokens": info.token_count,
            "lines": len(info.lines),
            "line_list": info.lines,
            "nyquist": "PASS (Silk-Smooth)" if info.is_nyquist_safe else "SUB-NYQUIST",
            "artifact_path": str(artifact_file),
            "desc": item["desc"],
        }
        results.append(res)

        print(f"\n[+] {item['label']}:")
        print(f"    - Font       : {info.font_name.upper()} ({info.archetype}) @ {info.font_size_pt}pt")
        print(f"    - Pixel Size : {info.width_px} x {info.height_px} px")
        print(f"    - Latent Grid: {info.latent_w} x {info.latent_h} ({info.token_count} tokens)")
        print(f"    - Lines ({len(info.lines)}): {info.lines}")
        print(f"    - Status     : {'[PASS] SILK-SMOOTH' if info.is_nyquist_safe else '[WARNING] SUB-NYQUIST'}")
        print(f"    - Saved      : {artifact_file}")

    print("\n" + "=" * 100)
    print(" [i] Summary Table:")
    print("-" * 100)
    print(f"{'Label':<30} | {'Font':<12} | {'Size':<6} | {'Resolution':<12} | {'Latent':<8} | {'Tokens':<8} | {'Lines':<5}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:<30} | {r['font']:<12} | {r['font_size']}pt{'':<2} | {r['width']}x{r['height']:<6} | {r['latent']:<8} | {r['tokens']:<8} | {r['lines']:<5}")
    print("=" * 100)

if __name__ == "__main__":
    main()
