"""
scripts/render_all_templates_ratios.py

Renders all 4 commercial layouts across key production aspect ratios:
1. 1:1 Square (1024x1024) - Feed / Square Post
2. 9:16 Vertical (576x1024) - TikTok / Story / Reel
3. 16:9 Horizontal (1024x576) - Web Banner / Header
4. 4:5 Portrait (816x1024) - Facebook Mobile Feed

Uses synthetic backdrop gradients + real QR codes + full typography data
to rigorously evaluate responsive scaling, safe zones, and visual rhythm.
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image, ImageDraw

from tendoo.layouts import (
    PosterContent,
    analyze_color_harmony,
    get_layout,
)
from tendoo.typography_engine import PosterRenderer
from tendoo.qr import generate_qr_base64


OUTPUT_DIR = PROJECT_ROOT / "output_layout_renders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RATIOS = [
    ("1x1", 1024, 1024, "Vuông (1024x1024)"),
    ("9x16", 576, 1024, "Dọc Story/TikTok (576x1024)"),
    ("16x9", 1024, 576, "Ngang Banner (1024x576)"),
    ("4x5", 816, 1024, "Dọc Feed Chuẩn (816x1024)"),
]

LAYOUT_CASES = {
    "top_dome": {
        "headline": "TRÀ ĐÀO CAM SẢ\nTHANH MÁT TỰ NHIÊN",
        "pre_header": "BỘ SƯU TẬP MÙA HÈ",
        "slogan": "Thưởng thức trọn vẹn từng giọt tươi mát từ thiên nhiên thanh khiết",
        "offer_main": "MUA 2 TẶNG 1",
        "offer_sub": "Áp dụng cho toàn bộ đồ uống đá xay & trà trái cây",
        "dates": "01/09/2026 - 15/09/2026",
        "brand": "Tendoo Beverage",
        "hotline": "1900 8888",
        "address": "123 Hoàng Hoa Thám, Ba Đình, Hà Nội",
        "website_link": "https://tendoo.ai/summer-sale",
        "bg_type": "beverage",
    },
    "center_hourglass": {
        "headline": "BÁNH TRUNG THU\nHOÀNG GIA THƯỢNG HẠNG",
        "pre_header": "TẾT TRÔNG TRĂNG ĐOÀN VIÊN",
        "slogan": "Món quà trọn vẹn ân tình gửi gắm gia đình sum vầy",
        "offer_main": "CHIẾT KHẤU ĐẾN 20%",
        "offer_sub": "Tặng kèm hộp trà sen Tây Hồ hảo hạng",
        "dates": "15/09/2026 - 01/10/2026",
        "brand": "Tendoo Mooncake",
        "hotline": "0988 123 456",
        "address": "Phố Cổ Hàng Buồm, Hoàn Kiếm, Hà Nội",
        "website_link": "https://tendoo.ai/mooncake",
        "applicable": "Áp dụng cho hóa đơn từ 1.500.000đ",
        "bg_type": "festive",
    },
    "bottom_platform": {
        "headline": "SUV THẾ HỆ MỚI 2026\nCHINH PHỤC MỌI CUNG ĐƯỜNG",
        "pre_header": "PHIÊN BẢN GIỚI HẠN",
        "slogan": "Động cơ Hybrid thông minh kết hợp hệ dẫn động 4 bánh toàn thời gian",
        "offer_main": "ƯU ĐÃI 100 TRIỆU",
        "offer_sub": "Tặng gói bảo hiểm thân vỏ & 3 năm bảo dưỡng",
        "dates": "01/09/2026 - 30/09/2026",
        "brand": "Tendoo Motors",
        "hotline": "1900 6868",
        "address": "Showroom Phú Mỹ Hưng, Quận 7, TP. HCM",
        "website_link": "https://tendoo.ai/suv-2026",
        "applicable": "Hỗ trợ trả góp 0% lãi suất qua ngân hàng",
        "bg_type": "automotive",
    },
    "split_column": {
        "headline": "ÁO MĂNG TÔ THU ĐÔNG\nDẠ LÔNG CỪU CAO CẤP",
        "pre_header": "COLLECTION 2026",
        "slogan": "Chất liệu dạ dệt tay thủ công từ vùng Tuscany, tôn vinh nét thanh lịch",
        "offer_main": "ƯU ĐÃI ĐẾN 25%",
        "offer_sub": "Tặng khăn choàng lụa tơ tằm cho đơn từ 3 triệu",
        "dates": "01/09/2026 - 25/09/2026",
        "brand": "Tendoo Atelier",
        "hotline": "1900 9999",
        "address": "Tràng Tiền Plaza, Hoàn Kiếm, Hà Nội",
        "website_link": "https://tendoo.ai/lookbook",
        "applicable": "Số lượng giới hạn 200 chiếc toàn quốc",
        "bg_type": "fashion",
    },
}


def create_synthetic_bg(w: int, h: int, bg_type: str) -> Image.Image:
    """Creates a high-contrast aesthetic synthetic backdrop simulating DiT inpainting."""
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    if bg_type == "beverage":
        # Amber/peach warm glow in lower half, soft daylight on top
        for y in range(h):
            t = y / float(h)
            r = int(18 + t * 45)
            g = int(24 + t * 30)
            b = int(36 + t * 15)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
    elif bg_type == "festive":
        # Midnight purple with golden center glow
        for y in range(h):
            t = y / float(h)
            r = int(22 + t * 40)
            g = int(14 + t * 25)
            b = int(34 + t * 10)
            draw.line([(0, y), (y, y)], fill=(r, g, b))
    elif bg_type == "automotive":
        # Dark asphalt slate with electric cyan rim light
        for y in range(h):
            t = y / float(h)
            r = int(12 + t * 15)
            g = int(16 + t * 20)
            b = int(26 + t * 35)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
    else:  # fashion
        # Minimalist neutral warm charcoal on right, deep slate on left
        for x in range(w):
            t = x / float(w)
            r = int(14 + t * 35)
            g = int(16 + t * 30)
            b = int(22 + t * 28)
            draw.line([(x, 0), (x, h)], fill=(r, g, b))

    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_all_renders():
    print("=" * 80)
    print("🚀 BẮT ĐẦU RENDER BỘ SƯU TẬP 4 LAYOUT x 4 TỶ LỆ KHUNG HÌNH")
    print("=" * 80)

    results = []
    t_start_all = time.time()

    for layout_key, case_data in LAYOUT_CASES.items():
        layout = get_layout(layout_key)
        qr_uri = generate_qr_base64(case_data["website_link"])

        content = PosterContent(
            headline=case_data["headline"],
            pre_header=case_data["pre_header"],
            slogan=case_data["slogan"],
            offer_main=case_data["offer_main"],
            offer_sub=case_data["offer_sub"],
            dates=case_data["dates"],
            brand=case_data["brand"],
            hotline=case_data["hotline"],
            address=case_data["address"],
            website_link=case_data["website_link"],
            qr_data_uri=qr_uri or "",
            applicable=case_data.get("applicable", ""),
        )

        for ratio_code, w, h, ratio_name in RATIOS:
            t0 = time.time()
            bg_img = create_synthetic_bg(w, h, case_data["bg_type"])
            bg_uri = pil_to_data_uri(bg_img)

            # Analyze palette
            safe_zone = layout.get_safe_zone()
            palette = analyze_color_harmony(np.array(bg_img), safe_zone, color_mode="auto")

            # Render HTML
            html_str = layout.render_html(
                content=content,
                palette=palette,
                bg_data_uri=bg_uri,
                width=w,
                height=h,
            )

            out_filename = f"{layout_key}_{ratio_code}_{w}x{h}.png"
            out_path = OUTPUT_DIR / out_filename

            # Save HTML for inspection
            html_path = OUTPUT_DIR / f"{layout_key}_{ratio_code}_{w}x{h}.html"
            html_path.write_text(html_str, encoding="utf-8")

            # Rasterize with Playwright
            PosterRenderer.render(
                html_content=html_str,
                output_image_path=out_path,
                width=w,
                height=h,
            )

            dur = time.time() - t0
            print(f"  ✓ [{layout.display_name}] Ratio {ratio_name}: {out_filename} ({dur:.2f}s)")
            results.append({
                "layout_key": layout_key,
                "layout_name": layout.display_name,
                "ratio_code": ratio_code,
                "ratio_name": ratio_name,
                "width": w,
                "height": h,
                "file_path": str(out_path),
                "file_name": out_filename,
            })

    total_dur = time.time() - t_start_all
    print("=" * 80)
    print(f"🎉 HOÀN THÀNH TOÀN BỘ {len(results)} BẢN RENDER TRONG {total_dur:.1f}s!")
    print(f"📁 Thư mục xuất file: {OUTPUT_DIR.resolve()}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_all_renders()
