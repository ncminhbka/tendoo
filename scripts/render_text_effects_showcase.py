"""
scripts/render_text_effects_showcase.py

Renders a targeted showcase of the 4 Hero Text Effects:
1. 'embossed' - In nổi 3D mạ vàng hoàng gia (Chiseled Gold Relief)
2. 'shadow'   - Bóng đổ chiều sâu studio điện ảnh (Deep Studio Shadow)
3. 'led'      - Đèn LED Backlit phát sáng viền (Backlit LED Halo)
4. 'neon'     - Phát quang Neon rực rỡ (Vibrant Neon Glow)

Saves output renders to output_layout_renders/showcase_effect_*.png
"""

import base64
import io
import sys
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


EFFECT_CASES = [
    {
        "effect": "embossed",
        "name": "in_noi_3d_gold",
        "layout": "center_hourglass",
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
        "bg_color": (18, 14, 28),
    },
    {
        "effect": "shadow",
        "name": "bong_do_studio_shadow",
        "layout": "top_dome",
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
        "bg_color": (22, 28, 38),
    },
    {
        "effect": "led",
        "name": "den_led_backlit",
        "layout": "bottom_platform",
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
        "bg_color": (10, 15, 26),
    },
    {
        "effect": "neon",
        "name": "phat_quang_neon",
        "layout": "top_dome",
        "headline": "ĐẠI TIỆC ÂM NHẠC\nNEON CYBER NIGHT",
        "pre_header": "SPECIAL EVENT 2026",
        "slogan": "Bùng nổ cảm xúc cùng hệ thống ánh sáng laser và âm thanh đỉnh cao",
        "offer_main": "VÉ VÀO CỔNG MIỄN PHÍ",
        "offer_sub": "Tặng kèm 01 đồ uống phát quang cho 100 khách đầu tiên",
        "dates": "20/09/2026 - 22/09/2026",
        "brand": "Tendoo Club",
        "hotline": "1900 7777",
        "address": "Phố Đi Bộ Bùi Viện, Quận 1, TP. HCM",
        "website_link": "https://tendoo.ai/neon-party",
        "bg_color": (12, 8, 22),
    },
    {
        "effect": "chrome",
        "name": "chrome_bach_kim",
        "layout": "bottom_platform",
        "headline": "SIÊU XE THỂ THAO 2026\nĐỈNH CAO TỐC ĐỘ VƯỢT TRỘI",
        "pre_header": "FLAGSHIP HYPERCAR",
        "slogan": "Trải nghiệm mã lực thuần khiết với khung sườn sợi carbon siêu nhẹ",
        "offer_main": "TẶNG BỘ VÀNH BẠCH KIM",
        "offer_sub": "Dành riêng cho 10 đơn đặt cọc đầu tiên",
        "dates": "01/10/2026 - 31/10/2026",
        "brand": "Tendoo Hypercar",
        "hotline": "1900 9999",
        "address": "Khu Đô Thị Sala, TP. Thủ Đức, TP. HCM",
        "website_link": "https://tendoo.ai/hypercar",
        "bg_color": (15, 18, 24),
    },
    {
        "effect": "engraved",
        "name": "khac_chim_sa_thach",
        "layout": "top_dome",
        "headline": "RƯỢU VANG CỔ ĐIỂN\nỦ THÙNG SỒI NĂM 1998",
        "pre_header": "VINTAGE RESERVE",
        "slogan": "Hương vị đậm đà kết tinh từ những vườn nho lâu đời vùng Bordeaux",
        "offer_main": "ƯU ĐÃI ĐẶC BIỆT 15%",
        "offer_sub": "Tặng kèm bộ ly pha lê cao cấp mạ bạc",
        "dates": "10/09/2026 - 25/09/2026",
        "brand": "Tendoo Cellar",
        "hotline": "1900 3333",
        "address": "Biệt Thự Cổ Pháp, Quận 1, TP. HCM",
        "website_link": "https://tendoo.ai/vintage-wine",
        "bg_color": (32, 28, 25),
    },
    {
        "effect": "holographic",
        "name": "hologram_xa_cu",
        "layout": "center_hourglass",
        "headline": "SON MÔI DƯỠNG ẨM\nÁNH KIM XÀ CỪ LẤP LÁNH",
        "pre_header": "GLITTER EDITION 2026",
        "slogan": "Tỏa sáng tự tin rực rỡ với sắc màu biến ảo theo góc độ ánh sáng",
        "offer_main": "MUA 1 TẶNG 1",
        "offer_sub": "Tặng túi đựng mỹ phẩm hologram cao cấp",
        "dates": "15/09/2026 - 30/09/2026",
        "brand": "Tendoo Beauty",
        "hotline": "1900 2222",
        "address": "Tầng 1 Vincom Center, Bà Triệu, Hà Nội",
        "website_link": "https://tendoo.ai/hologram-lip",
        "bg_color": (18, 12, 26),
    },
    {
        "effect": "outline",
        "name": "vien_rong_the_thao",
        "layout": "split_column",
        "headline": "BỘ SƯU TẬP STREETWEAR\nBỨT PHÁ MỌI GIỚI HẠN",
        "pre_header": "LIMITED DROP 2026",
        "slogan": "Thiết kế phá cách đậm chất thể thao đường phố dành riêng cho bạn",
        "offer_main": "GIẢM NGAY 30%",
        "offer_sub": "Áp dụng cho toàn bộ dòng sneaker & hoodie",
        "dates": "05/09/2026 - 20/09/2026",
        "brand": "Tendoo Athletics",
        "hotline": "1900 5555",
        "address": "456 Lê Duẩn, Thanh Khê, Đà Nẵng",
        "website_link": "https://tendoo.ai/streetwear",
        "bg_color": (14, 16, 20),
    },
]


def create_gradient_bg(w: int, h: int, base_rgb: tuple) -> Image.Image:
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    r0, g0, b0 = base_rgb
    for y in range(h):
        t = y / float(h)
        r = int(r0 + t * 25)
        g = int(g0 + t * 20)
        b = int(b0 + t * 30)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_showcase():
    print("=" * 80)
    print("🚀 BẮT ĐẦU RENDER BỘ HIỆU ỨNG CHỮ (EMBOSSED, SHADOW, LED, NEON)")
    print("=" * 80)

    w, h = 1024, 1024

    for case in EFFECT_CASES:
        layout = get_layout(case["layout"])
        qr_uri = generate_qr_base64(case["website_link"])

        content = PosterContent(
            headline=case["headline"],
            pre_header=case["pre_header"],
            slogan=case["slogan"],
            offer_main=case["offer_main"],
            offer_sub=case["offer_sub"],
            dates=case["dates"],
            brand=case["brand"],
            hotline=case["hotline"],
            address=case["address"],
            website_link=case["website_link"],
            qr_data_uri=qr_uri or "",
            applicable=case.get("applicable", ""),
            text_effect=case["effect"],
        )

        bg_img = create_gradient_bg(w, h, case["bg_color"])
        bg_uri = pil_to_data_uri(bg_img)

        safe_zone = layout.get_safe_zone()
        palette = analyze_color_harmony(np.array(bg_img), safe_zone, color_mode="auto")

        html_str = layout.render_html(
            content=content,
            palette=palette,
            bg_data_uri=bg_uri,
            width=w,
            height=h,
        )

        out_name = f"showcase_effect_{case['name']}_{w}x{h}.png"
        out_path = OUTPUT_DIR / out_name

        PosterRenderer.render(
            html_content=html_str,
            output_image_path=out_path,
            width=w,
            height=h,
        )
        print(f"  ✓ [{case['effect'].upper()}] {case['name']} -> {out_name}")

    print("=" * 80)
    print(f"🎉 ĐÃ HOÀN THÀNH TOÀN BỘ 4 HIỆU ỨNG CHỮ!")
    print("=" * 80)


if __name__ == "__main__":
    run_showcase()
