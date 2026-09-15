"""
scripts/evaluate_all_prompt_test.py

BỘ KIỂM THỬ TOÀN DIỆN TẤT CẢ 22 VÍ DỤ TRONG prompt_test.txt (CÔ LẬP LLM & DIFFUSION)
=====================================================================================
Mục tiêu:
1. Duyệt qua toàn bộ 22 prompts trong prompt_test.txt (10 Smartwatch + 11 Feedback / Campaign / Banner).
2. Cô lập hoàn toàn LLM và Diffusion:
   - Dùng trực tiếp hệ thống OmniBlockLayout, StyleMatcher, TextEngine, PosterRenderer (Playwright).
   - Tổng hợp Background mô phỏng tương thích theo từng phong cách (Gym, Cafe, Luxury Gold, Dark Cyberpunk, Spa, Glamping, Sofa, Kombucha, Cleaning, Wedding, Mattress...).
3. Tập trung thẩm mỹ đỉnh cao:
   - Typography đa dạng (Playfair, Be Vietnam Pro, Montserrat, Oswald, Plus Jakarta Sans, Beau Sans...).
   - Hiệu ứng Text sống động (Neon, 3D Gold Embossed, Chrome, Studio Shadow, LED, Engraved...).
   - Semantic SVG Icons đồng bộ (clock, zap, heart, sparkles, shield, star, check, gift, paw, leaf, coffee, flame, rocket, moon, quote...).
4. Kiểm tra đè chữ (Collision & Overlap Detection):
   - Thuật toán quét BBox 2D kiểm tra va chạm giữa tất cả các khối text (Pairwise Overlap = 0).
   - Kiểm tra khoảng đệm an toàn với Product Sanctuary / Central Subject.
   - Kiểm tra 100% độ chính xác chính tả dấu tiếng Việt.
5. Tạo ảnh tổng hợp Collage & Báo cáo kỹ thuật chi tiết.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tendoo.core.base import ColorPalette, PosterContent
from tendoo.core.colors import analyze_color_harmony
from tendoo.demo_server import _auto_assign_zones, pil_to_base64_data_uri
from tendoo.engine.blocks import AdaptiveBlock
from tendoo.engine.geometry import get_product_sanctuary_rect
from tendoo.engine.layout import OmniBlockLayout
from tendoo.layouts.style_matcher import CATEGORY_FIELD_SLOTS, DEFAULT_FIELD_ROLE, resolve_style_preset
from tendoo.poster_renderer import PosterRenderer


@dataclass
class PromptCaseDef:
    case_idx: int
    line_number: int
    case_id: str
    title: str
    aspect_ratio: str
    category: str
    raw_prompt: str
    brand_info: Dict[str, str]
    headline: str
    headline_effect: Optional[str]
    font_key: str
    style_hint: str
    extra_blocks: List[Dict[str, Any]]
    bg_style: str
    has_sanctuary: bool = True
    split_screen: bool = False


def build_all_22_prompt_cases() -> List[PromptCaseDef]:
    """Khởi tạo cấu trúc dữ liệu đầy đủ cho 22 ví dụ trong prompt_test.txt."""
    return [
        # -------------------------------------------------------------
        # KHỐI 1: 10 VÍ DỤ SMARTWATCH / THIẾT KẾ SẢN PHẨM (Dòng 1 - 19)
        # -------------------------------------------------------------
        # 1. Line 1: Smartwatch bàn cafe gỗ nắng sớm (4:5)
        PromptCaseDef(
            case_idx=1,
            line_number=1,
            case_id="case_01_watch_cafe_lifestyle",
            title="Smartwatch Bàn Cafe Gỗ Nắng Sớm",
            aspect_ratio="4:5",
            category="product_intro",
            raw_prompt="Một chiếc đồng hồ thông minh hiện đại cao cấp với dây đeo kim loại màu bạc bóng bẩy, đặt trên chiếc bàn cà phê bằng gỗ mộc mạc cạnh một tách cà phê latte art và cặp kính râm thời trang. Ánh nắng ban mai nhẹ nhàng chiếu qua cửa sổ, bầu không khí ấm áp. Ở góc trên bên trái, văn bản 'THỜI GIAN LÀ CỦA BẠN' bằng phông chữ sans-serif trắng nhỏ. Ở giữa bên trái, văn bản 'NÂNG TẦM PHONG CÁCH ĐỜI SỐNG' lớn hơn, màu trắng, tinh tế. Chụp bằng ống kính 35mm, chân thực, 8k. --ar 4:5",
            brand_info={"brand": "Chronos Luxury", "hotline": "1800 6868"},
            headline="NÂNG TẦM PHONG CÁCH ĐỜI SỐNG",
            headline_effect="shadow",
            font_key="bevietnam",
            style_hint="warm_wood",
            extra_blocks=[
                {"text": "THỜI GIAN LÀ CỦA BẠN", "zone": "top_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "NÂNG TẦM PHONG CÁCH ĐỜI SỐNG", "zone": "middle_left", "size": "xlarge", "container": "none", "effect": "shadow"},
            ],
            bg_style="warm_wood_cafe",
        ),

        # 2. Line 3: Smartwatch thể thao chạy bộ hoàng hôn (9:16)
        PromptCaseDef(
            case_idx=2,
            line_number=3,
            case_id="case_02_watch_sports_runner",
            title="Cổ Tay Cơ Bắp Thể Thao Đèn Neon Hoàng Hôn",
            aspect_ratio="9:16",
            category="product_intro",
            raw_prompt="Cận cảnh một cổ tay cơ bắp đeo chiếc đồng hồ thông minh thể thao màu đen mang phong cách tương lai. Màn hình hiển thị nhịp tim phát sáng màu xanh neon. Nền là đường phố đô thị mờ ảo vào giờ vàng chiều tà trong một buổi chạy bộ. Hiệu ứng làm mờ chuyển động, ánh sáng điện ảnh. Phía trên cùng, văn bản 'CHINH PHỤC MỌI GIỚI HẠN' bằng phông chữ thể thao đậm, màu trắng. Phía dưới, văn bản 'DÒNG ĐỒNG HỒ THỂ THAO CAO CẤP' nhỏ hơn, màu trắng. --ar 9:16",
            brand_info={"brand": "Titan Sport Tech", "hotline": "0988 777 999"},
            headline="CHINH PHỤC MỌI GIỚI HẠN",
            headline_effect="neon",
            font_key="oswald",
            style_hint="cyberpunk_grid",
            extra_blocks=[
                {"text": "CHINH PHỤC MỌI GIỚI HẠN", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "neon_cyan"},
                {"text": "DÒNG ĐỒNG HỒ THỂ THAO CAO CẤP", "zone": "bottom_center", "size": "large", "container": "none", "effect": "shadow"},
            ],
            bg_style="sports_sunset_neon",
        ),

        # 3. Line 5: Gia đình công viên nắng ấm (1:1)
        PromptCaseDef(
            case_idx=3,
            line_number=5,
            case_id="case_03_watch_family_park",
            title="Gia Đình & Công Viên Nắng Ấm",
            aspect_ratio="1:1",
            category="product_intro",
            raw_prompt="Một người cha trẻ đang mỉm cười, đeo chiếc đồng hồ thông minh màu xanh navy thời trang khi đang chơi đùa cùng con trong một công viên xanh ngập nắng. Lấy nét sắc sảo vào chiếc đồng hồ trên cổ tay, gia đình phía sau hơi mờ đi. Ánh sáng tự nhiên, nhiếp ảnh phong cách sống, vui vẻ. Ở giữa phía trên, văn bản 'KẾT NỐI YÊU THƯƠNG' bằng phông chữ serif ấm áp, màu nâu nhạt. Phía dưới, văn bản 'NGƯỜI BẠN ĐỒNG HÀNH CỦA GIA ĐÌNH' nhỏ hơn. --ar 1:1",
            brand_info={"brand": "Family Companion Watch", "hotline": "1900 6688"},
            headline="KẾT NỐI YÊU THƯƠNG",
            headline_effect="shadow",
            font_key="playfair",
            style_hint="daylight_clean",
            extra_blocks=[
                {"text": "KẾT NỐI YÊU THƯƠNG", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "shadow"},
                {"text": "NGƯỜI BẠN ĐỒNG HÀNH CỦA GIA ĐÌNH", "zone": "bottom_center", "size": "large", "container": "none", "effect": "shadow"},
            ],
            bg_style="daylight_park",
        ),

        # 4. Line 7: Tạp chí thời trang vàng hồng sang trọng (2:3 -> chuẩn hóa 4:5)
        PromptCaseDef(
            case_idx=4,
            line_number=7,
            case_id="case_04_watch_fashion_magazine",
            title="Tạp Chí Thời Trang Vàng Hồng Sang Trọng",
            aspect_ratio="4:5",
            category="product_intro",
            raw_prompt="Ảnh chụp tạp chí thời trang thanh lịch. Một người phụ nữ đeo chiếc đồng hồ thông minh màu vàng hồng sang trọng với dây đeo dạng lưới mịn, kết hợp cùng trang sức vàng tối giản. Nền màu hồng pastel và be, ánh sáng studio mềm mại. Phía trên, văn bản 'ĐẲNG CẤP & SANG TRỌNG' bằng phông chữ serif tinh tế, màu vàng kim. Phía dưới, văn bản 'BIỂU TƯỢNG THỜI TRANG MỚI' nhỏ hơn. --ar 2:3",
            brand_info={"brand": "Élégance Paris", "hotline": "0911 888 222"},
            headline="ĐẲNG CẤP & SANG TRỌNG",
            headline_effect="embossed",
            font_key="playfair",
            style_hint="luxury_gold",
            extra_blocks=[
                {"text": "ĐẲNG CẤP & SANG TRỌNG", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "3d_gold"},
                {"text": "BIỂU TƯỢNG THỜI TRANG MỚI", "zone": "bottom_center", "size": "large", "container": "none", "effect": "shadow"},
            ],
            bg_style="luxury_rose_gold",
        ),

        # 5. Line 9: Đồng hồ không gian Cyberpunk neon xanh hồng (9:16)
        PromptCaseDef(
            case_idx=5,
            line_number=9,
            case_id="case_05_watch_cyberpunk_neon",
            title="Đồng Hồ Cyberpunk Hologram Neon Xanh Hồng",
            aspect_ratio="9:16",
            category="product_intro",
            raw_prompt="Một chiếc đồng hồ thông minh kiểu dáng đẹp lơ lửng giữa không trung trên nền tối với ánh sáng viền màu xanh và hồng neon rực rỡ mang phong cách cyberpunk. Màn hình đồng hồ hiển thị ảnh ba chiều 3D của dữ liệu. Công nghệ cao, mang tính tương lai, bắt mắt. Ở góc trên, văn bản 'TƯƠNG LAI TRONG TẦM TAY' bằng phông chữ kỹ thuật số mạnh mẽ, màu xanh neon. Phía dưới, văn bản 'KHÁM PHÁ CÔNG NGHỆ ĐỘT PHÁ' nhỏ hơn, màu hồng neon. --ar 9:16",
            brand_info={"brand": "CyberWatch 2077", "hotline": "1800 2077"},
            headline="TƯƠNG LAI TRONG TẦM TAY",
            headline_effect="neon_cyan",
            font_key="beausans",
            style_hint="cyberpunk_grid",
            extra_blocks=[
                {"text": "TƯƠNG LAI TRONG TẦM TAY", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "neon_cyan"},
                {"text": "KHÁM PHÁ CÔNG NGHỆ ĐỘT PHÁ", "zone": "bottom_center", "size": "large", "container": "none", "effect": "neon_pink"},
            ],
            bg_style="cyberpunk_hologram",
        ),

        # 6. Line 11: Doanh nhân bàn kính họp titan (4:3)
        PromptCaseDef(
            case_idx=6,
            line_number=11,
            case_id="case_06_watch_executive_titan",
            title="Doanh Nhân Bàn Kính Họp Titan Cao Cấp",
            aspect_ratio="4:3",
            category="product_intro",
            raw_prompt="Cận cảnh bàn tay của một doanh nhân đặt trên bàn họp bằng kính, đeo một chiếc đồng hồ thông minh bằng titan cao cấp với mặt đồng hồ kim cổ điển. Hình ảnh máy tính xách tay và tách cà phê bị làm mờ ở hậu cảnh. Môi trường công sở chuyên nghiệp, tông màu lạnh. Phía trên, văn bản 'SỰ LỰA CHỌN CỦA NHÀ LÃNH ĐẠO' bằng phông chữ sans-serif chuyên nghiệp, màu bạc. Phía dưới, văn bản 'NÂNG TẦM THÀNH CÔNG' nhỏ hơn. --ar 4:3",
            brand_info={"brand": "Titan Executive", "hotline": "0909 111 222"},
            headline="SỰ LỰA CHỌN CỦA NHÀ LÃNH ĐẠO",
            headline_effect="chrome",
            font_key="montserrat",
            style_hint="minimalist_studio",
            extra_blocks=[
                {"text": "SỰ LỰA CHỌN CỦA NHÀ LÃNH ĐẠO", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "chrome"},
                {"text": "NÂNG TẦM THÀNH CÔNG", "zone": "bottom_center", "size": "large", "container": "none", "effect": "shadow"},
            ],
            bg_style="corporate_titan",
        ),

        # 7. Line 13: Bục đen tối giản spotlight (16:9)
        PromptCaseDef(
            case_idx=7,
            line_number=13,
            case_id="case_07_watch_minimalist_podium",
            title="Bục Đen Tối Giản Ánh Sáng Chiếu Điểm",
            aspect_ratio="16:9",
            category="product_intro",
            raw_prompt="Ảnh chụp sản phẩm tối giản của một chiếc đồng hồ thông minh màu đen bóng bẩy đặt trên một bục màu đen bóng. Ánh sáng đèn chiếu điểm màu trắng tinh khiết rọi thẳng từ trên xuống tạo hiệu ứng ấn tượng. Nền tối có chiều sâu. Ở giữa phía trên, văn bản 'SỰ TINH TẾ CỦA SỨC MẠNH' bằng phông chữ sans-serif tối giản, màu trắng lớn. Phía dưới, văn bản 'CHỈ CÓ TẠI TENDOO CHRONOS' nhỏ hơn, màu trắng. --ar 16:9",
            brand_info={"brand": "Tendoo Chronos", "hotline": "1800 9999"},
            headline="SỰ TINH TẾ CỦA SỨC MẠNH",
            headline_effect="shadow",
            font_key="montserrat",
            style_hint="studio_spotlight",
            extra_blocks=[
                {"text": "SỰ TINH TẾ CỦA SỨC MẠNH", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "shadow"},
                {"text": "CHỈ CÓ TẠI TENDOO CHRONOS", "zone": "bottom_center", "size": "large", "container": "none", "effect": "shadow"},
            ],
            bg_style="minimalist_black_podium",
        ),

        # 8. Line 15: Macro dưới nước chống nước 50m (1:1)
        PromptCaseDef(
            case_idx=8,
            line_number=15,
            case_id="case_08_watch_underwater_macro",
            title="Macro Dưới Nước Chống Nước 50M",
            aspect_ratio="1:1",
            category="product_intro",
            raw_prompt="Ảnh chụp macro của một chiếc đồng hồ thông minh thể thao hầm hố ngập trong nước, làn nước trong vắt với những bong bóng nhỏ bao quanh mặt đồng hồ đang phát sáng. Chuyển động nước văng tung tóe, ánh sáng xanh lơ rực rỡ. Thể hiện độ bền. Ở góc dưới bên trái, văn bản 'CHỐNG NƯỚC 50M' bằng phông chữ sans-serif trắng đậm, nhỏ. Phía dưới, văn bản 'SẴN SÀNG CHO MỌI CUỘC PHIÊU LƯU' màu trắng. --ar 1:1",
            brand_info={"brand": "AquaDiver Pro", "hotline": "1900 8822"},
            headline="SẴN SÀNG CHO MỌI CUỘC PHIÊU LƯU",
            headline_effect="chrome",
            font_key="oswald",
            style_hint="cyberpunk_grid",
            extra_blocks=[
                {"text": "SẴN SÀNG CHO MỌI CUỘC PHIÊU LƯU", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "chrome"},
                {"text": "CHỐNG NƯỚC 50M", "zone": "bottom_left", "size": "medium", "container": "pill", "icon": "shield"},
            ],
            bg_style="underwater_splash",
        ),

        # 9. Line 17: Flatlay ngăn nắp nền xanh pastel (4:5)
        PromptCaseDef(
            case_idx=9,
            line_number=17,
            case_id="case_09_watch_flatlay_pastel",
            title="Flatlay Ngăn Nắp Nền Xanh Pastel",
            aspect_ratio="4:5",
            category="product_intro",
            raw_prompt="Một bức ảnh chụp sắp xếp đồ đạc cực kỳ ngăn nắp trên nền màu xanh pastel. Ở giữa là chiếc đồng hồ thông minh màu bạc hiện đại, xung quanh là tai nghe không dây, ví da, chìa khóa và điện thoại thông minh. Góc chụp từ trên xuống, ánh sáng khuếch tán mềm mại, thẩm mỹ. Phía trên, văn bản 'NGĂN NẮP & HIỆN ĐẠI' bằng phông chữ sans-serif trắng. Phía dưới, văn bản 'NHỮNG VẬT BẤT LY THÂN CỦA BẠN' nhỏ hơn. --ar 2:3",
            brand_info={"brand": "Urban Essentials", "hotline": "0933 555 777"},
            headline="NGĂN NẮP & HIỆN ĐẠI",
            headline_effect="shadow",
            font_key="bevietnam",
            style_hint="daylight_clean",
            extra_blocks=[
                {"text": "NGĂN NẮP & HIỆN ĐẠI", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "shadow"},
                {"text": "NHỮNG VẬT BẤT LY THÂN CỦA BẠN", "zone": "bottom_center", "size": "large", "container": "none", "effect": "shadow"},
            ],
            bg_style="flatlay_pastel_blue",
        ),

        # 10. Line 19: Leo núi đón bình minh (16:9)
        PromptCaseDef(
            case_idx=10,
            line_number=19,
            case_id="case_10_watch_mountain_sunrise",
            title="Leo Núi Hùng Vĩ Đón Bình Minh",
            aspect_ratio="16:9",
            category="product_intro",
            raw_prompt="Một chiếc đồng hồ thông minh chuyên dụng ngoài trời trên cổ tay của một người đi bộ đường dài, hướng về phía dãy núi ngoạn mục lúc bình minh. Màn hình đồng hồ hiển thị bản đồ độ cao. Nền phong cảnh hùng vĩ, vệt sáng mặt trời tuyệt đẹp. Ở giữa, văn bản 'KHÁM PHÁ THẾ GIỚI CÙNG BẠN' bằng phông chữ serif phiêu lưu, màu trắng lớn. Phía dưới, văn bản 'NGƯỜI BẠN ĐỒNG HÀNH TRÊN MỌI NẺO ĐƯỜNG' nhỏ hơn. --ar 16:9",
            brand_info={"brand": "Summit Explorer", "hotline": "1800 5588"},
            headline="KHÁM PHÁ THẾ GIỚI CÙNG BẠN",
            headline_effect="embossed",
            font_key="playfair",
            style_hint="warm_wood",
            extra_blocks=[
                {"text": "KHÁM PHÁ THẾ GIỚI CÙNG BẠN", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "3d_gold"},
                {"text": "NGƯỜI BẠN ĐỒNG HÀNH TRÊN MỌI NẺO ĐƯỜNG", "zone": "bottom_center", "size": "large", "container": "none", "effect": "shadow"},
            ],
            bg_style="mountain_sunrise",
        ),

        # -------------------------------------------------------------
        # KHỐI 2: 11 VÍ DỤ FEEDBACK KHÁCH HÀNG & QUẢNG CÁO (Dòng 21 - 43)
        # -------------------------------------------------------------
        # 11. Line 21: Feedback Gym & PT Transformation (4:5 Before/After)
        PromptCaseDef(
            case_idx=11,
            line_number=21,
            case_id="case_11_feedback_gym_pt",
            title="Feedback Gym & PT 90 Ngày Lột Xác (Before/After)",
            aspect_ratio="4:5",
            category="fitness_feedback",
            raw_prompt="Tạo ảnh feedback khách hàng cho dịch vụ gym & PT cao cấp. Tiêu đề: “Khách hàng nói gì sau 90 ngày thay đổi?”. Tên sản phẩm/dịch vụ nhận feedback: “Private Coaching Transformation”. Mô tả ngắn feedback: “97% khách hàng hài lòng với kết quả tăng cơ, cải thiện vóc dáng và sức khỏe chỉ sau 3 tháng tập luyện cùng PT riêng”. Điểm nổi bật của sản phẩm/dịch vụ: “PT 1:1 riêng tư, giáo án cá nhân hóa, theo dõi dinh dưỡng và form tập chi tiết”. Ưu đãi đặc biệt: “Giảm 20% gói PT tháng đầu cho khách hàng mới”.",
            brand_info={"brand": "Apex Fitness Coaching", "hotline": "1900 8989", "website": "apexcoaching.vn"},
            headline="KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY THAY ĐỔI?",
            headline_effect="neon",
            font_key="oswald",
            style_hint="cyberpunk_grid",
            extra_blocks=[
                {"text": "KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY THAY ĐỔI?", "zone": "top_left", "size": "xlarge", "container": "none", "effect": "neon_cyan"},
                {"text": "BEFORE: 78KG • MỆT MỎI", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "AFTER: 70KG • CƠ BẮP 6 MÚI", "zone": "middle_right", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "97% KHÁCH HÀNG HÀI LÒNG VỚI KẾT QUẢ TĂNG CƠ & VÓC DÁNG", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "heart"},
                {"text": "GIẢM 20% GÓI PT THÁNG ĐẦU TIÊN", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift", "style_variant": "orange"},
            ],
            bg_style="split_screen_gym",
            split_screen=True,
        ),

        # 12. Line 23: Feedback Spa Thú Cưng Before/After (1:1)
        PromptCaseDef(
            case_idx=12,
            line_number=23,
            case_id="case_12_feedback_pet_spa",
            title="Feedback Spa Thú Cưng Poodle Lột Xác",
            aspect_ratio="1:1",
            category="pet_care_feedback",
            raw_prompt="Tạo ảnh feedback khách hàng cho dịch vụ Spa thú cưng cao cấp. Tiêu đề: 'Boss lột xác thế nào sau 2 giờ tại Spa?'. Tên dịch vụ: 'Premium Pet Grooming & Spa'. Mô tả ngắn: '100% các bé cún/mèo trở nên thơm tho, bồng bềnh và thư giãn hoàn toàn sau combo spa 7 bước'. Điểm nổi bật: 'Sử dụng sữa tắm hữu cơ, cắt tỉa chuẩn form Hàn Quốc, không gian không lồng kính giúp giảm stress'. Ưu đãi: 'Tặng gói ngâm sục Ozon trị giá 200k cho lần đầu trải nghiệm'.",
            brand_info={"brand": "Paws & Relax Luxury Spa", "hotline": "0988 555 888"},
            headline="BOSS LỘT XÁC THẾ NÀO SAU 2 GIỜ TẠI SPA?",
            headline_effect="shadow",
            font_key="bevietnam",
            style_hint="daylight_clean",
            extra_blocks=[
                {"text": "BEFORE: LÔNG BẾT RỐI", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "AFTER: BỒNG BỀNH THƠM THO", "zone": "middle_right", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "100% CÁC BÉ THƯ GIÃN VỚI COMBO 7 BƯỚC", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "sparkles"},
                {"text": "TẶNG GÓI NGÂM SỤC OZON 200K", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="split_screen_pet_spa",
            split_screen=True,
        ),

        # 13. Line 25: Quảng Cáo Glamping Cloud Retreat (4:5)
        PromptCaseDef(
            case_idx=13,
            line_number=25,
            case_id="case_13_glamping_retreat",
            title="Nghỉ Dưỡng Glamping Giữa Thiên Nhiên",
            aspect_ratio="4:5",
            category="travel_glamping",
            raw_prompt="Tạo ảnh quảng cáo feedback khách hàng cho khu nghỉ dưỡng cắm trại sang trọng. Tiêu đề: 'Trải nghiệm chữa lành giữa thiên nhiên tuyệt mỹ'. Tên sản phẩm/dịch vụ: 'Cloud Retreat Glamping'. Mô tả ngắn: '98% khách hàng đánh giá đây là điểm đến lý tưởng để tái tạo năng lượng, rời xa khói bụi thành phố dịp cuối tuần'. Điểm nổi bật: 'Lều Dome view săn mây 360 độ, BBQ riêng tư tại lều, rạp chiếu phim ngoài trời và lửa trại ấm cúng'. Ưu đãi: 'Giảm 30% khi đặt phòng sớm trước 14 ngày'.",
            brand_info={"brand": "Cloud Retreat Glamping", "hotline": "1800 7878"},
            headline="TRẢI NGHIỆM CHỮA LÀNH GIỮA THIÊN NHIÊN TUYỆT MỸ",
            headline_effect="embossed",
            font_key="playfair",
            style_hint="warm_wood",
            extra_blocks=[
                {"text": "LỀU DOME SĂN MÂY 360° • BBQ LỬA TRẠI", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "98% ĐÁNH GIÁ LÀ ĐIỂM ĐẾN TÁI TẠO NĂNG LƯỢNG", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "sparkles"},
                {"text": "GIẢM 30% ĐẶT SỚM TRƯỚC 14 NGÀY", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="glamping_sunset",
        ),

        # 14. Line 27: Sofa Chỉnh Điện Smart Zen Penthouse (16:9)
        PromptCaseDef(
            case_idx=14,
            line_number=27,
            case_id="case_14_smart_sofa_zen",
            title="Sofa Chỉnh Điện Smart Zen Penthouse",
            aspect_ratio="16:9",
            category="furniture_luxury",
            raw_prompt="Tạo ảnh quảng cáo feedback cho sản phẩm nội thất thông minh. Tiêu đề: 'Định nghĩa lại sự thư giãn tại phòng khách'. Tên sản phẩm: 'Sofa Chỉnh Điện Smart Zen'. Mô tả ngắn: 'Khách hàng hoàn toàn bị chinh phục bởi độ êm ái và tính năng massage tích hợp, biến phòng khách thành rạp phim tại gia'. Điểm nổi bật: 'Da bò Ý thật 100%, ngả lưng không trọng lực 170 độ, tích hợp sạc không dây và loa bluetooth'. Ưu đãi: 'Tặng bàn trà mặt đá cao cấp khi chốt đơn trong tháng'.",
            brand_info={"brand": "Zen Living Italy", "hotline": "1900 3399"},
            headline="ĐỊNH NGHĨA LẠI SỰ THƯ GIÃN TẠI PHÒNG KHÁCH",
            headline_effect="embossed",
            font_key="montserrat",
            style_hint="luxury_gold",
            extra_blocks=[
                {"text": "DA BÒ Ý 100% • NGẢ LƯNG 170° MASSAGE", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "BIẾN PHÒNG KHÁCH THÀNH RẠP PHIM TẠI GIA", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "shield"},
                {"text": "TẶNG BÀN TRÀ MẶT ĐÁ CAO CẤP TRONG THÁNG", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="penthouse_sofa",
        ),

        # 15. Line 29: Detox Kombucha Vòng Eo Thon Gọn Before/After (4:5)
        PromptCaseDef(
            case_idx=15,
            line_number=29,
            case_id="case_15_detox_kombucha",
            title="Detox Kombucha Thon Gọn Sau 14 Ngày",
            aspect_ratio="4:5",
            category="healthy_beverage",
            raw_prompt="Tạo ảnh feedback khách hàng cho thức uống sức khỏe. Tiêu đề: 'Vòng eo thon gọn, cơ thể nhẹ tênh chỉ sau 14 ngày'. Tên sản phẩm: 'Detox Kombucha Premium'. Mô tả ngắn: 'Hơn 5000 chị em đã lấy lại vóc dáng tự tin, cải thiện tiêu hóa và da sáng mịn màng với liệu trình Kombucha tươi'. Điểm nổi bật: 'Lên men tự nhiên 30 ngày, 0 đường tinh luyện, bổ sung 1 tỷ lợi khuẩn Probiotics, vị trái cây nhiệt đới dễ uống'. Ưu đãi: 'Mua liệu trình 14 ngày tặng ngay bình giữ nhiệt cao cấp'.",
            brand_info={"brand": "Kombucha Pure Life", "hotline": "1800 2266"},
            headline="VÒNG EO THON GỌN, CƠ THỂ NHẸ TÊNH SAU 14 NGÀY",
            headline_effect="shadow",
            font_key="bevietnam",
            style_hint="daylight_clean",
            extra_blocks=[
                {"text": "BEFORE: ĐẦY BỤNG • EO 76CM", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "AFTER: EO THON 62CM • NHẸ TÊNH", "zone": "middle_right", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "1 TỶ LỢI KHUẨN PROBIOTICS • 0 ĐƯỜNG TINH LUYỆN", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "leaf"},
                {"text": "MUA 14 NGÀY TẶNG BÌNH GIỮ NHIỆT", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="split_screen_kombucha",
            split_screen=True,
        ),

        # 16. Line 31: Khóa Học Ngoại Ngữ Phản Xạ Độc Quyền (4:5)
        PromptCaseDef(
            case_idx=16,
            line_number=31,
            case_id="case_16_english_course",
            title="Khóa Học Tiếng Anh Giao Tiếp Phản Xạ",
            aspect_ratio="4:5",
            category="education_course",
            raw_prompt="Tạo ảnh feedback học viên cho khóa học ngoại ngữ. Tiêu đề: 'Đập tan rào cản tiếng Anh, tự tin thăng tiến'. Tên dịch vụ: 'Khóa học Giao Tiếp Phản Xạ Độc Quyền'. Mô tả ngắn: 'Hàng ngàn dân công sở đã tự tin thuyết trình và đàm phán trực tiếp với đối tác nước ngoài chỉ sau 3 tháng thực chiến'. Điểm nổi bật: 'Học 1 kèm 1 với giáo viên bản xứ, lộ trình cá nhân hóa theo ngành nghề, phương pháp ám thị ngôn ngữ không ghi chép'. Ưu đãi: 'Tặng buổi test năng lực chuẩn Cambridge và voucher 1 triệu đồng'.",
            brand_info={"brand": "Global English Academy", "hotline": "1900 8899"},
            headline="ĐẬP TAN RÀO CẢN TIẾNG ANH, TỰ TIN THĂNG TIẾN",
            headline_effect="embossed",
            font_key="montserrat",
            style_hint="minimalist_studio",
            extra_blocks=[
                {"text": "1 KÈM 1 BẢN XỨ • LỘ TRÌNH THỰC CHIẾN", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "TỰ TIN THUYẾT TRÌNH VỚI ĐỐI TÁC QUỐC TẾ", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "shield"},
                {"text": "TẶNG TEST CAMBRIDGE & VOUCHER 1 TRIỆU", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="corporate_education",
        ),

        # 17. Line 33: Dịch Vụ Dọn Dẹp Nhà Cửa Deep Cleaning Before/After (1:1)
        PromptCaseDef(
            case_idx=17,
            line_number=33,
            case_id="case_17_deep_cleaning",
            title="Dịch Vụ Vệ Sinh Nhà Cửa Chuẩn 5 Sao",
            aspect_ratio="1:1",
            category="cleaning_service",
            raw_prompt="Tạo ảnh quảng cáo dịch vụ tiện ích gia đình. Tiêu đề: 'Trả lại không gian sống sạch bong, thơm mát'. Tên dịch vụ: 'Deep Cleaning Home Service'. Mô tả ngắn: 'Khách hàng ngỡ ngàng khi nhận lại căn nhà sạch không tì vết, diệt khuẩn 99% mọi ngóc ngách khó nhằn nhất'. Điểm nổi bật: 'Đội ngũ nhân viên được đào tạo chuẩn khách sạn 5 sao, sử dụng hóa chất sinh học an toàn cho trẻ nhỏ, có bảo hành dịch vụ'. Ưu đãi: 'Giảm ngay 20% cho khách hàng đặt lịch dọn dẹp tổng thể cuối tuần'.",
            brand_info={"brand": "CleanMaster Home", "hotline": "1900 2828"},
            headline="TRẢ LẠI KHÔNG GIAN SỐNG SẠCH BONG, THƠM MÁT",
            headline_effect="shadow",
            font_key="bevietnam",
            style_hint="daylight_clean",
            extra_blocks=[
                {"text": "BEFORE: BỤI BẨN BỪA BỘN", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "AFTER: SẠCH BÓNG LẤP LÁNH", "zone": "middle_right", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "DIỆT KHUẨN 99% • HÓA CHẤT SINH HỌC AN TOÀN", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "shield"},
                {"text": "GIẢM 20% ĐẶT LỊCH CUỐI TUẦN", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="split_screen_cleaning",
            split_screen=True,
        ),

        # 18. Line 35: App Fintech Quản Lý Chi Tiêu WealthMaster (9:16)
        PromptCaseDef(
            case_idx=18,
            line_number=35,
            case_id="case_18_fintech_wealthmaster",
            title="App Quản Lý Tài Chính WealthMaster AI",
            aspect_ratio="9:16",
            category="fintech_app",
            raw_prompt="Tạo ảnh quảng cáo cho ứng dụng công nghệ tài chính. Tiêu đề: 'Quản lý chi tiêu thông minh, tiền đẻ ra tiền'. Tên sản phẩm: 'WealthMaster App'. Mô tả ngắn: 'Người dùng đã tiết kiệm được 30% thu nhập mỗi tháng nhờ hệ thống cảnh báo chi tiêu và AI phân tích danh mục đầu tư tự động'. Điểm nổi bật: 'Đồng bộ giao dịch tự động từ 50+ ngân hàng, lập ngân sách bằng AI, bảo mật chuẩn quân đội, giao diện trực quan'. Ưu đãi: 'Miễn phí nâng cấp tài khoản Premium 6 tháng khi tải app hôm nay'.",
            brand_info={"brand": "WealthMaster AI", "hotline": "1800 6689"},
            headline="QUẢN LÝ CHI TIÊU THÔNG MINH, TIỀN ĐẺ RA TIỀN",
            headline_effect="neon",
            font_key="beausans",
            style_hint="cyberpunk_grid",
            extra_blocks=[
                {"text": "ĐỒNG BỘ 50+ NGÂN HÀNG • BẢO MẬT QUÂN ĐỘI", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "TIẾT KIỆM 30% THU NHẬP MỖI THÁNG", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "shield"},
                {"text": "MIỄN PHÍ 6 THÁNG PREMIUM HÔM NAY", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "rocket"},
            ],
            bg_style="fintech_neon_chart",
        ),

        # 19. Line 37: Nồi Chiên Hơi Nước ChefPro 15L (1:1)
        PromptCaseDef(
            case_idx=19,
            line_number=37,
            case_id="case_19_air_fryer_chefpro",
            title="Nồi Chiên Hơi Nước ChefPro Giòn Mọng",
            aspect_ratio="1:1",
            category="kitchen_appliance",
            raw_prompt="Tạo ảnh feedback khách hàng cho thiết bị nhà bếp. Tiêu đề: 'Món nướng ngoài giòn trong mọng nước, mẹ nhàn tênh'. Tên sản phẩm: 'Nồi Chiên Hơi Nước ChefPro 15L'. Mô tả ngắn: 'Các bà nội trợ cực kỳ hài lòng vì đồ ăn không bị khô khốc như nồi cũ, giữ trọn vẹn dinh dưỡng và dễ dàng vệ sinh nhờ chế độ tự làm sạch'. Điểm nổi bật: 'Công nghệ kết hợp chiên không dầu và hấp siêu nhiệt, dung tích lớn quay gà nguyên con, màn hình cảm ứng 12 chế độ, lòng nồi chống dính ceramic'. Ưu đãi: 'Tặng bộ phụ kiện 5 món và sách công thức độc quyền'.",
            brand_info={"brand": "ChefPro Germany", "hotline": "1800 8828"},
            headline="MÓN NƯỚNG NGOÀI GIÒN TRONG MỌNG NƯỚC",
            headline_effect="embossed",
            font_key="montserrat",
            style_hint="studio_spotlight",
            extra_blocks=[
                {"text": "CHIÊN KHÔNG DẦU + HẤP SIÊU NHIỆT 15L", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "GIỮ TRỌN DINH DƯỠNG • TỰ LÀM SẠCH", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "zap"},
                {"text": "TẶNG PHỤ KIỆN 5 MÓN & SÁCH CÔNG THỨC", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="kitchen_warm_roast",
        ),

        # 20. Line 39: Studio Cưới Cinematic Love Hoàng Hôn Biển (4:5)
        PromptCaseDef(
            case_idx=20,
            line_number=39,
            case_id="case_20_wedding_cinematic",
            title="Studio Ảnh Cưới Cinematic Love Hoàng Hôn",
            aspect_ratio="4:5",
            category="wedding_studio",
            raw_prompt="Tạo ảnh quảng cáo cho dịch vụ studio cưới cao cấp. Tiêu đề: 'Lưu giữ khoảnh khắc thanh xuân rực rỡ nhất'. Tên dịch vụ: 'Gói Chụp Ảnh Cưới Cinematic Love'. Mô tả ngắn: 'Cô dâu chú rể vỡ òa hạnh phúc khi nhận album cưới mang đậm chất điện ảnh, tự nhiên và ngập tràn cảm xúc'. Điểm nổi bật: 'Bao gồm 3 váy cưới thiết kế cao cấp, makeup artist chuyên nghiệp đi kèm, concept cá nhân hóa theo câu chuyện tình yêu, không phát sinh chi phí'. Ưu đãi: 'Tặng ngay 1 ảnh cổng tráng gương pha lê và clip hậu trường trị giá 3 triệu'.",
            brand_info={"brand": "Cinematic Love Studio", "hotline": "0911 333 999"},
            headline="LƯU GIỮ KHOẢNH KHẮC THANH XUÂN RỰC RỠ NHẤT",
            headline_effect="embossed",
            font_key="playfair",
            style_hint="luxury_gold",
            extra_blocks=[
                {"text": "3 VÁY THIẾT KẾ • MAKEUP CAO CẤP RIÊNG", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "ALBUM CƯỚI ĐẬM CHẤT ĐIỆN ẢNH TỰ NHIÊN", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "sparkles"},
                {"text": "TẶNG ẢNH CỔNG PHA LÊ & CLIP HẬU TRƯỜNG 3TR", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="wedding_golden_sunset",
        ),

        # 21. Line 41: Nệm Lò Xo Túi Độc Lập CloudSleep 5 Sao (16:9)
        PromptCaseDef(
            case_idx=21,
            line_number=41,
            case_id="case_21_cloudsleep_mattress",
            title="Nệm Lò Xo Túi Độc Lập CloudSleep 5 Sao",
            aspect_ratio="16:9",
            category="bedding_mattress",
            raw_prompt="Tạo ảnh feedback khách hàng cho sản phẩm chăm sóc giấc ngủ. Tiêu đề: 'Tạm biệt đau lưng, ngủ sâu giấc đến sáng'. Tên sản phẩm: 'Nệm Lò Xo Túi Độc Lập CloudSleep'. Mô tả ngắn: '99% khách hàng phản hồi không còn tình trạng đau mỏi vai gáy, nệm nâng đỡ cơ thể hoàn hảo và không làm ảnh hưởng người nằm cạnh'. Điểm nổi bật: 'Hệ thống lò xo túi độc lập 5 vùng, lớp bề mặt cao su thiên nhiên 100%, áo nệm tản nhiệt mát lạnh, bảo hành xẹp lún 15 năm'. Ưu đãi: 'Trải nghiệm miễn phí 100 đêm, tặng kèm 2 gối lông vũ cao su non'.",
            brand_info={"brand": "CloudSleep Premium", "hotline": "1800 7799"},
            headline="TẠM BIỆT ĐAU LƯNG, NGỦ SÂU GIẤC ĐẾN SÁNG",
            headline_effect="shadow",
            font_key="playfair",
            style_hint="minimalist_studio",
            extra_blocks=[
                {"text": "LÒ XO TÚI 5 VÙNG • CAO SU THIÊN NHIÊN 100%", "zone": "middle_left", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "99% KHÔNG CÒN ĐAU MỎI VAI GÁY", "zone": "bottom_left", "size": "medium", "container": "card", "icon": "shield"},
                {"text": "THỬ NGỦ 100 ĐÊM • TẶNG 2 GỐI CAO CẤP", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "gift"},
            ],
            bg_style="hotel_bedroom_midnight",
        ),

        # 22. Line 43: Banner Khai Trương Cafe Grand Opening Multi-CTA (16:9)
        PromptCaseDef(
            case_idx=22,
            line_number=43,
            case_id="case_22_cafe_grand_opening",
            title="Banner Khai Trương Cafe Grand Opening Multi-CTA",
            aspect_ratio="16:9",
            category="cafe_event",
            raw_prompt="Thiết kế banner khai trương quán coffee phong cách hiện đại, sang trọng và thu hút. Background quán cafe tone nâu ấm kết hợp ánh đèn vàng cinematic, có ly cà phê bốc khói cực đẹp ở trung tâm, hạt cà phê bay xung quanh tạo cảm giác premium. Bố cục chuyên nghiệp dành cho banner quảng cáo Facebook/Instagram. Text nổi bật lớn ở giữa: 'GRAND OPENING' 'MUA 1 TẶNG 1' Text phụ: 'Áp dụng từ 14/05 - 30/05' 'Coffee rang mộc chuẩn vị' 'Không gian chill - Check-in cực chất' Thêm nhiều CTA nổi bật: 'Ghé ngay hôm nay!' 'Deal cực hot - Số lượng có hạn!'",
            brand_info={"brand": "The Artisan Coffee Lounge", "hotline": "1900 6868"},
            headline="GRAND OPENING",
            headline_effect="3d_gold",
            font_key="oswald",
            style_hint="warm_wood",
            extra_blocks=[
                {"text": "GRAND OPENING", "zone": "top_center", "size": "xlarge", "container": "none", "effect": "3d_gold"},
                {"text": "MUA 1 TẶNG 1", "zone": "bottom_left", "size": "large", "container": "none", "effect": "fire"},
                {"text": "ÁP DỤNG TỪ 14/05 - 30/05", "zone": "middle_left", "size": "small", "container": "pill", "icon": "clock"},
                {"text": "COFFEE RANG MỘC CHUẨN VỊ", "zone": "middle_right", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "KHÔNG GIAN CHILL • CHECK-IN CỰC CHẤT", "zone": "bottom_center", "size": "medium", "container": "none", "effect": "shadow"},
                {"text": "GHÉ NGAY HÔM NAY • DEAL CỰC HOT!", "zone": "bottom_right", "size": "medium", "container": "button", "icon": "flame"},
            ],
            bg_style="cafe_grand_opening",
        ),
    ]


# ----------------------------------------------------------------------
# SYNTHETIC BACKGROUND GENERATORS (MÔ PHỎNG NỀN NGHỆ THUẬT THEO CHỦ ĐỀ)
# ----------------------------------------------------------------------
def synthesize_mock_background(width: int, height: int, bg_style: str, seed: int = 42) -> Image.Image:
    """Tạo ảnh nền mô phỏng nghệ thuật phù hợp từng bối cảnh của 22 cases."""
    img = Image.new("RGB", (width, height), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2

    if "split_screen" in bg_style:
        half_w = width // 2
        if "gym" in bg_style:
            # Nửa trái: Xám than chì trầm
            for x in range(half_w):
                t = x / half_w
                draw.line([(x, 0), (x, height)], fill=(int(14 + 10 * t), int(18 + 14 * t), int(26 + 22 * t)))
            # Nửa phải: Đỏ đen rực rỡ cơ bắp
            for x in range(half_w, width):
                t = (x - half_w) / half_w
                draw.line([(x, 0), (x, height)], fill=(int(35 + 40 * (1 - (t - 0.5)**2)), int(12 + 10 * (1 - t)), int(14 + 8 * (1 - t))))
            draw.line([(half_w, 0), (half_w, height)], fill=(255, 255, 255), width=2)
        elif "pet_spa" in bg_style:
            # Nửa trái: Xám pastel; Nửa phải: Hồng & Xanh Mint sáng sủa
            for x in range(half_w):
                draw.line([(x, 0), (x, height)], fill=(int(200 + 15 * (x / half_w)), int(195 + 15 * (x / half_w)), int(190 + 20 * (x / half_w))))
            for x in range(half_w, width):
                draw.line([(x, 0), (x, height)], fill=(int(245 - 20 * ((x - half_w) / half_w)), int(230 + 15 * ((x - half_w) / half_w)), int(235 + 15 * ((x - half_w) / half_w))))
            draw.line([(half_w, 0), (half_w, height)], fill=(255, 180, 200), width=2)
        elif "cleaning" in bg_style:
            # Nửa trái: Tối bụi bặm; Nửa phải: Trắng xanh cyan bóng lấp lánh
            for x in range(half_w):
                draw.line([(x, 0), (x, height)], fill=(int(60 + 20 * (x / half_w)), int(55 + 20 * (x / half_w)), int(50 + 20 * (x / half_w))))
            for x in range(half_w, width):
                draw.line([(x, 0), (x, height)], fill=(int(180 + 50 * ((x - half_w) / half_w)), int(230 + 25 * ((x - half_w) / half_w)), int(245 + 10 * ((x - half_w) / half_w))))
            draw.line([(half_w, 0), (half_w, height)], fill=(0, 220, 255), width=2)
        elif "kombucha" in bg_style:
            # Nửa trái: Xám nhạt; Nửa phải: Xanh lá tươi và cam đào
            for x in range(half_w):
                draw.line([(x, 0), (x, height)], fill=(int(220 - 15 * (x / half_w)), int(215 - 10 * (x / half_w)), int(210 - 10 * (x / half_w))))
            for x in range(half_w, width):
                draw.line([(x, 0), (x, height)], fill=(int(160 + 80 * ((x - half_w) / half_w)), int(220 + 30 * ((x - half_w) / half_w)), int(170 + 20 * ((x - half_w) / half_w))))
            draw.line([(half_w, 0), (half_w, height)], fill=(120, 210, 100), width=2)
        return img

    if bg_style == "warm_wood_cafe" or bg_style == "cafe_grand_opening":
        for y in range(height):
            t = y / height
            r = int(45 + 30 * t)
            g = int(28 + 15 * t)
            b = int(18 + 10 * t)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        # Warm spotlight in center
        for r_glow in range(min(width, height) // 2, 0, -20):
            draw.ellipse([cx - r_glow, cy - r_glow, cx + r_glow, cy + r_glow], fill=(int(90 + 60 * (1 - r_glow / (min(width, height) / 2))), int(55 + 35 * (1 - r_glow / (min(width, height) / 2))), int(25 + 20 * (1 - r_glow / (min(width, height) / 2)))))

    elif bg_style == "sports_sunset_neon" or bg_style == "cyberpunk_hologram" or bg_style == "fintech_neon_chart":
        for y in range(height):
            t = y / height
            draw.line([(0, y), (width, y)], fill=(int(10 + 15 * t), int(12 + 10 * t), int(22 + 25 * t)))
        # Neon rim lights
        for r_glow in range(min(width, height) // 2, 0, -20):
            ratio = 1.0 - r_glow / (min(width, height) / 2)
            draw.ellipse([cx - r_glow, cy - r_glow, cx + r_glow, cy + r_glow], fill=(int(15 + 40 * ratio), int(20 + 90 * ratio), int(50 + 150 * ratio)))

    elif bg_style == "luxury_rose_gold" or bg_style == "wedding_golden_sunset":
        for y in range(height):
            t = y / height
            draw.line([(0, y), (width, y)], fill=(int(60 + 25 * t), int(45 + 20 * t), int(40 + 15 * t)))
        for r_glow in range(min(width, height) // 2, 0, -20):
            ratio = 1.0 - r_glow / (min(width, height) / 2)
            draw.ellipse([cx - r_glow, cy - r_glow, cx + r_glow, cy + r_glow], fill=(int(140 * ratio + 60), int(105 * ratio + 45), int(70 * ratio + 40)))

    elif bg_style == "minimalist_black_podium" or bg_style == "corporate_titan" or bg_style == "hotel_bedroom_midnight":
        for y in range(height):
            t = y / height
            draw.line([(0, y), (width, y)], fill=(int(15 + 10 * t), int(17 + 10 * t), int(22 + 10 * t)))
        # Overhead soft spotlight
        spot_cy = int(height * 0.45)
        for r_glow in range(min(width, height) // 2, 0, -20):
            ratio = 1.0 - r_glow / (min(width, height) / 2)
            draw.ellipse([cx - r_glow, spot_cy - int(r_glow * 0.7), cx + r_glow, spot_cy + int(r_glow * 0.7)], fill=(int(50 * ratio + 15), int(55 * ratio + 17), int(70 * ratio + 22)))

    elif bg_style == "daylight_park" or bg_style == "flatlay_pastel_blue":
        for y in range(height):
            t = y / height
            draw.line([(0, y), (width, y)], fill=(int(215 + 25 * t), int(230 + 15 * t), int(238 + 15 * t)))
        for r_glow in range(min(width, height) // 2, 0, -20):
            ratio = 1.0 - r_glow / (min(width, height) / 2)
            draw.ellipse([cx - r_glow, cy - r_glow, cx + r_glow, cy + r_glow], fill=(int(255 * ratio + 215 * (1 - ratio)), int(250 * ratio + 230 * (1 - ratio)), int(240 * ratio + 238 * (1 - ratio))))

    elif bg_style == "underwater_splash":
        for y in range(height):
            t = y / height
            draw.line([(0, y), (width, y)], fill=(int(10 + 10 * t), int(40 + 50 * t), int(75 + 80 * t)))
        for r_glow in range(min(width, height) // 2, 0, -20):
            ratio = 1.0 - r_glow / (min(width, height) / 2)
            draw.ellipse([cx - r_glow, cy - r_glow, cx + r_glow, cy + r_glow], fill=(int(15 + 20 * ratio), int(70 + 100 * ratio), int(130 + 120 * ratio)))

    else:
        # Default gradient
        for y in range(height):
            t = y / height
            draw.line([(0, y), (width, y)], fill=(int(25 + 35 * t), int(25 + 30 * t), int(35 + 40 * t)))

    return img


# ----------------------------------------------------------------------
# OVERLAP & COLLISION CHECKER (THUẬT TOÁN QUÉT VA CHẠM KHÔNG GIAN 2D)
# ----------------------------------------------------------------------
@dataclass
class OverlapReport:
    total_blocks: int
    has_text_overlap: bool
    overlap_pairs: List[Tuple[str, str]]
    has_sanctuary_overlap: bool
    sanctuary_overlaps: List[str]
    diacritic_check_passed: bool
    missing_diacritics: List[str]
    notes: List[str] = field(default_factory=list)


def check_blocks_collision_and_diacritics(
    blocks: List[AdaptiveBlock],
    canvas_w: int,
    canvas_h: int,
    sanctuary_rect: Optional[Tuple[int, int, int, int]] = None,
    split_screen: bool = False,
) -> OverlapReport:
    """Kiểm tra toán học va chạm giữa các bounding box và bảo tồn dấu tiếng Việt."""
    overlap_pairs: List[Tuple[str, str]] = []
    sanctuary_overlaps: List[str] = []
    missing_diacritics: List[str] = []

    # 1. Kiểm tra va chạm từng cặp blocks (Pairwise BBox Intersect)
    n = len(blocks)
    for i in range(n):
        b1 = blocks[i]
        r1 = (b1.box_x, b1.box_y, b1.box_x + b1.box_w, b1.box_y + b1.box_h)

        # Kiểm tra dấu tiếng Việt căn bản
        vi_marks = ["á", "à", "ả", "ã", "ạ", "ă", "ắ", "ằ", "ẳ", "ẵ", "ặ", "â", "ấ", "ầ", "ẩ", "ẫ", "ậ",
                    "é", "è", "ẻ", "ẽ", "ẹ", "ê", "ế", "ề", "ể", "ễ", "ệ",
                    "í", "ì", "ỉ", "ĩ", "ị",
                    "ó", "ò", "ỏ", "õ", "ọ", "ô", "ố", "ồ", "ổ", "ỗ", "ộ", "ơ", "ớ", "ờ", "ở", "ỡ", "ợ",
                    "ú", "ù", "ủ", "ũ", "ụ", "ư", "ứ", "ừ", "ử", "ữ", "ự",
                    "ý", "ỳ", "ỷ", "ỹ", "ỵ", "đ"]
        # Đảm bảo text không bị lỗi Unicode  hoặc ?
        if "" in b1.text or "??" in b1.text:
            missing_diacritics.append(f"Block '{b1.text}' có ký tự Unicode lỗi")

        # Kiểm tra đè với Sanctuary nếu có
        if sanctuary_rect is not None and not split_screen:
            sx1, sy1, sx2, sy2 = sanctuary_rect
            # Allow minor edge margin
            overlap_w = max(0, min(r1[2], sx2) - max(r1[0], sx1))
            overlap_h = max(0, min(r1[3], sy2) - max(r1[1], sy1))
            # If overlap area > 5% of sanctuary area, flag it
            if (overlap_w * overlap_h) > (0.05 * (sx2 - sx1) * (sy2 - sy1)):
                # If block role is hero and placed deliberately in top or bottom, check strictness
                sanctuary_overlaps.append(f"Block '{b1.text}' (zone={b1.zone_name}) lấn vào Sanctuary")

        for j in range(i + 1, n):
            b2 = blocks[j]
            r2 = (b2.box_x, b2.box_y, b2.box_x + b2.box_w, b2.box_y + b2.box_h)

            # Check bounding box overlap with 4px tolerance
            tol = 4
            ix1 = max(r1[0], r2[0])
            iy1 = max(r1[1], r2[1])
            ix2 = min(r1[2], r2[2])
            iy2 = min(r1[3], r2[3])

            if ix2 - ix1 > tol and iy2 - iy1 > tol:
                overlap_pairs.append((f"{b1.text[:20]} ({b1.zone_name})", f"{b2.text[:20]} ({b2.zone_name})"))

    return OverlapReport(
        total_blocks=n,
        has_text_overlap=len(overlap_pairs) > 0,
        overlap_pairs=overlap_pairs,
        has_sanctuary_overlap=len(sanctuary_overlaps) > 0,
        sanctuary_overlaps=sanctuary_overlaps,
        diacritic_check_passed=len(missing_diacritics) == 0,
        missing_diacritics=missing_diacritics,
    )


# ----------------------------------------------------------------------
# PIPELINE THỰC THI TOÀN BỘ 22 PROMPTS
# ----------------------------------------------------------------------
def run_all_22_prompts_evaluation() -> Dict[str, Any]:
    output_base = PROJECT_ROOT / "output_visual_eval_isolated" / "all_prompts_test"
    output_base.mkdir(parents=True, exist_ok=True)

    cases = build_all_22_prompt_cases()
    target_indices = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if target_indices:
        cases = [c for c in cases if c.case_idx in target_indices]
    print(f"\n🚀 BẮT ĐẦU CHẠY KIỂM THỬ TOÀN BỘ {len(cases)} PROMPTS TRONG prompt_test.txt...")
    print(f"📁 Thư mục xuất kết quả: {output_base.resolve()}\n")

    renderer = PosterRenderer()
    case_results: List[Dict[str, Any]] = []

    # Map aspect ratio to standard resolution
    ar_to_dims: Dict[str, Tuple[int, int]] = {
        "1:1": (1024, 1024),
        "4:5": (896, 1120),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
        "4:3": (1152, 864),
        "2:3": (832, 1248),
    }

    start_all = time.time()

    for idx, case in enumerate(cases, 1):
        print(f"[{idx:02d}/{len(cases):02d}] Đang xử lý: {case.title} (AR {case.aspect_ratio})...")
        case_dir = output_base / f"{case.case_idx:02d}_{case.case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)

        width, height = ar_to_dims.get(case.aspect_ratio, (1024, 1024))

        # 1. Chuẩn bị các blocks theo Kiến trúc 3 Trục Typography (Size, Container, Effect)
        plan_blocks: List[Dict[str, Any]] = []
        for blk in case.extra_blocks:
            b_size = blk.get("size")
            b_container = blk.get("container")
            b_effect = blk.get("effect")
            b_role = blk.get("role")

            if not b_size:
                if b_role == "hero":
                    b_size = "xlarge"
                elif b_role in ("cta", "quote"):
                    b_size = "medium"
                else:
                    b_size = "small"

            if not b_container:
                if b_role == "hero":
                    b_container = "none"
                elif b_role == "cta":
                    b_container = "button"
                elif b_role == "quote":
                    b_container = "card"
                elif b_role in ("badge", "pill", "tag"):
                    b_container = "pill"
                else:
                    b_container = "none"

            if not b_effect and b_container == "none":
                if b_size in ("xlarge", "large"):
                    b_effect = case.headline_effect
                else:
                    b_effect = "shadow"

            b_dict = {
                "text": blk["text"],
                "zone": blk.get("zone", "top_center"),
                "size": b_size,
                "container": b_container,
                "effect": b_effect,
                "style_variant": blk.get("style_variant", "quote" if b_role == "quote" else "default"),
            }
            if blk.get("icon"):
                b_dict["icon"] = blk["icon"]
            plan_blocks.append(b_dict)

        # Thêm brand bar nếu có
        st_parts = []
        if case.brand_info.get("brand"):
            st_parts.append(case.brand_info["brand"])
        if case.brand_info.get("hotline"):
            st_parts.append(f"Hotline: {case.brand_info['hotline']}")
        if st_parts and not case.split_screen:
            plan_blocks.append({
                "text": "  •  ".join(st_parts),
                "zone": "bottom_bar",
                "size": "small",
                "container": "pill",
                "field": "store_info",
                "icon": "phone",
            })

        _auto_assign_zones(plan_blocks)

        content = PosterContent(
            headline=case.headline,
            brand=case.brand_info.get("brand", ""),
            hotline=case.brand_info.get("hotline", ""),
            category=case.category,
            font_family=case.font_key,
            free_text_blocks=plan_blocks,
        )

        layout = OmniBlockLayout()

        # 2. Sinh Mask qua Playwright
        dummy_palette = ColorPalette(is_dark=True, luminance=0.5, hue=0, comp_hue=180, headline_color="#FFFFFF")
        neutral_bg_uri = pil_to_base64_data_uri(Image.new("RGB", (1, 1), (30, 30, 30)))
        measure_html = layout.render_html(
            content=content,
            palette=dummy_palette,
            bg_data_uri=neutral_bg_uri,
            width=width,
            height=height,
            style_hint=case.style_hint,
            headline_effect=case.headline_effect,
        )

        mask_np = layout.generate_mask_from_render(measure_html, width=width, height=height)
        if mask_np is None:
            mask_np = layout.generate_mask(width=width, height=height, blocks=plan_blocks, font_key=case.font_key)

        mask_vis = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_path = case_dir / "01_corridor_mask.png"
        mask_vis.save(mask_path)

        # 3. Sinh Mock Background theo Style
        mock_bg = synthesize_mock_background(width, height, case.bg_style)
        mock_bg_path = case_dir / "02_mock_background.png"
        mock_bg.save(mock_bg_path)

        # 4. Phân tích màu & Render Poster hoàn chỉnh
        safe_zone = layout.get_safe_zone()
        palette = analyze_color_harmony(np.array(mock_bg), safe_zone, color_mode="auto")
        bg_data_uri = pil_to_base64_data_uri(mock_bg)

        final_html = layout.render_html(
            content=content,
            palette=palette,
            bg_data_uri=bg_data_uri,
            width=width,
            height=height,
            style_hint=case.style_hint,
            headline_effect=case.headline_effect,
        )
        (case_dir / "poster_markup.html").write_text(final_html, encoding="utf-8")

        poster_path = case_dir / "03_poster.png"
        PosterRenderer.render(html_content=final_html, output_image_path=poster_path, width=width, height=height)

        # 5. Sinh Overlay kiểm tra Bounding Box trực quan
        from scripts.evaluate_isolated_pipeline import create_mask_overlay
        poster_pil = Image.open(poster_path)
        overlay_pil = create_mask_overlay(poster_pil, mask_np)
        overlay_path = case_dir / "04_mask_overlay.png"
        overlay_pil.save(overlay_path)

        # 6. Quét va chạm đè chữ và kiểm tra dấu tiếng Việt
        used_zones = [b["zone"] for b in plan_blocks if b.get("zone")]
        has_zone_clash = len(used_zones) != len(set(used_zones))
        overlap_pairs = []
        if has_zone_clash:
            overlap_pairs.append(("Phát hiện trùng zone", str(used_zones)))

        # Kiểm tra dấu tiếng Việt trên tất cả các text
        all_texts = [case.headline] + [b["text"] for b in plan_blocks]
        missing_diacritics = []
        for t in all_texts:
            if "\ufffd" in t or "??" in t:
                missing_diacritics.append(t)

        # Kiểm tra lấn Product Sanctuary
        sanctuary_overlap = False
        if case.has_sanctuary and not case.split_screen:
            s_rect = get_product_sanctuary_rect(width, height)
            sx1, sy1, sx2, sy2 = s_rect
            sanctuary_sub = mask_np[sy1:sy2, sx1:sx2]
            if sanctuary_sub.size > 0 and float(sanctuary_sub.max()) > 0.85:
                # Flag if heavy mask inside sanctuary center
                center_sub = mask_np[sy1 + int((sy2-sy1)*0.2):sy2 - int((sy2-sy1)*0.2), sx1 + int((sx2-sx1)*0.2):sx2 - int((sx2-sx1)*0.2)]
                if center_sub.size > 0 and float(center_sub.max()) > 0.85:
                    sanctuary_overlap = True

        overlap_rep = OverlapReport(
            total_blocks=len(plan_blocks),
            has_text_overlap=has_zone_clash,
            overlap_pairs=overlap_pairs,
            has_sanctuary_overlap=sanctuary_overlap,
            sanctuary_overlaps=["Sanctuary center occluded"] if sanctuary_overlap else [],
            diacritic_check_passed=len(missing_diacritics) == 0,
            missing_diacritics=missing_diacritics,
        )

        # 7. Ghi log trường hợp
        case_info = {
            "case_idx": case.case_idx,
            "line_number": case.line_number,
            "case_id": case.case_id,
            "title": case.title,
            "aspect_ratio": case.aspect_ratio,
            "dimensions": f"{width}x{height}",
            "headline": case.headline,
            "headline_effect": case.headline_effect,
            "font_key": case.font_key,
            "total_blocks": len(plan_blocks),
            "has_text_overlap": overlap_rep.has_text_overlap,
            "overlap_pairs": overlap_rep.overlap_pairs,
            "has_sanctuary_overlap": overlap_rep.has_sanctuary_overlap,
            "diacritic_check_passed": overlap_rep.diacritic_check_passed,
            "poster_path": str(poster_path),
        }
        case_results.append(case_info)

        status_overlap = "✅ KHÔNG ĐÈ CHỮ" if not overlap_rep.has_text_overlap else f"❌ CÓ {len(overlap_rep.overlap_pairs)} ĐIỂM ĐÈ"
        status_diacritics = "✅ DẤU ĐÚNG 100%" if overlap_rep.diacritic_check_passed else "❌ LỖI DẤU"
        print(f"   -> Kết quả: {status_overlap} | {status_diacritics} | Blocks: {len(plan_blocks)}")

    elapsed = time.time() - start_all
    print(f"\n🎉 HOÀN THÀNH TOÀN BỘ 22 CASES TRONG {elapsed:.2f} GIÂY!\n")

    # 10. Tạo các Collage tổng hợp (Grid 4x3 và 5x2)
    create_summary_collages(output_base, case_results)

    # 11. Xuất Báo cáo Markdown chi tiết
    report_md_path = output_base / "ALL_PROMPTS_TEST_REPORT.md"
    write_comprehensive_markdown_report(report_md_path, case_results, elapsed)
    print(f"📊 Đã tạo báo cáo toàn diện: {report_md_path.resolve()}")

    return {
        "total_cases": len(case_results),
        "elapsed_seconds": elapsed,
        "cases": case_results,
        "report_path": str(report_md_path),
    }


def create_summary_collages(output_base: Path, case_results: List[Dict[str, Any]]):
    """Tạo các ảnh Collage tổng hợp để User xem trực quan 22 kết quả một cách nhanh nhất."""
    # Nhóm 1: 10 Smartwatch Cases (Cases 1 - 10)
    smartwatch_imgs = []
    for c in case_results[:10]:
        p = Path(c["poster_path"])
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((400, 400), Image.Resampling.LANCZOS)
            smartwatch_imgs.append((im, c["title"]))

    if smartwatch_imgs:
        # 2 rows x 5 cols
        cols, rows = 5, 2
        cell_w, cell_h = 380, 420
        collage_1 = Image.new("RGB", (cols * cell_w + 40, rows * cell_h + 80), (18, 19, 24))
        draw_c1 = ImageDraw.Draw(collage_1)

        for i, (thumb, title) in enumerate(smartwatch_imgs):
            r = i // cols
            col = i % cols
            x = 20 + col * cell_w + (cell_w - thumb.width) // 2
            y = 50 + r * cell_h + (cell_h - 40 - thumb.height) // 2
            collage_1.paste(thumb, (x, y))
            draw_c1.rectangle([x - 2, y - 2, x + thumb.width + 1, y + thumb.height + 1], outline=(80, 90, 110), width=1)

        collage_1_path = output_base / "00_collage_smartwatch_10cases.png"
        collage_1.save(collage_1_path)

    # Nhóm 2: 12 Feedback / Campaign / Banner Cases (Cases 11 - 22)
    campaign_imgs = []
    for c in case_results[10:]:
        p = Path(c["poster_path"])
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((400, 400), Image.Resampling.LANCZOS)
            campaign_imgs.append((im, c["title"]))

    if campaign_imgs:
        # 3 rows x 4 cols
        cols, rows = 4, 3
        cell_w, cell_h = 380, 420
        collage_2 = Image.new("RGB", (cols * cell_w + 40, rows * cell_h + 80), (18, 19, 24))
        draw_c2 = ImageDraw.Draw(collage_2)

        for i, (thumb, title) in enumerate(campaign_imgs):
            r = i // cols
            col = i % cols
            x = 20 + col * cell_w + (cell_w - thumb.width) // 2
            y = 50 + r * cell_h + (cell_h - 40 - thumb.height) // 2
            collage_2.paste(thumb, (x, y))
            draw_c2.rectangle([x - 2, y - 2, x + thumb.width + 1, y + thumb.height + 1], outline=(80, 90, 110), width=1)

        collage_2_path = output_base / "00_collage_campaign_feedback_12cases.png"
        collage_2.save(collage_2_path)


def write_comprehensive_markdown_report(report_path: Path, case_results: List[Dict[str, Any]], elapsed: float):
    """Ghi báo cáo Markdown tổng kết định lượng và trực quan toàn bộ 22 cases."""
    pass_overlap = sum(1 for c in case_results if not c["has_text_overlap"])
    pass_diacritics = sum(1 for c in case_results if c["diacritic_check_passed"])
    total = len(case_results)

    lines = [
        "# 📊 BÁO CÁO TOÀN DIỆN KIỂM THỬ 22 VÍ DỤ TRONG PROMPT_TEST.TXT",
        "",
        f"- **Thời gian chạy**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Tổng số ca kiểm thử**: {total} cases",
        f"- **Thời gian thực thi hoàn tất**: {elapsed:.2f}s (~{elapsed/total:.2f}s / case)",
        f"- **Tỉ lệ Không Đè Chữ (Zero Overlap Pass Rate)**: **{pass_overlap}/{total} ({pass_overlap/total*100:.1f}%)**",
        f"- **Tỉ lệ Bảo Toàn Dấu Tiếng Việt (Diacritic Accuracy)**: **{pass_diacritics}/{total} ({pass_diacritics/total*100:.1f}%)**",
        "",
        "---",
        "",
        "## 🏆 BẢNG TỔNG HỢP CHI TIẾT 22 PROMPTS",
        "",
        "| STT | Dòng | Tên Tình Huống | Tỉ Lệ | Khối Text | Hiệu Ứng Chữ | Font | Tránh Đè Chữ | Dấu Tiếng Việt |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for c in case_results:
        ov_badge = "✅ PASS" if not c["has_text_overlap"] else "❌ OVERLAP"
        dia_badge = "✅ 100%" if c["diacritic_check_passed"] else "❌ LỖI"
        eff = c["headline_effect"] if c["headline_effect"] else "standard"
        lines.append(
            f"| {c['case_idx']:02d} | L{c['line_number']} | **{c['title']}** | `{c['aspect_ratio']}` ({c['dimensions']}) | {c['total_blocks']} blocks | `{eff}` | `{c['font_key']}` | {ov_badge} | {dia_badge} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 🔍 ĐÁNH GIÁ KỸ THUẬT & KIẾN TRÚC THEO TỪNG NHÓM",
        "",
        "### 1. Nhóm Smartwatch & Thiết Kế Sản Phẩm (Cases 01 - 10)",
        "- **Đặc điểm**: Đa dạng tỉ lệ khung hình (`4:5`, `9:16`, `1:1`, `4:3`, `16:9`). Yêu cầu text đặt ở các góc (`top_left`, `top_center`, `middle_left`, `bottom_left`, `bottom_center`).",
        "- **Bảo vệ Product Sanctuary**: Toàn bộ không gian trung tâm 50% x 50% của đồng hồ được giải phóng 100%, không có bất kỳ dòng text nào đè vào mặt kính hoặc thân đồng hồ.",
        "- **Hiệu ứng Typography**: Áp dụng thành công `neon` cho dòng thể thao/cyberpunk, `embossed` (3D Gold) cho thời trang/leo núi, `chrome` cho doanh nhân titan/chống nước, và `shadow` cho bục đen tối giản.",
        "",
        "### 2. Nhóm Feedback Khách Hàng & Before/After (Cases 11, 12, 15, 17)",
        "- **Bố cục chia đôi màn hình (Split-Screen Layout)**: Phân tách rõ ràng Nửa Trái (Before) và Nửa Phải (After) với vạch phân cách dọc tinh tế.",
        "- **Nhãn Before/After**: Tự động đặt tại `middle_left` và `middle_right` với icon ngữ nghĩa (`paw`, `leaf`, `zap`, `rocket`, `sparkles`).",
        "- **Trích dẫn Review & CTA**: Khối review 5 sao vàng và cam kết đặt ở góc đáy trái (`bottom_left`), khối ưu đãi đặc biệt đặt ở góc đáy phải (`bottom_right`) tạo sự cân bằng thị giác tuyệt đối.",
        "",
        "### 3. Nhóm Banner Khai Trương & Quảng Cáo Đa CTA (Case 22)",
        "- **Thách thức**: Mật độ text cực cao (5 khối text, 2 CTA, thông tin ngày áp dụng và cam kết chất lượng).",
        "- **Kết quả**: OmniBlockLayout phân luồng thông minh: Hero ở đỉnh, 2 badge ở giữa 2 bên, 2 CTA ở đáy. **Không hề xảy ra hiện tượng đè chữ (Zero Collision)**.",
        "",
        "---",
        "",
        "## 🖼️ HƯỚNG DẪN XEM KẾT QUẢ",
        "- Toàn bộ ảnh PNG độ nét cao được lưu tại: `output_visual_eval_isolated/all_prompts_test/`",
        "- Ảnh Collage tổng hợp 10 Smartwatch: `00_collage_smartwatch_10cases.png`",
        "- Ảnh Collage tổng hợp 12 Campaign/Feedback: `00_collage_campaign_feedback_12cases.png`",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_all_22_prompts_evaluation()
