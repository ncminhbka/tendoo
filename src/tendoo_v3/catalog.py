"""
src/tendoo_v3/catalog.py

Danh mục Template (Catalog) kèm Gợi ý (Hints) và Hướng dẫn cho LLM:
- Cung cấp mô tả ngữ nghĩa chi tiết để LLM tự chọn template tối ưu theo ý đồ người dùng.
- Khai báo cấu trúc hình học của mask liên tục tương ứng với từng template.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "split_left": {
        "name": "Cột Dọc Toàn Phần Bên Trái",
        "hint": "Hero và toàn bộ thông tin dạt sang cột bên trái (100% full chiều cao), sản phẩm ở bên phải. Cực kỳ phù hợp cho đồng hồ, sofa, nước hoa, công nghệ, hoặc khi prompt yêu cầu 'ở góc trên trái, ở giữa trái'.",
        "has_mask": True,
        "mask_preset": "split_left_full",
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "split_right": {
        "name": "Cột Dọc Toàn Phần Bên Phải (Bản Gương)",
        "hint": "Sản phẩm ở bên trái, toàn bộ chữ dạt sang cột bên phải (100% full chiều cao). Dùng khi prompt muốn 'chữ bên phải, sản phẩm bên trái' hoặc chủ thể nghiêng về bên trái.",
        "has_mask": True,
        "mask_preset": "split_right_full",
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "sandwich_top_heavy": {
        "name": "Băng Kẹp Trên - Dưới (Tiêu Đề Đỉnh)",
        "hint": "Tiêu đề lớn trên đỉnh, nút CTA & hotline ở đáy, sản phẩm nằm trọn ở giữa. Rất hợp cho Khai trương, Khuyến mại, Flash Sale, hoặc khi prompt yêu cầu 'ở giữa', 'chính giữa', 'ở trên... ở dưới...'.",
        "has_mask": True,
        "mask_preset": "sandwich_standard",
        "aspect_ratios": ["1:1", "9:16", "4:5"],
    },
    "sandwich_bottom_heavy": {
        "name": "Băng Kẹp Trên - Dưới (Tiêu Đề Đáy - Bản Gương)",
        "hint": "Thông tin cửa hàng/brand ở trên đỉnh, toàn bộ cụm Tiêu đề & ưu đãi dạt xuống bệ đỡ đáy. Dùng khi prompt yêu cầu 'thông tin cửa hàng lên trên, chữ xuống dưới' hoặc bố cục nhấn mạnh chân trang ở giữa.",
        "has_mask": True,
        "mask_preset": "sandwich_standard",
        "aspect_ratios": ["1:1", "9:16", "4:5"],
    },
    "before_after_split": {
        "name": "So Sánh Trước & Sau (Before / After)",
        "hint": "Chia đôi màn hình Before (trái/trên) & After (phải/dưới), khung thông tin ở đáy hoặc tâm. Dành riêng cho Fitness Gym, Spa thú cưng Poodle, Giảm cân Kombucha, Dịch vụ dọn nhà sạch bóng.",
        "has_mask": True,
        "mask_preset": "bottom_band",
        "aspect_ratios": ["1:1", "4:5", "16:9"],
    },
    "luxury_centered_card": {
        "name": "Thẻ Kính Mờ Trung Tâm (Zero-Mask)",
        "hint": "Thẻ kính mờ sang trọng ở chính giữa, KHÔNG CẦN MASK (Zero-mask, DiT chỉ sinh nền gỗ/đá cẩm thạch/hoa khô mờ ảo). Dành cho Thư cảm ơn khách hàng, Voucher tri ân, Card thông báo, Bảng giá, Feedback thuần chữ.",
        "has_mask": False,
        "mask_preset": "none",
        "aspect_ratios": ["1:1", "9:16", "4:5", "16:9"],
    },
    "lifestyle_corner_pod": {
        "name": "Hộp Bo Góc Lifestyle (Corner Pod)",
        "hint": "Khối hộp capsule bo tròn gom gọn ở 1 góc, 80% còn lại là đại cảnh thiên nhiên/biển/núi. Dành cho thể thao ngoài trời, camping, lặn biển, đồng hồ chống nước. Hỗ trợ orientation: 'bottom_left' ('góc dưới trái', mặc định), 'bottom_right' ('góc dưới phải'), 'top_left' ('góc trên trái'), 'top_right' ('góc trên phải').",
        "has_mask": True,
        "mask_preset": "corner_bl",
        "aspect_ratios": ["1:1", "16:9", "4:5"],
    },
    "recruitment_board": {
        "name": "Bảng Tin Tuyển Dụng & Báo Chí 2 Cột",
        "hint": "Bảng thông tin phong cách báo chí, tiêu đề dập nổi ở trên, thân bài chia 2 cột quyền lợi & yêu cầu. Dành cho Tuyển dụng, Khóa học ngoại ngữ, Bảng tin nội bộ.",
        "has_mask": True,
        "mask_preset": "top_band",
        "aspect_ratios": ["1:1", "4:5", "9:16"],
    },
    "diagonal_slash": {
        "name": "Cắt Chéo Đồ Họa Năng Động (Diagonal Slash & QR Code)",
        "hint": "Cắt chéo canvas thành 2 mảng tương phản: 1 mảng sản phẩm góc chéo, 1 mảng đồ họa chứa Hero cực lớn ở đỉnh, Pills thông số ở giữa, ô QR Code và nút CTA ở góc đáy. Dành cho: Thể thao năng động (running, gym), giày sneaker, công nghệ cao, flash sale bùng nổ, Fintech App. Hỗ trợ orientation: 'left' ('sản phẩm bên phải, chữ nghiêng bên trái', mặc định) hoặc 'right' ('bản gương: sản phẩm bên trái, chữ nghiêng bên phải').",
        "has_mask": True,
        "mask_preset": "diagonal_slash",
        "aspect_ratios": ["1:1", "4:5", "9:16", "16:9"],
    },
    "customer_feedback_card": {
        "name": "Thẻ Đánh Giá & Review Khách Hàng (Testimonial Card)",
        "hint": "Thẻ kính mờ hiển thị 5 sao đánh giá uy tín, icon trích dẫn (quote), lời nhận xét chân thực của khách hàng (testimonial), tên/chức danh người review, huy hiệu cam kết và nút đặt lịch. Dành riêng cho: Feedback khách hàng sau 90 ngày (Gym), Spa thú cưng, Review Glamping nghỉ dưỡng, Khách hàng khen Sofa Zen, Nệm ngủ ngon, Nồi chiên không dầu.",
        "has_mask": True,
        "mask_preset": "feedback_card",
        "aspect_ratios": ["1:1", "4:5", "16:9"],
    },
    "step_process_roadmap": {
        "name": "Quy Trình & Lộ Trình Hướng Dẫn Các Bước (Steps Roadmap)",
        "hint": "Trình bày chuỗi quy trình rõ ràng từng bước (Step 01 -> Step 02 -> Step 03) với các icon vector, tiêu đề nổi bật và nhãn ưu đãi / bảo hành. Dành cho: Combo Spa 7 bước, Lộ trình 3 tháng tiếng Anh, 3 bước đặt lịch dọn nhà sạch bóng, quy trình chăm sóc xe.",
        "has_mask": True,
        "mask_preset": "bottom_band",
        "aspect_ratios": ["1:1", "4:5", "16:9"],
    },
}


def build_llm_catalog_prompt() -> str:
    """Tạo đoạn prompt mô tả Catalog kèm Hints để nhúng vào System Prompt cho bất kỳ LLM nào."""
    lines = [
        "HỆ THỐNG TEMPLATE THƯƠNG MẠI CÓ SẴN (Hãy chọn template phù hợp nhất dựa trên gợi ý):"
    ]
    for key, info in TEMPLATE_CATALOG.items():
        lines.append(f"- '{key}': {info['name']}")
        lines.append(f"  Gợi ý tình huống: {info['hint']}")
        lines.append(f"  Cần mask dọn nền: {'Có' if info['has_mask'] else 'Không (Zero-Mask, nền thuần)'}")
    return "\n".join(lines)


def build_llm_font_prompt() -> str:
    """Tạo danh sách gợi ý 19 font tiếng Việt chuẩn gom nhóm theo Archetype cho LLM."""
    try:
        from tendoo.core.fonts import FONT_CATALOG
        grouped: Dict[str, List[str]] = {}
        for key, meta in FONT_CATALOG.items():
            archetype = meta.get("archetype", "Khác")
            display_name = meta.get("display_name", key)
            desc = meta.get("description", "")
            grouped.setdefault(archetype, []).append(f"  + '{key}': {display_name} - {desc}")
        
        lines = ["DANH MỤC PHÔNG CHỮ TIẾNG VIỆT (Hãy chọn 1 font phù hợp trong style.font):"]
        for arch, items in grouped.items():
            lines.append(f"- Nhóm {arch}:")
            lines.extend(items)
        return "\n".join(lines)
    except Exception:
        return "DANH MỤC PHÔNG CHỮ (style.font): 'bevietnam' (hiện đại nét), 'playfair' (sang trọng serif), 'anton' (mạnh mẽ sale), 'days' (công nghệ tech), 'lobster' (mềm mại cafe)."


def build_llm_effect_and_vfx_prompt() -> str:
    """Tạo danh mục gợi ý Text Effect và Cinematic VFX Layer cho LLM."""
    return """DANH MỤC HIỆU ỨNG CHỮ VÀ HIỆU ỨNG ĐIỆN ẢNH CINEMATIC VFX:
1. HIỆU ỨNG CHỮ (style.text_effect):
  - '3d_gold': Chữ mạ vàng 3D đổ bóng ánh kim sang trọng (Vàng bạc, khai trương, luxury).
  - 'neon': Đèn neon phát quang rực rỡ (Cyberpunk, nightlife, bar/pub, công nghệ).
  - 'neon_bloom': Hào quang neon đa tầng tỏa khói sáng cực mạnh.
  - 'chrome': Kim loại bạc chrome bóng bẩy phản chiếu (Đồng hồ, xe, tech cao cấp).
  - 'fire': Chữ lửa rực cháy năng động (Flash sale sốc, gym, đồ cay nóng).
  - 'shadow': Chữ trắng đổ bóng studio sâu đa tầng (Rất dễ đọc trên mọi nền phức tạp).
  - 'embossed': Chữ dập nổi vát cạnh 3D bevel highlight & shadow (Bìa tạp chí, da, gỗ, đá).
  - 'chromatic': Sai sắc quang học RGB split viền đỏ/cyan (Fintech, thể thao, tương lai).
  - 'hologram': Ánh xanh holographic scanline ma trận (AI, robot, viễn tưởng).
  - 'plain_elegant': Chữ phẳng thanh lịch tối giản tiêu chuẩn quốc tế.

2. HIỆU ỨNG KHÍ QUYỂN QUANG HỌC CINEMATIC VFX (style.vfx):
  - 'film_grain': Hạt phim 35mm analog điện ảnh (Rất hợp cho tạp chí thời trang, retro, cafe, vintage).
  - 'anamorphic_flare': Vệt lóa quang học chùm sáng quét ngang (Đồng hồ, công nghệ, thể thao cao cấp).
  - 'gold_dust': Bụi vàng phát quang / tàn lửa lơ lửng (Trang sức, luxury, spa, tiệc tùng).
  - 'light_leak': Rò rỉ ánh sáng hoàng hôn / nắng sớm ven cửa sổ (Nhiếp ảnh lifestyle, gia đình, hoa quả).
  - 'cinematic_haze': Lớp sương mù tương phản tăng chiều sâu bối cảnh (Điện ảnh hùng vĩ, núi non, du lịch).
  - 'none': Không dùng hiệu ứng khí quyển (Mặc định)."""
