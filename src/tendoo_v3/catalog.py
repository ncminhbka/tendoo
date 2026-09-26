"""
src/tendoo_v3/catalog.py

Danh mục Template (Catalog) kèm Gợi ý (Hints) và Hướng dẫn cho LLM:
- Cung cấp mô tả ngữ nghĩa chi tiết để LLM tự chọn template tối ưu theo ý đồ người dùng.
- Khai báo cấu trúc hình học của mask liên tục tương ứng với từng template.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# `slots` (GĐ 0B, ROADMAP §6.2): nguồn sự thật DUY NHẤT về field nào template THỰC SỰ hiển
# thị -- lấy từ chính template.html (26/09), không phải từ hint. Field nào của plan không có
# trong `slots` sẽ KHÔNG xuất hiện trên poster (vd l_frame_showcase không có chỗ cho `cta`).
# `drives_geometry`: field này hiện diện hay không thì geometry.py đổi kích thước zone qua
# cờ tương ứng (has_qr/has_footer/has_message/has_freetext) -- thay cho 4 set hard-code cũ
# trong geometry.py. Cờ = True nếu BẤT KỲ slot nào khai báo cờ đó có nội dung.
# Các thuộc tính slot khác trong ví dụ §6.2 (required, supports_markup, max_items) CHƯA khai
# báo: chưa có code nào đọc chúng, và chưa có số đo cho max_items.
TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "split_left": {
        "name": "Cột Dọc Toàn Phần Bên Trái",
        "hint": "Hero và toàn bộ thông tin dạt sang cột bên trái (100% full chiều cao), sản phẩm ở bên phải. Cực kỳ phù hợp cho đồng hồ, sofa, nước hoa, công nghệ, hoặc khi prompt yêu cầu 'ở góc trên trái', 'ở giữa trái', 'chữ bên trái', 'cột trái', 'chia đôi trái'. Là lựa chọn AN TOÀN, ĐA DỤNG cho hầu hết brief KHÔNG nêu rõ vị trí (chứa được nhiều nội dung hơn corner_pod/l_frame_showcase vì chiếm full chiều cao, không chỉ 1 góc). Khác `diagonal_slash`: cột thẳng đứng gọn gàng, phong cách trang trọng/đa ngành, không phải cắt chéo góc cạnh năng động.",
        "has_mask": True,
        "mask_preset": "split_left_full",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {},
            "store_info": {},
            "qr_code": {},
        },
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "split_right": {
        "name": "Cột Dọc Toàn Phần Bên Phải (Bản Gương)",
        "hint": "Sản phẩm ở bên trái, toàn bộ chữ dạt sang cột bên phải (100% full chiều cao). Dùng khi prompt muốn 'chữ bên phải', 'cột phải', 'chia đôi phải', 'sản phẩm bên trái' hoặc chủ thể nghiêng về bên trái khung hình. Bản gương 100% của `split_left` -- cùng mức độ đa dụng, chọn khi vị trí mong muốn ngược lại.",
        "has_mask": True,
        "mask_preset": "split_right_full",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {},
            "store_info": {},
            "qr_code": {},
        },
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "sandwich_top_heavy": {
        "name": "Băng Kẹp Trên - Dưới (Tiêu Đề Đỉnh)",
        "hint": "Tiêu đề lớn trên đỉnh, nút CTA & hotline ở đáy, sản phẩm nằm trọn ở giữa. Rất hợp cho Khuyến mại, Giảm giá thông thường, giới thiệu sản phẩm, hoặc khi prompt yêu cầu 'ở giữa', 'chính giữa', 'ở trên... ở dưới...'. Là lựa chọn MẶC ĐỊNH AN TOÀN cho brief KHÔNG nêu rõ vị trí VÀ không thuộc loại nội dung đặc thù nào khác (menu, feedback, tuyển dụng...) -- phong cách trang trọng/tiết chế. Khác `grand_opening_banner`: cái này KHÔNG rực rỡ lễ hội, không hỗ trợ nhiều dòng CTA cùng lúc -- chỉ dùng grand_opening_banner khi prompt thực sự cần nhiều CTA/không khí sôi động.",
        "has_mask": True,
        "mask_preset": "sandwich_standard",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {},
            "store_info": {},
            "qr_code": {"drives_geometry": ('has_qr',)},
        },
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "sandwich_bottom_heavy": {
        "name": "Băng Kẹp Trên - Dưới (Tiêu Đề Đáy - Bản Gương)",
        "hint": "Thông tin cửa hàng/brand ở trên đỉnh, toàn bộ cụm Tiêu đề & ưu đãi dạt xuống bệ đỡ đáy. Dùng khi prompt yêu cầu 'thông tin cửa hàng lên trên', 'chữ xuống dưới', 'tiêu đề ở dưới', hoặc bố cục nhấn mạnh chân trang ở giữa. Bản gương của `sandwich_top_heavy` -- cùng mức độ trang trọng/đa dụng, chỉ đảo vị trí 2 khối.",
        "has_mask": True,
        "mask_preset": "sandwich_standard",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {},
            "store_info": {},
            "qr_code": {"drives_geometry": ('has_qr',)},
        },
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "before_after_split": {
        "name": "So Sánh Trước & Sau (Before / After)",
        "hint": "Chia đôi màn hình Before (trái/trên) & After (phải/dưới), khung thông tin ở đáy hoặc tâm. Dành riêng cho Fitness Gym, Spa thú cưng Poodle, Giảm cân Kombucha, Dịch vụ dọn nhà sạch bóng -- bất kỳ prompt nào có ý 'trước và sau', 'before after', 'so sánh', 'cải thiện rõ rệt', 'lột xác'. BẮT BUỘC điền `tag_left` (nhãn khối Before, mặc định 'BEFORE' nếu bỏ trống) và `tag_right` (nhãn khối After, mặc định 'AFTER') -- đây là 2 field ĐỊNH DANH riêng của template này, không dùng ở template khác. Có thể điền thêm `rating` nếu prompt có số sao đánh giá thật đi kèm kết quả (tuỳ chọn, không bắt buộc).",
        "has_mask": True,
        "mask_preset": "bottom_band",
        "slots": {
            "hero": {},
            "subhead": {},
            "extra_texts": {},
            "cta": {"drives_geometry": ('has_footer',)},
            "store_info": {"drives_geometry": ('has_footer',)},
            "qr_code": {"drives_geometry": ('has_footer',)},
            "rating": {},
            "tag_left": {},
            "tag_right": {},
        },
        "aspect_ratios": ["1:1", "4:5", "16:9", "9:16"],
    },
    "luxury_centered_card": {
        "name": "Thẻ Kính Mờ Thượng Lưu (Luxury Floating Glass Card)",
        "hint": "Thẻ kính mờ sang trọng với hiệu ứng ánh sáng kim hoàn, typography cao cấp, huy hiệu VIP và QR code không tì vết. Hỗ trợ 3 hướng: 'center' (chính giữa, tri ân / voucher), 'left' (thẻ bên trái, sản phẩm bên phải) và 'right' (bản gương 100%: thẻ bên phải, sản phẩm bên trái). Dùng cho thiệp mời VIP, thư tri ân, voucher/quà tặng KHÔNG kèm nội dung đánh giá/review thật. Khác `customer_feedback_card`: card này KHÔNG có sao đánh giá/lời nhận xét khách hàng -- nếu prompt có testimonial/rating thật, chọn customer_feedback_card thay vào đó.",
        "has_mask": True,
        "mask_preset": "luxury_card",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {"drives_geometry": ('has_message',)},
            "cta": {"drives_geometry": ('has_footer',)},
            "store_info": {"drives_geometry": ('has_footer',)},
            "qr_code": {"drives_geometry": ('has_footer',)},
            "rating": {},
        },
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "lifestyle_corner_pod": {
        "name": "Hộp Bo Góc Lifestyle (Corner Pod)",
        "hint": "Khối hộp capsule bo tròn NHỎ GỌN duy nhất ở 1 góc, 70-80% còn lại là đại cảnh thiên nhiên/biển/núi hoàn toàn mở. Dành cho thể thao ngoài trời, camping, lặn biển, đồng hồ chống nước, hoặc bất kỳ prompt nào yêu cầu đúng 'chữ/nội dung gom ở góc'. Hỗ trợ orientation: 'bottom_left' ('góc dưới trái', mặc định), 'bottom_right' ('góc dưới phải'), 'top_left' ('góc trên trái'), 'top_right' ('góc trên phải'). CHỈ chứa nội dung ngắn gọn (badge+hero+subhead+vài pill) -- nếu cần nhiều nội dung hơn (badge+benefit+dải thông tin đáy riêng) cho hàng cao cấp (ô tô, thời trang), dùng `l_frame_showcase` thay vào đó (khung rộng hơn, sức chứa lớn hơn, không phải 1 khối capsule đơn lẻ).",
        "has_mask": True,
        "mask_preset": "corner_bl",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {"drives_geometry": ('has_freetext',)},
            "cta": {"drives_geometry": ('has_footer',)},
            "store_info": {"drives_geometry": ('has_footer',)},
            "qr_code": {"drives_geometry": ('has_qr', 'has_footer')},
        },
        "aspect_ratios": ["1:1", "16:9", "4:5", "9:16"],
    },
    "recruitment_board": {
        "name": "Bảng Tin Tuyển Dụng & Báo Chí 2 Cột",
        "hint": "Bảng thông tin phong cách báo chí, tiêu đề dập nổi ở trên, thân bài chia 2 cột quyền lợi & yêu cầu. Dành cho Tuyển dụng, Khóa học ngoại ngữ, Bảng tin nội bộ -- bất kỳ prompt nào có ý 'tuyển dụng', 'cần tuyển', 'tìm đồng đội', 'gia nhập đội ngũ'.",
        "has_mask": True,
        "mask_preset": "top_band",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {},
            "store_info": {},
            "qr_code": {"drives_geometry": ('has_qr',)},
        },
        "aspect_ratios": ["1:1", "4:5", "9:16", "16:9"],
    },
    "diagonal_slash": {
        "name": "Cắt Chéo Đồ Họa Năng Động (Diagonal Slash & QR Code)",
        "hint": "Cắt chéo canvas thành 2 mảng tương phản: 1 mảng sản phẩm góc chéo, 1 mảng đồ họa chứa Hero cực lớn ở đỉnh, Pills thông số ở giữa, ô QR Code và nút CTA ở góc đáy. Dành cho: Thể thao năng động (running, gym), giày sneaker, công nghệ cao, flash sale bùng nổ, Fintech App -- bất kỳ prompt nào có ý 'năng động', 'mạnh mẽ', 'cắt chéo', 'sôi động', 'tốc độ'. Hỗ trợ orientation: 'left' ('sản phẩm bên phải, chữ nghiêng bên trái', mặc định) hoặc 'right' ('bản gương: sản phẩm bên trái, chữ nghiêng bên phải'). Khác `split_left`/`split_right`: đây có đường cắt chéo góc cạnh, năng lượng cao -- chỉ chọn khi prompt thực sự cần cảm giác động/thể thao/công nghệ, còn lại (đồ dùng thông thường, không khí trang trọng) dùng split_left/split_right.",
        "has_mask": True,
        "mask_preset": "diagonal_slash",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {},
            "store_info": {},
            "qr_code": {},
        },
        "aspect_ratios": ["1:1", "4:5", "9:16", "16:9"],
    },
    "customer_feedback_card": {
        "name": "Thẻ Đánh Giá & Review Khách Hàng (Testimonial Card)",
        "hint": "Thẻ kính mờ hiển thị 5 sao đánh giá uy tín, icon trích dẫn (quote), lời nhận xét chân thực của khách hàng (testimonial), tên/chức danh người review, huy hiệu cam kết và nút đặt lịch. Dành riêng cho: Feedback khách hàng sau 90 ngày (Gym), Spa thú cưng, Review Glamping nghỉ dưỡng, Khách hàng khen Sofa Zen, Nệm ngủ ngon, Nồi chiên không dầu -- BẮT BUỘC prompt có nội dung đánh giá/lời khen/sao thật (điền vào `testimonial`/`rating`/`reviewer_name`). Nếu chỉ là lời tri ân/voucher chung chung KHÔNG có review thật, dùng `luxury_centered_card` thay vào đó.",
        "has_mask": True,
        "mask_preset": "feedback_card",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {"drives_geometry": ('has_footer',)},
            "store_info": {"drives_geometry": ('has_footer',)},
            "qr_code": {"drives_geometry": ('has_footer',)},
            "testimonial": {},
            "reviewer_name": {},
            "rating": {},
        },
        "aspect_ratios": ["1:1", "4:5", "16:9", "9:16"],
    },
    "step_process_roadmap": {
        "name": "Quy Trình & Lộ Trình Hướng Dẫn Các Bước (Steps Roadmap)",
        "hint": "Trình bày chuỗi quy trình rõ ràng từng bước (Step 01 -> Step 02 -> Step 03) với các icon vector, tiêu đề nổi bật và nhãn ưu đãi / bảo hành. Dành cho: Combo Spa 7 bước, Lộ trình 3 tháng tiếng Anh, 3 bước đặt lịch dọn nhà sạch bóng, quy trình chăm sóc xe -- bất kỳ prompt nào có ý 'các bước', 'quy trình', 'hướng dẫn', 'lộ trình'. Điền `steps` (mảng chuỗi theo đúng thứ tự), KHÔNG dùng `extra_texts` cho nội dung này.",
        "has_mask": True,
        "mask_preset": "bottom_band",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {},
            "store_info": {},
            "qr_code": {"drives_geometry": ('has_qr',)},
            "steps": {},
        },
        "aspect_ratios": ["1:1", "4:5", "16:9", "9:16"],
    },
    "l_frame_showcase": {
        "name": "Khung Góc Chữ L Đẳng Cấp (L-Frame Showcase)",
        "hint": "Cụm Tiêu đề 3D, Badge Ưu đãi và Lợi ích neo ở góc trên bên trái, dải Thông tin showroom và QR code đặt ở đáy RIÊNG BIỆT, 65% diện tích trung tâm và bên phải mở hoàn toàn cho chủ thể. Cực kỳ tối ưu cho: Ô tô (SUV/Sedan), Thời trang (Áo khoác, Lookbook), Đồng hồ thông minh, Thiết bị cao cấp -- hàng cao cấp cần NHIỀU nội dung hơn 1 khối capsule đơn (tiêu đề + badge + benefit + dải liên hệ đáy). Khác `lifestyle_corner_pod`: đây là khung góc + dải đáy TÁCH RIÊNG (sức chứa lớn hơn), không phải 1 khối capsule bo tròn duy nhất -- dùng lifestyle_corner_pod cho brief chỉ cần nội dung ngắn gọn/phong cách outdoor-lifestyle. Hỗ trợ `orientation`: 'left' (mặc định, cụm chữ góc trên trái) hoặc 'right'/'top_right'/'bottom_right' (bản gương, cụm chữ dạt sang phải). `tag_left` (tuỳ chọn) hiển thị như tên thương hiệu nhỏ trong dải đáy, không phải field bắt buộc.",
        "has_mask": True,
        "mask_preset": "l_frame",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {"drives_geometry": ('has_freetext',)},
            "store_info": {},
            "qr_code": {},
            "tag_left": {},
        },
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "menu_price_board": {
        "name": "Bảng Giá & Thực Đơn Nhiều Dòng (Menu Price Board)",
        "hint": "Cột dọc liệt kê nhiều dòng tên món/dịch vụ kèm giá (dùng `extra_texts`, mỗi dòng 1 món dạng 'Tên món - Giá'), thông tin cửa hàng + QR ở đáy cột. Dành riêng cho: Menu quán ăn/cafe, bảng giá dịch vụ Spa/Salon theo gói, combo nhiều lựa chọn -- bất kỳ prompt nào có ý 'menu', 'thực đơn', 'bảng giá', 'giá dịch vụ', liệt kê từ 2 món/gói trở lên kèm giá cụ thể. Hỗ trợ orientation: 'left' (mặc định, cột chữ bên trái, ảnh bên phải) hoặc 'right' (bản gương).",
        "has_mask": True,
        "mask_preset": "menu_price_board",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {},
            "cta": {},
            "store_info": {},
            "qr_code": {},
        },
        "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
    },
    "grand_opening_banner": {
        "name": "Banner Khai Trương & Lễ Hội (Grand Opening Festive Banner)",
        "hint": "Tiêu đề lớn rực rỡ và thời hạn ưu đãi căn giữa ở đỉnh, bệ đỡ nút CTA chính và địa chỉ cửa hàng + QR ở đáy, trung tâm poster mở cho ly cafe bốc khói, món ăn hấp dẫn hoặc sản phẩm khai trương. Nếu có nhiều dòng ưu đãi/kêu gọi hành động phụ (vd 'Deal sốc', 'Ghé ngay hôm nay!'), đưa dòng MẠNH NHẤT vào `cta`, các dòng còn lại vào `extra_texts` (hiển thị dạng pill/bullet ngay dưới tiêu đề). Không chỉ dành riêng cho khai trương -- đây là lựa chọn TỔNG QUÁT tốt nhất cho MỌI banner ưu đãi/flash sale rộn ràng có nhiều dòng CTA/kêu gọi hành động cùng lúc (vd 'Mua 1 tặng 1', 'Deal sốc', nhiều nút CTA rải rác), không nhất thiết phải là sự kiện khai trương cửa hàng. Nếu prompt chỉ có 1 CTA duy nhất và không khí bình thường (không rực rỡ/lễ hội), dùng `sandwich_top_heavy` thay vào đó -- template đó tiết chế/trang trọng hơn.",
        "has_mask": True,
        "mask_preset": "festive_center",
        "slots": {
            "hero": {},
            "subhead": {},
            "badge": {},
            "extra_texts": {"drives_geometry": ('has_freetext',)},
            "cta": {},
            "store_info": {},
            "qr_code": {"drives_geometry": ('has_qr',)},
        },
        "aspect_ratios": ["1:1", "16:9", "9:16", "4:5"],
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
        from tendoo_core.fonts import FONT_CATALOG
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


def build_llm_effect_prompt() -> str:
    """Tạo danh mục gợi ý Text Effect cho LLM."""
    return """DANH MỤC HIỆU ỨNG CHỮ (style.text_effect):
  - '3d_gold': Chữ mạ vàng 3D đổ bóng ánh kim sang trọng (Vàng bạc, khai trương, luxury).
  - 'neon': Đèn neon phát quang rực rỡ (Cyberpunk, nightlife, bar/pub, công nghệ).
  - 'neon_bloom': Hào quang neon đa tầng tỏa khói sáng cực mạnh.
  - 'chrome': Kim loại bạc chrome bóng bẩy phản chiếu (Đồng hồ, xe, tech cao cấp).
  - 'fire': Chữ lửa rực cháy năng động (Flash sale sốc, gym, đồ cay nóng).
  - 'shadow': Chữ trắng đổ bóng studio sâu đa tầng (Rất dễ đọc trên mọi nền phức tạp).
  - 'embossed': Chữ dập nổi vát cạnh 3D bevel highlight & shadow (Bìa tạp chí, da, gỗ, đá).
  - 'chromatic': Sai sắc quang học RGB split viền đỏ/cyan (Fintech, thể thao, tương lai).
  - 'hologram': Ánh xanh holographic scanline ma trận (AI, robot, viễn tưởng).
  - 'plain_elegant': Chữ phẳng thanh lịch tối giản tiêu chuẩn quốc tế."""
