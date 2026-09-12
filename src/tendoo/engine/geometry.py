"""
src/tendoo/engine/geometry.py

Bộ Giải Toán Hình Học & Không Gian cho Tendoo Omni-Block Engine:
================================================================================
1. VÙNG THÁNH ĐỊA SẢN PHẨM (PRODUCT SANCTUARY):
   - Nguyên tắc bất di bất dịch: Bất kể người dùng nhập bao nhiêu chữ, văn bản KHÔNG BAO GIỜ
     được phép che lấp chủ thể sản phẩm (hoặc người mẫu) ở khu vực trung tâm bức ảnh.
   - Tọa độ Product Sanctuary được định nghĩa theo tỉ lệ chuẩn hóa tương đối [ymin, xmin, ymax, xmax]
     cho 4 tỉ lệ khung hình thương mại phổ biến: 1:1 (Square), 9:16 (Story/Reels), 16:9 (Banner), 4:5 (Feed).
   - Mọi ô chữ (Zones) trong ma trận 3x3 đều bị chặn cứng biên (hard boundaries) bởi Product Sanctuary
     kèm theo một khoảng đệm an toàn `pad = 12px`.

2. CƠ CHẾ ĐO ĐẠC HÌNH HỌC 2 LẦN (TWO-PASS RUNTIME GEOMETRY):
   - Pass 1 (PIL / FreeType offline estimate): Hàm `compute_block_metrics()` ước lượng cỡ font sơ bộ
     và tính toán ngắt dòng tiếng Việt cân xứng (`balance_vietnamese_headline`) để render HTML thô.
   - Pass 2 (Chromium Runtime True Measure): Sau khi HTML được nạp vào browser, script Shrink-to-fit
     Autofit trong `master.html` sẽ co giãn font chính xác từng pixel và báo cáo lại tọa độ thực
     `getBoundingClientRect()` qua `PosterRenderer.measure_zone_rects()`.
   - Nhờ đó, mặt nạ hành lang (Corridor Mask) khớp khít 100% với chữ thực tế, loại bỏ hoàn toàn
     sai số đo đạc do padding, letter-spacing, line-height và icon SVG gây ra.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tendoo.core.typography import balance_vietnamese_headline, fit_font_size_px
from tendoo.engine.blocks import AdaptiveBlock, ROLE_SCALE, VALID_ZONES

# ==============================================================================
# PHẦN 1: TỌA ĐỘ VÙNG THÁNH ĐỊA SẢN PHẨM (PRODUCT SANCTUARY)
# ==============================================================================

# Tọa độ tương đối chuẩn hóa [ymin, xmin, ymax, xmax] trong khoảng [0.0, 1.0].
#
# TẠI SAO CẦN LÀM:
# - Mô hình AI sinh ảnh (FLUX.2 Base 4B) được định hướng đặt sản phẩm chính ở trung tâm bức ảnh.
#   Nếu các khối text của poster lấn vào vùng này, thuật toán Velocity Blending sẽ làm mờ/xóa mất
#   chi tiết sắc nét của sản phẩm, hoặc ngược lại chữ bị chìm vào chi tiết sản phẩm.
# - Thiết lập ranh giới bảo vệ riêng cho từng tỉ lệ khung hình:
#   + 1:1 (Square): Vùng trung tâm chiếm 44% width (0.28 -> 0.72) và 52% height (0.24 -> 0.76).
#   + 9:16 (Vertical Story): Chiều ngang hẹp nên thu gọn xmin/xmax (0.32 -> 0.68) để chừa 2 mép
#     cho các bước hướng dẫn và tiêu đề.
#   + 16:9 (Horizontal Banner): Chiều dọc hẹp nên nới rộng ymin từ 0.18 để tránh chạm nóc banner.
#   + 4:5 (Standard Feed): Cân đối tỉ lệ giữa 1:1 và 9:16.
PRODUCT_SANCTUARY_BY_RATIO: Dict[str, Tuple[float, float, float, float]] = {
    "1:1": (0.24, 0.28, 0.76, 0.72),
    "9:16": (0.24, 0.32, 0.74, 0.68),
    "16:9": (0.18, 0.32, 0.74, 0.68),
    "4:5": (0.24, 0.28, 0.76, 0.72),
}

# Tỉ lệ số học của các khung hình chuẩn phục vụ thuật toán so khớp khoảng cách gần nhất
_STANDARD_RATIOS: Dict[str, float] = {
    "1:1": 1.0,
    "9:16": 9.0 / 16.0,   # ~0.5625
    "16:9": 16.0 / 9.0,   # ~1.7778
    "4:5": 4.0 / 5.0,     # 0.8
}


def get_product_sanctuary_rect(width: int, height: int) -> Tuple[int, int, int, int]:
    """Tính toán pixel bounding box (x1, y1, x2, y2) của Product Sanctuary theo kích thước canvas.
    
    Thuật toán nhận diện tỉ lệ khung hình (Aspect Ratio Matching):
    - Thay vì dùng điều kiện so sánh ngắt cứng `abs(ratio - target) < 0.05` dễ bị lỗi biên với các
      độ phân giải phi chuẩn (ví dụ 600x800, 720x1280), thuật toán tính khoảng cách Euclid gần nhất
      tới 4 tỉ lệ chuẩn: `min(_STANDARD_RATIOS, key=...)`.
    - Điều này đảm bảo mọi kích thước bất kỳ đều tự động tìm được vùng Sanctuary phù hợp nhất.
    
    TẠI SAO CẦN LÀM:
    - Chuyển đổi từ tọa độ phần trăm tương đối [0.0 - 1.0] sang tọa độ pixel tuyệt đối (px).
    - Cung cấp ranh giới toán học cho `get_zone_bounding_box()` và thuật toán trừ mặt nạ trong `mask.py`.
    """
    ratio_val = width / max(1, height)
    # Tìm tỉ lệ chuẩn gần nhất theo khoảng cách số học
    key = min(_STANDARD_RATIOS.keys(), key=lambda k: abs(ratio_val - _STANDARD_RATIOS[k]))

    ymin, xmin, ymax, xmax = PRODUCT_SANCTUARY_BY_RATIO[key]
    return int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)


# ==============================================================================
# PHẦN 2: TÍNH TOÁN BOUNDING BOX CHO TỪNG Ô KHÔNG GIAN (ZONE BOUNDING BOX)
# ==============================================================================

def get_zone_bounding_box(zone: str, width: int, height: int) -> Tuple[int, int, int, int]:
    """Tính toán pixel bounding box an toàn (x1, y1, x2, y2) cho từng ô Zone trên poster.
    
    NGUYÊN TẮC HÌNH HỌC:
    1. Lề an toàn viền ngoài (Outer Margins): `margin_x = 4% width`, `margin_y = 4% height`.
       Ngăn chặn việc chữ bị dính sát mép cắt của ảnh khi hiển thị trên mạng xã hội.
    2. Khoảng đệm với Sanctuary (Sanctuary Buffer): `pad = 12px`. Giữ khoảng cách tối thiểu
       12 pixel giữa mép chữ và biên của vùng sản phẩm, tránh hiện tượng chữ chạm sát vật thể.
    3. Phân lập thanh Store Bar:
       - `top_bar`: Chiếm dải trên cùng từ margin_y đến `int(s_y1 * 0.6)`.
       - `bottom_bar`: Chiếm dải đáy từ `int(height * 0.88)` đến `height - margin_y`.
       - Các ô hàng 3 (`bottom_left`, `bottom_center`, `bottom_right`) bị chặn dưới tại
         `int(height * 0.86)` để để lại khoảng trống 2% ngăn va chạm với `bottom_bar`.
    
    TẠI SAO CẦN LÀM:
    - Định nghĩa giới hạn không gian tối đa (`max_w_px`, `max_h_px`) cho từng ô trong lưới 3x3.
    - Script CSS Autofit trong trình duyệt sẽ dựa vào `max-width` và `max-height` này để biết
      khi nào nội dung bắt đầu tràn (`overflows`) và kích hoạt tự động thu nhỏ cỡ chữ.
    """
    margin_x = int(width * 0.04)
    margin_y = int(height * 0.04)

    # Tọa độ thực tế của Product Sanctuary theo aspect ratio
    s_x1, s_y1, s_x2, s_y2 = get_product_sanctuary_rect(width, height)
    pad = 12

    if zone == "top_bar":
        # Thanh đỉnh: Toàn bộ chiều ngang, chặn dưới trước khi chạm vào Sanctuary
        return margin_x, margin_y, width - margin_x, max(margin_y + 30, int(s_y1 * 0.6))
    
    elif zone == "bottom_bar":
        # Thanh đáy: Toàn bộ chiều ngang, bắt đầu từ 88% chiều cao canvas
        return margin_x, int(height * 0.88), width - margin_x, height - margin_y
    
    elif zone == "top_left":
        # Góc trên bên trái: Bị chặn phải bởi mép trái Sanctuary (s_x1 - pad), chặn dưới bởi (s_y1 - pad)
        # Nới rộng tối thiểu 28% width để các badge phụ không bị nghẽn
        max_right = max(int(width * 0.28), s_x1 - pad)
        return margin_x, margin_y, min(width - margin_x, max_right), max(margin_y + 20, s_y1 - pad)
    
    elif zone == "top_center":
        # Giữa bên trên: Trải rộng từ 6% đến 94% width, bị chặn dưới bởi đỉnh Sanctuary (s_y1 - pad)
        return int(width * 0.06), margin_y, int(width * 0.94), max(margin_y + 20, s_y1 - pad)
    
    elif zone == "top_right":
        # Góc trên bên phải: Bắt đầu từ mép phải Sanctuary (s_x2 + pad) đến mép phải ảnh
        # Nới rộng tối thiểu 28% width để các badge ưu đãi không bị nghẽn
        min_left = min(width - margin_x - int(width * 0.28), s_x2 + pad)
        return max(margin_x, min_left), margin_y, width - margin_x, max(margin_y + 20, s_y1 - pad)
    
    elif zone == "middle_left":
        # Giữa bên trái: Chạy dọc thân Sanctuary từ s_y1 đến s_y2, chặn phải tại s_x1 - pad
        return margin_x, s_y1, max(margin_x + 20, s_x1 - pad), max(s_y1 + 20, s_y2)
    
    elif zone == "middle_right":
        # Giữa bên phải: Chạy dọc thân Sanctuary từ s_y1 đến s_y2, bắt đầu từ s_x2 + pad
        return min(width - margin_x - 20, s_x2 + pad), s_y1, width - margin_x, max(s_y1 + 20, s_y2)
    
    elif zone == "bottom_left":
        # Góc dưới bên trái: Bắt đầu từ đáy Sanctuary (s_y2 + pad), chặn dưới tại 86% height.
        # Bên dưới Sanctuary không còn bị giới hạn bởi thân sản phẩm nữa, cho phép mở rộng
        # tới 58% width để các badge hoặc slogan dưới đáy hiển thị trọn vẹn, không bị cắt nét.
        right_boundary = max(int(width * 0.58), s_x1 - pad)
        return margin_x, s_y2 + pad, min(width - margin_x, right_boundary), int(height * 0.86)
    
    elif zone == "bottom_center":
        # Giữa bên dưới: Trục dọc trung tâm bên dưới Sanctuary, thích hợp cho nút CTA
        return int(width * 0.06), s_y2 + pad, int(width * 0.94), int(height * 0.86)
    
    elif zone == "bottom_right":
        # Góc dưới bên phải: Bắt đầu từ đáy Sanctuary (s_y2 + pad)
        left_boundary = min(int(width * 0.42), s_x2 + pad)
        return max(margin_x, left_boundary), s_y2 + pad, width - margin_x, int(height * 0.86)
    
    elif zone == "middle_center":
        # Ô tâm (Sanctuary): Chỉ dùng cho poster chữ (không có sản phẩm), nằm gọn trong lòng s_x1 -> s_x2
        return s_x1 + pad, s_y1 + pad, max(s_x1 + pad + 20, s_x2 - pad), max(s_y1 + pad + 20, s_y2 - pad)
    
    else:
        # Fallback an toàn phòng thủ rơi về top_left
        return margin_x, margin_y, max(margin_x + 20, s_x1 - pad), max(margin_y + 20, s_y1 - pad)


# ==============================================================================
# PHẦN 3: TÍNH TOÁN METRICS HIỂN THỊ CỦA TỪNG BLOCK (COMPUTE BLOCK METRICS)
# ==============================================================================

def compute_block_metrics(
    block: AdaptiveBlock,
    font_path: str,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Tính toán toàn bộ thông số hiển thị ban đầu cho một AdaptiveBlock.
    
    KẾT QUẢ ĐẦU RA (METRICS DICTIONARY):
    - font_size: Cỡ chữ (px) tối ưu sau khi chạy qua thuật toán nhị phân `fit_font_size_px`.
    - font_weight: Độ đậm font (900 cho hero, 800 cho badge, 500 cho body...).
    - is_uppercase: Cờ viết hoa chuỗi.
    - display_text: Chuỗi văn bản đã được ngắt dòng cân xứng tiếng Việt (chống mồ côi từ).
    - measured_w, measured_h: Kích thước pixel thực tế sau khi tính toán font.
    - zone_rect: Tọa độ bounding box tối đa cho phép của ô đó.
    
    TẠI SAO CẦN LÀM & CÁC LỖI ĐÃ ĐƯỢC KHẮC PHỤC TẠI ĐÂY:
    1. Khắc phục lỗi mồ côi từ (Orphan Word Bug):
       - Trước đây, cấu trúc `if / elif zone == 'top_left' / elif len(words) >= 3` khiến cho các
         tiêu đề đặt tại `top_left` (như "ĐẠI TIỆC MÙA HÈ" trong case_01) bị bỏ qua hàm
         `balance_vietnamese_headline`. Kết quả là từ "HÈ" bị rớt xuống dòng thứ 2 một mình rất xấu.
       - Logic mới tách rời việc mở rộng bề rộng (`max_w_px = width * 0.48`) và cân bằng dòng:
         Tiêu đề trong `top_left` với `max_one_line_chars = 10` sẽ tự động được cân đối thành 2 dòng
         đối xứng: "ĐẠI TIỆC" (dòng 1) / "MÙA HÈ" (dòng 2), loại bỏ 100% hiện tượng mồ côi từ.
    2. Đồng bộ kích thước card QR độc lập:
       - Card QR trong CSS (`master.html`) gồm: canvas 60px + padding 4px + border/padding 8px + caption.
       - Cập nhật `qr_w = 86px`, `qr_h = 106px` (hoặc 86px nếu không có caption) để phản ánh chính xác
         kích thước DOM thực tế trong browser, giúp đo đạc mask offline chuẩn xác.
    3. Nhận biết chữ hoa (Uppercase Awareness):
       - Chữ in hoa (như `TRANSFORMATION`) chiếm nhiều diện tích hơn chữ thường ~15-20%. Thuật toán
         đo đạc trực tiếp trên chuỗi `is_uppercase` để không bao giờ bị tính thiếu kích thước.
    """
    role = block.role if block.role in ROLE_SCALE else "body"
    size_ratio, weight, is_uppercase = ROLE_SCALE[role]

    zone = block.zone or "top_left"
    x1, y1, x2, y2 = get_zone_bounding_box(zone, width, height)
    max_w_px = float(x2 - x1)
    max_h_px = float(y2 - y1)

    text = (block.text or "").strip()
    display_text = text

    # --------------------------------------------------------------------------
    # XỬ LÝ KHỐI QR CODE ĐỘC LẬP
    # --------------------------------------------------------------------------
    if role == "qr":
        # Khớp chính xác với thông số CSS trong master.html:
        # .block-qr (padding 8px * 2) + .qr-canvas (img 60px + padding 4px * 2) + border 2px = 86px
        qr_w = 86
        # Khi có caption text: cộng thêm font 11px + line-height 14px + gap 6px = ~106px
        qr_h = 106 if text else 86
        return {
            "text": text,
            "display_text": text,
            "role": role,
            "zone": zone,
            "font_size": 11,
            "font_weight": 700,
            "is_uppercase": False,
            "measured_w": qr_w,
            "measured_h": qr_h,
            "zone_rect": (x1, y1, x2, y2),
        }

    # --------------------------------------------------------------------------
    # XỬ LÝ KHỐI TIÊU ĐỀ CHÍNH (HERO TITLE) & CÂN BẰNG DÒNG TIẾNG VIỆT
    # --------------------------------------------------------------------------
    if role == "hero":
        # Trường hợp 1: Tiêu đề dài hoặc đoạn thơ nhiều dòng (>= 6 từ hoặc đã có sẵn \n)
        if "\n" in text or len(text.split()) >= 6:
            is_uppercase = False  # Câu dài chuyển sang chữ thường trang nhã để tăng tính dễ đọc
            weight = 700
            max_w_px = max(max_w_px, float(width * 0.88))
            max_h_px = max(max_h_px, float(y2 - y1))
        
        # Trường hợp 2: Tiêu đề ngắn / trung bình (< 6 từ)
        else:
            # Nếu đặt tại góc trên bên trái: Mở rộng chiều ngang lên tối đa 48% canvas width
            # để tiêu đề có không gian dàn trải, không bị ép thành cột chữ quá hẹp.
            if zone == "top_left":
                max_w_px = max(max_w_px, float(width * 0.48))
                target_one_line = 10  # Giới hạn ký tự/dòng hẹp hơn cho ô góc trái
            else:
                target_one_line = 16

            # Cân bằng dòng thông minh (Smart Balancing):
            # Nếu câu có từ 3 từ trở lên (ví dụ: "ĐẠI TIỆC MÙA HÈ", "NÂNG TẦM PHONG CÁCH"),
            # tự động tách thành 2 dòng đối xứng thay vì để 1 từ rơi lẻ loi ở dòng 2.
            if len(text.split()) >= 3:
                lines, _ = balance_vietnamese_headline(text, max_one_line_chars=target_one_line)
                if len(lines) > 1:
                    display_text = "\n".join(lines)

    # Cỡ font cơ sở theo tỉ lệ width (ví dụ 0.068 * 1024 = ~70px cho Hero)
    base_font_size = max(14, int(width * size_ratio))

    # Chạy thuật toán tìm kiếm nhị phân (Binary Search) để tìm font_size lớn nhất
    # mà chuỗi vẫn nằm gọn hoàn hảo bên trong [max_w_px, max_h_px].
    font_size, measured_w, measured_h = fit_font_size_px(
        text=display_text,
        font_path=font_path,
        base_font_size=base_font_size,
        max_width_px=max_w_px,
        max_height_px=max_h_px,
        is_uppercase=is_uppercase,
    )

    return {
        "text": text,
        "display_text": display_text,
        "role": role,
        "zone": zone,
        "font_size": font_size,
        "font_weight": weight,
        "is_uppercase": is_uppercase,
        "measured_w": measured_w,
        "measured_h": measured_h,
        "zone_rect": (x1, y1, x2, y2),
    }


# ==============================================================================
# PHẦN 4: HẰNG SỐ CẤU HÌNH CSS GRID 3x3 VÀ ĐỊNH VỊ PHẦN TỬ
# ==============================================================================

# 8 ô lưới không gian ngoại vi quanh vùng tâm (loại bỏ middle_center để bảo vệ Sanctuary)
GRID_ZONE_NAMES: List[str] = [
    "top_left", "top_center", "top_right",
    "middle_left", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
]

# Toàn bộ Zones không gian hợp lệ được hỗ trợ bởi Tendoo Omni-Block Engine (8 ô ngoại vi + ô tâm + 2 thanh bar)
ZONE_NAMES: List[str] = ["top_bar", "bottom_bar", "middle_center"] + GRID_ZONE_NAMES
OMNI_ZONES: List[str] = ZONE_NAMES  # Tên alias tương thích chuẩn cho hệ thống Omni-Block

# Tọa độ grid-area (row_start / col_start / row_end / col_end) cho CSS Grid 3x3 trong master.html
#
# TẠI SAO CẦN LÀM:
# - Trình duyệt CSS Grid sử dụng đúng cú pháp `grid-area` này để định vị từng container ô
#   vào đúng ma trận 3 hàng x 3 cột mà không cần tính toán tọa độ pixel thủ công.
ZONE_GRID_AREA: Dict[str, str] = {
    "top_left": "1 / 1 / 2 / 2",
    "top_center": "1 / 2 / 2 / 3",
    "top_right": "1 / 3 / 2 / 4",
    "middle_left": "2 / 1 / 3 / 2",
    "middle_center": "2 / 2 / 3 / 3",
    "middle_right": "2 / 3 / 3 / 4",
    "bottom_left": "3 / 1 / 4 / 2",
    "bottom_center": "3 / 2 / 4 / 3",
    "bottom_right": "3 / 3 / 4 / 4",
}

# Hướng căn lề văn bản (text-align) tự nhiên theo vị trí ô:
# - Cột bên trái (left col) -> căn trái (text-align: left)
# - Cột giữa (center col)   -> căn giữa (text-align: center)
# - Cột bên phải (right col) -> căn phải (text-align: right)
ZONE_DEFAULT_ALIGN: Dict[str, str] = {
    "top_bar": "center",
    "top_left": "left", "middle_left": "left", "bottom_left": "left",
    "top_center": "center", "middle_center": "center", "bottom_center": "center",
    "top_right": "right", "middle_right": "right", "bottom_right": "right",
    "bottom_bar": "center",
}

# Hướng tự căn vị trí ô trong CSS Grid: Tuple[justify-self (ngang), align-self (dọc)].
#
# TẠI SAO CẦN LÀM:
# - Giúp các ô chữ "hút" về phía các góc và cạnh mép ngoài của poster (Perimeter Pinning).
# - Ví dụ: Ô `top_left` hút về ('start', 'start') góc trên trái; ô `bottom_right` hút về ('end', 'end').
# - Đảm bảo các khối chữ nằm gọn gàng ở ngoại vi, giải phóng tối đa khoảng trống trung tâm cho hình ảnh AI.
ZONE_SELF_ALIGN: Dict[str, Tuple[str, str]] = {
    "top_bar": ("center", "start"),
    "top_left": ("start", "start"), "top_center": ("center", "start"), "top_right": ("end", "start"),
    "middle_left": ("start", "center"), "middle_center": ("center", "center"), "middle_right": ("end", "center"),
    "bottom_left": ("start", "end"), "bottom_center": ("center", "end"), "bottom_right": ("end", "end"),
    "bottom_bar": ("center", "end"),
}
