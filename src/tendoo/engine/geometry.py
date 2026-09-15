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

from typing import Any, Dict, List, Optional, Set, Tuple

from tendoo.core.typography import balance_vietnamese_headline, fit_font_size_px
from tendoo.engine.blocks import AdaptiveBlock, SIZE_SCALE

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
    "16:9": (0.25, 0.30, 0.75, 0.70),
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
        # Thanh đỉnh: CSS thật (`.omni-top-bar` trong master.html) là 1 viên nang
        # `width: fit-content` căn giữa, neo tại `top: 3.5%`, cao ~padding 8px x2 +
        # font 14px -- KHÔNG phải dải toàn chiều rộng kéo dài tới 60% Sanctuary như
        # trước đây. Rect hẹp hơn này chỉ ảnh hưởng đường ước lượng offline PIL
        # (`generate_omni_corridor_mask`/`compute_block_metrics`) -- render_html không
        # bao giờ gọi get_zone_bounding_box cho "top_bar"/"bottom_bar" (store-info bar
        # dùng font cỡ cố định 14px, không đi qua compute_block_metrics).
        return margin_x, margin_y, width - margin_x, int(height * 0.10)

    elif zone == "bottom_bar":
        # Thanh đáy: viên nang tương tự, neo tại `bottom: 3.5%`.
        return margin_x, int(height * 0.90), width - margin_x, height - margin_y
    
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


def estimate_zone_available_box(
    zone: str,
    occupied_zones: Set[str],
    width: int,
    height: int,
    has_top_bar: bool = False,
    has_bottom_bar: bool = False,
) -> Tuple[int, int, int, int]:
    """Ước lượng bounding box (x1, y1, x2, y2) khả dụng THẬT SỰ của 1 zone, có tính đến
    các zone lân cận (`occupied_zones`) đang cùng có nội dung -- ví dụ `top_left` phải
    nhường bớt chỗ nếu `top_center` VÀ `top_right` cũng đang có chữ.

    TẠI SAO CẦN HÀM NÀY (audit 2026-09-12):
    - Logic thu hẹp bề rộng theo zone lân cận này trước đây chỉ tồn tại inline bên trong
      `engine/layout.py::render_html` (biến `zone_max_w`/`zone_max_h`, dùng để đặt CSS
      `max-width`/`max-height` cho script autofit của trình duyệt) -- HOÀN TOÀN tách biệt
      với `compute_block_metrics()` bên dưới (dùng `get_zone_bounding_box()` trần, không
      biết zone lân cận). Trên đường render chính (Chromium), sự lệch nhau này vô hại vì
      script shrink-to-fit trong master.html luôn co chữ vừa khít CSS max-width/max-height
      thật ở bước cuối. Nhưng trên đường fallback offline (`generate_omni_corridor_mask`,
      PIL, không có bước co chữ nào), `compute_block_metrics` là SỰ THẬT DUY NHẤT -- nếu nó
      ước lượng hào phóng hơn không gian thực sự có, mask sinh ra sẽ nhỏ hơn vùng chữ thật.
    - Hàm này tách logic đó ra làm nguồn chân lý DUY NHẤT, để cả `render_html()` (đường
      chính) và `compute_block_metrics()` (đường fallback) luôn đồng bộ 100% với nhau.
    """
    zx1, zy1, zx2, zy2 = get_zone_bounding_box(zone, width, height)

    if zone == "top_center":
        has_tl = "top_left" in occupied_zones
        has_tr = "top_right" in occupied_zones
        if not has_tl and not has_tr:
            max_w = max(1, int(width * 0.88))
        elif has_tl and has_tr:
            max_w = max(1, int(width * 0.36))
        else:
            max_w = max(1, int(width * 0.52))
    elif zone == "top_left":
        if "top_center" not in occupied_zones:
            max_w = max(int(width * 0.48), zx2 - zx1)
        else:
            has_tr = "top_right" in occupied_zones
            max_w = min(zx2 - zx1, int(width * (0.28 if has_tr else 0.35)))
    elif zone == "top_right":
        if "top_center" not in occupied_zones:
            max_w = max(int(width * 0.38), zx2 - zx1)
        else:
            has_tl = "top_left" in occupied_zones
            max_w = min(zx2 - zx1, int(width * (0.28 if has_tl else 0.35)))
    elif zone == "middle_center":
        has_ml = "middle_left" in occupied_zones
        has_mr = "middle_right" in occupied_zones
        if not has_ml and not has_mr:
            max_w = max(1, int(width * 0.88))
        elif has_ml and has_mr:
            max_w = max(1, int(width * 0.36))
        else:
            max_w = max(1, int(width * 0.50))
    elif zone in ("middle_left", "middle_right"):
        max_w = max(1, zx2 - zx1)
    elif zone == "bottom_center":
        has_bl = "bottom_left" in occupied_zones
        has_br = "bottom_right" in occupied_zones
        if not has_bl and not has_br:
            max_w = max(1, int(width * 0.88))
        elif has_bl and has_br:
            max_w = max(1, int(width * 0.34))
        else:
            max_w = max(1, int(width * 0.36))
    elif zone == "bottom_left":
        has_bc = "bottom_center" in occupied_zones
        max_w = min(zx2 - zx1, int(width * 0.34)) if has_bc else max(1, zx2 - zx1)
    elif zone == "bottom_right":
        has_bc = "bottom_center" in occupied_zones
        max_w = min(zx2 - zx1, int(width * 0.36)) if has_bc else max(1, zx2 - zx1)
    else:
        max_w = max(1, zx2 - zx1)

    base_h = zy2 - zy1
    if has_top_bar and zone in ("top_left", "top_center", "top_right"):
        grid_top_y = int(height * 0.13)
        max_h = min(base_h, max(30, zy2 - grid_top_y))
    elif has_bottom_bar and zone in ("bottom_left", "bottom_center", "bottom_right"):
        grid_bot_y = int(height * 0.87)
        max_h = min(base_h, max(30, grid_bot_y - zy1))
    else:
        max_h = max(1, base_h)

    return zx1, zy1, zx1 + max_w, zy1 + max_h


# ==============================================================================
# PHẦN 3: TÍNH TOÁN METRICS HIỂN THỊ CỦA TỪNG BLOCK (COMPUTE BLOCK METRICS)
# ==============================================================================

# Padding ngang thật (trái+phải, px) của từng container trong `master.html` -- CHƯA từng được
# trừ khỏi `max_w_px` trước 2026-09-13, khiến binary search tin rằng có nhiều chỗ hơn thực tế:
# `fit_font_size_px` chọn font/số dòng vừa khít bề rộng ZONE THÔ, nhưng khi Chromium render thật,
# `.block-card`/`.block-pill`/... còn bị padding CSS ăn bớt (`padding: 10px 18px` v.v., xem
# master.html) -- dòng chữ "vừa" theo PIL lại rộng hơn khung thật trong browser, `white-space:
# pre-line` (Fix C) khiến trình duyệt tự bẻ dòng THÊM một lần nữa -> nhiều dòng ngắn lởm chởm hơn
# hẳn so với những gì server tính (audit trực tiếp: case product_desc tính 4 dòng nhưng render
# thật ra 7 dòng). Số liệu xấp xỉ theo đúng CSS thật, không cần chính xác tuyệt đối từng
# style_variant (nhiều biến thể padding gần nhau) -- chỉ cần đủ để binary search không còn ảo
# tưởng có chỗ rộng hơn thực tế.
_CONTAINER_HPAD_PX: Dict[str, int] = {
    "card": 36,    # .block-card/.block-body: padding: 10px 18px
    "pill": 76,    # .block-pill/.block-badge: padding: 10px 38px (2026-09-13: was 8px 24px --
    "badge": 76,   # bumped alongside the master.html padding fix, see that file's comment;
    "button": 80,  # .block-cta: padding: 10px 40px (was 10px 28px, same fix)
}
# Icon SVG (~14-20px) + flex `gap: 8px` giữa icon và text -- chỉ cộng thêm khi block thực sự có icon.
_ICON_RESERVE_PX = 28


def compute_block_metrics(
    block: AdaptiveBlock,
    font_path: str,
    width: int,
    height: int,
    occupied_zones: Optional[Set[str]] = None,
    has_top_bar: bool = False,
    has_bottom_bar: bool = False,
) -> Dict[str, Any]:
    """Tính toán toàn bộ thông số hiển thị ban đầu cho một AdaptiveBlock.
    
    KẾT QUẢ ĐẦU RA (METRICS DICTIONARY):
    - font_size: Cỡ chữ (px) tối ưu sau khi chạy qua thuật toán nhị phân `fit_font_size_px`.
    - font_weight: Độ đậm font (900 cho xlarge, 800 cho large, 500 cho body...).
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
    4. `occupied_zones`/`has_top_bar`/`has_bottom_bar` (tùy chọn, mặc định None/False giữ
       nguyên hành vi cũ cho caller không truyền): khi được cung cấp, dùng
       `estimate_zone_available_box()` thay vì `get_zone_bounding_box()` trần, để khớp
       chính xác với không gian mà `render_html()` thực sự dành cho zone này khi có zone
       lân cận cùng chiếm chỗ (xem docstring `estimate_zone_available_box`).
    """
    size = block.size if block.size in SIZE_SCALE else "medium"
    container = block.container or "none"
    size_ratio, weight, is_uppercase = SIZE_SCALE[size]

    zone = block.zone or "top_left"
    if occupied_zones is not None:
        x1, y1, x2, y2 = estimate_zone_available_box(
            zone, occupied_zones, width, height,
            has_top_bar=has_top_bar, has_bottom_bar=has_bottom_bar,
        )
    else:
        x1, y1, x2, y2 = get_zone_bounding_box(zone, width, height)
    max_w_px = float(x2 - x1)
    max_h_px = float(y2 - y1)

    text = (block.text or "").strip()
    display_text = text

    # (2026-09-13 fix: trừ padding CSS thật + chỗ icon (nếu có) khỏi ngân sách bề rộng TRƯỚC khi
    # đo/co font -- xem giải thích ở `_CONTAINER_HPAD_PX` phía trên. Không áp dụng cho container
    # "none" (pure-typography không có padding hộp) hay "qr" (đã return sớm, không tới đây).
    hpad = _CONTAINER_HPAD_PX.get(container, 0)
    if block.icon:
        hpad += _ICON_RESERVE_PX
    if hpad:
        max_w_px = max(40.0, max_w_px - hpad)
        # Biên an toàn nhỏ ~5% (2026-09-13): PIL (FreeType) và Chromium (Skia) vẫn có thể đo bề
        # rộng cùng 1 chuỗi lệch nhau đôi chút (hinting/kerning/subpixel khác nhau) dù đo trên
        # CÙNG 1 font -- xem `layout.py::render_html`'s `body_font_path` cho nguyên nhân CHÍNH đã
        # tìm ra và sửa (trước đó card/pill/cta/badge bị đo bằng font TIÊU ĐỀ thay vì đúng font Be
        # Vietnam Pro mà CSS hardcode, gây lệch rất lớn -- 5% ở đây chỉ là biên dự phòng nhỏ cho
        # phần dư sau khi đã dùng đúng font).
        max_w_px = max(40.0, max_w_px * 0.95)

    # --------------------------------------------------------------------------
    # XỬ LÝ KHỐI QR CODE ĐỘC LẬP
    # --------------------------------------------------------------------------
    if container == "qr" or block.field == "qr_code" or getattr(block, "data_uri", None):
        qr_w = 86
        qr_h = 106 if text else 86
        return {
            "text": text,
            "display_text": text,
            "size": size,
            "container": "qr",
            "zone": zone,
            "font_size": 11,
            "font_weight": 700,
            "is_uppercase": False,
            "measured_w": qr_w,
            "measured_h": qr_h,
            "zone_rect": (x1, y1, x2, y2),
        }

    # --------------------------------------------------------------------------
    # XỬ LÝ PURE TYPOGRAPHY (CỠ XLARGE / LARGE, KHÔNG HỘP CHỨA)
    # --------------------------------------------------------------------------
    if container == "none" and size in ("xlarge", "large"):
        # Trường hợp 1: Tiêu đề dài hoặc đoạn thơ nhiều dòng (>= 6 từ hoặc đã có sẵn \n)
        if "\n" in text or len(text.split()) >= 6:
            is_uppercase = False  # Câu dài chuyển sang chữ thường trang nhã để tăng tính dễ đọc
            weight = 800 if size == "xlarge" else 700
            max_w_px = max(max_w_px, float(width * 0.88))
            max_h_px = max(max_h_px, float(y2 - y1))
        
        # Trường hợp 2: Tiêu đề ngắn / trung bình (< 6 từ)
        else:
            if zone in ("top_left", "middle_left", "bottom_left"):
                max_w_px = max(max_w_px, float(width * 0.52))
                target_one_line = 12
            elif zone in ("top_right", "middle_right", "bottom_right"):
                max_w_px = max(max_w_px, float(width * 0.52))
                target_one_line = 12
            else:
                target_one_line = 18

            if len(text.split()) >= 3:
                lines, _ = balance_vietnamese_headline(text, max_one_line_chars=target_one_line)
                if len(lines) > 1:
                    display_text = "\n".join(lines)

    # Cỡ font cơ sở theo tỉ lệ width (ví dụ 0.068 * 1024 = ~70px cho xlarge)
    base_font_size = max(14, int(width * size_ratio))

    # (2026-09-13 fix -- Finding C, phần 2: CHỈ `card`/`body` được phép xuống nhiều dòng --
    # đúng ý đồ thiết kế "chip ngắn 1 dòng" cho `pill`/`button`/`badge` (viền nhũ giá tiền, nút
    # CTA, rating...). Trước đây MỌI container đều dùng nguyên `max_h_px` của cả zone cell (khá
    # cao) làm ngân sách chiều cao cho binary search, khiến `fit_font_size_px` ngầm giả định được
    # phép xuống nhiều dòng và co font xuống rất nhỏ cho pill/badge có text dài -- rồi CSS
    # `nowrap` (master.html) lại ép hiển thị 1 dòng, tràn/vỡ hình. `allow_wrap=False` ép
    # `fit_font_size_px` đo `text` như đúng 1 dòng duy nhất (không ngộ nhận có chỗ xuống dòng),
    # co font tới khi cả chuỗi vừa 1 dòng; phần dư (nếu vẫn không vừa ở min_font_size) được CSS
    # ellipsis (master.html) xử lý.
    allow_multiline = container in ("card", "none")

    # (2026-09-13 fix -- cột hẹp bị Product Sanctuary ăn bớt bề rộng, vd `middle_left` trên
    # canvas 1:1 chỉ còn ~200px: thuật toán vốn LUÔN ưu tiên font_size LỚN NHẤT thoả điều kiện,
    # và `max_h_px` của cả zone cell khá rộng rãi nên "hợp lệ" ngay cả khi mỗi dòng chỉ vừa
    # 1-2 từ -- tạo 1 cột chữ lởm chởm từng dòng lẻ, khó đọc hơn hẳn so với co font nhỏ hơn 1
    # chút để mỗi dòng chứa nhiều từ hơn. Chỉ áp dụng cho `card` (prose dài: mô tả, quote, bước
    # hướng dẫn) -- ép trung bình tối thiểu ~3 từ/dòng bằng cách giới hạn số dòng tối đa theo số
    # từ thực có; quote dài (vd case 09, ~40 từ) gần như không bị ảnh hưởng vì trần này khá rộng.
    max_lines = max(1, len(text.split()) // 3) if container == "card" else None

    # Chạy thuật toán tìm kiếm nhị phân (Binary Search) để tìm font_size lớn nhất
    # mà chuỗi vẫn nằm gọn hoàn hảo bên trong [max_w_px, max_h_px].
    font_size, measured_w, measured_h, fit_lines = fit_font_size_px(
        text=display_text,
        font_path=font_path,
        base_font_size=base_font_size,
        max_width_px=max_w_px,
        max_height_px=max_h_px,
        is_uppercase=is_uppercase,
        allow_wrap=allow_multiline,
        max_lines=max_lines,
    )
    # Trước đây `display_text` chỉ được chèn `\n` xuống dòng cho nhánh pure-typography
    # (xlarge/large, container="none") ở trên. `card`/`body` (quote, mô tả dài) giữ nguyên
    # `text` gốc 1 dòng phẳng, dù `fit_font_size_px` đã ngầm tính font_size "như thể" chữ xuống
    # nhiều dòng để vừa `max_h_px` -- kết quả co xuống cực nhỏ nhưng HTML vẫn gửi 1 chuỗi không
    # `<br>`. Dùng `fit_lines` (đã đúng hoa/thường gốc, xem `wrap_and_measure`'s `display_text`
    # param) cho container được phép xuống dòng.
    if allow_multiline and len(fit_lines) > 1:
        display_text = "\n".join(fit_lines)

    return {
        "text": text,
        "display_text": display_text,
        "size": size,
        "container": container,
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
