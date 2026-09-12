"""
src/tendoo/engine/blocks.py

Mô hình Dữ Liệu Khối Chữ Thích Ứng (Adaptive Semantic Block) & Ánh Xạ 6 Danh Mục:
================================================================================
1. TRIẾT LÝ KIẾN TRÚC - "VẠN VẬT LÀ KHỐI" (EVERYTHING IS A BLOCK):
   - Trước đây (hệ thống Legacy), mỗi layout (Top Dome, Diagonal Slash, L-Frame) được code cứng
     các thẻ HTML tương ứng với từng trường form cụ thể (title, discount, dates, store_info).
     Cách làm cũ này gây ra các điểm nghẽn nghiêm trọng:
       + Không thể co giãn khi người dùng nhập thêm text tự do trong prompt.
       + Bị vỡ layout nếu người dùng bỏ trống một trường bắt buộc hoặc nhập câu quá dài.
       + Khó bảo trì vì phải viết CSS riêng cho từng template độc lập.
   - Kiến trúc mới giải quyết triệt để vấn đề này bằng cách trừu tượng hóa: Mọi thành phần hiển thị
     trên poster (từ Tiêu đề chính, Tag giảm giá, Trích dẫn review, Bước hướng dẫn cho đến Thông tin
     cửa hàng và Mã QR) đều là một đối tượng `AdaptiveBlock`.
   - Phân tách tuyệt đối 3 tầng:
       + Tầng Dữ liệu (blocks.py): Quản lý nội dung, vai trò ngữ nghĩa, ô không gian và khử trùng lặp.
       + Tầng Hình học & Mask (geometry.py, mask.py): Đo lường kích thước pixel, tạo hành lang bảo vệ.
       + Tầng Hiển thị (layout.py, master.html): Render HTML5/CSS Glassmorphism và Autofit tự động.

2. CÁC NGUYÊN TẮC THIẾT KẾ BẮT BUỘC ĐƯỢC BẢO ĐẢM TRONG FILE NÀY:
   - Tính Bất biến Không gian (Spatial Normalization): Chuẩn hóa mọi biến thể zone từ LLM.
   - Chống Trùng lặp Tuyệt đối (Content Fingerprint Deduplication): Không bao giờ render 2 lần cùng
     một nội dung do người dùng vừa điền form vừa lặp lại trong prompt.
   - Tính Tất định (Deterministic Fallback 100%): Tự động phân bổ bố cục thị giác tối ưu cho 6 danh
     mục thương mại (promo, product_intro, opening, feedback, recruitment, guide) kể cả khi không có LLM.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA KHÔNG GIAN 9 Ô & BỘ TỪ ĐIỂN ÁNH XẠ MỀM DẺO (FUZZY MAPPING)
# ==============================================================================

# Danh sách các Zones không gian được hệ thống hỗ trợ (9 ô nội vi ma trận 3x3 + 2 thanh bar mép ngoài).
#
# TẠI SAO CẦN LÀM:
# - Tạo danh sách trắng (whitelist) duy nhất để CSS Grid (`master.html`) và thuật toán tạo Mask
#   (`geometry.py`, `mask.py`) đồng bộ tọa độ 1-1 với nhau, triệt tiêu nguy cơ sai lệch định tuyến.
# - Phân tách ô `middle_center`: Vùng trung tâm ảnh được cách ly nghiêm ngặt thành Product Sanctuary
#   (vùng thánh địa của sản phẩm). Text chỉ được rơi vào `middle_center` khi người dùng chủ động
#   yêu cầu poster thuần typography (không có sản phẩm vật lý).
VALID_ZONES = [
    "top_bar",       # Thanh mép đỉnh toàn chiều rộng (chứa Store Info khi có yêu cầu đẩy lên trên)
    "top_left",      # Ô góc trên bên trái (thường dùng cho Tiêu đề hoặc Badge phụ)
    "top_center",    # Ô giữa bên trên (thường dùng cho Tiêu đề chính cân đối, Khai trương, Khuyến mại)
    "top_right",     # Ô góc trên bên phải (thường dùng cho Badge Giá, Tag Khuyến mại, Ngày tháng)
    "middle_left",   # Ô giữa bên trái (thường dùng cho Mô tả sản phẩm, Quy trình các bước, Quote)
    "middle_center", # Ô tâm (MẶC ĐỊNH BỊ CÁCH LY - chỉ dùng khi không có sản phẩm để tránh che chủ thể)
    "middle_right",  # Ô giữa bên phải (thường dùng cho Badge giảm giá, Đánh giá sao, Thông tin phụ)
    "bottom_left",   # Ô góc dưới bên trái (thường dùng cho Thời hạn, Ngày áp dụng, Hotline)
    "bottom_center", # Ô giữa bên dưới (thường dùng cho Nút kêu gọi hành động CTA Button)
    "bottom_right",  # Ô góc dưới bên phải (vị trí tối ưu cho Mã QR Code độc lập, Hạn nộp hồ sơ)
    "bottom_bar",    # Thanh mép đáy toàn chiều rộng (vị trí mặc định chuẩn mực của Store Info capsule)
]

# Bộ từ điển chuyển đổi biến thể chuỗi (Fuzzy Zone Alias Map).
#
# TẠI SAO CẦN LÀM:
# - Các mô hình ngôn ngữ lớn (LLM như Qwen3, GPT, Claude...) hoặc câu lệnh Prompt tự do của người dùng
#   thường sinh ra định dạng bất định (kebab-case `top-left`, space `top left`, viết tắt `tl`, `tc`...).
# - Nếu so khớp chuỗi cứng nhắc (exact match), hệ thống sẽ không nhận diện được, dẫn đến việc block chữ
#   bị gán nhầm vào zone mặc định hoặc làm vỡ cấu trúc CSS `grid-area` trong browser.
# - Bộ ánh xạ này đóng vai trò "màng lọc tiền xử lý" dung hòa mọi cách gọi tên của LLM và con người.
ZONE_ALIAS_MAP: Dict[str, str] = {
    # Hàng 1 (Top row)
    "top_left": "top_left", "top-left": "top_left", "top left": "top_left", "tl": "top_left",
    "top_center": "top_center", "top-center": "top_center", "top center": "top_center", "top": "top_center", "tc": "top_center",
    "top_right": "top_right", "top-right": "top_right", "top right": "top_right", "tr": "top_right",
    # Hàng 2 (Middle row)
    "middle_left": "middle_left", "middle-left": "middle_left", "mid-left": "middle_left", "mid_left": "middle_left", "center-left": "middle_left", "left": "middle_left", "ml": "middle_left",
    "middle_center": "middle_center", "middle-center": "middle_center", "mid-center": "middle_center", "center": "middle_center", "mc": "middle_center",
    "middle_right": "middle_right", "middle-right": "middle_right", "mid-right": "middle_right", "mid_right": "middle_right", "center-right": "middle_right", "right": "middle_right", "mr": "middle_right",
    # Hàng 3 (Bottom row)
    "bottom_left": "bottom_left", "bottom-left": "bottom_left", "bottom left": "bottom_left", "bl": "bottom_left",
    "bottom_center": "bottom_center", "bottom-center": "bottom_center", "bottom center": "bottom_center", "bottom": "bottom_center", "bc": "bottom_center",
    "bottom_right": "bottom_right", "bottom-right": "bottom_right", "bottom right": "bottom_right", "br": "bottom_right",
    # Hai thanh viền toàn cục (Global bars)
    "top_bar": "top_bar", "top-bar": "top_bar", "header": "top_bar",
    "bottom_bar": "bottom_bar", "bottom-bar": "bottom_bar", "footer": "bottom_bar",
}


def normalize_zone(zone: Optional[str], default: str = "top_left") -> str:
    """Chuẩn hóa mọi chuỗi zone đầu vào về 1 trong 11 ô không gian chuẩn của Tendoo.
    
    Quy trình chuẩn hóa 4 bước:
    1. Kiểm tra rỗng/None -> trả về zone mặc định an toàn (`top_left`).
    2. Chuyển chữ thường, cắt bỏ khoảng trắng thừa (`.strip().lower()`), tra cứu nhanh trong alias map.
    3. Chuẩn hóa dấu gạch ngang `-` và khoảng trắng thành dấu gạch dưới `_`, tra cứu lại.
    4. Kiểm tra xem chuỗi có nằm trong danh sách trắng `VALID_ZONES` hay không.
    Nếu thất bại toàn bộ, rơi về `default` để bảo vệ hệ thống không bao giờ crash.
    
    TẠI SAO CẦN LÀM:
    - Đảm bảo tính phòng thủ đa tầng (Defensive Programming). Toàn bộ các module downstream
      (Jinja2 template, CSS Grid area generator, Corridor Mask calculator) hoàn toàn an tâm
      rằng biến `zone` luôn luôn là chuỗi hợp lệ tuyệt đối, không cần phải try/except kiểm tra lại.
    """
    if not zone:
        return default
    cleaned = str(zone).strip().lower()
    if cleaned in ZONE_ALIAS_MAP:
        return ZONE_ALIAS_MAP[cleaned]
    canon = cleaned.replace("-", "_").replace(" ", "_")
    if canon in ZONE_ALIAS_MAP:
        return ZONE_ALIAS_MAP[canon]
    if canon in VALID_ZONES:
        return canon
    return default


# ==============================================================================
# PHẦN 2: THIẾT KẾ PHÂN CẤP THỊ GIÁC (VISUAL TYPOGRAPHY HIERARCHY)
# ==============================================================================

# Bảng quy chuẩn tỉ lệ kích thước, độ đậm (weight) và phong cách chữ theo vai trò.
# Cấu trúc mỗi tuple: (tỉ lệ cỡ chữ so với chiều rộng canvas, font-weight, is_uppercase)
#
# TẠI SAO CẦN LÀM:
# - Trong thiết kế quảng cáo chuyên nghiệp, sự phân cấp thị giác (Visual Hierarchy) là yếu tố
#   sống còn quyết định người xem sẽ chú ý vào đâu trước. Tiêu đề chính (Hero) phải vượt trội về
#   kích thước và độ đậm so với các dòng mô tả hay thông tin liên hệ.
# - Khi kích thước canvas thay đổi linh hoạt giữa các tỉ lệ (1:1 là 1024x1024, 9:16 là 576x1024,
#   16:9 là 1024x576), việc dùng tỉ lệ tương đối theo `width` (như 0.068 * width) giúp font chữ
#   tự động co giãn mượt mà, không bị hiện tượng chữ quá nhỏ trên banner 16:9 hay quá to trên story 9:16.
# - Riêng vai trò `qr`: có `size_ratio = 0.0` vì mã QR là hình ảnh ma trận điểm pixel, được quản lý
#   kích thước độc lập trong CSS card (`.block-qr`), không phụ thuộc vào tỉ lệ font văn bản.
ROLE_SCALE: Dict[str, Tuple[float, int, bool]] = {
    # Vai trò:   (tỉ lệ font / width, font-weight, viết hoa toàn bộ)
    "hero":       (0.068, 900, True),   # Tiêu đề chính: Chiếm sóng thị giác lớn nhất, đập vào mắt ngay
    "subtitle":   (0.030, 600, False),  # Tiêu đề phụ: Hỗ trợ ngữ nghĩa cho Hero, vừa vặn tinh tế
    "badge":      (0.024, 800, True),   # Huy hiệu / Pill tag (Giảm giá, Khai trương, Giá tiền): Đậm, gọn gàng
    "body":       (0.021, 500, False),  # Nội dung thân bài, câu mô tả sản phẩm, trích dẫn khách hàng
    "caption":    (0.016, 400, False),  # Chú thích nhỏ, điểm nổi bật, lưu ý phụ
    "meta":       (0.018, 500, False),  # Dữ liệu thời gian, hotline, hạn nộp hồ sơ
    "brand_bar":  (0.016, 500, False),  # Thanh thông tin thương hiệu (Brand • Hotline • Address)
    "step_list":  (0.020, 600, False),  # Danh sách bước thực hiện trong quy trình (01, 02, 03...)
    "qr":         (0.0,   400, False),  # Khối mã QR (kích thước tính theo pixel card, không tính theo font)
}


# ==============================================================================
# PHẦN 3: CẤU TRÚC DỮ LIỆU ĐẠI DIỆN KHỐI THÍCH ỨNG (ADAPTIVE BLOCK DATACLASS)
# ==============================================================================

@dataclass
class AdaptiveBlock:
    """Đại diện cho một khối chữ hoặc thành phần thích ứng độc lập trên poster quảng cáo.
    
    CÁC THUỘC TÍNH:
    - text: Chuỗi văn bản hiển thị (được tự động escape HTML khi render).
    - role: Vai trò ngữ nghĩa ('hero', 'subtitle', 'badge', 'body', 'meta', 'step_list', 'qr'...).
    - zone: Ô không gian được gán trên poster ('top_left', 'bottom_right', 'bottom_bar'...).
    - field: Tên trường form gốc sinh ra khối này (ví dụ 'discount', 'opening_date') phục vụ trace log.
    - style_variant: Biến thể thẩm mỹ CSS ('pill_badge', 'luxury_tag', 'calendar_box', 'cta_button'...).
    - icon: Tên biểu tượng SVG đính kèm ('phone', 'star', 'tag', 'calendar', 'gift', 'globe'...).
    - color: Mã màu tùy biến ghi đè nếu cần thiết.
    - step_index: Số thứ tự bước (chỉ dùng cho ngành hàng Guide quy trình: 1, 2, 3...).
    - data_uri: Chuỗi Base64 Data URI (chỉ dùng cho khối QR Code: 'data:image/png;base64,...').
    
    TẠI SAO CẦN LÀM:
    - Đây là đơn vị trao đổi dữ liệu chuẩn (Standard DTO) xuyên suốt từ tầng Form Input -> LLM Planner
      -> Layout Engine -> CSS Jinja2 Renderer -> Headless Browser Measure.
    - Tách rời hoàn toàn "Nội dung mang theo" khỏi "Cách hiển thị cụ thể": Một khối `badge` có thể
      hiển thị dạng viền vàng kim (`luxury_tag`) hoặc viên thuốc chuyển màu (`pill_badge`) chỉ bằng cách
      thay đổi thuộc tính `style_variant`, không cần can thiệp logic code xử lý text.
    """
    text: str = ""
    role: str = "body"
    zone: Optional[str] = None
    field: Optional[str] = None
    style_variant: str = "default"
    icon: Optional[str] = None
    color: Optional[str] = None
    step_index: Optional[int] = None
    data_uri: Optional[str] = None

    def __post_init__(self) -> None:
        if self.text:
            from tendoo.layouts.text_engine import strip_emojis
            self.text = strip_emojis(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary chuẩn phục vụ lưu log, gửi JSON API hoặc cache."""
        d = {
            "text": self.text,
            "role": self.role,
            "zone": self.zone,
            "field": self.field,
            "style_variant": self.style_variant,
        }
        if self.icon:
            d["icon"] = self.icon
        if self.color:
            d["color"] = self.color
        if self.step_index is not None:
            d["step_index"] = self.step_index
        if self.data_uri:
            d["data_uri"] = self.data_uri
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveBlock:
        """Khôi phục đối tượng AdaptiveBlock từ dictionary (từ LLM Planner sidecar hoặc request)."""
        s_idx = data.get("step_index")
        parsed_step = int(s_idx) if s_idx is not None and str(s_idx).isdigit() else (s_idx if isinstance(s_idx, int) else None)
        return cls(
            text=str(data.get("text", "")).strip(),
            role=data.get("role", "body"),
            zone=data.get("zone"),
            field=data.get("field"),
            style_variant=data.get("style_variant", "default"),
            icon=data.get("icon"),
            color=data.get("color"),
            step_index=parsed_step,
            data_uri=data.get("data_uri"),
        )


# ==============================================================================
# PHẦN 4: THUẬT TOÁN KHỬ TRÙNG LẶP NỘI DUNG (CONTENT FINGERPRINT DEDUPLICATION)
# ==============================================================================

def normalize_fingerprint(text: str) -> str:
    """Tạo chuỗi vân tay chuẩn hóa (canonical fingerprint) phục vụ việc phát hiện trùng lặp.
    
    Quy trình chuẩn hóa vân tay:
    1. Chuyển toàn bộ chuỗi về chữ thường không phân biệt hoa thường (`.lower()`).
    2. Loại bỏ sạch sẽ toàn bộ dấu câu và ký tự đặc biệt, nhưng BẢO TỒN ký tự ngôi sao (★, ☆).
    3. Gom toàn bộ khoảng trắng thừa, tab, ký tự xuống dòng `\n` về đúng 1 dấu cách đơn.
    """
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s★☆]", "", t)
    return " ".join(t.split())


def deduplicate_blocks(
    blocks: List[AdaptiveBlock],
    hero_title: Optional[str] = None,
    similarity_threshold: float = 0.88,
) -> List[AdaptiveBlock]:
    """Thuật toán Content Fingerprint khử trùng lặp 100% bảo vệ tính thẩm mỹ của poster."""
    seen_fingerprints: List[str] = []
    result: List[AdaptiveBlock] = []

    # Ưu tiên hero title xét duyệt đầu tiên để giữ lại kích thước lớn nhất cho thông điệp chính
    sorted_blocks = sorted(blocks, key=lambda b: 0 if b.role == "hero" else 1)

    for b in sorted_blocks:
        # Khối QR Code và Rating Badge là thành phần độc lập, không phải văn bản so khớp trùng lặp
        if b.role == "qr" or b.field == "feedback_rating" or b.style_variant == "rating_badge":
            result.append(b)
            continue

        text = b.text.strip()
        if not text:
            continue

        fp = normalize_fingerprint(text)
        if not fp:
            continue

        # Kiểm tra trùng lặp với các khối đã ghi nhận trước đó
        is_dup = False
        for s_fp in seen_fingerprints:
            if fp == s_fp:
                is_dup = True
                break
            if difflib.SequenceMatcher(None, fp, s_fp).ratio() >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            result.append(b)
            seen_fingerprints.append(fp)

    return result


# ==============================================================================
# PHẦN 5: BỘ DỊCH NGỮ NGHĨA MẶC ĐỊNH CHO 6 NGÀNH HÀNG (DETERMINISTIC FALLBACK)
# ==============================================================================

def format_rating_stars(val: Any) -> str:
    """Chuyển đổi linh hoạt giá trị rating thành chuỗi 5 ngôi sao vàng chuẩn mực (★★★★★).
    
    TẠI SAO CẦN LÀM:
    - Người dùng có thể nhập số '5', '5.0', '5 sao', '5 ★★★★★', '⭐⭐⭐⭐⭐' hoặc 4.5.
    - Để đạt tính thẩm mỹ cao nhất trên poster quảng cáo, huy hiệu rating (`rating_badge`)
      phải hiển thị trực quan các ngôi sao vàng (ví dụ: '★★★★★' hoặc '★★★★☆') thay vì chỉ hiển thị
      đơn độc một con số khô khan.
    - Ký tự '★' (\u2605) là vector typographical glyph chuẩn, không bị vỡ font hay biến thành tofu (□).
    """
    if not val:
        return "★★★★★"
    s = str(val).strip().replace("⭐", "★")
    # Nếu có phân số tỉ lệ điểm như "5.0 / 5.0" hoặc "4.8 / 5.0"
    if "/" in s:
        num_stars = s.count("★")
        if num_stars == 0:
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            score = float(m.group(1)) if m else 5.0
            full = min(5, max(1, int(round(score))))
            stars = "★" * full + "☆" * (5 - full)
            return f"{stars} {s}"
        return s

    num_stars = s.count("★")
    if num_stars >= 5 or ("5" in s and num_stars >= 1):
        return "★★★★★"
    if num_stars >= 1:
        return "★" * min(5, num_stars)
    # Tìm kiếm con số trong chuỗi (ví dụ '5', '5.0', '4', '4.5')
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        try:
            score = float(m.group(1))
            full_stars = min(5, max(1, int(round(score))))
            return "★" * full_stars + "☆" * (5 - full_stars)
        except Exception:
            pass
    return "★★★★★"


def map_category_to_default_blocks(
    category: str,
    title: Optional[str],
    fields: Dict[str, str],
    store_info: Optional[Dict[str, str]] = None,
    spatial_override_zone: Optional[str] = None,
) -> List[AdaptiveBlock]:
    """Ánh xạ ngữ nghĩa mặc định từ các trường Form sang danh sách AdaptiveBlock cho 6 Danh mục.
    
    CÁC DANH MỤC ĐƯỢC HỖ TRỢ:
    1. promo (Khuyến mại - Giảm giá)
    2. product_intro (Giới thiệu sản phẩm mới)
    3. opening (Khai trương cửa hàng / chi nhánh)
    4. feedback (Đánh giá & Trải nghiệm khách hàng)
    5. recruitment (Tuyển dụng nhân sự)
    6. guide (Quy trình hướng dẫn theo các bước đánh số)
    
    TẠI SAO CẦN LÀM:
    - BẢO ĐẢM TÍNH TẤT ĐỊNH 100% (DETERMINISTIC RELIABILITY): Khi hệ thống hoạt động mà không có LLM
      sidecar (do lỗi mạng, hết tài nguyên hoặc người dùng tắt chế độ AI phức tạp), poster vẫn phải
      được tạo ra với bố cục chuyên nghiệp, cân đối và chuẩn thẩm mỹ thị giác.
    - PHÂN BỔ NGOẠI VI BẢO VỆ SANCTUARY: Mọi khối chữ được tính toán đặt ở các ô viền (Perimeter Zones:
      top_left, top_center, top_right, middle_left, bottom_right...), tuyệt đối không để chữ rơi vào
      tâm ảnh (`middle_center`), đảm bảo khu vực trung tâm luôn sạch sẽ để hiển thị sản phẩm chính.
    """
    cat = (category or "promo").lower().strip()
    blocks: List[AdaptiveBlock] = []

    # --------------------------------------------------------------------------
    # 1. TIÊU ĐỀ CHÍNH (HERO TITLE)
    # --------------------------------------------------------------------------
    # - Đối với các ngành cần sự trang trọng, cân bằng chính diện (promo, opening, guide, feedback):
    #   Ưu tiên đặt tại `top_center` để tạo trục đối xứng thị giác mạnh mẽ.
    # - Đối với ngành giới thiệu sản phẩm (`product_intro`): Đặt tại `top_left` để nhường toàn bộ
    #   không gian trung tâm và góc phải phía trên cho ánh sáng và dáng đứng của sản phẩm.
    if title and title.strip():
        blocks.append(AdaptiveBlock(
            text=title.strip(),
            role="hero",
            zone="top_center" if cat in ("promo", "opening", "guide", "feedback") else "top_left",
            field="title",
        ))

    # --------------------------------------------------------------------------
    # 2. PHÂN BỔ KHỐI CHỮ THEO ĐẶC THÙ TỪNG DANH MỤC THƯƠNG MẠI
    # --------------------------------------------------------------------------
    if cat == "product_intro":
        # Tên sản phẩm: Nếu form chưa có title riêng thì lấy tên làm Hero; nếu đã có thì làm Subtitle
        name = fields.get("product_name", "")
        if name:
            if not title:
                blocks.append(AdaptiveBlock(text=name, role="hero", zone="top_left", field="product_name"))
            else:
                blocks.append(AdaptiveBlock(text=name, role="subtitle", zone="top_left", field="product_name"))
        
        # Giá tiền: Đặt tại `top_right` đối xứng với tên sản phẩm, dùng phong cách thẻ vàng sang trọng (`luxury_tag`)
        price = fields.get("price", "")
        if price:
            blocks.append(AdaptiveBlock(text=price, role="badge", zone="top_right", field="price", icon="tag", style_variant="luxury_tag"))
        
        # Mô tả ngắn và điểm nổi bật: Gom về cạnh sườn trái `middle_left` tạo thành một khối thông tin gọn gàng
        desc = fields.get("product_desc", "")
        if desc:
            blocks.append(AdaptiveBlock(text=desc, role="body", zone="middle_left", field="product_desc"))
        hl = fields.get("highlights", "")
        if hl:
            blocks.append(AdaptiveBlock(text=hl, role="caption", zone="middle_left", field="highlights", icon="check"))

    elif cat == "opening":
        # Ngày khai trương: Đặt tại `middle_right` dạng hộp lịch (`calendar_box`), nổi bật để khách ghi nhớ thời gian
        o_date = fields.get("opening_date", "")
        if o_date:
            blocks.append(AdaptiveBlock(text=o_date, role="badge", zone="middle_right", field="opening_date", icon="calendar", style_variant="calendar_box"))
        
        # Ưu đãi ngày mở bán: Đặt ngay dưới tiêu đề chính tại `top_center`
        promo = fields.get("opening_promo", "")
        if promo:
            blocks.append(AdaptiveBlock(text=promo, role="subtitle", zone="top_center", field="opening_promo"))
        
        # Hotline đặt bàn / liên hệ: Đặt ở góc dưới bên trái `bottom_left`
        contact = fields.get("booking_contact", "")
        if contact:
            blocks.append(AdaptiveBlock(text=contact, role="meta", zone="bottom_left", field="booking_contact", icon="phone"))

    elif cat == "feedback":
        # Đối tượng / Gói dịch vụ được đánh giá: Hiển thị trang nhã ở `top_center`
        target = fields.get("feedback_target", "")
        if target:
            if not title:
                blocks.append(AdaptiveBlock(text=target, role="hero", zone="top_center", field="feedback_target"))
            else:
                blocks.append(AdaptiveBlock(text=target, role="subtitle", zone="top_center", field="feedback_target"))
        
        # Lời nhận xét (Quote): Đặt tại `middle_left` với style thẻ trích dẫn (`quote`) có vạch nhấn màu accent
        quote = fields.get("feedback_quote", "")
        if quote:
            blocks.append(AdaptiveBlock(text=f'“{quote}”', role="body", zone="middle_left", field="feedback_quote", style_variant="quote"))
        
        # Đánh giá sao: Hiển thị 5 sao vàng nổi bật (`rating_badge`) tại `middle_left` ngay dưới lời nhận xét
        rating = fields.get("feedback_rating") or fields.get("rating") or "5"
        if rating:
            disp_rating = format_rating_stars(rating)
            blocks.append(AdaptiveBlock(text=disp_rating, role="badge", zone="middle_left", field="feedback_rating", icon="star", style_variant="rating_badge"))
        
        # Ưu đãi tri ân đặc biệt: Đặt tại `bottom_right` dưới dạng nút kêu gọi hành động bắt mắt (`cta_button`)
        offer = fields.get("special_offer", "")
        if offer:
            blocks.append(AdaptiveBlock(text=offer, role="badge", zone="bottom_right", field="special_offer", icon="gift", style_variant="cta_button"))

    elif cat == "recruitment":
        # Vị trí ứng tuyển: Đặt ở `top_center` nổi bật
        pos = fields.get("job_position", "")
        if pos:
            blocks.append(AdaptiveBlock(text=pos, role="badge", zone="top_center", field="job_position"))
        
        # Mô tả quyền lợi & công việc: Nằm gọn ở `middle_left`
        desc = fields.get("job_desc", "")
        if desc:
            blocks.append(AdaptiveBlock(text=desc, role="body", zone="middle_left", field="job_desc"))
        
        # Hạn nộp & phương thức nộp hồ sơ: Gom chung vào góc `bottom_right` dạng thẻ lịch hẹn
        deadline = fields.get("apply_deadline", "")
        method = fields.get("apply_method", "")
        meta_parts = []
        if deadline:
            meta_parts.append(f"Hạn: {deadline}")
        if method:
            meta_parts.append(f"Nộp: {method}")
        if meta_parts:
            blocks.append(AdaptiveBlock(text=" | ".join(meta_parts), role="meta", zone="bottom_right", field="apply_deadline", icon="clock", style_variant="calendar_box"))

    elif cat == "guide":
        # Quy trình nhiều bước: Hỗ trợ cả list từ form và chuỗi nối bằng dấu gạch đứng '|'
        # Toàn bộ các bước được gom vào `middle_left` dưới dạng thẻ bước đánh số (`step_list`)
        # Số thứ tự `step_index` (01, 02, 03...) sẽ được CSS render thành vòng tròn nổi bật
        steps_val = fields.get("guide_steps", "")
        if isinstance(steps_val, list):
            step_list = [str(s).strip() for s in steps_val if str(s).strip()]
        elif isinstance(steps_val, str) and steps_val.strip():
            step_list = [s.strip() for s in steps_val.split("|") if s.strip()]
        else:
            step_list = []
        for idx, s in enumerate(step_list, 1):
            blocks.append(AdaptiveBlock(
                text=s,
                role="step_list",
                zone="middle_left",
                step_index=idx,
                field=f"step_{idx}",
            ))

    else:
        # Mặc định: Ngành Khuyến mại (promo)
        # Tag giảm giá: Đặt ở `middle_right` dạng viên thuốc đỏ cam thu hút ánh nhìn (`pill_badge`)
        discount = fields.get("discount", "") or fields.get("offer_main", "")
        if discount:
            blocks.append(AdaptiveBlock(text=discount, role="badge", zone="middle_right", field="discount", icon="tag", style_variant="pill_badge"))
        
        # Sản phẩm áp dụng: Hiển thị nhẹ nhàng dưới tiêu đề chính
        applied = fields.get("applied_product", "") or fields.get("offer_sub", "")
        if applied:
            blocks.append(AdaptiveBlock(text=applied, role="subtitle", zone="top_center", field="applied_product"))
        
        # Thời hạn chương trình: Đặt ở `bottom_left` dạng thẻ thời gian
        d_start = fields.get("date_start", "")
        d_end = fields.get("date_end", "")
        dates = fields.get("dates", "")
        if not dates and (d_start or d_end):
            dates = f"Từ {d_start} - Đến {d_end}".strip(" -")
        if dates:
            blocks.append(AdaptiveBlock(text=dates, role="meta", zone="bottom_left", field="dates", icon="calendar", style_variant="calendar_box"))

    # --------------------------------------------------------------------------
    # 3. THÔNG TIN CỬA HÀNG (STORE INFORMATION CAPSULE)
    # --------------------------------------------------------------------------
    # TẠI SAO CẦN LÀM:
    # - Gom toàn bộ Brand, Hotline, Address thành 1 chuỗi duy nhất cách nhau bởi dấu chấm " • "
    #   và đưa vào `bottom_bar` (hoặc `top_bar` nếu tham số spatial_override_zone yêu cầu).
    # - Tránh việc mỗi thông tin cửa hàng phân mảnh thành các ô lẻ tẻ gây rối mắt.
    # - Định dạng capsule pill có `width: fit-content` đảm bảo thanh thông tin ôm vừa khít chữ,
    #   không bị thừa khoảng đen vô nghĩa sang hai bên mép.
    if store_info:
        store_parts = []
        brand = store_info.get("store_name") or store_info.get("brand")
        phone = store_info.get("phone") or store_info.get("hotline")
        addr = store_info.get("address")
        if brand:
            store_parts.append(brand)
        if phone:
            store_parts.append(f"Hotline: {phone}")
        if addr:
            store_parts.append(addr)

        if store_parts:
            store_zone = spatial_override_zone or "bottom_bar"
            blocks.append(AdaptiveBlock(
                text="  •  ".join(store_parts),
                role="brand_bar",
                zone=store_zone,
                field="store_info",
                icon="phone" if phone else "globe",
            ))

    return blocks
