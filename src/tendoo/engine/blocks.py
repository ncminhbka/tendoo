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
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tendoo.core.category_schema import get_field_block_spec

logger = logging.getLogger(__name__)

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

# Bảng quy chuẩn tỉ lệ kích thước, độ đậm (weight) và phong cách chữ theo quy mô (size).
# Cấu trúc mỗi tuple: (tỉ lệ cỡ chữ so với chiều rộng canvas, font-weight, is_uppercase)
#
# TẠI SAO CẦN LÀM:
# - Trong thiết kế quảng cáo chuyên nghiệp, sự phân cấp thị giác (Visual Hierarchy) là yếu tố
#   sống còn quyết định người xem sẽ chú ý vào đâu trước. Tiêu đề chính (xlarge) phải vượt trội về
#   kích thước và độ đậm so với các dòng mô tả hay thông tin liên hệ.
# - Khi kích thước canvas thay đổi linh hoạt giữa các tỉ lệ (1:1 là 1024x1024, 9:16 là 576x1024,
#   16:9 là 1024x576), việc dùng tỉ lệ tương đối theo `width` (như 0.068 * width) giúp font chữ
#   tự động co giãn mượt mà, không bị hiện tượng chữ quá nhỏ trên banner 16:9 hay quá to trên story 9:16.
# (2026-09-13: tăng theo thang Golden Ratio ~1.618 -- tỉ lệ trước đây (0.068/0.048/0.028/0.018,
# tương ứng bậc kề nhau 1.417/1.714/1.556) không đều và hơi nhỏ cho tiêu đề poster. Nguyên tắc
# thiết kế (typographic modular scale): nội dung dẫn dắt bằng tiêu đề lớn như poster quảng cáo nên
# dùng thang "mạnh tay" Golden Ratio (1.618) thay vì thang nhẹ Perfect Fourth (1.333) vốn hợp hơn
# cho văn bản web thông thường -- tiêu đề cần đủ lớn để đọc được từ xa. `fit_font_size_px()` vẫn
# luôn co nhỏ lại nếu không đủ chỗ nên tăng tỉ lệ mục tiêu ở đây không phá vỡ bảo đảm chống tràn.)
# (2026-09-13, cùng ngày: người dùng phản hồi trực tiếp trên ảnh render thật rằng medium/small
# (CTA, card, pill, quote, thông số...) trông hơi nhỏ so với xlarge/large -- tăng thêm ~19-20%
# cho riêng 2 bậc này (giữ nguyên xlarge/large). Đây là CỠ MỤC TIÊU/trần trên -- fit_font_size_px()
# vẫn luôn co nhỏ lại khi nội dung dài không đủ chỗ, tăng ở đây không tự nó đảm bảo mọi khối to hơn
# nếu zone quá hẹp cho nội dung đó; xem thêm fix zone riêng cho feedback_rating bên dưới, vốn là
# nguyên nhân CHÍNH khiến case feedback bị co nhỏ tới floor runtime, không phải do tỉ lệ này.)
SIZE_SCALE: Dict[str, Tuple[float, int, bool]] = {
    # Quy mô:   (tỉ lệ font / width, font-weight, viết hoa toàn bộ)
    "xlarge":   (0.084, 900, True),   # Cực lớn: Tiêu đề chính (~86px @1024, từ 0.068)
    "large":    (0.052, 800, True),   # Lớn: Tiêu đề phụ, thông điệp 2 (~53px @1024, từ 0.048)
    "medium":   (0.038, 700, False),  # Vừa: Slogan, nút CTA, trích dẫn (~39px @1024, từ 0.032)
    "small":    (0.024, 600, False),  # Nhỏ: Thông số, ngày tháng, pill (~25px @1024, từ 0.020)
}

# Danh sách trắng các giá trị `container` hợp lệ. "qr" được thêm vào đây 2026-09-12 --
# trước đó bị thiếu, khiến `container="qr"` (do layout.py._extract_blocks gán cho khối
# QR) bị âm thầm reset về "none" ngay tại đây; QR vẫn hoạt động được nhờ các điểm kiểm
# tra `data_uri`/`field=="qr_code"` song song ở nơi khác, nhưng bản thân giá trị
# `container` chưa từng thực sự giữ được "qr" cho tới bản sửa này.
VALID_CONTAINERS: Tuple[str, ...] = ("none", "button", "pill", "card", "qr")

# Danh sách trắng các giá trị `effect` hợp lệ (khớp với các CSS class `.effect-*` thật
# sự tồn tại trong engine/templates/master.html, và với VALID_EFFECTS của
# llm_render_plan_server.py -- xem AdaptiveBlock.__post_init__/from_dict bên dưới).
# Trước 2026-09-12, `effect` không hề được validate (khác với `size`/`container`) -- 1
# effect không hợp lệ (vd từ 1 render-plan LLM) sinh ra class CSS "effect-xxx" không
# khớp rule nào, chữ câm lặng rơi về trơn không hiệu ứng, không log cảnh báo.
VALID_EFFECTS: frozenset = frozenset({
    "none", "auto", "neon_cyan", "neon_pink", "neon_green", "neon_amber",
    "3d_gold", "chrome", "fire", "shadow", "plain", "embossed",
})


# ==============================================================================
# PHẦN 3: CẤU TRÚC DỮ LIỆU ĐẠI DIỆN KHỐI THÍCH ỨNG (ADAPTIVE BLOCK DATACLASS)
# ==============================================================================

@dataclass
class AdaptiveBlock:
    """Đại diện cho một khối chữ hoặc thành phần thích ứng độc lập trên poster quảng cáo.
    
    KIẾN TRÚC 3 TRỤC TRỰC QUAN (3-AXIS VISUAL TYPOGRAPHY):
    1. Trục Quy mô (size): 'xlarge' | 'large' | 'medium' | 'small'
    2. Trục Bao bọc (container): 'none' (Pure text) | 'button' (CTA) | 'pill' (Badge) | 'card' (Box mờ)
    3. Trục Hiệu ứng (effect): 'neon_cyan', 'neon_pink', '3d_gold', 'chrome', 'shadow', 'fire', 'plain'...
    """
    text: str = ""
    zone: Optional[str] = None
    size: str = "medium"
    container: str = "none"
    effect: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    field: Optional[str] = None
    style_variant: str = "default"
    step_index: Optional[int] = None
    data_uri: Optional[str] = None

    def __post_init__(self) -> None:
        if self.text:
            from tendoo.core.typography import strip_emojis
            self.text = strip_emojis(self.text)
        if self.size not in SIZE_SCALE:
            self.size = "medium"
        if self.container not in VALID_CONTAINERS:
            self.container = "none"
        if self.effect is not None and self.effect not in VALID_EFFECTS:
            logger.warning(
                f"AdaptiveBlock: unknown effect '{self.effect}' (text={self.text[:40]!r}), "
                f"discarded -- no CSS rule matches it, would otherwise silently render as plain text."
            )
            self.effect = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary chuẩn phục vụ lưu log, gửi JSON API hoặc cache."""
        d = {
            "text": self.text,
            "zone": self.zone,
            "size": self.size,
            "container": self.container,
            "style_variant": self.style_variant,
        }
        if self.effect:
            d["effect"] = self.effect
        if self.icon:
            d["icon"] = self.icon
        if self.color:
            d["color"] = self.color
        if self.field:
            d["field"] = self.field
        if self.step_index is not None:
            d["step_index"] = self.step_index
        if self.data_uri:
            d["data_uri"] = self.data_uri
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveBlock:
        """Khôi phục đối tượng AdaptiveBlock từ dictionary."""
        s_idx = data.get("step_index")
        parsed_step = int(s_idx) if s_idx is not None and str(s_idx).isdigit() else (s_idx if isinstance(s_idx, int) else None)
        raw_size = data.get("size")
        raw_container = data.get("container")
        raw_effect = data.get("effect")

        size = raw_size if raw_size in SIZE_SCALE else "medium"
        container = raw_container if raw_container in VALID_CONTAINERS else "none"
        if raw_effect is not None and raw_effect not in VALID_EFFECTS:
            logger.warning(f"AdaptiveBlock.from_dict: unknown effect '{raw_effect}', discarded.")
            raw_effect = None

        return cls(
            text=str(data.get("text", "")).strip(),
            zone=data.get("zone"),
            size=size,
            container=container,
            effect=raw_effect,
            icon=data.get("icon"),
            color=data.get("color"),
            field=data.get("field"),
            style_variant=data.get("style_variant", "default"),
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
    similarity_threshold: float = 0.88,
) -> List[AdaptiveBlock]:
    """Thuật toán Content Fingerprint khử trùng lặp 100% bảo vệ tính thẩm mỹ của poster."""
    seen_fingerprints: List[str] = []
    result: List[AdaptiveBlock] = []

    # Ưu tiên các khối cỡ chữ lớn (xlarge > large) xét duyệt đầu tiên để giữ lại kích thước lớn nhất cho thông điệp chính
    sorted_blocks = sorted(blocks, key=lambda b: 0 if b.size == "xlarge" else (1 if b.size == "large" else 2))

    for b in sorted_blocks:
        # Khối QR Code và Rating Badge là thành phần độc lập, không phải văn bản so khớp trùng lặp
        if b.data_uri or b.field in ("qr_code", "feedback_rating") or b.style_variant == "rating_badge":
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
    # (2026-09-12 fix: đã bỏ nhánh `or ("5" in s and num_stars >= 1)` -- bất kỳ chuỗi nào
    # chứa ít nhất 1 "★" VÀ ký tự "5" ở đâu đó (năm, giá, số không liên quan) đều bị ép
    # thành 5 sao đầy. Logic còn lại (đếm "★" thật, rồi fallback regex số) đã tự xử lý
    # đúng mọi trường hợp hợp lệ trong docstring của hàm mà không cần nhánh này.)
    if num_stars >= 5:
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
    # 1. TIÊU ĐỀ CHÍNH (PRIMARY HEADLINE)
    # --------------------------------------------------------------------------
    # - Đối với các ngành cần sự trang trọng, cân bằng chính diện (promo, opening, guide, feedback):
    #   Ưu tiên đặt tại `top_center` để tạo trục đối xứng thị giác mạnh mẽ.
    # - Đối với ngành giới thiệu sản phẩm (`product_intro`): Đặt tại `top_left` để nhường toàn bộ
    #   không gian trung tâm và góc phải phía trên cho ánh sáng và dáng đứng của sản phẩm.
    if title and title.strip():
        blocks.append(AdaptiveBlock(
            text=title.strip(),
            size="xlarge",
            container="none",
            zone="top_center" if cat in ("promo", "opening", "guide", "feedback") else "top_left",
            field="title",
        ))

    # --------------------------------------------------------------------------
    # 2. PHÂN BỔ KHỐI CHỮ THEO ĐẶC THÙ TỪNG DANH MỤC THƯƠNG MẠI
    # --------------------------------------------------------------------------
    if cat == "product_intro":
        name = fields.get("product_name", "")
        if name:
            spec = dict(get_field_block_spec(cat, "product_name"))
            if not title:
                spec["size"] = "xlarge"
            blocks.append(AdaptiveBlock(text=name, zone="top_left", field="product_name", **spec))
        
        price = fields.get("price", "")
        if price:
            blocks.append(AdaptiveBlock(text=price, zone="top_right", field="price", **get_field_block_spec(cat, "price")))

        desc = fields.get("product_desc", "")
        if desc:
            blocks.append(AdaptiveBlock(text=desc, zone="middle_left", field="product_desc", **get_field_block_spec(cat, "product_desc")))
        hl = fields.get("highlights", "")
        if hl:
            blocks.append(AdaptiveBlock(text=hl, zone="middle_left", field="highlights", **get_field_block_spec(cat, "highlights")))

    elif cat == "opening":
        o_date = fields.get("opening_date", "")
        if o_date:
            blocks.append(AdaptiveBlock(text=o_date, zone="middle_right", field="opening_date", **get_field_block_spec(cat, "opening_date")))

        promo = fields.get("opening_promo", "")
        if promo:
            blocks.append(AdaptiveBlock(text=promo, zone="top_center", field="opening_promo", **get_field_block_spec(cat, "opening_promo")))

        contact = fields.get("booking_contact", "")
        if contact:
            blocks.append(AdaptiveBlock(text=contact, zone="bottom_left", field="booking_contact", **get_field_block_spec(cat, "booking_contact")))

    elif cat == "feedback":
        target = fields.get("feedback_target", "")
        if target:
            spec = dict(get_field_block_spec(cat, "feedback_target"))
            if not title:
                spec["size"] = "xlarge"
            blocks.append(AdaptiveBlock(text=target, zone="top_center", field="feedback_target", **spec))

        quote = fields.get("feedback_quote", "")
        if quote:
            blocks.append(AdaptiveBlock(text=f'“{quote}”', zone="middle_left", field="feedback_quote", **get_field_block_spec(cat, "feedback_quote")))

        # (2026-09-13 fix: rating badge trước đây cũng dùng chung "middle_left" với
        # feedback_quote -- 2 khối chồng nhau trong 1 cột hẹp bị Product Sanctuary ăn bớt
        # bề rộng (~230px @1024) khiến script Autofit runtime trong master.html (co CẢ
        # zone, không chỉ riêng phần tử tràn, xem "Pass 2: Sibling Zone Anti-Collision")
        # co cả quote lẫn rating xuống gần floor 8px dù offline đã tính font hợp lý
        # (~20-22px) -- xác nhận trực tiếp bằng cách đọc HTML nguồn thật đã render, thấy
        # font-size offline vẫn ổn nhưng ảnh cuối cùng lại rất nhỏ. Chuyển rating sang
        # "bottom_left" (dưới Sanctuary, rộng rãi hơn nhiều, không ai dùng cho feedback)
        # để không còn chen chúc cùng quote nữa.
        rating = fields.get("feedback_rating") or fields.get("rating") or "5"
        if rating:
            disp_rating = format_rating_stars(rating)
            blocks.append(AdaptiveBlock(text=disp_rating, zone="bottom_left", field="feedback_rating", **get_field_block_spec(cat, "feedback_rating")))

        offer = fields.get("special_offer", "")
        if offer:
            blocks.append(AdaptiveBlock(text=offer, zone="bottom_right", field="special_offer", **get_field_block_spec(cat, "special_offer")))

    elif cat == "recruitment":
        pos = fields.get("job_position", "")
        if pos:
            blocks.append(AdaptiveBlock(text=pos, zone="top_center", field="job_position", **get_field_block_spec(cat, "job_position")))

        desc = fields.get("job_desc", "")
        if desc:
            blocks.append(AdaptiveBlock(text=desc, zone="middle_left", field="job_desc", **get_field_block_spec(cat, "job_desc")))

        deadline = fields.get("apply_deadline", "")
        method = fields.get("apply_method", "")
        meta_parts = []
        if deadline:
            meta_parts.append(f"Hạn: {deadline}")
        if method:
            meta_parts.append(f"Nộp: {method}")
        if meta_parts:
            blocks.append(AdaptiveBlock(text=" | ".join(meta_parts), zone="bottom_right", field="apply_deadline", **get_field_block_spec(cat, "apply_deadline")))

    elif cat == "guide":
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
                size="medium",
                container="card",
                zone="middle_left",
                step_index=idx,
                field=f"step_{idx}",
                style_variant="step_list",
            ))

    else:
        # Mặc định: Ngành Khuyến mại (promo) -- tra bảng bằng key "promo" tường minh (không dùng
        # `cat`), vì nhánh else này là bucket bắt mọi category không khớp ở trên, kể cả 1 chuỗi
        # `cat` lạ không tồn tại trong FIELD_BLOCK_SPECS.
        discount = fields.get("discount", "") or fields.get("offer_main", "")
        if discount:
            blocks.append(AdaptiveBlock(text=discount, zone="top_center", field="discount", **get_field_block_spec("promo", "discount")))

        applied = fields.get("applied_product", "") or fields.get("offer_sub", "")
        if applied:
            blocks.append(AdaptiveBlock(text=applied, zone="middle_right", field="applied_product", **get_field_block_spec("promo", "applied_product")))

        d_start = fields.get("date_start", "")
        d_end = fields.get("date_end", "")
        dates = fields.get("dates", "")
        if not dates and (d_start or d_end):
            dates = f"Từ {d_start} - Đến {d_end}".strip(" -")
        if dates:
            blocks.append(AdaptiveBlock(text=dates, zone="bottom_left", field="dates", **get_field_block_spec("promo", "dates")))

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
                size="small",
                container="pill",
                zone=store_zone,
                field="store_info",
                icon="phone" if phone else "globe",
            ))

    return blocks
