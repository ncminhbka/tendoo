"""
src/tendoo/core/components.py

Tầng cốt lõi quản lý Component đồ họa & SVG Icons:
- SVG Icons vector chuẩn (phone, location, calendar, star, check, gift, tag, clock, globe...).
- Component HTML renderers cho các danh mục thương mại.

TẠI SAO CẦN MODULE NÀY TRONG HỆ THỐNG TENDOO AI:
1. TẠI SAO DÙNG INLINE SVG VECTOR THAY VÌ FONT ICONS HOẶC ẢNH NGOÀI (CDN):
   - Khi render poster thương mại qua Headless Chromium (Playwright/Puppeteer), việc phụ thuộc
     vào web font (như FontAwesome, Material Icons) hay ảnh CDN ngoài tiềm ẩn rủi ro:
     + Chậm trễ mạng hoặc timeout trong môi trường mạng nội bộ cô lập (Offline/Air-gapped server).
     + Hiện tượng FOUC (Flash of Unstyled Content): Chromium chụp ảnh trước khi font kịp render,
       khiến icon bị biến thành ký tự ô vuông lỗi (tofu glyph).
     + Kích thước không co giãn mượt mà theo deviceScaleFactor.
   - Inline SVG nhúng thẳng vào DOM, 100% tự chứa (zero external dependency), render tức thì
     với độ nét vector tuyệt đối ở mọi độ phân giải (1024x1024, 4K, 8K).

2. TẠI SAO CẦN BẢNG ÁNH XẠ NGỮ NGHĨA `ICON_SVG_BY_NAME`:
   - Phân tách tầng dữ liệu và tầng biểu diễn: LLM (Layout Planner) hoặc parser chỉ cần trả về
     chuỗi string ngắn gọn (vd: `"phone"`, `"location"`, `"star"`).
   - Tầng dựng hình (master.html & blocks.py) tra cứu O(1) để lấy SVG chuẩn xác mà không cần
     nhồi mã SVG cồng kềnh vào luồng JSON API.

3. TẠI SAO CÁC HÀM GET_COMPONENT_CSS VÀ RENDER_CATEGORY_BODY LÀ STUBS:
   - Trong kiến trúc OmniBlock mới ("Everything is an Adaptive Block"), toàn bộ định kiểu
     được quản lý tập trung qua CSS variables trong master.html và các AdaptiveBlock.
   - Các hàm này được giữ lại dưới dạng pure stubs để bảo đảm interface an toàn mà không
     phụ thuộc vào bất kỳ thư viện ngoại lai nào.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from tendoo.core.base import (
    ARROW_RIGHT_ICON_SVG,
    BRAIN_ICON_SVG,
    CALENDAR_ICON_SVG,
    CHART_UP_ICON_SVG,
    CHECK_ICON_SVG,
    CLOCK_ICON_SVG,
    FIRE_ICON_SVG,
    GIFT_ICON_SVG,
    GLOBE_ICON_SVG,
    HEART_ICON_SVG,
    LOCATION_ICON_SVG,
    PALETTE_ICON_SVG,
    PHONE_ICON_SVG,
    ROCKET_ICON_SVG,
    SHIELD_ICON_SVG,
    SPARKLES_ICON_SVG,
    STAR_ICON_SVG,
    TAG_ICON_SVG,
    TROPHY_ICON_SVG,
    USERS_ICON_SVG,
    ZAP_ICON_SVG,
)

# Bảng tra cứu SVG vector theo tên ngữ nghĩa ngắn gọn
ICON_SVG_BY_NAME: Dict[str, str] = {
    # Store & Contact
    "phone": PHONE_ICON_SVG,
    "location": LOCATION_ICON_SVG,
    "globe": GLOBE_ICON_SVG,
    "calendar": CALENDAR_ICON_SVG,
    "clock": CLOCK_ICON_SVG,
    "check": CHECK_ICON_SVG,
    "star": STAR_ICON_SVG,
    "arrow_right": ARROW_RIGHT_ICON_SVG,
    "gift": GIFT_ICON_SVG,
    "tag": TAG_ICON_SVG,
    # Technology, Marketing & SaaS (Mới)
    "rocket": ROCKET_ICON_SVG,
    "chart": CHART_UP_ICON_SVG,
    "chart_up": CHART_UP_ICON_SVG,
    "trending_up": CHART_UP_ICON_SVG,
    "palette": PALETTE_ICON_SVG,
    "art": PALETTE_ICON_SVG,
    "brain": BRAIN_ICON_SVG,
    "ai": BRAIN_ICON_SVG,
    "cpu": BRAIN_ICON_SVG,
    "zap": ZAP_ICON_SVG,
    "lightning": ZAP_ICON_SVG,
    "sparkles": SPARKLES_ICON_SVG,
    "magic": SPARKLES_ICON_SVG,
    "fire": FIRE_ICON_SVG,
    "flame": FIRE_ICON_SVG,
    "shield": SHIELD_ICON_SVG,
    "security": SHIELD_ICON_SVG,
    "trophy": TROPHY_ICON_SVG,
    "award": TROPHY_ICON_SVG,
    "heart": HEART_ICON_SVG,
    "users": USERS_ICON_SVG,
}

# ==============================================================================
# BỘ MÁY SUY LUẬN NGỮ NGHĨA ICON TỰ ĐỘNG (SEMANTIC ICON MATCHER)
# ==============================================================================

SEMANTIC_ICON_PATTERNS: List[Tuple[str, List[str]]] = [
    ("rocket", [
        r"tốc\s*độ", r"ánh\s*sáng", r"tên\s*lửa", r"bứt\s*phá", r"siêu\s*tốc",
        r"hỏa\s*tiễn", r"khởi\s*động", r"launch", r"rocket", r"speed", r"fast"
    ]),
    ("chart_up", [
        r"chuyển\s*đổi", r"tối\s*ưu", r"tăng\s*trưởng", r"doanh\s*thu", r"lợi\s*nhuận",
        r"hiệu\s*quả", r"phát\s*triển", r"nâng\s*tầm", r"growth", r"convert", r"roi"
    ]),
    ("palette", [
        r"phong\s*cách", r"sáng\s*tạo", r"thiết\s*kế", r"màu\s*sắc", r"nghệ\s*thuật",
        r"giao\s*diện", r"đa\s*dạng", r"creative", r"design", r"style", r"art"
    ]),
    ("brain", [
        r"thông\s*minh", r"dữ\s*liệu", r"\bai\b", r"trí\s*tuệ", r"xử\s*lý",
        r"tự\s*động", r"công\s*nghệ", r"thuật\s*toán", r"smart", r"data", r"intel"
    ]),
    ("fire", [
        r"bùng\s*nổ", r"cực\s*hot", r"cháy\s*hàng", r"siêu\s*phẩm", r"nóng\s*bỏng",
        r"bán\s*chạy", r"hot", r"fire", r"best\s*seller"
    ]),
    ("zap", [
        r"tức\s*thì", r"chớp\s*nhoáng", r"nhanh\s*chóng", r"năng\s*lượng", r"siêu\s*nhanh",
        r"ngay\s*lập\s*tức", r"lightning", r"flash"
    ]),
    ("sparkles", [
        r"kỳ\s*diệu", r"đột\s*phá", r"tuyệt\s*hảo", r"tỏa\s*sáng", r"lung\s*linh",
        r"đặc\s*biệt", r"cao\s*cấp", r"magic", r"sparkle", r"premium"
    ]),
    ("shield", [
        r"uy\s*tín", r"bảo\s*đảm", r"cam\s*kết", r"an\s*toàn", r"chính\s*hãng",
        r"bảo\s*hành", r"chất\s*lượng", r"tin\s*cậy", r"secure", r"safe"
    ]),
    ("gift", [
        r"ưu\s*đãi", r"quà\s*tặng", r"miễn\s*phí", r"tặng\s*kèm", r"voucher",
        r"tri\s*ân", r"phần\s*thưởng", r"gift", r"free"
    ]),
    ("tag", [
        r"giảm\s*giá", r"sale", r"khuyến\s*mại", r"chiết\s*khấu", r"tiết\s*kiệm",
        r"đồng\s*giá", r"deal", r"offer"
    ]),
    ("trophy", [
        r"vô\s*địch", r"dẫn\s*đầu", r"số\s*1", r"hàng\s*đầu", r"quán\s*quân",
        r"giải\s*thưởng", r"top\s*1", r"winner"
    ]),
    ("heart", [
        r"yêu\s*thích", r"tận\s*tâm", r"sức\s*khỏe", r"chăm\s*sóc", r"hài\s*lòng",
        r"yêu\s*thương", r"passion", r"care"
    ]),
    ("users", [
        r"cộng\s*đồng", r"khách\s*hàng", r"thành\s*viên", r"người\s*dùng",
        r"đội\s*ngũ", r"chuyên\s*gia", r"community", r"team"
    ]),
]


def infer_icon_from_text(text: str) -> Optional[str]:
    """Tự động suy luận icon SVG phù hợp dựa trên nội dung câu văn tiếng Việt.
    
    Ví dụ:
    - "Sáng tạo tốc độ ánh sáng" -> "rocket"
    - "Tối ưu hóa chuyển đổi" -> "chart_up"
    - "Đa dạng phong cách" -> "palette"
    - "Xử lý dữ liệu thông minh" -> "brain"
    """
    if not text:
        return None
    cleaned = text.strip().lower()
    for icon_name, patterns in SEMANTIC_ICON_PATTERNS:
        for pat in patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                return icon_name
    return None


def get_component_css() -> str:
    """Stub tương thích: Trong kiến trúc OmniBlock, CSS được quản lý tập trung trong master.html."""
    return ""


def render_category_body(category: str = "", content: Any = None, *args, **kwargs) -> str:
    """Stub tương thích: Trong kiến trúc OmniBlock, toàn bộ nội dung được phân rã thành AdaptiveBlock."""
    if content and hasattr(content, "category_body_html"):
        return content.category_body_html or ""
    return ""


__all__ = [
    "ARROW_RIGHT_ICON_SVG",
    "BRAIN_ICON_SVG",
    "CALENDAR_ICON_SVG",
    "CHART_UP_ICON_SVG",
    "CHECK_ICON_SVG",
    "CLOCK_ICON_SVG",
    "FIRE_ICON_SVG",
    "GIFT_ICON_SVG",
    "GLOBE_ICON_SVG",
    "HEART_ICON_SVG",
    "ICON_SVG_BY_NAME",
    "LOCATION_ICON_SVG",
    "PALETTE_ICON_SVG",
    "PHONE_ICON_SVG",
    "ROCKET_ICON_SVG",
    "SHIELD_ICON_SVG",
    "SPARKLES_ICON_SVG",
    "STAR_ICON_SVG",
    "TAG_ICON_SVG",
    "TROPHY_ICON_SVG",
    "USERS_ICON_SVG",
    "ZAP_ICON_SVG",
    "get_component_css",
    "infer_icon_from_text",
    "render_category_body",
]

