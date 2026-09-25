"""
src/tendoo_v3/icons.py

Hệ thống Biểu tượng Vector SVG & QR Code Engine cho Tendoo v3:
- Zero-dependency: Vẽ trực tiếp bằng inline SVG với `stroke="currentColor"`, không phụ thuộc webfont.
- Semantic Icon Matcher: Tự động phân tích ngữ nghĩa câu từ tiếng Việt (chống nước, ưu đãi, pin, bảo hành...)
  để gán biểu tượng vector tương ứng cho từng pill / tính năng.
- QR Code Engine: Khung mã QR đồ họa hiện đại kèm nhãn quét ("QUÉT MÃ NGAY / NHẬN DEAL").
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# BỘ ICON VECTOR SVG CHUẨN THƯƠNG MẠI
# ==============================================================================

STAR_ICON_SVG = (
    '<svg class="icon-svg icon-star" width="16" height="16" viewBox="0 0 24 24" fill="#FFB300" stroke="#FFB300" stroke-width="1" '
    'style="display:inline-block; vertical-align:-2px; margin-right:2px;">'
    '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>'
    '</svg>'
)

CHECK_ICON_SVG = (
    '<svg class="icon-svg icon-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px;">'
    '<polyline points="20 6 9 17 4 12"></polyline>'
    '</svg>'
)

PHONE_ICON_SVG = (
    '<svg class="icon-svg icon-phone" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.9;">'
    '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>'
    '</svg>'
)

LOCATION_ICON_SVG = (
    '<svg class="icon-svg icon-location" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.9;">'
    '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>'
    '<circle cx="12" cy="10" r="3"></circle>'
    '</svg>'
)

CALENDAR_ICON_SVG = (
    '<svg class="icon-svg icon-calendar" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.9;">'
    '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>'
    '<line x1="16" y1="2" x2="16" y2="6"></line>'
    '<line x1="8" y1="2" x2="8" y2="6"></line>'
    '<line x1="3" y1="10" x2="21" y2="10"></line>'
    '</svg>'
)

CLOCK_ICON_SVG = (
    '<svg class="icon-svg icon-clock" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.9;">'
    '<circle cx="12" cy="12" r="10"></circle>'
    '<polyline points="12 6 12 12 16 14"></polyline>'
    '</svg>'
)

GIFT_ICON_SVG = (
    '<svg class="icon-svg icon-gift" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<polyline points="20 12 20 22 4 22 4 12"></polyline>'
    '<rect x="2" y="7" width="20" height="5"></rect>'
    '<line x1="12" y1="22" x2="12" y2="7"></line>'
    '<path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z"></path>'
    '<path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z"></path>'
    '</svg>'
)

TAG_ICON_SVG = (
    '<svg class="icon-svg icon-tag" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.9;">'
    '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path>'
    '<line x1="7" y1="7" x2="7.01" y2="7"></line>'
    '</svg>'
)

SHIELD_ICON_SVG = (
    '<svg class="icon-svg icon-shield" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>'
    '</svg>'
)

FIRE_ICON_SVG = (
    '<svg class="icon-svg icon-fire" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"></path>'
    '</svg>'
)

ZAP_ICON_SVG = (
    '<svg class="icon-svg icon-zap" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>'
    '</svg>'
)

SPARKLES_ICON_SVG = (
    '<svg class="icon-svg icon-sparkles" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>'
    '<path d="M5 3v4"></path><path d="M19 17v4"></path><path d="M3 5h4"></path><path d="M17 19h4"></path>'
    '</svg>'
)

ROCKET_ICON_SVG = (
    '<svg class="icon-svg icon-rocket" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path>'
    '<path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path>'
    '</svg>'
)

CHART_UP_ICON_SVG = (
    '<svg class="icon-svg icon-chart" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline>'
    '<polyline points="16 7 22 7 22 13"></polyline>'
    '</svg>'
)

BRAIN_ICON_SVG = (
    '<svg class="icon-svg icon-brain" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"></path>'
    '<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"></path>'
    '</svg>'
)

USERS_ICON_SVG = (
    '<svg class="icon-svg icon-users" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>'
    '<circle cx="9" cy="7" r="4"></circle>'
    '<path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>'
    '<path d="M16 3.13a4 4 0 0 1 0 7.75"></path>'
    '</svg>'
)

QUOTE_ICON_SVG = (
    '<svg class="icon-svg icon-quote" width="22" height="22" viewBox="0 0 24 24" fill="currentColor" opacity="0.4" '
    'style="display:inline-block; vertical-align:-4px; margin-right:6px;">'
    '<path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z"></path>'
    '<path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z"></path>'
    '</svg>'
)

ARROW_RIGHT_ICON_SVG = (
    '<svg class="icon-svg icon-arrow-right" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-left:4px;">'
    '<line x1="5" y1="12" x2="19" y2="12"></line>'
    '<polyline points="12 5 19 12 12 19"></polyline>'
    '</svg>'
)

WATER_DROP_ICON_SVG = (
    '<svg class="icon-svg icon-water" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>'
    '</svg>'
)

BATTERY_ICON_SVG = (
    '<svg class="icon-svg icon-battery" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<rect x="1" y="6" width="18" height="12" rx="2" ry="2"></rect>'
    '<line x1="23" y1="13" x2="23" y2="11"></line>'
    '<line x1="6" y1="10" x2="6" y2="14"></line>'
    '<line x1="10" y1="10" x2="10" y2="14"></line>'
    '</svg>'
)

GLOBE_ICON_SVG = (
    '<svg class="icon-svg icon-globe" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.9;">'
    '<circle cx="12" cy="12" r="10"></circle>'
    '<line x1="2" y1="12" x2="22" y2="12"></line>'
    '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>'
    '</svg>'
)

TROPHY_ICON_SVG = (
    '<svg class="icon-svg icon-trophy" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path>'
    '<path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path>'
    '<path d="M4 22h16"></path>'
    '<path d="M10 14.66V17c0 .55-.45.98-.96 1.2-1.11.48-1.71 1.04-1.71 1.8h9.34c0-.76-.6-1.32-1.71-1.8-.51-.22-.96-.65-.96-1.2v-2.34"></path>'
    '<path d="M18 2H6v7a6 6 0 0 0 12 0V2z"></path>'
    '</svg>'
)

HEART_ICON_SVG = (
    '<svg class="icon-svg icon-heart" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"></path>'
    '</svg>'
)

PALETTE_ICON_SVG = (
    '<svg class="icon-svg icon-palette" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<circle cx="13.5" cy="6.5" r=".5" fill="currentColor"></circle>'
    '<circle cx="17.5" cy="10.5" r=".5" fill="currentColor"></circle>'
    '<circle cx="8.5" cy="7.5" r=".5" fill="currentColor"></circle>'
    '<circle cx="6.5" cy="12.5" r=".5" fill="currentColor"></circle>'
    '<path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.563-2.512 5.563-5.563C22 6.5 17.5 2 12 2z"></path>'
    '</svg>'
)

LEAF_ICON_SVG = (
    '<svg class="icon-svg icon-leaf" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.95;">'
    '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"></path>'
    '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"></path>'
    '</svg>'
)

# Dấu chấm/sao trang trí (✦) dùng làm bullet cho freetext-block/message-line ở nhiều
# template -- CỐ TÌNH dùng SVG thật (fill="currentColor" để tự nhận đúng theme_color
# qua CSS `color:` của span cha) thay vì ký tự Unicode ✦ (U+2726) chèn cứng trong HTML:
# ký tự Unicode phụ thuộc font HỆ ĐIỀU HÀNH có chứa glyph đó hay không (Arial/Segoe UI
# Symbol chỉ đảm bảo có trên Windows -- server Linux thường KHÔNG có, gây lỗi tofu ô
# vuông trống dù CSS font-family đã khai đúng). SVG vẽ tay không phụ thuộc font nào,
# hiển thị giống hệt trên mọi OS/server.
BULLET_SPARKLE_SVG = (
    '<svg class="icon-svg icon-bullet-sparkle" width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" '
    'style="display:inline-block; vertical-align:-0.05em; flex-shrink:0;">'
    '<path d="M12 2 L14.5 9.5 L22 12 L14.5 14.5 L12 22 L9.5 14.5 L2 12 L9.5 9.5 Z"></path>'
    '</svg>'
)

ICONS_MAP: Dict[str, str] = {
    "star": STAR_ICON_SVG,
    "check": CHECK_ICON_SVG,
    "phone": PHONE_ICON_SVG,
    "location": LOCATION_ICON_SVG,
    "calendar": CALENDAR_ICON_SVG,
    "clock": CLOCK_ICON_SVG,
    "gift": GIFT_ICON_SVG,
    "tag": TAG_ICON_SVG,
    "shield": SHIELD_ICON_SVG,
    "fire": FIRE_ICON_SVG,
    "zap": ZAP_ICON_SVG,
    "sparkles": SPARKLES_ICON_SVG,
    "rocket": ROCKET_ICON_SVG,
    "chart_up": CHART_UP_ICON_SVG,
    "brain": BRAIN_ICON_SVG,
    "users": USERS_ICON_SVG,
    "quote": QUOTE_ICON_SVG,
    "water": WATER_DROP_ICON_SVG,
    "battery": BATTERY_ICON_SVG,
    "globe": GLOBE_ICON_SVG,
    "trophy": TROPHY_ICON_SVG,
    "heart": HEART_ICON_SVG,
    "palette": PALETTE_ICON_SVG,
    "arrow_right": ARROW_RIGHT_ICON_SVG,
    "leaf": LEAF_ICON_SVG,
}


# ==============================================================================
# BỘ MÁY SUY LUẬN NGỮ NGHĨA ICON TỰ ĐỘNG (SEMANTIC ICON MATCHER)
# ==============================================================================

SEMANTIC_ICON_PATTERNS: List[Tuple[str, List[str]]] = [
    ("check", [
        r"đầy\s*đủ", r"bao\s*gồm", r"trọn\s*gói", r"hoàn\s*thiện", r"xác\s*nhận",
        r"đạt\s*yêu\s*cầu", r"đáp\s*ứng", r"included", r"complete", r"verified"
    ]),
    ("leaf", [
        r"tự\s*nhiên", r"hữu\s*cơ", r"organic", r"thuần\s*chay", r"thảo\s*mộc",
        r"thân\s*thiện\s*môi\s*trường", r"nguyên\s*chất", r"tươi", r"xanh\s*sạch",
        r"bền\s*vững", r"eco", r"vegan"
    ]),
    ("phone", [
        r"hotline", r"liên\s*hệ", r"tư\s*vấn", r"gọi", r"điện\s*thoại", r"sđt", r"tel",
        r"1800", r"1900", r"\b0[3|5|7|8|9]\d{8}\b", r"zalo", r"call"
    ]),
    ("location", [
        r"địa\s*chỉ", r"cửa\s*hàng", r"showroom", r"chi\s*nhánh", r"ghé\s*ngay",
        r"tại\s*hà\s*nội", r"tại\s*tphcm", r"cơ\s*sở", r"quận", r"đường", r"phố",
        r"location", r"address"
    ]),
    ("globe", [
        r"website", r"web", r"online", r"toàn\s*quốc", r"toàn\s*cầu", r"internet",
        r"link", r"http", r"www\.", r"\.vn\b", r"\.com\b", r"trực\s*tuyến"
    ]),
    ("calendar", [
        r"khai\s*trương", r"mở\s*bán", r"áp\s*dụng\s*từ", r"đến\s*ngày", r"lịch\s*trình",
        r"hạn\s*chót", r"deadline", r"thứ\s*[2-7]", r"chủ\s*nhật", r"calendar", r"thời\s*hạn\s*nộp"
    ]),
    ("clock", [
        r"thời\s*gian", r"giờ\s*mở\s*cửa", r"giờ", r"phút", r"ngày", r"tuần", r"tháng",
        r"lịch\s*trình", r"24/7", r"chỉ\s*còn", r"thời\s*lượng", r"clock", r"time"
    ]),
    ("trophy", [
        r"vô\s*địch", r"dẫn\s*đầu", r"số\s*1", r"hàng\s*đầu", r"quán\s*quân",
        r"giải\s*thưởng", r"top\s*1", r"winner", r"best\s*choice", r"đạt\s*chuẩn"
    ]),
    ("heart", [
        r"yêu\s*thích", r"tận\s*tâm", r"sức\s*khỏe", r"chăm\s*sóc", r"hài\s*lòng",
        r"yêu\s*thương", r"passion", r"care", r"spa", r"thẩm\s*mỹ", r"dưỡng\s*sinh",
        r"thư\s*giãn", r"massage", r"healthy"
    ]),
    ("palette", [
        r"phong\s*cách", r"sáng\s*tạo", r"thiết\s*kế", r"màu\s*sắc", r"nghệ\s*thuật",
        r"giao\s*diện", r"đa\s*dạng", r"creative", r"design", r"style", r"art", r"custom",
        r"thủ\s*công", r"handmade", r"nghệ\s*nhân", r"thêu\s*tay", r"đan\s*tay"
    ]),
    ("water", [
        r"chống\s*nước", r"kháng\s*nước", r"ngập\s*nước", r"5\s*atm", r"50m", r"waterproof",
        r"bọt\s*khí", r"sục\s*ozon", r"nước\s*khoáng", r"tinh\s*khiết"
    ]),
    ("battery", [
        r"pin", r"năng\s*lượng", r"mặt\s*trời", r"sạc", r"battery", r"power", r"dự\s*phòng"
    ]),
    ("shield", [
        r"bảo\s*hành", r"cam\s*kết", r"an\s*toàn", r"chính\s*hãng", r"uy\s*tín",
        r"chuẩn", r"quân\s*đội", r"titanium", r"thép", r"độ\s*bền", r"bảo\s*mật",
        r"diệt\s*khuẩn", r"bảo\s*vệ", r"chứng\s*nhận"
    ]),
    ("gift", [
        r"ưu\s*đãi", r"quà\s*tặng", r"miễn\s*phí", r"tặng", r"voucher", r"mua\s*1\s*tặng\s*1",
        r"tri\s*ân", r"tặng\s*kèm", r"combo\s*quà", r"gift", r"free"
    ]),
    ("tag", [
        r"giảm\s*\d+%", r"giảm\s*giá", r"sale", r"đồng\s*giá", r"deal", r"tiết\s*kiệm",
        r"giá\s*tốt", r"giá\s*sốc", r"chỉ\s*từ", r"price", r"vnđ", r"đ"
    ]),
    ("fire", [
        r"bùng\s*nổ", r"cực\s*hot", r"cháy\s*hàng", r"siêu\s*phẩm", r"hot", r"đặc\s*biệt",
        r"nóng\s*bỏng", r"bán\s*chạy", r"best\s*seller", r"hot\s*deal"
    ]),
    ("zap", [
        r"tức\s*thì", r"nhanh\s*chóng", r"ngay", r"chớp\s*nhoáng", r"phản\s*xạ",
        r"tốc\s*độ", r"chớp\s*mắt", r"lightning", r"flash\s*sale",
        r"dễ\s*dàng", r"tiện\s*lợi", r"đơn\s*giản", r"nhanh\s*gọn", r"gọn\s*nhẹ"
    ]),
    ("rocket", [
        r"khởi\s*động", r"bứt\s*phá", r"tiên\s*phong", r"đột\s*phá", r"launch", r"siêu\s*tốc"
    ]),
    ("chart_up", [
        r"tăng\s*trưởng", r"vóc\s*dáng", r"thăng\s*tiến", r"kết\s*quả", r"thay\s*đổi",
        r"lột\s*xác", r"hiệu\s*quả", r"tiêu\s*hóa", r"tối\s*ưu", r"roi", r"growth"
    ]),
    ("sparkles", [
        r"sang\s*trọng", r"đẳng\s*cấp", r"vàng", r"rose\s*gold", r"tinh\s*tế",
        r"lấp\s*lánh", r"cao\s*cấp", r"premium", r"kim\s*cương", r"tỏa\s*sáng",
        r"độc\s*quyền", r"giới\s*hạn", r"duy\s*nhất", r"exclusive", r"limited"
    ]),
    ("brain", [
        r"thông\s*minh", r"\bai\b", r"công\s*nghệ", r"hologram", r"nơ-ron",
        r"tự\s*động", r"smart", r"data", r"thuật\s*toán", r"trí\s*tuệ\s*nhân\s*tạo"
    ]),
    ("users", [
        r"khách\s*hàng", r"học\s*viên", r"gia\s*đình", r"cặp\s*đôi", r"đội\s*ngũ",
        r"chuyên\s*gia", r"1\s*kèm\s*1", r"pt\s*1:1", r"cộng\s*đồng", r"thành\s*viên",
        r"hội\s*viên", r"người\s*dùng"
    ]),
]


# Khi câu văn không khớp bất kỳ nhóm ngữ nghĩa cụ thể nào ở trên, KHÔNG còn đổ đồng về
# 1 icon "check" cố định (nguyên nhân khiến dấu tick lặp lại tràn lan trên poster khi
# nhiều pill cùng chứa nội dung chung chung) -- xoay vòng qua 1 tập icon trung tính,
# vẫn TẤT ĐỊNH theo nội dung (cùng 1 câu luôn ra cùng icon giữa các lần render, để
# không "nhấp nháy" đổi icon khi regenerate cùng nội dung), nhưng các câu khác nhau
# trải đều ra nhiều icon khác nhau thay vì chỉ 1 lựa chọn duy nhất.
_FALLBACK_ICON_POOL = ["check", "star", "tag", "sparkles"]


def infer_semantic_icon(text: str) -> str:
    """Tự động suy luận icon SVG phù hợp dựa trên nội dung câu văn tiếng Việt."""
    if not text:
        return "check"
    cleaned = text.strip().lower()
    for icon_name, patterns in SEMANTIC_ICON_PATTERNS:
        for pat in patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                return icon_name
    idx = sum(ord(c) for c in cleaned) % len(_FALLBACK_ICON_POOL)
    return _FALLBACK_ICON_POOL[idx]


def get_icon_svg(name: str) -> str:
    """Lấy mã HTML SVG theo tên icon."""
    return ICONS_MAP.get(name, CHECK_ICON_SVG)


def parse_store_info_items(store_info: Optional[str]) -> List[Dict[str, str]]:
    """Phân tách chuỗi store_info (ngăn cách bởi |, •, ;, hoặc xuống dòng)
    thành danh sách các mục có gắn icon SVG tương ứng (Hotline -> phone, Địa chỉ -> location, Web -> globe...).
    """
    if not store_info:
        return []
    parts = re.split(r"[\n|•;]+", store_info)
    items = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        icon_name = infer_semantic_icon(p)
        icon_svg = get_icon_svg(icon_name)
        items.append({
            "text": p,
            "icon_name": icon_name,
            "icon_svg": icon_svg,
        })
    return items


# ==============================================================================
# BỘ RENDER MÃ QR VECTOR HIỆN ĐẠI (AESTHETIC QR CODE SVG)
# ==============================================================================

def render_qr_code_svg(
    label: str = "QUÉT MÃ NGAY",
    theme_color: str = "#FFB300",
    size_px: int = 110,
) -> str:
    """Tạo mã QR code vector SVG phong cách đồ họa phẳng cực kỳ sắc nét."""
    clean_label = (label or "QUÉT MÃ NGAY").strip()
    if len(clean_label) > 18:
        lbl_font_size = "8.5px"
        lbl_letter_spacing = "0.4px"
    elif len(clean_label) > 12:
        lbl_font_size = "10px"
        lbl_letter_spacing = "0.6px"
    else:
        lbl_font_size = "11.5px"
        lbl_letter_spacing = "0.8px"

    return f"""
    <div class="qr-card-component" style="display:inline-flex; flex-direction:column; align-items:center; gap:6px; background:rgba(255,255,255,0.08); backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.22); border-radius:14px; padding:10px 12px; box-shadow:0 8px 24px rgba(0,0,0,0.35); max-width:{size_px + 28}px;">
      <svg width="{size_px}" height="{size_px}" viewBox="0 0 100 100" fill="none" style="display:block; border-radius:8px; background:#fff; padding:6px;">
        <!-- QR Finder Patterns (Top-Left) -->
        <rect x="6" y="6" width="26" height="26" rx="4" fill="#000" />
        <rect x="11" y="11" width="16" height="16" rx="2" fill="#fff" />
        <rect x="15" y="15" width="8" height="8" rx="1.5" fill="{theme_color}" />
        
        <!-- QR Finder Patterns (Top-Right) -->
        <rect x="68" y="6" width="26" height="26" rx="4" fill="#000" />
        <rect x="73" y="11" width="16" height="16" rx="2" fill="#fff" />
        <rect x="77" y="15" width="8" height="8" rx="1.5" fill="{theme_color}" />
        
        <!-- QR Finder Patterns (Bottom-Left) -->
        <rect x="6" y="68" width="26" height="26" rx="4" fill="#000" />
        <rect x="11" y="73" width="16" height="16" rx="2" fill="#fff" />
        <rect x="15" y="77" width="8" height="8" rx="1.5" fill="{theme_color}" />
        
        <!-- Data Matrix Modules -->
        <rect x="38" y="8" width="6" height="6" rx="1" fill="#222" />
        <rect x="48" y="8" width="6" height="6" rx="1" fill="#222" />
        <rect x="56" y="14" width="6" height="6" rx="1" fill="#222" />
        <rect x="38" y="20" width="6" height="6" rx="1" fill="#222" />
        <rect x="48" y="26" width="6" height="6" rx="1" fill="#222" />
        
        <rect x="8" y="38" width="6" height="6" rx="1" fill="#222" />
        <rect x="20" y="44" width="6" height="6" rx="1" fill="#222" />
        <rect x="8" y="52" width="6" height="6" rx="1" fill="#222" />
        
        <rect x="36" y="36" width="28" height="28" rx="6" fill="{theme_color}" fill-opacity="0.15" stroke="{theme_color}" stroke-width="1.5" />
        <circle cx="50" cy="50" r="7" fill="{theme_color}" />
        <polyline points="47 50 49.5 52.5 53.5 48" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
        
        <rect x="70" y="38" width="6" height="6" rx="1" fill="#222" />
        <rect x="82" y="44" width="6" height="6" rx="1" fill="#222" />
        <rect x="74" y="54" width="6" height="6" rx="1" fill="#222" />
        <rect x="86" y="54" width="6" height="6" rx="1" fill="#222" />
        
        <rect x="38" y="72" width="6" height="6" rx="1" fill="#222" />
        <rect x="48" y="78" width="6" height="6" rx="1" fill="#222" />
        <rect x="58" y="72" width="6" height="6" rx="1" fill="#222" />
        <rect x="40" y="86" width="6" height="6" rx="1" fill="#222" />
        <rect x="52" y="86" width="6" height="6" rx="1" fill="#222" />
        <rect x="68" y="74" width="6" height="6" rx="1" fill="#222" />
        <rect x="78" y="82" width="6" height="6" rx="1" fill="#222" />
        <rect x="86" y="72" width="6" height="6" rx="1" fill="#222" />
        <rect x="86" y="86" width="6" height="6" rx="1" fill="#222" />
      </svg>
      <span style="font-size:{lbl_font_size}; font-weight:800; letter-spacing:{lbl_letter_spacing}; text-transform:uppercase; color:#fff; text-shadow:0 1px 3px rgba(0,0,0,0.8); text-align:center; word-break:keep-all; max-width:100%;">{clean_label}</span>
    </div>
    """


def render_star_rating_svg(count: Any = 5, fill_color: str = "#FFB300") -> str:
    """Tạo chuỗi HTML dãy 5 sao đánh giá uy tín (hỗ trợ int, float, string, emoji)."""
    import re

    num_stars = 5
    if count is not None:
        if isinstance(count, (int, float)):
            num_stars = int(round(count))
        elif isinstance(count, str):
            star_emojis = count.count("⭐") + count.count("★")
            if star_emojis > 0:
                num_stars = star_emojis
            else:
                m = re.search(r"(\d+(?:\.\d+)?)", count)
                if m:
                    try:
                        num_stars = int(round(float(m.group(1))))
                    except Exception:
                        num_stars = 5
                else:
                    num_stars = 5
        else:
            try:
                num_stars = int(count)
            except Exception:
                num_stars = 5

    num_stars = max(1, min(5, num_stars))

    star = (
        f'<svg width="18" height="18" viewBox="0 0 24 24" fill="{fill_color}" stroke="{fill_color}" stroke-width="1" '
        f'style="display:inline-block; vertical-align:-3px; margin-right:3px; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5));">'
        f'<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>'
        f'</svg>'
    )
    return "".join(star for _ in range(num_stars))
