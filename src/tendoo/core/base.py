"""
src/tendoo/core/base.py

Giao Diện Cốt Lõi & Cấu Trúc Dữ Liệu Trung Tâm Cho Tendoo Engine:
================================================================================
1. VAI TRÒ KIẾN TRÚC:
   - Đây là tệp "Gốc của Gốc" (Root of Core) trong toàn bộ hệ thống Tendoo AI.
   - Định nghĩa các đối tượng dữ liệu truyền tay (Data Transfer Objects - DTO)
     xuyên suốt mọi tầng: từ Web UI, Server API, Layout Engine, Jinja2 Renderer
     cho đến các thuật toán AI Diffusion.
   - Đảm bảo tính độc lập tuyệt đối giữa các module: Module xử lý màu (`color_engine.py`)
     chỉ cần biết `ColorPalette`, Module tạo layout (`layout.py`) chỉ cần biết `PosterContent`
     và kế thừa `BaseLayout`.

2. CÁC THÀNH PHẦN CHÍNH TRONG FILE:
   - Hệ thống Biểu tượng Vector Độc Lập (Zero-Dependency Vector SVGs): 10 biểu tượng
     chuẩn thương mại (điện thoại, lịch, vị trí, ngôi sao, quà tặng...) vẽ bằng SVG thuần,
     không phụ thuộc vào font icon bên ngoài (như FontAwesome), hiển thị sắc nét 100% trên
     mọi hệ điều hành và headless browser.
   - `PosterContent`: Dataclass trung tâm lưu trữ toàn bộ dữ liệu người dùng nhập cho 6 ngành hàng,
     kèm danh sách khối chữ tự do `free_text_blocks` và cấu hình QR code.
   - `ColorPalette`: Bộ token màu sắc tự động tính toán từ ảnh nền, xuất trực tiếp ra biến CSS root.
   - `BaseLayout`: Lớp trừu tượng (ABC) quy định hợp đồng cho mọi Layout Topology của Tendoo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==============================================================================
# PHẦN 1: HỆ THỐNG BIỂU TƯỢNG VECTOR SVG KHÔNG PHỤ THUỘC FONT (CROSS-PLATFORM ICONS)
# ==============================================================================

# TẠI SAO CẦN LÀM:
# - Các biểu tượng font thông thường (như Unicode emoji hay Icon Webfonts) thường bị lỗi "ô vuông"
#   (Tofu bug) trên các môi trường Linux / Docker / JupyterLab thiếu font hệ thống.
# - Vẽ trực tiếp bằng SVG Inline với `stroke="currentColor"` cho phép icon tự động đổi màu theo
#   màu chữ xung quanh, hiển thị cực kỳ sắc nét ở mọi độ phân giải (1K, 2K, 4K) và luôn đồng nhất.

CALENDAR_ICON_SVG = (
    '<svg class="icon-calendar" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.9;">'
    '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>'
    '<line x1="16" y1="2" x2="16" y2="6"></line>'
    '<line x1="8" y1="2" x2="8" y2="6"></line>'
    '<line x1="3" y1="10" x2="21" y2="10"></line>'
    '</svg>'
)

PHONE_ICON_SVG = (
    '<svg class="icon-phone" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.9;">'
    '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>'
    '</svg>'
)

LOCATION_ICON_SVG = (
    '<svg class="icon-location" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.9;">'
    '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>'
    '<circle cx="12" cy="10" r="3"></circle>'
    '</svg>'
)

GLOBE_ICON_SVG = (
    '<svg class="icon-globe" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.9;">'
    '<circle cx="12" cy="12" r="10"></circle>'
    '<line x1="2" y1="12" x2="22" y2="12"></line>'
    '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>'
    '</svg>'
)

GIFT_ICON_SVG = (
    '<svg class="icon-gift" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.9;">'
    '<polyline points="20 12 20 22 4 22 4 12"></polyline>'
    '<rect x="2" y="7" width="20" height="5"></rect>'
    '<line x1="12" y1="22" x2="12" y2="7"></line>'
    '<path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z"></path>'
    '<path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z"></path>'
    '</svg>'
)

TAG_ICON_SVG = (
    '<svg class="icon-tag" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.9;">'
    '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path>'
    '<line x1="7" y1="7" x2="7.01" y2="7"></line>'
    '</svg>'
)

STAR_ICON_SVG = (
    '<svg class="icon-star" width="16" height="16" viewBox="0 0 24 24" fill="#FFB300" stroke="#FFB300" stroke-width="1" '
    'style="display:inline-block; vertical-align:-2px; margin-right:2px;">'
    '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>'
    '</svg>'
)

CLOCK_ICON_SVG = (
    '<svg class="icon-clock" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.9;">'
    '<circle cx="12" cy="12" r="10"></circle>'
    '<polyline points="12 6 12 12 16 14"></polyline>'
    '</svg>'
)

CHECK_ICON_SVG = (
    '<svg class="icon-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; color:#10B981;">'
    '<polyline points="20 6 9 17 4 12"></polyline>'
    '</svg>'
)

ARROW_RIGHT_ICON_SVG = (
    '<svg class="icon-arrow-right" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-left:4px;">'
    '<line x1="5" y1="12" x2="19" y2="12"></line>'
    '<polyline points="12 5 19 12 12 19"></polyline>'
    '</svg>'
)

# --- ICON CÔNG NGHỆ, MARKETING, SÁNG TẠO & SAAS ---

ROCKET_ICON_SVG = (
    '<svg class="icon-rocket" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path>'
    '<path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path>'
    '<path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"></path>'
    '<path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"></path>'
    '</svg>'
)

CHART_UP_ICON_SVG = (
    '<svg class="icon-chart-up" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline>'
    '<polyline points="16 7 22 7 22 13"></polyline>'
    '</svg>'
)

PALETTE_ICON_SVG = (
    '<svg class="icon-palette" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<circle cx="13.5" cy="6.5" r=".5" fill="currentColor"></circle>'
    '<circle cx="17.5" cy="10.5" r=".5" fill="currentColor"></circle>'
    '<circle cx="8.5" cy="7.5" r=".5" fill="currentColor"></circle>'
    '<circle cx="6.5" cy="12.5" r=".5" fill="currentColor"></circle>'
    '<path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.563-2.512 5.563-5.563C22 6.5 17.5 2 12 2z"></path>'
    '</svg>'
)

BRAIN_ICON_SVG = (
    '<svg class="icon-brain" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"></path>'
    '<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"></path>'
    '</svg>'
)

ZAP_ICON_SVG = (
    '<svg class="icon-zap" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>'
    '</svg>'
)

SPARKLES_ICON_SVG = (
    '<svg class="icon-sparkles" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>'
    '<path d="M5 3v4"></path><path d="M19 17v4"></path><path d="M3 5h4"></path><path d="M17 19h4"></path>'
    '</svg>'
)

FIRE_ICON_SVG = (
    '<svg class="icon-fire" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"></path>'
    '</svg>'
)

SHIELD_ICON_SVG = (
    '<svg class="icon-shield" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>'
    '</svg>'
)

TROPHY_ICON_SVG = (
    '<svg class="icon-trophy" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path>'
    '<path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path>'
    '<path d="M4 22h16"></path>'
    '<path d="M10 14.66V17c0 .55-.45.98-.96 1.2-1.11.48-1.71 1.04-1.71 1.8h9.34c0-.76-.6-1.32-1.71-1.8-.51-.22-.96-.65-.96-1.2v-2.34"></path>'
    '<path d="M18 2H6v7a6 6 0 0 0 12 0V2z"></path>'
    '</svg>'
)

HEART_ICON_SVG = (
    '<svg class="icon-heart" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"></path>'
    '</svg>'
)

USERS_ICON_SVG = (
    '<svg class="icon-users" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:6px; opacity:0.95;">'
    '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path>'
    '<circle cx="9" cy="7" r="4"></circle>'
    '<path d="M22 21v-2a4 4 0 0 0-3-3.87"></path>'
    '<path d="M16 3.13a4 4 0 0 1 0 7.75"></path>'
    '</svg>'
)


# ==============================================================================
# PHẦN 2: CẤU TRÚC DỮ LIỆU NỘI DUNG POSTER (POSTER CONTENT)
# ==============================================================================

@dataclass
class PosterContent:
    """Cấu trúc dữ liệu trung tâm lưu trữ toàn bộ nội dung mà người dùng nhập từ Form/API.
    
    CÁC NHÓM TRƯỜNG:
    1. Tiêu đề & Thông điệp cơ bản (Headline, Slogan, Pre-header).
    2. Thông tin cửa hàng & Khuyến mại chung (Brand, Hotline, Address, Website, Dates, Offers).
    3. Mã QR (qr_data_uri, qr_zone, qr_label).
    4. Các trường chuyên biệt cho 6 danh mục thương mại:
       - product_intro: product_name, price, product_desc, highlights.
       - opening: opening_date, opening_promo, booking_contact.
       - feedback: feedback_target, feedback_quote, feedback_rating, special_offer.
       - recruitment: job_position, job_desc, apply_deadline, apply_method.
       - guide: steps (List[str]).
    5. Cấu hình thẩm mỹ & kế hoạch bố cục:
       - font_family, text_effect, custom_css.
       - free_text_blocks: Chứa danh sách các block tự do (List[AdaptiveBlock] hoặc List[Dict]).
    
    TẠI SAO CẦN LÀM:
    - Gom toàn bộ trạng thái của một yêu cầu thiết kế vào một nguồn chân lý (Single Source of Truth).
    - Cung cấp phương thức `to_dict()` và `from_dict()` an toàn tuyệt đối cho JSON serialization,
      tự động chuyển đổi các instance `AdaptiveBlock` lồng bên trong thành dict thuần để không
      bao giờ bị lỗi `TypeError: Object of type AdaptiveBlock is not JSON serializable`.
    """
    # 1. Tiêu đề & Khuyến mại cơ bản (Mặc định rỗng an toàn, tránh lỗi thiếu positional argument)
    headline: str = ""
    pre_header: str = ""
    slogan: str = ""
    offer_main: str = ""
    offer_sub: str = ""
    dates: str = ""
    brand: str = ""
    hotline: str = ""
    applicable: str = ""
    address: str = ""
    website_link: str = ""
    qr_data_uri: str = ""
    category: str = "promo"
    text_effect: str = "auto"
    font_family: str = "auto"
    custom_css: str = ""

    # 2. Danh mục: Giới thiệu sản phẩm (Product Intro)
    price: str = ""
    product_name: str = ""
    product_desc: str = ""
    highlights: str = ""

    # 3. Danh mục: Khai trương (Opening Banner)
    opening_date: str = ""
    opening_promo: str = ""
    booking_contact: str = ""

    # 4. Danh mục: Đánh giá khách hàng (Customer Feedback)
    feedback_target: str = ""
    feedback_quote: str = ""
    feedback_rating: str = ""
    special_offer: str = ""

    # 5. Danh mục: Tuyển dụng (Recruitment)
    job_position: str = ""
    job_desc: str = ""
    apply_deadline: str = ""
    apply_method: str = ""

    # 6. Danh mục: Quy trình hướng dẫn (Guide)
    steps: List[str] = field(default_factory=list)

    # Khối nội dung HTML tùy biến (nếu có)
    category_body_html: str = ""

    # Kế hoạch bố cục không gian đa khối (Omni Adaptive Blocks)
    # Hỗ trợ chứa cả Dict hoặc AdaptiveBlock instance
    free_text_blocks: List[Any] = field(default_factory=list)

    # Vị trí ô hiển thị QR Code (mặc định 'bottom_right')
    qr_zone: Optional[str] = "bottom_right"
    qr_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary thuần túy (100% JSON Serializability Safe).
        
        TẠI SAO CẦN LÀM:
        - Nếu trong `free_text_blocks` chứa các đối tượng `AdaptiveBlock` (dataclass), việc gọi
          `list(self.free_text_blocks)` thông thường sẽ giữ nguyên đối tượng python, khiến hàm
          `json.dumps()` bị crash với lỗi `TypeError: Object is not JSON serializable`.
        - Logic dưới đây chủ động phát hiện phương thức `to_dict()` của block con để serialize triệt để.
        """
        blocks_serialized: List[Dict[str, Any]] = []
        for b in self.free_text_blocks:
            if hasattr(b, "to_dict") and callable(b.to_dict):
                blocks_serialized.append(b.to_dict())
            elif isinstance(b, dict):
                blocks_serialized.append(dict(b))
            else:
                blocks_serialized.append(b)

        return {
            "headline": self.headline,
            "pre_header": self.pre_header,
            "slogan": self.slogan,
            "offer_main": self.offer_main,
            "offer_sub": self.offer_sub,
            "dates": self.dates,
            "brand": self.brand,
            "hotline": self.hotline,
            "applicable": self.applicable,
            "address": self.address,
            "website_link": self.website_link,
            "qr_data_uri": self.qr_data_uri,
            "category": self.category,
            "text_effect": self.text_effect,
            "font_family": self.font_family,
            "custom_css": self.custom_css,
            "price": self.price,
            "product_name": self.product_name,
            "product_desc": self.product_desc,
            "highlights": self.highlights,
            "opening_date": self.opening_date,
            "opening_promo": self.opening_promo,
            "booking_contact": self.booking_contact,
            "feedback_target": self.feedback_target,
            "feedback_quote": self.feedback_quote,
            "feedback_rating": self.feedback_rating,
            "special_offer": self.special_offer,
            "job_position": self.job_position,
            "job_desc": self.job_desc,
            "apply_deadline": self.apply_deadline,
            "apply_method": self.apply_method,
            "steps": list(self.steps),
            "category_body_html": self.category_body_html,
            "free_text_blocks": blocks_serialized,
            "qr_zone": self.qr_zone,
            "qr_label": self.qr_label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PosterContent:
        """Khôi phục đối tượng PosterContent an toàn từ dictionary payload hoặc cache."""
        steps_val = data.get("steps", [])
        if isinstance(steps_val, list):
            steps = [str(s) for s in steps_val]
        else:
            steps = [str(steps_val)] if steps_val else []

        return cls(
            headline=str(data.get("headline", "")),
            pre_header=str(data.get("pre_header", "")),
            slogan=str(data.get("slogan", "")),
            offer_main=str(data.get("offer_main", "")),
            offer_sub=str(data.get("offer_sub", "")),
            dates=str(data.get("dates", "")),
            brand=str(data.get("brand", "")),
            hotline=str(data.get("hotline", "")),
            applicable=str(data.get("applicable", "")),
            address=str(data.get("address", "")),
            website_link=str(data.get("website_link", "")),
            qr_data_uri=str(data.get("qr_data_uri", "")),
            category=str(data.get("category", "promo")),
            text_effect=str(data.get("text_effect", "auto")),
            font_family=str(data.get("font_family", "auto")),
            custom_css=str(data.get("custom_css", "")),
            price=str(data.get("price", "")),
            product_name=str(data.get("product_name", "")),
            product_desc=str(data.get("product_desc", "")),
            highlights=str(data.get("highlights", "")),
            opening_date=str(data.get("opening_date", "")),
            opening_promo=str(data.get("opening_promo", "")),
            booking_contact=str(data.get("booking_contact", "")),
            feedback_target=str(data.get("feedback_target", "")),
            feedback_quote=str(data.get("feedback_quote", "")),
            feedback_rating=str(data.get("feedback_rating", "")),
            special_offer=str(data.get("special_offer", "")),
            job_position=str(data.get("job_position", "")),
            job_desc=str(data.get("job_desc", "")),
            apply_deadline=str(data.get("apply_deadline", "")),
            apply_method=str(data.get("apply_method", "")),
            steps=steps,
            category_body_html=str(data.get("category_body_html", "")),
            free_text_blocks=list(data.get("free_text_blocks", [])),
            qr_zone=data.get("qr_zone", "bottom_right"),
            qr_label=str(data.get("qr_label", "")),
        )


# ==============================================================================
# PHẦN 3: BỘ TOKEN MÀU HÒA SẮC TỰ ĐỘNG (COLOR PALETTE TOKENS)
# ==============================================================================

@dataclass
class ColorPalette:
    """Bộ token màu sắc tự động trích xuất từ ảnh nền và đạt chuẩn tương phản WCAG AA.
    
    CÁC THUỘC TÍNH MÀU SẮC:
    - is_dark: Cờ boolean cho biết nền ảnh là tối hay sáng (ảnh hưởng đến việc dùng chữ trắng hay đen).
    - luminance: Độ sáng thực tế của vùng an toàn trên ảnh nền (tính theo công thức quang học Y).
    - hue, comp_hue: Góc màu chủ đạo và góc màu bổ túc trên vòng tròn màu bánh xe (Color Wheel).
    - headline_color: Màu chữ tiêu đề chính (đảm bảo độ tương phản tối thiểu 4.5:1 với nền).
    - headline_is_gradient: Cờ cho biết tiêu đề có dùng dải màu gradient hay màu đơn sắc.
    - sub_color: Màu chữ phụ cho mô tả và thông tin chi tiết.
    - text_shadow: Đổ bóng CSS để tách bạch nét chữ khỏi các chi tiết nhiễu hạt của nền ảnh.
    - badge_bg, badge_text, badge_shadow: Màu nền, màu chữ và bóng đổ của các huy hiệu/viên thuốc.
    - glass_bg, glass_border: Màu nền kính mờ và viền phản chiếu ánh sáng (Glassmorphism).
    - accent_color: Màu điểm nhấn (vàng kim, xanh ngọc, cam cháy...) tạo sự tương phản thu hút.
    - footer_text, footer_bg: Màu sắc thanh thông tin cửa hàng ở chân trang.
    
    TẠI SAO CẦN LÀM:
    - Tự động hóa hoàn toàn việc phối màu: Người dùng không cần kiến thức mỹ thuật, hệ thống tự động
      "hút màu" từ bức ảnh do AI sinh ra để tạo nên một thiết kế có sự hài hòa tuyệt đối về thị giác.
    - Phương thức `to_css_vars()` đồng thời xuất cả tên biến viết tắt (`--accent`, `--headline`, `--sub`)
      lẫn tên biến đầy đủ (`--accent-color`, `--headline-color`, `--sub-color`) để bảo đảm 100%
      tương thích với mọi file template HTML/CSS cũ và mới.
    """
    is_dark: bool
    luminance: float
    hue: int
    comp_hue: int
    headline_color: str
    headline_is_gradient: bool = False
    sub_color: str = "#E0E0E0"
    text_shadow: str = "0 2px 8px rgba(0, 0, 0, 0.6)"
    badge_bg: str = "#FF5722"
    badge_text: str = "#FFFFFF"
    badge_shadow: str = "0 4px 14px rgba(255, 87, 34, 0.4)"
    glass_bg: str = "rgba(255, 255, 255, 0.12)"
    glass_border: str = "rgba(255, 255, 255, 0.25)"
    accent_color: str = "#FFB300"
    footer_text: str = "#B0BEC5"
    footer_bg: str = "rgba(0, 0, 0, 0.5)"

    def to_css_vars(self) -> str:
        """Xuất bộ token thành các biến CSS Root Variable chuẩn (:root)."""
        return f"""
        :root {{
            --is-dark: {'1' if self.is_dark else '0'};
            --headline: {self.headline_color};
            --headline-color: {self.headline_color};
            --sub: {self.sub_color};
            --sub-color: {self.sub_color};
            --text-color: {self.sub_color};
            --text-shadow: {self.text_shadow};
            --badge-bg: {self.badge_bg};
            --badge-text: {self.badge_text};
            --badge-shadow: {self.badge_shadow};
            --glass-bg: {self.glass_bg};
            --glass-border: {self.glass_border};
            --accent: {self.accent_color};
            --accent-color: {self.accent_color};
            --footer-text: {self.footer_text};
            --footer-bg: {self.footer_bg};
        }}
        """.strip()


# ==============================================================================
# PHẦN 4: GIAO DIỆN LỚP TRỪU TƯỢNG BỐ CỤC (BASE LAYOUT INTERFACE)
# ==============================================================================

class BaseLayout(ABC):
    """Lớp trừu tượng (Interface Contract) mà mọi Layout Topology của Tendoo bắt buộc phải tuân thủ.
    
    TẠI SAO CẦN LÀM:
    - Áp dụng nguyên lý Đảo ngược phụ thuộc (Dependency Inversion Principle): Tầng điều phối server
      (`demo_server.py`) chỉ phụ thuộc vào interface trừu tượng `BaseLayout`, không phụ thuộc vào cách
      triển khai chi tiết của từng layout (như `OmniBlockLayout`).
    - Cho phép hoán đổi hoặc nâng cấp các engine layout mới mà không làm gãy pipeline sinh ảnh.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Định danh kỹ thuật dạng chuỗi (machine identifier), ví dụ: 'omni'."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Tên hiển thị thân thiện trên giao diện người dùng, ví dụ: 'Omni-Block Matrix 3x3'."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Mô tả ngắn gọn về đặc thù và ngữ cảnh nên sử dụng của layout này."""
        pass

    @abstractmethod
    def generate_mask(
        self,
        width: int,
        height: int,
        **kwargs,
    ) -> np.ndarray:
        """Sinh ma trận mặt nạ hành lang 2D float32 numpy array [H, W] với giá trị trong dải [0.0, 1.0].
        - Giá trị 1.0: Vùng hành lang chữ (áp dụng 100% Corridor Prompt để dọn dẹp sạch nền).
        - Giá trị 0.0: Vùng ảnh tự nhiên và Product Sanctuary (áp dụng 100% Scene Prompt cho sản phẩm).
        """
        pass

    @abstractmethod
    def get_corridor_prompt(self, style_hint: str = "daylight") -> str:
        """Trả về câu prompt hành lang thủ tục tối ưu cho vùng chữ an toàn của layout này."""
        pass

    @abstractmethod
    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """Trả về tọa độ chuẩn hóa (y1, x1, y2, x2) của vùng chữ chính để trích xuất màu sắc hài hòa."""
        pass

    @abstractmethod
    def render_html(
        self,
        content: PosterContent,
        palette: ColorPalette,
        bg_data_uri: str = "",
        width: int = 1024,
        height: int = 1024,
        headline_effect: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Render mã nguồn HTML5/CSS3 hoàn chỉnh sẵn sàng cho Chromium chụp ảnh màn hình."""
        pass
