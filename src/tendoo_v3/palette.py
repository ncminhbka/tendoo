"""tendoo_v3.palette -- bộ token màu tự động đạt chuẩn WCAG AA.

Trích nguyên văn `ColorPalette` từ `src/tendoo/core/base.py` của repo cũ. Phần còn
lại của base.py (`PosterContent`, `BaseLayout`) thuộc kiến trúc v1 và KHÔNG được
mang sang -- tendoo_v3 chưa bao giờ dùng tới chúng.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    badge_border: str = "rgba(255, 255, 255, 0.3)"
    badge_shadow: str = "0 4px 14px rgba(255, 87, 34, 0.4)"
    glass_bg: str = "rgba(255, 255, 255, 0.12)"
    glass_border: str = "rgba(255, 255, 255, 0.25)"
    accent_color: str = "#FFB300"
    on_card_accent: str = "#FFB300"
    on_card_title: str = "#FFFFFF"
    on_canvas_accent: str = "#FFB300"
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
            --badge-border: {self.badge_border};
            --badge-shadow: {self.badge_shadow};
            --glass-bg: {self.glass_bg};
            --glass-border: {self.glass_border};
            --accent: {self.accent_color};
            --accent-color: {self.accent_color};
            --on-card-accent: {self.on_card_accent};
            --on-card-title: {self.on_card_title};
            --on-canvas-accent: {self.on_canvas_accent};
            --footer-text: {self.footer_text};
            --footer-bg: {self.footer_bg};
        }}
        """.strip()


__all__ = ["ColorPalette"]
