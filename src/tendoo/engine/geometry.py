"""
src/tendoo/engine/geometry.py

Bộ giải toán Hình Học & Không Gian cho Tendoo Omni-Block Engine:
- Định vị Vùng Thánh Địa Sản Phẩm (Product Sanctuary): Tuyệt đối không cho chữ đè lên sản phẩm/chủ thể.
- Tính toán Bounding Box chính xác cho từng Block và Zone.
- Đo đạc font metrics bằng PIL:
  * Tự động đo `text.upper()` khi role có `is_uppercase=True` (chữa triệt để lỗi cắt mép "TRANSFORMATION").
  * Tự động cân bằng dòng `balance_vietnamese_headline` cho tiêu đề Hero (chữa triệt để lỗi 1 từ 1 dòng).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tendoo.core.typography import balance_vietnamese_headline, fit_font_size_px
from tendoo.engine.blocks import AdaptiveBlock, ROLE_SCALE

# Tọa độ tương đối [ymin, xmin, ymax, xmax] của Product Sanctuary theo tỷ lệ khung hình
PRODUCT_SANCTUARY_BY_RATIO: Dict[str, Tuple[float, float, float, float]] = {
    "1:1": (0.24, 0.28, 0.76, 0.72),
    "9:16": (0.24, 0.16, 0.72, 0.84),
    "16:9": (0.16, 0.28, 0.84, 0.72),
    "4:5": (0.24, 0.24, 0.76, 0.76),
}


def get_product_sanctuary_rect(width: int, height: int) -> Tuple[int, int, int, int]:
    """Trả về pixel rect (x1, y1, x2, y2) của Product Sanctuary."""
    ratio_val = width / max(1, height)
    if abs(ratio_val - 1.0) < 0.05:
        key = "1:1"
    elif abs(ratio_val - 9 / 16) < 0.05:
        key = "9:16"
    elif abs(ratio_val - 16 / 9) < 0.05:
        key = "16:9"
    elif abs(ratio_val - 4 / 5) < 0.05:
        key = "4:5"
    else:
        key = "1:1"

    ymin, xmin, ymax, xmax = PRODUCT_SANCTUARY_BY_RATIO[key]
    return int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)


def get_zone_bounding_box(zone: str, width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Trả về pixel bounding box an toàn (x1, y1, x2, y2) cho từng Zone.
    Đảm bảo các zone không lấn vào Product Sanctuary ở trung tâm.
    """
    margin_x = int(width * 0.04)
    margin_y = int(height * 0.04)

    # Các biên an toàn của Product Sanctuary ở trung tâm:
    # x in [0.26, 0.74], y in [0.22, 0.78]
    sanctuary_top = int(height * 0.22)
    sanctuary_bottom = int(height * 0.78)
    sanctuary_left = int(width * 0.26)
    sanctuary_right = int(width * 0.74)

    if zone == "top_bar":
        return margin_x, margin_y, width - margin_x, int(height * 0.12)
    elif zone == "bottom_bar":
        return margin_x, int(height * 0.90), width - margin_x, height - margin_y
    elif zone == "top_left":
        return margin_x, margin_y, sanctuary_left, sanctuary_top
    elif zone == "top_center":
        return int(width * 0.08), margin_y, int(width * 0.92), sanctuary_top
    elif zone == "top_right":
        return sanctuary_right, margin_y, width - margin_x, sanctuary_top
    elif zone == "middle_left":
        return margin_x, sanctuary_top, sanctuary_left, sanctuary_bottom
    elif zone == "middle_right":
        return sanctuary_right, sanctuary_top, width - margin_x, sanctuary_bottom
    elif zone == "bottom_left":
        return margin_x, sanctuary_bottom, sanctuary_left + int(width * 0.12), height - margin_y
    elif zone == "bottom_center":
        return int(width * 0.08), sanctuary_bottom, int(width * 0.92), height - margin_y
    elif zone == "bottom_right":
        return sanctuary_right - int(width * 0.12), sanctuary_bottom, width - margin_x, height - margin_y
    else:
        # Fallback an toàn
        return margin_x, margin_y, sanctuary_left, sanctuary_top


def compute_block_metrics(
    block: AdaptiveBlock,
    font_path: str,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """
    Tính toán metrics hiển thị cho một block:
    - font_size tối ưu
    - hiển thị đa dòng cân đối
    - bề rộng, bề cao
    - text_transform (uppercase)
    """
    role = block.role if block.role in ROLE_SCALE else "body"
    size_ratio, weight, is_uppercase = ROLE_SCALE[role]

    zone = block.zone or "top_left"
    x1, y1, x2, y2 = get_zone_bounding_box(zone, width, height)
    max_w_px = float(x2 - x1)
    max_h_px = float(y2 - y1)

    text = block.text.strip()
    display_text = text

    # Với vai trò hero: cân bằng dòng tiếng Việt để tránh 1 từ 1 dòng
    if role == "hero" and "\n" not in text and len(text.split()) >= 3:
        lines, _ = balance_vietnamese_headline(text)
        if len(lines) > 1:
            display_text = "\n".join(lines)

    base_font_size = max(14, int(width * size_ratio))

    # ĐO ĐẠC VỚI UPPERCASE AWARENESS:
    # Nếu role là hero (in hoa), đo trên chuỗi UPPERCASE để không bao giờ bị cắt mép
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


ZONE_NAMES: List[str] = [
    "top_left", "top_center", "top_right",
    "middle_left", "center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
]

ZONE_GRID_AREA: Dict[str, str] = {
    "top_left": "1 / 1 / 2 / 2",
    "top_center": "1 / 2 / 2 / 3",
    "top_right": "1 / 3 / 2 / 4",
    "middle_left": "2 / 1 / 3 / 2",
    "center": "2 / 2 / 3 / 3",
    "middle_right": "2 / 3 / 3 / 4",
    "bottom_left": "3 / 1 / 4 / 2",
    "bottom_center": "3 / 2 / 4 / 3",
    "bottom_right": "3 / 3 / 4 / 4",
}

ZONE_DEFAULT_ALIGN: Dict[str, str] = {
    "top_left": "left", "middle_left": "left", "bottom_left": "left",
    "top_center": "center", "center": "center", "bottom_center": "center",
    "top_right": "right", "middle_right": "right", "bottom_right": "right",
}

ZONE_SELF_ALIGN: Dict[str, Tuple[str, str]] = {
    "top_left": ("start", "start"), "top_center": ("center", "start"), "top_right": ("end", "start"),
    "middle_left": ("start", "center"), "center": ("center", "center"), "middle_right": ("end", "center"),
    "bottom_left": ("start", "end"), "bottom_center": ("center", "end"), "bottom_right": ("end", "end"),
}

