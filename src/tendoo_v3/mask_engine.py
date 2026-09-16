"""
src/tendoo_v3/mask_engine.py

Bộ sinh Parametric Continuous Mask Thẩm Mỹ (Aesthetic Mask Engine):
- Tạo các khối mask lớn liên tục (Continuous Blocks) theo đúng vùng chữ THẬT của
  từng template -- lấy số liệu từ `geometry.py`, KHÔNG còn tự đoán % độc lập ở đây
  nữa (đó chính là nguyên nhân đã đo thấy: sandwich_top_heavy's pill-row từng nằm
  97-100% ngoài mask, recruitment_board's cả thân bài không được bảo vệ...).
- Triệt tiêu hoàn toàn hiện tượng nứt viền, răng cưa (seam artifacts) nhờ dải làm mờ
  Gaussian chuẩn quang học.
- Độc lập 100% với GPU, chạy siêu tốc trên CPU.

NGUYÊN TẮC (giữ đúng yêu cầu 2026-09-15): mỗi vùng là 1 khối LIÊN TỤC (rounded-rect
hoặc polygon), KHÔNG vụn thành nhiều mảnh nhỏ bám theo từng dòng chữ -- một template
có thể có NHIỀU vùng (vd sandwich: top + bottom), nhưng mỗi vùng luôn là 1 khối gọn.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tendoo_v3.geometry import Rect, get_zones


def _draw_rounded_zone(draw: "ImageDraw.ImageDraw", rect: Rect, width: int, height: int, fill: int) -> None:
    x1, y1, x2, y2 = rect
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return
    radius = max(2, int(min(24.0, (x2 - x1) * 0.06, (y2 - y1) * 0.06)))
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill)


def _draw_diagonal_slash(
    draw: "ImageDraw.ImageDraw",
    width: float,
    height: float,
    fill: int,
    orientation: Optional[str] = "left",
) -> None:
    # Khớp .diagonal-content { width: 44% } (xem geometry.py::_diagonal_slash, cho dư biên 47%).
    if orientation in ("right", "top_right", "bottom_right"):
        poly_points = [
            (0.42 * width, 0),
            (width, 0),
            (width, height),
            (0.53 * width, height),
        ]
    else:
        poly_points = [
            (0, 0),
            (0.58 * width, 0),
            (0.47 * width, height),
            (0, height),
        ]
    draw.polygon(poly_points, fill=fill)


def generate_template_mask(
    template: str,
    width: int,
    height: int,
    orientation: Optional[str] = None,
    blur_radius_px: int = 18,
) -> np.ndarray:
    """Sinh ma trận float32 [H, W] trong khoảng [0.0, 1.0]:
    - 0.0: Vùng Scene (sản phẩm, mẫu ảnh, bối cảnh)
    - 1.0: Vùng Corridor (dọn sạch nền cho chữ đặt lên)
    - Ranh giới chuyển tiếp mềm mại dạng sigmoid nhờ Gaussian Blur.

    `template`: tên template thật trong TEMPLATE_CATALOG (vd "sandwich_top_heavy") --
    KHÔNG còn nhận preset string rời rạc như bản cũ (mask_preset trong catalog.py chỉ
    còn mang tính mô tả/hiển thị, việc vẽ mask nay đọc thẳng geometry.py theo tên
    template để đảm bảo 1-1 với CSS thật, không còn 2 template cùng dùng chung 1 preset
    rồi lệch hình học như "bottom_band" từng bị before_after_split và
    step_process_roadmap dùng chung dù 2 template có bố cục hoàn toàn khác nhau).
    """
    if template == "diagonal_slash":
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        _draw_diagonal_slash(draw, float(width), float(height), fill=255, orientation=orientation)
    else:
        zones = get_zones(template, width, height, orientation=orientation)
        if not zones:
            return np.zeros((height, width), dtype=np.float32)
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        for rect in zones.values():
            _draw_rounded_zone(draw, rect, width, height, fill=255)

    if blur_radius_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius_px))

    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    return mask_np


def save_mask_preview(mask_np: np.ndarray, output_path: str) -> None:
    """Lưu ảnh mask PNG để trực quan hóa kiểm tra."""
    u8 = (np.clip(mask_np, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(output_path)


__all__ = [
    "generate_template_mask",
    "save_mask_preview",
]
