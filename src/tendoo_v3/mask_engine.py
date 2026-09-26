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
    # Khớp chuẩn tendoo_legacy: góc cắt chéo ~35°-45° thể thao, diện tích ~24%-35%
    aspect = width / height
    if aspect < 0.7:  # 9:16
        x_top = 0.52
        x_bottom = 0.18
    elif aspect >= 1.5:  # 16:9
        x_top = 0.38
        x_bottom = 0.10
    else:  # 1:1, 4:5
        x_top = 0.46
        x_bottom = 0.12

    if orientation in ("right", "top_right", "bottom_right"):
        poly_points = [
            ((1.0 - x_top) * width, 0),
            (width, 0),
            (width, height),
            ((1.0 - x_bottom) * width, height),
        ]
    else:
        poly_points = [
            (0, 0),
            (x_top * width, 0),
            (x_bottom * width, height),
            (0, height),
        ]
    draw.polygon(poly_points, fill=fill)


def generate_template_mask(
    template: str,
    width: int,
    height: int,
    orientation: Optional[str] = None,
    blur_radius_px: int = 36,
    density: float = 1.0,
    has_qr: Optional[bool] = None,
    has_footer: Optional[bool] = None,
    has_message: Optional[bool] = None,
    has_freetext: Optional[bool] = None,
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

    `density` (0.35-1.0, xem geometry.py::compute_density_score): PHẢI truyền đúng
    giá trị đã dùng để render CSS (xem renderer.py::content_density) cho cùng 1
    plan/run -- mask và CSS đọc chung `get_zones()` nên phải cùng 1 density mới khớp
    nhau, nếu không mask sẽ vẽ 1 vùng khác kích thước với card CSS thật đã render.

    `has_qr`/`has_footer`/`has_message`/`has_freetext`: tương tự -- PHẢI khớp cờ đã dùng
    khi render CSS; truyền `**renderer.compute_geometry_flags(plan)` (cùng hàm
    build_template_html dùng), nếu không mask sẽ lệch kích thước với box CSS thật.
    """
    if template == "diagonal_slash":
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        _draw_diagonal_slash(draw, float(width), float(height), fill=255, orientation=orientation)
    else:
        zones = get_zones(
            template,
            width,
            height,
            orientation=orientation,
            density=density,
            has_qr=has_qr,
            has_footer=has_footer,
            has_message=has_message,
            has_freetext=has_freetext,
        )
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
    if mask_np.dtype == np.uint8:
        u8 = mask_np
    elif float(mask_np.max()) > 1.0:
        u8 = np.clip(mask_np, 0.0, 255.0).astype(np.uint8)
    else:
        u8 = (np.clip(mask_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(output_path)


__all__ = [
    "generate_template_mask",
    "save_mask_preview",
]
