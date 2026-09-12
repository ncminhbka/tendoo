"""
src/tendoo/engine/mask.py

Bộ sinh Parametric Corridor Mask Hợp Nhất (Single Unified Mask Engine):
========================================================================
- Hợp nhất bounding box của tất cả các AdaptiveBlock đang hoạt động thành 1 Binary Mask mềm duy nhất.
- Hỗ trợ kiến trúc 2-branch velocity blending: 1 Scene Branch + 1 Corridor Branch chung.
- Giữ nguyên chi phí tính toán O(1) bất biến (~6s/ảnh trên 2x A30), bất kể số lượng khối text.

TẠI SAO CẦN MODULE NÀY TRONG HỆ THỐNG TENDOO AI:
1. TẠI SAO CẦN PARAMETRIC CORRIDOR MASK HỢP NHẤT (SINGLE UNIFIED MASK):
   - Trong kiến trúc 2-branch Velocity Blending, FLUX.2 DiT chạy 2 luồng tích phân ODE:
     + Luồng 1 (Scene Branch): Sinh bối cảnh tự nhiên, sản phẩm 3D và ánh sáng môi trường.
     + Luồng 2 (Corridor Branch): Dọn sạch nền, làm dịu các họa tiết phức tạp nơi văn bản tọa lạc.
   - Nếu mỗi khối chữ có một mask riêng biệt, chi phí tính toán sẽ tăng tuyến tính O(N) theo số lượng
     block (N lần inference ODE -> thời gian sinh ảnh bùng nổ lên hàng phút).
   - Hợp nhất tất cả các bounding box vào 1 ma trận Mask duy nhất giúp hệ thống bảo toàn chi phí
     tính toán O(1) bất biến (~6s/ảnh trên 2x A30), bất kể poster chứa 1 block hay 8 blocks.

2. TẠI SAO CẦN 2 HÀM SINH MASK (OFFLINE PIL vs ONLINE CHROMIUM):
   - `generate_omni_corridor_mask` (Offline PIL): Dùng trong môi trường CLI tinh gọn, unit test,
     hoặc các server không cài sẵn Headless Chromium. Hàm này sử dụng PIL FreeType metrics để
     ước lượng vùng an toàn. Vì có sai số nhỏ so với CSS render thực tế, hàm sử dụng padding rộng hơn
     (padding_px=28, blur_radius_px=16) để đảm bảo an toàn tuyệt đối.
   - `generate_omni_corridor_mask_from_rects` (Online Chromium): Được gọi trong Production Pipeline
     thông qua `PosterRenderer.measure_zone_rects`. Bounding box được đo trực tiếp bởi chính engine
     Chromium sẽ vẽ pixel cuối cùng. Vì độ chính xác đạt 100% từng pixel, padding có thể thu gọn lại
     (padding_px=10, blur_radius_px=12), tối đa hóa diện tích bối cảnh nghệ thuật cho AI sinh ảnh.

3. TẠI SAO CẦN GAUSSIAN BLUR LÀM MỀM BIÊN (SOFT MASK TRANSITION):
   - Nếu sử dụng Hard Mask nhị phân (giá trị nhảy gắt 0 -> 1), tại ranh giới của chữ sẽ xuất hiện
     vết cắt sắc nhọn giả tạo (edge artifacts) do sự đứt gãy gradient vận tốc ODE Flow Matching.
   - Gaussian Blur tạo ra dải chuyển tiếp sigmoid mượt mà (0.0 -> 0.3 -> 0.7 -> 1.0), cho phép
     ánh sáng môi trường và màu sắc từ bối cảnh thẩm thấu tự nhiên vào vùng nền của chữ.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tendoo.engine.blocks import AdaptiveBlock
from tendoo.engine.geometry import compute_block_metrics, get_zone_bounding_box


def generate_omni_corridor_mask(
    width: int,
    height: int,
    blocks: List[AdaptiveBlock],
    font_path: str,
    padding_px: int = 28,
    blur_radius_px: int = 16,
    int_max: float = 1.0,
) -> np.ndarray:
    """
    Sinh 1 ma trận mask float32 [height, width] với giá trị [0.0, 1.0] (Offline PIL Estimator):
    - 0.0: Vùng Scene (Sản phẩm, chủ thể, bối cảnh)
    - int_max: Vùng Corridor (Làm sạch, làm dịu nền cho chữ đặt lên)
    - Biên chuyển tiếp mượt mà nhờ Gaussian Blur.
    """
    # Nếu không có block nào, trả về mask toàn 0.0 (thuần scene)
    if not blocks:
        return np.zeros((height, width), dtype=np.float32)

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    fill_val = int(min(255, max(0, int(int_max * 255))))

    # Vẽ các vùng chữ vào mask
    for b in blocks:
        text = b.text.strip()
        if not text:
            continue

        metrics = compute_block_metrics(b, font_path, width, height)
        zone = b.zone or "top_left"
        zx1, zy1, zx2, zy2 = metrics["zone_rect"]

        w = metrics["measured_w"]
        h = metrics["measured_h"]

        # Định vị hình chữ nhật bounding box trong zone
        if "left" in zone:
            bx1 = zx1
            bx2 = min(zx2, bx1 + int(w) + padding_px * 2)
        elif "right" in zone:
            bx2 = zx2
            bx1 = max(zx1, bx2 - int(w) - padding_px * 2)
        else:
            # center / bar
            cx = (zx1 + zx2) // 2
            half_w = int(w / 2) + padding_px
            bx1 = max(zx1, cx - half_w)
            bx2 = min(zx2, cx + half_w)

        if "top" in zone or zone == "top_bar":
            by1 = zy1
            by2 = min(zy2, by1 + int(h) + padding_px * 2)
        elif "bottom" in zone or zone == "bottom_bar":
            by2 = zy2
            by1 = max(zy1, by2 - int(h) - padding_px * 2)
        else:
            # middle
            cy = (zy1 + zy2) // 2
            half_h = int(h / 2) + padding_px
            by1 = max(zy1, cy - half_h)
            by2 = min(zy2, cy + half_h)

        # Kiểm tra tính hợp lệ của tọa độ trước khi vẽ để chống lỗi Pillow ValueError
        if bx2 <= bx1 or by2 <= by1:
            continue

        # Giới hạn bán kính bo góc an toàn không vượt quá nửa kích thước hình chữ nhật
        box_w = bx2 - bx1
        box_h = by2 - by1
        safe_radius = max(2, min(16, box_w // 2, box_h // 2))

        # Vẽ hình chữ nhật bo góc nhẹ
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=safe_radius, fill=fill_val)

    # Làm mượt biên với Gaussian Blur để velocity blending hòa trộn tự nhiên
    if blur_radius_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius_px))

    mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
    return mask_arr.clip(0.0, 1.0)


def generate_omni_corridor_mask_from_rects(
    width: int,
    height: int,
    rects: List[Tuple[float, float, float, float]],
    padding_px: int = 10,
    blur_radius_px: int = 12,
    int_max: float = 1.0,
) -> np.ndarray:
    """
    Sinh mask từ các bounding box THẬT đo được bằng chính engine sẽ render pixel cuối
    (Chromium, qua `PosterRenderer.measure_zone_rects`) -- KHÔNG suy luận offline bằng PIL.

    TẠI SAO ĐÂY LÀ PHƯƠNG PHÁP CHUẨN PRODUCTION:
    - Đo trực tiếp DOM layout thực tế bao gồm toàn bộ padding, gap, icon SVG, và letter-spacing.
    - Sai số hình học = 0.0px.
    - Cho phép giảm padding_px xuống còn 10px thay vì 28px, giải phóng tối đa diện tích
      cho bối cảnh sản phẩm nghệ thuật của FLUX.2 DiT.
    """
    if not rects:
        return np.zeros((height, width), dtype=np.float32)

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    fill_val = int(min(255, max(0, int(int_max * 255))))

    for (rx1, ry1, rx2, ry2) in rects:
        bx1 = max(0, int(rx1) - padding_px)
        by1 = max(0, int(ry1) - padding_px)
        bx2 = min(width, int(rx2) + padding_px)
        by2 = min(height, int(ry2) + padding_px)
        if bx2 <= bx1 or by2 <= by1:
            continue

        box_w = bx2 - bx1
        box_h = by2 - by1
        safe_radius = max(2, min(16, box_w // 2, box_h // 2))
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=safe_radius, fill=fill_val)

    if blur_radius_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius_px))

    mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
    return mask_arr.clip(0.0, 1.0)


__all__ = [
    "generate_omni_corridor_mask",
    "generate_omni_corridor_mask_from_rects",
]

