"""
src/tendoo/engine/mask.py

Bộ sinh Parametric Corridor Mask Hợp Nhất (Single Unified Mask):
- Hợp nhất bounding box của tất cả các AdaptiveBlock đang hoạt động thành 1 Binary Mask mềm duy nhất.
- Hỗ trợ kiến trúc 2-branch velocity blending: 1 Scene Branch + 1 Corridor Branch chung.
- Giữ nguyên chi phí tính toán O(1) bất biến (~6s/ảnh trên 2x A30).
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
    padding_px: int = 24,
    blur_radius_px: int = 36,
    int_max: float = 0.85,
) -> np.ndarray:
    """
    Sinh 1 ma trận mask float32 [height, width] với giá trị [0.0, 1.0]:
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

        # Vẽ hình chữ nhật bo góc nhẹ
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=16, fill=fill_val)

    # Làm mượt biên với Gaussian Blur để velocity blending hòa trộn tự nhiên
    if blur_radius_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius_px))

    mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
    return mask_arr.clip(0.0, 1.0)
