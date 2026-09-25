"""
src/tendoo_v3/mock_backgrounds.py

Tạo các ảnh nền Mock thẩm mỹ cao (High-Aesthetic Mock Backdrops) cho quá trình kiểm thử:
- Dark Luxury Studio (Nền tối ánh sáng studio, viền sáng mềm).
- Warm Oak Wood Flatlay (Mặt bàn gỗ sồi mộc mạc ấm áp).
- Luxury White Marble (Đá cẩm thạch trắng viền vân vàng kim).
- Pastel Spa Pet (Tone pastel hồng kem và xanh mint cho Spa thú cưng).
- Midnight Cinematic Gym (Phòng gym tương phản cao, ánh đèn vàng ấm & đỏ).
- Tech Cyber Neon (Nền công nghệ số xanh cyan mờ ảo).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_gradient_backdrop(
    width: int,
    height: int,
    tone: str = "dark_luxury",
) -> Image.Image:
    """Tạo ảnh PIL bối cảnh giả lập (Mock Backdrop) chất lượng cao theo từng phong cách."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    w_f = float(width)
    h_f = float(height)

    if tone == "dark_luxury":
        # Gradient studio tối với điểm sáng chéo (vignette mềm)
        for y in range(height):
            ratio = y / h_f
            r = int(18 + 12 * (1.0 - ratio))
            g = int(20 + 10 * (1.0 - ratio))
            b = int(28 + 16 * (1.0 - ratio))
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        # Thêm vệt sáng studio mềm
        light_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(light_img)
        cx, cy = int(width * 0.7), int(height * 0.4)
        rad = int(min(width, height) * 0.55)
        l_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(220, 200, 160, 40))
        light_img = light_img.filter(ImageFilter.GaussianBlur(radius=int(min(width, height) * 0.15)))
        img.paste(Image.alpha_composite(img.convert("RGBA"), light_img).convert("RGB"), (0, 0))

    elif tone == "warm_rustic":
        # Nền gỗ sồi ấm áp / vàng nâu
        for y in range(height):
            ratio = y / h_f
            r = int(48 + 35 * ratio)
            g = int(32 + 25 * ratio)
            b = int(20 + 16 * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        # Vệt sáng hoàng hôn chiếu qua cửa sổ
        light_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(light_img)
        cx, cy = int(width * 0.5), int(height * 0.5)
        rad = int(min(width, height) * 0.6)
        l_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(255, 180, 100, 50))
        light_img = light_img.filter(ImageFilter.GaussianBlur(radius=int(min(width, height) * 0.2)))
        img.paste(Image.alpha_composite(img.convert("RGBA"), light_img).convert("RGB"), (0, 0))

    elif tone == "pastel":
        # Tone hồng kem & xanh mint mềm mại cho Spa / Mỹ phẩm
        for y in range(height):
            ratio = y / h_f
            r = int(245 - 20 * ratio)
            g = int(230 + 15 * ratio)
            b = int(235 + 10 * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        light_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(light_img)
        cx, cy = int(width * 0.3), int(height * 0.7)
        rad = int(min(width, height) * 0.5)
        l_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(180, 240, 220, 60))
        light_img = light_img.filter(ImageFilter.GaussianBlur(radius=int(min(width, height) * 0.18)))
        img.paste(Image.alpha_composite(img.convert("RGBA"), light_img).convert("RGB"), (0, 0))

    elif tone == "cinema_red":
        # Phòng gym / thể thao: Đen tuyền kết hợp vệt sáng đỏ & vàng cinematic
        for y in range(height):
            ratio = y / h_f
            r = int(28 + 20 * ratio)
            g = int(10 + 5 * ratio)
            b = int(12 + 6 * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        light_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(light_img)
        cx, cy = int(width * 0.5), int(height * 0.5)
        rad = int(min(width, height) * 0.5)
        l_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(240, 40, 30, 45))
        light_img = light_img.filter(ImageFilter.GaussianBlur(radius=int(min(width, height) * 0.16)))
        img.paste(Image.alpha_composite(img.convert("RGBA"), light_img).convert("RGB"), (0, 0))

    elif tone == "cyber_neon":
        # Công nghệ / Fintech / Cyberpunk: Tối sẫm viền neon cyan
        for y in range(height):
            ratio = y / h_f
            r = int(8 + 10 * ratio)
            g = int(12 + 18 * ratio)
            b = int(24 + 35 * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        light_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(light_img)
        cx, cy = int(width * 0.7), int(height * 0.5)
        rad = int(min(width, height) * 0.5)
        l_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(0, 240, 255, 45))
        light_img = light_img.filter(ImageFilter.GaussianBlur(radius=int(min(width, height) * 0.16)))
        img.paste(Image.alpha_composite(img.convert("RGBA"), light_img).convert("RGB"), (0, 0))

    else:  # light_clean / marble
        # Nền đá cẩm thạch trắng sáng
        for y in range(height):
            ratio = y / h_f
            val = int(235 - 15 * ratio)
            draw.line([(0, y), (width, y)], fill=(val, val, val + 2))

    return img
