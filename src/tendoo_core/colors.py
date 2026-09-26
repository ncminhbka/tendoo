"""
src/tendoo/core/colors.py

Automated Color & Contrast Harmony Engine (WCAG 2.1 & ITU-R BT.601):
====================================================================
- Tự động phân tích độ chói cảm nhận (Perceived Luminance) và sắc tướng chủ đạo (Dominant Hue)
  từ các pixel bối cảnh thực tế tại safe zone.
- Đảm bảo 100% chuẩn tương phản WCAG 2.1 (AA / AAA) cho toàn bộ tiêu đề, phụ đề và thẻ CTA.
- Tự động chuyển đổi giữa Palette A (Nền tối / Chữ phát quang ngọc trai) và Palette B (Nền sáng / Chữ trầm đậm).

(2026-09-13: gộp từ `layouts/color_engine.py` -- trước đó `core/colors.py` chỉ là 1 vỏ
bọc re-export mỏng, còn toàn bộ logic thật nằm ở `layouts/`, khiến tên thư mục "core" nói
ngược với nơi chứa logic thật. `core/` giờ là tầng nền tảng DUY NHẤT, không còn `layouts/`
song song nữa -- xem `tendoo/layouts/__init__.py`, giờ chỉ còn giữ lại
`layouts/base.py` làm cầu nối tương thích bắt buộc cho `tendoo_legacy`.)

TẠI SAO CẦN MODULE NÀY TRONG HỆ THỐNG TENDOO AI:
1. TẠI SAO BẮT BUỘC ĐO ĐỘ CHÓI ITU-R BT.601 (Y = 0.299R + 0.587G + 0.114B):
   - Mắt người không nhạy cảm đều với các kênh màu: mắt phản ứng mạnh nhất với màu xanh lá lục (58.7%),
     kế tiếp là đỏ (29.9%) và yếu nhất với xanh dương (11.4%).
   - Nếu chỉ tính trung bình cộng đơn giản (R+G+B)/3, một bức ảnh có nền vàng chanh rực rỡ (rất chói mắt)
     sẽ bị đánh giá sai là nền tối, dẫn đến sinh chữ trắng khiến người dùng không đọc được chữ.
   - Công thức ITU-R BT.601 phản ánh chính xác quang sai sinh lý của mắt người. Khi Y < 128, hệ thống
     tự động chuyển sang Palette A (chữ sáng trên nền tối); khi Y >= 128, kích hoạt Palette B (chữ trầm trên nền sáng).

2. TẠI SAO DÙNG GÓC ĐỐI XỨNG (COMPLEMENTARY HUE = HUE + 180°) CHO THẺ CTA PILL:
   - Theo định lý bánh xe màu sắc Itten, hai màu nằm đối diện nhau 180 độ mang lại độ tương phản cảm nhận
     mạnh nhất (Visual Contrast Pop) mà không gây cảm giác xung đột thị giác (Color Clash).
   - Nút hành động kêu gọi mua sắm (CTA / Promo Badge) cần nổi bật ngay trong 3 giây đầu tiên tiếp xúc
     với khách hàng, và việc dùng Complementary Hue giúp nút CTA luôn nổi bật trên nền mà vẫn hòa quyện
     tự nhiên vào tổng thể bố cục.

3. TẠI SAO BẢO TOÀN LUMINANCE NỀN THẬT KHI NHẬN `user_hue` OVERRIDE:
   - Khi khách hàng chỉ định "màu chủ đạo thương hiệu" (Brand Color, ví dụ: màu đỏ Viettel hay cam Shopee),
     nếu ghi đè toàn bộ màu sắc mà bỏ qua nền ảnh thật thì chữ có nguy cơ bị chìm nghỉm vào bối cảnh.
   - Thuật toán chỉ ghi đè tham số Hue (sắc thái), trong khi toàn bộ logic quyết định tương phản
     (chữ trắng hay chữ đen, đổ bóng sáng hay tối) vẫn giữ nguyên theo độ chói thực tế của pixel ảnh nền.
"""

from __future__ import annotations

import colorsys
import re
from typing import Optional, Tuple, Union

import numpy as np

from tendoo_core.palette import ColorPalette


def parse_color_to_rgb(color_str: str) -> Tuple[float, float, float]:
    """Chuyển đổi các định dạng màu CSS (hex, rgb, rgba, hsl, hsla) sang (r, g, b) trong khoảng [0.0, 1.0]."""
    if not color_str:
        return (1.0, 1.0, 1.0)
    s = color_str.strip().lower()

    # 1. Hex format
    if s.startswith("#") or (len(s) in (3, 6, 8) and all(c in "0123456789abcdef" for c in s)):
        clean = s.lstrip("#")
        if len(clean) == 3:
            clean = "".join(c * 2 for c in clean)
        if len(clean) >= 6:
            try:
                r = int(clean[0:2], 16) / 255.0
                g = int(clean[2:4], 16) / 255.0
                b = int(clean[4:6], 16) / 255.0
                return (r, g, b)
            except ValueError:
                pass

    # 2. rgb / rgba format: rgb(255, 255, 255) / rgba(12, 16, 24, 0.8)
    m_rgb = re.search(r"rgba?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", s)
    if m_rgb:
        r = float(m_rgb.group(1)) / 255.0
        g = float(m_rgb.group(2)) / 255.0
        b = float(m_rgb.group(3)) / 255.0
        return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))

    # 3. hsl / hsla format: hsl(210, 50%, 80%)
    m_hsl = re.search(r"hsla?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)%?\s*,\s*([\d.]+)%?", s)
    if m_hsl:
        h = float(m_hsl.group(1)) % 360.0
        sat = float(m_hsl.group(2)) / 100.0 if float(m_hsl.group(2)) > 1.0 else float(m_hsl.group(2))
        lum = float(m_hsl.group(3)) / 100.0 if float(m_hsl.group(3)) > 1.0 else float(m_hsl.group(3))
        r, g, b = colorsys.hls_to_rgb(h / 360.0, lum, sat)
        return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))

    return (1.0, 1.0, 1.0)


def calculate_wcag_luminance(color: Union[str, Tuple[float, float, float]]) -> float:
    """
    Tính Relative Luminance theo chuẩn quang học WCAG 2.1 (sRGB gamma expansion):
    L = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin trong khoảng [0.0..1.0].
    """
    if isinstance(color, str):
        r, g, b = parse_color_to_rgb(color)
    else:
        r, g, b = color
        if r > 1.0 or g > 1.0 or b > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0

    def _to_linear(c: float) -> float:
        c = max(0.0, min(1.0, c))
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = _to_linear(r)
    g_lin = _to_linear(g)
    b_lin = _to_linear(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def calculate_contrast_ratio(color_a: str, color_b: str) -> float:
    """
    Tính tỉ số tương phản WCAG 2.1 giữa 2 màu:
    CR = (L_max + 0.05) / (L_min + 0.05) trong khoảng [1.0..21.0].
    """
    l1 = calculate_wcag_luminance(color_a)
    l2 = calculate_wcag_luminance(color_b)
    l_max = max(l1, l2)
    l_min = min(l1, l2)
    return (l_max + 0.05) / (l_min + 0.05)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Chuyển đổi r, g, b [0.0..1.0] hoặc [0..255] thành '#RRGGBB'."""
    if r <= 1.0 and g <= 1.0 and b <= 1.0:
        ir, ig, ib = int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0))
    else:
        ir, ig, ib = int(round(r)), int(round(g)), int(round(b))
    return f"#{max(0, min(255, ir)):02X}{max(0, min(255, ig)):02X}{max(0, min(255, ib)):02X}"


def get_contrasting_text_color(bg_color: str) -> str:
    """Trả về '#0F172A' (chữ đậm) hoặc '#FFFFFF' (chữ trắng) -- màu nào có tỉ số WCAG cao hơn.

    Trước đây dùng ngưỡng cứng luminance > 0.40, trong khi điểm hoà thật giữa 2 lựa chọn ở
    khoảng 0.18 -> mọi màu độ sáng trung bình (hồng, cam, cyan, vàng đồng: 0.18-0.40) nhận chữ
    trắng dù chữ đậm tương phản gấp 2-3 lần. Đo 26/09 (squint §4.5 điều kiện 3): CTA chữ trắng
    trên hồng #F472B6 chỉ 2.65:1 (< cả ngưỡng 3.0 chữ lớn); chữ đậm đạt ~7:1."""
    dark, light = "#0F172A", "#FFFFFF"
    return dark if calculate_contrast_ratio(dark, bg_color) >= calculate_contrast_ratio(light, bg_color) else light


def ensure_contrast(
    fg_color: str,
    bg_color: str,
    min_ratio: float = 4.5,
    preserve_hue: bool = True,
) -> str:
    """
    Tự động thẩm định và điều chỉnh độ sáng (Lightness) của fg_color để đạt tỉ số
    tương phản WCAG 2.1 tối thiểu (mặc định 4.5:1 cho text, 3.0:1 cho icon).
    Bảo toàn 100% góc sắc thái (Hue) và độ bão hòa (Saturation) của màu gốc.
    """
    if not fg_color:
        return "#FFFFFF" if calculate_wcag_luminance(bg_color) < 0.45 else "#0F172A"

    current_ratio = calculate_contrast_ratio(fg_color, bg_color)
    if current_ratio >= min_ratio:
        return fg_color

    r, g, b = parse_color_to_rgb(fg_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    bg_lum = calculate_wcag_luminance(bg_color)

    prefer_lighten = (bg_lum < 0.45)

    if prefer_lighten:
        best_candidate = fg_color
        for test_l in [0.65, 0.72, 0.80, 0.86, 0.92, 0.97]:
            cr, cg, cb = colorsys.hls_to_rgb(h, test_l, min(s, 0.95))
            cand_hex = rgb_to_hex(cr, cg, cb)
            if calculate_contrast_ratio(cand_hex, bg_color) >= min_ratio:
                return cand_hex
            best_candidate = cand_hex
        for test_l in [0.25, 0.15, 0.06]:
            cr, cg, cb = colorsys.hls_to_rgb(h, test_l, min(s, 0.95))
            cand_hex = rgb_to_hex(cr, cg, cb)
            if calculate_contrast_ratio(cand_hex, bg_color) >= min_ratio:
                return cand_hex
        return best_candidate
    else:
        best_candidate = fg_color
        for test_l in [0.38, 0.28, 0.20, 0.14, 0.08]:
            cr, cg, cb = colorsys.hls_to_rgb(h, test_l, min(s, 0.95))
            cand_hex = rgb_to_hex(cr, cg, cb)
            if calculate_contrast_ratio(cand_hex, bg_color) >= min_ratio:
                return cand_hex
            best_candidate = cand_hex
        for test_l in [0.85, 0.92, 0.97]:
            cr, cg, cb = colorsys.hls_to_rgb(h, test_l, min(s, 0.95))
            cand_hex = rgb_to_hex(cr, cg, cb)
            if calculate_contrast_ratio(cand_hex, bg_color) >= min_ratio:
                return cand_hex
        return best_candidate


def compute_relative_luminance(rgb_array: np.ndarray) -> float:
    """
    Tính độ chói cảm nhận theo chuẩn ITU-R BT.601:
    Y = 0.299*R + 0.587*G + 0.114*B trong khoảng [0..255].
    """
    if rgb_array.size == 0:
        return 128.0
    r = rgb_array[:, :, 0].astype(np.float32)
    g = rgb_array[:, :, 1].astype(np.float32)
    b = rgb_array[:, :, 2].astype(np.float32)
    return float(np.mean(0.299 * r + 0.587 * g + 0.114 * b))


def extract_dominant_hsv(rgb_array: np.ndarray) -> Tuple[int, float, float]:
    """
    Trích xuất sắc tướng trung vị (dominant hue 0..359 độ), độ bão hòa sat [0..1]
    và giá trị độ sáng val [0..1].
    """
    if rgb_array.size == 0:
        return (0, 0.0, 0.5)
    med_rgb = np.median(rgb_array.reshape(-1, 3), axis=0).astype(float) / 255.0
    h, s, v = colorsys.rgb_to_hsv(med_rgb[0], med_rgb[1], med_rgb[2])
    return (int(h * 360), float(s), float(v))


def hsl_to_rgb(h: float, s: float, lightness: float) -> Tuple[float, float, float]:
    """Chuyển đổi H [0..360], S [0..1], L [0..1] sang RGB [0..255]."""
    r, g, b = colorsys.hls_to_rgb(h / 360.0, lightness, s)
    return (r * 255.0, g * 255.0, b * 255.0)


def hex_to_hue(hex_str: str) -> Optional[int]:
    """
    Chuyển đổi chuỗi màu hex '#RRGGBB' hoặc '#RGB' thành góc Hue [0..359] độ.
    Trả về None nếu chuỗi rỗng hoặc không hợp lệ.
    """
    if not hex_str:
        return None
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
    except ValueError:
        return None
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 360)


def analyze_color_harmony(
    img_np: np.ndarray,
    crop_zone: Tuple[float, float, float, float],
    color_mode: str = "auto",
    font_style: str = "modern_sans",
    user_hue: Optional[int] = None,
) -> ColorPalette:
    """
    Tính toán bảng màu toán học hài hòa cho tiêu đề, thẻ CTA, đổ bóng và kính mờ (glassmorphism)
    dựa trên pixel thực tế của nền ảnh.

    Args:
        img_np: Mảng ảnh RGB [H, W, 3], uint8 trong khoảng [0..255].
        crop_zone: Tọa độ chuẩn hóa (y1, x1, y2, x2) định nghĩa safe zone cần phân tích màu.
        color_mode: 'auto', 'dark', hoặc 'light'.
        font_style: 'modern_sans' hoặc 'luxury_serif'.
        user_hue: Góc Hue [0..359] độ ghi đè màu chủ đạo do người dùng chọn (xem hex_to_hue()).
    """
    h, w, _ = img_np.shape
    y1, x1, y2, x2 = crop_zone
    crop = img_np[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]

    lum = compute_relative_luminance(crop)
    extracted_hue, sat, val = extract_dominant_hsv(crop)
    hue = user_hue if user_hue is not None else extracted_hue

    if color_mode == "dark":
        is_dark = True
    elif color_mode == "light":
        is_dark = False
    else:
        # Ngưỡng WCAG chuẩn phân biệt nền tối và sáng
        is_dark = (lum < 128.0)

    # Thẻ CTA tương phản cao sử dụng góc đối xứng (180 độ đối xứng trên bánh xe màu)
    comp_hue = (hue + 180) % 360

    if is_dark:
        # ==========================================
        # PALETTE A: NỀN TỐI / CHỮ SÁNG PHÁT QUANG
        # ==========================================
        if font_style == "luxury_serif":
            # Gradient vàng 24K ánh kim cao cấp
            headline_color = (
                "linear-gradient(180deg, #FFFDF2 0%, #F8DC88 35%, #DAA520 70%, #B8860B 100%)"
            )
            headline_is_gradient = True
        else:
            # Trắng ngọc trai pha sắc thái nhẹ của bối cảnh
            headline_color = f"hsl({hue}, 20%, 97%)"
            headline_is_gradient = False

        # Phụ đề: Ánh bạc thanh lịch độ sáng cao
        sub_color = f"hsl({hue}, 18%, 88%)"

        # Đổ bóng kép sắc nét chống nhòe
        text_shadow = "0 1px 4px rgba(0, 0, 0, 0.95), 0 2px 10px rgba(0, 0, 0, 0.70)"

        # Thẻ CTA Pill Badge
        badge_bg = (
            f"linear-gradient(135deg, hsl({comp_hue}, 95%, 54%) 0%, "
            f"hsl({(comp_hue + 25) % 360}, 90%, 45%) 100%)"
        )

        # Tính độ chói cảm nhận của nền badge để đảm bảo tương phản WCAG
        badge_rgb = hsl_to_rgb(comp_hue, 0.95, 0.54)
        badge_lum = 0.299 * badge_rgb[0] + 0.587 * badge_rgb[1] + 0.114 * badge_rgb[2]

        # Nếu badge thuộc dải màu sáng (vàng, cam sáng, xanh lá chanh) -> dùng chữ carbon đậm. Ngược lại -> chữ trắng tuyết
        if badge_lum > 135.0 or (30 <= comp_hue <= 95):
            badge_text = "#0D0D14"
            badge_border = "rgba(0, 0, 0, 0.20)"
        else:
            badge_text = "#FFFFFF"
            badge_border = "rgba(255, 255, 255, 0.38)"

        badge_shadow = f"0 4px 18px hsla({comp_hue}, 90%, 50%, 0.50)"
        glass_bg = "rgba(255, 255, 255, 0.12)"
        glass_border = "rgba(255, 255, 255, 0.25)"

        accent_color = f"hsl({comp_hue}, 95%, 72%)"
        footer_text = f"hsl({hue}, 15%, 82%)"
        footer_bg = "rgba(10, 10, 10, 0.65)"
    else:
        # ==========================================
        # PALETTE B: NỀN SÁNG / CHỮ TRẦM ĐẬM ĐẮC TẢ
        # ==========================================
        text_hue_sat = min(sat * 1.2, 0.75)
        headline_color = f"hsl({hue}, {int(text_hue_sat * 100)}%, 14%)"
        headline_is_gradient = False

        sub_color = f"hsl({hue}, {int(text_hue_sat * 80)}%, 26%)"
        text_shadow = "0 1px 3px rgba(255, 255, 255, 0.90), 0 1px 4px rgba(0, 0, 0, 0.08)"

        badge_bg = (
            f"linear-gradient(135deg, hsl({comp_hue}, 95%, 48%) 0%, "
            f"hsl({(comp_hue + 20) % 360}, 90%, 38%) 100%)"
        )
        badge_text = "#FFFFFF"
        badge_border = "rgba(255, 255, 255, 0.38)"
        badge_shadow = f"0 4px 14px hsla({comp_hue}, 90%, 45%, 0.38)"
        glass_bg = "rgba(255, 255, 255, 0.85)"
        glass_border = f"hsl({hue}, 25%, 82%)"
        accent_color = f"hsl({comp_hue}, 95%, 36%)"
        footer_text = f"hsl({hue}, 40%, 18%)"
        footer_bg = "rgba(255, 255, 255, 0.88)"

    # Thẩm định và tạo bộ token tương phản an toàn chuẩn WCAG 2.1
    if is_dark:
        effective_card_bg = "#0C101A"
        on_card_accent = ensure_contrast(accent_color, effective_card_bg, min_ratio=4.5)
        on_card_title = ensure_contrast(f"hsl({hue}, 80%, 78%)", effective_card_bg, min_ratio=4.5)
        on_canvas_accent = ensure_contrast(accent_color, "#000000", min_ratio=4.5)
    else:
        effective_card_bg = "#FFFFFF"
        on_card_accent = ensure_contrast(accent_color, effective_card_bg, min_ratio=4.5)
        on_card_title = ensure_contrast(f"hsl({hue}, 80%, 25%)", effective_card_bg, min_ratio=4.5)
        on_canvas_accent = ensure_contrast(accent_color, "#FFFFFF", min_ratio=4.5)

    return ColorPalette(
        is_dark=is_dark,
        luminance=lum,
        hue=hue,
        comp_hue=comp_hue,
        headline_color=headline_color,
        headline_is_gradient=headline_is_gradient,
        sub_color=sub_color,
        text_shadow=text_shadow,
        badge_bg=badge_bg,
        badge_text=badge_text,
        badge_border=badge_border,
        badge_shadow=badge_shadow,
        glass_bg=glass_bg,
        glass_border=glass_border,
        accent_color=accent_color,
        on_card_accent=on_card_accent,
        on_card_title=on_card_title,
        on_canvas_accent=on_canvas_accent,
        footer_text=footer_text,
        footer_bg=footer_bg,
    )


__all__ = [
    "ColorPalette",
    "analyze_color_harmony",
    "calculate_contrast_ratio",
    "calculate_wcag_luminance",
    "compute_relative_luminance",
    "ensure_contrast",
    "extract_dominant_hsv",
    "get_contrasting_text_color",
    "hex_to_hue",
    "hsl_to_rgb",
    "parse_color_to_rgb",
    "rgb_to_hex",
]
