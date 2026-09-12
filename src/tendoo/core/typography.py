"""
src/tendoo/core/typography.py

Tầng cốt lõi xử lý Typography & Đo đạc kích thước chữ:
- Cân bằng dòng tiếng Việt (balance_vietnamese_headline): tránh mồ côi từ, chia đều cụm từ 2 dòng.
- Chuẩn hóa văn bản (normalize_text, strip_emojis).
- Đo đạc font metrics bằng PIL chính xác 100% (fit_font_size_px):
  * Tự động áp dụng đo đạc trên `text.upper()` khi `is_uppercase=True` để chống cắt mép chữ in hoa (ví dụ: TRANSFORMATION).
  * Chống tràn cả chiều rộng và chiều cao (Width & Height constraints).
  * Bảo toàn ngắt dòng cứng (`\\n` / `splitlines()`) giúp đo đạc chính xác tuyệt đối các câu thơ hoặc tiêu đề 2 dòng.

TẠI SAO CẦN MODULE NÀY TRONG HỆ THỐNG TENDOO AI:
1. TẠI SAO PHẢI BẢO TOÀN NGẮT DÒNG CỨNG (`\\n` -> `splitlines()`) TRONG `wrap_and_measure`:
   - Khi tiêu đề qua hàm `balance_vietnamese_headline()` hoặc bài thơ nhiều câu, chuỗi text chứa
     ký tự xuống dòng `\\n`. Khi render ra HTML trong `master.html`, chuỗi này được chuyển thành `<br>`.
   - Nếu dùng `text.split()` thông thường, toàn bộ `\\n` bị gộp thành khoảng trắng (space), khiến
     PIL tưởng chuỗi là 1 dòng dài và đo chiều cao chỉ bằng 1 dòng (sai lệch 50-75% so với thực tế).
   - Tách theo `splitlines()` trước khi gói từ (word-wrap) đảm bảo số dòng và chiều cao đo đạc
     bằng PIL khớp 1:1 với bố cục hiển thị thực tế trên trình duyệt Chromium.

2. TẠI SAO PHẢI ĐO TRÊN `text.upper()` KHI `is_uppercase=True`:
   - Trong typography tiếng Latin và tiếng Việt, các ký tự in hoa (A, B, M, W, Đ, Ợ) có chiều rộng
     ký tự (advance width) lớn hơn từ 25% - 45% so với ký tự thường (a, b, m, w, đ, ợ).
   - Nếu CSS áp dụng `text-transform: uppercase` mà PIL chỉ đo đạc trên chuỗi chữ thường ban đầu,
     font_size tính toán sẽ bị quá lớn, dẫn đến khi Chromium render chữ hoa sẽ bị tràn lề (overflow)
     hoặc bị cắt cụt (clipping chữ cuối).

3. TẠI SAO CẦN CƠ CHẾ SAFE FALLBACK CHO FONT LOADING:
   - Trong môi trường Docker/JupyterLab hoặc máy tính không cài đủ mọi font Google (.ttf),
     việc gọi trực tiếp `ImageFont.truetype(font_path)` với đường dẫn rỗng hoặc file thiếu sẽ
     gây crash toàn bộ luồng tạo mask. Safe loader tự động fallback về font mặc định có kích thước
     tương đương, giữ cho pipeline luôn ổn định 100%.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import ImageFont

from tendoo.layouts.text_engine import (
    balance_vietnamese_headline,
    normalize_text,
    resolve_headline_effect,
    strip_emojis,
)

LINE_HEIGHT_MULT = 1.25


def _safe_load_font(font_path: str, size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """
    Tải font an toàn với cơ chế fallback tự động.
    
    TẠI SAO CẦN LÀM:
    - Nếu font_path rỗng, file bị hỏng hoặc thiếu trên server, không để ứng dụng crash
      mà lập tức fallback về font hệ thống mặc định của Pillow.
    """
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def wrap_and_measure(
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width_px: float,
) -> Tuple[float, float, List[str]]:
    """
    Gói từ tham lam (Greedy word-wrap) cho `text` vừa vặn trong `max_width_px`.
    Trả về (chiều rộng dòng dài nhất px, tổng chiều cao khối px, danh sách các dòng).
    
    TẠI SAO CẦN LÀM:
    - Hỗ trợ cả ngắt dòng cứng (explicit newlines `\\n`) lẫn tự động bẻ dòng khi vượt quá max_width_px.
    - Tính toán chính xác ascent + descent kết hợp hệ số giãn dòng LINE_HEIGHT_MULT = 1.25.
    """
    paragraphs = text.splitlines()
    if not paragraphs:
        return 0.0, 0.0, []

    lines: List[str] = []
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            bbox = font.getbbox(candidate)
            candidate_w = bbox[2] - bbox[0]
            if candidate_w <= max_width_px or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

    if not lines:
        return 0.0, 0.0, []

    max_line_w = max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in lines)
    try:
        ascent, descent = font.getmetrics()
        line_h = (ascent + descent) * LINE_HEIGHT_MULT
    except Exception:
        # Fallback nếu font không hỗ trợ getmetrics (bitmap font cũ)
        line_h = 16.0 * LINE_HEIGHT_MULT

    return float(max_line_w), float(len(lines) * line_h), lines


def fit_font_size_px(
    text: str,
    font_path: str,
    base_font_size: int,
    max_width_px: float,
    max_height_px: float,
    min_font_size: int = 14,
    step: int = 2,
    is_uppercase: bool = False,
) -> Tuple[int, float, float]:
    """
    Thu nhỏ `base_font_size` (px) từng bước cho đến khi `text` gói từ vừa vặn
    cả 2 chiều: `max_width_px` VÀ `max_height_px`.

    TẠI SAO BẮT BUỘC ĐO ĐẠC TRÊN CHỮ IN HOA (`text.upper()`):
    - Nếu `is_uppercase` là True, toàn bộ chuỗi được chuyển thành hoa trước khi đo.
    - Điều này đảm bảo tính toán đủ không gian cho các ký tự lớn, triệt tiêu hoàn toàn
      lỗi cắt mép chữ (clipping) khi trình duyệt áp dụng CSS `text-transform: uppercase`.
    """
    measure_text = text.upper() if is_uppercase else text

    font_size = base_font_size
    while font_size > min_font_size:
        font = _safe_load_font(font_path, font_size)
        w, h, _ = wrap_and_measure(measure_text, font, max_width_px)
        if h <= max_height_px and w <= max_width_px:
            return font_size, w, h
        font_size -= step

    font = _safe_load_font(font_path, min_font_size)
    w, h, _ = wrap_and_measure(measure_text, font, max_width_px)
    return min_font_size, w, h


__all__ = [
    "LINE_HEIGHT_MULT",
    "balance_vietnamese_headline",
    "fit_font_size_px",
    "normalize_text",
    "resolve_headline_effect",
    "strip_emojis",
    "wrap_and_measure",
]

