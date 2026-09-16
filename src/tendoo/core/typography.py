"""
src/tendoo/core/typography.py

Tầng cốt lõi xử lý Typography & Đo đạc kích thước chữ (gộp 2026-09-13 từ
`layouts/text_engine.py` + `core/typography.py` cũ -- trước đó `core/typography.py` chỉ
re-export từ `layouts/text_engine.py`, `core/` giờ là tầng nền tảng DUY NHẤT):
- Vietnamese Compound Word Protection + Orphan Word Prevention (`balance_vietnamese_headline`).
- Tofu Glyph Eradication (`strip_emojis`) + chuẩn hóa văn bản (`normalize_text`).
- 8 hiệu ứng chữ CSS phong phú (`resolve_headline_effect`) -- LƯU Ý: hiện là dead code
  (kết quả headline_fill_css/wrap_filter_css chưa từng được master.html tiêu thụ; hiệu ứng
  chữ thực tế dùng cơ chế effect-*/`STYLE_HINT_DEFAULT_EFFECT` đơn giản hơn trong
  engine/layout.py + core/style.py) -- giữ lại vì vẫn có thể hữu ích nếu mở rộng CSS sau này.
- Đo đạc font metrics bằng PIL chính xác 100% (`fit_font_size_px`):
  * Tự động áp dụng đo đạc trên `text.upper()` khi `is_uppercase=True` để chống cắt mép chữ in hoa (ví dụ: TRANSFORMATION).
  * Chống tràn cả chiều rộng và chiều cao (Width & Height constraints).
  * Bảo toàn ngắt dòng cứng (`\\n` / `splitlines()`) giúp đo đạc chính xác tuyệt đối các câu thơ hoặc tiêu đề 2 dòng.

TẠI SAO CẦN MODULE NÀY TRONG HỆ THỐNG TENDOO AI:
1. TẠI SAO PHẢI BẢO VỆ TỪ GHÉP TIẾNG VIỆT BẰNG HÀM CHI PHÍ NĂNG LƯỢNG:
   - Ngữ pháp tiếng Việt mang tính đơn lập, nghĩa của câu phụ thuộc vào các khối từ ghép 2 từ tố.
     Nếu một tiêu đề như "ĐẠI TIỆC MÙA HÈ" bị ngắt thành "ĐẠI TIỆC MÙA" / "HÈ", từ "HÈ" bị bỏ rơi đơn độc,
     làm mất đi tính trang trọng và mạch lạc của thương hiệu quảng cáo.
   - Hàm `balance_vietnamese_headline` tính toán năng lượng thẩm mỹ:
     Cost = |len(line1) - target| + |len(line2) - target| + Penalty(bẻ từ ghép: 60) + Penalty(mồ côi từ: 40).
     Điểm ngắt tối ưu luôn giữ trọn vẹn cụm từ ghép ("ĐẠI TIỆC" / "MÙA HÈ"), tạo nên sự cân đối thị giác hoàn hảo.

2. TẠI SAO PHẢI XÓA EMOJI BẰNG REGEX THAY VÌ ĐỂ TRÌNH DUYỆT TỰ XỬ LÝ:
   - Môi trường Linux Docker / JupyterLab thường thiếu font màu emoji chuyên dụng (như Apple Color Emoji).
   - Khi gặp emoji thô từ prompt của người dùng (🚀, 🔥, ✨), Chromium hiển thị ô vuông rác (tofu glyph □).
   - `strip_emojis` thanh lọc 100% ký tự biểu cảm rác mà vẫn bảo toàn tuyệt đối toàn bộ dấu phụ tiếng Việt
     (Á, Ệ, Ộ, Ứ, Ờ), dấu ngoặc kép và ký hiệu tiền tệ (₫, $).

3. TẠI SAO PHẢI BẢO TOÀN NGẮT DÒNG CỨNG (`\\n` -> `splitlines()`) TRONG `wrap_and_measure`:
   - Khi tiêu đề qua hàm `balance_vietnamese_headline()` hoặc bài thơ nhiều câu, chuỗi text chứa
     ký tự xuống dòng `\\n`. Khi render ra HTML trong `master.html`, chuỗi này được chuyển thành `<br>`.
   - Nếu dùng `text.split()` thông thường, toàn bộ `\\n` bị gộp thành khoảng trắng (space), khiến
     PIL tưởng chuỗi là 1 dòng dài và đo chiều cao chỉ bằng 1 dòng (sai lệch 50-75% so với thực tế).
   - Tách theo `splitlines()` trước khi gói từ (word-wrap) đảm bảo số dòng và chiều cao đo đạc
     bằng PIL khớp 1:1 với bố cục hiển thị thực tế trên trình duyệt Chromium.

4. TẠI SAO PHẢI ĐO TRÊN `text.upper()` KHI `is_uppercase=True`:
   - Trong typography tiếng Latin và tiếng Việt, các ký tự in hoa (A, B, M, W, Đ, Ợ) có chiều rộng
     ký tự (advance width) lớn hơn từ 25% - 45% so với ký tự thường (a, b, m, w, đ, ợ).
   - Nếu CSS áp dụng `text-transform: uppercase` mà PIL chỉ đo đạc trên chuỗi chữ thường ban đầu,
     font_size tính toán sẽ bị quá lớn, dẫn đến khi Chromium render chữ hoa sẽ bị tràn lề (overflow)
     hoặc bị cắt cụt (clipping chữ cuối).

5. TẠI SAO CẦN CƠ CHẾ SAFE FALLBACK CHO FONT LOADING:
   - Trong môi trường Docker/JupyterLab hoặc máy tính không cài đủ mọi font Google (.ttf),
     việc gọi trực tiếp `ImageFont.truetype(font_path)` với đường dẫn rỗng hoặc file thiếu sẽ
     gây crash toàn bộ luồng tạo mask. Safe loader tự động fallback về font mặc định có kích thước
     tương đương, giữ cho pipeline luôn ổn định 100%.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import ImageFont

logger = logging.getLogger(__name__)

LINE_HEIGHT_MULT = 1.25

# ==============================================================================
# PHẦN 1: CHUẨN HÓA VĂN BẢN & CÂN DÒNG TIẾNG VIỆT (từ layouts/text_engine.py cũ)
# ==============================================================================

# Danh mục 37 cụm từ ghép tiếng Việt thương mại bắt buộc không được ngắt dòng đôi
VIETNAMESE_COMPOUND_WORDS: Set[str] = {
    "chống ồn",
    "chủ động",
    "thế hệ",
    "không dây",
    "công nghệ",
    "tự nhiên",
    "thanh mát",
    "đặc biệt",
    "tri ân",
    "khách hàng",
    "khuyến mại",
    "ưu đãi",
    "mùa hè",
    "mùa thu",
    "mùa đông",
    "mùa xuân",
    "trung thu",
    "đẳng cấp",
    "cao cấp",
    "hoàn hảo",
    "thời trang",
    "sản phẩm",
    "nước hoa",
    "âm thanh",
    "đỉnh cao",
    "trà đào",
    "cam sả",
    "bánh nướng",
    "bánh dẻo",
    "giảm giá",
    "tặng kèm",
    "bảo hành",
    "chính hãng",
    "toàn quốc",
    "bùng nổ",
    "tuyệt hảo",
    "sang trọng",
    "tinh tế",
}


# Unicode Emoji, Pictograph, and Miscellaneous Symbols that lack glyphs in standard typography fonts
# NOTE: Typographical star glyphs ★ (★) and ☆ (☆) are explicitly EXCLUDED to allow star rating badges.
EMOJI_PATTERN = re.compile(
    r"["
    r"\U0001F000-\U0001FAFF"  # Emojis & Pictographs (1F300-1F9FF, 1FA00-1FAFF, symbols, etc.)
    r"☀-☄☇-➿"  # Misc symbols & Dingbats (weather, arrows) - EXCLUDES ★ (★) and ☆ (☆)
    r"⌀-⏿"          # Misc Technical
    r"⭑-⭕"          # Symbols - EXCLUDES ⭐ (⭐) which is converted to ★
    r"︀-️"          # Variation Selectors
    r"‍"                  # Zero-width joiner
    r"]+",
    flags=re.UNICODE,
)


def strip_emojis(text: str) -> str:
    """
    Strips raw Unicode emojis and unrenderable pictographic symbols to prevent
    them from displaying as square missing-glyph tofu boxes (□) in headless Chromium / Playwright.
    Preserves 100% of Vietnamese diacritics, currency marks, dashes, quotes, and punctuation.
    Preserves star rating glyphs (★, ☆) while converting emoji stars (⭐) to standard typographical stars (★).
    """
    if not text:
        return ""
    cleaned = text.replace("⭐", "★")
    cleaned = EMOJI_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" ?\n ?", "\n", cleaned)
    return cleaned.strip()


def normalize_text(text: str) -> str:
    """Cleans up literal escapes, strips unrenderable emojis (tofu prevention), and normalizes whitespace."""
    if not text:
        return ""
    cleaned = (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\r\n", "\n")
        .strip()
    )
    cleaned = strip_emojis(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


_COMPOUND_STRIP_CHARS = ".,!?;:\"'…“”"


def is_compound_pair(w1: str, w2: str) -> bool:
    """Checks if two consecutive words form a protected compound in lowercase.

    Strips trailing/leading punctuation from each word before comparing (2026-09-12
    fix): a headline like "MÙA HÈ!" used to build the lookup key "mùa hè!" (with the
    "!" still attached), which never matches the punctuation-free "mùa hè" entry in
    VIETNAMESE_COMPOUND_WORDS -- the anti-broken-compound penalty silently failed to
    fire on real ad copy whenever a compound word was followed by punctuation.
    """
    w1c = w1.strip().lower().strip(_COMPOUND_STRIP_CHARS)
    w2c = w2.strip().lower().strip(_COMPOUND_STRIP_CHARS)
    pair = f"{w1c} {w2c}"
    return pair in VIETNAMESE_COMPOUND_WORDS


def balance_vietnamese_headline(
    raw_headline: str,
    max_one_line_chars: int = 16,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Intelligently splits a Vietnamese headline into 1 or 2 visually and semantically balanced lines.

    Rules:
      - If raw_headline already contains explicit user-specified '\\n', respect user intent!
      - If total chars <= max_one_line_chars: Keep as 1 line.
      - If total chars > max_one_line_chars: Evaluates all split points using cost function:
          Cost = |len(line1) - len(line2)| + Penalty(Breaking Compound Word) + Penalty(Orphan Word)
      - Returns optimal 2 lines and calculated font sizing metrics.
    """
    clean = normalize_text(raw_headline)
    if not clean:
        return [], {"font_size": 44, "line_height": 1.15, "letter_spacing": 0.0}

    # Explicit user split
    if "\n" in clean:
        lines = [ln.strip() for ln in clean.split("\n") if ln.strip()]
        metrics = compute_font_ladder(lines)
        return lines, metrics

    words = clean.split()
    total_chars = len(clean)

    # Short headline -> single line
    if total_chars <= max_one_line_chars or len(words) <= 2:
        lines = [clean]
        metrics = compute_font_ladder(lines)
        return lines, metrics

    # Automatic 2-line split search
    target_len = total_chars / 2.0
    best_split_idx = 1
    min_cost = float("inf")

    for i in range(1, len(words)):
        w_prev = words[i - 1]
        w_curr = words[i]

        line1 = " ".join(words[:i])
        line2 = " ".join(words[i:])

        # Base cost: visual asymmetry between line 1 and line 2
        char_diff = abs(len(line1) - target_len) + abs(len(line2) - target_len)
        cost = char_diff

        # 1. Heavy penalty if splitting in the middle of a protected compound word
        if is_compound_pair(w_prev, w_curr):
            cost += 60.0

        # 2. Penalty for orphan word on line 2 (e.g. line 2 is only 1 word <= 4 chars)
        if len(words[i:]) == 1 and len(words[i]) <= 4:
            cost += 40.0

        # 3. Penalty for orphan word on line 1 (e.g. line 1 is only 1 word <= 3 chars)
        if i == 1 and len(words[0]) <= 3:
            cost += 40.0

        if cost < min_cost:
            min_cost = cost
            best_split_idx = i

    line1 = " ".join(words[:best_split_idx])
    line2 = " ".join(words[best_split_idx:])
    lines = [line1, line2]
    metrics = compute_font_ladder(lines)
    return lines, metrics


def compute_font_ladder(lines: List[str]) -> Dict[str, Any]:
    """
    Computes responsive font size, line-height, and tracking based on line count and character density.
    Adheres to golden-ratio typographic scale and extreme scale contrast for commercial posters.
    Designed for 1024x1024 poster canvas coordinates.
    """
    if not lines:
        return {"font_size": 52, "line_height": 1.10, "letter_spacing": 0.0}

    max_len = max(len(ln) for ln in lines)
    num_lines = len(lines)

    if num_lines == 1:
        if max_len <= 10:
            font_size = 78
            line_height = 1.05
            letter_spacing = -0.8
        elif max_len <= 15:
            font_size = 68
            line_height = 1.08
            letter_spacing = -0.5
        elif max_len <= 22:
            font_size = 56
            line_height = 1.10
            letter_spacing = -0.3
        else:
            font_size = 46
            line_height = 1.12
            letter_spacing = 0.0
    elif num_lines == 2:
        if max_len <= 14:
            font_size = 64
            line_height = 1.08
            letter_spacing = -0.5
        elif max_len <= 20:
            font_size = 54
            line_height = 1.10
            letter_spacing = -0.3
        elif max_len <= 28:
            font_size = 46
            line_height = 1.12
            letter_spacing = -0.1
        else:
            font_size = 38
            line_height = 1.15
            letter_spacing = 0.0
    else:
        # 3 or more lines
        if max_len <= 16:
            font_size = 48
            line_height = 1.12
            letter_spacing = -0.2
        elif max_len <= 24:
            font_size = 40
            line_height = 1.15
            letter_spacing = 0.0
        else:
            font_size = 32
            line_height = 1.18
            letter_spacing = 0.0

    return {
        "font_size": font_size,
        "line_height": line_height,
        "letter_spacing": letter_spacing,
    }


def resolve_headline_effect(
    effect: str = "auto",
    headline_text: str = "",
    category: str = "",
    layout_name: str = "",
    palette_is_dark: bool = True,
    accent_color: str = "#FFB300",
    headline_color: str = "",
) -> Tuple[str, str, str]:
    """
    Renders rich CSS typography effects for primary titles:
    1. 'embossed' (In nổi 3D / Chiseled 3D Gold): 24K gold or platinum 3D relief with specular highlight.
    2. 'shadow' (Bóng đổ chiều sâu / Deep Studio Shadow): Multi-layer ambient occlusion + directional drop shadows.
    3. 'led' (Đèn LED Backlit / Tech Halo): Illuminated reverse channel letters with crisp face and ambient halo.
    4. 'neon' (Phát quang Neon / Vibrant Neon Tube): Multi-radius gas-discharge tube light with white core.
    5. 'chrome' (Chrome Bạch Kim Tráng Gương / Liquid Chrome): High-specular horizon reflection metallic finish.
    6. 'engraved' (Khắc Chìm Sa Thạch / Deep Engraved): Inverted light relief debossed into rock or leather.
    7. 'holographic' (Hologram Ánh Kim Xà Cừ / Iridescent Foil): Multi-spectrum diagonal rainbow iridescent luster.
    8. 'outline' (Viền Rỗng Thể Thao Hiện Đại / Ghost Stroke): Bold modern athletic wireframe stroke.
    9. 'auto': Intelligently selects the optimal effect based on text semantics, category, and layout.

    Returns:
        (resolved_effect_name, headline_fill_css, wrap_filter_css)
    """
    EFFECT_ALIASES = {
        # Embossed / 3D Gold
        "embossed": "embossed",
        "in_noi_3d_gold": "embossed",
        "in_noi_3d": "embossed",
        "in_noi": "embossed",
        "3d_gold": "embossed",
        "gold": "embossed",
        "ma_vang": "embossed",
        "vang_kim": "embossed",
        "vang_24k": "embossed",
        "3d_embossed": "embossed",
        "gold_metallic": "embossed",
        "gold_foil": "embossed",
        "relief": "embossed",
        # LED Backlit
        "led": "led",
        "den_led_backlit": "led",
        "den_led": "led",
        "backlit": "led",
        "halo": "led",
        "led_glow": "led",
        "led_light": "led",
        # Neon
        "neon": "neon",
        "neon_glow": "neon",
        "neon_light": "neon",
        "den_neon": "neon",
        "phat_quang_neon": "neon",
        "phat_quang": "neon",
        "phat_sang": "neon",
        "glow": "neon",
        # Chrome
        "chrome": "chrome",
        "chrome_bach_kim": "chrome",
        "chrome_bac": "chrome",
        "bach_kim": "chrome",
        "liquid_chrome": "chrome",
        "silver": "chrome",
        "kim_loai": "chrome",
        # Engraved
        "engraved": "engraved",
        "khac_chim_sa_thach": "engraved",
        "khac_chim": "engraved",
        "khac_da": "engraved",
        "debossed": "engraved",
        # Holographic
        "holographic": "holographic",
        "hologram": "holographic",
        "hologram_xa_cu": "holographic",
        "xa_cu": "holographic",
        "iridescent": "holographic",
        # Outline / Wireframe
        "outline": "outline",
        "vien_rong_the_thao": "outline",
        "vien_rong": "outline",
        "wireframe": "outline",
        "stroke": "outline",
        # Studio Shadow
        "shadow": "shadow",
        "bong_do_studio_shadow": "shadow",
        "bong_do_studio": "shadow",
        "studio_shadow": "shadow",
        "drop_shadow": "shadow",
        "bong_do": "shadow",
        "deep_shadow": "shadow",
    }

    clean_effect = (effect or "auto").strip().lower()
    clean_effect = EFFECT_ALIASES.get(clean_effect, clean_effect)

    if clean_effect == "auto":
        text_lower = (headline_text or "").lower()
        if any(k in text_lower for k in ["neon", "phát quang", "quán bar", "đêm", "night", "cyber", "glow", "edm", "club"]):
            clean_effect = "neon"
        elif any(k in text_lower for k in ["chrome", "kim loại", "siêu xe", "sport", "racing", "tốc độ", "bạch kim", "flagship"]):
            clean_effect = "chrome"
        elif any(k in text_lower for k in ["hologram", "xà cừ", "lấp lánh", "glitter", "kpop", "gen z", "trang sức", "mỹ phẩm", "skincare", "son môi"]):
            clean_effect = "holographic"
        elif any(k in text_lower for k in ["khắc chìm", "sa thạch", "đá cổ", "sử thi", "cổ kính", "rượu vang", "da thật", "vintage"]):
            clean_effect = "engraved"
        elif any(k in text_lower for k in ["streetwear", "thể thao", "fitness", "sneaker", "chạy bộ", "marathon", "gym", "rỗng"]):
            clean_effect = "outline"
        elif any(k in text_lower for k in ["led", "hi-res", "công nghệ", "hybrid", "tai nghe", "audio", "digital", "suv"]):
            clean_effect = "led"
        elif any(k in text_lower for k in ["hoàng gia", "thượng hạng", "vàng", "gold", "trung thu", "tết", "quà tặng", "xa xỉ", "dạ lông cừu", "atelier", "luxury"]):
            clean_effect = "embossed"
        else:
            clean_effect = "shadow"

    if clean_effect == "embossed":
        # 1. 3D In nổi kim loại mạ vàng / bạch kim chiseled relief
        if palette_is_dark:
            fill_css = (
                "background: linear-gradient(180deg, #FFFFFF 0%, #FEE599 28%, #E5B842 62%, #A87612 92%, #7A5308 100%); "
                "-webkit-background-clip: text; "
                "-webkit-text-fill-color: transparent; "
                "text-shadow: none;"
            )
            filter_css = (
                "filter: drop-shadow(0 1px 0 rgba(255, 255, 255, 0.95)) "
                "drop-shadow(0 2.5px 0 #9E6E0F) "
                "drop-shadow(0 4.5px 1px #5C3E04) "
                "drop-shadow(0 10px 22px rgba(0, 0, 0, 0.92));"
            )
        else:
            fill_css = (
                f"color: {headline_color or '#1E293B'}; "
                "text-shadow: 0 1px 0 rgba(255, 255, 255, 0.95), 0 -1px 0 rgba(0, 0, 0, 0.25), 0 3px 6px rgba(0, 0, 0, 0.35);"
            )
            filter_css = "filter: drop-shadow(0 3px 8px rgba(0, 0, 0, 0.22));"

    elif clean_effect == "led":
        # 2. Đèn LED Backlit (Reverse Channel Letters): Mặt chữ sáng rõ nét, ánh sáng hắt lưng (halo-lit) và bóng đổ tách biệt
        if palette_is_dark:
            fill_css = (
                f"color: #FFFFFF; "
                f"text-shadow: 0 0 1px #FFFFFF, 0 0 8px {accent_color}, 0 0 20px {accent_color}, 0 4px 14px rgba(0, 0, 0, 0.95);"
            )
            filter_css = "filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.75));"
        else:
            fill_css = (
                f"color: {headline_color or '#0F172A'}; "
                f"text-shadow: 0 0 8px {accent_color}, 0 0 18px {accent_color}, 0 2px 4px rgba(0, 0, 0, 0.15);"
            )
            filter_css = "filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.20));"

    elif clean_effect == "neon":
        # 3. Phát quang Neon ống thủy tinh: Lõi sáng, viền neon rực rỡ và khoảng cách chữ thoáng
        if palette_is_dark:
            fill_css = (
                f"color: #FFFFFF; "
                f"letter-spacing: 0.03em; "
                f"text-shadow: 0 0 2px #FFFFFF, 0 0 7px {accent_color}, 0 0 18px {accent_color}, 0 0 36px {accent_color}, 0 2px 10px rgba(0, 0, 0, 0.90);"
            )
            filter_css = "filter: drop-shadow(0 3px 8px rgba(0, 0, 0, 0.50));"
        else:
            fill_css = (
                f"color: {accent_color}; "
                f"letter-spacing: 0.03em; "
                f"text-shadow: 0 0 2px {accent_color}, 0 0 8px {accent_color}, 0 2px 6px rgba(0, 0, 0, 0.18);"
            )
            filter_css = "filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.15));"

    elif clean_effect == "chrome":
        # 4. Chrome Bạch Kim Tráng Gương (Liquid Chrome / Cyber Horizon)
        if palette_is_dark:
            fill_css = (
                "background: linear-gradient(180deg, #FFFFFF 0%, #DCE5ED 25%, #7D8E9E 48%, #141C24 51%, #3E4F61 55%, #B8C7D6 80%, #FFFFFF 100%); "
                "-webkit-background-clip: text; "
                "-webkit-text-fill-color: transparent; "
                "text-shadow: none;"
            )
            filter_css = (
                "filter: drop-shadow(0 1px 0 rgba(255, 255, 255, 0.95)) "
                "drop-shadow(0 2.5px 0 #2E3842) "
                "drop-shadow(0 8px 20px rgba(0, 0, 0, 0.92));"
            )
        else:
            fill_css = (
                "background: linear-gradient(180deg, #475569 0%, #1E293B 48%, #0F172A 51%, #334155 70%, #64748B 100%); "
                "-webkit-background-clip: text; "
                "-webkit-text-fill-color: transparent; "
                "text-shadow: none;"
            )
            filter_css = "filter: drop-shadow(0 2px 8px rgba(15, 23, 42, 0.25));"

    elif clean_effect == "engraved":
        # 5. Khắc Chìm Sa Thạch / Da Thật (Deep Engraved / Letterpress)
        if palette_is_dark:
            fill_css = (
                "color: #D4CEBF; "
                "text-shadow: 0 -2.5px 3px rgba(0, 0, 0, 0.95), 0 1.5px 1px rgba(255, 255, 255, 0.40), 0 0 1px rgba(0, 0, 0, 0.90);"
            )
            filter_css = "filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.60));"
        else:
            fill_css = (
                "color: #6B5B4E; "
                "text-shadow: 0 -1.5px 2px rgba(0, 0, 0, 0.50), 0 1.5px 1px rgba(255, 255, 255, 0.90);"
            )
            filter_css = "filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.15));"

    elif clean_effect == "holographic":
        # 6. Hologram Ánh Kim Xà Cừ (Iridescent Holographic Foil)
        if palette_is_dark:
            fill_css = (
                "background: linear-gradient(135deg, #FFFFFF 0%, #E0C3FC 22%, #8EC5FC 45%, #F5D0FE 68%, #BAE6FD 85%, #FED7AA 100%); "
                "-webkit-background-clip: text; "
                "-webkit-text-fill-color: transparent; "
                "text-shadow: none;"
            )
            filter_css = (
                "filter: drop-shadow(0 0 10px rgba(186, 230, 253, 0.65)) "
                "drop-shadow(0 4px 16px rgba(0, 0, 0, 0.88));"
            )
        else:
            fill_css = (
                "background: linear-gradient(135deg, #7C3AED 0%, #2563EB 30%, #DB2777 60%, #EA580C 100%); "
                "-webkit-background-clip: text; "
                "-webkit-text-fill-color: transparent; "
                "text-shadow: none;"
            )
            filter_css = "filter: drop-shadow(0 2px 8px rgba(124, 58, 237, 0.25));"

    elif clean_effect == "outline":
        # 7. Viền Rỗng Thể Thao Hiện Đại (Ghost Wireframe Stroke)
        if palette_is_dark:
            fill_css = (
                "color: transparent; "
                "-webkit-text-stroke: 2.2px #FFFFFF; "
                "letter-spacing: 0.04em; "
                "text-shadow: 0 4px 14px rgba(0, 0, 0, 0.85);"
            )
            filter_css = "filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.60));"
        else:
            fill_css = (
                f"color: transparent; "
                f"-webkit-text-stroke: 2.2px {headline_color or '#0F172A'}; "
                "letter-spacing: 0.04em; "
                "text-shadow: 0 2px 6px rgba(15, 23, 42, 0.20);"
            )
            filter_css = "filter: drop-shadow(0 1px 4px rgba(0, 0, 0, 0.15));"

    else:
        # 8. 'shadow' (Mặc định): Bóng đổ chiều sâu studio cao cấp
        if palette_is_dark:
            fill_css = (
                f"color: {headline_color or '#FFFFFF'}; "
                "text-shadow: 0 2px 4px rgba(0, 0, 0, 0.95), 0 6px 16px rgba(0, 0, 0, 0.85), 0 16px 32px rgba(0, 0, 0, 0.75), 0 28px 60px rgba(0, 0, 0, 0.65);"
            )
            filter_css = "filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.70));"
        else:
            fill_css = (
                f"color: {headline_color or '#0F172A'}; "
                "text-shadow: 0 1px 2px rgba(255, 255, 255, 0.85), 0 4px 12px rgba(15, 23, 42, 0.28), 0 12px 28px rgba(15, 23, 42, 0.18);"
            )
            filter_css = "filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.15));"

    return clean_effect, fill_css, filter_css


# ==============================================================================
# PHẦN 2: ĐO ĐẠC FONT METRICS BẰNG PIL (font-fitting cho render/mask offline)
# ==============================================================================

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
        except Exception as e:
            # Silent before 2026-09-12: falling through to PIL's non-Unicode bitmap
            # default font with zero warning means the offline geometry estimate
            # (wrap_and_measure/fit_font_size_px) silently measures Vietnamese text
            # against the WRONG glyph shapes -- same "lose Vietnamese diacritics with no
            # trace" bug class this project has hit before (glyph_engine.py, before the
            # Omni-Block rewrite).
            logger.warning(f"Failed to load font '{font_path}' at size {size} ({e}); falling back to PIL's non-Unicode default font -- Vietnamese diacritics will not measure/render correctly in this estimate.")
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def wrap_and_measure(
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width_px: float,
    display_text: Optional[str] = None,
) -> Tuple[float, float, List[str]]:
    """
    Gói từ tham lam (Greedy word-wrap) cho `text` vừa vặn trong `max_width_px`.
    Trả về (chiều rộng dòng dài nhất px, tổng chiều cao khối px, danh sách các dòng).

    TẠI SAO CẦN LÀM:
    - Hỗ trợ cả ngắt dòng cứng (explicit newlines `\\n`) lẫn tự động bẻ dòng khi vượt quá max_width_px.
    - Tính toán chính xác ascent + descent kết hợp hệ số giãn dòng LINE_HEIGHT_MULT = 1.25.

    `display_text` (tùy chọn, 2026-09-13): khi `text` đã bị `.upper()` hoá để đo (xem
    `fit_font_size_px`'s `is_uppercase`), `display_text` là bản GỐC giữ nguyên hoa/thường. Ranh
    giới ngắt dòng vẫn được quyết định theo bbox đo trên `text` (khớp đúng CSS
    `text-transform: uppercase` sẽ áp dụng lúc render), nhưng chuỗi trả về trong `lines` lấy từ
    `display_text` theo đúng vị trí từ tương ứng -- để nội dung HTML thực sự giữ đúng chữ hoa/thường
    gốc (chỉ hiển thị hoa qua CSS), không bị ép hoa cứng vào chính chuỗi text. `str.upper()` bảo
    toàn số từ/thứ tự nên ánh xạ theo chỉ số luôn khớp 1-1.
    """
    paragraphs = text.splitlines()
    if not paragraphs:
        return 0.0, 0.0, []
    disp_paragraphs = (display_text or text).splitlines()

    lines: List[str] = []
    measure_lines: List[str] = []
    for para, disp_para in zip(paragraphs, disp_paragraphs):
        words = para.split()
        disp_words = disp_para.split()
        if not words:
            continue
        current = ""
        disp_current = ""
        for word, disp_word in zip(words, disp_words):
            candidate = f"{current} {word}".strip()
            bbox = font.getbbox(candidate)
            candidate_w = bbox[2] - bbox[0]
            disp_candidate = f"{disp_current} {disp_word}".strip()
            if candidate_w <= max_width_px or not current:
                current, disp_current = candidate, disp_candidate
            else:
                lines.append(disp_current)
                measure_lines.append(current)
                current, disp_current = word, disp_word
        if current:
            lines.append(disp_current)
            measure_lines.append(current)

    if not lines:
        return 0.0, 0.0, []

    max_line_w = max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in measure_lines)
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
    allow_wrap: bool = True,
    max_lines: Optional[int] = None,
) -> Tuple[int, float, float, List[str]]:
    """
    Thu nhỏ `base_font_size` (px) từng bước cho đến khi `text` gói từ vừa vặn
    cả 2 chiều: `max_width_px` VÀ `max_height_px`.

    TẠI SAO BẮT BUỘC ĐO ĐẠC TRÊN CHỮ IN HOA (`text.upper()`):
    - Nếu `is_uppercase` là True, toàn bộ chuỗi được chuyển thành hoa trước khi đo.
    - Điều này đảm bảo tính toán đủ không gian cho các ký tự lớn, triệt tiêu hoàn toàn
      lỗi cắt mép chữ (clipping) khi trình duyệt áp dụng CSS `text-transform: uppercase`.

    TẠI SAO TRẢ THÊM `lines` (2026-09-13 fix):
    - Trước đây hàm chỉ trả `(font_size, w, h)`, bỏ luôn danh sách `lines` mà
      `wrap_and_measure()` đã tính ra để đo `h`. Nghĩa là font_size được chọn "như thể" chữ đã
      xuống nhiều dòng, nhưng caller không có cách nào lấy lại đúng các dòng đó để chèn `\\n`
      thật vào HTML -- text.length dài (quote, mô tả) vẫn render thành 1 chuỗi phẳng, bị CSS
      `white-space: nowrap` khoá cứng ép về 1 dòng, cực nhỏ, khó đọc. Trả thêm `lines` để mọi
      caller (không chỉ nhánh pure-typography) có thể tự chèn xuống dòng đúng theo đúng phép đo.

    `allow_wrap=False` (2026-09-13, cho pill/button/badge -- xem geometry.py's
    `compute_block_metrics`): ép đo `text` như 1 dòng DUY NHẤT tuyệt đối (truyền `max_width_px`
    cực lớn xuống `wrap_and_measure` để thuật toán tham lam không bao giờ ngắt dòng), rồi so
    khớp `w` đo được với `max_width_px` THẬT để quyết định co font -- khớp đúng ý đồ "chip 1
    dòng": co dần tới khi cả chuỗi vừa 1 dòng, không co lố vì ngộ nhận có chỗ xuống nhiều dòng.

    `max_lines` (2026-09-13, cho `card` ở zone hẹp -- vd `middle_left` bị Product Sanctuary ăn
    bớt bề rộng): thuật toán vốn LUÔN chọn font_size LỚN NHẤT thoả `max_width_px`/`max_height_px`
    -- với 1 cột hẹp, font lớn nhất vẫn hợp lệ (nhờ `max_height_px` rộng rãi của cả zone cell)
    lại chỉ vừa 1-2 từ/dòng, tạo 1 cột dài lởm chởm từng dòng lẻ, xấu và khó đọc hơn hẳn so với
    việc co font nhỏ hơn để mỗi dòng chứa nhiều từ hơn. Khi đặt `max_lines`, `len(lines) >
    max_lines` cũng bị coi là "chưa vừa" (giống điều kiện `h`/`w`), buộc tiếp tục co cho tới khi
    số dòng nằm trong giới hạn đọc được.
    """
    measure_text = text.upper() if is_uppercase else text
    wrap_width_px = max_width_px if allow_wrap else 10**7

    font_size = base_font_size
    while font_size > min_font_size:
        font = _safe_load_font(font_path, font_size)
        w, h, lines = wrap_and_measure(measure_text, font, wrap_width_px, display_text=text)
        if h <= max_height_px and w <= max_width_px and (max_lines is None or len(lines) <= max_lines):
            return font_size, w, h, lines
        font_size -= step

    font = _safe_load_font(font_path, min_font_size)
    w, h, lines = wrap_and_measure(measure_text, font, wrap_width_px, display_text=text)
    return min_font_size, w, h, lines


__all__ = [
    "EMOJI_PATTERN",
    "LINE_HEIGHT_MULT",
    "VIETNAMESE_COMPOUND_WORDS",
    "balance_vietnamese_headline",
    "compute_font_ladder",
    "fit_font_size_px",
    "is_compound_pair",
    "normalize_text",
    "resolve_headline_effect",
    "strip_emojis",
    "wrap_and_measure",
]
