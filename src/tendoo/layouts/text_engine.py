"""
Vietnamese Typography & Adaptive Text Balancing Engine.

Features:
1. Semantic Compound Word Protection: Preserves Vietnamese compounds (e.g. 'CHỐNG ỒN', 'CÔNG NGHỆ',
   'THANH MÁT', 'ĐẶC BIỆT') from being broken across lines.
2. Orphan Word Prevention: Guarantees line 2 doesn't end with a lone orphan word.
3. Adaptive Font Sizing Ladder: Responsive type scale tailored for 1024x1024 canvas.
4. Robust Newline Normalization.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


# Common commercial & technical Vietnamese 2-word compounds that MUST NOT be split across lines
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


def normalize_text(text: str) -> str:
    """Cleans up literal escapes ('\\n', '\\N', '\\r\\n') and strips trailing whitespace."""
    if not text:
        return ""
    cleaned = (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\r\n", "\n")
        .strip()
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def is_compound_pair(w1: str, w2: str) -> bool:
    """Checks if two consecutive words form a protected compound in lowercase."""
    pair = f"{w1.strip().lower()} {w2.strip().lower()}"
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
        lines = [l.strip() for l in clean.split("\n") if l.strip()]
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

    max_len = max(len(l) for l in lines)
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
    Renders rich CSS typography effects for hero titles:
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
        "3d_gold": "embossed",
        "gold": "embossed",
        "relief": "embossed",
        # LED Backlit
        "led": "led",
        "den_led_backlit": "led",
        "den_led": "led",
        "backlit": "led",
        "halo": "led",
        # Neon
        "neon": "neon",
        "phat_quang_neon": "neon",
        "phat_quang": "neon",
        "glow": "neon",
        # Chrome
        "chrome": "chrome",
        "chrome_bach_kim": "chrome",
        "bach_kim": "chrome",
        "liquid_chrome": "chrome",
        "silver": "chrome",
        # Engraved
        "engraved": "engraved",
        "khac_chim_sa_thach": "engraved",
        "khac_chim": "engraved",
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
        elif layout_name == "center_hourglass":
            clean_effect = "embossed"
        elif layout_name == "bottom_platform":
            clean_effect = "led" if ("hybrid" in text_lower or "công nghệ" in text_lower) else "shadow"
        elif layout_name == "split_column":
            clean_effect = "embossed"
        elif layout_name == "diagonal_slash":
            clean_effect = "outline" if ("sport" in text_lower or "thể thao" in text_lower) else "chrome"
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
        fill_css = (
            f"color: #FFFFFF; "
            f"text-shadow: 0 0 1px #FFFFFF, 0 0 8px {accent_color}, 0 0 20px {accent_color}, 0 4px 14px rgba(0, 0, 0, 0.95);"
        )
        filter_css = "filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.75));"

    elif clean_effect == "neon":
        # 3. Phát quang Neon ống thủy tinh: Lõi trắng sáng, viền neon rực rỡ và khoảng cách chữ thoáng (không bị dính nét)
        fill_css = (
            f"color: #FFFFFF; "
            f"letter-spacing: 0.03em; "
            f"text-shadow: 0 0 2px #FFFFFF, 0 0 7px {accent_color}, 0 0 18px {accent_color}, 0 0 36px {accent_color}, 0 2px 10px rgba(0, 0, 0, 0.90);"
        )
        filter_css = "filter: drop-shadow(0 3px 8px rgba(0, 0, 0, 0.50));"

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

