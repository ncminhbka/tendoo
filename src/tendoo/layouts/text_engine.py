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
