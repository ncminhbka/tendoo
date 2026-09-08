"""
tests/test_text_engine.py

Unit tests for text_engine and icon vectorization:
- Validates strip_emojis removes emojis, Dingbats, variation selectors without tofu boxes.
- Validates 100% preservation of Vietnamese diacritics, currency marks, numbers, and punctuation.
- Validates normalize_text sanitizes copy-pasted emojis from user inputs.
- Validates vector SVG icon constants in base.py.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tendoo.layouts.text_engine import strip_emojis, normalize_text, EMOJI_PATTERN
from tendoo.layouts.base import (
    CALENDAR_ICON_SVG,
    PHONE_ICON_SVG,
    LOCATION_ICON_SVG,
    GLOBE_ICON_SVG,
    GIFT_ICON_SVG,
    TAG_ICON_SVG,
    STAR_ICON_SVG,
    CLOCK_ICON_SVG,
    CHECK_ICON_SVG,
    ARROW_RIGHT_ICON_SVG,
)


class TestStripEmojis:
    def test_strip_common_emojis(self):
        sample = "🔥 SIÊU GIẢM GIÁ 50% 🔥"
        assert strip_emojis(sample) == "SIÊU GIẢM GIÁ 50%"

    def test_strip_contact_emojis(self):
        sample = "📞 0988 123 456 - 📍 128 Trần Duy Hưng"
        assert strip_emojis(sample) == "0988 123 456 - 128 Trần Duy Hưng"

    def test_strip_arrows_and_symbols(self):
        sample = "NHẬN VOUCHER ➔ NGAY HÔM NAY 🎁"
        assert strip_emojis(sample) == "NHẬN VOUCHER NGAY HÔM NAY"

    def test_strip_food_and_objects(self):
        sample = "🍔 THE BURGER CRAFT 🍕"
        assert strip_emojis(sample) == "THE BURGER CRAFT"

    def test_preserve_vietnamese_diacritics(self):
        vietnamese_text = (
            "Trăm năm trong cõi người ta, chữ tài chữ mệnh khéo là ghét nhau. "
            "Trải qua một cuộc bể dâu, những điều trông thấy mà đau đớn lòng. "
            "Á À Ả Ã Ạ Ă Ắ Ằ Ẳ Ẵ Ặ Â Ấ Ầ Ẩ Ẫ Ậ É È Ẻ Ẽ Ẹ Ê Ế Ề Ể Ễ Ệ "
            "Í Ì Ỉ Ĩ Ị Ó Ò Ỏ Õ Ọ Ô Ố Ồ Ổ Ỗ Ộ Ơ ỚỜ Ở Ỡ Ợ Ú Ù Ủ Ũ Ụ Ư Ứ Ừ Ử Ữ Ự "
            "Ý Ỳ Ỷ Ỹ Ỵ Đ đ"
        )
        assert strip_emojis(vietnamese_text) == vietnamese_text

    def test_preserve_currency_and_numbers(self):
        pricing = "Giá chỉ từ 1.290.000đ - 2.500.000₫ (Tiết kiệm 35%)"
        assert strip_emojis(pricing) == pricing

    def test_preserve_quotes_and_punctuation(self):
        text = '“Sản phẩm tuyệt vời!” • Bảo hành 24 tháng — Miễn phí vận chuyển.'
        assert strip_emojis(text) == text


class TestNormalizeText:
    def test_normalize_strips_emojis_and_normalizes_spaces(self):
        raw = "   🔥   ĐẠI TIỆC   ÂM NHẠC   2026   🎉   "
        expected = "ĐẠI TIỆC ÂM NHẠC 2026"
        assert normalize_text(raw) == expected

    def test_normalize_multiline_strips_emojis(self):
        raw = "🔥 KHUYẾN MÃI LỚN\n📞 LIÊN HỆ NGAY\n📍 HÀ NỘI"
        expected = "KHUYẾN MÃI LỚN\nLIÊN HỆ NGAY\nHÀ NỘI"
        assert normalize_text(raw) == expected


class TestVectorSVGIcons:
    @pytest.mark.parametrize("svg_str,icon_name", [
        (CALENDAR_ICON_SVG, "calendar"),
        (PHONE_ICON_SVG, "phone"),
        (LOCATION_ICON_SVG, "location"),
        (GLOBE_ICON_SVG, "globe"),
        (GIFT_ICON_SVG, "gift"),
        (TAG_ICON_SVG, "tag"),
        (STAR_ICON_SVG, "star"),
        (CLOCK_ICON_SVG, "clock"),
        (CHECK_ICON_SVG, "check"),
        (ARROW_RIGHT_ICON_SVG, "arrow-right"),
    ])
    def test_svg_structure(self, svg_str, icon_name):
        assert svg_str.startswith("<svg"), f"{icon_name} should start with <svg"
        assert svg_str.endswith("</svg>"), f"{icon_name} should end with </svg>"
        assert 'viewBox="0 0 24 24"' in svg_str or 'viewBox="0 0 ' in svg_str
        assert not EMOJI_PATTERN.search(svg_str), f"{icon_name} must not contain raw unicode emojis"
