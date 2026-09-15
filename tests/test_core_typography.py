"""
tests/test_core_typography.py

Covers src/tendoo/core/typography.py (merged 2026-09-13 from the former
`layouts/text_engine.py` + the pre-existing PIL font-fitting half, see that module's
docstring):
- Validates strip_emojis removes emojis, Dingbats, variation selectors without tofu boxes.
- Validates 100% preservation of Vietnamese diacritics, currency marks, numbers, and punctuation.
- Validates normalize_text sanitizes copy-pasted emojis from user inputs.
- Validates Vietnamese headline line-balancing and PIL-based font-fitting (no clipping on
  uppercase text).
- Validates vector SVG icon constants in core/base.py.
"""

import pytest

from tendoo.core.base import (
    ARROW_RIGHT_ICON_SVG,
    CALENDAR_ICON_SVG,
    CHECK_ICON_SVG,
    CLOCK_ICON_SVG,
    GIFT_ICON_SVG,
    GLOBE_ICON_SVG,
    LOCATION_ICON_SVG,
    PHONE_ICON_SVG,
    STAR_ICON_SVG,
    TAG_ICON_SVG,
)
from tendoo.core.fonts import FONT_CATALOG, FONTS_DIR
from tendoo.core.typography import (
    EMOJI_PATTERN,
    balance_vietnamese_headline,
    fit_font_size_px,
    normalize_text,
    strip_emojis,
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


def test_uppercase_font_fitting_no_clipping():
    """(Moved 2026-09-13 from tests/test_omni_engine.py -- fits with the other
    typography-measurement coverage in this file.)"""
    font_path = str(FONTS_DIR / FONT_CATALOG['bevietnam']['file'])
    fsize, w, h, _lines = fit_font_size_px(
        text='Private Coaching Transformation',
        font_path=font_path,
        base_font_size=74,
        max_width_px=430.0,
        max_height_px=200.0,
        is_uppercase=True,
    )
    assert w <= 430.0
    assert fsize <= 42


def test_headline_line_balancing():
    """(Moved 2026-09-13 from tests/test_omni_engine.py.)"""
    lines, metrics = balance_vietnamese_headline('Tai Nghe Sonic Pro')
    assert len(lines) == 2
    assert lines[0] == 'Tai Nghe' and lines[1] == 'Sonic Pro'
    assert metrics['font_size'] >= 50


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
