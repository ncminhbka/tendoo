"""GĐ 7a (R2) -- ngắt dòng theo nghĩa: khoảng trắng không ngắt chỉ đổi LOẠI khoảng trắng, không đổi chữ."""

from __future__ import annotations

from tendoo_v3.hero_markup import NBSP, bind_nonbreaking
from tendoo_v3.schema import _normalize_for_verbatim_check


def test_binds_compounds_and_number_units():
    assert bind_nonbreaking("TRÀ SỮA TENDOO") == f"TRÀ{NBSP}SỮA TENDOO"
    assert bind_nonbreaking("CHỈ TỪ 12 TRIỆU") == f"CHỈ TỪ 12{NBSP}TRIỆU"
    assert bind_nonbreaking("Cà phê, trà sữa") == f"Cà{NBSP}phê, trà{NBSP}sữa"


def test_hero_mode_keeps_only_number_units():
    """Tiêu đề cột hẹp: nối cứng từ ghép làm autofit thu nhỏ cả hero (đo R2) -> chỉ số + đơn vị."""
    assert bind_nonbreaking("CHUYÊN VIÊN 3 BƯỚC", compounds=False) == f"CHUYÊN VIÊN 3{NBSP}BƯỚC"


def test_never_changes_text_verbatim():
    for s in ("KHAI TRƯƠNG CỬA HÀNG MỚI GIẢM 50 %", "Dịch vụ tuyệt vời, khách hàng hài lòng.", "", None):
        out = bind_nonbreaking(s)
        if s:
            assert _normalize_for_verbatim_check(out) == _normalize_for_verbatim_check(s)
        else:
            assert out == s
