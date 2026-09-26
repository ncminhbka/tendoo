"""Cổng 2 -- Thành viên (ROADMAP §2.5): phát hiện lựa chọn ngoài danh mục và nội dung sẽ mất."""

from __future__ import annotations

from dataclasses import replace

from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.validators import check_plan


def _plan(**kw) -> TendooCreativePlan:
    base = dict(template="sandwich_top_heavy", hero="GIẢM 25%", style=StyleConfig(font="bevietnam", text_effect="3d_gold", background_tone="dark_luxury"))
    base.update(kw)
    return TendooCreativePlan(**base)


def test_valid_plan_passes():
    assert check_plan(_plan(subhead="Toàn bộ menu", cta="ĐẶT NGAY", qr_code="https://x")) == []


def test_aliases_are_members():
    assert check_plan(_plan(style=StyleConfig(font="bevietnam", text_effect="bold_clean", background_tone="pastel"))) == []


def test_unknown_template_flagged():
    assert any("không có trong catalog" in i for i in check_plan(_plan(template="khong_ton_tai")))


def test_unknown_style_values_flagged():
    issues = check_plan(_plan(style=StyleConfig(font="cinzel", text_effect="gold_3d", background_tone="pastel_soft")))
    assert any("font 'cinzel'" in i for i in issues)
    assert any("text_effect 'gold_3d'" in i for i in issues)
    assert any("background_tone 'pastel_soft'" in i for i in issues)


def test_field_without_slot_reported_as_lost():
    # l_frame_showcase không có chỗ cho CTA trong template.html -> CTA sẽ mất khỏi poster.
    issues = check_plan(_plan(template="l_frame_showcase", cta="ĐẶT LỊCH LÁI THỬ"))
    assert any("KHÔNG hiển thị ['cta']" in i for i in issues)


def test_missing_required_field_reported():
    issues = check_plan(_plan(template="customer_feedback_card", testimonial="Rất hài lòng"))
    assert any("thiếu field chuyên biệt ['reviewer_name']" in i for i in issues)


def test_check_does_not_mutate_plan():
    p = _plan(template="l_frame_showcase", cta="X", style=StyleConfig(font="cinzel", text_effect="gold_3d", background_tone="pastel_soft"))
    before = replace(p)
    check_plan(p)
    assert p == before
