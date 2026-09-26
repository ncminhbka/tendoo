"""Cổng 3 -- dung lượng (ROADMAP §2.4-2.5, GĐ 3): Python phủ quyết template LLM đề xuất bằng sức
chứa ĐO ĐƯỢC, không bao giờ đổi sang template làm mất chữ hay khác loại nội dung."""

from __future__ import annotations

import pytest

from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.routing import capacity, nearest_aspect, route_template
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.validators import content_chars

LONG = ["Miễn phí giao hàng nội thành cho mọi đơn hàng trong tháng", "Tặng quà cho 100 khách hàng đầu tiên mỗi ngày",
        "Tích điểm đổi quà hấp dẫn cho thành viên thân thiết", "Bảo hành chính hãng 12 tháng trên toàn quốc",
        "Trả góp lãi suất 0% qua thẻ tín dụng", "Đổi trả miễn phí trong 7 ngày nếu lỗi nhà sản xuất"]
# Dòng dài ~110 ký tự: chỉ 6 dòng đầu được đếm (LIST_LIMITS), nên vượt sức chứa phải nhờ độ dài.
LONG = [x + ", áp dụng tại toàn bộ hệ thống cửa hàng và kênh trực tuyến chính thức" for x in LONG]


def _plan(template, n_extra=0, **kw):
    base = dict(template=template, hero="SIÊU SALE 50%", subhead="Áp dụng cho toàn bộ hệ thống cửa hàng trên toàn quốc tới hết tháng 9",
                extra_texts=(LONG * 3)[:n_extra], style=StyleConfig(font="bevietnam"))
    return TendooCreativePlan(**{**base, **kw})


@pytest.mark.parametrize("given,expected", [("1:1", "1:1"), ("3:4", "4:5"), ("2:3", "9:16"), ("21:9", "16:9"), ("rác", "1:1")])
def test_nearest_aspect(given, expected):
    assert nearest_aspect(given) == expected


def test_plan_within_capacity_is_untouched():
    plan = _plan("lifestyle_corner_pod", n_extra=1)
    assert route_template(plan, "1:1") == (plan, None)


def test_overflowing_plan_moves_to_bigger_template_same_intent_no_text_lost():
    plan = _plan("lifestyle_corner_pod", n_extra=5, orientation="top_left", cta="MUA NGAY", store_info="Hotline: 1900 8888")
    chars = content_chars(plan)
    assert chars > capacity("lifestyle_corner_pod", "1:1", "capacity_chars_safe")
    routed, why = route_template(plan, "1:1")
    assert routed.template != "lifestyle_corner_pod" and "đổi sang" in why
    info = TEMPLATE_CATALOG[routed.template]
    assert not info.get("specialized") and "product_showcase" in info["visual_intents"]
    assert capacity(routed.template, "1:1", "capacity_chars_safe") >= chars
    assert all(f in info["slots"] for f in ("hero", "subhead", "extra_texts", "cta", "store_info"))
    assert routed.orientation is None and routed.extra_texts == plan.extra_texts  # nội dung giữ nguyên văn


def test_specialized_template_is_never_left_or_entered():
    """recruitment_board và menu_price_board cùng intent matrix_board, nhưng tuyển dụng -> menu là sai loại."""
    plan = _plan("recruitment_board", n_extra=6)
    routed, why = route_template(plan, "9:16")
    assert routed is plan and "chuyên biệt" in why
    for _ in range(3):
        routed, _ = route_template(_plan("split_left", n_extra=6), "16:9")
        assert not TEMPLATE_CATALOG[routed.template].get("specialized")


def test_no_candidate_keeps_plan_and_reports_gap():
    plan = _plan("grand_opening_banner", n_extra=6)  # festive_event: chỉ 1 template phục vụ
    routed, why = route_template(plan, "1:1")
    assert routed is plan and "LỖ HỔNG" in why


def test_gate3_runs_on_every_planner_branch():
    """Planner dự phòng (không LLM) cũng qua Cổng 3 -- gắn ở `_pack`, điểm ra duy nhất."""
    from tendoo_v3.llm_planner import generate_creative_plan

    form = {"title": "Siêu sale 50%", "highlights": "\n".join(LONG * 2), "product_desc": LONG[0]}
    _, trace = generate_creative_plan(form, prompt="chữ gom ở góc trên trái", aspect_ratio="1:1", return_debug=True)
    assert "gate3" in trace
