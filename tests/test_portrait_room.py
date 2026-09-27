"""Khung dọc (renderer._apply_portrait_room, ROADMAP §4.7): trần tiêu đề + nội dung chính nới theo sqrt(cao/rộng), subhead
KHÔNG nới (trần thứ bậc); trần cũ của tiêu đề truyền cho autofit (Bước 1a: không giữ cỡ lớn nếu thêm dòng). Biến window
luôn được đặt lại (Playwright set_content giữ window giữa các lần dựng)."""

from __future__ import annotations

import math

from tendoo_v3.renderer import _apply_portrait_room, _portrait_script
from tendoo_v3.schema import TendooCreativePlan


def _budget():
    return {"hero": {"max_h": 200, "min_font": 30, "max_font": 60},
            "subhead": {"max_h": 80, "min_font": 20, "max_font": 24},
            "menu_list": {"max_h": 220, "min_font": 14, "max_font": 23}}


def test_portrait_grows_hero_and_content_not_subhead():
    b = _apply_portrait_room(_budget(), 576, 1024)
    k = math.sqrt(1024 / 576)
    # tiêu đề chỉ nới tới mức "đủ to" của C6 (10% cạnh sqrt(576*1024) = 76.8px), không tới 60 x 1.33 = 80
    assert b["hero"]["max_font"] == 76.5 and b["hero"]["base_max_font"] == 60
    big = _apply_portrait_room({"hero": {"max_h": 300, "min_font": 30, "max_font": 90}}, 576, 1024)
    assert big["hero"]["max_font"] == 90  # đã trên mức đủ to -> giữ nguyên
    assert b["menu_list"]["max_font"] == math.floor(23 * k * 2) / 2
    assert b["subhead"]["max_font"] == 24
    assert _apply_portrait_room(_budget(), 1024, 1024) == _budget()  # vuông / ngang: không đổi


def test_portrait_script_always_resets_window_state():
    menu = TendooCreativePlan(template="menu_price_board", hero="MENU TRÀ SỮA")
    square = _portrait_script("menu_price_board", menu, 1024, 1024, _budget())
    assert "__tendooHeroBaseMax = null" in square and "__tendooContentMaxRatio = null" in square
    tall = _portrait_script("menu_price_board", menu, 576, 1024, _apply_portrait_room(_budget(), 576, 1024))
    assert "__tendooHeroBaseMax = 60;" in tall and "__tendooContentMaxRatio = 2.0;" in tall
