"""Chữ tô gradient (3d_gold/chrome/fire/hologram): Bước 5 KHÔNG được thêm quầng (text-shadow) lên nó -- Chromium vẽ
bóng ĐÈ lên nền cắt theo chữ, cả tiêu đề thành vệt đen (27/09, thiệp VIP GPT thật, người duyệt: "chữ đen thui").
Thiếu tương phản -> làm phẳng về một màu đặc. Squint C3 từng bỏ sót vì lấy lần đo tốt hơn giữa có/không bóng."""

from __future__ import annotations

import pytest

from tendoo_v3.renderer import build_template_html
from tendoo_v3.schema import StyleConfig, TendooCreativePlan

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_template_test import generate_harsh_backdrop_data_uri  # noqa: E402

CASES = [
    ("luxury_centered_card", "3d_gold", "#D4AF37", "dark_luxury"),
    ("sandwich_top_heavy", "chrome", "#94A3B8", "light_clean"),
    ("split_left", "hologram", "#38BDF8", "dark_luxury"),
    ("diagonal_slash", "fire", "#F97316", "vibrant"),
]


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser.new_page()
        browser.close()


@pytest.mark.parametrize("template,effect,theme,tone", CASES)
def test_gradient_text_never_gets_a_shadow_over_its_fill(page, template, effect, theme, tone):
    plan = TendooCreativePlan(template=template, hero="ĐÊM TRI ÂN KHÁCH HÀNG VIP", subhead="Tối thứ bảy 18/10 · Khách sạn Majestic",
                              style=StyleConfig(font="playfair", theme_color=theme, text_effect=effect, background_tone=tone))
    for seed in ("a", "b", "c"):
        bg = generate_harsh_backdrop_data_uri(1024, 1024, theme, tone, seed=f"{template}-{seed}")
        page.set_viewport_size({"width": 1024, "height": 1024})
        page.set_content(build_template_html(plan, bg, 1024, 1024), wait_until="load")
        page.wait_for_function("window.__tendooAutofitDone === true", timeout=8000)
        bad = page.evaluate("""() => Array.from(document.querySelectorAll('[data-autofit], [data-autofit] *')).filter(n => {
            const cs = getComputedStyle(n);
            const clipText = (cs.webkitBackgroundClip || cs.backgroundClip) === 'text' && parseFloat((cs.webkitTextFillColor.match(/[\\d.]+/g) || [0,0,0,1])[3] ?? 1) < 0.5;
            return clipText && (n.getAttribute('style') || '').includes('text-shadow');
        }).map(n => n.className)""")
        assert not bad, f"{template}/{effect}/{seed}: quầng đè lên chữ gradient ở {bad}"
