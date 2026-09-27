"""Probe VA CHẠM DẤU tiếng Việt (GĐ 7, ROADMAP §10.11): với MỌI font × câu mẫu nhiều dấu chồng (Ấ Ầ Ẩ Ẫ Ậ, Ự...),
ở ĐÚNG sàn line-height hệ thống dùng cho font đó (fonts.MIN_STACK_LINE_HEIGHT; mặc định 1.05), đo ĐIỂM ẢNH thật (Chromium -- AGENTS §4.4):
chụp riêng từng dòng tại đúng vị trí của nó trong khối 2 dòng; mực của dòng 2 không được chồng lên mực dòng 1."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from tendoo_v3.fonts import DEFAULT_STACK_LINE_HEIGHT, FONT_CATALOG, MIN_STACK_LINE_HEIGHT, resolve_font

SAMPLES = [("ẤN TƯỢNG", "QUẢ NGỌT"), ("HỘI NGHỊ", "ẢNH ĐẸP"), ("NGÀN ƯU ĐÃI", "TỰ HÀO")]
KNOWN_TIGHT: set = set()


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser.new_page(viewport={"width": 1400, "height": 600})
        browser.close()


def _ink(page, hide: str) -> np.ndarray:
    page.evaluate(f"""() => {{ for (const s of document.querySelectorAll('#b span')) s.style.visibility = 'visible';
                              if ('{hide}') document.getElementById('{hide}').style.visibility = 'hidden'; }}""")
    img = np.asarray(Image.open(io.BytesIO(page.screenshot(clip={"x": 0, "y": 0, "width": 1400, "height": 600}))).convert("L"))
    return img < 128


@pytest.mark.parametrize("font", sorted(FONT_CATALOG))
def test_stacked_diacritics_do_not_touch_line_above(page, font):
    _, face_css, family = resolve_font(font_key=font)
    # Sàn line-height hệ thống dùng cho font này (hero_phrase.css --hero-min-lh). Hạ sàn trong fonts.py -> test đỏ.
    lh = MIN_STACK_LINE_HEIGHT.get(font, DEFAULT_STACK_LINE_HEIGHT)
    bad = []
    for l1, l2 in SAMPLES:
        page.set_content(f"""<html><head><style>{face_css}
          body {{ margin: 0; background: #fff; }}
          #b {{ font-family: {family}; font-weight: 900; font-size: 110px; line-height: {lh}; color: #000;
                padding: 60px 40px; white-space: nowrap; }}
        </style></head><body><div id="b"><span id="a">{l1}</span><br><span id="c">{l2}</span></div></body></html>""", wait_until="load")
        page.evaluate("document.fonts.ready")
        top = _ink(page, "c")   # chỉ dòng 1
        low = _ink(page, "a")   # chỉ dòng 2
        overlap = int((top & low).sum())
        if overlap > 0:
            ys = np.where((top & low).any(axis=1))[0]
            bad.append(f"{l1}/{l2}: dấu dòng 2 chồng nét dòng 1 ({overlap}px, y {ys.min()}-{ys.max()})")
    if font in KNOWN_TIGHT:
        pytest.xfail(f"{font}: đã biết sát dấu -- {bad}")
    assert not bad, f"{font}: {bad}"
