"""GĐ 10 (ROADMAP §10.9) -- dùng được thật: biến thể typography không chạy lại diffusion, logo brand kit, xuất in."""

from __future__ import annotations

import base64

import pytest
from PIL import Image

from tendoo_v3.geometry import get_zones
from tendoo_v3.renderer import build_template_html, compute_geometry_flags, export_print
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.variants import generate_variants

BG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
LOGO = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="100"><rect width="300" height="100" fill="#111"/></svg>').decode()


def _plan(**kw):
    base = dict(template="sandwich_top_heavy", hero="GIẢM 30% TOÀN BỘ MENU", subhead="Áp dụng đến hết chủ nhật", cta="ĐẶT NGAY",
                hero_parts=[{"t": "GIẢM", "role": "prefix"}, {"t": "30%", "role": "stat"}, {"t": "TOÀN BỘ MENU", "role": "suffix"}],
                visual_intent="big_number_deal", style=StyleConfig(font="anton", theme_color="#E11D48"))
    return TendooCreativePlan(**{**base, **kw})


def test_variants_keep_background_contract():
    """Biến thể giữ template/orientation/nội dung (-> cùng vùng chữ = cùng mask nền đã sinh), khác kiểu chữ."""
    p = _plan()
    vs = generate_variants(p, 4)
    assert len(vs) == 4 and vs[0] is p
    zones0 = get_zones(p.template, 1024, 1024, orientation=p.orientation, **compute_geometry_flags(p))
    keys = set()
    for v in vs:
        assert (v.template, v.orientation, v.hero, v.hero_parts, v.subhead, v.cta) == (p.template, p.orientation, p.hero, p.hero_parts, p.subhead, p.cta)
        assert get_zones(v.template, 1024, 1024, orientation=v.orientation, **compute_geometry_flags(v)) == zones0
        keys.add((v.style_pack, v.lockup))
    assert len(keys) == 4  # 4 kiểu khác nhau
    assert generate_variants(p, 4) == vs  # tất định


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        yield browser.new_page()
        browser.close()


@pytest.mark.parametrize("w,h", [(1024, 1024), (576, 1024), (1024, 576)])
def test_brand_logo_placed_in_free_corner_never_on_text(page, w, h):
    page.set_viewport_size({"width": w, "height": h})
    page.set_content(build_template_html(_plan(brand_logo=LOGO), BG, w, h), wait_until="load")
    page.wait_for_function("window.__tendooAutofitDone === true", timeout=8000)
    r = page.evaluate("""() => {
      const img = document.querySelector('.tk-brand-logo');
      if (!img || img.dataset.tendooPlaced !== '1') return 'not placed';
      const L = img.getBoundingClientRect();
      for (const el of document.querySelectorAll('[data-autofit]')) {
        const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
        while (w.nextNode()) { if (!w.currentNode.textContent.trim()) continue;
          const rg = document.createRange(); rg.selectNodeContents(w.currentNode);
          for (const b of rg.getClientRects()) if (b.width > 1 && L.right > b.left && L.left < b.right && L.bottom > b.top && L.top < b.bottom) return 'overlap';
        }
      }
      return 'ok';
    }""")
    assert r == "ok"


@pytest.mark.e2e  # cần Chromium thật (conftest thay PosterRenderer.render bằng ảnh giả <= 1024px)
def test_export_print_a4_300dpi(tmp_path):
    out, size = export_print(_plan(), BG, tmp_path / "print.png", 1024, 1024)
    assert max(size) == round(297 / 25.4 * 300)  # 3508 px cạnh dài
    with Image.open(out) as im:
        assert round(im.info["dpi"][0]) == 300
