"""
tests/test_engine_blocks.py

Covers src/tendoo/engine/blocks.py, engine/geometry.py, engine/mask.py, engine/layout.py:
AdaptiveBlock dataclass, content-fingerprint dedup, corridor mask generation, and a
render_html smoke test. Category-schema/mapping coverage lives in
test_core_category_schema.py; font-fitting/line-balancing coverage lives in
test_core_typography.py (both moved out 2026-09-13 as part of the src/ reorg).
"""

from __future__ import annotations

from tendoo.core.base import ColorPalette, PosterContent
from tendoo.core.fonts import FONT_CATALOG, FONTS_DIR
from tendoo.engine.blocks import (
    AdaptiveBlock,
    deduplicate_blocks,
    map_category_to_default_blocks,
)
from tendoo.engine.geometry import PRODUCT_SANCTUARY_BY_RATIO
from tendoo.engine.layout import OmniBlockLayout
from tendoo.engine.mask import generate_omni_corridor_mask


def test_adaptive_block_dataclass():
    b = AdaptiveBlock(text='MUA 1 TANG 1', size='medium', container='pill', zone='top_right', icon='tag')
    d = b.to_dict()
    assert d['text'] == 'MUA 1 TANG 1'
    assert d['size'] == 'medium'
    assert d['container'] == 'pill'
    assert d['zone'] == 'top_right'
    assert d['icon'] == 'tag'
    b2 = AdaptiveBlock.from_dict(d)
    assert b2.text == b.text and b2.size == b.size and b2.container == b.container


def test_store_info_spatial_override_to_top_bar():
    blocks = map_category_to_default_blocks(
        category='promo',
        title='SIEU SALE',
        fields={'discount': '30%'},
        store_info={'brand': 'Tendoo Fashion', 'phone': '0987654321'},
        spatial_override_zone='top_bar',
    )
    store_block = next((b for b in blocks if b.field == 'store_info'), None)
    assert store_block is not None
    assert store_block.zone == 'top_bar'


def test_deduplicate_blocks_content_fingerprint():
    # deduplicate_blocks() has no 'hero_title' parameter -- it decides which duplicate
    # to keep via each block's own `size` (xlarge > large > other, see
    # engine/blocks.py::deduplicate_blocks's own sort key), not an external hint.
    blocks = [
        AdaptiveBlock(text='Private Coaching Transformation', size='xlarge'),
        AdaptiveBlock(text='Private Coaching Transformation', size='medium'),
        AdaptiveBlock(text='97% khach hang hai long', size='medium'),
    ]
    deduped = deduplicate_blocks(blocks)
    assert len(deduped) == 2
    assert deduped[0].size == 'xlarge' and deduped[0].text == 'Private Coaching Transformation'
    assert deduped[1].text == '97% khach hang hai long'


def test_product_sanctuary_rects():
    for ratio_str, (ymin, xmin, ymax, xmax) in PRODUCT_SANCTUARY_BY_RATIO.items():
        assert 0.0 < xmin < xmax < 1.0
        assert 0.0 < ymin < ymax < 1.0


def test_omni_corridor_mask_generation():
    font_path = str(FONTS_DIR / FONT_CATALOG['bevietnam']['file'])
    blocks = [
        AdaptiveBlock(text='TIEU DE HERO', size='xlarge', zone='top_center'),
        AdaptiveBlock(text='GIAM 50%', size='medium', container='pill', zone='middle_right'),
    ]
    mask = generate_omni_corridor_mask(
        width=1024,
        height=1024,
        blocks=blocks,
        font_path=font_path,
    )
    assert mask.shape == (1024, 1024)
    assert mask.min() == 0.0
    assert mask.max() > 0.7


def test_omni_layout_render_html():
    layout = OmniBlockLayout()
    content = PosterContent(
        headline='TRẢI NGHIỆM ĐỈNH CAO',
        offer_main='GIẢM 40%',
        brand='Tendoo Audio',
        hotline='1900 8888',
        category='promo',
    )
    palette = ColorPalette(is_dark=True, luminance=0.15, hue=210, comp_hue=30, headline_color='#FFFFFF')
    html = layout.render_html(content, palette, bg_data_uri='', width=1024, height=1024)
    # A headline renders as `<div class="block-pure-text size-xlarge ...">`, not
    # 'block-hero' -- see engine/layout.py::render_html's container-class emission
    # (block-pure-text/block-cta/block-pill/block-card/block-qr, no 'block-hero' exists).
    assert 'block-pure-text' in html
    assert 'size-xlarge' in html
    assert 'TRẢI NGHIỆM' in html
    assert 'ĐỈNH CAO' in html
    assert 'GIẢM 40%' in html
    assert '1900 8888' in html
    assert 'omni-bottom-bar' in html
