from __future__ import annotations
import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / 'src']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tendoo.engine.blocks import (
    AdaptiveBlock,
    VALID_ZONES,
    ROLE_SCALE,
    deduplicate_blocks,
    map_category_to_default_blocks,
    normalize_fingerprint,
)
from tendoo.engine.geometry import (
    PRODUCT_SANCTUARY_BY_RATIO,
    compute_block_metrics,
    get_product_sanctuary_rect,
    get_zone_bounding_box,
)
from tendoo.engine.mask import generate_omni_corridor_mask
from tendoo.engine.layout import OmniBlockLayout
from tendoo.layouts.base import PosterContent, ColorPalette
from tendoo.core.fonts import FONT_CATALOG, FONTS_DIR
from tendoo.core.typography import balance_vietnamese_headline, fit_font_size_px


def test_adaptive_block_dataclass():
    b = AdaptiveBlock(text='MUA 1 TANG 1', role='badge', zone='top_right', icon='tag')
    d = b.to_dict()
    assert d['text'] == 'MUA 1 TANG 1'
    assert d['role'] == 'badge'
    assert d['zone'] == 'top_right'
    assert d['icon'] == 'tag'
    b2 = AdaptiveBlock.from_dict(d)
    assert b2.text == b.text and b2.role == b.role


def test_category_mappings_all_six():
    categories = ['promo', 'product_intro', 'opening', 'feedback', 'recruitment', 'guide']
    for cat in categories:
        fields = {
            'discount': 'GIAM 50%',
            'product_name': 'Tai Nghe Sonic Pro',
            'opening_date': '14/05/2026',
            'feedback_target': 'Dich vu PT 1:1',
            'job_position': 'Senior AI Engineer',
            'guide_steps': 'Buoc 1: Cai dat | Buoc 2: Khoi chay',
        }
        blocks = map_category_to_default_blocks(
            category=cat,
            title='TIEU DE CHINH',
            fields=fields,
            store_info={'brand': 'Tendoo Studio', 'phone': '0123456789'},
        )
        assert len(blocks) >= 2
        roles = [b.role for b in blocks]
        assert 'hero' in roles
        assert any(b.zone == 'bottom_bar' for b in blocks)


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
    blocks = [
        AdaptiveBlock(text='Private Coaching Transformation', role='hero'),
        AdaptiveBlock(text='Private Coaching Transformation', role='body'),
        AdaptiveBlock(text='97% khach hang hai long', role='body'),
    ]
    deduped = deduplicate_blocks(blocks, hero_title='Private Coaching Transformation')
    assert len(deduped) == 2
    assert deduped[0].role == 'hero' and deduped[0].text == 'Private Coaching Transformation'
    assert deduped[1].text == '97% khach hang hai long'


def test_uppercase_font_fitting_no_clipping():
    font_path = str(FONTS_DIR / FONT_CATALOG['bevietnam']['file'])
    fsize, w, h = fit_font_size_px(
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
    lines, metrics = balance_vietnamese_headline('Tai Nghe Sonic Pro')
    assert len(lines) == 2
    assert lines[0] == 'Tai Nghe' and lines[1] == 'Sonic Pro'
    assert metrics['font_size'] >= 50


def test_product_sanctuary_rects():
    for ratio_str, (ymin, xmin, ymax, xmax) in PRODUCT_SANCTUARY_BY_RATIO.items():
        assert 0.0 < xmin < xmax < 1.0
        assert 0.0 < ymin < ymax < 1.0


def test_omni_corridor_mask_generation():
    font_path = str(FONTS_DIR / FONT_CATALOG['bevietnam']['file'])
    blocks = [
        AdaptiveBlock(text='TIEU DE HERO', role='hero', zone='top_center'),
        AdaptiveBlock(text='GIAM 50%', role='badge', zone='middle_right'),
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
    assert 'block-hero' in html
    assert 'TRẢI NGHIỆM ĐỈNH CAO' in html
    assert 'GIẢM 40%' in html
    assert '1900 8888' in html
    assert 'omni-bottom-bar' in html
