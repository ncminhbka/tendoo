#!/usr/bin/env python3
"""
scripts/demo_full_promo_poster.py

==================================================================================================
TENDOO AI -- MULTI-CASE COMMERCIAL POSTER PIPELINE BENCHMARK
==================================================================================================

OBJECTIVE:
  End-to-End autonomous generation of production-ready commercial posters without third-party APIs.
  Proves that our Unified Architecture (Script-Based Layout + FLUX.2 Klein Inpainting + HTML Typography)
  works across diverse commercial domains and does NOT overfit to Mid-Autumn, lanterns, or moonlight.

BENCHMARK CASES:
  1. `case1_mid_autumn`:
     - Domain: Traditional festive promotional poster (Direct 1:1 Benchmark vs Gemini).
     - Layout: 3-Tier Sandwich (Header on top, family dinner in mid-band, offer & footer at bottom).
     - Style: Warm amber, nostalgic festive illustration, serif typography.

  2. `case2_leather_wallet`:
     - Domain: Luxury men's fashion / leather craft showcase (Image 2 Wallet counterpart).
     - Layout: 2-Tier Hero Top (Top 42% dark negative space, Bottom 58% wallet on black slate stone).
     - Style: Dark luxury, black slate, metallic gold 3D gradient typography, studio rim lighting.

  3. `case3_fresh_beverage`:
     - Domain: Modern F&B / Summer iced fruit tea & kombucha.
     - Layout: 2-Tier Fresh Splash (Top 36% mint cream negative space, Bottom 64% iced tea splash).
     - Style: Vibrant daytime morning sunlight, fresh water splash, energetic geometric typography.

EXECUTION:
  # Run all 3 diverse cases (takes ~6-8 seconds total on 2x A30):
  python scripts/demo_full_promo_poster.py --cases all

  # Run only the Mid-Autumn direct benchmark vs Gemini:
  python scripts/demo_full_promo_poster.py --cases case1_mid_autumn

  # Run only the Luxury Wallet case:
  python scripts/demo_full_promo_poster.py --cases case2_leather_wallet

  # Run only the Fresh Beverage case:
  python scripts/demo_full_promo_poster.py --cases case3_fresh_beverage
==================================================================================================
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from flux2 import util
from flux2.masked_generation import (
    build_reference_tokens_from_latent,
    build_region_token_mask,
    build_region_token_mask_multi,
    denoise_unified_inpaint,
    soften_region_mask,
)
from flux2.sampling import (
    batched_prc_txt,
    default_images_prep,
    denoise as denoise_baseline,
    get_schedule,
    prc_img,
    prc_txt,
)
from tendoo.poster_renderer import PosterRenderer

# ==================================================================================================
# 1. THREE DIVERSE BENCHMARK CASES
# ==================================================================================================

CASES: Dict[str, Dict[str, Any]] = {
    "case1_mid_autumn": {
        "title_id": "Mid-Autumn Festive Promo (Direct Gemini Benchmark)",
        "layout_type": "sandwich_3tier",
        "scene_prompt": (
            "A cozy nostalgic Mid-Autumn festival family dinner in an ancient courtyard, three "
            "generations smiling warmly around a round wooden table with tea set, mooncakes, pomelo, "
            "glowing paper lanterns hanging from old eaves, soft golden ambient light, traditional "
            "Vietnamese architectural background, people enjoying festive food, cinematic warm lighting"
        ),
        "inpaint_prompt": (
            "Soft warm golden amber gradient background with glowing moon halo, quiet peaceful "
            "atmosphere, clean empty copy space, smooth texture, no lanterns, no people, no tables, "
            "no clutter, minimal background"
        ),
        "rects": [
            ((0.02, 0.28), (0.05, 0.95)),  # Top Header band (y: 2% to 28%)
            ((0.65, 0.98), (0.05, 0.95)),  # Bottom Footer/Offer band (y: 65% to 98%)
        ],
        "content": {
            "headline": "CHƯƠNG TRÌNH\nKHUYẾN MẠI TRUNG THU",
            "slogan": "Đón đêm rằm đoàn viên, ấm áp và trọn vẹn",
            "offer_title": "ƯU ĐÃI ĐẶC BIỆT",
            "offer_main": "GIẢM 15% CHO CÁC SẢN PHẨM QUÀ TẶNG TRUNG THU",
            "offer_sub": "FREESHIP ĐƠN TỪ 200K",
            "applicable": "ÁP DỤNG CHO: Bánh pía, Bánh bông lan, Hoa hướng dương và các loại hạt dinh dưỡng.",
            "dates": "THỜI GIAN: 19/09/2026 - 23/09/2026",
            "shop_name": "Tendoo Shop",
            "hotline": "SĐT: +84334842155",
            "web": "Web: 0334842155.tendoo.click",
            "address": "Đ/c: Phú Ngãi, Huyện Ba Tri, Tỉnh Bến Tre",
        },
        "theme": "mid_autumn",
    },
    "case2_leather_wallet": {
        "title_id": "Luxury Men's Leather Wallet Showcase (Dark Studio)",
        "layout_type": "hero_top_2tier",
        "scene_prompt": (
            "Commercial product photography of a luxury black genuine leather bifold wallet "
            "standing upright and slightly open on a textured black slate stone slab, dark moody "
            "studio background with dramatic golden rim lighting catching the leather texture and "
            "fine edge stitching, subtle specular reflection on polished stone surface, professional luxury product photography"
        ),
        "inpaint_prompt": (
            "Dark solid black studio gradient background, smooth elegant negative space, minimal "
            "clean dark lighting, empty copy space, no wallet, no pedestal, no objects, pure copy space"
        ),
        "rects": [
            ((0.02, 0.44), (0.05, 0.95)),  # Top Hero Header band (y: 2% to 44%)
        ],
        "content": {
            "headline": "GỌN GÀNG\nLỊCH LÃM",
            "subtitle_1": "Thiết kế gập đôi mỏng nhẹ",
            "subtitle_2": "Nhiều ngăn đựng thẻ và tiền mặt tiện ích",
            "badge": "DA BÒ THẬT 100%",
            "brand": "TENDOO LEATHER CRAFT",
            "hotline": "Hotline: 0334842155",
        },
        "theme": "luxury_gold",
    },
    "case3_fresh_beverage": {
        "title_id": "Fresh Summer Fruit Tea & Detox Beverage (Daylight Splash)",
        "layout_type": "fresh_splash_2tier",
        "scene_prompt": (
            "Commercial beverage photography of a tall clear glass of iced peach lemongrass tea with "
            "crystal clear ice cubes, fresh sliced peaches, orange slices and vibrant green mint leaves, "
            "dynamic water splash droplets frozen in mid-air, bright morning sunlight shining through "
            "condensation on glass, clean light pastel cream and mint green studio background, crisp refreshing summer beverage photography"
        ),
        "inpaint_prompt": (
            "Clean bright pastel cream and soft mint green gradient background, diffused morning "
            "sunlight, fresh airy atmosphere, empty open copy space, no glass, no fruits, no water "
            "splash, no ice, clean minimalist copy space"
        ),
        "rects": [
            ((0.02, 0.38), (0.05, 0.95)),  # Top Fresh Header band (y: 2% to 38%)
        ],
        "content": {
            "headline": "THANH MÁT TỰ NHIÊN\nBỪNG TỈNH NĂNG LƯỢNG",
            "subtitle": "Trà đào cam sả 100% trái cây nhiệt đới tươi sạch",
            "badge": "MUA 2 TẶNG 1",
            "offer_highlight": "TẶNG BÌNH GIỮ NHIỆT CAO CẤP",
            "dates": "Áp dụng duy nhất trong tuần lễ khai trương",
            "shop_name": "Tendoo Tea & Coffee",
            "hotline": "Đặt hàng: 0334842155",
        },
        "theme": "fresh_mint",
    },
}


# ==================================================================================================
# 2. HTML TYPOGRAPHY TEMPLATE BUILDERS
# ==================================================================================================

def pil_to_base64_data_uri(img: Image.Image) -> str:
    """Converts a PIL Image into an inlined base64 data URI for zero-network Playwright rendering."""
    import io
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def build_case1_mid_autumn_html(bg_data_uri: str, c: Dict[str, str], w: int, h: int) -> str:
    """Renders Case 1: Mid-Autumn Promotional Poster matching the Gemini benchmark layout."""
    hl_lines = c["headline"].split("\n")
    hl_html = "<br>".join(hl_lines)
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Montserrat:wght@400;600;700;800;900&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {w}px; height: {h}px; overflow: hidden;
    background: url('{bg_data_uri}') no-repeat center center / cover;
    font-family: 'Montserrat', sans-serif;
    position: relative;
  }}
  /* --- TOP HEADER BAND --- */
  .header-band {{
    position: absolute; top: 3.5%; left: 4%; right: 4%;
    text-align: center; color: #4A1A02;
  }}
  .headline {{
    font-family: 'Playfair Display', serif;
    font-size: 38px; font-weight: 900;
    line-height: 1.15; letter-spacing: 0.5px;
    color: #381203;
    text-shadow: 0 1px 2px rgba(255,245,210,0.8), 0 2px 8px rgba(0,0,0,0.15);
    text-transform: uppercase;
  }}
  .slogan {{
    margin-top: 10px;
    font-size: 15px; font-weight: 600;
    color: #4A1E08; letter-spacing: 0.2px;
    text-shadow: 0 1px 1px rgba(255,255,255,0.7);
  }}

  /* --- BOTTOM OFFER BAND --- */
  .footer-band {{
    position: absolute; bottom: 1.8%; left: 4%; right: 4%;
    text-align: center; color: #FFF;
  }}
  .offer-title {{
    font-size: 22px; font-weight: 900;
    color: #FFDE7A; text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.8);
    margin-bottom: 6px;
  }}
  .offer-main {{
    font-size: 14.5px; font-weight: 800;
    color: #FFF; line-height: 1.35;
    text-shadow: 0 2px 4px rgba(0,0,0,0.9);
    margin-bottom: 4px;
  }}
  .offer-sub {{
    font-size: 17px; font-weight: 900;
    color: #FFD24D; letter-spacing: 0.5px;
    text-shadow: 0 2px 6px rgba(0,0,0,0.9);
    margin-bottom: 8px;
  }}
  .applicable {{
    font-size: 12px; font-weight: 600;
    color: #FFE6B3; line-height: 1.3;
    text-shadow: 0 1px 3px rgba(0,0,0,0.9);
    margin-bottom: 6px;
  }}
  .dates {{
    font-size: 13.5px; font-weight: 700;
    color: #FFCC00; letter-spacing: 0.3px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.9);
    margin-bottom: 12px;
  }}
  .shop-footer {{
    display: flex; justify-content: space-between; align-items: flex-end;
    font-size: 10px; font-weight: 500;
    color: rgba(255,255,255,0.9);
    text-shadow: 0 1px 3px rgba(0,0,0,0.95);
    border-top: 1px solid rgba(255,215,0,0.3);
    padding-top: 6px;
  }}
  .shop-brand {{
    font-size: 15px; font-weight: 900; color: #FFF;
  }}
  .shop-meta {{
    text-align: right; line-height: 1.4;
  }}
</style>
</head>
<body>
  <div class="header-band">
    <div class="headline">{hl_html}</div>
    <div class="slogan">{c["slogan"]}</div>
  </div>
  <div class="footer-band">
    <div class="offer-title">{c["offer_title"]}</div>
    <div class="offer-main">{c["offer_main"]}</div>
    <div class="offer-sub">{c["offer_sub"]}</div>
    <div class="applicable">{c["applicable"]}</div>
    <div class="dates">{c["dates"]}</div>
    <div class="shop-footer">
      <div class="shop-brand">{c["shop_name"]}</div>
      <div class="shop-meta">
        <div>{c["hotline"]} | {c["web"]}</div>
        <div>{c["address"]}</div>
      </div>
    </div>
  </div>
</body>
</html>"""


def build_case2_wallet_html(bg_data_uri: str, c: Dict[str, str], w: int, h: int) -> str:
    """Renders Case 2: Luxury Leather Wallet Showcase matching Image 2 style."""
    hl_lines = c["headline"].split("\n")
    hl_html = "<br>".join(hl_lines)
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700&family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Montserrat:wght@400;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {w}px; height: {h}px; overflow: hidden;
    background: url('{bg_data_uri}') no-repeat center center / cover;
    font-family: 'Montserrat', sans-serif;
    position: relative;
  }}
  .hero-container {{
    position: absolute; top: 4%; left: 6%; right: 6%;
    text-align: center;
  }}
  .metallic-headline {{
    font-family: 'Oswald', sans-serif;
    font-size: 76px; font-weight: 700;
    line-height: 0.95; letter-spacing: 2px;
    text-transform: uppercase;
    background: linear-gradient(180deg, #FFF6D1 0%, #D8B257 35%, #AA8022 70%, #6E4E08 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 6px 16px rgba(0,0,0,0.9));
  }}
  .subtitle-box {{
    margin-top: 18px;
    font-family: 'Playfair Display', serif;
    color: #E2E2E2;
    font-size: 15px; font-weight: 400;
    line-height: 1.45; letter-spacing: 0.3px;
    text-shadow: 0 2px 6px rgba(0,0,0,0.9);
  }}
  .badge-tag {{
    display: inline-block;
    margin-top: 14px;
    border: 1px solid rgba(216,178,87,0.6);
    background: rgba(0,0,0,0.4);
    color: #D8B257;
    font-size: 10px; font-weight: 700;
    letter-spacing: 2px; padding: 4px 14px;
    border-radius: 20px;
    text-transform: uppercase;
  }}
  .brand-mark {{
    position: absolute; bottom: 3%; left: 6%; right: 6%;
    display: flex; justify-content: space-between;
    font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
    color: rgba(255,255,255,0.6);
    border-top: 1px solid rgba(255,255,255,0.15);
    padding-top: 8px;
  }}
</style>
</head>
<body>
  <div class="hero-container">
    <div class="metallic-headline">{hl_html}</div>
    <div class="subtitle-box">
      <div>{c["subtitle_1"]}</div>
      <div>{c["subtitle_2"]}</div>
    </div>
    <div class="badge-tag">{c["badge"]}</div>
  </div>
  <div class="brand-mark">
    <div>{c["brand"]}</div>
    <div>{c["hotline"]}</div>
  </div>
</body>
</html>"""


def build_case3_beverage_html(bg_data_uri: str, c: Dict[str, str], w: int, h: int) -> str:
    """Renders Case 3: Fresh Summer Fruit Tea & Detox Beverage."""
    hl_lines = c["headline"].split("\n")
    hl_html = "<br>".join(hl_lines)
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {w}px; height: {h}px; overflow: hidden;
    background: url('{bg_data_uri}') no-repeat center center / cover;
    font-family: 'Montserrat', sans-serif;
    position: relative;
  }}
  .fresh-header {{
    position: absolute; top: 4%; left: 5%; right: 5%;
    text-align: center;
  }}
  .fresh-headline {{
    font-size: 30px; font-weight: 900;
    line-height: 1.15; text-transform: uppercase;
    color: #0E4723;
    text-shadow: 0 1px 3px rgba(255,255,255,0.9), 0 3px 10px rgba(0,0,0,0.1);
  }}
  .fresh-sub {{
    margin-top: 8px;
    font-size: 13.5px; font-weight: 700;
    color: #1F6B38;
    text-shadow: 0 1px 2px rgba(255,255,255,0.8);
  }}
  .offer-pill {{
    display: inline-flex; align-items: center; gap: 8px;
    margin-top: 10px;
    background: linear-gradient(135deg, #FF6B35 0%, #E84A15 100%);
    color: #FFF;
    padding: 6px 16px; border-radius: 25px;
    font-size: 13px; font-weight: 800;
    box-shadow: 0 4px 12px rgba(232,74,21,0.35);
  }}
  .offer-highlight {{
    margin-top: 6px;
    font-size: 11.5px; font-weight: 700;
    color: #D33A00; letter-spacing: 0.2px;
  }}
  .beverage-footer {{
    position: absolute; bottom: 2.5%; left: 5%; right: 5%;
    display: flex; justify-content: space-between;
    font-size: 11px; font-weight: 700; color: #164726;
    background: rgba(255,255,255,0.7);
    padding: 6px 14px; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
</style>
</head>
<body>
  <div class="fresh-header">
    <div class="fresh-headline">{hl_html}</div>
    <div class="fresh-sub">{c["subtitle"]}</div>
    <div>
      <div class="offer-pill">{c["badge"]}</div>
    </div>
    <div class="offer-highlight">{c["offer_highlight"]}</div>
  </div>
  <div class="beverage-footer">
    <div>{c["shop_name"]}</div>
    <div>{c["hotline"]}</div>
  </div>
</body>
</html>"""


HTML_DISPATCH = {
    "case1_mid_autumn": build_case1_mid_autumn_html,
    "case2_leather_wallet": build_case2_wallet_html,
    "case3_fresh_beverage": build_case3_beverage_html,
}


# ==================================================================================================
# 3. RUNTIME PIPELINE EXECUTION
# ==================================================================================================

def get_fallback_font(size: int = 24, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Tries cross-platform system fonts before falling back to default."""
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_pil_typography_fallback(
    bg_img: Image.Image,
    case_info: Dict[str, Any],
    width: int,
    height: int,
) -> Image.Image:
    """Fallback PIL-based renderer when Playwright or headless browser is unavailable."""
    canvas = bg_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    c = case_info["content"]
    theme = case_info.get("theme", "mid_autumn")

    font_title = get_fallback_font(size=int(width * 0.055), bold=True)
    font_sub = get_fallback_font(size=int(width * 0.032), bold=False)
    font_badge = get_fallback_font(size=int(width * 0.028), bold=True)
    font_meta = get_fallback_font(size=int(width * 0.024), bold=False)

    if theme == "mid_autumn":
        draw.rectangle([(int(width * 0.04), int(height * 0.03)), (int(width * 0.96), int(height * 0.26))], fill=(255, 245, 220, 200), outline=(180, 100, 20, 255), width=2)
        hl = c.get("headline", "")
        draw.text((int(width * 0.5), int(height * 0.09)), hl, fill=(60, 20, 5), font=font_title, anchor="mm", align="center")
        draw.text((int(width * 0.5), int(height * 0.19)), c.get("slogan", ""), fill=(120, 50, 10), font=font_sub, anchor="mm", align="center")

        draw.rectangle([(int(width * 0.04), int(height * 0.68)), (int(width * 0.96), int(height * 0.96))], fill=(255, 248, 235, 220), outline=(200, 120, 30, 255), width=2)
        draw.text((int(width * 0.5), int(height * 0.72)), c.get("offer_main", ""), fill=(180, 40, 10), font=font_badge, anchor="mm")
        draw.text((int(width * 0.5), int(height * 0.77)), c.get("offer_sub", ""), fill=(60, 20, 5), font=font_sub, anchor="mm")
        draw.text((int(width * 0.5), int(height * 0.83)), c.get("applicable", ""), fill=(90, 40, 10), font=font_meta, anchor="mm")
        draw.text((int(width * 0.5), int(height * 0.88)), c.get("dates", ""), fill=(140, 50, 10), font=font_meta, anchor="mm")
        draw.text((int(width * 0.5), int(height * 0.93)), f"{c.get('shop_name', '')} | {c.get('hotline', '')}", fill=(40, 20, 10), font=font_meta, anchor="mm")

    elif theme == "luxury_gold":
        draw.rectangle([(int(width * 0.05), int(height * 0.04)), (int(width * 0.95), int(height * 0.40))], fill=(10, 10, 10, 180), outline=(216, 178, 87, 200), width=2)
        hl = c.get("headline", "")
        draw.text((int(width * 0.5), int(height * 0.14)), hl, fill=(235, 195, 100), font=font_title, anchor="mm", align="center")
        draw.text((int(width * 0.5), int(height * 0.24)), f"{c.get('subtitle_1', '')}\n{c.get('subtitle_2', '')}", fill=(220, 220, 220), font=font_sub, anchor="mm", align="center")
        draw.text((int(width * 0.5), int(height * 0.33)), c.get("badge", ""), fill=(216, 178, 87), font=font_badge, anchor="mm")
        draw.text((int(width * 0.5), int(height * 0.96)), f"{c.get('brand', '')} | {c.get('hotline', '')}", fill=(200, 200, 200), font=font_meta, anchor="mm")

    else:
        draw.rectangle([(int(width * 0.04), int(height * 0.04)), (int(width * 0.96), int(height * 0.35))], fill=(245, 255, 245, 200), outline=(30, 130, 60, 200), width=2)
        hl = c.get("headline", "")
        draw.text((int(width * 0.5), int(height * 0.12)), hl, fill=(15, 75, 35), font=font_title, anchor="mm", align="center")
        draw.text((int(width * 0.5), int(height * 0.20)), c.get("subtitle", ""), fill=(30, 110, 55), font=font_sub, anchor="mm")
        draw.text((int(width * 0.5), int(height * 0.26)), c.get("badge", ""), fill=(230, 80, 20), font=font_badge, anchor="mm")
        draw.text((int(width * 0.5), int(height * 0.31)), c.get("offer_highlight", ""), fill=(200, 60, 10), font=font_meta, anchor="mm")
        draw.text((int(width * 0.5), int(height * 0.96)), f"{c.get('shop_name', '')} | {c.get('hotline', '')}", fill=(20, 60, 30), font=font_meta, anchor="mm")

    out_composite = Image.alpha_composite(canvas, overlay)
    return out_composite.convert("RGB")


def run_case(
    case_key: str,
    model: Any,
    ae: Any,
    text_encoder: Any,
    ae_dtype: torch.dtype,
    device: str,
    aux_device: str,
    out_dir: Path,
    steps: int = 8,
    guidance: float = 1.5,
    seed: int = 42,
    width: int = 576,
    height: int = 1024,
) -> Dict[str, Any]:
    case_info = CASES[case_key]
    case_dir = out_dir / case_key
    case_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🎬 EXECUTING: {case_info['title_id']} [{case_key}]")
    print("=" * 80)

    h_lat = height // 16
    w_lat = width // 16
    C = 128

    # 1. Build Multi-rect Soft Mask
    rects = case_info["rects"]
    region_mask_flat = build_region_token_mask_multi(h_lat, w_lat, rects)
    soft_mask = soften_region_mask(region_mask_flat, h_lat, w_lat, feather_iters=3)

    # 2. Generate Scene Baseline
    print(f"  ▶ Step 1: Generating scene baseline ({steps} steps Distill)...")
    torch.manual_seed(seed)
    z_init = torch.randn(1, C, h_lat, w_lat, device=device, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device)
    img_ids = img_ids.unsqueeze(0).to(device)

    timesteps = get_schedule(num_steps=steps, image_seq_len=img_tokens.shape[1])
    with torch.no_grad():
        ctx_scene = text_encoder([case_info["scene_prompt"]]).to(torch.bfloat16)
        ctx_scene, ctx_scene_ids = prc_txt(ctx_scene[0])
        ctx_scene = ctx_scene.unsqueeze(0).to(device)
        ctx_scene_ids = ctx_scene_ids.unsqueeze(0).to(device)

        out_scene = denoise_baseline(model, img_tokens, img_ids, ctx_scene, ctx_scene_ids, timesteps, guidance=guidance)
        z_dec = out_scene[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
        x_dec = ae.decode(z_dec).float()
        x_dec = ((x_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        baseline_pil = Image.fromarray(x_dec)

    baseline_path = case_dir / "01_baseline_raw.png"
    baseline_pil.save(baseline_path)
    print(f"    [✓] Baseline generated -> {baseline_path.name}")

    # 3. Run Unified Inpaint to Clean the Copy Space
    print(f"  ▶ Step 2: Unified Inpaint clearing copy space ({steps} steps Distill)...")
    t0 = time.time()
    with torch.no_grad():
        prep_tensor = default_images_prep(baseline_pil)
        if isinstance(prep_tensor, list):
            prep_tensor = prep_tensor[0]
        prep_tensor = prep_tensor.unsqueeze(0).to(device=aux_device, dtype=ae_dtype)
        z_orig = ae.encode(prep_tensor)[0].to(device=device, dtype=torch.bfloat16)

        ref_tokens, ref_ids = build_reference_tokens_from_latent(z_orig, t_offset=10.0, device=device)
        z_orig_tokens, img_ids = prc_img(z_orig)
        z_orig_tokens = z_orig_tokens.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        img_ids = img_ids.unsqueeze(0).to(device=device)

        ctx_inp = text_encoder([case_info["inpaint_prompt"]]).to(torch.bfloat16)
        ctx_inp, ctx_inp_ids = prc_txt(ctx_inp[0])
        ctx_inp = ctx_inp.unsqueeze(0).to(device)
        ctx_inp_ids = ctx_inp_ids.unsqueeze(0).to(device)

        noise_fixed = torch.randn_like(z_orig_tokens)
        out_inpaint = denoise_unified_inpaint(
            model=model,
            z_orig_tokens=z_orig_tokens,
            img_ids=img_ids,
            ref_tokens=ref_tokens,
            ref_ids=ref_ids,
            soft_mask=soft_mask,
            txt=ctx_inp,
            txt_ids=ctx_inp_ids,
            timesteps=timesteps,
            guidance=guidance,
            noise_fixed=noise_fixed,
            is_cfg=False,
        )
        z_inpaint_dec = out_inpaint[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
        x_inpaint_dec = ae.decode(z_inpaint_dec).float()
        x_inpaint_dec = ((x_inpaint_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        cleaned_bg_pil = Image.fromarray(x_inpaint_dec)
    dur_inpaint = time.time() - t0

    cleaned_bg_path = case_dir / "02_cleaned_background.png"
    cleaned_bg_pil.save(cleaned_bg_path)
    print(f"    [✓] Copy space cleared in {dur_inpaint:.2f}s -> {cleaned_bg_path.name}")

    # 4. Synthesize Responsive HTML Typography Overlay
    print("  ▶ Step 3: Overlaying vector typography (Playwright Chromium)...")
    bg_data_uri = pil_to_base64_data_uri(cleaned_bg_pil)
    html_fn = HTML_DISPATCH[case_key]
    html_str = html_fn(bg_data_uri, case_info["content"], width, height)

    html_path = case_dir / "03_poster_template.html"
    html_path.write_text(html_str, encoding="utf-8")
    print(f"    [✓] HTML template saved -> {html_path.name}")

    final_poster_path = case_dir / "03_final_poster.png"
    try:
        PosterRenderer.render(html_content=html_str, output_image_path=final_poster_path, width=width, height=height)
        print(f"    [✓] Final poster rendered (Playwright) -> {final_poster_path.name}")
    except Exception as e:
        print(f"    [!] Playwright render unavailable ({e}). Running PIL typography fallback...")
        final_poster_pil = draw_pil_typography_fallback(cleaned_bg_pil, case_info, width, height)
        final_poster_pil.save(final_poster_path)
        print(f"    [✓] Final poster rendered (PIL Fallback) -> {final_poster_path.name}")

    # 5. Build 3-Panel Comparison Strip
    w, h = baseline_pil.size
    final_poster_pil = Image.open(final_poster_path).convert("RGB")
    comp = Image.new("RGB", (w * 3, h), (18, 18, 18))
    comp.paste(baseline_pil, (0, 0))
    comp.paste(cleaned_bg_pil, (w, 0))
    comp.paste(final_poster_pil, (w * 2, 0))

    draw = ImageDraw.Draw(comp)
    font = get_fallback_font(size=26, bold=True)

    draw.rectangle([(10, 10), (250, 48)], fill=(0, 0, 0, 200))
    draw.text((20, 14), "1. RAW BASELINE", fill=(255, 255, 255), font=font)

    draw.rectangle([(w + 10, 10), (w + 290, 48)], fill=(0, 0, 0, 200))
    draw.text((w + 20, 14), "2. CLEANED COPY SPACE", fill=(255, 200, 50), font=font)

    draw.rectangle([(w * 2 + 10, 10), (w * 2 + 300, 48)], fill=(0, 0, 0, 200))
    draw.text((w * 2 + 20, 14), "3. FINAL COMPOSITE", fill=(80, 255, 80), font=font)

    comparison_path = case_dir / "04_comparison_strip.png"
    comp.save(comparison_path)
    print(f"    [✓] Comparison strip saved -> {comparison_path.name}")

    return {
        "case": case_key,
        "title": case_info["title_id"],
        "dur_inpaint_s": round(dur_inpaint, 2),
        "baseline": str(baseline_path.name),
        "cleaned_bg": str(cleaned_bg_path.name),
        "final_poster": str(final_poster_path.name),
        "comparison_strip": str(comparison_path.name),
    }


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Multi-Case Commercial Poster Benchmark")
    parser.add_argument("--cases", type=str, default="all",
                        choices=["all", "case1_mid_autumn", "case2_leather_wallet", "case3_fresh_beverage"],
                        help="Which case(s) to benchmark ('all', 'case1_mid_autumn', 'case2_leather_wallet', 'case3_fresh_beverage').")
    parser.add_argument("--steps", type=int, default=8, help="ODE steps for Distill (default 8 steps, ~2s).")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale (default 1.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=576, help="Canvas width.")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Primary DiT GPU.")
    parser.add_argument("--out-dir", type=str, default="output_demo_promo_posters", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 TENDOO AI MULTI-CASE COMMERCIAL POSTER BENCHMARK")
    print("=" * 80)

    device = args.device
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        aux_device = "cuda:1"
        print(f"  [Device] Multi-GPU: DiT on {device}, AE & TextEncoder on {aux_device}")
    elif num_gpus == 1:
        aux_device = "cuda:0"
        print(f"  [Device] Single GPU: All components on {device}")
    else:
        device = aux_device = "cpu"
        print("  [Device] CPU execution")

    # Load Model
    model_name = "flux.2-klein-4b"
    pdata = Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B"))
    distill_cand = pdata / "flux-2-klein-4b.safetensors"
    if distill_cand.exists() and "KLEIN_4B_MODEL_PATH" not in os.environ:
        os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_cand)

    print("\n⏳ Loading FLUX.2 Klein 4B Distill...")
    model = util.load_flow_model(model_name, device=device)
    model.eval()

    ae = util.load_ae(model_name, device=aux_device)
    ae.eval()
    ae_dtype = next(ae.parameters()).dtype

    text_encoder = util.load_text_encoder(model_name, device=aux_device)

    # Determine which cases to run
    cases_to_run = list(CASES.keys()) if args.cases == "all" else [args.cases]

    results: List[Dict[str, Any]] = []
    t_start_all = time.time()

    for ckey in cases_to_run:
        res = run_case(
            case_key=ckey,
            model=model,
            ae=ae,
            text_encoder=text_encoder,
            ae_dtype=ae_dtype,
            device=device,
            aux_device=aux_device,
            out_dir=out_dir,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
        results.append(res)

    total_time = time.time() - t_start_all

    # Save summary manifest
    summary_path = out_dir / "summary_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_runtime_seconds": round(total_time, 2),
            "cases_evaluated": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"🎉 HOÀN THÀNH TOÀN BỘ BENCHMARK ({len(results)} CASES) TRONG {total_time:.2f}s!")
    print("=" * 80)
    print(f"{'Case ID':<24} | {'Domain':<38} | {'Inpaint Time':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['case']:<24} | {r['title']:<38} | {r['dur_inpaint_s']}s")
    print("-" * 80)
    print(f"Tất cả kết quả lưu tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
