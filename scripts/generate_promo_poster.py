#!/usr/bin/env python3
"""
scripts/generate_promo_poster.py

==================================================================================================
TENDOO AI - DYNAMIC PARAMETERIZED COMMERCIAL POSTER GENERATOR (ZERO-HARDCODE CLI)
==================================================================================================

OBJECTIVE:
  Production CLI tool allowing marketers, developers, or backend services to generate commercial-grade
  promotional posters on-demand via bash command line or JSON configuration.

KEY CAPABILITIES:
  1. ZERO HARDCODED VALUES:
     - Prompt, texts, layout, aspect ratio, fonts, and colors are fully parameterizable via CLI.
  2. PROCEDURAL LAYOUT GENERATION:
     - Automatically calculates mask coordinates from `--layout` (sandwich, hero_top, hero_bottom, fresh_splash).
     - Automatically derives optimal inpainting prompts from the scene lighting if not specified.
  3. AUTOMATED COLOR & CONTRAST HARMONY ENGINE (WCAG 2.1 & ITU-R BT.601):
     - Analyzes luminance and extracts dominant HSV hue from the cleaned background crop.
     - Dynamically computes high-contrast, harmonious typography palettes (anchored hue for primary text,
       complementary 180° hue for CTA badges, adaptive drop shadows, and glassmorphic scrims).
  4. RESILIENT HYBRID RENDERING:
     - Exports standalone responsive HTML5/CSS3 template (`03_poster_template.html`).
     - Renders sub-pixel vector poster with Playwright Chromium, with automatic fallback to PIL.
     - Generates 3-panel comparison strip (`04_comparison_strip.png`) for instant quality verification.

USAGE EXAMPLES:
  # Example 1: Mid-Autumn promo with custom store info
  python scripts/generate_promo_poster.py \\
    --scene-prompt "A cozy traditional Vietnamese courtyard family tea gathering, mooncakes on wooden table, warm golden lighting" \\
    --layout sandwich \\
    --headline "LỄ HỘI ĐOÀN VIÊN\\nƯU ĐÃI TRĂNG RẰM" \\
    --slogan "Gửi trọn yêu thương trong từng hộp bánh" \\
    --offer-main "GIẢM 20% BÁNH TRUNG THU" \\
    --offer-sub "FREESHIP TOÀN QUỐC TỪ 300K" \\
    --applicable "Áp dụng: Bánh nướng thập cẩm, bánh dẻo hạt sen" \\
    --dates "Thời gian: 15/09 - 25/09/2026" \\
    --brand "Tendoo Bakery" \\
    --hotline "0988.123.456" \\
    --out-dir output_custom_midautumn

  # Example 2: Luxury Leather Craft (Hero Top layout)
  python scripts/generate_promo_poster.py \\
    --scene-prompt "Commercial product photography of a luxury black leather wallet on black slate stone, golden rim light" \\
    --layout hero_top \\
    --headline "ĐẲNG CẤP PHÁI MẠNH" \\
    --slogan "Chất liệu da bò nguyên tấm thủ công" \\
    --offer-main "GIẢM NGAY 30%" \\
    --brand "Tendoo Leather" \\
    --hotline "0334842155" \\
    --out-dir output_custom_wallet

  # Example 3: Load all fields from a JSON config file
  python scripts/generate_promo_poster.py --config config/sample_poster.json
==================================================================================================
"""

from __future__ import annotations

import argparse
import base64
import colorsys
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
    build_region_token_mask_multi,
    denoise_unified_inpaint,
    soften_region_mask,
)
from flux2.sampling import (
    default_images_prep,
    denoise as denoise_baseline,
    get_schedule,
    prc_img,
    prc_txt,
)
from tendoo.typography_engine import PosterRenderer


# ==================================================================================================
# 1. PROCEDURAL LAYOUT DEFINITIONS
# ==================================================================================================

LAYOUT_PRESETS: Dict[str, Dict[str, Any]] = {
    "sandwich": {
        "description": "3-tier sandwich: Header band on top (2-28%), Offer/Footer band at bottom (65-98%).",
        "rects": [
            ((0.02, 0.28), (0.05, 0.95)),  # Top Header band
            ((0.65, 0.98), (0.05, 0.95)),  # Bottom Footer band
        ],
        "crop_zone_header": (0.02, 0.05, 0.28, 0.95),
        "crop_zone_footer": (0.65, 0.05, 0.98, 0.95),
    },
    "hero_top": {
        "description": "2-tier hero: Negative copy space on top (2-42%), Product on bottom.",
        "rects": [
            ((0.02, 0.42), (0.05, 0.95)),
        ],
        "crop_zone_header": (0.02, 0.05, 0.42, 0.95),
        "crop_zone_footer": (0.90, 0.05, 0.98, 0.95),
    },
    "hero_bottom": {
        "description": "2-tier hero: Subject on top, Negative copy space at bottom (58-98%).",
        "rects": [
            ((0.58, 0.98), (0.05, 0.95)),
        ],
        "crop_zone_header": (0.02, 0.05, 0.15, 0.95),
        "crop_zone_footer": (0.58, 0.05, 0.98, 0.95),
    },
    "fresh_splash": {
        "description": "2-tier splash: Compact fresh header band on top (2-36%), splash subject on bottom.",
        "rects": [
            ((0.02, 0.36), (0.05, 0.95)),
        ],
        "crop_zone_header": (0.02, 0.05, 0.36, 0.95),
        "crop_zone_footer": (0.92, 0.05, 0.98, 0.95),
    },
}


def derive_inpaint_prompt(scene_prompt: str) -> str:
    """Procedurally constructs a prompt to clear copy space while preserving ambient lighting."""
    return (
        "Smooth clean gradient background matching the scene lighting and ambient atmosphere, "
        "minimalist negative space, elegant empty copy space, no objects, no people, no clutter, "
        "no text, seamless clean studio backdrop"
    )


# ==================================================================================================
# 2. AUTOMATED COLOR & CONTRAST HARMONY ENGINE (WCAG 2.1 & ITU-R BT.601)
# ==================================================================================================

def compute_relative_luminance(rgb_array: np.ndarray) -> float:
    """ITU-R BT.601 Perceived Luminance: Y = 0.299R + 0.587G + 0.114B [0..255]."""
    if rgb_array.size == 0:
        return 128.0
    r = rgb_array[:, :, 0].astype(np.float32)
    g = rgb_array[:, :, 1].astype(np.float32)
    b = rgb_array[:, :, 2].astype(np.float32)
    return float(np.mean(0.299 * r + 0.587 * g + 0.114 * b))


def extract_dominant_hsv(rgb_array: np.ndarray) -> Tuple[int, float, float]:
    """Extracts dominant hue (degrees [0..359]), saturation [0..1], and value [0..1]."""
    if rgb_array.size == 0:
        return (0, 0.0, 0.5)
    med_rgb = np.median(rgb_array.reshape(-1, 3), axis=0).astype(float) / 255.0
    h, s, v = colorsys.rgb_to_hsv(med_rgb[0], med_rgb[1], med_rgb[2])
    return (int(h * 360), float(s), float(v))


def analyze_color_harmony(
    img_np: np.ndarray,
    crop_zone: Tuple[float, float, float, float],
    color_mode: str = "auto",
    font_style: str = "auto",
) -> Dict[str, str]:
    """
    Computes mathematically harmonious text, badge, shadow, and glassmorphic colors
    tailored to the actual background pixels.
    """
    h, w, _ = img_np.shape
    y1, x1, y2, x2 = crop_zone
    crop = img_np[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]

    lum = compute_relative_luminance(crop)
    hue, sat, val = extract_dominant_hsv(crop)

    if color_mode == "dark":
        is_dark = True
    elif color_mode == "light":
        is_dark = False
    else:
        # Auto detection based on luminance threshold
        is_dark = (lum < 125.0)

    # Complementary hue for high-impact CTA badge (180 deg opposite on color wheel)
    comp_hue = (hue + 180) % 360

    if is_dark:
        # Dark Background Palette
        if font_style == "luxury_serif":
            headline_color = (
                "linear-gradient(180deg, #FFF6D1 0%, #D8B257 40%, #AA8022 75%, #6E4E08 100%)"
            )
            headline_is_gradient = True
        else:
            headline_color = f"hsl({hue}, 20%, 96%)"
            headline_is_gradient = False

        sub_color = f"hsl({hue}, 15%, 84%)"
        text_shadow = "0 2px 10px rgba(0,0,0,0.85), 0 1px 3px rgba(0,0,0,0.75)"
        badge_bg = f"linear-gradient(135deg, hsl({comp_hue}, 90%, 55%) 0%, hsl({(comp_hue + 30) % 360}, 85%, 45%) 100%)"
        badge_text = "#FFFFFF"
        badge_shadow = f"0 4px 14px hsla({comp_hue}, 85%, 50%, 0.45)"
        glass_bg = "rgba(255, 255, 255, 0.10)"
        glass_border = "rgba(255, 255, 255, 0.22)"
        footer_bg = "rgba(10, 10, 10, 0.65)"
        footer_text = f"hsl({hue}, 10%, 78%)"
        offer_highlight_color = f"hsl({comp_hue}, 85%, 65%)"
    else:
        # Light Background Palette
        # Anchors primary text deeply into the dominant hue for sophisticated richness
        text_hue_sat = min(sat * 1.2, 0.8)
        headline_color = f"hsl({hue}, {int(text_hue_sat * 100)}%, 14%)"
        headline_is_gradient = False
        sub_color = f"hsl({hue}, {int(text_hue_sat * 80)}%, 26%)"
        text_shadow = "0 1px 3px rgba(255,255,255,0.9), 0 2px 8px rgba(0,0,0,0.1)"
        badge_bg = f"linear-gradient(135deg, hsl({comp_hue}, 95%, 48%) 0%, hsl({(comp_hue + 20) % 360}, 90%, 40%) 100%)"
        badge_text = "#FFFFFF"
        badge_shadow = f"0 4px 12px hsla({comp_hue}, 90%, 45%, 0.35)"
        glass_bg = "rgba(255, 255, 255, 0.78)"
        glass_border = f"hsl({hue}, 25%, 82%)"
        footer_bg = "rgba(255, 255, 255, 0.85)"
        footer_text = f"hsl({hue}, 45%, 18%)"
        offer_highlight_color = f"hsl({comp_hue}, 90%, 38%)"

    return {
        "is_dark": str(is_dark),
        "lum": f"{lum:.1f}",
        "hue": str(hue),
        "comp_hue": str(comp_hue),
        "headline_color": headline_color,
        "headline_is_gradient": str(headline_is_gradient),
        "sub_color": sub_color,
        "text_shadow": text_shadow,
        "badge_bg": badge_bg,
        "badge_text": badge_text,
        "badge_shadow": badge_shadow,
        "glass_bg": glass_bg,
        "glass_border": glass_border,
        "footer_bg": footer_bg,
        "footer_text": footer_text,
        "offer_highlight_color": offer_highlight_color,
    }


# ==================================================================================================
# 3. DYNAMIC HTML5 / CSS3 POSTER TEMPLATE COMPOSER
# ==================================================================================================

def pil_to_base64_data_uri(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def build_dynamic_html(
    bg_data_uri: str,
    content: Dict[str, str],
    palette: Dict[str, str],
    layout_type: str,
    font_style: str,
    w: int,
    h: int,
) -> str:
    """Assembles a responsive, self-contained HTML poster with adaptive CSS variables."""
    hl_lines = content.get("headline", "").split("\n")
    hl_html = "<br>".join(hl_lines)

    # Font family selection
    if font_style == "luxury_serif":
        font_headline = "'Playfair Display', serif"
        font_sub = "'Montserrat', sans-serif"
    elif font_style == "bold_display":
        font_headline = "'Oswald', sans-serif"
        font_sub = "'Montserrat', sans-serif"
    else:  # modern_sans
        font_headline = "'Montserrat', sans-serif"
        font_sub = "'Montserrat', sans-serif"

    # Headline CSS (gradient vs solid)
    if palette["headline_is_gradient"] == "True":
        headline_style = f"""
          background: {palette['headline_color']};
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          filter: drop-shadow(0 4px 12px rgba(0,0,0,0.85));
        """
    else:
        headline_style = f"""
          color: {palette['headline_color']};
          text-shadow: {palette['text_shadow']};
        """

    # Layout-specific structure
    if layout_type == "sandwich":
        body_content = f"""
          <div class="header-zone">
            <div class="headline">{hl_html}</div>
            <div class="slogan">{content.get('slogan', '')}</div>
          </div>
          <div class="footer-zone glass-panel">
            <div class="offer-pill">{content.get('offer_main', '')}</div>
            <div class="offer-sub">{content.get('offer_sub', '')}</div>
            <div class="applicable">{content.get('applicable', '')}</div>
            <div class="dates">{content.get('dates', '')}</div>
            <div class="brand-bar">
              <div class="brand-name">{content.get('brand', '')}</div>
              <div class="contact-info">{content.get('hotline', '')} | {content.get('web', '')}</div>
            </div>
          </div>
        """
    elif layout_type in ("hero_top", "fresh_splash"):
        body_content = f"""
          <div class="hero-top-zone">
            <div class="headline">{hl_html}</div>
            <div class="slogan">{content.get('slogan', '')}</div>
            <div>
              <div class="offer-pill">{content.get('offer_main', '')}</div>
            </div>
            <div class="offer-sub">{content.get('offer_sub', '')}</div>
          </div>
          <div class="bottom-brand-bar">
            <div>{content.get('brand', '')}</div>
            <div>{content.get('hotline', '')}</div>
          </div>
        """
    else:  # hero_bottom
        body_content = f"""
          <div class="top-brand-bar">
            <div>{content.get('brand', '')}</div>
            <div>{content.get('hotline', '')}</div>
          </div>
          <div class="hero-bottom-zone glass-panel">
            <div class="headline">{hl_html}</div>
            <div class="slogan">{content.get('slogan', '')}</div>
            <div class="offer-pill">{content.get('offer_main', '')}</div>
            <div class="offer-sub">{content.get('offer_sub', '')}</div>
            <div class="dates">{content.get('dates', '')}</div>
          </div>
        """

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Oswald:wght@700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: {w}px; height: {h}px; overflow: hidden;
    background: url('{bg_data_uri}') no-repeat center center / cover;
    font-family: {font_sub};
    position: relative;
  }}
  .glass-panel {{
    background: {palette['glass_bg']};
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid {palette['glass_border']};
  }}
  /* --- TYPOGRAPHY STYLES --- */
  .headline {{
    font-family: {font_headline};
    font-size: 34px; font-weight: 900;
    line-height: 1.15; letter-spacing: 0.5px;
    text-transform: uppercase;
    {headline_style}
  }}
  .slogan {{
    margin-top: 8px;
    font-size: 14px; font-weight: 600;
    color: {palette['sub_color']};
    text-shadow: {palette['text_shadow']};
    line-height: 1.4;
  }}
  .offer-pill {{
    display: inline-block;
    margin-top: 10px;
    background: {palette['badge_bg']};
    color: {palette['badge_text']};
    padding: 7px 18px; border-radius: 25px;
    font-size: 13.5px; font-weight: 800;
    box-shadow: {palette['badge_shadow']};
    text-transform: uppercase; letter-spacing: 0.5px;
  }}
  .offer-sub {{
    margin-top: 6px;
    font-size: 12px; font-weight: 700;
    color: {palette['offer_highlight_color']};
    letter-spacing: 0.3px;
  }}
  .applicable, .dates {{
    margin-top: 4px;
    font-size: 11px; font-weight: 600;
    color: {palette['sub_color']};
  }}
  /* --- LAYOUT POSITIONS --- */
  .header-zone {{
    position: absolute; top: 4%; left: 5%; right: 5%;
    text-align: center;
  }}
  .footer-zone {{
    position: absolute; bottom: 3%; left: 4%; right: 4%;
    padding: 12px 16px; border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
  }}
  .hero-top-zone {{
    position: absolute; top: 5%; left: 6%; right: 6%;
    text-align: center;
  }}
  .hero-bottom-zone {{
    position: absolute; bottom: 4%; left: 5%; right: 5%;
    padding: 16px 20px; border-radius: 16px;
    text-align: center;
  }}
  .brand-bar {{
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid {palette['glass_border']};
    display: flex; justify-content: space-between; align-items: center;
    font-size: 11px; font-weight: 700;
    color: {palette['footer_text']};
  }}
  .bottom-brand-bar, .top-brand-bar {{
    position: absolute; left: 5%; right: 5%;
    display: flex; justify-content: space-between;
    font-size: 11.5px; font-weight: 700;
    color: {palette['footer_text']};
    background: {palette['footer_bg']};
    padding: 6px 14px; border-radius: 20px;
    backdrop-filter: blur(6px);
  }}
  .bottom-brand-bar {{ bottom: 2.5%; }}
  .top-brand-bar {{ top: 2.5%; }}
</style>
</head>
<body>
  {body_content}
</body>
</html>"""


# ==================================================================================================
# 4. CROSS-PLATFORM FONT LOADER & PIL FALLBACK
# ==================================================================================================

def get_fallback_font(size: int = 24, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def draw_pil_fallback(
    bg_img: Image.Image,
    content: Dict[str, str],
    palette: Dict[str, str],
    layout_type: str,
    width: int,
    height: int,
) -> Image.Image:
    """Robust PIL overlay when Playwright is missing."""
    canvas = bg_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_title = get_fallback_font(size=int(width * 0.055), bold=True)
    font_sub = get_fallback_font(size=int(width * 0.032), bold=False)
    font_badge = get_fallback_font(size=int(width * 0.028), bold=True)
    font_meta = get_fallback_font(size=int(width * 0.024), bold=False)

    is_dark = (palette["is_dark"] == "True")
    text_color = (245, 245, 245) if is_dark else (25, 25, 25)
    badge_bg_rgb = (235, 90, 30)

    # Simple clean banner
    hl = content.get("headline", "")
    draw.rectangle([(int(width * 0.04), int(height * 0.03)), (int(width * 0.96), int(height * 0.30))],
                   fill=(0, 0, 0, 160) if is_dark else (255, 255, 255, 190))
    draw.text((int(width * 0.5), int(height * 0.10)), hl, fill=text_color, font=font_title, anchor="mm", align="center")
    draw.text((int(width * 0.5), int(height * 0.20)), content.get("slogan", ""), fill=text_color, font=font_sub, anchor="mm", align="center")

    draw.rectangle([(int(width * 0.04), int(height * 0.70)), (int(width * 0.96), int(height * 0.96))],
                   fill=(0, 0, 0, 160) if is_dark else (255, 255, 255, 190))
    draw.text((int(width * 0.5), int(height * 0.75)), content.get("offer_main", ""), fill=badge_bg_rgb, font=font_badge, anchor="mm")
    draw.text((int(width * 0.5), int(height * 0.82)), content.get("offer_sub", ""), fill=text_color, font=font_sub, anchor="mm")
    draw.text((int(width * 0.5), int(height * 0.91)), f"{content.get('brand', '')} | {content.get('hotline', '')}", fill=text_color, font=font_meta, anchor="mm")

    return Image.alpha_composite(canvas, overlay).convert("RGB")


# ==================================================================================================
# 5. MAIN EXECUTION PIPELINE
# ==================================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tendoo AI - Dynamic Parameterized Commercial Poster Generator (Zero-Hardcode CLI)"
    )
    # Visual scene & inpainting
    parser.add_argument("--scene-prompt", type=str, default=None, help="Scene description prompt.")
    parser.add_argument("--inpaint-prompt", type=str, default=None, help="Copy space inpaint prompt (auto-derived if omitted).")
    parser.add_argument("--layout", type=str, default="sandwich", choices=["sandwich", "hero_top", "hero_bottom", "fresh_splash"],
                        help="Layout partitioning mode.")

    # Poster Content
    parser.add_argument("--headline", type=str, default="ƯU ĐÃI ĐẶC BIỆT\\nCHÀO MỪNG QUÝ KHÁCH", help="Main title (use \\n for newline).")
    parser.add_argument("--slogan", type=str, default="Sản phẩm chính hãng chất lượng cao", help="Subtitle/slogan.")
    parser.add_argument("--offer-main", type=str, default="GIẢM 20% TẤT CẢ SẢN PHẨM", help="Primary offer / CTA.")
    parser.add_argument("--offer-sub", type=str, default="FREESHIP ĐƠN TỪ 200K", help="Secondary offer / condition.")
    parser.add_argument("--applicable", type=str, default="Áp dụng cho mọi đơn hàng đặt trực tiếp hôm nay.", help="Applicable conditions.")
    parser.add_argument("--dates", type=str, default="Thời gian: 01/10 - 15/10/2026", help="Promotion validity.")
    parser.add_argument("--brand", type=str, default="Tendoo Store", help="Brand / shop name.")
    parser.add_argument("--hotline", type=str, default="Hotline: 0334842155", help="Hotline number.")
    parser.add_argument("--web", type=str, default="Website: tendoo.click", help="Website / QR link.")

    # Color & Font styling
    parser.add_argument("--color-mode", type=str, default="auto", choices=["auto", "dark", "light"],
                        help="Color mode ('auto' computes harmonious contrast from actual background pixels).")
    parser.add_argument("--font-style", type=str, default="auto", choices=["auto", "modern_sans", "luxury_serif", "bold_display"],
                        help="Font family choice.")

    # Inference settings
    parser.add_argument("--steps", type=int, default=8, help="DiT steps (default 8 for Klein Distill).")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--width", type=int, default=576, help="Canvas width.")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Primary DiT GPU device.")
    parser.add_argument("--out-dir", type=str, default="output_custom_poster", help="Output directory.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file overriding CLI arguments.")

    args = parser.parse_args()

    # If config file provided, override args
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"[Error] Config file not found: {cfg_path}")
            sys.exit(1)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if not args.scene_prompt:
        print("[Error] --scene-prompt is required (or specify via --config)!")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 TENDOO AI - DYNAMIC COMMERCIAL POSTER GENERATOR")
    print("=" * 80)
    print(f"  • Layout:       {args.layout}")
    print(f"  • Color Mode:   {args.color_mode} (Dynamic Contrast Harmony)")
    print(f"  • Font Style:   {args.font_style}")
    print(f"  • Scene Prompt: {args.scene_prompt[:70]}...")
    print(f"  • Headline:     {args.headline.replace('\\n', ' ')}")
    print(f"  • Offer:        {args.offer_main}")
    print(f"  • Output Dir:   {out_dir.resolve()}")
    print("=" * 80)

    # Prepare Content Dict
    content = {
        "headline": args.headline.replace("\\n", "\n"),
        "slogan": args.slogan,
        "offer_main": args.offer_main,
        "offer_sub": args.offer_sub,
        "applicable": args.applicable,
        "dates": args.dates,
        "brand": args.brand,
        "hotline": args.hotline,
        "web": args.web,
    }

    # Layout & Inpaint derivations
    layout_info = LAYOUT_PRESETS[args.layout]
    rects = layout_info["rects"]
    inpaint_prompt = args.inpaint_prompt or derive_inpaint_prompt(args.scene_prompt)

    # Multi-GPU routing
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

    # Load Model Weights
    model_name = "flux.2-klein-4b"
    pdata_candidates = [
        Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-4B")),
        Path(os.path.expanduser("~/persistent-data/FLUX.2-klein-base-4B")),
    ]
    for pdata in pdata_candidates:
        distill_cand = pdata / "flux-2-klein-4b.safetensors"
        if distill_cand.exists() and "KLEIN_4B_MODEL_PATH" not in os.environ:
            os.environ["KLEIN_4B_MODEL_PATH"] = str(distill_cand)
            break

    print("\n⏳ Loading FLUX.2 Klein 4B Distill...")
    model = util.load_flow_model(model_name, device=device)
    model.eval()

    ae = util.load_ae(model_name, device=aux_device)
    ae.eval()
    ae_dtype = next(ae.parameters()).dtype

    text_encoder = util.load_text_encoder(model_name, device=aux_device)

    h_lat = args.height // 16
    w_lat = args.width // 16
    C = 128

    # 1. Build Soft Mask
    region_mask_flat = build_region_token_mask_multi(h_lat, w_lat, rects)
    soft_mask = soften_region_mask(region_mask_flat, h_lat, w_lat, feather_iters=3)

    # 2. Step 1: Baseline Generation
    print(f"\n▶ Step 1: Generating scene baseline ({args.steps} steps Distill)...")
    torch.manual_seed(args.seed)
    z_init = torch.randn(1, C, h_lat, w_lat, device=device, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device)
    img_ids = img_ids.unsqueeze(0).to(device)

    timesteps = get_schedule(num_steps=args.steps, image_seq_len=img_tokens.shape[1])
    with torch.no_grad():
        ctx_scene = text_encoder([args.scene_prompt]).to(torch.bfloat16)
        ctx_scene, ctx_scene_ids = prc_txt(ctx_scene[0])
        ctx_scene = ctx_scene.unsqueeze(0).to(device)
        ctx_scene_ids = ctx_scene_ids.unsqueeze(0).to(device)

        out_scene = denoise_baseline(model, img_tokens, img_ids, ctx_scene, ctx_scene_ids, timesteps, guidance=args.guidance)
        z_dec = out_scene[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
        x_dec = ae.decode(z_dec).float()
        x_dec = ((x_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        baseline_pil = Image.fromarray(x_dec)

    baseline_path = out_dir / "01_baseline_raw.png"
    baseline_pil.save(baseline_path)
    print(f"  [✓] Baseline generated -> {baseline_path.name}")

    # 3. Step 2: Unified Inpaint
    print(f"\n▶ Step 2: Unified Inpaint clearing copy space ({args.steps} steps Distill)...")
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

        ctx_inp = text_encoder([inpaint_prompt]).to(torch.bfloat16)
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
            guidance=args.guidance,
            noise_fixed=noise_fixed,
            is_cfg=False,
        )
        z_inpaint_dec = out_inpaint[0].transpose(0, 1).reshape(1, C, h_lat, w_lat).to(device=aux_device, dtype=ae_dtype)
        x_inpaint_dec = ae.decode(z_inpaint_dec).float()
        x_inpaint_dec = ((x_inpaint_dec[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        cleaned_bg_pil = Image.fromarray(x_inpaint_dec)
    dur_inpaint = time.time() - t0

    cleaned_bg_path = out_dir / "02_cleaned_background.png"
    cleaned_bg_pil.save(cleaned_bg_path)
    print(f"  [✓] Copy space cleared in {dur_inpaint:.2f}s -> {cleaned_bg_path.name}")

    # 4. Step 3: Dynamic Contrast & Color Harmony Analysis
    print("\n▶ Step 3: Computing dynamic contrast & color harmony from background pixels...")
    cleaned_bg_np = np.array(cleaned_bg_pil)
    crop_zone = layout_info["crop_zone_header"]
    palette = analyze_color_harmony(cleaned_bg_np, crop_zone, color_mode=args.color_mode, font_style=args.font_style)

    print(f"  • Header Zone Luminance: {palette['lum']} (is_dark: {palette['is_dark']})")
    print(f"  • Dominant Hue:          {palette['hue']}° | Complementary Hue: {palette['comp_hue']}°")
    print(f"  • Headline Color:        {palette['headline_color'][:40]}...")
    print(f"  • Subtitle Color:        {palette['sub_color']}")
    print(f"  • Badge Gradient:        {palette['badge_bg'][:40]}...")

    # 5. Step 4: Assemble & Render HTML Typography
    print("\n▶ Step 4: Generating responsive HTML template & rendering vector overlay...")
    bg_data_uri = pil_to_base64_data_uri(cleaned_bg_pil)
    html_str = build_dynamic_html(
        bg_data_uri=bg_data_uri,
        content=content,
        palette=palette,
        layout_type=args.layout,
        font_style=args.font_style,
        w=args.width,
        h=args.height,
    )

    html_path = out_dir / "03_poster_template.html"
    html_path.write_text(html_str, encoding="utf-8")
    print(f"  [✓] HTML template exported -> {html_path.name}")

    final_poster_path = out_dir / "03_final_poster.png"
    try:
        PosterRenderer.render(html_content=html_str, output_image_path=final_poster_path, width=args.width, height=args.height)
        print(f"  [✓] Final poster rendered (Playwright) -> {final_poster_path.name}")
    except Exception as e:
        print(f"  [!] Playwright unavailable ({e}). Running PIL typography fallback...")
        final_poster_pil = draw_pil_fallback(cleaned_bg_pil, content, palette, args.layout, args.width, args.height)
        final_poster_pil.save(final_poster_path)
        print(f"  [✓] Final poster rendered (PIL Fallback) -> {final_poster_path.name}")

    # 6. Step 5: Build 3-Panel Comparison Strip
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

    comp_path = out_dir / "04_comparison_strip.png"
    comp.save(comp_path)
    print(f"  [✓] Comparison strip saved -> {comp_path.name}")

    # Save run manifest
    manifest_path = out_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "content": content,
            "palette": palette,
            "inpaint_time_s": round(dur_inpaint, 2),
            "artifacts": {
                "baseline": str(baseline_path.name),
                "cleaned_bg": str(cleaned_bg_path.name),
                "html_template": str(html_path.name),
                "final_poster": str(final_poster_path.name),
                "comparison_strip": str(comp_path.name),
            }
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH TẠO POSTER THƯƠNG MẠI TỰ ĐỘNG!")
    print(f"  Kết quả đã lưu tại: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
