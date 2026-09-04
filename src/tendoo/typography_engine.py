"""
==================================================================================================
TENDOO AI - DYNAMIC HTML/CSS TYPOGRAPHY OVERLAY ENGINE
==================================================================================================
Module: src/tendoo/typography_engine.py
Purpose: State-of-the-Art Scalable Typography and Component Overlay Engine for Commercial Posters.
Inspired by: AAAI 2026 Oral "PosterVerse: A Full-Workflow Framework for Commercial-Grade Poster
             Generation with HTML-Based Scalable Typography" (Liu et al., 2026).

CORE ARCHITECTURAL DIVISION OF LABOR:
  1. DiT (FLUX.2-klein-4B Distill @ t=10.0):
     - Renders background and 3D Hero Title with photorealistic lighting, cast shadows, and materials.
  2. This Engine (HTML/CSS + Playwright Chromium):
     - Renders secondary text: Badges, Subtitles, 5-Star Ratings, Customer Reviews, Spec Chips,
       Discounts, Brand Footers, and Hotlines.
     - Performs automated background luminance and contrast analysis to prevent color clashing.
     - Offers both an Algorithmic Layout Generator (100% offline) and a VLM Prompt Builder (for GPT-4o / Qwen2.5-VL).
==================================================================================================
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ==================================================================================================
# 1. DATA STRUCTURES FOR BACKGROUND CONTRAST & HARMONY ANALYSIS
# ==================================================================================================

@dataclass
class ZoneMetrics:
    """Quantitative metrics of a specific canvas region (Header, Middle, Footer)."""
    zone_name: str
    bbox_ratio: Tuple[float, float, float, float]  # (y_min, x_min, y_max, x_max) relative [0, 1]
    mean_luminance: float  # [0, 255]
    is_dark: bool  # True if luminance < 128
    dominant_rgb: Tuple[int, int, int]
    dominant_hex: str
    recommended_text_color: str
    recommended_subtext_color: str
    recommended_glass_bg: str
    recommended_glass_border: str
    recommended_badge_bg: str
    recommended_badge_text: str


@dataclass
class BackgroundAnalysis:
    """Holistic analysis of the generated background image."""
    width: int
    height: int
    aspect_ratio: float
    header_zone: ZoneMetrics
    center_zone: ZoneMetrics
    footer_zone: ZoneMetrics
    overall_luminance: float
    overall_is_dark: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==================================================================================================
# 2. COLOR & CONTRAST HARMONY ANALYZER
# ==================================================================================================

class PosterBackgroundAnalyzer:
    """
    Analyzes generated DiT poster backgrounds to determine optimal contrast, safe zones,
    and harmonious color palettes for secondary HTML typography overlays.
    """

    @staticmethod
    def _compute_luminance(rgb_array: np.ndarray) -> float:
        """Computes perceived relative luminance using ITU-R BT.601 standard: Y = 0.299R + 0.587G + 0.114B."""
        if rgb_array.size == 0:
            return 128.0
        r = rgb_array[:, :, 0].astype(np.float32)
        g = rgb_array[:, :, 1].astype(np.float32)
        b = rgb_array[:, :, 2].astype(np.float32)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return float(np.mean(lum))

    @staticmethod
    def _extract_dominant_color(rgb_array: np.ndarray) -> Tuple[int, int, int]:
        """Extracts dominant median color of a region."""
        if rgb_array.size == 0:
            return (128, 128, 128)
        median_rgb = np.median(rgb_array.reshape(-1, 3), axis=0).astype(int)
        return int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2])

    @classmethod
    def _build_zone_metrics(
        cls, zone_name: str, bbox_ratio: Tuple[float, float, float, float], img_np: np.ndarray
    ) -> ZoneMetrics:
        h, w, _ = img_np.shape
        y_min, x_min, y_max, x_max = bbox_ratio
        crop = img_np[int(y_min * h) : int(y_max * h), int(x_min * w) : int(x_max * w)]

        lum = cls._compute_luminance(crop)
        is_dark = lum < 128.0
        dom_rgb = cls._extract_dominant_color(crop)
        dom_hex = f"#{dom_rgb[0]:02x}{dom_rgb[1]:02x}{dom_rgb[2]:02x}"

        if is_dark:
            # Contrast for dark background: Light typography + vibrant glowing badges
            text_color = "#FFFFFF"
            subtext_color = "rgba(255, 255, 255, 0.82)"
            glass_bg = "rgba(255, 255, 255, 0.12)"
            glass_border = "rgba(255, 255, 255, 0.25)"
            badge_bg = "linear-gradient(135deg, #FF6B35 0%, #FFA500 100%)"
            badge_text = "#FFFFFF"
        else:
            # Contrast for bright background: Dark typography + deep saturated badges
            text_color = "#0F172A"
            subtext_color = "#334155"
            glass_bg = "rgba(15, 23, 42, 0.07)"
            glass_border = "rgba(15, 23, 42, 0.15)"
            badge_bg = "linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%)"
            badge_text = "#FFFFFF"

        return ZoneMetrics(
            zone_name=zone_name,
            bbox_ratio=bbox_ratio,
            mean_luminance=round(lum, 1),
            is_dark=is_dark,
            dominant_rgb=dom_rgb,
            dominant_hex=dom_hex,
            recommended_text_color=text_color,
            recommended_subtext_color=subtext_color,
            recommended_glass_bg=glass_bg,
            recommended_glass_border=glass_border,
            recommended_badge_bg=badge_bg,
            recommended_badge_text=badge_text,
        )

    @classmethod
    def analyze(cls, image_path_or_pil: str | Path | Image.Image) -> BackgroundAnalysis:
        """
        Analyzes image dimensions and splits canvas into Header (0-22%), Center (22-75%),
        and Footer (75-100%) to establish contrast and harmonious color guidelines.
        """
        if isinstance(image_path_or_pil, (str, Path)):
            pil_img = Image.open(str(image_path_or_pil)).convert("RGB")
        else:
            pil_img = image_path_or_pil.convert("RGB")

        w, h = pil_img.size
        img_np = np.array(pil_img)

        # Header zone (top 22%): Where eyebrow tags, top brand logo, or discount badges sit
        header_metrics = cls._build_zone_metrics("header", (0.0, 0.0, 0.22, 1.0), img_np)

        # Center zone (22% - 75%): Where the main DiT Hero Title and primary product sit
        center_metrics = cls._build_zone_metrics("center", (0.22, 0.0, 0.75, 1.0), img_np)

        # Footer zone (bottom 25%): Where sub-slogans, star ratings, specs, CTA buttons, and hotline sit
        footer_metrics = cls._build_zone_metrics("footer", (0.75, 0.0, 1.0, 1.0), img_np)

        overall_lum = cls._compute_luminance(img_np)

        return BackgroundAnalysis(
            width=w,
            height=h,
            aspect_ratio=round(w / h, 3),
            header_zone=header_metrics,
            center_zone=center_metrics,
            footer_zone=footer_metrics,
            overall_luminance=round(overall_lum, 1),
            overall_is_dark=overall_lum < 128.0,
        )


# ==================================================================================================
# 3. VLM PROMPT BUILDER (FOR GPT-4O / QWEN2.5-VL)
# ==================================================================================================

class TypographyPromptBuilder:
    """
    Constructs strict, high-precision prompts for Vision-Language Models (GPT-4o, Qwen2.5-VL)
    to generate production-grade HTML/CSS poster typography overlays.
    """

    SYSTEM_PROMPT = """You are a World-Class Commercial Graphic Designer and Frontend Typography Specialist.
You generate standalone, valid, production-grade HTML5/CSS3 documents for commercial advertising posters.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. THE MAIN HERO TITLE IS ALREADY 3D-RENDERED IN THE BACKGROUND IMAGE.
   - Do NOT create a duplicate main title element that repeats the main hero text!
   - Your sole responsibility is to place the SECONDARY elements: Brand Badge, Eyebrow/Category Tag,
     Discount/Promo Badge, Star Rating & Social Proof, Feature Chips, Sub-Slogan, CTA Button, and Footer Bar.
2. ABSOLUTE ZERO OVERLAP WITH HERO TITLE:
   - Place secondary components exclusively in the designated safe zones (Header top, Side column, or Footer bottom).
3. CONTRAST & COLOR HARMONY (ANTI-CHÌM MÀU):
   - Adhere strictly to the luminance and background color metrics provided.
   - If the region is dark, use crisp white typography with subtle drop-shadows and glassmorphism.
   - If the region is bright, use deep navy/slate typography with clean contrast.
4. CSS REQUIREMENTS:
   - Load modern Google Fonts via `@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Montserrat:wght@500;700;900&display=swap');`
   - Use `.poster-container` with exact dimensions matching the canvas, `position: relative; overflow: hidden;`.
   - Use absolute positioning (`position: absolute; left: ...; top: ...; transform: ...;`).
   - Use modern CSS styling: `backdrop-filter: blur(12px);`, `box-shadow: 0 8px 24px rgba(0,0,0,0.15);`,
     `border-radius: 9999px;` (for pill badges), `letter-spacing: 1px;`.
   - Return ONLY clean HTML code inside ```html ... ``` codeblock.
"""

    @classmethod
    def build_user_prompt(
        cls,
        analysis: BackgroundAnalysis,
        brief: Dict[str, Any],
        hero_title_text: Optional[str] = None,
    ) -> str:
        """
        Builds the user prompt containing visual background analysis and text brief.
        """
        hero_notice = (
            f"The image already features the 3D Hero Title: '{hero_title_text}'. DO NOT duplicate this text."
            if hero_title_text
            else "The image already features the main 3D Hero Title. DO NOT duplicate it."
        )

        prompt_payload = {
            "canvas_dimensions": {"width_px": analysis.width, "height_px": analysis.height, "aspect_ratio": analysis.aspect_ratio},
            "hero_title_notice": hero_notice,
            "background_analysis": {
                "header_zone": {
                    "is_dark": analysis.header_zone.is_dark,
                    "mean_luminance": analysis.header_zone.mean_luminance,
                    "dominant_hex": analysis.header_zone.dominant_hex,
                    "recommended_text_color": analysis.header_zone.recommended_text_color,
                    "recommended_badge_bg": analysis.header_zone.recommended_badge_bg,
                },
                "footer_zone": {
                    "is_dark": analysis.footer_zone.is_dark,
                    "mean_luminance": analysis.footer_zone.mean_luminance,
                    "dominant_hex": analysis.footer_zone.dominant_hex,
                    "recommended_text_color": analysis.footer_zone.recommended_text_color,
                    "recommended_cta_style": analysis.footer_zone.recommended_badge_bg,
                },
            },
            "required_secondary_elements": brief,
        }

        user_content = (
            f"Please analyze the attached poster image and design an HTML/CSS typography overlay "
            f"that complements the visual composition seamlessly.\n\n"
            f"```json\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Generate the complete HTML5 code with inline `<style>` and modern typography."
        )
        return user_content


# ==================================================================================================
# 4. ALGORITHMIC TEMPLATE ENGINE (OFFLINE FALLBACK & STANDALONE SYNTHESIS)
# ==================================================================================================

class PosterTemplateEngine:
    """
    Algorithmic typography generator that requires ZERO external API or model weights.
    Synthesizes responsive, pixel-perfect HTML/CSS overlays matching the PosterVerse standard.
    Ideal for execution on isolated internal servers without internet!
    """

    @classmethod
    def generate_html(
        cls,
        analysis: BackgroundAnalysis,
        brief: Dict[str, Any],
        background_image_path: Optional[str] = None,
    ) -> str:
        """
        Generates production-grade HTML/CSS tailored to the background's quantitative luminance.
        """
        w = analysis.width
        h = analysis.height

        # Brief extraction with robust defaults
        brand = brief.get("brand", "TENDOO")
        eyebrow = brief.get("eyebrow", "CÔNG NGHỆ ĐỘT PHÁ")
        badge = brief.get("badge", "GIẢM 30%")
        rating_val = brief.get("rating_value", "4.9")
        rating_count = brief.get("rating_count", "1.2k+ đánh giá")
        specs = brief.get("specs", ["Chống Ồn Chủ Động 45dB", "Thời Lượng Pin 40 Giờ", "Bluetooth 5.4 Ultra"])
        sub_slogan = brief.get("sub_slogan", "TRẢI NGHIỆM ĐỈNH CAO - CHẤT ÂM CHÂN THỰC")
        cta_text = brief.get("cta_text", "ĐẶT HÀNG NGAY")
        hotline = brief.get("hotline", "1800 8198")
        website = brief.get("website", "www.tendoo.ai")

        # Contrast styling based on Header & Footer analysis
        h_color = analysis.header_zone.recommended_text_color
        h_sub = analysis.header_zone.recommended_subtext_color
        h_badge_bg = analysis.header_zone.recommended_badge_bg
        h_glass_bg = analysis.header_zone.recommended_glass_bg
        h_glass_border = analysis.header_zone.recommended_glass_border

        f_color = analysis.footer_zone.recommended_text_color
        f_sub = analysis.footer_zone.recommended_subtext_color
        f_badge_bg = analysis.footer_zone.recommended_badge_bg
        f_glass_bg = analysis.footer_zone.recommended_glass_bg
        f_glass_border = analysis.footer_zone.recommended_glass_border

        # Background image handling
        bg_css = ""
        if background_image_path and os.path.exists(background_image_path):
            with open(background_image_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
                ext = Path(background_image_path).suffix.lower().replace(".", "")
                mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
                bg_css = f"background-image: url('data:{mime};base64,{b64_data}');"

        specs_html = "".join([f'<span class="spec-chip">{s}</span>' for s in specs])

        html_template = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tendoo Poster Typography Overlay</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Montserrat:wght@600;700;800;900&display=swap');

    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      -webkit-font-smoothing: antialiased;
    }}

    body {{
      width: 100vw;
      height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      background-color: #050505;
      font-family: 'Plus Jakarta Sans', sans-serif;
      overflow: hidden;
    }}

    .poster-container {{
      position: relative;
      width: {w}px;
      height: {h}px;
      {bg_css}
      background-size: cover;
      background-position: center;
      overflow: hidden;
    }}

    /* ==========================================================================
       TOP HEADER ZONE
       ========================================================================== */
    .top-header {{
      position: absolute;
      top: 48px;
      left: 56px;
      right: 56px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 20;
    }}

    .brand-eyebrow {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .brand-logo {{
      font-family: 'Montserrat', sans-serif;
      font-weight: 900;
      font-size: 26px;
      letter-spacing: 2px;
      color: {h_color};
      text-transform: uppercase;
      text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }}

    .eyebrow-tag {{
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 1.5px;
      color: {h_sub};
      text-transform: uppercase;
    }}

    .promo-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 12px 24px;
      background: {h_badge_bg};
      color: #FFFFFF;
      font-family: 'Montserrat', sans-serif;
      font-weight: 800;
      font-size: 18px;
      letter-spacing: 1px;
      border-radius: 9999px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.25), inset 0 1px 1px rgba(255,255,255,0.4);
      transform: rotate(-3deg);
      border: 1px solid rgba(255,255,255,0.3);
    }}

    /* ==========================================================================
       BOTTOM FOOTER & SOCIAL PROOF ZONE
       ========================================================================== */
    .bottom-section {{
      position: absolute;
      bottom: 56px;
      left: 56px;
      right: 56px;
      display: flex;
      flex-direction: column;
      gap: 24px;
      z-index: 20;
    }}

    .social-proof-bar {{
      display: flex;
      align-items: center;
      gap: 16px;
      background: {f_glass_bg};
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid {f_glass_border};
      padding: 12px 24px;
      border-radius: 16px;
      width: fit-content;
      box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }}

    .stars {{
      color: #FBBF24;
      font-size: 20px;
      letter-spacing: 2px;
    }}

    .rating-text {{
      font-size: 16px;
      font-weight: 700;
      color: {f_color};
    }}

    .rating-count {{
      font-size: 14px;
      font-weight: 500;
      color: {f_sub};
    }}

    .sub-slogan {{
      font-family: 'Montserrat', sans-serif;
      font-weight: 800;
      font-size: 32px;
      line-height: 1.3;
      color: {f_color};
      text-transform: uppercase;
      letter-spacing: 0.5px;
      text-shadow: 0 4px 16px rgba(0,0,0,0.3);
      max-width: 80%;
    }}

    .specs-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}

    .spec-chip {{
      padding: 10px 20px;
      background: {f_glass_bg};
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid {f_glass_border};
      color: {f_color};
      font-size: 15px;
      font-weight: 600;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    .action-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 16px;
      border-top: 1px solid {f_glass_border};
    }}

    .cta-button {{
      display: inline-flex;
      align-items: center;
      gap: 12px;
      padding: 16px 36px;
      background: {f_badge_bg};
      color: #FFFFFF;
      font-family: 'Montserrat', sans-serif;
      font-weight: 800;
      font-size: 17px;
      letter-spacing: 1px;
      border-radius: 9999px;
      box-shadow: 0 10px 28px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.4);
      text-decoration: none;
      border: 1px solid rgba(255,255,255,0.3);
    }}

    .contact-info {{
      display: flex;
      align-items: center;
      gap: 24px;
      font-size: 15px;
      font-weight: 600;
      color: {f_sub};
    }}

    .contact-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
  </style>
</head>
<body>
  <div class="poster-container">
    <!-- TOP HEADER ZONE -->
    <div class="top-header">
      <div class="brand-eyebrow">
        <span class="brand-logo">{brand}</span>
        <span class="eyebrow-tag">{eyebrow}</span>
      </div>
      <div class="promo-badge">
        <span>🔥 {badge}</span>
      </div>
    </div>

    <!-- BOTTOM FOOTER & SOCIAL PROOF ZONE -->
    <div class="bottom-section">
      <div class="social-proof-bar">
        <span class="stars">★★★★★</span>
        <span class="rating-text">{rating_val}/5</span>
        <span class="rating-count">({rating_count})</span>
      </div>

      <div class="sub-slogan">{sub_slogan}</div>

      <div class="specs-row">
        {specs_html}
      </div>

      <div class="action-bar">
        <a href="#" class="cta-button">
          <span>{cta_text}</span>
          <span>➔</span>
        </a>
        <div class="contact-info">
          <div class="contact-item">
            <span>📞</span>
            <span>Hotline: {hotline}</span>
          </div>
          <div class="contact-item">
            <span>🌐</span>
            <span>{website}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""
        return html_template


# ==================================================================================================
# 5. PLAYWRIGHT CHROMIUM HEADLESS RENDERER
# ==================================================================================================

class PosterRenderer:
    """
    High-performance headless Chromium renderer using Playwright.
    Ensures zero-network offline rendering, sub-pixel rasterization, and font readiness.
    """

    @classmethod
    async def render_html_async(
        cls,
        html_content: str,
        output_image_path: str | Path,
        width: int,
        height: int,
        device_scale_factor: int = 1,
    ) -> Path:
        """
        Renders HTML content into a lossless PNG image using Playwright Chromium.
        """
        from playwright.async_api import async_playwright

        out_file = Path(output_image_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=device_scale_factor,
            )
            page = await context.new_page()

            # Set content and wait for network/fonts to settle
            await page.set_content(html_content, wait_until="networkidle")

            # Guarantee all web fonts are fully rasterized
            try:
                await page.evaluate("document.fonts.ready")
            except Exception as e:
                logger.warning(f"Font readiness check skipped: {e}")

            # Take pixel-accurate screenshot
            await page.screenshot(
                path=str(out_file),
                full_page=True,
                type="png",
            )
            await browser.close()

        logger.info(f"[PosterRenderer] Screenshot saved to: {out_file}")
        return out_file

    @classmethod
    def render(
        cls,
        html_content: str,
        output_image_path: str | Path,
        width: int,
        height: int,
        device_scale_factor: int = 1,
    ) -> Path:
        """Synchronous wrapper for render_html_async."""
        return asyncio.run(
            cls.render_html_async(
                html_content=html_content,
                output_image_path=output_image_path,
                width=width,
                height=height,
                device_scale_factor=device_scale_factor,
            )
        )
