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
from string import Template
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
#
# TEMPLATE LIBRARY, NOT LIVE GENERATION: each category gets a hand-designed, QA'd-once layout
# (originally authored in scripts/demo_diverse_html_cases.py as 4 standalone flat-CSS mockups,
# migrated here and adapted to composite over a REAL diffusion-generated background instead of a
# flat gradient). Stage 1 (the blueprint/hero-selector LLM call, see scripts/test_hero_selector.py)
# picks WHICH template via `category` -- this is a free extra field on an already-happening call,
# not an additional model invocation. Live VLM HTML generation (TypographyPromptBuilder + api_*
# modes below) is reserved for DRAFTING new templates offline, reviewed by a human before being
# hardened into this registry -- never called live in the production hot path. See AGENTS.md
# discussion: this mirrors how PosterVerse's own PosterDNA dataset was built (LLM draft -> human
# correction -> trusted asset), just without committing to fine-tuning anything ourselves.
#
# IMPORTANT: each template here renders ONLY secondary elements (badges, quotes, chips, footers).
# The HERO text is NEVER re-created here -- it was already baked into `background_image_path` by
# FLUX.2 glyph injection (photoreal 3D/material integration). Duplicating it here would violate
# the same "don't re-render what diffusion already drew" rule as the VLM system prompt below.

def _bg_image_css(background_image_path: Optional[str]) -> str:
    """Shared helper: embeds a background image as base64 CSS, or returns '' if none given."""
    if background_image_path and os.path.exists(background_image_path):
        with open(background_image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
            ext = Path(background_image_path).suffix.lower().replace(".", "")
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
            return f"background-image: url('data:{mime};base64,{b64_data}'); background-size: cover; background-position: center;"
    return ""


class PosterTemplateEngine:
    """
    Algorithmic typography generator that requires ZERO external API or model weights.
    Synthesizes responsive, pixel-perfect HTML/CSS overlays matching the PosterVerse standard.
    Ideal for execution on isolated internal servers without internet!

    `generate_html(category=...)` dispatches into a small TEMPLATE LIBRARY (grand_opening,
    feedback, recruitment, menu) instead of one fixed layout -- pass the category chosen by the
    Stage 1 blueprint. Falls back to the original generic layout for "generic" or any unknown
    category, so existing callers (e.g. test_poster_typography_overlay.py's default flow) are
    unaffected.
    """

    @classmethod
    def generate_html(
        cls,
        analysis: BackgroundAnalysis,
        brief: Dict[str, Any],
        background_image_path: Optional[str] = None,
        category: str = "generic",
    ) -> str:
        """
        Dispatches to the template matching (category, orientation), defaulting to the generic
        layout. Orientation is a coarse "portrait" (h > w -- 9:16, 2:3...) vs "landscape/square"
        (w >= h -- 1:1, 4:5, 16:9...) split: these two buckets need genuinely different STRUCTURE
        (single-column stack vs side-by-side/grid), not just proportional scaling of the same
        layout -- see AGENTS.md discussion: fixed-px templates tuned for 1024x1024 broke outright
        (wrapped text, overlapping bands) when reused on 576x1024 or 1024x576 unchanged.
        """
        orientation = "portrait" if analysis.height > analysis.width else "landscape"
        dispatch = {
            ("grand_opening", "landscape"): cls._generate_grand_opening,
            ("grand_opening", "portrait"): cls._generate_grand_opening_portrait,
            ("feedback", "landscape"): cls._generate_feedback_card,
            ("feedback", "portrait"): cls._generate_feedback_card_portrait,
            ("recruitment", "landscape"): cls._generate_recruitment,
            ("recruitment", "portrait"): cls._generate_recruitment_portrait,
            ("menu", "landscape"): cls._generate_menu,
            ("menu", "portrait"): cls._generate_menu_portrait,
        }
        fn = dispatch.get((category, orientation))
        if fn is not None:
            return fn(analysis, brief, background_image_path)
        return cls._generate_generic(analysis, brief, background_image_path)

    @classmethod
    def _generate_generic(
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

    # ----------------------------------------------------------------------------------------
    # TEMPLATE LIBRARY (migrated from scripts/demo_diverse_html_cases.py, adapted to composite
    # over a real background image instead of a flat CSS gradient, and with the flat-CSS "hero"
    # text element removed from each -- that text is already baked into the photo by diffusion).
    # Uses string.Template ($placeholder) instead of f-strings/str.format to avoid having to
    # escape the hundreds of literal CSS "{ }" in each layout.
    # ----------------------------------------------------------------------------------------

    _GRAND_OPENING_TPL = Template("""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Grand Opening</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; font-family: 'Plus Jakarta Sans', sans-serif; }
    .poster {
      position: relative; width: ${w}px; height: ${h}px; overflow: hidden;
      $bg_css
      box-shadow: 0 25px 60px rgba(0,0,0,0.8);
    }
    .header { position: absolute; top: 48px; left: 56px; right: 56px; display: flex; justify-content: space-between; align-items: center; z-index: 20; }
    .brand-title { font-family: 'Montserrat', sans-serif; font-size: 24px; font-weight: 900; color: #FFB703; letter-spacing: 2px; text-transform: uppercase; text-shadow: 0 0 20px rgba(255, 183, 3, 0.5); }
    .date-pill { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(12px); border: 1px solid rgba(255, 183, 3, 0.4); padding: 10px 22px; border-radius: 999px; font-size: 14px; font-weight: 700; color: #FFF; letter-spacing: 1px; }
    .burst-badge {
      position: absolute; top: 220px; right: 70px; width: 170px; height: 170px;
      background: linear-gradient(135deg, #E63946 0%, #D90429 100%); border-radius: 50%;
      display: flex; flex-direction: column; justify-content: center; align-items: center;
      box-shadow: 0 12px 35px rgba(230, 57, 70, 0.6), inset 0 3px 6px rgba(255, 255, 255, 0.5);
      border: 4px dashed #FFF; transform: rotate(12deg); z-index: 25;
    }
    .badge-sub { font-size: 14px; font-weight: 800; color: #FFF; letter-spacing: 2px; text-transform: uppercase; }
    .badge-main { font-family: 'Montserrat', sans-serif; font-size: 52px; font-weight: 900; color: #FFF; line-height: 0.95; }
    .badge-off { font-size: 16px; font-weight: 900; color: #FFD166; letter-spacing: 1.5px; }
    .bottom-bar {
      position: absolute; bottom: 50px; left: 56px; right: 56px; z-index: 20;
      background: rgba(20, 10, 5, 0.75); backdrop-filter: blur(20px); border: 1px solid rgba(255, 183, 3, 0.25);
      border-radius: 24px; padding: 24px 36px; display: flex; justify-content: space-between; align-items: center;
      box-shadow: 0 15px 40px rgba(0,0,0,0.6);
    }
    .deal-info { display: flex; flex-direction: column; gap: 4px; }
    .deal-title { font-family: 'Montserrat', sans-serif; font-size: 20px; font-weight: 800; color: #FFF; }
    .deal-sub { font-size: 14px; font-weight: 500; color: #FFB703; }
    .cta-btn { background: linear-gradient(135deg, #FB8500 0%, #FFB703 100%); color: #000; font-family: 'Montserrat', sans-serif; font-weight: 900; font-size: 17px; letter-spacing: 0.5px; padding: 16px 36px; border-radius: 999px; text-decoration: none; box-shadow: 0 8px 25px rgba(251, 133, 0, 0.5); border: 1px solid rgba(255,255,255,0.4); }
  </style>
</head>
<body>
  <div class="poster">
    <div class="header">
      <div class="brand-title">$brand</div>
      <div class="date-pill">$date_range</div>
    </div>
    <div class="burst-badge">
      <span class="badge-sub">$badge_label</span>
      <span class="badge-main">$badge_percent</span>
      <span class="badge-off">$badge_sub</span>
    </div>
    <div class="bottom-bar">
      <div class="deal-info">
        <div class="deal-title">$address</div>
        <div class="deal-sub">$offer_desc</div>
      </div>
      <a href="#" class="cta-btn">$cta_text</a>
    </div>
  </div>
</body>
</html>""")

    @classmethod
    def _generate_grand_opening(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        return cls._GRAND_OPENING_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            brand=brief.get("brand", "🍔 THE BURGER CRAFT"),
            date_range=brief.get("date_range", "DUY NHẤT 05.09 - 15.09.2026"),
            badge_label=brief.get("badge_label", "GIẢM"),
            badge_percent=brief.get("badge_percent", "50%"),
            badge_sub=brief.get("badge_sub", "TOÀN MENU"),
            address=brief.get("address", "📍 128 Nguyễn Trãi, Phường Bến Thành, Quận 1"),
            offer_desc=brief.get("offer_desc", "Tặng 01 Coca-Cola mát lạnh cho hóa đơn từ 99K • Hotline: 1900 8899"),
            cta_text=brief.get("cta_text", "NHẬN VOUCHER ➔"),
        )

    _FEEDBACK_TPL = Template("""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Customer Feedback</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; font-family: 'Plus Jakarta Sans', sans-serif; }
    .poster { position: relative; width: ${w}px; height: ${h}px; overflow: hidden; $bg_css box-shadow: 0 25px 60px rgba(0,0,0,0.12); border-radius: 32px; }
    /* NOTE: bottom-stack uses flex-column + gap (not per-element top:Npx) precisely so this
       template survives BOTH 1024x1024 (roomy) and shorter landscape canvases like 1024x576
       (16:9) without elements overlapping or being pushed off-canvas -- see AGENTS.md discussion:
       fixed top:250px/top:620px/bottom:50px broke outright once height dropped from 1024 to 576. */
    .top-bar { position: absolute; top: 4%; left: 5.5%; right: 5.5%; display: flex; justify-content: space-between; align-items: center; z-index: 20; }
    .spa-logo { font-family: 'Quicksand', sans-serif; font-size: 26px; font-weight: 800; color: #0E9F6E; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 60%; text-shadow: 0 2px 8px rgba(255,255,255,0.6); }
    .spa-badge { background: #FFE4E6; color: #E02424; font-family: 'Quicksand', sans-serif; font-weight: 800; font-size: 14px; padding: 10px 20px; border-radius: 999px; border: 1px solid #FECDD3; white-space: nowrap; }
    .bottom-stack { position: absolute; bottom: 4%; left: 5.5%; right: 5.5%; display: flex; flex-direction: column; gap: 1.8%; z-index: 20; max-height: 78%; }
    .feedback-card {
      background: rgba(255, 255, 255, 0.82); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border: 2px solid rgba(255, 255, 255, 0.9); border-radius: 24px; padding: 3% 3.5%;
      box-shadow: 0 20px 40px rgba(0, 150, 110, 0.12), 0 1px 3px rgba(0,0,0,0.05);
      display: flex; flex-direction: column; gap: 1.4%;
    }
    .review-header { display: flex; justify-content: space-between; align-items: center; }
    .stars { color: #F59E0B; font-size: 24px; letter-spacing: 3px; }
    .verified-pill { display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 700; color: #057A55; background: #DEF7EC; padding: 5px 12px; border-radius: 999px; white-space: nowrap; }
    .quote-text { font-size: 19px; line-height: 1.5; color: #374151; font-weight: 500; font-style: italic; }
    .customer-info { display: flex; align-items: center; gap: 14px; border-top: 1px solid #E5E7EB; padding-top: 1.2%; }
    .avatar { width: 44px; height: 44px; flex-shrink: 0; border-radius: 50%; background: #D1FAE5; display: flex; justify-content: center; align-items: center; font-size: 22px; border: 2px solid #0E9F6E; }
    .cust-name { font-size: 16px; font-weight: 700; color: #111928; }
    .cust-sub { font-size: 12px; color: #6B7280; font-weight: 500; }
    .features-row { display: flex; justify-content: space-between; gap: 12px; }
    .f-pill { flex: 1; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px; padding: 14px; text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.04); }
    .f-icon { font-size: 24px; margin-bottom: 6px; }
    .f-text { font-size: 12.5px; font-weight: 700; color: #1F2A37; line-height: 1.25; }
    .bottom-cta-strip { background: linear-gradient(135deg, #0E9F6E 0%, #057A55 100%); border-radius: 20px; padding: 2.2% 3%; display: flex; justify-content: space-between; align-items: center; gap: 12px; box-shadow: 0 12px 30px rgba(14, 159, 110, 0.35); }
    .offer-box { color: #FFFFFF; min-width: 0; }
    .offer-title { font-family: 'Quicksand', sans-serif; font-size: 19px; font-weight: 800; }
    .offer-desc { font-size: 13px; opacity: 0.9; margin-top: 2px; }
    .btn-booking { flex-shrink: 0; background: #FFFFFF; color: #046C4E; font-family: 'Quicksand', sans-serif; font-weight: 800; font-size: 15px; padding: 12px 24px; border-radius: 999px; text-decoration: none; white-space: nowrap; }
  </style>
</head>
<body>
  <div class="poster">
    <div class="top-bar">
      <div class="spa-logo">$brand</div>
      <div class="spa-badge">$top_badge</div>
    </div>
    <div class="bottom-stack">
      <div class="feedback-card">
        <div class="review-header">
          <div class="stars">$stars</div>
          <div class="verified-pill">$verified_label</div>
        </div>
        <div class="quote-text">$quote_text</div>
        <div class="customer-info">
          <div class="avatar">$avatar_emoji</div>
          <div>
            <div class="cust-name">$customer_name</div>
            <div class="cust-sub">$customer_sub</div>
          </div>
        </div>
      </div>
      <div class="features-row">
        $features_html
      </div>
      <div class="bottom-cta-strip">
        <div class="offer-box">
          <div class="offer-title">$offer_title</div>
          <div class="offer-desc">$offer_desc</div>
        </div>
        <a href="#" class="btn-booking">$cta_text</a>
      </div>
    </div>
  </div>
</body>
</html>""")

    @classmethod
    def _generate_feedback_card(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        features = brief.get("features", [
            {"icon": "🌿", "text": "Chất Lượng Hữu Cơ 100% Nhập Khẩu"},
            {"icon": "✂️", "text": "Chuyên Nghiệp Theo Yêu Cầu Riêng"},
            {"icon": "🕊️", "text": "Không Gian Mở, Trải Nghiệm Thoải Mái"},
        ])
        features_html = "".join(
            f'<div class="f-pill"><div class="f-icon">{f.get("icon","✨")}</div>'
            f'<div class="f-text">{f.get("text","")}</div></div>'
            for f in features
        )
        return cls._FEEDBACK_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            brand=brief.get("brand", "🐾 PAWPARADISE SPA"),
            top_badge=brief.get("top_badge", "✨ CHUẨN FORM HÀN QUỐC"),
            stars=brief.get("stars", "★★★★★"),
            verified_label=brief.get("verified_label", "✔ ĐÃ TRẢI NGHIỆM DỊCH VỤ"),
            quote_text=brief.get("quote_text", "Dịch vụ tuyệt vời, nhân viên chuyên nghiệp và tận tâm, chắc chắn sẽ quay lại!"),
            avatar_emoji=brief.get("avatar_emoji", "🐩"),
            customer_name=brief.get("customer_name", "Khách hàng thân thiết"),
            customer_sub=brief.get("customer_sub", "Đã trải nghiệm dịch vụ Premium"),
            features_html=features_html,
            offer_title=brief.get("offer_title", "🎁 ƯU ĐÃI ĐẶC BIỆT CHO KHÁCH MỚI"),
            offer_desc=brief.get("offer_desc", "Áp dụng cho khách hàng đặt lịch trải nghiệm lần đầu tiên trong tuần này!"),
            cta_text=brief.get("cta_text", "ĐẶT LỊCH NGAY ➔"),
        )

    _RECRUITMENT_TPL = Template("""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Recruitment</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Montserrat:wght@700;800;900&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; font-family: 'Plus Jakarta Sans', sans-serif; }
    .poster { position: relative; width: ${w}px; height: ${h}px; overflow: hidden; $bg_css box-shadow: 0 25px 60px rgba(0,0,0,0.8); }
    .rec-header { position: absolute; top: 44px; left: 56px; right: 56px; display: flex; justify-content: space-between; align-items: center; z-index: 20; }
    .company-logo { font-family: 'Montserrat', sans-serif; font-size: 22px; font-weight: 900; color: #38BDF8; letter-spacing: 2px; }
    .urgency-badge { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #F87171; font-size: 13px; font-weight: 700; padding: 8px 18px; border-radius: 999px; letter-spacing: 1px; }
    .frosted-box { position: absolute; top: 110px; left: 56px; right: 56px; bottom: 44px; background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.14); border-radius: 28px; padding: 40px 48px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 20px 50px rgba(0,0,0,0.5); z-index: 20; }
    .pos-title-group { display: flex; justify-content: space-between; align-items: center; }
    .pos-label { font-size: 14px; font-weight: 700; color: #38BDF8; letter-spacing: 2px; text-transform: uppercase; }
    .salary-tag { background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%); color: #FFFFFF; font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 20px; padding: 12px 24px; border-radius: 14px; box-shadow: 0 8px 20px rgba(2, 132, 199, 0.4); border: 1px solid rgba(255,255,255,0.25); }
    .two-col-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; margin: 24px 0; }
    .col-title { font-size: 16px; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
    .checklist { list-style: none; display: flex; flex-direction: column; gap: 12px; }
    .check-item { display: flex; align-items: flex-start; gap: 12px; font-size: 15px; color: #E2E8F0; line-height: 1.45; font-weight: 500; }
    .check-icon { color: #38BDF8; font-weight: 900; font-size: 16px; }
    .rec-footer { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 24px; }
    .contact-block { display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: #94A3B8; }
    .contact-email { color: #38BDF8; font-weight: 700; font-size: 16px; }
    .apply-btn { background: linear-gradient(135deg, #38BDF8 0%, #0284C7 100%); color: #020617; font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 17px; padding: 16px 40px; border-radius: 999px; text-decoration: none; box-shadow: 0 8px 25px rgba(56, 189, 248, 0.4); border: 1px solid rgba(255,255,255,0.4); }
  </style>
</head>
<body>
  <div class="poster">
    <div class="rec-header">
      <div class="company-logo">$company</div>
      <div class="urgency-badge">$deadline</div>
    </div>
    <div class="frosted-box">
      <div class="pos-title-group">
        <div class="pos-label">$pos_label</div>
        <div class="salary-tag">$salary</div>
      </div>
      <div class="two-col-grid">
        <div>
          <div class="col-title">📋 YÊU CẦU ỨNG VIÊN</div>
          <ul class="checklist">$requirements_html</ul>
        </div>
        <div>
          <div class="col-title">🎁 QUYỀN LỢI ĐẶC QUYỀN</div>
          <ul class="checklist">$benefits_html</ul>
        </div>
      </div>
      <div class="rec-footer">
        <div class="contact-block">
          <div>$contact_line1</div>
          <div class="contact-email">$contact_email</div>
        </div>
        <a href="#" class="apply-btn">$cta_text</a>
      </div>
    </div>
  </div>
</body>
</html>""")

    @classmethod
    def _generate_recruitment(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        requirements = brief.get("requirements", [
            "Tối thiểu 2 năm kinh nghiệm trong lĩnh vực liên quan.",
            "Có tư duy chủ động, khả năng làm việc độc lập tốt.",
        ])
        benefits = brief.get("benefits", [
            "Thưởng dự án theo quý, đãi ngộ cạnh tranh.",
            "Môi trường làm việc hiện đại, đồng nghiệp thân thiện.",
        ])
        req_html = "".join(f'<li class="check-item"><span class="check-icon">✔</span><span>{r}</span></li>' for r in requirements)
        ben_html = "".join(f'<li class="check-item"><span class="check-icon">★</span><span>{b}</span></li>' for b in benefits)
        return cls._RECRUITMENT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            company=brief.get("company", "⚡ TENDOO AI RESEARCH LAB"),
            deadline=brief.get("deadline", "HẠN NỘP: 30.09.2026"),
            pos_label=brief.get("pos_label", "WE ARE HIRING • FULL-TIME POSITION"),
            salary=brief.get("salary", "THOẢ THUẬN"),
            requirements_html=req_html,
            benefits_html=ben_html,
            contact_line1=brief.get("contact_line1", "Gửi CV & Portfolio trực tiếp về hòm thư:"),
            contact_email=brief.get("contact_email", "careers@tendoo.ai"),
            cta_text=brief.get("cta_text", "ỨNG TUYỂN NGAY ➔"),
        )

    _MENU_TPL = Template("""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Menu</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; font-family: 'Plus Jakarta Sans', sans-serif; }
    .poster { position: relative; width: ${w}px; height: ${h}px; overflow: hidden; $bg_css box-shadow: 0 25px 60px rgba(0,0,0,0.8); padding: 56px; display: flex; flex-direction: column; justify-content: flex-end; gap: 24px; }
    .sub-brand { font-size: 14px; font-weight: 700; color: #D97706; letter-spacing: 4px; text-transform: uppercase; text-align: center; }
    .menu-desc { font-style: italic; font-size: 15px; color: #E7E5E4; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.6); }
    .menu-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; background: rgba(10,6,4,0.55); backdrop-filter: blur(16px); border-radius: 24px; padding: 32px; }
    .cat-title { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; color: #F59E0B; border-bottom: 1px solid rgba(245, 158, 11, 0.3); padding-bottom: 8px; margin-bottom: 16px; }
    .item-list { display: flex; flex-direction: column; gap: 14px; }
    .menu-row { display: flex; flex-direction: column; gap: 3px; }
    .row-top { display: flex; align-items: baseline; justify-content: space-between; }
    .item-name { font-size: 16px; font-weight: 700; color: #FFFFFF; }
    .dotted-line { flex-grow: 1; border-bottom: 1px dotted rgba(255,255,255,0.3); margin: 0 10px; }
    .item-price { font-family: 'Playfair Display', serif; font-size: 18px; font-weight: 700; color: #F59E0B; }
    .badge-star { font-size: 10px; font-weight: 800; background: #EF4444; color: #FFF; padding: 2px 6px; border-radius: 4px; margin-left: 6px; text-transform: uppercase; }
    .menu-footer { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 16px; padding: 16px 28px; display: flex; justify-content: space-between; align-items: center; }
    .foot-note { font-size: 13.5px; color: #FFFFFF; }
    .foot-hotline { font-weight: 700; color: #F59E0B; font-size: 15px; }
  </style>
</head>
<body>
  <div class="poster">
    <div class="sub-brand">$sub_brand</div>
    <div class="menu-desc">$tagline</div>
    <div class="menu-grid">$categories_html</div>
    <div class="menu-footer">
      <div class="foot-note">$footer_note</div>
      <div class="foot-hotline">$hotline</div>
    </div>
  </div>
</body>
</html>""")

    @classmethod
    def _generate_menu(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        categories = brief.get("categories", [
            {"title": "🍔 MÓN CHÍNH", "items": [
                {"name": "Món Đặc Trưng", "price": "89.000đ", "badge": "BEST SELLER"},
                {"name": "Món Signature", "price": "149.000đ"},
            ]},
            {"title": "🍹 ĐỒ UỐNG", "items": [
                {"name": "Thức Uống Đặc Biệt", "price": "49.000đ", "badge": "HOT"},
                {"name": "Thức Uống Nhẹ", "price": "45.000đ"},
            ]},
        ])
        cat_html_parts = []
        for cat in categories:
            items_html = "".join(
                '<div class="menu-row"><div class="row-top">'
                f'<span class="item-name">{it.get("name","")}'
                + (f'<span class="badge-star">{it["badge"]}</span>' if it.get("badge") else "")
                + '</span><span class="dotted-line"></span>'
                f'<span class="item-price">{it.get("price","")}</span></div></div>'
                for it in cat.get("items", [])
            )
            cat_html_parts.append(
                f'<div><div class="cat-title">{cat.get("title","")}</div>'
                f'<div class="item-list">{items_html}</div></div>'
            )
        return cls._MENU_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            sub_brand=brief.get("sub_brand", "ARTISAN DINING EXPERIENCE"),
            tagline=brief.get("tagline", "Thưởng thức tinh hoa ẩm thực thủ công từ nguyên liệu cao cấp"),
            categories_html="".join(cat_html_parts),
            footer_note=brief.get("footer_note", "✨ Giảm 10% tổng hóa đơn khi check-in tại quán"),
            hotline=brief.get("hotline", "📞 Hotline: 1800 8198"),
        )


    # ----------------------------------------------------------------------------------------
    # PORTRAIT VARIANTS (9:16, 2:3, 4:5 -- h > w). Same brief keys as their landscape
    # counterparts above, but restructured, not just rescaled: a top header (absolute, overlays
    # wherever the hero photo puts its own header-safe zone) + a BOTTOM STACK CONTAINER that is
    # itself absolutely positioned but whose CHILDREN flow via flex-column/gap -- so however many
    # secondary blocks exist, they stack without needing per-element top:Npx tuning. Font sizes
    # use vw units so the same markup scales across the whole portrait bucket (576px..832px wide),
    # not just the one exact width it was eyeballed against. Long single-line labels get
    # white-space:nowrap + text-overflow:ellipsis as a safety net against the wrapping breakage
    # seen when the landscape templates were reused unchanged on a 576x1024 canvas.
    # ----------------------------------------------------------------------------------------

    _GRAND_OPENING_PORTRAIT_TPL = Template("""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Grand Opening (Portrait)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:100vw; height:100vh; font-family:'Plus Jakarta Sans', sans-serif; }
  .poster { position:relative; width:${w}px; height:${h}px; overflow:hidden; $bg_css }
  .header { position:absolute; top:4%; left:6%; right:6%; display:flex; justify-content:space-between; align-items:center; gap:12px; z-index:20; }
  .brand-title { font-size:4.2vw; font-weight:900; color:#FFB703; letter-spacing:1px; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:58%; text-shadow:0 0 16px rgba(255,183,3,0.5); }
  .date-pill { background:rgba(255,255,255,0.12); backdrop-filter:blur(10px); border:1px solid rgba(255,183,3,0.4); padding:2vw 3.5vw; border-radius:999px; font-size:2.8vw; font-weight:700; color:#FFF; white-space:nowrap; }
  .bottom-stack { position:absolute; bottom:4%; left:6%; right:6%; display:flex; flex-direction:column; gap:3%; z-index:20; }
  .badge-pill { align-self:center; background:linear-gradient(135deg,#E63946 0%,#D90429 100%); border:3px dashed #FFF; border-radius:999px; padding:3vw 6vw; text-align:center; box-shadow:0 10px 28px rgba(230,57,70,0.55); }
  .badge-main-p { font-size:7vw; font-weight:900; color:#FFF; line-height:1; }
  .badge-off-p { font-size:3vw; font-weight:800; color:#FFD166; letter-spacing:1px; }
  .info-block { background:rgba(20,10,5,0.75); backdrop-filter:blur(18px); border:1px solid rgba(255,183,3,0.25); border-radius:20px; padding:5vw; display:flex; flex-direction:column; gap:2.5vw; }
  .deal-title { font-size:4vw; font-weight:800; color:#FFF; }
  .deal-sub { font-size:3.2vw; font-weight:500; color:#FFB703; }
  .cta-btn { text-align:center; background:linear-gradient(135deg,#FB8500 0%,#FFB703 100%); color:#000; font-weight:900; font-size:4vw; letter-spacing:0.5px; padding:3.5vw; border-radius:999px; text-decoration:none; box-shadow:0 8px 22px rgba(251,133,0,0.5); }
</style></head>
<body><div class="poster">
  <div class="header"><div class="brand-title">$brand</div><div class="date-pill">$date_range</div></div>
  <div class="bottom-stack">
    <div class="badge-pill"><div class="badge-main-p">$badge_percent</div><div class="badge-off-p">$badge_label $badge_sub</div></div>
    <div class="info-block">
      <div class="deal-title">$address</div>
      <div class="deal-sub">$offer_desc</div>
      <a href="#" class="cta-btn">$cta_text</a>
    </div>
  </div>
</div></body></html>""")

    @classmethod
    def _generate_grand_opening_portrait(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        return cls._GRAND_OPENING_PORTRAIT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            brand=brief.get("brand", "🍔 THE BURGER CRAFT"),
            date_range=brief.get("date_range", "05.09 - 15.09"),
            badge_label=brief.get("badge_label", "GIẢM"),
            badge_percent=brief.get("badge_percent", "50%"),
            badge_sub=brief.get("badge_sub", "TOÀN MENU"),
            address=brief.get("address", "📍 128 Nguyễn Trãi, Q1"),
            offer_desc=brief.get("offer_desc", "Tặng 01 Coca-Cola cho hóa đơn từ 99K • Hotline: 1900 8899"),
            cta_text=brief.get("cta_text", "NHẬN VOUCHER ➔"),
        )

    _FEEDBACK_PORTRAIT_TPL = Template("""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Feedback (Portrait)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:100vw; height:100vh; font-family:'Plus Jakarta Sans', sans-serif; }
  .poster { position:relative; width:${w}px; height:${h}px; overflow:hidden; $bg_css }
  .top-bar { position:absolute; top:4%; left:6%; right:6%; display:flex; justify-content:space-between; align-items:center; gap:10px; z-index:20; }
  .spa-logo { font-family:'Quicksand',sans-serif; font-size:4.2vw; font-weight:800; color:#0E9F6E; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:60%; text-shadow:0 2px 6px rgba(255,255,255,0.6); }
  .spa-badge { background:#FFE4E6; color:#E02424; font-weight:800; font-size:2.6vw; padding:1.8vw 3vw; border-radius:999px; white-space:nowrap; }
  .bottom-stack { position:absolute; bottom:3%; left:6%; right:6%; display:flex; flex-direction:column; gap:2.5%; z-index:20; }
  .feedback-card { background:rgba(255,255,255,0.85); backdrop-filter:blur(18px); border:2px solid rgba(255,255,255,0.9); border-radius:22px; padding:5vw; display:flex; flex-direction:column; gap:2.5vw; }
  .review-header { display:flex; justify-content:space-between; align-items:center; }
  .stars { color:#F59E0B; font-size:4.5vw; letter-spacing:2px; }
  .verified-pill { font-size:2.4vw; font-weight:700; color:#057A55; background:#DEF7EC; padding:1.2vw 2.6vw; border-radius:999px; white-space:nowrap; }
  .quote-text { font-size:3.6vw; line-height:1.5; color:#374151; font-weight:500; font-style:italic; }
  .customer-info { display:flex; align-items:center; gap:3vw; border-top:1px solid #E5E7EB; padding-top:3vw; }
  .avatar { width:9vw; height:9vw; border-radius:50%; background:#D1FAE5; display:flex; justify-content:center; align-items:center; font-size:4.5vw; border:2px solid #0E9F6E; flex-shrink:0; }
  .cust-name { font-size:3.2vw; font-weight:700; color:#111928; }
  .cust-sub { font-size:2.6vw; color:#6B7280; font-weight:500; }
  .features-col { display:flex; flex-direction:column; gap:2vw; }
  .f-pill { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px; padding:3vw 4vw; display:flex; align-items:center; gap:3vw; }
  .f-icon { font-size:5vw; }
  .f-text { font-size:3vw; font-weight:700; color:#1F2A37; }
  .bottom-cta-strip { background:linear-gradient(135deg,#0E9F6E 0%,#057A55 100%); border-radius:18px; padding:4vw 5vw; display:flex; flex-direction:column; gap:2vw; }
  .offer-title { font-family:'Quicksand',sans-serif; font-size:3.6vw; font-weight:800; color:#FFF; }
  .offer-desc { font-size:2.8vw; color:#FFF; opacity:0.9; }
  .btn-booking { align-self:flex-start; background:#FFFFFF; color:#046C4E; font-weight:800; font-size:3.2vw; padding:2.8vw 5vw; border-radius:999px; text-decoration:none; }
</style></head>
<body><div class="poster">
  <div class="top-bar"><div class="spa-logo">$brand</div><div class="spa-badge">$top_badge</div></div>
  <div class="bottom-stack">
    <div class="feedback-card">
      <div class="review-header"><div class="stars">$stars</div><div class="verified-pill">$verified_label</div></div>
      <div class="quote-text">$quote_text</div>
      <div class="customer-info"><div class="avatar">$avatar_emoji</div><div><div class="cust-name">$customer_name</div><div class="cust-sub">$customer_sub</div></div></div>
    </div>
    <div class="features-col">$features_html</div>
    <div class="bottom-cta-strip">
      <div class="offer-title">$offer_title</div>
      <div class="offer-desc">$offer_desc</div>
      <a href="#" class="btn-booking">$cta_text</a>
    </div>
  </div>
</div></body></html>""")

    @classmethod
    def _generate_feedback_card_portrait(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        features = brief.get("features", [
            {"icon": "🌿", "text": "Chất Lượng Hữu Cơ 100% Nhập Khẩu"},
            {"icon": "✂️", "text": "Chuyên Nghiệp Theo Yêu Cầu Riêng"},
            {"icon": "🕊️", "text": "Không Gian Mở, Trải Nghiệm Thoải Mái"},
        ])
        features_html = "".join(
            f'<div class="f-pill"><div class="f-icon">{f.get("icon","✨")}</div><div class="f-text">{f.get("text","")}</div></div>'
            for f in features
        )
        return cls._FEEDBACK_PORTRAIT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            brand=brief.get("brand", "🐾 PAWPARADISE SPA"),
            top_badge=brief.get("top_badge", "✨ CHUẨN HÀN QUỐC"),
            stars=brief.get("stars", "★★★★★"),
            verified_label=brief.get("verified_label", "✔ ĐÃ TRẢI NGHIỆM"),
            quote_text=brief.get("quote_text", "Dịch vụ tuyệt vời, nhân viên chuyên nghiệp và tận tâm, chắc chắn sẽ quay lại!"),
            avatar_emoji=brief.get("avatar_emoji", "🐩"),
            customer_name=brief.get("customer_name", "Khách hàng thân thiết"),
            customer_sub=brief.get("customer_sub", "Đã trải nghiệm dịch vụ Premium"),
            features_html=features_html,
            offer_title=brief.get("offer_title", "🎁 ƯU ĐÃI ĐẶC BIỆT CHO KHÁCH MỚI"),
            offer_desc=brief.get("offer_desc", "Áp dụng khi đặt lịch lần đầu trong tuần này!"),
            cta_text=brief.get("cta_text", "ĐẶT LỊCH NGAY ➔"),
        )

    _RECRUITMENT_PORTRAIT_TPL = Template("""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Recruitment (Portrait)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Montserrat:wght@700;800;900&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:100vw; height:100vh; font-family:'Plus Jakarta Sans', sans-serif; }
  .poster { position:relative; width:${w}px; height:${h}px; overflow:hidden; $bg_css }
  .rec-header { position:absolute; top:3.5%; left:6%; right:6%; display:flex; justify-content:space-between; align-items:center; gap:10px; z-index:20; }
  .company-logo { font-size:3.6vw; font-weight:900; color:#38BDF8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:60%; }
  .urgency-badge { background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#F87171; font-size:2.4vw; font-weight:700; padding:1.6vw 3vw; border-radius:999px; white-space:nowrap; }
  .frosted-box { position:absolute; bottom:3%; left:6%; right:6%; background:rgba(15,23,42,0.7); backdrop-filter:blur(20px); border:1px solid rgba(255,255,255,0.14); border-radius:22px; padding:5vw; display:flex; flex-direction:column; gap:3vw; max-height:70%; }
  .salary-tag { align-self:flex-start; background:linear-gradient(135deg,#0284C7 0%,#0369A1 100%); color:#FFF; font-weight:800; font-size:3.4vw; padding:2vw 4vw; border-radius:12px; }
  .col-title { font-size:2.8vw; font-weight:800; color:#94A3B8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:1.5vw; }
  .checklist { list-style:none; display:flex; flex-direction:column; gap:2vw; }
  .check-item { display:flex; align-items:flex-start; gap:2vw; font-size:2.8vw; color:#E2E8F0; line-height:1.4; font-weight:500; }
  .check-icon { color:#38BDF8; font-weight:900; }
  .rec-footer { display:flex; flex-direction:column; gap:2vw; border-top:1px solid rgba(255,255,255,0.1); padding-top:3vw; }
  .contact-email { color:#38BDF8; font-weight:700; font-size:3vw; }
  .apply-btn { text-align:center; background:linear-gradient(135deg,#38BDF8 0%,#0284C7 100%); color:#020617; font-weight:800; font-size:3.4vw; padding:3.2vw; border-radius:999px; text-decoration:none; }
</style></head>
<body><div class="poster">
  <div class="rec-header"><div class="company-logo">$company</div><div class="urgency-badge">$deadline</div></div>
  <div class="frosted-box">
    <div class="salary-tag">$salary</div>
    <div>
      <div class="col-title">📋 $pos_label</div>
      <ul class="checklist">$requirements_html</ul>
    </div>
    <div>
      <div class="col-title">🎁 QUYỀN LỢI</div>
      <ul class="checklist">$benefits_html</ul>
    </div>
    <div class="rec-footer">
      <div style="font-size:2.6vw;color:#94A3B8;">$contact_line1</div>
      <div class="contact-email">$contact_email</div>
      <a href="#" class="apply-btn">$cta_text</a>
    </div>
  </div>
</div></body></html>""")

    @classmethod
    def _generate_recruitment_portrait(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        requirements = brief.get("requirements", [
            "Tối thiểu 2 năm kinh nghiệm trong lĩnh vực liên quan.",
            "Có tư duy chủ động, khả năng làm việc độc lập tốt.",
        ])
        benefits = brief.get("benefits", [
            "Thưởng dự án theo quý, đãi ngộ cạnh tranh.",
            "Môi trường làm việc hiện đại, đồng nghiệp thân thiện.",
        ])
        req_html = "".join(f'<li class="check-item"><span class="check-icon">✔</span><span>{r}</span></li>' for r in requirements)
        ben_html = "".join(f'<li class="check-item"><span class="check-icon">★</span><span>{b}</span></li>' for b in benefits)
        return cls._RECRUITMENT_PORTRAIT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            company=brief.get("company", "⚡ TENDOO AI LAB"),
            deadline=brief.get("deadline", "HẠN: 30.09"),
            pos_label=brief.get("pos_label", "YÊU CẦU ỨNG VIÊN"),
            salary=brief.get("salary", "THOẢ THUẬN"),
            requirements_html=req_html,
            benefits_html=ben_html,
            contact_line1=brief.get("contact_line1", "Gửi CV & Portfolio:"),
            contact_email=brief.get("contact_email", "careers@tendoo.ai"),
            cta_text=brief.get("cta_text", "ỨNG TUYỂN NGAY ➔"),
        )

    _MENU_PORTRAIT_TPL = Template("""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Menu (Portrait)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:100vw; height:100vh; font-family:'Plus Jakarta Sans', sans-serif; }
  .poster { position:relative; width:${w}px; height:${h}px; overflow:hidden; $bg_css padding:5vw; display:flex; flex-direction:column; justify-content:flex-end; gap:3vw; }
  .sub-brand { font-size:2.6vw; font-weight:700; color:#D97706; letter-spacing:2px; text-transform:uppercase; text-align:center; }
  .menu-desc { font-style:italic; font-size:2.8vw; color:#E7E5E4; text-align:center; text-shadow:0 2px 6px rgba(0,0,0,0.6); }
  .menu-stack { display:flex; flex-direction:column; gap:5vw; background:rgba(10,6,4,0.6); backdrop-filter:blur(14px); border-radius:20px; padding:5vw; max-height:60%; overflow:hidden; }
  .cat-title { font-family:'Playfair Display',serif; font-size:4vw; font-weight:700; color:#F59E0B; border-bottom:1px solid rgba(245,158,11,0.3); padding-bottom:1.5vw; margin-bottom:2vw; }
  .item-list { display:flex; flex-direction:column; gap:2.5vw; }
  .menu-row { display:flex; flex-direction:column; gap:0.5vw; }
  .row-top { display:flex; align-items:baseline; justify-content:space-between; gap:2vw; }
  .item-name { font-size:3.2vw; font-weight:700; color:#FFFFFF; }
  .dotted-line { flex-grow:1; border-bottom:1px dotted rgba(255,255,255,0.3); margin:0 1vw; }
  .item-price { font-family:'Playfair Display',serif; font-size:3.4vw; font-weight:700; color:#F59E0B; white-space:nowrap; }
  .badge-star { font-size:2vw; font-weight:800; background:#EF4444; color:#FFF; padding:0.5vw 1.5vw; border-radius:4px; margin-left:1.5vw; }
  .menu-footer { background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.25); border-radius:14px; padding:3vw 4vw; display:flex; flex-direction:column; gap:1.5vw; }
  .foot-note { font-size:2.6vw; color:#FFFFFF; }
  .foot-hotline { font-weight:700; color:#F59E0B; font-size:2.8vw; }
</style></head>
<body><div class="poster">
  <div class="sub-brand">$sub_brand</div>
  <div class="menu-desc">$tagline</div>
  <div class="menu-stack">$categories_html</div>
  <div class="menu-footer"><div class="foot-note">$footer_note</div><div class="foot-hotline">$hotline</div></div>
</div></body></html>""")

    @classmethod
    def _generate_menu_portrait(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        categories = brief.get("categories", [
            {"title": "🍔 MÓN CHÍNH", "items": [
                {"name": "Món Đặc Trưng", "price": "89.000đ", "badge": "BEST SELLER"},
                {"name": "Món Signature", "price": "149.000đ"},
            ]},
            {"title": "🍹 ĐỒ UỐNG", "items": [
                {"name": "Thức Uống Đặc Biệt", "price": "49.000đ", "badge": "HOT"},
            ]},
        ])
        cat_html_parts = []
        for cat in categories:
            items_html = "".join(
                '<div class="menu-row"><div class="row-top">'
                f'<span class="item-name">{it.get("name","")}'
                + (f'<span class="badge-star">{it["badge"]}</span>' if it.get("badge") else "")
                + '</span><span class="dotted-line"></span>'
                f'<span class="item-price">{it.get("price","")}</span></div></div>'
                for it in cat.get("items", [])
            )
            cat_html_parts.append(
                f'<div><div class="cat-title">{cat.get("title","")}</div>'
                f'<div class="item-list">{items_html}</div></div>'
            )
        return cls._MENU_PORTRAIT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            sub_brand=brief.get("sub_brand", "ARTISAN DINING EXPERIENCE"),
            tagline=brief.get("tagline", "Thưởng thức tinh hoa ẩm thực thủ công"),
            categories_html="".join(cat_html_parts),
            footer_note=brief.get("footer_note", "✨ Giảm 10% khi check-in tại quán"),
            hotline=brief.get("hotline", "📞 Hotline: 1800 8198"),
        )


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
