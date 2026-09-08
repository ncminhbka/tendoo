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
import colorsys
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

def _bg_data_uri(background_image_path: Optional[str]) -> Optional[str]:
    """Shared helper: builds a base64 data: URI for an image path, or None if none/missing.
    Split out of `_bg_image_css` so the "Type Collage" giant-title technique (see
    PosterTemplateEngine._giant_title_block) can reference the SAME encoded photo without
    re-reading/re-encoding the file a second time per template."""
    if background_image_path and os.path.exists(background_image_path):
        with open(background_image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(background_image_path).suffix.lower().replace(".", "")
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        return f"data:{mime};base64,{b64_data}"
    return None


def _bg_image_css(background_image_path: Optional[str]) -> str:
    """Shared helper: embeds a background image as base64 CSS (full `.poster` background
    treatment: cover + centered), or returns '' if none given."""
    uri = _bg_data_uri(background_image_path)
    if uri:
        return f"background-image: url('{uri}'); background-size: cover; background-position: center;"
    return ""


def _bg_image_only_css(background_image_path: Optional[str]) -> str:
    """Just the `background-image: url(...)` declaration, no size/position -- for callers (the
    Type Collage giant-title technique) that need to set their OWN background-size/position to
    align a crop of the same photo, rather than the `.poster`-wide cover+center treatment."""
    uri = _bg_data_uri(background_image_path)
    return f"background-image: url('{uri}');" if uri else ""


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
            # before_after (single-image split-scene) deliberately reuses feedback's own render
            # code, not a new template: every real prompt_test.txt example (lines 21/23/29/33)
            # describes before/after as ONE diffusion-drawn split composition always paired with
            # review-card content -- the exact shape `feedback` already renders. Stage 1's job is
            # to write a `background_prompt` describing the split scene; Stage 4 needs no new
            # layout code for it. See docs/DESIGN_PRINCIPLES.md and AGENTS.md Rule 23 (dataset
            # topology already names "Ảnh Before-After" as a variant of this same half/half shape).
            ("before_after", "landscape"): cls._generate_feedback_card,
            ("before_after", "portrait"): cls._generate_feedback_card_portrait,
            ("recruitment", "landscape"): cls._generate_recruitment,
            ("recruitment", "portrait"): cls._generate_recruitment_portrait,
            ("menu", "landscape"): cls._generate_menu,
            ("menu", "portrait"): cls._generate_menu_portrait,
            # product_ad uses ONE implementation for both orientations -- unlike the card-heavy
            # templates above, it's just 2 positioned text blocks over a full-bleed photo, and vw-
            # relative sizing scales fine across the whole 4:5/9:16/1:1/16:9 range without needing
            # a structurally different portrait layout.
            ("product_ad", "landscape"): cls._generate_product_ad,
            ("product_ad", "portrait"): cls._generate_product_ad,
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
        <span>{badge}</span>
      </div>
    </div>

    <!-- BOTTOM FOOTER & SOCIAL PROOF ZONE -->
    <div class="bottom-section">
      <div class="social-proof-bar">
        <span class="stars"><svg width="14" height="14" viewBox="0 0 24 24" fill="#FFB300"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg><svg width="14" height="14" viewBox="0 0 24 24" fill="#FFB300"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg><svg width="14" height="14" viewBox="0 0 24 24" fill="#FFB300"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg><svg width="14" height="14" viewBox="0 0 24 24" fill="#FFB300"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg><svg width="14" height="14" viewBox="0 0 24 24" fill="#FFB300"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg></span>
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
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="display:inline-block; vertical-align:-2px; margin-left:4px;"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
        </a>
        <div class="contact-info">
          <div class="contact-item">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline-block; vertical-align:-2px; margin-right:4px;"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg>
            <span>Hotline: {hotline}</span>
          </div>
          <div class="contact-item">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline-block; vertical-align:-2px; margin-right:4px;"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
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

    # Card-stack templates (feedback/recruitment/grand_opening) hold much more content than a
    # single title -- a MER rect big enough for one line of text is nowhere near big enough for a
    # whole review card + features + CTA strip, so this uses a much higher floor than
    # `MIN_SAFE_RECT_HEIGHT_PCT` (10%, tuned for product_ad's single title zone -- see below).
    # Tuned against the real Case B probe (scripts/probe_mer_representative_cases.py): the
    # existing hand-designed fixed layout's own card group occupies roughly ~35-40% of canvas
    # height, so a MER result below ~30% is very likely too cramped to hold this much content.
    MIN_SAFE_RECT_HEIGHT_PCT_CARD = 30.0

    @classmethod
    def _derive_theme_palette(
        cls, accent_hex: Optional[str], defaults: Dict[str, str], secondary_hex: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Fixes a real gap the user caught by inspection: feedback/recruitment/menu/grand_opening
        each hardcode their OWN brand accent color unconditionally, regardless of the poster's
        actual subject matter -- a "cinematic wedding" feedback poster rendered with
        PawParadise Spa's green branding, because nothing in the template's CSS could ever be
        anything other than green. This builds a coherent tonal palette (accent / accent_dark /
        accent_darker / accent_tint / accent_rgb) from ONE brief-supplied hex color
        (`brief["brand_color"]`) via HSL lightness shifts, instead of requiring the caller to hand-
        pick 3-4 separate coordinated hex values. Falls back to `defaults` (each template's own
        historical hardcoded hex values) when no override is given, or the given hex fails to
        parse -- zero regression for every existing caller that doesn't pass `brand_color`.

        `defaults` must supply the same 5 keys this returns: accent, accent_dark, accent_darker,
        accent_tint, accent_rgb (the last for `rgba(var(--accent-rgb), alpha)` usage in CSS, since
        a CSS custom property holding a hex string can't be alpha-blended directly).

        `secondary_hex` -- ADDED after a real audit of prompt_test.txt found 10/11 feedback/
        before_after lines ask for a genuine TWO-TONE palette (vd "tone đen đỏ", "hồng pastel và
        xanh mint"), which a single accent color can never express (it only shades ONE hue).
        Optional and additive: when given (and it parses), overlays `secondary`/`secondary_rgb`/
        `secondary_tint_rgb` computed the same HSL way as the primary accent; when absent, those 3
        keys fall through from `defaults` unchanged (each caller's own historical hardcoded values,
        e.g. feedback's badge red) -- zero regression for every caller that doesn't pass it
        (currently only feedback's badge actually reads `--secondary`; other categories don't
        declare the CSS custom property at all, so the extra keys are simply unused, harmless).

        NOT an auto-pick-from-measured-luminance system like product_ad's `_auto_pick_style` --
        these templates have many differently-colored UI pieces (badges, gradients, text) rather
        than one text zone to measure, so there's no single "zone luminance" to read the way
        product_ad does. This only makes the color CONFIGURABLE; auto-selecting a good one from
        the background is a further enhancement, not done here.
        """
        def _shade(hh: float, ll: float, ss: float) -> Tuple[str, str]:
            rr, gg, bb = colorsys.hls_to_rgb(hh, max(0.0, min(1.0, ll)), ss)
            ri, gi, bi = int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255))
            return f"#{ri:02x}{gi:02x}{bi:02x}", f"{ri},{gi},{bi}"

        def _parse(hex_str: str) -> Optional[Tuple[int, int, int]]:
            hexs = hex_str.lstrip("#")
            try:
                return tuple(int(hexs[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
            except (ValueError, IndexError):
                return None

        result = dict(defaults)

        accent_rgb_parsed = _parse(accent_hex) if accent_hex else None
        if accent_rgb_parsed:
            h, l, s = colorsys.rgb_to_hls(*(c / 255.0 for c in accent_rgb_parsed))
            accent_hex_out, accent_rgb_out = _shade(h, l, s)
            accent_dark_hex, accent_dark_rgb = _shade(h, l - 0.15, s)
            accent_darker_hex, accent_darker_rgb = _shade(h, l - 0.28, s)
            result.update({
                "accent": accent_hex_out,
                "accent_dark": accent_dark_hex,
                "accent_darker": accent_darker_hex,
                "accent_tint": _shade(h, l + (1.0 - l) * 0.85, min(s, 0.35))[0],
                # A second, slightly paler tint -- some templates use 2 distinct light tints for
                # different pieces (e.g. feedback's verified-pill background vs avatar background).
                "accent_tint2": _shade(h, l + (1.0 - l) * 0.78, min(s, 0.30))[0],
                "accent_rgb": accent_rgb_out,
                "accent_dark_rgb": accent_dark_rgb,
                "accent_darker_rgb": accent_darker_rgb,
            })

        secondary_rgb_parsed = _parse(secondary_hex) if secondary_hex else None
        if secondary_rgb_parsed:
            h2, l2, s2 = colorsys.rgb_to_hls(*(c / 255.0 for c in secondary_rgb_parsed))
            secondary_hex_out, secondary_rgb_out = _shade(h2, l2, s2)
            result.update({
                "secondary": secondary_hex_out,
                "secondary_rgb": secondary_rgb_out,
                "secondary_tint_rgb": _shade(h2, l2 + (1.0 - l2) * 0.85, min(s2, 0.35))[1],
            })
        return result

    @classmethod
    def _safe_rect_style_attr(cls, brief: Dict[str, Any]) -> str:
        """
        Cap do 2 (MER, src/tendoo/layout_geometry.py) hook for the "one big absolutely-positioned
        content container" templates -- feedback's `.bottom-stack`, recruitment's `.frosted-box`,
        grand_opening's `.bottom-bar`/`.bottom-stack`. Generalizes product_ad's
        `_zone_css_from_rect` to an INLINE style override spliced onto the container's existing
        div, instead of building a zone from scratch -- these templates already have a
        fully-designed fixed layout; Cap do 2 should nudge the box away from a detected
        face/product, not replace the whole thing.

        Returns '' (a safe no-op -- the template's own fixed CSS wins, exactly as before this
        hook existed) if `brief` has no `"safe_rect"`, it's missing required keys, or it's too
        short to trust for a content block this size (see MIN_SAFE_RECT_HEIGHT_PCT_CARD above).
        NOT wired into `menu`/`menu_portrait` yet -- those templates flow content from a flex
        container (`justify-content: flex-end`), not one absolutely-positioned box, so this
        simple inline-style-splice approach doesn't apply there without a bigger restructure.

        Returned string includes a leading space so it can be spliced directly after a
        `class="..."` attribute in the HTML, e.g. `<div class="bottom-stack"$safe_rect_style>`.

        REAL BUG found+fixed after the first real-pipeline run on live model content (2026-09-05,
        see memory css-hero-title-overlay-direction.md): the original version also set `bottom`
        from the rect and stripped `max-height` (`max-height:none`) -- this forces an EXACT
        computed height on the container (top+bottom both fixed = height is whatever's between
        them), regardless of whether the actual content (review card + features + CTA strip) needs
        more room than that. On a real run this genuinely happened -- MER found a real, correctly
        placed top edge (clearing 2 detected faces) but the resulting height was shorter than the
        content's natural height, and the CTA button + a whole feature row were silently pushed
        past the canvas edge and clipped by `.poster`'s `overflow:hidden` -- not just visually
        cramped, GONE from the rendered poster entirely. Only `top` (+ left/right) is overridden
        now; `bottom` and `max-height` are left as the template's own CSS class already defines
        them, preserving the "anchored to bottom, grows upward, capped at max-height" behavior
        these templates were designed around. Trade-off: an obstacle sitting very close to the
        BOTTOM of the frame is no longer specifically avoided by this hook -- a smaller, rarer risk
        than silently deleting the CTA button.
        """
        rect_pct = brief.get("safe_rect")
        if not rect_pct:
            return ""
        try:
            top, left, right, height = (
                rect_pct["top_pct"], rect_pct["left_pct"], rect_pct["right_pct"], rect_pct["height_pct"],
            )
        except (KeyError, TypeError):
            return ""
        if height < cls.MIN_SAFE_RECT_HEIGHT_PCT_CARD:
            return ""
        # `bottom:auto` explicitly CANCELS the template class's own fixed `bottom:4%`-style rule --
        # setting only `top` here is not enough on its own, because the class rule still applies
        # for any property the inline style doesn't touch. With both `top` (inline) and `bottom`
        # (class) specified, the box's height is STILL forced to the distance between them --
        # confirmed on a real re-test after the first fix attempt: identical clipped output,
        # because the safe_rect's own bottom_pct happened to numerically match the class's default
        # anyway. `bottom:auto` makes height content-driven (shrink-to-fit), anchored at `top` and
        # growing downward -- guarantees the full card+features+CTA always renders, at the cost of
        # the box no longer hugging the canvas bottom edge the way the original fixed design did.
        return f' style="top:{top}%; left:{left}%; right:{right}%; bottom:auto;"'

    @classmethod
    def _corner_card_top_override(cls, brief: Dict[str, Any]) -> str:
        """
        Cấp độ 2 hook for feedback's compact `.corner-card` (redesigned 2026-09-05, per direct
        user request -- see the template's own comment). Deliberately NARROWER than
        `_safe_rect_style_attr`: only nudges `top` (+ `bottom:auto` so it can still grow downward
        without being height-capped) to clear an obstacle near the card's default position --
        does NOT touch `left`/`right`/`width` at all. Reusing `_safe_rect_style_attr` here was
        tried first and confirmed wrong on a real render: setting `right` from the MER rect fights
        the CSS class's own fixed `width` (an "over-constrained" box per the CSS abspos spec), and
        the card rendered noticeably WIDER than its intended compact size. The whole point of this
        redesign is a small corner card, not a full-width one -- width must stay exactly what the
        class defines, regardless of what Cấp độ 2 found.
        """
        rect_pct = brief.get("safe_rect")
        if not rect_pct:
            return ""
        try:
            top, height = rect_pct["top_pct"], rect_pct["height_pct"]
        except (KeyError, TypeError):
            return ""
        if height < cls.MIN_SAFE_RECT_HEIGHT_PCT_CARD:
            return ""
        return f' style="top:{top}%; bottom:auto;"'

    # ----------------------------------------------------------------------------------------
    # "TYPE COLLAGE" -- oversized see-through title (DESIGN_PRINCIPLES.md #6), added to
    # recruitment/menu only. ADDITIVE: doesn't replace/remove any existing element, just paints
    # one extra big headline-shaped "window" into the same photo already used as `.poster`'s own
    # background, in the empty space each of those 2 templates naturally has.
    #
    # The crop-math this relies on: PosterBackgroundAnalyzer.analyze() opens the SAME file later
    # passed as `background_image_path` (see PosterBackgroundAnalyzer.analyze, ~line 152), so
    # `analysis.width`/`analysis.height` are always exactly the photo's native pixel size --
    # `.poster`'s own `background-size:cover` therefore never actually scales/crops anything (a
    # `cover` fit where the box and the image are the same size is a no-op, scale factor 1). That
    # means: for a child positioned at (left_px, top_px) inside `.poster`, giving IT the same
    # photo with `background-size: <full poster w>px <full poster h>px; background-position:
    # -{left_px}px -{top_px}px;` shows EXACTLY the same photo pixels that would be visible there
    # if the child were transparent -- not an arbitrary crop. Using `background-size:cover` sized
    # to the child's own (much smaller) box instead would recompute an unrelated scale/crop --
    # deliberately NOT done here.
    # ----------------------------------------------------------------------------------------

    _GIANT_TITLE_DARK_CSS = "filter: invert(1) brightness(1.15) contrast(1.05); mix-blend-mode: screen;"
    _GIANT_TITLE_LIGHT_CSS = "mix-blend-mode: soft-light;"
    _EMOJI_PREFIX_RE = re.compile(r"^[\s\U0001F300-\U0001FAFF☀-➿]+")

    @classmethod
    def _giant_title_blend_css(cls, analysis: BackgroundAnalysis) -> str:
        """Picks the blend technique from the ACTUAL measured luminance of the zone the giant
        title sits in (same `header_zone` PosterBackgroundAnalyzer already computes, same pattern
        as `_auto_pick_style`) -- soft-light reads nicely on a bright photo but goes muddy/too dark
        on a dark one; invert+screen guarantees a bright, visible letterform on a dark photo at
        the cost of literal photo fidelity through the glyphs."""
        return cls._GIANT_TITLE_DARK_CSS if analysis.header_zone.is_dark else cls._GIANT_TITLE_LIGHT_CSS

    @classmethod
    def _giant_title_text(cls, raw: str) -> str:
        """Strips a leading emoji (COLR/emoji glyphs paint their own opaque bitmap in Chromium,
        ignoring `background-clip:text`/`color:transparent` masking, which would show as a solid
        ugly glyph breaking the see-through effect) -- only for the giant-title copy; the
        original element (`.company-logo`/`.sub-brand`) keeps its emoji untouched."""
        stripped = cls._EMOJI_PREFIX_RE.sub("", raw).strip()
        return stripped or raw

    @classmethod
    def _giant_title_block(
        cls, analysis: BackgroundAnalysis, background_image_path: Optional[str], text_raw: str,
        left_px: int, top_px: int, width_px: int, height_px: int, z_index: int, font_clamp: str,
    ) -> str:
        """Builds the giant see-through title <div>, or '' (safe no-op) if there's no photo to
        show through -- nothing to "window" into without one."""
        if not background_image_path or not os.path.exists(background_image_path):
            return ""
        text = cls._giant_title_text(text_raw)
        style = (
            f"position:absolute; left:{left_px}px; top:{top_px}px; width:{width_px}px; height:{height_px}px; "
            f"z-index:{z_index}; overflow:hidden; pointer-events:none; "
            f"font-family:'Montserrat',sans-serif; font-weight:900; text-transform:uppercase; "
            f"letter-spacing:-2px; line-height:0.92; white-space:nowrap; font-size:{font_clamp}; "
            f"{_bg_image_only_css(background_image_path)} "
            f"background-size:{analysis.width}px {analysis.height}px; "
            f"background-position:-{left_px}px -{top_px}px; background-repeat:no-repeat; "
            f"-webkit-background-clip:text; background-clip:text; color:transparent; -webkit-text-fill-color:transparent; "
            f"{cls._giant_title_blend_css(analysis)}"
        )
        return f'<div class="giant-title" style="{style}">{text}</div>'

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
      --accent: $accent; --accent-dark: $accent_dark; --accent-darker: $accent_darker;
      --accent-tint: $accent_tint; --accent-rgb: $accent_rgb; --accent-darker-rgb: $accent_darker_rgb;
    }
    .header { position: absolute; top: 48px; left: 56px; right: 56px; display: flex; justify-content: space-between; align-items: center; z-index: 20; }
    .brand-title { font-family: 'Montserrat', sans-serif; font-size: 24px; font-weight: 900; color: var(--accent); letter-spacing: 2px; text-transform: uppercase; text-shadow: 0 0 20px rgba(var(--accent-rgb), 0.5); }
    .date-pill { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(12px); border: 1px solid rgba(var(--accent-rgb), 0.4); padding: 10px 22px; border-radius: 999px; font-size: 14px; font-weight: 700; color: #FFF; letter-spacing: 1px; }
    .burst-badge {
      position: absolute; top: 220px; right: 70px; width: 170px; height: 170px;
      background: linear-gradient(135deg, #E63946 0%, #D90429 100%); border-radius: 50%;
      display: flex; flex-direction: column; justify-content: center; align-items: center;
      box-shadow: 0 12px 35px rgba(230, 57, 70, 0.6), inset 0 3px 6px rgba(255, 255, 255, 0.5);
      border: 4px dashed #FFF; transform: rotate(12deg); z-index: 25;
    }
    .badge-sub { font-size: 14px; font-weight: 800; color: #FFF; letter-spacing: 2px; text-transform: uppercase; }
    .badge-main { font-family: 'Montserrat', sans-serif; font-size: 52px; font-weight: 900; color: #FFF; line-height: 0.95; }
    .badge-off { font-size: 16px; font-weight: 900; color: var(--accent-tint); letter-spacing: 1.5px; }
    .bottom-bar {
      position: absolute; bottom: 50px; left: 56px; right: 56px; z-index: 20;
      background: rgba(20, 10, 5, 0.75); backdrop-filter: blur(20px); border: 1px solid rgba(var(--accent-rgb), 0.25);
      border-radius: 24px; padding: 24px 36px; display: flex; justify-content: space-between; align-items: center;
      box-shadow: 0 15px 40px rgba(0,0,0,0.6);
    }
    .deal-info { display: flex; flex-direction: column; gap: 4px; }
    .deal-title { font-family: 'Montserrat', sans-serif; font-size: 20px; font-weight: 800; color: #FFF; }
    .deal-sub { font-size: 14px; font-weight: 500; color: var(--accent); }
    .cta-btn { background: linear-gradient(135deg, var(--accent-darker) 0%, var(--accent) 100%); color: #000; font-family: 'Montserrat', sans-serif; font-weight: 900; font-size: 17px; letter-spacing: 0.5px; padding: 16px 36px; border-radius: 999px; text-decoration: none; box-shadow: 0 8px 25px rgba(var(--accent-darker-rgb), 0.5); border: 1px solid rgba(255,255,255,0.4); }
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
    <div class="bottom-bar"$safe_rect_style>
      <div class="deal-info">
        <div class="deal-title">$address</div>
        <div class="deal-sub">$offer_desc</div>
      </div>
      <a href="#" class="cta-btn">$cta_text</a>
    </div>
  </div>
</body>
</html>""")

    _GRAND_OPENING_DEFAULT_PALETTE = {
        "accent": "#FFB703", "accent_dark": "#FB8500", "accent_darker": "#FB8500",
        "accent_tint": "#FFD166", "accent_rgb": "255,183,3", "accent_darker_rgb": "251,133,0",
    }

    @classmethod
    def _generate_grand_opening(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        palette = cls._derive_theme_palette(brief.get("brand_color"), cls._GRAND_OPENING_DEFAULT_PALETTE)
        return cls._GRAND_OPENING_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            brand=brief.get("brand", "THE BURGER CRAFT"),
            date_range=brief.get("date_range", "DUY NHẤT 05.09 - 15.09.2026"),
            badge_label=brief.get("badge_label", "GIẢM"),
            badge_percent=brief.get("badge_percent", "50%"),
            badge_sub=brief.get("badge_sub", "TOÀN MENU"),
            address=brief.get("address", "128 Nguyễn Trãi, Phường Bến Thành, Quận 1"),
            offer_desc=brief.get("offer_desc", "Tặng 01 Coca-Cola mát lạnh cho hóa đơn từ 99K • Hotline: 1900 8899"),
            cta_text=brief.get("cta_text", "NHẬN VOUCHER"),
            safe_rect_style=cls._safe_rect_style_attr(brief),
            **palette,
        )

    _FEEDBACK_TPL = Template("""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Customer Feedback</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Dancing+Script:wght@700&family=Playfair+Display:ital,wght@0,700;1,400&family=Oswald:wght@600;700&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; font-family: 'Plus Jakarta Sans', sans-serif; }
    .poster {
      position: relative; width: ${w}px; height: ${h}px; overflow: hidden; $bg_css box-shadow: 0 25px 60px rgba(0,0,0,0.12); border-radius: 32px;
      --accent: $accent; --accent-dark: $accent_dark; --accent-tint: $accent_tint; --accent-tint2: $accent_tint2; --accent-rgb: $accent_rgb;
      --secondary: $secondary; --secondary-tint-rgb: $secondary_tint_rgb;
      --mood-quote-font: $mood_quote_font; --mood-name-font: $mood_name_font;
    }
    .top-bar { position: absolute; top: 4%; left: 5.5%; right: 5.5%; display: flex; justify-content: space-between; align-items: center; z-index: 20; gap: 12px; }
    /* Wraps to 2 lines instead of forcing 1-line ellipsis -- a real bug found in production: a
       longer English brand string ("Deep Cleaning Home Service") got sliced mid-word into "...".
       2-line wrap loses far less of the actual name than a hard truncation. */
    .spa-logo { font-family: 'Quicksand', sans-serif; font-size: 22px; font-weight: 800; color: var(--accent); max-width: 62%; line-height: 1.2; text-shadow: 0 2px 8px rgba(255,255,255,0.6); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    /* Translucent, not solid -- a large-ish badge with an opaque fill blocks a real chunk of the
       photo behind it (user feedback, 2026-09-05: "badge text lớn chiếm kha khá diện tích... nên
       để badge trong suốt hoặc mờ để nhìn xuyên được"). backdrop-filter keeps the text readable
       against whatever's behind it without needing a fully opaque fill.
       Recolored to `--secondary` (was hardcoded #E02424/rgba(255,228,230,...)) -- real gap found
       auditing prompt_test.txt: a 2-tone brand request ("tone đen đỏ", "hồng pastel và xanh
       mint"...) never reached this element even when `brand_color` WAS set, since it was pinned to
       one hardcoded hue regardless. Default palette value for `--secondary` equals the old
       hardcoded hex exactly, so a brief without a 2nd color renders byte-identical. */
    .spa-badge { background: rgba(var(--secondary-tint-rgb), 0.4); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); color: var(--secondary); font-family: 'Quicksand', sans-serif; font-weight: 800; font-size: 13px; padding: 9px 18px; border-radius: 999px; border: 1px solid rgba(var(--secondary-tint-rgb), 0.55); white-space: nowrap; flex-shrink: 0; }
    /* REDESIGNED (2026-09-05, per direct user request): the old design was a nearly-full-width
       stack (card + features-row + CTA strip) covering most of the photo -- appropriate for
       menu/recruitment (genuinely info-dense categories) but wrong for feedback, whose whole point
       is to let a real customer/before-after PHOTO read as the hero. Now ONE small corner card,
       ~38% wide, auto height (not stretched to fill a big region) -- the photo does the talking,
       text stays compact and secondary. */
    /* Genuinely see-through, not just "slightly less than fully opaque" -- user feedback,
       2026-09-05: the previous 0.92 opacity read as solid white in practice. Heavier blur (28px,
       up from 20px) compensates so the dark quote/name text stays legible against whatever
       texture shows through. */
    .corner-card {
      position: absolute; left: 5.5%; bottom: 5%; width: 38%; max-width: 420px;
      background: rgba(255,255,255,0.55); backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);
      border: 1.5px solid rgba(255,255,255,0.6); border-radius: 20px; padding: 4% 4.5% 4.5%;
      display: flex; flex-direction: column; gap: 10px;
      box-shadow: 0 20px 45px rgba(0,0,0,0.22), 0 1px 3px rgba(0,0,0,0.05);
      z-index: 20;
    }
    .review-header-mini { display: flex; justify-content: space-between; align-items: center; }
    .stars-mini { color: #F59E0B; font-size: 15px; letter-spacing: 2px; }
    .verified-mini { font-size: 10px; font-weight: 700; color: var(--accent-dark); background: var(--accent-tint); padding: 3px 9px; border-radius: 999px; white-space: nowrap; }
    /* Clamped to 3 lines -- a long quote grows the card only up to a point, never pushes the
       whole thing tall enough to fight Cấp độ 2's safe_rect for room (see memory
       css-hero-title-overlay-direction.md, the "MER vs content height" open issue this also
       helps with, even though it's not a full fix for that issue on its own). */
    .quote-mini { font-family: var(--mood-quote-font); font-size: 13px; line-height: 1.38; color: #374151; font-weight: 500; font-style: italic; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }
    .customer-mini { display: flex; align-items: center; gap: 8px; }
    .avatar-mini { width: 30px; height: 30px; flex-shrink: 0; border-radius: 50%; background: var(--accent-tint2); display: flex; justify-content: center; align-items: center; font-size: 15px; border: 1.5px solid var(--accent); }
    .cust-name-mini { font-family: var(--mood-name-font); font-size: 12.5px; font-weight: 700; color: #111928; line-height: 1.2; }
    .cust-sub-mini { font-size: 10px; color: #6B7280; font-weight: 500; }
    .tags-mini { display: flex; flex-wrap: wrap; gap: 5px; }
    /* Translucent, matching the card now (not solid white) -- an opaque pill sitting on top of a
       see-through card read as a separate shape "poking out" past the card's own edge (user
       feedback, 2026-09-05), especially once a long feature label made the pill stretch across
       most of the card's width. */
    /* Wraps to 2 lines instead of truncating with "..." -- same fix already applied to the brand
       name: a real feature label ("Nhân viên được đào tạo chuẩn khách sạn 5 sao") is long enough
       to get cut mid-word if forced onto one line, which is worse than a shorter pill wrapping to
       2 lines. `border-radius` reduced from a full pill (999px) to a softer chip shape, since a
       full pill looks odd once a tag is tall enough to hold 2 lines. */
    .tag-mini { font-size: 10px; font-weight: 600; color: #1F2A37; background: rgba(255,255,255,0.55); border: 1px solid rgba(229,231,235,0.7); border-radius: 12px; padding: 4px 9px; line-height: 1.3; max-width: 100%; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .cta-row-mini { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding-top: 6px; border-top: 1px solid rgba(0,0,0,0.07); }
    .offer-mini { font-size: 11px; font-weight: 700; color: var(--accent-dark); line-height: 1.25; flex: 1; min-width: 0; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .cta-mini { flex-shrink: 0; background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%); color: #FFFFFF; font-weight: 700; font-size: 11.5px; padding: 7px 14px; border-radius: 999px; text-decoration: none; white-space: nowrap; box-shadow: 0 6px 16px rgba(var(--accent-rgb), 0.35); }
  </style>
</head>
<body>
  <div class="poster">
    <div class="top-bar">
      <div class="spa-logo">$brand</div>
      $badge_html
    </div>
    <div class="corner-card"$safe_rect_style>
      <div class="review-header-mini">
        $stars_html
        <span class="verified-mini">$verified_label</span>
      </div>
      <div class="quote-mini">$quote_text</div>
      <div class="customer-mini">
        <div class="avatar-mini">$avatar_emoji</div>
        <div>
          <div class="cust-name-mini">$customer_name</div>
          <div class="cust-sub-mini">$customer_sub</div>
        </div>
      </div>
      $features_block_html
      <div class="cta-row-mini">
        $offer_html
        <a href="#" class="cta-mini">$cta_text</a>
      </div>
    </div>
  </div>
</body>
</html>""")

    _FEEDBACK_DEFAULT_PALETTE = {
        "accent": "#0E9F6E", "accent_dark": "#057A55", "accent_darker": "#057A55",
        "accent_tint": "#DEF7EC", "accent_tint2": "#D1FAE5",
        "accent_rgb": "14,159,110", "accent_darker_rgb": "5,122,85",
        # Secondary tone -- matches today's hardcoded .spa-badge red/pink exactly (255,228,230 tint
        # bg + #E02424 text), so a caller that doesn't pass a 2nd color gets byte-identical output.
        "secondary": "#E02424", "secondary_rgb": "224,36,36", "secondary_tint_rgb": "255,228,230",
    }

    # font_mood -- ADDED after the same prompt_test.txt audit: feedback/before_after had NO font
    # customization at all (unlike product_ad's title_style/title_font), yet 6/11 lines ask for a
    # specific typography mood ("chữ ký bay bổng mềm mại" for a wedding, "typography năng động"...).
    # Closed enum (not a free-text font name) -- deliberately avoids repeating the exact bug already
    # found+fixed once for product_ad's title_font (an invented/unloaded font name silently falls
    # through to the generic sans-serif with no warning). Each entry names 2 already-`@import`ed
    # families (quote text, customer name) -- "warm_friendly" is the literal pre-existing default
    # (Quicksand/Plus Jakarta Sans), named explicitly so picking it (or omitting font_mood) is a
    # true no-op.
    _FEEDBACK_FONT_MOODS = {
        "warm_friendly": {"quote_font": "'Quicksand', sans-serif", "name_font": "'Quicksand', sans-serif"},
        "elegant_script": {"quote_font": "'Dancing Script', cursive", "name_font": "'Playfair Display', serif"},
        "bold_sporty": {"quote_font": "'Oswald', sans-serif", "name_font": "'Oswald', sans-serif"},
        "clean_readable": {"quote_font": "'Plus Jakarta Sans', sans-serif", "name_font": "'Plus Jakarta Sans', sans-serif"},
    }
    _FEEDBACK_DEFAULT_FONT_MOOD = "warm_friendly"

    # hidden_elements -- ADDED for the same reason: testers sometimes explicitly don't want a piece
    # of the card ("không cần hiện rating sao"). Closed enum of the 4 genuinely optional pieces
    # (CTA button and the quote itself are never optional -- the whole card is pointless without
    # them). Blanking the whole wrapper element (not just its text) so hiding something doesn't
    # leave an empty translucent pill/row-shaped gap behind.
    _FEEDBACK_HIDEABLE_ELEMENTS = {"badge", "stars", "features", "offer"}

    @classmethod
    def _resolve_font_mood(cls, font_mood: Optional[str]) -> Dict[str, str]:
        return cls._FEEDBACK_FONT_MOODS.get(font_mood, cls._FEEDBACK_FONT_MOODS[cls._FEEDBACK_DEFAULT_FONT_MOOD])

    @classmethod
    def _build_feedback_conditional_html(
        cls, brief: Dict[str, Any],
        top_badge_default: str = "CHUẨN FORM HÀN QUỐC", stars_default: str = "5",
    ) -> Dict[str, str]:
        hidden = {h for h in (brief.get("hidden_elements") or []) if h in cls._FEEDBACK_HIDEABLE_ELEMENTS}

        badge_html = "" if "badge" in hidden else f'<div class="spa-badge">{brief.get("top_badge", top_badge_default)}</div>'
        stars_val = brief.get("stars", stars_default)
        if isinstance(stars_val, str) and stars_val.strip().lstrip("-").isdigit():
            star_count = max(0, min(5, int(stars_val)))
        elif isinstance(stars_val, (int, float)):
            star_count = max(0, min(5, int(stars_val)))
        else:
            star_count = 5
        star_svg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="#FFB300" style="display:inline-block; vertical-align:-1px; margin-right:1px;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>'
        stars_html = "" if "stars" in hidden else f'<span class="stars-mini">{star_svg * star_count}</span>'

        features = brief.get("features", [
            {"text": "Chất Lượng Hữu Cơ 100% Nhập Khẩu"},
            {"text": "Chuyên Nghiệp Theo Yêu Cầu Riêng"},
            {"text": "Không Gian Mở, Trải Nghiệm Thoải Mái"},
        ])
        check_icon = '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="display:inline-block; vertical-align:-1px; margin-right:3px; color:#10B981;"><polyline points="20 6 9 17 4 12"></polyline></svg>'
        features_inner = "".join(
            f'<span class="tag-mini">{check_icon}{f.get("text","")}</span>'
            for f in features[:3]
        )
        features_block_html = "" if "features" in hidden else f'<div class="tags-mini">{features_inner}</div>'

        offer_title = brief.get("offer_title", "ƯU ĐÃI ĐẶC BIỆT CHO KHÁCH MỚI")
        offer_desc = brief.get("offer_desc", "")
        offer_desc_suffix = f"<br/>{offer_desc}" if offer_desc else ""
        offer_html = '<span class="offer-mini"></span>' if "offer" in hidden else f'<span class="offer-mini">{offer_title}{offer_desc_suffix}</span>'

        return {
            "badge_html": badge_html, "stars_html": stars_html,
            "features_block_html": features_block_html, "offer_html": offer_html,
        }

    @classmethod
    def _generate_feedback_card(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        conditional = cls._build_feedback_conditional_html(brief)
        palette = cls._derive_theme_palette(
            brief.get("brand_color"), cls._FEEDBACK_DEFAULT_PALETTE, secondary_hex=brief.get("secondary_color"),
        )
        font_mood = cls._resolve_font_mood(brief.get("font_mood"))
        return cls._FEEDBACK_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            brand=brief.get("brand", "PAWPARADISE SPA"),
            verified_label=brief.get("verified_label", "ĐÃ TRẢI NGHIỆM DỊCH VỤ"),
            quote_text=brief.get("quote_text", "Dịch vụ tuyệt vời, nhân viên chuyên nghiệp và tận tâm, chắc chắn sẽ quay lại!"),
            avatar_emoji=brief.get("avatar_emoji", "P"),
            customer_name=brief.get("customer_name", "Khách hàng thân thiết"),
            customer_sub=brief.get("customer_sub", "Đã trải nghiệm dịch vụ Premium"),
            cta_text=brief.get("cta_text", "ĐẶT LỊCH NGAY"),
            safe_rect_style=cls._corner_card_top_override(brief),
            mood_quote_font=font_mood["quote_font"], mood_name_font=font_mood["name_font"],
            **conditional,
            **palette,
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
    .poster {
      position: relative; width: ${w}px; height: ${h}px; overflow: hidden; $bg_css box-shadow: 0 25px 60px rgba(0,0,0,0.8);
      isolation: isolate;
      --accent: $accent; --accent-dark: $accent_dark; --accent-darker: $accent_darker;
      --accent-rgb: $accent_rgb; --accent-dark-rgb: $accent_dark_rgb;
    }
    .rec-header { position: absolute; top: 44px; left: 56px; right: 56px; display: flex; justify-content: space-between; align-items: center; z-index: 20; }
    .company-logo { font-family: 'Montserrat', sans-serif; font-size: 22px; font-weight: 900; color: var(--accent); letter-spacing: 2px; }
    .urgency-badge { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #F87171; font-size: 13px; font-weight: 700; padding: 8px 18px; border-radius: 999px; letter-spacing: 1px; }
    .frosted-box { position: absolute; top: 110px; left: 56px; right: 56px; bottom: 44px; background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.14); border-radius: 28px; padding: 40px 48px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 20px 50px rgba(0,0,0,0.5); z-index: 20; }
    .pos-title-group { display: flex; justify-content: space-between; align-items: center; }
    .pos-label { font-size: 14px; font-weight: 700; color: var(--accent); letter-spacing: 2px; text-transform: uppercase; }
    /* Translucent, not a solid gradient fill -- a badge this size with an opaque fill blocks a
       real chunk of the photo behind it (user feedback, 2026-09-05). backdrop-filter keeps the
       bold white text readable without needing full opacity. */
    .salary-tag { background: rgba(var(--accent-dark-rgb), 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); color: #FFFFFF; font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 20px; padding: 12px 24px; border-radius: 14px; box-shadow: 0 8px 20px rgba(var(--accent-dark-rgb), 0.3); border: 1px solid rgba(255,255,255,0.3); }
    .two-col-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; margin: 24px 0; }
    .col-title { font-size: 16px; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
    .checklist { list-style: none; display: flex; flex-direction: column; gap: 12px; }
    .check-item { display: flex; align-items: flex-start; gap: 12px; font-size: 15px; color: #E2E8F0; line-height: 1.45; font-weight: 500; }
    .check-icon { color: var(--accent); font-weight: 900; font-size: 16px; }
    .rec-footer { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 24px; }
    .contact-block { display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: #94A3B8; }
    .contact-email { color: var(--accent); font-weight: 700; font-size: 16px; }
    .apply-btn { background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%); color: #020617; font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 17px; padding: 16px 40px; border-radius: 999px; text-decoration: none; box-shadow: 0 8px 25px rgba(var(--accent-rgb), 0.4); border: 1px solid rgba(255,255,255,0.4); }
  </style>
</head>
<body>
  <div class="poster">
    $giant_title_html
    <div class="rec-header">
      <div class="company-logo">$company</div>
      <div class="urgency-badge">$deadline</div>
    </div>
    <div class="frosted-box"$safe_rect_style>
      <div class="pos-title-group">
        <div class="pos-label">$pos_label</div>
        <div class="salary-tag">$salary</div>
      </div>
      <div class="two-col-grid">
        <div>
          <div class="col-title">YÊU CẦU ỨNG VIÊN</div>
          <ul class="checklist">$requirements_html</ul>
        </div>
        <div>
          <div class="col-title">QUYỀN LỢI ĐẶC QUYỀN</div>
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

    _RECRUITMENT_DEFAULT_PALETTE = {
        "accent": "#38BDF8", "accent_dark": "#0284C7", "accent_darker": "#0369A1",
        "accent_tint": "#DBEAFE", "accent_tint2": "#DBEAFE",
        "accent_rgb": "56,189,248", "accent_dark_rgb": "2,132,199", "accent_darker_rgb": "3,105,161",
    }

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
        req_html = "".join(f'<li class="check-item"><span class="check-icon"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="display:inline-block; vertical-align:-1px; color:#10B981;"><polyline points="20 6 9 17 4 12"></polyline></svg></span><span>{r}</span></li>' for r in requirements)
        ben_html = "".join(f'<li class="check-item"><span class="check-icon"><svg width="12" height="12" viewBox="0 0 24 24" fill="#FFB300"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg></span><span>{b}</span></li>' for b in benefits)
        palette = cls._derive_theme_palette(brief.get("brand_color"), cls._RECRUITMENT_DEFAULT_PALETTE)
        company = brief.get("company", "TENDOO AI RESEARCH LAB")
        giant_title_html = cls._giant_title_block(
            analysis, background_image_path, company,
            left_px=round(analysis.width * 0.04), top_px=0,
            width_px=round(analysis.width * 0.92), height_px=round(analysis.height * 0.30),
            z_index=1, font_clamp="clamp(48px, 9vw, 160px)",
        )
        return cls._RECRUITMENT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            company=company,
            deadline=brief.get("deadline", "HẠN NỘP: 30.09.2026"),
            pos_label=brief.get("pos_label", "WE ARE HIRING • FULL-TIME POSITION"),
            salary=brief.get("salary", "THOẢ THUẬN"),
            requirements_html=req_html,
            benefits_html=ben_html,
            contact_line1=brief.get("contact_line1", "Gửi CV & Portfolio trực tiếp về hòm thư:"),
            contact_email=brief.get("contact_email", "careers@tendoo.ai"),
            cta_text=brief.get("cta_text", "ỨNG TUYỂN NGAY"),
            safe_rect_style=cls._safe_rect_style_attr(brief),
            giant_title_html=giant_title_html,
            **palette,
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
    .poster {
      position: relative; width: ${w}px; height: ${h}px; overflow: hidden; $bg_css box-shadow: 0 25px 60px rgba(0,0,0,0.8); padding: 56px; display: flex; flex-direction: column; justify-content: flex-end; gap: 24px;
      isolation: isolate;
      --accent: $accent; --accent-dark: $accent_dark; --accent-rgb: $accent_rgb;
    }
    .sub-brand { font-size: 14px; font-weight: 700; color: var(--accent-dark); letter-spacing: 4px; text-transform: uppercase; text-align: center; }
    .menu-desc { font-style: italic; font-size: 15px; color: #E7E5E4; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.6); }
    .menu-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; background: rgba(10,6,4,0.55); backdrop-filter: blur(16px); border-radius: 24px; padding: 32px; }
    .cat-title { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; color: var(--accent); border-bottom: 1px solid rgba(var(--accent-rgb), 0.3); padding-bottom: 8px; margin-bottom: 16px; }
    .item-list { display: flex; flex-direction: column; gap: 14px; }
    .menu-row { display: flex; flex-direction: column; gap: 3px; }
    .row-top { display: flex; align-items: baseline; justify-content: space-between; }
    .item-name { font-size: 16px; font-weight: 700; color: #FFFFFF; }
    .dotted-line { flex-grow: 1; border-bottom: 1px dotted rgba(255,255,255,0.3); margin: 0 10px; }
    .item-price { font-family: 'Playfair Display', serif; font-size: 18px; font-weight: 700; color: var(--accent); }
    .badge-star { font-size: 10px; font-weight: 800; background: rgba(239,68,68,0.45); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); color: #FFF; padding: 2px 6px; border-radius: 4px; margin-left: 6px; text-transform: uppercase; }
    .menu-footer { background: rgba(var(--accent-rgb), 0.1); border: 1px solid rgba(var(--accent-rgb), 0.25); border-radius: 16px; padding: 16px 28px; display: flex; justify-content: space-between; align-items: center; }
    .foot-note { font-size: 13.5px; color: #FFFFFF; }
    .foot-hotline { font-weight: 700; color: var(--accent); font-size: 15px; }
  </style>
</head>
<body>
  <div class="poster">
    $giant_title_html
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

    _MENU_DEFAULT_PALETTE = {
        "accent": "#F59E0B", "accent_dark": "#D97706", "accent_darker": "#D97706",
        "accent_tint": "#FEF3C7", "accent_tint2": "#FEF3C7", "accent_rgb": "245,158,11",
    }

    @classmethod
    def _generate_menu(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        categories = brief.get("categories", [
            {"title": "MÓN CHÍNH", "items": [
                {"name": "Món Đặc Trưng", "price": "89.000đ", "badge": "BEST SELLER"},
                {"name": "Món Signature", "price": "149.000đ"},
            ]},
            {"title": "ĐỒ UỐNG", "items": [
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
        palette = cls._derive_theme_palette(brief.get("brand_color"), cls._MENU_DEFAULT_PALETTE)
        sub_brand = brief.get("sub_brand", "ARTISAN DINING EXPERIENCE")
        giant_title_html = cls._giant_title_block(
            analysis, background_image_path, sub_brand,
            left_px=round(analysis.width * 0.06), top_px=round(analysis.height * 0.06),
            width_px=round(analysis.width * 0.88), height_px=round(analysis.height * 0.34),
            z_index=-1, font_clamp="clamp(44px, 8vw, 150px)",
        )
        return cls._MENU_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            sub_brand=sub_brand,
            tagline=brief.get("tagline", "Thưởng thức tinh hoa ẩm thực thủ công từ nguyên liệu cao cấp"),
            categories_html="".join(cat_html_parts),
            footer_note=brief.get("footer_note", "Giảm 10% tổng hóa đơn khi check-in tại quán"),
            hotline=brief.get("hotline", "Hotline: 1800 8198"),
            giant_title_html=giant_title_html,
            **palette,
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
  .poster {
    position:relative; width:${w}px; height:${h}px; overflow:hidden; $bg_css
    --accent: $accent; --accent-dark: $accent_dark; --accent-darker: $accent_darker;
    --accent-tint: $accent_tint; --accent-rgb: $accent_rgb; --accent-darker-rgb: $accent_darker_rgb;
  }
  /* Font-sizes below fixed (2026-09-06, real audit finding, not a guess): every `vw` value in this
     portrait template was picked independently of the landscape version's PX values, and on the
     real portrait canvas (1024px wide, vs landscape's 1536px -- see `run_full_pipeline.py`'s
     `_pick_image_size`) the resulting text was actually LARGER in real pixels than landscape's,
     despite the canvas itself being smaller (~44% the area) -- backwards from "ảnh nhỏ hơn thì chữ
     phải nhỏ hơn". Recalibrated so portrait_px ~= landscape_px * 0.9 (deliberately a bit SMALLER,
     not just equal) at the real 1024px width: vw = landscape_px * 0.9 / 1024 * 100. */
  .header { position:absolute; top:4%; left:6%; right:6%; display:flex; justify-content:space-between; align-items:center; gap:12px; z-index:20; }
  .brand-title { font-size:2.1vw; font-weight:900; color:var(--accent); letter-spacing:1px; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:58%; text-shadow:0 0 16px rgba(var(--accent-rgb),0.5); }
  .date-pill { background:rgba(255,255,255,0.12); backdrop-filter:blur(10px); border:1px solid rgba(var(--accent-rgb),0.4); padding:1.4vw 2.6vw; border-radius:999px; font-size:1.2vw; font-weight:700; color:#FFF; white-space:nowrap; }
  .bottom-stack { position:absolute; bottom:4%; left:6%; right:6%; display:flex; flex-direction:column; gap:3%; z-index:20; }
  .badge-pill { align-self:center; background:linear-gradient(135deg,#E63946 0%,#D90429 100%); border:3px dashed #FFF; border-radius:999px; padding:3vw 6vw; text-align:center; box-shadow:0 10px 28px rgba(230,57,70,0.55); }
  .badge-main-p { font-size:4.6vw; font-weight:900; color:#FFF; line-height:1; }
  .badge-off-p { font-size:1.4vw; font-weight:800; color:var(--accent-tint); letter-spacing:1px; }
  .info-block { background:rgba(20,10,5,0.75); backdrop-filter:blur(18px); border:1px solid rgba(var(--accent-rgb),0.25); border-radius:20px; padding:5vw; display:flex; flex-direction:column; gap:2.5vw; }
  .deal-title { font-size:1.8vw; font-weight:800; color:#FFF; }
  .deal-sub { font-size:1.2vw; font-weight:500; color:var(--accent); }
  .cta-btn { text-align:center; background:linear-gradient(135deg,var(--accent-darker) 0%,var(--accent) 100%); color:#000; font-weight:900; font-size:1.5vw; letter-spacing:0.5px; padding:3.5vw; border-radius:999px; text-decoration:none; box-shadow:0 8px 22px rgba(var(--accent-darker-rgb),0.5); }
</style></head>
<body><div class="poster">
  <div class="header"><div class="brand-title">$brand</div><div class="date-pill">$date_range</div></div>
  <div class="bottom-stack"$safe_rect_style>
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
        palette = cls._derive_theme_palette(brief.get("brand_color"), cls._GRAND_OPENING_DEFAULT_PALETTE)
        return cls._GRAND_OPENING_PORTRAIT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            brand=brief.get("brand", "THE BURGER CRAFT"),
            date_range=brief.get("date_range", "05.09 - 15.09"),
            badge_label=brief.get("badge_label", "GIẢM"),
            badge_percent=brief.get("badge_percent", "50%"),
            badge_sub=brief.get("badge_sub", "TOÀN MENU"),
            address=brief.get("address", "128 Nguyễn Trãi, Q1"),
            offer_desc=brief.get("offer_desc", "Tặng 01 Coca-Cola cho hóa đơn từ 99K • Hotline: 1900 8899"),
            cta_text=brief.get("cta_text", "NHẬN VOUCHER"),
            safe_rect_style=cls._safe_rect_style_attr(brief),
            **palette,
        )

    _FEEDBACK_PORTRAIT_TPL = Template("""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Feedback (Portrait)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Dancing+Script:wght@700&family=Playfair+Display:ital,wght@0,700;1,400&family=Oswald:wght@600;700&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:100vw; height:100vh; font-family:'Plus Jakarta Sans', sans-serif; }
  .poster {
    position:relative; width:${w}px; height:${h}px; overflow:hidden; $bg_css
    --accent: $accent; --accent-dark: $accent_dark; --accent-tint: $accent_tint; --accent-tint2: $accent_tint2; --accent-rgb: $accent_rgb;
    --secondary: $secondary; --secondary-tint-rgb: $secondary_tint_rgb;
    --mood-quote-font: $mood_quote_font; --mood-name-font: $mood_name_font;
  }
  .top-bar { position:absolute; top:4%; left:6%; right:6%; display:flex; justify-content:space-between; align-items:center; gap:10px; z-index:20; }
  .spa-logo { font-family:'Quicksand',sans-serif; font-size:4vw; font-weight:800; color:var(--accent); max-width:60%; line-height:1.2; text-shadow:0 2px 6px rgba(255,255,255,0.6); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
  .spa-badge { background:rgba(var(--secondary-tint-rgb),0.4); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); color:var(--secondary); font-weight:800; font-size:2.4vw; padding:1.6vw 2.8vw; border-radius:999px; border:1px solid rgba(var(--secondary-tint-rgb),0.55); white-space:nowrap; flex-shrink:0; }
  /* REDESIGNED (2026-09-05, per direct user request) -- see landscape .corner-card comment:
     was a near-full-width stack covering most of the photo; now one small corner card so the
     customer/before-after photo stays the visual hero. */
  .corner-card {
    position:absolute; left:6%; bottom:5%; width:58%;
    background:rgba(255,255,255,0.55); backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px);
    border:1.5px solid rgba(255,255,255,0.6); border-radius:18px; padding:4.5vw 4.5vw 5vw;
    display:flex; flex-direction:column; gap:2.2vw;
    box-shadow:0 2.5vw 5vw rgba(0,0,0,0.28);
    z-index:20;
  }
  .review-header-mini { display:flex; justify-content:space-between; align-items:center; }
  .stars-mini { color:#F59E0B; font-size:3.6vw; letter-spacing:1px; }
  .verified-mini { font-size:2.2vw; font-weight:700; color:var(--accent-dark); background:var(--accent-tint); padding:0.8vw 2.2vw; border-radius:999px; white-space:nowrap; }
  .quote-mini { font-family:var(--mood-quote-font); font-size:2.9vw; line-height:1.36; color:#374151; font-weight:500; font-style:italic; display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical; overflow:hidden; }
  .customer-mini { display:flex; align-items:center; gap:2.2vw; }
  .avatar-mini { width:7.5vw; height:7.5vw; border-radius:50%; background:var(--accent-tint2); display:flex; justify-content:center; align-items:center; font-size:3.6vw; border:1.5px solid var(--accent); flex-shrink:0; }
  .cust-name-mini { font-family:var(--mood-name-font); font-size:2.9vw; font-weight:700; color:#111928; line-height:1.2; }
  .cust-sub-mini { font-size:2.3vw; color:#6B7280; font-weight:500; }
  .tags-mini { display:flex; flex-wrap:wrap; gap:1.4vw; }
  .tag-mini { font-size:2.2vw; font-weight:600; color:#1F2A37; background:rgba(255,255,255,0.55); border:1px solid rgba(229,231,235,0.7); border-radius:2.2vw; padding:1vw 2.2vw; line-height:1.3; max-width:100%; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
  .cta-row-mini { display:flex; align-items:center; justify-content:space-between; gap:2vw; padding-top:1.5vw; border-top:1px solid rgba(0,0,0,0.07); }
  .offer-mini { font-size:2.5vw; font-weight:700; color:var(--accent-dark); line-height:1.25; flex:1; min-width:0; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
  .cta-mini { flex-shrink:0; background:linear-gradient(135deg,var(--accent) 0%,var(--accent-dark) 100%); color:#FFFFFF; font-weight:700; font-size:2.6vw; padding:1.6vw 3.2vw; border-radius:999px; text-decoration:none; white-space:nowrap; }

  /* REVERTED the 2026-09-06 blanket shrink (width 58%->42% + every font scaled 0.8x): user pointed
     at line21_real_gpt4o_render.png and clarified the actual preference -- the CARD's own internal
     text (quote/name/tags) should shrink, but the BRAND TITLE (`.spa-logo`, in `.top-bar`, outside
     `.corner-card`) must stay large/prominent regardless. A single blanket scale factor couldn't
     express that (it shrank the title along with everything else). Replaced with an OPT-IN
     `.compact-card` class (driven by `brief["card_size"] == "compact"`) that overrides ONLY
     `.corner-card` and its children below -- `.spa-logo`/`.spa-badge` are never touched by it, so
     the title stays exactly as prominent in compact mode as in normal mode. Default (`card_size`
     absent/"normal") renders BYTE-IDENTICAL to before this whole compact-card exploration began. */
  .poster.compact-card .corner-card { width:42%; max-width:380px; padding:3.6vw 3.6vw 4vw; gap:1.8vw; box-shadow:0 2vw 4vw rgba(0,0,0,0.28); border-radius:16px; }
  .poster.compact-card .stars-mini { font-size:2.9vw; }
  .poster.compact-card .verified-mini { font-size:1.8vw; padding:0.6vw 1.8vw; }
  .poster.compact-card .quote-mini { font-size:2.3vw; }
  .poster.compact-card .avatar-mini { width:6vw; height:6vw; font-size:2.9vw; }
  .poster.compact-card .cust-name-mini { font-size:2.3vw; }
  .poster.compact-card .cust-sub-mini { font-size:1.8vw; }
  .poster.compact-card .tags-mini { gap:1.1vw; }
  .poster.compact-card .tag-mini { font-size:1.8vw; border-radius:1.8vw; padding:0.8vw 1.8vw; }
  /* Stacked (column), not the base row layout -- REAL bug found testing the original shrink: at
     42% width there's no longer room for offer text + CTA button side by side (text wrapped 1-2
     characters per line). Only applied in compact mode -- normal mode's 58%-wide row layout never
     had this problem. */
  .poster.compact-card .cta-row-mini { flex-direction:column; align-items:stretch; gap:1.6vw; padding-top:1.2vw; }
  .poster.compact-card .offer-mini { font-size:2vw; flex:none; }
  .poster.compact-card .cta-mini { font-size:2.1vw; padding:1.3vw 2.6vw; flex-shrink:1; text-align:center; }
</style></head>
<body><div class="poster$compact_class">
  <div class="top-bar"><div class="spa-logo">$brand</div>$badge_html</div>
  <div class="corner-card"$safe_rect_style>
    <div class="review-header-mini">$stars_html<span class="verified-mini">$verified_label</span></div>
    <div class="quote-mini">$quote_text</div>
    <div class="customer-mini"><div class="avatar-mini">$avatar_emoji</div><div><div class="cust-name-mini">$customer_name</div><div class="cust-sub-mini">$customer_sub</div></div></div>
    $features_block_html
    <div class="cta-row-mini">$offer_html<a href="#" class="cta-mini">$cta_text</a></div>
  </div>
</div></body></html>""")

    @classmethod
    def _generate_feedback_card_portrait(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        conditional = cls._build_feedback_conditional_html(brief, top_badge_default="CHUẨN HÀN QUỐC")
        palette = cls._derive_theme_palette(
            brief.get("brand_color"), cls._FEEDBACK_DEFAULT_PALETTE, secondary_hex=brief.get("secondary_color"),
        )
        font_mood = cls._resolve_font_mood(brief.get("font_mood"))
        compact_class = " compact-card" if brief.get("card_size") == "compact" else ""
        return cls._FEEDBACK_PORTRAIT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            compact_class=compact_class,
            brand=brief.get("brand", "PAWPARADISE SPA"),
            verified_label=brief.get("verified_label", "ĐÃ TRẢI NGHIỆM"),
            quote_text=brief.get("quote_text", "Dịch vụ tuyệt vời, nhân viên chuyên nghiệp và tận tâm, chắc chắn sẽ quay lại!"),
            avatar_emoji=brief.get("avatar_emoji", "P"),
            customer_name=brief.get("customer_name", "Khách hàng thân thiết"),
            customer_sub=brief.get("customer_sub", "Đã trải nghiệm dịch vụ Premium"),
            cta_text=brief.get("cta_text", "ĐẶT LỊCH NGAY"),
            safe_rect_style=cls._corner_card_top_override(brief),
            mood_quote_font=font_mood["quote_font"], mood_name_font=font_mood["name_font"],
            **conditional,
            **palette,
        )

    _RECRUITMENT_PORTRAIT_TPL = Template("""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Recruitment (Portrait)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Montserrat:wght@700;800;900&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:100vw; height:100vh; font-family:'Plus Jakarta Sans', sans-serif; }
  .poster {
    position:relative; width:${w}px; height:${h}px; overflow:hidden; $bg_css
    isolation: isolate;
    --accent: $accent; --accent-dark: $accent_dark; --accent-darker: $accent_darker; --accent-dark-rgb: $accent_dark_rgb;
  }
  /* Font-sizes recalibrated (2026-09-06, same audit/fix as grand_opening portrait above): every
     `vw` value here independently produced LARGER real pixels than the landscape version despite
     this canvas being smaller (1024px vs 1536px wide) -- vw = landscape_px * 0.9 / 1024 * 100. */
  .rec-header { position:absolute; top:3.5%; left:6%; right:6%; display:flex; justify-content:space-between; align-items:center; gap:10px; z-index:20; }
  .company-logo { font-size:1.9vw; font-weight:900; color:var(--accent); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:60%; }
  .urgency-badge { background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#F87171; font-size:1.1vw; font-weight:700; padding:1vw 2vw; border-radius:999px; white-space:nowrap; }
  .frosted-box { position:absolute; bottom:3%; left:6%; right:6%; background:rgba(15,23,42,0.7); backdrop-filter:blur(20px); border:1px solid rgba(255,255,255,0.14); border-radius:22px; padding:5vw; display:flex; flex-direction:column; gap:3vw; max-height:70%; z-index:20; }
  .salary-tag { align-self:flex-start; background:rgba(var(--accent-dark-rgb),0.45); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); color:#FFF; font-weight:800; font-size:1.8vw; padding:1.3vw 2.6vw; border-radius:12px; border:1px solid rgba(255,255,255,0.3); }
  .col-title { font-size:1.4vw; font-weight:800; color:#94A3B8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:1.5vw; }
  .checklist { list-style:none; display:flex; flex-direction:column; gap:2vw; }
  .check-item { display:flex; align-items:flex-start; gap:2vw; font-size:1.3vw; color:#E2E8F0; line-height:1.4; font-weight:500; }
  .check-icon { color:var(--accent); font-weight:900; }
  .rec-footer { display:flex; flex-direction:column; gap:2vw; border-top:1px solid rgba(255,255,255,0.1); padding-top:3vw; }
  .contact-email { color:var(--accent); font-weight:700; font-size:1.4vw; }
  .apply-btn { text-align:center; background:linear-gradient(135deg,var(--accent) 0%,var(--accent-dark) 100%); color:#020617; font-weight:800; font-size:1.5vw; padding:3.2vw; border-radius:999px; text-decoration:none; }
</style></head>
<body><div class="poster">
  $giant_title_html
  <div class="rec-header"><div class="company-logo">$company</div><div class="urgency-badge">$deadline</div></div>
  <div class="frosted-box"$safe_rect_style>
    <div class="salary-tag">$salary</div>
    <div>
      <div class="col-title">$pos_label</div>
      <ul class="checklist">$requirements_html</ul>
    </div>
    <div>
      <div class="col-title">QUYỀN LỢI</div>
      <ul class="checklist">$benefits_html</ul>
    </div>
    <div class="rec-footer">
      <div style="font-size:1.2vw;color:#94A3B8;">$contact_line1</div>
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
        req_html = "".join(f'<li class="check-item"><span class="check-icon"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="display:inline-block; vertical-align:-1px; color:#10B981;"><polyline points="20 6 9 17 4 12"></polyline></svg></span><span>{r}</span></li>' for r in requirements)
        ben_html = "".join(f'<li class="check-item"><span class="check-icon"><svg width="12" height="12" viewBox="0 0 24 24" fill="#FFB300"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg></span><span>{b}</span></li>' for b in benefits)
        palette = cls._derive_theme_palette(brief.get("brand_color"), cls._RECRUITMENT_DEFAULT_PALETTE)
        company = brief.get("company", "TENDOO AI LAB")
        giant_title_html = cls._giant_title_block(
            analysis, background_image_path, company,
            left_px=round(analysis.width * 0.04), top_px=round(analysis.height * 0.11),
            width_px=round(analysis.width * 0.92), height_px=round(analysis.height * 0.55),
            z_index=1, font_clamp="clamp(40px, 13vw, 140px)",
        )
        return cls._RECRUITMENT_PORTRAIT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            company=company,
            deadline=brief.get("deadline", "HẠN: 30.09"),
            pos_label=brief.get("pos_label", "YÊU CẦU ỨNG VIÊN"),
            salary=brief.get("salary", "THOẢ THUẬN"),
            requirements_html=req_html,
            benefits_html=ben_html,
            contact_line1=brief.get("contact_line1", "Gửi CV & Portfolio:"),
            contact_email=brief.get("contact_email", "careers@tendoo.ai"),
            cta_text=brief.get("cta_text", "ỨNG TUYỂN NGAY"),
            safe_rect_style=cls._safe_rect_style_attr(brief),
            giant_title_html=giant_title_html,
            **palette,
        )

    _MENU_PORTRAIT_TPL = Template("""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Menu (Portrait)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:100vw; height:100vh; font-family:'Plus Jakarta Sans', sans-serif; }
  .poster {
    position:relative; width:${w}px; height:${h}px; overflow:hidden; $bg_css padding:5vw; display:flex; flex-direction:column; justify-content:flex-end; gap:3vw;
    isolation: isolate;
    --accent: $accent; --accent-dark: $accent_dark; --accent-rgb: $accent_rgb;
  }
  /* Font-sizes recalibrated (2026-09-06, same audit/fix as grand_opening/recruitment portrait) --
     vw = landscape_px * 0.9 / 1024 * 100, see grand_opening portrait's comment for the full story. */
  .sub-brand { font-size:1.2vw; font-weight:700; color:var(--accent-dark); letter-spacing:2px; text-transform:uppercase; text-align:center; }
  .menu-desc { font-style:italic; font-size:1.3vw; color:#E7E5E4; text-align:center; text-shadow:0 2px 6px rgba(0,0,0,0.6); }
  .menu-stack { display:flex; flex-direction:column; gap:5vw; background:rgba(10,6,4,0.6); backdrop-filter:blur(14px); border-radius:20px; padding:5vw; max-height:60%; overflow:hidden; }
  .cat-title { font-family:'Playfair Display',serif; font-size:1.9vw; font-weight:700; color:var(--accent); border-bottom:1px solid rgba(var(--accent-rgb),0.3); padding-bottom:1.5vw; margin-bottom:2vw; }
  .item-list { display:flex; flex-direction:column; gap:2.5vw; }
  .menu-row { display:flex; flex-direction:column; gap:0.5vw; }
  .row-top { display:flex; align-items:baseline; justify-content:space-between; gap:2vw; }
  .item-name { font-size:1.4vw; font-weight:700; color:#FFFFFF; }
  .dotted-line { flex-grow:1; border-bottom:1px dotted rgba(255,255,255,0.3); margin:0 1vw; }
  .item-price { font-family:'Playfair Display',serif; font-size:1.6vw; font-weight:700; color:var(--accent); white-space:nowrap; }
  .badge-star { font-size:0.9vw; font-weight:800; background:rgba(239,68,68,0.45); backdrop-filter:blur(5px); -webkit-backdrop-filter:blur(5px); color:#FFF; padding:0.4vw 1.2vw; border-radius:4px; margin-left:1.5vw; }
  .menu-footer { background:rgba(var(--accent-rgb),0.1); border:1px solid rgba(var(--accent-rgb),0.25); border-radius:14px; padding:3vw 4vw; display:flex; flex-direction:column; gap:1.5vw; }
  .foot-note { font-size:1.2vw; color:#FFFFFF; display:flex; align-items:center; gap:1vw; }
  .foot-hotline { font-weight:700; color:var(--accent); font-size:1.3vw; display:flex; align-items:center; gap:1vw; }
</style></head>
<body><div class="poster">
  $giant_title_html
  <div class="sub-brand">$sub_brand</div>
  <div class="menu-desc">$tagline</div>
  <div class="menu-stack">$categories_html</div>
  <div class="menu-footer">
    <div class="foot-note"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>$footer_note</div>
    <div class="foot-hotline"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>$hotline</div>
  </div>
</div></body></html>""")

    @classmethod
    def _generate_menu_portrait(cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None) -> str:
        categories = brief.get("categories", [
            {"title": "MÓN CHÍNH", "items": [
                {"name": "Món Đặc Trưng", "price": "89.000đ", "badge": "BEST SELLER"},
                {"name": "Món Signature", "price": "149.000đ"},
            ]},
            {"title": "ĐỒ UỐNG", "items": [
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
        palette = cls._derive_theme_palette(brief.get("brand_color"), cls._MENU_DEFAULT_PALETTE)
        sub_brand = brief.get("sub_brand", "ARTISAN DINING EXPERIENCE")
        giant_title_html = cls._giant_title_block(
            analysis, background_image_path, sub_brand,
            left_px=round(analysis.width * 0.06), top_px=round(analysis.height * 0.05),
            width_px=round(analysis.width * 0.88), height_px=round(analysis.height * 0.38),
            z_index=-1, font_clamp="clamp(36px, 11vw, 130px)",
        )
        return cls._MENU_PORTRAIT_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            sub_brand=sub_brand,
            tagline=brief.get("tagline", "Thưởng thức tinh hoa ẩm thực thủ công"),
            categories_html="".join(cat_html_parts),
            footer_note=brief.get("footer_note", "Giảm 10% khi check-in tại quán"),
            hotline=brief.get("hotline", "Hotline: 1800 8198"),
            giant_title_html=giant_title_html,
            **palette,
        )

    # ----------------------------------------------------------------------------------------
    # PRODUCT_AD -- the "100%-overlay" direction's core template: simple title+subtitle over a
    # full-bleed photo, matching the exact shape of prompt_test.txt lines 1-19 (every one of them
    # is just 2 positioned/styled text blocks, nothing more). Unlike the diffusion-hero direction,
    # NEITHER block is baked into the photo -- both are pure CSS, using the hero style presets
    # validated in scripts/test_css_hero_title_styles.py (neon_glow confirmed genuinely
    # competitive with diffusion on a dark/neon scene; metallic_3d/gold_foil are solid but lack
    # environment-color reflection; dark-on-light styles for bright/pastel scenes are NOT YET
    # built -- see memory css-hero-title-overlay-direction.md).
    # ----------------------------------------------------------------------------------------

    HERO_STYLE_CSS = {
        "metallic_3d": """
            background: linear-gradient(180deg, #FFFFFF 0%, #D8D8D8 35%, #8A8A8A 55%, #C8C8C8 70%, #FFFFFF 100%);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;
            filter: drop-shadow(0px 1px 0px #b0b0b0) drop-shadow(0px 2px 0px #999999) drop-shadow(0px 3px 0px #808080)
                    drop-shadow(0px 4px 0px #666666) drop-shadow(0px 5px 2px rgba(0,0,0,0.5)) drop-shadow(0px 8px 14px rgba(0,0,0,0.6));
        """,
        "neon_glow": """
            color: #FFFFFF;
            text-shadow: 0 0 4px #FFFFFF, 0 0 10px #FFFFFF, 0 0 18px #00F0FF, 0 0 34px #00F0FF,
                         0 0 60px #00B8FF, 0 2px 2px rgba(0,0,0,0.4);
        """,
        "neon_glow_pink": """
            color: #FFFFFF;
            text-shadow: 0 0 4px #FFFFFF, 0 0 10px #FFFFFF, 0 0 18px #FF3DAE, 0 0 34px #FF3DAE,
                         0 0 60px #E000A0, 0 2px 2px rgba(0,0,0,0.4);
        """,
        "gold_foil": """
            background: linear-gradient(180deg, #FFF6D8 0%, #F5D485 20%, #C9971F 45%, #FFE9A8 55%, #B8860B 75%, #FFF2C4 100%);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;
            filter: drop-shadow(0px 1px 0px rgba(255,255,255,0.5)) drop-shadow(0px 3px 4px rgba(0,0,0,0.5)) drop-shadow(0px 6px 14px rgba(0,0,0,0.55));
        """,
        "plain_light": "color: #FFFFFF; text-shadow: 0 2px 14px rgba(0,0,0,0.55);",
        "plain_dark": "color: #1A1208; text-shadow: 0 2px 10px rgba(255,255,255,0.5);",
        # --- Dark-on-light styles, for bright/pastel scenes (prompt_test.txt lines 5/7/17/23/
        # 29/33/39...) where the light-on-dark trio above has no contrast to work with. ---
        "embossed_dark": """
            background: linear-gradient(180deg, #4a4a4a 0%, #2b2b2b 40%, #1a1a1a 70%, #3a3a3a 100%);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;
            filter: drop-shadow(0px -1px 0px rgba(255,255,255,0.65)) drop-shadow(0px 1px 1px rgba(255,255,255,0.3))
                    drop-shadow(0px 2px 3px rgba(0,0,0,0.22)) drop-shadow(0px 5px 10px rgba(0,0,0,0.16));
        """,
        "gold_deep": """
            background: linear-gradient(180deg, #8a6a1f 0%, #b8860b 25%, #6b4d0a 55%, #a17d1a 75%, #4a3407 100%);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;
            filter: drop-shadow(0px 1px 0px rgba(255,255,255,0.45)) drop-shadow(0px 2px 3px rgba(0,0,0,0.22))
                    drop-shadow(0px 5px 10px rgba(0,0,0,0.18));
        """,
        "pastel_pop": """
            color: #D6318F;
            -webkit-text-stroke: 2px #FFFFFF; paint-order: stroke fill;
            text-shadow: 0 3px 8px rgba(0,0,0,0.18), 0 1px 3px rgba(0,0,0,0.14);
        """,
    }

    # Which style each zone's dark/light reading should prefer when the caller doesn't pin one
    # explicitly -- see _auto_pick_style().
    _LIGHT_BG_STYLE_ORDER = ["embossed_dark", "gold_deep", "pastel_pop"]
    _DARK_BG_STYLE_ORDER = ["neon_glow", "metallic_3d", "gold_foil"]

    # Real bug found+fixed after the first real Stage-1 (gpt-4o-mini) pipeline run (2026-09-05, see
    # memory css-hero-title-overlay-direction.md): Stage 1 isn't told a strict enum for
    # title_style/subtitle_style, so it invents plausible-sounding-but-invalid names ("elegant",
    # "bold", "adventure"...). The OLD code (`brief.get("title_style") or _auto_pick_style(...)`)
    # treated ANY non-empty string as "caller has an opinion", so an invalid name never fell
    # through to auto-pick -- it silently hit HERO_STYLE_CSS.get(title_style, ...["plain_light"]),
    # turning every poster flat white with zero metallic/neon/gold/glow effect. Same root cause hit
    # title_font/subtitle_font: a generic CSS keyword like "serif"/"sans-serif" (not a real font
    # name) got quoted as a specific font-family in the template (`font-family: '$title_font',
    # sans-serif;` -> `font-family: 'serif', sans-serif;`), so the browser can't find a font
    # literally named "serif" and silently falls through to the generic sans-serif at the end --
    # the requested serif LOOK is lost even though nothing crashed or looked obviously broken.
    _GENERIC_FONT_FALLBACK = {
        "serif": "Playfair Display", "sans-serif": "Montserrat", "monospace": "Montserrat",
        "cursive": "Dancing Script", "fantasy": "Montserrat", "system-ui": "Montserrat",
    }

    @classmethod
    def _resolve_font(cls, font_name: Optional[str], default: str) -> str:
        """Returns `font_name` as-is if it looks like a real font name, or a curated real font
        matching the intent if it's actually a generic CSS family keyword, or `default` if empty.
        A genuine (if uncommon) font name that isn't in this file's @import list still degrades
        safely -- the browser just falls through to the stack's own trailing generic keyword,
        which is the normal/correct CSS behavior; only the generic-keyword-as-a-name case above is
        actually broken and worth guarding against here."""
        if not font_name:
            return default
        return cls._GENERIC_FONT_FALLBACK.get(font_name.strip().lower(), font_name)

    @classmethod
    def _auto_pick_style(cls, analysis: BackgroundAnalysis, position: str, style_hint: Optional[str] = None) -> str:
        """
        Picks a HERO_STYLE_CSS key based on the ACTUAL measured luminance of the zone the text
        will land in (not a guess made before the image existed) -- reads header/center/footer
        zone from PosterBackgroundAnalyzer depending on the position's vertical component. Prefers
        `style_hint` (e.g. "neon", "gold", "metallic", "embossed", "pastel") if it names a style in
        the right light/dark family; otherwise defaults to that family's first (best-tested) entry.
        """
        v = position.replace("_", "-").split("-")[0]
        zone = {"top": analysis.header_zone, "middle": analysis.center_zone, "bottom": analysis.footer_zone}.get(v, analysis.center_zone)
        family = cls._LIGHT_BG_STYLE_ORDER if not zone.is_dark else cls._DARK_BG_STYLE_ORDER

        if style_hint:
            hint = style_hint.lower()
            for key in family:
                if hint in key:
                    return key
        return family[0]

    @staticmethod
    def _zone_css(position: str) -> str:
        """
        Maps a canonical 3x3 zone name (matching the "góc trên bên trái", "ở giữa phía trên"...
        position language prompt_test.txt itself uses) to absolute-positioning CSS. Falls back to
        top-center for an unrecognized value rather than raising, since this is fed by an LLM
        blueprint that could occasionally emit something slightly off-spec.
        """
        parts = position.replace("_", "-").split("-")
        v = parts[0] if len(parts) > 0 else "top"
        h = parts[1] if len(parts) > 1 else "center"
        css = "position:absolute; "
        css += {"top": "top:6%;", "middle": "top:50%; transform:translateY(-50%);", "bottom": "bottom:6%;"}.get(v, "top:6%;")
        css += {"left": " left:6%; right:auto; text-align:left;",
                "right": " right:6%; left:auto; text-align:right;",
                "center": " left:6%; right:6%; text-align:center;"}.get(h, " left:6%; right:6%; text-align:center;")
        return css

    # Below this height fraction of the canvas, a detected safe_rect (Cấp độ 2 MER,
    # src/tendoo/layout_geometry.py) is considered too cramped to trust for a title -- falls back
    # to the fixed zone rather than squeezing text into a sliver. Not a fully-designed fallback
    # policy (that decision -- shrink content vs. accept light touch/overlap -- is still open per
    # memory css-hero-title-overlay-direction.md), just a minimum sanity guard.
    MIN_SAFE_RECT_HEIGHT_PCT = 10.0

    # Below this WIDTH fraction of the canvas, MER's raw output is rejected for a title even if
    # its area/height look fine -- a narrow-tall rectangle can be a perfectly valid zero-overlap
    # answer geometrically while still forcing an ugly extra line wrap, since title text needs
    # width more than height (unlike the bottom-stack cards MER was originally scoped for). Value
    # is grounded in 2 real before/after comparisons (scripts/probe_mer_representative_cases.py,
    # see memory css-hero-title-overlay-direction.md): a 37%-width rect forced a cramped 3-line
    # wrap (prompt7 fashion-portrait case), while 46.8-48.3%-width rects gave a clean 2-line wrap
    # (Case C hiker case) -- 40% sits between the two, closer to the "bad" data point so it stays
    # conservative until more real data points narrow it further.
    MIN_SAFE_RECT_WIDTH_PCT = 40.0

    @classmethod
    def _zone_css_from_rect(cls, rect_pct: Dict[str, float]) -> Optional[str]:
        """
        Builds absolute-positioning CSS directly from an EmptyRect.as_css_percent()-shaped dict
        (src/tendoo/layout_geometry.py's Cấp độ 2 MER output: left_pct/top_pct/right_pct/
        bottom_pct/height_pct/width_pct) instead of a fixed 3x3 zone -- lets a detected
        product/hero bounding box on THIS specific generated image move the text out of the way,
        rather than guessing a zone ahead of time (see prompt_test.txt line 1's demonstrated case:
        a middle-left title zone overlapped the product because nothing checked where the product
        actually landed). Returns None (caller should fall back to _zone_css) if the rect is
        missing required keys, too short (MIN_SAFE_RECT_HEIGHT_PCT), or too NARROW
        (MIN_SAFE_RECT_WIDTH_PCT) to trust for a title -- both are real, independently-measured
        failure modes, not just a theoretical concern (see round-2 probe in memory
        css-hero-title-overlay-direction.md).
        """
        try:
            top, left, right = rect_pct["top_pct"], rect_pct["left_pct"], rect_pct["right_pct"]
            height, width = rect_pct["height_pct"], rect_pct["width_pct"]
        except (KeyError, TypeError):
            return None
        if height < cls.MIN_SAFE_RECT_HEIGHT_PCT or width < cls.MIN_SAFE_RECT_WIDTH_PCT:
            return None
        return f"position:absolute; top:{top}%; left:{left}%; right:{right}%; text-align:center;"

    _PRODUCT_AD_TPL = Template("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Playfair+Display:wght@700;800;900&family=Dancing+Script:wght@700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { width: 100vw; height: 100vh; }
  .poster { position: relative; width: ${w}px; height: ${h}px; overflow: hidden; $bg_css }
  .title-text {
    font-family: '$title_font', sans-serif; font-weight: 900; font-size: $title_size;
    line-height: 1.15; letter-spacing: 0.5px; text-transform: $title_transform;
    $title_style_css
  }
  .subtitle-text {
    font-family: '$subtitle_font', sans-serif; font-weight: 600; font-size: $subtitle_size;
    line-height: 1.3; letter-spacing: 0.5px;
    $subtitle_style_css
  }
  .stack-gap { margin-top: 2.5%; }
</style></head>
<body>
  <div class="poster">
    $body_html
  </div>
</body></html>""")

    # DESIGN_PRINCIPLES.md #1 (typographic scale): subtitle size should derive from title size via
    # a fixed modular-scale ratio, not an unrelated hand-picked literal (the old default pairing,
    # clamp(30px,7vw,64px) title vs clamp(14px,3vw,24px) subtitle, is a ~2.67x ratio that matches
    # none of the standard scales). Premium/metallic styles get the more dramatic Golden Ratio;
    # everything else gets the safer Perfect Fourth.
    _PREMIUM_STYLE_FAMILY = {"gold_foil", "gold_deep", "metallic_3d"}
    _CLAMP_RE = re.compile(r"clamp\(\s*([\d.]+)(\w+)\s*,\s*([\d.]+)(\w+)\s*,\s*([\d.]+)(\w+)\s*\)")

    @classmethod
    def _modular_ratio(cls, title_style: str) -> float:
        return 1.618 if title_style in cls._PREMIUM_STYLE_FAMILY else 1.333

    @classmethod
    def _scaled_clamp_str(cls, base_clamp: str, ratio: float) -> str:
        """Derives a proportionally-smaller clamp() string from a larger one using `ratio`,
        instead of an unrelated hand-picked literal. Falls back to `base_clamp` unchanged if it
        doesn't parse as clamp(...) -- backward-compatible with a raw px/vw value."""
        m = cls._CLAMP_RE.match(base_clamp.strip())
        if not m:
            return base_clamp
        parts = []
        for i in range(0, 6, 2):
            num, unit = m.group(i + 1), m.group(i + 2)
            scaled = float(num) / ratio
            parts.append(f"{scaled:.2f}".rstrip("0").rstrip(".") + unit)
        return f"clamp({parts[0]}, {parts[1]}, {parts[2]})"

    # DESIGN_PRINCIPLES.md #5 (legibility scrim): a soft gradient behind the title, not just
    # text-shadow -- the 4 card-heavy templates already have `backdrop-filter:blur()`, product_ad
    # had nothing. Keyed off the TEXT's own light/dark style family (not the raw zone luminance),
    # so an explicit style override still gets a scrim that actually helps THAT text stay legible.
    # Wraps the title content in an inline-block inner div (shrink-wraps to the text's own size)
    # so the scrim hugs the text rather than spanning the whole (often much wider) positioned zone
    # -- deliberately NOT touching the outer zone div's own position/transform math.
    @classmethod
    def _title_scrim_css(cls, title_style: str) -> str:
        if title_style in cls._DARK_BG_STYLE_ORDER:
            tint = "0,0,0"
        else:
            tint = "255,255,255"
        return (
            f"display:inline-block; background: linear-gradient(to bottom, rgba({tint},0.32), rgba({tint},0)); "
            f"border-radius: 12px; padding: 1.5% 2%;"
        )

    @classmethod
    def _generate_product_ad(
        cls, analysis: BackgroundAnalysis, brief: Dict[str, Any], background_image_path: Optional[str] = None
    ) -> str:
        title_position = brief.get("title_position", "top-center")
        subtitle_position = brief.get("subtitle_position")  # None -> stack under title (most prompts: "phía dưới")

        # Style: honor an explicit pick ONLY if it's a real key in HERO_STYLE_CSS, else auto-select
        # from the ACTUAL measured luminance of the zone the text lands in (PosterBackgroundAnalyzer
        # already computed this from the real generated image -- no need to guess blind before the
        # image existed). Validated by MEMBERSHIP, not truthiness -- an invalid/invented style name
        # from Stage 1 (e.g. "elegant", "bold") must fall through to auto-pick, not silently resolve
        # to plain_light at the CSS lookup below (see _GENERIC_FONT_FALLBACK's comment for the real
        # bug this fixes).
        title_style_raw = brief.get("title_style")
        title_style = title_style_raw if title_style_raw in cls.HERO_STYLE_CSS else cls._auto_pick_style(
            analysis, title_position, brief.get("style_theme")
        )
        subtitle_style_raw = brief.get("subtitle_style")
        subtitle_style = subtitle_style_raw if subtitle_style_raw in cls.HERO_STYLE_CSS else cls._auto_pick_style(
            analysis, subtitle_position or title_position, brief.get("style_theme")
        )

        title_size = brief.get("title_size", "clamp(30px, 7vw, 64px)")
        default_subtitle_size = cls._scaled_clamp_str(title_size, cls._modular_ratio(title_style))

        title_html = f'<div class="title-text">{brief.get("title_text", "TIÊU ĐỀ SẢN PHẨM")}</div>'
        subtitle_html = f'<div class="subtitle-text">{brief.get("subtitle_text", "Dòng mô tả phụ")}</div>'
        scrim_css = cls._title_scrim_css(title_style)

        # Cấp độ 2 (detection + MER, src/tendoo/layout_geometry.py) integration: if the caller
        # already ran detection on THIS generated image and passed the resulting safe rectangle,
        # it overrides the fixed title_position zone -- otherwise fall back to the static 3x3
        # grid as before (fully backward compatible; safe_rect is optional).
        safe_rect = brief.get("safe_rect")
        title_zone_css = (cls._zone_css_from_rect(safe_rect) if safe_rect else None) or cls._zone_css(title_position)

        # REAL BUG found testing a live Stage 1 (gpt-4o) call: told to "BỎ TRỐNG subtitle_position"
        # (leave it blank) for the stacked case, the model emitted an EMPTY STRING "" rather than
        # omitting the key or using JSON null. `subtitle_position is None` doesn't catch "" (a
        # different, truthy-in-JSON-terms value that isn't equal to title_position either), so the
        # check below silently fell through to the INDEPENDENT branch -- title and subtitle got two
        # separate absolutely-positioned boxes (title at the safe_rect's top, subtitle hardcoded at
        # a fixed `top:6%`) instead of one stacked block, and the two visibly overlapped/collided
        # once the title wrapped to 2 lines. Using a falsy check (not just `is None`) treats ""
        # the same as an omitted/null value -- matches run_full_pipeline.py's own
        # `subtitle_position and subtitle_position != title_position` check, which already used
        # `and` (falsy-safe) rather than `is not None`.
        if not subtitle_position or subtitle_position == title_position:
            # Stacked mode: both blocks share ONE zone, subtitle flows directly below title --
            # matches the dominant prompt_test.txt pattern ("Ở góc trên... Phía dưới...", i.e.
            # subtitle is relative to title's position, not an independently-placed zone).
            body_html = (
                f'<div style="{title_zone_css}"><div style="{scrim_css}">'
                f'{title_html}<div class="stack-gap">{subtitle_html}</div></div></div>'
            )
        else:
            # Independent mode: title and subtitle explicitly occupy different zones (e.g.
            # prompt_test.txt line 1's inverted case -- small subtitle top-left, larger title
            # middle-left). REAL BUG found+fixed after the first real-pipeline run (2026-09-05,
            # see memory): `safe_rect` only ever protected the TITLE zone here -- the subtitle's
            # independent zone always used the plain fixed 3x3 grid, completely unprotected by
            # Cấp độ 2, even when a real product/obstacle sat exactly where that fixed zone landed
            # (prompt7-line1's own case: "middle-left" subtitle rendered squarely over the coffee
            # cup). Made worse by (1.1)'s modular-scale change, which made the subtitle noticeably
            # bigger by default -- a bigger box in an unprotected fixed zone collides more often.
            # Fix: honor `brief["subtitle_safe_rect"]` (a SEPARATE MER result computed for the
            # subtitle's own anchor band -- see scripts/run_full_pipeline.py's call_stage3) the same
            # way `safe_rect` protects the title; falls back to the old fixed zone if absent, so
            # this is fully backward-compatible for any caller that doesn't compute one.
            subtitle_safe_rect = brief.get("subtitle_safe_rect")
            subtitle_zone_css = (
                cls._zone_css_from_rect(subtitle_safe_rect) if subtitle_safe_rect else None
            ) or cls._zone_css(subtitle_position)
            body_html = (
                f'<div style="{title_zone_css}"><div style="{scrim_css}">{title_html}</div></div>'
                f'<div style="{subtitle_zone_css}">{subtitle_html}</div>'
            )

        return cls._PRODUCT_AD_TPL.substitute(
            w=analysis.width, h=analysis.height, bg_css=_bg_image_css(background_image_path),
            title_font=cls._resolve_font(brief.get("title_font"), "Montserrat"),
            title_size=title_size,
            title_transform=brief.get("title_transform", "uppercase"),
            title_style_css=cls.HERO_STYLE_CSS[title_style],
            subtitle_font=cls._resolve_font(brief.get("subtitle_font"), "Montserrat"),
            subtitle_size=brief.get("subtitle_size", default_subtitle_size),
            subtitle_style_css=cls.HERO_STYLE_CSS[subtitle_style],
            body_html=body_html,
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
            launch_args = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ]
            launch_kwargs: Dict[str, Any] = {
                "headless": True,
                "args": launch_args,
            }

            import shutil
            sys_chrome = (
                os.environ.get("PLAYWRIGHT_CHROME_PATH")
                or shutil.which("chromium-browser")
                or shutil.which("chromium")
                or shutil.which("google-chrome")
                or shutil.which("google-chrome-stable")
            )
            if sys_chrome:
                logger.info(f"[PosterRenderer] Found system browser at: {sys_chrome}")
                launch_kwargs["executable_path"] = sys_chrome

            browser = await p.chromium.launch(**launch_kwargs)
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=device_scale_factor,
            )
            page = await context.new_page()

            # Set content: wait for networkidle so external web fonts (Google Fonts) can download
            try:
                await page.set_content(html_content, wait_until="networkidle", timeout=6000)
            except Exception as e:
                logger.warning(f"[PosterRenderer] 'networkidle' wait failed or timed out ({e}), proceeding with rendered DOM...")
                try:
                    await page.set_content(html_content, wait_until="domcontentloaded", timeout=2000)
                except Exception:
                    pass

            # Settle fonts if available, otherwise continue smoothly without blocking
            try:
                await asyncio.wait_for(page.evaluate("document.fonts.ready"), timeout=4.0)
            except Exception as e:
                logger.debug(f"[PosterRenderer] Font readiness check skipped or timed out: {e}")


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
