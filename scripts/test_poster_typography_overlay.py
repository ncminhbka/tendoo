#!/usr/bin/env python3
"""
scripts/test_poster_typography_overlay.py

==================================================================================================
TENDOO AI - POSTER HTML TYPOGRAPHY OVERLAY TEST BENCH (PLAYWRIGHT & CHROMIUM)
==================================================================================================

OBJECTIVE:
  Test and validate the HTML-based scalable typography overlay pipeline locally using Playwright & Chromium.
  Ensures that secondary elements (Badges, Ratings, Customer Quotes, Spec Chips, Footers)
  are rendered with sub-pixel crispness and harmonious contrast over generated DiT posters.

MODES:
  1. --mode template    : [100% Offline] Uses Algorithmic Contrast-Aware Layout Engine.
                          No external API or LLM required. Ideal for internal offline server!
  2. --mode prompt_only : Analyzes image luminance and prints the exact prompt for GPT-4o / Qwen2.5-VL.
  3. --mode render_html : Renders an existing HTML file (e.g. received from GPT-4o) directly onto the image.
  4. --mode api_gpt4o   : Calls OpenAI GPT-4o API (requires OPENAI_API_KEY) with image + analysis.
  5. --mode api_gemini  : Calls Google Gemini API (requires GEMINI_API_KEY) with image + analysis.

EXAMPLE COMMANDS:
  # 1. Offline algorithmic template test (Instant, zero API):
  python scripts/test_poster_typography_overlay.py --image path/to/image.png --mode template

  # 2. Generate prompt to send to GPT-4o:
  python scripts/test_poster_typography_overlay.py --image path/to/image.png --mode prompt_only

  # 3. Render HTML returned by GPT-4o:
  python scripts/test_poster_typography_overlay.py --image path/to/image.png --html_input gpt4o_output.html --mode render_html
==================================================================================================
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tendoo.typography_engine import (
    BackgroundAnalysis,
    PosterBackgroundAnalyzer,
    PosterRenderer,
    PosterTemplateEngine,
    TypographyPromptBuilder,
)


def extract_html_codeblock(response_text: str) -> str:
    """Extracts raw HTML code from an LLM response containing markdown codeblocks."""
    # Match ```html ... ```
    match = re.search(r"```(?:html)?\s*(<!DOCTYPE.+?|<html.+?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Match raw <!DOCTYPE ... </html> or <html ... </html>
    match_raw = re.search(r"(<!DOCTYPE.+?</html>|<html.+?</html>)", response_text, re.DOTALL | re.IGNORECASE)
    if match_raw:
        return match_raw.group(1).strip()

    return response_text.strip()


def call_gpt4o_api(image_path: str, prompt: str, api_key: str | None = None) -> str:
    """Calls OpenAI GPT-4o Vision API with the poster image and design prompt."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] 'openai' package not installed. Run: pip install openai")
        sys.exit(1)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("[ERROR] OPENAI_API_KEY not found in environment or --api_key argument.")
        sys.exit(1)

    client = OpenAI(api_key=key)

    with open(image_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower().replace(".", "")
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

    print("  [API] Sending image & prompt to GPT-4o...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": TypographyPromptBuilder.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_img}", "detail": "high"},
                    },
                ],
            },
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    content = response.choices[0].message.content or ""
    return extract_html_codeblock(content)


def call_gemini_api(image_path: str, prompt: str, api_key: str | None = None) -> str:
    """Calls Google Gemini API with the poster image and design prompt."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("[ERROR] 'google-genai' package not installed. Run: pip install google-genai")
        sys.exit(1)

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("[ERROR] GEMINI_API_KEY not found in environment or --api_key argument.")
        sys.exit(1)

    client = genai.Client(api_key=key)
    from PIL import Image
    pil_img = Image.open(image_path)

    print("  [API] Sending image & prompt to Gemini 2.5 Flash...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[pil_img, TypographyPromptBuilder.SYSTEM_PROMPT + "\n\n" + prompt],
    )

    text = response.text or ""
    return extract_html_codeblock(text)


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Poster HTML Typography Overlay Test Bench")
    parser.add_argument("--image", type=str, required=True, help="Path to input poster / background image")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["template", "prompt_only", "render_html", "api_gpt4o", "api_gemini"],
        default="template",
        help="Execution mode (default: template)",
    )
    parser.add_argument("--hero_title", type=str, default=None, help="Existing 3D Hero Title text in image")
    parser.add_argument(
        "--category", type=str, default="generic",
        choices=["generic", "grand_opening", "feedback", "recruitment", "menu"],
        help="Which template from the library to use (--mode template only). Pick this via the "
             "Stage 1 blueprint/hero-selector category, not per-request guessing.",
    )
    parser.add_argument("--brief_json", type=str, default=None, help="Path to custom brief JSON file")
    parser.add_argument("--html_input", type=str, default=None, help="Path to existing HTML file (for render_html mode)")
    parser.add_argument("--output_dir", type=str, default="output_typography_tests", help="Output directory")
    parser.add_argument("--device_scale", type=int, default=1, help="Playwright device scale factor (1 or 2)")
    parser.add_argument("--api_key", type=str, default=None, help="Optional API key for GPT-4o or Gemini")

    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("🚀 TENDOO AI - POSTER HTML TYPOGRAPHY OVERLAY ENGINE")
    print("=" * 100)
    print(f"  Input Image       : {img_path.resolve()}")
    print(f"  Execution Mode    : {args.mode.upper()}")
    print(f"  Template Category : {args.category} (--mode template only)")
    print(f"  Output Directory  : {out_dir.resolve()}")

    # 1. Step 1: Analyze background luminance and contrast
    print("\n[Step 1/3] Running Quantitative Background & Contrast Analysis...")
    analysis: BackgroundAnalysis = PosterBackgroundAnalyzer.analyze(img_path)
    print(f"  Canvas Dimensions : {analysis.width}x{analysis.height}px (Ratio: {analysis.aspect_ratio})")
    print(f"  Header Zone       : Luminance={analysis.header_zone.mean_luminance:.1f} ({'DARK' if analysis.header_zone.is_dark else 'BRIGHT'}) | Dom={analysis.header_zone.dominant_hex}")
    print(f"  Footer Zone       : Luminance={analysis.footer_zone.mean_luminance:.1f} ({'DARK' if analysis.footer_zone.is_dark else 'BRIGHT'}) | Dom={analysis.footer_zone.dominant_hex}")
    print(f"  Overall           : Luminance={analysis.overall_luminance:.1f} ({'DARK' if analysis.overall_is_dark else 'BRIGHT'})")

    # Load custom brief if provided, else use default commercial advertising brief
    brief: Dict[str, Any] = {
        "brand": "TENDOO SOUND",
        "eyebrow": "ÂM THANH KHÔNG DÂY HI-RES",
        "badge": "GIẢM 30% HÔM NAY",
        "rating_value": "4.9",
        "rating_count": "2.8k+ đánh giá",
        "specs": [
            "Chống Ồn Chủ Động 45dB",
            "Thời Lượng Pin 40 Giờ",
            "Màng Loa Beryllium 40mm",
            "Bluetooth 5.4 Ultra-Low Latency",
        ],
        "sub_slogan": "CHINH PHỤC CHẤT ÂM PHÒNG THU ĐỈNH CAO",
        "cta_text": "SỞ HỮU NGAY",
        "hotline": "1800 8198",
        "website": "www.tendoo.ai",
    }
    if args.brief_json and Path(args.brief_json).exists():
        with open(args.brief_json, "r", encoding="utf-8") as f:
            brief.update(json.load(f))

    # 2. Step 2: Handle selected mode
    html_content: str = ""
    stem = img_path.stem

    if args.mode == "prompt_only":
        print("\n[Step 2/3] Generating Prompt for GPT-4o / Qwen2.5-VL...")
        prompt = TypographyPromptBuilder.build_user_prompt(analysis, brief, hero_title_text=args.hero_title)
        prompt_file = out_dir / f"{stem}_vlm_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write("=== SYSTEM PROMPT ===\n")
            f.write(TypographyPromptBuilder.SYSTEM_PROMPT)
            f.write("\n\n=== USER PROMPT ===\n")
            f.write(prompt)
        print(f"\n[✓] Prompt generated and saved to: {prompt_file.resolve()}\n")
        print("--- COPY PROMPT BELOW TO GPT-4O ---")
        print(prompt)
        return

    elif args.mode == "template":
        print("\n[Step 2/3] Generating Algorithmic Contrast-Harmonious HTML/CSS (Offline Mode)...")
        html_content = PosterTemplateEngine.generate_html(
            analysis=analysis,
            brief=brief,
            background_image_path=str(img_path),
            category=args.category,
        )

    elif args.mode == "render_html":
        if not args.html_input or not Path(args.html_input).exists():
            print("[ERROR] Please specify a valid HTML file with --html_input")
            sys.exit(1)
        print(f"\n[Step 2/3] Reading input HTML from: {args.html_input}...")
        with open(args.html_input, "r", encoding="utf-8") as f:
            html_content = f.read()

        # If background image is not embedded as base64, inject it
        if "data:image/" not in html_content and str(img_path) not in html_content:
            with open(img_path, "rb") as f:
                b64_bg = base64.b64encode(f.read()).decode("utf-8")
                ext = img_path.suffix.lower().replace(".", "")
                mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
                bg_inject = f".poster-container {{ background-image: url('data:{mime};base64,{b64_bg}'); background-size: cover; }}"
                html_content = html_content.replace("</style>", f"{bg_inject}\n</style>")

    elif args.mode == "api_gpt4o":
        print("\n[Step 2/3] Invoking GPT-4o API...")
        prompt = TypographyPromptBuilder.build_user_prompt(analysis, brief, hero_title_text=args.hero_title)
        html_content = call_gpt4o_api(str(img_path), prompt, api_key=args.api_key)

    elif args.mode == "api_gemini":
        print("\n[Step 2/3] Invoking Gemini API...")
        prompt = TypographyPromptBuilder.build_user_prompt(analysis, brief, hero_title_text=args.hero_title)
        html_content = call_gemini_api(str(img_path), prompt, api_key=args.api_key)

    # Save generated HTML
    html_file = out_dir / f"{stem}_{args.mode}.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [✓] HTML Saved    : {html_file.name}")

    # 3. Step 3: Render HTML via Playwright Chromium
    print("\n[Step 3/3] Rendering HTML with Playwright Headless Chromium...")
    out_png = out_dir / f"{stem}_{args.mode}_final.png"
    PosterRenderer.render(
        html_content=html_content,
        output_image_path=out_png,
        width=analysis.width,
        height=analysis.height,
        device_scale_factor=args.device_scale,
    )

    print("\n" + "=" * 90)
    print("🎉 SUCCESS! POSTER COMPOSITE COMPLETED")
    print("=" * 90)
    print(f"  Source Image      : {img_path.name}")
    print(f"  Generated HTML    : {html_file.resolve()}")
    print(f"  Final Composite   : {out_png.resolve()}")
    print(f"  Resolution        : {analysis.width}x{analysis.height}px")
    print("=" * 90)


if __name__ == "__main__":
    main()
