#!/usr/bin/env python3
"""
Test extraction fidelity of gpt-4o-mini on artistic cafe poster.
Checks:
1. Exact line breaks (whether multiline text has \n or is returned as a list of lines).
2. Bounding box accuracy on 1024x1024 poster.
"""

import base64
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY not found in .env")
    exit(1)

client = OpenAI(api_key=api_key)

image_path = Path("test_cafe_gpt_image_2_output/pure_prompt_cafe_poster_named_fonts.png")
if not image_path.exists():
    print(f"[ERROR] Image not found: {image_path}")
    exit(1)

with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

system_prompt = """
You are an expert OCR & Graphic Layout Analyst.
Analyze the provided image and extract all visible typography blocks.
For each text block, you MUST determine:
1. "lines": A JSON list of strings, representing the exact line-by-line breakdown as physically rendered on the image. If words are stacked vertically, each line MUST be a separate string in the list.
2. "full_text_with_newlines": The exact text with explicit '\\n' where line breaks occur.
3. "bbox_norm": [ymin, xmin, ymax, xmax] normalized from 0 to 1000.
4. "role": one of ["headline", "subtitle", "cta_badge", "body", "other"].
5. "estimated_font_style": e.g. "condensed sans-serif", "serif", "brush script", "rounded".

Output ONLY valid JSON matching this schema:
{
  "blocks": [
    {
      "lines": ["Line 1", "Line 2"],
      "full_text_with_newlines": "Line 1\\nLine 2",
      "bbox_norm": [ymin, xmin, ymax, xmax],
      "role": "...",
      "estimated_font_style": "..."
    }
  ]
}
"""

print(f"[*] Sending image {image_path.name} to gpt-4o-mini for layout & line break inspection...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Inspect this poster and extract all text blocks, their exact line breaks, and bounding boxes.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ],
    temperature=0.0,
)

content = response.choices[0].message.content
print("\n" + "=" * 80)
print(" [*] RAW GPT-4O-MINI VISION EXTRACTION RESULT")
print("=" * 80)
try:
    parsed = json.loads(content)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
except Exception:
    print(content)
print("=" * 80)
