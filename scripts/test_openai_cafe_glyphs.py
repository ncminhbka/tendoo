import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import urllib.request
import base64

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY not found in .env")
    sys.exit(1)

client = OpenAI(api_key=api_key)

glyph_dir = Path("test_cafe_glyphs")
out_dir = Path("test_cafe_gpt_image_2_output")
out_dir.mkdir(exist_ok=True)

g1 = glyph_dir / "01_headline_GRAND_OPENING.png"
g2 = glyph_dir / "02_subtitle_THOI_GIAN_CHAT_LUONG.png"
g3 = glyph_dir / "03_cta_DENSE_PROMO.png"

print(f"Checking glyph files:")
print(f"  G1: {g1} (exists: {g1.exists()})")
print(f"  G2: {g2} (exists: {g2.exists()})")
print(f"  G3: {g3} (exists: {g3.exists()})")

# Test 1: Multiple images in one call (already done)
print("\n--- TEST 1: Already generated cafe_poster_3glyphs_gpt_image_2_low.png (Skipping) ---")


# Helper to save response
def save_image_response(resp, target_file):
    if resp.data[0].url:
        urllib.request.urlretrieve(resp.data[0].url, target_file)
        print(f"[OK] Saved from URL to: {target_file}")
    elif resp.data[0].b64_json:
        with open(target_file, "wb") as f:
            f.write(base64.b64decode(resp.data[0].b64_json))
        print(f"[OK] Saved from b64 to: {target_file}")
    else:
        print(f"[WARN] No image data returned for {target_file}")

# Test 2: Single glyph test (e.g. G1 Anton)
print("\n--- TEST 2: Sending individual glyph G1 (Headline - Anton) ---")
try:
    with open(g1, "rb") as f1:
        response = client.images.edit(
            model="gpt-image-2",
            image=f1,
            prompt="A cafe poster wooden signboard. Render the exact text and preserve the exact bold condensed font style from the reference image: 'GRAND OPENING / MUA 1 TẶNG 1'. Maintain exact typography and diacritics.",
            quality="low",
            size="1024x1024",
        )
        out_path = out_dir / "glyph_01_anton_gpt_image_2_low.png"
        save_image_response(response, out_path)
except Exception as e:
    print(f"[ERROR in G1]: {e}")

# Test 3: Single glyph test (G3 Pacifico Cursive)
print("\n--- TEST 3: Sending individual glyph G3 (CTA - Pacifico Cursive) ---")
try:
    with open(g3, "rb") as f3:
        response = client.images.edit(
            model="gpt-image-2",
            image=f3,
            prompt="A decorative cafe promo sticker badge on a rustic wooden table. Render the exact text and preserve the exact cursive script font style from the reference image: 'Ghé ngay hôm nay! / Deal cực hot - Số lượng có hạn!'. Maintain exact typography and cursive handwriting shape.",
            quality="low",
            size="1024x1024",
        )
        out_path = out_dir / "glyph_03_pacifico_gpt_image_2_low.png"
        save_image_response(response, out_path)
except Exception as e:
    print(f"[ERROR in G3]: {e}")

# Test 4: Single glyph test (G2 BeVietnamPro)
print("\n--- TEST 4: Sending individual glyph G2 (Subtitle - BeVietnamPro-Black) ---")
try:
    with open(g2, "rb") as f2:
        response = client.images.edit(
            model="gpt-image-2",
            image=f2,
            prompt="A cafe banner subtitle panel. Render the exact text and preserve the exact geometric black sans-serif font style from the reference image: 'Áp dụng từ 14/05 - 30/05 / Coffee rang mộc chuẩn vị'. Maintain exact typography and diacritics.",
            quality="low",
            size="1024x1024",
        )
        out_path = out_dir / "glyph_02_bevietnam_gpt_image_2_low.png"
        save_image_response(response, out_path)
except Exception as e:
    print(f"[ERROR in G2]: {e}")

print("\n[DONE] Finished tests.")
