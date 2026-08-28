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

out_dir = Path("test_cafe_gpt_image_2_output")
out_dir.mkdir(exist_ok=True)

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

# TEST 1: Cafe poster with 3 explicit font names in prompt (No glyph images)
print("\n--- TEST 1: Pure Prompt with Explicit Font Names (Anton, Be Vietnam Pro, Pacifico) ---")
prompt_poster = (
    "A cozy vintage cafe promotional poster with warm wooden bar background, ambient lighting. "
    "Include 3 distinct text sections with strict typography: "
    "1. Top Headline: text 'GRAND OPENING / MUA 1 TẶNG 1' rendered strictly in 'Anton' font (condensed heavy display sans-serif). "
    "2. Middle Subtitle: text 'Áp dụng từ 14/05 - 30/05 / Coffee rang mộc chuẩn vị' rendered strictly in 'Be Vietnam Pro' font (geometric modern sans-serif). "
    "3. Bottom Badge: text 'Ghé ngay hôm nay! / Deal cực hot - Số lượng có hạn!' rendered strictly in 'Pacifico' font (retro casual brush script). "
    "Render correct Vietnamese diacritics and strictly adhere to the requested font typefaces."
)

try:
    response = client.images.generate(
        model="gpt-image-2",
        prompt=prompt_poster,
        quality="low",
        size="1024x1024",
    )
    out_path = out_dir / "pure_prompt_cafe_poster_named_fonts.png"
    save_image_response(response, out_path)
except Exception as e:
    print(f"[ERROR in Test 1]: {e}")

# TEST 2: Direct Typography Showcase with 3 distinct font styles (Anton, Playfair Display, Pacifico)
print("\n--- TEST 2: Typography Specimen Board (Anton, Playfair Display, Pacifico) ---")
prompt_showcase = (
    "A clean graphic design typography specimen poster on dark textured paper, showing 3 distinct horizontal rows of typography: "
    "Row 1 (Top): Large text 'HƯƠNG VỊ ĐẬM ĐÀ' typed strictly in 'Anton' font (tall condensed sans-serif, uppercase, bold). "
    "Row 2 (Middle): Elegant text 'Nghệ Thuật Cà Phê Cổ Điển' typed strictly in 'Playfair Display' font (high-contrast editorial serif, classic italic serifs). "
    "Row 3 (Bottom): Casual text 'Thưởng thức từng khoảnh khắc' typed strictly in 'Pacifico' font (flowing cursive brush script, rounded loops). "
    "Each row must strictly match the visual geometry and letterforms of the named Google Font. Perfect Vietnamese accents."
)

try:
    response = client.images.generate(
        model="gpt-image-2",
        prompt=prompt_showcase,
        quality="low",
        size="1024x1024",
    )
    out_path = out_dir / "pure_prompt_typography_showcase_3fonts.png"
    save_image_response(response, out_path)
except Exception as e:
    print(f"[ERROR in Test 2]: {e}")

print("\n[DONE] Finished pure prompt font adherence tests.")
