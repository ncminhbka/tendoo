"""
================================================================================
TENDOO AI - FONT UNICODE COVERAGE & GLYPH BITMAP FORENSIC INSPECTION
Checks Vietnamese Unicode coverage across all project fonts and inspects the
exact rendered glyph bitmap of 'CHỐNG ỒN CHỦ ĐỘNG'.
================================================================================
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent
FONTS_DIR = ROOT_DIR / "fonts"
OUTPUT_DIR = ROOT_DIR / "docs" / "font_verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Full Vietnamese Unicode test suite
VN_UPPERCASE = "A À Á Ả Ã Ạ Ă Ằ Ắ Ẳ Ẵ Ặ Â Ầ Ấ Ẩ Ẫ Ậ E È É Ẻ Ẽ Ẹ Ê Ề Ế Ể Ễ Ệ I Ì Í Ỉ Ĩ Ị O Ò Ó Ỏ Õ Ọ Ô Ồ Ố Ổ Ỗ Ộ ƠỜ Ớ Ở Ỡ Ợ U Ù Ú Ủ Ũ Ụ Ư Ừ Ứ Ử Ữ Ự Y Ỳ Ý Ỷ Ỹ Ỵ Đ"
VN_LOWERCASE = "a à á ả ã ạ ă ằ ắ ẳ ẵ ặ â ầ ấn ẩ ẫ ậ e è é ẻ ẽ ẹ ê ề ế ể ễ ệ i ì í ỉ ĩ ị o ò ó ỏ õ ọ ô ồ ố ổ ỗ ộ ơ ờ ớ ở ỡ ợ u ù ú ủ ũ ụ ư ừ ứ ử ữ ự y ỳ ý ỷ ỹ ỵ đ"
TARGET_PHRASE = "CHỐNG ỒN CHỦ ĐỘNG"


fonts_to_test = [
    ("BeVietnamPro-Black", FONTS_DIR / "BeVietnamPro-Black.ttf"),
    ("Anton-Regular", FONTS_DIR / "Anton-Regular.ttf"),
    ("PlayfairDisplay", FONTS_DIR / "PlayfairDisplay.ttf"),
    ("Pacifico-Regular", FONTS_DIR / "Pacifico-Regular.ttf"),
    ("SedgwickAveDisplay", FONTS_DIR / "SedgwickAveDisplay-Regular.ttf"),
    ("Oswald", FONTS_DIR / "Oswald.ttf"),
    ("DancingScript", FONTS_DIR / "DancingScript.ttf"),
    ("SVN-Gotham Ultra", FONTS_DIR / "SVN-Gotham Ultra.otf"),
]


def test_phrase_render():
    """Renders TARGET_PHRASE in 768x224 and 640x320 with each font."""
    print("=" * 80)
    print(f"🔍 INSPECTING GLYPH BITMAP FOR: '{TARGET_PHRASE}'")
    print("=" * 80)

    for name, fpath in fonts_to_test:
        if not fpath.exists():
            print(f"⚠️ Font not found: {fpath}")
            continue

        # Standard Box 768x224
        img = Image.new("RGB", (768, 224), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Binary search best size
        opt_font = None
        opt_size = 24
        for sz in range(20, 150):
            try:
                tf = ImageFont.truetype(str(fpath), size=sz)
                bbox = tf.getbbox(TARGET_PHRASE)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w <= 768 * 0.85 and h <= 224 * 0.85:
                    opt_font = tf
                    opt_size = sz
            except Exception:
                break

        if opt_font:
            bbox = opt_font.getbbox(TARGET_PHRASE)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = (768 - w) // 2 - bbox[0]
            y = (224 - h) // 2 - bbox[1]
            draw.text((x, y), TARGET_PHRASE, fill=(255, 255, 255), font=opt_font)

            out_file = OUTPUT_DIR / f"phrase_{name}_768x224.png"
            img.save(out_file)
            print(f"  -> ✅ [{name}] Rendered at size {opt_size}px (bbox: {w}x{h}px) -> {out_file.name}")


def test_unicode_coverage():
    """Renders complete Vietnamese alphabet and vowel chart for each font."""
    print("\n" + "=" * 80)
    print("🔍 CHECKING COMPLETE VIETNAMESE UNICODE COVERAGE ACROSS FONTS")
    print("=" * 80)

    rows = []
    for name, fpath in fonts_to_test:
        if not fpath.exists():
            continue

        try:
            font_title = ImageFont.truetype(str(FONTS_DIR / "BeVietnamPro-Black.ttf"), size=22)
            font_body = ImageFont.truetype(str(fpath), size=24)
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            continue

        panel = Image.new("RGB", (1200, 160), color=(15, 15, 20))
        draw = ImageDraw.Draw(panel)

        # Draw Font Name header
        draw.text((20, 10), f"Font: {name}", fill=(255, 215, 0), font=font_title)

        # Draw Uppercase Vowels
        draw.text((20, 50), VN_UPPERCASE, fill=(255, 255, 255), font=font_body)

        # Draw Lowercase Vowels
        draw.text((20, 100), VN_LOWERCASE, fill=(200, 230, 255), font=font_body)

        out_file = OUTPUT_DIR / f"unicode_coverage_{name}.png"
        panel.save(out_file)
        print(f"  -> ✅ [{name}] Unicode coverage test saved -> {out_file.name}")
        rows.append(panel)

    # Stitch all font panels vertically
    if rows:
        total_h = sum(r.height for r in rows)
        master = Image.new("RGB", (1200, total_h), color=(0, 0, 0))
        curr_y = 0
        for r in rows:
            master.paste(r, (0, curr_y))
            curr_y += r.height
        master_path = OUTPUT_DIR / "ALL_FONTS_UNICODE_VERIFICATION_SHEET.png"
        master.save(master_path)
        print(f"\n📊 Master Unicode Verification Sheet saved -> {master_path}")


if __name__ == "__main__":
    test_phrase_render()
    test_unicode_coverage()
