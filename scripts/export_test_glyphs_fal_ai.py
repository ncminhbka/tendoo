import os
from PIL import Image, ImageDraw, ImageFont

def render_crisp_glyph(text, font_path, width=768, height=224, output_path="glyph.png"):
    # Create black canvas
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Binary search for optimal font size to fill ~80% of width and height
    target_w = width * 0.85
    target_h = height * 0.70
    
    font_size = 120
    for size in range(120, 20, -2):
        try:
            test_font = ImageFont.truetype(font_path, size=size)
            bbox = draw.textbbox((0, 0), text, font=test_font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= target_w and h <= target_h:
                font_size = size
                break
        except Exception:
            continue
            
    font = ImageFont.truetype(font_path, size=font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Precise centering
    x = (width - text_w) // 2 - bbox[0]
    y = (height - text_h) // 2 - bbox[1]
    
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"[OK] Generated: {output_path} | Font Size: {font_size}px | Box: {width}x{height}")


if __name__ == "__main__":
    out_dir = "test_fal_ai_glyphs"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Headline (Anton)
    render_crisp_glyph(
        text="ÂM THANH ĐỈNH CAO",
        font_path="fonts/Anton-Regular.ttf",
        width=768,
        height=224,
        output_path=os.path.join(out_dir, "01_headline_AM_THANH_DINH_CAO.png")
    )
    
    # 2. Subtitle (BeVietnamPro-Black)
    render_crisp_glyph(
        text="CHỐNG ỒN CHỦ ĐỘNG",
        font_path="fonts/BeVietnamPro-Black.ttf",
        width=768,
        height=224,
        output_path=os.path.join(out_dir, "02_subtitle_CHONG_ON_CHU_DONG.png")
    )
    
    # 3. CTA Badge (Pacifico)
    render_crisp_glyph(
        text="MUA 1 TẶNG 1",
        font_path="fonts/Pacifico-Regular.ttf",
        width=768,
        height=224,
        output_path=os.path.join(out_dir, "03_cta_MUA_1_TANG_1.png")
    )
