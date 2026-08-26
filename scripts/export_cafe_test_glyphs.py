import os
from PIL import Image, ImageDraw, ImageFont

def render_multiline_glyph(text, font_path, width=768, height=256, output_path="glyph.png"):
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    num_lines = len(lines)
    
    # Binary search for font size
    target_w = width * 0.88
    target_h = height * 0.80
    
    font_size = 100
    for size in range(100, 16, -2):
        try:
            test_font = ImageFont.truetype(font_path, size=size)
            # Calculate total multiline bounding box
            max_line_w = 0
            total_h = 0
            line_heights = []
            
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=test_font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                max_line_w = max(max_line_w, lw)
                line_heights.append(lh)
            
            line_spacing = int(size * 0.25)
            total_h = sum(line_heights) + (num_lines - 1) * line_spacing
            
            if max_line_w <= target_w and total_h <= target_h:
                font_size = size
                break
        except Exception:
            continue
            
    font = ImageFont.truetype(font_path, size=font_size)
    line_spacing = int(font_size * 0.25)
    
    # Recompute exact heights
    line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_widths = [bbox[2] - bbox[0] for bbox in line_bboxes]
    line_heights = [bbox[3] - bbox[1] for bbox in line_bboxes]
    total_h = sum(line_heights) + (num_lines - 1) * line_spacing
    
    # Start y
    curr_y = (height - total_h) // 2
    for i, line in enumerate(lines):
        bbox = line_bboxes[i]
        lw = line_widths[i]
        lh = line_heights[i]
        x = (width - lw) // 2 - bbox[0]
        y = curr_y - bbox[1]
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        curr_y += lh + line_spacing
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"[OK] Generated: {output_path} | Font Size: {font_size}px | Box: {width}x{height}")

if __name__ == "__main__":
    out_dir = "test_cafe_glyphs"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Headline (2 dòng: GRAND OPENING + MUA 1 TẶNG 1)
    render_multiline_glyph(
        text="GRAND OPENING\nMUA 1 TẶNG 1",
        font_path="fonts/Anton-Regular.ttf",
        width=768,
        height=288,
        output_path=os.path.join(out_dir, "01_headline_GRAND_OPENING.png")
    )
    
    # 2. Subtitle (2 dòng: Áp dụng từ 14/05 - 30/05 + Coffee rang mộc chuẩn vị)
    render_multiline_glyph(
        text="Áp dụng từ 14/05 - 30/05\nCoffee rang mộc chuẩn vị",
        font_path="fonts/BeVietnamPro-Black.ttf",
        width=640,
        height=192,
        output_path=os.path.join(out_dir, "02_subtitle_THOI_GIAN_CHAT_LUONG.png")
    )
    
    # 3. Dense Multi-line CTA (2 dòng: Ghé ngay hôm nay! + Deal cực hot - Số lượng có hạn!)
    render_multiline_glyph(
        text="Ghé ngay hôm nay!\nDeal cực hot - Số lượng có hạn!",
        font_path="fonts/Pacifico-Regular.ttf",
        width=640,
        height=224,
        output_path=os.path.join(out_dir, "03_cta_DENSE_PROMO.png")
    )
