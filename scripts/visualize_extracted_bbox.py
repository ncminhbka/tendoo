import json
from PIL import Image, ImageDraw

img_path = "test_cafe_gpt_image_2_output/pure_prompt_cafe_poster_named_fonts.png"
img = Image.open(img_path).convert("RGB")
w, h = img.size

draw = ImageDraw.Draw(img)

# Data extracted from gpt-4o-mini
blocks = [
  {
    "lines": ["GRAND OPENING"],
    "bbox_norm": [90, 150, 180, 850],
    "role": "headline"
  },
  {
    "lines": ["THỜI GIAN CHẤT LƯỢNG", "TẠI TỪNG GIỌT CÀ PHÊ"],
    "bbox_norm": [210, 200, 310, 800],
    "role": "subtitle"
  },
  {
    "lines": ["Ghé ngay hôm nay!", "Deal cực hot - Số lượng có hạn!"],
    "bbox_norm": [780, 100, 880, 900],
    "role": "cta_badge"
  }
]

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

for b, color in zip(blocks, colors):
    ymin, xmin, ymax, xmax = b["bbox_norm"]
    # Denormalize
    px_ymin = int(ymin * h / 1000)
    px_xmin = int(xmin * w / 1000)
    px_ymax = int(ymax * h / 1000)
    px_xmax = int(xmax * w / 1000)
    draw.rectangle([px_xmin, px_ymin, px_xmax, px_ymax], outline=color, width=4)

img.save("tests/test_bbox_overlay.png")
print("[OK] Saved overlay to tests/test_bbox_overlay.png")
