"""
scripts/test_typography_effects.py

Kiểm thử và so sánh trực quan các hiệu ứng Typography nâng cao của Tendoo AI:
1. 'neon' (Phát quang Neon Đỏ / Vibrant Neon Glow)
2. 'embossed' (In nổi 3D Mạ Vàng 24K / Chiseled 3D Gold)
3. 'chrome' (Bạch Kim Tráng Gương / Liquid Chrome Cyber)
4. 'shadow' (Bóng đổ Chiều Sâu Studio / Deep Ambient Occlusion)

Xuất từng poster độc lập và tạo ảnh ghép lưới so sánh trực quan (2x2 grid collage).
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tendoo.core.base import ColorPalette, PosterContent
from tendoo.core.colors import analyze_color_harmony
from tendoo.demo_server import pil_to_base64_data_uri
from tendoo.engine.blocks import AdaptiveBlock
from tendoo.engine.layout import OmniBlockLayout
from tendoo.poster_renderer import PosterRenderer
from test_before_after_feedback import synthesize_split_screen_gym_background


def run_effects_comparison():
    print("=" * 80)
    print("KIỂM THỬ VÀ SO SÁNH TRỰC QUAN CÁC HIỆU ỨNG TYPOGRAPHY NÂNG CAO")
    print("=" * 80)

    w, h = 816, 1024
    out_dir = Path("output_visual_eval_isolated/case_effects_showcase")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tạo ảnh nền Gym Before/After
    scene_bg = synthesize_split_screen_gym_background(w, h, seed=42)

    # 2. Khối nội dung cơ bản
    blocks_input = [
        AdaptiveBlock(
            text="KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY?",
            role="hero",
            zone="top_center",
        ),
        AdaptiveBlock(
            text="PRIVATE COACHING TRANSFORMATION",
            role="subtitle",
            zone="top_center",
        ),
        AdaptiveBlock(
            text="NGÀY 01 • BEFORE (82KG)",
            role="badge",
            zone="middle_left",
            style_variant="feature_card",
            icon="clock",
        ),
        AdaptiveBlock(
            text="NGÀY 90 • AFTER (72KG 6 MÚI)",
            role="badge",
            zone="middle_right",
            style_variant="pill_badge",
            icon="rocket",
        ),
        AdaptiveBlock(
            text="“97% khách hàng tăng cơ, giảm mỡ ngoạn mục và phục hồi thể lực sau 3 tháng tập cùng PT riêng”",
            role="body",
            zone="bottom_left",
            style_variant="quote",
        ),
        AdaptiveBlock(
            text="★★★★★ 5.0 / 5.0",
            role="badge",
            zone="bottom_left",
            style_variant="rating_badge",
            icon="star",
        ),
        AdaptiveBlock(
            text="GIẢM 20% GÓI PT THÁNG ĐẦU",
            role="badge",
            zone="bottom_right",
            style_variant="cta_button",
            icon="gift",
        ),
        AdaptiveBlock(
            text="Tendoo Fitness & Yoga  •  Hotline: 0988 888 999  •  HLV Cá Nhân 1:1",
            role="brand_bar",
            zone="bottom_bar",
            icon="phone",
        ),
    ]

    content = PosterContent(
        headline="KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY?",
        category="feedback",
        brand="Tendoo Fitness & Yoga",
        hotline="0988 888 999",
        free_text_blocks=[b.__dict__ for b in blocks_input],
    )

    layout = OmniBlockLayout()

    # Sinh mask đo đạc
    dummy_palette = ColorPalette(is_dark=True, luminance=0.5, hue=0, comp_hue=180, headline_color="#FFFFFF")
    neutral_bg_uri = pil_to_base64_data_uri(Image.new("RGB", (1, 1), (30, 30, 30)))
    measure_html = layout.render_html(
        content=content,
        palette=dummy_palette,
        bg_data_uri=neutral_bg_uri,
        width=w,
        height=h,
    )
    mask_np = layout.generate_mask_from_render(measure_html, width=w, height=h)
    if mask_np is None:
        mask_np = layout.generate_mask(width=w, height=h, blocks=blocks_input)

    # Velocity blending
    scene_arr = np.array(scene_bg, dtype=np.float32)
    corridor_arr = scene_arr * 0.70
    mask_3d = np.expand_dims(mask_np.clip(0.0, 1.0), axis=-1)
    blended_arr = ((1.0 - mask_3d) * scene_arr + mask_3d * corridor_arr).clip(0, 255).astype(np.uint8)
    blended_bg = Image.fromarray(blended_arr, mode="RGB")
    bg_data_uri = pil_to_base64_data_uri(blended_bg)
    palette = analyze_color_harmony(blended_arr, layout.get_safe_zone(), color_mode="auto")

    # Danh sách 4 hiệu ứng thử nghiệm
    effects_to_test = [
        ("neon", "01_poster_neon.png", "1. PHÁT QUANG NEON (Neon Glow)"),
        ("embossed", "02_poster_3d_gold.png", "2. IN NỔI 3D MẠ VÀNG 24K (Chiseled Gold)"),
        ("chrome", "03_poster_chrome.png", "3. CHROME BẠCH KIM (Liquid Chrome)"),
        ("shadow", "04_poster_studio_shadow.png", "4. BÓNG ĐỔ STUDIO (Deep Shadow)"),
    ]

    rendered_images = []

    for effect_key, filename, label in effects_to_test:
        print(f"\n---> Đang kết xuất hiệu ứng: {label}...")
        final_html = layout.render_html(
            content=content,
            palette=palette,
            bg_data_uri=bg_data_uri,
            width=w,
            height=h,
            headline_effect=effect_key,
        )
        out_path = out_dir / filename
        PosterRenderer.render(final_html, out_path, width=w, height=h)
        print(f"     Đã lưu: {out_path}")
        rendered_images.append((Image.open(out_path), label))

    # 3. Tạo ảnh ghép lưới 2x2 để so sánh trực quan
    print("\n---> Đang tạo ảnh ghép so sánh 2x2 Grid Collage...")
    thumb_w, thumb_h = 500, int(500 * (h / w))
    header_h = 44
    grid_w = thumb_w * 2 + 60
    grid_h = (thumb_h + header_h) * 2 + 60

    collage = Image.new("RGB", (grid_w, grid_h), (18, 20, 26))
    draw = ImageDraw.Draw(collage)

    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        title_font = ImageFont.load_default()

    positions = [
        (20, 20),
        (thumb_w + 40, 20),
        (20, thumb_h + header_h + 30),
        (thumb_w + 40, thumb_h + header_h + 30),
    ]

    for idx, (img, label) in enumerate(rendered_images):
        pos_x, pos_y = positions[idx]
        # Vẽ header nhãn hiệu ứng
        draw.rectangle([pos_x, pos_y, pos_x + thumb_w, pos_y + 32], fill=(30, 35, 48))
        draw.text((pos_x + 12, pos_y + 6), label, fill=(255, 255, 255), font=title_font)
        # Resize và dán poster
        thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        collage.paste(thumb, (pos_x, pos_y + 36))

    collage_file = out_dir / "00_comparison_2x2_collage.png"
    collage.save(collage_file)
    print(f"\n[THÀNH CÔNG] Đã tạo ảnh ghép so sánh 2x2 tại: {collage_file}")


if __name__ == "__main__":
    run_effects_comparison()
