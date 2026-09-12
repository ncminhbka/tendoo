"""
scripts/test_before_after_feedback.py

Kiểm thử Bố cục Feedback Before/After Chia Đôi 2 Hình Trái - Phải (Gym Fitness):
- Bối cảnh chia đôi: Nửa trái (Before - tông lạnh trầm), Nửa phải (After - tông đỏ đen rực rỡ, ánh sáng cơ bắp).
- Vạch chia dọc tinh tế ở giữa màn hình.
- Tiêu đề chính ở đỉnh, nhãn Before ở cột trái, nhãn After ở cột phải.
- Review 5 sao vàng, trích dẫn feedback khách hàng và CTA ưu đãi giảm 20%.
- Kiểm tra toàn bộ cơ chế Velocity Blending Mask và kết xuất ảnh PNG hoàn chỉnh.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tendoo.core.base import ColorPalette, PosterContent
from tendoo.core.colors import analyze_color_harmony
from tendoo.demo_server import pil_to_base64_data_uri
from tendoo.engine.blocks import AdaptiveBlock
from tendoo.engine.geometry import get_product_sanctuary_rect
from tendoo.engine.layout import OmniBlockLayout
from tendoo.poster_renderer import PosterRenderer


def synthesize_split_screen_gym_background(width: int, height: int, seed: int = 42) -> Image.Image:
    """Tạo ảnh nền nghệ thuật mô phỏng chia đôi Before / After cho dịch vụ Gym & PT cao cấp.
    
    CẤU TRÚC 2 NỬA:
    - Nửa Trái (Before): Tông lạnh than chì - xanh xám trầm, ánh sáng khuếch tán mờ ảo,
      mô phỏng vóc dáng người trước khi tập.
    - Nửa Phải (After): Tông đỏ đen tương phản cao (charcoal & crimson rim-light), ánh sáng
      tập trung sắc bén tạo khối cơ bắp cuồn cuộn, bóng tạ ấm áp.
    - Trục giữa (Split divider): Đường phân cách dọc 2px ánh sáng neon chuyển tiếp mềm mại.
    """
    img = Image.new("RGB", (width, height), (15, 17, 23))
    draw = ImageDraw.Draw(img)

    half_w = width // 2

    # 1. Nửa Trái (Before - Mờ ảo, tông xanh xám lạnh #18202c)
    for x in range(half_w):
        t = x / half_w
        base_r = int(14 + 10 * t)
        base_g = int(18 + 14 * t)
        base_b = int(26 + 22 * t)
        draw.line([(x, 0), (x, height)], fill=(base_r, base_g, base_b))

    # Đèn mờ dịu hậu cảnh nửa trái (Before silhouette)
    left_cx, left_cy = int(half_w * 0.5), int(height * 0.52)
    for r in range(260, 0, -15):
        ratio = r / 260.0
        c_r = int(18 + 20 * (1 - ratio))
        c_g = int(26 + 32 * (1 - ratio))
        c_b = int(38 + 48 * (1 - ratio))
        draw.ellipse([left_cx - r, left_cy - int(r * 1.3), left_cx + r, left_cy + int(r * 1.3)], fill=(c_r, c_g, c_b))

    # 2. Nửa Phải (After - Đỏ đen rực rỡ, cơ bắp, rim light điện ảnh)
    for x in range(half_w, width):
        t = (x - half_w) / half_w
        base_r = int(28 + 40 * (1.0 - (t - 0.5)**2))
        base_g = int(12 + 10 * (1.0 - t))
        base_b = int(14 + 8 * (1.0 - t))
        draw.line([(x, 0), (x, height)], fill=(base_r, base_g, base_b))

    # Ánh sáng đỏ neon kịch tính tôn vinh vóc dáng săn chắc (After highlight)
    right_cx, right_cy = int(half_w + half_w * 0.5), int(height * 0.50)
    for r in range(300, 0, -15):
        ratio = r / 300.0
        c_r = int(180 * (1 - ratio) + 25 * ratio)
        c_g = int(24 * (1 - ratio) + 12 * ratio)
        c_b = int(28 * (1 - ratio) + 14 * ratio)
        draw.ellipse([right_cx - r, right_cy - int(r * 1.3), right_cx + r, right_cy + int(r * 1.3)], fill=(c_r, c_g, c_b))

    # Thêm chi tiết bóng cơ bắp / tạ đòn ở góc phải
    for r in range(160, 0, -10):
        ratio = r / 160.0
        glow_r = int(240 * (1 - ratio) + 40 * ratio)
        glow_g = int(60 * (1 - ratio) + 15 * ratio)
        glow_b = int(40 * (1 - ratio) + 15 * ratio)
        draw.ellipse([right_cx - r, right_cy - r, right_cx + r, right_cy + r], fill=(glow_r, glow_g, glow_b))

    # 3. Vạch chia đôi màn hình ở chính giữa (Vertical Split Divider)
    draw.line([(half_w - 1, 0), (half_w - 1, height)], fill=(60, 60, 70), width=1)
    draw.line([(half_w, 0), (half_w, height)], fill=(255, 255, 255), width=2)
    draw.line([(half_w + 1, 0), (half_w + 1, height)], fill=(220, 38, 38), width=1)

    # Thêm chút hạt noise nghệ thuật
    rng = np.random.RandomState(seed)
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, 1.8, (height, width, 3))
    arr = (arr + noise).clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.8))


def run_before_after_test():
    """Chạy toàn bộ pipeline kiểm thử layout chia đôi Before / After."""
    print("=" * 80)
    print("KIỂM THỬ BỐ CỤC FEEDBACK BEFORE / AFTER CHIA ĐÔI (GYM TRANSFORMATION)")
    print("=" * 80)

    # Khung hình 4:5 chuẩn Instagram / Facebook Fitness Ad
    w, h = 816, 1024
    out_dir = Path("output_visual_eval_isolated/case_before_after_gym_feedback")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Định nghĩa các khối chữ theo yêu cầu của Prompt 21
    blocks_input = [
        # Tiêu đề chính ở đỉnh (trang trọng, cân đối trên cả 2 nửa)
        AdaptiveBlock(
            text="KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY?",
            role="hero",
            zone="top_center",
        ),
        # Subtitle định vị dịch vụ
        AdaptiveBlock(
            text="PRIVATE COACHING TRANSFORMATION",
            role="subtitle",
            zone="top_center",
        ),
        # Nhãn Nửa Trái: BEFORE
        AdaptiveBlock(
            text="NGÀY 01 • BEFORE (82KG)",
            role="badge",
            zone="middle_left",
            style_variant="feature_card",
            icon="clock",
        ),
        # Nhãn Nửa Phải: AFTER (Kèm icon tên lửa / bứt phá)
        AdaptiveBlock(
            text="NGÀY 90 • AFTER (72KG 6 MÚI)",
            role="badge",
            zone="middle_right",
            style_variant="pill_badge",
            icon="rocket",
        ),
        # Lời nhận xét khách hàng ở góc dưới bên trái
        AdaptiveBlock(
            text="“97% khách hàng tăng cơ, giảm mỡ ngoạn mục và phục hồi thể lực sau 3 tháng tập cùng PT riêng”",
            role="body",
            zone="bottom_left",
            style_variant="quote",
        ),
        # Rating 5 sao vàng
        AdaptiveBlock(
            text="★★★★★ 5.0 / 5.0",
            role="badge",
            zone="bottom_left",
            style_variant="rating_badge",
            icon="star",
        ),
        # Nút CTA ưu đãi giảm 20% ở góc dưới bên phải
        AdaptiveBlock(
            text="GIẢM 20% GÓI PT THÁNG ĐẦU",
            role="badge",
            zone="bottom_right",
            style_variant="cta_button",
            icon="gift",
        ),
        # Thanh thương hiệu ở đáy
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

    # 2. Đo đạc Bounding Box qua Chromium & Sinh Corridor Mask
    dummy_palette = ColorPalette(is_dark=True, luminance=0.5, hue=0, comp_hue=180, headline_color="#FFFFFF")
    neutral_bg_uri = pil_to_base64_data_uri(Image.new("RGB", (1, 1), (30, 30, 30)))
    measure_html = layout.render_html(
        content=content,
        palette=dummy_palette,
        bg_data_uri=neutral_bg_uri,
        width=w,
        height=h,
    )

    print("[1/4] Đang đo đạc hình học trực tuyến và sinh Corridor Mask...")
    mask_np = layout.generate_mask_from_render(measure_html, width=w, height=h)
    if mask_np is None:
        mask_np = layout.generate_mask(width=w, height=h, blocks=blocks_input)

    mask_vis = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    mask_vis.save(out_dir / "01_corridor_mask.png")

    # 3. Tạo ảnh nền mock chia đôi Before/After hòa trộn Velocity Blending với mask
    print("[2/4] Đang tạo ảnh nền Before / After chia đôi và hòa trộn Velocity Blending...")
    scene_bg = synthesize_split_screen_gym_background(w, h, seed=42)

    # Áp dụng 2-branch velocity blending: vùng trong mask làm dịu 30% để chữ HTML nổi bật
    scene_arr = np.array(scene_bg, dtype=np.float32)
    corridor_arr = scene_arr * 0.70  # làm dịu nhẹ vùng chữ
    mask_3d = np.expand_dims(mask_np.clip(0.0, 1.0), axis=-1)
    blended_arr = ((1.0 - mask_3d) * scene_arr + mask_3d * corridor_arr).clip(0, 255).astype(np.uint8)
    blended_bg = Image.fromarray(blended_arr, mode="RGB")
    blended_bg.save(out_dir / "02_mock_background.png")

    # 4. Phân tích màu sắc & Render HTML cuối cùng bằng Chromium
    print("[3/4] Đang phân tích màu sắc và kết xuất poster hoàn chỉnh...")
    palette = analyze_color_harmony(blended_arr, layout.get_safe_zone(), color_mode="auto")
    bg_data_uri = pil_to_base64_data_uri(blended_bg)

    final_html = layout.render_html(
        content=content,
        palette=palette,
        bg_data_uri=bg_data_uri,
        width=w,
        height=h,
        headline_effect="neon_glow",
    )
    (out_dir / "poster_markup.html").write_text(final_html, encoding="utf-8")

    poster_file = out_dir / "03_poster.png"
    PosterRenderer.render(final_html, poster_file, width=w, height=h)

    # 5. So khớp viền dạ quang (Mask Overlay)
    print("[4/4] Đang tạo ảnh so khớp viền dạ quang (Mask Overlay)...")
    from evaluate_isolated_pipeline import create_mask_overlay
    poster_pil = Image.open(poster_file)
    overlay_pil = create_mask_overlay(poster_pil, mask_np)
    overlay_pil.save(out_dir / "04_mask_overlay.png")

    # 6. Ghi báo cáo chỉ số
    s_x1, s_y1, s_x2, s_y2 = get_product_sanctuary_rect(w, h)
    sanctuary_sub = mask_np[s_y1:s_y2, s_x1:s_x2]
    leakage_max = float(sanctuary_sub.max()) if sanctuary_sub.size > 0 else 0.0
    mask_coverage = float((mask_np > 0.30).mean() * 100.0)

    report_txt = f"""================================================================================
BÁO CÁO KẾT QUẢ TEST: FEEDBACK GYM BEFORE / AFTER (SPLIT-SCREEN)
================================================================================
- Tỉ lệ khung hình: 4:5 ({w}x{h} px)
- Bối cảnh: Phòng gym chia đôi 2 nửa (Before lạnh than chì vs After rực đỏ neon)
- Phân bổ khối:
  1. top_center: Hero Title + Subtitle
  2. middle_left: Nhãn Before (Ngày 01)
  3. middle_right: Nhãn After (Ngày 90) kèm icon tên lửa (rocket)
  4. bottom_left: Quote feedback khách hàng + Rating 5 sao vàng
  5. bottom_right: Nút CTA Giảm 20% kèm icon hộp quà (gift)
  6. bottom_bar: Thông tin thương hiệu & Hotline
- Tỉ lệ phủ mask: {mask_coverage:.1f}%
- Trạng thái vùng tâm: {"AN TOÀN" if leakage_max <= 0.35 else "CẢNH BÁO"} (Leak max: {leakage_max:.2f})
- Thư mục xuất: {out_dir}
================================================================================
"""
    (out_dir / "05_report.txt").write_text(report_txt, encoding="utf-8")

    # In thông báo hoàn tất bằng utf-8 an toàn
    import sys
    sys.stdout.buffer.write(f"\n[THÀNH CÔNG] Đã xuất toàn bộ kết quả kiểm thử tại:\n{out_dir}\n".encode("utf-8"))


if __name__ == "__main__":
    run_before_after_test()
