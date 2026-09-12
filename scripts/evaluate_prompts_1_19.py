"""
scripts/evaluate_prompts_1_19.py

BỘ KIỂM THỬ TRỰC QUAN CÔ LẬP CÁC PROMPT YÊU CẦU TEXT TỪ DÒNG 1-19 (prompt_test.txt)
===================================================================================
Tập trung kiểm thử khả năng xử lý của LLM và OmniBlock Layout khi:
- Người dùng không điền form mà gõ toàn bộ yêu cầu trong prompt tự do.
- Prompt chỉ định rõ ràng từng câu text kèm vị trí không gian (góc trên, ở giữa, phía dưới).
- Prompt yêu cầu phong cách ánh sáng, màu sắc và kiểu dáng phông chữ cụ thể.
- Kiểm tra cơ chế sinh Corridor Mask, bảo vệ Product Sanctuary, và kết xuất Playwright.

Xuất kết quả ra: output_visual_eval_isolated/prompts_1_19/
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tendoo.core.base import ColorPalette, PosterContent
from tendoo.core.colors import analyze_color_harmony
from tendoo.demo_server import (
    _auto_assign_zones,
    pil_to_base64_data_uri,
)
from tendoo.engine.geometry import get_product_sanctuary_rect
from tendoo.engine.layout import OmniBlockLayout
from tendoo.layouts.style_matcher import CATEGORY_FIELD_SLOTS, DEFAULT_FIELD_ROLE
from tendoo.llm_render_plan_server import validate_render_plan
from tendoo.poster_renderer import PosterRenderer
from scripts.evaluate_isolated_pipeline import (
    create_diffusion_mock_background,
    create_mask_overlay,
)


@dataclass
class PromptTestScenario:
    case_id: str
    line_number: int
    title: str
    aspect_ratio: str
    category: str
    raw_prompt: str
    brand_info: Dict[str, str]
    simulated_llm_plan: Dict[str, Any]


def get_prompts_1_19_scenarios() -> List[PromptTestScenario]:
    return [
        # 1. Line 1: Smartwatch bàn cafe gỗ nắng sớm (4:5)
        PromptTestScenario(
            case_id="prompt_line_01_cafe_lifestyle",
            line_number=1,
            title="Line 1: Smartwatch Bàn Cafe Gỗ Nắng Sớm (4:5)",
            aspect_ratio="4:5",
            category="product_intro",
            raw_prompt="Một chiếc đồng hồ thông minh hiện đại cao cấp với dây đeo kim loại màu bạc bóng bẩy, đặt trên chiếc bàn cà phê bằng gỗ mộc mạc cạnh một tách cà phê latte art và cặp kính râm thời trang. Ánh nắng ban mai nhẹ nhàng chiếu qua cửa sổ, bầu không khí ấm áp. Ở góc trên bên trái, văn bản 'THỜI GIAN LÀ CỦA BẠN' bằng phông chữ sans-serif trắng nhỏ. Ở giữa bên trái, văn bản 'NÂNG TẦM PHONG CÁCH ĐỜI SỐNG' lớn hơn, màu trắng, tinh tế. Chụp bằng ống kính 35mm, chân thực, 8k. --ar 4:5",
            brand_info={"brand": "Chronos Luxury", "hotline": "1800 6868"},
            simulated_llm_plan={
                "title": {"action": "use_prompt", "text": "NÂNG TẦM PHONG CÁCH ĐỜI SỐNG"},
                "extra_blocks": [
                    {
                        "field": None,
                        "text": "THỜI GIAN LÀ CỦA BẠN",
                        "zone": "top_left",
                        "role": "badge",
                        "icon": "clock",
                    },
                    {
                        "field": None,
                        "text": "NÂNG TẦM PHONG CÁCH ĐỜI SỐNG",
                        "zone": "middle_left",
                        "role": "hero",
                    },
                ],
                "scene_prompt": "A modern luxury smartwatch with sleek polished silver metallic strap, resting on a rustic wooden coffee table next to an artisan latte art cup and stylish sunglasses, warm gentle morning sunlight streaming through window, 35mm photography, ultra realistic",
                "style_hint": "warm_wood",
                "font_key": "bevietnam",
            },
        ),

        # 2. Line 3: Smartwatch thể thao chạy bộ hoàng hôn (9:16)
        PromptTestScenario(
            case_id="prompt_line_03_sports_runner",
            line_number=3,
            title="Line 3: Cổ Tay Cơ Bắp Thể Thao Đèn Neon Hoàng Hôn (9:16)",
            aspect_ratio="9:16",
            category="product_intro",
            raw_prompt="Cận cảnh một cổ tay cơ bắp đeo chiếc đồng hồ thông minh thể thao màu đen mang phong cách tương lai. Màn hình hiển thị nhịp tim phát sáng màu xanh neon. Nền là đường phố đô thị mờ ảo vào giờ vàng chiều tà trong một buổi chạy bộ. Hiệu ứng làm mờ chuyển động, ánh sáng điện ảnh. Phía trên cùng, văn bản 'CHINH PHỤC MỌI GIỚI HẠN' bằng phông chữ thể thao đậm, màu trắng. Phía dưới, văn bản 'DÒNG ĐỒNG HỒ THỂ THAO CAO CẤP' nhỏ hơn, màu trắng. --ar 9:16",
            brand_info={"brand": "Titan Sport Tech", "hotline": "0988 777 999"},
            simulated_llm_plan={
                "title": {"action": "use_prompt", "text": "CHINH PHỤC MỌI GIỚI HẠN"},
                "extra_blocks": [
                    {
                        "field": None,
                        "text": "CHINH PHỤC MỌI GIỚI HẠN",
                        "zone": "top_center",
                        "role": "hero",
                    },
                    {
                        "field": None,
                        "text": "DÒNG ĐỒNG HỒ THỂ THAO CAO CẤP",
                        "zone": "bottom_center",
                        "role": "badge",
                        "icon": "zap",
                    },
                ],
                "scene_prompt": "Extreme close-up of an athletic muscular wrist wearing a futuristic black sports smartwatch, glowing neon blue heart rate monitor display, blurred golden hour sunset urban street during an evening run, cinematic motion blur lighting, 8k",
                "style_hint": "cyberpunk_grid",
                "font_key": "beausans",
            },
        ),

        # 3. Line 5: Gia đình công viên nắng ấm (1:1)
        PromptTestScenario(
            case_id="prompt_line_05_family_park",
            line_number=5,
            title="Line 5: Gia Đình & Công Viên Nắng Ấm (1:1)",
            aspect_ratio="1:1",
            category="product_intro",
            raw_prompt="Một người cha trẻ đang mỉm cười, đeo chiếc đồng hồ thông minh màu xanh navy thời trang khi đang chơi đùa cùng con trong một công viên xanh ngập nắng. Lấy nét sắc sảo vào chiếc đồng hồ trên cổ tay, gia đình phía sau hơi mờ đi. Ánh sáng tự nhiên, nhiếp ảnh phong cách sống, vui vẻ. Ở giữa phía trên, văn bản 'KẾT NỐI YÊU THƯƠNG' bằng phông chữ serif ấm áp, màu nâu nhạt. Phía dưới, văn bản 'NGƯỜI BẠN ĐỒNG HÀNH CỦA GIA ĐÌNH' nhỏ hơn. --ar 1:1",
            brand_info={"brand": "Family Companion Watch", "hotline": "1900 6688"},
            simulated_llm_plan={
                "title": {"action": "use_prompt", "text": "KẾT NỐI YÊU THƯƠNG"},
                "extra_blocks": [
                    {
                        "field": None,
                        "text": "KẾT NỐI YÊU THƯƠNG",
                        "zone": "top_center",
                        "role": "hero",
                    },
                    {
                        "field": None,
                        "text": "NGƯỜI BẠN ĐỒNG HÀNH CỦA GIA ĐÌNH",
                        "zone": "bottom_center",
                        "role": "badge",
                        "icon": "heart",
                    },
                ],
                "scene_prompt": "A smiling young father wearing a stylish navy blue smartwatch playing with his child in a sunlit lush green park, sharp focus on the smartwatch on his wrist with softly blurred family background, natural cheerful lifestyle photography",
                "style_hint": "daylight_clean",
                "font_key": "playfair",
            },
        ),

        # 4. Line 7: Tạp chí thời trang vàng hồng sang trọng (4:5)
        PromptTestScenario(
            case_id="prompt_line_07_fashion_magazine",
            line_number=7,
            title="Line 7: Tạp Chí Thời Trang Vàng Hồng Sang Trọng (4:5)",
            aspect_ratio="4:5",
            category="product_intro",
            raw_prompt="Ảnh chụp tạp chí thời trang thanh lịch. Một người phụ nữ đeo chiếc đồng hồ thông minh màu vàng hồng sang trọng với dây đeo dạng lưới mịn, kết hợp cùng trang sức vàng tối giản. Nền màu hồng pastel và be, ánh sáng studio mềm mại. Phía trên, văn bản 'ĐẲNG CẤP & SANG TRỌNG' bằng phông chữ serif tinh tế, màu vàng kim. Phía dưới, văn bản 'BIỂU TƯỢNG THỜI TRANG MỚI' nhỏ hơn. --ar 2:3",
            brand_info={"brand": "Élégance Paris", "hotline": "0911 888 222"},
            simulated_llm_plan={
                "title": {"action": "use_prompt", "text": "ĐẲNG CẤP & SANG TRỌNG"},
                "extra_blocks": [
                    {
                        "field": None,
                        "text": "ĐẲNG CẤP & SANG TRỌNG",
                        "zone": "top_center",
                        "role": "hero",
                    },
                    {
                        "field": None,
                        "text": "BIỂU TƯỢNG THỜI TRANG MỚI",
                        "zone": "bottom_center",
                        "role": "badge",
                        "icon": "sparkles",
                    },
                ],
                "scene_prompt": "High fashion editorial magazine shoot, an elegant woman wearing an opulent rose gold smartwatch with fine mesh strap, minimal gold jewelry, soft studio diffused lighting against warm beige and pastel pink backdrop, 8k luxury style",
                "style_hint": "luxury_gold",
                "font_key": "playfair",
            },
        ),

        # 5. Line 13: Tối giản bục đen spotlight (16:9)
        PromptTestScenario(
            case_id="prompt_line_13_minimalist_podium",
            line_number=13,
            title="Line 13: Bục Đen Tối Giản Ánh Sáng Chiếu Điểm (16:9)",
            aspect_ratio="16:9",
            category="product_intro",
            raw_prompt="Ảnh chụp sản phẩm tối giản của một chiếc đồng hồ thông minh màu đen bóng bẩy đặt trên một bục màu đen bóng. Ánh sáng đèn chiếu điểm màu trắng tinh khiết rọi thẳng từ trên xuống tạo hiệu ứng ấn tượng. Nền tối có chiều sâu. Ở giữa phía trên, văn bản 'SỰ TINH TẾ CỦA SỨC MẠNH' bằng phông chữ sans-serif tối giản, màu trắng lớn. Phía dưới, văn bản 'CHỈ CÓ TẠI TENDOO CHRONOS' nhỏ hơn, màu trắng. --ar 16:9",
            brand_info={"brand": "Tendoo Chronos Flagship", "hotline": "1800 9999"},
            simulated_llm_plan={
                "title": {"action": "use_prompt", "text": "SỰ TINH TẾ CỦA SỨC MẠNH"},
                "extra_blocks": [
                    {
                        "field": None,
                        "text": "SỰ TINH TẾ CỦA SỨC MẠNH",
                        "zone": "top_center",
                        "role": "hero",
                    },
                    {
                        "field": None,
                        "text": "CHỈ CÓ TẠI TENDOO CHRONOS",
                        "zone": "bottom_center",
                        "role": "badge",
                        "icon": "shield",
                    },
                ],
                "scene_prompt": "Minimalist high-end commercial product shot of a glossy black smartwatch resting on a polished black reflective pedestal, crisp pure white overhead spotlight beaming straight down creating high drama and deep shadows, ultra realistic",
                "style_hint": "studio_spotlight",
                "font_key": "montserrat",
            },
        ),

        # 6. Line 15: Macro dưới nước chống nước 50m (1:1)
        PromptTestScenario(
            case_id="prompt_line_15_underwater_macro",
            line_number=15,
            title="Line 15: Macro Dưới Nước Chống Nước 50M (1:1)",
            aspect_ratio="1:1",
            category="product_intro",
            raw_prompt="Ảnh chụp macro của một chiếc đồng hồ thông minh thể thao hầm hố ngập trong nước, làn nước trong vắt với những bong bóng nhỏ bao quanh mặt đồng hồ đang phát sáng. Chuyển động nước văng tung tóe, ánh sáng xanh lơ rực rỡ. Thể hiện độ bền. Ở góc dưới bên trái, văn bản 'CHỐNG NƯỚC 50M' bằng phông chữ sans-serif trắng đậm, nhỏ. Phía dưới, văn bản 'SẴN SÀNG CHO MỌI CUỘC PHIÊU LƯU' màu trắng. --ar 1:1",
            brand_info={"brand": "AquaDiver Pro", "hotline": "1900 8822"},
            simulated_llm_plan={
                "title": {"action": "use_prompt", "text": "SẴN SÀNG CHO MỌI CUỘC PHIÊU LƯU"},
                "extra_blocks": [
                    {
                        "field": None,
                        "text": "CHỐNG NƯỚC 50M",
                        "zone": "bottom_left",
                        "role": "badge",
                        "icon": "shield",
                    },
                    {
                        "field": None,
                        "text": "SẴN SÀNG CHO MỌI CUỘC PHIÊU LƯU",
                        "zone": "top_center",
                        "role": "hero",
                    },
                ],
                "scene_prompt": "Macro splash photography of a rugged athletic sports smartwatch submerged in crystalline clear water, tiny effervescent bubbles surrounding the glowing watch face, dynamic water droplets and cyan aquatic lighting, hyper realistic",
                "style_hint": "cyberpunk_grid",
                "font_key": "beausans",
            },
        ),

        # 7. Line 19: Đỉnh núi bình minh dã ngoại (16:9)
        PromptTestScenario(
            case_id="prompt_line_19_mountain_sunrise",
            line_number=19,
            title="Line 19: Đỉnh Núi Hoàng Hôn & Khám Phá Thế Giới (16:9)",
            aspect_ratio="16:9",
            category="product_intro",
            raw_prompt="Một chiếc đồng hồ thông minh chuyên dụng ngoài trời trên cổ tay của một người đi bộ đường dài, hướng về phía dãy núi ngoạn mục lúc bình minh. Màn hình đồng hồ hiển thị bản đồ độ cao. Nền phong cảnh hùng vĩ, vệt sáng mặt trời tuyệt đẹp. Ở giữa, văn bản 'KHÁM PHÁ THẾ GIỚI CÙNG BẠN' bằng phông chữ serif phiêu lưu, màu trắng lớn. Phía dưới, văn bản 'NGƯỜI BẠN ĐỒNG HÀNH TRÊN MỌI NẺO ĐƯỜNG' nhỏ hơn. --ar 16:9",
            brand_info={"brand": "Apex Trekker", "hotline": "0933 555 111"},
            simulated_llm_plan={
                "title": {"action": "use_prompt", "text": "KHÁM PHÁ THẾ GIỚI CÙNG BẠN"},
                "extra_blocks": [
                    {
                        "field": None,
                        "text": "KHÁM PHÁ THẾ GIỚI CÙNG BẠN",
                        "zone": "top_center",
                        "role": "hero",
                    },
                    {
                        "field": None,
                        "text": "NGƯỜI BẠN ĐỒNG HÀNH TRÊN MỌI NẺO ĐƯỜNG",
                        "zone": "bottom_center",
                        "role": "badge",
                        "icon": "globe",
                    },
                ],
                "scene_prompt": "A rugged outdoor expedition smartwatch on the wrist of a mountain hiker pointed toward a breathtaking mountain range at sunrise, glowing topographical map on screen, golden sun flare, epic landscape photography",
                "style_hint": "luxury_gold",
                "font_key": "playfair",
            },
        ),
    ]


def run_prompts_1_19_suite():
    output_base = PROJECT_ROOT / "output_visual_eval_isolated" / "prompts_1_19"
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    scenarios = get_prompts_1_19_scenarios()
    layout = OmniBlockLayout()

    print("\n" + "=" * 85)
    print("CHẠY BỘ KIỂM THỬ CÁC PROMPT YÊU CẦU TEXT THÊM TỪ DÒNG 1-19 (7 SCENARIOS)")
    print(f"Thư mục lưu kết quả: {output_base}")
    print("=" * 85 + "\n")

    summary_list = []
    t_start = time.time()

    for idx, sc in enumerate(scenarios, 1):
        print(f"[{idx:02d}/{len(scenarios):02d}] Đang chạy {sc.case_id} ({sc.aspect_ratio}, Line {sc.line_number})...")
        case_dir = output_base / sc.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        if sc.aspect_ratio == "9:16":
            w, h = 576, 1024
        elif sc.aspect_ratio == "16:9":
            w, h = 1024, 576
        elif sc.aspect_ratio == "4:5":
            w, h = 816, 1024
        else:
            w, h = 1024, 1024

        # Validate LLM plan
        valid_plan, errors = validate_render_plan(sc.simulated_llm_plan, sc.category, {})
        assert valid_plan is not None

        style_hint = valid_plan.get("style_hint") or "studio_spotlight"
        font_key = valid_plan.get("font_key") or "bevietnam"

        # Dựng blocks
        plan_blocks: List[Dict[str, Any]] = []
        final_title = valid_plan["title"].get("text")

        for b in valid_plan.get("extra_blocks", []):
            blk = {
                "text": b["text"],
                "zone": b.get("zone"),
                "role": b.get("role") or "body",
                "style_variant": "pill_badge" if b.get("role") == "badge" else None,
            }
            if b.get("icon"):
                blk["icon"] = b["icon"]
            if b.get("color"):
                blk["color"] = b["color"]
            plan_blocks.append(blk)

        # Thông tin brand / store
        st_parts = []
        if sc.brand_info.get("brand"):
            st_parts.append(sc.brand_info["brand"])
        if sc.brand_info.get("hotline"):
            st_parts.append(f"Hotline: {sc.brand_info['hotline']}")
        if st_parts:
            plan_blocks.append({
                "text": "  •  ".join(st_parts),
                "zone": "bottom_bar",
                "role": "brand_bar",
                "field": "store_info",
                "icon": "phone",
            })

        _auto_assign_zones(plan_blocks)

        content = PosterContent(
            headline=final_title or "",
            brand=sc.brand_info.get("brand", ""),
            hotline=sc.brand_info.get("hotline", ""),
            category=sc.category,
            font_family=font_key,
            free_text_blocks=plan_blocks,
        )

        # 1. Sinh Mask qua Playwright
        dummy_palette = ColorPalette(is_dark=True, luminance=0.5, hue=0, comp_hue=180, headline_color="#FFFFFF")
        neutral_bg_uri = pil_to_base64_data_uri(Image.new("RGB", (1, 1), (50, 50, 50)))
        measure_html = layout.render_html(
            content=content,
            palette=dummy_palette,
            bg_data_uri=neutral_bg_uri,
            width=w,
            height=h,
            style_hint=style_hint,
        )

        mask_np = layout.generate_mask_from_render(measure_html, width=w, height=h)
        if mask_np is None:
            mask_np = layout.generate_mask(width=w, height=h, blocks=plan_blocks, font_key=font_key)

        mask_vis = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_vis.save(case_dir / "01_corridor_mask.png")

        # 2. Sinh Mock Background theo Mask
        mock_bg_pil = create_diffusion_mock_background(w, h, style_hint=style_hint, mask_np=mask_np)
        mock_bg_pil.save(case_dir / "02_mock_background.png")

        # 3. Render Poster hoàn chỉnh
        safe_zone = layout.get_safe_zone()
        palette = analyze_color_harmony(np.array(mock_bg_pil), safe_zone, color_mode="auto")
        bg_data_uri = pil_to_base64_data_uri(mock_bg_pil)

        final_html = layout.render_html(
            content=content,
            palette=palette,
            bg_data_uri=bg_data_uri,
            width=w,
            height=h,
            style_hint=style_hint,
        )
        (case_dir / "poster_markup.html").write_text(final_html, encoding="utf-8")

        poster_file = case_dir / "03_poster.png"
        PosterRenderer.render(html_content=final_html, output_image_path=poster_file, width=w, height=h)

        # 4. Mask Overlay
        poster_pil = Image.open(poster_file)
        overlay_pil = create_mask_overlay(poster_pil, mask_np)
        overlay_pil.save(case_dir / "04_mask_overlay.png")

        # 5. Lưu LLM plan JSON
        (case_dir / "llm_plan.json").write_text(json.dumps(valid_plan, indent=2, ensure_ascii=False), encoding="utf-8")

        # 6. Chỉ số kỹ thuật
        s_x1, s_y1, s_x2, s_y2 = get_product_sanctuary_rect(w, h)
        sanctuary_sub = mask_np[s_y1:s_y2, s_x1:s_x2]
        leakage_max = float(sanctuary_sub.max()) if sanctuary_sub.size > 0 else 0.0
        mask_coverage = float((mask_np > 0.30).mean() * 100.0)

        report_txt = f"""================================================================================
BÁO CÁO KIỂM THỬ PROMPT DÒNG {sc.line_number}: {sc.case_id}
================================================================================
Tiêu đề: {sc.title}
Prompt gốc: {sc.raw_prompt}
Tỉ lệ khung hình: {sc.aspect_ratio} ({w}x{h} px)
Phong cách (Style Hint): {style_hint} | Phông chữ (Font): {font_key}

[1] CÁC KHỐI CHỮ TRÍCH XUẤT TỪ PROMPT:
"""
        for b_idx, b in enumerate(plan_blocks, 1):
            report_txt += f"  {b_idx:>2}. Zone: {b.get('zone'):<14} | Role: {b.get('role'):<10} | Text: '{b.get('text')}'\n"

        report_txt += f"""
[2] CHỈ SỐ MẶT NẠ HÀNH LANG (CORRIDOR MASK FIT):
  - Tỉ lệ phủ mask: {mask_coverage:.1f}%
  - Rò rỉ tâm sản phẩm (Sanctuary Leakage): {leakage_max:.2f}
  - Trạng thái: {"AN TOÀN" if leakage_max <= 0.40 else "CHẠM NHẸ MẶT NẠ"}
================================================================================
"""
        (case_dir / "05_report.txt").write_text(report_txt, encoding="utf-8")

        summary_list.append({
            "case_id": sc.case_id,
            "line": sc.line_number,
            "ratio": sc.aspect_ratio,
            "num_blocks": len(plan_blocks),
            "leakage": leakage_max,
            "coverage": mask_coverage,
            "style": style_hint,
        })
        print(f"       -> [XONG] {len(plan_blocks)} blocks | Mask: {mask_coverage:.1f}% | Leak: {leakage_max:.2f}")

    total_time = time.time() - t_start

    summary_md = f"""# Báo Cáo Tổng Hợp Kiểm Thử Các Prompt Dòng 1-19 (prompt_test.txt)
*Tổng thời gian thực thi: {total_time:.2f}s*

| STT | Dòng | Mã Kịch Bản | Tỉ Lệ | Blocks | Phủ Mask | Leak Tâm | Phong Cách | Trạng Thái |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
"""
    for idx, s in enumerate(summary_list, 1):
        status = "PASSED" if s["leakage"] <= 0.40 else "FEATHER_EDGE"
        summary_md += f"| {idx:02d} | L{s['line']:02d} | `{s['case_id']}` | {s['ratio']} | {s['num_blocks']} | {s['coverage']:.1f}% | {s['leakage']:.2f} | `{s['style']}` | **{status}** |\n"

    (output_base / "summary_report.md").write_text(summary_md, encoding="utf-8")
    print("\n" + "=" * 85)
    print("HOÀN THÀNH TẤT CẢ CÁC KỊCH BẢN TỪ DÒNG 1-19!")
    print(f"Toàn bộ ảnh và báo cáo đã lưu tại:\n{output_base}\n")


if __name__ == "__main__":
    run_prompts_1_19_suite()
