#!/usr/bin/env python3
"""
scripts/test_hero_selector.py

==================================================================================================
TENDOO AI - STAGE 1 "HERO SELECTOR + BLUEPRINT" PROMPT TEST (OPEN-SOURCE, ZERO FINE-TUNE)
==================================================================================================

WHY THIS SCRIPT?
  Inspired by PosterVerse (AAAI 2026, arXiv:2601.03993) Stage 1 "Blueprint Creation", but
  deliberately NOT fine-tuned -- per explicit decision, we only borrow the PIPELINE STRUCTURE
  (Blueprint -> Background -> Layout+Typography), not their training investment. This script
  tests whether a strong OFF-THE-SHELF, OPEN-SOURCE model (Qwen2.5-VL-7B-Instruct, text-only
  here -- Stage 1 never sees an image, exactly like PosterVerse's own Stage 1) can, via prompting
  alone, reliably:
    1. Parse a free-form Vietnamese ad request (any of Tendoo's 7 poster categories) into a
       structured blueprint.
    2. Pick EXACTLY ONE "hero" text block (the one that goes through FLUX.2 glyph injection @
       t=10 for photoreal 3D/material integration) -- per the decided hybrid architecture, every
       other text block is a "secondary" element destined for the HTML/CSS typography overlay
       (src/tendoo/typography_engine.py), never for the diffusion model.
    3. Produce a `background_prompt` that is 100% free of literal hero/secondary text strings --
       this is a hard correctness check, not a style preference: AGENTS.md Rule 33 showed that
       putting literal text back into the DiT-facing prompt corrupts even an isolated, otherwise-
       bulletproof glyph render. This script asserts that guarantee automatically on every case.

TEST CASES: one per poster category, reusing real content already in this repo instead of
inventing new copy:
  - promo / product_intro : prompt_test.txt line 1 (đồng hồ thông minh, 2 text blocks)
  - feedback               : prompt_test.txt line 21 (gym PT feedback, 5 text roles)
  - grand_opening          : generate_distill_4cases.py CASES_CONFIG["case1_burger_opening"]
  - recruitment            : generate_distill_4cases.py CASES_CONFIG["case3_recruitment"]
  - menu                   : generate_distill_4cases.py CASES_CONFIG["case4_restaurant_menu"]

USAGE:
  python scripts/test_hero_selector.py --model Qwen/Qwen2.5-VL-7B-Instruct
  python scripts/test_hero_selector.py --model Qwen/Qwen2.5-VL-7B-Instruct --case promo

NOTE: Rule 28 (the old HTML-output ban) has been retracted -- this script writes plain JSON
files anyway because that is the right format for structured data, not because HTML is banned.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================================================================================================
# 1. TEST CASES (real content already used elsewhere in the repo -- not invented for this test)
# ==================================================================================================

TEST_CASES: Dict[str, Dict[str, str]] = {
    "promo": {
        "category_hint": "promo / product_intro",
        "request": (
            "Một chiếc đồng hồ thông minh hiện đại cao cấp với dây đeo kim loại màu bạc bóng bẩy, "
            "đặt trên chiếc bàn cà phê bằng gỗ mộc mạc cạnh một tách cà phê latte art và cặp kính râm "
            "thời trang. Ánh nắng ban mai nhẹ nhàng chiếu qua cửa sổ, bầu không khí ấm áp. Ở góc trên "
            "bên trái, văn bản \"THỜI GIAN LÀ CỦA BẠN\" bằng phông chữ sans-serif trắng nhỏ. Ở giữa bên "
            "trái, văn bản \"NÂNG TẦM PHONG CÁCH ĐỜI SỐNG\" lớn hơn, màu trắng, tinh tế. Chụp bằng ống "
            "kính 35mm, chân thực, 8k. --ar 4:5"
        ),
    },
    "feedback": {
        "category_hint": "feedback khách hàng",
        "request": (
            "Tạo ảnh feedback khách hàng cho dịch vụ gym & PT cao cấp. Tiêu đề: \"Khách hàng nói gì sau "
            "90 ngày thay đổi?\". Tên sản phẩm/dịch vụ nhận feedback: \"Private Coaching Transformation\". "
            "Mô tả ngắn feedback: \"97% khách hàng hài lòng với kết quả tăng cơ, cải thiện vóc dáng và sức "
            "khỏe chỉ sau 3 tháng tập luyện cùng PT riêng\". Điểm nổi bật của sản phẩm/dịch vụ: \"PT 1:1 "
            "riêng tư, giáo án cá nhân hóa, theo dõi dinh dưỡng và form tập chi tiết\". Ưu đãi đặc biệt: "
            "\"Giảm 20% gói PT tháng đầu cho khách hàng mới\". Thiết kế phong cách fitness premium hiện "
            "đại, background phòng gym sang trọng, hiển thị review 5 sao và hình ảnh khách hàng "
            "before/after, tone đen đỏ mạnh mẽ, typography nổi bật, ánh sáng cinematic, bố cục chuyên "
            "nghiệp như quảng cáo fitness trên Facebook/Instagram, ultra realistic, high detail."
        ),
    },
    "grand_opening": {
        "category_hint": "banner khai trương",
        "request": (
            "Poster quảng cáo ẩm thực cao cấp, chiếc bánh burger bò đẫm phô mai tan chảy bốc khói "
            "nghi ngút trên thớt gỗ rustic sẫm màu, hạt tiêu và sốt bơ nấm bóng bẩy, ánh sáng studio "
            "vàng ấm tương phản cao. Tiêu đề lớn 3D mạ vàng kim loại nổi bật phát sáng ở phía trên: "
            "\"TƯNG BỪNG KHAI TRƯƠNG\". Thêm badge nhỏ góc trên: \"MUA 1 TẶNG 1\". Footer: hotline "
            "\"1900 6868\" và địa chỉ \"123 Nguyễn Huệ, Q1, TP.HCM\". Bố cục cân đối điện ảnh sang "
            "trọng, không có watermark."
        ),
    },
    "recruitment": {
        "category_hint": "tin tuyển dụng",
        "request": (
            "Poster tuyển dụng công nghệ cao, không gian văn phòng AI research lab hiện đại với tường "
            "kính và dải đèn led neon xanh dương cyberpunk, các kỹ sư làm việc mờ ảo có chiều sâu ở hậu "
            "cảnh. Tiêu đề lớn: \"SENIOR AI DESIGNER\". Mô tả ngắn: \"Mức lương 2000-3500 USD, làm việc "
            "hybrid, đãi ngộ bảo hiểm cao cấp\". Yêu cầu: \"3+ năm kinh nghiệm Figma, Midjourney, hiểu "
            "biết Generative AI\". CTA: \"Ứng tuyển ngay hôm nay\". Liên hệ: \"hr@tendoo.ai\"."
        ),
    },
    "menu": {
        "category_hint": "menu món ăn",
        "request": (
            "Ảnh thực đơn nhà hàng cao cấp, món phở bò tái nạm bốc khói nghi ngút trong tô sứ trắng "
            "trên bàn gỗ tối màu, rau thơm tươi xanh bên cạnh. Tiêu đề lớn: \"THỰC ĐƠN ĐẶC BIỆT\". Danh "
            "sách món: \"Phở Tái Nạm - 65.000đ\", \"Phở Gầu Sốt Vang - 75.000đ\", \"Phở Đặc Biệt - "
            "85.000đ\". Footer: \"Mở cửa 6h00 - 22h00 hàng ngày\"."
        ),
    },
}

# ==================================================================================================
# 2. HERO SELECTOR SYSTEM PROMPT
# ==================================================================================================

HERO_SELECTOR_SYSTEM_PROMPT = """Bạn là chuyên gia thiết kế poster thương mại, đóng vai trò TẦNG 1 (Blueprint + Hero Selector)
của một pipeline sinh poster gồm 3 tầng:
  Tầng 1 (BẠN): phân tích yêu cầu tự do của người dùng -> xuất blueprint JSON có cấu trúc.
  Tầng 2: một mô hình diffusion (FLUX.2) vẽ ẢNH NỀN + đúng 1 khối chữ "hero" bằng kỹ thuật
          glyph injection (tái tạo 100% chính tả, cho phép hiệu ứng vật lý thật: dập nổi 3D,
          mạ vàng, neon phát sáng, khắc chìm... vì chữ đó được "vẽ" thật vào cảnh).
  Tầng 3: một engine HTML/CSS overlay toàn bộ các khối chữ CÒN LẠI (subtitle, badge, CTA,
          rating, spec, thông tin liên hệ...) lên trên ảnh Tầng 2 -- những khối này KHÔNG được
          diffusion vẽ, chúng là chữ vector/HTML render chính xác tuyệt đối.

NHIỆM VỤ CỦA BẠN:
1. Đọc yêu cầu người dùng (có thể sơ sài hoặc rất chi tiết, thuộc 1 trong nhiều loại: khuyến mại,
   ưu đãi, giới thiệu sản phẩm, banner khai trương, feedback khách hàng, tin tuyển dụng, menu món,
   hoặc yêu cầu tự do khác).
2. Chọn ra ĐÚNG 1 khối chữ làm "hero" -- khối quan trọng/nổi bật nhất về mặt thị giác, thường là:
   tiêu đề chính, tên chiến dịch, hoặc 1 con số/thống kê ấn tượng (tuỳ ngữ cảnh, KHÔNG máy móc luôn
   chọn "câu đầu tiên"). TUYỆT ĐỐI chỉ 1 hero, không hơn, không kém -- kể cả khi bài không có khối
   nào thực sự nổi bật, vẫn phải chọn 1 khối hợp lý nhất để giữ kiến trúc nhất quán.
3. Toàn bộ khối chữ còn lại xếp vào "secondary_elements", mỗi khối gán 1 "role" phù hợp
   (subtitle | badge | cta | quote | spec | contact | rating | price_item | other).
4. Viết "background_prompt" mô tả CẢNH/PHONG CÁCH/CHẤT LIỆU cho Tầng 2 -- annotate rõ vị trí,
   quy mô, chất liệu của riêng khối hero (theo đúng 3 thành phần bắt buộc: vị trí không gian +
   quy mô/vai trò + vật lý/chất liệu/quang học). TUYỆT ĐỐI KHÔNG được chứa bất kỳ chuỗi chữ literal
   nào của hero hoặc của bất kỳ secondary_elements nào -- kể cả trong ngoặc kép, kể cả một phần --
   vì điều này đã được thực nghiệm xác nhận làm hỏng luôn cả khối chữ hero vốn dĩ hoàn hảo.
5. Điền "template_brief" -- đây là dữ liệu THẬT sẽ đổ trực tiếp vào layout HTML cố định của
   Tầng 3 (không qua model nào nữa), nên PHẢI dùng ĐÚNG BỘ KHOÁ theo "category" đã chọn bên dưới,
   không tự bịa thêm khoá lạ, không đổi tên khoá. Nếu yêu cầu người dùng có bao nhiêu mục
   (ví dụ 5 món ăn, 6 yêu cầu tuyển dụng...) thì liệt kê đủ bấy nhiêu -- KHÔNG giới hạn số lượng
   theo ví dụ dưới đây, ví dụ chỉ để minh hoạ hình dạng dữ liệu.

BỘ KHOÁ "template_brief" BẮT BUỘC THEO TỪNG CATEGORY:
- grand_opening: brand, date_range, badge_label, badge_percent, badge_sub, address, offer_desc, cta_text
- feedback: brand, top_badge, stars, verified_label, quote_text, avatar_emoji, customer_name, customer_sub,
            features (list các {"icon","text"}, SỐ LƯỢNG TUỲ THEO YÊU CẦU), offer_title, offer_desc, cta_text
- recruitment: company, deadline, pos_label, salary, requirements (list string, số lượng tuỳ ý),
               benefits (list string, số lượng tuỳ ý), contact_line1, contact_email, cta_text
- menu: sub_brand, tagline, categories (list các {"title", "items": [{"name","price","badge"(tuỳ chọn)}]},
        SỐ DANH MỤC VÀ SỐ MÓN MỖI DANH MỤC TUỲ THEO YÊU CẦU NGƯỜI DÙNG, không cố định 2x2), footer_note, hotline
- promo / offer / product_intro / freeform (dùng layout "generic"): brand, eyebrow, badge, rating_value,
        rating_count, specs (list string), sub_slogan, cta_text, hotline, website

Trả về DUY NHẤT 1 khối JSON hợp lệ theo đúng schema sau, không thêm giải thích:
{
  "category": "promo|offer|product_intro|grand_opening|feedback|recruitment|menu|freeform",
  "canvas": {"width": <int>, "height": <int>, "aspect_ratio": "<vd 9:16>"},
  "background_prompt": "<mô tả cảnh, phong cách, vị trí/chất liệu hero -- KHÔNG chứa chữ literal>",
  "hero": {
    "text": "<chuỗi chữ chính xác>",
    "role": "<vd: title, headline, stat_callout>",
    "style_hint": "<vd: 3D dập nổi mạ vàng, ở giữa bên trái>"
  },
  "secondary_elements": [
    {"text": "<chuỗi chữ chính xác>", "role": "<subtitle|badge|cta|quote|spec|contact|rating|price_item|other>", "style_hint": "<gợi ý ngắn>"}
  ],
  "template_brief": { "<đúng bộ khoá theo category đã chọn ở trên, điền đủ số lượng mục thực tế>": "..." }
}
"""

# Ground truth of what typography_engine.py's PosterTemplateEngine actually reads per category --
# kept here (not just in the prompt) so validate_blueprint() can check the model didn't invent or
# drop keys, independent of whatever the LLM was told to do.
TEMPLATE_BRIEF_SCHEMAS: Dict[str, List[str]] = {
    "grand_opening": ["brand", "date_range", "badge_label", "badge_percent", "badge_sub", "address", "offer_desc", "cta_text"],
    "feedback": ["brand", "top_badge", "stars", "verified_label", "quote_text", "avatar_emoji", "customer_name", "customer_sub", "features", "offer_title", "offer_desc", "cta_text"],
    "recruitment": ["company", "deadline", "pos_label", "salary", "requirements", "benefits", "contact_line1", "contact_email", "cta_text"],
    "menu": ["sub_brand", "tagline", "categories", "footer_note", "hotline"],
    "generic": ["brand", "eyebrow", "badge", "rating_value", "rating_count", "specs", "sub_slogan", "cta_text", "hotline", "website"],
}
# Categories that route through the "generic" layout (no dedicated template exists yet).
GENERIC_CATEGORIES = {"promo", "offer", "product_intro", "freeform"}

USER_PROMPT_TEMPLATE = "Yêu cầu người dùng ({category_hint}):\n{request}"


# ==================================================================================================
# 3. VALIDATION (automated Rule 33 guard)
# ==================================================================================================

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def validate_blueprint(blueprint: Dict[str, Any]) -> List[str]:
    """Returns a list of validation errors (empty list = clean)."""
    errors: List[str] = []

    required_top = ["category", "canvas", "background_prompt", "hero", "secondary_elements"]
    for key in required_top:
        if key not in blueprint:
            errors.append(f"Missing top-level key: {key}")
    if errors:
        return errors

    hero = blueprint["hero"]
    secondary = blueprint["secondary_elements"]
    bg_prompt_norm = _normalize(blueprint["background_prompt"])

    if not isinstance(hero, dict) or "text" not in hero or not hero["text"].strip():
        errors.append("hero.text is missing or empty")
    if not isinstance(secondary, list):
        errors.append("secondary_elements is not a list")

    # Rule 33 guard: background_prompt must NEVER contain the literal hero/secondary text.
    all_literal_texts = []
    if isinstance(hero, dict) and hero.get("text"):
        all_literal_texts.append(hero["text"])
    if isinstance(secondary, list):
        for el in secondary:
            if isinstance(el, dict) and el.get("text"):
                all_literal_texts.append(el["text"])

    for txt in all_literal_texts:
        txt_norm = _normalize(txt)
        if len(txt_norm) >= 4 and txt_norm in bg_prompt_norm:
            errors.append(
                f"[RULE 33 VIOLATION] background_prompt contains literal text: \"{txt}\" -- "
                f"this WILL poison even the isolated hero render per AGENTS.md Rule 33."
            )

    # Sanity: exactly one hero (schema already enforces this structurally, but double check
    # the model didn't stuff multiple texts into hero.text separated by newlines/semicolons
    # as a workaround).
    if isinstance(hero, dict) and hero.get("text") and len(hero["text"]) > 80:
        errors.append(
            f"[WARNING] hero.text is unusually long ({len(hero['text'])} chars) -- verify the "
            f"model didn't merge multiple text blocks into one 'hero'."
        )

    # template_brief guard: must exist, and must use the EXACT key set typography_engine.py's
    # PosterTemplateEngine actually reads for the chosen category -- catches the model inventing
    # keys (silently ignored -> missing content in the final poster) or dropping required ones
    # (falls back to a hardcoded default -- exactly the "still hardcoded" gap being tested here).
    template_brief = blueprint.get("template_brief")
    category = blueprint.get("category", "")
    schema_key = "generic" if category in GENERIC_CATEGORIES else category
    expected_keys = TEMPLATE_BRIEF_SCHEMAS.get(schema_key)

    if template_brief is None:
        errors.append("Missing template_brief -- Stage 3 would fall back to hardcoded Python defaults, not the user's actual request.")
    elif expected_keys is not None:
        actual_keys = set(template_brief.keys())
        missing = set(expected_keys) - actual_keys
        unexpected = actual_keys - set(expected_keys)
        if missing:
            errors.append(f"template_brief missing keys for category='{category}': {sorted(missing)} -- these will silently fall back to hardcoded demo defaults.")
        if unexpected:
            errors.append(f"[WARNING] template_brief has keys PosterTemplateEngine won't read for category='{category}': {sorted(unexpected)} -- silently ignored, wasted model output.")
    else:
        errors.append(f"[WARNING] category='{category}' has no known template_brief schema -- verify this is intentional (new category not yet in TEMPLATE_BRIEF_SCHEMAS).")

    return errors


# ==================================================================================================
# 4. MODEL RUNNER (open-source, text-only -- Stage 1 never needs vision)
# ==================================================================================================

def run_case(model, tokenizer, case_id: str, case: Dict[str, str], device: str) -> Dict[str, Any]:
    import torch

    user_prompt = USER_PROMPT_TEMPLATE.format(
        category_hint=case["category_hint"], request=case["request"]
    )
    messages = [
        {"role": "system", "content": HERO_SELECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    # Extract JSON block
    match = re.search(r"\{.*\}", response, re.DOTALL)
    raw_json = match.group(0) if match else response

    result: Dict[str, Any] = {"case_id": case_id, "raw_response": response}
    try:
        blueprint = json.loads(raw_json)
        result["blueprint"] = blueprint
        result["validation_errors"] = validate_blueprint(blueprint)
    except json.JSONDecodeError as e:
        result["blueprint"] = None
        result["validation_errors"] = [f"JSON parse failure: {e}"]

    return result


def render_preview(blueprint: Dict[str, Any], out_path: Path, case_id: str) -> Optional[Path]:
    """
    Closes the Stage1 -> Stage3 loop LOCALLY, with NO GPU / NO diffusion: fakes a Stage-2
    background (plain gradient + a dark rectangle standing in for wherever the hero would land)
    and feeds the blueprint's own `template_brief` + `category` straight into
    PosterTemplateEngine -- proving (or disproving) that Stage 1's output alone is enough to
    render a DIFFERENT poster's actual content, not the hardcoded Python defaults.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from PIL import Image, ImageDraw
    from tendoo.typography_engine import PosterBackgroundAnalyzer, PosterRenderer, PosterTemplateEngine

    canvas = blueprint.get("canvas", {}) or {}
    w, h = int(canvas.get("width", 1024)), int(canvas.get("height", 1024))

    fake_bg = Image.new("RGB", (w, h), color=(18, 14, 10))
    draw = ImageDraw.Draw(fake_bg)
    for y in range(h):
        shade = int(8 + (y / h) * 35)
        draw.line([(0, y), (w, y)], fill=(shade, shade // 2, shade // 3))
    draw.rectangle([w * 0.25, h * 0.08, w * 0.75, h * 0.28], fill=(45, 32, 20))  # fake hero silhouette

    bg_path = out_path / f"{case_id}_fake_bg.png"
    fake_bg.save(bg_path)

    analysis = PosterBackgroundAnalyzer.analyze(bg_path)
    category = blueprint.get("category", "generic")
    template_category = category if category not in GENERIC_CATEGORIES else "generic"
    brief = blueprint.get("template_brief") or {}

    html = PosterTemplateEngine.generate_html(
        analysis=analysis, brief=brief, background_image_path=str(bg_path), category=template_category
    )
    png_path = out_path / f"{case_id}_preview.png"
    PosterRenderer.render(html_content=html, output_image_path=png_path, width=w, height=h)
    return png_path


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Stage 1 Hero Selector prompt test (open-source)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                         help="HF model id or local path (text-only usage -- any open-weight "
                              "instruction-tuned LLM/VLM works; Qwen2.5-VL-7B-Instruct matches "
                              "PosterVerse's own model family)")
    parser.add_argument("--case", type=str, default="all", choices=list(TEST_CASES.keys()) + ["all"],
                         help="Which test case to run (default: all)")
    parser.add_argument("--output_dir", type=str, default="output_hero_selector_test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render_preview", action="store_true",
                         help="Also render a local PNG preview per case (fake gradient background "
                              "with a placeholder hero box, no GPU/diffusion) by feeding the "
                              "blueprint's own template_brief straight into PosterTemplateEngine -- "
                              "proves Stage 1's output alone can fill a DIFFERENT poster's content, "
                              "not just the hardcoded Python defaults. Requires playwright installed.")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(" [*] TENDOO AI - STAGE 1 HERO SELECTOR PROMPT TEST (OPEN-SOURCE, ZERO FINE-TUNE)")
    print("=" * 100)
    print(f"  Model      : {args.model}")
    print(f"  Cases      : {list(TEST_CASES.keys()) if args.case == 'all' else [args.case]}")

    print("\n[1/2] Loading model & tokenizer (text-only generation)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    ).eval()

    cases_to_run = TEST_CASES if args.case == "all" else {args.case: TEST_CASES[args.case]}

    print(f"\n[2/2] Running {len(cases_to_run)} case(s)...\n")
    results = []
    for case_id, case in cases_to_run.items():
        print("-" * 80)
        print(f"CASE: {case_id} ({case['category_hint']})")
        res = run_case(model, tokenizer, case_id, case, args.device)
        results.append(res)

        if res["blueprint"] is not None:
            hero_text = res["blueprint"].get("hero", {}).get("text", "???")
            n_secondary = len(res["blueprint"].get("secondary_elements", []))
            print(f"  Hero        : \"{hero_text}\"")
            print(f"  Secondary   : {n_secondary} block(s)")
        errors = res["validation_errors"]
        if errors:
            print(f"  [!] VALIDATION ISSUES ({len(errors)}):")
            for e in errors:
                print(f"      - {e}")
        else:
            print("  [OK] Validation clean.")

        result_file = out_path / f"{case_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        if args.render_preview and res["blueprint"] is not None:
            try:
                png_path = render_preview(res["blueprint"], out_path, case_id)
                print(f"  [preview] Rendered: {png_path}")
            except Exception as e:
                print(f"  [preview] FAILED: {e}")

    # ASCII summary
    print("\n" + "=" * 100)
    print(f"{'CASE':<16} | {'HERO TEXT':<40} | {'#SECONDARY':<10} | {'ISSUES':<8}")
    print("-" * 100)
    for res in results:
        bp = res.get("blueprint") or {}
        hero_text = (bp.get("hero", {}) or {}).get("text", "(parse failed)")[:38]
        n_sec = len(bp.get("secondary_elements", []) or [])
        n_issues = len(res["validation_errors"])
        print(f"{res['case_id']:<16} | {hero_text:<40} | {n_sec:<10} | {n_issues:<8}")
    print("=" * 100)
    print(f"\n[✓] Results saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
