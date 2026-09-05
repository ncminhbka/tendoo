#!/usr/bin/env python3
"""
scripts/test_hero_selector.py

==================================================================================================
TENDOO AI - STAGE 1 "HERO SELECTOR + BLUEPRINT" PROMPT TEST (OPEN-SOURCE, ZERO FINE-TUNE)
==================================================================================================

WHY THIS SCRIPT?
  Inspired by PosterVerse (AAAI 2026, arXiv:2601.03993) Stage 1 "Blueprint Creation", but
  deliberately NOT fine-tuned -- per explicit decision, we only borrow the PIPELINE STRUCTURE
  (Blueprint -> Background -> Layout+Typography), not their training investment.

  UPDATED for the "100%-overlay" architecture (superseding the earlier hybrid hero-via-diffusion
  design): diffusion NEVER draws any text at all anymore, not even a title -- Stage 2 is a pure
  prompt-to-image call (scripts/generate_textfree_background.py), and 100% of typography,
  including what used to be the "hero", is rendered by src/tendoo/typography_engine.py's
  PosterTemplateEngine. There is no more "hero vs secondary" split to make here -- ALL text is
  just structured content for whichever category template will render it.

  This script tests whether a strong OFF-THE-SHELF, OPEN-SOURCE model (Qwen2.5-VL-7B-Instruct,
  text-only here -- Stage 1 never sees an image, exactly like PosterVerse's own Stage 1) can, via
  prompting alone, reliably:
    1. Parse a free-form Vietnamese ad request into a structured blueprint: which category/
       template it matches, a scene-only `background_prompt`, a `style_theme` mood hint, and the
       full text content shaped exactly like that category's `template_brief` schema.
    2. Produce a `background_prompt` containing ZERO literal content text -- still enforced, but
       the reasoning changed: it's no longer "protects one isolated glyph reference" (AGENTS.md
       Rule 33 -- moot now, nothing is drawn as a glyph anymore), it's "keeps Stage 2's text-free
       generation from being tempted to hallucinate garbled text just because the prompt mentions
       words it should be picturing instead of writing."

TEST CASES: one per template category, reusing real content already in this repo instead of
inventing new copy:
  - product_ad   : prompt_test.txt line 1 (đồng hồ thông minh, 2 text blocks -- the exact shape
                   of prompt_test.txt lines 1-19)
  - feedback     : prompt_test.txt line 21 (gym PT feedback, 5 text roles)
  - grand_opening: generate_distill_4cases.py CASES_CONFIG["case1_burger_opening"]
  - recruitment  : generate_distill_4cases.py CASES_CONFIG["case3_recruitment"]
  - menu         : generate_distill_4cases.py CASES_CONFIG["case4_restaurant_menu"]

USAGE:
  python scripts/test_hero_selector.py --model Qwen/Qwen2.5-VL-7B-Instruct
  python scripts/test_hero_selector.py --model Qwen/Qwen2.5-VL-7B-Instruct --case product_ad

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
    "product_ad": {
        "category_hint": "product_ad (khuyến mại / ưu đãi / giới thiệu sản phẩm)",
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

HERO_SELECTOR_SYSTEM_PROMPT = """Bạn là chuyên gia thiết kế poster thương mại, đóng vai trò TẦNG 1 (Blueprint)
của một pipeline sinh poster gồm 4 tầng, theo đúng kiến trúc "100%-overlay" (không có mô hình
diffusion nào vẽ chữ, kể cả tiêu đề chính):
  Tầng 1 (BẠN): phân tích yêu cầu tự do của người dùng -> xuất blueprint JSON có cấu trúc.
  Tầng 2: một mô hình diffusion (FLUX.2) vẽ THUẦN ẢNH NỀN + SẢN PHẨM, KHÔNG VẼ BẤT KỲ CHỮ NÀO
          CẢ (kể cả tiêu đề) -- text-to-image thuần tuý theo đúng "background_prompt" bạn viết.
  Tầng 3: xác định vùng đặt chữ an toàn (né sản phẩm) trên ảnh Tầng 2 vừa sinh ra.
  Tầng 4: engine HTML/CSS (src/tendoo/typography_engine.py) render 100% MỌI khối chữ -- từ tiêu
          đề chính tới badge, CTA, rating, thông tin liên hệ -- đè lên ảnh Tầng 2. Không có khái
          niệm "hero vẽ bằng diffusion" nữa: MỌI chữ đều là HTML/CSS render chính xác tuyệt đối.

NHIỆM VỤ CỦA BẠN:
1. Đọc yêu cầu người dùng (có thể sơ sài hoặc rất chi tiết, thuộc 1 trong các loại: giới thiệu sản
   phẩm/khuyến mại/ưu đãi (category "product_ad"), banner khai trương ("grand_opening"), feedback
   khách hàng ("feedback"), tin tuyển dụng ("recruitment"), menu món ("menu"), hoặc tự do khác
   không khớp loại nào ("generic")).
2. Viết "background_prompt" mô tả THUẦN CẢNH/SẢN PHẨM/ÁNH SÁNG/BỐ CỤC cho Tầng 2 -- KHÔNG được
   chứa bất kỳ chuỗi chữ nội dung nào (kể cả trong ngoặc kép, kể cả một phần của tiêu đề) -- vì
   nếu nhắc tới chữ trong prompt, mô hình sinh ảnh dễ bị cám dỗ tự vẽ vài nét chữ mờ/sai vào ảnh
   dù không có glyph reference nào dẫn dắt, phá hỏng đúng mục tiêu "ảnh nền hoàn toàn sạch chữ"
   của Tầng 2. Được phép mô tả VỊ TRÍ/KHÔNG GIAN nên chừa trống (ví dụ "chừa khoảng trống sạch
   phía trên cho tiêu đề") -- đó là chỉ dẫn bố cục, không phải nội dung chữ.
3. Chọn "style_theme" -- 1 từ khoá gợi ý phong cách chữ chủ đạo cho toàn bài, chọn theo tông màu/
   mood mô tả trong yêu cầu (vd: "neon" cho cảnh cyberpunk/đèn màu, "gold" cho sang trọng/kim
   loại, "metallic" cho công nghệ/bạc, "embossed" cho tối giản/khắc chìm, "pastel" cho tông màu
   pastel/dễ thương). Tầng 4 sẽ tự đo độ sáng/tối thật của ảnh Tầng 2 rồi mới quyết định dùng style
   sáng-trên-tối hay tối-trên-sáng -- style_theme chỉ là gợi ý chọn GIỮA CÁC STYLE CÙNG PHE, không
   phải quyết định cuối cùng.
4. Điền "template_brief" -- dữ liệu THẬT sẽ đổ trực tiếp vào layout HTML cố định của Tầng 4 (không
   qua model nào nữa), nên PHẢI dùng ĐÚNG BỘ KHOÁ theo "category" đã chọn bên dưới, không tự bịa
   thêm khoá lạ, không đổi tên khoá. Nếu yêu cầu người dùng có bao nhiêu mục (ví dụ 5 món ăn, 6 yêu
   cầu tuyển dụng...) thì liệt kê đủ bấy nhiêu -- KHÔNG giới hạn số lượng theo ví dụ dưới đây, ví
   dụ chỉ để minh hoạ hình dạng dữ liệu.

BỘ KHOÁ "template_brief" THEO TỪNG CATEGORY (* = bắt buộc, còn lại tuỳ chọn -- không điền cũng
được, Tầng 4 tự chọn giá trị/style hợp lý):
- product_ad: *title_text, *title_position, *subtitle_text, subtitle_position, title_style,
        subtitle_style, title_font, subtitle_font
        (position là 1 trong 9 giá trị: top-left, top-center, top-right, middle-left,
        middle-center, middle-right, bottom-left, bottom-center, bottom-right -- đọc đúng theo
        ngôn ngữ vị trí của yêu cầu gốc, ví dụ "góc trên bên trái"->top-left, "ở giữa"->
        middle-center. Nếu yêu cầu nói khối 2 nằm "phía dưới" khối 1 mà không chỉ rõ toạ độ khác,
        BỎ TRỐNG subtitle_position để nó tự xếp ngay dưới title, đừng gán cùng 1 zone với title.)
- grand_opening: *brand, *date_range, *badge_label, *badge_percent, *badge_sub, *address,
        *offer_desc, *cta_text
- feedback: *brand, *top_badge, *stars, *verified_label, *quote_text, *avatar_emoji,
        *customer_name, *customer_sub, *features (list các {"icon","text"}, SỐ LƯỢNG TUỲ THEO YÊU
        CẦU), *offer_title, *offer_desc, *cta_text
- recruitment: *company, *deadline, *pos_label, *salary, *requirements (list string, số lượng tuỳ
        ý), *benefits (list string, số lượng tuỳ ý), *contact_line1, *contact_email, *cta_text
- menu: *sub_brand, *tagline, *categories (list các {"title", "items": [{"name","price",
        "badge"(tuỳ chọn)}]}, SỐ DANH MỤC VÀ SỐ MÓN MỖI DANH MỤC TUỲ THEO YÊU CẦU NGƯỜI DÙNG,
        không cố định 2x2), *footer_note, *hotline
- generic (chỉ dùng khi không khớp category nào ở trên): brand, eyebrow, badge, rating_value,
        rating_count, specs (list string), sub_slogan, cta_text, hotline, website

Trả về DUY NHẤT 1 khối JSON hợp lệ theo đúng schema sau, không thêm giải thích:
{
  "category": "product_ad|grand_opening|feedback|recruitment|menu|generic",
  "canvas": {"width": <int>, "height": <int>, "aspect_ratio": "<vd 9:16>"},
  "background_prompt": "<mô tả cảnh/sản phẩm/ánh sáng/bố cục -- TUYỆT ĐỐI KHÔNG chứa chữ literal>",
  "style_theme": "<neon|gold|metallic|embossed|pastel|...>",
  "template_brief": { "<đúng bộ khoá theo category đã chọn ở trên, điền đủ số lượng mục thực tế>": "..." }
}
"""

# Ground truth of what typography_engine.py's PosterTemplateEngine actually reads per category --
# kept here (not just in the prompt) so validate_blueprint() can check the model didn't invent or
# drop keys, independent of whatever the LLM was told to do. A plain list = every key required
# (legacy shape, still used by the 4 card-heavy templates below); a {"required":[...],
# "optional":[...]} dict = product_ad's shape, where style/font keys are genuinely optional
# (PosterTemplateEngine._auto_pick_style covers their absence).
TEMPLATE_BRIEF_SCHEMAS: Dict[str, Any] = {
    "product_ad": {
        "required": ["title_text", "title_position", "subtitle_text"],
        "optional": ["subtitle_position", "title_style", "subtitle_style", "title_font", "subtitle_font"],
    },
    "grand_opening": ["brand", "date_range", "badge_label", "badge_percent", "badge_sub", "address", "offer_desc", "cta_text"],
    "feedback": ["brand", "top_badge", "stars", "verified_label", "quote_text", "avatar_emoji", "customer_name", "customer_sub", "features", "offer_title", "offer_desc", "cta_text"],
    "recruitment": ["company", "deadline", "pos_label", "salary", "requirements", "benefits", "contact_line1", "contact_email", "cta_text"],
    "menu": ["sub_brand", "tagline", "categories", "footer_note", "hotline"],
    "generic": ["brand", "eyebrow", "badge", "rating_value", "rating_count", "specs", "sub_slogan", "cta_text", "hotline", "website"],
}

USER_PROMPT_TEMPLATE = "Yêu cầu người dùng ({category_hint}):\n{request}"


# ==================================================================================================
# 3. VALIDATION (no-literal-text-in-background_prompt guard + template_brief schema guard)
# ==================================================================================================

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _extract_literal_texts(node: Any) -> List[str]:
    """Recursively collects every string leaf-value out of a template_brief (which nests lists
    of dicts for categories like feedback's `features` or menu's `categories`/`items`) -- used to
    check none of them leaked into background_prompt, regardless of how deep they're nested."""
    texts: List[str] = []
    if isinstance(node, str):
        texts.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            texts.extend(_extract_literal_texts(v))
    elif isinstance(node, list):
        for item in node:
            texts.extend(_extract_literal_texts(item))
    return texts


def validate_blueprint(blueprint: Dict[str, Any]) -> List[str]:
    """Returns a list of validation errors (empty list = clean)."""
    errors: List[str] = []

    required_top = ["category", "canvas", "background_prompt", "template_brief"]
    for key in required_top:
        if key not in blueprint:
            errors.append(f"Missing top-level key: {key}")
    if errors:
        return errors

    bg_prompt_norm = _normalize(blueprint["background_prompt"])
    template_brief = blueprint["template_brief"]
    category = blueprint.get("category", "")

    # No-literal-text guard: background_prompt must never contain any literal content string from
    # template_brief. Not a glyph-poisoning concern anymore (nothing is drawn as a glyph in the
    # 100%-overlay architecture) -- the risk now is Stage 2's text-free generation being tempted
    # to hallucinate garbled text just because the prompt mentions words it should only picture.
    if isinstance(template_brief, dict):
        for txt in _extract_literal_texts(template_brief):
            if not isinstance(txt, str):
                continue
            txt_norm = _normalize(txt)
            if len(txt_norm) >= 4 and txt_norm in bg_prompt_norm:
                errors.append(
                    f"[LITERAL TEXT LEAK] background_prompt contains content text: \"{txt}\" -- "
                    f"risks Stage 2 hallucinating garbled text into a background meant to be 100% text-free."
                )
    else:
        errors.append("template_brief is not a dict")

    # template_brief guard: must use the EXACT key set typography_engine.py's PosterTemplateEngine
    # actually reads for the chosen category -- catches the model inventing keys (silently ignored
    # -> missing content in the final poster) or dropping required ones (falls back to a
    # hardcoded default -- exactly the "still hardcoded" gap this schema exists to catch).
    schema = TEMPLATE_BRIEF_SCHEMAS.get(category)
    if schema is None:
        errors.append(f"[WARNING] category='{category}' has no known template_brief schema -- verify this is intentional (new category not yet in TEMPLATE_BRIEF_SCHEMAS).")
    else:
        if isinstance(schema, dict):
            required_keys, optional_keys = schema.get("required", []), schema.get("optional", [])
        else:
            required_keys, optional_keys = schema, []
        actual_keys = set(template_brief.keys()) if isinstance(template_brief, dict) else set()
        missing = set(required_keys) - actual_keys
        unexpected = actual_keys - set(required_keys) - set(optional_keys)
        if missing:
            errors.append(f"template_brief missing keys for category='{category}': {sorted(missing)} -- these will silently fall back to hardcoded demo defaults.")
        if unexpected:
            errors.append(f"[WARNING] template_brief has keys PosterTemplateEngine won't read for category='{category}': {sorted(unexpected)} -- silently ignored, wasted model output.")

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
    brief = dict(blueprint.get("template_brief") or {})
    brief.setdefault("style_theme", blueprint.get("style_theme"))  # thread the top-level hint into brief

    html = PosterTemplateEngine.generate_html(
        analysis=analysis, brief=brief, background_image_path=str(bg_path), category=category
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
            bp = res["blueprint"]
            n_texts = len(_extract_literal_texts(bp.get("template_brief") or {}))
            print(f"  Category    : {bp.get('category', '???')}")
            print(f"  Style theme : {bp.get('style_theme', '(none)')}")
            print(f"  Text fields : {n_texts} literal string(s) in template_brief")
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
    print(f"{'CASE':<16} | {'CATEGORY':<14} | {'STYLE THEME':<12} | {'#TEXT FIELDS':<12} | {'ISSUES':<8}")
    print("-" * 100)
    for res in results:
        bp = res.get("blueprint") or {}
        category = bp.get("category", "(parse failed)")
        style_theme = bp.get("style_theme", "-")
        n_texts = len(_extract_literal_texts(bp.get("template_brief") or {}))
        n_issues = len(res["validation_errors"])
        print(f"{res['case_id']:<16} | {category:<14} | {style_theme:<12} | {n_texts:<12} | {n_issues:<8}")
    print("=" * 100)
    print(f"\n[✓] Results saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
