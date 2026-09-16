"""
src/tendoo/render_plan.py

Phase C Render-Plan schema, prompt construction, JSON parsing & validation --
pure functions, ZERO GPU/model/FastAPI dependency (2026-09-13, tách ra từ
`llm_render_plan_server.py` để tách rõ "logic thuần có thể unit-test không cần GPU" khỏi
"model loading + HTTP routing", đúng tinh thần "dễ đọc dễ bảo trì" -- xem
`llm_render_plan_server.py`'s module docstring cho toàn bộ quy tắc thiết kế Phase C (đặc
biệt: KHÔNG còn khái niệm "title" riêng cho model quyết định).

`llm_render_plan_server.py` (nạp Qwen3, chạy FastAPI sidecar) import module này cho toàn
bộ phần logic; `tests/test_render_plan.py` test module này trực tiếp, không cần mock model.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tendoo.core.category_schema import CATEGORY_FIELD_SLOTS
from tendoo.core.components import ICON_SVG_BY_NAME
from tendoo.core.fonts import FONT_CATALOG
from tendoo.engine.blocks import SIZE_SCALE, VALID_EFFECTS
from tendoo.engine.geometry import ZONE_NAMES

VALID_SIZES = set(SIZE_SCALE.keys())
# "qr" deliberately excluded here (unlike engine.blocks.VALID_CONTAINERS): the render-plan
# LLM never authors a QR block's container directly -- QR position comes through a
# dedicated field="qr" + zone extra_block (see validate_render_plan below), not container.
VALID_CONTAINERS = {"none", "button", "pill", "card"}
VALID_ICONS = set(ICON_SVG_BY_NAME.keys())
VALID_FONT_KEYS = set(FONT_CATALOG.keys()) | {"auto"}
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
# How similar (0..1, difflib ratio) a `field: null` block's text must be to an
# already-filled mandatory field's real value before it's treated as an accidental
# restatement and dropped -- see validate_render_plan()'s duplicate filter.
DUPLICATE_SIMILARITY_THRESHOLD = 0.72

__all__ = [
    "CATEGORY_FIELD_SLOTS",
    "DUPLICATE_SIMILARITY_THRESHOLD",
    "SYSTEM_PROMPT_TEMPLATE",
    "USER_PROMPT_TEMPLATE",
    "VALID_CONTAINERS",
    "VALID_FONT_KEYS",
    "VALID_ICONS",
    "VALID_SIZES",
    "build_messages",
    "is_render_plan_usable",
    "parse_render_plan_json",
    "validate_render_plan",
]


# ==================================================================================
# Prompt construction
# ==================================================================================

SYSTEM_PROMPT_TEMPLATE = """Bạn là bộ não dàn trang (layout brain) cho một hệ thống sinh poster thương mại. Nhiệm vụ CHỈ \
gồm 2 việc: (1) liệt kê các đoạn chữ PHỤ THÊM mà prompt tự do yêu cầu (nếu có, bao gồm cả 1 dòng tiêu đề/chữ lớn nổi \
bật nếu prompt muốn vậy), (2) viết prompt mô tả nền cho mô hình diffusion. Bạn KHÔNG quyết định bố cục (layout) -- \
việc đó do code xử lý, và KHÔNG có khái niệm "tiêu đề" riêng -- tiêu đề chỉ là 1 khối chữ cỡ xlarge/large như bất kỳ \
khối chữ nào khác.

QUY TẮC TUYỆT ĐỐI (không tự suy diễn khác):
- CÁC TRƯỜNG THÔNG TIN đã liệt kê bên dưới (dù trống hay đã điền) là NGỮ CẢNH tham khảo, không phải để bạn viết lại
  nội dung của chúng -- nếu trường đã điền, nó SẼ được vẽ nguyên văn giá trị thật (không qua bạn). Bạn CHỈ nêu ý kiến
  về các trường này qua "extra_blocks" khi: (i) trường đang TRỐNG và prompt tự do có mô tả rõ nội dung cho nó (lúc đó
  gắn "field" đúng tên trường + "text" là nội dung bạn trích được), hoặc (ii) người dùng yêu cầu VỊ TRÍ cụ thể cho nội
  dung của 1 trường (gắn "field" + "zone").
- TUYỆT ĐỐI KHÔNG tạo "extra_blocks" mới (field: null) cho nội dung TRÙNG Ý NGHĨA với 1 trường đã liệt kê (dù trường
  đó đang trống hay đã điền) -- nếu prompt tự do lặp lại/diễn giải lại đúng nội dung 1 trường, hãy gắn "field" tương
  ứng thay vì tạo khối chữ mới trùng lặp. Lỗi thật cần tránh: form đã điền "Từ ngày...đến ngày..." NHƯNG prompt cũng
  viết lại y hệt -- TUYỆT ĐỐI không tạo thêm 1 khối chữ "Từ ngày...đến ngày..." nữa, vì nó sẽ bị vẽ 2 LẦN.
- Chỉ dùng "extra_blocks" với field: null cho nội dung THỰC SỰ MỚI, không khớp bất kỳ trường nào đã liệt kê (ví dụ:
  1 dòng CTA phụ, 1 câu khẩu hiệu thêm mà form không có trường tương ứng, HOẶC 1 dòng tiêu đề/chữ lớn nổi bật mà
  prompt tự do nêu rõ muốn dùng THAY vì (hoặc thêm vào) tiêu đề đã điền -- dùng size="xlarge" hoặc "large",
  container="none" cho trường hợp này, giống hệt bất kỳ khối chữ lớn nào khác, KHÔNG có cơ chế "title" riêng).
- "zone" CHỈ điền khi người dùng yêu cầu RÕ RÀNG 1 vị trí cụ thể ("ở góc trên bên trái", "ở giữa", "phía dưới cùng"...)
  -- đừng tự ý gán zone nếu người dùng không yêu cầu vị trí.
- Mã QR: nếu người dùng yêu cầu vị trí cụ thể cho mã QR, tạo 1 block field="qr" kèm "zone" tương ứng (bỏ qua "text").
  Nếu không có yêu cầu, đừng tạo block này -- mã QR (nếu bật) sẽ tự hiển thị ở vị trí mặc định.
- MẶC ĐỊNH AN TOÀN là "extra_blocks": [] (mảng RỖNG). Chỉ thêm 1 phần tử khi có tín hiệu RÕ RÀNG trong prompt tự do
  (một vị trí cụ thể được nêu, hoặc một nội dung mới thực sự không khớp trường nào). Nếu không chắc prompt có yêu cầu vị trí/nội dung mới hay không -> ĐỪNG thêm block, để mảng rỗng.

QUAN TRỌNG -- CHỌN size THEO VAI TRÒ NỘI DUNG, không phải theo "muốn nó nổi bật":
- "xlarge": DUY NHẤT vai trò tiêu đề/thông điệp CHÍNH của cả poster -- 1 cụm từ NGẮN (tối đa ~5-6 chữ), luôn đi kèm
  container="none". Đây là khối chữ TO NHẤT trên poster, chỉ dùng khi nội dung THỰC SỰ là điểm nhấn số 1 (vd tên
  chương trình khuyến mãi, tiêu đề chính) -- không dùng cho nội dung phụ.
- "large": thông điệp PHỤ, hạng 2 sau tiêu đề chính -- vẫn 1 cụm từ ngắn, container="none". Dùng khi có 1 dòng quan
  trọng nhưng không phải điểm nhấn số 1 (vd tên vị trí tuyển dụng, tên sản phẩm).
- "medium": nội dung ở mức thông tin thường -- nút CTA (container="button"), badge giá/mã giảm giá (container="pill"),
  hoặc 1 đoạn mô tả/trích dẫn vài câu (container="card").
- "small": chi tiết phụ, ít quan trọng nhất -- ngày tháng, thông số, tag nhỏ, rating.
- Quy tắc chọn xlarge/large hay không: nếu nội dung là CẢ MỘT CÂU DÀI (trích feedback, mô tả sản phẩm...), TUYỆT ĐỐI
  KHÔNG gán xlarge/large cho nguyên câu đó (sẽ tràn khung vì render ở cỡ RẤT LỚN) -- hoặc rút gọn thành cụm từ ngắn thể
  hiện ý chính, hoặc dùng medium/small với container="card" thay vào đó.
- QUY TẮC NÀY LUÔN THẮNG YÊU CẦU NGƯỜI DÙNG, không có ngoại lệ: nếu người dùng yêu cầu "làm to hơn"/"làm nổi bật
  nhất"/"chiếm trọn tâm điểm" cho 1 câu dài, TUYỆT ĐỐI KHÔNG tăng size của cả câu đó lên xlarge/large để chiều theo
  đúng nghĩa đen -- vẫn giữ size="medium" (hoặc "small") với container="card", chỉ tăng độ nổi bật qua "effect"
  (vd "fire", "neon_cyan", "3d_gold")/"color", KHÔNG BAO GIỜ qua size/container. Đây là lỗi thật đã xảy ra: model
  từng làm xlarge nguyên 1 câu feedback dài ~30 chữ chỉ vì người dùng yêu cầu "làm to nhất có thể".
- Cách tự kiểm tra trước khi xuất JSON: ĐẾM số từ trong "text" của mỗi khối xlarge/large -- nếu > 6 từ, đó là lỗi,
  phải sửa lại (rút ngắn HOẶC đổi size="medium"+container="card") trước khi trả lời.

QUAN TRỌNG -- "button"/"pill" KHÔNG BAO GIỜ tự xuống dòng (luôn hiển thị 1 dòng duy nhất, phần thừa sẽ bị cắt thành
"...") -- CHỈ container="card" mới tự động xuống dòng khi nội dung dài. Vì vậy, bất kể size nào (medium hay small),
nội dung gán container="button"/"pill" LUÔN phải là 1 cụm từ NGẮN (tối đa ~4-5 chữ, giống mức độ ngắn gọn của
xlarge/large) -- ví dụ "Nhanh tay!", "Số lượng có hạn", "Ưu đãi hôm nay". TUYỆT ĐỐI không gán 1 câu dài cho
button/pill (vd "Số lượng có hạn, nhanh tay lên kẻo hết ưu đãi hấp dẫn này") -- nếu nội dung dài, dùng
container="card" (cho phép xuống dòng) thay vì button/pill. Cách tự kiểm tra: ĐẾM số từ trong "text" của mỗi khối
container="button"/"pill" -- nếu > 5 từ, đó là lỗi, phải đổi container="card" trước khi trả lời.

VÍ DỤ (câu feedback dài, người dùng yêu cầu "làm nổi bật nhất có thể"):
- SAI: {{"field": null, "text": "Tôi đã thử rất nhiều nơi nhưng chưa ở đâu chuyên nghiệp như ở đây", "size": "xlarge", "container": "none"}}  (>6 từ ở xlarge -- sẽ tràn khung)
- ĐÚNG: {{"field": null, "text": "Tôi đã thử rất nhiều nơi nhưng chưa ở đâu chuyên nghiệp như ở đây", "size": "medium", "container": "card", "effect": "shadow"}}

VÙNG ĐẶT CHỮ HỢP LỆ (zone, đúng 9 tên sau): {zones}
QUY MÔ CHỮ (size): {sizes} (xlarge: tiêu đề cực lớn; large: thông điệp lớn 2; medium: vừa, nút bấm, card; small: nhỏ, tag, ngày tháng)
BAO BỌC CHỮ (container): {containers} (none: chữ thuần không hộp cho xlarge/large; button: nút bấm CTA, LUÔN 1 dòng
không xuống dòng; pill: viên thuốc badge, LUÔN 1 dòng không xuống dòng; card: thẻ kính mờ, DUY NHẤT loại tự xuống dòng
khi nội dung dài)
HIỆU ỨNG ÁNH SÁNG CHO CHỮ (effect): {effects}
ICON TÙY CHỌN HỢP LỆ (icon): {icons} -- CHỈ dùng ĐÚNG 1 giá trị có trong danh sách này; nếu không có icon phù hợp,
BỎ HẲN key "icon" khỏi block đó (đừng viết icon="none" hay tự đặt tên icon khác không có trong danh sách).
FONT (font_key): "auto" là lựa chọn an toàn nhất trừ khi người dùng nêu rõ 1 font cụ thể trong danh sách: {fonts}.
STYLE_HINT: "auto" là lựa chọn an toàn nhất trừ khi người dùng mô tả rõ ràng 1 tông màu/ánh sáng cụ thể.
HEADLINE_EFFECT: "auto" là an toàn nhất hoặc chọn đúng 1 trong: auto, 3d_gold, neon_cyan, neon_pink, chrome, fire, shadow, plain.

QUAN TRỌNG -- scene_prompt (mô tả nền cho mô hình diffusion) TUYỆT ĐỐI KHÔNG được chứa bất kỳ chuỗi chữ nội dung nào
(không trích dẫn tiêu đề, không viết chữ sẽ hiện trên poster) -- toàn bộ chữ được vẽ riêng bằng HTML/CSS, không phải
bởi mô hình diffusion. Viết scene_prompt bằng tiếng Anh, mô tả thuần cảnh/sản phẩm/ánh sáng, không nhắc đến khái niệm
"text"/"chữ"/"title" dưới bất kỳ hình thức nào.

Khi "field" hoặc "zone" không khớp gì, dùng giá trị JSON null THẬT (không phải chuỗi ký tự "null" -- viết field: null,
KHÔNG viết field: "null").

CHỈ xuất ra DUY NHẤT một khối JSON hợp lệ, không kèm giải thích, đúng khuôn dạng sau:
{{"extra_blocks": [{{"field": "ten_truong hoặc null", "text": "...", "zone": "ten_zone hoặc null", "size": "xlarge|large|medium|small", "container": "none|button|pill|card", "effect": "neon_cyan|neon_pink|3d_gold|...", "icon": "..."}}], \
"scene_prompt": "...", "style_hint": "...", "font_key": "...", "headline_effect": "..."}}
"""

USER_PROMPT_TEMPLATE = """Danh mục (category): {category}

CÁC TRƯỜNG THÔNG TIN (tên trường: giá trị hiện tại, "(trống)" nếu chưa điền):
{category_fields_block}

PROMPT TỰ DO (prompt ảnh): {prompt}

Gợi ý phong cách người dùng chọn (style_pref, chỉ là gợi ý, có thể điều chỉnh theo prompt tự do): {style_pref}
"""


def _format_category_fields(category_fields: Dict[str, str]) -> str:
    if not category_fields:
        return "(danh mục này không có trường nào)"
    return "\n".join(f"- {k}: {v if v else '(trống)'}" for k, v in category_fields.items())


def build_messages(
    category: str,
    category_fields: Dict[str, str],
    prompt: str,
    style_pref: str,
) -> List[Dict[str, str]]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        zones=", ".join(ZONE_NAMES),
        sizes=", ".join(sorted(VALID_SIZES)),
        containers=", ".join(sorted(VALID_CONTAINERS)),
        effects=", ".join(sorted(VALID_EFFECTS)),
        icons=", ".join(sorted(VALID_ICONS)),
        fonts=", ".join(sorted(FONT_CATALOG.keys())),
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        category=category or "promo",
        category_fields_block=_format_category_fields(category_fields),
        prompt=prompt or "(không có)",
        style_pref=style_pref or "auto",
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ==================================================================================
# JSON extraction & parsing
# ==================================================================================

def _extract_first_json_object(raw_response: str) -> Optional[str]:
    """Scans for the first balanced {...} block via brace-depth counting, honoring
    string literals (so a `}` inside a quoted JSON string value never closes the
    object early) and backslash escapes within them.

    Replaces a naive `re.search(r"\\{.*\\}", ..., re.DOTALL)`, which is GREEDY: `.*`
    matches as much as possible, so if the model's raw output is
    `{"scene_prompt": ...} some trailing prose that happens to mention a brace: {ok}`
    the old regex captured from the FIRST `{` all the way to the LAST `}` in the whole
    response -- including that trailing prose -- producing either a hard JSONDecodeError
    (if the trailing text isn't valid JSON, the common case) or, worse, a silently wrong
    parse if it happened to still be valid-looking JSON. Chat models routinely emit
    trailing commentary after the JSON block despite instructions not to, so this isn't
    a theoretical risk. A balanced scanner starting from the first `{` and stopping at
    ITS matching `}` is immune to anything after that point."""
    start = raw_response.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(raw_response)):
        ch = raw_response[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw_response[start : i + 1]
    return None


def parse_render_plan_json(raw_response: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Extracts the first balanced {...} block from the model's raw text and
    JSON-decodes it. Never raises -- returns (None, [error]) on any parse failure."""
    candidate = _extract_first_json_object(raw_response)
    if candidate is None:
        return None, ["No JSON object found in model response."]
    try:
        return json.loads(candidate), []
    except json.JSONDecodeError as e:
        return None, [f"JSON parse failure: {e}"]


# ==================================================================================
# Validation -- pure functions, fully offline-testable (no GPU/model needed)
# ==================================================================================

def validate_render_plan(
    plan: Any,
    category: str,
    category_field_values: Dict[str, str],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Cleans and validates a raw (possibly malformed) render-plan dict from the model.

    `category_field_values`: the REAL current req values for CATEGORY_FIELD_SLOTS[category]
    (blank string if unfilled) -- used both to know which field names are valid for
    this category and to run the duplicate-content filter against already-filled ones.

    Never raises. Drops/corrects individual bad pieces rather than rejecting the whole
    plan outright (mirrors OmniLayout.render_html's own silent-drop tolerance for
    unknown zone/icon) -- every drop/correction is recorded in `errors` but doesn't by
    itself invalidate the plan. Returns (None, errors) only when the input isn't a dict
    at all.
    """
    errors: List[str] = []
    if not isinstance(plan, dict):
        return None, ["render plan is not a JSON object"]

    valid_fields = set(CATEGORY_FIELD_SLOTS.get(category, [])) | {"qr"}

    # --- extra_blocks ---
    raw_blocks = plan.get("extra_blocks")
    clean_blocks: List[Dict[str, Any]] = []
    if isinstance(raw_blocks, list):
        for i, b in enumerate(raw_blocks):
            if not isinstance(b, dict):
                errors.append(f"extra_blocks[{i}] is not an object, dropped")
                continue
            text = b.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"extra_blocks[{i}] has no non-empty 'text', dropped")
                continue
            field = b.get("field")
            if field is not None and field not in valid_fields:
                errors.append(f"extra_blocks[{i}] has unknown field '{field}' for category '{category}', treated as null")
                field = None
            zone = b.get("zone")
            if zone is not None and zone not in ZONE_NAMES:
                errors.append(f"extra_blocks[{i}] has unknown zone '{zone}', dropped (block kept)")
                zone = None
            size = b.get("size") if b.get("size") in VALID_SIZES else "medium"
            if b.get("size") is not None and b.get("size") not in VALID_SIZES:
                errors.append(f"extra_blocks[{i}] has unknown size '{b.get('size')}', defaulted to 'medium'")
            container = b.get("container") if b.get("container") in VALID_CONTAINERS else "none"
            if b.get("container") is not None and b.get("container") not in VALID_CONTAINERS:
                errors.append(f"extra_blocks[{i}] has unknown container '{b.get('container')}', defaulted to 'none'")

            # --- length-discipline safety net (2026-09-14) ---
            # A real-GPU capability probe (scripts/verify_render_plan_llm.py) against a
            # live Qwen3-4B-FP8 sidecar found the model follows SYSTEM_PROMPT_TEMPLATE's
            # "keep xlarge/large/button/pill text SHORT" instruction only ~40% of the
            # time across 28 real cases (17 violations) -- including an adversarial case
            # where the user's own request ("làm to nhất có thể") outweighed the system
            # prompt's own length rule. Prompt wording alone cannot be trusted to prevent
            # this reliably (the exact CSS-padding/ellipsis-truncation defects found and
            # fixed earlier this same session). Mirroring how an invalid size/container/
            # icon/font_key is already silently corrected above rather than trusted
            # blindly, apply the same graceful-degradation treatment to length: "card" is
            # the one container that actually wraps to multiple lines (see geometry.py's
            # `allow_multiline = container in ("card", "none")`), so downgrading into it
            # is always safe regardless of how long the text turns out to be.
            word_count = len(text.split())
            if container in ("button", "pill") and word_count > 5:
                errors.append(
                    f"extra_blocks[{i}] text is {word_count} words but container='{container}' "
                    f"never wraps (renders as a single-line chip) -- downgraded to container='card' "
                    f"to avoid ellipsis/tiny-font truncation at render time"
                )
                container = "card"
            if size in ("xlarge", "large") and word_count > 6:
                downgrade_note = f"extra_blocks[{i}] text is {word_count} words but size='{size}' is meant for a short hero phrase -- downgraded to size='medium'"
                size = "medium"
                if container == "none":
                    container = "card"
                    downgrade_note += ", container='card'"
                errors.append(downgrade_note + " to avoid render-time overflow")

            effect = b.get("effect") if b.get("effect") in VALID_EFFECTS else None
            clean_block: Dict[str, Any] = {
                "field": field,
                "text": text.strip(),
                "zone": zone,
                "size": size,
                "container": container,
            }
            if effect:
                clean_block["effect"] = effect
            icon = b.get("icon")
            if icon:
                if icon in VALID_ICONS:
                    clean_block["icon"] = icon
                else:
                    errors.append(f"extra_blocks[{i}] has unknown icon '{icon}', dropped")
            color = b.get("color")
            if color:
                if isinstance(color, str) and _HEX_COLOR_RE.match(color):
                    clean_block["color"] = color
                else:
                    errors.append(f"extra_blocks[{i}] has invalid color '{color}', dropped")
            clean_blocks.append(clean_block)
    elif raw_blocks is not None:
        errors.append("'extra_blocks' is not a list, ignored")

    # --- duplicate filter: drop field:null blocks that closely restate an
    # already-filled mandatory field's real value (the model failing to tag `field`
    # correctly per the system prompt's explicit instruction). Two heuristics, since a
    # restatement is often a short value embedded in a longer natural-language
    # sentence (whole-string ratio alone misses that): (1) the field's exact value
    # appears verbatim inside the block's text (or vice versa), (2) the two strings are
    # a close whole-string match (catches near-identical short blocks). ---
    filled_values = [v for v in category_field_values.values() if v and v.strip()]

    def _is_near_duplicate(text: str) -> bool:
        norm_text = text.lower()
        for fv in filled_values:
            norm_fv = fv.lower()
            if norm_fv in norm_text or norm_text in norm_fv:
                return True
            if difflib.SequenceMatcher(None, norm_text, norm_fv).ratio() >= DUPLICATE_SIMILARITY_THRESHOLD:
                return True
        return False

    deduped_blocks: List[Dict[str, Any]] = []
    for b in clean_blocks:
        if b["field"] is None and _is_near_duplicate(b["text"]):
            errors.append(
                f"extra_block text '{b['text'][:40]}...' looks like a near-duplicate "
                f"of an already-filled field, dropped"
            )
            continue
        deduped_blocks.append(b)

    # --- scene_prompt / style_hint / font_key ---
    scene_prompt = plan.get("scene_prompt")
    if not isinstance(scene_prompt, str) or not scene_prompt.strip():
        errors.append("missing/empty 'scene_prompt'")
        scene_prompt = ""
    else:
        scene_prompt = scene_prompt.strip()

    font_key = plan.get("font_key")
    if font_key not in VALID_FONT_KEYS:
        if font_key is not None:
            errors.append(f"unknown font_key '{font_key}', defaulted to 'auto'")
        font_key = "auto"

    style_hint = plan.get("style_hint")
    if not isinstance(style_hint, str) or not style_hint.strip():
        style_hint = "auto"
    # Style hint is passed through and re-validated downstream by detect_scene_lighting_tone
    # against the rich commercial styles supported by OmniBlockLayout.

    raw_effect = plan.get("headline_effect")
    headline_effect = "auto"
    if isinstance(raw_effect, str) and raw_effect.strip():
        try:
            from tendoo.core.typography import resolve_headline_effect
            resolved_effect, _, _ = resolve_headline_effect(effect=raw_effect.strip())
            headline_effect = resolved_effect
        except Exception:
            headline_effect = "auto"

    cleaned = {
        "extra_blocks": deduped_blocks,
        "scene_prompt": scene_prompt,
        "style_hint": style_hint,
        "font_key": font_key,
        "headline_effect": headline_effect,
    }
    return cleaned, errors


def is_render_plan_usable(plan: Optional[Dict[str, Any]]) -> bool:
    """A much lower bar than v1: a plan contributes value if it has a non-empty
    scene_prompt or any extra_blocks -- either alone is still worth using over the
    fully-deterministic fallback. A plan with NEITHER (e.g. total generation failure)
    is not worth threading through at all. (2026-09-13: dropped the "title decision"
    check -- there is no more dedicated title concept, see llm_render_plan_server.py's
    module docstring.)"""
    if not plan:
        return False
    return bool(plan.get("scene_prompt")) or bool(plan.get("extra_blocks"))
