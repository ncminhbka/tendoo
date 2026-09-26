"""
src/tendoo_v3/validators.py

Cổng 2 -- THÀNH VIÊN (ROADMAP §2.5): mọi lựa chọn trong plan phải thuộc danh mục đóng, và
mọi nội dung plan mang theo phải có chỗ hiển thị trên template đã chọn.

Cổng này CHỈ PHÁT HIỆN + GHI LOG, không sửa plan. Mỗi giá trị lạ đã có đường rơi về mặc
định an toàn ở đúng chỗ dùng nó (template -> llm_planner; font -> resolve_font gợi ý font;
hiệu ứng -> nhánh plain_elegant của get_effect_css; tông nền -> palette nền tối). Cố ý
KHÔNG chuẩn hoá tại đây: vd đổi tông lạ thành "dark_luxury" sẽ làm
renderer.compute_type_scale_ratio bắt được chữ "luxury" và âm thầm đổi thang cỡ chữ.
"""

from __future__ import annotations

import logging
from typing import List

from tendoo_core.fonts import FONT_ALIASES, FONT_CATALOG
from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.styles import BACKGROUND_TONES, TEXT_EFFECT_ALIASES, TEXT_EFFECTS

logger = logging.getLogger(__name__)

# Field nội dung người dùng nhìn thấy trên poster (không gồm style/prompt/markup).
CONTENT_FIELDS = (
    "hero", "subhead", "badge", "tag_left", "tag_right", "rating", "extra_texts",
    "cta", "store_info", "qr_code", "testimonial", "reviewer_name", "steps",
)


def check_plan(plan: TendooCreativePlan) -> List[str]:
    """Danh sách vấn đề (rỗng = qua cổng). Không raise, không sửa plan."""
    issues: List[str] = []
    info = TEMPLATE_CATALOG.get(plan.template)
    if info is None:
        issues.append(f"template '{plan.template}' không có trong catalog")
    else:
        slots = info.get("slots", {})
        dropped = [f for f in CONTENT_FIELDS if getattr(plan, f, None) and f not in slots]
        if dropped:
            issues.append(f"template '{plan.template}' KHÔNG hiển thị {dropped} -- nội dung này sẽ mất khỏi poster")
        missing = [f for f, spec in slots.items() if spec.get("required") and not getattr(plan, f, None)]
        if missing:
            issues.append(f"template '{plan.template}' thiếu field chuyên biệt {missing} -- poster có thể lệch bản chất template")

    font = (plan.style.font or "auto").lower().strip()
    if font != "auto" and font not in FONT_CATALOG and font not in FONT_ALIASES:
        issues.append(f"font '{plan.style.font}' không có trong danh mục -- sẽ tự gợi ý font khác")

    effect = (plan.style.text_effect or "plain_elegant").lower().strip()
    if TEXT_EFFECT_ALIASES.get(effect, effect) not in TEXT_EFFECTS:
        issues.append(f"text_effect '{plan.style.text_effect}' không có trong danh mục -- sẽ render như plain_elegant")

    tone = (plan.style.background_tone or "dark_luxury").lower().strip()
    if tone not in BACKGROUND_TONES:
        issues.append(f"background_tone '{plan.style.background_tone}' không có trong danh mục -- sẽ xử lý như nền tối")
    return issues


def log_plan_issues(plan: TendooCreativePlan) -> List[str]:
    issues = check_plan(plan)
    for issue in issues:
        logger.warning(f"[Cổng 2] {issue}")
    return issues


__all__ = ["CONTENT_FIELDS", "check_plan", "log_plan_issues"]
