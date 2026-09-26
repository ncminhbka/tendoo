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
from tendoo_v3.catalog import COMPONENT_STYLES, INTENT_PROFILES, TEMPLATE_CATALOG, resolve_intent
from tendoo_v3.components import STAMP_FONT_MIN, split_stat, stamp_ring
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.styles import BACKGROUND_TONES, TEXT_EFFECT_ALIASES, TEXT_EFFECT_INTENTS, TEXT_EFFECTS

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

    if plan.visual_intent is not None:
        if plan.visual_intent not in INTENT_PROFILES:
            issues.append(f"visual_intent '{plan.visual_intent}' không có trong danh mục -- dùng intent mặc định của template")
        elif info is not None and plan.visual_intent not in info.get("visual_intents", []):
            issues.append(f"visual_intent '{plan.visual_intent}' không hợp template '{plan.template}' -- dùng intent mặc định")

    font = (plan.style.font or "auto").lower().strip()
    if font != "auto" and font not in FONT_CATALOG and font not in FONT_ALIASES:
        issues.append(f"font '{plan.style.font}' không có trong danh mục -- sẽ tự gợi ý font khác")

    effect = (plan.style.text_effect or "plain_elegant").lower().strip()
    effect = TEXT_EFFECT_ALIASES.get(effect, effect)
    if effect not in TEXT_EFFECTS:
        issues.append(f"text_effect '{plan.style.text_effect}' không có trong danh mục -- sẽ render như plain_elegant")
    elif effect in TEXT_EFFECT_INTENTS and resolve_intent(plan.template, plan.visual_intent) not in TEXT_EFFECT_INTENTS[effect]:
        issues.append(f"text_effect '{effect}' không hợp intent '{resolve_intent(plan.template, plan.visual_intent)}' (chỉ dùng cho {list(TEXT_EFFECT_INTENTS[effect])})")

    tone = (plan.style.background_tone or "dark_luxury").lower().strip()
    if tone not in BACKGROUND_TONES:
        issues.append(f"background_tone '{plan.style.background_tone}' không có trong danh mục -- sẽ xử lý như nền tối")
    issues += _check_components(plan)
    return issues


def _check_components(plan: TendooCreativePlan) -> List[str]:
    """Linh kiện GĐ 2: thuộc danh mục, hợp intent (Luật 4 §4.4), và có đủ nội dung để vẽ."""
    issues: List[str] = []
    intent = resolve_intent(plan.template, plan.visual_intent)
    for field, options in COMPONENT_STYLES.items():
        value = getattr(plan, field)
        if value is None:
            continue
        if value not in options:
            issues.append(f"{field} '{value}' không có trong danh mục {sorted(options)} -- dùng mặc định")
        elif intent not in options[value]:
            issues.append(f"{field} '{value}' không hợp intent '{intent}' (chỉ dùng cho {list(options[value])})")
    if plan.badge_style not in (None, "pill") and not plan.badge:
        issues.append(f"badge_style '{plan.badge_style}' nhưng plan không có badge -- bỏ qua")
    if plan.badge_style == "capsule" and plan.badge and "|" not in plan.badge:
        issues.append("badge_style 'capsule' cần badge dạng 'TRÁI | PHẢI' -- sẽ render pill")
    if plan.badge_style == "stamp" and plan.badge and stamp_ring(plan.badge)[1] < STAMP_FONT_MIN:
        issues.append(f"badge quá dài cho con dấu ({len(plan.badge)} ký tự) -- chữ vòng sẽ khó đọc")
    if plan.stat_style in ("unit", "burst"):
        stats = [p.get("t", "") for p in plan.hero_parts if p.get("role") == "stat"]
        if not stats:
            issues.append(f"stat_style '{plan.stat_style}' cần hero_parts có đoạn role=stat -- bỏ qua")
        elif not any(split_stat(t) for t in stats):
            issues.append(f"stat_style '{plan.stat_style}': stat {stats} không có dạng số+đơn vị -- không tách đơn vị")
        if plan.stat_style == "burst" and any(len(t) > 5 for t in stats):
            issues.append(f"stat_style 'burst' hợp với stat ngắn (<= 5 ký tự); {stats} sẽ kéo sao thành elip, cắt mép chữ")
    return issues


def log_plan_issues(plan: TendooCreativePlan) -> List[str]:
    issues = check_plan(plan)
    for issue in issues:
        logger.warning(f"[Cổng 2] {issue}")
    return issues


__all__ = ["CONTENT_FIELDS", "check_plan", "log_plan_issues"]
