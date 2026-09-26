"""
src/tendoo_v3/routing.py

Cổng 3 -- DUNG LƯỢNG (ROADMAP §2.4-2.5): LLM đề xuất template, Python phủ quyết bằng SỐ ĐO.

Đếm ký tự thật của plan, tra sức chứa ĐO ĐƯỢC của template ở đúng tỉ lệ khung hình
(`catalog.capacity_chars*`, scripts/calibrate_capacity.py). Vượt -> đổi sang template CÙNG
intent, hiển thị ĐỦ mọi field plan mang theo (không làm mất chữ), sức chứa lớn hơn. Không có
template nào chứa nổi -> giữ nguyên và ghi log: đó là LỖ HỔNG THẬT trên bản đồ phủ sóng (§3.4).

Ngưỡng dùng để định tuyến: xem `ROUTING_CAPACITY_KEY` (chọn theo số đo 3D, ROADMAP GĐ 3).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional, Tuple

from tendoo_v3.catalog import TEMPLATE_CATALOG, resolve_intent
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.validators import CONTENT_FIELDS, content_chars

logger = logging.getLogger(__name__)

ASPECTS = {"1:1": 1.0, "9:16": 9 / 16, "16:9": 16 / 9, "4:5": 4 / 5}


def nearest_aspect(aspect: str) -> str:
    """'3:4' -> '4:5' (tỉ lệ gần nhất trong 4 khung đã đo). Chuỗi hỏng -> '1:1'."""
    if aspect in ASPECTS:
        return aspect
    try:
        w, h = (float(x) for x in str(aspect).split(":"))
        r = w / h
    except (ValueError, ZeroDivisionError):
        return "1:1"
    return min(ASPECTS, key=lambda k: abs(ASPECTS[k] - r))


def _fits_all_fields(plan: TendooCreativePlan, template: str) -> bool:
    """Template hiển thị ĐỦ mọi field plan có, và plan có đủ field bắt buộc của template."""
    slots = TEMPLATE_CATALOG[template].get("slots", {})
    if any(getattr(plan, f, None) and f not in slots for f in CONTENT_FIELDS):
        return False
    return all(getattr(plan, f, None) for f, spec in slots.items() if spec.get("required"))


def capacity(template: str, aspect: str, key: str) -> int:
    return TEMPLATE_CATALOG[template].get(key, {}).get(nearest_aspect(aspect), 0)


def route_template(plan: TendooCreativePlan, aspect: str, key: Optional[str] = None) -> Tuple[TendooCreativePlan, Optional[str]]:
    """(plan -- có thể đã đổi template, lý do đổi / cảnh báo hoặc None)."""
    key = key or ROUTING_CAPACITY_KEY
    if plan.template not in TEMPLATE_CATALOG:
        return plan, None
    chars = content_chars(plan)
    intent = resolve_intent(plan.template, plan.visual_intent)
    # Phủ quyết MẤT CHỮ (GĐ 3R: LLM thật đặt `cta` vào l_frame_showcase vốn không có chỗ cho cta):
    # đổi sang template cùng intent hiển thị đủ mọi field, còn chứa nổi. Không có -> giữ, Cổng 2 báo.
    if not TEMPLATE_CATALOG[plan.template].get("specialized") and not _fits_all_fields(plan, plan.template):
        full = [t for t, info in TEMPLATE_CATALOG.items()
                if t != plan.template and not info.get("specialized") and intent in info.get("visual_intents", [])
                and _fits_all_fields(plan, t) and capacity(t, aspect, key) >= chars]
        if full:
            best = max(full, key=lambda t: (capacity(t, aspect, "capacity_chars") >= chars, capacity(t, aspect, key)))
            why = f"[Cổng 3] '{plan.template}' không hiển thị hết field của plan -> đổi sang '{best}' (cùng intent '{intent}', đủ chỗ)"
            logger.info(why)
            return replace(plan, template=best, orientation=None), why
    if chars <= capacity(plan.template, aspect, key):
        return plan, None
    if TEMPLATE_CATALOG[plan.template].get("specialized"):
        why = f"[Cổng 3] {chars} ký tự vượt sức chứa '{plan.template}' ({capacity(plan.template, aspect, key)}, {aspect}) -- template chuyên biệt, không đổi; LỖ HỔNG phủ sóng"
        logger.warning(why)
        return plan, why
    cands = [
        t for t, info in TEMPLATE_CATALOG.items()
        if t != plan.template and not info.get("specialized") and intent in info.get("visual_intents", []) and _fits_all_fields(plan, t)
        and capacity(t, aspect, key) >= chars
    ]
    if not cands:
        why = f"[Cổng 3] {chars} ký tự vượt sức chứa '{plan.template}' ({capacity(plan.template, aspect, key)}, {aspect}) và không template cùng intent '{intent}' nào chứa nổi -- LỖ HỔNG phủ sóng, giữ nguyên"
        logger.warning(why)
        return plan, why
    # Ưu tiên template còn vừa theo sức chứa THẨM MỸ, rồi tới dư sức chứa nhiều nhất.
    best = max(cands, key=lambda t: (capacity(t, aspect, "capacity_chars") >= chars, capacity(t, aspect, key)))
    why = f"[Cổng 3] {chars} ký tự vượt sức chứa '{plan.template}' ({capacity(plan.template, aspect, key)}, {aspect}) -> đổi sang '{best}' ({capacity(best, aspect, key)}), cùng intent '{intent}'"
    logger.info(why)
    # orientation mang nghĩa riêng từng template -> về mặc định của template mới.
    return replace(plan, template=best, orientation=None), why


# Chốt sau số đo 3D (xem ROADMAP GĐ 3): sức chứa AN TOÀN (không mất chữ).
ROUTING_CAPACITY_KEY = "capacity_chars_safe"

__all__ = ["ROUTING_CAPACITY_KEY", "capacity", "nearest_aspect", "route_template"]
