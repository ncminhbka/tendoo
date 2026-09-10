"""
src/tendoo/engine/blocks.py

Mô hình Dữ Liệu Khối Chữ Thích Ứng (Adaptive Semantic Block) & Ánh Xạ 6 Danh Mục:
- Vạn vật là Khối (Everything is a Block): Mọi trường text (kể cả Store Info) đều là một AdaptiveBlock.
- Phân bổ mặc định tối ưu thị giác cho 6 Danh mục của Tendoo (promo, product_intro, opening, feedback, recruitment, guide).
- Thuật toán Content Fingerprint chống trùng lặp 100%: không bao giờ để một nội dung xuất hiện 2 lần trên cùng một poster.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Danh sách 11 Zones không gian được hỗ trợ
VALID_ZONES = [
    "top_bar",       # Toàn bộ thanh đỉnh (Header bar, Store info khi nhảy lên trên)
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
    "bottom_bar",    # Toàn bộ thanh đáy (Footer bar, Store info mặc định)
    # Lưu ý: "center" là vùng cấm Product Sanctuary, không cấp cho text tự động
]

# Scale và font ratio cho từng vai trò thị giác
ROLE_SCALE: Dict[str, Tuple[float, int, bool]] = {
    # role: (size_ratio_of_width, font_weight, is_uppercase)
    "hero": (0.068, 900, True),
    "subtitle": (0.030, 600, False),
    "badge": (0.024, 800, True),
    "body": (0.021, 500, False),
    "caption": (0.016, 400, False),
    "meta": (0.018, 500, False),
    "brand_bar": (0.016, 500, False),
    "step_list": (0.020, 600, False),
}


@dataclass
class AdaptiveBlock:
    """Đại diện cho một khối chữ độc lập trên poster."""
    text: str
    role: str = "body"
    zone: Optional[str] = None
    field: Optional[str] = None
    style_variant: str = "default"
    icon: Optional[str] = None
    color: Optional[str] = None
    step_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "text": self.text,
            "role": self.role,
            "zone": self.zone,
            "field": self.field,
            "style_variant": self.style_variant,
        }
        if self.icon:
            d["icon"] = self.icon
        if self.color:
            d["color"] = self.color
        if self.step_index is not None:
            d["step_index"] = self.step_index
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveBlock:
        return cls(
            text=str(data.get("text", "")).strip(),
            role=data.get("role", "body"),
            zone=data.get("zone"),
            field=data.get("field"),
            style_variant=data.get("style_variant", "default"),
            icon=data.get("icon"),
            color=data.get("color"),
            step_index=data.get("step_index"),
        )


def normalize_fingerprint(text: str) -> str:
    """Tạo chuỗi vân tay chuẩn hóa để so khớp trùng lặp nội dung."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def deduplicate_blocks(
    blocks: List[AdaptiveBlock],
    hero_title: Optional[str] = None,
    similarity_threshold: float = 0.88,
) -> List[AdaptiveBlock]:
    """
    Thuật toán Content Fingerprint khử trùng lặp 100%:
    - Đảm bảo mỗi nội dung (kể cả hero_title) xuất hiện CHÍNH XÁC 1 LẦN.
    - Lần xuất hiện đầu tiên (ưu tiên role='hero' nếu có) được giữ lại.
    - Mọi lần xuất hiện tiếp theo (nhắc lại ở body/subtitle/caption) bị loại bỏ hoàn toàn.
    """
    seen_fingerprints: List[str] = []
    result: List[AdaptiveBlock] = []

    # Sắp xếp để role hero được duyệt đầu tiên
    sorted_blocks = sorted(blocks, key=lambda b: 0 if b.role == "hero" else 1)

    for b in sorted_blocks:
        text = b.text.strip()
        if not text:
            continue
        fp = normalize_fingerprint(text)
        if not fp:
            continue

        # Kiểm tra trùng với các nội dung đã duyệt qua
        is_dup = False
        for s_fp in seen_fingerprints:
            if fp == s_fp:
                is_dup = True
                break
            if difflib.SequenceMatcher(None, fp, s_fp).ratio() >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            result.append(b)
            seen_fingerprints.append(fp)

    return result


def map_category_to_default_blocks(
    category: str,
    title: Optional[str],
    fields: Dict[str, str],
    store_info: Optional[Dict[str, str]] = None,
    spatial_override_zone: Optional[str] = None,
) -> List[AdaptiveBlock]:
    """
    Ánh xạ ngữ nghĩa mặc định từ form sang các AdaptiveBlock cho 6 Danh mục:
    1. promo (Khuyến mại)
    2. product_intro (Giới thiệu sản phẩm)
    3. opening (Khai trương)
    4. feedback (Đánh giá khách hàng)
    5. recruitment (Tuyển dụng)
    6. guide (Quy trình hướng dẫn)
    """
    cat = (category or "promo").lower().strip()
    blocks: List[AdaptiveBlock] = []

    # 1. Tiêu đề Hero
    if title and title.strip():
        blocks.append(AdaptiveBlock(
            text=title.strip(),
            role="hero",
            zone="top_center" if cat in ("promo", "opening", "guide", "feedback") else "top_left",
            field="title",
        ))

    # 2. Thông tin chung theo từng danh mục
    if cat == "product_intro":
        name = fields.get("product_name", "")
        if name:
            if not title:
                blocks.append(AdaptiveBlock(text=name, role="hero", zone="top_left", field="product_name"))
            else:
                blocks.append(AdaptiveBlock(text=name, role="subtitle", zone="top_left", field="product_name"))
        price = fields.get("price", "")
        if price:
            blocks.append(AdaptiveBlock(text=price, role="badge", zone="top_right", field="price", icon="tag"))
        desc = fields.get("product_desc", "")
        if desc:
            blocks.append(AdaptiveBlock(text=desc, role="body", zone="middle_left", field="product_desc"))
        hl = fields.get("highlights", "")
        if hl:
            blocks.append(AdaptiveBlock(text=hl, role="caption", zone="middle_left", field="highlights", icon="check"))

    elif cat == "opening":
        o_date = fields.get("opening_date", "")
        if o_date:
            blocks.append(AdaptiveBlock(text=o_date, role="badge", zone="middle_right", field="opening_date", icon="calendar"))
        promo = fields.get("opening_promo", "")
        if promo:
            blocks.append(AdaptiveBlock(text=promo, role="subtitle", zone="top_center", field="opening_promo"))
        contact = fields.get("booking_contact", "")
        if contact:
            blocks.append(AdaptiveBlock(text=contact, role="meta", zone="bottom_left", field="booking_contact", icon="phone"))

    elif cat == "feedback":
        target = fields.get("feedback_target", "")
        if target:
            if not title:
                blocks.append(AdaptiveBlock(text=target, role="hero", zone="top_center", field="feedback_target"))
            else:
                blocks.append(AdaptiveBlock(text=target, role="subtitle", zone="top_center", field="feedback_target"))
        quote = fields.get("feedback_quote", "")
        if quote:
            blocks.append(AdaptiveBlock(text=f'“{quote}”', role="body", zone="middle_left", field="feedback_quote", style_variant="quote"))
        rating = fields.get("feedback_rating", "5")
        if rating:
            blocks.append(AdaptiveBlock(text=f"{rating} ★★★★★", role="caption", zone="middle_left", field="feedback_rating", icon="star"))
        offer = fields.get("special_offer", "")
        if offer:
            blocks.append(AdaptiveBlock(text=offer, role="badge", zone="bottom_right", field="special_offer", icon="gift"))

    elif cat == "recruitment":
        pos = fields.get("job_position", "")
        if pos:
            blocks.append(AdaptiveBlock(text=pos, role="badge", zone="top_center", field="job_position"))
        desc = fields.get("job_desc", "")
        if desc:
            blocks.append(AdaptiveBlock(text=desc, role="body", zone="middle_left", field="job_desc"))
        deadline = fields.get("apply_deadline", "")
        method = fields.get("apply_method", "")
        meta_parts = []
        if deadline:
            meta_parts.append(f"Hạn: {deadline}")
        if method:
            meta_parts.append(f"Nộp: {method}")
        if meta_parts:
            blocks.append(AdaptiveBlock(text=" | ".join(meta_parts), role="meta", zone="bottom_right", field="apply_deadline", icon="clock"))

    elif cat == "guide":
        steps_raw = fields.get("guide_steps", "")
        if steps_raw:
            step_list = [s.strip() for s in steps_raw.split("|") if s.strip()]
            for idx, s in enumerate(step_list, 1):
                blocks.append(AdaptiveBlock(
                    text=s,
                    role="step_list",
                    zone="middle_left",
                    step_index=idx,
                    field=f"step_{idx}",
                ))

    else:
        # Default: promo
        discount = fields.get("discount", "") or fields.get("offer_main", "")
        if discount:
            blocks.append(AdaptiveBlock(text=discount, role="badge", zone="middle_right", field="discount", icon="tag"))
        applied = fields.get("applied_product", "") or fields.get("offer_sub", "")
        if applied:
            blocks.append(AdaptiveBlock(text=applied, role="subtitle", zone="top_center", field="applied_product"))
        d_start = fields.get("date_start", "")
        d_end = fields.get("date_end", "")
        dates = fields.get("dates", "")
        if not dates and (d_start or d_end):
            dates = f"Từ {d_start} - Đến {d_end}".strip(" -")
        if dates:
            blocks.append(AdaptiveBlock(text=dates, role="meta", zone="bottom_left", field="dates", icon="calendar"))

    # 3. Thông tin cửa hàng (Store Info)
    # Mặc định nằm ở bottom_bar. Nếu spatial_override_zone (ví dụ user đòi lên trên) -> đổi zone
    if store_info:
        store_parts = []
        brand = store_info.get("store_name") or store_info.get("brand")
        phone = store_info.get("phone") or store_info.get("hotline")
        addr = store_info.get("address")
        if brand:
            store_parts.append(brand)
        if phone:
            store_parts.append(f"Hotline: {phone}")
        if addr:
            store_parts.append(addr)

        if store_parts:
            store_zone = spatial_override_zone or "bottom_bar"
            blocks.append(AdaptiveBlock(
                text="  •  ".join(store_parts),
                role="brand_bar",
                zone=store_zone,
                field="store_info",
                icon="phone" if phone else None,
            ))

    return blocks
