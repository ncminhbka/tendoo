"""
src/tendoo/core/category_schema.py

Bảng Category -> Field -> Đặc tả thị giác (NGUỒN CHÂN LÝ DUY NHẤT):
====================================================================
Trước 2026-09-13, kiến thức "6 category thương mại có những field nào, mỗi field thuộc
vai trò/kích thước/kiểu dáng thị giác gì" tồn tại thành 2 bảng ĐỘC LẬP, không import lẫn
nhau:
  - `layouts/style_matcher.py`'s `CATEGORY_FIELD_SLOTS` (chỉ liệt kê tên field) +
    `DEFAULT_FIELD_ROLE` (field -> role ngữ nghĩa: badge/body/caption/subtitle).
  - `engine/blocks.py`'s `FIELD_BLOCK_SPECS` (field -> size/container/icon/style_variant),
    được thêm sau để vá bug `style_variant` bị mất trên nhánh có render_plan -- nhưng chỉ
    hợp nhất 2 nơi *gọi* trong `demo_server.py`, chưa hợp nhất 2 bảng *định nghĩa*.

2 bảng song song này liệt kê CÙNG 6 category, CÙNG tên field, viết tay 2 lần độc lập --
chính đây là dạng lỗi kiến trúc đã từng gây ra bug thật (style_variant bị mất trên 1
nhánh). File này gộp lại thành đúng 1 bảng `CATEGORY_SCHEMA`, và `CATEGORY_FIELD_SLOTS`/
`DEFAULT_FIELD_ROLE`/`get_field_block_spec()` (tên hàm/hằng giữ nguyên để mọi call site cũ
không đổi) đều dẫn xuất (derive) từ bảng chính này, không còn viết tay 2 lần.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FieldSpec:
    """Đặc tả thị giác đầy đủ cho 1 field của 1 category.

    `role`: vai trò ngữ nghĩa cũ (badge/body/caption/subtitle) -- vẫn được
    `llm_render_plan_server.py` dùng làm gợi ý ngữ cảnh cho model, tách biệt với
    `size`/`container`/`style_variant` (quyết định thị giác thật, do Python áp đặt, model
    không có quyền quyết định cho field bắt buộc -- xem module docstring của
    `llm_render_plan_server.py`).
    """
    role: str = "body"
    size: str = "medium"
    container: str = "card"
    icon: Optional[str] = None
    style_variant: Optional[str] = None
    # False cho các pseudo-field KHÔNG tương ứng 1 field form thật (vd "dates" = tổng hợp
    # date_start+date_end, chỉ dùng nội bộ bởi map_category_to_default_blocks()) -- loại
    # khỏi CATEGORY_FIELD_SLOTS dẫn xuất bên dưới, giữ đúng hành vi gốc (CATEGORY_FIELD_SLOTS
    # chỉ liệt kê field form thật, được demo_server.py lặp qua để đọc `getattr(req, field)`).
    is_form_field: bool = True


# Bảng chính: category -> field (theo đúng thứ tự hiển thị) -> FieldSpec.
CATEGORY_SCHEMA: Dict[str, Dict[str, FieldSpec]] = {
    "promo": {
        "discount": FieldSpec(role="badge", size="large", container="none"),
        "applied_product": FieldSpec(role="body", size="medium", container="pill", style_variant="pill_badge"),
        "date_start": FieldSpec(role="caption", size="small", container="card", icon="calendar", style_variant="calendar_box"),
        "date_end": FieldSpec(role="caption", size="small", container="card", icon="calendar", style_variant="calendar_box"),
        "dates": FieldSpec(role="caption", size="small", container="card", icon="calendar", style_variant="calendar_box", is_form_field=False),
    },
    "product_intro": {
        "product_name": FieldSpec(role="subtitle", size="large", container="none"),
        "price": FieldSpec(role="badge", size="medium", container="pill", icon="tag", style_variant="luxury_tag"),
        "product_desc": FieldSpec(role="body", size="medium", container="card"),
        "highlights": FieldSpec(role="body", size="small", container="card", icon="check"),
    },
    "opening": {
        "opening_date": FieldSpec(role="badge", size="medium", container="pill", icon="calendar", style_variant="calendar_box"),
        "opening_promo": FieldSpec(role="body", size="large", container="none"),
        "booking_contact": FieldSpec(role="caption", size="small", container="card", icon="phone"),
    },
    "feedback": {
        "feedback_target": FieldSpec(role="subtitle", size="large", container="none"),
        "feedback_quote": FieldSpec(role="body", size="medium", container="card", style_variant="quote"),
        "feedback_rating": FieldSpec(role="badge", size="small", container="pill", icon="star", style_variant="rating_badge"),
        "special_offer": FieldSpec(role="body", size="medium", container="button", icon="gift", style_variant="cta_button"),
    },
    "recruitment": {
        "job_position": FieldSpec(role="subtitle", size="large", container="none"),
        "job_desc": FieldSpec(role="body", size="medium", container="card"),
        "apply_deadline": FieldSpec(role="caption", size="small", container="card", icon="clock", style_variant="calendar_box"),
        "apply_method": FieldSpec(role="caption", size="small", container="card", icon="clock", style_variant="calendar_box"),
    },
    "guide": {
        # guide_steps được dựng bằng vòng lặp step_index riêng (đánh số 01/02/03...) ở cả
        # engine/blocks.py::map_category_to_default_blocks() lẫn demo_server.py's
        # render_plan branch -- KHÔNG đi qua get_field_block_spec(). Spec dưới đây chỉ để
        # tài liệu hoá đúng những gì 2 nơi đó thực sự render cho mỗi bước, không được tra
        # cứu trực tiếp.
        "guide_steps": FieldSpec(role="body", size="medium", container="card", style_variant="step_list"),
    },
}


def _to_spec_dict(spec: FieldSpec) -> Dict[str, str]:
    """Chuyển FieldSpec thành plain dict chỉ gồm size/container/icon/style_variant (bỏ
    `role`) -- đúng hình dạng mà `get_field_block_spec()` đã trả về từ trước, để mọi caller
    hiện tại (engine/blocks.py, demo_server.py) không cần đổi gì."""
    d: Dict[str, str] = {"size": spec.size, "container": spec.container}
    if spec.icon:
        d["icon"] = spec.icon
    if spec.style_variant:
        d["style_variant"] = spec.style_variant
    return d


def get_field_block_spec(category: str, field: str) -> Dict[str, str]:
    """Tra cứu đặc tả thị giác mặc định (size/container/icon/style_variant) cho 1 field của
    1 category, dùng chung bởi `map_category_to_default_blocks()` (nhánh tất định) và
    `run_pipeline_inference()`'s render_plan branch. Trả về dict rỗng nếu category/field
    không có đặc tả (caller tự áp dụng fallback hợp lý của riêng mình, ví dụ
    size="medium"/container="card")."""
    spec = CATEGORY_SCHEMA.get((category or "").lower().strip(), {}).get(field)
    return _to_spec_dict(spec) if spec is not None else {}


def get_field_role(field: str) -> str:
    """Tra cứu vai trò ngữ nghĩa (badge/body/caption/subtitle) của 1 field, không phân biệt
    category (tên field không trùng giữa các category) -- dùng bởi
    `llm_render_plan_server.py` làm ngữ cảnh gợi ý cho model. Mặc định "body" nếu field lạ."""
    return DEFAULT_FIELD_ROLE.get(field, "body")


# Dẫn xuất (derive) 2 hằng số cũ từ CATEGORY_SCHEMA -- KHÔNG viết tay lần thứ 2 nữa.
CATEGORY_FIELD_SLOTS: Dict[str, List[str]] = {
    cat: [f for f, spec in fields.items() if spec.is_form_field]
    for cat, fields in CATEGORY_SCHEMA.items()
}
DEFAULT_FIELD_ROLE: Dict[str, str] = {
    field: spec.role for fields in CATEGORY_SCHEMA.values() for field, spec in fields.items()
}


__all__ = [
    "CATEGORY_FIELD_SLOTS",
    "CATEGORY_SCHEMA",
    "DEFAULT_FIELD_ROLE",
    "FieldSpec",
    "get_field_block_spec",
    "get_field_role",
]
