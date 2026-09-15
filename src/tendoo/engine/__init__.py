"""
src/tendoo/engine

Tendoo Omni-Block Engine -- Bộ Dàn Trang Khối Đa Năng Thích Ứng:
================================================================
- blocks: AdaptiveBlock, 6 category mappings & Content Fingerprint dedup
- geometry: Product Sanctuary & 3x3 Spatial Grid Bounding Box math
- mask: Unified Parametric Corridor Mask cho 2-branch velocity blending O(1)
- layout: OmniBlockLayout kế thừa BaseLayout (Master Engine duy nhất)

TẠI SAO CẦN PACKAGE NÀY TRONG KIẾN TRÚC TENDOO AI:
- Thay thế hoàn toàn cho 7 layouts tĩnh trước đây bằng 1 engine thích ứng duy nhất.
- Tự động nhận diện cấu trúc thông tin của từng ngành hàng (ẩm thực, công nghệ, thời trang, sự kiện...).
- Định vị chữ thông minh xung quanh Product Sanctuary bằng lưới 3x3, bảo vệ tuyệt đối chủ thể chính.
- Tối ưu hóa hiệu năng tính toán: sinh mask O(1) bất biến (~6s/ảnh trên 2x NVIDIA A30).
"""

from typing import Dict, List, Optional, Union

from tendoo.engine.blocks import (
    AdaptiveBlock,
    SIZE_SCALE,
    VALID_ZONES,
    deduplicate_blocks,
    map_category_to_default_blocks,
    normalize_fingerprint,
    normalize_zone,
)
from tendoo.engine.geometry import (
    GRID_ZONE_NAMES,
    OMNI_ZONES,
    PRODUCT_SANCTUARY_BY_RATIO,
    ZONE_DEFAULT_ALIGN,
    ZONE_GRID_AREA,
    ZONE_NAMES,
    ZONE_SELF_ALIGN,
    compute_block_metrics,
    estimate_zone_available_box,
    get_product_sanctuary_rect,
    get_zone_bounding_box,
)
from tendoo.engine.layout import OmniBlockLayout
from tendoo.engine.mask import (
    generate_omni_corridor_mask,
    generate_omni_corridor_mask_from_rects,
)

# (2026-09-13: get_layout()/list_layouts() chuyển từ `layouts/registry.py` -- nơi đó cần
# lazy-import OmniBlockLayout để phá vòng lặp import engine<->layouts giả tạo. Ở đây,
# OmniBlockLayout đã là import bình thường ngay phía trên, nên không còn cần lazy loading:
# package "engine" tự nhiên là nơi đúng cho 1 registry chỉ phục vụ đúng 1 layout nó chứa.)
_OMNI_INSTANCE: Optional[OmniBlockLayout] = None


def _get_omni_instance() -> OmniBlockLayout:
    """Khởi tạo và lưu cache duy nhất một instance của OmniBlockLayout."""
    global _OMNI_INSTANCE
    if _OMNI_INSTANCE is None:
        _OMNI_INSTANCE = OmniBlockLayout()
    return _OMNI_INSTANCE


def get_layout(name: Union[str, dict, None] = None) -> OmniBlockLayout:
    """
    Trả về OmniBlockLayout - động cơ dàn trang đa khối tự thích ứng duy nhất của Tendoo AI.

    TẠI SAO CẦN LÀM:
    - Toàn bộ các yêu cầu sinh poster trong hệ thống đều được phục vụ bởi OmniBlockLayout.
    - Không còn bất kỳ rủi ro vỡ template cứng hay lỗi tải thư viện phụ thuộc cũ.
    """
    return _get_omni_instance()


def list_layouts() -> List[Dict[str, str]]:
    """Trả về danh sách layout chính thức phục vụ hiển thị trên giao diện người dùng."""
    omni = _get_omni_instance()
    return [
        {
            "name": "omni",
            "display_name": omni.display_name,
            "description": omni.description,
        }
    ]


__all__ = [
    "AdaptiveBlock",
    "GRID_ZONE_NAMES",
    "OMNI_ZONES",
    "OmniBlockLayout",
    "PRODUCT_SANCTUARY_BY_RATIO",
    "SIZE_SCALE",
    "VALID_ZONES",
    "ZONE_DEFAULT_ALIGN",
    "ZONE_GRID_AREA",
    "ZONE_NAMES",
    "ZONE_SELF_ALIGN",
    "compute_block_metrics",
    "deduplicate_blocks",
    "estimate_zone_available_box",
    "generate_omni_corridor_mask",
    "generate_omni_corridor_mask_from_rects",
    "get_layout",
    "get_product_sanctuary_rect",
    "get_zone_bounding_box",
    "list_layouts",
    "map_category_to_default_blocks",
    "normalize_fingerprint",
    "normalize_zone",
]

