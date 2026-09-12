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

from tendoo.engine.blocks import (
    AdaptiveBlock,
    ROLE_SCALE,
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
    get_product_sanctuary_rect,
    get_zone_bounding_box,
)
from tendoo.engine.layout import OmniBlockLayout
from tendoo.engine.mask import (
    generate_omni_corridor_mask,
    generate_omni_corridor_mask_from_rects,
)

__all__ = [
    "AdaptiveBlock",
    "GRID_ZONE_NAMES",
    "OMNI_ZONES",
    "OmniBlockLayout",
    "PRODUCT_SANCTUARY_BY_RATIO",
    "ROLE_SCALE",
    "VALID_ZONES",
    "ZONE_DEFAULT_ALIGN",
    "ZONE_GRID_AREA",
    "ZONE_NAMES",
    "ZONE_SELF_ALIGN",
    "compute_block_metrics",
    "deduplicate_blocks",
    "generate_omni_corridor_mask",
    "generate_omni_corridor_mask_from_rects",
    "get_product_sanctuary_rect",
    "get_zone_bounding_box",
    "map_category_to_default_blocks",
    "normalize_fingerprint",
    "normalize_zone",
]

