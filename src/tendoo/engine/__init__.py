"""
src/tendoo/engine

Tendoo Omni-Block Engine -- Bộ Dàn Trang Khối Đa Năng Thích Ứng:
- blocks: AdaptiveBlock, 6 category mappings & Content Fingerprint dedup
- geometry: Product Sanctuary & Bounding Box math
- mask: Unified Parametric Corridor Mask cho 2-branch velocity blending O(1)
- layout: OmniBlockLayout kế thừa BaseLayout
"""

from tendoo.engine.blocks import (
    AdaptiveBlock,
    VALID_ZONES,
    ROLE_SCALE,
    deduplicate_blocks,
    map_category_to_default_blocks,
    normalize_fingerprint,
)
from tendoo.engine.geometry import (
    PRODUCT_SANCTUARY_BY_RATIO,
    compute_block_metrics,
    get_product_sanctuary_rect,
    get_zone_bounding_box,
)
from tendoo.engine.mask import generate_omni_corridor_mask
from tendoo.engine.layout import OmniBlockLayout

__all__ = [
    "AdaptiveBlock",
    "VALID_ZONES",
    "ROLE_SCALE",
    "deduplicate_blocks",
    "map_category_to_default_blocks",
    "normalize_fingerprint",
    "PRODUCT_SANCTUARY_BY_RATIO",
    "compute_block_metrics",
    "get_product_sanctuary_rect",
    "get_zone_bounding_box",
    "generate_omni_corridor_mask",
    "OmniBlockLayout",
]
