"""
src/tendoo_v3/renderer.py

Engine kết nối HTML/CSS & Playwright Chromium Renderer cho Tendoo v3:
- Nạp template Jinja2 từ `src/tendoo_v3/templates/<template>/template.html`.
- Nhúng 100% font tiếng Việt Unicode Base64 offline.
- Tự động áp dụng các hiệu ứng ánh sáng (3D Gold, Neon, Chrome) và CSS Glassmorphism.
- Chụp ảnh PNG không nén (lossless) thông qua Headless Chromium.
"""

from __future__ import annotations

import base64
import colorsys
from pathlib import Path
from typing import Any, Dict, Optional

import jinja2
import numpy as np
from PIL import Image

from tendoo_core.colors import ensure_contrast
from tendoo_core.fonts import resolve_font
from tendoo_core.poster_renderer import PosterRenderer
from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.geometry import compute_density_score, get_zones
from tendoo_v3.icons import (
    BULLET_SPARKLE_SVG,
    get_icon_svg,
    infer_semantic_icon,
    parse_store_info_items,
    render_qr_code_svg,
    render_star_rating_svg,
)
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.styles import (
    COMMON_AUTOFIT_JS,
    get_adaptive_palette,
    get_effect_css,
)

# Chặn nội dung dạng list phình vô hạn TRƯỚC khi tới template -- mask/CSS của mỗi
# zone là 1 khối kích thước cố định (xem geometry.py), autofit chỉ CO CHỮ chứ không
# tự sinh thêm không gian, nên số lượng phần tử vẫn phải có trần hợp lý làm lưới an
# toàn cuối cùng (đã đo thấy thật: extra-tag-row/board-col-left không giới hạn số
# dòng có thể đẩy nội dung tràn ra ngoài vùng đã dành cho nó).
MAX_EXTRA_TEXTS = 6
MAX_STORE_ITEMS = 4
MAX_STEPS = 6


def compute_type_scale_ratio(style: StyleConfig) -> float:
    """Chọn tỷ lệ modular scale (docs/DESIGN_PRINCIPLES.md mục 1) dùng để derive
    subhead_max_font từ hero_max_font -- thay vì mỗi budget function tự hard-code
    1 hằng số subhead max-font cố định không phụ thuộc hero_max_font thật của
    branch đang chạy (nguyên nhân tỷ lệ hero:subhead trôi dạt 2.7x-4.4x tùy branch
    nội dung, đã xác nhận qua review thực tế cả 3 hàm compute_*_budget bên dưới).
    1.618 (Golden Ratio) cho theme vàng/luxury/kim loại, 1.333 (Perfect Fourth) mặc định.
    """
    signal = f"{style.text_effect} {style.background_tone} {style.theme_color}".lower()
    if any(k in signal for k in ("gold", "luxury", "metallic", "chrome", "hologram")):
        return 1.618
    return 1.333


def _derive_subhead_max_font(hero_max_f: float, ratio: float, ceiling: float) -> float:
    """subhead_max_font = hero_max_font / ratio, chặn trần bởi `ceiling` để chữ phụ
    không phình to bất thường khi text ngắn (autofit vẫn co theo max_height thật,
    nhưng trần max_font riêng cần hợp lý để giữ đúng thứ bậc thị giác hero > subhead)."""
    return round(min(hero_max_f / ratio, ceiling), 1)


def compute_sandwich_top_budget(
    plan: TendooCreativePlan,
    top_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách chiều cao động cho dải đỉnh của sandwich_top_heavy."""
    has_badge = bool(plan.badge)
    has_subhead = bool(plan.subhead)
    has_extra = bool(plan.extra_texts)
    is_narrow = (width / height) < 0.7  # 9:16
    is_wide = (width / height) >= 1.5   # 16:9

    # Trừ padding và gap để tính chiều cao khả dụng thật sự (avail_h)
    pad_tb = 12.0 if is_wide else 16.0
    gap_val = 3.0 if is_wide else 4.0
    num_items = sum([has_badge, True, has_subhead, has_extra])
    gap_total = max(0, num_items - 1) * gap_val
    avail_h = max(50.0, top_height - pad_tb - gap_total)

    # Cỡ chữ tối đa cho Hero phụ thuộc vào tỉ lệ khung hình
    if is_wide:
        hero_max_f = 72.0
        hero_min_f = 24.0
    elif is_narrow:
        hero_max_f = 86.0
        hero_min_f = 26.0
    else:  # 1:1, 4:5
        hero_max_f = 96.0
        hero_min_f = 28.0

    # Phân bổ ngân sách chiều cao động:
    # Badge luôn là 1 pill gọn gàng, chiều cao thực tế tối thiểu là 24px (16:9) hoặc 26-28px (các tỉ lệ khác).
    if has_badge:
        badge_h = 24.0 if is_wide else max(26.0, avail_h * 0.12)
        rem_h = max(30.0, avail_h - badge_h)
    else:
        badge_h = 0.0
        rem_h = avail_h

    if not has_subhead and not has_extra:
        # Chỉ có Hero (hoặc Badge + Hero)
        hero_h = rem_h * 0.96
        subhead_h = 0.0
        extra_h = 0.0
        if has_badge:
            hero_max_f = min(hero_max_f, 90.0)
    elif has_subhead and not has_extra:
        # Hero + Subhead
        hero_h = rem_h * 0.68
        subhead_h = rem_h * 0.32
        extra_h = 0.0
        hero_max_f = min(hero_max_f, 84.0 if has_badge else 88.0)
    elif not has_subhead and has_extra:
        # Hero + Extra
        hero_h = rem_h * 0.64
        subhead_h = 0.0
        extra_h = rem_h * 0.36
        hero_max_f = min(hero_max_f, 76.0 if has_badge else 80.0)
    else:
        # Cả 3: Hero + Subhead + Extra
        # Tái cân bằng ngân sách: tăng tỷ trọng cho Extra (đặc biệt khi là Freetext Block nhiều dòng)
        # Giảm nhẹ Hero trần để tránh Hero chiếm 2 dòng to đè dải đỉnh.
        hero_h = rem_h * 0.44
        subhead_h = rem_h * 0.22
        extra_h = rem_h * 0.34
        hero_max_f = min(hero_max_f, 62.0)
        hero_min_f = 24.0

    ratio = compute_type_scale_ratio(plan.style)
    # TRẦN SUBHEAD THEO CHUẨN POSTER, KHÔNG THEO CHUẨN VĂN BẢN (2026-09-25):
    # trần cũ là hằng số cố định (20/26/28px) và LUÔN là ràng buộc thắng thế -- đo thật
    # qua probe_type_contrast cho thấy subhead bị ghim đúng ở trần 28px trong khi hero
    # chỉ đạt 43-45px, ra tương phản 1.6x, trong khi DESIGN_PRINCIPLES mục 1.2 đòi
    # 4x-10x (poster được nhìn trong 1.5-3 giây, không phải trang sách).
    # Thang 1.333/1.618 (Perfect Fourth / Golden Ratio) là thang cho VĂN BẢN nhiều cấp
    # liền kề, quá nông cho nhịp 2 cấp Hook-vs-Context của poster.
    # Nay trần subhead = 1/4 trần hero (đúng cận dưới 4x của squint test), chặn sàn 13px
    # để không rơi xuống mức không đọc nổi.
    # SÀN 13px CÒN LÀ MỘT TÍN HIỆU: khi nó bị kích hoạt (hero_max_f < 52) nghĩa là nội
    # dung đã quá dày so với dải đỉnh này -- đó đúng là lúc hợp đồng dung lượng phải
    # reroute sang template cỡ lớn hơn, chứ không phải lúc ép chữ nhỏ thêm nữa.
    subhead_ceiling = max(13.0, min(24.0, hero_max_f / 4.0))
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=subhead_ceiling)

    return {
        "badge": {"max_h": round(badge_h, 1), "min_font": 12.0, "max_font": 20.0 if not is_wide else 17.0},
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "extra": {"max_h": round(extra_h, 1), "min_font": 11.0, "max_font": 22.0 if not is_wide else 18.0},
    }


def compute_sandwich_top_heavy_bottom_budget(
    plan: TendooCreativePlan,
    bottom_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho cụm đáy của sandwich_top_heavy (Store Info + CTA + QR).
    Khác với sandwich_bottom_heavy (nơi store nằm ở Top Bar mỏng ~90-128px),
    sandwich_top_heavy có Bottom Band rộng rãi (chiều cao 16% canvas ~163.8px ở 1:1/4:5).
    Vì vậy, thông tin cửa hàng có không gian thoải mái để hiển thị nổi bật, dễ đọc."""
    is_narrow = (width / height) < 0.7  # 9:16
    is_wide = (width / height) >= 1.5   # 16:9

    # Đếm số lượng item store thực tế
    num_store = 0
    if plan.store_info:
        raw = plan.store_info
        if "•" in raw:
            parts = [p.strip() for p in raw.split("•") if p.strip()]
            num_store = len(parts)
        elif "|" in raw:
            parts = [p.strip() for p in raw.split("|") if p.strip()]
            num_store = len(parts)
        else:
            num_store = 1

    # Phân bổ ngân sách store theo số lượng item & tỉ lệ khung hình
    if num_store <= 1:
        # Chỉ có 1 hotline hoặc 1 địa chỉ duy nhất: Phóng to nổi bật
        if is_wide:
            store_max_f = 18.0
            store_min_f = 13.0
            store_max_h = 44.0
        elif is_narrow:
            store_max_f = 18.0
            store_min_f = 13.0
            store_max_h = 50.0
        else:  # 1:1, 4:5
            store_max_f = 22.0
            store_min_f = 14.0
            store_max_h = 56.0
    elif num_store <= 3:
        # 2-3 items (vd: Hotline + Địa chỉ + Giờ mở cửa ở sth_03)
        if is_wide:
            store_max_f = 14.5
            store_min_f = 11.5
            store_max_h = bottom_height * 0.75
        elif is_narrow:
            store_max_f = 16.0
            store_min_f = 12.0
            store_max_h = bottom_height * 0.55
        else:  # 1:1, 4:5 (bottom_height = 163.8px)
            store_max_f = 18.5  # Tăng mạnh từ 14.5px lên 18.5px
            store_min_f = 13.0
            store_max_h = bottom_height * 0.82  # ~134px
    else:
        # >= 4 items (nhiều chi nhánh, email, hotline như sth_04)
        if is_wide:
            store_max_f = 13.0
            store_min_f = 11.0
            store_max_h = bottom_height * 0.80
        elif is_narrow:
            store_max_f = 14.0
            store_min_f = 11.0
            store_max_h = bottom_height * 0.55
        else:  # 1:1, 4:5
            store_max_f = 15.5
            store_min_f = 11.5
            store_max_h = bottom_height * 0.85  # ~139px

    cta_max_f = 20.0 if not is_wide else 15.0
    cta_min_f = 13.0
    cta_max_h = 48.0 if not is_wide else 38.0

    return {
        "cta": {"max_h": round(cta_max_h, 1), "min_font": cta_min_f, "max_font": cta_max_f},
        "store": {"max_h": round(store_max_h, 1), "min_font": store_min_f, "max_font": store_max_f},
    }


def compute_sandwich_bottom_budget(
    plan: TendooCreativePlan,
    bottom_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách chiều cao động cho bệ đỡ đáy của sandwich_bottom_heavy."""
    has_subhead = bool(plan.subhead)
    has_extra = bool(plan.extra_texts)
    has_cta = bool(plan.cta or plan.qr_code)
    is_narrow = (width / height) < 0.7  # 9:16
    is_wide = (width / height) >= 1.5   # 16:9

    # Kiểm tra xem extra_texts có phải là block freetext / bullet nhiều dòng không
    has_freetext = False
    if plan.extra_texts:
        has_freetext = any(t.strip().startswith(("*", "-", "•")) for t in plan.extra_texts) or (
            sum(1 for t in plan.extra_texts if len(t) > 35 or len(t.split()) > 6) > len(plan.extra_texts) / 2
        )

    pad_tb = 10.0 if is_wide else 14.0
    gap_val = 2.5 if is_wide else 3.5
    num_items = sum([True, has_subhead, has_extra, has_cta])
    gap_total = max(0, num_items - 1) * gap_val
    avail_h = max(60.0, bottom_height - pad_tb - gap_total)

    # Nâng trần cỡ chữ Hero tối đa theo từng tỉ lệ
    if is_wide:
        hero_max_f = 72.0
        hero_min_f = 24.0
    elif is_narrow:
        hero_max_f = 88.0
        hero_min_f = 26.0
    else:  # 1:1, 4:5
        hero_max_f = 96.0
        hero_min_f = 28.0

    if not has_subhead and not has_extra and not has_cta:
        # Minimal: Chỉ có Hero duy nhất
        hero_h = avail_h * 0.95
        subhead_h = 0.0
        extra_h = 0.0
        cta_h = 0.0
    elif has_subhead and not has_extra and not has_cta:
        # Hero + Subhead
        hero_h = avail_h * 0.68
        subhead_h = avail_h * 0.32
        extra_h = 0.0
        cta_h = 0.0
        hero_max_f = min(hero_max_f, 90.0)
    elif not has_subhead and not has_extra and has_cta:
        # Hero + CTA
        hero_h = avail_h * 0.68
        subhead_h = 0.0
        extra_h = 0.0
        cta_h = avail_h * 0.32
        hero_max_f = min(hero_max_f, 90.0)
    elif has_subhead and not has_extra and has_cta:
        # Hero + Subhead + CTA (Light text, vd sbh_02, sbh_06, sbh_13)
        hero_h = avail_h * 0.58
        subhead_h = avail_h * 0.26
        extra_h = 0.0
        cta_h = avail_h * 0.16
        hero_max_f = min(hero_max_f, 84.0)
    elif not has_subhead and has_extra and has_cta:
        # Hero + Extra + CTA
        hero_h = avail_h * 0.50
        subhead_h = 0.0
        extra_h = avail_h * 0.28
        cta_h = avail_h * 0.22
        hero_max_f = min(hero_max_f, 76.0)
    elif has_subhead and has_extra and not has_cta:
        # Hero + Subhead + Extra
        hero_h = avail_h * 0.52
        subhead_h = avail_h * 0.24
        extra_h = avail_h * 0.24
        cta_h = 0.0
        hero_max_f = min(hero_max_f, 76.0)
    else:
        # Full 4 items: Hero + Subhead + Extra + CTA/QR (Medium & Heavy, vd sbh_03, sbh_04, sbh_08, sbh_11, sbh_15)
        if is_wide:
            hero_h = avail_h * 0.36
            subhead_h = avail_h * 0.20
            extra_h = avail_h * 0.22
            cta_h = avail_h * 0.22
            hero_max_f = min(hero_max_f, 56.0)
        else:
            hero_h = avail_h * 0.38
            subhead_h = avail_h * 0.20
            extra_h = avail_h * 0.24
            cta_h = avail_h * 0.18
            hero_max_f = min(hero_max_f, 62.0)
        hero_min_f = 22.0

    ratio = compute_type_scale_ratio(plan.style)
    subhead_ceiling = 20.0 if is_wide else (26.0 if is_narrow else 28.0)
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=subhead_ceiling)

    extra_min_f = 11.0 if has_freetext else 13.0
    extra_max_f = 22.0 if not is_wide else 18.0
    cta_min_f = 12.0 if is_wide else 13.0
    cta_max_f = 22.0 if not is_narrow else 18.0

    # Dynamic store budget cho dải đỉnh (Top Bar):
    # Phân biệt hotline đơn lẻ (cần cỡ chữ nổi bật, phóng to 18-21px)
    # vs danh sách nhiều địa chỉ chi nhánh (cần giới hạn 14-14.5px).
    is_single_store = not plan.store_info or (
        "•" not in plan.store_info
        and "|" not in plan.store_info
        and len(plan.store_info) < 35
    )

    if is_single_store:
        if is_wide:
            store_max_f = 17.5
            store_min_f = 13.0
            store_max_h = 44.0
        elif is_narrow:
            store_max_f = 16.5
            store_min_f = 12.0
            store_max_h = 44.0
        else:  # 1:1, 4:5
            store_max_f = 21.0
            store_min_f = 13.5
            store_max_h = 52.0
    else:
        if is_wide:
            store_max_f = 14.0
            store_min_f = 11.0
            store_max_h = 56.0
        else:
            store_max_f = 14.5
            store_min_f = 11.5
            store_max_h = 75.0

    return {
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "extra": {"max_h": round(extra_h, 1), "min_font": extra_min_f, "max_font": extra_max_f},
        "cta": {"max_h": round(cta_h, 1), "min_font": cta_min_f, "max_font": cta_max_f},
        "store": {"max_h": round(store_max_h, 1), "min_font": store_min_f, "max_font": store_max_f},
    }


def compute_split_budget(
    plan: TendooCreativePlan,
    col_height: float,
    col_width: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách chiều cao động cho cột split_left và split_right."""
    has_badge = bool(plan.badge)
    has_subhead = bool(plan.subhead)
    has_extra = bool(plan.extra_texts)
    has_cta = bool(plan.cta)
    has_qr = bool(plan.qr_code)
    has_store = bool(plan.store_info)
    is_wide = (width / height) >= 1.5
    is_narrow = (width / height) < 0.7

    pad_tb = 32.0 if is_wide else 44.0
    avail_h = max(200.0, col_height - pad_tb)

    if is_wide:  # 16:9 Landscape (h=576)
        hero_max_f = 64.0
        hero_min_f = 22.0
    elif is_narrow:  # 9:16 Narrow Portrait (h=1024, w=576)
        hero_max_f = 76.0
        hero_min_f = 24.0
    else:  # 1:1, 4:5
        hero_max_f = 88.0
        hero_min_f = 26.0

    # Phân bổ ngân sách theo mức độ hiện diện của các thành phần
    if not has_subhead and not has_extra:
        hero_h = avail_h * 0.54
        subhead_h = 0.0
        extra_h = 0.0
        bottom_h = avail_h * 0.42
        # Cấp số nhân cho Minimal Mode: khi không có subhead/extra, cho phép Hero phóng to
        # vượt trần tiêu chuẩn để chiếm lĩnh không gian, tạo ấn tượng poster thương mại mạnh mẽ
        if is_wide:
            hero_max_f = 86.0
        elif is_narrow:
            hero_max_f = 84.0
        else:
            hero_max_f = 96.0
    elif has_subhead and not has_extra:
        hero_h = avail_h * 0.44
        subhead_h = avail_h * 0.22
        extra_h = 0.0
        bottom_h = avail_h * 0.32
        hero_max_f = min(hero_max_f, 78.0)
    elif not has_subhead and has_extra:
        hero_h = avail_h * 0.44
        subhead_h = 0.0
        extra_h = avail_h * 0.24
        bottom_h = avail_h * 0.30
        hero_max_f = min(hero_max_f, 78.0)
    else:
        # Đầy đủ cả Subhead và Extra texts
        hero_h = avail_h * 0.38
        subhead_h = avail_h * 0.20
        extra_h = avail_h * 0.22
        bottom_h = avail_h * 0.20
        hero_max_f = min(hero_max_f, 68.0)
        hero_min_f = 20.0

    ratio = compute_type_scale_ratio(plan.style)
    if is_narrow:
        subhead_ceiling = 18.0
        extra_max_f = 15.0
        extra_min_f = 11.0
    elif is_wide:
        subhead_ceiling = 18.0
        extra_max_f = 16.0
        extra_min_f = 10.5
    else:  # 1:1, 4:5
        subhead_ceiling = 22.0
        extra_max_f = 18.0
        extra_min_f = 12.0

    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=subhead_ceiling)

    # Dynamic extra_min_f: if multiline freetext block, allow 10.5 to prevent wrapping overflow
    if plan.extra_texts and any(len(t) > 30 for t in plan.extra_texts):
        extra_min_f = min(extra_min_f, 10.5)

    # Dynamic store budget:
    # Phân biệt giữa 1 dòng hotline đơn lẻ (cần cỡ chữ nổi bật, dễ đọc)
    # và thông tin địa chỉ chuỗi chi nhánh nhiều dòng dài (cần khống chế trần để bẻ dòng gọn).
    is_single_store = not plan.store_info or (
        "•" not in plan.store_info
        and "|" not in plan.store_info
        and len(plan.store_info) < 35
    )

    if is_single_store:
        if is_wide:
            store_max_f = 17.0
            store_min_f = 12.0
            store_h = 52.0
        elif is_narrow:
            store_max_f = 16.0
            store_min_f = 12.0
            store_h = 52.0
        else:  # 1:1, 4:5
            store_max_f = 18.5
            store_min_f = 13.0
            store_h = 56.0
    else:
        # Multi-branch address: trần 14-14.5px để tránh bung cỡ chữ làm vỡ bố cục
        if is_wide:
            store_max_f = 14.0
            store_min_f = 11.0
            store_h = 80.0
        elif is_narrow:
            store_max_f = 14.5
            store_min_f = 11.0
            store_h = 95.0
        else:  # 1:1, 4:5
            store_max_f = 14.5
            store_min_f = 11.5
            store_h = 105.0

    return {
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 12.0, "max_font": subhead_max_f},
        "extra": {"max_h": round(extra_h, 1), "min_font": extra_min_f, "max_font": extra_max_f},
        "bottom": {"max_h": round(bottom_h, 1), "min_font": 13.0, "max_font": 20.0},
        "store": {"max_h": round(store_h, 1), "min_font": store_min_f, "max_font": store_max_f},
    }


def compute_step_process_roadmap_budget(
    plan: TendooCreativePlan,
    header_height: float,
    platform_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách chiều cao động cho dải header (badge/hero/subhead) và bệ
    platform (steps/cta/store) của step_process_roadmap.

    Các phân số cho tổ hợp Badge+Hero+Subhead (tổ hợp DUY NHẤT mà toàn bộ 16 test
    case hiện có sử dụng) và cho platform giữ NGUYÊN đúng giá trị đã đo đạc ổn định
    trước khi có hàm này (0.65/0.35 cho hero/subhead, 0.68/0.52 cho steps, 38/34px
    cho cta, 38/30px cho store) -- badge-pill trước đây hardcode font-size:11px
    (vi phạm mục 0 "không hardcode cho phần tử nội dung"), nay được cấp ngân sách
    autofit riêng CỘNG THÊM (không trừ vào hero/subhead) vì badge đo thực tế chỉ cao
    ~23px, không cạnh tranh không gian thật với hero/subhead (xem phép đo Playwright
    trong PLAN_PER_TEMPLATE_BUDGET_ENGINE.md).
    """
    has_badge = bool(plan.badge)
    has_subhead = bool(plan.subhead)
    is_narrow = (width / height) < 0.7   # 9:16
    is_wide = (width / height) >= 1.5    # 16:9

    # Sàn tuyệt đối 30px cho badge_h: ở 16:9 (header_height ~104-127px), fraction
    # 0.15/0.20 (15-25px) không đủ cho badge-pill render thật (font 11.5-16 sau khi
    # nâng trần -- đo Playwright thật: cao 22.6px) -- data-max-height nhỏ hơn content
    # khiến self-overflow (xác nhận trên cả step_process_roadmap và recruitment_board,
    # 2 template dùng chung cấu trúc header này). header là zone MỀM (không clip
    # cứng) nên nâng badge_h không ảnh hưởng hero/subhead thật đang có dư.
    if not has_badge and not has_subhead:
        badge_h = 0.0
        hero_h = header_height * 0.95
        subhead_h = 0.0
    elif has_badge and not has_subhead:
        badge_h = max(header_height * 0.20, 30.0)
        hero_h = header_height * 0.80
        subhead_h = 0.0
    elif not has_badge and has_subhead:
        badge_h = 0.0
        hero_h = header_height * 0.68
        subhead_h = header_height * 0.32
    else:
        badge_h = max(header_height * 0.15, 30.0)
        hero_h = header_height * 0.65
        subhead_h = header_height * 0.35

    hero_max_f, hero_min_f = 52.0, 20.0
    ratio = compute_type_scale_ratio(plan.style)
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=20.0)

    # Trần đã nâng theo đúng tiền lệ các template khác (user báo cáo trực tiếp:
    # "thông tin thêm"/"thông tin cửa hàng" nhìn nhỏ so với hero).
    if is_narrow:
        steps_h = platform_height * 0.52
        steps_min_f, steps_max_f = 14.0, 18.0
        cta_h, cta_min_f, cta_max_f = 34.0, 13.0, 17.0
        store_h, store_min_f, store_max_f = 30.0, 14.0, 18.0
    else:
        steps_h = platform_height * 0.68
        steps_min_f, steps_max_f = 14.0, 19.0
        cta_h, cta_min_f, cta_max_f = 38.0, 13.0, 17.0
        store_h, store_min_f, store_max_f = 38.0, 14.0, 18.0

    return {
        "badge": {"max_h": round(badge_h, 1), "min_font": 11.5, "max_font": 16.0},
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "steps": {"max_h": round(steps_h, 1), "min_font": steps_min_f, "max_font": steps_max_f},
        "cta": {"max_h": cta_h, "min_font": cta_min_f, "max_font": cta_max_f},
        "store": {"max_h": store_h, "min_font": store_min_f, "max_font": store_max_f},
    }


def compute_grand_opening_banner_budget(
    plan: TendooCreativePlan,
    header_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `grand_opening_banner`.

    `.opening-header-cluster` KHÔNG có height/max-height cố định -- nó tự phình
    theo tổng chiều cao các con (badge + hero + subhead + freetext), phần mở giữa
    canvas cho sản phẩm còn rất nhiều dư địa (xem geometry.py::_grand_opening_banner,
    header chỉ chiếm 30-34% đỉnh). Vì vậy các fraction bên dưới KHÔNG cần cộng lại
    = 1.0 -- mỗi fraction chỉ là trần max-height RIÊNG của từng phần tử autofit,
    độc lập nhau (logic này trước đây nằm thẳng trong template.html qua Jinja
    `{% set %}`, nay chuyển vào đây theo đúng chuẩn budget engine). Badge (kicker-
    capsule) trước đây hardcode font-size:12.5px (vi phạm mục 0), nay là autofit
    thật với ngân sách riêng nằm trong khoản dự trữ cố định `header_badge_reserve`
    đã trừ khỏi `header_avail_h` cho hero/subhead/freetext.
    `freetext_block`/`extra_tag_row` được cấp fraction LỚN NHẤT (0.60) vì là phần
    tử duy nhất có số dòng thay đổi thật (1-6 dòng) -- đã đo thấy tràn thật
    ~13-22px khi chỉ cấp ngang hero/subhead cho 3 dòng dài ở 16:9
    (case open_11_16x9_rich_features).
    Bottom bar (cta/store) giữ NGUYÊN đúng số hardcode px đã kiểm chứng ổn định
    trước đó, chỉ branch theo is_portrait_narrow (2 tầng dọc) vs chuẩn (3 cột ngang).
    """
    has_badge = bool(plan.badge)
    is_narrow = (width / height) < 0.7
    is_wide = (width / height) >= 1.5

    header_badge_reserve = (36.0 if is_wide else 52.0) if has_badge else 0.0
    header_avail_h = max(80.0, header_height - header_badge_reserve)

    # Trần nâng thêm 1 vòng nữa (user yêu cầu trực tiếp "tăng toàn bộ cỡ chữ" sau khi
    # đã nâng 1 lần trong cùng phiên) -- hero giữ nguyên 68 (đã ở mức trần cao nhất
    # dùng chung toàn dự án), các phần còn lại nâng thêm 1 bậc.
    if is_narrow:
        cta_h, cta_min_f, cta_max_f = 42.0, 13.0, 18.0
        store_h, store_min_f, store_max_f = 38.0, 14.0, 19.0
    else:
        cta_h, cta_min_f, cta_max_f = 46.0, 13.0, 18.0
        store_h, store_min_f, store_max_f = 44.0, 14.0, 19.0

    return {
        "badge": {"max_h": min(36.0 if is_wide else 40.0, header_badge_reserve) if has_badge else 0.0, "min_font": 12.0, "max_font": 17.0},
        "hero": {"max_h": round(header_avail_h * (0.50 if is_wide else 0.46), 1), "min_font": 22.0, "max_font": 68.0},
        "subhead": {"max_h": max(26.0, round(header_avail_h * (0.24 if is_wide else 0.185), 1)), "min_font": 13.0, "max_font": 22.0},
        "freetext_block": {"max_h": round(header_avail_h * 0.60, 1), "min_font": 14.0, "max_font": 20.0},
        "freetext_pills": {"max_h": round(header_avail_h * 0.60, 1), "min_font": 13.0, "max_font": 18.0},
        "cta": {"max_h": cta_h, "min_font": cta_min_f, "max_font": cta_max_f},
        "store": {"max_h": store_h, "min_font": store_min_f, "max_font": store_max_f},
    }


def compute_recruitment_board_budget(
    plan: TendooCreativePlan,
    header_height: float,
    board_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `recruitment_board`.

    Header (badge/hero/subhead) dùng đúng cấu trúc/giá trị đã kiểm chứng ổn định ở
    `compute_step_process_roadmap_budget()` (2 template có `.recruitment-header`/
    `.top-header` giống hệt nhau về gap/badge-pill/hero/subhead).

    Board là 1 card cố định (`.board-card`, bo góc, có background -- ĐÚNG loại
    "card" hợp lệ theo yeu_cau_templates.txt mục 1, không phải scrim) -- ở narrow
    (9:16) xếp CỘT DỌC (col-left rồi board-bottom-split chứa cta+store+qr), khác
    hẳn layout HÀNG NGANG (3 cột) của 1:1/4:5/16:9. `.board-col-qr` chứa QR kiosk
    KHÔNG tự autofit (kích thước gần-cố-định ~100-115px kể cả border/padding) nên
    ở narrow, chiều cao thật của `.board-bottom-split` bị QR áp đặo một sàn cố định
    (`qr_reserve`) bất kể cta/store budget được cấp bao nhiêu -- extra_h phải trừ
    đúng phần này để không tràn (đã xác nhận qua đo Playwright thật: case
    recruit_10_9x16_left_bullet tràn ~7-10px khi geometry.py's board_h còn 0.20,
    xem comment tại đó).
    """
    has_badge = bool(plan.badge)
    has_subhead = bool(plan.subhead)
    is_narrow = (width / height) < 0.7
    is_wide = (width / height) >= 1.5

    # Sàn tuyệt đối 30px cho badge_h: ở 16:9 (header_height ~104-127px), fraction
    # 0.15/0.20 (15-25px) không đủ cho badge-pill render thật (font 11.5-16 sau khi
    # nâng trần -- đo Playwright thật: cao 22.6px) -- data-max-height nhỏ hơn content
    # khiến self-overflow (xác nhận trên cả step_process_roadmap và recruitment_board,
    # 2 template dùng chung cấu trúc header này). header là zone MỀM (không clip
    # cứng) nên nâng badge_h không ảnh hưởng hero/subhead thật đang có dư.
    if not has_badge and not has_subhead:
        badge_h = 0.0
        hero_h = header_height * 0.95
        subhead_h = 0.0
    elif has_badge and not has_subhead:
        badge_h = max(header_height * 0.20, 30.0)
        hero_h = header_height * 0.80
        subhead_h = 0.0
    elif not has_badge and has_subhead:
        badge_h = 0.0
        hero_h = header_height * 0.68
        subhead_h = header_height * 0.32
    else:
        badge_h = max(header_height * 0.15, 30.0)
        hero_h = header_height * 0.65
        subhead_h = header_height * 0.35

    hero_max_f, hero_min_f = 52.0, 20.0
    ratio = compute_type_scale_ratio(plan.style)
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=20.0)

    # Trần đã nâng theo đúng tiền lệ các template khác (user báo cáo trực tiếp:
    # "thông tin thêm"/"thông tin cửa hàng" nhìn nhỏ so với hero).
    if is_narrow:
        pad_gap = 20.0 + 8.0       # padding dọc (10+10) + 1 gap giữa col-left/bottom-split
        qr_reserve = 115.0         # QR kiosk cố định (~108px) + border-top(1) + padding-top(6)
        title_reserve = 26.0       # .col-title (~20px) + gap nội bộ board-col-left (6px)
        extra_h = max(40.0, board_height - pad_gap - qr_reserve - title_reserve)
        extra_min_f, extra_max_f = 14.0, 18.0
        cta_h, cta_min_f, cta_max_f = 34.0, 13.0, 17.0
        store_h, store_min_f, store_max_f = 38.0, 14.0, 18.0
    else:
        extra_h = board_height * 0.78
        extra_min_f, extra_max_f = 14.0, 18.0
        cta_h, cta_min_f, cta_max_f = 38.0, 13.0, 17.0
        store_h, store_min_f, store_max_f = 54.0, 14.0, 18.0

    return {
        "badge": {"max_h": round(badge_h, 1), "min_font": 11.5, "max_font": 16.0},
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "extra": {"max_h": round(extra_h, 1), "min_font": extra_min_f, "max_font": extra_max_f},
        "cta": {"max_h": cta_h, "min_font": cta_min_f, "max_font": cta_max_f},
        "store": {"max_h": store_h, "min_font": store_min_f, "max_font": store_max_f},
    }


def compute_menu_price_board_budget(
    plan: TendooCreativePlan,
    content_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `menu_price_board` (cột dọc full-height, cùng họ
    `zones.content` với split_left/split_right nhưng nội dung chính là danh sách
    món/giá `menu_list` -- số dòng biến đổi mạnh (2-8 món), luôn được cấp fraction
    LỚN NHẤT. `.menu-column` dùng flex column `justify-content:space-between`
    (không có `overflow:hidden` riêng, chỉ dựa vào `.poster-canvas` ở ngoài) nên
    hero/subhead/menu_list/bottom_stack PHẢI tự budget hợp lý, không dựa may rủi
    vào width-binding như các template dải mỏng khác.
    """
    has_badge = bool(plan.badge)
    has_subhead = bool(plan.subhead)
    is_narrow = (width / height) < 0.7
    is_wide = (width / height) >= 1.5

    # Giữ nguyên đúng ceiling hero/subhead đã kiểm chứng ổn định trước đây (fraction
    # của content_height, không phải avail_h -- vì justify-content:space-between tự
    # co giãn khoảng trống giữa top-stack/menu-list/bottom-stack).
    hero_h = content_height * 0.20
    subhead_h = content_height * 0.09 if has_subhead else 0.0
    menu_h = content_height * 0.44

    hero_max_f, hero_min_f = 52.0, 20.0
    ratio = compute_type_scale_ratio(plan.style)
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=20.0)

    # Trần đã nâng theo đúng tiền lệ các template khác (user báo cáo trực tiếp:
    # "thông tin thêm"/"thông tin cửa hàng" nhìn nhỏ so với hero).
    if is_wide:
        cta_h, cta_min_f, cta_max_f = 42.0, 13.0, 17.0
        store_h, store_min_f, store_max_f = 70.0, 14.0, 18.0
    else:
        cta_h, cta_min_f, cta_max_f = 46.0, 13.0, 17.0
        store_h, store_min_f, store_max_f = 90.0, 14.0, 18.0

    return {
        "badge": {"max_h": 40.0 if has_badge else 0.0, "min_font": 11.5, "max_font": 16.0},
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "menu_list": {"max_h": round(menu_h, 1), "min_font": 14.0, "max_font": 19.0},
        "cta": {"max_h": cta_h, "min_font": cta_min_f, "max_font": cta_max_f},
        "store": {"max_h": store_h, "min_font": store_min_f, "max_font": store_max_f},
    }


def compute_diagonal_slash_budget(
    plan: TendooCreativePlan,
    content_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `diagonal_slash` (cột dọc full-height `zones.content`,
    cùng họ split_left/split_right/menu_price_board -- badge (kicker-tag)/hero/
    subhead ở `.top-section`, freetext dual-mode (block/pills), rồi cta/store/qr ở
    `.conversion-suite`). `.diagonal-content` đã có `overflow:hidden` riêng nên an
    toàn hơn `.menu-column` (menu_price_board) -- giữ nguyên đúng % đã kiểm chứng
    ổn định trước đây, chỉ chuyển vào hàm Python theo đúng chuẩn budget engine.
    """
    has_badge = bool(plan.badge)
    has_subhead = bool(plan.subhead)
    is_wide = (width / height) >= 1.5

    hero_h = content_height * 0.35
    subhead_h = content_height * 0.18 if has_subhead else 0.0
    freetext_h = content_height * 0.22

    hero_max_f, hero_min_f = 68.0, 20.0
    ratio = compute_type_scale_ratio(plan.style)
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=20.0)

    # Trần đã nâng theo đúng tiền lệ các template khác (user báo cáo trực tiếp:
    # "thông tin thêm"/"thông tin cửa hàng" nhìn nhỏ so với hero).
    if is_wide:
        cta_h, cta_min_f, cta_max_f = 42.0, 13.0, 17.0
        store_h, store_min_f, store_max_f = 70.0, 14.0, 18.0
    else:
        cta_h, cta_min_f, cta_max_f = 46.0, 13.0, 17.0
        store_h, store_min_f, store_max_f = 90.0, 14.0, 18.0

    return {
        "badge": {"max_h": 40.0 if has_badge else 0.0, "min_font": 11.5, "max_font": 16.0},
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "freetext_block": {"max_h": round(freetext_h, 1), "min_font": 14.0, "max_font": 19.0},
        "freetext_pills": {"max_h": round(freetext_h, 1), "min_font": 13.0, "max_font": 17.0},
        "cta": {"max_h": cta_h, "min_font": cta_min_f, "max_font": cta_max_f},
        "store": {"max_h": store_h, "min_font": store_min_f, "max_font": store_max_f},
    }


def compute_before_after_budget(
    plan: TendooCreativePlan,
    bottom_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `before_after_split`'s `.bottom-info-pod` (card cố
    định ở đáy, xem yeu_cau_templates.txt mục 1 -- card bo góc hợp lệ, không phải
    scrim). Chuyển đúng logic fraction từng nằm hardcode trực tiếp trong
    template.html (đã qua 1 vòng sửa lỗi tràn footer-row hôm 2026-09-21) sang hàm
    Python theo đúng chuẩn budget engine -- CHƯA có hàm riêng trước đây (khác các
    template khác trong catalog, đây là mảnh ghép cuối cùng thiếu).

    16:9 landscape dùng bố cục 2 CỘT (`land-left-col`: rating+hero+subhead,
    `land-right-col`: qr+action-stack chứa cta+store) -- 2 cột độc lập nhau nên
    percent không cần cộng dồn giữa 2 cột, chỉ cần đúng trong nội bộ mỗi cột.
    1:1/9:16/4:5 dùng bố cục XẾP CHỒNG 1 cột (rating+hero+subhead+freetext+
    footer-row chứa qr+cta+store).
    """
    has_subhead = bool(plan.subhead)
    has_freetext = bool(plan.extra_texts)
    has_stars = bool(plan.rating)
    has_qr = bool(plan.qr_code)
    has_cta = bool(plan.cta)
    has_store = bool(plan.store_info)
    has_footer = has_qr or has_cta or has_store
    is_wide = (width / height) >= 1.5
    is_narrow = (width / height) < 0.7

    ratio = compute_type_scale_ratio(plan.style)

    if is_wide:
        # Trần font đã nâng theo đúng tiền lệ l_frame_showcase/customer_feedback_card
        # (chữ ở đây từng nhỏ hơn hẳn cần thiết so với hero, user báo cáo trực tiếp).
        hero_max_f, hero_min_f = 46.0, 18.0
        subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=18.0)
        return {
            "hero": {"max_h": round(bottom_height * 0.55, 1), "min_font": hero_min_f, "max_font": hero_max_f},
            "subhead": {"max_h": round(bottom_height * 0.30, 1) if has_subhead else 0.0, "min_font": 13.0, "max_font": subhead_max_f},
            "freetext_block": {"max_h": 0.0, "min_font": 14.0, "max_font": 19.0},
            "freetext_pills": {"max_h": 0.0, "min_font": 13.0, "max_font": 17.0},
            "cta": {"max_h": round(bottom_height * 0.32, 1), "min_font": 13.0, "max_font": 17.0},
            "store": {"max_h": round(bottom_height * 0.55, 1), "min_font": 14.0, "max_font": 18.0},
        }

    # 1:1/9:16/4:5: XẾP CHỒNG 1 cột trong `.bottom-info-pod` (card cố định, KHÔNG có
    # `overflow:hidden` riêng) -- (rating-stars nếu có) + hero + subhead + freetext +
    # footer-row (qr + cta + store, footer-row là HÀNG NGANG nên chiều cao của nó bị
    # QR áp đặt sàn cố định ~96-115px, không cộng dồn cta+store). Fraction gốc
    # (0.29/0.16/0.20/0.15/0.15) KHÔNG hề trừ padding+rating-stars+gap+footer thật
    # trước khi tính % -- đo Playwright thật xác nhận tràn ~8px ở case đủ mọi phần tử
    # (ba_12_4x5_left_rich). Tính đúng `avail_stack` (= bottom_height trừ hết các
    # khoản cố định thật) rồi mới chia % cho hero/subhead/freetext -- freetext được
    # ưu tiên fraction LỚN NHẤT vì là phần tử duy nhất có số dòng biến đổi thật
    # (0-4 dòng), hero/subhead thường bị width-binding trước khi chạm ceiling.
    if is_narrow:
        pad_tb, gap_val = 24.0, 5.0
    else:  # 1:1, 4:5
        pad_tb, gap_val = 22.0, 4.0

    # `footer_reserve` từng CỐ ĐỊNH 100/118px bất kể qr/cta/store có rỗng hay không --
    # đúng lỗi đã fix ở customer_feedback_card/luxury_centered_card (QR ~96-115px mới
    # là driver thật cần khoản dự trữ lớn; không QR chỉ cần đủ 1 hàng cta-btn/store-
    # text ~46px; không có gì cả thì 0, .footer-row giờ cũng đã bọc {% if %} nên
    # không còn tồn tại "trace" nào trong DOM khi rỗng).
    if has_qr:
        footer_reserve = 100.0 if is_narrow else 118.0
    elif has_footer:
        footer_reserve = 46.0
    else:
        footer_reserve = 0.0

    stars_reserve = 22.0 if has_stars else 0.0
    num_items = sum([has_stars, True, has_subhead, has_freetext, has_footer])  # hero luôn có mặt
    gap_total = max(0, num_items - 1) * gap_val
    avail_stack = max(80.0, bottom_height - pad_tb - gap_total - stars_reserve - footer_reserve)

    if has_subhead and has_freetext:
        hero_h, subhead_h, freetext_h = avail_stack * 0.36, avail_stack * 0.18, avail_stack * 0.40
    elif has_subhead and not has_freetext:
        hero_h, subhead_h, freetext_h = avail_stack * 0.62, avail_stack * 0.36, 0.0
    elif not has_subhead and has_freetext:
        hero_h, subhead_h, freetext_h = avail_stack * 0.55, 0.0, avail_stack * 0.43
    else:
        hero_h, subhead_h, freetext_h = avail_stack * 0.95, 0.0, 0.0

    # Trần font đã nâng theo đúng tiền lệ l_frame_showcase/customer_feedback_card --
    # user báo cáo trực tiếp chữ trong before/after nhìn nhỏ so với hero.
    hero_max_f, hero_min_f = 58.0, 18.0
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=20.0)
    cta_h = 36.0 if is_narrow else 42.0
    store_h = 70.0 if is_narrow else 90.0
    return {
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "freetext_block": {"max_h": round(freetext_h, 1), "min_font": 14.0, "max_font": 19.0},
        "freetext_pills": {"max_h": round(freetext_h, 1), "min_font": 13.0, "max_font": 17.0},
        "cta": {"max_h": cta_h, "min_font": 13.0, "max_font": 17.0},
        "store": {"max_h": store_h, "min_font": 14.0, "max_font": 18.0},
    }


def compute_l_frame_showcase_budget(
    plan: TendooCreativePlan,
    top_cluster_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `l_frame_showcase`'s `.lframe-top-cluster`.

    PHÁT HIỆN LỖI MẤT CHỮ THẬT (không phải lý thuyết): `.lframe-top-cluster` có
    `overflow:hidden` + `max-height` CSS cố định, nhưng test script của template
    này (`scripts/test_l_frame_showcase.py::measure_text_elements_in_page`) CHỈ đo
    các phần tử LÁ (`el.children.length > 0` thì bỏ qua) -- không bao giờ kiểm tra
    chính `.lframe-top-cluster`, nên 16/16 case "PASS" trước đây không hề phát hiện
    được rằng case "heavy" (badge+subhead+freetext cùng có mặt) đã bị CẮT MẤT phần
    đáy freetext-block thật (xem ảnh `lframe_12_16x9_heavy_right/poster.png`) --
    scrollHeight thật 319px nhưng clientHeight chỉ 266px. Nguyên nhân giống hệt các
    template khác đã sửa: fraction hero/subhead/freetext (0.55/0.22/0.22, tổng 0.99)
    không hề trừ padding+gap+badge trước khi tính %, nên khi cả 3 phần tử cùng có
    mặt và cần nhiều chữ, tổng vượt xa container thật.

    Sửa đúng gốc: tính `avail` (= max-height thật của container trừ padding/gap/
    badge) rồi mới chia % theo tổ hợp phần tử hiện diện. Hàm này còn tính luôn
    `container_max_h` (gộp logic "-10" từng nằm rải rác 2 nơi trong CSS narrow/wide
    media query của template.html) để có DUY NHẤT 1 nguồn chân lý.
    """
    has_badge = bool(plan.badge)
    has_subhead = bool(plan.subhead)
    has_freetext = bool(plan.extra_texts)
    is_narrow = (width / height) < 0.7
    is_wide = (width / height) >= 1.5

    container_max_h = top_cluster_height - 10.0 if (is_narrow or is_wide) else top_cluster_height

    pad_tb = 10.0
    gap_val = 5.0 if is_wide else 8.0
    badge_reserve = 30.0 if has_badge else 0.0
    num_items = sum([True, has_badge, has_subhead, has_freetext])
    gap_total = max(0, num_items - 1) * gap_val
    avail = max(80.0, container_max_h - pad_tb - gap_total - badge_reserve)

    if has_subhead and has_freetext:
        # Đo thật (`lframe_12_16x9_heavy_right`, badge+subhead+3 dòng freetext dài):
        # freetext-block scrollHeight thật 106px nhưng ngân sách gốc (0.28*avail) chỉ
        # cấp 57px -- tràn ~49px thật. NHƯNG đo thêm case 1-2 dòng (thường render dạng
        # pill gọn, không phải block đoạn văn) cho thấy 0.28 vẫn ĐỦ dư dả -- ép cả 2
        # trường hợp dùng chung 1 tỉ lệ 0.57 (như bản sửa đầu tiên) khiến hero bị bóp
        # xuống sàn 20px kể cả case "medium" 2 dòng ngắn, không cần thiết (user báo cáo
        # trực tiếp hero nhìn không nổi bật). Chỉ nâng tỉ trọng freetext khi THỰC SỰ
        # nhiều dòng (>=3, đúng ngưỡng đã đo gây tràn) -- <=2 dòng giữ nguyên tỉ lệ gốc.
        num_extra = len(plan.extra_texts or [])
        if num_extra >= 3:
            hero_h, subhead_h, freetext_h = avail * 0.25, avail * 0.18, avail * 0.57
        else:
            hero_h, subhead_h, freetext_h = avail * 0.50, avail * 0.18, avail * 0.28
    elif has_subhead and not has_freetext:
        hero_h, subhead_h, freetext_h = avail * 0.72, avail * 0.24, 0.0
    elif not has_subhead and has_freetext:
        hero_h, subhead_h, freetext_h = avail * 0.62, 0.0, avail * 0.34
    else:
        hero_h, subhead_h, freetext_h = avail * 0.92, 0.0, 0.0

    # Trần hero/subhead/freetext/store/badge nâng thêm 1 vòng nữa (user yêu cầu trực
    # tiếp "tăng tất cả cỡ chữ" sau khi đã nâng 1 lần trong cùng phiên) -- 58 vốn đã ở
    # giữa dải so với các template khác (46-68, xem grep `hero_max_f, hero_min_f =`
    # trong renderer.py) nhưng vẫn nâng lên khớp trần cao nhất (68, dùng ở split_left/
    # right) vì autofit chỉ dùng trần này khi ĐỦ chỗ thật -- không có rủi ro tràn tự
    # thân, chỉ cần verify lại geometry.py sau khi bump (đã làm, xem test suite).
    hero_max_f, hero_min_f = 68.0, 20.0
    ratio = compute_type_scale_ratio(plan.style)
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=22.0)

    return {
        "container_max_h": round(container_max_h, 1),
        "badge": {"max_h": 34.0 if has_badge else 0.0, "min_font": 12.0, "max_font": 17.0},
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "freetext_block": {"max_h": round(freetext_h, 1), "min_font": 14.0, "max_font": 20.0},
        "freetext_pills": {"max_h": round(freetext_h, 1), "min_font": 13.0, "max_font": 18.0},
        "store": {"max_h": 34.0, "min_font": 14.0, "max_font": 19.0},
    }


def compute_lifestyle_corner_pod_budget(
    plan: TendooCreativePlan,
    pod_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `lifestyle_corner_pod`'s `.corner-pod` (card cố định
    ở 1 góc, `overflow:hidden`). `.footer-row` xếp CỘT DỌC (cta-qr-group rồi
    store-info-col bên dưới, khác các template khác xếp chúng cùng 1 hàng) -- QR
    component cố định (~90-115px tuỳ kích thước) áp đặt 1 sàn không co giãn cho
    `.cta-qr-group`, cộng thêm `store-info-col` bên dưới khiến `.footer-row` chiếm
    phần đáng kể của pod bất kể ngân sách cta/store được cấp bao nhiêu.

    Fraction gốc hero/subhead/freetext (0.33/0.15/0.13) không hề trừ padding+badge+
    gap+footer thật trước khi tính % -- đo Playwright thật xác nhận tràn ~15-17px ở
    case đủ mọi phần tử (pod_08_4x5_bottom_right_tea, `.corner-pod` scrollHeight
    507 > clientHeight 490). Đã nâng riêng `box_h` 1:1/4:5 trong geometry.py (0.48 ->
    0.52) để có thêm margin, kết hợp với tính đúng `avail` (trừ hết khoản cố định
    thật) trước khi chia % ưu tiên freetext (biến đổi nhiều nhất).
    """
    has_badge = bool(plan.badge)
    has_subhead = bool(plan.subhead)
    has_freetext = bool(plan.extra_texts)
    has_qr = bool(plan.qr_code)
    has_cta = bool(plan.cta)
    has_store = bool(plan.store_info)
    has_footer = has_qr or has_cta or has_store
    is_narrow = (width / height) < 0.7
    is_wide = (width / height) >= 1.5

    if is_narrow or is_wide:
        pad_tb, gap_val, qr_h = 32.0, 6.0, 90.0
    else:
        pad_tb, gap_val, qr_h = 40.0, 8.0, 115.0

    badge_reserve = 28.0 if has_badge else 0.0
    store_ceiling = 70.0
    # `footer_reserve` từng CỐ ĐỊNH qr_h+store_ceiling bất kể qr/cta/store có rỗng hay
    # không -- đúng lỗi đã fix ở customer_feedback_card/before_after_split/
    # luxury_centered_card: khi pod đã co nhỏ lại (has_qr=False/has_freetext=False,
    # xem geometry.py::_lifestyle_corner_pod) mà avail vẫn bị trừ hết khoản QR ảo
    # (96-115px) dù QR không hề tồn tại, hero bị bóp nhỏ hơn hẳn mức cần thiết ngay
    # trong pod đã nhỏ -- user báo cáo trực tiếp "no content đáng lẽ nhiều space hơn
    # thì chữ phải to hơn". QR mới là driver thật cần khoản dự trữ lớn; không QR chỉ
    # cần đủ 1 hàng cta-btn/store-text (~46px); không có gì cả thì 0.
    if has_qr:
        footer_reserve = qr_h + 6.0 + store_ceiling
    elif has_footer:
        footer_reserve = 46.0
    else:
        footer_reserve = 0.0

    num_items = sum([True, has_freetext, has_footer])  # top-section luôn có mặt
    gap_total = max(0, num_items - 1) * gap_val
    avail = max(80.0, pod_height - pad_tb - gap_total - badge_reserve - footer_reserve)

    if has_subhead and has_freetext:
        hero_h, subhead_h, freetext_h = avail * 0.50, avail * 0.20, avail * 0.30
    elif has_subhead and not has_freetext:
        hero_h, subhead_h, freetext_h = avail * 0.72, avail * 0.28, 0.0
    elif not has_subhead and has_freetext:
        hero_h, subhead_h, freetext_h = avail * 0.60, 0.0, avail * 0.40
    else:
        hero_h, subhead_h, freetext_h = avail * 0.95, 0.0, 0.0

    hero_max_f, hero_min_f = 54.0, 18.0
    ratio = compute_type_scale_ratio(plan.style)
    # Trần đã nâng theo đúng tiền lệ l_frame_showcase/before_after_split (chữ từng
    # nhỏ hơn hẳn cần thiết so với hero).
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=20.0)

    return {
        "badge": {"max_h": 34.0 if has_badge else 0.0, "min_font": 11.5, "max_font": 16.0},
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "freetext_block": {"max_h": round(freetext_h, 1), "min_font": 14.0, "max_font": 19.0},
        "freetext_pills": {"max_h": round(freetext_h, 1), "min_font": 13.0, "max_font": 17.0},
        "cta": {"max_h": 42.0, "min_font": 13.0, "max_font": 17.0},
        "store": {"max_h": store_ceiling, "min_font": 14.0, "max_font": 18.0},
    }


def compute_customer_feedback_budget(
    plan: TendooCreativePlan,
    header_height: float,
    card_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `customer_feedback_card`.

    `.feedback-card` (card cố định, `overflow:hidden`) là phần tử biến đổi nội dung
    NHIỀU NHẤT trong 10 template (độ dài testimonial rất khác nhau -- xem mục 4 của
    tài liệu kế hoạch) -- (stars-badge-row cố định) + testimonial + reviewer-info
    (nhỏ, gần cố định) + freetext (0-N dòng) + footer-action-row (qr/cta/store,
    QR cố định ~90-118px áp đặt sàn không co giãn, giống các template khác).

    Đo Playwright thật cho thấy fraction gốc (testimonial 0.30, freetext 0.18) dù
    trông hợp lý riêng lẻ nhưng tổng nội dung thật (testimonial+reviewer+freetext)
    đã chiếm gần hết avail thật (127.5/129px, dư đúng ~1.5px) trên card 1:1/4:5 --
    XÁC NHẬN đây là thiết kế "chạy được nhờ may rủi" giống hệt các template khác đã
    sửa, dù test hiện tại (đã kiểm tra `.feedback-card` đúng chuẩn, không như lỗ hổng
    l_frame_showcase) vẫn PASS 16/16 vì chưa case nào vừa đủ để lộ ra. Đã nâng
    `card_frac_h` 1:1/4:5 trong geometry.py (0.315 -> 0.34) để có margin thật, kết
    hợp tính đúng `avail` (trừ hết khoản cố định) rồi ưu tiên testimonial (nội dung
    chính) > freetext (biến đổi nhì) > reviewer (luôn ngắn, gần cố định).
    """
    has_reviewer = bool(plan.reviewer_name)
    has_freetext = bool(plan.extra_texts)
    has_stars = bool(plan.rating)
    has_qr = bool(plan.qr_code)
    has_cta = bool(plan.cta)
    has_store = bool(plan.store_info)
    has_footer = has_qr or has_cta or has_store
    is_narrow = (width / height) < 0.7
    is_wide = (width / height) >= 1.5

    # ---- Header (hero/subhead, không có container cố định riêng -- giữ nguyên
    # đúng % đã kiểm chứng ổn định, header_height chỉ dùng làm mốc tỉ lệ) ----
    hero_h = header_height * 0.65
    subhead_h = header_height * 0.35 if bool(plan.subhead) else 0.0
    # Trần đã nâng theo đúng tiền lệ l_frame_showcase/before_after_split/
    # lifestyle_corner_pod -- user báo cáo trực tiếp TẤT CẢ cỡ chữ trong
    # customer_feedback_card đều quá nhỏ.
    hero_max_f, hero_min_f = 60.0, 20.0
    ratio = compute_type_scale_ratio(plan.style)
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=22.0)

    # ---- Feedback card ----
    pad_tb = 28.0
    gap_val = 6.0
    stars_reserve = 22.0 if has_stars else 0.0
    reviewer_reserve = 24.0 if has_reviewer else 0.0
    # `footer_reserve` từng CỐ ĐỊNH 90/118px bất kể qr/cta/store có rỗng hay không --
    # lỗi thật user báo cáo (`e2e_04_feedback_form_only`, không có qr/cta/store): ngân
    # sách vẫn bị trừ hết cho 1 `.footer-action-row` rỗng (chỉ còn `margin-top:auto`
    # đẩy 1 div rỗng sát đáy), vừa lãng phí chiều cao đáng lẽ dành cho testimonial/
    # freetext vừa để hở khoảng trống giữa freetext/reviewer và đáy card. QR (90-118px,
    # kích thước kiosk cố định) mới là thành phần thật sự cần khoản dự trữ lớn -- nếu
    # không có QR, chỉ cần đủ 1 hàng cta-btn/store-text (~46px).
    if has_qr:
        footer_reserve = 90.0 if (is_narrow or is_wide) else 118.0
    elif has_footer:
        footer_reserve = 46.0
    else:
        footer_reserve = 0.0
    num_items = sum([True, True, has_reviewer, has_freetext, has_footer])  # stars + testimonial luôn có mặt
    gap_total = max(0, num_items - 1) * gap_val
    avail = max(90.0, card_height - pad_tb - gap_total - stars_reserve - reviewer_reserve - footer_reserve)

    # Đo thật (case badge+5 sao+verified-chip+testimonial+reviewer+3-dòng freetext,
    # ảnh render server user gửi): freetext-block bị kẹt cứng ở SÀN font (14px) dù
    # testimonial (60% avail) dư thừa hẳn -- testimonial 1-2 câu ngắn chỉ cần ~66px
    # trong khi được cấp avail*0.60=98px, còn freetext 3 dòng cần ~70-95px tuỳ font
    # nhưng chỉ được cấp avail*0.40=66px, không đủ ngay cả ở sàn. Testimonial luôn có
    # `testimonial_max_f` (28px) chặn trần riêng nên không cần % lớn để "phòng hờ" --
    # rebalance theo SỐ DÒNG freetext thật, giống pattern đã áp dụng ở
    # l_frame_showcase (num_extra_texts >= 3).
    num_extra_texts = len(plan.extra_texts or [])
    if has_freetext and num_extra_texts >= 3:
        testimonial_h, freetext_h = avail * 0.42, avail * 0.58
    elif has_freetext:
        testimonial_h, freetext_h = avail * 0.60, avail * 0.40
    else:
        testimonial_h, freetext_h = avail * 0.95, 0.0

    # Trần font testimonial từng CỐ ĐỊNH 24px bất kể `avail` còn trống bao nhiêu -- vì
    # `_DENSITY_AWARE_TEMPLATES` bị tắt (xem docs/PLAN_PER_TEMPLATE_BUDGET_ENGINE.md mục 0,
    # KHÔNG được hồi sinh), `.feedback-card` luôn giữ nguyên kích thước tối đa dù nội dung
    # thật rất ngắn (case chỉ có testimonial ngắn, không freetext) -- testimonial dừng ở
    # 24px dù còn thừa hàng trăm px avail, để lại khoảng trống bất hợp lý (user báo cáo
    # trực tiếp qua ảnh case tủ lạnh chiên không dầu). Đây là kỹ thuật "rebalance % TRONG 1
    # zone cố định" ĐƯỢC PHÉP (không phải density-score bị cấm) -- nới trần lên 32px khi
    # testimonial không phải chia sẻ avail với freetext, để nó tự lớn lấp khoảng trống thay
    # vì autofit vẫn tự chọn font NHỎ HƠN nếu nội dung dài không đủ chỗ (không có rủi ro tràn).
    # Trần nâng thêm 1 vòng (user yêu cầu trực tiếp "cỡ chữ quá nhỏ, tăng toàn bộ" sau
    # khi đã nâng 1 lần trong cùng phiên) -- card này có margin mỏng nhất trong toàn bộ
    # 10 template (~1.5px đo được ở docstring trên), chỉ nâng vừa phải + verify lại
    # ngay bằng suite thay vì nâng mạnh như các template có margin rộng hơn.
    testimonial_max_f = 28.0 if has_freetext else 36.0

    return {
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "testimonial": {"max_h": round(testimonial_h, 1), "min_font": 14.0, "max_font": testimonial_max_f},
        "reviewer": {"max_h": 24.0, "min_font": 13.0, "max_font": 16.0},
        "freetext_block": {"max_h": round(freetext_h, 1), "min_font": 14.0, "max_font": 20.0},
        "freetext_pills": {"max_h": round(freetext_h, 1), "min_font": 13.0, "max_font": 18.0},
        "cta": {"max_h": 42.0, "min_font": 13.0, "max_font": 18.0},
        "store": {"max_h": 55.0, "min_font": 14.0, "max_font": 19.0},
    }


def compute_luxury_centered_card_budget(
    plan: TendooCreativePlan,
    card_height: float,
    width: int,
    height: int,
) -> Dict[str, Dict[str, float]]:
    """Tính toán ngân sách cho `luxury_centered_card`'s `.centered-glass-card`.

    SỬA DỨT ĐIỂM lỗi tự tràn pre-existing đã biết từ lâu (`lux_15_16x9_left_minimal`,
    badge+rating+subhead cùng có mặt ở 16:9, `.header-cluster` scrollHeight vượt
    clientHeight). Nguyên nhân: hero/subhead max_h cũ tính trực tiếp = card_height *
    (0.32 hoặc 0.50)/0.18 -- KHÔNG hề trừ badge-capsule (~27px) + rating-stars
    (~18px) + gap nội bộ header-cluster trước khi tính %, nên khi cả 2 phần tử
    cố định này cùng có mặt, hero/subhead ceiling vẫn được cấp như thể chúng không
    tồn tại -- header-cluster tự phình vượt quá không gian thật còn lại.

    Sửa đúng gốc: tính `avail_content` (= card_height trừ padding + footer thật)
    rồi chia cho header-cluster vs message-container (nếu có), sau đó trong nội bộ
    header-cluster mới trừ tiếp badge/stars/gap trước khi chia hero/subhead.
    """
    has_badge = bool(plan.badge)
    has_stars = bool(plan.rating)
    has_subhead = bool(plan.subhead)
    has_message = bool(plan.extra_texts)
    has_qr = bool(plan.qr_code)
    has_cta = bool(plan.cta)
    has_store = bool(plan.store_info)
    has_footer = has_qr or has_cta or has_store
    is_narrow = (width / height) < 0.7
    is_wide = (width / height) >= 1.5

    pad_tb = 26.0 if is_wide else (20.0 if is_narrow else 36.0)
    if has_qr:
        footer_reserve = 100.0 if is_narrow else 115.0
    elif has_footer:
        footer_reserve = 44.0 if is_narrow else 54.0
    else:
        footer_reserve = 0.0
    avail_content = max(110.0, card_height - pad_tb - footer_reserve)

    if has_message:
        header_budget = avail_content * 0.58
        message_h = avail_content * 0.42
    else:
        header_budget = avail_content * 0.95
        message_h = 0.0

    badge_reserve = 27.0 if has_badge else 0.0
    stars_reserve = 18.0 if has_stars else 0.0
    gap_val = 6.0
    num_items = sum([has_badge, has_stars, True, has_subhead])  # hero luôn có mặt
    gap_total = max(0, num_items - 1) * gap_val
    # Dự trữ thêm 12px "line-height rounding slack": đo Playwright thật cho thấy
    # hero-title ở font lớn (multi-line, sát trần ceiling) có scrollHeight thật cao
    # hơn clientHeight/box ~5-7px do bo tròn line-height/font-metrics của Chromium
    # (không phải lỗi tính toán budget) -- khoản chênh này cộng dồn lên
    # `.header-cluster` (không có overflow:hidden riêng) khiến nó tự báo tràn dù
    # tổng ngân sách đã đúng. Trừ thêm 1 khoản nhỏ để hero không bị ép sát trần.
    hero_subhead_avail = max(60.0, header_budget - badge_reserve - stars_reserve - gap_total - 12.0)

    if has_subhead:
        hero_h, subhead_h = hero_subhead_avail * 0.72, hero_subhead_avail * 0.28
    else:
        hero_h, subhead_h = hero_subhead_avail * 0.95, 0.0

    hero_max_f, hero_min_f = 64.0, 18.0
    ratio = compute_type_scale_ratio(plan.style)
    subhead_max_f = _derive_subhead_max_font(hero_max_f, ratio, ceiling=18.0)

    return {
        # `.header-cluster` không có CSS height/max-height cố định (tự phình theo
        # nội dung) -- so sánh scrollHeight với chính clientHeight của nó (như test
        # script vẫn làm cho phần tử không có `data-max-height`) rất DỄ VỠ: đo thật
        # cho thấy font script/cursive (chân bay bổng, dấu tiếng Việt cao) khiến
        # scrollHeight tự nhiên cao hơn clientHeight ~5px dù không hề mất chữ thật
        # (xem ảnh `lux_15_16x9_left_minimal/poster.png` -- không có gì bị cắt).
        # Cấp `header_budget` làm `data-max-height` cho template gắn vào
        # `.header-cluster` để test so sánh với NGÂN SÁCH THẬT đã tính (có margin
        # an toàn) thay vì so với clientHeight cảm tính dễ trôi theo font.
        "header_budget": round(header_budget, 1),
        "hero": {"max_h": round(hero_h, 1), "min_font": hero_min_f, "max_font": hero_max_f},
        "subhead": {"max_h": round(subhead_h, 1), "min_font": 13.0, "max_font": subhead_max_f},
        "message": {"max_h": round(message_h, 1), "min_font": 14.0, "max_font": 19.0},
        "cta": {"max_h": 40.0, "min_font": 13.0, "max_font": 17.0},
        # Trước đây `.store-info-row` (hotline/địa chỉ) là 1 font-size CSS CỐ ĐỊNH
        # 10px/9.5px, hoàn toàn không autofit -- trong khi hero cùng card có thể lên
        # tới 64px (xem `hero_max_f`). User báo cáo trực tiếp "cỡ chữ hotline ở luxury
        # card quá nhỏ". Thêm entry ngân sách này để template gắn `data-autofit` giống
        # hệt cơ chế đã dùng cho hero/subhead/message/cta.
        "store": {"max_h": 60.0, "min_font": 14.0, "max_font": 18.0},
    }


def compute_plan_content_density(plan: TendooCreativePlan, store_items: Optional[list] = None) -> float:
    """Bọc `geometry.py::compute_density_score()` để đọc field trực tiếp từ 1
    TendooCreativePlan -- dùng chung bởi `build_template_html()` (tính CSS) VÀ
    `demo_server.py::_run_variant_pipeline()` (tính mask) để đảm bảo CÙNG 1 density
    cho cùng 1 plan/run (mask và CSS đọc chung `get_zones()`, lệch density giữa 2
    nơi gọi sẽ khiến mask vẽ sai kích thước so với card CSS thật đã render).
    `store_items` truyền vào nếu đã parse sẵn (tránh parse `plan.store_info` 2 lần);
    nếu không truyền, tự parse lại từ `plan.store_info`."""
    if store_items is None:
        store_items = parse_store_info_items(plan.store_info)[:MAX_STORE_ITEMS]
    return compute_density_score(
        has_badge=bool(plan.badge),
        has_subhead=bool(plan.subhead),
        has_rating=bool(plan.rating),
        testimonial_len=len(plan.testimonial or ""),
        num_extra_texts=len(plan.extra_texts or []),
        extra_texts_total_len=sum(len(t) for t in (plan.extra_texts or [])),
        has_cta=bool(plan.cta),
        num_store_items=len(store_items),
        has_qr=bool(plan.qr_code),
    )


def _zone_to_ctx(rect) -> Dict[str, float]:
    x1, y1, x2, y2 = rect
    return {
        "x": x1,
        "y": y1,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": max(0.0, x2 - x1),
        "height": max(0.0, y2 - y1),
    }

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=False,
)


def pil_to_base64_data_uri(img: Image.Image, format: str = "PNG") -> str:
    """Chuyển đổi PIL Image thành Base64 Data URI."""
    import io
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _hex_to_rgb(hex_color: str) -> tuple:
    h = (hex_color or "#D4AF37").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (212, 175, 55)


def _rgb_to_hex(rgb: tuple) -> str:
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def sample_dominant_color(bg_data_uri: str) -> Optional[str]:
    """Lấy màu chủ đạo THẬT từ ảnh nền vừa sinh (quantize xuống 1 bảng màu nhỏ, chọn
    màu phổ biến nhất, loại các màu gần đen/gần trắng thuần vì đó thường là bóng đổ/
    highlight chứ không phải màu chất liệu thật của chủ thể) -- dùng để tô glow chữ
    "bám" đúng tông ảnh thật thay vì chỉ dựa 100% vào theme_color LLM tự chọn."""
    try:
        import io
        header, encoded = bg_data_uri.split(",", 1) if "," in bg_data_uri else ("", bg_data_uri)
        raw = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_small = img.resize((48, 48))
        palette_img = img_small.quantize(colors=6, method=Image.MEDIANCUT)
        palette = palette_img.getpalette()
        color_counts = sorted(palette_img.getcolors(), reverse=True)

        for count, idx in color_counts:
            r, g, b = palette[idx * 3: idx * 3 + 3]
            mx, mn = max(r, g, b), min(r, g, b)
            lightness = (mx + mn) / 2.0
            saturation = 0.0 if mx == mn else (mx - mn) / (255.0 - abs(2 * lightness - 255.0) + 1e-6)
            # Bỏ qua các màu gần đen/gần trắng thuần (bóng đổ/highlight, không phải
            # màu chất liệu chủ thể) -- vẫn giữ màu xám ấm/lạnh nhẹ nếu không còn lựa
            # chọn nào khác tốt hơn (fallback ở cuối vòng lặp).
            if 25 < lightness < 230 and saturation > 0.12:
                return _rgb_to_hex((r, g, b))

        if color_counts:
            _, idx = color_counts[0]
            r, g, b = palette[idx * 3: idx * 3 + 3]
            return _rgb_to_hex((r, g, b))
    except Exception:
        pass
    return None


def sample_bg_luminance_for_zone(
    bg_data_uri: str,
    zone: Optional[Dict[str, float]],
    canvas_w: int,
    canvas_h: int,
    percentile: float = 20.0,
) -> Optional[float]:
    """Đo độ chói THẬT (ITU-R BT.601, đồng nhất công thức với `sample_dominant_color`)
    của đúng vùng ảnh nền nằm dưới 1 zone pixel cụ thể (vd `zones.header`) -- dùng cho
    12/14 template có hero/subhead-title đặt TRỰC TIẾP lên `.bg-layer` (`background:
    transparent`, chỉ có 1 gradient mỏng bám mép + text-shadow, KHÔNG có card/pill đặc
    phía sau). `get_adaptive_palette()` chỉ chọn 1 tông màu categorical
    (`background_tone`) cho CẢ ảnh, nên khi ảnh thật có vùng sáng/tối cục bộ khác nhau
    (vd cửa sổ sáng ở góc trên nhưng mảng tường tối hơn ngay dưới subhead), chữ chọn
    theo tông tổng thể vẫn có thể chìm hẳn ở đúng vùng nó thực sự đứng -- xem báo cáo
    thật `Chuyên viên Kinh doanh B2B` chìm trên nền văn phòng sáng dù `hero-title` phía
    trên vẫn đọc được.

    Dùng PERCENTILE THẤP (mặc định p20) của luminance từng pixel, KHÔNG dùng trung bình
    cộng thô: 1 dòng chữ đơn lẻ (vd hero-title) thường trải NGANG hết chiều rộng zone,
    nên nếu ảnh có 1 mảng tối (tủ bếp, bóng đèn, khung cửa sổ...) nằm CẠNH 1 mảng sáng
    (cửa sổ) ngay dưới CÙNG 1 dòng chữ, trung bình cộng dễ ra mức "sáng vừa" khiến hệ
    thống chọn nhầm chữ tối/navy -- chữ tối đó vẫn đọc ổn ở phần ảnh sáng nhưng CHÌM HẲN
    (đen sì) ở đúng phần ảnh tối cạnh đó (case thật user báo cáo: hero-title dính đúng
    lỗi này khi ảnh có cửa sổ sáng cạnh khu tủ bếp/đèn tối hơn). Percentile thấp trả lời
    đúng câu hỏi cần thiết: "phần TỐI NHẤT của vùng này có đủ sáng để chữ tối vẫn đọc
    được không" -- chỉ chọn chữ tối khi CẢ vùng (kể cả phần tối nhất) đã đủ sáng, mặc
    định thiên về chữ trắng (an toàn hơn khi ảnh có độ tương phản nội bộ cao).

    Trả về None nếu decode lỗi hoặc zone rỗng -- nơi gọi phải tự fallback về palette cũ.
    """
    if not zone:
        return None
    try:
        import io
        header, encoded = bg_data_uri.split(",", 1) if "," in bg_data_uri else ("", bg_data_uri)
        raw = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.size != (canvas_w, canvas_h):
            img = img.resize((canvas_w, canvas_h))
        x1 = max(0, min(canvas_w - 1, int(zone.get("x1", 0))))
        y1 = max(0, min(canvas_h - 1, int(zone.get("y1", 0))))
        x2 = max(x1 + 1, min(canvas_w, int(zone.get("x2", canvas_w))))
        y2 = max(y1 + 1, min(canvas_h, int(zone.get("y2", canvas_h))))
        arr = np.asarray(img.crop((x1, y1, x2, y2)), dtype=np.float64) / 255.0
        lum = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        return float(np.percentile(lum, percentile))
    except Exception:
        return None


def sample_bg_hue_for_zone(
    bg_data_uri: str,
    zone: Optional[Dict[str, float]],
    canvas_w: int,
    canvas_h: int,
) -> Optional[Dict[str, float]]:
    """Đo hue/saturation TRUNG VỊ (không phải trung bình -- 1 vài pixel cực trị dễ kéo
    lệch trung bình cộng của hue là đại lượng góc, trung vị ổn định hơn nhiều) của đúng
    vùng ảnh dưới 1 zone, dùng CÙNG cách crop với `sample_bg_luminance_for_zone` (decode
    lại ảnh riêng, chấp nhận trùng lặp nhỏ để giữ 2 hàm độc lập/dễ kiểm chứng).

    Mục đích: nhuộm NHẸ (subtle tint) màu chữ trắng/đen thuần theo đúng tông màu THẬT
    của ảnh bên dưới (xem `get_zone_adaptive_text_colors`), thay vì luôn ra trắng/navy
    trơ -- yêu cầu trực tiếp của user muốn phần overlay "thẩm mỹ hơn", ăn nhập với ảnh
    thay vì tách biệt hoàn toàn. Trả về hue độ [0..360) + saturation [0..1] của ảnh
    THẬT, KHÔNG áp giới hạn saturation ở đây -- nơi gọi (`get_zone_adaptive_text_colors`)
    tự quyết định mức độ nhuộm an toàn cho từng vai trò chữ (primary/secondary).
    """
    if not zone:
        return None
    try:
        import io
        header, encoded = bg_data_uri.split(",", 1) if "," in bg_data_uri else ("", bg_data_uri)
        raw = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.size != (canvas_w, canvas_h):
            img = img.resize((canvas_w, canvas_h))
        x1 = max(0, min(canvas_w - 1, int(zone.get("x1", 0))))
        y1 = max(0, min(canvas_h - 1, int(zone.get("y1", 0))))
        x2 = max(x1 + 1, min(canvas_w, int(zone.get("x2", canvas_w))))
        y2 = max(y1 + 1, min(canvas_h, int(zone.get("y2", canvas_h))))
        crop = np.asarray(img.crop((x1, y1, x2, y2)), dtype=np.float64) / 255.0
        med_rgb = np.median(crop.reshape(-1, 3), axis=0)
        h, s, _v = colorsys.rgb_to_hsv(float(med_rgb[0]), float(med_rgb[1]), float(med_rgb[2]))
        return {"hue": h * 360.0, "sat": float(s)}
    except Exception:
        return None


def _tinted_hex(hue_deg: float, sat: float, lightness: float, sat_cap: float) -> str:
    """Tạo màu hex HSL từ hue thật của ảnh, ép saturation về mức RẤT nhẹ (`sat_cap`)
    để luôn ra 1 sắc thái tinh tế gần trắng/gần đen chứ không phải màu rực -- an toàn
    cho mọi tông ảnh (kể cả da người, đồ ăn... vốn có hue "rủi ro" nếu nhuộm đậm)."""
    effective_sat = max(0.0, min(sat_cap, sat * 0.5))
    r, g, b = colorsys.hls_to_rgb(hue_deg / 360.0, lightness, effective_sat)
    return _rgb_to_hex((r * 255.0, g * 255.0, b * 255.0))


def get_zone_adaptive_text_colors(
    bg_data_uri: str,
    zone: Optional[Dict[str, float]],
    canvas_w: int,
    canvas_h: int,
    fallback_primary: str,
    fallback_secondary: str,
    fallback_is_dark: bool = True,
    theme_color: Optional[str] = None,
    fallback_accent: Optional[str] = None,
) -> Dict[str, str]:
    """Chọn `text_primary`/`text_secondary` cho chữ đặt TRỰC TIẾP lên ảnh nền (không
    card/scrim đặc phía sau) dựa trên độ chói THẬT đo được của đúng vùng ảnh bên dưới,
    thay vì 1 tông màu categorical áp cho toàn ảnh -- rồi vẫn chạy qua `ensure_contrast`
    (WCAG 2.1) để đảm bảo tỉ lệ tối thiểu với đúng màu nền đo được (không chỉ đảo
    trắng/đen thô theo 1 ngưỡng luminance đơn). Hero-title (chữ to/đậm, được tính là
    "large text" theo WCAG) chỉ cần >=3.0:1; subhead-title (chữ thường, nhỏ hơn nhiều,
    đúng phần tử user báo cáo bị chìm) cần >=4.5:1 nghiêm ngặt hơn.

    `theme_color` (nếu truyền vào): tính thêm `accent` -- màu icon SVG (`on_card_accent`
    cũ) ensure_contrast NGAY với đúng nền cục bộ đo được, thay vì `card_bg_solid` giả
    định (chỉ đúng khi có card đặc phía sau). Dùng khi đã bỏ hẳn nền/scrim của card
    (user yêu cầu trực tiếp: "text đè lên ảnh thì phải nền trong suốt, chỉ dán đè lên
    thôi") -- icon giờ cũng nằm thẳng trên ảnh, cần tự thích ứng y hệt chữ."""
    lum = sample_bg_luminance_for_zone(bg_data_uri, zone, canvas_w, canvas_h)
    if lum is None:
        return {
            "text_primary": fallback_primary,
            "text_secondary": fallback_secondary,
            "is_dark": fallback_is_dark,
            "accent": fallback_accent or theme_color or fallback_primary,
        }

    gray_val = int(round(max(0.0, min(1.0, lum)) * 255))
    approx_bg_hex = _rgb_to_hex((gray_val, gray_val, gray_val))
    zone_is_dark = lum < 0.5

    # Nhuộm NHẸ (subtle tint) chữ trắng/navy thuần theo đúng hue thật của ảnh bên dưới
    # (yêu cầu trực tiếp của user: overlay "thẩm mỹ hơn", ăn nhập với ảnh thay vì tách
    # biệt hoàn toàn) -- saturation ép rất thấp (`_tinted_hex`'s `sat_cap`) nên luôn ra
    # sắc thái tinh tế, không rực, an toàn với mọi tông ảnh. Nếu không đo được hue
    # (decode lỗi) thì rơi về đúng trắng/navy thuần như trước, không đổi hành vi.
    hue_info = sample_bg_hue_for_zone(bg_data_uri, zone, canvas_w, canvas_h)
    if hue_info is not None:
        hue_deg, sat = hue_info["hue"], hue_info["sat"]
        if zone_is_dark:
            base_primary = _tinted_hex(hue_deg, sat, lightness=0.97, sat_cap=0.10)
            base_secondary = _tinted_hex(hue_deg, sat, lightness=0.89, sat_cap=0.16)
        else:
            base_primary = _tinted_hex(hue_deg, sat, lightness=0.11, sat_cap=0.10)
            base_secondary = _tinted_hex(hue_deg, sat, lightness=0.19, sat_cap=0.16)
    else:
        base_primary = "#FFFFFF" if zone_is_dark else "#0F172A"
        base_secondary = "#E2E8F0" if zone_is_dark else "#1E293B"

    return {
        "text_primary": ensure_contrast(base_primary, approx_bg_hex, min_ratio=3.0),
        "text_secondary": ensure_contrast(base_secondary, approx_bg_hex, min_ratio=4.5),
        "accent": ensure_contrast(theme_color, approx_bg_hex, min_ratio=3.0) if theme_color else (fallback_accent or base_primary),
        # `is_dark` (độ tối của ĐÚNG vùng ảnh cục bộ này) -- dùng để tính lại `effect_css`
        # (mục 2.42b trong build_template_html) theo đúng nền cục bộ, KHÔNG theo tông màu
        # categorical của cả ảnh. Lý do bắt buộc: mọi hiệu ứng chữ không dùng background-
        # clip gradient (neon/shadow/embossed/chromatic/neon_bloom/plain_elegant -- 6/10
        # hiệu ứng, gồm cả plain_elegant là default) tự đặt `color:` RIÊNG bên trong CSS
        # của chính nó, ghi đè lên `header_text_primary`/`header_text_secondary` đã tính
        # đúng ở trên vì `{{ effect_css }}` luôn được chèn SAU `color:` trong cùng 1 CSS
        # rule -- nếu effect_css vẫn tính is_dark theo categorical, nó âm thầm phá hỏng
        # toàn bộ fix zone-adaptive (user báo cáo thật: hero "CHỐNG NƯỚC 50M" vẫn navy
        # trên card tối dù `header_text_primary` đã đúng #FFFFFF).
        "is_dark": zone_is_dark,
    }


def _top_band_of_zone(
    zone: Optional[Dict[str, float]],
    band_height_px: float,
    offset_px: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Cắt 1 dải mỏng nằm gần đỉnh của `zone` thay vì lấy nguyên `zone`, để sample
    luminance RIÊNG cho từng phần tử (hero vs subhead) thay vì 1 trung bình chung cho
    cả zone. Lý do: user báo cáo thực tế subhead ("SENIOR AI RESEARCH ENGINEER") vẫn
    chìm dù hero phía trên đã đọc được -- ảnh thật thường có độ sáng khác nhau giữa
    dải hero (đỉnh) và dải ngay dưới nó (subhead), lấy trung bình luminance của CẢ
    `header_zone` (có thể cao bằng nguyên 1 cột full-height ở `split_left/right`,
    `diagonal_slash`, `menu_price_board`) làm nhoè mất sự khác biệt cục bộ đó.
    `offset_px` dịch dải xuống dưới đỉnh zone 1 khoảng (dùng để lấy dải subhead, nằm
    ngay dưới dải hero)."""
    if not zone:
        return None
    y1 = zone["y1"] + offset_px
    y2 = min(zone["y2"], y1 + band_height_px)
    if y2 <= y1:
        return None
    return {"x1": zone["x1"], "y1": y1, "x2": zone["x2"], "y2": y2}


def _bottom_band_of_zone(
    zone: Optional[Dict[str, float]],
    band_height_px: float,
) -> Optional[Dict[str, float]]:
    """Cắt 1 dải mỏng nằm gần ĐÁY của `zone` -- dùng để sample luminance RIÊNG cho
    footer-row (cta/qr/store) của các template "card/pod" (`CARD_ZONE_KEY_BY_TEMPLATE`),
    thay vì lấy 1 màu trung bình cho CẢ card rồi áp cho cả badge (đỉnh) lẫn footer
    (đáy). User báo cáo thật: badge/store-text bị chìm vào nền (trắng trên trắng, hoặc
    ngược lại) dù phần đầu card đọc được -- CÙNG NGUYÊN NHÂN đã sửa cho hero/subhead
    ở `_top_band_of_zone` (ảnh thật có độ sáng khác nhau giữa đỉnh và đáy 1 vùng card
    cao, lấy trung bình làm nhoè mất khác biệt cục bộ), chỉ khác trục (trên/dưới của
    CẢ card, không phải trong nội bộ header)."""
    if not zone:
        return None
    y2 = zone["y2"]
    y1 = max(zone["y1"], y2 - band_height_px)
    if y2 <= y1:
        return None
    return {"x1": zone["x1"], "y1": y1, "x2": zone["x2"], "y2": y2}


# Zone dùng làm mốc đo độ chói nền THẬT phía sau hero/subhead-title của từng template
# -- CHỈ khai báo cho 12/14 template có header đặt trực tiếp lên `.bg-layer` (xác nhận
# qua audit `background: transparent;` bao quanh `.hero-title`/`.hero-top-title` trong
# từng file template, xem chi tiết ở docstring `sample_bg_luminance_for_zone`).
# `luxury_centered_card` và `lifestyle_corner_pod` dùng `CARD_ZONE_KEY_BY_TEMPLATE`
# riêng bên dưới (không phải header) vì hero/subhead của 2 template này nằm trong toàn
# bộ vùng "card"/"pod", không phải 1 dải mỏng ở đỉnh.
HEADER_ZONE_KEY_BY_TEMPLATE: Dict[str, str] = {
    "customer_feedback_card": "header",
    "recruitment_board": "header",
    "step_process_roadmap": "header",
    "grand_opening_banner": "header",
    "sandwich_top_heavy": "top",
    "sandwich_bottom_heavy": "bottom",
    "before_after_split": "bottom",
    "split_left": "content",
    "split_right": "content",
    "menu_price_board": "content",
    "diagonal_slash": "content",
    "l_frame_showcase": "top_cluster",
}

# Zone dùng làm mốc đo độ chói nền THẬT cho TOÀN BỘ nội dung bên trong "card" của 5
# template dạng card/pod độc lập -- trước đây các card này có nền rgba đặc + backdrop-
# blur (giả định LUÔN đủ tối để chữ trắng cứng an toàn bất kể ảnh gì), nhưng user yêu
# cầu trực tiếp và nhắc lại nhiều lần: "text đè lên ảnh thì text phải nền trong suốt,
# chỉ dán đè lên thôi" -- đã bỏ hẳn nền/backdrop-filter của các card này (xem
# template.html từng file), nên MỌI chữ/icon bên trong giờ cũng nằm thẳng trên ảnh y
# hệt hero/subhead, cần cùng cơ chế zone-adaptive percentile-based, không còn được
# phép giả định "card luôn tối".
CARD_ZONE_KEY_BY_TEMPLATE: Dict[str, str] = {
    "customer_feedback_card": "card",
    "luxury_centered_card": "card",
    "lifestyle_corner_pod": "pod",
    "recruitment_board": "board",
    "step_process_roadmap": "platform",
    # `.bottom-info-pod` (extra-pill/store-item) không có nền riêng đủ tối, nằm thẳng
    # lên ảnh y hệt các card khác -- trước đây KHÔNG khai báo ở đây nên
    # `card_footer_text_secondary` bị tính theo fallback toàn cục thay vì sample đúng
    # vùng ảnh thật bên dưới, và template.html còn hardcode #E2E8F0 riêng (đã sửa) --
    # user báo cáo trực tiếp: chữ hoà mất vào nền sáng/pastel. Dùng chung zone "bottom"
    # với HEADER_ZONE_KEY_BY_TEMPLATE (cùng 1 khối .bottom-info-pod chứa cả 2).
    "before_after_split": "bottom",
}


def blend_hex_colors(color_a: str, color_b: str, weight_b: float = 0.5) -> str:
    """Trộn 2 màu hex theo trọng số (0.0 = 100% color_a, 1.0 = 100% color_b)."""
    weight_b = max(0.0, min(1.0, weight_b))
    ra, ga, ba = _hex_to_rgb(color_a)
    rb, gb, bb = _hex_to_rgb(color_b)
    return _rgb_to_hex((
        ra * (1 - weight_b) + rb * weight_b,
        ga * (1 - weight_b) + gb * weight_b,
        ba * (1 - weight_b) + bb * weight_b,
    ))


def build_template_html(
    plan: TendooCreativePlan,
    bg_data_uri: str,
    width: int,
    height: int,
    palette_override: Optional[Dict[str, str]] = None,
) -> str:
    """Biên dịch TendooCreativePlan thành chuỗi HTML5/CSS3 tự chứa 100%."""
    tpl_name = plan.template if plan.template in TEMPLATE_CATALOG else "sandwich_top_heavy"
    
    # 1. Resolve Font Unicode Tiếng Việt
    canonical_font, font_face_css, headline_font_css = resolve_font(
        font_key=plan.style.font,
        text_content=f"{plan.hero} {plan.subhead or ''} {plan.cta or ''} {plan.badge or ''} {plan.testimonial or ''}",
    )

    # 1.5. Màu Glow "bám" đúng tông ảnh nền THẬT: lấy màu chủ đạo thật từ bg_data_uri
    # (không phải đoán/hard-code) rồi trộn 50/50 với theme_color LLM chọn -- glow chữ
    # (neon/neon_bloom) nhờ vậy cảm giác "thuộc về" đúng tấm ảnh thay vì màu tách biệt
    # hoàn toàn khỏi cảnh. CTA/badge/pill (`theme_color`/`palette`)
    # CỐ TÌNH giữ nguyên màu gốc LLM/form chọn -- không đổi thương hiệu/độ tương phản
    # đã được chọn có chủ đích, chỉ riêng lớp "glow" mới bám ảnh thật.
    sampled_bg_color = sample_dominant_color(bg_data_uri)
    glow_color = blend_hex_colors(plan.style.theme_color, sampled_bg_color, weight_b=0.5) if sampled_bg_color else plan.style.theme_color

    # 2. Resolve Adaptive Palette & Text Effect CSS (Tự động thích ứng độ chói WCAG 2.1)
    palette = palette_override or get_adaptive_palette(
        background_tone=plan.style.background_tone,
        theme_color=plan.style.theme_color,
    )
    is_dark = bool(palette.get("is_dark", True))
    effect_css = get_effect_css(
        effect_name=plan.style.text_effect,
        theme_color=glow_color,
        is_dark=is_dark,
        text_color=palette.get("text_primary"),
    )

    # 2.35. store_items tính SỚM (trước cả zones) vì bước 2.4 (density-aware zone
    # sizing) cần biết số lượng store_items thật để ước lượng mật độ nội dung.
    store_items = parse_store_info_items(plan.store_info)[:MAX_STORE_ITEMS]

    # 2.4. Hình học vùng chữ THẬT của template này (xem geometry.py), tính SỚM ở đây
    # (không chỉ ở bước 5 bên dưới) vì bước 2.42 (màu chữ zone-adaptive) cần các zone
    # này để biết đúng vùng ảnh nào nằm dưới hero/subhead.
    #
    # `content_density` (xem geometry.py::compute_density_score) co giãn zone của 1
    # số template dạng "card/pod" độc lập (luxury_centered_card, customer_feedback_
    # card, lifestyle_corner_pod, recruitment_board, l_frame_showcase) theo ĐÚNG
    # lượng nội dung thật -- tránh khoảng trắng rỗng giữa card khi ít chữ (case thật
    # user báo cáo: luxury_centered_card badge+hero+subhead+2 bullet+cta để hở khoảng
    # trống lớn giữa bullet-list và CTA vì card cao cố định bất kể nội dung).
    content_density = compute_plan_content_density(plan, store_items=store_items)
    zones = {
        name: _zone_to_ctx(rect)
        for name, rect in get_zones(
            tpl_name,
            width,
            height,
            orientation=plan.orientation,
            density=content_density,
            has_qr=bool(plan.qr_code),
            has_footer=bool(plan.qr_code or plan.cta or plan.store_info),
            has_message=bool(plan.extra_texts),
            has_freetext=bool(plan.extra_texts),
        ).items()
    }

    # 2.42. Màu chữ THẬT cho hero/subhead-title đặt trực tiếp lên ảnh nền (xem
    # `get_zone_adaptive_text_colors` -- sửa lỗi thật: chữ chọn theo 1 tông màu
    # categorical cho cả ảnh vẫn có thể chìm ở đúng vùng cục bộ nó đứng). Template
    # không có mặt trong `HEADER_ZONE_KEY_BY_TEMPLATE` (hero nằm trong card/pod nền
    # đặc) giữ nguyên 100% `palette.text_primary/secondary` cũ.
    header_zone_key = HEADER_ZONE_KEY_BY_TEMPLATE.get(tpl_name)
    header_zone = zones.get(header_zone_key) if header_zone_key else None
    # Sample RIÊNG dải hero (đỉnh zone) và dải subhead (ngay dưới) thay vì 1 trung
    # bình chung cho cả `header_zone` -- xem lý do đầy đủ ở docstring `_top_band_of_zone`.
    hero_band = _top_band_of_zone(header_zone, band_height_px=height * 0.09)
    subhead_band = _top_band_of_zone(header_zone, band_height_px=height * 0.05, offset_px=height * 0.09)
    hero_zone_colors = get_zone_adaptive_text_colors(
        bg_data_uri=bg_data_uri,
        zone=hero_band or header_zone,
        canvas_w=width,
        canvas_h=height,
        fallback_primary=palette["text_primary"],
        fallback_secondary=palette["text_secondary"],
        fallback_is_dark=is_dark,
    )
    subhead_zone_colors = get_zone_adaptive_text_colors(
        bg_data_uri=bg_data_uri,
        zone=subhead_band or header_zone,
        canvas_w=width,
        canvas_h=height,
        fallback_primary=palette["text_primary"],
        fallback_secondary=palette["text_secondary"],
        fallback_is_dark=is_dark,
    )
    header_colors = {
        "text_primary": hero_zone_colors["text_primary"],
        "text_secondary": subhead_zone_colors["text_secondary"],
    }

    # 2.42b. `effect_css` gốc (dòng ~1296) tính is_dark theo tông CATEGORICAL của cả
    # ảnh -- nhưng vì `{{ effect_css }}` luôn được chèn SAU `color:` trong CSS của
    # `.hero-title`/`.hero-top-title`, và 6/10 hiệu ứng (neon/shadow/embossed/chromatic/
    # neon_bloom/plain_elegant -- gồm cả default) tự đặt `color:` riêng, nó ÂM THẦM GHI
    # ĐÈ lên toàn bộ fix zone-adaptive ở trên (user báo cáo thật: "CHỐNG NƯỚC 50M" vẫn
    # navy trên card tối). Cần 2 biến thể effect_css RIÊNG:
    # - `header_effect_css`: cho 12 template hero đặt TRỰC TIẾP lên ảnh (transparent bg)
    #   -- tính lại is_dark theo ĐÚNG độ tối cục bộ đã sample (`hero_zone_colors`).
    # - `card_effect_css`: cho `luxury_centered_card`/`lifestyle_corner_pod`.
    header_effect_css = get_effect_css(
        effect_name=plan.style.text_effect,
        theme_color=glow_color,
        is_dark=hero_zone_colors.get("is_dark", is_dark),
        text_color=hero_zone_colors.get("text_primary"),
    )

    # Hiệu ứng subtitle/subhead: loại bỏ hoàn toàn blur 6px gây mờ nét chữ
    # Nền tối: đổ bóng 1px nhẹ, sắc nét. Nền sáng: chữ tối tự nhiên, text-shadow: none
    subhead_is_dark = subhead_zone_colors.get("is_dark", is_dark)
    if subhead_is_dark:
        subhead_effect_css = "text-shadow: 0 1px 3px rgba(0, 0, 0, 0.65);"
    else:
        subhead_effect_css = "text-shadow: none;"

    # 2.42c. Màu chữ/icon THẬT cho TOÀN BỘ nội dung bên trong "card" của 5 template
    # dạng card/pod độc lập -- các card này đã bỏ hẳn nền rgba + backdrop-filter (yêu
    # cầu trực tiếp của user, nhắc lại nhiều lần: "text đè lên ảnh thì phải nền trong
    # suốt, chỉ dán đè lên thôi"), nên KHÔNG còn được phép giả định "card luôn tối" như
    # trước -- mọi chữ/icon bên trong giờ nằm thẳng trên ảnh, cần sample đúng vùng ảnh
    # bên dưới CHÍNH zone của card đó (xem CARD_ZONE_KEY_BY_TEMPLATE).
    #
    # ĐÍNH CHÍNH (user báo cáo thật: chữ trắng hoà vào nền trắng, badge chìm màu):
    # "dùng chung 1 bộ màu cho cả card" (comment cũ) là SAI với card cao -- ảnh thật
    # có độ sáng khác nhau giữa ĐỈNH card (badge/hero) và ĐÁY card (footer-row: cta/
    # qr/store), y hệt lý do đã sửa cho hero/subhead ở `_top_band_of_zone`. Sample
    # THÊM 1 dải riêng ở đáy card (`_bottom_band_of_zone`) cho footer-row, thay vì chỉ
    # 1 màu `card_colors` dùng chung cho mọi thứ.
    card_zone_key = CARD_ZONE_KEY_BY_TEMPLATE.get(tpl_name)
    card_zone = zones.get(card_zone_key) if card_zone_key else None
    card_zone_colors = get_zone_adaptive_text_colors(
        bg_data_uri=bg_data_uri,
        zone=card_zone,
        canvas_w=width,
        canvas_h=height,
        fallback_primary=palette["text_primary"],
        fallback_secondary=palette["text_secondary"],
        fallback_is_dark=is_dark,
        theme_color=plan.style.theme_color,
        fallback_accent=palette.get("on_card_accent"),
    )
    card_colors = {
        "text_primary": card_zone_colors["text_primary"],
        "text_secondary": card_zone_colors["text_secondary"],
        "accent": card_zone_colors["accent"],
    }
    card_effect_css = get_effect_css(
        effect_name=plan.style.text_effect,
        theme_color=glow_color,
        is_dark=card_zone_colors.get("is_dark", True),
        text_color=card_zone_colors.get("text_primary"),
    )

    footer_band = _bottom_band_of_zone(card_zone, band_height_px=height * 0.12)
    card_footer_zone_colors = get_zone_adaptive_text_colors(
        bg_data_uri=bg_data_uri,
        zone=footer_band or card_zone,
        canvas_w=width,
        canvas_h=height,
        fallback_primary=palette["text_primary"],
        fallback_secondary=palette["text_secondary"],
        fallback_is_dark=is_dark,
        theme_color=plan.style.theme_color,
        fallback_accent=palette.get("on_card_accent"),
    )
    card_footer_colors = {
        "text_primary": card_footer_zone_colors["text_primary"],
        "text_secondary": card_footer_zone_colors["text_secondary"],
        "accent": card_footer_zone_colors["accent"],
    }

    # Badge/freetext-block (và mọi "chip" tương tự) CỐ TÌNH giữ lại nền riêng (đen
    # 35-65% opacity, ĐƯỢC PHÉP theo yeu_cau_templates.txt mục 1 -- pill/card có nền
    # không phải scrim) -- khác `.bullet-gem`/`.store-item svg` (không nền, nằm thẳng
    # lên ảnh, cần contrast theo ẢNH). Lỗi thật user báo cáo (luxury_centered_card's
    # badge-capsule): dùng `card_accent` (contrast theo ẢNH phía sau) cho chữ/viền của
    # 1 phần tử có nền ĐEN CỐ ĐỊNH của riêng nó -- khi ảnh phía sau sáng, card_accent
    # chọn màu TỐI để nổi trên ảnh sáng, nhưng màu tối đó lại đặt trên nền đen riêng
    # của badge -> chữ chìm vào nền. Phải contrast với nền THẬT của badge (đen ~65%),
    # không phải ảnh. `ensure_contrast` với `#000000` xấp xỉ đúng nền tối cố định đó.
    card_chrome_text = ensure_contrast("#FFFFFF", "#000000", min_ratio=4.5)
    card_chrome_accent = ensure_contrast(plan.style.theme_color, "#000000", min_ratio=3.0)

    # 3. Rating Star SVG & QR Code Component (Thực tế quét được)
    stars_svg = render_star_rating_svg(count=plan.rating or 5, fill_color=plan.style.theme_color) if plan.rating else ""
    from tendoo_v3.qr import render_scannable_qr_component
    # Tự động scale QR size theo tỷ lệ khung hình để không bao giờ bị cắt ở 16:9 hoặc 9:16
    if width <= 600 or height <= 600:
        qr_size = 58
    elif width / height >= 1.5:
        qr_size = 62
    elif tpl_name == "sandwich_bottom_heavy":
        qr_size = 62
    elif tpl_name in ("step_process_roadmap", "recruitment_board"):
        qr_size = 70
    else:
        qr_size = 78
    qr_svg = render_scannable_qr_component(
        data=plan.qr_code,
        label=plan.qr_label or "QUÉT MÃ NGAY",
        theme_color=plan.style.theme_color,
        size_px=qr_size,
    ) if plan.qr_code else ""

    # 4. Semantic Pills & Tags (tự động gắn icon theo ngữ nghĩa câu từ tiếng Việt) --
    # cắt trần số lượng TRƯỚC khi render (xem MAX_EXTRA_TEXTS ở đầu file).
    extra_pills = []
    for item in plan.extra_texts[:MAX_EXTRA_TEXTS]:
        icon_name = infer_semantic_icon(item)
        icon_svg = get_icon_svg(icon_name)
        extra_pills.append({"text": item, "icon_name": icon_name, "icon_svg": icon_svg})

    tag_left_icon = get_icon_svg(infer_semantic_icon(plan.tag_left)) if plan.tag_left else ""
    tag_right_icon = get_icon_svg(infer_semantic_icon(plan.tag_right)) if plan.tag_right else ""
    cta_icon = get_icon_svg("arrow_right") if plan.cta else ""
    # store_items đã tính sớm ở bước 2.35 (để phục vụ content_density), dùng lại ở đây.
    steps = plan.steps[:MAX_STEPS]

    # 5. `zones` (hình học vùng chữ THẬT của template này, xem geometry.py -- nguồn
    # chân lý DUY NHẤT dùng chung với mask_engine.py) đã được tính sớm hơn ở bước 2.4.
    # Mỗi zone injected dưới dạng dict {x1,y1,x2,y2,width,
    # height} px, template dùng `zones.<ten_zone>.height` v.v. làm `data-max-height`/
    # `data-max-width` cho phần tử autofit thay vì đoán qua `parentElement.clientHeight`
    # (sai với mọi phần tử position:absolute có DOM-parent lớn hơn vùng thị giác thật).

    # Phân loại tỉ lệ và ngân sách dải đỉnh
    is_portrait_narrow = (width / height) < 0.7  # 9:16
    is_landscape_wide = (width / height) >= 1.5   # 16:9
    top_height = zones.get("top", {}).get("height", 246.0)
    top_budget = compute_sandwich_top_budget(plan, top_height, width, height)
    bottom_height = zones.get("bottom", {}).get("height", 286.0)
    if tpl_name == "sandwich_top_heavy":
        bottom_budget = compute_sandwich_top_heavy_bottom_budget(plan, bottom_height, width, height)
    else:
        bottom_budget = compute_sandwich_bottom_budget(plan, bottom_height, width, height)

    split_budget = None
    if tpl_name in ("split_left", "split_right") and "content" in zones:
        split_budget = compute_split_budget(
            plan=plan,
            col_height=zones["content"]["height"],
            col_width=zones["content"]["width"],
            width=width,
            height=height,
        )

    step_budget = compute_step_process_roadmap_budget(
        plan=plan,
        header_height=zones.get("header", {}).get("height", height * 0.15),
        platform_height=zones.get("platform", {}).get("height", height * 0.18),
        width=width,
        height=height,
    )

    open_budget = compute_grand_opening_banner_budget(
        plan=plan,
        header_height=zones.get("header", {}).get("height", height * 0.30),
        width=width,
        height=height,
    )

    recruit_budget = compute_recruitment_board_budget(
        plan=plan,
        header_height=zones.get("header", {}).get("height", height * 0.15),
        board_height=zones.get("board", {}).get("height", height * 0.20),
        width=width,
        height=height,
    )

    menu_budget = compute_menu_price_board_budget(
        plan=plan,
        content_height=zones.get("content", {}).get("height", height),
        width=width,
        height=height,
    )

    diagonal_budget = compute_diagonal_slash_budget(
        plan=plan,
        content_height=zones.get("content", {}).get("height", height),
        width=width,
        height=height,
    )

    before_after_budget = compute_before_after_budget(
        plan=plan,
        bottom_height=zones.get("bottom", {}).get("height", height * 0.33),
        width=width,
        height=height,
    )

    lframe_budget = compute_l_frame_showcase_budget(
        plan=plan,
        top_cluster_height=zones.get("top_cluster", {}).get("height", height * 0.40),
        width=width,
        height=height,
    )

    lifestyle_budget = compute_lifestyle_corner_pod_budget(
        plan=plan,
        pod_height=zones.get("pod", {}).get("height", height * 0.48),
        width=width,
        height=height,
    )

    feedback_budget = compute_customer_feedback_budget(
        plan=plan,
        header_height=zones.get("header", {}).get("height", height * 0.14),
        card_height=zones.get("card", {}).get("height", height * 0.32),
        width=width,
        height=height,
    )

    luxury_budget = compute_luxury_centered_card_budget(
        plan=plan,
        card_height=zones.get("card", {}).get("height", height * 0.72),
        width=width,
        height=height,
    )

    # Phân loại FreeText: dạng Bullet / Block văn bản vs Pills ngắn.
    # Trước đây dùng any() -- chỉ 1 dòng dài/nhiều từ trong cả mảng cũng ép TOÀN BỘ
    # mảng (kể cả các dòng ngắn khác) sang block mode, dù phần lớn dòng còn lại đủ
    # ngắn để làm pill (case thật: "Mua 1 Tặng 1 toàn bộ menu cà phê" (9 từ) kéo cả
    # 2 dòng ngắn "Tặng ly giữ nhiệt..."/"Vòng quay may mắn..." xuống khối bullet dọc
    # dù bản thân 2 dòng đó thừa sức làm pill ngang). Đổi sang: bullet marker tường
    # minh (*, -, •) vẫn là tín hiệu CHỦ Ý rõ ràng -- chỉ cần 1 dòng có là đủ; còn
    # ngưỡng độ dài chỉ trigger block mode khi ĐA SỐ (> 50%) dòng đều dài, phản ánh
    # đúng bản chất nội dung tổng thể thay vì bị 1 dòng ngoại lệ chi phối.
    is_freetext_block = False
    if plan.extra_texts:
        has_explicit_bullet = any(t.strip().startswith(("*", "-", "•")) for t in plan.extra_texts)
        num_long = sum(1 for t in plan.extra_texts if len(t) > 35 or len(t.split()) > 6)
        is_freetext_block = has_explicit_bullet or (num_long > len(plan.extra_texts) / 2)

    # 6. Nạp template Jinja2
    template_file = f"{tpl_name}/template.html"
    try:
        jinja_tpl = _JINJA_ENV.get_template(template_file)
    except Exception as e:
        jinja_tpl = _JINJA_ENV.get_template("sandwich_top_heavy/template.html")

    # Unified budget mapping for all templates
    current_budget = {}
    if tpl_name == "sandwich_top_heavy":
        current_budget = {
            **top_budget,
            "cta": bottom_budget.get("cta", {}),
            "store": bottom_budget.get("store", {}),
            "top": top_budget,
            "bottom": bottom_budget,
        }
    elif tpl_name == "sandwich_bottom_heavy":
        current_budget = {
            **bottom_budget,
            "badge": top_budget.get("badge", {}),
            "top": top_budget,
            "bottom": bottom_budget,
        }
    elif tpl_name in ("split_left", "split_right"):
        current_budget = split_budget or {}
    elif tpl_name == "step_process_roadmap":
        current_budget = step_budget
    elif tpl_name == "grand_opening_banner":
        current_budget = open_budget
    elif tpl_name == "recruitment_board":
        current_budget = recruit_budget
    elif tpl_name == "menu_price_board":
        current_budget = menu_budget
    elif tpl_name == "diagonal_slash":
        current_budget = diagonal_budget
    elif tpl_name == "before_after_split":
        current_budget = before_after_budget
    elif tpl_name == "l_frame_showcase":
        current_budget = lframe_budget
    elif tpl_name == "lifestyle_corner_pod":
        current_budget = lifestyle_budget
    elif tpl_name == "customer_feedback_card":
        current_budget = feedback_budget
    elif tpl_name == "luxury_centered_card":
        current_budget = luxury_budget
    else:
        current_budget = top_budget

    # 7. Render context
    context = {
        "budget": current_budget,
        "width": width,
        "height": height,
        "bg_data_uri": bg_data_uri,
        "font_face_css": font_face_css,
        "headline_font_css": headline_font_css,
        "theme_color": plan.style.theme_color,
        "glow_color": glow_color,
        "effect_css": effect_css,
        "header_effect_css": header_effect_css,
        "card_effect_css": card_effect_css,
        "subhead_effect_css": subhead_effect_css,
        "header_is_dark": hero_zone_colors.get("is_dark", is_dark),
        "subhead_is_dark": subhead_is_dark,
        "palette": palette,
        "header_text_primary": header_colors["text_primary"],
        "header_text_secondary": header_colors["text_secondary"],
        "card_text_primary": card_colors["text_primary"],
        "card_text_secondary": card_colors["text_secondary"],
        "card_accent": card_colors["accent"],
        "card_footer_text_primary": card_footer_colors["text_primary"],
        "card_footer_text_secondary": card_footer_colors["text_secondary"],
        "card_footer_accent": card_footer_colors["accent"],
        "card_chrome_text": card_chrome_text,
        "card_chrome_accent": card_chrome_accent,
        "hero": plan.hero,
        "hero_parts": plan.hero_parts,
        "subhead": plan.subhead,
        "badge": plan.badge,
        "tag_left": plan.tag_left,
        "tag_left_icon": tag_left_icon,
        "tag_right": plan.tag_right,
        "tag_right_icon": tag_right_icon,
        "stars_svg": stars_svg,
        "bullet_svg": BULLET_SPARKLE_SVG,
        "extra_texts": plan.extra_texts[:MAX_EXTRA_TEXTS],
        "extra_pills": extra_pills,
        "is_freetext_block": is_freetext_block,
        "is_portrait_narrow": is_portrait_narrow,
        "is_landscape_wide": is_landscape_wide,
        "top_budget": top_budget,
        "bottom_budget": bottom_budget,
        "split_budget": split_budget,
        "step_budget": step_budget,
        "open_budget": open_budget,
        "recruit_budget": recruit_budget,
        "menu_budget": menu_budget,
        "diagonal_budget": diagonal_budget,
        "before_after_budget": before_after_budget,
        "lframe_budget": lframe_budget,
        "lifestyle_budget": lifestyle_budget,
        "feedback_budget": feedback_budget,
        "luxury_budget": luxury_budget,
        "type_scale_ratio": compute_type_scale_ratio(plan.style),
        "cta": plan.cta,
        "cta_icon": cta_icon,
        "store_info": plan.store_info,
        "store_items": store_items,
        "qr_code": plan.qr_code,
        "qr_label": plan.qr_label,
        "qr_svg": qr_svg,
        "testimonial": plan.testimonial,
        "reviewer_name": plan.reviewer_name,
        "steps": steps,
        "orientation": plan.orientation or "bottom_left",
        "zones": zones,
        "autofit_js": COMMON_AUTOFIT_JS,
    }

    return jinja_tpl.render(**context)


def render_plan_to_poster(
    plan: TendooCreativePlan,
    bg_data_uri: str,
    output_image_path: Path | str,
    width: int,
    height: int,
    palette_override: Optional[Dict[str, str]] = None,
) -> Tuple[Path, str]:
    """Render plan thành ảnh poster PNG qua Playwright Chromium.
    Trả về (output_path, html_content).
    """
    html_content = build_template_html(
        plan=plan,
        bg_data_uri=bg_data_uri,
        width=width,
        height=height,
        palette_override=palette_override,
    )
    
    out_file = Path(output_image_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    PosterRenderer.render(
        html_content=html_content,
        output_image_path=out_file,
        width=width,
        height=height,
    )
    
    return out_file, html_content
