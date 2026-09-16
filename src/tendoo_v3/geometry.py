"""
src/tendoo_v3/geometry.py

NGUỒN CHÂN LÝ DUY NHẤT cho hình học "vùng chữ" (content zone) của mỗi template.
Trước bản này, `mask_engine.py` tự đoán % để vẽ mask VÀ mỗi `template.html` tự
khai báo % CSS riêng để đặt khối chữ thật -- hai nơi độc lập, dễ lệch nhau (đã
đo thấy thật: sandwich_top_heavy's pill-row nằm ngoài mask 97-100%,
recruitment_board's cả thân bài không được mask bảo vệ...). Giờ cả
`mask_engine.py` (vẽ mask) và `renderer.py` (tính `data-max-width/height` cho
script autofit) đều đọc CHUNG bảng này -- không còn 2 nguồn số tự ý trôi dạt.

NGUYÊN TẮC (2026-09-15, theo yêu cầu trực tiếp): mask phải LIÊN TỤC -- 1 khối
lớn gọn gàng mỗi vùng, không vụn thành nhiều mảnh nhỏ rời rạc bám theo từng
dòng chữ (đó là hướng continuous-mask, KHÔNG quay lại đo-per-block như v1/v2).
Kích thước mỗi khối được CHỌN cho đủ đẹp -- đủ chỗ cho nội dung thật thở,
không tham lam chiếm hết khung, cho sản phẩm/bối cảnh không gian riêng -- rồi
CHỮ TỰ CO (autofit, xem styles.py::COMMON_AUTOFIT_JS) để vừa khít bên trong,
thay vì khối phải phình ra đuổi theo chữ.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

Rect = Tuple[float, float, float, float]  # (x1, y1, x2, y2) tính bằng px


def _frac(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> Rect:
    return (x1 * w, y1 * h, x2 * w, y2 * h)


# ==================================================================================
# 1 hàm mỗi template, trả về {tên_vùng: Rect (px)} -- khớp CHÍNH XÁC với vị trí
# CSS thật đã đặt trong `templates/<template>/template.html` (đã đối chiếu số
# đo Chromium thật, xem scripts/check_tendoo_v3_mask_overflow.py).
# ==================================================================================

def _split_left(w: float, h: float) -> Dict[str, Rect]:
    # Khớp .split-column-left { width: 38% } -- thêm biên nhỏ (40%) làm dư an toàn.
    return {"content": _frac(0.0, 0.0, 0.40, 1.0, w, h)}


def _split_right(w: float, h: float) -> Dict[str, Rect]:
    return {"content": _frac(0.60, 0.0, 1.0, 1.0, w, h)}


def _sandwich_top_heavy(w: float, h: float) -> Dict[str, Rect]:
    # Dải trên gọn gàng: 0 -> 28% (badge + hero 1-2 dòng + subhead + 1 hàng pill).
    # Dải đáy: 86% -> 100% (cao 14%). Khoảng giữa (28% đến 86% = 58% canvas) mở cho DiT.
    return {
        "top": _frac(0.0, 0.0, 1.0, 0.28, w, h),
        "bottom": _frac(0.0, 0.86, 1.0, 1.0, w, h),
    }


def _sandwich_bottom_heavy(w: float, h: float) -> Dict[str, Rect]:
    # .sandwich-top-bar height:12% (chỉ badge+store nhỏ), .sandwich-bottom-platform
    # height:28% (hero+subhead+pill+cta) -- KHÁC hẳn top_heavy, không được dùng
    # chung 1 preset như trước (đó chính là lý do hero từng chỉ được mask 7%).
    return {
        "top": _frac(0.0, 0.0, 1.0, 0.13, w, h),
        "bottom": _frac(0.0, 0.72, 1.0, 1.0, w, h),
    }


def _before_after_split(w: float, h: float) -> Dict[str, Rect]:
    # 2 nhãn góc BEFORE/AFTER: kích thước capsule thật (~padding 8px 18px,
    # font 13px) -- ước lượng hộp rộng rãi quanh vị trí top:28px/left|right:28px
    # đã khai báo trong CSS, cho dư biên chứ không đo pixel-perfect.
    label_w = min(190.0, 0.30 * w)
    label_h = min(56.0, 0.10 * h)
    return {
        "label_before": (16.0, 16.0, 16.0 + label_w, 16.0 + label_h),
        "label_after": (w - 16.0 - label_w, 16.0, w - 16.0, 16.0 + label_h),
        # .bottom-info-pod height:25% -- khớp đúng.
        "bottom": _frac(0.0, 0.75, 1.0, 1.0, w, h),
    }


def _luxury_centered_card(w: float, h: float) -> Dict[str, Rect]:
    return {}  # Zero-mask theo đúng thiết kế (nền mờ ảo thuần, không sản phẩm).


def _lifestyle_corner_pod(w: float, h: float, orientation: str = "bottom_left") -> Dict[str, Rect]:
    # Khớp .corner-pod { width: min(480px, 46%); max-height: calc(100% - 56px) }
    box_w = min(480.0, 0.46 * w)
    box_h = h - 56.0
    if orientation in ("bottom_right", "right"):
        return {"pod": (w - 28.0 - box_w, 28.0, w - 28.0, 28.0 + box_h)}
    if orientation == "top_left":
        return {"pod": (28.0, 28.0, 28.0 + box_w, 28.0 + box_h)}
    if orientation == "top_right":
        return {"pod": (w - 28.0 - box_w, 28.0, w - 28.0, 28.0 + box_h)}
    # bottom_left (mặc định)
    return {"pod": (28.0, h - 28.0 - box_h, 28.0 + box_w, h - 28.0)}


def _recruitment_board(w: float, h: float) -> Dict[str, Rect]:
    # .recruitment-header ở trên đỉnh: max-width: 85%, cao 18% (badge + hero + subhead)
    # .board-card ở đáy: cao 28% (từ 70% đến 98%)
    # Khoảng giữa từ y=0.18 đến y=0.70 (52% chiều cao poster) HOÀN TOÀN MỞ CHO CHỦ THỂ DiT!
    return {
        "header": _frac(0.0, 0.0, 0.85, 0.18, w, h),
        "board": _frac(0.02, 0.70, 0.98, 0.98, w, h),
    }


def _diagonal_slash(w: float, h: float, orientation: str = "left") -> Dict[str, Rect]:
    # Khớp .diagonal-content width:44% cố định (dư biên 47% chống bo tròn Gaussian blur).
    # orientation == "right": nội dung dạt sang cột chéo bên phải (bản gương).
    if orientation in ("right", "top_right", "bottom_right"):
        return {"content": _frac(0.53, 0.0, 1.0, 1.0, w, h)}
    return {"content": _frac(0.0, 0.0, 0.47, 1.0, w, h)}


def _customer_feedback_card(w: float, h: float) -> Dict[str, Rect]:
    # Header thanh mảnh ở đỉnh: cao 12%
    header = _frac(0.0, 0.0, 1.0, 0.12, w, h)
    if w > h:
        # Màn ngang: .feedback-card đặt ở góc trái: rộng 42%, cao 70%
        card = (32.0, 0.16 * h, 32.0 + 0.42 * w, 0.86 * h)
    else:
        # Màn vuông 1:1 hoặc dọc: .feedback-card là thẻ gọn gàng ở đáy (cao 26%, y từ 0.70 đến 0.96)
        # Khoảng giữa từ y=0.12 đến 0.70 (58% chiều cao poster) HOÀN TOÀN MỞ CHO SPA/CHỦ THỂ!
        card = (32.0, 0.70 * h, w - 32.0, 0.96 * h)
    return {"header": header, "card": card}


def _step_process_roadmap(w: float, h: float) -> Dict[str, Rect]:
    # .top-header thanh mảnh ở đỉnh: cao 15% (badge + hero + subhead)
    # .roadmap-platform bệ quy trình gọn gàng ở đáy: cao 22% (từ 0.78 đến 1.0)
    # Khoảng giữa từ y=0.15 đến 0.78 (63% chiều cao poster) HOÀN TOÀN MỞ CHO ẢNH GYM/FITNESS!
    return {
        "header": _frac(0.0, 0.0, 1.0, 0.15, w, h),
        "platform": _frac(0.0, 0.78, 1.0, 1.0, w, h),
    }


_GEOMETRY_FUNCS: Dict[str, Callable[..., Dict[str, Rect]]] = {
    "split_left": _split_left,
    "split_right": _split_right,
    "sandwich_top_heavy": _sandwich_top_heavy,
    "sandwich_bottom_heavy": _sandwich_bottom_heavy,
    "before_after_split": _before_after_split,
    "luxury_centered_card": _luxury_centered_card,
    "lifestyle_corner_pod": _lifestyle_corner_pod,
    "recruitment_board": _recruitment_board,
    "diagonal_slash": _diagonal_slash,
    "customer_feedback_card": _customer_feedback_card,
    "step_process_roadmap": _step_process_roadmap,
}


def get_zones(template: str, width: int, height: int, orientation: Optional[str] = None) -> Dict[str, Rect]:
    """Trả về {tên_vùng: (x1,y1,x2,y2) px} cho template này ở đúng width/height.
    Dict rỗng nghĩa là zero-mask (vd luxury_centered_card) hoặc template lạ."""
    fn = _GEOMETRY_FUNCS.get(template)
    if fn is None:
        return {}
    if template == "lifestyle_corner_pod":
        return fn(float(width), float(height), orientation=orientation or "bottom_left")
    if template == "diagonal_slash":
        return fn(float(width), float(height), orientation=orientation or "left")
    return fn(float(width), float(height))


__all__ = ["Rect", "get_zones"]
