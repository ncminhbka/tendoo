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

from tendoo_v3.catalog import TEMPLATE_CATALOG

Rect = Tuple[float, float, float, float]  # (x1, y1, x2, y2) tính bằng px


def _frac(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> Rect:
    return (x1 * w, y1 * h, x2 * w, y2 * h)


# ==================================================================================
# MẬT ĐỘ NỘI DUNG (density): dùng để co giãn zone THEO ĐÚNG lượng nội dung thật, thay
# vì luôn cấp 1 kích thước CỐ ĐỊNH bất kể ít/nhiều chữ -- tránh khoảng trống rỗng giữa
# card khi nội dung ít (case thật: luxury_centered_card badge+hero+subhead+2 bullet+cta
# nhưng card cao cố định 58-84% height, để hở khoảng trắng lớn giữa bullet-list và cta
# vì `justify-content: space-between` kéo giãn hết cỡ). CHỈ áp dụng cho các template có
# 1 "card/pod" độc lập kích thước không phụ thuộc layout khác trong cùng khung (luxury_
# centered_card, customer_feedback_card, lifestyle_corner_pod, recruitment_board,
# l_frame_showcase) -- KHÔNG áp dụng cho các template dạng "band" đã dùng flex+gap co
# theo nội dung tự nhiên rồi (sandwich_top/bottom_heavy, split_left/right, diagonal_
# slash, step_process_roadmap, grand_opening_banner, menu_price_board, before_after_
# split) để tránh động vào những zone đã qua kiểm chứng kỹ ở đợt sửa trước.
def compute_density_score(
    has_badge: bool = False,
    has_subhead: bool = False,
    has_rating: bool = False,
    testimonial_len: int = 0,
    num_extra_texts: int = 0,
    extra_texts_total_len: int = 0,
    has_cta: bool = False,
    num_store_items: int = 0,
    has_qr: bool = False,
) -> float:
    """Ước lượng mật độ nội dung [0.35, 1.0] dựa trên các field THẬT SỰ sẽ chiếm
    không gian trong card/pod -- 0.35 là sàn (không co nhỏ hơn để card không trông vỡ
    dáng khi gần như trống), 1.0 là kích thước tối đa hiện tại (nội dung đầy đủ nhất).
    `has_qr` từng bị thiếu hoàn toàn khỏi công thức dù QR là khối chiếm không gian cố
    định lớn nhất trong footer (~90-120px, xem `compute_customer_feedback_budget`/
    `compute_luxury_centered_card_budget` trong renderer.py) -- để trống mỗi QR không
    hề làm card nhỏ lại dù nội dung thật đã ít hơn hẳn."""
    score = 0.0
    if has_badge:
        score += 0.08
    if has_subhead:
        score += 0.10
    if has_rating:
        score += 0.05
    if testimonial_len:
        score += min(0.20, 0.06 + testimonial_len / 400.0)
    if num_extra_texts:
        score += min(0.35, num_extra_texts * 0.10 + extra_texts_total_len / 600.0)
    if has_cta:
        score += 0.08
    if num_store_items:
        score += min(0.14, num_store_items * 0.05)
    if has_qr:
        score += 0.14
    return max(0.35, min(1.0, score))


# ==================================================================================
# 1 hàm mỗi template, trả về {tên_vùng: Rect (px)} -- khớp CHÍNH XÁC với vị trí
# CSS thật đã đặt trong `templates/<template>/template.html` (đã đối chiếu số
# đo Chromium thật, xem scripts/check_tendoo_v3_mask_overflow.py).
# ==================================================================================

def _split_left(w: float, h: float) -> Dict[str, Rect]:
    # Thích ứng dẻo dai theo 4 tỉ lệ khung hình (1:1, 9:16, 16:9, 4:5):
    # - 9:16 (Narrow Portrait): width 48.5% (đủ rộng cho từ dài không wrap sớm)
    # - 16:9 (Wide Landscape): width 41.0% (độ mở 59% cho chủ thể)
    # - 1:1 / 4:5: width 44.0% (độ mở 56% cho chủ thể)
    # Diện tích mask luôn từ 41.0% - 48.5% (hoàn toàn <= 50% theo yêu cầu).
    if w / h < 0.7:
        frac_w = 0.485
    elif w / h >= 1.5:
        frac_w = 0.410
    else:
        frac_w = 0.440
    return {"content": _frac(0.0, 0.0, frac_w, 1.0, w, h)}


def _split_right(w: float, h: float) -> Dict[str, Rect]:
    if w / h < 0.7:
        frac_w = 0.485
    elif w / h >= 1.5:
        frac_w = 0.410
    else:
        frac_w = 0.440
    return {"content": _frac(1.0 - frac_w, 0.0, 1.0, 1.0, w, h)}


def _sandwich_top_heavy(w: float, h: float, has_qr: bool = True) -> Dict[str, Rect]:
    # Tự động điều chỉnh dẻo dai theo 4 tỉ lệ khung hình (1:1, 9:16, 16:9, 4:5):
    # - 16:9 (Landscape rộng hẹp h=576): Top 27% (155.5px), Bottom 19% (109.4px)
    # - 9:16 (Narrow Portrait w=576, h=1024): Top 31% (317.4px), Bottom 16% (163.8px)
    # - 1:1 / 4:5 (h=1024): Top 28.5% (291.8px), Bottom 16% (163.8px)
    # Tổng diện tích mask luôn dao động 44.5% - 47.0% (hoàn toàn <= 50% theo yêu cầu).
    if w / h >= 1.5:  # 16:9 Landscape
        if has_qr:
            bottom_start = 0.81
        else:
            bottom_start = 0.84
        top_frac = 0.27
    elif w / h < 0.7:  # 9:16 Narrow Portrait
        top_frac = 0.31
        if has_qr:
            bottom_start = 0.84
        else:
            bottom_start = 0.868
    else:  # 1:1, 4:5
        top_frac = 0.31
        if has_qr:
            bottom_start = 0.84
        else:
            bottom_start = 0.868

    return {
        "top": _frac(0.0, 0.0, 1.0, top_frac, w, h),
        "bottom": _frac(0.0, bottom_start, 1.0, 1.0, w, h),
    }


def _sandwich_bottom_heavy(w: float, h: float, has_qr: bool = True) -> Dict[str, Rect]:
    # .sandwich-top-bar: 12.5% (1:1/4:5/9:16) hoặc 17.0% (16:9 landscape) cho store + badge + QR
    # .sandwich-bottom-platform: 31.5% (16:9) hoặc 36.5% (1:1/4:5/9:16) cho hero+subhead+pill+cta
    # Tổng mask đảm bảo luôn dao động 48.5% - 49.0% (hoàn toàn <= 50% theo quy chuẩn).
    if w / h >= 1.5:  # 16:9 Landscape (w=1024, h=576)
        if has_qr:
            top_frac = 0.170  # ~97.9px (đủ badge + store + QR 64px)
        else:
            top_frac = 0.120  # ~69.1px (đủ badge + store không QR)
        # bottom_start = 0.685 -> chiều cao đáy là 0.315 * 576 = 181.4px
        # Tổng mask: 0.170 + 0.315 = 0.485 (48.5% <= 50% chuẩn mực)
        bottom_start = 0.685
    else:  # 1:1, 4:5, 9:16
        if has_qr:
            top_frac = 0.125  # ~128px ở h=1024
        else:
            top_frac = 0.088  # ~90px ở h=1024
        # bottom_start = 0.635 -> chiều cao đáy là 0.365 * 1024 = 373.8px
        # Tổng mask: 0.125 + 0.365 = 0.490 (49.0% <= 50% chuẩn mực)
        bottom_start = 0.635

    return {
        "top": _frac(0.0, 0.0, 1.0, top_frac, w, h),
        "bottom": _frac(0.0, bottom_start, 1.0, 1.0, w, h),
    }


def _before_after_split(w: float, h: float, orientation: str = "left", has_footer: bool = True) -> Dict[str, Rect]:
    # `has_footer` (= qr_code hoặc cta hoặc store_info): đo thật bằng Playwright
    # (probe_before_after2) cho thấy ở bố cục XẾP CHỒNG (1:1/9:16/4:5), khi không có
    # footer (dù freetext có hay không), nội dung thật chỉ cần 227.8-288.8px trong khi
    # pod vẫn cấp 378.9-389.1px -- co lại với margin an toàn ~39-49px. Ở 16:9 (bố cục 2
    # CỘT), land-left-col (hero+subhead+stars) không phụ thuộc footer -- có/không
    # footer đều cần ~135px như nhau -- KHÔNG co, giữ nguyên bất kể has_footer.
    label_w = min(190.0, 0.30 * w)
    label_h = min(56.0, 0.10 * h)
    aspect = w / h
    if aspect < 0.7:  # 9:16 Narrow Portrait
        if has_footer:
            # 0.33 từng đo thấy tràn thật ~7-11px: rating-stars+hero+subhead+freetext-block
            # +footer-row (QR component ~108px không được budget riêng, luôn cố định theo
            # qr_size trong renderer.py, không co giãn theo fraction) cộng gap/padding cần
            # ~349px nhưng 0.33*1024=337.9px không đủ -- nâng lên 0.38 (389.1px, dư margin
            # rộng vì hero-title tận dụng thêm chiều cao mới để tự tăng font theo đúng cơ
            # chế elastic, ăn lại 1 phần margin mỗi lần tăng -- 0.35 vẫn còn thiếu ~9px)
            bot_frac = 0.38
        else:
            # Đo thật (không footer, freetext có hoặc không): tối đa 288.8px --
            # 0.32 (327.7px) dư ~38.9px margin an toàn.
            bot_frac = 0.32
    elif aspect >= 1.5:  # 16:9 Wide Landscape
        # 0.28 (161.3px) đủ khi trần font hero/subhead còn thấp, nhưng sau khi nâng
        # trần (hero_max_f 42->46, subhead ceiling 16->18 theo yêu cầu tăng cỡ chữ)
        # land-left-col (hero 2 dòng + subhead) đo thật cần tới 168px ở case
        # ba_07_16x9_right_landscaping (hero+subhead dài) -- tràn ~7px. Nâng lên 0.32
        # (184.3px) dư ~16px margin an toàn.
        bot_frac = 0.32
    else:  # 1:1, 4:5
        if has_footer:
            # 0.33 từng đo thấy tràn thật ~8px ở case đủ rating+subhead+freetext+cta+
            # store+qr (ba_12_4x5_left_rich) -- cùng nguyên nhân với narrow ở trên (QR
            # ~113-115px trong footer-row không co giãn, chiếm mất phần lớn ngân sách
            # còn lại cho hero/subhead/freetext). Nâng lên 0.37 (378.9px ở height=1024)
            # -- xem compute_before_after_budget() trong renderer.py.
            bot_frac = 0.37
        else:
            # Đo thật (không footer, freetext có hoặc không, 1:1 và 4:5): tối đa 278.0px
            # -- 0.31 (317.4px) dư ~39.4px margin an toàn.
            bot_frac = 0.31

    if orientation in ("right", "top_right", "bottom_right", "reverse"):
        l_before = (w - 20.0 - label_w, 20.0, w - 20.0, 20.0 + label_h)
        l_after = (20.0, 20.0, 20.0 + label_w, 20.0 + label_h)
    else:
        l_before = (20.0, 20.0, 20.0 + label_w, 20.0 + label_h)
        l_after = (w - 20.0 - label_w, 20.0, w - 20.0, 20.0 + label_h)

    return {
        "label_before": l_before,
        "label_after": l_after,
        "bottom": _frac(0.0, 1.0 - bot_frac, 1.0, 1.0, w, h),
    }


def _luxury_centered_card(
    w: float,
    h: float,
    orientation: str = "center",
    density: float = 1.0,
    has_message: bool = True,
    has_footer: bool = True,
) -> Dict[str, Rect]:
    # Luxury Floating Glass Card (VIP Voucher / Tri Ân / Khách Hàng Thượng Lưu / Sự Kiện Xa Xỉ):
    # Hỗ trợ 3 hướng: 'center' (mặc định), 'left', 'right' (Mirror 100%).
    # Tổng diện tích thẻ luôn dao động từ 38% đến 47% (luôn <= 50% theo quy chuẩn),
    # chừa không gian thoáng đạt cho sản phẩm trang sức, đồng hồ, ô tô hoặc người mẫu DiT.
    # `density` (0.35-1.0, xem compute_density_score()) co giãn CHIỀU CAO card theo
    # đúng lượng nội dung thật -- ít chữ thì card thấp lại (co đối xứng quanh tâm dọc
    # ban đầu), tránh khoảng trắng rỗng giữa bullet-list và CTA do `justify-content:
    # space-between` kéo giãn hết cỡ 1 card cao cố định. `density` vẫn TẮT (không có
    # trong `_DENSITY_AWARE_TEMPLATES`) -- luôn = 1.0.
    #
    # `has_message`/`has_footer` (mirror đúng 2 cờ lớn nhất trong
    # compute_luxury_centered_card_budget(), đo thật bằng Playwright qua 5 tổ hợp):
    # khi CẢ 2 đều False (chỉ có header-cluster: badge/rating/hero/subhead), tổng nội
    # dung thật chỉ cần 254-341px (tuỳ tỉ lệ) trong khi `ch` gốc cấp 484-737px -- co
    # về mức có margin ~48-58px. Khi CÓ message HOẶC footer, margin đo được có thể rất
    # mỏng (case "full" ở 9:16 chỉ dư 0.5px!) -- TUYỆT ĐỐI giữ nguyên `ch` gốc, không
    # cố tách tiếp theo has_badge/has_stars/has_subhead riêng lẻ (rủi ro cao, chưa đo
    # đủ hết 32 tổ hợp để an toàn).
    has_content = has_message or has_footer
    aspect = w / h
    if aspect < 0.7:  # 9:16 Narrow Portrait (576x1024)
        cw = 0.80
        # ch = 0.58 (593.9px): Đảm bảo tổng diện tích mask (kể cả Gaussian blur)
        # luôn <= 48.5% <= 50.0% theo quy chuẩn, không vượt quá nửa bức ảnh.
        ch = 0.58 if has_content else 0.36
        cy = 0.21
        if orientation in ("left", "top_left", "bottom_left"):
            cx = 0.05
        elif orientation in ("right", "top_right", "bottom_right"):
            cx = 0.15
        else:
            cx = 0.10
    elif aspect >= 1.5:  # 16:9 Wide Landscape (1024x576)
        cw = 0.52
        ch = 0.84 if has_content else 0.53
        cy = 0.08
        if orientation in ("left", "top_left", "bottom_left"):
            cx = 0.05
        elif orientation in ("right", "top_right", "bottom_right"):
            cx = 0.43
        else:
            cx = 0.24
    else:  # 1:1, 4:5
        if aspect >= 0.95:  # 1:1 (1024x1024)
            cw = 0.64
            ch = 0.72 if has_content else 0.32
            cy = 0.14
        else:  # 4:5 (816x1024)
            cw = 0.70
            ch = 0.68 if has_content else 0.31
            cy = 0.16
        if orientation in ("left", "top_left", "bottom_left"):
            cx = 0.06
        elif orientation in ("right", "top_right", "bottom_right"):
            cx = 1.0 - cw - 0.06
        else:
            cx = (1.0 - cw) / 2.0

    ch_scaled = ch * density
    cy_scaled = cy + (ch - ch_scaled) / 2.0
    return {"card": _frac(cx, cy_scaled, cx + cw, cy_scaled + ch_scaled, w, h)}


def _lifestyle_corner_pod(
    w: float,
    h: float,
    orientation: str = "bottom_left",
    density: float = 1.0,
    has_qr: bool = True,
    has_freetext: bool = True,
    has_footer: bool = True,
) -> Dict[str, Rect]:
    # Hộp Bo Góc Capsule Cục Bộ (Corner Pod):
    # Chiếm ~23% - 32% diện tích góc, để lại >68% - 77% poster hoàn toàn thoáng đạt cho ảnh Lifestyle.
    # Hỗ trợ đầy đủ 4 hướng góc: bottom_left (mặc định), bottom_right, top_left, top_right.
    # `density` co giãn CHIỀU CAO box theo lượng nội dung thật (xem compute_density_
    # score()) -- cạnh neo góc (top hoặc bottom tuỳ orientation) giữ nguyên vị trí,
    # chỉ cạnh còn lại "co vào" để tránh khoảng trắng rỗng khi ít chữ. `density` vẫn
    # TẮT (không có trong `_DENSITY_AWARE_TEMPLATES`) -- luôn = 1.0.
    #
    # `.corner-pod` (template.html) chỉ set `max-height` (KHÔNG set `height` cứng) --
    # tự shrink-wrap theo nội dung thật, không có khoảng trống lộ liễu trong CSS. Chỉ
    # riêng MASK (box_h dùng cho get_zones()) vẫn cố định độc lập với DOM thật -- đo
    # Playwright cho thấy pod chỉ cao 181-294px (hero+cta, không qr/freetext) trong khi
    # mask vẫn cấp 450-532px. `has_qr or has_freetext` (2 driver chính -- QR cố định
    # 90-115px, freetext là khối biến đổi lớn nhất) -- khi CẢ 2 đều False, co box_h.
    # CÓ 1 trong 2 thì giữ NGUYÊN bản gốc (đo thật case dense ở 16:9 chỉ dư ~0.02px,
    # sát trần tuyệt đối -- TUYỆT ĐỐI không đụng).
    #
    # SỬA LẠI: `has_content` ban đầu chỉ xét has_qr/has_freetext, bỏ sót cta/store --
    # khi `compute_lifestyle_corner_pod_budget()`'s `footer_reserve` được sửa đúng
    # (không còn cố định bất kể cta/store rỗng hay không, xem renderer.py), case có
    # cta+store (nhưng không qr/freetext) vẫn cần footer_reserve~46px thật, trong khi
    # box_h lại co về mức "hoàn toàn trống" -- gây tràn thật (đo Playwright:
    # pod_nocontent_02_9x16 cần 324px, pod_nocontent_04_4x5 cần 333px). Dùng
    # `has_footer` (= qr/cta/store, đúng định nghĩa footer_reserve) thay vì chỉ has_qr.
    has_content = has_footer or has_freetext
    aspect = w / h
    if aspect < 0.7:  # 9:16 Narrow Portrait
        box_w = 0.72 * w
        if has_content:
            # Sau khi nâng trần font (đúng yêu cầu tăng cỡ chữ), đo lại thấy tràn ~9px
            # ở case đủ mọi phần tử (pod_06_9x16_bottom_right_perfume, scrollHeight
            # 460 > clientHeight 451) -- cùng nguyên nhân đã sửa ở nhánh 16:9. Nâng
            # 0.44 -> 0.46 (471.0px, dư ~11px margin an toàn).
            box_h = 0.46 * h
        else:
            box_h = 0.28 * h
    elif aspect >= 1.5:  # 16:9 Wide Landscape
        box_w = 0.42 * w
        if has_content:
            # 0.68 từng đo thấy tràn thật ~40-50px ra ngoài canvas khi badge+hero(2 dòng)+
            # subhead+freetext+footer(cta+store) cùng có mặt -- 0.78 vẫn còn <=50% diện
            # tích poster (0.42*0.78=32.8%, trong yêu cầu <=50%). Sau khi nâng trần font
            # badge/subhead (đúng yêu cầu tăng cỡ chữ), đo lại thấy tràn thêm ~11px ở
            # case đủ mọi phần tử (pod_07_16x9_bottom_right_interior, .corner-pod
            # scrollHeight 460 > clientHeight 449) vì các phần tử giờ dùng gần hết
            # max_h thay vì dư ra như khi bị trần font thấp chặn trước -- nâng lên 0.81
            # (466.6px, dư ~17px margin an toàn, vẫn <=50% diện tích: 0.42*0.81=34%).
            box_h = 0.81 * h
        else:
            # Đo thật (hero+cta, không qr/freetext): pod chỉ cao 214.3px -- 0.48
            # (276.5px) dư ~62px margin an toàn (chỉ ảnh hưởng mask, CSS tự shrink-wrap).
            box_h = 0.48 * h
    else:  # 1:1, 4:5
        if has_content:
            # 0.48 từng đo thấy tràn thật ~15-17px ở case đủ badge+hero(2 dòng)+subhead+
            # freetext+footer(cta+qr+store, QR ~115px không co giãn) cùng có mặt
            # (pod_08_4x5_bottom_right_tea, .corner-pod scrollHeight 507 > clientHeight
            # 490) -- cùng nguyên nhân đã sửa ở nhánh 16:9 bên trên. Nâng lên 0.52 (dư
            # margin ~20px) -- xem compute_lifestyle_corner_pod_budget() trong renderer.py.
            box_h = 0.52 * h
        else:
            # Đo thật (hero+cta, không qr/freetext, 1:1 và 4:5): pod cao tối đa 293.8px
            # (4:5) -- 0.36 (368.6px ở h=1024) dư ~75px margin an toàn.
            box_h = 0.36 * h
        box_w = 0.48 * w

    box_h_scaled = box_h * max(0.35, min(1.0, density))
    pad_x = 24.0
    pad_y = 24.0

    if orientation in ("bottom_right", "right"):
        return {"pod": (w - pad_x - box_w, h - pad_y - box_h_scaled, w - pad_x, h - pad_y)}
    elif orientation == "top_left":
        return {"pod": (pad_x, pad_y, pad_x + box_w, pad_y + box_h_scaled)}
    elif orientation == "top_right":
        return {"pod": (w - pad_x - box_w, pad_y, w - pad_x, pad_y + box_h_scaled)}
    else:  # bottom_left (mặc định)
        return {"pod": (pad_x, h - pad_y - box_h_scaled, pad_x + box_w, h - pad_y)}


def _diagonal_slash(w: float, h: float, orientation: str = "left") -> Dict[str, Rect]:
    # Khớp tendoo_legacy: x_top=0.46, x_bottom=0.12 (diện tích ~24%-35%)
    aspect = w / h
    if aspect < 0.7:  # 9:16
        top_f = 0.50
    elif aspect >= 1.5:  # 16:9
        top_f = 0.38
    else:  # 1:1, 4:5
        top_f = 0.44

    if orientation in ("right", "top_right", "bottom_right"):
        return {"content": _frac(1.0 - top_f, 0.0, 1.0, 1.0, w, h)}
    return {"content": _frac(0.0, 0.0, top_f, 1.0, w, h)}


def _customer_feedback_card(
    w: float, h: float, orientation: str = "left", density: float = 1.0, has_footer: bool = True
) -> Dict[str, Rect]:
    # Header thanh mảnh ở đỉnh
    # `density` co giãn CHIỀU CAO card theo lượng nội dung thật (xem compute_density_
    # score()) -- cạnh đáy (1:1/9:16/4:5) hoặc cạnh trên (16:9) giữ nguyên vị trí neo,
    # chỉ cạnh còn lại co vào, tránh khoảng trắng rỗng khi testimonial/freetext ngắn.
    # `density` vẫn TẮT (không có trong `_DENSITY_AWARE_TEMPLATES`) -- luôn = 1.0.
    #
    # `has_footer` (= qr_code hoặc cta hoặc store_info, đúng định nghĩa
    # `compute_customer_feedback_budget()` đã dùng): đo thật bằng Playwright cho thấy
    # `compute_customer_feedback_budget()` đã rebalance % NỘI BỘ rất tốt (testimonial
    # tự lớn khi không có freetext, footer_reserve co theo has_qr/has_footer) nhưng
    # bản thân `card_frac_h` (kích thước NGOÀI, quyết định mask) chưa từng co --
    # khi has_footer=False (không qr/cta/store), nội dung thật chỉ cần 108-141px
    # (tuỳ tỉ lệ) trong khi card vẫn giữ nguyên 338-415px. `has_qr` riêng lẻ (còn
    # cta/store) KHÔNG cần co (đo thật fill vẫn ~92%, sát current) nên chỉ dùng
    # `has_footer` (gộp cả 3), không tách has_qr riêng như các template khác.
    aspect = w / h
    density = max(0.35, min(1.0, density))
    header = _frac(0.0, 0.0, 1.0, 0.10 if aspect >= 1.5 else 0.14, w, h)
    if aspect >= 1.5:  # 16:9 Wide Landscape
        card_w = 0.42 * w
        if has_footer:
            # Sau khi nâng trần font (đúng yêu cầu tăng cỡ chữ, user báo cáo trực tiếp
            # "tất cả cỡ chữ đều quá nhỏ"), đo lại thấy tràn ~49px ở case đủ mọi phần
            # tử (fb_11_16x9_left_minimal, .feedback-card scrollHeight 464 >
            # clientHeight 415) -- cùng nguyên nhân đã gặp ở lifestyle_corner_pod/
            # l_frame_showcase. Card neo từ y1=0.16*h nên card_h tối đa an toàn (không
            # tràn ra ngoài canvas) chỉ là (1-0.16)*h=0.84*h -- 0.85 (thử ban đầu) đã
            # VƯỢT trần đó, gây tràn canvas. Chốt 0.82 (472.3px, dư ~8px so với 464px
            # đo được, dư ~11.5px tới mép canvas) -- NHƯNG sau khi đổi font UI phụ
            # (badge/verified-chip/extra-pill/store-item) từ font nghệ thuật đã chọn
            # sang `--ui-font` cố định (Be Vietnam Pro, xem template.html) để dễ đọc
            # hơn, metrics font mới rộng/cao hơn 1 chút cho cùng nội dung, đo lại thấy
            # tràn thật ~19px (scrollHeight 491 > clientHeight 472) ở
            # fb_15_16x9_right_minimal. Không đủ margin nếu chỉ tăng card_h (đã sát
            # trần y1=0.16h) -- hạ luôn y1 xuống 0.12h (card neo thấp hơn, gần header
            # hơn) để có thêm ~23px, nâng card_h lên 0.86h.
            card_h = 0.86 * h * density
        else:
            # Đo thật (không footer, bare_minimum/no_footer_at_all): content tối đa
            # 141.4px -- 0.32*h (184.3px) dư ~42.9px margin an toàn.
            card_h = 0.32 * h * density
        card_y1 = 0.12 * h if has_footer else 0.16 * h
        if orientation in ("right", "top_right", "bottom_right"):
            card = (w - 28.0 - card_w, card_y1, w - 28.0, card_y1 + card_h)
        else:
            card = (28.0, card_y1, 28.0 + card_w, card_y1 + card_h)
    else:  # 1:1, 9:16, 4:5
        if aspect < 0.7:  # 9:16 Narrow Portrait
            if has_footer:
                # 0.30 từng đo thấy tràn thật ~25-27px: stars-row+testimonial+reviewer+
                # freetext-block+footer-action-row (cta+store) cộng gap/padding cần ~332px
                # nhưng 0.30*1024=307px không đủ -- nâng lên 0.33 (337.9px, dư margin).
                # Sau khi nâng trần font, đo lại thấy tràn nhẹ ~4-5px (fb_06/fb_02,
                # scrollHeight 342-363 > clientHeight 338-358) -- nâng thêm lên 0.36
                # (368.6px, dư ~10px margin an toàn qua nhiều case).
                card_frac_h = 0.36
            else:
                # Đo thật (không footer): content tối đa 135.5px -- 0.17 (174.1px)
                # dư ~38.6px margin an toàn.
                card_frac_h = 0.17
        else:  # 1:1, 4:5
            if has_footer:
                # 0.315 từng đo thấy margin thật CHỈ ~1-2px ở case đủ mọi phần tử (fb_12/
                # 16_4x5_..._rich: testimonial+reviewer+freetext+footer QR-bound ~118px)
                # -- tổng cần ~319.5px trên card cao 321.7px, sát mép tới mức chỉ cần đổi
                # % phân bổ nội bộ (không đổi tổng nội dung) cũng đủ gây tràn. Nâng lên
                # 0.34 (348.2px, dư margin ~28px) -- xem compute_customer_feedback_budget()
                # trong renderer.py.
                card_frac_h = 0.34
            else:
                # Đo thật (không footer): content tối đa 108.5px -- 0.15 (153.6px)
                # dư ~45px margin an toàn. Sau khi nâng trần font thêm 1 vòng nữa (user
                # yêu cầu "tăng toàn bộ"), case rich (freetext dài) tràn thật ~9px
                # (scrollHeight 163 > clientHeight 154) -- nâng lên 0.17 (174.1px).
                # Đo thêm 1 case khác (badge+stars+subhead+testimonial+reviewer+freetext
                # 3 dòng, ảnh render thật trên server user gửi -- pill freetext cuối
                # cùng bị cắt): scrollHeight thật 198px > 0.17*1024=174.1px vẫn còn
                # thiếu ~24px -- 0.17 chưa đủ cho tổ hợp ĐẦY ĐỦ nhất (có cả stars +
                # verified-chip + testimonial + reviewer + 3-dòng freetext cùng lúc,
                # case trước chỉ đo thiếu 1-2 phần tử). Nâng lên 0.21 (215px) -- NHƯNG
                # card cao hơn khiến autofit testimonial tự chọn font to hơn nữa
                # (23->27.5px, "ăn lại" hết margin vừa thêm, đúng hiệu ứng đã gặp nhiều
                # lần trong phiên này: nới trần lộ ra chỗ thiếu mới) -- đo lại vẫn thiếu
                # ~10px thật (225 vs 215). Nâng hẳn lên 0.25 (256px) để có margin đủ
                # rộng chịu được vòng lặp tự-lớn này.
                card_frac_h = 0.25
        card = _frac(0.04, 0.98 - card_frac_h * density, 0.96, 0.98, w, h)

    return {"header": header, "card": card}


def _recruitment_board(
    w: float, h: float, orientation: str = "left", density: float = 1.0, has_qr: bool = True
) -> Dict[str, Rect]:
    # `density` co giãn CHIỀU CAO board (cạnh đáy neo cố định) theo lượng nội dung
    # quyền lợi/điều kiện thật -- xem compute_density_score(). Sàn RIÊNG cao hơn mức
    # chung [0.35,1.0] (0.35 gây tràn thật ~14/16 case: board_h gốc vốn đã nhỏ (17-
    # 22%) nên co xuống 35% còn quá bé để chứa col-title+1 pill+cta+qr tối thiểu) --
    # remap density [0.35,1.0] -> [0.80,1.0] để board không bao giờ co dưới 80% kích
    # thước gốc, vẫn có chút co giãn nhưng an toàn cho nội dung tối thiểu luôn hiện
    # diện (col-title, cta, qr không phải optional như ở luxury_centered_card).
    # `density` vẫn TẮT (không có trong `_DENSITY_AWARE_TEMPLATES`) -- luôn = 1.0.
    #
    # `has_qr`: đo thật bằng Playwright (probe_recruit_board) -- .board-card dùng
    # align-items:stretch (giống step_process_roadmap), QR kiosk cố định ~108-145px
    # là driver chính khi có QR (margin còn lại rất mỏng: 16:9 chỉ dư 6.8px, 9:16 dư
    # ~29px -- TUYỆT ĐỐI không đụng nhánh has_qr=True). Khi không QR, col-left/col-mid
    # tự nhiên chỉ cần 83-111px (1:1/4:5/16:9) hoặc tổng col-left+bottom-split ~164px
    # (9:16, layout cột dọc riêng) -- co board_h với margin an toàn rộng (~20-45px).
    # GĐ 4 (26/09): header 15%/18% -> 20%/22% (cùng lý do step_process_roadmap: hero bị chiều cao
    # khoá). Đạt đủ 4 điều kiện squint 2 -> 12/22 (hero phẳng), 16/16 (hero_parts); không mất chữ.
    aspect = w / h
    density = max(0.35, min(1.0, density))
    density = 0.80 + (density - 0.35) / (1.0 - 0.35) * (1.0 - 0.80)
    if aspect < 0.7:  # 9:16 Narrow Portrait
        head_h = 0.20
        if has_qr:
            # 0.20 từng đo thấy tràn thật ~7-10px qua Playwright (board-card scrollHeight
            # 224 > clientHeight 215 ở case recruit_10_9x16_left_bullet, 3 dòng extra
            # bullet) -- .board-card ở narrow xếp CỘT DỌC (col-left rồi board-bottom-split
            # chứa cta+store+qr), khác hẳn layout HÀNG NGANG của 1:1/4:5/16:9, nên cần
            # nhiều chiều cao hơn hẳn (không thể dùng chung 1 fraction với 2 nhánh kia).
            # Nâng lên 0.24 (đủ margin an toàn cho khối QR kiosk cố định ~108px + col-title
            # + 3 dòng extra) -- xem compute_recruitment_board_budget() trong renderer.py.
            board_h = 0.24
        else:
            # Đo thật (không QR, dense): colLeft 91.4px + board-bottom-split 63.1px
            # (~164.5px tổng cộng gap) -- 0.195 (199.7px) dư ~35px margin an toàn.
            board_h = 0.195
    elif aspect >= 1.5:  # 16:9 Wide Landscape
        head_h = 0.22
        if has_qr:
            board_h = 0.22
        else:
            # Đo thật (không QR, dense): max(colLeft 91.4, colMid 83.4) -- 0.19
            # (109.4px) dư ~18px margin an toàn.
            board_h = 0.19
    else:  # 1:1, 4:5
        head_h = 0.20
        if has_qr:
            board_h = 0.17
        else:
            # Đo thật (không QR, dense, cả 1:1 và 4:5): max(colLeft, colMid) tối đa
            # 110.8px -- 0.145 (148.5px ở h=1024) dư ~38px margin an toàn.
            board_h = 0.145

    header = _frac(0.04, 0.0, 0.96, head_h, w, h)
    board = _frac(0.04, 0.98 - board_h * density, 0.96, 0.98, w, h)
    return {"header": header, "board": board}


def _step_process_roadmap(w: float, h: float, orientation: str = "left", has_qr: bool = True) -> Dict[str, Rect]:
    # .top-header ở đỉnh: cao 20% (22% ở 16:9) -- GĐ 4 (26/09) nâng từ 15%/18%: hero tiêu đề dài
    # bị chiều cao khoá ở 36-41px < 2.5x subhead 19.5px -> C1 0/21; nay hero ~49.5px, C1 11/21,
    # đạt đủ 4 điều kiện 0 -> 9/21, không mất chữ (probe_type_hierarchy --template step_process_roadmap).
    # .roadmap-platform bệ quy trình tinh gọn: cao 16.5% ở 1:1/4:5 để triệt tiêu khoảng trống thừa
    # Khoảng giữa (>68% chiều cao poster) HOÀN TOÀN MỞ CHO ẢNH GYM/FITNESS/SPA/SẢN PHẨM!
    #
    # `has_qr`: đo thật bằng Playwright (probe_step_platform) -- CHỈ co ở 9:16, vì đó
    # là tỉ lệ DUY NHẤT có margin thật sự an toàn để co: dense-text 4 bước + cta +
    # store (không QR) chỉ cần flow_natural=103.5px trong khi 0.25*1024=256px đang cấp
    # -- dư ~150px. 1:1/4:5 (168.95px cấp) chỉ dư 10-33px so với dense-text đo được
    # (135.9-158.3px) -- KHÔNG đủ margin an toàn để co, giữ nguyên bất kể has_qr.
    # 16:9 (126.7px cấp) gần như sát trần ngay cả khi CÓ QR (qr_kiosk đo được 112.7px)
    # -- TUYỆT ĐỐI không đụng.
    aspect = w / h
    if aspect < 0.7:  # 9:16 Narrow Portrait
        head_h = 0.20
        if has_qr:
            # 0.22 từng đo thấy tràn thật ~15-17px: steps-grid (2x2 khi 4 bước) + gap 8px +
            # roadmap-bottom-split (cta/store cột + qr-kiosk badge+QR) + padding 22px cộng
            # lại ~240-253px nhưng 0.22-0.24*1024=225-246px không đủ tuỳ mật độ nội dung --
            # nâng lên 0.25 (256px, dư margin cho case nhiều bước/QR label dài nhất)
            plat_h = 0.25
        else:
            # Đo thật (không QR, 4 bước dài + cta + store): flow_natural tối đa 103.5px
            # -- 0.19 (194.6px) dư ~91px margin an toàn (rộng rãi vì chưa test hết mọi
            # tổ hợp số bước/độ dài).
            plat_h = 0.19
    elif aspect >= 1.5:  # 16:9 Wide Landscape
        head_h = 0.22
        plat_h = 0.22
    else:  # 1:1, 4:5
        head_h = 0.20
        plat_h = 0.165

    header = _frac(0.04, 0.0, 0.96, head_h, w, h)
    platform = _frac(0.0, 1.0 - plat_h, 1.0, 1.0, w, h)
    return {"header": header, "platform": platform}


def _l_frame_showcase(
    w: float, h: float, orientation: str = "left", density: float = 1.0, has_freetext: bool = True
) -> Dict[str, Rect]:
    # L-Frame Corner Anchor & Bottom Contact Bar:
    # Hiện thực hóa chuẩn mực từ 2 ảnh mẫu đính kèm (SUV Mercedes & Áo khoác măng tô dạ).
    # Hỗ trợ Mirror 100%: orientation 'left' (mặc định) hoặc 'right'.
    # Mở rộng pod_w ở 1:1 và 4:5 lên 0.56 để chữ không bị bó hẹp, không bị cắt viền
    # `density` co giãn CHIỀU CAO top_cluster (cạnh trên neo cố định) theo lượng nội
    # dung hero/badge/subhead/freetext thật -- xem compute_density_score(). KHÔNG áp
    # dụng cho bot_rect (store/QR) -- band đó đã co theo nội dung tự nhiên qua flex.
    # `density` vẫn TẮT (không có trong `_DENSITY_AWARE_TEMPLATES`) -- luôn = 1.0.
    #
    # `.lframe-top-cluster` chỉ set `max-height` (không set `height` cứng) -- tự
    # shrink-wrap, không có khoảng trống lộ liễu trong CSS (giống lifestyle_corner_
    # pod). Chỉ MASK cố định độc lập với DOM thật. Đo Playwright (badge+subhead, KHÔNG
    # freetext): top-cluster chỉ cần 174-304px (tuỳ tỉ lệ) trong khi mask cấp 276-410px.
    # `has_freetext=True` giữ NGUYÊN bản gốc (freetext là khối lớn nhất, đủ margin
    # nhưng chưa đo sát -- không mạo hiểm đụng vào nhánh có tràn thật đã biết ở 16:9,
    # xem `compute_l_frame_showcase_budget()`'s `line_height_slack`).
    density = max(0.35, min(1.0, density))
    if w / h < 0.7:  # 9:16 Narrow Portrait
        pod_w = 0.88
        # Đo thật (badge+subhead, không freetext): 288.5px -- 0.32 (327.7px) dư ~39px.
        pod_h = 0.36 if has_freetext else 0.32
        bot_h = 0.14
    elif w / h >= 1.5:  # 16:9 Wide Landscape
        pod_w = 0.44
        # Đo thật (badge+subhead, không freetext): 173.9px -- 0.36 (207.4px) dư ~33px.
        # 0.48 (has_freetext) từng đủ nhưng sau khi nâng thêm trần font hero/subhead/
        # freetext lần 2 (user yêu cầu "tăng tất cả"), case heavy (badge+subhead+3
        # dòng freetext) tràn thật ~8px (scrollHeight 275 > clientHeight 267) -- nâng
        # lên 0.52 (299.5px), dư ~15px margin an toàn.
        pod_h = 0.52 if has_freetext else 0.36
        bot_h = 0.16
    else:  # 1:1, 4:5
        # 0.56 từng đủ nhưng đo thật cho thấy hero-title (vd "TENDOO CHRONO MASTER")
        # luôn bị KẸT THEO CHIỀU RỘNG (scrollWidth == clientWidth == data-max-width
        # đúng 545px) trong khi chiều cao vẫn còn dư (chỉ dùng 61/86px budget) -- user
        # báo cáo trực tiếp qua ảnh render thật: hero trông nhỏ dù trần font (68px) và
        # max_h còn thừa rất nhiều. Nâng pod_w lên 0.64 để hero có thêm chỗ ngang phát
        # triển thay vì bị nghẽn cổ chai ở bề rộng box trước khi chạm hết chiều cao.
        pod_w = 0.64
        # Đo thật (badge+subhead, không freetext, cả 1:1 và 4:5): tối đa 304.1px --
        # 0.34 (348.2px) dư ~44px.
        pod_h = 0.40 if has_freetext else 0.34
        bot_h = 0.13

    pod_h_scaled = pod_h * density
    if orientation in ("right", "top_right"):
        top_rect = _frac(1.0 - pod_w, 0.0, 1.0, pod_h_scaled, w, h)
    else:
        top_rect = _frac(0.0, 0.0, pod_w, pod_h_scaled, w, h)

    bot_rect = _frac(0.0, 1.0 - bot_h, 1.0, 1.0, w, h)

    return {
        "top_left": top_rect,  # Giữ alias cho backward-compat
        "top_cluster": top_rect,
        "bottom": bot_rect,
    }


def _grand_opening_banner(
    w: float, h: float, has_freetext: bool = True, has_qr: bool = True
) -> Dict[str, Rect]:
    # Banner Khai Trương / Sự Kiện Hội Lễ (Hiện thực hóa Prompt 43):
    # - Cụm Header Trung Tâm Trên Đỉnh: x từ 5% đến 95%
    # - Bệ Đáy Chuyển Đổi & Địa Chỉ Ở Đáy
    # - Khoảng trung tâm (52%-58% chiều cao) HOÀN TOÀN MỞ CHO CHỦ THỂ / SẢN PHẨM!
    # Tổng diện tích mask dao động 41% - 47% (hoàn toàn <= 50% theo yêu cầu).
    #
    # `.opening-header-cluster`/`.opening-bottom-bar` (template.html) đã tự auto-size
    # theo content (không set height/max-height cứng) -- CSS không có khoảng trống lộ
    # liễu, chỉ MASK bị cố định độc lập với DOM thật. Đo Playwright thật (2 cờ riêng
    # biệt, vì header driver chính là freetext còn bottom driver chính là QR -- không
    # dùng chung 1 cờ): `has_freetext=False` -> header chỉ cần 120-185px (tuỳ tỉ lệ)
    # thay vì 196-307px; `has_qr=False` -> bottom chỉ cần 30-70px thay vì 92-174px.
    # CHỈ co nhánh False -- nhánh True (has_freetext=True/has_qr=True) giữ NGUYÊN y hệt
    # gốc, không đo lại (đo thử cho thấy 16:9's bottom có thể đã khá sát mép ngay cả ở
    # nhánh gốc -- ngoài phạm vi sửa lần này, không tự ý đổi).
    if w / h < 0.7:  # 9:16 Narrow Portrait
        head_h = 0.30 if has_freetext else 0.22
        bot_h = 0.17 if has_qr else 0.11
    elif w / h >= 1.5:  # 16:9 Wide Landscape
        head_h = 0.34 if has_freetext else 0.28
        bot_h = 0.16 if has_qr else 0.115
    else:  # 1:1, 4:5
        aspect = w / h
        if aspect >= 0.95:  # 1:1
            head_h = 0.30 if has_freetext else 0.18
        else:  # 4:5 (cột hẹp hơn -> badge/hero/subhead wrap nhiều hơn, cần margin lớn hơn 1:1)
            head_h = 0.30 if has_freetext else 0.225
        bot_h = 0.14 if has_qr else 0.075

    return {
        "header": _frac(0.05, 0.0, 0.95, head_h, w, h),
        "bottom": _frac(0.04, 1.0 - bot_h, 0.96, 1.0, w, h),
    }


def _menu_price_board(w: float, h: float, orientation: str = "left") -> Dict[str, Rect]:
    # Bảng Giá / Menu Nhiều Dòng (Menu Price Board):
    # Cột dọc chứa Hero + Badge + danh sách món/giá (extra_texts, mỗi dòng 1 món) +
    # Store/QR/CTA ở đáy -- cùng họ với split_left/split_right (cột full-height, mirror
    # trái/phải) nhưng rộng hơn 1 chút vì nội dung dạng list cần nhiều chỗ ngang hơn để
    # tên món + giá không bị bó hẹp. Vẫn luôn <= 50% diện tích theo yêu cầu.
    if w / h < 0.7:  # 9:16 Narrow Portrait
        frac_w = 0.50
    elif w / h >= 1.5:  # 16:9 Wide Landscape
        frac_w = 0.40
    else:  # 1:1, 4:5
        frac_w = 0.44

    if orientation in ("right", "top_right", "bottom_right"):
        return {"content": _frac(1.0 - frac_w, 0.0, 1.0, 1.0, w, h)}
    return {"content": _frac(0.0, 0.0, frac_w, 1.0, w, h)}


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
    "l_frame_showcase": _l_frame_showcase,
    "grand_opening_banner": _grand_opening_banner,
    "menu_price_board": _menu_price_board,
}


_DENSITY_AWARE_TEMPLATES: set = set()
# TẠM THỜI TẮT (2026-09-21) -- đã thử bật cho cả 5 template ("luxury_centered_card",
# "customer_feedback_card", "lifestyle_corner_pod", "recruitment_board",
# "l_frame_showcase") nhưng full stress-test 16 case/template phát hiện density co
# giãn theo công thức DÙNG CHUNG (compute_density_score) làm zone co quá tay, gây
# tràn chữ thật hàng loạt (customer_feedback_card 20/16 lỗi, luxury_centered_card
# 9/16 lỗi, recruitment_board 14/16 lỗi, lifestyle_corner_pod gần như toàn bộ 16/16
# lỗi -- chỉ l_frame_showcase pass sạch 16/16). Nguyên nhân gốc: công thức tính điểm
# density cộng dồn theo field (testimonial/rating/badge/subhead/extra_texts/cta/
# store_items) không hề "biết" template nào THỰC SỰ dùng field nào -- vd
# lifestyle_corner_pod/recruitment_board không có testimonial/rating nên KHÔNG BAO
# GIỜ đạt điểm density gần 1.0 dù nội dung đã dày đặc, khiến zone luôn bị co dù
# đang ở case "nhiều chữ nhất". Hạ tầng (compute_density_score, `get_zones(...,
# density=...)`, `compute_plan_content_density()` trong renderer.py, wiring qua
# mask_engine.py/demo_server.py) VẪN GIỮ NGUYÊN, chỉ tắt qua set rỗng này -- khi làm
# lại, cần 1 công thức RIÊNG cho từng template (biết chính xác field nào thực sự áp
# dụng cho template đó) thay vì 1 công thức chung áp cho tất cả, và phải test đủ
# 16 case/template TRƯỚC KHI bật lại bất kỳ template nào vào set này.


# Cờ hiện diện (has_qr/has_footer/has_message/has_freetext) mà hàm zone của template nhận
# -- khai báo trong catalog.py (`slots[...]["drives_geometry"]`), thay cho 4 set hard-code
# cũ. Khác _DENSITY_AWARE_TEMPLATES (1 điểm số density chung, đã CHỨNG MINH SAI): đây là
# boolean hiện diện thật của đúng field làm driver kích thước chính, mỗi template tự khai
# báo field nào (customer_feedback_card: has_footer = qr HOẶC cta HOẶC store, vì riêng QR
# không phải driver ở đó -- xem compute_customer_feedback_budget()).
def geometry_drivers(template: str) -> Dict[str, Tuple[str, ...]]:
    """{cờ: các slot kích hoạt cờ đó} theo khai báo trong catalog."""
    out: Dict[str, Tuple[str, ...]] = {}
    for slot, spec in TEMPLATE_CATALOG.get(template, {}).get("slots", {}).items():
        for flag in spec.get("drives_geometry", ()):
            out[flag] = out.get(flag, ()) + (slot,)
    return out


def get_zones(
    template: str,
    width: int,
    height: int,
    orientation: Optional[str] = None,
    density: float = 1.0,
    has_qr: Optional[bool] = None,
    has_footer: Optional[bool] = None,
    has_message: Optional[bool] = None,
    has_freetext: Optional[bool] = None,
) -> Dict[str, Rect]:
    """Trả về {tên_vùng: (x1,y1,x2,y2) px} cho template này ở đúng width/height.
    Dict rỗng nghĩa là zero-mask (vd luxury_centered_card) hoặc template lạ.

    `density` (0.35-1.0, xem compute_density_score()) chỉ có tác dụng với các
    template trong `_DENSITY_AWARE_TEMPLATES` (dạng 1 card/pod độc lập) -- co giãn
    zone theo đúng lượng nội dung thật thay vì luôn cấp kích thước tối đa cố định.
    Mặc định 1.0 (kích thước tối đa, hành vi CŨ y hệt) cho mọi lời gọi chưa truyền
    density -- tương thích ngược hoàn toàn.

    `has_qr`/`has_footer`/`has_message`/`has_freetext` (bool | None) chỉ có tác dụng với
    template khai báo cờ đó trong catalog (xem geometry_drivers()). `None` (mặc định,
    chưa truyền) -> coi như True (kích thước tối đa, hành vi CŨ y hệt)."""
    fn = _GEOMETRY_FUNCS.get(template)
    if fn is None:
        return {}
    density_kwarg = {"density": density} if template in _DENSITY_AWARE_TEMPLATES else {}
    given = {"has_qr": has_qr, "has_footer": has_footer, "has_message": has_message, "has_freetext": has_freetext}
    presence_kwarg: Dict[str, bool] = {
        flag: True if given[flag] is None else given[flag] for flag in geometry_drivers(template)
    }
    default_orientation = TEMPLATE_CATALOG.get(template, {}).get("default_orientation")
    orientation_kwarg = {"orientation": orientation or default_orientation} if default_orientation else {}
    return fn(float(width), float(height), **orientation_kwarg, **density_kwarg, **presence_kwarg)


__all__ = ["Rect", "geometry_drivers", "get_zones"]
