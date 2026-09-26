"""
src/tendoo_v3/hero_markup.py

Tách `hero_parts` TẤT ĐỊNH (ROADMAP §2.3, GĐ 3) -- hai việc:

1. "LLM lý tưởng" để ĐO (cô lập LLM khỏi phép đo): probe/calibrate chạy chính sách cỡ chữ trên
   đúng loại markup một LLM giỏi sẽ trả, mà không cần gọi LLM thật.
2. Đường dự phòng khi không có LLM (`fallback_heuristic_planner`) cũng có tiêu đề nhiều cỡ.

Chỉ tách khi có TÍN HIỆU RÕ -- con số kèm đơn vị, hoặc cụm từ móc thương mại. Tiêu đề thuần tên
sản phẩm ("GIÀY TÂY OXFORD DA BÒ Ý CAO CẤP") giữ PHẲNG: chọn chữ nào to ở đó là phán đoán thiết
kế của LLM, quy tắc không được đoán thay. Kết quả luôn qua Cổng 1 (nối lại == hero).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Cụm từ móc: vai trò "stat" không chỉ là con số (§2.3: "GRAND OPENING", "TUYỂN DỤNG", "MIỄN PHÍ").
# Thứ tự = ưu tiên (cụm dài trước để "MUA 1 TẶNG 1" thắng "MUA 1").
HOOK_PHRASES = (
    "MUA 1 TẶNG 1", "MUA 2 TẶNG 1", "BLACK FRIDAY", "FLASH SALE", "SIÊU SALE", "GRAND OPENING",
    "KHAI TRƯƠNG", "TUYỂN DỤNG", "CHIÊU MỘ", "MIỄN PHÍ", "ĐỒNG GIÁ", "SALE", "TUYỂN",
)
# Đơn vị/chữ đi kèm con số, gộp vào cùng đoạn stat ("7 BƯỚC", "5 SAO", "6 TUẦN").
UNIT_WORDS = {"BƯỚC", "SAO", "NGÀY", "TUẦN", "THÁNG", "GIỜ", "LẦN", "HỐ", "FT", "TRIỆU", "NGHÌN", "NGÀN", "TỶ", "ĐỒNG", "VNĐ", "K", "%"}
_UNIT_WORDS = UNIT_WORDS
# Từ ghép 2 âm tiết thông dụng trong quảng cáo -- tách qua 2 dòng ("CÀ | PHÊ") đọc như 2 từ rời.
# Danh sách THU HẸP có chủ đích (không phải từ điển): dùng để giữ liền khi ngắt dòng (GĐ 7a) và
# để probe_line_breaks.py đếm lỗi. Cụm 3 âm tiết ghi dạng đủ, tách thành các cặp liền kề.
_COMPOUND_PHRASES = "ƯU ĐÃI|KHUYẾN MÃI|GIẢM GIÁ|MIỄN PHÍ|KHAI TRƯƠNG|SẢN PHẨM|KHÁCH HÀNG|CỬA HÀNG|DỊCH VỤ|CHẤT LƯỢNG|CAO CẤP|THỜI TRANG|CÔNG NGHỆ|TRẢI NGHIỆM|CÀ PHÊ|TRÀ SỮA|ẨM THỰC|NHÀ HÀNG|THỰC ĐƠN|HẢI SẢN|BẢO HÀNH|CHÍNH HÃNG|GIAO HÀNG|TOÀN QUỐC|HỆ THỐNG|TUYỂN DỤNG|NHÂN VIÊN|CHUYÊN VIÊN|KỸ SƯ|DOANH NGHIỆP|THƯƠNG HIỆU|BỘ SƯU TẬP|SANG TRỌNG|ĐẲNG CẤP|THƯỢNG LƯU|HOÀNG GIA|NGHỈ DƯỠNG|DU LỊCH|KỲ NGHỈ|LÀM ĐẸP|THẨM MỸ|LÀN DA|CHĂM SÓC|LIỆU TRÌNH|SỨC KHỎE|PHÒNG KHÁM|NHA KHOA|BÁC SĨ|HỌC VIÊN|KHÓA HỌC|ĐÀO TẠO|QUY TRÌNH|GIẢI PHÁP|TRÍ TUỆ|NHÂN TẠO|ĐIỆN THOẠI|ĐỒNG HỒ|NƯỚC HOA|MỸ PHẨM|TRANG SỨC|KIM CƯƠNG|BIỆT THỰ|CĂN HỘ|BẤT ĐỘNG SẢN|NỘI THẤT|KHÔNG GIAN|THIÊN NHIÊN|TỰ NHIÊN|NGUYÊN BẢN|THỦ CÔNG|ĐỘC QUYỀN|ĐẶC BIỆT|HẤP DẪN|TUYỆT VỜI|HOÀN HẢO|TẬN TÂM|UY TÍN|HÀI LÒNG|TIN DÙNG|ĐẶT HÀNG|MUA SẮM|ĐĂNG KÝ|LIÊN HỆ|THÀNH VIÊN|TÍCH ĐIỂM|QUÀ TẶNG|SỐ LƯỢNG|CÓ HẠN|CUỐI TUẦN|HÔM NAY|ĐÊM NAY|MÙA HÈ|MÙA THU|GIÁNG SINH|TRUNG THU|NĂM MỚI|LỄ HỘI|SỰ KIỆN|TRIỂN LÃM|HỘI NGHỊ|THỂ THAO|THỂ HÌNH|VÓC DÁNG|TƯƠI SÁNG|RẠNG RỠ|THANH LỊCH|TINH TẾ|HIỆN ĐẠI|THÔNG MINH|SIÊU TỐC|ĐỈNH CAO|BÙNG NỔ|TƯNG BỪNG|CHÀO ĐÓN|TRI ÂN|CẢM ƠN|HÀ NỘI|SÀI GÒN|ĐÀ NẴNG|ĐÀ LẠT|PHÚ QUỐC|NHA TRANG".split("|")
COMPOUNDS = {f"{a} {b}" for ph in _COMPOUND_PHRASES for a, b in zip(ph.split(), ph.split()[1:])}
# Con số "đáng làm neo": có ký hiệu (%, K, Đ, +, X, HZ...) hoặc dạng 3N2Đ / 24/7 / 1:1 / 99.9.
_NUM_TOKEN = re.compile(r"^[-–+]?\d[\d.,:/]*(%|K|Đ|đ|\+|X|HZ|FT|N\d+Đ|H)?$", re.IGNORECASE)
_YEAR = re.compile(r"^(19|20)\d\d$")  # năm ("2026") là thông tin, không phải điểm neo


def _numeric_stat(words: List[str]) -> Optional[Tuple[int, int]]:
    """[i, j) của đoạn số làm neo: ưu tiên số có '%', rồi số có ký hiệu/đơn vị/từ dẫn. Số TRẦN
    ("CƠ SỞ 5 VIỆN") và năm ("2026") không làm neo -- thua cả cụm từ móc."""
    cands = []
    for i, w in enumerate(words):
        m = _NUM_TOKEN.match(w)
        if not m or _YEAR.match(w):
            continue
        j = i + 1 + (1 if i + 1 < len(words) and words[i + 1] in _UNIT_WORDS else 0)
        lead = i > 0 and words[i - 1] in ("SỐ", "THỨ")
        if not (m.group(1) or j > i + 1 or lead):
            continue  # số trần
        cands.append((0 if "%" in w else 1, i, (i - 1 if lead else i, j)))
    return min(cands)[2] if cands else None


def _hook_stat(words: List[str]) -> Optional[Tuple[int, int]]:
    for phrase in HOOK_PHRASES:
        p = phrase.split()
        for i in range(len(words) - len(p) + 1):
            if words[i:i + len(p)] == p:
                return i, i + len(p)
    return None


def suggest_hero_parts(hero: str) -> List[Dict[str, Any]]:
    """hero_parts cho `hero`, hoặc [] (giữ phẳng) nếu không có tín hiệu rõ."""
    words = (hero or "").split()
    if len(words) < 2:
        return []
    upper = [w.upper() for w in words]  # so khớp không phân biệt hoa/thường, xuất nguyên văn
    span = _numeric_stat(upper) or _hook_stat(upper)
    if span is None:
        return []
    i, j = span
    if i == 0 and j == len(words):
        return []  # cả câu là điểm neo -> phẳng đã là đúng
    parts = []
    if i > 0:
        parts.append({"t": " ".join(words[:i]), "role": "prefix"})
    parts.append({"t": " ".join(words[i:j]), "role": "stat", "emphasis": "accent"})
    if j < len(words):
        parts.append({"t": " ".join(words[j:]), "role": "suffix"})
    return parts


__all__ = ["COMPOUNDS", "HOOK_PHRASES", "UNIT_WORDS", "suggest_hero_parts"]
