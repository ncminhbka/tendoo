"""
src/tendoo/qr.py

QR Code Generator for Tendoo AI Commercial Posters:
===================================================
- Tạo mã QR vector sắc nét, tương phản cao dưới dạng chuỗi Base64 Data URI inline.
- Nhúng trực tiếp vào Master Template HTML của OmniBlockLayout và render qua Playwright Chromium.

TẠI SAO CẦN MODULE NÀY TRONG HỆ THỐNG TENDOO AI:
1. TẠI SAO BẮT BUỘC DÙNG BASE64 DATA URI INLINE THAY VÌ FILE ẢNH TẠM HOẶC API NGOÀI:
   - Nếu gọi API tạo QR bên ngoài (như Google Charts QR API), hệ thống sẽ gãy hoàn toàn khi chạy
     trong mạng nội bộ không có kết nối internet (Air-gapped Server 2x A30).
   - Nếu lưu thành file ảnh PNG tạm thời trên ổ đĩa (`temp/qr_123.png`), hệ thống tiềm ẩn rủi ro:
     + Tranh chấp tài nguyên (race condition) khi có nhiều người dùng tạo poster cùng lúc.
     + Trình duyệt Chromium có thể chụp canvas trước khi file kịp đồng bộ từ đĩa vào bộ nhớ cache.
   - Nhúng chuỗi Base64 Data URI trực tiếp vào thẻ `<img src="data:image/png;base64,...">` đảm bảo:
     + Hoạt động 100% offline, phi trạng thái (stateless), an toàn tuyệt đối cho môi trường đa luồng.
     + Không để lại rác tạm thời trên ổ cứng server.

2. TẠI SAO DÙNG CẤP ĐỘ SỬA LỖI `ERROR_CORRECT_M` (15% Error Correction):
   - Cấp độ M (Medium) cho phép camera điện thoại quét thành công mã QR ngay cả khi 15% bề mặt
     bị mờ hoặc lóa sáng do góc chụp thực tế ngoài đời.
   - Giữ cho ma trận điểm ảnh (module grid) không quá dày đặc, giúp mã QR hiển thị rõ ràng
     ở kích thước nhỏ (60x60px) trên poster điện thoại.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger("Tendoo.QR")


def generate_qr_base64(
    data: str,
    box_size: int = 8,
    border: int = 2,
    fill_color: str = "#000000",
    back_color: str = "#ffffff",
) -> Optional[str]:
    """
    Sinh mã QR dưới dạng chuỗi Data URI Base64 ("data:image/png;base64,...").
    
    Args:
        data: URL hoặc văn bản cần mã hóa.
        box_size: Kích thước pixel của từng ô module QR.
        border: Độ dày đường viền tĩnh an toàn (quiet zone border).
        fill_color: Màu pixel mã QR (mặc định đen).
        back_color: Màu nền mã QR (mặc định trắng tinh).
        
    Returns:
        Chuỗi Base64 Data URI hoặc None nếu dữ liệu rỗng hoặc có lỗi.
    """
    if not data or not data.strip():
        return None

    clean_data = data.strip()

    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_M

        qr = qrcode.QRCode(
            version=None,  # Tự động tính toán version kích thước tối ưu dựa trên độ dài chuỗi
            error_correction=ERROR_CORRECT_M,
            box_size=box_size,
            border=border,
        )
        qr.add_data(clean_data)
        qr.make(fit=True)

        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    except ImportError:
        logger.warning("Thư viện 'qrcode' chưa được cài đặt. Không thể sinh mã QR.")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi sinh mã QR cho chuỗi '{clean_data}': {e}")
        return None


__all__ = [
    "generate_qr_base64",
]

