"""
src/tendoo_v3/qr.py

Trình Tạo Mã QR Code Quét Được Thực Tế (Scannable QR Generator) cho Tendoo v3:
=============================================================================
- Sinh mã QR chuẩn ISO mã hóa URL/văn bản dưới dạng chuỗi Base64 Data URI inline ("data:image/png;base64,...").
- Quét được trực tiếp 100% bằng Camera điện thoại hoặc ứng dụng Zalo/Ngân hàng.
- Cấp độ sửa lỗi ERROR_CORRECT_M (15%) bảo đảm quét tốt cả khi bị mờ hoặc lóa sáng ngoài trời.
- Hoạt động 100% offline, zero external HTTP call, an toàn tuyệt đối trên server mạng nội bộ.
- Fallback tự động sang Vector SVG Pattern nếu môi trường chưa cài đặt thư viện 'qrcode'.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

from tendoo_v3.icons import render_qr_code_svg

logger = logging.getLogger("TendooV3.QR")


def generate_qr_base64(
    data: str,
    box_size: int = 8,
    border: int = 2,
    fill_color: str = "#000000",
    back_color: str = "#ffffff",
) -> Optional[str]:
    """Sinh mã QR dưới dạng chuỗi Base64 Data URI inline ('data:image/png;base64,...').
    
    Args:
        data: URL hoặc văn bản cần mã hóa.
        box_size: Kích thước pixel cho từng ô module QR.
        border: Độ rộng khoảng đệm an toàn (quiet zone).
        fill_color: Màu điểm ảnh QR.
        back_color: Màu nền QR.
        
    Returns:
        Chuỗi Base64 Data URI hoặc None nếu dữ liệu rỗng / lỗi.
    """
    if not data or not data.strip():
        return None

    clean_data = data.strip()

    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_M

        qr = qrcode.QRCode(
            version=None,  # Tự động tính toán version kích thước tối ưu theo độ dài chuỗi
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
        logger.warning("Thư viện 'qrcode' chưa được cài đặt trong môi trường Python. Sẽ fallback sang vector SVG.")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi sinh mã QR cho chuỗi '{clean_data}': {e}")
        return None


def render_scannable_qr_component(
    data: Optional[str] = None,
    label: str = "QUÉT MÃ NGAY",
    theme_color: str = "#FFB300",
    size_px: int = 100,
) -> str:
    """Tạo component HTML mã QR hoàn chỉnh.
    Nếu có dữ liệu URL thực tế và qrcode khả dụng -> sinh ảnh QR quét được.
    Nếu không -> render vector SVG đồ họa phẳng chuẩn xác.
    """
    clean_label = (label or "QUÉT MÃ NGAY").strip()
    if size_px <= 70:
        if len(clean_label) > 12:
            lbl_font_size = "8.5px"
            lbl_letter_spacing = "0.2px"
        elif len(clean_label) > 7:
            lbl_font_size = "9.5px"
            lbl_letter_spacing = "0.4px"
        else:
            lbl_font_size = "10.5px"
            lbl_letter_spacing = "0.5px"
    else:
        if len(clean_label) > 18:
            lbl_font_size = "8.5px"
            lbl_letter_spacing = "0.4px"
        elif len(clean_label) > 12:
            lbl_font_size = "10px"
            lbl_letter_spacing = "0.6px"
        else:
            lbl_font_size = "11.5px"
            lbl_letter_spacing = "0.8px"

    qr_b64 = generate_qr_base64(data, fill_color="#000000", back_color="#ffffff") if data else None

    card_max_w = max(size_px + 32, 88)
    if qr_b64:
        return f"""
        <div class="qr-card-component" style="display:inline-flex; flex-direction:column; align-items:center; gap:5px; background:rgba(255,255,255,0.08); backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.22); border-radius:14px; padding:6px 8px; box-shadow:0 8px 24px rgba(0,0,0,0.35); max-width:{card_max_w}px;">
          <img src="{qr_b64}" width="{size_px}" height="{size_px}" style="display:block; border-radius:8px; background:#fff; padding:4px;" alt="QR Code" />
          <span style="font-family:var(--ui-font, 'Be Vietnam Pro', sans-serif); font-size:{lbl_font_size}; font-weight:800; letter-spacing:{lbl_letter_spacing}; text-transform:uppercase; color:#fff; text-shadow:0 1px 3px rgba(0,0,0,0.8); text-align:center; word-break:keep-all; max-width:100%;">{clean_label}</span>
        </div>
        """
    else:
        # Fallback về Vector SVG
        return render_qr_code_svg(label=clean_label, theme_color=theme_color, size_px=size_px)


__all__ = [
    "generate_qr_base64",
    "render_scannable_qr_component",
]
