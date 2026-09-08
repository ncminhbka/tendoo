"""
src/tendoo/qr.py

QR Code Generator for Tendoo AI Commercial Posters.
Generates crisp, high-contrast QR codes as Base64 Data URIs suitable
for direct injection into HTML templates and rendering via Playwright.
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
    """Generates a QR code image from string data and returns a base64 data URI.
    
    Args:
        data: URL or text to encode.
        box_size: Pixel size of each QR box/module.
        border: Number of boxes for quiet zone border.
        fill_color: Foreground hex color or name.
        back_color: Background hex color or name.
        
    Returns:
        Base64 Data URI string "data:image/png;base64,...", or None on failure.
    """
    if not data or not data.strip():
        return None

    clean_data = data.strip()

    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_M

        qr = qrcode.QRCode(
            version=None,  # Auto-size based on content
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
        logger.warning("Package 'qrcode' not installed. Attempting pure-python or SVG fallback...")
        return None
    except Exception as e:
        logger.error(f"Failed to generate QR code for '{clean_data}': {e}")
        return None
