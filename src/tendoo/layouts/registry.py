"""
src/tendoo/layouts/registry.py

Pure OmniBlock Layout Registry:
===============================
- Đăng ký và phục vụ duy nhất động cơ dàn trang OmniBlockLayout (`omni`).
- Thực hiện triệt để triết lý: "Everything is an Adaptive Block".
- 100% độc lập, loại bỏ toàn bộ cơ chế lazy loading và tàn dư template cũ.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from tendoo.core.base import BaseLayout

_REGISTRY: Dict[str, BaseLayout] = {}
_OMNI_INSTANCE: Optional[BaseLayout] = None


def _get_omni_instance() -> BaseLayout:
    """Khởi tạo và lưu cache duy nhất một instance của OmniBlockLayout."""
    global _OMNI_INSTANCE
    if _OMNI_INSTANCE is None:
        from tendoo.engine.layout import OmniBlockLayout
        _OMNI_INSTANCE = OmniBlockLayout()
        _REGISTRY["omni"] = _OMNI_INSTANCE
    return _OMNI_INSTANCE


def register_layout(layout: BaseLayout) -> None:
    """Đăng ký một instance layout mới vào Registry."""
    _REGISTRY[layout.name] = layout


def get_layout(name: Union[str, dict, None] = None) -> BaseLayout:
    """
    Trả về OmniBlockLayout - động cơ dàn trang đa khối tự thích ứng duy nhất của Tendoo AI.
    
    TẠI SAO CẦN LÀM:
    - Toàn bộ các yêu cầu sinh poster trong hệ thống đều được phục vụ bởi OmniBlockLayout.
    - Không còn bất kỳ rủi ro vỡ template cứng hay lỗi tải thư viện phụ thuộc cũ.
    """
    return _get_omni_instance()


def list_layouts() -> List[Dict[str, str]]:
    """
    Trả về danh sách layout chính thức phục vụ hiển thị trên giao diện người dùng.
    """
    omni = _get_omni_instance()
    return [
        {
            "name": "omni",
            "display_name": omni.display_name,
            "description": omni.description,
        }
    ]


__all__ = [
    "get_layout",
    "list_layouts",
    "register_layout",
]
