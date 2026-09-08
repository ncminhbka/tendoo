"""
Layout Registry and Factory.

Centralizes discovery, registration, and dispatch of all poster layout topologies.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from tendoo.layouts.base import BaseLayout
from tendoo.layouts.bottom_platform.layout import BottomPlatformLayout
from tendoo.layouts.center_hourglass.layout import CenterHourglassLayout
from tendoo.layouts.diagonal_slash.layout import DiagonalSlashLayout
from tendoo.layouts.l_frame.layout import LFrameLayout
from tendoo.layouts.split_column.layout import SplitColumnLayout
from tendoo.layouts.top_dome.layout import TopDomeLayout


_REGISTRY: Dict[str, BaseLayout] = {}


def register_layout(layout: BaseLayout) -> None:
    """Registers a layout instance by its unique name."""
    _REGISTRY[layout.name] = layout


def get_layout(name: str | dict) -> BaseLayout:
    """Retrieves a registered layout by name (or metadata dict). Raises KeyError if not found."""
    if isinstance(name, dict) and "name" in name:
        name = name["name"]
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Unknown layout '{name}'. Available layouts: {available}")
    return _REGISTRY[name]


def list_layouts() -> List[Dict[str, str]]:
    """Returns metadata list of all available layouts for UI display."""
    return [
        {
            "name": layout.name,
            "display_name": layout.display_name,
            "description": layout.description,
        }
        for layout in _REGISTRY.values()
    ]


# Auto-register canonical layouts
register_layout(TopDomeLayout())
register_layout(CenterHourglassLayout())
register_layout(BottomPlatformLayout())
register_layout(SplitColumnLayout())
register_layout(DiagonalSlashLayout())
register_layout(LFrameLayout())


