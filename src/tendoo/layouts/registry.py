"""
src/tendoo/layouts/registry.py

Unified Layout Registry & Facade.
Directs modern layout requests to the Tendoo Omni-Block Engine (OmniBlockLayout)
while lazily delegating legacy template requests to `tendoo_legacy.layouts`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from tendoo.core.base import BaseLayout
from tendoo.engine.layout import OmniBlockLayout

_REGISTRY: Dict[str, BaseLayout] = {}
_OMNI_INSTANCE = OmniBlockLayout()

# Register OmniBlockLayout as the primary engine
_REGISTRY["omni"] = _OMNI_INSTANCE


def register_layout(layout: BaseLayout) -> None:
    """Registers a layout instance by its unique name."""
    _REGISTRY[layout.name] = layout


def get_layout(name: Union[str, dict]) -> BaseLayout:
    """Retrieves a registered layout by name (or metadata dict). Defaults to OmniBlockLayout."""
    if isinstance(name, dict) and "name" in name:
        name = name["name"]
    name_str = str(name).strip().lower()

    if name_str in _REGISTRY:
        return _REGISTRY[name_str]

    # Load legacy layouts on explicit request
    try:
        import sys
        from pathlib import Path
        src_dir = Path(__file__).resolve().parent.parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        if name_str == "top_dome":
            from tendoo_legacy.layouts.top_dome.layout import TopDomeLayout
            return TopDomeLayout()
        elif name_str == "bottom_platform":
            from tendoo_legacy.layouts.bottom_platform.layout import BottomPlatformLayout
            return BottomPlatformLayout()
        elif name_str == "center_hourglass":
            from tendoo_legacy.layouts.center_hourglass.layout import CenterHourglassLayout
            return CenterHourglassLayout()
        elif name_str == "split_column":
            from tendoo_legacy.layouts.split_column.layout import SplitColumnLayout
            return SplitColumnLayout()
        elif name_str == "diagonal_slash":
            from tendoo_legacy.layouts.diagonal_slash.layout import DiagonalSlashLayout
            return DiagonalSlashLayout()
        elif name_str == "l_frame":
            from tendoo_legacy.layouts.l_frame.layout import LFrameLayout
            return LFrameLayout()
        elif name_str == "freeform":
            from tendoo_legacy.layouts.freeform.layout import FreeformLayout
            return FreeformLayout()
    except Exception:
        pass

    # Default fallback to OmniBlockLayout
    return _OMNI_INSTANCE


def list_layouts() -> List[Dict[str, str]]:
    """Returns metadata list of available layouts for UI display and style matching."""
    items = [
        {
            "name": "omni",
            "display_name": _OMNI_INSTANCE.display_name,
            "description": _OMNI_INSTANCE.description,
        }
    ]
    legacy_names = [
        "top_dome",
        "bottom_platform",
        "center_hourglass",
        "split_column",
        "diagonal_slash",
        "l_frame",
        "freeform",
    ]
    for name in legacy_names:
        try:
            layout = get_layout(name)
            items.append({
                "name": layout.name,
                "display_name": layout.display_name,
                "description": layout.description,
            })
        except Exception:
            pass
    return items
