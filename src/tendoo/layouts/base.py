"""
Base interfaces and data structures for Tendoo Layouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PosterContent:
    """Structured form content submitted by user."""
    headline: str
    pre_header: str = ""
    slogan: str = ""
    offer_main: str = ""
    offer_sub: str = ""
    dates: str = ""
    brand: str = ""
    hotline: str = ""
    applicable: str = ""
    address: str = ""
    website_link: str = ""
    qr_data_uri: str = ""
    category: str = "promo"
    text_effect: str = "auto"
    custom_css: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "headline": self.headline,
            "pre_header": self.pre_header,
            "slogan": self.slogan,
            "offer_main": self.offer_main,
            "offer_sub": self.offer_sub,
            "dates": self.dates,
            "brand": self.brand,
            "hotline": self.hotline,
            "applicable": self.applicable,
            "address": self.address,
            "website_link": self.website_link,
            "qr_data_uri": self.qr_data_uri,
            "category": self.category,
            "text_effect": self.text_effect,
            "custom_css": self.custom_css,
        }


@dataclass
class ColorPalette:
    """Calculated color harmony tokens derived from the background image."""
    is_dark: bool
    luminance: float
    hue: int
    comp_hue: int
    headline_color: str
    headline_is_gradient: bool = False
    sub_color: str = "#E0E0E0"
    text_shadow: str = "0 2px 8px rgba(0, 0, 0, 0.6)"
    badge_bg: str = "#FF5722"
    badge_text: str = "#FFFFFF"
    badge_shadow: str = "0 4px 14px rgba(255, 87, 34, 0.4)"
    glass_bg: str = "rgba(255, 255, 255, 0.12)"
    glass_border: str = "rgba(255, 255, 255, 0.25)"
    accent_color: str = "#FFB300"
    footer_text: str = "#B0BEC5"
    footer_bg: str = "rgba(0, 0, 0, 0.5)"

    def to_css_vars(self) -> str:
        """Serializes tokens into CSS root variables."""
        return f"""
        :root {{
            --is-dark: {'1' if self.is_dark else '0'};
            --headline-color: {self.headline_color};
            --sub-color: {self.sub_color};
            --text-shadow: {self.text_shadow};
            --badge-bg: {self.badge_bg};
            --badge-text: {self.badge_text};
            --badge-shadow: {self.badge_shadow};
            --glass-bg: {self.glass_bg};
            --glass-border: {self.glass_border};
            --accent-color: {self.accent_color};
            --footer-text: {self.footer_text};
            --footer-bg: {self.footer_bg};
        }}
        """.strip()


class BaseLayout(ABC):
    """Abstract interface that every layout topology must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine identifier, e.g. 'top_dome', 'center_hourglass'."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for UI button."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief explanation of when to use this layout."""
        pass

    @abstractmethod
    def generate_mask(
        self,
        width: int,
        height: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Generates 2D float32 numpy array [H, W] in [0.0, 1.0].
        1.0 means 100% velocity corridor prompt (text safe zone).
        0.0 means 100% unconstrained scene prompt (product / background).
        """
        pass

    @abstractmethod
    def get_corridor_prompt(self, style_hint: str = "daylight") -> str:
        """Returns procedural corridor prompt tailored for this layout's safe zone."""
        pass

    @abstractmethod
    def get_safe_zone(self) -> Tuple[float, float, float, float]:
        """
        Returns normalized (y1, x1, y2, x2) bounding box of the primary text safe zone.
        Used by the Color Wheel Engine to crop and extract harmonious palette.
        """
        pass

    @abstractmethod
    def render_html(
        self,
        content: PosterContent,
        palette: ColorPalette,
        bg_data_uri: str,
        width: int,
        height: int,
    ) -> str:
        """Renders complete HTML5/CSS3 document for Playwright rendering."""
        pass
