"""
src/tendoo/core

Tầng cốt lõi đồ họa & typography cho hệ thống Tendoo AI:
========================================================
- base: BaseLayout, PosterContent, ColorPalette, SVG icons vector
- fonts: Quản lý Font Catalog 19 họ font thương mại tiếng Việt & font-face CSS
- colors: Color harmony (đơn sắc, tương phản, tam giác màu) & WCAG contrast ratio
- typography: Bounding box math, greedy word-wrapping, PIL font fitting & Vietnamese line balancing
- components: SVG icons vector & visual building blocks (O(1) semantic icon lookup)
- style: Category slot definitions, style presets & color guidance injection

TẠI SAO CẦN TẦNG CORE ĐỘC LẬP:
- Tách biệt rõ ràng giữa Business Logic / Data Models (core) và Spatial Layout / Rendering (engine & renderer).
- Cho phép bất kỳ module nào (LLM planner, API server, layout generator, visual test suite)
  dễ dàng import các hằng số, kiểu dữ liệu, và công cụ typography mà không bị dính vòng lặp phụ thuộc (circular imports).
"""

# (2026-09-12 cleanup: this package used to re-export ~30 names from its submodules
# here so callers could `from tendoo.core import X` -- confirmed via a repo-wide grep
# that NOTHING anywhere (src/, tests/, scripts/, tendoo_legacy/) actually imports
# through that path; every real caller imports the specific submodule directly, e.g.
# `from tendoo.core.base import PosterContent` or `from tendoo.core.colors import
# analyze_color_harmony`. Removed the dead aggregator; the submodules themselves are
# unaffected and remain directly importable exactly as before.)
