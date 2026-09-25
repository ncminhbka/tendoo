"""tendoo_core -- hạ tầng dùng chung của Tendoo AI.

Ba thành phần, tất cả đều được `tendoo_v3` dùng ở mọi lần render:

  - `colors`          : phối màu tự động + cưỡng chế tương phản WCAG 2.1
                        (`ensure_contrast` chính là Luật 3 trong ROADMAP.md)
  - `fonts`           : danh mục 19 font tiếng Việt + phân giải font
  - `poster_renderer` : engine Playwright/Chromium headless HTML -> PNG
  - `palette`         : dataclass `ColorPalette`

Tách khỏi package `tendoo` (v1) ngày 2026-09-25: đây là phần DUY NHẤT của v1 mà
v3 thật sự cần (~1.680/7.905 dòng). Phần còn lại của v1 ở lại repo cũ.
"""
