"""
src/tendoo/layouts

(2026-09-13: gói này giờ chỉ còn giữ `layouts/base.py` -- cầu nối tương thích BẮT BUỘC cho
`src/tendoo_legacy/` (7 layout + `component_engine.py` import cố định
`from tendoo.layouts.base import BaseLayout, ColorPalette, PosterContent, *_ICON_SVG`).
Toàn bộ logic thật trước đây nằm rải rác ở `layouts/color_engine.py`, `font_engine.py`,
`style_matcher.py`, `text_engine.py`, `registry.py` đã chuyển hẳn vào `tendoo.core`/
`tendoo.engine` (nguồn chân lý DUY NHẤT bây giờ) -- import từ đó, không import qua gói
này, trừ khi bạn đang bảo trì `tendoo_legacy`.
"""
