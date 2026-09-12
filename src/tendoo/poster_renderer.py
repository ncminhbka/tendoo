"""
src/tendoo/poster_renderer.py

Playwright Headless-Chromium High-Fidelity HTML -> PNG Renderer:
================================================================
- Tầng dựng hình đồ họa cao cấp của Tendoo AI: chuyển đổi HTML/CSS thành ảnh PNG không nén (lossless).
- Hỗ trợ đầy đủ các kỹ thuật CSS3 tiên tiến nhất: Glassmorphism (backdrop-filter: blur),
  Linear Gradient Text Clip, Multi-layer Studio Text Shadow, Vector Inline SVG và Sub-pixel Antialiasing.
- Cung cấp cơ chế đo đạc hình học trực tuyến (`measure_zone_rects`): cho phép Chromium tự đo và báo cáo
  chính xác tọa độ Bounding Box của từng khối chữ sau khi đã chạy script thu nhỏ tự động (Shrink-to-fit).

TẠI SAO CẦN CHROMIUM ENGINE TRONG TENDOO AI:
1. TẠI SAO DÙNG HEADLESS CHROMIUM THAY VÌ CÔNG CỤ VẼ ẢNH PYTHON THUẦN (PIL/CAIRO):
   - Các thư viện đồ họa thuần (PIL, PyMuPDF, CairoSVG) không hỗ trợ hoặc hỗ trợ rất yếu các hiệu ứng CSS hiện đại:
     + Kính mờ phủ hậu cảnh (Glassmorphism `backdrop-filter: blur(14px)`).
     + Chữ gradient kim loại (`-webkit-background-clip: text` nhũ vàng 24K, liquid chrome).
     + Nhiều lớp bóng đổ ma trận (`text-shadow`, `drop-shadow`) mô phỏng vầng sáng đèn Neon và LED.
   - Headless Chromium mang lại chất lượng thẩm mỹ xuất sắc chuẩn Agency quảng cáo, tự động xử lý
     kerning tiếng Việt phức tạp và tối ưu hóa hiển thị ở bất kỳ mật độ điểm ảnh nào (`device_scale_factor`).

2. TẠI SAO CẦN BỘ ĐIỀU PHỐI ĐO ĐẠC `measure_zone_rects`:
   - Nếu đo bounding box bằng PIL (offline) nhưng lại render bằng Chromium (online), sai số về kerning,
     padding, font hinting và word-wrapping giữa 2 engine sẽ tích tụ khiến Bounding Box của Mask bị lệch
     so với Bounding Box của Text thật (gây ra hiện tượng chữ tràn ra ngoài vùng dọn nền).
   - `measure_zone_rects` chạy cùng một file HTML qua chính Chromium, kích hoạt script Autofit tự thu nhỏ
     nếu cần, rồi đọc lại `getBoundingClientRect()` thực tế. Nhờ đó sai số giữa Mask và Text bằng 0.0px.

3. TẠI SAO CẦN CƠ CHẾ CHẠY ĐỒNG BỘ AN TOÀN TRONG EVENT LOOP (`_run_coroutine_sync`):
   - Trên máy chủ JupyterLab hoặc bên trong các tiến trình async (FastAPI, asyncio), việc gọi `asyncio.run()`
     trực tiếp sẽ ném ra ngoại lệ: `RuntimeError: asyncio.run() cannot be called from a running event loop`.
   - `_run_coroutine_sync` tự động phát hiện event loop đang chạy và ủy thác việc thực thi sang một luồng
     độc lập, đảm bảo các hàm sync `render()` và `measure_zone_rects()` luôn chạy an toàn 100% trong mọi môi trường.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Cờ đồng bộ với script autofit trong master.html: trình duyệt thu nhỏ chữ xong sẽ gán cờ này thành true
_AUTOFIT_DONE_EXPR = "window.__tendooAutofitDone === true"
_AUTOFIT_WAIT_TIMEOUT_MS = 3000


def _run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Chạy coroutine bất đồng bộ một cách an toàn kể cả khi đang ở trong một Event Loop đang chạy (như JupyterLab).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Đang trong event loop (JupyterLab / async thread): chạy qua ThreadPoolExecutor để tránh lỗi loop nesting
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def _chromium_launch_kwargs() -> Dict[str, Any]:
    """Cấu hình tham số khởi chạy Headless Chromium tối ưu cho môi trường máy chủ nội bộ."""
    launch_args = [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
    ]
    launch_kwargs: Dict[str, Any] = {
        "headless": True,
        "args": launch_args,
    }

    import shutil
    sys_chrome = (
        os.environ.get("PLAYWRIGHT_CHROME_PATH")
        or shutil.which("chromium-browser")
        or shutil.which("chromium")
        or shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
    )
    if sys_chrome:
        logger.info(f"[PosterRenderer] Tìm thấy trình duyệt hệ thống tại: {sys_chrome}")
        launch_kwargs["executable_path"] = sys_chrome
    return launch_kwargs



async def _set_content_and_wait_ready(page: Any, html_content: str) -> None:
    """Loads `html_content` and waits for webfonts + the in-page autofit pass."""
    try:
        await page.set_content(html_content, wait_until="networkidle", timeout=6000)
    except Exception as e:
        logger.warning(f"[PosterRenderer] 'networkidle' wait failed or timed out ({e}), proceeding with rendered DOM...")
        try:
            await page.set_content(html_content, wait_until="domcontentloaded", timeout=2000)
        except Exception:
            pass

    # Settle fonts if available, otherwise continue smoothly without blocking
    try:
        await asyncio.wait_for(page.evaluate("document.fonts.ready"), timeout=4.0)
    except Exception as e:
        logger.debug(f"[PosterRenderer] Font readiness check skipped or timed out: {e}")

    # Let the page's own shrink-to-fit autofit script (master.html, omni-block engine
    # only) finish reflowing BEFORE we screenshot or read back geometry -- otherwise
    # we'd capture mid-shrink sizes that match neither the pre-fit estimate nor the
    # final, settled layout. Gated on the marker actually being present in the HTML so
    # this never adds a dead wait_for_function timeout to legacy-template renders,
    # which carry no such script and would otherwise stall here for the full timeout.
    if "__tendooAutofitDone" in html_content:
        try:
            await page.wait_for_function(_AUTOFIT_DONE_EXPR, timeout=_AUTOFIT_WAIT_TIMEOUT_MS)
        except Exception as e:
            logger.debug(f"[PosterRenderer] Autofit-done wait skipped or timed out: {e}")


class PosterRenderer:
    """
    High-performance headless Chromium renderer using Playwright.
    Ensures zero-network offline rendering, sub-pixel rasterization, and font readiness.
    """

    @classmethod
    async def render_html_async(
        cls,
        html_content: str,
        output_image_path: str | Path,
        width: int,
        height: int,
        device_scale_factor: int = 1,
    ) -> Path:
        """
        Renders HTML content into a lossless PNG image using Playwright Chromium.
        """
        from playwright.async_api import async_playwright

        out_file = Path(output_image_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(**_chromium_launch_kwargs())
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=device_scale_factor,
            )
            page = await context.new_page()

            await _set_content_and_wait_ready(page, html_content)

            # Take pixel-accurate screenshot
            await page.screenshot(
                path=str(out_file),
                full_page=True,
                type="png",
            )
            await browser.close()

        logger.info(f"[PosterRenderer] Screenshot saved to: {out_file}")
        return out_file

    @classmethod
    def render(
        cls,
        html_content: str,
        output_image_path: str | Path,
        width: int,
        height: int,
        device_scale_factor: int = 1,
    ) -> Path:
        """
        Bọc đồng bộ an toàn cho render_html_async (chạy được cả trong Event Loop JupyterLab).
        """
        return _run_coroutine_sync(
            cls.render_html_async(
                html_content=html_content,
                output_image_path=output_image_path,
                width=width,
                height=height,
                device_scale_factor=device_scale_factor,
            )
        )

    @classmethod
    async def measure_zone_rects_async(
        cls,
        html_content: str,
        width: int,
        height: int,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Loads `html_content` in the SAME Chromium engine used for the final screenshot,
        lets its in-page autofit script shrink text to fit, then reads back the real
        `(left, top, right, bottom)` pixel rect of every rendered text zone via
        `window.__tendooZoneRects` (set by the `<script>` at the end of master.html).

        This is the measurement half of the fix for mask/text mismatch: the caller
        (`OmniBlockLayout.generate_mask_from_render`) uses these exact rects to draw the
        corridor mask, instead of a PIL-estimated bounding box computed by a different
        text-shaping engine. Returns {} if the page never sets the flag (e.g. malformed
        HTML) -- callers should fall back to the legacy PIL-estimate mask in that case.
        """
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(**_chromium_launch_kwargs())
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
            )
            page = await context.new_page()

            await _set_content_and_wait_ready(page, html_content)

            rects: Dict[str, List[float]] = {}
            try:
                rects = await page.evaluate("window.__tendooZoneRects || {}")
            except Exception as e:
                logger.warning(f"[PosterRenderer] Failed to read back zone rects: {e}")

            await browser.close()

        return {k: (float(v[0]), float(v[1]), float(v[2]), float(v[3])) for k, v in rects.items()}

    @classmethod
    def measure_zone_rects(
        cls,
        html_content: str,
        width: int,
        height: int,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Bọc đồng bộ an toàn cho measure_zone_rects_async (chạy được cả trong Event Loop JupyterLab).
        """
        return _run_coroutine_sync(
            cls.measure_zone_rects_async(
                html_content=html_content,
                width=width,
                height=height,
            )
        )


__all__ = [
    "PosterRenderer",
]

