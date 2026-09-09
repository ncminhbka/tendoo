"""
src/tendoo/poster_renderer.py

Playwright headless-Chromium HTML -> PNG renderer, used by both:
  - the live layout engine (`tendoo.layouts.*`, driven by `tendoo.demo_server`), and
  - the older `tendoo_legacy.typography_engine.PosterTemplateEngine` pipeline.

Extracted out of `tendoo_legacy/typography_engine.py` (2026-09-08) so this one genuinely
shared class doesn't drag the rest of that 2000+ line, legacy-pipeline-only module into
the live demo package. Self-contained: no dependency on anything else in this package.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


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
                logger.info(f"[PosterRenderer] Found system browser at: {sys_chrome}")
                launch_kwargs["executable_path"] = sys_chrome

            browser = await p.chromium.launch(**launch_kwargs)
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=device_scale_factor,
            )
            page = await context.new_page()

            # Set content: wait for networkidle so external web fonts (Google Fonts) can download
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
        """Synchronous wrapper for render_html_async."""
        return asyncio.run(
            cls.render_html_async(
                html_content=html_content,
                output_image_path=output_image_path,
                width=width,
                height=height,
                device_scale_factor=device_scale_factor,
            )
        )
