"""
src/tendoo_v2/pipeline.py

End-to-end glue for the "wire tạm v2 vào production, dùng LLM đã host" round
(2026-09-14): form fields + free prompt -> (deterministic store-info block) +
(LLM-curated content blocks, llm_client.py) -> solver.solve() -> renderer.
render_blocks_to_html() -> (optional) PosterRenderer screenshot.

Store info (STORE_INFO_KEYS in llm_client.py) is deliberately built HERE, in code,
never sent to the LLM -- the one authority carve-out business stated explicitly
("trừ phần thông tin cửa hàng"). Everything else goes through the LLM's curation.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from tendoo_v2.llm_client import STORE_INFO_KEYS, call_llm_for_blocks
from tendoo_v2.renderer import render_blocks_to_html
from tendoo_v2.schema import Block


def _build_store_info_block(fields: Dict[str, str]) -> Optional[Block]:
    """Store/contact fields always render verbatim, unconditionally, pinned to a
    fixed spot -- never the LLM's decision (see llm_client.py's STORE_INFO_KEYS
    docstring). Joined into ONE meta block; empty/missing keys are just skipped."""
    parts = [fields[k].strip() for k in ("store_name", "phone", "address") if fields.get(k, "").strip()]
    if not parts:
        return None
    return Block(text=" • ".join(parts), role="meta", field="store_info", zone="bottom_center")


def synthetic_background_data_uri(width: int, height: int) -> str:
    """Plain warm-dark gradient placeholder -- this round wires the LLM stage, NOT
    the real diffusion background (that lives in src/tendoo's own pipeline). Pass a
    real `bg_data_uri` to run_pipeline() once tested against actual diffusion output."""
    top = np.array([58, 38, 26], dtype=np.float32)
    bottom = np.array([18, 12, 9], dtype=np.float32)
    t = np.linspace(0, 1, height).reshape(height, 1, 1)
    arr = top * (1 - t) + bottom * t
    arr = np.repeat(arr, width, axis=1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def run_pipeline(
    fields: Dict[str, str],
    free_prompt: str,
    width: int = 1024,
    height: int = 1024,
    bg_data_uri: Optional[str] = None,
    font_key: str = "bevietnam",
    model: Optional[str] = None,
    output_image_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs the full form-fields+prompt -> real hosted LLM -> solver -> HTML pipeline.
    Never raises on an LLM failure (network/timeout/bad JSON) -- falls back to a
    store-info-only poster and reports the error, matching demo_server.py's "an LLM
    sidecar hiccup never breaks the request" philosophy. Set `output_image_path` to
    also render a PNG via the real Chromium engine (tendoo.poster_renderer.PosterRenderer)."""
    llm_blocks, llm_errors = call_llm_for_blocks(fields, free_prompt, model=model)

    store_block = _build_store_info_block(fields)
    all_blocks: List[Block] = ([store_block] if store_block else []) + llm_blocks

    bg_uri = bg_data_uri or synthetic_background_data_uri(width, height)
    html = render_blocks_to_html(all_blocks, width=width, height=height, bg_data_uri=bg_uri, font_key=font_key)

    result: Dict[str, Any] = {
        "html": html,
        "blocks": all_blocks,
        "llm_block_count": len(llm_blocks),
        "llm_errors": llm_errors,
    }

    if output_image_path is not None:
        from tendoo.poster_renderer import PosterRenderer
        PosterRenderer.render(html_content=html, output_image_path=output_image_path, width=width, height=height)
        result["output_image_path"] = str(output_image_path)

    return result


__all__ = ["run_pipeline", "synthetic_background_data_uri", "STORE_INFO_KEYS"]
