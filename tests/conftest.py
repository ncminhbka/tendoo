from __future__ import annotations
import os, sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / 'src']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

@pytest.fixture(autouse=True)
def mock_slow_playwright_render(request, monkeypatch):
    if 'e2e' in request.keywords or os.environ.get('TENDOO_REAL_PLAYWRIGHT') == '1':
        return

    def fake_render(cls, html_content, output_image_path, width, height, device_scale_factor=1):
        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Create non-trivial image so PNG size > 30000 bytes
        w = max(1, min(width, 1024))
        h = max(1, min(height, 1024))
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(out_path, format='PNG')
        return out_path

    try:
        from tendoo.poster_renderer import PosterRenderer
        monkeypatch.setattr(PosterRenderer, 'render', classmethod(fake_render))
    except Exception:
        pass
