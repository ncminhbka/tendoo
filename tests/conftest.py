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
    """Fakes the FINAL image render (PosterRenderer.render) with a cheap random-noise
    PNG instead of a real Playwright/Chromium screenshot. Opt out with @pytest.mark.e2e
    or TENDOO_REAL_PLAYWRIGHT=1.

    (2026-09-13: this does NOT cover PosterRenderer.measure_zone_rects -- the SEPARATE
    real-Chromium call OmniBlockLayout.generate_mask_from_render() makes for every
    pipeline run to measure text geometry. See mock_slow_zone_measurement below, added
    the same day after confirming empirically (a direct timed call, ~1.34s each) that
    THIS was the actual per-test Chromium-launch cost this fixture never covered.)
    """
    if 'e2e' in request.keywords or os.environ.get('TENDOO_REAL_PLAYWRIGHT') == '1':
        return

    def fake_render(cls, html_content, output_image_path, width, height, device_scale_factor=1, overflow_report=None):
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
        from tendoo_core.poster_renderer import PosterRenderer
        monkeypatch.setattr(PosterRenderer, 'render', classmethod(fake_render))
    except Exception:
        pass


@pytest.fixture(autouse=True)
def mock_slow_zone_measurement(request, monkeypatch):
    """Fakes PosterRenderer.measure_zone_rects (used by
    OmniBlockLayout.generate_mask_from_render as its "preferred path" corridor-mask
    measurement) to return {} instead of launching a real Chromium browser.
    generate_mask_from_render() already treats {} as "measurement unavailable" and falls
    back to the PIL-estimated generate_mask() -- a real, already-covered, GPU-free code
    path -- so this is a safe, representative fake, not a bypass of real logic.

    Root cause this fixes: every real pipeline call (run_pipeline_inference /
    POST /api/generate, ~30 tests in test_demo_server.py + test_omni_end_to_end.py)
    launched a full real Chromium instance for this measurement -- confirmed directly
    (not just inferred) by timing a standalone call: ~1.34s/launch, unconditionally, even
    in IS_MOCK_MODE (which only ever disabled the diffusion model, never Chromium).
    Opt out the same way as mock_slow_playwright_render (@pytest.mark.e2e or
    TENDOO_REAL_PLAYWRIGHT=1) for a test that genuinely needs the real measured mask.
    """
    if 'e2e' in request.keywords or os.environ.get('TENDOO_REAL_PLAYWRIGHT') == '1':
        return

    def fake_measure(cls, html_content, width, height):
        return {}

    try:
        from tendoo_core.poster_renderer import PosterRenderer
        monkeypatch.setattr(PosterRenderer, 'measure_zone_rects', classmethod(fake_measure))
    except Exception:
        pass


@pytest.fixture(autouse=True)
def redirect_demo_server_output_dir(request, monkeypatch, tmp_path):
    """Redirects demo_server.OUTPUT_DIR to a per-test tmp_path so pipeline tests never
    write real files under the project's real output_demo_server/ directory. Only
    touches `tendoo.demo_server` when it's already imported (most test files never
    import it at all) -- avoids paying its import cost for every test in the suite just
    for this.

    Root cause this fixes: OUTPUT_DIR = PROJECT_ROOT / "output_demo_server" was a
    hardcoded real path, never overridden anywhere in the suite -- confirmed
    empirically: a full test run left ~29-30 fresh real run_<timestamp>/ directories on
    disk every time (bounded only by production code's own _prune_old_output_runs(),
    which also does a real synchronous shutil.rmtree() inside the timed test body once
    the 100-folder cap is hit).
    """
    if 'e2e' in request.keywords or os.environ.get('TENDOO_REAL_PLAYWRIGHT') == '1':
        return
    demo_server = sys.modules.get('tendoo.demo_server')
    if demo_server is not None:
        out_dir = tmp_path / 'output_demo_server'
        out_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(demo_server, 'OUTPUT_DIR', out_dir)


@pytest.fixture(autouse=True)
def redirect_render_plan_log_path(request, monkeypatch, tmp_path):
    """Redirects llm_render_plan_server.RENDER_PLAN_LOG_PATH to a per-test tmp_path so
    the request/response logging added 2026-09-14 (see that module's
    _log_render_plan_io()) never writes real lines into the project's real
    logs/render_plan_requests.jsonl during a pytest run -- same rationale as
    redirect_demo_server_output_dir above, only touches the module if it's already
    imported."""
    if 'e2e' in request.keywords or os.environ.get('TENDOO_REAL_PLAYWRIGHT') == '1':
        return
    srv = sys.modules.get('tendoo.llm_render_plan_server')
    if srv is not None:
        monkeypatch.setattr(srv, 'RENDER_PLAN_LOG_PATH', tmp_path / 'render_plan_requests.jsonl')
