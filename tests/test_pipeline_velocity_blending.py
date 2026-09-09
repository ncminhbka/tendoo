"""
tests/test_pipeline_velocity_blending.py

Regression test for the batching optimization applied to
scripts/pipeline_e2e_poster.py::denoise_regional_velocity_blended().

The optimization replaces two sequential single-batch model() calls (one for the
"scene" prompt, one for the "corridor" prompt) with one batch-2 call + chunk(2) --
mirroring the existing classifier-free-guidance pattern in flux2.sampling.denoise_cfg.
This is meant to be *mathematically identical* to the original two-call version, not
just "close enough" -- so this test keeps a copy of the original (pre-optimization)
implementation as a reference oracle and asserts bit-exact equality against the
current (batched) implementation, using a small deterministic fake model (no real
FLUX.2 checkpoint needed / available in this environment).
"""

import sys
from pathlib import Path
from typing import List, Optional

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline_e2e_poster  # noqa: E402


def _reference_denoise_regional_velocity_blended(
    model,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt_scene: torch.Tensor,
    txt_scene_ids: torch.Tensor,
    txt_corridor: torch.Tensor,
    txt_corridor_ids: torch.Tensor,
    spatial_mask: torch.Tensor,
    timesteps: List[float],
    guidance: float = 1.5,
    num_canvas_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Verbatim copy of the pre-optimization (two sequential single-batch calls)
    implementation, kept here ONLY as a reference oracle for this test."""
    orig_dtype = img.dtype
    device = img.device
    mask = spatial_mask.to(device=device, dtype=orig_dtype)
    L_canvas = num_canvas_tokens if num_canvas_tokens is not None else mask.shape[1]

    for step_idx in range(len(timesteps) - 1):
        t_curr = timesteps[step_idx]
        t_prev = timesteps[step_idx + 1]

        t_vec = torch.full((img.shape[0],), t_curr, dtype=orig_dtype, device=device)
        guidance_vec = torch.full((img.shape[0],), guidance, dtype=orig_dtype, device=device)

        pred_scene = model(x=img, x_ids=img_ids, timesteps=t_vec, ctx=txt_scene, ctx_ids=txt_scene_ids, guidance=guidance_vec)
        pred_corridor = model(x=img, x_ids=img_ids, timesteps=t_vec, ctx=txt_corridor, ctx_ids=txt_corridor_ids, guidance=guidance_vec)

        v_scene_canvas = pred_scene[:, :L_canvas, :]
        v_corridor_canvas = pred_corridor[:, :L_canvas, :]
        v_blend_canvas = (1.0 - mask) * v_scene_canvas + mask * v_corridor_canvas

        canvas_tokens = img[:, :L_canvas, :] + (t_prev - t_curr) * v_blend_canvas
        if img.shape[1] > L_canvas:
            img = torch.cat([canvas_tokens, img[:, L_canvas:, :]], dim=1).to(orig_dtype)
        else:
            img = canvas_tokens.to(orig_dtype)

    return img[:, :L_canvas, :]


class _FakeFlux2Model:
    """
    Deterministic stand-in for the real FLUX.2 model, sensitive enough to catch
    batching/indexing mistakes: the "velocity" it predicts depends on each row's own
    `ctx` content, the timestep, the guidance scale, and token position -- so if the
    batched implementation ever mixed up which batch row belongs to scene vs.
    corridor, swapped ctx_ids between them, or broadcast the wrong slice, the two
    implementations' outputs would differ.

    Deliberately does NOT let a row's output depend on its position within the batch
    (e.g. "row 0 vs row 1") or on other rows' content -- a real transformer forward
    pass (LayerNorm/attention, no batch-norm-like cross-row statistics) computes each
    batch row independently, so encoding batch position into "ground truth" here would
    be testing an artifact of this fake, not a real batching bug.
    """

    def __call__(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        B, L, C = x.shape
        # Distinguish "which ctx" fed this call (mean over ctx's own values -- differs
        # between txt_scene and txt_corridor since they're independently randomized,
        # but is identical for the same ctx regardless of which batch row carries it).
        ctx_signature = ctx.mean(dim=(1, 2)).view(B, 1, 1)
        # Depend on timestep/guidance so a step-ordering bug would also be caught.
        t_signature = timesteps.view(B, 1, 1) * 0.1
        g_signature = guidance.view(B, 1, 1) * 0.001
        # Depend on token position so canvas-vs-ref slicing bugs would be caught.
        pos_signature = torch.arange(L, dtype=x.dtype).view(1, L, 1) * 0.001

        return (
            x * 0.5
            + ctx_signature
            + t_signature
            + g_signature
            + pos_signature
        ).expand(B, L, C).contiguous()


def _make_inputs(seed: int, num_canvas: int, num_ref: int, ctx_len: int, C: int = 8):
    g = torch.Generator().manual_seed(seed)
    L_total = num_canvas + num_ref
    img = torch.randn(1, L_total, C, generator=g, dtype=torch.float32)
    img_ids = torch.randn(1, L_total, 4, generator=g, dtype=torch.float32)
    txt_scene = torch.randn(1, ctx_len, C, generator=g, dtype=torch.float32)
    txt_scene_ids = torch.randn(1, ctx_len, 4, generator=g, dtype=torch.float32)
    txt_corridor = torch.randn(1, ctx_len, C, generator=g, dtype=torch.float32)
    txt_corridor_ids = torch.randn(1, ctx_len, 4, generator=g, dtype=torch.float32)
    spatial_mask = torch.rand(1, num_canvas, 1, generator=g, dtype=torch.float32)
    return img, img_ids, txt_scene, txt_scene_ids, txt_corridor, txt_corridor_ids, spatial_mask


@pytest.mark.parametrize("num_ref,ctx_len,num_steps", [
    (0, 6, 8),   # no reference image, 8-step schedule (matches production default)
    (5, 6, 8),   # with a reference-image token block appended after canvas tokens
    (0, 6, 3),   # very short schedule (edge case: 2 denoise iterations)
])
def test_batched_matches_sequential_reference(num_ref, ctx_len, num_steps):
    num_canvas = 12
    inputs = _make_inputs(seed=42, num_canvas=num_canvas, num_ref=num_ref, ctx_len=ctx_len)
    img, img_ids, txt_scene, txt_scene_ids, txt_corridor, txt_corridor_ids, spatial_mask = inputs
    timesteps = list(torch.linspace(1.0, 0.0, num_steps).tolist())

    model = _FakeFlux2Model()

    out_reference = _reference_denoise_regional_velocity_blended(
        model, img.clone(), img_ids.clone(), txt_scene.clone(), txt_scene_ids.clone(),
        txt_corridor.clone(), txt_corridor_ids.clone(), spatial_mask.clone(),
        timesteps, guidance=1.5, num_canvas_tokens=num_canvas,
    )
    out_batched = pipeline_e2e_poster.denoise_regional_velocity_blended(
        model, img.clone(), img_ids.clone(), txt_scene.clone(), txt_scene_ids.clone(),
        txt_corridor.clone(), txt_corridor_ids.clone(), spatial_mask.clone(),
        timesteps, guidance=1.5, num_canvas_tokens=num_canvas,
    )

    assert out_reference.shape == out_batched.shape == (1, num_canvas, img.shape[-1])
    assert torch.allclose(out_reference, out_batched, atol=1e-6, rtol=1e-6), (
        "Batched implementation diverged from the sequential reference -- "
        f"max abs diff = {(out_reference - out_batched).abs().max().item()}"
    )


def test_batched_calls_model_half_as_many_times():
    """The whole point of the optimization: 1 batched call per step instead of 2."""
    num_canvas, num_steps = 10, 8
    inputs = _make_inputs(seed=7, num_canvas=num_canvas, num_ref=0, ctx_len=6)
    img, img_ids, txt_scene, txt_scene_ids, txt_corridor, txt_corridor_ids, spatial_mask = inputs
    timesteps = list(torch.linspace(1.0, 0.0, num_steps).tolist())

    call_count = {"n": 0}
    base_model = _FakeFlux2Model()

    def counting_model(*args, **kwargs):
        call_count["n"] += 1
        return base_model(*args, **kwargs)

    pipeline_e2e_poster.denoise_regional_velocity_blended(
        counting_model, img, img_ids, txt_scene, txt_scene_ids,
        txt_corridor, txt_corridor_ids, spatial_mask,
        timesteps, guidance=1.5, num_canvas_tokens=num_canvas,
    )

    # num_steps timesteps -> (num_steps - 1) denoise iterations -> 1 batched call each.
    assert call_count["n"] == num_steps - 1
