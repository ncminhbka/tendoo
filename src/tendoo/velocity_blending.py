"""
src/tendoo/velocity_blending.py

Single-Pass Regional Velocity Blending: the production DiT sampling algorithm behind
the live demo's "reserve a text-safe corridor without covering the product" behavior
(as opposed to the earlier, abandoned approach of generating freely and then running
object detection to find empty space after the fact -- see project memory).

Co-evolves two prompts (a "scene" framing prompt and a "corridor" negative-space
prompt) from the same noise, blending their predicted velocities per-token according
to a spatial mask at every denoising step:

    v_blend = (1.0 - M) * v_scene + M * v_corridor

Moved here (2026-09) from `scripts/pipeline_e2e_poster.py` -- that script still owns
its own mask builders / campaign presets / CLI (a standalone test harness), and now
imports these two functions from here instead of defining them, so `tendoo.demo_server`
(the live server) no longer has to import from a `scripts/` file to run real inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from flux2.sampling import prc_img

__all__ = ["denoise_regional_velocity_blended", "load_and_encode_ref_image"]


def denoise_regional_velocity_blended(
    model: Any,
    img: torch.Tensor,             # (1, L_total, C) initial noise + optional ref tokens
    img_ids: torch.Tensor,         # (1, L_total, 4) position coordinate ids
    txt_scene: torch.Tensor,       # scene prompt embeddings
    txt_scene_ids: torch.Tensor,
    txt_corridor: torch.Tensor,    # corridor prompt embeddings
    txt_corridor_ids: torch.Tensor,
    spatial_mask: torch.Tensor,    # (1, L_canvas, 1) float tensor in [0, 1]
    timesteps: List[float],        # ODE schedule (e.g. 8 steps)
    guidance: float = 1.5,
    num_canvas_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Co-evolves scene framing and physical negative space simultaneously from noise:
      v_blend = (1.0 - M) * v_scene + M * v_corridor
    Supports optional In-Context Reference Product Conditioning (RoPE t=10.0).

    PERF: the scene and corridor branches share the exact same `x`/`x_ids`/`timesteps`/
    `guidance` at every step -- only `ctx` differs -- so both are run as ONE batch-2
    forward pass (`torch.cat([...], dim=0)` + `pred.chunk(2)`) instead of two sequential
    single-batch calls. This mirrors the existing classifier-free-guidance pattern in
    `flux2.sampling.denoise_cfg` and is mathematically identical to the previous
    two-call version -- same two forward passes' worth of compute, just issued as one
    kernel-launch-amortized batched call instead of two sequential ones. Requires
    `txt_scene`/`txt_corridor` to already share the same sequence length (true here:
    `Qwen3TextEncoder.forward` always pads to a fixed `MAX_LENGTH`, so this never needs
    to pad/truncate the two prompts to match each other).

    NOTE: a second optimization (reusing `model.forward_kv_extract`/`forward_kv_cached`
    to skip recomputing the reference-image tokens' K/V at every step) was investigated
    and deliberately NOT applied here -- see `scripts/verify_ref_kv_cache_hypothesis.py`
    and the project memory. That path uses a *different* reference-token modulation
    scheme (a fixed `ref_fixed_timestep`, default 0.0) than the uniform per-step `t_vec`
    modulation this function currently applies to ref tokens -- i.e. it is not a
    value-preserving cache, it is a different sampling formulation. A real-GPU
    comparison found the two paths diverge substantially by the final step (cosine
    similarity ~0.93, max pixel diff ~217/255 on one test case) -- confirmed, not
    hypothetical -- so it was not swapped in without further validation of whether
    reference-product fidelity is actually preserved either way.
    """
    orig_dtype = img.dtype
    device = img.device
    mask = spatial_mask.to(device=device, dtype=orig_dtype)
    L_canvas = num_canvas_tokens if num_canvas_tokens is not None else mask.shape[1]

    # Batch the two conditioning branches along dim 0 once, up front.
    img_b = torch.cat([img, img], dim=0)
    img_ids_b = torch.cat([img_ids, img_ids], dim=0)
    txt_b = torch.cat([txt_scene, txt_corridor], dim=0)
    txt_ids_b = torch.cat([txt_scene_ids, txt_corridor_ids], dim=0)

    for step_idx in range(len(timesteps) - 1):
        t_curr = timesteps[step_idx]
        t_prev = timesteps[step_idx + 1]

        t_vec = torch.full((img_b.shape[0],), t_curr, dtype=orig_dtype, device=device)
        guidance_vec = torch.full((img_b.shape[0],), guidance, dtype=orig_dtype, device=device)

        # Single batched forward pass computes both the scene and corridor velocity
        # predictions together (batch 0 = scene, batch 1 = corridor).
        pred_b = model(
            x=img_b,
            x_ids=img_ids_b,
            timesteps=t_vec,
            ctx=txt_b,
            ctx_ids=txt_ids_b,
            guidance=guidance_vec,
        )
        pred_scene, pred_corridor = pred_b[0:1], pred_b[1:2]

        # Regional Flow Matching velocity blending applied strictly to Canvas tokens
        v_scene_canvas = pred_scene[:, :L_canvas, :]
        v_corridor_canvas = pred_corridor[:, :L_canvas, :]
        v_blend_canvas = (1.0 - mask) * v_scene_canvas + mask * v_corridor_canvas

        # Euler ODE step on canvas tokens (single copy -- both batch slots must stay
        # identical going into the next step, since they represent the same co-evolving
        # image conditioned on two different prompts, not two independent samples).
        canvas_tokens = img_b[0:1, :L_canvas, :] + (t_prev - t_curr) * v_blend_canvas
        if img_b.shape[1] > L_canvas:
            new_img = torch.cat([canvas_tokens, img_b[0:1, L_canvas:, :]], dim=1).to(orig_dtype)
        else:
            new_img = canvas_tokens.to(orig_dtype)
        img_b = torch.cat([new_img, new_img], dim=0)

    return img_b[0:1, :L_canvas, :]


def load_and_encode_ref_image(
    ref_image_path: Path,
    ae: Any,
    device: str,
    target_dim: int = 512,
    time_offset: float = 10.0,
    ae_device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes user product image into VAE latent with canonical In-Context RoPE offset.

    `device` is where the RETURNED tokens must end up (the DiT's device, to be
    concatenated with the canvas tokens). `ae_device` is where the `ae` module itself
    actually lives -- on a single-GPU (or CPU) setup these are the same device and
    `ae_device` can be omitted, but on a multi-GPU setup (DiT on cuda:0, AE on cuda:1,
    as util.load_ae(..., device=aux_device) sets up) they differ, and encoding must
    run on `ae_device` or torch raises a cross-device RuntimeError from conv2d.
    """
    if ae_device is None:
        ae_device = device

    pil_img = Image.open(ref_image_path).convert("RGB")
    pil_img.thumbnail((target_dim, target_dim), Image.Resampling.LANCZOS)
    w = max(16, (pil_img.width // 16) * 16)
    h = max(16, (pil_img.height // 16) * 16)
    pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)

    arr = (np.array(pil_img).astype(np.float32) / 127.5 - 1.0)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=ae_device, dtype=torch.bfloat16)

    with torch.no_grad():
        z_ref = ae.encode(tensor)

    ref_toks, ref_ids = prc_img(z_ref[0])
    ref_toks = ref_toks.unsqueeze(0).to(device)
    ref_ids = ref_ids.unsqueeze(0).to(device)
    ref_ids[:, :, 0] = time_offset  # Pretrained discrete RoPE offset (t=10.0)
    return ref_toks, ref_ids
