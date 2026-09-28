"""
Các cách tạo VÙNG YÊN cho chữ trên ảnh nền diffusion -- đối chứng với velocity blending 2 luồng (velocity_blending.py).

Bối cảnh (28/09): blend mask cứng ở MỌI bước cắt cụt chủ thể ngay mép mask (poster đồng hồ split_left: cánh tay
mất nửa) và tốn batch 2 mỗi bước. Module này thêm 3 thứ, đều ở tầng ngoài, chỉ gọi `model(...)` như BFL
(không sửa src/flux2/):

1. `composition_hint(mask)`   -- câu bố cục SINH TỪ HÌNH HỌC mask (không để LLM tự viết), nối vào scene_prompt.
                                 Chỉ dùng chữ, không số/tỉ lệ (AGENTS §4.2).
2. `denoise_blend_early(...)` -- trộn 2 luồng CHỈ ở k bước đầu (bố cục lớn hình thành ở bước nhiễu cao), phần còn
                                 lại luồng scene tự chạy -> tự "hàn" đường biên. Chi phí DiT ~ (N + k) / N.
3. `denoise_x0_lowpass(...)`  -- MỘT luồng: mỗi bước suy ra ảnh sạch dự đoán x̂0 = x_t - t·v, trong vùng mask bỏ bớt
                                 tần số cao của x̂0 (giữ màu/ánh sáng), tính lại v. Không có prompt thứ hai tranh chấp
                                 -> không có đường cắt. Chi phí DiT ~ 1×.

Toán (quy ước BFL: t đi 1 -> 0, x_t = (1-t)·x0 + t·ε, bước Euler x += (t_prev - t)·v nên v = ε - x0):
  x̂0 = x_t - t·v,  ε̂ = x_t + (1-t)·v
  x0' = x̂0 - s·M·HP(x̂0)           với HP = x̂0 - LowPass(x̂0) trên lưới token (h_lat × w_lat)
  v'  = ε̂ - x0' = v + s·M·HP(x̂0)   (ε̂ giữ nguyên -> chỉ đổi phần ảnh sạch)

CHƯA ĐO: chưa biết làm mờ trên latent 128 kênh giải mã ra bokeh hay mảng xám bệt -- đó là việc của
scripts/bench_saliency_modes.py trên máy chủ. Không kết luận trước khi có ảnh.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from tendoo_v3.velocity_blending import denoise_regional_velocity_blended, denoise_scene_only


# --------------------------------------------------------------------------------------------------------------
# 1. Câu bố cục sinh từ mask
# --------------------------------------------------------------------------------------------------------------

def composition_hint(mask: np.ndarray) -> str:
    """Câu tiếng Anh mô tả chủ thể nằm ở PHÍA MỞ và vùng chữ để trống, suy từ mask (H, W) ∈ [0, 1].

    Chỉ dùng chữ, không có số, phần trăm, tỉ lệ khung (Qwen3 hiểu nhầm thành chữ cần vẽ -- AGENTS §4.2).
    Mask rỗng -> chuỗi rỗng (không có gì phải chừa)."""
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim != 2 or float(m.mean()) < 0.01:
        return ""
    h, w = m.shape
    rows, cols = m.mean(axis=1), m.mean(axis=0)
    band = max(1, h // 6)
    top, bottom = float(rows[:band].mean()), float(rows[-band:].mean())
    bandw = max(1, w // 6)
    left, right = float(cols[:bandw].mean()), float(cols[-bandw:].mean())
    total = float(m.sum())
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((m * xs).sum() / total) / w - 0.5
    cy = float((m * ys).sum() / total) / h - 0.5

    empty = "softly blurred empty background with no objects"
    # Hai dải trên + dưới (sandwich, l_frame): chủ thể ở dải giữa.
    if top > 0.5 and bottom > 0.5 and left < 0.5 and right < 0.5:
        return (", main subject fully contained in the middle band of the frame, well clear of the top and bottom edges, "
                f"calm {empty} along the top and bottom")
    # Khối ở tâm (thẻ luxury): chủ thể dạt ra quanh mép.
    if abs(cx) < 0.08 and abs(cy) < 0.08:
        return f", elements arranged toward the edges of the frame, calm uncluttered center, {empty} in the middle"

    horiz = "left" if cx < 0 else "right"
    vert = "top" if cy < 0 else "bottom"
    opp = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    if abs(cx) >= 1.3 * abs(cy):
        return (f", main subject placed entirely in the {opp[horiz]} half of the frame and not crossing into the "
                f"{horiz} side, the {horiz} side of the frame is {empty}")
    if abs(cy) >= 1.3 * abs(cx):
        return (f", main subject placed entirely in the {opp[vert]} part of the frame and not reaching the {vert} edge, "
                f"the {vert} part of the frame is {empty}")
    return (f", main subject placed toward the {opp[vert]} {opp[horiz]} area of the frame, away from the {vert} {horiz} "
            f"corner, the {vert} {horiz} corner is {empty}")


# --------------------------------------------------------------------------------------------------------------
# 2. Trộn 2 luồng chỉ ở k bước đầu
# --------------------------------------------------------------------------------------------------------------

def denoise_blend_early(
    model: Any,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt_scene: torch.Tensor,
    txt_scene_ids: torch.Tensor,
    txt_corridor: torch.Tensor,
    txt_corridor_ids: torch.Tensor,
    spatial_mask: torch.Tensor,
    timesteps: List[float],
    blend_steps: int,
    guidance: float = 4.0,
    num_canvas_tokens: Optional[int] = None,
    txt_uncond: Optional[torch.Tensor] = None,
    txt_uncond_ids: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """k = `blend_steps` bước đầu: đúng `denoise_regional_velocity_blended`; các bước sau: đúng `denoise_scene_only`
    trên latent đã trộn. k >= số bước -> y hệt blend toàn phần; k = 0 -> y hệt scene-only (test_saliency_guidance)."""
    L = num_canvas_tokens if num_canvas_tokens is not None else spatial_mask.shape[1]
    n = len(timesteps) - 1
    k = max(0, min(int(blend_steps), n))
    cfg = dict(txt_uncond=txt_uncond, txt_uncond_ids=txt_uncond_ids, cfg_scale=cfg_scale)
    x = img
    if k > 0:
        canvas = denoise_regional_velocity_blended(
            model, x, img_ids, txt_scene, txt_scene_ids, txt_corridor, txt_corridor_ids, spatial_mask,
            timesteps[: k + 1], guidance=guidance, num_canvas_tokens=L, **cfg,
        )
        if k == n:
            return canvas
        # Hàm blend chỉ trả token canvas -> gắn lại token ảnh tham chiếu (không bị Euler cập nhật, như đường cũ).
        x = torch.cat([canvas, img[:, L:, :]], dim=1) if img.shape[1] > L else canvas
    return denoise_scene_only(
        model, x, img_ids, txt_scene, txt_scene_ids, timesteps[k:], guidance=guidance, num_canvas_tokens=L, **cfg,
    )


# --------------------------------------------------------------------------------------------------------------
# 3. Một luồng, bỏ tần số cao của ảnh sạch dự đoán trong vùng mask
# --------------------------------------------------------------------------------------------------------------

def _gaussian_lowpass(tokens: torch.Tensor, h_lat: int, w_lat: int, sigma: float) -> torch.Tensor:
    """Làm mờ Gauss tách 2 chiều trên lưới token. tokens: (1, h_lat*w_lat, C) theo thứ tự (h w) của prc_img."""
    if sigma <= 0:
        return tokens
    c = tokens.shape[-1]
    grid = tokens[0].transpose(0, 1).reshape(1, c, h_lat, w_lat).float()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=tokens.device, dtype=torch.float32)
    k1 = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    # Pad phản chiếu tối đa (kích thước - 1) mỗi chiều -- lưới nhỏ (test) vẫn chạy được.
    ph, pw = min(radius, h_lat - 1), min(radius, w_lat - 1)
    kh = k1[radius - ph: radius + ph + 1] if ph < radius else k1
    kw = k1[radius - pw: radius + pw + 1] if pw < radius else k1
    kh, kw = kh / kh.sum(), kw / kw.sum()
    g = F.pad(grid, (0, 0, ph, ph), mode="reflect") if ph > 0 else grid
    g = F.conv2d(g, kh.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    g = F.pad(g, (pw, pw, 0, 0), mode="reflect") if pw > 0 else g
    g = F.conv2d(g, kw.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return g.reshape(1, c, h_lat * w_lat).transpose(1, 2).to(tokens.dtype)


def denoise_x0_lowpass(
    model: Any,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt_scene: torch.Tensor,
    txt_scene_ids: torch.Tensor,
    spatial_mask: torch.Tensor,
    timesteps: List[float],
    h_lat: int,
    w_lat: int,
    strength: float = 0.85,
    sigma: float = 2.5,
    active_steps: Optional[int] = None,
    guidance: float = 4.0,
    num_canvas_tokens: Optional[int] = None,
    txt_uncond: Optional[torch.Tensor] = None,
    txt_uncond_ids: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """Một luồng scene (batch 1; bản base + CFG: batch 2 như denoise_scene_only). Ở `active_steps` bước đầu (None = mọi
    bước), vận tốc được sửa v' = v + s·M·HP(x̂0) -- xem docstring module. strength=0 -> y hệt denoise_scene_only."""
    orig_dtype = img.dtype
    device = img.device
    L = num_canvas_tokens if num_canvas_tokens is not None else spatial_mask.shape[1]
    mask = spatial_mask.to(device=device, dtype=torch.float32)
    n = len(timesteps) - 1
    active = n if active_steps is None else max(0, min(int(active_steps), n))

    use_cfg = txt_uncond is not None and cfg_scale != 1.0
    nb = 2 if use_cfg else 1
    ctx = torch.cat([txt_scene, txt_uncond], dim=0) if use_cfg else txt_scene
    ctx_ids = torch.cat([txt_scene_ids, txt_uncond_ids], dim=0) if use_cfg else txt_scene_ids

    x = img
    for step_idx in range(n):
        t_curr, t_prev = timesteps[step_idx], timesteps[step_idx + 1]
        pred = model(
            x=torch.cat([x] * nb, dim=0) if use_cfg else x,
            x_ids=torch.cat([img_ids] * nb, dim=0) if use_cfg else img_ids,
            timesteps=torch.full((nb,), t_curr, dtype=orig_dtype, device=device),
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=torch.tensor([guidance] * nb, dtype=orig_dtype, device=device),
        )
        if use_cfg:
            pred = pred[1:2] + cfg_scale * (pred[0:1] - pred[1:2])

        v = pred[:, :L, :].float()
        if step_idx < active and strength > 0:
            x0_hat = x[:, :L, :].float() - t_curr * v
            hp = x0_hat - _gaussian_lowpass(x0_hat, h_lat, w_lat, sigma)
            v = v + strength * mask * hp

        canvas = x[:, :L, :].float() + (t_prev - t_curr) * v
        x = torch.cat([canvas.to(orig_dtype), x[:, L:, :]], dim=1) if x.shape[1] > L else canvas.to(orig_dtype)

    return x[:, :L, :]
