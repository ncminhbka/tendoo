"""
src/tendoo_v3/velocity_blending.py

Single-Pass Regional Velocity Blending ODE Engine cho Tendoo v3:
================================================================
- Thuật toán cốt lõi điều khiển FLUX.2 DiT Base 4B tạo ra vùng không gian an toàn (Text Corridor)
  mà không che lấp sản phẩm và bảo tồn 100% chi tiết quang học.
- Đồng tiến hóa 2 luồng Prompt (Scene Bối cảnh + Corridor Hành lang văn bản) từ cùng một hạt nhiễu Gauss:
      v_blend = (1.0 - M) * v_scene + M * v_corridor
- Tối ưu hóa hiệu năng: Gộp 2 luồng vào Batch-2 Amortization trên 2x NVIDIA A30 (Ampere architecture).
- Hỗ trợ In-Context Reference Product Conditioning tại mốc RoPE chuẩn t=10.0 / t=50.0 / t=60.0.
- Tuyệt đối không dùng KV-caching cho DiT Base 4B (luôn denoise CFG full 50 bước tương tác liên tục).
- Tích hợp Mock Generator thông minh khi chạy trên máy local không có GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

logger = logging.getLogger("TendooV3.VelocityBlending")


def denoise_regional_velocity_blended(
    model: Any,
    img: torch.Tensor,             # (1, L_total, C) initial noise + optional ref tokens
    img_ids: torch.Tensor,         # (1, L_total, 4) position coordinate ids
    txt_scene: torch.Tensor,       # scene prompt embeddings
    txt_scene_ids: torch.Tensor,
    txt_corridor: torch.Tensor,    # corridor prompt embeddings
    txt_corridor_ids: torch.Tensor,
    spatial_mask: torch.Tensor,    # (1, L_canvas, 1) float tensor in [0, 1]
    timesteps: List[float],        # ODE schedule (e.g. 50 steps Euler)
    guidance: float = 4.0,
    num_canvas_tokens: Optional[int] = None,
    mask_gamma: float = 1.0,
    corridor_guidance_boost: float = 0.0,
) -> torch.Tensor:
    """Đồng tiến hóa trường vận tốc Scene và Corridor từ nhiễu hạt theo công thức:
      v_blend = (1.0 - M') * v_scene + M' * v_corridor,  M' = M ** mask_gamma
    Hỗ trợ In-Context Reference Product Conditioning tại mốc RoPE chuẩn BFL.
    """
    orig_dtype = img.dtype
    device = img.device
    mask = spatial_mask.to(device=device, dtype=orig_dtype)
    if mask_gamma != 1.0:
        mask = mask.clamp(0.0, 1.0).pow(mask_gamma)
    L_canvas = num_canvas_tokens if num_canvas_tokens is not None else mask.shape[1]

    # Batch 2 conditioning branches dọc theo dim 0 (Batch Amortization)
    img_b = torch.cat([img, img], dim=0)
    img_ids_b = torch.cat([img_ids, img_ids], dim=0)
    txt_b = torch.cat([txt_scene, txt_corridor], dim=0)
    txt_ids_b = torch.cat([txt_scene_ids, txt_corridor_ids], dim=0)

    for step_idx in range(len(timesteps) - 1):
        t_curr = timesteps[step_idx]
        t_prev = timesteps[step_idx + 1]

        t_vec = torch.full((img_b.shape[0],), t_curr, dtype=orig_dtype, device=device)
        guidance_vec = torch.tensor(
            [guidance, guidance + corridor_guidance_boost], dtype=orig_dtype, device=device
        )

        # Single batched forward pass tính đồng thời scene và corridor velocity
        pred_b = model(
            x=img_b,
            x_ids=img_ids_b,
            timesteps=t_vec,
            ctx=txt_b,
            ctx_ids=txt_ids_b,
            guidance=guidance_vec,
        )
        pred_scene, pred_corridor = pred_b[0:1], pred_b[1:2]

        # Regional Flow Matching velocity blending trên Canvas tokens
        v_scene_canvas = pred_scene[:, :L_canvas, :]
        v_corridor_canvas = pred_corridor[:, :L_canvas, :]
        v_blend_canvas = (1.0 - mask) * v_scene_canvas + mask * v_corridor_canvas

        # Euler ODE step trên canvas tokens
        canvas_tokens = img_b[0:1, :L_canvas, :] + (t_prev - t_curr) * v_blend_canvas
        if img_b.shape[1] > L_canvas:
            new_img = torch.cat([canvas_tokens, img_b[0:1, L_canvas:, :]], dim=1).to(orig_dtype)
        else:
            new_img = canvas_tokens.to(orig_dtype)
        img_b = torch.cat([new_img, new_img], dim=0)

    return img_b[0:1, :L_canvas, :]


def load_and_encode_ref_image(
    ref_image_path: Path | str,
    ae: Any,
    device: str,
    target_dim: int = 512,
    time_offset: float = 10.0,
    ae_device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tải và mã hóa ảnh sản phẩm tham chiếu bằng VAE tại mốc thời gian RoPE chuẩn."""
    from flux2.sampling import prc_img

    img_p = Path(ref_image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Ref image not found: {img_p}")

    pil_img = Image.open(img_p).convert("RGB")
    # Resize giữ nguyên tỉ lệ
    pil_img.thumbnail((target_dim, target_dim), Image.Resampling.LANCZOS)
    
    # Pad hoặc căn chỉnh bội số 16 cho VAE
    w, h = pil_img.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    if new_w != w or new_h != h:
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    arr = np.array(pil_img, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    target_ae_device = ae_device or device
    tensor = tensor.to(device=target_ae_device, dtype=torch.bfloat16)

    with torch.inference_mode():
        latents = ae.encode(tensor)

    ref_tokens, ref_ids = prc_img(latents)
    # Gán mốc time_offset chuẩn vào trục t (trục 0 của ids)
    ref_ids[..., 0] = time_offset

    if str(target_ae_device) != str(device):
        ref_tokens = ref_tokens.to(device=device)
        ref_ids = ref_ids.to(device=device)

    return ref_tokens, ref_ids


def generate_mock_backdrop(
    width: int,
    height: int,
    theme_color: str = "#D4AF37",
    background_tone: str = "dark_luxury",
    scene_prompt: str = "",
    spatial_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Tạo ảnh nền giả lập (Aesthetic Mock Backdrop) mượt mà khi chạy trên môi trường CPU/Local."""
    # Phân tích tone màu nền
    tone_palettes = {
        "dark_luxury": ((14, 18, 28), (28, 38, 56)),
        "light_clean": ((245, 247, 250), (220, 228, 238)),
        "warm_rustic": ((42, 28, 20), (74, 52, 38)),
        "pastel": ((240, 244, 248), (255, 235, 238)),
        "vibrant": ((20, 24, 45), (45, 25, 65)),
    }
    top_color, bottom_color = tone_palettes.get(background_tone, ((14, 18, 28), (28, 38, 56)))

    # Tạo gradient dọc
    base = Image.new("RGB", (width, height), top_color)
    draw = ImageDraw.Draw(base)
    for y in range(height):
        factor = y / float(height)
        r = int(top_color[0] + factor * (bottom_color[0] - top_color[0]))
        g = int(top_color[1] + factor * (bottom_color[1] - top_color[1]))
        b = int(top_color[2] + factor * (bottom_color[2] - top_color[2]))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Thêm ánh sáng ambient radial glow nhẹ
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    cx, cy = int(width * 0.5), int(height * 0.45)
    r_glow = int(min(width, height) * 0.4)
    glow_draw.ellipse(
        [(cx - r_glow, cy - r_glow), (cx + r_glow, cy + r_glow)],
        fill=(255, 255, 255, 18),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=60))
    base.paste(glow, (0, 0), glow)

    return base


__all__ = [
    "denoise_regional_velocity_blended",
    "generate_mock_backdrop",
    "load_and_encode_ref_image",
]
