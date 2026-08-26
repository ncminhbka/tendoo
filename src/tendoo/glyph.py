"""
================================================================================
TENDOO AI - GLYPH RENDERING & IN-CONTEXT 4D ROPE ENCODING MODULE
================================================================================
"""

import math
import os
from pathlib import Path
from typing import Union

from einops import rearrange
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from flux2.autoencoder import AutoEncoder
from flux2.sampling import prc_img

# Auto-detect repository root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FONTS_DIR = ROOT_DIR / "fonts"

# Built-in Font Registry with Friendly Aliases
FONT_REGISTRY = {
    # Core Standard Fonts:
    "bevietnam": str(FONTS_DIR / "BeVietnamPro-Black.ttf"),
    "playfair": str(FONTS_DIR / "PlayfairDisplay.ttf"),
    "anton": str(FONTS_DIR / "Anton-Regular.ttf"),
    "pacifico": str(FONTS_DIR / "Pacifico-Regular.ttf"),
    "graffiti": str(FONTS_DIR / "SedgwickAveDisplay-Regular.ttf"),
    "sedgwick": str(FONTS_DIR / "SedgwickAveDisplay-Regular.ttf"),
    "dancing": str(FONTS_DIR / "DancingScript.ttf"),
    "oswald": str(FONTS_DIR / "Oswald.ttf"),
    # Fresh, Summer, Playful SVN Fonts:
    "cookies": str(FONTS_DIR / "SVN-Cookies.ttf"),
    "grocery": str(FONTS_DIR / "SVN-Grocery Rounded.ttf"),
    "gretoon": str(FONTS_DIR / "SVN-Gretoon.ttf"),
    "blowbrush": str(FONTS_DIR / "SVN-Blow Brush.ttf"),
    "brush": str(FONTS_DIR / "SVN-Blow Brush.ttf"),
    "holidays": str(FONTS_DIR / "SVN-Holidays.ttf"),
    "clementine": str(FONTS_DIR / "SVN-Clementine.ttf"),
    "harabaras": str(FONTS_DIR / "SVN-Harabaras.ttf"),
    "lolapeluza": str(FONTS_DIR / "SVN-Lolapeluza Black.ttf"),
    "gotham": str(FONTS_DIR / "SVN-Gotham Ultra.otf"),
}


def resolve_font_path(font_name_or_path: str | None) -> str:
    """
    Resolves font alias (e.g. 'playfair', 'bevietnam') or validates custom file path.
    Falls back to OS system fonts if none are matched.
    """
    if font_name_or_path:
        key = font_name_or_path.lower().strip()
        if key in FONT_REGISTRY and os.path.exists(FONT_REGISTRY[key]):
            return FONT_REGISTRY[key]
        if os.path.exists(font_name_or_path):
            return font_name_or_path

    # Fallback to defaults
    for p in [
        FONT_REGISTRY["bevietnam"],
        FONT_REGISTRY["playfair"],
        FONT_REGISTRY["anton"],
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]:
        if os.path.exists(p):
            return p

    raise RuntimeError("❌ No valid Vietnamese Unicode font found!")


def create_glyph_image(
    text: str,
    target_width: int = 512,
    target_height: int = 224,
    font_path: str | None = None,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    padding_ratio: float = 0.08,
) -> Image.Image:
    """
    Renders tight-crop Vietnamese glyph bitmap with automatic line wrapping
    and binary-search sizing adhering to the Glyph Scaling Law.
    """
    assert target_width > 0 and target_height > 0
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16

    font_path = resolve_font_path(font_path)

    pad_w = int(target_width * padding_ratio)
    pad_h = int(target_height * padding_ratio)
    max_w = target_width - 2 * pad_w
    max_h = target_height - 2 * pad_h

    text = text.replace("\\n", "\n")

    if "\n" in text:
        candidate_layouts = [[line.strip() for line in text.split("\n") if line.strip()]]
    else:
        words = text.split()
        candidate_layouts = []
        if len(words) >= 4:
            mid = len(words) // 2
            candidate_layouts.append([" ".join(words[:mid]), " ".join(words[mid:])])
        if len(words) >= 6:
            p1 = len(words) // 3
            p2 = 2 * len(words) // 3
            candidate_layouts.append([" ".join(words[:p1]), " ".join(words[p1:p2]), " ".join(words[p2:])])
        candidate_layouts.append([text])

    best_font = None
    best_lines = None
    best_size = 0

    for lines in candidate_layouts:
        low, high = 14, 200
        opt_font = None
        opt_size = 0

        while low <= high:
            mid_size = (low + high) // 2
            try:
                test_font = ImageFont.truetype(font_path, size=mid_size)
            except Exception:
                test_font = ImageFont.load_default()

            total_h = 0
            max_line_w = 0

            for line in lines:
                bbox = test_font.getbbox(line)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                max_line_w = max(max_line_w, lw)
                total_h += lh

            line_spacing = int(mid_size * 0.20) * (len(lines) - 1)
            total_h += line_spacing

            if max_line_w <= max_w and total_h <= max_h:
                opt_font = test_font
                opt_size = mid_size
                low = mid_size + 1
            else:
                high = mid_size - 1

        if opt_size > best_size:
            best_size = opt_size
            best_font = opt_font
            best_lines = lines

    if best_font is None:
        best_font = ImageFont.truetype(font_path, size=20)
        best_lines = candidate_layouts[-1]
        best_size = 20

    img = Image.new("RGB", (target_width, target_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    line_heights = []
    line_widths = []
    for line in best_lines:
        bbox = best_font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(best_size * 0.20)
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)
    curr_y = (target_height - total_block_h) // 2

    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        curr_x = (target_width - lw) // 2
        bbox = best_font.getbbox(line)
        draw.text((curr_x - bbox[0], curr_y - bbox[1]), line, fill=text_color, font=best_font)
        curr_y += line_heights[i] + line_spacing

    return img


def encode_glyph_to_incontext_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a tight-crop glyph image into 4D RoPE coordinate tokens.
    Handles device placement across multi-GPU setups automatically.
    """
    ae_device = next(ae.parameters()).device if hasattr(ae, "parameters") else device

    g_arr = np.array(glyph_img).astype(np.float32) / 127.5 - 1.0
    g_tensor = torch.from_numpy(g_arr).permute(2, 0, 1).unsqueeze(0).to(device=ae_device, dtype=torch.bfloat16)

    with torch.no_grad():
        g_latent = ae.encode(g_tensor)

    ref_tokens, _ = prc_img(g_latent[0])
    ref_tokens = ref_tokens.unsqueeze(0).to(device)

    g_h, g_w = g_latent.shape[2], g_latent.shape[3]
    t_coords = torch.full((g_h, g_w), fill_value=float(t_offset), dtype=torch.float32, device=device)
    h_coords = torch.arange(g_h, dtype=torch.float32, device=device).unsqueeze(1).expand(g_h, g_w)
    w_coords = torch.arange(g_w, dtype=torch.float32, device=device).unsqueeze(0).expand(g_h, g_w)
    l_coords = torch.zeros((g_h, g_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0).to(device)

    return ref_tokens, ref_ids


def encode_product_to_incontext_tokens(
    ae: AutoEncoder,
    image_path_or_img: Union[str, Image.Image],
    t_offset: float = 50.0,
    target_size: int = 1024,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a natural product image into 4D RoPE tokens.
    """
    ae_device = next(ae.parameters()).device if hasattr(ae, "parameters") else device

    if isinstance(image_path_or_img, str):
        img = Image.open(image_path_or_img).convert("RGB")
    else:
        img = image_path_or_img.convert("RGB")

    # Resize keeping aspect ratio, clamped to multiple of 16
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    w, h = img.size
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    img = img.resize((w, h), Image.Resampling.LANCZOS)

    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=ae_device, dtype=torch.bfloat16)

    with torch.no_grad():
        latent = ae.encode(tensor)

    prod_tokens, _ = prc_img(latent[0])
    prod_tokens = prod_tokens.unsqueeze(0).to(device)

    p_h, p_w = latent.shape[2], latent.shape[3]
    t_coords = torch.full((p_h, p_w), fill_value=float(t_offset), dtype=torch.float32, device=device)
    h_coords = torch.arange(p_h, dtype=torch.float32, device=device).unsqueeze(1).expand(p_h, p_w)
    w_coords = torch.arange(p_w, dtype=torch.float32, device=device).unsqueeze(0).expand(p_h, p_w)
    l_coords = torch.zeros((p_h, p_w), dtype=torch.float32, device=device)

    prod_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    prod_ids = rearrange(prod_ids, "h w d -> (h w) d").unsqueeze(0).to(device)

    return prod_tokens, prod_ids
