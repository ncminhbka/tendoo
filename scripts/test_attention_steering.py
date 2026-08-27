"""
================================================================================
TENDOO AI - TRAINING-FREE ATTENTION STEERING & REGIONAL ROUTING BENCHMARK
Model: FLUX.2 klein 4B Base + Qwen3-4B-FP8 + AE
Platform: 2x NVIDIA A30 (48GB VRAM) / Single GPU Compatible

Scientific Objective:
Examine whether In-Memory Attention Logit Boosting and Regional Soft Sigmoid
Masking (with small negative bias for style context retention) can eliminate
Softmax Attention Bleeding / Competition between Slot 10 (Headline) and
Slot 20 (Subtitle) WITHOUT REQUIRING LORA FINE-TUNING.

Key Architectural Guarantees:
1. Upstream Core Frozen: Zero edits to src/flux2/ files on disk.
2. In-Memory Runtime Context: Overrides causal_attn_fn in RAM during execution
   and restores original BFL code cleanly upon completion.
3. Soft Sigmoid Boundary: Smooth spatial transition prevents harsh seams.
4. Controllable Context Bias: Uses beta_suppress = -3.0 (instead of -inf)
   to prevent slot competition while preserving global stylistic harmony.
5. Timestep-Aware Scheduling: Active primarily during geometry formation steps.
================================================================================
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path
from dataclasses import dataclass, field

# Auto-configure PYTHONPATH to include src/
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Auto-configure Offline Mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import native FLUX.2 modules
import flux2.model as flux_model
from flux2.autoencoder import AutoEncoder
from flux2.model import Flux2
from flux2.sampling import (
    batched_prc_txt,
    get_schedule,
    prc_img,
)
from flux2.text_encoder import load_qwen3_embedder
from flux2.util import (
    load_ae,
    load_flow_model,
)

# Built-in Font Registry
FONT_REGISTRY = {
    "bevietnam": str(ROOT_DIR / "fonts" / "BeVietnamPro-Black.ttf"),
    "playfair": str(ROOT_DIR / "fonts" / "PlayfairDisplay.ttf"),
    "anton": str(ROOT_DIR / "fonts" / "Anton-Regular.ttf"),
    "pacifico": str(ROOT_DIR / "fonts" / "Pacifico-Regular.ttf"),
    "sedgwick": str(ROOT_DIR / "fonts" / "SedgwickAveDisplay-Regular.ttf"),
    "dancing": str(ROOT_DIR / "fonts" / "DancingScript.ttf"),
    "oswald": str(ROOT_DIR / "fonts" / "Oswald.ttf"),
    "cookies": str(ROOT_DIR / "fonts" / "SVN-Cookies.ttf"),
    "grocery": str(ROOT_DIR / "fonts" / "SVN-Grocery Rounded.ttf"),
    "gotham": str(ROOT_DIR / "fonts" / "SVN-Gotham Ultra.otf"),
}


def resolve_font_path(font_name_or_path: str | None) -> str:
    """Resolves font alias or validates file path."""
    if font_name_or_path:
        key = font_name_or_path.lower().strip()
        if key in FONT_REGISTRY and os.path.exists(FONT_REGISTRY[key]):
            return FONT_REGISTRY[key]
        if os.path.exists(font_name_or_path):
            return font_name_or_path

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
    tight_crop: bool = True,
    force_single_line: bool = False,
) -> Image.Image:
    """
    Renders TRUE TIGHT-CROP Vietnamese glyph bitmap with automatic line wrapping,
    binary-search font sizing, and exact bounding-box cropping snapped to multiples of 16.
    """
    assert target_width > 0 and target_height > 0
    envelope_w = (target_width // 16) * 16
    envelope_h = (target_height // 16) * 16

    font_path = resolve_font_path(font_path)

    pad_w = int(envelope_w * padding_ratio)
    pad_h = int(envelope_h * padding_ratio)
    max_w = envelope_w - 2 * pad_w
    max_h = envelope_h - 2 * pad_h

    text = text.replace("\\n", "\n")

    if force_single_line:
        candidate_layouts = [[text.replace("\n", " ").strip()]]
    elif "\n" in text:
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

    spacing_ratio = 0.32 if len(candidate_layouts[0]) >= 2 else 0.20

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

            curr_spacing = int(mid_size * spacing_ratio) * (len(lines) - 1)
            total_h += curr_spacing

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

    line_heights = []
    line_widths = []
    for line in best_lines:
        bbox = best_font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = int(best_size * spacing_ratio)
    total_block_h = sum(line_heights) + line_spacing * (len(best_lines) - 1)
    total_block_w = max(line_widths)

    img = Image.new("RGB", (envelope_w, envelope_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    curr_y = (envelope_h - total_block_h) // 2
    for i, line in enumerate(best_lines):
        lw = line_widths[i]
        lh = line_heights[i]
        bbox = best_font.getbbox(line)
        curr_x = (envelope_w - lw) // 2 - bbox[0]
        draw.text((curr_x, curr_y - bbox[1]), line, font=best_font, fill=text_color)
        curr_y += lh + line_spacing

    if not tight_crop:
        return img

    # Exact bounding box crop snapped to 16
    gray = img.convert("L")
    bbox = gray.getbbox()
    if bbox is None:
        return img

    bx0, by0, bx1, by1 = bbox
    pad_box = int(best_size * 0.25)
    bx0 = max(0, bx0 - pad_box)
    by0 = max(0, by0 - pad_box)
    bx1 = min(envelope_w, bx1 + pad_box)
    by1 = min(envelope_h, by1 + pad_box)

    bw = bx1 - bx0
    bh = by1 - by0

    target_bw = max(16, int(np.ceil(bw / 16.0) * 16))
    target_bh = max(16, int(np.ceil(bh / 16.0) * 16))

    crop_img = Image.new("RGB", (target_bw, target_bh), color=bg_color)
    draw_crop = ImageDraw.Draw(crop_img)

    paste_x = (target_bw - bw) // 2
    paste_y = (target_bh - bh) // 2

    crop_img.paste(img.crop((bx0, by0, bx1, by1)), (paste_x, paste_y))
    return crop_img


def encode_glyph_to_incontext_tokens(
    ae: AutoEncoder,
    glyph_img: Image.Image,
    t_offset: float = 10.0,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes a tight-crop glyph image into 4D RoPE tokens."""
    np_arr = np.array(glyph_img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        latent = ae.encode(tensor)

    ref_tokens, _ = prc_img(latent[0])
    ref_tokens = ref_tokens.unsqueeze(0)

    g_h, g_w = latent.shape[2], latent.shape[3]
    t_coords = torch.full((g_h, g_w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(g_h, dtype=torch.float32, device=device).unsqueeze(1).expand(g_h, g_w)
    w_coords = torch.arange(g_w, dtype=torch.float32, device=device).unsqueeze(0).expand(g_h, g_w)
    l_coords = torch.zeros((g_h, g_w), dtype=torch.float32, device=device)

    ref_ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ref_ids = rearrange(ref_ids, "h w d -> (h w) d").unsqueeze(0)

    return ref_tokens, ref_ids


# ==============================================================================
# IN-MEMORY ATTENTION STEERING ENGINE (ZERO UPSTREAM FILE CHANGES)
# ==============================================================================

@dataclass
class AttentionSteeringConfig:
    """Configuration state for runtime attention steering."""
    mode: str = "none"  # "none", "null_test", "boost_only", "soft_regional", "scheduled_soft_regional"
    active: bool = False
    current_timestep: float = 1.0

    # Layout dimensions
    num_txt: int = 512
    num_canvas: int = 2304
    num_ref10: int = 300
    num_ref20: int = 300
    lat_h: int = 64
    lat_w: int = 36

    # Steering hyperparameters
    boost_val_10: float = 0.0
    boost_val_20: float = 2.0
    suppress_val: float = -3.0       # Small negative bias to retain style context while preventing collision
    split_y: float = 0.48            # Normalized vertical boundary between Title and Subtitle
    sharpness: float = 14.0          # Sigmoid sharpness factor for smooth transition
    t_start: float = 0.85            # Timestep threshold to start active steering
    t_end: float = 0.20              # Timestep threshold to ramp down steering for pure optical shading

    # Quantitative Attention Visualization & Layer-Averaging
    dump_attn: bool = False
    recorded_attn_stats: dict | None = None
    _accum_map10: list = field(default_factory=list)
    _accum_map20: list = field(default_factory=list)
    _accum_logit10: list = field(default_factory=list)
    _accum_logit20: list = field(default_factory=list)

    def finalize_stats(self) -> dict | None:
        """Aggregates and averages sampled attention metrics across all captured DiT layers."""
        if len(self._accum_logit10) == 0:
            return None
        mean_10 = float(np.mean(self._accum_logit10))
        mean_20 = float(np.mean(self._accum_logit20))
        map_10 = np.mean(self._accum_map10, axis=0)
        map_20 = np.mean(self._accum_map20, axis=0)
        self.recorded_attn_stats = {
            "map_10": map_10,
            "map_20": map_20,
            "mean_logit_10": mean_10,
            "mean_logit_20": mean_20,
            "logit_gap": mean_10 - mean_20,
            "num_layers_sampled": len(self._accum_logit10),
        }
        return self.recorded_attn_stats


class AttentionSteeringManager:
    """Manages in-memory patching of BFL causal_attn_fn without touching source files."""

    def __init__(self, config: AttentionSteeringConfig):
        self.config = config
        self.orig_causal_attn_fn = None

    def __enter__(self):
        self.orig_causal_attn_fn = flux_model.causal_attn_fn
        cfg = self.config

        def steered_causal_attn_fn(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            num_txt_tokens: int,
            num_ref_tokens: int,
            kv_cache: dict | None = None,
        ) -> torch.Tensor:
            if kv_cache is not None:
                return self.orig_causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)

            seq_len = q.shape[2]
            n_txt = cfg.num_txt
            n_canvas = cfg.num_canvas
            n_ref10 = cfg.num_ref10
            n_ref20 = cfg.num_ref20
            n_ref_total = n_ref10 + n_ref20

            # Debug assertion to strictly guarantee token sequence layout: [TXT][CANVAS][REF10][REF20]
            assert seq_len == n_txt + n_canvas + n_ref_total, (
                f"❌ Slicing Mismatch: seq_len={seq_len} != txt({n_txt}) + canvas({n_canvas}) + ref({n_ref_total})"
            )

            c_start = n_txt
            c_end = n_txt + n_canvas
            r10_start = c_end
            r10_end = r10_start + n_ref10
            r20_start = r10_end
            r20_end = r20_start + n_ref20
            curr_t = cfg.current_timestep

            # --- Quantitative Attention Measurement Hook (Multi-layer accumulation across t in [0.70, 0.85]) ---
            if cfg.dump_attn and (0.70 <= curr_t <= 0.85):
                with torch.no_grad():
                    b_idx = 1 if q.shape[0] > 1 else 0
                    q_c = q[b_idx : b_idx + 1, :, c_start:c_end, :]
                    k_r10 = k[b_idx : b_idx + 1, :, r10_start:r10_end, :]
                    k_r20 = k[b_idx : b_idx + 1, :, r20_start:r20_end, :]

                    head_dim = q.shape[-1]
                    scale = 1.0 / math.sqrt(head_dim)
                    num_heads = q.shape[1]

                    sim_10 = torch.einsum("b h c d, b h r d -> c r", q_c, k_r10) * scale / num_heads
                    sim_20 = torch.einsum("b h c d, b h r d -> c r", q_c, k_r20) * scale / num_heads

                    map_10 = sim_10.mean(dim=-1).view(cfg.lat_h, cfg.lat_w).float().cpu().numpy()
                    map_20 = sim_20.mean(dim=-1).view(cfg.lat_h, cfg.lat_w).float().cpu().numpy()

                    cfg._accum_map10.append(map_10)
                    cfg._accum_map20.append(map_20)
                    cfg._accum_logit10.append(float(sim_10.mean().item()))
                    cfg._accum_logit20.append(float(sim_20.mean().item()))

            # --- Pass 1: Strict Unconditional Fallback to Native BFL Attention ---
            # Guarantees zero circular comparison: Pass 1 ALWAYS runs native BFL orig_causal_attn_fn!
            if not cfg.active or cfg.mode == "none":
                return self.orig_causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)

            # --- Pass 2, 3, 4: Unified Joint SDPA with Proper Causal Masking ---
            attn_bias = torch.zeros((1, 1, seq_len, seq_len), dtype=q.dtype, device=q.device)

            # Causal Isolation for Ref Tokens:
            # Ref tokens must ONLY self-attend to reference keys [r10_start:r20_end].
            # Block Ref queries from attending to Txt [0:c_start] and Canvas [c_start:r10_start]:
            attn_bias[:, :, r10_start:r20_end, :r10_start] = -10000.0

            if cfg.mode == "null_test":
                # Pass 2: Null Control.
                # Canvas -> Ref has 0.0 bias (unsteered). Txt and Canvas attend to all keys naturally.
                pass

            elif cfg.mode == "boost_only":
                # Amplify Canvas -> Slot 20 keys uniformly
                attn_bias[:, :, c_start:c_end, r20_start:r20_end] = cfg.boost_val_20

            elif cfg.mode in ["soft_regional", "scheduled_soft_regional"]:
                h_coords = torch.arange(cfg.lat_h, device=q.device, dtype=torch.float32)
                y_grid = (h_coords.unsqueeze(1).expand(cfg.lat_h, cfg.lat_w).flatten() + 0.5) / float(cfg.lat_h)

                # Sigmoid transition: 0 (top) -> 1 (bottom)
                w_20 = torch.sigmoid(cfg.sharpness * (y_grid - cfg.split_y))
                w_10 = 1.0 - w_20

                # User's brilliant recommendation: retain context with suppress_val (-3.0)
                bias_10 = cfg.suppress_val + (cfg.boost_val_10 - cfg.suppress_val) * w_10
                bias_20 = cfg.suppress_val + (cfg.boost_val_20 - cfg.suppress_val) * w_20

                time_scale = 1.0
                if cfg.mode == "scheduled_soft_regional":
                    if curr_t > cfg.t_start:
                        time_scale = max(0.0, (1.0 - curr_t) / (1.0 - cfg.t_start))
                    elif curr_t < cfg.t_end:
                        time_scale = max(0.0, curr_t / cfg.t_end)
                    else:
                        time_scale = 1.0

                bias_10 = (bias_10 * time_scale).to(dtype=q.dtype).view(1, 1, n_canvas, 1)
                bias_20 = (bias_20 * time_scale).to(dtype=q.dtype).view(1, 1, n_canvas, 1)

                # ONLY steer the interaction between Canvas queries and Ref keys!
                attn_bias[:, :, c_start:c_end, r10_start:r10_end] = bias_10
                attn_bias[:, :, c_start:c_end, r20_start:r20_end] = bias_20

            # Execute SINGLE FULL JOINT ATTENTION CALL with true causal isolation for ref
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
            return rearrange(out, "b h n d -> b n (h d)")





        flux_model.causal_attn_fn = steered_causal_attn_fn
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.orig_causal_attn_fn is not None:
            flux_model.causal_attn_fn = self.orig_causal_attn_fn


def denoise_steered_cfg(
    model: Flux2,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: list[float],
    guidance: float,
    img_cond_seq: torch.Tensor,
    img_cond_seq_ids: torch.Tensor,
    steering_cfg: AttentionSteeringConfig,
) -> torch.Tensor:
    """Executes Euler ODE Flow Matching with True CFG and active attention steering updates."""
    # Duplicate img and img_ids across batch dimension for CFG [uncond, cond]
    img = torch.cat([img, img], dim=0)
    img_ids = torch.cat([img_ids, img_ids], dim=0)

    if img_cond_seq is not None:
        assert img_cond_seq_ids is not None
        img_cond_seq = torch.cat([img_cond_seq, img_cond_seq], dim=0)
        img_cond_seq_ids = torch.cat([img_cond_seq_ids, img_cond_seq_ids], dim=0)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        steering_cfg.current_timestep = float(t_curr)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=None,
        )

        if img_cond_seq is not None:
            pred = pred[:, : img.shape[1]]

        pred_uncond, pred_cond = pred.chunk(2)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        pred = torch.cat([pred, pred], dim=0)

        img = img + (t_prev - t_curr) * pred

    return img.chunk(2)[0]



# ==============================================================================
# BENCHMARK RUNNER & GRID STITCHER
# ==============================================================================

BENCHMARK_SUITE = [
    {
        "id": "pass1_baseline_vanilla",
        "title": "Pass 1: Baseline (Vanilla)",
        "subtitle": "Standard BFL Attention Fallback (Full Bidirectional Joint Attention)",
        "mode": "none",
    },
    {
        "id": "pass2_null_control",
        "title": "Pass 2: Null Control (Bias=0.0)",
        "subtitle": "Steered Code Path with Bias=0.0 (Sanity Check)",
        "mode": "null_test",
    },
    {
        "id": "pass_boost_only",
        "title": "Pass: Pure Logit Boost",
        "subtitle": "Global Beta_20 = +2.0 (No Spatial Masking)",
        "mode": "boost_only",
    },
    {
        "id": "pass3_soft_regional",
        "title": "Pass 3: Soft Regional Routing",
        "subtitle": "Top vs Bottom Sigmoid (+2.0 inside, -3.0 context bias)",
        "mode": "soft_regional",
    },
    {
        "id": "pass4_scheduled_soft_regional",
        "title": "Pass 4: Scheduled Soft Regional",
        "subtitle": "Sigmoid Routing Scheduled on Timesteps [0.85, 0.20]",
        "mode": "scheduled_soft_regional",
    },
]




def save_attention_heatmap(
    heatmap_2d: np.ndarray,
    out_path: str | Path,
    title: str = "Attention Logits Map",
    cmap_name: str = "magma",
):
    """Saves a 2D attention logit map as an upsampled, colorized heatmap PNG."""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 7), dpi=150)
        im = ax.imshow(heatmap_2d, cmap=cmap_name, aspect="auto")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # Fallback to pure PIL grayscale if matplotlib is not available
        norm = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-6)
        img = Image.fromarray((norm * 255).astype(np.uint8))
        img = img.resize((360, 640), Image.Resampling.BILINEAR)
        img.save(str(out_path))


def stitch_comparison_panel(
    results: list[tuple[dict, Image.Image]],
    output_path: str,
    title_text: str,
    subtitle_text: str,
    prompt_text: str,
):
    """Stitches a side-by-side master comparison panel with descriptive headers."""
    num_cols = len(results)
    if num_cols < 2:
        return
    col_w, col_h = results[0][1].size

    header_h = 170
    footer_h = 80
    margin = 20
    spacing = 16

    total_w = margin * 2 + col_w * num_cols + spacing * (num_cols - 1)
    total_h = header_h + col_h + footer_h + margin * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 17, 23))
    draw = ImageDraw.Draw(canvas)


    try:
        font_main = ImageFont.truetype(FONT_REGISTRY["bevietnam"], 32)
        font_sub = ImageFont.truetype(FONT_REGISTRY["bevietnam"], 18)
        font_col_t = ImageFont.truetype(FONT_REGISTRY["bevietnam"], 22)
        font_col_s = ImageFont.truetype(FONT_REGISTRY["bevietnam"], 14)
        font_foot = ImageFont.truetype(FONT_REGISTRY["bevietnam"], 16)
    except Exception:
        font_main = font_sub = font_col_t = font_col_s = font_foot = ImageFont.load_default()

    # Master Banner Header
    draw.text(
        (margin, margin),
        "🚀 TENDOO AI: TRAINING-FREE ATTENTION STEERING & REGIONAL ROUTING BENCHMARK",
        font=font_main,
        fill=(255, 215, 0),
    )
    draw.text(
        (margin, margin + 44),
        f"📝 Slot 10 (t=10.0, Top): '{title_text}'  |  Slot 20 (t=20.0, Bottom): '{subtitle_text}'",
        font=font_sub,
        fill=(240, 240, 240),
    )
    prompt_preview = prompt_text[:140] + ("..." if len(prompt_text) > 140 else "")
    draw.text(
        (margin, margin + 74),
        f"🎨 Prompt: \"{prompt_preview}\"",
        font=font_sub,
        fill=(180, 190, 205),
    )
    draw.text(
        (margin, margin + 104),
        "🔬 Hypothesis: Soft Sigmoid Regional Masking with -3.0 Context Bias prevents collision without style disparity.",
        font=font_sub,
        fill=(130, 215, 245),
    )

    y_img = header_h + margin

    # Draw Columns
    for idx, (meta, img) in enumerate(results):
        x_img = margin + idx * (col_w + spacing)

        # Draw Column Banner
        card_bg = (26, 32, 44) if idx == 0 else (22, 42, 60)
        draw.rectangle([x_img, y_img - 48, x_img + col_w, y_img - 4], fill=card_bg)
        accent_color = (255, 100, 100) if idx == 0 else ((100, 200, 255) if idx < 3 else (120, 255, 150))
        draw.text((x_img + 12, y_img - 44), meta["title"], font=font_col_t, fill=accent_color)
        draw.text((x_img + 12, y_img - 22), meta["subtitle"], font=font_col_s, fill=(200, 210, 225))

        # Paste Image
        canvas.paste(img, (x_img, y_img))
        draw.rectangle([x_img, y_img, x_img + col_w, y_img + col_h], outline=(50, 65, 85), width=2)

    # Footer
    y_foot = y_img + col_h + 16
    draw.text(
        (margin, y_foot),
        "💡 Insight: In Softmax, exp(+2.0) / exp(-3.0) gives 148x local advantage, while keeping a 5% baseline gradient for global style coherence.",
        font=font_foot,
        fill=(160, 175, 195),
    )
    draw.text(
        (margin, y_foot + 26),
        "Zero modifications to BFL upstream core (src/flux2/). Execution executed entirely in-memory via PyTorch SDPA float mask.",
        font=font_foot,
        fill=(120, 135, 155),
    )

    canvas.save(output_path)
    print(f"📊 Master 4-Way Comparison Panel saved: {output_path}")


def run_benchmark(
    text_title: str = "CÀ PHÊ SỮA ĐÁ",
    text_subtitle: str = "ĐẬM ĐÀ HƯƠNG VỊ VIỆT",
    font_title: str = "playfair",
    font_subtitle: str = "bevietnam",
    prompt: str = (
        "Poster quảng cáo quầy bar cà phê gỗ mộc mạc cổ điển, ánh sáng ven ấm áp tương phản cao, "
        "dòng chữ tiêu đề 3D mạ vàng đồng cổ sắc nét nổi bật ở phía trên, dòng chữ slogan màu trắng sắc nét "
        "ở chân quầy bar phía dưới, bố cục sạch sẽ gọn gàng, không có chữ ký, không có watermark, không có chữ trang trí thừa"
    ),
    width: int = 576,
    height: int = 1024,
    num_steps: int = 50,
    guidance: float = 4.5,
    seed: int = 42,
    boost_val_20: float = 2.0,
    suppress_val: float = -3.0,
    split_y: float = 0.48,
    sharpness: float = 14.0,
    output_dir: str = "output_attention_steering_benchmark",
    model_name: str = "flux.2-klein-base-4b",
    checkpoint_dir: str | None = None,
    mode_select: str = "all",
    single_line_sub: bool = True,
    dump_attn: bool = True,
):
    """Executes the Training-Free Attention Steering Benchmark."""
    start_total = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    width = (width // 16) * 16
    height = (height // 16) * 16
    lat_h = height // 16
    lat_w = width // 16

    print("=" * 80)
    print(" 🚀 TENDOO AI: TRAINING-FREE ATTENTION STEERING BENCHMARK (V2 SLICING)")
    print("=" * 80)
    print(f"📝 Title (Slot 10, Top)   : '{text_title}' (Font: {font_title})")
    print(f"📝 Subtitle (Slot 20, Btm): '{text_subtitle}' (Font: {font_subtitle}, SingleLine: {single_line_sub})")
    print(f"📐 Canvas Resolution      : {width}x{height} pixels (Latent: {lat_h}x{lat_w} = {lat_h * lat_w} tokens)")
    print(f"⚙️ Hyperparams            : Boost = +{boost_val_20}, Suppress = {suppress_val}, Split Y = {split_y}, Sharpness = {sharpness}")
    print(f"⚙️ Sampling               : {num_steps} steps, CFG = {guidance}, Seed = {seed}")
    print(f"🎯 Target Model           : {model_name} | Dump Attn: {dump_attn}")
    print("=" * 80)

    # 1. Device Allocation
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_dit = torch.device("cuda:0")
        device_ae = torch.device("cuda:1")
        device_te = torch.device("cuda:1")
        print("🚀 Dual GPU Mode: DiT on GPU 0 | VAE & Qwen3 on GPU 1")
    else:
        device_dit = device_ae = device_te = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Single Device Mode: {device_dit}")

    # 2. Render In-Context Glyphs
    print("\n[1/4] Generating In-Context Vietnamese Glyphs...")
    resolved_font_title = resolve_font_path(font_title)
    resolved_font_sub = resolve_font_path(font_subtitle)

    glyph_title = create_glyph_image(
        text=text_title,
        target_width=min(width - 64, 512),
        target_height=192,
        font_path=resolved_font_title,
    )
    glyph_sub = create_glyph_image(
        text=text_subtitle,
        target_width=min(width - 32, 544) if single_line_sub else min(width - 64, 512),
        target_height=144 if single_line_sub else 224,
        font_path=resolved_font_sub,
        force_single_line=single_line_sub,
    )



    glyph_title.save(out_path / "glyph_title_t10_preview.png")
    glyph_sub.save(out_path / "glyph_subtitle_t20_preview.png")
    print("  -> Saved glyph previews to disk.")

    # 3. Load Models
    print("\n[2/4] Loading FLUX.2 Klein 4B Base Models...")
    if checkpoint_dir:
        os.environ["PERSISTENT_DATA_DIR"] = checkpoint_dir

    model = load_flow_model(model_name, device=device_dit)
    ae = load_ae(model_name, device=device_ae)
    text_encoder = load_qwen3_embedder(variant="4B", device=device_te)

    # Encode Text Prompt
    print("  -> Encoding Prompt via Qwen3-4B-FP8...")
    with torch.no_grad():
        txt = text_encoder(["", prompt])
        txt, txt_ids = batched_prc_txt(txt)
        txt = txt.to(device_dit)
        txt_ids = txt_ids.to(device_dit)

    if num_gpus < 2:
        del text_encoder
        torch.cuda.empty_cache()

    # Encode Glyphs to 4D RoPE Tokens
    print("  -> Encoding Glyphs to 4D RoPE Tokens...")
    ref_tokens_10, ref_ids_10 = encode_glyph_to_incontext_tokens(
        ae=ae, glyph_img=glyph_title, t_offset=10.0, device=device_ae
    )
    ref_tokens_20, ref_ids_20 = encode_glyph_to_incontext_tokens(
        ae=ae, glyph_img=glyph_sub, t_offset=20.0, device=device_ae
    )

    all_ref_tokens = torch.cat([ref_tokens_10, ref_tokens_20], dim=1).to(device_dit)
    all_ref_ids = torch.cat([ref_ids_10, ref_ids_20], dim=1).to(device_dit)

    num_ref10 = ref_tokens_10.shape[1]
    num_ref20 = ref_tokens_20.shape[1]
    num_canvas = lat_h * lat_w
    num_txt = txt.shape[1]

    print(f"  -> Reference Tokens: Slot 10 = {num_ref10} tokens, Slot 20 = {num_ref20} tokens")
    print(f"  -> Canvas Tokens   : {num_canvas} tokens (Grid {lat_h}x{lat_w})")
    print(f"  -> Total Combined Sequence: {num_txt + num_canvas + num_ref10 + num_ref20} tokens")

    # Fixed Initial Latent Noise (Strict Identical Noise across all passes)
    torch.manual_seed(seed)
    z_init = torch.randn(1, 128, lat_h, lat_w, device=device_dit, dtype=torch.bfloat16)
    img_tokens, img_ids = prc_img(z_init[0])
    img_tokens = img_tokens.unsqueeze(0).to(device_dit)
    img_ids = img_ids.unsqueeze(0).to(device_dit)

    timesteps = get_schedule(num_steps=num_steps, image_seq_len=num_canvas)

    # 4. Execute Benchmark Passes
    print("\n[3/4] Running Benchmark Passes...")
    test_cases = BENCHMARK_SUITE if mode_select == "all" else [c for c in BENCHMARK_SUITE if mode_select in c["id"]]

    results = []

    for sc in test_cases:
        print("\n" + "-" * 80)
        print(f"▶️ RUNNING: {sc['title']}")
        print(f"   Description: {sc['subtitle']}")
        print(f"   Mode       : {sc['mode']}")
        print("-" * 80)

        pass_cfg = AttentionSteeringConfig(
            mode=sc["mode"],
            active=(sc["mode"] != "none"),
            num_txt=num_txt,
            num_canvas=num_canvas,
            num_ref10=num_ref10,
            num_ref20=num_ref20,
            lat_h=lat_h,
            lat_w=lat_w,
            boost_val_10=0.0,
            boost_val_20=boost_val_20,
            suppress_val=suppress_val,
            split_y=split_y,
            sharpness=sharpness,
            dump_attn=dump_attn,
        )

        t_pass_start = time.time()

        with AttentionSteeringManager(pass_cfg):
            with torch.no_grad():
                out_latent = denoise_steered_cfg(
                    model=model,
                    img=img_tokens.clone(),
                    img_ids=img_ids.clone(),
                    txt=txt,
                    txt_ids=txt_ids,
                    timesteps=timesteps,
                    guidance=guidance,
                    img_cond_seq=all_ref_tokens,
                    img_cond_seq_ids=all_ref_ids,
                    steering_cfg=pass_cfg,
                )

                out_latent = rearrange(out_latent, "b (h w) c -> b c h w", h=lat_h, w=lat_w)
                out_pixels = ae.decode(out_latent.to(device_ae))
                out_pixels = ((out_pixels[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
                pass_img = Image.fromarray(out_pixels)

        elapsed = time.time() - t_pass_start
        pass_file = out_path / f"{sc['id']}.png"
        pass_img.save(pass_file)
        print(f"   ✅ Finished in {elapsed:.2f}s | Saved: {pass_file.name}")

        # Quantitative attention analysis and heatmap export
        stats = pass_cfg.finalize_stats() if pass_cfg.dump_attn else None
        if stats is not None:
            print(f"   📊 [LAYER-AVERAGED ATTENTION METRICS - {sc['title']}]")
            print(f"      • Captured DiT Layers           : {stats['num_layers_sampled']}")
            print(f"      • Canvas -> Ref10 (Title)   Logit: {stats['mean_logit_10']:+.4f}")
            print(f"      • Canvas -> Ref20 (Subtitle) Logit: {stats['mean_logit_20']:+.4f}")
            print(f"      • Intrinsic Logit Gap (ΔS)       : {stats['logit_gap']:+.4f}")
            print(f"      • Optimal Compensatory Beta (β*) : {stats['logit_gap']:+.4f}")

            hm10_path = out_path / f"attn_heatmap_{sc['id']}_title.png"
            hm20_path = out_path / f"attn_heatmap_{sc['id']}_subtitle.png"
            save_attention_heatmap(stats["map_10"], hm10_path, title=f"{sc['title']} -> Title (t=10)")
            save_attention_heatmap(stats["map_20"], hm20_path, title=f"{sc['title']} -> Subtitle (t=20)")
            print(f"      🖼️ Heatmaps saved: {hm10_path.name} & {hm20_path.name}")


        results.append((sc, pass_img))

    # 5. Master Panel Stitching
    if len(results) >= 2:
        print(f"\n[4/4] Stitching Master Comparison Panel ({len(results)} columns)...")
        master_panel_file = out_path / "ATTENTION_STEERING_COMPARISON.png"
        stitch_comparison_panel(
            results=results,
            output_path=str(master_panel_file),
            title_text=text_title,
            subtitle_text=text_subtitle,
            prompt_text=prompt,
        )

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print("🎉 ATTENTION STEERING BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"⏱️ Total Time: {total_elapsed:.2f}s")
    print(f"📁 Output Directory: {out_path.resolve()}")
    if len(results) >= 2:
        print(f"📊 Master Grid Image: {out_path / 'ATTENTION_STEERING_COMPARISON.png'}")
    print("=" * 80 + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tendoo AI - Training-Free Attention Steering Benchmark"
    )
    parser.add_argument("--text_title", type=str, default="CÀ PHÊ SỮA ĐÁ", help="Headline text (Slot 10, Top)")
    parser.add_argument("--text_subtitle", type=str, default="ĐẬM ĐÀ HƯƠNG VỊ VIỆT", help="Subtitle text (Slot 20, Bottom)")
    parser.add_argument("--font_title", type=str, default="playfair", help="Font alias for Title")
    parser.add_argument("--font_subtitle", type=str, default="bevietnam", help="Font alias for Subtitle")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Poster quảng cáo quầy bar cà phê gỗ mộc mạc cổ điển, ánh sáng ven ấm áp tương phản cao, "
            "dòng chữ tiêu đề 3D mạ vàng đồng cổ sắc nét nổi bật ở phía trên, dòng chữ slogan màu trắng sắc nét "
            "ở chân quầy bar phía dưới, bố cục sạch sẽ gọn gàng, không có chữ ký, không có watermark, không có chữ trang trí thừa"
        ),
        help="Text prompt (describes material, optics, surfaces; NO literal text repetition)",
    )

    parser.add_argument("--width", type=int, default=576, help="Canvas width (default: 576 for 9:16)")
    parser.add_argument("--height", type=int, default=1024, help="Canvas height (default: 1024 for 9:16)")
    parser.add_argument("--steps", type=int, default=50, help="Euler ODE steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG Guidance scale (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Steering parameters
    parser.add_argument("--boost_val", type=float, default=2.0, help="Positive boost beta for Slot 20 (default: 2.0)")
    parser.add_argument("--suppress_val", type=float, default=-3.0, help="Negative bias for non-target region (default: -3.0)")
    parser.add_argument("--split_y", type=float, default=0.48, help="Vertical split threshold (default: 0.48)")
    parser.add_argument("--sharpness", type=float, default=14.0, help="Sigmoid transition sharpness (default: 14.0)")

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "baseline", "null", "boost", "regional", "scheduled"],
        help="Execution mode (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_attention_steering_benchmark",
        help="Output directory for generated benchmark images",
    )
    parser.add_argument("--model_name", type=str, default="flux.2-klein-base-4b", help="FLUX.2 model variant")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to persistent-data")
    parser.add_argument("--single_line_sub", action="store_true", default=True, help="Force subtitle glyph onto single line")
    parser.add_argument("--no_single_line_sub", action="store_false", dest="single_line_sub", help="Allow multi-line wrapping for subtitle")
    parser.add_argument("--dump_attn", action="store_true", default=True, help="Record and visualize quantitative attention heatmaps")
    parser.add_argument("--no_dump_attn", action="store_false", dest="dump_attn", help="Disable attention visualization")

    args = parser.parse_args()

    run_benchmark(
        text_title=args.text_title,
        text_subtitle=args.text_subtitle,
        font_title=args.font_title,
        font_subtitle=args.font_subtitle,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        boost_val_20=args.boost_val,
        suppress_val=args.suppress_val,
        split_y=args.split_y,
        sharpness=args.sharpness,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        mode_select=args.mode,
        single_line_sub=args.single_line_sub,
        dump_attn=args.dump_attn,
    )

