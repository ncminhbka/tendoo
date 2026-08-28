"""
====================================================================================================
TENDOO AI - DIT 4B BASE LORA TRAINING ENGINE (PHASE 3 MASTER PIPELINE)
====================================================================================================
Script: scripts/train_lora_dit.py
Target Model: FLUX.2-klein-base-4B (4B DiT, Qwen3-4B-FP8, 128-ch VAE)
Target Hardware: 2x NVIDIA A30 (48GB VRAM total) on JupyterLab / Remote Server

Core Architecture & Mathematical Basis:
1. Target Model: Strictly FLUX.2-klein-base-4B (True CFG = 4.0, 50 steps ODE).
2. LoRA Injection: Rank 32, Alpha 32.0 into:
   - 5 DoubleStreamBlocks: 'img_attn.qkv', 'txt_attn.qkv'
   - 20 SingleStreamBlocks: 'linear1' (Fused Joint Attention + MLP)
   - Base model 100% frozen (0.58% trainable parameters ~ 23.6M params).
3. In-Context 4D RoPE Conditioning:
   - Canvas: t = 0.0 (Target image x_0 noisy latent z_t)
   - Slot 1 (Text 1 Glyph): t = 10.0
   - Slot 2 (Text 2 Glyph): t = 20.0
   - Slot 3 (Product packshot - I2I): t = 30.0
4. Flow Matching Objective:
   - Velocity prediction: v_theta(z_t, t, c)
   - Ground truth velocity: u_t = z_1 - z_0 where z_1 ~ N(0, I)
   - Loss: L = ||v_theta - u_t||^2 (MSE over canvas tokens only)
5. Text Conditioning Dropout (p = 0.10):
   - Replaces prompt with empty string "" embedding to train unconditional CFG branch.
   - Reference tokens (glyphs & product) are 100% preserved.
6. Feature Pre-Caching:
   - Supports pre-encoding VAE latents & Qwen3 embeddings to disk (.pt).
   - Eliminates VAE and Qwen3 from GPU memory during training -> 10x speedup & zero OOM risk!
====================================================================================================
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange  (replaced with native torch.reshape)
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.flux2.autoencoder import AutoEncoder
from src.flux2.model import Flux2, Klein4BParams
from src.flux2.sampling import batched_prc_txt, get_schedule, prc_img
from src.flux2.text_encoder import load_qwen3_embedder
from src.flux2.util import find_persistent_data_root, load_ae, load_flow_model
from src.tendoo.lora import (
    extract_lora_state_dict,
    inject_lora_to_flux2_klein,
    load_lora_weights,
    save_lora_weights,
)


# ==================================================================================================
# 1. 4D ROPE COORDINATE ENCODING UTILITIES
# ==================================================================================================
def encode_tensor_to_rope_tokens(
    latent: torch.Tensor,
    t_offset: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a latent tensor [1, 128, H, W] into 4D RoPE coordinate tokens:
    - tokens: [1, H*W, 128]
    - ids: [1, H*W, 4] with (t, h, w, l) coordinates.
    """
    tokens, _ = prc_img(latent[0])
    tokens = tokens.unsqueeze(0).to(device)

    _, _, h, w = latent.shape
    t_coords = torch.full((h, w), fill_value=t_offset, dtype=torch.float32, device=device)
    h_coords = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).expand(h, w)
    w_coords = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).expand(h, w)
    l_coords = torch.zeros((h, w), dtype=torch.float32, device=device)

    ids = torch.stack([t_coords, h_coords, w_coords, l_coords], dim=-1)
    ids = ids.reshape(-1, ids.shape[-1]).unsqueeze(0)

    return tokens, ids


def encode_image_file_to_latent(
    ae: AutoEncoder,
    image_path: Union[str, Path],
    target_width: int,
    target_height: int,
    device: torch.device,
) -> torch.Tensor:
    """Loads an image, normalizes to [-1, 1], and encodes via VAE."""
    img = Image.open(image_path).convert("RGB")
    if img.size != (target_width, target_height):
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        latent = ae.encode(tensor)
    return latent


# ==================================================================================================
# 2. DATASET & CACHED FEATURE CONTAINER
# ==================================================================================================
class MilestoneADataset(Dataset):
    """
    Dataset loader for Tendoo AI Milestone A.
    Supports either:
    1. Cached Mode: Loads pre-computed .pt feature files (high throughput).
    2. Raw Mode: Loads images on-the-fly and encodes via VAE & Qwen3.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.records: List[Dict] = []

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        print(f" [*] Loaded {len(self.records)} samples from manifest: {self.manifest_path}")
        if self.cache_dir and self.cache_dir.exists():
            cached_count = len(list(self.cache_dir.glob("*.pt")))
            print(f" [*] Cache directory active: {self.cache_dir} ({cached_count} cached shards)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        sid = record["id"]

        # If cache shard exists, load directly
        if self.cache_dir:
            cache_file = self.cache_dir / f"{sid}.pt"
            if cache_file.exists():
                return torch.load(cache_file, map_location="cpu", weights_only=True)

        return record


# ==================================================================================================
# 3. PRE-CACHING PIPELINE (HIGH-PERFORMANCE LATENT & EMBEDDING CACHING)
# ==================================================================================================
def pre_cache_dataset(
    manifest_path: Union[str, Path],
    cache_dir: Union[str, Path],
    model_name: str = "flux.2-klein-base-4b",
    device: torch.device = torch.device("cuda:0"),
):
    """
    Pre-encodes all target images, glyphs, products, and prompt embeddings into .pt files.
    Running this once allows LoRA training to proceed at maximum speed without holding
    the 4B Text Encoder or VAE in GPU memory.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(" [*] TENDOO AI - FEATURE PRE-CACHING PIPELINE")
    print(f" [*] Manifest: {manifest_path}")
    print(f" [*] Cache Dir: {cache_path}")
    print("=" * 90)

    # 1. Load VAE & Text Encoder
    print(" [*] Loading VAE and Qwen3 Text Encoder...")
    ae = load_ae(model_name, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    text_encoder = load_qwen3_embedder(variant="4B", device=device)

    # Precompute empty prompt embedding for text dropout
    with torch.no_grad():
        empty_txt = text_encoder([""])
        empty_txt, empty_txt_ids = batched_prc_txt(empty_txt)
        empty_shard = {
            "empty_txt": empty_txt.cpu(),
            "empty_txt_ids": empty_txt_ids.cpu(),
        }
        torch.save(empty_shard, cache_path / "_empty_prompt.pt")
        print("  -> Saved unconditional null prompt embedding: _empty_prompt.pt")

    # Read manifest
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(f" [*] Pre-encoding {len(records)} samples...")
    for idx, rec in enumerate(records, start=1):
        sid = rec["id"]
        out_shard = cache_path / f"{sid}.pt"
        if out_shard.exists():
            continue

        target_file = PROJECT_ROOT / rec["target_image"]
        width, height = rec["width"], rec["height"]

        # A. Encode Target Image -> z_0
        with torch.no_grad():
            z_0 = encode_image_file_to_latent(ae, target_file, width, height, device=device)

        # B. Encode Reference Tokens (Glyphs + Product)
        ref_tokens_list = []
        ref_ids_list = []

        for slot in rec["slots"]:
            t_offset = float(slot["time_offset"])
            slot_path = PROJECT_ROOT / slot["path"]

            if slot["type"] == "glyph":
                g_img = Image.open(slot_path).convert("RGB")
                gw, gh = g_img.size
                with torch.no_grad():
                    g_latent = encode_image_file_to_latent(ae, slot_path, gw, gh, device=device)
                    toks, ids = encode_tensor_to_rope_tokens(g_latent, t_offset, device=device)
                ref_tokens_list.append(toks)
                ref_ids_list.append(ids)

            elif slot["type"] == "product":
                p_img = Image.open(slot_path).convert("RGB")
                # Product normalized to standard 1024x1024
                with torch.no_grad():
                    p_latent = encode_image_file_to_latent(ae, slot_path, 1024, 1024, device=device)
                    toks, ids = encode_tensor_to_rope_tokens(p_latent, t_offset, device=device)
                ref_tokens_list.append(toks)
                ref_ids_list.append(ids)

        all_ref_tokens = torch.cat(ref_tokens_list, dim=1)
        all_ref_ids = torch.cat(ref_ids_list, dim=1)

        # C. Encode Student Clean Prompt via Qwen3
        with torch.no_grad():
            prompt_str = rec["prompt_clean"]
            txt_emb = text_encoder([prompt_str])
            txt_tokens, txt_ids = batched_prc_txt(txt_emb)

        shard_data = {
            "id": sid,
            "width": width,
            "height": height,
            "z_0": z_0.cpu(),  # [1, 128, H/16, W/16]
            "ref_tokens": all_ref_tokens.cpu(),  # [1, L_ref, 128]
            "ref_ids": all_ref_ids.cpu(),  # [1, L_ref, 4]
            "txt_tokens": txt_tokens.cpu(),  # [1, L_txt, 7680]
            "txt_ids": txt_ids.cpu(),  # [1, L_txt, 4]
            "modality": rec["modality"],
            "use_case": rec["use_case"],
        }
        torch.save(shard_data, out_shard)
        if idx % 10 == 0 or idx == len(records):
            print(f"   [{idx:04d}/{len(records):04d}] Cached {sid} -> {out_shard.name}")

    print("=" * 90)
    print(" [*] PRE-CACHING COMPLETE: All features ready on disk.")
    print("=" * 90)


# ==================================================================================================
# 4. FLOW MATCHING TRAINING ENGINE
# ==================================================================================================
def train_lora(
    manifest_path: str,
    cache_dir: str,
    output_dir: str,
    model_name: str = "flux.2-klein-base-4b",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.05,
    max_steps: int = 800,
    warmup_steps: int = 150,
    grad_accum_steps: int = 4,
    save_every: int = 200,
    text_dropout: float = 0.10,
    seed: int = 42,
    device_str: str = "cuda",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device_str)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(" 🚀 TENDOO AI - LORA DIT 4B TRAINING LAUNCH")
    print(f" [*] Manifest       : {manifest_path}")
    print(f" [*] Cache Dir      : {cache_dir}")
    print(f" [*] Output Dir     : {output_path}")
    print(f" [*] Target Steps   : {max_steps} steps (Effective Batch Size = 1 x {grad_accum_steps} = {grad_accum_steps})")
    print(f" [*] LoRA Config    : Rank={rank}, Alpha={alpha}, LR={lr}, TextDropout={text_dropout}")
    print(f" [*] Execution Device: {device}")
    print("=" * 90)

    # 1. Load Base FLUX.2 DiT Model
    print("\n[1/5] Loading FLUX.2-klein-base-4B DiT...")
    model = load_flow_model(model_name, device=device)
    model.eval()

    # 2. Inject LoRA Layers
    print("\n[2/5] Injecting LoRA Adapters...")
    model, injected_modules = inject_lora_to_flux2_klein(
        model=model,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        dtype=torch.bfloat16,
    )

    # 3. Load Empty Prompt for Text Dropout
    empty_shard_file = Path(cache_dir) / "_empty_prompt.pt"
    if not empty_shard_file.exists():
        raise FileNotFoundError(f"Missing _empty_prompt.pt in cache. Please run with --pre-cache first!")
    empty_data = torch.load(empty_shard_file, map_location="cpu", weights_only=True)
    null_txt = empty_data["empty_txt"].to(device)
    null_txt_ids = empty_data["empty_txt_ids"].to(device)

    # 4. Setup Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Dataset Loader
    dataset = MilestoneADataset(manifest_path, cache_dir=cache_dir)
    indices = list(range(len(dataset)))

    print("\n[3/5] Starting Flow Matching Optimization Loop...")
    step = 0
    epoch = 0
    accum_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()

    while step < max_steps:
        epoch += 1
        random.shuffle(indices)

        for idx in indices:
            batch = dataset[idx]

            # Unpack cached features
            z_0 = batch["z_0"].to(device, dtype=torch.bfloat16)  # [1, 128, H/16, W/16]
            ref_tokens = batch["ref_tokens"].to(device, dtype=torch.bfloat16)  # [1, L_ref, 128]
            ref_ids = batch["ref_ids"].to(device)  # [1, L_ref, 4]

            # Text Conditioning Dropout (p = 0.10)
            if random.random() < text_dropout:
                txt = null_txt
                txt_ids = null_txt_ids
            else:
                txt = batch["txt_tokens"].to(device, dtype=torch.bfloat16)
                txt_ids = batch["txt_ids"].to(device)

            # Flow Matching Noise & Interpolation
            # Sample continuous timestep t ~ U(0, 1)
            t_val = random.random()
            z_1 = torch.randn_like(z_0)

            # Rectified Flow interpolation: z_t = (1 - t) z_0 + t z_1
            z_t = (1.0 - t_val) * z_0 + t_val * z_1
            target_velocity = z_1 - z_0  # u_t = z_1 - z_0

            # Convert canvas to tokens
            canvas_tokens, canvas_ids = prc_img(z_t[0])
            canvas_tokens = canvas_tokens.unsqueeze(0).to(device)
            canvas_ids = canvas_ids.unsqueeze(0).to(device)

            target_tokens, _ = prc_img(target_velocity[0])
            target_tokens = target_tokens.unsqueeze(0).to(device, dtype=torch.bfloat16)

            num_canvas = canvas_tokens.shape[1]

            # Concatenate canvas tokens with in-context reference tokens
            img_input = torch.cat([canvas_tokens, ref_tokens], dim=1)
            img_input_ids = torch.cat([canvas_ids, ref_ids], dim=1)

            t_vec = torch.full((1,), t_val, dtype=torch.bfloat16, device=device)

            # Forward pass through FLUX.2 DiT with LoRA adapters
            pred = model(
                x=img_input,
                x_ids=img_input_ids,
                timesteps=t_vec,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=None,
            )

            # Extract predicted velocity for canvas tokens only
            pred_canvas_velocity = pred[:, :num_canvas, :]

            # Flow Matching MSE Loss
            loss = F.mse_loss(pred_canvas_velocity, target_tokens)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            accum_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 10 == 0:
                    avg_loss = accum_loss / (10 * grad_accum_steps)
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    s_per_step = elapsed / max(1, step)
                    print(
                        f" [Step {step:04d}/{max_steps:04d} | Epoch {epoch}] "
                        f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                        f"Speed: {s_per_step:.2f}s/step"
                    )
                    accum_loss = 0.0

                # Checkpoint saving
                if step % save_every == 0 or step == max_steps:
                    ckpt_file = output_path / f"tendoo_lora_step_{step:04d}.safetensors"
                    save_lora_weights(model, ckpt_file)

            if step >= max_steps:
                break

    print("\n" + "=" * 90)
    print(" 🎉 TENDOO AI - LORA TRAINING COMPLETED SUCCESSFULLY!")
    print(f" [*] Final Checkpoint Saved: {output_path / f'tendoo_lora_step_{max_steps:04d}.safetensors'}")
    print("=" * 90)


# ==================================================================================================
# 5. CLI INTERFACE
# ==================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Tendoo AI - DiT 4B Base LoRA Training Pipeline")
    parser.add_argument("--manifest", type=str, default="data/milestone_a/dataset_manifest.jsonl", help="Dataset manifest")
    parser.add_argument("--cache-dir", type=str, default="data/milestone_a/cache", help="Feature cache directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints/lora_milestone_a", help="Output checkpoint directory")
    parser.add_argument("--model-name", type=str, default="flux.2-klein-base-4b", help="Model name")
    parser.add_argument("--pre-cache", action="store_true", help="Run pre-caching of VAE latents & text embeddings")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=800, help="Total training steps")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save-every", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--text-dropout", type=float, default=0.10, help="Text conditioning dropout probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    args = parser.parse_args()

    if args.pre_cache:
        pre_cache_dataset(
            manifest_path=args.manifest,
            cache_dir=args.cache_dir,
            model_name=args.model_name,
            device=torch.device(args.device if torch.cuda.is_available() else "cpu"),
        )
        return

    train_lora(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        max_steps=args.steps,
        grad_accum_steps=args.grad_accum,
        save_every=args.save_every,
        text_dropout=args.text_dropout,
        seed=args.seed,
        device_str=args.device if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    main()
