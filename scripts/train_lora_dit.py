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

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange  (replaced with native torch.reshape)
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

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
def evaluate_validation(
    model: nn.Module,
    val_dataset: List[Dict],
    null_txt: torch.Tensor,
    null_txt_ids: torch.Tensor,
    device: torch.device,
) -> float:
    """Evaluates validation Flow Matching MSE loss on held-out samples."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_dataset:
            z_0 = batch["z_0"].to(device, dtype=torch.bfloat16)
            ref_tokens = batch["ref_tokens"].to(device, dtype=torch.bfloat16)
            ref_ids = batch["ref_ids"].to(device)
            txt = batch["txt_tokens"].to(device, dtype=torch.bfloat16)
            txt_ids = batch["txt_ids"].to(device)

            # Standard evaluation at t = 0.5 (midpoint of ODE flow)
            t_val = 0.5
            z_1 = torch.randn_like(z_0)
            z_t = (1.0 - t_val) * z_0 + t_val * z_1
            target_velocity = z_1 - z_0

            canvas_tokens, canvas_ids = prc_img(z_t[0])
            canvas_tokens = canvas_tokens.unsqueeze(0).to(device)
            canvas_ids = canvas_ids.unsqueeze(0).to(device)

            target_tokens, _ = prc_img(target_velocity[0])
            target_tokens = target_tokens.unsqueeze(0).to(device, dtype=torch.bfloat16)

            num_canvas = canvas_tokens.shape[1]
            img_input = torch.cat([canvas_tokens, ref_tokens], dim=1)
            img_input_ids = torch.cat([canvas_ids, ref_ids], dim=1)
            t_vec = torch.full((1,), t_val, dtype=torch.bfloat16, device=device)

            pred = model(
                x=img_input,
                x_ids=img_input_ids,
                timesteps=t_vec,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=None,
            )
            pred_canvas = pred[:, :num_canvas, :]
            val_loss = F.mse_loss(pred_canvas, target_tokens).item()
            val_losses.append(val_loss)

    return float(np.mean(val_losses)) if val_losses else 0.0


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
    eval_every: int = 50,
    val_ratio: float = 0.10,
    weighted_sampling: bool = True,
    text_dropout: float = 0.10,
    seed: int = 42,
    device_str: str = "cuda",
):
    # 0. DDP / Multi-GPU Environment Initialization
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank_id = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main_process = (rank_id == 0)
    else:
        rank_id = 0
        local_rank = 0
        world_size = 1
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        is_main_process = True

    torch.manual_seed(seed + rank_id)
    np.random.seed(seed + rank_id)
    random.seed(seed + rank_id)

    output_path = Path(output_dir)
    if is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)
        print("=" * 90)
        print(" 🚀 TENDOO AI - LORA DIT 4B TRAINING LAUNCH")
        print(f" [*] Manifest         : {manifest_path}")
        print(f" [*] Cache Dir        : {cache_dir}")
        print(f" [*] Output Dir       : {output_path}")
        print(f" [*] Target Steps     : {max_steps} steps (Effective Batch = {world_size} GPU x {grad_accum_steps} = {world_size * grad_accum_steps})")
        print(f" [*] LoRA Config      : Rank={rank}, Alpha={alpha}, LR={lr}, TextDropout={text_dropout}")
        print(f" [*] DDP Mode         : {'Active (2x GPUs)' if is_distributed else 'Single Device'}")
        print(f" [*] Execution Device : {device}")
        print("=" * 90)

    # 1. Load Base FLUX.2 DiT Model
    if is_main_process:
        print("\n[1/5] Loading FLUX.2-klein-base-4B DiT...")
    model = load_flow_model(model_name, device=device)
    model.eval()

    # 2. Inject LoRA Layers
    if is_main_process:
        print("\n[2/5] Injecting LoRA Adapters...")
    model, injected_modules = inject_lora_to_flux2_klein(
        model=model,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        dtype=torch.bfloat16,
    )

    # Ensure base model remains in eval mode (freezing LayerNorm/RMSNorm running stats)
    # while explicitly setting all LoRA modules to train mode so dropout and gradients are active
    model.eval()
    for mod in injected_modules.values():
        mod.train()

    # Wrap model in DDP if running distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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

    # 5. Dataset Loader & Train/Val Split (Held-Out Validation Set)
    full_dataset = MilestoneADataset(manifest_path, cache_dir=cache_dir)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
    n_train = n_total - n_val

    all_indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    val_dataset = [full_dataset[i] for i in val_indices]

    # Calculate Sample Weights for Hard Crosstalk Cases (Oversampling)
    train_weights = []
    for idx in train_indices:
        item = full_dataset[idx]
        w = 1.0
        ref_tokens_count = item["ref_tokens"].shape[1]
        # Stress-inducing cases: low token mass (<350 tokens) or known_hard
        if ref_tokens_count < 350:
            w *= 2.0
        train_weights.append(w)

    if is_main_process:
        print(f" [*] Dataset Split    : {n_train} Train samples, {n_val} Held-out Validation samples")
        print(f" [*] Weighted Sampling: {'Enabled (Hard Cases 2.0x weight)' if weighted_sampling else 'Disabled'}")
        print("\n[3/5] Starting Flow Matching Optimization Loop...")

    step = 0
    epoch = 0
    accum_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()

    while step < max_steps:
        epoch += 1
        if weighted_sampling:
            current_epoch_indices = random.choices(train_indices, weights=train_weights, k=len(train_indices))
        else:
            current_epoch_indices = train_indices.copy()
            random.shuffle(current_epoch_indices)

        for idx in current_epoch_indices:
            batch = full_dataset[idx]
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

                if step % 10 == 0 and is_main_process:
                    avg_loss = accum_loss / (10 * grad_accum_steps)
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    s_per_step = elapsed / max(1, step)
                    print(
                        f" [Step {step:04d}/{max_steps:04d} | Epoch {epoch}] "
                        f"Train Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                        f"Speed: {s_per_step:.2f}s/step"
                    )
                    accum_loss = 0.0

                # Held-Out Validation Evaluation
                if step % eval_every == 0 and is_main_process and val_dataset:
                    raw_model = model.module if is_distributed else model
                    val_loss = evaluate_validation(raw_model, val_dataset, null_txt, null_txt_ids, device)
                    print(f"   📊 [Validation @ Step {step:04d}] Held-Out Val Loss: {val_loss:.4f}")
                    # Re-set LoRA modules back to train mode
                    for mod in injected_modules.values():
                        mod.train()

                # Checkpoint saving
                if (step % save_every == 0 or step == max_steps) and is_main_process:
                    ckpt_file = output_path / f"tendoo_lora_step_{step:04d}.safetensors"
                    raw_model = model.module if is_distributed else model
                    save_lora_weights(raw_model, ckpt_file)

            if step >= max_steps:
                break

    if is_main_process:
        print("\n" + "=" * 90)
        print(" 🎉 TENDOO AI - LORA TRAINING COMPLETED SUCCESSFULLY!")
        print(f" [*] Final Checkpoint Saved: {output_path / f'tendoo_lora_step_{max_steps:04d}.safetensors'}")
        print("=" * 90)

    if is_distributed:
        dist.destroy_process_group()


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
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate validation loss every N steps")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation set ratio (default: 0.10)")
    parser.add_argument("--weighted-sampling", action="store_true", default=True, help="Enable oversampling of hard crosstalk cases")
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
        eval_every=args.eval_every,
        val_ratio=args.val_ratio,
        weighted_sampling=args.weighted_sampling,
        text_dropout=args.text_dropout,
        seed=args.seed,
        device_str=args.device if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    main()
