"""
====================================================================================================
TENDOO AI - DIT 4B BASE LORA TRAINING ENGINE (PHASE 3 MASTER PIPELINE) — v3 (audited)
====================================================================================================
Script: scripts/train_lora_dit.py
Target Model: FLUX.2-klein-base-4B (4B DiT, Qwen3-4B-FP8, 128-ch VAE)
Target Hardware: 2x NVIDIA A30 (48GB VRAM total) on JupyterLab / Remote Server

Changes in this revision vs the previous draft:
  1. DDP now actually SHARDS the epoch's sample order across ranks (rank_id::world_size on a
     rank-shared resampled order) instead of every rank independently iterating the full dataset.
     Previously DDP was correct but delivered ~0% real speedup — this fixes that.
  2. Weighted sampling now reads PER-SLOT token counts (saved at pre-cache time) instead of the
     aggregate ref_tokens length, so a thin text slot sitting next to a large product image is no
     longer invisible to the oversampling logic. Also honors an optional manifest "known_hard": true
     field.
  3. --weighted-sampling is now a real togglable flag (BooleanOptionalAction: --weighted-sampling /
     --no-weighted-sampling), replacing the previous store_true+default=True dead flag.
  4. Added --print-forward-signature: dumps the REAL runtime signature of Flux2.forward via
     inspect.signature() so you can verify num_ref_tokens / ref_fixed_timestep claims yourself
     instead of trusting pasted/paraphrased source. Run this before trusting anything else below.
  5. Cache format changed (adds ref_slot_lengths / ref_slot_types / known_hard per shard) — you must
     re-run --pre-cache; old .pt shards from the previous script version are not compatible.

STILL UNVERIFIED BY CLAUDE (carried forward as an assumption, not a confirmed fact):
  - Whether Flux2.forward() truly ignores per-token timestep distinction for ref vs canvas tokens,
    and whether ref_fixed_timestep is genuinely dead code only reachable via forward_kv_extract.
    Run --print-forward-signature yourself and read the real forward() body before large-scale
    training; this script does not (and cannot, from here) confirm that claim.
====================================================================================================
"""

from __future__ import annotations

import argparse
import inspect
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
# 0. RUNTIME SIGNATURE VERIFICATION (use this before trusting any static-analysis claim)
# ==================================================================================================
def print_forward_signature() -> None:
    """
    Dumps the ACTUAL runtime signature of Flux2.forward (and forward_kv_extract, if present)
    so you can verify, from your own environment, whether num_ref_tokens / ref_fixed_timestep
    claims made in prior discussion are accurate — rather than trusting pasted/paraphrased code.
    """
    print("=" * 90)
    print(" [*] Flux2.forward signature:")
    print("     ", inspect.signature(Flux2.forward))
    print(" [*] Flux2.forward source (first 40 lines):")
    try:
        src_lines = inspect.getsource(Flux2.forward).splitlines()
        for line in src_lines[:40]:
            print("     ", line)
    except (OSError, TypeError) as e:
        print(f"     <could not retrieve source: {e}>")

    if hasattr(Flux2, "forward_kv_extract"):
        print("\n [*] Flux2.forward_kv_extract signature:")
        print("     ", inspect.signature(Flux2.forward_kv_extract))
    else:
        print("\n [*] Flux2.forward_kv_extract: NOT FOUND on this class.")
    print("=" * 90)


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
    Cache-only in this revision (raw on-the-fly encoding was dead-code risk; pre-cache is required).
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        cache_dir: Union[str, Path],
    ):
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.records: List[Dict[str, Any]] = []

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        print(f" [*] Loaded {len(self.records)} samples from manifest: {self.manifest_path}")
        if self.cache_dir.exists():
            cached_count = len(list(self.cache_dir.glob("*.pt")))
            print(f" [*] Cache directory active: {self.cache_dir} ({cached_count} cached shards)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sid = record["id"]
        cache_file = self.cache_dir / f"{sid}.pt"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Missing cache shard for id={sid}: {cache_file}. "
                f"Run with --pre-cache first (cache format changed in this revision — "
                f"old shards from a previous script version are not compatible)."
            )
        return torch.load(cache_file, map_location="cpu", weights_only=True)


# ==================================================================================================
# 3. PRE-CACHING PIPELINE (HIGH-PERFORMANCE LATENT & EMBEDDING CACHING)
# ==================================================================================================
def pre_cache_dataset(
    manifest_path: Union[str, Path],
    cache_dir: Union[str, Path],
    model_name: str = "flux.2-klein-base-4b",
    device: torch.device = torch.device("cuda:0"),
) -> None:
    """
    Pre-encodes all target images, glyphs, products, and prompt embeddings into .pt files.

    v3 change: also stores PER-SLOT token counts/types (ref_slot_lengths, ref_slot_types,
    ref_slot_t_offsets) so training-time weighted sampling can identify a specific thin/weak
    slot inside a sample, instead of only seeing the aggregate ref_tokens length.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(" [*] TENDOO AI - FEATURE PRE-CACHING PIPELINE (v3, per-slot metadata)")
    print(f" [*] Manifest: {manifest_path}")
    print(f" [*] Cache Dir: {cache_path}")
    print("=" * 90)

    print(" [*] Loading VAE and Qwen3 Text Encoder...")
    ae = load_ae(model_name, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    text_encoder = load_qwen3_embedder(variant="4B", device=device)

    with torch.no_grad():
        empty_txt = text_encoder([""])
        empty_txt, empty_txt_ids = batched_prc_txt(empty_txt)
        empty_shard = {
            "empty_txt": empty_txt.cpu(),
            "empty_txt_ids": empty_txt_ids.cpu(),
        }
        torch.save(empty_shard, cache_path / "_empty_prompt.pt")
        print("  -> Saved unconditional null prompt embedding: _empty_prompt.pt")

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

        with torch.no_grad():
            z_0 = encode_image_file_to_latent(ae, target_file, width, height, device=device)

        ref_tokens_list: List[torch.Tensor] = []
        ref_ids_list: List[torch.Tensor] = []
        ref_slot_lengths: List[int] = []
        ref_slot_types: List[str] = []
        ref_slot_t_offsets: List[float] = []

        for slot in rec["slots"]:
            t_offset = float(slot["time_offset"])
            slot_path = PROJECT_ROOT / slot["path"]

            if slot["type"] == "glyph":
                g_img = Image.open(slot_path).convert("RGB")
                gw, gh = g_img.size
                with torch.no_grad():
                    g_latent = encode_image_file_to_latent(ae, slot_path, gw, gh, device=device)
                    toks, ids = encode_tensor_to_rope_tokens(g_latent, t_offset, device=device)
            elif slot["type"] == "product":
                with torch.no_grad():
                    p_latent = encode_image_file_to_latent(ae, slot_path, 1024, 1024, device=device)
                    toks, ids = encode_tensor_to_rope_tokens(p_latent, t_offset, device=device)
            else:
                raise ValueError(f"Unknown slot type: {slot['type']!r} in sample {sid}")

            ref_tokens_list.append(toks)
            ref_ids_list.append(ids)
            ref_slot_lengths.append(toks.shape[1])
            ref_slot_types.append(slot["type"])
            ref_slot_t_offsets.append(t_offset)

        all_ref_tokens = torch.cat(ref_tokens_list, dim=1)
        all_ref_ids = torch.cat(ref_ids_list, dim=1)

        with torch.no_grad():
            prompt_str = rec["prompt_clean"]
            txt_emb = text_encoder([prompt_str])
            txt_tokens, txt_ids = batched_prc_txt(txt_emb)

        shard_data = {
            "id": sid,
            "width": width,
            "height": height,
            "z_0": z_0.cpu(),
            "ref_tokens": all_ref_tokens.cpu(),
            "ref_ids": all_ref_ids.cpu(),
            "ref_slot_lengths": ref_slot_lengths,       # NEW: per-slot token count, same order as slots
            "ref_slot_types": ref_slot_types,            # NEW: "glyph" | "product", same order
            "ref_slot_t_offsets": ref_slot_t_offsets,    # NEW: RoPE t-offset per slot
            "txt_tokens": txt_tokens.cpu(),
            "txt_ids": txt_ids.cpu(),
            "modality": rec["modality"],
            "use_case": rec["use_case"],
            "known_hard": bool(rec.get("known_hard", False)),  # NEW: optional manifest-declared flag
        }
        torch.save(shard_data, out_shard)
        if idx % 10 == 0 or idx == len(records):
            print(f"   [{idx:04d}/{len(records):04d}] Cached {sid} -> {out_shard.name}")

    print("=" * 90)
    print(" [*] PRE-CACHING COMPLETE: All features ready on disk.")
    print("=" * 90)


# ==================================================================================================
# 4. WEIGHTED SAMPLING — PER-SLOT AWARE
# ==================================================================================================
def compute_sample_weight(
    shard: Dict[str, Any],
    thin_glyph_token_threshold: int = 350,
    thin_glyph_weight: float = 2.0,
    known_hard_weight: float = 2.0,
) -> float:
    """
    Weight a training sample by how likely it is to be a hard crosstalk case.

    Unlike checking the AGGREGATE ref_tokens length (which is dominated by any product-image
    slot and almost never trips a low threshold), this looks at the MINIMUM token count among
    only the "glyph" (text) slots in the sample — the actual quantity that was found to
    correlate with subtitle/CTA dropout under concurrent-slot crosstalk.
    Multiple applicable conditions multiply (capped) rather than override each other.
    """
    weight = 1.0

    slot_types = shard.get("ref_slot_types", [])
    slot_lengths = shard.get("ref_slot_lengths", [])
    glyph_lengths = [
        length for stype, length in zip(slot_types, slot_lengths) if stype == "glyph"
    ]
    if glyph_lengths and min(glyph_lengths) < thin_glyph_token_threshold:
        weight *= thin_glyph_weight

    if shard.get("known_hard", False):
        weight *= known_hard_weight

    return weight


def build_epoch_indices(
    train_indices: List[int],
    train_weights: List[float],
    weighted_sampling: bool,
    epoch: int,
    base_seed: int,
    rank_id: int,
    world_size: int,
) -> List[int]:
    """
    Builds this epoch's sample order, IDENTICAL across all ranks (uses a rank-independent RNG
    seeded from base_seed+epoch, not the per-rank-seeded global `random` module), then shards it
    contiguously by rank via strided slicing. This is what actually delivers DDP speedup — the
    previous version had every rank independently resample the FULL dataset, so 2 GPUs did 2x the
    redundant work instead of splitting it.
    """
    shared_rng = random.Random(base_seed * 100_003 + epoch)  # rank-independent, epoch-varying
    if weighted_sampling:
        full_order = shared_rng.choices(train_indices, weights=train_weights, k=len(train_indices))
    else:
        full_order = train_indices.copy()
        shared_rng.shuffle(full_order)
    return full_order[rank_id::world_size]


# ==================================================================================================
# 5. VALIDATION LOOP (flow-matching MSE — a proxy signal, NOT a substitute for OCR-based eval)
# ==================================================================================================
def evaluate_validation(
    model: nn.Module,
    val_dataset: List[Dict[str, Any]],
    null_txt: torch.Tensor,
    null_txt_ids: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Evaluates validation Flow Matching MSE loss on held-out samples at a fixed t=0.5.

    NOTE: this is a loss-based proxy only. A falling validation MSE indicates the LoRA is not
    catastrophically overfitting/diverging, but it does NOT confirm that Vietnamese text renders
    correctly or that crosstalk is resolved — that requires an actual inference pass + OCR check
    (see run_qualitative_eval below, which is a stub: Claude does not have your inference/OCR
    pipeline and cannot implement this part for you).
    """
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_dataset:
            z_0 = batch["z_0"].to(device, dtype=torch.bfloat16)
            ref_tokens = batch["ref_tokens"].to(device, dtype=torch.bfloat16)
            ref_ids = batch["ref_ids"].to(device)
            txt = batch["txt_tokens"].to(device, dtype=torch.bfloat16)
            txt_ids = batch["txt_ids"].to(device)

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


def run_qualitative_eval(ckpt_path: Path) -> None:
    """
    STUB. Claude does not have access to your inference/sampling wrapper or OCR pipeline
    (PaddleOCR setup, prompt/slot construction for a real generation pass) and cannot implement
    this for you from this conversation. Wire this to: load ckpt_path as LoRA weights on top of
    the base model, run denoise_cfg for a small held-out probe set (ideally including your known
    t=20 / thin-glyph hard cases), then OCR the rendered text and report per-sample accuracy.
    Call this alongside evaluate_validation() at your eval cadence, not as a replacement for it.
    """
    pass


# ==================================================================================================
# 6. FLOW MATCHING TRAINING ENGINE
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
    eval_every: int = 50,
    val_ratio: float = 0.10,
    weighted_sampling: bool = True,
    thin_glyph_token_threshold: int = 350,
    text_dropout: float = 0.10,
    seed: int = 42,
    device_str: str = "cuda",
) -> None:
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

    # Per-rank RNG for noise/dropout sampling (divergence across ranks here is fine and desired —
    # each rank should see different noise draws for its shard). Epoch ordering uses a SEPARATE
    # rank-independent RNG (see build_epoch_indices) so it does not consume from this stream.
    torch.manual_seed(seed + rank_id)
    np.random.seed(seed + rank_id)
    random.seed(seed + rank_id)

    output_path = Path(output_dir)
    if is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)
        print("=" * 90)
        print(" 🚀 TENDOO AI - LORA DIT 4B TRAINING LAUNCH (v3, audited)")
        print(f" [*] Manifest         : {manifest_path}")
        print(f" [*] Cache Dir        : {cache_dir}")
        print(f" [*] Output Dir       : {output_path}")
        print(f" [*] Target Steps     : {max_steps} steps (grad_accum={grad_accum_steps}, world_size={world_size})")
        print(f" [*] LoRA Config      : Rank={rank}, Alpha={alpha}, LR={lr}, TextDropout={text_dropout}")
        print(f" [*] DDP Mode         : {'Active — sharded (' + str(world_size) + ' GPUs)' if is_distributed else 'Single Device'}")
        print(f" [*] Execution Device : {device}")
        print("=" * 90)

    if is_main_process:
        print("\n[1/5] Loading FLUX.2-klein-base-4B DiT...")
    model = load_flow_model(model_name, device=device)
    model.eval()

    if is_main_process:
        print("\n[2/5] Injecting LoRA Adapters...")
    model, injected_modules = inject_lora_to_flux2_klein(
        model=model,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        dtype=torch.bfloat16,
    )

    # Base model stays in eval() (frozen norm stats); LoRA submodules explicitly set to train()
    # so their dropout is active. Order matters: eval() first (recurses over everything including
    # the newly injected modules), THEN selectively flip LoRA modules back to train().
    model.eval()
    for mod in injected_modules.values():
        mod.train()

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    empty_shard_file = Path(cache_dir) / "_empty_prompt.pt"
    if not empty_shard_file.exists():
        raise FileNotFoundError(f"Missing _empty_prompt.pt in cache. Please run with --pre-cache first!")
    empty_data = torch.load(empty_shard_file, map_location="cpu", weights_only=True)
    null_txt = empty_data["empty_txt"].to(device)
    null_txt_ids = empty_data["empty_txt_ids"].to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dataset + Train/Val Split (identical split on every rank: uses `seed`, not `seed+rank_id`)
    full_dataset = MilestoneADataset(manifest_path, cache_dir=cache_dir)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
    n_train = n_total - n_val

    split_rng = random.Random(seed)
    all_indices = list(range(n_total))
    split_rng.shuffle(all_indices)
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    val_dataset = [full_dataset[i] for i in val_indices]

    # Per-slot-aware sample weights (computed once; loads each train shard once)
    train_weights = [
        compute_sample_weight(full_dataset[idx], thin_glyph_token_threshold=thin_glyph_token_threshold)
        for idx in train_indices
    ]

    if is_main_process:
        n_weighted = sum(1 for w in train_weights if w > 1.0)
        print(f" [*] Dataset Split    : {n_train} Train samples, {n_val} Held-out Validation samples")
        print(f" [*] Weighted Sampling: {'Enabled' if weighted_sampling else 'Disabled'} "
              f"({n_weighted}/{n_train} train samples flagged as hard cases)")
        print("\n[3/5] Starting Flow Matching Optimization Loop...")

    step = 0
    epoch = 0
    accum_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()

    while step < max_steps:
        epoch += 1
        current_epoch_indices = build_epoch_indices(
            train_indices, train_weights, weighted_sampling, epoch, seed, rank_id, world_size
        )

        for idx in current_epoch_indices:
            batch = full_dataset[idx]
            z_0 = batch["z_0"].to(device, dtype=torch.bfloat16)
            ref_tokens = batch["ref_tokens"].to(device, dtype=torch.bfloat16)
            ref_ids = batch["ref_ids"].to(device)

            if random.random() < text_dropout:
                txt = null_txt
                txt_ids = null_txt_ids
            else:
                txt = batch["txt_tokens"].to(device, dtype=torch.bfloat16)
                txt_ids = batch["txt_ids"].to(device)

            t_val = random.random()
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
            pred_canvas_velocity = pred[:, :num_canvas, :]

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

                if step % eval_every == 0 and is_main_process and val_dataset:
                    raw_model = model.module if is_distributed else model
                    val_loss = evaluate_validation(raw_model, val_dataset, null_txt, null_txt_ids, device)
                    print(f"   📊 [Validation @ Step {step:04d}] Held-Out Val Loss (proxy only, not OCR): {val_loss:.4f}")
                    for mod in injected_modules.values():
                        mod.train()

                if (step % save_every == 0 or step == max_steps) and is_main_process:
                    ckpt_file = output_path / f"tendoo_lora_step_{step:04d}.safetensors"
                    raw_model = model.module if is_distributed else model
                    save_lora_weights(raw_model, ckpt_file)
                    run_qualitative_eval(ckpt_file)  # currently a no-op stub — see docstring

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
# 7. CLI INTERFACE
# ==================================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Tendoo AI - DiT 4B Base LoRA Training Pipeline")
    parser.add_argument("--manifest", type=str, default="data/milestone_a/dataset_manifest.jsonl")
    parser.add_argument("--cache-dir", type=str, default="data/milestone_a/cache")
    parser.add_argument("--output-dir", type=str, default="checkpoints/lora_milestone_a")
    parser.add_argument("--model-name", type=str, default="flux.2-klein-base-4b")
    parser.add_argument("--pre-cache", action="store_true", help="Run pre-caching of VAE latents & text embeddings")
    parser.add_argument("--print-forward-signature", action="store_true",
                         help="Print the real runtime Flux2.forward signature/source and exit — "
                              "use this to verify num_ref_tokens/ref_fixed_timestep claims yourself")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--text-dropout", type=float, default=0.10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--weighted-sampling", action=argparse.BooleanOptionalAction, default=True,
                         help="Oversample hard crosstalk cases (thin glyph slots / known_hard). "
                              "Use --no-weighted-sampling to disable.")
    parser.add_argument("--thin-glyph-token-threshold", type=int, default=350,
                         help="Glyph slots with fewer tokens than this get oversampled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.print_forward_signature:
        print_forward_signature()
        return

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
        thin_glyph_token_threshold=args.thin_glyph_token_threshold,
        text_dropout=args.text_dropout,
        seed=args.seed,
        device_str=args.device if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    main()