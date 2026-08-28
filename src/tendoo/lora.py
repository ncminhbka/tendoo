"""
====================================================================================================
TENDOO AI - PEFT LORA INJECTION ENGINE FOR FLUX.2-KLEIN-BASE-4B
====================================================================================================
Module: src/tendoo/lora.py
Purpose:
    Production-grade Low-Rank Adaptation (LoRA) module for FLUX.2 Klein 4B DiT:
    1. Injects trainable rank-32 adapters into:
       - 5 DoubleStreamBlocks: 'img_attn.qkv' and 'txt_attn.qkv' (Attention routing)
       - 20 SingleStreamBlocks: 'linear1' (Fused Joint QKV Attention + MLP)
    2. Freezes 100% of upstream BFL core weights (zero base model degradation).
    3. Initializes B matrix to ZERO, guaranteeing ΔW = 0 at step 0 (exact identity preservation).
    4. Provides standard safetensors serialization and checkpointing.
====================================================================================================
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.
    Computes: W_out = W_base(x) + (x @ A^T @ B^T) * (alpha / r)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 32,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Freeze base layer completely
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Dropout
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # LoRA weights
        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), dtype=dtype))

        # Reset parameters: A with Kaiming uniform, B with ZERO
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear projection
        result = self.base_layer(x)

        # LoRA delta projection (casted to LoRA parameter dtype)
        x_cast = x.to(self.lora_A.dtype)
        lora_out = (self.dropout(x_cast) @ self.lora_A.T) @ self.lora_B.T
        lora_out = (lora_out * self.scaling).to(result.dtype)

        return result + lora_out


def inject_lora_to_flux2_klein(
    model: nn.Module,
    r: int = 32,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, Dict[str, LoRALinear]]:
    """
    Injects LoRA modules into target linear layers of FLUX.2-klein-base-4B:
    - 5 DoubleStreamBlocks: img_attn.qkv, txt_attn.qkv
    - 20 SingleStreamBlocks: linear1
    """
    # 1. Freeze entire model
    for param in model.parameters():
        param.requires_grad = False

    injected_modules: Dict[str, LoRALinear] = {}

    # 2. Inject into DoubleStreamBlocks (5 blocks)
    if hasattr(model, "double_blocks"):
        for b_idx, block in enumerate(model.double_blocks):
            # img_attn.qkv
            if hasattr(block, "img_attn") and hasattr(block.img_attn, "qkv"):
                name = f"double_blocks.{b_idx}.img_attn.qkv"
                wrapped = LoRALinear(
                    block.img_attn.qkv,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    dtype=dtype,
                )
                block.img_attn.qkv = wrapped
                injected_modules[name] = wrapped

            # txt_attn.qkv
            if hasattr(block, "txt_attn") and hasattr(block.txt_attn, "qkv"):
                name = f"double_blocks.{b_idx}.txt_attn.qkv"
                wrapped = LoRALinear(
                    block.txt_attn.qkv,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    dtype=dtype,
                )
                block.txt_attn.qkv = wrapped
                injected_modules[name] = wrapped

    # 3. Inject into SingleStreamBlocks (20 blocks)
    if hasattr(model, "single_blocks"):
        for b_idx, block in enumerate(model.single_blocks):
            # linear1
            if hasattr(block, "linear1"):
                name = f"single_blocks.{b_idx}.linear1"
                wrapped = LoRALinear(
                    block.linear1,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    dtype=dtype,
                )
                block.linear1 = wrapped
                injected_modules[name] = wrapped

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = (trainable_params / total_params) * 100.0

    print("=" * 90)
    print(" [*] TENDOO AI - LORA INJECTION COMPLETE")
    print(f" [*] Injected Modules: {len(injected_modules)} layers")
    print(f" [*] Base Model Parameters: {total_params / 1e9:.2f}B (Frozen)")
    print(f" [*] Trainable LoRA Parameters: {trainable_params / 1e6:.2f}M ({trainable_ratio:.3f}%)")
    print(f" [*] Rank (r): {r} | Alpha: {lora_alpha} | Dropout: {lora_dropout} | Dtype: {dtype}")
    print("=" * 90)

    return model, injected_modules


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts only the trainable LoRA parameters into a clean state dict."""
    lora_sd: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_sd[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_sd[f"{name}.lora_B"] = module.lora_B.data.clone()
    return lora_sd


def save_lora_weights(model: nn.Module, save_path: Union[str, Path]):
    """Saves LoRA weights to safetensors format."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sd = extract_lora_state_dict(model)
    save_file(sd, str(p))
    print(f" [*] Saved LoRA weights ({len(sd)} tensors, {p.stat().st_size / (1024*1024):.2f} MB) -> {p}")


def load_lora_weights(model: nn.Module, load_path: Union[str, Path], strict: bool = True):
    """Loads LoRA weights from safetensors file into injected model."""
    p = Path(load_path)
    if not p.exists():
        raise FileNotFoundError(f"LoRA weights not found at: {p}")

    sd = load_file(str(p))
    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_A = f"{name}.lora_A"
            key_B = f"{name}.lora_B"
            if key_A in sd and key_B in sd:
                module.lora_A.data.copy_(sd[key_A])
                module.lora_B.data.copy_(sd[key_B])
                loaded_count += 1
            elif strict:
                raise KeyError(f"Missing weights for {name} in {p}")

    print(f" [*] Loaded LoRA weights into {loaded_count} layers from: {p}")
