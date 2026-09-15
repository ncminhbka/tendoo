import base64
import io
import os
import sys
from pathlib import Path

import torch.nn as _nn


def _load_state_dict_verbose(module: "_nn.Module", sd: dict, label: str) -> None:
    """
    Loads a state dict and ALWAYS reports missing/unexpected keys explicitly,
    even when falling back to strict=False. A silent strict=False fallback can
    hide a broken key-conversion (e.g. Diffusers->BFL VAE remap) behind a model
    that "loads fine" but is missing real weights.
    """
    try:
        module.load_state_dict(sd, strict=True, assign=True)
        print(f"  -> [{label}] Loaded with strict=True (0 missing, 0 unexpected).")
    except Exception as e:
        print(f"Warning: Strict state dict loading failed for [{label}] ({e}), retrying non-strict...")
        result = module.load_state_dict(sd, strict=False, assign=True)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        print(f"  -> [{label}] Non-strict load: {len(missing)} missing key(s), {len(unexpected)} unexpected key(s).")
        if missing:
            print(f"     Missing (first 10): {missing[:10]}")
        if unexpected:
            print(f"     Unexpected (first 10): {unexpected[:10]}")
        if missing or unexpected:
            print(
                "     ⚠️  This checkpoint did NOT fully match the target architecture. "
                "Outputs may be silently degraded (e.g. VAE decode quality). "
                "Do not treat this as a successful load without reviewing the lists above."
            )

import huggingface_hub
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from .text_encoder import load_mistral_small_embedder, load_qwen3_embedder

FLUX2_MODEL_INFO = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b-kv": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B-kv",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b-kv.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_KV_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
        "use_kv_cache": True,
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
        "text_encoder_load_fn": load_mistral_small_embedder,
        "model_path": "FLUX2_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": True,
    },
}


def find_persistent_data_root() -> str | None:
    if "FLUX_CHECKPOINT_DIR" in os.environ and os.path.exists(os.environ["FLUX_CHECKPOINT_DIR"]):
        return os.environ["FLUX_CHECKPOINT_DIR"]

    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "persistent-data", "FLUX.2-klein-base-4B"),
        "/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
        "/persistent-data/FLUX.2-klein-base-4B",
        os.path.join(home, "persistent-data"),
        "/home/jovyan/persistent-data",
        "/persistent-data",
    ]

    cwd = Path.cwd()
    for base in [cwd, Path(__file__).resolve().parent.parent.parent]:
        curr = base
        for _ in range(5):
            candidates.append(str(curr / "persistent-data" / "FLUX.2-klein-base-4B"))
            candidates.append(str(curr / "persistent-data"))
            if curr.parent == curr:
                break
            curr = curr.parent

    for c in candidates:
        if c and os.path.exists(c):
            if os.path.exists(os.path.join(c, "FLUX.2-klein-base-4B")):
                return os.path.join(c, "FLUX.2-klein-base-4B")
            if (
                os.path.exists(os.path.join(c, "flux-2-klein-base-4b.safetensors"))
                or os.path.exists(os.path.join(c, "vae"))
                or os.path.exists(os.path.join(c, "text_encoder"))
            ):
                return c
    return None


def convert_diffusers_dit_to_bfl(
    sd: dict[str, torch.Tensor], depth: int, depth_single_blocks: int
) -> dict[str, torch.Tensor]:
    """
    Convert a HuggingFace Diffusers-format FLUX.2 DiT checkpoint (`x_embedder`,
    `context_embedder`, `transformer_blocks.*`, `single_transformer_blocks.*`, ...)
    into this module's BFL-native `Flux2` key layout (`img_in`, `txt_in`,
    `double_blocks.*`, `single_blocks.*`, ...).

    Added 2026-09-15: a persistent-data checkpoint at
    .../FLUX.2-klein-base-4B/transformer/diffusion_pytorch_model.safetensors turned
    out to be Diffusers-format on real-GPU load, and load_flow_model() had no
    conversion for the DiT (only load_ae() had one, for the VAE) -- every DiT weight
    silently stayed on the meta device and `.to(device)` crashed with "Cannot copy
    out of meta tensor; no data!". This mapping was derived key-by-key from that real
    missing/unexpected-key dump (every key on both sides accounted for), NOT verified
    end-to-end against the actual checkpoint yet (no GPU in this dev environment) --
    re-run load_flow_model() after this change; `_load_state_dict_verbose` will print
    the resulting missing/unexpected counts. If it isn't 0/0, report the exact
    remaining keys back rather than assuming this mapping is complete -- the most
    likely failure point is single_blocks' fused QKV+MLP layout (see note below).

    Two real structural differences, not just renames:
    1. DoubleStreamBlock's img_attn/txt_attn.qkv is ONE fused Linear(dim, 3*dim) in
       BFL; Diffusers exposes it as 3 separate Linears (to_q/to_k/to_v for the image
       stream, add_q_proj/add_k_proj/add_v_proj for the text/"added" stream) --
       concatenated here along dim=0 in [Q, K, V] order to match model.py's
       `rearrange(qkv, "B L (K H D) -> K B H L D", K=3, ...)` unpacking.
    2. SingleStreamBlock's linear1 fuses QKV *and* the MLP's first projection into
       ONE Linear in BFL; Diffusers' `to_qkv_mlp_proj` appears to already be fused
       the same way (unlike the double-block case) -- ASSUMED here to need only a
       rename, not a re-concatenation. This is the one part of this mapping most
       likely to be wrong if Diffusers' internal layout differs -- a shape mismatch
       here will surface as an explicit load_state_dict error, not silent corruption.
    """
    new_sd: dict[str, torch.Tensor] = {}

    def pop(key: str) -> torch.Tensor | None:
        return sd.pop(key, None)

    # --- Top-level embedders ---
    if (w := pop("x_embedder.weight")) is not None:
        new_sd["img_in.weight"] = w
    if (w := pop("context_embedder.weight")) is not None:
        new_sd["txt_in.weight"] = w
    if (w := pop("time_guidance_embed.timestep_embedder.linear_1.weight")) is not None:
        new_sd["time_in.in_layer.weight"] = w
    if (w := pop("time_guidance_embed.timestep_embedder.linear_2.weight")) is not None:
        new_sd["time_in.out_layer.weight"] = w
    # Guidance embedding only applies to non-distilled/guidance-embed variants;
    # Klein 4B/9B set use_guidance_embed=False so these are expected absent here.
    if (w := pop("time_guidance_embed.guidance_embedder.linear_1.weight")) is not None:
        new_sd["guidance_in.in_layer.weight"] = w
    if (w := pop("time_guidance_embed.guidance_embedder.linear_2.weight")) is not None:
        new_sd["guidance_in.out_layer.weight"] = w

    # --- Shared (NOT per-block) AdaLN modulation projections ---
    for diff_name, bfl_name in [
        ("double_stream_modulation_img.linear.weight", "double_stream_modulation_img.lin.weight"),
        ("double_stream_modulation_txt.linear.weight", "double_stream_modulation_txt.lin.weight"),
        ("single_stream_modulation.linear.weight", "single_stream_modulation.lin.weight"),
    ]:
        if (w := pop(diff_name)) is not None:
            new_sd[bfl_name] = w

    # --- Final layer ---
    if (w := pop("norm_out.linear.weight")) is not None:
        new_sd["final_layer.adaLN_modulation.1.weight"] = w
    if (w := pop("proj_out.weight")) is not None:
        new_sd["final_layer.linear.weight"] = w

    # --- Double (joint img/txt) stream blocks ---
    for i in range(depth):
        p = f"transformer_blocks.{i}."
        b = f"double_blocks.{i}."

        q, k, v = pop(p + "attn.to_q.weight"), pop(p + "attn.to_k.weight"), pop(p + "attn.to_v.weight")
        if q is not None and k is not None and v is not None:
            new_sd[b + "img_attn.qkv.weight"] = torch.cat([q, k, v], dim=0)
        if (w := pop(p + "attn.norm_q.weight")) is not None:
            new_sd[b + "img_attn.norm.query_norm.scale"] = w
        if (w := pop(p + "attn.norm_k.weight")) is not None:
            new_sd[b + "img_attn.norm.key_norm.scale"] = w
        if (w := pop(p + "attn.to_out.0.weight")) is not None:
            new_sd[b + "img_attn.proj.weight"] = w
        if (w := pop(p + "ff.linear_in.weight")) is not None:
            new_sd[b + "img_mlp.0.weight"] = w
        if (w := pop(p + "ff.linear_out.weight")) is not None:
            new_sd[b + "img_mlp.2.weight"] = w

        aq, ak, av = pop(p + "attn.add_q_proj.weight"), pop(p + "attn.add_k_proj.weight"), pop(p + "attn.add_v_proj.weight")
        if aq is not None and ak is not None and av is not None:
            new_sd[b + "txt_attn.qkv.weight"] = torch.cat([aq, ak, av], dim=0)
        if (w := pop(p + "attn.norm_added_q.weight")) is not None:
            new_sd[b + "txt_attn.norm.query_norm.scale"] = w
        if (w := pop(p + "attn.norm_added_k.weight")) is not None:
            new_sd[b + "txt_attn.norm.key_norm.scale"] = w
        if (w := pop(p + "attn.to_add_out.weight")) is not None:
            new_sd[b + "txt_attn.proj.weight"] = w
        if (w := pop(p + "ff_context.linear_in.weight")) is not None:
            new_sd[b + "txt_mlp.0.weight"] = w
        if (w := pop(p + "ff_context.linear_out.weight")) is not None:
            new_sd[b + "txt_mlp.2.weight"] = w

    # --- Single stream blocks ---
    for i in range(depth_single_blocks):
        p = f"single_transformer_blocks.{i}."
        b = f"single_blocks.{i}."

        if (w := pop(p + "attn.to_qkv_mlp_proj.weight")) is not None:
            new_sd[b + "linear1.weight"] = w
        if (w := pop(p + "attn.to_out.weight")) is not None:
            new_sd[b + "linear2.weight"] = w
        if (w := pop(p + "attn.norm_q.weight")) is not None:
            new_sd[b + "norm.query_norm.scale"] = w
        if (w := pop(p + "attn.norm_k.weight")) is not None:
            new_sd[b + "norm.key_norm.scale"] = w

    # Anything left in `sd` wasn't recognized by this mapping -- keep it (already-
    # renamed keys were popped out above) so _load_state_dict_verbose's report still
    # surfaces it as "unexpected" instead of silently dropping it.
    new_sd.update(sd)
    return new_sd


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        weight_path = None
        if config["model_path"] in os.environ and os.path.exists(os.environ[config["model_path"]]):
            weight_path = os.environ[config["model_path"]]
        else:
            p_root = find_persistent_data_root()
            if p_root:
                candidates = [
                    os.path.join(p_root, config["filename"]),
                    os.path.join(p_root, "transformer", "diffusion_pytorch_model.safetensors"),
                ]
                for cp in candidates:
                    if os.path.exists(cp):
                        weight_path = cp
                        print(f"Found local FLUX.2 weights at: {cp}")
                        break

        if weight_path is None:
            # download from huggingface
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except Exception as e:
                print(
                    f"Failed to access model repository on HuggingFace and local file not found ({config['filename']}). "
                    f"Error: {e}. Please set environment variable {config['model_path']} to local file path."
                )
                sys.exit(1)

    if not debug_mode:
        params = FLUX2_MODEL_INFO[model_name.lower()]["params"]
        with torch.device("meta"):
            model = Flux2(params).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))

        # Auto-convert Diffusers-format DiT checkpoints to this module's BFL layout
        # (mirrors load_ae()'s existing Diffusers-VAE auto-conversion below).
        if any(
            k.startswith("x_embedder.") or k.startswith("context_embedder.") or k.startswith("transformer_blocks.")
            for k in sd.keys()
        ):
            print("  -> Detected Diffusers format DiT keys, converting to BFL Flux2 format...")
            sd = convert_diffusers_dit_to_bfl(sd, depth=params.depth, depth_single_blocks=params.depth_single_blocks)

        _load_state_dict_verbose(model, sd, label=f"DiT:{model_name}")
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_text_encoder(model_name: str, device: str | torch.device = "cuda"):
    config = FLUX2_MODEL_INFO[model_name.lower()]
    return config["text_encoder_load_fn"](device=device)


def convert_diffusers_vae_to_bfl(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("vae."):
            new_k = new_k[4:]

        # quant_conv / post_quant_conv
        if new_k.startswith("quant_conv."):
            new_k = "encoder." + new_k
        elif new_k.startswith("post_quant_conv."):
            new_k = "decoder." + new_k

        # conv_norm_out -> norm_out
        new_k = new_k.replace("conv_norm_out.", "norm_out.")

        # Encoder: down_blocks.{i} -> down.{i} (same order)
        if "encoder.down_blocks." in new_k:
            new_k = new_k.replace("encoder.down_blocks.", "encoder.down.")
            new_k = new_k.replace(".downsamplers.0.conv.", ".downsample.conv.")
            new_k = new_k.replace(".conv_shortcut.", ".nin_shortcut.")
            new_k = new_k.replace(".resnets.", ".block.")

        # Decoder: up_blocks.{i} -> up.{3 - i} (reversed order in BFL)
        elif "decoder.up_blocks." in new_k:
            parts = new_k.split(".")
            # parts example: ['decoder', 'up_blocks', '0', 'resnets', '0', ...]
            level_idx = int(parts[2])
            bfl_level = 3 - level_idx
            parts[1] = "up"
            parts[2] = str(bfl_level)
            new_k = ".".join(parts)

            new_k = new_k.replace(".upsamplers.0.conv.", ".upsample.conv.")
            new_k = new_k.replace(".conv_shortcut.", ".nin_shortcut.")
            new_k = new_k.replace(".resnets.", ".block.")

        # mid_block -> mid
        if ".mid_block." in new_k:
            new_k = new_k.replace(".mid_block.resnets.0.", ".mid.block_1.")
            new_k = new_k.replace(".mid_block.resnets.1.", ".mid.block_2.")
            new_k = new_k.replace(".mid_block.attentions.0.group_norm.", ".mid.attn_1.norm.")
            new_k = new_k.replace(".mid_block.attentions.0.to_q.", ".mid.attn_1.q.")
            new_k = new_k.replace(".mid_block.attentions.0.to_k.", ".mid.attn_1.k.")
            new_k = new_k.replace(".mid_block.attentions.0.to_v.", ".mid.attn_1.v.")
            new_k = new_k.replace(".mid_block.attentions.0.to_out.0.", ".mid.attn_1.proj_out.")

        # Shape adjustment: Diffusers attention Linear (512, 512) -> BFL Conv2d (512, 512, 1, 1)
        if any(att in new_k for att in [".attn_1.q.weight", ".attn_1.k.weight", ".attn_1.v.weight", ".attn_1.proj_out.weight"]):
            if len(v.shape) == 2:
                v = v.view(v.shape[0], v.shape[1], 1, 1)

        new_sd[new_k] = v
    return new_sd


def load_ae(model_name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]
    weight_path = None

    if "AE_MODEL_PATH" in os.environ and os.path.exists(os.environ["AE_MODEL_PATH"]):
        weight_path = os.environ["AE_MODEL_PATH"]
    else:
        p_root = find_persistent_data_root()
        if p_root:
            candidates = [
                os.path.join(p_root, "vae", "diffusion_pytorch_model.safetensors"),
                os.path.join(p_root, config["filename_ae"]),
                os.path.join(p_root, "vae", config["filename_ae"]),
            ]
            for cp in candidates:
                if os.path.exists(cp):
                    weight_path = cp
                    print(f"Found local AutoEncoder weights at: {cp}")
                    break

    if weight_path is None:
        # download from huggingface
        try:
            ae_repo = config.get("ae_repo_id", config["repo_id"])
            weight_path = huggingface_hub.hf_hub_download(
                repo_id=ae_repo,
                filename=config["filename_ae"],
                repo_type="model",
            )
        except Exception as e:
            print(
                f"Failed to access AE repository on HuggingFace and local file not found ({config['filename_ae']}). "
                f"Error: {e}. Please set environment variable AE_MODEL_PATH to local file path."
            )
            sys.exit(1)

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    
    # Auto-convert diffusers format to BFL format if detected
    if any(k.startswith("encoder.down_blocks.") or k.startswith("quant_conv.") or k.startswith("decoder.up_blocks.") for k in sd.keys()):
        print("  -> Detected Diffusers format VAE keys, converting to BFL AutoEncoder format...")
        sd = convert_diffusers_vae_to_bfl(sd)
    elif any(k.startswith("vae.") for k in sd.keys()):
        sd = {k.replace("vae.", ""): v for k, v in sd.items()}

    _load_state_dict_verbose(ae, sd, label=f"AutoEncoder:{model_name}")

    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
