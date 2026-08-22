# ENVIRONMENT AND WORKFLOW RULES FOR ANTIGRAVITY AGENT

## 1. HARDWARE & ENVIRONMENT CONSTRAINTS
- **Local Machine (Windows PC)**: Low compute resources, NO GPU. 
  - ONLY use for: code editing, script writing, architecture inspection, git versioning.
  - NEVER execute heavy model inference / training scripts locally.
- **Remote Server (Internal Network / JupyterLab)**:
  - 2x NVIDIA A30 (24GB VRAM each = 48GB VRAM).
  - Directory layout on server:
    ```
    /home/jovyan/
    ├── persistent-data/
    │   └── FLUX.2-klein-base-4B/
    │       ├── flux-2-klein-base-4b.safetensors   (7.3GB - DiT)
    │       ├── text_encoder/                      (Qwen3-4B-FP8 weights)
    │       ├── tokenizer/                         (Tokenizer files)
    │       ├── vae/diffusion_pytorch_model.safetensors (161MB - VAE)
    │       └── transformer/
    └── work/                                      (Cloned repo workspace)
    ```
  - Isolated network: code is deployed via GitHub repo (push/pull), zip archives, or copy-paste into JupyterLab.
  - Executes all model runs, inference experiments, VAE fine-tuning, and LoRA training.

## 2. CODE DELIVERY STANDARDS
- All scripts meant for remote execution must be self-contained, well-commented, support CUDA/BF16, and provide clear CLI arguments.
- When finishing a development phase, provide clear instructions for the user on which files to pull/copy to JupyterLab and the exact command to run on the server.

## 3. TECHNICAL TRUTH & PEER REVIEW
- Maintain rigorous, honest, and objective technical peer review.
- Strictly adhere to the 3-Pillar complementary design (RoPE Spatial Binding + Tight Crop Glyph + DiT LoRA).

## 4. SOLE TARGET MODEL: FLUX.2-klein-base-4B
- All development, LoRA training, and inference scripts strictly target **FLUX.2-klein-base-4B**:
  - DiT: `Klein4BParams` (5 DoubleBlocks, 20 SingleBlocks, hidden_size=3072, num_heads=24, axes_dim=[32,32,32,32], theta=2000).
  - Text Encoder: `Qwen3-4B-FP8` (Layers [9, 18, 27] -> 7680 dim).
  - VAE: 128 channels, 16x downsampling.
  - Inference: 50 steps Euler ODE, CFG guidance = 4.0.
  - No 9B, No 32B, No 4-step distilled models.
