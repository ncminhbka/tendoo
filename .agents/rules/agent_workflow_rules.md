# ENVIRONMENT AND WORKFLOW RULES FOR ANTIGRAVITY AGENT

## 1. HARDWARE & ENVIRONMENT CONSTRAINTS
- **Local Machine (Windows PC)**: Low compute resources, NO GPU. 
  - ONLY use for: code editing, script writing, architecture inspection, git versioning.
  - NEVER execute heavy model inference / training scripts locally.
- **Remote Server (Internal Network / JupyterLab)**:
  - 2x NVIDIA A30 (24GB VRAM each = 48GB VRAM).
  - Directory layout on server:
    - `/persistent-data/FLUX.2-klein-base-4B/` (Pre-downloaded model checkpoints)
    - `work/` (Cloned repo codebase, sibling to persistent-data)
  - Isolated network: code is deployed via GitHub repo (push/pull), zip archives, or copy-paste into JupyterLab.
  - Executes all model runs, inference experiments, VAE fine-tuning, and LoRA training.

## 2. CODE DELIVERY STANDARDS
- All scripts meant for remote execution must be self-contained, well-commented, support CUDA/BF16, and provide clear CLI arguments.
- When finishing a development phase, provide clear instructions for the user on which files to pull/copy to JupyterLab and the exact command to run on the server.

## 3. TECHNICAL TRUTH & PEER REVIEW
- Maintain rigorous, honest, and objective technical peer review.
- Strictly adhere to the 3-Pillar complementary design (RoPE Spatial Binding + Tight Crop Glyph + DiT LoRA).
