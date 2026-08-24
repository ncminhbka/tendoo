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

## 3. TECHNICAL TRUTH, NO FLATTERY & CRITICAL THINKING
- **Tuyệt đối không nịnh bợ (Zero Flattery)**: Không dùng các lời khen sáo rỗng (như "nhận định xuất sắc", "câu hỏi tuyệt vời", "rất sắc bén"...). Đi thẳng vào cốt lõi kỹ thuật.
- **Trung thực, thẳng thắn 100% (Honest & Candid)**: Đóng vai trò đồng nghiệp phản biện độc lập; thẳng thắn chỉ ra các lỗ hổng toán học, rủi ro bộ nhớ, điểm bất hợp lý trong logic hay hạn chế thực tế của mô hình.
- **Suy nghĩ sâu sắc & cẩn trọng (Deep & Thorough Thinking)**: Phân tích kỹ cơ chế hoạt động thực tế, kiến trúc DiT/Transformer và luồng dữ liệu trước khi phát ngôn hoặc viết code; tuyệt đối không phán đoán hời hợt, không giả định thiếu căn cứ.
- **Nguyên tắc Kiềng 3 chân**: Tuân thủ nghiêm ngặt mô hình bổ trợ 3 chân (RoPE Spatial Binding + Tight Crop Glyph + DiT LoRA).

## 4. SOLE TARGET MODEL: FLUX.2-klein-base-4B
- All development, LoRA training, and inference scripts strictly target **FLUX.2-klein-base-4B**:
  - DiT: `Klein4BParams` (5 DoubleBlocks, 20 SingleBlocks, hidden_size=3072, num_heads=24, axes_dim=[32,32,32,32], theta=2000).
  - Text Encoder: `Qwen3-4B-FP8` (Layers [9, 18, 27] -> 7680 dim).
  - VAE: 128 channels, 16x downsampling.
  - Inference: 50 steps Euler ODE, CFG guidance = 4.0.
  - No 9B, No 32B, No 4-step distilled models.

## 5. LESSONS LEARNED & ARCHITECTURAL PITFALLS (BÀI HỌC KỸ THUẬT BẮT BUỘC GHI NHỚ)
- ⛔ **TUYỆT ĐỐI KHÔNG DÙNG KV-CACHING CHO DiT BASE 4B (`forward_kv_cached`)**:
  - **Nguyên nhân**: Cơ chế KV-caching của BFL chỉ thiết kế cho mô hình Distilled 4 bước (9B-KV). Đối với Base 4B (50 bước ODE + CFG 4.0), việc trích xuất và đóng băng KV của Reference token tại Step 0 ($t=1.0$ khi canvas là 100% nhiễu hạt) sẽ cắt đứt sự tương tác động giữa nét chữ và canvas qua các timestep, khiến chữ bị biến thành ký tự rác.
  - **Quy tắc bắt buộc**: Luôn dùng `denoise_cfg` full 50 bước tương tác liên tục `[Canvas, Ref]` qua `model.forward()`. Với $L_{\text{ref}} \le 256$ tokens (Tight Crop), tốc độ chạy trên 2x A30 vẫn đạt yêu cầu mà chất lượng chữ đạt đỉnh.

