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
- **Nguyên tắc Kiềng 3 chân (Chuẩn v3)**: Tuân thủ nghiêm ngặt mô hình bổ trợ 3 chân (In-Context Time-Offset Conditioning $t=10, 20...$ + Tight Crop Glyph + DiT LoRA).

## 4. SOLE TARGET MODEL: FLUX.2-klein-base-4B
- All development, LoRA training, and inference scripts strictly target **FLUX.2-klein-base-4B**:
  - DiT: `Klein4BParams` (5 DoubleBlocks, 20 SingleBlocks, hidden_size=3072, num_heads=24, axes_dim=[32,32,32,32], theta=2000).
  - Text Encoder: `Qwen3-4B-FP8` (Layers [9, 18, 27] -> 7680 dim).
  - VAE: 128 channels, 16x downsampling.
  - Inference: 50 steps Euler ODE, CFG guidance = 4.0.
  - No 9B, No 32B, No 4-step distilled models.

## 5. LESSONS LEARNED & ARCHITECTURAL PITFALLS (BÀI HỌC KỸ THUẬT BẮT BUỘC GHI NHỚ)
- ⛔ **ĐÓNG BĂNG UPSTREAM CORE (`src/flux2/`) — TUYỆT ĐỐI HẠN CHẾ SỬA CODE GỐC**:
  - **Nguyên nhân**: Can thiệp trực tiếp vào mã nguồn gốc của BFL (`model.py`, `sampling.py`, `autoencoder.py`) dễ gây lỗi hồi quy ngầm (silent regressions), phá vỡ các giả định toán học của mô hình và làm mất mốc đối chứng (Ground Truth) khi debug.
  - **Quy tắc bắt buộc**: Giữ nguyên vẹn 100% các file gốc của BFL. Mọi logic mở rộng của Tendoo AI (RoPE Spatial Binding, Glyph Rendering, LoRA Training Pipelines, Custom Wrappers) phải được phát triển ở tầng riêng bên ngoài (`scripts/`, `src/tendoo/`), chỉ gọi các API chuẩn của BFL (`model.forward()`, `ae.encode()`, `ae.decode()`).
- ⛔ **TUYỆT ĐỐI KHÔNG DÙNG KV-CACHING CHO DiT BASE 4B (`forward_kv_cached`)**:
  - **Nguyên nhân**: Cơ chế KV-caching của BFL chỉ thiết kế cho mô hình Distilled 4 bước (9B-KV). Đối với Base 4B (50 bước ODE + CFG 4.0), việc trích xuất và đóng băng KV của Reference token tại Step 0 ($t=1.0$ khi canvas là 100% nhiễu hạt) sẽ cắt đứt sự tương tác động giữa nét chữ và canvas qua các timestep, khiến chữ bị biến thành ký tự rác.
  - **Quy tắc bắt buộc**: Luôn dùng `denoise_cfg` full 50 bước tương tác liên tục `[Canvas, Ref]` qua `model.forward()`. Với $L_{\text{ref}} \le 256$ tokens (Tight Crop), tốc độ chạy trên 2x A30 vẫn đạt yêu cầu mà chất lượng chữ đạt đỉnh.
- ⛔ **KHÔNG LẶP LẠI NGUYÊN VĂN NỘI DUNG CHỮ TRONG TEXT PROMPT (REPRESENTATION CLASH)**:
  - **Nguyên nhân**: Khi đưa nguyên văn chuỗi text tiếng Việt vào Prompt, `Qwen3-4B-FP8` cố gắng tự sinh chữ từ kiến thức tiền huấn luyện (vốn yếu về dấu tiếng Việt $\rightarrow$ sinh ra chữ lỗi). Tín hiệu lỗi này xung đột trực tiếp với tín hiệu In-Context Glyph Bitmap chuẩn từ VAE, khiến DiT bị "phân tâm" và làm vỡ nát nét chữ.
  - **Quy tắc bắt buộc**: Trong Text Prompt, **TUYỆT ĐỐI KHÔNG LẶP LẠI NỘI DUNG CHỮ CẦN VẼ**. Chỉ mô tả vai trò (`tiêu đề`, `slogan`), vị trí (`ở trên`, `ở dưới`) và chất liệu/màu sắc (`đèn neon xanh`, `chữ vàng dập nổi`). Hãy để In-Context Glyph đảm nhiệm $100\%$ nội dung chính tả.
- ⛔ **NGƯỠNG PHÂN GIẢI LATENT TỐI THIỂU CHO GLYPH BITMAP ($\ge 10$ TOKENS HEIGHT)**:
  - **Nguyên nhân**: VAE nén $16\times$. Các câu dài nhiều từ nếu bị ép vào box có chiều cao $< 128\text{px}$ ($< 8$ latent tokens) sẽ khiến các dấu phụ (`Á`, `Ệ`, `Ộ`) bị thu nhỏ dưới 1 pixel, gây nghẽn cổ chai giải mã ở VAE Decoder.
  - **Quy tắc bắt buộc**: Kích thước chiều cao Box của Glyph phải đạt tối thiểu $160\text{px}$ ($\ge 10$ latent tokens). Với slogan dài $\ge 4$ từ, tự động tăng chiều cao lên $192\text{px}$ ($12$ latent tokens) để đảm bảo độ sắc nét $100\%$.
- ⛔ **BẮT BUỘC PHÂN TÁCH TIME OFFSET ($t=10, 20...$) KÈM ĐỊNH HƯỚNG BỀ MẶT CHO ĐA THỰC THỂ (MULTI-ENTITY DISAMBIGUATION RULE)**:
  - **Nguyên nhân**: Gộp các khối text độc lập về cùng $t=10.0$ tại $(0, 0)$ sẽ làm DiT trộn lẫn từ ngữ và in đè lên nhau. Tách $t=10.0, 20.0$ nhưng thiếu 2 bề mặt vật thể trong Prompt sẽ làm mất khối $t=20.0$.
  - **Quy tắc bắt buộc**: Gán Time Offset riêng cho từng thực thể (Ref 1: $t=10.0$, Ref 2: $t=20.0$, Ref 3: $t=30.0$) + Định hình rõ 2 bề mặt vật thể tương ứng trong Prompt + Đảm bảo Glyph height $\ge 160-192\text{px}$.
- ⛔ **TUYỆT ĐỐI KHÔNG ĐƯA TỈ LỆ KHUNG HÌNH VÀ THÔNG SỐ ĐỘ PHÂN GIẢI VÀO PROMPT (DIMENSIONS / RESOLUTION POLLUTION)**:
  - **Nguyên nhân**: Đưa `"9:16"`, `"16:9"`, `"8k"`, `"4k"` vào Prompt khiến Text Encoder Qwen3 kích hoạt DiT vẽ chuỗi số/chữ rác lên các bề mặt hoặc đáy ảnh.
  - **Quy tắc bắt buộc**: Không bao giờ ghi thông số kích thước/tỉ lệ vào Prompt. Chỉ khai báo qua tham số CLI `--width` và `--height`.
- ⛔ **BẮT BUỘC TUÂN THỦ CÁC MỐC TIME OFFSET TIỀN HUẤN LUYỆN ($t \in \{10.0, 20.0, 30.0\}$)**:
  - **Nguyên nhân**: BFL pretrain FLUX.2 độc quyền với $t = 10 \times k$. Các mốc $t < 10.0$ (như $t=5.0$) là Out-of-Distribution, khiến mô hình bỏ qua $t=5.0$ và dồn toàn bộ attention vào $t=10.0$.
  - **Quy tắc bắt buộc**: Luôn dùng $t=10.0$ cho Ref 1 (Sản phẩm/Title), $t=20.0$ cho Ref 2 (Slogan/Biển 2), $t=30.0$ cho Ref 3 (Logo). LoRA (Giai đoạn 3) sẽ học ma trận $\Delta W$ dựa trên đúng các mốc này để tái cân bằng attention.
- ⛔ **KHỐI LƯỢNG TOKEN & NĂNG LỰC TIẾP NHẬN ĐA PHƯƠNG THỨC Ở MỐC THỜI GIAN XA**:
  - **Phát hiện**: Ảnh sản phẩm $4096$ tokens ở $t=60.0$ giữ y hệt $100\%$ do mật độ đặc trưng dày đặc, trong khi Glyph thưa thớt $\sim 320$ tokens bị mất. Điều này chứng minh kiến trúc 4D RoPE hoàn toàn có thể truyền tải thông tin ở mốc xa, và LoRA Giai đoạn 3 sẽ kích hoạt năng lực này cho Glyph chữ tiếng Việt!







