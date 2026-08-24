# HƯỚNG DẪN HOẠT ĐỘNG DÀNH CHO AGENT (AGENT INSTRUCTIONS & ENVIRONMENT CONSTRAINTS)

## 📌 1. MÔI TRƯỜNG PHÁT TRIỂN & PHÂN VÙNG THỰC THI (ENVIRONMENT PARTITIONING)

### 💻 A. MÁY CÁ NHÂN CỦA USER (LOCAL DEVELOPMENT MACHINE - WINDOWS):
- **Phần cứng**: Máy cá nhân, **KHÔNG CÓ GPU / KHÔNG ĐỦ TÀI NGUYÊN** để chạy mô hình AI nặng.
- **Vai trò**: Dùng DUY NHẤT để:
  + Đọc hiểu, phân tích mã nguồn, tài liệu và paper.
  + Viết code, debug cú pháp, đóng gói module, viết scripts kiểm thử/huấn luyện.
  + Quản lý phiên bản mã nguồn (Git commit, git push, tạo file ZIP).
- **⚠️ ĐIỀU CẤM KỴ**: **TUYỆT ĐỐI KHÔNG CHẠY** các lệnh inference mô hình nặng (`cli.py`, tải checkpoint DiT 4B, train LoRA, load weights lớn) trên máy local này!

---

### 🚀 B. MÁY CHỦ THỰC THI (REMOTE COMPUTING SERVER - 2x NVIDIA A30 48GB):
- **Phần cứng**: **2x NVIDIA A30 (24GB VRAM x 2 = 48GB VRAM)**.
- **Môi trường kết nối**: Máy chủ nằm trong **MẠNG NỘI BỘ (Internal Network)**, truy cập thông qua **JupyterLab**.
- **Cấu trúc thư mục máy chủ thực tế (Source of Truth)**:
  ```
  ~/ (thư mục gốc JupyterLab: /home/jovyan/)
  ├── persistent-data/
  │   └── FLUX.2-klein-base-4B/
  │       ├── flux-2-klein-base-4b.safetensors   (7.3GB - Weights DiT chuẩn BFL)
  │       ├── text_encoder/                      (Weights Qwen3-4B-FP8)
  │       │   ├── config.json
  │       │   ├── model-00001-of-00002.safetensors
  │       │   ├── model-00002-of-00002.safetensors
  │       │   └── model.safetensors.index.json
  │       ├── tokenizer/                         (Tokenizer của Qwen3)
  │       │   ├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt...
  │       ├── vae/
  │       │   └── diffusion_pytorch_model.safetensors (161MB - VAE)
  │       └── transformer/
  │           └── diffusion_pytorch_model.safetensors (7.3GB - Diffusers DiT)
  └── work/                                      <-- Nơi clone repo Tendoo AI (ngang hàng)
  ```
- **Kênh truyền tải mã nguồn**:
  + Qua **GitHub Repository** (Git push từ local -> Git pull trong thư mục `work/` trên server).
  + Hoặc đóng gói tệp **ZIP** / copy paste trực tiếp mã nguồn vào JupyterLab.
- **Vai trò**:
  + Chạy các thực nghiệm suy luận (Inference Gate tests, RoPE Binding).
  + Chạy huấn luyện (Fine-tune VAE Decoder, LoRA DiT 4B Base).
  + Xuất log, ảnh kết quả và lưu checkpoint. Kết quả sau đó được chuyển về máy local để đánh giá.

---

## 🎯 2. QUY TRÌNH LÀM VIỆC CHUẨN (STANDARD AGENT WORKFLOW)

```
[ BƯỚC 1: LOCAL AGENT ]
  Viết code hoàn chỉnh, độc lập, có tài liệu hướng dẫn và script tự động.
         │
         ▼
[ BƯỚC 2: ĐÓNG GÓI & ĐẨY CODE ]
  Commit Git / Hướng dẫn lệnh Git hoặc tạo file ZIP để User chuyển sang JupyterLab.
         │
         ▼
[ BƯỚC 3: USER CHẠY TRÊN SERVER ]
  User chạy script trên 2x GPU A30 qua JupyterLab Terminal / Notebook.
         │
         ▼
[ BƯỚC 4: NHẬN KẾT QUẢ VỀ LOCAL ]
  User copy log / ảnh kết quả về máy local -> Agent đọc log và phân tích phản biện tiếp.
```

## 📋 3. NGUYÊN TẮC GIAO TIẾP VÀ KỸ THUẬT (COMMUNICATION & TECHNICAL PRINCIPLES)

1. **Phong cách đồng nghiệp phản biện**:
   - Khách quan, trung thực 100%, không nịnh bợ, không lạc quan tếu.
   - Sẵn sàng chỉ ra lỗ hổng toán học, rủi ro bộ nhớ và sai số kiến trúc.
2. **Nguyên tắc Kiềng 3 Chân (3-Pillar Complementary Rule - Chuẩn v3)**:
   - In-Context Time-Offset Conditioning ($t=10, 20...$ tại tọa độ $(0, 0)$) (giải quyết Đa khối Text & Định vị tự nhiên).
   - Tight Crop Bitmap (giải quyết Chi phí Sequence Length, tiết kiệm $>80\%$ tokens).
   - LoRA DiT 4B Base (giải quyết Chất liệu & Ánh sáng trên vật liệu phức tạp).
   - Ba giải pháp này bổ trợ cho nhau, không thay thế nhau.
3. **Mã nguồn thực thi**: Mọi script viết ra để chạy trên Server phải tự chứa (self-contained), có xử lý exception, hỗ trợ GPU CUDA, và tối ưu cho cấu hình 2x GPU A30 (Ampere architecture, BF16/FP16, DDP).

---

## 🎯 4. MÔ HÌNH MỤC TIÊU DUY NHẤT CỦA DỰ ÁN (SOLE TARGET MODEL)

Dự án này **CHỈ TẬP TRUNG DUY NHẤT VÀO MÔ HÌNH**:
👉 **`FLUX.2-klein-base-4B`** (Không dùng bản Distilled 4-step, không dùng 9B, không dùng Dev 32B).

- **Kiến trúc DiT**: `Klein4BParams` (5 DoubleStreamBlocks, 20 SingleStreamBlocks, $d_{\text{model}} = 3072$, 24 attention heads, 4D RoPE `axes_dim = [32, 32, 32, 32]`, $\theta = 2000$).
- **Text Encoder**: `Qwen3-4B-FP8` (Trích xuất 3 tầng cố định: `[9, 18, 27]` $\rightarrow$ Context dimension = $2560 \times 3 = \mathbf{7680}$).
- **AutoEncoder (VAE)**: 128 latent channels, nén $16\times$ không gian.
- **Quy chuẩn suy luận (Inference Defaults)**:
  + Euler ODE Flow Matching: `num_steps = 50`.
  + Classifier-Free Guidance (CFG): `guidance = 4.0` (Do đây là bản Base, bắt buộc có CFG để chất lượng đạt đỉnh).
- **Mọi code, script, LoRA fine-tuning và tối ưu hóa từ nay về sau**: Đều viết và cấu hình riêng cho `FLUX.2-klein-base-4B`.

---

## ⛔ 5. BÀI HỌC KỸ THUẬT BẮT BUỘC GHI NHỚ (LESSONS LEARNED)

1. **ĐÓNG BĂNG UPSTREAM CORE (`src/flux2/`) — TUYỆT ĐỐI HẠN CHẾ SỬA CODE GỐC**:
   - **Nguyên nhân**: Can thiệp trực tiếp vào mã nguồn gốc của BFL (`model.py`, `sampling.py`, `autoencoder.py`) dễ gây lỗi hồi quy ngầm (silent regressions), phá vỡ các giả định toán học của mô hình và làm mất mốc đối chứng (Ground Truth) khi debug.
   - **Quy tắc**: Giữ nguyên vẹn 100% các file gốc của BFL. Mọi logic mở rộng của Tendoo AI (RoPE Spatial Binding, Glyph Rendering, LoRA Training Pipelines, Custom Wrappers) phải được phát triển ở tầng riêng bên ngoài (`scripts/`, `src/tendoo/`), chỉ gọi các API chuẩn của BFL (`model.forward()`, `ae.encode()`, `ae.decode()`).

2. **TUYỆT ĐỐI KHÔNG DÙNG KV-CACHING CHO DiT BASE 4B (`forward_kv_cached`)**:
   - **Nguyên nhân**: Cơ chế KV-caching đóng băng Key/Value của Reference token tại Step 0 ($t=1.0$ khi canvas là 100% nhiễu hạt), làm mất sự tương tác thích ứng động giữa nét chữ và canvas qua 50 bước ODE, khiến chữ bị biến thành ký tự rác.
   - **Quy tắc**: Luôn dùng `denoise_cfg` full 50 bước tương tác liên tục `[Canvas, Ref]` qua `model.forward()`. Với $L_{\text{ref}} \le 256$ tokens (Tight Crop), tốc độ chạy trên 2x A30 hoàn toàn đảm bảo mà chất lượng vẽ chữ đạt đỉnh.


