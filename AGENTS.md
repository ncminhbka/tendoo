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

3. **KHÔNG LẶP LẠI NGUYÊN VĂN NỘI DUNG CHỮ TRONG TEXT PROMPT (REPRESENTATION CLASH)**:
   - **Nguyên nhân**: Khi đưa nguyên văn chuỗi text tiếng Việt vào Prompt, `Qwen3-4B-FP8` cố gắng tự sinh chữ từ kiến thức tiền huấn luyện (vốn yếu về dấu tiếng Việt $\rightarrow$ sinh ra chữ lỗi). Tín hiệu lỗi này xung đột trực tiếp với tín hiệu In-Context Glyph Bitmap chuẩn từ VAE, khiến DiT bị "phân tâm" và làm vỡ nát nét chữ.
   - **Quy tắc**: Trong Text Prompt, **TUYỆT ĐỐI KHÔNG LẶP LẠI NỘI DUNG CHỮ CẦN VẼ**. Chỉ mô tả vai trò (`tiêu đề`, `slogan`), vị trí (`ở trên`, `ở dưới`) và chất liệu/màu sắc (`đèn neon xanh`, `chữ vàng dập nổi`). Hãy để In-Context Glyph đảm nhiệm $100\%$ nội dung chính tả.

4. **NGƯỠNG PHÂN GIẢI LATENT TỐI THIỂU CHO GLYPH BITMAP ($\ge 10$ TOKENS HEIGHT)**:
   - **Nguyên nhân**: VAE nén $16\times$. Các câu dài nhiều từ nếu bị ép vào box có chiều cao $< 128\text{px}$ ($< 8$ latent tokens) sẽ khiến các dấu phụ (`Á`, `Ệ`, `Ộ`) bị thu nhỏ dưới 1 pixel, gây nghẽn cổ chai giải mã ở VAE Decoder.
   - **Quy tắc**: Kích thước chiều cao Box của Glyph phải đạt tối thiểu $160\text{px}$ ($\ge 10$ latent tokens). Với slogan dài $\ge 4$ từ, tự động tăng chiều cao lên $192\text{px}$ ($12$ latent tokens) để đảm bảo độ sắc nét $100\%$.

5. **BẮT BUỘC PHÂN TÁCH TIME OFFSET ($t=10, 20...$) KÈM ĐỊNH HƯỚNG BỀ MẶT CHO ĐA THỰC THỂ (MULTI-ENTITY DISAMBIGUATION RULE)**:
   - **Nguyên nhân**:
     - Nếu gộp các khối text độc lập về cùng mốc $t=10.0$ tại $(0, 0)$, DiT thấy tọa độ không-thời gian trùng lặp $\rightarrow$ Trộn lẫn từ ngữ của 2 câu thành chuỗi lai tạp và in đè lên nhau.
     - Nếu tách mốc $t=10.0$ và $t=20.0$ nhưng KHÔNG có 2 bề mặt vật thể rõ ràng trong Prompt $\rightarrow$ Khối $t=20.0$ bị suy hao và biến mất.
   - **Quy tắc bắt buộc**:
     - Mỗi thực thể độc lập (Sản phẩm, Tiêu đề, Slogan) **BẮT BUỘC mang Time Offset riêng**: Thực thể 1 ở $t=10.0$, Thực thể 2 ở $t=20.0$, Thực thể 3 ở $t=30.0$.
     - Prompt **BẮT BUỘC định hình 2 bề mặt vật thể riêng biệt** tương ứng (ví dụ: *biển hiệu trên cao* cho $t=10.0$ và *màn hình đế bục/thân sản phẩm* cho $t=20.0$).
     - Đảm bảo từng Glyph đạt độ phân giải $\ge 10-12$ latent tokens height ($160-192\text{px}$) để tín hiệu ở $t=20.0$ sắc nét $100\%$.

6. **TUYỆT ĐỐI KHÔNG ĐƯA TỈ LỆ KHUNG HÌNH VÀ THÔNG SỐ ĐỘ PHÂN GIẢI VÀO PROMPT (DIMENSIONS / RESOLUTION POLLUTION)**:
   - **Nguyên nhân**: Khi đưa các chuỗi như `"9:16"`, `"16:9"`, `"8k"`, `"4k"`, `"1080p"` vào Prompt, Text Encoder `Qwen3` hiểu nhầm đây là chuỗi ký tự cần hiển thị trên sản phẩm/ảnh $\rightarrow$ DiT sẽ tự động vẽ các cụm số và chữ rác (như `3:16 9 168`, `8K0...`) lên các bề mặt hoặc góc đáy của bức ảnh.
   - **Quy tắc bắt buộc**: **KHÔNG BAO GIỜ GHI CÁC THÔNG SỐ KÍCH THƯỚC/TỈ LỆ VÀO TEXT PROMPT**. Tỉ lệ khung hình và kích thước chỉ được khai báo duy nhất qua các tham số CLI `--width` và `--height`.

7. **BẮT BUỘC TUÂN THỦ CÁC MỐC TIME OFFSET TIỀN HUẤN LUYỆN ($t \in \{10.0, 20.0, 30.0\}$) (CANONICAL PRETRAINED OFFSETS RULE)**:
   - **Nguyên nhân**: BFL tiền huấn luyện FLUX.2 độc quyền với các mốc thời gian rời rạc $t = 10 \times k$ ($10.0, 20.0, 30.0$). Trọng số của các Attention Heads đã được tối ưu sâu để nhận diện $t=10.0$ là kênh tham chiếu chính. Khi đưa vào các mốc $t < 10.0$ (như $t=5.0$), mô hình hoàn toàn không có biểu diễn tiền huấn luyện (Out-of-Distribution) nên sẽ bỏ qua $t=5.0$ và dồn toàn bộ sự chú ý vào $t=10.0$.
   - **Quy tắc bắt buộc & Định hướng Huấn luyện LoRA (Giai đoạn 3)**:
     - Cả trong suy luận (Inference) và huấn luyện LoRA (Training): **TUYỆT ĐỐI KHÔNG DÙNG $t < 10.0$**.
     - Bắt buộc chuẩn hóa 3 kênh thời gian:
       + **Kênh 1 ($t = 10.0$)**: Ảnh Sản phẩm chính HOẶC Tiêu đề chính (Title).
       + **Kênh 2 ($t = 20.0$)**: Slogan phụ HOẶC Biển hiệu 2.
       + **Kênh 3 ($t = 30.0$)**: Logo / Tem nhãn thương hiệu.
     - **Mục tiêu của LoRA DiT 4B (Giai đoạn 3)**: Tái cân bằng ma trận Attention Query/Key trên đúng các mốc $10.0, 20.0, 30.0$ này để biến kênh $t=20.0$ và $t=30.0$ thành vững chắc $100\%$ mà không phụ thuộc vào câu từ của Prompt.

8. **KHỐI LƯỢNG TOKEN & TIỀM NĂNG TIẾP NHẬN ĐA PHƯƠNG THỨC Ở CÁC MỐC THỜI GIAN XA (TOKEN MASS & MULTI-MODAL TIME HORIZON RULE)**:
   - **Phát hiện thực nghiệm đối chứng (`exp44` vs `exp45`)**:
     + Đặt **Glyph chữ thưa thớt ($\sim 320$ tokens)** tại $t=60.0$ $\rightarrow$ Chữ bị mất hoàn toàn do suy hao góc quay RoPE và thiếu khối lượng token kích hoạt.
     + Đặt **Ảnh Sản phẩm thật tự nhiên ($4096$ tokens)** tại $t=60.0$ $\rightarrow$ Sản phẩm trong ảnh sinh ra được giữ **Y HỆT $100\%$ so với ảnh thật**!
   - **Ý nghĩa sống còn đối với Giai đoạn 3 (Huấn luyện LoRA)**:
     + Kiến trúc 4D RoPE của FLUX.2 **hoàn toàn có năng lực truyền tải và bảo toàn thông tin ở các mốc thời gian xa ($t \ge 30, 60$)**.
     + Lý do chữ bị yếu ở mốc $t=20, 30$ trên mô hình gốc chỉ vì BFL chưa từng huấn luyện DiT đọc Glyph chữ đen trắng ở các mốc này.
     + Khi huấn luyện LoRA ở Giai đoạn 3, LoRA sẽ kích hoạt phản xạ chú ý cho Glyph chữ, giúp năng lực vẽ chữ ở các mốc $t=20, 30$ đạt độ chính xác $100\%$ vững chắc ngang ngửa khả năng giữ ảnh sản phẩm thật!

9. **BẢN CHẤT GỐC RỄ NẰM Ở MA TRẬN ATTENTION ROPE, KHÔNG PHẢI DO CÂU TỪ PROMPT (INDEPENDENT ATTENTION ROUTING RULE)**:
   - **Phân tích bản chất**: Việc mô hình Base zero-shot cần Prompt "gợi mở 2 bề mặt" thực chất chỉ là một "chiếc nạng cứu trợ tạm thời" để bù đắp cho tín hiệu Attention bị suy hao ở $t=20.0$. Thực nghiệm `exp45` (giữ nguyên $100\%$ sản phẩm thật ở $t=60.0$ mà không cần prompt chi tiết) đã chứng minh gốc rễ hoàn toàn nằm ở trọng số Attention $W_Q, W_K$ của DiT cho loại token đó.
   - **Mục tiêu giải phóng của LoRA (Giai đoạn 3)**: Huấn luyện LoRA tối ưu hóa ma trận Attention cho Glyph chữ ở $t=20.0, 30.0$, giải phóng người dùng khỏi việc phải "viết prompt văn mẫu", cho phép prompt hoàn toàn tự nhiên, ngắn gọn mà mô hình vẫn tự động định vị và vẽ đúng $100\%$ đa khối text.








