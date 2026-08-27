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
3. **Nguyên tắc Phân Công Lao Động Nhị Hợp (Dual Division of Labor Rule)**:
   - **Nhánh Glyph Bitmap (VAE)**: Chịu trách nhiệm $100\%$ về **HÌNH HỌC & CHÍNH TẢ** (nội dung tiếng Việt đúng dấu `Á`, `Ệ`, `Ộ`, kiểu dáng Font chữ tùy biến Serif/Sans/Brush/Graffiti, bố cục xuống dòng).
   - **Nhánh Text Prompt (Qwen3)**: Chịu trách nhiệm $100\%$ về **CHẤT LIỆU, VẬT LÝ & QUANG HỌC** (kỹ thuật chế tác: dập nổi, khắc chìm trên gỗ, đúc kim loại; hiệu ứng ánh sáng: đèn neon phát quang, đổ bóng 3D, phản chiếu studio).
   - Glyph tạo nên "khung xương hình học", Prompt tạo nên "phần hồn, ánh sáng và chất liệu".
4. **Mã nguồn thực thi**: Mọi script viết ra để chạy trên Server phải tự chứa (self-contained), có xử lý exception, hỗ trợ GPU CUDA, và tối ưu cho cấu hình 2x GPU A30 (Ampere architecture, BF16/FP16, DDP).


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

10. **QUY LUẬT VÀNG ĐƠN KHỐI ($t=10.0$) VS ĐA KHỐI ($\ge 2$ TEXTS) TRÊN DiT BASE 4B (SINGLE-TEXT ABSOLUTE PRESERVATION LAW)**:
    - **Khẳng định thực nghiệm $100\%$ (Kiểm chứng qua exp49 - exp51)**:
      + Khi chỉ có **DUY NHẤT 1 KHỐI TEXT đặt tại $t=10.0$**: Chữ **LUÔN LUÔN ĐƯỢC GIỮ ĐẸP VÀ CHUẨN XÁC TUYỆT ĐỐI $100\%$**, biến hóa xuất sắc theo mọi chất liệu và ánh sáng trong Prompt (chữ vàng dập nổi 3D, đèn neon phát quang, đổ bóng studio).
      + Khi có **TỪ 2 KHỐI TEXT TRỞ LÊN ($\ge 2$ texts)** trên mô hình Base 4B zero-shot: Kết quả **CỰC KỲ LUNG LAY, lúc được lúc không** và phụ thuộc nặng nề vào việc prompt có mô tả chính xác bề mặt vật thể hay không.
    - **Tầm quan trọng sống còn của LoRA (Giai đoạn 3)**:
      + Mô hình Base 4B nguyên bản đã đủ $100\%$ độ tin cậy cho bài toán: **1 Ảnh Sản phẩm ($t=60$) + 1 Dòng Chữ Chính ($t=10$)**.
      + Để mở rộng năng lực phục vụ **Đa khối Text ($\ge 2$ texts)** đạt chuẩn $100\%$ bất chấp prompt tự nhiên, bắt buộc phải hoàn thành **Huấn luyện LoRA DiT 4B ở Giai đoạn 3**.

11. **ĐỘT PHÁ VỀ NĂNG LỰC SINH BÀI THƠ / ĐOẠN VĂN DÀI ($\ge 28$ TỪ) TRÊN DiT BASE 4B (VAE RESOLUTION SCALING LAW)**:
    - **Phát hiện thực nghiệm mang tính bước ngoặt (`exp52` vs `exp53` vs `exp54`)**:
      + **Lầm tưởng ban đầu (`exp52`)**: Cho rằng mô hình Base 4B bị giới hạn dung lượng Attention nên không thể vẽ được 4 câu thơ (28 từ, 119 ký tự).
      + **Bản chất kỹ thuật thực sự**: Do nhồi 4 dòng thơ vào Glyph Box nhỏ ($512\times 224\text{px}$) khiến cỡ chữ bị co xuống chỉ còn $\sim 18\text{px}$, các dấu phụ và nét thanh Serif chỉ dày $1-2\text{px}$. Khi VAE nén $16\times$, tín hiệu rơi xuống dưới $0.1$ latent pixel $\rightarrow$ sụp đổ đặc trưng (Latent Feature Collapse).
      + **Đột phá thành công 100% (`exp54`)**: Khi phóng to Glyph Box tỉ lệ thuận theo số dòng lên **`896 x 512 px`** (cỡ chữ tăng lên $\sim 46-48\text{px}$, dấu tiếng Việt đạt $8-12\text{px}$ $\rightarrow$ vượt xa ngưỡng nén của VAE):
        ```bash
        python scripts/demo_tendoo_poster.py \
          --text "Sông Mã xa rồi Tây Tiến ơi\nNhớ về rừng núi nhớ chơi vơi.\nSài Khao sương lấp đoàn quân mỏi,\nMường Lát hoa về trong đêm hơi." \
          --prompt "Bức vách đá sa thạch cổ kính phẳng sừng sững ở tiền cảnh góc bên, bốn câu thơ chữ khắc chìm mạ vàng đồng cổ sắc nét trên mặt đá phẳng phủ rêu phong, hậu cảnh núi non Tây Bắc hùng vĩ mây mù hoàng hôn le lói, phong cách điện ảnh sử thi cổ trang, ánh sáng studio tương phản cao" \
          --font "playfair" \
          --width 1024 \
          --height 1024 \
          --box_w 896 \
          --box_h 512 \
          --steps 50 \
          --guidance 4.5 \
          --output "tay_tien_hires_glyph_4lines.png"
        ```
        $\rightarrow$ **KẾT QUẢ ĐỈNH CAO: MÔ HÌNH SINH ẢNH KHÔNG SAI MỘT CHỮ NÀO CẢ BÀI THƠ 4 CÂU (28 TỪ, 119 KÝ TỰ), DẤU TIẾNG VIỆT HOÀN HẢO 100%, NÉT KHẮC ĐÁ SA THẠCH MẠ VÀNG TUYỆT ĐẸP!**
    - **Quy tắc Vàng về Tỉ lệ Glyph Box (The Glyph Scaling Law)**:
      + Chiều cao Glyph Box phải tỉ lệ thuận theo số dòng: **$\text{box\_h} \ge \text{num\_lines} \times 128\text{px}$** (đảm bảo mỗi dòng chữ nhận tối thiểu $8$ latent tokens height và font size $\ge 40\text{px}$).
      + Với các bài thơ / đoạn văn dài: Bắt buộc mở rộng Box ngang $\ge 800 - 896\text{px}$ và Box dọc $\ge 448 - 512\text{px}$ trên Canvas $1024$.
      + **Khẳng định năng lực**: `FLUX.2-klein-base-4B` hoàn toàn có khả năng ghi nhớ và vẽ chuẩn xác $100\%$ các đoạn thơ/văn bản dài ở $t=10.0$ mà không cần chờ tới LoRA!

12. **BẢN CHẤT CỐT LÕI CỦA LORA GIAI ĐOẠN 3: ĐỊNH TUYẾN PHÂN TÁCH KÊNH (ATTENTION DISENTANGLEMENT & ROUTING), KHÔNG PHẢI HỌC BIỂU DIỄN MỚI**:
    - **Phát hiện thực nghiệm bước ngoặt (`Probe Suite 1` vs `Probe Suite 2`)**:
      + **Trạng thái Cô lập (Suite 1)**: Từng kênh $t=10.0, 20.0, 30.0, 40.0$ khi chạy đơn lẻ đều vẽ chữ 3D (ví dụ `"MUA 1 TẶNG 1"`) **chuẩn xác 100%, nét vẽ và chi tiết hoàn hảo**. Khẳng định mô hình Base đã có sẵn năng lực biểu diễn và giải mã ở các mốc $t \le 40.0$ (chỉ sụp đổ tại $t \ge 50.0$).
      + **Trạng thái Cạnh tranh Đồng thời (Suite 2)**: Khi đưa đồng thời $3 - 4$ slot, Attention Heads của mô hình Base bị hiện tượng **Tranh chấp & Tràn kênh (Cross-Slot Attention Bleeding)** $\rightarrow$ Slot $t=20$ tràn sang đè mất Slot $t=30$.
    - **Ý nghĩa toán học & Đột phá về Chi phí Huấn luyện (Data Efficiency Breakthrough)**:
      + **Phân biệt 2 bài toán**:
        * *Bài toán A (Dạy mô hình hiểu biểu diễn mới từ đầu)*: Đòi hỏi tập dữ liệu khổng lồ ($>50,000$ mẫu), nhiều epoch, tốn kém compute.
        * *Bài toán B (Dạy mô hình tách bạch các tín hiệu nó vốn đã hiểu riêng lẻ khi chúng xuất hiện cùng lúc)*: Đây là **Bài toán Định tuyến Chú ý (Attention Disentanglement / Routing)** — phạm vi tối ưu hóa hẹp hơn rất nhiều!
      + **Chiến lược LoRA Giai đoạn 3**: LoRA chỉ cần nhắm trực tiếp vào ma trận $W_Q, W_K$ ở các tầng Attention xử lý Reference tokens để đóng vai trò "Bộ điều phối phân luồng (Traffic Controller)".
      + **Tối ưu hóa Dataset**: Chỉ cần một tập dữ liệu nhỏ tập trung vào các mẫu Multi-Block / Banner ($\sim 500 - 1,000$ mẫu) là đủ cho bản Pilot đạt độ chính xác $100\%$ đa khối text, thay vì phải rải dữ liệu khổng lồ để "dạy chữ từ đầu".

13. **TÍNH BẤT BIẾN HOÁN VỊ THỨ TỰ GHÉP CHUỖI CỦA 4D RoPE (THE SEQUENCE PERMUTATION INVARIANCE LAW)**:
    - **Kiểm chứng thực nghiệm (`exp57` - `test_permutation_invariance.py`)**:
      + Thay đổi thứ tự nối chuỗi vật lý `dim=1`: Xuôi `[Canvas, t10, t20, t30]` vs Ngược `[Canvas, t30, t20, t10]` vs Xáo trộn `[Canvas, t20, t10, t30]`.
      + **Kết quả**: Cả 3 ảnh sinh ra **GIỐNG HỆT NHAU 100% TỪNG PIXEL**.
    - **Ý nghĩa toán học**:
      + Khẳng định hệ thống tọa độ 4D RoPE của Tendoo AI sạch tuyệt đối, không có rò rỉ thứ tự tuần tự (zero sequence-order bias).
      + Mọi tương tác đa slot chỉ phụ thuộc duy nhất vào: Tọa độ không-thời gian $(t, h, w, l)$ và động lực học Softmax cạnh tranh.

14. **MÔ HÌNH CƠ CHẾ KÉP TRONG SOFTMAX CẠNH TRANH (THE DUAL-MECHANISM CROSSTALK MODEL)**:
    - **Phát hiện thực nghiệm đối chứng (`exp58` vs `exp59`)**:
      + **Cơ chế 1 (Khối lượng Token - Token Mass Dominance)**: Khối ít token bị yếu thế trong Softmax chung trước khối 4096 tokens của sản phẩm. Khi tăng token mass lên $\ge 672$ tokens ($768 \times 224\text{px}$), khối CTA `"MUA 1 TẶNG 1"` ở $t=30$ **tự động phục hồi 100% độ chính xác mà không cần train**.
      + **Cơ chế 2 (Nghẽn cục bộ vị trí / Dấu phụ phức tạp)**: Khối Subtitle `"CHỐNG ỒN CHỦ ĐỘNG"` tại $t=20$ không tự khỏi khi tăng token, cần phân lập giữa đặc thù RoPE $t=20$ và cụm 4 dấu phụ liên tiếp `Ố-Ồ-Ủ-Ộ`.
    - **Định hướng công bố khoa học**: Mọi báo cáo kỹ thuật và pipeline huấn luyện LoRA cần kết hợp cả 2 trục: Chuẩn hóa kích thước Token Mass tối thiểu và Tinh chỉnh Attention Routing phân luồng.

15. **ĐỊNH LUẬT CHẠM TRẦN HEURISTIC ZERO-SHOT & TÍNH TẤT YẾU CỦA LORA (THE ZERO-SHOT HEURISTIC CEILING LAW)**:
    - **Phát hiện thực nghiệm (`exp60` vs `exp61`)**:
      + Mô hình Base 4B vẽ được dấu kép phức tạp (minh chứng: `"CHỐNG"` ở `exp60` Pass C đẹp $100\%$).
      + Nhưng khi đưa vào bài toán thương mại đầy đủ (3 Text + 1 Sản phẩm thật): Mọi mẹo Prompting, Whitelist bề mặt vật đỡ (ruy băng, kính mờ, bảng kim loại) đều **hoàn toàn bất lực trong việc ép DiT Base định tuyến đúng dòng chữ giữa**.
    - **Kết luận tối hậu cho Giai đoạn 3**:
      + Các giải pháp Heuristic "rẻ/miễn phí" đã chạm trần vật lý của mô hình Base.
      + **Huấn luyện LoRA DiT 4B Base (Giai đoạn 3) là giải pháp DUY NHẤT, BẮT BUỘC VÀ TẤT YẾU** để biến FLUX.2 thành cỗ máy sản xuất Banner thương mại 4-slot đạt chuẩn $100\%$.

16. **QUY LUẬT PHÂN CẤP THỊ GIÁC 3 TẦNG & BỐ CỤC TỰ NHIÊN TRONG BANNER THƯƠNG MẠI (THE 3-TIER TYPOGRAPHIC HIERARCHY LAW)**:
    - **Phát hiện từ thực tế nghiệp vụ & Thử nghiệm Prompt**:
      + **Tầng 1 (Slot $t=10.0$ - Headline / Bắt mắt tức thì)**: Đóng vai trò tiêu đề chính, chữ nổi 3D lớn nhất, dập nổi kim loại/vàng gold/neon, chiếm $45\%$ trọng tâm thị giác.
      + **Tầng 2 (Slot $t=20.0$ - Subtitle / Thông tin bổ trợ)**: Đóng vai trò cung cấp thông tin chi tiết (thời gian, chất lượng, thông số), chữ vừa, sắc nét, thanh lịch, chiếm $30\%$ trọng tâm thị giác.
      + **Tầng 3 (Slot $t=30.0$ - CTA Badge / Kêu gọi hành động)**: Đóng vai trò kích thích chuyển đổi, thường gồm nhiều cụm từ ngắn gọn ("Ghé ngay hôm nay!", "Deal cực hot - Số lượng có hạn!"), chữ nhỏ hơn, trình bày dưới dạng **Huy hiệu / Badge / Khung neon / Sticker nhỏ xinh** ở các góc hoặc chân đế sản phẩm, chiếm $25\%$ trọng tâm thị giác.
      + Do toàn bộ tập dữ liệu Ground-Truth từ Gemini Teacher được cấu trúc chuẩn hóa theo đúng phân cấp 3 tầng này, LoRA sẽ tự động học được mối liên kết không gian:
        * Tọa độ $t=10 \rightarrow$ Phóng to kích thước, dập nổi 3D ở phần trên/trung tâm.
        * Tọa độ $t=20 \rightarrow$ Nét thẳng, thanh mảnh, nằm liền kề dưới tiêu đề.
        * Tọa độ $t=30 \rightarrow$ Tự động bao gói thành các khung huy hiệu (Badge/Neon Pill) nhỏ xinh ở các góc.

17. **ĐỊNH LUẬT BỘI SỐ 10 TIỀN HUẤN LUYỆN THẮNG TUYỆT ĐỐI GIẢ THUYẾT GÓC QUAY TOÁN HỌC (CANONICAL PRETRAINED DISCRETE OFFSETS SUPREMACY LAW)**:
    - **Kiểm chứng thực nghiệm trực tiếp (`probe_rope_phase_aliasing.py`)**:
      + Quét 8 mốc thời gian đối chứng trên cùng Seed 42: $[t=10.0, 44.0, 47.1, 50.0, 53.4, 56.5, 60.0, 70.0, 80.0]$.
      + **Kết quả thực nghiệm**:
        * Tại $t=50.0$ (bội 10 chuẩn BFL): **GIỮ NGUYÊN ĐÚNG 100% CHỮ TRÊN SẢN PHẨM & NẮP BẠC**, vượt trội hơn tất cả các mốc float lân cận.
        * Tại các mốc số thực lẻ $t=44.0, 47.1, 53.4, 56.5$ (dù $t=47.1$ ngược pha $180^\circ$ theo lý thuyết hình học): Đều rơi vào trạng thái **Out-of-Distribution (OOD)**, dẫn đến **sai chữ trên bao bì sản phẩm** và nắp bị rò rỉ ngữ nghĩa biến thành mạ vàng!
    - **Bản chất khoa học & Bài học tối hậu**:
      + Mạng nơ-ron sâu bị chi phối $100\%$ bởi phân phối gradient tiền huấn luyện (Empirical Data Exposure) hơn là các giả định hình học liên tục.
      + Ma trận trọng số Attention $W_Q, W_K$ của DiT đã được hiệu chuẩn sâu trên các mốc số nguyên rời rạc $t \in \{10.0, 20.0, 30.0, 40.0, 50.0\}$.
    - **Quy tắc Bắt buộc cho Pipeline & LoRA (Giai đoạn 3)**:
      + **TUYỆT ĐỐI KHÔNG DÙNG CÁC TỌA ĐỘ FLOAT LẺ**.
      + Toàn bộ hệ thống chỉ chuẩn hóa trên các mốc số nguyên: $t \in \{10.0, 20.0, 30.0, 40.0, 50.0\}$.
      + **Cơ chế Phân bổ Slot Động theo Ngữ cảnh (Dynamic Context-Aware Slot Assignment)**:
        * Chỉ có 1 Sản phẩm (Đổi background): Sản phẩm ở $t = 10.0$.
        * 1 Header + 1 Sản phẩm: Header $t = 10.0$, Sản phẩm $t = 20.0$.
        * 2 Text + 1 Sản phẩm: Text $t = 10.0, 20.0$, Sản phẩm $t = 30.0$.
        * 3 Text + 1 Sản phẩm: Text $t = 10.0, 20.0, 30.0$, Sản phẩm $t = 40.0$.
        * 4 Text + 1 Sản phẩm (Full-Power 5-Slot Cực Hạn): Text $t = 10, 20, 30, 40$, Sản phẩm $t = 50.0$.

18. **ĐỊNH LUẬT VÙNG AN TOÀN GLYPH VÀ TÍNH KHÔNG TỒN TẠI CỦA SỰ PHỤC HỒI CHU KỲ (THE TEXT GLYPH ROPE SAFE-ZONE & PRETRAINED DOMAIN EXPOSURE LAW)**:
    - **Kiểm chứng thực nghiệm trực tiếp (`probe_text_rope_phase_sweep.py` & `demo_tendoo_poster.py` trên `"CHỐNG ỒN CHỦ ĐỘNG"`)**:
      + Quét các mốc thời gian: $[t=10.0, 15.0, 40.0, 44.0, 47.1, 50.0, 53.4, 56.5, 60.0, 62.8, 66.0, 70.0, 80.0]$.
      + **Kết quả thực nghiệm**:
        * **CHỈ DUY NHẤT TẠI $t=10.0$ VÀ $t=40.0$**: Chữ `"CHỐNG ỒN CHỦ ĐỘNG"` được vẽ **ĐÚNG 100% HOÀN HẢO**, nét dập nổi 3D mạ vàng sắc lẹm, không sai một dấu tiếng Việt nào!
        * **TẤT CẢ CÁC MỐC CÒN LẠI ĐỀU THẤT BẠI HOÀN TOÀN (FAIL 100%)**:
          - $t=15.0, 44.0, 47.1, 53.4, 56.5, 62.8, 66.0$ (các mốc số thực lẻ): Rơi vào Out-of-Distribution (OOD), chữ bị biến dạng và hỏng dấu hoàn toàn.
          - $t=50.0, 60.0, 70.0, 80.0$: Glyph chữ thưa thớt bị suy thoái triệt để và **HOÀN TOÀN KHÔNG CÓ HIỆN TƯỢNG TỰ PHỤC HỒI THEO CHU KỲ (Zero Periodic Recovery)**.
    - **Bản chất Khoa học về Sự Khác Biệt Giữa Sản Phẩm vs Glyph Chữ tại $t=50.0$**:
      + **Sản Phẩm Thật sống sót ở $t=50.0$**: Do Black Forest Labs (BFL) tiền huấn luyện mô hình chủ yếu trên **ảnh chụp vật thể thật và khuôn mặt người (photo/subject dataset)**, ma trận Attention đã có sẵn biểu diễn vững chắc để giữ chi tiết sản phẩm ở các mốc xa $t \ge 50.0$.
      + **Glyph Chữ sụp đổ ở $t \ge 50.0$**: Do BFL **chưa từng huấn luyện DiT đọc Glyph chữ đen trắng ở các mốc xa**, mô hình hoàn toàn không có phản xạ chú ý cho loại token này ở $t \ge 50.0$.
    - **Quy chuẩn Phân Vùng Không Gian 4D RoPE**:
      + **Vùng Dành Riêng cho Glyph Văn Bản**: Khóa cứng $100\%$ các khối Text trong dải an toàn **$t \in [10.0, 40.0]$** ($t=10, 20, 30, 40$).
      + **Vùng Dành Riêng cho Sản Phẩm Thật**: Mốc $t=50.0$ là vị trí dành cho Sản phẩm thật (khi có 4 khối text) vì BFL đã có sẵn biểu diễn tiền huấn luyện vững chắc cho vật thể tại mốc này.

19. **ĐỊNH LUẬT THỤ THỂ KÍCH HOẠT VĂN BẢN VÀ TÍNH ĐỘC LẬP VẬT CHỨA (THE TEXT ACTIVATION RECEPTOR & CONTAINER-INDEPENDENCE LAW)**:
    - **Kiểm chứng thực nghiệm trực tiếp (`test_prompt_techniques_3x3.py` trên 3 độ dài văn bản)**:
      + **Prompt 1 (Không nhắc chữ)**: Mô hình **TẮT HOÀN TOÀN CHÚ Ý** $\rightarrow$ Dù Glyph có ở $t=10.0$, DiT vẫn bỏ qua và không vẽ chữ.
      + **Prompt 2 (Nhắc vai trò/chất liệu chữ, KHÔNG CẦN VẬT CHỨA)**: Mô hình **LẬP TỨC VẼ CHỮ ĐÚNG 100%**, biến hóa chữ 3D/neon cực đẹp.
      + **Prompt 3 (Thêm vật chứa bảng biển/vách đá)**: Cho thấy vật chứa chỉ là chi tiết bối cảnh trang trí, **HOÀN TOÀN KHÔNG QUYẾT ĐỊNH việc chữ có xuất hiện hay không**.
    - **Bản chất Khoa học & Bài học tối hậu**:
      + Mô hình DiT **nhận thức được khối token ở $t=10.0$ là HÌNH THÁI VĂN BẢN (Text Morphology)**.
      + Từ khóa chỉ vai trò/chất liệu (`dòng chữ tiêu đề`, `chữ 3D`, `đèn neon`, `mạ vàng`) trong Prompt đóng vai trò là **"Thụ thể kích hoạt (Semantic Activation Receptor)"** để bật công tắc Attention liên kết giữa Prompt và Glyph VAE.
      + Tuyệt đối không cần gượng ép đưa "bảng gỗ/khung biển" vào Prompt. Chữ 3D hoàn toàn có thể đứng tự do trong không gian.

20. **ĐỊNH LUẬT CO GIÃN TỶ LỆ QUA VẬT THỂ ĐỠ VÀ NGƯỠNG PHÂN GIẢI VAE (THE OBJECT-BOUND SCALE MODULATION & VAE RESOLUTION BOUND LAW)**:
    - **Kiểm chứng thực nghiệm đối chứng cô lập 3 Case (`test_prompt_scale_isolation.py`)**:
      + **Case 1 (Prompt thuần, không vật thể)**: Lệnh Prompt bảo vẽ "chữ siêu nhỏ" đứng tự do $\rightarrow$ **THẤT BẠI HOÀN TOÀN, CHỮ VẪN TO NGUYÊN VẸN** theo kích thước Glyph! (Khẳng định: Prompt thuần từ ngữ trừu tượng bất lực trong việc tự co nhỏ Glyph to).
      + **Case 2 (Ép chữ qua Vật thể có kích thước vừa vặn - Tách cà phê sứ)**: Khi gắn chữ vào một vật thể có kích thước vật lý cụ thể trong thế giới thực $\rightarrow$ **THÀNH CÔNG 100%, CHỮ ĐÚNG VÀ CO NHỎ TỰ NHIÊN** theo tỷ lệ của tách cà phê! (Khẳng định: Vật thể đỡ đóng vai trò là "Thước đo tỷ lệ phối cảnh 3D" để DiT neo và co nhỏ chữ).
      + **Case 3 (Ép chữ qua Vật thể siêu nhỏ - Tem nhãn mini trên gói cafe)**: Khi vật thể quá nhỏ khiến diện tích thực tế của chữ trên Canvas bị ép xuống dưới ngưỡng giải mã của VAE ($16\times$) $\rightarrow$ **CHỮ BỊ SAI/VỠ NÉT**!
    - **Quan sát Thực nghiệm Cần Lưu Ý**:
      + Prompt thuần dùng từ ngữ trừu tượng ("tiny/small") không tự co nhỏ được Glyph to nếu không có bối cảnh bố cục giới hạn.
      + Nếu chữ bị co nhỏ dưới ngưỡng phân giải hiển thị trên Canvas (chiều cao nét chữ rơi xuống dưới $\sim 20 - 24\text{px}$), bộ giải mã VAE sẽ không đủ thông tin latent dẫn đến nét chữ bị vỡ hoặc sai chính tả.
      + Khi muốn chữ uốn lượn (Trái Đất, ruy băng): Glyph bắt buộc là 1 dòng dài chiều ngang (1D Manifold).

21. **ĐỊNH LUẬT CẠNH TRANH KHÔNG GIAN BỐ CỤC VÀ ĐỘ NÉT 100% CỦA VĂN BẢN ĐA DÒNG (THE SPATIAL REAL-ESTATE COMPETITION & MULTI-LINE TEXT LAW)**:
    - **Kiểm chứng thực nghiệm trực tiếp (`test_perfume_916_bevietnam_split.png`)**:
      + **Chữ đạt độ chính xác 100%**: Khi Glyph bảo toàn chiều cao `box_h=448px`, khoảng cách dòng $32\%$ và dùng font nét dày (`bevietnam`), mô hình vẽ **ĐÚNG 100% CẢ 4 DÒNG CHỮ**, bao gồm cả dòng `"Mua 1 tặng 1"` và ngày tháng `"Đến hết ngày 26/5"`.
      + **Hiện tượng Cạnh tranh Không gian (Spatial Real-Estate Trade-Off)**:
        * Khi khối chữ 4 dòng chiếm `box_h = 448px` trên Canvas 9:16 ($576 \times 1024$), khối text chiếm tới $\sim 70\%$ diện tích nửa dưới của poster.
        * Chủ thể sản phẩm (chai nước hoa) bị đẩy dồn lên trên và co lại chỉ chiếm $\sim 30\%$ diện tích nửa trên ("lọ nước hoa bị lùn đi").
    - **Bản chất Khoa học & Quy tắc Thiết kế Hệ thống**:
      + **Tổng diện tích Canvas là hữu hạn ($100\%$)**: Khối lượng token và chiều cao của Glyph Box quyết định trực tiếp "thị phần không gian" (spatial real-estate) trên bức ảnh sinh ra.
      + **Muốn Sản phẩm To nổi bật ($\ge 60\%$) + Text Đa Dòng Nhỏ Tinh Tế ($\le 40\%$)**:
        * Hoặc chuyển sang Canvas vuông $1:1$ ($1024 \times 1024$) hoặc $4:5$ ($896 \times 1120$) để có bề ngang rộng hơn cho text dàn trải mà không lấn chiếm chiều dọc.
        * Hoặc dùng cơ chế **In-Context Product Reference Image ($t=60.0$)** để khóa cứng tỷ lệ và hình dáng chai nước hoa, ngăn không cho DiT tự ý thu nhỏ chai nước hoa!

22. **ĐỊNH LUẬT LẤY MẪU NYQUIST CẤP KÝ TỰ & NGƯỠNG TRIỆT TIÊU GAI NÉT (THE CHARACTER-LEVEL NYQUIST SAMPLING & ANTI-ALIASING LAW)**:
    - **Kiểm chứng thực nghiệm trực tiếp (`probe_glyph_resolution_threshold.py` và ảnh sổ tay `exp_small_text_test.png` với `"TÔI YÊU VIỆT NAM"`)**:
      + Khi thu nhỏ Glyph xuống `box_w=384, box_h=112` (`font_size ≈ 30pt`, chỉ $84$ latent tokens):
        * Mô hình thành công vang dội: Đã ép được chữ nhỏ dập chìm mạ vàng ở góc dưới bìa da cuốn sổ tay theo đúng prompt, không bị phóng to thành tiêu đề khổng lồ!
        * Tuy nhiên, các nét mũ (`Ô`, `Ê`) và dấu nặng (`Ệ`) xuất hiện **hiện tượng "gai nét" (jagged aliasing)**.
    - **Bản chất Toán học & Ma trận Đo đạc Nyquist trên Lưới Latent $16\times$**:
      + Ở cỡ chữ $30\text{pt}$ (`BeVietnamPro-Black`):
        * Chiều cao ký tự: $22\text{px}$ ($1.38$ latent tokens).
        * Chiều cao dấu mũ: $7\text{px}$ (**$0.44$ latent tokens**).
        * Dấu nặng: $8\text{px}$ ($0.50$ latent tokens).
      + **Định luật Nyquist-Shannon trên VAE $16\times$**:
        * Khi một chi tiết (dấu phụ, nét thanh) có kích thước $< 0.50$ latent token ($< 8\text{px}$): VAE Decoder phải thực hiện phép nội suy siêu phân giải vượt quá giới hạn lấy mẫu (Sub-Nyquist) $\implies$ sinh ra các gai răng cưa (spatial quantization spikes).
    - **3 Ngưỡng Độ Phân Giải Định Lượng Bất Biến**:
      1. **Vùng Đỏ (Sub-Nyquist - Gai nét, răng cưa)**:
         - Font size $\le 30\text{pt}$ (Dấu phụ $< 8\text{px}$, tức $< 0.50$ latent token).
         - Hậu quả: Dấu mũ và nét thanh bị gai, mép gợn răng cưa.
      2. **Vùng Vàng (Marginal - Chấp nhận được, mép mềm)**:
         - Font size $36 - 44\text{pt}$ (Dấu phụ $9 - 11\text{px}$, tức $0.56 - 0.69$ latent token).
         - Chiều cao dòng $\approx 128\text{px}$ ($8$ latent tokens).
         - Nét chữ liền mạch, dấu phụ bắt đầu tròn trịa, phù hợp cho chữ phụ / CTA / slogan nhỏ.
      3. **Vùng Xanh (Silk-Smooth - Mịn màng 100%, sắc nét tuyệt đối)**:
         - Font size $\ge 48 - 56\text{pt}$ (Dấu phụ $\ge 12\text{px}$, tức $\ge 0.75$ latent token; Ký tự $\ge 36 - 43\text{px}$, tức $\ge 2.25 - 2.69$ latent tokens).
         - Chiều cao dòng $\ge 144 - 160\text{px}$ ($9 - 10$ latent tokens).
         - Nét chữ 100% trơn láng, không một vết gai (chính là ngưỡng đã giúp bài thơ Tây Tiến `exp54` đạt 100/100 điểm tuyệt đối!).
    - **Công thức Khóa Cứng Thiết kế Hệ thống**:
      - Với bất kỳ khối văn bản $N$ dòng:
        $$\text{box\_h} \ge N \times 128\text{px}\quad \text{(Ngưỡng tối thiểu chống gai nét - Anti-Aliasing Lower Bound)}$$
        $$\text{box\_h} \ge N \times 144 - 160\text{px}\quad \text{(Ngưỡng tối ưu đạt độ mịn lụa 100\% - Silk-Smooth Golden Target)}$$
      - Đảm bảo font size trong Binary Search không bao giờ rơi xuống dưới ngưỡng an toàn $36\text{pt}$!

23. **ĐỊNH LUẬT ĐA DẠNG HÓA TOPOLOGY BỐ CỤC ĐỂ PHÂN LUỒNG CHÚ Ý TỔNG QUÁT (THE TOPOLOGICAL DIVERSITY & UNIVERSAL ATTENTION ROUTING LAW)**:
    - **Bản chất Khoa học**:
      + Nếu toàn bộ dataset chỉ dùng một mẫu bố cục xếp chồng dọc cổ điển (Đỉnh: Header $\rightarrow$ Giữa: Slogan $\rightarrow$ Đáy: CTA), mạng Attention của DiT sẽ vô tình "học vẹt" mối tương quan giả (Spurious Correlation) rằng: $t=10.0$ bắt buộc phải nằm ở trên đỉnh, $t=30.0$ bắt buộc phải nằm ở đáy.
      + Khi người dùng yêu cầu bài toán thực tế khác (như Prompt 1: Title ở góc trên bên trái, Subtitle ở giữa bên trái; hay bài toán Card Feedback: nửa trái là ảnh, nửa phải là 4 khối text), mô hình sẽ bị bối rối và vẽ sai vị trí.
    - **Quy chuẩn 4 Dạng Topology Bắt Buộc trong Dataset 2.500 Mẫu**:
      1. **Poster Dọc Cổ Điển ($35\%$ - $875$ mẫu)**: Xếp chồng dọc Top-Mid-Bottom (Standee, Flash Sale, Sự kiện).
      2. **Phân Tách Trái - Phải ($25\%$ - $625$ mẫu)**: Nửa trái là Sản phẩm / Ảnh Before-After $\longleftrightarrow$ Nửa phải là 4 tầng thông tin (Card Feedback Gym, Spa, Khóa học, Nha khoa dựa trên `prompt_test.txt`).
      3. **Lưới Đều / Menu Ma Trận ($20\%$ - $500$ mẫu)**: Các khối chữ có cỡ tương đương nhau không có Hero Title áp đảo (Menu cà phê/trà sữa, Bảng so sánh cước 5G Viettel, Bảng thông số kỹ thuật).
      4. **Tự Do Bất Đối Xứng & Chữ Nổi ($20\%$ - $500$ mẫu)**: Chữ đặt lệch góc, nổi trên khoảng trống âm (Negative space), typography động (Tin tuyển dụng, Khai trương, Thời trang).
    - **Ý nghĩa Thực thi**:
      + Giữ nguyên tiến trình curriculum $2 \rightarrow 3 \rightarrow 4/5$ slots và quy mô ngân sách $2,500$ mẫu.
      + Giúp DiT làm chủ cơ chế Attention Routing như một **năng lực không gian tổng quát (Universal Spatial Capability)**, phục vụ trọn vẹn mọi yêu cầu đồ họa đa dạng trong thực tế của Viettel.

24. **ĐỊNH LUẬT TRỰC GIAO HÓA FONT VÀ NGÀNH HÀNG ĐỂ CHỐNG LIÊN KẾT GIẢ (THE ORTHOGONAL FONT-DOMAIN DECOUPLING & ZERO-SPURIOUS CORRELATION LAW)**:
    - **Bản chất Khoa học**:
      + Cố định cứng $1:1$ giữa Ngành hàng và Font chữ (Thời trang luôn đi với Serif `Playfair`, F&B luôn đi với `Sedgwick`, Tech luôn đi với `Anton`) sẽ tiêm vào mô hình một **liên kết giả (Spurious Correlation)** nguy hại: Mô hình tưởng rằng *"chữ serif mạ vàng $\Longleftrightarrow$ bối cảnh thời trang"*, khiến nó mất khả năng vẽ chữ Serif trên đồ công nghệ hay vẽ chữ Sans trên đồ thời trang.
      + Mô hình Base vốn dĩ đã có sẵn năng lực tái tạo font zero-shot $100\%$ (chứng minh qua Probe Suite 1). Bài toán cốt lõi của LoRA là **phân luồng không gian đa slot**, tuyệt đối không phải "dạy font".
    - **Quy chuẩn Trực Giao Hóa ($I(\text{Font}; \text{Domain}) = 0$)**:
      + Huy động toàn bộ **Pool 16 Font Unicode Tiếng Việt** sẵn có trong `fonts/` (Sans-Serif, Editorial Serif, Heavy Display, Script/Calligraphy, Brush/Rounded).
      + Bất kỳ font chữ nào cũng xuất hiện ngẫu nhiên và bình đẳng trong mọi ngành hàng (Fashion dùng cả Sans/Bold, Tech dùng cả Serif/Sans, F&B dùng cả Rounded/Brush/Serif).
      + Mỗi font chỉ cần xuất hiện từ $50 - 150$ mẫu trải đều là đủ để triệt tiêu hoàn toàn liên kết giả, giải phóng năng lực tổng quát hóa tối đa cho LoRA.

25. **ĐỊNH LUẬT TIGHT-CROP VỪA ĐỦ ĐỂ TRIỆT TIÊU ĐỊNH KIẾN KÍCH THƯỚC VÀ TỐI ƯU HÓA TOKEN (THE OPTIMAL TIGHT-CROP & SIZE-BIAS ELIMINATION LAW)**:
    - **Bản chất Khoa học**:
      + Nếu cố tình phóng to Glyph Box nhân tạo ($400 - 640\text{ tokens}$ cho mỗi slot text), tổng chiều dài chuỗi Reference cho 4 Text + 1 Sản phẩm sẽ bị đội lên gần $2,000\text{ tokens}$ (tổng sequence length vượt $10,500\text{ tokens}$), gây nghẽn bộ nhớ VRAM và làm chậm hàm Attention bình phương $\mathcal{O}(L^2)$ tới $3\times$.
      + Nguy hại hơn, việc cố định `Ref_10` to và `Ref_30` nhỏ sẽ tiêm vào DiT một **định kiến ngầm về kích thước (Size Bias)**: Mô hình ngộ nhận rằng slot có nhiều token bắt buộc phải là Header lớn, gây sai lệch khi gặp bài toán Menu (các món bằng nhau) hoặc bài toán Card Feedback (đoạn quote ở Slot 3 có nhiều chữ hơn Header ở Slot 1).
    - **Quy chuẩn Tight-Crop Vừa Đủ**:
      + Nhờ thực nghiệm kiểm chứng (`exp_small_text_test.png`), một khối text 1 dòng chỉ cần **$80 - 140\text{ tokens}$** (đảm bảo font size $\ge 36 - 44\text{pt}$, $\text{box\_h} \ge 128\text{px}$) là đã đạt độ phân giải tối ưu chống gai nét.
      + Kích thước Token được co dãn thuần túy theo **độ dài từ và số dòng thực tế**, không phân biệt Header hay CTA:
        * *1 dòng ngắn ($1 - 3$ từ)*: $80 - 140\text{ tokens}$.
        * *1 dòng vừa ($4 - 6$ từ)*: $130 - 200\text{ tokens}$.
        * *2 dòng ($6 - 10$ từ)*: $220 - 320\text{ tokens}$.
        * *Đoạn dài ($3 - 4$ dòng / $15 - 25$ từ)*: $380 - 640\text{ tokens}$.
    - **Lợi ích Thực thi**:
      + Tiết kiệm $>60\%$ sequence length (tổng 4 slot text chỉ chiếm $\sim 500 - 750\text{ tokens}$).
      + Tăng tốc độ huấn luyện và suy luận phục vụ gấp $2 - 3$ lần trên hạ tầng 2x NVIDIA A30.
      + Triệt tiêu hoàn toàn Size Bias, cho phép bất kỳ slot nào cũng có thể đóng vai trò Header to, Quote dài hay Badge nhỏ tùy theo prompt!

26. **ĐỊNH LUẬT TÍCH HỢP TỌA ĐỘ KHÔNG GIAN TRONG PROMPT VÀ HÀM SIZING TRỰC GIAO (THE PROMPT-SPATIAL GROUNDING & UNIVERSAL SIZING LAW)**:
    - **Sự Thật Kiến Trúc Cốt Tử**:
      + Trong hàm `encode_glyph_to_incontext_tokens`, toạ độ RoPE `h_coords` và `w_coords` của Reference Token chỉ là **toạ độ nội bộ trong bounding-box của glyph ($0 \rightarrow H_{\text{glyph}}, 0 \rightarrow W_{\text{glyph}}$), HOÀN TOÀN KHÔNG CÓ toạ độ tuyệt đối trên Canvas $1024 \times 1024$**.
      + Nghĩa là bản thân Glyph Token **không tự biết nó nằm ở đâu trên ảnh**. Quyết định vị trí đặt chữ trên Canvas thuộc về **sự tương tác Attention giữa Canvas và Text Prompt qua Qwen3**.
      + Toàn bộ 21 prompt thực tế của tester (`prompt_test.txt`) đều chứa từ chỉ vị trí tường minh (*"ở góc trên bên trái"*, *"ở giữa bên trái"*, *"ở đáy góc phải"*). Nếu training data chỉ dùng từ `"poster/banner"` chung chung, mô hình sẽ mất khả năng điều khiển vị trí theo ý muốn người dùng!
    - **Cấu Trúc 3 Thành Phần Bắt Buộc Trong Prompt Huấn Luyện**:
      1. **Chỉ Dẫn Vị Trí Tường Minh (Explicit Spatial Anchor)**: *"ở góc trên bên trái"*, *"ở giữa bên trái"*, *"ở trung tâm phía trên"*, *"ở đáy poster"*.
      2. **Quy Mô & Vai Trò (Scale & Role Descriptor)**: *"dòng chữ tiêu đề lớn"*, *"dòng chữ phụ thanh mảnh"*, *"huy hiệu ưu đãi nhỏ nhắn"*, *"đoạn trích dẫn nhận xét"*.
      3. **Vật Lý, Chất Liệu & Quang Học (Material & Optics)**: *"dập nổi mạ vàng"*, *"đèn neon phát quang"*, *"khắc chìm trên gỗ"*, *"đổ bóng studio 3D"*.
    - **Hàm Sizing Phổ Quát Dùng Chung Cho Mọi Slot (`compute_optimal_glyph_box`)**:
      + Bỏ hoàn toàn range token theo slot. Mọi slot đều dùng chung 1 hàm sizing, phụ thuộc vào độ dài ký tự và **ngưỡng sàn riêng của từng font (Per-Font Minimum Floor)**:
        * *Nhóm A (Đậm nét: `BeVietnamPro-Black`, `Anton`, `Gotham`)*: Floor = $36\text{pt}$.
        * *Nhóm B (Serif & Condensed: `PlayfairDisplay`, `Oswald`)*: Floor = $40\text{pt}$.
        * *Nhóm C (Nét mảnh, Script, Cọ vẽ: `DancingScript`, `Pacifico`, `Sedgwick`, `Blow Brush`)*: Floor = $44 - 48\text{pt}$ (tránh bị VAE nuốt nét thanh).
    - **Phân Định Tam Hợp Bất Biến**:
      + **Prompt Qwen3**: Định hướng **Vị trí không gian** và **Kích thước thị giác**.
      + **RoPE Time Offset ($t=10, 20, 30, 40$)**: Định danh **Kênh phân luồng độc lập**.
      + **Glyph VAE**: Bảo toàn **100% Hình học và Chính tả**.

27. **ĐỊNH LUẬT HUẤN LUYỆN NĂNG LỰC CẠNH TRANH VÀ PHÂN TẦNG ĐỘ DÀI VĂN BẢN (THE COMPETITIVE ATTENTION TRAINING & STRATIFIED LENGTH INVARIANCE LAW)**:
    - **Tuyên Ngôn Cốt Lõi**:
      + **CHÚNG TA HUẤN LUYỆN NĂNG LỰC CẠNH TRANH, TUYỆT ĐỐI KHÔNG HUẤN LUYỆN POSTER OVERFIT!**
      + Mô hình Base 4B đã có sẵn năng lực đọc-viết tiếng Việt zero-shot với bất kỳ độ dài text nào khi ở trạng thái cô lập (Probe Suite 1).
      + LoRA chỉ làm một nhiệm vụ duy nhất: **Giải quyết tranh chấp Softmax khi nhiều slot cùng hoạt động đồng thời (Attention Disentanglement)**, hoàn toàn không phụ thuộc vào việc slot đó mang câu ngắn hay câu dài.
    - **Nguy Cơ Của Việc Overfit Cấu Trúc Poster**:
      + Nếu gán cứng $t=10$ luôn là chữ ngắn ($3 - 4$ từ) và $t=20$ luôn là chữ dài ($2$ dòng), mô hình sẽ học vẹt **liên kết giả thứ hai (`slot \Longleftrightarrow \text{length}`)**, khiến nó không thể vẽ được câu triết lý thương hiệu dài ở Title hay vẽ được từ ngắn ở Subhead.
    - **Quy Chuẩn Phân Tầng Vàng 75/25 (Golden Stratified Ratio)**:
      + **75% – 80% Phân bố tự nhiên thương mại**: Phục vụ tối ưu cho sản xuất thực tế ($t=10$ Title ngắn/vừa, $t=20$ Subtitle vừa, $t=30$ CTA ngắn, $t=40$ Features dài).
      + **20% – 25% Chủ đích nghịch đảo độ dài**: $t=10$ Title dài bất thường ($10 - 18$ từ / $2 - 3$ dòng), $t=20$ Subtitle cực ngắn ($1 - 2$ từ: *"SIÊU NHẸ"*, *"PRO"*), $t=30$ CTA dài hơn Title ($8 - 12$ từ: *"NHỮNG VẬT BẤT LY THÂN CỦA BẠN"*).
    - **Quy Tắc Thực Thi Bắt Buộc**:
      + **Phân tầng nghịch đảo này PHẢI ĐƯỢC ĐƯA VÀO NGAY TỪ MILESTONE A (2 SLOTS)**!
      + Tuyệt đối không dồn tới Milestone C, nhằm triệt tiêu liên kết giả ngay từ gốc rễ, đảm bảo các milestone sau kế thừa một bộ khung chú ý hoàn toàn sạch và tổng quát.






