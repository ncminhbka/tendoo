# 🎯 KẾ HOẠCH & LỘ TRÌNH HUẤN LUYỆN LORA DiT 4B BASE (PHASE 3 MASTER ROADMAP)

- **Dự án**: Tendoo AI – Hệ Thống Sinh Banner Quảng Cáo Thương Mại Đa Khối Chữ Tiếng Việt
- **Mô hình Mục Tiêu Duy Nhất**: **`FLUX.2-klein-base-4B`** (Bản Base nguyên bản, Euler ODE Flow Matching, CFG = 4.0).
- **Hạ tầng Thực thi**: **2x NVIDIA A30 (24GB VRAM $\times 2 = 48$GB VRAM)**, Ampere Architecture, DDP (`accelerate`).
- **Bản chất Toán học Cốt lõi**: Huấn luyện bài toán **Định Tuyến & Phân Tách Chú Ý (Attention Disentanglement & Routing)** trên không gian 4D RoPE, giải quyết hiện tượng tranh chấp Softmax và rò rỉ đặc trưng Ref-to-Ref mà **KHÔNG CẦN DẠY LẠI BIỂU DIỄN CHỮ TỪ ĐẦU**.

---

## 📊 1. MA TRẬN PHÂN TÍCH KỸ THUẬT & ĐỊNH HÌNH THIẾT KẾ (TECHNICAL BASIS)

Dựa trên 60 chuỗi thực nghiệm đối chứng từ `exp01` đến `exp60`, lộ trình huấn luyện được xây dựng trên các chân lý kỹ thuật đã được chứng minh $100\%$:

| Thành phần Kiến trúc | Phát hiện Thực nghiệm / Chân lý Toán học | Giải pháp Kỹ thuật trong Pipeline Huấn luyện |
| :--- | :--- | :--- |
| **1. Softmax Joint Attention** | Toàn bộ Key $K$ của Canvas, Sản phẩm (4096 tokens) và các Glyph bị gom chung vào 1 Softmax duy nhất $\rightarrow$ gây ra hiện tượng Token Mass Dominance đè bẹp các khối nhỏ. | Chuẩn hóa kích thước Token Mass tối thiểu $\ge 672$ tokens ($768 \times 224\text{px}$) cho mọi khối text trong Dataset. |
| **2. Target LoRA Layers** | FLUX.2 không có module Cross-Attention riêng; Canvas và Ref dùng chung `img_attn.qkv` (DoubleBlocks) và `linear1` (SingleBlocks). 80% độ sâu mô hình nằm ở 20 SingleBlocks. | Tiêm LoRA trực tiếp vào: `img_attn.qkv` (5 DoubleBlocks) và phần Q, K, V của `linear1` (20 SingleBlocks). Rank $r=32$, $\alpha=32$. |
| **3. Uniform AdaLN Modulation** | `ref_fixed_timestep = 0.0` cố định cho mọi slot $\rightarrow$ mô hình không có biên độ ưu tiên giữa các slot, toàn bộ nhận diện slot dồn vào RoPE. | LoRA sẽ tối ưu hóa ma trận $W_Q, W_K$ để thích ứng nhạy bén với các dải tần số góc quay RoPE $\mathbf{R}(\Delta t)$ của từng slot ($t=10, 20, 30$). |
| **4. Upstream Semantic Clash** | Khi prompt lặp lại nguyên văn text tiếng Việt, Text Encoder `Qwen3-4B-FP8` gây nhiễu và xung đột với tín hiệu Glyph từ VAE. | Tích hợp Upstream LLM (Gemini Flash / Qwen-2.5) tự động bóc tách text và làm sạch prompt (Prompt Sanitization). |
| **5. Phân bổ Slot Chuẩn** | Mốc $t \le 40.0$ là vùng hoạt động an toàn tuyệt đối. Mốc $t \ge 50.0$ bắt đầu suy hao góc pha RoPE đối với glyph chữ. | Khóa cứng 4 Slot chuẩn: $t=10.0$ (Headline), $t=20.0$ (Subtitle), $t=30.0$ (CTA Badge), $t=40.0$ (Ảnh Sản Phẩm Thật). |

---

## 🗂️ 2. THIẾT KẾ DỮ LIỆU & QUY TRÌNH CHẾ TẠO DATASET (DISTILLATION ENGINE)

### 2.1. Quy cách một Training Sample chuẩn:
Mỗi mẫu huấn luyện là một bộ tứ thống nhất:
$$\text{Sample}_i = \left( \text{Prompt}_{\text{clean}}, \; \text{Ref}_{10}, \; \text{Ref}_{20}, \; \text{Ref}_{30}, \; \text{Ref}_{\text{prod\_40}}, \; \mathbf{X}_{\text{target}} \right)$$

1. **`Prompt_clean`**: Mô tả phong cách thị giác, bố cục, ánh sáng studio, chất liệu 3D, **TUYỆT ĐỐI KHÔNG CHỨA CHỮ NGUYÊN VĂN**.
2. **`Ref_10` (Headline Glyph)**: Kích thước $768 \times 224$ (672 tokens), font nghệ thuật (Anton / Playfair / Oswald).
3. **`Ref_20` (Subtitle Glyph)**: Kích thước $768 \times 224$ (672 tokens), font thông tin (BeVietnam / Sans).
4. **`Ref_30` (CTA Badge Glyph)**: Kích thước $768 \times 224$ (672 tokens), font uốn lượn/dạ quang (Pacifico / Sedgwick).
5. **`Ref_prod_40` (Ảnh Sản phẩm Thật)**: Kích thước $1024 \times 1024$ (4096 tokens), ảnh sản phẩm studio sạch nền.
6. **`X_target` (Ảnh Ground-Truth $1024 \times 1024$)**: Ảnh poster hoàn chỉnh đạt chuẩn thương mại, chữ hiển thị $100\%$ đúng hình học của 3 glyph và hòa quyện ánh sáng.

---

### 2.2. Phân bố Lĩnh vực & Quy mô Dataset (Domain Sizing Matrix):

```
                                  TỔNG QUY MÔ DATASET: 2,500 SAMPLES
                                                  │
         ┌───────────────────┬────────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                    ▼                   ▼                   ▼
    [ ☕ F&B / Cafe ]   [ 📱 Công Nghệ ]    [ 👗 Thời Trang ]   [ 💆 Spa / Mỹ Phẩm ] [ 🛍️ Siêu Thị / FMCG ]
       500 samples         500 samples          500 samples          500 samples          500 samples
     (Khắc gỗ, neon)     (Kim loại, LED)      (Thanh lịch, gold)   (Tối giản, pastel)   (Dập nổi, pop-art)
```

---

### 2.3. Quy trình Chế tạo Ground-Truth (Multimodal Distillation Pipeline):

```
       [ 1. Template Layout Generator (Python) ]
       • Sinh ngẫu nhiên: Lĩnh vực, Headline, Subtitle, CTA, Font, Chất liệu, Bố cục
                          │
                          ▼
       [ 2. Backend Dynamic Glyph Engine ]
       • Xuất 3 ảnh Glyph Bitmap (768x224, tight crop, đúng Unicode tiếng Việt)
                          │
                          ├─────────────────────────────────────────────────┐
                          ▼                                                 ▼
        [ 3A. Nhánh Gemini 2.0 In-Context (70%) ]        [ 3B. Nhánh Programmatic Shader (30%) ]
        • Đưa 3 Glyph + Ảnh Sản Phẩm vào Gemini API      • Render chữ font TTF lên nền Background sạch
        • Prompt: "Giữ 100% hình học glyph, render 3D"   • OpenCV/Shader: Thêm drop shadow, viền kim loại,
        • Xuất ảnh Ground Truth tự nhiên hoàn hảo          ánh sáng phát quang neon
                          │                                                 │
                          └───────────────────────┬─────────────────────────┘
                                                  ▼
                                [ 4. Automated Quality Filter ]
                                • OCR Check: Độ khớp ký tự tiếng Việt >= 98%
                                • SSIM / Color distribution check
                                • Đóng gói thành WebDataset / Sharded HDF5
```

### 2.4. Ma trận Đa Tỉ Lệ Khung Hình (Aspect Ratio Bucketing Strategy):

Để chống hiện tượng **Spatial Coordinate Overfitting** (học vẹt bố cục vuông 1:1) và phục vụ trực tiếp các định dạng thương mại thực tế, tập dữ liệu huấn luyện được phân bổ đa dạng qua 4 bucket chuẩn (bảo toàn diện tích $\approx 1\text{ Megapixel} \approx 4,032 - 4,096\text{ tokens}$):

| Tỉ Lệ Bucket | Kích Thước Pixel | Latent Grid ($16\times$) | Token Canvas | Tỷ Trọng | Ứng Dụng Nghiệp Vụ Thương Mại |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1:1** (Vuông) | $1024 \times 1024$ | $64 \times 64$ | $4,096$ tokens | **35%** | Bài đăng Feed Facebook, Instagram, E-commerce Post |
| **9:16** (Dọc cao) | $768 \times 1344$ | $48 \times 84$ | $4,032$ tokens | **35%** | **TikTok Ads**, Instagram Reels, Story, Standee quảng cáo |
| **4:5** (Dọc vừa) | $896 \times 1152$ | $56 \times 72$ | $4,032$ tokens | **15%** | Instagram Portrait Post (Tối ưu diện tích màn hình mobile) |
| **16:9** (Ngang) | $1344 \times 768$ | $84 \times 48$ | $4,032$ tokens | **15%** | Facebook Fanpage Cover, Website Banner, TV Display |

> [!TIP]
> **TỐI ƯU HÓA BỘ NHỚ**: Do mỗi GPU nhận `batch_size = 1`, việc luân chuyển giữa các bucket tỉ lệ khác nhau trong DataLoader diễn ra **hoàn toàn tự nhiên, không cần zero-padding và không làm tăng thêm 1 MB VRAM nào**!

---

## ⚙️ 3. THIẾT KẾ KIẾN TRÚC LORA & HYPERPARAMETERS


### 3.1. Cấu hình PEFT LoRA Injection:
```python
# Cấu hình LoRA tối ưu cho FLUX.2 Klein 4B Base
lora_config = {
    "r": 32,                          # Rank 32 đủ dung lượng học phân luồng đa slot
    "lora_alpha": 32,                 # Scaling factor = 1.0
    "lora_dropout": 0.05,
    "target_modules": [
        "img_attn.qkv",               # 5 DoubleStreamBlocks (Query, Key, Value ảnh/glyph)
        "linear1",                    # 20 SingleStreamBlocks (Joint QKV + MLP projection)
    ],
    "bias": "none",
    "dtype": "bfloat16"
}
```

* **Tổng tham số mô hình Base 4B**: $\approx 4.08 \times 10^9$ parameters.
* **Tổng tham số LoRA cần huấn luyện**: $\approx 35.4 \times 10^6$ parameters (**chỉ chiếm $\mathbf{0.86\%}$ mô hình**).
* **Kích thước file trọng số LoRA lưu trữ**: $\approx \mathbf{70.8\text{ MB}}$ (`.safetensors`).

---

### 3.2. Bảng Siêu Tham Số Huấn Luyện (Hyperparameters):

| Tham số Huấn luyện | Giá trị Cấu hình | Cơ sở Lý thuyết / Tính toán |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$) | Chuẩn cho Diffusion Transformers |
| **Weight Decay** | $0.01$ | Chống overfit trên các font hiếm |
| **Learning Rate** | $1.0 \times 10^{-4}$ | Tốc độ tối ưu cho LoRA Rank 32 |
| **LR Scheduler** | `CosineAnnealingLR` với 150 warmup steps | Đảm bảo ổn định ở các step đầu |
| **Batch Size per GPU** | $1$ sample ($1024 \times 1024$) | Do độ dài chuỗi sequence dài $\sim 10,208$ tokens |
| **Gradient Accumulation** | $4$ steps | Tạo Effective Batch Size $= 1 \times 2 \times 4 = \mathbf{8}$ |
| **Precision Mode** | `bfloat16` Native Mixed Precision | Tối ưu kiến trúc Tensor Core A30, chống tràn số |
| **Gradient Checkpointing** | Kích hoạt trên toàn bộ 25 Blocks | Giảm bộ nhớ kích hoạt trung gian $>65\%$ |
| **Max Gradient Norm** | $1.0$ (Gradient Clipping) | Chống hiện tượng gradient spike khi gặp glyph phức tạp |
| **Hàm Mất Mát (Loss)** | Flow Matching MSE Loss: $\mathcal{L} = \| v_\theta - (x_1 - x_0) \|^2$ | Chuẩn Flow Matching Euler ODE của BFL |

---

## 📈 4. LỘ TRÌNH HUẤN LUYỆN 3 MỐC (CURRICULUM LEARNING ROADMAP)

Để đảm bảo gradient hội tụ mượt mà và không làm "sốc" ma trận Attention của mô hình Base, quá trình train được chia làm 3 Phase tăng dần độ phức tạp:

```
[ GIAI ĐOẠN 3.1: MILESTONE A ] ──────► [ GIAI ĐOẠN 3.2: MILESTONE B ] ──────► [ GIAI ĐOẠN 3.3: MILESTONE C ]
  (1 Text t=10 + 1 SP t=40)              (2 Texts t=10,20 + 1 SP t=40)           (3 Texts t=10,20,30 + 1 SP t=40)
  • 600 Steps (~45 phút)                  • 1,200 Steps (~1.5 giờ)                • 2,200 Steps (~2.8 giờ)
  • Học hòa trộn SP & Headline            • Học tách kênh Subtitle (t=20)         • Khóa toàn bộ 4-slot Production
```

### 📍 Chi tiết từng Milestone:

#### 🔹 Milestone A: Đồng bộ Hòa trộn 1 Sản Phẩm ($t=40$) + 1 Headline ($t=10$)
* **Mục tiêu**: Dạy LoRA phân luồng mượt mà giữa khối token khổng lồ của sản phẩm (4096 tokens) và khối chữ chính (672 tokens), đảm bảo chữ luôn ăn khớp ánh sáng với sản phẩm.
* **Số bước**: `600 steps` (khoảng 2 epochs với tập 500 samples).
* **Tiêu chuẩn nghiệm thu**: Headline đạt độ chính xác $100\%$, sản phẩm đạt độ giống thật $\ge 98\%$.

#### 🔹 Milestone B: Phân tách Chú ý 2 Khối Text ($t=10, 20$) + Sản Phẩm ($t=40$)
* **Mục tiêu**: Kích hoạt khả năng phân tách kênh $t=20$ (Subtitle), triệt tiêu hoàn toàn hiện tượng Ref-to-Ref contamination (rò rỉ can nhiễu giữa các ref) và ngăn chặn DiT tự sinh chữ rác Lorem Ipsum.
* **Số bước**: `1,200 steps`.
* **Tiêu chuẩn nghiệm thu**: Cả Headline và Subtitle đều render chuẩn $100\%$ chữ và dấu tiếng Việt trên cùng 1 ảnh.

#### 🔹 Milestone C: Toàn diện 3 Khối Text ($t=10, 20, 30$) + Sản Phẩm ($t=40$)
* **Mục tiêu**: Khóa cứng toàn bộ ma trận Attention cho bố cục chuẩn thương mại hoàn chỉnh (Headline 3D + Subtitle thông tin + CTA Badge phát sáng + Sản phẩm thật).
* **Số bước**: `2,200 steps`.
* **Tiêu chuẩn nghiệm thu**: Chạy bộ Benchmark 20 test case độc lập đạt tỷ lệ chính xác $\ge \mathbf{95\%}$ tổng thể và $\mathbf{100\%}$ trên các mẫu banner chuẩn.

---

## ⏱️ 5. BẢNG DỰ TOÁN TÀI NGUYÊN & THỜI GIAN (RESOURCE & TIMELINE ESTIMATES)

### 5.1. Dự toán Bộ nhớ VRAM trên 2x NVIDIA A30 (48GB Tổng cộng):

| Thành phần Bộ nhớ | Dung lượng GPU 0 | Dung lượng GPU 1 | Ghi chú Kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Trọng số DiT 4B Base (BF16)** | $8.2\text{ GB}$ | $8.2\text{ GB}$ | Đóng băng 100% gradient |
| **Trọng số LoRA + Optimizer States** | $1.2\text{ GB}$ | $1.2\text{ GB}$ | AdamW states (FP32 master weights) |
| **Latent Cache + Text Embeddings** | $1.5\text{ GB}$ | $1.5\text{ GB}$ | Pre-computed embeddings |
| **Activations (Gradient Checkpointed)** | $6.8\text{ GB}$ | $6.8\text{ GB}$ | Sequence dài 10,208 tokens |
| **CUDA Workspace & PyTorch Overhead** | $1.2\text{ GB}$ | $1.2\text{ GB}$ | Bộ đệm phân mảnh |
| 📊 **TỔNG VRAM SỬ DỤNG MỖI GPU** | **$18.9\text{ GB}$ / $24\text{ GB}$** | **$18.9\text{ GB}$ / $24\text{ GB}$** | 🟢 **Dư an toàn $\approx 5.1\text{ GB}$ Headroom!** |

---

### 5.2. Dự toán Thời gian Thực thi (Timeline Estimates):

```
╔══════════════════════════════════════════════════════════╦══════════════╦═════════════════════════════════╗
║ Hạng Mục Công Việc                                       ║ Thời Gian    ║ Compute / Nhân Lực Cần Thiết    ║
╠══════════════════════════════════════════════════════════╬══════════════╬═════════════════════════════════╣
║ 1. Viết Script Sinh Dataset & Gemini Distillation        ║ 0.5 Ngày     ║ Agent viết code trên Local      ║
║ 2. Sinh tập Dữ liệu Pilot (500 mẫu) qua Gemini API       ║ 2 – 3 Giờ    ║ Chạy nền qua API                ║
║ 3. Xây dựng Pipeline Train LoRA (`train_lora_dit.py`)    ║ 0.5 Ngày     ║ DDP Accelerate trên 2x A30      ║
║ 4. Huấn luyện Milestone A & B (1,800 steps)              ║ ~2.5 Giờ     ║ 2x GPU A30 chạy liên tục        ║
║ 5. Đánh giá & Điều chỉnh Hyperparameters                 ║ 0.5 Ngày     ║ Chạy Benchmark Suite 20 mẫu     ║
║ 6. Huấn luyện Full Milestone C (4,000 steps tổng)        ║ ~5.0 Giờ     ║ 2x GPU A30 chạy qua đêm         ║
║ 7. Đóng gói Inference Pipeline & Upstream LLM Router     ║ 1.0 Ngày     ║ Hoàn thiện sản phẩm End-to-End  ║
╠══════════════════════════════════════════════════════════╬══════════════╬═════════════════════════════════╣
║ 🏆 TỔNG THỜI GIAN HOÀN THÀNH GIAI ĐOẠN 3                ║ **3 – 4 NGÀY**║ Compute Server 2x A30 sẵn sàng  ║
╚══════════════════════════════════════════════════════════╩══════════════╩═════════════════════════════════╝
```

---

## 🛡️ 6. CHIẾN LƯỢC ĐÁNH GIÁ & DỰ PHÒNG RỦI RO (EVALUATION & CONTINGENCY)

### 6.1. Bộ Đánh Giá Tự Động (Automated Evaluation Suite):
* Cứ sau mỗi **500 steps**, trainer tự động tạm dừng 60 giây và sinh ảnh trên **5 mẫu Golden Test Cases** cố định:
  1. *Test 1 (F&B)*: Poster Cafe Grand Opening 3 tầng chữ.
  2. *Test 2 (Tech)*: Poster Tai nghe chống ồn với cụm 4 dấu kép `Ố-Ồ-Ủ-Ộ`.
  3. *Test 3 (Fashion)*: Poster Flash Sale Thời trang cao cấp.
  4. *Test 4 (Literature)*: Bài thơ Tây Tiến 4 câu (kiểm tra độ bền chữ dài).
  5. *Test 5 (Product Anchor)*: Giày Sneaker thật $t=40$ + 2 khối text.
* Toàn bộ ảnh eval được tự động ghép vào panel: **`eval_checkpoints/STEP_XXXX_COMPARISON.png`** để User theo dõi trực quan độ tiến bộ qua từng checkpoint.

---

### 6.2. Kế hoạch Dự phòng Rủi ro (Fallback Mechanisms):

1. **Rủi ro 1: Overfitting trên font chữ của Dataset**:
   * *Triệu chứng*: Mô hình chỉ vẽ đúng font trong tập train, vẽ xấu font lạ.
   * *Giải pháp*: Tăng Dropout lên $0.1$, bổ sung Random Font Augmentation trong DataLoader (chọn ngẫu nhiên trong 8 font hệ thống ở mỗi epoch).
2. **Rủi ro 2: Gradient Spike do độ dài Sequence lớn**:
   * *Triệu chứng*: Loss đột ngột vọt lên `NaN` hoặc `Inf`.
   * *Giải pháp*: Gradient clipping chặt ở mức $0.5$, kích hoạt `torch.cuda.amp.GradScaler` cho bfloat16.
3. **Rủi ro 3: Tràn VRAM khi sequence đạt đỉnh**:
   * *Triệu chứng*: Lỗi `CUDA Out of Memory`.
   * *Giải pháp*: Kích hoạt CPU Offloading cho Optimizer States hoặc tăng Gradient Accumulation lên 8 và giảm batch size.

---

## 🎯 7. KẾT LUẬN & ĐỀ XUẤT PHÊ DUYỆT

Kế hoạch huấn luyện này được thiết kế theo đúng tiêu chuẩn kỹ thuật hàng đầu:
* **Tận dụng $100\%$ thế mạnh của Base 4B** (không xóa bỏ những gì mô hình đã biết).
* **Định vị đúng trọng tâm bài toán** (Attention Disentanglement & Routing).
* **Tiết kiệm tối đa tài nguyên** (Chỉ cần vài giờ compute trên 2x A30 thay vì nhiều ngày).

Bạn vui lòng xem xét toàn bộ bản Kế hoạch chi tiết này. Nếu bạn đồng thuận, chúng ta sẽ bắt tay ngay vào việc lập trình **Script Sinh Dataset Đa Phương Thức từ Gemini (`scripts/generate_distilled_dataset.py`)**!
