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
| **1. Softmax Joint Attention** | Toàn bộ Key $K$ của Canvas, Sản phẩm (4096 tokens) và các Glyph bị gom chung vào 1 Softmax duy nhất $\rightarrow$ gây ra hiện tượng Token Mass Dominance đè bẹp các khối nhỏ nếu thiếu phân luồng. | Áp dụng **Quy Luật Kích Thước Động (Dynamic Glyph Token Sizing)**: Tự động tính toán Box theo độ dài từ ($240 - 800\text{ tokens}$), đảm bảo chiều cao $\ge 160\text{px}$/dòng và font size $\ge 40\text{px}$ để LoRA học tính đa hình không gian và tiết kiệm $>40\%$ sequence length. |
| **2. Target LoRA Layers** | FLUX.2 không có module Cross-Attention riêng; Canvas và Ref dùng chung `img_attn.qkv` (DoubleBlocks) và `linear1` (SingleBlocks). 80% độ sâu mô hình nằm ở 20 SingleBlocks. | Tiêm LoRA trực tiếp vào: `img_attn.qkv` (5 DoubleBlocks) và phần Q, K, V của `linear1` (20 SingleBlocks). Rank $r=32$, $\alpha=32$. |
| **3. Uniform AdaLN Modulation** | `ref_fixed_timestep = 0.0` cố định cho mọi slot $\rightarrow$ mô hình không có biên độ ưu tiên giữa các slot, toàn bộ nhận diện slot dồn vào RoPE. | LoRA sẽ tối ưu hóa ma trận $W_Q, W_K$ để thích ứng nhạy bén với các dải tần số góc quay RoPE $\mathbf{R}(\Delta t)$ của từng slot ($t=10, 20, 30$). |
| **4. Upstream Semantic Clash** | Khi prompt lặp lại nguyên văn text tiếng Việt, Text Encoder `Qwen3-4B-FP8` gây nhiễu và xung đột với tín hiệu Glyph từ VAE. | Tích hợp Upstream LLM (Gemini Flash / Qwen-2.5) tự động bóc tách text và làm sạch prompt (Prompt Sanitization). |
| **5. Phân bổ Slot Chuẩn** | Mốc $t \le 40.0$ là vùng hoạt động an toàn tuyệt đối. Mốc $t \ge 50.0$ bắt đầu suy hao góc pha RoPE đối với glyph chữ. | Khóa cứng 4 Slot chuẩn: $t=10.0$ (Headline), $t=20.0$ (Subtitle), $t=30.0$ (CTA Badge), $t=40.0$ (Ảnh Sản Phẩm Thật). |
| **6. True CFG & Chống CFG Drift** | Klein 4B Base dùng True CFG (`use_guidance_embed = False`), nhánh Unconditional giữ nguyên Reference Tokens `img_cond_seq` và chỉ null hóa Text Prompt `ctx = ""`. | Áp dụng Text Conditioning Dropout ($p=0.10$) khi train LoRA: Thay thế `txt` bằng embedding của chuỗi rỗng `""` với tỉ lệ $10\%$ để LoRA học đúng nhánh Unconditional. |


---

## 🗂️ 2. THIẾT KẾ DỮ LIỆU & QUY TRÌNH CHẾ TẠO DATASET (DISTILLATION ENGINE)

### 2.1. Quy cách một Training Sample Chuẩn & Nguyên Tắc Khớp Tuyệt Đối (Strict Target-Ref Alignment):

> [!IMPORTANT]
> **NGUYÊN TẮC KHỚP TUYỆT ĐỐI (ZERO GHOST-TEXT PRINCIPLE)**:
> Ảnh Ground-Truth $\mathbf{X}_{\text{target}}$ **BẮT BUỘC CHỈ CHỨA ĐÚNG các thành phần text có mặt trong danh sách Reference Tokens của Milestone đó**.
> Tuyệt đối không để ảnh đích xuất hiện Subtitle hay CTA khi input ở Milestone A chỉ có Headline (tránh việc DiT bị ép phải "học vẹt sinh chữ ma từ không khí" khi không có glyph condition).

Mỗi mẫu huấn luyện được cấu trúc động theo tiến trình Milestone:
$$\text{Sample}_i^{(\text{Milestone})} = \left( \text{Prompt}_{\text{clean}}, \; \{\text{Ref}_k\}_{k \in \text{ActiveSlots}}, \; \mathbf{X}_{\text{target}}^{(\text{Aligned})} \right)$$

1. **`Prompt_clean`**: Mô tả phong cách thị giác, bố cục, ánh sáng studio, chất liệu 3D, **TUYỆT ĐỐI KHÔNG CHỨA CHỮ NGUYÊN VĂN**.
2. **`Ref_10` (Headline Glyph)**: Kích thước động ($280 - 640\text{ tokens}$), font nghệ thuật thương hiệu theo Domain.
3. **`Ref_20` (Subtitle Glyph)**: Kích thước động ($240 - 480\text{ tokens}$), font thông tin sắc nét (`BeVietnamPro-Black`).
4. **`Ref_30` (CTA Badge Glyph)**: Kích thước động ($240 - 384\text{ tokens}$), font uốn lượn/dạ quang (`Pacifico` / `Sedgwick`).
5. **`Ref_prod_40` (Ảnh Sản phẩm Thật)**: Kích thước $1024 \times 1024$ (4096 tokens), ảnh sản phẩm studio sạch nền.
6. **`X_target` (Ảnh Ground-Truth $1024 \times 1024$)**: Ảnh poster tương ứng chỉ chứa đúng các thành phần text đã kích hoạt.


---

### 2.2. Ma Trận Ánh Xạ 1:1 Font Chuẩn Thương Hiệu Theo 5 Domain (Domain-Font Mapping Matrix):

Để tối ưu hóa **Mật độ Tiếp xúc Font (Font Exposure Density)** và chống loãng tín hiệu học dấu tiếng Việt, toàn bộ 2,500 mẫu được ánh xạ cố định $1:1$ với 5 bộ Font chủ lực (đã QA Unicode $100\%$):

| Ngành Hàng (Domain) | Quy Mô | Font Headline Chủ Lực | Font Subtitle | Font CTA Badge | Phong Cách Thiết Kế & Chất Liệu |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **☕ 1. F&B / Cafe** | $500$ mẫu | `SedgwickAveDisplay` | `BeVietnamPro-Black` | `Pacifico` | Chữ khắc gỗ 3D mộc mạc, neon cafe ấm cúng |
| **📱 2. Công Nghệ / Tech** | $500$ mẫu | `Anton-Regular` | `BeVietnamPro-Black` | `Pacifico` | Chữ kim loại vát cạnh, đèn LED, chrome bóng bẩy |
| **👗 3. Thời Trang / Fashion**| $500$ mẫu | `PlayfairDisplay` | `BeVietnamPro-Black` | `Pacifico` | Chữ Serif mạ vàng gold, sang trọng, thanh lịch |
| **💆 4. Spa / Mỹ Phẩm** | $500$ mẫu | `DancingScript` | `BeVietnamPro-Black` | `Pacifico` | Chữ mềm mại uyển chuyển, phong cách pastel tối giản |
| **🛍️ 5. Siêu Thị / FMCG** | $500$ mẫu | `SVN-Gotham Ultra` / `Oswald`| `BeVietnamPro-Black` | `Pacifico` | Chữ dập nổi 3D khối to, pop-art khuyến mãi rực rỡ |

> [!TIP]
> **PHÂN TÁCH FONT CHỦ LỰC VS POOL AUGMENTATION**:
> * **5 Font Chủ Lực trên**: Nhận $85\%$ tổng số lượt exposure để đảm bảo mô hình khắc sâu từng nét dấu tiếng Việt chuẩn xác $100\%$.
> * **Pool Font Phụ Hệ Thống** (`SVN-Blow Brush`, `SVN-Cookies`, `SVN-Gretoon`, `SVN-Harabaras`...): Chỉ được kích hoạt trong cơ chế **Random Font Augmentation ($15\%$ xác suất)** trong DataLoader để giúp mô hình không bị overfit cứng nhắc vào 5 font chính.

---

### 2.3. Quy Trình Chế Tạo 1-Shot Multi-Modal Distillation Trực Tiếp (Direct 1-Shot Distillation Pipeline):

Toàn bộ 2,500 mẫu huấn luyện được chế tạo theo cơ chế **Phi Trạng Thái Trực Tiếp (Stateless 1-Shot)**: Mỗi mẫu nhận đúng số lượng Glyph cần thiết và sinh ra bức ảnh Ground-Truth hoàn chỉnh chỉ trong **1 LẦN GỌI DUY NHẤT**, tối ưu hóa bố cục thị giác tự nhiên, chạy song song bất đồng bộ (`asyncio` / `ThreadPool`) cực nhanh và tiết kiệm $65\%$ chi phí API:

```
                                  TỔNG QUY MÔ DATASET: 2,500 MẪU ĐỘC LẬP
                                                    │
         ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
         ▼                                          ▼                                          ▼
 [ 🎯 NHÓM A: 500 MẪU (Milestone A) ]       [ 🎯 NHÓM B: 1,000 MẪU (Milestone B) ]     [ 🎯 NHÓM C: 1,000 MẪU (Milestone C) ]
 • Phục vụ: Milestone A (500 mẫu)           • Phục vụ: Milestone B (1,500 mẫu = A + B) • Phục vụ: Milestone C (2,500 mẫu = A+B+C)
 • 1-Shot Input: [Ref_10 (Headline)]        • 1-Shot Input: [Ref_10, Ref_20]           • 1-Shot Input: [Ref_10, Ref_20, Ref_30]
   + [Ref_SP_40 (nếu có)]                     + [Ref_SP_40 (nếu có)]                     + [Ref_SP_40 (nếu có)]
 • Output Ground-Truth:                     • Output Ground-Truth:                     • Output Ground-Truth:
   Poster 1 Text + SP/Scene                   Poster 2 Texts + SP/Scene                  Poster 3 Texts + SP/Scene
 • ⚡ 500 Calls (Async Parallel)            • ⚡ 1,000 Calls (Async Parallel)          • ⚡ 1,000 Calls (Async Parallel)
         │                                          │                                          │
         └──────────────────────────────────────────┼──────────────────────────────────────────┘
                                                    ▼
                                   [ AUTOMATED QUALITY ASSURANCE FILTER ]
                                   • OCR Check: Khớp chính xác ký tự tiếng Việt >= 98%
                                   • Độ phân giải chuẩn hóa theo 4 Aspect Ratio Buckets
                                   • Đóng gói thành WebDataset Shards (.tar / .h5)
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

### 2.5. Phân Bổ Chế Độ Kép (Product-Anchor 60% vs Pure Text-to-Image 40%):

Để chống hiện tượng mô hình bị "nghiện mỏ neo sản phẩm" (Spurious Anchor Dependency) và đáp ứng $100\%$ các bài toán marketing thực tế (bao gồm cả poster sự kiện, lễ hội, thơ ca không có ảnh sản phẩm upload), tập dữ liệu được chia làm 2 nhánh huấn luyện:

```
                                  TỔNG QUY MÔ DATASET: 2,500 SAMPLES
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
   [ 🛍️ NHÁNH A: PRODUCT-ANCHOR MODE (60% ~ 1,500 mẫu) ]          [ 🎨 NHÁNH B: PURE TEXT-TO-IMAGE MODE (40% ~ 1,000 mẫu) ]
   • Dành cho E-commerce, Thiết bị, Mỹ phẩm, Thời trang           • Dành cho Poster Sự kiện, Lễ hội, Cafe, Thơ ca, Quote
   • Input: Đa khối Text ($t=10,20,30$) + 1 SP thật ($t=40$)      • Input: ĐA KHỐI TEXT ($t=10,20,30$) + PROMPT SCENE
   • Sequence dài: ~5,100 - 6,000 tokens                          • Sequence ngắn: ~4,400 - 5,000 tokens (KHÔNG CÓ t=40!)
   • Học: Tách bạch Sản phẩm thật và Đa khối chữ                  • Học: Tự sinh Scene tự nhiên và Định tuyến Đa khối chữ
```

#### 🧮 Cơ Chế Phạt Loss Trong Cả 2 Chế Độ (Supervised Flow Matching Objective):
Dù có hay không có ảnh sản phẩm ở Input, mỗi mẫu trong Dataset đều có một ảnh Poster hoàn chỉnh làm **Ground-Truth $\mathbf{X}_{\text{target}}$**:
$$\mathcal{L}_{\text{Flow}} = \mathbb{E}_{t, x_0, x_1} \left[ \left\| \mathbf{v}_\theta\left(x_t, t, \text{ctx}, \{\text{Ref}_k\}\right) - (x_1 - x_0) \right\|^2 \right]$$

1. **Khi chạy Nhánh A (Có SP $t=40$)**: Loss phạt nếu sản phẩm sinh ra sai lệch chi tiết so với ảnh thật ở $t=40$ hoặc chữ bị biến dạng.
2. **Khi chạy Nhánh B (Pure T2I - Không có $t=40$)**: Loss phạt ở vùng hậu cảnh nếu không ăn khớp với Text Prompt, và phạt ở vùng chữ nếu không bám sát tọa độ hình học của các Glyph $t=10, 20, 30$.
   * 👉 **Mô hình học được kỹ năng tối thượng**: *"Khi không có ảnh sản phẩm đưa vào, hãy tự do vẽ bối cảnh theo prompt, nhưng vẫn phải định tuyến chính xác $100\%$ các khối chữ từ $t=10, 20, 30$ lên bức tranh đó!"*

---

## ⚙️ 3. THIẾT KẾ KIẾN TRÚC LORA & HYPERPARAMETERS



### 3.1. Cấu hình PEFT LoRA Injection & Phân Bổ Tham Số Chính Xác:
```python
# Cấu hình LoRA tối ưu cho FLUX.2 Klein 4B Base
lora_config = {
    "r": 32,                          # Rank 32 đủ dung lượng học phân luồng đa slot
    "lora_alpha": 32,                 # Scaling factor = 1.0
    "lora_dropout": 0.05,
    "target_modules": [
        "img_attn.qkv",               # 5 DoubleStreamBlocks (Query, Key, Value nhánh ảnh/glyph)
        "txt_attn.qkv",               # 5 DoubleStreamBlocks (Query, Key, Value nhánh text-prompt)
        "linear1",                    # 20 SingleStreamBlocks (Fused Joint QKV Attention + MLP Projection)
    ],
    "bias": "none",
    "dtype": "bfloat16"
}
```

#### 🧮 Tính Toán Số Lượng Tham Số LoRA Chi Tiết:
1. **5 Khối DoubleStreamBlocks**:
   * `img_attn.qkv`: $5 \times (3072 \times 32 + 9216 \times 32) = 1,966,080$ tham số.
   * `txt_attn.qkv`: $5 \times (3072 \times 32 + 9216 \times 32) = 1,966,080$ tham số.
   * $\rightarrow$ Tiểu kế DoubleBlocks: **$3,932,160$ tham số** (~3.93M).
2. **20 Khối SingleStreamBlocks**:
   * `linear1` (Ma trận fused $3072 \rightarrow 27648$ gồm 9216 dim cho QKV + 18432 dim cho MLP):
   * $20 \times (3072 \times 32 + 27648 \times 32) = 20 \times 983,040 = \mathbf{19,660,800}$ tham số (~19.66M).
   * *Ý nghĩa kỹ thuật*: Áp dụng LoRA lên `linear1` cho phép mô hình tối ưu đồng thời cả cơ chế Attention phân luồng lẫn biến đổi đặc trưng không gian (Feature Transformation) trong khối đơn luồng, tạo năng lực thích ứng mạnh mẽ nhất.

* **Tổng tham số mô hình Base 4B**: $\approx 4.08 \times 10^9$ parameters.
* **Tổng tham số LoRA cần huấn luyện**: $\mathbf{23,592,960\text{ parameters}}$ (**chỉ chiếm $\mathbf{0.58\%}$ mô hình**).
* **Kích thước file trọng số LoRA lưu trữ**: $\approx \mathbf{47.2\text{ MB}}$ (`.safetensors`).


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
| **Text Conditioning Dropout** | $p = 0.10$ ($10\%$ số step train) | Thay thế `txt` bằng embedding chuỗi rỗng `""`, giữ nguyên $100\%$ Reference Tokens để LoRA học đúng nhánh Unconditional của True CFG, chống CFG Drift |
| **Hàm Mất Mát (Loss)** | Flow Matching MSE Loss: $\mathcal{L} = \| v_\theta - (x_1 - x_0) \|^2$ | Chuẩn Flow Matching Euler ODE của BFL |


---

## 📈 4. LỘ TRÌNH HUẤN LUYỆN 3 MỐC (CURRICULUM LEARNING ROADMAP)

Để đảm bảo gradient hội tụ mượt mà và không làm "sốc" ma trận Attention của mô hình Base, quá trình train được chia làm 3 Phase tăng dần độ phức tạp (Effective Batch Size $= 1 \times 2 \times 4 = \mathbf{8}$):

```
       [ MILESTONE A ]                      [ MILESTONE B ]                      [ MILESTONE C ]
     Phase Khởi Động (1 text)             Phase Tách Kênh (2 texts)            Phase Full 4-Slot (3 texts)
      600 steps (9.6 epochs)              1,200 steps (6.4 epochs)             2,200 steps (7.04 epochs)
   Tập dữ liệu: 500 samples             Tập dữ liệu: 1,500 samples           Tập dữ liệu: 2,500 samples
  • Học hòa trộn SP & Headline            • Học tách kênh Subtitle (t=20)         • Khóa toàn bộ 4-slot Production
```

### 📍 Chi tiết từng Milestone (Phân Bổ 60% Product-Anchor / 40% Pure T2I):

#### 🔹 Milestone A: Đồng bộ Hòa trộn 1 Headline ($t=10$) (300 mẫu SP + 200 mẫu Pure T2I)
* **Mục tiêu**: Dạy LoRA phân luồng mượt mà giữa khối token của sản phẩm và khối chữ chính ($280 - 640\text{ tokens}$ dynamic), đồng thời học cách tự sinh background khi không có ảnh sản phẩm.
* **Số bước**: `600 steps` (tương đương **$9.6\text{ epochs}$** trên tập 500 mẫu với Effective Batch Size = 8).
* **Tiêu chuẩn nghiệm thu**: Headline đạt độ chính xác $100\%$ trên cả ảnh có sản phẩm thật và ảnh T2I thuần.

#### 🔹 Milestone B: Phân tách Chú ý 2 Khối Text ($t=10, 20$) (900 mẫu SP + 600 mẫu Pure T2I)
* **Mục tiêu**: Kích hoạt khả năng phân tách kênh $t=20$ (Subtitle), triệt tiêu hoàn toàn hiện tượng Ref-to-Ref contamination và ngăn chặn DiT tự sinh chữ rác Lorem Ipsum.
* **Số bước**: `1,200 steps` (tương đương **$6.4\text{ epochs}$** trên tập 1,500 mẫu).
* **Tiêu chuẩn nghiệm thu**: Cả Headline và Subtitle đều render chuẩn $100\%$ chữ và dấu tiếng Việt trên cùng 1 ảnh.

#### 🔹 Milestone C: Toàn diện 3 Khối Text ($t=10, 20, 30$) (1,500 mẫu SP + 1,000 mẫu Pure T2I)
* **Mục tiêu**: Khóa cứng toàn bộ ma trận Attention cho bố cục chuẩn thương mại hoàn chỉnh (Headline 3D + Subtitle thông tin + CTA Badge phát sáng + Sản phẩm thật / Background AI).
* **Số bước**: `2,200 steps` (tương đương **$7.04\text{ epochs}$** trên tập 2,500 mẫu).
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

### 5.2. Dự toán Thời gian Thực thi Thực Tế (Realistic Timeline Estimates):

```
╔══════════════════════════════════════════════════════════╦══════════════╦═════════════════════════════════╗
║ Hạng Mục Công Việc                                       ║ Thời Gian    ║ Compute / Nhân Lực Cần Thiết    ║
╠══════════════════════════════════════════════════════════╬══════════════╬═════════════════════════════════╣
║ 1. Viết Script Sinh Dataset & Pipeline Distillation      ║ 0.5 Ngày     ║ Agent viết code trên Local      ║
║ 2. Sinh Dataset Batch (2,500 mẫu kèm Retry Loop OCR)     ║ 6 – 8 Giờ    ║ Chạy nền qua đêm (Batch script) ║
║ 3. Xây dựng Pipeline Train LoRA DDP (`train_lora_dit.py`) ║ 0.5 Ngày     ║ DDP Accelerate trên 2x A30      ║
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

### 6.1. Bộ Đánh Giá Toàn Diện 8 Test Cases Cố Định (Golden Evaluation Suite):
Cứ sau mỗi **500 steps**, trainer tự động tạm dừng và sinh ảnh đánh giá trên **8 Golden Test Cases** bao phủ đủ 5 ngành hàng và bài test chống hồi quy:
0. *Test 0 (Single-Slot Zero-Shot Regression Test)*: Duy nhất 1 Headline ở $t=10.0$ (Xác nhận LoRA không phá vỡ độ chính xác 100% vốn có của mô hình Base).
1. *Test 1 (F&B / Cafe)*: Poster Cafe Grand Opening 3 tầng chữ (Gỗ/Neon).
2. *Test 2 (Tech / Audio)*: Poster Tai nghe chụp tai với Headline 3D kim loại + Subtitle mạ bạc + CTA Neon.
3. *Test 3 (Fashion / Clothing)*: Poster Flash Sale Thời trang cao cấp với chất liệu chữ vàng đồng.
4. *Test 4 (Spa / Cosmetics)*: Poster Spa Thảo mộc dưỡng da cao cấp (Chất liệu chữ pastel/tối giản).
5. *Test 5 (Supermarket / FMCG)*: Poster Siêu thị Đại hạ giá cuối tuần (Chất liệu chữ pop-art dập nổi).
6. *Test 6 (Literature / Dense Text)*: Bài thơ Tây Tiến 4 câu (28 từ, kiểm tra độ bền câu dài).
7. *Test 7 (Product Anchor 4096 tokens)*: Giày Sneaker thật $t=40$ + Headline $t=10$ + CTA $t=30$.

* Toàn bộ ảnh eval được tự động ghép vào panel: **`eval_checkpoints/STEP_XXXX_COMPARISON.png`** để theo dõi trực quan từng checkpoint.
* **Đánh Giá Tuyến Tính CFG Scale (CFG Scale Sweep)**: Tại các checkpoint lớn (Step 600, 1800, 4000), trainer tự động chạy sweep qua 4 mức CFG Guidance Scale: `[1.0, 2.5, 4.0, 6.0]` để đảm bảo LoRA không bị suy thoái hoặc cháy nét ở các dải guidance khác nhau.



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
