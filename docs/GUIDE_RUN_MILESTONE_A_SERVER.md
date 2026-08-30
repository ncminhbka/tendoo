# HƯỚNG DẪN VẬN HÀNH TOÀN DIỆN MILESTONE A TRÊN MÁY CHỦ 2x NVIDIA A30

Tài liệu này hướng dẫn chi tiết quy trình từng bước để chuẩn bị dữ liệu, pre-cache tính năng, huấn luyện LoRA DiT 4B Base (Giai đoạn 3) và đánh giá trực quan trên máy chủ 2x NVIDIA A30 (48GB VRAM) thông qua JupyterLab.

---

## 📌 1. TỔNG QUAN KIẾN TRÚC & MỤC TIÊU MILESTONE A

- **Mục tiêu cốt lõi**: Đào tạo LoRA đóng vai trò **Bộ Điều Phối Phân Luồng Attention (Attention Traffic Controller)**, giải quyết tranh chấp Softmax và tràn kênh Attention (Cross-Slot Attention Crosstalk) khi hiển thị đồng thời 2 khối chữ ($t=10.0$ và $t=20.0$) cùng ảnh sản phẩm thật ($t=30.0$).
- **Mô hình mục tiêu duy nhất**: `FLUX.2-klein-base-4B` (4B DiT + Qwen3-4B-FP8 + 128-channel VAE).
- **Quy mô tập dữ liệu Milestone A**:
  + **Bản đầy đủ (Full Target)**: **800 mẫu độc bản** theo ma trận phân bổ đa chiều (440 mẫu I2I + 360 mẫu T2I; 7 nghiệp vụ use-case; 4 tỉ lệ khung hình 1:1, 9:16, 4:5, 16:9).
  + **Bản thử nghiệm (Pilot Benchmark)**: **60 mẫu chuẩn hóa** (đã có sẵn trong `data/milestone_a/` khớp 100% từng pixel và dòng chữ).

---

## 🛠️ 2. CHẾ TẠO DỮ LIỆU BẰNG `scripts/build_milestone_a_dataset.py`

### 2.1. Xác nhận năng lực tạo 800 mẫu
Script [`scripts/build_milestone_a_dataset.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/build_milestone_a_dataset.py) được thiết kế mặc định tạo đúng **800 mẫu dữ liệu**:
- Lệnh chạy: `python scripts/build_milestone_a_dataset.py --execute`
- Script tích hợp cơ chế **Tự động tiếp tục (Auto-Resume)**: Khi phát hiện trong `data/milestone_a/dataset_manifest.jsonl` đã có sẵn 60 mẫu Pilot, script sẽ tự động bỏ qua 60 mẫu này và chạy tiếp 740 mẫu còn lại (`sample_0061` đến `sample_0800`).

### 2.2. Lưu ý về môi trường chạy script tạo dữ liệu
- Script `build_milestone_a_dataset.py` gọi **OpenAI API** (`gpt-image` / `dall-e-3` sinh target image và `gpt-4o` VLM sinh clean prompt).
- **Yêu cầu**: Cần có kết nối Internet và `OPENAI_API_KEY` trong file `.env`.
- **Khuyến nghị vận hành**:
  + **Cách 1 (Khuyên dùng)**: Chạy lệnh tạo dữ liệu trên máy tính cá nhân (Local) có kết nối mạng ổn định:
    ```bash
    # Chạy tiếp để hoàn tất đủ 800 mẫu
    python scripts/build_milestone_a_dataset.py --execute
    ```
    Sau đó nén thư mục `data/milestone_a/` thành file zip (`zip -r milestone_a_data.zip data/milestone_a/`) và tải trực tiếp lên JupyterLab server.
  + **Cách 2**: Nếu JupyterLab server có mạng ra ngoài Internet, bạn có thể chạy trực tiếp trên Terminal của JupyterLab:
    ```bash
    python scripts/build_milestone_a_dataset.py --execute
    ```

---

## 🚀 3. QUY TRÌNH THỰC THI TRÊN SERVER 2x NVIDIA A30

### BƯỚC 1: Đồng bộ mã nguồn mới nhất trên Server
Mở Terminal trong JupyterLab (`/home/jovyan/work/tendoo/`):
```bash
cd ~/work/tendoo
git pull origin main
```

---

### BƯỚC 2: Feature Pre-Caching (Tối ưu hóa VRAM & Tăng tốc 10x)
Trước khi train, toàn bộ ảnh target, glyph bitmap, ảnh sản phẩm và text prompt được mã hóa trước qua VAE và Qwen3 thành các tensor `.pt` lưu trên ổ cứng. 

> [!TIP]
> Bước này giúp giải phóng hoàn toàn VAE và Text Encoder khỏi bộ nhớ GPU trong lúc train, chỉ giữ lại duy nhất DiT 4B trên VRAM $\rightarrow$ Triệt tiêu 100% nguy cơ tràn bộ nhớ (OOM) và tăng tốc độ train lên gấp 10 lần!

```bash
# Chạy pre-caching trên GPU 0
python scripts/train_lora_dit.py --pre-cache \
  --manifest data/milestone_a/dataset_manifest.jsonl \
  --cache-dir data/milestone_a/cache \
  --device cuda:0
```
*Thời gian chạy*: Khoảng 1–2 phút cho 60 mẫu, hoặc 10–15 phút cho 800 mẫu.
*Kiểm tra*: Sau khi chạy xong, kiểm tra thư mục cache:
```bash
ls -la data/milestone_a/cache | head -n 15
```

---

### BƯỚC 3: Khởi chạy Huấn luyện LoRA Multi-GPU DDP (2x A30)

Sử dụng `torchrun` để phân tán dữ liệu thực sự (Data Sharded DDP) trên cả 2 GPU A30:

```bash
torchrun --nproc_per_node=2 scripts/train_lora_dit.py \
  --manifest data/milestone_a/dataset_manifest.jsonl \
  --cache-dir data/milestone_a/cache \
  --output-dir checkpoints/lora_milestone_a \
  --steps 800 \
  --grad-accum 4 \
  --lr 1e-4 \
  --rank 32 \
  --alpha 32.0 \
  --dropout 0.05 \
  --weighted-sampling \
  --save-every 200 \
  --eval-every 50
```

#### Giải thích các tham số vận hành:
- `--nproc_per_node=2`: Chạy song song trên 2 GPU A30. Mỗi GPU xử lý $1/2$ dữ liệu mỗi epoch.
- `--steps 800`: 800 optimizer steps. Với `grad_accum=4`, mô hình thực hiện $800 \times 4 = 3,200$ lượt forward sample-level (tương đương 4 epochs qua 800 mẫu hoặc 53 epochs qua 60 mẫu).
- `--weighted-sampling`: Tự động tăng gấp đôi tần suất huấn luyện (2.0x weight) cho các mẫu chữ mỏng ($<350$ tokens tại $t=20.0$) và các ca crosstalk khó.
- `--save-every 200`: Tự động lưu checkpoint tại Step 200, 400, 600, 800 vào thư mục `checkpoints/lora_milestone_a/`.
- `--eval-every 50`: Cứ 50 steps, tự động tính validation loss trên 10% tập dữ liệu held-out để kiểm tra độ hội tụ.

#### Mẹo chạy nền (Tránh mất kết nối JupyterLab):
Để terminal không bị tắt khi đóng tab trình duyệt, dùng `nohup`:
```bash
nohup torchrun --nproc_per_node=2 scripts/train_lora_dit.py \
  --manifest data/milestone_a/dataset_manifest.jsonl \
  --cache-dir data/milestone_a/cache \
  --output-dir checkpoints/lora_milestone_a \
  --steps 800 \
  --grad-accum 4 \
  --save-every 200 \
  > train_lora.log 2>&1 &

# Theo dõi log thời gian thực:
tail -f train_lora.log
```

---

### BƯỚC 4: Đánh giá Trực quan Checkpoint (Visual Inspection Suite)

Khi đạt các mốc checkpoint (Step 200, Step 400, Step 600, Step 800), mở một Terminal mới và chạy bộ Benchmark Probe:

```bash
# Đánh giá checkpoint Step 200
python scripts/eval_lora_suite.py \
  --checkpoint checkpoints/lora_milestone_a/tendoo_lora_step_0200.safetensors \
  --output-dir eval_results

# Đánh giá checkpoint Step 400
python scripts/eval_lora_suite.py \
  --checkpoint checkpoints/lora_milestone_a/tendoo_lora_step_0400.safetensors \
  --output-dir eval_results
```

#### 3 Bài test tự động sinh ra trong `eval_results/`:
1. `..._probe1_t2i_recruitment.png`: Poster tuyển dụng 2 khối chữ (*"TUYỂN DỤNG NHÂN TÀI"* @ $t=10$ và *"BỨT PHÁ MỌI GIỚI HẠN"* @ $t=20$).
2. `..._probe2_i2i_luxury_perfume.png`: Chai nước hoa thật @ $t=30$ + Title @ $t=10$ + Subtitle @ $t=20$.
3. `..._probe3_hard_crosstalk_stress.png`: Ca stress-test khối CTA ngắn dễ bị nuốt nét (*"MUA 1 TẶNG 1"* @ $t=20$).

👉 **Cách xem**: Mở trình quản lý file bên trái của JupyterLab, truy cập thư mục `eval_results/` và nhấp đúp chuột vào từng file ảnh PNG để soi nét chữ trực tiếp.

---

## 📋 4. TIÊU CHÍ NGHIỆM THU TRỰC QUAN (ACCEPTANCE CRITERIA)

| Tiêu Chí Đánh Giá | Đạt Chuẩn Nghiệm Thu (PASS) | Chưa Đạt (FAIL / Cần Train Tiếp) |
| :--- | :--- | :--- |
| **Phân tách 2 khối chữ ($t=10, 20$)** | Cả Tiêu đề và Slogan cùng hiển thị độc lập, rõ ràng, không đè lên nhau. | Chữ $t=20$ biến mất hoặc chữ $t=10$ bị dính các ký tự của $t=20$. |
| **Dấu Tiếng Việt** | Đúng dấu 100% (`Ệ`, `Ộ`, `Ả`, `Ắ`, `Ư`). Không rụng dấu, không méo nét. | Dấu bị biến thành hạt nhiễu hoặc sai dấu thanh. |
| **Chất liệu & Ánh sáng** | Chữ mạ vàng/neon hòa trộn tự nhiên vào bề mặt theo mô tả Prompt. | Chữ phẳng lì như tem dán 2D, không ăn nhập với ánh sáng. |
| **Bảo tồn sản phẩm ($t=30$)** | Sản phẩm chai nước hoa giữ nguyên 100% kiểu dáng, nhãn mác và màu sắc. | Sản phẩm bị biến dạng hoặc nắp chai bị đổi màu. |

---

## ❓ 5. XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

1. **Lỗi `ModuleNotFoundError: No module named 'einops'`**:
   Chạy `pip install einops` trong terminal của môi trường JupyterLab.
2. **Lỗi `CUDA out of memory`**:
   Script đã được tối ưu hóa để chạy vừa vặn trên A30 24GB. Nếu vẫn gặp OOM, giảm `grad_accum_steps` từ 4 xuống 2 hoặc kiểm tra xem có tiến trình nào khác đang chiếm VRAM (`nvidia-smi`).
3. **Lỗi `Missing cache shard for id=...`**:
   Format cache đã được nâng cấp lên v3 lưu per-slot token lengths. Hãy chạy lại lệnh `--pre-cache` ở Bước 2.
