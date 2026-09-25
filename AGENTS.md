# HƯỚNG DẪN CHO AGENT — REPO `tendoo-v3`

> Đọc file này trước, rồi đọc [ROADMAP.md](ROADMAP.md) trước khi sửa bất kỳ dòng code nào.

---

## 1. MÔI TRƯỜNG & PHÂN VÙNG THỰC THI

### 💻 Máy cá nhân (Windows, KHÔNG GPU)

Dùng để: đọc/phân tích mã nguồn, viết code, debug, đóng gói, quản lý Git, **và chạy toàn bộ
tầng render chữ** (Playwright + Chromium chạy trên CPU rất tốt — mọi phép đo cỡ chữ, tương
phản, tràn chữ đều làm được ở đây).

**TUYỆT ĐỐI KHÔNG** chạy trên máy này: tải checkpoint DiT 4B, inference diffusion, train LoRA,
load weights lớn.

### 🚀 Máy chủ (2× NVIDIA A30, 24GB VRAM mỗi card, truy cập qua JupyterLab)

```
~/ (/home/jovyan/)
├── persistent-data/
│   └── FLUX.2-klein-base-4B/
│       ├── flux-2-klein-base-4b.safetensors   (7.3GB — weights DiT chuẩn BFL)
│       ├── text_encoder/                      (Qwen3-4B-FP8)
│       ├── tokenizer/
│       ├── vae/
│       └── transformer/
└── work/                                      <- clone repo tendoo-v3 ở đây
```

Dùng để: inference diffusion, huấn luyện, xuất log/ảnh/checkpoint.
Mã nguồn chuyển qua **GitHub** (push từ local → pull trong `work/`) hoặc file ZIP.

### Quy trình chuẩn

```
[1] Local: viết code hoàn chỉnh, tự chứa, có script tự động
      ↓
[2] Commit & push (hoặc đóng gói ZIP)
      ↓
[3] Server: chạy trên 2× A30 qua JupyterLab
      ↓
[4] Local: nhận log/ảnh về, agent đọc và phân tích phản biện
```

Mọi script chạy trên server phải **tự chứa**, có xử lý exception, hỗ trợ CUDA, tối ưu cho
Ampere (BF16/FP16, DDP).

---

## 2. KIẾN TRÚC HIỆN TẠI

- **Chữ**: 100% render bằng **HTML/CSS overlay** (Jinja2 + Chromium headless qua Playwright).
  DiT **không** nhận glyph bitmap nào. Không bao giờ để diffusion vẽ chữ.
- **Diffusion**: chỉ sinh **ảnh nền/bối cảnh**. `velocity_blending.py` đồng tiến hoá 2 luồng
  prompt (Scene + Corridor) từ cùng một nhiễu, trộn theo `v = (1-M)·v_scene + M·v_corridor`,
  mục đích là chừa ra **vùng negative space an toàn** để overlay chữ lên trên — KHÔNG phải để
  vẽ chữ trong vùng đó.
- **Model**: mặc định `FLUX.2-klein-4B` bản **Distilled** (`--model distill`, `num_steps=8`,
  `guidance=1.5`). Bản Base (`--model base`, 50 bước, `guidance=4.0`) là lựa chọn thứ hai.
- **Text encoder**: `Qwen3-4B-FP8` (3 tầng `[9, 18, 27]` → context dim 7680), mã hoá prompt
  scene/corridor. Hoàn toàn tách biệt với việc dùng LLM API ngoài cho `llm_planner.py`.
- **VAE**: 128 latent channels, nén 16×.
- **In-Context Reference Product**: `load_and_encode_ref_image` gán mốc RoPE `t=10.0` để giữ
  ảnh sản phẩm thật. **Chỉ dùng cho ảnh sản phẩm, không dùng cho chữ.**

---

## 3. NGUYÊN TẮC GIAO TIẾP

**Phong cách đồng nghiệp phản biện**: khách quan, trung thực 100%, không nịnh, không lạc quan
tếu. Sẵn sàng chỉ ra lỗ hổng toán học, rủi ro bộ nhớ, sai số kiến trúc — kể cả khi điều đó
bác bỏ đề xuất của chính mình ở lượt trước.

---

## 4. NĂM NGUYÊN TẮC KỸ THUẬT BẤT BIẾN

Mỗi nguyên tắc dưới đây đều đã trả giá bằng một lỗi thật. Chi tiết và số đo: [ROADMAP.md](ROADMAP.md).

### 4.1. Đóng băng `src/flux2/`

Giữ nguyên 100% mã gốc BFL (`model.py`, `sampling.py`, `autoencoder.py`). Can thiệp trực tiếp
gây hồi quy ngầm, phá giả định toán học và làm mất mốc đối chứng khi debug. Mọi mở rộng của
Tendoo phát triển ở tầng ngoài, chỉ gọi API chuẩn (`model.forward()`, `ae.encode()`, `ae.decode()`).

### 4.2. Không đưa thông số kích thước/tỉ lệ vào prompt

Chuỗi như `"9:16"`, `"16:9"`, `"8k"`, `"4k"`, `"1080p"` khiến Qwen3 hiểu nhầm là chữ cần hiển
thị → DiT vẽ số rác lên ảnh. Kích thước **chỉ** khai báo qua tham số `--width`/`--height`.

### 4.3. Không đưa nội dung chữ thật vào prompt diffusion

Qwen3 xử lý chuỗi trong ngoặc kép như một tác vụ text-to-image, nhưng nó **không hiểu đúng
hình dạng nét chữ tiếng Việt**. Kết quả: DiT nhận hai tín hiệu "cần vẽ chữ" cạnh tranh nhau và
làm hỏng cả những thứ vốn đang đúng. Prompt chỉ mô tả **bối cảnh, chất liệu, ánh sáng**.

### 4.4. Mọi phát biểu về cỡ chữ phải ĐO qua Chromium

Ước lượng từ hằng số `max_font` cho ra 2.2x; đo thật chỉ **1.62x** — vì hero bị ngân sách
**chiều cao** khoá trước khi chạm trần. Không suy diễn, hãy render và đọc DOM.

### 4.5. Không thêm hằng số hiệu chỉnh tay nếu chưa có phép đo

Mọi hằng số trong `geometry.py`/`renderer.py` phải trỏ được về một probe cụ thể. Bài học:
`_DENSITY_AWARE_TEMPLATES` từng phải tắt toàn bộ vì một công thức chung "không biết template
nào thực sự dùng field nào".

---

## 5. TRƯỚC KHI SỬA CODE

| Bạn định sửa | Đọc trước |
| :--- | :--- |
| Ngân sách cỡ chữ, `compute_*_budget` | ROADMAP §4.1, §4.2 |
| Thêm/sửa template | ROADMAP §3, §6.2 |
| Prompt LLM, schema | ROADMAP §2 |
| Mask, geometry | ROADMAP §3.2, §5 |
| Hiệu ứng, thư viện JS/CSS | ROADMAP §4.4 (có danh sách thư viện **đã bị loại** kèm lý do) |
| Test | ROADMAP §4.5 — test phải kiểm thẩm mỹ, không chỉ kiểm không-crash |

---

## 6. LỊCH SỬ CẦN BIẾT

Repo này tách khỏi repo `Tendoo AI` ngày 25/09/2026. Repo cũ chứa 4 thế hệ code và một khối
lớn nghiên cứu về hướng **render chữ native qua DiT bằng glyph bitmap VAE** — hướng đó **đã bị
thay thế hoàn toàn bằng text overlay** và **không áp dụng cho code hiện tại**. Nếu cần tham
chiếu lịch sử đó, xem repo cũ (đã archive); đừng mang các quy tắc của nó sang đây.
