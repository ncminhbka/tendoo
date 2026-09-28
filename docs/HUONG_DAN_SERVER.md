# HƯỚNG DẪN CHẠY DEMO TRÊN MÁY CHỦ GPU (2× A30, JupyterLab)

Mục tiêu: kéo mã mới về máy chủ, kiểm tra, rồi mở giao diện demo sinh poster **end to end** — LLM viết nội dung →
FLUX.2-klein **bản distill** vẽ nền (8 bước) → trình duyệt sắp chữ → poster PNG.

Mọi lệnh gõ trong **Terminal của JupyterLab** (File → New → Terminal). Làm lần lượt từ bước 1.

---

## Tóm tắt 7 bước

```bash
cd ~/work && (git -C tendoo-v3 pull || git clone https://github.com/ncminhbka/tendoo.git tendoo-v3)   # 1. lấy mã
cd ~/work/tendoo-v3 && pip install -r requirements.txt                                           # 2. thư viện
python scripts/check_server.py                                                                   # 3. tự kiểm tra
cp -n src/tendoo_v3/.env.example .env && nano .env                                               # 4. chọn LLM
nohup python src/tendoo_v3/demo_server.py --model distill --port 8088 > demo.log 2>&1 &           # 5. chạy demo
# 6. mở trình duyệt:  https://<địa-chỉ-jupyter>/user/<tên-bạn>/proxy/8088/
python scripts/run_e2e_server.py                                                                 # 7. test tự động 20 mẫu
```

Chi tiết từng bước và cách xử lý sự cố ở dưới.

---

## Bước 1 — Lấy mã mới nhất

**Lần đầu** (chưa có thư mục `~/work/tendoo-v3`):

```bash
cd ~/work
git clone https://github.com/ncminhbka/tendoo.git tendoo-v3
```

(Repo riêng tư: HTTPS sẽ hỏi tên + Personal Access Token của GitHub; máy chủ đã có SSH key thì dùng `git@github.com:ncminhbka/tendoo.git`. Máy chủ không ra được GitHub →
trên máy cá nhân chạy `git archive -o tendoo-v3.zip HEAD`, tải file ZIP lên JupyterLab rồi `unzip tendoo-v3.zip -d ~/work/tendoo-v3`.)

**Những lần sau:**

```bash
cd ~/work/tendoo-v3 && git pull
```

Kiểm tra đúng bản: `git log --oneline -1` phải trùng commit mới nhất trên GitHub.

## Bước 2 — Cài thư viện

```bash
cd ~/work/tendoo-v3
pip install -r requirements.txt
```

- Máy chủ đã có torch bản CUDA thì pip giữ nguyên, không cài đè.
- **Chromium cho phần sắp chữ** — cần một trình duyệt headless:
  - máy chủ có mạng: `python -m playwright install chromium`
  - máy chủ **không** có mạng: nếu đã có sẵn chromium/google-chrome thì hệ thống tự tìm; hoặc chỉ định
    `export PLAYWRIGHT_CHROME_PATH=/đường/dẫn/chromium`; hoặc chép thư mục `~/.cache/ms-playwright` từ một máy Linux
    khác đã cài.
  - báo thiếu thư viện hệ thống (libnss3, libatk…): `python -m playwright install-deps` (cần quyền root) — hoặc nhờ
    quản trị cài.
- Font đã nằm sẵn trong repo (`fonts/`), không cần cài gì.

## Bước 3 — Tự kiểm tra (bắt buộc, 30 giây)

```bash
python scripts/check_server.py
```

Script **không** tải model lên GPU; chỉ kiểm tra từng mục và in `[ĐẠT]` / `[LỖI]` kèm dòng `SỬA:`. Cần đạt:

| Mục | Phải thấy |
| :--- | :--- |
| GPU | `cuda:0 NVIDIA A30 24GB`, `cuda:1 NVIDIA A30 24GB` |
| DiT distill | `flux-2-klein-4b.safetensors` (~7–8GB) |
| VAE, Text encoder, Tokenizer | đường dẫn trong `~/persistent-data/FLUX.2-klein-...` |
| transformers | ≥ 4.51 |
| Chromium | "chạy được" |
| LLM | xem Bước 4 |

**File distill nằm ở đâu?** Hệ thống tự tìm `flux-2-klein-4b.safetensors` trong:
`~/persistent-data/FLUX.2-klein-4B/`, rồi `~/persistent-data/FLUX.2-klein-base-4B/` (cùng các bản `/home/jovyan/...`,
`/persistent-data/...`). VAE / text_encoder / tokenizer lấy từ thư mục nào có. File distill để chỗ khác thì chạy:

```bash
python scripts/check_server.py --model_path /đường/dẫn/flux-2-klein-4b.safetensors
```

và thêm đúng `--model_path` đó vào lệnh chạy demo ở Bước 5.

Cuối cùng phải in: **`SẴN SÀNG chạy GPU thật.`**

## Bước 4 — Chọn LLM viết nội dung (file `.env`)

```bash
cp -n src/tendoo_v3/.env.example .env
nano .env        # Ctrl+O lưu, Ctrl+X thoát
```

Chọn **một** trong hai (sửa dòng `TENDOO_V3_LLM_BACKEND`):

| Lựa chọn | Đặt trong `.env` | Khi nào dùng |
| :--- | :--- | :--- |
| **A. LLM nội bộ qua mạng** (khuyên dùng nếu có) | `TENDOO_V3_LLM_BACKEND=api` + `TENDOO_V3_LLM_BASE_URL=...` + `TENDOO_V3_LLM_API_KEY=...` + `TENDOO_V3_LLM_MODEL=...` | Cụm LLM nội bộ (vd `http://10.221.155.3:8004/v1`) còn chạy — nội dung tốt hơn, không tốn VRAM máy chủ |
| **B. Qwen3-4B ngay trên máy chủ** | `TENDOO_V3_LLM_BACKEND=local` | Không có LLM nội bộ — chạy offline trên `cuda:1`, nội dung kém hơn model lớn |

- Endpoint không cần key: thêm `TENDOO_V3_LLM_ALLOW_NO_KEY=1`.
- `.env` chứa khoá — **không commit** (đã nằm trong `.gitignore`).
- File mẫu đặt cổng 7860; lệnh ở Bước 5 truyền `--port 8088` nên cổng theo lệnh.
- Chạy lại `python scripts/check_server.py` để thấy mục LLM `[ĐẠT]`.

## Bước 5 — Chạy máy chủ demo

```bash
cd ~/work/tendoo-v3
nohup python src/tendoo_v3/demo_server.py --model distill --port 8088 > demo.log 2>&1 &
tail -f demo.log          # xem log; Ctrl+C để thoát xem (máy chủ vẫn chạy)
```

Chờ 1–3 phút nạp model. Log **đúng** sẽ có:

```
✅ Found local Distilled DiT weights (FLUX.2-klein-4B): .../flux-2-klein-4b.safetensors
✅ All Diffusion models loaded and warm in VRAM in XX.Xs!
[GPU 0] Occupied: ...   [GPU 1] Occupied: ...
Uvicorn running on http://0.0.0.0:8088
```

Kiểm tra nhanh ở terminal khác:

```bash
curl -s localhost:8088/api/health
```

Phải thấy `"mock_mode":false`, `"models_loaded":true`, `"model_name":"flux.2-klein-4b"`.
Nếu `"mock_mode":true` → máy chủ vẫn chạy nhưng **ảnh nền là ảnh giả** — xem mục Sự cố.

Dừng máy chủ: `pkill -f demo_server.py`.

## Bước 6 — Mở giao diện

- **Qua JupyterLab** (cách thường dùng): mở tab trình duyệt mới, sửa địa chỉ JupyterLab đang dùng thành
  `https://<địa-chỉ-jupyter>/user/<tên-bạn>/proxy/8088/` — **nhớ dấu `/` ở cuối**.
  (Địa chỉ JupyterLab dạng `.../user/<tên>/lab` → thay `lab` bằng `proxy/8088/`. Cần extension jupyter-server-proxy,
  thường có sẵn.)
- **Trực tiếp** (nếu mạng nội bộ cho phép): `http://<IP-máy-chủ>:8088/`.

Trên giao diện: chọn loại poster, điền thông tin, chọn khung hình → nút **Tạo Poster Thương Mại Ngay**. Kết quả có 3 tab: *Poster Hoàn Thiện*, *Background Sạch (DiT)*, *Mặt Nạ An Toàn (Mask)*. Mỗi poster mất khoảng vài giây
(LLM) + vài giây (FLUX distill 8 bước) + ~1–2 giây (sắp chữ). Ảnh lưu tại
`~/work/tendoo-v3/output_tendoo_v3/demo_runs/run_<thời-gian>/variant_<n>/` gồm `poster.png`, `background.png`,
`mask.png`, `plan.json`.

**Sửa chữ không vẽ lại nền:** dưới poster có ô **"Sửa chữ trên nền này"** hiện các dòng chữ poster đang có. Sửa
rồi bấm **Cập nhật chữ** (khoảng 1–2 giây, không dùng GPU vẽ nền), hoặc **Thêm 3 kiểu chữ** để xem 3 cách trình bày khác
trên cùng nền — bấm vào ảnh nhỏ để xem và tải. Xoá hẳn hoặc thêm một dòng làm đổi vùng chữ thì giao diện báo cần tạo lại
poster. Chữ mới quá dài bị cắt thì giao diện báo đỏ — rút ngắn lại.

Xem LLM đã trả gì cho lần sinh gần nhất: `https://.../proxy/8088/api/v3/llm-debug/latest`.

## Bước 7 — Test tự động 20 mẫu (thay cho bấm tay)

Khi máy chủ demo đang chạy (Bước 5), mở **terminal khác**:

```bash
cd ~/work/tendoo-v3
python scripts/run_e2e_server.py                 # 20 mẫu trong tests/e2e_server_cases.json
```

Script gửi lần lượt 20 mẫu tới `/api/generate` **đúng như giao diện gửi** (14 mẫu tái hiện 6 loại poster trên giao
diện × 4 khung hình, 6 mẫu nhu cầu khác qua trường `prompt`: thông báo, thiệp chúc, thư mời, menu, combo, Tết thuần
chữ), rồi với mỗi mẫu: tải poster + ảnh nền, đọc LLM có chạy thật không, chấm 6 điều kiện thẩm mỹ trên nền thật, và
thử **sửa chữ trên nền cũ** (như nút "Cập nhật chữ").
In mỗi mẫu một dòng, cuối cùng tổng kết:

```
20/20 mẫu ra poster | LLM thật 20/20 | trung bình 6.5s/mẫu | đạt cả 6 điều kiện 15/20 | sửa chữ trên nền cũ 20/20
Xem: output_probe/e2e_0928_0930/index.html
```

- Mở `index.html` trong JupyterLab (chuột phải → Open in New Browser Tab) để xem cả 20 poster cạnh nhau.
- Dòng `!` = mẫu đó LLM không chạy thật (dùng plan dự phòng) — xem lý do in kèm, và Bước 4.
- Dòng `✗` = mẫu lỗi hẳn — xem `demo.log`.
- Chỉ chạy vài mẫu: `--only e02,e18`. Máy chủ ở cổng khác: `--url http://127.0.0.1:8090`. Thêm/sửa mẫu: sửa
  `tests/e2e_server_cases.json` (mỗi mẫu = JSON giao diện gửi).
- Gửi kết quả về: `zip -r e2e.zip output_probe/e2e_*` rồi tải về.

---

## Sự cố thường gặp

| Hiện tượng | Nguyên nhân / cách xử lý |
| :--- | :--- |
| `"mock_mode":true`, log có `Failed to load GPU models` | Đọc dòng lỗi ngay trên đó trong `demo.log`. Thiếu file → Bước 3 (`--model_path`). Lỗi ở bước **TextEncoder** nhắc FP8 / compute capability → A30 (compute 8.0) không hỗ trợ FP8 phần cứng: cập nhật `pip install -U transformers accelerate` rồi thử lại; vẫn lỗi → báo lại kèm log (cần bản text encoder bf16). |
| Log: `Không tìm thấy flux-2-klein-4b.safetensors lẫn ...` | Không thấy trọng số → chạy với `--model_path`. |
| Log: `Không import được flux2` | transformers quá cũ → `pip install 'transformers>=4.51'`. |
| `CUDA out of memory` | Đang dùng LLM cục bộ (B) cùng card với VAE + text encoder → chuyển sang A nếu được; hoặc kiểm tra `nvidia-smi` xem tiến trình khác chiếm GPU (`pkill -f demo_server.py` rồi chạy lại). |
| Poster ra nhưng nội dung "cứng", giống mẫu | LLM không gọi được → hệ thống dùng plan dự phòng theo quy tắc. Xem `api/v3/llm-debug/latest` (trường `status`, `error`) và Bước 4. |
| Lỗi mở Chromium / poster trắng | Bước 2 mục Chromium. |
| `Address already in use` | Cổng 8088 bận → `pkill -f demo_server.py` hoặc đổi `--port 8090` (và URL proxy tương ứng). |
| Trang proxy trắng / 404 | Thiếu dấu `/` cuối URL; hoặc JupyterLab không có server-proxy → dùng cách truy cập trực tiếp. |

---

## Tuỳ chọn — đo tốc độ và gửi kết quả về

```bash
python scripts/bench_maskless.py --seeds 5 --model distill     # tốc độ có/không mask -> output_probe/bench_maskless.json
cd ~/work/tendoo-v3 && zip -r ket_qua_$(date +%m%d).zip output_tendoo_v3/demo_runs output_probe/bench_maskless* demo.log
```

Tải file ZIP về máy cá nhân (chuột phải → Download trong JupyterLab) để phân tích.

## Ghi chú kỹ thuật (cho người bảo trì)

- Model mặc định: **distill** (`flux-2-klein-4b.safetensors`, 8 bước, không CFG). `--model base` vẫn chạy được — từ
  28/09 bản base có CFG đúng chuẩn BFL (`velocity_blending`, `cfg_scale=guidance`), nhưng chậm (50 bước × 3 nhánh).
- GPU: DiT trên `cuda:0`; VAE + text encoder (+ Qwen cục bộ nếu chọn B) trên `cuda:1` — đổi bằng `--device_dit`,
  `--device_aux`.
- `--mock`: chạy giao diện không cần GPU (ảnh nền giả) — dùng để thử giao diện.
- Mọi yêu cầu sinh ảnh chạy tuần tự (1 người dùng tại một thời điểm là tối ưu).
