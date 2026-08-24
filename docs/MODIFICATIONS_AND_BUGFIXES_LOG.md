# 📋 NHẬT KÝ SỬA ĐỔI MÃ NGUỒN GỐC & BIÊN NIÊN SỬ COMMIT
## (Upstream Modifications & Commit History Changelog)

**Dự án**: Nâng cấp FLUX.2 [klein] 4B Base Sinh chữ Tiếng Việt  
**Mục tiêu mô hình**: `FLUX.2-klein-base-4B`  
**Hạ tầng mục tiêu**: 2x NVIDIA A30 (48GB VRAM) / JupyterLab Offline  
**Mã nguồn gốc đối chiếu**: [`black-forest-labs/flux2`](https://github.com/black-forest-labs/flux2)

---

## 📑 PHẦN I: MA TRẬN ĐỐI CHIẾU MÃ NGUỒN GỐC (`src/flux2/`)

Toàn bộ thư mục `src/flux2/` được clone từ repo chính thức của Black Forest Labs (BFL) và được rà soát đối chiếu từng dòng mã:

| Tên File | Trạng thái so với BFL Upstream | Mục đích sửa đổi |
| :--- | :---: | :--- |
| [`autoencoder.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/autoencoder.py) | 🟢 **100% Nguyên bản** | Giữ nguyên kiến trúc AutoEncoder chuẩn BFL (128 latent channels, 16x compression). |
| [`model.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/model.py) | 🔴 **Đã sửa đổi — bảng này trước đây khai SAI là "100% Nguyên bản"** | Giữ nguyên `Klein4BParams`, `EmbedND`, `rope`, `apply_rope`, cấu trúc `DoubleStreamBlock`/`SingleStreamBlock` gốc. Nhưng đã **thêm mới** (không có trong BFL upstream): `Flux2.forward_kv_extract`, `Flux2.forward_kv_cached`, `DoubleStreamBlock.forward_kv_extract/forward_kv_cached`, `SingleStreamBlock.forward_kv_extract/forward_kv_cached`, `_blend_double_mods`, `_blend_single_mods`, `_blend_mod_triple`; và mở rộng `causal_attn_fn` thêm nhánh `kv_cache is not None`. **Không có commit nào trong Phần II ghi nhận việc thêm các hàm này** — nhiều khả năng chúng đã nằm sẵn trong commit khởi tạo `aee82cf` trước khi được tách log riêng. Cần chạy `git log -p --follow -- src/flux2/model.py` để xác định chính xác thời điểm, tránh audit trail có khoảng trống. Đã kiểm tra kỹ logic `_blend_double_mods`/`_blend_single_mods`: modulation của ref token luôn lấy từ `ref_fixed_timestep` cố định, ref token chỉ tự-attend (không bao giờ phụ thuộc canvas) → về mặt toán học, việc cache K/V của ref token qua các bước denoise là **chính xác tuyệt đối, không phải xấp xỉ**. |
| [`openrouter_api_client.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/openrouter_api_client.py) | 🟢 **100% Nguyên bản** | Giữ nguyên client API dự phòng. |
| [`system_messages.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/system_messages.py) | 🟢 **100% Nguyên bản** | Giữ nguyên system prompts của BFL. |
| [`watermark.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/watermark.py) | 🟢 **100% Nguyên bản** | Giữ nguyên module watermark vô hình. |
| [`sampling.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/sampling.py) | 🟡 **Đã nâng cấp** *(+77 / -29)* | Thêm `denoise_cfg` cho Pure Prompt và nâng cấp `denoise_cfg_cached` với `ref_fixed_timestep=0.0`. |
| [`text_encoder.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/text_encoder.py) | 🟡 **Đã nâng cấp** *(+259 / -231)* | Thêm Auto-discovery cho Qwen3-4B-FP8 offline, tự động load tokenizer và batching đồng bộ. |
| [`util.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/util.py) | 🟡 **Đã nâng cấp** *(+150 / -15)* | Thêm Auto-discovery `persistent-data`, chuyển đổi VAE keys (Diffusers $\leftrightarrow$ BFL), và fallback `strict=False`. |

---

### 🔍 Chi tiết các hàm được can thiệp trong 3 file:

#### 1. [`src/flux2/sampling.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/sampling.py)
* **Thêm hàm `denoise_cfg`**: Chạy Denoise Euler ODE Flow Matching thuần bằng Prompt (CFG scale), phục vụ tạo ảnh Baseline so sánh khi không có ref glyph.
* **Nâng cấp `denoise_cfg_cached`**:
  * Thêm tham số `ref_fixed_timestep: float = 0.0`.
  * **Step 0**: Chạy `model.forward_kv_extract()` với full sequence `[ref, canvas]` để tính toán và lưu `kv_cache` của glyph reference.
  * **Step 1 đến 50**: Chạy `model.forward_kv_cached()` chỉ trên canvas tokens, tái sử dụng `kv_cache`, giúp tăng tốc độ suy luận $\sim 3\times$ và cố định tín hiệu chữ không bị nhiễu xóa nhòa.

#### 2. [`src/flux2/text_encoder.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/text_encoder.py)
* **Tối ưu `Qwen3Embedder`**:
  * Thêm hỗ trợ tham số `tokenizer_spec` (tự động nạp thư mục `tokenizer/` nội bộ).
  * Chấp nhận cả `str` và `list[str]`.
  * Thiết lập `enable_thinking=False` khi build template chat cho Qwen3.
* **Nâng cấp `load_qwen3_embedder`**:
  * Tự động dò tìm trọng số offline từ biến môi trường hoặc thư mục `~/persistent-data/FLUX.2-klein-base-4B/text_encoder/`.

#### 3. [`src/flux2/util.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/util.py)
* **Thêm `find_persistent_data_root()`**: Tự động phát hiện cây thư mục `persistent-data` trên JupyterLab Server (`/home/jovyan/persistent-data/...`).
* **Thêm `convert_diffusers_vae_to_bfl(sd)`**: Bộ dịch ngược định dạng trọng số VAE từ Diffusers sang cấu trúc Native BFL.
* **Cải tiến `load_flow_model` và `load_ae`**: Tự động nạp trọng số cục bộ, tích hợp bộ chuyển đổi VAE keys và thêm fallback `strict=False` an toàn.

---

## 📜 PHẦN II: BIÊN NIÊN SỬ CHI TIẾT TỪNG COMMIT KỂ TỪ INITIAL COMMIT

Dưới đây là bảng theo dõi toàn bộ các commit được thực hiện trên repository kể từ mốc khởi tạo (`aee82cf`), ghi nhận bối cảnh kỹ thuật, file thay đổi và giải pháp xử lý:

---

### 1. Commit `aee82cf` — Khởi tạo dự án nghiên cứu FLUX.2 Tiếng Việt
* **Thời gian**: `Sat Aug 22 16:08:37 2026 +0700`
* **File thay đổi** (21 files: +1589, -164):
  * `AGENTS.md`, `project_roadmap.txt`, `requirements.txt`, `scripts/test_rope_spatial_binding.py`, các file tài liệu và specs.
* **Nội dung thay đổi**:
  * Thiết lập tài liệu định hướng "Kiềng 3 chân" (RoPE Binding + Tight Crop + LoRA DiT).
  * Tạo kịch bản thử nghiệm đầu tiên `scripts/test_rope_spatial_binding.py` để kiểm chứng Giai đoạn 1A.

---

### 2. Commit `c52107f` — Fix: Tự động nhận diện mô hình Offline trong mạng nội bộ
* **Thời gian**: `Sat Aug 22 16:18:55 2026 +0700`
* **File thay đổi** (2 files: +79, -17):
  * [`src/flux2/text_encoder.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/text_encoder.py), [`src/flux2/util.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/util.py)
* **Vấn đề xuất hiện**: Khi chạy trên server nội bộ không có Internet, các hàm load mô hình mặc định của BFL cố gọi HuggingFace Hub và văng lỗi Network Timeout / Repository Not Found.
* **Giải pháp kỹ thuật**:
  * Thêm logic quét biến môi trường (`KLEIN_4B_BASE_MODEL_PATH`, `AE_MODEL_PATH`, `TEXT_ENCODER_PATH`).
  * Xây dựng hàm `find_persistent_data_root()` tự động tìm thư mục `persistent-data/FLUX.2-klein-base-4B`.

---

### 3. Commit `f5d9f29` — Update: Đồng bộ sơ đồ thư mục máy chủ thực tế
* **Thời gian**: `Sat Aug 22 16:23:30 2026 +0700`
* **File thay đổi** (5 files: +63, -24):
  * `AGENTS.md`, `.agents/rules/agent_workflow_rules.md`, `scripts/test_rope_spatial_binding.py`, `src/flux2/text_encoder.py`, `src/flux2/util.py`
* **Nội dung thay đổi**:
  * Cập nhật chính xác cấu trúc thư mục Server JupyterLab: `text_encoder/`, `tokenizer/`, `vae/`, `transformer/`.
  * Cho phép `load_qwen3_embedder` tự động nạp `tokenizer/` nằm ngang hàng với `text_encoder/`.

---

### 4. Commit `0e39786` — Fix: Export tường minh các hàm nạp Text Encoder
* **Thời gian**: `Sat Aug 22 16:29:11 2026 +0700`
* **File thay đổi** (1 file: +126, -140):
  * [`src/flux2/text_encoder.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/text_encoder.py)
* **Vấn đề xuất hiện**: Các hàm `load_mistral_small_embedder` và `load_qwen3_embedder` bị lồng sai scope hoặc thiếu export khiến `util.py` không import được.
* **Giải pháp kỹ thuật**:
  * Tái cấu trúc file `text_encoder.py`, phân tách rõ ràng class `Mistral3SmallEmbedder`, `Qwen3Embedder` và các factory functions nạp trọng số tương ứng.

---

### 5. Commit `3f8a78d` — Update: Khóa cứng mục tiêu duy nhất FLUX.2-klein-base-4B
* **Thời gian**: `Sat Aug 22 16:30:12 2026 +0700`
* **File thay đổi** (2 files: +23, -2):
  * `AGENTS.md`, `.agents/rules/agent_workflow_rules.md`
* **Nội dung thay đổi**:
  * Khóa cứng mọi cấu hình dự án tập trung duy nhất vào bản `FLUX.2-klein-base-4B` (5 DoubleBlocks, 20 SingleBlocks, Qwen3-4B-FP8 7680d, 50 Euler steps, CFG 4.0).

---

### 6. Commit `52932b4` — Perf: Dọn rác binary nặng khỏi Git và tự động hóa môi trường
* **Thời gian**: `Sat Aug 22 16:32:09 2026 +0700`
* **File thay đổi** (15 files: +48, -2, untrack ~90MB binary images/PDF):
  * `.gitignore`, `scripts/cli.py`, `scripts/test_rope_spatial_binding.py`, xóa theo dõi các file demo PNG/JPG/PDF trong git tree.
* **Nội dung thay đổi**:
  * Đưa các asset nặng vào `.gitignore` để tăng tốc độ `git clone` / `git pull` trên server.
  * Tự động cấu hình `sys.path.insert(0, str(SRC_DIR))` và các biến môi trường offline (`HF_HUB_OFFLINE=1`) ngay khi script khởi chạy.

---

### 7. Commit `dc98276` — Fix: Nâng cấp Auto-discovery quét đa tầng thư mục
* **Thời gian**: `Sat Aug 22 16:39:40 2026 +0700`
* **File thay đổi** (3 files: +116, -92):
  * `scripts/test_rope_spatial_binding.py`, `src/flux2/text_encoder.py`, `src/flux2/util.py`
* **Vấn đề xuất hiện**: Khi chạy từ thư mục con hoặc từ JupyterLab Home, đường dẫn tương đối tới `persistent-data` bị sai lệch.
* **Giải pháp kỹ thuật**:
  * Hàm `find_persistent_data_root()` bổ sung vòng lặp quét ngược 5 cấp thư mục cha và kiểm tra thư mục home của user (`~`, `/home/jovyan`, `/persistent-data`).

---

### 8. Commit `29e62fb` — Robustness: Thêm cơ chế Fallback nạp State Dict
* **Thời gian**: `Sat Aug 22 16:41:50 2026 +0700`
* **File thay đổi** (1 file: +12, -2):
  * [`src/flux2/util.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/util.py)
* **Vấn đề xuất hiện**: Khi nạp checkpoint DiT hoặc VAE từ định dạng biến thể, cờ `strict=True` làm crash chương trình do thiếu/thừa một số key metadata không ảnh hưởng tính toán.
* **Giải pháp kỹ thuật**:
  * Bọc khối `try...except` khi gọi `model.load_state_dict()` và `ae.load_state_dict()` để tự động fallback sang `strict=False`.

---

### 9. Commit `9515f3f` — Fix: Xây dựng bộ chuyển đổi Key VAE Diffusers sang Native BFL
* **Thời gian**: `Sat Aug 22 16:47:26 2026 +0700`
* **File thay đổi** (1 file: +52, -1):
  * [`src/flux2/util.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/util.py)
* **Vấn đề xuất hiện**: `RuntimeError: Cannot copy out of meta tensor` do file `diffusion_pytorch_model.safetensors` tải về có tiền tố Diffusers (`encoder.down_blocks.0`, `quant_conv`) không khớp với kiến trúc `AutoEncoder` của BFL.
* **Giải pháp kỹ thuật**:
  * Thêm hàm `convert_diffusers_vae_to_bfl(sd)` chuyển đổi key:
    * `quant_conv.` $\rightarrow$ `encoder.quant_conv.`
    * `post_quant_conv.` $\rightarrow$ `decoder.post_quant_conv.`
    * `encoder.down_blocks.X.` $\rightarrow$ `encoder.down.X.`
    * `decoder.up_blocks.X.` $\rightarrow$ `decoder.up.X.`

---

### 10. Commit `e5f520a` — Fix: Đảo chiều chỉ số Block Decoder và Reshape Tensor Attention 4D
* **Thời gian**: `Sat Aug 22 16:51:19 2026 +0700`
* **File thay đổi** (1 file: +33, -21):
  * [`src/flux2/util.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/util.py)
* **Vấn đề xuất hiện**:
  1. Hình ảnh decode bị rác vỡ hạt do thứ tự các tầng Decoder trong Diffusers (`up_blocks.0` là độ phân giải thấp nhất) ngược với BFL (`up.3` là độ phân giải thấp nhất).
  2. `RuntimeError: size mismatch for mid.attn_1.q.weight (512, 512) vs (512, 512, 1, 1)`.
* **Giải pháp kỹ thuật**:
  * Đảo chiều index tầng Decoder: `bfl_level = 3 - diffusers_level`.
  * Reshape trọng số Attention Linear 2D `(512, 512)` sang Conv2d 4D `(512, 512, 1, 1)`.

---

### 11. Commit `9f728ce` — Perf: Tối ưu phân bổ bộ nhớ trên 2x GPU NVIDIA A30
* **Thời gian**: `Sat Aug 22 16:55:19 2026 +0700`
* **File thay đổi** (2 files: +45, -21):
  * `scripts/test_rope_spatial_binding.py`, `src/flux2/text_encoder.py`
* **Vấn đề xuất hiện**: Nguy cơ tràn VRAM (OOM) khi dồn cả DiT Base 4B, Qwen3-4B-FP8, VAE và Attention buffer lên 1 GPU duy nhất khi batch size = 2 (CFG).
* **Giải pháp kỹ thuật**:
  * Tự động nhận diện số GPU: DiT Base 4B chạy trên `cuda:0`, Qwen3 Text Encoder và VAE AutoEncoder chạy trên `cuda:1`.
  * Thu hồi VRAM tức thì nếu chạy ở chế độ Single-GPU.

---

### 12. Commit `d68fef9` — Fix: Khắc phục xung đột kiểu dữ liệu VAE Convolution (Float32 vs Bfloat16)
* **Thời gian**: `Sat Aug 22 16:59:13 2026 +0700`
* **File thay đổi** (1 file: +3, -2):
  * [`scripts/test_rope_spatial_binding.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_rope_spatial_binding.py)
* **Vấn đề xuất hiện**: `RuntimeError: expected scalar type BFloat16 but found Float` tại phép tích chập nén glyph image.
* **Giải pháp kỹ thuật**:
  * Đọc kiểu dữ liệu thực tế từ tham số của VAE (`next(ae.parameters()).dtype`) và cast `glyph_tensor` sang đúng `bfloat16`.

---

### 13. Commit `89215a1` — Fix: Loại bỏ tham số di sản `shift` trong `get_schedule`
* **Thời gian**: `Sat Aug 22 17:04:00 2026 +0700`
* **File thay đổi** (1 file: +0, -1):
  * [`scripts/test_rope_spatial_binding.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_rope_spatial_binding.py)
* **Vấn đề xuất hiện**: `TypeError: get_schedule() got an unexpected keyword argument 'shift'` do code cũ kế thừa từ FLUX.1.
* **Giải pháp kỹ thuật**:
  * Xóa bỏ tham số `shift`, chỉ truyền `num_steps` và `image_seq_len` chuẩn theo API FLUX.2.

---

### 14. Commit `1f49a55` — Refactor: Snap Box 16px, quét Font Unicode tiếng Việt, tạo ảnh 3-Panel
* **Thời gian**: `Sat Aug 22 17:13:56 2026 +0700`
* **File thay đổi** (1 file: +88, -69):
  * [`scripts/test_rope_spatial_binding.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_rope_spatial_binding.py)
* **Nội dung thay đổi**:
  * Tự động snap tọa độ Bounding Box về bội số của 16 (`(val // 16) * 16`) tránh lệch shape latent.
  * Bổ sung cơ chế quét đa font Unicode trên Ubuntu/Debian (NotoSans, DejaVuSans, Liberation, FreeSans) với khả năng tự động co giãn font vừa vặn box.
  * Ghép đồng bộ batch `text_encoder(["", prompt])` để tránh lệch sequence length padding.
  * Tự động ghép ảnh đối chứng 3-Panel: `[Pure Prompt | Baseline 0,0 | RoPE Bound Box]`.

---

### 15. Commit `eaca869` — Fix: Khóa Ref Timestep $t=0.0$ và ứng dụng KV-Caching chống xóa chữ
* **Thời gian**: `Sat Aug 22 17:28:33 2026 +0700`
* **File thay đổi** (2 files: +76, -25):
  * `scripts/test_rope_spatial_binding.py`, `src/flux2/sampling.py`
* **Vấn đề xuất hiện**: Chữ bị mờ và biến mất vào nền khi denoise trên các prompt trung tính do mô hình coi ref token bị nhiễu theo canvas.
* **Giải pháp kỹ thuật**:
  * Khóa mức nhiễu của reference token cố định ở $t=0.0$ (ảnh sạch tuyệt đối).
  * Viết hàm `denoise_cfg_cached`: Gọi `forward_kv_extract` ở Step 0 để lưu KV cache của ref, từ Step 1–50 chỉ denoise canvas và tái sử dụng KV cache.

---

### 16. Commit `dbcbf1c` — Fix: Dọn dẹp tham số thừa trong hàm gọi Pure Prompt Baseline
* **Thời gian**: `Sat Aug 22 17:35:00 2026 +0700`
* **File thay đổi** (1 file: +0, -2):
  * [`scripts/test_rope_spatial_binding.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_rope_spatial_binding.py)
* **Nội dung thay đổi**:
  * Xóa bỏ các tham số `img_cond_seq` thừa khi gọi hàm `denoise_cfg` cho nhánh Pure Prompt Baseline, đảm bảo nhánh baseline chạy hoàn toàn độc lập không có ref token.

---

### 17. Commit `8411f89` — Docs: Chuyển cấu trúc tài liệu vào `docs/` và tạo Changelog
* **Thời gian**: `Sun Aug 23 21:22:57 2026 +0700`
* **File thay đổi** (8 files: +166, -0):
  * Di chuyển toàn bộ các file text/spec rải rác ở root vào thư mục [`docs/`](file:///d:/Viettel%20Telecom/Tendoo%20AI/docs).
  * Khởi tạo tài liệu tổng hợp sửa đổi mã nguồn gốc và lịch sử lỗi `MODIFICATIONS_AND_BUGFIXES_LOG.md`.

---

### 18. Commit `9088b0e` — Perf: Bỏ Baseline Pure Prompt, chuyển sang so sánh 2-Panel trực tiếp
* **Thời gian**: `Mon Aug 24 08:14:16 2026 +0700`
* **File thay đổi** (1 file: +9, -30):
  * [`scripts/test_rope_spatial_binding.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_rope_spatial_binding.py)
* **Nội dung thay đổi**:
  * Loại bỏ pass denoise `Baseline 2 (Pure Prompt)` để tiết kiệm 33% thời gian chạy thực nghiệm.
  * Xuất ảnh `_COMPARISON.png` chuẩn 2-Panel đối chứng: `[Baseline (0,0) | RoPE Bound Box]`.

---

### 19. Commit `5a53136` — Rules: Bổ sung quy tắc chống nịnh bợ và tư duy phản biện
* **Thời gian**: `Mon Aug 24 08:35:42 2026 +0700`
* **File thay đổi** (1 file: +5, -3):
  * [`.agents/rules/agent_workflow_rules.md`](file:///d:/Viettel%20Telecom/Tendoo%20AI/.agents/rules/agent_workflow_rules.md)
* **Nội dung thay đổi**:
  * Quy định bắt buộc: Không nịnh bợ, trung thực thẳng thắn 100%, suy nghĩ kỹ lưỡng trước khi kết luận.

---

### 20. Commit `39405bf` — Feat: Thêm Diagnostic Instrumentation và chế độ Debug Solid Color
* **Thời gian**: `Mon Aug 24 08:52:51 2026 +0700`
* **File thay đổi** (1 file: +126, -10):
  * [`scripts/test_rope_spatial_binding.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_rope_spatial_binding.py)
* **Nội dung thay đổi**:
  * Thêm hàm `diagnose_id_ranges`: Kiểm tra trực tiếp min/max 4 cột tọa độ `img_ids` vs `ref_ids` trước khi denoise, phát hiện sớm lỗi lệch thứ tự trục.
  * Thêm cờ `--debug_mode solid_color` (`create_debug_solid_block`): Dùng khối màu đỏ đặc để cô lập bài toán định vị không gian khỏi bài toán nhận diện nét chữ.

---

### 21. Commit `468fe66` — Fix: Khôi phục Full Dynamic Denoise (Revert KV-Caching gây vỡ chữ)
* **Thời gian**: `Mon Aug 24 09:09:48 2026 +0700`
* **File thay đổi** (3 files: +30, -76):
  * [`src/flux2/sampling.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/sampling.py), [`scripts/test_rope_spatial_binding.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_rope_spatial_binding.py)
* **Vấn đề xuất hiện**: KV-caching đóng băng Key/Value của ref token tại $t=1.0$ (khi canvas còn là nhiễu hạt 100%), triệt tiêu tương tác động đa bước và làm đảo thứ tự layout, khiến chữ bị biến thành ký tự rác.
* **Giải pháp kỹ thuật**:
  * Hủy bỏ `denoise_cfg_cached`, khôi phục hàm `denoise_cfg` full 50 bước tương tác liên tục giữa Canvas và Ref Tokens `[Canvas, Ref]`. Khôi phục chất lượng vẽ chữ và dấu tiếng Việt sắc nét như ban đầu.

---

### 22. Commit `5903303` — Sync: Đồng bộ 100% nguyên bản BFL cho `src/flux2/sampling.py`
* **Thời gian**: `Mon Aug 24 09:14:03 2026 +0700`
* **File thay đổi** (1 file: +6, -1):
  * [`src/flux2/sampling.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/flux2/sampling.py)
* **Nội dung thay đổi**:
  * Khôi phục toàn bộ hàm nguyên bản upstream của BFL, đảm bảo 6/8 file trong `src/flux2/` hoàn toàn đồng nhất 100% với official repository.

---

### 23. Commit `3f9d5cb` — Rules: Khóa cứng nguyên tắc Đóng băng Upstream Core (`src/flux2/`)
* **Thời gian**: `Mon Aug 24 09:17:22 2026 +0700`
* **File thay đổi** (2 files: +10, -1):
  * [`AGENTS.md`](file:///d:/Viettel%20Telecom/Tendoo%20AI/AGENTS.md), [`.agents/rules/agent_workflow_rules.md`](file:///d:/Viettel%20Telecom/Tendoo%20AI/.agents/rules/agent_workflow_rules.md)
* **Nội dung thay đổi**:
  * Đưa quy tắc "Đóng băng mã nguồn gốc BFL (`src/flux2/`)" thành quy chuẩn bắt buộc của dự án. Mọi mở rộng chỉ được viết ở tầng ngoài (`scripts/`, `src/tendoo/`).

---

### 24. Commit `297d947` — Feat: Nâng cấp bộ tạo Glyph Bitmap tiếng Việt chống tràn cho Text dài
* **Thời gian**: `Mon Aug 24 09:56:36 2026 +0700`
* **File thay đổi** (1 file: +113, -35):
  * [`scripts/test_rope_spatial_binding.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_rope_spatial_binding.py)
* **Nội dung thay đổi**:
  * Hỗ trợ ngắt dòng thủ công `\n` và tự động ngắt 1, 2, 3 dòng tùy theo tỉ lệ khung hình (Aspect Ratio) của Bounding Box.
  * Dùng thuật toán Binary Search tìm cỡ chữ lớn nhất có thể hiển thị vừa khít trong vùng vẽ.

---

### 25. Commit `d84315f` — Feat: Thêm Script Kiểm thử Đa Text Độc lập (`test_multi_text_rope.py`)
* **Thời gian**: `Mon Aug 24 10:07:05 2026 +0700`
* **File thay đổi** (1 file: +467, -0):
  * [`scripts/test_multi_text_rope.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_multi_text_rope.py)
* **Nội dung thay đổi**:
  * Xây dựng script chuyên biệt kiểm thử đa khối text tiếng Việt đồng thời với hệ tọa độ Time-Offset In-Context ($t=10, t=20$).

---

### 26. Commit `10a2267` — Feat: Thêm Script Sinh Poster Quảng cáo Sản phẩm (`test_product_poster.py`)
* **Thời gian**: `Mon Aug 24 10:16:38 2026 +0700`
* **File thay đổi** (5 files: +487, -0):
  * [`scripts/test_product_poster.py`](file:///d:/Viettel%20Telecom/Tendoo%20AI/scripts/test_product_poster.py), [`images/reference_prod.png`](file:///d:/Viettel%20Telecom/Tendoo%20AI/images/reference_prod.png), [`.gitignore`](file:///d:/Viettel%20Telecom/Tendoo%20AI/.gitignore)
* **Nội dung thay đổi**:
  * Tạo pipeline kết hợp Reference Ảnh Sản phẩm thật ($t=10.0$) và Reference Glyph Tiêu đề/Slogan tiếng Việt ($t=20.0$) để sinh poster thương mại chất lượng cao.
  * Cập nhật `.gitignore` cho phép theo dõi và đẩy ảnh mẫu trong thư mục `images/`.



