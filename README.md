# FLUX.2

**Frontier Visual Intelligence** — Công nghệ tạo và chỉnh sửa hình ảnh tiên tiến nhất (SOTA) từ [Black Forest Labs](https://bfl.ai).

---

<p align="center">
<a href="https://docs.bfl.ai">Tài liệu API</a> •
<a href="https://huggingface.co/black-forest-labs">Hugging Face</a> •
<a href="https://bfl.ai/blog">Blog</a>
</p>

Repository này chứa mã nguồn suy luận (inference) tối giản để thực hiện sinh ảnh và chỉnh sửa hình ảnh với các mô hình mở trọng số của dòng FLUX.2.

## Tin tức

- **[15.01.2026]** Hôm nay, chúng tôi phát hành dòng mô hình FLUX.2 [klein], đây là dòng mô hình nhanh nhất của chúng tôi từ trước đến nay. Khả năng sinh ảnh dưới 1 giây (sub-second) trên GPU phổ thông. Đọc thêm tại [bài viết blog của chúng tôi](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence).
- **[25.11.2025]** Chúng tôi phát hành FLUX.2 [dev], một mô hình 32 tỷ tham số (32B) phục vụ tạo ảnh từ văn bản và chỉnh sửa ảnh (hỗ trợ ảnh đơn tham chiếu và đa tham chiếu).

## Tổng quan các mô hình

| Tên | Step-distilled | Guidance-distilled | Text-to-Image | Chỉnh sửa ảnh (Đơn tham chiếu) | Chỉnh sửa ảnh (Đa tham chiếu) | Giấy phép |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | ✅ | ✅ | ✅ | ✅ | ✅ | [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| [FLUX.2 [klein] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | ✅ | ✅ | ✅ | ✅ | ✅ | [Giấy phép phi thương mại FLUX](model_licenses/LICENSE-FLUX-NON-COMMERICAL) |
| [FLUX.2 [klein] 9B KV](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) | ✅ | ✅ | ✅ | ✅ | ✅ | [Giấy phép phi thương mại FLUX](model_licenses/LICENSE-FLUX-NON-COMMERICAL) |
| [FLUX.2 [klein] 4B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | ❌ | ❌ | ✅ | ✅ | ✅ | [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| [FLUX.2 [klein] 9B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) | ❌ | ❌ | ✅ | ✅ | ✅ | [Giấy phép phi thương mại FLUX](model_licenses/LICENSE-FLUX-NON-COMMERICAL) |
| [FLUX.2 [dev]](https://huggingface.co/black-forest-labs/FLUX.2-dev) | ❌ | ✅ | ✅ | ✅ | ✅ | [Giấy phép phi thương mại FLUX](model_licenses/LICENSE-FLUX-NON-COMMERICAL) |

**Tất cả các mô hình đều hỗ trợ**: Text-to-Image ✅ | Chỉnh sửa đơn tham chiếu ✅ | Chỉnh sửa đa tham chiếu ✅

## Tôi nên chọn mô hình nào?

| Nhu cầu | Khuyên dùng |
|---|---|
| Ứng dụng thời gian thực, quy trình tương tác nhanh | [klein] 4B, 9B, hoặc 9B KV (bản chưng cất - distilled) |
| GPU người dùng cá nhân (ví dụ: RTX 3090/4070) | [klein] 4B |
| Fine-tuning, huấn luyện LoRA | [klein] Base hoặc FLUX.2 [dev] |
| Chất lượng cao nhất, không bị giới hạn độ trễ | FLUX.2 [dev] |

## `FLUX.2 [klein]`

FLUX.2 [klein] là dòng mô hình nhanh nhất của chúng tôi — có thể sinh và chỉnh sửa (nhiều) ảnh trong thời gian dưới 1 giây mà không làm suy giảm chất lượng. Được thiết kế chuyên biệt cho các ứng dụng thời gian thực, thử nghiệm sáng tạo lặp đi lặp lại và triển khai trên phần cứng người dùng thông thường.

### Các khả năng cốt lõi
- **Suy luận dưới 1 giây (Sub-second inference)** — Tạo hoặc chỉnh sửa ảnh chỉ trong chưa đầy một giây trên phần cứng hiện đại.
- **Tạo & chỉnh sửa hợp nhất** — Tích hợp Text-to-image, chỉnh sửa ảnh và đa tham chiếu trong cùng một mô hình duy nhất.
- **Chạy trên GPU cá nhân** — Bản Klein 4B chỉ cần khoảng ~8GB VRAM (từ RTX 3090/4070 trở lên).
- **Giấy phép Apache 2.0 trên bản 4B** — Mã nguồn mở, tự do fine-tune và thương mại hóa.

### Hiệu năng

Các mô hình Klein xác lập ranh giới Pareto mới về tương quan giữa chất lượng (Elo) so với độ trễ (Latency) và dung lượng VRAM trên cả 3 tác vụ: text-to-image, chỉnh sửa đơn tham chiếu và sinh ảnh đa tham chiếu:

<p align="center">
<img src="assets/klein_benchmark.jpg" alt="FLUX.2 [klein] vs Baselines — Elo vs Latency and VRAM" width="800"/>
</p>
<sub>Điểm Elo càng cao + Độ trễ/VRAM càng thấp = Càng tốt.</sub>

### Dòng mô hình Klein

| Mô hình | Phù hợp nhất cho |
|:---|:---|
| **[klein] 4B** | Tốc độ tối đa, phần cứng phổ thông, triển khai thiết bị biên (edge) |
| **[klein] 9B** | Chất lượng text-to-image cao; đối với chỉnh sửa ảnh, bản 9B KV nhanh hơn ở chất lượng tương đương |
| **[klein] 9B KV** | Tỷ lệ chất lượng/độ trễ tốt nhất, nhanh hơn 4B khi chỉnh sửa ảnh đa tham chiếu nhờ [KV caching](docs/flux2_klein_kv_cache.md) |
| **[klein] 4B Base** | Fine-tuning trên phần cứng giới hạn, tùy biến hoàn toàn |
| **[klein] 9B Base** | Nghiên cứu, huấn luyện LoRA, độ đa dạng đầu ra tối đa |

**Phân biệt bản Chưng cất (Distilled) vs Bản Nền tảng (Base):**
- Sử dụng **Distilled** (4 bước) cho các ứng dụng sản phẩm thực tế và sinh ảnh thời gian thực.
- Sử dụng **Base** (50 bước) cho fine-tuning, huấn luyện LoRA và sự linh hoạt tối đa.

**Giấy phép (Licensing):** Các bản 4B phát hành theo giấy phép [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md). Các bản 9B sử dụng [Giấy phép phi thương mại FLUX.2-dev](model_licenses/LICENSE-FLUX-DEV).

### Ví dụ Text-to-image

Ví dụ tập trung vào tính chân thực:
![t2i-klein-grid](assets/t2i_klein_realism.jpg)

Ví dụ tập trung vào độ đa dạng đầu ra:
![t2i-klein-others](assets/t2i_klein_others.jpg)

### Ví dụ Chỉnh sửa ảnh

![i2i-klein](assets/i2i_klein.jpg)

## `FLUX.2 [dev]`

`FLUX.2 [dev]` là mô hình flow matching transformer 32 tỷ tham số (32B) có khả năng sinh và chỉnh sửa (nhiều) ảnh. Mô hình được phát hành theo [Giấy phép phi thương mại FLUX.2-dev](model_licenses/LICENSE-FLUX-DEV) và có thể tải tại [đây](https://huggingface.co/black-forest-labs/FLUX.2-dev).

Lưu ý rằng đoạn script bên dưới cho `FLUX.2 [dev]` đòi hỏi dung lượng VRAM đáng kể (tương đương GPU H100). Chúng tôi đã hợp tác với Hugging Face để tạo ra các phiên bản lượng tử hóa có thể chạy trên phần cứng người dùng cá nhân; bên dưới là hướng dẫn cách chạy trên RTX 4090 với remote text-encoder, đối với các mức kích thước lượng tử hóa và cấu hình khác, hãy xem [hướng dẫn lượng tử hóa diffusers tại đây](docs/flux2_dev_hf.md).

### Ví dụ Text-to-image

![t2i-grid](assets/teaser_generation.png)

### Ví dụ Chỉnh sửa ảnh

![edit-grid](assets/teaser_editing.png)

### Tăng cường câu lệnh (Prompt upsampling)

`FLUX.2 [dev]` được hưởng lợi đáng kể từ kỹ thuật prompt upsampling. Script suy luận dưới đây cung cấp tùy chọn sử dụng cả prompt upsampling cục bộ bằng chính mô hình được dùng làm text encoder ([`Mistral-Small-3.2-24B-Instruct-2506`](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)), hoặc tùy chọn sử dụng bất kỳ mô hình nào trên [OpenRouter](https://openrouter.ai/) thông qua lệnh gọi API.

Xem [hướng dẫn upsampling](docs/flux2_with_prompt_upsampling.md) để biết thêm chi tiết và hướng dẫn khi nào nên sử dụng.

## `FLUX.2` autoencoder

AutoEncoder của FLUX.2 đã được cải tiến vượt bậc so với [AutoEncoder của FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors). AutoEncoder được phát hành theo giấy phép [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) và có thể tải tại [đây](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors). Để biết thêm chi tiết, xem [bài viết kỹ thuật của chúng tôi](https://bfl.ai/research/representation-comparison).

## Cài đặt cục bộ

Mã suy luận đã được thử nghiệm trên GB200 với CUDA 12.9 và Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir
```

## Chạy giao diện dòng lệnh (CLI)

Trước khi chạy CLI, bạn có thể tải các trọng số từ [đây](https://huggingface.co/black-forest-labs/FLUX.2-dev) và thiết lập các biến môi trường sau:

```bash
export FLUX2_MODEL_PATH="<flux2_path>"
export AE_MODEL_PATH="<ae_path>"
export KLEIN_4B_MODEL_PATH="<klein_4b_path>"
export KLEIN_4B_BASE_MODEL_PATH="<klein_4b_base_path>"
export KLEIN_9B_MODEL_PATH="<klein_9b_path>"
export KLEIN_9B_KV_MODEL_PATH="<klein_9b_kv_path>"
export KLEIN_9B_BASE_MODEL_PATH="<klein_9b_base_path>"
```

Nếu bạn không thiết lập các biến môi trường, trọng số sẽ tự động được tải xuống từ Hugging Face Hub.

Bạn có thể bắt đầu phiên tương tác dòng lệnh để thực hiện tạo ảnh từ văn bản cũng như chỉnh sửa (một hoặc nhiều) hình ảnh bằng lệnh sau:

```bash
PYTHONPATH=src python scripts/cli.py
```

## Thủy vân vô hình (Watermarking)

Chúng tôi đã bổ sung tùy chọn nhúng thủy vân vô hình trực tiếp vào các hình ảnh được sinh ra thông qua [thư viện invisible-watermark](https://github.com/ShieldMnt/invisible-watermark).

Ngoài ra, chúng tôi khuyến nghị áp dụng các giải pháp ghi nhận siêu dữ liệu cho các sản phẩm đầu ra của bạn, chẳng hạn như tiêu chuẩn [C2PA](https://c2pa.org/).

## Trích dẫn (Citation)

Nếu bạn thấy mã nguồn hoặc các mô hình này hữu ích cho nghiên cứu của mình, vui lòng trích dẫn như sau:

```bib
@misc{flux-2-2025,
    author={Black Forest Labs},
    title={{FLUX.2: Frontier Visual Intelligence}},
    year={2025},
    howpublished={\url{https://bfl.ai/blog/flux-2}},
}
```
