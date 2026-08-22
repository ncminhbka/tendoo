# FLUX.2 [klein] 9B KV Cache

FLUX.2 [klein] 9B KV là một biến thể của mô hình klein 9B được tối ưu hóa cho tác vụ chỉnh sửa ảnh nhanh (fast image editing). Mô hình sử dụng kỹ thuật KV caching nhằm tránh việc phải tính toán lại attention một cách dư thừa trên các token của ảnh tham chiếu (reference image tokens) ở mỗi bước khử nhiễu (denoising step), mang lại tốc độ xử lý nhanh hơn đáng kể khi làm việc với các ảnh tham chiếu.

## Cơ chế hoạt động

Trong quy trình chỉnh sửa ảnh tiêu chuẩn, mỗi bước khử nhiễu sẽ ghép nối (concatenate) các token của ảnh tham chiếu với các token nhiễu đầu ra và thực hiện full attention trên toàn bộ chuỗi sequence. Vì các token tham chiếu không thay đổi qua các bước, điều này gây lãng phí tài nguyên tính toán.

Biến thể KV cache chia quá trình khử nhiễu thành hai giai đoạn:

1. **Bước 0 — `forward_kv_extract`**: Thực hiện một lượt forward pass đầy đủ bao gồm cả các token tham chiếu. Trích xuất và lưu vào bộ nhớ đệm (cache) các phép chiếu key/value (KV) cho các token tham chiếu này.
2. **Từ bước 1 trở đi — `forward_kv_cached`**: Chỉ thực hiện forward pass với các token đầu ra và token văn bản. Tái sử dụng các KV của ảnh tham chiếu đã lưu trong cache thông qua phép ghép nối (concatenation) tại các tầng attention.

Điều này có nghĩa là các token của ảnh tham chiếu chỉ cần xử lý duy nhất một lần, bất kể số bước khử nhiễu là bao nhiêu.

## Mức độ tăng tốc (Speedup)

Tốc độ tăng tốc phụ thuộc vào tỷ lệ giữa số lượng token tham chiếu và token đầu ra. Càng nhiều ảnh tham chiếu và độ phân giải đầu ra càng nhỏ thì hiệu quả tăng tốc càng lớn:

| Số lượng ảnh tham chiếu (1024x1024 mỗi ảnh) | 512x512 | 768x768 | 1024x1024 | 1440x1440 |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 1.78x | 1.57x | 1.40x | 1.21x |
| 2 | 2.16x | 1.97x | 1.77x | 1.46x |
| 3 | 2.43x | 2.21x | 1.99x | 1.69x |
| 4 | 2.66x | 2.44x | 2.22x | 1.85x |

## Cài đặt

Làm theo [hướng dẫn cài đặt tiêu chuẩn](../README.md#local-installation).

## Cách sử dụng

### Biến môi trường

```bash
export KLEIN_9B_KV_MODEL_PATH="/path/to/flux-2-klein-9b-kv.safetensors"
export AE_MODEL_PATH="/path/to/ae.safetensors"  # tùy chọn, sẽ tự động tải nếu chưa thiết lập
```

### CLI

CLI sẽ tự động sử dụng luồng KV cache khi bạn chọn mô hình `flux.2-klein-9b-kv`:

```bash
PYTHONPATH=src python scripts/cli.py --model_name flux.2-klein-9b-kv
```

Sau đó cung cấp các ảnh tham chiếu để chỉnh sửa:

```
> input_images="ref1.jpg,ref2.jpg"
> prompt="a cat wearing sunglasses"
> run
```
