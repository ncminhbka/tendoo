# Tăng cường câu lệnh (Prompt upsampling) với FLUX.2

Kỹ thuật tăng cường câu lệnh (Prompt upsampling) sử dụng một mô hình thị giác - ngôn ngữ lớn (Vision-Language Model) để mở rộng và làm phong phú câu lệnh của bạn trước khi đưa vào sinh ảnh. Điều này giúp cải thiện đáng kể kết quả đối với các tác vụ tạo ảnh phức tạp và đòi hỏi khả năng suy luận logic cao.

## Khi nào nên sử dụng prompt upsampling

Prompt upsampling đặc biệt hiệu quả đối với các câu lệnh đòi hỏi khả năng suy luận hoặc diễn giải phức tạp:

- **Tạo văn bản trong ảnh**: Tạo meme, áp phích (poster) hoặc các hình ảnh đòi hỏi mô hình phải tạo ra văn bản sáng tạo hoặc phù hợp với ngữ cảnh.
- **Chỉ dẫn dựa trên hình ảnh**: Các prompt mà trong đó ảnh đầu vào có chứa văn bản đè lên, mũi tên hoặc các chú thích cần được diễn giải (ví dụ: "follow the instructions in the image", "read the diagram and generate the result").
- **Suy luận code và toán học**: Tạo các hình ảnh minh họa thuật toán, khái niệm toán học hoặc sơ đồ luồng code (code flow diagrams) nơi cấu trúc logic đóng vai trò quan trọng.

Đối với các câu lệnh đơn giản, trực tiếp (ví dụ: "a red car"), prompt upsampling có thể không mang lại lợi ích rõ rệt.

## Các phương pháp

Chúng tôi cung cấp hai phương pháp cho kỹ thuật prompt upsampling:

### 1. Prompt upsampling qua API (Khuyên dùng)

Prompt upsampling dựa trên API thông qua [OpenRouter](https://openrouter.ai/) nhìn chung mang lại kết quả tốt hơn nhờ tận dụng các mô hình mạnh mẽ hơn.

Thiết lập API key của bạn dưới dạng biến môi trường:

```bash
export OPENROUTER_API_KEY="<api_key>"
```

Sau đó chạy CLI với chế độ upsampling được bật:
```bash
export PYTHONPATH=src
python scripts/cli.py --upsample_prompt_mode=openrouter
```

Bạn có thể chuyển đổi giữa các mô hình khác nhau bằng cách dùng cờ `--openrouter_model=<tên_mô_hình>`.

Ngoài ra, bạn cũng có thể khởi động CLI đơn giản bằng lệnh:

```bash
export PYTHONPATH=src
python scripts/cli.py
```

và chọn mô hình prompt upsampling tương tác trực tiếp trong giao diện dòng lệnh.

**Ví dụ kết quả đầu ra:**

| Prompt: "Make a meme about generating memes with this model" |
|:---:|
| <img src="../assets/t2i_upsample_example.png" alt="Output" width="512"> |

### 2. Prompt upsampling cục bộ (Local)

Prompt upsampling cục bộ sử dụng mô hình [`Mistral-Small-3.2-24B-Instruct-2506`](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506), đây cũng chính là mô hình được dùng để mã hóa văn bản (text encoder) trong `FLUX.2 [dev]`. Tùy chọn này không yêu cầu API key nhưng phần mở rộng prompt có thể kém chi tiết hơn.

Để bật prompt upsampling cục bộ, hãy sử dụng `--upsample_prompt_mode=local`.

**Ví dụ:**

<table>
  <tr>
    <th colspan="2" style="text-align: center;">Prompt: "Describe what the red arrow is seeing"</th>
  </tr>
  <tr>
    <th>Input</th>
    <th>Output</th>
  </tr>
  <tr>
    <td align="center"><img src="../assets/i2i_upsample_input.png" alt="Input image"></td>
    <td align="center"><img src="../assets/i2i_upsample_example.png" alt="Output image"></td>
  </tr>
</table>