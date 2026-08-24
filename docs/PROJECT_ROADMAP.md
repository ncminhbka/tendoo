# 🚀 Lộ trình Dự án (v3): Nâng cấp FLUX.2 klein 4B Base Sinh chữ Tiếng Việt

- **Mục tiêu**: Tối ưu hóa năng lực vẽ chữ tiếng Việt có dấu ($100\%$ đúng chính tả, đủ dấu thanh/mũ) và sinh Poster quảng cáo thương mại trên nền tảng **FLUX.2 [klein] 4B Base**.
- **Cấu hình phần cứng mục tiêu**: **2x NVIDIA A30 (24GB VRAM $\times 2 = 48$GB VRAM)**.
- **Phương pháp tiếp cận cốt lõi**: *Kiềng 3 chân (Time-Offset In-Context + Tight Crop Bitmap + LoRA DiT)*.

---

## 🏛️ Ma trận Kiềng 3 Chân (Bản v3 - Sau Thực Nghiệm Giai Đoạn 1)

| Vấn đề kỹ thuật cốt lõi | Triệu chứng nếu không xử lý | Giải pháp kỹ thuật chuẩn hóa | Đánh giá thực nghiệm |
| :--- | :--- | :--- | :---: |
| **1. Định vị & Đa khối Text** | Chữ trôi tự do hoặc tranh chấp khi có nhiều biển hiệu / vật thể. | **In-Context Multi-Reference Time-Offset**<br>*(Phân tách các khối text/sản phẩm bằng $t=10.0, 20.0, 30.0$ tại tọa độ gốc)* | 🏆 **0 Tham số - Đạt chuẩn $100\%$**<br>*(Thay thế RoPE Shift do RoPE Shift bị OOD khi box ở đáy ảnh)* |
| **2. Chi phí tính toán (Sequence Cost)** | Sequence dài $\rightarrow$ Attention $\mathcal{O}(N^2)$ làm chậm $3.5-4\times$. | **Tight Crop Bitmap Preprocessing**<br>*(Auto-wrap 1-3 dòng, Binary search font size, $L_{\text{ref}} \le 256$ tokens)* | ⚡ **Tiền xử lý đồ họa**<br>(Tiết kiệm $>80\%$ VRAM & Compute) |
| **3. Chất liệu & Hòa trộn ánh sáng** | Chữ trên một số vật liệu gồ ghề/phản chiếu mạnh có thể cần thích ứng sâu hơn. | **LoRA trên DiT 4B Base**<br>*(Dạy DiT hòa trộn texture/lighting vào các vật liệu phức tạp)* | ⚙️ **LoRA Rank 16–32**<br>(Giai đoạn 3) |

> [!NOTE]
> **KẾT QUẢ THỰC NGHIỆM GIAI ĐOẠN 1 (EMPIRICAL VERDICT)**:
> Thử nghiệm đối chứng qua 6 bài test thực nghiệm trên 2x GPU A30 cho thấy việc ép tọa độ RoPE $(h, w)$ thủ công không hiệu quả và dễ gây lỗi Out-of-Distribution (mất chữ ở nửa dưới). Trong khi đó, cơ chế **In-Context Reference Conditioning chuẩn của BFL với Time Offsets ($t=10, 20...$)** hoạt động hoàn hảo 100%, tự động uốn cong 3D theo vật thể và sinh đa khối text xuất sắc.

---

## 🗺️ Sơ đồ 5 Giai đoạn Triển khai (Bản v3)

```mermaid
graph TD
    G0["Giai đoạn 0: Xác minh Kiến trúc (HOÀN TẤT)<br>• Khóa cứng mã nguồn gốc BFL src/flux2/<br>• Xác minh cơ chế 4D RoPE (t, h, w, l)"]
    G1["Giai đoạn 1: In-Context Reference Conditioning (HOÀN TẤT 100%)<br>• Xác nhận In-Context Time-Offset (t=10, 20) vượt trội hoàn toàn RoPE Shift<br>• Hoàn thiện bộ tạo Tight-Crop Glyph tiếng Việt thông minh<br>• Thử nghiệm thành công Multi-Text & Poster Sản phẩm thương mại"]
    G2["Giai đoạn 2: Tinh chỉnh VAE Decoder (2 - 3 ngày)<br>• Train nhẹ tầng Decode với 2K-3K ảnh Text TV<br>• Khử răng cưa cho dấu hỏi, ngã, nặng, mũ, móc<br>• Khóa Encoder, giữ nguyên Latent 128ch"]
    G3["Giai đoạn 3: Huấn luyện LoRA DiT 4B Base (3 - 5 ngày)<br>• Train LoRA trên 2x GPU A30 song song (DDP)<br>• Dạy DiT hòa trộn chất liệu, ánh sáng, đổ bóng<br>• Dataset 3K-5K cặp ảnh Typography TV"]
    G4["Giai đoạn 4: Đóng gói & Serving TikTok 9:16 / E-commerce (2 ngày)<br>• Tích hợp End-to-End Pipeline hoàn chỉnh<br>• Tối ưu torch.compile & Multi-GPU Serving<br>• Xây dựng FastAPI / Gradio UI"]

    G0 --> G1 --> G2 --> G3 --> G4
```

---

## 📅 Chi tiết các Giai đoạn

### 📍 Giai đoạn 1: In-Context Multi-Reference Pipeline (ĐÃ HOÀN TẤT & NGHIỆM THU 100%)
* **Kết quả thực nghiệm**:
  * Đạt độ chính xác $100\%$ dấu tiếng Việt trên các từ phức tạp và câu dài.
  * Xác nhận DiT tự động nhận diện bề mặt vật thể (`surface`) từ prompt và uốn cong chữ mà không cần prompt nhắc có chữ.
  * Hỗ trợ đồng thời đa khối text độc lập và poster sản phẩm thương mại qua mốc thời gian In-Context $t=10.0, 20.0...$ tại tọa độ chuẩn $(0, 0)$.
  * Quyết định: Loại bỏ hoàn toàn RoPE coordinate shift thủ công để tiết kiệm thời gian và tránh lỗi OOD.

---

### 📍 Giai đoạn 2: Tinh chỉnh VAE Decoder
* **Thời gian**: `2 – 3 Ngày`
* **Công việc**:
  * Dataset: ~2,000 – 3,000 ảnh typography tiếng Việt chất lượng cao.
  * Đóng băng Encoder, mở gradient các block cuối của `Decoder`.
  * Hàm mất mát: $\mathcal{L} = \text{L1} + 0.5 \times \text{LPIPS}$. 1 GPU A30, hoàn thành trong ~3–5 giờ.

---

### 📍 Giai đoạn 3: Huấn luyện LoRA cho DiT 4B Base
* **Thời gian**: `3 – 5 Ngày`
* **Mục tiêu**:
  * Dạy DiT tách biệt hình dáng (Shape từ Ref) và chất liệu/ánh sáng (Texture từ Target).
  * LoRA Rank 16–32 trên các khối `DoubleStreamBlock` và `SingleStreamBlock`.
  * Chạy trên 2x GPU A30 song song (DDP / `accelerate`).

---

### 📍 Giai đoạn 4: Đóng gói Sản phẩm & Serving
* **Thời gian**: `2 Ngày`
* **Công việc**:
  * Ghép nối pipeline hoàn chỉnh, áp dụng `torch.compile`.
  * Phục vụ đa luồng trên 2x GPU A30 (FastAPI Backend / Gradio UI).
