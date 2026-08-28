# SUB-PLAN CHUYÊN SÂU: PIPELINE CHẾ TẠO DỮ LIỆU MILESTONE A (800 MẪU ĐỘC BẢN)

Tài liệu này là đặc tả kỹ thuật chi tiết cho hợp phần **Data Generation Engine** của **Milestone A (800 mẫu huấn luyện)** nhằm phục vụ mở khóa năng lực định tuyến phân luồng 2 Slots ($t=10.0$ và $t=20.0$) trên mô hình `FLUX.2-klein-base-4B`.

---

## 📊 1. MA TRẬN PHÂN BỔ ĐA CHIỀU 800 MẪU (MULTIDIMENSIONAL MATRIX)

Để ngăn chặn tuyệt đối hiện tượng mô hình bị "học vẹt", toàn bộ 800 mẫu huấn luyện được kiểm soát đa biến số độc lập:

### 1.1. Ma trận 7 Nghiệp vụ Use-Case Thực tế của Tendoo AI $\times$ Modality Split

| STT | Nghiệp Vụ Sử Dụng Thực Tế | Quy Mô Mẫu | Tỷ Lệ | I2I (SP-Anchor: 1 SP + 1 Text) | Pure T2I (2 Texts: Slot 1 + Slot 2) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Poster Khuyến Mại / Flash Sale** | **170** | $21.25\%$ | $90$ mẫu (Đồ gia dụng, công nghệ, thời trang) | $80$ mẫu (Dịch vụ F&B, vé sự kiện, voucher) |
| **2** | **Poster / Banner Quảng Cáo Sản Phẩm** | **170** | $21.25\%$ | $170$ mẫu (Chai nước hoa, đồng hồ, điện thoại) | $0$ mẫu (100% là sản phẩm thật) |
| **3** | **Card Feedback / Đánh Giá Khách Hàng** | **110** | $13.75\%$ | $60$ mẫu (Ảnh khách dùng sản phẩm) | $50$ mẫu (Feedback dịch vụ Gym, Spa, Khóa học) |
| **4** | **Banner Khai Trương / Sự Kiện** | **100** | $12.50\%$ | $40$ mẫu (Khai trương cửa hàng điện máy/cà phê) | $60$ mẫu (Banner sự kiện, hội thảo, triển lãm) |
| **5** | **Ảnh Tin Tuyển Dụng (Recruitment)** | **90** | $11.25\%$ | $20$ mẫu (Tuyển dụng văn phòng có ảnh trụ sở) | $70$ mẫu (Thẻ tuyển dụng đồ họa hiện đại) |
| **6** | **Quy Trình / Hướng Dẫn (2 Bước)** | **80** | $10.00\%$ | $30$ mẫu (Hướng dẫn sử dụng thiết bị/sản phẩm) | $50$ mẫu (Quy trình dùng app, đăng ký dịch vụ) |
| **7** | **Sáng Tạo Tự Do / Quote / Bìa Sách** | **80** | $10.00\%$ | $30$ mẫu (Bìa sách, album, tác phẩm nghệ thuật) | $50$ mẫu (Trích dẫn danh ngôn, thơ ca chữ nổi) |
| **Σ** | **TỔNG CỘNG MILESTONE A** | **800** | **$100\%$** | **440 mẫu ($55\%$)** | **360 mẫu ($45\%$)** |

---

### 1.2. Ma trận 4 Tỉ lệ Khung hình (Aspect Ratio Bucketing)

Tỉ lệ khung hình được phân bổ cố định và trải đều trên cả 2 nhánh I2I và T2I:

| Tỉ Lệ Bucket | Kích Thước Pixel | Latent Patch ($16\times$) | Tỷ Lệ | Số Lượng Mẫu I2I | Số Lượng Mẫu T2I | Tổng Mẫu | Ứng Dụng Nghiệp Vụ Thực Tế |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1:1 (Vuông)** | $1024 \times 1024$ | $64 \times 64$ ($4096$ tokens) | **$35\%$** | $154$ | $126$ | **$280$ mẫu** | Feed Facebook, Instagram, E-commerce Post |
| **9:16 (Dọc)** | $768 \times 1344$ | $48 \times 84$ ($4032$ tokens) | **$35\%$** | $154$ | $126$ | **$280$ mẫu** | TikTok Ads, Reels, Story, Standee quảng cáo |
| **4:5 (Dọc vừa)**| $896 \times 1152$ | $56 \times 72$ ($4032$ tokens) | **$15\%$** | $66$ | $54$ | **$120$ mẫu** | Instagram Portrait tối ưu diện tích màn hình mobile |
| **16:9 (Ngang)** | $1344 \times 768$ | $84 \times 48$ ($4032$ tokens) | **$15\%$** | $66$ | $54$ | **$120$ mẫu** | Banner Website, Fanpage Cover, Biển hiệu sự kiện |
| **TỔNG** | — | — | **$100\%$** | **$440$** | **$360$** | **$800$ mẫu** | — |

---

### 1.3. Ma trận Phân tầng Độ dài Văn bản (Golden 75/25 Ratio)

Để chống lại việc DiT tự ý suy diễn rằng *"Slot 1 luôn ngắn, Slot 2 luôn dài"*:
* **Phân tầng Tự nhiên Thương mại ($75\% = 600$ mẫu)**:
  - Slot 1 ($t=10.0$): Ngắn đến vừa ($2 - 5$ từ, $1$ dòng hoặc $2$ dòng ngắn).
  - Slot 2 ($t=20.0$): Vừa đến dài ($4 - 12$ từ, $1 - 2$ dòng).
* **Phân tầng Nghịch đảo Chủ đích ($25\% = 200$ mẫu)**:
  - Slot 1 ($t=10.0$) **CỰC DÀI**: $8 - 16$ từ ($2 - 3$ dòng) (triết lý, câu mở đầu, đoạn thơ).
  - Slot 2 ($t=20.0$) **CỰC NGẮN**: $1 - 3$ từ (*"MIỄN PHÍ"*, *"5G PRO"*, *"ĐỒNG GIÁ 99K"*, *"BƯỚC 1"*).

---

### 1.4. Phân bổ Tiếp xúc Tự nhiên Văn hóa Việt (Incidental Cultural Baking)
* **Số lượng mẫu có bối cảnh văn hóa Việt**: **120 mẫu (chiếm đúng $15\%$ tổng 800 mẫu)**.
* **Cơ cấu**:
  - *50 mẫu Ẩm thực & F&B truyền thống*: Phở Thìn, bánh mì kẹp thịt, bún chả, cà phê phin rang mộc, chè sen Long Nhãn.
  - *40 mẫu Đời sống, Du lịch & Lễ hội*: Tà áo dài truyền thống, nón lá, chợ hoa Tết rực rỡ, hoàng hôn hồ Gươm, phố cổ Hội An rực rỡ đèn lồng.
  - *30 mẫu Viễn thông & Doanh nghiệp Việt*: Trạm phát sóng Viettel 5G, bảng hiệu Viettel Money hiện đại, chuyển đổi số quốc gia.
* **Nguyên tắc kỹ thuật**: 
  - Chỉ tạo bối cảnh thị giác chân thực qua Prompt Teacher để Teacher vẽ đúng.
  - Không gán loss hay mục tiêu fine-tune riêng $\lambda_{\text{culture}}$, tránh làm loãng gradient của bài toán định tuyến chú ý (Spatial Attention Routing).

---

## 🎨 2. NGUYÊN TẮC TRỰC GIAO HÓA FONT VÀ SÀN PHÂN GIẢI KÉP

### 2.1. Phân bổ Font Ngẫu nhiên Độc lập ($I(\text{Font}; \text{Domain}) = 0$)
* Tuyệt đối không gán chết một font vào một ngành hàng.
* Mỗi mẫu huấn luyện chọn ngẫu nhiên độc lập trong kho **16 Font Unicode**:
  - **Slot 1**: Bốc ngẫu nhiên 1 trong 16 font với xác suất đều đặn $P = 1/16$ ($\approx 50$ mẫu/font).
  - **Slot 2**: $75\%$ trường hợp chọn font khác Slot 1 (để học cách phối hợp 2 font khác phong cách); $25\%$ trường hợp chọn cùng font (để học phân cấp kích thước đồng bộ).
* **Tổng lượt xuất hiện mỗi font trong 800 mẫu**: $\approx 100$ lần, đảm bảo phá vỡ hoàn toàn mọi định kiến giả giữa phong cách hình học của chữ và ngữ cảnh của ảnh.

### 2.2. Khóa Cứng Sàn Phân Giải Kép (Locked Dual-Floor Architecture)
* **`BeVietnamPro-Black`**: Sàn tối thiểu **$32\text{pt}$** (độ phân giải tối thiểu cho nét chữ không bị gãy).
* **Toàn bộ 15 font còn lại** (`anton`, `gotham`, `lolapeluza`, `gretoon`, `playfair`, `oswald`, `harabaras`, `dancing`, `pacifico`, `sedgwick`, `blowbrush`, `clementine`, `cookies`, `grocery`, `holidays`): Sàn tối thiểu **$36\text{pt}$**.

---

## 📝 3. KHO NGỮ LIỆU 1,600 CỤM VĂN BẢN ĐỘC BẢN (ZERO-DUPLICATION TEXT CORPUS)

Để triệt tiêu hiện tượng lặp lại instance gây overfit từ ngữ:
* Xây dựng một kho dữ liệu gồm **1,600 cụm text tiếng Việt nguyên bản** (tương ứng 800 cặp Text 1 & Text 2) trải đều trên 10 lĩnh vực:
  1. *Công nghệ, Viễn thông & AI (Viettel 5G, Cloud)*: 100 cặp.
  2. *F&B, Cà phê, Trà sữa & Nhà hàng*: 120 cặp.
  3. *Mỹ phẩm, Spa & Chăm sóc da*: 100 cặp.
  4. *Thời trang, Giày dép & Streetwear*: 90 cặp.
  5. *Giáo dục, Khóa học & Tiếng Anh*: 70 cặp.
  6. *Thể thao, Gym, Yoga & Fitness*: 70 cặp.
  7. *Nội thất, Gia dụng & Không gian sống*: 70 cặp.
  8. *Y tế, Nha khoa & Chăm sóc sức khỏe*: 60 cặp.
  9. *Bất động sản & Du lịch*: 60 cặp.
  10. *Văn hóa Việt Nam (Ẩm thực, Di tích, Lễ hội)*: 60 cặp.
* **Nguyên tắc lấy mẫu không hoàn lại (Sampling without replacement)**: Mỗi cặp văn bản chỉ được gán cho đúng $1$ mẫu huấn luyện duy nhất. **Không có bất kỳ 2 mẫu nào trùng text trong toàn bộ dataset!**

---

## 🔄 4. QUY TRÌNH CHẾ TẠO DỮ LIỆU TIỀN ĐỊNH (PRE-DETERMINED TYPOGRAPHY PIPELINE)

```
[ BƯỚC 1: SPECIFICATION SAMPLER ]
  Pipeline chọn ngẫu nhiên bộ tham số cho mẫu:
  • Use-Case (1/7), Domain (1/10), Aspect Ratio (1/4), Length Stratum (75/25).
  • Cặp Text: Text 1 (Ví dụ: 1 dòng) và Text 2 (Ví dụ: 2 dòng ngắt bằng \n).
  • Cặp Font: Font 1 (sàn 32/36pt) và Font 2 (sàn 32/36pt).
                               │
                               ▼
[ BƯỚC 2: RENDER GLYPH BITMAPS (100% TIỀN ĐỊNH) ]
  • Gọi `glyph_engine.py`:
    - Glyph 1: Render Text 1 theo đúng cấu trúc dòng và sàn font size.
    - Glyph 2: Render Text 2 theo đúng cấu trúc dòng và sàn font size.
  ===> Thu được `glyph_1.png` và `glyph_2.png` hoàn mỹ về mặt hình học!
                               │
                               ▼
[ BƯỚC 3: PROMPT TEACHER (CHỈ THỊ RÕ CẤU TRÚC DÒNG) ]
  • Tạo Prompt chi tiết gửi sang OpenAI API (`gpt-image-2` quality="low"):
    "Commercial recruitment poster for tech corporation. At top-center, on 1 single wide line,
     bold metallic golden text says 'KỸ SƯ TRÍ TUỆ NHÂN TẠO'. Below it, on 2 neat lines,
     clean white text says: Line 1: 'Thu nhập hấp dẫn', Line 2: 'Môi trường sáng tạo'..."
  • Nhận về ảnh Target Ground Truth `target_{id}.png`.
                                │
                                ▼
[ BƯỚC 4: SINH STUDENT CLEAN PROMPT (QUY TẮC ORDINAL (1)/(2)) ]
  • Xóa 100% chữ nguyên văn để chống Representation Clash cho Qwen3.
  • Gắn thẻ định danh thứ tự chuẩn:
    "(1) Dòng chữ vị trí tuyển dụng ở phía trên dập nổi mạ vàng sang trọng.
     (2) Khối thông tin đãi ngộ 2 dòng ở bên dưới nét chữ màu trắng thanh mảnh rõ nét."
```

---

## 🛡️ 5. CHIẾN LƯỢC KIỂM SOÁT CHI PHÍ & 3 CHỐT AN TOÀN API (3-TIER SAFETY GATES)

| Giai Đoạn Thực Hiện | Quy Mô Mẫu | Ngân Sách API Dự Kiến | Mục Tiêu & Tiêu Chí Nghiệm Thu (Pass Criteria) |
| :--- | :---: | :---: | :--- |
| **Đợt 1: Smoke Test** | **10 mẫu** | $\approx 0.20\text{ USD}$ | • Mở xem trực tiếp 10 ảnh: Xác nhận `gpt-image-2` tuân thủ đúng số dòng đã chỉ định.<br>• Xác nhận Glyph render đúng sàn phân giải và không lỗi font.<br>• Xác nhận Student Prompt đúng chuẩn `(1)` / `(2)` không dính chữ thật. |
| **Đợt 2: Pilot Micro-Run** | **50 mẫu mới**<br>*(Đạt đủ 60 mẫu)* | $\approx 1.00\text{ USD}$ | • Đóng gói 60 mẫu đưa lên server 2x A30 chạy thử nghiệm 50 steps ODE.<br>• Xác nhận hàm loss Flow Matching giảm đều từ $\sim 0.85 \rightarrow \le 0.25$.<br>• Xác nhận không dính lỗi OOM và gradient ổn định. |
| **Đợt 3: Sản Xuất Toàn Bộ** | **740 mẫu còn lại**<br>*(Hoàn tất 800 mẫu)* | $\approx 14.80\text{ USD}$ | • Chạy đa luồng tự động có cơ chế Checkpoint & Resume (nếu đứt mạng tự động chạy tiếp từ mẫu dở dang).<br>• Đóng gói toàn bộ 800 mẫu thành dataset hoàn chỉnh. |
| **TỔNG NGÂN SÁCH API** | **800 mẫu** | **$\approx 16.00\text{ USD}$** | **(~400.000 VNĐ) — An toàn, tiết kiệm và được kiểm soát chặt chẽ từng bước!** |

---

## 📦 6. CẤU TRÚC LƯU TRỮ VÀ DỮ LIỆU ĐẦU RA

Sau khi hoàn tất, thư mục dữ liệu Milestone A sẽ có cấu trúc chuẩn hóa:

```
data/milestone_a/
├── dataset_manifest.jsonl      # 800 dòng metadata JSON
├── targets/                    # 800 ảnh Ground Truth do Teacher sinh (PNG)
│   ├── target_0001.png
│   └── ...
├── glyphs/                     # 1,600 ảnh Glyph đen-trắng do glyph_engine render
│   ├── glyph_0001_slot10.png
│   ├── glyph_0001_slot20.png
│   └── ...
└── products/                   # 440 ảnh sản phẩm thật (cho nhánh SP-Anchor)
    ├── prod_0001.png
    └── ...
```

Mỗi dòng trong `dataset_manifest.jsonl`:
```json
{
  "id": "sample_0001",
  "modality": "t2i",
  "use_case": "recruitment",
  "aspect_ratio": "9:16",
  "width": 768,
  "height": 1344,
  "prompt_clean": "(1) Dòng chữ vị trí tuyển dụng ở phía trên dập nổi mạ vàng... (2) Khối thông tin đãi ngộ 2 dòng ở bên dưới...",
  "target_image": "targets/target_0001.png",
  "slots": [
    {
      "time_offset": 10.0,
      "glyph_path": "glyphs/glyph_0001_slot10.png",
      "font": "anton",
      "font_size_pt": 36,
      "text": "KỸ SƯ TRÍ TUỆ NHÂN TẠO",
      "lines": ["KỸ SƯ TRÍ TUỆ NHÂN TẠO"]
    },
    {
      "time_offset": 20.0,
      "glyph_path": "glyphs/glyph_0001_slot20.png",
      "font": "bevietnam",
      "font_size_pt": 32,
      "text": "Thu nhập hấp dẫn\nMôi trường sáng tạo",
      "lines": ["Thu nhập hấp dẫn", "Môi trường sáng tạo"]
    }
  ]
}
```
