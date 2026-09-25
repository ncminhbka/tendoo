# TENDOO AI — CẨM NANG THIẾT KẾ POSTER THƯƠNG MẠI (COMMERCIAL POSTER DESIGN BIBLE)

> **Tài liệu tham chiếu chuẩn mực dành cho Designer, Kỹ sư Frontend và Agentic LLM.**  
> Đúc kết từ thực tiễn thiết kế poster thương mại quốc tế và bộ mẫu thực tế: chuyển hóa các nguyên lý mỹ thuật thị giác thành **công thức có số đo, có code CSS/HTML mẫu**, và xây dựng **kiến trúc mở rộng cho LLM quản lý hàng trăm template**.

---

## 📌 MỤC LỤC
1. [PHẦN 1: CÁC NGUYÊN LÝ THỊ GIÁC BẤT BIẾN CỦA POSTER THƯƠNG MẠI](#phần-1-các-nguyên-lý-thị-giác-bất-biến-của-poster-thương-mại)
2. [PHẦN 2: TỪ ĐIỂN NĂNG LỰC THẨM MỸ HTML & CSS (WHAT CAN HTML/CSS DO?)](#phần-2-từ-điển-năng-lực-thẩm-mỹ-html--css-what-can-htmlcss-do)
3. [PHẦN 3: PHÂN TÍCH 5 CA NGHIÊN CỨU THỰC TẾ (CASE STUDIES TỪ ẢNH MẪU)](#phần-3-phân-tích-5-ca-nghiên-cứu-thực-tế-case-studies-từ-ảnh-mẫu)
4. [PHẦN 4: BẢN ĐỒ TEMPLATE THEO Ý ĐỊNH THỊ GIÁC (VISUAL INTENT TAXONOMY)](#phần-4-bản-đồ-template-theo-ý-định-thị-giác-visual-intent-taxonomy)
5. [PHẦN 5: KIẾN TRÚC AGENTIC LLM ĐỂ QUẢN LÝ BÙNG NỔ HÀNG TRĂM TEMPLATE](#phần-5-kiến-trúc-agentic-llm-để-quản-lý-bùng-nổ-hàng-trăm-template)

---

# PHẦN 1: CÁC NGUYÊN LÝ THỊ GIÁC BẤT BIẾN CỦA POSTER THƯƠNG MẠI

Poster thương mại **không phải là trang sách hay bài báo**. Người xem lướt qua một poster quảng cáo trên phố hay trên mạng xã hội chỉ trong **1.5 đến 3 giây**. Nếu mọi dòng chữ đều viết đều nhau tăm tắp cùng một kích thước và màu sắc, poster sẽ biến thành một "khối văn bản xám xịt" (wall of text) và người xem sẽ bỏ qua ngay lập tức.

### 1. Luật Ba Tầng Thị Giác (The 3-Level Visual Hierarchy & 3-Second Rule)
Một poster thương mại chuẩn quốc tế luôn chia nội dung thành đúng 3 cấp độ:
- **Cấp 1 - Điểm Neo Móc (The Hook / 70% Thị Giác)**: Con số khuyến mãi khổng lồ ("50%", "169K", "2 TRIỆU") hoặc từ khóa giật gân ("GRAND OPENING", "SKIN CARE"). Người xem phải đọc được từ khoảng cách 3-5 mét hoặc khi lướt newsfeed ở tốc độ cao.
- **Cấp 2 - Ngữ Cảnh Bổ Trợ (The Context / 20% Thị Giác)**: Dòng giải thích ngắn gọn lý do vì sao ưu đãi này có giá trị ("Combo nướng chào đông", "Trải nghiệm thực tế ảo", thời hạn diễn ra).
- **Cấp 3 - Chi Tiết Hành Động (The Details / 10% Thị Giác)**: Nút CTA ("ĐẶT BÀN NGAY"), địa chỉ, hotline, mã QR.

### 2. Tương Phản Khắc Nghiệt (Extreme Scale Contrast) & The Squint Test
- Trong văn bản thông thường, tiêu đề lớn gấp 1.5x hoặc 2x văn bản thân.
- Nhưng trong **poster thương mại**, tỷ lệ tương phản giữa điểm nhấn chính (Hero/Number) và chữ phụ phải đạt từ **4x đến 10x**!
- **Quy tắc Kiểm Tra Nheo Mắt (The Squint Test)**: Khi lùi ra xa và nheo mắt lại sao cho toàn bộ poster mờ đi: Nếu bạn vẫn nhìn thấy rõ mồn một Điểm Neo Cấp 1 và Chủ thể sản phẩm, poster đó đạt chuẩn. Nếu thấy mọi chữ nhòe nhoẹt ngang nhau, bố cục đã thất bại.

### 3. Nghệ Thuật Chữ To Nhỏ Xen Kẽ Cùng Dòng (Inline Hierarchical Rhythm)
Trong poster thực tế, người thiết kế **rất hiếm khi viết một câu với cùng một cỡ chữ**. Họ phân rã một câu thành:
- **Từ bổ trợ (nhỏ, mỏng, thanh thoát)**: "CÓ CƠ HỘI NHẬN NGAY", "CHỈ TỪ:", "GIẢM TỚI", "DAYS LEFT".
- **Từ khóa giá trị (khổng lồ, đậm, màu nhấn, 3D)**: "**2**", "**169K**", "**50%**", "**1**".
- **Đơn vị / Hậu tố (vừa, viết hoa, dạt góc)**: "TRIỆU ĐỒNG", "VNĐ", "COUNTDOWN".
Sự nhấp nhô to nhỏ ngay trên một dòng tạo ra **nhịp điệu thị giác (visual cadence)**, dẫn dắt ánh nhìn trực diện vào giá trị hấp dẫn nhất.

### 4. Quy Luật Luồng Mắt Tự Nhiên (Z-Pattern & F-Pattern Flow)
- **Bố cục Z-Pattern (cho poster có sản phẩm ở giữa - Sandwich Layout)**:
  1. Mắt bắt đầu ở **Góc trên cùng** (Logo / Tag phân loại).
  2. Quét ngang qua **Tiêu đề lớn / Hero** ở đỉnh.
  3. Rơi chéo xuống **Sản phẩm trung tâm** (Ly cà phê, đĩa thịt nướng, lọ mỹ phẩm).
  4. Quét ngang qua **Chân trang** (Hotline bên trái $\rightarrow$ Nút CTA ở giữa $\rightarrow$ Mã QR quét lấy vé bên phải).
- **Bố cục L-Frame / Diagonal**: Dành cho poster có nhiều thông số kỹ thuật hoặc quy trình nhiều bước.

### 5. Thang Cỡ Chữ Chuẩn (Modular Typographic Scale)
Không bao giờ chọn cỡ chữ ngẫu nhiên. Cỡ chữ của các cấp liền kề phải tuân theo tỷ lệ toán học:
- **1.333 (Perfect Fourth)**: Chuẩn cân bằng, áp dụng cho phần lớn poster bán lẻ, tin tức, tuyển dụng.
- **1.618 (Golden Ratio)**: Chuẩn tương phản cao, chuyên dùng cho poster Luxury, Kim loại, Mỹ phẩm, Sự kiện cao cấp.

### 6. Quy Tắc Phối Màu 60 - 30 - 10
- **60% Màu Chủ Đạo**: Nền không gian (Scene/Corridor do Diffusion sinh ra).
- **30% Màu Phụ Trợ**: Các khối card kính mờ, khung hộp chữ, dải bệ đỡ.
- **10% Màu Nhấn (Accent Glow)**: Duy nhất dành cho **Điểm Neo Cấp 1 (Hero Number/CTA Button/Icon Sparkle)**. Không bao giờ dùng màu nhấn bừa bãi ra toàn poster.

---

# PHẦN 2: TỪ ĐIỂN NĂNG LỰC THẨM MỸ HTML & CSS (WHAT CAN HTML/CSS DO?)

Rất nhiều người lầm tưởng HTML/CSS chỉ làm được các khối hộp phẳng đơn điệu. Thực tế, Chromium engine hiện đại hỗ trợ những kỹ thuật đồ họa cực kỳ mạnh mẽ, đủ sức thay thế Photoshop trong việc render chữ thương mại tự động.

### 1. Kỹ Thuật Chữ To Nhỏ Xen Kẽ Cùng Dòng (Inline Scaled Typography)
Dùng Flexbox inline căn trục đáy (`align-items: baseline`):

```html
<!-- Mẫu: CÓ CƠ HỘI NHẬN NGAY [ 2 ] TRIỆU ĐỒNG -->
<div class="inline-hero-phrase">
  <span class="phrase-prefix">CÓ CƠ HỘI NHẬN NGAY</span>
  <span class="phrase-hero-num">2</span>
  <span class="phrase-suffix">TRIỆU ĐỒNG</span>
</div>
```

```css
.inline-hero-phrase {
  display: inline-flex;
  align-items: baseline;
  justify-content: center;
  gap: 0.25em;
  font-family: var(--ui-font);
}
.phrase-prefix, .phrase-suffix {
  font-size: 0.38em;       /* Bằng ~38% cỡ của con số chính */
  font-weight: 700;
  color: #FFFFFF;
  letter-spacing: 0.5px;
}
.phrase-hero-num {
  font-size: 1em;           /* Con số chiếm 100% kích thước font của khối */
  font-weight: 900;
  color: #FFD700;
  line-height: 1;
  text-shadow: 0 4px 16px rgba(255, 215, 0, 0.6), 0 2px 4px rgba(0,0,0,0.8);
}
```

---

### 2. Kỹ Thuật Chữ Vàng Kim Loại 3D (Metallic Gold Emboss & Extrusion)
Tạo khối kim loại nổi như được đúc bằng vàng thật thông qua chuỗi bóng đổ nhiều lớp (`multi-layered text-shadow`) và gradient mặt cắt:

```css
.text-metallic-gold-3d {
  font-weight: 900;
  text-transform: uppercase;
  /* Mặt chữ gradient vàng bóng */
  background: linear-gradient(180deg, #FFFFFF 0%, #FFE680 25%, #E5A910 50%, #8A5A00 75%, #FFEFA6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  /* Viền kim loại sắc nét */
  -webkit-text-stroke: 1px rgba(255, 240, 150, 0.6);
  /* Khối 3D đùn ra phía sau bằng multi-shadow (1px -> 8px) */
  filter: drop-shadow(0 2px 0 #9A6600)
          drop-shadow(0 4px 0 #6B4400)
          drop-shadow(0 6px 0 #422800)
          drop-shadow(0 10px 15px rgba(0, 0, 0, 0.9));
}
```

---

### 3. Kỹ Thuật Chữ 3D Phồng Mọng Nước / Thạch Gel (Puffy Water Droplet / Gel 3D)
Dành cho poster Mỹ phẩm, Dưỡng ẩm, Nước giải khát mùa hè:

```css
.text-water-gel-3d {
  font-weight: 900;
  color: #E0F7FA;
  background: linear-gradient(180deg, #FFFFFF 0%, #80DEEA 40%, #00ACC1 70%, #006064 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  /* Ánh viền nước bóng loáng */
  -webkit-text-stroke: 1.5px rgba(255, 255, 255, 0.85);
  /* Đổ bóng kép: ánh sáng xuyên qua nước + bóng đáy mềm */
  filter: drop-shadow(0 3px 6px rgba(0, 172, 193, 0.7))
          drop-shadow(0 8px 20px rgba(0, 0, 0, 0.5));
}
```

---

### 4. Kỹ Thuật Chữ Retro Viền Đôi Đồ Nướng / BBQ (Chalkboard Pop Outline)
Mặt chữ màu vàng/cam ấm, viền trắng kép nổi bật trên nền bảng đen hoặc khói nướng:

```css
.text-bbq-retro-outline {
  font-weight: 900;
  color: #FFA000;
  -webkit-text-stroke: 2px #FFFFFF;
  /* Bóng đổ khối cứng (hard shadow không blur) tạo cảm giác truyện tranh/biển hiệu cổ */
  text-shadow: 4px 4px 0px #000000, 6px 6px 0px rgba(0, 0, 0, 0.4);
  letter-spacing: 1px;
}
```

---

### 5. Dải Ruy Băng Lượn Sóng & Đuôi Nheo (Ribbon Banner)
Tạo dải băng rôn mềm mại phía sau con số ưu đãi bằng `clip-path`:

```html
<div class="ribbon-banner">
  <span class="ribbon-content">CHỈ TỪ: 169K</span>
</div>
```

```css
.ribbon-banner {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(90deg, #FF8F00, #FFC107, #FF8F00);
  color: #000000;
  font-weight: 900;
  padding: 0.4em 1.8em;
  /* Cắt 2 đầu thành đuôi nheo cờ đuôi cá */
  clip-path: polygon(10px 0%, calc(100% - 10px) 0%, 100% 50%, calc(100% - 10px) 100%, 10px 100%, 0% 50%);
  box-shadow: 0 4px 15px rgba(255, 143, 0, 0.5);
  border-top: 2px solid #FFF;
  border-bottom: 2px solid #FFF;
}
```

---

### 6. Viên Nang 2 Nửa Màu (Split-Capsule / Dual-Color Badge)
Phân tách rõ ràng giữa [NHÃN VẤN ĐỀ] và [GIẢI PHÁP / GIÁ TRỊ] như poster Dược mỹ phẩm:

```html
<div class="split-capsule">
  <span class="capsule-label">PROBLEM</span>
  <span class="capsule-value">THE RIGHT FORMULA</span>
</div>
```

```css
.split-capsule {
  display: inline-flex;
  align-items: center;
  border-radius: 9999px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.25);
  box-shadow: 0 3px 10px rgba(0,0,0,0.3);
  font-size: 13px;
  font-weight: 700;
}
.capsule-label {
  background: #D32F2F;   /* Đỏ nổi bật */
  color: #FFFFFF;
  padding: 0.35em 0.85em;
  border-radius: 9999px 0 0 9999px;
}
.capsule-value {
  background: rgba(255, 255, 255, 0.85); /* Kem sáng */
  color: #1A1A1A;
  padding: 0.35em 0.95em;
}
```

---

### 7. Chữ Uốn Cong Cánh Cung (Curved / Arc Text qua SVG)
Chuẩn cho các sự kiện Vòng quay may mắn, Hội chợ, Grand Opening:

```html
<svg viewBox="0 0 500 120" class="curved-text-svg">
  <path id="curvePath" d="M 30,100 Q 250,20 470,100" fill="transparent"/>
  <text class="curved-title">
    <textPath href="#curvePath" startOffset="50%" text-anchor="middle">
      VÒNG QUAY MAY MẮN
    </textPath>
  </text>
</svg>
```

---

# PHẦN 3: PHÂN TÍCH 5 CA NGHIÊN CỨU THỰC TẾ (CASE STUDIES TỪ ẢNH MẪU)

| Ảnh Mẫu | Thể Loại | Điểm Nhấn Cốt Lõi | Kỹ Thuật Đồ Họa Cần Học Tập |
| :--- | :--- | :--- | :--- |
| **Ảnh 1: Grand Opening Countdown** | Bất động sản / Khai trương xa xỉ | **Số "1" khổng lồ** chiếm trọn tâm điểm thị giác. | - Hero Number chiếm 60% chiều cao không gian.<br>- Chữ kim loại vàng dập nổi 3D.<br>- Hạt bụi vàng lấp lánh (Gold dust / Sparks) bao quanh vầng sáng. |
| **Ảnh 2: Combo Nướng Chào Đông** | Ẩm thực nhà hàng BBQ | **Ruy băng "CHỈ TỪ: 169K"** và 3 hộp set giá. | - Chữ retro kẻ viền đôi màu vàng cam.<br>- Ribbon đuôi nheo uốn lượn.<br>- Grid 3 hộp giá tiền đồng bộ (Set 1-2, Set 3-4, Set 5-6). |
| **Ảnh 3: Super Skin Care** | Mỹ phẩm dưỡng ẩm / Serum | **Chữ "SKIN CARE" 3D mọng nước** & Giọt nước trong suốt. | - Kỹ thuật Gel 3D / Water reflection.<br>- Chữ phụ "SUPER" nghiêng thể thao góc trên.<br>- Khung nhãn góc cạnh kỹ thuật `[Professional water lock]`. |
| **Ảnh 4: Vòng Quay May Mắn** | Minigame / Trúng thưởng | **Chữ cong Arc** và **Số "2" khổng lồ** giữa dòng. | - Tiêu đề uốn lượn theo vành đai vòng tròn.<br>- Cụm từ to nhỏ xen kẽ: "CÓ CƠ HỘI NHẬN NGAY [2] TRIỆU ĐỒNG" (số 2 to gấp 3 lần).<br>- Chữ 3D viền trắng đổ bóng cam rực rỡ. |
| **Ảnh 5: Targeted Hair Treatment** | Chăm sóc cá nhân / Giáng sinh | **Thanh tuyết phủ** và **Capsule 2 nửa màu**. | - Ribbon có tuyết phủ trên mép trên (Snow cap).<br>- Split Capsule Badge phân chia Vấn đề & Giải pháp.<br>- Nút CTA "ORDER NOW" tương phản trên nền trắng viền đỏ. |

---

# PHẦN 4: BẢN ĐỒ TEMPLATE THEO Ý ĐỊNH THỊ GIÁC (VISUAL INTENT TAXONOMY)

Thay vì gọi tên template theo hình học khô khan (`sandwich_top`, `split_left`), hệ thống tương lai sẽ ánh xạ theo **Ý Định Truyền Thông (Visual Intent)**:

```
VISUAL INTENT TAXONOMY
│
├── 1. intent = "big_number_deal" (Nhấn mạnh giá sốc / Phần trăm giảm / Countdown)
│    ├── Mẫu: "CHỈ TỪ 169K", "GIẢM 50%", "1 DAY COUNTDOWN"
│    └── Đặc thù: Dành 60% dải chứa chữ cho Hero Number, chữ bổ trợ dạt xung quanh.
│
├── 2. intent = "festive_curved_banner" (Sự kiện rộn ràng / Khai trương / Minigame)
│    ├── Mẫu: Vòng quay may mắn, Hội chợ tết, Khai xuân đại lộc.
│    └── Đặc thù: Tiêu đề cong cánh cung (Arc Text), dải ruy băng đuôi nheo (Ribbon).
│
├── 3. intent = "clean_cosmetics_water" (Dược mỹ phẩm / Công nghệ sinh học / Spa)
│    ├── Mẫu: Serum căng bóng, Nước khoáng khóa ẩm, Trị liệu chuyên sâu.
│    └── Đặc thù: Hiệu ứng Gel 3D, giọt nước trong suốt, Split Capsule nhãn kỹ thuật.
│
├── 4. intent = "luxury_gold_elegance" (Bất động sản / Hội nghị VIP / Đồng hồ)
│    ├── Mẫu: Hội nghị thượng đỉnh AI, Đêm tiệc tri ân, Ra mắt căn hộ mẫu.
│    └── Đặc thù: Chữ mạ vàng Metallic 3D, Type scale tỷ lệ vàng 1.618, nền đen/xanh đêm huyền bí.
│
└── 5. intent = "food_restaurant_menu" (Ẩm thực / Combo lẩu nướng / Cafe)
     ├── Mẫu: Đại tiệc hải sản, Combo nướng chào đông, Trà sữa deal hời.
     └── Đặc thù: Bảng giá theo set, badge retro viền đôi, chữ ấm áp kích thích vị giác.
```

---

# PHẦN 5: KIẾN TRÚC AGENTIC LLM ĐỂ QUẢN LÝ BÙNG NỔ HÀNG TRĂM TEMPLATE

Khi số lượng template tăng từ 10 lên 100+, chúng ta **tuyệt đối không nhồi toàn bộ mã HTML/CSS hay mô tả chi tiết của 100 template vào Context Prompt của LLM**. Làm như vậy sẽ gây nghẽn Token, tăng chi phí và khiến LLM bị "ảo giác" (hallucination).

### Giải Pháp: Mô Hình Phân Luồng Agentic 2 Tầng (Two-Tier Routing)

```
[ User Prompt: "Poster hội nghị AI, nhấn mạnh giảm 50% vé sớm, phong cách sang trọng" ]
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: LLM INTENT PARSER (Trọng lượng nhẹ - Siêu rẻ)                  │
│ Nhiệm vụ: Không cần biết code template, chỉ trích xuất JSON Intent:    │
│ {                                                                      │
│   "visual_intent": "big_number_deal",                                  │
│   "theme_vibe": "luxury_gold",                                         │
│   "hero_number": "50%",                                                │
│   "highlight_phrase": "GIẢM 50% VÉ SỚM",                              │
│   "aspect_ratio": "1:1"                                                │
│ }                                                                      │
└────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BỘ ĐIỀU PHỐI HỆ THỐNG (Vector Catalog Matcher / Registry Query)         │
│ - Tra cứu nhanh trong CATALOG_INDEX (Metadata nhẹ, không tốn Token):  │
│   Tìm thấy Template tối ưu: `luxury_countdown_stat` hoặc              │
│   `sandwich_top_heavy` kích hoạt cờ `use_hero_number=True`.            │
└────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: TEMPLATE SPECIALIST AGENT (Nạp đúng 1 template duy nhất)       │
│ - Chỉ nạp đúng Schema và Budget của Template được chọn.               │
│ - Điền nội dung vào các slot: hero, hero_number, subhead, cta.         │
│ - Xuất ra TendooCreativePlan hoàn chỉnh.                               │
└────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: ENGINE THỰC THI (geometry.py -> renderer.py -> Playwright)     │
│ - geometry tính toán corridor mask cho Diffusion.                     │
│ - renderer tính trần/sàn cỡ chữ cho từng slot (kể cả số khổng lồ).    │
│ - Chromium chụp ảnh poster thành phẩm hoàn hảo không lỗi font/lệch lề. │
└────────────────────────────────────────────────────────────────────────┘
```

### Schema Mở Rộng Sẵn Sàng Cho Micro-Styling:
Để hỗ trợ chữ to nhỏ xen kẽ và các hiệu ứng nâng cao, schema `TendooCreativePlan` sẽ được mở rộng linh hoạt:
```json
{
  "template": "sandwich_top_heavy",
  "visual_intent": "big_number_deal",
  "hero": "ĐẠI TIỆC HẢI SẢN HOÀNG GIA",
  "hero_stat": {
    "prefix": "CHỈ TỪ",
    "number": "169K",
    "unit": "VNĐ/SET",
    "style": "ribbon_banner"
  },
  "style": {
    "font": "bevietnam",
    "text_effect": "metallic_gold",
    "theme_color": "#FFD700"
  }
}
```

---

## 🎯 KẾT LUẬN & HƯỚNG ĐI TIẾP THEO
1. File tài liệu này đóng vai trò **Bộ Quy Chuẩn Tối Thượng (Single Source of Truth)** về thẩm mỹ poster cho dự án Tendoo AI.
2. Mọi template mới xây dựng sau này sẽ tuân thủ:
   - Sử dụng chung linh kiện từ `ui_components.css` và `ui_macros.html`.
   - Hỗ trợ các kỹ thuật chữ to nhỏ nhịp điệu (Inline Contrast).
   - Bảo đảm an toàn diện tích Mask $\le 50\%$ từ `geometry.py`.
3. Khi bạn muốn bổ sung template nào (vd: Mẫu số to 3D, Mẫu dải băng ruy băng, Mẫu dược mỹ phẩm giọt nước), chúng ta chỉ cần tạo thư mục template và kế thừa 100% các thành phần nền tảng đã xây dựng.
