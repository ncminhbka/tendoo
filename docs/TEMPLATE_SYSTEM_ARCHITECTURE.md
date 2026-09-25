# KIẾN TRÚC HỆ THỐNG TEMPLATE ĐA SIZE & CHIẾN LƯỢC AGENTIC TENDOO AI
## (MULTI-SIZE TEMPLATE ARCHITECTURE & AGENTIC DISPATCH STRATEGY)

> **Tài liệu chiến lược kỹ thuật & thiết kế song hành cùng [docs/DESIGN_PRINCIPLES.md](file:///d:/Viettel%20Telecom/Tendoo%20AI/docs/DESIGN_PRINCIPLES.md)**  
> Giải quyết triệt để nghịch lý: *"Làm sao để poster vừa ĐẸP NHƯ MAY ĐO, vừa KHÔNG BAO GIỜ BỊ VỠ CHỮ?"*

---

## 📌 MỤC LỤC
1. [BƯỚC NGOẶT TRIẾT LÝ: TẠM BIỆT ÁO FREESIZE](#-1-bước-ngoặt-triết-lý-tạm-biệt-áo-freesize)
2. [TẠI SAO BÂY GIỜ CHÚNG TA TỰ TIN LÀM NHIỀU TEMPLATE?](#-2-tại-sao-bây-giờ-chúng-ta-tự-tin-làm-nhiều-template)
3. [MA TRẬN 6 DANH MỤC × 4 CỠ CHỮ (THE 6x4 SYSTEM GRID)](#-3-ma-trận-6-danh-mục--4-cỡ-chữ-the-6x4-system-grid)
4. [BẢN HỢP ĐỒNG NỘI DUNG (CONTENT CONTRACT) CHO TỪNG SIZE](#-4-bản-hợp-đồng-nội-dung-content-contract-cho-từng-size)
5. [MÔ HÌNH PHÂN LUỒNG AGENTIC LLM (SMART SIZING & ROUTING)](#-5-mô-hình-phân-luồng-agentic-llm-smart-sizing--routing)
6. [5 LOẠI MASK BỐ CỤC CHUẨN CỦA DESIGNER (THE 5 CANONICAL MASKS)](#-6-5-loại-mask-bố-cục-chuẩn-của-designer-the-5-canonical-masks)
7. [ĐỘT PHÁ: CHẾ ĐỘ KHÔNG CẦN MASK CHO POSTER THUẦN CHỮ (MASKLESS MODE)](#-7-đột-phá-chế-độ-không-cần-mask-cho-poster-thuần-chữ-maskless-mode)
8. [PHANH AN TOÀN CUỐI CÙNG (LAST-MILE GUARDRAIL)](#-8-phanh-an-toàn-cuối-cùng-last-mile-guardrail)
9. [KẾ HOẠCH TÍCH HỢP CÔNG CỤ & THƯ VIỆN ĐỒ HỌA BỔ TRỢ (GRAPHICS TOOLKIT ROADMAP)](#-9-kế-hoạch-tích-hợp-công-cụ--thư-viện-đồ-họa-bổ-trợ-graphics-toolkit-roadmap)

---

## 📌 1. BƯỚC NGOẶT TRIẾT LÝ: TẠM BIỆT ÁO FREESIZE

### 1.1. Bản chất vì sao "Áo Freesize" giết chết thẩm mỹ?
Trước đây, 14 template được thiết kế theo tư duy **"Một template cố gắng co giãn chống chịu cho mọi thể loại" (One-Size-Fits-All)**:
- Người dùng nhập 1 từ cũng phải vừa, nhập 40 từ cũng phải vừa.
- Có 0 pill cũng được, có 6 pill cũng phải vừa.
- **Hệ quả tất yếu**: Designer buộc phải xếp các khối thành các dải phẳng, thẳng hàng, đều tăm tắp (`flex-direction: column; overflow: hidden; align-items: center;`). Không thể cho chữ nghiêng ngả đè lên nhau, không thể phóng to con số khổng lồ, không thể uốn lượn ruy băng. Kết quả là poster an toàn nhưng **đơn điệu, nhạt nhòa, thiếu cá tính nghệ thuật**.

### 1.2. Giải pháp: Chuẩn bị sẵn cỡ áo (S, M, L, XL) cho từng thể loại nội dung
Thay vì bắt 1 template phải gánh mọi khối lượng chữ:
- **Template Size S (Minimal Punch)**: Chuyên trị **ít text** (1 tiêu đề ngắn 2–3 từ + CTA). Chữ được bung to cực đại, hoành tráng, chiếm trọn tâm điểm.
- **Template Size M (Hero Stat / Focal Focus)**: Chuyên trị **nội dung có điểm nhấn số** (Tiêu đề + Con số to 3D khổng lồ "50%", "169K" + 1–2 tag ngắn).
- **Template Size L (Standard Balanced)**: Chuyên trị **độ dài tiêu chuẩn** (Tiêu đề + Slogan + 3 tag ưu đãi + Store Info + CTA + QR).
- **Template Size XL (Editorial / Text-Heavy)**: Chuyên trị **nhiều text** (Đoạn văn dài, bài thơ 4 câu, nhiều chi nhánh, chính sách dài). Bố cục chia 2 cột kiểu tạp chí/bảng tin, không gian thở rộng rãi, không bị xén chữ.

---

## 📌 2. TẠI SAO BÂY GIỜ CHÚNG TA TỰ TIN LÀM NHIỀU TEMPLATE?

Trước đây, ý tưởng làm 40–60 template là "bất khả thi" vì mỗi template viết tay 400 dòng CSS/HTML trùng lặp (60 template $\approx$ 24.000 dòng code, không ai bảo trì nổi).

Nhưng với **Kiến Trúc Phân Tầng V3 Đã Hoàn Thiện**:
1. **Linh kiện dùng chung (Shared Components)**: [components/ui_components.css](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/tendoo_v3/templates/components/ui_components.css) đã chuẩn hóa 100% style của Badge, Hero, Subhead, Pills, Store Info, CTA, QR.
2. **Thư viện Macro (Shared Macros)**: [components/ui_macros.html](file:///d:/Viettel%20Telecom/Tendoo%20AI/src/tendoo_v3/templates/components/ui_macros.html) đã chuẩn hóa 100% logic HTML Zero-Trace.
3. **Mỗi template mới giờ đây chỉ là một khung Layout dài đúng ~60–80 dòng code**:
   - Khi muốn tạo một template mới, chỉ cần khai báo vị trí các zone trong `geometry.py` và gọi các macro vào `template.html`.
   - **Chi phí tạo 1 template mới chỉ mất 15–20 phút** và **hoàn toàn không làm tăng gánh nặng bảo trì**!

---

## 📌 3. MA TRẬN 6 DANH MỤC × 4 CỠ CHỮ (THE 6x4 SYSTEM GRID)

Hệ thống được cấu trúc thành **24 Template Nòng Cốt** bao quát 100% nhu cầu thực tế từ [yeu_cau.txt](file:///d:/Viettel%20Telecom/Tendoo%20AI/yeu_cau.txt):

```
┌──────────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐
│ CỠ S (Ít Text / Punch)   │ CỠ M (Số To / Focal)     │ CỠ L (Tiêu Chuẩn Đầy Đủ) │ CỠ XL (Nhiều Text/Bảng)  │
├──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 🛍️ 1. KHUYẾN MẠI / SALE  │                          │                          │                          │
│ promo_minimal_bold       │ promo_hero_number_3d     │ promo_sandwich_standard  │ promo_combo_price_grid   │
│ (Tiêu đề khổng lồ + CTA) │ (Số 50%, 169K to giữa)   │ (Đỉnh kẹp đáy đầy đủ)    │ (Bảng combo nhiều món/giá)│
├──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 💎 2. GIỚI THIỆU SẢN PHẨM│                          │                          │                          │
│ product_zen_corner       │ product_spec_capsule     │ product_standard_card    │ product_spec_sheet_table │
│ (Dạt góc, 80% khoe ảnh)  │ (Split capsule kỹ thuật) │ (Tên sp + mô tả + giá)   │ (Bảng thông số chi tiết) │
├──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 🎊 3. BANNER KHAI TRƯƠNG │                          │                          │                          │
│ opening_grand_arch       │ opening_countdown_days   │ opening_ceremony_ribbon  │ opening_event_agenda     │
│ (Chữ cong uốn lượn)      │ (Đếm ngược 1 DAY/3 DAYS) │ (Cắt băng khánh thành)   │ (Lịch trình & nhiều quà) │
├──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ ⭐ 4. FEEDBACK KHÁCH HÀNG│                          │                          │                          │
│ feedback_5star_badge     │ feedback_quote_hero      │ feedback_centered_glass  │ feedback_multi_review    │
│ (5 sao lớn + 1 câu khen) │ (Ngoặc kép trích dẫn to) │ (Card kính mờ review)    │ (Nhiều nhận xét cùng lúc)│
├──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 👔 5. TUYỂN DỤNG         │                          │                          │                          │
│ recruit_hiring_punch     │ recruit_salary_focus     │ recruit_two_column_board │ recruit_full_policy      │
│ ("WE ARE HIRING" áp đảo) │ (Mức lương to nổi bật)   │ (Bảng Quyền lợi vs Yêu cầu│ (Mô tả JD chi tiết)     │
├──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 🗺️ 6. QUY TRÌNH HƯỚNG DẪN│                          │                          │                          │
│ step_quick_two_steps     │ step_horizontal_3cols    │ step_roadmap_ziczac      │ step_detailed_sop        │
│ (2 bước siêu gọn)        │ (3 bước nối mũi tên)     │ (Lộ trình ziczac uốn lượn│ (Nhiều bước có mô tả sâu)│
└──────────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 📌 4. BẢN HỢP ĐỒNG NỘI DUNG (CONTENT CONTRACT) CHO TỪNG SIZE

Để template không bao giờ bị vỡ, mỗi cỡ template đều có một **Hợp Đồng Dung Lượng (Capacity Limits)** rõ ràng:

| Cỡ Template | Giới hạn Tiêu Đề (Hero) | Giới hạn Mô tả (Subhead) | Thành phần bổ trợ | Trường hợp kích hoạt lý tưởng |
| :--- | :--- | :--- | :--- | :--- |
| **Size S** | $\le 4$ từ (vd: "SALE SẬP SÀN") | Không có hoặc $\le 6$ từ | 1 CTA Button duy nhất | Form chỉ điền tiêu đề + ảnh, không điền chi tiết. |
| **Size M** | 1 Số ngắn $\le 5$ ký tự (`50%`, `169K`) + $\le 3$ từ bổ trợ | $\le 10$ từ | 1–2 Pill từ khóa | Form có điền mức giảm giá sốc, ngày countdown. |
| **Size L** | $\le 8$ từ (chia 1–2 dòng đẹp) | $\le 16$ từ | 3 Pills + Store Info + CTA + QR | Form điền đầy đủ các trường cơ bản. |
| **Size XL** | $\le 12$ từ | Đoạn văn hoặc danh sách $\ge 4$ dòng | Bảng 2 cột, lưới giá tiền | Form có mô tả công việc dài, menu nhiều combo giá. |

---

## 📌 5. MÔ HÌNH PHÂN LUỒNG AGENTIC LLM (SMART SIZING & ROUTING)

LLM sẽ không bao giờ nhồi nhét mã CSS hay cố đoán cỡ chữ pixel. Toàn bộ quy trình diễn ra tự động qua 3 bước:

```
[ BƯỚC 1: ĐO ĐẠC NỘI DUNG (CONTENT PROFILING) ]
  Người dùng gửi Form + Prompt tự do.
  LLM Parser đọc dữ liệu và đo đạc:
  - Danh mục: "promotional_sale"
  - Phân tích văn bản: Có số ưu đãi rõ ràng không? -> Có: "giảm 50%"
  - Tổng số từ: Ngắn (< 15 từ)
         │
         ▼
[ BƯỚC 2: CHỌN CỠ ÁO & TEMPLATE (SIZE MATCHING) ]
  - Khớp với Danh mục 1 + Cỡ M (Hero Number Focus).
  - Chọn chính xác Template: `promo_hero_number_3d`.
  - Tự tin 100% không bị vỡ vì template này sinh ra là để đón nhận con số ngắn!
         │
         ▼
[ BƯỚC 3: BÓC TÁCH SLOT NỘI DUNG (SLOT FITTING) ]
  LLM điền vào JSON Plan sạch sẽ:
  {
    "template": "promo_hero_number_3d",
    "hero_prefix": "GIẢM TỚI",
    "hero_number": "50%",
    "hero_suffix": "TOÀN BỘ MENU",
    "cta": "NHẬN MÃ NGAY"
  }
         │
         ▼
[ BƯỚC 4: RENDERER & CHROMIUM THỰC THI ]
  - geometry.py cấp corridor mask cho Diffusion.
  - template gọi macro render_hero_stat("GIẢM TỚI", "50%", "TOÀN BỘ MENU").
  - Chữ "50%" bung to 3D rực rỡ, chữ phụ nằm nép tinh tế, đẹp hoàn hảo như designer vẽ tay!
```

---

## 📌 6. 5 LOẠI MASK BỐ CỤC CHUẨN CỦA DESIGNER (THE 5 CANONICAL MASKS)

Thay vì tạo ra hàng chục kiểu mask ngẫu hứng, các designer chuyên nghiệp trong thực tế chỉ sử dụng **5 Họ Mask chuẩn mực** sau:

1. **Họ Sandwich (Đỉnh & Đáy)**:
   - Dành cho: Đồ ăn, thức uống, sản phẩm trung tâm (ly cà phê bốc khói, đĩa hải sản).
   - Vùng mask: Dải đỉnh (Top 25–31%) + Dải đáy (Bottom 15–18%). Mở toang 55% trung tâm cho sản phẩm thở.
2. **Họ Split Side (Cột sườn Trái / Phải)**:
   - Dành cho: Xe hơi, người mẫu thời trang đứng lệch bên, bảng thông số kỹ thuật.
   - Vùng mask: Cột dọc chiếm 38–45% chiều rộng bên trái hoặc bên phải.
3. **Họ Centered / Corner Pod (Khối Card Kính Mờ)**:
   - Dành cho: Feedback khách hàng, Mỹ phẩm cao cấp, Thiệp mời sự kiện.
   - Vùng mask: Khối chữ nhật bo tròn góc nằm ở chính giữa (cao 45–60%) hoặc dạt góc dưới.
4. **Họ Bottom Platform (Bệ Đỡ Đáy Sâu)**:
   - Dành cho: Poster phong cảnh, bất động sản, dự án mở rộng với khoảng trời mênh mông.
   - Vùng mask: Bệ đáy sâu chiếm 35–42% chân poster.
5. **Họ Diagonal / Slanted (Cắt Chéo Năng Động)**:
   - Dành cho: Thể thao, giày sneaker, sự kiện âm nhạc, công nghệ bứt phá.
   - Vùng mask: Đường cắt chéo 15–25 độ chia đôi khung hình.

---

## 📌 7. ĐỘT PHÁ: CHẾ ĐỘ KHÔNG CẦN MASK CHO POSTER THUẦN CHỮ (MASKLESS MODE)

> **Phát kiến mang tính bước ngoặt**: Không phải poster nào cũng có sản phẩm vật lý cụ thể!  
> Rất nhiều poster thương mại lấy **CHỮ LÀM NHÂN VẬT CHÍNH (Typography-First Poster)**.

### 7.1. Khi nào áp dụng Maskless Mode?
- Poster Đại Hạ Giá: *"SALE SẬP SÀN 50% - DUY NHẤT HÔM NAY"*.
- Poster Tuyển Dụng Lớn: *"WE ARE HIRING - GIA NHẬP ĐỘI NGŨ CÔNG NGHỆ"*.
- Poster Hội Nghị / Workshop: Thông báo diễn giả, lịch trình hội thảo.
- Poster Chúc Mừng Năm Mới / Giáng Sinh / Lời cảm ơn tri ân khách hàng.

### 7.2. Lợi ích vượt trội của Maskless Mode:
1. **Tốc độ sinh nền Diffusion tăng gấp đôi (2x Speedup)**:
   - Do không có sản phẩm cần bảo vệ, hệ thống **không cần chạy Velocity Blending 2 luồng prompt** (`v_scene` và `v_corridor`).
   - Diffusion chỉ chạy **1 luồng prompt duy nhất** tạo ra nền texture nghệ thuật (Atmospheric Background) đồng nhất (ví dụ: nền hạt bụi vàng kim lấp lánh, nền gradient cyber neon, nền khói lượn sóng).
2. **Tự do thẩm mỹ tuyệt đối**:
   - Chữ và các khối thẻ HTML/CSS có thể dàn trải, lơ lửng, bay bổng khắp toàn bộ khung poster theo phong cách Typography Poster quốc tế mà không bị gò bó bởi bất kỳ ranh giới mask nào!
   - Không lo viền chữ chạm sát mép mask hay bị cắt gọt.

---

## 📌 8. PHANH AN TOÀN CUỐI CÙNG (LAST-MILE GUARDRAIL)

Dù LLM đã phân loại cỡ áo, bản thân mỗi template vẫn được trang bị **3 lớp bảo vệ tự động**:
1. **Lớp 1: CSS `align-items: baseline` và `flex-wrap: wrap`**: Giúp cụm chữ to nhỏ tự cân chỉnh theo trục đáy, không bị lệch hàng.
2. **Lớp 2: Thuật toán Binary Search Autofit trong Chromium**: Nếu người dùng nhập chữ hơi dài hơn chuẩn Size 1 chút (ví dụ `102.000đ` thay vì `102K`), Autofit JS sẽ tự động co nhẹ 5% – 10% để vừa khít vùng đệm mà mắt thường không nhận ra sự suy giảm.
3. **Lớp 3: Fallback Tràn Dòng**: Nếu nội dung vượt quá giới hạn cực đoan, chữ tự động ngắt dòng thông minh (`word-break: keep-all`) thay vì bị xén ngang mép container.

---

## 📌 9. KẾ HOẠCH TÍCH HỢP CÔNG CỤ & THƯ VIỆN ĐỒ HỌA BỔ TRỢ (GRAPHICS TOOLKIT ROADMAP)

Hệ thống render của Tendoo V3 chạy trên **Chromium headless thật qua Playwright**. Đây là lợi thế cạnh tranh áp đảo so với các công cụ tĩnh (như Vercel Satori hay PIL/Cairo): **Chromium hỗ trợ 100% sức mạnh đồ họa web hiện đại (CSS 3D, SVG Filters, Canvas 2D, Web Fonts, JS DOM Calculation)**.

Để nâng tầm poster từ "khung bố cục cứng nhắc" lên chuẩn "tác phẩm thiết kế thương mại", hệ thống sẽ tích hợp có chọn lọc các thư viện và công cụ siêu nhẹ sau:

### 9.1. Bảng phân công công cụ & thư viện mục tiêu

| Công cụ / Thư viện | Kích thước & Loại | Vị trí tích hợp trong Repo | Công dụng cụ thể cho Poster | Giai đoạn triển khai |
| :--- | :--- | :--- | :--- | :--- |
| **`Open Props` Tokens** | 0 KB (CSS Tokens) | `components/ui_components.css` | **Bóng đổ đa tầng siêu thực (Smooth Layered Shadows)**: Cung cấp các biến `--shadow-3`, `--shadow-4`, `--shadow-5` cho thẻ kính mờ, badge, CTA nổi bật như thiết kế Apple/Figma. | **Giai đoạn 1 (Làm ngay)** |
| **SVG Filters Engine** | 0 KB (Native SVG) | `components/svg_filters.html` | **Chất liệu 3D Kim loại & Gel nước**: Dùng thẻ `<filter>` ẩn (`feSpecularLighting`, `feDistantLight`, `feGaussianBlur`) tạo hiệu ứng chữ đúc vàng ánh kim 3D và bóng chữ nổi khối thực thụ. | **Giai đoạn 1 (Làm ngay)** |
| **`CircleType.js`** | ~4 KB (Local JS) | `components/circletype.min.js` | **Uốn cong chữ cánh cung (Arc Text)**: Dành riêng cho tiêu đề uốn lượn của poster Khai trương (Grand Opening), Minigame / Vòng quay may mắn, và Con dấu retro tròn. | **Giai đoạn 2 (Tuần 2)** |
| **`Canvas-Confetti`** | ~8 KB (Local JS) | `components/confetti_overlay.js` | **Bụi vàng & Hạt lấp lánh (Sparkles & Gold Dust)**: Tự động rải hạt confetti/bụi sao trên một lớp canvas trong suốt nằm giữa nền diffusion và chữ, tạo không khí lễ hội/khuyến mại hoành tráng. | **Giai đoạn 2 (Tuần 2)** |
| **`Chroma.js` / `Colord`** | ~5 KB (Local JS) | `renderer.py` / headless JS | **Tự động đo tương phản WCAG AA/AAA**: Đo độ sáng (luminance) vùng nền sau khi Diffusion sinh ảnh, tự động đảo màu chữ (trắng $\leftrightarrow$ đen/vàng) nếu nền quá sáng hoặc quá tối ngoài dự tính. | **Giai đoạn 3 (Tuần 3)** |

### 9.2. Nguyên tắc triển khai thư viện (Zero-Bloat Rule)
1. **100% Offline & Local**: Tuyệt đối không gọi CDN từ internet bên ngoài. Tất cả file JS/CSS đều được lưu local trong thư mục `src/tendoo_v3/templates/components/` để đảm bảo chạy hoàn hảo trên server nội bộ (offline network).
2. **On-Demand Loading**: Thư viện nào cần cho template nào thì template đó mới kích hoạt; template tiêu chuẩn không load thừa JS.
3. **Zero-Trace Render**: Toàn bộ script chỉ chạy một lần duy nhất trong quá trình Playwright nạp trang và chụp ảnh PNG, sau đó trình duyệt đóng lại, không để lại bất kỳ rác bộ nhớ nào.

---

## 🎯 KẾT LUẬN

1. **Hệ thống Template Đa Size (S / M / L / XL)** kết hợp **5 Họ Mask Chuẩn** và **Chế độ Maskless** tạo nên một ma trận hoàn chỉnh, vững chắc.
2. **LLM làm đúng chuyên môn**: Làm người thợ may đo đạc chữ và chọn đúng cỡ áo.
3. **Template & Engine làm đúng chuyên môn**: Đảm bảo an toàn toán học, thẩm mỹ đỉnh cao và tốc độ tối ưu.
