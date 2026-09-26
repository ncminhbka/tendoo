# KẾ HOẠCH TỔNG THỂ TENDOO V3 — ĐỊNH HƯỚNG KIẾN TRÚC & THI CÔNG

> **Bản chốt 25/09/2026.** Đây là tài liệu định hướng của repo `tendoo-v3`.
> Mọi quyết định trong tài liệu này đều gắn với một phép đo thật
> chạy trên Chromium, không có quyết định nào dựa trên suy luận thuần tuý. Chỗ nào chưa đo,
> tài liệu ghi rõ là **chưa đo**.

---

## 0. TÀI LIỆU NÀY THAY THẾ CÁI GÌ

| Tài liệu | Trạng thái |
| :--- | :--- |
| `docs/TEMPLATE_SYSTEM_ARCHITECTURE.md` §1, §2, §3, §4, §5, §8, §9 | **BỊ THAY THẾ** bởi tài liệu này |
| `docs/TEMPLATE_SYSTEM_ARCHITECTURE.md` §6 (5 họ mask), §7 (maskless) | **GIỮ**, có bổ sung ràng buộc ở §5 dưới đây |
| `docs/DESIGN_PRINCIPLES.md` §1 (nguyên lý thị giác), §2 (từ điển CSS), §3 (case study) | **GIỮ NGUYÊN** — đây là phần vững nhất của cả 2 tài liệu |
| `docs/DESIGN_PRINCIPLES.md` §4 (taxonomy), §5 (agentic 2 tầng) | **BỊ THAY THẾ** bởi §2 và §3 dưới đây |
| `ke_hoach_cong_viec.txt` (repo cũ) | Nhật ký thi công + số đo chi tiết, **ở lại repo cũ đã archive** |

Lý do thay thế, tóm tắt: hai tài liệu cũ đề xuất **2 trục phân loại mâu thuẫn nhau**
(6 ngành × 4 cỡ vs 5 visual_intent), quy công cho ma trận cỡ áo cả việc "không vỡ chữ" lẫn
việc "đẹp" (thực ra là 2 trục độc lập), và im lặng hoàn toàn về số phận 14 template đang chạy.

---

## 1. CĂN CỨ ĐO ĐẠC (EVIDENCE BASE)

Toàn bộ kế hoạch này dựng trên 3 phép đo đã chạy ngày 25/09/2026:

| Phép đo | Phạm vi | Kết quả then chốt |
| :--- | :--- | :--- |
| `probe_type_contrast` | 12 case, `sandwich_top_heavy` | Tương phản hero:subhead thật **1.62x TB** (chuẩn đòi 4–10x). **0/12 tràn.** |
| `probe_hero_markup_ab` | 9 cặp A/B | Markup ngữ nghĩa → **4.10x**, tương phản trong dòng **2.63x**, vẫn 0 tràn, 323/323 test xanh |
| `probe_capacity_map` | **212 case thật**, 13 template | Chỉ **11/212 (5%)** đạt 4x. 9/13 template bị **subhead** ghim trần. 6 case có tín hiệu tràn mà test suite không bắt. |

**Ba kết luận rút ra, chi phối toàn bộ tài liệu:**

1. **Hệ thống hiện tại an toàn tuyệt đối nhưng phẳng.** 0% vỡ chữ, ~40% điểm thẩm mỹ.
   Đúng "áo oversize ai cũng mặc được".
2. **Nguyên nhân phẳng nằm ở CHÍNH SÁCH NGÂN SÁCH dùng chung, không nằm ở layout.**
   Engine chống tràn bằng cách **hạ trần hero**, tức hy sinh đúng Điểm Neo Cấp 1. Sửa ở
   1 chỗ (`compute_sandwich_top_budget`) đã đưa template đó lên đầu bảng.
3. **Nghịch lý tailored-vs-oversize là dichotomy giả.** Case `short/1_1` đạt 4.6x ngay từ
   baseline — template *vốn may đo*, nó chỉ sụp khi bị nhồi. Có **hợp đồng dung lượng
   cưỡng chế ở tầng routing** thì may đo mà vẫn không bao giờ vỡ.

---

## 2. CÂU HỎI 1 — LLM NHẬN GÌ, XUẤT GÌ

### 2.1. Nguyên tắc phân công: ba việc, ba chủ thể

| Chủ thể | Việc | Vì sao giao cho nó |
| :--- | :--- | :--- |
| **LLM** | Ngữ nghĩa: hiểu brief, chọn ý đồ thị giác, **cắt vai trò chữ**, chọn font/hiệu ứng, viết prompt cảnh | Đây là phán đoán ngôn ngữ — việc duy nhất LLM làm tốt hơn code |
| **Python** | Đo đạc: đếm ký tự, tra hợp đồng dung lượng, chọn cỡ, chốt template, reroute | Tất định, miễn phí, không bao giờ ảo giác |
| **Chromium** | Hình học: autofit binary-search, đo DOM thật, phát hiện tràn | Là engine render thật — mọi ước lượng ngoài nó đều lệch |

**Bốn điều LLM TUYỆT ĐỐI KHÔNG LÀM** (mỗi điều đều có bằng chứng thất bại trong lịch sử dự án):

1. **Không đếm chữ / không chọn cỡ áo.** Đếm ký tự là việc Python làm đúng 100%, LLM làm sai
   và tốn tiền.
2. **Không chỉ định toạ độ.** Đã chốt từ trước, giữ nguyên.
3. **Không biên tập nội dung bắt buộc.** Field bắt buộc render nguyên văn từ `req`.
4. **Không tự sinh chữ trong ảnh diffusion.** Chữ 100% là overlay HTML/CSS.

### 2.2. INPUT — LLM nhận gì

```
┌─ TẦNG 1: INTENT PARSER (rẻ, nhẹ) ─────────────────────────────┐
│ NHẬN:                                                          │
│  1. Brief thô của người dùng (văn bản tự do)                  │
│  2. Các field form bắt buộc đã điền (nếu có)                  │
│  3. Danh mục INTENT (6–8 mục, mỗi mục 1 dòng mô tả)           │
│  4. Danh mục font (19 font, gom theo archetype)               │
│  5. Danh mục hiệu ứng chữ (10 mục)                            │
│  6. Tỉ lệ khung hình đích (1:1 / 9:16 / 16:9 / 4:5)           │
│                                                                │
│ KHÔNG NHẬN (cố ý):                                             │
│  ✗ Mã HTML/CSS của bất kỳ template nào                        │
│  ✗ Bảng capacity / budget / ngân sách cỡ chữ                  │
│  ✗ Toạ độ zone, số đo pixel                                   │
│  ✗ Prose mô tả dài của cả 14 template (hiện `catalog.py` đang  │
│    nhồi ~200 từ/template vào prompt — sẽ cắt xuống 1 dòng)     │
└────────────────────────────────────────────────────────────────┘
```

**Thay đổi so với hiện tại:** `build_llm_catalog_prompt()` đang đẩy toàn bộ hint dài của
14 template vào system prompt. Phần lớn nội dung hint đó là để LLM **phân biệt sức chứa**
(*"chứa được nhiều nội dung hơn corner_pod vì chiếm full chiều cao"*) — việc mà Python sắp
làm bằng số đo. Cắt được phần này giúp prompt ngắn, rẻ, ít ảo giác hơn.

### 2.3. OUTPUT — LLM xuất gì

```jsonc
{
  // — Ý ĐỒ (LLM quyết) —
  "visual_intent": "big_number_deal",      // 1 trong 6–8 intent, xem §3.3
  "category": "promotional_sale",          // chỉ để chọn style pack, KHÔNG chọn layout
  "layout_hint": "text_top",               // thô: text_top|text_bottom|text_left|
                                           // text_right|corner|center|null
                                           // CHỈ điền khi brief nói rõ; null = để Python chọn
  "template_suggestion": "sandwich_top_heavy",  // GỢI Ý, Python có quyền phủ quyết

  // — NỘI DUNG (LLM cắt vai trò, KHÔNG biên tập) —
  "hero": "GIẢM TỚI 25% TOÀN BỘ MENU",
  "hero_parts": [                          // ★ TẦNG MARKUP — đóng góp lớn nhất về thẩm mỹ
    { "t": "GIẢM TỚI",     "role": "prefix" },
    { "t": "25%",          "role": "stat", "emphasis": "accent" },
    { "t": "TOÀN BỘ MENU", "role": "suffix" }
  ],
  "subhead": "Áp dụng toàn hệ thống tới hết chủ nhật",
  "badge": "ƯU ĐÃI CÓ HẠN",
  "extra_texts": ["Miễn phí giao hàng", "Tặng kèm topping"],
  "cta": "ĐẶT NGAY",
  "store_info": "Tendoo | Hotline: 1900 8888",
  "qr_code": "https://tendoo.ai/promo",

  // — PHONG CÁCH (LLM chọn trong danh mục đóng) —
  "style": {
    "font": "bevietnam",
    "theme_color": "#FF3B30",
    "text_effect": "shadow",
    "background_tone": "dark_luxury"
  },

  // — PROMPT DIFFUSION (chỉ cho ảnh nền, KHÔNG chứa chữ cần vẽ) —
  "scene_prompt": "...",
  "corridor_prompt": "..."
}
```

**Điểm mới quan trọng nhất là `hero_parts`.** Đây là thứ biến một dòng chữ phẳng thành
cụm có điểm neo — đúng cơ chế mà poster thương mại thật dùng ("25%" to, "GIẢM TỚI" nhỏ,
cùng một dòng, căn trục đáy). Đo được: **2.63x tương phản, tốn 0 pixel chiều cao.**

**Vai trò `stat` không chỉ dành cho con số.** Bất kỳ cụm nào là cái móc đều gán `stat`:
`"GRAND OPENING"`, `"TUYỂN DỤNG"`, `"MIỄN PHÍ"`.

### 2.4. LLM ĐỀ XUẤT — PYTHON PHỦ QUYẾT

Đây là thay đổi kiến trúc then chốt, và là điểm thực thi mà "hợp đồng dung lượng" hiện đang
thiếu hoàn toàn:

```
LLM trả template_suggestion
        │
        ▼
Python đếm ký tự thật từng slot
        │
        ▼
Tra capacity ĐO ĐƯỢC của template đó ở đúng tỉ lệ khung hình
        │
        ├── vừa  → dùng template LLM đề xuất
        └── vượt → REROUTE sang template cùng intent, cùng layout_hint,
                   nhưng band sức chứa lớn hơn. Ghi log lý do reroute.
                   Không có template nào chứa nổi → đó là LỖ HỔNG THẬT
                   trên bản đồ phủ sóng (xem §3.4)
```

Cách này giữ được thế mạnh ngữ nghĩa của LLM (nó hiểu "poster tuyển dụng" nên dùng bảng 2 cột),
đồng thời vô hiệu hoá điểm yếu của nó (không biết 340 ký tự có nhét vừa dải đỉnh 280px không).
Và nó triển khai được **tăng dần**: hôm nay LLM vẫn chọn template như cũ, Python chỉ thêm
quyền phủ quyết — không phải viết lại luồng.

### 2.5. BỐN CỔNG KIỂM SOÁT (VALIDATION GATES)

Mọi output của LLM đi qua 4 cổng trước khi ra ảnh. Cổng nào hỏng cũng **fail-safe**, không
bao giờ làm sai chữ:

| # | Cổng | Kiểm gì | Hỏng thì làm gì | Trạng thái |
| :-- | :--- | :--- | :--- | :--- |
| 1 | **Nguyên văn** | Nối `hero_parts` phải tái tạo đúng `hero` | Vứt markup, render phẳng | ✅ **ĐÃ LÀM** |
| 2 | **Thành viên** | `font`/`text_effect`/`template`/`intent` phải thuộc danh mục đóng | Rơi về mặc định hợp lệ | ✅ **0B-2 (26/09)** `validators.py::check_plan` — kiểm template/font/text_effect/background_tone + field **không có chỗ hiển thị** (sẽ mất) + field `required` bị thiếu. Chỉ **phát hiện + log**; việc rơi về mặc định giữ nguyên ở nơi dùng (cố ý: chuẩn hoá tông nền lạ thành `dark_luxury` sẽ làm `compute_type_scale_ratio` bắt chữ "luxury" và đổi thang cỡ). `intent` chưa có trong schema → chưa kiểm |
| 3 | **Dung lượng** | Ký tự thật vs capacity đo được | Reroute sang band lớn hơn | ❌ **CHƯA CÓ** |
| 4 | **Tràn** | Đo DOM sau render: có phần tử nào kẹt ở sàn mà vẫn tràn không | Log + reroute + đánh dấu case | ✅ **0C (26/09)** — `window.__tendooOverflow` (styles.py Bước 4) phân loại `clipped`/`overlap` (mất chữ thật) vs `spill` (tràn vào chỗ trống); `render_plan_to_poster(overflow_report=…)` log WARNING khi mất chữ; demo_server trả `overflow` + `text_lost` mỗi poster. **Chưa reroute** (cần capacity GĐ 1). Chưa kiểm chữ đè lên hình (QR/icon) |

Cổng 1 đã chứng minh cần thiết: markup sai một ký tự là chữ sai chính tả trên ảnh giao khách.
Cổng 4 là **hồi quy so với v1** — `src/tendoo/engine/templates/master.html:677` (repo cũ) từng có bộ
phát hiện tràn; v3 vẫn *tính* verdict tràn trong binary search (`styles.py:427`) nhưng **vứt
kết quả đi**. Hệ quả đo được: 6/212 case đang tràn thật mà 323/323 test vẫn xanh.

---

## 3. CÂU HỎI 2 — TEMPLATE CHIA THẾ NÀO

### 3.1. Ba trục độc lập, không trộn lẫn

Sai lầm gốc của tài liệu cũ là gộp ngành hàng, cỡ chữ và bố cục vào một ma trận phẳng 6×4.
Thực tế có **ba trục hoàn toàn độc lập**:

| Trục | Là gì | Quyết định điều gì | Ai chọn |
| :--- | :--- | :--- | :--- |
| **1. HỌ MASK** | Hình học vùng chữ trên canvas | Zone, corridor mask cho diffusion | Python (từ template) |
| **2. SỨC CHỨA** | Bao nhiêu ký tự vẫn giữ được tương phản | Template nào đủ sức chứa brief này | **Python, bằng SỐ ĐO** |
| **3. INTENT** | Ý đồ truyền thông | Ngưỡng tương phản, style pack, hiệu ứng | LLM |

> **NGÀNH HÀNG KHÔNG PHẢI MỘT TRỤC TEMPLATE.** Một poster khuyến mại và một poster tuyển dụng,
> cùng lượng chữ, cùng họ mask, thì **layout giống hệt nhau** — chỉ khác chữ, màu, font,
> icon. Ngành hàng là **style pack (dữ liệu)**, không phải template (file).
> Chỉ riêng việc tách trục này đã cắt ma trận 24 ô xuống ~15 file, mà phủ rộng hơn.

### 3.2. Trục 1 — 14 template hiện có ánh xạ vào họ mask

| Họ mask | Hình học | Template hiện có | mask_preset |
| :--- | :--- | :--- | :--- |
| **A. Sandwich** (2 dải) | Dải đỉnh 25–31% + dải đáy 15–18%, giữa mở cho sản phẩm | `sandwich_top_heavy`, `sandwich_bottom_heavy` | `sandwich_standard` |
| **B. Băng đơn** (1 dải ngang) | Một dải đỉnh **hoặc** một bệ đáy 35–42% | `recruitment_board` (top), `before_after_split` (bottom), `step_process_roadmap` (bottom) | `top_band`, `bottom_band` |
| **C. Cột sườn** | Cột dọc 38–45% trái hoặc phải | `split_left`, `split_right`, `l_frame_showcase` (cột + đáy = chữ L) | `split_*_full`, `l_frame` |
| **D. Khối nổi** (card / pod) | Khối bo góc ở tâm hoặc dạt góc | `luxury_centered_card`, `customer_feedback_card`, `lifestyle_corner_pod` | `luxury_card`, `feedback_card`, `corner_bl` |
| **E. Cắt chéo** | Đường chéo 15–25° chia đôi khung | `diagonal_slash` | `diagonal_slash` |
| **F. Đặc thù** (ngoài 5 họ chuẩn) | Lưới đều / trung tâm lễ hội | `menu_price_board`, `grand_opening_banner` | `menu_price_board`, `festive_center` |

**Nhận xét:** 12/14 template rơi gọn vào 5 họ chuẩn; 2 cái còn lại là đặc thù có lý do chính
đáng (lưới menu không có hero áp đảo; banner lễ hội cần tâm mở). Tài liệu cũ nói "5 họ mask"
— con số đó **đúng**, và đây là phần vững nhất của nó.

### 3.3. Trục 3 — Danh mục INTENT (thay cho "6 danh mục ngành")

Intent quyết định **ngưỡng tương phản**, nên nó phải là danh mục đóng, ít mục, không trùng lắp:

| intent | Đặc trưng thị giác | contrast_target | Template phù hợp |
| :--- | :--- | :--- | :--- |
| `big_number_deal` | Con số/phần trăm áp đảo, `hero_parts` gần như bắt buộc | **≥ 4.0x** | Sandwich, Cột sườn, Cắt chéo |
| `hook_headline` | Cụm từ khoá lớn, không có số | **≥ 4.0x** | Sandwich, Băng đơn, Khối nổi |
| `product_showcase` | Sản phẩm là chính, chữ nép | **≥ 4.0x** | Cột sườn, Khối nổi (pod) |
| `testimonial_trust` | Sao + trích dẫn + tên người | **≥ 3.0x** | Khối nổi (card) |
| `festive_event` | Khai trương, lễ hội, minigame | **≥ 3.5x** | Đặc thù (festive), Sandwich |
| `matrix_board` | **Các khối chữ CỠ TƯƠNG ĐƯƠNG, KHÔNG có hero áp đảo** | **≥ 2.5x** hero-vs-item, và **các item đều nhau (±15%)** | Lưới menu, Băng đơn (2 cột), Quy trình |

> **ĐÍNH CHÍNH TỰ NÊU:** phép đo `probe_capacity_map` chấm cả 13 template bằng một ngưỡng 4x
> duy nhất. Đó là **chấm sai đề** với nhóm `matrix_board`: một bảng menu mà tên món nhỏ hơn
> tiêu đề 4 lần là **sai chuẩn thiết kế**, không phải đạt chuẩn. Vì vậy 3 template đội sổ
> (`step_process_roadmap` 1.80, `recruitment_board` 1.89, `before_after_split` 1.99) **có thể
> không hề sai** — chúng bị đo bằng thước của thể loại khác. Bản đo lần 2 bắt buộc phải chấm
> theo đúng profile của từng intent.

### 3.4. Trục 2 — Sức chứa: ĐO, KHÔNG TUYÊN BỐ

Đây là chỗ ý tưởng **S/M/L/XL sống sót**, nhưng đổi vai trò:

> **Cũ:** S/M/L/XL là *bản vẽ thi công* — tuyên bố 24 ô rồi đi xây cho đủ.
> **Mới:** S/M/L/XL là *bản đồ phủ sóng* — mỗi template đã có sẵn sức chứa nội tại do hình học
> zone quyết định; ta **đo** nó, rồi **chỗ trống trên bản đồ mới là danh sách cần xây**.

Quy trình hiệu chuẩn (tự động, không hiệu chỉnh tay):

```
Với mỗi (template × tỉ lệ khung hình):
  tăng dần lượng nội dung
    → đo tương phản thật qua Chromium
    → đo có tràn không
  ĐIỂM GÃY = mốc ký tự cuối cùng còn giữ được contrast_target của intent đó
  ⇒ ghi vào catalog.py thành capacity_chars
```

Band sức chứa (ranh giới sẽ chốt bằng bản đo lần 2):

| Band | Ký tự nội dung | Đặc trưng |
| :--- | :--- | :--- |
| **S** | ≤ 60 | Hero bung tối đa 85–90px, không hoặc rất ít chữ phụ |
| **M** | 61 – 130 | Hero + subhead + 1–2 pill |
| **L** | 131 – 240 | Đầy đủ badge/hero/subhead/pills/store/CTA/QR |
| **XL** | > 240 | Bảng 2 cột, lưới giá, đoạn văn dài |

**Bản đo LẦN 1 (25/09, 212 case) — đọc kèm cảnh báo:**

| template | TB | max | đạt ≥4x | phần tử phụ to nhất |
| :--- | ---: | ---: | :--- | :--- |
| `sandwich_top_heavy` | **3.12** | 4.89 | 4/16 | subhead ← *đã sửa ở Mốc 0A* |
| `split_right` / `split_left` | 3.24 / 3.16 | 4.44 | 1/20 | subhead, cta |
| `luxury_centered_card` | 3.10 | 3.63 | 0/16 | subhead |
| `diagonal_slash` | 2.98 | 4.09 | 2/16 | subhead |
| `sandwich_bottom_heavy` | 2.74 | 5.09 | 3/16 | subhead, store_info |
| `l_frame_showcase` | 2.56 | 3.60 | 0/13 | extra-tag-row |
| `menu_price_board` | 2.52 | 3.12 | 0/15 | subhead |
| `grand_opening_banner` | 2.48 | 3.54 | 0/16 | freetext, cta |
| `lifestyle_corner_pod` | 2.09 | 2.44 | 0/16 | subhead |
| `before_after_split` | 1.99 | 2.49 | 0/16 | subhead |
| `recruitment_board` | 1.89 | 2.29 | 0/16 | subhead |
| `step_process_roadmap` | 1.80 | 1.97 | 0/16 | subhead |

> ⚠️ **BẢNG NÀY ĐANG ĐO CHÍNH SÁCH, CHƯA ĐO HÌNH HỌC.** Mốc 0A mới sửa trần subhead cho đúng
> **1/14** hàm `compute_*_budget` — và đúng cái đó đứng đầu bảng. Cột cuối cho thấy **9/13
> template bị chính SUBHEAD ghim trần**, tức cùng một nguyên nhân đã chứng minh là sửa được.
> **Không được dùng bảng này để quyết định xây template mới.** Phải đo lại sau §4.1 + §4.2.

### 3.5. Khi nào mới được xây template mới

Đúng một điều kiện: **bản đồ phủ sóng lần 2 cho thấy một ô (họ mask × intent × band) trống,
VÀ có brief thật rơi vào ô đó.** Không xây theo trực giác, không xây cho đủ ma trận.

Dự đoán (chưa đo, ghi ra để sau này đối chiếu): vùng **S/M phủ dày** (sandwich, split, pod,
diagonal, luxury — 8–9 template), vùng **XL mỏng** (chỉ recruitment/menu/step). Nếu đúng,
số template cần xây là **3–5 cái, gần như toàn bộ ở band XL**, không phải 24.

---

## 4. CÂU HỎI 3 — ĐẢM BẢO THẨM MỸ THẾ NÀO

Thẩm mỹ trong hệ thống này **không để cho cảm tính**. Nó được quy về 5 luật đo được, và mọi
luật đều có ngưỡng số để CI chặn.

### 4.1. LUẬT 1 — Thang cỡ chữ chuẩn poster (KHÔNG phải chuẩn văn bản)

**Bệnh:** thang 1.333 (Perfect Fourth) / 1.618 (Golden Ratio) là thang cho **văn bản nhiều
cấp liền kề**. Poster chỉ có 3 cấp và được nhìn trong 1.5–3 giây — thang đó quá nông. Tệ hơn,
trần các cấp dưới hiện là **hằng số cố định** (subhead 20/26/28px) nên **luôn thắng thế**:
đo thật thấy subhead bị ghim đúng ở trần 28px trong khi hero chỉ đạt 43–45px.

**Luật mới — mọi cấp dưới suy xuống từ trần hero, không còn hằng số cố định:**

```python
subhead_max = clamp(hero_max / 4.0,  13, 24)   # ✅ ĐÃ LÀM cho compute_sandwich_top_budget
cta_max     = clamp(hero_max / 4.5,  13, 22)   # ❌ chưa
extra_max   = clamp(hero_max / 5.0,  11, 20)   # ❌ chưa  ← thủ phạm đảo ngược thứ bậc
badge_max   = clamp(hero_max / 5.5,  11, 18)   # ❌ chưa
store_max   = clamp(hero_max / 5.5,  11, 18)   # ❌ chưa
```

Luật này cưỡng chế đúng **Luật Ba Tầng Thị Giác** của DESIGN_PRINCIPLES §1.1:
`Cấp 1 (hero) ≫ Cấp 2 (subhead) > Cấp 3 (pill/cta/store)`.

**Lỗi đảo ngược thứ bậc đang tồn tại, đo được:** `medium/1_1` có extra pills **21.5px** trong
khi subhead chỉ **15px** — Chi tiết Cấp 3 đang **to hơn** Ngữ cảnh Cấp 2. Đây là việc phải vá
ngay.

**Sàn không phải để ép chữ nhỏ thêm — sàn là TÍN HIỆU.** Khi `subhead_max` chạm sàn 13px
(tức `hero_max < 52`), nghĩa là nội dung đã quá dày cho template này ⇒ kích hoạt Cổng 3,
**reroute**, chứ không phải ép nhỏ tiếp.

> **KẾT QUẢ ĐO GĐ 0A+ (26/09) — công thức `cta/extra/badge/store` ở trên ĐÃ BỊ BÁC, không áp.**
> Đo bằng `scripts/probe_type_hierarchy.py` trên 379 case (toàn bộ suite JSON):
>
> | | Baseline | Công thức §4.1 nguyên văn | **Đã áp: Cấp 3 ≤ Cấp 2** |
> | :--- | ---: | ---: | ---: |
> | Case có Cấp 3 to hơn subhead | 53 | 0 | **0** |
> | Trung vị cỡ Cấp 3 | 16.5px | **12px** | 16.5px |
> | Phần tử Cấp 3 < 13px | 17/993 | **575/993** | 17/993 |
> | Subhead < 14px | 0/290 | 93/290 | 0/290 |
> | Case đạt ≥4x | 23 | 78 | 24 |
> | Hero đổi cỡ | — | 0 case | 0 case |
>
> Vì sao bác: công thức chia theo **trần** hero, mà 9/14 template có trần hero chỉ 52–68px ⇒
> chữ phụ rơi xuống 11–13px **kể cả ở poster ÍT chữ** — Cổng 3 (reroute) không cứu được vì
> poster ít chữ không có lý do để reroute. Toàn bộ mức tăng 23→78 case ≥4x đến từ **thu nhỏ
> chữ phụ**, hero không to thêm pixel nào.
>
> Đã áp thay thế (2 lớp, chỉ co — không bao giờ phóng, nên không sinh tràn mới):
> 1. `renderer.py::_apply_tier_caps` — trần mọi phần tử Cấp 3 ≤ trần subhead (khi có subhead).
>    Riêng lớp này còn 22 case đảo bậc: subhead hay bị **chiều cao** khoá dưới trần của nó.
> 2. Bước 3 trong `styles.py::COMMON_AUTOFIT_JS` — sau autofit, phần tử Cấp 3 nào to hơn
>    subhead **đo thật** thì co về bằng nó. Danh sách class theo cấp: `styles.py::TIER*_CLASSES`.
>
> Chỉ đúng 53 case từng đảo bậc bị co chữ phụ (tối đa 6px). **Bài học rút ra cho các GĐ sau:**
> tương phản thấp ở 9 template **không phải do chữ phụ to — mà do hero nhỏ** (bị chiều cao
> khoá). Đòn bẩy thật là làm hero TO lên (Luật 2 `hero_parts`, phân bổ chiều cao), không phải
> ép chữ phụ nhỏ xuống.

### 4.2. LUẬT 2 — Tương phản trong dòng (nguồn tương phản thứ hai, miễn phí chiều cao)

Luật 1 có trần vật lý: trong một dải đỉnh cao 280px, hero không thể to mãi. Nguồn tương phản
thứ hai phải đến từ **bên trong chính dòng hero**:

```
.hero-seg--stat    → 1.00em   (điểm neo, được phép chiếm thị giác)
.hero-seg--prefix  → 0.38em   (chữ dẫn, nép)
.hero-seg--suffix  → 0.38em   (chữ đuôi, nép)
                     ⇒ 2.63x tương phản, TỐN 0 PIXEL CHIỀU CAO
```

Kỹ thuật: `display:flex; align-items:baseline; flex-wrap:wrap`, mọi cỡ con dùng đơn vị `em`
để tự co theo font-size mà autofit gán cho phần tử cha.

**Hiệu ứng phụ đã đo, quan trọng hơn cả con số 2.63x:** markup làm hero base tăng **43 → 61.5px**
— tức nó **giải phóng hero khỏi ràng buộc chiều cao**, đưa trần `max_font` trở lại làm ràng
buộc thật. Nhờ đó Luật 1 mới có tác dụng. **Hai luật cộng hưởng, không cộng tuyến tính.**

### 4.3. LUẬT 3 — Màu 60-30-10 và tương phản nền

- **60%** nền (ảnh diffusion) · **30%** khối card/dải đỡ · **10%** màu nhấn.
- Màu nhấn **chỉ** dành cho: `hero_parts[role=stat][emphasis=accent]` và nút CTA. Không rải
  màu nhấn ra toàn poster.
- Mọi màu chữ **bắt buộc** đi qua `ensure_contrast()` với **màu nền ĐO ĐƯỢC** của đúng vùng
  đó (đã có: `sample_bg_luminance_for_zone`, WCAG 2.1, ≥3.0:1 large text, ≥4.5:1 small text).
- `-webkit-text-fill-color` bắt buộc trên màu nhấn để thắng `background-clip:text` của các
  hiệu ứng chất liệu (`3d_gold`, `chrome`) — nếu không, màu nhấn bị nuốt.
- **Khoảng trống còn lại:** hiện đo **luminance trung bình** của zone. Nền nhiễu cao (trung
  bình "tối" nhưng có vệt sáng chói ngay dưới nét chữ) vẫn chìm chữ. Cần bổ sung đo
  **worst-case patch / phương sai**, vượt ngưỡng mới bật scrim.

### 4.4. LUẬT 4 — Danh mục hiệu ứng thị giác (đóng, gắn với intent)

**Giữ và dùng:**

| Nhóm | Thành phần | Trạng thái |
| :--- | :--- | :--- |
| Hiệu ứng chữ | 10 mục sẵn có: `3d_gold`, `neon`, `neon_bloom`, `chrome`, `fire`, `shadow`, `embossed`, `chromatic`, `hologram`, `plain_elegant` | ✅ có |
| Bóng đổ đa tầng | Open Props Shadow Tokens (`--shadow-3..5`) | ❌ làm |
| Chất liệu 3D | SVG Filters (`feSpecularLighting`, `feDistantLight`) — kim loại, gel nước | ❌ làm |
| Linh kiện đồ hoạ | Hero Stat, Ribbon Banner (`clip-path` đuôi nheo), Split Capsule | ❌ làm |
| Chữ uốn cung | **SVG `<textPath>` thuần** | ❌ làm |
| Hạt lấp lánh | Rải hạt **tĩnh có seed** (CSS radial-gradient / `<circle>` sinh theo seed) | ❌ làm |

**Loại bỏ, kèm lý do kỹ thuật:**

| Thư viện | Vì sao loại |
| :--- | :--- |
| **CircleType.js** | Đo DOM rồi ghi lại từng span lúc load → (a) phải chờ layout settle trước khi Playwright chụp; (b) **xung đột trực tiếp với binary-search autofit** — autofit đổi font-size thì vòng cung phải tính lại. `<textPath>` cho kết quả tương đương, zero JS, zero timing. |
| **Canvas-Confetti** | Là thư viện **animation** chạy bằng `requestAnimationFrame`. Chụp 1 khung hình = đóng băng ở frame ngẫu nhiên → **render không tất định**, phá tính lặp lại của test suite. |
| **Chroma.js / Colord** | Làm lại thứ Python đã có (`get_zone_adaptive_text_colors`, `ensure_contrast`). Hai bản cài đặt cùng một logic bằng hai ngôn ngữ = hai nơi phải sửa. |

**Luật gắn hiệu ứng với intent:** mỗi hiệu ứng khai báo intent nào được dùng, để tránh
`neon` rơi vào thiệp mời VIP hay `3d_gold` rơi vào poster tuyển dụng kỹ thuật.

### 4.5. LUẬT 5 — SQUINT TEST TỰ ĐỘNG HOÁ (nghiệm thu thẩm mỹ bằng CI)

Đây là đóng góp lớn nhất về mặt quy trình: **biến "đẹp" từ cảm nhận thành 4 điều kiện chặn
được trong CI.** Mỗi lần render test, đo DOM và kiểm:

| # | Điều kiện | Ngưỡng |
| :-- | :--- | :--- |
| 1 | **Tương phản điểm neo** = cỡ `stat` (hoặc cả hero) / cỡ phần tử phụ **to nhất** | ≥ `contrast_target` của intent (§3.3) |
| 2 | **Chống bức tường chữ** — không được có ≥3 phần tử nằm trong khoảng ±20% cỡ của nhau | 0 vi phạm |
| 3 | **Tương phản nền** — WCAG với màu nền đo thật | ≥3.0:1 hero, ≥4.5:1 chữ nhỏ |
| 4 | **Không tràn** — không phần tử `[data-autofit]` nào kẹt ở sàn mà `scrollHeight > maxH` | 0 vi phạm |

Điều kiện 2 là phiên bản code của chính câu cảnh báo trong DESIGN_PRINCIPLES §1.2:
*"Nếu thấy mọi chữ nhòe nhoẹt ngang nhau, bố cục đã thất bại."*

**Tình trạng hiện tại:** test suite 323 case **chỉ kiểm không-crash**, không kiểm một điều
kiện nào ở trên. Bằng chứng: 6 case đang tràn thật mà 323/323 vẫn xanh.

---

## 5. CÂU HỎI 4 — DÙNG CÁC LAYOUT CƠ BẢN NÀO

### 5.1. Năm họ mask chuẩn (giữ nguyên từ tài liệu cũ — phần vững nhất của nó)

| Họ | Vùng mask | Dành cho | Band sức chứa dự kiến |
| :--- | :--- | :--- | :--- |
| **A. Sandwich** | Dải đỉnh 25–31% + dải đáy 15–18%; mở 55% giữa | Đồ ăn, thức uống, sản phẩm đặt giữa | S → M |
| **B. Băng đơn** | Một dải đỉnh, hoặc bệ đáy sâu 35–42% | Phong cảnh, bất động sản, bảng tin, quy trình | M → XL |
| **C. Cột sườn** | Cột dọc 38–45% trái/phải (biến thể chữ L thêm dải đáy) | Xe hơi, thời trang, đồng hồ, bảng thông số | M → L |
| **D. Khối nổi** | Card bo góc ở tâm, hoặc pod dạt góc | Feedback, mỹ phẩm cao cấp, thiệp mời | S → M |
| **E. Cắt chéo** | Đường chéo 15–25° chia đôi khung | Thể thao, sneaker, âm nhạc, fintech | S → M |

Ràng buộc chung: **diện tích mask ≤ 50% canvas** (đã cưỡng chế trong test hiện có).

### 5.2. Hai layout đặc thù (ngoài 5 họ, có lý do chính đáng)

- **Lưới đều / Menu ma trận** (`menu_price_board`): các khối chữ cỡ tương đương, **không có
  hero áp đảo** — đây là `intent = matrix_board`, chấm bằng ngưỡng 2.5x chứ không phải 4x.
- **Trung tâm lễ hội** (`grand_opening_banner`): tâm mở cho chữ cong + hạt lấp lánh.

### 5.3. Chế độ thứ sáu — MASKLESS (không phải một họ mask)

Dành cho poster **lấy chữ làm nhân vật chính** (sale thuần chữ, tuyển dụng lớn, chúc mừng).
Bỏ luồng corridor → `denoise_regional_velocity_blended` chạy batch 1 thay vì batch 2.

**Ba ràng buộc bắt buộc, không được bỏ qua:**

1. **Chỉ áp dụng cho background ÍT CHI TIẾT** (gradient, bokeh, bụi vàng, khói, texture phẳng).
   Bỏ corridor = bỏ **cơ chế duy nhất** tạo vùng tĩnh có bảo đảm; nền rậm rạp sẽ khiến chữ khó
   đọc **hơn cả** khi có mask. (Các ví dụ trong tài liệu cũ đều là nền ít chi tiết — đã trực
   giác đúng nhưng chưa phát biểu thành luật.)
2. **Luật 3 (tương phản nền) phải hoàn thiện TRƯỚC.** Maskless phụ thuộc 100% vào nó.
3. **Không ghi "nhanh gấp 2 lần" vào tài liệu bàn giao trước khi đo.** Bỏ corridor giảm ~½
   FLOPs của **riêng pha DiT**; wall-clock còn text-encode + VAE decode + Playwright nên
   end-to-end sẽ **thấp hơn** 2x.

---

## 6. CÂU HỎI 5 — CODE THẾ NÀO DỄ BẢO TRÌ

### 6.1. Bệnh hiện tại: 1 template = 6 điểm chạm

| Nơi | Phải sửa gì cho mỗi template mới |
| :--- | :--- |
| `templates/<tên>/template.html` | ~420 dòng (đã migrate: ~170) |
| `geometry.py` | 1 hàm zone riêng (30–80 dòng) + ghi danh vào 4 set cờ |
| `renderer.py` | 1 hàm `compute_*_budget()` riêng (60–110 dòng), **hiệu chỉnh tay theo số đo Playwright** |
| `catalog.py` | 1 entry |
| `llm_prompts.py` | prose mô tả riêng |
| `llm_planner.py` | nhánh xử lý riêng |
| `tests/*.json` | 1–3 bộ suite × 16 case |

**Tổng ≈ 600 dòng trải 6 file + 16–48 lần render nghiệm thu.** Con số "15–20 phút/template"
trong tài liệu cũ chỉ tính phần HTML — tức ~25% khối lượng thật.

Triệu chứng của bệnh này đã hiện rõ trong code: `geometry.py` có **23 dòng comment** giải
thích vì sao `_DENSITY_AWARE_TEMPLATES` phải tắt — nguyên nhân gốc ghi thẳng trong comment là
*"công thức không hề biết template nào THỰC SỰ dùng field nào"*.

### 6.2. Kiến trúc đích: 1 template = 2 điểm chạm

```
catalog.py  ──[khai báo declarative, máy đọc được]──┐
                                                     ├─→ system prompt cho LLM
                                                     ├─→ validate schema (Cổng 2)
                                                     ├─→ kwarg cho get_zones()
                                                     ├─→ hợp đồng dung lượng (Cổng 3)
                                                     └─→ ngưỡng nghiệm thu CI (Luật 5)
```

Mỗi template khai báo **một lần, một chỗ**:

```python
"sandwich_top_heavy": {
    "name": "Băng Kẹp Trên - Dưới",
    "hint": "...",                      # 1 dòng, KHÔNG còn 200 từ
    "mask_family": "sandwich",
    "mask_preset": "sandwich_standard",
    "visual_intents": ["big_number_deal", "hook_headline", "product_showcase"],
    "contrast_target": 4.0,             # suy từ intent, ghi đè được
    "slots": {                          # ★ nguồn sự thật DUY NHẤT về field
        "hero":        {"required": True,  "supports_markup": True},
        "subhead":     {"required": False},
        "badge":       {"required": False},
        "extra_texts": {"required": False, "max_items": 4},
        "cta":         {"required": False},
        "store_info":  {"required": False},
        "qr_code":     {"required": False, "drives_geometry": True},
    },
    "capacity_chars": {"1:1": 130, "9:16": 118, "16:9": 92, "4:5": 126},  # ★ ĐO, không đoán
    "aspect_ratios": ["1:1", "9:16", "16:9", "4:5"],
}
```

**Xoá được nhờ khai báo này:** 4 set hard-code trong `geometry.py`
(`_HAS_QR_TEMPLATES`, `_HAS_FOOTER_TEMPLATES`, `_HAS_MESSAGE_TEMPLATES`,
`_HAS_FREETEXT_TEMPLATES`) — thay bằng đọc `slots[...].drives_geometry`. Và toàn bộ prose
lặp lại trong `llm_prompts.py`.

### 6.3. Phân tầng trách nhiệm

| Tầng | File | Trách nhiệm | Nguyên tắc |
| :--- | :--- | :--- | :--- |
| **Khai báo** | `catalog.py` | Dữ liệu thuần, không logic | Nguồn sự thật duy nhất |
| **Hình học** | `geometry.py` | `(template, w, h, cờ) → zones` | Hàm thuần, không đọc plan |
| **Ngân sách** | `renderer.py` | `(zones, nội dung, intent) → cỡ chữ` | **MỘT hàm chung** + override tối thiểu, thay 14 hàm chép tay |
| **Layout** | `templates/*/template.html` | Chỉ bố cục, gọi macro | ~170 dòng, không CSS chép tay |
| **Linh kiện** | `templates/components/` | Macro + CSS dùng chung | Zero-Trace: rỗng → không sinh thẻ |
| **Kiểm soát** | validator mới | 4 cổng §2.5 | Fail-safe, không bao giờ sai chữ |
| **Nghiệm thu** | `tests/` | 4 điều kiện §4.5 | Đo DOM thật, không chỉ kiểm crash |

Ghi chú về tầng Ngân sách: 14 hàm `compute_*_budget()` hiện nay **không thật sự khác nhau về
thuật toán** — chúng khác nhau ở các hằng số tỉ lệ chiều cao và trần cỡ chữ. Sau khi áp Luật 1
(mọi cấp suy từ hero), phần lớn khác biệt đó tan biến ⇒ gom về một hàm chung nhận `zones` +
`slots` + `intent`, còn override chỉ giữ ở những chỗ thật sự có hình học đặc thù.

### 6.4. Bốn quy tắc bất biến cho mọi thay đổi về sau

1. **Không thêm hằng số hiệu chỉnh tay mới nếu chưa có probe đo.** Mọi hằng số trong
   `geometry.py`/`renderer.py` phải trỏ được về một phép đo.
2. **Mọi phát biểu về cỡ chữ phải đo qua Chromium.** Bài học đã trả giá: ước lượng từ trần
   `max_font` cho 2.2x, đo thật chỉ 1.62x — vì hero bị ngân sách **chiều cao** khoá trước khi
   chạm trần.
3. **Test phải kiểm THẨM MỸ, không chỉ kiểm không-crash.** 323/323 xanh trong khi 6 case tràn
   thật là bằng chứng bộ test hiện tại chưa đủ.
4. **Đóng băng `src/flux2/` — không sửa mã gốc BFL.** Mọi mở rộng nằm ở tầng ngoài (quy tắc cũ, vẫn áp dụng).

---

## 7. LỘ TRÌNH THI CÔNG

Không có deadline cứng. Xếp theo **thứ tự phụ thuộc**, mỗi giai đoạn có tiêu chí nghiệm thu
đo được.

| GĐ | Nội dung | Nghiệm thu (Definition of Done) | Trạng thái |
| :-- | :--- | :--- | :--- |
| **0A** | Tầng markup ngữ nghĩa (`hero_parts` + `.hero-phrase` + macro) và thang cỡ chữ poster cho `sandwich_top_heavy` | Tương phản 1.62x → 4.10x, 0 tràn, test xanh | ✅ **XONG 25/09** |
| **0A+** | Lan Luật 1 ra 13 hàm budget còn lại; vá đảo ngược thứ bậc Cấp 2 vs Cấp 3 (extra/cta/badge) | Không còn phần tử Cấp 3 nào to hơn Cấp 2 trên cả 212 case | ✅ **XONG 26/09** — 0/379 case đảo bậc (từ 53), cỡ chữ phụ giữ nguyên ngoài 53 case đó; **không** dùng công thức chia theo hero (xem kết quả đo cuối §4.1) |
| **0B** | `catalog.py` declarative (§6.2); migrate 13 template sang component layer; Cổng 2+3 | Thêm template mới chỉ chạm **2 file**; 4 set cờ hard-code bị xoá | 🔄 **Đang làm.** ✅ 0B-2 Cổng 2 (xem §2.5). ✅ 0B-1 (26/09): `slots` khai báo cho 14 template (lấy từ template.html, có test giữ đồng bộ), 4 set cờ `_HAS_*_TEMPLATES` đã xoá, 3 bản sao cách tính cờ gom về `renderer.compute_geometry_flags` — HTML + mask của 379 case **giống hệt từng byte** trước/sau. ⚠️ Cổng 3 cần `capacity_chars` **đo ở GĐ 1** — 0B chỉ dựng được khung, chưa có số để reroute |
| **0C** | Cổng 4 — phát hiện tràn lúc chạy (`window.__tendoo_overflow` + `page.evaluate`) | 6 case tràn hiện tại bị bắt và phân loại đúng (thật / báo động giả) | ✅ **XONG 26/09** (làm TRƯỚC 0B-3 theo §8: cần lưới phát hiện tràn trước khi viết lại HTML 13 template). Trên 379 case: 17 phần tử / 15 case vượt ngân sách → **2 mất chữ thật** (`sbh_11_16x9_heavy` ×2 suite: dòng email dải đỉnh bị cắt nửa) + 15 báo động giả. Đối chiếu bằng mắt 8/17 phần tử, khớp 8/8. Test `test_no_new_text_loss` chốt danh sách |
| **1** | Squint test tự động hoá (§4.5) vào CI; đo lại bản đồ phủ sóng **lần 2** theo đúng profile intent | Có bảng sức chứa thật ⇒ điền `capacity_chars` vào catalog | |
| **2** | Linh kiện UI: Hero Stat, Ribbon, Split Capsule, Open Props shadows, SVG filters, `<textPath>` | Render mẫu đạt 4 điều kiện §4.5 với đủ các hiệu ứng | |
| **3** | Đưa `hero_parts` + `visual_intent` vào system prompt LLM; hạ hint xuống 1 dòng/template | Poster sinh từ brief thật (không phải plan viết tay) đạt §4.5 | |
| **4** | Siết tương phản nền: worst-case patch + ngưỡng bật scrim | Chữ đạt WCAG AA cả trên nền có vệt sáng cục bộ | |
| **5** | Maskless Mode + **đo** tốc độ thật | Poster thuần chữ đạt §4.5; có số wall-clock thật | |
| **6** | Template mới — **chỉ cho các ô trống mà bản đồ lần 2 chỉ ra** | Mỗi cái: 2 file + 16-case suite đạt §4.5 | Dự kiến 3–5 cái, không phải 24 |

---

## 8. RỦI RO ĐÃ BIẾT

| Rủi ro | Mức | Giảm thiểu |
| :--- | :--- | :--- |
| Bản đồ sức chứa lần 1 bị hiểu nhầm là kết luận cuối | **Cao** | Đã ghi cảnh báo ngay trong §3.4; cấm dùng để quyết định xây template |
| Ngưỡng 4x áp nhầm cho `matrix_board` | **Cao** | `contrast_target` khai báo theo intent, không phải hằng số toàn cục |
| Gom 14 hàm budget về 1 hàm chung gây hồi quy hàng loạt | Trung bình | Làm sau 0C (đã có cổng phát hiện tràn) và sau khi CI có §4.5; migrate từng cái, đo từng cái |
| LLM cắt `hero_parts` sai chỗ (gán `stat` cho cụm không phải cái móc) | Trung bình | Không làm sai chữ (Cổng 1 bảo đảm); chỉ giảm thẩm mỹ. Đo qua §4.5 điều kiện 1 |
| Markup làm hero chiếm nhiều dòng hơn khi wrap | Thấp | Đã quan sát ngược lại (hero base tăng 43→61.5px); vẫn phải theo dõi ở 9:16 |
| `from_dict()` nhận `style` là object thì âm thầm rơi về mặc định | Thấp | Lỗi sai-thầm-lặng, đã ghi nhận, vá ở 0B |
| §4.5 điều kiện 2 ("≥3 phần tử trong ±20%") mâu thuẫn với chính thứ bậc 3 tầng: poster có ≥3 chi tiết Cấp 3 (hotline + địa chỉ + CTA…) gần như luôn vi phạm. Đo 26/09: **195/379** case vi phạm ở baseline, **205/379** sau 0A+ (chữ phụ bị co về cùng cỡ subhead) | **Cao** | Định nghĩa lại trước GĐ 1 (vd chỉ xét giữa các CẤP khác nhau, không xét trong cùng Cấp 3) — nếu đưa nguyên văn vào CI sẽ chặn hơn nửa số poster |
| Khi subhead bị chiều cao khoá nhỏ, 0A+ kéo cả hotline/CTA xuống theo (vd `sth_03_1x1_medium`: subhead 15px ⇒ store/cta 18–19.5 → 15px) | Trung bình | Chưa sửa. Đòn bẩy đúng là cấp thêm chiều cao cho subhead/hero, không phải nới Cấp 3 |
| Field LLM trả về nhưng template không có chỗ hiển thị bị **mất âm thầm** — vd `l_frame_showcase` không có `cta` trong template, không case test nào có cta cho template này. Thấy được nhờ `slots` (0B-1) | **Cao** | Cổng 2 (0B-2): log cảnh báo khi plan có field ngoài `slots`; quyết định thiết kế (thêm chỗ cho CTA hay cấm LLM chọn) là việc của người duyệt |
| **Dữ liệu test dùng tên style không tồn tại** (Cổng 2 bắt được, 26/09): 84/379 case dùng `text_effect` lạ (`gold_3d`, `chrome_silver`, `warm_glow`, `subtle_depth`…) ⇒ render thành chữ phẳng — các hiệu ứng đó **chưa từng được test thật**; 19 case tông nền lạ (`pastel_soft` ý là nền SÁNG nhưng bị xử lý như nền tối); 13 case font ngoài danh mục (`cinzel`, `cormorant`…) | ~~Cao~~ **Đã sửa 26/09** | 116 giá trị trong 24 file suite đổi về tên hợp lệ gần nghĩa nhất (`gold_3d`→`3d_gold`, `*_glow`→`neon_bloom` theo alias `glow` có sẵn, `pastel_soft`→`pastel`, `cinzel`/`cormorant`→`playfair`…; bảng đầy đủ trong commit). Đo lại 379 case: 0 đảo bậc, đúng 15 case tràn cũ, không case mới. Cổng 2 trên suite: 0 lỗi style (còn 14 cảnh báo `required` — các case sparse CỐ Ý thiếu field) |
| **Nền giả trong test luôn tối** — `run_template_test.generate_mock_backdrop_data_uri` bỏ qua `background_tone` ⇒ poster tông sáng (`pastel`, `light_clean`) **chưa từng được test trên nền sáng**; lỗi kiểu chữ sáng trên nền sáng sẽ lọt | Trung bình | Cho nền giả theo tông (đã có `velocity_blending` mock theo tông). Sẽ đổi ảnh gốc của các case tông sáng → làm cùng lúc với đo tương phản nền (GĐ 4) |
| LLM **không được cho danh sách `background_tone` hợp lệ** (prompt chỉ có 1 ví dụ) ⇒ tự nghĩ tên tông | Trung bình | Thêm danh mục tông vào system prompt ở GĐ 3 (cần thử với LLM thật) |
| `sbh_11_16x9_heavy` (sandwich_bottom_heavy 16:9, 4 dòng store ở dải đỉnh) **mất chữ thật**: dòng email bị cắt — Cổng 4 bắt được (0C), chưa sửa. 13 case "tràn" còn lại là báo động giả (hiển thị đủ) | Trung bình | Sửa hình học dải đỉnh 16:9 hoặc giới hạn số dòng store — cần probe đo trước khi đổi hằng số (§6.4 quy tắc 1) |

---

## 9. TÓM TẮT MỘT TRANG

- **LLM nhận** brief + danh mục intent/font/hiệu ứng (KHÔNG nhận capacity, CSS, toạ độ).
  **LLM xuất** intent + nội dung đã **cắt vai trò** (`hero_parts`) + style + prompt cảnh.
  **LLM đề xuất template, Python phủ quyết bằng số đo.**
- **Template chia theo 3 trục độc lập:** họ mask (5+2) × sức chứa (**đo được**, S/M/L/XL) ×
  intent (6). **Ngành hàng KHÔNG phải trục template** — nó là style pack.
- **Thẩm mỹ được bảo đảm bằng 5 luật đo được:** thang cỡ chữ suy từ hero (không hằng số cố
  định) · tương phản trong dòng qua markup · màu 60-30-10 + WCAG trên nền đo thật · danh mục
  hiệu ứng đóng · **squint test tự động hoá thành 4 điều kiện CI**.
- **Layout cơ bản:** 5 họ mask chuẩn + 2 đặc thù + 1 chế độ maskless (có 3 ràng buộc).
- **Dễ bảo trì bằng cách** kéo 1 template từ **6 điểm chạm xuống 2**, với `catalog.py`
  declarative làm nguồn sự thật duy nhất cho prompt, validate, geometry, hợp đồng và CI.
- **Số template cần xây thêm: dự kiến 3–5, không phải 24** — và chỉ xây sau khi bản đồ phủ
  sóng lần 2 chỉ ra ô trống thật.
