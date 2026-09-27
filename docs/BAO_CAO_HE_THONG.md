# BÁO CÁO TỔNG THỂ — HỆ THỐNG TENDOO V3 HOẠT ĐỘNG NHƯ THẾ NÀO

> Cập nhật: 27/09/2026 · Commit gần nhất: `899f8a0` · Dành cho người điều hành dự án (không cần nền tảng thiết kế).
> Tài liệu kỹ thuật chi tiết và lịch sử quyết định: [ROADMAP.md](../ROADMAP.md). Quy tắc cho người/AI sửa code: [AGENTS.md](../AGENTS.md).

---

## 0. TÓM TẮT MỘT TRANG

**Tendoo v3 làm gì:** người dùng điền một form ngắn (loại poster, tiêu đề, giá, địa chỉ…) và/hoặc gõ một câu mô tả,
chọn khổ ảnh (1:1, 4:5, 9:16, 16:9). Hệ thống trả về **poster quảng cáo hoàn chỉnh**: ảnh nền đẹp + chữ tiếng Việt
sắc nét, đúng thứ bậc, đủ tương phản, đọc được trên điện thoại.

**Ý tưởng cốt lõi — chia việc cho đúng "người":**

| Ai | Làm gì | Không bao giờ làm gì |
| :--- | :--- | :--- |
| **LLM** (hiện tạm dùng GPT) | Hiểu brief, viết/cắt nội dung, chọn mẫu, chọn phong cách, mô tả cảnh nền | Quyết định cỡ chữ, toạ độ |
| **Python** (luật cố định) | Kiểm tra, phủ quyết, tính kích thước, sửa lỗi | Viết lại nội dung của người dùng |
| **Diffusion** (FLUX.2, trên GPU) | Vẽ **ảnh nền**, chừa sẵn vùng trống cho chữ | **Vẽ chữ** (AI vẽ chữ tiếng Việt sai dấu) |
| **Trình duyệt Chromium** | Sắp chữ bằng HTML/CSS, đo thật, chụp ảnh | — |

**Hiện trạng đo được (27/09):**
- 17 mẫu (template) phủ **12 nhu cầu** người dùng; 28 font tiếng Việt (9 font miễn phí thương mại mới); 8 bộ phong cách theo dịp; 5 kiểu cụm tiêu đề.
- Bộ kiểm thử tự động: **491 poster mẫu**, 182 đạt cả 5 điều kiện thẩm mỹ, 179 đạt cả 6; **476 test** tự động đều xanh.
- Poster sinh từ **GPT thật + ảnh nền thật** (26 đề bài thực tế): **22/26 đạt cả 5 điều kiện**, 0 poster mất chữ.

---

## 1. ĐẦU VÀO VÀ ĐẦU RA

**Đầu vào**
- Form theo loại: khuyến mãi, giới thiệu sản phẩm, khai trương, đánh giá khách hàng, tuyển dụng, hướng dẫn/quy trình,
  thông báo, thiệp chúc mừng, thư mời, trước/sau, minigame, combo/bảng giá.
- Một câu mô tả tự do (vd *"Poster Tết thuần chữ, nền đỏ bụi vàng lấp lánh"*).
- Khổ ảnh; tuỳ chọn: ảnh sản phẩm thật, link QR, logo/màu/font thương hiệu.

**Đầu ra**
- Ảnh poster PNG (màn hình 1024px; bản in A4 300dpi = 3508px).
- File HTML của poster (sửa được), ảnh nền, mask, bản kế hoạch (JSON) và báo cáo kiểm tra.
- Tuỳ chọn: nhiều phương án kiểu chữ trên cùng một nền.

---

## 2. MỘT POSTER ĐƯỢC TẠO RA QUA 8 BƯỚC

```
Form + mô tả
   │
   ▼
[1] LLM lập kế hoạch ──► bản kế hoạch (plan): mẫu, nội dung đã cắt vai trò, phong cách, mô tả cảnh
   │
   ▼
[2] Python kiểm tra & phủ quyết (Cổng 1-2-3): đúng chữ? hợp lệ? mẫu có chứa nổi nội dung không?
   │
   ▼
[3] Tính vùng chữ + mask ──► biết chỗ nào trên ảnh sẽ đặt chữ
   │
   ▼
[4] Diffusion vẽ ảnh nền, giữ vùng chữ yên tĩnh (hoặc: không mask nếu poster thuần chữ / tấm thiệp)
   │
   ▼
[5] Dựng HTML: mẫu + chữ + phong cách + hoạ tiết + logo
   │
   ▼
[6] Chromium tự co giãn chữ (autofit) cho vừa khung, đúng thứ bậc, đọc được trên điện thoại
   │
   ▼
[7] Chỉnh tương phản theo ĐIỂM ẢNH THẬT của nền (đổi màu chữ / quầng / lớp mờ); đặt hoạ tiết, logo
   │
   ▼
[8] Cổng 4 báo mất chữ → chụp PNG
```

### Bước 1 — LLM lập kế hoạch
File: `llm_planner.py`, `llm_prompts.py`. LLM nhận form + mô tả + **danh mục sinh tự động từ code** (các mẫu, kiểu ý
đồ, linh kiện, bộ phong cách, font). LLM **không** nhận kích thước pixel hay CSS. LLM trả về một bản kế hoạch gồm:

- `template` (mẫu), `visual_intent` (ý đồ thị giác — xem §4.3).
- Nội dung: `hero` (tiêu đề), `subhead`, `badge`, `extra_texts`, `cta` (nút), `store_info`, `body` (đoạn nội dung
  thiệp), `steps`, `testimonial`…
- **`hero_parts`** — tiêu đề được **cắt theo vai trò**: `prefix` (chữ dẫn nhỏ) / `stat` (điểm neo to nhất) /
  `suffix` (chữ đuôi). Ví dụ *"GIẢM TỚI 25% TOÀN BỘ MENU"* → `GIẢM TỚI` + **`25%`** + `TOÀN BỘ MENU`. Đây là nguồn
  tương phản thị giác lớn nhất mà không tốn thêm chỗ.
- Phong cách: font, màu chủ đạo, hiệu ứng chữ, tông nền, linh kiện, kiểu cụm tiêu đề (`lockup`), bộ phong cách.
- `scene_prompt` — mô tả cảnh nền **không có chữ, không có kích thước**.

Không có LLM (mất mạng/không có khoá) → **bộ lập kế hoạch dự phòng** bằng luật, vẫn ra poster dùng được.

### Bước 2 — Python kiểm tra và phủ quyết (Cổng 1–3, chi tiết §5)
LLM **đề xuất**, Python **quyết**: chữ phải khớp nguyên văn; lựa chọn phải thuộc danh mục; mẫu phải chứa nổi nội dung
(đo trước); chữ lặp lại bị bỏ; poster thuần chữ tự chuyển sang mẫu chữ-làm-chính.

### Bước 3–4 — Vùng chữ, mask và ảnh nền
- `geometry.py`: mỗi mẫu khai báo **vùng chữ** (toạ độ) cho từng khổ ảnh. Cùng một nguồn dùng cho cả CSS lẫn mask →
  chữ luôn nằm đúng chỗ nền được chừa.
- `mask_engine.py`: vẽ **mask** (vùng trắng = chỗ đặt chữ), viền mềm, **luôn ≤ 50% diện tích** để còn chỗ cho sản phẩm.
- `velocity_blending.py` (GPU): FLUX.2 chạy **hai luồng cùng lúc** từ cùng một nhiễu — luồng *cảnh* (sản phẩm, bối
  cảnh) và luồng *hành lang* (nền mịn, yên tĩnh) — rồi trộn theo mask. Kết quả: ảnh liền mạch, vùng chữ tự nhiên dịu đi.
- **Không mask** khi: poster thuần chữ (`maskless`, nền ít chi tiết phủ cả khung) hoặc mẫu tự vẽ nền đặc dưới chữ
  (tấm thiệp).
- Ảnh sản phẩm thật (nếu có) được đưa vào qua cơ chế "ảnh tham chiếu" để giữ nguyên sản phẩm.
- **Hiện tại** trên máy local: ảnh nền do **GPT** vẽ (mô phỏng vùng chừa bằng lời) để kiểm tầng chữ; FLUX chạy trên
  máy chủ 2×A30.

### Bước 5 — Dựng HTML
`renderer.py` ghép: mẫu (`templates/<tên>/template.html`, Jinja2) + nội dung + **ngân sách cỡ chữ** (trần/sàn cho
từng phần tử, tính theo vùng và khổ ảnh) + bảng màu thích ứng + linh kiện + font nhúng sẵn (không cần internet).

### Bước 6 — Autofit: chữ tự co giãn trong trình duyệt
Script `styles.COMMON_AUTOFIT_JS` chạy **trong Chromium**, đo chữ thật (không đoán). Theo thứ tự:

| Bước | Việc | Vì sao |
| :-- | :--- | :--- |
| 1 | Tìm cỡ **to nhất** vừa khung cho từng phần tử (tìm nhị phân) | Chữ to hết mức có thể |
| 1 (lockup) | Đo cả kiểu cụm đặc biệt lẫn xếp ngang; chỉ giữ kiểu đặc biệt nếu con số không nhỏ đi quá ngưỡng | Đẹp nhưng không đánh đổi độ nổi |
| 1b, 3c | **Cứu chữ**: phần tử bị cắt thì cho co xuống dưới sàn (tới 12px) | Không bao giờ mất chữ |
| 2 | Chống cắt chữ (co khoảng cách, đệm) | |
| 3, 3a | **Thứ bậc**: phụ đề ≤ tiêu đề ÷ 1.6; chữ nhỏ (nút, nhãn…) ≤ 0.8 × phụ đề | Mắt biết đọc cái gì trước |
| 3d | Chữ **bị cắt/ra ngoài khung** → chỉ co chữ **cùng vùng**, chữ phụ trước, tiêu đề sau | Sửa tràn mà không phá tiêu đề |
| 4 | **Cổng 4**: báo phần tử bị cắt / đè nhau / tràn | Phát hiện mất chữ |

Sàn cỡ chữ theo **Luật 6 (đọc được trên điện thoại)**: poster thường được xem trên màn ~375px ngang, nên mọi chữ
≥ 10px *trên màn*, tiêu đề hướng tới ≥ 20px *trên màn*. Tức là cỡ chữ **tỉ lệ theo bề ngang poster**.

### Bước 7 — Tương phản theo điểm ảnh thật
Sau khi chữ đã chốt vị trí, script đọc **từng điểm ảnh nền dưới từng dòng chữ**, lấy ô xấu nhất rồi xử lý:
1. **Màu nhấn** (con số): giữ sắc, pha sáng/tối tới đủ tương phản với cả ô tối nhất lẫn ô sáng nhất.
2. **Chữ thường**: đảo trắng ↔ xanh đậm nếu cực kia tốt hơn rõ; quyết theo **nhóm** (một dòng một màu).
3. **Chữ ánh kim** (vàng, chrome, lửa…): thiếu tương phản → làm phẳng về một màu đặc (không dùng quầng — quầng vẽ đè
   lên nét làm chữ đen thui).
4. Thiếu ít → **quầng mềm** quanh chữ. Thiếu nhiều (nền lẫn sáng-tối) → **lớp mờ** sau dòng chữ, độ đậm tính theo số đo.

Cùng lúc đặt **hoạ tiết** của bộ phong cách và **logo** thương hiệu vào chỗ trống — không bao giờ chạm chữ.

### Bước 8 — Chụp ảnh
Chromium chụp PNG; báo cáo Cổng 4 đi kèm (mất chữ thật được ghi cảnh báo).

---

## 3. 17 MẪU (TEMPLATE) VÀ NHU CẦU CHÚNG PHỦ

| Mẫu | Bố cục | Hợp cho |
| :--- | :--- | :--- |
| `split_left` / `split_right` | Cột chữ full chiều cao bên trái/phải, sản phẩm phía còn lại | Đa dụng, sản phẩm |
| `sandwich_top_heavy` / `sandwich_bottom_heavy` | Dải chữ đỉnh + dải đáy | Khuyến mãi, lễ hội |
| `diagonal_slash` | Cột chữ cắt chéo năng động | Sale, thể thao, công nghệ |
| `l_frame_showcase` | Khung chữ L ôm góc | Sản phẩm cao cấp |
| `lifestyle_corner_pod` | Hộp chữ nhỏ bo góc | Nước hoa, lifestyle |
| `luxury_centered_card` | Thẻ chữ ở giữa | Thiệp mời, sang trọng |
| `grand_opening_banner` | Băng rôn khai trương | Khai trương, sự kiện |
| `type_showcase` | Chữ làm nhân vật chính, con số khổng lồ giữa khung | Sale thuần chữ, chúc mừng |
| `notice_card` | **Tấm thiệp giấy** đặc (không mask), 16:9 hai cột | Thông báo, thiệp chúc mừng, thư mời |
| `menu_price_board` | Bảng giá "tên món ..... giá" | Menu, combo |
| `step_process_roadmap` | Các bước đánh số | Hướng dẫn, quy trình |
| `recruitment_board` | Bảng tin 2 cột | Tuyển dụng |
| `customer_feedback_card` | Thẻ đánh giá + ưu đãi | Review có kèm ưu đãi |
| `quote_spotlight` | **Câu trích dẫn là chữ to nhất** | Đánh giá khách hàng |
| `before_after_split` | Hai nửa trước / sau | Giảm cân, dọn dẹp, spa |

Mẫu **chuyên biệt** (menu, tuyển dụng, các bước, đánh giá, trước/sau, thiệp) không bị Cổng 3 đổi sang mẫu khác.

**12 nhu cầu** trên trang nghiệm thu: khuyến mãi · giới thiệu sản phẩm · khai trương · đánh giá · tuyển dụng ·
hướng dẫn · thông báo · thiệp chúc mừng · thư mời · trước/sau · minigame · combo.

---

## 4. CÁC "VIÊN GẠCH" TẠO NÊN THẨM MỸ

### 4.1. Font (28 font tiếng Việt, `fonts.py`)
- Đủ 134 ký tự có dấu; nhúng sẵn vào poster; độ đậm thật (không tô đậm giả).
- **Ghép font theo thực hành designer**: font cá tính chỉ cho điểm neo (tiêu đề/con số); chữ phụ, danh sách, các bước
  luôn dùng font dễ đọc (Be Vietnam Pro). Font viết tay chỉ cho tiêu đề ≤ 5 từ và **không bao giờ viết hoa toàn bộ**
  (tiêu đề IN HOA → tự đổi sang font in).
- **9 font mới (27/09 tối)** tải từ kho Google Fonts, giấy phép OFL (được dùng thương mại): 6 font thư pháp/viết tay
  (Great Vibes, Alex Brush, Lobster, Charm, Sriracha, Pattaya) + 3 font nhiều độ đậm (Montserrat, Cormorant, Barlow
  Condensed). Bộ phong cách spa/sang trọng dùng Cormorant + dòng viết tay Alex Brush/Great Vibes; tuyển dụng dùng Montserrat.
- 13 font SVN-* cũ **chưa rõ giấy phép thương mại** — cần người hỏi nhà phát hành.
- **Khoảng cách dòng tối thiểu theo font** (đo bằng điểm ảnh): 4 font để dấu tiếng Việt dòng dưới đè lên dòng trên ở
  khoảng cách hẹp (anton, gretoon, oswald, pacifico, greatvibes, charm) → được giữ khoảng cách lớn hơn.

### 4.2. Kiểu cụm tiêu đề — lockup (5 kiểu, rút từ bộ poster tham chiếu)
| Kiểu | Trông như | Cần |
| :--- | :--- | :--- |
| `stat_stack` | chữ dẫn nhỏ / **CON SỐ khổng lồ** (% nhấc lên) / chữ đuôi | con số |
| `script_over_caps` | *chữ viết tay nghiêng* đè lên **CHỮ IN HOA** | cụm cảm xúc ngắn + cụm chính |
| `band` | **CỤM MÓC** + mô tả nằm trong dải màu | cụm móc + đuôi ngắn |
| `bracket_title` | tiêu đề giữa 4 góc ngoặc khung | tiêu đề trang trọng |
| `stat_seal` | con số ngắn trong vòng tròn + dải băng | sale, lễ hội |

LLM bỏ trống → Python tự đề xuất. Autofit tự quay về xếp ngang nếu kiểu đặc biệt làm con số nhỏ đi.

### 4.3. Ý đồ thị giác (intent) — quyết định mức "nổi" cần đạt
| Intent | Tiêu đề phải to gấp ≥ | Ví dụ |
| :--- | :-- | :--- |
| `big_number_deal` | 3.0 × chữ phụ | Giảm 50%, đồng giá 99K |
| `hook_headline` | 2.8 × | Cụm từ móc |
| `festive_event` | 2.8 × | Tết, khai trương |
| `product_showcase` | 2.5 × | Giới thiệu sản phẩm |
| `testimonial_trust` | 2.4 × | Đánh giá |
| `matrix_board` | 2.0 × | Menu, các bước, tuyển dụng (nhiều thông tin) |

Mỗi linh kiện/hiệu ứng khai báo intent được dùng (tránh "chữ neon trên thiệp mời VIP").

### 4.4. Linh kiện đồ hoạ và hiệu ứng
- Nhãn (`badge_style`): viên thuốc, dải băng, viên nang 2 màu, con dấu tròn chữ chạy vòng.
- Con số (`stat_style`): tách đơn vị nhỏ (50 + %), sao nổ phía sau.
- Trang trí (`decor`): hạt lấp lánh quanh con số.
- Hiệu ứng chữ: đổ bóng, vàng 3D, chrome, neon, lửa, hologram, kim loại nổi, gel bóng… (hiệu ứng nổi khối chỉ cho font đậm).

### 4.5. Bộ phong cách theo dịp + nhận diện thương hiệu (`style_packs.py`)
- **8 bộ**: Tết, Trung Thu, sale đồ ăn, cà phê, spa/làm đẹp, công nghệ, sang trọng, tuyển dụng. Mỗi bộ phối sẵn font,
  màu, hiệu ứng, linh kiện, **hoạ tiết** (cành mai, đèn lồng, nhánh lá, lưới chấm, khung góc, vạch nhãn — SVG tự vẽ).
- Hoạ tiết đặt **sau** khi chữ đã chốt, chỉ vào chỗ trống, không chạm chữ/khối/logo; cùng nội dung → cùng kết quả.
- **Brand kit** (màu, font, logo của người dùng) **ghi đè** bộ phong cách; logo tự tìm góc trống.

---

## 5. BỐN CỔNG KIỂM SOÁT

| Cổng | Kiểm gì | Nếu sai |
| :--- | :--- | :--- |
| **1 — Nguyên văn** | `hero_parts` ghép lại phải đúng từng ký tự tiêu đề | Bỏ phần cắt vai trò, tiêu đề về dạng thường (không bao giờ sai chữ) |
| **2 — Hợp lệ** (`validators.py`) | Mẫu/intent/font/hiệu ứng/linh kiện/bộ phong cách có trong danh mục và hợp nhau; số từ (tiêu đề ≤ 7, phụ đề ≤ 10…); trường bắt buộc; bỏ chữ lặp | Cảnh báo + dùng giá trị an toàn |
| **3 — Sức chứa** (`routing.py`) | Số ký tự ≤ sức chứa **đã đo** của mẫu ở khổ đó; mẫu có chỗ cho mọi trường; poster thuần chữ → `type_showcase` | Đổi sang mẫu cùng ý đồ chứa nổi |
| **4 — Tràn** (trong autofit) | Chữ bị cắt / đè / ra ngoài khung / ngoài khung cha | Tự co chữ trong vùng; còn thì báo "mất chữ" |

---

## 6. SÁU LUẬT THẨM MỸ VÀ 5 ĐIỀU KIỆN NGHIỆM THU TỰ ĐỘNG

**Luật** (chi tiết ROADMAP §4): (1) thang cỡ chữ kiểu poster, không phải kiểu văn bản · (2) tương phản trong dòng qua
`hero_parts` · (3) màu 60-30-10 và tương phản nền đo thật · (4) danh mục hiệu ứng đóng, gắn ý đồ · (5) "nheo mắt
nhìn" tự động (squint test) · (6) đọc được trên điện thoại.

**6 điều kiện** chấm mọi poster (`scripts/probe_type_hierarchy.py`):

| # | Điều kiện | Nghĩa dễ hiểu |
| :-- | :--- | :--- |
| C1 | Điểm neo | Tiêu đề to gấp đủ lần chữ phụ (theo intent) |
| C2 | Không tường chữ | Không có ≥ 3 phần tử khác cấp to gần bằng nhau |
| C3 | Tương phản nền | Mọi dòng chữ đạt chuẩn WCAG ở ô nền xấu nhất (đo trên ảnh chụp thật) |
| C4 | Không mất chữ | Không chữ nào bị cắt/đè |
| C5 | Đọc được trên điện thoại | Chữ ≥ 10px trên màn 375px, tiêu đề ≥ 20px |
| C6 | Không phí chỗ | Tiêu đề / nội dung chính không bị trần cỡ chữ giữ nhỏ khi còn ≥ 40% chỗ (thêm 27/09 — bắt lỗi "chữ nhỏ giữa khoảng trống" ở khung dọc mà C1–C5 bỏ sót) |

---

## 7. TÍNH NĂNG DÙNG THẬT

- **Nhiều kiểu chữ trên một nền** (`variants.py`, `scripts/render_variants.py`): giữ mẫu + vùng chữ ⇒ không cần vẽ lại
  nền (không tốn GPU), đổi bộ phong cách / kiểu cụm tiêu đề.
- **Sửa chữ sau khi sinh** (`POST /api/v3/retext` trên máy chủ demo): sửa tiêu đề, nút… hoặc xin thêm kiểu chữ; thay
  đổi làm đổi vùng chữ bị từ chối (vì nền đã vẽ theo vùng cũ).
- **Xuất in** (`renderer.export_print`): A4 300dpi, bố cục y hệt bản màn hình.
- **Logo, màu, font thương hiệu**; **QR** tự sinh từ link.
- **Máy chủ demo** (`demo_server.py`): giao diện web + API; chạy chế độ giả lập khi không có GPU.

---

## 8. KIỂM THỬ VÀ SỐ ĐO

**Tự động (chạy mỗi lần sửa):** 476 test (`pytest tests/`), gồm:
- 4 **"bánh cóc"** chất lượng — số poster đạt từng điều kiện **không được giảm** so với mốc: bộ chính (491 poster),
  bộ "LLM lý tưởng" (80), nền khắc nghiệt (123, có vệt sáng/mảng tối), nền không mask (18).
- Kiểm dấu tiếng Việt 28 font, hoạ tiết không chạm chữ, logo, bản in, sửa chữ, hợp đồng LLM, sức chứa, mask ≤ 50%…

**Số đo bộ chính (491 poster, 27/09):**

| C1 neo | C2 không tường | C3 tương phản | C4 không mất chữ | Đạt 4 | C5 điện thoại | **Đạt cả 5** | C6 không phí chỗ | **Đạt cả 6** |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 218 | 428 | 485 | 485 | 217 | 268 | **182** | 486 | **179** |

**Với LLM + ảnh nền thật** (`scripts/run_real_llm.py`, 26 đề bài, lượt v8): **21/26 đạt cả 5**, 0 mất chữ. Trang xem:
`output_probe/nghiem_thu_v8/index.html` (gom theo 12 nhu cầu).

**Bộ poster tham chiếu**: 135 mẫu thiết kế chuyên nghiệp (Canva, 9 nhóm) — `references/posters/index.html`, chỉ lưu
máy local để so sánh, không đẩy lên GitHub.

---

## 9. BẢN ĐỒ MÃ NGUỒN

```
src/
├── flux2/                 Mã gốc FLUX.2 (Black Forest Labs) — ĐÓNG BĂNG, không sửa
└── tendoo_v3/
    ├── catalog.py          Danh mục 17 mẫu, intent, linh kiện, giới hạn — NGUỒN SỰ THẬT DUY NHẤT (prompt LLM sinh từ đây)
    ├── schema.py           Cấu trúc bản kế hoạch (plan)
    ├── llm_planner.py      Gọi LLM (API / Qwen local / dự phòng) + Cổng 1-3
    ├── llm_prompts.py      Hướng dẫn cho LLM
    ├── validators.py       Cổng 2, bỏ chữ lặp
    ├── routing.py          Cổng 3 — đổi mẫu theo sức chứa
    ├── hero_markup.py      Cắt tiêu đề "LLM lý tưởng", nối từ không ngắt dòng
    ├── geometry.py         Vùng chữ từng mẫu × khổ
    ├── mask_engine.py      Vẽ mask
    ├── velocity_blending.py Diffusion 2 luồng (GPU)
    ├── renderer.py         Ngân sách cỡ chữ, bảng màu, dựng HTML, chụp, xuất in
    ├── styles.py           Hiệu ứng chữ, hằng số Luật 6, script autofit + tương phản
    ├── components.py       Linh kiện, lockup
    ├── style_packs.py      Bộ phong cách, hoạ tiết, logo
    ├── variants.py         Nhiều kiểu chữ trên một nền
    ├── fonts.py / colors.py / palette.py   Font, màu, WCAG
    ├── poster_renderer.py  Điều khiển Chromium
    ├── demo_server.py      Máy chủ web + API
    └── templates/          17 mẫu HTML + components/ (CSS, macro dùng chung)
scripts/                    Đo, chấm, chạy GPT thật, trang nghiệm thu, tải poster tham chiếu, biến thể
tests/                      440 test + 38 bộ dữ liệu poster mẫu + 4 mốc chất lượng
```

---

## 10. CÁCH CHẠY

```bash
# Cài đặt (máy local, không cần GPU cho tầng chữ)
pip install -r requirements.txt && python -m playwright install chromium

# Toàn bộ test
PYTHONPATH=src python -m pytest tests/ -q

# Chấm 5 điều kiện trên bộ poster mẫu
PYTHONPATH=src python scripts/probe_type_hierarchy.py --bg

# Chạy GPT thật (khoá trong .env) + ảnh nền thật, rồi dựng trang nghiệm thu
PYTHONPATH=src python scripts/run_real_llm.py --model gpt-5.4-mini --images --tag v9
PYTHONPATH=src python scripts/build_acceptance_report.py --tag gpt-5.4-mini_v9

# Nhiều kiểu chữ + logo + bản in cho một poster
PYTHONPATH=src python scripts/render_variants.py --tag gpt-5.4-mini_v8 --id b01_promo_coffee_1x1 --n 4 --print

# Máy chủ demo (GPU cho diffusion thật; không GPU = giả lập)
PYTHONPATH=src python src/tendoo_v3/demo_server.py --model distill
```

---

## 11. GIỚI HẠN ĐÃ BIẾT VÀ VIỆC CÒN LẠI

**Giới hạn đo được**
- **Khung ngang 16:9** là khó nhất: banner ngang xem trên điện thoại buộc mọi chữ phải to, chữ phụ lấn chỗ tiêu đề →
  một số poster tiêu đề chưa đủ nổi (C1). Đã thử nới cột chữ (sneaker) và dời đoạn nội dung sang cột phải (thiệp) — cả
  hai gây hồi quy nên hoàn tác. Ở khung 1024×576 mọi chữ phụ đã ở cỡ tối thiểu: đây là giới hạn **lượng chữ**, không phải bố cục.
- Tấm thiệp 16:9 với đoạn nội dung dài: chữ về gần sàn.
- 6 poster mẫu nội dung cực dày (chủ yếu 16:9) vẫn mất chữ ở mức chữ nhỏ nhất cho phép.
- Máy đo **không chấm được cái đẹp** — chỉ chấm được "đọc được, đúng thứ bậc". Mắt người vẫn là thước đo cuối.
- Ảnh nền hiện do GPT vẽ; GPT không luôn chừa đúng vùng chữ như FLUX + mask thật.

**Việc còn lại**
| Việc | Cần |
| :--- | :--- |
| Nghiệm thu với nền **FLUX thật**, đo tốc độ chế độ không mask, chạy Qwen trên máy chủ | GPU (máy chủ 2×A30) |
| **Chấm mù** poster hệ thống vs poster designer: công cụ đã sẵn — mở `references/blind_review/index.html`, chấm 26 cặp, tải phiếu, chạy `scripts/score_blind_review.py` | Người chấm (3–5 người) |
| Kiểm giấy phép 13 font SVN-* (nếu không được phép → thay bằng font OFL) | Người liên hệ nhà phát hành |
| Khung 16:9: hạ sức chứa (ít dòng phụ hơn) hoặc khuyên khung dọc khi nội dung dày | Máy local |

---

## 12. THUẬT NGỮ

| Từ | Nghĩa |
| :--- | :--- |
| Template / mẫu | Bố cục cố định (vị trí các khối chữ) |
| Plan / bản kế hoạch | Kết quả LLM: mẫu + nội dung + phong cách + mô tả cảnh |
| Mask | Ảnh đen trắng đánh dấu vùng sẽ đặt chữ để diffusion vẽ dịu đi |
| Diffusion / FLUX | Mô hình AI vẽ ảnh nền |
| Autofit | Script tự co giãn cỡ chữ trong trình duyệt |
| hero / stat / prefix / suffix | Tiêu đề / điểm neo to nhất / chữ dẫn / chữ đuôi |
| Lockup | Kiểu sắp đặt cụm tiêu đề |
| Intent | Ý đồ thị giác, quyết định mức nổi cần đạt |
| Squint test | "Nheo mắt nhìn" — chấm tự động 5 điều kiện |
| WCAG | Chuẩn quốc tế về độ tương phản chữ–nền |
| Maskless | Chế độ không mask cho poster thuần chữ |
| Style pack / brand kit | Bộ phong cách theo dịp / nhận diện thương hiệu của khách |
| px trên màn | Kích thước chữ khi poster hiển thị vừa bề ngang điện thoại 375px |
