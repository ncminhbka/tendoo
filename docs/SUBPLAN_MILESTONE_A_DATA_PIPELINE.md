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
* **Quy mô**: Đúng **120 mẫu ($15\%$ tổng 800 mẫu)**, chia đều cân bằng:
  - **60 mẫu nhánh SP-Anchor**: Dùng ảnh Packshot của 10 sản phẩm đóng gói / chế tác thương mại Việt Nam (Mì Hảo Hảo, Phin cafe, Trà sen, SIM 5G Viettel, Giày Biti's).
  - **60 mẫu nhánh Pure T2I**: Dành cho ẩm thực hữu cơ tự nhiên & bối cảnh không gian sống (Tô phở Thìn bốc khói, Bánh mì kẹp thịt giòn rụm, Áo dài chợ hoa, Phố cổ Hội An).
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

## 📦 3. DANH MỤC 50 SẢN PHẨM PACKSHOT TUYỂN CHỌN (8 THƯ MỤC DOMAIN) & BỘ LỌC NGỮ NGHĨA

Người dùng sẽ trực tiếp tải/thu thập 50 ảnh sản phẩm chất lượng cao (nền trắng hoặc trong suốt, tỉ lệ 1:1, tối thiểu $512\times 512\text{px}$) thả vào các thư mục tương ứng trong `data/products/`:

### 3.1. Thư mục `data/products/cosmetics/` (8 sản phẩm)
| Tên File | Đối Tượng Sản Phẩm (Gợi ý tìm kiếm) | Chủ Đề Text Bắt Buộc Khớp Ngữ Nghĩa |
| :--- | :--- | :--- |
| `01_nuoc_hoa_luxury.png` | Chai nước hoa thủy tinh cao cấp (Dior, Chanel...) | *Hương thơm quyến rũ / Lưu hương 24h / Nốt hương quý phái* |
| `02_serum_duong_am.png` | Lọ serum nhỏ giọt thủy tinh (Estee Lauder, The Ordinary) | *Căng bóng mịn màng / Cấp ẩm chuyên sâu / Tái tạo làn da* |
| `03_kem_duong_da.png` | Hộp kem dưỡng da mặt hình tròn | *Dưỡng trắng tự nhiên / Mờ thâm nám / Trẻ hóa làn da* |
| `04_son_moi_matte.png` | Thỏi son môi lì vặn nắp vuông/tròn (MAC, YSL) | *Sắc đỏ thời thượng / Lâu trôi mịn lì / Chuẩn màu tôn da* |
| `05_kem_chong_nang.png` | Tuýp kem chống nắng đứng | *Bảo vệ tối ưu SPF50+ / Kháng nước kiềm dầu / Chống tia UV* |
| `06_sua_rua_mat.png` | Tuýp sữa rửa mặt tạo bọt | *Sạch sâu dịu nhẹ / Cân bằng độ ẩm / Ngừa mụn kiềm dầu* |
| `07_phan_nuoc_cushion.png` | Hộp phấn nước cushion tròn mở nắp nhẹ | *Lớp nền mỏng mịn / Che phủ khuyết điểm / Tự nhiên suốt ngày dài* |
| `08_dau_goi_dau.png` | Chai dầu gội đầu vòi nhấn sang trọng | *Bồng bềnh suôn mượt / Ngăn rụng tóc / Tinh dầu tự nhiên* |

### 3.2. Thư mục `data/products/fnb/` (7 sản phẩm)
| Tên File | Đối Tượng Sản Phẩm (Gợi ý tìm kiếm) | Chủ Đề Text Bắt Buộc Khớp Ngữ Nghĩa |
| :--- | :--- | :--- |
| `09_phin_cafe_nhom.png` 🇻🇳 | Bộ phin nhôm pha cà phê truyền thống Việt Nam | *Cà phê nguyên chất / Đậm đà phong vị / Hương thơm truyền thống* |
| `10_tui_cafe_rang_moc.png` 🇻🇳 | Túi giấy kraft cà phê hạt rang mộc (Robusta/Arabica) | *Hạt Robusta mộc / Rang xay thủ công / Vị đắng thanh khiết* |
| `11_lon_nuoc_tang_luc.png` | Lon nhôm nước tăng lực (Red Bull, Monster) | *Bật tung năng lượng / Sảng khoái tức thì / Tỉnh táo tập trung* |
| `12_chai_tra_xanh.png` | Chai nhựa trà xanh thanh mát đóng chai | *Thanh lọc cơ thể / Vị trà thanh mát / Chiết xuất lá trà tươi* |
| `13_hop_sua_hat.png` | Hộp giấy sữa hạt dinh dưỡng (óc chó, hạnh nhân) | *Dinh dưỡng thuần khiết / Giàu canxi tự nhiên / Thanh nhẹ lành mạnh* |
| `14_lon_bia_craft.png` | Lon bia thủ công in họa tiết nghệ thuật | *Men bia ủ mộc / Hương hoa bia sảng khoái / Đỉnh cao hương vị* |
| `15_chai_ruou_vang.png` | Chai rượu vang thủy tinh cổ cao sang trọng | *Ủ thùng gỗ sồi / Nồng nàn quyến rũ / Đẳng cấp tiệc sang* |

### 3.3. Thư mục `data/products/tech/` (7 sản phẩm)
| Tên File | Đối Tượng Sản Phẩm (Gợi ý tìm kiếm) | Chủ Đề Text Bắt Buộc Khớp Ngữ Nghĩa |
| :--- | :--- | :--- |
| `16_tai_nghe_tws.png` | Hộp tai nghe không dây TWS (AirPods Pro, Sony) | *Chống ồn chủ động / Âm bass sống động / Pin 30 giờ liên tục* |
| `17_smartwatch.png` | Đồng hồ thông minh màn hình AMOLED (Apple Watch...) | *Theo dõi sức khỏe / Kháng nước 5ATM / Trợ lý thông minh* |
| `18_loa_bluetooth.png` | Loa bluetooth di động (JBL, Marshall) | *Âm thanh vòm 360 / Âm bass cực căng / Tiệc tùng thả ga* |
| `19_chuot_gaming.png` | Chuột máy tính không dây gaming công thái học | *Độ nhạy siêu cao / Cảm biến quang học / Thiết kế công thái học* |
| `20_ban_phim_co.png` | Bàn phím cơ không dây layout gọn gàng | *Switch êm ái / Đèn nền RGB / Cảm giác gõ đỉnh cao* |
| `21_sac_du_phong.png` | Cục sạc dự phòng nhỏ gọn hiện đại | *Sạc nhanh 65W / Dung lượng 20000mAh / Nhỏ gọn tiện lợi* |
| `22_tay_cam_game.png` | Tay cầm chơi game không dây (PlayStation/Xbox) | *Rung phản hồi chân thực / Không độ trễ / Tương thích đa nền tảng* |

### 3.4. Thư mục `data/products/fashion/` (6 sản phẩm)
| Tên File | Đối Tượng Sản Phẩm (Gợi ý tìm kiếm) | Chủ Đề Text Bắt Buộc Khớp Ngữ Nghĩa |
| :--- | :--- | :--- |
| `23_giay_sneaker_bitis.png` 🇻🇳 | Giày thể thao năng động *(Đã có: `images/shoes.jpeg`)* | *Siêu nhẹ êm chân / Bước đi tự tin / Thiết kế phá cách* |
| `24_kinh_mat_thoi_trang.png` | Kính râm thời trang gọng kim loại | *Chống tia UV400 / Thời thượng cá tính / Tôn vinh góc mặt* |
| `25_dong_ho_kim_loai.png` | Đồng hồ đeo tay dây kim loại mạ bạc/vàng | *Sang trọng lịch lãm / Bộ máy chuẩn xác / Đẳng cấp quý ông* |
| `26_tui_xach_da.png` | Túi xách nữ chất liệu da cao cấp | *Chất da cao cấp / Tinh tế từng đường kim / Phong cách thanh lịch* |
| `27_vi_da_nam.png` | Ví da nam gập đôi cầm tay cao cấp | *Gọn gàng tiện dụng / Da bò nguyên tấm / Bền đẹp cùng thời gian* |
| `28_non_la_viet_nam.png` 🇻🇳 | Chiếc nón lá truyền thống chụp packshot | *Nét đẹp truyền thống / Hồn quê đất Việt / Duyên dáng thanh tao* |

### 3.5. Thư mục `data/products/home/` (6 sản phẩm)
| Tên File | Đối Tượng Sản Phẩm (Gợi ý tìm kiếm) | Chủ Đề Text Bắt Buộc Khớp Ngữ Nghĩa |
| :--- | :--- | :--- |
| `29_binh_giu_nhiet.png` | Bình giữ nhiệt inox nắp gỗ tối giản (Lock&Lock) | *Giữ nhiệt 24 giờ / Thép không gỉ 304 / Thiết kế tối giản* |
| `30_may_say_toc.png` | Máy sấy tóc tạo ion âm hiện đại (Dyson style) | *Sấy khô siêu tốc / Chăm sóc tóc bóng mượt / Nhẹ nhàng êm ái* |
| `31_ban_ui_hoi_nuoc.png` | Bàn ủi hơi nước cầm tay mini | *Phẳng phiu tức thì / Diệt khuẩn 99% / Nhỏ gọn du lịch* |
| `32_may_xay_sinh_to.png` | Máy xay sinh tố mini cầm tay sạc pin | *Xay nhuyễn mịn / Tiện lợi mọi nơi / Sống khỏe mỗi ngày* |
| `33_noi_chien_khong_dau.png` | Nồi chiên không dầu điện tử | *Giảm 85% dầu mỡ / Chín đều giòn rụm / Nấu nướng thảnh thơi* |
| `34_den_ban_led.png` | Đèn bàn học/làm việc LED chống cận thông minh | *Ánh sáng bảo vệ mắt / Tùy chỉnh 3 màu / Cảm ứng thông minh* |

### 3.6. Thư mục `data/products/fmcg/` (6 sản phẩm)
| Tên File | Đối Tượng Sản Phẩm (Gợi ý tìm kiếm) | Chủ Đề Text Bắt Buộc Khớp Ngữ Nghĩa |
| :--- | :--- | :--- |
| `35_mi_hao_hao.png` 🇻🇳 | Gói mì Hảo Hảo Tôm chua cay *(Đã có: `images/hao_hao.jpg`)* | *Vị tôm chua cay / Sợi mì dai giòn / Hương vị quốc dân* |
| `36_hop_tra_sen_tay_ho.png` 🇻🇳 | Hộp thiếc trà ướp hoa sen Tây Hồ | *Hương sen thanh khiết / Trà búp tân cương / Tinh hoa ẩm thực Việt* |
| `37_chai_nuoc_mam_phu_quoc.png` 🇻🇳 | Chai thủy tinh nước mắm truyền thống | *Đậm đà cốt cá cơm / Ủ chượp truyền thống / Vị ngon nguyên bản* |
| `38_hop_cao_sao_vang.png` 🇻🇳 | Hộp thiếc tròn đỏ cao sao vàng cổ điển | *Liệu pháp dân gian / Tinh dầu tràm quế / Thương hiệu vượt thời gian* |
| `39_hu_yen_sao_khanh_hoa.png` 🇻🇳 | Hũ thủy tinh yến sào chưng sẵn | *Yến đảo nguyên chất / Bồi bổ sức khỏe / Quà tặng sức khỏe* |
| `40_hop_banh_quy_bo.png` | Hộp thiếc bánh quy bơ phong cách châu Âu (Danisa) | *Thơm lừng bơ sữa / Giòn tan trong miệng / Món ngon sum vầy* |

### 3.7. Thư mục `data/products/telecom_viettel/` (5 sản phẩm)
| Tên File | Đối Tượng Sản Phẩm (Gợi ý tìm kiếm) | Chủ Đề Text Bắt Buộc Khớp Ngữ Nghĩa |
| :--- | :--- | :--- |
| `41_modem_wifi6_viettel.png` 🇻🇳 | Thiết bị Modem Home Wifi 6 màu trắng của Viettel | *Phủ sóng toàn diện / Tốc độ gigabit / Mượt mà không giật lag* |
| `42_phoi_sim_5g_viettel.png` 🇻🇳 | Phôi thẻ SIM 5G Viettel màu đỏ trắng | *Tốc độ vượt trội / Kết nối tương lai / Phủ sóng toàn quốc* |
| `43_smart_camera_viettel.png` 🇻🇳 | Thiết bị Camera an ninh thông minh Viettel Home | *Hình ảnh 2K sắc nét / Đàm thoại 2 chiều / Lưu trữ an toàn* |
| `44_thiet_bi_v_tracking.png` 🇻🇳 | Thiết bị định vị giám sát hành trình Viettel | *Định vị GPS chính xác / Giám sát 24/7 / Quản lý phương tiện hiệu quả* |
| `45_hop_tv360_box.png` 🇻🇳 | Thiết bị đầu thu Android TV Box TV360 Viettel | *Thế giới giải trí đỉnh cao / Hàng trăm kênh HD / Xem mượt mọi lúc* |

### 3.8. Thư mục `data/products/fitness/` (5 sản phẩm)
| Tên File | Đối Tượng Sản Phẩm (Gợi ý tìm kiếm) | Chủ Đề Text Bắt Buộc Khớp Ngữ Nghĩa |
| :--- | :--- | :--- |
| `46_binh_lac_shaker.png` | Bình lắc thể thao tập gym đựng Whey Protein | *Khuấy tan siêu nhanh / Nhựa an toàn BPA-Free / Đồng hành cùng gymer* |
| `47_tham_yoga.png` | Cuộn thảm tập yoga TPE cao cấp | *Chống trơn trượt / Đàn hồi êm ái / Tự tin trong từng động tác* |
| `48_gang_tay_gym.png` | Đôi găng tay thể thao tập tạ chống chai tay | *Bảo vệ cổ tay / Thoáng khí êm ái / Tăng cường lực kéo* |
| `49_con_lan_massage.png` | Con lăn bọt xốp Foam Roller giãn cơ sau tập | *Giảm đau mỏi cơ / Phục hồi chấn thương / Thư giãn sau tập* |
| `50_day_nhay_toc_do.png` | Dây nhảy thể dục lõi cáp tốc độ cao | *Đốt cháy calo / Ổ bi xoay mượt mà / Rèn luyện sức bền* |

---

## 🔄 4. QUY TRÌNH CHẾ TẠO DỮ LIỆU TIỀN ĐỊNH (PRE-DETERMINED TYPOGRAPHY PIPELINE)

```
[ BƯỚC 1: SPECIFICATION SAMPLER ]
  Pipeline chọn ngẫu nhiên bộ tham số cho mẫu:
  • Use-Case (1/7), Domain (1/8), Aspect Ratio (1/4), Length Stratum (75/25).
  • Nếu là I2I: Bốc ảnh sản phẩm từ đúng thư mục `data/products/<domain>/` tương ứng.
  • Cặp Text: Lấy text được chỉ định riêng cho sản phẩm đó (tránh cọc cạch ngữ nghĩa).
  • Cặp Font: Bốc ngẫu nhiên trực giao từ 16 fonts (sàn 32pt cho BeVietnam, 36pt cho font khác).
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
[ BƯỚC 4: SINH STUDENT CLEAN PROMPT (PYTHON RULE-BASED ENGINE) ]
  • Ghép câu bằng Python Combinatorial Builder với từ điển thuộc tính đa dạng:
    "(1) Dòng chữ vị trí tuyển dụng ở phía trên dập nổi mạ vàng sang trọng.
     (2) Khối thông tin đãi ngộ 2 dòng ở bên dưới nét chữ màu trắng thanh mảnh rõ nét."
  • Chạy hàm `validate_prompt_clean()`:
    - Bắt buộc có thẻ định danh `(1)` và `(2)`.
    - Khẳng định 100% KHÔNG chứa chuỗi chữ nguyên văn (Anti-leak test).
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
└── products/                   # Thư mục gốc chứa 50 ảnh sản phẩm thật tuyển chọn
    ├── cosmetics/
    ├── fnb/
    ├── tech/
    ├── fashion/
    ├── home/
    ├── fmcg/
    ├── telecom_viettel/
    └── fitness/
```
