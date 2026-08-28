# -*- coding: utf-8 -*-
"""
====================================================================================================
TENDOO AI - HIGH-DIVERSITY MULTI-ANGLE TEXT CORPUS (MILESTONE A: 800 SAMPLES)
====================================================================================================
This corpus delivers 350+ unique, professional Vietnamese advertising copy pairs across:
1. 50 Packshot Products (8 domains) with 5 content angles each:
   - Angle 1: Product Identity / Brand Headline
   - Angle 2: Flash Sale / Hot Promotion
   - Angle 3: Lifestyle / Emotional Benefit
   - Angle 4: Inverted Stratum A (Long Title 8-14 words @ Slot 1, Short Badge 1-3 words @ Slot 2)
   - Angle 5: Inverted Stratum B (Narrative/Quality @ Slot 1, Urgency Badge @ Slot 2)
2. 8 General T2I Use Cases (Flash Sale, Product Ad, Feedback, Recruitment, Guide, Quote, Banner, Cultural Vietnam)
   divided into standard and inverted length strata (Golden 75/25 Ratio).
3. Known-Hard Concurrency Stress Test Pairs.
====================================================================================================
"""

from typing import Dict, List, Tuple

# ==================================================================================================
# 1. 50 PRODUCTS MULTI-ANGLE TEXT CORPUS
# ==================================================================================================
PRODUCT_TEXT_CORPUS: Dict[str, Dict[str, Dict[str, List[Tuple[str, str]]]]] = {
    # ----------------------------------------------------------------------------------------------
    # COSMETICS (8 products)
    # ----------------------------------------------------------------------------------------------
    "cosmetics": {
        "01_nuoc_hoa_luxury": {
            "standard": [
                ("NƯỚC HOA CAO CẤP", "Hương thơm quý phái\nLưu hương 24 giờ"),
                ("TINH HOA NƯỚC PHÁP", "Nốt hương quyến rũ nồng nàn\nTôn vinh vẻ đẹp kiêu sa"),
                ("ĐẠI TIỆC MÙI HƯƠNG", "Ưu đãi 40% bộ sưu tập mới\nTặng kèm hộp quà sang trọng"),
            ],
            "inverted": [
                ("MÙI HƯƠNG THANH LỊCH ĐÁNH THỨC MỌI GIÁC QUAN\nKHẲNG ĐỊNH ĐẲNG CẤP KHÁC BIỆT", "GIẢM 50%"),
                ("NỐT HƯƠNG QUÝ PHÁI ĐỒNG HÀNH TRONG TỪNG KHOẢNH KHẮC\nTÔN VINH VẺ ĐẸP KIÊU SA", "HOT DEAL"),
            ],
        },
        "02_serum_duong_am": {
            "standard": [
                ("SERUM DƯỠNG ẨM", "Căng bóng mịn màng\nCấp ẩm chuyên sâu"),
                ("TINH CHẤT PHỤC HỒI", "Tái tạo tế bào da\nNgăn ngừa lão hóa sớm"),
                ("LÀN DA TƯƠI TRẺ", "Thẩm thấu tức thì vào sâu tế bào\nCải thiện độ đàn hồi tự nhiên"),
            ],
            "inverted": [
                ("BÍ QUYẾT LÀN DA CĂNG BÓNG MỊN MÀNG TỰ NHIÊN\nPHỤC HỒI TỔN THƯƠNG TỪ SÂU BÊN TRONG", "MUA 1 TẶNG 1"),
                ("CẤP ẨM CHUYÊN SÂU CHO LÀN DA RẠNG RỠ MỖI NGÀY\nCHỐNG LẠI DẤU HIỆU THỜI GIAN", "TIẾT KIỆM 40%"),
            ],
        },
        "03_kem_duong_da": {
            "standard": [
                ("KEM DƯỠNG TRẮNG", "Trẻ hóa làn da\nMờ thâm nám tự nhiên"),
                ("NUÔI DƯỠNG CHUYÊN SÂU", "Chiết xuất thảo dược thiên nhiên\nBảo vệ hàng rào độ ẩm"),
                ("RẠNG RỠ MỖI SỚM MAI", "Chăm sóc làn da ban đêm\nThức giấc với vẻ đẹp rạng ngời"),
            ],
            "inverted": [
                ("GIẢI PHÁP TRẺ HÓA LÀN DA TOÀN DIỆN CHO PHÁI ĐẸP\nXÓA MỜ THÂM NÁM VÀ LÃO HÓA", "GIẢM 35%"),
                ("NUÔI DƯỠNG LÀN DA TRẮNG MỊN MÀNG KHÔNG TỲ VẾT\nAN TOÀN CHO MỌI LOẠI DA", "HOT DEAL"),
            ],
        },
        "04_son_moi_matte": {
            "standard": [
                ("SON MÔI MỊN LÌ", "Sắc đỏ thời thượng\nChuẩn màu lâu trôi"),
                ("SẮC ĐỎ NỒNG NÀN", "Tôn vinh bờ môi quyến rũ\nChất son nhung mịn không khô môi"),
                ("BỘ SƯU TẬP MÙA HÈ", "Tự tin tỏa sáng mọi ánh nhìn\nBảng màu đa dạng phong cách"),
            ],
            "inverted": [
                ("CHẤT SON MỊN LÌ NHƯ NHUNG TÔN VINH VẺ ĐẸP ĐỘC BẢN\nCHUẨN MÀU SUỐT CẢ NGÀY DÀI", "ƯU ĐÃI 50%"),
                ("BÙNG NỔ NĂNG LƯỢNG VỚI SẮC ĐỎ THỜI THƯỢNG\nCHẠM ĐẾN TRÁI TIM ĐỐI PHƯƠNG", "GIÁ SỐC"),
            ],
        },
        "05_kem_chong_nang": {
            "standard": [
                ("KEM CHỐNG NẮNG", "Bảo vệ tối ưu SPF50+\nKháng nước kiềm dầu"),
                ("LÁ CHẮN TIA UV", "Ngăn ngừa sạm nám tối đa\nKết cấu mỏng nhẹ không nhờn rít"),
                ("TỰ TIN DƯỚI NẮNG HÈ", "Bảo vệ quang phổ rộng toàn diện\nThích hợp cho mọi hoạt động ngoài trời"),
            ],
            "inverted": [
                ("BẢO VỆ TỐI ƯU CHO LÀN DA TRƯỚC TIA CỰC TÍM VÀ ÁNH SÁNG XANH\nKHÁNG NƯỚC BỀN BỈ SUỐT NGÀY DÀI", "GIẢM 30%"),
                ("LÁ CHẮN VỮNG CHẮC BẢO VỆ LÀN DA SUỐT 12 GIỜ\nKIỀM DẦU HIỆU QUẢ", "MUA 1 TẶNG 1"),
            ],
        },
        "06_sua_rua_mat": {
            "standard": [
                ("SỮA RỬA MẶT", "Sạch sâu dịu nhẹ\nCân bằng độ ẩm tự nhiên"),
                ("LÀM SẠCH TINH KHIẾT", "Lấy đi bụi bẩn và bã nhờn\nDưỡng chất thiên nhiên dịu mát"),
                ("KHỞI ĐẦU NGÀY MỚI", "Bọt mịn êm ái thư giãn\nCảm giác sảng khoái tức thì"),
            ],
            "inverted": [
                ("LÀM SẠCH SÂU TẬN LỖ CHÂN LÔNG KHÔNG GÂY KHÔ RÁT\nBẢO VỆ ĐỘ ẨM TỰ NHIÊN CHO DA", "QUÀ TẶNG 0Đ"),
                ("CẢM GIÁC THƯ THÁI DỊU NHẸ NHƯ CHĂM SÓC TẠI SPA\nLOẠI BỎ HOÀN TOÀN BÃ NHỜN", "GIẢM 45%"),
            ],
        },
        "07_phan_nuoc_cushion": {
            "standard": [
                ("PHẤN NƯỚC CUSHION", "Lớp nền mỏng mịn\nChe phủ hoàn hảo suốt ngày"),
                ("LỚP NỀN TRONG VEO", "Hiệu ứng căng bóng tự nhiên\nKiềm dầu giữ tông lâu trôi"),
                ("TRANG ĐIỂM HOÀN HẢO", "Tiết kiệm thời gian mỗi sáng\nChỉ số chống nắng tích hợp cao"),
            ],
            "inverted": [
                ("CHE PHỦ HOÀN HẢO MỌI KHUYẾT ĐIỂM VỚI LỚP NỀN TRONG SUỐT\nTỰ NHIÊN CẢ NGÀY KHÔNG XUỐNG TÔNG", "GIẢM 40%"),
                ("HIỆU ỨNG CĂNG BÓNG MỊN MÀNG CHUẨN PHONG CÁCH HÀN QUỐC\nDƯỠNG ẨM TỐI ƯU", "HOT SALE"),
            ],
        },
        "08_dau_goi_dau": {
            "standard": [
                ("DẦU GỘI DƯỠNG CHẤT", "Bồng bềnh suôn mượt\nChiết xuất tinh dầu tự nhiên"),
                ("PHỤC HỒI HƯ TỔN", "Giảm gãy rụng tóc hiệu quả\nNuôi dưỡng nang tóc chắc khỏe"),
                ("HƯƠNG THƠM THẢO MỘC", "Thư giãn da đầu tinh tế\nMái tóc bồng bềnh tràn đầy sức sống"),
            ],
            "inverted": [
                ("GIẢI PHÁP PHỤC HỒI MÁI TÓC HƯ TỔN TỪ GỐC ĐẾN NGỌN\nBỒNG BỀNH SUÔN MƯỢT TỰ NHIÊN", "MUA 2 TẶNG 1"),
                ("NUÔI DƯỠNG MÁI TÓC CHẮC KHỎE BỚT GÃY RỤNG SAU 14 NGÀY\nTINH DẦU THIÊN NHIÊN", "GIẢM 35%"),
            ],
        },
    },

    # ----------------------------------------------------------------------------------------------
    # F&B (FOOD & BEVERAGE) (7 products)
    # ----------------------------------------------------------------------------------------------
    "fnb": {
        "09_phin_cafe_nhom": {
            "standard": [
                ("CÀ PHÊ PHIN NGUYÊN CHẤT", "Đậm đà phong vị Việt\nHương thơm nồng nàn truyền thống"),
                ("PHONG VỊ TRUYỀN THỐNG", "Chắt lọc từng giọt tinh túy\nKhởi đầu ngày mới tràn đầy cảm hứng"),
                ("CÀ PHÊ VIỆT NAM", "Đậm đà hương vị quê hương\nRang xay mộc chuẩn vị xưa"),
            ],
            "inverted": [
                ("CHẮT LỌC TỪNG GIỌT ĐẮNG TINH TÚY ĐẬM ĐÀ BẢN SẮC VIỆT\nĐÁNH THỨC NGUỒN CẢM HỨNG SÁNG TẠO", "ĐỒNG GIÁ 29K"),
                ("THƯỞNG THỨC HƯƠNG VỊ CÀ PHÊ PHIN TRUYỀN THỐNG ĐÍCH THỰC\nKHỞI ĐẦU NGÀY MỚI NĂNG ĐỘNG", "FREESHIP"),
            ],
        },
        "10_tui_cafe_rang_moc": {
            "standard": [
                ("HẠT ROBUSTA RANG MỘC", "Hương vị nguyên bản\nRang xay thủ công tinh tế"),
                ("ARABICA THƯỢNG HẠNG", "Hương thơm hoa quả nhẹ nhàng\nHậu vị ngọt sâu lắng đọng"),
                ("CÀ PHÊ ĐẶC SẢN CAO NGUYÊN", "Thu hái chín cây 100%\nChế biến ướt tiêu chuẩn quốc tế"),
            ],
            "inverted": [
                ("HẠT CÀ PHÊ NGUYÊN CHẤT ĐƯỢC THU HÁI VÀ RANG XAY THỦ CÔNG\nĐẬM ĐÀ HƯƠNG VỊ CAO NGUYÊN ĐẠI NGÀN", "GIẢM 30%"),
                ("TRẢI NGHIỆM HƯƠNG THƠM NGUYÊN BẢN TỪ NHỮNG HẠT CÀ PHÊ MỘC\nKHÔNG CHẤT BẢO QUẢN", "HOT DEAL"),
            ],
        },
        "11_lon_nuoc_tang_luc": {
            "standard": [
                ("BẬT TUNG NĂNG LƯỢNG", "Sảng khoái tức thì\nTỉnh táo chinh phục thử thách"),
                ("BỨT PHÁ GIỚI HẠN", "Bổ sung vitamin và khoáng chất\nĐập tan mọi cơn mệt mỏi"),
                ("TỈNH TÁO TẬP TRUNG", "Hương vị thơm ngon sảng khoái\nĐồng hành cùng game thủ và tài xế"),
            ],
            "inverted": [
                ("BẬT TUNG NGUỒN NĂNG LƯỢNG ĐẬP TAN MỌI MỆT MỎI ÁP LỰC\nCHINH PHỤC MỌI THỬ THÁCH ĐỈNH CAO", "MUA 5 TẶNG 1"),
                ("SẢNG KHOÁI TỨC THÌ TIẾP THÊM SỨC MẠNH VƯỢT TRỘI\nBỀN BỈ ĐẾN CUỐI NGÀY", "GIÁ SỐC"),
            ],
        },
        "12_chai_tra_xanh": {
            "standard": [
                ("TRÀ XANH THANH MÁT", "Chiết xuất lá trà tươi\nThanh lọc cơ thể mỗi ngày"),
                ("THANH NHIỆT CUỘC SỐNG", "Giàu chất chống oxy hóa\nVị đắng nhẹ hậu ngọt tự nhiên"),
                ("HƯƠNG VỊ THIÊN NHIÊN", "Giải khát tức thì sảng khoái\nKhông chất bảo quản an toàn"),
            ],
            "inverted": [
                ("CHIẾT XUẤT TỪ NHỮNG BÚP TRÀ XANH TƯƠI MÁT NƠI VÙNG ĐỒI CAO\nTHANH LỌC CƠ THỂ VÀ TÂM HỒN", "GIẢM 25%"),
                ("THƯỞNG THỨC VỊ THANH KHIẾT MÁT LÀNH CỦA TRÀ XANH TỰ NHIÊN\nBỔ SUNG NĂNG LƯỢNG LÀNH MẠNH", "FREESHIP"),
            ],
        },
        "13_hop_sua_hat": {
            "standard": [
                ("SỮA HẠT DINH DƯỠNG", "Thuần khiết tự nhiên\nGiàu canxi không đường"),
                ("DINH DƯỠNG LÀNH MẠNH", "Hạnh nhân và óc chó tự nhiên\nTốt cho tim mạch và vóc dáng"),
                ("BỮA SÁNG XANH TIỆN LỢI", "Cung cấp năng lượng sạch\nThơm ngon bổ dưỡng cho cả nhà"),
            ],
            "inverted": [
                ("NGUỒN DINH DƯỠNG THUẦN KHIẾT TỪ CÁC LOẠI HẠT QUÝ THIÊN NHIÊN\nCHĂM SÓC SỨC KHỎE GIA ĐÌNH TOÀN DIỆN", "TIẾT KIỆM 35%"),
                ("SỰ KẾT HỢP HOÀN HẢO GIỮA VỊ THƠM BÉO TỰ NHIÊN VÀ CANXI DỒI DÀO\nKHÔNG CHỨA ĐƯỜNG TINH LUYỆN", "HOT SALE"),
            ],
        },
        "14_lon_bia_craft": {
            "standard": [
                ("BIA THỦ CÔNG CAO CẤP", "Hương hoa bia sảng khoái\nMen bia ủ mộc thượng hạng"),
                ("MEN BIA NGUYÊN BẢN", "Quy trình lên men truyền thống\nĐộ cồn cân bằng hậu vị đậm đà"),
                ("BỮA TIỆC BÙNG NỔ", "Gắn kết bạn bè nâng ly chúc mừng\nHương vị độc bản khác biệt"),
            ],
            "inverted": [
                ("TUYỆT PHẨM BIA THỦ CÔNG LÊN MEN TRUYỀN THỐNG CHUẨN CHÂU ÂU\nĐẬM ĐÀ TỪNG GIỌT HƯƠNG HOA BIA", "COMBO ƯU ĐÃI"),
                ("NÂNG LY GẮN KẾT NHỮNG KHOẢNH KHẮC ĐÁNG NHỚ CÙNG TRI KỶ\nHƯƠNG VỊ NỒNG NÀN ĐỘC BẢN", "GIẢM 30%"),
            ],
        },
        "15_chai_ruou_vang": {
            "standard": [
                ("VANG ĐỎ THƯỢNG HẠNG", "Ủ thùng gỗ sồi lâu năm\nNồng nàn đẳng cấp quý phái"),
                ("HƯƠNG VỊ QUÝ TỘC", "Nho Cabernet Sauvignon tuyển chọn\nSóng sánh sắc đỏ quyến rũ"),
                ("TIỆC SANG ĐẲNG CẤP", "Món quà ngoại giao tinh tế\nNâng tầm bàn tiệc sum vầy"),
            ],
            "inverted": [
                ("TUYỆT TÁC RƯỢU VANG Ủ LÂU NĂM TRONG THÙNG GỖ SỒI PHÁP\nNỒNG NÀN HƯƠNG TRÁI CÂY CHÍN MỌNG", "MUA 1 TẶNG 1"),
                ("MÓN QUÀ TRI ÂN THƯỢNG HẠNG DÀNH TẶNG ĐỐI TÁC VÀ GIA ĐÌNH\nĐẲNG CẤP HOÀNG GIA", "GIẢM 40%"),
            ],
        },
    },

    # ----------------------------------------------------------------------------------------------
    # TECH (7 products)
    # ----------------------------------------------------------------------------------------------
    "tech": {
        "16_tai_nghe_tws": {
            "standard": [
                ("TAI NGHE CHỐNG ỒN", "Âm bass sống động\nPin 30 giờ liên tục"),
                ("ÂM THANH VÒM 3D", "Công nghệ chống ồn chủ động ANC\nĐàm thoại sắc nét khử tạp âm"),
                ("KẾT NỐI KHÔNG DÂY", "Bluetooth 5.3 siêu tốc không trễ\nThiết kế công thái học đeo êm ái"),
            ],
            "inverted": [
                ("ĐẮM CHÌM TRONG KHÔNG GIAN ÂM NHẠC SỐNG ĐỘNG TÁCH BIỆT THẾ GIỚI\nCÔNG NGHỆ KHỬ ỒN KỸ THUẬT SỐ ĐỈNH CAO", "GIẢM 50%"),
                ("TRẢI NGHIỆM CHẤT ÂM PHÒNG THU TRONG MỘT THIẾT KẾ SIÊU NHỎ GỌN\nPIN BỀN BỈ 30 GIỜ", "FLASH SALE"),
            ],
        },
        "17_smartwatch": {
            "standard": [
                ("ĐỒNG HỒ THÔNG MINH", "Theo dõi sức khỏe 24/7\nKháng nước chuẩn 5ATM"),
                ("TRỢ LÝ SỨC KHỎE", "Đo nhịp tim và nồng độ oxy SpO2\nHơn 100 chế độ tập luyện thể thao"),
                ("PHONG CÁCH THỜI THƯỢNG", "Màn hình AMOLED sắc nét rực rỡ\nNhận thông báo cuộc gọi tức thì"),
            ],
            "inverted": [
                ("NGƯỜI BẠN ĐỒNG HÀNH TOÀN DIỆN CHO SỨC KHỎE VÀ LỐI SỐNG NĂNG ĐỘNG\nQUẢN LÝ MỌI HOẠT ĐỘNG TRÊN CỔ TAY", "ƯU ĐÃI 40%"),
                ("THEO DÕI CHỈ SỐ SỨC KHỎE VÀ GIẤC NGỦ CHUYÊN SÂU TỪNG GIÂY PHÚT\nKHÁNG NƯỚC 5ATM", "HOT DEAL"),
            ],
        },
        "18_loa_bluetooth": {
            "standard": [
                ("LOA BLUETOOTH DI ĐỘNG", "Âm thanh vòm 360 độ\nKhuấy động mọi bữa tiệc"),
                ("ÂM BASS BÙNG NỔ", "Công suất mạnh mẽ sống động\nChống nước chuẩn IPX7 bền bỉ"),
                ("TIỆC VUI BẤT TẬN", "Thời lượng pin lên tới 24 giờ\nĐèn LED RGB nhấp nháy theo nhạc"),
            ],
            "inverted": [
                ("KHUẤY ĐỘNG MỌI KHÔNG GIAN BỮA TIỆC VỚI DẢI BASS SIÊU TRẦM NỘI LỰC\nÂM THANH VÒM 360 ĐỘ LAN TỎA", "GIẢM 35%"),
                ("GIAI ĐIỆU BÙNG CHÁY THEO TỪNG BƯỚC CHÂN PHƯỢT THỦ VÀ DÃ NGOẠI\nCHỐNG NƯỚC BỀN BỈ", "MUA 1 TẶNG 1"),
            ],
        },
        "19_chuot_gaming": {
            "standard": [
                ("CHUỘT GAMING KHÔNG DÂY", "Độ nhạy cực cao\nThiết kế công thái học đỉnh cao"),
                ("CẢM BIẾN QUANG HỌC", "Tốc độ phản hồi 1ms siêu nhanh\nSwitch cơ học 80 triệu lần bấm"),
                ("CHIẾN GAME VÔ ĐỐI", "Trọng lượng siêu nhẹ chỉ 60g\nChinh phục mọi đấu trường Esports"),
            ],
            "inverted": [
                ("VŨ KHÍ TỐI THƯỢNG GIÚP GAME THỦ LÀM CHỦ MỌI TRẬN CHIẾN TỐC ĐỘ\nĐỘ CHÍNH XÁC TUYỆT ĐỐI KHÔNG ĐỘ TRỄ", "GIẢM 45%"),
                ("THIẾT KẾ CÔNG THÁI HỌC VỪA VẶN LÒNG BÀN TAY CHIẾN GAME KHÔNG MỎI\nCẢM BIẾN QUANG HỌC ĐỈNH CAO", "HOT SALE"),
            ],
        },
        "20_ban_phim_co": {
            "standard": [
                ("BÀN PHÍM CƠ CAO CẤP", "Gõ phím êm ái\nĐèn nền RGB rực rỡ"),
                ("CẢM GIÁC GÕ ĐỈNH CAO", "Hot-swap thay switch nhanh chóng\nKeycap PBT double-shot bền bỉ"),
                ("GÓC LÀM VIỆC ĐẲNG CẤP", "Kết nối 3 chế độ đa năng tiện lợi\nPin sạc dùng cả tháng"),
            ],
            "inverted": [
                ("TRẢI NGHIỆM GÕ PHÍM MƯỢT MÀ ÊM TAI NÂNG TẦM HIỆU SUẤT LÀM VIỆC\nĐÈN LED RGB RỰC RỠ TÙY BIẾN", "TIẾT KIỆM 40%"),
                ("BÀN PHÍM CƠ CÔNG THÁI HỌC HOÀN HẢO CHO CODER VÀ GAME THỦ CHUYÊN NGHIỆP\nBỀN BỈ 100 TRIỆU LẦN NHẤN", "GIÁ SỐC"),
            ],
        },
        "21_sac_du_phong": {
            "standard": [
                ("SẠC NHANH ĐA NĂNG", "Công suất 65W vượt trội\nNhỏ gọn tiện lợi mang đi"),
                ("DUNG LƯỢNG 20000MAH", "Sạc cùng lúc 3 thiết bị an toàn\nCông nghệ sạc nhanh PD và QC"),
                ("NĂNG LƯỢNG KHÔNG NGỪNG", "Lõi pin Lithium Polymer bền bỉ\nBảo vệ quá dòng và quá nhiệt"),
            ],
            "inverted": [
                ("TRẠM SẠC DỰ PHÒNG CÔNG SUẤT CAO NẠP NHANH CHO CẢ LAPTOP VÀ ĐIỆN THOẠI\nAN TOÀN TUYỆT ĐỐI CHỐNG CHÁY NỔ", "GIẢM 50%"),
                ("NGUỒN NĂNG LƯỢNG BỀN BỈ ĐỒNG HÀNH TRÊN MỌI HÀNH TRÌNH CHUYẾN ĐI\nSẠC NHANH 65W", "MUA 1 TẶNG 1"),
            ],
        },
        "22_tay_cam_game": {
            "standard": [
                ("TAY CẦM CHƠI GAME", "Rung phản hồi chân thực\nKhông độ trễ trên mọi thiết bị"),
                ("CẦN GẠT CHỐNG TRÔI", "Cảm biến Hall Effect siêu bền\nTương thích PC, Console và Mobile"),
                ("CHIẾN GAME CHUYÊN NGHIỆP", "Phím bấm phản hồi xúc giác nảy\nThiết kế cầm nắm chắc chắn"),
            ],
            "inverted": [
                ("CẢM NHẬN TỪNG CHUYỂN ĐỘNG VÀ VA CHẠM CHÂN THỰC TRONG THẾ GIỚI ẢO\nTAY CẦM CHƠI GAME KHÔNG DÂY THẾ HỆ MỚI", "GIẢM 30%"),
                ("ĐIỀU KHIỂN CHÍNH XÁC TỪNG TIK TAK VỚI CẦN GẠT CHỐNG TRÔI ĐỘT PHÁ\nKẾT NỐI KHÔNG ĐỘ TRỄ", "HOT DEAL"),
            ],
        },
    },

    # ----------------------------------------------------------------------------------------------
    # FASHION (6 products)
    # ----------------------------------------------------------------------------------------------
    "fashion": {
        "23_giay_sneaker_bitis": {
            "standard": [
                ("GIÀY THỂ THAO NĂNG ĐỘNG", "Siêu nhẹ êm chân\nBước đi bứt phá tự tin"),
                ("BƯỚC ĐI TỰ HÀO", "Đế bọt khí đàn hồi êm ái\nThoáng khí tối đa cả ngày dài"),
                ("PHONG CÁCH ĐƯỜNG PHỐ", "Thiết kế thời thượng phá cách\nDễ dàng phối mọi trang phục"),
            ],
            "inverted": [
                ("ĐÔI GIÀY ĐỒNG HÀNH TRÊN MỌI CUNG ĐƯỜNG CHINH PHỤC ƯỚC MƠ\nÊM ÁI VÀ BỀN BỈ THEO THỜI GIAN", "GIẢM 40%"),
                ("BỨT PHÁ MỌI GIỚI HẠN VẬN ĐỘNG VỚI CÔNG NGHỆ ĐẾ SIÊU NHẸ\nTỰ TIN TRONG TỪNG BƯỚC CHẠY", "FLASH SALE"),
            ],
        },
        "24_kinh_mat_thoi_trang": {
            "standard": [
                ("KÍNH RÂM THỜI THƯỢNG", "Chống tia UV400\nTôn vinh phong cách cá nhân"),
                ("BẢO VỆ ĐÔI MẮT", "Tròng phân cực chống chói lóa\nGọng kim loại titan siêu nhẹ"),
                ("ĐẲNG CẤP MÙA HÈ", "Phụ kiện không thể thiếu khi du lịch\nThiết kế thời trang dẫn đầu xu hướng"),
            ],
            "inverted": [
                ("BẢO VỆ ĐÔI MẮT KHỎI TIA CỰC TÍM VỚI PHONG CÁCH LỊCH LÃM ĐẲNG CẤP\nTRÒNG KÍNH PHÂN CỰC CHỐNG CHÓI", "GIẢM 50%"),
                ("TÔN VINH ĐƯỜNG NÉT KHUÔN MẶT VÀ KHẲNG ĐỊNH GU THỜI TRANG ĐỘC BẢN\nGỌNG KIM LOẠI CAO CẤP", "HOT DEAL"),
            ],
        },
        "25_dong_ho_kim_loai": {
            "standard": [
                ("ĐỒNG HỒ KIM LOẠI SANG TRỌNG", "Đẳng cấp quý ông\nBộ máy cơ học chuẩn xác"),
                ("THỜI GIAN VÔ GIÁ", "Thép không gỉ 316L sáng bóng\nMặt kính sapphire chống trầy xước"),
                ("BIỂU TƯỢNG THÀNH ĐẠT", "Thiết kế lộ cơ tinh xảo\nKhẳng định vị thế người dẫn đầu"),
            ],
            "inverted": [
                ("BIỂU TƯỢNG CỦA SỰ CHÍNH XÁC VÀ BẢN LĨNH PHÁI MẠNH ĐẲNG CẤP\nBỘ MÁY CƠ TỰ ĐỘNG TINH XẢO", "GIẢM 35%"),
                ("NÂNG TẦM PHONG CÁCH QUÝ ÔNG LỊCH LÃM VỚI THIẾT KẾ ĐỒNG HỒ CAO CẤP\nBẢO HÀNH 5 NĂM", "ƯU ĐÃI VIP"),
            ],
        },
        "26_tui_xach_da": {
            "standard": [
                ("TÚI XÁCH DA THẬT", "Chất da cao cấp\nTinh tế từng đường kim mũi chỉ"),
                ("THANH LỊCH NỮ TÍNH", "Ngăn chứa đồ thông minh tiện dụng\nPhụ kiện kim loại mạ vàng sang trọng"),
                ("ĐỒNG HÀNH CÔNG SỞ", "Phù hợp cả đi làm và dạo phố\nĐộ bền vượt trội cùng năm tháng"),
            ],
            "inverted": [
                ("TINH HOA CHẾ TÁC THỦ CÔNG TỪ CHẤT LIỆU DA NGUYÊN TẤM THƯỢNG HẠNG\nTÔN VINH NÉT QUÝ PHÁI KIÊU KỲ", "GIẢM 45%"),
                ("CHIẾC TÚI XÁCH HOÀN HẢO DÀNH RIÊNG CHO QUÝ CÔ HIỆN ĐẠI TỰ TIN\nKHÔNG GIAN TIỆN DỤNG", "MUA 1 TẶNG 1"),
            ],
        },
        "27_vi_da_nam": {
            "standard": [
                ("VÍ DA CẦM TAY", "Da bò nguyên tấm\nBền đẹp cùng thời gian"),
                ("GỌN GÀNG LỊCH LÃM", "Thiết kế gập đôi mỏng nhẹ\nNhiều ngăn đựng thẻ và tiền mặt tiện ích"),
                ("QUÀ TẶNG Ý NGHĨA", "Món quà tinh tế cho người đàn ông\nĐóng gói hộp xi sang trọng"),
            ],
            "inverted": [
                ("CHẤT DA BÒ THẬT NGUYÊN TẤM CÀNG DÙNG CÀNG BÓNG ĐẸP THEO NĂM THÁNG\nTHIẾT KẾ ĐẲNG CẤP TINH TẾ", "TIẾT KIỆM 40%"),
                ("PHỤ KIỆN BỎ TÚI BỀN BỈ KHÔNG THỂ THIẾU CỦA NGƯỜI ĐÀN ÔNG THÀNH ĐẠT\nCHỐNG NƯỚC HIỆU QUẢ", "GIÁ SỐC"),
            ],
        },
        "28_non_la_viet_nam": {
            "standard": [
                ("NÓN LÁ DUYÊN DÁNG", "Nét đẹp truyền thống\nHồn quê đất Việt ngàn năm"),
                ("TINH HOA LÀNG NGHỀ", "Đan tỉ mỉ từ lá cọ phơi sương\nBiểu tượng văn hóa dịu dàng"),
                ("QUÀ TẶNG QUÊ HƯƠNG", "Kỷ vật lưu niệm cho bạn bè quốc tế\nChe nắng che mưa thủy chung"),
            ],
            "inverted": [
                ("BIỂU TƯỢNG VĂN HÓA TRUYỀN THỐNG DỊU DÀNG CỦA NGƯỜI PHỤ NỮ VIỆT NAM\nĐAN THỦ CÔNG TỪ LÁ NÓN TINH KHÔI", "ĐẶC SẢN VIỆT"),
                ("CHỞ CHE NẮNG MƯA VÀ LƯU GIỮ HỒN THƠ CỦA LÀNG QUÊ YÊU DẤU\nQUÀ TẶNG Ý NGHĨA", "GIÁ GỐC"),
            ],
        },
    },

    # ----------------------------------------------------------------------------------------------
    # HOME (6 products)
    # ----------------------------------------------------------------------------------------------
    "home": {
        "29_binh_giu_nhiet": {
            "standard": [
                ("BÌNH GIỮ NHIỆT INOX", "Giữ nhiệt suốt 24 giờ\nThép không gỉ an toàn"),
                ("ĐỒNG HÀNH MỖI NGÀY", "Giữ nóng 12h giữ lạnh 24h\nNắp vặn kín chống rò rỉ nước"),
                ("SỐNG XANH TIỆN LỢI", "Giảm thiểu rác thải nhựa một lần\nThiết kế tối giản sang trọng"),
            ],
            "inverted": [
                ("GIỮ TRỌN VẸN HƯƠNG VỊ NÓNG LẠNH YÊU THÍCH SUỐT 24 GIỜ LIÊN TỤC\nTHÉP KHÔNG GỈ AN TOÀN TUYỆT ĐỐI", "GIẢM 40%"),
                ("NGƯỜI BẠN TIỆN DỤNG CUNG CẤP NƯỚC ẤM TRÊN BÀN LÀM VIỆC VÀ TẬP GYM\nTHIẾT KẾ TỐI GIẢN", "MUA 1 TẶNG 1"),
            ],
        },
        "30_may_say_toc": {
            "standard": [
                ("MÁY SẤY TÓC ION ÂM", "Sấy khô siêu tốc\nBảo vệ tóc bóng mượt"),
                ("CHĂM SÓC CHUYÊN SÂU", "Hàng triệu ion âm chống tĩnh điện\nCảm biến kiểm soát nhiệt thông minh"),
                ("TẠO KIỂU TẠI NHÀ", "Đầu sấy từ tính xoay 360 độ\nTóc bồng bềnh như bước ra từ salon"),
            ],
            "inverted": [
                ("CÔNG NGHỆ SẤY KHÔ SIÊU TỐC VÀ DƯỠNG TÓC BÓNG MƯỢT VỚI HÀNG TRIỆU ION ÂM\nKHÔNG GÂY KHÔ XƠ GÃY RỤNG", "TIẾT KIỆM 35%"),
                ("BẢO VỆ TÓC KHỎI TỔN THƯƠNG NHIỆT VÀ TẠO KIỂU DỄ DÀNG NGAY TẠI NHÀ\nĐỘNG CƠ KHÔNG CHỔI THAN", "HOT DEAL"),
            ],
        },
        "31_ban_ui_hoi_nuoc": {
            "standard": [
                ("BÀN ỦI HƠI NƯỚC", "Phẳng phiu tức thì\nKháng khuẩn 99% áo quần"),
                ("ỦI ĐỨNG CẦM TAY", "Luồng hơi nước áp lực mạnh mẽ\nPhù hợp mọi chất liệu vải mềm"),
                ("TIỆN LỢI DU LỊCH", "Thiết kế gập gọn mang đi công tác\nLàm mới trang phục chỉ sau 1 phút"),
            ],
            "inverted": [
                ("ỦI PHẲNG PHIU MỌI NẾP NHĂN VÀ TIÊU DIỆT VI KHUẨN TRÊN QUẦN ÁO NHANH CHÓNG\nLUỒNG HƠI NƯỚC ÁP SUẤT CAO", "GIẢM 30%"),
                ("GIẢI PHÁP LÀM MỚI TRANG PHỤC CÔNG SỞ GỌN NHẸ CHO NGƯỜI BẬN RỘN\nKHÔNG CHÁY VẢI", "FLASH SALE"),
            ],
        },
        "32_may_xay_sinh_to": {
            "standard": [
                ("MÁY XAY SINH TỐ MINI", "Xay nhuyễn mịn đa năng\nSống khỏe tươi vui mỗi ngày"),
                ("TIỆN LỢI MỌI NƠI", "Cối xay kiêm bình nước thể thao\nPin sạc dùng được 15 lần xay"),
                ("DINH DƯỠNG TƯƠI MÁT", "Lưỡi dao thép 6 cánh sắc bén\nXay đá và trái cây trong 30 giây"),
            ],
            "inverted": [
                ("THƯỞNG THỨC MỖI NGÀY MỘT LY SINH TỐ TƯƠI MÁT GIÀU VITAMIN ĐẦY DINH DƯỠNG\nCỐI XAY CẦM TAY TIỆN LỢI", "GIẢM 50%"),
                ("XAY NHUYỄN MỊN TRÁI CÂY VÀ ĐÁ VIÊN TỨC THÌ CHỈ VỚI MỘT NÚT BẤM\nSỐNG LÀNH MẠNH", "MUA 1 TẶNG 1"),
            ],
        },
        "33_noi_chien_khong_dau": {
            "standard": [
                ("NỒI CHIÊN KHÔNG DẦU", "Giảm 85% chất béo\nChín vàng giòn rụm thơm ngon"),
                ("CÔNG NGHỆ NƯỚNG 360", "Lưu thông khí nóng tuần hoàn đều\nKhông cần lật trở món ăn"),
                ("BỮA CƠM TIỆN LỢI", "Dung tích lớn 8 lít cho cả nhà\nMàn hình cảm ứng 12 chế độ tự động"),
            ],
            "inverted": [
                ("BÍ QUYẾT NẤU NƯỚNG GIẢM ĐẾN TÁM MƯƠI LĂM PHẦN TRĂM DẦU MỠ THỪA\nMÓN ĂN CHÍN VÀNG GIÒN RỤM", "TIẾT KIỆM 40%"),
                ("CHĂM SÓC SỨC KHỎE CẢ GIA ĐÌNH VỚI NHỮNG BỮA ĂN KHÔNG DẦU MỠ TIỆN LỢI\nDUNG TÍCH KHỦNG 8 LÍT", "HOT SALE"),
            ],
        },
        "34_den_ban_led": {
            "standard": [
                ("ĐÈN BÀN CHỐNG CẬN", "Ánh sáng dịu mắt\nTùy chỉnh 3 chế độ thông minh"),
                ("BẢO VỆ THỊ LỰC", "Không chớp nháy không bức xạ hại\nĐộ hoàn màu CRI cao sắc nét"),
                ("GÓC HỌC TẬP THÔNG MINH", "Thân đèn gập linh hoạt đa hướng\nCổng sạc điện thoại tích hợp"),
            ],
            "inverted": [
                ("NGUỒN ÁNH SÁNG TỰ NHIÊN BẢO VỆ ĐÔI MẮT KHỎI NGUY CƠ CẬN THỊ HỌC ĐƯỜNG\nCÔNG NGHỆ LED CHỐNG CHÓI MỎI MẮT", "GIẢM 35%"),
                ("NGƯỜI BẠN ĐỒNG HÀNH TIN CẬY CHO NHỮNG ĐÊM DÀI HỌC TẬP VÀ LÀM VIỆC\nTIẾT KIỆM ĐIỆN", "QUÀ TẶNG KÈM"),
            ],
        },
    },

    # ----------------------------------------------------------------------------------------------
    # FMCG (FAST MOVING CONSUMER GOODS) (6 products)
    # ----------------------------------------------------------------------------------------------
    "fmcg": {
        "35_mi_hao_hao": {
            "standard": [
                ("MÌ HẢO HẢO TÔM CHUA CAY", "Sợi mì dai giòn đậm vị\nHương vị quốc dân gắn kết"),
                ("HƯƠNG VỊ QUỐC DÂN", "Chua cay bùng nổ vị giác\nĐậm đà phong vị bữa cơm gia đình"),
                ("BỮA ĂN NHANH TIỆN LỢI", "Sẵn sàng chỉ sau 3 phút\nĐồng hành cùng bao thế hệ Việt"),
            ],
            "inverted": [
                ("HƯƠNG VỊ TÔM CHUA CAY QUỐC DÂN GẮN LIỀN VỚI KÝ ỨC CỦA BAO THẾ HỆ\nSỢI MÌ VÀNG DAI GIÒN ĐẬM ĐÀ", "THÙNG TIẾT KIỆM"),
                ("BÙNG NỔ VỊ GIÁC VỚI GÓI NƯỚC SỐT CHUA CAY ĐẬM ĐÀ KHÓ CƯỠNG\nẤM ÁP TỪNG BỮA ĂN ĐÊM", "GIÁ TẬN GỐC"),
            ],
        },
        "36_hop_tra_sen_tay_ho": {
            "standard": [
                ("TRÀ SEN TÂY HỒ", "Hương sen thanh khiết\nTinh hoa trà búp Tân Cương"),
                ("TINH HOA TRÀ ĐẠO", "Ướp thủ công từ gạo sen Bách Diệp\nThức uống cung đình tao nhã"),
                ("QUÀ TẶNG TRÂN QUÝ", "Món quà văn hóa ngàn năm Thăng Long\nVị chát thanh hậu ngọt sâu"),
            ],
            "inverted": [
                ("CHẮT LỌC TINH TÚY ĐẤT TRỜI TRONG TỪNG BÚP TRÀ ƯỚP HƯƠNG HOA SEN TÂY HỒ\nNÉT ĐẸP TAO NHÃ TRUYỀN THỐNG VIỆT", "QUÀ TẾT CAO CẤP"),
                ("HƯƠNG THƠM THANH KHIẾT LẮNG ĐỌNG HỒN THIÊNG NGHÌN NĂM VĂN HIẾN\nTRÀ SEN THƯỢNG HẠNG", "GIẢM 20%"),
            ],
        },
        "37_chai_nuoc_mam_phu_quoc": {
            "standard": [
                ("NƯỚC MẮM CỐT PHÚ QUỐC", "Đậm đà vị cá cơm truyền thống\nỦ chượp ròng rã tự nhiên"),
                ("ĐẬM ĐÀ VỊ BIỂN", "Độ đạm tự nhiên nguyên chất 40 độ\nThùng gỗ bời lời ủ truyền thống"),
                ("LINH HỒN ẨM THỰC", "Gia vị tinh túy của bữa cơm gia đình\nMàu cánh gián óng ánh sóng sánh"),
            ],
            "inverted": [
                ("Ủ CHƯỢP RÒNG RÃ TRONG THÙNG GỖ BỜI LỜI NƠI ĐẢO NGỌC PHÚ QUỐC\nĐẬM ĐÀ GIỌT NƯỚC MẮM CỐT TRUYỀN THỐNG", "GIẢM 25%"),
                ("LINH HỒN CỦA MỌI MÓN ĂN GIA ĐÌNH VIỆT VỚI ĐỘ ĐẠM TỰ NHIÊN NGUYÊN BẢN\nCHỨNG NHẬN CHỈ DẪN ĐỊA LÝ", "HOT DEAL"),
            ],
        },
        "38_hop_cao_sao_vang": {
            "standard": [
                ("CAO SAO VÀNG CỔ ĐIỂN", "Tinh dầu tràm quế tự nhiên\nThương hiệu vượt thời gian"),
                ("LIỆU PHÁP DÂN GIAN", "Làm ấm cơ thể giảm cảm mạo\nHương thơm thảo mộc quen thuộc"),
                ("KỶ VẬT THỜI GIAN", "Hộp thiếc đỏ ngôi sao vàng nhỏ gọn\nCó mặt trong mọi tủ thuốc gia đình"),
            ],
            "inverted": [
                ("THƯƠNG HIỆU QUỐC DÂN VƯỢT THỜI GIAN ĐỒNG HÀNH CHĂM SÓC SỨC KHỎE\nTINH DẦU THẢO MỘC TỰ NHIÊN", "COMBO 5 HỘP"),
                ("GIỮ ẤM CƠ THỂ VÀ XUA TAN CẢM MẠO NHỜ TINH HOA CÁC LOÀI THẢO MỘC\nKỶ VẬT GIA ĐÌNH", "GIÁ SỐC"),
            ],
        },
        "39_hu_yen_sao_khanh_hoa": {
            "standard": [
                ("YẾN SÀO KHÁNH HÒA", "Bồi bổ sức khỏe tinh anh\nQuà tặng trân quý cho gia đình"),
                ("TINH TÚY THIÊN NHIÊN", "Tổ yến đảo thiên nhiên nguyên chất\nChưng đường phèn thanh mát"),
                ("TĂNG CƯỜNG ĐỀ KHÁNG", "Dinh dưỡng dồi dào cho trẻ em và người già\nPhục hồi thể trạng nhanh chóng"),
            ],
            "inverted": [
                ("NGUỒN DINH DƯỠNG QUÝ GIÁ TỪ TỔ YẾN ĐẢO TỰ NHIÊN NƠI VÙNG BIỂN KHÁNH HÒA\nBỒI BỔ THỂ LỰC VÀ TRÍ LỰC", "MUA 6 TẶNG 1"),
                ("MÓN QUÀ THƯỢNG HẠNG GẮN KẾT YÊU THƯƠNG DÀNH TẶNG ÔNG BÀ CHA MẸ\nCHƯNG SẴN TIỆN LỢI", "GIẢM 30%"),
            ],
        },
        "40_hop_banh_quy_bo": {
            "standard": [
                ("BÁNH QUY BƠ THƯỢNG HẠNG", "Thơm lừng bơ sữa nguyên chất\nGiòn tan tròn vị yêu thương"),
                ("HƯƠNG VỊ HOÀNG GIA", "Công thức nướng truyền thống Đan Mạch\nĐa dạng hình dáng bắt mắt"),
                ("MÓN NGON SUM VẦY", "Thưởng thức cùng trà chiều ấm áp\nHộp thiếc in nổi sang trọng"),
            ],
            "inverted": [
                ("THƯỞNG THỨC HƯƠNG VỊ BƠ SỮA NỒNG NÀN GIÒN TAN TRONG TỪNG MIẾNG BÁNH\nCÔNG THỨC NƯỚNG TRUYỀN THỐNG CHÂU ÂU", "GIẢM 35%"),
                ("MÓN BÁNH QUY HỘP THIẾC TRANG NHÃ CHO NHỮNG BUỔI TRÀ CHIỀU SUM VẦY\nQUÀ TẶNG Ý NGHĨA", "HOT SALE"),
            ],
        },
    },

    # ----------------------------------------------------------------------------------------------
    # TELECOM & VIETTEL (5 products)
    # ----------------------------------------------------------------------------------------------
    "telecom_viettel": {
        "41_modem_wifi6_viettel": {
            "standard": [
                ("MODEM HOME WIFI 6", "Phủ sóng toàn diện ngôi nhà\nTốc độ Gigabit không giật lag"),
                ("KẾT NỐI KHÔNG DÂY TỐC ĐỘ CAO", "Băng thông siêu rộng chịu tải lớn\nCông nghệ Mesh loại bỏ vùng lõm sóng"),
                ("INTERNET THẾ HỆ MỚI", "Xem phim 4K chơi game mượt mà\nLắp đặt tận nhà miễn phí"),
            ],
            "inverted": [
                ("TRẢI NGHIỆM TỐC ĐỘ INTERNET GIGABIT VƯỢT TRỘI VÀ PHỦ SÓNG TOÀN BỘ KHÔNG GIAN\nCÔNG NGHỆ MESH WIFI 6 THẾ HỆ MỚI", "TRANG BỊ 0Đ"),
                ("LOẠI BỎ HOÀN TOÀN MỌI VÙNG LÕM SÓNG TRONG NGÔI NHÀ NHIỀU TẦNG CỦA BẠN\nKẾT NỐI KHÔNG ĐỘ TRỄ", "HOT DEAL"),
            ],
        },
        "42_phoi_sim_5g_viettel": {
            "standard": [
                ("SIM VIETTEL 5G SIÊU TỐC", "Tốc độ vượt trội kết nối tương lai\nƯu đãi data không giới hạn"),
                ("KỶ NGUYÊN SỐ 5G", "Tải tệp dung lượng lớn trong chớp mắt\nPhủ sóng toàn quốc từ thành thị đến nông thôn"),
                ("GÓI CƯỚC THÔNG MINH", "Thoại nội mạng thả ga miễn phí\nTặng data khủng xem truyền hình TV360"),
            ],
            "inverted": [
                ("CHẠM ĐẾN KỶ NGUYÊN SỐ VỚI MẠNG DI ĐỘNG 5G TỐC ĐỘ CAO HÀNG ĐẦU VIỆT NAM\nDATA KHÔNG GIỚI HẠN DUNG LƯỢNG", "ĐỔI SIM MIỄN PHÍ"),
                ("KẾT NỐI KHÔNG GIỚI HẠN VỚI HẠ TẦNG MẠNG 5G PHỦ KHẮP TOÀN BỘ TỈNH THÀNH\nSIÊU TỐC ĐỘ", "ƯU ĐÃI ĐỘC QUYỀN"),
            ],
        },
        "43_smart_camera_viettel": {
            "standard": [
                ("CAMERA THÔNG MINH VIETTEL", "Hình ảnh 2K sắc nét ban đêm\nLưu trữ đám mây bảo mật tuyệt đối"),
                ("AN TÂM MỌI LÚC MỌI NƠI", "Đàm thoại 2 chiều to rõ ràng\nCảnh báo chuyển động gửi về điện thoại"),
                ("BẢO VỆ TỔ ẤM GIA ĐÌNH", "Góc nhìn toàn cảnh xoay 360 độ\nSever đặt tại Việt Nam an toàn thông tin"),
            ],
            "inverted": [
                ("BẢO VỆ TỔ ẤM VÀ NGƯỜI THÂN YÊU 24/7 VỚI MẮT THẦN CAMERA AI THÔNG MINH\nLƯU TRỮ ĐÁM MÂY AN TOÀN TUYỆT ĐỐI", "CHỈ TỪ 30K/THÁNG"),
                ("QUAN SÁT SẮC NÉT TRONG ĐÊM VÀ ĐÀM THOẠI HAI CHIỀU MỌI LÚC MỌI NƠI\nCÔNG NGHỆ NHẬN DIỆN AI", "MIỄN PHÍ LẮP ĐẶT"),
            ],
        },
        "44_thiet_bi_v_tracking": {
            "standard": [
                ("ĐỊNH VỊ V-TRACKING", "Giám sát hành trình 24/7\nQuản lý phương tiện thông minh"),
                ("GIẢI PHÁP VẬN TẢI", "Hợp chuẩn bộ giao thông vận tải\nBáo cáo nhiên liệu và tốc độ chính xác"),
                ("QUẢN LÝ ĐỘI XE TỐI ƯU", "Cảnh báo vượt tốc độ qua ứng dụng\nLưu trữ lịch sử lộ trình suốt 1 năm"),
            ],
            "inverted": [
                ("GIẢI PHÁP ĐỊNH VỊ GPS GIÁM SÁT HÀNH TRÌNH XE DOANH NGHIỆP TOÀN DIỆN\nHỢP CHUẨN QUY ĐỊNH BỘ GIAO THÔNG VẬN TẢI", "TIẾT KIỆM 30%"),
                ("QUẢN LÝ ĐỘI XE VÀ ĐỊNH VỊ CHÍNH XÁC VỊ TRÍ PHƯƠNG TIỆN TỪNG GIÂY\nTỐI ƯU HÓA CHI PHÍ", "ƯU ĐÃI NĂM ĐẦU"),
            ],
        },
        "45_hop_tv360_box": {
            "standard": [
                ("TRUYỀN HÌNH TV360", "Thế giới giải trí không giới hạn\nHàng trăm kênh truyền hình chuẩn HD"),
                ("RẠP PHIM TẠI GIA", "Kho phim bom tấn 4K cập nhật hàng ngày\nTrực tiếp các giải bóng đá đỉnh cao"),
                ("ĐẦU THU THÔNG MINH", "Điều khiển tìm kiếm giọng nói tiếng Việt\nBiến tivi thường thành Smart TV"),
            ],
            "inverted": [
                ("THỎA SỨC GIẢI TRÍ VỚI HÀNG TRĂM KÊNH TRUYỀN HÌNH ĐỈNH CAO VÀ PHIM BOM TẤN\nĐIỀU KHIỂN TÌM KIẾM BẰNG GIỌNG NÓI", "TẶNG HỘP BOX 0Đ"),
                ("MANG CẢ THẾ GIỚI ĐIỆN ẢNH VÀ TRẬN CẦU BÓNG ĐÁ ĐỈNH CAO VỀ PHÒNG KHÁCH\nTRUYỀN HÌNH 4K SIÊU NÉT", "HOT SALE"),
            ],
        },
    },

    # ----------------------------------------------------------------------------------------------
    # FITNESS (5 products)
    # ----------------------------------------------------------------------------------------------
    "fitness": {
        "46_binh_lac_shaker": {
            "standard": [
                ("BÌNH LẮC SHAKER THỂ THAO", "Khuấy tan bột siêu nhanh\nNhựa nguyên sinh an toàn sức khỏe"),
                ("NGƯỜI BẠN GYMER", "Lưới đánh bột chống vón cục hiệu quả\nNắp đậy kín không rỉ một giọt"),
                ("THỜI TRANG PHÒNG TẬP", "Dung tích 700ml chuẩn tập luyện\nChất liệu nhựa cao cấp BPA Free"),
            ],
            "inverted": [
                ("CHIẾC BÌNH LẮC TIỆN DỤNG KHUẤY TAN PROTEIN SIÊU MỊN CHỈ VỚI VÀI LẦN LẮC\nNHỰA NGUYÊN SINH KHÔNG ĐỘC HẠI", "GIẢM 40%"),
                ("BỔ SUNG PROTEIN VÀ NƯỚC UỐNG ĐẦY ĐỦ SUỐT BUỔI TẬP CƯỜNG ĐỘ CAO\nDUNG TÍCH 700ML", "MUA 1 TẶNG 1"),
            ],
        },
        "47_tham_yoga": {
            "standard": [
                ("THẢM TẬP YOGA CAO CẤP", "Độ bám sàn chống trơn trượt\nÊm ái trong từng chuyển động"),
                ("TẬP LUYỆN DẺO DAI", "Chất liệu TPE sinh học thân thiện môi trường\nĐịnh tuyến chuẩn hỗ trợ đúng tư thế"),
                ("THƯ GIÃN CÂN BẰNG", "Độ dày 6mm bảo vệ khớp xương\nKháng khuẩn chống thấm mồ hôi"),
            ],
            "inverted": [
                ("ĐỒNG HÀNH TRÊN HÀNH TRÌNH TÌM LẠI SỰ CÂN BẰNG VÀ DẺO DAI CỦA CƠ THỂ\nĐỘ BÁM SÀN TUYỆT ĐỐI CHỐNG TRƠN TRƯỢT", "GIẢM 35%"),
                ("THẢM ĐỊNH TUYẾN CAO CẤP BẢO VỆ KHỚP VÀ TẬP YOGA CHUẨN TỪNG TƯ THẾ\nCHẤT LIỆU TPE ÊM ÁI", "TẶNG TÚI ĐỰNG"),
            ],
        },
        "48_gang_tay_gym": {
            "standard": [
                ("GĂNG TAY TẬP TẠ", "Bảo vệ cổ tay vững chắc\nThoáng khí chống chai tay"),
                ("TỰ TIN NÂNG TẠ", "Đệm lòng bàn tay chống trơn trợ lực\nĐai quấn cổ tay dài gia cố an toàn"),
                ("PHONG CÁCH GYM MẠNH MẼ", "Chất liệu sợi co giãn thoáng mồ hôi\nBền bỉ trong từng hiệp tập nặng"),
            ],
            "inverted": [
                ("BẢO VỆ TỐI ĐA CHO CỔ TAY VÀ LÒNG BÀN TAY TRƯỚC NHỮNG MỨC TẠ NẶNG\nTHIẾT KẾ THOÁNG KHÍ TRỢ LỰC BỀN BỈ", "TIẾT KIỆM 40%"),
                ("NÂNG CAO HIỆU SUẤT TẬP LUYỆN VÀ LOẠI BỎ HOÀN TOÀN CHAI SẦN TAY\nĐỆM SILICONE CHỐNG TRƯỢT", "GIÁ TỐT"),
            ],
        },
        "49_con_lan_massage": {
            "standard": [
                ("CON LĂN MASSAGE GIÃN CƠ", "Giảm căng cứng cơ bắp\nPhục hồi thần tốc sau luyện tập"),
                ("TRỊ LIỆU TẠI NHÀ", "Gai massage tác động sâu vào huyệt đạo\nGiải tỏa axit lactic sau khi chạy bộ"),
                ("THƯ GIÃN TOÀN DIỆN", "Lõi chịu tải lên đến 200kg\nChất liệu bọt xốp EVA cao cấp"),
            ],
            "inverted": [
                ("PHỤC HỒI THẦN TỐC CƠ BẮP VÀ GIẢI TỎA MỌI ĐAU MỎI CĂNG CỨNG SAU BUỔI TẬP\nCON LĂN MASSAGE TRỊ LIỆU CHUYÊN SÂU", "GIẢM 45%"),
                ("GIẢI TỎA CĂNG THẲNG CHO HỆ CƠ VÀ NÂNG CAO ĐỘ ĐÀN HỒI CHO CƠ THỂ\nCHẤT LIỆU BỌT XỐP CAO CẤP", "HOT DEAL"),
            ],
        },
        "50_day_nhay_toc_do": {
            "standard": [
                ("DÂY NHẢY TỐC ĐỘ CAO", "Lõi cáp thép bền bỉ\nĐốt cháy calo rèn luyện sức bền"),
                ("CARDIO GIẢM CÂN", "Vòng bi xoay 360 độ siêu mượt không xoắn\nĐiều chỉnh chiều dài linh hoạt"),
                ("VẬN ĐỘNG MỖI NGÀY", "Tay cầm bọc nhôm chống trượt bám tay\nĐốt mỡ toàn thân hiệu quả nhanh"),
            ],
            "inverted": [
                ("ĐỐT CHÁY MỠ THỪA HIỆU QUẢ VÀ NÂNG CAO SỨC BỀN TIM MẠCH VỚI TỐC ĐỘ CAO\nLÕI CÁP THÉP XOAY BA TRĂM SÁU MƯƠI ĐỘ", "GIẢM 50%"),
                ("BÀI TẬP CARDIO ĐỐT CALO SIÊU TỐC TẠI NHÀ CHO MỌI LỨA TUỔI\nTAY CẦM CHỐNG TRƯỢT", "FREESHIP"),
            ],
        },
    },
}

# ==================================================================================================
# 2. GENERAL T2I CORPUS (BY USE-CASE & LENGTH STRATA)
# ==================================================================================================
GENERAL_T2I_CORPUS: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "flash_sale": {
        "standard": [
            ("LỄ HỘI MUA SẮM", "Giảm giá lên đến 50%\nSố lượng có hạn trong hôm nay"),
            ("SIÊU SALE GIỮA NĂM", "Hàng ngàn ưu đãi hấp dẫn\nMiễn phí vận chuyển toàn quốc"),
            ("GIỜ VÀNG GIÁ SỐC", "Đồng giá từ 99 ngàn đồng\nÁp dụng cho mọi đơn hàng"),
            ("XẢ KHO ĐÓN HÈ", "Giảm giá kịch sàn mọi mặt hàng\nNhanh tay kẻo lỡ"),
            ("ĐẠI TIỆC CUỐI TUẦN", "Tặng voucher 100K cho thành viên\nCơ hội trúng quà tặng giá trị"),
        ],
        "inverted": [
            ("CƠ HỘI MUA SẮM DUY NHẤT TRONG NĂM VỚI HÀNG NGÀN ƯU ĐÃI KHỦNG\nÁP DỤNG TRÊN TOÀN HỆ THỐNG", "GIẢM 70%"),
            ("ĐẠI TIỆC TRI ÂN KHÁCH HÀNG THÂN THIẾT VỚI NHỮNG PHẦN QUÀ GIÁ TRỊ\nSỐ LƯỢNG CÓ HẠN", "CHỈ HÔM NAY"),
            ("BÙNG NỔ CƠN LỐC GIẢM GIÁ KHÔNG TƯỞNG CHO TOÀN BỘ SẢN PHẨM\nĐỪNG BỎ LỠ CƠ HỘI NÀY", "MUA 1 TẶNG 1"),
        ],
    },
    "recruitment": {
        "standard": [
            ("KỸ SƯ TRÍ TUỆ NHÂN TẠO", "Thu nhập hấp dẫn cạnh tranh\nMôi trường sáng tạo mở"),
            ("CHUYÊN VIÊN MARKETING", "Phát triển tiềm năng vượt bậc\nĐãi ngộ hàng đầu ngành"),
            ("LẬP TRÌNH VIÊN BACKEND", "Làm việc linh hoạt hybrid\nDự án quy mô triệu người dùng"),
            ("TRƯỞNG NHÓM KINH DOANH", "Lương thưởng không giới hạn\nLộ trình thăng tiến rõ ràng"),
            ("QUẢN LÝ DỰ ÁN CÔNG NGHỆ", "Văn hóa chủ động bứt phá\nChế độ bảo hiểm toàn diện"),
        ],
        "inverted": [
            ("CHÀO ĐÓN NHỮNG NHÂN TÀI ĐỒNG HÀNH KIẾN TẠO NỀN TẢNG CÔNG NGHỆ TƯƠNG LAI\nMÔI TRƯỜNG LÀM VIỆC ĐẲNG CẤP QUỐC TẾ", "ỨNG TUYỂN NGAY"),
            ("GIA NHẬP ĐỘI NGŨ TIÊN PHONG TRONG LĨNH VỰC CHUYỂN ĐỔI SỐ DOANH NGHIỆP\nTHU NHẬP KHÔNG GIỚI HẠN", "TUYỂN DỤNG"),
            ("TÌM KIẾM NHỮNG CHIẾN BINH SALES BỨT PHÁ MỌI CHỈ TIÊU KINH DOANH\nĐÃI NGỘ KHỦNG", "HOT JOB"),
        ],
    },
    "two_step_guide": {
        "standard": [
            ("BƯỚC 1: ĐĂNG KÝ TÀI KHOẢN", "BƯỚC 2: BẮT ĐẦU TRẢI NGHIỆM\nHoàn toàn miễn phí"),
            ("BƯỚC 1: TẢI ỨNG DỤNG", "BƯỚC 2: NHẬN NGAY VOUCHER 50K\nÁp dụng cho đơn đầu tiên"),
            ("BƯỚC 1: QUÉT MÃ QR CODE", "BƯỚC 2: THANH TOÁN TỨC THÌ\nAn toàn và tiện lợi"),
            ("BƯỚC 1: CHỌN SẢN PHẨM", "BƯỚC 2: XÁC NHẬN GIAO TẬN NƠI\nĐổi trả trong 7 ngày"),
        ],
        "inverted": [
            ("QUY TRÌNH KÍCH HOẠT DỊCH VỤ VÀ NHẬN NGAY GÓI QUÀ TẶNG THÀNH VIÊN MỚI\nCHỈ VỚI HAI BƯỚC ĐƠN GIẢN", "BƯỚC 1: ĐĂNG KÝ"),
            ("HƯỚNG DẪN MỞ TÀI KHOẢN TRỰC TUYẾN VÀ ĐĂNG NHẬP ỨNG DỤNG AN TOÀN\nBẢO MẬT ĐA TẦNG", "BẮT ĐẦU NGAY"),
        ],
    },
    "creative_quote": {
        "standard": [
            ("HÃY THEO ĐUỔI ĐAM MÊ", "Thành công sẽ luôn mỉm cười\nKiên trì tạo nên sự khác biệt"),
            ("BƯỚC ĐI TẠO NÊN HÀNH TRÌNH", "Mỗi ngày là một khởi đầu mới\nĐừng ngại vượt qua thử thách"),
            ("SÁNG TẠO KHÔNG GIỚI HẠN", "Tự tin khẳng định bản sắc\nVươn tới những đỉnh cao mới"),
            ("HẠNH PHÚC TỪ ĐIỀU GIẢN ĐƠN", "Trân trọng từng khoảnh khắc\nSống trọn vẹn mỗi phút giây"),
            ("LÒNG KIÊN TRÌ CHIẾN THẮNG", "Vượt qua mọi rào cản khó khăn\nChạm tay vào giấc mơ lớn"),
        ],
        "inverted": [
            ("HÀNH TRÌNH VẠN DẶM BẮT ĐẦU TỪ MỘT BƯỚC CHÂN KIÊN ĐỊNH ĐẦU TIÊN\nHÃY TIN TƯỞNG VÀO CHÍNH BẢN THÂN BẠN", "ĐAM MÊ"),
            ("SỰ KHÁC BIỆT DUY NHẤT GIỮA NGƯỜI THÀNH CÔNG VÀ KẺ THẤT BẠI LÀ LÒNG BỀN BỈ\nĐỪNG BAO GIỜ TỪ BỎ", "KHÁT VỌNG"),
            ("MỖI BUỔI SÁNG THỨC DẬY LÀ MỘT CƠ HỘI MỚI ĐỂ BẠN HOÀN THIỆN CHÍNH MÌNH\nHÃY MỈM CƯỜI ĐÓN NHẬN", "TỰ TIN"),
        ],
    },
    "opening_banner": {
        "standard": [
            ("TƯNG BỪNG KHAI TRƯƠNG", "Giảm giá 30% toàn bộ dịch vụ\nChào đón khách hàng tuần đầu tiên"),
            ("ĐẠI TIỆC MỞ BÁN", "Quà tặng đặc biệt cho 100 khách đầu tiên\nCơ hội trúng thưởng hấp dẫn"),
            ("RA MẮT KHÔNG GIAN MỚI", "Trải nghiệm dịch vụ đẳng cấp\nƯu đãi độc quyền khai trương"),
            ("CHÀO ĐÓN CƠ SỞ MỚI", "Không gian sang trọng tiện nghi\nƯu đãi ngập tràn ngày mở cửa"),
        ],
        "inverted": [
            ("CHÀO ĐÓN KHÔNG GIAN MỚI HIỆN ĐẠI SANG TRỌNG VÀ ĐẲNG CẤP TRẢI NGHIỆM\nƯU ĐÃI ĐẶC BIỆT TUẦN ĐẦU KHAI TRƯƠNG", "GIẢM 50%"),
            ("ĐẠI TIỆC KHAI TRƯƠNG BÙNG NỔ VỚI HÀNG TRĂM QUÀ TẶNG MIỄN PHÍ\nCHÀO ĐÓN QUÝ KHÁCH", "CHECK-IN 0Đ"),
        ],
    },
    "customer_feedback": {
        "standard": [
            ("ĐÁNH GIÁ KHÁCH HÀNG", "Sản phẩm dùng rất tốt và ưng ý\nChất lượng vượt xa kỳ vọng"),
            ("CẢM NHẬN CHÂN THỰC", "Giao hàng nhanh đóng gói cẩn thận\nNhân viên tư vấn nhiệt tình chu đáo"),
            ("TRẢI NGHIỆM HÀI LÒNG", "Dùng rất êm và hiệu quả rõ rệt\nSẽ tiếp tục ủng hộ lâu dài"),
            ("NIỀM TIN TRỌN VẸN", "Độ hoàn thiện sản phẩm rất cao\nXứng đáng với từng đồng chi phí"),
        ],
        "inverted": [
            ("SỰ HÀI LÒNG VÀ NỤ CƯỜI CỦA KHÁCH HÀNG LÀ ĐỘNG LỰC LỚN NHẤT CỦA CHÚNG TÔI\nCAM KẾT CHẤT LƯỢNG HÀNG ĐẦU", "5 SAO"),
            ("CẢM ƠN QUÝ KHÁCH ĐÃ LUÔN TIN TƯỞNG VÀ ĐỒNG HÀNH SUỐT CHẶNG ĐƯỜNG VỪA QUA\nCHẤT LƯỢNG CHUẨN MỰC", "UY TÍN"),
        ],
    },
    "cultural_vietnam": {
        "standard": [
            ("PHỞ BÒ TÁI LĂN HÀ NỘI", "Hương vị truyền thống trăm năm\nNước dùng thanh ngọt đậm đà"),
            ("BÁNH MÌ KẸP THỊT NƯỚNG", "Vỏ bánh giòn rụm thơm lừng\nẨm thực đường phố trứ danh"),
            ("PHỐ CỔ HỘI AN HOÀNG HÔN", "Đèn lồng rực rỡ lung linh\nVẻ đẹp hoài niệm cổ kính"),
            ("ÁO DÀI HOA XUÂN", "Duyên dáng người con gái Việt\nTôn vinh nét đẹp truyền thống"),
            ("CÀ PHÊ TRỨNG HÀ THÀNH", "Lớp kem trứng béo ngậy mềm mịn\nHương vị độc đáo khó quên"),
        ],
        "inverted": [
            ("TINH HOA ẨM THỰC TRUYỀN THỐNG VIỆT NAM CHINH PHỤC BẠN BÈ NĂM CHÂU\nHƯƠNG VỊ ĐẬM ĐÀ BẢN SẮC DÂN TỘC", "PHỞ VIỆT"),
            ("NÉT ĐẸP CỔ KÍNH RÊU PHONG VÀ BÌNH YÊN BÊN DÒNG SÔNG HOÀI THƠ MỘNG\nDI SẢN VĂN HÓA", "HỘI AN"),
            ("CHIẾC ÁO DÀI THƯỚT THA TÔN VINH VẺ ĐẸP THANH TAO CỦA PHỤ NỮ VIỆT\nDUYÊN DÁNG ĐẤT TRỜI", "VIỆT NAM"),
        ],
    },
}
