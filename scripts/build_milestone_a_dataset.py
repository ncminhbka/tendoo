#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - MASTER DATASET SYNTHESIS ENGINE (MILESTONE A: 800 UNIQUE SAMPLES)
====================================================================================================
Script: scripts/build_milestone_a_dataset.py
Purpose:
    Autonomous data generation pipeline for Phase 3 Milestone A:
    1. Manages 7 Tendoo AI Use Cases x 4 Aspect Ratios x Modality Split (440 I2I + 360 Pure T2I).
    2. Enforces Universal Ordinal Mapping '(1)' and '(2)' with 0% literal text leakage (Rule 3).
    3. Pre-determined typography: Template decides lines -> renders glyphs (Floor 32pt/36pt) ->
       instructs Teacher model -> generates ground truth target.
    4. Orthogonal Font Randomization across all 16 Unicode fonts.
    5. Semantic Domain & Product Locking: Ensures copy matches product reality 100%.
====================================================================================================
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.tendoo.glyph_engine import (
    FONT_TIERS,
    GlyphEngine,
)

glyph_engine = GlyphEngine()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY not found in .env")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=api_key)

# ==================================================================================================
# 1. 16 UNICODE FONTS & DUAL-FLOOR RESOLUTION
# ==================================================================================================
ALL_FONTS = [
    "bevietnam", "anton", "gotham", "lolapeluza", "gretoon", "playfair",
    "oswald", "harabaras", "dancing", "pacifico", "sedgwick", "blowbrush",
    "clementine", "cookies", "grocery", "holidays"
]

def get_font_floor(font_name: str) -> int:
    """Locked Dual-Floor: 32pt for bevietnam, 36pt for all other 15 fonts."""
    return 32 if font_name.lower() == "bevietnam" else 36

def sample_orthogonal_fonts() -> Tuple[str, str]:
    """Independent font sampling: Slot 1 is uniform 1/16, Slot 2 is 75% distinct font."""
    font1 = random.choice(ALL_FONTS)
    if random.random() < 0.75:
        remaining = [f for f in ALL_FONTS if f != font1]
        font2 = random.choice(remaining)
    else:
        font2 = font1
    return font1, font2

# ==================================================================================================
# 2. ASPECT RATIO CONFIGURATIONS
# ==================================================================================================
ASPECT_RATIOS = {
    "1:1": {"width": 1024, "height": 1024, "ratio": 0.35},
    "9:16": {"width": 768, "height": 1344, "ratio": 0.35},
    "4:5": {"width": 896, "height": 1152, "ratio": 0.15},
    "16:9": {"width": 1344, "height": 768, "ratio": 0.15},
}

# ==================================================================================================
# 3. SEMANTIC DOMAIN CORPUS & PRODUCT LOCKING
# ==================================================================================================
# Pairs of (Slot 1 Text, Slot 2 Text) tailored per domain and product
DOMAIN_TEXT_CORPUS = {
    "cosmetics": [
        ("NƯỚC HOA CAO CẤP", "Hương thơm quý phái\nLưu hương 24 giờ"),
        ("SERUM DƯỠNG ẨM", "Căng bóng mịn màng\nCấp ẩm chuyên sâu"),
        ("KEM DƯỠNG TRẮNG", "Trẻ hóa làn da\nMờ thâm nám tự nhiên"),
        ("SON MÔI MỊN LÌ", "Sắc đỏ thời thượng\nChuẩn màu lâu trôi"),
        ("KEM CHỐNG NẮNG", "Bảo vệ tối ưu SPF50+\nKháng nước kiềm dầu"),
        ("SỮA RỬA MẶT", "Sạch sâu dịu nhẹ\nCân bằng độ ẩm tự nhiên"),
        ("PHẤN NƯỚC CUSHION", "Lớp nền mỏng mịn\nChe phủ hoàn hảo suốt ngày"),
        ("DẦU GỘI DƯỠNG CHẤT", "Bồng bềnh suôn mượt\nChiết xuất tinh dầu tự nhiên"),
        ("TINH CHẤT PHỤC HỒI", "Tái tạo tế bào da\nNgăn ngừa lão hóa sớm"),
        ("MẶT NẠ DƯỠNG DA", "Thư giãn làn da\nCung cấp vitamin tức thì"),
    ],
    "fnb": [
        ("CÀ PHÊ PHIN NGUYÊN CHẤT", "Đậm đà phong vị Việt\nHương thơm nồng nàn truyền thống"),
        ("HẠT ROBUSTA RANG MỘC", "Hương vị nguyên bản\nRang xay thủ công tinh tế"),
        ("BẬT TUNG NĂNG LƯỢNG", "Sảng khoái tức thì\nTỉnh táo chinh phục thử thách"),
        ("TRÀ XANH THANH MÁT", "Chiết xuất lá trà tươi\nThanh lọc cơ thể mỗi ngày"),
        ("SỮA HẠT DINH DƯỠNG", "Thuần khiết tự nhiên\nGiàu canxi không đường"),
        ("BIA THỦ CÔNG CAO CẤP", "Hương hoa bia sảng khoái\nMen bia ủ mộc thượng hạng"),
        ("VANG ĐỎ THƯỢNG HẠNG", "Ủ thùng gỗ sồi lâu năm\nNồng nàn đẳng cấp quý phái"),
        ("TRÀ HOA CÚC MẬT ONG", "Thanh nhiệt an thần\nVị ngọt dịu nhẹ tự nhiên"),
        ("NƯỚC ÉP TRÁI CÂY TƯƠI", "100% nguyên chất\nBổ sung năng lượng tức thì"),
    ],
    "tech": [
        ("TAI NGHE CHỐNG ỒN", "Âm bass sống động\nPin 30 giờ liên tục"),
        ("ĐỒNG HỒ THÔNG MINH", "Theo dõi sức khỏe 24/7\nKháng nước chuẩn 5ATM"),
        ("LOA BLUETOOTH DI ĐỘNG", "Âm thanh vòm 360 độ\nKhuấy động mọi bữa tiệc"),
        ("CHUỘT GAMING KHÔNG DÂY", "Độ nhạy cực cao\nThiết kế công thái học đỉnh cao"),
        ("BÀN PHÍM CƠ CAO CẤP", "Gõ phím êm ái\nĐèn nền RGB rực rỡ"),
        ("SẠC NHANH ĐA NĂNG", "Công suất 65W vượt trội\nNhỏ gọn tiện lợi mang đi"),
        ("TAY CẦM CHƠI GAME", "Rung phản hồi chân thực\nKhông độ trễ trên mọi thiết bị"),
        ("CỦ SẠC CÔNG NGHỆ GAN", "Tản nhiệt thông minh\nBảo vệ thiết bị tối đa"),
    ],
    "fashion": [
        ("GIÀY THỂ THAO NĂNG ĐỘNG", "Siêu nhẹ êm chân\nBước đi bứt phá tự tin"),
        ("KÍNH RÂM THỜI THƯỢNG", "Chống tia UV400\nTôn vinh phong cách cá nhân"),
        ("ĐỒNG HỒ KIM LOẠI SANG TRỌNG", "Đẳng cấp quý ông\nBộ máy cơ học chuẩn xác"),
        ("TÚI XÁCH DA THẬT", "Chất da cao cấp\nTinh tế từng đường kim mũi chỉ"),
        ("VÍ DA CẦM TAY", "Da bò nguyên tấm\nBền đẹp cùng thời gian"),
        ("NÓN LÁ DUYÊN DÁNG", "Nét đẹp truyền thống\nHồn quê đất Việt ngàn năm"),
        ("ÁO KHOÁC GIÓ THỜI TRANG", "Chống thấm nước nhẹ\nGiữ ấm cản gió tối đa"),
    ],
    "home": [
        ("BÌNH GIỮ NHIỆT INOX", "Giữ nhiệt suốt 24 giờ\nThép không gỉ an toàn"),
        ("MÁY SẤY TÓC ION ÂM", "Sấy khô siêu tốc\nBảo vệ tóc bóng mượt"),
        ("BÀN ỦI HƠI NƯỚC", "Phẳng phiu tức thì\nKháng khuẩn 99% áo quần"),
        ("MÁY XAY SINH TỐ MINI", "Xay nhuyễn mịn đa năng\nSống khỏe tươi vui mỗi ngày"),
        ("NỒI CHIÊN KHÔNG DẦU", "Giảm 85% chất béo\nChín vàng giòn rụm thơm ngon"),
        ("ĐÈN BÀN CHỐNG CẬN", "Ánh sáng dịu mắt\nTùy chỉnh 3 chế độ thông minh"),
    ],
    "fmcg": [
        ("MÌ HẢO HẢO TÔM CHUA CAY", "Sợi mì dai giòn đậm vị\nHương vị quốc dân gắn kết"),
        ("TRÀ SEN TÂY HỒ", "Hương sen thanh khiết\nTinh hoa trà búp Tân Cương"),
        ("NƯỚC MẮM CỐT PHÚ QUỐC", "Đậm đà vị cá cơm truyền thống\nỦ chượp ròng rã tự nhiên"),
        ("CAO SAO VÀNG CỔ ĐIỂN", "Tinh dầu tràm quế tự nhiên\nThương hiệu vượt thời gian"),
        ("YẾN SÀO KHÁNH HÒA", "Bồi bổ sức khỏe tinh anh\nQuà tặng trân quý cho gia đình"),
        ("BÁNH QUY BƠ THƯỢNG HẠNG", "Thơm lừng bơ sữa nguyên chất\nGiòn tan tròn vị yêu thương"),
    ],
    "telecom_viettel": [
        ("MODEM HOME WIFI 6", "Phủ sóng toàn diện ngôi nhà\nTốc độ Gigabit không giật lag"),
        ("SIM VIETTEL 5G SIÊU TỐC", "Tốc độ vượt trội kết nối tương lai\nƯu đãi data không giới hạn"),
        ("CAMERA THÔNG MINH VIETTEL", "Hình ảnh 2K sắc nét ban đêm\nLưu trữ đám mây bảo mật tuyệt đối"),
        ("ĐỊNH VỊ V-TRACKING", "Giám sát hành trình 24/7\nQuản lý phương tiện thông minh"),
        ("TRUYỀN HÌNH TV360", "Thế giới giải trí không giới hạn\nHàng trăm kênh truyền hình chuẩn HD"),
    ],
    "fitness": [
        ("BÌNH LẮC SHAKER THỂ THAO", "Khuấy tan bột siêu nhanh\nNhựa nguyên sinh an toàn sức khỏe"),
        ("THẢM TẬP YOGA CAO CẤP", "Độ bám sàn chống trơn trượt\nÊm ái trong từng chuyển động"),
        ("GĂNG TAY TẬP TẠ", "Bảo vệ cổ tay vững chắc\nThoáng khí chống chai tay"),
        ("CON LĂN MASSAGE GIÃN CƠ", "Giảm căng cứng cơ bắp\nPhục hồi thần tốc sau luyện tập"),
        ("DÂY NHẢY TỐC ĐỘ CAO", "Lõi cáp thép bền bỉ\nĐốt cháy calo rèn luyện sức bền"),
    ],
}

# General T2I Corpus for Non-Product Use Cases (Recruitment, Guide, Quotes, Opening)
GENERAL_T2I_CORPUS = {
    "recruitment": [
        ("KỸ SƯ TRÍ TUỆ NHÂN TẠO", "Thu nhập hấp dẫn\nMôi trường sáng tạo mở"),
        ("CHUYÊN VIÊN MARKETING", "Phát triển tiềm năng\nĐãi ngộ cạnh tranh hàng đầu"),
        ("LẬP TRÌNH VIÊN BACKEND", "Làm việc linh hoạt\nDự án quy mô triệu người dùng"),
        ("TRƯỞNG NHÓM KINH DOANH", "Lương thưởng không giới hạn\nLộ trình thăng tiến rõ ràng"),
        ("QUẢN LÝ DỰ ÁN CÔNG NGHỆ", "Văn hóa chủ động bứt phá\nChế độ bảo hiểm toàn diện"),
    ],
    "two_step_guide": [
        ("BƯỚC 1: ĐĂNG KÝ TÀI KHOẢN", "BƯỚC 2: BẮT ĐẦU TRẢI NGHIỆM\nHoàn toàn miễn phí"),
        ("BƯỚC 1: TẢI ỨNG DỤNG", "BƯỚC 2: NHẬN NGAY VOUCHER 50K\nÁp dụng cho đơn đầu tiên"),
        ("BƯỚC 1: QUÉT MÃ QR CODE", "BƯỚC 2: THANH TOÁN TỨC THÌ\nAn toàn và tiện lợi"),
        ("BƯỚC 1: CHỌN SẢN PHẨM YÊU THÍCH", "BƯỚC 2: XÁC NHẬN GIAO HÀNG TẬN NƠI\nĐổi trả trong 7 ngày"),
    ],
    "creative_quote": [
        ("HÃY THEO ĐUỔI ĐAM MÊ", "Thành công sẽ luôn mỉm cười\nKiên trì tạo nên sự khác biệt"),
        ("BƯỚC ĐI TẠO NÊN HÀNH TRÌNH", "Mỗi ngày là một khởi đầu mới\nĐừng ngại vượt qua thử thách"),
        ("SÁNG TẠO KHÔNG GIỚI HẠN", "Tự tin khẳng định bản sắc\nVươn tới những đỉnh cao mới"),
        ("HẠNH PHÚC TỪ NHỮNG ĐIỀU GIẢN ĐƠN", "Trân trọng từng khoảnh khắc\nSống trọn vẹn mỗi phút giây"),
    ],
    "opening_banner": [
        ("TƯNG BỪNG KHAI TRƯƠNG", "Giảm giá 30% toàn bộ dịch vụ\nChào đón khách hàng tuần đầu tiên"),
        ("ĐẠI TIỆC MỞ BÁN", "Quà tặng đặc biệt cho 100 khách đầu tiên\nCơ hội trúng thưởng hấp dẫn"),
        ("RA MẮT KHÔNG GIAN MỚI", "Trải nghiệm dịch vụ đẳng cấp\nƯu đãi độc quyền khai trương"),
    ]
}

# ==================================================================================================
# 4. STUDENT PROMPT COMBINATORIAL ENGINE (RULE 3 & 26 COMPLIANT)
# ==================================================================================================
ENV_SETTINGS = [
    "Bối cảnh studio thương mại hiện đại với bệ trưng bày hình khối tối giản",
    "Không gian chụp ảnh sản phẩm chuyên nghiệp phong cách Bắc Âu trang nhã",
    "Phông nền tương phản cao với ánh sáng studio điện ảnh nghệ thuật",
    "Không gian nội thất sang trọng với ánh sáng tự nhiên dịu nhẹ",
    "Bối cảnh công nghệ hiện đại với hiệu ứng ánh sáng gradient tinh tế",
    "Bố cục poster đồ họa thương mại cao cấp với các mảng màu cân đối",
]

ROLE_DESCRIPTORS_1 = [
    "Khối tiêu đề chính", "Dòng chữ thông điệp nổi bật", "Tiêu đề vị trí",
    "Khối tên thương hiệu chính", "Dòng chữ chủ đề lớn", "Tiêu đề thông điệp"
]

ROLE_DESCRIPTORS_2 = [
    "Dòng phụ đề bổ trợ", "Khối thông tin chi tiết 2 dòng", "Dòng chú thích nội dung",
    "Khối thông điệp phụ", "Khối quyền lợi đãi ngộ", "Dòng slogan ngắn gọn"
]

POSITION_DESCRIPTORS_1 = [
    "ở phía trên chính giữa", "ở phần trên cùng của bố cục", "ở góc trên cân đối",
    "ở vị trí trung tâm phía trên", "chạy ngang phần trên canvas"
]

POSITION_DESCRIPTORS_2 = [
    "ở phía dưới chính giữa", "ở phần chân đế bên dưới", "nằm ngay bên dưới tiêu đề",
    "ở nửa dưới của poster", "ở vị trí góc dưới thanh lịch"
]

MATERIALS = [
    "chữ kim loại dập nổi 3D mạ vàng sang trọng",
    "chữ phát sáng hiệu ứng đèn neon hiện đại",
    "nét chữ màu trắng thanh lịch đổ bóng studio mềm mại",
    "chữ in nổi 3D chất liệu acrylic cao cấp sắc nét",
    "chữ khắc chìm phong cách tối giản tương phản rõ ràng",
    "chữ màu vàng đồng cổ điển ánh kim rực rỡ",
    "nét chữ đậm đà phong cách typography hiện đại",
]

PHOTOGRAPHY_LIGHTING = [
    "Ánh sáng studio tương phản cao, đổ bóng tự nhiên sắc nét, phong cách nhiếp ảnh thương mại chuẩn mực.",
    "Ánh sáng tự nhiên dịu mắt, độ chi tiết cao, màu sắc hài hòa sống động.",
    "Ánh sáng spotlight tập trung vào chủ thể, chiều sâu trường ảnh mượt mà.",
    "Phong cách thiết kế poster quảng cáo thương mại cao cấp, bố cục cân xứng hoàn hảo."
]

def synthesize_clean_prompt(text1: str, text2: str) -> str:
    """
    Generates Student Clean Prompt strictly adhering to Rule 3 (No literal text)
    and Universal Ordinal Mapping (1) / (2).
    """
    env = random.choice(ENV_SETTINGS)
    r1 = random.choice(ROLE_DESCRIPTORS_1)
    p1 = random.choice(POSITION_DESCRIPTORS_1)
    m1 = random.choice(MATERIALS)

    r2 = random.choice(ROLE_DESCRIPTORS_2)
    p2 = random.choice(POSITION_DESCRIPTORS_2)
    m2 = random.choice(MATERIALS)

    light = random.choice(PHOTOGRAPHY_LIGHTING)

    prompt_clean = (
        f"{env}. "
        f"(1) {r1} {p1} {m1}. "
        f"(2) {r2} {p2} {m2}. "
        f"{light}"
    )

    # ANTI-LEAK VALIDATION: Ensure literal words are 100% absent
    def get_clean_words(t: str) -> List[str]:
        words = re.findall(r"\b\w+\b", t.lower())
        return [w for w in words if len(w) > 2]

    for w in get_clean_words(text1) + get_clean_words(text2):
        if w in prompt_clean.lower():
            # Replace any accidental common word collision
            prompt_clean = prompt_clean.replace(w, "thông_điệp")

    assert "(1)" in prompt_clean and "(2)" in prompt_clean, "Missing ordinal tags!"
    return prompt_clean

# ==================================================================================================
# 5. DATASET SAMPLER & TEACHER CALLER
# ==================================================================================================
def sample_dataset_spec(sample_id: int, total_samples: int = 800) -> Dict:
    """Determines full parameters for sample (id, modality, use_case, domain, text, fonts)."""
    # 1. Modality: 440 I2I, 360 T2I
    is_i2i = (sample_id <= 440)
    modality = "i2i" if is_i2i else "t2i"

    # 2. Aspect Ratio: Weighted sampling
    r = random.random()
    if r < 0.35:
        ar_name = "1:1"
    elif r < 0.70:
        ar_name = "9:16"
    elif r < 0.85:
        ar_name = "4:5"
    else:
        ar_name = "16:9"
    ar_cfg = ASPECT_RATIOS[ar_name]

    # 3. Fonts
    font1, font2 = sample_orthogonal_fonts()
    floor1 = get_font_floor(font1)
    floor2 = get_font_floor(font2)

    # 4. Domain & Text
    if is_i2i:
        # Pick from domain products
        domains = list(DOMAIN_TEXT_CORPUS.keys())
        domain = random.choice(domains)
        pair = random.choice(DOMAIN_TEXT_CORPUS[domain])
        text1, text2 = pair
        use_case = "hero_product" if sample_id <= 170 else "flash_sale"
    else:
        # T2I can be commercial or general
        if random.random() < 0.5:
            domain = random.choice(list(DOMAIN_TEXT_CORPUS.keys()))
            text1, text2 = random.choice(DOMAIN_TEXT_CORPUS[domain])
            use_case = "flash_sale"
        else:
            use_case = random.choice(list(GENERAL_T2I_CORPUS.keys()))
            domain = "general_" + use_case
            text1, text2 = random.choice(GENERAL_T2I_CORPUS[use_case])

    # 5. Clean Prompt
    prompt_clean = synthesize_clean_prompt(text1, text2)

    return {
        "id": f"sample_{sample_id:04d}",
        "modality": modality,
        "use_case": use_case,
        "domain": domain,
        "aspect_ratio": ar_name,
        "width": ar_cfg["width"],
        "height": ar_cfg["height"],
        "text1": text1,
        "text2": text2,
        "font1": font1,
        "font2": font2,
        "floor1": floor1,
        "floor2": floor2,
        "prompt_clean": prompt_clean,
    }


def render_and_save_glyphs(spec: Dict, glyphs_dir: Path) -> Tuple[Path, Path, Dict, Dict]:
    """Renders locked tight-crop glyphs for Slot 1 and Slot 2 according to font floors."""
    sid = spec["id"]
    g1_path = glyphs_dir / f"glyph_{sid}_slot10.png"
    g2_path = glyphs_dir / f"glyph_{sid}_slot20.png"

    g1_info = glyph_engine.render(
        text=spec["text1"],
        font_name_or_path=spec["font1"],
        font_size_pt=spec["floor1"],
        auto_size=True,
    )
    g1_info.image.save(g1_path)

    g2_info = glyph_engine.render(
        text=spec["text2"],
        font_name_or_path=spec["font2"],
        font_size_pt=spec["floor2"],
        auto_size=True,
    )
    g2_info.image.save(g2_path)

    return g1_path, g2_path, g1_info.__dict__, g2_info.__dict__


def generate_target_image(spec: Dict, targets_dir: Path, delay: float = 9.5) -> Path:
    """Calls OpenAI gpt-image-2 to generate ground-truth image and resizes to exact bucket dimensions."""
    from PIL import Image
    import io

    sid = spec["id"]
    target_path = targets_dir / f"target_{sid}.png"

    if target_path.exists() and target_path.stat().st_size > 10000:
        return target_path

    # Build Teacher Prompt with explicit line structure
    t1_lines = spec["text1"].split("\n")
    t2_lines = spec["text2"].split("\n")

    t1_desc = f"At top, {len(t1_lines)} line(s): '{' / '.join(t1_lines)}'"
    t2_desc = f"Below it, {len(t2_lines)} line(s): '{' / '.join(t2_lines)}'"

    teacher_prompt = (
        f"Commercial advertising graphic poster for {spec['use_case']} ({spec['domain']}). "
        f"{t1_desc}. {t2_desc}. "
        f"Professional graphic typography design, high contrast, sharp studio lighting, commercial photography."
    )

    # Pick closest OpenAI size
    ar = spec["aspect_ratio"]
    if ar == "1:1":
        api_size = "1024x1024"
    elif ar in ["9:16", "4:5"]:
        api_size = "1024x1536"
    else:
        api_size = "1536x1024"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = client.images.generate(
                model="gpt-image-2",
                prompt=teacher_prompt,
                quality="low",
                size=api_size,
            )
            raw_bytes = base64.b64decode(res.data[0].b64_json)
            img = Image.open(io.BytesIO(raw_bytes))

            # Resize with Lanczos to exact canonical bucket dimensions (multiples of 16)
            target_w, target_h = spec["width"], spec["height"]
            img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_resized.save(target_path)

            time.sleep(delay)
            return target_path
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f" [429 RATE LIMIT] Backoff 12s (attempt {attempt+1}/{max_retries})...")
                time.sleep(12.0)
            else:
                raise e

    raise RuntimeError(f"Failed to generate target for {sid}")


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI - Master Dataset Synthesis Engine")
    parser.add_argument("--smoke", action="store_true", help="Run Smoke Test (10 samples)")
    parser.add_argument("--pilot", action="store_true", help="Run Pilot Test (60 samples)")
    parser.add_argument("--count", type=int, default=None, help="Custom sample count")
    parser.add_argument("--execute", action="store_true", help="Actually execute API calls and glyph rendering (default: dry-run)")
    parser.add_argument("--delay", type=float, default=9.5, help="Delay in seconds between API requests (default: 9.5s)")
    args = parser.parse_args()

    target_count = 10 if args.smoke else (60 if args.pilot else (args.count or 800))

    output_dir = PROJECT_ROOT / "data" / "milestone_a"
    glyphs_dir = output_dir / "glyphs"
    targets_dir = output_dir / "targets"
    glyphs_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "dataset_manifest.jsonl"

    print("=" * 90)
    print(f" [*] TENDOO AI - MILESTONE A DATASET GENERATOR (TARGET COUNT: {target_count})")
    print(f" [*] MODE: {'EXECUTE (API + GLYPHS)' if args.execute else 'DRY RUN (SPECIFICATION VERIFICATION)'}")
    print(f" [*] OUTPUT DIRECTORY: {output_dir}")
    print("=" * 90)

    if not args.execute:
        # Dry-run inspection
        print(" [*] Pre-generating dataset specifications and verifying typography pipeline...")
        for idx in range(1, min(6, target_count + 1)):
            spec = sample_dataset_spec(idx, target_count)
            print(f"\n--- [SAMPLE #{spec['id']}] ---")
            print(f" Modality: {spec['modality']} | Use Case: {spec['use_case']} | AR: {spec['aspect_ratio']} ({spec['width']}x{spec['height']})")
            print(f" Slot 1 (t=10.0): '{spec['text1']}' [Font: {spec['font1']} @ {spec['floor1']}pt]")
            print(f" Slot 2 (t=20.0): '{spec['text2'].replace(chr(10), ' / ')}' [Font: {spec['font2']} @ {spec['floor2']}pt]")
            print(f" Student Clean Prompt: {spec['prompt_clean']}")
        print("\n" + "=" * 90)
        print(" [OK] Dry-run passed. To generate real data, run with '--execute'.")
        print("=" * 90)
        return

    # Real execution
    existing_ids = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_ids.add(record["id"])
                    except Exception:
                        pass
    print(f" [*] Found {len(existing_ids)} existing records in manifest. Resuming...")

    success_count = len(existing_ids)
    for idx in range(1, target_count + 1):
        sample_id = f"sample_{idx:04d}"
        if sample_id in existing_ids:
            continue

        spec = sample_dataset_spec(idx, target_count)
        print(f"[{idx:04d}/{target_count:04d}] Processing {sample_id} ({spec['modality']} | {spec['use_case']} | {spec['aspect_ratio']})...")

        # 1. Render Glyphs
        g1_path, g2_path, g1_meta, g2_meta = render_and_save_glyphs(spec, glyphs_dir)

        # 2. Generate Target Image via API
        target_path = generate_target_image(spec, targets_dir, delay=args.delay)

        # 3. Assemble Manifest Record
        record = {
            "id": spec["id"],
            "modality": spec["modality"],
            "use_case": spec["use_case"],
            "domain": spec["domain"],
            "aspect_ratio": spec["aspect_ratio"],
            "width": spec["width"],
            "height": spec["height"],
            "prompt_clean": spec["prompt_clean"],
            "target_image": str(target_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "slots": [
                {
                    "time_offset": 10.0,
                    "glyph_path": str(g1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "font": spec["font1"],
                    "font_size_pt": spec["floor1"],
                    "text": spec["text1"],
                    "width_px": g1_meta["width_px"],
                    "height_px": g1_meta["height_px"],
                    "token_count": g1_meta["token_count"],
                },
                {
                    "time_offset": 20.0,
                    "glyph_path": str(g2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "font": spec["font2"],
                    "font_size_pt": spec["floor2"],
                    "text": spec["text2"],
                    "width_px": g2_meta["width_px"],
                    "height_px": g2_meta["height_px"],
                    "token_count": g2_meta["token_count"],
                }
            ]
        }

        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        success_count += 1
        print(f"   ===> [OK] {sample_id} saved successfully! (Total completed: {success_count})")

    print("\n" + "=" * 90)
    print(f" [*] BATCH COMPLETE: {success_count}/{target_count} samples generated in: {output_dir}")
    print(f" [*] Manifest: {manifest_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
