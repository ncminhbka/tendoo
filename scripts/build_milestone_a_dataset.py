#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - MASTER DATASET SYNTHESIS ENGINE (MILESTONE A: 800 UNIQUE SAMPLES)
====================================================================================================
Script: scripts/build_milestone_a_dataset.py

SLOT ARCHITECTURE (corrected per clarification):
    - Text1 (title)    @ t=10.0  -> ALWAYS present (I2I and T2I)
    - Text2 (subtitle) @ t=20.0  -> ALWAYS present (I2I and T2I)
    - Product ref      @ t=30.0  -> I2I ONLY (single-ref product placement is already reliable
                                      at any t-slot per prior probing; it rides "for free" and is
                                      NOT the thing Milestone A is trying to unlock)
    => Milestone A's actual learning target in both modalities is the SAME: reliable concurrent
       2-slot TEXT rendering (t=10 + t=20). I2I just has one extra, already-solved, slot on top.

Key fixes vs. the previous draft:
    1. total_samples was ignored -> --smoke/--pilot silently only ever produced i2i/hero_product
       samples. Now every ratio-based branch is computed from `total_samples`, not hardcoded 800/440/170.
    2. I2I samples used to leak an uncontioned `text2` into the teacher prompt (no glyph, no
       description backing it) -> ground-truth had "invisible" text the model had no signal for.
       Now I2I always renders BOTH glyphs, same as T2I; the only I2I-specific addition is the
       product slot.
    3. Product<->text pairing was implicit/positional (fragile: reorder or add a file and every
       downstream pairing silently shifts). Now explicit dict keyed by filename stem.
    4. I2I ground-truth target now generated via images.edit() with the REAL product photo as
       input reference, so the "ground truth" the model is trained to reproduce actually contains
       the same product pixels that will condition FLUX at t=30 (previously the teacher only ever
       saw a text description of the product and hallucinated its own version -> identity mismatch
       between training target and training reference).
    5. Known-hard stress pairs now carry an explicit, semantically sane domain instead of a fully
       random one.
    6. Added an explicit held-out split (`split: train|val`) so Milestone A produces something you
       can actually evaluate generalization against, not just training loss.
    7. Optional LLM-authored student prompts (--llm-prompts) for phrasing diversity beyond the
       fixed combinatorial word lists, with the anti-leak assertion still enforced as a hard gate
       (LLM proposes, code disposes: on validation failure it retries, then falls back to the
       deterministic combinatorial builder).
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
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.tendoo.glyph_engine import GlyphEngine  # noqa: E402

glyph_engine = GlyphEngine()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY not found in .env")
    sys.exit(1)

from openai import OpenAI  # noqa: E402

client = OpenAI(api_key=api_key)

# ==================================================================================================
# 1. TIME-OFFSET SLOT CONSTANTS
# ==================================================================================================
T_TEXT1 = 10.0
T_TEXT2 = 20.0
T_PRODUCT = 30.0  # I2I only; already reliable per prior probing, not part of what we're training

# ==================================================================================================
# 2. FONTS (7 x 100% OFL, COMMERCIAL-SAFE) & DUAL-FLOOR RESOLUTION
# ==================================================================================================
ALL_FONTS = [
    "bevietnam",  # Clean Modern Geometric Sans (OFL) - Floor 32pt
    "anton",      # Heavy Condensed Bold Display (OFL) - Floor 36pt
    "playfair",   # Elegant High-Contrast Serif (OFL) - Floor 36pt
    "oswald",     # Gothic Condensed Display Sans (OFL) - Floor 36pt
    "pacifico",   # Casual Fun Brush Script (OFL) - Floor 36pt
    "dancing",    # Dynamic Cursive Script (OFL) - Floor 36pt
    "sedgwick",   # Street Urban Marker / Graffiti (OFL) - Floor 36pt
]


def get_font_floor(font_name: str) -> int:
    return 32 if font_name.lower() == "bevietnam" else 36


def sample_orthogonal_fonts() -> Tuple[str, str]:
    font1 = random.choice(ALL_FONTS)
    if random.random() < 0.75:
        font2 = random.choice([f for f in ALL_FONTS if f != font1])
    else:
        font2 = font1
    return font1, font2


# ==================================================================================================
# 3. ASPECT RATIO BUCKETS (direct 1:1 pixel sizes for gpt-image-2, no resize needed)
# ==================================================================================================
ASPECT_RATIOS = {
    "1:1": {"width": 1024, "height": 1024, "weight": 0.35},
    "9:16": {"width": 768, "height": 1344, "weight": 0.35},
    "4:5": {"width": 896, "height": 1152, "weight": 0.15},
    "16:9": {"width": 1344, "height": 768, "weight": 0.15},
}
# Exact 20-sample balanced cycle matching 35% 1:1, 35% 9:16, 15% 4:5, 15% 16:9 per Sub-plan Table 1.2
# Evenly interleaved so aspect ratios are uniformly distributed across domains & use-cases.
AR_CYCLE_20: List[str] = [
    "1:1", "9:16", "4:5", "1:1", "9:16", "16:9", "1:1", "9:16", "1:1", "9:16",
    "4:5", "1:1", "9:16", "16:9", "1:1", "9:16", "4:5", "1:1", "9:16", "16:9",
]


def determine_aspect_ratio(sample_id: int, total_samples: int, is_i2i: bool) -> str:
    i2i_cutoff = round(total_samples * 0.55)
    rel_idx = (sample_id - 1) if is_i2i else (sample_id - i2i_cutoff - 1)
    return AR_CYCLE_20[rel_idx % len(AR_CYCLE_20)]


# ==================================================================================================
# 4. PRODUCT <-> TEXT PAIRING & GENERAL T2I CORPUS (HIGH-DIVERSITY MULTI-ANGLE & STRATIFIED)
# ==================================================================================================
try:
    from scripts.corpus_milestone_a import PRODUCT_TEXT_CORPUS, GENERAL_T2I_CORPUS
except ImportError:
    from corpus_milestone_a import PRODUCT_TEXT_CORPUS, GENERAL_T2I_CORPUS


# Known-hard concurrency stress pairs covering 4 empirical pain points from sub-plan 1.6:
# 1. Diacritic Cluster Stress (3-4 dense tone marks in a row)
# 2. Extreme Asymmetric Token Mass (product 4096 tokens vs tiny text badge)
# 3. Zero Surface Anchor (abstract/floating 3D text without cue words)
# 4. Boundary Coordinates on extreme aspect ratios (9:16 & 16:9)
KNOWN_HARD_PAIRS = [
    # 1. Diacritic Clusters
    {"text1": "CHỐNG ỒN CHỦ ĐỘNG", "text2": "Khử tạp âm kỹ thuật số\nĐắm chìm trong âm nhạc đỉnh cao", "domain": "tech"},
    {"text1": "Ủ CHƯỢP TRUYỀN THỐNG", "text2": "Cá cơm tươi nguyên chất\nĐậm đà phong vị biển xanh", "domain": "fmcg"},
    {"text1": "ĐỔI MỚI SÁNG TẠO TOÀN DIỆN", "text2": "Bứt phá mọi giới hạn\nĐịnh hình kỷ nguyên số", "domain": "telecom_viettel"},
    {"text1": "BỘT GIẶT ĐẬM ĐẶC BẢO VỆ MÀU VẢI", "text2": "Đánh bay vết bẩn cứng đầu\nLưu hương thơm mát suốt ngày dài", "domain": "home"},
    {"text1": "KHỞI ĐẦU ĐỔI MỚI PHÁT TRIỂN", "text2": "Kiến tạo tương lai số\nNâng tầm vị thế doanh nghiệp", "domain": "telecom_viettel"},
    {"text1": "NƯỚC MẮM CỐT ĐẶC SẢN NGUYÊN CHẤT", "text2": "Ủ chượp tự nhiên từ cá cơm than\nHương vị truyền thống trăm năm", "domain": "fmcg"},
    # 2. Extreme Asymmetric Token Mass
    {"text1": "TIỆM CÀ PHÊ ANH QUÂN GÓC PHỐ NHỎ BÌNH YÊN", "text2": "GIẢM 50%", "domain": "fnb"},
    {"text1": "BỘ DƯỠNG TRẮNG PHỤC HỒI TÁI TẠO LÀN DA CHUYÊN SÂU", "text2": "HOT SALE", "domain": "cosmetics"},
    {"text1": "GIẢI PHÁP KẾT NỐI KHÔNG DÂY TỐC ĐỘ CAO CHO MỌI NGÔI NHÀ", "text2": "0 ĐỒNG", "domain": "telecom_viettel"},
    {"text1": "DÒNG ĐỒNG HỒ KIM LOẠI CAO CẤP CHỐNG NƯỚC VƯỢT TRỘI", "text2": "5 SAO", "domain": "fashion"},
    {"text1": "CHƯƠNG TRÌNH KHUYẾN MẠI MÙA HÈ BÙNG NỔ NĂNG LƯỢNG", "text2": "SALE", "domain": "fitness"},
    # 3. Zero Surface Anchor (floating 3D text without physical cue words)
    {"text1": "SỨC MẠNH VÔ HÌNH", "text2": "Đánh thức tiềm năng vô hạn\nVượt qua mọi giới hạn bản thân", "domain": "fitness"},
    {"text1": "KHÔNG GIAN VÔ CỰC", "text2": "Âm thanh lan tỏa đa chiều\nCảm xúc thăng hoa bất tận", "domain": "tech"},
    {"text1": "TỰ DO BỨT PHÁ", "text2": "Làm chủ hành trình cuộc đời\nTự tin khẳng định phong cách", "domain": "fashion"},
    {"text1": "ÁNH SÁNG TƯƠNG LAI", "text2": "Công nghệ dẫn đầu xu thế\nTrải nghiệm đỉnh cao mỗi ngày", "domain": "tech"},
    # 4. Boundary Coordinates on extreme aspect ratios (9:16 & 16:9)
    {"text1": "ĐỈNH CAO THIẾT KẾ ĐỒ HỌA SỐNG ĐỘNG", "text2": "ĐẲNG CẤP THƯƠNG HIỆU QUỐC TẾ\nKHẲNG ĐỊNH VỊ THẾ DẪN ĐẦU THỊ TRƯỜNG", "domain": "telecom_viettel"},
    {"text1": "TRẢI NGHIỆM ĐIỆN ẢNH ĐỈNH CAO TRONG TẦM TAY", "text2": "HÀNG NGÀN BỘ PHIM BOM TẤN\nCẬP NHẬT LIÊN TỤC MỖI NGÀY", "domain": "telecom_viettel"},
    {"text1": "BỨT PHÁ MỌI GIỚI HẠN TỐC ĐỘ VẬN ĐỘNG", "text2": "RÈN LUYỆN THỂ LỰC BỀN BỈ\nCHINH PHỤC ĐỈNH CAO DANH VỌNG", "domain": "fitness"},
    {"text1": "HƯƠNG VỊ THANH TAO TỪ THIÊN NHIÊN HOANG SƠ", "text2": "CHẮT LỌC TINH TÚY ĐẤT TRỜI\nAN LÀNH CHO SỨC KHỎE GIA ĐÌNH", "domain": "fnb"},
    {"text1": "KHOẢNH KHẮC THĂNG HOA CÙNG ĐAM MÊ BẤT TẬN", "text2": "ĐỒNG HÀNH TRÊN MỌI CUNG ĐƯỜNG\nTỰ HÀO BẢN SẮC VIỆT NAM", "domain": "fashion"},
]

# Detailed typography styling instructions passed to Teacher (gpt-image-2)
# ensuring the generated poster's visual font geometry matches the VAE glyph bitmap 1:1.
FONT_STYLE_DESCRIPTORS = {
    "bevietnam": "in clean, ultra-legible modern geometric sans-serif lettering (Be Vietnam Pro style), with uniform stroke weight",
    "anton": "in massive, heavy bold condensed sans-serif display lettering (Anton style), tall impactful block letters",
    "playfair": "in high-contrast luxury editorial serif lettering (Playfair Display style), with delicate thin serifs and thick vertical stems",
    "oswald": "in tall, narrow condensed gothic sans-serif lettering (Oswald style), structured modern poster lettering",
    "pacifico": "in flowing casual retro brush script calligraphy (Pacifico style), rounded cursive strokes with organic handmade feel",
    "dancing": "in elegant dynamic cursive script calligraphy (Dancing Script style), fluid handwriting with lively bouncy ascenders",
    "sedgwick": "in expressive urban street graffiti marker lettering (Sedgwick Ave style), edgy handcrafted raw display strokes",
}

# ==================================================================================================
# 5. STUDENT CLEAN-PROMPT BUILDER (RULE: never leak literal text; keep (1)/(2)/(3) spatial anchors)
# ==================================================================================================
# ==================================================================================================
# 5. DOMAIN-AWARE VISUAL CONTEXTS & CLEAN-PROMPT BUILDER (NO "CỌC CẠCH" MISMATCHES)
# ==================================================================================================
DOMAIN_VISUAL_CONTEXTS: Dict[str, Dict[str, List[str]]] = {
    "fitness": {
        "seeds": [
            "Phong cách phòng tập thể thao cao cấp, ánh sáng spotlight tương phản mạnh mẽ, phông nền bê tông mài và kim loại tối giản.",
            "Phong cách thể thao năng động bùng nổ, ánh sáng góc nghiêng mạnh mẽ, tạo bóng đổ dứt khoát tôn vinh tinh thần vận động.",
            "Không gian tập luyện chuyên nghiệp hiện đại, tone màu xám đậm thể thao, ánh sáng studio sắc sảo.",
        ],
        "envs": [
            "Bối cảnh studio thể thao hiện đại với phông nền bê tông mài tối giản",
            "Không gian phòng tập cao cấp với ánh sáng spotlight tương phản mạnh",
            "Phông nền xám đen mờ sang trọng tôn vinh tối đa tinh thần thể thao năng động",
        ],
        "materials_1": [
            "chữ kim loại dập nổi 3D mạ vàng ánh kim mạnh mẽ",
            "nét chữ typography đậm đà phong cách thể thao hiện đại dập chìm",
            "chữ kim loại chrome bạc phản chiếu ánh sáng studio sắc nét",
        ],
        "materials_2": [
            "nét chữ màu trắng thanh lịch đổ bóng studio sắc nét",
            "chữ decal mờ tinh giản với độ tương phản sắc nét",
            "dòng chữ viền led phát sáng dịu mắt tạo chiều sâu không gian",
        ],
    },
    "cosmetics": {
        "seeds": [
            "Phong cách studio mỹ phẩm cao cấp với bệ đá cẩm thạch trắng, ánh sáng softbox khuếch tán dịu nhẹ, tone màu pastel thanh khiết.",
            "Phong cách chăm sóc sắc đẹp spa sang trọng, ánh sáng tự nhiên tinh khôi tôn vinh sự tươi mới.",
            "Phong cách tạp chí làm đẹp quốc tế, tương phản mềm mại, bóng đổ mờ ảo mang lại cảm giác dịu dàng quý phái.",
        ],
        "envs": [
            "Không gian chụp ảnh mỹ phẩm chuyên nghiệp phong cách Bắc Âu trang nhã",
            "Bối cảnh studio spa sang trọng với bệ đá cẩm thạch trắng tinh khôi",
            "Phông nền tone màu pastel dịu nhẹ với ánh sáng tự nhiên êm dịu",
        ],
        "materials_1": [
            "chữ in nổi chất liệu acrylic cao cấp trong suốt bóng bẩy",
            "chữ kim loại dập nổi 3D mạ vàng hồng ánh kim sang trọng",
            "chữ vàng đồng cổ điển ánh kim dập nổi tương phản cao",
        ],
        "materials_2": [
            "nét chữ màu trắng thanh lịch đổ bóng studio mềm mại",
            "dòng chữ màu vàng nhạt tinh tế hài hòa với bố cục",
            "chữ decal mờ tinh giản với độ tương phản sắc nét",
        ],
    },
    "tech": {
        "seeds": [
            "Phong cách công nghệ hiện đại với bệ trưng bày hình khối mạ chrome tối giản, ánh sáng tương phản điện ảnh sắc sảo.",
            "Phong cách Cyberpunk tương lai với ánh sáng neon dịu, phông nền gradient trừu tượng công nghệ cao.",
            "Phong cách tối giản công nghệ cao với các đường nét viền phản quang sắc nét, chiều sâu trường ảnh mượt mà.",
        ],
        "envs": [
            "Bối cảnh công nghệ hiện đại với hiệu ứng ánh sáng gradient tinh tế",
            "Không gian studio công nghệ cao với bệ trưng bày hình khối mạ chrome",
            "Phông nền đen mờ sang trọng tôn vinh tối đa các chi tiết viền phản quang",
        ],
        "materials_1": [
            "chữ kim loại chrome bạc phản chiếu ánh sáng studio sắc nét",
            "chữ phát sáng hiệu ứng đèn neon uốn lượn hiện đại rực rỡ",
            "chữ kim loại dập nổi 3D mạ bạc công nghệ sắc sảo",
        ],
        "materials_2": [
            "nét chữ phát quang viền led dịu mắt tạo chiều sâu không gian",
            "dòng chữ màu bạc ánh kim thanh mảnh trang nhã",
            "chữ decal mờ tinh giản với độ tương phản sắc nét",
        ],
    },
    "telecom_viettel": {
        "seeds": [
            "Phong cách thương mại viễn thông hiện đại của Viettel, bệ trưng bày công nghệ số sắc nét, ánh sáng tương phản cao.",
            "Không gian dịch vụ số tương lai với ánh sáng viền công nghệ, phông nền gradient hiện đại thanh lịch.",
            "Bố cục poster truyền thông công nghệ viễn thông cao cấp, ánh sáng spotlight tập trung tôn vinh thiết bị kết nối.",
        ],
        "envs": [
            "Bối cảnh công nghệ số hiện đại với hiệu ứng ánh sáng gradient tinh tế",
            "Không gian studio viễn thông cao cấp với bệ trưng bày hình khối tối giản",
            "Phông nền tương phản cao với ánh sáng studio điện ảnh nghệ thuật",
        ],
        "materials_1": [
            "chữ kim loại dập nổi 3D mạ bạc ánh kim sắc sảo",
            "chữ phát sáng hiệu ứng đèn neon hiện đại trang nhã",
            "chữ kim loại dập nổi 3D mạ vàng sang trọng",
        ],
        "materials_2": [
            "nét chữ màu trắng thanh lịch đổ bóng studio mềm mại",
            "dòng chữ màu bạc ánh kim thanh mảnh trang nhã",
            "nét chữ phát quang viền led dịu mắt tạo chiều sâu không gian",
        ],
    },
    "fnb": {
        "seeds": [
            "Phong cách ẩm thực ấm cúng với bề mặt gỗ mộc mạc, ánh sáng mềm tôn vinh hương vị và màu sắc tự nhiên.",
            "Không gian quán cà phê phong cách Retro hoài niệm, ánh sáng đèn vàng dịu nhẹ, tạo cảm giác gần gũi và thư thái.",
            "Bối cảnh studio ẩm thực chuyên nghiệp, ánh sáng spotlight chiếu xiên làm nổi bật độ tươi ngon và chi tiết đồ uống.",
        ],
        "envs": [
            "Không gian ẩm thực và đồ uống ấm cúng với mặt bàn gỗ tự nhiên mộc mạc",
            "Bối cảnh quán cà phê sang trọng phong cách Retro với ánh sáng ấm dịu",
            "Bố cục poster ẩm thực thương mại cao cấp với các mảng màu cân đối",
        ],
        "materials_1": [
            "chữ vàng đồng cổ điển ánh kim dập nổi tương phản cao",
            "chữ gỗ khắc mộc tinh xảo với vân gỗ tự nhiên ấm áp",
            "chữ kim loại dập nổi 3D mạ vàng sang trọng",
        ],
        "materials_2": [
            "nét chữ màu trắng thanh lịch đổ bóng studio mềm mại",
            "dòng chữ màu vàng nhạt tinh tế hài hòa với bố cục",
            "chữ decal mờ tinh giản với độ tương phản sắc nét",
        ],
    },
    "fashion": {
        "seeds": [
            "Phong cách tạp chí thời trang quốc tế Editorial/Vogue, bố cục thanh lịch, tương phản ánh sáng điện ảnh sắc nét.",
            "Không gian studio thời trang cao cấp với phông nền xám trung tính, ánh sáng softbox tôn vinh chất liệu vải.",
            "Phong cách thời trang đường phố Streetwear năng động, ánh sáng rim-light tương phản mạnh tôn vinh cá tính.",
        ],
        "envs": [
            "Không gian studio thời trang cao cấp với phông nền xám trung tính trang nhã",
            "Bố cục poster thời trang editorial với các mảng màu cân xứng hoàn hảo",
            "Phông nền tương phản cao với ánh sáng studio điện ảnh nghệ thuật",
        ],
        "materials_1": [
            "chữ kim loại dập nổi 3D mạ vàng ánh kim sang trọng",
            "nét chữ typography đậm đà phong cách thời trang hiện đại",
            "chữ sơn mài đen bóng viền kim loại sang trọng",
        ],
        "materials_2": [
            "nét chữ màu trắng thanh lịch đổ bóng studio mềm mại",
            "dòng chữ màu bạc ánh kim thanh mảnh trang nhã",
            "chữ decal mờ tinh giản với độ tương phản sắc nét",
        ],
    },
    "cultural_vietnam": {
        "seeds": [
            "Phong cách không gian văn hóa Việt Nam thanh tao, phông nền sa thạch cổ kính, ánh nắng tự nhiên le lói dịu dàng.",
            "Bối cảnh phố cổ hoài niệm với tường vàng rêu phong, ánh sáng ban mai nhẹ nhàng tôn vinh vẻ đẹp truyền thống.",
            "Không gian nghệ thuật thủ công truyền thống ấm cúng, nền giấy dó mộc mạc, đậm đà bản sắc dân tộc.",
        ],
        "envs": [
            "Không gian văn hóa truyền thống với phông nền tường vàng sa thạch cổ kính",
            "Bối cảnh phố cổ hoài niệm thanh bình với ánh sáng tự nhiên dịu nhẹ",
            "Không gian nghệ thuật truyền thống với nền giấy dó và gỗ mộc ấm cúng",
        ],
        "materials_1": [
            "chữ vàng đồng cổ điển ánh kim dập nổi tương phản cao",
            "chữ gỗ khắc mộc tinh xảo với vân gỗ tự nhiên ấm áp",
            "chữ kim loại dập nổi 3D mạ vàng đồng truyền thống",
        ],
        "materials_2": [
            "nét chữ màu trắng thanh lịch đổ bóng tự nhiên mềm mại",
            "chữ khắc chìm phong cách tối giản tương phản rõ ràng",
            "dòng chữ màu vàng nhạt tinh tế hài hòa với bố cục",
        ],
    },
    "home": {
        "seeds": [
            "Phong cách không gian sống gia đình hiện đại Bắc Âu, ánh sáng tự nhiên tràn ngập từ cửa sổ, tone màu gỗ và trắng ấm áp.",
            "Bối cảnh nội thất gia đình trang nhã, phông nền tối giản hiện đại tôn vinh sự tiện nghi và an lành.",
            "Studio chụp ảnh đồ gia dụng chuyên nghiệp, ánh sáng softbox khuếch tán đều, tạo cảm giác sạch sẽ và đáng tin cậy.",
        ],
        "envs": [
            "Không gian nội thất gia đình sang trọng với ánh sáng tự nhiên dịu nhẹ",
            "Không gian chụp ảnh phong cách Bắc Âu trang nhã với tone màu sáng ấm",
            "Bối cảnh studio gia dụng hiện đại với bệ trưng bày tối giản",
        ],
        "materials_1": [
            "chữ kim loại dập nổi 3D mạ vàng ánh kim sang trọng",
            "chữ in nổi chất liệu acrylic cao cấp trong suốt bóng bẩy",
            "nét chữ typography đậm đà phong cách hiện đại dập chìm",
        ],
        "materials_2": [
            "nét chữ màu trắng thanh lịch đổ bóng studio mềm mại",
            "chữ decal mờ tinh giản với độ tương phản sắc nét",
            "dòng chữ màu vàng nhạt tinh tế hài hòa với bố cục",
        ],
    },
    "fmcg": {
        "seeds": [
            "Phong cách poster tiêu dùng nhanh thương mại cao cấp, bệ trưng bày tối giản, ánh sáng studio tương phản sắc nét.",
            "Không gian chụp ảnh bao bì sản phẩm chuyên nghiệp, ánh sáng softbox tôn vinh màu sắc tươi tắn và nhãn mác rõ ràng.",
            "Bố cục poster thương mại hiện đại với các khối màu tươi sáng, ánh sáng rạng rỡ thân thiện.",
        ],
        "envs": [
            "Bối cảnh studio thương mại hiện đại với bệ trưng bày hình khối tối giản",
            "Không gian chụp ảnh sản phẩm chuyên nghiệp với ánh sáng tự nhiên dịu mắt",
            "Bố cục poster đồ họa thương mại cao cấp với các mảng màu cân đối",
        ],
        "materials_1": [
            "chữ kim loại dập nổi 3D mạ vàng ánh kim sang trọng",
            "chữ vàng đồng cổ điển ánh kim dập nổi tương phản cao",
            "chữ in nổi chất liệu acrylic cao cấp bóng bẩy",
        ],
        "materials_2": [
            "nét chữ màu trắng thanh lịch đổ bóng studio mềm mại",
            "chữ decal mờ tinh giản với độ tương phản sắc nét",
            "dòng chữ màu vàng nhạt tinh tế hài hòa với bố cục",
        ],
    },
}


def get_domain_context(domain: str) -> Dict[str, List[str]]:
    clean_domain = domain.replace("general_", "").lower()
    if clean_domain in DOMAIN_VISUAL_CONTEXTS:
        return DOMAIN_VISUAL_CONTEXTS[clean_domain]
    # Fallback to fmcg context
    return DOMAIN_VISUAL_CONTEXTS["fmcg"]


SYNTACTIC_FLOW_HINTS = [
    "Khởi đầu câu bằng vị trí không gian rồi đến chất liệu và vai trò khối chữ (ví dụ: 'Ở phần trên cùng, tiêu đề chính được chế tác bằng...').",
    "Khởi đầu câu bằng chất liệu chế tác rồi mới đến vị trí (ví dụ: 'Sử dụng chữ kim loại mạ vàng sang trọng, tiêu đề chính tọa lạc tại...').",
    "Tập trung miêu tả hiệu ứng quang học và đổ bóng studio của khối chữ trước khi mô tả vị trí.",
    "Sử dụng câu ghép mô tả sự chuyển tiếp tự nhiên giữa tiêu đề chính và phụ đề bổ trợ.",
    "Mô tả trực diện phong cách typography hiện đại với sự hòa quyện giữa chất liệu chữ và phông nền studio.",
]

ROLE_DESCRIPTORS_1 = [
    "Khối tiêu đề chính", "Dòng chữ thông điệp nổi bật", "Tiêu đề vị trí",
    "Khối tên thương hiệu chính", "Dòng chữ chủ đề lớn", "Tiêu đề thông điệp",
]

ROLE_DESCRIPTORS_2 = [
    "Dòng phụ đề bổ trợ", "Khối thông tin chi tiết 2 dòng", "Dòng chú thích nội dung",
    "Khối thông điệp phụ", "Khối quyền lợi đãi ngộ", "Dòng slogan ngắn gọn",
]

POSITION_DESCRIPTORS_1 = [
    "ở phía trên chính giữa", "ở phần trên cùng của bố cục", "ở góc trên cân đối",
    "ở vị trí trung tâm phía trên", "chạy ngang phần trên canvas", "tọa lạc trang trọng ở nửa trên",
]

POSITION_DESCRIPTORS_2 = [
    "ở phía dưới chính giữa", "ở phần chân đế bên dưới", "nằm ngay bên dưới tiêu đề",
    "ở nửa dưới của poster", "ở vị trí góc dưới thanh lịch", "bố trí cân xứng ở phần chân poster",
]

PHOTOGRAPHY_LIGHTING = [
    "Ánh sáng studio tương phản cao, đổ bóng tự nhiên sắc nét, phong cách nhiếp ảnh thương mại chuẩn mực.",
    "Ánh sáng tự nhiên dịu mắt, độ chi tiết cao, màu sắc hài hòa sống động.",
    "Ánh sáng spotlight tập trung vào chủ thể, chiều sâu trường ảnh mượt mà.",
    "Phong cách thiết kế poster quảng cáo thương mại cao cấp, bố cục cân xứng hoàn hảo.",
    "Ánh sáng viền rim-light sắc sảo tách bạch chủ thể trên nền tối, tạo chiều sâu quang học tinh tế.",
    "Ánh sáng khuếch tán softbox đa chiều, loại bỏ phản chiếu gắt, tôn vinh chất liệu chân thực.",
]

PRODUCT_PATTERNS = [
    "(3) Sản phẩm thật được đặt trang trọng làm tiêu điểm trung tâm, đón ánh sáng studio nổi bật toàn bộ kiểu dáng.",
    "(3) Ở vị trí trọng tâm khung hình, sản phẩm thật nổi bật với độ sắc nét và chi tiết bề mặt chân thực.",
    "(3) Sản phẩm thật được trưng bày ngay ngắn ở phần chân đế, tạo liên kết thị giác hài hòa với các khối chữ.",
    "(3) Đặt tại trung tâm bố cục, sản phẩm thật bắt sáng tự nhiên với chiều sâu trường ảnh mượt mà.",
]

# 6 diverse syntactic patterns for (1) and (2) in combinatorial fallback
COMBINATORIAL_PATTERNS = [
    # Pattern 1: Vị trí trước
    lambda r1, p1, m1, r2, p2, m2: (
        f"(1) {p1.capitalize()}, {r1.lower()} được chế tác tinh xảo bằng {m1}. "
        f"(2) {p2.capitalize()}, {r2.lower()} nổi bật với {m2}."
    ),
    # Pattern 2: Chất liệu trước
    lambda r1, p1, m1, r2, p2, m2: (
        f"(1) Sử dụng {m1}, {r1.lower()} được bố trí trang nhã {p1}. "
        f"(2) {r2.capitalize()} với {m2} được đặt {p2} tạo điểm nhấn cân đối."
    ),
    # Pattern 3: Động từ thị giác / ánh sáng
    lambda r1, p1, m1, r2, p2, m2: (
        f"(1) {r1} làm bằng {m1} thu hút ánh nhìn {p1}. "
        f"(2) Đi kèm ngay {p2} là {r2.lower()} mang {m2} bổ trợ thông tin rõ nét."
    ),
    # Pattern 4: Thiết kế bố cục cân đối
    lambda r1, p1, m1, r2, p2, m2: (
        f"(1) Thiết kế bố trí {r1.lower()} bằng {m1} tọa lạc {p1}. "
        f"(2) Nửa dưới khung hình đón nhận {r2.lower()} {p2} với {m2} thanh lịch."
    ),
    # Pattern 5: Mô tả trực diện tinh tế
    lambda r1, p1, m1, r2, p2, m2: (
        f"(1) {r1} {p1} làm nổi bật hiệu ứng {m1}. "
        f"(2) {r2} {p2} trình bày bằng {m2} sắc sảo và mạch lạc."
    ),
    # Pattern 6: Tương phản sáng tối / studio
    lambda r1, p1, m1, r2, p2, m2: (
        f"(1) Ánh sáng tôn vinh {r1.lower()} mang {m1} hiện diện {p1}. "
        f"(2) Hòa quyện cùng tổng thể, {r2.lower()} {p2} thể hiện bằng {m2} tinh tế."
    ),
]


def _leaks(candidate: str, *texts: str) -> bool:
    cand_low = candidate.lower()
    for t in texts:
        for line in t.split("\n"):
            line = line.strip()
            if line and line.lower() in cand_low:
                return True
    # Also forbid dimension/resolution leaks
    for forbidden in ["1024x1024", "768x1344", "896x1152", "1344x768", "9:16", "16:9", "4:5", "1:1", "4k", "8k"]:
        if forbidden in cand_low:
            return True
    return False


USE_CASE_DESCRIPTIONS: Dict[str, str] = {
    "hero_product": "Poster quảng cáo thương mại cao cấp tôn vinh sản phẩm",
    "flash_sale": "Poster thông báo chương trình khuyến mại ưu đãi đặc biệt",
    "customer_feedback": "Thẻ ảnh đánh giá phản hồi (card feedback) trải nghiệm khách hàng",
    "opening_banner": "Banner sự kiện khai trương ra mắt cơ sở hoặc dịch vụ mới",
    "recruitment": "Thẻ tin tuyển dụng nhân sự chuyên nghiệp và uy tín",
    "two_step_guide": "Infographic poster hướng dẫn quy trình 2 bước trực quan",
    "creative_quote": "Poster tranh ảnh nghệ thuật trích dẫn danh ngôn truyền cảm hứng",
}


def combinatorial_clean_prompt(spec: Dict, has_product: bool) -> str:
    text1, text2 = spec["text1"], spec["text2"]
    domain = spec.get("domain", "general")
    ctx = get_domain_context(domain)
    raw_uc = spec.get("use_case", "commercial")
    uc_desc = USE_CASE_DESCRIPTIONS.get(raw_uc, "Poster đồ họa thương mại cao cấp")
    prod_name = _clean_product_name(spec["product_path"]) if has_product and spec.get("product_path") else ""
    subject_desc = f"sản phẩm {prod_name}" if prod_name else f"chủ đề {domain}"

    env = random.choice(ctx["envs"])
    r1, p1, m1 = random.choice(ROLE_DESCRIPTORS_1), random.choice(POSITION_DESCRIPTORS_1), random.choice(ctx["materials_1"])
    r2, p2, m2 = random.choice(ROLE_DESCRIPTORS_2), random.choice(POSITION_DESCRIPTORS_2), random.choice(ctx["materials_2"])
    light = random.choice(PHOTOGRAPHY_LIGHTING)
    pattern_fn = random.choice(COMBINATORIAL_PATTERNS)
    typography_part = pattern_fn(r1, p1, m1, r2, p2, m2)

    parts = [f"{uc_desc} cho {subject_desc}, {env.lower()}.", typography_part]
    if has_product:
        parts.append(random.choice(PRODUCT_PATTERNS))
    parts.append(light)
    prompt_clean = " ".join(parts)
    assert not _leaks(prompt_clean, text1, text2), f"Leak detected in combinatorial prompt: {prompt_clean}"
    assert "(1)" in prompt_clean and "(2)" in prompt_clean
    if has_product:
        assert "(3)" in prompt_clean
    return prompt_clean


LLM_SYSTEM_PROMPT = """Bạn là chuyên gia Art Director biên soạn Clean Prompt cho mô hình DiT (FLUX.2) trong nền tảng Tendoo AI.
Trong thực tế, Clean Prompt này là kết quả sau khi LLM Enhancer xử lý prompt thô của người dùng (đã bóc tách nội dung chữ ra làm VAE Glyphs, và thay bằng các mỏ neo (1), (2), (3)).

NHIỆM VỤ CỐT LÕI:
Viết MỘT đoạn văn ngắn gọn (khoảng 3-4 câu) bằng tiếng Việt mô tả đầy đủ:
1. YÊU CẦU NGHIỆP VỤ & BỐI CẢNH THỰC TẾ (TUYỆT ĐỐI KHÔNG ĐƯỢC BỎ MẤT YÊU CẦU NGƯỜI DÙNG):
   - Mở đầu bằng yêu cầu nghiệp vụ rõ ràng: Ví dụ "Poster quảng cáo thương mại cho sản phẩm...", "Banner sự kiện khai trương...", "Thẻ tin tuyển dụng chuyên nghiệp cho...", "Thẻ card feedback khách hàng cho..."
   - Mô tả bối cảnh không gian sống/nhiếp ảnh studio thực tế liên quan mật thiết đến sản phẩm/chủ đề (ví dụ: bàn ăn gia đình ấm cúng với nguyên liệu tươi ngon; phòng gym cao cấp với tạ đòn; phố cổ hoàng hôn với tường vàng và đèn lồng đỏ; studio công nghệ cao với bệ chrome...).
2. CÁC THẺ MỎ NEO KHÔNG GIAN BẮT BUỘC ĐỂ ĐỊNH VỊ CHỮ VÀ SẢN PHẨM:
   - Thẻ "(1)": Đứng trước câu mô tả HÌNH THỨC của khối chữ thứ nhất (vị trí ở đâu, chất liệu chữ 3D mạ vàng / acrylic / kim loại / thư pháp, hướng chiếu sáng).
   - Thẻ "(2)": Đứng trước câu mô tả HÌNH THỨC của khối chữ thứ hai (vị trí bên dưới, chất liệu chữ trắng thanh mảnh / decal mờ / viền led).
{product_rule}
3. CẢNH BÁO TUYỆT ĐỐI VỀ NỘI DUNG CHỮ:
   - TUYỆT ĐỐI KHÔNG LẶP LẠI NỘI DUNG CHỮ THẬT (Representation Clash).
   - TUYỆT ĐỐI KHÔNG tự bịa ra slogan hay câu chữ giả. Chỉ mô tả HÌNH THỨC VẬT LÝ của nét chữ.
   - TUYỆT ĐỐI KHÔNG ghi độ phân giải hay tỉ lệ khung hình (như '1024x1024', '9:16', '4k', '8k', '--ar').
4. Trả về DUY NHẤT một đoạn văn xuôi hoàn chỉnh, không gạch đầu dòng, không tiêu đề."""


def llm_clean_prompt(spec: Dict, has_product: bool, max_retries: int = 3) -> Optional[str]:
    """Ask an LLM to author a diverse student prompt with natural variation around anchors (1), (2), (3).
    Enforces anti-leak and anchor presence as a hard gate.
    """
    text1, text2 = spec["text1"], spec["text2"]
    product_rule = "3. BẮT BUỘC có thẻ \"(3)\" đứng trước mô tả vị trí của SẢN PHẨM THẬT trong bố cục." if has_product else ""
    system = LLM_SYSTEM_PROMPT.format(product_rule=product_rule)

    domain = spec.get("domain", "general")
    ctx = get_domain_context(domain)
    style_seed = random.choice(ctx["seeds"])
    syntax_hint = random.choice(SYNTACTIC_FLOW_HINTS)

    raw_uc = spec.get("use_case", "commercial")
    uc_desc = USE_CASE_DESCRIPTIONS.get(raw_uc, "Poster đồ họa thương mại cao cấp")
    prod_name = _clean_product_name(spec["product_path"]) if has_product and spec.get("product_path") else ""
    subject_desc = f"sản phẩm {prod_name}" if prod_name else f"chủ đề {domain}"

    num_words_1 = len(text1.split())
    num_lines_1 = len(text1.split("\n"))
    num_words_2 = len(text2.split())
    num_lines_2 = len(text2.split("\n"))

    user_msg = (
        f"Yêu cầu nghiệp vụ cốt lõi: {uc_desc} cho {subject_desc}\n"
        f"Ngành hàng/Lĩnh vực: {domain}\n"
        f"Gợi ý phong cách & không gian: {style_seed}\n"
        f"Gợi ý cấu trúc câu: {syntax_hint}\n"
        f"Đặc điểm khối chữ (1): Tiêu đề chính ({num_words_1} từ, {num_lines_1} dòng)\n"
        f"Đặc điểm khối chữ (2): Phụ đề bổ trợ ({num_words_2} từ, {num_lines_2} dòng)\n"
        f"Có sản phẩm thật trong ảnh: {'Có (bắt buộc có thẻ (3) mô tả vị trí sản phẩm thật)' if has_product else 'Không'}"
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.85,
                max_tokens=280,
            )
            candidate = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"   [LLM prompt WARN] call failed (attempt {attempt+1}): {e}")
            continue

        if _leaks(candidate, text1, text2):
            print(f"   [LLM prompt WARN] leak detected on attempt {attempt+1}, retrying...")
            continue
        if "(1)" not in candidate or "(2)" not in candidate:
            print(f"   [LLM prompt WARN] missing ordinal tags on attempt {attempt+1}, retrying...")
            continue
        if has_product and "(3)" not in candidate:
            print(f"   [LLM prompt WARN] missing (3) product tag on attempt {attempt+1}, retrying...")
            continue
        return candidate

    return None  # caller falls back to deterministic combinatorial builder


def build_clean_prompt(spec: Dict, has_product: bool, use_llm: bool) -> str:
    if use_llm:
        result = llm_clean_prompt(spec, has_product)
        if result is not None:
            return result
        print("   [LLM prompt] all retries failed validation -> falling back to combinatorial builder")
    return combinatorial_clean_prompt(spec, has_product)


# ==================================================================================================
# 6. SAMPLER
# ==================================================================================================
def _pick_product_for_domain(domain: str) -> Optional[Path]:
    prod_folder = PROJECT_ROOT / "data" / "products" / domain
    if not prod_folder.exists():
        return None
    files = sorted(f for f in prod_folder.glob("*.*") if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return random.choice(files) if files else None


# ==================================================================================================
# 6. BUSINESS USE-CASE QUOTA MATRIX (SUB-PLAN 1.1: 800-SAMPLE GROUND TRUTH ALLOCATION)
# ==================================================================================================
# Target sample counts for I2I (55% = 440 total per Sub-plan Table 1.1)
# Note: Real packshots must strictly feature product-tailored commercial copy (no recruitment or abstract quotes on consumer products).
I2I_USE_CASE_TARGETS: List[Tuple[str, int]] = [
    ("hero_product", 190),        # 43.18% of I2I (Product identity, features, branding)
    ("flash_sale", 120),          # 27.27% of I2I (Hot discounts, promotions)
    ("customer_feedback", 70),    # 15.91% of I2I (Product ratings, user satisfaction)
    ("opening_banner", 35),       # 7.95% of I2I (Flagship launch, new arrival)
    ("two_step_guide", 25),       # 5.68% of I2I (Usage guide, unbox/experience)
]

# Target sample counts for Pure T2I (45% = 360 total per Sub-plan Table 1.1)
T2I_USE_CASE_TARGETS: List[Tuple[str, int]] = [
    ("recruitment", 90),          # 25.00% of T2I (Modern employer branding, tech hiring)
    ("creative_quote", 80),       # 22.22% of T2I (Inspirational quotes, wisdom, typography art)
    ("flash_sale", 60),           # 16.67% of T2I (Services, vouchers, travel, dining)
    ("cultural_vietnam", 60),     # 16.67% of T2I (Vietnamese cuisine, heritage, street life)
    ("opening_banner", 35),       # 9.72% of T2I (Store grand opening, exhibition, event)
    ("two_step_guide", 35),       # 9.72% of T2I (App onboarding, digital procedure)
]


def determine_use_case(sample_id: int, total_samples: int, is_i2i: bool) -> str:
    i2i_cutoff = round(total_samples * 0.55)
    targets = I2I_USE_CASE_TARGETS if is_i2i else T2I_USE_CASE_TARGETS
    total_weight = sum(w for _, w in targets)

    if is_i2i:
        rel_idx = sample_id - 1
        sample_span = max(1, i2i_cutoff)
    else:
        rel_idx = sample_id - i2i_cutoff - 1
        sample_span = max(1, total_samples - i2i_cutoff)

    progress = (rel_idx + 0.5) / sample_span
    acc = 0.0
    for uc, weight in targets:
        acc += weight / total_weight
        if progress < acc:
            return uc
    return targets[-1][0]


def adapt_text_for_aspect_ratio(text: str, ar_name: str) -> str:
    """Adapts line breaks to match the geometric aspect ratio of the canvas.
    On narrow 9:16 (768px width), prevents overflow and uncontrolled line breaks by
    proactively structuring >=3 word titles into balanced multi-line text.
    On wide 16:9 (1344px width, 768px height), prevents tall multi-line stacks that consume vertical space.
    """
    if not text:
        return text

    if ar_name == "9:16":
        # Narrow vertical canvas: 1 single long line will overflow or force GPT to break unpredictably.
        # Proactively wrap >=3 words if not already broken.
        if "\n" not in text:
            words = text.split()
            if len(words) == 3:
                return f"{words[0]} {words[1]}\n{words[2]}"
            elif len(words) == 4:
                return f"{words[0]} {words[1]}\n{words[2]} {words[3]}"
            elif len(words) == 5:
                return f"{words[0]} {words[1]}\n{words[2]} {words[3]} {words[4]}"
            elif len(words) >= 6:
                # Break into roughly 3-4 word lines
                lines = []
                for i in range(0, len(words), 3):
                    lines.append(" ".join(words[i:i+3]))
                return "\n".join(lines)
    elif ar_name == "16:9":
        # Wide horizontal canvas: abundant width (1344px), scarce height (768px).
        # Avoid >=3 stacked lines. If text has 3+ lines, rebalance into 1-2 lines.
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) >= 3:
            all_words = " ".join(lines).split()
            half = len(all_words) // 2
            return " ".join(all_words[:half]) + "\n" + " ".join(all_words[half:])

    return text


def sample_dataset_spec(sample_id: int, total_samples: int) -> Dict:
    # 1. Modality: 55% I2I / 45% T2I, proportional to total_samples (fixes the hardcoded-800 bug)
    i2i_cutoff = round(total_samples * 0.55)
    is_i2i = sample_id <= i2i_cutoff
    modality = "i2i" if is_i2i else "t2i"

    # 2. Aspect ratio
    ar_name = determine_aspect_ratio(sample_id, total_samples, is_i2i)
    ar_cfg = ASPECT_RATIOS[ar_name]

    # 3. Fonts
    font1, font2 = sample_orthogonal_fonts()
    floor1, floor2 = get_font_floor(font1), get_font_floor(font2)

    # 4. Determine Use-Case based on Sub-plan 1.1 Matrix
    use_case = determine_use_case(sample_id, total_samples, is_i2i)

    # 5. Length Stratum: EXACT 75% standard commercial (600 samples), 25% inverted (200 samples)
    # Using deterministic cycle: sample_id % 4 == 0 ensures exactly 1 in 4 samples is inverted.
    # When total_samples == 800: exactly 200 inverted (110 I2I + 90 T2I) and 600 standard (330 I2I + 270 T2I).
    is_inverted = (sample_id % 4 == 0)
    length_stratum = "inverted" if is_inverted else "standard"

    # 6. Cohort: ~12.5% known-hard stress reproductions (exactly 100 samples when total_samples == 800)
    is_known_hard = (sample_id % 8 == 0)
    cohort = "known_hard" if is_known_hard else "standard"

    product_path: Optional[Path] = None

    if is_known_hard:
        pair = random.choice(KNOWN_HARD_PAIRS)
        text1, text2, domain = pair["text1"], pair["text2"], pair["domain"]
        if is_i2i:
            product_path = _pick_product_for_domain(domain)
    elif is_i2i:
        # I2I ALWAYS has 2 text slots + 1 product slot (per corrected architecture).
        # In I2I, typography MUST strictly match the product packshot!
        domain = random.choice(list(PRODUCT_TEXT_CORPUS.keys()))
        stem = random.choice(list(PRODUCT_TEXT_CORPUS[domain].keys()))
        prod_folder = PROJECT_ROOT / "data" / "products" / domain
        candidates = list(prod_folder.glob(f"{stem}.*"))
        product_path = candidates[0] if candidates else _pick_product_for_domain(domain)

        # STRICT PRODUCT-COUPLED COPY: Never leak disconnected quotes/recruitment onto products
        options = PRODUCT_TEXT_CORPUS[domain][stem][length_stratum]
        text1, text2 = random.choice(options)
    else:
        # Pure T2I: 2 text slots, no product.
        if use_case == "cultural_vietnam":
            domain = "cultural_vietnam"
            options = GENERAL_T2I_CORPUS["cultural_vietnam"][length_stratum]
            text1, text2 = random.choice(options)
        elif use_case == "flash_sale":
            options = GENERAL_T2I_CORPUS["flash_sale"][length_stratum]
            text1, text2 = random.choice(options)
            domain = random.choice(["fmcg", "fashion", "tech", "home"])
        else:
            domain = "general_" + use_case
            options = GENERAL_T2I_CORPUS[use_case][length_stratum]
            text1, text2 = random.choice(options)

    # GEOMETRIC LINE ADAPTATION: Proactively align text line breaks with aspect ratio
    text1 = adapt_text_for_aspect_ratio(text1, ar_name)
    text2 = adapt_text_for_aspect_ratio(text2, ar_name)

    return {
        "id": f"sample_{sample_id:04d}",
        "cohort": cohort,
        "length_stratum": length_stratum,
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
        "product_path": product_path,
    }


# ==================================================================================================
# 7. GLYPH RENDERING (both modalities always render BOTH glyphs)
# ==================================================================================================
def render_and_save_glyphs(spec: Dict, glyphs_dir: Path):
    sid = spec["id"]
    g1_path = glyphs_dir / f"glyph_{sid}_slot10.png"
    g2_path = glyphs_dir / f"glyph_{sid}_slot20.png"

    g1_info = glyph_engine.render(
        text=spec["text1"],
        font_name_or_path=spec["font1"],
        font_size_pt=spec["floor1"],
        force_single_line=("\n" not in spec["text1"]),
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

    return g1_path, g2_path, g1_info, g2_info


# ==================================================================================================
# 8. TEACHER CALL (gpt-image-2): generate() for T2I, edit() with real product photo for I2I
# ==================================================================================================
def _clean_product_name(product_path: Path) -> str:
    return re.sub(r"^\d+[\s_-]*", "", product_path.stem).replace("_", " ")


def _format_lines_desc(lines: List[str], position: str, font_style: str) -> str:
    n = len(lines)
    if n == 1:
        return f"{position}, on 1 single line {font_style}: '{lines[0]}'"
    else:
        line_specs = ", ".join(f"Line {i+1}: '{line}'" for i, line in enumerate(lines))
        return f"{position}, stacked vertically on {n} lines ({line_specs}) {font_style}"


def _build_teacher_prompt(spec: Dict, g1_lines: List[str], g2_lines: List[str]) -> str:
    f1_style = FONT_STYLE_DESCRIPTORS.get(spec["font1"], "in clean modern bold typography")
    f2_style = FONT_STYLE_DESCRIPTORS.get(spec["font2"], "in clean typography")

    t1_desc = _format_lines_desc(g1_lines, "At top", f1_style)
    t2_desc = _format_lines_desc(g2_lines, "Below it", f2_style)

    prod_desc = ""
    if spec["modality"] == "i2i" and spec.get("product_path"):
        prod_desc = (
            f"The exact product shown in the reference image (a {_clean_product_name(spec['product_path'])}) "
            f"must appear placed prominently, unmodified in identity/shape/label, in the lower or center portion. "
        )

    negative_rule = (
        "CRITICAL TYPOGRAPHY RESTRICTION: Render ONLY the exact text specified above, with fully correct "
        "Vietnamese diacritics (tone marks) -- do not drop, alter, or simplify any dấu. "
        "DO NOT add any other words, badges, discount numbers, phone numbers, website URLs, or decorative "
        "gibberish text. There must be ZERO extraneous text anywhere on the canvas."
    )

    return (
        f"Commercial advertising graphic poster for {spec['use_case']} ({spec['domain']}). "
        f"{prod_desc}"
        f"{t1_desc}. {t2_desc}. "
        f"{negative_rule} "
        f"Professional graphic typography design, high contrast, sharp studio lighting, commercial photography."
    )


def map_to_openai_size(target_w: int, target_h: int) -> str:
    """Maps target aspect ratio to supported OpenAI API sizes:
    1024x1024 (1:1), 1024x1536 (portrait), 1536x1024 (landscape).
    """
    if target_w == target_h:
        return "1024x1024"
    elif target_w < target_h:
        return "1024x1536"
    else:
        return "1536x1024"


def generate_target_image(spec: Dict, g1_lines: List[str], g2_lines: List[str], targets_dir: Path, delay: float) -> Path:
    from PIL import Image, ImageOps
    import io
    import urllib.request

    sid = spec["id"]
    target_path = targets_dir / f"target_{sid}.png"
    if target_path.exists() and target_path.stat().st_size > 10000:
        return target_path

    teacher_prompt = _build_teacher_prompt(spec, g1_lines, g2_lines)
    api_size = map_to_openai_size(spec["width"], spec["height"])

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if spec["modality"] == "i2i" and spec.get("product_path"):
                # Pass file handle directly to OpenAI image edit endpoint
                with open(spec["product_path"], "rb") as prod_file:
                    res = client.images.edit(
                        model="gpt-image-2",
                        image=prod_file,
                        prompt=teacher_prompt,
                        quality="low",
                        size=api_size,
                    )
            else:
                res = client.images.generate(
                    model="gpt-image-2",
                    prompt=teacher_prompt,
                    quality="low",
                    size=api_size,
                )

            # Retrieve image bytes whether returned via url or b64_json
            if getattr(res.data[0], "b64_json", None):
                raw_bytes = base64.b64decode(res.data[0].b64_json)
                img = Image.open(io.BytesIO(raw_bytes))
            elif getattr(res.data[0], "url", None):
                req = urllib.request.Request(res.data[0].url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    img = Image.open(io.BytesIO(resp.read()))
            else:
                raise RuntimeError(f"No image data returned from API for {sid}")

            # Ensure exact target bucket dimensions (1024x1024, 768x1344, 896x1152, 1344x768)
            target_size = (spec["width"], spec["height"])
            if img.size != target_size:
                img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)

            img.save(target_path)
            time.sleep(delay)
            return target_path
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"   [429 RATE LIMIT] Backoff 12s (attempt {attempt+1}/{max_retries})...")
                time.sleep(12.0)
            else:
                print(f"   [API ERROR on {sid}] attempt {attempt+1}: {e}")
                time.sleep(3.0)
                if attempt == max_retries - 1:
                    raise

    raise RuntimeError(f"Failed to generate target for {sid}")


# ==================================================================================================
# 9. MAIN
# ==================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Tendoo AI - Master Dataset Synthesis Engine")
    parser.add_argument("--smoke", action="store_true", help="Run Smoke Test (10 samples)")
    parser.add_argument("--pilot", action="store_true", help="Run Pilot Test (60 samples)")
    parser.add_argument("--count", type=int, default=None, help="Custom sample count")
    parser.add_argument("--execute", action="store_true", help="Actually execute API calls (default: dry-run)")
    parser.add_argument("--delay", type=float, default=9.5, help="Delay in seconds between API requests")
    parser.add_argument("--llm-prompts", action="store_true", help="Use LLM-authored student prompts (falls back to combinatorial on failure)")
    args = parser.parse_args()

    total_samples = 10 if args.smoke else (60 if args.pilot else (args.count or 800))

    output_dir = PROJECT_ROOT / "data" / "milestone_a"
    glyphs_dir = output_dir / "glyphs"
    targets_dir = output_dir / "targets"
    glyphs_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.jsonl"

    print("=" * 90)
    print(f" [*] TENDOO AI - MILESTONE A DATASET GENERATOR (TARGET COUNT: {total_samples})")
    print(f" [*] MODE: {'EXECUTE (API + GLYPHS)' if args.execute else 'DRY RUN (SPECIFICATION VERIFICATION)'}")
    print(f" [*] STUDENT PROMPTS: {'LLM-authored (w/ combinatorial fallback)' if args.llm_prompts else 'Combinatorial only'}")
    print(f" [*] OUTPUT DIRECTORY: {output_dir}")
    print("=" * 90)

    if not args.execute:
        num_to_print = min(15, total_samples) if total_samples > 10 else total_samples
        for idx in range(1, num_to_print + 1):
            spec = sample_dataset_spec(idx, total_samples)
            has_product = spec["modality"] == "i2i" and spec.get("product_path") is not None
            prompt_clean = build_clean_prompt(spec, has_product, args.llm_prompts)
            print(f"\n--- [SAMPLE #{spec['id']}] cohort={spec['cohort']} stratum={spec['length_stratum']} ---")
            print(f" Modality: {spec['modality']} | Use Case: {spec['use_case']} | Domain: {spec['domain']} | AR: {spec['aspect_ratio']} ({spec['width']}x{spec['height']})")
            print(f" Slot t=10.0 (text1): '{spec['text1']}' [Font: {spec['font1']} @ {spec['floor1']}pt]")
            print(f" Slot t=20.0 (text2): '{spec['text2'].replace(chr(10), ' / ')}' [Font: {spec['font2']} @ {spec['floor2']}pt]")
            if has_product:
                print(f" Slot t=30.0 (product): {spec['product_path'].name}")
            print(f" Student Clean Prompt: {prompt_clean}")
        print("\n" + "=" * 90)
        print(" [OK] Dry-run passed. To generate real data, run with '--execute'.")
        print("=" * 90)
        return

    existing_ids = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
    print(f" [*] Found {len(existing_ids)} existing records in manifest. Resuming...")

    success_count = len(existing_ids)
    for idx in range(1, total_samples + 1):
        sample_id = f"sample_{idx:04d}"
        if sample_id in existing_ids:
            continue

        spec = sample_dataset_spec(idx, total_samples)
        print(f"[{idx:04d}/{total_samples:04d}] {sample_id} ({spec['modality']} | {spec['use_case']} | {spec['aspect_ratio']} | {spec['length_stratum']})...")

        g1_path, g2_path, g1_info, g2_info = render_and_save_glyphs(spec, glyphs_dir)
        has_product = spec["modality"] == "i2i" and spec.get("product_path") is not None
        prompt_clean = build_clean_prompt(spec, has_product, args.llm_prompts)

        target_path = generate_target_image(spec, g1_info.lines, g2_info.lines, targets_dir, delay=args.delay)

        slots = [
            {
                "time_offset": T_TEXT1, "type": "glyph",
                "path": str(g1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "font": spec["font1"], "font_size_pt": spec["floor1"], "text": spec["text1"],
                "width_px": g1_info.width_px, "height_px": g1_info.height_px, "token_count": g1_info.token_count,
            },
            {
                "time_offset": T_TEXT2, "type": "glyph",
                "path": str(g2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "font": spec["font2"], "font_size_pt": spec["floor2"], "text": spec["text2"],
                "width_px": g2_info.width_px, "height_px": g2_info.height_px, "token_count": g2_info.token_count,
            },
        ]
        if has_product:
            slots.append({
                "time_offset": T_PRODUCT, "type": "product",
                "path": str(spec["product_path"].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "product_name": spec["product_path"].stem,
            })

        record = {
            "id": spec["id"],
            "cohort": spec["cohort"],
            "length_stratum": spec["length_stratum"],
            "modality": spec["modality"],
            "use_case": spec["use_case"],
            "domain": spec["domain"],
            "aspect_ratio": spec["aspect_ratio"],
            "width": spec["width"],
            "height": spec["height"],
            "prompt_clean": prompt_clean,
            "target_image": str(target_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "slots": slots,
        }

        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        success_count += 1
        print(f"   ===> [OK] {sample_id} saved. (Total completed: {success_count})")

    print("\n" + "=" * 90)
    print(f" [*] BATCH COMPLETE: {success_count}/{total_samples} samples generated in: {output_dir}")
    print(f" [*] Manifest: {manifest_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()