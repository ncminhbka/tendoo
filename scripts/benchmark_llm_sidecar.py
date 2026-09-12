"""
scripts/benchmark_llm_sidecar.py

BỘ KIỂM THỬ VÀ ĐÁNH GIÁ ĐỊNH LƯỢNG NĂNG LỰC LLM SIDECAR (QWEN) TRONG TENDOO AI:
=============================================================================
Đánh giá khách quan, độc lập xem mô hình Qwen3-4B-FP8 (hoặc các phiên bản khác)
có đủ năng lực đảm nhiệm vai trò "Bộ não Dàn trang & Sinh Prompt (Layout Brain)"
hay bắt buộc phải nâng cấp lên mô hình lớn hơn (Qwen2.5-7B, Qwen2.5-14B, API Gemini).

6 TRỤC ĐÁNH GIÁ (EVALUATION PILLARS):
------------------------------------
1. JSON Syntax Compliance: Tỉ lệ xuất ra đúng cú pháp JSON 100% không bị vỡ.
2. Schema & Zone Validity: Không hallucinate zone lạ ngoài 9 ô Omni-Block và 2 thanh bar.
3. Spatial Placement Accuracy: Độ chính xác khi prompt yêu cầu vị trí ("ở trên", "chia đôi 2 hình"...).
4. Scene Prompt Cleanliness: TUYỆT ĐỐI không rò rỉ chuỗi text tiếng Việt hoặc tỉ lệ "16:9", "4K" vào diffusion.
5. Entity Extraction & Sizing: Bóc tách đúng vai trò (hero, badge, quote) và giữ Hero ngắn gọn (<= 6 từ).
6. Inference Latency & Efficiency: Đo đạc tốc độ suy luận (thời gian giây / prompt, tokens/sec).

CÁC CHẾ ĐỘ CHẠY (MODES):
------------------------
- direct:  Nạp trực tiếp weights qua Transformers trên GPU A30 (Server).
- sidecar: Gửi HTTP request tới FastAPI sidecar đang chạy (port 7861).
- mock:    Chạy giả lập offline trên máy local Windows để kiểm tra toàn bộ rubric chấm điểm và báo cáo.

SỬ DỤNG TRÊN SERVER:
--------------------
  python scripts/benchmark_llm_sidecar.py --mode direct
  python scripts/benchmark_llm_sidecar.py --mode sidecar --url http://127.0.0.1:7861
  python scripts/benchmark_llm_sidecar.py --mode mock
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tendoo.core.components import ICON_SVG_BY_NAME
from tendoo.core.fonts import FONT_CATALOG
from tendoo.core.style import CATEGORY_FIELD_SLOTS
from tendoo.engine.blocks import ROLE_SCALE
from tendoo.engine.geometry import ZONE_NAMES

VALID_ROLES = set(ROLE_SCALE.keys())
VALID_ICONS = set(ICON_SVG_BY_NAME.keys())
VALID_ZONES = set(ZONE_NAMES)
VALID_EFFECTS = {
    "auto", "embossed", "neon", "chrome", "shadow",
    "led", "engraved", "holographic", "outline",
    "ma_vang", "in_noi", "3d_gold", "neon_glow",
}


# ==============================================================================
# 1. ĐỊNH NGHĨA TEST CASE & DỮ LIỆU ĐỐI CHỨNG
# ==============================================================================

@dataclass
class BenchmarkCase:
    id: str
    name: str
    category: str
    form_fields: Dict[str, str]
    title_value: str
    prompt: str
    style_pref: str = "auto"
    # Kỳ vọng kiểm thử:
    expected_title_action: Optional[str] = None  # use_prompt | generate | none
    expected_title_keywords: List[str] = field(default_factory=list)
    expected_zones: List[str] = field(default_factory=list)
    expected_roles: List[str] = field(default_factory=list)
    expected_icons: List[str] = field(default_factory=list)
    expected_effect: Optional[str] = None
    forbidden_scene_strings: List[str] = field(default_factory=list)
    min_extra_blocks: int = 0
    max_hero_words: int = 7
    notes: str = ""


def build_benchmark_suite() -> List[BenchmarkCase]:
    """12 Ca kiểm thử toàn diện từ thực tế ngành hàng và prompt_test.txt."""
    return [
        # Case 01: Khai trương cà phê - Định vị trung tâm & Đa CTA
        BenchmarkCase(
            id="case_01_coffee_grand_opening",
            name="Khai trương quán cà phê (Định vị trung tâm & CTA)",
            category="opening",
            form_fields={},
            title_value="",
            prompt=(
                "Thiết kế banner khai trương quán coffee phong cách hiện đại sang trọng. "
                "Background quán cafe tone nâu ấm, ly cà phê bốc khói trung tâm. "
                "Text nổi bật lớn ở giữa: “GRAND OPENING”. "
                "Text trên cùng: “MUA 1 TẶNG 1”. "
                "Áp dụng từ 14/05 - 30/05. "
                "Góc dưới bên phải có nút CTA: “Ghé ngay hôm nay!”. "
                "Chữ neon phát quang tinh tế. Tỉ lệ 16:9, chất lượng 4K."
            ),
            expected_title_action="use_prompt",
            expected_title_keywords=["GRAND OPENING"],
            expected_zones=["middle_center", "top_center", "bottom_right"],
            expected_effect="neon",
            forbidden_scene_strings=["GRAND OPENING", "MUA 1 TẶNG 1", "Ghé ngay hôm nay", "16:9", "4K", "4k"],
            min_extra_blocks=2,
            notes="Kiểm tra: Bắt đúng title lớn ở giữa, nhận diện effect neon, lọc sạch chuỗi chữ và 16:9 khỏi scene_prompt.",
        ),

        # Case 02: Biến đổi vóc dáng Gym - Bố cục Chia đôi Before / After
        BenchmarkCase(
            id="case_02_gym_before_after_split",
            name="Gym Fitness Transformation (Chia đôi Before / After)",
            category="feedback",
            form_fields={},
            title_value="",
            prompt=(
                "Tạo ảnh feedback khách hàng tập gym vóc dáng thay đổi ngoạn mục dạng Before After chia đôi 2 hình trái phải. "
                "Tiêu đề chính: “KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY?”. "
                "Nửa bên trái: nhãn “NGÀY 01 • BEFORE (82KG)”. "
                "Nửa bên phải: nhãn “NGÀY 90 • AFTER (72KG 6 MÚI)” kèm icon tên lửa. "
                "Góc dưới bên phải có nút: “GIẢM 20% GÓI PT THÁNG ĐẦU”. "
                "Chữ in nổi 3D mạ vàng. Background chia đôi: bên trái xám than chì mờ, bên phải đỏ đen cơ bắp."
            ),
            expected_title_action="use_prompt",
            expected_title_keywords=["KHÁCH HÀNG NÓI GÌ"],
            expected_zones=["top_center", "middle_left", "middle_right", "bottom_right"],
            expected_icons=["rocket"],
            expected_effect="embossed",
            forbidden_scene_strings=["KHÁCH HÀNG NÓI GÌ", "BEFORE", "AFTER", "82KG", "6 MÚI", "GIẢM 20%"],
            min_extra_blocks=3,
            notes="Kiểm tra: Phân bổ đúng middle_left (Before) và middle_right (After), icon rocket, hiệu ứng mạ vàng (embossed).",
        ),

        # Case 03: Ứng dụng Tài chính Fintech - Trích xuất Brief Dày đặc
        BenchmarkCase(
            id="case_03_fintech_app_dense",
            name="Ứng dụng Tài chính WealthMaster (Trích xuất Brief dày)",
            category="product_intro",
            form_fields={},
            title_value="",
            prompt=(
                "Tạo ảnh quảng cáo cho ứng dụng công nghệ tài chính. "
                "Tiêu đề: “Quản lý chi tiêu thông minh, tiền đẻ ra tiền”. "
                "Tên sản phẩm: “WealthMaster App”. "
                "Mô tả ngắn: “Người dùng đã tiết kiệm được 30% thu nhập mỗi tháng nhờ hệ thống cảnh báo chi tiêu và AI phân tích”. "
                "Điểm nổi bật: “Đồng bộ giao dịch tự động từ 50+ ngân hàng, bảo mật chuẩn quân đội”. "
                "Ưu đãi: “Miễn phí nâng cấp tài khoản Premium 6 tháng khi tải app hôm nay”. "
                "Background các biểu đồ tài chính tăng trưởng mờ ảo màu xanh lá, tone xanh navy đậm."
            ),
            expected_title_action="use_prompt",
            expected_title_keywords=["Quản lý chi tiêu"],
            expected_roles=["hero", "badge", "body"],
            expected_icons=["chart_up", "shield", "gift"],
            forbidden_scene_strings=["Quản lý chi tiêu", "WealthMaster", "30%", "Premium"],
            min_extra_blocks=3,
            max_hero_words=8,
            notes="Kiểm tra: Không nhét cả đoạn mô tả vào hero; phân tách đúng role; scene_prompt tiếng Anh thuần cảnh biểu đồ và iPhone 3D.",
        ),

        # Case 04: Nồi Chiên Hơi Nước - Đánh giá 5 Sao & Quote Review
        BenchmarkCase(
            id="case_04_kitchen_feedback_quote",
            name="Nồi chiên hơi nước ChefPro (Review 5 sao & Quote)",
            category="feedback",
            form_fields={},
            title_value="",
            prompt=(
                "Tạo ảnh feedback khách hàng cho thiết bị nhà bếp. "
                "Tiêu đề: “Món nướng ngoài giòn trong mọng nước, mẹ nhàn tênh”. "
                "Tên sản phẩm: “Nồi Chiên Hơi Nước ChefPro 15L”. "
                "Mô tả ngắn feedback: “Các bà nội trợ cực kỳ hài lòng vì đồ ăn không bị khô khốc, giữ trọn vẹn dinh dưỡng”. "
                "Ưu đãi: “Tặng bộ phụ kiện 5 món độc quyền”. "
                "Hiển thị đánh giá 5 sao. Tone màu đen bóng của thiết bị và vàng cam của thức ăn, gà quay bốc khói."
            ),
            expected_title_action="use_prompt",
            expected_title_keywords=["Món nướng"],
            expected_roles=["hero", "body", "badge"],
            expected_icons=["star", "gift"],
            forbidden_scene_strings=["ChefPro", "Món nướng", "giòn", "5 sao"],
            min_extra_blocks=2,
            notes="Kiểm tra: Nhận diện review 5 sao gán icon star; quote vào body; không vẽ chữ lên gà quay trong scene_prompt.",
        ),

        # Case 05: Chụp Ảnh Cưới Studio - Typography Bay Bổng Lãng Mạn
        BenchmarkCase(
            id="case_05_wedding_studio_luxury",
            name="Studio Ảnh Cưới Cinematic Love (Thẩm mỹ & Font)",
            category="promo",
            form_fields={},
            title_value="",
            prompt=(
                "Tạo ảnh quảng cáo cho dịch vụ studio cưới cao cấp. "
                "Tiêu đề: “Lưu giữ khoảnh khắc thanh xuân rực rỡ nhất”. "
                "Tên dịch vụ: “Gói Chụp Ảnh Cưới Cinematic Love”. "
                "Ưu đãi: “Tặng ngay 1 ảnh cổng tráng gương pha lê trị giá 3 triệu”. "
                "Tone màu trắng, be và ánh sáng vàng ấm golden hour, typography font chữ ký bay bổng mềm mại lãng mạn."
            ),
            expected_title_action="use_prompt",
            expected_title_keywords=["thanh xuân"],
            forbidden_scene_strings=["Cinematic Love", "thanh xuân", "3 triệu"],
            min_extra_blocks=1,
            notes="Kiểm tra: Gợi ý font lãng mạn (playfair hoặc alex_brush), style_hint ấm áp, scene_prompt lãng mạn không có text.",
        ),

        # Case 06: Nệm Lò Xo CloudSleep - Thông Số Kỹ Thuật
        BenchmarkCase(
            id="case_06_sleep_mattress_specs",
            name="Nệm Lò Xo CloudSleep (Thông số & Trải nghiệm)",
            category="feedback",
            form_fields={},
            title_value="",
            prompt=(
                "Tạo ảnh feedback khách hàng cho sản phẩm chăm sóc giấc ngủ. "
                "Tiêu đề: “Tạm biệt đau lưng, ngủ sâu giấc đến sáng”. "
                "Sản phẩm: “Nệm Lò Xo Túi CloudSleep”. "
                "Đặc điểm: “99% khách hàng phản hồi không còn đau mỏi vai gáy, lò xo túi độc lập 5 vùng”. "
                "Ưu đãi: “Trải nghiệm miễn phí 100 đêm, tặng 2 gối lông vũ”. "
                "Hình ảnh 3D góc cắt lớp của nệm, tone midnight blue êm ái."
            ),
            expected_title_action="use_prompt",
            expected_title_keywords=["đau lưng"],
            forbidden_scene_strings=["CloudSleep", "đau lưng", "100 đêm"],
            min_extra_blocks=2,
            notes="Kiểm tra: Trích xuất trọn vẹn ưu đãi 100 đêm và đặc tính kỹ thuật 5 vùng mà không làm rối layout.",
        ),

        # Case 07: Chống Rò Rỉ Kích Thước & Từ Khóa Cấm (Anti-Pollution Stress Test)
        BenchmarkCase(
            id="case_07_anti_pollution_stress",
            name="Chống rò rỉ độ phân giải & text (Anti-Pollution)",
            category="product_intro",
            form_fields={},
            title_value="",
            prompt=(
                "Quảng cáo tai nghe Bluetooth Sony WH-1000XM5 chính hãng. "
                "Tỉ lệ 9:16 story, chuẩn 8K, 4K siêu sắc nét, 1080p photorealistic. "
                "Tiêu đề ở góc trên: “ĐỈNH CAO ÂM THANH KHÔNG GIAN”. "
                "Dưới đáy: “Chống ồn chủ động đỉnh cao 99%”. "
                "Hiệu ứng chữ bóng đổ studio sâu thẳm."
            ),
            expected_title_action="use_prompt",
            expected_title_keywords=["ĐỈNH CAO"],
            expected_zones=["top_center", "top_left", "bottom_left", "bottom_center"],
            expected_effect="shadow",
            forbidden_scene_strings=["9:16", "16:9", "8K", "8k", "4K", "4k", "1080p", "ĐỈNH CAO ÂM THANH"],
            min_extra_blocks=1,
            notes="QUAN TRỌNG: scene_prompt TUYỆT ĐỐI không chứa các chuỗi rác như '9:16', '8k', '4k' khiến DiT vẽ số bậy lên ảnh.",
        ),

        # Case 08: Chống Trùng Lặp Nội Dung (Deduplication Stress Test)
        BenchmarkCase(
            id="case_08_dedup_form_and_prompt",
            name="Chống trùng lặp nội dung (Dedup Protection)",
            category="promo",
            form_fields={
                "title": "SIÊU SALE MÙA HÈ",
                "discount": "GIẢM ĐẾN 50%",
                "date_start": "01/06/2026",
                "date_end": "15/06/2026",
            },
            title_value="SIÊU SALE MÙA HÈ",
            prompt=(
                "Banner khuyến mại rực rỡ mùa hè. "
                "Tiêu đề SIÊU SALE MÙA HÈ, áp dụng giảm đến 50% từ 01/06/2026 đến 15/06/2026. "
                "Yêu cầu thêm một nút kêu gọi hành động ở góc dưới bên phải: “SĂN DEAL NGAY KẺO HẾT” kèm icon hộp quà."
            ),
            expected_title_action="none",  # Form đã có title, prompt lặp lại y hệt -> không cần action use_prompt/generate
            expected_zones=["bottom_right"],
            expected_icons=["gift"],
            forbidden_scene_strings=["SIÊU SALE", "GIẢM ĐẾN 50%", "SĂN DEAL"],
            min_extra_blocks=1,
            notes="Kiểm tra: Nhận biết title và discount đã có trong form -> KHÔNG tạo block trùng lặp, CHỈ tạo nút SĂN DEAL NGAY.",
        ),

        # Case 09: Tự Động Sinh Tiêu Đề Khi Form Trống (Auto Title Generation)
        BenchmarkCase(
            id="case_09_auto_title_generation",
            name="Tự động sáng tác Tiêu đề phù hợp ngữ cảnh",
            category="recruitment",
            form_fields={
                "job_position": "Senior Backend Python / AI Engineer",
                "salary": "2500$ - 3500$",
            },
            title_value="",
            prompt=(
                "Công ty công nghệ Tendoo tuyển dụng lập trình viên Backend AI cấp cao. "
                "Lương thưởng hấp dẫn, môi trường làm việc hybrid tự do sáng tạo, văn phòng quận Cầu Giấy. "
                "Phong cách công nghệ tương lai hiện đại."
            ),
            expected_title_action="generate",  # Title trống, prompt không có cụm trong ngoặc kép -> phải tự sáng tác
            expected_roles=["hero"],
            forbidden_scene_strings=["Backend", "Python", "2500$"],
            min_extra_blocks=0,
            max_hero_words=6,
            notes="Kiểm tra: action='generate', tự nghĩ ra tiêu đề tuyển dụng ngắn gọn hấp dẫn (vd: GIA NHẬP ĐỘI NGŨ TENDOO).",
        ),

        # Case 10: Thử Thách 4 Góc Không Gian (Quad-Corner Spatial Challenge)
        BenchmarkCase(
            id="case_10_quad_corner_spatial",
            name="Định vị 4 góc màn hình (Spatial Precision)",
            category="promo",
            form_fields={},
            title_value="",
            prompt=(
                "Banner giải đấu Esports game sinh tồn. "
                "Góc trên bên trái đặt nhãn: “GIẢI ĐẤU MÙA THU 2026”. "
                "Góc trên bên phải đặt ngày: “KHỞI TRANH 20/10”. "
                "Góc dưới bên trái hiển thị: “TỔNG THƯỞNG 500 TRIỆU” kèm icon cúp vàng. "
                "Góc dưới bên phải có nút: “ĐĂNG KÝ THI ĐẤU NGAY”. "
                "Chữ kim loại chrome bạch kim sáng loáng."
            ),
            expected_title_action="none",  # Không có hero ở giữa, toàn bộ là các góc
            expected_zones=["top_left", "top_right", "bottom_left", "bottom_right"],
            expected_icons=["trophy"],
            expected_effect="chrome",
            forbidden_scene_strings=["GIẢI ĐẤU", "500 TRIỆU", "20/10", "ĐĂNG KÝ"],
            min_extra_blocks=3,
            notes="Kiểm tra: Nhận diện chính xác 4 góc không gian; effect chrome; không ép tạo hero ở giữa khi người dùng chỉ định 4 góc.",
        ),

        # Case 11: Nhận Diện Icon Vector Ngữ Nghĩa (Vector Icon Matcher)
        BenchmarkCase(
            id="case_11_semantic_icons_challenge",
            name="Nhận diện Icon Vector theo ngữ cảnh",
            category="promo",
            form_fields={},
            title_value="",
            prompt=(
                "Quảng cáo dịch vụ giao vận hỏa tốc liên tỉnh. "
                "Tiêu đề: “GIAO HÀNG TỐC ĐỘ ÁNH SÁNG”. "
                "Dòng 1: “Tốc độ tên lửa siêu nhanh” kèm biểu tượng tên lửa. "
                "Dòng 2: “Bảo mật kiện hàng 100%” kèm biểu tượng khiên bảo vệ. "
                "Dòng 3: “Tổng đài hỗ trợ 24/7” kèm biểu tượng điện thoại: 1900 6868."
            ),
            expected_title_action="use_prompt",
            expected_title_keywords=["TỐC ĐỘ ÁNH SÁNG"],
            expected_icons=["rocket", "shield", "phone"],
            min_extra_blocks=3,
            notes="Kiểm tra: Bắt chuẩn 3 icons rocket, shield, phone từ ngôn ngữ tự nhiên tiếng Việt.",
        ),

        # Case 12: Prompt Cực Ngắn - Khả Năng Xử Lý Tối Thiểu (Minimal Edge Case)
        BenchmarkCase(
            id="case_12_minimal_extreme_brief",
            name="Prompt cực ngắn (Graceful Minimal Handling)",
            category="opening",
            form_fields={},
            title_value="",
            prompt="Phòng khám nha khoa cao cấp Dr. Smile.",
            expected_title_action="generate",
            forbidden_scene_strings=["Dr. Smile", "Nha khoa", "Phòng khám"],
            min_extra_blocks=0,
            notes="Kiểm tra: Không crash khi prompt chỉ có 1 câu ngắn; tự sinh title nha khoa và scene_prompt phòng khám sạch sẽ bằng tiếng Anh.",
        ),
    ]


# ==============================================================================
# 2. BỘ CHẤM ĐIỂM ĐỊNH LƯỢNG (AUTOMATED QUANTITATIVE RUBRIC)
# ==============================================================================

@dataclass
class CaseScoreResult:
    case_id: str
    case_name: str
    is_json_valid: bool
    json_score: float               # max 20
    schema_score: float             # max 20
    spatial_score: float            # max 20
    scene_cleanliness_score: float  # max 20
    entity_recall_score: float      # max 20
    total_score: float              # max 100
    latency_sec: float
    raw_response: str
    parsed_plan: Optional[Dict[str, Any]]
    validation_errors: List[str]
    audit_notes: List[str]

    @property
    def passed(self) -> bool:
        return self.total_score >= 80.0


def evaluate_case_output(
    case: BenchmarkCase,
    raw_response: str,
    latency_sec: float,
) -> CaseScoreResult:
    """Chấm điểm định lượng từng trường hợp theo thang điểm 100 chuẩn khoa học."""
    audit: List[str] = []
    val_errors: List[str] = []

    # --------------------------------------------------------------------------
    # 1. JSON Syntax Score (Max 20 điểm)
    # --------------------------------------------------------------------------
    json_score = 0.0
    parsed_plan = None
    is_valid_json = False

    # Tìm khối JSON
    from tendoo.llm_render_plan_server import parse_render_plan_json, validate_render_plan
    parsed_candidate, parse_errs = parse_render_plan_json(raw_response)

    if parsed_candidate is not None:
        is_valid_json = True
        json_score = 20.0
        audit.append("[PASS] JSON hợp lệ 100%")
    else:
        json_score = 0.0
        val_errors.extend(parse_errs)
        audit.append(f"[FAIL] Lỗi phân tích cú pháp JSON: {parse_errs}")

    # --------------------------------------------------------------------------
    # 2. Schema & Zone Validity Score (Max 20 điểm)
    # --------------------------------------------------------------------------
    schema_score = 20.0
    cleaned_plan = None

    if is_valid_json and isinstance(parsed_candidate, dict):
        cleaned_plan, val_errs = validate_render_plan(parsed_candidate, case.category, case.form_fields)
        val_errors.extend(val_errs)

        # Kiểm tra các lỗi nghiêm trọng về Zone/Role
        for err in val_errs:
            if "unknown zone" in err:
                schema_score = max(0.0, schema_score - 7.0)
                audit.append(f"[TRỪ ĐIỂM] Hallucinate Zone không hợp lệ: {err}")
            elif "unknown role" in err:
                schema_score = max(0.0, schema_score - 5.0)
                audit.append(f"[TRỪ ĐIỂM] Hallucinate Role không hợp lệ: {err}")
            elif "unknown title.action" in err:
                schema_score = max(0.0, schema_score - 4.0)
                audit.append(f"[TRỪ ĐIỂM] Action Title sai: {err}")
            elif "missing/empty 'scene_prompt'" in err:
                schema_score = max(0.0, schema_score - 10.0)
                audit.append(f"[TRỪ ĐIỂM] Thiếu scene_prompt")
    else:
        schema_score = 0.0

    # --------------------------------------------------------------------------
    # 3. Spatial Placement Accuracy Score (Max 20 điểm)
    # --------------------------------------------------------------------------
    spatial_score = 20.0
    if cleaned_plan:
        found_zones = set()
        for b in cleaned_plan.get("extra_blocks", []):
            if b.get("zone"):
                found_zones.add(b["zone"])

        if case.expected_zones:
            matched_zones = set(case.expected_zones).intersection(found_zones)
            recall = len(matched_zones) / len(case.expected_zones)
            spatial_score = round(20.0 * recall, 1)
            if recall < 1.0:
                missing = set(case.expected_zones) - found_zones
                audit.append(f"[TRỪ ĐIỂM] Thiếu các Zone mong đợi: {missing} (Đạt {len(matched_zones)}/{len(case.expected_zones)})")
            else:
                audit.append(f"[PASS] Định vị không gian khớp 100%: {matched_zones}")
    else:
        spatial_score = 0.0

    # --------------------------------------------------------------------------
    # 4. Scene Prompt Cleanliness & Anti-Pollution Score (Max 20 điểm)
    # --------------------------------------------------------------------------
    scene_score = 20.0
    if cleaned_plan and cleaned_plan.get("scene_prompt"):
        sp = cleaned_plan["scene_prompt"]

        # Kiểm tra rò rỉ từ cấm
        leakages = []
        for forbidden in case.forbidden_scene_strings:
            if forbidden.lower() in sp.lower():
                leakages.append(forbidden)

        if leakages:
            scene_score = max(0.0, scene_score - len(leakages) * 7.0)
            audit.append(f"[CẢNH BÁO RÒ RỈ] Scene prompt dính từ cấm/chuỗi text/kích thước: {leakages}")

        # Kiểm tra tiếng Việt sót lại trong scene_prompt
        vietnamese_chars = re.findall(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", sp.lower())
        if len(vietnamese_chars) >= 3:
            scene_score = max(0.0, scene_score - 10.0)
            audit.append("[TRỪ ĐIỂM] Scene prompt chưa dịch trọn vẹn sang tiếng Anh, vẫn dính tiếng Việt")
        elif not leakages:
            audit.append("[PASS] Scene prompt sạch 100%, không rò rỉ text/tỉ lệ")
    else:
        scene_score = 0.0

    # --------------------------------------------------------------------------
    # 5. Entity Extraction, Role & Sizing Score (Max 20 điểm)
    # --------------------------------------------------------------------------
    entity_score = 20.0
    if cleaned_plan:
        title_obj = cleaned_plan.get("title", {})
        action = title_obj.get("action")
        title_text = title_obj.get("text") or ""

        # Kiểm tra Title Action
        if case.expected_title_action and action != case.expected_title_action:
            entity_score = max(0.0, entity_score - 8.0)
            audit.append(f"[TRỪ ĐIỂM] Title action kỳ vọng '{case.expected_title_action}' nhưng model xuất '{action}'")

        # Kiểm tra Title Length (Chống Hero quá dài gây vỡ layout)
        if title_text:
            word_count = len(title_text.split())
            if word_count > case.max_hero_words:
                entity_score = max(0.0, entity_score - 7.0)
                audit.append(f"[TRỪ ĐIỂM] Tiêu đề quá dài ({word_count} từ > {case.max_hero_words} từ cho phép)")

        # Kiểm tra từ khóa tiêu đề nếu có
        if case.expected_title_keywords and title_text:
            has_kw = any(kw.lower() in title_text.lower() for kw in case.expected_title_keywords)
            if not has_kw:
                entity_score = max(0.0, entity_score - 5.0)
                audit.append(f"[TRỪ ĐIỂM] Tiêu đề thiếu từ khóa kỳ vọng {case.expected_title_keywords}")

        # Kiểm tra số lượng extra blocks tối thiểu
        blocks_count = len(cleaned_plan.get("extra_blocks", []))
        if blocks_count < case.min_extra_blocks:
            entity_score = max(0.0, entity_score - 5.0)
            audit.append(f"[TRỪ ĐIỂM] Trích xuất thiếu blocks ({blocks_count} < tối thiểu {case.min_extra_blocks})")

        # Kiểm tra Effect nếu có kỳ vọng
        if case.expected_effect:
            eff = cleaned_plan.get("headline_effect", "auto")
            if eff == case.expected_effect:
                audit.append(f"[PASS] Bắt đúng Headline Effect: {eff}")
            else:
                entity_score = max(0.0, entity_score - 3.0)
                audit.append(f"[TRỪ ĐIỂM] Effect kỳ vọng '{case.expected_effect}', nhận '{eff}'")

        # Kiểm tra Icons nếu có kỳ vọng
        if case.expected_icons:
            found_icons = {b.get("icon") for b in cleaned_plan.get("extra_blocks", []) if b.get("icon")}
            matched_icons = set(case.expected_icons).intersection(found_icons)
            if matched_icons:
                audit.append(f"[PASS] Gán đúng Icons ngữ nghĩa: {matched_icons}")
            else:
                entity_score = max(0.0, entity_score - 4.0)
                audit.append(f"[TRỪ ĐIỂM] Thiếu các Icons kỳ vọng: {case.expected_icons}")
    else:
        entity_score = 0.0

    total = round(json_score + schema_score + spatial_score + scene_score + entity_score, 1)

    return CaseScoreResult(
        case_id=case.id,
        case_name=case.name,
        is_json_valid=is_valid_json,
        json_score=json_score,
        schema_score=schema_score,
        spatial_score=spatial_score,
        scene_cleanliness_score=scene_score,
        entity_recall_score=entity_score,
        total_score=total,
        latency_sec=latency_sec,
        raw_response=raw_response,
        parsed_plan=cleaned_plan,
        validation_errors=val_errors,
        audit_notes=audit,
    )


# ==============================================================================
# 3. TRÌNH MÔ PHỎNG & CLIENT KẾT NỐI
# ==============================================================================

def mock_llm_response(case: BenchmarkCase) -> str:
    """Phản hồi mẫu giả lập cho chế độ offline/mock kiểm tra rubric chấm điểm."""
    if case.id == "case_01_coffee_grand_opening":
        return json.dumps({
            "title": {"action": "use_prompt", "text": "GRAND OPENING"},
            "extra_blocks": [
                {"field": None, "text": "MUA 1 TẶNG 1", "zone": "top_center", "role": "badge", "icon": "sparkles"},
                {"field": "opening_promo", "text": "Áp dụng từ 14/05 - 30/05", "zone": "top_center", "role": "subtitle", "icon": "calendar"},
                {"field": None, "text": "Ghé ngay hôm nay!", "zone": "bottom_right", "role": "badge", "icon": "gift"}
            ],
            "scene_prompt": "A modern luxury coffee shop interior with warm brown tones and cinematic golden lighting, an exquisite steaming cup of coffee in the center, floating roasted coffee beans, high depth of field, 8k commercial photography",
            "style_hint": "warm_cafe",
            "font_key": "playfair",
            "headline_effect": "neon"
        }, ensure_ascii=False)

    elif case.id == "case_02_gym_before_after_split":
        return json.dumps({
            "title": {"action": "use_prompt", "text": "KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY?"},
            "extra_blocks": [
                {"field": None, "text": "NGÀY 01 • BEFORE (82KG)", "zone": "middle_left", "role": "badge", "icon": "clock"},
                {"field": None, "text": "NGÀY 90 • AFTER (72KG 6 MÚI)", "zone": "middle_right", "role": "badge", "icon": "rocket"},
                {"field": None, "text": "GIẢM 20% GÓI PT THÁNG ĐẦU", "zone": "bottom_right", "role": "badge", "icon": "gift"}
            ],
            "scene_prompt": "A vertical split-screen fitness advertisement photography, left side showing a man slightly out of shape in dim cool slate gym lighting, right side showing a dramatic athletic transformation with shredded six pack abs under high contrast crimson rim-light, central dividing line",
            "style_hint": "athletic_split",
            "font_key": "playfair",
            "headline_effect": "embossed"
        }, ensure_ascii=False)

    elif case.id == "case_07_anti_pollution_stress":
        # Giả lập model xử lý sạch sẽ không rò rỉ 9:16 hay 8k
        return json.dumps({
            "title": {"action": "use_prompt", "text": "ĐỈNH CAO ÂM THANH KHÔNG GIAN"},
            "extra_blocks": [
                {"field": None, "text": "Chống ồn chủ động đỉnh cao 99%", "zone": "bottom_left", "role": "body", "icon": "zap"}
            ],
            "scene_prompt": "Premium flagship over-ear wireless headphones floating gracefully against a dark minimalist studio backdrop, subtle blue rim lighting, product commercial photography, sleek matte texture",
            "style_hint": "dark_studio",
            "font_key": "montserrat",
            "headline_effect": "shadow"
        }, ensure_ascii=False)

    elif case.id == "case_08_dedup_form_and_prompt":
        return json.dumps({
            "title": {"action": "none", "text": None},
            "extra_blocks": [
                {"field": None, "text": "SĂN DEAL NGAY KẺO HẾT", "zone": "bottom_right", "role": "badge", "icon": "gift"}
            ],
            "scene_prompt": "A vibrant summer promotional scene with bright golden sunlight and tropical elements, clean photography background",
            "style_hint": "daylight",
            "font_key": "be_vietnam_pro",
            "headline_effect": "auto"
        }, ensure_ascii=False)

    elif case.id == "case_09_auto_title_generation":
        return json.dumps({
            "title": {"action": "generate", "text": "GIA NHẬP ĐỘI NGŨ TENDOO"},
            "extra_blocks": [
                {"field": "job_position", "text": "Senior Backend Python / AI Engineer", "zone": "middle_left", "role": "body", "icon": "brain"},
                {"field": "salary", "text": "2500$ - 3500$", "zone": "middle_left", "role": "badge", "icon": "sparkles"}
            ],
            "scene_prompt": "A futuristic high-tech AI software company office with neon digital data visualization, ultra modern workspace, soft cinematic lighting",
            "style_hint": "cyber_tech",
            "font_key": "be_vietnam_pro",
            "headline_effect": "led"
        }, ensure_ascii=False)

    else:
        # Fallback tổng quát cho các case khác
        return json.dumps({
            "title": {"action": "use_prompt" if "Tiêu đề" in case.prompt else "generate", "text": "KHÁM PHÁ NGAY HÔM NAY"},
            "extra_blocks": [
                {"field": None, "text": "Ưu đãi độc quyền", "zone": "bottom_right", "role": "badge", "icon": "gift"},
                {"field": None, "text": "Chất lượng vượt trội", "zone": "bottom_left", "role": "body", "icon": "shield"}
            ],
            "scene_prompt": "A high-end clean commercial photography scene perfectly lit for professional product advertising, soft studio lighting, ultra sharp depth of field",
            "style_hint": "auto",
            "font_key": "auto",
            "headline_effect": "auto"
        }, ensure_ascii=False)


def call_sidecar_http(url: str, case: BenchmarkCase, timeout: float = 30.0) -> Tuple[str, float]:
    """Gửi HTTP request tới FastAPI sidecar đang chạy."""
    endpoint = f"{url.rstrip('/')}/api/render-plan"
    payload = {
        "category": case.category,
        "category_fields": case.form_fields,
        "title_value": case.title_value,
        "prompt": case.prompt,
        "style_pref": case.style_pref,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            t1 = time.perf_counter()
            body = resp.read().decode("utf-8")
            return body, (t1 - t0)
    except Exception as e:
        t1 = time.perf_counter()
        return f'{{"error": "{str(e)}"}}', (t1 - t0)


def generate_with_transformers(
    model: Any,
    tokenizer: Any,
    device: str,
    case: BenchmarkCase,
) -> Tuple[str, float]:
    """Chạy inference trực tiếp qua GPU Transformers."""
    from tendoo.llm_render_plan_server import build_messages, generate_raw_response

    messages = build_messages(
        category=case.category,
        category_fields=case.form_fields,
        title_value=case.title_value,
        prompt=case.prompt,
        style_pref=case.style_pref,
    )
    t0 = time.perf_counter()
    raw = generate_raw_response(model, tokenizer, device, messages, max_new_tokens=1024)
    t1 = time.perf_counter()
    return raw, (t1 - t0)


# ==============================================================================
# 4. TRÌNH THỰC THI & XUẤT BÁO CÁO TOÀN DIỆN
# ==============================================================================

def run_benchmark(
    mode: str = "mock",
    model_path: Optional[str] = None,
    sidecar_url: str = "http://127.0.0.1:7861",
    device: str = "cuda:0",
    out_dir_path: str = "output_llm_benchmark",
) -> Dict[str, Any]:
    """Chạy toàn bộ quy trình benchmark và tính toán kết luận."""
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("BỘ ĐÁNH GIÁ ĐỊNH LƯỢNG NĂNG LỰC LLM SIDECAR (QWEN) - TENDOO AI")
    print(f"Chế độ thực thi: {mode.upper()} | Thiết bị: {device}")
    print("=" * 90 + "\n")

    cases = build_benchmark_suite()
    model = None
    tokenizer = None

    if mode == "direct":
        print(f"[1/3] Đang nạp mô hình từ đường dẫn: {model_path}...")
        from tendoo.llm_render_plan_server import _resolve_qwen3_checkpoint_path, load_model
        actual_path = model_path or _resolve_qwen3_checkpoint_path("4b")
        print(f"      Resolved model path: {actual_path}")
        model, tokenizer = load_model(actual_path, device=device)
        print("      [Nạp mô hình thành công lên GPU!]")
    elif mode == "sidecar":
        print(f"[1/3] Đang kết nối tới HTTP Sidecar service tại: {sidecar_url}...")
    else:
        print("[1/3] Đang chạy ở chế độ MOCK (Kiểm tra logic rubric & rubric scoring offline)...")

    results: List[CaseScoreResult] = []

    print(f"[2/3] Bắt đầu đánh giá {len(cases)} ca kiểm thử tiêu chuẩn...")
    for idx, case in enumerate(cases, 1):
        print(f"\n--- [{idx:02d}/{len(cases):02d}] {case.name} ({case.id}) ---")

        if mode == "direct":
            raw_output, latency = generate_with_transformers(model, tokenizer, device, case)
        elif mode == "sidecar":
            raw_output, latency = call_sidecar_http(sidecar_url, case)
        else:
            t0 = time.perf_counter()
            raw_output = mock_llm_response(case)
            latency = time.perf_counter() - t0

        res = evaluate_case_output(case, raw_output, latency)
        results.append(res)

        status_tag = "PASS" if res.passed else "FAIL"
        print(f"    -> Kết quả: [{status_tag}] | Tổng điểm: {res.total_score}/100 | Thời gian: {latency:.2f}s")
        print(f"       Chi tiết điểm: JSON={res.json_score}/20, Schema={res.schema_score}/20, KhôngGian={res.spatial_score}/20, SceneClean={res.scene_cleanliness_score}/20, ThựcThể={res.entity_recall_score}/20")
        for audit_msg in res.audit_notes:
            print(f"       * {audit_msg}")

    # ==========================================================================
    # TỔNG HỢP CHỈ SỐ & BẢNG KẾT LUẬN TOÀN DIỆN
    # ==========================================================================
    avg_total = sum(r.total_score for r in results) / len(results)
    avg_json = sum(r.json_score for r in results) / len(results)
    avg_schema = sum(r.schema_score for r in results) / len(results)
    avg_spatial = sum(r.spatial_score for r in results) / len(results)
    avg_scene = sum(r.scene_cleanliness_score for r in results) / len(results)
    avg_entity = sum(r.entity_recall_score for r in results) / len(results)
    avg_latency = sum(r.latency_sec for r in results) / len(results)
    pass_count = sum(1 for r in results if r.passed)
    pass_rate = (pass_count / len(results)) * 100.0

    # Phán quyết khách quan dựa trên ngưỡng điểm
    if avg_total >= 88.0 and pass_rate >= 83.0 and avg_json >= 19.0:
        verdict = "ĐỦ KHẢ NĂNG (SUFFICIENT - READY FOR PRODUCTION)"
        verdict_color = "XANH (TỐT)"
        recommendation = (
            "Mô hình Qwen3-4B-FP8 hoàn toàn đủ năng lực đảm nhiệm vai trò Sidecar Dàn Trang & Sinh Scene Prompt. "
            "Tỉ lệ tuân thủ JSON cao, bóc tách thực thể chuẩn xác, không rò rỉ text vào diffusion. "
            "KHÔNG CẦN nâng cấp lên mô hình lớn hơn (7B/14B), tiết kiệm tối đa VRAM cho DiT và LoRA!"
        )
    elif avg_total >= 75.0:
        verdict = "CẦN TỐI ƯU THÊM (CONDITIONAL / MARGINAL)"
        verdict_color = "VÀNG (CẢNH BÁO)"
        recommendation = (
            "Qwen3-4B-FP8 xử lý được các ca cơ bản nhưng có sai sót ở các ca không gian phức tạp hoặc rò rỉ text nhẹ. "
            "Khuyến nghị: Giữ 4B nhưng bổ sung lớp Guardrail Regex Sanitize ở tầng Python "
            "hoặc cân nhắc kiểm thử thêm Qwen2.5-7B-Instruct nếu yêu cầu độ chính xác 100% không qua sanitize."
        )
    else:
        verdict = "KHÔNG ĐỦ KHẢ NĂNG (INSUFFICIENT - MUST UPGRADE)"
        verdict_color = "ĐỎ (KHÔNG ĐẠT)"
        recommendation = (
            "Mô hình 4B thường xuyên vỡ cú pháp JSON, hallucinate zone ngoài danh mục hoặc rò rỉ text nghiêm trọng. "
            "BẮT BUỘC NÂNG CẤP: Chuyển sang Qwen2.5-7B-Instruct hoặc Qwen2.5-14B-Instruct-GPTQ, hoặc tích hợp API ngoài."
        )

    summary = {
        "mode": mode,
        "total_cases": len(cases),
        "passed_cases": pass_count,
        "pass_rate_percent": round(pass_rate, 1),
        "avg_total_score": round(avg_total, 1),
        "avg_json_score": round(avg_json, 1),
        "avg_schema_score": round(avg_schema, 1),
        "avg_spatial_score": round(avg_spatial, 1),
        "avg_scene_cleanliness_score": round(avg_scene, 1),
        "avg_entity_recall_score": round(avg_entity, 1),
        "avg_latency_sec": round(avg_latency, 2),
        "verdict": verdict,
        "verdict_level": verdict_color,
        "recommendation": recommendation,
    }

    # Xuất file JSON chi tiết
    json_report = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    (out_dir / "benchmark_report.json").write_text(json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Xuất file Markdown Báo cáo
    md_content = f"""# BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG NĂNG LỰC LLM SIDECAR (QWEN)

- **Thời gian thực hiện**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Chế độ kiểm thử**: `{mode}`
- **Số lượng ca kiểm thử**: {len(cases)}
- **Tỉ lệ đạt (Pass Rate)**: **{pass_rate:.1f}%** ({pass_count}/{len(cases)})
- **Điểm số trung bình toàn diện**: **{avg_total:.1f} / 100**
- **Độ trễ trung bình mỗi prompt**: **{avg_latency:.2f}s**

---

## 1. KẾT LUẬN & KHUYẾN NGHỊ LỰA CHỌN MÔ HÌNH (VERDICT)

### 👉 KẾT QUẢ: **{verdict}** (Mức độ: {verdict_color})

**Nhận định kỹ thuật**:
{recommendation}

---

## 2. BẢNG ĐIỂM CHI TIẾT THEO 5 TRỤC NĂNG LỰC

| Trục Đánh Giá | Điểm Trung Bình (Max 20) | Tỉ Lệ Đạt Chuẩn | Nhận Xét Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| **1. Cú pháp JSON (Syntax)** | **{avg_json:.1f}** | {(avg_json/20)*100:.1f}% | Đo đạc độ ổn định cấu trúc JSON, không lỗi cú pháp. |
| **2. Tuân thủ Schema (Zone/Role)** | **{avg_schema:.1f}** | {(avg_schema/20)*100:.1f}% | Kiểm soát chống hallucinate zone ngoài 9 ô Omni-Block. |
| **3. Định vị Không gian (Spatial)** | **{avg_spatial:.1f}** | {(avg_spatial/20)*100:.1f}% | Khả năng ánh xạ yêu cầu vị trí ("ở trên", "chia đôi"). |
| **4. Độ sạch Scene Prompt (Diffusion)**| **{avg_scene:.1f}** | {(avg_scene/20)*100:.1f}% | Chống rò rỉ text tiếng Việt & thông số "16:9", "4K". |
| **5. Phân tách Thực thể & Sizing** | **{avg_entity:.1f}** | {(avg_entity/20)*100:.1f}% | Phân bổ role hero/badge/quote và giới hạn độ dài hero <= 6 từ. |

---

## 3. KẾT QUẢ CHI TIẾT TỪNG CA KIỂM THỬ

| ID | Tên Ca Kiểm Thử | Trạng Thái | Tổng Điểm | Thời Gian | Ghi Chú Audit Chính |
| :--- | :--- | :---: | :---: | :---: | :--- |
"""
    for r in results:
        status_badge = "✅ PASS" if r.passed else "❌ FAIL"
        key_audit = "; ".join(r.audit_notes[:2])
        md_content += f"| `{r.case_id}` | {r.case_name} | {status_badge} | **{r.total_score:.1f}** | {r.latency_sec:.2f}s | {key_audit} |\n"

    (out_dir / "benchmark_report.md").write_text(md_content, encoding="utf-8")

    # In bảng tổng kết ra console
    print("\n" + "=" * 90)
    print("TỔNG KẾT KẾT QUẢ BENCHMARK LLM SIDECAR")
    print("=" * 90)
    print(f"Tổng số ca kiểm thử   : {len(cases)}")
    print(f"Số ca đạt chuẩn (>=80): {pass_count}/{len(cases)} ({pass_rate:.1f}%)")
    print(f"Điểm số trung bình    : {avg_total:.1f} / 100")
    print(f"Thời gian trung bình  : {avg_latency:.2f} giây / prompt")
    print("-" * 90)
    print(f"PHÁN QUYẾT            : {verdict}")
    print(f"KHUYẾN NGHỊ           : {recommendation}")
    print("=" * 90)
    print(f"Báo cáo chi tiết đã lưu tại:\n  - {out_dir / 'benchmark_report.md'}\n  - {out_dir / 'benchmark_report.json'}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI - LLM Sidecar Comprehensive Benchmark Suite")
    parser.add_argument("--mode", choices=["direct", "sidecar", "mock"], default="mock",
                        help="Chế độ chạy: 'direct' (Transformers GPU), 'sidecar' (HTTP service), 'mock' (offline test).")
    parser.add_argument("--model-path", default=None,
                        help="Đường dẫn thư mục weights Qwen (mặc định tự động dò persistent-data).")
    parser.add_argument("--url", default="http://127.0.0.1:7861",
                        help="URL của LLM Sidecar nếu chạy ở mode sidecar.")
    parser.add_argument("--device", default="cuda:0",
                        help="Thiết bị GPU chạy inference (vd: cuda:0).")
    parser.add_argument("--output-dir", default="output_llm_benchmark",
                        help="Thư mục xuất báo cáo JSON & Markdown.")
    args = parser.parse_args()

    run_benchmark(
        mode=args.mode,
        model_path=args.model_path,
        sidecar_url=args.url,
        device=args.device,
        out_dir_path=args.output_dir,
    )


if __name__ == "__main__":
    main()
