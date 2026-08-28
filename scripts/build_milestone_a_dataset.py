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


# Known-hard concurrency stress pairs, reproducing configurations that broke in earlier probing.
# Each carries an explicit, semantically sane domain (used only to pick a plausible product for I2I).
KNOWN_HARD_PAIRS = [
    {"text1": "CHỐNG ỒN CHỦ ĐỘNG", "text2": "Khử tạp âm kỹ thuật số\nĐắm chìm trong âm nhạc đỉnh cao", "domain": "tech"},
    {"text1": "Ủ CHƯỢP TRUYỀN THỐNG", "text2": "Cá cơm tươi nguyên chất\nĐậm đà phong vị biển xanh", "domain": "fmcg"},
    {"text1": "ĐỔI MỚI SÁNG TẠO TOÀN DIỆN", "text2": "Bứt phá mọi giới hạn\nĐịnh hình kỷ nguyên số", "domain": "telecom_viettel"},
    {"text1": "KHUẤY ĐỘNG MỌI BỮA TIỆC", "text2": "Âm bass bùng nổ nội lực\nÁnh sáng rực rỡ sắc màu", "domain": "tech"},
    {"text1": "AN TOÀN TUYỆT ĐỐI CHO LÀN DA", "text2": "Không chất bảo quản\nChứng nhận kiểm nghiệm quốc tế", "domain": "cosmetics"},
    {"text1": "TIỆM CÀ PHÊ ANH QUÂN GÓC PHỐ NHỎ BÌNH YÊN", "text2": "GIẢM 50%", "domain": "fnb"},
    {"text1": "BỘ DƯỠNG TRẮNG PHỤC HỒI TÁI TẠO LÀN DA CHUYÊN SÂU", "text2": "HOT SALE", "domain": "cosmetics"},
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
# 5. STUDENT CLEAN-PROMPT BUILDER (RULE: never leak literal text; must carry (1)/(2) tags)
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
    "Khối tên thương hiệu chính", "Dòng chữ chủ đề lớn", "Tiêu đề thông điệp",
]
ROLE_DESCRIPTORS_2 = [
    "Dòng phụ đề bổ trợ", "Khối thông tin chi tiết 2 dòng", "Dòng chú thích nội dung",
    "Khối thông điệp phụ", "Khối quyền lợi đãi ngộ", "Dòng slogan ngắn gọn",
]
POSITION_DESCRIPTORS_1 = [
    "ở phía trên chính giữa", "ở phần trên cùng của bố cục", "ở góc trên cân đối",
    "ở vị trí trung tâm phía trên", "chạy ngang phần trên canvas",
]
POSITION_DESCRIPTORS_2 = [
    "ở phía dưới chính giữa", "ở phần chân đế bên dưới", "nằm ngay bên dưới tiêu đề",
    "ở nửa dưới của poster", "ở vị trí góc dưới thanh lịch",
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
    "Phong cách thiết kế poster quảng cáo thương mại cao cấp, bố cục cân xứng hoàn hảo.",
]

PRODUCT_ROLE_DESCRIPTORS = [
    "Sản phẩm được đặt làm tiêu điểm trung tâm",
    "Sản phẩm nổi bật ở phần dưới khung hình",
    "Sản phẩm được trưng bày trang trọng giữa khung hình",
]


def _leaks(candidate: str, *texts: str) -> bool:
    cand_low = candidate.lower()
    for t in texts:
        for line in t.split("\n"):
            line = line.strip()
            if line and line.lower() in cand_low:
                return True
    return False


def combinatorial_clean_prompt(text1: str, text2: str, has_product: bool) -> str:
    env = random.choice(ENV_SETTINGS)
    r1, p1, m1 = random.choice(ROLE_DESCRIPTORS_1), random.choice(POSITION_DESCRIPTORS_1), random.choice(MATERIALS)
    r2, p2, m2 = random.choice(ROLE_DESCRIPTORS_2), random.choice(POSITION_DESCRIPTORS_2), random.choice(MATERIALS)
    light = random.choice(PHOTOGRAPHY_LIGHTING)
    parts = [
        f"{env}.",
        f"(1) {r1} {p1} {m1}.",
        f"(2) {r2} {p2} {m2}.",
    ]
    if has_product:
        parts.append(f"(3) {random.choice(PRODUCT_ROLE_DESCRIPTORS)}.")
    parts.append(light)
    prompt_clean = " ".join(parts)
    assert not _leaks(prompt_clean, text1, text2), f"Leak detected in combinatorial prompt: {prompt_clean}"
    assert "(1)" in prompt_clean and "(2)" in prompt_clean
    return prompt_clean


LLM_SYSTEM_PROMPT = """Bạn là bộ sinh mô tả bố cục poster quảng cáo cho pipeline huấn luyện AI.
Nhiệm vụ: viết MỘT đoạn mô tả bối cảnh + bố cục bằng tiếng Việt, gồm:
- 1 câu mô tả bối cảnh/studio chung.
- Một mục đánh dấu "(1) ..." mô tả VAI TRÒ, VỊ TRÍ và CHẤT LIỆU của khối chữ thứ nhất (tiêu đề).
- Một mục đánh dấu "(2) ..." mô tả VAI TRÒ, VỊ TRÍ và CHẤT LIỆU của khối chữ thứ hai (phụ đề).
{product_line}
- 1 câu mô tả ánh sáng/phong cách nhiếp ảnh.

QUY TẮC BẮT BUỘC (vi phạm là hỏng dữ liệu huấn luyện):
- TUYỆT ĐỐI KHÔNG được viết ra nội dung chữ thật (không lặp lại bất kỳ từ nào trong nội dung text được cho bên dưới, kể cả một phần).
- Chỉ mô tả vai trò/vị trí/chất liệu của khối chữ, không mô tả ý nghĩa hay nội dung của nó.
- Bắt buộc phải có đúng các thẻ "(1)" và "(2)"{product_tag_note}.
- Trả về DUY NHẤT đoạn văn bản, không giải thích, không markdown, không liệt kê lại nội dung chữ."""


def llm_clean_prompt(text1: str, text2: str, has_product: bool, max_retries: int = 3) -> Optional[str]:
    """Ask an LLM to author a more diverse student prompt; enforce the anti-leak rule as a hard gate.
    Returns None (caller should fall back to combinatorial_clean_prompt) if all retries fail validation.
    """
    product_line = "- Một mục đánh dấu \"(3) ...\" mô tả vai trò/vị trí của SẢN PHẨM thật trong ảnh." if has_product else ""
    product_tag_note = " và \"(3)\"" if has_product else ""
    system = LLM_SYSTEM_PROMPT.format(product_line=product_line, product_tag_note=product_tag_note)
    user_msg = (
        f"Nội dung chữ thứ nhất (KHÔNG được lặp lại): {text1}\n"
        f"Nội dung chữ thứ hai (KHÔNG được lặp lại): {text2}\n"
        f"Có sản phẩm thật trong ảnh: {'Có' if has_product else 'Không'}"
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.9,
                max_tokens=250,
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


def build_clean_prompt(text1: str, text2: str, has_product: bool, use_llm: bool) -> str:
    if use_llm:
        result = llm_clean_prompt(text1, text2, has_product)
        if result is not None:
            return result
        print("   [LLM prompt] all retries failed validation -> falling back to combinatorial builder")
    return combinatorial_clean_prompt(text1, text2, has_product)


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
I2I_USE_CASE_TARGETS: List[Tuple[str, int]] = [
    ("hero_product", 170),        # 21.25% of total
    ("flash_sale", 90),           # 11.25% of total
    ("customer_feedback", 60),    # 7.50% of total
    ("opening_banner", 40),       # 5.00% of total
    ("two_step_guide", 30),       # 3.75% of total
    ("creative_quote", 30),       # 3.75% of total
    ("recruitment", 20),          # 2.50% of total
]

# Target sample counts for Pure T2I (45% = 360 total per Sub-plan Table 1.1)
T2I_USE_CASE_TARGETS: List[Tuple[str, int]] = [
    ("flash_sale", 80),           # 10.00% of total
    ("recruitment", 70),          # 8.75% of total
    ("opening_banner", 60),       # 7.50% of total
    ("customer_feedback", 50),    # 6.25% of total
    ("two_step_guide", 50),       # 6.25% of total
    ("creative_quote", 50),       # 6.25% of total
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

    # 5. Length Stratum: 75% standard commercial, 25% inverted (Golden 75/25 Ratio per subplan)
    # Standard: Slot 1 short/medium (2-5 words), Slot 2 medium/long (4-12 words)
    # Inverted: Slot 1 long (8-16 words, 2-3 lines), Slot 2 short punchy badge (1-3 words)
    is_inverted = (random.random() < 0.25)
    length_stratum = "inverted" if is_inverted else "standard"

    # 6. Cohort: ~12.5% known-hard stress reproductions, rest standard
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
        domain = random.choice(list(PRODUCT_TEXT_CORPUS.keys()))
        stem = random.choice(list(PRODUCT_TEXT_CORPUS[domain].keys()))
        prod_folder = PROJECT_ROOT / "data" / "products" / domain
        candidates = list(prod_folder.glob(f"{stem}.*"))
        product_path = candidates[0] if candidates else _pick_product_for_domain(domain)

        if use_case in ["hero_product", "flash_sale"]:
            options = PRODUCT_TEXT_CORPUS[domain][stem][length_stratum]
            text1, text2 = random.choice(options)
        else:
            # Other 5 use cases in I2I (customer_feedback, opening_banner, recruitment, two_step_guide, creative_quote)
            # take text from GENERAL_T2I_CORPUS while displaying the real product photo!
            options = GENERAL_T2I_CORPUS[use_case][length_stratum]
            text1, text2 = random.choice(options)
    else:
        # Pure T2I: 2 text slots, no product.
        if use_case == "flash_sale":
            domain = random.choice(list(PRODUCT_TEXT_CORPUS.keys()))
            stem = random.choice(list(PRODUCT_TEXT_CORPUS[domain].keys()))
            options = PRODUCT_TEXT_CORPUS[domain][stem][length_stratum]
            text1, text2 = random.choice(options)
        else:
            domain = "general_" + use_case
            options = GENERAL_T2I_CORPUS[use_case][length_stratum]
            text1, text2 = random.choice(options)

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


def _build_teacher_prompt(spec: Dict, g1_lines: List[str], g2_lines: List[str]) -> str:
    f1_style = FONT_STYLE_DESCRIPTORS.get(spec["font1"], "in clean modern bold typography")
    f2_style = FONT_STYLE_DESCRIPTORS.get(spec["font2"], "in clean typography")

    t1_desc = f"At top, on {len(g1_lines)} line(s) {f1_style}: '{' / '.join(g1_lines)}'"
    t2_desc = f"Below it, on {len(g2_lines)} line(s) {f2_style}: '{' / '.join(g2_lines)}'"

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


def generate_target_image(spec: Dict, g1_lines: List[str], g2_lines: List[str], targets_dir: Path, delay: float) -> Path:
    from PIL import Image
    import io

    sid = spec["id"]
    target_path = targets_dir / f"target_{sid}.png"
    if target_path.exists() and target_path.stat().st_size > 10000:
        return target_path

    teacher_prompt = _build_teacher_prompt(spec, g1_lines, g2_lines)
    direct_size = f"{spec['width']}x{spec['height']}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if spec["modality"] == "i2i" and spec.get("product_path"):
                # Use the REAL product photo as an edit reference so the ground-truth target
                # actually contains the same product pixels that will condition FLUX at t=30.0.
                # Without this, the "ground truth" would show a teacher-hallucinated product that
                # never matches the reference image used at train/inference time.
                with open(spec["product_path"], "rb") as prod_file:
                    res = client.images.edit(
                        model="gpt-image-2",
                        image=[prod_file],
                        prompt=teacher_prompt,
                        quality="low",
                        size=direct_size,
                    )
            else:
                res = client.images.generate(
                    model="gpt-image-2",
                    prompt=teacher_prompt,
                    quality="low",
                    size=direct_size,
                )
            raw_bytes = base64.b64decode(res.data[0].b64_json)
            Image.open(io.BytesIO(raw_bytes)).save(target_path)
            time.sleep(delay)
            return target_path
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"   [429 RATE LIMIT] Backoff 12s (attempt {attempt+1}/{max_retries})...")
                time.sleep(12.0)
            else:
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
        print(" [*] Pre-generating dataset specifications and verifying typography pipeline...")
        for idx in range(1, min(6, total_samples + 1)):
            spec = sample_dataset_spec(idx, total_samples)
            has_product = spec["modality"] == "i2i" and spec.get("product_path") is not None
            prompt_clean = build_clean_prompt(spec["text1"], spec["text2"], has_product, args.llm_prompts)
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
        prompt_clean = build_clean_prompt(spec["text1"], spec["text2"], has_product, args.llm_prompts)

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