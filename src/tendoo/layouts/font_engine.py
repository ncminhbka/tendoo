"""
src/tendoo/layouts/font_engine.py

Central Font Engine for Tendoo Studio:
- Registers 19 verified Vietnamese typography fonts across 6 aesthetic archetypes.
- Generates zero-dependency @font-face CSS using Base64 Data URIs (cached in-memory).
- Provides category-aware intelligent font recommendation (Auto mode).
- Guarantees 100% offline rendering fidelity for Playwright on remote servers.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Locate fonts directory robustly
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent.parent
FONTS_DIR = _PROJECT_ROOT / "fonts"
if not FONTS_DIR.exists():
    FONTS_DIR = Path("fonts")


# Registry of 19 QA-verified Vietnamese Unicode fonts
FONT_CATALOG: Dict[str, Dict[str, Any]] = {
    # Nhóm 1: Hiện đại & Chuẩn mực (Modern Clean)
    "bevietnam": {
        "file": "BeVietnamPro-Black.ttf",
        "css_family": "Be Vietnam Pro",
        "display_name": "Be Vietnam Pro (Hiện đại / Siêu nét)",
        "archetype": "Hiện đại & Chuẩn mực",
        "format": "truetype",
        "fallback": "'Montserrat', 'Plus Jakarta Sans', sans-serif",
        "description": "Sans-serif quốc tế chuẩn mực, dễ đọc tuyệt đối trên mọi độ phân giải.",
    },
    "gotham": {
        "file": "SVN-Gotham Ultra.otf",
        "css_family": "SVN-Gotham Ultra",
        "display_name": "SVN-Gotham Ultra (Quyền lực / Doanh nghiệp)",
        "archetype": "Hiện đại & Chuẩn mực",
        "format": "opentype",
        "fallback": "'Montserrat', 'Be Vietnam Pro', sans-serif",
        "description": "Hình học uy quyền, lý tưởng cho tập đoàn, viễn thông, tài chính B2B.",
    },
    "harabaras": {
        "file": "SVN-Harabaras.ttf",
        "css_family": "SVN-Harabaras",
        "display_name": "SVN-Harabaras (Geometric / Thân thiện)",
        "archetype": "Hiện đại & Chuẩn mực",
        "format": "truetype",
        "fallback": "'Plus Jakarta Sans', sans-serif",
        "description": "Bo tròn góc nhẹ, năng động cho ứng dụng, thiết bị thông minh, startup.",
    },

    # Nhóm 2: Mạnh mẽ & Giảm giá sốc (Impact & Sport)
    "anton": {
        "file": "Anton-Regular.ttf",
        "css_family": "Anton",
        "display_name": "Anton (Cô đọng / Giảm giá sốc)",
        "archetype": "Mạnh mẽ & Flash Sale",
        "format": "truetype",
        "fallback": "'Oswald', 'Montserrat', sans-serif",
        "description": "Chữ cao dóng dày dặn, giật gân, tác động thị giác cực mạnh cho flash sale.",
    },
    "days": {
        "file": "SVN-Days.otf",
        "css_family": "SVN-Days",
        "display_name": "SVN-Days (Tương lai / Thể thao / Tech)",
        "archetype": "Mạnh mẽ & Flash Sale",
        "format": "opentype",
        "fallback": "'Anton', 'Montserrat', sans-serif",
        "description": "Hình khối tương lai mạnh mẽ, hoàn hảo cho đồ thể thao, gaming, xe cộ, robot.",
    },
    "hemihead": {
        "file": "SVN-Hemi Head.ttf",
        "css_family": "SVN-Hemi Head",
        "display_name": "SVN-Hemi Head (Tốc độ / Đua xe / Đậm nét)",
        "archetype": "Mạnh mẽ & Flash Sale",
        "format": "truetype",
        "fallback": "'Anton', sans-serif",
        "description": "Nét chữ nghiêng tốc độ, cơ khí mạnh mẽ cho ô tô, phụ kiện xe, thể thao.",
    },
    "lolapeluza": {
        "file": "SVN-Lolapeluza Black.ttf",
        "css_family": "SVN-Lolapeluza Black",
        "display_name": "SVN-Lolapeluza (Siêu dày / Đại tiệc Sale)",
        "archetype": "Mạnh mẽ & Flash Sale",
        "format": "truetype",
        "fallback": "'Anton', 'Montserrat', sans-serif",
        "description": "Chữ cực dày khối, tạo cảm giác đại tiệc xả kho hoành tráng, đông vui.",
    },
    "oswald": {
        "file": "Oswald.ttf",
        "css_family": "Oswald",
        "display_name": "Oswald (Thanh mảnh / Gothic / Specs)",
        "archetype": "Mạnh mẽ & Flash Sale",
        "format": "truetype",
        "fallback": "'Anton', 'Be Vietnam Pro', sans-serif",
        "description": "Gothic cao ráo, chuyên nghiệp cho thông số kỹ thuật, thời trang nam.",
    },

    # Nhóm 3: Sang trọng & Thẩm mỹ (Luxury & Editorial)
    "playfair": {
        "file": "PlayfairDisplay.ttf",
        "css_family": "Playfair Display",
        "display_name": "Playfair Display (Quý phái / Editorial Serif)",
        "archetype": "Sang trọng & Thẩm mỹ",
        "format": "truetype",
        "fallback": "'Times New Roman', serif",
        "description": "Serif tương phản cao kiểu Vogue/Harper's Bazaar cho trang sức, nước hoa, bất động sản.",
    },
    "dancing": {
        "file": "DancingScript.ttf",
        "css_family": "Dancing Script",
        "display_name": "Dancing Script (Cursive / Spa & Beauty)",
        "archetype": "Sang trọng & Thẩm mỹ",
        "format": "truetype",
        "fallback": "'Playfair Display', cursive",
        "description": "Chữ viết tay uyển chuyển, nhẹ nhàng cho spa, thẩm mỹ viện, mỹ phẩm organic.",
    },
    "clementine": {
        "file": "SVN-Clementine.ttf",
        "css_family": "SVN-Clementine",
        "display_name": "SVN-Clementine (Calligraphy / Tiệc cưới)",
        "archetype": "Sang trọng & Thẩm mỹ",
        "format": "truetype",
        "fallback": "'Playfair Display', cursive",
        "description": "Thư pháp hoàng gia thanh tao cho thiệp mừng, quà tặng cao cấp, tiệc cưới.",
    },

    # Nhóm 4: Ẩm thực & Bánh kẹo (F&B & Playful)
    "pacifico": {
        "file": "Pacifico-Regular.ttf",
        "css_family": "Pacifico",
        "display_name": "Pacifico (Cọ vẽ Retro / Cafe & Đồ uống)",
        "archetype": "F&B, Cafe & Bánh kẹo",
        "format": "truetype",
        "fallback": "'Plus Jakarta Sans', cursive",
        "description": "Cọ vẽ mùa hè phóng khoáng, số 1 cho quán cafe, trà sữa, ẩm thực đường phố.",
    },
    "cookies": {
        "file": "SVN-Cookies.ttf",
        "css_family": "SVN-Cookies",
        "display_name": "SVN-Cookies (Bánh ngọt / Trà sữa / Kem)",
        "archetype": "F&B, Cafe & Bánh kẹo",
        "format": "truetype",
        "fallback": "'Plus Jakarta Sans', cursive",
        "description": "Chữ tròn xoe như bánh quy, cực kỳ đáng yêu cho tiệm bánh, kẹo ngọt, mẹ & bé.",
    },
    "gretoon": {
        "file": "SVN-Gretoon.ttf",
        "css_family": "SVN-Gretoon",
        "display_name": "SVN-Gretoon (Pop-Art 3D / Đồ ăn vặt)",
        "archetype": "F&B, Cafe & Bánh kẹo",
        "format": "truetype",
        "fallback": "'Montserrat', sans-serif",
        "description": "Pop-art truyện tranh 3D vui nhộn cho snack, rạp phim, đồ ăn vặt thanh thiếu niên.",
    },
    "grocery": {
        "file": "SVN-Grocery Rounded.ttf",
        "css_family": "SVN-Grocery Rounded",
        "display_name": "SVN-Grocery (Bảng phấn Vintage / Thực phẩm sạch)",
        "archetype": "F&B, Cafe & Bánh kẹo",
        "format": "truetype",
        "fallback": "'Plus Jakarta Sans', sans-serif",
        "description": "Chữ viết bảng phấn mộc mạc cho siêu thị hữu cơ, nông trại sạch, cafe vintage.",
    },

    # Nhóm 5: Đường phố & Thể thao (Street Art & Action)
    "blowbrush": {
        "file": "SVN-Blow Brush.ttf",
        "css_family": "SVN-Blow Brush",
        "display_name": "SVN-Blow Brush (Marker cọ khô / Streetwear)",
        "archetype": "Đường phố & Thể thao",
        "format": "truetype",
        "fallback": "'Impact', 'Montserrat', sans-serif",
        "description": "Nét marker khô đường phố cá tính, hợp cho streetwear, sneaker, hiphop.",
    },
    "sedgwick": {
        "file": "SedgwickAveDisplay-Regular.ttf",
        "css_family": "Sedgwick Ave Display",
        "display_name": "Sedgwick Ave (Graffiti / Văn hóa đường phố)",
        "archetype": "Đường phố & Thể thao",
        "format": "truetype",
        "fallback": "'Montserrat', cursive",
        "description": "Graffiti tự do đậm chất nghệ thuật đô thị, trượt ván, âm nhạc underground.",
    },
    "guerrilla": {
        "file": "SVN-Guerrilla.ttf",
        "css_family": "SVN-Guerrilla",
        "display_name": "SVN-Guerrilla (Stencil / Quân đội / Underground)",
        "archetype": "Đường phố & Thể thao",
        "format": "truetype",
        "fallback": "'Impact', sans-serif",
        "description": "Font Stencil dập khuôn quân đội, góc cạnh, mạnh mẽ cho dã ngoại, phượt.",
    },

    # Nhóm 6: Lễ hội & Sự kiện (Festive & Celebration)
    "holidays": {
        "file": "SVN-Holidays.ttf",
        "css_family": "SVN-Holidays",
        "display_name": "SVN-Holidays (Lễ hội / Tết / Party)",
        "archetype": "Lễ hội & Tết",
        "format": "truetype",
        "fallback": "'Playfair Display', cursive",
        "description": "Chữ rộn ràng phong vị Tết cổ truyền, tiệc Giáng sinh, Gala tri ân, đón lộc xuân.",
    },
}

# Alias dictionary for convenience
FONT_ALIASES: Dict[str, str] = {
    "brush": "blowbrush",
    "graffiti": "sedgwick",
    "street": "blowbrush",
    "food": "pacifico",
    "cafe": "pacifico",
    "bakery": "cookies",
    "luxury": "playfair",
    "serif": "playfair",
    "sale": "anton",
    "tech": "days",
    "sport": "days",
    "festive": "holidays",
    "tet": "holidays",
}


@lru_cache(maxsize=32)
def _read_and_encode_font(file_path_str: str) -> str:
    """Reads a font file and encodes it to Base64 (cached in RAM)."""
    p = Path(file_path_str)
    if not p.exists():
        return ""
    data = p.read_bytes()
    return base64.b64encode(data).decode("ascii")


def recommend_font(category: str, style_hint: str = "", text_content: str = "") -> str:
    """
    Intelligently recommends the best typography font key based on:
    - Commercial category (promo, opening, feedback, recruitment, product_intro, guide)
    - Visual style hint (luxury, sport, daylight, tech, festive...)
    - Optional keyword scan in headline/description text.
    """
    cat = (category or "").lower().strip()
    hint = (style_hint or "").lower().strip()
    txt = (text_content or "").lower().strip()

    # 1. Check style hint priority
    if any(k in hint for k in ["sport", "speed", "cyber", "racing", "gaming"]):
        return "days"
    if any(k in hint for k in ["luxury", "gold", "champagne", "silk"]):
        return "playfair"
    if any(k in hint for k in ["festive", "moon", "tet", "xuan", "party"]):
        return "holidays"
    if any(k in hint for k in ["asphalt", "rock", "stone", "dark"]):
        return "anton"

    # 2. Check text content keywords
    if any(k in txt for k in ["cafe", "cà phê", "trà sữa", "nước ép", "sinh tố", "bistro", "ẩm thực", "summer", "mùa hè"]):
        return "pacifico"
    if any(k in txt for k in ["bánh", "kem", "bakery", "sweet", "ngọt", "trẻ em", "đồ chơi"]):
        return "cookies"
    if any(k in txt for k in ["spa", "thẩm mỹ", "massage", "nước hoa", "son", "mỹ phẩm"]):
        return "dancing" if "spa" in txt or "organic" in txt else "playfair"
    if any(k in txt for k in ["tết", "giáng sinh", "chúc mừng", "tri ân", "gala"]):
        return "holidays"
    if any(k in txt for k in ["streetwear", "sneaker", "hiphop", "skateboard", "giày"]):
        return "blowbrush"

    # 3. Category fallbacks
    if cat == "promo":
        return "anton"
    elif cat == "opening":
        return "pacifico"
    elif cat == "feedback":
        return "playfair"
    elif cat == "recruitment":
        return "gotham"
    elif cat == "product_intro":
        return "days" if "sport" in hint or "tech" in hint else "bevietnam"
    elif cat == "guide":
        return "bevietnam"

    return "bevietnam"


def resolve_font(
    font_key: Optional[str] = None,
    category: str = "promo",
    style_hint: str = "",
    text_content: str = "",
) -> Tuple[str, str, str]:
    """
    Resolves the font key and generates:
    1. canonical font_key (e.g. 'anton')
    2. font_face_css (@font-face block with Base64 data URI)
    3. headline_font_css (font-family stack for CSS)
    """
    k = (font_key or "auto").lower().strip()
    if k in FONT_ALIASES:
        k = FONT_ALIASES[k]

    if k == "auto" or k not in FONT_CATALOG:
        k = recommend_font(category=category, style_hint=style_hint, text_content=text_content)

    meta = FONT_CATALOG.get(k, FONT_CATALOG["bevietnam"])
    file_path = FONTS_DIR / meta["file"]

    font_b64 = _read_and_encode_font(str(file_path))
    fmt = meta["format"]
    css_family = meta["css_family"]
    fallback = meta["fallback"]

    if font_b64:
        font_face_css = f"""
    @font-face {{
      font-family: '{css_family}';
      src: url('data:font/{fmt};charset=utf-8;base64,{font_b64}') format('{fmt}');
      font-weight: normal;
      font-style: normal;
      font-display: swap;
    }}
        """.strip()
        headline_font_css = f"'{css_family}', {fallback}"
    else:
        # Graceful fallback if file is missing
        font_face_css = ""
        headline_font_css = f"'{css_family}', {fallback}"

    return k, font_face_css, headline_font_css


def list_font_options() -> List[Dict[str, Any]]:
    """Returns full catalog structured for UI dropdown consumption."""
    archetypes: Dict[str, List[Dict[str, Any]]] = {}
    for key, meta in FONT_CATALOG.items():
        arch = meta["archetype"]
        if arch not in archetypes:
            archetypes[arch] = []
        archetypes[arch].append({
            "key": key,
            "display_name": meta["display_name"],
            "css_family": meta["css_family"],
            "description": meta["description"],
        })

    result = []
    for arch_name, font_list in archetypes.items():
        result.append({
            "group_name": arch_name,
            "fonts": font_list,
        })
    return result
