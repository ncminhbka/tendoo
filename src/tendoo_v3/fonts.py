"""
src/tendoo/core/fonts.py

Central Typography Font Engine for Tendoo Studio:
=================================================
- Quản lý 28 họ font chữ Unicode tiếng Việt chuẩn (Việt hóa 100% dấu thanh) qua 6 phong cách thẩm mỹ.
- Tự động sinh khối CSS `@font-face` nhúng trực tiếp chuỗi Base64 Data URI (cached trong RAM bằng lru_cache).
- Khuyến nghị font thông minh theo ngành nghề, phong cách thị giác và từ khóa nội dung (Auto mode).
- Cam kết độ trung thực hiển thị 100% offline cho Chromium Playwright trên server nội bộ.

(2026-09-13: gộp từ `layouts/font_engine.py` -- trước đó `core/fonts.py` chỉ là 1 vỏ bọc
re-export mỏng. `core/` giờ là tầng nền tảng DUY NHẤT chứa logic thật, không còn
`layouts/` song song nữa.)

TẠI SAO CẦN MODULE NÀY TRONG HỆ THỐNG TENDOO AI:
1. TẠI SAO BẮT BUỘC NHÚNG FONT BASE64 DATA URI THAY VÌ LINK GOOGLE FONTS:
   - Trong môi trường máy chủ nội bộ cô lập (Offline/Air-gapped server) như JupyterLab 2x A30,
     trình duyệt headless Chromium không thể tải font từ CDN bên ngoài (fonts.googleapis.com).
   - Nếu gọi font ngoài, trang web sẽ bị timeout hoặc rơi vào hiện tượng FOUC (Flash of Unstyled Content),
     Chromium chụp ảnh canvas trước khi font kịp render, dẫn đến chữ bị biến dạng thành font Times New Roman
     mặc định hoặc mất dấu tiếng Việt (Tofu Glyphs).
   - Nhúng Base64 Data URI biến file HTML thành một tài liệu tự chứa 100% (self-contained), render tức thì
     với độ sắc nét vector tuyệt đối tại bất kỳ độ phân giải nào (1024x1024, 4K).
   - Thuật toán `lru_cache` lưu trữ các chuỗi Base64 trong RAM, giúp tốc độ sinh HTML đạt dưới 1ms.

2. TẠI SAO LUÔN NHÚNG KÈM BE VIETNAM PRO CHO CÁC KHỐI CHỮ PHỤ:
   - Các font tiêu đề nghệ thuật (Display/Brush/Calligraphy như Pacifico, Blow Brush, Dancing Script)
     rất ấn tượng cho tiêu đề lớn, nhưng hoàn toàn không thích hợp cho chữ nhỏ (12-14px) như hotline,
     địa chỉ cửa hàng, điều kiện áp dụng hay danh sách bước thực hiện.
   - Nhúng kèm Be Vietnam Pro đảm bảo toàn bộ các thành phần UI phụ (Store bar, badge, step badge)
     luôn có font Sans-serif quốc tế chuẩn mực, siêu nét và dễ đọc tuyệt đối.
"""

from __future__ import annotations

import base64
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Xác định đường dẫn thư mục fonts một cách an toàn và linh hoạt
# (không đổi độ sâu thư mục khi gộp từ layouts/ -- core/ nằm cùng cấp `src/tendoo/<pkg>/`)
_CURRENT_DIR = Path(__file__).resolve().parent
# (2026-09-25) Sâu 2 cấp trong repo mới (src/tendoo_v3/, gộp từ tendoo_core 27/09) thay vì 3 cấp như repo
# cũ (src/tendoo/core/) -- sửa để FONTS_DIR vẫn trỏ đúng <repo>/fonts.
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
FONTS_DIR = _PROJECT_ROOT / "fonts"
if not FONTS_DIR.exists():
    FONTS_DIR = Path("fonts")


# Bảng tra cứu 28 họ font (19 gốc + 9 OFL mở rộng 27/09: scripts/fetch_fonts.py) tiếng Việt chuẩn đã kiểm thử chất lượng (QA Verified)

FONT_CATALOG: Dict[str, Dict[str, Any]] = {
    # Nhóm 1: Hiện đại & Chuẩn mực (Modern Clean)
    "bevietnam": {
        "file": "BeVietnamPro-Black.ttf",
        # Họ font nhiều độ đậm (ROADMAP §10.7): trước 26/09 chỉ nhúng Black nhưng khai báo
        # `font-weight: normal` -> CSS xin 600/800 thì Chromium lấy Black rồi TÔ ĐẬM GIẢ thêm.
        # WOFF2 (nén không mất dữ liệu, nhỏ ~3.5x so với TTF) để 6 độ đậm không làm chậm render.
        "weights": {
            400: "BeVietnamPro-Regular.woff2",
            500: "BeVietnamPro-Medium.woff2",
            600: "BeVietnamPro-SemiBold.woff2",
            700: "BeVietnamPro-Bold.woff2",
            800: "BeVietnamPro-ExtraBold.woff2",
            900: "BeVietnamPro-Black.woff2",
        },
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

    "montserrat": {
        "file": "Montserrat.ttf",
        "weights": {"100 900": "Montserrat.woff2"},
        "css_family": "Montserrat",
        "display_name": "Montserrat (Hình học / Nhiều độ đậm)",
        "archetype": "Hiện đại & Chuẩn mực",
        "format": "truetype",
        "fallback": "'Be Vietnam Pro', sans-serif",
        "description": "Sans hình học đủ độ đậm 100-900 (variable), gọn gàng cho tuyển dụng, doanh nghiệp, giáo dục.",
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

    "barlowcond": {
        "file": "BarlowCondensed-Black.ttf",
        "weights": {"100 900": "BarlowCondensed-Black.woff2"},
        "css_family": "Barlow Condensed",
        "display_name": "Barlow Condensed Black (Chữ hẹp / Con số lớn)",
        "archetype": "Mạnh mẽ & Flash Sale",
        "format": "truetype",
        "fallback": "'Anton', 'Oswald', sans-serif",
        "description": "Chữ hẹp đen dày, con số khổng lồ vẫn vừa khung -- flash sale, giá, thể thao.",
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

    "greatvibes": {
        "file": "GreatVibes-Regular.ttf",
        "weights": {"100 900": "GreatVibes-Regular.woff2"},
        "css_family": "Great Vibes",
        "display_name": "Great Vibes (Thư pháp trang trọng / Thiệp cưới)",
        "archetype": "Sang trọng & Thẩm mỹ",
        "format": "truetype",
        "fallback": "'Playfair Display', cursive",
        "description": "Thư pháp nét thanh nét đậm trang trọng cho thiệp mời, tiệc cưới, trang sức, spa cao cấp.",
    },
    "alexbrush": {
        "file": "AlexBrush-Regular.ttf",
        "weights": {"100 900": "AlexBrush-Regular.woff2"},
        "css_family": "Alex Brush",
        "display_name": "Alex Brush (Cọ mềm nghiêng / Mỹ phẩm)",
        "archetype": "Sang trọng & Thẩm mỹ",
        "format": "truetype",
        "fallback": "'Playfair Display', cursive",
        "description": "Cọ mềm nghiêng thanh lịch cho mỹ phẩm, nail, hoa, quà tặng.",
    },
    "cormorant": {
        "file": "Cormorant.ttf",
        "weights": {"100 900": "Cormorant.woff2"},
        "css_family": "Cormorant",
        "display_name": "Cormorant (Serif thanh mảnh / Nhiều độ đậm)",
        "archetype": "Sang trọng & Thẩm mỹ",
        "format": "truetype",
        "fallback": "'Playfair Display', serif",
        "description": "Serif Garamond thanh mảnh đủ độ đậm (variable) cho nước hoa, khách sạn, bất động sản cao cấp.",
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

    "lobster": {
        "file": "Lobster-Regular.ttf",
        "weights": {"100 900": "Lobster-Regular.woff2"},
        "css_family": "Lobster",
        "display_name": "Lobster (Script đậm Retro / Tiệm bánh)",
        "archetype": "F&B, Cafe & Bánh kẹo",
        "format": "truetype",
        "fallback": "'Pacifico', cursive",
        "description": "Script đậm nối nét kiểu biển hiệu retro cho tiệm bánh, burger, quán ăn gia đình.",
    },
    "charm": {
        "file": "Charm-Bold.ttf",
        "weights": {"100 900": "Charm-Bold.woff2"},
        "css_family": "Charm",
        "display_name": "Charm (Viết tay bút mực / Cafe thủ công)",
        "archetype": "F&B, Cafe & Bánh kẹo",
        "format": "truetype",
        "fallback": "'Dancing Script', cursive",
        "description": "Viết tay bút mực mềm mại cho cafe thủ công, trà, đồ handmade, Trung thu.",
    },
    "sriracha": {
        "file": "Sriracha-Regular.ttf",
        "weights": {"100 900": "Sriracha-Regular.woff2"},
        "css_family": "Sriracha",
        "display_name": "Sriracha (Viết tay thân thiện / Trà sữa)",
        "archetype": "F&B, Cafe & Bánh kẹo",
        "format": "truetype",
        "fallback": "'Plus Jakarta Sans', cursive",
        "description": "Viết tay tròn trịa thân thiện cho trà sữa, đồ uống giới trẻ, lớp học, mẹ & bé.",
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
    "pattaya": {
        "file": "Pattaya-Regular.ttf",
        "weights": {"100 900": "Pattaya-Regular.woff2"},
        "css_family": "Pattaya",
        "display_name": "Pattaya (Cọ đậm / Tết & Sale lễ hội)",
        "archetype": "Lễ hội & Tết",
        "format": "truetype",
        "fallback": "'Lobster', cursive",
        "description": "Nét cọ đậm rộn ràng cho Tết, lễ hội, khai trương, sale mùa lễ.",
    },
}

# Alias dictionary for convenience
# Font ĐỌC ĐƯỢC ở cỡ nhỏ, dòng dài (sans/serif văn bản). Còn lại là font TRƯNG BÀY (script, cọ, graffiti,
# condensed/ultra-black): chỉ cho tiêu đề -- quy tắc ghép font của designer "display cho headline, sans cho
# thân chữ". Đo thật GĐ 3R: Pacifico cho bảng giá / các bước chăm sóc da -> danh sách khó đọc.
BODY_SAFE_FONTS = frozenset({"bevietnam", "harabaras", "playfair", "oswald", "montserrat"})
# LINE-HEIGHT TỐI THIỂU để dấu tiếng Việt chồng (Ấ Ầ Ự...) dòng dưới KHÔNG chạm nét dòng trên -- ĐO bằng điểm ảnh
# Chromium, chữ 900 in hoa, 3 câu mẫu (tests/test_diacritics.py, 27/09). Font không có ở đây: 1.05 là đủ.
MIN_STACK_LINE_HEIGHT: Dict[str, float] = {"anton": 1.2, "gretoon": 1.3, "oswald": 1.1, "pacifico": 1.4,
                                           "greatvibes": 1.25, "charm": 1.15}
DEFAULT_STACK_LINE_HEIGHT = 1.05

# Font VIẾT TAY / cọ mềm (lockup script_over_caps dùng làm dòng viết tay; đồng thời không dùng làm dòng in hoa).
SCRIPT_FONTS = frozenset({"dancing", "clementine", "pacifico", "cookies", "holidays",
                          "greatvibes", "alexbrush", "lobster", "charm", "sriracha", "pattaya"})
DEFAULT_SCRIPT_FONT = "dancing"


SCRIPT_HERO_MAX_WORDS = 5


def script_unfit(font_key: str, headline_text: str) -> Optional[str]:
    """Lý do font viết tay KHÔNG hợp đoạn chữ hiển thị bằng font tiêu đề (None = hợp). Thực hành designer:
    (1) script chỉ cho 1 cụm ngắn <= 5 từ (27/09, thông báo nghỉ Tết 8 từ SVN-Holidays khó đọc);
    (2) KHÔNG BAO GIỜ viết hoa toàn bộ chữ viết tay -- chữ hoa script nối nét rối, mất nhịp (27/09, pack cafe
    Lobster "CÀ PHÊ PHIN": đo bằng mắt kém hẳn Playfair + dòng viết tay thường)."""
    if font_key not in SCRIPT_FONTS:
        return None
    words = headline_text.split()
    if len(words) > SCRIPT_HERO_MAX_WORDS:
        return f"{len(words)} từ (> {SCRIPT_HERO_MAX_WORDS})"
    if is_all_caps(headline_text):
        return "viết hoa toàn bộ"
    return None


def is_all_caps(text: str) -> bool:
    """Đoạn chữ viết hoa toàn bộ (>= 80% chữ cái hoa, >= 4 chữ cái -- '50%' hay 'Tết' không tính)."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) >= 0.8 * len(letters)


def script_font(headline_key: str, preferred: Optional[str] = None) -> Tuple[str, str]:
    """(font_face_css cần nhúng THÊM, chuỗi font-family) cho dòng viết tay của lockup: font tiêu đề nếu nó
    đã là viết tay (không nhúng thêm), ngược lại font viết tay của style pack (`preferred`, vd luxury -> Great Vibes)
    hoặc Dancing Script (hỗ trợ đủ dấu tiếng Việt)."""
    if headline_key in SCRIPT_FONTS:
        meta = FONT_CATALOG[headline_key]
        return "", f"'{meta['css_family']}', {meta['fallback']}"
    meta = FONT_CATALOG[preferred if preferred in SCRIPT_FONTS else DEFAULT_SCRIPT_FONT]
    return "\n\n".join(_font_faces(meta)), f"'{meta['css_family']}', {meta['fallback']}"

FONT_ALIASES: Dict[str, str] = {
    "be_vietnam_pro": "bevietnam",
    "bevietnampro": "bevietnam",
    "be_vietnam": "bevietnam",
    "inter": "bevietnam",
    "roboto": "bevietnam",
    "sans": "bevietnam",
    "modern": "bevietnam",
    "clean": "bevietnam",
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
        # Silent before 2026-09-12: returned "" with zero warning -- on an air-gapped
        # server (this module's own stated deployment target) this can mean the poster
        # silently falls back to whatever generic font Chromium has installed, which may
        # not cover Vietnamese diacritics, with nothing in the logs to explain why.
        logger.warning(f"[fonts] Font file not found: {file_path_str} -- this font will not be embedded, falling back to its CSS fallback stack (may not render Vietnamese diacritics correctly).")
        return ""
    data = p.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _font_faces(meta: Dict[str, Any]) -> List[str]:
    """Các khối @font-face (Base64) của 1 họ font. Họ có "weights" -> 1 khối/độ đậm THẬT để
    Chromium chọn đúng file thay vì tô đậm giả; họ 1 file -> 1 khối `font-weight: normal`."""
    # Họ 1 file: khai báo PHẠM VI 100 900 thay vì `normal` (400) -- CSS xin 900 cho hero thì Chromium
    # KHÔNG tô đậm giả lên font vốn đã đậm (Anton, Gotham Ultra...); font variable (Oswald, Playfair,
    # Dancing Script) còn được độ đậm THẬT theo trục wght (GĐ 7b, R3).
    files = meta.get("weights") or {"100 900": meta["file"]}
    family, faces = meta["css_family"], []
    for weight, fname in files.items():
        fmt = "woff2" if fname.endswith(".woff2") else meta["format"]
        b64 = _read_and_encode_font(str(FONTS_DIR / fname))
        if b64:
            faces.append(f"""@font-face {{
      font-family: '{family}';
      src: url('data:font/{fmt};charset=utf-8;base64,{b64}') format('{fmt}');
      font-weight: {weight};
      font-style: normal;
      font-display: swap;
      unicode-range: U+0-9F, U+A1-10FFFF;
    }}""")
    # unicode-range bỏ U+00A0 (khoảng trắng không ngắt -- hero_markup.bind_nonbreaking chèn giữa số và đơn vị):
    # SVN-Days VẼ một chữ ở vị trí NBSP -> "12 tuần" hiện "12Atuần" (27/09, poster gym GPT thật). NBSP lấy từ font
    # dự phòng = khoảng trắng thật, vẫn không ngắt dòng.
    return faces


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
    # "moonbeam" is a real OMNI_STYLES lighting/mood style (unrelated to Tết/lunar
    # festivities) that happens to contain the substring "moon" -- excluded explicitly so
    # it doesn't collide with the free-text "moon" keyword below (2026-09-12 fix).
    if hint != "moonbeam" and any(k in hint for k in ["festive", "moon", "tet", "xuan", "party"]):
        return "holidays"
    if any(k in hint for k in ["asphalt", "rock", "stone", "dark"]):
        return "anton"

    # 2. Check text content keywords
    if any(k in txt for k in ["cafe", "cà phê", "trà", "trà đào", "trà sữa", "nước ép", "sinh tố", "bistro", "ẩm thực", "summer", "mùa hè"]):
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

    css_family = meta["css_family"]
    fallback = meta["fallback"]
    # headline_font_css vẫn nêu font mong muốn + chuỗi dự phòng kể cả khi file thiếu
    # (`_read_and_encode_font` đã log cảnh báo) -- khi đó Chromium dùng font hệ thống.
    headline_font_css = f"'{css_family}', {fallback}"

    faces = _font_faces(meta)
    # Luôn nhúng kèm Be Vietnam Pro Base64 để phục vụ các thành phần UI
    # (badge, body, store info, step list) mà không cần internet/Google Fonts.
    if k != "bevietnam":
        bv_faces = _font_faces(FONT_CATALOG["bevietnam"])
        if not bv_faces:
            logger.warning("[fonts] Mandatory Be Vietnam Pro embed unavailable -- badges/store-info/step-list text will fall back to whatever system font Chromium finds, which may not render Vietnamese diacritics correctly.")
        faces += bv_faces

    font_face_css = "\n\n".join(faces)
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


__all__ = [
    "BODY_SAFE_FONTS",
    "DEFAULT_STACK_LINE_HEIGHT",
    "MIN_STACK_LINE_HEIGHT",
    "DEFAULT_SCRIPT_FONT",
    "SCRIPT_FONTS",
    "script_font",
    "script_unfit",
    "is_all_caps",
    "SCRIPT_HERO_MAX_WORDS",
    "FONT_ALIASES",
    "FONT_CATALOG",
    "FONTS_DIR",
    "list_font_options",
    "recommend_font",
    "resolve_font",
]
