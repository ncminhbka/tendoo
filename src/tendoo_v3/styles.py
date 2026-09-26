"""
src/tendoo_v3/styles.py

Thư viện hiệu ứng CSS & Macro đồ họa cao cấp cho Tendoo v3:
- Hiệu ứng chữ: 3D Gold, Neon Glow, Liquid Chrome, Minimal Shadow, Plain Elegant.
- Kính mờ cao cấp (Glassmorphism): backdrop-filter blur, viền phát sáng đa tầng, đổ bóng ma trận.
- Tự động nhúng font tiếng Việt Base64 (100% offline).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tendoo_core.colors import ensure_contrast, get_contrasting_text_color
from tendoo_core.fonts import resolve_font

# Danh mục đóng hiệu ứng chữ -- đúng các nhánh get_effect_css() bên dưới xử lý. Tên lạ rơi
# về "plain_elegant" (nhánh else). Dùng chung với validators.py (Cổng 2).
TEXT_EFFECTS = (
    "3d_gold", "neon", "neon_bloom", "chrome", "fire", "shadow",
    "embossed", "chromatic", "hologram", "plain_elegant",
)
# Alias thường dùng từ prompt/LLM.
TEXT_EFFECT_ALIASES = {
    "gold_metallic": "3d_gold",
    "gold": "3d_gold",
    "neon_glow": "neon_bloom",
    "glow": "neon_bloom",
    "bold_clean": "plain_elegant",
    "clean": "plain_elegant",
}
# Tông nền có xử lý riêng ở get_adaptive_palette() / velocity_blending.py / comment của nó.
# Tên lạ: palette coi là nền tối, mock backdrop dùng màu dark_luxury.
BACKGROUND_TONES = ("dark_luxury", "light_clean", "warm_rustic", "pastel", "vibrant", "cyber_neon", "cinema_red")


def get_effect_css(
    effect_name: str,
    theme_color: str = "#D4AF37",
    is_dark: bool = True,
    text_color: Optional[str] = None,
) -> str:
    """Trả về các quy tắc CSS cho hiệu ứng chữ đã chọn, tự động thích ứng theo nền tối/sáng.

    `text_color` (tuỳ chọn): màu chữ THẬT đã tính zone-adaptive (có thể đã nhuộm nhẹ
    theo hue ảnh, xem `renderer.py::_tinted_hex`) -- CHỈ áp dụng cho 4 hiệu ứng "màu
    phẳng + đổ bóng" (shadow/embossed/chromatic/plain_elegant), nơi `color:` vốn là
    lựa chọn thẩm mỹ tự do (không gánh ý nghĩa thị giác riêng). KHÔNG áp dụng cho
    neon/neon_bloom (lõi trắng rực là chủ đích thiết kế của hiệu ứng glow, nhuộm màu sẽ
    làm mờ đục lõi) hay 3d_gold/chrome/fire/hologram (gradient multi-stop đã là bản sắc
    màu riêng, không có khái niệm "màu chữ" đơn lẻ để thay). Mặc định `None` giữ
    NGUYÊN hành vi cũ (trắng/navy thuần) khi không truyền vào."""
    eff = (effect_name or "plain_elegant").lower().strip()

    eff = TEXT_EFFECT_ALIASES.get(eff, eff)

    if is_dark:
        # ==========================================
        # HIỆU ỨNG TRÊN NỀN TỐI (DARK BACKGROUND)
        # ==========================================
        if eff == "3d_gold":
            return """
            background: linear-gradient(135deg, #FFF59D 0%, #FFD700 25%, #FFA000 60%, #FFE082 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: none !important;
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.98)) drop-shadow(0 4px 14px rgba(0, 0, 0, 0.85)) drop-shadow(0 0 1px #000000);
            """
        elif eff == "neon":
            c = theme_color if theme_color else "#00F0FF"
            return f"""
            color: #FFFFFF;
            text-shadow:
              0 0 5px #FFFFFF,
              0 0 10px {c},
              0 0 20px {c},
              0 0 40px {c},
              0 0 60px {c};
            """
        elif eff == "chrome":
            return """
            background: linear-gradient(180deg, #FFFFFF 0%, #D8E0EC 35%, #8892A2 50%, #E8F0FE 52%, #B0BAC8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: none !important;
            filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.95)) drop-shadow(0 0 15px rgba(200, 220, 255, 0.5));
            """
        elif eff == "fire":
            return """
            background: linear-gradient(180deg, #FFFF99 0%, #FFCC00 25%, #FF6600 65%, #CC0000 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: none !important;
            filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.95)) drop-shadow(0 0 25px rgba(255, 102, 0, 0.7));
            """
        elif eff == "shadow":
            return f"""
            color: {text_color or "#FFFFFF"};
            text-shadow:
              0 2px 4px rgba(0, 0, 0, 0.95),
              0 6px 16px rgba(0, 0, 0, 0.8),
              0 12px 32px rgba(0, 0, 0, 0.6);
            """
        elif eff == "embossed":
            return f"""
            color: {text_color or "#E2E8F0"};
            text-shadow:
              1px 1px 0 rgba(255, 255, 255, 0.45),
              -1px -1px 0 rgba(0, 0, 0, 0.65),
              0 2px 8px rgba(0, 0, 0, 0.75);
            """
        elif eff == "chromatic":
            return f"""
            color: {text_color or "#FFFFFF"};
            text-shadow:
              -2.5px 0 0 rgba(255, 0, 80, 0.85),
              2.5px 0 0 rgba(0, 240, 255, 0.85),
              0 2px 10px rgba(0, 0, 0, 0.8);
            """
        elif eff == "neon_bloom":
            c = theme_color if theme_color else "#00F0FF"
            return f"""
            color: #FFFFFF;
            text-shadow:
              0 0 4px #FFFFFF,
              0 0 12px {c},
              0 0 24px {c},
              0 0 48px {c},
              0 0 80px {c}99;
            filter: drop-shadow(0 0 15px {c}66);
            """
        elif eff == "hologram":
            return """
            background: linear-gradient(180deg, #E0F2FE 0%, #38BDF8 40%, #0284C7 70%, #0369A1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.8)) drop-shadow(0 0 20px rgba(14, 165, 233, 0.4));
            """
        else:  # plain_elegant
            return f"""
            color: {text_color or "#FFFFFF"};
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.75), 0 4px 20px rgba(0, 0, 0, 0.5);
            """
    else:
        # ==========================================
        # HIỆU ỨNG TRÊN NỀN SÁNG (LIGHT BACKGROUND)
        # ==========================================
        if eff == "3d_gold":
            return """
            background: linear-gradient(135deg, #B8860B 0%, #D4AF37 35%, #996515 70%, #B8860B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: none !important;
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.65)) drop-shadow(0 0 1px rgba(0, 0, 0, 0.85));
            """
        elif eff in ("neon", "neon_bloom"):
            c = theme_color if theme_color else "#0284C7"
            return f"""
            color: #0F172A;
            text-shadow:
              0 0 4px #FFFFFF,
              0 0 12px {c},
              0 0 24px {c}88;
            filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.35));
            """
        elif eff == "chrome":
            return """
            background: linear-gradient(180deg, #1E293B 0%, #475569 35%, #0F172A 52%, #334155 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: none !important;
            filter: drop-shadow(0 1px 3px rgba(255, 255, 255, 0.9)) drop-shadow(0 2px 8px rgba(0, 0, 0, 0.25));
            """
        elif eff == "fire":
            return """
            background: linear-gradient(180deg, #EA580C 0%, #C2410C 40%, #991B1B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.35));
            """
        elif eff == "shadow":
            return f"""
            color: {text_color or "#0F172A"};
            text-shadow:
              0 1px 2px rgba(255, 255, 255, 0.95),
              0 4px 14px rgba(0, 0, 0, 0.22);
            """
        elif eff == "embossed":
            return f"""
            color: {text_color or "#1E293B"};
            text-shadow:
              1px 1px 0 rgba(255, 255, 255, 0.90),
              -1px -1px 0 rgba(0, 0, 0, 0.35),
              0 2px 8px rgba(0, 0, 0, 0.15);
            """
        elif eff == "chromatic":
            return f"""
            color: {text_color or "#0F172A"};
            text-shadow:
              -2px 0 0 rgba(225, 29, 72, 0.8),
              2px 0 0 rgba(14, 165, 233, 0.8),
              0 1px 3px rgba(255, 255, 255, 0.9);
            """
        elif eff == "hologram":
            return """
            background: linear-gradient(180deg, #0369A1 0%, #0284C7 40%, #075985 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 4px rgba(56, 189, 248, 0.5));
            """
        else:  # plain_elegant on light background
            return f"""
            color: {text_color or "#0F172A"};
            text-shadow: 0 1px 3px rgba(255, 255, 255, 0.95), 0 2px 8px rgba(0, 0, 0, 0.12);
            """


def get_adaptive_palette(background_tone: str, theme_color: str = "#FFB300") -> Dict[str, str]:
    """Tính toán bảng màu thích ứng cho Transparent Glass Overlay, loại bỏ hoàn toàn các vệt đen."""
    tone = (background_tone or "dark_luxury").lower().strip()
    tc = theme_color if theme_color else "#FFB300"
    is_dark = tone not in ("pastel", "light_clean")

    if not is_dark:
        card_bg_solid = "#FFFFFF"
        on_card_accent = ensure_contrast(tc, card_bg_solid, min_ratio=4.5)
        on_card_title = ensure_contrast(f"{tc}", card_bg_solid, min_ratio=4.5)
        on_canvas_accent = ensure_contrast(tc, "#FFFFFF", min_ratio=4.5)
        badge_text = get_contrasting_text_color(tc)

        return {
            "is_dark": False,
            "text_primary": "#0F172A",
            "text_secondary": "rgba(15, 23, 42, 0.82)",
            "glass_bg": "rgba(255, 255, 255, 0.70)",
            "glass_border": "rgba(255, 255, 255, 0.90)",
            "glass_shadow": "0 16px 36px rgba(0, 0, 0, 0.10)",
            "text_shadow": "0 1px 2px rgba(255, 255, 255, 0.8)",
            "pill_bg": "rgba(255, 255, 255, 0.75)",
            "pill_border": f"{tc}55",
            "pill_text": "#0F172A",
            "cta_bg": tc,
            "cta_text": badge_text,
            "badge_bg": tc,
            "badge_text": badge_text,
            "on_card_accent": on_card_accent,
            "on_card_title": on_card_title,
            "on_canvas_accent": on_canvas_accent,
            "gradient_overlay": "linear-gradient(to bottom, rgba(255, 255, 255, 0.85) 0%, rgba(255, 255, 255, 0.40) 70%, rgba(255, 255, 255, 0) 100%)",
        }
    elif tone == "cyber_neon":
        card_bg_solid = "#0A0F1C"
        on_card_accent = ensure_contrast(tc, card_bg_solid, min_ratio=4.5)
        on_card_title = ensure_contrast(f"{tc}", card_bg_solid, min_ratio=4.5)
        on_canvas_accent = ensure_contrast(tc, "#000000", min_ratio=4.5)
        badge_text = get_contrasting_text_color(tc)

        return {
            "is_dark": True,
            "text_primary": "#FFFFFF",
            "text_secondary": "rgba(255, 255, 255, 0.88)",
            "glass_bg": "rgba(10, 15, 28, 0.48)",
            "glass_border": f"{tc}66",
            "glass_shadow": f"0 16px 40px rgba(0, 0, 0, 0.7), 0 0 24px {tc}33",
            "text_shadow": "0 2px 10px rgba(0, 0, 0, 0.9)",
            "pill_bg": "rgba(255, 255, 255, 0.08)",
            "pill_border": f"{tc}88",
            "pill_text": "#FFFFFF",
            "cta_bg": tc,
            "cta_text": badge_text,
            "badge_bg": tc,
            "badge_text": badge_text,
            "on_card_accent": on_card_accent,
            "on_card_title": on_card_title,
            "on_canvas_accent": on_canvas_accent,
            "gradient_overlay": "linear-gradient(to bottom, rgba(10, 15, 28, 0.75) 0%, rgba(10, 15, 28, 0.35) 70%, rgba(10, 15, 28, 0) 100%)",
        }
    else:  # dark_luxury, warm_rustic, cinema_red
        card_bg_solid = "#0C101A"
        on_card_accent = ensure_contrast(tc, card_bg_solid, min_ratio=4.5)
        on_card_title = ensure_contrast(f"{tc}", card_bg_solid, min_ratio=4.5)
        on_canvas_accent = ensure_contrast(tc, "#000000", min_ratio=4.5)
        badge_text = get_contrasting_text_color(tc)

        return {
            "is_dark": True,
            "text_primary": "#FFFFFF",
            "text_secondary": "rgba(255, 255, 255, 0.85)",
            "glass_bg": "rgba(12, 16, 24, 0.46)",  # Kính mờ siêu nhẹ, ảnh nền hiển thị xuyên suốt
            "glass_border": "rgba(255, 255, 255, 0.18)",
            "glass_shadow": "0 16px 40px rgba(0, 0, 0, 0.5)",
            "text_shadow": "0 2px 10px rgba(0, 0, 0, 0.85)",
            "pill_bg": "rgba(255, 255, 255, 0.09)",
            "pill_border": "rgba(255, 255, 255, 0.20)",
            "pill_text": "#FFFFFF",
            "cta_bg": tc,
            "cta_text": badge_text,
            "badge_bg": tc,
            "badge_text": badge_text,
            "on_card_accent": on_card_accent,
            "on_card_title": on_card_title,
            "on_canvas_accent": on_canvas_accent,
            "gradient_overlay": "linear-gradient(to bottom, rgba(12, 16, 24, 0.75) 0%, rgba(12, 16, 24, 0.35) 70%, rgba(12, 16, 24, 0) 100%)",
        }


def palette_from_color_harmony(cp: Any, theme_color: Optional[str] = None) -> Dict[str, str]:
    """Chuyển đổi đối tượng ColorPalette từ tendoo.core.colors thành dict tương thích template v3 với bảo đảm tương phản WCAG."""
    is_dark = bool(getattr(cp, "is_dark", True))
    tc = theme_color or getattr(cp, "accent_color", "#D4AF37")

    raw_headline = getattr(cp, "headline_color", "#FFFFFF" if is_dark else "#0F172A")
    if "gradient" in str(raw_headline).lower():
        solid_primary = "#FFFFFF" if is_dark else "#0F172A"
    else:
        solid_primary = str(raw_headline)

    sub_color = getattr(cp, "sub_color", "rgba(255, 255, 255, 0.85)" if is_dark else "rgba(15, 23, 42, 0.82)")
    badge_bg = getattr(cp, "badge_bg", "rgba(255, 255, 255, 0.09)" if is_dark else "rgba(255, 255, 255, 0.75)")
    badge_border = getattr(cp, "badge_border", f"{tc}88" if is_dark else f"{tc}55")
    badge_text = getattr(cp, "badge_text", "#FFFFFF" if is_dark else "#0F172A")
    accent_color = getattr(cp, "accent_color", tc)
    cta_text = get_contrasting_text_color(accent_color)

    card_bg_solid = "#0C101A" if is_dark else "#FFFFFF"
    on_card_accent = getattr(cp, "on_card_accent", None) or ensure_contrast(accent_color, card_bg_solid, min_ratio=4.5)
    on_card_title = getattr(cp, "on_card_title", None) or ("#F8DC88" if is_dark else "#0F172A")
    on_canvas_accent = getattr(cp, "on_canvas_accent", None) or ensure_contrast(accent_color, "#000000" if is_dark else "#FFFFFF", min_ratio=4.5)

    if is_dark:
        return {
            "is_dark": True,
            "text_primary": solid_primary,
            "text_secondary": sub_color,
            "glass_bg": "rgba(12, 16, 24, 0.46)",
            "glass_border": "rgba(255, 255, 255, 0.18)",
            "glass_shadow": "0 16px 40px rgba(0, 0, 0, 0.5)",
            "text_shadow": getattr(cp, "text_shadow", "0 2px 10px rgba(0, 0, 0, 0.85)") or "0 2px 10px rgba(0, 0, 0, 0.85)",
            "pill_bg": badge_bg,
            "pill_border": badge_border,
            "pill_text": badge_text,
            "cta_bg": accent_color,
            "cta_text": cta_text,
            "badge_bg": tc,
            "badge_text": get_contrasting_text_color(tc),
            "on_card_accent": on_card_accent,
            "on_card_title": on_card_title,
            "on_canvas_accent": on_canvas_accent,
            "gradient_overlay": "linear-gradient(to bottom, rgba(12, 16, 24, 0.75) 0%, rgba(12, 16, 24, 0.35) 70%, rgba(12, 16, 24, 0) 100%)",
        }
    else:
        return {
            "is_dark": False,
            "text_primary": solid_primary,
            "text_secondary": sub_color,
            "glass_bg": "rgba(255, 255, 255, 0.70)",
            "glass_border": "rgba(255, 255, 255, 0.90)",
            "glass_shadow": "0 16px 36px rgba(0, 0, 0, 0.10)",
            "text_shadow": getattr(cp, "text_shadow", "0 1px 2px rgba(255, 255, 255, 0.8)") or "0 1px 2px rgba(255, 255, 255, 0.8)",
            "pill_bg": badge_bg,
            "pill_border": badge_border,
            "pill_text": badge_text,
            "cta_bg": accent_color,
            "cta_text": cta_text,
            "badge_bg": tc,
            "badge_text": get_contrasting_text_color(tc),
            "on_card_accent": on_card_accent,
            "on_card_title": on_card_title,
            "on_canvas_accent": on_canvas_accent,
            "gradient_overlay": "linear-gradient(to bottom, rgba(255, 255, 255, 0.85) 0%, rgba(255, 255, 255, 0.40) 70%, rgba(255, 255, 255, 0) 100%)",
        }


from tendoo_v3.icons import render_qr_code_svg, render_star_rating_svg


# Class autofit theo cấp thị giác (DESIGN_PRINCIPLES §1.1) -- nguồn DUY NHẤT, dùng cho
# bước giữ thứ bậc trong COMMON_AUTOFIT_JS và cho scripts/probe_type_hierarchy.py.
# Không có menu-list/steps-grid/testimonial-quote: nội dung chính của template dạng bảng.
TIER1_CLASSES = ("hero-title", "hero-top-title")
TIER2_CLASSES = ("subhead-title", "subhead-date", "subhead-benefit")
TIER3_CLASSES = (
    "badge-pill", "badge-capsule", "kicker-tag", "kicker-capsule", "cta-btn",
    "store-info-row", "store-info-col", "store-details-row", "store-text", "store-item",
    "extra-tag-row", "freetext-block", "flexible-stack", "extra-pills-wrap",
    "message-container", "reviewer-info",
)
CONTENT_CLASSES = ("menu-list", "steps-grid", "testimonial-quote")

COMMON_AUTOFIT_JS = """
<script>
(function() {
  function fitElements() {
    // 1. Tự động co giãn kích cỡ 2 chiều bằng thuật toán Binary Search Fill-to-Bounds
    const autofitEls = document.querySelectorAll('[data-autofit]');
    autofitEls.forEach(el => {
      const parent = el.parentElement;
      if (!parent) return;
      let maxH = parent.clientHeight;
      let maxW = parent.clientWidth;
      if (el.dataset.maxHeight) maxH = parseFloat(el.dataset.maxHeight);
      if (el.dataset.maxWidth) maxW = parseFloat(el.dataset.maxWidth);

      let computed = window.getComputedStyle(el);
      let baseSize = parseFloat(computed.fontSize) || 28;
      let minSize = parseFloat(el.dataset.minFont) || 12;
      let maxSize = parseFloat(el.dataset.maxFont) || (baseSize * 1.5);
      if (maxSize < minSize) maxSize = minSize + 10;

      // Nếu KHÔNG có data-max-width tường minh, mặc định lấy parent.clientWidth làm
      // trần -- SAI khi chính phần tử có CSS `width`/`max-width` cố định hẹp hơn
      // parent thật (vd pill CTA `width: 95%` trong khối is_portrait_narrow): thuật
      // toán tưởng còn dư chỗ (theo parent) nên phóng to chữ vượt khỏi khung chính
      // nó, tràn ra 2 bên (bug thực tế: "MỞ THẺ ONLINE TRONG 3 PHÚT" tràn pill ở
      // diagonal_slash). Đo clientWidth của CHÍNH phần tử tại maxSize (ngưỡng font
      // lớn nhất có thể thử) để bắt đúng trần CSS tự thân (dù là `width:X%` cố định
      // hay `max-width:X%` chỉ chặn khi nội dung đủ lớn) -- lấy min() với trần cũ,
      // không ảnh hưởng phần tử auto-width bình thường (tự nới theo nội dung, luôn
      // rộng hơn hoặc bằng trần cũ nên min() vẫn chọn đúng trần cũ).
      if (!el.dataset.maxWidth) {
        const prevFontSize = el.style.fontSize;
        el.style.fontSize = maxSize + 'px';
        const selfCapW = el.clientWidth;
        el.style.fontSize = prevFontSize;
        if (selfCapW > 0 && selfCapW < maxW) {
          maxW = selfCapW;
        }
      }

      // Tìm kiếm nhị phân 10 bước: Tìm cỡ chữ LỚN NHẤT CÓ THỂ trong dải [minSize, maxSize]
      let low = minSize;
      let high = maxSize;
      let bestSize = minSize;
      
      for (let iter = 0; iter < 10; iter++) {
        let mid = (low + high) / 2;
        el.style.fontSize = mid + 'px';
        
        // Dynamic Line-Height: Chữ nhỏ / nhiều dòng tự động nén line-height để không chiếm diện tích dọc
        if (mid < 26) {
          el.style.lineHeight = '1.15';
        } else if (mid < 36) {
          el.style.lineHeight = '1.2';
        } else {
          el.style.lineHeight = '1.3';
        }
        
        // Kiểm tra xem có nằm gọn trong bounding box hình học không
        // Cho phép dung sai 2.0px đối với scrollWidth để không bị nghẽn bởi làm tròn subpixel của width: 100%
        if (el.scrollHeight <= maxH && el.scrollWidth <= maxW + 2.0) {
          bestSize = mid;
          low = mid; // Vừa vặn -> Thử phóng to lên thêm
        } else {
          high = mid; // Tràn -> Thu nhỏ lại
        }
      }
      
      // Áp dụng cỡ chữ tối ưu cuối cùng (làm tròn 0.5px)
      let finalFont = Math.floor(bestSize * 2) / 2;
      el.style.fontSize = finalFont + 'px';
      if (finalFont < 26) {
        el.style.lineHeight = '1.15';
      } else if (finalFont < 36) {
        el.style.lineHeight = '1.2';
      } else {
        el.style.lineHeight = '1.3';
      }

      // Dynamic Letter-Spacing: Nếu text phải co về gần sàn (dưới 32px), co nhẹ kerning
      if (finalFont <= 32) {
        el.style.letterSpacing = '-0.5px';
      }

      // Nếu là container chứa danh sách pills/cards (như .flexible-stack, .extra-tag-row, .steps-grid, .board-col-left, .store-info-col)
      // mà vẫn chớm tràn, tự động co gap và padding của các phần tử con để giữ trọn vẹn 100% nội dung
      if (el.scrollHeight > maxH && el.children.length > 0) {
        el.style.gap = '3px';
        Array.from(el.children).forEach(child => {
          child.style.paddingTop = '2px';
          child.style.paddingBottom = '2px';
          child.style.paddingLeft = '6px';
          child.style.paddingRight = '6px';
        });
      }
    });

    // 2. NGUYÊN TẮC BẤT DI BẤT DỊCH: KHÔNG MẤT CHỮ (ZERO-LOSS & ANTI-CLIPPING RULE)
    // Thay vì ẩn phần tử bằng display:none làm mất thông tin của người dùng (vi phạm yeu_cau_templates.txt),
    // ta tự động co padding, gap và font-size của các pills/items để tất cả đều hiển thị trọn vẹn.
    //
    // CẢNH BÁO ĐÃ SỬA (2026-09-21): fitElements() chạy 2-3 lần/trang (DOMContentLoaded +
    // load + setTimeout dự phòng). Trước đây mỗi lần chạy, nếu container vẫn tràn, code
    // này trừ thêm 0.5px vào ĐÚNG con đó qua inline style -- nhưng ghi inline style lên
    // con phá luôn `font-size: inherit` mà nhiều class con (vd .store-item, .message-line)
    // đang dùng để nhận cỡ chữ tính toán ở Bước 1 -- khiến 2 hệ quả cộng dồn: (a) lần chạy
    // sau, binary-search ở Bước 1 không còn ảnh hưởng con nữa (con đã "chốt cứng" bằng
    // inline riêng, không còn kế thừa), (b) mỗi lần fitElements() gọi lại, trừ thêm 0.5px
    // NỮA -- tụt dần không giới hạn qua nhiều lần chạy, có thể xuống dưới sàn 12px hệ thống
    // type-scale mới dù bản thân container đã tính đúng cỡ lớn hơn nhiều. Sửa bằng 2 lớp:
    // (1) cờ `tendooAntiClipDone` chặn áp dụng quá 1 lần cho cùng 1 container -- không còn
    // cộng dồn qua nhiều lần fitElements() gọi lại; (2) nâng sàn cứng 9.5 -> 11.5 (sát đúng
    // sàn Label-tier 12px hệ thống, chỉ chừa margin nhỏ cho đúng nghĩa "giải pháp cuối cùng
    // chống tràn" chứ không phải mức bình thường).
    const pillContainers = document.querySelectorAll('.flexible-stack, .extra-tag-row, .steps-grid, .board-col-left, .store-info-col, .store-info-row, .store-details-row, .extra-pills-wrap, .freetext-block, .menu-list, .message-container');
    pillContainers.forEach(container => {
      if (container.dataset.tendooAntiClipDone === '1') return;
      const maxAllowedH = container.dataset.maxHeight ? parseFloat(container.dataset.maxHeight) : container.clientHeight;
      if (container.scrollHeight > maxAllowedH + 1.0) {
        container.dataset.tendooAntiClipDone = '1';
        container.style.gap = '2px';
        Array.from(container.children).forEach(child => {
          child.style.paddingTop = '2px';
          child.style.paddingBottom = '2px';
          child.style.paddingLeft = '5px';
          child.style.paddingRight = '5px';
          let curF = parseFloat(window.getComputedStyle(child).fontSize);
          if (curF > 11.5) {
            child.style.fontSize = (curF - 0.5) + 'px';
          }
        });
      }
    });

    // Kiểm tra an toàn bổ sung ở cấp độ cột/khối tổng thể
    const layoutColumns = document.querySelectorAll('.split-column-left, .split-column-right, .board-card, .feedback-card, .roadmap-platform');
    layoutColumns.forEach(col => {
      const colRect = col.getBoundingClientRect();
      const bottomStack = col.querySelector('.bottom-stack, .conversion-bar, .footer-action-row');
      const limitBottom = bottomStack ? (bottomStack.getBoundingClientRect().top - 8.0) : (colRect.bottom - 6.0);
      const items = col.querySelectorAll('.extra-item, .step-card');
      for (let i = items.length - 1; i >= 0; i--) {
        const item = items[i];
        if (item.style.display === 'none') continue;
        const itemRect = item.getBoundingClientRect();
        if (itemRect.bottom > limitBottom + 2.0) {
          // Co font và padding thay vì ẩn ngay lập tức
          item.style.padding = '2px 4px';
          item.style.fontSize = '10px';
        }
      }
    });

    // 3. THỨ BẬC CẤP 2 > CẤP 3 THEO CỠ ĐO THẬT (GĐ 0A+, ROADMAP §4.1). Chặn trần ở
    // renderer.py::_apply_tier_caps chưa đủ: subhead hay bị CHIỀU CAO khoá dưới trần
    // của nó, trong khi cta/store vẫn đạt trần -- đo thật còn 22/379 case đảo bậc chỉ
    // với chặn trần. Ở đây phần tử Cấp 3 nào to hơn subhead THỰC TẾ thì co về đúng
    // bằng nó. Chỉ co, không bao giờ phóng -> không thể sinh tràn mới.
    function tendooEffFont(el) {
      let eff = 0;
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode;
        if (!node.textContent.trim()) continue;
        const host = node.parentElement;
        if (host.closest('svg') || host.classList.contains('deco-bullet') || host.classList.contains('freetext-bullet')) continue;
        const r = host.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) continue;
        eff = Math.max(eff, parseFloat(getComputedStyle(host).fontSize) || 0);
      }
      return eff;
    }
    let tier2Font = 0;
    document.querySelectorAll('__TENDOO_TIER2_SELECTOR__').forEach(el => {
      tier2Font = Math.max(tier2Font, tendooEffFont(el));
    });
    if (tier2Font > 0) {
      document.querySelectorAll('__TENDOO_TIER3_SELECTOR__').forEach(el => {
        const eff = tendooEffFont(el);
        if (eff <= tier2Font) return;
        const k = tier2Font / eff;
        const cur = parseFloat(getComputedStyle(el).fontSize) || 0;
        el.style.fontSize = (Math.floor(cur * k * 2) / 2) + 'px';
        // Con đã bị ghi cỡ px inline (anti-clip ở Bước 2, hoặc autofit lồng) không còn
        // kế thừa cha -> phải co cùng tỉ lệ, nếu không vẫn to hơn subhead.
        el.querySelectorAll('*').forEach(ch => {
          if (ch.style.fontSize && ch.style.fontSize.endsWith('px')) {
            ch.style.fontSize = (Math.floor(parseFloat(ch.style.fontSize) * k * 2) / 2) + 'px';
          }
        });
      });
    }

    window.__tendooAutofitDone = true;
  }

  if (document.readyState === 'complete') {
    fitElements();
  } else {
    window.addEventListener('load', fitElements);
    document.addEventListener('DOMContentLoaded', fitElements);
  }
  setTimeout(fitElements, 150);
})();
</script>
""".replace("__TENDOO_TIER2_SELECTOR__", ", ".join("." + c for c in TIER2_CLASSES)).replace(
    "__TENDOO_TIER3_SELECTOR__", ", ".join("." + c for c in TIER3_CLASSES)
)
