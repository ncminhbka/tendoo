"""
src/tendoo_v3/styles.py

Thư viện hiệu ứng CSS & Macro đồ họa cao cấp cho Tendoo v3:
- Hiệu ứng chữ: 3D Gold, Neon Glow, Liquid Chrome, Minimal Shadow, Plain Elegant.
- Kính mờ cao cấp (Glassmorphism): backdrop-filter blur, viền phát sáng đa tầng, đổ bóng ma trận.
- Tự động nhúng font tiếng Việt Base64 (100% offline).
"""

from __future__ import annotations

from typing import Dict, Tuple

from tendoo.core.fonts import resolve_font


def get_effect_css(effect_name: str, theme_color: str = "#D4AF37") -> str:
    """Trả về các quy tắc CSS cho hiệu ứng chữ đã chọn."""
    eff = (effect_name or "plain_elegant").lower().strip()

    if eff == "3d_gold":
        return f"""
        background: linear-gradient(135deg, #FFE082 0%, #FFD700 25%, #FFA000 50%, #FFD54F 75%, #FFE082 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.9)) drop-shadow(0 8px 18px rgba(255, 179, 0, 0.35));
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
        filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.85)) drop-shadow(0 0 15px rgba(200, 220, 255, 0.4));
        """
    elif eff == "fire":
        return """
        background: linear-gradient(180deg, #FFFF99 0%, #FFCC00 25%, #FF6600 65%, #CC0000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.9)) drop-shadow(0 0 25px rgba(255, 102, 0, 0.6));
        """
    elif eff == "shadow":
        return """
        color: #FFFFFF;
        text-shadow:
          0 2px 4px rgba(0, 0, 0, 0.95),
          0 6px 16px rgba(0, 0, 0, 0.8),
          0 12px 32px rgba(0, 0, 0, 0.6);
        """
    elif eff == "embossed":
        return """
        color: #E2E8F0;
        text-shadow:
          1px 1px 0 rgba(255, 255, 255, 0.45),
          -1px -1px 0 rgba(0, 0, 0, 0.65),
          0 2px 8px rgba(0, 0, 0, 0.75);
        """
    elif eff == "chromatic":
        return """
        color: #FFFFFF;
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
        return """
        color: #FFFFFF;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.75), 0 4px 20px rgba(0, 0, 0, 0.5);
        """


def get_adaptive_palette(background_tone: str, theme_color: str = "#FFB300") -> Dict[str, str]:
    """Tính toán bảng màu thích ứng cho Transparent Glass Overlay, loại bỏ hoàn toàn các vệt đen."""
    tone = (background_tone or "dark_luxury").lower().strip()
    tc = theme_color if theme_color else "#FFB300"

    if tone in ("pastel", "light_clean"):
        return {
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
            "cta_text": "#FFFFFF" if tone == "pastel" else "#000000",
            "gradient_overlay": "linear-gradient(to bottom, rgba(255, 255, 255, 0.85) 0%, rgba(255, 255, 255, 0.40) 70%, rgba(255, 255, 255, 0) 100%)",
        }
    elif tone == "cyber_neon":
        return {
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
            "cta_text": "#000000",
            "gradient_overlay": "linear-gradient(to bottom, rgba(10, 15, 28, 0.75) 0%, rgba(10, 15, 28, 0.35) 70%, rgba(10, 15, 28, 0) 100%)",
        }
    else:  # dark_luxury, warm_rustic, cinema_red
        return {
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
            "cta_text": "#000000",
            "gradient_overlay": "linear-gradient(to bottom, rgba(12, 16, 24, 0.75) 0%, rgba(12, 16, 24, 0.35) 70%, rgba(12, 16, 24, 0) 100%)",
        }


def palette_from_color_harmony(cp: Any, theme_color: Optional[str] = None) -> Dict[str, str]:
    """Chuyển đổi đối tượng ColorPalette từ tendoo.core.colors thành dict 12 khóa tương thích template v3."""
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
    cta_text = "#000000" if not is_dark else "#FFFFFF"

    if is_dark:
        return {
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
            "gradient_overlay": "linear-gradient(to bottom, rgba(12, 16, 24, 0.75) 0%, rgba(12, 16, 24, 0.35) 70%, rgba(12, 16, 24, 0) 100%)",
        }
    else:
        return {
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
            "gradient_overlay": "linear-gradient(to bottom, rgba(255, 255, 255, 0.85) 0%, rgba(255, 255, 255, 0.40) 70%, rgba(255, 255, 255, 0) 100%)",
        }


def get_vfx_overlay_html_and_css(
    vfx_name: Optional[str],
    width: int,
    height: int,
    theme_color: str = "#D4AF37",
) -> str:
    """Sinh thẻ HTML/SVG cho hiệu ứng quang học & khí quyển Cinematic VFX."""
    vfx = (vfx_name or "none").lower().strip()
    if vfx in ("none", "", "null", "false"):
        return ""

    if vfx == "film_grain":
        # Procedural 35mm film grain qua SVG feTurbulence
        return f"""
    <div class="vfx-layer vfx-film-grain" style="position:absolute; inset:0; pointer-events:none; z-index:4; mix-blend-mode:overlay; opacity:0.24;">
      <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="position:absolute; width:100%; height:100%;">
        <filter id="tendoo-grain-{width}">
          <feTurbulence type="fractalNoise" baseFrequency="0.75" numOctaves="3" stitchTiles="stitch"/>
          <feColorMatrix type="saturate" values="0"/>
        </filter>
        <rect width="100%" height="100%" filter="url(#tendoo-grain-{width})"/>
      </svg>
    </div>
    """
    elif vfx == "anamorphic_flare":
        c = theme_color if theme_color else "#00F0FF"
        return f"""
    <div class="vfx-layer vfx-anamorphic-flare" style="position:absolute; inset:0; pointer-events:none; z-index:4; overflow:hidden;">
      <div style="position:absolute; top:28%; left:-10%; right:-10%; height:3px; background:radial-gradient(ellipse 70% 100% at 50% 50%, {c} 0%, rgba(59,130,246,0.5) 45%, transparent 90%); mix-blend-mode:screen; filter:blur(0.5px);"></div>
      <div style="position:absolute; top:27%; left:30%; width:40%; height:20px; background:radial-gradient(ellipse at 50% 50%, {c}66 0%, transparent 70%); mix-blend-mode:screen; filter:blur(6px);"></div>
    </div>
    """
    elif vfx in ("gold_dust", "sparkles"):
        return """
    <div class="vfx-layer vfx-gold-dust" style="position:absolute; inset:0; pointer-events:none; z-index:4; overflow:hidden; mix-blend-mode:color-dodge;">
      <div style="position:absolute; top:12%; left:18%; width:5px; height:5px; border-radius:50%; background:#FFE082; box-shadow:0 0 10px #FFD700; filter:blur(0.5px);"></div>
      <div style="position:absolute; top:22%; left:75%; width:7px; height:7px; border-radius:50%; background:#FFF59D; box-shadow:0 0 14px #FFA000; filter:blur(0.8px);"></div>
      <div style="position:absolute; top:45%; left:12%; width:4px; height:4px; border-radius:50%; background:#FFE082; box-shadow:0 0 8px #FFD700;"></div>
      <div style="position:absolute; top:65%; left:82%; width:6px; height:6px; border-radius:50%; background:#FFF8E1; box-shadow:0 0 12px #FFCA28; filter:blur(0.5px);"></div>
      <div style="position:absolute; top:80%; left:25%; width:5px; height:5px; border-radius:50%; background:#FFE082; box-shadow:0 0 10px #FFD700;"></div>
      <div style="position:absolute; top:35%; left:88%; width:8px; height:8px; border-radius:50%; background:#FFF59D; box-shadow:0 0 16px #FF8F00; filter:blur(1px);"></div>
    </div>
    """
    elif vfx == "light_leak":
        return """
    <div class="vfx-layer vfx-light-leak" style="position:absolute; top:0; right:0; width:70%; height:70%; background:radial-gradient(circle at 100% 0%, rgba(255, 190, 70, 0.42) 0%, rgba(255, 120, 40, 0.16) 45%, transparent 75%); mix-blend-mode:screen; pointer-events:none; z-index:4;"></div>
    """
    elif vfx == "cinematic_haze":
        return """
    <div class="vfx-layer vfx-cinematic-haze" style="position:absolute; bottom:0; left:0; right:0; height:48%; background:linear-gradient(to top, rgba(0,0,0,0.68) 0%, rgba(0,0,0,0.22) 50%, transparent 100%); pointer-events:none; z-index:4;"></div>
    """
    return ""


from tendoo_v3.icons import render_qr_code_svg, render_star_rating_svg


COMMON_AUTOFIT_JS = """
<script>
(function() {
  function fitElements() {
    // 1. Thu nhỏ font chữ và co giãn padding/gap cho các phần tử [data-autofit]
    const autofitEls = document.querySelectorAll('[data-autofit]');
    autofitEls.forEach(el => {
      const parent = el.parentElement;
      if (!parent) return;
      let maxH = parent.clientHeight;
      let maxW = parent.clientWidth;
      if (el.dataset.maxHeight) maxH = parseFloat(el.dataset.maxHeight);
      if (el.dataset.maxWidth) maxW = parseFloat(el.dataset.maxWidth);
      
      let curSize = parseFloat(window.getComputedStyle(el).fontSize) || 16;
      let minSize = parseFloat(el.dataset.minFont) || 10;
      
      let step = 0;
      while ((el.scrollHeight > maxH || el.scrollWidth > maxW) && curSize > minSize && step < 50) {
        curSize -= 0.5;
        el.style.fontSize = curSize + 'px';
        step++;
      }

      // Nếu là container chứa danh sách pills/cards (như .flexible-stack, .extra-tag-row, .steps-grid, .board-col-left)
      // mà vẫn chớm tràn, tự động co gap và padding của các phần tử con để cố gắng giữ trọn vẹn nội dung
      if (el.scrollHeight > maxH && el.children.length > 0) {
        el.style.gap = '4px';
        Array.from(el.children).forEach(child => {
          child.style.paddingTop = '4px';
          child.style.paddingBottom = '4px';
          child.style.paddingLeft = '8px';
          child.style.paddingRight = '8px';
        });
      }
    });

    // 2. NGUYÊN TẮC BẤT DI BẤT DỊCH: CHỐNG CẮT PILL NỬA VỜI (ANTI-PILL-CLIPPING RULE)
    // Quét qua mọi container chứa pills/items/cards (.flexible-stack, .extra-tag-row, .steps-grid, .board-col-left, v.v.)
    // Sử dụng tọa độ viewport (getBoundingClientRect) chính xác tuyệt đối.
    // Nếu mép đáy của bất kỳ phần tử con nào vượt quá mép đáy an toàn của container,
    // ẨN NGAY LẬP TỨC (display: none) phần tử đó. TUYỆT ĐỐI KHÔNG BAO GIỜ HIỂN THỊ MỘT VIÊN PILL BỊ XÉN ĐÔI!
    const pillContainers = document.querySelectorAll('.flexible-stack, .extra-tag-row, .steps-grid, .board-col-left');
    pillContainers.forEach(container => {
      const containerRect = container.getBoundingClientRect();
      const maxAllowedH = container.dataset.maxHeight ? parseFloat(container.dataset.maxHeight) : container.clientHeight;
      const effectiveBottom = Math.min(containerRect.bottom, containerRect.top + maxAllowedH);
      const children = Array.from(container.children);
      
      for (let i = children.length - 1; i >= 0; i--) {
        const child = children[i];
        if (child.style.display === 'none') continue;
        const childRect = child.getBoundingClientRect();
        
        if (childRect.bottom > effectiveBottom + 1.5) {
          child.style.display = 'none';
        }
      }
    });

    // Kiểm tra an toàn bổ sung ở cấp độ cột/khối tổng thể (đảm bảo pills không đè lên CTA/bottom-stack)
    const layoutColumns = document.querySelectorAll('.split-column-left, .split-column-right, .board-card, .feedback-card, .roadmap-platform, .sandwich-top-band');
    layoutColumns.forEach(col => {
      const colRect = col.getBoundingClientRect();
      const bottomStack = col.querySelector('.bottom-stack, .conversion-bar, .footer-action-row');
      const limitBottom = bottomStack ? (bottomStack.getBoundingClientRect().top - 8.0) : (colRect.bottom - 6.0);
      const items = col.querySelectorAll('.extra-item, .extra-pill, .step-card');
      for (let i = items.length - 1; i >= 0; i--) {
        const item = items[i];
        if (item.style.display === 'none') continue;
        const itemRect = item.getBoundingClientRect();
        if (itemRect.bottom > limitBottom) {
          item.style.display = 'none';
        }
      }
    });

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
"""
