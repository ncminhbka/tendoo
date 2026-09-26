"""
src/tendoo_v3/components.py

Linh kiện đồ hoạ GĐ 2 (ROADMAP §4.4): tính TOÀN BỘ phần "quyết định" ở Python -- lựa chọn hiệu
lực, vị trí, cỡ, màu, toạ độ hạt -- để HTML/CSS chỉ vẽ lại, không có JS đo đạc hay ngẫu nhiên lúc
chạy (bài học CircleType.js / Canvas-Confetti ở §4.4: render phải tất định, không đụng autofit).

Mọi linh kiện là TUỲ CHỌN: plan không yêu cầu -> `build_components` trả None -> template render
y hệt trước GĐ 2 (kiểm bằng visual_diff: 379 case giống từng điểm ảnh).

  badge_style  pill (mặc định) | ribbon (dải băng đuôi nheo) | capsule ("A | B" hai màu) |
               stamp (con dấu tròn, chữ uốn cung bằng SVG <textPath>, đặt ở góc trống ngoài vùng chữ)
  stat_style   plain (mặc định) | unit (đơn vị "%"/"K"/"TRIỆU" nhỏ, nhấc lên) | burst (unit + nền sao nổ)
  decor        none (mặc định) | sparkles (hạt lấp lánh tĩnh, seed theo nội dung, tránh vùng chữ)
"""

from __future__ import annotations

import html
import json
import logging
import math
import random
import re
import zlib
from typing import Any, Dict, List, Optional, Tuple

from tendoo_core.colors import calculate_contrast_ratio, get_contrasting_text_color, parse_color_to_rgb, rgb_to_hex
from tendoo_v3.styles import TIER1_CLASSES

logger = logging.getLogger(__name__)

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2

# Tỉ lệ cỡ dấu so với cạnh ngắn của khung hình, và lề tới mép. Chọn bằng mắt trên bộ mẫu GĐ 2
# (scripts/render_components_showcase.py): 0.22 đủ đọc chữ vòng ở khung 576px mà không lấn sản phẩm.
STAMP_SIZE_RATIO = 0.22
STAMP_MARGIN_RATIO = 0.04
# Chữ vòng dấu: bán kính đường chạy chữ và cỡ chữ tối đa trong hệ toạ độ viewBox 0..100.
# Cỡ 9.5 -> đỉnh chữ hoa ~ r 37 + 0.72*9.5 = 43.8 < vòng răng cưa r 45.
_STAMP_ARC_R = 37.0
_STAMP_FONT_MAX = 9.5
STAMP_FONT_MIN = 5.5  # dưới mức này chữ vòng không đọc được ở khung 576px -> validator cảnh báo
_STAMP_CHAR_W = 0.66   # bề rộng trung bình 1 ký tự hoa Be Vietnam Pro ExtraBold, tính theo em

# Hạt lấp lánh quanh hero: số hạt, khoảng cách ra ngoài mép hộp hero và bán kính hạt -- cả hai
# tính theo cỡ chữ hero sau autofit (em) để co giãn đúng tỉ lệ ở mọi khung hình.
_SPARKLE_COUNT = 16
_SPARKLE_GAP = (0.15, 1.1)
_SPARKLE_SIZE = (0.14, 0.38)

# Đơn vị tách khỏi con số trong stat: "50%", "-30%", "2 TRIỆU", "500K", "99.000Đ", "10+".
_STAT_UNIT_RE = re.compile(r"^([-–+]?\d[\d.,]*)(\s*(?:%|\+|K|k|Đ|đ|VNĐ|TR|TRIỆU|NGHÌN|NGÀN|TỶ|X|x))$")


def split_stat(text: str) -> Optional[Tuple[str, str]]:
    """'50%' -> ('50', '%'). None nếu stat không phải số+đơn vị (vd 'MIỄN PHÍ') -> render như cũ."""
    m = _STAT_UNIT_RE.match(text or "")
    return (m.group(1), m.group(2)) if m else None


def star_polygon(points: int = 24, inner: float = 0.88) -> str:
    """clip-path polygon() của sao nổ `points` cánh (toạ độ %), dùng cho stat_style=burst."""
    pts = []
    for i in range(points * 2):
        a = math.pi * i / points - math.pi / 2
        r = 50 if i % 2 == 0 else 50 * inner
        pts.append(f"{50 + r * math.cos(a):.2f}% {50 + r * math.sin(a):.2f}%")
    return f"polygon({', '.join(pts)})"


def _intersects(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _zone_boxes(zones: Dict[str, Any]) -> List[Box]:
    out = []
    for z in zones.values():
        if isinstance(z, dict) and {"x1", "y1", "x2", "y2"} <= z.keys():
            out.append((z["x1"], z["y1"], z["x2"], z["y2"]))
        elif hasattr(z, "x1"):
            out.append((z.x1, z.y1, z.x2, z.y2))
    return out


def find_stamp_box(zone_boxes: List[Box], width: int, height: int) -> Optional[Box]:
    """Chỗ đặt con dấu không đè vùng chữ nào: thử 4 góc khung hình trước, rồi sát mép phải/trái
    ngay dưới hoặc ngay trên từng vùng chữ (khung dọc: dải chữ phủ hết bề ngang nên 4 góc đều
    kẹt -- bản đầu rơi về pill ở 2/3 mẫu 9:16). None = hết chỗ (vd luxury_centered_card) -> pill."""
    d = STAMP_SIZE_RATIO * min(width, height)
    m = STAMP_MARGIN_RATIO * min(width, height)
    right, left = width - m - d, m
    candidates = [(right, m), (left, m), (right, height - m - d), (left, height - m - d)]
    for z in zone_boxes:
        candidates += [(right, z[3] + m), (left, z[3] + m), (right, z[1] - m - d), (left, z[1] - m - d)]
    for x, y in candidates:
        box = (x, y, x + d, y + d)
        inside = box[1] >= 0 and box[3] <= height
        if inside and not any(_intersects(box, z) for z in zone_boxes):
            return box
    return None


def stamp_ring(badge: str) -> Tuple[str, float]:
    """Chuỗi chạy quanh vòng dấu (lặp badge cho kín vòng) và cỡ chữ (đơn vị viewBox)."""
    circ = 2 * math.pi * _STAMP_ARC_R
    unit = f"{badge.strip().upper()} • "
    repeats = max(1, int(circ / (len(unit) * _STAMP_CHAR_W * _STAMP_FONT_MAX)))
    ring = unit * repeats
    font = min(_STAMP_FONT_MAX, circ * 0.97 / (len(ring) * _STAMP_CHAR_W))
    return ring, round(font, 2)


def sparkles_html(seed_text: str, color: str, anchor_selector: str) -> str:
    """Hạt lấp lánh 4 cánh TỤ QUANH HERO (bản đầu rải khắp nền -> nhìn như nhiễu, xem ảnh mẫu GĐ 2).

    Python quyết định mọi thứ theo seed (cùng nội dung -> cùng ảnh): góc, khoảng cách, cỡ, màu.
    JS chỉ quy đổi ra px quanh hộp hero ĐÃ CHỐT sau autofit (móc `__tendooAfterFit`, chạy trước
    cờ xong). Tâm = con số stat nếu có (điểm neo thị giác), không thì cả hero. Nhiều hạt hơn cần
    thiết (_SPARKLE_COUNT) vì hạt chạm dòng chữ / con dấu hoặc ra ngoài khung bị bỏ lúc chạy."""
    rng = random.Random(zlib.crc32(seed_text.encode("utf-8")))
    specs = []
    while len(specs) < _SPARKLE_COUNT:
        deg = rng.uniform(0, 360)
        specs.append([round(math.radians(deg), 3), round(rng.uniform(*_SPARKLE_GAP), 3), round(rng.uniform(*_SPARKLE_SIZE), 3),
                      "#FFFFFF" if rng.random() < 0.45 else color, round(rng.uniform(0.6, 0.95), 2)])
    return f"""<svg class="tk-sparkles" style="position:absolute;inset:0;width:100%;height:100%;overflow:visible;"></svg>
<script>
window.__tendooAfterFit = function() {{
  const host = document.querySelector('.tk-sparkles');
  const hero = document.querySelector('.hero-seg--stat') || document.querySelector({json.dumps(anchor_selector)});
  if (!host || !hero) return;
  // Nhấc lên trên mảng nền cụm chữ (nhiều template vẽ panel ở z-index 10, che mất lớp decor z 5)
  // nhưng KHÔNG đè chữ: hạt nào chạm hộp chữ đã đo đều bị bỏ.
  const canvas = host.closest('.poster-canvas') || document.body;
  canvas.appendChild(host);
  host.style.zIndex = 12; host.style.pointerEvents = 'none';
  // Hộp của TỪNG DÒNG CHỮ thật (không phải hộp chứa): hộp hero rộng cả khung sẽ chặn mọi hạt.
  const texts = [];
  for (const el of document.querySelectorAll('[data-autofit], .tk-stamp')) {{
    if (el.classList.contains('tk-stamp')) {{ texts.push(el.getBoundingClientRect()); continue; }}
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    while (w.nextNode()) {{
      if (!w.currentNode.textContent.trim()) continue;
      const rg = document.createRange(); rg.selectNodeContents(w.currentNode);
      for (const b of rg.getClientRects()) if (b.width > 1) texts.push(b);
    }}
  }}
  const c = host.getBoundingClientRect(), r = hero.getBoundingClientRect();
  const fs = parseFloat(getComputedStyle(hero).fontSize) || 40;
  const cx = r.left + r.width / 2 - c.left, cy = r.top + r.height / 2 - c.top;
  const NS = 'http://www.w3.org/2000/svg';
  host.innerHTML = '';
  for (const [a, gap, size, fill, op] of {json.dumps(specs)}) {{
    const x = cx + Math.cos(a) * (r.width / 2 + gap * fs), y = cy + Math.sin(a) * (r.height / 2 + gap * fs);
    const s = size * fs;
    if (x < s || y < s || x > c.width - s || y > c.height - s) continue;
    const X = x + c.left, Y = y + c.top;
    if (texts.some(b => X + s > b.left && X - s < b.right && Y + s > b.top && Y - s < b.bottom)) continue;
    const p = document.createElementNS(NS, 'path');
    p.setAttribute('transform', 'translate(' + x.toFixed(1) + ' ' + y.toFixed(1) + ') scale(' + s.toFixed(2) + ')');
    p.setAttribute('fill', fill); p.setAttribute('opacity', op);
    p.setAttribute('d', 'M0-1C.12-.12.12-.12 1 0C.12.12.12.12 0 1C-.12.12-.12.12-1 0C-.12-.12-.12-.12 0-1Z');
    host.appendChild(p);
    texts.push({{left: X - s, right: X + s, top: Y - s, bottom: Y + s}});  // hạt sau không chồng hạt trước
  }}
}};
</script>"""


def stamp_svg(badge: str, box: Box, theme_color: str, text_color: str, uid: str = "tk-stamp") -> str:
    ring, font = stamp_ring(badge)
    circ = 2 * math.pi * _STAMP_ARC_R
    r = _STAMP_ARC_R
    x1, y1, x2, _ = box
    return f"""<svg class="tk-stamp" viewBox="0 0 100 100" aria-label="{html.escape(badge)}" style="position:absolute;left:{x1:.1f}px;top:{y1:.1f}px;width:{x2 - x1:.1f}px;height:{x2 - x1:.1f}px;transform:rotate(-12deg);filter:drop-shadow(0 6px 10px rgba(0,0,0,.45));overflow:visible;">
  <defs><path id="{uid}-arc" d="M{50 - r},50 a{r},{r} 0 1,1 {2 * r},0 a{r},{r} 0 1,1 -{2 * r},0"/></defs>
  <circle cx="50" cy="50" r="49" fill="{theme_color}"/>
  <circle cx="50" cy="50" r="45.5" fill="none" stroke="{text_color}" stroke-width="0.9" stroke-dasharray="1.6 1.4" opacity="0.8"/>
  <circle cx="50" cy="50" r="29" fill="none" stroke="{text_color}" stroke-width="0.9" opacity="0.8"/>
  <text font-family="'Be Vietnam Pro', sans-serif" font-weight="800" font-size="{font}" fill="{text_color}"><textPath href="#{uid}-arc" textLength="{circ * 0.97:.1f}" lengthAdjust="spacing">{html.escape(ring)}</textPath></text>
  <path transform="translate(50 50) scale(17)" fill="{text_color}" d="M0-1L.235-.324.951-.309.38.124.588.809 0 .4-.588.809-.38.124-.951-.309-.235-.324Z"/>
</svg>"""



def accessible_fill(theme: str, min_ratio: float = 4.5) -> Tuple[str, str]:
    """(màu nền khối, màu chữ) cho khối màu nhấn chứa CHỮ NHỎ (nửa viên nang, dải băng...). Giữ
    sắc thương hiệu, chỉ trộn dần về đen (chữ trắng) hoặc trắng (chữ navy) tới khi đạt WCAG 4.5:1 --
    lấy hướng thay đổi ÍT nhất. Đo GĐ 5: xanh #3B82F6 cả chữ trắng (3.68) lẫn navy (4.2) đều trượt."""
    text = get_contrasting_text_color(theme)
    if calculate_contrast_ratio(theme, text) >= min_ratio:
        return theme, text
    r, g, b = parse_color_to_rgb(theme)
    best = None
    for toward, txt in (((0, 0, 0), "#FFFFFF"), ((255, 255, 255), "#0F172A")):
        for i in range(1, 11):
            t = i / 10
            cand = rgb_to_hex(r * (1 - t) + toward[0] * t, g * (1 - t) + toward[1] * t, b * (1 - t) + toward[2] * t)
            if calculate_contrast_ratio(cand, txt) >= min_ratio:
                if best is None or t < best[0]:
                    best = (t, cand, txt)
                break
    return (best[1], best[2]) if best else (theme, text)

def build_components(plan: Any, zones: Dict[str, Any], width: int, height: int) -> Optional[Dict[str, Any]]:
    """Dữ liệu render linh kiện cho template (biến Jinja `components`), hoặc None nếu plan không
    dùng linh kiện nào -> template giữ nguyên hành vi trước GĐ 2."""
    badge_style = (plan.badge_style or "pill") if plan.badge else "pill"
    stat_style = plan.stat_style or "plain"
    decor = plan.decor or "none"
    if badge_style == "pill" and stat_style == "plain" and decor == "none":
        return None

    theme = plan.style.theme_color
    on_theme = get_contrasting_text_color(theme)
    # Khối màu nhấn chứa chữ nhỏ (nửa phải viên nang, dải băng): nền chỉnh sắc độ để đạt 4.5:1.
    fill, on_fill = accessible_fill(theme)
    boxes = _zone_boxes(zones)
    comp: Dict[str, Any] = {"on_theme": on_theme, "fill": fill, "on_fill": on_fill, "decor_html": ""}

    if badge_style == "capsule":
        left, sep, right = plan.badge.partition("|")
        if sep and left.strip() and right.strip():
            comp["capsule"] = (left.strip(), right.strip())
        else:
            badge_style = "pill"  # validator đã cảnh báo; không đoán chỗ tách
    stamp_box = None
    if badge_style == "stamp":
        stamp_box = find_stamp_box(boxes, width, height)
        if stamp_box is None:
            logger.info(f"[components] '{plan.template}' không còn góc trống cho con dấu -> badge pill")
            badge_style = "pill"
        else:
            comp["decor_html"] += stamp_svg(plan.badge, stamp_box, theme, on_theme)
    if decor == "sparkles":
        anchor = ", ".join("." + c for c in TIER1_CLASSES)
        comp["decor_html"] = sparkles_html(plan.hero, theme, anchor) + comp["decor_html"]
    if stat_style in ("unit", "burst"):
        comp["burst_polygon"] = star_polygon() if stat_style == "burst" else None

    comp.update(badge_style=badge_style, stat_style=stat_style, decor=decor)
    return comp


def enrich_hero_parts(hero_parts: List[Dict[str, Any]], stat_style: str) -> List[Dict[str, Any]]:
    """Thêm num/unit cho đoạn stat khi stat_style cần tách đơn vị. Nội dung chữ KHÔNG đổi
    (num + unit == t, Cổng 1 nguyên văn vẫn đúng)."""
    if stat_style not in ("unit", "burst"):
        return hero_parts
    out = []
    for seg in hero_parts:
        seg = dict(seg)
        if seg.get("role") == "stat":
            parts = split_stat(seg.get("t", ""))
            if parts:
                seg["num"], seg["unit"] = parts
                # Ký hiệu (%, K, +, Đ) nhấc lên ngang đỉnh số; đơn vị là CHỮ (TRIỆU, NGHÌN...) giữ
                # đường chân chữ -- nhấc lên thì dấu tiếng Việt đụng đỉnh số (ảnh mẫu GĐ 2).
                seg["unit_word"] = len(parts[1].strip()) > 1
        out.append(seg)
    return out


__all__ = ["STAMP_FONT_MIN", "build_components", "enrich_hero_parts", "find_stamp_box", "sparkles_html", "split_stat", "stamp_ring", "star_polygon"]
