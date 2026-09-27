"""
src/tendoo_v3/style_packs.py -- STYLE PACK + HOẠ TIẾT (GĐ 8, ROADMAP §10.5-10.6).

Style pack là DỮ LIỆU, không phải template: một tổ hợp font / màu / hiệu ứng / linh kiện / hoạ tiết đã phối sẵn
theo dịp hoặc ngành -- để poster "có gu" nhất quán thay vì LLM ghép từng trường rời rạc. Brand kit của người
dùng (plan.brand_*) ghi đè pack.

Hoạ tiết (ornament): SVG vẽ tay, TẤT ĐỊNH theo seed, NEO vào chữ/vùng chữ -- đặt SAU khi autofit chốt vị trí:
  - chỉ nằm trong vùng chữ của template (vùng nền đã được giữ yên -- không đè sản phẩm),
  - KHÔNG chạm hộp của bất kỳ dòng chữ nào (cùng cơ chế đo với Cổng 4), không chồng hoạ tiết khác,
  - không có chỗ -> bỏ hoạ tiết đó (không bao giờ ép).
"""

from __future__ import annotations

import json
import random
import zlib
from dataclasses import replace
from typing import Any, Dict, List, Optional

# Mỗi pack: style (ghi đè plan.style trừ background_tone -- tông nền theo cảnh), linh kiện mặc định (chỉ điền khi
# LLM bỏ trống), hoạ tiết, intent phù hợp (Cổng 2), gợi ý cảnh (prompt).
STYLE_PACKS: Dict[str, Dict[str, Any]] = {
    "tet": {
        "name": "Tết Nguyên Đán", "intents": ("festive_event", "big_number_deal", "hook_headline"),
        "style": {"font": "playfair", "theme_color": "#C8102E", "text_effect": "3d_gold"},
        "components": {"badge_style": "ribbon", "decor": "sparkles"},
        "ornaments": ["blossom", "lantern", "corner_frame"],
        "scene": "hoa mai vàng, đèn lồng đỏ, bokeh vàng ấm",
    },
    "trung_thu": {
        "name": "Trung Thu", "intents": ("festive_event", "hook_headline", "big_number_deal"),
        "style": {"font": "playfair", "theme_color": "#F59E0B", "text_effect": "shadow"},
        "components": {"decor": "sparkles"},
        "ornaments": ["lantern", "dots", "corner_frame"],
        "scene": "trăng tròn, đèn lồng cam, trời xanh đêm",
    },
    "fnb_sale": {
        "name": "Sale đồ ăn / đồ uống", "intents": ("big_number_deal", "hook_headline", "product_showcase"),
        "style": {"font": "anton", "theme_color": "#E11D48", "text_effect": "shadow"},
        "components": {"badge_style": "ribbon", "stat_style": "unit"},
        "ornaments": ["dots", "flank_lines"],
        "scene": "món ăn cận cảnh, màu rực, ánh sáng studio",
    },
    "cafe": {
        "name": "Cà phê / trà sữa", "intents": ("hook_headline", "product_showcase", "big_number_deal", "matrix_board"),
        "style": {"font": "playfair", "theme_color": "#A16207", "text_effect": "plain_elegant"},
        "components": {},
        "ornaments": ["leaf", "flank_lines"],
        "scene": "gỗ ấm, hạt cà phê, nắng sớm",
    },
    "spa_beauty": {
        "name": "Spa / làm đẹp", "intents": ("product_showcase", "testimonial_trust", "hook_headline", "festive_event"),
        "style": {"font": "playfair", "theme_color": "#DB2777", "text_effect": "plain_elegant"},
        "components": {},
        "ornaments": ["leaf", "corner_frame"],
        "scene": "hồng pastel, hoa, lụa mềm",
    },
    "tech": {
        "name": "Công nghệ", "intents": ("product_showcase", "big_number_deal", "hook_headline", "matrix_board"),
        "style": {"font": "days", "theme_color": "#06B6D4", "text_effect": "shadow"},
        "components": {"badge_style": "capsule"},
        "ornaments": ["corner_frame", "dots"],
        "scene": "tối, ánh xanh neon, bề mặt kim loại",
    },
    "luxury": {
        "name": "Sang trọng", "intents": ("product_showcase", "hook_headline", "festive_event"),
        "style": {"font": "playfair", "theme_color": "#D4AF37", "text_effect": "plain_elegant"},
        "components": {},
        "ornaments": ["corner_frame", "flank_lines"],
        "scene": "đen, vàng kim, ánh sáng dịu",
    },
    "recruit": {
        "name": "Tuyển dụng", "intents": ("matrix_board", "hook_headline"),
        "style": {"font": "bevietnam", "theme_color": "#2563EB", "text_effect": "plain_elegant"},
        "components": {},
        "ornaments": ["dots", "corner_frame"],
        "scene": "văn phòng sáng, hiện đại",
    },
}


def apply_style_pack(plan: Any) -> Any:
    """Plan sau khi áp pack (trả plan gốc nếu không có pack hợp lệ). Style của pack GHI ĐÈ font/màu/hiệu ứng (tổ hợp
    đã phối sẵn); linh kiện chỉ điền khi LLM bỏ trống; brand kit (brand_color/brand_font) ghi đè cả pack."""
    pack = STYLE_PACKS.get(getattr(plan, "style_pack", None) or "")
    style = plan.style
    if pack:
        style = replace(style, **pack["style"])
        comp = {k: v for k, v in pack["components"].items() if getattr(plan, k, None) is None}
        plan = replace(plan, **comp) if comp else plan
    if getattr(plan, "brand_color", None):
        style = replace(style, theme_color=plan.brand_color)
    if getattr(plan, "brand_font", None):
        style = replace(style, font=plan.brand_font)
    return replace(plan, style=style) if style is not plan.style else plan


# ---- Hoạ tiết ------------------------------------------------------------------------------------------------
# Hình vẽ trong hộp đơn vị [-1, 1] (JS scale theo cỡ đặt). Nét mảnh, màu nhấn / màu phụ.
_MOTIFS = {
    # cành mai: nhánh cong + 3 bông 5 cánh + nụ
    "blossom": [
        ("path", {"d": "M-1 .9C-.4 .5 .1 .1 .95-.85", "fill": "none", "stroke": "$dark", "stroke-width": ".05", "stroke-linecap": "round"}),
        *[("g", {"transform": f"translate({x} {y}) scale({s})", "fill": "$gold",
                 "_petals": True}) for x, y, s in ((-.45, .45, .28), (.15, -.05, .34), (.62, -.55, .24))],
        ("circle", {"cx": "-.8", "cy": ".72", "r": ".07", "fill": "$gold"}),
        ("circle", {"cx": ".4", "cy": "-.35", "r": ".06", "fill": "$gold"}),
    ],
    # đèn lồng treo: dây + thân bầu + chóp + tua
    "lantern": [
        ("path", {"d": "M0-1V-.62", "stroke": "$gold", "stroke-width": ".04"}),
        ("rect", {"x": "-.22", "y": "-.64", "width": ".44", "height": ".1", "rx": ".03", "fill": "$gold"}),
        ("ellipse", {"cx": "0", "cy": "-.1", "rx": ".46", "ry": ".46", "fill": "$main"}),
        ("path", {"d": "M0-.55V.35M-.28-.47Q-.46-.1 -.28.28M.28-.47Q.46-.1 .28.28", "fill": "none", "stroke": "$gold", "stroke-width": ".035", "opacity": ".85"}),
        ("rect", {"x": "-.22", "y": ".32", "width": ".44", "height": ".1", "rx": ".03", "fill": "$gold"}),
        ("path", {"d": "M-.1.42L-.14.95M0 .42V1M.1.42L.14.95", "stroke": "$main", "stroke-width": ".05", "stroke-linecap": "round"}),
    ],
    # nhánh lá (nguyệt quế): cuống cong + 5 lá
    "leaf": [
        ("path", {"d": "M-1 .15Q0 -.35 1 .15", "fill": "none", "stroke": "$main", "stroke-width": ".05", "stroke-linecap": "round"}),
        *[("path", {"d": f"M{x} {y}q.12-.28 .3-.3q-.04.26-.3.3z", "fill": "$main", "opacity": ".85"}) for x, y in ((-.8, .05), (-.4, -.1), (0, -.15), (.4, -.1))],
        *[("path", {"d": f"M{x} {y}q.12.28 .3.3q-.04-.26-.3-.3z", "fill": "$main", "opacity": ".7"}) for x, y in ((-.6, .0), (-.2, -.08), (.2, -.08))],
    ],
    # lưới chấm 4x3
    "dots": [("circle", {"cx": f"{-.75 + i * .5:.2f}", "cy": f"{-.5 + j * .5:.2f}", "r": ".07", "fill": "$main", "opacity": ".75"})
             for i in range(4) for j in range(3)],
}
_PETALS = "".join(
    f'<ellipse cx="0" cy="-.55" rx=".32" ry=".5" transform="rotate({a})"/>' for a in range(0, 360, 72)
) + '<circle r=".28" fill="#8B1A1A"/>'


def _motif_svg(name: str, colors: Dict[str, str]) -> str:
    parts = []
    for tag, attrs in _MOTIFS[name]:
        attrs = dict(attrs)
        petals = attrs.pop("_petals", False)
        a = " ".join(f'{k}="{colors.get(v[1:], v) if isinstance(v, str) and v.startswith("$") else v}"' for k, v in attrs.items())
        parts.append(f"<{tag} {a}>{_PETALS if petals else ''}</{tag}>" if tag == "g" else f"<{tag} {a}/>")
    return "".join(parts)


def ornaments_html(pack_name: Optional[str], theme: str, seed_text: str, zones: List[List[float]]) -> str:
    """SVG + JS đặt hoạ tiết của pack sau khi autofit chốt vị trí chữ (móc __tendooAfterFitHooks). Rỗng nếu
    không có pack. `zones` = các vùng chữ [x1,y1,x2,y2] px của template (nơi được phép đặt hoạ tiết góc)."""
    pack = STYLE_PACKS.get(pack_name or "")
    if not pack or not zones:
        return ""
    colors = {"main": theme, "gold": "#E9B949", "dark": "#5B3A1E"}
    rng = random.Random(zlib.crc32(seed_text.encode("utf-8")))
    motifs = {m: _motif_svg(m, colors) for m in pack["ornaments"] if m in _MOTIFS}
    corner_order = ["tl", "tr", "bl", "br"]
    rng.shuffle(corner_order)
    spec = {"ornaments": pack["ornaments"], "motifs": motifs, "corners": corner_order, "zones": zones, "color": theme}
    return f"""<svg class="tk-ornaments" style="position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none;"></svg>
<script>
(window.__tendooAfterFitHooks = window.__tendooAfterFitHooks || []).push(function() {{
  const S = {json.dumps(spec, ensure_ascii=False)};
  const host = document.querySelector('.tk-ornaments');
  if (!host) return;
  const canvas = host.closest('.poster-canvas') || document.body;
  canvas.appendChild(host); host.style.zIndex = 12;
  host.innerHTML = '';
  const c = canvas.getBoundingClientRect(), NS = 'http://www.w3.org/2000/svg';
  // Hộp từng DÒNG chữ đã chốt (cùng cách Cổng 4 đo) + hộp linh kiện đồ hoạ đã đặt.
  const busy = [];
  // + hộp KHỐI đồ hoạ (nhãn, nút, viên thông tin, hộp nội dung): hoạ tiết không đè cả nền của chúng.
  for (const el of document.querySelectorAll('.badge-pill, .badge-capsule, .kicker-tag, .notice-label, .cta-btn, .store-item, .extra-pill, .notice-facts, .step-card, .qr-col, .qr-kiosk'))
    if (el.getBoundingClientRect().width > 0) busy.push(el.getBoundingClientRect());
  for (const el of document.querySelectorAll('[data-autofit], .tk-stamp, .tk-sparkles path')) {{
    if (!el.hasAttribute('data-autofit')) {{ busy.push(el.getBoundingClientRect()); continue; }}
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    while (w.nextNode()) {{
      if (!w.currentNode.textContent.trim()) continue;
      const rg = document.createRange(); rg.selectNodeContents(w.currentNode);
      for (const b of rg.getClientRects()) if (b.width > 1) busy.push(b);
    }}
  }}
  const pad = Math.max(6, c.width * 0.008);
  const free = (x1, y1, x2, y2) => x1 >= 0 && y1 >= 0 && x2 <= c.width && y2 <= c.height &&
    !busy.some(b => x2 + pad > b.left - c.left && x1 - pad < b.right - c.left && y2 + pad > b.top - c.top && y1 - pad < b.bottom - c.top);
  const put = (svg, x1, y1, x2, y2, flip) => {{
    const g = document.createElementNS(NS, 'g');
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, sx = (x2 - x1) / 2, sy = (y2 - y1) / 2;
    g.setAttribute('transform', `translate(${{cx.toFixed(1)}} ${{cy.toFixed(1)}}) scale(${{(flip ? -sx : sx).toFixed(2)}} ${{sy.toFixed(2)}})`);
    g.innerHTML = svg; host.appendChild(g);
    busy.push({{left: x1 + c.left, right: x2 + c.left, top: y1 + c.top, bottom: y2 + c.top}});
  }};
  const line = (x1, y1, x2, y2, wdt, op) => {{
    const p = document.createElementNS(NS, 'path');
    p.setAttribute('d', `M${{x1.toFixed(1)}} ${{y1.toFixed(1)}}L${{x2.toFixed(1)}} ${{y2.toFixed(1)}}`);
    p.setAttribute('stroke', S.color); p.setAttribute('stroke-width', wdt); p.setAttribute('opacity', op);
    p.setAttribute('stroke-linecap', 'round'); p.setAttribute('fill', 'none'); host.appendChild(p);
  }};
  const hero = document.querySelector('.hero-title, .hero-top-title, .quote-hero');
  const label = document.querySelector('.notice-label, .badge-pill, .kicker-tag, .badge-capsule');
  for (const name of S.ornaments) {{
    if (name === 'corner_frame' && hero) {{
      // 4 góc chữ L quanh cụm tiêu đề (hộp hero, đệm 0.25em) -- chỉ khi cả 4 góc đều trống.
      const r = hero.getBoundingClientRect(), fs = parseFloat(getComputedStyle(hero).fontSize) || 40;
      const m = fs * 0.25, L = Math.min(fs * 0.7, r.width * 0.12), t = Math.max(1.5, fs * 0.035);
      const x1 = r.left - c.left - m, y1 = r.top - c.top - m, x2 = r.right - c.left + m, y2 = r.bottom - c.top + m;
      if (x1 < 0 || y1 < 0 || x2 > c.width || y2 > c.height) continue;
      const cs = [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]];
      if (!cs.every(([x, y, dx, dy]) => free(Math.min(x, x + dx * L), Math.min(y, y + dy * L), Math.max(x, x + dx * L), Math.max(y, y + dy * L)))) continue;
      for (const [x, y, dx, dy] of cs) {{ line(x, y, x + dx * L, y, t, .85); line(x, y, x, y + dy * L, t, .85); }}
    }} else if (name === 'flank_lines' && label) {{
      // 2 vạch mảnh hai bên nhãn nhỏ (kiểu thiệp in).
      const r = label.getBoundingClientRect(), h = r.height, L = h * 1.6, g = h * 0.5, y = r.top - c.top + h / 2;
      const a = [r.left - c.left - g - L, y - 1, r.left - c.left - g, y + 1], b = [r.right - c.left + g, y - 1, r.right - c.left + g + L, y + 1];
      if (free(...a) && free(...b)) {{ line(a[0], y, a[2], y, 1.5, .8); line(b[0], y, b[2], y, 1.5, .8); }}
      else if (free(b[0], b[1], b[0] + L * 2, b[3])) line(b[0], y, b[0] + L * 2, y, 1.5, .8);  // nhãn căn trái: "NHÃN ——"
      else if (free(a[2] - L * 2, a[1], a[2], a[3])) line(a[2] - L * 2, y, a[2], y, 1.5, .8);  // nhãn căn phải
    }} else if (S.motifs[name]) {{
      // Hoạ tiết góc: thử các góc của từng vùng chữ theo thứ tự seed; lồng đèn chỉ ở góc TRÊN (treo).
      let done = false;
      // Vùng chữ trước; không vừa -> GÓC POSTER (lề ngoài thường là nền trống, sản phẩm ở giữa).
      const m = Math.min(c.width, c.height) * 0.94;
      const cz = [(c.width - m) / 2, (c.height - m) / 2, (c.width + m) / 2, (c.height + m) / 2];
      for (const z of [...S.zones, cz]) {{
        const zw = z[2] - z[0], zh = z[3] - z[1];
        // Cỡ theo vùng chữ: cành mai / đèn lồng là mảng trang trí thật sự (~1/3 cạnh vùng), lưới chấm nhỏ hơn.
        const k0 = {{dots: 0.16, leaf: 0.34, blossom: 0.36, lantern: 0.26}}[name] || 0.25;
        const s = Math.min(zw, zh) * k0;
        // Vùng quá nhỏ (dải đáy...) -> hoạ tiết thành chấm tí hon, vô nghĩa: bỏ vùng này (không ép).
        if (s < Math.min(c.width, c.height) * 0.07) continue;
        const sy = name === 'lantern' ? s * 1.5 : (name === 'leaf' ? s * 0.5 : (name === 'dots' ? s * 0.75 : s));
        const ins = Math.min(zw, zh) * 0.04;
        for (const k of S.corners) {{
          if (name === 'lantern' && k[0] === 'b') continue;
          const x1 = k[1] === 'l' ? z[0] + ins : z[2] - ins - s, y1 = k[0] === 't' ? z[1] + (name === 'lantern' ? 0 : ins) : z[3] - ins - sy;
          if (free(x1, y1, x1 + s, y1 + sy)) {{ put(S.motifs[name], x1, y1, x1 + s, y1 + sy, k[1] === 'r' && name === 'blossom'); done = true; break; }}
        }}
        if (done) break;
      }}
    }}
  }}
}});
</script>"""


__all__ = ["STYLE_PACKS", "apply_style_pack", "ornaments_html"]
