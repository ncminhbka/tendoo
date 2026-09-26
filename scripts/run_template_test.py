#!/usr/bin/env python3
"""
scripts/run_template_test.py

Unified Test Runner CLI cho 14 commercial templates của Tendoo v3.
Kiểm định tự động toàn diện qua Playwright Headless với 5 Tiêu chuẩn cứng:
1. selfOverflow == 0: Không cắt chữ, không tràn hộp nội bộ.
2. canvasOverflow == False: Không phần tử nào lọt ra ngoài canvas.
3. maskArea <= 50.0%: Tuân thủ quy chuẩn Diffusion corridor in-painting.
4. zeroPlaceholders == True: Không để lại bất kỳ khung rỗng/card trống khi thiếu data.
5. Typography Metrics: Đo kích thước font Hero, Subhead, Store và kiểm tra chống lỗi Tofu.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from tendoo_v3.geometry import get_zones
from tendoo_v3.mask_engine import generate_template_mask, save_mask_preview
from tendoo_v3.mock_backgrounds import create_gradient_backdrop
from tendoo_v3.renderer import pil_to_base64_data_uri, render_plan_to_poster
from tendoo_v3.schema import StyleConfig, TendooCreativePlan
from tendoo_v3.styles import LIGHT_BACKGROUND_TONES, TONE_BACKDROP_COLORS

TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_BASE_DIR = PROJECT_ROOT / "output_tendoo_v3"

ALL_TEMPLATES = [
    "sandwich_top_heavy",
    "sandwich_bottom_heavy",
    "split_left",
    "split_right",
    "diagonal_slash",
    "l_frame_showcase",
    "step_process_roadmap",
    "recruitment_board",
    "menu_price_board",
    "luxury_centered_card",
    "lifestyle_corner_pod",
    "customer_feedback_card",
    "before_after_split",
    "grand_opening_banner",
]


def generate_mock_backdrop_data_uri(
    width: int, height: int, theme_hex: str = "#FF3366", background_tone: str = "dark_luxury"
) -> str:
    """Tạo gradient backdrop cho poster, THEO TÔNG NỀN của plan.

    Trước đây luôn nền tối -> poster tông sáng (pastel/light_clean) chưa từng được test trên nền
    sáng: palette chọn chữ đậm cho nền sáng rồi đặt lên nền tối (đo 26/09: giá menu đỏ đậm trên
    nền gần đen, WCAG 1.78:1). Tông tối giữ NGUYÊN thuật toán cũ (ảnh không đổi)."""
    r = int(theme_hex[1:3], 16) if len(theme_hex) >= 7 else 255
    g = int(theme_hex[3:5], 16) if len(theme_hex) >= 7 else 51
    b = int(theme_hex[5:7], 16) if len(theme_hex) >= 7 else 102

    cx = int(width * 0.65)
    cy = int(height * 0.45)
    radius = int(min(width, height) * 0.55)

    if background_tone in LIGHT_BACKGROUND_TONES:
        top, bottom = TONE_BACKDROP_COLORS[background_tone]
        img = Image.new("RGB", (width, height), color=top)
        draw = ImageDraw.Draw(img)
        for y in range(0, height, 8):
            t = y / max(1, height - 1)
            draw.rectangle([0, y, width, y + 8], fill=tuple(int(top[k] * (1 - t) + bottom[k] * t) for k in range(3)))
        for i in range(radius, 0, -20):
            a = 0.25 * (1.0 - i / radius)  # quầng màu theme nhạt dần, trộn tuyến tính trên nền sáng
            col = tuple(int(bottom[k] * (1 - a) + (r, g, b)[k] * a) for k in range(3))
            draw.ellipse([cx - i, cy - i, cx + i, cy + i], fill=col)
    else:
        img = Image.new("RGB", (width, height), color=(10, 15, 26))
        draw = ImageDraw.Draw(img)
        for i in range(radius, 0, -20):
            alpha = int(50 * (1.0 - i / radius))
            col = (
                min(255, 10 + int(r * alpha / 255)),
                min(255, 15 + int(g * alpha / 255)),
                min(255, 26 + int(b * alpha / 255)),
            )
            draw.ellipse([cx - i, cy - i, cx + i, cy + i], fill=col)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"



def generate_harsh_backdrop_data_uri(
    width: int, height: int, theme_hex: str = "#FF3366", background_tone: str = "dark_luxury", seed: str = ""
) -> str:
    """Nền giả KHẮC NGHIỆT cho GĐ 4 (ROADMAP §4.3): nền theo tông như trên + các vệt sáng cục bộ
    (nền tối) hoặc mảng tối cục bộ (nền sáng) -- vừa mờ mềm (đèn, nắng) vừa cạnh sắc (ô cửa sổ,
    khung). Mô phỏng ảnh FLUX thật: trung bình vùng "tối" nhưng có vệt chói ngay dưới nét chữ.
    Tất định theo `seed` (mã case)."""
    import random
    import zlib

    from PIL import ImageFilter

    base = Image.open(io.BytesIO(base64.b64decode(generate_mock_backdrop_data_uri(width, height, theme_hex, background_tone).split(",", 1)[1]))).convert("RGB")
    rng = random.Random(zlib.crc32(f"{seed}|{width}x{height}".encode("utf-8")))
    light_tone = background_tone in LIGHT_BACKGROUND_TONES
    short = min(width, height)
    soft = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(soft)
    for _ in range(rng.randint(4, 7)):  # đốm mềm
        rx, ry = short * rng.uniform(0.06, 0.2), short * rng.uniform(0.04, 0.14)
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=rng.randint(150, 255))
    soft = soft.filter(ImageFilter.GaussianBlur(short * 0.03))
    hard = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(hard)
    for _ in range(rng.randint(2, 4)):  # khối cạnh sắc
        w_, h_ = short * rng.uniform(0.08, 0.25), short * rng.uniform(0.03, 0.12)
        x, y = rng.uniform(0, width - w_), rng.uniform(0, height - h_)
        d.rectangle([x, y, x + w_, y + h_], fill=rng.randint(170, 240))
    mask = Image.fromarray(np.maximum(np.asarray(soft), np.asarray(hard)))
    spot = (22, 24, 32) if light_tone else (255, 244, 220)  # mảng tối trên nền sáng / vệt đèn ấm trên nền tối
    img = Image.composite(Image.new("RGB", (width, height), spot), base, mask)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def generate_lowdetail_backdrop_data_uri(
    width: int, height: int, theme_hex: str = "#FF3366", background_tone: str = "dark_luxury", seed: str = ""
) -> str:
    """Nền giả MASKLESS (GĐ 5, ROADMAP §5.3): nền ÍT CHI TIẾT đúng loại prompt maskless cho phép
    (gradient + bokeh mềm + bụi sáng li ti) phủ CẢ khung -- không có vùng tĩnh nào dưới chữ như khi
    có corridor. Tất định theo `seed`."""
    import random
    import zlib

    from PIL import ImageFilter

    base = Image.open(io.BytesIO(base64.b64decode(generate_mock_backdrop_data_uri(width, height, theme_hex, background_tone).split(",", 1)[1]))).convert("RGB")
    rng = random.Random(zlib.crc32(f"lowdetail|{seed}|{width}x{height}".encode("utf-8")))
    light_tone = background_tone in LIGHT_BACKGROUND_TONES
    short = min(width, height)
    th = tuple(int(theme_hex[i:i + 2], 16) for i in (1, 3, 5)) if len(theme_hex) >= 7 else (255, 200, 120)
    bokeh = Image.new("RGB", (width, height), (0, 0, 0))
    amask = Image.new("L", (width, height), 0)
    db, dm = ImageDraw.Draw(bokeh), ImageDraw.Draw(amask)
    for _ in range(rng.randint(14, 22)):  # đốm bokeh mềm khắp khung
        r = short * rng.uniform(0.03, 0.12)
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        col = th if rng.random() < 0.5 else ((255, 236, 200) if not light_tone else (255, 255, 255))
        db.ellipse([x - r, y - r, x + r, y + r], fill=col)
        dm.ellipse([x - r, y - r, x + r, y + r], fill=rng.randint(60, 140))
    for _ in range(int(width * height / 2500)):  # bụi sáng li ti
        x, y, r = rng.uniform(0, width), rng.uniform(0, height), rng.uniform(0.6, 1.8)
        db.ellipse([x - r, y - r, x + r, y + r], fill=(255, 245, 220))
        dm.ellipse([x - r, y - r, x + r, y + r], fill=rng.randint(80, 200))
    amask = amask.filter(ImageFilter.GaussianBlur(short * 0.012))
    img = Image.composite(bokeh, base, amask)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

def parse_case_to_plan(case: Dict[str, Any], default_template: str) -> Tuple[TendooCreativePlan, int, int]:
    """Phân tích case thành TendooCreativePlan bất kể schema nested hay flat."""
    width = case.get("width", 1024)
    height = case.get("height", 1024)
    orient = case.get("orientation", "left")

    src = case["plan"] if ("plan" in case and isinstance(case["plan"], dict)) else case

    style_raw = src.get("style", {})
    if isinstance(style_raw, StyleConfig):
        style_obj = style_raw
    elif isinstance(style_raw, dict):
        style_obj = StyleConfig(
            font=style_raw.get("font", "bevietnam"),
            theme_color=style_raw.get("theme_color", "#FFD700"),
            text_effect=style_raw.get("text_effect", "bold_clean"),
            background_tone=style_raw.get("background_tone", "dark_luxury"),
        )
    else:
        style_obj = StyleConfig(font="bevietnam", theme_color="#FFD700")

    tpl = src.get("template", default_template)
    
    plan = TendooCreativePlan(
        template=tpl,
        orientation=src.get("orientation", orient),
        hero=src.get("hero", ""),
        subhead=src.get("subhead"),
        badge=src.get("badge"),
        tag_left=src.get("tag_left"),
        tag_right=src.get("tag_right"),
        rating=src.get("rating"),
        extra_texts=src.get("extra_texts", []),
        cta=src.get("cta"),
        store_info=src.get("store_info"),
        qr_code=src.get("qr_code"),
        qr_label=src.get("qr_label"),
        testimonial=src.get("testimonial"),
        reviewer_name=src.get("reviewer_name"),
        steps=src.get("steps", []),
        style=style_obj,
        scene_prompt=src.get("scene_prompt", ""),
        corridor_prompt=src.get("corridor_prompt", ""),
        visual_intent=src.get("visual_intent"),
        badge_style=src.get("badge_style"),
        stat_style=src.get("stat_style"),
        decor=src.get("decor"),
    )
    if src.get("hero_parts"):
        # Qua đúng chốt nguyên văn (Cổng 1) của đường LLM thật -- lệch chữ thì markup bị vứt.
        plan.hero_parts = TendooCreativePlan.from_dict({"hero": plan.hero, "hero_parts": src["hero_parts"]}).hero_parts
    return plan, width, height


def inspect_page_dom(page, width: int, height: int, has_store: bool, has_qr: bool) -> Dict[str, Any]:
    """Kiểm tra toàn diện DOM Playwright theo 5 tiêu chuẩn."""
    js = r"""
    ([width, height, hasStore, hasQr]) => {
      const out = {
        leafElements: [],
        selfOverflowCount: 0,
        canvasOverflow: false,
        placeholderViolations: [],
        tofuViolations: [],
        heroFontSize: 0,
        subheadFontSize: 0,
        storeFontSize: 0,
        badgeFontSize: 0
      };

      // 1. Quét tất cả phần tử lá chứa text
      const allEls = document.querySelectorAll('body *');
      for (const el of allEls) {
        // Chỉ xét leaf text node
        if (el.children.length > 0) continue;
        const text = (el.textContent || '').trim();
        if (!text) continue;

        const cs = getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity || '1') === 0) continue;

        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) continue;

        const fontSize = parseFloat(cs.fontSize) || 0;
        const cls = (el.className && el.className.toString) ? el.className.toString() : '';
        const parentCls = (el.parentElement && el.parentElement.className) ? el.parentElement.className.toString() : '';

        // Thu thập font size chính
        if (cls.includes('hero') || parentCls.includes('hero')) {
          out.heroFontSize = Math.max(out.heroFontSize, fontSize);
        } else if (cls.includes('subhead') || parentCls.includes('subhead')) {
          out.subheadFontSize = Math.max(out.subheadFontSize, fontSize);
        } else if (cls.includes('store') || parentCls.includes('store')) {
          out.storeFontSize = Math.max(out.storeFontSize, fontSize);
        } else if (cls.includes('badge') || parentCls.includes('badge')) {
          out.badgeFontSize = Math.max(out.badgeFontSize, fontSize);
        }

        // Tofu check: Nếu có ký tự trang trí mà font-family không có Arial / Segoe UI Symbol
        if (/[\u2726\u2605\u25CF\u25C6\u2714\u25B6]/.test(text)) {
          const ff = cs.fontFamily.toLowerCase();
          if (!ff.includes('arial') && !ff.includes('segoe ui symbol') && !ff.includes('sans-serif')) {
            out.tofuViolations.push({ text, font: cs.fontFamily, cls });
          }
        }

        // Kiểm tra tràn hộp (Self-Overflow)
        const isSelfOverflowX = el.scrollWidth > el.clientWidth + 2.0;
        const isSelfOverflowY = (cs.overflow === 'hidden' || cs.overflowY === 'hidden')
          ? (el.scrollHeight > el.clientHeight + 4.0)
          : (el.dataset.maxHeight ? el.scrollHeight > parseFloat(el.dataset.maxHeight) + 1.5 : el.scrollHeight > el.clientHeight + 14.0);
        if (isSelfOverflowX || isSelfOverflowY) {
          out.selfOverflowCount++;
        }

        // Kiểm tra tràn ngoài canvas
        if (r.left < -2.0 || r.top < -2.0 || r.right > width + 2.0 || r.bottom > height + 2.0) {
          out.canvasOverflow = true;
        }

        out.leafElements.push({
          tag: el.tagName,
          cls,
          parentCls,
          text: text.slice(0, 40),
          fontSize,
          selfOverflowX: isSelfOverflowX,
          selfOverflowY: isSelfOverflowY
        });
      }

      // 2. Kiểm tra Zero-Trace / No-Placeholder
      // Khi hasQr == false, kiểm tra xem có container QR nào rỗng còn hiển thị không
      if (!hasQr) {
        const qrSelectors = ['.qr-card-pod', '.top-qr-col', '.bottom-qr-card', '.qr-col', '.qr-box'];
        for (const sel of qrSelectors) {
          const matched = document.querySelectorAll(sel);
          for (const m of matched) {
            const r = m.getBoundingClientRect();
            const cs = getComputedStyle(m);
            const hasBg = cs.backgroundColor !== 'rgba(0, 0, 0, 0)' && cs.backgroundColor !== 'transparent';
            const hasBorder = parseFloat(cs.borderWidth || '0') > 0;
            if ((r.width > 2 && r.height > 2) && (hasBg || hasBorder || m.children.length > 0)) {
              out.placeholderViolations.push({ type: 'empty_qr_card', selector: sel, rect: { w: r.width, h: r.height } });
            }
          }
        }
      }

      // Khi hasStore == false, kiểm tra xem có container store nào rỗng còn hiển thị không
      if (!hasStore) {
        const storeSelectors = ['.store-brand-card', '.store-info-row', '.store-info-col', '.store-details-row'];
        for (const sel of storeSelectors) {
          const matched = document.querySelectorAll(sel);
          for (const m of matched) {
            const r = m.getBoundingClientRect();
            const cs = getComputedStyle(m);
            const text = (m.textContent || '').trim();
            const hasBg = cs.backgroundColor !== 'rgba(0, 0, 0, 0)' && cs.backgroundColor !== 'transparent';
            const hasBorder = parseFloat(cs.borderWidth || '0') > 0;
            if ((r.width > 2 && r.height > 2) && (hasBg || hasBorder || text.length > 0)) {
              out.placeholderViolations.push({ type: 'empty_store_card', selector: sel, rect: { w: r.width, h: r.height }, text });
            }
          }
        }
      }

      return out;
    }
    """
    return page.evaluate(js, [width, height, has_store, has_qr])


def build_gallery_html(template_name: str, results: List[Dict[str, Any]]) -> str:
    """Tạo file HTML Gallery trực quan báo cáo kết quả."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    cards_html = []
    for r in results:
        cid = r["id"]
        title = r["title"]
        w, h = r["width"], r["height"]
        hero = r["hero"]
        hero_font = r.get("hero_font", 0)
        overflow_count = r["overflow_count"]
        canvas_overflow = r["canvas_overflow"]
        mask_area = r["mask_area"]
        placeholders = r["placeholder_violations"]
        passed_case = r["passed"]

        status_text = "PASS (0 lỗi)" if passed_case else f"FAIL ({overflow_count} tràn, {len(placeholders)} placeholder)"
        status_color = "#10B981" if passed_case else "#EF4444"
        ratio_lbl = "1:1" if w == h else ("9:16" if w < 600 else ("16:9" if w > 1000 else "4:5"))

        cards_html.append(f"""
        <div class="case-card {'failed' if not passed_case else ''}">
          <div class="card-header">
            <span class="case-badge">{cid}</span>
            <span class="ratio-badge">{ratio_lbl} ({w}x{h})</span>
          </div>
          <h3 class="case-title">{title}</h3>
          <p class="case-hero">Hero: <b>"{hero}"</b></p>
          <div class="meta-row">
            <span>Hero Font: <b style="color:#FFD700;">{hero_font:.1f}px</b></span>
            <span>Mask: <b>{mask_area:.1f}%</b></span>
            <span>Trạng thái: <b style="color:{status_color};">{status_text}</b></span>
          </div>
          <div class="media-preview">
            <div class="preview-col">
              <span class="preview-label">Final Poster</span>
              <a href="{cid}/poster.png" target="_blank">
                <img src="{cid}/poster.png" alt="Poster" class="img-poster" loading="lazy">
              </a>
            </div>
            <div class="preview-col">
              <span class="preview-label">DiT Mask ({mask_area:.1f}%)</span>
              <a href="{cid}/mask.png" target="_blank">
                <img src="{cid}/mask.png" alt="Mask" class="img-mask" loading="lazy">
              </a>
            </div>
          </div>
          <div class="card-footer">
            <a href="{cid}/poster.html" target="_blank" class="btn-link">Mở File HTML</a>
          </div>
        </div>
        """)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Gallery Kiểm Thử: {template_name.upper()}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0B0F19;
      color: #E2E8F0;
      padding: 32px 24px;
    }}
    .header {{
      text-align: center;
      margin-bottom: 32px;
    }}
    .header h1 {{
      font-size: 26px;
      font-weight: 800;
      color: #FFFFFF;
      margin-bottom: 8px;
    }}
    .stats-bar {{
      display: inline-flex;
      gap: 16px;
      background: rgba(255, 255, 255, 0.05);
      padding: 8px 24px;
      border-radius: 9999px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      margin-top: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 24px;
      max-width: 1600px;
      margin: 0 auto;
    }}
    .case-card {{
      background: #151D2F;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .case-card.failed {{
      border-color: #EF4444;
      background: #1C1523;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .case-badge {{
      font-size: 11px;
      font-weight: 700;
      background: rgba(59, 130, 246, 0.2);
      color: #60A5FA;
      padding: 3px 8px;
      border-radius: 6px;
    }}
    .ratio-badge {{
      font-size: 11px;
      background: rgba(255, 255, 255, 0.1);
      color: #CBD5E1;
      padding: 3px 8px;
      border-radius: 6px;
    }}
    .case-title {{
      font-size: 14px;
      font-weight: 600;
      color: #F8FAFC;
      line-height: 1.35;
    }}
    .case-hero {{
      font-size: 12px;
      color: #94A3B8;
    }}
    .meta-row {{
      display: flex;
      justify-content: space-between;
      font-size: 11.5px;
      background: rgba(0, 0, 0, 0.25);
      padding: 6px 10px;
      border-radius: 6px;
    }}
    .media-preview {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .preview-col {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .preview-label {{
      font-size: 10px;
      color: #64748B;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .img-poster, .img-mask {{
      width: 100%;
      height: 190px;
      object-fit: contain;
      background: #070B14;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .card-footer {{
      display: flex;
      justify-content: flex-end;
    }}
    .btn-link {{
      font-size: 11px;
      color: #38BDF8;
      text-decoration: none;
      background: rgba(56, 189, 248, 0.1);
      padding: 4px 10px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Gallery Kiểm Thử: {template_name.upper()}</h1>
    <div class="stats-bar">
      <span>Tổng số: <b>{total}</b></span>
      <span>Đạt chuẩn: <b style="color:#10B981;">{passed}</b></span>
      <span>Lỗi: <b style="color:{'#EF4444' if failed > 0 else '#94A3B8'};">{failed}</b></span>
      <span>Tỉ lệ đạt: <b style="color:{'#10B981' if pass_rate == 100 else '#F59E0B'};">{pass_rate:.1f}%</b></span>
    </div>
  </div>
  <div class="grid">
    {"".join(cards_html)}
  </div>
</body>
</html>"""


def run_test_suite_for_template(
    template_name: str,
    suite_path: Path,
    browser,
    gallery_dir: Path,
    case_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Chạy toàn bộ các ca trong một suite file cho 1 template."""
    with open(suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if case_filter:
        cases = [c for c in cases if c["id"] == case_filter]
        if not cases:
            return []

    gallery_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"\n--- Chạy Suite: {suite_path.name} ({len(cases)} cases) cho {template_name.upper()} ---")

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        title = case["title"]
        plan, w, h = parse_case_to_plan(case, default_template=template_name)

        case_dir = gallery_dir / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        output_png = case_dir / "poster.png"
        output_html = case_dir / "poster.html"
        output_mask = case_dir / "mask.png"

        t0 = time.perf_counter()

        # 1. Background mock
        bg_data_uri = generate_mock_backdrop_data_uri(w, h, theme_hex=plan.style.theme_color, background_tone=plan.style.background_tone)

        # 2. Render HTML & PNG
        _, html_content = render_plan_to_poster(
            plan=plan,
            bg_data_uri=bg_data_uri,
            output_image_path=output_png,
            width=w,
            height=h,
        )
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 3. Mask calculation
        from tendoo_v3.renderer import compute_geometry_flags, compute_plan_content_density
        density = compute_plan_content_density(plan)
        mask = generate_template_mask(
            template=plan.template,
            width=w,
            height=h,
            orientation=plan.orientation,
            density=density,
            **compute_geometry_flags(plan),
        )
        mask_area_pct = float(np.sum(mask > 0.5) / (w * h) * 100.0)
        save_mask_preview(mask, output_mask)

        # 4. Playwright DOM Inspection
        page = browser.new_page(viewport={"width": w, "height": h})
        page.set_content(html_content, wait_until="networkidle")

        has_store = bool(plan.store_info)
        has_qr = bool(plan.qr_code)
        dom_metrics = inspect_page_dom(page, w, h, has_store=has_store, has_qr=has_qr)
        page.close()

        dur = time.perf_counter() - t0

        self_overflow = dom_metrics["selfOverflowCount"]
        canvas_overflow = dom_metrics["canvasOverflow"]
        placeholder_violations = dom_metrics["placeholderViolations"]
        tofu_violations = dom_metrics["tofuViolations"]

        passed = (
            self_overflow == 0
            and not canvas_overflow
            and mask_area_pct <= 50.0
            and len(placeholder_violations) == 0
            and len(tofu_violations) == 0
        )

        status_str = "PASS" if passed else "FAIL"
        err_details = []
        if self_overflow > 0:
            overflowing_els = [el for el in dom_metrics["leafElements"] if el["selfOverflowX"] or el["selfOverflowY"]]
            el_summaries = [f"{el['cls'] or el['tag']} ('{el['text'][:20]}...')" for el in overflowing_els]
            err_details.append(f"{self_overflow} self-overflow: [{'; '.join(el_summaries)}]")
        if canvas_overflow:
            err_details.append("canvas-overflow")
        if mask_area_pct > 50.0:
            err_details.append(f"mask {mask_area_pct:.1f}% > 50%")
        if len(placeholder_violations) > 0:
            err_details.append(f"{len(placeholder_violations)} placeholder leaks")
        if len(tofu_violations) > 0:
            err_details.append(f"{len(tofu_violations)} tofu icons")

        err_msg = f" ({', '.join(err_details)})" if err_details else ""
        print(f"  [{idx:02d}/{len(cases)}] {status_str}: {cid} ({w}x{h}) | Hero: {dom_metrics['heroFontSize']:.1f}px | Mask: {mask_area_pct:.1f}%{err_msg} [{dur:.2f}s]")

        results.append({
            "id": cid,
            "title": title,
            "template": template_name,
            "width": w,
            "height": h,
            "hero": plan.hero,
            "hero_font": dom_metrics["heroFontSize"],
            "subhead_font": dom_metrics["subheadFontSize"],
            "store_font": dom_metrics["storeFontSize"],
            "overflow_count": self_overflow,
            "canvas_overflow": canvas_overflow,
            "mask_area": mask_area_pct,
            "placeholder_violations": placeholder_violations,
            "tofu_violations": tofu_violations,
            "passed": passed,
            "duration_s": dur,
        })

    return results


def find_suites_for_template(template_name: str, include_sparse: bool = True) -> List[Path]:
    """Tìm tất cả các file suite JSON liên quan đến template."""
    suites = []
    
    # 1. Standard suite
    std_file = TESTS_DIR / f"test_{template_name}_suite.json"
    if std_file.exists():
        suites.append(std_file)
        
    if include_sparse:
        # 2. Sparse / no-qr suites
        candidates = [
            TESTS_DIR / f"test_{template_name}_sparse_suite.json",
            TESTS_DIR / f"test_{template_name}_no_qr_suite.json",
            TESTS_DIR / f"test_{template_name}_no_content_suite.json",
            TESTS_DIR / f"test_{template_name}_no_footer_suite.json",
        ]
        for c in candidates:
            if c.exists() and c not in suites:
                suites.append(c)
                
    return suites


def main():
    parser = argparse.ArgumentParser(description="Unified Test Runner CLI for Tendoo v3 Templates")
    parser.add_argument("--template", type=str, choices=ALL_TEMPLATES + ["all"], default=None, help="Template cần test")
    parser.add_argument("--suite", type=str, default=None, help="Đường dẫn file suite JSON tùy chỉnh")
    parser.add_argument("--sparse", action="store_true", help="Chỉ chạy sparse/no-data suite")
    parser.add_argument("--all-suites", action="store_true", help="Chạy cả Standard và Sparse suites")
    parser.add_argument("--case", type=str, default=None, help="Chỉ chạy 1 case id cụ thể")
    parser.add_argument("--all", action="store_true", help="Chạy kiểm thử cho TOÀN BỘ 14 templates")
    parser.add_argument("--baseline", action="store_true", help="Chạy chế độ baseline benchmark (lưu kết quả phân tích)")
    parser.add_argument("--report-json", type=str, default=None, help="File JSON lưu báo cáo tổng hợp")
    args = parser.parse_args()

    target_templates = []
    if args.all or args.template == "all":
        target_templates = ALL_TEMPLATES
    elif args.template:
        target_templates = [args.template]
    else:
        print("[!] Không chỉ định template. Dùng --template <name> hoặc --all.")
        sys.exit(1)

    print("=" * 80)
    print("TENDOO V3 - UNIFIED TEMPLATE TEST RUNNER CLI")
    print(f"Templates mục tiêu: {', '.join(target_templates)}")
    print(f"Chế độ: {'Baseline Benchmark' if args.baseline else 'Standard Validation'}")
    print("=" * 80)

    all_results = {}
    total_passed = 0
    total_cases = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for tpl in target_templates:
            gallery_dir = OUTPUT_BASE_DIR / f"{tpl}_gallery"
            suites_to_run = []

            if args.suite:
                suites_to_run = [Path(args.suite)]
            elif args.sparse:
                suites_to_run = [s for s in find_suites_for_template(tpl, include_sparse=True) if "suite.json" in s.name and f"test_{tpl}_suite.json" != s.name]
            elif args.all_suites or args.baseline:
                suites_to_run = find_suites_for_template(tpl, include_sparse=True)
            else:
                suites_to_run = find_suites_for_template(tpl, include_sparse=False)

            if not suites_to_run:
                print(f"[-] Không tìm thấy test suite nào cho {tpl}!")
                continue

            tpl_results = []
            for suite_path in suites_to_run:
                res = run_test_suite_for_template(tpl, suite_path, browser, gallery_dir, case_filter=args.case)
                tpl_results.extend(res)

            # Xuất Gallery HTML cho template
            gallery_html = build_gallery_html(tpl, tpl_results)
            gallery_file = gallery_dir / "gallery.html"
            with open(gallery_file, "w", encoding="utf-8") as f:
                f.write(gallery_html)
            print(f"\n[+] Xuất Gallery: {gallery_file}")

            tpl_passed = sum(1 for r in tpl_results if r["passed"])
            tpl_total = len(tpl_results)
            print(f"[SUMMARY {tpl.upper()}]: {tpl_passed}/{tpl_total} PASS ({tpl_passed/tpl_total*100:.1f}%)")

            all_results[tpl] = tpl_results
            total_passed += tpl_passed
            total_cases += tpl_total

    print("\n" + "=" * 80)
    print(f"TỔNG KẾT TOÀN DIỆN: {total_passed}/{total_cases} CASES PASS ({(total_passed/total_cases*100) if total_cases > 0 else 0:.1f}%)")
    print("=" * 80)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[+] Báo cáo đã lưu tại: {report_path}")

    # Nếu có lỗi và không phải baseline mode, trả về exit code 1
    if not args.baseline and total_passed < total_cases:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
