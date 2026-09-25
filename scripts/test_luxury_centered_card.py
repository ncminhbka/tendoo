import sys
import io
import json
import time
import base64
from pathlib import Path
from typing import Dict, Any, List

# Đảm bảo mã nguồn tiếng Việt in ra màn hình Windows không bị lỗi cp1252
sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục src vào Python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

from tendoo_v3.schema import TendooCreativePlan, StyleConfig
from tendoo_v3.renderer import render_plan_to_poster


def generate_mock_background(width: int, height: int, orientation: str = "center", theme_hex: str = "#D4AF37") -> str:
    """Tạo mock background mờ tối điện ảnh với điểm nhấn ánh sáng chủ thể.
    - Nếu orientation là 'left': Chủ thể sáng ở bên phải (x ~ 72%).
    - Nếu orientation là 'right': Chủ thể sáng ở bên trái (x ~ 28%).
    - Nếu orientation là 'center': Chủ thể tỏa sáng đối xứng huyền ảo xung quanh trung tâm.
    """
    img = Image.new("RGB", (width, height), (8, 11, 18))
    draw = ImageDraw.Draw(img)

    r = int(theme_hex[1:3], 16)
    g = int(theme_hex[3:5], 16)
    b = int(theme_hex[5:7], 16)

    if orientation in ("right", "top_right", "bottom_right"):
        center_x = int(width * 0.25)
        center_y = int(height * 0.50)
    elif orientation in ("left", "top_left", "bottom_left"):
        center_x = int(width * 0.75)
        center_y = int(height * 0.50)
    else:
        center_x = int(width * 0.50)
        center_y = int(height * 0.50)

    radius = int(min(width, height) * 0.46)

    for i in range(radius, 0, -16):
        alpha = int(45 * (1.0 - i / radius))
        glow_col = (
            min(255, 8 + int(r * alpha / 255)),
            min(255, 11 + int(g * alpha / 255)),
            min(255, 18 + int(b * alpha / 255)),
        )
        draw.ellipse([center_x - i, center_y - i, center_x + i, center_y + i], fill=glow_col)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"


def inspect_page_metrics(page, width: int, height: int) -> List[Dict[str, Any]]:
    """Đo lường chính xác các phần tử DOM trên trang bằng Chromium."""
    js = f"""
    () => {{
      const out = [];
      const width = {width};
      const height = {height};
      const els = document.querySelectorAll(
        '.centered-glass-card, .header-cluster, .badge-capsule, .rating-stars, .hero-title, .subhead-title, .message-container, .message-line, .footer-action, .footer-left, .cta-btn, .store-item, .qr-col'
      );
      for (const el of els) {{
        const r = el.getBoundingClientRect();
        const cs = window.getComputedStyle(el);
        const text = (el.innerText || '').trim();
        if (r.width <= 0 || r.height <= 0) continue;

        let selfOverX = false;
        let selfOverY = false;
        if (el.dataset && el.dataset.maxWidth) {{
          const maxW = parseFloat(el.dataset.maxWidth);
          if (el.scrollWidth > maxW + 4.0) selfOverX = true;
        }} else {{
          if (el.scrollWidth > el.clientWidth + 4.0) selfOverX = true;
        }}

        if (el.dataset && el.dataset.maxHeight) {{
          const maxH = parseFloat(el.dataset.maxHeight);
          if (el.scrollHeight > maxH + 4.0) selfOverY = true;
        }} else {{
          if (el.scrollHeight > el.clientHeight + 4.0) selfOverY = true;
        }}

        // Kiểm tra tràn khỏi Canvas poster (width x height)
        let canvasOver = false;
        if (r.left < -2 || r.top < -2 || r.right > width + 2 || r.bottom > height + 2) {{
          canvasOver = true;
        }}

        out.push({{
          tag: el.tagName,
          className: el.className,
          text: text.slice(0, 35),
          left: Math.round(r.left),
          top: Math.round(r.top),
          width: Math.round(r.width),
          height: Math.round(r.height),
          fontSize: cs.fontSize,
          selfOverX,
          selfOverY,
          canvasOver
        }});
      }}
      return out;
    }}
    """
    return page.evaluate(js)


def generate_gallery_html(results: List[Dict[str, Any]], title_name: str) -> str:
    cards = []
    for r in results:
        cid = r["id"]
        title = r["title"]
        w = r["width"]
        h = r["height"]
        aspect = f"{w}:{h}"
        orient = r.get("orientation", "center")
        hero_font = r.get("hero_font", "N/A")
        self_over = r["self_over"]
        canvas_over = r["canvas_over"]

        status_class = "pass" if (self_over == 0 and canvas_over == 0) else "fail"
        status_text = "PASS" if status_class == "pass" else f"FAIL ({self_over} self, {canvas_over} canvas)"

        cards.append(f"""
        <div class="case-card {status_class}">
          <div class="card-header">
            <span class="card-badge {status_class}">{status_text}</span>
            <div class="card-meta">
              <span class="ratio-pill">{aspect}</span>
              <span class="orient-pill">Orient: {orient}</span>
              <span class="time-pill">{r['duration']:.2f}s</span>
            </div>
          </div>
          <div class="card-body">
            <div class="poster-preview">
              <a href="{cid}/poster.html" target="_blank">
                <img src="{cid}/poster.png" alt="{cid}">
              </a>
            </div>
            <div class="card-details">
              <h3 class="case-title">{title}</h3>
              <div class="case-id"><code>{cid}</code></div>
              <div class="metrics-list">
                <div class="metric-row">
                  <span class="m-label">Kích thước:</span>
                  <span class="m-val">{w} x {h} px</span>
                </div>
                <div class="metric-row">
                  <span class="m-label">Hero Font:</span>
                  <span class="m-val font-highlight">{hero_font}</span>
                </div>
                <div class="metric-row">
                  <span class="m-label">Lỗi Tự Tràn:</span>
                  <span class="m-val {'err' if self_over > 0 else 'ok'}">{self_over}</span>
                </div>
                <div class="metric-row">
                  <span class="m-label">Lỗi Tràn Canvas:</span>
                  <span class="m-val {'err' if canvas_over > 0 else 'ok'}">{canvas_over}</span>
                </div>
              </div>
              <div class="action-links">
                <a href="{cid}/poster.html" target="_blank" class="btn-link">Mở HTML</a>
                <a href="{cid}/poster.png" target="_blank" class="btn-link">Xem Ảnh</a>
              </div>
            </div>
          </div>
        </div>
        """)

    cards_joined = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Gallery Kiểm Thử: {title_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0B0F19;
      color: #E2E8F0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 32px 24px;
    }}
    .header {{
      max-width: 1600px;
      margin: 0 auto 32px;
      padding-bottom: 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .header h1 {{
      font-size: 26px;
      font-weight: 800;
      color: #F8FAFC;
      margin-bottom: 8px;
    }}
    .header p {{
      color: #94A3B8;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
      gap: 24px;
      max-width: 1600px;
      margin: 0 auto;
    }}
    .case-card {{
      background: #151C2C;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: #1A2337;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .card-badge {{
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
      color: #fff;
    }}
    .card-badge.pass {{ background: #10B981; }}
    .card-badge.fail {{ background: #EF4444; }}
    .card-meta {{
      display: flex;
      gap: 8px;
    }}
    .ratio-pill, .orient-pill, .time-pill {{
      font-size: 11.5px;
      background: rgba(255, 255, 255, 0.08);
      padding: 3px 8px;
      border-radius: 6px;
      color: #CBD5E1;
    }}
    .card-body {{
      display: flex;
      padding: 16px;
      gap: 16px;
    }}
    .poster-preview {{
      flex: 0 0 200px;
      height: 240px;
      background: #0D131F;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .poster-preview img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      transition: transform 0.2s;
    }}
    .poster-preview img:hover {{
      transform: scale(1.04);
    }}
    .card-details {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .case-title {{
      font-size: 15px;
      font-weight: 700;
      color: #F1F5F9;
      line-height: 1.35;
    }}
    .case-id code {{
      font-size: 11.5px;
      color: #38BDF8;
      background: rgba(56, 189, 248, 0.1);
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .metrics-list {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12.5px;
      margin-top: 4px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      border-bottom: 1px dashed rgba(255, 255, 255, 0.06);
      padding-bottom: 3px;
    }}
    .m-label {{ color: #94A3B8; }}
    .m-val {{ font-weight: 600; color: #E2E8F0; }}
    .m-val.ok {{ color: #10B981; }}
    .m-val.err {{ color: #EF4444; font-weight: 800; }}
    .font-highlight {{ color: #F59E0B; }}
    .action-links {{
      margin-top: auto;
      display: flex;
      gap: 8px;
      padding-top: 10px;
    }}
    .btn-link {{
      font-size: 12px;
      padding: 6px 12px;
      border-radius: 6px;
      text-decoration: none;
      background: #1E293B;
      color: #38BDF8;
      border: 1px solid rgba(56, 189, 248, 0.3);
      font-weight: 600;
      transition: all 0.2s;
    }}
    .btn-link:hover {{
      background: #0284C7;
      color: #fff;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Báo Cáo Kiểm Thử Tự Động: {title_name}</h1>
    <p>Tổng cộng {len(results)} kịch bản kiểm thử (4 tỉ lệ x 3 hướng Mirror/Center x Đa dạng phong cách xa xỉ)</p>
  </div>
  <div class="grid">
    {cards_joined}
  </div>
</body>
</html>
"""


def main():
    import argparse

    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="luxury_centered_card stress-test runner")
    parser.add_argument("--suite", type=str, default=str(root_dir / "tests" / "test_luxury_centered_card_suite.json"))
    parser.add_argument("--output", type=str, default=str(root_dir / "output_tendoo_v3" / "luxury_centered_card_gallery"))
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"================================================================================")
    print(f"🚀 BẮT ĐẦU KIỂM THỬ: luxury_centered_card (Tổng số: {len(cases)} test cases)")
    print(f"================================================================================")

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for idx, case in enumerate(cases, 1):
            cid = case["id"]
            title = case["title"]
            w = case["width"]
            h = case["height"]
            hero = case["hero"]
            subhead = case.get("subhead", "")
            badge = case.get("badge", "")
            extra_texts = case.get("extra_texts", [])
            cta = case.get("cta", "")
            store_info = case.get("store_info", "")
            orientation = case.get("orientation", "center")
            rating = case.get("rating", 5)

            case_dir = out_dir / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            style_dict = case.get("style", {})
            theme_hex = style_dict.get("theme_color", "#D4AF37")
            bg_data_uri = generate_mock_background(w, h, orientation=orientation, theme_hex=theme_hex)

            style_obj = StyleConfig(
                font=style_dict.get("font", "playfair"),
                theme_color=theme_hex,
                text_effect=style_dict.get("text_effect", "3d_gold"),
                background_tone=style_dict.get("background_tone", "dark_luxury"),
            )

            plan = TendooCreativePlan(
                template="luxury_centered_card",
                hero=hero,
                subhead=subhead,
                badge=badge,
                rating=rating,
                extra_texts=extra_texts,
                cta=cta,
                store_info=store_info,
                qr_code=case.get("qr_code"),
                qr_label=case.get("qr_label", "QUÉT MÃ"),
                orientation=orientation,
                style=style_obj,
            )

            t0 = time.time()
            output_png = case_dir / "poster.png"
            output_html = case_dir / "poster.html"

            out_path, html_content = render_plan_to_poster(
                plan=plan,
                bg_data_uri=bg_data_uri,
                output_image_path=output_png,
                width=w,
                height=h,
            )

            with open(output_html, "w", encoding="utf-8") as f:
                f.write(html_content)

            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(f"file:///{output_html.resolve().as_posix()}")
            page.wait_for_timeout(350)

            # Chụp ảnh chất lượng cao
            page.screenshot(path=str(output_png))

            metrics = inspect_page_metrics(page, w, h)
            page.close()

            dur = time.time() - t0

            # Tính toán lỗi
            self_over_count = sum(1 for m in metrics if m["selfOverX"] or m["selfOverY"])
            canvas_over_count = sum(1 for m in metrics if m["canvasOver"])

            # Lấy cỡ font Hero
            hero_m = next((m for m in metrics if "hero-title" in m["className"]), None)
            hero_font = hero_m["fontSize"] if hero_m else "N/A"

            status_icon = "✅" if (self_over_count == 0 and canvas_over_count == 0) else "❌"
            print(f"[{idx:02d}/{len(cases)}] {status_icon} {cid} ({w}x{h}, orient={orientation:<6}) - HeroFont: {hero_font} - Overflow: {self_over_count + canvas_over_count} - Dur: {dur:.2f}s")

            results.append({
                "id": cid,
                "title": title,
                "width": w,
                "height": h,
                "orientation": orientation,
                "duration": dur,
                "self_over": self_over_count,
                "canvas_over": canvas_over_count,
                "hero_font": hero_font,
                "metrics": metrics
            })

        browser.close()

    # Tạo Gallery Báo Cáo
    gallery_html = generate_gallery_html(results, "luxury_centered_card")
    with open(out_dir / "gallery.html", "w", encoding="utf-8") as f:
        f.write(gallery_html)

    tot_self = sum(r["self_over"] for r in results)
    tot_canvas = sum(r["canvas_over"] for r in results)
    print(f"\n================================================================================")
    print(f"🏁 HOÀN TẤT KIỂM THỬ luxury_centered_card:")
    print(f"   - Tổng test cases: {len(results)}")
    print(f"   - Tổng lỗi tự tràn (Self Overflow): {tot_self}")
    print(f"   - Tổng lỗi tràn Canvas (Canvas Overflow): {tot_canvas}")
    print(f"   - Báo cáo Gallery: {out_dir / 'gallery.html'}")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()
