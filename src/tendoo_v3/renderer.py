"""
src/tendoo_v3/renderer.py

Engine kết nối HTML/CSS & Playwright Chromium Renderer cho Tendoo v3:
- Nạp template Jinja2 từ `src/tendoo_v3/templates/<template>/template.html`.
- Nhúng 100% font tiếng Việt Unicode Base64 offline.
- Tự động áp dụng các hiệu ứng ánh sáng (3D Gold, Neon, Chrome) và CSS Glassmorphism.
- Chụp ảnh PNG không nén (lossless) thông qua Headless Chromium.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, Optional

import jinja2
from PIL import Image

from tendoo.core.fonts import resolve_font
from tendoo.poster_renderer import PosterRenderer
from tendoo_v3.catalog import TEMPLATE_CATALOG
from tendoo_v3.geometry import get_zones
from tendoo_v3.icons import (
    get_icon_svg,
    infer_semantic_icon,
    parse_store_info_items,
    render_qr_code_svg,
    render_star_rating_svg,
)
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.styles import (
    COMMON_AUTOFIT_JS,
    get_adaptive_palette,
    get_effect_css,
    get_vfx_overlay_html_and_css,
)

# Chặn nội dung dạng list phình vô hạn TRƯỚC khi tới template -- mask/CSS của mỗi
# zone là 1 khối kích thước cố định (xem geometry.py), autofit chỉ CO CHỮ chứ không
# tự sinh thêm không gian, nên số lượng phần tử vẫn phải có trần hợp lý làm lưới an
# toàn cuối cùng (đã đo thấy thật: extra-tag-row/board-col-left không giới hạn số
# dòng có thể đẩy nội dung tràn ra ngoài vùng đã dành cho nó).
MAX_EXTRA_TEXTS = 6
MAX_STORE_ITEMS = 3
MAX_STEPS = 6


def _zone_to_ctx(rect) -> Dict[str, float]:
    x1, y1, x2, y2 = rect
    return {
        "x": x1,
        "y": y1,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": max(0.0, x2 - x1),
        "height": max(0.0, y2 - y1),
    }

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=False,
)


def pil_to_base64_data_uri(img: Image.Image, format: str = "PNG") -> str:
    """Chuyển đổi PIL Image thành Base64 Data URI."""
    import io
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_template_html(
    plan: TendooCreativePlan,
    bg_data_uri: str,
    width: int,
    height: int,
    palette_override: Optional[Dict[str, str]] = None,
) -> str:
    """Biên dịch TendooCreativePlan thành chuỗi HTML5/CSS3 tự chứa 100%."""
    tpl_name = plan.template if plan.template in TEMPLATE_CATALOG else "sandwich_top_heavy"
    
    # 1. Resolve Font Unicode Tiếng Việt
    canonical_font, font_face_css, headline_font_css = resolve_font(
        font_key=plan.style.font,
        text_content=f"{plan.hero} {plan.subhead or ''} {plan.testimonial or ''}",
    )

    # 2. Resolve Text Effect CSS & Adaptive Palette (Không vệt đen)
    effect_css = get_effect_css(
        effect_name=plan.style.text_effect,
        theme_color=plan.style.theme_color,
    )
    palette = palette_override or get_adaptive_palette(
        background_tone=plan.style.background_tone,
        theme_color=plan.style.theme_color,
    )

    # 2.5. Resolve Cinematic VFX Layer (film grain, anamorphic flare, gold dust, etc.)
    vfx_name = getattr(plan.style, "vfx", "none")
    vfx_html = get_vfx_overlay_html_and_css(
        vfx_name=vfx_name,
        width=width,
        height=height,
        theme_color=plan.style.theme_color,
    )

    # 3. Rating Star SVG & QR Code Component (Thực tế quét được)
    stars_svg = render_star_rating_svg(count=plan.rating or 5, fill_color=plan.style.theme_color) if plan.rating else ""
    from tendoo_v3.qr import render_scannable_qr_component
    qr_svg = render_scannable_qr_component(
        data=plan.qr_code,
        label=plan.qr_label or "QUÉT MÃ NGAY",
        theme_color=plan.style.theme_color,
    ) if plan.qr_code else ""

    # 4. Semantic Pills & Tags (tự động gắn icon theo ngữ nghĩa câu từ tiếng Việt) --
    # cắt trần số lượng TRƯỚC khi render (xem MAX_EXTRA_TEXTS ở đầu file).
    extra_pills = []
    for item in plan.extra_texts[:MAX_EXTRA_TEXTS]:
        icon_name = infer_semantic_icon(item)
        icon_svg = get_icon_svg(icon_name)
        extra_pills.append({"text": item, "icon_name": icon_name, "icon_svg": icon_svg})

    tag_left_icon = get_icon_svg(infer_semantic_icon(plan.tag_left)) if plan.tag_left else ""
    tag_right_icon = get_icon_svg(infer_semantic_icon(plan.tag_right)) if plan.tag_right else ""
    cta_icon = get_icon_svg("arrow_right") if plan.cta else ""
    store_items = parse_store_info_items(plan.store_info)[:MAX_STORE_ITEMS]
    steps = plan.steps[:MAX_STEPS]

    # 5. Hình học vùng chữ THẬT của template này (xem geometry.py) -- nguồn chân lý
    # DUY NHẤT dùng chung với mask_engine.py, để CSS/autofit và mask không còn lệch
    # nhau. Mỗi zone injected dưới dạng dict {x1,y1,x2,y2,width,height} px, template
    # dùng `zones.<ten_zone>.height` v.v. làm `data-max-height`/`data-max-width` cho
    # phần tử autofit thay vì đoán qua `parentElement.clientHeight` (sai với mọi phần
    # tử position:absolute có DOM-parent lớn hơn vùng thị giác thật của nó).
    zones = {name: _zone_to_ctx(rect) for name, rect in get_zones(tpl_name, width, height, orientation=plan.orientation).items()}

    # 6. Nạp template Jinja2
    template_file = f"{tpl_name}/template.html"
    try:
        jinja_tpl = _JINJA_ENV.get_template(template_file)
    except Exception as e:
        jinja_tpl = _JINJA_ENV.get_template("sandwich_top_heavy/template.html")

    # 7. Render context
    context = {
        "width": width,
        "height": height,
        "bg_data_uri": bg_data_uri,
        "font_face_css": font_face_css,
        "headline_font_css": headline_font_css,
        "theme_color": plan.style.theme_color,
        "effect_css": effect_css,
        "palette": palette,
        "hero": plan.hero,
        "subhead": plan.subhead,
        "badge": plan.badge,
        "tag_left": plan.tag_left,
        "tag_left_icon": tag_left_icon,
        "tag_right": plan.tag_right,
        "tag_right_icon": tag_right_icon,
        "stars_svg": stars_svg,
        "extra_texts": plan.extra_texts[:MAX_EXTRA_TEXTS],
        "extra_pills": extra_pills,
        "cta": plan.cta,
        "cta_icon": cta_icon,
        "store_info": plan.store_info,
        "store_items": store_items,
        "qr_code": plan.qr_code,
        "qr_label": plan.qr_label,
        "qr_svg": qr_svg,
        "testimonial": plan.testimonial,
        "reviewer_name": plan.reviewer_name,
        "steps": steps,
        "orientation": plan.orientation or "bottom_left",
        "zones": zones,
        "vfx_html": vfx_html,
        "autofit_js": COMMON_AUTOFIT_JS,
    }

    return jinja_tpl.render(**context)


def render_plan_to_poster(
    plan: TendooCreativePlan,
    bg_data_uri: str,
    output_image_path: Path | str,
    width: int,
    height: int,
    palette_override: Optional[Dict[str, str]] = None,
) -> Tuple[Path, str]:
    """Render plan thành ảnh poster PNG qua Playwright Chromium.
    Trả về (output_path, html_content).
    """
    html_content = build_template_html(
        plan=plan,
        bg_data_uri=bg_data_uri,
        width=width,
        height=height,
        palette_override=palette_override,
    )
    
    out_file = Path(output_image_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    PosterRenderer.render(
        html_content=html_content,
        output_image_path=out_file,
        width=width,
        height=height,
    )
    
    return out_file, html_content
