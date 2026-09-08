"""
Adaptive Visual Component Engine for Tendoo AI.
==============================================
Renders high-contrast, professional, typography-aligned HTML/CSS modules
for the 6 primary commercial business intents across all layout topologies:
  1. promo: Promotional discount badge + conditions + date chip.
  2. product_intro: Prominent price tag + tech spec tags + description.
  3. opening: Grand opening ribbon + calendar date + booking contact.
  4. feedback: Liquid glassmorphism quote card + 5-star rating + target.
  5. recruitment: Role badge + salary/benefits box + deadline + apply method.
  6. guide: Vertical/Horizontal Stepper Timeline with connected steps.
"""

from __future__ import annotations

import html
from typing import List, Optional

from tendoo.layouts.base import CALENDAR_ICON_SVG, ColorPalette, PosterContent
from tendoo.layouts.text_engine import normalize_text


# Clean inline vector stars for rating
STAR_ICON_SVG = (
    '<svg class="icon-star" width="16" height="16" viewBox="0 0 24 24" fill="#FFB300" stroke="#FFB300" stroke-width="1" '
    'style="display:inline-block; vertical-align:-2px; margin-right:2px;">'
    '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>'
    '</svg>'
)

CLOCK_ICON_SVG = (
    '<svg class="icon-clock" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; opacity:0.9;">'
    '<circle cx="12" cy="12" r="10"></circle>'
    '<polyline points="12 6 12 12 16 14"></polyline>'
    '</svg>'
)

CHECK_ICON_SVG = (
    '<svg class="icon-check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-2px; margin-right:5px; color:#10B981;">'
    '<polyline points="20 6 9 17 4 12"></polyline>'
    '</svg>'
)


def render_category_body(
    content: PosterContent,
    palette: ColorPalette,
    layout_name: str = "top_dome",
) -> str:
    """
    Assembles the adaptive visual component HTML tailored for the content's category.
    If content.category_body_html is already provided, returns it directly.
    """
    if content.category_body_html and content.category_body_html.strip():
        return content.category_body_html.strip()

    cat = (content.category or "promo").lower().strip()

    # Border contrast based on badge text
    if palette.badge_text == "#0D0D14":
        badge_border = "rgba(0, 0, 0, 0.22)"
    else:
        badge_border = "rgba(255, 255, 255, 0.38)"

    if cat == "product_intro":
        return _render_product_intro(content, palette, layout_name, badge_border)
    elif cat == "opening":
        return _render_opening(content, palette, layout_name, badge_border)
    elif cat == "feedback":
        return _render_feedback(content, palette, layout_name, badge_border)
    elif cat == "recruitment":
        return _render_recruitment(content, palette, layout_name, badge_border)
    elif cat == "guide":
        return _render_guide(content, palette, layout_name, badge_border)
    else:
        # Default: promo (100% backward compatible)
        return _render_promo(content, palette, layout_name, badge_border)


def _render_promo(
    content: PosterContent,
    palette: ColorPalette,
    layout_name: str,
    badge_border: str,
) -> str:
    """Standard promotional discount offer capsule + conditions + dates."""
    offer_main = normalize_text(content.offer_main or content.discount if hasattr(content, 'discount') else content.offer_main)
    offer_sub = normalize_text(content.offer_sub or content.applied if hasattr(content, 'applied') else content.offer_sub)
    dates = normalize_text(content.dates)

    parts = []
    if offer_main:
        parts.append(
            f'<div class="cat-comp-promo-badge badge-wrapper">'
            f'  <div class="badge-pill">{html.escape(offer_main)}</div>'
            f'</div>'
        )
    if offer_sub:
        parts.append(f'<div class="cat-comp-promo-sub offer-sub">{html.escape(offer_sub)}</div>')
    if dates:
        parts.append(
            f'<div class="cat-comp-promo-dates dates-chip">'
            f'  {CALENDAR_ICON_SVG}<span>{html.escape(dates)}</span>'
            f'</div>'
        )

    if not parts:
        return ""
    return f'<div class="category-body-container category-promo-container">\n  ' + "\n  ".join(parts) + "\n</div>"


def _render_product_intro(
    content: PosterContent,
    palette: ColorPalette,
    layout_name: str,
    badge_border: str,
) -> str:
    """Product showcase: prominent price badge + product name + spec highlights."""
    price = normalize_text(content.price or content.offer_main)
    product_name = normalize_text(content.product_name)
    product_desc = normalize_text(content.product_desc or content.offer_sub)
    highlights_raw = normalize_text(content.highlights or content.applicable)

    parts = []

    # 1. Product Name Badge
    if product_name:
        parts.append(
            f'<div class="cat-comp-prod-name-tag">'
            f'  <span class="prod-tag-text">{html.escape(product_name)}</span>'
            f'</div>'
        )

    # 2. Prominent Price Badge
    if price:
        parts.append(
            f'<div class="cat-comp-price-wrap">'
            f'  <div class="cat-comp-price-pill badge-pill">'
            f'    <span class="price-prefix">GIÁ BÁN:</span>'
            f'    <span class="price-number">{html.escape(price)}</span>'
            f'  </div>'
            f'</div>'
        )

    # 3. Product Description
    if product_desc:
        parts.append(f'<div class="cat-comp-prod-desc offer-sub">{html.escape(product_desc)}</div>')

    # 4. Feature Highlights Chips
    if highlights_raw:
        chips = [c.strip() for c in highlights_raw.replace(";", ",").replace("•", ",").replace("\n", ",").split(",") if c.strip()]
        if chips:
            chips_html = "".join(f'<span class="spec-chip">{CHECK_ICON_SVG}{html.escape(c)}</span>' for c in chips[:4])
            parts.append(f'<div class="cat-comp-specs-row">{chips_html}</div>')

    if not parts:
        return ""
    return f'<div class="category-body-container category-product-container">\n  ' + "\n  ".join(parts) + "\n</div>"


def _render_opening(
    content: PosterContent,
    palette: ColorPalette,
    layout_name: str,
    badge_border: str,
) -> str:
    """Opening banner: Grand opening date + opening gift/promo + booking hotline."""
    opening_date = normalize_text(content.opening_date or content.dates)
    opening_promo = normalize_text(content.opening_promo or content.offer_main)
    booking_contact = normalize_text(content.booking_contact or content.offer_sub)

    parts = []

    # 1. Opening Date Chip
    if opening_date:
        parts.append(
            f'<div class="cat-comp-opening-date dates-chip">'
            f'  {CALENDAR_ICON_SVG}<span class="date-label">NGÀY KHAI TRƯƠNG:</span>'
            f'  <span class="date-val">{html.escape(opening_date)}</span>'
            f'</div>'
        )

    # 2. Grand Opening Promo Ribbon Badge
    if opening_promo:
        parts.append(
            f'<div class="cat-comp-opening-promo badge-wrapper">'
            f'  <div class="badge-pill">{html.escape(opening_promo)}</div>'
            f'</div>'
        )

    # 3. Booking / Table Reservation Chip
    if booking_contact:
        parts.append(
            f'<div class="cat-comp-booking-chip">'
            f'  <span>📞 LIÊN HỆ ĐẶT CHỖ: {html.escape(booking_contact)}</span>'
            f'</div>'
        )

    if not parts:
        return ""
    return f'<div class="category-body-container category-opening-container">\n  ' + "\n  ".join(parts) + "\n</div>"


def _render_feedback(
    content: PosterContent,
    palette: ColorPalette,
    layout_name: str,
    badge_border: str,
) -> str:
    """Customer testimonial: Liquid glass quote card + 5-star rating + target product."""
    target = normalize_text(content.feedback_target or content.pre_header)
    quote = normalize_text(content.feedback_quote or content.offer_sub or content.slogan)
    rating = normalize_text(content.feedback_rating or content.offer_main or "5.0 / 5.0")
    special_offer = normalize_text(content.special_offer)

    parts = []

    # Star icons generator
    star_count = 5
    if "4" in rating and "5" not in rating.split("/")[0]:
        star_count = 4
    stars_html = STAR_ICON_SVG * star_count

    # Liquid Glassmorphism Quote Box
    parts.append(
        f'<div class="cat-comp-feedback-card">'
        f'  <div class="feedback-card-header">'
        f'    <div class="feedback-stars">{stars_html}</div>'
        f'    <div class="feedback-rating-text">{html.escape(rating)}</div>'
        f'  </div>'
        f'  <div class="feedback-quote-content">'
        f'    <span class="feedback-quote-mark">“</span>'
        f'    <span class="feedback-quote-text">{html.escape(quote or "Sản phẩm trên cả tuyệt vời, chất lượng hoàn hảo và dịch vụ rất chu đáo!")}</span>'
        f'    <span class="feedback-quote-mark">”</span>'
        f'  </div>'
        + (f'  <div class="feedback-target-tag">— Khách hàng đánh giá: <b>{html.escape(target)}</b></div>' if target else "")
        + f'</div>'
    )

    # Special Loyalty Offer Tag
    if special_offer:
        parts.append(
            f'<div class="cat-comp-special-offer badge-pill">'
            f'  <span>🎁 {html.escape(special_offer)}</span>'
            f'</div>'
        )

    return f'<div class="category-body-container category-feedback-container">\n  ' + "\n  ".join(parts) + "\n</div>"


def _render_recruitment(
    content: PosterContent,
    palette: ColorPalette,
    layout_name: str,
    badge_border: str,
) -> str:
    """Recruitment poster: Position badge + salary/benefits + deadline + apply method."""
    job_position = normalize_text(content.job_position or content.offer_main or "VỊ TRÍ TUYỂN DỤNG")
    job_desc = normalize_text(content.job_desc or content.offer_sub)
    apply_deadline = normalize_text(content.apply_deadline or content.dates)
    apply_method = normalize_text(content.apply_method or content.applicable)

    parts = []

    # 1. Job Role Badge
    if job_position:
        parts.append(
            f'<div class="cat-comp-job-role badge-wrapper">'
            f'  <div class="badge-pill">{html.escape(job_position)}</div>'
            f'</div>'
        )

    # 2. Job Description & Benefits Box
    if job_desc:
        parts.append(
            f'<div class="cat-comp-job-desc-box">'
            f'  <div class="job-desc-text">{html.escape(job_desc)}</div>'
            f'</div>'
        )

    # 3. Action Badges Row (Deadline & Apply Method)
    action_items = []
    if apply_deadline:
        action_items.append(
            f'<div class="dates-chip">'
            f'  {CLOCK_ICON_SVG}<span>Hạn nộp: {html.escape(apply_deadline)}</span>'
            f'</div>'
        )
    if apply_method:
        action_items.append(
            f'<div class="dates-chip apply-method-chip">'
            f'  <span>Ứng tuyển: {html.escape(apply_method)}</span>'
            f'</div>'
        )

    if action_items:
        parts.append(f'<div class="cat-comp-job-actions">{"".join(action_items)}</div>')

    if not parts:
        return ""
    return f'<div class="category-body-container category-recruitment-container">\n  ' + "\n  ".join(parts) + "\n</div>"


def _render_guide(
    content: PosterContent,
    palette: ColorPalette,
    layout_name: str,
    badge_border: str,
) -> str:
    """Usage Guide: Vertical/Horizontal Stepper Timeline with connecting lines."""
    steps = content.steps if content.steps else []
    if not steps:
        raw = content.offer_sub or content.applicable or ""
        if raw:
            steps = [s.strip() for s in raw.split("\n") if s.strip()]

    if not steps:
        steps = [
            "Bước 1: Chọn sản phẩm hoặc quét mã QR",
            "Bước 2: Xác nhận thông tin và đặt mua",
            "Bước 3: Nhận hàng nhanh chóng tận nhà",
        ]

    step_items_html = []
    for idx, step_text in enumerate(steps, start=1):
        clean_step = normalize_text(step_text)
        clean_disp = clean_step
        if clean_disp.lower().startswith(f"bước {idx}:"):
            clean_disp = clean_disp[len(f"bước {idx}:"):].strip()
        elif clean_disp.lower().startswith(f"bước {idx} -"):
            clean_disp = clean_disp[len(f"bước {idx} -"):].strip()

        step_items_html.append(
            f'<div class="timeline-step">'
            f'  <div class="step-num-node">{idx}</div>'
            f'  <div class="step-content-node">'
            f'    <div class="step-title">Bước {idx}</div>'
            f'    <div class="step-desc">{html.escape(clean_disp)}</div>'
            f'  </div>'
            f'</div>'
        )

    stepper_html = f'<div class="cat-comp-stepper-timeline">{"".join(step_items_html)}</div>'
    return f'<div class="category-body-container category-guide-container">\n  {stepper_html}\n</div>'


def get_component_css() -> str:
    """
    Returns standard CSS rules for all adaptive category components.
    Injected cleanly into all layout templates.
    """
    return """
    /* ==========================================================================
       ADAPTIVE VISUAL COMPONENT ENGINE STYLES
       ========================================================================== */
    .category-body-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 100%;
      max-width: 96%;
      gap: 10px;
      margin-top: 4px;
    }

    /* Product Intro Styles */
    .cat-comp-prod-name-tag {
      display: inline-flex;
      align-items: center;
      padding: 4px 16px;
      border-radius: 9999px;
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      color: var(--accent-color);
      font-size: 13.5px;
      font-weight: 800;
      letter-spacing: 2px;
      text-transform: uppercase;
      text-shadow: var(--text-shadow);
    }

    .badge-pill, .cat-comp-price-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 9px 28px;
      border-radius: 9999px;
      background: var(--badge-bg);
      color: var(--badge-text);
      box-shadow: var(--badge-shadow);
      font-family: 'Montserrat', sans-serif;
      font-size: 19px;
      font-weight: 900;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      border: 1px solid rgba(255, 255, 255, 0.38);
      white-space: nowrap;
    }

    .price-prefix {
      font-size: 13.5px;
      font-weight: 700;
      opacity: 0.85;
      letter-spacing: 1.5px;
    }

    .price-number {
      font-size: 23px;
      font-weight: 900;
      letter-spacing: 1px;
    }

    .cat-comp-specs-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 2px;
    }

    .spec-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 4px 12px;
      border-radius: 6px;
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      font-size: 13px;
      font-weight: 700;
      color: var(--sub-color);
      letter-spacing: 0.4px;
      text-shadow: var(--text-shadow);
    }

    /* Opening Banner Styles */
    .cat-comp-opening-date {
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0.8px;
    }

    .date-label {
      font-size: 12.5px;
      opacity: 0.85;
      letter-spacing: 1.2px;
      margin-right: 4px;
    }

    .cat-comp-booking-chip {
      display: inline-flex;
      align-items: center;
      padding: 6px 18px;
      border-radius: 9999px;
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      color: var(--accent-color);
      font-size: 14.5px;
      font-weight: 800;
      letter-spacing: 0.6px;
      text-shadow: var(--text-shadow);
    }

    /* Customer Feedback Styles */
    .cat-comp-feedback-card {
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      padding: 14px 22px;
      border-radius: 16px;
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
      width: 100%;
      max-width: 620px;
      gap: 8px;
    }

    .feedback-card-header {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .feedback-rating-text {
      font-size: 14px;
      font-weight: 800;
      color: #FFB300;
      letter-spacing: 0.5px;
    }

    .feedback-quote-content {
      font-size: 16.5px;
      font-weight: 600;
      font-style: italic;
      line-height: 1.45;
      color: var(--sub-color);
      text-shadow: var(--text-shadow);
      text-wrap: balance;
    }

    .feedback-quote-mark {
      font-family: 'Playfair Display', serif;
      font-size: 24px;
      font-weight: 900;
      color: var(--accent-color);
      line-height: 1;
      vertical-align: -2px;
    }

    .feedback-target-tag {
      font-size: 13px;
      font-weight: 500;
      color: var(--sub-color);
      letter-spacing: 0.4px;
      text-shadow: var(--text-shadow);
    }

    .cat-comp-special-offer {
      font-size: 15px;
      padding: 7px 22px;
    }

    /* Recruitment Styles */
    .cat-comp-job-desc-box {
      padding: 10px 20px;
      border-radius: 12px;
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      font-size: 15.5px;
      font-weight: 600;
      color: var(--sub-color);
      text-align: center;
      line-height: 1.4;
      text-shadow: var(--text-shadow);
      max-width: 600px;
    }

    .cat-comp-job-actions {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    /* Stepper Timeline Styles (Usage Guide) */
    .cat-comp-stepper-timeline {
      display: flex;
      flex-direction: column;
      gap: 10px;
      width: 100%;
      max-width: 580px;
      position: relative;
      padding-left: 6px;
    }

    .timeline-step {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      text-align: left;
      position: relative;
    }

    .step-num-node {
      width: 32px;
      height: 32px;
      min-width: 32px;
      border-radius: 50%;
      background: var(--badge-bg);
      color: var(--badge-text);
      box-shadow: var(--badge-shadow);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Montserrat', sans-serif;
      font-size: 15px;
      font-weight: 900;
      border: 1px solid rgba(255, 255, 255, 0.4);
      z-index: 2;
    }

    .step-content-node {
      display: flex;
      flex-direction: column;
      gap: 2px;
      padding-top: 2px;
    }

    .step-title {
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--accent-color);
      text-shadow: var(--text-shadow);
    }

    .step-desc {
      font-size: 15.5px;
      font-weight: 600;
      color: var(--sub-color);
      line-height: 1.35;
      text-shadow: var(--text-shadow);
    }

    /* Promo Offer Sub & Dates Chip */
    .cat-comp-promo-badge {
      display: flex;
      justify-content: center;
      width: 100%;
    }

    .cat-comp-promo-sub, .offer-sub {
      font-size: 15.5px;
      font-weight: 600;
      color: var(--sub-color);
      text-shadow: var(--text-shadow);
      text-align: center;
      line-height: 1.4;
      text-wrap: balance;
    }

    .cat-comp-promo-dates, .dates-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 16px;
      border-radius: 9999px;
      background: var(--glass-bg);
      border: 1px solid var(--glass-border);
      color: var(--sub-color);
      font-size: 13px;
      font-weight: 700;
      text-shadow: var(--text-shadow);
    }

    /* Asymmetrical Layout Alignments (Split Column, Diagonal Slash, L-Frame) */
    .editorial-column .category-body-container,
    .diagonal-content-stack .category-body-container,
    .lframe-left-column .category-body-container {
      align-items: flex-start;
      text-align: left;
    }

    .editorial-column .cat-comp-promo-badge,
    .diagonal-content-stack .cat-comp-promo-badge,
    .lframe-left-column .cat-comp-promo-badge {
      justify-content: flex-start;
    }

    .editorial-column .cat-comp-promo-sub,
    .diagonal-content-stack .cat-comp-promo-sub,
    .lframe-left-column .cat-comp-promo-sub {
      text-align: left;
    }
    """
