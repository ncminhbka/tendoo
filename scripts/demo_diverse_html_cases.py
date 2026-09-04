#!/usr/bin/env python3
"""
scripts/demo_diverse_html_cases.py

Demonstrates how HTML/CSS Scalable Typography effortlessly solves diverse commercial use-cases:
  1. Grand Opening Burger Banner with 50% Off Burst Badge
  2. Customer Feedback & Review Card for Pet Spa (Prompt 23 in prompt_test.txt)
  3. High-Density Text Recruitment Poster with Frosted Glassmorphic Box
  4. Restaurant / Food Menu with Dotted Leaders & Price Tags
"""

import asyncio
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tendoo.typography_engine import PosterRenderer

OUTPUT_DIR = Path("output_diverse_cases")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================================================================
# CASE 1: BURGER GRAND OPENING (50% OFF BADGE, NEON AMBER VIBE)
# ==================================================================================================
HTML_BURGER = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Burger Grand Opening</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center;
      background: #0a0503; font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .poster {
      position: relative; width: 1024px; height: 1024px; overflow: hidden;
      background: radial-gradient(circle at 50% 45%, #2a1205 0%, #120601 60%, #050201 100%);
      box-shadow: 0 25px 60px rgba(0,0,0,0.8);
    }
    /* Ambient Glows */
    .glow-orange {
      position: absolute; width: 600px; height: 600px; border-radius: 50%;
      background: radial-gradient(circle, rgba(255, 107, 0, 0.25) 0%, rgba(255, 60, 0, 0) 70%);
      top: 200px; left: 212px; pointer-events: none; filter: blur(40px);
    }
    /* Header Brand */
    .header {
      position: absolute; top: 48px; left: 56px; right: 56px;
      display: flex; justify-content: space-between; align-items: center; z-index: 20;
    }
    .brand-title {
      font-family: 'Montserrat', sans-serif; font-size: 24px; font-weight: 900;
      color: #FFB703; letter-spacing: 2px; text-transform: uppercase;
      text-shadow: 0 0 20px rgba(255, 183, 3, 0.5);
    }
    .date-pill {
      background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 183, 3, 0.4); padding: 10px 22px;
      border-radius: 999px; font-size: 14px; font-weight: 700; color: #FFF;
      letter-spacing: 1px;
    }
    /* Hero Opening Title */
    .hero-title-box {
      position: absolute; top: 120px; left: 56px; right: 56px; text-align: center; z-index: 15;
    }
    .eyebrow {
      font-size: 18px; font-weight: 800; color: #FB8500; letter-spacing: 6px;
      text-transform: uppercase; margin-bottom: 8px; text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    .hero-main {
      font-family: 'Montserrat', sans-serif; font-size: 68px; font-weight: 900;
      line-height: 1.1; text-transform: uppercase; letter-spacing: 1px;
      background: linear-gradient(180deg, #FFFFFF 0%, #FFE8B8 50%, #FFB703 100%);
      -webkit-background-clip: text; color: transparent;
      filter: drop-shadow(0 6px 20px rgba(255, 136, 0, 0.6));
    }
    /* 50% OFF Explosive Badge */
    .burst-badge {
      position: absolute; top: 220px; right: 70px; width: 170px; height: 170px;
      background: linear-gradient(135deg, #E63946 0%, #D90429 100%);
      border-radius: 50%; display: flex; flex-direction: column; justify-content: center; align-items: center;
      box-shadow: 0 12px 35px rgba(230, 57, 70, 0.6), inset 0 3px 6px rgba(255, 255, 255, 0.5);
      border: 4px dashed #FFF; transform: rotate(12deg); z-index: 25;
      animation: pulse 2s infinite alternate;
    }
    .badge-sub { font-size: 14px; font-weight: 800; color: #FFF; letter-spacing: 2px; text-transform: uppercase; }
    .badge-main { font-family: 'Montserrat', sans-serif; font-size: 52px; font-weight: 900; color: #FFF; line-height: 0.95; }
    .badge-off { font-size: 16px; font-weight: 900; color: #FFD166; letter-spacing: 1.5px; }

    /* Mockup Graphic Area (Burger Silhouette/Placeholder) */
    .burger-center-circle {
      position: absolute; top: 380px; left: 50%; transform: translateX(-50%);
      width: 420px; height: 420px; border-radius: 50%;
      background: radial-gradient(circle, rgba(255, 183, 3, 0.15) 0%, rgba(0,0,0,0) 70%);
      border: 2px dashed rgba(255, 183, 3, 0.3); display: flex; justify-content: center; align-items: center;
    }
    .center-label {
      font-size: 80px; filter: drop-shadow(0 10px 30px rgba(255, 107, 0, 0.8));
    }

    /* Bottom Info & CTA */
    .bottom-bar {
      position: absolute; bottom: 50px; left: 56px; right: 56px; z-index: 20;
      background: rgba(20, 10, 5, 0.75); backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 183, 3, 0.25); border-radius: 24px;
      padding: 24px 36px; display: flex; justify-content: space-between; align-items: center;
      box-shadow: 0 15px 40px rgba(0,0,0,0.6);
    }
    .deal-info { display: flex; flex-direction: column; gap: 4px; }
    .deal-title { font-family: 'Montserrat', sans-serif; font-size: 20px; font-weight: 800; color: #FFF; }
    .deal-sub { font-size: 14px; font-weight: 500; color: #FFB703; }
    .cta-btn {
      background: linear-gradient(135deg, #FB8500 0%, #FFB703 100%);
      color: #000; font-family: 'Montserrat', sans-serif; font-weight: 900;
      font-size: 17px; letter-spacing: 0.5px; padding: 16px 36px; border-radius: 999px;
      text-decoration: none; box-shadow: 0 8px 25px rgba(251, 133, 0, 0.5);
      border: 1px solid rgba(255,255,255,0.4);
    }
  </style>
</head>
<body>
  <div class="poster">
    <div class="glow-orange"></div>
    <div class="header">
      <div class="brand-title">🍔 THE BURGER CRAFT</div>
      <div class="date-pill">DUY NHẤT 05.09 - 15.09.2026</div>
    </div>
    <div class="hero-title-box">
      <div class="eyebrow">TƯNG BỪNG KHAI TRƯƠNG CHI NHÁNH MỚI</div>
      <div class="hero-main">ĂN THẢ GA • KHÔNG LO GIÁ</div>
    </div>
    <div class="burst-badge">
      <span class="badge-sub">GIẢM</span>
      <span class="badge-main">50%</span>
      <span class="badge-off">TOÀN MENU</span>
    </div>
    <div class="burger-center-circle">
      <div class="center-label">🍔✨</div>
    </div>
    <div class="bottom-bar">
      <div class="deal-info">
        <div class="deal-title">📍 128 Nguyễn Trãi, Phường Bến Thành, Quận 1</div>
        <div class="deal-sub">Tặng 01 Coca-Cola mát lạnh cho hóa đơn từ 99K • Hotline: 1900 8899</div>
      </div>
      <a href="#" class="cta-btn">NHẬN VOUCHER ➔</a>
    </div>
  </div>
</body>
</html>"""

# ==================================================================================================
# CASE 2: PET SPA CUSTOMER FEEDBACK (PROMPT 23 IN prompt_test.txt)
# ==================================================================================================
HTML_PET_SPA = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pet Spa Customer Feedback</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center;
      background: #eaf8f5; font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .poster {
      position: relative; width: 1024px; height: 1024px; overflow: hidden;
      background: linear-gradient(145deg, #F0FAF7 0%, #FFF5F7 50%, #F5FBF9 100%);
      box-shadow: 0 25px 60px rgba(0,0,0,0.12); border-radius: 32px;
    }
    /* Top Bar */
    .top-bar {
      position: absolute; top: 48px; left: 56px; right: 56px;
      display: flex; justify-content: space-between; align-items: center; z-index: 20;
    }
    .spa-logo {
      font-family: 'Quicksand', sans-serif; font-size: 26px; font-weight: 800;
      color: #0E9F6E; display: flex; align-items: center; gap: 8px;
    }
    .spa-badge {
      background: #FFE4E6; color: #E02424; font-family: 'Quicksand', sans-serif;
      font-weight: 800; font-size: 14px; padding: 10px 20px; border-radius: 999px;
      border: 1px solid #FECDD3; box-shadow: 0 4px 12px rgba(224, 36, 36, 0.1);
    }
    /* Title Box */
    .title-box {
      position: absolute; top: 120px; left: 56px; right: 56px; text-align: center;
    }
    .title-pre { font-size: 15px; font-weight: 700; color: #047481; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 6px; }
    .main-title {
      font-family: 'Quicksand', sans-serif; font-size: 52px; font-weight: 800;
      color: #111928; line-height: 1.2;
    }
    .highlight { color: #0E9F6E; }

    /* Frosted Glass Customer Testimonial Card */
    .feedback-card {
      position: absolute; top: 250px; left: 70px; right: 70px;
      background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px); border: 2px solid rgba(255, 255, 255, 0.9);
      border-radius: 28px; padding: 40px 48px;
      box-shadow: 0 20px 40px rgba(0, 150, 110, 0.08), 0 1px 3px rgba(0,0,0,0.05);
      z-index: 20;
    }
    .review-header {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
    }
    .stars { color: #F59E0B; font-size: 26px; letter-spacing: 4px; }
    .verified-pill {
      display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 700;
      color: #057A55; background: #DEF7EC; padding: 6px 14px; border-radius: 999px;
    }
    .quote-text {
      font-size: 21px; line-height: 1.6; color: #374151; font-weight: 500;
      font-style: italic; margin-bottom: 24px; position: relative;
    }
    .quote-text::before {
      content: "“"; font-size: 70px; color: #A7F3D0; font-family: serif;
      position: absolute; left: -32px; top: -25px; line-height: 1; opacity: 0.6;
    }
    .customer-info {
      display: flex; align-items: center; gap: 16px; border-top: 1px solid #E5E7EB;
      padding-top: 18px;
    }
    .avatar {
      width: 52px; height: 52px; border-radius: 50%; background: #D1FAE5;
      display: flex; justify-content: center; align-items: center; font-size: 26px;
      border: 2px solid #0E9F6E;
    }
    .cust-name { font-size: 17px; font-weight: 700; color: #111928; }
    .cust-sub { font-size: 13px; color: #6B7280; font-weight: 500; }

    /* Highlight Feature Pills */
    .features-row {
      position: absolute; top: 620px; left: 70px; right: 70px;
      display: flex; justify-content: space-between; gap: 14px; z-index: 20;
    }
    .f-pill {
      flex: 1; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 18px;
      padding: 20px; text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    .f-icon { font-size: 28px; margin-bottom: 8px; }
    .f-text { font-size: 14px; font-weight: 700; color: #1F2A37; line-height: 1.3; }

    /* Promo Offer Ribbon & CTA */
    .bottom-cta-strip {
      position: absolute; bottom: 50px; left: 70px; right: 70px;
      background: linear-gradient(135deg, #0E9F6E 0%, #057A55 100%);
      border-radius: 22px; padding: 22px 36px; display: flex;
      justify-content: space-between; align-items: center;
      box-shadow: 0 12px 30px rgba(14, 159, 110, 0.35); z-index: 20;
    }
    .offer-box { color: #FFFFFF; }
    .offer-title { font-family: 'Quicksand', sans-serif; font-size: 22px; font-weight: 800; }
    .offer-desc { font-size: 14px; opacity: 0.9; margin-top: 2px; }
    .btn-booking {
      background: #FFFFFF; color: #046C4E; font-family: 'Quicksand', sans-serif;
      font-weight: 800; font-size: 16px; padding: 14px 30px; border-radius: 999px;
      text-decoration: none; box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
  </style>
</head>
<body>
  <div class="poster">
    <div class="top-bar">
      <div class="spa-logo">🐾 PAWPARADISE SPA</div>
      <div class="spa-badge">✨ CHUẨN FORM HÀN QUỐC</div>
    </div>
    <div class="title-box">
      <div class="title-pre">KHÁCH HÀNG THẬT • TRẢI NGHIỆM THẬT</div>
      <div class="main-title">Boss Lột Xác Thế Nào <span class="highlight">Sau 2 Giờ Spa?</span></div>
    </div>

    <!-- Customer Feedback Card -->
    <div class="feedback-card">
      <div class="review-header">
        <div class="stars">★★★★★</div>
        <div class="verified-pill">✔ ĐÃ TRẢI NGHIỆM DỊCH VỤ</div>
      </div>
      <div class="quote-text">
        Bé Poodle Miu nhà mình đi spa về thơm nức nở suốt 1 tuần, lông xốp bồng bềnh mềm như kẹo bông gòn! Kỹ thuật viên cắt tỉa mông trái tim siêu khéo, không gian mở không lồng kính nên bé không hề bị stress hay run sợ. 10/10 điểm uy tín!
      </div>
      <div class="customer-info">
        <div class="avatar">🐩</div>
        <div>
          <div class="cust-name">Chị Thu Hằng & Bé Poodle Miu</div>
          <div class="cust-sub">Khách hàng gói Premium Grooming 7 Bước • Quận Đống Đa, Hà Nội</div>
        </div>
      </div>
    </div>

    <!-- Features Row -->
    <div class="features-row">
      <div class="f-pill">
        <div class="f-icon">🌿</div>
        <div class="f-text">Sữa Tắm Hữu Cơ 100% Nhập Khẩu</div>
      </div>
      <div class="f-pill">
        <div class="f-icon">✂️</div>
        <div class="f-text">Cắt Tỉa Chuyên Nghiệp Theo Yêu Cầu</div>
      </div>
      <div class="f-pill">
        <div class="f-icon">🕊️</div>
        <div class="f-text">Không Gian Mở Không Lồng Kính</div>
      </div>
    </div>

    <!-- Bottom Action Strip -->
    <div class="bottom-cta-strip">
      <div class="offer-box">
        <div class="offer-title">🎁 TẶNG GÓI NGÂM SỤC OZON TRỊ GIÁ 200K</div>
        <div class="offer-desc">Áp dụng cho khách hàng đặt lịch trải nghiệm lần đầu tiên trong tuần này!</div>
      </div>
      <a href="#" class="btn-booking">ĐẶT LỊCH NGAY ➔</a>
    </div>
  </div>
</body>
</html>"""

# ==================================================================================================
# CASE 3: HIGH-DENSITY RECRUITMENT POSTER (FROSTED GLASS CONTAINER)
# ==================================================================================================
HTML_RECRUITMENT = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Recruitment Poster</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Montserrat:wght@700;800;900&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center;
      background: #020617; font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .poster {
      position: relative; width: 1024px; height: 1024px; overflow: hidden;
      background: radial-gradient(circle at 80% 20%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
      box-shadow: 0 25px 60px rgba(0,0,0,0.8);
    }
    /* Top Header */
    .rec-header {
      position: absolute; top: 44px; left: 56px; right: 56px;
      display: flex; justify-content: space-between; align-items: center; z-index: 20;
    }
    .company-logo {
      font-family: 'Montserrat', sans-serif; font-size: 22px; font-weight: 900;
      color: #38BDF8; letter-spacing: 2px;
    }
    .urgency-badge {
      background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4);
      color: #F87171; font-size: 13px; font-weight: 700; padding: 8px 18px;
      border-radius: 999px; letter-spacing: 1px;
    }

    /* Main Frosted Glass Box */
    .frosted-box {
      position: absolute; top: 110px; left: 56px; right: 56px; bottom: 44px;
      background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 28px; padding: 40px 48px; display: flex; flex-direction: column;
      justify-content: space-between; box-shadow: 0 20px 50px rgba(0,0,0,0.5); z-index: 20;
    }

    .pos-title-group { display: flex; justify-content: space-between; align-items: flex-start; }
    .pos-label { font-size: 14px; font-weight: 700; color: #38BDF8; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
    .pos-name { font-family: 'Montserrat', sans-serif; font-size: 40px; font-weight: 900; color: #FFFFFF; line-height: 1.1; }
    .salary-tag {
      background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
      color: #FFFFFF; font-family: 'Montserrat', sans-serif; font-weight: 800;
      font-size: 20px; padding: 12px 24px; border-radius: 14px;
      box-shadow: 0 8px 20px rgba(2, 132, 199, 0.4); border: 1px solid rgba(255,255,255,0.25);
    }

    /* Dense Content Columns */
    .two-col-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 36px; margin: 24px 0;
    }
    .col-title {
      font-size: 16px; font-weight: 800; color: #94A3B8; text-transform: uppercase;
      letter-spacing: 1px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
    }
    .checklist { list-style: none; display: flex; flex-direction: column; gap: 12px; }
    .check-item {
      display: flex; align-items: flex-start; gap: 12px; font-size: 15px; color: #E2E8F0;
      line-height: 1.45; font-weight: 500;
    }
    .check-icon { color: #38BDF8; font-weight: 900; font-size: 16px; }

    /* Footer Apply Action */
    .rec-footer {
      display: flex; justify-content: space-between; align-items: center;
      border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 24px;
    }
    .contact-block { display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: #94A3B8; }
    .contact-email { color: #38BDF8; font-weight: 700; font-size: 16px; }
    .apply-btn {
      background: linear-gradient(135deg, #38BDF8 0%, #0284C7 100%);
      color: #020617; font-family: 'Montserrat', sans-serif; font-weight: 800;
      font-size: 17px; padding: 16px 40px; border-radius: 999px; text-decoration: none;
      box-shadow: 0 8px 25px rgba(56, 189, 248, 0.4); border: 1px solid rgba(255,255,255,0.4);
    }
  </style>
</head>
<body>
  <div class="poster">
    <div class="rec-header">
      <div class="company-logo">⚡ TENDOO AI RESEARCH LAB</div>
      <div class="urgency-badge">HẠN NỘP: 30.09.2026</div>
    </div>

    <!-- Frosted Glass Container -->
    <div class="frosted-box">
      <div class="pos-title-group">
        <div>
          <div class="pos-label">WE ARE EXPANDING • FULL-TIME POSITION</div>
          <div class="pos-name">SENIOR AI PRODUCT DESIGNER</div>
        </div>
        <div class="salary-tag">25 - 35 TRIỆU / THÁNG</div>
      </div>

      <div class="two-col-grid">
        <!-- Col 1: Requirements -->
        <div>
          <div class="col-title">📋 YÊU CẦU ỨNG VIÊN</div>
          <ul class="checklist">
            <li class="check-item"><span class="check-icon">✔</span><span>Tối thiểu 2 năm kinh nghiệm UI/UX hoặc Visual Branding cho sản phẩm công nghệ.</span></li>
            <li class="check-item"><span class="check-icon">✔</span><span>Thành thạo Figma, Design System, Adobe Creative Suite và AI tools.</span></li>
            <li class="check-item"><span class="check-icon">✔</span><span>Tư duy thẩm mỹ hiện đại, am hiểu sâu sắc về bố cục và typography thương mại.</span></li>
            <li class="check-item"><span class="check-icon">✔</span><span>Khả năng chủ động dẫn dắt tính năng sản phẩm từ ý tưởng đến thực thi.</span></li>
          </ul>
        </div>

        <!-- Col 2: Benefits -->
        <div>
          <div class="col-title">🎁 QUYỀN LỢI ĐẶC QUYỀN</div>
          <ul class="checklist">
            <li class="check-item"><span class="check-icon">★</span><span>Thưởng dự án theo quý + Chương trình cổ phần ưu đãi ESOP cho nhân sự giỏi.</span></li>
            <li class="check-item"><span class="check-icon">★</span><span>Trang bị MacBook Pro M3 Max + Màn hình Studio Display 4K cao cấp.</span></li>
            <li class="check-item"><span class="check-icon">★</span><span>Làm việc Hybrid linh hoạt (2 ngày remote/tuần), giờ giấc tự do.</span></li>
            <li class="check-item"><span class="check-icon">★</span><span>Bảo hiểm sức khỏe VIP toàn diện, du lịch nghỉ dưỡng cao cấp 2 lần/năm.</span></li>
          </ul>
        </div>
      </div>

      <div class="rec-footer">
        <div class="contact-block">
          <div>Gửi CV & Portfolio trực tiếp về hòm thư:</div>
          <div class="contact-email">careers@tendoo.ai | Hotline: 0988 123 456</div>
        </div>
        <a href="#" class="apply-btn">ỨNG TUYỂN NGAY ➔</a>
      </div>
    </div>
  </div>
</body>
</html>"""

# ==================================================================================================
# CASE 4: FOOD / BEVERAGE MENU WITH DOTTED LEADERS
# ==================================================================================================
HTML_MENU = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Food Menu</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center;
      background: #0f0a07; font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .poster {
      position: relative; width: 1024px; height: 1024px; overflow: hidden;
      background: radial-gradient(circle at 50% 30%, #1f140e 0%, #120b08 60%, #080403 100%);
      box-shadow: 0 25px 60px rgba(0,0,0,0.8); padding: 56px;
      display: flex; flex-direction: column; justify-content: space-between;
    }
    .menu-header { text-align: center; }
    .sub-brand { font-size: 14px; font-weight: 700; color: #D97706; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 6px; }
    .menu-title {
      font-family: 'Playfair Display', serif; font-size: 54px; font-weight: 900;
      color: #FFFBEB; letter-spacing: 1px;
    }
    .menu-desc { font-style: italic; font-size: 15px; color: #A8A29E; margin-top: 4px; }

    /* Menu Grid */
    .menu-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin: 30px 0;
    }
    .cat-title {
      font-family: 'Playfair Display', serif; font-size: 24px; font-weight: 700;
      color: #F59E0B; border-bottom: 1px solid rgba(245, 158, 11, 0.3);
      padding-bottom: 8px; margin-bottom: 18px;
    }
    .item-list { display: flex; flex-direction: column; gap: 16px; }
    .menu-row { display: flex; flex-direction: column; gap: 3px; }
    .row-top { display: flex; align-items: baseline; justify-content: space-between; }
    .item-name { font-size: 17px; font-weight: 700; color: #FFFFFF; }
    .dotted-line { flex-grow: 1; border-bottom: 1px dotted rgba(255,255,255,0.3); margin: 0 10px; }
    .item-price { font-family: 'Playfair Display', serif; font-size: 19px; font-weight: 700; color: #F59E0B; }
    .badge-star {
      font-size: 10px; font-weight: 800; background: #EF4444; color: #FFF;
      padding: 2px 6px; border-radius: 4px; margin-left: 6px; text-transform: uppercase;
    }
    .item-ing { font-size: 12.5px; color: #78716C; }

    /* Menu Footer */
    .menu-footer {
      background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.2);
      border-radius: 16px; padding: 16px 28px; display: flex;
      justify-content: space-between; align-items: center;
    }
    .foot-note { font-size: 13.5px; color: #D6D3D1; }
    .foot-hotline { font-weight: 700; color: #F59E0B; font-size: 15px; }
  </style>
</head>
<body>
  <div class="poster">
    <div class="menu-header">
      <div class="sub-brand">ARTISAN DINING EXPERIENCE</div>
      <div class="menu-title">BẾP NHÀ TENDOO</div>
      <div class="menu-desc">Thưởng thức tinh hoa ẩm thực thủ công từ nguyên liệu cao cấp</div>
    </div>

    <div class="menu-grid">
      <!-- Cat 1: Burgers -->
      <div>
        <div class="cat-title">🍔 BÒ NƯỚNG & BURGER</div>
        <div class="item-list">
          <div class="menu-row">
            <div class="row-top">
              <span class="item-name">Smash Burger Phô Mai Chảy <span class="badge-star">BEST SELLER</span></span>
              <span class="dotted-line"></span>
              <span class="item-price">89.000đ</span>
            </div>
            <div class="item-ing">Bò Black Angus xay tươi, phô mai Cheddar kép, sốt bơ nấm</div>
          </div>
          <div class="menu-row">
            <div class="row-top">
              <span class="item-name">Wagyu Truffle Burger <span class="badge-star">CHEF CHOICE</span></span>
              <span class="dotted-line"></span>
              <span class="item-price">149.000đ</span>
            </div>
            <div class="item-ing">Thịt bò Wagyu vân mỡ, sốt nấm Truffle đen thượng hạng</div>
          </div>
          <div class="menu-row">
            <div class="row-top">
              <span class="item-name">Gà Giòn Cay Sốt Mật Ong</span>
              <span class="dotted-line"></span>
              <span class="item-price">79.000đ</span>
            </div>
            <div class="item-ing">Đùi gà rút xương chiên giòn rụm, sốt mật ong mù tạt</div>
          </div>
        </div>
      </div>

      <!-- Cat 2: Drinks & Sides -->
      <div>
        <div class="cat-title">🍹 ĐỒ UỐNG & MÓN ĂN KÈM</div>
        <div class="item-list">
          <div class="menu-row">
            <div class="row-top">
              <span class="item-name">Trà Sữa Trân Châu Nướng <span class="badge-star">HOT</span></span>
              <span class="dotted-line"></span>
              <span class="item-price">49.000đ</span>
            </div>
            <div class="item-ing">Trà đen ủ lạnh, sữa tươi thanh trùng, trân châu đường đen</div>
          </div>
          <div class="menu-row">
            <div class="row-top">
              <span class="item-name">Trà Đào Cam Sả Tươi</span>
              <span class="dotted-line"></span>
              <span class="item-price">45.000đ</span>
            </div>
            <div class="item-ing">Đào miếng giòn ngọt, sả cây tươi, cam vàng thanh mát</div>
          </div>
          <div class="menu-row">
            <div class="row-top">
              <span class="item-name">Khoai Tây Lắc Phô Mai Truffle</span>
              <span class="dotted-line"></span>
              <span class="item-price">39.000đ</span>
            </div>
            <div class="item-ing">Khoai tây Bỉ chiên giòn, bột phô mai Cheddar & mùi thơm nấm</div>
          </div>
        </div>
      </div>
    </div>

    <div class="menu-footer">
      <div class="foot-note">✨ Giảm 10% tổng hóa đơn khi check-in tại quán • Freeship bán kính 3km</div>
      <div class="foot-hotline">📞 Hotline Đặt Bàn: 1800 8198</div>
    </div>
  </div>
</body>
</html>"""


def render_case(name: str, html: str, w: int, h: int):
    html_file = OUTPUT_DIR / f"{name}.html"
    png_file = OUTPUT_DIR / f"{name}.png"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Rendering [{name}] via Playwright Chromium ({w}x{h})...")
    PosterRenderer.render(html_content=html, output_image_path=png_file, width=w, height=h)
    print(f"  ✓ Saved: {png_file.name}")


def main():
    print("=" * 80)
    print("🚀 RENDERING 4 DIVERSE COMMERCIAL USE-CASES VIA PLAYWRIGHT CHROMIUM")
    print("=" * 80)
    render_case("case1_burger_opening", HTML_BURGER, 1024, 1024)
    render_case("case2_pet_spa_feedback", HTML_PET_SPA, 1024, 1024)
    render_case("case3_recruitment_dense_glass", HTML_RECRUITMENT, 1024, 1024)
    render_case("case4_restaurant_menu", HTML_MENU, 1024, 1024)
    print("\n[✓] All 4 diverse cases rendered successfully!")


if __name__ == "__main__":
    main()
