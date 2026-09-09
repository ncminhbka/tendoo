"""
Cap do 2 (detection + MER) -- CPU-only representative-case probe, round 2.

Round 1 (see memory css-hero-title-overlay-direction.md) only tested ONE composition family
(pastel fashion portrait, product_ad, face dead-center). This script tests 3 MORE compositions,
each picked to stress a different aspect of the detection+MER mechanism, per the user's request
to "test dai dien cac case khac, roi moi ghi nhan diem manh / diem han che":

  A) case_a_product_only  -- product_ad, NO face at all (pure product-on-pedestal shot).
                              Detector: YOLO-World only (no face-detection noise possible).
                              Search region: TOP-anchored (title_position="top-center").
  B) case_b_multi_face    -- feedback-shaped composition (wedding couple), TWO faces, and the
                              couple's bodies span nearly the FULL canvas height -- stress-tests
                              whether the feedback template's fixed bottom-stack zone (which has
                              NO safe_rect hook at all today) actually collides with real content.
                              Detector: OpenCV Haar cascade (faces) -- rendered TWICE: once with
                              the template's current fixed position (as-is, to see if it actually
                              breaks), once with the MER-computed safe rect fed in manually via a
                              quick monkey-patched zone (feedback has no safe_rect param yet).
  C) case_c_already_clear -- product_ad, middle-center title, small distant person + huge empty
                              sky. Sanity/baseline: MER should NOT invent an artificially-narrow
                              safe area when the scene is already clear -- this is the case where
                              Cap do 2 should be a no-op (or very close to one).

No GPU used anywhere in this script.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tendoo.poster_renderer import PosterRenderer  # noqa: E402
from tendoo_legacy.typography_engine import PosterBackgroundAnalyzer, PosterTemplateEngine  # noqa: E402
from tendoo_legacy.layout_geometry import compute_safe_rect_for_category  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output_hero_selector_test"


def detect_product_yoloworld(image_path: Path, classes: list[str]):
    from ultralytics import YOLOWorld
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(classes)
    t0 = time.time()
    results = model.predict(str(image_path), conf=0.15, verbose=False)
    dt = time.time() - t0
    boxes = []
    for r in results:
        for b in r.boxes:
            cls_name = r.names[int(b.cls[0])]
            conf = float(b.conf[0])
            xyxy = [float(v) for v in b.xyxy[0]]
            boxes.append((cls_name, conf, xyxy))
    return boxes, dt


def detect_faces_opencv(image_path: Path):
    import cv2
    t0 = time.time()
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    dt = time.time() - t0
    boxes = [(float(x), float(y), float(x + w), float(y + h)) for (x, y, w, h) in faces]
    return boxes, dt


def render(category: str, brief: dict, bg_path: Path, out_name: str, w: int, h: int):
    analysis = PosterBackgroundAnalyzer.analyze(bg_path)
    html = PosterTemplateEngine.generate_html(
        analysis=analysis, brief=brief, background_image_path=str(bg_path), category=category
    )
    out_path = OUT_DIR / out_name
    PosterRenderer.render(html_content=html, output_image_path=out_path, width=w, height=h)
    return out_path


def case_a():
    print("\n=== CASE A: product_only (product_ad, no face) ===")
    bg = OUT_DIR / "case_a_product_only_bg.png"
    boxes, dt = detect_product_yoloworld(bg, ["smart watch", "watch", "wristwatch"])
    print(f"YOLO-World ({dt:.3f}s): {boxes}")
    W, H = 1536, 1024
    forbidden = [b[2] for b in boxes if b[1] >= 0.20]
    rect = compute_safe_rect_for_category(W, H, "landscape", forbidden, vertical_anchor="top")
    rect_pct = rect.as_css_percent(W, H)
    print(f"forbidden used: {forbidden}")
    print(f"safe_rect: {rect_pct}")

    brief_fixed = {
        "title_text": "SỰ TINH TẾ CỦA SỨC MẠNH", "title_position": "top-center",
        "subtitle_text": "CHỈ CÓ TẠI TENDOO", "style_theme": "metallic",
    }
    render("product_ad", brief_fixed, bg, "case_a_FIXED.png", W, H)

    brief_mer = dict(brief_fixed)
    brief_mer["safe_rect"] = rect_pct
    render("product_ad", brief_mer, bg, "case_a_MER.png", W, H)
    print("wrote case_a_FIXED.png / case_a_MER.png")


def case_b():
    print("\n=== CASE B: multi_face (feedback-shaped, wedding couple) ===")
    bg = OUT_DIR / "case_b_multi_face_bg.png"
    faces, dt = detect_faces_opencv(bg)
    print(f"OpenCV Haar cascade ({dt:.3f}s): {faces}")
    W, H = 1536, 1024

    # 1) Render with the feedback template AS-IS (fixed bottom-stack, no safe_rect hook exists
    #    for this category at all today) -- does it actually collide with real content?
    brief = {
        "brand": "💍 CINEMATIC LOVE",
        "top_badge": "✨ LUXURY WEDDING",
        "quote_text": "Cô dâu chú rể vỡ òa hạnh phúc khi nhận album cưới mang đậm chất điện ảnh!",
        "customer_name": "Anh Minh & Chị Lan",
        "customer_sub": "Khách hàng Gói Cinematic Love",
        "offer_title": "Ưu đãi đặc biệt",
        "offer_desc": "Tặng ảnh cổng tráng gương pha lê",
        "cta_text": "ĐẶT LỊCH NGAY",
    }
    render("feedback", brief, bg, "case_b_FIXED.png", W, H)

    # 2) Where would Cap do 2 actually put things? Bottom-anchored search region (matching the
    #    template's own bottom-stack convention), avoiding detected faces.
    rect = compute_safe_rect_for_category(W, H, "landscape", list(faces), vertical_anchor="bottom")
    rect_pct = rect.as_css_percent(W, H)
    print(f"safe_rect (bottom-anchored region, faces as forbidden): {rect_pct}")
    print("NOTE: feedback template has NO safe_rect param wired in yet (gap #4, see report) --")
    print("      this rect is reported for characterization only, not rendered into a poster.")


def case_c():
    print("\n=== CASE C: already_clear (product_ad, distant hiker + huge empty sky) ===")
    bg = OUT_DIR / "case_c_already_clear_bg.png"
    faces, dt_f = detect_faces_opencv(bg)
    print(f"OpenCV Haar cascade ({dt_f:.3f}s): {faces}")
    boxes, dt_y = detect_product_yoloworld(bg, ["person", "hiker", "backpack"])
    print(f"YOLO-World ({dt_y:.3f}s): {boxes}")
    W, H = 1536, 1024
    forbidden = list(faces) + [b[2] for b in boxes if b[1] >= 0.20]
    rect = compute_safe_rect_for_category(W, H, "landscape", forbidden, vertical_anchor="middle")
    rect_pct = rect.as_css_percent(W, H)
    region_area = (W * 0.945 - W * 0.055) * (H * 0.65 - H * 0.30)
    print(f"forbidden used: {forbidden}")
    print(f"safe_rect: {rect_pct}")
    print(f"region area={region_area:.0f} vs rect area={rect.area:.0f}"
          f"  -> {'NO-OP as expected (clear scene)' if rect.area / region_area > 0.85 else 'MER shrank the region -- unexpected for a clear scene'}")

    brief_fixed = {
        "title_text": "KHÁM PHÁ THẾ GIỚI CÙNG BẠN", "title_position": "middle-center",
        "subtitle_text": "NGƯỜI BẠN ĐỒNG HÀNH TRÊN MỌI NẺO ĐƯỜNG", "style_theme": "plain",
    }
    render("product_ad", brief_fixed, bg, "case_c_FIXED.png", W, H)
    brief_mer = dict(brief_fixed)
    brief_mer["safe_rect"] = rect_pct
    render("product_ad", brief_mer, bg, "case_c_MER.png", W, H)
    print("wrote case_c_FIXED.png / case_c_MER.png")


if __name__ == "__main__":
    case_a()
    case_b()
    case_c()
