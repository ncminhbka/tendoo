#!/usr/bin/env python3
"""
scripts/run_real_llm.py -- GĐ 3R (ROADMAP §7): đối chiếu hợp đồng prompt với LLM THẬT (OpenAI API)
và thử chữ trên ảnh nền THẬT (mô hình ảnh OpenAI thay FLUX -- chỉ để kiểm tầng chữ; FLUX vẫn là
mô hình nền chính trên máy chủ).

Khoá đọc từ OPENAI_API_KEY trong .env (không in ra, không ghi vào kết quả). Mỗi lời gọi tốn phí:
plan được cache theo (model, brief), ảnh cache theo (model ảnh, brief, scene_prompt).

  PYTHONPATH=src python scripts/run_real_llm.py --model gpt-5.4-mini            # chỉ plan + Cổng
  PYTHONPATH=src python scripts/run_real_llm.py --model gpt-5.4-mini --images   # + ảnh nền thật + squint
Kết quả: output_probe/real_llm/<model>/ (plans.json, report.json, poster_*.png)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SIZES = {"1:1": (1024, 1024), "9:16": (576, 1024), "16:9": (1024, 576), "4:5": (816, 1024)}
IMG_REQ = {"1:1": "1024x1024", "9:16": "1024x1536", "16:9": "1536x1024", "4:5": "1024x1536"}


def _openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    env = PROJECT_ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("Thiếu OPENAI_API_KEY (.env)")
    return key


def composition_hint(plan, aspect: str) -> str:
    """Mô phỏng corridor của velocity blending (mô hình ảnh OpenAI không nhận mask): mô tả bằng lời
    vùng chữ của template (lấy từ geometry.get_zones) cần để trống, mềm, ít chi tiết."""
    from tendoo_v3.geometry import get_zones

    if plan.maskless:
        return " Low-detail background across the whole frame, no dominant subject."
    w, h = SIZES[aspect]
    parts = []
    for x1, y1, x2, y2 in get_zones(plan.template, w, h, orientation=plan.orientation).values():
        horiz = "left" if x2 <= w * 0.6 else ("right" if x1 >= w * 0.4 else "")
        vert = "top" if y2 <= h * 0.6 else ("bottom" if y1 >= h * 0.4 else "")
        where = " ".join(v for v in (vert, horiz) if v) or "center"
        parts.append(f"the {where} {round((x2 - x1) * (y2 - y1) / (w * h) * 100)}% area")
    return (" Composition: keep " + ", ".join(parts) + " of the frame as calm, softly blurred, uncluttered negative space"
            " (smooth tones continuing the scene); place the main subject in the remaining area.")


def generate_background(scene_prompt: str, aspect: str, key: str, model: str, cache: Path) -> str:
    """Ảnh nền không chữ từ scene_prompt -> data URI đúng kích thước khung (cover-crop)."""
    import requests
    from PIL import Image

    h = hashlib.sha1(f"{model}|{aspect}|{scene_prompt}".encode("utf-8")).hexdigest()[:16]
    f = cache / f"bg_{h}.png"
    if not f.exists():
        prompt = (scene_prompt + ". Absolutely no text, letters, numbers, logos or watermarks anywhere in the image.")
        r = requests.post("https://api.openai.com/v1/images/generations", timeout=240,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "prompt": prompt, "size": IMG_REQ[aspect], "n": 1})
        r.raise_for_status()
        f.write_bytes(base64.b64decode(r.json()["data"][0]["b64_json"]))
    img = Image.open(f).convert("RGB")
    w, hgt = SIZES[aspect]
    sc = max(w / img.width, hgt / img.height)
    img = img.resize((round(img.width * sc), round(img.height * sc)))
    x, y = (img.width - w) // 2, (img.height - hgt) // 2
    buf = io.BytesIO()
    img.crop((x, y, x + w, y + hgt)).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--image-model", default="gpt-image-2")
    ap.add_argument("--images", action="store_true", help="sinh ảnh nền thật + đo squint (tốn phí ảnh)")
    ap.add_argument("--only", default=None, help="chỉ chạy brief có id chứa chuỗi này")
    ap.add_argument("--tag", default="", help="hậu tố thư mục kết quả (vd v2 sau khi sửa prompt) -- plan cache riêng")
    args = ap.parse_args()
    key = _openai_key()
    os.environ.update({"TENDOO_V3_LLM_BACKEND": "api", "TENDOO_V3_LLM_BASE_URL": "https://api.openai.com/v1",
                       "TENDOO_V3_LLM_API_KEY": key, "TENDOO_V3_LLM_MODEL": args.model, "TENDOO_V3_LLM_TIMEOUT_S": "180"})

    from tendoo_v3.llm_planner import generate_creative_plan  # đọc biến môi trường lúc import
    from tendoo_v3.validators import check_plan

    out = PROJECT_ROOT / "output_probe" / "real_llm" / (args.model + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    cache = PROJECT_ROOT / "output_probe" / "real_llm" / "_img_cache"
    cache.mkdir(parents=True, exist_ok=True)
    briefs = json.loads((PROJECT_ROOT / "tests" / "real_briefs.json").read_text(encoding="utf-8"))
    if args.only:
        briefs = [b for b in briefs if args.only in b["id"]]
    plans_file = out / "plans.json"
    plans = json.loads(plans_file.read_text(encoding="utf-8")) if plans_file.exists() else {}

    rows = []
    for b in briefs:
        if b["id"] not in plans:
            t0 = time.time()
            plan, trace = generate_creative_plan(b["form"], prompt=b["prompt"], aspect_ratio=b["aspect"], return_debug=True)
            raw = (trace.get("output") or {}).get("extracted_json") or {}
            plans[b["id"]] = {"plan": plan.to_dict(), "status": trace.get("status"), "mode": trace.get("mode"),
                              "latency": round(time.time() - t0, 1), "raw_hero_parts": raw.get("hero_parts"),
                              "gate3": trace.get("gate3", []), "error": str(trace.get("error") or "")[:300]}
            plans_file.write_text(json.dumps(plans, ensure_ascii=False, indent=1), encoding="utf-8")
        rec = plans[b["id"]]
        from tendoo_v3.schema import TendooCreativePlan

        plan = TendooCreativePlan.from_dict(json.loads(json.dumps(rec["plan"])))
        raw_parts = rec.get("raw_hero_parts")
        row = {"id": b["id"], "status": rec["status"], "template": plan.template, "intent": plan.visual_intent,
               "hero": plan.hero, "raw_parts": bool(raw_parts), "parts_kept": bool(plan.hero_parts),
               "components": {k: getattr(plan, k) for k in ("badge_style", "stat_style", "decor") if getattr(plan, k)},
               "maskless": plan.maskless, "gate2": check_plan(plan), "gate3": rec.get("gate3"), "latency": rec.get("latency")}
        rows.append((b, plan, row))

    if args.images:
        from playwright.sync_api import sync_playwright

        from probe_type_hierarchy import measure_plan
        from tendoo_v3.renderer import build_template_html

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            for b, plan, row in rows:
                w, h = SIZES[b["aspect"]]
                try:
                    bg = generate_background(plan.scene_prompt + composition_hint(plan, b["aspect"]), b["aspect"], key, args.image_model, cache)
                except Exception as e:  # ảnh lỗi không được chặn cả lượt
                    row["image_error"] = f"{type(e).__name__}: {str(e)[:160]}"
                    continue
                r = measure_plan(page, plan, w, h, with_bg=True, bg_override=bg)
                row.update({k: r.get(k) for k in ("contrast", "contrast_target", "c1_anchor", "c2_no_wall", "c3_bg", "c4_no_loss", "c5_phone", "squint_pass", "pass5", "phone_min")})
                page.set_viewport_size({"width": w, "height": h})
                page.set_content(build_template_html(plan, bg, w, h), wait_until="load")
                page.wait_for_function("window.__tendooAutofitDone === true", timeout=8000)
                page.screenshot(path=str(out / f"poster_{b['id']}.png"))
            browser.close()

    report = [r for _, _, r in rows]
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ok = [r for r in report if r["status"] == "success"]
    print(f"{args.model}: {len(ok)}/{len(report)} plan từ LLM (còn lại rơi về dự phòng)")
    print(f"  hero_parts LLM trả: {sum(r['raw_parts'] for r in ok)}/{len(ok)}  qua Cổng 1: {sum(r['parts_kept'] for r in ok)}/{sum(r['raw_parts'] for r in ok)}")
    print(f"  plan không vấn đề Cổng 2: {sum(not r['gate2'] for r in ok)}/{len(ok)}   Cổng 3 đổi/báo: {sum(bool(r['gate3']) for r in ok)}")
    for r in report:
        if r["status"] != "success":
            print(f"  ! {r['id']}: {plans[r['id']].get('error')}")
        sq = "" if "c1_anchor" not in r else ("  squint " + "".join("✓" if r.get(k) else "✗" for k in ("c1_anchor", "c2_no_wall", "c3_bg", "c4_no_loss", "c5_phone")) + f" tp={r.get('contrast')}")
        print(f"  {r['id']:<26} {r['status'] or '':<8} {r['template']:<22} {str(r['intent']):<18} parts={'✓' if r['parts_kept'] else ('✗' if r['raw_parts'] else '-')}"
              f" {r['components'] or ''}{' maskless' if r['maskless'] else ''}{sq}{('  G2: ' + '; '.join(r['gate2'])[:120]) if r['gate2'] else ''}{('  ' + r['image_error']) if r.get('image_error') else ''}")


if __name__ == "__main__":
    main()
