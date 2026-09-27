#!/usr/bin/env python3
"""
scripts/render_variants.py -- GĐ 10: N biến thể typography trên CÙNG ảnh nền (không chạy lại diffusion) + logo
thương hiệu + bản in.

  PYTHONPATH=src python scripts/render_variants.py --tag gpt-5.4-mini_v8 --id b01_promo_coffee_1x1 --n 4 \
      [--logo path/to/logo.png] [--print]
  -> output_probe/variants/<id>/variant_{i}.png + sheet.png (+ print_A4_300dpi.png)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "scripts")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataclasses import replace  # noqa: E402

from PIL import Image  # noqa: E402

import run_real_llm as R  # noqa: E402
from tendoo_v3.renderer import export_print, render_plan_to_poster  # noqa: E402
from tendoo_v3.routing import route_template  # noqa: E402
from tendoo_v3.schema import TendooCreativePlan  # noqa: E402
from tendoo_v3.validators import dedupe_plan  # noqa: E402
from tendoo_v3.variants import generate_variants  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--logo", default=None, help="ảnh logo (png/svg) -- brand kit")
    ap.add_argument("--print", dest="do_print", action="store_true", help="xuất thêm bản in A4 300dpi của biến thể 0")
    args = ap.parse_args()
    plans = json.loads((PROJECT_ROOT / "output_probe" / "real_llm" / args.tag / "plans.json").read_text(encoding="utf-8"))
    brief = {b["id"]: b for b in json.loads((PROJECT_ROOT / "tests" / "real_briefs.json").read_text(encoding="utf-8"))}[args.id]
    aspect = brief["aspect"]
    w, h = R.SIZES[aspect]
    plan, _ = route_template(dedupe_plan(TendooCreativePlan.from_dict(plans[args.id]["plan"])), aspect)
    if args.logo:
        mime = "image/svg+xml" if args.logo.endswith(".svg") else "image/png"
        plan = replace(plan, brand_logo=f"data:{mime};base64," + base64.b64encode(Path(args.logo).read_bytes()).decode())
    bg = R.generate_background(plan.scene_prompt + R.composition_hint(plan, aspect), aspect, R._openai_key(), "gpt-image-2",
                               PROJECT_ROOT / "output_probe" / "real_llm" / "_img_cache")
    out = PROJECT_ROOT / "output_probe" / "variants" / args.id
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for i, v in enumerate(generate_variants(plan, args.n)):
        f, _ = render_plan_to_poster(v, bg, out / f"variant_{i}.png", w, h)
        files.append(f)
        print(f"variant_{i}: pack={v.style_pack} lockup={v.lockup} font={v.style.font}")
    T = 420
    sheet = Image.new("RGB", (T * len(files), T), "white")
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((T - 8, T - 8))
        sheet.paste(im, (i * T + (T - im.width) // 2, (T - im.height) // 2))
    sheet.save(out / "sheet.png")
    if args.do_print:
        p, size = export_print(plan, bg, out / "print_A4_300dpi.png", w, h)
        print(f"bản in: {p} {size}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
