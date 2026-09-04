#!/usr/bin/env python3
"""
scripts/probe_layout_safezone_detection.py

==================================================================================================
TENDOO AI - CẤP ĐỘ 2 PROBE: OBJECT DETECTION + MAXIMAL EMPTY RECTANGLE (GPU FEASIBILITY TEST)
==================================================================================================

WHY THIS SCRIPT?
  Answers a specific conditional decision ("nếu 2 GPU A30 cho phép, và độ trễ cho phép"): does
  adding a lightweight open-vocabulary detector (Grounding DINO) to find the hero-title and
  product bounding boxes -- so a Maximal Empty Rectangle (src/tendoo/layout_geometry.py) can dynamically
  place the HTML secondary-content card instead of a fixed safe-zone percentage -- actually fit
  the existing 2x A30 VRAM budget (DiT on cuda:0, VAE+Qwen3 on cuda:1) and run fast enough to be
  worth it?

  This is a FEASIBILITY PROBE, not yet wired into the production template flow: measures latency
  (model load vs per-image inference, separately) and VRAM delta, and saves a visualization (
  detected boxes + the resulting safe rectangle) for visual QA. Wiring the computed rect into
  typography_engine.py's templates is a follow-up once this confirms it's worth the cost.

WHY A DETECTOR, NOT A VLM: this only needs BOUNDING BOXES for a couple of known-vocabulary
classes (the hero text region, the product) -- an open-vocabulary detector solves exactly this in
tens of ms with no risk of the "numerical coordinate blindness" a VLM would have if asked to
output pixel coordinates directly (see AGENTS.md discussion).

WHY Grounding DINO, NOT YOLO-World: YOLO-World's weights are hosted on GitHub Releases, not
Hugging Face Hub -- unusable on a network whose only egress is an HF-repo-mirroring proxy.
Grounding DINO ("IDEA-Research/grounding-dino-tiny") is a genuine Hugging Face Hub model repo,
loaded via standard `transformers.AutoModelForZeroShotObjectDetection.from_pretrained(...)`, so it
downloads via the exact `{repo_id}/resolve/{revision}/{file}` pattern such a proxy forwards.

USAGE (on a network with normal Hugging Face access):
  python scripts/probe_layout_safezone_detection.py --image images/commercial_steps8_g1.5_seed123_576x1024.png \\
      --hero_classes "3d text,title lettering,neon sign" --product_classes "headphones"

USAGE (internal network, HF reachable only through a mirroring proxy):
  1. Mirror the repo to a local directory using whatever internal downloader script already
     works for you (must fetch every file the repo lists, not just the one .safetensors --
     config.json/preprocessor_config.json/tokenizer files etc. are also required):
       python <your_hf_proxy_downloader>.py IDEA-Research/grounding-dino-tiny --out ./models/grounding-dino-tiny
     Do NOT commit that downloader script or its auth token into this repo -- keep it local /
     outside version control, it carries an internal credential.
  2. Point this probe at the local mirror instead of the repo id:
       python scripts/probe_layout_safezone_detection.py --image <path> \\
           --model_dir ./models/grounding-dino-tiny --category feedback --orientation portrait
==================================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def get_gpu_mem_mb(device: str = "cuda:0") -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return -1.0
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    except Exception:
        return -1.0


def run_detection(
    image_path: str,
    hero_classes: List[str],
    product_classes: List[str],
    device: str,
    conf: float = 0.15,
    model_dir: str = "IDEA-Research/grounding-dino-tiny",
) -> Tuple[Dict[str, Any], float, float]:
    """
    Loads Grounding DINO (open-vocab detection via free-text prompts), runs it once.
    Returns (result_dict, load_time_s, infer_time_s).

    NOTE: uses Grounding DINO instead of ultralytics YOLO-World deliberately -- YOLO-World's
    weights are hosted on GitHub Releases, not Hugging Face Hub, which doesn't work through an
    internal network that only has an HF-mirroring proxy. Grounding DINO ("IDEA-Research/
    grounding-dino-tiny") is a genuine, canonical Hugging Face Hub model repo loaded via standard
    `transformers.AutoModelForZeroShotObjectDetection.from_pretrained(...)`, so it downloads via
    the exact `{repo_id}/resolve/{revision}/{file}` URL pattern an HF-repo-mirroring proxy
    forwards. `model_dir` can be either that HF repo id (if the proxy/network allows on-the-fly
    `from_pretrained` calls) OR a local directory the repo was already mirrored into (e.g. via
    your own internal proxy download script) -- transformers treats both identically.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    mem_before = get_gpu_mem_mb(device)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).to(device).eval()
    load_time = time.time() - t0
    mem_after_load = get_gpu_mem_mb(device)

    all_classes = hero_classes + product_classes
    # Grounding DINO expects a single lowercase prompt with classes separated by " . "
    text_prompt = " . ".join(c.lower() for c in all_classes) + " ."
    image = Image.open(image_path).convert("RGB")

    t1 = time.time()
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=conf, text_threshold=conf,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]
    infer_time = time.time() - t1
    mem_after_infer = get_gpu_mem_mb(device)

    hero_set = {c.lower() for c in hero_classes}
    boxes_out = []
    for label, score, box in zip(results["labels"], results["scores"], results["boxes"]):
        cls_name = label.strip()
        xyxy = box.tolist()
        boxes_out.append({
            "class": cls_name,
            "is_hero_class": any(h in cls_name or cls_name in h for h in hero_set),
            "confidence": round(float(score), 3),
            "bbox_xyxy": [round(v, 1) for v in xyxy],
        })

    result = {
        "image": image_path,
        "canvas_size": [image.size[0], image.size[1]],  # (w, h)
        "detections": boxes_out,
        "gpu_mem_mb": {
            "before_load": round(mem_before, 1),
            "after_load": round(mem_after_load, 1),
            "after_infer": round(mem_after_infer, 1),
            "model_footprint_mb": round(mem_after_load - mem_before, 1),
        },
    }
    return result, load_time, infer_time


def visualize(image_path: str, detections: List[Dict[str, Any]], safe_rect, out_path: str) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        color = (255, 80, 80) if d["is_hero_class"] else (80, 160, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 4, max(0, y1 - 18)), f'{d["class"]} {d["confidence"]}', fill=color)

    if safe_rect is not None and safe_rect.area > 0:
        draw.rectangle([safe_rect.x1, safe_rect.y1, safe_rect.x2, safe_rect.y2], outline=(80, 255, 120), width=5)
        draw.text((safe_rect.x1 + 6, safe_rect.y1 + 6), "SAFE RECT (MER)", fill=(80, 255, 120))

    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI Cấp độ 2: detection + MER feasibility probe")
    parser.add_argument("--image", type=str, required=True, help="Path to an already-generated poster image")
    parser.add_argument("--hero_classes", type=str, default="3d text,embossed title,neon sign,lettering",
                         help="Comma-separated open-vocab classes for the hero title region")
    parser.add_argument("--product_classes", type=str, default="product,headphones,bottle,shoe,watch,food",
                         help="Comma-separated open-vocab classes for the product/subject")
    parser.add_argument("--category", type=str, default="feedback",
                         choices=["grand_opening", "feedback", "recruitment", "menu"],
                         help="Which template's bottom-stack search region convention to use")
    parser.add_argument("--orientation", type=str, default=None, choices=["portrait", "landscape"],
                         help="Override orientation (default: inferred from image aspect ratio)")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="cuda:1",
                         help="Device to run the detector on (default cuda:1 -- same GPU as VAE/Qwen3, "
                              "since this runs as post-processing after DiT on cuda:0 is already done)")
    parser.add_argument("--output_dir", type=str, default="output_layout_safezone_probe")
    parser.add_argument("--model_dir", type=str, default="IDEA-Research/grounding-dino-tiny",
                         help="HF repo id, or a local directory the repo was already mirrored into "
                              "(e.g. via an internal HF-proxy downloader) if direct HF access is blocked.")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hero_classes = [c.strip() for c in args.hero_classes.split(",") if c.strip()]
    product_classes = [c.strip() for c in args.product_classes.split(",") if c.strip()]

    print("=" * 100)
    print(" [*] TENDOO AI - CẤP ĐỘ 2: DETECTION + MER FEASIBILITY PROBE")
    print("=" * 100)
    print(f"  Image           : {args.image}")
    print(f"  Model dir/repo  : {args.model_dir}")
    print(f"  Hero classes    : {hero_classes}")
    print(f"  Product classes : {product_classes}")
    print(f"  Device          : {args.device}")

    result, load_time, infer_time = run_detection(
        args.image, hero_classes, product_classes, args.device, conf=args.conf, model_dir=args.model_dir
    )

    print(f"\n  [Timing] Model load  : {load_time:.3f}s (one-time, amortized across a whole batch/session)")
    print(f"  [Timing] Inference   : {infer_time*1000:.1f}ms (per image -- this is the number that matters for per-request latency)")
    print(f"  [VRAM]   Model footprint on {args.device}: {result['gpu_mem_mb']['model_footprint_mb']:.1f} MB")
    print(f"\n  [Detections] {len(result['detections'])} box(es):")
    for d in result["detections"]:
        tag = "HERO" if d["is_hero_class"] else "PRODUCT"
        print(f"    [{tag:7s}] {d['class']:20s} conf={d['confidence']:.2f}  bbox={d['bbox_xyxy']}")

    # Compute MER
    from tendoo.layout_geometry import compute_safe_rect_for_category

    canvas_w, canvas_h = result["canvas_size"]
    orientation = args.orientation or ("portrait" if canvas_h > canvas_w else "landscape")
    forbidden = [tuple(d["bbox_xyxy"]) for d in result["detections"]]
    safe_rect = compute_safe_rect_for_category(canvas_w, canvas_h, orientation, forbidden)

    print(f"\n  [MER] Orientation bucket: {orientation}")
    print(f"  [MER] Safe rect (px)   : ({safe_rect.x1:.0f}, {safe_rect.y1:.0f}) -> ({safe_rect.x2:.0f}, {safe_rect.y2:.0f})")
    print(f"  [MER] Safe rect (% of canvas): {safe_rect.as_css_percent(canvas_w, canvas_h)}")
    print(f"  [MER] Area: {100*safe_rect.area/(canvas_w*canvas_h):.1f}% of canvas")
    if safe_rect.area < 0.15 * canvas_w * canvas_h:
        print("  [!] WARNING: safe rect is small (<15% of canvas) -- template content may need to "
              "shrink/reduce, or this category's search-region assumption doesn't fit this image.")

    result["orientation"] = orientation
    result["safe_rect_px"] = [safe_rect.x1, safe_rect.y1, safe_rect.x2, safe_rect.y2]
    result["safe_rect_css_pct"] = safe_rect.as_css_percent(canvas_w, canvas_h)
    result["timing_s"] = {"load": round(load_time, 3), "inference": round(infer_time, 4)}

    result_file = out_path / "detection_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    viz_path = out_path / "detection_visualization.png"
    visualize(args.image, result["detections"], safe_rect, str(viz_path))

    print(f"\n[✓] Saved: {result_file} , {viz_path}\n")
    print("=" * 100)
    print("  ĐỌC KẾT QUẢ:")
    print("  - Inference < ~100ms/ảnh + VRAM footprint vài trăm MB -> khả thi, đáng wire vào production.")
    print("  - Bounding box detect được có khớp bằng mắt (xem detection_visualization.png) không --")
    print("    nếu 'HERO' box không khớp đúng vùng chữ 3D thật, cần đổi hero_classes hoặc đổi model.")
    print("=" * 100)


if __name__ == "__main__":
    main()
