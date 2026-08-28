#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - ACQUIRE REMAINING PRODUCTS WITH REALISTIC LIFESTYLE CONTEXT (NO WHITE BACKGROUND)
====================================================================================================
Script: scripts/acquire_remaining_lifestyle_products.py
Purpose:
    Generates the remaining 18 products for Milestone A using rich, authentic, natural lifestyle
    environmental backgrounds (wooden tables, kitchen counters, gym floors, TV consoles, desks).
    - 0% white background: all scenes have realistic domestic, commercial, or athletic environments.
    - Strictly avoids OpenAI rate limits (9.5s delay + backoff).
====================================================================================================
"""

import base64
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY not found in .env")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=api_key)

# 18 Remaining Products with 100% Non-White Lifestyle Backgrounds
LIFESTYLE_REMAINING_PRODUCTS = [
    # --- Home Appliances (4) ---
    {
        "folder": "home",
        "filename": "31_ban_ui_hoi_nuoc.png",
        "prompt": "A commercial lifestyle product photo of a handheld compact garment steamer iron in minimalist matte white and champagne gold, standing upright on a rustic light wood laundry shelf next to folded clean linen shirts, warm domestic natural ambient light, shallow depth of field",
    },
    {
        "folder": "home",
        "filename": "32_may_xay_sinh_to.png",
        "prompt": "A commercial lifestyle photo of a portable USB personal blender with pastel mint green base and clear cup filled with fresh strawberry smoothie, standing on a bright modern kitchen wooden breakfast bar with sliced fruits nearby, sunny morning daylight",
    },
    {
        "folder": "home",
        "filename": "33_noi_chien_khong_dau.png",
        "prompt": "A sleek modern digital air fryer in matte black with chrome handle, placed on a contemporary kitchen dark granite countertop against a white subway tile backsplash, subtle warm under-cabinet kitchen lighting, sharp commercial product photo",
    },
    {
        "folder": "home",
        "filename": "34_den_ban_led.png",
        "prompt": "A minimalist architectural LED desk lamp in matte white aluminum, illuminated and standing on a solid oak study desk next to an open notebook and ceramic coffee cup, cozy home office evening ambiance",
    },

    # --- FMCG & Packaged Foods (4) ---
    {
        "folder": "fmcg",
        "filename": "36_hop_tra_sen_tay_ho.png",
        "prompt": "A commercial product photo of an authentic Vietnamese West Lake lotus scented green tea decorative tin box with gold embossed lotus flower, resting on a traditional dark bamboo tea tray with two small ceramic teacups, warm serene tea ceremony ambiance",
    },
    {
        "folder": "fmcg",
        "filename": "37_chai_nuoc_mam_phu_quoc.png",
        "prompt": "A commercial product photo of a premium glass bottle of traditional Vietnamese Phu Quoc anchovy fish sauce with dark amber liquid and gold seal, standing on a rustic wooden cutting board in a warm authentic Vietnamese kitchen setting with fresh red chili and garlic beside it",
    },
    {
        "folder": "fmcg",
        "filename": "39_hu_yen_sao_khanh_hoa.png",
        "prompt": "A commercial product photo of a luxury glass jar of Vietnamese bird's nest soup with golden hexagonal lid, resting on an elegant polished red mahogany wooden tray with a small porcelain spoon, warm premium ambient lighting",
    },
    {
        "folder": "fmcg",
        "filename": "40_hop_banh_quy_bo.png",
        "prompt": "A classic circular royal blue embossed butter cookie tin box, resting on a cozy wooden coffee table covered with a lace table runner next to a porcelain teapot, warm afternoon tea setting",
    },

    # --- Viettel Telecom & Smart Devices (5) ---
    {
        "folder": "telecom_viettel",
        "filename": "41_modem_wifi6_viettel.png",
        "prompt": "A commercial product photo of a modern white Viettel Home Wifi 6 mesh router device with vertical antennas and red logo accent, neatly placed on a wooden living room TV entertainment console next to a small potted succulent plant, realistic modern apartment interior",
    },
    {
        "folder": "telecom_viettel",
        "filename": "42_phoi_sim_5g_viettel.png",
        "prompt": "A commercial product photo of a Viettel 5G high-speed SIM card plastic frame in vibrant red and white corporate colors, lying on a dark textured slate work desk beside a modern glass smartphone, clean tech office lighting",
    },
    {
        "folder": "telecom_viettel",
        "filename": "43_smart_camera_viettel.png",
        "prompt": "A commercial product photo of a compact modern smart home security camera in matte white with rotating lens, mounted on a light gray painted living room interior wall shelf, realistic indoor home setting",
    },
    {
        "folder": "telecom_viettel",
        "filename": "44_thiet_bi_v_tracking.png",
        "prompt": "A commercial product photo of a rugged compact black automotive GPS tracking device box with LED indicator lights, resting on the clean leather dashboard of a modern car, realistic automotive interior lighting",
    },
    {
        "folder": "telecom_viettel",
        "filename": "45_hop_tv360_box.png",
        "prompt": "A commercial product photo of a sleek matte black Android TV streaming set-top box and accompanying slim remote controller, resting on a dark wood entertainment media console beside a TV screen, cozy home theater lighting",
    },

    # --- Fitness & Gym (5) ---
    {
        "folder": "fitness",
        "filename": "46_binh_lac_shaker.png",
        "prompt": "A commercial fitness product photo of a gym protein shaker bottle in translucent smoke black, standing upright on a black textured rubber gym floor right next to a heavy cast iron dumbbell, dramatic fitness studio lighting",
    },
    {
        "folder": "fitness",
        "filename": "47_tham_yoga.png",
        "prompt": "A commercial fitness photo of a rolled-up dual-layer TPE yoga mat in muted slate blue with carrying strap, lying on a sunlit natural blonde hardwood yoga studio floor with soft window shadows, serene wellness ambiance",
    },
    {
        "folder": "fitness",
        "filename": "48_gang_tay_gym.png",
        "prompt": "A pair of high-performance black workout weightlifting gym gloves with wrist support wraps, placed naturally on the textured knurled steel barbell in a workout gym, athletic mood lighting",
    },
    {
        "folder": "fitness",
        "filename": "49_con_lan_massage.png",
        "prompt": "A commercial fitness photo of a high-density black textured foam muscle roller for recovery, resting on a clean gym exercise mat near a gym bench, active fitness training background",
    },
    {
        "folder": "fitness",
        "filename": "50_day_nhay_toc_do.png",
        "prompt": "A professional speed jump rope with knurled red aluminum handles and thin steel cable, neatly coiled on an athletic gym wooden floor, dynamic sports photography lighting",
    },
]


def acquire_item(item: dict, target_dir: Path, delay: float = 9.5, overwrite: bool = False) -> bool:
    folder = target_dir / item["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    out_file = folder / item["filename"]

    if out_file.exists() and out_file.stat().st_size > 10000 and not overwrite:
        print(f" [SKIP] Already exists: {item['folder']}/{item['filename']}")
        return True

    print(f" [GEN-LIFESTYLE] {item['folder']}/{item['filename']}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = client.images.generate(
                model="gpt-image-2",
                prompt=item["prompt"],
                quality="low",
                size="1024x1024",
            )
            img_bytes = base64.b64decode(res.data[0].b64_json)
            with open(out_file, "wb") as f:
                f.write(img_bytes)
            print(f"   ===> [OK] Saved ({len(img_bytes) // 1024} KB) -> {out_file.name}")
            time.sleep(delay)
            return True
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                print(f"   ===> [429 RATE LIMIT] Backoff 12s (attempt {attempt+1}/{max_retries})...")
                time.sleep(12.0)
            else:
                print(f"   ===> [FAIL] Error on {item['filename']}: {e}")
                return False
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tendoo AI - Lifestyle Products Acquisition Engine")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files to ensure lifestyle backgrounds")
    parser.add_argument("--delay", type=float, default=9.5, help="Delay in seconds between requests (default: 9.5s)")
    args = parser.parse_args()

    target_dir = PROJECT_ROOT / "data" / "products"
    target_dir.mkdir(parents=True, exist_ok=True)

    items = LIFESTYLE_REMAINING_PRODUCTS
    print("=" * 90)
    print(f" [*] TENDOO AI - LIFESTYLE PRODUCTS ACQUISITION ENGINE (TOTAL: {len(items)} REMAINING)")
    print(" [*] 100% NON-WHITE ENVIRONMENTAL BACKGROUNDS (WOOD, MARBLE, GYM, LIVING ROOM)")
    print(f" [*] OVERWRITE MODE: {args.overwrite}")
    print("=" * 90)

    success = 0
    for idx, item in enumerate(items, 1):
        print(f"[{idx:02d}/{len(items)}]", end="")
        ok = acquire_item(item, target_dir, delay=args.delay, overwrite=args.overwrite)
        if ok:
            success += 1

    print("\n" + "=" * 90)
    print(f" [*] FINISHED: {success}/{len(items)} lifestyle products generated in: {target_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
