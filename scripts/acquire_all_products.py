#!/usr/bin/env python3
"""
====================================================================================================
TENDOO AI - AUTOMATED PRODUCT PACKSHOT ACQUISITION ENGINE (50 TARGET PRODUCTS)
====================================================================================================
Script: scripts/acquire_all_products.py
Purpose:
    Automatically synthesizes / downloads pure 1024x1024 commercial studio packshots on seamless
    solid white background for the 50 Milestone A reference products.
    - Uses OpenAI gpt-image-2 (quality="low") for flawless, high-contrast, zero-watermark packshots.
    - Automatically organizes into data/products/{cosmetics, fnb, tech, fashion, home, fmcg, telecom_viettel, fitness}/
    - Skips existing files so user can overwrite any individual file with manual images.
====================================================================================================
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY not found in .env")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=api_key)

# 50 Products Registry
PRODUCTS_REGISTRY = [
    # 1. Cosmetics (8)
    {
        "folder": "cosmetics",
        "filename": "01_nuoc_hoa_luxury.png",
        "prompt": "A commercial studio product packshot of a luxury amber glass perfume bottle with gold metallic cap, centered on pure solid white background, sharp focus, 8k product photography",
    },
    {
        "folder": "cosmetics",
        "filename": "02_serum_duong_am.png",
        "prompt": "A luxury cosmetic glass dropper serum bottle, clean modern aesthetic, centered product photography on pure seamless solid white background, high-end commercial packaging, studio lighting, sharp focus",
    },
    {
        "folder": "cosmetics",
        "filename": "03_kem_duong_da.png",
        "prompt": "A commercial studio product packshot of a round luxury facial moisturizer cream jar with metallic rose gold lid, centered on pure seamless white background, elegant commercial cosmetic packaging",
    },
    {
        "folder": "cosmetics",
        "filename": "04_son_moi_matte.png",
        "prompt": "A sleek luxury matte red lipstick in a square black and gold tube, slightly twisted up showing smooth vibrant red bullet, centered on pure solid white background, studio commercial lighting",
    },
    {
        "folder": "cosmetics",
        "filename": "05_kem_chong_nang.png",
        "prompt": "A commercial studio packshot of an upright sunscreen tube, clean modern white and yellow bottle design, centered on seamless pure white background, crisp cosmetic packaging",
    },
    {
        "folder": "cosmetics",
        "filename": "06_sua_rua_mat.png",
        "prompt": "A commercial studio packshot of a modern facial cleanser tube, refreshing soft pastel blue and white bottle, centered on seamless pure solid white background, commercial lighting",
    },
    {
        "folder": "cosmetics",
        "filename": "07_phan_nuoc_cushion.png",
        "prompt": "A round luxury cosmetic cushion compact foundation case, slightly open showing clean puff and mirror, centered on pure seamless white background, high-end beauty product shot",
    },
    {
        "folder": "cosmetics",
        "filename": "08_dau_goi_dau.png",
        "prompt": "A luxury modern shampoo pump bottle, rich amber translucent plastic with black pump dispenser, centered on pure solid white background, premium salon hair care packshot",
    },

    # 2. FnB & Beverages (7)
    {
        "folder": "fnb",
        "filename": "09_phin_cafe_nhom.png",
        "prompt": "A commercial studio packshot of a traditional Vietnamese aluminum coffee phin filter set, clean polished metal, centered on pure solid white background, sharp studio photography",
    },
    {
        "folder": "fnb",
        "filename": "10_tui_cafe_rang_moc.png",
        "prompt": "A commercial packshot of a stand-up kraft paper pouch bag of whole bean artisan coffee, elegant minimal typography, centered on pure seamless solid white background",
    },
    {
        "folder": "fnb",
        "filename": "11_lon_nuoc_tang_luc.png",
        "prompt": "A commercial studio packshot of a slim 250ml metallic aluminum energy drink can, dynamic modern graphic design, centered on pure seamless solid white background, crisp reflections",
    },
    {
        "folder": "fnb",
        "filename": "12_chai_tra_xanh.png",
        "prompt": "A commercial packshot of a 500ml clear PET bottle of green tea with clear refreshing golden-green tea liquid inside, clean modern label, centered on pure solid white background",
    },
    {
        "folder": "fnb",
        "filename": "13_hop_sua_hat.png",
        "prompt": "A commercial packshot of a 1-liter carton of organic almond milk with screw cap, fresh natural packaging design, centered on pure seamless solid white background",
    },
    {
        "folder": "fnb",
        "filename": "14_lon_bia_craft.png",
        "prompt": "A commercial studio packshot of a 330ml craft beer aluminum can featuring artistic botanical illustration label, centered on pure seamless solid white background",
    },
    {
        "folder": "fnb",
        "filename": "15_chai_ruou_vang.png",
        "prompt": "A commercial studio packshot of a tall Bordeaux red wine glass bottle with deep red foil capsule and elegant vintage paper label, centered on pure solid white background",
    },

    # 3. Tech & Gadgets (7)
    {
        "folder": "tech",
        "filename": "16_tai_nghe_tws.png",
        "prompt": "A commercial studio packshot of sleek white true wireless earbuds resting in an open charging case, centered on pure seamless solid white background, high-end Apple style lighting",
    },
    {
        "folder": "tech",
        "filename": "17_smartwatch.png",
        "prompt": "A commercial studio packshot of a modern smartwatch with black fluoroelastomer strap and vibrant curved AMOLED display, centered on pure seamless solid white background",
    },
    {
        "folder": "tech",
        "filename": "18_loa_bluetooth.png",
        "prompt": "A commercial studio packshot of a compact portable cylindrical bluetooth speaker with fabric mesh grill and silicone accents, centered on pure seamless solid white background",
    },
    {
        "folder": "tech",
        "filename": "19_chuot_gaming.png",
        "prompt": "A commercial studio packshot of an ergonomic matte black wireless gaming mouse with subtle RGB rim lighting, centered on pure seamless solid white background",
    },
    {
        "folder": "tech",
        "filename": "20_ban_phim_co.png",
        "prompt": "A commercial studio packshot of a compact 75% mechanical wireless keyboard with retro two-tone PBT keycaps and aluminum frame, top-angled view on pure solid white background",
    },
    {
        "folder": "tech",
        "filename": "21_sac_du_phong.png",
        "prompt": "A commercial studio packshot of a slim anodized aluminum portable power bank with digital LED battery percentage display, centered on pure seamless solid white background",
    },
    {
        "folder": "tech",
        "filename": "22_tay_cam_game.png",
        "prompt": "A commercial studio packshot of a modern ergonomic wireless gamepad controller in matte white with textured grips, centered on pure seamless solid white background",
    },

    # 4. Fashion & Accessories (6)
    {
        "folder": "fashion",
        "filename": "23_giay_sneaker_bitis.jpeg",
        "prompt": "",  # Already exists
    },
    {
        "folder": "fashion",
        "filename": "24_kinh_mat_thoi_trang.png",
        "prompt": "A commercial studio packshot of luxury designer aviator sunglasses with gold titanium frame and gradient dark lenses, angled view on pure seamless solid white background",
    },
    {
        "folder": "fashion",
        "filename": "25_dong_ho_kim_loai.png",
        "prompt": "A luxury Swiss men's chronograph wristwatch with stainless steel oyster bracelet and deep blue sunray dial, centered on pure seamless solid white background, sharp studio lighting",
    },
    {
        "folder": "fashion",
        "filename": "26_tui_xach_da.png",
        "prompt": "A high-end luxury women's structured leather handbag in warm cognac brown with gold hardware clasp, centered on pure seamless solid white background, studio photography",
    },
    {
        "folder": "fashion",
        "filename": "27_vi_da_nam.png",
        "prompt": "A sleek luxury men's bifold wallet made of full-grain black leather with fine stitching, slightly opened on pure seamless solid white background, studio commercial shot",
    },
    {
        "folder": "fashion",
        "filename": "28_non_la_viet_nam.png",
        "prompt": "A commercial studio packshot of an authentic traditional Vietnamese conical palm leaf hat (Non La), pristine natural woven texture, angled on pure solid white background",
    },

    # 5. Home Appliances (6)
    {
        "folder": "home",
        "filename": "29_binh_giu_nhiet.png",
        "prompt": "A modern Scandinavian stainless steel thermal water bottle with natural bamboo wooden cap, powder-coated matte sage green, centered on pure seamless solid white background",
    },
    {
        "folder": "home",
        "filename": "30_may_say_toc.png",
        "prompt": "A commercial studio packshot of an ultra-modern ionic hair dryer in matte iron and fuchsia pink, high-tech hollow cylindrical head, centered on pure solid white background",
    },
    {
        "folder": "home",
        "filename": "31_ban_ui_hoi_nuoc.png",
        "prompt": "A handheld compact garment steamer iron in minimalist matte white and champagne gold, upright on pure seamless solid white background, commercial home appliance packshot",
    },
    {
        "folder": "home",
        "filename": "32_may_xay_sinh_to.png",
        "prompt": "A portable USB personal blender bottle with clear Tritan cup and pastel mint green motor base, centered on pure seamless solid white background, commercial studio lighting",
    },
    {
        "folder": "home",
        "filename": "33_noi_chien_khong_dau.png",
        "prompt": "A sleek modern digital air fryer in matte black with chrome handle and touchscreen display, centered on pure seamless solid white background, premium kitchen appliance shot",
    },
    {
        "folder": "home",
        "filename": "34_den_ban_led.png",
        "prompt": "A minimalist architectural LED desk lamp in matte white aluminum, flexible slim neck and touch control dimmer base, upright on pure seamless solid white background",
    },

    # 6. FMCG & Packaged Food (6)
    {
        "folder": "fmcg",
        "filename": "35_mi_hao_hao.jpg",
        "prompt": "",  # Already exists
    },
    {
        "folder": "fmcg",
        "filename": "36_hop_tra_sen_tay_ho.png",
        "prompt": "A commercial studio packshot of an authentic Vietnamese West Lake lotus scented green tea decorative tin box with gold embossed lotus flower, centered on pure solid white background",
    },
    {
        "folder": "fmcg",
        "filename": "37_chai_nuoc_mam_phu_quoc.png",
        "prompt": "A commercial studio packshot of a premium glass bottle of traditional Vietnamese Phu Quoc anchovy fish sauce with dark amber liquid and gold seal, centered on pure white background",
    },
    {
        "folder": "fmcg",
        "filename": "38_hop_cao_sao_vang.png",
        "prompt": "A commercial studio product packshot of the iconic Vietnamese Golden Star Balm small red circular tin can (Cao Sao Vang) with yellow five-pointed star, centered on pure solid white background",
    },
    {
        "folder": "fmcg",
        "filename": "39_hu_yen_sao_khanh_hoa.png",
        "prompt": "A commercial studio packshot of a luxury glass jar of Vietnamese bird's nest soup with golden hexagonal lid, clear rich natural texture, centered on pure solid white background",
    },
    {
        "folder": "fmcg",
        "filename": "40_hop_banh_quy_bo.png",
        "prompt": "A commercial packshot of a classic circular royal blue embossed butter cookie tin box, centered on pure seamless solid white background, studio commercial packaging photography",
    },

    # 7. Viettel Telecom (5)
    {
        "folder": "telecom_viettel",
        "filename": "41_modem_wifi6_viettel.png",
        "prompt": "A commercial studio packshot of a high-tech modern white Home Wifi 6 mesh router device with vertical antennas and subtle red logo accent, centered on pure solid white background",
    },
    {
        "folder": "telecom_viettel",
        "filename": "42_phoi_sim_5g_viettel.png",
        "prompt": "A commercial studio packshot of a 5G high-speed mobile network plastic SIM card holder frame in bright red and white corporate colors, centered on pure seamless solid white background",
    },
    {
        "folder": "telecom_viettel",
        "filename": "43_smart_camera_viettel.png",
        "prompt": "A commercial studio packshot of a compact modern smart home security camera in matte white with glossy black 360-degree rotating lens, centered on pure solid white background",
    },
    {
        "folder": "telecom_viettel",
        "filename": "44_thiet_bi_v_tracking.png",
        "prompt": "A commercial studio packshot of a rugged compact black automotive GPS tracking device box with LED status indicators, centered on pure seamless solid white background",
    },
    {
        "folder": "telecom_viettel",
        "filename": "45_hop_tv360_box.png",
        "prompt": "A commercial studio packshot of a sleek matte black Android TV streaming set-top box and accompanying slim remote controller, centered on pure seamless solid white background",
    },

    # 8. Fitness & Sports (5)
    {
        "folder": "fitness",
        "filename": "46_binh_lac_shaker.png",
        "prompt": "A commercial studio packshot of a gym protein shaker bottle in translucent smoke black with mixing ball and measurement markings, centered on pure seamless solid white background",
    },
    {
        "folder": "fitness",
        "filename": "47_tham_yoga.png",
        "prompt": "A rolled-up premium dual-layer TPE yoga fitness mat tied with carrying strap, muted slate blue and charcoal gray, centered on pure seamless solid white background",
    },
    {
        "folder": "fitness",
        "filename": "48_gang_tay_gym.png",
        "prompt": "A commercial studio packshot of a pair of high-performance black workout weightlifting gym gloves with integrated wrist support wrap, centered on pure solid white background",
    },
    {
        "folder": "fitness",
        "filename": "49_con_lan_massage.png",
        "prompt": "A commercial studio packshot of a high-density black textured foam muscle roller for physical therapy and fitness recovery, centered on pure seamless solid white background",
    },
    {
        "folder": "fitness",
        "filename": "50_day_nhay_toc_do.png",
        "prompt": "A commercial studio packshot of a professional speed jump rope with knurled aluminum handles and thin red steel cable, neatly coiled on pure seamless solid white background",
    },
]


def acquire_product(item: dict, target_dir: Path) -> bool:
    folder = target_dir / item["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    out_file = folder / item["filename"]

    # If already exists with valid size > 10KB, skip
    if out_file.exists() and out_file.stat().st_size > 10000:
        print(f" [SKIP] Already exists ({out_file.stat().st_size // 1024} KB): {item['folder']}/{item['filename']}")
        return True

    if not item["prompt"]:
        print(f" [WARN] No prompt defined for: {item['filename']}")
        return False

    print(f" [GEN] Synthesizing packshot: {item['folder']}/{item['filename']}...")
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
            return True
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                print(f"   ===> [429 RATE LIMIT] Waiting 12s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(12.0)
            else:
                print(f"   ===> [FAIL] Error generating {item['filename']}: {e}")
                return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Tendoo AI - Auto Packshot Acquisition Engine")
    parser.add_argument("--count", type=int, default=50, help="Number of products to process (default: 50)")
    parser.add_argument("--folder", type=str, default=None, help="Filter by specific folder (e.g. cosmetics, fnb)")
    parser.add_argument("--delay", type=float, default=9.5, help="Delay between requests in seconds (default: 9.5s to respect 7 IPM limit)")
    args = parser.parse_args()

    target_dir = PROJECT_ROOT / "data" / "products"
    target_dir.mkdir(parents=True, exist_ok=True)

    items = PRODUCTS_REGISTRY
    if args.folder:
        items = [p for p in items if p["folder"] == args.folder]
    items = items[:args.count]

    print("=" * 90)
    print(f" [*] TENDOO AI - PACKSHOT ACQUISITION ENGINE (TARGET: {len(items)} PRODUCTS, DELAY: {args.delay}s)")
    print("=" * 90)

    success = 0
    for idx, item in enumerate(items, 1):
        print(f"[{idx:02d}/{len(items)}]", end="")
        ok = acquire_product(item, target_dir)
        if ok:
            success += 1
        # Delay only if we actually generated (not skipped)
        folder = target_dir / item["folder"]
        out_file = folder / item["filename"]
        if out_file.exists() and ok:
            time.sleep(args.delay)

    print("\n" + "=" * 90)
    print(f" [*] ACQUISITION COMPLETE: {success}/{len(items)} products available in: {target_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
