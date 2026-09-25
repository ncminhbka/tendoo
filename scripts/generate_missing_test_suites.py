#!/usr/bin/env python3
"""
scripts/generate_missing_test_suites.py

Sinh toàn diện các test suites còn thiếu cho 14 commercial templates của Tendoo v3:
1. test_split_right_suite.json (16-case standard suite) dựa trên test_split_left_suite.json.
2. Sinh các sparse / no-data test suites chuẩn hóa (bao gồm: no-qr, no-store, no-chrome, minimal hero-only)
   cho các template còn thiếu:
   - split_left
   - split_right
   - diagonal_slash
   - customer_feedback_card
   - menu_price_board
   - before_after_split
   - l_frame_showcase
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"

def generate_split_right_standard_suite():
    left_suite_path = TESTS_DIR / "test_split_left_suite.json"
    right_suite_path = TESTS_DIR / "test_split_right_suite.json"
    
    if not left_suite_path.exists():
        print(f"[-] ERROR: {left_suite_path} not found!")
        return
        
    with open(left_suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    right_cases = []
    for c in cases:
        rc = json.loads(json.dumps(c))
        rc["id"] = rc["id"].replace("sl_", "sr_")
        if "plan" in rc:
            rc["plan"]["template"] = "split_right"
            if "corridor_prompt" in rc["plan"]:
                rc["plan"]["corridor_prompt"] = rc["plan"]["corridor_prompt"].replace("on left", "on right").replace("on the left", "on the right")
        else:
            rc["template"] = "split_right"
        right_cases.append(rc)
        
    with open(right_suite_path, "w", encoding="utf-8") as f:
        json.dump(right_cases, f, ensure_ascii=False, indent=2)
    print(f"[+] Generated {right_suite_path} ({len(right_cases)} cases)")

def _get_target_dict(case_obj):
    """Trả về dict chứa các trường (plan hoặc chính case_obj)."""
    if "plan" in case_obj and isinstance(case_obj["plan"], dict):
        return case_obj["plan"]
    return case_obj

def generate_sparse_suite_for_template(template_name: str, standard_suite_filename: str, output_sparse_filename: str):
    std_path = TESTS_DIR / standard_suite_filename
    out_path = TESTS_DIR / output_sparse_filename
    
    if not std_path.exists():
        print(f"[-] ERROR: Standard suite {std_path} not found for {template_name}!")
        return
        
    with open(std_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    # Lọc lấy đại diện 4 tỉ lệ (1:1, 9:16, 16:9, 4:5)
    ratios_cases = {}
    for c in cases:
        w, h = c["width"], c["height"]
        key = f"{w}x{h}"
        if key not in ratios_cases:
            ratios_cases[key] = []
        ratios_cases[key].append(c)
        
    sparse_cases = []
    ratio_keys = ["1024x1024", "576x1024", "1024x576", "816x1024"]
    
    for rkey in ratio_keys:
        candidates = ratios_cases.get(rkey, [])
        if not candidates:
            continue
        base_c = candidates[0]
        w, h = base_c["width"], base_c["height"]
        r_label = "1x1" if w == h else ("9x16" if w < 600 else ("16x9" if w > 1000 else "4x5"))
        
        # Test Case 1: Tắt QR (giữ store)
        c1 = json.loads(json.dumps(base_c))
        c1["id"] = f"{template_name}_sparse_{r_label}_no_qr"
        c1["title"] = f"[{template_name.upper()}] {r_label} - Tắt QR Code (Chỉ Giữ Store Info)"
        t1 = _get_target_dict(c1)
        t1["template"] = template_name
        t1["qr_code"] = None
        t1["qr_label"] = None
        if not t1.get("store_info") and not t1.get("store_items"):
            t1["store_info"] = "Hotline: 1900 8888 • Hệ thống toàn quốc"
        sparse_cases.append(c1)
        
        # Test Case 2: Tắt Store (giữ QR)
        c2 = json.loads(json.dumps(base_c))
        c2["id"] = f"{template_name}_sparse_{r_label}_no_store"
        c2["title"] = f"[{template_name.upper()}] {r_label} - Tắt Thông Tin Cửa Hàng (Chỉ Giữ QR)"
        t2 = _get_target_dict(c2)
        t2["template"] = template_name
        t2["store_info"] = None
        t2["store_items"] = None
        t2["tag_left"] = None
        t2["tag_right"] = None
        if not t2.get("qr_code"):
            t2["qr_code"] = f"https://tendoo.ai/{template_name}"
            t2["qr_label"] = "QUÉT MÃ MUA"
        sparse_cases.append(c2)
        
        # Test Case 3: Zero Chrome / Minimal Hero Only
        c3 = json.loads(json.dumps(base_c))
        c3["id"] = f"{template_name}_sparse_{r_label}_minimal_hero_only"
        c3["title"] = f"[{template_name.upper()}] {r_label} - Hero Tối Giản Cực Đại (Không Store, Không QR, Không Subhead)"
        t3 = _get_target_dict(c3)
        t3["template"] = template_name
        t3["subhead"] = ""
        t3["badge"] = ""
        t3["tag_left"] = None
        t3["tag_right"] = None
        t3["extra_texts"] = []
        t3["cta"] = ""
        t3["store_info"] = None
        t3["store_items"] = None
        t3["qr_code"] = None
        t3["qr_label"] = None
        sparse_cases.append(c3)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sparse_cases, f, ensure_ascii=False, indent=2)
    print(f"[+] Generated {out_path} ({len(sparse_cases)} cases)")

def main():
    print("=== SYNCHRONIZING & GENERATING MISSING TEST SUITES ===")
    
    # 1. Standard suite for split_right
    generate_split_right_standard_suite()
    
    # 2. Missing sparse suites
    templates_needing_sparse = [
        ("split_left", "test_split_left_suite.json", "test_split_left_sparse_suite.json"),
        ("split_right", "test_split_right_suite.json", "test_split_right_sparse_suite.json"),
        ("diagonal_slash", "test_diagonal_slash_suite.json", "test_diagonal_slash_sparse_suite.json"),
        ("customer_feedback_card", "test_customer_feedback_card_suite.json", "test_customer_feedback_card_sparse_suite.json"),
        ("menu_price_board", "test_menu_price_board_suite.json", "test_menu_price_board_sparse_suite.json"),
        ("before_after_split", "test_before_after_split_suite.json", "test_before_after_split_sparse_suite.json"),
        ("l_frame_showcase", "test_l_frame_showcase_suite.json", "test_l_frame_showcase_sparse_suite.json"),
    ]
    
    for tpl, std_file, out_file in templates_needing_sparse:
        generate_sparse_suite_for_template(tpl, std_file, out_file)
        
    print("\n[OK] All test suites synchronized!")

if __name__ == "__main__":
    main()
