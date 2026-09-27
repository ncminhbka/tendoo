#!/usr/bin/env python3
"""
scripts/score_blind_review.py -- tổng hợp phiếu ĐÁNH GIÁ MÙ (scripts/build_blind_review.py).

Đọc references/blind_review/key.json (cặp nào poster hệ thống nằm bên nào) + mọi file phiếu trong
references/blind_review/results/*.json -> tỉ lệ "hệ thống thắng / ngang / thua" chung và theo nhu cầu,
kèm khoảng tin cậy Wilson 95% (ít phiếu -> khoảng rộng: đọc khoảng, đừng đọc con số giữa).

  python scripts/score_blind_review.py [--json out.json]

Chỉ số chính ROADMAP §10.10: tỉ lệ "không thua" = (thắng + ngang) / tổng -- poster hệ thống "không phân biệt được /
ưng hơn" so với poster designer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIR = PROJECT_ROOT / "references" / "blind_review"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def tally(votes: list[str]) -> dict:
    n = len(votes)
    win, tie = votes.count("win"), votes.count("tie")
    lo, hi = wilson(win + tie, n)
    return {"n": n, "win": win, "tie": tie, "loss": n - win - tie,
            "not_worse": round((win + tie) / n, 3) if n else None, "not_worse_ci95": [round(lo, 3), round(hi, 3)]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    key = json.loads((DIR / "key.json").read_text(encoding="utf-8"))["pairs"]
    files = sorted((DIR / "results").glob("*.json"))
    if not files:
        print(f"Chưa có phiếu nào trong {DIR / 'results'}")
        return 1
    allv, by_need, by_rater = [], defaultdict(list), defaultdict(list)
    for f in files:
        sheet = json.loads(f.read_text(encoding="utf-8"))
        for pid, v in sheet["votes"].items():
            if pid not in key:
                continue  # phiếu của vòng cũ (bộ cặp khác)
            k = key[pid]
            res = "tie" if v["v"] == "tie" else ("win" if v["v"] == k["ours"] else "loss")
            allv.append(res)
            by_need[k["need"]].append(res)
            by_rater[sheet.get("rater", f.stem)].append(res)
    report = {"files": len(files), "overall": tally(allv),
              "by_need": {n: tally(v) for n, v in sorted(by_need.items())},
              "by_rater": {r: tally(v) for r, v in sorted(by_rater.items())}}
    o = report["overall"]
    print(f"{o['n']} phiếu từ {len(files)} người chấm: hệ thống thắng {o['win']}, ngang {o['tie']}, thua {o['loss']}")
    print(f"KHÔNG THUA poster designer: {o['not_worse']:.0%} (khoảng tin cậy 95%: {o['not_worse_ci95'][0]:.0%}–{o['not_worse_ci95'][1]:.0%})")
    for n, t in report["by_need"].items():
        print(f"  {n:<14} {t['n']:>3} phiếu  thắng {t['win']:>2} ngang {t['tie']:>2} thua {t['loss']:>2}  không thua {t['not_worse']:.0%}")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
