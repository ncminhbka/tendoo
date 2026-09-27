"""
src/tendoo_v3/variants.py -- BIẾN THỂ TYPOGRAPHY KHÔNG CHẠY LẠI DIFFUSION (GĐ 10, ROADMAP §10.9).

Người dùng thường muốn "cho xem vài kiểu chữ khác" trên CÙNG một ảnh nền. Diffusion đắt (GPU), chữ rẻ (HTML) ->
giữ nguyên template + orientation (tức giữ vùng chữ = giữ mask/corridor mà nền đã được sinh theo), chỉ đổi lớp chữ:
lockup, style pack (font/màu/hiệu ứng/hoạ tiết), linh kiện. Nội dung chữ KHÔNG đổi (Cổng 1 vẫn đúng).
"""

from __future__ import annotations

import random
import zlib
from dataclasses import replace
from typing import Any, List

from tendoo_v3.catalog import resolve_intent
from tendoo_v3.components import LOCKUP_REQUIRES, suggest_lockup
from tendoo_v3.style_packs import STYLE_PACKS


def _lockups_for(plan: Any) -> List[str]:
    roles = {p.get("role") for p in (plan.hero_parts or [])}
    ok = [name for name, need in LOCKUP_REQUIRES.items() if set(need) <= roles]
    return ["none"] + ok if ok else ["none"]


def generate_variants(plan: Any, n: int = 4) -> List[Any]:
    """[plan gốc, ...n-1 biến thể] -- tất định theo nội dung. Mỗi biến thể khác bản gốc ở ÍT NHẤT style pack hoặc
    lockup; cùng template/orientation/nội dung. Pack chỉ lấy trong danh mục hợp intent (Luật 4)."""
    intent = resolve_intent(plan.template, plan.visual_intent)
    rng = random.Random(zlib.crc32(f"{plan.template}|{plan.hero}".encode("utf-8")))
    packs = [k for k, p in STYLE_PACKS.items() if intent in p["intents"] and k != plan.style_pack]
    rng.shuffle(packs)
    lockups = _lockups_for(plan)
    current_lockup = plan.lockup if plan.lockup is not None else suggest_lockup(plan.hero_parts or [])
    alt_lockups = [x for x in lockups if x != current_lockup] or lockups
    out, seen = [plan], {(plan.style_pack, current_lockup)}
    i = 0
    while len(out) < n and i < 4 * n:
        pack = packs[i % len(packs)] if packs else plan.style_pack
        lockup = alt_lockups[i % len(alt_lockups)] if i % 2 else current_lockup
        i += 1
        if (pack, lockup) in seen:
            continue
        seen.add((pack, lockup))
        # Linh kiện do pack điền lại (bỏ lựa chọn cũ của bản gốc để pack mới thể hiện đúng tính cách).
        out.append(replace(plan, style_pack=pack, lockup=lockup, badge_style=None, stat_style=None, decor=None))
    return out


__all__ = ["generate_variants"]
