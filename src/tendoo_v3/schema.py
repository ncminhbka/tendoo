"""
src/tendoo_v3/schema.py

Schema Output chuẩn hóa dành cho LLM (LLM Creative Plan Schema):
- Tối giản, trực quan, hướng ngữ nghĩa, KHÔNG làm rối LLM với các phép toán tọa độ.
- Chứa trường `extra_texts` (mảng chuỗi) để tự do nhồi thêm mọi thông tin phụ mà không làm vỡ khung.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StyleConfig:
    font: str = "bevietnam"
    theme_color: str = "#D4AF37"
    text_effect: str = "plain_elegant"  # 3d_gold, neon, chrome, plain_elegant, shadow, fire, embossed, chromatic, neon_bloom, hologram
    background_tone: str = "dark_luxury"  # dark_luxury, light_clean, warm_rustic, pastel, vibrant


def _normalize_for_verbatim_check(s: str) -> str:
    """Chuẩn hóa để đối chiếu nguyên văn: bỏ mọi khoảng trắng (kể cả \\n do xuống dòng
    thủ công) và không phân biệt hoa/thường. Dùng cho chốt chặn của `hero_parts`."""
    return "".join(str(s).split()).lower()


@dataclass
class TendooCreativePlan:
    template: str
    hero: str
    # MARKUP NGỮ NGHĨA CỦA HERO (Semantic Hero Markup) -- tầng mua được "tương phản
    # trong dòng" mà DESIGN_PRINCIPLES mục 1.3 mô tả: "CÓ CƠ HỘI NHẬN NGAY [2] TRIỆU
    # ĐỒNG", "GIẢM TỚI [25%] TOÀN BỘ MENU". Mỗi phần tử: {"t": <chuỗi>, "role":
    # "prefix"|"stat"|"suffix"|"body", "emphasis": "accent"|None}.
    #
    # KHÔNG VI PHẠM LUẬT "LLM KHÔNG CÓ QUYỀN BIÊN TẬP NỘI DUNG": markup chỉ CẮT ĐOẠN và
    # GẮN NHÃN, không được thêm/bớt/sửa một ký tự nào. `from_dict()` cưỡng chế điều đó
    # bằng cách nối các đoạn lại và đối chiếu nguyên văn với `hero` -- lệch một ký tự là
    # vứt toàn bộ markup, rơi về render phẳng như cũ (fail-safe, không bao giờ sai chữ).
    hero_parts: List[Dict[str, Any]] = field(default_factory=list)
    subhead: Optional[str] = None
    badge: Optional[str] = None
    tag_left: Optional[str] = None   # Dành cho before/after: vd "BEFORE"
    tag_right: Optional[str] = None  # Dành cho before/after: vd "AFTER"
    rating: Optional[int] = None     # Đánh giá sao: 1-5
    extra_texts: List[str] = field(default_factory=list)
    cta: Optional[str] = None
    store_info: Optional[str] = None
    qr_code: Optional[str] = None       # URL hoặc text mã QR (vd: "https://tendoo.ai/app" hoặc "QUET_MA")
    qr_label: Optional[str] = "QUÉT MÃ NGAY"
    testimonial: Optional[str] = None   # Lời nhận xét thực tế của khách hàng (vd: "97% khách hàng hài lòng...")
    reviewer_name: Optional[str] = None # Tên khách hàng / chức danh (vd: "Ngọc Lan - Hội viên VIP")
    steps: List[str] = field(default_factory=list) # Quy trình các bước (vd: ["Tư vấn 1:1", "Lên phác đồ", "Bảo hành 1 năm"])
    # CHỈ có ý nghĩa với template "lifestyle_corner_pod": "bottom_left" (mặc định) |
    # "bottom_right" | "top_left" | "top_right" -- vị trí đặt capsule góc;
    # Hoặc "diagonal_slash": "left" (mặc định) | "right" (bản gương).
    orientation: Optional[str] = None
    # Ý đồ thị giác (ROADMAP §3.3) -- quyết định ngưỡng tương phản squint test. None = intent mặc
    # định của template (catalog `visual_intents[0]`). LLM điền ở GĐ 3.
    visual_intent: Optional[str] = None

    style: StyleConfig = field(default_factory=StyleConfig)

    scene_prompt: str = ""
    corridor_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TendooCreativePlan:
        data = dict(data)
        style_raw = data.pop("style", {})
        if isinstance(style_raw, dict):
            style = StyleConfig(
                font=style_raw.get("font", "bevietnam"),
                theme_color=style_raw.get("theme_color", "#D4AF37"),
                text_effect=style_raw.get("text_effect", "plain_elegant"),
                background_tone=style_raw.get("background_tone", "dark_luxury"),
            )
        else:
            style = StyleConfig()

        # Đảm bảo extra_texts luôn là list string
        raw_extra = data.pop("extra_texts", [])
        if isinstance(raw_extra, list):
            extra_texts = [str(x).strip() for x in raw_extra if str(x).strip()]
        elif isinstance(raw_extra, str) and raw_extra.strip():
            extra_texts = [raw_extra.strip()]
        else:
            extra_texts = []

        # Đảm bảo store_info luôn là string phẳng (crash thật gặp trên server: LLM
        # nhỏ (gemma-4-E4B-it) đôi khi trả store_info thành dict {"address":...,
        # "hotline":...} thay vì string -- parse_store_info_items()/icons.py chỉ
        # nhận string, ném TypeError không bắt được, sập cả request 500).
        raw_store_info = data.pop("store_info", None)
        if isinstance(raw_store_info, dict):
            store_info = " • ".join(str(v).strip() for v in raw_store_info.values() if str(v).strip()) or None
        elif isinstance(raw_store_info, list):
            store_info = " • ".join(str(v).strip() for v in raw_store_info if str(v).strip()) or None
        elif raw_store_info is not None:
            store_info = str(raw_store_info).strip() or None
        else:
            store_info = None

        # Đảm bảo steps luôn là list string
        raw_steps = data.pop("steps", [])
        if isinstance(raw_steps, list):
            steps = [str(x).strip() for x in raw_steps if str(x).strip()]
        elif isinstance(raw_steps, str) and raw_steps.strip():
            steps = [raw_steps.strip()]
        else:
            steps = []

        # MARKUP HERO: chỉ giữ lại nếu nối các đoạn lại KHỚP NGUYÊN VĂN với `hero`.
        hero_str = data.get("hero", "")
        raw_parts = data.pop("hero_parts", [])
        hero_parts: List[Dict[str, Any]] = []
        if isinstance(raw_parts, list) and raw_parts:
            cleaned: List[Dict[str, Any]] = []
            for seg in raw_parts:
                if not isinstance(seg, dict):
                    cleaned = []
                    break
                text = str(seg.get("t", seg.get("text", ""))).strip()
                if not text:
                    continue
                role = str(seg.get("role", "body")).strip().lower()
                if role not in ("prefix", "stat", "suffix", "body"):
                    role = "body"
                emphasis = seg.get("emphasis")
                emphasis = str(emphasis).strip().lower() if emphasis else None
                if emphasis not in ("accent", None):
                    emphasis = None
                cleaned.append({"t": text, "role": role, "emphasis": emphasis})
            # Chốt chặn nguyên văn: tổng các đoạn phải tái tạo đúng `hero`.
            if cleaned and _normalize_for_verbatim_check(
                "".join(s["t"] for s in cleaned)
            ) == _normalize_for_verbatim_check(hero_str):
                hero_parts = cleaned

        # Chuẩn hóa rating thành int (1-5) an toàn
        raw_rating = data.get("rating")
        parsed_rating: Optional[int] = None
        if raw_rating is not None and str(raw_rating).strip():
            if isinstance(raw_rating, (int, float)):
                parsed_rating = max(1, min(5, int(round(raw_rating))))
            elif isinstance(raw_rating, str):
                star_count = raw_rating.count("⭐") + raw_rating.count("★")
                if star_count > 0:
                    parsed_rating = max(1, min(5, star_count))
                else:
                    import re
                    m = re.search(r"(\d+(?:\.\d+)?)", raw_rating)
                    if m:
                        try:
                            parsed_rating = max(1, min(5, int(round(float(m.group(1))))))
                        except Exception:
                            parsed_rating = 5
                    else:
                        parsed_rating = 5

        return cls(
            template=data.get("template", "sandwich_top_heavy"),
            hero=hero_str,
            hero_parts=hero_parts,
            subhead=data.get("subhead"),
            badge=data.get("badge"),
            tag_left=data.get("tag_left"),
            tag_right=data.get("tag_right"),
            rating=parsed_rating,
            extra_texts=extra_texts,
            cta=data.get("cta"),
            store_info=store_info,
            qr_code=data.get("qr_code"),
            qr_label=data.get("qr_label", "QUÉT MÃ NGAY"),
            testimonial=data.get("testimonial"),
            reviewer_name=data.get("reviewer_name"),
            steps=steps,
            orientation=data.get("orientation"),
            visual_intent=data.get("visual_intent"),
            style=style,
            # "background_prompt" chấp nhận thêm làm alias: hệ thống prompt LLM từng
            # dùng tên này trước khi thống nhất về "scene_prompt" (2026-09-15) -- giữ
            # lại để không âm thầm mất nội dung nếu 1 phiên bản prompt cũ còn tồn tại.
            scene_prompt=data.get("scene_prompt") or data.get("background_prompt") or "",
            corridor_prompt=data.get("corridor_prompt"),
        )

