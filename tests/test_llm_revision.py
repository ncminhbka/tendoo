"""Vòng sửa độ dài của LLM (llm_planner._revise_lengths): chữ vượt giới hạn số từ -> hỏi lại LLM ĐÚNG trường vi phạm,
chỉ nhận bản sửa khi ít vi phạm hơn; Python không tự cắt chữ. Khung 16:9 dùng giới hạn chặt hơn (catalog.TEXT_WORD_LIMITS_WIDE)."""

from __future__ import annotations

import json

import tendoo_v3.llm_planner as lp
from tendoo_v3.schema import TendooCreativePlan
from tendoo_v3.validators import length_issues

BASE = {"template": "diagonal_slash", "visual_intent": "product_showcase", "hero": "NITRO RUN", "cta": "MUA NGAY",
        "scene_prompt": "Running shoe on dark podium, zero text", "corridor_prompt": "soft",
        "style": {"font": "bevietnam", "theme_color": "#2563EB", "text_effect": "shadow", "background_tone": "dark_luxury"}}
LONG = dict(BASE, subhead="Đế đệm khí siêu nhẹ cho những buổi chạy bộ đường dài mỗi ngày")  # 14 từ
SHORT = dict(BASE, subhead="Đế đệm khí, siêu nhẹ")


class _Resp:
    def __init__(self, obj):
        self.status_code, self._obj = 200, obj

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self._obj, ensure_ascii=False)}}]}


def _run(monkeypatch, replies, aspect="16:9"):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return _Resp(replies[len(calls) - 1])

    monkeypatch.setattr(lp.requests, "post", fake_post)
    monkeypatch.setenv("TENDOO_V3_LLM_BACKEND", "api")
    monkeypatch.setenv("TENDOO_V3_LLM_API_KEY", "test")
    monkeypatch.setattr(lp, "_save_debug_trace", lambda *a, **k: None)
    plan, trace = lp.generate_creative_plan({"title": "x"}, prompt="giày chạy", aspect_ratio=aspect, return_debug=True)
    return plan, trace, calls


def test_wide_frame_has_stricter_limits():
    p = TendooCreativePlan.from_dict(json.loads(json.dumps(dict(BASE, subhead="Đế đệm khí nhẹ cho chạy bộ"))))  # 7 từ
    assert length_issues(p, "1:1") == []
    assert length_issues(p, "16:9") == ["`subhead` đang 7 từ, tối đa 6 từ (khung ngang 16:9)"]


def test_revision_adopted_when_shorter(monkeypatch):
    plan, trace, calls = _run(monkeypatch, [LONG, SHORT])
    assert len(calls) == 2
    ask = calls[1]["messages"][-1]["content"]
    assert "`subhead` đang 14 từ, tối đa 6 từ" in ask and calls[1]["messages"][-2]["role"] == "assistant"
    assert plan.subhead == "Đế đệm khí, siêu nhẹ" and trace["revision"]["adopted"] is True


def test_revision_rejected_when_not_better_and_skipped_when_clean(monkeypatch):
    plan, trace, calls = _run(monkeypatch, [LONG, LONG])
    assert plan.subhead == LONG["subhead"] and trace["revision"]["adopted"] is False
    plan, trace, calls = _run(monkeypatch, [SHORT])
    assert len(calls) == 1 and "revision" not in trace


def test_revision_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TENDOO_V3_LLM_REVISE", "0")
    _, trace, calls = _run(monkeypatch, [LONG])
    assert len(calls) == 1 and "revision" not in trace


def test_revision_rejected_when_it_drops_a_field(monkeypatch):
    no_cta = {k: v for k, v in SHORT.items() if k != "cta"}
    plan, trace, _ = _run(monkeypatch, [LONG, no_cta])
    assert plan.cta == "MUA NGAY" and trace["revision"]["adopted"] is False and trace["revision"]["lost_fields"] == ["cta"]


def test_local_qwen_branch_returns_llm_plan(monkeypatch):
    """Nhánh Qwen CỤC BỘ (máy chủ GPU) phải trả plan của LLM, không âm thầm rơi về dự phòng (27/09: NameError)."""
    monkeypatch.setenv("TENDOO_V3_LLM_BACKEND", "local")
    monkeypatch.setattr(lp, "load_local_qwen3", lambda: (object(), object()))
    monkeypatch.setattr(lp, "generate_local_qwen_response", lambda m, t, msgs, **k: json.dumps(SHORT, ensure_ascii=False))
    monkeypatch.setattr(lp, "_save_debug_trace", lambda *a, **k: None)
    plan, trace = lp.generate_creative_plan({"title": "x"}, prompt="giày chạy", aspect_ratio="16:9", return_debug=True)
    assert trace["status"] == "success" and trace["mode"] == "local_qwen3_4b", trace.get("error")
    assert plan.hero == "NITRO RUN"
