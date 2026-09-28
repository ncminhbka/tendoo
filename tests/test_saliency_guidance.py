"""Kiểm TOÁN HỌC 2 cách tạo vùng yên mới (saliency_guidance.py) bằng mô hình giả trên CPU (không tải DiT).
Chất lượng ảnh thật chỉ đo được trên máy chủ: scripts/bench_saliency_modes.py."""

from __future__ import annotations

import re

import numpy as np
import torch

from tendoo_v3.saliency_guidance import (
    _gaussian_lowpass,
    composition_hint,
    denoise_blend_early,
    denoise_x0_lowpass,
)
from tendoo_v3.velocity_blending import denoise_regional_velocity_blended, denoise_scene_only
from test_maskless import FakeDiT


def _inputs(h=3, w=4, C=8, ref=3):
    torch.manual_seed(0)
    L = h * w
    img = torch.randn(1, L + ref, C)
    ids = torch.randn(1, L + ref, 4)
    scene, corridor = torch.randn(1, 5, C), torch.randn(1, 5, C)
    sids = torch.zeros(1, 5, 4)
    mask = torch.rand(1, L, 1)
    return img, ids, scene, corridor, sids, mask, L, h, w


TS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]


def test_blend_early_endpoints_match_existing_paths():
    img, ids, scene, corr, sids, mask, L, *_ = _inputs()
    full = denoise_regional_velocity_blended(FakeDiT(), img, ids, scene, sids, corr, sids, mask, TS, num_canvas_tokens=L)
    only = denoise_scene_only(FakeDiT(), img, ids, scene, sids, TS, num_canvas_tokens=L)
    assert torch.allclose(denoise_blend_early(FakeDiT(), img, ids, scene, sids, corr, sids, mask, TS, 99,
                                              num_canvas_tokens=L), full, atol=1e-6)
    assert torch.allclose(denoise_blend_early(FakeDiT(), img, ids, scene, sids, corr, sids, mask, TS, 0,
                                              num_canvas_tokens=L), only, atol=1e-6)


def test_blend_early_cost_and_frozen_reference():
    """k bước batch 2 rồi batch 1; token ảnh tham chiếu không đổi suốt quá trình."""
    img, ids, scene, corr, sids, mask, L, *_ = _inputs()
    seen = []

    class Spy(FakeDiT):
        def __call__(self, x, *a, **k):
            seen.append(x[:, L:, :].clone())
            return super().__call__(x, *a, **k)

    spy = Spy()
    out = denoise_blend_early(spy, img, ids, scene, sids, corr, sids, mask, TS, 2, num_canvas_tokens=L)
    assert spy.calls == [2, 2, 1, 1, 1]
    assert out.shape == (1, L, img.shape[2])
    assert all(torch.equal(s[:1], img[:, L:, :]) for s in seen)


def test_x0_lowpass_zero_strength_or_zero_mask_is_scene_only():
    img, ids, scene, _, sids, mask, L, h, w = _inputs()
    only = denoise_scene_only(FakeDiT(), img, ids, scene, sids, TS, num_canvas_tokens=L)
    fake = FakeDiT()
    a = denoise_x0_lowpass(fake, img, ids, scene, sids, mask, TS, h, w, strength=0.0, num_canvas_tokens=L)
    b = denoise_x0_lowpass(FakeDiT(), img, ids, scene, sids, torch.zeros_like(mask), TS, h, w, num_canvas_tokens=L)
    assert torch.allclose(a, only, atol=1e-5) and torch.allclose(b, only, atol=1e-5)
    assert fake.calls == [1] * (len(TS) - 1)  # một luồng, batch 1


def test_x0_lowpass_formula_one_step():
    """v' = v + s·M·(x̂0 - LP(x̂0)), x̂0 = x_t - t·v -- tính tay một bước."""
    img, ids, scene, _, sids, mask, L, h, w = _inputs(ref=0)
    ts = [0.7, 0.3]
    out = denoise_x0_lowpass(FakeDiT(), img, ids, scene, sids, mask, ts, h, w, strength=0.6, sigma=1.0, guidance=2.0)
    v = FakeDiT()(img, ids, torch.tensor([0.7]), scene, sids, torch.tensor([2.0]))
    x0 = img - 0.7 * v
    v2 = v + 0.6 * mask * (x0 - _gaussian_lowpass(x0, h, w, 1.0))
    assert torch.allclose(out, img + (0.3 - 0.7) * v2, atol=1e-5)


def test_lowpass_keeps_constant_field_and_token_order():
    c = torch.full((1, 12, 3), 2.5)
    assert torch.allclose(_gaussian_lowpass(c, 3, 4, 2.0), c, atol=1e-5)
    # Một điểm sáng ở (hàng 4, cột 6) của lưới 9×11 -> sau làm mờ, cực đại vẫn ở đúng ô đó (thứ tự (h w) như prc_img).
    x = torch.zeros(1, 99, 1)
    x[0, 4 * 11 + 6, 0] = 1.0
    y = _gaussian_lowpass(x, 9, 11, 1.0)
    assert int(y[0, :, 0].argmax()) == 4 * 11 + 6
    assert abs(float(y.sum()) - 1.0) < 1e-4  # xa mép: bảo toàn tổng


def test_composition_hint_follows_mask_side_and_has_no_numbers():
    h = w = 64
    left = np.zeros((h, w), np.float32)
    left[:, : w // 3] = 1
    hint_l = composition_hint(left)
    assert "right half" in hint_l and "left side" in hint_l
    assert "left half" in composition_hint(left[:, ::-1].copy())
    top = np.zeros((h, w), np.float32)
    top[: h // 3] = 1
    assert "bottom part" in composition_hint(top)
    sandwich = np.zeros((h, w), np.float32)
    sandwich[: h // 4] = 1
    sandwich[-h // 5:] = 1
    assert "middle band" in composition_hint(sandwich)
    corner = np.zeros((h, w), np.float32)
    corner[h // 2:, : w // 2] = 1
    assert "top right" in composition_hint(corner)
    assert composition_hint(np.zeros((h, w), np.float32)) == ""
    for m in (left, top, sandwich, corner):
        assert not re.search(r"\d", composition_hint(m))  # AGENTS §4.2: không số/tỉ lệ trong prompt
