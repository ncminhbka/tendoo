"""GĐ 5 -- Maskless Mode (ROADMAP §5.3). Kiểm TOÁN HỌC bằng mô hình giả trên CPU (không tải DiT):
bỏ corridor phải trùng khớp đường blend cũ với mask ≡ 0, và mỗi bước chỉ 1 forward batch 1."""

from __future__ import annotations

import torch

from tendoo_v3.velocity_blending import denoise_regional_velocity_blended, denoise_scene_only


class FakeDiT:
    """Vận tốc tất định phụ thuộc x, t, ngữ cảnh và guidance -- đủ để lộ mọi sai lệch trộn/batch."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        self.calls.append(x.shape[0])
        t = timesteps.view(-1, 1, 1)
        g = guidance.view(-1, 1, 1)
        return torch.tanh(x) * (1 + t) + ctx.mean(dim=1, keepdim=True) * 0.1 * g + x_ids[..., :1] * 0.01


def _inputs(L=12, C=8, ref=3):
    torch.manual_seed(0)
    img = torch.randn(1, L + ref, C)
    ids = torch.randn(1, L + ref, 4)
    scene, corridor = torch.randn(1, 5, C), torch.randn(1, 5, C)
    sids = torch.zeros(1, 5, 4)
    return img, ids, scene, corridor, sids, L


def test_scene_only_equals_blend_with_zero_mask():
    img, ids, scene, corridor, sids, L = _inputs()
    ts = [1.0, 0.75, 0.5, 0.25, 0.0]
    blended = denoise_regional_velocity_blended(FakeDiT(), img, ids, scene, sids, corridor, sids,
                                                torch.zeros(1, L, 1), ts, guidance=4.0, num_canvas_tokens=L)
    fake = FakeDiT()
    only = denoise_scene_only(fake, img, ids, scene, sids, ts, guidance=4.0, num_canvas_tokens=L)
    assert torch.allclose(blended, only, atol=1e-6)
    assert only.shape == (1, L, img.shape[2])
    assert fake.calls == [1] * (len(ts) - 1)  # batch 1, không phải batch 2


def test_scene_only_keeps_reference_tokens_frozen():
    """Token ảnh sản phẩm tham chiếu (RoPE t=10) không bị Euler cập nhật -- giống đường cũ."""
    img, ids, scene, _, sids, L = _inputs()
    seen = []

    class Spy(FakeDiT):
        def __call__(self, x, *a, **k):
            seen.append(x[:, L:, :].clone())
            return super().__call__(x, *a, **k)

    denoise_scene_only(Spy(), img, ids, scene, sids, [1.0, 0.5, 0.0], num_canvas_tokens=L)
    assert all(torch.equal(s, img[:, L:, :]) for s in seen)
