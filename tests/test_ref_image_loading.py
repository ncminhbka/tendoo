import sys
from pathlib import Path
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flux2.sampling import prc_img
from tendoo_v3.velocity_blending import (
    denoise_regional_velocity_blended,
    load_and_encode_ref_image,
)
from tendoo_v3.demo_server import GenerateRequest


class FakeAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def encode(self, x):
        b, c, h, w = x.shape
        return torch.randn(b, 128, h // 16, w // 16, dtype=torch.bfloat16, device=x.device)


class FakeDiT(torch.nn.Module):
    def __call__(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        # x is (B, L_total, C)
        # Returns predicted velocity with same shape as x
        return torch.zeros_like(x)


def test_load_and_encode_ref_image_shapes_and_rope(tmp_path):
    img_path = tmp_path / "test_product.png"
    img = Image.new("RGB", (512, 512), (200, 100, 50))
    img.save(img_path)

    fake_ae = FakeAE()
    
    ref_tokens, ref_ids = load_and_encode_ref_image(
        ref_image_path=img_path,
        ae=fake_ae,
        device="cpu",
        target_dim=512,
        time_offset=10.0,
    )

    assert ref_tokens.dim() == 3, f"Expected 3D tokens (1, L, 128), got {ref_tokens.shape}"
    assert ref_ids.dim() == 3, f"Expected 3D ids (1, L, 4), got {ref_ids.shape}"
    assert ref_tokens.shape[0] == 1
    assert ref_tokens.shape[2] == 128
    assert ref_ids.shape[0] == 1
    assert ref_ids.shape[2] == 4
    assert ref_tokens.shape[1] == ref_ids.shape[1]
    assert (ref_ids[:, :, 0] == 10.0).all(), "RoPE time offset not set to 10.0"


def test_velocity_blending_with_ref_tokens():
    """Verify that denoise_regional_velocity_blended properly handles canvas + ref tokens."""
    device = "cpu"
    L_canvas = 64
    L_ref = 32
    C = 128

    z_canvas = torch.randn(1, L_canvas, C, dtype=torch.bfloat16, device=device)
    ids_canvas = torch.zeros(1, L_canvas, 4, device=device)

    z_ref = torch.randn(1, L_ref, C, dtype=torch.bfloat16, device=device)
    ids_ref = torch.zeros(1, L_ref, 4, device=device)
    ids_ref[:, :, 0] = 10.0

    img_total = torch.cat([z_canvas, z_ref], dim=1)
    img_ids_total = torch.cat([ids_canvas, ids_ref], dim=1)

    txt_scene = torch.randn(1, 16, C, dtype=torch.bfloat16, device=device)
    txt_scene_ids = torch.zeros(1, 16, 4, device=device)
    txt_corr = torch.randn(1, 16, C, dtype=torch.bfloat16, device=device)
    txt_corr_ids = torch.zeros(1, 16, 4, device=device)

    mask = torch.ones(1, L_canvas, 1, dtype=torch.bfloat16, device=device) * 0.5
    timesteps = [1.0, 0.5, 0.0]

    model = FakeDiT()
    out = denoise_regional_velocity_blended(
        model=model,
        img=img_total,
        img_ids=img_ids_total,
        txt_scene=txt_scene,
        txt_scene_ids=txt_scene_ids,
        txt_corridor=txt_corr,
        txt_corridor_ids=txt_corr_ids,
        spatial_mask=mask,
        timesteps=timesteps,
        guidance=4.0,
        num_canvas_tokens=L_canvas,
    )

    # Must return strictly the canvas tokens
    assert out.shape == (1, L_canvas, C), f"Expected (1, {L_canvas}, {C}), got {out.shape}"


def test_form_dict_excludes_large_base64():
    """Verify that req.model_dump excludes image_base64 from LLM prompts."""
    req = GenerateRequest(
        category="promo",
        title="Trà Đào",
        image_base64="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
        ref_image_b64=None,
    )
    dumped = req.model_dump(
        exclude={"prompt", "template", "aspect_ratio", "ref_image_b64", "image_base64", "fast_preview"}
    )
    dumped["has_product_image"] = bool(req.ref_image_b64 or req.image_base64)

    assert "image_base64" not in dumped
    assert "ref_image_b64" not in dumped
    assert dumped["has_product_image"] is True


def test_rating_parsing_and_svg_render():
    """Verify that rating strings with stars/emojis/floats do not cause TypeError."""
    from tendoo_v3.icons import render_star_rating_svg
    from tendoo_v3.schema import TendooCreativePlan
    from tendoo_v3.renderer import build_template_html

    # Test direct SVG rendering with various types
    svg_emoji = render_star_rating_svg(count="⭐⭐⭐⭐⭐ 5.0 / 5.0")
    assert svg_emoji.count("<svg") == 5

    svg_str_num = render_star_rating_svg(count="4")
    assert svg_str_num.count("<svg") == 4

    svg_float_str = render_star_rating_svg(count="4.8")
    assert svg_float_str.count("<svg") == 5

    svg_int = render_star_rating_svg(count=3)
    assert svg_int.count("<svg") == 3

    # Test plan schema parsing
    plan_dict = {
        "template": "customer_feedback_card",
        "hero": "Hảo Hảo dai ngon đậm vị",
        "testimonial": "Chua cay ngon tuyệt",
        "rating": "⭐⭐⭐⭐⭐ 5.0 / 5.0",
    }
    plan = TendooCreativePlan.from_dict(plan_dict)
    assert plan.rating == 5

    # Test HTML template compilation with rating
    html = build_template_html(
        plan=plan,
        bg_data_uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
        width=1024,
        height=1024,
    )
    assert "stars-wrap" in html or "rating-stars" in html or "<svg" in html
