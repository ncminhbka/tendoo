#!/usr/bin/env python3
"""
scripts/bench_saliency_modes.py -- ĐỐI CHỨNG 4 CÁCH TẠO VÙNG YÊN CHO CHỮ (chạy trên máy chủ 2× A30)
=====================================================================================================

Câu hỏi (28/09): blend 2 luồng ở mọi bước cắt cụt chủ thể ở mép mask (đồng hồ split_left) và tốn batch 2.
Có cách nào chừa chỗ cho chữ mà rẻ hơn và không cắt? Script này KHÔNG kết luận -- nó đo để người quyết.

Chế độ (cùng seed, cùng nhiễu ban đầu, cùng ca):
  blend_full      velocity blending hiện tại, mọi bước (mốc đối chứng)
  blend_early     trộn chỉ k bước đầu (--early-frac), sau đó luồng scene tự chạy
  maskless_hint   không mask, 1 luồng, scene_prompt + câu bố cục sinh từ mask (saliency_guidance.composition_hint)
  x0_lowpass      1 luồng + câu bố cục, bỏ tần số cao của ảnh sạch dự đoán trong vùng mask

Đo (mỗi ảnh):
  dit_s           thời gian vòng khử nhiễu (đã khởi động GPU trước, không tính mã hoá prompt / VAE)
  zone_grad       độ gắt cạnh trung bình trong vùng chữ (mask > 0.5), nền ảnh [0,1] -- thấp = yên
  zone_detail     tỉ lệ điểm ảnh vùng chữ có cạnh gắt (grad > 0.08) -- thấp = yên
  zone_lum_spread độ chênh sáng p95 - p5 trong vùng chữ -- cao = khó chọn một màu chữ
  seam_align      cạnh ảnh ở dải biên mask có VUÔNG GÓC với biên không (0.64 ≈ ngẫu nhiên, → 1 = có đường cắt)
  seam_energy     năng lượng cạnh ở dải biên / năng lượng cạnh phần ảnh ngoài mask (> 1 = biên gắt hơn phần còn lại)
  halo / scrim    số dòng chữ Bước 7 phải thêm quầng / lớp mờ khi dựng poster thật lên nền đó (ít = nền chừa chỗ tốt)
  text_lost       Cổng 4 báo mất chữ
  mirror_quieter  (Tầng 2) bố cục soi gương có vùng chữ yên hơn không -- nếu có, đổi bố cục là miễn phí

seam_* là THƯỚC ĐO GIÁN TIẾP: nó không biết "cánh tay" là gì. Kết luận về cắt cụt phải nhìn ảnh trên trang
index.html. Tương tự không chấm được "đẹp".

Chạy:
  PYTHONPATH=src python scripts/bench_saliency_modes.py                      # 12 ca × 3 seed × 4 chế độ
  PYTHONPATH=src python scripts/bench_saliency_modes.py --cases watch_pov_split_left --seeds 42
  PYTHONPATH=src python scripts/bench_saliency_modes.py --modes blend_full,x0_lowpass --x0-strength 0.7
  PYTHONPATH=src python scripts/bench_saliency_modes.py --mock               # chạy thử trên CPU, nền giả

Kết quả: output_probe/saliency_bench/<tag>/{index.html, results.json, summary.json, <ca>/<seed>/<mode>_*.png}
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BenchSaliency")

ALL_MODES = ["blend_full", "blend_early", "maskless_hint", "x0_lowpass"]

MODEL_DIRS = [
    "~/persistent-data/FLUX.2-klein-4B",
    "/home/jovyan/persistent-data/FLUX.2-klein-4B",
    "~/persistent-data/FLUX.2-klein-base-4B",
    "/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
]

# Bố cục soi gương: đổi sang cái này không cần sinh lại nền (chữ là HTML).
MIRRORS = {
    ("split_left", None): ("split_right", None),
    ("split_right", None): ("split_left", None),
    ("diagonal_slash", "left"): ("diagonal_slash", "right"),
    ("diagonal_slash", "right"): ("diagonal_slash", "left"),
    ("lifestyle_corner_pod", "bottom_left"): ("lifestyle_corner_pod", "bottom_right"),
    ("lifestyle_corner_pod", "bottom_right"): ("lifestyle_corner_pod", "bottom_left"),
    ("l_frame_showcase", "left"): ("l_frame_showcase", "right"),
    ("l_frame_showcase", "right"): ("l_frame_showcase", "left"),
}


@dataclass
class Case:
    id: str
    template: str
    scene_prompt: str
    corridor_prompt: str
    hero: str
    subhead: str = ""
    badge: str = ""
    extra_texts: List[str] = field(default_factory=list)
    cta: str = ""
    orientation: Optional[str] = None
    width: int = 1024
    height: int = 1024
    theme_color: str = "#38BDF8"
    font: str = "bevietnam"
    background_tone: str = "dark_luxury"


# 12 ca: đủ họ mask có chủ thể dễ bị cắt (cột bên, cắt chéo, góc, chữ L) + dải trên/dưới + thẻ tâm; 3 khung hình.
# Ca đầu là đúng poster lỗi 28/09. Prompt tiếng Anh, không chữ cần vẽ, không kích thước (AGENTS §4.2, §4.3).
CASES: List[Case] = [
    Case("watch_pov_split_left", "split_left",
         "Commercial outdoor product photography of a rugged smartwatch on a wrist, showing a detailed terrain map on "
         "the display, epic mountain ridge at golden sunrise, cinematic lighting, 85mm lens",
         "Distant misty mountain background in extreme soft focus, creamy bokeh, clean negative space without objects, "
         "matching ambient mountain lighting",
         "KHÁM PHÁ THẾ GIỚI CÙNG BẠN", "Người Bạn Đồng Hành Trên Mọi Nẻo Đường", "CHUYÊN DỤNG NGOÀI TRỜI",
         ["Bản đồ độ cao chi tiết", "Thiết kế bền bỉ"], "KHÁM PHÁ NGAY"),
    Case("barista_pour_split_right", "split_right",
         "Artisan cafe photography of a barista hand pouring hot water from a gooseneck kettle into a glass coffee "
         "dripper, warm steam rising, soft morning window light",
         "Warm dark wooden cafe interior in extreme bokeh, soft ambient light, completely empty negative space",
         "HƯƠNG VỊ CÀ PHÊ NGUYÊN BẢN", "Rang xay thủ công từ hạt Cầu Đất", "CÀ PHÊ NGUYÊN CHẤT",
         ["100% Robusta mộc"], "THƯỞNG THỨC NGAY", theme_color="#F59E0B", background_tone="warm_rustic"),
    Case("runner_split_left_916", "split_left",
         "Sports photography of a female runner mid-stride on a coastal road at dawn, dynamic motion, athletic wear, "
         "golden rim light, shallow depth of field",
         "Soft blurred ocean horizon and dawn sky in extreme bokeh, clean negative space without objects",
         "CHẠY VÌ CHÍNH BẠN", "Giày chạy bộ đệm êm cả ngày", "BỘ SƯU TẬP MỚI", [], "MUA NGAY",
         width=576, height=1024, theme_color="#F97316"),
    Case("sneaker_diagonal_left", "diagonal_slash",
         "Commercial photography of a vibrant neon running shoe hovering in mid air, dynamic motion trails, splash of "
         "water droplets, dramatic studio lighting",
         "Deep dark studio backdrop with soft motion blur, clean open negative space without objects",
         "BỨT PHÁ MỌI GIỚI HẠN", "Đệm khí siêu nhẹ cho mọi cự ly", "BẬT NẢY TỐI ĐA", ["Nhẹ chỉ 165g"], "MUA NGAY",
         orientation="left", theme_color="#10B981"),
    Case("guitar_diagonal_right_169", "diagonal_slash",
         "Concert photography of a guitarist playing an electric guitar on stage, colorful stage lights and haze, "
         "energetic atmosphere",
         "Dark stage haze with soft colorful light glow in extreme bokeh, clean negative space without objects",
         "ĐÊM NHẠC ROCK", "Sân khấu ngoài trời cuối tuần này", "VÉ CÓ HẠN", [], "ĐẶT VÉ",
         orientation="right", width=1024, height=576, theme_color="#E11D48"),
    Case("perfume_corner_pod", "lifestyle_corner_pod",
         "Luxury product photography of an elegant crystal perfume bottle on a polished marble pedestal, pink silk "
         "drapery, soft morning sunlight",
         "Soft pink silk drapery in creamy bokeh, gentle pastel glow, clean negative space without props",
         "HƯƠNG NƯỚC HOA PHÁP", "Tinh hoa quyến rũ", "GIẢM 25% HÔM NAY", [], "XEM CHI TIẾT",
         orientation="bottom_left", theme_color="#EC4899", font="playfair", background_tone="pastel"),
    Case("dog_owner_corner_pod", "lifestyle_corner_pod",
         "Lifestyle photography of a woman hugging a golden retriever in a sunny park, warm afternoon light, joyful "
         "candid moment",
         "Soft green park lawn and trees in extreme bokeh, warm light, clean negative space without objects",
         "YÊU THƯƠNG MỖI NGÀY", "Thức ăn hạt cho thú cưng", "MỚI", [], "MUA NGAY",
         orientation="bottom_right", theme_color="#22C55E", background_tone="light_clean"),
    Case("car_l_frame", "l_frame_showcase",
         "Automotive photography of a sleek silver electric SUV driving on a mountain road at sunset, motion blur on "
         "the wheels, cinematic lighting",
         "Soft blurred sunset sky and distant hills in extreme bokeh, clean negative space without objects",
         "LÁI XE XANH TƯƠNG LAI", "Pin 500 km một lần sạc", "RA MẮT", ["Sạc nhanh 30 phút"], "LÁI THỬ",
         orientation="left", theme_color="#0EA5E9"),
    Case("chef_sandwich_top", "sandwich_top_heavy",
         "Food photography of a chef hands plating a gourmet steak dish with sauce drizzle, restaurant kitchen, warm "
         "dramatic light",
         "Warm dark restaurant kitchen in extreme bokeh, soft amber glow, clean negative space without objects",
         "GIẢM 30% BÍT TẾT", "Chỉ trong tuần lễ khai trương", "ƯU ĐÃI", ["Tặng nước ép"], "ĐẶT BÀN",
         theme_color="#EF4444"),
    Case("family_sandwich_bottom_45", "sandwich_bottom_heavy",
         "Lifestyle photography of a happy family of four having a picnic on the grass, children laughing, sunny day, "
         "wide shot",
         "Soft green meadow in extreme bokeh, warm sunlight, clean negative space without objects",
         "NGÀY HỘI GIA ĐÌNH", "Vui chơi miễn phí cho bé", "CHỦ NHẬT NÀY", [], "ĐĂNG KÝ",
         width=820, height=1024, theme_color="#F59E0B", background_tone="light_clean"),
    Case("yoga_before_after", "before_after_split",
         "Wellness photography of a woman doing a yoga pose on a mat by a large window, calm morning light, "
         "minimalist studio",
         "Soft light minimalist studio floor in extreme bokeh, clean negative space without objects",
         "CƠ THỂ NHẸ NHÀNG", "Lớp yoga buổi sáng 8 tuần", "KHOÁ MỚI", [], "GIỮ CHỖ",
         theme_color="#8B5CF6", background_tone="light_clean"),
    Case("ring_luxury_card", "luxury_centered_card",
         "Luxury jewelry photography of a diamond ring and pearl necklace arranged on dark velvet, soft sparkle, "
         "elegant low key lighting",
         "Dark velvet surface in extreme soft focus, subtle warm glow, clean negative space without objects",
         "BỘ SƯU TẬP VĨNH CỬU", "Trang sức kim cương thủ công", "ĐỘC QUYỀN", [], "KHÁM PHÁ",
         theme_color="#D4AF37", font="playfair"),
]


# ------------------------------------------------------------------------------------------------------------------
# Đo trên ảnh (CPU, numpy)
# ------------------------------------------------------------------------------------------------------------------

def _gray(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def _sobel(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.pad(g, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:] - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]) / 8.0
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:] - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]) / 8.0
    return gx, gy


def zone_metrics(bg: Image.Image, mask: np.ndarray) -> Dict[str, Optional[float]]:
    g = _gray(bg)
    gx, gy = _sobel(g)
    mag = np.hypot(gx, gy)
    zone = mask > 0.5
    out: Dict[str, Optional[float]] = {"zone_grad": None, "zone_detail": None, "zone_lum_spread": None,
                                       "seam_align": None, "seam_energy": None}
    if zone.sum() < 50:
        return out
    out["zone_grad"] = round(float(mag[zone].mean()), 4)
    out["zone_detail"] = round(float((mag[zone] > 0.08).mean()), 4)
    lz = g[zone]
    out["zone_lum_spread"] = round(float(np.percentile(lz, 95) - np.percentile(lz, 5)), 4)

    # Dải biên mask: nơi mask chuyển tiếp. Pháp tuyến của biên = hướng gradient của mask.
    mx, my = _sobel(mask.astype(np.float32))
    mmag = np.hypot(mx, my)
    band = (mask > 0.2) & (mask < 0.8) & (mmag > 1e-4)
    outside = mask < 0.05
    if band.sum() >= 50 and outside.sum() >= 50:
        nx, ny = mx[band] / mmag[band], my[band] / mmag[band]
        along_normal = np.abs(gx[band] * nx + gy[band] * ny)
        total = mag[band] + 1e-6
        # Trung bình có trọng số theo độ gắt: chỉ cạnh mạnh mới "bỏ phiếu" về hướng.
        out["seam_align"] = round(float((along_normal).sum() / total.sum()), 4)
        out["seam_energy"] = round(float(mag[band].mean() / (mag[outside].mean() + 1e-6)), 4)
    return out


# ------------------------------------------------------------------------------------------------------------------
# Dựng poster thật lên nền (đọc báo cáo quầng/lớp mờ của Bước 7 + Cổng 4)
# ------------------------------------------------------------------------------------------------------------------

def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_plan(case: Case, template: Optional[str] = None, orientation: Optional[str] = None):
    from tendoo_v3.schema import StyleConfig, TendooCreativePlan
    return TendooCreativePlan(
        template=template or case.template,
        hero=case.hero,
        subhead=case.subhead or None,
        badge=case.badge or None,
        extra_texts=list(case.extra_texts),
        cta=case.cta or None,
        orientation=orientation if template else case.orientation,
        style=StyleConfig(font=case.font, theme_color=case.theme_color, text_effect="plain_elegant",
                          background_tone=case.background_tone),
        scene_prompt=case.scene_prompt,
        corridor_prompt=case.corridor_prompt,
    )


def plan_mask(plan, w: int, h: int) -> np.ndarray:
    """Đúng mask demo_server dùng (cùng density + cờ hình học như CSS)."""
    from tendoo_v3.mask_engine import generate_template_mask
    from tendoo_v3.renderer import compute_geometry_flags, compute_plan_content_density
    return generate_template_mask(template=plan.template, width=w, height=h, orientation=plan.orientation,
                                  density=compute_plan_content_density(plan), **compute_geometry_flags(plan))


async def _render_async(html: str, out_png: Path, w: int, h: int) -> Tuple[list, list]:
    from playwright.async_api import async_playwright
    from tendoo_v3.poster_renderer import _chromium_launch_kwargs, _set_content_and_wait_ready
    async with async_playwright() as p:
        browser = await p.chromium.launch(**_chromium_launch_kwargs())
        try:
            page = await (await browser.new_context(viewport={"width": w, "height": h})).new_page()
            await _set_content_and_wait_ready(page, html)
            await page.screenshot(path=str(out_png), full_page=True, type="png")
            halo = await page.evaluate("window.__tendooHalo || []")
            overflow = await page.evaluate("window.__tendooOverflow || []")
        finally:
            await browser.close()
    return halo, overflow


def render_poster(plan, bg: Image.Image, out_png: Path, w: int, h: int) -> Dict[str, Any]:
    from tendoo_v3.poster_renderer import _run_coroutine_sync
    from tendoo_v3.renderer import build_template_html
    html = build_template_html(plan=plan, bg_data_uri=_data_uri(bg), width=w, height=h)
    halo, overflow = _run_coroutine_sync(_render_async(html, out_png, w, h))
    return {
        "halo": sum(1 for r in halo if "bg" in r),
        "scrim": sum(1 for r in halo if "scrim" in r),
        "text_lost": any(o.get("verdict") in ("clipped", "overlap") for o in overflow),
    }


# ------------------------------------------------------------------------------------------------------------------
# Diffusion
# ------------------------------------------------------------------------------------------------------------------

class Engine:
    """Nạp DiT (cuda:0) + VAE/Qwen3 (cuda:1 nếu có). --mock: không nạp gì, nền giả tất định."""

    def __init__(self, model: str, mock: bool):
        self.mock = mock
        self.model_name = "mock"
        if mock:
            self.num_steps, self.guidance = 8, 1.0
            return
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("Không có CUDA. Chạy trên máy chủ, hoặc thêm --mock để thử luồng trên CPU.")
        self.torch = torch
        self.model_name = self._resolve(model)
        from flux2 import util
        self.util = util
        self.dev_dit = os.environ.get("TENDOO_V3_DEVICE_DIT", "cuda:0")
        self.dev_aux = os.environ.get("TENDOO_V3_DEVICE_AUX", "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        logger.info(f"Nạp {self.model_name}: DiT -> {self.dev_dit}, VAE + Qwen3 -> {self.dev_aux}")
        self.dit = util.load_flow_model(self.model_name, device=self.dev_dit).eval()
        self.ae = util.load_ae(self.model_name, device=self.dev_aux).eval()
        self.te = util.load_text_encoder(self.model_name, device=self.dev_aux)
        self.ae_dtype = next(self.ae.parameters()).dtype
        distilled = util.FLUX2_MODEL_INFO[self.model_name]["guidance_distilled"]
        if self.model_name == "flux.2-klein-4b":
            self.num_steps, self.guidance = 8, 1.5
        else:
            self.num_steps, self.guidance = 50, 4.0
        self.uncond = None
        self.cfg_scale = 1.0
        if not distilled:  # bản base: CFG với prompt rỗng (như demo_server)
            self.uncond = self.encode("")
            self.cfg_scale = self.guidance

    @staticmethod
    def _resolve(model: str) -> str:
        want_base = model == "base"
        for d in MODEL_DIRS:
            p = Path(os.path.expanduser(d))
            distill, base = p / "flux-2-klein-4b.safetensors", p / "flux-2-klein-base-4b.safetensors"
            if not want_base and distill.is_file():
                os.environ.setdefault("KLEIN_4B_MODEL_PATH", str(distill))
                return "flux.2-klein-4b"
            if model != "distill" and base.is_file():
                os.environ.setdefault("KLEIN_4B_BASE_MODEL_PATH", str(base))
                return "flux.2-klein-base-4b"
        if model == "distill" and Path(os.environ.get("KLEIN_4B_MODEL_PATH", "")).is_file():
            return "flux.2-klein-4b"
        if Path(os.environ.get("KLEIN_4B_BASE_MODEL_PATH", "")).is_file():
            return "flux.2-klein-base-4b"
        if Path(os.environ.get("KLEIN_4B_MODEL_PATH", "")).is_file():
            return "flux.2-klein-4b"
        raise SystemExit(f"Không tìm thấy checkpoint ({model}) trong {MODEL_DIRS}. Đặt KLEIN_4B_MODEL_PATH / "
                         "KLEIN_4B_BASE_MODEL_PATH hoặc FLUX_CHECKPOINT_DIR.")

    def encode(self, prompt: str):
        from flux2.sampling import prc_txt
        with self.torch.inference_mode():
            ctx = self.te([prompt]).to(self.torch.bfloat16)
        ctx, ids = prc_txt(ctx[0])
        return ctx.unsqueeze(0).to(self.dev_dit), ids.unsqueeze(0).to(self.dev_dit)

    def generate(self, mode: str, case: Case, mask: np.ndarray, seed: int, args) -> Tuple[Image.Image, float, str]:
        """Trả (ảnh nền, giây vòng khử nhiễu, scene_prompt thật đã dùng)."""
        from tendoo_v3.saliency_guidance import composition_hint
        w, h = case.width, case.height
        hinted = mode in ("maskless_hint", "x0_lowpass") or (mode.startswith("blend") and args.hint_blend)
        prompt = case.scene_prompt + (composition_hint(mask) if hinted else "")
        if self.mock:
            from tendoo_v3.velocity_blending import generate_mock_backdrop
            rng = np.random.default_rng(seed)
            m = None if mode in ("maskless_hint",) else mask
            bg = generate_mock_backdrop(width=w, height=h, theme_color="#%06x" % int(rng.integers(0, 0xFFFFFF)),
                                        background_tone=case.background_tone, spatial_mask=m)
            return bg, 0.0, prompt

        torch = self.torch
        from flux2.sampling import get_schedule, prc_img
        from tendoo_v3.saliency_guidance import denoise_blend_early, denoise_x0_lowpass
        from tendoo_v3.velocity_blending import denoise_regional_velocity_blended, denoise_scene_only

        w_lat, h_lat = w // 16, h // 16
        m_small = Image.fromarray((mask * 255).astype(np.uint8)).resize((w_lat, h_lat), Image.Resampling.BICUBIC)
        m_flat = torch.from_numpy(np.asarray(m_small, dtype=np.float32) / 255.0).reshape(1, -1, 1).to(self.dev_dit)

        cs, cs_ids = self.encode(prompt)
        cc = cc_ids = None
        if mode in ("blend_full", "blend_early"):
            cc, cc_ids = self.encode(case.corridor_prompt)

        torch.manual_seed(seed)
        z = torch.randn(1, 128, h_lat, w_lat, device=self.dev_dit, dtype=torch.bfloat16)
        tok, ids = prc_img(z[0])
        tok, ids = tok.unsqueeze(0).to(self.dev_dit), ids.unsqueeze(0).to(self.dev_dit)
        ts = get_schedule(num_steps=self.num_steps, image_seq_len=tok.shape[1])
        cfg = dict(txt_uncond=self.uncond[0] if self.uncond else None,
                   txt_uncond_ids=self.uncond[1] if self.uncond else None, cfg_scale=self.cfg_scale)
        common = dict(guidance=self.guidance, num_canvas_tokens=tok.shape[1], **cfg)
        k = max(1, round(args.early_frac * self.num_steps))

        for d in {self.dev_dit, self.dev_aux}:
            torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        with torch.inference_mode():
            if mode == "blend_full":
                out = denoise_regional_velocity_blended(self.dit, tok, ids, cs, cs_ids, cc, cc_ids, m_flat, ts, **common)
            elif mode == "blend_early":
                out = denoise_blend_early(self.dit, tok, ids, cs, cs_ids, cc, cc_ids, m_flat, ts, k, **common)
            elif mode == "maskless_hint":
                out = denoise_scene_only(self.dit, tok, ids, cs, cs_ids, ts, **common)
            elif mode == "x0_lowpass":
                active = None if args.x0_active_frac >= 1 else max(1, round(args.x0_active_frac * self.num_steps))
                out = denoise_x0_lowpass(self.dit, tok, ids, cs, cs_ids, m_flat, ts, h_lat, w_lat,
                                         strength=args.x0_strength, sigma=args.x0_sigma, active_steps=active, **common)
            else:
                raise ValueError(mode)
        torch.cuda.synchronize(self.dev_dit)
        dit_s = time.perf_counter() - t0

        zd = out[0].transpose(0, 1).reshape(1, 128, h_lat, w_lat).to(device=self.dev_aux, dtype=self.ae_dtype)
        with torch.inference_mode():
            x = self.ae.decode(zd).float()
        arr = ((x[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr), dit_s, prompt

    def warmup(self, args):
        """Một lượt nhỏ mỗi chế độ để CUDA/cuBLAS khởi động -- không tính vào số đo."""
        if self.mock:
            return
        logger.info("Khởi động GPU (không tính giờ)...")
        c = CASES[0]
        mask = plan_mask(make_plan(c), c.width, c.height)
        for mode in args.modes:
            self.generate(mode, c, mask, 0, args)


# ------------------------------------------------------------------------------------------------------------------
# Trang kết quả
# ------------------------------------------------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def summarize(rows: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Any]:
    out = {}
    base_t = _mean([r["dit_s"] for r in rows if r["mode"] == "blend_full"])
    for m in modes:
        rs = [r for r in rows if r["mode"] == m and not r.get("error")]
        if not rs:
            continue
        t = _mean([r["dit_s"] for r in rs])
        out[m] = {
            "n": len(rs),
            "dit_s": t,
            "dit_vs_blend_full": round(t / base_t, 3) if (t and base_t) else None,
            "zone_grad": _mean([r["zone_grad"] for r in rs]),
            "zone_detail": _mean([r["zone_detail"] for r in rs]),
            "zone_lum_spread": _mean([r["zone_lum_spread"] for r in rs]),
            "seam_align": _mean([r["seam_align"] for r in rs]),
            "seam_energy": _mean([r["seam_energy"] for r in rs]),
            "posters_with_scrim": sum(1 for r in rs if r["scrim"] > 0),
            "posters_with_halo": sum(1 for r in rs if r["halo"] > 0),
            "text_lost": sum(1 for r in rs if r["text_lost"]),
            "mirror_quieter": sum(1 for r in rs if r.get("mirror_quieter")),
        }
    return out


def build_html(rows: List[Dict[str, Any]], summary: Dict[str, Any], meta: Dict[str, Any], modes: List[str]) -> str:
    import html as H
    cols = ["n", "dit_s", "dit_vs_blend_full", "zone_grad", "zone_detail", "zone_lum_spread", "seam_align",
            "seam_energy", "posters_with_scrim", "posters_with_halo", "text_lost", "mirror_quieter"]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        f"<tr><td><b>{m}</b></td>" + "".join(f"<td>{summary[m].get(c)}</td>" for c in cols) + "</tr>"
        for m in modes if m in summary)
    groups: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["case"], r["seed"]), {})[r["mode"]] = r
    blocks = []
    for (cid, seed), by_mode in groups.items():
        cells = []
        for m in modes:
            r = by_mode.get(m)
            if not r:
                continue
            if r.get("error"):
                cells.append(f"<div class='cell'><h4>{m}</h4><pre>{H.escape(r['error'][:600])}</pre></div>")
                continue
            info = (f"DiT {r['dit_s']}s · vùng chữ grad {r['zone_grad']} · chi tiết {r['zone_detail']} · "
                    f"seam {r['seam_align']}/{r['seam_energy']} · quầng {r['halo']} · lớp mờ {r['scrim']}"
                    + (" · <b>MẤT CHỮ</b>" if r["text_lost"] else "")
                    + (" · gương yên hơn" if r.get("mirror_quieter") else ""))
            cells.append(
                f"<div class='cell'><h4>{m}</h4>"
                f"<img loading='lazy' src='{r['bg']}' title='nền'><img loading='lazy' src='{r['poster']}' title='poster'>"
                f"<p>{info}</p><details><summary>prompt</summary><code>{H.escape(r['prompt'])}</code></details></div>")
        blocks.append(f"<section><h3>{cid} · seed {seed}</h3><div class='row'>{''.join(cells)}</div></section>")
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Đối chứng vùng yên</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:14px system-ui,sans-serif;margin:16px;background:#111;color:#eee}}table{{border-collapse:collapse}}
td,th{{border:1px solid #444;padding:4px 8px;text-align:right}}th{{background:#222}}section{{margin:24px 0}}
.row{{display:flex;gap:12px;overflow-x:auto}}.cell{{min-width:300px;max-width:340px}}.cell img{{width:100%;display:block;
margin-bottom:4px}}code{{font-size:11px;white-space:pre-wrap}}h4{{margin:4px 0}}pre{{white-space:pre-wrap;color:#f88}}</style>
</head><body><h1>Đối chứng 4 cách tạo vùng yên cho chữ</h1>
<p>{H.escape(json.dumps(meta, ensure_ascii=False))}</p>
<p>Thấp hơn là tốt hơn: zone_grad, zone_detail, seam_align (0.64 ≈ ngẫu nhiên), seam_energy, số poster phải bật lớp mờ.
seam_* chỉ là thước đo gián tiếp; chuyện chủ thể có bị cắt cụt không thì phải nhìn ảnh bên dưới.</p>
<table><tr><th>mode</th>{head}</tr>{body}</table>{''.join(blocks)}</body></html>"""


# ------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["auto", "distill", "base"], default="auto")
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--cases", default="", help="danh sách id, phẩy ngăn cách; trống = cả 12")
    ap.add_argument("--seeds", default="42,7,2026")
    ap.add_argument("--early-frac", type=float, default=0.35, help="blend_early: tỉ lệ bước đầu được trộn (8 bước -> 3)")
    ap.add_argument("--x0-strength", type=float, default=0.85)
    ap.add_argument("--x0-sigma", type=float, default=2.5, help="độ mờ theo ô latent (1 ô = 16px)")
    ap.add_argument("--x0-active-frac", type=float, default=1.0, help="x0_lowpass: tỉ lệ bước đầu được áp (1 = mọi bước)")
    ap.add_argument("--hint-blend", action="store_true", help="nối câu bố cục cả vào 2 chế độ blend")
    ap.add_argument("--no-poster", action="store_true", help="chỉ sinh nền, bỏ dựng poster Chromium")
    ap.add_argument("--mock", action="store_true", help="không GPU: nền giả, chỉ để thử luồng script")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--out", default="output_probe/saliency_bench")
    args = ap.parse_args()
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in args.modes if m not in ALL_MODES]
    if bad:
        raise SystemExit(f"Chế độ lạ: {bad}. Có: {ALL_MODES}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    want = {c.strip() for c in args.cases.split(",") if c.strip()}
    cases = [c for c in CASES if not want or c.id in want]
    if want - {c.id for c in cases}:
        raise SystemExit(f"Ca lạ: {sorted(want - {c.id for c in cases})}. Có: {[c.id for c in CASES]}")

    out_dir = PROJECT_ROOT / args.out / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(args.model, args.mock)
    engine.warmup(args)
    meta = {"model": engine.model_name, "steps": engine.num_steps, "guidance": engine.guidance,
            "modes": args.modes, "seeds": seeds, "early_frac": args.early_frac, "x0_strength": args.x0_strength,
            "x0_sigma": args.x0_sigma, "x0_active_frac": args.x0_active_frac, "hint_blend": args.hint_blend,
            "mock": args.mock}
    logger.info(f"Cấu hình: {meta}")

    rows: List[Dict[str, Any]] = []
    results_path = out_dir / "results.json"
    for case in cases:
        plan = make_plan(case)
        mask = plan_mask(plan, case.width, case.height)
        mirror = MIRRORS.get((case.template, case.orientation))
        mirror_plan = make_plan(case, *mirror) if mirror else None
        mirror_mask = plan_mask(mirror_plan, case.width, case.height) if mirror_plan else None
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"{case.id}_mask.png")
        for seed in seeds:
            d = out_dir / case.id / str(seed)
            d.mkdir(parents=True, exist_ok=True)
            for mode in args.modes:
                row: Dict[str, Any] = {"case": case.id, "template": case.template, "size": f"{case.width}x{case.height}",
                                       "seed": seed, "mode": mode}
                try:
                    bg, dit_s, prompt = engine.generate(mode, case, mask, seed, args)
                    bg_path = d / f"{mode}_bg.png"
                    bg.save(bg_path)
                    row.update(dit_s=round(dit_s, 3), prompt=prompt, bg=str(bg_path.relative_to(out_dir)).replace("\\", "/"))
                    row.update(zone_metrics(bg, mask))
                    if mirror_mask is not None and row["zone_grad"] is not None:
                        alt = zone_metrics(bg, mirror_mask)["zone_grad"]
                        row["mirror_zone_grad"] = alt
                        row["mirror_quieter"] = bool(alt is not None and alt < 0.8 * row["zone_grad"])
                    row.update(halo=0, scrim=0, text_lost=False, poster=row["bg"])
                    if not args.no_poster:
                        p_path = d / f"{mode}_poster.png"
                        row.update(render_poster(plan, bg, p_path, case.width, case.height))
                        row["poster"] = str(p_path.relative_to(out_dir)).replace("\\", "/")
                    logger.info(f"[{case.id} s{seed} {mode}] DiT {row['dit_s']}s grad {row['zone_grad']} "
                                f"seam {row['seam_align']}/{row['seam_energy']} quầng {row['halo']} lớp mờ {row['scrim']}")
                except Exception as e:  # một ca hỏng không làm dừng cả lượt
                    logger.error(f"[{case.id} s{seed} {mode}] LỖI: {e}")
                    row["error"] = traceback.format_exc()
                    if not args.mock and "out of memory" in str(e).lower():
                        engine.torch.cuda.empty_cache()
                rows.append(row)
                results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = summarize(rows, args.modes)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (out_dir / "index.html").write_text(build_html(rows, summary, meta, args.modes), encoding="utf-8")
    logger.info("TÓM TẮT:\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info(f"Trang xem: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
