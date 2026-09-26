# Tendoo V3 — Hệ thống sinh poster thương mại tiếng Việt

Sinh poster quảng cáo tiếng Việt chất lượng thương mại: **ảnh nền do diffusion sinh ra**,
**chữ do HTML/CSS overlay render qua Chromium**. Hai tầng tách bạch hoàn toàn — mô hình
diffusion không bao giờ phải vẽ chữ, nên không bao giờ sai chính tả hay hỏng dấu tiếng Việt.

---

## Kiến trúc một trang

```
Brief người dùng
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ LLM (llm_planner)  — NGỮ NGHĨA                                  │
│  • chọn visual_intent + style (font, hiệu ứng, màu)             │
│  • CẮT VAI TRÒ nội dung: "GIẢM TỚI" / "25%" / "TOÀN BỘ MENU"    │
│  • viết prompt cảnh cho diffusion                               │
│  KHÔNG: đếm chữ · chọn cỡ · chỉ định toạ độ · sửa nội dung      │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Python — ĐO ĐẠC & QUYẾT ĐỊNH                                    │
│  • đếm ký tự, tra hợp đồng dung lượng, chốt template + cỡ       │
│  • geometry.py  → vùng chữ (zones)                              │
│  • mask_engine  → corridor mask cho diffusion                   │
│  • renderer.py  → ngân sách cỡ chữ theo thang poster            │
└─────────────────────────────────────────────────────────────────┘
      │
      ├──────────────────────────┬──────────────────────────────┐
      ▼                          ▼                              ▼
┌──────────────┐      ┌────────────────────┐      ┌──────────────────────┐
│ FLUX.2-klein │      │ Velocity Blending  │      │ Chromium (Playwright)│
│ 4B (DiT)     │ ───► │ scene ⊕ corridor   │ ───► │ overlay chữ + autofit│
│ + Qwen3-4B   │      │ chừa negative space│      │ → PNG thành phẩm     │
└──────────────┘      └────────────────────┘      └──────────────────────┘
```

---

## Cấu trúc thư mục

```
src/
├── flux2/          Mã gốc Black Forest Labs (FLUX.2) — ĐÓNG BĂNG, không sửa
└── tendoo_v3/      Toàn bộ sản phẩm: catalog, geometry, mask, renderer, 16 template + hạ tầng màu/font/Chromium
fonts/              19 font tiếng Việt có dấu đầy đủ
tests/              Bộ test + 36 suite JSON dữ liệu nghiệm thu template
scripts/            Script render & nghiệm thu (run_template_test, probe_type_hierarchy, run_real_llm, build_acceptance_report)
docs/               Nguyên lý thiết kế & kiến trúc
ROADMAP.md          ★ Định hướng kiến trúc và thi công — ĐỌC TRƯỚC KHI SỬA CODE
```

### Hạ tầng dùng chung (trong `tendoo_v3`)

Trước 27/09 nằm ở package riêng `tendoo_core`; nay gộp vào `tendoo_v3` (một package duy nhất):

| Module | Vai trò |
| :--- | :--- |
| `colors.py` | Phối màu tự động từ ảnh nền + cưỡng chế tương phản WCAG 2.1 (`ensure_contrast`) |
| `fonts.py` | Danh mục 19 font tiếng Việt, phân giải font, nhúng base64 |
| `poster_renderer.py` | Engine Playwright/Chromium headless: HTML → PNG, và đo DOM thật |
| `palette.py` | `ColorPalette` — bộ token màu |

---

## Cài đặt

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

Máy dev (không GPU) chạy được toàn bộ tầng render chữ. Tầng diffusion cần GPU — xem `AGENTS.md`.

## Chạy thử

```bash
# Render một poster (dùng nền giả lập, không cần GPU)
PYTHONPATH=src python scripts/run_template_test.py --template sandwich_top_heavy

# Toàn bộ test
PYTHONPATH=src python -m pytest tests/ -q

# Server demo (cần GPU cho diffusion thật; không có GPU sẽ chạy chế độ mock)
PYTHONPATH=src python src/tendoo_v3/demo_server.py --model distill
```

---

## Nguyên tắc bất biến

1. **Chữ không bao giờ do diffusion vẽ.** 100% overlay HTML/CSS.
2. **Không sửa `src/flux2/`.** Mọi mở rộng nằm ở tầng ngoài.
3. **Mọi phát biểu về cỡ chữ phải đo qua Chromium**, không suy từ hằng số — bài học đã
   trả giá: ước lượng cho 2.2x, đo thật chỉ 1.62x.
4. **Không thêm hằng số hiệu chỉnh tay nếu chưa có phép đo** chống lưng cho nó.
5. **Test phải kiểm thẩm mỹ, không chỉ kiểm không-crash.**

Chi tiết và căn cứ đo đạc của từng nguyên tắc: [ROADMAP.md](ROADMAP.md).

---

## Lịch sử

Repo này tách ra ngày **25/09/2026** từ repo `Tendoo AI` — nơi có 4 thế hệ code cùng tồn tại
(`tendoo_legacy`, `tendoo` v1, `tendoo_v2`, `tendoo_v3`). Chỉ v3 + phần hạ tầng v1 thực sự
được dùng được mang sang; ~22.000 dòng của các thế hệ cũ ở lại repo gốc (đã archive).

Xem [`docs/MIGRATION.md`](docs/MIGRATION.md) để biết chính xác cái gì được mang sang và vì sao.
