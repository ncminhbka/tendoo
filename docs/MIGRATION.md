# DI TRÚ SANG REPO `tendoo-v3` — 25/09/2026

Ghi lại chính xác cái gì được mang sang, cái gì ở lại, và vì sao — để sau này không ai phải
đoán, và để biết chỗ nào còn tìm lại được nếu cần.

---

## 1. Vì sao tách repo

Repo cũ `Tendoo AI` có **bốn thế hệ code cùng tồn tại**:

| Package | File | Dòng | Ai dùng |
| :--- | ---: | ---: | :--- |
| `src/tendoo_legacy` | 31 | 7.550 | Chỉ 7 script untracked; trong v1/v3 chỉ còn trong **comment** |
| `src/tendoo` (v1) | 24 | 7.905 | v3 dùng **4 module**; v2 dùng; 136 test |
| `src/tendoo_v2` | 7 | 1.042 | Nhánh `engine_version="v2"` **bên trong demo_server của v1**; 12 test |
| `src/tendoo_v3` | 15 | 7.721 | **Sản phẩm hiện tại** |
| `src/flux2` | 9 | 3.397 | Upstream BFL, đóng băng |

v3 chỉ chiếm **28%** code trong `src/`. Tệ hơn, v3 **không tự đứng được**: nó import từ v1 ở
4 module — gồm cả engine Playwright và hàm cưỡng chế tương phản WCAG. Nên không thể chỉ đơn
giản xoá v1.

Tách repo giải quyết cả hai: v3 trở nên tự chứa, và ~22.000 dòng của các thế hệ cũ không còn
nằm trong tầm mắt người đọc code mới.

---

## 2. Cái gì được mang sang

| Nguồn (repo cũ) | Đích (repo này) | Ghi chú |
| :--- | :--- | :--- |
| `src/flux2/` | `src/flux2/` | Nguyên vẹn, đóng băng |
| `src/tendoo_v3/` | `src/tendoo_v3/` | Nguyên vẹn, chỉ sửa import |
| `src/tendoo/core/colors.py` | `src/tendoo_core/colors.py` | |
| `src/tendoo/core/fonts.py` | `src/tendoo_core/fonts.py` | Sửa độ sâu `FONTS_DIR`: 3 → 2 cấp |
| `src/tendoo/poster_renderer.py` | `src/tendoo_core/poster_renderer.py` | |
| `src/tendoo/core/base.py` → class `ColorPalette` | `src/tendoo_core/palette.py` | Chỉ trích đúng 1 class (71 dòng / 558) |
| `fonts/` | `fonts/` | 19 font |
| 8 test .py + 36 suite JSON | `tests/` | Xem §4 |
| 20 script nghiệm thu template | `scripts/` | Xem §5 |
| `KE_HOACH_TONG_THE_TENDOO_V3.md` | `ROADMAP.md` | Sửa đường dẫn theo layout mới |
| `templates/DESIGN_PRINCIPLES.md`, `templates/TEMPLATE_SYSTEM_ARCHITECTURE.md` | `docs/` | Chuyển ra khỏi thư mục template |

### Vì sao `tendoo_core` gồm đúng 4 file đó

Đây là **toàn bộ** những gì `tendoo_v3` thực sự import từ v1, không hơn:

```
from tendoo.core.colors import ensure_contrast, get_contrasting_text_color, analyze_color_harmony
from tendoo.core.fonts  import resolve_font, FONT_CATALOG, list_font_options
from tendoo.poster_renderer import PosterRenderer
```

≈ **1.680 / 7.905 dòng** của v1. Phần còn lại ở lại repo cũ.

`base.py` không được mang nguyên: v3 chỉ cần `ColorPalette`, còn `PosterContent` và
`BaseLayout` là khái niệm kiến trúc v1 mà v3 chưa bao giờ đụng tới.

---

## 3. Cái gì ở lại (và tìm lại ở đâu)

| Ở lại repo cũ | Dòng | Lý do |
| :--- | ---: | :--- |
| `src/tendoo_legacy/` | 7.550 | Chỉ còn script untracked import; trong code sống chỉ còn ở comment |
| `src/tendoo/` phần legacy (demo_server, engine/, render_plan, llm_render_plan_server, typography, components, category_schema, style, layouts, qr, velocity_blending) | ~6.200 | Thuộc kiến trúc v1, v3 không dùng |
| `src/tendoo_v2/` | 1.042 | Nhánh engine thực nghiệm gắn chặt vào demo_server của v1 |
| `scripts/archive/` | 27.045 | **Chưa từng được add vào git** — xoá là mất vĩnh viễn, nên để nguyên tại chỗ |
| 148 test của v1/v2 | — | Phủ code không được mang sang |

Repo cũ **được archive, không xoá**. Mọi thứ trên vẫn tìm lại được ở đó.

---

## 4. Test: mang gì, bỏ gì

**Mang sang 8 file (99 test, tất cả đang xanh):**

| File | Phủ gì |
| :--- | :--- |
| `test_core_colors.py`, `test_core_fonts.py` | `tendoo_core` (hạ tầng) |
| `test_wcag_contrast_safeguard.py` | Tương phản WCAG, `tendoo_core` + v3 |
| `test_v3_upgrades.py` | Catalog, geometry, planner, renderer |
| `test_sandwich_top_heavy_matrix.py` | Mask + render template |
| `test_demo_server_endpoints.py`, `test_ref_image_loading.py` | API server, ref image |
| `conftest.py` | — |

**Đã cắt 2 test** khỏi `test_core_fonts.py`: `test_render_poster_with_explicit_font` và
`test_render_poster_with_auto_font_family` — chúng dựng poster qua `tendoo.engine.get_layout("omni")`,
tức layout engine của v1 không được mang sang. Phần chúng thực sự kiểm (nhúng font vào HTML)
vẫn còn được phủ bởi các test font khác.

> ⚠️ **Khoảng trống đã biết, ghi lại để không quên:** `pytest tests/` **KHÔNG render 14 template**.
> 36 file `tests/test_*_suite.json` chỉ được các script trong `scripts/` đọc. Đây chính là lý do
> 6 case tràn chữ thật vẫn lọt qua khi toàn bộ test xanh. Vá khoảng trống này là hạng mục **0C**
> trong ROADMAP.

---

## 5. Script: vì sao mang 20 file đó

Ở repo cũ, `.gitignore` ignore **toàn bộ `scripts/`** và chỉ allowlist 8 file — nên **124/132
script nằm ngoài quản lý phiên bản**, bao gồm toàn bộ tầng nghiệm thu 14 template. 20 script
mang sang đây là các script render/nghiệm thu template, nay **được git theo dõi đầy đủ**.

---

## 6. Kiểm chứng sau di trú

| Kiểm tra | Kết quả |
| :--- | :--- |
| Còn tham chiếu `tendoo.` / `tendoo_v2` / `tendoo_legacy` nào không | **Không còn dòng nào** |
| `import tendoo_core` + `import tendoo_v3` | OK — 19 font, `FONTS_DIR` tồn tại, 14 template |
| `pytest tests/` | **99/99 xanh** |
| Render thật đầu-cuối qua Playwright (có tầng markup `hero_parts`) | OK — PNG 457 KB, 3 đoạn vai trò |

**Hai lỗi đã mắc và đã sửa trong lúc di trú** (ghi lại để cảnh báo cho lần trích class tương tự):

1. Trích `ColorPalette` bắt đầu từ dòng `class ...` nên **mất decorator `@dataclass`** ở dòng
   ngay trên → `ColorPalette() takes no arguments`. Dùng `node.decorator_list` chứ đừng dùng
   `node.lineno`.
2. Cắt hàm bằng regex làm hỏng cú pháp file → phải làm lại bằng AST.
