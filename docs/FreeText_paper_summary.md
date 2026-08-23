# FreeText: Training-Free Text Rendering in Diffusion Transformers via Attention Localization and Spectral Glyph Injection

**arXiv:** 2601.00535v1 [cs.CV], 2 Jan 2026
**Tác giả:** Ruiqiang Zhang, Hengyi Wang, Chang Liu, Guanjie Wang, Zehua Ma, Weiming Zhang — Anhui Province Key Laboratory of Digital Security, University of Science and Technology of China

---

## 1. Bài toán và động lực

Các mô hình T2I quy mô lớn (Stable Diffusion, FLUX, Qwen-Image) sinh ảnh open-domain tốt nhưng vẫn render text sai: lỗi chính tả, thiếu nét, méo hình, và **"semantic drift" / "semantic leakage"** — model vẽ *khái niệm* của từ thay vì *chính từ đó* (vd: render chữ "Car" thành hình một chiếc xe hơi thay vì chữ cái "C-a-r"). Vấn đề đặc biệt nặng với chữ tượng hình (Trung Quốc): phân bố ký tự long-tail, nhiều ký tự hiếm/ít gặp, và nhiều ký tự trông giống nhau về mặt cấu trúc nét (radical).

**Hai hướng tồn tại trước đây:**

| Hướng | Đại diện | Nhược điểm |
|---|---|---|
| Retraining-based | TextDiffuser, AnyText, GlyphControl, GlyphDraw, UniGlyph | Cần data/compute lớn, có thể dịch chuyển phân bố sinh ảnh và phong cách khỏi model gốc |
| Layout-conditioned | (thường đi kèm nhóm trên) | Ép cứng vị trí text bằng box/mask, giới hạn tự do prompt, giảm khả năng lập kế hoạch bố cục nội tại (scene planning) của model |

**Insight cốt lõi của paper:** model đã *nhận diện* text tốt hơn nhiều so với khả năng *render* nó ở mức pixel. Từ đó, thay vì dạy model "cách viết" từ đầu (retraining), tách bài toán thành hai bài toán con nhỏ hơn mà model gốc vốn đã có tiềm năng: **viết ở đâu** (WHERE) và **viết cái gì** (WHAT).

---

## 2. Phương pháp

Framework hoàn toàn **training-free, plug-and-play**, không sửa tham số/kiến trúc, chỉ can thiệp ở **inference time**. Gồm 2 giai đoạn:

### 2.1. Giai đoạn 1 — Định vị vùng viết bằng attention nội sinh (WHERE)

**Ý tưởng:** không dùng OCR hay VLM detector hậu kỳ (fragile), mà đọc trực tiếp attribution không gian từ attention image-to-text (I2T) sẵn có trong kiến trúc DiT/MMDiT khi model đang sinh ảnh.

**Bước 1 — Trích attention:**
Gọi `A^(t,l) ∈ R^(H×W×N_text)` là attention I2T trung bình theo head, tại timestep `t` và block DiT thứ `l`. Với span text mục tiêu `s`, xác định tập token con `T_s` của nó, rồi bổ sung thêm một số **sink-like token** (token phản hồi cao ổn định qua các layer/head) để tạo tập token neo `T̃_s`. Bản đồ định vị sơ bộ:

```
M^(t,l)(x,y) = (1/|T̃_s|) · Σ_{k∈T̃_s} A^(t,l)_{x,y,k}
```
sau đó chuẩn hoá tuyến tính về [0,1].

**Bước 2 — Chọn timestep–layer:** Gộp attention qua *mọi* timestep/block sẽ nhiễu (bước đầu thô/toàn cục, bước giữa tập trung đúng vùng viết, bước cuối lại loãng ra do refine toàn cục; block nông nhấn hình học cục bộ, block sâu tích hợp ngữ nghĩa toàn cục). Vì vậy chọn tập cặp `(t,l)` tốt nhất bằng **soft IoU** với một mask tham chiếu `Y`:

```
IoU(t,l) = ⟨M^(t,l), Y⟩ / (‖M^(t,l)‖₁ + ‖Y‖₁ − ⟨M^(t,l), Y⟩)
```

Chọn top-K cặp tạo tập `S`, rồi gộp:
```
M(x,y) = (1/|S|) · Σ_{(t,l)∈S} M^(t,l)(x,y)
```

**Bước 3 — Hậu xử lý theo topology:** Gộp lân cận cục bộ để triệt outlier nhỏ và tăng cường các cụm liên thông → nhị phân hoá `M` thành `B ∈ {0,1}^(H×W)` bằng ngưỡng thích ứng (tối đa hoá phương sai liên lớp, kiểu Otsu) → chạy **DBSCAN** trên pixel foreground để lấy các vùng liên thông ứng viên `{C_i}`, loại nhiễu thưa.

Mỗi vùng `C_i` được chấm điểm trên `M` gốc:
```
q_i = |{(x,y) ∈ C_i | M(x,y) > τ}| / |C_i|
```
(τ là một quantile cao của `M`). Chọn vùng tốt nhất, resize về độ phân giải latent → mask nhị phân cuối cùng `R ∈ {0,1}^(H_lat × W_lat)`.

### 2.2. Giai đoạn 2 — Spectral-Modulated Glyph Injection / SGMI (WHAT)

**Bước 1 — Chiếu latent khớp mức nhiễu (noise-aligned latent projection):**
Raster hoá chuỗi text mục tiêu `s` thành ảnh glyph tham chiếu `I_glyph` đặt tại vùng `R`, mã hoá bằng **đúng VAE của model gốc**:
```
z_ref = E_VAE(I_glyph) ∈ R^(C×H_lat×W_lat)
```
Tại timestep `t` với lịch trình nhiễu `(α_t, σ_t)`, khớp mức nhiễu bằng forward diffusion chuẩn:
```
z_ref^(t) = α_t · z_ref + σ_t · ε,   ε ~ N(0, I)
```

**Bước 2 — Điều biến phổ Log-Gabor:**
Trên `z_ref^(t)`, áp bộ lọc Log-Gabor `G(ρ,θ)` trong miền tần số 2D để **tăng cường dải tần trung-cao** (mang cấu trúc glyph/nét chữ) và **triệt tần số thấp** (nền/background) lẫn **tần số rất cao** (nhiễu):
```
ẑ_ref,c^(t)  = F(z_ref,c^(t))
ẑ_sgmi,c^(t)(ρ,θ) = G(ρ,θ) · ẑ_ref,c^(t)(ρ,θ)
z_sgmi,c^(t) = F⁻¹(ẑ_sgmi,c^(t))
```
(F, F⁻¹ là FFT 2D thuận/nghịch, áp riêng từng kênh `c`).

**Bước 3 — Tiêm không-thời gian có suy giảm dần (annealed spatiotemporal injection):**
Chỉ tiêm prior trong một **cửa sổ timestep sớm-giữa**, tránh phá vỡ lập kế hoạch toàn cục ở bước đầu hoặc refine chi tiết ở bước cuối:
```
t_start = 0.8T,   t_end = 0.6T
```
Với `t ∈ [t_start, t_end]`, trọng số cosine-anneal:
```
λ(t) = (1/2)·(1 + cos(π · (t − t_start)/(t_end − t_start)))
```
Cập nhật latent đang denoise bằng **thay thế có mask** (masked replacement, kiểu blended latent diffusion):
```
z̃^(t) = (I − λ(t)·R) ⊙ z^(t)  +  λ(t)·R ⊙ z_sgmi^(t)
```
Ngoài khoảng `[t_start, t_end]`, `z^(t)` giữ nguyên hoàn toàn.

### 2.3. CLT-Bench — benchmark chữ Hán long-tail (đóng góp phụ)

Benchmark riêng để đo mức độ suy giảm hiệu năng từ ký tự phổ biến/đơn giản → hiếm/phức tạp. Điểm độ khó ký tự `c` kết hợp số nét chuẩn hoá `κ(c)` và hạng tần suất `r(c)`:
```
K(c) = (κ(c) − κ_min)/(κ_max − κ_min)
R(c) = (r(c) − r_min)/(r_max − r_min)
D(c) = (w_s·K(c) + w_f·R(c)) / (w_s + w_f)  ∈ [0,1]
```
Với đoạn text `{txt_i}`, số ký tự `N_chars`, số vùng/đoạn `N_seg`:
```
C_char = (1/N_chars) · Σ_j D(c_j)
C_len  = min(N_chars/N_max, 1)
C_seg  = min((N_seg−1)/(M_max−1), 1)
Score  = (w_char·C_char + w_len·C_len + w_seg·C_seg) / (w_char+w_len+w_seg)  ∈ [0,1]
```
Prompt được phân tầng theo `Score` để tạo các tập con từ dễ/phổ biến → khó/hiếm.

---

## 3. Thực nghiệm — findings chính

**Setup:** so sánh *Base* vs *Base + FreeText* trên 4 model nền: Qwen-Image, FLUX.1-dev, SD3.5-L, SD3-M. FreeText chỉ hoạt động ở inference time, không sửa tham số/kiến trúc/thêm nhánh học được. Benchmark: longText-Benchmark (en/zh, đoạn văn dài), CVTG (2–5 vùng text, prompt ngắn), CLT-Bench (Hán tự hiếm, chỉ trên Qwen-Image). Metric: **NED** (Normalized Edit Distance, qua OCR engine), **CLIPScore**, **AestheticScore** (LAION), **VQA Score** (VLM-based).

### Bảng 1 — Qwen-Image & FLUX.1-dev (longText-Benchmark, CVTG)

| Model | Setting | Subset | NED↑ | CLIP↑ | Aes↑ | VQA↑ |
|---|---|---|---|---|---|---|
| Qwen-Image | Base | longText-en | 0.625 | 0.858 | 4.912 | 2.650 |
| Qwen-Image | +FreeText | longText-en | 0.713 | 0.864 | 5.013 | 4.177 |
| FLUX.1-dev | Base | longText-en | 0.598 | 0.863 | 5.365 | 2.563 |
| FLUX.1-dev | +FreeText | longText-en | 0.690 | 0.868 | 5.342 | 4.211 |
| Qwen-Image | Base | longText-zh | 0.639 | 0.474 | 4.607 | 3.657 |
| Qwen-Image | +FreeText | longText-zh | 0.694 | 0.537 | 4.749 | 4.211 |
| Qwen-Image | Base | CVTG | 0.574 | 0.781 | 4.386 | 2.756 |
| Qwen-Image | +FreeText | CVTG | 0.619 | 0.794 | 4.391 | 3.469 |
| FLUX.1-dev | Base | CVTG | 0.712 | 0.836 | 5.910 | 4.050 |
| FLUX.1-dev | +FreeText | CVTG | 0.722 | 0.839 | 5.936 | 4.952 |

→ NED và VQA Score tăng nhất quán ở mọi setting (đọc được rõ hơn); CLIPScore/AestheticScore gần như không đổi → **ít ảnh hưởng đến alignment ngữ nghĩa và thẩm mỹ**.

### Bảng 2 — SD3-M / SD3.5-L (CVTG only — model nhạy với prompt dài)

| Model | Setting | NED↑ | CLIP↑ | Aes↑ | VQA↑ |
|---|---|---|---|---|---|
| SD3.5-L | Base | 0.848 | 0.879 | 5.634 | 3.849 |
| SD3.5-L | +FreeText | 0.864 | 0.871 | 5.608 | 4.595 |
| SD3-M | Base | 0.616 | 0.851 | 5.906 | 2.903 |
| SD3-M | +FreeText | 0.669 | 0.852 | 5.917 | 3.674 |

### Bảng 3 — CLT-Bench (Qwen-Image, Hán tự hiếm)

| Setting | NED↑ |
|---|---|
| Base | 0.458 |
| +FreeText | 0.488 |

**Finding quan trọng nhất của bảng này (đọc kỹ ở mục Hạn chế bên dưới):** mức tăng trên CLT-Bench nhỏ hơn hẳn so với các benchmark khác.

### Lan truyền lợi ích chéo vùng (benefit propagation)

Sửa đúng một vùng text bằng FreeText có thể cải thiện luôn cả các vùng text khác chưa được xử lý trực tiếp, thể hiện qua VQA Score tăng ở mức toàn cục. Lý do: self-attention toàn cục trong DiT/MMDiT khiến patch token trộn thông tin toàn cục ở mỗi bước; lỗi nặng ở một vùng gây nhiễu cập nhật ở vùng khác, sửa lỗi trọng điểm giảm nhiễu lan.

### Chiến lược localization (ablation lựa chọn token)

| Setting | IoU↑ |
|---|---|
| Entity-only (chỉ token của span text) | 0.495 |
| Sink-only (chỉ sink token) | 0.479 |
| Entity + Sink | **0.561** |

→ Sink-only ổn định theo thời gian hơn nhưng trần thấp hơn; kết hợp Entity+Sink cho IoU tốt nhất.

### So sánh với VLM localization (closed-source)

| Method | IoU↑ |
|---|---|
| doubao-seed-1-6-251015 | 0.325 |
| gemini-2.5-flash-lite | 0.139 |
| gpt-5.1 | 0.159 |
| qwen3-vl-plus-2025-09-23 | 0.195 |
| **FreeText (ours)** | **0.561** |

→ Pipeline "nhận diện rồi định vị" (recognize-then-localize) của VLM dễ gãy khi gặp text nhiều dòng, nền lộn xộn, glyph méo — lỗi nhận diện lan sang lỗi định vị. Đọc trực tiếp attention I2T tránh được chuỗi lỗi này.

### Ablation SGMI

| Model | Settings | NED↑ | CLIP↑ | Aes↑ | VQA↑ |
|---|---|---|---|---|---|
| Qwen-Image | B (Base) | 0.625 | 0.858 | 4.912 | 2.650 |
| Qwen-Image | +F−SGMI (bỏ lọc phổ) | 0.686 | 0.860 | 5.027 | 3.724 |
| Qwen-Image | +F (đầy đủ) | 0.713 | 0.864 | 5.013 | 4.177 |
| FLUX.1-dev | B | 0.598 | 0.863 | 5.365 | 2.563 |
| FLUX.1-dev | +F−SGMI | 0.671 | 0.865 | 5.361 | 3.816 |
| FLUX.1-dev | +F | 0.690 | 0.868 | 5.342 | 4.211 |

→ Bỏ SGMI làm giảm NED và VQA Score rõ rệt, trong khi CLIP/Aes gần như không đổi → SGMI đóng góp chủ yếu vào **độ dễ đọc của text**, không phải thẩm mỹ chung. Quan sát định tính thêm: chỉ tiêm tần số thấp → mất cấu trúc nét (mất chi tiết stroke); chỉ tiêm tần số cao (nơi ngữ nghĩa chiếm ưu thế) → gây "concept-texture intrusion" (rò rỉ ngữ nghĩa vào texture). **Kết luận cốt lõi:** mấu chốt của điều biến miền tần số không phải là "tiêm nhiều thông tin hơn" mà là "tiêm đúng dải tần".

### Hiệu năng inference (đo trên NVIDIA A6000, bfloat16, 928×928, 50 bước)

| Model | Setting | Time (s)↓ | Mem (GB)↓ |
|---|---|---|---|
| Qwen-Image | Base | 37.64 | 53.76 |
| Qwen-Image | +FreeText | 42.33 | 54.35 |
| FLUX.1-dev | Base | 41.56 | 31.44 |
| FLUX.1-dev | +FreeText | 47.17 | 32.17 |
| SD3.5-L | Base | 35.03 | 26.11 |
| SD3.5-L | +FreeText | 41.17 | 26.91 |
| SD3-M | Base | 9.85 | 14.53 |
| SD3-M | +FreeText | 11.47 | 14.97 |

→ Overhead vừa phải: tăng latency end-to-end **~12–18%**, tăng bộ nhớ đỉnh **dưới 1GB** (chủ yếu từ Stage 1 — tích luỹ và chọn lọc attention I2T trước khi tiêm).

---

## 4. Hạn chế (đọc kỹ trước khi áp dụng)

1. **Không dạy được ký tự hoàn toàn mới, chỉ khuếch đại cái model đã "biết":** Tự paper thừa nhận trên CLT-Bench (Hán tự hiếm), mức tăng NED nhỏ hơn hẳn so với các benchmark khác — kết luận trực tiếp của nhóm tác giả: **SGMI hiệu quả nhất khi model gốc đã có sẵn representation khả dụng cho ký tự mục tiêu; nó tăng cường cấu trúc glyph có sẵn chứ không giúp model học ký tự hoàn toàn xa lạ từ đầu.** Đây là giới hạn về bản chất phương pháp (không phải giới hạn kỹ thuật có thể vá thêm), vì cơ chế cốt lõi (spectral blending) chỉ *khuếch đại* tín hiệu glyph đã yếu-nhưng-có-mặt trong latent của model, không *tạo mới* từ hư không.

2. **Chỉ test trên 4 model, không có FLUX.2/Qwen3-based model:** Toàn bộ thực nghiệm giới hạn ở Qwen-Image, FLUX.1-dev, SD3.5-L, SD3-M — dùng text encoder T5-XXL/CLIP hoặc tương đương, chưa có model nào dùng Qwen3 causal LM làm text encoder. Việc xác định token subsequence của span mục tiêu và chọn sink-like token ổn định cần làm lại/kiểm chứng riêng cho mỗi kiến trúc text-encoder mới.

3. **Cần biết trước chính xác chuỗi text cần render:** Phương pháp nhận `s` (target text span) như input tường minh — không tự sinh ra nội dung text cần vẽ, chỉ đảm bảo vẽ đúng nội dung đã cho trước.

4. **Cần quyền truy cập nội bộ mô hình (white-box):** Bắt buộc phải hook được attention map I2T ở nhiều layer/timestep trong quá trình sampling, và can thiệp trực tiếp vào latent giữa các bước denoise → không thể dùng qua API đóng hộp (closed-source, black-box), chỉ khả thi với model self-host có toàn quyền truy cập vòng lặp lấy mẫu.

5. **Overhead không phải zero:** dù "moderate", vẫn tăng 12–18% latency và cần thêm bước tính toán attention selection (Stage 1) — với ứng dụng cần latency thấp/thời gian thực, đây là chi phí cần cân nhắc.

6. **Localization dựa trên soft IoU cần một reference mask `Y`** để chọn timestep–layer tốt nhất (công thức IoU(t,l)) — chi tiết về nguồn gốc/cách xây `Y` trong pha vận hành thực tế (không phải lúc thiết kế/calibrate phương pháp) không được nêu đầy đủ trong phần method đã trích ở đây; đây là điểm cần làm rõ thêm khi triển khai.

7. **Kết luận (§5) của chính tác giả:** hướng nghiên cứu tiếp theo được đề xuất là **kiểm chứng tính tổng quát của phương pháp bằng cách thích nghi nó sang các model nền mới đang xuất hiện** — ngầm xác nhận đây vẫn là câu hỏi mở, chưa được chứng minh sẽ hoạt động tốt trên mọi kiến trúc DiT tương lai.

---

## 5. Related work được nhắc tới (để tra cứu thêm)

- **TextDiffuser** (Chen et al. 2023) — layout predictor học trên corpus có annotation OCR lớn.
- **AnyText** (2023), **GlyphControl**, **GlyphDraw**, **UniGlyph** — nhánh ControlNet-style/control branch riêng, cần data glyph/mask bổ sung.
- **Attend-and-Excite** (Chefer et al. 2023) — cơ sở lý thuyết cho phần chọn timestep-layer dựa attention.
- **Blended Latent Diffusion** (Avrahami et al.) — nền tảng kỹ thuật cho bước masked replacement ở Stage 2.
- **Vision Transformers Need Registers** (Darcet et al. 2023), **Attention sink trong LLM** — cơ sở lý thuyết cho khái niệm sink-like token dùng làm neo định vị.
