"""
src/flux2/masked_generation.py

==================================================================================================
SPIKE (2026-09-06): "reserve a region for text BEFORE generation" trên FLUX.2 klein 4B -- CHƯA
TEST được (môi trường viết code này không có GPU/không có weights FLUX.2 -- xem cảnh báo cuối
file). Viết THÊM (không sửa `model.py`/`sampling.py` gốc) để an toàn, dễ rollback, dựa đọc trực
tiếp code thật của 2 file đó (không suy đoán).

Bối cảnh & lựa chọn của user (không phải feature "học vẹt spell chữ" -- xem
docs/PHASE_3_LORA_TRAINING_ROADMAP.md dòng đầu: model 4B ĐÃ viết đúng tiếng Việt 100% khi cô lập,
vấn đề là tranh chấp attention khi nhiều slot cùng active). Ở đây mục tiêu KHÁC: không đặt chữ gì
vào latent cả -- chỉ ép 1 VÙNG PIXEL cụ thể (nơi CSS sẽ overlay chữ THẬT sau này, như toàn bộ
pipeline `typography_engine.py` vẫn đang làm) đừng bị model vẽ vật thể/sản phẩm đè lên, bằng 2 cơ
chế độc lập (chọn 1 hoặc kết hợp cả 2):

  A. `denoise_reserve()` -- Masked latent replacement kiểu RePaint (Lugmayr et al. 2022), thích
     nghi đúng công thức rectified-flow của FLUX.2 (xác nhận từ `sampling.py:305`:
     `img = img + (t_prev - t_curr) * pred`, tức quy ước `x_t = (1-t)*x_data + t*noise`).
     "MỀM" hơn hẳn 1 patch phẳng chết cứng -- chỉ khoá mạnh ở các bước ĐẦU (t gần 1, lúc model
     đang quyết định BỐ CỤC TOÀN CỤC: đặt sản phẩm ở đâu), rồi THẢ HẲN ra ở các bước cuối (t nhỏ,
     lúc model refine chi tiết/texture) -- để vùng đó vẫn có texture ảnh thật tự nhiên (không phải
     ô chết), đúng lo ngại "ô đen/giả tạo" user đã nêu.

  B. Attention bias (monkeypatch `causal_attn_fn`, không sửa file gốc) -- "động vào attention"
     mạnh hơn: chặn bớt trọng số attention từ CÁC TOKEN ẢNH TRONG VÙNG RESERVE (query) tới CÁC
     TOKEN CHỮ (txt, key) mô tả sản phẩm -- ví dụ từ "watch"/"smart watch" trong prompt tiếng Anh.
     Chỉ ức chế đúng khái niệm "sản phẩm" lan vào vùng đó, KHÔNG khoá cứng nội dung -- vùng đó vẫn
     tự do vẽ ánh sáng/bóng/bàn/vải tuỳ theo phần prompt còn lại. Rủi ro/effort cao hơn A -- cần
     TỰ ĐIỀN phần map "từ sản phẩm -> vị trí token" bằng tokenizer thật của `text_encoder.py` (chưa
     đọc file đó, không đoán mù).

CẢ HAI đều cần `region_mask`: 1 tensor boolean trên đúng chuỗi token ẢNH CHÍNH (không phải ref/txt)
-- suy ra từ `sampling.py:prc_img` (dòng 141-151): token thứ `i` (sau `rearrange(x, "c h w -> (h
w) c")`) ứng với toạ độ latent `h_ids[i] = i // w_lat`, `w_ids[i] = i % w_lat` (row-major, ĐÃ xác
nhận đọc trực tiếp code, không suy đoán). `h_lat`/`w_lat` là kích thước latent SAU VAE (không phải
pixel thật) -- KHÔNG tự đoán hệ số downsample ở đây: dự án đã có sẵn `compute_optimal_glyph_box`
(nhắc tới trong docs/PHASE_3_LORA_TRAINING_ROADMAP.md mục 2.1.2) giải đúng bài toán pixel->token
này cho glyph slot -- NÊN TÁI DÙNG hàm đó thay vì viết lại ở đây, tránh 2 nơi tính hệ số khác nhau.
==================================================================================================
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, Optional

import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.nn import functional as F

from . import model as _model_module  # monkeypatch target for mechanism B


# ==================================================================================================
# Vùng reserve, biểu diễn dưới dạng mask trên chuỗi token ảnh chính
# ==================================================================================================

def build_region_token_mask(h_lat: int, w_lat: int, row_frac: tuple[float, float], col_frac: tuple[float, float], device=None) -> Tensor:
    """
    Trả về bool tensor shape (h_lat*w_lat,), True tại các token thuộc vùng reserve.
    `row_frac`/`col_frac` là (start, end) theo TỈ LỆ 0..1 trên khung latent (không phải pixel) --
    nếu bạn có safe_rect dạng EmptyRect.as_css_percent() (đã dùng xuyên suốt pipeline CSS hiện tại,
    xem src/tendoo/layout_geometry.py), quy đổi: row_frac = (top_pct/100, 1-bottom_pct/100),
    col_frac = (left_pct/100, 1-right_pct/100) -- TÁI DÙNG đúng safe_rect đã có, không tính lại.

    Thứ tự token: row-major theo (h, w) -- khớp CHÍNH XÁC `rearrange(x, "c h w -> (h w) c")` ở
    `sampling.py:prc_img` (dòng 150), đã đọc trực tiếp, không suy đoán.
    """
    r0 = round(row_frac[0] * h_lat)
    r1 = round(row_frac[1] * h_lat)
    c0 = round(col_frac[0] * w_lat)
    c1 = round(col_frac[1] * w_lat)
    mask2d = torch.zeros((h_lat, w_lat), dtype=torch.bool, device=device)
    mask2d[r0:r1, c0:c1] = True
    return mask2d.flatten()


def build_region_token_mask_multi(
    h_lat: int, w_lat: int, rects: list[tuple[tuple[float, float], tuple[float, float]]], device=None,
) -> Tensor:
    """
    Ban HOP (union) cua `build_region_token_mask` -- moi phan tu trong `rects` la 1 cap
    (row_frac, col_frac) CUNG DINH DANG nhu ham tren. Dung khi vung reserve THUC TE la nhieu dong
    text co do rong khac nhau xep chong len nhau (hinh bac thang/chu T -- vd tieu de 2 dong, dong
    1 rong hon dong 2), KHONG phai 1 hinh chu nhat don gian -- dung y hinh dung nguoi dung ve ra
    (mask theo TUNG DONG text, khong phai 1 box lon bao tron ca khoi, cung khong phai ve vien tung
    con chu). Voi 1 phan tu duy nhat trong `rects`, ket qua giong het `build_region_token_mask`.
    """
    mask2d = torch.zeros((h_lat, w_lat), dtype=torch.bool, device=device)
    for row_frac, col_frac in rects:
        r0 = round(row_frac[0] * h_lat)
        r1 = round(row_frac[1] * h_lat)
        c0 = round(col_frac[0] * w_lat)
        c1 = round(col_frac[1] * w_lat)
        mask2d[r0:r1, c0:c1] = True
    return mask2d.flatten()


# ==================================================================================================
# CƠ CHẾ A -- masked latent replacement (RePaint-style), KHÔNG cần sửa model.py
# ==================================================================================================

def default_lock_schedule(t: float, t_full_lock_above: float = 0.85, t_release_below: float = 0.55) -> float:
    """
    λ(t) mặc định: khoá MẠNH (λ=1) khi t còn cao (bước đầu, model đang quyết định BỐ CỤC TOÀN CỤC
    -- đặt sản phẩm ở đâu), rồi anneal cosine xuống 0 và THẢ HẲN (λ=0) ở nửa sau quá trình khử
    nhiễu -- để chi tiết/texture ảnh thật tự do hình thành trong vùng đó thay vì bị khoá thành 1
    patch chết. Ngưỡng 0.85/0.55 là ĐIỂM KHỞI ĐẦU HỢP LÝ dựa trên cùng logic FreeText's
    early-mid injection window (docs/FreeText_paper_summary.md, t_start=0.8T/t_end=0.6T).

    CẢNH BÁO (đã kiểm chứng bằng số liệu thật từ test trên server, seq_len=6144, model distill 4
    bước): ngưỡng TUYỆT ĐỐI theo t này được hiệu chỉnh hình dung cho lịch trình SNR-shift ~50 bước
    của model base -- với model 4 bước, đường cong shift co cụm khác hẳn (mu khác), timesteps thực
    tế đo được là [1.0, 0.9306, 0.8171, 0.5982, 0.0] --> lam(t_prev) tương ứng ra
    [1.0, 0.9706, 0.0624, 0.0] -- tức KHOÁ CỨNG 2/4 bước (50% tổng bước), không phải ~15% như tính
    toán cho 50 bước. Kết quả thật: vùng reserve thành 1 mảng xám có vân, lạc hẳn tông màu so với
    phần còn lại -- đúng hiện tượng "ô giả tạo" đã lo từ đầu. Dùng
    `build_index_based_lock_schedule()` bên dưới thay cho hàm này khi số bước ít (<=10) -- nó tính
    theo VỊ TRÍ BƯỚC (index), không phụ thuộc hình dạng đường cong t.
    """
    if t >= t_full_lock_above:
        return 1.0
    if t <= t_release_below:
        return 0.0
    span = t_full_lock_above - t_release_below
    x = (t - t_release_below) / span  # 0..1
    return 0.5 * (1 - math.cos(math.pi * x))


def build_index_based_lock_schedule(
    timesteps: list[float], lock_frac: float = 0.25, anneal_frac: float = 0.25, lock_strength: float = 1.0,
) -> Callable[[float], float]:
    """
    Ban thay the cho `default_lock_schedule` -- tinh lam THEO VI TRI BUOC (index trong danh sach
    `timesteps` thuc te da qua SNR-shift), KHONG theo gia tri t tuyet doi. Sinh ra de sua dung
    "OOM lan 3" + "ket qua thuc te tren server" -- xem canh bao trong docstring
    `default_lock_schedule` o tren: cung 1 nguong t=0.85/0.55 khoa 2/4 buoc voi model 4-buoc
    (qua nang) nhung chi khoa ~15% buoc voi model 50-buoc (dung y).

    `timesteps`: list day du tu `sampling.get_schedule(num_steps, image_seq_len)` (do dai
    num_steps+1, tu 1.0 xuong 0.0) -- PHAI truyen dung list nay (khong phai list khac) vi ham tra
    ve 1 closure tra cuu theo GIA TRI t_prev THUC trong chinh list nay (khop chinh xac tung buoc
    cua vong lap `denoise_reserve`).

    `lock_frac`: ti le so buoc DAU TIEN bi khoa (lam dinh = lock_strength) -- it nhat 1 buoc.
    `anneal_frac`: ti le so buoc TIEP THEO dung de anneal cosine tu lock_strength xuong 0.0 -- sau
    do lam=0.0 han cho toi het. Vi du num_steps=4, lock_frac=0.25 -> khoa cung dung buoc 1;
    anneal_frac=0.25 -> round(0.25*4)=1 buoc anneal, nhung ANNEAL CAN TOI THIEU 2 BUOC de x chay
    tu >0 den 1 (khong bi ket cung o x=0) -- nen khi round ra <2, ham TU DONG BO QUA anneal, khoa
    dung 1 buoc dau roi THA HAN 3 buoc con lai.

    `lock_strength` (MOI, xem CANH BAO ket qua thuc te ben duoi): dinh lam toi da khi bi "khoa"
    -- MAC DINH 1.0 (khoa CUNG hoan toan, giu nguyen hanh vi cu). Voi model qua it buoc (vd 4 buoc
    distill), CHI 1 buoc bi khoa CUNG (lam=1.0) o t gan 1.0 (buoc dau, quyet dinh BO CUC/TEXTURE
    TOAN CUC) da du de "cam ket cung" vung do vao 1 mau/texture co dinh -- 3 buoc con lai (du da
    lam=0.0, khong con ep lai z_known nua) KHONG DU "ngan sach" Euler-step de model ve lai/hoa
    tron texture do vao phong cach anh con lai (moi buoc rectified-flow o day la 1 buoc NHAY LON,
    khong phai 1 buoc refine nho nhu diffusion nhieu buoc) -- ket qua THUC TE (server, 2026-09-07):
    van ra mang xam co van lac tong, GIONG HET truoc khi sua tu default_lock_schedule sang ham
    nay. Ha `lock_strength` xuong (vd 0.4-0.6) de buoc dau chi "goi y nhe" thay vi ep cung hoan
    toan -- CHUA verify that tren GPU, can user tu do lai.
    """
    n = len(timesteps) - 1  # so buoc thuc su (Euler step), timesteps co n+1 diem
    t_prev_list = timesteps[1:]  # t_prev cua tung buoc, dung thu tu duyet trong denoise_reserve
    lock_upto_idx = max(1, round(lock_frac * n))
    anneal_steps = max(0, round(anneal_frac * n))
    anneal_upto_idx = min(n, lock_upto_idx + anneal_steps)

    lam_by_tprev: dict[float, float] = {}
    for i, t_prev in enumerate(t_prev_list):
        if i < lock_upto_idx:
            lam = lock_strength
        elif anneal_steps >= 2 and i < anneal_upto_idx:
            span = anneal_upto_idx - lock_upto_idx  # >= 2 o day
            x = (i - lock_upto_idx + 1) / span  # (0, 1], khong bao gio dung yen o 0
            lam = lock_strength * 0.5 * (1 + math.cos(math.pi * x))  # ~lock_strength -> 0.0
        else:
            lam = 0.0
        lam_by_tprev[t_prev] = lam

    def _schedule(t: float) -> float:
        # Tra cuu gan dung (float roundtrip qua .tolist() co the lech epsilon nho) thay vi ep bang
        # tuyet doi -- van an toan vi cac t_prev cach nhau xa hon nhieu so epsilon float.
        closest = min(lam_by_tprev.keys(), key=lambda tp: abs(tp - t))
        return lam_by_tprev[closest]

    return _schedule


def denoise_reserve(
    model,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: list[float],
    guidance: float,
    region_mask: Tensor,          # bool, shape (L_img,) -- xem build_region_token_mask
    z_known: Tensor,              # clean latent noi dung "don gian" muon giu trong vung, shape (L_img_masked, C) hoac (1, C) de broadcast
    noise_fixed: Optional[Tensor] = None,   # co dinh xuyen suot cac buoc (RePaint) -- neu None, tu sample 1 lan
    lock_schedule: Callable[[float], float] = default_lock_schedule,
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
) -> Tensor:
    """
    Ban sao co chinh sua cua `sampling.denoise()` -- GIU NGUYEN toan bo logic goc (khong doi hanh
    vi khi region_mask rong/lock_schedule luon tra 0), CHI THEM buoc "ep lai" vung reserve sau moi
    Euler step. Dùng cho main-canvas img (khong dung cho anh tham chieu/ref).

    QUAN TRONG (khac voi 1 patch mau phang co dinh): `z_known` KHONG BAT BUOC la mau phang -- co
    the la anh mo/it chi tiet (vd nen ban/vai lua da lam mo manh) de vung nay, khi tha lock o cuoi,
    tiep tuc duoc model "ve tiep" chi tiet tu nhien tren nen mau/anh sang do, thay vi tu 1 gia tri
    hoan toan xa la voi phan con lai cua anh.
    """
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    device, dtype = img.device, img.dtype

    if noise_fixed is None:
        noise_fixed = torch.randn(z_known.shape[-1] if z_known.dim() == 1 else img.shape[-1], device=device, dtype=dtype)
        noise_fixed = noise_fixed.expand_as(img[0, region_mask]) if img.dim() == 3 else noise_fixed

    region_idx = region_mask.nonzero(as_tuple=True)[0]

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input, img_input_ids = img, img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(x=img_input, x_ids=img_input_ids, timesteps=t_vec, ctx=txt, ctx_ids=txt_ids, guidance=guidance_vec)
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

        lam = lock_schedule(t_prev)
        if lam > 0.0:
            # z_known_t theo DUNG quy uoc rectified-flow cua FLUX.2: x_t = (1-t)*x_data + t*noise
            # (xac nhan tu chinh cong thuc Euler step ben tren, khong suy doan).
            z_known_t = (1 - t_prev) * z_known + t_prev * noise_fixed
            img[:, region_idx, :] = lam * z_known_t + (1 - lam) * img[:, region_idx, :]

    return img


# ==================================================================================================
# CO CHE B -- attention bias qua monkeypatch causal_attn_fn (KHONG sua model.py)
# ==================================================================================================
# CANH BAO RO: phan nay MOI CHI LA KHUNG SUON (scaffold), CHUA HOAN CHINH -- thieu 1 manh bat buoc:
# map "tu san pham trong prompt (vd 'watch')" -> vi tri token trong `txt` sequence, thu duoc bang
# tokenizer THAT cua Qwen3 text encoder (`src/flux2/text_encoder.py` -- CHUA DOC file nay, khong
# doan mu cach goi tokenizer). Ban CAN tu dien ham `find_product_txt_token_span()` ben duoi bang
# tokenizer thuc te truoc khi dung duoc co che nay.

_current_attn_bias: Optional[Tensor] = None  # module-level "current bias" -- doc boi ban patched


def _causal_attn_fn_with_bias(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache=None):
    """Ban sao gan giong `model.causal_attn_fn` goc, CHI khac: cong them `_current_attn_bias` (neu
    co) vao nhanh "txt+img attend to all" truoc softmax, qua tham so `attn_mask` cua
    `scaled_dot_product_attention` (PyTorch ho tro attn_mask dang additive bias, khong chi boolean)."""
    from einops import rearrange

    bias = _current_attn_bias  # snapshot, tranh race neu doi giua chung

    if kv_cache is not None:
        k_ref, v_ref = kv_cache["k_ref"], kv_cache["v_ref"]
        q_txt, q_img = q[:, :, :num_txt_tokens, :], q[:, :, num_txt_tokens:, :]
        k_txt, v_txt = k[:, :, :num_txt_tokens, :], v[:, :, :num_txt_tokens, :]
        k_img, v_img = k[:, :, num_txt_tokens:, :], v[:, :, num_txt_tokens:, :]
        q_txt_img = torch.cat([q_txt, q_img], dim=2)
        k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
        v_all = torch.cat([v_txt, v_ref, v_img], dim=2)
        out = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, attn_mask=bias, is_causal=False)
    else:
        ref_start, ref_end = num_txt_tokens, num_txt_tokens + num_ref_tokens
        q_txt, q_ref, q_img = q[:, :, :ref_start, :], q[:, :, ref_start:ref_end, :], q[:, :, ref_end:, :]
        k_txt, v_txt = k[:, :, :ref_start, :], v[:, :, :ref_start, :]
        k_ref, v_ref = k[:, :, ref_start:ref_end, :], v[:, :, ref_start:ref_end, :]
        k_img, v_img = k[:, :, ref_end:, :], v[:, :, ref_end:, :]
        q_txt_img = torch.cat([q_txt, q_img], dim=2)
        k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
        v_all = torch.cat([v_txt, v_ref, v_img], dim=2)
        attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, attn_mask=bias, is_causal=False)
        attn_txt = attn_txt_img[:, :, :ref_start, :]
        attn_img = attn_txt_img[:, :, ref_start:, :]
        attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)
        out = torch.cat([attn_txt, attn_ref, attn_img], dim=2)

    return rearrange(out, "b h n d -> b n (h d)")


@contextmanager
def attention_bias_context(bias: Optional[Tensor]):
    """Vd: `with attention_bias_context(my_bias): denoise(...)` -- patch tam thoi
    `flux2.model.causal_attn_fn`, tu dong khoi phuc ham goc khi thoat scope (kho ke ca loi)."""
    global _current_attn_bias
    original_fn = _model_module.causal_attn_fn
    _current_attn_bias = bias
    _model_module.causal_attn_fn = _causal_attn_fn_with_bias
    try:
        yield
    finally:
        _model_module.causal_attn_fn = original_fn
        _current_attn_bias = None


def find_product_txt_token_span(
    tokenizer, prompt: str, product_phrase: str, max_length: int = 512,
) -> tuple[int, int]:
    """
    Map `product_phrase` (vd "smart watch", phai xuat hien NGUYEN VAN, khong bien the, trong
    `prompt`) -> (start, end) VI TRI TOKEN THAT trong chuoi `txt` ma FLUX.2 nhan (tuc dung vi tri
    trong `input_ids` cua Qwen3Embedder.forward(), src/flux2/text_encoder.py:389-407 -- da doc
    truc tiep code that, khong doan mu):
      1. Boc prompt bang DUNG chat template Qwen3Embedder dung (`apply_chat_template(...,
         tokenize=False, add_generation_prompt=True, enable_thinking=False)`) -- PHAI giong
         y het, vi vi tri token se lech neu boc khac di.
      2. Tokenize LAI voi `return_offsets_mapping=True` (Qwen3 dung fast tokenizer -- Rust backend
         -- ho tro offset mapping; neu AutoTokenizer.from_pretrained(...) tra ve slow tokenizer vi
         ly do nao do, ham nay se bao loi ro rang thay vi am tham sai).
      3. Tim token nao co character-offset giao voi vi tri `product_phrase` xuat hien trong text
         da boc template.

    CANH BAO: neu `product_phrase` xuat hien NHIEU LAN trong prompt, ham nay lay LAN XUAT HIEN
    DAU TIEN -- tu kiem tra lai neu prompt cua ban co the lap tu.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    char_start = text.find(product_phrase)
    if char_start == -1:
        raise ValueError(
            f"product_phrase {product_phrase!r} khong xuat hien nguyen van trong prompt sau khi "
            f"boc chat template -- kiem tra lai chinh ta/bien the."
        )
    char_end = char_start + len(product_phrase)

    enc = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True,
        max_length=max_length, return_offsets_mapping=True,
    )
    if "offset_mapping" not in enc:
        raise RuntimeError(
            "Tokenizer khong tra ve offset_mapping -- co the day la slow tokenizer (khong phai "
            "fast/Rust-backed). Kiem tra AutoTokenizer.from_pretrained(...) co dang tra ve "
            "PreTrainedTokenizerFast khong."
        )
    offsets = enc["offset_mapping"][0].tolist()

    token_start, token_end = None, None
    for i, (s, e) in enumerate(offsets):
        if s == e:  # special/padding token
            continue
        if s < char_end and e > char_start:
            if token_start is None:
                token_start = i
            token_end = i + 1
    if token_start is None:
        raise ValueError(
            "Khong map duoc product_phrase vao token nao -- co the bi truncate (prompt qua "
            f"{max_length} token) hoac loi offset."
        )
    return token_start, token_end


# ==================================================================================================
# CO CHE C -- reference-image conditioning THAT (denoise_cached + forward_kv_extract), KHONG phai
# ghi de latent tho bao nhu Co che A. Them sau khi Co che A cho ket qua mo/xam that tren server
# (2026-09-07) -- root-cause: Co che A ep ghi de latent GIUA CHUNG qua trinh denoise, model chua
# tung duoc huan luyen de xu ly kieu can thiep nay (dac biet voi model distill 4-buoc, xem canh bao
# trong build_index_based_lock_schedule/denoise_reserve o tren). FLUX.2 klein DUOC TAI LIEU HOA la
# co "multi-reference editing capabilities" -- co che THAT cho dieu nay la anh tham chieu duoc dua
# vao qua `img_cond_seq`/`img_cond_seq_ids` (xem `sampling.denoise_cached()`, da doc truc tiep code
# that): anh tham chieu duoc coi la "clean" (ref_fixed_timestep=0.0 trong `Flux2.forward_kv_extract`)
# va model ATTEND toi no qua KV-cache XUYEN SUOT qua trinh denoise anh chinh (KHONG ghi de truc
# tiep pixel/latent nao) -- day la co che DA DUOC HUAN LUYEN (documented), khong phai hack tu che
# nhu Co che A. Ham `encode_image_refs` trong `sampling.py` (dung that trong scripts/cli.py) da lam
# dung viec nay -- CAC HAM DUOI DAY MIRROR CHINH XAC quy uoc cua no (t_off = scale + scale*t,
# scale=10.0) nhung nhan device/dtype tuong minh (khong hardcode .cuda() nhu ban goc) de tuong
# thich voi kieu tach GPU DiT/AE cua `scripts/test_flux2_reserve_region.py`.
# ==================================================================================================

def build_reserve_guide_image(
    width: int, height: int, rects: list[tuple[tuple[float, float], tuple[float, float]]],
    bg_top_rgb: tuple[int, int, int] = (250, 238, 222),
    bg_bottom_rgb: tuple[int, int, int] = (238, 220, 196),
    scrim_rgb: tuple[int, int, int] = (205, 190, 168),
) -> Image.Image:
    """
    Dung 1 anh PIL DON GIAN lam 'anh tham chieu' cho Co che C -- KHONG phai anh that/khong can
    dep, chi la 1 goi y bo cuc THO: nen la 1 gradient mem doc (giu tong mau chung, it thong tin
    cu the ve noi dung de KHONG ap dat bo cuc phan con lai cua canvas), vung reserve (`rects`,
    CUNG DINH DANG (row_frac,col_frac) nhu build_region_token_mask/_multi) duoc to PHANG 1 mau
    tuong phan ro rang -- tin hieu "vung nay giu don gian, dung ve vat the phuc tap" cho model
    tham chieu qua attention (KHONG phai ep giu nguyen pixel -- day khong phai inpainting cung).

    `rects` PHAI cung 1 danh sach da dung cho `build_region_token_mask_multi` (kich thuoc token) --
    dam bao vung to scrim tren anh pixel THAT KHOP vi tri voi vung mask token, vi day chinh la diem
    manh cua co che nay: anh tham chieu duoc encode O DUNG DO PHAN GIAI CANVAS CHINH (width/height
    truyen vao day PHAI bang canvas that) nen RoPE h/w id cua no khop 1-1 voi anh chinh (chi khac
    o "t" id qua `encode_reserve_guide_ref`) -- tao tuong ung khong gian THAT giua tham chieu va
    anh dang denoise, khong chi la "1 anh vi du chung chung".
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        f = y / max(1, height - 1)
        r = round(bg_top_rgb[0] + (bg_bottom_rgb[0] - bg_top_rgb[0]) * f)
        g = round(bg_top_rgb[1] + (bg_bottom_rgb[1] - bg_top_rgb[1]) * f)
        b = round(bg_top_rgb[2] + (bg_bottom_rgb[2] - bg_top_rgb[2]) * f)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    for row_frac, col_frac in rects:
        x0, x1 = col_frac[0] * width, col_frac[1] * width
        y0, y1 = row_frac[0] * height, row_frac[1] * height
        draw.rectangle([x0, y0, x1, y1], fill=scrim_rgb)
    return img


def encode_reserve_guide_ref(
    ae, ae_device, ae_dtype, guide_image: Image.Image, t_offset: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """
    Ma hoa `guide_image` (PIL, CUNG kich thuoc canvas chinh) thanh (img_cond_seq, img_cond_seq_ids)
    de dung truc tiep voi `sampling.denoise_cached()`. Mirror CHINH XAC quy uoc cua
    `encode_image_refs()` that trong `sampling.py` (dung boi `scripts/cli.py` production):
    `t_off = scale + scale*t` voi `scale=10.0` -- GIU NGUYEN gia tri 10.0 (khong bay dat) de dung
    dung quy uoc production, cho model phan biet day la "anh tham chieu" (t=10) khac voi anh chinh
    dang denoise (t=0 mac dinh trong `prc_img`).

    Tra ve tensor CON O `ae_device` -- goi ham nay xong PHAI TU `.to(device=dit_device)` truoc khi
    ghep vao `denoise_cached()` (giong het cach `z_known`/`ctx` da chuyen device trong script), vi
    ham nay khong biet DiT dang chay o GPU nao (co the khac AE, xem --aux-device).
    """
    from .sampling import default_prep, prc_img  # tranh circular import o module-level

    x = default_prep(guide_image, limit_pixels=None)  # (C,H,W), da chuan hoa ve [-1,1]
    x = x.unsqueeze(0).to(device=ae_device, dtype=ae_dtype)
    with torch.no_grad():
        z = ae.encode(x)[0]  # (C, h_lat, w_lat) -- bo batch dim
    # QUAN TRONG: encode_image_refs() that (sampling.py) dung t_off KIEU int64 (tu torch.arange(...)
    # + scale la python int) -- torch.arange(h)/torch.arange(w) trong prc_img cung mac dinh int64.
    # Neu truyen t_coord dang float se lech dtype voi 2 truc con lai trong torch.cartesian_prod --
    # ep long() de dung y het quy uoc production, tranh loi/upcast am tham.
    t_coord = torch.tensor([int(round(t_offset))], dtype=torch.long)
    ref_tokens, ref_ids = prc_img(z, t_coord=t_coord)  # (L_ref, C), (L_ref, 4)
    ref_tokens = ref_tokens.unsqueeze(0).to(torch.bfloat16)  # (1, L_ref, C)
    ref_ids = ref_ids.unsqueeze(0)  # (1, L_ref, 4)
    return ref_tokens, ref_ids


def build_product_suppression_bias(
    num_txt_tokens: int, num_ref_tokens: int, num_img_tokens: int,
    region_mask: Tensor, product_txt_token_span: tuple[int, int],
    suppress_value: float = -30.0, device=None, dtype=None,
) -> Tensor:
    """
    Bias additive shape (num_txt_tokens+num_img_tokens, num_txt_tokens+num_ref_tokens+num_img_tokens)
    -- khop dung layout [txt, ref, img] (khong-cache nhanh) cua `causal_attn_fn`. Dat gia tri rat am
    (~softmax(-30) ~ 0) tai (query=img token trong region_mask, key=txt token thuoc
    product_txt_token_span) -- moi cho khac = 0 (khong doi hanh vi).

    `product_txt_token_span`: (start, end) VI TRI TOKEN THAT trong `txt`, PHAI tu tinh bang
    tokenizer that cua Qwen3 (src/flux2/text_encoder.py) -- KHONG doan mu o day.
    """
    L_q = num_txt_tokens + num_img_tokens
    L_k = num_txt_tokens + num_ref_tokens + num_img_tokens
    bias = torch.zeros((L_q, L_k), device=device, dtype=dtype)

    img_query_offset = num_txt_tokens  # trong q_txt_img, img bat dau ngay sau txt
    region_query_idx = region_query_idx = img_query_offset + region_mask.nonzero(as_tuple=True)[0]

    p0, p1 = product_txt_token_span
    bias[region_query_idx.unsqueeze(1), p0:p1] = suppress_value
    return bias


# ==================================================================================================
# CO CHE D (2026-09-07) -- Sequential exact-trajectory latent injection + True CFG, tren
# flux.2-klein-base-4b (50 buoc). Them sau khi Co che A (ghi de latent tu 1 patch mau gia) va
# Co che C (reference-conditioning mem qua denoise_cached) DEU khong cho hieu qua ro tren server
# that (3 lan chay, 2 canh khac nhau): C bi model "lo" di vi khong co chi dan text lien he ro rang
# den anh tham chieu (dung y user chi ra); A bi han che boi model 4-buoc khong du "duong bang" de
# hoa tron 1 gia tri gia (khong phai noi dung THAT).
#
# Co che nay ke thua TRUC TIEP tu 1 file DA CO SAN, DA CHAY DUOC trong repo --
# `scripts/probe_sequential_inpainting.py` (doc truc tiep, khong doan mu) -- von dung cho viec
# GHEP TIEU DE+PHU DE (2 lan sinh anh rieng, moi lan giu nguyen vung da sinh truoc do), khong phai
# cho viec "tranh vat the lon xon". Diem manh cua co che nay so voi Co che A: thay vi ep vung
# reserve ve 1 GIA TRI GIA (mau phang), no ghi lai TOAN BO trajectory THAT tu 1 lan sinh anh
# THAT KHAC (Pass 1 -- vd chi ve "bau troi/trang don gian"), roi o Pass 2 (sinh canh day du), buoc
# nao cung EP vung reserve ve DUNG latent THAT cua Pass 1 tai buoc tuong ung -- vi day la noi dung
# ĐÃ ĐUỢC MODEL TU SINH RA (khong phai gia tri ngoai lai), model o Pass 2 "thay" no nhu 1 phan tu
# nhien cua chinh canvas dang ve, khong bi coi la "vat la".
#
# Model BAT BUOC la flux.2-klein-base-4b (hoac klein-9b-base) -- KHONG phai ban distill: base co
# 50 buoc (du "duong bang" de hoa tron bien), va params.use_guidance_embed=False cho CA 2 bien the
# 4B (xac nhan tu util.py: ca "flux.2-klein-4b" lan "flux.2-klein-base-4b" dung chung Klein4BParams)
# nen guidance PHAI lam qua True CFG thu cong (batch doi ["", prompt], KHONG qua guidance_embed) --
# dung Y HET pattern DA CO SAN va DA CHAY DUOC trong probe_sequential_inpainting.py (dong 173-197),
# khong phai tu nghi ra.
# ==================================================================================================

def soften_region_mask(region_mask: Tensor, h_lat: int, w_lat: int, feather_iters: int = 3) -> Tensor:
    """
    Bien 1 bool mask PHANG (tu build_region_token_mask/_multi, shape (h_lat*w_lat,)) thanh 1 mask
    MEM [0,1] CUNG SHAPE -- lam mem bien qua vai lan average-pool 3x3 lien tiep (xap xi Gaussian
    blur, khong can dependency ngoai torch). Thay cho `compute_inpainting_mask` trong
    `probe_sequential_inpainting.py` (o do chi la 1 dai ngang don gian, cosine ramp theo 1 truc y)
    -- o day TONG QUAT cho BAT KY hinh dang mask nao (ke ca hop nhieu hinh chu nhat khong loi nhu
    `title_two_line`), vi lam mem hau-ky tren chinh mask 2D thay vi tinh cong thuc rieng cho tung
    hinh dang.

    `feather_iters` cang lon, bien cang rong/mem hon -- 3 la diem khoi dau hop ly (moi lan avg_pool
    3x3 mo rong vung anh huong 1 pixel latent moi phia, 3 lan ~ 3 pixel latent ~ 48px anh that voi
    downsample factor 16).
    """
    m = region_mask.float().reshape(1, 1, h_lat, w_lat)
    for _ in range(feather_iters):
        m = F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    return m.reshape(-1).clamp(0.0, 1.0)


def denoise_trajectory_cfg(
    model, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor,
    timesteps: list[float], guidance: float,
) -> tuple[Tensor, list[Tensor]]:
    """
    Euler ODE True-CFG, GHI LAI TOAN BO trajectory (1 phan tu moi buoc, CHUYEN VE CPU ngay de
    khong don VRAM). Mirror CHINH XAC `denoise_single_slot_with_trajectory` trong
    `probe_sequential_inpainting.py` (dong 155-202, da doc truc tiep) -- CHI khac: KHONG ghep
    ref_tokens (khong can anh tham chieu glyph o day, chi la 2 lan sinh anh full-scene thuan tuy).

    `txt`/`txt_ids` PHAI da la CFG-batch (["", prompt] -> batch=2, dung `batched_prc_txt` hoac
    tuong duong) -- ham nay se TU DOUBLE `img`/`img_ids` de khop batch=2, KHONG double txt (txt
    da double tu truoc, chi 1 lan encode).
    """
    orig_dtype = img.dtype
    trajectory: list[Tensor] = [img.clone().cpu()]
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((2,), t_curr, dtype=img.dtype, device=img.device)
        img_cfg = torch.cat([img, img], dim=0)
        img_ids_cfg = torch.cat([img_ids, img_ids], dim=0)
        pred = model(x=img_cfg, x_ids=img_ids_cfg, timesteps=t_vec, ctx=txt, ctx_ids=txt_ids, guidance=None)
        pred_uncond, pred_cond = pred.chunk(2)
        v_pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        img = (img + (t_prev - t_curr) * v_pred).to(orig_dtype)
        trajectory.append(img.clone().cpu())
    return img, trajectory


def denoise_trajectory_inpaint_exact(
    model, img_init: Tensor, known_trajectory: list[Tensor], soft_mask: Tensor,
    img_ids: Tensor, txt: Tensor, txt_ids: Tensor, timesteps: list[float], guidance: float,
) -> Tensor:
    """
    Mirror CHINH XAC `denoise_flow_matching_inpaint_exact` trong `probe_sequential_inpainting.py`
    (dong 205-281, da doc truc tiep) -- CHI khac: KHONG ghep ref_tokens, va nhan `soft_mask` da lam
    mem san (tu `soften_region_mask`) thay vi 1 mask cosine-ramp 1-truc.

    `soft_mask`: shape (L_img,), gia tri [0,1] -- 0.0 = giu Y HET latent that cua `known_trajectory`
    (Pass 1) tai moi buoc, 1.0 = de Pass 2 (model dang chay day, dieu kien boi `txt`/`txt_ids`
    RIENG cua Pass 2) tu quyet dinh hoan toan, gia tri trung gian = hoa tron tuyen tinh (vung bien
    mem). `known_trajectory`: list tra ve tu `denoise_trajectory_cfg` cua Pass 1 (PHAI cung do dai
    `timesteps`, PHAI cung h_lat/w_lat canvas).

    QUAN TRONG (dung y het file goc): moi buoc EP LAI vung mask=0 ve DUNG latent that cua Pass 1
    tai CHINH XAC buoc do (khong phai xap xi (1-t)*z+t*noise nhu Co che A) -- day la diem khac
    biet cot loi giup tranh "o gia tao": noi dung duoc ghep vao la noi dung THAT model da tung
    sinh ra, khong phai gia tri ngoai lai.
    """
    orig_dtype = img_init.dtype
    device = img_init.device
    mask = soft_mask.to(device=device, dtype=orig_dtype).view(1, -1, 1)  # (1, L_img, 1) de broadcast kenh

    img = img_init.clone()
    for step_idx in range(len(timesteps) - 1):
        t_curr, t_prev = timesteps[step_idx], timesteps[step_idx + 1]
        t_vec = torch.full((2,), t_curr, dtype=img.dtype, device=img.device)
        img_cfg = torch.cat([img, img], dim=0)
        img_ids_cfg = torch.cat([img_ids, img_ids], dim=0)
        pred = model(x=img_cfg, x_ids=img_ids_cfg, timesteps=t_vec, ctx=txt, ctx_ids=txt_ids, guidance=None)
        pred_uncond, pred_cond = pred.chunk(2)
        v_pred = pred_uncond + guidance * (pred_cond - pred_uncond)

        img_model_next = img + (t_prev - t_curr) * v_pred
        img_known_next = known_trajectory[step_idx + 1].to(device=device, dtype=orig_dtype)
        img = (mask * img_model_next + (1.0 - mask) * img_known_next).to(orig_dtype)

    # Buoc cuoi (t=0): ep dung latent sach cuoi cung cua Pass 1 tai vung mask=0, dung y het file goc.
    final_known = known_trajectory[-1].to(device=device, dtype=orig_dtype)
    img = (mask * img + (1.0 - mask) * final_known).to(orig_dtype)
    return img
