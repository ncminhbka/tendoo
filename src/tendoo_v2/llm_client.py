"""
src/tendoo_v2/llm_client.py

Wires tendoo_v2's role/group/zone schema (see schema.py's docstring) to a REAL,
already-hosted LLM instead of the hand-written mock Blocks used so far this round --
per explicit instruction: "wire tạm v2 vào production, dùng LLM đã host ... để tôi
lên server test, lỗi gì thì fix sau" (2026-09-14). Deliberately minimal: talks
OpenAI-compatible /v1/chat/completions HTTP (the LLM team's already-running sidecar,
NOT the separate local-transformers Qwen3 sidecar in llm_render_plan_server.py --
different infra, different model list), gets back a JSON array of Blocks.

Endpoint/model: an internal OpenAI-compatible sidecar the LLM team already hosts (see
TENDOO_V2_LLM_BASE_URL below). Model list included Qwen/Qwen3.6-35B-A3B,
Qwen/Qwen3.8-27B, google/gemma-4-31B-it, Qwen/Qwen3.5-4B (+ 2 embedding/reranker
models, irrelevant here). Default is Qwen/Qwen3.8-27B (2026-09-14: explicitly picked
over the earlier smaller default when wiring this into production, per direct
instruction) -- override via TENDOO_V2_LLM_MODEL env var to try the others.

SECRET HANDLING (2026-09-14 fix): the bearer token has NO hardcoded fallback (unlike
an earlier revision of this file, which briefly committed the real internal token as
a source-code default -- already pushed to remote history at that point; rotating
the token server-side is the real fix, this file change only stops it from
recurring). TENDOO_V2_LLM_API_KEY MUST be set in the environment; every call fails
fast with a clear error otherwise (see call_llm_raw below) rather than silently
sending an empty/placeholder Authorization header.

DELIBERATELY REUSES 2 hard-won lessons from src/tendoo/render_plan.py +
llm_render_plan_server.py (do not reintroduce either bug):
  1. Qwen3's thinking mode can eat the entire token budget before ever reaching the
     JSON answer (confirmed on a real server run, 2026-09-10) -- disabled via
     `chat_template_kwargs: {"enable_thinking": false}` in the request body (the vLLM/
     OpenAI-compatible-server convention for Qwen3), plus a defensive <think> strip.
  2. A naive greedy `re.search(r"\\[.*\\]", ..., re.DOTALL)` over-captures if the model
     emits trailing prose after the JSON -- ported render_plan.py's balanced-bracket
     scanner idea, adapted for a top-level JSON ARRAY instead of an object.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from tendoo.engine.geometry import GRID_ZONE_NAMES
from tendoo_v2.schema import Block, ROLE_VALUES

# The 9 positions an LLM-extracted `zone` may legitimately name -- the 8 perimeter
# grid cells (auto-assignable to floating blocks too) PLUS "middle_center" (the
# true dead-center spot, deliberately excluded from GRID_ZONE_NAMES/auto-assignment
# since it usually belongs to the product/subject -- reachable only via an
# explicit pin, i.e. a real user request naming it). Single source of truth for
# validating the LLM's raw `zone` string; matches tendoo.engine.geometry's own
# canonical zone vocabulary instead of hand-duplicating it.
VALID_LLM_ZONES = frozenset(GRID_ZONE_NAMES) | {"middle_center"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LLM_BASE_URL = os.environ.get("TENDOO_V2_LLM_BASE_URL", "http://10.221.155.3:8004/v1")
# No hardcoded fallback on purpose -- see module docstring's "SECRET HANDLING" note.
LLM_API_KEY = os.environ.get("TENDOO_V2_LLM_API_KEY")
LLM_MODEL = os.environ.get("TENDOO_V2_LLM_MODEL", "Qwen/Qwen3.8-27B")
LLM_TIMEOUT_S = float(os.environ.get("TENDOO_V2_LLM_TIMEOUT_S", "45"))

# Same convention as render_plan.py's RENDER_PLAN_LOG_PATH: a plain module-level
# path so it can be redirected in tests, never written to during pytest runs.
LLM_LOG_PATH = PROJECT_ROOT / "logs" / "tendoo_v2_llm_requests.jsonl"

# Fields that are NEVER handed to the LLM's curation authority -- store/contact info
# stays outside its scope entirely (matches llm_render_plan_server.py's rule #1:
# "Store/contact fields ... always render -- unconditional, handled entirely outside
# this module"). Business's one stated exception ("trừ phần thông tin cửa hàng") to
# otherwise-full LLM curation authority (2026-09-14 discussion) lives here, in code,
# not as a prompt instruction the model could ignore.
STORE_INFO_KEYS = frozenset({"store_name", "phone", "address"})

__all__ = [
    "LLM_BASE_URL", "LLM_MODEL", "LLM_LOG_PATH", "STORE_INFO_KEYS",
    "build_messages", "call_llm_raw", "parse_blocks_json",
    "call_llm_for_blocks",
]


# ==================================================================================
# Prompt construction
# ==================================================================================

SYSTEM_PROMPT = """Bạn là chuyên gia biên tập nội dung (content curator) cho một hệ thống tạo poster \
thương mại tiếng Việt. Bạn KHÔNG quyết định vị trí/kích thước pixel -- một solver tất định ở tầng \
sau sẽ tự lo việc đó. Việc của bạn CHỈ là: chọn nội dung nào nên vẽ, xếp hạng độ quan trọng, và nhóm \
các nội dung nên đứng cạnh nhau.

BẠN NHẬN: danh sách các trường đã điền (có thể trống) + 1 prompt tự do (mô tả ý muốn của người dùng).
BẠN TRẢ VỀ: DUY NHẤT một mảng JSON các "block" chữ sẽ được vẽ lên poster, không kèm giải thích, không \
markdown, không thẻ <think>. Mỗi block có dạng:
{{"text": "...", "role": "hero|subhead|body|meta|cta", "group": "ten_nhom hoặc null", \
"zone": "ten_zone hoặc null", "field": "ten_truong_nguon hoặc null"}}

QUY TẮC:
1. "role" là XẾP HẠNG độ quan trọng, KHÔNG phải cỡ chữ: "hero" = thông điệp nổi bật nhất (có thể có \
   NHIỀU hero nếu nội dung thực sự có nhiều điểm nhấn ngang nhau -- solver sẽ tự tách vị trí, bạn \
   không cần lo việc đó); "subhead" = phụ đề bổ trợ cho 1 hero; "body" = đoạn mô tả/câu dài; "cta" = \
   lời kêu gọi hành động; "meta" = chi tiết phụ (ngày tháng, thông số, tag ngắn).
2. "group": đặt CÙNG 1 tên group cho các block PHẢI đứng cạnh nhau thành 1 cụm (ví dụ 1 hero luôn đi \
   kèm 1 subhead của riêng nó, hoặc các bước hướng dẫn 1-2-3 đứng thành 1 cột). Để null nếu block đó \
   đứng độc lập.
3. "zone" CHỈ điền khi prompt tự do nêu RÕ RÀNG 1 vị trí cụ thể (ví dụ "góc trên bên trái" -> \
   "top_left", "phía dưới cùng" -> "bottom_center", "chính giữa"/"ở giữa khung hình" -> \
   "middle_center"). Các zone hợp lệ, PHẢI CHÉP Y NGUYÊN 1 trong 9 chuỗi sau (không tự bịa biến \
   thể khác như "center"/"middle"/"giua"): top_left, top_center, top_right, middle_left, \
   middle_center, middle_right, bottom_left, bottom_center, bottom_right. Riêng "middle_center" \
   CHỈ dùng khi prompt yêu cầu rõ ràng đặt chữ ở NGAY GIỮA khung hình -- không tự chọn zone này khi \
   prompt không nói vị trí, vì "chính giữa" thường là chỗ dành cho sản phẩm/chủ thể ảnh nền. TUYỆT \
   ĐỐI KHÔNG tự bịa 1 vị trí nếu prompt không nói -- để null, solver sẽ tự chọn chỗ hợp lý.
4. "field": nếu nội dung của block bắt nguồn từ 1 trường trong danh sách đã điền, ghi lại TÊN trường đó \
   (cho phép TÁCH 1 trường dài thành NHIỀU block ngắn hơn, mỗi block cùng field đó -- ví dụ trường \
   product_desc dài có thể tách thành 2 block "meta" ngắn). Nếu block là nội dung mới trích từ prompt \
   tự do (không khớp trường nào), để null.
5. Bạn ĐƯỢC PHÉP: chọn lọc (không cần vẽ hết mọi trường đã điền), gộp, tách, và viết lại ngắn gọn/dễ \
   hiểu hơn (ví dụ đổi định dạng ngày). TUYỆT ĐỐI KHÔNG được bịa thêm sự kiện/số liệu không có trong \
   trường/prompt đã cho (không hallucinate).
6. KHÔNG PHẢI xử lý thông tin cửa hàng (tên/SĐT/địa chỉ) -- các trường đó đã bị loại khỏi danh sách \
   gửi cho bạn, không nằm trong phạm vi bạn quyết định.
7. Nếu không có nội dung nào đáng vẽ (mọi trường trống, prompt không có yêu cầu chữ), trả về mảng RỖNG: [].

CHỈ xuất ra DUY NHẤT 1 mảng JSON, không kèm bất kỳ chữ nào khác, không dùng markdown code fence."""

USER_PROMPT_TEMPLATE = """CÁC TRƯỜNG ĐÃ ĐIỀN (không phải thông tin cửa hàng):
{fields_block}

PROMPT TỰ DO: {free_prompt}
"""


def _format_fields(fields: Dict[str, str]) -> str:
    curatable = {k: v for k, v in fields.items() if k not in STORE_INFO_KEYS}
    if not curatable:
        return "(không có trường nào)"
    return "\n".join(f"- {k}: {v if v else '(trống)'}" for k, v in curatable.items())


def build_messages(fields: Dict[str, str], free_prompt: str) -> List[Dict[str, str]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        fields_block=_format_fields(fields),
        free_prompt=free_prompt or "(không có)",
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ==================================================================================
# HTTP call
# ==================================================================================

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def call_llm_raw(messages: List[Dict[str, str]], model: Optional[str] = None, timeout: float = LLM_TIMEOUT_S) -> str:
    """POSTs to the hosted OpenAI-compatible /v1/chat/completions endpoint, returns
    the assistant message's raw text content. Raises requests.RequestException /
    KeyError on transport or malformed-response failure -- callers (call_llm_for_blocks)
    catch broadly so a sidecar hiccup never crashes the caller, mirroring
    demo_server.py's "never let an LLM sidecar failure break the request" philosophy."""
    if not LLM_API_KEY:
        raise RuntimeError(
            "TENDOO_V2_LLM_API_KEY is not set -- refusing to call the LLM endpoint "
            "without a real bearer token (see llm_client.py's SECRET HANDLING note)."
        )
    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 900,
        # vLLM/OpenAI-compatible-server convention for Qwen3 -- see module docstring's
        # lesson (1). Harmless no-op on non-Qwen3 models (gemma/etc. ignore it).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ==================================================================================
# JSON array extraction & parsing (balanced-bracket scan -- see module docstring's
# lesson (2); mirrors render_plan.py::_extract_first_json_object but for a top-level
# array instead of an object).
# ==================================================================================

def _extract_first_json_array(raw_response: str) -> Optional[str]:
    start = raw_response.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(raw_response)):
        ch = raw_response[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw_response[start : i + 1]
    return None


def parse_blocks_json(raw_response: str) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    """Never raises -- returns (None, [error]) on any parse failure."""
    cleaned = _THINK_TAG_RE.sub("", raw_response).strip()
    candidate = _extract_first_json_array(cleaned)
    if candidate is None:
        return None, ["No JSON array found in model response."]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        return None, [f"JSON parse failure: {e}"]
    if not isinstance(parsed, list):
        return None, [f"Expected a JSON array, got {type(parsed).__name__}"]
    return parsed, []


def _to_blocks(raw_items: List[Dict[str, Any]]) -> Tuple[List[Block], List[str]]:
    """Converts raw dicts to validated Block instances, dropping (not crashing on)
    anything malformed -- an unreliable small model WILL occasionally emit a bad
    role or a non-dict item; one bad block must never sink the whole plan."""
    blocks: List[Block] = []
    errors: List[str] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict) or not item.get("text"):
            errors.append(f"item {idx}: missing/invalid 'text', dropped")
            continue
        role = item.get("role") or "body"
        if role not in ROLE_VALUES:
            errors.append(f"item {idx}: invalid role={role!r}, defaulted to 'body'")
            role = "body"
        zone = item.get("zone") or None
        if zone is not None and zone not in VALID_LLM_ZONES:
            # NEVER silently trust an unrecognized zone string through to the
            # solver: get_zone_bounding_box() falls back to a top_left-shaped
            # rect for any unknown name (geometry.py's defensive default), which
            # would place content at the WRONG position with zero visible error
            # -- e.g. the model writing "center" instead of "middle_center" would
            # silently land the block at top-left instead of the user's actual
            # "chính giữa" request. Drop to None instead (matches the "solver
            # picks a sensible spot" fallback for a block with no real zone),
            # and surface it so the caller/log can see the model hallucinated.
            errors.append(f"item {idx}: invalid zone={zone!r} (not in VALID_LLM_ZONES), dropped to None")
            zone = None
        try:
            blocks.append(Block(
                text=str(item["text"]),
                role=role,
                field=item.get("field") or None,
                group=item.get("group") or None,
                zone=zone,
            ))
        except ValueError as e:
            errors.append(f"item {idx}: {e}, dropped")
    return blocks, errors


def _log_llm_io(
    fields: Dict[str, str], free_prompt: str, model: str,
    raw_response: Optional[str], blocks: Optional[List[Dict[str, Any]]],
    errors: List[str], latency_s: float,
) -> None:
    """Best-effort JSONL append -- mirrors render_plan.py's _log_render_plan_io.
    A logging failure must never break the actual request."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {"fields": fields, "free_prompt": free_prompt, "model": model},
            "output": {"raw_response": raw_response, "blocks": blocks, "errors": errors},
            "latency_s": round(latency_s, 2),
        }
        LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [tendoo_v2 LLM log] Notice: failed to write log entry: {e}")


def call_llm_for_blocks(
    fields: Dict[str, str], free_prompt: str, model: Optional[str] = None, timeout: float = LLM_TIMEOUT_S,
) -> Tuple[List[Block], List[str]]:
    """End-to-end: build prompt -> call hosted LLM -> parse -> validate -> Block list.
    NEVER raises -- any failure (network, timeout, bad JSON, bad role) surfaces as an
    entry in the returned errors list and an empty/partial block list, so a caller can
    always fall back gracefully (e.g. store-info-only poster) instead of crashing."""
    resolved_model = model or LLM_MODEL
    t0 = time.time()
    messages = build_messages(fields, free_prompt)
    raw_response: Optional[str] = None
    try:
        raw_response = call_llm_raw(messages, model=resolved_model, timeout=timeout)
    except Exception as e:
        errors = [f"LLM call failed: {e}"]
        _log_llm_io(fields, free_prompt, resolved_model, None, None, errors, time.time() - t0)
        return [], errors

    raw_items, parse_errors = parse_blocks_json(raw_response)
    if raw_items is None:
        preview = raw_response[:400].replace("\n", " ")
        errors = parse_errors + [f"raw model output preview: {preview!r}"]
        _log_llm_io(fields, free_prompt, resolved_model, raw_response, None, errors, time.time() - t0)
        return [], errors

    blocks, conversion_errors = _to_blocks(raw_items)
    errors = parse_errors + conversion_errors
    _log_llm_io(fields, free_prompt, resolved_model, raw_response, raw_items, errors, time.time() - t0)
    return blocks, errors
