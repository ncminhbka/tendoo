#!/usr/bin/env python3
"""
src/tendoo_v3/llm_sidecar.py

Standalone LLM Sidecar Service cho Tendoo v3 (Học tập từ Phase C llm_render_plan_server.py):
==========================================================================================
- Chạy như một tiến trình độc lập (Microservice), không phụ thuộc vào tiến trình Diffusion.
- Nạp trực tiếp checkpoint Qwen3-4B-FP8 có sẵn trên máy chủ (/home/jovyan/persistent-data/...)
  lên GPU (mặc định cuda:1 để không tranh chấp với DiT 4B trên cuda:0).
- Cung cấp:
  1. Endpoint chuẩn OpenAI: `POST /v1/chat/completions` (tương thích curl, Postman, vLLM client).
  2. Endpoint trực tiếp: `POST /api/v3/creative-plan` (tiếp nhận form + prompt -> xuất TendooCreativePlan).
  3. Endpoint kiểm tra sức khỏe: `GET /api/health`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from tendoo_v3.llm_planner import (
    SYSTEM_PROMPT,
    extract_balanced_json,
    generate_local_qwen_response,
    load_local_qwen3,
    free_local_qwen3,
    resolve_qwen3_checkpoint_path,
    _finalize_plan_from_dict,
    fallback_heuristic_planner,
)

logger = logging.getLogger("TendooV3.LLMSidecar")

# Global model pointers
MODEL: Any = None
TOKENIZER: Any = None
DEVICE: str = "cuda:1"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "Qwen/Qwen3-4B-FP8"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 1536
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    extra_body: Optional[Dict[str, Any]] = None


class CreativePlanApiRequest(BaseModel):
    form_data: Dict[str, Any] = {}
    prompt: str = ""
    aspect_ratio: str = "1:1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER, DEVICE
    logger.info("=" * 70)
    logger.info("🚀 STARTING TENDOO v3 LOCAL QWEN3-4B SIDECAR")
    logger.info(f"   Target Device: {DEVICE}")
    logger.info("=" * 70)
    try:
        MODEL, TOKENIZER = load_local_qwen3(device=DEVICE)
        logger.info("✅ Qwen3-4B Model & Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load local Qwen3 weights: {e}", exc_info=True)
    yield
    logger.info("Releasing sidecar GPU resources...")
    free_local_qwen3()


app = FastAPI(title="Tendoo v3 Local LLM Sidecar", lifespan=lifespan)


@app.get("/api/health")
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
        "model_path": resolve_qwen3_checkpoint_path("4B"),
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Endpoint tương thích chuẩn OpenAI / vLLM chat completions."""
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Local Qwen3 model not loaded")

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    max_tokens = req.max_tokens or 1536

    t0 = time.time()
    try:
        content = generate_local_qwen_response(
            MODEL, TOKENIZER, raw_messages, max_new_tokens=max_tokens
        )
        latency = round(time.time() - t0, 3)
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "Qwen/Qwen3-4B-FP8",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "latency_seconds": latency,
        }
    except Exception as e:
        logger.error(f"Error during text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v3/creative-plan")
async def api_creative_plan(req: CreativePlanApiRequest):
    """Endpoint trực tiếp tiếp nhận form + prompt sinh TendooCreativePlan v3."""
    if MODEL is None or TOKENIZER is None:
        logger.warning("Sidecar model not loaded, falling back to heuristic.")
        plan = fallback_heuristic_planner(req.form_data, req.prompt, req.aspect_ratio)
        return {"status": "fallback_heuristic", "plan": plan.to_dict()}

    import json
    user_payload = {
        "form_fields": req.form_data,
        "user_prompt": req.prompt,
        "aspect_ratio": req.aspect_ratio,
    }
    user_message = f"Dữ liệu người dùng nhập:\n```json\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n```"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    t0 = time.time()
    try:
        raw_text = generate_local_qwen_response(MODEL, TOKENIZER, messages, max_new_tokens=1536)
        extracted = extract_balanced_json(raw_text)
        if not extracted:
            plan = fallback_heuristic_planner(req.form_data, req.prompt, req.aspect_ratio)
            return {"status": "fallback_unparseable", "plan": plan.to_dict(), "raw_text": raw_text}

        plan = _finalize_plan_from_dict(extracted, req.form_data)
        return {
            "status": "success",
            "plan": plan.to_dict(),
            "latency_seconds": round(time.time() - t0, 3),
        }
    except Exception as e:
        logger.error(f"Sidecar generation error: {e}", exc_info=True)
        plan = fallback_heuristic_planner(req.form_data, req.prompt, req.aspect_ratio)
        return {"status": "fallback_exception", "error": str(e), "plan": plan.to_dict()}


def parse_args():
    parser = argparse.ArgumentParser(description="Tendoo v3 Local Qwen3 Sidecar Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8004, help="Port to listen on (e.g. 8004 or 7861)")
    parser.add_argument("--device", type=str, default=None, help="Device to load model on (default: cuda:1 if multi-GPU, else cuda:0)")
    return parser.parse_args()


def main():
    global DEVICE
    args = parse_args()
    if args.device:
        DEVICE = args.device
    else:
        import torch
        DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
