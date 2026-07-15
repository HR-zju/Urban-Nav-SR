#!/usr/bin/env python3
"""OpenAI-compatible text adapter for official LLaVA-NeXT legacy checkpoints.

UrbanNav Module 1 is text-only.  This adapter intentionally uses the official
LLaVA-NeXT loader for the checkpoint's custom ``LlavaQwenForCausalLM`` class,
while compiling OpenAI system/user messages into the repository's Qwen ChatML
conversation template.  Image/video requests are rejected rather than being
silently ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = Field(default=512, ge=1, le=2048)
    temperature: float | None = 0.0


class LegacyLlavaTextServer:
    def __init__(self, model_path: str, model_id: str) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.tokenizer: Any = None
        self.model: Any = None
        self.lock = asyncio.Lock()

    def load(self) -> None:
        disable_torch_init()
        self.tokenizer, self.model, _image_processor, _context_len = load_pretrained_model(
            self.model_path,
            model_base=None,
            # The official loader dispatches legacy Qwen LLaVA checkpoints by
            # model name, not the config architecture string.
            model_name="llava-next-video-qwen",
            device_map="auto",
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
        )
        self.model.eval()

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise ValueError("message.content must be a string for text-only serving")
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("unsupported structured message item")
            kind = item.get("type")
            if kind == "text":
                parts.append(str(item.get("text", "")))
            elif kind in {"image_url", "input_image", "video_url", "input_video"}:
                raise ValueError("this UrbanNav LLaVA endpoint accepts text-only requests")
            else:
                raise ValueError(f"unsupported message content type: {kind!r}")
        return "\n".join(parts)

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in messages:
            role = message.get("role")
            text = self._message_text(message)
            if role == "system":
                system_parts.append(text)
            elif role == "user":
                user_parts.append(text)
            elif role == "assistant":
                # Replay runners do not rely on historical assistant turns,
                # but retaining them is safer than silently discarding state.
                user_parts.append(f"Previous assistant response:\n{text}")
            else:
                raise ValueError(f"unsupported role: {role!r}")
        if not user_parts:
            raise ValueError("at least one user message is required")

        # This legacy Video checkpoint is based on Qwen1.5-32B. Its released
        # LLaVA training prompt uses the original Qwen ChatML template.
        conv = conv_templates["qwen_1_5"].copy()
        if system_parts:
            conv.system = "<|im_start|>system\n" + "\n\n".join(system_parts)
        conv.append_message(conv.roles[0], "\n\n".join(user_parts))
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate(self, request: ChatRequest) -> str:
        prompt = self._build_prompt(request.messages)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                # The legacy LlavaQwen override consumes ``inputs`` and then
                # constructs multimodal embeddings itself. Passing the modern
                # HF ``input_ids`` keyword leaves that positional input empty.
                inputs=input_ids,
                attention_mask=attention_mask,
                do_sample=bool((request.temperature or 0.0) > 0.0),
                temperature=max(float(request.temperature or 0.0), 1e-5),
                max_new_tokens=request.max_tokens or 512,
                use_cache=True,
            )
        # Legacy LLaVA under older Transformers returns prompt + completion.
        # Newer Transformers may return only the completion when generation
        # starts from ``inputs_embeds``. Support both return conventions.
        token_ids = output_ids[0]
        if token_ids.shape[0] > input_ids.shape[1]:
            token_ids = token_ids[input_ids.shape[1] :]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def make_app(server: LegacyLlavaTextServer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        server.load()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": server.model_id, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        if request.model != server.model_id:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        try:
            async with server.lock:
                content = await asyncio.to_thread(server.generate, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": server.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(make_app(LegacyLlavaTextServer(args.model_path, args.model_id)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
