#!/usr/bin/env python3
"""Expose a local Hugging Face chat model through a minimal OpenAI-compatible API.

This server is intentionally limited to non-streaming, text-only chat requests.
It is used for benchmark models whose custom architectures are not supported by
vLLM, while preserving the client contract used by RouteServer evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoProcessor


class ChatMessage(BaseModel):
    """One OpenAI-compatible chat message."""

    role: str
    content: Any


class ChatRequest(BaseModel):
    """Subset of chat-completions fields used by the benchmark client."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=0.0, ge=0.0)
    stream: bool = False


def _text_content(content: Any) -> str:
    """Normalize OpenAI string or content-part payloads to text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _model_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert OpenAI messages to the model's text chat-template structure."""

    return [
        {"role": message.role, "content": [{"type": "text", "text": _text_content(message.content)}]}
        for message in messages
    ]


def parse_args() -> argparse.Namespace:
    """Parse server configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--batch-timeout-ms", type=int, default=0)
    return parser.parse_args()


@dataclass
class _GenerationJob:
    """One queued request with a future for its decoded continuation."""

    messages: list[ChatMessage]
    max_new_tokens: int
    temperature: float
    future: asyncio.Future[str]


class _MicroBatcher:
    """Short-window, same-generation-parameter batching for HF generation."""

    def __init__(
        self,
        *,
        processor: Any,
        model: Any,
        max_batch_size: int,
        batch_timeout_ms: int,
    ) -> None:
        self.processor = processor
        self.model = model
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_ms / 1000.0
        self.queue: asyncio.Queue[_GenerationJob] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.worker = asyncio.create_task(self._run(), name="hf-generation-microbatcher")

    async def close(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    async def submit(
        self,
        *,
        messages: list[ChatMessage],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        await self.queue.put(
            _GenerationJob(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                future=future,
            )
        )
        return await future

    def _generate_batch(self, batch: list[_GenerationJob]) -> list[str]:
        prompts = [
            self.processor.apply_chat_template(
                _model_messages(job.messages), tokenize=False, add_generation_prompt=True
            )
            for job in batch
        ]
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to("cuda")
        temperature = batch[0].temperature
        kwargs: dict[str, Any] = {
            "max_new_tokens": batch[0].max_new_tokens,
            "do_sample": temperature > 0,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        return self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deferred: list[_GenerationJob] = []
            deadline = time.monotonic() + self.batch_timeout_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if (
                    candidate.max_new_tokens == first.max_new_tokens
                    and candidate.temperature == first.temperature
                ):
                    batch.append(candidate)
                else:
                    deferred.append(candidate)

            try:
                completions = await asyncio.to_thread(self._generate_batch, batch)
                for job, completion in zip(batch, completions, strict=True):
                    if not job.future.done():
                        job.future.set_result(completion)
            except Exception as exc:
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                for _ in batch + deferred:
                    self.queue.task_done()
                for job in deferred:
                    await self.queue.put(job)


def main() -> None:
    """Load the model once and serve OpenAI-compatible text completions."""

    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    batcher = _MicroBatcher(
        processor=processor,
        model=model,
        max_batch_size=max(1, args.max_batch_size),
        batch_timeout_ms=max(0, args.batch_timeout_ms),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await batcher.start()
        yield
        await batcher.close()

    app = FastAPI(title="HF Text Chat", lifespan=lifespan)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": args.model_id, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported")
        if request.model and request.model != args.model_id:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        max_new_tokens = min(request.max_tokens or args.max_new_tokens, args.max_new_tokens)
        completion = await batcher.submit(
            messages=request.messages,
            max_new_tokens=max_new_tokens,
            temperature=float(request.temperature or 0.0),
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": completion}, "finish_reason": "stop"}],
        }

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
