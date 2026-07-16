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
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Subset of chat-completions fields used by UrbanNav evaluation.

    Args:
        model: Model identifier exposed by the local service.
        messages: Ordered OpenAI-compatible chat messages.
        max_tokens: Maximum generated completion tokens.
        temperature: Sampling temperature; zero enables greedy decoding.
    """

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = Field(default=512, ge=1, le=2048)
    temperature: float | None = 0.0


def _configure_decoder_padding(tokenizer: Any, model: Any) -> None:
    """Configure safe left padding for decoder-only batched generation.

    Args:
        tokenizer: Tokenizer used to encode legacy ChatML prompts.
        model: Loaded LLaVA language model.

    Returns:
        None.
    """

    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "pad_token_id", None) is None:
        generation_config.pad_token_id = pad_token_id


def _continuation_token_ids(output_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Remove an echoed padded prompt only when it is an exact token prefix.

    Legacy LLaVA releases differ in whether generation from ``inputs_embeds``
    returns ``prompt + completion`` or only completion tokens. Exact prefix
    matching avoids truncating a long completion in the latter case.

    Args:
        output_ids: Generated token matrix with shape ``[B, T_out]``.
        input_ids: Left-padded prompt token matrix with shape ``[B, T_in]``.

    Returns:
        Token matrix containing only completion tokens when a prompt echo is
        present, otherwise the original output matrix.

    Raises:
        ValueError: If the tensors are not rank-two or batch sizes differ.
    """

    if output_ids.ndim != 2 or input_ids.ndim != 2:
        raise ValueError("input_ids and output_ids must be rank-two tensors")
    if output_ids.shape[0] != input_ids.shape[0]:
        raise ValueError("input and output batch sizes must match")
    input_width = input_ids.shape[1]
    if output_ids.shape[1] >= input_width and torch.equal(output_ids[:, :input_width], input_ids):
        return output_ids[:, input_width:]
    return output_ids


class LegacyLlavaTextServer:
    """Serve one legacy LLaVA checkpoint through text chat completions.

    Args:
        model_path: Local model checkpoint path.
        model_id: Model identifier exposed by the API.
        repetition_penalty: Greedy-decoding repetition penalty.
        no_repeat_ngram_size: Repeated token n-gram size to forbid.
    """

    def __init__(
        self,
        model_path: str,
        model_id: str,
        *,
        repetition_penalty: float = 1.08,
        no_repeat_ngram_size: int = 32,
    ) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.repetition_penalty = max(1.0, float(repetition_penalty))
        self.no_repeat_ngram_size = max(0, int(no_repeat_ngram_size))
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load the official legacy LLaVA checkpoint once.

        Returns:
            None.
        """

        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

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
        _configure_decoder_padding(self.tokenizer, self.model)

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
        from llava.conversation import conv_templates

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

    def generate_batch(self, requests: list[ChatRequest]) -> list[str]:
        """Generate one completion per request in a single model call.

        Args:
            requests: Non-empty requests sharing generation parameters.

        Returns:
            Decoded completion strings in request order.

        Raises:
            ValueError: If requests are empty or use different generation
                parameters.
        """

        if not requests:
            raise ValueError("requests must not be empty")
        max_tokens = requests[0].max_tokens or 512
        temperature = float(requests[0].temperature or 0.0)
        if any((request.max_tokens or 512) != max_tokens for request in requests):
            raise ValueError("batched requests must share max_tokens")
        if any(float(request.temperature or 0.0) != temperature for request in requests):
            raise ValueError("batched requests must share temperature")

        prompts = [self._build_prompt(request.messages) for request in requests]
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "use_cache": True,
        }
        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids = self.model.generate(
                # The legacy LlavaQwen override consumes ``inputs`` and then
                # constructs multimodal embeddings itself. Passing the modern
                # HF ``input_ids`` keyword leaves that positional input empty.
                inputs=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        completion_ids = _continuation_token_ids(output_ids, input_ids)
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

    def generate(self, request: ChatRequest) -> str:
        """Generate one completion through the batch-compatible path.

        Args:
            request: OpenAI-compatible request to execute.

        Returns:
            Decoded model completion.
        """

        return self.generate_batch([request])[0]


@dataclass
class _GenerationJob:
    """One queued completion request and its asynchronous result future."""

    request: ChatRequest
    future: asyncio.Future[str]


class LegacyLlavaMicroBatcher:
    """Group compatible legacy LLaVA requests into bounded microbatches.

    Args:
        server: Loaded legacy LLaVA generation server.
        max_batch_size: Maximum requests in one model call.
        batch_timeout_ms: Collection window after receiving the first request.
    """

    def __init__(
        self,
        server: LegacyLlavaTextServer,
        *,
        max_batch_size: int = 1,
        batch_timeout_ms: int = 0,
    ) -> None:
        self.server = server
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_timeout_s = max(0, int(batch_timeout_ms)) / 1000.0
        self.queue: asyncio.Queue[_GenerationJob] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the single generation worker.

        Returns:
            None.
        """

        self.worker = asyncio.create_task(self._run(), name="legacy-llava-microbatcher")

    async def close(self) -> None:
        """Cancel and await the generation worker.

        Returns:
            None.
        """

        if self.worker is not None:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    async def submit(self, request: ChatRequest) -> str:
        """Queue a request and await its decoded completion.

        Args:
            request: Compatible request to enqueue.

        Returns:
            Decoded model completion.
        """

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        await self.queue.put(_GenerationJob(request=request, future=future))
        return await future

    @staticmethod
    def _compatible(left: ChatRequest, right: ChatRequest) -> bool:
        return (left.max_tokens or 512) == (right.max_tokens or 512) and float(
            left.temperature or 0.0
        ) == float(right.temperature or 0.0)

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
                if self._compatible(first.request, candidate.request):
                    batch.append(candidate)
                else:
                    deferred.append(candidate)

            try:
                completions = await asyncio.to_thread(
                    self.server.generate_batch,
                    [job.request for job in batch],
                )
                for job, completion in zip(batch, completions, strict=True):
                    if not job.future.done():
                        job.future.set_result(completion)
            except Exception as exc:
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                for _job in batch + deferred:
                    self.queue.task_done()
                for job in deferred:
                    await self.queue.put(job)


def make_app(
    server: LegacyLlavaTextServer,
    *,
    max_batch_size: int = 1,
    batch_timeout_ms: int = 0,
) -> FastAPI:
    """Build the OpenAI-compatible application around one model instance.

    Args:
        server: Legacy LLaVA model service.
        max_batch_size: Maximum compatible requests per generation call.
        batch_timeout_ms: Queue collection window in milliseconds.

    Returns:
        Configured FastAPI application.
    """

    batcher = LegacyLlavaMicroBatcher(
        server,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        server.load()
        await batcher.start()
        yield
        await batcher.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": server.model_id, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        if request.model != server.model_id:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        try:
            content = await batcher.submit(request)
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
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--batch-timeout-ms", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=32)
    args = parser.parse_args()

    import uvicorn

    server = LegacyLlavaTextServer(
        args.model_path,
        args.model_id,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    app = make_app(
        server,
        max_batch_size=max(1, args.max_batch_size),
        batch_timeout_ms=max(0, args.batch_timeout_ms),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
