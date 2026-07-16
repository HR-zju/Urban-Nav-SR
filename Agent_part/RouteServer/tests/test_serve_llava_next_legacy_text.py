from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.evaluation.serve_llava_next_legacy_text import (
    ChatRequest,
    LegacyLlavaMicroBatcher,
    _configure_decoder_padding,
    _continuation_token_ids,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 7
        self._pad_token = None

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        self._pad_token = value
        if value == self.eos_token:
            self.pad_token_id = self.eos_token_id


class _FakeServer:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.batches: list[list[ChatRequest]] = []

    def generate_batch(self, requests: list[ChatRequest]) -> list[str]:
        self.batches.append(requests)
        if self.failure is not None:
            raise self.failure
        return [str(request.messages[0]["content"]) for request in requests]


def _request(text: str) -> ChatRequest:
    return ChatRequest(
        model="legacy-llava",
        messages=[{"role": "user", "content": text}],
        max_tokens=64,
        temperature=0.0,
    )


async def _run_inline(function, *args):
    return function(*args)


def test_configure_decoder_padding_uses_left_padding_and_eos() -> None:
    tokenizer = _FakeTokenizer()
    model = SimpleNamespace(generation_config=SimpleNamespace(pad_token_id=None))

    _configure_decoder_padding(tokenizer, model)

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert model.generation_config.pad_token_id == tokenizer.eos_token_id


def test_continuation_tokens_remove_only_an_exact_prompt_prefix() -> None:
    input_ids = torch.tensor([[0, 2, 3], [0, 4, 5]])
    completions = torch.tensor([[9, 10], [11, 12]])
    echoed = torch.cat([input_ids, completions], dim=1)
    completion_only = torch.tensor([[8, 9, 10, 11], [12, 13, 14, 15]])

    assert torch.equal(_continuation_token_ids(echoed, input_ids), completions)
    assert torch.equal(_continuation_token_ids(completion_only, input_ids), completion_only)


def test_compatible_requests_share_one_model_call() -> None:
    async def exercise() -> tuple[list[str], list[int]]:
        server = _FakeServer()
        batcher = LegacyLlavaMicroBatcher(server, max_batch_size=2, batch_timeout_ms=100)
        await batcher.start()
        try:
            outputs = await asyncio.gather(
                batcher.submit(_request("first")),
                batcher.submit(_request("second")),
            )
        finally:
            await batcher.close()
        return outputs, [len(batch) for batch in server.batches]

    with patch("tools.evaluation.serve_llava_next_legacy_text.asyncio.to_thread", new=_run_inline):
        outputs, batch_sizes = asyncio.run(exercise())
    assert outputs == ["first", "second"]
    assert batch_sizes == [2]


def test_batch_failure_reaches_every_waiting_request() -> None:
    async def exercise() -> tuple[list[object], int]:
        server = _FakeServer(failure=RuntimeError("generation failed"))
        batcher = LegacyLlavaMicroBatcher(server, max_batch_size=2, batch_timeout_ms=100)
        await batcher.start()
        try:
            outputs = await asyncio.gather(
                batcher.submit(_request("first")),
                batcher.submit(_request("second")),
                return_exceptions=True,
            )
        finally:
            await batcher.close()
        return outputs, len(server.batches)

    with patch("tools.evaluation.serve_llava_next_legacy_text.asyncio.to_thread", new=_run_inline):
        outputs, batch_count = asyncio.run(exercise())
    assert batch_count == 1
    assert all(isinstance(output, RuntimeError) for output in outputs)
