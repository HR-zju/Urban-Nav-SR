from __future__ import annotations

from types import SimpleNamespace

from tools.evaluation.serve_hf_text_chat import _configure_decoder_padding


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


def test_configure_decoder_padding_uses_left_padding_and_eos() -> None:
    tokenizer = _FakeTokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(generation_config=SimpleNamespace(pad_token_id=None))

    _configure_decoder_padding(processor, model)

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert model.generation_config.pad_token_id == tokenizer.eos_token_id
