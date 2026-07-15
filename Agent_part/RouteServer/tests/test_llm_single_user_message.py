from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from mcp_server.llm import LLMSettings, _apply_chat_template_kwargs


class SingleUserMessageAdapterTest(unittest.TestCase):
    """Tests for local VLM servers that reject system chat roles."""

    def test_default_preserves_chat_messages(self) -> None:
        payload = {
            "model": "molmo2-4b",
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Request: find coffee"},
            ],
        }

        with patch.dict(os.environ, {"LLM_SINGLE_USER_MESSAGE": "0"}, clear=False):
            adapted = _apply_chat_template_kwargs(payload, _settings())

        self.assertEqual(adapted["messages"], payload["messages"])

    def test_single_user_message_collapses_roles(self) -> None:
        payload = {
            "model": "molmo2-4b",
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Request: find coffee"},
                {"role": "assistant", "content": '{"search_keywords":"cafe"}'},
                {"role": "user", "content": "Request: find tea"},
            ],
        }

        with patch.dict(os.environ, {"LLM_SINGLE_USER_MESSAGE": "1"}, clear=False):
            adapted = _apply_chat_template_kwargs(payload, _settings())

        self.assertEqual(len(adapted["messages"]), 1)
        self.assertEqual(adapted["messages"][0]["role"], "user")
        content = adapted["messages"][0]["content"]
        self.assertIn("[SYSTEM INSTRUCTIONS]\nReturn JSON only.", content)
        self.assertIn("[EXAMPLE ASSISTANT OUTPUT]\n", content)
        self.assertIn("Request: find tea", content)


def _settings() -> LLMSettings:
    return LLMSettings(
        provider="openai",
        api_style="chat_completions",
        api_key="dummy",
        model="molmo2-4b",
        base_url="http://127.0.0.1:18452/v1",
    )


if __name__ == "__main__":
    unittest.main()
