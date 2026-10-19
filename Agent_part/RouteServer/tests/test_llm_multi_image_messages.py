from __future__ import annotations

import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from mcp_server.llm import LLMSettings, VisionAnalyzer


LLM_FILE = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "llm.py"


def _function_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = [arg.arg for arg in node.args.args]
    if node.args.kwonlyargs:
        args.extend(arg.arg for arg in node.args.kwonlyargs)
    return args


class LlmMultiImageMessagesTest(unittest.TestCase):
    def test_vision_analyzer_accepts_extra_image_urls(self) -> None:
        tree = ast.parse(LLM_FILE.read_text(encoding="utf-8"))
        vision_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VisionAnalyzer"
        )
        build_messages = next(
            node for node in vision_class.body if isinstance(node, ast.FunctionDef) and node.name == "build_messages"
        )
        analyze = next(
            node for node in vision_class.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "analyze"
        )
        self.assertIn("extra_image_urls", _function_arg_names(build_messages))
        self.assertIn("extra_image_urls", _function_arg_names(analyze))


    def test_vision_request_controls_are_explicit_opt_ins(self) -> None:
        settings = LLMSettings(
            provider="openai",
            api_style="chat_completions",
            api_key="test",
            model="test-model",
            base_url="http://127.0.0.1:1/v1",
        )
        analyzer = VisionAnalyzer(settings=settings)

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(analyzer._vision_max_tokens())
            self.assertFalse(analyzer._vision_json_response_enabled())

        with patch.dict(
            os.environ,
            {
                "LLM_VISION_MAX_TOKENS": "1536",
                "LLM_VISION_RESPONSE_FORMAT_JSON": "true",
            },
            clear=True,
        ):
            self.assertEqual(analyzer._vision_max_tokens(), 1536)
            self.assertTrue(analyzer._vision_json_response_enabled())

    def test_invalid_vision_token_cap_fails_closed(self) -> None:
        analyzer = VisionAnalyzer(
            settings=LLMSettings(
                provider="openai",
                api_style="chat_completions",
                api_key="test",
                model="test-model",
                base_url="http://127.0.0.1:1/v1",
            )
        )
        with patch.dict(
            os.environ,
            {"LLM_VISION_MAX_TOKENS": "not-an-integer"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "LLM_VISION_MAX_TOKENS must be an integer",
            ):
                analyzer._vision_max_tokens()


if __name__ == "__main__":
    unittest.main()
