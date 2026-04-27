from __future__ import annotations

import ast
from pathlib import Path
import unittest


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

    def test_arbiter_functions_accept_extra_image_urls(self) -> None:
        tree = ast.parse(LLM_FILE.read_text(encoding="utf-8"))
        arbitrate_remote = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "arbitrate_pov_candidates_with_llm"
        )
        arbitrate_local = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "arbitrate_local_pov_candidates_with_llm"
        )
        self.assertIn("extra_image_urls", _function_arg_names(arbitrate_remote))
        self.assertIn("extra_image_urls", _function_arg_names(arbitrate_local))


if __name__ == "__main__":
    unittest.main()
