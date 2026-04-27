from __future__ import annotations

import ast
from pathlib import Path
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "brain" / "arbiter.py"


def _function_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = [arg.arg for arg in node.args.args]
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    return args


class ArbiterPromptContextTest(unittest.TestCase):
    def test_question_builders_accept_new_alignment_context_args(self) -> None:
        tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
        question_builder = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_direct_prompt_question"
        )
        verifier_builder = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_direct_prompt_verifier_question"
        )
        for node in [question_builder, verifier_builder]:
            names = _function_arg_names(node)
            self.assertIn("alignment_report", names)
            self.assertIn("overlay_legend", names)
            self.assertIn("summary_source", names)

    def test_source_contains_alignment_context_keys(self) -> None:
        text = _MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"alignment_report": alignment_report', text)
        self.assertIn('"overlay_legend": overlay_legend', text)
        self.assertIn('"summary_source": str(summary_source or "from_same_mask_entity")', text)


if __name__ == "__main__":
    unittest.main()
