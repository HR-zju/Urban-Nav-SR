from __future__ import annotations

import ast
from pathlib import Path
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "brain" / "arbiter.py"
_SERVICE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "services" / "visual_prompt_service.py"


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
            self.assertIn("som_summary", names)
            self.assertIn("som_candidate_hints", names)
            self.assertIn("som_overlay_legend", names)
            self.assertIn("som_spatial_constraints", names)
        self.assertIn("segmentation_validation", _function_arg_names(verifier_builder))

    def test_source_contains_alignment_context_keys(self) -> None:
        text = _MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"alignment_report": alignment_report', text)
        self.assertIn('"overlay_legend": overlay_legend', text)
        self.assertIn('"summary_source": str(summary_source or "from_same_mask_entity")', text)
        self.assertIn('"som_summary": som_summary', text)
        self.assertIn('"som_candidate_hints": som_candidate_hints', text)
        self.assertIn('"som_overlay_legend": som_overlay_legend', text)
        self.assertIn('"som_spatial_constraints": som_spatial_constraints', text)
        self.assertNotIn("segmentation_safe_corridor_hints", text)
        self.assertIn("Direct regression", text)
        self.assertIn('"segmentation_validation": segmentation_validation', text)

    def test_visual_prompt_service_does_not_promote_segmentation_geometry_to_final(self) -> None:
        text = _SERVICE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("_build_segmentation_safe_corridor_hint", text)
        self.assertNotIn("_prefer_segmentation_safe_corridor", text)
        self.assertNotIn("segmentation_safe_corridor_guard", text)
        self.assertNotIn("segmentation_safe_corridor_hints", text)
        self.assertNotIn("llm_direct_verified_generator_kept", text)
        self.assertNotIn("verifier_invalid_kept_generator", text)


if __name__ == "__main__":
    unittest.main()
