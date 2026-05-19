from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "brain" / "arbiter.py"
_SERVICE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "services" / "visual_prompt_service.py"


def _load_arbiter_module():
    class VisualPromptDecision:
        def __init__(
            self,
            *,
            selected_candidate_id=None,
            selected_prompt_type=None,
            instruction="",
            confidence=0.0,
            reason="",
            reject_all=False,
            fallback_used=False,
            arbiter_source=None,
            rendered_prompt=None,
            debug=None,
        ):
            self.selected_candidate_id = selected_candidate_id
            self.selected_prompt_type = selected_prompt_type
            self.instruction = instruction
            self.confidence = confidence
            self.reason = reason
            self.reject_all = reject_all
            self.fallback_used = fallback_used
            self.arbiter_source = arbiter_source
            self.rendered_prompt = rendered_prompt
            self.debug = debug or {}

    package = types.ModuleType("mcp_server")
    route_context = types.ModuleType("mcp_server.route_context")
    schema = types.ModuleType("mcp_server.route_context.schema")
    schema.VisualPromptDecision = VisualPromptDecision
    previous = {
        "mcp_server": sys.modules.get("mcp_server"),
        "mcp_server.route_context": sys.modules.get("mcp_server.route_context"),
        "mcp_server.route_context.schema": sys.modules.get("mcp_server.route_context.schema"),
    }
    sys.modules["mcp_server"] = package
    sys.modules["mcp_server.route_context"] = route_context
    sys.modules["mcp_server.route_context.schema"] = schema
    spec = importlib.util.spec_from_file_location("mcp_server.brain.arbiter", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


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
        self.assertIn('"prompt_prefix": _prompt_cache_prefix(', text)
        self.assertIn('"som_summary": som_summary', text)
        self.assertIn('"som_candidate_hints": som_candidate_hints', text)
        self.assertIn('"som_overlay_legend": som_overlay_legend', text)
        self.assertIn('"prompt_encoding": "compact_json_with_constraint_dsl"', text)
        self.assertIn("_compact_json_dumps(", text)
        self.assertIn("_compact_spatial_constraints_for_llm(som_spatial_constraints)", text)
        self.assertIn('"hard_rules_dsl": _join_dsl_clauses([', text)
        self.assertIn('"verification_objectives_dsl": _join_dsl_clauses([', text)
        self.assertNotIn("segmentation_safe_corridor_hints", text)
        self.assertIn("Direct regression", text)
        self.assertIn('"segmentation_validation": segmentation_validation', text)

    def test_minimal_output_coerces_to_full_path_corridor(self) -> None:
        arbiter = _load_arbiter_module()
        decision = arbiter.coerce_direct_prompt_decision(
            raw={
                "r": 0,
                "t": "c",
                "p": [
                    [[800, 827], [631, 827], [969, 827]],
                    [[777, 764], [629, 764], [924, 764]],
                    [[777, 702], [688, 702], [865, 702]],
                ],
                "c": 0.91,
            },
            route_context={"local_semantics": {"short_action_summary": "Continue forward."}},
            image_width=1600,
            image_height=900,
        )
        self.assertIsNotNone(decision)
        self.assertFalse(decision.reject_all)
        self.assertEqual(decision.selected_prompt_type, "path_corridor")
        self.assertEqual(decision.rendered_prompt["termination_semantics"], "continue_visible_corridor")
        self.assertEqual(len(decision.rendered_prompt["path_samples"]), 3)
        self.assertEqual(decision.rendered_prompt["path_samples"][0]["support_label"], "preferred_traversable")
        self.assertEqual(decision.rendered_prompt["path_uv"][0], [800.0, 827.0])
        self.assertIn("_minimal_output", decision.debug["raw"])

    def test_minimal_prompt_payload_is_ascii_safe(self) -> None:
        arbiter = _load_arbiter_module()
        old_value = os.environ.get("LVLM_MINIMAL_OUTPUT")
        os.environ["LVLM_MINIMAL_OUTPUT"] = "1"
        try:
            system_prompt = arbiter.build_direct_prompt_system_prompt()
            question = arbiter.build_direct_prompt_question(
                route_context={"local_semantics": {"short_action_summary": "继续直行"}},
                visual_regions={},
                prompt_proposals=[],
                pov_prompt_candidates=[],
                image_width=1600,
                image_height=900,
                navigation_mode="foot",
                fpv_segmentation_summary={"note": "人行道与绿化带"},
                som_summary={"labels": ["可通行", "障碍物"]},
            )
        finally:
            if old_value is None:
                os.environ.pop("LVLM_MINIMAL_OUTPUT", None)
            else:
                os.environ["LVLM_MINIMAL_OUTPUT"] = old_value
        system_prompt.encode("ascii")
        question.encode("ascii")
        self.assertIn("\\u7ee7\\u7eed\\u76f4\\u884c", question)

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
