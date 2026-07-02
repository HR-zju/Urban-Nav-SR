from __future__ import annotations

import json
import re
import unittest

from mcp_server.agent.finish_schema import finish_trajectory_matches, parse_finish_trajectory_norm
from mcp_server.agent.schemas import AGENT_STAGE2_CASE_SCHEMA_VERSION, AgentStage2Case
from mcp_server.agent.stage2 import build_final_assistant_text
from mcp_server.agent.tool_parser import parse_finish_text
from mcp_server.eval.exporters import trace_to_ms_swift_stage2_row
from mcp_server.eval.metrics import compute_trace_metrics
from mcp_server.eval.stage2_validator import validate_ms_swift_stage2_row, validate_stage2_trace
from mcp_server.teacher.agentic_stage2_trace_builder import AgenticStage2TraceBuilder


_FINISH_RE = re.compile(r"<finish>.*?</finish>", re.DOTALL)


class AgentFinishSchemaTest(unittest.TestCase):
    def test_final_assistant_builder_outputs_canonical_grid1000_finish(self) -> None:
        case = _sample_case()

        text = build_final_assistant_text(case, call=None, response=None, source="unit")
        finish, errors = parse_finish_text(text)

        self.assertEqual(errors, [])
        self.assertIsNotNone(finish)
        assert finish is not None
        self.assertEqual(finish["trajectory_grid1000"], [[500, 900], [501, 820], [520, 740]])
        self.assertEqual(finish["trajectory_coord_frame"], "norm1000")
        self.assertNotIn("trajectory", finish)
        self.assertTrue(finish_trajectory_matches(finish, case.ground_truth["pixel_trajectory"]))
        self.assertEqual(parse_finish_trajectory_norm(finish), case.ground_truth["pixel_trajectory"])

    def test_validators_accept_legacy_finish_trajectory_for_backward_compatibility(self) -> None:
        case = _sample_case()
        trace = AgenticStage2TraceBuilder().build_trace(
            case,
            call=None,
            tool_decision_reason="local FPV and BEV are enough",
        )
        finish, errors = parse_finish_text(trace.turns[-1].content)
        self.assertEqual(errors, [])
        assert finish is not None
        legacy_finish = dict(finish)
        legacy_finish.pop("trajectory_grid1000", None)
        legacy_finish.pop("trajectory_coord_frame", None)
        legacy_finish["trajectory"] = case.ground_truth["pixel_trajectory"]
        trace.turns[-1].content = _replace_finish(trace.turns[-1].content, legacy_finish)

        metric_result = compute_trace_metrics(trace, case)
        row = trace_to_ms_swift_stage2_row(trace, case)

        self.assertEqual(metric_result.metrics["finish_trajectory_consistency"], 1.0)
        self.assertTrue(validate_stage2_trace(trace, case).valid)
        self.assertTrue(validate_ms_swift_stage2_row(row).valid)

    def test_validators_reject_mismatched_canonical_finish_trajectory(self) -> None:
        case = _sample_case()
        trace = AgenticStage2TraceBuilder().build_trace(
            case,
            call=None,
            tool_decision_reason="local FPV and BEV are enough",
        )
        finish, errors = parse_finish_text(trace.turns[-1].content)
        self.assertEqual(errors, [])
        assert finish is not None
        bad_finish = dict(finish)
        bad_finish["trajectory_grid1000"] = [[0, 0] for _ in case.ground_truth["pixel_trajectory"]]
        trace.turns[-1].content = _replace_finish(trace.turns[-1].content, bad_finish)

        trace_validation = validate_stage2_trace(trace, case)
        row_validation = validate_ms_swift_stage2_row(trace_to_ms_swift_stage2_row(trace, case))

        self.assertFalse(trace_validation.valid)
        self.assertIn("finish_trajectory_mismatch", trace_validation.errors)
        self.assertFalse(row_validation.valid)
        self.assertIn("row_finish_trajectory_mismatch", row_validation.errors)


def _sample_case() -> AgentStage2Case:
    return AgentStage2Case(
        case_id="finish-schema",
        schema_version=AGENT_STAGE2_CASE_SCHEMA_VERSION,
        user_request="Continue along the visible sidewalk.",
        observation={"images": {"fpv": "/data/fpv.jpg", "bev": "/data/bev.jpg"}},
        ground_truth={
            "tool_policy": "no_tool",
            "pixel_trajectory": [[0.5, 0.9], [0.501, 0.82], [0.52, 0.74]],
            "intent": "go_straight",
        },
    )


def _replace_finish(text: str, payload: dict) -> str:
    replacement = f"<finish>{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}</finish>"
    return _FINISH_RE.sub(replacement, text, count=1)


if __name__ == "__main__":
    unittest.main()
