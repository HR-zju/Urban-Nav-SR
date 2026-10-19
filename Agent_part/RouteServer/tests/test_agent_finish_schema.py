from __future__ import annotations

import json
import math
import unittest

from mcp_server.agent.finish_schema import parse_finish_trajectory_norm
from mcp_server.agent.schemas import AGENT_STAGE2_CASE_SCHEMA_VERSION, AgentStage2Case
from mcp_server.agent.stage2 import (
    READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
    READOUT_PLACEMENT_USER_PROMPT_V1,
    build_final_assistant_text,
)
from mcp_server.agent.tool_parser import parse_finish_text
from mcp_server.eval.agent_tools.exporters import trace_to_ms_swift_stage2_row
from mcp_server.eval.agent_tools.metrics import compute_trace_metrics
from mcp_server.eval.agent_tools.stage2_validator import validate_ms_swift_stage2_row, validate_stage2_trace
from mcp_server.teacher.agent_tools.agentic_stage2_trace_builder import AgenticStage2TraceBuilder, AgenticStage2TraceBuilderConfig


class AgentFinishSchemaTest(unittest.TestCase):
    def test_final_assistant_builder_outputs_v2_symbolic_finish(self) -> None:
        case = _sample_case()

        text = build_final_assistant_text(case, call=None, response=None, source="unit")
        finish, errors = parse_finish_text(text)

        self.assertEqual(errors, [])
        self.assertIsNotNone(finish)
        assert finish is not None
        self.assertEqual(finish["schema_version"], "urbannav_finish_v2")
        self.assertEqual(finish["decision_type"], "navigate")
        self.assertNotIn("trajectory", finish)
        self.assertNotIn("trajectory_grid1000", finish)
        self.assertIsNone(parse_finish_trajectory_norm(finish))
        self.assertIn("<TaskGrounding>", text)
        self.assertIn("<readout>[SYS1_0]", text)

    def test_validators_accept_legacy_finish_trajectory_for_backward_compatibility(self) -> None:
        case = _sample_case()
        trace = AgenticStage2TraceBuilder(
            AgenticStage2TraceBuilderConfig(readout_placement=READOUT_PLACEMENT_USER_PROMPT_V1)
        ).build_trace(
            case,
            call=None,
            tool_decision_reason="local FPV and BEV are enough",
        )
        trace.turns[-1].content = _legacy_final_text(case.ground_truth["pixel_trajectory"])

        metric_result = compute_trace_metrics(trace, case)
        row = trace_to_ms_swift_stage2_row(
            trace,
            case,
            readout_placement=READOUT_PLACEMENT_USER_PROMPT_V1,
        )

        self.assertEqual(metric_result.metrics["finish_trajectory_consistency"], 1.0)
        self.assertTrue(validate_stage2_trace(trace, case).valid)
        self.assertTrue(validate_ms_swift_stage2_row(row).valid)

    def test_legacy_validators_reject_mismatched_finish_trajectory(self) -> None:
        case = _sample_case()
        trace = AgenticStage2TraceBuilder(
            AgenticStage2TraceBuilderConfig(readout_placement=READOUT_PLACEMENT_USER_PROMPT_V1)
        ).build_trace(
            case,
            call=None,
            tool_decision_reason="local FPV and BEV are enough",
        )
        trace.turns[-1].content = _legacy_final_text(
            [[0.0, 0.0] for _ in case.ground_truth["pixel_trajectory"]]
        )

        trace_validation = validate_stage2_trace(trace, case)
        row_validation = validate_ms_swift_stage2_row(
            trace_to_ms_swift_stage2_row(
                trace,
                case,
                readout_placement=READOUT_PLACEMENT_USER_PROMPT_V1,
            )
        )

        self.assertFalse(trace_validation.valid)
        self.assertIn("finish_trajectory_mismatch", trace_validation.errors)
        self.assertFalse(row_validation.valid)
        self.assertIn("row_finish_trajectory_mismatch", row_validation.errors)

    def test_exporter_supports_assistant_suffix_system1_readout_v2(self) -> None:
        case = _sample_case()
        trace = AgenticStage2TraceBuilder().build_trace(
            case,
            call=None,
            tool_decision_reason="local FPV and BEV are enough",
        )

        row = trace_to_ms_swift_stage2_row(
            trace,
            case,
            readout_placement=READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
        )

        self.assertTrue(validate_ms_swift_stage2_row(row).valid)
        self.assertEqual(row["contract_meta"]["system1_readout_placement"], READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2)
        self.assertEqual(row["contract_meta"]["system1_readout_lm_label"], -100)
        self.assertIn("[SYS1_0]", row["messages"][-1]["content"])
        self.assertGreater(row["messages"][-1]["content"].find("<readout>"), row["messages"][-1]["content"].find("</finish>"))
        self.assertNotIn("[TRAJ_0]", row["messages"][-2]["content"])
        finish, errors = parse_finish_text(row["messages"][-1]["content"])
        self.assertEqual(errors, [])
        assert finish is not None
        self.assertEqual(finish["decision_type"], "navigate")
        self.assertNotIn("trajectory_grid1000", finish)
        self.assertNotIn("trajectory", finish)

    def test_v2_exporter_canonicalizes_system1_trajectory_ego_t(self) -> None:
        case = _sample_case()
        case.ground_truth["trajectory_ego_t"] = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.5],
            [1.0, 1.0, 0.2, 0.5],
        ]
        trace = AgenticStage2TraceBuilder().build_trace(
            case,
            call=None,
            tool_decision_reason="local FPV and BEV are enough",
        )

        row = trace_to_ms_swift_stage2_row(
            trace,
            case,
            readout_placement=READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
        )

        self.assertTrue(validate_ms_swift_stage2_row(row).valid)
        self.assertEqual(row["contract_meta"]["system1_target_field"], "trajectory_ego_t")
        self.assertEqual(row["contract_meta"]["system1_target_format"], "ego_pose_xyt_flu_v1")
        self.assertEqual(row["trajectory_ego_t"][0], [0.0, 0.0, 0.0])
        self.assertEqual(row["trajectory_ego_t"][1], [1.0, 0.0, 0.0])
        self.assertEqual(row["trajectory_ego_t"][2][:2], [1.0, -1.0])
        self.assertAlmostEqual(row["trajectory_ego_t"][2][2], -math.pi / 2.0)
        self.assertEqual(row["trajectory_policy"]["coordinate_frame"], "current_ego_flu")
        self.assertEqual(row["trajectory_policy"]["representation"], "cumulative_relative_pose")
        self.assertEqual(
            row["trajectory_policy"]["axis_names"],
            ["x_forward_m", "y_left_m", "yaw_rel_ccw_rad"],
        )
        self.assertEqual(row["trajectory_policy"]["yaw_source"], "path_tangent")
        self.assertIs(row["trajectory_policy"]["normalized"], False)
        self.assertEqual(row["system1_supervision"]["trajectory_ego_t"], row["trajectory_ego_t"])

    def test_v2_exporter_preserves_declared_canonical_flu_trajectory(self) -> None:
        case = _sample_case()
        case.ground_truth.update(
            {
                "trajectory_ego_t": [[0.5, 0.1, 0.05], [1.0, 0.3, 0.2]],
                "trajectory_policy": {
                    "system1_target_format": "ego_pose_xyt_flu_v1",
                    "yaw_source": "odometry",
                    "dt_sec": 0.2,
                },
            }
        )
        trace = AgenticStage2TraceBuilder().build_trace(
            case,
            call=None,
            tool_decision_reason="local FPV and BEV are enough",
        )

        row = trace_to_ms_swift_stage2_row(
            trace,
            case,
            readout_placement=READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
        )

        self.assertTrue(validate_ms_swift_stage2_row(row).valid)
        for actual, expected in zip(row["trajectory_ego_t"], [[0.5, 0.1, 0.05], [1.0, 0.3, 0.2]]):
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value)
        self.assertEqual(row["trajectory_policy"]["yaw_source"], "odometry")
        self.assertEqual(row["trajectory_policy"]["dt_sec"], 0.2)

    def test_v2_validator_rejects_legacy_four_channel_system1_trajectory(self) -> None:
        case = _sample_case()
        trace = AgenticStage2TraceBuilder().build_trace(
            case,
            call=None,
            tool_decision_reason="local FPV and BEV are enough",
        )
        row = trace_to_ms_swift_stage2_row(
            trace,
            case,
            readout_placement=READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
        )
        row["trajectory_ego_t"] = [[0.0, 0.0, 0.0, 0.0]]
        row["system1_supervision"]["trajectory_ego_t"] = row["trajectory_ego_t"]

        validation = validate_ms_swift_stage2_row(row)

        self.assertFalse(validation.valid)
        self.assertIn("row_system1_trajectory_ego_t_width:4", validation.errors)
        self.assertIn("row_system1_supervision_trajectory_ego_t_not_canonical", validation.errors)


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


def _legacy_final_text(trajectory: list[list[float]]) -> str:
    """Build one historical V1 target for read-compatibility tests."""

    finish = {
        "trajectory": trajectory,
        "intent": "go_straight",
        "confidence_level": "normal",
        "trajectory_type": "forward",
        "route_action": "continue_navigation",
    }
    return (
        "<think>\n"
        '<Scene>{"summary":"sidewalk"}</Scene>\n'
        '<FPVScene>{"navigable_area":"center"}</FPVScene>\n'
        '<BEVContext>{"available":true,"intent":"go_straight"}</BEVContext>\n'
        '<CrossView>{"alignment":"aligned"}</CrossView>\n'
        '<Plan>{"motion_primitive":"go_straight"}</Plan>\n'
        "</think>\n"
        f"<finish>{json.dumps(finish, separators=(',', ':'))}</finish>"
    )


if __name__ == "__main__":
    unittest.main()
