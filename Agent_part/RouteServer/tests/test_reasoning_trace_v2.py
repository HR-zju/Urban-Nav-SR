from __future__ import annotations

import json
import importlib.util
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

from mcp_server.agent.reasoning_text import adapt_v1_reasoning_text, parse_reasoning_target_v2, render_reasoning_target_v2
from mcp_server.agent.reasoning_validation import validate_reasoning_target_v2
from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse
from mcp_server.agent.stage2 import READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2, stage2_evidence_catalog
from mcp_server.contracts.reasoning import (
    ApplicableConstraintV2,
    ConstraintApplicabilityV2,
    CrossModalAlignmentV2,
    DecisionV2,
    EvidenceGroundingV2,
    FinishV2,
    GroundedFactV2,
    ReasoningTargetV2,
    ReasoningTraceV2,
    TaskGroundingV2,
)
from mcp_server.eval.exporters import trace_to_ms_swift_stage2_row
from mcp_server.eval.stage2_validator import validate_ms_swift_stage2_row, validate_stage2_trace
from mcp_server.teacher.agentic_stage2_trace_builder import AgenticStage2TraceBuilder


class ReasoningTraceV2ContractTest(unittest.TestCase):
    def test_full_target_round_trip_and_grounding_validation(self) -> None:
        target = _valid_target()
        text = render_reasoning_target_v2(
            target,
            readout_tokens=[f"[SYS1_{index}]" for index in range(16)],
        )

        parsed = parse_reasoning_target_v2(text)
        validation = validate_reasoning_target_v2(
            parsed.target,
            evidence_catalog=_evidence_catalog(),
            travel_mode="foot",
            assistant_text=text,
            require_navigation_readout=True,
        )

        self.assertTrue(parsed.valid, parsed.errors)
        self.assertTrue(validation.valid, validation.errors)
        self.assertNotIn("trajectory_grid1000", text)
        self.assertLess(text.index("<TaskGrounding>"), text.index("<Decision>"))
        self.assertLess(text.index("</finish>"), text.index("<readout>"))

    def test_finish_must_match_decision(self) -> None:
        trace = _valid_target().reasoning
        with self.assertRaises(ValidationError):
            ReasoningTargetV2(
                reasoning=trace,
                finish=FinishV2(
                    decision_type="navigate",
                    motion="turn_right",
                    speed="slow",
                    social_action="yield",
                    confidence_level="high",
                    trajectory_type="local_navigation",
                ),
            )

    def test_alignment_availability_must_match_bev_modality(self) -> None:
        base = _valid_target()
        alignment = base.reasoning.cross_modal_alignment.model_copy(
            update={"evidence_refs": ["route.active", "bev.current", "corridor.left"]}
        )
        target = ReasoningTargetV2(
            reasoning=base.reasoning.model_copy(update={"cross_modal_alignment": alignment}),
            finish=base.finish,
        )

        fpv_only = validate_reasoning_target_v2(
            target,
            evidence_catalog=_evidence_catalog(),
            available_modalities={"fpv", "temporal_fpv"},
        )
        fpv_bev = validate_reasoning_target_v2(
            target,
            evidence_catalog=[*_evidence_catalog(), "bev.current"],
            available_modalities={"fpv", "temporal_fpv", "bev"},
        )

        self.assertFalse(fpv_only.valid)
        self.assertIn("alignment_available_without_bev", fpv_only.errors)
        self.assertIn("alignment_content_without_bev", fpv_only.errors)
        self.assertTrue(fpv_bev.valid, fpv_bev.errors)

    def test_social_temporal_constraint_requires_track_and_path_refs(self) -> None:
        target = _valid_target()
        payload = target.reasoning.model_dump(mode="json")
        payload["constraint_applicability"] = {
            "applicable": [
                {
                    "constraint": "yield_to_crossing_agent",
                    "trigger_refs": ["fpv.current", "corridor.left"],
                    "effect": "wait",
                }
            ],
            "not_applicable": [],
        }
        payload["decision"]["social_action"] = "yield"
        modified_trace = ReasoningTraceV2.model_validate(payload)
        modified_finish = target.finish.model_copy(update={"social_action": "yield"})
        modified = ReasoningTargetV2(reasoning=modified_trace, finish=modified_finish)

        result = validate_reasoning_target_v2(modified, evidence_catalog=_evidence_catalog())

        self.assertFalse(result.valid)
        self.assertIn("constraint_missing_track_ref:yield_to_crossing_agent", result.errors)

    def test_legacy_adapter_drops_text_trajectory_and_weak_social_rule(self) -> None:
        legacy = _legacy_text()

        parsed = adapt_v1_reasoning_text(legacy)

        self.assertTrue(parsed.valid, parsed.errors)
        self.assertTrue(parsed.adapted_from_v1)
        self.assertEqual(parsed.target.finish.schema_version, "urbannav_finish_v2")
        self.assertEqual(parsed.target.reasoning.constraint_applicability.applicable, [])
        rendered = render_reasoning_target_v2(parsed.target)
        self.assertNotIn("trajectory_grid1000", rendered)
        self.assertNotIn("<FPVScene>", rendered)

    def test_tool_result_use_has_one_causal_position_and_matching_finish(self) -> None:
        base = _valid_target()
        evidence = base.reasoning.evidence_grounding.model_copy(
            update={"tool_evidence_refs": ["tool.map_tool.search_poi"]}
        )
        target = ReasoningTargetV2(
            reasoning=base.reasoning.model_copy(update={"evidence_grounding": evidence}),
            finish=base.finish.model_copy(update={"tool_used": True}),
        )
        tool_use = {"tool_used": True, "tool": "map_tool", "op": "search_poi"}
        text = render_reasoning_target_v2(
            target,
            readout_tokens=[f"[SYS1_{index}]" for index in range(16)],
            tool_result_use=tool_use,
        )
        parsed = parse_reasoning_target_v2(text)
        validation = validate_reasoning_target_v2(
            parsed.target,
            evidence_catalog=[*_evidence_catalog(), "tool.map_tool.search_poi"],
            assistant_text=text,
            require_navigation_readout=True,
            tool_result_use=parsed.tool_result_use,
        )

        self.assertTrue(parsed.valid, parsed.errors)
        self.assertTrue(validation.valid, validation.errors)
        self.assertLess(text.index("</EvidenceGrounding>"), text.index("<ToolResultUse>"))
        self.assertLess(text.index("</ToolResultUse>"), text.index("<CrossModalAlignment>"))

        tool_block = '<ToolResultUse>{"tool_used":true,"tool":"map_tool","op":"search_poi"}</ToolResultUse>\n'
        misplaced = text.replace(tool_block, "").replace("<TaskGrounding>", tool_block + "<TaskGrounding>", 1)
        self.assertIn("tool_result_use_position", " ".join(parse_reasoning_target_v2(misplaced).errors))

    def test_ms_swift_consumer_schema_matches_routeserver_contract(self) -> None:
        code_root = Path(__file__).resolve().parents[4]
        consumer_path = code_root / "ms-swift/swift/sidewalk_nav/reasoning.py"
        if not consumer_path.is_file():
            self.skipTest("sibling ms-swift checkout is not available")
        spec = importlib.util.spec_from_file_location("ms_swift_reasoning_contract", consumer_path)
        assert spec is not None and spec.loader is not None
        consumer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = consumer
        spec.loader.exec_module(consumer)

        self.assertEqual(
            _semantic_schema(ReasoningTargetV2.model_json_schema()),
            _semantic_schema(consumer.ReasoningTargetV2.model_json_schema()),
        )

    def test_multiframe_images_do_not_implicitly_create_bev_evidence(self) -> None:
        fpv_only = AgentStage2Case(
            case_id="multiframe-fpv",
            user_request="Continue.",
            observation={"images": ["old.jpg", "current.jpg"]},
        )
        fpv_bev = AgentStage2Case(
            case_id="multiframe-bev",
            user_request="Continue.",
            observation={
                "images": ["old.jpg", "current.jpg", "bev.jpg"],
                "image_roles": ["fpv_history", "fpv", "bev"],
            },
        )

        self.assertEqual(fpv_only.image_roles, ["fpv", "fpv"])
        self.assertNotIn("bev.current", stage2_evidence_catalog(fpv_only))
        self.assertIn("bev.current", stage2_evidence_catalog(fpv_bev))

class Stage2ReasoningV2IntegrationTest(unittest.TestCase):
    def test_tool_and_no_tool_rows_are_v2_and_valid(self) -> None:
        builder = AgenticStage2TraceBuilder()
        no_tool_case = _stage2_case("no-tool")
        tool_case = _stage2_case("tool")
        call = ToolCall(
            tool="map_tool",
            arguments={"op": "search_poi", "query": "bench", "location": "120.0,30.0"},
            call_id="map-call-1",
        )

        for case, trace in (
            (no_tool_case, builder.build_trace(no_tool_case, call=None)),
            (tool_case, builder.build_trace(tool_case, call=call)),
        ):
            row = trace_to_ms_swift_stage2_row(
                trace,
                case=case,
                readout_placement=READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
            )
            trace_result = validate_stage2_trace(trace, case=case)
            row_result = validate_ms_swift_stage2_row(row)
            final_text = row["messages"][-1]["content"]

            self.assertTrue(trace_result.valid, trace_result.errors)
            self.assertTrue(row_result.valid, row_result.errors)
            self.assertIn("<TaskGrounding>", final_text)
            self.assertIn("<readout>[SYS1_0]", final_text)
            self.assertEqual(row["contract_meta"]["reasoning_schema_version"], "urbannav_reasoning_trace_v2")

    def test_user_interaction_row_has_no_system1_supervision(self) -> None:
        case = _stage2_case("ask")
        call = ToolCall(
            tool="user_interaction_tool",
            arguments={"op": "request_clarification", "question": "Which entrance should I use?"},
            call_id="ask-1",
        )
        response = ToolResponse(
            tool="user_interaction_tool",
            success=True,
            call_id="ask-1",
            result={
                "op": "request_clarification",
                "await_user": True,
                "requires_user_response": True,
                "interaction_mode": "blocking",
                "question": "Which entrance should I use?",
                "motion_policy_while_waiting": "stop",
                "dialogue_state": {"pending_user_interaction": {"interaction_id": "ask-1"}},
            },
        )
        trace = AgenticStage2TraceBuilder().build_trace(case, call=call, response=response)
        row = trace_to_ms_swift_stage2_row(trace, case=case)
        final_text = row["messages"][-1]["content"]

        self.assertTrue(validate_stage2_trace(trace, case=case).valid)
        self.assertTrue(validate_ms_swift_stage2_row(row).valid)
        self.assertIn('"decision_type":"ask_user"', final_text)
        self.assertNotIn("[SYS1_", final_text)
        self.assertNotIn("trajectory_ego_t", row)
        self.assertEqual(row["system1_supervision"], {"enabled": False, "target_type": "none"})


def _valid_target() -> ReasoningTargetV2:
    trace = ReasoningTraceV2(
        task_grounding=TaskGroundingV2(
            goal_ref="goal.active",
            route_ref="route.active",
            local_objective="continue_route",
            instruction_constraints=["use_crosswalk"],
        ),
        evidence_grounding=EvidenceGroundingV2(
            task_evidence_refs=["instruction.span_1"],
            route_evidence_refs=["route.active", "waypoint.wp_13"],
            fpv_evidence_refs=["fpv.current", "track.person_4", "corridor.left"],
            relevant_facts=[
                GroundedFactV2(fact_code="route_turns_left", evidence_refs=["waypoint.wp_13"]),
                GroundedFactV2(fact_code="crossing_agent_on_path", evidence_refs=["track.person_4", "corridor.left"]),
            ],
        ),
        cross_modal_alignment=CrossModalAlignmentV2(
            available=True,
            desired_direction="left",
            route_to_fpv="left",
            selected_corridor_ref="corridor.left",
            status="locally_blocked",
            evidence_refs=["route.active", "corridor.left", "track.person_4"],
        ),
        constraint_applicability=ConstraintApplicabilityV2(
            applicable=[
                ApplicableConstraintV2(
                    constraint="yield_to_crossing_agent",
                    trigger_refs=["track.person_4", "corridor.left"],
                    effect="wait",
                )
            ],
        ),
        decision=DecisionV2(
            motion="turn_left",
            speed="slow",
            social_action="yield",
            selected_corridor_ref="corridor.left",
            fallback="stop",
            confidence="high",
            reason_codes=["crossing_agent_on_path"],
        ),
    )
    return ReasoningTargetV2(
        reasoning=trace,
        finish=FinishV2(
            decision_type="navigate",
            motion="turn_left",
            speed="slow",
            social_action="yield",
            confidence_level="high",
            trajectory_type="local_navigation",
            route_action="continue_navigation",
            tool_used=False,
        ),
    )


def _evidence_catalog() -> list[str]:
    return [
        "instruction.span_1",
        "goal.active",
        "route.active",
        "waypoint.wp_13",
        "fpv.current",
        "track.person_4",
        "corridor.left",
    ]


def _legacy_text() -> str:
    finish = {
        "trajectory_grid1000": [[500, 900], [500, 800]],
        "trajectory_coord_frame": "norm1000",
        "intent": "go_straight",
        "confidence_level": "normal",
        "trajectory_type": "forward",
        "route_action": "continue_navigation",
    }
    return (
        "<think>\n"
        '<Scene>{"summary":"sidewalk"}</Scene>\n'
        '<FPVScene>{"blocked_by":["pedestrian"],"local_risk":"medium"}</FPVScene>\n'
        '<BEVContext>{"available":true,"intent":"go_straight"}</BEVContext>\n'
        '<CrossView>{"safe_corridor":"center_corridor","alignment":"aligned","confidence":"high"}</CrossView>\n'
        '<Plan>{"motion_primitive":"go_straight","speed_policy":"slow","social_action":"yield","contingency":"stop"}</Plan>\n'
        "</think>\n"
        f"<finish>{json.dumps(finish, separators=(',', ':'))}</finish>"
    )


def _stage2_case(suffix: str) -> AgentStage2Case:
    return AgentStage2Case(
        case_id=f"reasoning-v2-{suffix}",
        user_request="Continue along the sidewalk.",
        observation={
            "images": {"fpv": f"/{suffix}-fpv.jpg", "bev": f"/{suffix}-bev.jpg"},
            "current_location": "120.0,30.0",
            "route_session_id": "route-1",
            "mode": "foot",
        },
        ground_truth={
            "intent": "go_straight",
            "pixel_trajectory": [[0.5, 0.9], [0.5, 0.8]],
        },
    )


def _semantic_schema(value):
    """Drop documentation-only keys before cross-repository schema comparison."""

    if isinstance(value, dict):
        return {
            key: _semantic_schema(item)
            for key, item in value.items()
            if key not in {"title", "description"}
        }
    if isinstance(value, list):
        return [_semantic_schema(item) for item in value]
    return value


if __name__ == "__main__":
    unittest.main()
