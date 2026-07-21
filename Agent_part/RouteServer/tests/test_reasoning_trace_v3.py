from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v3_text import parse_reasoning_target_v3, render_reasoning_target_v3
from mcp_server.contracts.reasoning_v3 import (
    LocalTraversabilityV3,
    NavigationContextGroundingV3,
    NavigationSourceGroundingV3,
    TraversabilityRegionV3,
)
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_composer import compose_reasoning_target_v3
from mcp_server.data.reasoning_v3_labels import (
    build_deterministic_supervision_v3,
    derive_corridor_from_fpv_trajectory_v3,
    derive_direction_from_trajectory_v3,
)
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from mcp_server.teacher.reasoning_v3_teacher import (
    parse_reasoning_v3_decision_rationale,
    parse_reasoning_v3_teacher_observation,
)

from reasoning_v3_test_utils import accepted_annotation, decision_rationale_payload, observation_payload


def test_reasoning_v3_round_trip_uses_exact_six_section_contract() -> None:
    """Round-trip text must preserve the V3.3 six-section order."""

    source = _source_with_bev()
    target = accepted_annotation(source, corridor="corridor.right", direction="right").target
    assert target is not None

    text = render_reasoning_target_v3(target, readout_tokens=["[SYS1_0]", "[SYS1_1]"])
    parsed = parse_reasoning_target_v3(text)

    assert parsed.valid, parsed.errors
    assert parsed.target == target
    tags = (
        "TaskGrounding",
        "SceneGrounding",
        "LocalTraversability",
        "NavigationContextGrounding",
        "InteractionAndRuleGrounding",
        "Decision",
    )
    assert [text.index(f"<{tag}>") for tag in tags] == sorted(text.index(f"<{tag}>") for tag in tags)
    assert text.index("</finish>") < text.index("<readout>")
    assert "EvidenceGrounding" not in text
    assert "ConstraintApplicability" not in text


def test_navigation_context_requires_real_source_and_resolved_target() -> None:
    """Positive navigation context cannot be source-free or corridor-free."""

    with pytest.raises(ValidationError):
        NavigationContextGroundingV3(
            applicable=False,
            source_groundings=[
                NavigationSourceGroundingV3(
                    source_type="bev",
                    source_ref="bev.current",
                    directive="right",
                    target_corridor_ref="corridor.right",
                    status="aligned",
                    reliability="high",
                    evidence_refs=["bev.current", "corridor.right"],
                )
            ],
            fused_direction="right",
            fused_target_corridor_ref="corridor.right",
            source_consistency="single_source",
            fused_execution_status="clear",
        )
    with pytest.raises(ValidationError, match="target corridor"):
        NavigationSourceGroundingV3(
            source_type="bev",
            source_ref="bev.current",
            directive="right",
            status="aligned",
            reliability="high",
            evidence_refs=["bev.current", "fpv.current"],
        )


def test_selected_corridor_must_be_direct_not_conditional_or_blocked() -> None:
    """Only directly traversable FPV support is immediately executable."""

    conditional = TraversabilityRegionV3(
        region_ref="corridor.center",
        surface="crosswalk",
        state="conditional",
        required_condition="wait_for_walk_signal",
        evidence_refs=["fpv.current", "corridor.center"],
    )
    with pytest.raises(ValidationError, match="direct region"):
        LocalTraversabilityV3(
            applicable=True,
            regions=[conditional],
            preferred_clear_corridor_ref="corridor.center",
            visibility="clear",
            local_risk="medium",
            confidence="high",
        )


def test_blocked_navigation_context_uses_direct_fallback_corridor() -> None:
    """Blocked route intent stays grounded while Decision executes a clear fallback."""

    source = build_reasoning_source_v3(
        {
            "id": "context-blocked",
            "dataset_source": "sidewalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Turn right at the next opening.",
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )
    payload = observation_payload(source, corridor="corridor.right", direction="right")
    payload["local_traversability"]["regions"] = [
        {
            "region_ref": "corridor.center",
            "surface": "sidewalk",
            "state": "direct",
            "required_condition": None,
            "blocker_refs": [],
            "reason": None,
            "severity": "unknown",
            "evidence_refs": ["fpv.current", "corridor.center"],
        },
        {
            "region_ref": "corridor.right",
            "surface": "sidewalk",
            "state": "blocked",
            "required_condition": None,
            "blocker_refs": [],
            "reason": "parked_vehicle_blocks_route",
            "severity": "full",
            "evidence_refs": ["fpv.current", "corridor.right"],
        },
    ]
    payload["local_traversability"]["preferred_clear_corridor_ref"] = "corridor.center"
    grounding = payload["navigation_context_grounding"]["source_groundings"][0]
    grounding.update(
        {
            "target_corridor_ref": "corridor.right",
            "status": "locally_blocked",
            "evidence_refs": ["instruction.active", "fpv.current", "corridor.right"],
        }
    )
    parsed = parse_reasoning_v3_teacher_observation(json.dumps(payload))
    assert parsed.observation is not None, parsed.errors

    rationale = parse_reasoning_v3_decision_rationale(
        json.dumps(decision_rationale_payload(source, parsed.observation))
    )
    assert rationale.rationale is not None, rationale.errors
    result = compose_reasoning_target_v3(source, parsed.observation, rationale.rationale)
    context = result.target.reasoning.navigation_context_grounding
    decision = result.target.reasoning.decision
    assert context.fused_direction == "right"
    assert context.fused_target_corridor_ref == "corridor.right"
    assert context.fused_execution_status == "locally_blocked"
    assert decision.selected_corridor_ref == "corridor.center"
    assert decision.motion == "go_straight"


def test_social_wait_preserves_grounded_future_corridor() -> None:
    """A temporary social stop must not erase the aligned route corridor."""

    source = build_reasoning_source_v3(
        {
            "id": "social-wait",
            "dataset_source": "sidewalk",
            "images": ["history.jpg", "current.jpg"],
            "image_roles": ["fpv.history_00", "fpv.current"],
            "instruction": "Continue straight.",
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )
    payload = observation_payload(source, corridor="corridor.center", direction="straight")
    payload["scene_grounding"]["entities"] = [
        {
            "entity_ref": "entity.person_01",
            "type": "pedestrian",
            "lateral_position": "center",
            "depth_position": "near",
            "motion_state": "crossing_path",
            "evidence_refs": ["fpv.history_00", "fpv.current"],
            "confidence": "high",
        }
    ]
    payload["interaction_and_rule_grounding"] = {
        "applicable": True,
        "interactions": [
            {
                "actor_ref": "entity.person_01",
                "relation": "crossing_path",
                "affected_corridor_ref": "corridor.center",
                "temporal_basis": "multi_frame",
                "required_behavior": "wait",
                "rule": "pedestrian_priority",
                "evidence_refs": ["fpv.history_00", "fpv.current", "corridor.center"],
                "confidence": "high",
            }
        ],
        "traffic_rules": [],
    }
    parsed = parse_reasoning_v3_teacher_observation(json.dumps(payload))
    assert parsed.observation is not None, parsed.errors

    rationale = parse_reasoning_v3_decision_rationale(
        json.dumps(decision_rationale_payload(source, parsed.observation))
    )
    assert rationale.rationale is not None, rationale.errors
    result = compose_reasoning_target_v3(source, parsed.observation, rationale.rationale)
    trace = result.target.reasoning
    assert trace.navigation_context_grounding.fused_execution_status == "clear"
    assert trace.local_traversability.preferred_clear_corridor_ref == "corridor.center"
    assert trace.decision.selected_corridor_ref == "corridor.center"
    assert trace.decision.motion == "stop"
    assert trace.decision.social_action == "wait"
    assert result.target.finish.motion == "stop"


def test_trusted_stop_without_visible_reason_disables_decision_reasoning_loss() -> None:
    """A trusted stop remains the outcome but cannot rewrite visible observations.

    This guards against hindsight hallucination: the first teacher sees a clear
    corridor, while the private demonstration says stop. The composer keeps both
    facts and masks Decision/finish reasoning instead of inventing a blocker.
    """

    source = build_reasoning_source_v3(
        {
            "id": "unsupported-demonstrated-stop",
            "dataset_source": "sidewalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Continue along the pedestrian route.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            "trajectory_policy": {"system1_target_format": "ego_pose_xyt_flu_v1"},
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )
    parsed = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source, corridor="corridor.center", direction="straight"))
    )
    assert parsed.observation is not None, parsed.errors
    rationale = parse_reasoning_v3_decision_rationale(
        json.dumps(decision_rationale_payload(source, parsed.observation))
    )
    assert rationale.rationale is not None, rationale.errors

    result = compose_reasoning_target_v3(source, parsed.observation, rationale.rationale)

    assert result.target.reasoning.local_traversability.preferred_clear_corridor_ref == "corridor.center"
    assert result.target.reasoning.decision.motion == "stop"
    assert result.target.reasoning.decision.rationale_status == "unsupported"
    assert result.section_loss_mask.decision is False


def test_trusted_trajectory_and_frodobots_trust_boundary() -> None:
    """Trajectory authority remains dataset-specific and private."""

    direction, details = derive_direction_from_trajectory_v3([[0.0, 0.0, 0.0], [2.0, 1.0, 0.2]])
    sidewalk = build_deterministic_supervision_v3(
        {
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, 1.0, 0.2]],
            "trajectory_policy": {"system1_target_format": "ego_pose_xyt_flu_v1"},
            "instruction": "Continue safe pedestrian navigation.",
        },
        dataset_source="sidewalk",
    )
    frodobots = build_deterministic_supervision_v3(
        {
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [9.0, -4.0, -0.5]],
            "bev_direction": "turn_left",
            "instruction": "Continue safe pedestrian navigation.",
        },
        dataset_source="frodobots",
    )

    assert direction == "left"
    assert details["endpoint_theta_deg"] > 10.0
    assert sidewalk.demonstrated_direction == "left"
    assert sidewalk.system1_enabled is True
    assert frodobots.demonstrated_direction == "unknown"
    assert frodobots.condition_direction == "left"
    assert frodobots.system1_enabled is False


def test_egowalk_motion_does_not_force_image_third_without_projection_qc() -> None:
    """Odometry direction and FPV corridor are independent without calibrated projection."""

    egowalk = build_deterministic_supervision_v3(
        {
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, -1.0, -0.2]],
            "trajectory_policy": {"system1_target_format": "ego_pose_xyt_flu_v1"},
            "caption": "Continue along the accessible path.",
        },
        dataset_source="egowalk",
    )
    corridor, details = derive_corridor_from_fpv_trajectory_v3(
        [[500.0, 900.0], [800.0, 700.0]],
        coord_frame="norm1000",
    )

    assert egowalk.demonstrated_direction == "right"
    assert egowalk.demonstrated_corridor_ref is None
    assert corridor == "corridor.right"
    assert details["endpoint_x_norm"] == pytest.approx(0.8)


def _source_with_bev():
    """Build one FPV+BEV source with a directional instruction."""

    return build_reasoning_source_v3(
        {
            "id": "foxglove-bev",
            "dataset_source": "foxglove",
            "images": ["history.jpg", "current.jpg", "bev.jpg"],
            "image_roles": ["fpv.history_00", "fpv.current", "bev.current"],
            "instruction": "Turn right and continue on the pedestrian route.",
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="foxglove"),
    )
