from __future__ import annotations

from mcp_server.agent.reasoning_v4_text import render_reasoning_section_v4
from mcp_server.contracts.direction_profile import DirectionLabelStamp
from mcp_server.contracts.openloop_benchmark import (
    OpenLoopPredictionV1,
    OpenLoopReferenceV1,
)
from mcp_server.contracts.reasoning_v4 import DecisionV4
from mcp_server.eval.openloop_common import join_openloop_artifacts_v1
from mcp_server.eval.stage_i_openloop import (
    evaluate_stage_i_decision_only_v1,
)


def _stamp(label_index: int, label_name: str, choice: str, angle: float) -> DirectionLabelStamp:
    return DirectionLabelStamp(
        profile_id="fpv_degree10_terminal_centroid_v2",
        profile_version="2.0",
        label_index=label_index,
        label_name=label_name,
        choice=choice,
        angle_deg=angle,
        source="pixel_trajectory_terminal_centroid_rule_v2",
        bin_mode="degree10_with_stop",
        derivation_version="pixel_trajectory_terminal_centroid_rule_v2",
    )


def _reference(
    sample_id: str,
    stamp: DirectionLabelStamp,
    *,
    task_type: str = "decision",
) -> OpenLoopReferenceV1:
    decision = DecisionV4(
        selected_action_label=stamp.label_name,
        speed="normal",
        response="proceed",
    )
    return OpenLoopReferenceV1(
        sample_id=sample_id,
        parent_sample_id=f"parent:{sample_id}",
        source_group_id=f"source:{sample_id}",
        split_group_id=f"clip:{sample_id}",
        stage="stage_i",
        task_type=task_type,
        dataset_source="foxglove",
        target={"decision": decision.model_dump(mode="json")},
        label_provenance="deterministic_gt",
        direction_stamp=stamp,
    )


def _prediction(reference: OpenLoopReferenceV1, decision: DecisionV4) -> OpenLoopPredictionV1:
    return OpenLoopPredictionV1(
        sample_id=reference.sample_id,
        parent_sample_id=reference.parent_sample_id,
        stage="stage_i",
        task_type=reference.task_type,
        response=render_reasoning_section_v4("Decision", decision),
        input_digest="a" * 64,
    )


def test_decision_only_scorecard_excludes_training_view_metrics() -> None:
    straight = _reference("straight", _stamp(9, "straight", "J", 0.0))
    left_50 = _reference("left-50", _stamp(4, "left_50", "E", -50.0))
    references = {row.sample_id: row for row in (straight, left_50)}
    predictions = {
        "straight": _prediction(
            straight,
            DecisionV4(
                selected_action_label="right_10",
                speed="normal",
                response="proceed",
            ),
        ),
        "left-50": _prediction(
            left_50,
            DecisionV4(
                selected_action_label="left_60",
                speed="normal",
                response="proceed",
            ),
        ),
    }

    report = evaluate_stage_i_decision_only_v1(
        join_openloop_artifacts_v1(references, predictions)
    )
    profile = report["decision"]["profiles"][
        "fpv_degree10_terminal_centroid_v2"
    ]

    assert report["evaluation_scope"] == "stage_i_direction_only"
    assert report["direct_task_type"] == "decision"
    assert "full_cot" not in report
    assert "vqa" not in report
    assert profile["strict_direction_accuracy"] == 0.0
    assert profile["within_10_accuracy"] == 1.0
    assert profile["within_20_accuracy"] == 1.0
    assert profile["common_12way"]["strict_direction_accuracy"] == 0.5


def test_direction_only_scorecard_accepts_v2_reasoning_dropout() -> None:
    reference = _reference(
        "dropout-straight",
        _stamp(9, "straight", "J", 0.0),
        task_type="reasoning_dropout",
    )
    report = evaluate_stage_i_decision_only_v1(
        join_openloop_artifacts_v1(
            {reference.sample_id: reference},
            {
                reference.sample_id: _prediction(
                    reference,
                    DecisionV4(
                        selected_action_label="straight",
                        speed="normal",
                        response="proceed",
                    ),
                )
            },
        )
    )

    assert report["direct_task_type"] == "reasoning_dropout"
    assert report["decision"]["profiles"]["fpv_degree10_terminal_centroid_v2"]["strict_direction_accuracy"] == 1.0
