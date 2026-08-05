from __future__ import annotations

import json

import pytest

from mcp_server.agent.reasoning_v4_text import (
    render_reasoning_section_v4,
    render_reasoning_target_v4,
)
from mcp_server.agent.vqa_v2_text import render_vqa_answer_v2
from mcp_server.contracts.direction_profile import DirectionLabelStamp
from mcp_server.contracts.openloop_benchmark import (
    CheckpointExposureEntryV1,
    CheckpointExposureRegistryV1,
    OpenLoopCheckpointIdentityV1,
    OpenLoopContextIdentityV1,
    OpenLoopFailureV1,
    OpenLoopInputV1,
    OpenLoopMessageV1,
    OpenLoopMetricPolicyV1,
    OpenLoopPredictionV1,
    OpenLoopReferenceV1,
    OpenLoopRendererPolicyV1,
    OpenLoopRepositoryStateV1,
    OpenLoopRunManifestV1,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    ReasoningTargetV4,
    ReasoningTraceV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGroundingV4,
)
from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION,
    StageIITrajectoryPolicyV1,
    StageIITrajectoryTargetV1,
    stage_ii_digest_v1,
)
from mcp_server.contracts.vqa_v2 import VQAQuestionV2, build_vqa_answer_v2
from mcp_server.eval.openloop_common import (
    audit_checkpoint_exposure_v1,
    audit_output_input_digests_v1,
    canonical_row_digest_v1,
    coverage_metrics_v1,
    join_openloop_artifacts_v1,
)
from mcp_server.eval.openloop_registry import (
    build_checkpoint_exposure_registry_v1,
)
from mcp_server.eval.stage_i_openloop import (
    evaluate_decision_openloop_v1,
    evaluate_full_cot_openloop_v1,
    evaluate_vqa_openloop_v1,
)
from mcp_server.eval.stage_ii_cfm_openloop import (
    evaluate_stage_ii_cfm_openloop_v1,
)


_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64


def _stamp(label_name: str) -> DirectionLabelStamp:
    labels = (
        "stop",
        "straight",
        "left_10",
        "right_10",
        "left_20",
        "right_20",
        "left_30",
        "right_30",
        "left_40",
        "right_40",
        "left_50plus",
        "right_50plus",
    )
    index = labels.index(label_name)
    angles = (
        None,
        0.0,
        -10.0,
        10.0,
        -20.0,
        20.0,
        -30.0,
        30.0,
        -40.0,
        40.0,
        -50.0,
        50.0,
    )
    return DirectionLabelStamp(
        profile_id="ego_degree10_12way_terminal_centroid_v2",
        profile_version="2.0",
        label_index=index,
        label_name=label_name,
        choice=chr(ord("A") + index),
        angle_deg=angles[index],
        source=(
            "finish_stop_rule" if label_name == "stop" else "trajectory_ego_t"
        ),
        bin_mode="degree10_12way_with_stop",
        derivation_version="ego_x_right_y_front_terminal_centroid_rule_v2",
    )


def _reasoning_target() -> ReasoningTargetV4:
    return ReasoningTargetV4(
        reasoning=ReasoningTraceV4(
            task_grounding=TaskGroundingV4(
                objective="follow_sidewalk",
                requirements=["avoid_obstacles", "stay_on_walkable_surface"],
                ambiguities=[],
            ),
            scene_grounding=SceneGroundingV4(
                scene_type="urban_sidewalk",
                walkable_support="sidewalk",
                key_entities=["pedestrian"],
                traffic_context="none",
            ),
            local_traversability=LocalTraversabilityV4(
                direction_affordances=[]
            ),
            navigation_context_grounding=NavigationContextGroundingV4(
                task_direction=TaskDirectionV4(
                    status="insufficient_evidence",
                    source="none",
                )
            ),
            interaction_constraint_grounding=(
                InteractionConstraintGroundingV4(constraints=[])
            ),
            decision=DecisionV4(
                selected_action_label="straight",
                speed="normal",
                response="proceed",
            ),
        )
    )


def _reference(
    sample_id: str,
    *,
    task_type: str,
    target: dict,
    provenance: str = "reference_annotation",
    stamp: DirectionLabelStamp | None = None,
    dataset: str = "sidewalk",
    mask: list[bool] | None = None,
) -> OpenLoopReferenceV1:
    return OpenLoopReferenceV1(
        sample_id=sample_id,
        parent_sample_id=f"parent:{sample_id}",
        source_group_id=f"source:{sample_id}",
        split_group_id=f"clip:{sample_id}",
        stage="stage_ii" if task_type == "trajectory" else "stage_i",
        task_type=task_type,
        dataset_source=dataset,
        target=target,
        label_provenance=provenance,
        direction_stamp=stamp,
        trajectory_valid_mask=mask,
        trajectory_policy_id=(
            "urbannav_stage_ii_metric_flu_2s_16_v1"
            if task_type == "trajectory"
            else None
        ),
    )


def _prediction(
    reference: OpenLoopReferenceV1,
    *,
    response: str | None = None,
    trajectory: list[list[float]] | None = None,
    context_identity: dict | None = None,
    metadata: dict | None = None,
) -> OpenLoopPredictionV1:
    return OpenLoopPredictionV1(
        sample_id=reference.sample_id,
        parent_sample_id=reference.parent_sample_id,
        stage=reference.stage,
        task_type=reference.task_type,
        response=response,
        trajectory_ego_t=trajectory,
        metadata=metadata or {},
        input_digest=_HEX_B,
        context_identity=context_identity or {},
    )


def test_model_visible_input_rejects_private_supervision() -> None:
    with pytest.raises(ValueError, match="private supervision"):
        OpenLoopInputV1(
            sample_id="sample-1",
            parent_sample_id="parent-1",
            source_group_id="source-1",
            split_group_id="clip-1",
            split="validation",
            stage="stage_i",
            task_type="decision",
            dataset_source="sidewalk",
            messages=[OpenLoopMessageV1(role="user", content="<image>go")],
            images=["current.jpg"],
            image_roles=["fpv.current"],
            public_context={"direction_supervision": {"label": "straight"}},
            modality_profile_id="fpv_current_v1",
            direction_profile_id="ego_degree10_12way_terminal_centroid_v2",
            renderer_policy_digest=_HEX_A,
        )
    with pytest.raises(ValueError, match="assistant turns"):
        OpenLoopInputV1(
            sample_id="sample-2",
            parent_sample_id="parent-2",
            source_group_id="source-2",
            split_group_id="clip-2",
            split="validation",
            stage="stage_i",
            task_type="decision",
            dataset_source="sidewalk",
            messages=[OpenLoopMessageV1(role="assistant", content="target")],
            modality_profile_id="fpv_current_v1",
            renderer_policy_digest=_HEX_A,
        )


def test_openloop_v21_accepts_reasoning_dropout_and_reads_v20_rows() -> None:
    """V2 direct rendering adds one task without invalidating frozen V2.0 rows."""

    common = {
        "sample_id": "dropout-1",
        "parent_sample_id": "parent-1",
        "source_group_id": "source-1",
        "split_group_id": "clip-1",
        "split": "validation",
        "stage": "stage_i",
        "task_type": "reasoning_dropout",
        "dataset_source": "sidewalk",
        "messages": [OpenLoopMessageV1(role="user", content="<image>go")],
        "images": ["current.jpg"],
        "image_roles": ["fpv.current"],
        "modality_profile_id": "fpv_current_v1",
        "direction_profile_id": "ego_degree10_12way_terminal_centroid_v2",
        "renderer_policy_digest": _HEX_A,
    }
    current = OpenLoopInputV1.model_validate(common)
    legacy = OpenLoopInputV1.model_validate({
        **common,
        "schema_version": "urbannav_openloop_input_v2_0",
    })

    assert current.schema_version == "urbannav_openloop_input_v2_1"
    assert legacy.schema_version == "urbannav_openloop_input_v2_0"

def test_join_uses_full_reference_denominator_and_rejects_extra_ids() -> None:
    target = _reasoning_target()
    references = {
        name: _reference(
            name,
            task_type="full_cot",
            target={"reasoning_target": target.model_dump(mode="json")},
        )
        for name in ("a", "b")
    }
    predictions = {
        "a": _prediction(
            references["a"],
            response=render_reasoning_target_v4(target),
        )
    }
    joined = join_openloop_artifacts_v1(references, predictions)
    assert coverage_metrics_v1(joined)["output_coverage"] == 0.5
    score = evaluate_full_cot_openloop_v1(joined)
    assert score["mechanical"]["strict_parse_rate"] == 0.5

    extra_reference = _reference(
        "extra",
        task_type="full_cot",
        target={"reasoning_target": target.model_dump(mode="json")},
    )
    with pytest.raises(ValueError, match="unknown sample ids"):
        join_openloop_artifacts_v1(
            references,
            {**predictions, "extra": _prediction(extra_reference, response="bad")},
        )


def test_vqa_multiset_order_and_full_reference_provenance_macro() -> None:
    question = VQAQuestionV2(
        task_id="a2_explicit_requirement",
        template_id="a2_v1",
        question_text="Which requirements are explicit?",
        evidence_refs=["instruction.active"],
    )
    answer = build_vqa_answer_v2(
        question.task_id,
        {
            "requirements": ["avoid_obstacles", "stay_on_walkable_surface"],
            "status": "resolved",
        },
    )
    refs = [
        _reference(
            sample_id,
            task_type="vqa",
            target={
                "question": question.model_dump(mode="json"),
                "answer": answer.model_dump(mode="json"),
            },
            provenance="lvlm_weak",
        )
        for sample_id in ("vqa-a", "vqa-b")
    ]
    reordered = build_vqa_answer_v2(
        question.task_id,
        {
            "requirements": ["stay_on_walkable_surface", "avoid_obstacles"],
            "status": "resolved",
        },
    )
    rows = [
        join_openloop_artifacts_v1(
            {refs[0].sample_id: refs[0]},
            {
                refs[0].sample_id: _prediction(
                    refs[0],
                    response=render_vqa_answer_v2(reordered),
                )
            },
        )[0],
        join_openloop_artifacts_v1(
            {refs[1].sample_id: refs[1]},
            {},
        )[0],
    ]
    score = evaluate_vqa_openloop_v1(rows)
    task = score["task_metrics"][question.task_id]
    assert task["strict_exact_full_reference"] == 0.5
    assert task["parse_rate_full_reference"] == 0.5
    assert (
        score["provenance_scorecards"]["lvlm_weak"]["primary_task_macro"]
        == 0.5
    )
    assert score["provenance_scorecards"]["human_gold"]["primary_task_macro"] is None


def test_decision_reports_50plus_separately_from_point_angle() -> None:
    range_ref = _reference(
        "range",
        task_type="decision",
        target={
            "decision": {
                "selected_action_label": "left_50plus",
                "speed": "slow",
                "response": "detour",
            }
        },
        provenance="deterministic_gt",
        stamp=_stamp("left_50plus"),
    )
    center_ref = _reference(
        "center",
        task_type="decision",
        target={
            "decision": {
                "selected_action_label": "straight",
                "speed": "normal",
                "response": "proceed",
            }
        },
        provenance="deterministic_gt",
        stamp=_stamp("straight"),
    )
    rows = join_openloop_artifacts_v1(
        {row.sample_id: row for row in (range_ref, center_ref)},
        {
            "range": _prediction(
                range_ref,
                response=render_reasoning_section_v4(
                    "Decision",
                    DecisionV4(
                        selected_action_label="left_50plus",
                        speed="slow",
                        response="detour",
                    ),
                ),
            ),
            "center": _prediction(
                center_ref,
                response=render_reasoning_section_v4(
                    "Decision",
                    DecisionV4(
                        selected_action_label="left_50plus",
                        speed="slow",
                        response="detour",
                    ),
                ),
            ),
        },
    )
    profile = evaluate_decision_openloop_v1(rows)["profiles"][
        "ego_degree10_12way_terminal_centroid_v2"
    ]
    assert profile["range_50plus"]["accuracy"] == 1.0
    assert profile["exact_center_angular"]["reference_count"] == 1
    assert profile["exact_center_angular"]["defined_count"] == 0
    assert profile["exact_center_angular"]["coverage"] == 0.0


def _trajectory_target(valid_count: int) -> StageIITrajectoryTargetV1:
    rows = [
        [0.125 * (index + 1), 0.01 * (index + 1), 0.01 * (index + 1)]
        if index < valid_count
        else [0.0, 0.0, 0.0]
        for index in range(16)
    ]
    mask = [index < valid_count for index in range(16)]
    policy = StageIITrajectoryPolicyV1(
        source_format="ego_xy_t_speed_right_front_v1",
        trajectory_source="calibrated_pose_projection",
        yaw_source="path_tangent",
    )
    payload = {
        "schema_version": STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION,
        "trajectory_ego_t": rows,
        "action_mask": mask,
        "valid_step_count": valid_count,
        "policy": policy.model_dump(mode="json"),
        "source_target_digest": _HEX_A,
    }
    payload["target_digest"] = stage_ii_digest_v1(payload)
    return StageIITrajectoryTargetV1.model_validate(payload)


def _context_identity() -> OpenLoopContextIdentityV1:
    return OpenLoopContextIdentityV1(
        policy="shared_current_checkpoint_v1",
        reasoning_context_source="model_prediction",
        decision_context_source="model_prediction",
        reasoning_producer_checkpoint_digest=_HEX_C,
        decision_producer_checkpoint_digest=_HEX_C,
        pass_t_backbone_checkpoint_digest=_HEX_C,
        reasoning_renderer_digest=_HEX_A,
        decision_renderer_digest=_HEX_B,
        input_manifest_digest=_HEX_D,
        adapter_input_manifest_digest=_HEX_C,
        reasoning_cache_digest=_HEX_A,
        decision_cache_digest=_HEX_B,
        normalization_profile_digest=_HEX_D,
        flow_steps=8,
        seed=17,
    )


def _run_manifest() -> OpenLoopRunManifestV1:
    return OpenLoopRunManifestV1(
        run_id="stage-ii-test",
        benchmark_id="benchmark-test",
        benchmark_digest=_HEX_A,
        stage="stage_ii",
        repositories=[
            OpenLoopRepositoryStateV1(
                name="RouteServer", commit="abcdef1", dirty=False
            ),
            OpenLoopRepositoryStateV1(
                name="ms-swift", commit="abcdef2", dirty=False
            ),
        ],
        checkpoint=OpenLoopCheckpointIdentityV1(
            model_id="qwen-test",
            checkpoint_id="checkpoint-test",
            checkpoint_digest=_HEX_C,
            tokenizer_id="tokenizer-test",
            processor_id="processor-test",
        ),
        renderer_policy=OpenLoopRendererPolicyV1(
            reasoning_renderer_id="reasoning-renderer",
            reasoning_renderer_digest=_HEX_A,
            decision_renderer_id="decision-renderer",
            decision_renderer_digest=_HEX_B,
            max_context_tokens=4096,
        ),
        metric_policy_digest=_HEX_D,
        context_identity=_context_identity(),
        batch_size=1,
        precision="bf16",
    )


def test_stage_ii_fixed_horizon_coverage_rejects_short_prediction() -> None:
    full_ok_target = _trajectory_target(16)
    partial_target = _trajectory_target(8)
    full_short_target = _trajectory_target(16)
    full_ok = _reference(
        "full-ok",
        task_type="trajectory",
        target={"trajectory_target": full_ok_target.model_dump(mode="json")},
        provenance="deterministic_gt",
        stamp=_stamp("straight"),
        mask=full_ok_target.action_mask,
    )
    partial = _reference(
        "partial",
        task_type="trajectory",
        target={"trajectory_target": partial_target.model_dump(mode="json")},
        provenance="deterministic_gt",
        stamp=_stamp("straight"),
        mask=partial_target.action_mask,
        dataset="egowalk",
    )
    full_short = _reference(
        "full-short",
        task_type="trajectory",
        target={"trajectory_target": full_short_target.model_dump(mode="json")},
        provenance="deterministic_gt",
        stamp=_stamp("straight"),
        mask=full_short_target.action_mask,
    )
    identity = _context_identity().model_dump(mode="json")
    predictions = {
        "full-ok": _prediction(
            full_ok,
            trajectory=full_ok_target.trajectory_ego_t,
            context_identity=identity,
            metadata={"decision_context_label": "straight"},
        ),
        "full-short": _prediction(
            full_short,
            trajectory=full_short_target.trajectory_ego_t[:8],
            context_identity=identity,
        ),
    }
    failure = OpenLoopFailureV1(
        sample_id=partial.sample_id,
        parent_sample_id=partial.parent_sample_id,
        stage="stage_ii",
        task_type="trajectory",
        attempted_pass="pass_r",
        failure_code="context_parse_failed",
        input_digest=_HEX_B,
    )
    joined = join_openloop_artifacts_v1(
        {row.sample_id: row for row in (full_ok, partial, full_short)},
        predictions,
        {partial.sample_id: failure},
    )
    score = evaluate_stage_ii_cfm_openloop_v1(
        joined,
        run_manifest=_run_manifest(),
        metric_policy=OpenLoopMetricPolicyV1(),
    )
    assert score["strict_output"]["invalid_output_count"] == 1
    assert score["horizons"]["0.500s"]["eligible_reference_count"] == 3
    assert score["horizons"]["0.500s"]["geometry_coverage"] == pytest.approx(1 / 3)
    assert score["horizons"]["2.000s"]["eligible_reference_count"] == 2
    assert score["horizons"]["2.000s"]["geometry_coverage"] == 0.5
    assert score["three_pass_reliability"]["passes"]["pass_t"] == {
        "attempted": 2,
        "succeeded": 1,
    }
    assert (
        score["three_pass_reliability"]["failure_codes"][
            "invalid_pass_t_output"
        ]
        == 1
    )
    assert score["physical_motion"]["status"] == "N/A"


def test_stage_ii_reports_selected_l2_at_fixed_physical_horizons() -> None:
    target = _trajectory_target(16)
    reference = _reference(
        "fixed-l2",
        task_type="trajectory",
        target={"trajectory_target": target.model_dump(mode="json")},
        provenance="deterministic_gt",
        stamp=_stamp("straight"),
        mask=target.action_mask,
    )
    trajectory = [row[:] for row in target.trajectory_ego_t]
    trajectory[7][0] += 0.25
    trajectory[15][0] += 0.75
    prediction = _prediction(
        reference,
        trajectory=trajectory,
        context_identity=_context_identity().model_dump(mode="json"),
    )
    joined = join_openloop_artifacts_v1(
        {reference.sample_id: reference},
        {prediction.sample_id: prediction},
        {},
    )

    score = evaluate_stage_ii_cfm_openloop_v1(
        joined,
        run_manifest=_run_manifest(),
        metric_policy=OpenLoopMetricPolicyV1(),
    )

    one_second = score["horizons"]["1.000s"]
    two_seconds = score["horizons"]["2.000s"]
    assert one_second["step_count"] == 8
    assert two_seconds["step_count"] == 16
    assert one_second["primary_eligible_dataset_macro"][
        "conditional_l2_m"
    ] == pytest.approx(0.25)
    assert two_seconds["primary_eligible_dataset_macro"][
        "conditional_l2_m"
    ] == pytest.approx(0.75)
    assert one_second["primary_eligible_dataset_macro"][
        "conditional_l2_m"
    ] == one_second["primary_eligible_dataset_macro"]["conditional_fde_m"]
    assert score["selected_l2"]["primary_eligible_dataset_macro"] == {
        "l2_1s_m": pytest.approx(0.25),
        "l2_2s_m": pytest.approx(0.75),
    }


def test_production_context_identity_and_exposure_fail_closed() -> None:
    with pytest.raises(ValueError, match="equal Pass R/D/T"):
        OpenLoopContextIdentityV1(
            **{
                **_context_identity().model_dump(mode="json"),
                "decision_producer_checkpoint_digest": _HEX_D,
            }
        )

    input_row = OpenLoopInputV1(
        sample_id="sample-1",
        parent_sample_id="parent-1",
        source_group_id="source-1",
        split_group_id="clip-1",
        split="test",
        stage="stage_i",
        task_type="decision",
        dataset_source="sidewalk",
        messages=[OpenLoopMessageV1(role="user", content="<image>go")],
        images=["current.jpg"],
        image_roles=["fpv.current"],
        modality_profile_id="fpv_current_v1",
        renderer_policy_digest=_HEX_A,
    )
    registry = CheckpointExposureRegistryV1(
        checkpoint_digest=_HEX_C,
        entries=[
            CheckpointExposureEntryV1(
                parent_sample_id="another-parent",
                split_group_id="clip-1",
                manifest_digest=_HEX_D,
                stage="stage_i_train",
            )
        ],
    )
    with pytest.raises(ValueError, match="exposure overlap"):
        audit_checkpoint_exposure_v1(
            [input_row],
            registry,
            checkpoint_digest=_HEX_C,
        )


def test_output_digest_and_explicit_exposure_registry_fail_closed(
    tmp_path,
) -> None:
    input_row = OpenLoopInputV1(
        sample_id="sample-1",
        parent_sample_id="parent-1",
        source_group_id="source-1",
        split_group_id="clip-1",
        split="validation",
        stage="stage_i",
        task_type="decision",
        dataset_source="sidewalk",
        messages=[OpenLoopMessageV1(role="user", content="<image>go")],
        images=["current.jpg"],
        image_roles=["fpv.current"],
        modality_profile_id="fpv_current_v1",
        renderer_policy_digest=_HEX_A,
    )
    reference = _reference(
        "sample-1",
        task_type="decision",
        target={
            "decision": {
                "selected_action_label": "straight",
                "speed": "normal",
                "response": "proceed",
            }
        },
    )
    prediction = _prediction(reference, response="invalid")
    with pytest.raises(ValueError, match="input digest differs"):
        audit_output_input_digests_v1(
            {input_row.sample_id: input_row},
            {prediction.sample_id: prediction},
        )
    bound_prediction = prediction.model_copy(
        update={"input_digest": canonical_row_digest_v1(input_row)}
    )
    audit_output_input_digests_v1(
        {input_row.sample_id: input_row},
        {bound_prediction.sample_id: bound_prediction},
    )

    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text(
        json.dumps(
            {
                "parent": "parent-1",
                "lineage": {"group": "clip-1", "split": "train"},
                "fingerprint": "source-fp",
            }
        )
        + "\n"
        + json.dumps(
            {
                "parent": "heldout-parent",
                "lineage": {"group": "heldout-clip", "split": "test"},
                "fingerprint": "heldout-fp",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = tmp_path / "exposure.json"
    spec.write_text(
        json.dumps(
            {
                "manifests": [
                    {
                        "path": "train.jsonl",
                        "stage": "stage_i_train",
                        "parent_field": "parent",
                        "split_group_field": "lineage.group",
                        "source_fingerprint_field": "fingerprint",
                        "annotation_version": "v4.6",
                        "where": {"lineage.split": "train"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    built = build_checkpoint_exposure_registry_v1(
        spec,
        tmp_path / "registry.json",
        checkpoint_digest=_HEX_C,
    )
    assert len(built.entries) == 1
    assert built.entries[0].manifest_digest != _HEX_D
    with pytest.raises(ValueError, match="exposure overlap"):
        audit_checkpoint_exposure_v1(
            [input_row],
            built,
            checkpoint_digest=_HEX_C,
        )
