from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_target_v4,
    render_reasoning_target_v4,
)
from mcp_server.agent.reasoning_v4_prompt_policy import (
    STUDENT_DECISION_PROFILE_V4,
    STUDENT_FULL_COT_PROFILE_V4,
)
from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_section_v4,
    validate_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    GroundingOutputsV4,
    InputModalitiesV4,
    NavObservationV4,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    GoalAlignmentV4,
    InteractionConstraintV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    NavigationSourceAlignmentV4,
    REASONING_V4_CONTRACT_SCHEMA_DIGEST,
    ReasoningTargetV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGoalV4,
    TaskGroundingV4,
    TraversabilityRegionV4,
)
from mcp_server.data.reasoning_v4_composer import (
    compose_reasoning_target_v4,
)
from mcp_server.data.reasoning_v4_audit import (
    audit_reasoning_v4_artifacts,
    audit_reasoning_v4_source_coverage,
)
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v4_source import build_reasoning_source_v4
from mcp_server.data.reasoning_v4_views import (
    build_reasoning_v4_user_prompt,
    build_reasoning_training_views_v4,
)
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration
from mcp_server.teacher.reasoning_v4_batch import (
    ReasoningV4BatchConfig,
    annotate_reasoning_source_v4,
    label_reasoning_source_manifest_v4,
)
from mcp_server.teacher.reasoning_v4_profiles import (
    REASONING_V4_TEACHER_PROFILES,
)
from mcp_server.teacher.reasoning_v4_teacher import (
    build_reasoning_v4_section_question,
)


def _source() -> ReasoningSourceV4:
    observation = NavObservationV4(
        source_group_id="sidewalk:case-1",
        dataset_source="sidewalk",
        source_record_id="case-1",
        split_group_id="clip-1",
        images=["frame.jpg"],
        image_roles=["fpv.current"],
        instruction="Continue straight on the sidewalk.",
        evidence_catalog=[
            {
                "ref": "instruction.active",
                "modality": "instruction",
            },
            {"ref": "fpv.current", "modality": "fpv"},
        ],
        input_modalities=InputModalitiesV4(instruction=True),
    )
    payload = {
        "schema_version": "urbannav_reasoning_source_v4_0",
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": None,
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": [],
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return ReasoningSourceV4.model_validate(payload)


def _groundings() -> GroundingOutputsV4:
    return GroundingOutputsV4(
        task_grounding=TaskGroundingV4(
            objective="follow_instruction",
            goal=TaskGoalV4(
                goal_type="none",
                specification="absent",
            ),
        ),
        scene_grounding=SceneGroundingV4(),
        local_traversability=LocalTraversabilityV4(
            regions=[
                TraversabilityRegionV4(
                    region_ref="region.center",
                    surface="sidewalk",
                    lateral_position="center",
                    state="direct",
                    clearance="sufficient",
                    evidence_refs=["fpv.current"],
                )
            ]
        ),
        navigation_context_grounding=NavigationContextGroundingV4(
            source_alignments=[
                NavigationSourceAlignmentV4(
                    alignment_ref="alignment.instruction",
                    source_ref="instruction.active",
                    local_effect="straight",
                    aligned_region_refs=["region.center"],
                    status="supports",
                    evidence_refs=[
                        "instruction.active",
                        "fpv.current",
                    ],
                )
            ],
            task_direction=TaskDirectionV4(
                desired_motion="straight",
                candidate_region_refs=["region.center"],
                status="aligned",
                evidence_refs=[
                    "instruction.active",
                    "fpv.current",
                ],
            ),
        ),
        interaction_constraint_grounding=(
            InteractionConstraintGroundingV4()
        ),
    )


def _decision() -> DecisionV4:
    return DecisionV4(
        decision_type="navigate",
        motion="go_straight",
        speed="normal",
        social_action="none",
        selected_region_ref="region.center",
        fallback="stop",
        support_refs=["nav.task_direction", "region.center"],
        evidence_refs=["instruction.active", "fpv.current"],
    )


def test_compose_render_parse_and_validate_round_trip() -> None:
    source = _source()
    composition = compose_reasoning_target_v4(
        source,
        _groundings(),
        _decision(),
    )
    text = render_reasoning_target_v4(composition.target)
    parsed = parse_reasoning_target_v4(text)
    checked = validate_reasoning_target_v4(
        composition.target,
        observation=source.observation,
        assistant_text=text,
    )

    assert parsed.valid, parsed.errors
    assert parsed.target == composition.target
    assert checked.valid, checked.errors
    assert "trajectory" not in text.lower()
    assert text.index("<TaskGrounding>") < text.index("<Decision>")
    assert "<InteractionConstraintGrounding>" in text


def test_decision_support_refs_accept_defined_navigation_alignment() -> None:
    decision = _decision().model_copy(
        update={
            "support_refs": [
                "alignment.instruction",
                "nav.task_direction",
                "region.center",
            ]
        }
    )
    composition = compose_reasoning_target_v4(
        _source(),
        _groundings(),
        decision,
    )

    assert composition.target.reasoning.decision.support_refs[0] == (
        "alignment.instruction"
    )


def test_prompt_prioritizes_grounded_navigation_over_prohibition_lists() -> None:
    full_prompt = STUDENT_FULL_COT_PROFILE_V4.system_prompt
    decision_prompt = STUDENT_DECISION_PROFILE_V4.system_prompt

    assert "supports people with disabilities" in full_prompt
    assert "Every grounding section independently reads" in full_prompt
    assert "A single FPV image may support visible occupancy" in full_prompt
    assert "source-to-path relation" in full_prompt
    assert "only allowed decision_type values" in decision_prompt
    assert "DecisionV4.decision_type: navigate | stop" in decision_prompt
    assert "request_user=material task ambiguity" not in decision_prompt
    assert "Never use or claim GPS, pose, calibration" not in full_prompt


def test_student_prompt_exposes_deterministic_temporal_eligibility() -> None:
    source = _source()
    annotation = asyncio.run(
        annotate_reasoning_source_v4(
            source,
            backend=_FakeBackend(),
            config=ReasoningV4BatchConfig(),
        )
    )
    view = next(
        item
        for item in build_reasoning_training_views_v4(
            source,
            annotation,
        )
        if item.view_type == "full_cot_sft"
    )
    payload = json.loads(
        build_reasoning_v4_user_prompt(view).split("\n", 2)[2]
    )

    assert payload["temporal_evidence_eligible"] is False


def test_source_keeps_private_geometry_out_and_preserves_direction_stamp() -> None:
    source = build_reasoning_source_v4(
        {
            "id": "case-source",
            "dataset_source": "sidewalk",
            "images": ["old.jpg", "current.jpg", "bev.jpg"],
            "image_roles": [
                "fpv.history_00",
                "fpv.current",
                "bev.current",
            ],
            "instruction": "Continue straight.",
            "model_visible_context": {
                "goal": {"description": "entrance"},
                "gps": {"latitude": 1.0, "longitude": 2.0},
                "runtime_task_phase": "approaching_goal",
            },
            "canonical_direction": {
                "profile_id": "fpv_degree10_terminal_centroid_v2",
                "profile_version": "2.0",
                "label_index": 9,
                "label_name": "straight",
                "choice": "J",
                "angle_deg": 0.0,
                "source": "pixel_trajectory_terminal_centroid",
                "bin_mode": "degree10_with_stop",
                "derivation_version": (
                    "pixel_trajectory_terminal_centroid_rule_v2"
                ),
            },
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )

    assert source.observation.temporal_evidence_eligible is True
    assert source.direction_supervision is not None
    assert source.direction_supervision.label_index == 9
    visible = json.dumps(source.observation.model_visible_context)
    assert "latitude" not in visible
    assert "runtime_task_phase" not in visible
    assert "trajectory" not in json.dumps(
        source.observation.model_dump(mode="json")
    )


def test_source_extracts_explicit_instruction_from_legacy_user_message() -> None:
    source = build_reasoning_source_v4(
        {
            "id": "legacy-message-case",
            "dataset_source": "sidewalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
            "messages": [
                {"role": "system", "content": "Legacy training prompt."},
                {
                    "role": "user",
                    "content": (
                        "<image>\nTask: navigation\n"
                        "Instruction: Continue along the visible sidewalk.\n"
                        "Output exactly the legacy format."
                    ),
                },
            ],
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )

    assert (
        source.observation.instruction
        == "Continue along the visible sidewalk."
    )
    assert source.observation.input_modalities.instruction is True
    assert "instruction.active" in {
        entry.ref for entry in source.observation.evidence_catalog
    }


def test_source_does_not_invent_instruction_when_source_has_none() -> None:
    source = build_reasoning_source_v4(
        {
            "id": "fpv-only-local-navigation",
            "dataset_source": "sidewalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )

    assert source.observation.instruction is None
    assert source.observation.input_modalities.instruction is False
    assert "instruction.active" not in {
        entry.ref for entry in source.observation.evidence_catalog
    }


def test_source_drops_unverified_coordinate_destination() -> None:
    source = build_reasoning_source_v4(
        {
            "id": "coordinate-destination",
            "dataset_source": "sidewalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
            "destination": [116.397, 39.908],
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )

    assert "goal" not in source.observation.model_visible_context
    assert (
        "unverified_or_coordinate_destination_dropped"
        in source.warnings
    )


def test_nav_observation_rejects_private_context_even_if_prebuilt() -> None:
    with pytest.raises(
        ValueError,
        match="private prompt-visible context keys",
    ):
        NavObservationV4(
            source_group_id="sidewalk:private-context",
            dataset_source="sidewalk",
            source_record_id="private-context",
            split_group_id="clip-private",
            images=["current.jpg"],
            image_roles=["fpv.current"],
            model_visible_context={
                "goal": {
                    "description": "entrance",
                    "gps_pose": {"latitude": 39.908},
                }
            },
            evidence_catalog=[
                {"ref": "fpv.current", "modality": "fpv"},
            ],
            input_modalities=InputModalitiesV4(),
        )


def test_dynamic_constraint_requires_distinct_visible_fpv_frames() -> None:
    repeated = NavObservationV4(
        source_group_id="sidewalk:dynamic-repeated",
        dataset_source="sidewalk",
        source_record_id="dynamic-repeated",
        split_group_id="clip-dynamic",
        images=["same.jpg", "same.jpg"],
        image_roles=["fpv.history_00", "fpv.current"],
        evidence_catalog=[
            {"ref": "fpv.history_00", "modality": "fpv"},
            {"ref": "fpv.current", "modality": "fpv"},
        ],
        input_modalities=InputModalitiesV4(history_fpv=True),
        temporal_evidence_eligible=False,
    )
    section = InteractionConstraintGroundingV4(
        constraints=[
            InteractionConstraintV4(
                constraint_ref="constraint.crossing_agent",
                constraint_code="yield_to_crossing_agent",
                kind="social",
                source_refs=["fpv.current"],
                applies_to_region_refs=["region.center"],
                relation="crosses_candidate_path",
                temporal_basis="multi_frame",
                status="applicable",
                effect="yield",
                evidence_refs=["fpv.history_00", "fpv.current"],
            )
        ]
    )

    rejected = validate_reasoning_section_v4(
        "InteractionConstraintGrounding",
        section,
        observation=repeated,
    )
    distinct = NavObservationV4.model_validate(
        {
            **repeated.model_dump(mode="json"),
            "images": ["old.jpg", "current.jpg"],
            "temporal_evidence_eligible": True,
        }
    )
    accepted = validate_reasoning_section_v4(
        "InteractionConstraintGrounding",
        section,
        observation=distinct,
    )

    assert any(
        error.startswith("dynamic_constraint_requires_distinct_frames:")
        for error in rejected.errors
    )
    assert accepted.valid, accepted.errors


def test_incomplete_legacy_direction_is_dropped_without_losing_source() -> None:
    source = build_reasoning_source_v4(
        {
            "id": "legacy-egowalk-direction",
            "dataset_source": "egowalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Continue safely.",
            "direction_choice_label": 19,
            "direction_name": "stop",
            "direction_choice": "T",
            "direction_bin_mode": "degree10_with_stop",
            "direction_source": "low_displacement_rule",
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="egowalk"),
    )

    assert source.direction_supervision is None
    assert any(
        warning.startswith("incomplete_legacy_direction_dropped:")
        for warning in source.warnings
    )


@pytest.mark.parametrize(
    "decision_type",
    ("finish", "request_user", "request_replan"),
)
def test_stage_a_rejects_non_navigation_control_states(
    decision_type: str,
) -> None:
    unsupported_decision = _decision().model_copy(
        update={
            "decision_type": decision_type,
            "motion": "stop",
            "speed": "stop",
            "selected_region_ref": None,
        }
    )

    checked = validate_reasoning_section_v4(
        "Decision",
        unsupported_decision,
        observation=_source().observation,
    )
    assert checked.errors == [
        f"stage_a_decision_type_not_allowed:{decision_type}"
    ]

    with pytest.raises(
        ValueError,
        match=f"stage_a_decision_type_not_allowed:{decision_type}",
    ):
        compose_reasoning_target_v4(
            _source(),
            _groundings(),
            unsupported_decision,
        )


def test_blocked_task_candidate_can_drive_stop() -> None:
    groundings = _groundings()
    blocked_local = LocalTraversabilityV4(
        regions=[
            TraversabilityRegionV4(
                region_ref="region.center",
                surface="sidewalk",
                lateral_position="center",
                state="blocked",
                clearance="insufficient",
                limiting_factor_codes=["visible_obstacle"],
                evidence_refs=["fpv.current"],
            )
        ]
    )
    blocked_groundings = groundings.model_copy(
        update={"local_traversability": blocked_local}
    )
    decision = DecisionV4(
        decision_type="stop",
        motion="stop",
        speed="stop",
        social_action="none",
        fallback="stop",
        support_refs=["nav.task_direction", "region.center"],
        evidence_refs=["instruction.active", "fpv.current"],
    )

    composition = compose_reasoning_target_v4(
        _source(),
        blocked_groundings,
        decision,
    )

    assert composition.target.reasoning.decision.decision_type == "stop"


def test_explicit_goal_requires_navigation_goal_alignment() -> None:
    groundings = _groundings()
    task = TaskGroundingV4(
        objective="approach_goal",
        goal=TaskGoalV4(
            goal_type="language_goal",
            description="the entrance",
            specification="specified",
            source_refs=["instruction.active"],
        ),
    )
    mismatched = groundings.model_copy(
        update={"task_grounding": task}
    )

    with pytest.raises(
        ValueError,
        match="explicit_task_goal_requires_goal_alignment",
    ):
        compose_reasoning_target_v4(
            _source(),
            mismatched,
            _decision(),
        )


def test_ambiguous_goal_alignment_requires_two_candidates() -> None:
    with pytest.raises(
        ValueError,
        match="at least two candidates",
    ):
        GoalAlignmentV4(
            status="ambiguous",
            candidate_target_refs=["entity.entrance_1"],
            evidence_refs=["instruction.active", "fpv.current"],
        )


def test_decision_constraint_support_must_apply_to_selected_region() -> None:
    constraint = InteractionConstraintV4(
        constraint_ref="constraint.left_clearance",
        constraint_code="maintain_clearance",
        kind="local_safety",
        source_refs=["fpv.current"],
        applies_to_region_refs=["region.left"],
        relation="near_candidate_path",
        temporal_basis="single_frame",
        status="applicable",
        effect="slow",
        evidence_refs=["fpv.current"],
    )
    groundings = _groundings().model_copy(
        update={
            "interaction_constraint_grounding": (
                InteractionConstraintGroundingV4(
                    constraints=[constraint]
                )
            )
        }
    )
    decision = _decision().model_copy(
        update={
            "support_refs": [
                "nav.task_direction",
                "region.center",
                constraint.constraint_ref,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "decision_support_constraint_not_applicable_to_selected_region"
        ),
    ):
        compose_reasoning_target_v4(
            _source(),
            groundings,
            decision,
        )


def test_export_emits_five_grounding_one_decision_and_one_full_view() -> None:
    source = _source()
    composition = compose_reasoning_target_v4(
        source,
        _groundings(),
        _decision(),
    )
    from mcp_server.contracts.reasoning_dataset_v4 import (
        AnnotationStageRecordV4,
        ReasoningAnnotationV4,
    )
    from mcp_server.teacher.reasoning_v4_profiles import (
        REASONING_V4_TEACHER_PROFILES,
    )

    records = {
        name: AnnotationStageRecordV4(
            status="accepted",
            attempts=1,
            prompt_profile_id=profile.profile_id,
            prompt_policy_digest=profile.policy_digest,
        )
        for name, profile in REASONING_V4_TEACHER_PROFILES.items()
    }
    annotation = ReasoningAnnotationV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="accepted",
        grounding_outputs=_groundings(),
        decision=_decision(),
        target=composition.target,
        stage_records=records,
    )

    views = build_reasoning_training_views_v4(source, annotation)

    assert len(views) == 7
    assert sum(view.view_type == "grounding_section_sft" for view in views) == 5
    assert sum(view.view_type == "decision_section_sft" for view in views) == 1
    assert sum(view.view_type == "full_cot_sft" for view in views) == 1
    assert all(
        "direction_supervision" not in view.model_visible_context
        for view in views
    )


def test_non_navigation_full_view_omits_system1_readout_and_sidecar(
    tmp_path: Path,
) -> None:
    payload = _source().model_dump(mode="json")
    payload.pop("source_fingerprint")
    payload.update(
        {
            "trajectory_ego_t": [[0.0, 0.0, 0.0]],
            "trajectory_policy": {
                "frame": "current_ego_flu",
                "target_kind": "cumulative",
            },
            "action_mask": [True],
        }
    )
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    source = ReasoningSourceV4.model_validate(payload)
    decision = DecisionV4(
        decision_type="stop",
        motion="stop",
        speed="stop",
        social_action="wait",
        fallback="stop",
        support_refs=["nav.task_direction"],
        evidence_refs=["fpv.current"],
    )
    composition = compose_reasoning_target_v4(
        source,
        _groundings(),
        decision,
    )
    from mcp_server.contracts.reasoning_dataset_v4 import (
        AnnotationStageRecordV4,
        ReasoningAnnotationV4,
    )
    from mcp_server.teacher.reasoning_v4_profiles import (
        REASONING_V4_TEACHER_PROFILES,
    )

    annotation = ReasoningAnnotationV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="accepted",
        grounding_outputs=_groundings(),
        decision=decision,
        target=composition.target,
        stage_records={
            name: AnnotationStageRecordV4(
                status="accepted",
                attempts=1,
                prompt_profile_id=profile.profile_id,
                prompt_policy_digest=profile.policy_digest,
            )
            for name, profile in REASONING_V4_TEACHER_PROFILES.items()
        },
    )

    views = build_reasoning_training_views_v4(source, annotation)
    full_view = next(
        view
        for view in views
        if view.view_type == "full_cot_sft"
    )

    assert "<readout>" not in full_view.assistant_text
    assert full_view.trajectory_ego_t is None
    assert full_view.trajectory_policy is None
    assert full_view.action_mask is None
    assert (
        full_view.metadata["system1_omitted_reason"]
        == "decision_not_navigate"
    )

    source_jsonl = tmp_path / "source.jsonl"
    annotation_jsonl = tmp_path / "annotation.jsonl"
    view_jsonl = tmp_path / "views.jsonl"
    source_jsonl.write_text(
        json.dumps(source.model_dump(mode="json", exclude_none=True))
        + "\n",
        encoding="utf-8",
    )
    annotation_jsonl.write_text(
        json.dumps(
            annotation.model_dump(mode="json", exclude_none=True)
        )
        + "\n",
        encoding="utf-8",
    )
    view_jsonl.write_text(
        "".join(
            json.dumps(view.model_dump(mode="json", exclude_none=True))
            + "\n"
            for view in views
        ),
        encoding="utf-8",
    )

    report = audit_reasoning_v4_artifacts(
        source_jsonl,
        annotation_jsonl,
        view_jsonl,
        required_dataset_families=("sidewalk",),
    )

    assert report["passed"], report["errors"]
    assert report["metrics"]["annotation_acceptance_fraction"] == 1.0
    assert (
        report["metrics"]["stage_metrics"]["Decision"][
            "acceptance_fraction"
        ]
        == 1.0
    )


def test_source_coverage_audit_detects_private_visible_context(
    tmp_path: Path,
) -> None:
    payload = _source().model_dump(mode="json")
    payload.pop("source_fingerprint")
    payload["observation"]["model_visible_context"] = {
        "goal": {"description": "entrance"},
        "gps_pose": {"latitude": 1.0},
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    source_jsonl = tmp_path / "source.jsonl"
    source_jsonl.write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )

    report = audit_reasoning_v4_source_coverage(
        [source_jsonl],
        required_dataset_families=("sidewalk",),
        min_source_rows=1,
    )

    assert report["passed"] is False
    assert any(
        error.startswith("private_visible_context_keys:")
        for error in report["errors"]
    )


def test_decision_teacher_exposes_only_stage_a_decision_types() -> None:
    profile = REASONING_V4_TEACHER_PROFILES["Decision"]
    question = build_reasoning_v4_section_question(
        _source().observation,
        profile,
        groundings=_groundings(),
    )
    payload = json.loads(question.split("\n\n", 1)[1])

    assert payload["output_json_schema"]["properties"][
        "decision_type"
    ]["enum"] == ["navigate", "stop"]
    assert "supports blind or low-vision users" in profile.system_prompt
    assert "source-to-path relation" in REASONING_V4_TEACHER_PROFILES[
        "InteractionConstraintGrounding"
    ].system_prompt
    assert "Do not use GPS, pose, calibration" not in profile.system_prompt


class _FakeBackend:
    async def generate(
        self,
        observation: NavObservationV4,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        assert not hasattr(observation, "direction_supervision")
        assert not hasattr(observation, "trajectory_ego_t")
        del observation, system_prompt, image_detail
        values = {
            "TaskGrounding": _groundings().task_grounding,
            "SceneGrounding": _groundings().scene_grounding,
            "LocalTraversability": _groundings().local_traversability,
            "NavigationContextGrounding": (
                _groundings().navigation_context_grounding
            ),
            "InteractionConstraintGrounding": (
                _groundings().interaction_constraint_grounding
            ),
            "Decision": _decision(),
        }
        section_name = next(
            name
            for name in values
            if f'"requested_section":"{name}"' in question
        )
        return TeacherGeneration(
            answer=json.dumps(
                values[section_name].model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
            provider="fake",
            model="fake-v4",
            telemetry={
                "roundtrip_ms": 100,
                "response": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "prompt_tokens_details": {
                            "cached_tokens": 2,
                        },
                    }
                },
            },
        )


class _OneSectionFailureBackend(_FakeBackend):
    async def generate(
        self,
        observation: NavObservationV4,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        if '"requested_section":"SceneGrounding"' in question:
            return TeacherGeneration(
                answer="not-json",
                provider="fake",
                model="fake-v4",
                telemetry={},
            )
        return await super().generate(
            observation,
            question=question,
            system_prompt=system_prompt,
            image_detail=image_detail,
        )


def test_teacher_runs_independent_groundings_then_decision() -> None:
    annotation = asyncio.run(
        annotate_reasoning_source_v4(
            _source(),
            backend=_FakeBackend(),
            config=ReasoningV4BatchConfig(max_retries=0),
        )
    )

    assert annotation.status == "accepted"
    assert annotation.target is not None
    assert set(annotation.stage_records) == {
        "TaskGrounding",
        "SceneGrounding",
        "LocalTraversability",
        "NavigationContextGrounding",
        "InteractionConstraintGrounding",
        "Decision",
    }


def test_failed_grounding_keeps_other_accepted_section_views() -> None:
    source = _source()
    annotation = asyncio.run(
        annotate_reasoning_source_v4(
            source,
            backend=_OneSectionFailureBackend(),
            config=ReasoningV4BatchConfig(max_retries=0),
        )
    )

    assert annotation.status == "partial"
    assert annotation.grounding_outputs.scene_grounding is None
    assert annotation.stage_records["SceneGrounding"].status == "rejected"
    assert annotation.stage_records["Decision"].status == "skipped"
    assert annotation.decision is None
    assert annotation.target is None

    views = build_reasoning_training_views_v4(source, annotation)
    assert len(views) == 4
    assert {
        view.section_name
        for view in views
    } == {
        "TaskGrounding",
        "LocalTraversability",
        "NavigationContextGrounding",
        "InteractionConstraintGrounding",
    }
    assert all(
        view.view_type == "grounding_section_sft"
        for view in views
    )


def test_direction_mismatch_quarantines_full_view_without_rewriting_decision() -> None:
    payload = _source().model_dump(mode="json")
    payload.pop("source_fingerprint")
    payload["direction_supervision"] = {
        "profile_id": "fpv_degree10_terminal_centroid_v2",
        "profile_version": "2.0",
        "label_index": 8,
        "label_name": "left_10",
        "choice": "I",
        "angle_deg": -10.0,
        "source": "pixel_trajectory_terminal_centroid_rule_v2",
        "bin_mode": "degree10_with_stop",
        "derivation_version": (
            "pixel_trajectory_terminal_centroid_rule_v2"
        ),
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(
        payload
    )
    source = ReasoningSourceV4.model_validate(payload)

    annotation = asyncio.run(
        annotate_reasoning_source_v4(
            source,
            backend=_FakeBackend(),
            config=ReasoningV4BatchConfig(max_retries=0),
        )
    )

    assert annotation.status == "partial"
    assert annotation.decision == _decision()
    assert annotation.target is None
    assert any(
        error.startswith(
            "composition_failed:decision_direction_coarse_mismatch:"
        )
        for error in annotation.errors
    )
    assert annotation.stage_records["Decision"].status == "accepted"

    views = build_reasoning_training_views_v4(source, annotation)
    assert len(views) == 6
    assert sum(
        view.view_type == "grounding_section_sft"
        for view in views
    ) == 5
    assert sum(
        view.view_type == "decision_section_sft"
        for view in views
    ) == 1
    assert not any(view.view_type == "full_cot_sft" for view in views)
    assert all(view.direction_supervision is None for view in views)


def test_batch_report_aggregates_stage_usage_latency_and_cost(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "source.jsonl"
    annotation_jsonl = tmp_path / "annotation.jsonl"
    report_json = tmp_path / "report.json"
    source_jsonl.write_text(
        json.dumps(
            _source().model_dump(mode="json", exclude_none=True)
        )
        + "\n",
        encoding="utf-8",
    )

    report = asyncio.run(
        label_reasoning_source_manifest_v4(
            source_jsonl,
            annotation_jsonl,
            report_json=report_json,
            config=ReasoningV4BatchConfig(
                max_retries=0,
                input_cost_per_million_tokens=1.0,
                output_cost_per_million_tokens=2.0,
            ),
            backend_factory=lambda worker_id, key: _FakeBackend(),
        )
    )

    assert report["accepted_rows"] == 1
    assert report["telemetry_totals"]["provider_calls"] == 6
    assert report["telemetry_totals"]["roundtrip_ms"] == 600.0
    assert report["telemetry_totals"]["prompt_tokens"] == 60
    assert report["telemetry_totals"]["completion_tokens"] == 30
    assert report["telemetry_totals"]["cached_prompt_tokens"] == 12
    assert report["stage_metrics"]["Decision"]["accepted"] == 1
    assert (
        report["stage_metrics"]["Decision"]["acceptance_fraction"]
        == 1.0
    )
    assert report["cost_estimate"]["estimated_total"] == pytest.approx(
        0.00012
    )


def test_ms_swift_contract_and_prompt_digests_match() -> None:
    code_root = Path(__file__).resolve().parents[4]
    ms_root = code_root / "ms-swift"
    if not ms_root.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(ms_root))
    try:
        consumer = importlib.import_module("swift.sidewalk_nav.reasoning_v4")
        ms_prompt = importlib.import_module("swift.sidewalk_nav.prompt_v4")
        from mcp_server.agent.reasoning_v4_prompt_policy import (
            PROMPT_PROFILE_REGISTRY_V4,
        )

        assert (
            consumer.REASONING_V4_CONTRACT_SCHEMA_DIGEST
            == REASONING_V4_CONTRACT_SCHEMA_DIGEST
        )
        assert (
            consumer.ReasoningTrainingViewV4.model_json_schema()
            == ReasoningTrainingViewV4.model_json_schema()
        )
        route_digests = {
            key: profile.policy_digest
            for key, profile in PROMPT_PROFILE_REGISTRY_V4.items()
        }
        ms_digests = {
            key: profile.policy_digest
            for key, profile in ms_prompt.PROMPT_PROFILE_REGISTRY_V4.items()
        }
        assert ms_digests == route_digests

        target = compose_reasoning_target_v4(
            _source(),
            _groundings(),
            _decision(),
        ).target
        canonical_text = render_reasoning_target_v4(target)
        ms_parsed = consumer.parse_reasoning_target_v4(canonical_text)
        assert ms_parsed.valid, ms_parsed.errors
        assert (
            ms_parsed.target.model_dump(mode="json")
            == target.model_dump(mode="json")
        )

        invalid_text = canonical_text + "\nfree text"
        route_invalid = parse_reasoning_target_v4(invalid_text)
        ms_invalid = consumer.parse_reasoning_target_v4(invalid_text)
        assert route_invalid.valid is False
        assert ms_invalid.valid is False
        assert route_invalid.errors == ms_invalid.errors
    finally:
        sys.path.remove(str(ms_root))
