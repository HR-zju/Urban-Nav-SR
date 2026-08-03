from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_target_v4,
    render_reasoning_target_v4,
)
from mcp_server.agent.reasoning_v4_prompt_policy import (
    get_prompt_profile_v4,
)
from mcp_server.agent.reasoning_v4_validation import (
    sanitize_reasoning_section_evidence_v4,
)
from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_AFFORDANCE7_V1,
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    ActionAdjudicationPayloadV4,
    ActionBlindReviewV4,
    ActionReviewComponentAssessmentV4,
    ActionReviewDiagnosisPayloadV4,
    ActionReviewDiagnosisRecordV4,
    AnnotationStageRecordV4,
    CandidateStructureRecordV4,
    DirectionContextV4,
    DirectionSupervisionV4,
    GroundingOutputsV4,
    InputModalitiesV4,
    NavObservationV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
    DirectionAffordanceV4,
    InteractionConstraintGroundingV4,
    InteractionConstraintV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    ReasoningTargetV4,
    ReasoningTraceV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGroundingV4,
)
from mcp_server.data.reasoning_v4_compact import (
    parse_compact_full_cot_action_v4,
)
from mcp_server.data.reasoning_v4_audit import (
    audit_reasoning_v4_artifacts,
    audit_reasoning_v4_pilot,
)
from mcp_server.data.reasoning_v4_composer import (
    REASONING_V4_COMPOSER_VERSION,
    compose_reasoning_target_v4,
)
from mcp_server.data.reasoning_v4_reconciliation import (
    action_reconciliation_advisory_trigger_codes_v4,
    action_reconciliation_blocking_trigger_codes_v4,
    action_reconciliation_review_priority_v4,
    action_reconciliation_review_required_v4,
    build_action_reconciliation_audit_v4,
)
from mcp_server.data.reasoning_v4_migration import (
    migrate_reasoning_annotation_v4,
)
from mcp_server.data.reasoning_v4_reconciliation_review import (
    export_action_reconciliation_review_v4,
)
from mcp_server.data.reasoning_v4_source import (
    build_reasoning_source_v4,
)
from mcp_server.data.reasoning_v4_state import (
    derive_reasoning_release_state_v4,
)
from mcp_server.data.reasoning_v4_review import (
    bind_action_adjudication_v4,
    bind_action_review_diagnosis_v4,
    build_reasoning_repair_request_v4,
)
from mcp_server.data.reasoning_v4_task_canonicalization import (
    canonicalize_task_instruction_v4,
    normalize_task_instruction_v4,
    task_instruction_fingerprint_v4,
)
from mcp_server.data.reasoning_v4_views import (
    build_reasoning_training_views_v4,
)
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration
from mcp_server.teacher.reasoning_v4_batch import (
    ReasoningV4BatchConfig,
    annotate_reasoning_source_v4,
    repair_reasoning_annotation_v4,
)
from mcp_server.teacher.reasoning_v4_image_policy import (
    REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID,
)
from mcp_server.teacher.reasoning_v4_review import (
    build_reasoning_v4_blind_review_question,
    build_reasoning_v4_comparative_review_question,
    parse_reasoning_v4_review_answer,
)
from mcp_server.teacher.reasoning_v4_review_batch import (
    ReasoningV4ReviewBatchConfig,
    diagnose_reasoning_action_review_v4,
    label_reasoning_review_queue_v4,
)
from mcp_server.teacher.reasoning_v4_review_profiles import (
    REASONING_V4_REVIEW_PROFILES,
)
from tools.model_processing.build_reasoning_v4_review_html import (
    build_reasoning_v4_review_html,
)
from tools.model_processing.import_reasoning_v4_adjudications import (
    import_reasoning_v4_adjudications,
)


def _direction(label: str = "straight") -> DirectionSupervisionV4:
    labels = [
        *[f"left_{angle}" for angle in range(90, 0, -10)],
        "straight",
        *[f"right_{angle}" for angle in range(10, 100, 10)],
        "stop",
    ]
    index = labels.index(label)
    return DirectionSupervisionV4(
        profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
        profile_version="2.0",
        label_index=index,
        label_name=label,
        choice=chr(ord("A") + index),
        angle_deg=(
            None if label == "stop" else float((index - 9) * 10)
        ),
        source=(
            "finish_stop_rule"
            if label == "stop"
            else "pixel_trajectory_terminal_centroid_rule_v2"
        ),
        bin_mode="degree10_with_stop",
        derivation_version=(
            "pixel_trajectory_terminal_centroid_rule_v2"
        ),
    )


def _source(
    label: str = "straight",
    *,
    warnings: list[str] | None = None,
) -> ReasoningSourceV4:
    observation = NavObservationV4(
        source_group_id=f"sidewalk:case-{label}",
        dataset_source="sidewalk",
        source_record_id=f"case-{label}",
        split_group_id="clip-1",
        images=["frame.jpg"],
        image_roles=["fpv.current"],
        instruction="Follow the route.",
        evidence_catalog=[
            {"ref": "instruction.active", "modality": "instruction"},
            {"ref": "fpv.current", "modality": "fpv"},
        ],
        input_modalities=InputModalitiesV4(instruction=True),
        direction_context=DirectionContextV4(
            action_profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
            action_profile_version="2.0",
            affordance_probe_profile_id=FPV_DEGREE10_AFFORDANCE7_V1,
            affordance_probe_profile_version="1.0",
        ),
    )
    payload = {
        "schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION,
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": _direction(label).model_dump(
            mode="json"
        ),
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": list(warnings or []),
    }
    payload["source_fingerprint"] = (
        reasoning_source_fingerprint_v4(payload)
    )
    return ReasoningSourceV4.model_validate(payload)


def test_sidewalk_nested_metadata_produces_stable_v4_source_identity() -> None:
    row = {
        "fpv_image_path": "/tmp/frame.jpg",
        "sidewalk_meta": {
            "clip_id": "sidewalk-clip-a",
            "frame_idx": 41,
            "frame_name": "000042",
        },
        "direction_supervision": _direction().model_dump(mode="json"),
    }

    source = build_reasoning_source_v4(
        row,
        index=7,
        config=ReasoningSourceBuildConfig(
            dataset_source="sidewalk",
            enable_system1=False,
        ),
    )

    assert source.observation.source_record_id == (
        "sidewalk-clip-a:000042"
    )
    assert source.observation.source_group_id == (
        "sidewalk:sidewalk-clip-a:000042"
    )
    assert source.observation.split_group_id == "sidewalk-clip-a"


def _groundings(
    *,
    effect: str | None = None,
) -> GroundingOutputsV4:
    constraints = []
    if effect is not None:
        constraints.append(
            InteractionConstraintV4(
                kind="crossing_interaction",
                status="applicable",
                effect=effect,
                affected_direction_labels=(
                    []
                    if effect in {"slow", "yield", "stop"}
                    else ["straight"]
                ),
            )
        )
    return GroundingOutputsV4(
        task_grounding=TaskGroundingV4(
            objective="follow_route",
            requirements=["route_adherence"],
        ),
        scene_grounding=SceneGroundingV4(
            scene_type="urban_sidewalk",
            walkable_support="paved_sidewalk",
            key_entities=["pedestrian"],
            traffic_context="no_active_control_visible",
        ),
        local_traversability=LocalTraversabilityV4(
            direction_affordances=[
                DirectionAffordanceV4(
                    direction_label="straight",
                    affordance="direct",
                    clearance="sufficient",
                )
            ]
        ),
        navigation_context_grounding=NavigationContextGroundingV4(
            task_direction=TaskDirectionV4(
                status="not_visible",
                source="none",
            )
        ),
        interaction_constraint_grounding=(
            InteractionConstraintGroundingV4(
                constraints=constraints
            )
        ),
    )


def _stage_records() -> dict[str, AnnotationStageRecordV4]:
    names = (
        "TaskGrounding",
        "SceneGrounding",
        "LocalTraversability",
        "NavigationContextGrounding",
        "InteractionConstraintGrounding",
        "Decision",
    )
    records = {}
    for name in names:
        deterministic = name == "TaskGrounding"
        records[name] = AnnotationStageRecordV4(
            status="accepted",
            attempts=0 if deterministic else 1,
            producer_kind=(
                "deterministic" if deterministic else "lvlm"
            ),
            producer_profile_id=f"producer_{name.lower()}",
            derivation_version=f"{name.lower()}_v1",
            prompt_policy_digest=(
                None if deterministic else "a" * 64
            ),
        )
    return records


def _annotation(
    source: ReasoningSourceV4,
    *,
    proposal_label: str = "straight",
    effect: str | None = None,
) -> ReasoningAnnotationV4:
    groundings = _groundings(effect=effect)
    proposal = DecisionProposalV4(
        selected_action_label=proposal_label,
        speed=(
            "stop" if proposal_label == "stop" else "normal"
        ),
        response=(
            "stop" if proposal_label == "stop" else "proceed"
        ),
    )
    audit = build_action_reconciliation_audit_v4(
        source,
        groundings,
        proposal,
    )
    composition = compose_reasoning_target_v4(
        source,
        groundings,
        proposal,
    )
    target = composition.candidate_target
    return ReasoningAnnotationV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        record_status="complete",
        grounding_outputs=groundings,
        decision_proposal=proposal,
        action_reconciliation=audit,
        composer_candidate=target.reasoning.decision,
        candidate_target=target,
        candidate_structure_record=CandidateStructureRecordV4(
            composer_version=REASONING_V4_COMPOSER_VERSION,
            status="valid",
        ),
        stage_records=_stage_records(),
    )


def _review_diagnosis(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
) -> ActionReviewDiagnosisRecordV4:
    """Build one completed teacher-proposal-error diagnosis fixture."""

    audit = annotation.action_reconciliation
    assert audit is not None
    payload = ActionReviewDiagnosisPayloadV4(
        verdict="teacher_proposal_error",
        error_subtype="proposal_direction_error",
        affected_components=["DecisionProposal"],
        component_assessment=ActionReviewComponentAssessmentV4(
            teacher_proposal="likely_wrong",
            grounding="likely_correct",
            strong_label="likely_correct",
        ),
        reason_codes=["proposal_conflicts_with_visible_direction"],
        error_analysis=(
            "Visible context supports the strong direction more than the "
            "teacher proposal."
        ),
        repair_action="keep_without_change",
        fields_to_inspect=["decision_proposal.selected_action_label"],
        confidence=0.9,
    )
    diagnosis = bind_action_review_diagnosis_v4(
        source,
        annotation,
        payload,
    )
    stage_records = {
        name: AnnotationStageRecordV4(
            status="accepted",
            attempts=1,
            producer_kind="lvlm",
            producer_profile_id=f"review_{name}",
            derivation_version=f"review_{name}_v1",
            prompt_policy_digest="b" * 64,
        )
        for name in ("blind_visual", "comparative_diagnosis")
    }
    return ActionReviewDiagnosisRecordV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        audit_id=audit.audit_id,
        record_status="complete",
        blind_assessment=ActionBlindReviewV4(
            evidence_sufficiency="sufficient",
            visual_findings=["visible_turn_support"],
            relevant_components=["NavigationContextGrounding"],
            evidence_analysis="The route cue visibly bends toward one side.",
        ),
        diagnosis=diagnosis,
        stage_records=stage_records,
    )


def _keep_adjudication(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    diagnosis: ActionReviewDiagnosisRecordV4,
):
    """Build one human keep-without-change adjudication fixture."""

    return bind_action_adjudication_v4(
        source,
        annotation,
        diagnosis,
        ActionAdjudicationPayloadV4(
            diagnosis_disposition="confirmed",
            final_verdict="teacher_proposal_error",
            error_subtype="proposal_direction_error",
            affected_components=["DecisionProposal"],
            reason_codes=[
                "proposal_conflicts_with_visible_direction"
            ],
            repair_action="keep_without_change",
            fields_to_inspect=[
                "decision_proposal.selected_action_label"
            ],
        ),
        reviewer="reviewer-1",
        reviewed_at=datetime.now(timezone.utc),
    )


def test_v46_wire_has_exact_six_sections_and_no_finish() -> None:
    source = _source()
    target = _annotation(source).candidate_target
    assert target is not None

    text = render_reasoning_target_v4(target)
    parsed = parse_reasoning_target_v4(text)

    assert parsed.valid
    assert parsed.target == target
    assert "<Finish>" not in text
    assert "<finish>" not in text
    assert "<readout>" not in text
    assert text.count("<Decision>") == 1
    assert (
        parse_compact_full_cot_action_v4(
            text,
            action_profile_id=(
                source.observation.direction_context.action_profile_id
            ),
        ).selected_action_label
        == "straight"
    )


def test_removed_fields_are_rejected_instead_of_ignored() -> None:
    payload = {
        "scene_type": "urban_sidewalk",
        "walkable_support": "paved_sidewalk",
        "key_entities": [],
        "traffic_context": "none_visible",
        "entities": [],
    }
    with pytest.raises(ValidationError):
        SceneGroundingV4.model_validate(payload)


def test_task_direction_requires_minimal_visible_provenance() -> None:
    with pytest.raises(
        ValidationError,
        match="aligned task direction requires visible source",
    ):
        TaskDirectionV4(
            status="aligned",
            source="none",
            direction="left_20",
        )
    with pytest.raises(
        ValidationError,
        match="requires source=none",
    ):
        TaskDirectionV4(
            status="not_visible",
            source="bev_route",
        )


def test_only_avoid_constraint_may_scope_direction_labels() -> None:
    with pytest.raises(
        ValidationError,
        match="yield constraint must be global",
    ):
        InteractionConstraintV4(
            kind="crossing_interaction",
            status="applicable",
            effect="yield",
            affected_direction_labels=["straight"],
        )
    avoid = InteractionConstraintV4(
        kind="static_obstacle_avoidance",
        status="applicable",
        effect="avoid",
        affected_direction_labels=["right_10"],
    )
    assert [str(value) for value in avoid.affected_direction_labels] == [
        "right_10"
    ]


def test_scene_and_constraint_synonyms_are_deterministically_normalized() -> None:
    observation = _source().observation
    scene, scene_warnings = sanitize_reasoning_section_evidence_v4(
        "SceneGrounding",
        SceneGroundingV4(
            scene_type="urban_intersection",
            walkable_support="crosswalk",
            key_entities=[
                "car",
                "parked_vehicles",
                "delivery_scooter",
                "parked_bicycles",
                "bushes",
                "hedges",
                "trees",
                "signage",
                "street_lamp",
                "traffic_cones",
            ],
            traffic_context="active_pedestrian_signal",
        ),
        observation=observation,
    )
    constraints, constraint_warnings = (
        sanitize_reasoning_section_evidence_v4(
            "InteractionConstraintGrounding",
            InteractionConstraintGroundingV4(
                constraints=[
                    InteractionConstraintV4(
                        kind="traffic_signal",
                        status="applicable",
                        effect="yield",
                    ),
                    InteractionConstraintV4(
                        kind="sidewalk_obstacle",
                        status="applicable",
                        effect="avoid",
                        affected_direction_labels=["straight"],
                    ),
                ]
            ),
            observation=observation,
        )
    )

    assert isinstance(scene, SceneGroundingV4)
    assert [str(value) for value in scene.key_entities] == [
        "vehicle",
        "micromobility",
        "bicycle",
        "bush",
        "hedge",
        "tree",
        "sign",
        "street_light",
        "traffic_cone",
    ]
    assert str(scene.traffic_context) == "traffic_control_present"
    assert scene_warnings == ["scene_vocabulary_normalized"]
    assert isinstance(constraints, InteractionConstraintGroundingV4)
    assert [
        str(value.kind) for value in constraints.constraints
    ] == ["traffic_control", "local_obstacle"]
    assert constraint_warnings == ["constraint_vocabulary_normalized"]


def test_reasoning_migration_updates_groundings_candidate_and_audit() -> None:
    source = _source()
    annotation = _annotation(source)
    alias_scene = SceneGroundingV4(
        scene_type="urban_sidewalk",
        walkable_support="sidewalk",
        key_entities=["scooter"],
        traffic_context="traffic_light_visible",
    )
    groundings = annotation.grounding_outputs.model_copy(
        update={"scene_grounding": alias_scene}
    )
    assert annotation.candidate_target is not None
    old_trace = annotation.candidate_target.reasoning
    target = ReasoningTargetV4(
        reasoning=ReasoningTraceV4(
            task_grounding=old_trace.task_grounding,
            scene_grounding=alias_scene,
            local_traversability=old_trace.local_traversability,
            navigation_context_grounding=(
                old_trace.navigation_context_grounding
            ),
            interaction_constraint_grounding=(
                old_trace.interaction_constraint_grounding
            ),
            decision=old_trace.decision,
        )
    )
    assert annotation.decision_proposal is not None
    legacy = ReasoningAnnotationV4.model_validate(
        annotation.model_copy(
            update={
                "grounding_outputs": groundings,
                "candidate_target": target,
                "action_reconciliation": (
                    build_action_reconciliation_audit_v4(
                        source,
                        groundings,
                        annotation.decision_proposal,
                    )
                ),
            }
        ).model_dump(mode="json", exclude_none=True)
    )

    migrated, warnings = migrate_reasoning_annotation_v4(source, legacy)

    scene = migrated.grounding_outputs.scene_grounding
    assert scene is not None
    assert [str(value) for value in scene.key_entities] == [
        "micromobility"
    ]
    assert str(scene.traffic_context) == "traffic_control_present"
    assert migrated.candidate_target is not None
    assert migrated.candidate_target.reasoning.scene_grounding == scene
    assert migrated.action_reconciliation != legacy.action_reconciliation
    assert warnings == ("scene_vocabulary_normalized",)


@pytest.mark.parametrize(
    ("effect", "expected_speed", "expected_response"),
    [
        (None, "normal", "proceed"),
        ("slow", "slow", "slow"),
        ("yield", "slow", "yield"),
        ("avoid", "slow", "detour"),
    ],
)
def test_strong_direction_owns_action_and_effect_maps_tuple(
    effect: str | None,
    expected_speed: str,
    expected_response: str,
) -> None:
    source = _source("straight")
    proposal = DecisionProposalV4(
        selected_action_label="left_40",
        speed="normal",
        response="proceed",
    )

    result = compose_reasoning_target_v4(
        source,
        _groundings(effect=effect),
        proposal,
    )
    decision = result.candidate_target.reasoning.decision

    assert str(decision.selected_action_label) == "straight"
    assert decision.speed == expected_speed
    assert decision.response == expected_response


def test_moving_strong_direction_and_global_stop_is_invalid() -> None:
    with pytest.raises(
        ValueError,
        match="strong_moving_action_conflicts_with_stop_constraint",
    ):
        compose_reasoning_target_v4(
            _source("straight"),
            _groundings(effect="stop"),
            DecisionProposalV4(
                selected_action_label="stop",
                speed="stop",
                response="stop",
            ),
        )


def test_avoid_subset_means_detour_required_not_direction_forbidden() -> None:
    inside = compose_reasoning_target_v4(
        _source("straight"),
        _groundings(effect="avoid"),
        DecisionProposalV4(
            selected_action_label="straight",
            speed="slow",
            response="detour",
        ),
    ).candidate_target.reasoning.decision
    outside = compose_reasoning_target_v4(
        _source("right_10"),
        _groundings(effect="avoid"),
        DecisionProposalV4(
            selected_action_label="right_10",
            speed="normal",
            response="proceed",
        ),
    ).candidate_target.reasoning.decision
    stopped = compose_reasoning_target_v4(
        _source("stop"),
        _groundings(effect="avoid"),
        DecisionProposalV4(
            selected_action_label="stop",
            speed="stop",
            response="stop",
        ),
    ).candidate_target.reasoning.decision

    assert (inside.speed, inside.response) == ("slow", "detour")
    assert (outside.speed, outside.response) == ("normal", "proceed")
    assert (stopped.speed, stopped.response) == ("stop", "stop")


def test_stop_is_valid_short_horizon_action_without_finish() -> None:
    source = _source("stop")
    annotation = _annotation(source, proposal_label="stop")
    decision = annotation.candidate_target.reasoning.decision

    assert str(decision.selected_action_label) == "stop"
    assert (decision.speed, decision.response) == ("stop", "stop")
    assert "<Finish>" not in render_reasoning_target_v4(
        annotation.candidate_target
    )


def test_task_canonicalization_is_stable_and_registry_owned() -> None:
    variants = (
        "  FOLLOW the route!!! ",
        "ＦＯＬＬＯＷ　ＴＨＥ　ＲＯＵＴＥ。",
    )

    assert {
        normalize_task_instruction_v4(value) for value in variants
    } == {"follow the route"}
    assert len(
        {
            task_instruction_fingerprint_v4(value)
            for value in variants
        }
    ) == 1
    result = canonicalize_task_instruction_v4(variants[0])
    assert result is not None
    assert result.source == "versioned_registry"
    assert str(result.task_grounding.objective) == "follow_route"
    assert [
        str(value) for value in result.task_grounding.requirements
    ] == ["route_adherence"]
    project_template = canonicalize_task_instruction_v4(
        "Navigate safely on the sidewalk and follow the visible route."
    )
    assert project_template is not None
    assert project_template.task_grounding == TaskGroundingV4(
        objective="follow_route",
        requirements=["route_adherence"],
        ambiguities=[],
    )
    local_selection = canonicalize_task_instruction_v4(
        "Navigate safely within the sidewalk or other pedestrian-accessible "
        "space. Choose one short-horizon local motion direction from the "
        "provided candidate options."
    )
    assert local_selection is not None
    assert local_selection.task_grounding == TaskGroundingV4(
        objective="choose_local_motion_direction",
        requirements=[
            "pedestrian_accessible_navigation",
            "short_horizon_motion_selection",
        ],
        ambiguities=[],
    )


def test_reconciliation_requires_review_without_mutating_strong_action() -> None:
    source = _source("left_40")
    annotation = _annotation(
        source,
        proposal_label="straight",
    )
    audit = annotation.action_reconciliation

    assert audit is not None
    assert action_reconciliation_review_required_v4(audit)
    assert str(
        annotation.candidate_target.reasoning.decision
        .selected_action_label
    ) == "left_40"


def test_human_review_angle_threshold_is_independent_from_profile_tolerance() -> None:
    source = _source("straight")
    supported = _groundings()
    at_threshold = build_action_reconciliation_audit_v4(
        source,
        supported,
        DecisionProposalV4(
            selected_action_label="right_20",
            speed="normal",
            response="proceed",
        ),
    )
    above_threshold = build_action_reconciliation_audit_v4(
        source,
        supported,
        DecisionProposalV4(
            selected_action_label="right_30",
            speed="normal",
            response="proceed",
        ),
    )
    unsupported = supported.model_copy(
        update={
            "local_traversability": LocalTraversabilityV4(
                direction_affordances=[
                    DirectionAffordanceV4(
                        direction_label="straight",
                        affordance="blocked",
                        clearance="insufficient",
                        limiting_factors=["static_obstacle"],
                    )
                ]
            )
        }
    )
    threshold_but_unsupported = build_action_reconciliation_audit_v4(
        source,
        unsupported,
        DecisionProposalV4(
            selected_action_label="right_20",
            speed="normal",
            response="proceed",
        ),
    )
    within_tolerance_unsupported = build_action_reconciliation_audit_v4(
        source,
        unsupported,
        DecisionProposalV4(
            selected_action_label="right_10",
            speed="normal",
            response="proceed",
        ),
    )

    assert at_threshold is not None
    assert at_threshold.checks.proposal_strong_relation == "large_delta"
    assert not action_reconciliation_review_required_v4(at_threshold)
    assert above_threshold is not None
    assert {
        str(value) for value in above_threshold.review_trigger_codes
    } == {"proposal_direction_delta_exceeds_profile_tolerance"}
    assert threshold_but_unsupported is not None
    assert {
        str(value)
        for value in threshold_but_unsupported.review_trigger_codes
    } == {"strong_action_probe_unsupported"}
    assert not action_reconciliation_review_required_v4(
        threshold_but_unsupported
    )
    assert action_reconciliation_blocking_trigger_codes_v4(
        threshold_but_unsupported
    ) == ()
    assert action_reconciliation_advisory_trigger_codes_v4(
        threshold_but_unsupported
    ) == ("strong_action_probe_unsupported",)
    assert (
        action_reconciliation_review_priority_v4(
            threshold_but_unsupported
        )
        == "not_required"
    )
    assert within_tolerance_unsupported is not None
    assert (
        within_tolerance_unsupported.checks.proposal_strong_relation
        == "within_tolerance"
    )
    assert not action_reconciliation_review_required_v4(
        within_tolerance_unsupported
    )


def test_reconciliation_preserves_all_nearest_probe_ties() -> None:
    source = _source("right_20")
    groundings = _groundings()
    groundings = groundings.model_copy(
        update={
            "local_traversability": LocalTraversabilityV4(
                direction_affordances=[
                    DirectionAffordanceV4(
                        direction_label="right_10",
                        affordance="direct",
                        clearance="sufficient",
                    ),
                    DirectionAffordanceV4(
                        direction_label="right_30",
                        affordance="blocked",
                        clearance="insufficient",
                        limiting_factors=["static_obstacle"],
                    ),
                ]
            )
        }
    )
    audit = build_action_reconciliation_audit_v4(
        source,
        groundings,
        DecisionProposalV4(
            selected_action_label="right_20",
            speed="normal",
            response="proceed",
        ),
    )

    assert audit is not None
    assert [
        str(item.direction_label)
        for item in audit.probe_projections
    ] == ["right_10", "right_30"]
    assert audit.checks.probe_support == "mixed"
    assert "strong_action_probe_mixed" in {
        str(value) for value in audit.review_trigger_codes
    }


def test_preview_and_formal_release_are_separate() -> None:
    source = _source()
    annotation = _annotation(source)

    preview = build_reasoning_training_views_v4(
        source,
        annotation,
        release_track="review_preview",
    )
    formal = build_reasoning_training_views_v4(
        source,
        annotation,
        release_track="formal_training",
    )

    assert len(preview) == 1
    assert preview[0].release_track == "review_preview"
    assert preview[0].artifact_status == "preview_only"
    assert preview[0].artifact_training_eligible is False
    assert preview[0].formal_training_eligible is True
    assert len(formal) == 1
    assert formal[0].artifact_status == "training_ready"
    assert formal[0].artifact_training_eligible is True
    assert formal[0].formal_training_eligible is True
    assert formal[0].review_status == "not_required"


def test_v46_system1_view_requires_all_sixteen_mask_steps_valid() -> None:
    source = _source()
    annotation = _annotation(source)
    payload = build_reasoning_training_views_v4(
        source,
        annotation,
        release_track="formal_training",
    )[0].model_dump(mode="json", exclude_none=True)
    payload["trajectory_ego_t"] = [
        [0.1 * index, 0.0, 0.0] for index in range(16)
    ]
    payload["trajectory_policy"] = {
        "system1_target_field": "trajectory_ego_t",
    }
    payload["action_mask"] = [1] * 15 + [0]

    with pytest.raises(
        ValidationError,
        match="must mark all 16 steps valid",
    ):
        ReasoningTrainingViewV4.model_validate(payload)


def test_post_threshold_formal_gate_keeps_pending_candidates_out() -> None:
    source = _source("straight")
    annotations = [
        *[_annotation(source, proposal_label="straight") for _ in range(6)],
        _annotation(source, proposal_label="right_20"),
        *[_annotation(source, proposal_label="right_30") for _ in range(7)],
    ]

    formal = [
        view
        for annotation in annotations
        for view in build_reasoning_training_views_v4(
            source,
            annotation,
            release_track="formal_training",
        )
    ]
    preview = [
        view
        for annotation in annotations
        for view in build_reasoning_training_views_v4(
            source,
            annotation,
            release_track="review_preview",
        )
    ]

    assert len(preview) == 14
    assert len(formal) == 7
    assert all(
        view.formal_training_eligible
        and view.review_status in {"not_required", "resolved"}
        for view in formal
    )
    assert sum(
        view.review_status == "pending_human"
        for view in preview
    ) == 7


def test_triggered_review_requires_diagnosis_bound_adjudication() -> None:
    source = _source("left_40")
    annotation = _annotation(
        source,
        proposal_label="straight",
    )
    audit = annotation.action_reconciliation
    assert audit is not None

    pending = derive_reasoning_release_state_v4(
        source,
        annotation,
    )
    assert pending.review_status == "pending_human"
    assert pending.formal_training_eligible is False

    diagnosis = _review_diagnosis(source, annotation)
    adjudication = _keep_adjudication(
        source,
        annotation,
        diagnosis,
    )
    rejected = derive_reasoning_release_state_v4(
        source,
        annotation,
        adjudication=adjudication,
    )
    assert rejected.formal_training_eligible is False
    assert (
        "adjudication_diagnosis_lineage_invalid"
        in rejected.blockers
    )
    resolved = derive_reasoning_release_state_v4(
        source,
        annotation,
        diagnosis=diagnosis,
        adjudication=adjudication,
    )
    assert resolved.review_status == "resolved"
    assert resolved.formal_training_eligible is True


def test_adjudication_aware_artifact_audit_reports_six_axes(
    tmp_path: Path,
) -> None:
    source = _source("left_40")
    annotation = _annotation(
        source,
        proposal_label="straight",
    )
    audit = annotation.action_reconciliation
    assert audit is not None
    diagnosis = _review_diagnosis(source, annotation)
    adjudication = _keep_adjudication(
        source,
        annotation,
        diagnosis,
    )
    views = build_reasoning_training_views_v4(
        source,
        annotation,
        release_track="formal_training",
        diagnosis=diagnosis,
        adjudication=adjudication,
    )
    assert len(views) == 1
    paths = {
        "source": tmp_path / "sources.jsonl",
        "annotation": tmp_path / "annotations.jsonl",
        "diagnosis": tmp_path / "diagnoses.jsonl",
        "adjudication": tmp_path / "adjudications.jsonl",
        "view": tmp_path / "views.jsonl",
    }
    payloads = {
        "source": source,
        "annotation": annotation,
        "diagnosis": diagnosis,
        "adjudication": adjudication,
        "view": views[0],
    }
    for name, model in payloads.items():
        paths[name].write_text(
            json.dumps(
                model.model_dump(mode="json", exclude_none=True)
            )
            + "\n",
            encoding="utf-8",
        )

    report = audit_reasoning_v4_artifacts(
        paths["source"],
        paths["annotation"],
        paths["view"],
        diagnosis_jsonl=paths["diagnosis"],
        adjudication_jsonl=paths["adjudication"],
        required_dataset_families=("sidewalk",),
    )

    assert report["errors"] == []
    assert report["passed"] is True
    assert report["metrics"] == {
        "sources": 1,
        "record_complete": 1,
        "all_sections_accepted": 1,
        "candidate_structurally_valid": 1,
        "pending_human": 0,
        "formal_training_eligible": 1,
        "task_canonicalization_drift": 0,
        "probe_tie_loss": 0,
        "preview_training_leak": 0,
        "formal_ineligible_artifact_leak": 0,
    }


def test_natural_pilot_audit_reports_rates_and_exact_formal_filter(
    tmp_path: Path,
) -> None:
    """Pilot audit accepts pending rates but gates seven probes and leakage."""

    source = _source("straight")
    groundings = _groundings()
    groundings = groundings.model_copy(
        update={
            "local_traversability": LocalTraversabilityV4(
                direction_affordances=[
                    DirectionAffordanceV4(
                        direction_label=label,
                        affordance="conditional",
                        clearance="sufficient",
                        limiting_factors=["dynamic_occupancy"],
                    )
                    for label in (
                        "left_90",
                        "left_30",
                        "left_10",
                        "straight",
                        "right_10",
                        "right_30",
                        "right_90",
                    )
                ]
            )
        }
    )
    proposal = DecisionProposalV4(
        selected_action_label="straight",
        speed="normal",
        response="proceed",
    )
    reconciliation = build_action_reconciliation_audit_v4(
        source,
        groundings,
        proposal,
    )
    composition = compose_reasoning_target_v4(
        source,
        groundings,
        proposal,
    )
    annotation = ReasoningAnnotationV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        record_status="complete",
        grounding_outputs=groundings,
        decision_proposal=proposal,
        action_reconciliation=reconciliation,
        composer_candidate=(
            composition.candidate_target.reasoning.decision
        ),
        candidate_target=composition.candidate_target,
        candidate_structure_record=CandidateStructureRecordV4(
            composer_version=REASONING_V4_COMPOSER_VERSION,
            status="valid",
        ),
        stage_records=_stage_records(),
    )
    views = build_reasoning_training_views_v4(
        source,
        annotation,
        release_track="formal_training",
    )
    assert len(views) == 1
    paths = {
        "source": tmp_path / "sources.jsonl",
        "annotation": tmp_path / "annotations.jsonl",
        "view": tmp_path / "views.jsonl",
    }
    payloads = {
        "source": source,
        "annotation": annotation,
        "view": views[0],
    }
    for name, model in payloads.items():
        paths[name].write_text(
            json.dumps(
                model.model_dump(mode="json", exclude_none=True)
            )
            + "\n",
            encoding="utf-8",
        )

    report = audit_reasoning_v4_pilot(
        paths["source"],
        paths["annotation"],
        paths["view"],
        expected_rows=1,
    )

    assert report["errors"] == []
    assert report["passed"] is True
    assert report["metrics"]["pending_human_rate"] == 0.0
    assert report["metrics"]["formal_view_rows"] == 1
    assert report["metrics"]["formal_manifest_leakage_rate"] == 0.0
    assert report["metrics"]["action_tuple_conflict_rate"] == 0.0
    assert (
        report["metrics"]["post_normalization_vocab_violation_rate"]
        == 0.0
    )
    assert report["metrics"]["probe_rows_exactly_seven"] == 1
    assert report["metrics"]["all_seven_probes_identical_rate"] == 1.0
    assert report["metrics"]["all_seven_probes_blocked_rate"] == 0.0
    assert report["metrics"]["all_seven_probes_conditional_rate"] == 1.0


def test_natural_pilot_audit_accepts_quarantined_stop_conflict(
    tmp_path: Path,
) -> None:
    """A known strong-moving/global-stop collision remains review-only."""

    source = _source("straight")
    groundings = _groundings(effect="stop").model_copy(
        update={
            "local_traversability": LocalTraversabilityV4(
                direction_affordances=[
                    DirectionAffordanceV4(
                        direction_label=label,
                        affordance="blocked",
                        clearance="insufficient",
                        limiting_factors=["dynamic_occupancy"],
                    )
                    for label in (
                        "left_90",
                        "left_30",
                        "left_10",
                        "straight",
                        "right_10",
                        "right_30",
                        "right_90",
                    )
                ]
            )
        }
    )
    proposal = DecisionProposalV4(
        selected_action_label="stop",
        speed="stop",
        response="stop",
    )
    reconciliation = build_action_reconciliation_audit_v4(
        source,
        groundings,
        proposal,
    )
    candidate_error = (
        "candidate_construction_failed:"
        "strong_moving_action_conflicts_with_stop_constraint"
    )
    annotation = ReasoningAnnotationV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        record_status="complete",
        grounding_outputs=groundings,
        decision_proposal=proposal,
        action_reconciliation=reconciliation,
        composer_candidate=None,
        candidate_target=None,
        candidate_structure_record=CandidateStructureRecordV4(
            composer_version=REASONING_V4_COMPOSER_VERSION,
            status="invalid",
            errors=[candidate_error],
        ),
        stage_records=_stage_records(),
        errors=[candidate_error],
    )
    state = derive_reasoning_release_state_v4(
        source,
        annotation,
    )
    assert state.review_status == "pending_human"
    assert state.formal_training_eligible is False

    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    formal_path = tmp_path / "formal.jsonl"
    source_path.write_text(
        json.dumps(
            source.model_dump(mode="json", exclude_none=True)
        )
        + "\n",
        encoding="utf-8",
    )
    annotation_path.write_text(
        json.dumps(
            annotation.model_dump(mode="json", exclude_none=True)
        )
        + "\n",
        encoding="utf-8",
    )
    formal_path.write_text("", encoding="utf-8")

    report = audit_reasoning_v4_pilot(
        source_path,
        annotation_path,
        formal_path,
        expected_rows=1,
    )

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["metrics"]["candidate_structurally_valid"] == 0
    assert (
        report["metrics"]["candidate_fail_closed_review_only"]
        == 1
    )
    assert report["metrics"]["candidate_unexpected_invalid_rows"] == 0
    assert report["metrics"]["pending_human"] == 1
    assert report["metrics"]["formal_training_eligible"] == 0
    assert report["metrics"]["formal_view_rows"] == 0
    assert report["metrics"]["constraint_effect_conflict_rows"] == 1
    assert report["metrics"]["action_tuple_conflict_rows"] == 0
    assert report["metrics"]["probe_rows_exactly_seven"] == 1


class _FakeBackend:
    def __init__(self, *, decision_label: str = "straight") -> None:
        self.requested_sections: list[str] = []
        self.image_details: dict[str, str] = {}
        self.image_roles: dict[str, tuple[str, ...]] = {}
        self.decision_label = decision_label

    async def generate(
        self,
        observation: NavObservationV4,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        del system_prompt
        payload = json.loads(question.split("\n\n", 1)[1])
        section = payload["requested_section"]
        self.requested_sections.append(section)
        self.image_details[section] = image_detail
        self.image_roles[section] = tuple(observation.image_roles)
        answers = {
            "SceneGrounding": {
                "scene_type": "urban_sidewalk",
                "walkable_support": "paved_sidewalk",
                "key_entities": [],
                "traffic_context": "none_visible",
            },
            "LocalTraversability": {
                "direction_affordances": [
                    {
                        "direction_label": "straight",
                        "affordance": "direct",
                        "clearance": "sufficient",
                        "limiting_factors": [],
                    }
                ]
            },
            "NavigationContextGrounding": {
                "task_direction": {
                    "status": "not_visible",
                    "source": "none",
                    "direction": None,
                }
            },
            "InteractionConstraintGrounding": {
                "constraints": []
            },
            "Decision": {
                "selected_action_label": self.decision_label,
                "speed": "normal",
                "response": "proceed",
            },
        }
        return TeacherGeneration(
            answer=json.dumps(answers[section]),
            provider="fake",
            model="fake-vlm",
            telemetry={"provider": "fake", "model": "fake-vlm"},
        )


def test_known_task_registry_skips_task_teacher_call() -> None:
    backend = _FakeBackend()
    annotation = asyncio.run(
        annotate_reasoning_source_v4(
            _source(
                warnings=["history_fpv_is_repeated_padding"],
            ),
            backend=backend,
            config=ReasoningV4BatchConfig(max_retries=0),
        )
    )

    assert annotation.record_status == "complete"
    assert annotation.all_sections_accepted is True
    assert annotation.candidate_structure_record.status == "valid"
    assert annotation.stage_records[
        "TaskGrounding"
    ].producer_kind == "deterministic"
    assert annotation.metadata["task_canonicalization_source"] == (
        "versioned_registry"
    )
    assert annotation.metadata["input_diagnostics"] == [
        "history_fpv_is_repeated_padding"
    ]
    assert "history_fpv_is_repeated_padding" not in annotation.warnings
    assert not any(
        "task_canonicalization_source" in warning
        for warning in annotation.warnings
    )
    assert "TaskGrounding" not in backend.requested_sections
    assert set(backend.requested_sections) == {
        "SceneGrounding",
        "LocalTraversability",
        "NavigationContextGrounding",
        "InteractionConstraintGrounding",
        "Decision",
    }
    assert backend.image_details["NavigationContextGrounding"] == "high"
    assert all(
        detail == "low"
        for section, detail in backend.image_details.items()
        if section != "NavigationContextGrounding"
    )


def test_batch_routes_exact_section_images_and_records_policy() -> None:
    """The runner, provider payload, and provenance use one routed policy."""

    base = _source()
    roles = [
        "fpv.history_4",
        "fpv.history_3",
        "fpv.history_2",
        "fpv.history_1",
        "fpv.current",
        "bev.current",
    ]
    observation_payload = base.observation.model_dump(mode="python")
    observation_payload.update(
        {
            "images": [f"/tmp/{role}.jpg" for role in roles],
            "image_roles": roles,
            "evidence_catalog": [
                {"ref": "instruction.active", "modality": "instruction"},
                *[
                    {"ref": role, "modality": "fpv"}
                    for role in roles
                    if role.startswith("fpv.")
                ],
                {
                    "ref": "bev.current",
                    "modality": "bev",
                    "bev_route_evidence": {
                        "frame": "north_up",
                        "ego_forward": "up",
                        "route_overlay_visible": True,
                        "route_direction_authority": (
                            "visual_conditional"
                        ),
                        "local_safety_authority": False,
                    },
                },
            ],
            "input_modalities": {
                "fpv": True,
                "history_fpv": True,
                "instruction": True,
                "bev": True,
                "goal_image": False,
                "pixel_goal": False,
                "waypoint": False,
            },
            "temporal_evidence_eligible": True,
        }
    )
    source_payload = base.model_dump(mode="python")
    source_payload["observation"] = NavObservationV4.model_validate(
        observation_payload
    ).model_dump(mode="python")
    source_payload.pop("source_fingerprint")
    source_payload["source_fingerprint"] = (
        reasoning_source_fingerprint_v4(source_payload)
    )
    source = ReasoningSourceV4.model_validate(source_payload)
    backend = _FakeBackend()

    annotation = asyncio.run(
        annotate_reasoning_source_v4(
            source,
            backend=backend,
            config=ReasoningV4BatchConfig(
                max_retries=0,
                section_image_policy_id=(
                    REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID
                ),
            ),
        )
    )

    assert annotation.record_status == "complete"
    assert backend.image_roles == {
        "SceneGrounding": tuple(roles[:5]),
        "LocalTraversability": ("fpv.current",),
        "NavigationContextGrounding": (
            "fpv.current",
            "bev.current",
        ),
        "InteractionConstraintGrounding": tuple(roles[:5]),
        "Decision": ("fpv.current", "bev.current"),
    }
    assert annotation.metadata["section_image_policy_id"] == (
        REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID
    )
    assert annotation.stage_records["Decision"].telemetry[
        "selected_image_roles"
    ] == ["fpv.current", "bev.current"]
    preview = build_reasoning_training_views_v4(
        source,
        annotation,
    )[0]
    assert preview.images == source.observation.images
    assert preview.image_roles == source.observation.image_roles
    assert preview.metadata["section_image_policy_id"] == (
        REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID
    )
    assert preview.metadata["section_image_policy_digest"] == (
        annotation.metadata["section_image_policy_digest"]
    )


class _FakeReviewBackend:
    """Return valid blind and comparative reviewer payloads."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def generate(
        self,
        observation: NavObservationV4,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Return one deterministic payload selected by review_stage."""

        del observation, system_prompt, image_detail
        self.questions.append(question)
        payload = json.loads(question.split("\n\n", 1)[1])
        if payload["review_stage"] == "blind_visual":
            answer = {
                "evidence_sufficiency": "sufficient",
                "visual_findings": ["visible_turn_support"],
                "ambiguous_evidence": [],
                "relevant_components": [
                    "NavigationContextGrounding"
                ],
                "evidence_analysis": "The visible route cue supports a turn.",
            }
        else:
            answer = {
                "verdict": "teacher_proposal_error",
                "error_subtype": "proposal_direction_error",
                "affected_components": ["DecisionProposal"],
                "component_assessment": {
                    "teacher_proposal": "likely_wrong",
                    "grounding": "likely_correct",
                    "strong_label": "likely_correct",
                },
                "reason_codes": [
                    "proposal_conflicts_with_visible_direction"
                ],
                "error_analysis": (
                    "The proposal conflicts with the visible route cue."
                ),
                "repair_action": "keep_without_change",
                "fields_to_inspect": [
                    "decision_proposal.selected_action_label"
                ],
                "confidence": 0.9,
            }
        return TeacherGeneration(
            answer=json.dumps(answer),
            provider="fake",
            model="fake-reviewer",
            telemetry={"provider": "fake", "model": "fake-reviewer"},
        )


def test_review_prompts_keep_blind_and_comparative_boundaries() -> None:
    """Blind input excludes private labels; comparison excludes candidate."""

    source = _source("left_40")
    annotation = _annotation(source, proposal_label="straight")
    blind_profile = REASONING_V4_REVIEW_PROFILES["blind_visual"]
    blind_question = build_reasoning_v4_blind_review_question(
        source.observation,
        blind_profile,
    )
    for forbidden in (
        "direction_supervision",
        "decision_proposal",
        "action_reconciliation",
        "trajectory_ego_t",
        "candidate_target",
    ):
        assert forbidden not in blind_question
    blind = ActionBlindReviewV4(
        evidence_sufficiency="sufficient",
        visual_findings=["visible_turn_support"],
        relevant_components=["NavigationContextGrounding"],
        evidence_analysis="The route cue supports a visible turn.",
    )
    comparative = build_reasoning_v4_comparative_review_question(
        source,
        annotation,
        blind,
        REASONING_V4_REVIEW_PROFILES["comparative_diagnosis"],
    )
    assert "strong_direction_stamp" in comparative
    assert "teacher_decision_proposal" in comparative
    assert "deterministic_action_audit" in comparative
    assert "trajectory_ego_t" not in comparative
    assert "candidate_target" not in comparative


def test_review_parser_rejects_corrected_values() -> None:
    """LVLM filtering may diagnose but cannot output corrections."""

    result = parse_reasoning_v4_review_answer(
        json.dumps(
            {
                "verdict": "strong_label_error",
                "corrected_label": "left_40",
            }
        ),
        REASONING_V4_REVIEW_PROFILES["comparative_diagnosis"],
    )
    assert result.valid is False
    assert result.errors[0].startswith(
        "corrected_or_suggested_values_forbidden"
    )


def test_two_stage_review_and_partial_repair_are_append_only() -> None:
    """Reviewer diagnoses first; repair creates a new audit lineage."""

    source = _source("left_40")
    original = asyncio.run(
        annotate_reasoning_source_v4(
            source,
            backend=_FakeBackend(decision_label="straight"),
            config=ReasoningV4BatchConfig(max_retries=0),
        )
    )
    review_backend = _FakeReviewBackend()
    diagnosis = asyncio.run(
        diagnose_reasoning_action_review_v4(
            source,
            original,
            backend=review_backend,
            config=ReasoningV4ReviewBatchConfig(max_retries=0),
        )
    )
    assert diagnosis.record_status == "complete"
    assert len(review_backend.questions) == 2
    assert "direction_supervision" not in review_backend.questions[0]
    adjudication = bind_action_adjudication_v4(
        source,
        original,
        diagnosis,
        ActionAdjudicationPayloadV4(
            diagnosis_disposition="overridden",
            final_verdict="teacher_proposal_error",
            error_subtype="proposal_direction_error",
            affected_components=["DecisionProposal"],
            reason_codes=[
                "proposal_conflicts_with_visible_direction"
            ],
            repair_action="rerun_teacher_proposal",
            fields_to_inspect=[
                "decision_proposal.selected_action_label"
            ],
        ),
        reviewer="reviewer-1",
        reviewed_at=datetime.now(timezone.utc),
    )
    request = build_reasoning_repair_request_v4(adjudication)
    assert request is not None
    repair_backend = _FakeBackend(decision_label="left_40")
    repaired = asyncio.run(
        repair_reasoning_annotation_v4(
            source,
            original,
            request,
            backend=repair_backend,
            config=ReasoningV4BatchConfig(max_retries=0),
        )
    )
    assert repair_backend.requested_sections == ["Decision"]
    assert repaired.action_reconciliation is not None
    assert original.action_reconciliation is not None
    assert (
        repaired.action_reconciliation.audit_id
        != original.action_reconciliation.audit_id
    )
    assert repaired.metadata["repair_request_id"] == (
        request.repair_request_id
    )


def test_review_html_exports_payload_consumed_by_canonical_importer(
    tmp_path: Path,
) -> None:
    """Self-contained review output binds through exact parent identities."""

    source = _source("left_40")
    annotation = _annotation(source, proposal_label="straight")
    diagnosis = _review_diagnosis(source, annotation)
    paths = {
        "source": tmp_path / "sources.jsonl",
        "annotation": tmp_path / "annotations.jsonl",
        "diagnosis": tmp_path / "diagnoses.jsonl",
        "input": tmp_path / "human.unbound.jsonl",
        "output": tmp_path / "adjudications.jsonl",
        "html": tmp_path / "review.complete.html",
        "audit": tmp_path / "audits.jsonl",
        "queue": tmp_path / "queue.jsonl",
        "report": tmp_path / "review-report.json",
    }
    for name, model in (
        ("source", source),
        ("annotation", annotation),
        ("diagnosis", diagnosis),
    ):
        paths[name].write_text(
            json.dumps(
                model.model_dump(mode="json", exclude_none=True)
            )
            + "\n",
            encoding="utf-8",
        )
    summary = build_reasoning_v4_review_html(
        source_jsonl=paths["source"],
        annotation_jsonl=paths["annotation"],
        diagnosis_jsonl=paths["diagnosis"],
        output_html=paths["html"],
        embed_images=False,
    )
    document = paths["html"].read_text(encoding="utf-8")
    assert summary["metrics"]["pending_human"] == 1
    assert "Export adjudications JSONL" in document
    assert diagnosis.diagnosis is not None
    assert diagnosis.diagnosis.diagnosis_id in document
    queue_report = export_action_reconciliation_review_v4(
        paths["source"],
        paths["annotation"],
        audit_jsonl=paths["audit"],
        queue_jsonl=paths["queue"],
        report_json=paths["report"],
        diagnosis_jsonl=paths["diagnosis"],
    )
    assert queue_report["passed"] is True
    queue_row = json.loads(
        paths["queue"].read_text(encoding="utf-8")
    )
    assert queue_row["diagnosis_status"] == "complete"
    assert queue_row["lvlm_diagnosis"]["diagnosis_id"] == (
        diagnosis.diagnosis.diagnosis_id
    )
    audit = annotation.action_reconciliation
    assert audit is not None
    paths["input"].write_text(
        json.dumps(
            {
                "source_group_id": source.observation.source_group_id,
                "audit_id": audit.audit_id,
                "diagnosis_id": diagnosis.diagnosis.diagnosis_id,
                "reviewer": "reviewer-1",
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "diagnosis_disposition": "confirmed",
                "final_verdict": "teacher_proposal_error",
                "error_subtype": "proposal_direction_error",
                "affected_components": ["DecisionProposal"],
                "reason_codes": [
                    "proposal_conflicts_with_visible_direction"
                ],
                "repair_action": "keep_without_change",
                "fields_to_inspect": [
                    "decision_proposal.selected_action_label"
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = import_reasoning_v4_adjudications(
        source_jsonl=paths["source"],
        annotation_jsonl=paths["annotation"],
        diagnosis_jsonl=paths["diagnosis"],
        input_jsonl=paths["input"],
        output_jsonl=paths["output"],
    )
    assert report["appended_rows"] == 1
    imported = json.loads(
        paths["output"].read_text(encoding="utf-8")
    )
    assert imported["diagnosis_id"] == (
        diagnosis.diagnosis.diagnosis_id
    )
    assert imported["source_fingerprint"] == source.source_fingerprint


def test_review_batch_is_resumable_by_audit_identity(
    tmp_path: Path,
) -> None:
    """Completed diagnoses are appended once and reused on resume."""

    source = _source("left_40")
    annotation = _annotation(source, proposal_label="straight")
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    diagnosis_path = tmp_path / "diagnoses.jsonl"
    source_path.write_text(
        json.dumps(source.model_dump(mode="json", exclude_none=True))
        + "\n",
        encoding="utf-8",
    )
    annotation_path.write_text(
        json.dumps(
            annotation.model_dump(mode="json", exclude_none=True)
        )
        + "\n",
        encoding="utf-8",
    )
    created: list[_FakeReviewBackend] = []

    def backend_factory(
        worker_id: int,
        api_key: str | None,
    ) -> _FakeReviewBackend:
        """Create and retain one deterministic review backend."""

        del worker_id, api_key
        backend = _FakeReviewBackend()
        created.append(backend)
        return backend

    first = asyncio.run(
        label_reasoning_review_queue_v4(
            source_path,
            annotation_path,
            diagnosis_path,
            config=ReasoningV4ReviewBatchConfig(max_retries=0),
            backend_factory=backend_factory,
        )
    )
    assert first["passed"] is True
    assert first["counts"]["record_status:complete"] == 1
    assert len(created) == 1
    second = asyncio.run(
        label_reasoning_review_queue_v4(
            source_path,
            annotation_path,
            diagnosis_path,
            config=ReasoningV4ReviewBatchConfig(max_retries=0),
            backend_factory=backend_factory,
        )
    )
    assert second["passed"] is True
    assert second["pending_rows"] == 0
    assert second["counts"]["cached:complete"] == 1
    assert len(created) == 1


def test_v46_system_prompt_preserves_bev_route_authority_boundary() -> None:
    profile = get_prompt_profile_v4(
        "urbannav_reasoning_v4_6_compact_full_cot_sft_v1"
    )
    prompt = " ".join(profile.system_prompt.split())

    assert (
        "A prompt-visible BEV route overlay may support route direction "
        "and upcoming topology when its orientation is clear."
    ) in prompt
    assert (
        "The absence of local safety authority only prevents BEV from "
        "proving traversability, clearance, or collision safety."
    ) in prompt
    assert "Set task_direction.source=bev_route" in prompt
