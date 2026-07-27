from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_target_v4,
    render_reasoning_target_v4,
)
from mcp_server.agent.reasoning_v4_prompt_policy import (
    STUDENT_FULL_COT_PROFILE_V4,
)
from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_section_v4,
    validate_reasoning_target_v4,
)
from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    coarse_motion_for_direction_label,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    AnnotationStageRecordV4,
    DirectionOutputSpaceV4,
    DirectionSupervisionV4,
    GroundingOutputsV4,
    InputModalitiesV4,
    NavObservationV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
    TeacherActionConstraintV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
    DirectionAffordanceV4,
    GoalAlignmentV4,
    InteractionConstraintGroundingV4,
    InteractionConstraintV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    NavigationSourceAlignmentV4,
    ReasoningTargetV4,
    SceneEntityV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGoalV4,
    TaskGroundingV4,
)
from mcp_server.data.reasoning_v4_composer import (
    compose_reasoning_target_v4,
)
from mcp_server.data.reasoning_v4_audit import (
    _reasoning_v4_leakage_categories,
    audit_reasoning_v4_artifacts,
    audit_reasoning_v4_source_coverage,
)
from mcp_server.data.reasoning_v4_views import (
    build_reasoning_training_views_v4,
    build_reasoning_v4_user_prompt,
)
from mcp_server.data.reasoning_v4_source import (
    build_reasoning_source_v4,
)
from mcp_server.data.reasoning_source import (
    ReasoningSourceBuildConfig,
)
from mcp_server.teacher.reasoning_v4_profiles import (
    REASONING_V4_TEACHER_PROFILES,
)
from mcp_server.teacher.reasoning_v4_batch import (
    ReasoningV4BatchConfig,
    annotate_reasoning_source_v4,
)
from mcp_server.teacher.reasoning_v4_teacher import (
    build_reasoning_v4_section_question,
    parse_reasoning_v4_section_answer,
)
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration
from tools.model_processing.build_reasoning_v4_review_html import (
    build_reasoning_v4_review_html,
)


def _direction(label: str = "straight") -> DirectionSupervisionV4:
    labels = [
        f"left_{angle}" for angle in range(90, 0, -10)
    ] + [
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
        choice=(
            chr(ord("A") + index)
            if index < 19
            else "T"
        ),
        angle_deg=(
            None
            if label == "stop"
            else float((index - 9) * 10)
        ),
        source="fpv_trajectory_terminal_centroid",
        bin_mode="degree10_with_stop",
        derivation_version=(
            "pixel_trajectory_terminal_centroid_rule_v2"
        ),
    )


def _output_space() -> DirectionOutputSpaceV4:
    return DirectionOutputSpaceV4(
        profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
        profile_version="2.0",
        allowed_labels=[
            f"left_{angle}" for angle in range(90, 0, -10)
        ]
        + [
            "straight",
            *[
                f"right_{angle}"
                for angle in range(10, 100, 10)
            ],
            "stop",
        ],
    )


def _source(label: str = "straight") -> ReasoningSourceV4:
    observation = NavObservationV4(
        source_group_id="sidewalk:case-1",
        dataset_source="sidewalk",
        source_record_id="case-1",
        split_group_id="clip-1",
        images=["frame.jpg"],
        image_roles=["fpv.current"],
        instruction="Continue on the sidewalk.",
        evidence_catalog=[
            {
                "ref": "instruction.active",
                "modality": "instruction",
            },
            {"ref": "fpv.current", "modality": "fpv"},
        ],
        input_modalities=InputModalitiesV4(instruction=True),
        direction_output_space=_output_space(),
    )
    payload = {
        "schema_version": "urbannav_reasoning_source_v4_2",
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": _direction(label).model_dump(
            mode="json"
        ),
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": [],
    }
    payload["source_fingerprint"] = (
        reasoning_source_fingerprint_v4(payload)
    )
    return ReasoningSourceV4.model_validate(payload)


def _groundings(
    *,
    constraint: InteractionConstraintV4 | None = None,
) -> GroundingOutputsV4:
    return GroundingOutputsV4(
        task_grounding=TaskGroundingV4(
            objective="follow_instruction",
            goal=TaskGoalV4(
                goal_type="none",
                specification="absent",
            ),
        ),
        scene_grounding=SceneGroundingV4(
            entities=[
                SceneEntityV4(
                    entity_ref="entity.e01",
                    category="delivery_cart",
                    evidence_status="supported",
                    observation_quality="clear",
                    lateral_position="right",
                    depth_position="mid",
                    evidence_refs=["fpv.current"],
                )
            ]
        ),
        local_traversability=LocalTraversabilityV4(
            direction_affordances=[
                DirectionAffordanceV4(
                    direction_label="straight",
                    support_type="visible_sidewalk",
                    affordance="direct",
                    clearance="sufficient",
                    evidence_status="supported",
                    evidence_refs=["fpv.current"],
                )
            ]
        ),
        navigation_context_grounding=(
            NavigationContextGroundingV4(
                source_alignments=[
                    NavigationSourceAlignmentV4(
                        alignment_ref="alignment.a01",
                        source_ref="instruction.active",
                        alignment_role="task_intent",
                        supported_direction_labels=["straight"],
                        status="supports",
                        evidence_refs=[
                            "instruction.active",
                            "fpv.current",
                        ],
                    )
                ],
                task_direction=TaskDirectionV4(
                    candidate_direction_labels=["straight"],
                    status="aligned",
                    evidence_refs=[
                        "instruction.active",
                        "fpv.current",
                    ],
                ),
            )
        ),
        interaction_constraint_grounding=(
            InteractionConstraintGroundingV4(
                constraints=(
                    [constraint] if constraint is not None else []
                )
            )
        ),
    )


def _proposal(
    *,
    decision_type: str = "navigate",
    navigation_intent_direction_label: str | None = "straight",
    selected_action_label: str = "straight",
    speed: str = "normal",
    social_action: str = "none",
    support_refs: list[str] | None = None,
) -> DecisionProposalV4:
    return DecisionProposalV4(
        decision_type=decision_type,
        navigation_intent_direction_label=(
            navigation_intent_direction_label
        ),
        selected_action_label=selected_action_label,
        speed=speed,
        social_action=social_action,
        fallback="stop",
        support_refs=(
            support_refs
            if support_refs is not None
            else ["nav.task_direction", "alignment.a01"]
        ),
        evidence_refs=["instruction.active", "fpv.current"],
    )


def _annotation(
    source: ReasoningSourceV4,
    target: ReasoningTargetV4,
    proposal: DecisionProposalV4,
) -> ReasoningAnnotationV4:
    records = {
        name: AnnotationStageRecordV4(
            status="accepted",
            attempts=1,
            prompt_profile_id=profile.profile_id,
            prompt_policy_digest=profile.policy_digest,
        )
        for name, profile in REASONING_V4_TEACHER_PROFILES.items()
    }
    return ReasoningAnnotationV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="accepted",
        grounding_outputs=_groundings(),
        decision_proposal=proposal,
        decision=target.reasoning.decision,
        target=target,
        stage_records=records,
    )


def test_compose_render_parse_and_validate_round_trip() -> None:
    source = _source()
    proposal = _proposal()
    composition = compose_reasoning_target_v4(
        source,
        _groundings(),
        proposal,
    )
    text = render_reasoning_target_v4(composition.target)
    parsed = parse_reasoning_target_v4(text)
    checked = validate_reasoning_target_v4(
        composition.target,
        observation=source.observation,
        assistant_text=text,
    )

    assert parsed.valid
    assert parsed.target == composition.target
    assert checked.valid
    assert (
        composition.target.finish.demonstrated_direction.model_dump()
        == {"label": "straight"}
    )


@pytest.mark.parametrize(
    ("label", "motion"),
    [
        ("left_10", "go_straight"),
        ("right_10", "go_straight"),
        ("left_20", "turn_left"),
        ("right_20", "turn_right"),
        ("stop", "stop"),
    ],
)
def test_profile_owns_direction_to_motion_mapping(
    label: str,
    motion: str,
) -> None:
    assert coarse_motion_for_direction_label(
        FPV_DEGREE10_TERMINAL_CENTROID_V2,
        label,
    ) == motion
    stop_constraint = (
        InteractionConstraintV4(
            constraint_ref="constraint.k01",
            kind="traffic",
            source_type="pedestrian_signal",
            source_state="red_hand",
            relation="controls_current_path",
            status="applicable",
            effect="stop",
            evidence_refs=["fpv.current"],
        )
        if label == "stop"
        else None
    )
    proposal = (
        _proposal(
            decision_type="stop",
            selected_action_label="stop",
            speed="stop",
            social_action="wait",
            support_refs=["nav.task_direction", "constraint.k01"],
        )
        if label == "stop"
        else _proposal(selected_action_label=label)
    )
    target = compose_reasoning_target_v4(
        _source(label),
        _groundings(constraint=stop_constraint),
        proposal,
    ).target
    assert target.reasoning.decision.motion == motion


def test_source_builder_exposes_profile_without_selected_label() -> None:
    direction = _direction("left_30")
    source = build_reasoning_source_v4(
        {
            "dataset_source": "sidewalk",
            "source_record_id": "builder-case",
            "images": ["frame.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Keep left.",
            "direction_supervision": direction.model_dump(
                mode="json"
            ),
        },
        index=0,
        config=ReasoningSourceBuildConfig(
            dataset_source="sidewalk",
            enable_system1=False,
        ),
    )
    visible = source.observation.model_dump(mode="json")

    assert source.direction_supervision == direction
    assert visible["direction_output_space"]["profile_id"] == (
        direction.profile_id
    )
    serialized_visible = json.dumps(visible)
    assert '"label_index"' not in serialized_visible
    assert '"selected_label"' not in serialized_visible


def test_source_builder_does_not_invent_frodobots_profile() -> None:
    with pytest.raises(
        ValueError,
        match="no approved V4.2 direction profile",
    ):
        build_reasoning_source_v4(
            {
                "dataset_source": "frodobots",
                "source_record_id": "frodobots-case",
                "images": ["frame.jpg"],
                "image_roles": ["fpv.current"],
            },
            index=0,
            config=ReasoningSourceBuildConfig(
                dataset_source="frodobots",
                enable_system1=False,
            ),
        )


def test_schema_contains_no_geometric_area_contract() -> None:
    schema_text = json.dumps(
        ReasoningTargetV4.model_json_schema(),
        sort_keys=True,
    ).lower()
    assert "region" not in schema_text


def test_open_scene_and_affordance_codes_are_not_fixed_choices() -> None:
    entity = SceneEntityV4(
        entity_ref="entity.e42",
        category="novel_folded_mobility_device",
        evidence_status="supported",
        evidence_refs=["fpv.current"],
    )
    affordance = DirectionAffordanceV4(
        direction_label="straight",
        support_type="novel_surface_support_code",
        affordance="direct",
        evidence_status="supported",
        evidence_refs=["fpv.current"],
    )
    assert entity.category == "novel_folded_mobility_device"
    assert affordance.support_type == "novel_surface_support_code"


def test_semantic_scene_refs_must_be_neutral() -> None:
    with pytest.raises(ValidationError):
        SceneEntityV4(
            entity_ref="entity.parked_suv",
            category="suv",
            evidence_status="supported",
            evidence_refs=["fpv.current"],
        )


def test_direction_outside_public_allowlist_is_rejected() -> None:
    section = LocalTraversabilityV4(
        direction_affordances=[
            DirectionAffordanceV4(
                direction_label="u_turn",
                support_type="visible_walkway",
                affordance="direct",
                evidence_status="supported",
                evidence_refs=["fpv.current"],
            )
        ]
    )
    checked = validate_reasoning_section_v4(
        "LocalTraversability",
        section,
        observation=_source().observation,
    )
    assert not checked.valid
    assert "direction_labels_outside_public_output_space" in (
        "\n".join(checked.errors)
    )


def test_grounding_prompt_has_public_space_but_no_private_target() -> None:
    source = _source("left_30")
    profile = REASONING_V4_TEACHER_PROFILES[
        "LocalTraversability"
    ]
    prompt = build_reasoning_v4_section_question(
        source.observation,
        profile,
    )

    assert "direction_output_space" in prompt
    assert "teacher_only_action_constraint" not in prompt
    assert "direction_supervision" not in prompt
    assert "label_index" not in prompt
    assert "trajectory_terminal_centroid" not in prompt


def test_system_prompts_preserve_bev_route_and_safety_authority_boundary() -> None:
    prompts = (
        REASONING_V4_TEACHER_PROFILES[
            "NavigationContextGrounding"
        ].system_prompt,
        STUDENT_FULL_COT_PROFILE_V4.system_prompt,
    )
    statement = (
        "A prompt-visible BEV route overlay may support route direction and "
        "upcoming topology when its orientation is clear. The absence of "
        "local safety authority only prevents BEV from proving "
        "traversability, clearance, or collision safety."
    )

    for system_prompt in prompts:
        assert statement in " ".join(system_prompt.split())


def test_teacher_prompts_encode_smoke_discovered_fail_closed_checks() -> None:
    navigation_prompt = REASONING_V4_TEACHER_PROFILES[
        "NavigationContextGrounding"
    ].system_prompt
    interaction_prompt = REASONING_V4_TEACHER_PROFILES[
        "InteractionConstraintGrounding"
    ].system_prompt
    decision_prompt = REASONING_V4_TEACHER_PROFILES[
        "Decision"
    ].system_prompt
    normalized_interaction_prompt = " ".join(
        interaction_prompt.split()
    )
    normalized_decision_prompt = " ".join(decision_prompt.split())

    assert "status=supports with" in navigation_prompt
    assert "supported_direction_labels=[] is invalid" in (
        navigation_prompt
    )
    assert 'The default output is {"constraints":[]}' in (
        interaction_prompt
    )
    assert "do not emit clear_walkable_path" in (
        normalized_interaction_prompt
    )
    assert "If any slow constraint applies, speed=normal is invalid" in (
        normalized_decision_prompt
    )
    assert "If any avoid constraint covers" in (
        normalized_decision_prompt
    )


def test_private_action_constraint_is_decision_teacher_only() -> None:
    source = _source("left_30")
    private = TeacherActionConstraintV4(
        demonstrated_direction_label="left_30"
    )
    decision_prompt = build_reasoning_v4_section_question(
        source.observation,
        REASONING_V4_TEACHER_PROFILES["Decision"],
        groundings=_groundings(),
        teacher_action_constraint=private,
    )
    target = compose_reasoning_target_v4(
        _source(),
        _groundings(),
        _proposal(),
    ).target
    full_view = next(
        view
        for view in build_reasoning_training_views_v4(
            _source(),
            _annotation(_source(), target, _proposal()),
        )
        if view.view_type == "full_cot_sft"
    )
    student_prompt = build_reasoning_v4_user_prompt(full_view)

    assert "teacher_only_action_constraint" in decision_prompt
    assert "left_30" in decision_prompt
    assert "teacher_only_action_constraint" not in student_prompt
    assert "direction_supervision" not in student_prompt


def test_private_action_constraint_must_use_public_direction() -> None:
    with pytest.raises(
        ValueError,
        match="outside the public direction output space",
    ):
        build_reasoning_v4_section_question(
            _source().observation,
            REASONING_V4_TEACHER_PROFILES["Decision"],
            groundings=_groundings(),
            teacher_action_constraint=TeacherActionConstraintV4(
                demonstrated_direction_label="u_turn"
            ),
        )


def test_decision_teacher_schema_has_no_motion() -> None:
    schema = REASONING_V4_TEACHER_PROFILES[
        "Decision"
    ].output_json_schema
    assert "motion" not in schema["properties"]
    assert "navigation_intent_direction_label" in schema["properties"]
    assert "selected_action_label" in schema["properties"]


def test_numeric_confidence_and_cost_are_rejected() -> None:
    profile = REASONING_V4_TEACHER_PROFILES["Decision"]
    payload = _proposal().model_dump(mode="json")
    payload["confidence"] = 0.8
    parsed = parse_reasoning_v4_section_answer(
        json.dumps(payload),
        profile,
    )
    assert not parsed.valid
    assert "numeric_confidence_or_cost_forbidden" in (
        "\n".join(parsed.errors)
    )


@pytest.mark.parametrize(
    "source_ref",
    ["goal.pixel_01", "waypoint.next_01"],
)
def test_unfrozen_navigation_geometry_cannot_assert_direction(
    source_ref: str,
) -> None:
    with pytest.raises(ValidationError):
        NavigationSourceAlignmentV4(
            alignment_ref="alignment.a01",
            source_ref=source_ref,
            alignment_role="route_intent",
            supported_direction_labels=["straight"],
            status="supports",
            evidence_refs=[source_ref],
        )


def test_clear_bev_route_overlay_may_support_route_direction() -> None:
    alignment = NavigationSourceAlignmentV4(
        alignment_ref="alignment.a01",
        source_ref="bev.route",
        alignment_role="route_intent",
        supported_direction_labels=["straight"],
        status="supports",
        evidence_refs=["bev.route"],
    )

    assert alignment.supported_direction_labels == ["straight"]


def test_unfrozen_navigation_source_must_be_insufficient() -> None:
    alignment = NavigationSourceAlignmentV4(
        alignment_ref="alignment.a01",
        source_ref="waypoint.next_01",
        alignment_role="route_intent",
        status="insufficient",
        evidence_refs=["waypoint.next_01"],
    )
    assert alignment.supported_direction_labels == []

    with pytest.raises(
        ValidationError,
        match="must be insufficient",
    ):
        NavigationSourceAlignmentV4(
            alignment_ref="alignment.a01",
            source_ref="waypoint.next_01",
            alignment_role="route_intent",
            status="conflicts",
            evidence_refs=["waypoint.next_01"],
        )


def test_unfrozen_goal_geometry_cannot_assert_direction() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot support goal direction labels",
    ):
        GoalAlignmentV4(
            status="resolved",
            target_description="visible point prompt",
            supported_direction_labels=["straight"],
            evidence_refs=["goal.pixel_01", "fpv.current"],
        )


def test_task_direction_cannot_bypass_source_alignment_gate() -> None:
    with pytest.raises(
        ValidationError,
        match="same-section source support",
    ):
        NavigationContextGroundingV4(
            source_alignments=[
                NavigationSourceAlignmentV4(
                    alignment_ref="alignment.a01",
                    source_ref="waypoint.next_01",
                    alignment_role="route_intent",
                    status="insufficient",
                    evidence_refs=["waypoint.next_01"],
                )
            ],
            task_direction=TaskDirectionV4(
                candidate_direction_labels=["straight"],
                status="aligned",
                evidence_refs=["waypoint.next_01", "fpv.current"],
            ),
        )


def test_navigate_proposal_cannot_use_stop_speed() -> None:
    with pytest.raises(
        ValidationError,
        match="navigate decision proposal cannot use speed=stop",
    ):
        _proposal(speed="stop")


def test_constraint_effect_conflict_rejects_full_target() -> None:
    constraint = InteractionConstraintV4(
        constraint_ref="constraint.k01",
        kind="local_safety",
        source_type="visible_cart",
        source_state="blocking",
        affected_direction_labels=[],
        relation="blocks_candidate_path",
        status="applicable",
        effect="stop",
        evidence_refs=["fpv.current"],
    )
    with pytest.raises(
        ValueError,
        match="constraint_effect_conflicts_with_final_decision",
    ):
        compose_reasoning_target_v4(
            _source(),
            _groundings(constraint=constraint),
            _proposal(),
        )


def test_red_signal_stop_uses_no_direction_scope_or_label_expansion() -> None:
    constraint = InteractionConstraintV4(
        constraint_ref="constraint.k01",
        kind="traffic",
        source_type="pedestrian_signal",
        source_state="red_hand",
        relation="controls_current_path",
        status="applicable",
        effect="stop",
        evidence_refs=["fpv.current"],
    )
    proposal = _proposal(
        decision_type="stop",
        navigation_intent_direction_label="straight",
        selected_action_label="stop",
        speed="stop",
        social_action="wait",
        support_refs=["nav.task_direction", "constraint.k01"],
    )
    composition = compose_reasoning_target_v4(
        _source("left_10"),
        _groundings(constraint=constraint),
        proposal,
    )
    decision = composition.target.reasoning.decision

    assert constraint.affected_direction_labels == []
    assert decision.navigation_intent_direction_label == "straight"
    assert decision.selected_action_label == "stop"
    assert decision.motion == "stop"
    assert decision.speed == "stop"
    assert decision.social_action == "wait"
    assert decision.fallback == "stop"
    assert (
        composition.target.finish.demonstrated_direction.label
        == "left_10"
    )


def test_stop_constraint_rejects_direction_enumeration() -> None:
    with pytest.raises(
        ValidationError,
        match="stop constraint must not enumerate affected directions",
    ):
        InteractionConstraintV4(
            constraint_ref="constraint.k01",
            kind="traffic",
            source_type="pedestrian_signal",
            source_state="red_hand",
            affected_direction_labels=["straight"],
            relation="controls_current_path",
            status="applicable",
            effect="stop",
            evidence_refs=["fpv.current"],
        )


def test_avoid_constraint_requires_subset_directions() -> None:
    with pytest.raises(
        ValidationError,
        match="avoid constraint requires affected_direction_labels",
    ):
        InteractionConstraintV4(
            constraint_ref="constraint.k01",
            kind="social",
            source_type="cyclist",
            source_state="visible",
            relation="occupies_candidate_path",
            status="applicable",
            effect="avoid",
            evidence_refs=["fpv.current"],
        )


def test_subset_avoid_rejects_only_affected_selected_action() -> None:
    constraint = InteractionConstraintV4(
        constraint_ref="constraint.k01",
        kind="social",
        source_type="cyclist",
        source_state="visible",
        affected_direction_labels=["left_10", "left_20"],
        relation="occupies_candidate_path",
        status="applicable",
        effect="avoid",
        evidence_refs=["fpv.current"],
    )
    with pytest.raises(
        ValueError,
        match="constraint_effect_conflicts_with_final_decision",
    ):
        compose_reasoning_target_v4(
            _source("left_10"),
            _groundings(constraint=constraint),
            _proposal(selected_action_label="left_10"),
        )

    alternative = compose_reasoning_target_v4(
        _source("left_10"),
        _groundings(constraint=constraint),
        _proposal(selected_action_label="right_10"),
    )
    assert (
        alternative.target.reasoning.decision.selected_action_label
        == "right_10"
    )


def test_stop_without_visible_support_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="stop_action_lacks_visible_support",
    ):
        compose_reasoning_target_v4(
            _source("left_10"),
            _groundings(),
            _proposal(
                decision_type="stop",
                selected_action_label="stop",
                speed="stop",
                social_action="wait",
            ),
        )


def test_constraint_rejects_affected_direction_scope_field() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        InteractionConstraintV4.model_validate(
            {
                "constraint_ref": "constraint.k01",
                "kind": "traffic",
                "source_type": "pedestrian_signal",
                "source_state": "red_hand",
                "affected_direction_scope": "all_moving",
                "relation": "controls_current_path",
                "status": "applicable",
                "effect": "stop",
                "evidence_refs": ["fpv.current"],
            }
        )


def test_constraint_rejects_all_moving_label_expansion() -> None:
    constraint = InteractionConstraintV4(
        constraint_ref="constraint.k01",
        kind="local_safety",
        source_type="temporary_hazard",
        source_state="visible",
        affected_direction_labels=[
            label
            for label in _output_space().allowed_labels
            if label != "stop"
        ],
        relation="near_candidate_path",
        status="applicable",
        effect="slow",
        evidence_refs=["fpv.current"],
    )
    checked = validate_reasoning_section_v4(
        "InteractionConstraintGrounding",
        InteractionConstraintGroundingV4(constraints=[constraint]),
        observation=_source().observation,
    )

    assert (
        "global_constraint_must_not_enumerate_all_moving:"
        "constraint.k01"
    ) in checked.errors


def test_blocked_selected_direction_rejects_full_target() -> None:
    blocked = LocalTraversabilityV4(
        direction_affordances=[
            DirectionAffordanceV4(
                direction_label="straight",
                support_type="visible_sidewalk",
                affordance="blocked",
                clearance="insufficient",
                limiting_factor_codes=["visible_obstruction"],
                evidence_status="supported",
                evidence_refs=["fpv.current"],
            )
        ]
    )
    groundings = _groundings().model_copy(
        update={"local_traversability": blocked}
    )
    with pytest.raises(
        ValueError,
        match="weak_strong_grounding_conflict",
    ):
        compose_reasoning_target_v4(
            _source(),
            groundings,
            _proposal(),
        )


def test_demonstrated_direction_does_not_override_selected_action() -> None:
    composition = compose_reasoning_target_v4(
        _source("stop"),
        _groundings(),
        _proposal(),
    )

    assert composition.target.reasoning.decision.selected_action_label == (
        "straight"
    )
    assert composition.target.finish.demonstrated_direction.label == "stop"
    assert "demonstrated_navigation_intent_mismatch" in "\n".join(
        composition.warnings
    )


def test_views_keep_private_direction_on_full_view_only() -> None:
    source = _source()
    proposal = _proposal()
    target = compose_reasoning_target_v4(
        source,
        _groundings(),
        proposal,
    ).target
    views = build_reasoning_training_views_v4(
        source,
        _annotation(source, target, proposal),
    )

    assert {view.view_type for view in views} == {
        "grounding_section_sft",
        "decision_section_sft",
        "full_cot_sft",
    }
    for view in views:
        assert (
            view.direction_supervision is not None
        ) == (view.view_type == "full_cot_sft")
        assert (
            view.direction_output_space
            == source.observation.direction_output_space
        )


def test_training_view_contract_rejects_private_sidecar_drift() -> None:
    source = _source()
    proposal = _proposal()
    target = compose_reasoning_target_v4(
        source,
        _groundings(),
        proposal,
    ).target
    views = build_reasoning_training_views_v4(
        source,
        _annotation(source, target, proposal),
    )
    full = next(
        view for view in views if view.view_type == "full_cot_sft"
    )
    full_payload = full.model_dump(mode="json")
    full_payload["direction_supervision"] = None
    with pytest.raises(
        ValidationError,
        match="requires direction_supervision",
    ):
        ReasoningTrainingViewV4.model_validate(full_payload)

    section = next(
        view
        for view in views
        if view.view_type == "grounding_section_sft"
    )
    section_payload = section.model_dump(mode="json")
    section_payload["direction_supervision"] = (
        source.direction_supervision.model_dump(mode="json")
        if source.direction_supervision is not None
        else None
    )
    with pytest.raises(
        ValidationError,
        match="cannot carry private direction",
    ):
        ReasoningTrainingViewV4.model_validate(section_payload)


def test_synthetic_source_annotation_view_audit(
    tmp_path: Path,
) -> None:
    source = _source()
    proposal = _proposal()
    target = compose_reasoning_target_v4(
        source,
        _groundings(),
        proposal,
    ).target
    annotation = _annotation(source, target, proposal)
    views = build_reasoning_training_views_v4(
        source,
        annotation,
    )
    source_path = tmp_path / "source.jsonl"
    annotation_path = tmp_path / "annotation.jsonl"
    view_path = tmp_path / "views.jsonl"
    source_path.write_text(
        json.dumps(source.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    annotation_path.write_text(
        json.dumps(annotation.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    view_path.write_text(
        "".join(
            json.dumps(view.model_dump(mode="json")) + "\n"
            for view in views
        ),
        encoding="utf-8",
    )

    coverage = audit_reasoning_v4_source_coverage(
        [source_path],
        required_dataset_families=("sidewalk",),
        min_source_rows=1,
    )
    report = audit_reasoning_v4_artifacts(
        source_path,
        annotation_path,
        view_path,
        required_dataset_families=("sidewalk",),
    )

    assert coverage["passed"] is True
    assert report["passed"] is True
    assert report["counts"]["direction_affordance_count"] == 1
    assert all(
        item["rows"] == 0
        for item in report["metrics"][
            "accepted_teacher_answer_leakage"
        ].values()
    )


def test_synthetic_review_html_uses_v4_2_contract(
    tmp_path: Path,
) -> None:
    source = _source()
    proposal = _proposal()
    target = compose_reasoning_target_v4(
        source,
        _groundings(),
        proposal,
    ).target
    annotation = _annotation(source, target, proposal)
    views = build_reasoning_training_views_v4(
        source,
        annotation,
    )
    source_path = tmp_path / "source.jsonl"
    annotation_path = tmp_path / "annotation.jsonl"
    view_path = tmp_path / "views.jsonl"
    output_path = tmp_path / "reasoning_v4_2_review.html"
    source_path.write_text(
        json.dumps(source.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    annotation_path.write_text(
        json.dumps(annotation.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    view_path.write_text(
        "".join(
            json.dumps(view.model_dump(mode="json")) + "\n"
            for view in views
        ),
        encoding="utf-8",
    )

    report = build_reasoning_v4_review_html(
        source_jsonl=source_path,
        annotation_jsonl=annotation_path,
        views_jsonl=view_path,
        output_html=output_path,
        embed_images=False,
    )
    review_text = output_path.read_text(encoding="utf-8")

    assert report["schema_version"] == (
        "urbannav_reasoning_v4_2_review_report_v1"
    )
    assert report["annotation_status_counts"] == {
        "accepted": 1
    }
    assert report["rendered_views"] == len(views)
    assert "UrbanNav Reasoning V4.2 Review" in review_text
    assert "direction_affordances" in review_text


def test_audit_classifies_forbidden_teacher_claims() -> None:
    assert _reasoning_v4_leakage_categories(
        '{"region_ref":"region.r1"}'
    ) == {"semantic_region"}
    assert _reasoning_v4_leakage_categories(
        '{"confidence":0.78,"cost":0.2}'
    ) == {"numeric_confidence_or_cost"}
    assert _reasoning_v4_leakage_categories(
        '{"source_type":"route_theta_projection"}'
    ) == {"disabled_geometry"}


def test_synthetic_six_call_teacher_flow_is_leakage_bounded() -> None:
    groundings = _groundings()
    answers = {
        "TaskGrounding": groundings.task_grounding,
        "SceneGrounding": groundings.scene_grounding,
        "LocalTraversability": groundings.local_traversability,
        "NavigationContextGrounding": (
            groundings.navigation_context_grounding
        ),
        "InteractionConstraintGrounding": (
            groundings.interaction_constraint_grounding
        ),
        "Decision": _proposal(),
    }

    class FakeBackend:
        def __init__(self) -> None:
            self.questions: list[str] = []

        async def generate(
            self,
            observation,
            *,
            question: str,
            system_prompt: str,
            image_detail: str,
        ) -> TeacherGeneration:
            del observation, system_prompt, image_detail
            self.questions.append(question)
            section_name = next(
                name
                for name in answers
                if f'"requested_section":"{name}"' in question
            )
            value = answers[section_name]
            assert value is not None
            return TeacherGeneration(
                answer=json.dumps(
                    value.model_dump(mode="json"),
                ),
                provider="fake",
                model="fake-v4-1",
                telemetry={"roundtrip_ms": 1},
            )

    backend = FakeBackend()
    annotation = asyncio.run(
        annotate_reasoning_source_v4(
            _source(),
            backend=backend,
            config=ReasoningV4BatchConfig(max_retries=0),
        )
    )

    assert annotation.status == "accepted"
    grounding_questions = [
        question
        for question in backend.questions
        if '"requested_section":"Decision"' not in question
    ]
    assert len(grounding_questions) == 5
    assert all(
        "teacher_only_action_constraint" not in question
        for question in grounding_questions
    )
    decision_question = next(
        question
        for question in backend.questions
        if '"requested_section":"Decision"' in question
    )
    assert "teacher_only_action_constraint" in decision_question
