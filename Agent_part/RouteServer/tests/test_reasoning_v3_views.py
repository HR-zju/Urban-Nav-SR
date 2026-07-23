from __future__ import annotations

import pytest

from mcp_server.agent.reasoning_v3_prompt_policy import STUDENT_FULL_COT_PROFILE_V3
from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3
from mcp_server.data.reasoning_v3_views import (
    REASONING_V3_TRAINING_POLICY_VERSION,
    ReasoningViewBuildConfigV3,
    build_reasoning_training_views_v3,
)
from reasoning_v3_test_utils import accepted_annotation, build_test_source


def test_v3_4_audit_view_disables_all_losses_and_keeps_sidecars_separate() -> None:
    source = build_test_source()
    annotation = accepted_annotation(source)

    views = build_reasoning_training_views_v3(source, annotation)

    assert len(views) == 1
    view = views[0]
    assert view.task_type == "reasoning_v3_4_full_cot"
    assert view.loss_mask == {
        "lm": False,
        "direction_choice": False,
        "system1_action": False,
        "trajectory": False,
        "segmentation": False,
    }
    assert not any(view.section_loss_mask.model_dump(mode="python").values())
    assert view.direction_profile_id == "fpv_degree10_terminal_centroid_v2"
    assert view.direction_label_index == 12
    assert view.trajectory_ego_t == source.model_input.trajectory_ego_t
    assert view.metadata["direction_training_role"] == "audit_only"
    assert view.metadata["system1_training_role"] == "audit_only"
    assert view.metadata["reasoning_text_training_role"] == "audit_only"
    assert view.metadata["reasoning_training_policy_version"] == (
        REASONING_V3_TRAINING_POLICY_VERSION
    )
    assert view.prompt_profile_id == STUDENT_FULL_COT_PROFILE_V3.profile_id
    assert view.prompt_policy_digest == STUDENT_FULL_COT_PROFILE_V3.policy_digest


def test_v3_4_strong_action_profile_enables_only_independent_action_sidecars() -> None:
    source = build_test_source()
    annotation = accepted_annotation(source)

    view = build_reasoning_training_views_v3(
        source,
        annotation,
        config=ReasoningViewBuildConfigV3(training_profile="strong_action_only"),
    )[0]

    assert view.loss_mask == {
        "lm": False,
        "direction_choice": True,
        "system1_action": True,
        "trajectory": False,
        "segmentation": False,
    }
    assert not any(view.section_loss_mask.model_dump(mode="python").values())
    assert view.metadata["direction_training_role"] == "primary"
    assert view.metadata["system1_training_role"] == "primary"


def test_v3_4_strong_action_profile_does_not_couple_direction_to_system1() -> None:
    source = build_test_source(include_trajectory=False)
    annotation = accepted_annotation(source)

    view = build_reasoning_training_views_v3(
        source,
        annotation,
        config=ReasoningViewBuildConfigV3(training_profile="strong_action_only"),
    )[0]

    assert view.loss_mask["direction_choice"] is True
    assert view.loss_mask["system1_action"] is False
    assert view.direction_label_index == 12
    assert view.trajectory_ego_t is None


def test_v3_4_view_records_weak_section_availability_without_enabling_lm() -> None:
    source = build_test_source()
    annotation = accepted_annotation(source)
    assert annotation.section_loss_mask is not None

    view = build_reasoning_training_views_v3(source, annotation)[0]

    assert view.metadata["weak_section_availability"] == (
        annotation.section_loss_mask.model_dump(mode="json")
    )
    assert any(view.metadata["weak_section_availability"].values())
    assert view.loss_mask["lm"] is False
    assert not any(view.section_loss_mask.model_dump(mode="python").values())


def test_v3_4_observation_only_annotation_never_emits_training_view() -> None:
    source = build_test_source(include_direction=False)
    annotation = ReasoningAnnotationV3(
        source_group_id=source.model_input.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="observation_only",
        observation=accepted_annotation(build_test_source()).observation,
        attempts=1,
    )

    assert build_reasoning_training_views_v3(source, annotation) == []


def test_v3_4_shortcut_flag_is_audit_visible_but_blocks_strong_action_training() -> None:
    source = build_test_source(
        extra_row={
            "instruction": "Turn right.",
            "instruction_meta": {
                "source": "rule",
                "rule_version": "instruction_rule_v1",
                "motion_source": "trajectory_ego_terminal_centroid_rule_v1",
            },
        }
    )
    annotation = accepted_annotation(source)

    audit_views = build_reasoning_training_views_v3(source, annotation)
    strong_views = build_reasoning_training_views_v3(
        source,
        annotation,
        config=ReasoningViewBuildConfigV3(training_profile="strong_action_only"),
    )

    assert audit_views[0].metadata["p0_supervision_flags"] == [
        "target_derived_instruction_candidate"
    ]
    assert not any(audit_views[0].loss_mask.values())
    assert strong_views == []


def test_v3_4_removed_mixed_distillation_profiles_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported Reasoning V3.4 training profile"):
        ReasoningViewBuildConfigV3(  # type: ignore[arg-type]
            training_profile="legacy_mixed_teacher_distillation"
        )
    with pytest.raises(ValueError, match="section LM loss is always disabled"):
        ReasoningViewBuildConfigV3(require_complete_full_cot_supervision=True)
    with pytest.raises(ValueError, match="legacy mixed teacher distillation"):
        ReasoningViewBuildConfigV3(allow_flagged_legacy_sources=True)


def test_v3_4_export_rejects_weak_lm_provenance() -> None:
    source = build_test_source()
    annotation = accepted_annotation(source)
    weak = annotation.label_provenance["task_grounding"].model_copy(
        update={"loss_channel": "weak_lm"}
    )
    invalid = annotation.model_copy(
        update={
            "label_provenance": {
                **annotation.label_provenance,
                "task_grounding": weak,
            }
        }
    )

    with pytest.raises(ValueError, match="weak labels must be audit-only"):
        build_reasoning_training_views_v3(source, invalid)
