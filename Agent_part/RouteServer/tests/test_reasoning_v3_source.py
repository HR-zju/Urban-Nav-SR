from __future__ import annotations

import pytest

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from reasoning_v3_test_utils import (
    build_test_source,
    canonical_direction_record,
)


def test_v3_4_source_keeps_private_labels_out_of_model_visible_partition() -> None:
    source = build_test_source(
        extra_row={
            "model_visible_context": {
                "note": "Use the visible sidewalk.",
                "route_theta_deg": 35.0,
                "pose": [1.0, 2.0, 0.5],
                "gps": {"latitude": 1.0, "longitude": 2.0},
                "camera_calibration": {"fx": 500.0},
                "tracked_agents": [{"track_id": "person_1"}],
            },
            "route_points_ego_t": [[0.0, 0.0], [1.0, 1.0]],
            "goal_geometry": [1.0, 2.0],
        }
    )

    visible_text = str(source.model_input.model_visible_context).lower()
    for forbidden in (
        "route_theta",
        "latitude",
        "longitude",
        "camera_calibration",
        "track_id",
    ):
        assert forbidden not in visible_text
    assert source.model_input.route_points_ego_t is None
    assert source.model_input.goal_geometry is None
    assert source.model_input.teacher_private_context == {}
    assert source.teacher_supervision_context["fixed_labels"]["canonical_direction"] is not None
    assert source.teacher_supervision_context["source_boundary_filter"]["removed_counts"][
        "context_paths"
    ] > 0


@pytest.mark.parametrize(
    ("dataset_source", "profile_id", "label_index", "name", "choice", "angle"),
    [
        (
            "sidewalk",
            "fpv_degree10_terminal_centroid_v2",
            12,
            "right_30",
            "M",
            30.0,
        ),
        (
            "egowalk",
            "ego_degree10_12way_terminal_centroid_v2",
            4,
            "left_20",
            "E",
            -20.0,
        ),
    ],
)
def test_v3_4_source_requires_and_preserves_complete_direction_profile_stamp(
    dataset_source: str,
    profile_id: str,
    label_index: int,
    name: str,
    choice: str,
    angle: float,
) -> None:
    record = canonical_direction_record(
        profile_id=profile_id,
        label_index=label_index,
        name=name,
        choice=choice,
        angle_deg=angle,
    )
    source = build_test_source(
        dataset_source=dataset_source,
        extra_row={"canonical_direction": record},
    )

    assert source.teacher_supervision_context["fixed_labels"]["canonical_direction"] == record
    provenance = source.label_provenance["decision.direction"]
    assert provenance.source_kind == "deterministic_adapter"
    assert provenance.supervision == "strong"
    assert provenance.student_inferable is True
    assert provenance.loss_channel == "direction_choice"
    assert provenance.details == {
        "profile_id": record["profile_id"],
        "profile_version": record["profile_version"],
        "bin_mode": record["bin_mode"],
    }


def test_v3_4_direction_and_system1_are_independent_sidecar_channels() -> None:
    direction_only = build_test_source(include_trajectory=False)
    trajectory_only = build_test_source(include_direction=False)

    assert direction_only.model_input.system1_enabled is False
    assert set(direction_only.label_provenance) == {"decision.direction"}
    assert direction_only.label_provenance["decision.direction"].loss_channel == "direction_choice"

    assert trajectory_only.model_input.system1_enabled is True
    assert set(trajectory_only.label_provenance) == {"system1.trajectory"}
    system1 = trajectory_only.label_provenance["system1.trajectory"]
    assert system1.source_kind == "dataset_gt"
    assert system1.supervision == "strong"
    assert system1.student_inferable is True
    assert system1.loss_channel == "system1_action"
    assert trajectory_only.teacher_supervision_context["fixed_labels"][
        "canonical_direction"
    ] is None
    assert "canonical_direction_missing_observation_only" in trajectory_only.warnings
    assert not any(trajectory_only.section_loss_mask.model_dump(mode="python").values())


@pytest.mark.parametrize(
    ("dataset_source", "profile_id"),
    [
        ("sidewalk", "ego_degree10_12way_terminal_centroid_v2"),
        ("foxglove", "ego_degree10_12way_terminal_centroid_v2"),
        ("egowalk", "fpv_degree10_terminal_centroid_v2"),
    ],
)
def test_v3_4_rejects_cross_dataset_direction_profiles(
    dataset_source: str,
    profile_id: str,
) -> None:
    record = canonical_direction_record(
        profile_id=profile_id,
        label_index=4 if profile_id.startswith("ego_") else 12,
        name="left_20" if profile_id.startswith("ego_") else "right_30",
        choice="E" if profile_id.startswith("ego_") else "M",
        angle_deg=-20.0 if profile_id.startswith("ego_") else 30.0,
    )
    with pytest.raises(ValueError, match="not approved for dataset"):
        build_test_source(
            dataset_source=dataset_source,
            extra_row={"canonical_direction": record},
        )


def test_v3_4_never_derives_direction_from_trajectory_or_route_theta() -> None:
    source = build_test_source(
        include_direction=False,
        extra_row={
            "sidewalk_meta": {"route_theta_deg": 42.0},
            "model_visible_context": {"route_theta_deg": 42.0},
        },
    )
    assert source.teacher_supervision_context["fixed_labels"]["canonical_direction"] is None
    assert "route_theta" not in str(source.model_input.model_visible_context).lower()

    route_theta_record = canonical_direction_record()
    route_theta_record["source"] = "route_theta_fallback"
    with pytest.raises(ValueError, match="route_theta direction_source is forbidden"):
        build_test_source(extra_row={"canonical_direction": route_theta_record})


def test_v3_4_rejects_incomplete_or_unapproved_direction_profiles() -> None:
    incomplete = canonical_direction_record()
    incomplete.pop("profile_version")
    with pytest.raises(ValueError, match="incomplete"):
        build_test_source(extra_row={"canonical_direction": incomplete})

    unknown = canonical_direction_record()
    unknown["profile_id"] = "researcher_invented_profile"
    with pytest.raises(ValueError, match="not approved"):
        build_test_source(extra_row={"canonical_direction": unknown})

    inconsistent = canonical_direction_record()
    inconsistent["name"] = "left_30"
    with pytest.raises(ValueError, match="name does not match"):
        build_test_source(extra_row={"canonical_direction": inconsistent})


def test_v3_4_frodobots_disables_direction_and_continuous_system1_even_if_supplied() -> None:
    source = build_test_source(
        dataset_source="frodobots",
        include_direction=True,
        include_trajectory=True,
    )

    assert source.model_input.system1_enabled is False
    assert source.model_input.trajectory_ego_t is None
    assert source.teacher_supervision_context["fixed_labels"]["canonical_direction"] is None
    assert source.label_provenance == {}
    assert "frodobots_trajectory_ignored_by_policy" in source.warnings
    assert "frodobots_canonical_direction_ignored_by_policy" in source.warnings
    assert not any(source.section_loss_mask.model_dump(mode="python").values())


def test_v3_4_source_schema_and_fingerprint_are_frozen() -> None:
    source = build_test_source()
    assert source.schema_version == "urbannav_reasoning_source_v3_4"

    mutated = source.model_dump(mode="python")
    mutated["warnings"] = [*mutated["warnings"], "post_fingerprint_mutation"]
    with pytest.raises(ValueError, match="source_fingerprint"):
        ReasoningSourceV3.model_validate(mutated)

    legacy = source.model_dump(mode="python")
    legacy["schema_version"] = "urbannav_reasoning_source_v3_3"
    with pytest.raises(ValueError, match="urbannav_reasoning_source_v3_4"):
        ReasoningSourceV3.model_validate(legacy)


def test_v3_4_flags_target_derived_instruction_and_answer_projected_cot() -> None:
    source = build_test_source(
        include_direction=False,
        include_trajectory=False,
        extra_row={
            "instruction": "Turn right.",
            "instruction_meta": {
                "source": "rule",
                "rule_version": "instruction_rule_v1",
                "motion_source": "trajectory_ego_terminal_centroid_rule_v1",
            },
            "messages": [
                {"role": "user", "content": "choose"},
                {"role": "assistant", "content": "<think>label projected</think>"},
            ],
            "sidewalk_meta": {"direction_choice_mode": "choice_only"},
        },
    )

    assert source.teacher_supervision_context["p0_supervision_flags"] == [
        "answer_projected_direction_choice_cot_candidate",
        "target_derived_instruction_candidate",
    ]
