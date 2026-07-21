from __future__ import annotations

import pytest

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3


def test_v3_source_separates_model_input_from_teacher_truth() -> None:
    row = {
        "id": "sidewalk-1",
        "dataset_source": "sidewalk",
        "images": ["old.jpg", "current.jpg", "bev.jpg"],
        "image_roles": ["fpv.history_00", "fpv.current", "bev.current"],
        "instruction": "Turn right toward the next entrance.",
        "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, -1.0, -0.2]],
        "trajectory_fpv_t": [[0.5, 0.9], [0.8, 0.7]],
        "trajectory_fpv_coord_frame": "normalized_0_1",
        "trajectory_source": "calibrated_pose_projection",
    }

    source = build_reasoning_source_v3(
        row,
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )

    assert source.model_input.system1_enabled is True
    assert "trajectory_ego_t" not in source.model_input.model_visible_context
    fixed = source.teacher_supervision_context["fixed_labels"]
    assert fixed["demonstrated_direction"] == "right"
    # Instruction semantics are intentionally deferred to the LVLM teacher.
    assert fixed["condition_direction"] == "unknown"
    assert fixed["demonstrated_corridor_ref"] == "corridor.right"
    assert fixed["condition_source_directions"] == {}
    assert source.label_provenance["demonstrated_direction"].source == "gt_trajectory"
    assert "condition_direction" not in source.label_provenance
    assert source.label_provenance["demonstrated_corridor_ref"].source == "validated_fpv_projection"

    legacy_payload = source.model_dump(mode="json")
    legacy_payload["schema_version"] = "urbannav_reasoning_source_v3"
    with pytest.raises(ValueError, match="urbannav_reasoning_source_v3_3"):
        ReasoningSourceV3.model_validate(legacy_payload)


def test_v3_source_disables_frodobots_continuous_truth() -> None:
    row = {
        "id": "frodobots-1",
        "dataset_source": "frodobots",
        "images": ["current.jpg", "bev.jpg"],
        "image_roles": ["fpv.current", "bev.current"],
        "instruction": "Continue safe pedestrian navigation.",
        "trajectory_ego_t": [[0.0, 0.0, 0.0], [20.0, -10.0, -0.5]],
        "bev_direction": "turn_left",
    }

    source = build_reasoning_source_v3(
        row,
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="frodobots"),
    )

    assert source.model_input.system1_enabled is False
    assert source.model_input.trajectory_ego_t is None
    assert source.teacher_supervision_context["fixed_labels"]["demonstrated_direction"] == "unknown"
    assert source.teacher_supervision_context["fixed_labels"]["condition_direction"] == "left"
    assert source.label_provenance["condition_direction"].source == "bev_geometry"
    assert "frodobots_trajectory_ignored_by_policy" in source.warnings


def test_v3_source_authorizes_bev_context_after_interleaved_image_normalization() -> None:
    """Recovered SideWalk FPV/BEV pairs must not lose BEV section supervision."""

    images: list[str] = []
    for index in range(5):
        images.extend(
            [
                f"/dataset/clip/fpv/{index:06d}.jpg",
                f"/dataset/clip/bev/{index:06d}_bev.jpg",
            ]
        )
    source = build_reasoning_source_v3(
        {
            "id": "sidewalk-interleaved-bev",
            "dataset_source": "sidewalk",
            "images": images,
            "instruction": "Continue safe pedestrian navigation.",
            "sidewalk_meta": {"multiframe_pair_order": "fpv_bev_pairs"},
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )

    assert "bev" in source.model_input.available_modalities
    assert source.model_input.image_roles[-1] == "bev.current"
    assert source.section_loss_mask.navigation_context_grounding is True


def test_v3_source_requires_explicit_projection_qc_for_strong_corridor() -> None:
    source = build_reasoning_source_v3(
        {
            "id": "sidewalk-unverified-projection",
            "dataset_source": "sidewalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Continue straight.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [0.0, 2.0, 0.2]],
            "trajectory_fpv_t": [[0.5, 0.9], [0.8, 0.7]],
            "trajectory_fpv_coord_frame": "normalized_0_1",
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )

    assert source.teacher_supervision_context["fixed_labels"]["demonstrated_corridor_ref"] is None
    assert "demonstrated_corridor_ref" not in source.label_provenance
    assert "fpv_projection_ignored_without_explicit_qc" in source.warnings


def test_v3_source_does_not_derive_action_from_invalid_trajectory_contract() -> None:
    source = build_reasoning_source_v3(
        {
            "id": "sidewalk-invalid-trajectory",
            "dataset_source": "sidewalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Continue safely.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, 1.0, 0.2]],
            "trajectory_policy": {"system1_target_format": "unknown_coordinate_contract"},
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )

    assert source.model_input.system1_enabled is False
    assert source.teacher_supervision_context["fixed_labels"]["demonstrated_direction"] == "unknown"
    assert "demonstrated_direction" not in source.label_provenance
    assert source.section_loss_mask.decision is True


def test_v3_source_reads_nested_frodobots_bev_intent_and_theta_convention() -> None:
    base = {
        "dataset_source": "frodobots",
        "images": ["current.jpg", "bev.jpg"],
        "image_roles": ["fpv.current", "bev.current"],
        "instruction": "Continue safe pedestrian navigation.",
    }
    by_intent = build_reasoning_source_v3(
        {
            **base,
            "id": "frodobots-nested-intent",
            "sidewalk_meta": {"intent_gt": "turn_left", "route_theta_deg": -37.6},
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="frodobots"),
    )
    by_theta = build_reasoning_source_v3(
        {
            **base,
            "id": "frodobots-nested-theta",
            "sidewalk_meta": {"route_theta_deg": 31.0},
        },
        index=1,
        config=ReasoningSourceBuildConfig(dataset_source="frodobots"),
    )

    assert by_intent.teacher_supervision_context["fixed_labels"]["condition_direction"] == "left"
    assert by_intent.label_provenance["condition_direction"].details["source_directions"] == {"bev.current": "left"}
    assert by_theta.teacher_supervision_context["fixed_labels"]["condition_direction"] == "right"
    assert by_theta.label_provenance["condition_direction"].details["source_directions"] == {"bev.current": "right"}
