"""Focused tests for the Reasoning V4 annotation orchestration inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_server.agent.reasoning_v4_prompt_policy import (
    STUDENT_FULL_COT_PROFILE_V4,
    STUDENT_FULL_COT_PROFILE_V4_REGISTRY_V2,
)
from mcp_server.contracts.direction_profile import (
    DIRECTION_PROFILE_REGISTRY_V1,
    DIRECTION_PROFILE_REGISTRY_V2,
    EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    EGO_DEGREE10_TERMINAL_CENTROID_V2,
    FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    approved_stage_a_direction_profiles,
    resolve_stage_a_direction_profile,
)
from mcp_server.data.legacy.reasoning_v2.source import ReasoningSourceBuildConfig
from mcp_server.data.canonical.reasoning_v4_source import build_reasoning_source_v4
from mcp_server.teacher.reasoning_v4_profiles import (
    get_reasoning_v4_teacher_profile,
)
from tools.model_processing.project_reasoning_v4_fpv_directions import (
    project_reasoning_v4_fpv_direction_row,
)
from tools.model_processing.stamp_reasoning_v4_directions import (
    stamp_reasoning_v4_direction_manifest,
)

RUNNER_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "model_processing"
    / "run_reasoning_v4_qwen122b_dual_annotation.sh"
)
FULL_QUEUE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "model_processing"
    / "run_reasoning_v4_full_annotation_queue.sh"
)


def _legacy_direction_row(
    *,
    row_id: str,
    label_index: int,
    label_name: str,
    choice: str,
    angle_deg: float | None,
) -> dict[str, object]:
    """Build one old flat direction row without a V4 profile stamp."""

    return {
        "id": row_id,
        "direction_choice_label": label_index,
        "direction_name": label_name,
        "direction_choice": choice,
        "direction_angle_deg": angle_deg,
        "direction_source": (
            "finish_stop_rule"
            if label_name == "stop"
            else "pixel_trajectory_terminal_centroid_rule_v2"
        ),
        "direction_bin_mode": "degree10_with_stop",
        "pixel_trajectory": [[0.5, 0.8], [0.5, 0.2]],
    }


def test_runner_separates_machine_pilot_gate_from_launch_approval() -> None:
    """Machine audit is inspectable before the latch authorizes full launch."""

    script = RUNNER_PATH.read_text(encoding="utf-8")
    gate_case = script.split("  gate-pilot)", 1)[1].split("    ;;", 1)[0]
    launch_case = script.split("  launch-full)", 1)[1].split("    ;;", 1)[0]

    assert "validate_pilot_gate" in gate_case
    assert "PILOT_APPROVED" not in gate_case
    assert launch_case.index("PILOT_APPROVED") < launch_case.index(
        "validate_pilot_gate"
    )
    assert "tmux_session_exists_exact()" in script
    assert "grep -Fqx --" in script
    assert "tmux has-session -t" not in script


def test_full_queue_uses_exact_tmux_session_names() -> None:
    """Queue and child names must not collide through tmux prefix matching."""

    script = FULL_QUEUE_PATH.read_text(encoding="utf-8")

    assert "tmux_session_exists_exact()" in script
    assert "grep -Fqx --" in script
    assert "tmux has-session -t" not in script


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small test JSONL manifest."""

    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _legacy_sidewalk_12way_row(
    *,
    label_index: int,
    label_name: str,
    raw_angle_deg: float | None,
) -> dict[str, object]:
    """Build one consistent legacy Sidewalk 12-way direction row."""

    return {
        "direction_choice_label": label_index,
        "direction_name": label_name,
        "direction_choice": chr(ord("A") + label_index),
        "direction_angle_deg": (
            None if label_name == "stop" else raw_angle_deg
        ),
        "direction_raw_angle_deg": raw_angle_deg,
        "direction_source": (
            "finish_stop_rule"
            if label_name == "stop"
            else "pixel_trajectory_terminal_centroid_rule_v2"
        ),
        "direction_bin_mode": "degree10_12way_with_stop",
    }


def test_sidewalk_12way_projection_uses_raw_angle_for_frozen_20way() -> None:
    """Collapsed 50-plus labels recover a precise target from raw angle."""

    projected = project_reasoning_v4_fpv_direction_row(
        _legacy_sidewalk_12way_row(
            label_index=10,
            label_name="left_50plus",
            raw_angle_deg=-67.2,
        )
    )

    assert projected["direction_choice_label"] == 2
    assert projected["direction_name"] == "left_70"
    assert projected["direction_choice"] == "C"
    assert projected["direction_angle_deg"] == -70.0
    assert projected["direction_raw_angle_deg"] == -67.2
    assert projected["direction_bin_mode"] == "degree10_with_stop"
    assert projected["direction_profile"] == (
        "fpv_degree10_terminal_centroid_v2"
    )
    assert projected["direction_profile_id"] == (
        "fpv_degree10_terminal_centroid_v2"
    )
    assert projected["direction_label_index"] == 2
    assert projected["direction_label_name"] == "left_70"
    assert len(projected["direction_candidates"]) == 20


def test_registry_v1_is_frozen_and_registry_v2_has_four_profiles() -> None:
    """The running V4.6 registry remains readable beside the new matrix."""

    assert set(
        approved_stage_a_direction_profiles(
            DIRECTION_PROFILE_REGISTRY_V1
        )
    ) == {
        FPV_DEGREE10_TERMINAL_CENTROID_V2,
        EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    }
    assert set(
        approved_stage_a_direction_profiles(
            DIRECTION_PROFILE_REGISTRY_V2
        )
    ) == {
        FPV_DEGREE10_TERMINAL_CENTROID_V2,
        FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
        EGO_DEGREE10_TERMINAL_CENTROID_V2,
        EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    }
    assert STUDENT_FULL_COT_PROFILE_V4.policy_digest == (
        "09462230ab1439971e39fe1ac957b1d5"
        "a33c00a6680335ce5fe90f6ef8eeff0c"
    )
    assert (
        STUDENT_FULL_COT_PROFILE_V4_REGISTRY_V2.profile_id
        == "urbannav_reasoning_v4_6_compact_full_cot_sft_v2"
    )
    assert (
        STUDENT_FULL_COT_PROFILE_V4_REGISTRY_V2.policy_digest
        != STUDENT_FULL_COT_PROFILE_V4.policy_digest
    )


def test_teacher_profiles_are_versioned_by_direction_registry() -> None:
    """Registry-v2 annotation cannot hit registry-v1 teacher caches."""

    legacy = get_reasoning_v4_teacher_profile(
        "Decision",
        direction_registry_version=DIRECTION_PROFILE_REGISTRY_V1,
    )
    expanded = get_reasoning_v4_teacher_profile(
        "Decision",
        direction_registry_version=DIRECTION_PROFILE_REGISTRY_V2,
    )

    assert legacy.profile_id.endswith("_v10")
    assert expanded.profile_id.endswith("_v11")
    assert legacy.policy_digest != expanded.policy_digest
    assert legacy.derivation_version.endswith("_v2")
    assert expanded.derivation_version.endswith("_v3")
    assert legacy.policy_digest == (
        "ddad6e12bfd1e33bc521774a2d6577ba"
        "6b31daba67213da6f784377da9b66440"
    )
    routed_legacy = get_reasoning_v4_teacher_profile(
        "Decision",
        section_image_policy_id=(
            "reasoning_v4_section_images_routed_visual_v1"
        ),
        direction_registry_version=DIRECTION_PROFILE_REGISTRY_V1,
    )
    assert routed_legacy.policy_digest == (
        "2c4a5ebafd7b042964b4586b897e06c1"
        "9b875ed40f0f47c59f255325b9fe7fb2"
    )


def test_each_registry_v2_profile_freezes_geometry_contract() -> None:
    """Every profile owns geometry, units, stop, and derivation semantics."""

    for profile_id in (
        FPV_DEGREE10_TERMINAL_CENTROID_V2,
        FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
        EGO_DEGREE10_TERMINAL_CENTROID_V2,
        EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    ):
        profile = resolve_stage_a_direction_profile(profile_id)
        assert profile["derivation_strategy"]
        assert profile["coordinate_frame"]
        assert profile["coordinate_unit"]
        assert profile["trajectory_fields"]
        assert profile["derivation_version"]
        assert profile["stop_threshold"] > 0
        assert profile["route_theta_fallback"] is False


def test_sidewalk_projection_is_consumable_by_v4_source_builder() -> None:
    """A projected row must be a complete canonical V4 flat stamp."""

    row = _legacy_sidewalk_12way_row(
        label_index=10,
        label_name="left_50plus",
        raw_angle_deg=-67.2,
    )
    row.update(
        {
            "id": "sidewalk-projected-source",
            "fpv_image_path": "/tmp/current.jpg",
        }
    )
    projected = project_reasoning_v4_fpv_direction_row(row)

    source = build_reasoning_source_v4(
        projected,
        index=0,
        config=ReasoningSourceBuildConfig(
            dataset_source="sidewalk",
            enable_system1=False,
        ),
    )

    assert source.direction_supervision is not None
    assert source.direction_supervision.label_index == 2
    assert source.direction_supervision.label_name == "left_70"
    assert source.direction_supervision.choice == "C"
    assert source.direction_supervision.angle_deg == -70.0


def test_sidewalk_12way_projection_preserves_stop_semantics() -> None:
    """Legacy stop maps to the frozen 20-way stop slot."""

    projected = project_reasoning_v4_fpv_direction_row(
        _legacy_sidewalk_12way_row(
            label_index=0,
            label_name="stop",
            raw_angle_deg=None,
        )
    )

    assert projected["direction_choice_label"] == 19
    assert projected["direction_name"] == "stop"
    assert projected["direction_choice"] == "T"
    assert projected["direction_angle_deg"] is None
    assert projected["direction_source"] == "finish_stop_rule"


def test_sidewalk_12way_projection_does_not_require_raw_angle() -> None:
    """A same-vocabulary projection preserves a moving 12-way GT record."""

    row = _legacy_sidewalk_12way_row(
        label_index=10,
        label_name="left_50plus",
        raw_angle_deg=None,
    )
    projected = project_reasoning_v4_fpv_direction_row(
        row,
        target_profile_id=FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    )

    assert projected["direction_name"] == "left_50plus"
    assert projected["direction_label_index"] == 10
    assert projected["direction_raw_angle_deg"] is None


def test_sidewalk_projection_rederives_moving_label_from_raw_geometry() -> None:
    """A legacy source name is admissible only after raw-angle validation."""

    row = _legacy_sidewalk_12way_row(
        label_index=5,
        label_name="right_20",
        raw_angle_deg=18.4,
    )
    row["direction_source"] = "existing_direction_name"

    projected = project_reasoning_v4_fpv_direction_row(
        row,
        target_profile_id=FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    )

    assert projected["direction_name"] == "right_20"
    assert projected["direction_source"] == (
        "pixel_trajectory_terminal_centroid_rule_v2"
    )


def test_sidewalk_12way_to_20way_requires_raw_angle() -> None:
    """A collapsed 50-plus class cannot be invented as an exact 20-way bin."""

    row = _legacy_sidewalk_12way_row(
        label_index=10,
        label_name="left_50plus",
        raw_angle_deg=None,
    )

    with pytest.raises(
        ValueError,
        match="direction_raw_angle_deg",
    ):
        project_reasoning_v4_fpv_direction_row(
            row,
            target_profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
        )


def test_sidewalk_12way_projection_rejects_inconsistent_source_label() -> None:
    """Projection fails closed when the stored 12-way tuple disagrees."""

    row = _legacy_sidewalk_12way_row(
        label_index=1,
        label_name="straight",
        raw_angle_deg=0.0,
    )
    row["direction_choice"] = "A"

    with pytest.raises(
        ValueError,
        match="source direction choice/index mismatch",
    ):
        project_reasoning_v4_fpv_direction_row(row)


def _projected_fpv_source(
    *,
    dataset_source: str,
    target_profile_id: str,
) -> dict[str, object]:
    """Build one explicit FPV-profile source without data-family defaults."""

    row = _legacy_sidewalk_12way_row(
        label_index=1,
        label_name="straight",
        raw_angle_deg=0.0,
    )
    row.update({
        "id": f"{dataset_source}-{target_profile_id}",
        "fpv_image_path": "/tmp/current.jpg",
    })
    return project_reasoning_v4_fpv_direction_row(
        row,
        dataset_source=dataset_source,
        target_profile_id=target_profile_id,
    )


def test_same_dataset_accepts_two_explicit_fpv_profiles() -> None:
    """Dataset identity no longer chooses a single output vocabulary."""

    sources = [
        build_reasoning_source_v4(
            _projected_fpv_source(
                dataset_source="sidewalk",
                target_profile_id=profile_id,
            ),
            index=index,
            config=ReasoningSourceBuildConfig(
                dataset_source="sidewalk",
                enable_system1=False,
            ),
        )
        for index, profile_id in enumerate(
            (
                FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
                FPV_DEGREE10_TERMINAL_CENTROID_V2,
            )
        )
    ]

    assert {
        source.direction_supervision.profile_id
        for source in sources
        if source.direction_supervision is not None
    } == {
        FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
        FPV_DEGREE10_TERMINAL_CENTROID_V2,
    }


def test_different_datasets_accept_the_same_explicit_profile() -> None:
    """A registered profile is selected per row, not per dataset family."""

    profile_id = FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2
    sources = [
        build_reasoning_source_v4(
            _projected_fpv_source(
                dataset_source=dataset_source,
                target_profile_id=profile_id,
            ),
            index=index,
            config=ReasoningSourceBuildConfig(
                dataset_source=dataset_source,
                enable_system1=False,
            ),
        )
        for index, dataset_source in enumerate(
            ("sidewalk", "foxglove")
        )
    ]

    assert {
        source.observation.dataset_source for source in sources
    } == {"sidewalk", "foxglove"}
    assert all(
        source.direction_supervision is not None
        and source.direction_supervision.profile_id == profile_id
        for source in sources
    )


def test_egowalk_accepts_explicit_12way_and_20way_profiles() -> None:
    """Both ego output vocabularies share the same frozen geometry producer."""

    specifications = (
        (
            EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
            1,
            "B",
            "degree10_12way_with_stop",
        ),
        (
            EGO_DEGREE10_TERMINAL_CENTROID_V2,
            9,
            "J",
            "degree10_with_stop",
        ),
    )
    sources = []
    for index, (
        profile_id,
        label_index,
        choice,
        bin_mode,
    ) in enumerate(specifications):
        row = {
            "id": f"egowalk-{index}",
            "fpv_image_path": "/tmp/current.jpg",
            "direction_profile_id": profile_id,
            "direction_profile_version": "2.0",
            "direction_label_index": label_index,
            "direction_label_name": "straight",
            "direction_choice": choice,
            "direction_angle_deg": 0.0,
            "direction_source": "trajectory_ego_t",
            "direction_bin_mode": bin_mode,
            "direction_derivation_version": (
                "ego_x_right_y_front_terminal_centroid_rule_v2"
            ),
        }
        sources.append(
            build_reasoning_source_v4(
                row,
                index=index,
                config=ReasoningSourceBuildConfig(
                    dataset_source="egowalk",
                    enable_system1=False,
                ),
            )
        )

    assert {
        source.direction_supervision.profile_id
        for source in sources
        if source.direction_supervision is not None
    } == {
        EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
        EGO_DEGREE10_TERMINAL_CENTROID_V2,
    }


def test_frodobots_remains_direction_gt_fail_closed() -> None:
    """Framework profile flexibility does not invent FrodoBots GT."""

    row = _projected_fpv_source(
        dataset_source="frodobots",
        target_profile_id=FPV_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    )

    with pytest.raises(
        ValueError,
        match="no approved Reasoning V4 direction GT",
    ):
        build_reasoning_source_v4(
            row,
            index=0,
            config=ReasoningSourceBuildConfig(
                dataset_source="frodobots",
                enable_system1=False,
            ),
        )


def test_direction_stamper_selects_four_coarse_smoke_buckets(
    tmp_path: Path,
) -> None:
    """The smoke set is stratified without reading or deriving trajectory."""

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "stamped.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(
        input_path,
        [
            _legacy_direction_row(
                row_id="straight",
                label_index=9,
                label_name="straight",
                choice="J",
                angle_deg=0.0,
            ),
            _legacy_direction_row(
                row_id="left",
                label_index=8,
                label_name="left_10",
                choice="I",
                angle_deg=-10.0,
            ),
            _legacy_direction_row(
                row_id="right",
                label_index=10,
                label_name="right_10",
                choice="K",
                angle_deg=10.0,
            ),
            _legacy_direction_row(
                row_id="stop",
                label_index=19,
                label_name="stop",
                choice="T",
                angle_deg=None,
            ),
        ],
    )

    report = stamp_reasoning_v4_direction_manifest(
        input_path,
        output_path,
        report_json=report_path,
        dataset_source="foxglove",
        selection="coarse_direction",
        limit=4,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == [
        "straight",
        "left",
        "right",
        "stop",
    ]
    assert all(row["dataset_source"] == "foxglove" for row in rows)
    assert all(
        row["direction_supervision"]["profile_id"]
        == "fpv_degree10_terminal_centroid_v2"
        for row in rows
    )
    assert all(
        row["direction_supervision"]["derivation_version"]
        == "pixel_trajectory_terminal_centroid_rule_v2"
        for row in rows
    )
    assert report["direction_rederived"] is False
    assert report["trajectory_read"] is False
    assert report["counts"]["output_rows"] == 4


def test_direction_stamper_scales_coarse_smoke_buckets_to_sixteen(
    tmp_path: Path,
) -> None:
    """A larger smoke remains deterministic and balanced across four buckets."""

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "stamped.jsonl"
    report_path = tmp_path / "report.json"
    specifications = (
        ("straight", 9, "straight", "J", 0.0),
        ("left", 8, "left_10", "I", -10.0),
        ("right", 10, "right_10", "K", 10.0),
        ("stop", 19, "stop", "T", None),
    )
    _write_jsonl(
        input_path,
        [
            _legacy_direction_row(
                row_id=f"{bucket}-{repeat}",
                label_index=label_index,
                label_name=label_name,
                choice=choice,
                angle_deg=angle_deg,
            )
            for repeat in range(4)
            for (
                bucket,
                label_index,
                label_name,
                choice,
                angle_deg,
            ) in specifications
        ],
    )

    report = stamp_reasoning_v4_direction_manifest(
        input_path,
        output_path,
        report_json=report_path,
        dataset_source="foxglove",
        selection="coarse_direction",
        limit=16,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == [
        *(f"straight-{index}" for index in range(4)),
        *(f"left-{index}" for index in range(4)),
        *(f"right-{index}" for index in range(4)),
        *(f"stop-{index}" for index in range(4)),
    ]
    assert report["counts"]["input_rows"] == 16
    assert report["counts"]["output_rows"] == 16


def test_direction_stamper_selects_seeded_interesting_unique_clips(
    tmp_path: Path,
) -> None:
    """Interesting smoke combines consensus and random cross-clip samples."""

    input_path = tmp_path / "input.jsonl"
    interesting_a = tmp_path / "interesting_a.jsonl"
    interesting_b = tmp_path / "interesting_b.jsonl"
    output_path = tmp_path / "stamped.jsonl"
    repeated_output_path = tmp_path / "stamped_repeated.jsonl"
    report_path = tmp_path / "report.json"
    repeated_report_path = tmp_path / "report_repeated.json"
    specifications = (
        ("straight", 9, "straight", "J", 0.0),
        ("left", 8, "left_10", "I", -10.0),
        ("right", 10, "right_10", "K", 10.0),
        ("stop", 19, "stop", "T", None),
    )
    rows: list[dict[str, object]] = []
    consensus_rows: list[dict[str, object]] = []
    for (
        bucket,
        label_index,
        label_name,
        choice,
        angle_deg,
    ) in specifications:
        for repeat in range(6):
            row = _legacy_direction_row(
                row_id=f"{bucket}-{repeat}",
                label_index=label_index,
                label_name=label_name,
                choice=choice,
                angle_deg=angle_deg,
            )
            row["clip_id"] = f"clip-{bucket}-{repeat}"
            rows.append(row)
            if repeat < 3:
                consensus_rows.append(
                    {
                        "id": row["id"],
                        "clip_id": row["clip_id"],
                        "behavior_filter": {
                            "label": 1,
                            "valid": True,
                        },
                    }
                )
    interesting_rows = [
        {
            "id": row["id"],
            "clip_id": row["clip_id"],
            "behavior_filter": {
                "label": 1,
                "valid": True,
            },
        }
        for row in rows
    ]
    _write_jsonl(input_path, rows)
    _write_jsonl(interesting_a, interesting_rows)
    _write_jsonl(interesting_b, consensus_rows)

    common_arguments = {
        "dataset_source": "foxglove",
        "selection": "interesting_diverse",
        "limit": 16,
        "interesting_jsonls": [interesting_a, interesting_b],
        "random_seed": 17,
        "consensus_min_votes": 2,
    }
    report = stamp_reasoning_v4_direction_manifest(
        input_path,
        output_path,
        report_json=report_path,
        **common_arguments,
    )
    stamp_reasoning_v4_direction_manifest(
        input_path,
        repeated_output_path,
        report_json=repeated_report_path,
        **common_arguments,
    )

    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    repeated_rows = [
        json.loads(line)
        for line in repeated_output_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(output_rows) == 16
    assert len({row["clip_id"] for row in output_rows}) == 16
    assert [row["id"] for row in output_rows] == [
        row["id"] for row in repeated_rows
    ]
    assert report["counts"]["bucket:straight"] == 4
    assert report["counts"]["bucket:left"] == 4
    assert report["counts"]["bucket:right"] == 4
    assert report["counts"]["bucket:stop"] == 4
    assert (
        report["counts"]["selection_group:interesting_consensus"] == 8
    )
    assert report["counts"]["selection_group:interesting_random"] == 8
    assert len(report["selected_samples"]) == 16
    assert len(report["interesting_manifests"]) == 2
    assert all(
        item["sha256"]
        for item in report["interesting_manifests"]
    )


def test_direction_stamper_selects_seeded_natural_random_rows(
    tmp_path: Path,
) -> None:
    """Natural pilot scans all rows and is seeded without stratification."""

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "stamped.jsonl"
    repeated_output_path = tmp_path / "stamped_repeated.jsonl"
    changed_seed_output_path = tmp_path / "stamped_changed_seed.jsonl"
    report_path = tmp_path / "report.json"
    repeated_report_path = tmp_path / "report_repeated.json"
    changed_seed_report_path = tmp_path / "report_changed_seed.json"
    rows = []
    for index in range(40):
        if index < 30:
            specification = (9, "straight", "J", 0.0)
        elif index < 34:
            specification = (8, "left_10", "I", -10.0)
        elif index < 38:
            specification = (10, "right_10", "K", 10.0)
        else:
            specification = (19, "stop", "T", None)
        row = _legacy_direction_row(
            row_id=f"row-{index:02d}",
            label_index=specification[0],
            label_name=specification[1],
            choice=specification[2],
            angle_deg=specification[3],
        )
        row["clip_id"] = f"clip-{index // 2:02d}"
        rows.append(row)
    _write_jsonl(input_path, rows)

    common_arguments = {
        "dataset_source": "foxglove",
        "selection": "random",
        "limit": 12,
    }
    report = stamp_reasoning_v4_direction_manifest(
        input_path,
        output_path,
        report_json=report_path,
        random_seed=17,
        **common_arguments,
    )
    stamp_reasoning_v4_direction_manifest(
        input_path,
        repeated_output_path,
        report_json=repeated_report_path,
        random_seed=17,
        **common_arguments,
    )
    stamp_reasoning_v4_direction_manifest(
        input_path,
        changed_seed_output_path,
        report_json=changed_seed_report_path,
        random_seed=18,
        **common_arguments,
    )

    selected_ids = [
        json.loads(line)["id"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    repeated_ids = [
        json.loads(line)["id"]
        for line in repeated_output_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    changed_seed_ids = [
        json.loads(line)["id"]
        for line in changed_seed_output_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(selected_ids) == 12
    assert selected_ids == repeated_ids
    assert selected_ids != changed_seed_ids
    assert selected_ids != [f"row-{index:02d}" for index in range(12)]
    assert report["selection"] == "random"
    assert report["random_seed"] == 17
    assert report["counts"]["input_rows"] == 40
    assert report["counts"]["valid_input_rows"] == 40
    assert report["counts"]["output_rows"] == 12
    assert report["counts"]["selection_group:natural_random"] == 12
    assert len(report["selected_samples"]) == 12
    assert report["interesting_manifests"] == []


def test_direction_stamper_rejects_existing_label_identity_mismatch(
    tmp_path: Path,
) -> None:
    """A wrong stored choice fails closed instead of being repaired."""

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "stamped.jsonl"
    report_path = tmp_path / "report.json"
    row = _legacy_direction_row(
        row_id="bad",
        label_index=9,
        label_name="straight",
        choice="K",
        angle_deg=0.0,
    )
    _write_jsonl(input_path, [row])

    with pytest.raises(ValueError, match="direction label identity mismatch"):
        stamp_reasoning_v4_direction_manifest(
            input_path,
            output_path,
            report_json=report_path,
            dataset_source="foxglove",
        )
    assert not output_path.exists()
    assert not report_path.exists()
