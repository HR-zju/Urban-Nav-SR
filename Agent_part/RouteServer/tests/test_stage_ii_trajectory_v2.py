from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mcp_server.data.stage_ii.raw_trajectory_v2 import StageIIRawTrajectoryErrorV2, StageIIRawTrajectoryReaderV2
from mcp_server.data.stage_ii.trajectory import build_stage_ii_trajectory_target_v2
from mcp_server.data.stage_ii.trajectory_v2_corpus import build_stage_ii_trajectory_corpus_v2
from tools.model_processing.build_stage_ii_trajectory_4s32 import parse_args


ROUTE_ROOT = Path(__file__).resolve().parents[1]


def _build_target(rows):
    return build_stage_ii_trajectory_target_v2(
        sample_id="sidewalk:test",
        dataset_source="sidewalk",
        clip_id="clip-test",
        frame_idx=0,
        raw_trajectory=rows,
        action_mask=[True] * len(rows),
        source_adapter_id="sidewalk_npy_pose_4s_v2",
        trajectory_source="sidewalk_frame_pose_odometry",
        allowed_trajectory_sources={"sidewalk_frame_pose_odometry"},
        raw_timestamp_unit="nanosecond",
        yaw_offset_deg=0.0,
    )


def _straight_rows(*, end: float = 4.0, speed: float = 1.0):
    count = int(round(end / 0.125)) + 1
    return [
        [speed * index * 0.125, 0.0, 0.0, index * 0.125]
        for index in range(count)
    ]


def _manifest_row(clip_id: str) -> dict:
    return {
        "id": f"sidewalk:{clip_id}:000000",
        "trajectory_policy": {
            "schema_version": "sidewalk_time_conditioned_policy_v1",
            "time_unit": "seconds",
            "trajectory_source": "calibrated_pose_projection",
        },
        "sidewalk_meta": {
            "clip_id": clip_id,
            "frame_idx": 0,
            "frame_name": "000000.jpg",
            "trajectory_source": "calibrated_pose_projection",
        },
    }


def test_trajectory_builder_cli_defaults_to_full_horizon_and_explicitly_allows_prefixes() -> None:
    base_args = [
        "--dataset-manifest",
        "sidewalk=/tmp/source.jsonl",
        "--raw-root",
        "sidewalk=/tmp/raw",
        "--identity-registry-json",
        "/tmp/registry.json",
        "--output-dir",
        "/tmp/output",
    ]

    assert parse_args(base_args).allow_partial_horizon is False
    assert parse_args([*base_args, "--allow-partial-horizon"]).allow_partial_horizon is True


def _quaternion_rows(yaw: float, count: int, *, order: str) -> np.ndarray:
    xyzw = np.tile(
        np.asarray([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]),
        (count, 1),
    )
    if order == "xyzw":
        return xyzw
    return xyzw[:, [3, 0, 1, 2]]


def _write_npy_clip(
    root: Path,
    clip_id: str,
    *,
    positions: np.ndarray,
    yaw: float,
    order: str,
    write_metadata: bool,
) -> Path:
    clip_dir = root / clip_id
    clip_dir.mkdir(parents=True)
    count = len(positions)
    np.save(clip_dir / "frame_positions.npy", positions)
    np.save(
        clip_dir / "frame_orientations.npy",
        _quaternion_rows(yaw, count, order=order),
    )
    np.save(
        clip_dir / "frame_times.npy",
        np.arange(count, dtype=np.int64) * 125_000_000,
    )
    if write_metadata:
        (clip_dir / "mcap_extract_meta.json").write_text(
            json.dumps({"frame_orientations_order": order}),
            encoding="utf-8",
        )
    return clip_dir


def test_v2_builds_complete_four_second_metric_target() -> None:
    result = _build_target(_straight_rows())

    assert result.accepted
    assert result.target is not None
    assert result.target.valid_step_count == 32
    assert result.target.action_mask == [True] * 32
    assert result.target.trajectory_ego_t[0] == pytest.approx([0.125, 0.0, 0.0])
    assert result.target.trajectory_ego_t[-1] == pytest.approx([4.0, 0.0, 0.0])
    assert result.target.policy.policy_id == (
        "urbannav_stage_ii_metric_flu_4s_32_speed_audit_v2_1"
    )
    assert result.target.policy.source_speed_qc_mode == "audit_only"
    assert result.qc.source_horizon_seconds == pytest.approx(4.0)


def test_v2_rejects_partial_horizon_and_invalid_current_origin() -> None:
    short = _build_target(_straight_rows(end=3.875))
    assert not short.accepted
    assert short.qc.reason_codes == ("trajectory_full_horizon_missing",)

    invalid_origin = _straight_rows()
    invalid_origin[0][0] = 0.1
    result = _build_target(invalid_origin)
    assert not result.accepted
    assert result.qc.reason_codes == ("trajectory_current_origin_pose_invalid",)


def test_v2_partial_policy_emits_terminal_hold_with_false_suffix_mask() -> None:
    rows = _straight_rows(end=3.875)
    result = build_stage_ii_trajectory_target_v2(
        sample_id="sidewalk:partial",
        dataset_source="sidewalk",
        clip_id="clip-partial",
        frame_idx=0,
        raw_trajectory=rows,
        action_mask=[True] * len(rows),
        source_adapter_id="sidewalk_npy_pose_4s_v2",
        trajectory_source="sidewalk_frame_pose_odometry",
        allowed_trajectory_sources={"sidewalk_frame_pose_odometry"},
        raw_timestamp_unit="nanosecond",
        yaw_offset_deg=0.0,
        full_horizon_required=False,
    )

    assert result.accepted
    assert result.target is not None
    assert result.target.valid_step_count == 31
    assert result.target.action_mask == [True] * 31 + [False]
    assert result.target.trajectory_ego_t[-1] == result.target.trajectory_ego_t[-2]
    assert result.target.policy.full_horizon_required is False
    assert result.qc.target_valid_step_count == 31


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        (
            [[0.0, 0.0, 0.0, 0.0]]
            + [[0.1 * time, 0.0, 0.0, time] for time in np.arange(0.625, 4.001, 0.125)],
            "trajectory_source_gap_exceeded",
        ),
        (
            [[6.1 * time, 0.0, 0.0, time] for time in np.arange(0.0, 4.001, 0.5)],
            "trajectory_source_step_exceeded",
        ),
        (
            [[0.0125 * index, 0.0, 0.625 * index, 0.125 * index] for index in range(33)],
            "trajectory_source_yaw_rate_exceeded",
        ),
    ],
)
def test_v2_rejects_raw_motion_anomalies(rows, reason: str) -> None:
    result = _build_target(rows)

    assert not result.accepted
    assert result.qc.reason_codes == (reason,)


def test_v2_keeps_high_speed_as_an_audited_sample() -> None:
    result = _build_target(_straight_rows(speed=8.0))

    assert result.accepted
    assert result.target is not None
    assert result.qc.max_source_speed_mps == pytest.approx(8.0)
    assert result.qc.reason_codes == (
        "trajectory_build_accepted",
        "trajectory_source_speed_audit_exceeded",
    )


def test_v2_interpolates_yaw_across_pi_on_the_shortest_arc() -> None:
    rows = []
    for index in range(33):
        time = index * 0.125
        if math.isclose(time, 1.125):
            continue
        if time <= 1.0:
            yaw = 3.1 * time
        else:
            yaw = -3.1
        rows.append([0.1 * time, 0.0, yaw, time])

    result = _build_target(rows)

    assert result.accepted
    assert result.target is not None
    assert abs(result.target.trajectory_ego_t[8][2]) > 3.13


@pytest.mark.parametrize(
    ("dataset", "yaw", "axis", "order", "write_metadata"),
    [
        ("sidewalk", 0.0, "x", "wxyz", False),
        ("sidewalk", math.pi / 2.0, "y", "xyzw", True),
        ("foxglove", 0.0, "y", "xyzw", True),
    ],
)
def test_raw_npy_adapters_preserve_forward_axis(
    tmp_path: Path,
    dataset: str,
    yaw: float,
    axis: str,
    order: str,
    write_metadata: bool,
) -> None:
    root = tmp_path / dataset
    count = 41
    positions = np.zeros((count, 3), dtype=np.float64)
    positions[:, 0 if axis == "x" else 1] = np.arange(count) * 0.125
    _write_npy_clip(
        root,
        "clip-forward",
        positions=positions,
        yaw=yaw,
        order=order,
        write_metadata=write_metadata,
    )
    reader = StageIIRawTrajectoryReaderV2({dataset: root})

    window = reader.extract(dataset, _manifest_row("clip-forward"))
    target = build_stage_ii_trajectory_target_v2(
        sample_id=f"{dataset}:forward",
        dataset_source=dataset,
        clip_id=window.clip_id,
        frame_idx=window.frame_idx,
        raw_trajectory=window.trajectory_ego_timed,
        action_mask=window.action_mask,
        source_adapter_id=window.source_adapter_id,
        trajectory_source=window.trajectory_source,
        allowed_trajectory_sources={window.trajectory_source},
        raw_timestamp_unit=window.raw_timestamp_unit,
        yaw_offset_deg=window.yaw_offset_deg,
    )

    assert target.accepted
    assert target.target is not None
    assert target.target.trajectory_ego_t[-1] == pytest.approx([4.0, 0.0, 0.0], abs=1e-6)


def test_raw_npy_adapter_exposes_source_prefix_for_partial_policy(tmp_path: Path) -> None:
    root = tmp_path / "sidewalk"
    positions = np.zeros((25, 3), dtype=np.float64)
    positions[:, 0] = np.arange(25) * 0.125
    _write_npy_clip(
        root,
        "clip-partial",
        positions=positions,
        yaw=0.0,
        order="wxyz",
        write_metadata=False,
    )
    reader = StageIIRawTrajectoryReaderV2({"sidewalk": root})

    with pytest.raises(StageIIRawTrajectoryErrorV2) as exc_info:
        reader.extract("sidewalk", _manifest_row("clip-partial"))
    assert exc_info.value.code == "trajectory_full_horizon_missing"

    window = reader.extract(
        "sidewalk",
        _manifest_row("clip-partial"),
        full_horizon_required=False,
    )
    assert window.trajectory_ego_timed[-1][-1] == pytest.approx(3.0)
    target = build_stage_ii_trajectory_target_v2(
        sample_id="sidewalk:partial",
        dataset_source="sidewalk",
        clip_id=window.clip_id,
        frame_idx=window.frame_idx,
        raw_trajectory=window.trajectory_ego_timed,
        action_mask=window.action_mask,
        source_adapter_id=window.source_adapter_id,
        trajectory_source=window.trajectory_source,
        allowed_trajectory_sources={window.trajectory_source},
        raw_timestamp_unit=window.raw_timestamp_unit,
        yaw_offset_deg=window.yaw_offset_deg,
        full_horizon_required=False,
    )
    assert target.accepted
    assert target.target is not None
    assert target.target.action_mask == [True] * 24 + [False] * 8


def test_raw_npy_adapter_rejects_static_export_and_bad_clock(tmp_path: Path) -> None:
    root = tmp_path / "sidewalk"
    clip = _write_npy_clip(
        root,
        "clip-static",
        positions=np.zeros((41, 3), dtype=np.float64),
        yaw=0.0,
        order="wxyz",
        write_metadata=False,
    )
    reader = StageIIRawTrajectoryReaderV2({"sidewalk": root})
    with pytest.raises(StageIIRawTrajectoryErrorV2) as exc_info:
        reader.extract("sidewalk", _manifest_row("clip-static"))
    assert exc_info.value.code == "trajectory_static_pose_clip"

    positions = np.zeros((41, 3), dtype=np.float64)
    positions[:, 0] = np.arange(41) * 0.125
    np.save(clip / "frame_positions.npy", positions)
    times = np.arange(41, dtype=np.int64) * 125_000_000
    times[2] = times[1]
    np.save(clip / "frame_times.npy", times)
    reader = StageIIRawTrajectoryReaderV2({"sidewalk": root})
    with pytest.raises(StageIIRawTrajectoryErrorV2) as exc_info:
        reader.extract("sidewalk", _manifest_row("clip-static"))
    assert exc_info.value.code == "trajectory_time_not_strictly_increasing"


def test_raw_npy_adapter_rejects_explicitly_invalid_pose_window(
    tmp_path: Path,
) -> None:
    root = tmp_path / "foxglove"
    positions = np.zeros((41, 3), dtype=np.float64)
    positions[:, 0] = np.arange(41) * 0.125
    clip = _write_npy_clip(
        root,
        "clip-missing-odom",
        positions=positions,
        yaw=0.0,
        order="xyzw",
        write_metadata=True,
    )
    pose_valid = np.ones(41, dtype=np.bool_)
    pose_valid[8] = False
    np.save(clip / "frame_pose_valid.npy", pose_valid)
    reader = StageIIRawTrajectoryReaderV2({"foxglove": root})

    with pytest.raises(StageIIRawTrajectoryErrorV2) as exc_info:
        reader.extract("foxglove", _manifest_row("clip-missing-odom"))

    assert exc_info.value.code == "trajectory_raw_pose_invalid"


def test_raw_npy_adapter_maps_corrupt_file_to_stable_rejection(tmp_path: Path) -> None:
    root = tmp_path / "sidewalk"
    positions = np.zeros((41, 3), dtype=np.float64)
    positions[:, 0] = np.arange(41) * 0.125
    clip = _write_npy_clip(
        root,
        "clip-corrupt",
        positions=positions,
        yaw=0.0,
        order="wxyz",
        write_metadata=False,
    )
    (clip / "frame_positions.npy").write_bytes(b"not-a-numpy-file")
    reader = StageIIRawTrajectoryReaderV2({"sidewalk": root})

    with pytest.raises(StageIIRawTrajectoryErrorV2) as exc_info:
        reader.extract("sidewalk", _manifest_row("clip-corrupt"))

    assert exc_info.value.code == "trajectory_raw_read_failed"


def test_v2_corpus_writes_portable_sidecar_qc_and_report(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    positions = np.zeros((41, 3), dtype=np.float64)
    positions[:, 0] = np.arange(41) * 1.0
    _write_npy_clip(
        raw_root,
        "clip-corpus",
        positions=positions,
        yaw=0.0,
        order="wxyz",
        write_metadata=False,
    )
    manifest = tmp_path / "sidewalk.jsonl"
    manifest.write_text(json.dumps(_manifest_row("clip-corpus")) + "\n", encoding="utf-8")

    report = build_stage_ii_trajectory_corpus_v2(
        dataset_manifests={"sidewalk": [manifest]},
        raw_roots={"sidewalk": raw_root},
        identity_registry_json=(
            ROUTE_ROOT / "config/stage_ii_cfm/trajectory_source_registry_v1.json"
        ),
        output_dir=tmp_path / "output",
    )

    assert report["complete"] is True
    assert report["counts"]["accepted"] == 1
    assert report["counts"]["rejected_total"] == 0
    assert report["counts"]["audit:trajectory_source_speed_audit_exceeded"] == 1
    assert report["per_manifest"]["sidewalk/sidewalk.jsonl"]["accepted"] == 1
    sidecars = []
    for split in ("train", "validation", "test"):
        path = tmp_path / "output" / f"trajectory_sidecars_v2_{split}.jsonl"
        sidecars.extend(json.loads(line) for line in path.read_text().splitlines())
    assert len(sidecars) == 1
    assert sidecars[0]["source_manifest_name"] == "sidewalk/sidewalk.jsonl"
    assert len(sidecars[0]["trajectory_target"]["trajectory_ego_t"]) == 32
    assert sidecars[0]["trajectory_qc"]["reason_codes"] == [
        "trajectory_build_accepted",
        "trajectory_source_speed_audit_exceeded",
    ]
    assert str(raw_root) not in json.dumps(sidecars[0])


def test_v2_corpus_propagates_partial_horizon_policy(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    positions = np.zeros((25, 3), dtype=np.float64)
    positions[:, 0] = np.arange(25) * 0.125
    _write_npy_clip(
        raw_root,
        "clip-corpus-partial",
        positions=positions,
        yaw=0.0,
        order="wxyz",
        write_metadata=False,
    )
    manifest = tmp_path / "sidewalk.jsonl"
    manifest.write_text(
        json.dumps(_manifest_row("clip-corpus-partial")) + "\n",
        encoding="utf-8",
    )

    report = build_stage_ii_trajectory_corpus_v2(
        dataset_manifests={"sidewalk": [manifest]},
        raw_roots={"sidewalk": raw_root},
        identity_registry_json=(
            ROUTE_ROOT / "config/stage_ii_cfm/trajectory_source_registry_v1.json"
        ),
        output_dir=tmp_path / "output",
        full_horizon_required=False,
    )

    assert report["complete"] is True
    assert report["counts"]["accepted"] == 1
    sidecars = []
    for split in ("train", "validation", "test"):
        path = tmp_path / "output" / f"trajectory_sidecars_v2_{split}.jsonl"
        sidecars.extend(json.loads(line) for line in path.read_text().splitlines())
    assert len(sidecars) == 1
    target = sidecars[0]["trajectory_target"]
    assert target["action_mask"] == [True] * 24 + [False] * 8
    assert target["policy"]["full_horizon_required"] is False
