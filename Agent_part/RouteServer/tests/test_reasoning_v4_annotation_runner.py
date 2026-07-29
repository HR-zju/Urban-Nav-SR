"""Focused tests for the Reasoning V4 annotation orchestration inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.model_processing.stamp_reasoning_v4_directions import (
    stamp_reasoning_v4_direction_manifest,
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small test JSONL manifest."""

    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
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
