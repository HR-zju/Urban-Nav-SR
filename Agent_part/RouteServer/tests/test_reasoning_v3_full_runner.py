from __future__ import annotations

import json

import pytest

from tools.model_processing.run_reasoning_v3_full import (
    _dataset_family,
    _partition_round_robin,
    _safe_name,
    parse_args,
)


def test_v3_4_full_runner_maps_only_canonical_dataset_families() -> None:
    assert _dataset_family("sidewalk_main") == "sidewalk"
    assert _dataset_family("Foxglove-clean") == "foxglove"
    assert _dataset_family("frodobots") == "frodobots"
    assert _dataset_family("egowalk_part_b") == "egowalk"
    with pytest.raises(ValueError, match="unsupported"):
        _dataset_family("unknown")


def test_v3_4_full_runner_sanitizes_dataset_names() -> None:
    assert _safe_name("sidewalk extra/v1") == "sidewalk_extra_v1"
    with pytest.raises(ValueError, match="invalid dataset name"):
        _safe_name("///")


def test_v3_4_full_runner_partitions_jsonl_stably_without_loss(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"index": index}) + "\n" for index in range(10)),
        encoding="utf-8",
    )
    shards = [tmp_path / f"shard_{index}.jsonl" for index in range(4)]

    assert _partition_round_robin(source, shards) == [3, 3, 2, 2]
    values = [
        [json.loads(line)["index"] for line in path.read_text(encoding="utf-8").splitlines()]
        for path in shards
    ]
    assert values == [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]


def test_v3_4_full_runner_exposes_only_minimal_training_profiles(tmp_path) -> None:
    audit = parse_args(
        ["finalize", "--run-root", str(tmp_path), "--training-profile", "audit_only"]
    )
    strong = parse_args(
        [
            "finalize",
            "--run-root",
            str(tmp_path),
            "--training-profile",
            "strong_action_only",
        ]
    )

    assert audit.training_profile == "audit_only"
    assert strong.training_profile == "strong_action_only"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "finalize",
                "--run-root",
                str(tmp_path),
                "--training-profile",
                "legacy_mixed_teacher_distillation",
            ]
        )
