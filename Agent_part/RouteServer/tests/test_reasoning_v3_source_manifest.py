from __future__ import annotations

import json

import pytest

from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_source import build_reasoning_source_manifest_v3


def test_v3_source_manifest_reports_dataset_supervision_eligibility(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "reasoning_v3_source.jsonl"
    rows = [
        {
            "id": "sidewalk-1",
            "dataset_source": "sidewalk",
            "images": ["sidewalk-current.jpg", "sidewalk-bev.jpg"],
            "image_roles": ["fpv.current", "bev.current"],
            "instruction": "Continue straight.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, 0.1, 0.02]],
            "trajectory_fpv_t": [[0.5, 0.9], [0.51, 0.7]],
            "trajectory_fpv_coord_frame": "normalized_0_1",
        },
        {
            "id": "frodobots-1",
            "dataset_source": "frodobots",
            "images": ["frodo-current.jpg", "frodo-bev.jpg"],
            "image_roles": ["fpv.current", "bev.current"],
            "instruction": "Continue safe pedestrian navigation.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [20.0, -10.0, -0.5]],
            "bev_direction": "turn_left",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = build_reasoning_source_manifest_v3(
        input_path,
        output_path,
        config=ReasoningSourceBuildConfig(dataset_source="mixed"),
    )

    assert report["counts"]["input_rows"] == 2
    assert report["counts"]["written_rows"] == 2
    assert report["counts"]["system1_rows"] == 1
    assert report["counts"]["lm_only_rows"] == 1
    assert report["counts"]["dataset:sidewalk"] == 1
    assert report["counts"]["dataset:frodobots"] == 1
    assert report["counts"]["label:demonstrated_direction:strong"] == 1
    # Only FrodoBots BEV geometry is a deterministic context direction here.
    assert report["counts"]["label:condition_direction:strong"] == 1
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2
    assert written[0]["schema_version"] == "urbannav_reasoning_source_v3_3"
    assert "teacher_supervision_context" in written[0]
    assert "teacher_supervision_context" not in written[0]["model_input"]


def test_v3_source_manifest_resolves_relative_images_from_media_root(tmp_path) -> None:
    media_root = tmp_path / "dataset"
    fpv = media_root / "media" / "fpv.jpg"
    bev = media_root / "media" / "bev.jpg"
    fpv.parent.mkdir(parents=True)
    fpv.write_bytes(b"fpv")
    bev.write_bytes(b"bev")
    input_path = tmp_path / "relative.jsonl"
    output_path = tmp_path / "relative_v3.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "relative-1",
                "dataset_source": "sidewalk",
                "images": ["media/fpv.jpg", "media/bev.jpg"],
                "image_roles": ["fpv.current", "bev.current"],
                "instruction": "Continue straight.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    build_reasoning_source_manifest_v3(
        input_path,
        output_path,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk", media_root=media_root),
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["model_input"]["images"] == [str(fpv.resolve()), str(bev.resolve())]


def test_v3_source_manifest_rejects_missing_media_root_image(tmp_path) -> None:
    input_path = tmp_path / "missing.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "missing-1",
                "dataset_source": "egowalk",
                "images": ["media/missing.jpg"],
                "image_roles": ["fpv.current"],
                "instruction": "Continue straight.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not exist under media_root"):
        build_reasoning_source_manifest_v3(
            input_path,
            tmp_path / "missing_v3.jsonl",
            config=ReasoningSourceBuildConfig(dataset_source="egowalk", media_root=tmp_path),
        )
