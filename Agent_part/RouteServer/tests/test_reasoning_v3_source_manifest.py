from __future__ import annotations

import json

import pytest

from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_source import build_reasoning_source_manifest_v3
from reasoning_v3_test_utils import canonical_direction_record


def test_v3_4_source_manifest_reports_direction_and_system1_independently(
    tmp_path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "sources.jsonl"
    rows = [
        {
            "id": "sidewalk-direction-and-system1",
            "dataset_source": "sidewalk",
            "images": ["sidewalk.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Turn right.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, -1.0, -0.2]],
            "canonical_direction": canonical_direction_record(),
        },
        {
            "id": "frodobots-observation-only",
            "dataset_source": "frodobots",
            "images": ["frodobots.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Continue safely.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [20.0, -10.0, -0.5]],
            "canonical_direction": canonical_direction_record(),
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

    counts = report["counts"]
    assert counts["input_rows"] == counts["written_rows"] == 2
    assert counts["system1_rows"] == 1
    assert counts["non_system1_rows"] == 1
    assert counts["canonical_direction_rows"] == 1
    assert counts["observation_only_rows"] == 1
    assert counts["label:decision.direction:strong"] == 1
    assert counts["label:system1.trajectory:strong"] == 1
    written = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["schema_version"] == "urbannav_reasoning_source_v3_4" for row in written)
    assert all(not any(row["section_loss_mask"].values()) for row in written)


def test_v3_4_source_manifest_skip_errors_records_invalid_direction_profile(
    tmp_path,
) -> None:
    input_path = tmp_path / "invalid.jsonl"
    output_path = tmp_path / "valid.jsonl"
    invalid = canonical_direction_record()
    invalid["profile_version"] = "999"
    input_path.write_text(
        json.dumps(
            {
                "id": "invalid-direction-profile",
                "dataset_source": "sidewalk",
                "images": ["current.jpg"],
                "image_roles": ["fpv.current"],
                "instruction": "Continue.",
                "canonical_direction": invalid,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_reasoning_source_manifest_v3(
        input_path,
        output_path,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
        skip_errors=True,
    )

    assert report["counts"]["rejected_rows"] == 1
    assert "profile stamp mismatch" in report["errors"][0]["error"]
    assert output_path.read_text(encoding="utf-8") == ""


def test_v3_4_source_manifest_resolves_relative_images_from_media_root(
    tmp_path,
) -> None:
    media_root = tmp_path / "dataset"
    fpv = media_root / "media" / "fpv.jpg"
    fpv.parent.mkdir(parents=True)
    fpv.write_bytes(b"fpv")
    input_path = tmp_path / "relative.jsonl"
    output_path = tmp_path / "relative_v3.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "relative-1",
                "dataset_source": "sidewalk",
                "images": ["media/fpv.jpg"],
                "image_roles": ["fpv.current"],
                "instruction": "Continue straight.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    build_reasoning_source_manifest_v3(
        input_path,
        output_path,
        config=ReasoningSourceBuildConfig(
            dataset_source="sidewalk",
            media_root=media_root,
        ),
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["model_input"]["images"] == [str(fpv.resolve())]


def test_v3_4_source_manifest_rejects_missing_media_root_image(tmp_path) -> None:
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
            config=ReasoningSourceBuildConfig(
                dataset_source="egowalk",
                media_root=tmp_path,
            ),
        )
