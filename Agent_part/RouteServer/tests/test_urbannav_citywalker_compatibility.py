"""Focused public-admission coverage for the CityWalker-to-UrbanNav bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkMessageV1,
    BenchmarkReferenceV1,
)
from mcp_server.data.benchmark.urbannav_citywalker_compatibility import (
    audit_urbannav_citywalker_compatibility_v1,
    build_urbannav_citywalker_compatible_package_v1,
    urbannav_citywalker_native_default_profile_v1,
)
from mcp_server.data.benchmark.urbannav_dataset_plugin import urbannav_stage_ii_default_profile_v1
from mcp_server.eval.openloop.benchmark_plugin_package import (
    build_benchmark_package_manifest_v1,
    load_benchmark_package_v1,
    write_benchmark_rows_v1,
)


def _source_input(sample_id: str,
                  *,
                  with_goal: bool = True) -> BenchmarkInputV1:
    """Create one synthetic public UrbanNav row with explicit history binding."""

    context = {
        "coordinate_frame":
        "current_ego_x_forward_y_left_v1",
        "history_xy_m": [[-0.8, 0.0], [-0.6, 0.0], [-0.4, 0.0], [-0.2, 0.0],
                         [0.0, 0.0]],
        "history_image_indices": [1, 2, 3, 4, 5],
    }
    if with_goal:
        context["target_xy_m"] = [3.0, 0.5]
    return BenchmarkInputV1(
        sample_id=sample_id,
        parent_sample_id=f"{sample_id}:parent",
        source_group_id="source_group",
        split_group_id="split_group",
        split="validation",
        dataset_plugin_id="urbannav_stage_ii_projection_v1",
        benchmark_profile_id="source_urbannav_profile",
        dataset_source="sidewalk",
        messages=[
            BenchmarkMessageV1(role="user", content="public instruction")
        ],
        images=[
            "/tmp/bev.png", "/tmp/f0.jpg", "/tmp/f1.jpg", "/tmp/f2.jpg",
            "/tmp/f3.jpg", "/tmp/f4.jpg"
        ],
        image_roles=[
            "bev.current", "fpv.history_4", "fpv.history_3", "fpv.history_2",
            "fpv.history_1", "fpv.current"
        ],
        public_context=context,
        modality_profile_id="urbannav_stage_ii_visible_context_v1",
    )


def _source_reference(input_row: BenchmarkInputV1) -> BenchmarkReferenceV1:
    """Create the hidden pair after public-row construction for package IO only."""

    return BenchmarkReferenceV1(
        sample_id=input_row.sample_id,
        parent_sample_id=input_row.parent_sample_id,
        source_group_id=input_row.source_group_id,
        split_group_id=input_row.split_group_id,
        split=input_row.split,
        dataset_plugin_id=input_row.dataset_plugin_id,
        benchmark_profile_id=input_row.benchmark_profile_id,
        dataset_source=input_row.dataset_source,
        trajectory=[[1.0, 0.0], [2.0, 0.0]],
        trajectory_timestamps_s=[1.0, 2.0],
        trajectory_valid_mask=[True, True],
        coordinate_frame="current_ego_x_forward_y_left_v1",
    )


def test_public_compatibility_audit_rejects_missing_goal_without_reference_access(
):
    """A missing public goal must become a funnel rejection rather than a guess."""

    eligible = _source_input("eligible")
    missing_goal = _source_input("missing-goal", with_goal=False)
    report = audit_urbannav_citywalker_compatibility_v1({
        eligible.sample_id:
        eligible,
        missing_goal.sample_id:
        missing_goal
    })
    assert report["denominator_count"] == 2
    assert report["eligible_count"] == 1
    assert report["reject_reason_counts"] == {"missing_public_goal": 1}
    assert report["entries"][0]["selected_image_indices"] == [1, 2, 3, 4, 5]
    assert report["entries"][0]["public_scale_value"] == pytest.approx(0.2)


def test_compatible_package_reduces_to_five_fpv_and_preserves_hidden_denominator(
        tmp_path: Path):
    """Admission must happen from public rows before matching their references."""

    source_profile = urbannav_stage_ii_default_profile_v1(
        profile_id="source_urbannav_profile", horizons_s=(1.0, 2.0))
    eligible = _source_input("eligible")
    rejected = _source_input("missing-goal", with_goal=False)
    source_root = tmp_path / "source"
    write_benchmark_rows_v1(source_root / "inputs.jsonl", [eligible, rejected])
    write_benchmark_rows_v1(
        source_root / "references.jsonl",
        [_source_reference(eligible),
         _source_reference(rejected)],
    )
    build_benchmark_package_manifest_v1(
        package_id="source",
        profile=source_profile,
        split="validation",
        input_jsonl=source_root / "inputs.jsonl",
        reference_jsonl=source_root / "references.jsonl",
        output_json=source_root / "package.json",
    )
    report = build_urbannav_citywalker_compatible_package_v1(
        source_package_json=source_root / "package.json",
        output_root=tmp_path / "derived",
        package_id="derived",
        profile=urbannav_citywalker_native_default_profile_v1(),
    )
    assert report["denominator_count"] == 2
    assert report["eligible_count"] == 1
    _, inputs, references = load_benchmark_package_v1(report["package_path"])
    input_row = inputs["eligible"]
    assert input_row.images == [
        "/tmp/f0.jpg", "/tmp/f1.jpg", "/tmp/f2.jpg", "/tmp/f3.jpg",
        "/tmp/f4.jpg"
    ]
    assert input_row.modality_profile_id == "urbannav_citywalker_native_5fpv_public_history_goal_v1"
    assert input_row.public_context["history_image_indices"] == [0, 1, 2, 3, 4]
    assert references[
        "eligible"].benchmark_profile_id == "urbannav_citywalker_native_openloop_v1"
