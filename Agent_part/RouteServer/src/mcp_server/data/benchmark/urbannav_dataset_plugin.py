"""Project frozen UrbanNav Stage-II open-loop artifacts into plugin packages.

The legacy OpenLoop V2 contract remains authoritative for existing Stage-I/II
runs.  This adapter is deliberately one-way: it consumes already-frozen,
model-visible Stage-II trajectory inputs plus their hidden targets and writes a
model-independent package for cross-benchmark metrics.  It never synthesizes
new geometry, prompts, or labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkMessageV1,
    BenchmarkProfileV1,
    BenchmarkReferenceV1,
    BenchmarkTimePolicyV1,
)
from mcp_server.contracts.openloop_benchmark import OpenLoopInputV1, OpenLoopReferenceV1
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkDatasetPluginV1
from mcp_server.eval.openloop.benchmark_plugin_package import (
    build_benchmark_package_manifest_v1,
    write_benchmark_rows_v1,
)


URBANNAV_DATASET_PLUGIN_ID_V1 = "urbannav_stage_ii_projection_v1"
URBANNAV_INPUT_PROFILE_ID_V1 = "urbannav_stage_ii_visible_context_v1"
URBANNAV_PREDICTION_GEOMETRY_ID_V1 = "urbannav_current_ego_flu_xy_v1"
URBANNAV_COORDINATE_FRAME_V1 = "current_ego_x_forward_y_left_v1"


class UrbanNavDatasetPluginV1(BenchmarkDatasetPluginV1):
    """Project existing canonical UrbanNav Stage-II packages into generic rows."""

    plugin_id = URBANNAV_DATASET_PLUGIN_ID_V1

    def build_package(self, **kwargs: Any) -> dict[str, Any]:
        """Materialize one generic package from legacy Stage-II artifacts.

        Args:
            **kwargs: See :func:`build_urbannav_stage_ii_benchmark_package_v1`.

        Returns:
            JSON-compatible package build report.
        """

        return build_urbannav_stage_ii_benchmark_package_v1(**kwargs)


def urbannav_stage_ii_default_profile_v1(
    *,
    profile_id: str = "urbannav_stage_ii_openloop_v1",
    horizons_s: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> BenchmarkProfileV1:
    """Return the generic plugin profile for existing UrbanNav Stage-II targets.

    Args:
        profile_id: Stable package/evaluation profile identity.
        horizons_s: Physical metric horizons already supported by the legacy
            Stage-II target time grid.

    Returns:
        Generic UrbanNav Stage-II profile with common and diagnostic suites.
    """

    return BenchmarkProfileV1(
        profile_id=profile_id,
        dataset_plugin_id=URBANNAV_DATASET_PLUGIN_ID_V1,
        model_input_profile_id=URBANNAV_INPUT_PROFILE_ID_V1,
        prediction_geometry_id=URBANNAV_PREDICTION_GEOMETRY_ID_V1,
        metric_suite_ids=("trajectory_geometry_core_v1", "urbannav_openloop_metrics_v1"),
        time_policy=BenchmarkTimePolicyV1(
            policy_id="urbannav_stage_ii_fixed_time_v1",
            horizons_s=horizons_s,
            min_angle_reference_distance_m=1e-3,
        ),
    )


def build_urbannav_stage_ii_benchmark_package_v1(
    *,
    legacy_input_jsonl: str | Path,
    legacy_reference_jsonl: str | Path,
    output_root: str | Path,
    split: str,
    profile: BenchmarkProfileV1,
    package_id: str,
) -> dict[str, Any]:
    """Write a generic benchmark package from frozen legacy Stage-II rows.

    Args:
        legacy_input_jsonl: Legacy model-visible Stage-II input artifact.
        legacy_reference_jsonl: Legacy hidden Stage-II reference artifact.
        output_root: Generic package output root.
        split: Required frozen split selector.
        profile: Generic profile that accepts the UrbanNav projection.
        package_id: Immutable output package id.

    Returns:
        Build report with immutable artifact locations and row counts.

    Raises:
        ValueError: If legacy Stage-II rows or the generic profile drift.
    """

    if profile.dataset_plugin_id != URBANNAV_DATASET_PLUGIN_ID_V1:
        raise ValueError("UrbanNav generic profile has a different dataset plugin")
    if profile.model_input_profile_id != URBANNAV_INPUT_PROFILE_ID_V1:
        raise ValueError("UrbanNav generic profile has a different input profile")
    if profile.prediction_geometry_id != URBANNAV_PREDICTION_GEOMETRY_ID_V1:
        raise ValueError("UrbanNav generic profile has a different output geometry")
    legacy_inputs = _load_legacy_models_v1(legacy_input_jsonl, OpenLoopInputV1)
    legacy_references = _load_legacy_models_v1(legacy_reference_jsonl, OpenLoopReferenceV1)
    if set(legacy_inputs) != set(legacy_references):
        raise ValueError("legacy Stage-II input/reference sample ids differ")
    inputs: list[BenchmarkInputV1] = []
    references: list[BenchmarkReferenceV1] = []
    for sample_id, legacy_input in legacy_inputs.items():
        legacy_reference = legacy_references[sample_id]
        if legacy_input.split != split:
            continue
        input_row, reference_row = project_urbannav_stage_ii_row_v1(legacy_input, legacy_reference, profile=profile)
        inputs.append(input_row)
        references.append(reference_row)
    if not inputs:
        raise ValueError(f"no legacy Stage-II rows selected for split={split!r}")
    output = Path(output_root).expanduser().resolve()
    input_path = output / "inputs.jsonl"
    reference_path = output / "references.jsonl"
    manifest_path = output / "package.json"
    write_benchmark_rows_v1(input_path, inputs)
    write_benchmark_rows_v1(reference_path, references)
    manifest = build_benchmark_package_manifest_v1(
        package_id=package_id,
        profile=profile,
        split=split,
        input_jsonl=input_path,
        reference_jsonl=reference_path,
        output_json=manifest_path,
        source_fingerprints={
            "legacy_input_jsonl": _sha256_path_v1(legacy_input_jsonl),
            "legacy_reference_jsonl": _sha256_path_v1(legacy_reference_jsonl),
        },
    )
    return {
        "dataset_plugin_id": URBANNAV_DATASET_PLUGIN_ID_V1,
        "package_id": package_id,
        "package_path": str(manifest_path),
        "input_count": manifest.input_count,
        "reference_count": manifest.reference_count,
    }


def project_urbannav_stage_ii_row_v1(
    legacy_input: OpenLoopInputV1,
    legacy_reference: OpenLoopReferenceV1,
    *,
    profile: BenchmarkProfileV1,
) -> tuple[BenchmarkInputV1, BenchmarkReferenceV1]:
    """Project one frozen Stage-II legacy pair without altering visible content.

    Args:
        legacy_input: Valid model-visible legacy Stage-II trajectory input.
        legacy_reference: Aligned hidden legacy Stage-II trajectory reference.
        profile: Generic plugin profile accepting this projection.

    Returns:
        Exact generic visible input and hidden trajectory reference.

    Raises:
        ValueError: If Stage-II target structure or pair identity is invalid.
    """

    if legacy_input.stage != "stage_ii" or legacy_input.task_type != "trajectory":
        raise ValueError("only legacy Stage-II trajectory inputs can be projected")
    if legacy_reference.stage != "stage_ii" or legacy_reference.task_type != "trajectory":
        raise ValueError("only legacy Stage-II trajectory references can be projected")
    identity_fields = ("sample_id", "parent_sample_id", "source_group_id", "split_group_id", "dataset_source")
    if any(getattr(legacy_input, name) != getattr(legacy_reference, name) for name in identity_fields):
        raise ValueError("legacy Stage-II input/reference identity drift")
    target = legacy_reference.target.get("trajectory_target")
    if not isinstance(target, dict):
        raise ValueError("legacy Stage-II reference is missing trajectory_target")
    trajectory = target.get("trajectory_ego_t")
    mask = target.get("action_mask")
    policy = target.get("policy")
    if not isinstance(trajectory, list) or not isinstance(mask, list) or not isinstance(policy, dict):
        raise ValueError("legacy Stage-II target is structurally invalid")
    target_dt_s = policy.get("target_dt_sec")
    if not isinstance(target_dt_s, (int, float)) or float(target_dt_s) <= 0.0:
        raise ValueError("legacy Stage-II trajectory policy has no positive target_dt_sec")
    direction_label = None
    if legacy_reference.direction_stamp is not None:
        direction_label = legacy_reference.direction_stamp.name
    projected_input = BenchmarkInputV1(
        sample_id=legacy_input.sample_id,
        parent_sample_id=legacy_input.parent_sample_id,
        source_group_id=legacy_input.source_group_id,
        split_group_id=legacy_input.split_group_id,
        split=legacy_input.split,
        dataset_plugin_id=URBANNAV_DATASET_PLUGIN_ID_V1,
        benchmark_profile_id=profile.profile_id,
        dataset_source=legacy_input.dataset_source,
        messages=[
            BenchmarkMessageV1(role=message.role, content=message.content)
            for message in legacy_input.messages
            if message.role != "assistant"
        ],
        images=legacy_input.images,
        image_roles=legacy_input.image_roles,
        public_context=legacy_input.public_context,
        modality_profile_id=URBANNAV_INPUT_PROFILE_ID_V1,
    )
    projected_reference = BenchmarkReferenceV1(
        sample_id=legacy_reference.sample_id,
        parent_sample_id=legacy_reference.parent_sample_id,
        source_group_id=legacy_reference.source_group_id,
        split_group_id=legacy_reference.split_group_id,
        split=legacy_input.split,
        dataset_plugin_id=URBANNAV_DATASET_PLUGIN_ID_V1,
        benchmark_profile_id=profile.profile_id,
        dataset_source=legacy_reference.dataset_source,
        trajectory=[[float(point[0]), float(point[1])] for point in trajectory],
        trajectory_timestamps_s=[float(target_dt_s) * (index + 1) for index in range(len(trajectory))],
        trajectory_valid_mask=[bool(item) for item in mask],
        coordinate_frame=URBANNAV_COORDINATE_FRAME_V1,
        label_provenance="deterministic_gt",
        subgroup_metadata={
            "direction_label": direction_label,
            "legacy_trajectory_policy_id": legacy_reference.trajectory_policy_id,
            "legacy_trajectory_target_digest": target.get("target_digest"),
        },
    )
    return projected_input, projected_reference


def _load_legacy_models_v1(path: str | Path, model_type: type[Any]) -> dict[str, Any]:
    """Load unique strict legacy JSONL models keyed by their sample id."""

    rows: dict[str, Any] = {}
    for payload in iter_jsonl(path):
        row = model_type.model_validate(payload)
        if row.sample_id in rows:
            raise ValueError(f"duplicate legacy sample_id: {row.sample_id}")
        rows[row.sample_id] = row
    return rows


def _sha256_path_v1(path: str | Path) -> str:
    """Return SHA256 through the generic package implementation without import cycles."""

    from mcp_server.eval.openloop.benchmark_plugin_package import artifact_sha256_v1

    return artifact_sha256_v1(path)


__all__ = [
    "URBANNAV_COORDINATE_FRAME_V1",
    "URBANNAV_DATASET_PLUGIN_ID_V1",
    "URBANNAV_INPUT_PROFILE_ID_V1",
    "URBANNAV_PREDICTION_GEOMETRY_ID_V1",
    "UrbanNavDatasetPluginV1",
    "build_urbannav_stage_ii_benchmark_package_v1",
    "project_urbannav_stage_ii_row_v1",
    "urbannav_stage_ii_default_profile_v1",
]
