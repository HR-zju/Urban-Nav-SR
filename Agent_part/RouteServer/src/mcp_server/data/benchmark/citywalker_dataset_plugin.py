"""Freeze CityWalker Teleop data as public inputs and hidden references.

This module reads the reference repository's three-line pose-label files but
does not import, mutate, or depend on its training dataset implementation.
Target-point selection is intentionally deterministic here because CityWalker
uses a random draw in ``TeleopDataset.__getitem__``.  The frozen draw becomes
part of the stable ``sample_id`` and is consequently reproducible across model
adapters and metric suites.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkMessageV1,
    BenchmarkProfileV1,
    BenchmarkReferenceV1,
    BenchmarkTimePolicyV1,
)
from mcp_server.contracts.scene_event_tags import CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkDatasetPluginV1
from mcp_server.eval.openloop.benchmark_plugin_package import (
    artifact_sha256_v1,
    build_benchmark_package_manifest_v1,
    write_benchmark_rows_v1,
)
from mcp_server.eval.openloop.benchmark_plugin_media import build_benchmark_media_manifest_v1


CITYWALKER_DATASET_PLUGIN_ID_V1 = "citywalker_teleop_v1"
CITYWALKER_INPUT_PROFILE_ID_V1 = "citywalker_fpv_pose_goal_prompt_v1"
CITYWALKER_PREDICTION_GEOMETRY_ID_V1 = "citywalker_current_pose_xy_v1"
CITYWALKER_COORDINATE_FRAME_V1 = "citywalker_current_pose_x_right_y_front_v1"
# CityWalker human labels and UrbanNav's derived audit tags share names/order,
# but differ in provenance.  Keep this alias for public-package compatibility.
CITYWALKER_CATEGORY_ORDER_V1 = CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1

_SPLIT_SEQUENCE_IDS = {
    "train": tuple(range(1, 9)),
    "validation": (9,),
    "test": tuple(range(10, 19)),
}


@dataclass(frozen=True)
class CityWalkerPoseRecordV1:
    """One parsed CityWalker pose-label triplet.

    Args:
        timestamp_s: Source GPS timestamp in seconds.
        translation_xyz: Pose translation in source world coordinates.
        rotation_rotvec: Axis-angle pose rotation in source world coordinates.
        frame_index: Source forward-camera frame index.
        categories: Ordered overlap-permitted CityWalker category flags.
    """

    timestamp_s: float
    translation_xyz: tuple[float, float, float]
    rotation_rotvec: tuple[float, float, float]
    frame_index: int
    categories: tuple[bool, ...]


class CityWalkerDatasetPluginV1(BenchmarkDatasetPluginV1):
    """Build deterministic CityWalker Teleop benchmark packages."""

    plugin_id = CITYWALKER_DATASET_PLUGIN_ID_V1

    def build_package(self, **kwargs: Any) -> dict[str, Any]:
        """Materialize one CityWalker split into frozen input/reference JSONL.

        Args:
            **kwargs: See :func:`build_citywalker_benchmark_package_v1`.

        Returns:
            JSON-compatible build report and package manifest identity.
        """

        return build_citywalker_benchmark_package_v1(**kwargs)


def citywalker_default_profile_v1(
    *,
    profile_id: str = "citywalker_teleop_openloop_v1",
    horizons_s: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
) -> BenchmarkProfileV1:
    """Return the first frozen direct-evaluation CityWalker profile.

    Args:
        profile_id: Stable benchmark profile identifier.
        horizons_s: Common physical horizons; uncovered horizons report coverage
            rather than being silently extrapolated.

    Returns:
        Legal CityWalker benchmark profile.
    """

    return BenchmarkProfileV1(
        profile_id=profile_id,
        dataset_plugin_id=CITYWALKER_DATASET_PLUGIN_ID_V1,
        model_input_profile_id=CITYWALKER_INPUT_PROFILE_ID_V1,
        prediction_geometry_id=CITYWALKER_PREDICTION_GEOMETRY_ID_V1,
        metric_suite_ids=("trajectory_geometry_core_v1", "citywalker_openloop_metrics_v1"),
        time_policy=BenchmarkTimePolicyV1(
            policy_id="citywalker_source_gps_time_linear_xy_v1",
            horizons_s=horizons_s,
            min_angle_reference_distance_m=1.0,
        ),
    )


def build_citywalker_benchmark_package_v1(
    *,
    pose_label_dir: str | Path,
    image_root_dir: str | Path,
    output_root: str | Path,
    split: str,
    profile: BenchmarkProfileV1,
    package_id: str,
    target_seed: int = 20260808,
    context_size: int = 5,
    waypoint_length: int = 5,
    search_window: int = 50,
    arrived_threshold_frames: int = 5,
    arrived_probability: float = 0.3,
    allow_missing_images: bool = False,
) -> dict[str, Any]:
    """Build one immutable CityWalker package without invoking a model or API.

    The public prompt contains only past FPV images, deterministic local history,
    and a deterministic point goal.  Five future poses, source timestamps,
    arrival labels, and CityWalker categories remain exclusively in the hidden
    reference ledger.

    Args:
        pose_label_dir: Directory containing ``pose_traj_XX.txt`` triplets.
        image_root_dir: Directory containing ``traj_XX/forward_XXXX.jpg``.
        output_root: Package artifact root.
        split: Official CityWalker ``train``, ``validation``, or ``test`` split.
        profile: Frozen generic plugin profile.
        package_id: Immutable package identity.
        target_seed: Global deterministic target-point sampling seed.
        context_size: Number of model-visible historical frames.
        waypoint_length: Number of hidden future trajectory points.
        search_window: Maximum source future window used for target selection.
        arrived_threshold_frames: Near-goal target range width.
        arrived_probability: Deterministic per-case probability of near-goal draw.
        allow_missing_images: Keep metadata-only cases for CI; never use this for
            a real visual benchmark package.

    Returns:
        Build report containing artifact paths, counts, and source digests.

    Raises:
        ValueError: If the profile/split/CityWalker source contract is invalid.
        FileNotFoundError: If required source files or images are missing.
    """

    _validate_citywalker_build_args_v1(
        split=split,
        profile=profile,
        context_size=context_size,
        waypoint_length=waypoint_length,
        search_window=search_window,
        arrived_threshold_frames=arrived_threshold_frames,
        arrived_probability=arrived_probability,
    )
    output = Path(output_root).expanduser().resolve()
    pose_root = Path(pose_label_dir).expanduser().resolve()
    image_root = Path(image_root_dir).expanduser().resolve()
    inputs: list[BenchmarkInputV1] = []
    references: list[BenchmarkReferenceV1] = []
    source_fingerprints: dict[str, str] = {}
    missing_images: list[str] = []
    for sequence_id in _SPLIT_SEQUENCE_IDS[split]:
        pose_path = pose_root / f"pose_traj_{sequence_id:02d}.txt"
        records = load_citywalker_pose_records_v1(pose_path)
        source_fingerprints[pose_path.name] = artifact_sha256_v1(pose_path)
        sequence_inputs, sequence_references, sequence_missing = build_citywalker_cases_v1(
            records=records,
            sequence_id=sequence_id,
            split=split,
            profile=profile,
            image_root=image_root,
            target_seed=target_seed,
            context_size=context_size,
            waypoint_length=waypoint_length,
            search_window=search_window,
            arrived_threshold_frames=arrived_threshold_frames,
            arrived_probability=arrived_probability,
        )
        inputs.extend(sequence_inputs)
        references.extend(sequence_references)
        missing_images.extend(sequence_missing)
    if missing_images and not allow_missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(f"CityWalker package has {len(missing_images)} missing prompt images: {preview}")
    input_path = output / "inputs.jsonl"
    reference_path = output / "references.jsonl"
    media_manifest_path = output / "media_manifest.jsonl"
    manifest_path = output / "package.json"
    write_benchmark_rows_v1(input_path, inputs)
    write_benchmark_rows_v1(reference_path, references)
    media_manifest_report = None
    if not missing_images:
        media_manifest_report = build_benchmark_media_manifest_v1(
            inputs=inputs,
            source_root=image_root,
            output_jsonl=media_manifest_path,
        )
    manifest = build_benchmark_package_manifest_v1(
        package_id=package_id,
        profile=profile,
        split=split,
        input_jsonl=input_path,
        reference_jsonl=reference_path,
        output_json=manifest_path,
        media_manifest_jsonl=(None if media_manifest_report is None else media_manifest_path),
        source_fingerprints=source_fingerprints,
    )
    return {
        "dataset_plugin_id": CITYWALKER_DATASET_PLUGIN_ID_V1,
        "package_id": manifest.package_id,
        "package_path": str(manifest_path),
        "input_path": str(input_path),
        "reference_path": str(reference_path),
        "input_count": len(inputs),
        "reference_count": len(references),
        "missing_prompt_image_count": len(missing_images),
        "allow_missing_images": allow_missing_images,
        "media_manifest": media_manifest_report,
        "source_fingerprints": source_fingerprints,
    }


def load_citywalker_pose_records_v1(path: str | Path) -> list[CityWalkerPoseRecordV1]:
    """Parse one reference CityWalker pose-label file without importing its code.

    Args:
        path: ``pose_traj_XX.txt`` source path.

    Returns:
        Ordered parsed source records.

    Raises:
        ValueError: If triplets, timestamps, poses, or category widths drift.
    """

    source_path = Path(path).expanduser()
    lines = [line.strip() for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError(f"CityWalker pose-label file does not contain three-line records: {source_path}")
    records: list[CityWalkerPoseRecordV1] = []
    prior_time = -math.inf
    for offset in range(0, len(lines), 3):
        gps_tokens = lines[offset].split(",")
        pose_tokens = lines[offset + 1].split(",")
        category_tokens = lines[offset + 2].split(",")
        if len(gps_tokens) < 1 or len(pose_tokens) < 8 or len(category_tokens) != len(CITYWALKER_CATEGORY_ORDER_V1):
            raise ValueError(f"malformed CityWalker record at {source_path}:{offset + 1}")
        timestamp_s = float(gps_tokens[0])
        if not math.isfinite(timestamp_s) or timestamp_s <= prior_time:
            raise ValueError(f"CityWalker GPS timestamps are not strictly increasing: {source_path}:{offset + 1}")
        prior_time = timestamp_s
        values = tuple(float(item) for item in pose_tokens[1:7])
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite CityWalker pose at {source_path}:{offset + 2}")
        category_values = tuple(_parse_binary_flag_v1(item, source_path, offset + 3) for item in category_tokens)
        records.append(
            CityWalkerPoseRecordV1(
                timestamp_s=timestamp_s,
                translation_xyz=(values[0], values[1], values[2]),
                rotation_rotvec=(values[3], values[4], values[5]),
                frame_index=int(pose_tokens[7]),
                categories=category_values,
            )
        )
    if not records:
        raise ValueError(f"CityWalker pose-label file is empty: {source_path}")
    return records


def build_citywalker_cases_v1(
    *,
    records: list[CityWalkerPoseRecordV1],
    sequence_id: int,
    split: str,
    profile: BenchmarkProfileV1,
    image_root: Path,
    target_seed: int,
    context_size: int,
    waypoint_length: int,
    search_window: int,
    arrived_threshold_frames: int,
    arrived_probability: float,
) -> tuple[list[BenchmarkInputV1], list[BenchmarkReferenceV1], list[str]]:
    """Materialize deterministic cases for one CityWalker sequence.

    Args:
        records: Parsed source pose-label stream.
        sequence_id: Official two-digit CityWalker sequence number.
        split: Frozen official split.
        profile: Legal generic benchmark profile.
        image_root: Source FPV root.
        target_seed: Deterministic target sampler seed.
        context_size: Historical frame count.
        waypoint_length: Future reference length.
        search_window: Target selection window.
        arrived_threshold_frames: Near-goal target boundary.
        arrived_probability: Near-goal selection probability.

    Returns:
        Inputs, references, and missing image paths.
    """

    usable = len(records) - context_size - max(arrived_threshold_frames * 2, waypoint_length)
    inputs: list[BenchmarkInputV1] = []
    references: list[BenchmarkReferenceV1] = []
    missing_images: list[str] = []
    stride = context_size if split == "train" else 1
    for pose_start in range(0, max(usable, 0), stride):
        future_start = pose_start + context_size
        future_end = min(future_start + search_window, len(records))
        future = records[future_start:future_end]
        if len(future) < waypoint_length:
            continue
        target_index, arrived = _select_citywalker_target_v1(
            future_length=len(future),
            waypoint_length=waypoint_length,
            arrived_threshold_frames=arrived_threshold_frames,
            arrived_probability=arrived_probability,
            target_seed=target_seed,
            sequence_id=sequence_id,
            pose_start=pose_start,
        )
        history = records[pose_start : pose_start + context_size]
        reference_future = records[future_start : future_start + waypoint_length]
        current = history[-1]
        history_xy = [_transform_citywalker_pose_xy_v1(item, current) for item in history]
        reference_xy = [_transform_citywalker_pose_xy_v1(item, current) for item in reference_future]
        target_xy = _transform_citywalker_pose_xy_v1(future[target_index], current)
        source_group_id = f"citywalker_sequence_{sequence_id:02d}"
        # Frame ids are video-frame names, not a guaranteed unique source-row
        # key.  Keep the pose-stream anchor in the parent identity so repeated
        # or repaired video frame ids cannot collapse distinct benchmark cases.
        parent_sample_id = (
            f"citywalker:{sequence_id:02d}:pose:{pose_start:06d}:frame:{current.frame_index:04d}"
        )
        sample_id = f"{parent_sample_id}:target:{target_index:03d}:seed:{target_seed}"
        image_dir = _citywalker_sequence_image_dir_v1(image_root, sequence_id)
        images = [str(image_dir / f"forward_{item.frame_index:04d}.jpg") for item in history]
        missing_images.extend(path for path in images if not Path(path).is_file())
        messages = [
            BenchmarkMessageV1(
                role="system",
                content=(
                    "You are an embodied pedestrian-navigation policy. Use only the visible FPV history, "
                    "local pose history, and point goal. Predict a future local xy trajectory."
                ),
            ),
            BenchmarkMessageV1(
                role="user",
                content=(
                    "<FPVHistory>" + "<image>" * context_size + "</FPVHistory>\n"
                    "<NavigationCondition>{\"coordinate_frame\":\"citywalker_current_pose_x_right_y_front_v1\","
                    f"\"history_xy_m\":{_compact_xy_json_v1(history_xy)},"
                    f"\"target_xy_m\":[{target_xy[0]:.6f},{target_xy[1]:.6f}],"
                    f"\"trajectory_point_count\":{waypoint_length}}}</NavigationCondition>\n"
                    "Return the future local xy trajectory and, if supported, an arrival probability."
                ),
            ),
        ]
        inputs.append(
            BenchmarkInputV1(
                sample_id=sample_id,
                parent_sample_id=parent_sample_id,
                source_group_id=source_group_id,
                split_group_id=source_group_id,
                split=split,
                dataset_plugin_id=CITYWALKER_DATASET_PLUGIN_ID_V1,
                benchmark_profile_id=profile.profile_id,
                dataset_source="citywalker_teleop",
                messages=messages,
                images=images,
                image_roles=["fpv_history"] * len(images),
                public_context={
                    "coordinate_frame": CITYWALKER_COORDINATE_FRAME_V1,
                    "history_xy_m": history_xy,
                    "target_xy_m": target_xy,
                    "trajectory_point_count": waypoint_length,
                },
                modality_profile_id=CITYWALKER_INPUT_PROFILE_ID_V1,
            )
        )
        references.append(
            BenchmarkReferenceV1(
                sample_id=sample_id,
                parent_sample_id=parent_sample_id,
                source_group_id=source_group_id,
                split_group_id=source_group_id,
                split=split,
                dataset_plugin_id=CITYWALKER_DATASET_PLUGIN_ID_V1,
                benchmark_profile_id=profile.profile_id,
                dataset_source="citywalker_teleop",
                trajectory=[list(point) for point in reference_xy],
                trajectory_timestamps_s=[item.timestamp_s - current.timestamp_s for item in reference_future],
                trajectory_valid_mask=[True] * len(reference_xy),
                coordinate_frame=CITYWALKER_COORDINATE_FRAME_V1,
                arrival_label=arrived,
                scenario_labels=tuple(
                    label for label, enabled in zip(CITYWALKER_CATEGORY_ORDER_V1, current.categories) if enabled
                ),
                label_provenance="deterministic_gt",
                field_provenance={
                    "trajectory": "deterministic_gt",
                    "arrival_label": "deterministic_gt",
                    "scenario_labels": "reference_annotation",
                },
                subgroup_metadata={
                    "sequence_id": sequence_id,
                    "frame_index": current.frame_index,
                    "arrival_label_derivation": "citywalker_frozen_target_sampler_v1",
                },
            )
        )
    return inputs, references, missing_images


def _validate_citywalker_build_args_v1(
    *,
    split: str,
    profile: BenchmarkProfileV1,
    context_size: int,
    waypoint_length: int,
    search_window: int,
    arrived_threshold_frames: int,
    arrived_probability: float,
) -> None:
    """Validate source/profile settings before materializing any artifact."""

    if split not in _SPLIT_SEQUENCE_IDS:
        raise ValueError(f"unsupported CityWalker split: {split}")
    if profile.dataset_plugin_id != CITYWALKER_DATASET_PLUGIN_ID_V1:
        raise ValueError("CityWalker profile has a different dataset_plugin_id")
    if profile.model_input_profile_id != CITYWALKER_INPUT_PROFILE_ID_V1:
        raise ValueError("CityWalker profile has a different model_input_profile_id")
    if profile.prediction_geometry_id != CITYWALKER_PREDICTION_GEOMETRY_ID_V1:
        raise ValueError("CityWalker profile has a different prediction_geometry_id")
    if min(context_size, waypoint_length, search_window, arrived_threshold_frames) <= 0:
        raise ValueError("CityWalker frame counts must be positive")
    if search_window < waypoint_length + arrived_threshold_frames + 1:
        raise ValueError("CityWalker search_window cannot support the far-target range")
    if not 0.0 <= arrived_probability <= 1.0:
        raise ValueError("arrived_probability must be in [0, 1]")


def _select_citywalker_target_v1(
    *,
    future_length: int,
    waypoint_length: int,
    arrived_threshold_frames: int,
    arrived_probability: float,
    target_seed: int,
    sequence_id: int,
    pose_start: int,
) -> tuple[int, bool]:
    """Freeze a CityWalker target-point draw using a stable per-case seed."""

    material = f"{target_seed}:{sequence_id}:{pose_start}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random(seed)
    arrived = rng.random() < arrived_probability
    maximum = future_length - 1
    if arrived:
        lower = waypoint_length
        upper = min(waypoint_length + arrived_threshold_frames, maximum)
    else:
        lower = waypoint_length + arrived_threshold_frames
        upper = maximum
    if lower > upper:
        raise ValueError("CityWalker target range is empty; source length/configuration drifted")
    return rng.randint(lower, upper), arrived


def _transform_citywalker_pose_xy_v1(
    pose: CityWalkerPoseRecordV1,
    current: CityWalkerPoseRecordV1,
) -> tuple[float, float]:
    """Transform a source pose into CityWalker's current-frame x-right/y-front xy.

    The transformation matches the reference implementation's
    ``inv(T_current) @ T_pose`` followed by its documented ``[y, -x]`` axis
    conversion, but is implemented here without importing SciPy or CityWalker.
    """

    rotation = _rotation_matrix_from_rotvec_v1(current.rotation_rotvec)
    delta = tuple(pose.translation_xyz[index] - current.translation_xyz[index] for index in range(3))
    local = _mat_vec_transpose_v1(rotation, delta)
    return local[1], -local[0]


def _rotation_matrix_from_rotvec_v1(rotvec: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    """Return Rodrigues rotation matrix for one finite axis-angle vector."""

    x, y, z = rotvec
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-12:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ax, ay, az = x / angle, y / angle, z / angle
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1.0 - cosine
    return (
        (cosine + ax * ax * one_minus, ax * ay * one_minus - az * sine, ax * az * one_minus + ay * sine),
        (ay * ax * one_minus + az * sine, cosine + ay * ay * one_minus, ay * az * one_minus - ax * sine),
        (az * ax * one_minus - ay * sine, az * ay * one_minus + ax * sine, cosine + az * az * one_minus),
    )


def _mat_vec_transpose_v1(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Multiply a vector by the transpose of a 3x3 matrix."""

    result = tuple(
        sum(matrix[row][column] * vector[row] for row in range(3))
        for column in range(3)
    )
    return float(result[0]), float(result[1]), float(result[2])


def _parse_binary_flag_v1(value: str, source_path: Path, line_number: int) -> bool:
    """Parse CityWalker category flags without silently coercing malformed values."""

    if value not in {"0", "1"}:
        raise ValueError(f"invalid CityWalker category flag at {source_path}:{line_number}: {value!r}")
    return value == "1"


def _citywalker_sequence_image_dir_v1(image_root: Path, sequence_id: int) -> Path:
    """Resolve the reference's ``traj_nav`` and legacy ``traj`` media layouts."""

    for name in (f"traj_nav_{sequence_id:02d}", f"traj_{sequence_id:02d}"):
        candidate = image_root / name
        if candidate.is_dir():
            return candidate
    return image_root / f"traj_nav_{sequence_id:02d}"


def _compact_xy_json_v1(points: Iterable[tuple[float, float]]) -> str:
    """Render small prompt-visible xy arrays with stable decimal JSON syntax."""

    return "[" + ",".join(f"[{x:.6f},{y:.6f}]" for x, y in points) + "]"


__all__ = [
    "CITYWALKER_CATEGORY_ORDER_V1",
    "CITYWALKER_COORDINATE_FRAME_V1",
    "CITYWALKER_DATASET_PLUGIN_ID_V1",
    "CITYWALKER_INPUT_PROFILE_ID_V1",
    "CITYWALKER_PREDICTION_GEOMETRY_ID_V1",
    "CityWalkerDatasetPluginV1",
    "CityWalkerPoseRecordV1",
    "build_citywalker_benchmark_package_v1",
    "build_citywalker_cases_v1",
    "citywalker_default_profile_v1",
    "load_citywalker_pose_records_v1",
]
