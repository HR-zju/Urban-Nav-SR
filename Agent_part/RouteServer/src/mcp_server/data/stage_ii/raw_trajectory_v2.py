"""Read trustworthy four-second trajectories from raw pose/odometry files."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SIDEWALK_RAW_ADAPTER_V2 = "sidewalk_npy_pose_4s_v2"
FOXGLOVE_RAW_ADAPTER_V2 = "foxglove_npy_pose_4s_v2"
EGOWALK_RAW_ADAPTER_V2 = "egowalk_parquet_pose_4s_v2"


@dataclass(frozen=True)
class StageIIRawTrajectoryWindowV2:
    """Carry one current-origin raw trajectory window into the V2 builder.

    Args:
        clip_id: Raw clip or EgoWalk trajectory identity.
        frame_idx: Current video/source frame index.
        source_adapter_id: Versioned raw reader identity.
        trajectory_source: Dataset-owned odometry provenance.
        raw_timestamp_unit: Original clock unit.
        yaw_offset_deg: Odom-to-base rotation used for FLU XY projection.
        trajectory_ego_timed: Current origin and future FLU poses with time.
    """

    clip_id: str
    frame_idx: int
    source_adapter_id: str
    trajectory_source: str
    raw_timestamp_unit: str
    yaw_offset_deg: float
    trajectory_ego_timed: tuple[tuple[float, float, float, float], ...]

    @property
    def action_mask(self) -> tuple[bool, ...]:
        """Return the explicit all-valid raw-pose mask."""

        return (True,) * len(self.trajectory_ego_timed)


class StageIIRawTrajectoryErrorV2(ValueError):
    """Carry one stable raw-source rejection code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class _PoseSeries:
    """Store one cached global pose series and its source clock."""

    positions_xy: np.ndarray
    yaws: np.ndarray
    timestamps: np.ndarray
    frames: np.ndarray
    timestamp_unit: str
    orientation_order: str
    globally_static: bool
    pose_valid: np.ndarray


class StageIIRawTrajectoryReaderV2:
    """Resolve manifest parents against bounded cached raw pose files.

    Args:
        raw_roots: Dataset family to raw dataset root.  SideWalk/Foxglove roots
            may contain nested clip directories; EgoWalk must contain ``data``.
        cache_size: Maximum loaded clips/parquets retained in memory.

    Raises:
        ValueError: If roots are missing or unsupported datasets are supplied.
    """

    def __init__(
        self,
        raw_roots: Mapping[str, str | Path],
        *,
        cache_size: int = 8,
    ) -> None:
        self.raw_roots = {
            str(dataset): Path(path).expanduser()
            for dataset, path in raw_roots.items()
        }
        if not self.raw_roots or any(not path.is_dir() for path in self.raw_roots.values()):
            raise ValueError("every Stage II V2 raw root must be an existing directory")
        unsupported = set(self.raw_roots) - {"sidewalk", "foxglove", "egowalk"}
        if unsupported:
            raise ValueError(f"unsupported Stage II V2 raw datasets: {sorted(unsupported)}")
        self.cache_size = max(int(cache_size), 1)
        self._cache: OrderedDict[tuple[str, str], _PoseSeries] = OrderedDict()
        self._clip_indexes: dict[str, dict[str, Path]] = {}

    def extract(
        self,
        dataset_source: str,
        row: Mapping[str, Any],
    ) -> StageIIRawTrajectoryWindowV2:
        """Read one strict current-to-4-second raw-pose window.

        Args:
            dataset_source: ``sidewalk``, ``foxglove``, or ``egowalk``.
            row: Prepared source-manifest row used only for identity/stamps.

        Returns:
            Current-origin FLU timed rows with real odometry yaw.

        Raises:
            StageIIRawTrajectoryErrorV2: If identity, raw files, continuity,
                instruction horizon, or pose arrays are not trustworthy.
        """

        dataset = str(dataset_source)
        if dataset not in self.raw_roots:
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_raw_root_missing",
                f"no raw root supplied for dataset={dataset}",
            )
        if dataset in {"sidewalk", "foxglove"}:
            return self._extract_npy(dataset, row)
        if dataset == "egowalk":
            return self._extract_egowalk(row)
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_dataset_unsupported",
            f"dataset={dataset} has no V2 raw adapter",
        )

    def _extract_npy(
        self,
        dataset: str,
        row: Mapping[str, Any],
    ) -> StageIIRawTrajectoryWindowV2:
        """Extract SideWalk/Foxglove poses from aligned NPY arrays."""

        meta = row.get("sidewalk_meta")
        if not isinstance(meta, Mapping):
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_identity_missing",
                "sidewalk_meta is required by the NPY pose adapter",
            )
        clip_id = str(meta.get("clip_id") or row.get("clip_id") or "").strip()
        frame_idx = _nonnegative_int(meta.get("frame_idx", row.get("frame_idx")))
        if not clip_id:
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_identity_missing",
                "clip_id is required by the NPY pose adapter",
            )
        declared_source = _coherent_declared_source(row)
        if declared_source not in {"calibrated_pose_projection", "stationary_pose_hold"}:
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_source_not_allowed",
                f"prepared source={declared_source!r} is not pose-calibrated",
            )
        series = self._series(dataset, clip_id)
        if series.globally_static:
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_static_pose_clip",
                "whole-clip frame_positions cannot distinguish real stops from broken pose export",
            )
        # SideWalk 的历史 InternNav/ROS pose yaw 已经以机器人前向为零轴；
        # Foxglove 的批处理资产则沿用 Phase1 明确记录的 +90 度 yaw 校准。
        # 两种来源不能共用一个 offset，否则会把前进位移旋转到横向轴。
        yaw_offset = 0.0 if dataset == "sidewalk" else 90.0
        adapter = SIDEWALK_RAW_ADAPTER_V2 if dataset == "sidewalk" else FOXGLOVE_RAW_ADAPTER_V2
        source = f"{dataset}_frame_pose_odometry"
        rows = _relative_window(
            series,
            current_index=frame_idx,
            horizon_seconds=4.0,
            yaw_offset_deg=yaw_offset,
        )
        return StageIIRawTrajectoryWindowV2(
            clip_id=clip_id,
            frame_idx=frame_idx,
            source_adapter_id=adapter,
            trajectory_source=source,
            raw_timestamp_unit="nanosecond",
            yaw_offset_deg=yaw_offset,
            trajectory_ego_timed=rows,
        )

    def _extract_egowalk(
        self,
        row: Mapping[str, Any],
    ) -> StageIIRawTrajectoryWindowV2:
        """Extract one EgoWalk window from the original odometry parquet."""

        clip_id = str(row.get("clip_id") or "").strip()
        frame_idx = _nonnegative_int(row.get("frame_idx"))
        timestamp = _nonnegative_int(row.get("timestamp"))
        if not clip_id:
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_identity_missing",
                "EgoWalk clip_id is required",
            )
        source = str(_field_value(row, "trajectory_policy.source") or "").strip()
        if source != "egowalk_odometry":
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_source_not_allowed",
                f"EgoWalk source must be egowalk_odometry, got {source!r}",
            )
        annotation_end = _field_value(row, "egowalk_annotation.end_ts")
        if annotation_end is None or int(annotation_end) < timestamp + 4000:
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_instruction_horizon_missing",
                "EgoWalk caption interval does not remain active for the full 4-second target",
            )
        series = self._series("egowalk", clip_id)
        matches = np.flatnonzero(
            (series.timestamps == timestamp) & (series.frames == frame_idx)
        )
        if matches.size != 1:
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_current_pose_not_found",
                "EgoWalk timestamp/frame does not identify exactly one parquet row",
            )
        current_index = int(matches[0])
        rows = _relative_window(
            series,
            current_index=current_index,
            horizon_seconds=4.0,
            yaw_offset_deg=0.0,
        )
        _reject_egowalk_backward_discontinuity(rows)
        return StageIIRawTrajectoryWindowV2(
            clip_id=clip_id,
            frame_idx=frame_idx,
            source_adapter_id=EGOWALK_RAW_ADAPTER_V2,
            trajectory_source="egowalk_odometry",
            raw_timestamp_unit="millisecond",
            yaw_offset_deg=0.0,
            trajectory_ego_timed=rows,
        )

    def _series(self, dataset: str, clip_id: str) -> _PoseSeries:
        """Load one pose series through a bounded LRU cache."""

        key = (dataset, clip_id)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        if dataset == "egowalk":
            try:
                series = _load_egowalk_series(self.raw_roots[dataset], clip_id)
            except StageIIRawTrajectoryErrorV2:
                raise
            except (KeyError, OSError, ValueError) as exc:
                raise StageIIRawTrajectoryErrorV2(
                    "trajectory_raw_read_failed",
                    f"failed to read EgoWalk pose series for clip={clip_id}: {exc}",
                ) from exc
        else:
            path = self._clip_index(dataset).get(clip_id)
            if path is None:
                raise StageIIRawTrajectoryErrorV2(
                    "trajectory_raw_clip_missing",
                    f"raw clip not found for dataset={dataset}, clip={clip_id}",
                )
            try:
                series = _load_npy_series(path, dataset=dataset)
            except StageIIRawTrajectoryErrorV2:
                raise
            except (OSError, ValueError) as exc:
                raise StageIIRawTrajectoryErrorV2(
                    "trajectory_raw_read_failed",
                    f"failed to read NPY pose series for clip={clip_id}: {exc}",
                ) from exc
        self._cache[key] = series
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return series

    def _clip_index(self, dataset: str) -> dict[str, Path]:
        """Index NPY clip directories and reject ambiguous clip identities."""

        existing = self._clip_indexes.get(dataset)
        if existing is not None:
            return existing
        index: dict[str, Path] = {}
        for positions_path in self.raw_roots[dataset].rglob("frame_positions.npy"):
            clip_dir = positions_path.parent
            if not (clip_dir / "frame_orientations.npy").is_file():
                continue
            clip_id = clip_dir.name
            previous = index.get(clip_id)
            if previous is not None and previous.resolve() != clip_dir.resolve():
                raise ValueError(
                    f"duplicate raw clip identity for dataset={dataset}: {clip_id}"
                )
            index[clip_id] = clip_dir
        self._clip_indexes[dataset] = index
        return index


def _load_npy_series(clip_dir: Path, *, dataset: str) -> _PoseSeries:
    """Load aligned position, quaternion, and nanosecond arrays."""

    times_path = clip_dir / "frame_times.npy"
    if not times_path.is_file():
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_raw_time_missing",
            f"frame_times.npy is missing: {clip_dir}",
        )
    positions = np.asarray(np.load(clip_dir / "frame_positions.npy"), dtype=np.float64)
    orientations = np.asarray(np.load(clip_dir / "frame_orientations.npy"), dtype=np.float64)
    timestamps = np.asarray(np.load(times_path), dtype=np.float64).reshape(-1)
    pose_valid_path = clip_dir / "frame_pose_valid.npy"
    pose_valid = (
        np.asarray(np.load(pose_valid_path), dtype=np.bool_).reshape(-1)
        if pose_valid_path.is_file()
        else np.ones(len(timestamps), dtype=np.bool_)
    )
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_raw_pose_shape_invalid",
            "frame_positions.npy must have shape [N,>=2]",
        )
    if orientations.ndim != 2 or orientations.shape[1] < 4:
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_raw_orientation_shape_invalid",
            "frame_orientations.npy must have shape [N,>=4]",
        )
    count = len(timestamps)
    if (
        len(positions) != count
        or len(orientations) != count
        or len(pose_valid) != count
    ):
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_raw_array_length_mismatch",
            "pose/orientation/time arrays must have identical length",
        )
    _require_strict_pose_clock(timestamps)
    order = _orientation_order(clip_dir, dataset=dataset)
    yaws = _quaternion_rows_to_yaw(orientations[:, :4], order=order)
    return _PoseSeries(
        positions_xy=positions[:, :2],
        yaws=yaws,
        timestamps=timestamps,
        frames=np.arange(count, dtype=np.int64),
        timestamp_unit="nanosecond",
        orientation_order=order,
        globally_static=_positions_are_globally_static(positions[:, :2]),
        pose_valid=pose_valid,
    )


def _load_egowalk_series(root: Path, clip_id: str) -> _PoseSeries:
    """Load the exact EgoWalk odometry columns without media decoding."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("pandas is required for EgoWalk V2 trajectory data") from exc
    path = root / "data" / f"{clip_id}.parquet"
    if not path.is_file():
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_raw_clip_missing",
            f"EgoWalk odometry parquet is missing: {path}",
        )
    columns = [
        "timestamp",
        "frame",
        "cart_x",
        "cart_y",
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
    ]
    frame = pd.read_parquet(path, columns=columns)
    positions = frame[["cart_x", "cart_y"]].to_numpy(dtype=np.float64)
    quaternions = frame[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(dtype=np.float64)
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    _require_strict_pose_clock(timestamps)
    return _PoseSeries(
        positions_xy=positions,
        yaws=_quaternion_rows_to_yaw(quaternions, order="xyzw"),
        timestamps=timestamps,
        frames=frame["frame"].to_numpy(dtype=np.int64),
        timestamp_unit="millisecond",
        orientation_order="xyzw",
        globally_static=False,
        pose_valid=np.ones(len(timestamps), dtype=np.bool_),
    )


def _require_strict_pose_clock(timestamps: np.ndarray) -> None:
    """Reject non-finite, duplicate, or decreasing raw pose timestamps."""

    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_raw_time_invalid",
            "raw pose timestamps must contain at least two finite values",
        )
    if not np.all(np.diff(values) > 0.0):
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_time_not_strictly_increasing",
            "the complete raw pose clock must be strictly increasing",
        )


def _relative_window(
    series: _PoseSeries,
    *,
    current_index: int,
    horizon_seconds: float,
    yaw_offset_deg: float,
) -> tuple[tuple[float, float, float, float], ...]:
    r"""Transform a raw global pose window into current-ego FLU coordinates.

    For map displacement \(\Delta p=(\Delta x,\Delta y)\) and current base yaw
    \(\psi_0\), the target coordinates are

    \[
    x_f=\Delta x\cos\psi_0+\Delta y\sin\psi_0,\qquad
    y_l=-\Delta x\sin\psi_0+\Delta y\cos\psi_0.
    \]
    """

    count = len(series.timestamps)
    if current_index < 0 or current_index >= count:
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_current_pose_not_found",
            f"current pose index {current_index} is outside [0,{count})",
        )
    divisor = 1e9 if series.timestamp_unit == "nanosecond" else 1e3
    current_time = float(series.timestamps[current_index])
    end_time = current_time + float(horizon_seconds) * divisor
    end_index = int(np.searchsorted(series.timestamps, end_time, side="left"))
    if end_index >= count:
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_full_horizon_missing",
            "raw pose series ends before the 4-second target horizon",
        )
    selection = slice(current_index, end_index + 1)
    positions = np.asarray(series.positions_xy[selection], dtype=np.float64)
    yaws = np.asarray(series.yaws[selection], dtype=np.float64)
    times = (np.asarray(series.timestamps[selection], dtype=np.float64) - current_time) / divisor
    pose_valid = np.asarray(series.pose_valid[selection], dtype=np.bool_)
    if not pose_valid.all():
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_raw_pose_invalid",
            "selected raw pose window contains an explicitly invalid odometry frame",
        )
    if len(times) < 2 or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(yaws)):
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_raw_pose_non_finite",
            "selected raw pose window contains missing, NaN, or Inf values",
        )
    if not np.all(np.diff(times) > 0.0):
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_time_not_strictly_increasing",
            "selected raw pose timestamps must be strictly increasing",
        )
    yaw0 = float(yaws[0])
    base_yaw = yaw0 + math.radians(float(yaw_offset_deg))
    delta = positions - positions[0]
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    x_forward = delta[:, 0] * cos_yaw + delta[:, 1] * sin_yaw
    y_left = -delta[:, 0] * sin_yaw + delta[:, 1] * cos_yaw
    unwrapped = np.unwrap(yaws)
    yaw_relative = np.asarray([_wrap_angle(value - unwrapped[0]) for value in unwrapped])
    return tuple(
        (
            float(x_forward[index]),
            float(y_left[index]),
            float(yaw_relative[index]),
            float(times[index]),
        )
        for index in range(len(times))
    )


def _orientation_order(clip_dir: Path, *, dataset: str) -> str:
    """Resolve quaternion order from extraction metadata or dataset contract."""

    meta_path = clip_dir / "mcap_extract_meta.json"
    if meta_path.is_file():
        try:
            value = str(
                json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "frame_orientations_order"
                )
                or ""
            ).lower()
        except (OSError, json.JSONDecodeError):
            value = ""
        if value in {"xyzw", "wxyz"}:
            return value
    if dataset == "sidewalk":
        return "wxyz"
    raise StageIIRawTrajectoryErrorV2(
        "trajectory_orientation_order_missing",
        "Foxglove frame_orientations_order must be recorded in extraction metadata",
    )


def _quaternion_rows_to_yaw(rows: np.ndarray, *, order: str) -> np.ndarray:
    """Normalize quaternion rows and return full 3D yaw."""

    values = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    norms = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(norms) & (norms > 1e-8)
    normalized = np.full_like(values, np.nan)
    normalized[valid] = values[valid] / norms[valid, None]
    if order == "xyzw":
        x, y, z, w = normalized.T
    elif order == "wxyz":
        w, x, y, z = normalized.T
    else:  # pragma: no cover - protected by adapter
        raise ValueError(f"unsupported quaternion order: {order}")
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _positions_are_globally_static(positions: np.ndarray) -> bool:
    """Reject whole-clip zero/static exports, not legitimate local stops."""

    values = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    finite = values[np.isfinite(values).all(axis=1)]
    if len(finite) < 2:
        return True
    path_length = float(np.linalg.norm(np.diff(finite, axis=0), axis=1).sum())
    displacement = float(np.linalg.norm(finite[-1] - finite[0]))
    return displacement <= 0.5 and path_length <= 1.0


def _reject_egowalk_backward_discontinuity(
    rows: tuple[tuple[float, float, float, float], ...],
) -> None:
    """Preserve the original EgoWalk forward-continuity segmentation gate."""

    for left, right in zip(rows, rows[1:]):
        dx = float(right[0] - left[0])
        dy = float(right[1] - left[1])
        distance = math.hypot(dx, dy)
        if distance < 0.01:
            continue
        forward_dot = dx * math.cos(float(left[2])) + dy * math.sin(float(left[2]))
        if forward_dot < -0.01:
            raise StageIIRawTrajectoryErrorV2(
                "trajectory_source_backward_discontinuity",
                "EgoWalk 4-second window crosses an original continuity boundary",
            )


def _coherent_declared_source(row: Mapping[str, Any]) -> str:
    """Require prepared trajectory source stamps to agree."""

    values = {
        str(value).strip()
        for value in (
            _field_value(row, "trajectory_policy.trajectory_source"),
            _field_value(row, "sidewalk_meta.trajectory_source"),
        )
        if value is not None and str(value).strip()
    }
    if len(values) != 1:
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_source_conflict",
            f"prepared trajectory source stamps disagree: {sorted(values)}",
        )
    return next(iter(values))


def _field_value(row: Mapping[str, Any], path: str) -> Any:
    """Resolve one dot-separated mapping path."""

    value: Any = row
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _nonnegative_int(value: Any) -> int:
    """Parse one required non-negative integer identity field."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_identity_missing",
            f"required integer identity is invalid: {value!r}",
        ) from exc
    if parsed < 0:
        raise StageIIRawTrajectoryErrorV2(
            "trajectory_identity_missing",
            f"required integer identity is negative: {parsed}",
        )
    return parsed


def _wrap_angle(value: float) -> float:
    """Wrap one angle into ``[-pi, pi)``."""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "EGOWALK_RAW_ADAPTER_V2",
    "FOXGLOVE_RAW_ADAPTER_V2",
    "SIDEWALK_RAW_ADAPTER_V2",
    "StageIIRawTrajectoryErrorV2",
    "StageIIRawTrajectoryReaderV2",
    "StageIIRawTrajectoryWindowV2",
]
