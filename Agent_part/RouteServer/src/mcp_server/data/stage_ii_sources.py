"""Frozen dataset-source adaptation for UrbanNav Stage II trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.stage_ii_cfm import (
    StageIITrajectorySourceRegistryV1,
    StageIITrajectorySourceSpecV1,
)


@dataclass(frozen=True)
class StageIIAdaptedTrajectorySourceV1:
    """Carry one registry-validated raw trajectory into the canonical builder.

    Args:
        sample_id: Unique dataset parent identity.
        source_group_id: Stable per-parent source identity.
        split_group_id: Leakage-safe dataset/clip identity.
        dataset_source: Canonical dataset family.
        raw_trajectory: Timed source rows after the frozen origin policy.
        action_mask: Explicit valid-source prefix after the frozen mask policy.
        source_format: Exact row layout.
        trajectory_source: Dataset-owned geometry provenance.
        yaw_source: Odometry or deterministic path-tangent authority.
        mask_provenance: Exact policy that produced the source mask.
        origin_provenance: Exact policy that handled current ego origin.
        source_horizon_seconds: Last valid source timestamp.
    """

    sample_id: str
    source_group_id: str
    split_group_id: str
    dataset_source: str
    raw_trajectory: tuple[tuple[Any, ...], ...]
    action_mask: tuple[bool, ...]
    source_format: str
    trajectory_source: str
    yaw_source: str
    mask_provenance: str
    origin_provenance: str
    source_horizon_seconds: float


class StageIISourceAdapterErrorV1(ValueError):
    """Carry a stable source-adaptation failure code.

    Args:
        code: Stable lower-snake-case audit reason.
        detail: Human-readable bounded diagnostic.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def load_stage_ii_trajectory_source_registry_v1(
    path: str | Path,
) -> StageIITrajectorySourceRegistryV1:
    """Load and validate one immutable Stage II source registry.

    Args:
        path: Registry JSON path.

    Returns:
        Fully validated frozen registry.

    Raises:
        FileNotFoundError: If the registry does not exist.
        ValidationError: If schema or digest validation fails.
    """

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return StageIITrajectorySourceRegistryV1.model_validate(payload)


def adapt_stage_ii_trajectory_source_row_v1(
    row: Mapping[str, Any],
    spec: StageIITrajectorySourceSpecV1,
) -> StageIIAdaptedTrajectorySourceV1:
    """Adapt one real dataset row according to its exact frozen registry entry.

    Args:
        row: Source-manifest row.
        spec: Frozen dataset source specification.

    Returns:
        Compact source payload accepted by the canonical trajectory builder.

    Raises:
        StageIISourceAdapterErrorV1: If identity, stamps, mask, origin, or
            provenance do not match the registry.
    """

    if spec.status != "enabled":
        raise StageIISourceAdapterErrorV1(
            "trajectory_dataset_disabled",
            f"dataset={spec.dataset_source} is disabled: {spec.disabled_reason}",
        )
    _validate_required_values(row, spec.required_values)
    identity = _identity_from_fields(row, spec.sample_identity_fields)
    split_identity = _identity_from_fields(row, spec.split_group_fields)
    sample_id = f"{spec.dataset_source}:{identity}"
    split_group_id = f"{spec.dataset_source}:{split_identity}"
    raw_value = _field_value(row, spec.trajectory_field)
    if not isinstance(raw_value, list):
        raise StageIISourceAdapterErrorV1(
            "trajectory_field_invalid",
            f"{spec.trajectory_field} must be a list",
        )
    raw_rows = [list(item) if isinstance(item, (list, tuple)) else item for item in raw_value]
    trajectory_source = _coherent_source_value(row, spec.trajectory_source_fields)
    if trajectory_source not in set(spec.allowed_trajectory_sources):
        raise StageIISourceAdapterErrorV1(
            "trajectory_source_not_allowed",
            f"trajectory_source={trajectory_source!r} is not admitted for {spec.dataset_source}",
        )

    # 缺失 mask 只能由 registry 中逐数据集冻结的 timed-row policy 补出；
    # 任何未注册 row 仍 fail closed，不能退化为通用的 all-true 猜测。
    if spec.mask_policy == "explicit_field_v1":
        mask_value = _field_value(row, str(spec.action_mask_field))
        if not isinstance(mask_value, list):
            raise StageIISourceAdapterErrorV1(
                "trajectory_mask_missing",
                f"{spec.action_mask_field} must contain the explicit source mask",
            )
        source_mask = list(mask_value)
    else:
        if row.get("action_mask") is not None:
            raise StageIISourceAdapterErrorV1(
                "trajectory_mask_contract_drift",
                "validated_timed_rows_v1 does not accept an unstamped action_mask override",
            )
        source_mask = [True] * len(raw_rows)

    if spec.origin_policy == "require_explicit_origin_v1":
        _validate_explicit_origin(raw_rows, source_format=spec.source_format)
    else:
        _validate_future_only_rows(raw_rows, source_format=spec.source_format)
        raw_rows.insert(0, [0.0, 0.0, 0.0, 0.0])
        source_mask.insert(0, True)
    if len(source_mask) != len(raw_rows):
        raise StageIISourceAdapterErrorV1(
            "trajectory_mask_length_mismatch",
            "source mask length must equal trajectory length after origin policy",
        )
    source_horizon = _source_time(raw_rows[-1], source_format=spec.source_format)
    if not math.isfinite(source_horizon) or source_horizon <= 0.0:
        raise StageIISourceAdapterErrorV1(
            "trajectory_source_horizon_invalid",
            "last source timestamp must be finite and positive",
        )
    return StageIIAdaptedTrajectorySourceV1(
        sample_id=sample_id,
        source_group_id=sample_id,
        split_group_id=split_group_id,
        dataset_source=str(spec.dataset_source),
        raw_trajectory=tuple(tuple(item) for item in raw_rows),
        action_mask=tuple(source_mask),
        source_format=spec.source_format,
        trajectory_source=trajectory_source,
        yaw_source=spec.yaw_source,
        mask_provenance=spec.mask_policy,
        origin_provenance=spec.origin_policy,
        source_horizon_seconds=float(source_horizon),
    )


def stage_ii_source_row_identity_v1(
    row: Mapping[str, Any],
    spec: StageIITrajectorySourceSpecV1,
) -> tuple[str, str]:
    """Resolve parent and split identities before full trajectory adaptation.

    Args:
        row: Source-manifest row.
        spec: Frozen dataset source specification.

    Returns:
        ``(sample_id, split_group_id)`` with the canonical dataset prefix.
    """

    identity = _identity_from_fields(row, spec.sample_identity_fields)
    split_identity = _identity_from_fields(row, spec.split_group_fields)
    return (
        f"{spec.dataset_source}:{identity}",
        f"{spec.dataset_source}:{split_identity}",
    )


def stage_ii_assign_split_v1(
    split_group_id: str,
    registry: StageIITrajectorySourceRegistryV1,
) -> str:
    """Assign one whole group with the frozen Stage I SHA-256 policy.

    Args:
        split_group_id: Dataset-prefixed leakage-safe group identity.
        registry: Registry carrying the exact seed and ratios.

    Returns:
        One of ``train``, ``validation``, or ``test``.
    """

    policy = registry.split_policy
    digest = hashlib.sha256(
        f"{policy.seed}:{split_group_id}".encode("utf-8")
    ).hexdigest()
    fraction = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if fraction < policy.train:
        return "train"
    if fraction < policy.train + policy.validation:
        return "validation"
    return "test"


def _validate_required_values(
    row: Mapping[str, Any],
    required_values: Mapping[str, Any],
) -> None:
    """Require exact source-schema stamps before using any trajectory value."""

    for path, expected in required_values.items():
        actual = _field_value(row, path)
        if actual != expected:
            raise StageIISourceAdapterErrorV1(
                "trajectory_source_stamp_mismatch",
                f"{path} expected {expected!r}, got {actual!r}",
            )


def _identity_from_fields(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Join exact required row paths into one stable identity."""

    values: list[str] = []
    for path in fields:
        value = _field_value(row, path)
        if value is None or isinstance(value, (dict, list, tuple)):
            raise StageIISourceAdapterErrorV1(
                "trajectory_identity_missing",
                f"identity field {path} is missing or non-scalar",
            )
        rendered = str(value).strip()
        if not rendered:
            raise StageIISourceAdapterErrorV1(
                "trajectory_identity_missing",
                f"identity field {path} is empty",
            )
        values.append(rendered)
    return ":".join(values)


def _field_value(row: Mapping[str, Any], path: str) -> Any:
    """Resolve one dot-separated mapping path without array traversal."""

    value: Any = row
    for component in str(path).split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _coherent_source_value(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Require all present source stamps to agree exactly."""

    values = {
        str(value).strip()
        for field in fields
        if (value := _field_value(row, field)) is not None and str(value).strip()
    }
    if not values:
        raise StageIISourceAdapterErrorV1(
            "trajectory_source_missing",
            f"none of the source fields are present: {list(fields)}",
        )
    if len(values) != 1:
        raise StageIISourceAdapterErrorV1(
            "trajectory_source_conflict",
            f"trajectory source stamps disagree: {sorted(values)}",
        )
    return next(iter(values))


def _validate_explicit_origin(rows: Sequence[Any], *, source_format: str) -> None:
    """Require a finite current-ego zero pose at source time zero."""

    if not rows or not isinstance(rows[0], (list, tuple)) or len(rows[0]) != 4:
        raise StageIISourceAdapterErrorV1(
            "trajectory_origin_missing",
            "source must begin with one width-4 current-origin row",
        )
    try:
        x = float(rows[0][0])
        y = float(rows[0][1])
        time_sec = _source_time(rows[0], source_format=source_format)
    except (TypeError, ValueError) as exc:
        raise StageIISourceAdapterErrorV1(
            "trajectory_origin_invalid",
            "source origin must be numeric",
        ) from exc
    if not all(math.isfinite(value) for value in (x, y, time_sec)):
        raise StageIISourceAdapterErrorV1(
            "trajectory_origin_invalid",
            "source origin must be finite",
        )
    if abs(x) > 1e-6 or abs(y) > 1e-6 or abs(time_sec) > 1e-6:
        raise StageIISourceAdapterErrorV1(
            "trajectory_origin_not_current_ego",
            "source origin must be [0,0] at t=0",
        )


def _validate_future_only_rows(rows: Sequence[Any], *, source_format: str) -> None:
    """Require future-only rows before prepending the known current ego origin."""

    if not rows or not isinstance(rows[0], (list, tuple)) or len(rows[0]) != 4:
        raise StageIISourceAdapterErrorV1(
            "trajectory_future_rows_missing",
            "future-only source must contain width-4 rows",
        )
    try:
        first_time = _source_time(rows[0], source_format=source_format)
    except (TypeError, ValueError) as exc:
        raise StageIISourceAdapterErrorV1(
            "trajectory_future_time_invalid",
            "first future timestamp must be numeric",
        ) from exc
    if not math.isfinite(first_time) or first_time <= 0.0:
        raise StageIISourceAdapterErrorV1(
            "trajectory_future_time_invalid",
            "future-only source must start strictly after t=0",
        )


def _source_time(row: Sequence[Any], *, source_format: str) -> float:
    """Return the time component for one registered source format."""

    index = 2 if source_format == "ego_xy_t_speed_right_front_v1" else 3
    return float(row[index])


__all__ = [
    "StageIIAdaptedTrajectorySourceV1",
    "StageIISourceAdapterErrorV1",
    "adapt_stage_ii_trajectory_source_row_v1",
    "load_stage_ii_trajectory_source_registry_v1",
    "stage_ii_assign_split_v1",
    "stage_ii_source_row_identity_v1",
]
