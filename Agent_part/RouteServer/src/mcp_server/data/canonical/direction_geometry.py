"""Construct private continuous direction-geometry sidecars.

The canonical direction stamp deliberately stores a discrete candidate center.
This module preserves the upstream continuous angle separately, so a 12-way
training target remains backward compatible while sampling and evaluation can
reason about a sample's distance to a quantization boundary.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from mcp_server.contracts.direction_profile import (
    quantize_direction_angle,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionGeometrySupervisionV1,
    DirectionSupervisionV4,
)


DIRECTION_GEOMETRY_SIDECAR_DERIVATION_VERSION = (
    "direction_geometry_sidecar_quantizer_v1"
)
DEFAULT_BOUNDARY_AMBIGUITY_THRESHOLD_DEG = 2.0


def build_direction_geometry_supervision_v1(
    row: Mapping[str, Any],
    *,
    direction: DirectionSupervisionV4,
    boundary_ambiguity_threshold_deg: float = DEFAULT_BOUNDARY_AMBIGUITY_THRESHOLD_DEG,
) -> DirectionGeometrySupervisionV1:
    """Build one private geometry sidecar from recorded upstream fields.

    The function never reconstructs an angle from a label center.  It accepts
    only a recorded continuous value such as ``direction_raw_angle_deg``;
    absent raw geometry is represented explicitly as ``unavailable``.  This
    prevents ``left_50plus`` from being silently treated as exactly ``-50``.

    Args:
        row: Raw/projected row that may retain direction geometry fields.
        direction: Validated canonical discrete direction stamp.
        boundary_ambiguity_threshold_deg: Positive distance-to-boundary gate.

    Returns:
        Immutable geometry sidecar consistent with ``direction``.

    Raises:
        ValueError: If a retained raw angle conflicts with the discrete label
            or an upstream numeric value is non-finite.
    """

    threshold = float(boundary_ambiguity_threshold_deg)
    if not 0.0 < threshold <= 45.0:
        raise ValueError("boundary_ambiguity_threshold_deg must be within (0, 45]")
    profile = resolve_stage_a_direction_profile(direction.profile_id)
    raw_angle = _optional_finite_float(
        _first_present(
            row,
            "direction_raw_angle_deg",
            "raw_angle_deg",
        ),
        field="direction_raw_angle_deg",
    )
    displacement = _optional_finite_float(
        _first_present(
            row,
            "direction_displacement",
            "displacement",
            "direction_projection.displacement",
            "direction_projection.displacement_m",
        ),
        field="direction_displacement",
        nonfinite_as_none=True,
    )
    is_stop = str(direction.label_name) == "stop"
    if is_stop:
        if raw_angle is not None:
            raise ValueError("stop direction cannot carry direction_raw_angle_deg")
        return DirectionGeometrySupervisionV1(
            profile_id=direction.profile_id,
            profile_version=direction.profile_version,
            label_index=direction.label_index,
            label_name=direction.label_name,
            is_stop=True,
            raw_angle_deg=None,
            displacement=displacement,
            stop_threshold=float(profile["stop_threshold"]),
            coordinate_frame=str(profile["coordinate_frame"]),
            coordinate_unit=str(profile["coordinate_unit"]),
            nearest_boundary_deg=None,
            boundary_distance_deg=None,
            boundary_ambiguity_threshold_deg=threshold,
            confidence="stop",
            angle_valid=False,
            confidence_weight=1.0,
            stop_rule_verified=(
                displacement is not None
                and displacement < float(profile["stop_threshold"])
            ),
            source=direction.source,
            derivation_version=DIRECTION_GEOMETRY_SIDECAR_DERIVATION_VERSION,
        )
    if raw_angle is None:
        return DirectionGeometrySupervisionV1(
            profile_id=direction.profile_id,
            profile_version=direction.profile_version,
            label_index=direction.label_index,
            label_name=direction.label_name,
            is_stop=False,
            raw_angle_deg=None,
            displacement=displacement,
            stop_threshold=float(profile["stop_threshold"]),
            coordinate_frame=str(profile["coordinate_frame"]),
            coordinate_unit=str(profile["coordinate_unit"]),
            nearest_boundary_deg=None,
            boundary_distance_deg=None,
            boundary_ambiguity_threshold_deg=threshold,
            confidence="unavailable",
            angle_valid=False,
            confidence_weight=0.0,
            stop_rule_verified=(
                displacement is not None
                and displacement >= float(profile["stop_threshold"])
            ),
            source=direction.source,
            derivation_version=DIRECTION_GEOMETRY_SIDECAR_DERIVATION_VERSION,
        )
    quantized = quantize_direction_angle(direction.profile_id, raw_angle)
    if (quantized.label_index, quantized.label_name) != (
        direction.label_index,
        str(direction.label_name),
    ):
        raise ValueError(
            "direction_raw_angle_deg quantizes to a different strong label: "
            f"raw={raw_angle}, expected={direction.label_name}, "
            f"actual={quantized.label_name}"
        )
    boundary_distance = quantized.boundary_distance_deg
    if boundary_distance is None:
        confidence = "high_confidence"
        confidence_weight = 1.0
    elif boundary_distance < threshold:
        confidence = "boundary_ambiguous"
        confidence_weight = 0.25
    else:
        confidence = "high_confidence"
        confidence_weight = 1.0
    return DirectionGeometrySupervisionV1(
        profile_id=direction.profile_id,
        profile_version=direction.profile_version,
        label_index=direction.label_index,
        label_name=direction.label_name,
        is_stop=False,
        raw_angle_deg=raw_angle,
        displacement=displacement,
        stop_threshold=float(profile["stop_threshold"]),
        coordinate_frame=str(profile["coordinate_frame"]),
        coordinate_unit=str(profile["coordinate_unit"]),
        nearest_boundary_deg=quantized.nearest_boundary_deg,
        boundary_distance_deg=boundary_distance,
        boundary_ambiguity_threshold_deg=threshold,
        confidence=confidence,
        angle_valid=True,
        confidence_weight=confidence_weight,
        stop_rule_verified=(
            displacement is not None
            and displacement >= float(profile["stop_threshold"])
        ),
        source=direction.source,
        derivation_version=DIRECTION_GEOMETRY_SIDECAR_DERIVATION_VERSION,
    )


def direction_geometry_distribution_report(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize geometry availability and boundary confidence deterministically.

    Args:
        rows: Rows carrying a nested ``direction_geometry`` sidecar.

    Returns:
        JSON-compatible dataset report.  Missing sidecars are counted instead
        of being silently inferred from discrete candidate centers.
    """

    counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    boundary_distances: list[float] = []
    for row in rows:
        geometry = row.get("direction_geometry")
        if not isinstance(geometry, Mapping):
            counts["geometry:missing"] += 1
            continue
        typed = DirectionGeometrySupervisionV1.model_validate(geometry)
        counts["geometry:present"] += 1
        counts[f"confidence:{typed.confidence}"] += 1
        counts[f"angle_valid:{str(typed.angle_valid).lower()}"] += 1
        counts[f"stop_rule_verified:{str(typed.stop_rule_verified).lower()}"] += 1
        labels[str(typed.label_name)] += 1
        if typed.boundary_distance_deg is not None:
            boundary_distances.append(float(typed.boundary_distance_deg))
    quantiles = _quantiles(boundary_distances)
    return {
        "schema_version": "urbannav_direction_geometry_distribution_report_v1",
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "label_counts": dict(sorted(labels.items())),
        "boundary_distance_deg": quantiles,
    }


def _first_present(row: Mapping[str, Any], *paths: str) -> Any:
    """Read the first present dotted-path value without mutating a row."""

    for path in paths:
        current: Any = row
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return None


def _optional_finite_float(
    value: Any,
    *,
    field: str,
    nonfinite_as_none: bool = False,
) -> float | None:
    """Normalize one optional finite numeric field."""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(result) and nonfinite_as_none:
        # 位移只服务于 stop-rule 审计；历史 NaN 不能覆盖已保留的有效 raw angle。
        return None
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    """Return stable summary quantiles without a NumPy dependency."""

    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = int(round((len(ordered) - 1) * fraction))
        return float(ordered[index])

    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p50": at(0.50),
        "p90": at(0.90),
        "max": float(ordered[-1]),
    }


__all__ = [
    "DEFAULT_BOUNDARY_AMBIGUITY_THRESHOLD_DEG",
    "DIRECTION_GEOMETRY_SIDECAR_DERIVATION_VERSION",
    "build_direction_geometry_supervision_v1",
    "direction_geometry_distribution_report",
]
