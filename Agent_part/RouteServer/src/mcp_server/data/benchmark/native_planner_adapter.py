"""Offline adapters that join native localization with independent planner logs.

The two-stage join is deliberate: a recorded GPS track can establish the
robot's current location, but it is not itself a route instruction.  Only a
separate planner response with typed provenance becomes public waypoint
guidance after this module emits ``PlannerRouteRecordWgs84V1``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mcp_server.contracts.language_waypoint_benchmark import (
    NativePlannerAdapterConfigV1,
    NativePlannerAnchorV1,
    NativePlannerRouteLogV1,
    PlannerRouteRecordWgs84V1,
    WaypointGuidanceProvenanceV1,
    Wgs84CoordinateV1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic


def build_foxglove_native_gps_anchors_v1(
    *,
    source_jsonl: str | Path,
    foxglove_raw_root: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    heading_lookback_frames: int = 120,
    min_heading_displacement_m: float = 0.25,
    timestamp_scale_to_seconds: float = 1e-9,
) -> dict[str, Any]:
    """Build current-pose anchors from Foxglove native GPS files.

    The heading is estimated from an earlier non-stationary GPS observation,
    never from a future frame.  ``frame_gps.npy`` therefore contributes only
    current localization.  It must still be joined with a separate independent
    planner ledger before waypoint guidance can exist.

    Args:
        source_jsonl: Canonical Reasoning V4 source manifest.
        foxglove_raw_root: Directory containing ``{clip}/frame_gps.npy`` and
            ``{clip}/frame_times.npy`` files.
        output_jsonl: Destination for one anchor status row per Foxglove source.
        report_json: Deterministic audit report destination.
        heading_lookback_frames: Maximum prior raw frames considered for heading.
        min_heading_displacement_m: Minimum past-to-current GPS displacement.
        timestamp_scale_to_seconds: Explicit conversion from raw frame time units.

    Returns:
        Counts, source fingerprints, and availability diagnostics.

    Raises:
        ValueError: If the requested pose policy is not physically meaningful.
    """

    if heading_lookback_frames < 1:
        raise ValueError("heading_lookback_frames must be positive")
    if min_heading_displacement_m <= 0.0 or not math.isfinite(min_heading_displacement_m):
        raise ValueError("min_heading_displacement_m must be positive and finite")
    if timestamp_scale_to_seconds <= 0.0 or not math.isfinite(timestamp_scale_to_seconds):
        raise ValueError("timestamp_scale_to_seconds must be positive and finite")

    raw_root = Path(foxglove_raw_root)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    anchors: list[NativePlannerAnchorV1] = []
    counts: Counter[str] = Counter()
    seen_parent_ids: set[str] = set()
    for payload in iter_jsonl(source_jsonl):
        source = ReasoningSourceV4.model_validate(payload)
        observation = source.observation
        if observation.dataset_source != "foxglove":
            counts["dataset_source_not_foxglove"] += 1
            continue
        parent_id = observation.source_group_id
        if parent_id in seen_parent_ids:
            raise ValueError(f"duplicate source_group_id: {parent_id}")
        seen_parent_ids.add(parent_id)
        clip_id, frame_index = _parse_foxglove_source_record_id_v1(observation.source_record_id)
        if clip_id is None or frame_index is None:
            anchors.append(
                _unavailable_anchor_v1(
                    parent_sample_id=parent_id,
                    dataset_source=observation.dataset_source,
                    source_record_id=observation.source_record_id,
                    reason="source_record_id_unparseable",
                )
            )
            counts["source_record_id_unparseable"] += 1
            continue
        try:
            arrays = cache.get(clip_id)
            if arrays is None:
                arrays = _load_foxglove_native_arrays_v1(raw_root=raw_root, clip_id=clip_id)
                cache[clip_id] = arrays
            gps, times = arrays
        except (OSError, ValueError) as exc:
            reason = str(exc)
            anchors.append(
                _unavailable_anchor_v1(
                    parent_sample_id=parent_id,
                    dataset_source=observation.dataset_source,
                    source_record_id=observation.source_record_id,
                    reason=reason,
                    raw_clip_id=clip_id,
                    raw_frame_index=frame_index,
                )
            )
            counts[reason] += 1
            continue
        if frame_index >= len(gps) or frame_index >= len(times):
            anchors.append(
                _unavailable_anchor_v1(
                    parent_sample_id=parent_id,
                    dataset_source=observation.dataset_source,
                    source_record_id=observation.source_record_id,
                    reason="raw_frame_index_out_of_range",
                    raw_clip_id=clip_id,
                    raw_frame_index=frame_index,
                )
            )
            counts["raw_frame_index_out_of_range"] += 1
            continue
        latitude_deg, longitude_deg = float(gps[frame_index, 0]), float(gps[frame_index, 1])
        timestamp_s = float(times[frame_index]) * timestamp_scale_to_seconds
        if not _valid_wgs84_v1(latitude_deg, longitude_deg) or not math.isfinite(timestamp_s):
            anchors.append(
                _unavailable_anchor_v1(
                    parent_sample_id=parent_id,
                    dataset_source=observation.dataset_source,
                    source_record_id=observation.source_record_id,
                    reason="native_gps_or_timestamp_invalid",
                    raw_clip_id=clip_id,
                    raw_frame_index=frame_index,
                    observation_timestamp_s=max(0.0, timestamp_s) if math.isfinite(timestamp_s) else 0.0,
                )
            )
            counts["native_gps_or_timestamp_invalid"] += 1
            continue
        heading_rad = _past_gps_heading_rad_v1(
            gps,
            frame_index=frame_index,
            lookback_frames=heading_lookback_frames,
            min_displacement_m=min_heading_displacement_m,
        )
        if heading_rad is None:
            anchors.append(
                NativePlannerAnchorV1(
                    anchor_id=_stable_id_v1("native_planner_anchor", parent_id),
                    parent_sample_id=parent_id,
                    dataset_source=observation.dataset_source,
                    source_record_id=observation.source_record_id,
                    observation_timestamp_s=timestamp_s,
                    current_wgs84=Wgs84CoordinateV1(
                        latitude_deg=latitude_deg,
                        longitude_deg=longitude_deg,
                    ),
                    localization_source="unavailable",
                    raw_clip_id=clip_id,
                    raw_frame_index=frame_index,
                    available=False,
                    reason="native_past_heading_unavailable",
                )
            )
            counts["native_past_heading_unavailable"] += 1
            continue
        anchors.append(
            NativePlannerAnchorV1(
                anchor_id=_stable_id_v1("native_planner_anchor", parent_id),
                parent_sample_id=parent_id,
                dataset_source=observation.dataset_source,
                source_record_id=observation.source_record_id,
                observation_timestamp_s=timestamp_s,
                current_wgs84=Wgs84CoordinateV1(
                    latitude_deg=latitude_deg,
                    longitude_deg=longitude_deg,
                ),
                current_heading_rad=heading_rad,
                localization_source="native_gps_past_bearing",
                raw_clip_id=clip_id,
                raw_frame_index=frame_index,
                available=True,
                reason="native_gps_anchor_available",
            )
        )
        counts["native_gps_anchor_available"] += 1
    write_jsonl_atomic(output_jsonl, (anchor.model_dump(mode="json") for anchor in anchors))
    report = {
        "schema_version": "urbannav_foxglove_native_gps_anchor_report_v1",
        "source_count": len(seen_parent_ids),
        "anchor_count": len(anchors),
        "available_count": sum(anchor.available for anchor in anchors),
        "unavailable_count": sum(not anchor.available for anchor in anchors),
        "status_counts": dict(sorted(counts.items())),
        "heading_lookback_frames": heading_lookback_frames,
        "min_heading_displacement_m": min_heading_displacement_m,
        "timestamp_scale_to_seconds": timestamp_scale_to_seconds,
        "source_jsonl_sha256": _artifact_sha256_v1(source_jsonl),
        "raw_root": str(raw_root),
    }
    _write_json_v1(report_json, report)
    return report


def build_foxglove_native_gps_anchors_from_stage_i_views_v1(
    *,
    source_jsonl: str | Path,
    foxglove_raw_root: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    heading_lookback_frames: int = 120,
    min_heading_displacement_m: float = 0.25,
    timestamp_scale_to_seconds: float = 1e-9,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build current-pose anchors from compiled FullCoT Stage I views.

    The existing source adapter consumes the pre-teacher V4 source envelope.
    This entry point covers the already compiled Stage I view available in the
    current local artifact set.  It reads only the physical source identity
    and native past/current GPS arrays; the view's private trajectory fields
    are never inspected.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    raw_root = Path(foxglove_raw_root)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    anchors: list[NativePlannerAnchorV1] = []
    counts: Counter[str] = Counter()
    for payload in iter_jsonl(source_jsonl):
        if payload.get("dataset_source") != "foxglove":
            continue
        if limit is not None and len(anchors) >= limit:
            break
        source_group_id = str(payload.get("source_group_id") or "")
        prefix = "foxglove:"
        source_record_id = source_group_id[len(prefix):] if source_group_id.startswith(prefix) else ""
        clip_id, frame_index = _parse_foxglove_source_record_id_v1(source_record_id)
        if not source_record_id or clip_id is None or frame_index is None:
            counts["source_record_id_unparseable"] += 1
            continue
        try:
            arrays = cache.get(clip_id)
            if arrays is None:
                arrays = _load_foxglove_native_arrays_v1(raw_root=raw_root, clip_id=clip_id)
                cache[clip_id] = arrays
            gps, times = arrays
        except (OSError, ValueError) as exc:
            counts[str(exc)] += 1
            continue
        if frame_index >= len(gps) or frame_index >= len(times):
            counts["raw_frame_index_out_of_range"] += 1
            continue
        latitude_deg, longitude_deg = float(gps[frame_index, 0]), float(gps[frame_index, 1])
        timestamp_s = float(times[frame_index]) * timestamp_scale_to_seconds
        heading_rad = _past_gps_heading_rad_v1(
            gps,
            frame_index=frame_index,
            lookback_frames=heading_lookback_frames,
            min_displacement_m=min_heading_displacement_m,
        )
        if (
            not _valid_wgs84_v1(latitude_deg, longitude_deg)
            or not math.isfinite(timestamp_s)
            or heading_rad is None
        ):
            counts["native_pose_unavailable"] += 1
            continue
        anchors.append(
            NativePlannerAnchorV1(
                anchor_id=_stable_id_v1("native_planner_anchor", source_group_id),
                parent_sample_id=source_group_id,
                dataset_source="foxglove",
                source_record_id=source_record_id,
                observation_timestamp_s=timestamp_s,
                current_wgs84=Wgs84CoordinateV1(
                    latitude_deg=latitude_deg,
                    longitude_deg=longitude_deg,
                ),
                current_heading_rad=heading_rad,
                localization_source="native_gps_past_bearing",
                raw_clip_id=clip_id,
                raw_frame_index=frame_index,
                available=True,
                reason="native_gps_anchor_available",
            )
        )
        counts["native_gps_anchor_available"] += 1
    write_jsonl_atomic(output_jsonl, (anchor.model_dump(mode="json") for anchor in anchors))
    report = {
        "schema_version": "urbannav_foxglove_stage_i_view_anchor_report_v1",
        "source_jsonl": str(source_jsonl),
        "raw_root": str(raw_root),
        "source_role": "compiled_stage_i_fullcot_view",
        "target_role": "current_pose_only_before_independent_planner_join",
        "counts": dict(sorted(counts.items())),
        "anchor_count": len(anchors),
        "heading_lookback_frames": heading_lookback_frames,
        "min_heading_displacement_m": min_heading_displacement_m,
        "timestamp_scale_to_seconds": timestamp_scale_to_seconds,
        "source_jsonl_sha256": _artifact_sha256_v1(source_jsonl),
    }
    _write_json_v1(report_json, report)
    return report


def build_planner_records_from_native_logs_v1(
    *,
    anchor_jsonl: str | Path,
    planner_log_jsonl: str | Path,
    output_jsonl: str | Path,
    decisions_jsonl: str | Path,
    report_json: str | Path,
    config: NativePlannerAdapterConfigV1 | None = None,
) -> dict[str, Any]:
    """Join typed independent planner logs to timestamped native anchors.

    A planner response is accepted only when it names the same dataset/native
    source record, is no newer than the observation, is not stale, and agrees
    with the localization anchor in position and heading.  The function never
    reads trajectory sidecars or recorded future GPS points.

    Args:
        anchor_jsonl: Native localization anchor ledger.
        planner_log_jsonl: Independently produced typed planner-log ledger.
        output_jsonl: Accepted ``PlannerRouteRecordWgs84V1`` output ledger.
        decisions_jsonl: One deterministic accept/reject decision per anchor.
        report_json: Aggregate audit report destination.
        config: Fail-closed timing and pose-consistency thresholds.

    Returns:
        Accepted/rejected counts and immutable source fingerprints.

    Raises:
        ValueError: If either ledger has conflicting duplicate identities.
    """

    cfg = config or NativePlannerAdapterConfigV1()
    anchors = _load_unique_anchors_v1(anchor_jsonl)
    logs_by_native_key = _load_planner_logs_v1(planner_log_jsonl)
    records: list[PlannerRouteRecordWgs84V1] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for parent_id, anchor in sorted(anchors.items()):
        decision: dict[str, Any] = {
            "parent_sample_id": parent_id,
            "dataset_source": anchor.dataset_source,
            "source_record_id": anchor.source_record_id,
            "anchor_id": anchor.anchor_id,
            "status": "rejected",
        }
        if not anchor.available:
            decision["reason"] = f"anchor_{anchor.reason}"
            decisions.append(decision)
            counts[decision["reason"]] += 1
            continue
        key = (anchor.dataset_source, anchor.source_record_id)
        candidates = logs_by_native_key.get(key, [])
        if not candidates:
            decision["reason"] = "planner_log_missing"
            decisions.append(decision)
            counts[decision["reason"]] += 1
            continue
        earlier = [item for item in candidates if item.plan_timestamp_s <= anchor.observation_timestamp_s]
        if not earlier:
            decision["reason"] = "planner_log_after_observation"
            decisions.append(decision)
            counts[decision["reason"]] += 1
            continue
        log = max(earlier, key=lambda item: (item.plan_timestamp_s, item.route_log_id))
        plan_age_s = anchor.observation_timestamp_s - log.plan_timestamp_s
        decision.update({"route_log_id": log.route_log_id, "plan_age_s": plan_age_s})
        if plan_age_s > cfg.max_plan_age_s:
            decision["reason"] = "planner_record_stale"
            decisions.append(decision)
            counts[decision["reason"]] += 1
            continue
        assert anchor.current_wgs84 is not None and anchor.current_heading_rad is not None
        origin_distance_m = _wgs84_distance_m_v1(anchor.current_wgs84, log.current_wgs84)
        heading_delta_rad = _angular_delta_rad_v1(anchor.current_heading_rad, log.current_heading_rad)
        decision.update(
            {
                "origin_distance_m": origin_distance_m,
                "heading_delta_rad": heading_delta_rad,
            }
        )
        if origin_distance_m > cfg.max_origin_distance_m:
            decision["reason"] = "planner_origin_mismatch"
            decisions.append(decision)
            counts[decision["reason"]] += 1
            continue
        if heading_delta_rad > cfg.max_heading_delta_rad:
            decision["reason"] = "planner_heading_mismatch"
            decisions.append(decision)
            counts[decision["reason"]] += 1
            continue
        records.append(
            PlannerRouteRecordWgs84V1(
                record_id=_stable_id_v1("native_planner_route", parent_id, log.route_log_id),
                parent_sample_id=parent_id,
                observation_timestamp_s=anchor.observation_timestamp_s,
                current_wgs84=anchor.current_wgs84,
                current_heading_rad=anchor.current_heading_rad,
                goal_wgs84=log.goal_wgs84,
                planner_polyline_wgs84=log.planner_polyline_wgs84,
                provenance=WaypointGuidanceProvenanceV1(
                    planner_id=log.planner_id,
                    route_version=log.route_version,
                    plan_timestamp_s=log.plan_timestamp_s,
                    datum=log.datum,
                    source_kind=log.source_kind,
                    independent_of_future_target=True,
                ),
                reroute=log.reroute,
            )
        )
        decision["status"] = "accepted"
        decision["reason"] = "independent_native_planner_route_available"
        decisions.append(decision)
        counts[decision["reason"]] += 1
    write_jsonl_atomic(output_jsonl, (record.model_dump(mode="json") for record in records))
    write_jsonl_atomic(decisions_jsonl, decisions)
    report = {
        "schema_version": "urbannav_native_planner_route_adapter_report_v1",
        "anchor_count": len(anchors),
        "planner_log_count": sum(len(items) for items in logs_by_native_key.values()),
        "accepted_record_count": len(records),
        "rejected_count": len(decisions) - len(records),
        "decision_counts": dict(sorted(counts.items())),
        "config": cfg.model_dump(mode="json"),
        "anchor_jsonl_sha256": _artifact_sha256_v1(anchor_jsonl),
        "planner_log_jsonl_sha256": _artifact_sha256_v1(planner_log_jsonl),
    }
    _write_json_v1(report_json, report)
    return report


def _parse_foxglove_source_record_id_v1(source_record_id: str) -> tuple[str | None, int | None]:
    """Split a ``{clip}_{frame:06d}`` record without guessing clip internals."""

    clip_id, separator, suffix = str(source_record_id).rpartition("_")
    if not separator or not clip_id or not suffix.isdigit():
        return None, None
    return clip_id, int(suffix)


def _load_foxglove_native_arrays_v1(*, raw_root: Path, clip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only native GPS and timestamps required for a current-pose anchor."""

    clip_root = raw_root / clip_id
    gps_path = clip_root / "frame_gps.npy"
    times_path = clip_root / "frame_times.npy"
    if not gps_path.is_file() or not times_path.is_file():
        raise ValueError("native_gps_or_frame_times_missing")
    gps = np.load(gps_path, mmap_mode="r")
    times = np.load(times_path, mmap_mode="r")
    if gps.ndim != 2 or gps.shape[1] < 2 or times.ndim != 1:
        raise ValueError("native_gps_or_frame_times_schema_invalid")
    return gps, times


def _past_gps_heading_rad_v1(
    gps: np.ndarray,
    *,
    frame_index: int,
    lookback_frames: int,
    min_displacement_m: float,
) -> float | None:
    """Estimate compass heading from an earlier native GPS point only."""

    current_lat, current_lon = float(gps[frame_index, 0]), float(gps[frame_index, 1])
    if not _valid_wgs84_v1(current_lat, current_lon):
        return None
    lower = max(-1, frame_index - lookback_frames - 1)
    for prior_index in range(frame_index - 1, lower, -1):
        prior_lat, prior_lon = float(gps[prior_index, 0]), float(gps[prior_index, 1])
        if not _valid_wgs84_v1(prior_lat, prior_lon):
            continue
        prior = Wgs84CoordinateV1(latitude_deg=prior_lat, longitude_deg=prior_lon)
        current = Wgs84CoordinateV1(latitude_deg=current_lat, longitude_deg=current_lon)
        if _wgs84_distance_m_v1(prior, current) < min_displacement_m:
            continue
        north_m, east_m = _north_east_delta_m_v1(prior, current)
        return math.atan2(east_m, north_m) % (2.0 * math.pi)
    return None


def _unavailable_anchor_v1(
    *,
    parent_sample_id: str,
    dataset_source: str,
    source_record_id: str,
    reason: str,
    raw_clip_id: str | None = None,
    raw_frame_index: int | None = None,
    observation_timestamp_s: float = 0.0,
) -> NativePlannerAnchorV1:
    """Create a typed unavailable anchor instead of silently dropping a parent."""

    return NativePlannerAnchorV1(
        anchor_id=_stable_id_v1("native_planner_anchor", parent_sample_id),
        parent_sample_id=parent_sample_id,
        dataset_source=dataset_source,
        source_record_id=source_record_id,
        observation_timestamp_s=observation_timestamp_s,
        raw_clip_id=raw_clip_id,
        raw_frame_index=raw_frame_index,
        available=False,
        reason=reason,
    )


def _load_unique_anchors_v1(path: str | Path) -> dict[str, NativePlannerAnchorV1]:
    """Load native anchors keyed by canonical physical parent."""

    rows: dict[str, NativePlannerAnchorV1] = {}
    for payload in iter_jsonl(path):
        anchor = NativePlannerAnchorV1.model_validate(payload)
        if anchor.parent_sample_id in rows:
            raise ValueError(f"duplicate native planner anchor parent: {anchor.parent_sample_id}")
        rows[anchor.parent_sample_id] = anchor
    return rows


def _load_planner_logs_v1(path: str | Path) -> dict[tuple[str, str], list[NativePlannerRouteLogV1]]:
    """Load planner logs grouped by their explicit native join key."""

    grouped: dict[tuple[str, str], list[NativePlannerRouteLogV1]] = defaultdict(list)
    seen_ids: set[str] = set()
    for payload in iter_jsonl(path):
        log = NativePlannerRouteLogV1.model_validate(payload)
        if log.route_log_id in seen_ids:
            raise ValueError(f"duplicate native planner route log id: {log.route_log_id}")
        seen_ids.add(log.route_log_id)
        grouped[(log.dataset_source, log.source_record_id)].append(log)
    return grouped


def _valid_wgs84_v1(latitude_deg: float, longitude_deg: float) -> bool:
    """Return whether a finite latitude/longitude pair is within WGS84 bounds."""

    return (
        math.isfinite(latitude_deg)
        and math.isfinite(longitude_deg)
        and -90.0 <= latitude_deg <= 90.0
        and -180.0 <= longitude_deg <= 180.0
    )


def _north_east_delta_m_v1(origin: Wgs84CoordinateV1, point: Wgs84CoordinateV1) -> tuple[float, float]:
    """Approximate a nearby WGS84 displacement on a local tangent plane."""

    earth_radius_m = 6_378_137.0
    north_m = earth_radius_m * math.radians(point.latitude_deg - origin.latitude_deg)
    east_m = earth_radius_m * math.cos(math.radians(origin.latitude_deg)) * math.radians(
        point.longitude_deg - origin.longitude_deg
    )
    return north_m, east_m


def _wgs84_distance_m_v1(origin: Wgs84CoordinateV1, point: Wgs84CoordinateV1) -> float:
    """Return local tangent-plane distance for nearby planner-anchor checks."""

    north_m, east_m = _north_east_delta_m_v1(origin, point)
    return math.hypot(north_m, east_m)


def _angular_delta_rad_v1(left: float, right: float) -> float:
    """Return the unsigned wrapped delta between two compass headings."""

    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def _stable_id_v1(*parts: str) -> str:
    """Return a deterministic bounded identity for an offline adapter artifact."""

    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{parts[0]}:{digest}"


def _artifact_sha256_v1(path: str | Path) -> str:
    """Hash an immutable ledger without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_v1(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one deterministic small adapter report."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "build_foxglove_native_gps_anchors_v1",
    "build_foxglove_native_gps_anchors_from_stage_i_views_v1",
    "build_planner_records_from_native_logs_v1",
]
