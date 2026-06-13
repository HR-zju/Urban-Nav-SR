#!/usr/bin/env python3
"""Repair odometry pose files for prepared SideWalk clips without re-extracting media.

This tool is intentionally narrower than ``process_chunks_26_29.py``.  It reads
the original MCAP ``/odom`` stream, synchronizes it to the full FPV
``frame_times.npy`` timeline when available, and rewrites only odometry-derived metadata:

    trajectory.json
    frames_manifest.json
    frame_positions.npy
    frame_orientations.npy
    frame_linear_velocities.npy
    frame_angular_velocities.npy
    frame_speeds.npy
    mcap_extract_meta.json

It does not touch FPV images, ``front.mp4``, BEV images, segmentation outputs, or
CoT labels.  Use this when a prepared clip already has valid media/BEV but
``frame_positions.npy`` was accidentally written as all zeros.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mcap_to_sidewalk_inputs import (
    OdometrySample,
    _choose_optional_channel,
    _decode_chunk,
    _iter_chunk_records,
    _iter_mcap_records,
    _nearest_sample,
    _parse_channel,
    _parse_message_record,
    _parse_odometry_motion,
    _parse_schema,
)


def _positions_are_static(
    positions: np.ndarray,
    *,
    max_displacement_m: float = 0.5,
    max_path_length_m: float = 1.0,
) -> bool:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return True
    xy = arr[:, :2]
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    if xy.shape[0] < 2:
        return True
    deltas = np.diff(xy, axis=0)
    path_length = float(np.sum(np.linalg.norm(deltas, axis=1)))
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    return displacement <= max_displacement_m and path_length <= max_path_length_m


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_optional_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _resolve_frame_image_path(clip_dir: Path, frame_idx: int) -> Path | None:
    for image_dir_name in ("images", "front_images"):
        image_dir = clip_dir / image_dir_name
        if not image_dir.is_dir():
            continue
        candidates = [f"{int(frame_idx):06d}", str(int(frame_idx)), f"{int(frame_idx) + 1:06d}", str(int(frame_idx) + 1)]
        for stem in candidates:
            for suffix in (".jpg", ".jpeg", ".png"):
                path = image_dir / f"{stem}{suffix}"
                if path.is_file():
                    return path.resolve()
    return None


def _trajectory_from_full_fpv_timeline(clip_dir: Path, existing_trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a dense FPV-frame timeline for odom repair.

    Older BEV generation wrote ``trajectory.json`` at the BEV sample rate.  For
    Phase1 GT, odom arrays must be synchronized to FPV frames instead; otherwise
    future trajectory labels span BEV intervals.  Prefer full ``frame_times.npy``
    and ``frame_gps.npy`` when they are longer than the existing trajectory.
    """

    frame_times = _load_optional_array(clip_dir / "frame_times.npy")
    frame_gps = _load_optional_array(clip_dir / "frame_gps.npy")
    if frame_times is None or len(frame_times) <= len(existing_trajectory):
        return existing_trajectory

    n = int(len(frame_times))
    if frame_gps is not None and len(frame_gps) > 0:
        n = min(n, int(len(frame_gps)))

    dense: list[dict[str, Any]] = []
    front_path = clip_dir / "front.mp4"
    for idx in range(n):
        image_path = _resolve_frame_image_path(clip_dir, idx)
        row: dict[str, Any] = {
            "frame_idx": int(idx),
            "frame_name": f"{int(idx):06d}",
            "timestamp": int(frame_times[idx]),
        }
        if frame_gps is not None and idx < len(frame_gps):
            row["lat"] = float(frame_gps[idx, 0])
            row["lng"] = float(frame_gps[idx, 1])
        if image_path is not None:
            row["image_path"] = str(image_path)
        if front_path.is_file():
            row["fpv_video_path"] = str(front_path.resolve())
        dense.append(row)
    return dense


def _read_odom_samples(mcap_path: Path, *, odom_topic: str = "auto") -> tuple[list[OdometrySample], dict[str, Any]]:
    """Read only the Odometry stream from MCAP.

    The function reuses the low-level MCAP readers from ``mcap_to_sidewalk_inputs``
    so that ROS1/ROS2 serialization behavior stays identical to the full
    extractor.
    """

    schemas: dict[int, Any] = {}
    channels: dict[int, Any] = {}
    message_counts: dict[str, int] = {}

    for opcode, payload in _iter_mcap_records(mcap_path):
        if opcode == 3:
            schema = _parse_schema(payload)
            schemas[schema.schema_id] = schema
        elif opcode == 4:
            channel = _parse_channel(payload, schemas)
            channels[channel.channel_id] = channel

    odom_channel = _choose_optional_channel(
        channels,
        requested_topic=odom_topic,
        schema_substrings=["Odometry"],
        role="odom",
    )
    if odom_channel is None:
        return [], {
            "odom_topic": None,
            "odom_schema": None,
            "message_counts": message_counts,
            "error": "odom_channel_not_found",
        }

    samples: list[OdometrySample] = []
    parse_errors = 0
    for opcode, payload in _iter_mcap_records(mcap_path):
        if opcode != 6:
            continue
        chunk = _decode_chunk(payload)
        for inner_opcode, inner_payload in _iter_chunk_records(chunk):
            if inner_opcode != 5:
                continue
            channel_id, timestamp_ns, message_payload = _parse_message_record(inner_payload)
            channel = channels.get(channel_id)
            if channel is None:
                continue
            message_counts[channel.topic] = message_counts.get(channel.topic, 0) + 1
            if channel_id != odom_channel.channel_id:
                continue
            try:
                samples.append(
                    _parse_odometry_motion(
                        message_payload,
                        timestamp_ns=timestamp_ns,
                        source_topic=channel.topic,
                        source_schema=channel.schema_name,
                        message_encoding=channel.message_encoding,
                    )
                )
            except Exception:
                parse_errors += 1

    samples.sort(key=lambda item: int(item.timestamp_ns))
    return samples, {
        "odom_topic": odom_channel.topic,
        "odom_schema": odom_channel.schema_name,
        "odom_message_encoding": odom_channel.message_encoding,
        "message_counts": message_counts,
        "odom_parse_errors": parse_errors,
    }


def _odom_payload(sample: OdometrySample | None, delta_ns: int | None, *, max_delta_ns: int) -> dict[str, Any]:
    if sample is None or delta_ns is None or int(delta_ns) > int(max_delta_ns):
        return {"odom_source": "missing"}
    return {
        "odom_source": "odom",
        "odom_topic": sample.source_topic,
        "odom_schema": sample.source_schema,
        "odom_sync_delta_s": round(float(delta_ns) / 1e9, 6),
        "odom_position_x": float(sample.position_x or 0.0),
        "odom_position_y": float(sample.position_y or 0.0),
        "odom_position_z": float(sample.position_z or 0.0),
        "odom_orientation_x": float(sample.orientation_x or 0.0),
        "odom_orientation_y": float(sample.orientation_y or 0.0),
        "odom_orientation_z": float(sample.orientation_z or 0.0),
        "odom_orientation_w": float(sample.orientation_w if sample.orientation_w is not None else 1.0),
        "odom_linear_velocity_x": float(sample.linear_velocity_x or 0.0),
        "odom_linear_velocity_y": float(sample.linear_velocity_y or 0.0),
        "odom_linear_velocity_z": float(sample.linear_velocity_z or 0.0),
        "odom_angular_velocity_x": float(sample.angular_velocity_x or 0.0),
        "odom_angular_velocity_y": float(sample.angular_velocity_y or 0.0),
        "odom_angular_velocity_z": float(sample.angular_velocity_z or 0.0),
    }


def _has_odom(row: dict[str, Any]) -> bool:
    return str(row.get("odom_source") or "").lower() in {"odom", "odom_filled_nearest"}


def _fill_missing_odom_rows(rows: list[dict[str, Any]]) -> int:
    """Fill isolated missing odom rows from the nearest synchronized odom row.

    Writing zeros for missing rows creates kilometer-scale jumps after one bad
    sync, which is worse than explicitly carrying the nearest pose.  The repair
    report still records how many frames were missing and filled so these clips
    can be audited or filtered later.
    """

    valid_indices = [idx for idx, row in enumerate(rows) if str(row.get("odom_source") or "").lower() == "odom"]
    if not valid_indices:
        return 0
    odom_keys = [
        "odom_topic",
        "odom_schema",
        "odom_position_x",
        "odom_position_y",
        "odom_position_z",
        "odom_orientation_x",
        "odom_orientation_y",
        "odom_orientation_z",
        "odom_orientation_w",
        "odom_linear_velocity_x",
        "odom_linear_velocity_y",
        "odom_linear_velocity_z",
        "odom_angular_velocity_x",
        "odom_angular_velocity_y",
        "odom_angular_velocity_z",
    ]
    filled = 0
    for idx, row in enumerate(rows):
        if str(row.get("odom_source") or "").lower() != "missing":
            continue
        nearest = min(valid_indices, key=lambda valid_idx: abs(valid_idx - idx))
        source = rows[nearest]
        for key in odom_keys:
            if key in source:
                row[key] = source[key]
        row["odom_source"] = "odom_filled_nearest"
        row["odom_fill_source_frame_idx"] = int(source.get("frame_idx", nearest) or nearest)
        filled += 1
    return filled


def _heading_deg_to_ros_xyzw(heading_deg: float) -> tuple[float, float, float, float]:
    """Convert compass heading degrees into a planar ROS map quaternion.

    ``heading_deg`` follows the usual navigation convention: 0=north,
    90=east.  ROS/map yaw is 0 along +x/east and positive counter-clockwise,
    so yaw = 90 - heading.
    """

    yaw = math.radians(90.0 - float(heading_deg))
    half = 0.5 * yaw
    return 0.0, 0.0, float(math.sin(half)), float(math.cos(half))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _apply_orientation_source(row: dict[str, Any], *, orientation_source: str) -> str:
    """Update row orientation fields and return the effective source."""

    if orientation_source != "trajectory_heading":
        row["orientation_source"] = "odom_quaternion"
        return "odom_quaternion"

    heading_deg = _finite_float(row.get("heading_deg"))
    if heading_deg is None:
        row["orientation_source"] = "odom_quaternion_fallback_missing_heading"
        return "odom_quaternion_fallback_missing_heading"

    for key in ("x", "y", "z", "w"):
        odom_key = f"odom_orientation_{key}"
        if odom_key in row:
            row[f"raw_{odom_key}"] = row[odom_key]
    qx, qy, qz, qw = _heading_deg_to_ros_xyzw(heading_deg)
    row["odom_orientation_x"] = qx
    row["odom_orientation_y"] = qy
    row["odom_orientation_z"] = qz
    row["odom_orientation_w"] = qw
    row["orientation_source"] = "trajectory_heading"
    row["orientation_heading_deg"] = heading_deg
    row["orientation_yaw_ros_deg"] = 90.0 - heading_deg
    return "trajectory_heading"


def _write_odom_arrays(
    clip_dir: Path,
    trajectory: list[dict[str, Any]],
    *,
    orientation_source: str,
    dry_run: bool,
) -> dict[str, Any]:
    n = len(trajectory)
    frame_positions = np.zeros((n, 3), dtype=np.float64)
    frame_orientations = np.zeros((n, 4), dtype=np.float64)
    frame_orientations[:, 3] = 1.0
    frame_linear_velocities = np.zeros((n, 3), dtype=np.float64)
    frame_angular_velocities = np.zeros((n, 3), dtype=np.float64)
    orientation_counts: dict[str, int] = {}

    for i, row in enumerate(trajectory):
        if not _has_odom(row):
            continue
        effective_orientation_source = _apply_orientation_source(row, orientation_source=orientation_source)
        orientation_counts[effective_orientation_source] = int(orientation_counts.get(effective_orientation_source, 0)) + 1

        frame_positions[i, 0] = float(row.get("odom_position_x", 0.0) or 0.0)
        frame_positions[i, 1] = float(row.get("odom_position_y", 0.0) or 0.0)
        frame_positions[i, 2] = float(row.get("odom_position_z", 0.0) or 0.0)

        # ROS quaternion order on disk: x, y, z, w.
        frame_orientations[i, 0] = float(row.get("odom_orientation_x", 0.0) or 0.0)
        frame_orientations[i, 1] = float(row.get("odom_orientation_y", 0.0) or 0.0)
        frame_orientations[i, 2] = float(row.get("odom_orientation_z", 0.0) or 0.0)
        frame_orientations[i, 3] = float(row.get("odom_orientation_w", 1.0) or 1.0)

        frame_linear_velocities[i, 0] = float(row.get("odom_linear_velocity_x", 0.0) or 0.0)
        frame_linear_velocities[i, 1] = float(row.get("odom_linear_velocity_y", 0.0) or 0.0)
        frame_linear_velocities[i, 2] = float(row.get("odom_linear_velocity_z", 0.0) or 0.0)

        frame_angular_velocities[i, 0] = float(row.get("odom_angular_velocity_x", 0.0) or 0.0)
        frame_angular_velocities[i, 1] = float(row.get("odom_angular_velocity_y", 0.0) or 0.0)
        frame_angular_velocities[i, 2] = float(row.get("odom_angular_velocity_z", 0.0) or 0.0)

    frame_speeds = np.linalg.norm(frame_linear_velocities[:, :2], axis=1).astype(np.float64)
    if not dry_run:
        np.save(clip_dir / "frame_positions.npy", frame_positions)
        np.save(clip_dir / "frame_orientations.npy", frame_orientations)
        np.save(clip_dir / "frame_linear_velocities.npy", frame_linear_velocities)
        np.save(clip_dir / "frame_angular_velocities.npy", frame_angular_velocities)
        np.save(clip_dir / "frame_speeds.npy", frame_speeds)

    valid_positions = frame_positions[np.linalg.norm(frame_positions[:, :2], axis=1) > 1e-9]
    if len(valid_positions) >= 2:
        xy_displacement_m = float(np.linalg.norm(valid_positions[-1, :2] - valid_positions[0, :2]))
        xy_path_length_m = float(np.sum(np.linalg.norm(np.diff(valid_positions[:, :2], axis=0), axis=1)))
    else:
        xy_displacement_m = 0.0
        xy_path_length_m = 0.0
    return {
        "frame_count": n,
        "odom_frame_count": int(sum(1 for row in trajectory if _has_odom(row))),
        "xy_displacement_m": round(xy_displacement_m, 6),
        "xy_path_length_m": round(xy_path_length_m, 6),
        "speed_mean_mps": round(float(np.mean(frame_speeds)) if n else 0.0, 6),
        "speed_max_mps": round(float(np.max(frame_speeds)) if n else 0.0, 6),
        "orientation_source_requested": orientation_source,
        "orientation_source_counts": orientation_counts,
    }


def _update_frames_manifest(clip_dir: Path, trajectory: list[dict[str, Any]], *, dry_run: bool) -> None:
    records: list[dict[str, Any]] = []
    existing = _load_json(clip_dir / "frames_manifest.json", [])
    existing_by_idx: dict[int, dict[str, Any]] = {}
    if isinstance(existing, list):
        for item in existing:
            if not isinstance(item, dict):
                continue
            try:
                existing_by_idx[int(item.get("frame_idx", len(existing_by_idx)))] = item
            except Exception:
                continue

    for i, row in enumerate(trajectory):
        frame_idx = int(row.get("frame_idx", i) or i)
        image_path = row.get("image_path") or str(clip_dir / "images" / f"{frame_idx:06d}.jpg")
        record = dict(existing_by_idx.get(frame_idx, {}))
        record.update(
            {
                "frame_idx": frame_idx,
                "frame_name": str(row.get("frame_name") or Path(str(image_path)).stem),
                "timestamp": int(row["timestamp"]),
                "image_path": str(image_path),
            }
        )
        record["odom"] = {
            key: value
            for key, value in row.items()
            if key.startswith("odom_")
        }
        records.append(record)

    if not dry_run:
        (clip_dir / "frames_manifest.json").write_text(
            json.dumps(records, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


def _update_extract_meta(
    clip_dir: Path,
    *,
    mcap_path: Path,
    odom_meta: dict[str, Any],
    repair_stats: dict[str, Any],
    dry_run: bool,
) -> None:
    path = clip_dir / "mcap_extract_meta.json"
    meta = _load_json(path, {})
    if not isinstance(meta, dict):
        meta = {}
    raw_counts = meta.get("raw_counts")
    if not isinstance(raw_counts, dict):
        raw_counts = {}
    raw_counts["odom_motion_samples"] = int(odom_meta.get("odom_sample_count", 0))
    meta["raw_counts"] = raw_counts
    meta["odom_topic"] = odom_meta.get("odom_topic")
    meta["frame_orientations_order"] = "xyzw"
    meta["odom_repair"] = {
        "schema_version": "sidewalk_odom_repair_v1",
        "mcap_path": str(mcap_path),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "odom_schema": odom_meta.get("odom_schema"),
        "odom_message_encoding": odom_meta.get("odom_message_encoding"),
        "stats": repair_stats,
    }
    if not dry_run:
        path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")


def repair_clip_odom(
    *,
    mcap_path: Path,
    clip_dir: Path,
    odom_topic: str,
    max_sync_delta_s: float,
    orientation_source: str,
    only_static_positions: bool,
    dry_run: bool,
) -> dict[str, Any]:
    trajectory_path = clip_dir / "trajectory.json"
    if not trajectory_path.exists():
        return {"clip": clip_dir.name, "status": "skipped", "reason": "missing_trajectory_json"}

    existing_trajectory = _load_json(trajectory_path, [])
    if not isinstance(existing_trajectory, list) or not existing_trajectory:
        return {"clip": clip_dir.name, "status": "skipped", "reason": "empty_trajectory_json"}
    trajectory = _trajectory_from_full_fpv_timeline(clip_dir, existing_trajectory)

    if only_static_positions:
        positions_path = clip_dir / "frame_positions.npy"
        if positions_path.exists():
            try:
                positions = np.load(positions_path)
                positions_dense_enough = len(positions) >= len(trajectory)
                if positions_dense_enough and not _positions_are_static(positions):
                    return {"clip": clip_dir.name, "status": "skipped", "reason": "positions_not_static"}
            except Exception:
                pass

    samples, odom_meta = _read_odom_samples(mcap_path, odom_topic=odom_topic)
    if not samples:
        return {
            "clip": clip_dir.name,
            "status": "failed",
            "reason": odom_meta.get("error") or "no_odom_samples",
            "mcap": str(mcap_path),
        }

    timestamps = [int(sample.timestamp_ns) for sample in samples]
    max_delta_ns = int(float(max_sync_delta_s) * 1e9)
    sync_deltas: list[float] = []
    missing = 0
    repaired: list[dict[str, Any]] = []
    for row in trajectory:
        timestamp = int(row["timestamp"])
        sample, delta_ns = _nearest_sample(samples, timestamps, timestamp)
        payload = _odom_payload(sample, delta_ns, max_delta_ns=max_delta_ns)
        if payload.get("odom_source") != "odom":
            missing += 1
        else:
            sync_deltas.append(float(payload["odom_sync_delta_s"]))
        # Preserve all existing non-odom fields and replace only odom keys.
        clean_row = {key: value for key, value in row.items() if not key.startswith("odom_")}
        clean_row.update(payload)
        repaired.append(clean_row)

    odom_meta["odom_sample_count"] = len(samples)
    filled_missing = _fill_missing_odom_rows(repaired)
    repair_stats = _write_odom_arrays(
        clip_dir,
        repaired,
        orientation_source=orientation_source,
        dry_run=dry_run,
    )
    repair_stats.update(
        {
            "input_trajectory_frames": len(existing_trajectory),
            "repair_timeline_source": "frame_times_npy" if len(trajectory) > len(existing_trajectory) else "trajectory_json",
            "missing_odom_frames": missing,
            "filled_missing_odom_frames": filled_missing,
            "max_sync_delta_s": round(max(sync_deltas), 6) if sync_deltas else None,
            "mean_sync_delta_s": round(float(np.mean(sync_deltas)), 6) if sync_deltas else None,
        }
    )

    if not dry_run:
        trajectory_path.write_text(json.dumps(repaired, ensure_ascii=True, indent=2), encoding="utf-8")
    _update_frames_manifest(clip_dir, repaired, dry_run=dry_run)
    _update_extract_meta(
        clip_dir,
        mcap_path=mcap_path,
        odom_meta=odom_meta,
        repair_stats=repair_stats,
        dry_run=dry_run,
    )

    status = "ok" if missing == 0 else "partial"
    return {
        "clip": clip_dir.name,
        "status": status,
        "mcap": str(mcap_path),
        **repair_stats,
    }


def _task_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    return repair_clip_odom(
        mcap_path=args[0],
        clip_dir=args[1],
        odom_topic=args[2],
        max_sync_delta_s=args[3],
        orientation_source=args[4],
        only_static_positions=args[5],
        dry_run=args[6],
    )


def _discover_tasks(
    *,
    dataset_root: Path,
    sidewalk_root: Path,
    chunks: list[str],
    limit: int | None,
) -> list[tuple[Path, Path]]:
    tasks: list[tuple[Path, Path]] = []
    flat_mcaps = sorted(dataset_root.glob("*.mcap"))
    if flat_mcaps:
        for mcap_path in flat_mcaps:
            clip_dir = sidewalk_root / mcap_path.stem
            if clip_dir.exists():
                tasks.append((mcap_path, clip_dir))
                if limit is not None and len(tasks) >= limit:
                    return tasks
        if tasks:
            return tasks

    for chunk in chunks:
        chunk_name = str(chunk)
        if not chunk_name.startswith("chunk_"):
            chunk_name = f"chunk_{chunk_name}"
        mcap_dir = dataset_root / chunk_name / "mcap"
        sidewalk_chunk = sidewalk_root / chunk_name
        if not mcap_dir.exists() or not sidewalk_chunk.exists():
            continue
        for mcap_path in sorted(mcap_dir.glob("*.mcap")):
            clip_dir = sidewalk_chunk / mcap_path.stem
            if clip_dir.exists():
                tasks.append((mcap_path, clip_dir))
                if limit is not None and len(tasks) >= limit:
                    return tasks
    return tasks


def _write_report(path: Path | None, rows: list[dict[str, Any]], *, dry_run: bool) -> None:
    if path is None or dry_run:
        return
    summary: dict[str, Any] = {
        "schema_version": "sidewalk_odom_repair_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "status_counts": {},
        "rows": rows,
    }
    for row in rows:
        status = str(row.get("status") or "unknown")
        summary["status_counts"][status] = int(summary["status_counts"].get(status, 0)) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair prepared SideWalk clips with odometry from original MCAP files.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Raw MCAP dataset root containing chunk_xx/mcap/*.mcap")
    parser.add_argument("--sidewalk-root", type=Path, required=True, help="Prepared SideWalk root containing chunk_xx/<clip_id>/")
    parser.add_argument("--chunks", nargs="+", default=["26", "27", "28", "29"], help="Chunk ids, e.g. 26 27 28 29")
    parser.add_argument("--odom-topic", default="auto", help="Odometry topic. Use auto for schema-based discovery.")
    parser.add_argument("--max-sync-delta-s", type=float, default=0.25, help="Maximum image-to-odom sync delta.")
    parser.add_argument(
        "--orientation-source",
        choices=["odom", "trajectory_heading"],
        default="odom",
        help=(
            "Orientation written to frame_orientations.npy. Use trajectory_heading for realworld clips whose "
            "MCAP odom quaternion is not the base_link forward yaw."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() // 2), help="Parallel workers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--only-static-positions", action="store_true", help="Repair only clips whose existing frame_positions.npy is static.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without writing files.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    tasks = _discover_tasks(
        dataset_root=args.dataset_root,
        sidewalk_root=args.sidewalk_root,
        chunks=args.chunks,
        limit=args.limit,
    )
    print(f"[odom-repair] discovered {len(tasks)} existing clips")
    if not tasks:
        return

    rows: list[dict[str, Any]] = []
    worker_args = [
        (
            mcap_path,
            clip_dir,
            args.odom_topic,
            float(args.max_sync_delta_s),
            str(args.orientation_source),
            bool(args.only_static_positions),
            bool(args.dry_run),
        )
        for mcap_path, clip_dir in tasks
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.num_workers))) as executor:
        futures = {executor.submit(_task_worker, item): item for item in worker_args}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:
                mcap_path, clip_dir, *_ = futures[future]
                row = {"clip": clip_dir.name, "mcap": str(mcap_path), "status": "failed", "reason": str(exc)}
            rows.append(row)
            status = row.get("status")
            detail = row.get("reason") or f"odom={row.get('odom_frame_count', 0)}/{row.get('frame_count', 0)} disp={row.get('xy_displacement_m', 0)}m"
            print(f"[{index}/{len(tasks)}] {status} {row.get('clip')} {detail}")

    _write_report(args.report_json, rows, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
