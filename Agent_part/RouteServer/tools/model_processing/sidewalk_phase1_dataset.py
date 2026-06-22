#!/usr/bin/env python3
"""Build Phase1 SideWalk System2 training artifacts.

This script is the Agent-side bridge between raw/prepared SideWalk clips and
ms-swift training.  It intentionally writes a lightweight, stable contract:

* source JSONL rows consumed by ``swift.sidewalk_nav.dataset_builder``
* per-frame trajectory labels
* per-frame BEV topology metadata
* per-frame QA bundles using the ms-swift SideWalk QA schema

Phase1 does not require SAM/SoM/segmentation outputs.  Production trajectory
labels are generated from InternNav clip poses and calibrated camera
parameters, matching ``internnav.dataset.sidewalk_trajectory_generator``.  A
GNSS/HFOV heuristic path remains available only as an explicit debug fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


QA_SCHEMA_VERSION = "sidewalk_bev_fpv_qa_v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are a sidewalk navigation assistant. Infer local navigation intent from "
    "FPV and BEV observations and output feasible FPV visual prompt trajectories."
)
DEFAULT_INSTRUCTION = "Navigate safely on the sidewalk and follow the visible route."


@dataclass(frozen=True)
class Phase1Config:
    num_points: int = 16
    future_frames: int = 50
    future_horizon_sec: float | None = 2.5
    frame_step: int = 1
    hfov_deg: float = 90.0
    min_future_distance_m: float = 0.3
    trajectory_min_y_step: float = 0.005
    max_bev_time_delta_ms: float = 300.0
    frame_rate_hz: float = 20.0


@dataclass(frozen=True)
class CameraParams:
    """Calibrated FPV camera parameters used for pose-to-image projection."""

    K: np.ndarray
    image_width: int
    image_height: int
    camera_height_meters: float

    @classmethod
    def from_json(cls, path: Path) -> "CameraParams":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "K" in data:
            K = np.asarray(data["K"], dtype=np.float64)
        elif "camera_intrinsic" in data:
            K = np.asarray(data["camera_intrinsic"], dtype=np.float64)
        else:
            raise ValueError(f"camera_params.json missing K/camera_intrinsic: {path}")
        K = K.reshape(3, 3)
        return cls(
            K=K,
            image_width=int(data.get("image_width") or data.get("width")),
            image_height=int(data.get("image_height") or data.get("height")),
            camera_height_meters=float(data.get("camera_height_meters", 0.6)),
        )


@dataclass(frozen=True)
class FrameRecord:
    clip_dir: Path
    clip_id: str
    frame_name: str
    frame_idx: int
    image_path: Path | None
    bev_image_path: Path | None
    fpv_video_path: Path | None = None
    bev_frame_idx: int | None = None
    bev_timestamp: int | None = None
    bev_time_delta_ms: float | None = None
    bev_mapping_source: str | None = None
    lat: float | None = None
    lng: float | None = None
    heading_deg: float | None = None
    timestamp: int | None = None
    speed_mps: float | None = None
    video_frame_idx: int | None = None
    video_frame_rate_hz: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    position_xy: tuple[float, float] | None = None
    orientation_wxyz: tuple[float, float, float, float] | None = None
    camera_params: CameraParams | None = None
    pose_source: str | None = None
    pose_fallback_reason: str | None = None


def discover_clip_dirs(root: Path) -> list[Path]:
    """Return clip directories in either InternNav or legacy JSON format."""

    root = root.expanduser().resolve()
    if _is_internnav_clip(root) or (root / "trajectory.json").is_file():
        return [root]
    dirs = {
        path
        for path in root.iterdir()
        if path.is_dir() and (_is_internnav_clip(path) or (path / "trajectory.json").is_file())
    }
    dirs.update({
        path.parent
        for path in root.rglob("trajectory.json")
        if path.parent.is_dir()
    })
    dirs.update(
        path.parent
        for path in root.rglob("frame_positions.npy")
        if (path.parent / "frame_orientations.npy").is_file()
    )
    return sorted(dirs)


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"trajectory.json must be a list: {path}")
    return [item for item in data if isinstance(item, dict)]


def _is_internnav_clip(path: Path) -> bool:
    return (path / "frame_positions.npy").is_file() and (path / "frame_orientations.npy").is_file()


def _positions_are_static(
    positions: np.ndarray,
    *,
    max_displacement_m: float = 0.5,
    max_path_length_m: float = 1.0,
) -> bool:
    """Return true when odom/map positions cannot support trajectory labels.

    Some prepared SideWalk chunks contain valid ``frame_gps.npy`` and
    ``trajectory.json`` but have ``frame_positions.npy`` filled with zeros.
    Treating those zeros as calibrated map poses produces a fake stop-hold
    trajectory for every frame.  This guard keeps the calibrated path for
    normal InternNav clips while forcing static-pose clips onto the explicit
    GNSS fallback path.
    """

    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return True
    xy = arr[:, :2]
    finite_mask = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite_mask]
    if xy.shape[0] < 2:
        return True
    deltas = np.diff(xy, axis=0)
    path_length = float(np.sum(np.linalg.norm(deltas, axis=1)))
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    return displacement <= float(max_displacement_m) and path_length <= float(max_path_length_m)


def _load_camera_params(clip_dir: Path) -> CameraParams | None:
    for path in (clip_dir / "camera_params.json", clip_dir.parent / "camera_params.json"):
        if path.is_file():
            return CameraParams.from_json(path)
    return None


def _load_optional_array(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _load_frame_orientation_order(clip_dir: Path) -> str | None:
    """Read the on-disk quaternion order when the MCAP extractor records it."""

    meta_path = clip_dir / "mcap_extract_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    order = str(meta.get("frame_orientations_order") or "").strip().lower()
    return order if order in {"xyzw", "wxyz"} else None


def _load_front_video_fps(clip_dir: Path) -> float | None:
    for meta_name in ("mcap_extract_meta.json", "conversion_report.json"):
        meta_path = clip_dir / meta_name
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("front_video_fps", "video_fps"):
            value = meta.get(key)
            if value is None:
                continue
            try:
                fps = float(value)
            except Exception:
                continue
            if fps > 0:
                return fps
    return None


def _orientation_row_to_wxyz(row: np.ndarray, *, order: str | None) -> tuple[float, float, float, float]:
    """Convert stored quaternion rows into the builder's internal wxyz order.

    RouteServer/ROS extraction stores quaternions as ``xyzw``.  Older unit
    tests and some prepared InternNav clips used ``wxyz``.  The extractor now
    records the order in ``mcap_extract_meta.json``; for legacy clips without
    metadata we use a conservative identity-quaternion heuristic.
    """

    q = np.asarray(row[:4], dtype=np.float64)
    if str(order or "").lower() == "xyzw":
        x, y, z, w = q
        return float(w), float(x), float(y), float(z)
    if str(order or "").lower() == "wxyz":
        w, x, y, z = q
        return float(w), float(x), float(y), float(z)

    # Identity in ROS xyzw is [0, 0, 0, 1], while old wxyz is [1, 0, 0, 0].
    if abs(float(q[0])) < 0.25 and abs(float(q[1])) < 0.25 and (abs(float(q[2])) > 0.25 or abs(float(q[3])) > 0.25):
        x, y, z, w = q
        return float(w), float(x), float(y), float(z)
    w, x, y, z = q
    return float(w), float(x), float(y), float(z)


def build_phase1_dataset(
    sidewalk_root: str,
    output_root: str,
    *,
    split_name: str = "train",
    config: Phase1Config | None = None,
    limit: int | None = None,
    max_clips: int | None = None,
    allow_missing_bev: bool = False,
    allow_heuristic_trajectory: bool = False,
    write_overlays: bool = False,
) -> dict[str, Any]:
    """Build source JSONL and offline QA artifacts for Phase1 training."""

    cfg = config or Phase1Config()
    input_root = Path(sidewalk_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_jsonl = manifest_dir / f"phase1_source_{split_name}.jsonl"
    qa_index_jsonl = manifest_dir / f"phase1_qa_index_{split_name}.jsonl"

    source_rows: list[dict[str, Any]] = []
    qa_index_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": "sidewalk_phase1_dataset_v1",
        "sidewalk_root": str(input_root),
        "output_root": str(output),
        "split_name": split_name,
        "allow_heuristic_trajectory": bool(allow_heuristic_trajectory),
        "max_clips": max_clips,
        "clips_total": 0,
        "clips_used": 0,
        "frames_written": 0,
        "frames_skipped": 0,
        "skip_reasons": {},
        "intent_counts": {},
        "trajectory_source_counts": {},
    }

    for clip_dir in discover_clip_dirs(input_root):
        if limit is not None and len(source_rows) >= limit:
            break
        if max_clips is not None and int(report["clips_total"]) >= int(max_clips):
            break
        report["clips_total"] += 1
        try:
            frame_records = list(
                _iter_clip_frame_records(
                    clip_dir,
                    output_root=output,
                    allow_missing_bev=allow_missing_bev,
                    allow_heuristic_trajectory=allow_heuristic_trajectory,
                    max_bev_time_delta_ms=cfg.max_bev_time_delta_ms,
                )
            )
        except Exception as exc:
            _inc(report["skip_reasons"], f"clip_load_error:{type(exc).__name__}")
            continue
        if not frame_records:
            _inc(report["skip_reasons"], "empty_or_unusable_clip")
            continue

        clip_rows_before = len(source_rows)
        for local_idx in range(0, len(frame_records), max(int(cfg.frame_step), 1)):
            if limit is not None and len(source_rows) >= limit:
                break
            record = frame_records[local_idx]
            future = select_future_records(frame_records, local_idx, config=cfg)
            if not future:
                report["frames_skipped"] += 1
                _inc(report["skip_reasons"], "no_future_frames")
                continue
            try:
                sample = build_phase1_sample(
                    record,
                    future,
                    output_root=output,
                    config=cfg,
                    allow_heuristic_trajectory=allow_heuristic_trajectory,
                    write_overlay=write_overlays,
                )
            except Exception as exc:
                report["frames_skipped"] += 1
                _inc(report["skip_reasons"], f"sample_error:{type(exc).__name__}")
                continue
            source_rows.append(sample["source_row"])
            qa_index_rows.append(sample["qa_index_row"])
            intent = sample["source_row"]["sidewalk_meta"].get("intent_gt")
            trajectory_source = sample["source_row"]["sidewalk_meta"].get("trajectory_source")
            _inc(report["intent_counts"], str(intent or "unknown"))
            _inc(report["trajectory_source_counts"], str(trajectory_source or "unknown"))

        if len(source_rows) > clip_rows_before:
            report["clips_used"] += 1


    _write_jsonl(source_jsonl, source_rows)
    _write_jsonl(qa_index_jsonl, qa_index_rows)
    report["frames_written"] = len(source_rows)
    report["source_jsonl"] = str(source_jsonl)
    report["qa_index_jsonl"] = str(qa_index_jsonl)
    report_path = manifest_dir / f"phase1_dataset_report_{split_name}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def select_future_records(frame_records: list[FrameRecord], current_index: int, *, config: Phase1Config) -> list[FrameRecord]:
    """Select future FPV/odom records for trajectory supervision.

    The production target is a fixed time horizon, not a fixed number of
    already-decimated records.  Chunk0 is 20Hz, so the historical
    ``future_frames=50`` naturally spans about 2.5s.  Some later chunks have
    irregular or lower effective FPV rates after decoding; using 50 records
    there stretches the trajectory to 4s+.  When timestamps are available, use
    ``future_horizon_sec`` first and keep all future records whose arrival time
    is inside that horizon.  ``future_frames`` remains the fallback for clips
    without timestamps or for explicit legacy runs with horizon <= 0.
    """

    if current_index < 0 or current_index >= len(frame_records):
        return []
    current = frame_records[current_index]
    candidates = frame_records[current_index + 1 :]
    if not candidates:
        return []

    horizon = config.future_horizon_sec
    if horizon is not None and float(horizon) > 0.0 and current.timestamp is not None:
        selected: list[FrameRecord] = []
        for item in candidates:
            if item.timestamp is None:
                selected = []
                break
            dt = _timestamp_delta_seconds(current.timestamp, item.timestamp)
            if dt <= 0.0:
                continue
            if dt <= float(horizon) + 1e-9:
                selected.append(item)
                continue
            break
        if selected:
            return selected

    return candidates[: max(int(config.future_frames), 1)]


def build_phase1_sample(
    record: FrameRecord,
    future: list[FrameRecord],
    *,
    output_root: Path,
    config: Phase1Config,
    allow_heuristic_trajectory: bool = False,
    write_overlay: bool = False,
) -> dict[str, Any]:
    """Build one ms-swift source row plus sidecar labels for one frame."""

    record = materialize_fpv_image(record, output_root, frame_rate_hz=config.frame_rate_hz)
    with Image.open(record.image_path) as image:
        image_width, image_height = image.size

    target_pack = build_phase1_targets(
        record,
        future,
        config=config,
        allow_heuristic_trajectory=allow_heuristic_trajectory,
    )
    local_path = target_pack["local_path"]
    trajectory_norm = target_pack["trajectory_norm"]
    trajectory_fpv_t = target_pack["trajectory_fpv_t"]
    trajectory_fpv_tv = target_pack["trajectory_fpv_tv"]
    trajectory_ego_t = target_pack["trajectory_ego_t"]
    trajectory_policy = target_pack["trajectory_policy"]
    trajectory_source = str(target_pack["trajectory_source"])
    route_theta_deg = bev_theta_deg(local_path[-1, 0], local_path[-1, 1])
    intent = intent_from_theta(route_theta_deg, distance_m=float(np.linalg.norm(local_path[-1])))
    anchor = anchor_from_theta(route_theta_deg)
    bev_meta = build_bev_patch_meta(
        record,
        local_path,
        route_theta_deg=route_theta_deg,
        intent=intent,
        anchor=anchor,
    )

    clip_output = output_root / "clips" / record.clip_id
    label_dir = clip_output / "trajectory_labels"
    bev_meta_dir = clip_output / "bev_patch_meta"
    qa_dir = clip_output / "qa"
    overlay_dir = clip_output / "overlays"
    label_dir.mkdir(parents=True, exist_ok=True)
    bev_meta_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    label_path = label_dir / f"{record.frame_name}.json"
    bev_meta_path = bev_meta_dir / f"{record.frame_name}.json"
    qa_path = qa_dir / f"{record.frame_name}_qa.json"
    overlay_path = overlay_dir / f"{record.frame_name}_debug.jpg"

    label_payload = {
        "schema_version": "sidewalk_phase1_trajectory_v2",
        "clip_id": record.clip_id,
        "frame_idx": record.frame_idx,
        "frame_name": record.frame_name,
        "image_size": {"width": image_width, "height": image_height},
        "trajectory_norm": trajectory_norm.tolist(),
        "trajectory_fpv_t": trajectory_fpv_t,
        "trajectory_fpv_tv": trajectory_fpv_tv,
        "trajectory_pixel": to_pixel_trajectory(trajectory_norm, image_width, image_height).tolist(),
        "local_path_m": local_path.astype(float).tolist(),
        "trajectory_ego_t": trajectory_ego_t,
        "trajectory_policy": trajectory_policy,
        "route_theta_deg": round(float(route_theta_deg), 3),
        "intent_gt": intent,
        "anchor": anchor,
        "trajectory_source": trajectory_source,
        "pose_source": record.pose_source,
        "pose_fallback_reason": record.pose_fallback_reason,
        "projection": target_pack.get("projection", {}),
    }
    _drop_none_fields(label_payload)
    label_path.write_text(json.dumps(label_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    bev_meta_path.write_text(json.dumps(bev_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    qa_bundle = build_qa_bundle(
        record=record,
        image_width=image_width,
        image_height=image_height,
        trajectory_norm=trajectory_norm,
        trajectory_fpv_t=trajectory_fpv_t,
        trajectory_ego_t=trajectory_ego_t,
        trajectory_policy=trajectory_policy,
        bev_meta=bev_meta,
        label_path=label_path,
        bev_meta_path=bev_meta_path,
        trajectory_source=trajectory_source,
    )
    qa_path.write_text(json.dumps(qa_bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        write_debug_overlay(record.image_path, trajectory_norm, overlay_path)

    sidewalk_meta = {
        "source_dataset": "sidewalk_phase1_agent",
        "clip_id": record.clip_id,
        "frame_idx": record.frame_idx,
        "frame_name": record.frame_name,
        "timestamp": record.timestamp,
        "bev_frame_idx": record.bev_frame_idx,
        "bev_timestamp": record.bev_timestamp,
        "bev_time_delta_ms": record.bev_time_delta_ms,
        "bev_mapping_source": record.bev_mapping_source,
        "lat": record.lat,
        "lng": record.lng,
        "heading_deg": record.heading_deg,
        "intent_gt": intent,
        "route_theta_deg": round(float(route_theta_deg), 3),
        "trajectory_source": trajectory_source,
        "pose_source": record.pose_source,
        "pose_fallback_reason": record.pose_fallback_reason,
        "bev_patch_meta_path": str(bev_meta_path),
        "trajectory_label_path": str(label_path),
        "qa_bundle_path": str(qa_path),
    }
    _drop_none_fields(sidewalk_meta)

    source_row = {
        "id": f"{record.clip_id}_{record.frame_name}",
        "clip_id": record.clip_id,
        "frame_idx": record.frame_idx,
        "frame_name": record.frame_name,
        "fpv_image_path": str(record.image_path),
        "bev_image_path": str(record.bev_image_path) if record.bev_image_path else None,
        "pixel_trajectory": trajectory_norm.astype(float).tolist(),
        "trajectory_fpv_t": trajectory_fpv_t,
        "trajectory_fpv_tv": trajectory_fpv_tv,
        "trajectory_ego_t": trajectory_ego_t,
        "trajectory_policy": trajectory_policy,
        "image_width": image_width,
        "image_height": image_height,
        "instruction": DEFAULT_INSTRUCTION,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "trajectory_label_path": str(label_path),
        "bev_patch_meta_path": str(bev_meta_path),
        "qa_bundle_path": str(qa_path),
        "route_context": {
            "route_theta_deg": round(float(route_theta_deg), 3),
            "intent_gt": intent,
            "anchor": anchor,
        },
        "sidewalk_meta": sidewalk_meta,
    }
    if source_row["bev_image_path"] is None:
        source_row.pop("bev_image_path")

    qa_index_row = {
        "qa_bundle_path": str(qa_path),
        "clip_id": record.clip_id,
        "frame_idx": record.frame_idx,
        "fpv_image_path": str(record.image_path),
        "bev_image_path": str(record.bev_image_path) if record.bev_image_path else None,
        "num_samples": len(qa_bundle["samples"]),
    }
    return {"source_row": source_row, "qa_index_row": qa_index_row}


def build_phase1_targets(
    record: FrameRecord,
    future: list[FrameRecord],
    *,
    config: Phase1Config,
    allow_heuristic_trajectory: bool,
) -> dict[str, Any]:
    """Generate calibrated FPV trajectory and matching local BEV path.

    The calibrated branch mirrors InternNav's SidewalkTrajectoryGenerator:
    future map/odom points are transformed into the current robot frame and
    then projected with camera intrinsics.  The legacy GNSS/HFOV path is kept
    only for explicit smoke-test fallback.
    """

    if record.position_xy is not None and record.orientation_wxyz is not None:
        sampled_future = sample_future_records(future, config.num_points)
        local_path = pose_local_path_right_front(record, sampled_future)
        if local_path.shape[0] == 0:
            local_path = np.asarray([[0.0, 1.0], [0.0, 3.0], [0.0, 6.0]], dtype=np.float32)
            local_path = resample_polyline(local_path, config.num_points)
        if record.camera_params is not None:
            trajectory_norm = calibrated_fpv_trajectory(record, sampled_future, config=config)
            time_pack = build_time_conditioned_targets(
                record,
                sampled_future,
                local_path=local_path,
                trajectory_norm=trajectory_norm,
                config=config,
                trajectory_source="calibrated_pose_projection",
            )
            return {
                "local_path": local_path,
                "trajectory_norm": trajectory_norm,
                **time_pack,
                "trajectory_source": "calibrated_pose_projection",
                "projection": {
                    "coordinate_chain": "map_pose_to_robot_to_camera_to_fpv",
                    "camera_height_meters": record.camera_params.camera_height_meters,
                    "camera_image_size": [record.camera_params.image_width, record.camera_params.image_height],
                    "postprocess": {
                        "anchor_norm": [0.5, 0.9],
                        "monotonic_forward_y": True,
                        "min_y_step": float(config.trajectory_min_y_step),
                    },
                },
            }
        if allow_heuristic_trajectory:
            trajectory_norm = fpv_trajectory_from_local_path(local_path, config=config)
            time_pack = build_time_conditioned_targets(
                record,
                sampled_future,
                local_path=local_path,
                trajectory_norm=trajectory_norm,
                config=config,
                trajectory_source="heuristic_pose_projection",
            )
            return {
                "local_path": local_path,
                "trajectory_norm": trajectory_norm,
                **time_pack,
                "trajectory_source": "heuristic_pose_projection",
                "projection": {"coordinate_chain": "map_pose_to_robot_to_hfov_fpv"},
            }
        raise ValueError("camera_params required for calibrated FPV trajectory")

    if allow_heuristic_trajectory:
        local_path = legacy_gnss_local_path(record, future, config=config)
        trajectory_norm = fpv_trajectory_from_local_path(local_path, config=config)
        sampled_future = sample_future_records(future, config.num_points)
        time_pack = build_time_conditioned_targets(
            record,
            sampled_future,
            local_path=local_path,
            trajectory_norm=trajectory_norm,
            config=config,
            trajectory_source="heuristic_gnss_projection",
        )
        return {
            "local_path": local_path,
            "trajectory_norm": trajectory_norm,
            **time_pack,
            "trajectory_source": "heuristic_gnss_projection",
            "projection": {"coordinate_chain": "gnss_heading_to_hfov_fpv"},
        }
    raise ValueError("InternNav pose/camera data required; pass --allow-heuristic-trajectory for debug fallback")


def build_time_conditioned_targets(
    record: FrameRecord,
    sampled_future: list[FrameRecord],
    *,
    local_path: np.ndarray,
    trajectory_norm: np.ndarray,
    config: Phase1Config,
    trajectory_source: str,
) -> dict[str, Any]:
    """Build time-conditioned trajectory labels without changing legacy XY GT.

    ``pixel_trajectory`` remains the backward-compatible Phase1 target.  The
    additional fields here make the label explicit about *when* the robot
    reaches each waypoint, which is required to reason about dynamic obstacles:
    a future executed path may cross the current location of a pedestrian only
    after that pedestrian has moved.  Without dynamic masks/tracks this function
    does not rewrite the path; it records that dynamic conflict auditing has not
    been evaluated yet.
    """

    traj = np.asarray(trajectory_norm, dtype=np.float32).reshape(-1, 2)
    local = np.asarray(local_path, dtype=np.float32).reshape(-1, 2)
    target_n = int(config.num_points)
    if traj.shape[0] != target_n:
        traj = resample_polyline(traj, target_n)
    if local.shape[0] != target_n:
        local = resample_polyline(local, target_n)

    dt_values = _trajectory_dt_seconds(record, sampled_future, target_n, config=config)
    local_time = np.zeros((target_n, 2), dtype=np.float32)
    if target_n > 1 and local.shape[0] > 1:
        local_time[1:] = local[1:target_n]
    speeds = _trajectory_speeds_mps(record, sampled_future, local_time, dt_values)

    fpv_t = [
        [_round_float(traj[i, 0]), _round_float(traj[i, 1]), _round_float(dt_values[i])]
        for i in range(target_n)
    ]
    fpv_tv = [
        [
            _round_float(traj[i, 0]),
            _round_float(traj[i, 1]),
            _round_float(dt_values[i]),
            _round_float(speeds[i]),
        ]
        for i in range(target_n)
    ]
    ego_t = [
        [
            _round_float(local_time[i, 0]),
            _round_float(local_time[i, 1]),
            _round_float(dt_values[i]),
            _round_float(speeds[i]),
        ]
        for i in range(target_n)
    ]
    return {
        "trajectory_fpv_t": fpv_t,
        "trajectory_fpv_tv": fpv_tv,
        "trajectory_ego_t": ego_t,
        "trajectory_policy": {
            "schema_version": "sidewalk_time_conditioned_policy_v1",
            "primary_temporal_target": "trajectory_fpv_t",
            "speed_target": "trajectory_fpv_tv",
            "static_compatibility_target": "pixel_trajectory",
            "trajectory_role": "executed_future_path_projected_to_current_fpv",
            "trajectory_source": trajectory_source,
            "time_unit": "seconds",
            "speed_unit": "mps",
            "frame_rate_hz_fallback": float(config.frame_rate_hz),
            "speed_policy": speed_policy_from_profile(speeds),
            "dynamic_obstacle_audit": {
                "status": "not_evaluated_no_dynamic_masks",
                "has_dynamic_masks": False,
                "current_frame_conflict": "unknown",
                "note": (
                    "Future odom is an executed path, not a guaranteed current-frame "
                    "free-space mask. Add dynamic masks/tracks before rewriting labels "
                    "around moving obstacles."
                ),
            },
        },
    }


def _trajectory_dt_seconds(
    record: FrameRecord,
    sampled_future: list[FrameRecord],
    num_points: int,
    *,
    config: Phase1Config,
) -> list[float]:
    """Return monotonic arrival times for trajectory indices.

    Index 0 is the current robot anchor.  Later indices are matched to the
    corresponding future pose when possible.  If timestamps are unavailable, the
    frame-index delta is converted with ``config.frame_rate_hz``.
    """

    dt_values = [0.0]
    fallback_step = 1.0 / max(float(config.frame_rate_hz), 1e-6)
    for i in range(1, int(num_points)):
        item = sampled_future[min(i, len(sampled_future) - 1)] if sampled_future else None
        dt = None
        if item is not None:
            dt = _record_delta_seconds(record, item, config=config)
        if dt is None:
            dt = float(i) * fallback_step
        if dt <= dt_values[-1]:
            dt = dt_values[-1] + fallback_step
        dt_values.append(float(dt))
    return dt_values


def _record_delta_seconds(record: FrameRecord, item: FrameRecord, *, config: Phase1Config) -> float | None:
    if record.timestamp is not None and item.timestamp is not None:
        return _timestamp_delta_seconds(record.timestamp, item.timestamp)
    frame_delta = int(item.frame_idx) - int(record.frame_idx)
    if frame_delta > 0:
        return float(frame_delta) / max(float(config.frame_rate_hz), 1e-6)
    return None


def _timestamp_delta_seconds(start: int, end: int) -> float:
    """Convert timestamp deltas from common SideWalk units to seconds."""

    delta = float(int(end) - int(start))
    abs_delta = abs(delta)
    if abs_delta >= 1_000_000.0:
        return delta / 1e9  # nanoseconds, used by frame_times.npy
    if abs_delta >= 1_000.0:
        return delta / 1e3  # milliseconds
    return delta  # seconds or small synthetic ticks


def _trajectory_speeds_mps(
    record: FrameRecord,
    sampled_future: list[FrameRecord],
    local_time: np.ndarray,
    dt_values: list[float],
) -> list[float]:
    speeds: list[float] = []
    for i in range(len(dt_values)):
        item = sampled_future[min(i, len(sampled_future) - 1)] if sampled_future else None
        if item is not None and item.speed_mps is not None:
            speeds.append(max(float(item.speed_mps), 0.0))
            continue
        if i == 0:
            speeds.append(max(float(record.speed_mps or 0.0), 0.0))
            continue
        dt = max(float(dt_values[i]) - float(dt_values[i - 1]), 1e-6)
        distance = float(np.linalg.norm(local_time[i] - local_time[i - 1]))
        speeds.append(max(distance / dt, 0.0))
    if len(speeds) > 1 and speeds[0] <= 1e-6:
        speeds[0] = speeds[1]
    return speeds


def speed_policy_from_profile(speeds: list[float]) -> str:
    if not speeds:
        return "unknown"
    mean_speed = float(np.mean(np.asarray(speeds, dtype=np.float32)))
    max_speed = float(np.max(np.asarray(speeds, dtype=np.float32)))
    if max_speed < 0.15:
        return "stop"
    if mean_speed < 0.6:
        return "slow"
    if mean_speed < 1.5:
        return "normal"
    return "fast_or_unverified"


def _round_float(value: Any, digits: int = 5) -> float:
    return round(float(value), int(digits))


def legacy_gnss_local_path(record: FrameRecord, future: list[FrameRecord], *, config: Phase1Config) -> np.ndarray:
    if record.lat is None or record.lng is None or record.heading_deg is None:
        raise ValueError("legacy GNSS fallback requires lat/lng/heading_deg")
    local_points = []
    for item in future:
        if item.lat is None or item.lng is None:
            continue
        local_points.append(local_xy_m(record.lat, record.lng, item.lat, item.lng, record.heading_deg))
    local_points_arr = np.asarray(local_points, dtype=np.float32).reshape(-1, 2)
    local_points_arr = _drop_near_or_behind(local_points_arr, min_distance_m=config.min_future_distance_m)
    if local_points_arr.shape[0] == 0:
        local_points_arr = np.asarray([[0.0, 1.0], [0.0, 3.0], [0.0, 6.0]], dtype=np.float32)
    return resample_polyline(local_points_arr, config.num_points)


def sample_future_records(future: list[FrameRecord], num_points: int) -> list[FrameRecord]:
    if not future:
        return []
    if len(future) == 1:
        return [future[0] for _ in range(int(num_points))]
    indices = np.linspace(0, len(future) - 1, int(num_points), dtype=int)
    return [future[int(idx)] for idx in indices]


def pose_local_path_right_front(record: FrameRecord, sampled_future: list[FrameRecord]) -> np.ndarray:
    if record.position_xy is None or record.orientation_wxyz is None:
        return np.zeros((0, 2), dtype=np.float32)
    points = []
    for item in sampled_future:
        if item.position_xy is None:
            continue
        forward_m, left_m = robot_forward_left_m(record.position_xy, record.orientation_wxyz, item.position_xy)
        points.append([-left_m, forward_m])
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def calibrated_fpv_trajectory(record: FrameRecord, sampled_future: list[FrameRecord], *, config: Phase1Config) -> np.ndarray:
    if record.position_xy is None or record.orientation_wxyz is None or record.camera_params is None:
        raise ValueError("calibrated projection requires position, orientation, and camera params")
    cp = record.camera_params
    points = []
    for item in sampled_future:
        if item.position_xy is None:
            continue
        u, v = project_pose_to_fpv(record.position_xy, record.orientation_wxyz, item.position_xy, cp)
        points.append([u / float(cp.image_width), v / float(cp.image_height)])
    if not points:
        points = [[0.5, 0.9] for _ in range(int(config.num_points))]
    while len(points) < int(config.num_points):
        points.append(points[-1])
    trajectory = np.asarray(points[: int(config.num_points)], dtype=np.float32)
    return smooth_forward_fpv_trajectory(trajectory, config=config)


def smooth_forward_fpv_trajectory(trajectory_norm: np.ndarray, *, config: Phase1Config) -> np.ndarray:
    """Anchor and smooth FPV waypoints so they move monotonically forward.

    Calibrated projection is correct geometrically, but very near future poses
    can project to the bottom image border.  If we simply replace the first
    projected point with the robot anchor ``(0.5, 0.9)``, the next point may sit
    lower than the anchor and create a small visual backtrack.  This function
    keeps the calibrated points where possible, but repairs such near-field
    kinks by interpolating toward the next valid forward point.
    """

    arr = np.asarray(trajectory_norm, dtype=np.float32).reshape(-1, 2).copy()
    target_n = int(config.num_points)
    if arr.shape[0] == 0:
        arr = np.repeat(np.asarray([[0.5, 0.9]], dtype=np.float32), target_n, axis=0)
    if arr.shape[0] < target_n:
        arr = np.concatenate([arr, np.repeat(arr[-1:], target_n - arr.shape[0], axis=0)], axis=0)
    arr = arr[:target_n]
    arr = np.clip(arr, 0.0, 1.0)
    arr[0] = [0.5, 0.9]

    min_step = max(float(config.trajectory_min_y_step), 0.0)
    for i in range(1, arr.shape[0]):
        previous = arr[i - 1].copy()
        max_allowed_y = max(0.02, float(previous[1]) - min_step)
        if float(arr[i, 1]) <= max_allowed_y:
            continue

        replacement = arr[i].copy()
        for j in range(i + 1, arr.shape[0]):
            if float(arr[j, 1]) <= max_allowed_y:
                alpha = 1.0 / float(j - i + 1)
                replacement = previous * (1.0 - alpha) + arr[j] * alpha
                break
        replacement[1] = min(float(replacement[1]), max_allowed_y)
        replacement[0] = float(np.clip(replacement[0], 0.0, 1.0))
        replacement[1] = float(np.clip(replacement[1], 0.0, 1.0))
        arr[i] = replacement

    return arr.astype(np.float32)


def build_bev_patch_meta(
    record: FrameRecord,
    local_path: np.ndarray,
    *,
    route_theta_deg: float,
    intent: str,
    anchor: str,
) -> dict[str, Any]:
    """Build local BEV route metadata used by Stage1 topology QA.

    Earlier versions also emitted coarse BEV grid cells.  Those cells were
    derived from arbitrary exporter extents rather than external annotation, so
    they are no longer used as CoT/SFT targets.  Keep the continuous local route
    polyline instead; it is directly derived from the same future trajectory as
    the FPV visual prompt target.
    """

    target_xy = local_path[-1]
    meta = {
        "schema_version": "sidewalk_bev_patch_meta_v1",
        "clip_id": record.clip_id,
        "frame_idx": record.frame_idx,
        "frame_name": record.frame_name,
        "coordinate_frame": "ego_aligned_local_meters",
        "route_polyline_local_m": local_path.astype(float).tolist(),
        "route_target_local_m": [float(target_xy[0]), float(target_xy[1])],
        "route_theta_deg": round(float(route_theta_deg), 3),
        "intent_gt": intent,
        "target_direction": anchor,
        "confidence_level_gt": "normal",
        "pose": {
            "lat": record.lat,
            "lng": record.lng,
            "heading_deg": record.heading_deg,
            "timestamp": record.timestamp,
        },
        "bev_observation": {
            "bev_image_path": str(record.bev_image_path) if record.bev_image_path else None,
            "bev_frame_idx": record.bev_frame_idx,
            "bev_timestamp": record.bev_timestamp,
            "fpv_timestamp": record.timestamp,
            "time_delta_ms": record.bev_time_delta_ms,
            "mapping_source": record.bev_mapping_source,
        },
    }
    _drop_none_fields(meta["pose"])
    return meta


def build_qa_bundle(
    *,
    record: FrameRecord,
    image_width: int,
    image_height: int,
    trajectory_norm: np.ndarray,
    trajectory_fpv_t: list[list[float]],
    trajectory_ego_t: list[list[float]],
    trajectory_policy: dict[str, Any],
    bev_meta: dict[str, Any],
    label_path: Path,
    bev_meta_path: Path,
    trajectory_source: str = "unknown",
) -> dict[str, Any]:
    route_theta = float(bev_meta["route_theta_deg"])
    intent = str(bev_meta["intent_gt"])
    anchor = str(bev_meta["target_direction"])
    qa_base = f"{record.clip_id}_{record.frame_name}"
    bev_context = build_cot_bev_context(
        bev_meta=bev_meta,
        route_theta=route_theta,
        intent=intent,
        anchor=anchor,
    )
    fpv_scene = build_cot_fpv_scene()
    cross_view = build_cot_cross_view(intent=intent, anchor=anchor)
    finish = build_cot_finish(trajectory_norm=trajectory_norm, intent=intent)
    samples = [
        {
            "sample_id": f"{qa_base}_bev_route_direction",
            "task_family": "A_bev",
            "task_type": "bev_route_direction",
            "answer_type": "direction_json",
            "question": "Estimate the route lookahead direction from the top-down BEV route context.",
            "teacher_answer": json_tag("BEVContext", bev_context),
            "structured_target": bev_context,
            "evidence": {"bev_patch_meta_path": str(bev_meta_path)},
            "valid": True,
            "confidence": 0.9,
            "difficulty": "normal",
        },
        {
            "sample_id": f"{qa_base}_bev_navigation_stage",
            "task_family": "A_bev",
            "task_type": "bev_navigation_stage",
            "answer_type": "mcq",
            "question": "Classify the local navigation stage from the BEV route context.",
            "teacher_answer": json_tag("BEVContext", bev_context),
            "structured_target": bev_context,
            "evidence": {"bev_patch_meta_path": str(bev_meta_path)},
            "valid": True,
            "confidence": 0.86,
            "difficulty": "normal",
        },
        {
            "sample_id": f"{qa_base}_crossview_topology_grounding",
            "task_family": "X_crossview",
            "task_type": "crossview_topology_grounding",
            "answer_type": "grounding_json",
            "question": "Does the FPV trajectory endpoint agree with the BEV route intent?",
            "teacher_answer": json_tag("CrossView", cross_view),
            "structured_target": cross_view,
            "evidence": {
                "trajectory_label_path": str(label_path),
                "bev_patch_meta_path": str(bev_meta_path),
            },
            "valid": True,
            "confidence": 0.78,
            "difficulty": "normal",
        },
        {
            "sample_id": f"{qa_base}_point_prompt",
            "task_family": "P_prompt",
            "task_type": "point_prompt",
            "answer_type": "point_with_theta_json",
            "question": "What is the FPV target point implied by the GT trajectory?",
            "teacher_answer": json_tag("finish", finish),
            "structured_target": finish,
            "evidence": {"trajectory_label_path": str(label_path)},
            "valid": True,
            "confidence": 0.9,
            "difficulty": "normal",
        },
        {
            "sample_id": f"{qa_base}_trajectory_intent_check",
            "task_family": "P_prompt",
            "task_type": "trajectory_intent_check",
            "answer_type": "structured_text",
            "question": "Check whether the FPV trajectory is consistent with the BEV route intent.",
            "teacher_answer": json_tag("finish", finish),
            "structured_target": finish,
            "evidence": {"trajectory_label_path": str(label_path)},
            "valid": True,
            "confidence": 0.72,
            "difficulty": "normal",
        },
    ]
    return {
        "schema_version": QA_SCHEMA_VERSION,
        "clip_name": record.clip_id,
        "fpv_frame_idx": record.frame_idx,
        "bev_frame_idx": record.bev_frame_idx if record.bev_frame_idx is not None else record.frame_idx,
        "bev_time_delta_ms": record.bev_time_delta_ms,
        "time_sec": None,
        "image_size": {"width": image_width, "height": image_height},
        "fpv_image_path": str(record.image_path),
        "bev_image_path": str(record.bev_image_path) if record.bev_image_path else None,
        "route_context": {
            "intent_gt": intent,
            "route_theta_deg": round(route_theta, 3),
            "anchor": anchor,
        },
        "local_bev_context": bev_meta,
        "segmentation_context_ref": {},
        "som_context": {},
        "gt_pixel_trajectory": trajectory_norm.astype(float).tolist(),
        "gt_trajectory_fpv_t": trajectory_fpv_t,
        "gt_trajectory_ego_t": trajectory_ego_t,
        "metadata": {
            "source": "agent_phase1_dataset_builder",
            "trajectory_source": trajectory_source,
            "trajectory_policy": trajectory_policy,
            "trajectory_label_path": str(label_path),
            "bev_patch_meta_path": str(bev_meta_path),
            "bev_mapping_source": record.bev_mapping_source,
            "phase": "phase1_no_segmentation",
        },
        "samples": samples,
    }


def _iter_clip_frame_records(
    clip_dir: Path,
    *,
    output_root: Path,
    allow_missing_bev: bool,
    allow_heuristic_trajectory: bool,
    max_bev_time_delta_ms: float | None,
) -> Iterable[FrameRecord]:
    if _is_internnav_clip(clip_dir):
        positions = np.asarray(np.load(clip_dir / "frame_positions.npy"), dtype=np.float64)
        if _positions_are_static(positions) and (clip_dir / "trajectory.json").is_file():
            if not allow_heuristic_trajectory:
                raise ValueError(
                    "static frame_positions.npy requires --allow-heuristic-trajectory "
                    f"to use trajectory.json GNSS fallback: {clip_dir}"
                )
            trajectory = load_trajectory(clip_dir / "trajectory.json")
            yield from _iter_legacy_trajectory_frame_records(
                clip_dir,
                trajectory,
                allow_missing_bev=allow_missing_bev,
                pose_fallback_reason="static_frame_positions",
            )
            return
        yield from _iter_internnav_frame_records(
            clip_dir,
            output_root=output_root,
            allow_missing_bev=allow_missing_bev,
            max_bev_time_delta_ms=max_bev_time_delta_ms,
        )
        return
    if not allow_heuristic_trajectory:
        raise ValueError("legacy trajectory.json input requires --allow-heuristic-trajectory")
    trajectory = load_trajectory(clip_dir / "trajectory.json")
    if not trajectory:
        return
    yield from _iter_legacy_trajectory_frame_records(
        clip_dir,
        trajectory,
        allow_missing_bev=allow_missing_bev,
    )


def _iter_legacy_trajectory_frame_records(
    clip_dir: Path,
    trajectory: list[dict[str, Any]],
    *,
    allow_missing_bev: bool,
    pose_fallback_reason: str | None = None,
) -> Iterable[FrameRecord]:
    clip_id = clip_dir.name
    images_dir = clip_dir / "images"
    bev_dir = clip_dir / "bev_images"
    fpv_video_path = (clip_dir / "front.mp4").resolve() if (clip_dir / "front.mp4").is_file() else None
    for fallback_idx, row in enumerate(trajectory):
        try:
            frame_idx = _optional_int(row.get("frame_idx"))
            if frame_idx is None:
                frame_idx = int(fallback_idx)
            frame_name = str(row.get("frame_name") or f"{int(frame_idx):06d}")
            try:
                image_path = _resolve_image_path(row, images_dir)
                frame_name = image_path.stem
                frame_idx = int(frame_name) if frame_name.isdigit() else int(frame_idx)
            except FileNotFoundError:
                if fpv_video_path is None:
                    continue
                image_path = None
            timestamp = _optional_int(row.get("timestamp"))
            bev_path, bev_match = _resolve_internnav_bev_match(clip_dir, frame_idx, frame_name, timestamp)
            if bev_path is None:
                bev_path = _resolve_bev_path(row, bev_dir, frame_name)
                bev_match = {
                    "bev_frame_idx": None,
                    "bev_timestamp": None,
                    "bev_time_delta_ms": None,
                    "bev_mapping_source": "exact_legacy_frame_name" if bev_path is not None else None,
                }
            if bev_path is None and not allow_missing_bev:
                continue
            yield FrameRecord(
                clip_dir=clip_dir,
                clip_id=clip_id,
                frame_name=frame_name,
                frame_idx=frame_idx,
                image_path=image_path,
                bev_image_path=bev_path,
                fpv_video_path=fpv_video_path,
                bev_frame_idx=bev_match.get("bev_frame_idx"),
                bev_timestamp=bev_match.get("bev_timestamp"),
                bev_time_delta_ms=bev_match.get("bev_time_delta_ms"),
                bev_mapping_source=bev_match.get("bev_mapping_source"),
                lat=float(row["lat"]),
                lng=float(row["lng"]),
                heading_deg=float(row.get("heading_deg", 0.0) or 0.0),
                timestamp=timestamp,
                speed_mps=_optional_float(row.get("speed_mps") if row.get("speed_mps") is not None else row.get("speed")),
                roll_deg=_optional_float(row.get("roll_deg")),
                pitch_deg=_optional_float(row.get("pitch_deg")),
                pose_source="trajectory_json_gnss",
                pose_fallback_reason=pose_fallback_reason,
            )
        except Exception:
            continue


def _iter_internnav_frame_records(
    clip_dir: Path,
    *,
    output_root: Path,
    allow_missing_bev: bool,
    max_bev_time_delta_ms: float | None,
) -> Iterable[FrameRecord]:
    clip_id = clip_dir.name
    positions = np.asarray(np.load(clip_dir / "frame_positions.npy"), dtype=np.float64)
    orientations = np.asarray(np.load(clip_dir / "frame_orientations.npy"), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"frame_positions.npy must have shape [N, >=2]: {clip_dir}")
    if orientations.ndim != 2 or orientations.shape[1] < 4:
        raise ValueError(f"frame_orientations.npy must have shape [N, >=4]: {clip_dir}")
    camera_params = _load_camera_params(clip_dir)
    orientation_order = _load_frame_orientation_order(clip_dir)
    front_video_fps = _load_front_video_fps(clip_dir)
    frame_times = _load_optional_array(clip_dir / "frame_times.npy")
    frame_speeds = _load_optional_array(clip_dir / "frame_speeds.npy")
    frame_video_indices = _load_optional_array(clip_dir / "frame_video_indices.npy")
    frame_count = min(len(positions), len(orientations))
    fpv_video_path = (clip_dir / "front.mp4").resolve() if (clip_dir / "front.mp4").is_file() else None
    for frame_idx in range(frame_count):
        image_path = _resolve_existing_internnav_image_path(clip_dir, frame_idx)
        if image_path is None and fpv_video_path is None:
            continue
        frame_name = image_path.stem if image_path is not None else f"{int(frame_idx):06d}"
        timestamp = int(frame_times[frame_idx]) if frame_times is not None and frame_idx < len(frame_times) else None
        bev_path, bev_match = _resolve_internnav_bev_match(clip_dir, frame_idx, frame_name, timestamp)
        if _bev_match_too_far(bev_match, max_bev_time_delta_ms=max_bev_time_delta_ms):
            bev_path = None
            bev_match = {
                "bev_frame_idx": None,
                "bev_timestamp": None,
                "bev_time_delta_ms": None,
                "bev_mapping_source": "rejected_by_time_delta",
            }
        if bev_path is None and not allow_missing_bev:
            continue
        yield FrameRecord(
            clip_dir=clip_dir,
            clip_id=clip_id,
            frame_name=frame_name,
            frame_idx=frame_idx,
            image_path=image_path,
            bev_image_path=bev_path,
            fpv_video_path=fpv_video_path,
            bev_frame_idx=bev_match.get("bev_frame_idx"),
            bev_timestamp=bev_match.get("bev_timestamp"),
            bev_time_delta_ms=bev_match.get("bev_time_delta_ms"),
            bev_mapping_source=bev_match.get("bev_mapping_source"),
            timestamp=timestamp,
            speed_mps=(
                float(frame_speeds[frame_idx])
                if frame_speeds is not None and frame_idx < len(frame_speeds)
                else None
            ),
            video_frame_idx=(
                int(frame_video_indices[frame_idx])
                if frame_video_indices is not None and frame_idx < len(frame_video_indices)
                else None
            ),
            video_frame_rate_hz=front_video_fps,
            position_xy=(float(positions[frame_idx, 0]), float(positions[frame_idx, 1])),
            orientation_wxyz=_orientation_row_to_wxyz(orientations[frame_idx, :4], order=orientation_order),
            camera_params=camera_params,
            pose_source="frame_positions",
        )


def _resolve_image_path(row: dict[str, Any], images_dir: Path) -> Path:
    raw = str(row.get("image_path") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return path.resolve()
        candidate = images_dir / path.name
        if candidate.is_file():
            return candidate.resolve()
    frame = str(row.get("frame_name") or row.get("frame_idx") or "").strip()
    if frame:
        stem = f"{int(frame):06d}" if frame.isdigit() else frame
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve FPV image for trajectory row under {images_dir}")


def _resolve_bev_path(row: dict[str, Any], bev_dir: Path, frame_name: str) -> Path | None:
    for key in ("bev_image_path", "bev_image_path_ego_aligned"):
        raw = str(row.get(key) or "").strip()
        if raw and Path(raw).expanduser().is_file():
            return Path(raw).expanduser().resolve()
    for suffix in ("_bev.jpg", "_bev.png", ".jpg", ".png"):
        candidate = bev_dir / f"{frame_name}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_existing_internnav_image_path(clip_dir: Path, frame_idx: int) -> Path | None:
    for image_dir_name in ("images", "front_images"):
        image_dir = clip_dir / image_dir_name
        if not image_dir.is_dir():
            continue
        prefer_one_based = image_dir_name == "front_images" or _image_dir_is_one_based(image_dir)
        for stem in _frame_name_candidates(frame_idx, prefer_one_based=prefer_one_based):
            for suffix in (".jpg", ".jpeg", ".png"):
                candidate = image_dir / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate.resolve()
    return None


def materialize_fpv_image(record: FrameRecord, output_root: Path, *, frame_rate_hz: float | None = None) -> FrameRecord:
    """Ensure the selected FPV sample has a JPG without extracting full clips."""

    if record.image_path is not None and record.image_path.is_file():
        return record
    if record.fpv_video_path is None or not record.fpv_video_path.is_file():
        raise FileNotFoundError(f"Cannot resolve FPV image for {record.clip_id}:{record.frame_idx}")
    image_path = _extract_video_frame(
        record.fpv_video_path,
        record.video_frame_idx if record.video_frame_idx is not None else record.frame_idx,
        output_root / "clips" / record.clip_id / "fpv_frames",
        frame_rate_hz=record.video_frame_rate_hz if record.video_frame_idx is not None else frame_rate_hz,
    )
    if image_path is None:
        raise FileNotFoundError(f"Cannot extract FPV image for {record.clip_id}:{record.frame_idx}")
    return replace(record, image_path=image_path)


def _resolve_internnav_bev_match(
    clip_dir: Path,
    frame_idx: int,
    frame_name: str,
    timestamp: int | None,
) -> tuple[Path | None, dict[str, Any]]:
    for bev_dir_name in ("bev_images", "bev_images_ego_aligned"):
        bev_dir = clip_dir / bev_dir_name
        if not bev_dir.is_dir():
            continue
        prefer_one_based = _bev_dir_is_one_based(bev_dir)
        exact_idx = int(frame_idx) + 1 if prefer_one_based else int(frame_idx)
        stems = [frame_name, f"{exact_idx:06d}", str(exact_idx)]
        for stem in dict.fromkeys(stems):
            for suffix in ("_bev.jpg", "_bev.png", ".jpg", ".png"):
                candidate = bev_dir / f"{stem}{suffix}"
                if candidate.is_file():
                    meta = _read_bev_image_meta(candidate)
                    return candidate.resolve(), _bev_match_meta(
                        path=candidate,
                        query_frame_idx=frame_idx,
                        query_timestamp=timestamp,
                        raw_meta=meta,
                        mapping_source="exact_frame_name",
                    )

    entries = _load_bev_index_entries(clip_dir)
    if not entries:
        return None, {}
    best = _nearest_bev_entry(entries, frame_idx=frame_idx, timestamp=timestamp)
    if best is None:
        return None, {}
    path = Path(str(best["path"])).expanduser()
    if not path.is_file():
        return None, {}
    return path.resolve(), _bev_match_meta(
        path=path,
        query_frame_idx=frame_idx,
        query_timestamp=timestamp,
        raw_meta=best,
        mapping_source="nearest_timestamp" if timestamp is not None and best.get("timestamp") is not None else "nearest_frame_idx",
    )


def _read_bev_image_meta(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(path.suffix + ".bevmeta.json")
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_bev_index_entries(clip_dir: Path) -> list[dict[str, Any]]:
    return list(_load_bev_index_entries_cached(str(clip_dir.resolve())))


@lru_cache(maxsize=256)
def _load_bev_index_entries_cached(clip_dir_text: str) -> tuple[dict[str, Any], ...]:
    clip_dir = Path(clip_dir_text)
    for index_path in (clip_dir / "bev_index.json", clip_dir / "bev_images" / "bev_index.json"):
        if index_path.is_file():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                entries = data.get("entries") if isinstance(data, dict) else None
                if isinstance(entries, list):
                    return tuple(_normalize_bev_index_entry(item, index_path.parent) for item in entries if isinstance(item, dict))
            except Exception:
                pass

    entries: list[dict[str, Any]] = []
    for bev_dir_name in ("bev_images", "bev_images_ego_aligned"):
        bev_dir = clip_dir / bev_dir_name
        if not bev_dir.is_dir():
            continue
        for meta_path in sorted(bev_dir.glob("*.bevmeta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            image_path = Path(str(meta_path)[: -len(".bevmeta.json")])
            if not image_path.is_file():
                continue
            item = dict(meta)
            item["path"] = str(image_path)
            entries.append(_normalize_bev_index_entry(item, bev_dir))
    return tuple(entries)


def _normalize_bev_index_entry(item: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    raw_path = item.get("bev_image_path") or item.get("path")
    path = Path(str(raw_path)).expanduser() if raw_path else base_dir / f"{item.get('frame_name', '')}_bev.jpg"
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    frame_idx = _optional_int(item.get("frame_idx") if item.get("frame_idx") is not None else item.get("source_frame_idx"))
    timestamp = _optional_int(item.get("timestamp") if item.get("timestamp") is not None else item.get("source_timestamp_ns"))
    frame_name = str(item.get("frame_name") or item.get("source_frame_name") or (f"{frame_idx:06d}" if frame_idx is not None else path.stem))
    return {
        "path": str(path),
        "frame_idx": frame_idx,
        "frame_name": frame_name,
        "timestamp": timestamp,
        "bev_sample_fps": item.get("bev_sample_fps"),
        "bev_alignment": item.get("bev_alignment") or "ego_aligned",
    }


def _nearest_bev_entry(entries: list[dict[str, Any]], *, frame_idx: int, timestamp: int | None) -> dict[str, Any] | None:
    candidates = entries
    if not candidates:
        return None
    if timestamp is not None and any(entry.get("timestamp") is not None for entry in candidates):
        return min(
            candidates,
            key=lambda entry: abs(int(entry.get("timestamp")) - int(timestamp)) if entry.get("timestamp") is not None else 10**30,
        )
    return min(
        candidates,
        key=lambda entry: abs(int(entry.get("frame_idx")) - int(frame_idx)) if entry.get("frame_idx") is not None else 10**12,
    )


def _bev_match_meta(
    *,
    path: Path,
    query_frame_idx: int,
    query_timestamp: int | None,
    raw_meta: dict[str, Any],
    mapping_source: str,
) -> dict[str, Any]:
    bev_frame_idx = _optional_int(raw_meta.get("frame_idx") if raw_meta.get("frame_idx") is not None else raw_meta.get("source_frame_idx"))
    bev_timestamp = _optional_int(raw_meta.get("timestamp") if raw_meta.get("timestamp") is not None else raw_meta.get("source_timestamp_ns"))
    if bev_frame_idx is None:
        stem = path.name.split("_bev", 1)[0]
        bev_frame_idx = int(stem) if stem.isdigit() else None
    delta_ms = None
    if query_timestamp is not None and bev_timestamp is not None:
        delta_ms = abs(float(int(query_timestamp) - int(bev_timestamp))) / 1e6
    return {
        "bev_frame_idx": bev_frame_idx,
        "bev_timestamp": bev_timestamp,
        "bev_time_delta_ms": delta_ms,
        "bev_mapping_source": mapping_source,
    }


def _bev_match_too_far(bev_match: dict[str, Any], *, max_bev_time_delta_ms: float | None) -> bool:
    """Return True when a nearest-BEV match is outside the accepted time window."""

    if max_bev_time_delta_ms is None or float(max_bev_time_delta_ms) <= 0:
        return False
    delta = bev_match.get("bev_time_delta_ms")
    if delta is None:
        return False
    try:
        return float(delta) > float(max_bev_time_delta_ms)
    except Exception:
        return False


def _image_dir_is_one_based(image_dir: Path) -> bool:
    if any((image_dir / f"000000{suffix}").is_file() for suffix in (".jpg", ".jpeg", ".png")):
        return False
    return any((image_dir / f"000001{suffix}").is_file() for suffix in (".jpg", ".jpeg", ".png"))


def _bev_dir_is_one_based(bev_dir: Path) -> bool:
    zero_names = ["000000_bev.jpg", "000000_bev.png", "000000.jpg", "000000.png"]
    one_names = ["000001_bev.jpg", "000001_bev.png", "000001.jpg", "000001.png"]
    if any((bev_dir / name).is_file() for name in zero_names):
        return False
    return any((bev_dir / name).is_file() for name in one_names)


def _frame_name_candidates(frame_idx: int, *, prefer_one_based: bool = False) -> list[str]:
    zero_based = [
        f"{int(frame_idx):06d}",
        str(int(frame_idx)),
    ]
    one_based = [
        f"{int(frame_idx) + 1:06d}",
        str(int(frame_idx) + 1),
    ]
    return one_based + zero_based if prefer_one_based else zero_based + one_based


def _can_try_cv2() -> bool:
    """Avoid noisy OpenCV imports when the environment has a known NumPy ABI mismatch."""

    if os.environ.get("SIDEWALK_PHASE1_TRY_CV2") in {"1", "true", "TRUE", "yes"}:
        return True
    try:
        import numpy as np  # type: ignore

        major = int(str(np.__version__).split(".", 1)[0])
        if major >= 2:
            return False
    except Exception:
        pass
    return True


def _extract_video_frame(video_path: Path, frame_idx: int, output_dir: Path, *, frame_rate_hz: float | None = None) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{int(frame_idx):06d}.jpg"
    if output_path.is_file():
        return output_path.resolve()
    try:
        seek_seconds = None
        if frame_rate_hz is not None and float(frame_rate_hz) > 0:
            seek_seconds = max(0.0, float(frame_idx) / float(frame_rate_hz))
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]
        if seek_seconds is not None:
            cmd.extend(["-ss", f"{seek_seconds:.6f}"])
        cmd.extend(["-i", str(video_path)])
        if seek_seconds is None:
            cmd.extend(["-vf", f"select=eq(n\\,{int(frame_idx)})", "-vframes", "1"])
        else:
            cmd.extend(["-frames:v", "1"])
        cmd.append(str(output_path))
        subprocess.run(
            cmd,
            check=True,
        )
        if output_path.is_file():
            return output_path.resolve()
    except Exception:
        pass
    try:
        if not _can_try_cv2():
            raise ImportError("Skipping cv2 because NumPy >= 2 is active; use ffmpeg/decord fallback")
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = capture.read()
            if ok and frame is not None:
                cv2.imwrite(str(output_path), frame)
                return output_path.resolve()
        finally:
            capture.release()
    except Exception:
        pass
    try:
        import decord  # type: ignore

        reader = decord.VideoReader(str(video_path))
        if int(frame_idx) < len(reader):
            image = Image.fromarray(reader[int(frame_idx)].asnumpy())
            image.save(output_path, quality=95)
            return output_path.resolve()
    except Exception:
        pass
    return None


def local_xy_m(lat0: float, lng0: float, lat: float, lng: float, heading_deg: float) -> tuple[float, float]:
    """Project WGS84 delta into ego local meters, x=right, y=front."""

    mean_lat = math.radians((float(lat0) + float(lat)) * 0.5)
    east_m = (float(lng) - float(lng0)) * 111320.0 * max(0.1, math.cos(mean_lat))
    north_m = (float(lat) - float(lat0)) * 110540.0
    theta = math.radians(float(heading_deg))
    x_m = east_m * math.cos(theta) - north_m * math.sin(theta)
    y_m = east_m * math.sin(theta) + north_m * math.cos(theta)
    return float(x_m), float(y_m)


def bev_theta_deg(x_m: float, y_m: float) -> float:
    return max(-90.0, min(90.0, math.degrees(math.atan2(float(x_m), max(float(y_m), 1e-6)))))


def quat_to_yaw(quat_wxyz: tuple[float, float, float, float]) -> float:
    w, _x, _y, z = quat_wxyz
    return float(2.0 * math.atan2(float(z), float(w)))


def robot_forward_left_m(
    current_xy: tuple[float, float],
    current_quat_wxyz: tuple[float, float, float, float],
    target_xy: tuple[float, float],
) -> tuple[float, float]:
    """Transform map/odom XY into current robot frame, x=forward, y=left."""

    yaw = quat_to_yaw(current_quat_wxyz)
    dx = float(target_xy[0]) - float(current_xy[0])
    dy = float(target_xy[1]) - float(current_xy[1])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x_forward = dx * cos_yaw + dy * sin_yaw
    y_left = -dx * sin_yaw + dy * cos_yaw
    return float(x_forward), float(y_left)


def project_pose_to_fpv(
    current_xy: tuple[float, float],
    current_quat_wxyz: tuple[float, float, float, float],
    target_xy: tuple[float, float],
    camera: CameraParams,
) -> tuple[float, float]:
    """Project one future odom/map point to FPV pixels with InternNav convention."""

    x_forward, y_left = robot_forward_left_m(current_xy, current_quat_wxyz, target_xy)
    z_cam = x_forward
    x_cam = -y_left
    y_cam = -float(camera.camera_height_meters)
    if z_cam <= 0.01:
        return float(camera.image_width) / 2.0, float(camera.image_height) * 0.9
    fx, fy = float(camera.K[0, 0]), float(camera.K[1, 1])
    cx, cy = float(camera.K[0, 2]), float(camera.K[1, 2])
    u = fx * x_cam / z_cam + cx
    v = cy - fy * y_cam / z_cam
    u = float(np.clip(u, 0.0, float(camera.image_width) - 1.0))
    v = float(np.clip(v, 0.0, float(camera.image_height) - 1.0))
    return u, v


def intent_from_theta(theta_deg: float, *, distance_m: float) -> str:
    if distance_m < 0.75:
        return "stop"
    if theta_deg <= -25.0:
        return "turn_left"
    if theta_deg >= 25.0:
        return "turn_right"
    return "go_straight"


def anchor_from_theta(theta_deg: float) -> str:
    if theta_deg <= -45.0:
        return "left_front"
    if theta_deg <= -15.0:
        return "slight_left_front"
    if theta_deg >= 45.0:
        return "right_front"
    if theta_deg >= 15.0:
        return "slight_right_front"
    return "front"


def stage_from_intent(intent: str) -> str:
    if intent == "stop":
        return "stop_or_arrive"
    if intent in {"turn_left", "turn_right"}:
        return "approach_turn"
    return "straight"


def json_tag(tag: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"<{tag}>{body}</{tag}>"


def build_cot_bev_context(
    *,
    bev_meta: dict[str, Any],
    route_theta: float,
    intent: str,
    anchor: str,
) -> dict[str, Any]:
    return {
        "scene_summary": "local BEV route context",
        "route_summary": f"route continues from ego toward {anchor}",
        "route_direction_bin": route_direction_bin(route_theta, intent, anchor),
        "intent": intent,
        "target_direction": anchor,
        "bev_anchor": anchor
    }


def build_cot_fpv_scene() -> dict[str, Any]:
    return {
        "navigable_support": ["visible_sidewalk_or_pedestrian_support"],
        "conditional_support": [],
        "risk_objects": [],
        "visibility": "uncertain",
        "local_risk": "unknown",
    }


def build_cot_cross_view(*, intent: str, anchor: str) -> dict[str, Any]:
    region = fpv_region_from_direction(anchor)
    return {
        "bev_anchor": anchor,
        "fpv_anchor": f"{region}_visible_navigation_support",
        "near_reference": "trajectory_endpoint",
        "blind_area_context": {
            "left": "unknown without dense region evidence",
            "right": "unknown without dense region evidence",
            "front": "trajectory endpoint indicates visible support",
        },
        "fpv_route_region": region,
        "route_to_fpv_direction": route_to_fpv_direction(anchor),
        "safe_corridor": safe_corridor_from_region(region),
        "alignment": "aligned",
        "confidence": "medium",
    }


def build_cot_finish(*, trajectory_norm: np.ndarray, intent: str) -> dict[str, Any]:
    trajectory = np.asarray(trajectory_norm, dtype=np.float32).reshape(-1, 2).astype(float).tolist()
    return {
        "trajectory": trajectory,
        "intent": intent,
        "confidence_level": "normal",
        "trajectory_type": trajectory_type(trajectory),
        "trajectory_validity": "valid",
    }


def route_direction_bin(theta_deg: float, intent: str, target_direction: str) -> str:
    if str(intent) == "stop" or str(target_direction) == "stop":
        return "stop"
    theta = float(theta_deg)
    if theta <= -15.0:
        return "left"
    if theta < -5.0:
        return "slight_left"
    if theta <= 5.0:
        return "front"
    if theta <= 15.0:
        return "slight_right"
    return "right"



def fpv_region_from_direction(direction: str) -> str:
    if direction == "left_front":
        return "left"
    if direction == "right_front":
        return "right"
    if direction == "stop":
        return "unknown"
    return "center"


def route_to_fpv_direction(direction: str) -> str:
    return {
        "left_front": "forward_left",
        "right_front": "forward_right",
        "front": "forward_center",
        "stop": "stop",
    }.get(direction, "unknown")


def safe_corridor_from_region(region: str) -> str:
    return {
        "left": "left_corridor",
        "right": "right_corridor",
        "center": "center_corridor",
        "unknown": "unknown",
    }.get(region, "unknown")


def trajectory_type(trajectory: list[list[float]]) -> str:
    arr = np.asarray(trajectory or [], dtype=np.float32).reshape(-1, 2)
    if arr.size == 0:
        return "unknown"
    if len(arr) > 1 and float(np.max(arr[:, 1]) - np.min(arr[:, 1])) < 0.08:
        return "stop_hold"
    delta_x = float(arr[-1, 0] - arr[0, 0]) if len(arr) > 1 else 0.0
    if delta_x < -0.035:
        return "slight_left_center"
    if delta_x > 0.035:
        return "slight_right_center"
    return "straight_center"


def _drop_none_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Remove optional fields that are absent before persisting JSON artifacts."""

    for key in [key for key, value in data.items() if value is None]:
        data.pop(key, None)
    return data


def resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((int(num_points), 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, int(num_points), axis=0)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    if float(cumulative[-1]) <= 1e-6:
        return np.repeat(points[:1], int(num_points), axis=0)
    target = np.linspace(0.0, float(cumulative[-1]), int(num_points))
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def fpv_trajectory_from_local_path(local_path: np.ndarray, *, config: Phase1Config) -> np.ndarray:
    """Convert local meters to normalized FPV waypoints with a conservative camera heuristic."""

    points = np.asarray(local_path, dtype=np.float32).reshape(-1, 2)
    hfov_rad = math.radians(max(float(config.hfov_deg), 1.0))
    distances = np.linalg.norm(points, axis=1)
    max_distance = max(float(np.max(distances)), 1.0)
    result = []
    for i, (x_m, y_m) in enumerate(points):
        theta = math.atan2(float(x_m), max(float(y_m), 1e-3))
        x_norm = 0.5 + theta / hfov_rad
        progress = min(float(distances[i]) / max_distance, 1.0)
        y_norm = 0.9 - 0.62 * progress
        result.append([float(np.clip(x_norm, 0.02, 0.98)), float(np.clip(y_norm, 0.12, 0.92))])
    if result:
        result[0] = [0.5, 0.9]
    return np.asarray(result, dtype=np.float32)


def to_pixel_trajectory(trajectory_norm: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    arr = np.asarray(trajectory_norm, dtype=np.float32).reshape(-1, 2).copy()
    arr[:, 0] *= float(image_width)
    arr[:, 1] *= float(image_height)
    return arr


def write_debug_overlay(image_path: Path, trajectory_norm: np.ndarray, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    points = to_pixel_trajectory(trajectory_norm, image.width, image.height)
    xy = [(float(x), float(y)) for x, y in points]
    if len(xy) >= 2:
        draw.line(xy, fill=(255, 0, 255), width=5)
    for i, (x, y) in enumerate(xy):
        radius = 5 if i else 7
        fill = (34, 197, 94) if i == 0 else (255, 0, 255)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def _drop_near_or_behind(points: np.ndarray, *, min_distance_m: float) -> np.ndarray:
    kept = []
    for x_m, y_m in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        if float(y_m) <= 0.05:
            continue
        if math.hypot(float(x_m), float(y_m)) < float(min_distance_m):
            continue
        kept.append([float(x_m), float(y_m)])
    return np.asarray(kept, dtype=np.float32).reshape(-1, 2)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _inc(counter: dict[str, Any], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SideWalk Phase1 source JSONL and QA artifacts")
    parser.add_argument("--sidewalk-root", required=True, help="Root containing prepared SideWalk clips")
    parser.add_argument("--output-root", required=True, help="Output root for Phase1 artifacts")
    parser.add_argument("--split-name", default="train", help="Manifest split name")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of source samples to write across all clips.")
    parser.add_argument("--max-clips", type=int, default=None, help="Maximum number of discovered clips to process, after sorted discovery order.")
    parser.add_argument("--frame-step", type=int, default=10, help="Generate one sample every N FPV frames. Default 10 gives about 2fps for 20Hz FPV clips.")
    parser.add_argument("--future-frames", type=int, default=50, help="Fallback number of future frames when timestamps are unavailable or --future-horizon-sec <= 0.")
    parser.add_argument(
        "--future-horizon-sec",
        type=float,
        default=2.5,
        help="Preferred future trajectory time horizon in seconds. Use <=0 to fall back to --future-frames.",
    )
    parser.add_argument("--num-points", type=int, default=16)
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--min-future-distance-m", type=float, default=0.3)
    parser.add_argument("--trajectory-min-y-step", type=float, default=0.005)
    parser.add_argument("--frame-rate-hz", type=float, default=20.0, help="Fallback FPV frame rate used to derive dt when frame_times.npy is missing.")
    parser.add_argument(
        "--max-bev-time-delta-ms",
        type=float,
        default=300.0,
        help="Maximum allowed FPV/BEV timestamp delta. Use <=0 to disable this check.",
    )
    parser.add_argument("--allow-missing-bev", action="store_true")
    parser.add_argument(
        "--allow-heuristic-trajectory",
        action="store_true",
        help="Allow legacy GNSS/HFOV or uncalibrated pose fallback. Disabled by default for production labels.",
    )
    parser.add_argument("--write-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Phase1Config(
        num_points=int(args.num_points),
        future_frames=int(args.future_frames),
        future_horizon_sec=(float(args.future_horizon_sec) if float(args.future_horizon_sec) > 0.0 else None),
        frame_step=int(args.frame_step),
        hfov_deg=float(args.hfov_deg),
        min_future_distance_m=float(args.min_future_distance_m),
        trajectory_min_y_step=float(args.trajectory_min_y_step),
        max_bev_time_delta_ms=float(args.max_bev_time_delta_ms),
        frame_rate_hz=float(args.frame_rate_hz),
    )
    report = build_phase1_dataset(
        args.sidewalk_root,
        args.output_root,
        split_name=str(args.split_name),
        config=config,
        limit=args.limit,
        max_clips=args.max_clips,
        allow_missing_bev=bool(args.allow_missing_bev),
        allow_heuristic_trajectory=bool(args.allow_heuristic_trajectory),
        write_overlays=bool(args.write_overlays),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
