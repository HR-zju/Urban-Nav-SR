#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SAM3_SCRIPT = "/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/prepare_frames_and_run_batch_sam3.py"


def _sorted_frame_files(camera_dir: Path) -> list[Path]:
    frames = sorted(camera_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No extracted JPG frames found: {camera_dir}")
    return frames


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise RuntimeError(f"Missing CSV file: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_sequence_csv(sequence_dir: Path, prefix: str) -> Path:
    matches = sorted(sequence_dir.glob(f"{prefix}_*.csv"))
    if not matches:
        raise RuntimeError(f"Missing {prefix}_*.csv in {sequence_dir}")
    return matches[0]


def _load_front_timestamps(sequence_dir: Path) -> list[dict[str, Any]]:
    csv_path = _find_sequence_csv(sequence_dir, "front_camera_timestamps")
    rows = _load_csv_rows(csv_path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            parsed.append(
                {
                    "frame_id": int(str(row.get("frame_id", "")).strip()),
                    "timestamp_s": float(str(row.get("timestamp", "")).strip()),
                }
            )
        except Exception:
            continue
    if not parsed:
        raise RuntimeError(f"No usable front camera timestamps in {csv_path}")
    return parsed


def _load_gps_samples(sequence_dir: Path) -> list[dict[str, Any]]:
    csv_path = _find_sequence_csv(sequence_dir, "gps_data")
    rows = _load_csv_rows(csv_path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            lat = float(str(row.get("latitude", "")).strip())
            lng = float(str(row.get("longitude", "")).strip())
            timestamp_ms = int(float(str(row.get("timestamp", "")).strip()))
        except Exception:
            continue
        parsed.append({"lat": lat, "lng": lng, "timestamp_ms": timestamp_ms})
    if not parsed:
        raise RuntimeError(f"No usable GPS samples in {csv_path}")
    return parsed


def _safe_json_loads(text: str | None) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _load_imu_samples(sequence_dir: Path) -> list[dict[str, Any]]:
    csv_path = _find_sequence_csv(sequence_dir, "imu_data")
    rows = _load_csv_rows(csv_path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            timestamp_ms = int(float(str(row.get("timestamp", "")).strip()))
        except Exception:
            continue
        parsed.append(
            {
                "timestamp_ms": timestamp_ms,
                "compass": _safe_json_loads(row.get("compass")),
                "accelerometer": _safe_json_loads(row.get("accelerometer")),
                "gyroscope": _safe_json_loads(row.get("gyroscope")),
            }
        )
    if not parsed:
        raise RuntimeError(f"No usable IMU samples in {csv_path}")
    return parsed


def _nearest_index(sorted_values: list[float | int], target: float | int) -> int:
    if not sorted_values:
        raise RuntimeError("Cannot search nearest item in an empty sequence")
    position = bisect_left(sorted_values, target)
    if position <= 0:
        return 0
    if position >= len(sorted_values):
        return len(sorted_values) - 1
    before = sorted_values[position - 1]
    after = sorted_values[position]
    if abs(float(after) - float(target)) < abs(float(target) - float(before)):
        return position
    return position - 1


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    y = math.sin(delta_lon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def _estimate_heading_from_gps(gps_samples: list[dict[str, Any]], gps_index: int) -> float | None:
    if not gps_samples:
        return None
    if len(gps_samples) == 1:
        return None
    current = gps_samples[gps_index]
    if gps_index + 1 < len(gps_samples):
        nxt = gps_samples[gps_index + 1]
        if abs(float(nxt["lat"]) - float(current["lat"])) > 1e-9 or abs(float(nxt["lng"]) - float(current["lng"])) > 1e-9:
            return _bearing_deg(current["lat"], current["lng"], nxt["lat"], nxt["lng"])
    if gps_index > 0:
        prev = gps_samples[gps_index - 1]
        if abs(float(prev["lat"]) - float(current["lat"])) > 1e-9 or abs(float(prev["lng"]) - float(current["lng"])) > 1e-9:
            return _bearing_deg(prev["lat"], prev["lng"], current["lat"], current["lng"])
    return None


def _extract_compass_vector(imu_sample: dict[str, Any]) -> dict[str, float] | None:
    compass = imu_sample.get("compass")
    if not isinstance(compass, list) or not compass:
        return None
    first = compass[0]
    if not isinstance(first, list) or len(first) < 3:
        return None
    try:
        return {
            "x": float(first[0]),
            "y": float(first[1]),
            "z": float(first[2]),
        }
    except Exception:
        return None


def _compass_heading_candidate_deg(compass_vector: dict[str, float] | None) -> float | None:
    if not compass_vector:
        return None
    x = float(compass_vector.get("x", 0.0))
    y = float(compass_vector.get("y", 0.0))
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return None
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _load_class_mapping(mapping_path: Path) -> tuple[dict[int, str], dict[str, int]]:
    if not mapping_path.exists():
        raise RuntimeError(f"Missing class mapping: {mapping_path}")
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw = payload.get("class_to_idx") or {}
    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    for name, class_id in raw.items():
        try:
            numeric_id = int(class_id)
        except Exception:
            continue
        name_to_id[str(name)] = numeric_id
        id_to_name[numeric_id] = str(name)
    return id_to_name, name_to_id


def _summarize_segmentation(mask_path: Path, mapping_path: Path) -> dict[str, Any]:
    id_to_name, name_to_id = _load_class_mapping(mapping_path)
    mask = np.load(mask_path)
    unique_ids, counts = np.unique(mask, return_counts=True)
    present_classes: list[dict[str, Any]] = []
    for class_id, pixel_count in sorted(
        zip(unique_ids.tolist(), counts.tolist()),
        key=lambda item: item[1],
        reverse=True,
    ):
        present_classes.append(
            {
                "class_id": int(class_id),
                "class_name": id_to_name.get(int(class_id), f"unknown_{class_id}"),
                "pixel_count": int(pixel_count),
                "pixel_ratio": round(float(pixel_count) / float(mask.size), 6),
            }
        )
    traversable_names = ["sidewalk", "crosswalk", "road", "driveway"]
    traversable_ids = [name_to_id[name] for name in traversable_names if name in name_to_id]
    traversable_pixels = 0
    if traversable_ids:
        traversable_pixels = int(np.isin(mask, traversable_ids).sum())
    return {
        "mask_path": str(mask_path),
        "class_mapping_path": str(mapping_path),
        "shape_hw": [int(mask.shape[0]), int(mask.shape[1])],
        "dtype": str(mask.dtype),
        "present_classes": present_classes,
        "traversable_class_ids": traversable_ids,
        "traversable_class_names": [id_to_name[class_id] for class_id in traversable_ids if class_id in id_to_name],
        "traversable_pixel_ratio": round(float(traversable_pixels) / float(mask.size), 6) if mask.size else 0.0,
    }


def _run_sam3_pipeline(args: argparse.Namespace) -> None:
    command = [
        "conda",
        "run",
        "-n",
        args.sam3_env_name,
        "python",
        args.sam3_script,
        "--rides_base_dir",
        args.rides_base_dir,
        "--frames_base_dir",
        args.frames_base_dir,
        "--inference_output_base_dir",
        args.sam3_output_base_dir,
        "--recordings_subdir",
        args.recordings_subdir,
        "--stream_pattern",
        args.stream_pattern,
        "--camera_view",
        args.camera_view,
        "--sequence_names",
        args.sequence_name,
        "--fps",
        str(args.fps),
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--min_score",
        str(args.min_score),
        "--finetuned",
        "true" if args.finetuned else "false",
        "--checkpoint_path",
        args.checkpoint_path,
    ]
    subprocess.run(command, check=True)


def _build_agent_command(sample: dict[str, Any], prompt: str, mode: str) -> str:
    gps = sample["gps"]
    heading = sample.get("heading_deg")
    image_path = sample["image_path"]
    parts = [
        "python",
        "examples/zero_shot_visual_prompt_agent_client.py",
        "--prompt",
        json.dumps(prompt, ensure_ascii=False),
        "--lat",
        str(gps["lat"]),
        "--lon",
        str(gps["lng"]),
        "--mode",
        mode,
        "--image-path",
        json.dumps(image_path, ensure_ascii=False),
    ]
    if heading is not None:
        parts.extend(["--heading-deg", str(heading)])
    segmentation = sample.get("fpv_segmentation") or {}
    segmentation_npy_path = segmentation.get("segmentation_npy_path")
    if isinstance(segmentation_npy_path, str) and segmentation_npy_path.strip():
        parts.extend(["--fpv-segmentation-npy", json.dumps(segmentation_npy_path, ensure_ascii=False)])
        summary = segmentation.get("summary") or {}
        class_mapping_path = summary.get("class_mapping_path")
        if isinstance(class_mapping_path, str) and class_mapping_path.strip():
            parts.extend(["--fpv-segmentation-class-mapping", json.dumps(class_mapping_path, ensure_ascii=False)])
        visualization_path = segmentation.get("visualization_path")
        if isinstance(visualization_path, str) and visualization_path.strip():
            parts.extend(["--fpv-segmentation-visualization", json.dumps(visualization_path, ensure_ascii=False)])
    return " ".join(parts)


def _build_manifest(
    *,
    sequence_dir: Path,
    frames_dir: Path,
    sam3_dir: Path,
    fps: float,
    prompt: str | None,
    mode: str,
    frame_name: str | None,
    frame_limit: int | None,
    use_imu_heading_fallback: bool,
) -> dict[str, Any]:
    if fps <= 0:
        raise RuntimeError("--fps must be positive so extracted frames can be aligned back to source timestamps")

    extracted_frames = _sorted_frame_files(frames_dir)
    front_timestamps = _load_front_timestamps(sequence_dir)
    gps_samples = _load_gps_samples(sequence_dir)
    imu_samples = _load_imu_samples(sequence_dir)
    gps_times = [sample["timestamp_ms"] for sample in gps_samples]
    imu_times = [sample["timestamp_ms"] for sample in imu_samples]

    mapping_path = sam3_dir / "class_mapping.json"
    front_start_s = float(front_timestamps[0]["timestamp_s"])

    selected_frames = extracted_frames
    if frame_name:
        selected_frames = [frame for frame in extracted_frames if frame.stem == frame_name]
        if not selected_frames:
            raise RuntimeError(f"Frame {frame_name} not found in {frames_dir}")
    elif frame_limit is not None and frame_limit > 0:
        selected_frames = extracted_frames[:frame_limit]

    samples: list[dict[str, Any]] = []
    for extracted_index, image_path in enumerate(selected_frames, start=1):
        original_extracted_index = extracted_frames.index(image_path) + 1
        approx_timestamp_s = front_start_s + float(original_extracted_index - 1) / float(fps)
        front_idx = _nearest_index([entry["timestamp_s"] for entry in front_timestamps], approx_timestamp_s)
        front_sample = front_timestamps[front_idx]
        target_timestamp_ms = int(round(float(front_sample["timestamp_s"]) * 1000.0))
        gps_idx = _nearest_index(gps_times, target_timestamp_ms)
        imu_idx = _nearest_index(imu_times, target_timestamp_ms)
        gps_sample = gps_samples[gps_idx]
        imu_sample = imu_samples[imu_idx]
        heading_deg = _estimate_heading_from_gps(gps_samples, gps_idx)

        mask_path = sam3_dir / f"{image_path.stem}_segmentation.npy"
        visualization_path = sam3_dir / f"{image_path.stem}_visualization.jpg"
        segmentation = _summarize_segmentation(mask_path, mapping_path) if mask_path.exists() else None
        compass_vector = _extract_compass_vector(imu_sample)
        imu_heading_candidate = _compass_heading_candidate_deg(compass_vector)
        resolved_heading = heading_deg
        if resolved_heading is None and use_imu_heading_fallback:
            resolved_heading = imu_heading_candidate

        sample: dict[str, Any] = {
            "sequence_name": sequence_dir.name,
            "camera_view": frames_dir.name,
            "frame_name": image_path.stem,
            "extracted_frame_index": original_extracted_index,
            "source_frame_id": int(front_sample["frame_id"]),
            "image_path": str(image_path),
            "source_timestamp_s": float(front_sample["timestamp_s"]),
            "gps": {
                "lat": float(gps_sample["lat"]),
                "lng": float(gps_sample["lng"]),
                "timestamp_ms": int(gps_sample["timestamp_ms"]),
            },
            "imu": {
                "timestamp_ms": int(imu_sample["timestamp_ms"]),
                "compass_vector": compass_vector,
                "heading_candidate_deg": round(float(imu_heading_candidate), 3) if imu_heading_candidate is not None else None,
            },
            "heading_deg": round(float(resolved_heading), 3) if resolved_heading is not None else None,
            "heading_source": "gps_bearing" if heading_deg is not None else ("imu_compass_raw" if resolved_heading is not None else None),
            "fpv_segmentation": {
                "segmentation_npy_path": str(mask_path) if mask_path.exists() else None,
                "visualization_path": str(visualization_path) if visualization_path.exists() else None,
                "summary": segmentation,
            },
        }

        project_input = {
            "image_path": sample["image_path"],
            "gps": {"lat": sample["gps"]["lat"], "lng": sample["gps"]["lng"]},
            "heading_deg": sample["heading_deg"],
            "fpv_segmentation": sample["fpv_segmentation"],
        }
        sample["project_input"] = project_input

        if prompt:
            sample["agent_demo"] = {
                "prompt": prompt,
                "mode": mode,
                "command": _build_agent_command(sample, prompt=prompt, mode=mode),
            }
        samples.append(sample)

    return {
        "dataset_type": "fodorobot_fpv_with_sam3",
        "sequence_name": sequence_dir.name,
        "camera_view": frames_dir.name,
        "fps": fps,
        "sample_count": len(samples),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 for a FodoRobot ride and convert FPV segmentation outputs into an UrbanNav project input manifest."
    )
    parser.add_argument("--rides-base-dir", required=True, help="Base directory of FodoRobot ride folders")
    parser.add_argument("--frames-base-dir", required=True, help="Base directory of extracted frame folders")
    parser.add_argument("--sam3-output-base-dir", required=True, help="Base directory of SAM3 frame outputs")
    parser.add_argument("--sequence-name", required=True, help="Ride sequence name, e.g. ride_16446_20240115041752")
    parser.add_argument("--camera-view", default="camera_front", help="Camera view name used by SAM3")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS used by SAM3 preprocessing")
    parser.add_argument("--mode", default="foot", choices=["foot", "bike", "car"], help="Suggested routing mode for demo command")
    parser.add_argument("--prompt", default=None, help="Optional navigation prompt for generating a ready-to-run agent command")
    parser.add_argument("--frame-name", default=None, help="Optional extracted frame stem, e.g. 000005")
    parser.add_argument("--frame-limit", type=int, default=1, help="How many samples to emit when --frame-name is omitted")
    parser.add_argument("--save-manifest", default=None, help="Optional JSON output path")
    parser.add_argument("--use-imu-heading-fallback", action="store_true", help="Use raw IMU compass XY as heading when GPS bearing is unavailable")
    parser.add_argument("--run-sam3", action="store_true", help="Run the external SAM3 extraction+inference pipeline before building the manifest")
    parser.add_argument("--sam3-env-name", default="sam3", help="Conda env name used for SAM3")
    parser.add_argument("--sam3-script", default=DEFAULT_SAM3_SCRIPT, help="Path to prepare_frames_and_run_batch_sam3.py")
    parser.add_argument("--recordings-subdir", default="recordings", help="Recordings subdir passed through to the SAM3 runner")
    parser.add_argument("--stream-pattern", default="*_video.m3u8", help="Stream pattern passed through to the SAM3 runner")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU ids passed to the SAM3 runner")
    parser.add_argument("--min-score", type=float, default=0.3, help="SAM3 min score threshold")
    parser.add_argument("--finetuned", action=argparse.BooleanOptionalAction, default=True, help="Whether to use the finetuned SAM3 checkpoint")
    parser.add_argument("--checkpoint-path", default=None, help="SAM3 checkpoint path; required when --run-sam3 is set")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_sam3 and not str(args.checkpoint_path or "").strip():
        raise RuntimeError("--checkpoint-path is required when --run-sam3 is enabled")

    sequence_dir = Path(args.rides_base_dir).expanduser() / args.sequence_name
    frames_dir = Path(args.frames_base_dir).expanduser() / args.sequence_name / args.camera_view
    sam3_dir = Path(args.sam3_output_base_dir).expanduser() / args.sequence_name / args.camera_view

    if args.run_sam3:
        _run_sam3_pipeline(args)

    manifest = _build_manifest(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        sam3_dir=sam3_dir,
        fps=args.fps,
        prompt=args.prompt,
        mode=args.mode,
        frame_name=args.frame_name,
        frame_limit=args.frame_limit,
        use_imu_heading_fallback=args.use_imu_heading_fallback,
    )

    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.save_manifest:
        output_path = Path(args.save_manifest).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved manifest to: {output_path}")
    print(text)


if __name__ == "__main__":
    main()
