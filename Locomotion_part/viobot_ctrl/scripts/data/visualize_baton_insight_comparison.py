#!/usr/bin/env python3
"""Synchronize and compare Baton and Insight trajectories on Baton camera frames.

The Baton input is a ROS 2 MCAP file or a directory of split MCAP files containing
compressed images and nav_msgs/Odometry.  The Insight input contains
geometry_msgs/PoseStamped.  Insight's device-relative header clock is mapped onto
the Baton header clock using MCAP log time as a coarse anchor and angular-speed
correlation as the fine alignment signal.

Both pose streams are interpolated at the camera timestamps.  Their future paths
are converted independently into a common forward/left vehicle convention and
projected with the camera model used by visualize_go_zju_mcap_gt.py.
"""

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import visualize_go_zju_mcap_gt as gt  # noqa: E402


DEFAULT_BATON_INPUT = SCRIPT_PATH.parent / "bag_20260728_071949"
DEFAULT_INSIGHT_INPUT = SCRIPT_PATH.parent / "insight_0728_2"
DEFAULT_OUTPUT_ROOT = SCRIPT_PATH.parent / "baton_insight_comparison"
DEFAULT_INSIGHT_TOPIC = "/insight/vio_20hz"
DEFAULT_HORIZONS_S = (4.0, 3.0, 2.0, 1.0)
BATON_COLOR = (80, 225, 90)
INSIGHT_COLOR = (225, 90, 210)
CACHE_VERSION = 1


@dataclass
class PoseStream:
    name: str
    paths: list
    topic: str
    header_times_ns: np.ndarray
    log_times_ns: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray


@dataclass
class PrimaryData:
    pose_stream: PoseStream
    image_times_ns: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize Baton Odometry and Insight PoseStamped, then render both "
            "future trajectories on the Baton camera stream."
        )
    )
    parser.add_argument(
        "--baton",
        type=Path,
        default=DEFAULT_BATON_INPUT,
        help="Baton MCAP file or directory containing split MCAP files.",
    )
    parser.add_argument(
        "--insight",
        type=Path,
        default=DEFAULT_INSIGHT_INPUT,
        help="Insight MCAP file or rosbag2 directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for video, JSON, and CSV reports.",
    )
    parser.add_argument(
        "--image-topic",
        default=gt.DEFAULT_IMAGE_TOPIC,
        help="Baton compressed-image topic.",
    )
    parser.add_argument(
        "--baton-odom-topic",
        default=gt.DEFAULT_ODOM_TOPIC,
        help="Baton nav_msgs/Odometry topic.",
    )
    parser.add_argument(
        "--insight-pose-topic",
        default=DEFAULT_INSIGHT_TOPIC,
        help="Insight geometry_msgs/PoseStamped topic.",
    )
    parser.add_argument(
        "--insight-time-correction-s",
        type=float,
        help=(
            "Fixed correction added after mapping Insight header time through its "
            "median MCAP log/header offset. Omit to estimate it automatically."
        ),
    )
    parser.add_argument(
        "--sync-search-s",
        type=float,
        default=5.0,
        help="Half-width of the automatic fine time-alignment search.",
    )
    parser.add_argument(
        "--sync-window-s",
        type=float,
        default=100.0,
        help="Window length used to diagnose time-offset stability.",
    )
    parser.add_argument(
        "--baton-yaw-offset-deg",
        type=float,
        help="Fixed Baton sensor-yaw to vehicle-forward correction; auto if omitted.",
    )
    parser.add_argument(
        "--insight-yaw-offset-deg",
        type=float,
        help="Fixed Insight sensor-yaw to vehicle-forward correction; auto if omitted.",
    )
    parser.add_argument(
        "--baton-lever-arm",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X_FORWARD", "Y_LEFT"),
        help="Baton sensor position from the common vehicle reference point, in metres.",
    )
    parser.add_argument(
        "--insight-lever-arm",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X_FORWARD", "Y_LEFT"),
        help="Insight sensor position from the common vehicle reference point, in metres.",
    )
    parser.add_argument(
        "--insight-lateral-convention",
        choices=("left_positive", "right_positive"),
        default="left_positive",
        help="Flip Insight's local lateral path if its convention is right-positive.",
    )
    parser.add_argument(
        "--horizons-s",
        type=float,
        nargs="+",
        default=list(DEFAULT_HORIZONS_S),
        help="Future trajectory horizons in seconds.",
    )
    parser.add_argument(
        "--path-sample-hz",
        type=float,
        default=20.0,
        help="Sampling rate used to draw each future path.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=20.0,
        help="Output-video playback rate.",
    )
    parser.add_argument(
        "--start-s",
        type=float,
        default=0.0,
        help="Start offset in seconds from the synchronized common interval.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=0.0,
        help="Optional rendered duration; zero renders the full common interval.",
    )
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        default=list(gt.DEFAULT_INTRINSICS),
        metavar=("FX", "FY", "CX", "CY"),
        help="Camera intrinsics: fx, fy, cx, cy.",
    )
    parser.add_argument(
        "--distortion",
        type=float,
        nargs=3,
        default=list(gt.DEFAULT_DISTORTION),
        metavar=("K1", "K2", "K3"),
        help="Radial_3 distortion coefficients.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=0.6,
        help="Physical camera height in metres; repository calibration records 0.6.",
    )
    parser.add_argument(
        "--camera-x-offset",
        type=float,
        default=-0.1,
        help="Camera translation along vehicle-forward x before projection.",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=gt.DEFAULT_PITCH_DEG,
        help="Camera up-pitch in degrees.",
    )
    parser.add_argument(
        "--path-width",
        type=float,
        default=0.18,
        help="Half-width in metres of each rendered trajectory band.",
    )
    parser.add_argument(
        "--video-name",
        default="baton_insight_comparison.mp4",
        help="Output MP4 filename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract, synchronize, and report metrics without rendering video.",
    )
    return parser.parse_args()


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def resolve_mcaps(path):
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".mcap":
            raise ValueError(f"Not an MCAP file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    paths = sorted(
        (candidate for candidate in path.glob("*.mcap") if candidate.is_file()),
        key=natural_key,
    )
    if not paths:
        raise ValueError(f"No .mcap files found in: {path}")
    return paths


def decoded_pose(message):
    pose_container = message.pose
    return (
        pose_container.pose
        if hasattr(pose_container, "pose")
        else pose_container
    )


def append_pose_record(records, decoded_message, log_time_ns):
    pose = decoded_pose(decoded_message)
    stamp_ns = gt.message_stamp_ns(decoded_message, log_time_ns)
    records.append(
        (
            stamp_ns,
            int(log_time_ns),
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
    )


def finalize_pose_stream(name, paths, topic, records):
    if not records:
        raise RuntimeError(f"No poses found for {name} on topic {topic}")
    values = np.asarray(records, dtype=np.float64)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    keep = np.r_[values[1:, 0] != values[:-1, 0], True]
    values = values[keep]
    if len(values) < 2:
        raise RuntimeError(f"{name} contains fewer than two distinct pose timestamps")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite pose values")
    quaternions = gt.make_quaternions_continuous(values[:, 5:9])
    return PoseStream(
        name=name,
        paths=list(paths),
        topic=topic,
        header_times_ns=values[:, 0].astype(np.int64),
        log_times_ns=values[:, 1].astype(np.int64),
        positions=values[:, 2:5],
        quaternions=quaternions,
    )


def load_primary(paths, image_topic, odom_topic):
    make_reader, DecoderFactory = gt.import_mcap_runtime()
    image_times = []
    pose_records = []
    previous_image_time = None
    for path in paths:
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, channel, message, decoded_message in reader.iter_decoded_messages(
                topics=[image_topic, odom_topic]
            ):
                if channel.topic == image_topic:
                    stamp_ns = gt.message_stamp_ns(decoded_message, message.log_time)
                    if previous_image_time is not None and stamp_ns < previous_image_time:
                        raise RuntimeError(
                            "Baton image header timestamps are not monotonic across "
                            f"the ordered MCAP files; offending file: {path}"
                        )
                    image_times.append(stamp_ns)
                    previous_image_time = stamp_ns
                elif channel.topic == odom_topic:
                    append_pose_record(
                        pose_records, decoded_message, message.log_time
                    )
    if not image_times:
        raise RuntimeError(f"No images found on topic {image_topic}")
    return PrimaryData(
        pose_stream=finalize_pose_stream(
            "baton", paths, odom_topic, pose_records
        ),
        image_times_ns=np.asarray(image_times, dtype=np.int64),
    )


def load_pose_stream(name, paths, topic):
    make_reader, DecoderFactory = gt.import_mcap_runtime()
    records = []
    for path in paths:
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, _, message, decoded_message in reader.iter_decoded_messages(
                topics=[topic]
            ):
                append_pose_record(records, decoded_message, message.log_time)
    return finalize_pose_stream(name, paths, topic, records)


def quaternion_yaws(quaternions):
    return np.unwrap(
        np.asarray(
            [gt.quaternion_xyzw_to_yaw(quaternion) for quaternion in quaternions],
            dtype=np.float64,
        )
    )


def wrap_angles(angles):
    return (np.asarray(angles) + math.pi) % (2.0 * math.pi) - math.pi


def angular_speed_curve(times_s, quaternions, sample_hz=20.0, smooth_s=0.25):
    time_deltas = np.diff(times_s)
    if np.any(time_deltas <= 0.0):
        raise ValueError("Pose timestamps must be strictly increasing")
    dots = np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1))
    angles = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
    angular_speeds = angles / time_deltas
    middle_times = 0.5 * (times_s[:-1] + times_s[1:])
    step = 1.0 / sample_hz
    grid = np.arange(
        math.ceil(middle_times[0] * sample_hz) / sample_hz,
        math.floor(middle_times[-1] * sample_hz) / sample_hz,
        step,
    )
    if len(grid) < 20:
        raise RuntimeError("Pose stream is too short for automatic synchronization")
    values = np.interp(grid, middle_times, angular_speeds)
    kernel_size = max(1, int(round(sample_hz * smooth_s)))
    if kernel_size > 1:
        kernel = np.full(kernel_size, 1.0 / kernel_size)
        values = np.convolve(values, kernel, mode="same")
    return grid, values


def normalized_correlation(
    shift_s,
    baton_grid,
    baton_signal,
    insight_grid,
    insight_signal,
    interval=None,
):
    low = max(baton_grid[0], insight_grid[0] - shift_s)
    high = min(baton_grid[-1], insight_grid[-1] - shift_s)
    if interval is not None:
        low = max(low, interval[0])
        high = min(high, interval[1])
    if high - low < 10.0:
        return -math.inf, 0
    sample_times = np.arange(low, high, 0.05)
    baton_values = np.interp(sample_times, baton_grid, baton_signal)
    insight_values = np.interp(
        sample_times + shift_s, insight_grid, insight_signal
    )
    baton_limits = np.percentile(baton_values, (2.0, 98.0))
    insight_limits = np.percentile(insight_values, (2.0, 98.0))
    baton_values = np.clip(baton_values, *baton_limits)
    insight_values = np.clip(insight_values, *insight_limits)
    baton_std = float(np.std(baton_values))
    insight_std = float(np.std(insight_values))
    if baton_std < 1e-8 or insight_std < 1e-8:
        return -math.inf, len(sample_times)
    baton_values = (baton_values - np.mean(baton_values)) / baton_std
    insight_values = (insight_values - np.mean(insight_values)) / insight_std
    return float(np.mean(baton_values * insight_values)), len(sample_times)


def search_best_shift(
    baton_grid,
    baton_signal,
    insight_grid,
    insight_signal,
    search_s,
    interval=None,
    center_s=0.0,
):
    if search_s <= 0.0:
        raise ValueError("--sync-search-s must be positive")
    coarse_step = 0.02
    coarse_candidates = np.arange(
        center_s - search_s,
        center_s + search_s + coarse_step * 0.5,
        coarse_step,
    )
    coarse_results = [
        (
            normalized_correlation(
                shift,
                baton_grid,
                baton_signal,
                insight_grid,
                insight_signal,
                interval,
            )[0],
            float(shift),
        )
        for shift in coarse_candidates
    ]
    best_coarse = max(coarse_results)
    if not math.isfinite(best_coarse[0]):
        raise RuntimeError("No informative overlap found for automatic synchronization")

    fine_step = 0.001
    fine_candidates = np.arange(
        best_coarse[1] - coarse_step,
        best_coarse[1] + coarse_step + fine_step * 0.5,
        fine_step,
    )
    fine_results = []
    for shift in fine_candidates:
        correlation, count = normalized_correlation(
            shift,
            baton_grid,
            baton_signal,
            insight_grid,
            insight_signal,
            interval,
        )
        fine_results.append((correlation, float(shift), count))
    return max(fine_results)


def estimate_time_mapping(
    baton,
    insight,
    fixed_correction_s,
    search_s,
    window_s,
):
    origin_ns = int(baton.header_times_ns[0])
    baton_times_s = (baton.header_times_ns - origin_ns).astype(np.float64) / 1e9

    insight_log_header_offsets = (
        insight.log_times_ns.astype(np.float64)
        - insight.header_times_ns.astype(np.float64)
    )
    coarse_offset_ns = float(np.median(insight_log_header_offsets))
    insight_coarse_times_s = (
        insight.header_times_ns.astype(np.float64)
        - float(insight.header_times_ns[0])
    ) / 1e9
    insight_coarse_times_s += (
        float(insight.header_times_ns[0]) + coarse_offset_ns - origin_ns
    ) / 1e9

    baton_grid, baton_signal = angular_speed_curve(
        baton_times_s, baton.quaternions
    )
    insight_grid, insight_signal = angular_speed_curve(
        insight_coarse_times_s, insight.quaternions
    )

    if fixed_correction_s is None:
        correlation, shift_s, sample_count = search_best_shift(
            baton_grid,
            baton_signal,
            insight_grid,
            insight_signal,
            search_s=search_s,
        )
        correction_s = -shift_s
        mode = "auto_angular_speed"
    else:
        correction_s = float(fixed_correction_s)
        shift_s = -correction_s
        correlation, sample_count = normalized_correlation(
            shift_s,
            baton_grid,
            baton_signal,
            insight_grid,
            insight_signal,
        )
        mode = "fixed"

    mapped_insight_times_s = insight_coarse_times_s + correction_s
    overlap_start = max(baton_grid[0], mapped_insight_times_s[0])
    overlap_end = min(baton_grid[-1], mapped_insight_times_s[-1])
    window_results = []
    if window_s >= 20.0 and overlap_end - overlap_start >= window_s:
        window_start = overlap_start
        while window_start + window_s <= overlap_end + 1e-9:
            window_center = window_start + 0.5 * window_s
            result = search_best_shift(
                baton_grid,
                baton_signal,
                insight_grid,
                insight_signal,
                search_s=min(0.15, search_s),
                center_s=shift_s,
                interval=(window_start, window_start + window_s),
            )
            window_results.append(
                {
                    "center_from_baton_start_s": float(window_center),
                    "correlation": float(result[0]),
                    "shift_s": float(result[1]),
                    "correction_s": float(-result[1]),
                }
            )
            window_start += window_s

    header_to_baton_offset_s = (
        coarse_offset_ns / 1e9 + correction_s
    )
    return {
        "origin_ns": origin_ns,
        "baton_times_s": baton_times_s,
        "insight_times_s": mapped_insight_times_s,
        "mode": mode,
        "coarse_header_to_log_offset_s": coarse_offset_ns / 1e9,
        "fine_shift_s": float(shift_s),
        "correction_s": float(correction_s),
        "insight_header_to_baton_header_offset_s": float(
            header_to_baton_offset_s
        ),
        "correlation": float(correlation),
        "correlation_sample_count": int(sample_count),
        "window_results": window_results,
    }


def estimate_yaw_offset_deg(times_s, positions, quaternions):
    yaws = quaternion_yaws(quaternions)
    target_indices = np.searchsorted(times_s, times_s + 1.0)
    valid = target_indices < len(times_s)
    source_indices = np.flatnonzero(valid)
    displacements = (
        positions[target_indices[source_indices], :2]
        - positions[source_indices, :2]
    )
    distances = np.linalg.norm(displacements, axis=1)
    moving = distances > 1.0
    source_indices = source_indices[moving]
    displacements = displacements[moving]
    distances = distances[moving]
    if len(source_indices) < 20:
        raise RuntimeError("Insufficient motion to estimate a yaw correction")
    motion_headings = np.arctan2(displacements[:, 1], displacements[:, 0])
    offsets = wrap_angles(motion_headings - yaws[source_indices])

    keep = np.ones(len(offsets), dtype=bool)
    mean_offset = 0.0
    for _ in range(3):
        weights = distances[keep]
        phasors = np.exp(1j * offsets[keep])
        mean_offset = float(np.angle(np.sum(weights * phasors)))
        residuals = np.abs(wrap_angles(offsets - mean_offset))
        keep = residuals <= math.radians(45.0)
        if np.count_nonzero(keep) < 20:
            break
    residual_degrees = np.degrees(
        np.abs(wrap_angles(offsets - mean_offset))
    )
    return math.degrees(mean_offset), {
        "sample_count": int(len(offsets)),
        "residual_median_deg": float(np.median(residual_degrees)),
        "residual_p90_deg": float(np.percentile(residual_degrees, 90.0)),
    }


def resolve_yaw_offset(fixed_value, times_s, stream):
    if fixed_value is not None:
        return float(fixed_value), {"mode": "fixed"}
    value, diagnostics = estimate_yaw_offset_deg(
        times_s, stream.positions, stream.quaternions
    )
    diagnostics["mode"] = "auto_motion_heading"
    return float(value), diagnostics


def interpolate_pose(times_s, stream, sample_times_s):
    positions = gt.interpolate_positions(
        sample_times_s, times_s, stream.positions
    )
    quaternions = gt.interpolate_quaternions_slerp(
        sample_times_s, times_s, stream.quaternions
    )
    return positions, quaternions


def positions_at_vehicle_reference(
    sensor_positions,
    quaternions,
    yaw_offset_deg,
    lever_arm,
):
    base_yaws = np.asarray(
        [
            gt.quaternion_xyzw_to_yaw(quaternion)
            for quaternion in quaternions
        ],
        dtype=np.float64,
    ) + math.radians(yaw_offset_deg)
    positions_xy = sensor_positions[:, :2].copy()
    lever_x, lever_y = lever_arm
    if abs(lever_x) > 1e-12 or abs(lever_y) > 1e-12:
        cos_yaw = np.cos(base_yaws)
        sin_yaw = np.sin(base_yaws)
        world_lever = np.column_stack(
            (
                cos_yaw * lever_x - sin_yaw * lever_y,
                sin_yaw * lever_x + cos_yaw * lever_y,
            )
        )
        positions_xy -= world_lever
    return positions_xy, base_yaws


def sample_local_path(
    stream,
    stream_times_s,
    sample_times_s,
    yaw_offset_deg,
    lever_arm,
    flip_lateral=False,
):
    positions, quaternions = interpolate_pose(
        stream_times_s, stream, sample_times_s
    )
    base_positions, base_yaws = positions_at_vehicle_reference(
        positions, quaternions, yaw_offset_deg, lever_arm
    )
    local_path = gt.rotate_points_xy(
        base_positions - base_positions[0], -base_yaws[0]
    )
    if flip_lateral:
        local_path[:, 1] *= -1.0
    return local_path, base_yaws


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def prepare_comparison(
    primary,
    insight,
    time_mapping,
    horizons_s,
    start_s,
    max_duration_s,
    baton_yaw_offset_deg,
    insight_yaw_offset_deg,
    baton_lever_arm,
    insight_lever_arm,
    flip_insight_lateral,
):
    origin_ns = time_mapping["origin_ns"]
    image_times_s = (
        primary.image_times_ns.astype(np.float64) - origin_ns
    ) / 1e9
    baton_times_s = time_mapping["baton_times_s"]
    insight_times_s = time_mapping["insight_times_s"]
    max_horizon_s = max(horizons_s)

    common_start_s = max(
        image_times_s[0], baton_times_s[0], insight_times_s[0]
    )
    common_end_s = min(
        image_times_s[-1], baton_times_s[-1], insight_times_s[-1]
    )
    render_start_s = common_start_s + start_s
    latest_frame_s = common_end_s - max_horizon_s
    if max_duration_s > 0.0:
        render_end_s = min(latest_frame_s, render_start_s + max_duration_s)
    else:
        render_end_s = latest_frame_s
    if render_start_s >= render_end_s:
        raise RuntimeError(
            "Requested render interval is empty after synchronization and "
            "future-horizon coverage checks"
        )

    selected = (image_times_s >= render_start_s) & (
        image_times_s < render_end_s
    )
    source_indices = np.flatnonzero(selected).astype(np.int64)
    frame_times_s = image_times_s[selected]
    if not len(frame_times_s):
        raise RuntimeError("No camera frames fall inside the common interval")

    metric_rows = []
    metric_values = {
        horizon: {"longitudinal": [], "lateral": [], "distance": [], "heading": []}
        for horizon in horizons_s
    }
    for horizon_s in sorted(horizons_s):
        future_times = frame_times_s + horizon_s
        baton_targets = np.column_stack((frame_times_s, future_times)).reshape(-1)
        insight_targets = baton_targets
        baton_positions, baton_quaternions = interpolate_pose(
            baton_times_s, primary.pose_stream, baton_targets
        )
        insight_positions, insight_quaternions = interpolate_pose(
            insight_times_s, insight, insight_targets
        )
        baton_reference, baton_yaws = positions_at_vehicle_reference(
            baton_positions,
            baton_quaternions,
            baton_yaw_offset_deg,
            baton_lever_arm,
        )
        insight_reference, insight_yaws = positions_at_vehicle_reference(
            insight_positions,
            insight_quaternions,
            insight_yaw_offset_deg,
            insight_lever_arm,
        )
        baton_reference = baton_reference.reshape(-1, 2, 2)
        insight_reference = insight_reference.reshape(-1, 2, 2)
        baton_yaws = baton_yaws.reshape(-1, 2)
        insight_yaws = insight_yaws.reshape(-1, 2)

        baton_endpoints = np.empty((len(frame_times_s), 2), dtype=np.float64)
        insight_endpoints = np.empty_like(baton_endpoints)
        for index in range(len(frame_times_s)):
            baton_endpoints[index] = gt.rotate_points_xy(
                (
                    baton_reference[index, 1]
                    - baton_reference[index, 0]
                ).reshape(1, 2),
                -baton_yaws[index, 0],
            )[0]
            insight_endpoints[index] = gt.rotate_points_xy(
                (
                    insight_reference[index, 1]
                    - insight_reference[index, 0]
                ).reshape(1, 2),
                -insight_yaws[index, 0],
            )[0]
        if flip_insight_lateral:
            insight_endpoints[:, 1] *= -1.0

        differences = insight_endpoints - baton_endpoints
        distances = np.linalg.norm(differences, axis=1)
        baton_heading_changes = wrap_angles(
            baton_yaws[:, 1] - baton_yaws[:, 0]
        )
        insight_heading_changes = wrap_angles(
            insight_yaws[:, 1] - insight_yaws[:, 0]
        )
        if flip_insight_lateral:
            insight_heading_changes *= -1.0
        heading_differences = np.degrees(
            wrap_angles(insight_heading_changes - baton_heading_changes)
        )

        metric_values[horizon_s]["longitudinal"].extend(differences[:, 0])
        metric_values[horizon_s]["lateral"].extend(differences[:, 1])
        metric_values[horizon_s]["distance"].extend(distances)
        metric_values[horizon_s]["heading"].extend(heading_differences)
        for index, frame_time in enumerate(frame_times_s):
            metric_rows.append(
                {
                    "frame_index": index,
                    "baton_header_time_ns": int(
                        round(frame_time * 1e9 + origin_ns)
                    ),
                    "horizon_s": float(horizon_s),
                    "insight_minus_baton_longitudinal_m": float(
                        differences[index, 0]
                    ),
                    "insight_minus_baton_lateral_m": float(
                        differences[index, 1]
                    ),
                    "endpoint_distance_m": float(distances[index]),
                    "heading_change_difference_deg": float(
                        heading_differences[index]
                    ),
                }
            )

    metric_summary = {}
    for horizon, values in metric_values.items():
        metric_summary[str(horizon)] = {
            "longitudinal_signed_m": summarize_values(values["longitudinal"]),
            "lateral_signed_m": summarize_values(values["lateral"]),
            "endpoint_distance_m": summarize_values(values["distance"]),
            "heading_change_signed_deg": summarize_values(values["heading"]),
        }
    return {
        "source_indices": source_indices,
        "frame_times_s": frame_times_s,
        "common_start_s": float(common_start_s),
        "common_end_s": float(common_end_s),
        "render_start_s": float(render_start_s),
        "render_end_s": float(render_end_s),
        "metric_rows": metric_rows,
        "metric_summary": metric_summary,
    }


def draw_comparison_legend(image, horizons_s):
    x0, y0 = 24, 24
    width, height = 330, 112
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, image, 0.30, 0.0, image)
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), (230, 230, 230), 1)
    entries = (("Baton odometry", BATON_COLOR), ("Insight VIO", INSIGHT_COLOR))
    for index, (label, color) in enumerate(entries):
        y = y0 + 29 + 27 * index
        cv2.line(image, (x0 + 14, y), (x0 + 55, y), color, 7, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (x0 + 67, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    horizon_text = "markers: " + "/".join(
        f"{horizon:g}s" for horizon in sorted(horizons_s)
    )
    cv2.putText(
        image,
        horizon_text,
        (x0 + 14, y0 + 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )


def draw_path_markers(image, projected_center, sample_times, frame_time, horizons, color):
    height, width = image.shape[:2]
    for horizon in sorted(horizons):
        index = int(np.argmin(np.abs(sample_times - (frame_time + horizon))))
        point = projected_center[index]
        if not np.all(np.isfinite(point)):
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < width and 0 <= y < height):
            continue
        cv2.circle(image, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)


def draw_frame_difference(image, baton_path, insight_path, horizon_s):
    delta = insight_path[-1] - baton_path[-1]
    distance = float(np.linalg.norm(delta))
    text = (
        f"{horizon_s:g}s agreement: d={distance:.2f}m  "
        f"long={delta[0]:+.2f}m  lat={delta[1]:+.2f}m"
    )
    text_size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )
    x = 24
    y = image.shape[0] - 28
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x - 9, y - text_size[1] - 11),
        (x + text_size[0] + 9, y + 8),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.70, image, 0.30, 0.0, image)
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def iter_selected_images(paths, topic, source_indices):
    make_reader, DecoderFactory = gt.import_mcap_runtime()
    source_indices = np.asarray(source_indices, dtype=np.int64)
    wanted = 0
    source_index = 0
    for path in paths:
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, _, _, decoded_message in reader.iter_decoded_messages(
                topics=[topic]
            ):
                if (
                    wanted < len(source_indices)
                    and source_index == source_indices[wanted]
                ):
                    yield decoded_message.data
                    wanted += 1
                    if wanted == len(source_indices):
                        return
                source_index += 1
    raise RuntimeError(
        f"Image streams ended after {wanted}/{len(source_indices)} selected frames"
    )


def render_video(
    output_path,
    primary,
    insight,
    comparison,
    time_mapping,
    image_topic,
    horizons_s,
    sample_hz,
    output_fps,
    baton_yaw_offset_deg,
    insight_yaw_offset_deg,
    baton_lever_arm,
    insight_lever_arm,
    flip_insight_lateral,
    intrinsics,
    distortion,
    camera_height,
    camera_x_offset,
    pitch_deg,
    path_width,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    writer = None
    codec = None
    frame_size = None
    frame_times_s = comparison["frame_times_s"]
    max_horizon_s = max(horizons_s)
    sample_offsets = np.arange(
        0.0,
        max_horizon_s + 0.5 / sample_hz,
        1.0 / sample_hz,
    )
    image_iterator = iter_selected_images(
        primary.pose_stream.paths,
        image_topic,
        comparison["source_indices"],
    )

    try:
        for frame_index, encoded_image in enumerate(image_iterator):
            frame = gt.decode_compressed_image(encoded_image)
            if writer is None:
                height, width = frame.shape[:2]
                frame_size = (width, height)
                writer, codec = gt.open_video_writer(
                    temporary_path, width, height, output_fps
                )
            elif (frame.shape[1], frame.shape[0]) != frame_size:
                raise RuntimeError("Image resolution changed inside the Baton bag")

            frame_time = frame_times_s[frame_index]
            sample_times = frame_time + sample_offsets
            baton_path, _ = sample_local_path(
                primary.pose_stream,
                time_mapping["baton_times_s"],
                sample_times,
                baton_yaw_offset_deg,
                baton_lever_arm,
            )
            insight_path, _ = sample_local_path(
                insight,
                time_mapping["insight_times_s"],
                sample_times,
                insight_yaw_offset_deg,
                insight_lever_arm,
                flip_lateral=flip_insight_lateral,
            )

            projected = []
            for path, color in (
                (insight_path, INSIGHT_COLOR),
                (baton_path, BATON_COLOR),
            ):
                center, left, right = gt.project_path_curves(
                    path,
                    intrinsic_params=intrinsics,
                    distortion_coeffs=distortion,
                    camera_height=camera_height,
                    camera_x_offset=camera_x_offset,
                    pitch_deg=pitch_deg,
                    path_width=path_width,
                )
                gt.overlay_projected_path(
                    frame,
                    waypoints=path,
                    points_center=center,
                    points_left=left,
                    points_right=right,
                    color=color,
                    base_alpha=0.08,
                )
                projected.append((center, color))
            for center, color in projected:
                draw_path_markers(
                    frame,
                    center,
                    sample_times,
                    frame_time,
                    horizons_s,
                    color,
                )
            draw_comparison_legend(frame, horizons_s)
            draw_frame_difference(
                frame, baton_path, insight_path, max_horizon_s
            )
            writer.write(frame)

            rendered = frame_index + 1
            if rendered == 1 or rendered % 100 == 0 or rendered == len(frame_times_s):
                print(f"[render] {rendered}/{len(frame_times_s)}")
    except Exception:
        if writer is not None:
            writer.release()
        temporary_path.unlink(missing_ok=True)
        raise

    if writer is None:
        raise RuntimeError("No frames were rendered")
    writer.release()
    temporary_path.replace(output_path)
    print(f"[done] {output_path} (codec={codec})")
    return codec


def write_metrics_csv(path, rows):
    fieldnames = (
        "frame_index",
        "baton_header_time_ns",
        "horizon_s",
        "insight_minus_baton_longitudinal_m",
        "insight_minus_baton_lateral_m",
        "endpoint_distance_m",
        "heading_change_difference_deg",
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def input_signatures(paths):
    return [gt.input_signature(path) for path in paths]


def validate_args(args):
    if args.sync_search_s <= 0.0:
        raise ValueError("--sync-search-s must be positive")
    if args.sync_window_s < 0.0:
        raise ValueError("--sync-window-s cannot be negative")
    if args.start_s < 0.0:
        raise ValueError("--start-s cannot be negative")
    if args.max_duration_s < 0.0:
        raise ValueError("--max-duration-s cannot be negative")
    if args.output_fps <= 0.0 or args.path_sample_hz <= 0.0:
        raise ValueError("FPS values must be positive")
    if args.camera_height <= 0.0 or args.path_width <= 0.0:
        raise ValueError("Camera height and path width must be positive")
    if not args.horizons_s or any(value <= 0.0 for value in args.horizons_s):
        raise ValueError("--horizons-s must contain positive values")
    if len(set(args.horizons_s)) != len(args.horizons_s):
        raise ValueError("--horizons-s cannot contain duplicates")
    if Path(args.video_name).name != args.video_name:
        raise ValueError("--video-name must be a filename, not a path")


def main():
    args = parse_args()
    validate_args(args)
    baton_paths = resolve_mcaps(args.baton)
    insight_paths = resolve_mcaps(args.insight)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[extract] Baton: {len(baton_paths)} MCAP file(s); "
        f"Insight: {len(insight_paths)} MCAP file(s)"
    )
    primary = load_primary(
        baton_paths, args.image_topic, args.baton_odom_topic
    )
    insight = load_pose_stream(
        "insight", insight_paths, args.insight_pose_topic
    )
    print(
        f"[extract] images={len(primary.image_times_ns)}, "
        f"baton_poses={len(primary.pose_stream.positions)}, "
        f"insight_poses={len(insight.positions)}"
    )

    time_mapping = estimate_time_mapping(
        primary.pose_stream,
        insight,
        fixed_correction_s=args.insight_time_correction_s,
        search_s=args.sync_search_s,
        window_s=args.sync_window_s,
    )
    print(
        "[sync] mode={mode}, correction={correction_s:+.3f}s, "
        "correlation={correlation:.4f}".format(**time_mapping)
    )

    baton_yaw_offset_deg, baton_yaw_diagnostics = resolve_yaw_offset(
        args.baton_yaw_offset_deg,
        time_mapping["baton_times_s"],
        primary.pose_stream,
    )
    insight_yaw_offset_deg, insight_yaw_diagnostics = resolve_yaw_offset(
        args.insight_yaw_offset_deg,
        time_mapping["insight_times_s"],
        insight,
    )
    print(
        f"[orientation] Baton yaw correction={baton_yaw_offset_deg:+.3f}deg; "
        f"Insight={insight_yaw_offset_deg:+.3f}deg"
    )

    horizons_s = tuple(sorted(set(args.horizons_s), reverse=True))
    comparison = prepare_comparison(
        primary=primary,
        insight=insight,
        time_mapping=time_mapping,
        horizons_s=horizons_s,
        start_s=args.start_s,
        max_duration_s=args.max_duration_s,
        baton_yaw_offset_deg=baton_yaw_offset_deg,
        insight_yaw_offset_deg=insight_yaw_offset_deg,
        baton_lever_arm=tuple(args.baton_lever_arm),
        insight_lever_arm=tuple(args.insight_lever_arm),
        flip_insight_lateral=args.insight_lateral_convention
        == "right_positive",
    )
    print(
        f"[interval] frames={len(comparison['frame_times_s'])}, "
        f"start={comparison['render_start_s']:.3f}s, "
        f"end={comparison['render_end_s']:.3f}s from Baton origin"
    )

    metrics_path = output_root / "baton_insight_metrics.csv"
    write_metrics_csv(metrics_path, comparison["metric_rows"])
    video_path = output_root / args.video_name
    codec = None
    if not args.dry_run:
        codec = render_video(
            output_path=video_path,
            primary=primary,
            insight=insight,
            comparison=comparison,
            time_mapping=time_mapping,
            image_topic=args.image_topic,
            horizons_s=horizons_s,
            sample_hz=args.path_sample_hz,
            output_fps=args.output_fps,
            baton_yaw_offset_deg=baton_yaw_offset_deg,
            insight_yaw_offset_deg=insight_yaw_offset_deg,
            baton_lever_arm=tuple(args.baton_lever_arm),
            insight_lever_arm=tuple(args.insight_lever_arm),
            flip_insight_lateral=args.insight_lateral_convention
            == "right_positive",
            intrinsics=tuple(args.intrinsics),
            distortion=tuple(args.distortion),
            camera_height=args.camera_height,
            camera_x_offset=args.camera_x_offset,
            pitch_deg=args.pitch_deg,
            path_width=args.path_width,
        )

    report = {
        "cache_version": CACHE_VERSION,
        "inputs": {
            "baton": input_signatures(baton_paths),
            "insight": input_signatures(insight_paths),
            "image_topic": args.image_topic,
            "baton_odom_topic": args.baton_odom_topic,
            "insight_pose_topic": args.insight_pose_topic,
        },
        "counts": {
            "images": int(len(primary.image_times_ns)),
            "baton_poses": int(len(primary.pose_stream.positions)),
            "insight_poses": int(len(insight.positions)),
            "rendered_or_selected_frames": int(len(comparison["frame_times_s"])),
        },
        "time_alignment": {
            key: value
            for key, value in time_mapping.items()
            if key not in {"baton_times_s", "insight_times_s"}
        },
        "orientation_alignment": {
            "baton_yaw_offset_deg": baton_yaw_offset_deg,
            "baton_diagnostics": baton_yaw_diagnostics,
            "insight_yaw_offset_deg": insight_yaw_offset_deg,
            "insight_diagnostics": insight_yaw_diagnostics,
            "insight_lateral_convention": args.insight_lateral_convention,
        },
        "lever_arms_vehicle_xy_m": {
            "baton": list(args.baton_lever_arm),
            "insight": list(args.insight_lever_arm),
            "note": (
                "Defaults are zero. Supply measured sensor positions relative to "
                "one common vehicle reference point for lever-arm compensation."
            ),
        },
        "interval": {
            "common_start_from_baton_origin_s": comparison["common_start_s"],
            "common_end_from_baton_origin_s": comparison["common_end_s"],
            "render_start_from_baton_origin_s": comparison["render_start_s"],
            "render_end_from_baton_origin_s": comparison["render_end_s"],
            "requested_start_s": args.start_s,
            "requested_max_duration_s": args.max_duration_s,
        },
        "camera": {
            "intrinsics": list(args.intrinsics),
            "distortion": list(args.distortion),
            "camera_height_m": args.camera_height,
            "camera_x_offset_m": args.camera_x_offset,
            "pitch_deg": args.pitch_deg,
            "path_half_width_m": args.path_width,
        },
        "comparison": {
            "horizons_s": list(horizons_s),
            "metric_meaning": (
                "Agreement between two devices after independent local-frame "
                "normalization; neither device is treated as ground truth."
            ),
            "summary": comparison["metric_summary"],
        },
        "outputs": {
            "metrics_csv": str(metrics_path),
            "video": None if args.dry_run else str(video_path),
            "video_codec": codec,
        },
    }
    report_path = output_root / "baton_insight_comparison.json"
    gt.write_json_atomic(report_path, report)
    print(f"[done] {metrics_path}")
    print(f"[done] {report_path}")
    if args.dry_run:
        print("[done] dry run requested; video was not rendered")


if __name__ == "__main__":
    main()
