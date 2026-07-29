#!/usr/bin/env python3
"""Process sidewalk dataset (npy + mp4) to generate BEV maps."""

from __future__ import annotations

import sys
import os
import math
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

import numpy as np
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR / "bev"))
sys.path.insert(0, str(TOOLS_DIR / "math_tools"))

from frodobot_bev_enhanced import (
    render_bev_map,
    MAP_PROVIDERS,
    ProcessOptions,
)
from pose_ekf import PoseEKF


def _quaternion_to_euler_deg(*, x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert ROS quaternion xyzw into roll/pitch/yaw degrees."""
    sinr_cosp = 2.0 * (float(w) * float(x) + float(y) * float(z))
    cosr_cosp = 1.0 - 2.0 * (float(x) * float(x) + float(y) * float(y))
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (float(w) * float(y) - float(z) * float(x))))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (float(w) * float(z) + float(x) * float(y))
    cosy_cosp = 1.0 - 2.0 * (float(y) * float(y) + float(z) * float(z))
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _calculate_heading_from_gps(
    lat1: float, lng1: float, lat2: float, lng2: float
) -> float:
    """Calculate compass heading (0=North, clockwise) from two GPS points."""
    # Check if points are the same (no movement)
    if abs(lat1 - lat2) < 1e-7 and abs(lng1 - lng2) < 1e-7:
        return 0.0

    d_lon = math.radians(lng2 - lng1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)

    y = math.sin(d_lon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(y, x))
    compass_heading = (bearing + 360.0) % 360.0
    return compass_heading


def _distance_m_between_gps(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    lat_mid = math.radians((float(lat1) + float(lat2)) * 0.5)
    dlat = math.radians(float(lat2) - float(lat1))
    dlng = math.radians(float(lng2) - float(lng1))
    earth_radius_m = 6371000.0
    dx = earth_radius_m * dlng * math.cos(lat_mid)
    dy = earth_radius_m * dlat
    return math.hypot(dx, dy)


def _circular_mean_deg(values: list[float]) -> float:
    valid = [float(v) for v in values if not math.isnan(float(v))]
    if not valid:
        return float("nan")
    x_sum = 0.0
    y_sum = 0.0
    for value in valid:
        rad = math.radians(value)
        x_sum += math.cos(rad)
        y_sum += math.sin(rad)
    if abs(x_sum) < 1e-12 and abs(y_sum) < 1e-12:
        return float("nan")
    return (math.degrees(math.atan2(y_sum, x_sum)) + 360.0) % 360.0


def _calculate_robust_gps_headings(
    positions: list[tuple[float, float]],
    *,
    min_distance_m: float = 0.75,
    max_lookahead_frames: int = 40,
) -> list[float]:
    """Estimate heading from GPS using points far enough apart to beat jitter/stops."""
    headings: list[float] = []
    n = len(positions)
    for i, (lat, lng) in enumerate(positions):
        heading = float("nan")

        upper = min(n, i + max_lookahead_frames + 1)
        for j in range(i + 1, upper):
            next_lat, next_lng = positions[j]
            if _distance_m_between_gps(lat, lng, next_lat, next_lng) >= min_distance_m:
                heading = _calculate_heading_from_gps(lat, lng, next_lat, next_lng)
                break

        if math.isnan(heading):
            lower = max(-1, i - max_lookahead_frames - 1)
            for j in range(i - 1, lower, -1):
                prev_lat, prev_lng = positions[j]
                if _distance_m_between_gps(prev_lat, prev_lng, lat, lng) >= min_distance_m:
                    heading = _calculate_heading_from_gps(prev_lat, prev_lng, lat, lng)
                    break

        headings.append(heading)

    valid_headings = [h for h in headings if not math.isnan(h)]
    fallback = valid_headings[0] if valid_headings else 0.0

    # Fill gaps from nearest usable heading so stationary sections preserve travel direction.
    filled = headings.copy()
    last = float("nan")
    for i, h in enumerate(filled):
        if math.isnan(h):
            if not math.isnan(last):
                filled[i] = last
        else:
            last = h
    next_h = float("nan")
    for i in range(n - 1, -1, -1):
        if math.isnan(filled[i]):
            filled[i] = next_h if not math.isnan(next_h) else fallback
        else:
            next_h = filled[i]

    return [normalize_heading_to_compass(h) for h in filled]


def _is_yaw_observation_reliable(yaw_list: list[float], ori_arr: np.ndarray, selected_indices: list[int]) -> tuple[bool, str]:
    if not yaw_list:
        return False, "empty yaw list"

    norms = []
    for data_idx in selected_indices[: min(len(selected_indices), 200)]:
        q = ori_arr[data_idx]
        norms.append(float(np.linalg.norm(q)))
    median_norm = float(np.median(norms)) if norms else 0.0
    if median_norm < 0.5:
        return False, f"invalid quaternion norm median={median_norm:.3f}"

    mean_heading = _circular_mean_deg(yaw_list)
    deviations = [abs(((h - mean_heading + 180.0) % 360.0) - 180.0) for h in yaw_list]
    p95_dev = float(np.percentile(deviations, 95)) if deviations else 0.0
    if p95_dev < 0.5:
        return False, f"near-constant yaw p95_dev={p95_dev:.3f}"

    return True, f"yaw p95_dev={p95_dev:.3f}"


def normalize_heading_to_compass(heading_deg: float) -> float:
    """Normalize heading to compass convention: 0=North, clockwise positive, range [0,360)."""
    h = float(heading_deg)
    h = h % 360.0
    if h < 0:
        h += 360.0
    return h


def _extract_frames_from_mp4(mp4_path: Path, output_dir: Path, sample_fps: float = 1.0) -> list[Path]:
    """Extract frames from mp4 at given sample rate."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to extract frames
    # Use fps filter to sample at given rate
    output_pattern = str(images_dir / "%06d.jpg")

    # Try to extract frames at sample_fps
    cmd = [
        "ffmpeg",
        "-i",
        str(mp4_path),
        "-vf",
        f"fps={sample_fps}",
        "-q:v",
        "2",
        "-y",
        output_pattern,
    ]

    print(f"Extracting frames from {mp4_path.name}...")
    import subprocess
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

    # List the extracted frames
    frames = sorted(images_dir.glob("*.jpg"))
    print(f"Extracted {len(frames)} frames")
    return frames


def _extract_video_frame_by_index(mp4_path: Path, frame_idx: int, output_path: Path) -> Path:
    """Extract one FPV frame only when a downstream consumer needs a JPG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        return output_path
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(mp4_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = capture.read()
            if ok and frame is not None:
                cv2.imwrite(str(output_path), frame)
                return output_path
        finally:
            capture.release()
    except Exception:
        pass
    try:
        import decord  # type: ignore

        reader = decord.VideoReader(str(mp4_path))
        if int(frame_idx) < len(reader):
            Image.fromarray(reader[int(frame_idx)].asnumpy()).save(output_path, quality=95)
            return output_path
    except Exception as exc:
        raise RuntimeError(f"Failed to extract frame {frame_idx} from {mp4_path}") from exc
    raise RuntimeError(f"Failed to extract frame {frame_idx} from {mp4_path}")


def _find_existing_fpv_image(clip_dir: Path, frame_idx: int) -> Optional[Path]:
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


def _select_indices_by_fps(times_arr: np.ndarray, sample_fps: float) -> list[int]:
    """Select frame indices using timestamps instead of integer stride."""

    n = int(len(times_arr))
    if n <= 0:
        return []
    fps = float(sample_fps)
    if fps <= 0:
        return list(range(n))
    times = [int(x) for x in np.asarray(times_arr).reshape(-1).tolist()]
    period_ns = int(round(1e9 / fps))
    if period_ns <= 0:
        return list(range(n))
    selected = [0]
    next_target = int(times[0]) + period_ns
    for idx in range(1, n):
        if int(times[idx]) >= next_target:
            selected.append(idx)
            while next_target <= int(times[idx]):
                next_target += period_ns
    return selected


def _estimate_fps_from_timestamps(times_arr: np.ndarray, default_fps: float = 20.0) -> float:
    """Estimate the source frame rate from monotonic timestamps."""

    arr = np.asarray(times_arr).reshape(-1)
    if len(arr) < 2:
        return float(default_fps)
    diffs = np.diff(arr.astype(np.int64))
    valid = diffs[diffs > 0]
    if len(valid) == 0:
        return float(default_fps)
    median_dt_s = float(np.median(valid)) / 1e9
    if median_dt_s <= 0:
        return float(default_fps)
    return 1.0 / median_dt_s


def _route_path_for_bev_frame(
    route_points: list[tuple[float, float]],
    frame_pos: int,
    *,
    route_scope: str,
    lookahead_s: float,
    bev_sample_fps: float,
) -> tuple[list[tuple[float, float]] | None, dict[str, Any]]:
    """Return the route overlay to draw for one BEV training frame.

    The route overlay is part of the BEV task condition, not a debug artifact.
    ``global_full`` draws the whole route, including fixed start/end points.
    ``local_future`` is kept for ablations that want a bounded lookahead path
    starting from the current ego pose.
    """

    normalized_scope = str(route_scope or "global_full")
    if normalized_scope == "none" or not route_points:
        return None, {"scope": "none", "point_count": 0}

    safe_pos = max(0, min(int(frame_pos), len(route_points) - 1))
    if normalized_scope == "global_full":
        return list(route_points), {
            "scope": "global_full",
            "point_count": len(route_points),
            "start_sample_index": 0,
            "end_sample_index": len(route_points) - 1,
            "route_endpoint_semantics": "recorded_clip_endpoint",
            "route_progress_authority": "none",
        }

    horizon_frames = max(1, int(math.ceil(max(float(lookahead_s), 0.0) * max(float(bev_sample_fps), 1e-6))))
    end_pos = min(len(route_points) - 1, safe_pos + horizon_frames)
    if end_pos <= safe_pos:
        return None, {
            "scope": "local_future",
            "point_count": 0,
            "start_sample_index": safe_pos,
            "end_sample_index": end_pos,
            "route_endpoint_semantics": "crop_boundary",
            "route_progress_authority": "none",
        }
    local_path = list(route_points[safe_pos : end_pos + 1])
    return local_path, {
        "scope": "local_future",
        "point_count": len(local_path),
        "start_sample_index": safe_pos,
        "end_sample_index": end_pos,
        "lookahead_s": float(lookahead_s),
        "route_endpoint_semantics": "crop_boundary",
        "route_progress_authority": "none",
    }


def _bev_route_evidence_metadata(
    route_overlay_meta: dict[str, Any],
    *,
    bev_alignment: str,
) -> dict[str, Any]:
    """Project renderer metadata into the Reasoning V4 BEV capability contract."""

    alignment = str(bev_alignment)
    overlay_visible = (
        str(route_overlay_meta.get("scope") or "none") != "none"
        and int(route_overlay_meta.get("point_count") or 0) > 0
    )
    endpoint_semantics = str(
        route_overlay_meta.get("route_endpoint_semantics") or "unknown"
    )
    return {
        "frame": "ego" if alignment == "ego_aligned" else "north_up",
        "ego_forward": "up" if alignment == "ego_aligned" else "unknown",
        "route_overlay_visible": overlay_visible,
        "route_direction_authority": (
            "visual_conditional"
            if overlay_visible and alignment == "ego_aligned"
            else "none"
        ),
        "route_progress_authority": str(
            route_overlay_meta.get("route_progress_authority") or "none"
        ),
        "route_terminal_marker_visible": (
            overlay_visible and endpoint_semantics != "unknown"
        ),
        "route_endpoint_semantics": endpoint_semantics,
        "local_safety_authority": False,
        "producer_contract_id": "sidewalk_bev_route_renderer",
        "producer_contract_version": "1.0",
    }


def generate_bev_mp4(
    bev_images_dir: Path,
    output_dir: Path,
    fps: float = 1.0,
    output_filename: str = "bev_video.mp4",
    timestamps_ns: Optional[list[int]] = None,
) -> Path:
    """Generate BEV video from BEV images."""
    bev_images = sorted(list(bev_images_dir.glob("*_bev.jpg")))
    if not bev_images:
        raise RuntimeError("No BEV images found to generate video")

    mp4_path = output_dir / output_filename
    durations, encode_fps = _bev_video_timing(
        num_frames=len(bev_images),
        fallback_fps=fps,
        timestamps_ns=timestamps_ns,
    )

    # Use ffmpeg to generate video
    # First create a list file
    list_file = output_dir / "bev_images_list.txt"
    with open(list_file, "w") as f:
        for img, duration in zip(bev_images, durations):
            # Use absolute path in the list file
            abs_path = img.resolve()
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {duration:.9f}\n")

        # Add the last frame again to avoid cutoff
        if bev_images:
            abs_path = bev_images[-1].resolve()
            f.write(f"file '{abs_path}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264",
            "-vf", "fps={}".format(encode_fps),
            "-pix_fmt", "yuv420p",
            str(mp4_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        # Clean up list file
        if list_file.exists():
            list_file.unlink()

    print(f"Generated BEV video: {mp4_path} (fps={encode_fps:.3f}, frames={len(bev_images)})")
    return mp4_path


def _bev_video_timing(
    *,
    num_frames: int,
    fallback_fps: float,
    timestamps_ns: Optional[list[int]],
) -> tuple[list[float], float]:
    """Return per-frame durations and a stable CFR encode fps.

    BEV images are aligned to sampled FPV frames.  Using a stale default fps
    when writing the mp4 makes videos appear too long or too short.  Prefer the
    actual frame timestamps and only fall back to the CLI sampling fps when
    timestamp inference is unavailable.
    """

    safe_fps = float(fallback_fps) if float(fallback_fps or 0.0) > 0 else 1.0
    fallback_duration = 1.0 / safe_fps
    if num_frames <= 0:
        return [], safe_fps

    if timestamps_ns and len(timestamps_ns) == num_frames:
        ts = [int(t) for t in timestamps_ns]
        diffs = [
            (ts[i + 1] - ts[i]) / 1e9
            for i in range(len(ts) - 1)
            if ts[i + 1] > ts[i]
        ]
        if diffs:
            median_duration = float(np.median(diffs))
            valid_durations = [
                max(1e-3, min(float(dt), median_duration * 5.0))
                for dt in diffs
            ]
            last_duration = valid_durations[-1] if valid_durations else median_duration
            durations = valid_durations + [last_duration]
            encode_fps = 1.0 / max(float(np.median(valid_durations)), 1e-3)
            return durations, encode_fps

    return [fallback_duration for _ in range(num_frames)], safe_fps


def process_sidewalk_ride(
    sidewalk_dir: Path,
    output_dir: Path,
    sample_fps: float = 1.0,
    bev_sample_fps: float = 2.0,
    map_provider: str = "osm",
    use_pose_ekf: bool = False,
    use_ekf_position: bool = False,
    ekf_q_pos: float = 1e-3,
    ekf_q_theta: float = 1e-2,
    ekf_r_xy: float = 9.0,
    ekf_r_theta: float = 0.5,
    extract_fpv_images: bool = False,
    write_north_up_bev: bool = False,
    write_bev_videos: bool = False,
    draw_bev_trajectory: bool = True,
    bev_route_scope: str = "global_full",
    bev_route_lookahead_s: float = 8.0,
    bev_alignment: str = "ego_aligned",
    bev_jpeg_quality: int = 90,
) -> list[dict[str, Any]]:
    """Process one sidewalk ride directory."""
    print(f"\n{'='*60}")
    print(f"Processing: {sidewalk_dir.name}")
    print(f"{'='*60}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if BEV video already exists - regenerate if yes
    for bev_video_name in ("bev_video.mp4", "bev_video_ego_aligned.mp4", "bev_video_north_up.mp4"):
        bev_video_path = sidewalk_dir / bev_video_name
        if bev_video_path.exists():
            print(f"BEV video already exists, removing to regenerate: {bev_video_path}")
            bev_video_path.unlink()
    # Also remove old BEV image dirs if they exist.
    for bev_images_name in ("bev_images", "bev_images_ego_aligned", "bev_images_north_up"):
        bev_images_dir = sidewalk_dir / bev_images_name
        if bev_images_dir.exists():
            shutil.rmtree(bev_images_dir)

    # Load npy files
    gps_arr = np.load(sidewalk_dir / "frame_gps.npy")
    ori_arr = np.load(sidewalk_dir / "frame_orientations.npy")
    times_arr = np.load(sidewalk_dir / "frame_times.npy")
    imu_omega_arr = None
    imu_omega_path = sidewalk_dir / "frame_angular_velocities.npy"
    if imu_omega_path.exists():
        imu_omega_arr = np.load(imu_omega_path)

    num_frames_total = len(gps_arr)
    print(f"Total frames in dataset: {num_frames_total}")

    mp4_path = sidewalk_dir / "front.mp4"
    if not mp4_path.exists():
        raise RuntimeError(f"front.mp4 not found: {mp4_path}")

    total_duration_s = (times_arr[-1] - times_arr[0]) / 1e9
    selected_indices = _select_indices_by_fps(times_arr, bev_sample_fps)
    print(
        f"Total duration: {total_duration_s:.1f}s, "
        f"selected {len(selected_indices)} BEV frames at {bev_sample_fps:g}fps"
    )

    all_indices = list(range(num_frames_total))
    trajectory_sample_fps = _estimate_fps_from_timestamps(times_arr, default_fps=float(sample_fps))

    # Step 2: Build a full per-FPV-frame trajectory.  BEV images are sampled
    # below, but trajectory.json must stay dense so downstream Phase1 builders
    # can choose their own FPV sampling rate and then timestamp-match BEV.
    trajectory = []

    # Smooth GPS positions first so the BEV center does not jitter with raw point noise.
    raw_positions: list[tuple[float, float]] = []
    for data_idx in all_indices:
        raw_positions.append((float(gps_arr[data_idx, 0]), float(gps_arr[data_idx, 1])))

    window_size = 7
    smoothed_positions: list[tuple[float, float]] = []
    for i in range(len(raw_positions)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(raw_positions), i + window_size // 2 + 1)
        window = raw_positions[window_start:window_end]
        lat_sum = sum(lat for lat, _ in window)
        lng_sum = sum(lng for _, lng in window)
        smoothed_positions.append((lat_sum / len(window), lng_sum / len(window)))

    # Calculate GPS headings from points far enough apart. Adjacent 20 FPS GPS
    # samples can be duplicated or dominated by jitter, especially while stopped.
    raw_headings = _calculate_robust_gps_headings(smoothed_positions)

    default_heading = raw_headings[0] if raw_headings else 0.0

    # Apply sliding window smoothing to headings.
    smoothed_headings = raw_headings.copy()
    window_size = 7

    for i in range(len(raw_headings)):
        window = []
        for j in range(max(0, i - window_size // 2), min(len(raw_headings), i + window_size // 2 + 1)):
            h = raw_headings[j]
            if not math.isnan(h):
                window.append(h)

        if window:
            smoothed_headings[i] = _circular_mean_deg(window)
        else:
            smoothed_headings[i] = default_heading

    # BEV position and route overlay must stay tied to the measured GNSS path.
    # The current PoseEKF has no translational odometry input, so using its x/y
    # state can shrink long clips and make FPV frames appear behind the BEV route
    # stage. Keep EKF position opt-in for diagnostics, but use it for heading
    # smoothing only in the normal dataset path.
    position_source = "gps_smoothed"
    heading_source = "gps_robust"

    if use_pose_ekf:
        print("Using Pose EKF for heading smoothing; BEV position remains GPS-smoothed.")
        yaw_list = []
        for data_idx in all_indices:
            qx = float(ori_arr[data_idx, 0])
            qy = float(ori_arr[data_idx, 1])
            qz = float(ori_arr[data_idx, 2])
            qw = float(ori_arr[data_idx, 3])
            _, _, yaw_deg = _quaternion_to_euler_deg(x=qx, y=qy, z=qz, w=qw)
            # Convert ROS/ENU yaw (0 = east, CCW-positive) to compass heading
            # (0 = north, CW-positive) so EKF and BEV renderer use the same convention.
            compass_yaw = (90.0 - float(yaw_deg)) % 360.0
            yaw_list.append(compass_yaw)

        first_idx = all_indices[0]
        use_yaw_obs, yaw_reason = _is_yaw_observation_reliable(yaw_list, ori_arr, all_indices)
        heading_obs = yaw_list if use_yaw_obs else smoothed_headings
        heading_obs_source = "quaternion" if use_yaw_obs else "gps"
        init_heading = heading_obs[0] if heading_obs else default_heading
        if use_yaw_obs:
            print(f"Using quaternion yaw observations for EKF heading ({yaw_reason})")
        else:
            print(f"[WARN] Quaternion yaw unreliable ({yaw_reason}). Using robust GPS bearing observations for EKF heading.")
        ekf = PoseEKF.init_from_latlng_heading(float(gps_arr[first_idx, 0]), float(gps_arr[first_idx, 1]), init_heading, pos_var=25.0, theta_var=0.5)
        fused_positions = []
        fused_headings = []
        prev_time_ns = int(times_arr[first_idx])
        for i, data_idx in enumerate(all_indices):
            t_ns = int(times_arr[data_idx])
            dt = (t_ns - prev_time_ns) / 1e9 if i > 0 else 0.0
            if imu_omega_arr is not None and data_idx < len(imu_omega_arr):
                omega = float(imu_omega_arr[data_idx, 2])
            else:
                cur_yaw = yaw_list[i]
                prev_yaw = yaw_list[i - 1] if i > 0 else cur_yaw
                diff_rad = math.radians(cur_yaw - prev_yaw)
                diff_rad = (diff_rad + math.pi) % (2 * math.pi) - math.pi
                omega = diff_rad / max(dt, 1e-6)
            ekf.predict(omega, dt, q_pos=ekf_q_pos, q_theta=ekf_q_theta)
            ekf.update_gps_and_yaw(
                float(gps_arr[data_idx, 0]),
                float(gps_arr[data_idx, 1]),
                yaw_deg=heading_obs[i],
                r_xy=ekf_r_xy,
                r_theta=ekf_r_theta,
            )
            latf, lngf = ekf.get_latlng()
            fused_positions.append((latf, lngf))
            fused_headings.append(ekf.get_heading_deg())
            prev_time_ns = t_ns

        if use_ekf_position:
            print("[WARN] --use-ekf-position enabled: BEV center/route will use EKF x/y. This is diagnostic and can drift without translational odometry.")
            smoothed_positions = fused_positions
            position_source = "ekf_position"
        else:
            position_source = "gps_smoothed"
        smoothed_headings = fused_headings
        heading_source = f"ekf_{heading_obs_source}"
        print(f"[DEBUG] First 5 frame heading comparisons (quaternion_yaw_deg | gps_heading_deg | ekf_heading_deg, obs={heading_obs_source}):")
        for k in range(min(5, len(all_indices))):
            qyaw = yaw_list[k]
            gps_h = raw_headings[k] if k < len(raw_headings) else None
            ekf_h = smoothed_headings[k]
            print(f"  frame {all_indices[k]}: {qyaw:.6f} | {gps_h:.6f} | {ekf_h:.6f}")

    for traj_pos, data_idx in enumerate(all_indices):
        lat, lng = smoothed_positions[traj_pos]
        heading_deg = smoothed_headings[traj_pos]
        if math.isnan(heading_deg):
            heading_deg = default_heading

        qx = float(ori_arr[data_idx, 0])
        qy = float(ori_arr[data_idx, 1])
        qz = float(ori_arr[data_idx, 2])
        qw = float(ori_arr[data_idx, 3])
        roll_deg, pitch_deg, _ = _quaternion_to_euler_deg(x=qx, y=qy, z=qz, w=qw)

        image_path = _find_existing_fpv_image(sidewalk_dir, data_idx)
        if extract_fpv_images:
            image_path = _extract_video_frame_by_index(mp4_path, data_idx, output_dir / "images" / f"{data_idx:06d}.jpg")

        item = {
            "frame_idx": int(data_idx),
            "frame_name": f"{int(data_idx):06d}",
            "timestamp": int(times_arr[data_idx]),
            "lat": lat,
            "lng": lng,
            "heading_deg": heading_deg,
            "fpv_video_path": str(mp4_path.resolve()),
            "attitude_source": heading_source,
            "position_source": position_source,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
        }
        if image_path is not None:
            item["image_path"] = str(image_path.resolve())
        trajectory.append(item)

    # Save dense trajectory.json for every FPV frame. BEV-only sampling is saved
    # separately below so downstream dataset builders do not double-downsample.
    traj_path = output_dir / "trajectory.json"
    traj_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"Saved full trajectory: {traj_path} ({len(trajectory)} frames)")

    bev_trajectory = [trajectory[int(idx)] for idx in selected_indices if int(idx) < len(trajectory)]
    bev_sample_traj_path = output_dir / "bev_sample_trajectory.json"
    bev_sample_traj_path.write_text(json.dumps(bev_trajectory, indent=2), encoding="utf-8")
    print(f"Saved BEV-sampled trajectory: {bev_sample_traj_path} ({len(bev_trajectory)} frames)")

    # Step 3: Generate minimal BEV maps in the original sidewalk dir. Phase1
    # keeps one ego-aligned BEV by default, and the BEV training input includes
    # the route overlay.  The route/start/end are projected with the exact same
    # rotate-then-crop transform as the OSM raster so they stay anchored to map
    # features even though the ego pose is centered in the image.
    bev_images_dir = sidewalk_dir / "bev_images"
    bev_images_north_up_dir = sidewalk_dir / "bev_images_north_up"
    bev_images_dir.mkdir(parents=True, exist_ok=True)
    if write_north_up_bev:
        bev_images_north_up_dir.mkdir(parents=True, exist_ok=True)

    provider_config = MAP_PROVIDERS.get(map_provider, MAP_PROVIDERS["osm"])
    ego_options = ProcessOptions(
        map_provider=map_provider,
        output_size=1024,
        zoom=19,
        cache_dir=Path("/tmp/bev_cache"),
        draw_trajectory=bool(draw_bev_trajectory and str(bev_route_scope) != "none"),
        add_attribution=True,
        bev_alignment=str(bev_alignment),
    )
    north_up_options = None
    if write_north_up_bev and str(bev_alignment) != "north_up":
        north_up_options = ProcessOptions(
            map_provider=map_provider,
            output_size=1024,
            zoom=19,
            cache_dir=Path("/tmp/bev_cache"),
            draw_trajectory=bool(draw_bev_trajectory and str(bev_route_scope) != "none"),
            add_attribution=True,
            bev_alignment="north_up",
        )

    # Global trajectory - all FPV frame GPS points. BEV images are sparse, but
    # the route overlay should still represent the complete clip path.
    global_nav_path = smoothed_positions

    samples = []
    bev_index_entries = []
    mode_text = str(bev_alignment) if north_up_options is None else f"{bev_alignment} and north-up"
    print(f"Generating {mode_text} BEV maps for {len(bev_trajectory)} sampled frames...")
    for i, frame in enumerate(tqdm(bev_trajectory, desc="Generating BEV")):
        lat = frame["lat"]
        lng = frame["lng"]
        heading_deg = frame["heading_deg"]
        frame_name = str(frame.get("frame_name") or f"{int(frame.get('frame_idx', i)):06d}")
        fpv_image_path = Path(frame["image_path"]) if frame.get("image_path") else None
        route_overlay_path, route_overlay_meta = _route_path_for_bev_frame(
            global_nav_path,
            int(frame.get("frame_idx", i)),
            route_scope=str(bev_route_scope) if draw_bev_trajectory else "none",
            lookahead_s=float(bev_route_lookahead_s),
            bev_sample_fps=float(trajectory_sample_fps),
        )

        try:
            ego_bev_image = render_bev_map(
                lat=lat,
                lng=lng,
                heading_deg=heading_deg,
                navigation_path_gps=route_overlay_path,
                options=ego_options,
                provider_config=provider_config,
            )
            north_up_bev_image = None
            if north_up_options is not None:
                north_up_bev_image = render_bev_map(
                    lat=lat,
                    lng=lng,
                    heading_deg=heading_deg,
                    navigation_path_gps=route_overlay_path,
                    options=north_up_options,
                    provider_config=provider_config,
                )

            bev_filename = f"{frame_name}_bev.jpg"
            bev_path = bev_images_dir / bev_filename
            ego_bev_image.save(bev_path, quality=int(bev_jpeg_quality))
            bev_north_up_path = None
            if north_up_bev_image is not None:
                bev_north_up_path = bev_images_north_up_dir / bev_filename
                north_up_bev_image.save(bev_north_up_path, quality=int(bev_jpeg_quality))

            bev_route_evidence = _bev_route_evidence_metadata(
                route_overlay_meta,
                bev_alignment=str(bev_alignment),
            )

            # Write a small metadata file next to each BEV image with the heading actually used for rendering
            try:
                meta = {
                    "source_frame_idx": int(frame.get("frame_idx", i)),
                    "source_frame_name": frame_name,
                    "source_timestamp_ns": int(frame["timestamp"]),
                    "used_heading_deg": float(heading_deg),
                    "attitude_source": frame.get("attitude_source", "unknown"),
                    "position_source": frame.get("position_source", "unknown"),
                    "use_ekf_position": bool(use_ekf_position),
                    "bev_alignment": str(bev_alignment),
                    "bev_sample_fps": float(bev_sample_fps),
                    "draw_trajectory": bool(draw_bev_trajectory),
                    "bev_route_overlay": route_overlay_meta,
                    "bev_route_evidence": bev_route_evidence,
                }
                meta_path = bev_path.with_suffix(bev_path.suffix + ".bevmeta.json")
                meta_path.write_text(json.dumps(meta), encoding="utf-8")
                if bev_north_up_path is not None:
                    north_up_meta = dict(meta)
                    north_up_meta["bev_alignment"] = "north_up"
                    north_up_meta["bev_route_evidence"] = (
                        _bev_route_evidence_metadata(
                            route_overlay_meta,
                            bev_alignment="north_up",
                        )
                    )
                    north_up_meta_path = bev_north_up_path.with_suffix(bev_north_up_path.suffix + ".bevmeta.json")
                    north_up_meta_path.write_text(json.dumps(north_up_meta), encoding="utf-8")
            except Exception:
                pass

            bev_index_entry = {
                "frame_idx": int(frame.get("frame_idx", i)),
                "frame_name": frame_name,
                "timestamp": int(frame["timestamp"]),
                "bev_image_path": str(bev_path.resolve()),
                "bev_alignment": str(bev_alignment),
                "bev_sample_fps": float(bev_sample_fps),
                "position_source": frame.get("position_source", "unknown"),
                "attitude_source": frame.get("attitude_source", "unknown"),
                "draw_trajectory": bool(draw_bev_trajectory),
                "bev_route_overlay": route_overlay_meta,
                "bev_route_evidence": bev_route_evidence,
            }
            bev_index_entries.append(bev_index_entry)
            sample = {
                "ride_id": sidewalk_dir.name,
                "source": "sidewalk_npy",
                "fpv_image_path": str(fpv_image_path) if fpv_image_path is not None else None,
                "fpv_video_path": str(mp4_path.resolve()),
                "fpv_frame_idx": int(frame.get("frame_idx", i)),
                "bev_image_path": str(bev_path),
                "bev_image_alignment": str(bev_alignment),
                "bev_route_evidence": bev_route_evidence,
                "metadata": {
                    "lat": lat,
                    "lng": lng,
                    "timestamp": frame["timestamp"],
                    "heading_deg": heading_deg,
                    "position_source": frame.get("position_source", "unknown"),
                    "attitude_source": frame.get("attitude_source", "unknown"),
                    "bev_route_overlay": route_overlay_meta,
                    "bev_route_evidence": bev_route_evidence,
                    "bev_outputs": {
                        str(bev_alignment): str(bev_path),
                    },
                },
                "instruction": "",
                "action": "",
                "plan": "",
                "think": "",
            }
            if str(bev_alignment) == "ego_aligned":
                sample["bev_image_path_ego_aligned"] = str(bev_path)
            elif str(bev_alignment) == "north_up":
                sample["bev_image_path_north_up"] = str(bev_path)
            if bev_north_up_path is not None:
                sample["bev_image_path_north_up"] = str(bev_north_up_path)
                sample["metadata"]["bev_outputs"]["north_up"] = str(bev_north_up_path)
            samples.append(sample)
        except Exception as e:
            print(f"Warning: failed frame {i}: {e}")

    # Save samples.json in output dir
    samples_path = output_dir / "samples.json"
    samples_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} samples: {samples_path}")
    bev_index = {
        "schema_version": "sidewalk_bev_index_v1",
        "clip_id": sidewalk_dir.name,
        "bev_sample_fps": float(bev_sample_fps),
        "bev_alignment": str(bev_alignment),
        "draw_trajectory": bool(draw_bev_trajectory),
        "bev_route_scope": str(bev_route_scope) if draw_bev_trajectory else "none",
        "bev_route_lookahead_s": float(bev_route_lookahead_s),
        "entries": bev_index_entries,
    }
    bev_index_path = sidewalk_dir / "bev_index.json"
    bev_index_path.write_text(json.dumps(bev_index, indent=2), encoding="utf-8")
    (bev_images_dir / "bev_index.json").write_text(json.dumps(bev_index, indent=2), encoding="utf-8")
    print(f"Saved BEV index: {bev_index_path}")

    # Step 4: Generate BEV mp4 video in the original sidewalk dir
    if samples and write_bev_videos:
        try:
            print("Generating BEV videos...")
            bev_timestamps_ns = [int(frame["timestamp"]) for frame in bev_trajectory if frame.get("timestamp") is not None]
            if len(bev_timestamps_ns) != len(bev_trajectory):
                bev_timestamps_ns = None
            generate_bev_mp4(bev_images_dir, sidewalk_dir, bev_sample_fps, timestamps_ns=bev_timestamps_ns)
            generate_bev_mp4(
                bev_images_dir,
                sidewalk_dir,
                bev_sample_fps,
                output_filename="bev_video_ego_aligned.mp4",
                timestamps_ns=bev_timestamps_ns,
            )
            if write_north_up_bev:
                generate_bev_mp4(
                    bev_images_north_up_dir,
                    sidewalk_dir,
                    bev_sample_fps,
                    output_filename="bev_video_north_up.mp4",
                    timestamps_ns=bev_timestamps_ns,
                )
        except Exception as e:
            print(f"Warning: failed to generate BEV video: {e}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Process sidewalk dataset to BEV maps")
    parser.add_argument(
        "--sidewalk-root",
        type=Path,
        default="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0",
        help="Root of sidewalk dataset, or a single ride directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default="/media/ubuntu/data1/sr_nav_chunk0_final",
        help="Legacy output root directory. Ignored by default; use --write-to-output-root to restore old behavior.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=20.0,
        help="Legacy FPV extraction sampling rate. Phase1 minimal mode does not extract FPV by default.",
    )
    parser.add_argument(
        "--bev-sample-fps",
        type=float,
        default=2.0,
        help="Sampling rate for persisted BEV images. Use 2fps for Phase1 storage control.",
    )
    parser.add_argument(
        "--map-provider",
        type=str,
        default="osm",
        help="Map provider (osm, carto, etc.)",
    )
    parser.add_argument(
        "--ride-filter",
        type=str,
        default=None,
        help="Process only rides matching this substring",
    )
    parser.add_argument(
        "--use-pose-ekf",
        action="store_true",
        help="Use 2D pose EKF for heading smoothing. BEV position stays GPS-smoothed unless --use-ekf-position is set.",
    )
    parser.add_argument(
        "--use-ekf-position",
        action="store_true",
        help="Diagnostic legacy mode: also use EKF x/y for BEV center/route. Avoid for dataset generation unless translational odometry is modeled.",
    )
    parser.add_argument("--ekf-q-pos", type=float, default=1e-3, help="EKF process noise strength for x/y")
    parser.add_argument("--ekf-q-theta", type=float, default=1e-2, help="EKF process noise strength for theta")
    parser.add_argument("--ekf-r-xy", type=float, default=9.0, help="EKF GPS position observation variance")
    parser.add_argument("--ekf-r-theta", type=float, default=0.5, help="EKF yaw/compass observation variance")
    parser.add_argument("--extract-fpv-images", action="store_true", help="Persist sampled FPV JPGs. Off by default to avoid duplicating front.mp4.")
    parser.add_argument("--write-north-up-bev", action="store_true", help="Also write north-up BEV images for debugging.")
    parser.add_argument("--write-bev-videos", action="store_true", help="Write BEV mp4 debug videos. Off by default for minimal Phase1 data.")
    parser.add_argument(
        "--draw-bev-trajectory",
        action="store_true",
        help="Compatibility flag: force drawing the full recorded clip trajectory on BEV images.",
    )
    parser.add_argument(
        "--no-bev-route-overlay",
        action="store_true",
        help="Disable route overlay in BEV training images. By default the full route is drawn.",
    )
    parser.add_argument(
        "--bev-route-scope",
        default="global_full",
        choices=["local_future", "global_full", "none"],
        help="Route overlay in BEV images. global_full draws the whole task route; local_future is an ablation/debug mode.",
    )
    parser.add_argument(
        "--bev-route-lookahead-s",
        type=float,
        default=8.0,
        help="Lookahead horizon in seconds for local_future BEV route overlay.",
    )
    parser.add_argument(
        "--bev-alignment",
        default="ego_aligned",
        choices=["ego_aligned", "north_up"],
        help="BEV frame for saved bev_images. ego_aligned rotates robot heading up; north_up keeps OSM north-up while centering ego.",
    )
    parser.add_argument("--bev-jpeg-quality", type=int, default=90, help="JPEG quality for persisted BEV images")
    parser.add_argument(
        "--write-to-output-root",
        action="store_true",
        help="Legacy behavior: write trajectory/samples/FPV extraction under --output-root instead of the original clip directory.",
    )
    args = parser.parse_args()
    if not args.write_to_output_root:
        print("[INFO] Writing generated artifacts into each original clip directory. --output-root is ignored.")

    # Check if sidewalk_root is a single ride directory (has frame_gps.npy and front.mp4)
    sidewalk_root = args.sidewalk_root
    is_single_ride = (sidewalk_root / "frame_gps.npy").exists() and (sidewalk_root / "front.mp4").exists()

    ride_dirs = []
    if is_single_ride:
        ride_dirs = [sidewalk_root]
        print(f"Processing single ride directory: {sidewalk_root.name}")
    else:
        # Find all ride directories
        ride_dirs = sorted([d for d in sidewalk_root.iterdir() if d.is_dir()])
        if args.ride_filter:
            ride_dirs = [d for d in ride_dirs if args.ride_filter in d.name]
        print(f"Found {len(ride_dirs)} rides to process")

    all_samples = []
    for ride_dir in ride_dirs:
        if args.write_to_output_root and is_single_ride:
            output_dir = args.output_root
        elif args.write_to_output_root:
            output_dir = args.output_root / ride_dir.name
        else:
            output_dir = ride_dir
        draw_route_overlay = not bool(args.no_bev_route_overlay)
        bev_route_scope = str(args.bev_route_scope)
        if args.draw_bev_trajectory:
            draw_route_overlay = True
            bev_route_scope = "global_full"
        try:
            samples = process_sidewalk_ride(
                ride_dir,
                output_dir,
                sample_fps=args.sample_fps,
                bev_sample_fps=args.bev_sample_fps,
                map_provider=args.map_provider,
                use_pose_ekf=args.use_pose_ekf,
                use_ekf_position=args.use_ekf_position,
                ekf_q_pos=args.ekf_q_pos,
                ekf_q_theta=args.ekf_q_theta,
                ekf_r_xy=args.ekf_r_xy,
                ekf_r_theta=args.ekf_r_theta,
                extract_fpv_images=bool(args.extract_fpv_images),
                write_north_up_bev=bool(args.write_north_up_bev),
                write_bev_videos=bool(args.write_bev_videos),
                draw_bev_trajectory=draw_route_overlay,
                bev_route_scope=bev_route_scope,
                bev_route_lookahead_s=float(args.bev_route_lookahead_s),
                bev_alignment=str(args.bev_alignment),
                bev_jpeg_quality=int(args.bev_jpeg_quality),
            )
            all_samples.extend(samples)
        except Exception as e:
            print(f"\nError processing {ride_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined samples
    if all_samples:
        combined_root = args.output_root if args.write_to_output_root else sidewalk_root
        combined_path = combined_root / "all_samples.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(all_samples, indent=2), encoding="utf-8")
        print(f"\n{'='*60}")
        print(f"Done!")
        print(f"Total samples processed: {len(all_samples)}")
        print(f"Combined samples: {combined_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
