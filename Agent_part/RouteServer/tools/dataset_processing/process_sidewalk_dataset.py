#!/usr/bin/env python3
"""Process sidewalk dataset (npy + mp4) to generate BEV maps."""

import sys
import os
import math
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Any
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


def generate_bev_mp4(
    bev_images_dir: Path,
    output_dir: Path,
    fps: float = 1.0,
) -> Path:
    """Generate BEV video from BEV images."""
    bev_images = sorted(list(bev_images_dir.glob("*_bev.jpg")))
    if not bev_images:
        raise RuntimeError("No BEV images found to generate video")

    mp4_path = output_dir / "bev_video.mp4"

    # Use ffmpeg to generate video
    # First create a list file
    list_file = output_dir / "bev_images_list.txt"
    with open(list_file, "w") as f:
        for img in bev_images:
            # Use absolute path in the list file
            abs_path = img.resolve()
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {1.0/fps}\n")

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
            "-vf", "fps={}".format(fps),
            "-pix_fmt", "yuv420p",
            str(mp4_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        # Clean up list file
        if list_file.exists():
            list_file.unlink()

    print(f"Generated BEV video: {mp4_path}")
    return mp4_path


def process_sidewalk_ride(
    sidewalk_dir: Path,
    output_dir: Path,
    sample_fps: float = 1.0,
    map_provider: str = "osm",
    use_pose_ekf: bool = False,
    ekf_q_pos: float = 1e-3,
    ekf_q_theta: float = 1e-2,
    ekf_r_xy: float = 9.0,
    ekf_r_theta: float = 0.5,
) -> list[dict[str, Any]]:
    """Process one sidewalk ride directory."""
    print(f"\n{'='*60}")
    print(f"Processing: {sidewalk_dir.name}")
    print(f"{'='*60}")

    # Check if BEV video already exists - regenerate if yes
    bev_video_path = sidewalk_dir / "bev_video.mp4"
    if bev_video_path.exists():
        print(f"BEV video already exists, removing to regenerate: {bev_video_path}")
        bev_video_path.unlink()
    # Also remove old bev_images if exists
    bev_images_dir = sidewalk_dir / "bev_images"
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

    # Step 1: Extract frames from video
    mp4_path = sidewalk_dir / "front.mp4"
    if not mp4_path.exists():
        raise RuntimeError(f"front.mp4 not found: {mp4_path}")

    # Extract frames and get their indices
    extracted_frames = _extract_frames_from_mp4(mp4_path, output_dir, sample_fps)

    # Estimate which indices we extracted (approximate)
    total_duration_s = (times_arr[-1] - times_arr[0]) / 1e9
    num_samples = len(extracted_frames)
    print(f"Total duration: {total_duration_s:.1f}s, sampling {num_samples} frames")

    # Calculate step - make sure we have matching number of GPS/pose for extracted images
    step = max(1, num_frames_total // max(1, num_samples))
    selected_indices = list(range(0, num_frames_total, step))[:num_samples]
    print(f"Selected indices: {len(selected_indices)} (step={step})")

    # If mismatch, adjust
    if len(selected_indices) != len(extracted_frames):
        selected_indices = selected_indices[: len(extracted_frames)]

    # Step 2: Build trajectory
    trajectory = []

    # Smooth GPS positions first so the BEV center does not jitter with raw point noise.
    raw_positions: list[tuple[float, float]] = []
    for data_idx in selected_indices:
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

    # First calculate all raw headings from the smoothed positions.
    raw_headings = []
    for i in range(len(smoothed_positions)):
        lat, lng = smoothed_positions[i]

        # Calculate heading from GPS - use current point and next point (or previous if last)
        if i < len(smoothed_positions) - 1:
            next_lat, next_lng = smoothed_positions[i + 1]
            h = _calculate_heading_from_gps(lat, lng, next_lat, next_lng)
        elif i > 0:
            prev_lat, prev_lng = smoothed_positions[i - 1]
            h = _calculate_heading_from_gps(prev_lat, prev_lng, lat, lng)
        else:
            h = 0.0

        raw_headings.append(h)

    default_heading = raw_headings[0] if raw_headings else 0.0

    # Apply sliding window smoothing to headings
    smoothed_headings = raw_headings.copy()
    window_size = 7  # 7-point window

    for i in range(len(raw_headings)):
        # Collect nearby headings
        window = []
        for j in range(max(0, i - window_size // 2), min(len(raw_headings), i + window_size // 2 + 1)):
            h = raw_headings[j]
            if not math.isnan(h):
                window.append(h)

        if window:
            # Handle angle wrap-around (0-360) properly by converting to unit vectors
            x_sum = 0.0
            y_sum = 0.0
            for h in window:
                rad = math.radians(h)
                x_sum += math.cos(rad)
                y_sum += math.sin(rad)

            avg_x = x_sum / len(window)
            avg_y = y_sum / len(window)
            smoothed_h = math.degrees(math.atan2(avg_y, avg_x))
            smoothed_h = (smoothed_h + 360.0) % 360.0  # Normalize to 0-360
            smoothed_headings[i] = smoothed_h
        else:
            smoothed_headings[i] = default_heading

    # Now process each frame with smoothed position and smoothed heading
    # Optionally run Pose EKF fusion to further smooth position+heading
    if use_pose_ekf:
        print("Using Pose EKF fusion for position and heading...")
        # build yaw list from quaternion orientations for selected indices
        yaw_list = []
        for data_idx in selected_indices:
            qx = float(ori_arr[data_idx, 0])
            qy = float(ori_arr[data_idx, 1])
            qz = float(ori_arr[data_idx, 2])
            qw = float(ori_arr[data_idx, 3])
            _, _, yaw_deg = _quaternion_to_euler_deg(x=qx, y=qy, z=qz, w=qw)
            # Convert ROS/ENU yaw (0 = east, CCW-positive) to compass heading
            # (0 = north, CW-positive) so EKF and BEV renderer use the same convention.
            compass_yaw = (90.0 - float(yaw_deg)) % 360.0
            yaw_list.append(compass_yaw)

        # Initialize EKF at first frame
        first_idx = selected_indices[0]
        # Detect if yaw_list appears unreliable (near-constant), in which case prefer GPS bearing
        yaw_range = max(yaw_list) - min(yaw_list) if yaw_list else 0.0
        use_yaw_obs = True
        if yaw_range < 0.5:
            # yaw is nearly constant -> likely invalid quaternions
            use_yaw_obs = False
            # compute initial heading from GPS bearing between first two smoothed positions if possible
            if len(smoothed_positions) >= 2:
                lat0, lng0 = smoothed_positions[0]
                lat1, lng1 = smoothed_positions[1]
                init_heading = _calculate_heading_from_gps(lat0, lng0, lat1, lng1)
            else:
                init_heading = raw_headings[0] if raw_headings else 0.0
            print(f"[WARN] Yaw observations appear constant (range={yaw_range:.3f}). Disabling yaw updates; initializing EKF heading from GPS: {init_heading:.3f}")
            ekf = PoseEKF.init_from_latlng_heading(float(gps_arr[first_idx, 0]), float(gps_arr[first_idx, 1]), init_heading, pos_var=25.0, theta_var=0.5)
        else:
            ekf = PoseEKF.init_from_latlng_heading(float(gps_arr[first_idx, 0]), float(gps_arr[first_idx, 1]), yaw_list[0], pos_var=25.0, theta_var=0.5)
        fused_positions = []
        fused_headings = []
        prev_time_ns = int(times_arr[first_idx])
        for i, data_idx in enumerate(selected_indices):
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
            if use_yaw_obs:
                ekf.update_gps_and_yaw(
                    float(gps_arr[data_idx, 0]),
                    float(gps_arr[data_idx, 1]),
                    yaw_deg=yaw_list[i],
                    r_xy=ekf_r_xy,
                    r_theta=ekf_r_theta,
                )
            else:
                ekf.update_gps(float(gps_arr[data_idx, 0]), float(gps_arr[data_idx, 1]), r_xy=ekf_r_xy)
            latf, lngf = ekf.get_latlng()
            fused_positions.append((latf, lngf))
            fused_headings.append(ekf.get_heading_deg())
            prev_time_ns = t_ns

        smoothed_positions = fused_positions
        smoothed_headings = fused_headings
        # Print a quick comparison for the first few frames: quaternion yaw, gps heading, ekf heading
        print("[DEBUG] First 5 frame heading comparisons (quaternion_yaw_deg | gps_heading_deg | ekf_heading_deg):")
        for k in range(min(5, len(selected_indices))):
            qyaw = yaw_list[k]
            gps_h = raw_headings[k] if k < len(raw_headings) else None
            ekf_h = smoothed_headings[k]
            print(f"  frame {k}: {qyaw:.6f} | {gps_h:.6f} | {ekf_h:.6f}")

    for img_idx, (img_path, data_idx) in enumerate(zip(extracted_frames, selected_indices)):
        # GPS format looks like: [lat, lng, alt, ?, ?]
        lat, lng = smoothed_positions[img_idx]

        # Use smoothed heading
        heading_deg = smoothed_headings[img_idx]

        # Fallback if heading is NaN
        if math.isnan(heading_deg):
            heading_deg = default_heading

        # Still get roll/pitch from quaternion (though they're 0)
        qx = float(ori_arr[data_idx, 0])
        qy = float(ori_arr[data_idx, 1])
        qz = float(ori_arr[data_idx, 2])
        qw = float(ori_arr[data_idx, 3])
        roll_deg, pitch_deg, _ = _quaternion_to_euler_deg(x=qx, y=qy, z=qz, w=qw)

        # Use original FPV frame index (data_idx) for filename
        # This ensures alignment with InternNav's frame indexing
        img_name = f"{data_idx:06d}.jpg"
        final_img_path = output_dir / "images" / img_name
        if img_path != final_img_path:
            if final_img_path.exists():
                final_img_path.unlink()
            img_path.rename(final_img_path)

        # Add to trajectory
        trajectory.append(
            {
                "timestamp": int(times_arr[data_idx]),
                "lat": lat,
                "lng": lng,
                "heading_deg": heading_deg,
                "image_path": str(final_img_path.resolve()),
                "attitude_source": "ekf" if use_pose_ekf else "gps",
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
            }
        )

    # Save trajectory.json
    traj_path = output_dir / "trajectory.json"
    traj_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"Saved trajectory: {traj_path}")

    # Step 3: Generate BEV maps with global trajectory - in the original sidewalk dir
    bev_images_dir = sidewalk_dir / "bev_images"
    bev_images_dir.mkdir(parents=True, exist_ok=True)

    provider_config = MAP_PROVIDERS.get(map_provider, MAP_PROVIDERS["osm"])
    options = ProcessOptions(
        map_provider=map_provider,
        output_size=1024,
        zoom=19,
        cache_dir=Path("/tmp/bev_cache"),
        draw_trajectory=True,
        add_attribution=True,
    )

    # Global trajectory - all GPS points
    global_nav_path = smoothed_positions

    samples = []
    print(f"Generating BEV maps for {len(trajectory)} frames...")
    for i, frame in enumerate(tqdm(trajectory, desc="Generating BEV")):
        lat = frame["lat"]
        lng = frame["lng"]
        heading_deg = frame["heading_deg"]
        fpv_image_path = Path(frame["image_path"])

        try:
            bev_image = render_bev_map(
                lat=lat,
                lng=lng,
                heading_deg=heading_deg,
                navigation_path_gps=global_nav_path,
                options=options,
                provider_config=provider_config,
            )

            bev_filename = f"{fpv_image_path.stem}_bev.jpg"
            bev_path = bev_images_dir / bev_filename
            bev_image.save(bev_path, quality=95)

            # Write a small metadata file next to each BEV image with the heading actually used for rendering
            try:
                meta = {
                    "used_heading_deg": float(heading_deg),
                    "attitude_source": frame.get("attitude_source", "unknown"),
                }
                meta_path = bev_path.with_suffix(bev_path.suffix + ".bevmeta.json")
                meta_path.write_text(json.dumps(meta), encoding="utf-8")
            except Exception:
                pass

            sample = {
                "ride_id": sidewalk_dir.name,
                "source": "sidewalk_npy",
                "fpv_image_path": str(fpv_image_path),
                "bev_image_path": str(bev_path),
                "metadata": {
                    "lat": lat,
                    "lng": lng,
                    "timestamp": frame["timestamp"],
                    "heading_deg": heading_deg,
                },
                "instruction": "",
                "action": "",
                "plan": "",
                "think": "",
            }
            samples.append(sample)
        except Exception as e:
            print(f"Warning: failed frame {i}: {e}")

    # Save samples.json in output dir
    samples_path = output_dir / "samples.json"
    samples_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} samples: {samples_path}")

    # Step 4: Generate BEV mp4 video in the original sidewalk dir
    if samples:
        try:
            print("Generating BEV video...")
            generate_bev_mp4(bev_images_dir, sidewalk_dir, sample_fps)
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
        help="Output root directory",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Sampling rate for frames",
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
        help="Use 2D pose EKF to fuse orientation (from quaternions) and GPS for smoother pose/heading",
    )
    parser.add_argument("--ekf-q-pos", type=float, default=1e-3, help="EKF process noise strength for x/y")
    parser.add_argument("--ekf-q-theta", type=float, default=1e-2, help="EKF process noise strength for theta")
    parser.add_argument("--ekf-r-xy", type=float, default=9.0, help="EKF GPS position observation variance")
    parser.add_argument("--ekf-r-theta", type=float, default=0.5, help="EKF yaw/compass observation variance")
    args = parser.parse_args()

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
        if is_single_ride:
            output_dir = args.output_root
        else:
            output_dir = args.output_root / ride_dir.name
        try:
            samples = process_sidewalk_ride(
                ride_dir,
                output_dir,
                sample_fps=args.sample_fps,
                map_provider=args.map_provider,
                use_pose_ekf=args.use_pose_ekf,
                ekf_q_pos=args.ekf_q_pos,
                ekf_q_theta=args.ekf_q_theta,
                ekf_r_xy=args.ekf_r_xy,
                ekf_r_theta=args.ekf_r_theta,
            )
            all_samples.extend(samples)
        except Exception as e:
            print(f"\nError processing {ride_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined samples
    if all_samples:
        combined_path = args.output_root / "all_samples.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(all_samples, indent=2), encoding="utf-8")
        print(f"\n{'='*60}")
        print(f"Done!")
        print(f"Total samples processed: {len(all_samples)}")
        print(f"Combined samples: {combined_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
