#!/usr/bin/env python3
"""Process realword MCAP files into sidewalk chunk_0 format.

This script:
1. Extracts data from MCAP using mcap_to_sidewalk_inputs.py
2. Converts to chunk_0 format (npy files + front.mp4)
3. Generates BEV video using process_sidewalk_dataset.py
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Any
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _quat_xyzw_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return float(qx), float(qy), float(qz), float(qw)


def _has_odom_pose(frame: dict[str, Any]) -> bool:
    return str(frame.get("odom_source") or "").lower() == "odom"


def mcap_to_chunk_format(
    mcap_path: Path,
    output_chunk_dir: Path,
    temp_extract_dir: Path,
    sample_fps: float = 1.0,
    image_topic: str = "auto",
    gps_topic: str = "auto",
    heading_topic: str = "auto",
    imu_topic: str = "auto",
    pose_topic: str = "auto",
    generate_bev: bool = True,
):
    """Convert a single MCAP file to SideWalk chunk format.

    Args:
        mcap_path: Input MCAP file to decode.
        output_chunk_dir: Parent directory that receives one clip directory per
            MCAP stem.
        temp_extract_dir: Temporary directory for extracted frames and
            intermediate ffmpeg lists.
        sample_fps: Target frame sampling rate for extracted FPV images.
        image_topic: Image topic name, or ``auto``.
        gps_topic: NavSatFix topic name, or ``auto``.
        heading_topic: Heading topic name, or ``auto``.
        imu_topic: Optional IMU topic name, ``auto``, or ``none``.
        pose_topic: Optional pose topic name, ``auto``, or ``none``.
        generate_bev: Whether to run the downstream BEV generation stage.

    Returns:
        The created clip directory.

    Raises:
        RuntimeError: If MCAP extraction fails or required extracted artifacts
            are missing.
    """

    print(f"\n{'='*80}")
    print(f"Processing: {mcap_path.name}")
    print(f"{'='*80}")

    # Create output directories
    output_dir = output_chunk_dir / mcap_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract from MCAP
    print("\n[1/5] Extracting data from MCAP...")
    extract_dir = temp_extract_dir / mcap_path.stem

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "mcap_to_sidewalk_inputs.py"),
        "--mcap-path", str(mcap_path),
        "--output-dir", str(extract_dir),
        "--sample-fps", str(sample_fps),
        "--image-topic", image_topic,
        "--gps-topic", gps_topic,
        "--heading-topic", heading_topic,
        "--imu-topic", imu_topic,
        "--pose-topic", pose_topic,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting from MCAP:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Failed to extract MCAP: {mcap_path}")

    # Load trajectory
    trajectory_path = extract_dir / "trajectory.json"
    if not trajectory_path.exists():
        raise RuntimeError(f"trajectory.json not found: {trajectory_path}")

    trajectory = json.loads(trajectory_path.read_text())
    print(f"Extracted {len(trajectory)} frames")

    # Keep extraction metadata with the prepared clip.  Downstream Phase1
    # builders use it to recover conventions such as quaternion storage order.
    for sidecar_name in ("mcap_extract_meta.json", "frames_manifest.json"):
        sidecar_path = extract_dir / sidecar_name
        if sidecar_path.is_file():
            shutil.copy2(sidecar_path, output_dir / sidecar_name)

    # Step 2: Build npy files
    print("\n[2/5] Creating npy files...")

    num_frames = len(trajectory)

    # frame_times.npy (int64)
    frame_times = np.array([f["timestamp"] for f in trajectory], dtype=np.int64)
    np.save(output_dir / "frame_times.npy", frame_times)

    # frame_gps.npy (float64, shape [N, 5] - lat, lng, alt, 0, 0)
    frame_gps = np.zeros((num_frames, 5), dtype=np.float64)
    for i, f in enumerate(trajectory):
        frame_gps[i, 0] = f["lat"]
        frame_gps[i, 1] = f["lng"]
        frame_gps[i, 2] = 0.0  # alt, not in trajectory
        frame_gps[i, 3] = 0.0
        frame_gps[i, 4] = 0.0
    np.save(output_dir / "frame_gps.npy", frame_gps)

    # frame_orientations.npy stores ROS quaternion order: x, y, z, w.
    frame_orientations = np.zeros((num_frames, 4), dtype=np.float64)
    frame_orientations[:, 3] = 1.0
    for i, f in enumerate(trajectory):
        if _has_odom_pose(f):
            frame_orientations[i, 0] = float(f.get("odom_orientation_x", 0.0) or 0.0)
            frame_orientations[i, 1] = float(f.get("odom_orientation_y", 0.0) or 0.0)
            frame_orientations[i, 2] = float(f.get("odom_orientation_z", 0.0) or 0.0)
            frame_orientations[i, 3] = float(f.get("odom_orientation_w", 1.0) or 1.0)
        elif "roll_deg" in f and "pitch_deg" in f and "attitude_yaw_deg" in f:
            roll = np.radians(f["roll_deg"])
            pitch = np.radians(f["pitch_deg"])
            yaw = np.radians(f["attitude_yaw_deg"])
            qx, qy, qz, qw = _quat_xyzw_from_euler(roll, pitch, yaw)
            frame_orientations[i, 0] = qx
            frame_orientations[i, 1] = qy
            frame_orientations[i, 2] = qz
            frame_orientations[i, 3] = qw
    np.save(output_dir / "frame_orientations.npy", frame_orientations)

    # frame_positions.npy (float64, shape [N, 3]) from nearest /odom pose.
    frame_positions = np.zeros((num_frames, 3), dtype=np.float64)
    for i, f in enumerate(trajectory):
        if not _has_odom_pose(f):
            continue
        frame_positions[i, 0] = float(f.get("odom_position_x", 0.0) or 0.0)
        frame_positions[i, 1] = float(f.get("odom_position_y", 0.0) or 0.0)
        frame_positions[i, 2] = float(f.get("odom_position_z", 0.0) or 0.0)
    np.save(output_dir / "frame_positions.npy", frame_positions)

    # frame_angular_velocities.npy (float64, shape [N, 3])
    frame_angular_velocities = np.zeros((num_frames, 3), dtype=np.float64)
    for i, f in enumerate(trajectory):
        if not _has_odom_pose(f):
            continue
        frame_angular_velocities[i, 0] = float(f.get("odom_angular_velocity_x", 0.0) or 0.0)
        frame_angular_velocities[i, 1] = float(f.get("odom_angular_velocity_y", 0.0) or 0.0)
        frame_angular_velocities[i, 2] = float(f.get("odom_angular_velocity_z", 0.0) or 0.0)
    np.save(output_dir / "frame_angular_velocities.npy", frame_angular_velocities)

    # frame_linear_velocities.npy (float64, shape [N, 3])
    frame_linear_velocities = np.zeros((num_frames, 3), dtype=np.float64)
    for i, f in enumerate(trajectory):
        if not _has_odom_pose(f):
            continue
        frame_linear_velocities[i, 0] = float(f.get("odom_linear_velocity_x", 0.0) or 0.0)
        frame_linear_velocities[i, 1] = float(f.get("odom_linear_velocity_y", 0.0) or 0.0)
        frame_linear_velocities[i, 2] = float(f.get("odom_linear_velocity_z", 0.0) or 0.0)
    np.save(output_dir / "frame_linear_velocities.npy", frame_linear_velocities)

    # frame_speeds.npy (float64, shape [N])
    frame_speeds = np.linalg.norm(frame_linear_velocities[:, :2], axis=1).astype(np.float64)
    np.save(output_dir / "frame_speeds.npy", frame_speeds)

    print(f"Created npy files for {num_frames} frames")

    # Step 3: Create front.mp4 from images
    print("\n[3/5] Creating front.mp4...")
    images_dir = extract_dir / "images"
    image_files = sorted(images_dir.glob("*.jpg"))

    if not image_files:
        raise RuntimeError(f"No images found: {images_dir}")

    # Use ffmpeg to create video
    front_mp4 = output_dir / "front.mp4"

    # Create a list file for ffmpeg
    list_file = extract_dir / "images_list.txt"
    with open(list_file, "w") as f:
        for img in image_files:
            abs_path = img.resolve()
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {1.0/sample_fps}\n")
        # Add last frame again
        if image_files:
            abs_path = image_files[-1].resolve()
            f.write(f"file '{abs_path}'\n")

    # Run ffmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-vf", f"fps={sample_fps}",
        "-pix_fmt", "yuv420p",
        str(front_mp4),
    ]

    print(f"Running ffmpeg...")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: ffmpeg failed, but continuing:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    else:
        print(f"Created: {front_mp4}")

    # Step 4: Generate BEV using process_sidewalk_dataset.py
    if not generate_bev:
        print("\n[4/5] Skipping BEV generation (--skip-bev)")
        print(f"\n[5/5] Done!")
        print(f"Output directory: {output_dir}")
        print(f"Files created:")
        for f in sorted(output_dir.iterdir()):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
        return output_dir

    print("\n[4/5] Generating BEV video...")

    # Use a temporary output dir for process_sidewalk_dataset.py
    bev_temp_dir = temp_extract_dir / f"{mcap_path.stem}_bev_out"

    bev_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "process_sidewalk_dataset.py"),
        "--sidewalk-root", str(output_chunk_dir),
        "--output-root", str(bev_temp_dir),
        "--sample-fps", str(sample_fps),
        "--ride-filter", mcap_path.stem,
    ]

    print(f"Running: {' '.join(bev_cmd)}")
    result = subprocess.run(bev_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: BEV generation had errors:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    else:
        print(f"BEV generation complete")

    print(f"\n[5/5] Done!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Process realword MCAP files into sidewalk chunk_0 format"
    )
    parser.add_argument(
        "--mcap-dir",
        type=Path,
        default="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/realword",
        help="Directory containing MCAP files",
    )
    parser.add_argument(
        "--output-chunk-dir",
        type=Path,
        default="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/realword_processed",
        help="Output directory for processed chunks",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default="/tmp/realword_mcap_temp",
        help="Temporary directory for extraction",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=20.0,
        help="Sampling rate for frames",
    )
    parser.add_argument(
        "--image-topic",
        default="/camera_x001/image_raw/compressed",
        help="Image topic",
    )
    parser.add_argument(
        "--gps-topic",
        default="/baton/gnss/spp_fix",
        help="GPS topic",
    )
    parser.add_argument(
        "--heading-topic",
        default="/baton/gnss/heading",
        help="Heading topic",
    )
    parser.add_argument(
        "--imu-topic",
        default="none",
        help="IMU topic",
    )
    parser.add_argument(
        "--pose-topic",
        default="none",
        help="Pose topic",
    )
    parser.add_argument(
        "--skip-bev",
        action="store_true",
        help="Only decode MCAP into SideWalk clip files; do not generate BEV maps/videos.",
    )
    args = parser.parse_args()

    # Find MCAP files
    mcap_files = sorted(args.mcap_dir.glob("*.mcap"))
    print(f"Found {len(mcap_files)} MCAP files")

    # Process each MCAP
    for mcap_path in mcap_files:
        try:
            # Check if already processed
            output_dir = args.output_chunk_dir / mcap_path.stem
            bev_video = output_dir / "bev_video.mp4"
            if bev_video.exists():
                print(f"\nSkipping already processed: {mcap_path.name}")
                continue

            # Check if npy files exists, if not generate
            frame_gps = output_dir / "frame_gps.npy"
            if not frame_gps.exists():
                print(f"\nProcessing npy/mp4 for: {mcap_path.name}")
                mcap_to_chunk_format(
                    mcap_path=mcap_path,
                    output_chunk_dir=args.output_chunk_dir,
                    temp_extract_dir=args.temp_dir,
                    sample_fps=args.sample_fps,
                    image_topic=args.image_topic,
                    gps_topic=args.gps_topic,
                    heading_topic=args.heading_topic,
                    imu_topic=args.imu_topic,
                    pose_topic=args.pose_topic,
                    generate_bev=not args.skip_bev,
                )
            else:
                print(f"\nnpy files already exist for: {mcap_path.name}")

        except Exception as e:
            print(f"\nError processing {mcap_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("All done!")
    print(f"Output directory: {args.output_chunk_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
