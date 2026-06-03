#!/usr/bin/env python3
"""Process chunk_26, 27, 28, 29 into sidewalk format.

This script:
1. Extracts data from MCAP using mcap_to_sidewalk_inputs.py
2. Converts to sidewalk format (frame_*.npy + front.mp4)
3. Optionally generates BEV video using process_sidewalk_dataset.py
4. Optionally generates Phase2/WalkGPT annotations using run_sam3_som_phase.py
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Any, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
MODEL_PROCESSING_DIR = TOOLS_DIR / "model_processing"


def _load_json_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _should_use_pose_ekf_from_meta(meta: Optional[dict[str, Any]]) -> bool:
    if not isinstance(meta, dict):
        return True
    quality = meta.get("pose_quality", {})
    if not isinstance(quality, dict):
        return True
    if not bool(quality.get("overall_pose_quality_valid", False)):
        return True
    return False


def process_single_mcap(
    args_tuple: tuple,
) -> tuple[bool, str, Optional[str]]:
    """Process a single MCAP file - wrapper for parallel processing."""
    (mcap_path, output_dir, temp_extract_dir, sample_fps,
     image_topic, gps_topic, heading_topic, imu_topic, pose_topic,
     generate_bev, bev_output_root, generate_phase2, generate_walkgpt,
     checkpoint_path, sam3_env_name, gpus, skip_existing,
    use_pose_ekf, ekf_q_pos, ekf_q_theta, ekf_r_xy, ekf_r_theta, force_reextract) = args_tuple

    try:
        mcap_to_sidewalk_format(
            mcap_path=mcap_path,
            output_dir=output_dir,
            temp_extract_dir=temp_extract_dir,
            sample_fps=sample_fps,
            image_topic=image_topic,
            gps_topic=gps_topic,
            heading_topic=heading_topic,
            imu_topic=imu_topic,
            pose_topic=pose_topic,
            generate_bev=generate_bev,
            bev_output_root=bev_output_root,
            quiet=True,
            use_pose_ekf=use_pose_ekf,
            ekf_q_pos=ekf_q_pos,
            ekf_q_theta=ekf_q_theta,
            ekf_r_xy=ekf_r_xy,
            ekf_r_theta=ekf_r_theta,
            force_reextract=force_reextract,
        )

        # Generate Phase2 annotations if needed
        if generate_phase2:
            phase2_cmd = [
                sys.executable,
                str(MODEL_PROCESSING_DIR / "run_sam3_som_phase.py"),
                "--clip_dir", str(output_dir),
                "--run_sam3",
                "--sam3_env_name", sam3_env_name,
                "--gpus", gpus,
                "--min_score", "0.3",
                "--sample_fps", str(sample_fps),
                "--disable_route_support",
                "--generate_video",
                "--cleanup_temp",
            ]
            if checkpoint_path:
                phase2_cmd.extend(["--checkpoint_path", str(checkpoint_path)])
            result = subprocess.run(phase2_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return (False, mcap_path.name, f"Phase2 failed: {result.stderr}")

        return (True, mcap_path.name, None)
    except Exception as e:
        import traceback
        return (False, mcap_path.name, f"{str(e)}\n{traceback.format_exc()}")


def mcap_to_sidewalk_format(
    mcap_path: Path,
    output_dir: Path,
    temp_extract_dir: Path,
    sample_fps: float = 20.0,
    image_topic: str = "auto",
    gps_topic: str = "auto",
    heading_topic: str = "auto",
    imu_topic: str = "auto",
    pose_topic: str = "auto",
    generate_bev: bool = False,
    bev_output_root: Optional[Path] = None,
    use_pose_ekf: bool = False,
    ekf_q_pos: float = 1e-3,
    ekf_q_theta: float = 1e-2,
    ekf_r_xy: float = 9.0,
    ekf_r_theta: float = 0.5,
    force_reextract: bool = False,
    quiet: bool = False,
):
    """Convert a single MCAP file to sidewalk format."""

    if not quiet:
        print(f"\n{'='*80}")
        print(f"Processing: {mcap_path.name}")
        print(f"{'='*80}")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    frame_gps = output_dir / "frame_gps.npy"
    front_mp4 = output_dir / "front.mp4"
    bev_video = output_dir / "bev_video.mp4"

    if frame_gps.exists() and front_mp4.exists() and not force_reextract:
        existing_meta = _load_json_file(output_dir / "mcap_extract_meta.json")
        effective_use_pose_ekf = bool(use_pose_ekf or _should_use_pose_ekf_from_meta(existing_meta))
        if generate_bev and bev_output_root:
            print(f"Base sidewalk files already exist; regenerating BEV only: {output_dir.name}")
            bev_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "process_sidewalk_dataset.py"),
                "--sidewalk-root", str(output_dir),
                "--output-root", str(output_dir),
                "--sample-fps", str(sample_fps),
            ]
            if effective_use_pose_ekf:
                bev_cmd.append("--use-pose-ekf")
                bev_cmd.extend(["--ekf-q-pos", str(ekf_q_pos)])
                bev_cmd.extend(["--ekf-q-theta", str(ekf_q_theta)])
                bev_cmd.extend(["--ekf-r-xy", str(ekf_r_xy)])
                bev_cmd.extend(["--ekf-r-theta", str(ekf_r_theta)])
            print(f"Running: {' '.join(bev_cmd)}")
            result = subprocess.run(bev_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: BEV generation had errors:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            else:
                print(f"BEV generation complete")
            return output_dir

        print(f"Skipping - already converted: {output_dir.name}")
        return output_dir

    if force_reextract and (frame_gps.exists() or front_mp4.exists() or (output_dir / "trajectory.json").exists()):
        print(f"Force re-extract enabled; refreshing extracted files for: {output_dir.name}")

    # Step 1: Extract from MCAP
    if not quiet:
        print("\n[1/4] Extracting data from MCAP...")
    extract_dir = temp_extract_dir / mcap_path.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

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

    if not quiet:
        print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error extracting from MCAP: {mcap_path.name}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Failed to extract MCAP: {mcap_path}")

    # Load trajectory
    trajectory_path = extract_dir / "trajectory.json"
    if not trajectory_path.exists():
        raise RuntimeError(f"trajectory.json not found: {trajectory_path}")

    extract_meta_path = extract_dir / "mcap_extract_meta.json"
    extract_meta = _load_json_file(extract_meta_path)

    trajectory = json.loads(trajectory_path.read_text())
    if not quiet:
        print(f"Extracted {len(trajectory)} frames")

    # Keep extracted images alongside the chunk outputs so the dataset is self-contained.
    output_images_dir = output_dir / "images"
    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)
    shutil.copytree(extract_dir / "images", output_images_dir)
    for frame in trajectory:
        image_path = Path(frame["image_path"])
        frame["image_path"] = str(output_images_dir / image_path.name)

    # Keep extractor meta alongside the normalized clip outputs so downstream steps can read it.
    if extract_meta_path.exists():
        shutil.copy2(extract_meta_path, output_dir / extract_meta_path.name)

    normalized_trajectory_path = output_dir / "trajectory.json"
    normalized_trajectory_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=True), encoding="utf-8")

    images_dir = output_images_dir
    image_files = sorted(images_dir.glob("*.jpg"))

    # Step 2: Build npy files
    if not quiet:
        print("\n[2/4] Creating npy files...")

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

    # frame_orientations.npy (float64, shape [N, 4] - x, y, z, w quaternion)
    frame_orientations = np.zeros((num_frames, 4), dtype=np.float64)
    for i, f in enumerate(trajectory):
        if "roll_deg" in f and "pitch_deg" in f and "attitude_yaw_deg" in f:
            # Convert euler back to quaternion (approximate, just for compatibility)
            roll = np.radians(f["roll_deg"])
            pitch = np.radians(f["pitch_deg"])
            yaw = np.radians(f["attitude_yaw_deg"])
            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            frame_orientations[i, 0] = qx
            frame_orientations[i, 1] = qy
            frame_orientations[i, 2] = qz
            frame_orientations[i, 3] = qw
    np.save(output_dir / "frame_orientations.npy", frame_orientations)

    # frame_positions.npy (float64, shape [N, 3])
    frame_positions = np.zeros((num_frames, 3), dtype=np.float64)
    np.save(output_dir / "frame_positions.npy", frame_positions)

    # frame_angular_velocities.npy (float64, shape [N, 3])
    frame_angular_velocities = np.zeros((num_frames, 3), dtype=np.float64)
    extracted_imu_motion = extract_dir / "frame_angular_velocities.npy"
    if extracted_imu_motion.exists():
        imu_motion = np.load(extracted_imu_motion)
        # Support two formats from extractor:
        # - [timestamp, omega_x, omega_y, omega_z] (shape [N,4])
        # - [omega_x, omega_y, omega_z] (shape [N,3])
        if imu_motion.ndim == 2 and imu_motion.shape[1] >= 4:
            imu_times = imu_motion[:, 0].astype(np.int64)
            imu_omega = imu_motion[:, 1:4]
            for i, f in enumerate(trajectory):
                timestamp = int(f["timestamp"])
                nearest = int(np.argmin(np.abs(imu_times - timestamp)))
                frame_angular_velocities[i, :] = imu_omega[nearest, :]
        elif imu_motion.ndim == 2 and imu_motion.shape[1] == 3:
            # No timestamps available: assume samples align by index. Use nearest index clamped.
            for i, _ in enumerate(trajectory):
                idx = min(i, imu_motion.shape[0] - 1)
                frame_angular_velocities[i, :] = imu_motion[idx, :]
    np.save(output_dir / "frame_angular_velocities.npy", frame_angular_velocities)

    # frame_linear_velocities.npy (float64, shape [N, 3])
    frame_linear_velocities = np.zeros((num_frames, 3), dtype=np.float64)
    np.save(output_dir / "frame_linear_velocities.npy", frame_linear_velocities)

    # frame_speeds.npy (float64, shape [N])
    frame_speeds = np.zeros(num_frames, dtype=np.float64)
    np.save(output_dir / "frame_speeds.npy", frame_speeds)

    if not quiet:
        print(f"Created npy files for {num_frames} frames")

    # Step 3: Create front.mp4 from images
    if not quiet:
        print("\n[3/4] Creating front.mp4...")
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

    if not quiet:
        print(f"Running ffmpeg...")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0 and not quiet:
        print(f"Warning: ffmpeg failed, but continuing:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    elif not quiet:
        print(f"Created: {front_mp4}")

    effective_use_pose_ekf = bool(use_pose_ekf or _should_use_pose_ekf_from_meta(extract_meta))
    if not quiet:
        if effective_use_pose_ekf:
            reason = "CLI enabled" if use_pose_ekf else "extract_meta pose_quality invalid/missing"
            print(f"Using EKF heading smoothing for BEV generation ({reason}); position remains GPS-smoothed")
        else:
            print("Using raw heading data for BEV generation (pose quality looks valid)")

    # Step 4: Generate BEV (optional)
    if generate_bev and bev_output_root:
        if not quiet:
            print("\n[4/4] Generating BEV video...")

        bev_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "process_sidewalk_dataset.py"),
            "--sidewalk-root", str(output_dir),
            "--output-root", str(output_dir),
            "--sample-fps", str(sample_fps),
        ]
        if effective_use_pose_ekf:
            bev_cmd.append("--use-pose-ekf")
            bev_cmd.extend(["--ekf-q-pos", str(ekf_q_pos)])
            bev_cmd.extend(["--ekf-q-theta", str(ekf_q_theta)])
            bev_cmd.extend(["--ekf-r-xy", str(ekf_r_xy)])
            bev_cmd.extend(["--ekf-r-theta", str(ekf_r_theta)])

        if not quiet:
            print(f"Running: {' '.join(bev_cmd)}")
        result = subprocess.run(bev_cmd, capture_output=True, text=True)
        if result.returncode != 0 and not quiet:
            print(f"Warning: BEV generation had errors:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        elif not quiet:
            print(f"BEV generation complete")

    if not quiet:
        print(f"\nDone!")
        print(f"Output directory: {output_dir}")
        print(f"Files created:")
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")

    return output_dir


def process_chunk(
    chunk_dir: Path,
    sidewalk_root: Path,
    temp_dir: Path,
    sample_fps: float = 20.0,
    generate_bev: bool = False,
    generate_phase2: bool = False,
    generate_walkgpt: bool = False,
    checkpoint_path: Optional[Path] = None,
    sam3_env_name: str = "sam3",
    gpus: str = "0",
    skip_existing: bool = True,
    num_workers: int = 4,
    use_pose_ekf: bool = False,
    ekf_q_pos: float = 1e-3,
    ekf_q_theta: float = 1e-2,
    ekf_r_xy: float = 9.0,
    ekf_r_theta: float = 0.5,
    force_reextract: bool = False,
):
    """Process all MCAP files in a chunk directory with parallel processing."""

    mcap_dir = chunk_dir / "mcap"
    if not mcap_dir.exists():
        print(f"Warning: mcap dir not found: {mcap_dir}")
        return

    mcap_files = sorted(mcap_dir.glob("*.mcap"))
    print(f"\n{'='*80}")
    print(f"Processing chunk: {chunk_dir.name}")
    print(f"Found {len(mcap_files)} MCAP files")
    print(f"Using {num_workers} parallel workers")
    print(f"{'='*80}")

    # Prepare task list
    tasks = []
    skipped_count = 0

    for mcap_path in mcap_files:
        output_dir = sidewalk_root / chunk_dir.name / mcap_path.stem

        # Check if already processed
        if skip_existing:
            frame_gps = output_dir / "frame_gps.npy"
            front_mp4 = output_dir / "front.mp4"
            if frame_gps.exists() and front_mp4.exists():
                # Check if we need to run Phase2 or BEV
                need_more = False
                bev_ready = (output_dir / "bev_video.mp4").exists()

                if generate_bev and not bev_ready:
                    need_more = True
                if generate_phase2:
                    fpv_seg_dir = output_dir / "fpv_segmentation"
                    som_dir = output_dir / "som"
                    phase2_core_ready = fpv_seg_dir.exists() and som_dir.exists()

                    if not phase2_core_ready:
                        need_more = True

                if not need_more:
                    skipped_count += 1
                    continue

        # Create a unique temp dir for each worker
        worker_temp_dir = temp_dir / f"worker_{os.getpid()}_{mcap_path.stem}"

        tasks.append((
            mcap_path, output_dir, worker_temp_dir, sample_fps,
            "auto", "auto", "auto", "auto", "auto",
            generate_bev, sidewalk_root / chunk_dir.name,
            generate_phase2, generate_walkgpt,
            checkpoint_path, sam3_env_name, gpus, skip_existing,
            use_pose_ekf, ekf_q_pos, ekf_q_theta, ekf_r_xy, ekf_r_theta, force_reextract
        ))

    print(f"Skipped {skipped_count} already processed files")
    print(f"Processing {len(tasks)} files with {num_workers} workers...\n")

    # Process in parallel
    success_count = skipped_count
    failed_files = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_mcap, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            mcap_path = task[0]
            try:
                success, filename, error_msg = future.result()
                if success:
                    success_count += 1
                    print(f"[{i}/{len(tasks)}] ✓ {filename}")
                else:
                    failed_files.append((filename, error_msg))
                    print(f"[{i}/{len(tasks)}] ✗ {filename} failed")
                    if error_msg:
                        print(f"    Error: {error_msg[:200]}...")
            except Exception as e:
                failed_files.append((mcap_path.name, str(e)))
                print(f"[{i}/{len(tasks)}] ✗ {mcap_path.name} exception: {e}")

    print(f"\nChunk {chunk_dir.name} complete: {success_count}/{len(mcap_files)} succeeded")
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for name, error in failed_files[:10]:
            print(f"  - {name}: {error[:100]}...")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Process chunk_26, 27, 28, 29 into sidewalk format"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="/media/ubuntu/data1/SR-Nav Dataset",
        help="Root directory of SR-Nav Dataset",
    )
    parser.add_argument(
        "--sidewalk-root",
        type=Path,
        default="/media/ubuntu/data1/SR-Nav Dataset/sidewalk",
        help="Output directory for sidewalk format",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default="/tmp/chunk_mcap_temp",
        help="Temporary directory for extraction",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        nargs="+",
        default=["26", "27", "28", "29"],
        help="Chunks to process (default: 26 27 28 29)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=20.0,
        help="Sampling rate for frames",
    )
    parser.add_argument(
        "--generate-bev",
        action="store_true",
        help="Generate BEV video",
    )
    parser.add_argument(
        "--generate-phase2",
        action="store_true",
        help="Generate Phase2 annotations",
    )
    parser.add_argument(
        "--generate-walkgpt",
        action="store_true",
        help="Generate WalkGPT annotations",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default="/media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt",
        help="Path to SAM3 checkpoint",
    )
    parser.add_argument(
        "--sam3-env-name",
        type=str,
        default="sam3",
        help="SAM3 conda environment name",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU IDs to use (comma separated)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already processed clips",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Rerun MCAP extraction even when frame_gps.npy/front.mp4 already exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel workers (default: half CPU count)",
    )
    parser.add_argument(
        "--use-pose-ekf",
        action="store_true",
        help="Use EKF fusion with IMU angular velocity and yaw/compass during BEV generation",
    )
    parser.add_argument("--ekf-q-pos", type=float, default=1e-3, help="EKF process noise strength for x/y")
    parser.add_argument("--ekf-q-theta", type=float, default=1e-2, help="EKF process noise strength for theta")
    parser.add_argument("--ekf-r-xy", type=float, default=9.0, help="EKF GPS observation variance")
    parser.add_argument("--ekf-r-theta", type=float, default=0.5, help="EKF yaw/compass observation variance")
    args = parser.parse_args()

    # Create output directories
    args.sidewalk_root.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    # Process each chunk
    for chunk_num in args.chunks:
        chunk_dir = args.dataset_root / f"chunk_{chunk_num}"
        if not chunk_dir.exists():
            print(f"Warning: chunk dir not found: {chunk_dir}")
            continue

        process_chunk(
            chunk_dir=chunk_dir,
            sidewalk_root=args.sidewalk_root,
            temp_dir=args.temp_dir,
            sample_fps=args.sample_fps,
            generate_bev=args.generate_bev,
            generate_phase2=args.generate_phase2,
            generate_walkgpt=args.generate_walkgpt,
            checkpoint_path=args.checkpoint_path,
            sam3_env_name=args.sam3_env_name,
            gpus=args.gpus,
            skip_existing=args.skip_existing,
            num_workers=args.num_workers,
            use_pose_ekf=args.use_pose_ekf,
            ekf_q_pos=args.ekf_q_pos,
            ekf_q_theta=args.ekf_q_theta,
            ekf_r_xy=args.ekf_r_xy,
            ekf_r_theta=args.ekf_r_theta,
            force_reextract=args.force_reextract,
        )

    print(f"\n{'='*80}")
    print("All done!")
    print(f"Output directory: {args.sidewalk_root}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
