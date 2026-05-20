#!/usr/bin/env python3
"""Process sidewalk dataset (npy + mp4) to generate BEV maps."""

import sys
import os
import math
import json
import argparse
import subprocess
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
from PIL import Image

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent))

from frodobot_bev_enhanced import (
    render_bev_map,
    MAP_PROVIDERS,
    ProcessOptions,
)


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
    d_lon = math.radians(lng2 - lng1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)

    y = math.sin(d_lon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(y, x))
    compass_heading = (bearing + 360.0) % 360.0
    return compass_heading


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
) -> list[dict[str, Any]]:
    """Process one sidewalk ride directory."""
    print(f"\n{'='*60}")
    print(f"Processing: {sidewalk_dir.name}")
    print(f"{'='*60}")

    # Load npy files
    gps_arr = np.load(sidewalk_dir / "frame_gps.npy")
    ori_arr = np.load(sidewalk_dir / "frame_orientations.npy")
    times_arr = np.load(sidewalk_dir / "frame_times.npy")

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
    for img_idx, (img_path, data_idx) in enumerate(zip(extracted_frames, selected_indices)):
        # GPS format looks like: [lat, lng, alt, ?, ?]
        lat = float(gps_arr[data_idx, 0])
        lng = float(gps_arr[data_idx, 1])

        # Calculate heading from GPS - use current point and next point (or previous if last)
        if img_idx < len(selected_indices) - 1:
            next_data_idx = selected_indices[img_idx + 1]
            next_lat = float(gps_arr[next_data_idx, 0])
            next_lng = float(gps_arr[next_data_idx, 1])
            heading_deg = _calculate_heading_from_gps(lat, lng, next_lat, next_lng)
        elif img_idx > 0:
            prev_data_idx = selected_indices[img_idx - 1]
            prev_lat = float(gps_arr[prev_data_idx, 0])
            prev_lng = float(gps_arr[prev_data_idx, 1])
            # Use previous heading (same direction)
            heading_deg = _calculate_heading_from_gps(prev_lat, prev_lng, lat, lng)
        else:
            # Only one point - default to 0
            heading_deg = 0.0

        # Still get roll/pitch from quaternion (though they're 0)
        qx = float(ori_arr[data_idx, 0])
        qy = float(ori_arr[data_idx, 1])
        qz = float(ori_arr[data_idx, 2])
        qw = float(ori_arr[data_idx, 3])
        roll_deg, pitch_deg, _ = _quaternion_to_euler_deg(x=qx, y=qy, z=qz, w=qw)

        # Rename image to be 000001.jpg style
        img_name = f"{img_idx+1:06d}.jpg"
        final_img_path = output_dir / "images" / img_name
        if img_path != final_img_path:
            img_path.rename(final_img_path)

        # Add to trajectory
        trajectory.append(
            {
                "timestamp": int(times_arr[data_idx]),
                "lat": lat,
                "lng": lng,
                "heading_deg": heading_deg,
                "image_path": str(final_img_path.resolve()),
                "attitude_source": "gps",
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
    global_nav_path = [(float(gps_arr[i, 0]), float(gps_arr[i, 1])) for i in selected_indices]

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
        help="Root of sidewalk dataset",
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
        default=1.0,
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
    args = parser.parse_args()

    # Find all ride directories
    sidewalk_root = args.sidewalk_root
    ride_dirs = sorted([d for d in sidewalk_root.iterdir() if d.is_dir()])

    if args.ride_filter:
        ride_dirs = [d for d in ride_dirs if args.ride_filter in d.name]

    print(f"Found {len(ride_dirs)} rides to process")

    all_samples = []
    for ride_dir in ride_dirs:
        output_dir = args.output_root / ride_dir.name
        try:
            samples = process_sidewalk_ride(
                ride_dir,
                output_dir,
                sample_fps=args.sample_fps,
                map_provider=args.map_provider,
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
