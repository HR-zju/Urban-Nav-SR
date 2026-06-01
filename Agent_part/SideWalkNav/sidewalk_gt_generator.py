#!/usr/bin/env python3
"""
Standalone FPV Trajectory Projection Tool

This script projects future GPS trajectory points onto current FPV image.

Usage:
    # Single frame
    python sidewalk_gt_generator.py --clip-dir "/path/to/clip" --frame-idx 100 --output traj.jpg

    # Batch process entire clip and save visualization mp4
    python sidewalk_gt_generator.py --clip-dir "/path/to/clip" --batch --output-dir "./output"

    # Process entire chunk (all clips)
    python sidewalk_gt_generator.py --chunk-dir "/path/to/chunk" --batch --output-dir "./output"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Lazy import for video writer
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class CameraParams:
    """Camera intrinsic and geometric parameters."""
    K: np.ndarray                    # 3x3 camera intrinsic matrix
    image_width: int                # image width in pixels
    image_height: int               # image height in pixels
    camera_height_meters: float = 0.41  # camera height from ground in meters

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraParams":
        return cls(
            K=np.array(data["K"], dtype=np.float64),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            camera_height_meters=float(data.get("camera_height_meters", 0.3)),
        )


def quat_to_yaw(quat: np.ndarray) -> float:
    """Convert quaternion to yaw angle."""
    w, x, y, z = quat
    return 2 * np.arctan2(z, w)


def project_map_to_fpv_single(
    current_pos: np.ndarray,
    current_quat: np.ndarray,
    target_pos: np.ndarray,
    camera: CameraParams,
) -> Tuple[float, float]:
    """
    Project a single map coordinate point to FPV image pixel coordinates.

    Args:
        current_pos: Current robot position (x, y) in meters
        current_quat: Current robot orientation quaternion (w, x, y, z)
        target_pos: Target point position (x, y) in meters
        camera: Camera parameters

    Returns:
        (u, v): Image pixel coordinates
    """
    yaw = quat_to_yaw(current_quat)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    # Compute relative position
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]

    # Rotate to robot local coordinate (x=forward, y=left)
    x_robot = dx * cos_yaw + dy * sin_yaw
    y_robot = -dx * sin_yaw + dy * cos_yaw

    # Transform to camera coordinate (z=forward, x=right, y=down)
    z_cam = x_robot
    x_cam = -y_robot
    y_cam = -camera.camera_height_meters

    # Point is too close to camera, return bottom center
    if z_cam <= 0.01:
        return camera.image_width / 2.0, camera.image_height * 0.9

    # Camera intrinsic projection
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]

    u = fx * x_cam / z_cam + cx
    v = cy - fy * y_cam / z_cam

    # Special handling for points that would be too low in the image
    bottom_limit = camera.image_height * 0.95
    if v > bottom_limit:
        # For points that project too low, smoothly adjust them upwards
        v = bottom_limit - (v - bottom_limit) * 0.5
        # Make sure it doesn't go above the bottom center point
        v = max(v, camera.image_height * 0.85)

    # Clip to image bounds
    u = np.clip(u, 0.0, camera.image_width - 1.0)
    v = np.clip(v, 0.0, camera.image_height - 1.0)

    return float(u), float(v)


def generate_fpv_trajectory(
    positions: np.ndarray,
    orientations: np.ndarray,
    current_frame_idx: int,
    camera: CameraParams,
    num_trajectory_points: int = 16,
    num_future_steps: int = 50,
    first_point_at_bottom: bool = True,
) -> np.ndarray:
    """
    Generate future trajectory projection on FPV image.

    Args:
        positions: Position array for all frames (N, 2) in meters
        orientations: Quaternion array for all frames (N, 4)
        current_frame_idx: Current frame index
        camera: Camera parameters
        num_trajectory_points: Number of output trajectory points
        num_future_steps: Number of future steps to predict
        first_point_at_bottom: Whether to fix first point at bottom center

    Returns:
        trajectory: (num_trajectory_points, 2) pixel coordinate array
    """
    num_frames = len(positions)
    end_frame_idx = min(current_frame_idx + num_future_steps, num_frames - 1)
    # Start sampling from a reasonable distance ahead
    start_offset = 10
    start_frame_idx = min(current_frame_idx + start_offset, end_frame_idx)

    # Uniformly sample within future frame range
    traj_frame_indices = np.linspace(
        start_frame_idx, end_frame_idx, num_trajectory_points - 1, dtype=int
    )

    current_pos = positions[current_frame_idx, :2]
    current_quat = orientations[current_frame_idx]

    # Project point by point
    trajectory = []
    for frame_idx in traj_frame_indices:
        target_pos = positions[frame_idx, :2]
        u, v = project_map_to_fpv_single(
            current_pos, current_quat, target_pos, camera
        )
        trajectory.append([u, v])

    trajectory = np.array(trajectory, dtype=np.float32)

    # Add the bottom center point as the first point
    first_point = np.array([[camera.image_width / 2.0, camera.image_height * 0.9]], dtype=np.float32)
    trajectory = np.vstack([first_point, trajectory])

    return trajectory


def normalize_trajectory(trajectory_px: np.ndarray, camera: CameraParams) -> np.ndarray:
    """Normalize pixel coordinates to [0, 1] range."""
    traj = trajectory_px.copy()
    traj[:, 0] /= camera.image_width
    traj[:, 1] /= camera.image_height
    return traj


def draw_trajectory_on_image(
    image: Any,
    trajectory_px: np.ndarray,
    point_radius: int = 8,
    line_width: int = 3,
) -> Any:
    """Draw trajectory on FPV image with color gradient from green (near) to red (far)."""
    if not HAS_PIL:
        raise ImportError("Install Pillow: pip install Pillow")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    traj = trajectory_px.astype(int)

    # Draw line (cyan)
    for i in range(len(traj) - 1):
        draw.line([tuple(traj[i]), tuple(traj[i+1])], fill=(0,255,255), width=line_width)

    # Draw points (green -> red gradient)
    for i, (x, y) in enumerate(traj):
        alpha = i / max(len(traj) - 1, 1)
        r = int(0 * (1 - alpha) + 255 * alpha)
        g = int(255 * (1 - alpha) + 0 * alpha)
        b = 0
        draw.ellipse([x-point_radius, y-point_radius, x+point_radius, y+point_radius],
                    fill=(r, g, b), outline=(255,255,255), width=2)

    return image


def load_sidewalk_clip_data(clip_dir: Path) -> Dict[str, Any]:
    """
    Load a sidewalk clip data.

    Directory structure:
        chunk_dir/
            camera_params.json
            clip_000000/
                frame_positions.npy
                frame_orientations.npy
                front.mp4 or images/
    """
    clip_dir = Path(clip_dir)

    # Load position and orientation
    positions = np.load(clip_dir / "frame_positions.npy")
    orientations = np.load(clip_dir / "frame_orientations.npy")

    # Load camera parameters from parent directory
    chunk_dir = clip_dir.parent
    with open(chunk_dir / "camera_params.json", "r", encoding="utf-8") as f:
        camera_params_dict = json.load(f)
    camera = CameraParams.from_dict(camera_params_dict)

    return {
        "positions": positions,
        "orientations": orientations,
        "camera": camera,
        "clip_dir": clip_dir,
    }


def load_fpv_image(clip_dir: Path, frame_idx: int) -> Optional[Image.Image]:
    """
    Load FPV image for specified frame.

    First try to read from mp4, fallback to images directory.
    Note: In dataset images start from 000001.jpg, frame_idx starts from 0.
    """
    mp4_path = clip_dir / "front.mp4"
    if mp4_path.exists():
        try:
            import decord
            vr = decord.VideoReader(str(mp4_path))
            if frame_idx < len(vr):
                return Image.fromarray(vr[frame_idx].asnumpy())
        except ImportError:
            pass

    fpv_dir = clip_dir / "images"
    if fpv_dir.exists():
        # In dataset images start from 000001.jpg
        for pattern in [f"{frame_idx+1:06d}.jpg", f"{frame_idx:06d}.jpg", f"{frame_idx+1}.jpg", f"{frame_idx}.jpg"]:
            img_path = fpv_dir / pattern
            if img_path.exists():
                return Image.open(img_path)

    return None


def process_clip_batch(
    clip_dir: Path,
    output_dir: Optional[Path] = None,
    num_trajectory_points: int = 16,
    num_future_steps: int = 50,
    point_radius: int = 8,
    line_width: int = 3,
    save_mp4: bool = True,
    fps: int = 10,
) -> Dict[str, Any]:
    """
    Batch process all frames of a clip, generate trajectory GT and optionally save visualization mp4.

    Args:
        clip_dir: Path to clip directory
        output_dir: Output directory, if None save to clip_dir itself
        num_trajectory_points: Number of trajectory points
        num_future_steps: Number of future steps
        point_radius: Visualization point radius
        line_width: Visualization line width
        save_mp4: Whether to save visualization mp4
        fps: mp4 frame rate

    Returns:
        Processing statistics
    """
    data = load_sidewalk_clip_data(clip_dir)
    positions, orientations, camera = data["positions"], data["orientations"], data["camera"]
    num_frames = len(positions)

    clip_name = clip_dir.name
    # Save to clip directory by default
    clip_output_dir = clip_dir if output_dir is None else output_dir / clip_name
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing clip: {clip_name} ({num_frames} frames)")

    # Store all trajectories
    all_trajectories = []
    all_trajectories_norm = []
    viz_frames = []

    # Process frame by frame
    for frame_idx in range(num_frames):
        trajectory_px = generate_fpv_trajectory(
            positions, orientations, frame_idx, camera,
            num_trajectory_points, num_future_steps
        )
        trajectory_norm = normalize_trajectory(trajectory_px, camera)

        all_trajectories.append(trajectory_px)
        all_trajectories_norm.append(trajectory_norm)

        # Generate visualization frame
        if save_mp4 and HAS_PIL and HAS_CV2:
            fpv_image = load_fpv_image(clip_dir, frame_idx)
            if fpv_image:
                viz = draw_trajectory_on_image(fpv_image, trajectory_px,
                                              point_radius, line_width)
                viz_frames.append(np.array(viz))

        # Progress
        if (frame_idx + 1) % 50 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames")

    # Save all trajectory data
    trajectories_npy = np.array(all_trajectories_norm, dtype=np.float32)
    np.save(clip_output_dir / "trajectories.npy", trajectories_npy)

    # Save metadata
    metadata = {
        "clip_name": clip_name,
        "num_frames": num_frames,
        "num_trajectory_points": num_trajectory_points,
        "num_future_steps": num_future_steps,
        "image_width": camera.image_width,
        "image_height": camera.image_height,
    }
    with open(clip_output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save mp4
    mp4_path = None
    if save_mp4 and viz_frames and HAS_CV2:
        mp4_path = clip_output_dir / "trajectory_viz.mp4"
        height, width = viz_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))

        for frame in viz_frames:
            # RGB -> BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"  Saved visualization MP4: {mp4_path}")

    return {
        "clip_name": clip_name,
        "num_frames": num_frames,
        "trajectories_npy": clip_output_dir / "trajectories.npy",
        "metadata_json": clip_output_dir / "metadata.json",
        "viz_mp4": mp4_path,
    }


def process_chunk(
    chunk_dir: Path,
    output_dir: Optional[Path] = None,
    num_trajectory_points: int = 16,
    num_future_steps: int = 50,
    point_radius: int = 8,
    line_width: int = 3,
    save_mp4: bool = True,
    fps: int = 10,
) -> List[Dict[str, Any]]:
    """
    Batch process all clips in a chunk.

    Args:
        chunk_dir: Path to chunk directory
        output_dir: Output directory, if None save to each clip's own directory
        ... other args same as process_clip_batch ...

    Returns:
        List of processing statistics for each clip
    """
    chunk_dir = Path(chunk_dir)
    chunk_name = chunk_dir.name

    # Find all clip directories
    clip_dirs = sorted([d for d in chunk_dir.iterdir() if d.is_dir() and (d / "frame_positions.npy").exists()])

    print(f"Processing chunk: {chunk_name}")
    print(f"Found {len(clip_dirs)} clips")

    results = []
    for i, clip_dir in enumerate(clip_dirs):
        print(f"\n[{i+1}/{len(clip_dirs)}]", end=" ")
        # If output_dir is provided, use it; otherwise pass None to save in clip_dir
        clip_output_dir = output_dir / chunk_name if output_dir is not None else None
        result = process_clip_batch(
            clip_dir,
            clip_output_dir,
            num_trajectory_points,
            num_future_steps,
            point_radius,
            line_width,
            save_mp4,
            fps,
        )
        results.append(result)

    return results


def main():
    DEFAULT_DATA_ROOT = "/media/ubuntu/data1/SR-Nav Dataset/sidewalk"

    parser = argparse.ArgumentParser(description="Standalone FPV Trajectory Projection")

    # Input source (three options, prefer chunk-name)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--chunk-name", type=str, default="chunk_0", help=f"Chunk name under {DEFAULT_DATA_ROOT} (default: chunk_0)")
    input_group.add_argument("--clip-dir", type=str, help="Path to single clip directory")
    input_group.add_argument("--chunk-dir", type=str, help="Path to full chunk directory")

    # Mode selection
    parser.add_argument("--batch", action="store_true", default=True, help="Enable batch mode (process all frames, default: True)")

    # Single frame mode params
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index for single frame mode")
    parser.add_argument("--output", type=str, default=None, help="Output path for single frame mode")

    # Batch mode params
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for batch mode, default to save in each clip's own directory")
    parser.add_argument("--no-mp4", action="store_true", help="Don't save visualization MP4")
    parser.add_argument("--fps", type=int, default=10, help="MP4 frame rate")

    # Trajectory params
    parser.add_argument("--num-points", type=int, default=16, help="Number of trajectory points")
    parser.add_argument("--num-future", type=int, default=50, help="Number of future steps to predict")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization in single frame mode")
    parser.add_argument("--point-radius", type=int, default=8, help="Point radius for visualization")
    parser.add_argument("--line-width", type=int, default=3, help="Line width for visualization")

    args = parser.parse_args()

    # Resolve input path
    if args.clip_dir is None and args.chunk_dir is None:
        # Use default chunk name
        args.chunk_dir = str(Path(DEFAULT_DATA_ROOT) / args.chunk_name)

    # Batch mode
    if args.batch:
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        if args.chunk_dir:
            # Process entire chunk
            process_chunk(
                Path(args.chunk_dir),
                output_dir,
                args.num_points,
                args.num_future,
                args.point_radius,
                args.line_width,
                not args.no_mp4,
                args.fps,
            )
        elif args.clip_dir:
            # Process single clip
            process_clip_batch(
                Path(args.clip_dir),
                output_dir,
                args.num_points,
                args.num_future,
                args.point_radius,
                args.line_width,
                not args.no_mp4,
                args.fps,
            )
        print("\nBatch processing complete!")
        return

    # Single frame mode
    if not args.clip_dir:
        print("Error: --clip-dir is required for single frame mode")
        sys.exit(1)

    data = load_sidewalk_clip_data(Path(args.clip_dir))
    positions, orientations, camera = data["positions"], data["orientations"], data["camera"]

    print(f"Clip: {data['clip_dir'].name}")
    print(f"Total frames: {len(positions)}")
    print(f"Image size: {camera.image_width} x {camera.image_height}")

    trajectory_px = generate_fpv_trajectory(
        positions, orientations, args.frame_idx, camera,
        args.num_points, args.num_future
    )

    trajectory_norm = normalize_trajectory(trajectory_px, camera)

    print(f"\nTrajectory shape: {trajectory_px.shape}")
    print(f"Start (pixel): ({trajectory_px[0,0]:.1f}, {trajectory_px[0,1]:.1f})")
    print(f"Start (normalized): ({trajectory_norm[0,0]:.4f}, {trajectory_norm[0,1]:.4f})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(out_path.with_suffix(".npy"), trajectory_norm)
        print(f"\nSaved trajectory to: {out_path.with_suffix('.npy')}")

        metadata = {
            "frame_idx": args.frame_idx,
            "image_width": camera.image_width,
            "image_height": camera.image_height,
            "trajectory_pixels": trajectory_px.tolist(),
            "trajectory_normalized": trajectory_norm.tolist(),
        }
        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if not args.no_viz and HAS_PIL:
            fpv_image = load_fpv_image(data["clip_dir"], args.frame_idx)
            if fpv_image:
                viz = draw_trajectory_on_image(fpv_image, trajectory_px,
                                              args.point_radius, args.line_width)
                viz.save(out_path.with_suffix(".jpg"), quality=95)
                print(f"Saved visualization to: {out_path.with_suffix('.jpg')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
