"""SideWalk Clip Data Structure and Loader

Encapsulates all data for a single clip and loading logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np

# Lazy import for decord
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

from .sidewalk_trajectory_generator import CameraParams


@dataclass
class SidewalkClip:
    """Encapsulates all data for a single clip, replacing dict usage.

    Provides type-safe data access.
    """
    clip_dir: Path
    positions: np.ndarray  # (N, 2)
    orientations: np.ndarray  # (N, 4)
    num_frames: int
    camera_params: Optional[CameraParams]
    bev_image_paths: List[Path]
    use_mp4: bool
    frame_times: Optional[np.ndarray] = None
    speeds: Optional[np.ndarray] = None
    video_reader: Optional[object] = None  # decord.VideoReader

    @property
    def bev_num_frames(self) -> int:
        """Get BEV image count."""
        return len(self.bev_image_paths)

    def close_video(self):
        """Clean up video resources."""
        if self.video_reader is not None:
            # decord doesn't need explicit closing, but keep the interface
            self.video_reader = None


class SidewalkClipLoader:
    """Responsible for clip loading and validation.

    Separates data loading logic from main Dataset.
    """

    @staticmethod
    def load(clip_dir: Path, use_mp4: bool = True) -> Optional[SidewalkClip]:
        """Load a single clip.

        Args:
            clip_dir: Clip directory path
            use_mp4: Whether to prefer MP4 video

        Returns:
            SidewalkClip object, or None if loading fails
        """
        # Validate required files
        positions_file = clip_dir / "frame_positions.npy"
        orientations_file = clip_dir / "frame_orientations.npy"
        if not positions_file.exists() or not orientations_file.exists():
            return None

        # Load basic trajectory data
        positions = np.load(positions_file)
        orientations = np.load(orientations_file)
        num_frames = len(positions)

        # Check image data
        bev_image_paths = SidewalkClipLoader._load_bev_paths(clip_dir)
        if not bev_image_paths:
            return None

        has_front_mp4 = (clip_dir / "front.mp4").exists()
        has_front_images = (clip_dir / "front_images").exists()
        if not has_front_mp4 and not has_front_images:
            return None

        # Load camera parameters
        camera_params = SidewalkClipLoader._load_camera_params(clip_dir)

        # Load other optional data
        frame_times = SidewalkClipLoader._load_frame_times(clip_dir, num_frames)
        speeds = SidewalkClipLoader._load_speeds(clip_dir)

        # Create clip object
        clip = SidewalkClip(
            clip_dir=clip_dir,
            positions=positions,
            orientations=orientations,
            num_frames=num_frames,
            camera_params=camera_params,
            bev_image_paths=bev_image_paths,
            use_mp4=use_mp4 and has_front_mp4,
            frame_times=frame_times,
            speeds=speeds,
        )

        # Initialize video reader
        if clip.use_mp4 and DECORD_AVAILABLE:
            clip.video_reader = SidewalkClipLoader._init_video_reader(clip_dir)
            if clip.video_reader is None:
                clip.use_mp4 = False

        return clip

    @staticmethod
    def _load_bev_paths(clip_dir: Path) -> List[Path]:
        """Load BEV image path list.

        Args:
            clip_dir: Clip directory

        Returns:
            Ordered BEV image path list
        """
        bev_dir = clip_dir / "bev_images"
        if not bev_dir.exists():
            return []
        return sorted(bev_dir.glob("*_bev.jpg"))

    @staticmethod
    def _load_camera_params(clip_dir: Path) -> Optional[CameraParams]:
        """Load camera parameters.

        Tries clip directory first, then chunk directory.

        Args:
            clip_dir: Clip directory

        Returns:
            CameraParams object, or None if not found
        """
        for path in [clip_dir / "camera_params.json", clip_dir.parent / "camera_params.json"]:
            if path.exists():
                return CameraParams.from_json(path)
        return None

    @staticmethod
    def _load_frame_times(clip_dir: Path, num_frames: int) -> Optional[np.ndarray]:
        """Load frame timestamps.

        Args:
            clip_dir: Clip directory
            num_frames: Expected number of frames

        Returns:
            Frame timestamp array, or None if not found or mismatched
        """
        file = clip_dir / "frame_times.npy"
        if not file.exists():
            return None
        times = np.load(file)
        if len(times) != num_frames:
            return None
        return times

    @staticmethod
    def _load_speeds(clip_dir: Path) -> Optional[np.ndarray]:
        """Load speed data.

        Args:
            clip_dir: Clip directory

        Returns:
            Speed array, or None if not found
        """
        file = clip_dir / "frame_speeds.npy"
        if file.exists():
            return np.load(file)
        return None

    @staticmethod
    def _init_video_reader(clip_dir: Path) -> Optional[object]:
        """Initialize decord video reader.

        Args:
            clip_dir: Clip directory

        Returns:
            decord.VideoReader object, or None on failure
        """
        if not DECORD_AVAILABLE:
            return None
        try:
            return decord.VideoReader(str(clip_dir / "front.mp4"))
        except Exception:
            return None
