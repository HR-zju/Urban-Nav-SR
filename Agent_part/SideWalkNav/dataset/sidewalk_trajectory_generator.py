"""
SideWalk Trajectory GT Generator - FPV Projection Module.

This module encapsulates the core logic for generating FPV pixel trajectory GT
from map coordinates. Decoupled from dataset class for reusability.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np


@dataclass
class CameraParams:
    """Camera parameters for FPV projection."""
    K: np.ndarray  # (3, 3) Camera intrinsics
    D: np.ndarray  # (4,) Distortion coefficients
    camera_height_meters: float
    image_width: int
    image_height: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraParams":
        """Create from dictionary."""
        return cls(
            K=np.array(data["K"]),
            D=np.array(data["D"]),
            camera_height_meters=float(data["camera_height_meters"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "CameraParams":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""
    num_trajectory_points: int = 16
    num_future_steps: int = 50
    first_point_at_bottom_center: bool = True
    normalize_output: bool = True


class SidewalkTrajectoryGenerator:
    """
    Generates FPV pixel trajectory GT from map coordinates.

    Core features:
    - Quaternion to yaw conversion
    - Map → Robot local coordinates transformation
    - Robot → Camera coordinates transformation
    - Projection to FPV pixel coordinates
    - Trajectory sampling and normalization
    """

    def __init__(
        self,
        camera_params: Optional[CameraParams] = None,
        config: Optional[TrajectoryConfig] = None,
    ):
        """
        Initialize trajectory generator.

        Args:
            camera_params: Camera intrinsic parameters. If None, dummy fallback
                mode is used.
            config: Trajectory generation configuration.
        """
        self.camera_params = camera_params
        self.config = config or TrajectoryConfig()

    def set_camera_params(self, camera_params: CameraParams) -> None:
        """Set or update camera parameters."""
        self.camera_params = camera_params

    @staticmethod
    def quat_to_yaw(quat: np.ndarray) -> float:
        """
        Convert quaternion (w, x, y, z) to yaw angle (radians).

        Args:
            quat: (4,) quaternion [w, x, y, z]

        Returns:
            Yaw angle in radians
        """
        w, x, y, z = quat
        return 2 * np.arctan2(z, w)

    def project_map_to_fpv(
        self,
        curr_pos: np.ndarray,
        curr_quat: np.ndarray,
        target_pos: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Project a single map coordinate to FPV pixel coordinates.

        Args:
            curr_pos: (2,) Current position [x, y] in map space
            curr_quat: (4,) Current orientation [w, x, y, z]
            target_pos: (2,) Target position [x, y] in map space

        Returns:
            (u, v) pixel coordinates, NOT normalized
        """
        if self.camera_params is None:
            raise RuntimeError("Camera parameters not set for projection")

        cp = self.camera_params
        yaw = self.quat_to_yaw(curr_quat)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Step 1: Map → Robot local coordinates
        dx = target_pos[0] - curr_pos[0]
        dy = target_pos[1] - curr_pos[1]
        x_robot = dx * cos_yaw + dy * sin_yaw
        y_robot = -dx * sin_yaw + dy * cos_yaw

        # Step 2: Robot → Camera coordinates
        # Camera coordinate system (OpenCV): +x right, +y down, +z forward
        z_cam = x_robot
        x_cam = -y_robot
        y_cam = -cp.camera_height_meters

        # Handle point behind camera
        if z_cam <= 0.01:
            return cp.image_width / 2.0, cp.image_height * 0.9

        # Step 3: Projection to FPV pixel coordinates
        fx, fy = cp.K[0, 0], cp.K[1, 1]
        cx, cy = cp.K[0, 2], cp.K[1, 2]

        u = fx * x_cam / z_cam + cx
        v = cy - fy * y_cam / z_cam

        # Clamp to image bounds
        u = np.clip(u, 0.0, cp.image_width - 1.0)
        v = np.clip(v, 0.0, cp.image_height - 1.0)

        return float(u), float(v)

    def generate_trajectory(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        current_frame_idx: int,
    ) -> np.ndarray:
        """
        Generate full FPV pixel trajectory GT.

        Args:
            positions: (N, 2) Position array [x, y] in map space
            orientations: (N, 4) Orientation array [w, x, y, z]
            current_frame_idx: Current frame index (start point)

        Returns:
            If normalize_output=True (default):
                (num_trajectory_points, 2) NORMALIZED coordinates [0, 1]
            Else:
                (num_trajectory_points, 2) Pixel coordinates
        """
        num_points = self.config.num_trajectory_points
        num_future = self.config.num_future_steps
        curr_pos = positions[current_frame_idx, :2]
        curr_quat = orientations[current_frame_idx]

        if self.camera_params is None:
            # Dummy fallback mode - no camera params
            dummy_traj = np.zeros((num_points, 2), dtype=np.float32)
            dummy_traj[:, 0] = 0.5  # center x
            dummy_traj[:, 1] = np.linspace(0.95, 0.3, num_points)
            return dummy_traj

        cp = self.camera_params

        # Sample trajectory frames (start from +1 to avoid x_robot=0 issue)
        end_frame_idx = min(current_frame_idx + num_future, len(positions) - 1)
        start_frame_idx = current_frame_idx + 1
        if start_frame_idx > end_frame_idx:
            start_frame_idx = current_frame_idx

        traj_frame_indices = np.linspace(start_frame_idx, end_frame_idx, num_points, dtype=int)

        # Project each point
        trajectory = []
        for frame_idx in traj_frame_indices:
            target_pos = positions[frame_idx, :2]
            u, v = self.project_map_to_fpv(curr_pos, curr_quat, target_pos)
            trajectory.append([u, v])

        trajectory = np.array(trajectory, dtype=np.float32)

        # Force first point at bottom center
        if self.config.first_point_at_bottom_center:
            trajectory[0, 0] = cp.image_width / 2.0
            trajectory[0, 1] = cp.image_height * 0.9

        # Normalize to [0, 1] if needed
        if self.config.normalize_output:
            trajectory[:, 0] /= cp.image_width
            trajectory[:, 1] /= cp.image_height

        return trajectory

    def generate_dummy_trajectory(self) -> np.ndarray:
        """
        Generate a dummy trajectory (straight from bottom center).

        Returns:
            (num_trajectory_points, 2) normalized coordinates
        """
        num_points = self.config.num_trajectory_points
        dummy_traj = np.zeros((num_points, 2), dtype=np.float32)
        dummy_traj[:, 0] = 0.5
        dummy_traj[:, 1] = np.linspace(0.95, 0.3, num_points)
        return dummy_traj

    def get_bottom_center_pixel(self) -> Tuple[float, float]:
        """Get bottom center pixel position (normalized)."""
        if self.camera_params is None:
            return 0.5, 0.9
        return 0.5, 0.9
