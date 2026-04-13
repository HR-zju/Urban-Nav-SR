#!/usr/bin/env python3
"""ROS2 RealSense ONNX runner with trajectory overlay and async video recording.

This script is based on `test_nav_flow_il_realsense.py` (RealSense live
inference node) and reuses the visualization and asynchronous video writer
patterns from `run_onnx_from_mp4.py`.

Key features:
- Subscribes to RealSense color compressed image and odometry.
- Maintains a sliding window of 21 frames for ONNX inference.
- Uses `project_waypoints_to_fisheye_image_with_polygon_new` to overlay
  predicted waypoints onto each incoming RGB frame.
- Caches the last successful waypoint prediction (`last_wp_np`) and reuses
  it for frames between inferences.
- When `--save_vis` is enabled, pushes rendered BGR frames into an
  `AsyncVideoWriter` for non-blocking MP4 recording.

This runner is designed to avoid blocking the control loop while still
producing a high-quality overlay video for later inspection.
"""

import argparse
import datetime
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
import torch
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray

import ast


# -----------------------------------------------------------------------------
# Utility: safe dynamic load of visualization function from vis_gt_single_gt.py
# -----------------------------------------------------------------------------


def load_function_from_file(filepath: str, func_name: str):
    """Extract imports + a single function from a file and load into a namespace.

    This avoids executing any top-level demo code in the source file.
    """

    with open(filepath, "r") as f:
        src = f.read()
    tree = ast.parse(src, filepath)

    new_nodes = []
    func_node = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            new_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node
            new_nodes.append(node)
            break

    if func_node is None:
        raise ImportError(f"Function {func_name} not found in {filepath}")

    mod = ast.Module(body=new_nodes, type_ignores=[])
    code = compile(ast.fix_missing_locations(mod), filepath, "exec")
    ns = {}
    exec(code, ns)
    return ns[func_name]


# Resolve path to vis_gt_single_gt.py from tools/ directory
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_VIS_FILE = os.path.join(_REPO_ROOT, "tools", "vis_gt_single_gt.py")
project_waypoints_to_fisheye_image_with_polygon_new = load_function_from_file(
    _VIS_FILE, "project_waypoints_to_fisheye_image_with_polygon_new"
)


# -----------------------------------------------------------------------------
# Async video writer (ported from run_onnx_from_mp4.py)
# -----------------------------------------------------------------------------


class AsyncVideoWriter:
    """Background VideoWriter with dynamic size check and bounded queue.

    This keeps disk I/O and heavy overlay work off the main loop. The
    main ROS callback should push raw BGR frames plus an optional
    waypoint array, and this worker thread will handle resizing to the
    target resolution and drawing trajectories before writing to disk.
    """

    def __init__(self, out_path: str, fps: float, fourcc: int, max_queue: int = 128):
        self.out_path = out_path
        self.fps = float(fps)
        self.fourcc = fourcc
        self.max_queue = max_queue

        # Queue of (frame_bgr, wp_np) tuples, where wp_np is an
        # optional (N, 2) array of waypoints in robot frame.
        self._queue = mp.Queue(maxsize=max_queue)
        self._stop_event = mp.Event()
        self._process = mp.Process(
            target=self._worker,
            args=(self._queue, self._stop_event, self.out_path, self.fps, self.fourcc),
            daemon=True,
        )
        self._queue_full_logged = False

    def start(self):
        self._process.start()

    def write(self, frame_bgr: np.ndarray, wp_np: Optional[np.ndarray]):
        """Queue a frame and optional waypoints for writing.

        This call is designed to be non-blocking from the main ROS
        callback. If the internal queue is full, the frame is dropped
        rather than blocking the control loop.
        """

        if self._stop_event.is_set():
            return
        try:
            frame_copy = frame_bgr.copy()
            wp_copy = None
            if wp_np is not None:
                wp_copy = np.array(wp_np, copy=True)
            self._queue.put_nowait((frame_copy, wp_copy))
        except queue.Full:
            if not self._queue_full_logged:
                print("[overlay] queue full, dropping frames to keep callback non-blocking")
                self._queue_full_logged = True
        except Exception as e:  # noqa: BLE001
            print(f"[overlay] queue put error: {e}")

    @staticmethod
    def _worker(
        frame_queue,
        stop_event,
        out_path: str,
        fps: float,
        fourcc: int,
    ):
        writer = None
        writer_size = None
        frames_written = 0
        init_error_logged = False

        while True:
            if stop_event.is_set() and frame_queue.empty():
                break
            try:
                item = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                continue

            frame_bgr, wp_np = item

            try:
                # Resize to the target resolution and apply trajectory
                # overlay if waypoints are available.
                try:
                    # First bring frame to the canonical 1920x1080 size.
                    frame_resized = cv2.resize(
                        frame_bgr,
                        (1920, 1080),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    if wp_np is not None:
                        rgb_img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        try:
                            frame_resized = project_waypoints_to_fisheye_image_with_polygon_new(
                                wp_np,
                                [790.0, 1920.0 / 2.0, 1080.0 / 2.0, 0.0],
                                rgb_img,
                                save_fig=False,
                            )
                        except Exception as e:  # noqa: BLE001
                            # Fallback to plain image if overlay fails.
                            print(f"[overlay] draw error in worker: {e}")
                            frame_resized = frame_resized
                except Exception as e:  # noqa: BLE001
                    print(f"[overlay] resize/overlay pipeline error: {e}")
                    frame_resized = frame_bgr

                if writer is None:
                    h, w = frame_resized.shape[:2]
                    writer_size = (w, h)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    writer = cv2.VideoWriter(out_path, fourcc, fps, writer_size)
                    if not writer.isOpened():
                        if not init_error_logged:
                            print(f"[overlay] ERROR: failed to open VideoWriter at {out_path}")
                            init_error_logged = True
                        writer = None

                    if writer is None:
                        # Cannot write; drop frame but continue draining queue
                        continue

                frame_out = frame_resized
                if writer_size is not None:
                    h, w = frame_resized.shape[:2]
                    w0, h0 = writer_size
                    if (w, h) != (w0, h0):
                        frame_out = cv2.resize(
                            frame_resized,
                            (w0, h0),
                            interpolation=cv2.INTER_LINEAR,
                        )

                writer.write(frame_out)
                frames_written += 1
                if frames_written % 30 == 0:
                    print(f"[overlay] frames written (async): {frames_written}")
            except Exception as e:  # noqa: BLE001
                # Avoid spamming FFmpeg warnings; log once per failure type
                print(f"[overlay] write error: {e}")

        if writer is not None:
            writer.release()
            writer = None

    def close(self):
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass

        if self._process.is_alive():
            self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

        try:
            self._queue.close()
            self._queue.join_thread()
        except Exception:  # noqa: BLE001
            pass


# -----------------------------------------------------------------------------
# Helpers from test_nav_flow_il_realsense
# -----------------------------------------------------------------------------


def load_target_positions_from_file(filename: str = "odom_poses.txt"):
    """Load target positions from txt file if present, else fall back to default.

    The original ROS script represents poses as 7D vectors:
    [x, y, z, qx, qy, qz, qw]. We preserve this format so that
    `transform_target_pose` and `pose_to_matrix` can treat targets in the
    same way as odom poses.
    """

    # Default goal: 3 m ahead in x, identity orientation.
    default = [[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    try:
        if not os.path.exists(filename):
            return default

        targets = []
        with open(filename, "r", encoding="utf-8") as f:
            # Expected header: x,y,z,qx,qy,qz,qw
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if not parts:
                    continue
                # If the file only provides x,y (legacy), pad to 7D.
                if len(parts) >= 7:
                    x, y, z = map(float, parts[:3])
                    qx, qy, qz, qw = map(float, parts[3:7])
                elif len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = 0.0
                    qx = qy = qz = 0.0
                    qw = 1.0
                else:
                    continue
                targets.append([x, y, z, qx, qy, qz, qw])

        return targets or default
    except Exception as e:  # noqa: BLE001
        print(f"[target] error loading target positions: {e}, using default {default}")
        return default


def trajectory_nms(trajectory_raw: torch.Tensor, score_raw: torch.Tensor,
                   max_trajectories: int = 6, distance_threshold: float = 2.0):
    """Apply NMS to trajectories based on endpoint distance.

    Args:
        trajectory_raw: [B, N, T, 3]
        score_raw: [B, N]
    Returns:
        selected_trajectories: [B, max_trajectories, T, 3]
        selected_scores: [B, max_trajectories]
    """

    B, N, T, _ = trajectory_raw.shape  # noqa: N806
    selected_trajectories = []
    selected_scores = []
    for b in range(B):
        trajs = trajectory_raw[b]
        scores = score_raw[b]
        sorted_indices = torch.argsort(scores, descending=True)
        keep_indices = []
        for i in sorted_indices:
            if len(keep_indices) >= max_trajectories:
                break
            should_keep = True
            for kept_idx in keep_indices:
                endpoint_dist = torch.norm(trajs[i][-1, :2] - trajs[kept_idx][-1, :2])
                if endpoint_dist < distance_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep_indices.append(int(i))
        while len(keep_indices) < max_trajectories:
            keep_indices.append(keep_indices[-1] if keep_indices else 0)
        keep_indices_t = torch.tensor(keep_indices[:max_trajectories], device=trajectory_raw.device)
        selected_trajectories.append(trajs[keep_indices_t])
        selected_scores.append(scores[keep_indices_t])
    return torch.stack(selected_trajectories), torch.stack(selected_scores)


# -----------------------------------------------------------------------------
# RealSense + ONNX + overlay ROS2 node
# -----------------------------------------------------------------------------


class RealSenseNavflowNode(Node):
    """ROS2 node: subscribe RealSense images + odom, run ONNX, overlay, and record.

    This is a distilled version of ImagePoseSyncNode from test_nav_flow_il_realsense,
    adapted to:
    - cache last waypoint prediction as `last_wp_np` (Nx2 in robot frame),
    - run ONNX in a background processing thread, and
    - overlay `last_wp_np` onto every accepted RGB frame using the
      same projection and color-space contract as the mp4 runner.
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        device: torch.device,
        target_xy,
        distance_threshold: float,
        frames: int = 21,
        output_fps: float = 30.0,
        save_vis: bool = False,
        video_writer: Optional[AsyncVideoWriter] = None,
    ):
        super().__init__("navflow_realsense_runner")
        self.session = session
        self.device = device
        self.target_xy = target_xy
        self.distance_threshold = distance_threshold
        self.frames = frames
        self.output_fps = float(output_fps) if float(output_fps) > 0.0 else 30.0
        self.save_vis = save_vis
        self.video_writer = video_writer
        # Zero-overhead bypass when both recording and save_vis are disabled.
        self.enable_visualization = bool(self.video_writer is not None or self.save_vis)

        # Sliding window of image tensors + poses used for ONNX
        self.image_batch = deque(maxlen=self.frames)

        self.pose_buffer = deque(maxlen=20)
        self.localpose_buffer = deque(maxlen=20)

        self.lock = threading.Lock()
        self.last_image_time = 0.0
        self.last_video_time = 0.0
        self.inference_event = threading.Event()

        # Counter to track when new frames are appended to image_batch
        self.update_counter = 0
        self.last_processed_counter = 0

        # Latest waypoint prediction in robot frame (N,2)
        self.last_wp_np: Optional[np.ndarray] = None

        # Intrinsic params for projection: f, cx, cy, k
        self.intrinsic_params = [790.0, 1920.0 / 2.0, 1080.0 / 2.0, 0.0]

        # Metrics and logging
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_count = 0

        # Publishers
        self.waypoint_pub = self.create_publisher(Float32MultiArray, "waypoint", 1)
        self.goal_pub = self.create_publisher(Bool, "/topoplan/reached_goal", 1)

        # Autonomy enable flag
        self.enable_2 = False
        self.autonomy_sub2 = self.create_subscription(
            Bool,
            "/cmd_vel/autonomy_enable",
            self.button_b_callback,
            10,
        )

        # Odom subscriptions
        self.localodom_sub = self.create_subscription(
            Odometry,
            "/mfla/dr_odom",
            self.localodom_callback,
            20,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/mfla/dr_odom",
            self.odom_callback,
            100,
        )

        # RealSense compressed image subscription
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/camera/camera/color/image_raw/compressed",
            self.image_callback,
            10,
        )

        self.get_logger().info("Warming up ONNX model (this may take 10+ seconds)...")
        # 构建一个全 0 的 dummy 输入
        dummy_obs = np.zeros((1, self.frames, 3, 288, 512), dtype=np.float32)
        dummy_metric = np.array([[0.0, 0.51, -0.26, 0.26]], dtype=np.float32)
        dummy_embodiment = np.array([[0]], dtype=np.int64)
        try:
            self.session.run(None, {
                "observation": dummy_obs,
                # "embodiment_id": dummy_embodiment,
                "metric_spacing": dummy_metric
            })
            self.get_logger().info("ONNX warmup complete!")
        except Exception as e:
            self.get_logger().warn(f"Warmup failed: {e}")
        # -------------------------------------------------------------------

        # Background processing thread for ONNX
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

        self.get_logger().info("RealSense NavFlow node initialized (save_vis=%s)" % self.save_vis)

    # ------------------------------------------------------------------
    # Odom + pose handling
    # ------------------------------------------------------------------

    def button_b_callback(self, msg: Bool):
        try:
            self.enable_2 = bool(msg.data)
        except Exception:  # noqa: BLE001
            self.enable_2 = False

    def localodom_callback(self, msg: Odometry):
        try:
            with self.lock:
                self.localpose_buffer.append((msg.header.stamp, msg.pose))
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Error in local odom callback: {e}")

    def odom_callback(self, msg: Odometry):
        try:
            with self.lock:
                self.pose_buffer.append((msg.header.stamp, msg.pose))
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Error in odom callback: {e}")

    def find_closest_localpose(self, image_stamp):
        with self.lock:
            if not self.localpose_buffer:
                return None
            image_nsec = image_stamp.nanosec + image_stamp.sec * 1e9
            closest_pose = None
            for pose_stamp, pose in self.localpose_buffer:
                pose_nsec = pose_stamp.nanosec + pose_stamp.sec * 1e9
                time_diff = pose_nsec - image_nsec
                if abs(time_diff) < 1e8 or time_diff > 0:
                    closest_pose = pose
                    break
            if closest_pose is None:
                closest_pose = self.localpose_buffer[-1][1]
            return closest_pose

    def find_closest_pose(self, image_stamp):
        with self.lock:
            if not self.pose_buffer:
                return None
            image_nsec = image_stamp.nanosec + image_stamp.sec * 1e9
            closest_pose = None
            for pose_stamp, pose in self.pose_buffer:
                pose_nsec = pose_stamp.nanosec + pose_stamp.sec * 1e9
                time_diff = pose_nsec - image_nsec
                if abs(time_diff) < 1e8 or time_diff > 0:
                    closest_pose = pose
                    break
            if closest_pose is None:
                closest_pose = self.pose_buffer[-1][1]
            return closest_pose

    # ------------------------------------------------------------------
    # Image callback: capture, resize, push to batch, overlay, and record
    # ------------------------------------------------------------------

    def image_callback(self, msg: CompressedImage):
        try:
            # Decode compressed image
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image_bgr is None:
                self.get_logger().warning("Failed to decode compressed image")
                return

            # Timestamp for rate control of ONNX enqueue (video itself runs at
            # full input frame rate).
            image_stamp = msg.header.stamp
            current_time = (image_stamp.nanosec + image_stamp.sec * 1e9) * 1e-9
            time_since_last = current_time - self.last_image_time

            # Source-side strict throttling for video recording to prevent
            # writer queue overflow and playback time compression on weak CPUs.
            if self.enable_visualization and self.video_writer is not None:
                min_video_interval = 1.0 / self.output_fps
                if (current_time - self.last_video_time) >= min_video_interval:
                    self.video_writer.write(image_bgr, self.last_wp_np)
                    self.last_video_time = current_time

            # Only every ~6 Hz frame is preprocessed for the ONNX model; all
            # intermediate frames skip the heavy PIL/torch pipeline.
            if time_since_last < 1.0 / 6.0:
                return

            # Pose sync (only needed when we actually feed the model).
            pose = self.find_closest_pose(image_stamp)
            localpose = self.find_closest_localpose(image_stamp)
            if pose is None or localpose is None:
                self.get_logger().warning("No pose yet for current image")
                return

            # BGR -> RGB for model preprocessing
            rgb_data = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Resize via PIL to 1920x1080 (same as other runners)
            pil_img = PILImage.fromarray(rgb_data, "RGB")
            desired_height = 1080
            desired_width = 1920
            W, H = pil_img.size
            scale_H, scale_W = desired_height / H, desired_width / W
            new_H, new_W = int(H * scale_H), int(W * scale_W)
            if (new_H, new_W) != (H, W):
                pil_img = pil_img.resize((new_W, new_H), PILImage.BILINEAR)
            img_np = np.array(pil_img)  # RGB, HxWx3

            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            _, H_resized, W_resized = img_tensor.shape
            pad_height = desired_height - H_resized
            pad_width = desired_width - W_resized
            if pad_height > 0 or pad_width > 0:
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                img_tensor = TF.pad(img_tensor, (pad_left, pad_top, pad_right, pad_bottom))
            elif pad_height < 0 or pad_width < 0:
                img_tensor = TF.center_crop(img_tensor, (desired_height, desired_width))

            assert img_tensor.shape[1] == desired_height and img_tensor.shape[2] == desired_width

            # Push into sliding window batch for ONNX thread at ~6 Hz.
            with self.lock:
                self.image_batch.append((img_tensor, pose, localpose))
                # Mark that a new frame has been appended so the
                # processing thread can trigger a fresh ONNX run.
                self.update_counter += 1
                if (
                    len(self.image_batch) == self.frames
                    and self.update_counter > self.last_processed_counter
                ):
                    self.inference_event.set()
            self.last_image_time = current_time

        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Error in image callback: {e}")

    # ------------------------------------------------------------------
    # Processing thread: run ONNX on full window and publish waypoints
    # ------------------------------------------------------------------

    def process_frames(self):
        while rclpy.ok():
            if not self.inference_event.wait(timeout=0.1):
                continue

            with self.lock:
                # Only process when the window is full *and* new
                # frames have been appended since the last run.
                if (
                    len(self.image_batch) == self.frames
                    and self.update_counter > self.last_processed_counter
                ):
                    batch_snapshot = list(self.image_batch)
                    self.last_processed_counter = self.update_counter
                else:
                    batch_snapshot = None

            if batch_snapshot is None:
                self.inference_event.clear()
                continue

            self.process_batch(batch_snapshot)

    @staticmethod
    def pose_to_matrix(pose):
        """Convert a pose-like input to a 4x4 transform matrix.

        The original code assumes a full 7D pose [x, y, z, qx, qy, qz, qw].
        To be more robust against underspecified targets (e.g. [x, y] or
        [x, y, z]), we auto-complete missing components as follows:

        - len==2  -> [x, y, 0.0, 0.0, 0.0, 0.0, 1.0]
        - len==3  -> [x, y, z, 0.0, 0.0, 0.0, 1.0]

        For len>=7 we use the first 7 entries, matching the original
        behavior. If the input has length 4-6 it is considered malformed
        and we raise a ValueError to surface the issue clearly.
        """

        pose_arr = np.asarray(pose, dtype=float).reshape(-1)

        if pose_arr.size == 2:
            x, y = pose_arr
            pose_arr = np.array([x, y, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        elif pose_arr.size == 3:
            x, y, z = pose_arr
            pose_arr = np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=float)
        elif pose_arr.size < 7:
            raise ValueError(
                f"Expected pose to have length 2, 3, or >=7, got {pose_arr.size}"
            )
        # For size >= 7, keep original semantics by using the first 7 entries.
        pose_arr = pose_arr[:7]

        position = pose_arr[:3]
        rotation = R.from_quat(pose_arr[3:7])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix

    @staticmethod
    def poses_to_matrices(poses):
        positions = poses[:, :3]
        quats = poses[:, 3:]
        rotations = R.from_quat(quats)
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = positions
        return matrices

    def transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self.poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]
        return positions

    def transform_target_pose(self, target_pose, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        target_pose_matrix = self.pose_to_matrix(target_pose)
        transformed_target_matrix = np.matmul(current_pose_inv, target_pose_matrix)
        target_position = transformed_target_matrix[:3, 3]
        return target_position

    @staticmethod
    def resample_waypoints_equal_distance(waypoints, num_points: Optional[int] = None):
        batch_size, num_wp, _ = waypoints.shape
        if num_points is None:
            num_points = num_wp
        device = waypoints.device
        resampled = []
        for b in range(batch_size):
            wp = waypoints[b].cpu().numpy()
            diffs = np.diff(wp, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
            total_length = cumulative_dist[-1]
            if total_length < 1e-6:
                resampled.append(wp)
                continue
            if num_points > 1:
                sample_distances = np.linspace(0, total_length, num_points)
            else:
                sample_distances = [total_length]
            resampled_wp = np.zeros((num_points, 2))
            for i, s_dist in enumerate(sample_distances):
                idx = np.searchsorted(cumulative_dist, s_dist, side="right") - 1
                idx = np.clip(idx, 0, len(cumulative_dist) - 2)
                if idx < len(cumulative_dist) - 1:
                    t = (s_dist - cumulative_dist[idx]) / (
                        cumulative_dist[idx + 1] - cumulative_dist[idx] + 1e-8
                    )
                    t = np.clip(t, 0, 1)
                    resampled_wp[i] = wp[idx] + t * (wp[idx + 1] - wp[idx])
                else:
                    resampled_wp[i] = wp[-1]
            resampled.append(resampled_wp)
        return torch.tensor(np.array(resampled), device=device, dtype=waypoints.dtype)

    def process_batch(self, batch_snapshot):
        """Run ONNX on the latest batch and publish / cache waypoints.

        `batch_snapshot` is a list of (img_tensor, pose, localpose) tuples.
        """

        self.get_logger().info(f"Processing batch of {len(batch_snapshot)} frames")
        if not batch_snapshot:
            self.get_logger().warning("Empty batch snapshot, skip processing")
            return

        snap_image, snap_pose, snap_localpose = batch_snapshot[-1]

        image_list = []
        pose_list = []
        for image, pose, _localpose in batch_snapshot:
            image_list.append(image)
            pose_list.append(
                [
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ]
            )

        last_localpose_snapshot = [
            snap_localpose.pose.position.x,
            snap_localpose.pose.position.y,
            snap_localpose.pose.position.z,
            snap_localpose.pose.orientation.w,
            snap_localpose.pose.orientation.x,
            snap_localpose.pose.orientation.y,
            snap_localpose.pose.orientation.z,
        ]

        input_poses = np.array(pose_list)
        current_pose = input_poses[-1]
        post_odom_temp = self.transform_poses(input_poses, current_pose)[:, [0, 1]]
        target_diff_temp = self.transform_target_pose(np.array(self.target_xy[0]), current_pose)[
            np.newaxis, [0, 1]
        ]

        post_odom = post_odom_temp.copy()
        target_diff = target_diff_temp.copy()
        post_odom[:, 0] = -post_odom_temp[:, 1]
        post_odom[:, 1] = post_odom_temp[:, 0]
        target_diff[:, 0] = -target_diff_temp[:, 1]
        target_diff[:, 1] = target_diff_temp[:, 0]

        self.get_logger().info(f"delta (x,y) = {target_diff[0][0], target_diff[0][1]}")

        transformed_input_positions = np.concatenate([post_odom, target_diff], axis=0)
        step_scale = torch.tensor(
            np.linalg.norm(np.diff(post_odom, axis=0), axis=1).mean()
        )
        step_scale = torch.clamp(step_scale, min=1e-2)
        step_scale = torch.tensor(1)
        transformed_input_positions /= 0.25

        video_frames = torch.stack(image_list, dim=0).unsqueeze(0).to(self.device)
        input_positions = torch.tensor(transformed_input_positions).unsqueeze(0).to(self.device)

        video_frames_raw = video_frames.clone()
        video_frames_small = torch.nn.functional.interpolate(
            video_frames_raw[0], (288, 512)
        ).unsqueeze(0)

        input_positions = input_positions.to(torch.float32).to(self.device)

        goal = input_positions[:, -1, :]
        goal = torch.cat([goal[:, 1:2], -goal[:, 0:1]], dim=1).reshape(1, 2)
        dis = torch.norm(goal, dim=1)
        goal[..., 0] = goal[..., 0].clamp(0.1, 500.0)
        goal[..., 1] = goal[..., 1].clamp(-50.0, 50.0)
        if torch.abs(torch.arctan2(goal[0, 1], goal[0, 0])) < np.pi / 9:
            goal[..., 1] = 0.0

        past_traj = torch.cat(
            [
                input_positions[:, :-2, 1:2],
                -input_positions[:, :-2, 0:1],
            ],
            dim=2,
        ).reshape(1, -1, 2) * 0.25
        past_traj = torch.cat([past_traj], dim=1)

        goal = torch.stack(
            [
                dis.clamp(0.1, 200.0) / 200.0,
                torch.cos(torch.arctan2(goal[:, 1], goal[:, 0])),
                torch.sin(torch.arctan2(goal[:, 1], goal[:, 0])),
            ],
            dim=-1,
        )

        past_mask = torch.ones([1, self.frames]).float().to(self.device)
        video_frames_small = torch.cat([video_frames_small], dim=1)

        input_obs = video_frames_small[0]
        with torch.no_grad():
            metric_spacing = np.array([[0.0, 0.51, -0.26, 0.26]], dtype=np.float32).reshape(1, 4)
            embodiment_id = np.array([[0]], dtype=np.int64)
            t0 = time.perf_counter()
            output = self.session.run(
                None,
                {
                    "observation": input_obs.unsqueeze(0).cpu().numpy().astype(np.float32),
                    # "embodiment_id": embodiment_id,
                    "metric_spacing": metric_spacing,
                },
            )
            t1 = time.perf_counter()
            infer_dt = max(t1 - t0, 1e-8)
            infer_fps = 1.0 / infer_dt
            self.get_logger().info(
                f"[perf] onnx inference: {infer_dt*1000:.2f} ms, fps={infer_fps:.2f}"
            )

            traj_raw = output[0]
            score_raw = output[1]
            traj_raw_t = torch.from_numpy(traj_raw)
            score_raw_t = torch.from_numpy(score_raw)

            traj, score = traj_raw_t.clone(), score_raw_t.clone()
            
            traj = traj[0]
            score = score[0]
            score = torch.softmax(score, dim=0)
            
            traj_final = traj[torch.argmax(score)]
            wp_pred = traj_final[..., :2].unsqueeze(0)

            wp_pred = torch.cat(
                [
                    torch.zeros_like(wp_pred[..., 0:1, :]),
                    wp_pred[..., :-1, :],
                ],
                dim=-2,
            )
            wp_pred[..., 0] = wp_pred[..., 0].clamp(0.1, 20.0)

        # Cache last waypoint prediction (robot frame, Nx2)
        self.last_wp_np = (
            wp_pred.detach()[0, :, :2]
            .cpu()
            .numpy()
        )

        wp_pred_cam = torch.cat(
            [
                -wp_pred[:, :, 1:2],
                wp_pred[:, :, 0:1],
            ],
            dim=-1,
        )

        waypoint_msg = Float32MultiArray()
        wp_pred_np = wp_pred_cam.detach().cpu().numpy()

        if self.save_vis:
            # Save snapshot image + metadata for offline analysis.
            img_np = snap_image.permute(1, 2, 0).detach().cpu().numpy()
            if img_np.dtype in (np.float32, np.float64):
                img_np = (img_np * 255).astype(np.uint8)

            enable_state = np.array(self.enable_2)
            folder_path = os.path.join(_REPO_ROOT, "vis", f"navflow{self.timestamp}")
            os.makedirs(folder_path, exist_ok=True)
            output_path = os.path.join(folder_path, f"sample_{self.image_count}.png")
            cv2.imwrite(output_path, img_np[..., ::-1])  # RGB->BGR

            pose_snapshot_np = np.array(
                [
                    snap_pose.pose.position.x,
                    snap_pose.pose.position.y,
                    snap_pose.pose.position.z,
                    snap_pose.pose.orientation.w,
                    snap_pose.pose.orientation.x,
                    snap_pose.pose.orientation.y,
                    snap_pose.pose.orientation.z,
                ]
            )

            np.save(os.path.join(folder_path, f"sample_{self.image_count}_traj.npy"), traj_raw)
            np.save(os.path.join(folder_path, f"sample_{self.image_count}_score.npy"), score_raw)
            np.save(os.path.join(folder_path, f"sample_{self.image_count}_auto.npy"), enable_state)
            np.save(os.path.join(folder_path, f"sample_{self.image_count}_pose.npy"), pose_snapshot_np)
            np.save(
                os.path.join(folder_path, f"sample_{self.image_count}_pose_aligned.npy"),
                np.array(last_localpose_snapshot),
            )

            # Also save the final waypoint prediction explicitly for convenience
            np.save(
                os.path.join(folder_path, f"sample_{self.image_count}_wp.npy"),
                self.last_wp_np,
            )

            self.image_count += 1

        # Publish waypoint message (same format as original script)
        wp_list = wp_pred_np.tolist()
        self.get_logger().info(
            f"waypoint (x,y) = {wp_list[0][-1][1], -wp_list[0][-1][0]}"
        )
        if wp_list[0][-1][1] > 0.1:
            tmp_list = []
            for i in range(0, len(wp_list[0]), 2):
                tmp_list += [wp_list[0][i][1]]
                tmp_list += [-wp_list[0][i][0]]
            waypoint_msg.data = tmp_list
        else:
            tmp_list = []
            for i in range(0, len(wp_list[0]), 2):
                tmp_list += [0.0]
                tmp_list += [0.0]
            waypoint_msg.data = tmp_list

        self.waypoint_pub.publish(waypoint_msg)
        reached_goal = False
        reached_goal_msg = Bool()
        reached_goal_msg.data = bool(reached_goal)
        self.goal_pub.publish(reached_goal_msg)
        self.inference_event.clear()


# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense NavFlow ONNX runner")
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to ONNX model (e.g., onnx_ckpts/s1_ours_coco_sft_stage2_goal_less.onnx)",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=10.0,
        help="Distance threshold used in odom-based logic (default: 10)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=21,
        help="Temporal window length T for ONNX input (default: 21)",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Enable saving offline PNG/NPY samples for analysis (vis/ directory)",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=30.0,
        help="Nominal FPS for the saved overlay video (default: 30Hz)",
    )
    parser.add_argument(
        "--no_record_video",
        action="store_false",
        dest="record_video",
        help="Disable MP4 video recording (AsyncVideoWriter)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ONNX session and device
    available_providers = ort.get_available_providers()
    providers = []
    provider_options = []

    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
        provider_options.append(
            {
                "gpu_mem_limit": str(8 * 1024 * 1024 * 1024),
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
            }
        )

    if "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")
        provider_options.append({})

    if not providers:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]

    print(f"[onnx] loading model from {args.onnx} with providers {providers}")
    session = ort.InferenceSession(
        args.onnx,
        providers=providers,
        provider_options=provider_options,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}")

    target_xy = load_target_positions_from_file()
    print(f"[target] loaded {len(target_xy)} target positions: {target_xy}")

    # Async video writer: record overlay MP4 unless explicitly disabled.
    async_writer = None
    if getattr(args, "record_video", True):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_dir = os.path.join(_REPO_ROOT, "vis", "overlay")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"realsense_navflow_overlay_{ts}.mp4")
        print(
            f"[overlay] enabling AsyncVideoWriter at {out_path}, fps={args.output_fps}"
        )
        async_writer = AsyncVideoWriter(out_path, args.output_fps, fourcc)
        async_writer.start()

    rclpy.init(args=None)
    node = RealSenseNavflowNode(
        session=session,
        device=device,
        target_xy=target_xy,
        distance_threshold=args.distance_threshold,
        frames=args.frames,
        output_fps=args.output_fps,
        save_vis=args.save_vis,
        video_writer=async_writer,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[ros] KeyboardInterrupt, shutting down...")
    finally:
        if async_writer is not None:
            async_writer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
