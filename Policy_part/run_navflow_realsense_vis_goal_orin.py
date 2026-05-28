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
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray

import ast

import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
_VIS_FILE = os.path.join(_REPO_ROOT, "Policy_part", "tools", "vis_gt_single_gt.py")
project_waypoints_to_fisheye_image_with_polygon_new = load_function_from_file(
    _VIS_FILE, "project_waypoints_to_fisheye_image_with_polygon_new"
)


def create_onnx_session(model_path: str) -> ort.InferenceSession:
    available_providers = ort.get_available_providers()
    providers = []
    provider_options = []

    gpu_mem_limit_gb = float(os.environ.get("NAVFLOW_ORT_GPU_MEM_LIMIT_GB", "4"))
    gpu_mem_limit = int(gpu_mem_limit_gb * 1024 * 1024 * 1024)

    disable_trt = os.environ.get("NAVFLOW_DISABLE_TRT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not disable_trt and "TensorrtExecutionProvider" in available_providers:
        providers.append("TensorrtExecutionProvider")
        provider_options.append(
            {
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": os.environ.get(
                    "NAVFLOW_TRT_ENGINE_CACHE",
                    os.path.expanduser("~/.cache/navflow_trt"),
                ),
                "trt_max_workspace_size": str(gpu_mem_limit),
            }
        )

    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
        provider_options.append(
            {
                "gpu_mem_limit": str(gpu_mem_limit),
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

    attempts = [(providers, provider_options)]
    if "TensorrtExecutionProvider" in providers:
        trt_idx = providers.index("TensorrtExecutionProvider")
        cuda_providers = providers[:trt_idx] + providers[trt_idx + 1 :]
        cuda_options = provider_options[:trt_idx] + provider_options[trt_idx + 1 :]
        attempts.append((cuda_providers, cuda_options))
    if providers != ["CPUExecutionProvider"] and "CPUExecutionProvider" in available_providers:
        attempts.append((["CPUExecutionProvider"], [{}]))

    print(f"[onnx] available providers: {available_providers}")
    last_error = None
    for attempt_providers, attempt_options in attempts:
        try:
            print(
                f"[onnx] loading model from {model_path} with providers {attempt_providers}"
            )
            session = ort.InferenceSession(
                model_path,
                providers=attempt_providers,
                provider_options=attempt_options,
            )
            print(f"[onnx] active providers: {session.get_providers()}")
            return session
        except Exception as exc:
            last_error = exc
            print(f"[onnx] provider attempt failed: {attempt_providers}: {exc}")
    raise last_error


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

    def __init__(
        self,
        out_path: str,
        fps: float,
        fourcc: int,
        max_queue: int = 128,
        overlay_camera_pitch_deg: float = 8.0,
    ):
        self.out_path = out_path
        self.fps = float(fps)
        self.fourcc = fourcc
        self.max_queue = max_queue
        self.overlay_camera_pitch_deg = float(overlay_camera_pitch_deg)

        # Queue of (frame_bgr, wp_np, frame_time) tuples, where wp_np is an
        # optional (N, 2) array of waypoints in robot frame.
        self._queue = mp.Queue(maxsize=max_queue)
        self._stop_event = mp.Event()
        self._process = mp.Process(
            target=self._worker,
            args=(
                self._queue,
                self._stop_event,
                self.out_path,
                self.fps,
                self.fourcc,
                self.overlay_camera_pitch_deg,
            ),
            daemon=True,
        )
        self._queue_full_logged = False

    def start(self):
        self._process.start()

    def write(self, frame_bgr: np.ndarray, wp_np: Optional[np.ndarray], frame_time: Optional[float] = None):
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
            t_copy = float(frame_time) if frame_time is not None else time.time()
            self._queue.put_nowait((frame_copy, wp_copy, t_copy))
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
        overlay_camera_pitch_deg: float,
    ):
        writer = None
        writer_size = None
        frames_written = 0
        init_error_logged = False
        pending_items = []
        pending_times = []

        def estimate_fps_from_times(times, fallback_fps):
            if len(times) < 2:
                return max(float(fallback_fps), 1.0)
            arr = np.asarray(times, dtype=np.float64)
            deltas = np.diff(arr)
            deltas = deltas[(deltas > 1e-4) & (deltas < 1.0)]
            if deltas.size == 0:
                return max(float(fallback_fps), 1.0)
            est = 1.0 / float(np.median(deltas))
            return float(np.clip(est, 1.0, 120.0))

        def ensure_writer(frame_resized):
            nonlocal writer, writer_size, init_error_logged
            if writer is not None:
                return True

            h, w = frame_resized.shape[:2]
            writer_size = (w, h)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            estimated_fps = estimate_fps_from_times(pending_times, fps)
            print(f"[overlay] init VideoWriter fps={estimated_fps:.2f}")
            writer = cv2.VideoWriter(out_path, fourcc, estimated_fps, writer_size)
            if not writer.isOpened():
                if not init_error_logged:
                    print(f"[overlay] ERROR: failed to open VideoWriter at {out_path}")
                    init_error_logged = True
                writer = None
                return False
            return True

        while True:
            if stop_event.is_set() and frame_queue.empty():
                break
            try:
                item = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                continue

            frame_bgr, wp_np, frame_time = item
            pending_times.append(frame_time)
            if len(pending_times) > 240:
                pending_times = pending_times[-240:]

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
                                camera_pitch_deg=overlay_camera_pitch_deg,
                            )
                        except Exception as e:  # noqa: BLE001
                            # Fallback to plain image if overlay fails.
                            print(f"[overlay] draw error in worker: {e}")
                            frame_resized = frame_resized
                except Exception as e:  # noqa: BLE001
                    print(f"[overlay] resize/overlay pipeline error: {e}")
                    frame_resized = frame_bgr

                pending_items.append(frame_resized)

                # Delay writer initialization briefly to estimate real input FPS.
                should_init = (writer is None) and (len(pending_items) >= 12 or (stop_event.is_set() and len(pending_items) > 0))
                if should_init and not ensure_writer(frame_resized):
                    continue

                if writer is None:
                    continue

                while pending_items:
                    frame_out = pending_items.pop(0)
                    if writer_size is not None:
                        h, w = frame_out.shape[:2]
                        w0, h0 = writer_size
                        if (w, h) != (w0, h0):
                            frame_out = cv2.resize(
                                frame_out,
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
        show_window: bool = True,
        publish_overlay: bool = True,
        overlay_topic: str = "/navflow/overlay/image_raw/compressed",
        overlay_transport: str = "compressed",
        overlay_jpeg_quality: int = 85,
        overlay_camera_pitch_deg: float = 8.0,
        video_writer: Optional[AsyncVideoWriter] = None,
    ):
        super().__init__("navflow_realsense_runner")
        self.session = session
        self.device = device
        self.model_input_names = [inp.name for inp in self.session.get_inputs()]
        self.target_xy = target_xy
        self.distance_threshold = distance_threshold
        self.frames = frames
        self.output_fps = float(output_fps) if float(output_fps) > 0.0 else 30.0
        self.save_vis = save_vis
        self.show_window = bool(show_window)
        self.publish_overlay = bool(publish_overlay)
        self.overlay_topic = str(overlay_topic)
        self.overlay_transport = str(overlay_transport)
        self.overlay_jpeg_quality = int(np.clip(int(overlay_jpeg_quality), 1, 100))
        self.overlay_camera_pitch_deg = float(overlay_camera_pitch_deg)
        self.video_writer = video_writer
        self.window_name = "navflow_realsense_overlay"
        # Zero-overhead bypass when both recording and save_vis are disabled.
        self.enable_visualization = bool(
            self.video_writer is not None
            or self.save_vis
            or self.show_window
            or self.publish_overlay
        )

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Sliding window of image tensors + poses used for ONNX
        self.image_batch = deque(maxlen=self.frames)

        self.pose_buffer = deque(maxlen=20)
        self.localpose_buffer = deque(maxlen=20)

        self.lock = threading.Lock()
        self.last_image_time = 0.0
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
        if self.publish_overlay and self.overlay_transport == "compressed":
            self.overlay_pub = self.create_publisher(CompressedImage, self.overlay_topic, 1)
        elif self.publish_overlay:
            self.overlay_pub = self.create_publisher(Image, self.overlay_topic, 1)
        else:
            self.overlay_pub = None
        if self.overlay_pub is not None:
            self.get_logger().info(
                f"Publishing overlay image topic: {self.overlay_topic} "
                f"({self.overlay_transport})"
            )

        # Autonomy enable flag
        self.enable_2 = False
        self.goal_reached = False
        self.autonomy_sub2 = self.create_subscription(
            Bool,
            "/cmd_vel/autonomy_enable",
            self.button_b_callback,
            10,
        )

        # Odom subscriptions
        self.localodom_sub = self.create_subscription(
            Odometry,
            "/globalEstimator/global_odometry",
            self.localodom_callback,
            20,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/globalEstimator/global_odometry",
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

        self.get_logger().info(f"ONNX input names: {self.model_input_names}")

        self.get_logger().info("Warming up ONNX model (this may take 10+ seconds)...")
        # 构建一个全 0 的 dummy 输入
        dummy_obs = np.zeros((1, self.frames, 3, 288, 512), dtype=np.float32)
        dummy_goal = np.array([[0.1, 1.0, 0.0]], dtype=np.float32)
        try:
            warmup_feed = self.build_onnx_feed(dummy_obs, dummy_goal)
            self.session.run(None, warmup_feed)
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

                if not self.goal_reached and len(self.target_xy) > 0:
                    dx = self.target_xy[0][0] - msg.pose.pose.position.x
                    dy = self.target_xy[0][1] - msg.pose.pose.position.y
                    pose_diff = math.sqrt(dx ** 2 + dy ** 2)

                    if pose_diff < self.distance_threshold:
                        if len(self.target_xy) > 1:
                            del self.target_xy[0]
                            self.get_logger().info(
                                f"Reached current target, {len(self.target_xy)} targets remaining"
                            )
                        else:
                            self.goal_reached = True
                            self.get_logger().info("Reached final target")

            reached_goal_msg = Bool()
            reached_goal_msg.data = bool(self.goal_reached)
            self.goal_pub.publish(reached_goal_msg)
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

    def render_overlay_frame(self, frame_bgr: np.ndarray):
        frame_resized = cv2.resize(
            frame_bgr,
            (1920, 1080),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.last_wp_np is None:
            return frame_resized

        rgb_img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        try:
            frame_out = project_waypoints_to_fisheye_image_with_polygon_new(
                self.last_wp_np,
                self.intrinsic_params,
                rgb_img,
                save_fig=False,
                camera_pitch_deg=self.overlay_camera_pitch_deg,
            )
            return frame_out
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f"Overlay render failed: {e}")
            return frame_resized

    @staticmethod
    def bgr_to_image_msg(frame_bgr: np.ndarray, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(frame_bgr.shape[0])
        msg.width = int(frame_bgr.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(frame_bgr.shape[1] * frame_bgr.shape[2])
        msg.data = frame_bgr.tobytes()
        return msg

    def bgr_to_compressed_image_msg(self, frame_bgr: np.ndarray, header) -> Optional[CompressedImage]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.overlay_jpeg_quality]
        ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_params)
        if not ok:
            self.get_logger().warning("Failed to JPEG-encode overlay image")
            return None
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        return msg

    def image_callback(self, msg: CompressedImage):
        try:
            # Decode compressed image
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image_bgr is None:
                self.get_logger().warning("Failed to decode compressed image")
                return

            if self.show_window or self.overlay_pub is not None:
                display_frame = self.render_overlay_frame(image_bgr)
                if self.overlay_pub is not None:
                    if self.overlay_transport == "compressed":
                        overlay_msg = self.bgr_to_compressed_image_msg(display_frame, msg.header)
                    else:
                        overlay_msg = self.bgr_to_image_msg(display_frame, msg.header)
                    if overlay_msg is not None:
                        self.overlay_pub.publish(overlay_msg)
                if self.show_window:
                    cv2.imshow(self.window_name, display_frame)
                    cv2.waitKey(1)

            # Timestamp for rate control of ONNX enqueue (video itself runs at
            # full input frame rate).
            image_stamp = msg.header.stamp
            current_time = (image_stamp.nanosec + image_stamp.sec * 1e9) * 1e-9
            time_since_last = current_time - self.last_image_time

            # Record every incoming frame for video; no fixed-rate throttle.
            if self.enable_visualization and self.video_writer is not None:
                self.video_writer.write(image_bgr, self.last_wp_np, frame_time=current_time)

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

            try:
                self.process_batch(batch_snapshot)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"Error in process_batch: {e}")
                self.inference_event.clear()

    def build_onnx_feed(self, observation: np.ndarray, goal_point: Optional[np.ndarray] = None):
        """Build ONNX feed dict based on actual model input names."""

        metric_spacing = np.array([[0.0, 0.51, -0.26, 0.26]], dtype=np.float32).reshape(1, 4)
        embodiment_id = np.array([[0]], dtype=np.int64)

        input_names = set(self.model_input_names)
        feed = {}
        if "observation" in input_names:
            feed["observation"] = observation
        elif self.model_input_names:
            # Fallback: some exported models rename the first input.
            feed[self.model_input_names[0]] = observation
        else:
            raise RuntimeError("ONNX model has no inputs")

        if "metric_spacing" in input_names:
            feed["metric_spacing"] = metric_spacing
        if "embodiment_id" in input_names:
            feed["embodiment_id"] = embodiment_id
        if "goal_point" in input_names:
            if goal_point is None:
                raise RuntimeError("ONNX model expects goal_point but no goal was provided")
            feed["goal_point"] = goal_point.astype(np.float32)
        return feed

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
        # Snapshot shared state once to avoid races with localodom_callback.
        with self.lock:
            target_snapshot = list(self.target_xy)
            goal_reached_snapshot = self.goal_reached

        if not target_snapshot:
            self.get_logger().warning("No target pose available, skip this batch")
            reached_goal_msg = Bool()
            reached_goal_msg.data = bool(goal_reached_snapshot)
            self.goal_pub.publish(reached_goal_msg)
            self.inference_event.clear()
            return

        # Keep target semantics consistent with sequential deletion logic:
        # localodom_callback always removes the first target when reached.
        current_target_pose = np.array(target_snapshot[0])
        target_diff_temp = self.transform_target_pose(current_target_pose, current_pose)[np.newaxis, [0, 1]]

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

        goal = input_positions[:, -1, :].clone()
        goal = torch.cat([goal[:, 1:2], -goal[:, 0:1]], dim=1).reshape(1, 2)
        dis = torch.norm(goal, dim=1) * 0.25
        goal[..., 0] = goal[..., 0].clamp(0.1, 200.0)
        goal[..., 1] = goal[..., 1].clamp(-500.0, 500.0)

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
                dis.clamp(0.1, 100.0) / 100.0,
                torch.cos(torch.arctan2(goal[:, 1], goal[:, 0]).clamp(-torch.pi / 2.0, torch.pi / 2.0)),
                torch.sin(torch.arctan2(goal[:, 1], goal[:, 0]).clamp(-torch.pi / 2.0, torch.pi / 2.0)),
            ],
            dim=-1,
        )

        past_mask = torch.ones([1, self.frames]).float().to(self.device)
        video_frames_small = torch.cat([video_frames_small], dim=1)

        input_obs = video_frames_small[0]
        with torch.no_grad():
            observation = input_obs.unsqueeze(0).cpu().numpy().astype(np.float32)
            goal_point = goal.cpu().numpy().reshape(1, 3).astype(np.float32)
            onnx_feed = self.build_onnx_feed(observation, goal_point)
            t0 = time.perf_counter()
            output = self.session.run(None, onnx_feed)
            t1 = time.perf_counter()
            infer_dt = max(t1 - t0, 1e-8)
            infer_fps = 1.0 / infer_dt
            self.get_logger().info(
                f"[perf] onnx inference: {infer_dt*1000:.2f} ms, fps={infer_fps:.2f}"
            )

            traj_raw = output[0]
            score_raw = output[1]
            if (
                traj_raw.ndim >= 3
                and score_raw.ndim >= 2
                and traj_raw.shape[1] > 1
                and score_raw.shape[1] > 1
            ):
                traj_raw = traj_raw[:, 1:]
                score_raw = score_raw[:, 1:]
            traj_raw_t = torch.from_numpy(traj_raw)
            score_raw_t = torch.from_numpy(score_raw)

            traj, score = trajectory_nms(traj_raw_t, score_raw_t)
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

            # plotting
            wp_to_plot = self.last_wp_np

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plt.subplots_adjust(wspace=0.3)
            
            self.get_logger().info(f"DEBUG -> Global Robot Pos: {current_pose[:2]}")
            self.get_logger().info(f"DEBUG -> Relative Target (Local): {target_diff_temp[0]}")

            ax1.imshow(img_np)  # original RGB image
            ax1.axis('off')
            ax1.set_title("Current View")
            ax2.axis('equal')
            ax2.plot(wp_to_plot[:, 0], wp_to_plot[:, 1],
                        's-', label='Predicted Waypoints', color='#DB6057')
            ax2.axhline(0, color='black', lw=1, ls='--') 
            ax2.axvline(0, color='black', lw=1, ls='--') 
            ax2.plot(target_diff_temp[0][0], target_diff_temp[0][1],
                        marker='*', markersize=15, label='Target Coordinate', color='#A157DB')
            ax2.legend()
            ax2.set_title('Coordinates BEV Trajectory (m) ', fontsize=20)
            ax2.set_xlabel('X (m)', fontsize=20)
            ax2.set_ylabel('Y (m)', fontsize=20)
            ax2.tick_params(axis='both', labelsize=18)
            ax2.grid(True)
            ax2.legend()

            enable_state = np.array(self.enable_2)
            folder_path = os.path.join(_REPO_ROOT, "Policy_part", "vis", f"navflow{self.timestamp}")
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(os.path.join(folder_path, f"vis_{self.image_count}.png"))
            plt.close(fig)
            # output_path = os.path.join(folder_path, f"sample_{self.image_count}.png")
            # cv2.imwrite(output_path, img_np[..., ::-1])  # RGB->BGR

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
        if wp_list[0][-1][1] > 0.12:
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
        reached_goal_msg = Bool()
        reached_goal_msg.data = bool(self.goal_reached)
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
        "--record_video",
        action="store_true",
        help="Enable MP4 video recording (disabled by default)",
    )
    parser.add_argument(
        "--no_show_window",
        action="store_false",
        dest="show_window",
        help="Disable realtime OpenCV preview window",
    )
    parser.add_argument(
        "--overlay_topic",
        type=str,
        default=None,
        help="Topic for the rendered overlay frame. Defaults depend on --overlay_transport.",
    )
    parser.add_argument(
        "--overlay_transport",
        type=str,
        choices=("compressed", "raw"),
        default="compressed",
        help="Publish overlay as sensor_msgs/CompressedImage or raw sensor_msgs/Image",
    )
    parser.add_argument(
        "--overlay_jpeg_quality",
        type=int,
        default=85,
        help="JPEG quality for compressed overlay publishing (1-100)",
    )
    parser.add_argument(
        "--overlay_camera_pitch_deg",
        type=float,
        default=8.0,
        help=(
            "Visualization-only camera pitch in degrees. Positive tilts the "
            "virtual camera upward and moves projected ground paths lower."
        ),
    )
    parser.add_argument(
        "--no_publish_overlay",
        action="store_false",
        dest="publish_overlay",
        help="Disable publishing the rendered overlay image topic",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="odom_poses.txt",
        help="Path to target poses file (default: odom_poses.txt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.overlay_topic is None:
        if args.overlay_transport == "compressed":
            args.overlay_topic = "/navflow/overlay/image_raw/compressed"
        else:
            args.overlay_topic = "/navflow/overlay/image_raw"

    session = create_onnx_session(args.onnx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}")

    target_xy = load_target_positions_from_file(args.targets)
    print(f"[target] loaded {len(target_xy)} target positions: {target_xy}")

    # Async video writer: record overlay MP4 unless explicitly disabled.
    async_writer = None
    if getattr(args, "record_video", False):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_dir = os.path.join(_REPO_ROOT, "Policy_part", "vis", "overlay")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"realsense_navflow_overlay_{ts}.mp4")
        print(
            f"[overlay] enabling AsyncVideoWriter at {out_path} "
            f"(fps estimate from input, fallback={args.output_fps})"
        )
        async_writer = AsyncVideoWriter(
            out_path,
            args.output_fps,
            fourcc,
            overlay_camera_pitch_deg=args.overlay_camera_pitch_deg,
        )
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
        show_window=args.show_window,
        publish_overlay=args.publish_overlay,
        overlay_topic=args.overlay_topic,
        overlay_transport=args.overlay_transport,
        overlay_jpeg_quality=args.overlay_jpeg_quality,
        overlay_camera_pitch_deg=args.overlay_camera_pitch_deg,
        video_writer=async_writer,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[ros] KeyboardInterrupt, shutting down...")
    finally:
        if async_writer is not None:
            async_writer.close()
        if args.show_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
