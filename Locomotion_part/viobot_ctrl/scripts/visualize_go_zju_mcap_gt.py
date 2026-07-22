#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MCAPS = [
    "/media/zjy/RTL9210C/go_ZJU/2026-05-10T16_51_56.028407621Z__bag_20260510_164655_4_5__rec_0eQJIsrpj5N6MbLL.mcap",
    "/media/zjy/RTL9210C/go_ZJU/2026-05-10T16_54_56.121499793Z__bag_20260510_164655_1_8__rec_0eSONXqVlJ0MUnow.mcap",
    "/media/zjy/RTL9210C/go_ZJU/2026-05-10T17_04_56.29076299Z__bag_20260510_164655_4_18__rec_0eQJIsefTGS8lIQL.mcap",
    "/media/zjy/RTL9210C/go_ZJU/2026-05-10T17_19_56.387870766Z__bag_20260510_164655_4_33__rec_0eQJIsegLdspHaqS.mcap",
]
DEFAULT_IMAGE_TOPIC = "/camera_x001/image_raw/compressed"
DEFAULT_ODOM_TOPIC = "/baton/stereo3/odometry"
DEFAULT_INTRINSICS = (749.2058, 749.2058, 1003.1, 526.5258)
DEFAULT_DISTORTION = (1.5847e-5, -1.3585e-7, 4.7386e-11)
DEFAULT_PITCH_DEG = 8.0
DEFAULT_CAMERA_HEIGHT = 0.68
DEFAULT_YAW_OFFSET_DEG = 90.0
# Keep the preferred view's nested, color-coded horizons while extending the
# longest band to the model's full 4 s prediction window at 20 Hz.
DEFAULT_HORIZONS = (80, 60, 40, 20)
DEFAULT_COLORS = [
    (100, 230, 100),
    (100, 100, 255),
    (255, 100, 100),
    (100, 255, 255),
]
CACHE_VERSION = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render odometry GT future trajectories on images from ROS2 MCAP files."
    )
    parser.add_argument(
        "--mcaps",
        nargs="+",
        default=DEFAULT_MCAPS,
        help="ROS2 MCAP files to visualize.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that will contain the script outputs.",
    )
    parser.add_argument(
        "--image-topic",
        default=DEFAULT_IMAGE_TOPIC,
        help="Compressed image topic.",
    )
    parser.add_argument(
        "--odom-topic",
        default=DEFAULT_ODOM_TOPIC,
        help="Odometry topic.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Future horizons in frames.",
    )
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        default=list(DEFAULT_INTRINSICS),
        metavar=("FX", "FY", "CX", "CY"),
        help="Camera intrinsics: fx, fy, cx, cy.",
    )
    parser.add_argument(
        "--distortion",
        type=float,
        nargs=3,
        default=list(DEFAULT_DISTORTION),
        metavar=("K1", "K2", "K3"),
        help="Radial_3 distortion coefficients k1, k2, k3.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Camera height in meters.",
    )
    parser.add_argument(
        "--camera-x-offset",
        type=float,
        default=-0.1,
        help="Camera translation along ego x before projection.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Output video FPS and frame-to-seconds conversion for the legend.",
    )
    parser.add_argument(
        "--offset-ms",
        type=float,
        default=0.0,
        help="Camera timestamp offset in milliseconds.",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=DEFAULT_YAW_OFFSET_DEG,
        help="Yaw correction from odom heading to base_link heading.",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=DEFAULT_PITCH_DEG,
        help="Camera up-pitch in degrees.",
    )
    parser.add_argument(
        "--yaw-offset-mode",
        choices=["fixed", "auto"],
        default="fixed",
        help="Use a fixed yaw offset or try auto-selecting its sign from motion.",
    )
    parser.add_argument(
        "--odom-lateral-convention",
        choices=["left_positive", "right_positive"],
        default="left_positive",
        help=(
            "Meaning of the second ego coordinate after yaw rotation. "
            "Use right_positive when negative yaw denotes a left turn."
        ),
    )
    parser.add_argument(
        "--path-width",
        type=float,
        default=0.22,
        help="Path half-width in meters; the rendered band is twice this value.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cached extraction outputs even if they already exist.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=0.0,
        help="Optional duration cap from the first image frame. 0 renders all frames.",
    )
    return parser.parse_args()


def normalize_quaternions(quaternions):
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Odometry contains a zero-length quaternion")
    return quaternions / norms


def make_quaternions_continuous(quaternions):
    """Choose a consistent sign for equivalent q/-q quaternion samples."""
    continuous = normalize_quaternions(np.asarray(quaternions, dtype=np.float64)).copy()
    for index in range(1, len(continuous)):
        if float(np.dot(continuous[index - 1], continuous[index])) < 0.0:
            continuous[index] *= -1.0
    return continuous


def quaternion_xyzw_to_yaw(quaternion):
    x, y, z, w = quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def rotate_points_xy(points, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return points @ rotation.T


def sort_and_deduplicate_poses(pose_times, poses):
    pose_times = np.asarray(pose_times, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64)
    if len(pose_times) != len(poses) or poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("Invalid odometry timestamp/pose arrays")
    if len(pose_times) < 2:
        raise RuntimeError("At least two odometry messages are required")

    order = np.argsort(pose_times, kind="stable")
    pose_times = pose_times[order]
    poses = poses[order]

    # Keep the last sample at duplicate timestamps.
    keep = np.r_[pose_times[1:] != pose_times[:-1], True]
    pose_times = pose_times[keep]
    poses = poses[keep]
    if len(pose_times) < 2:
        raise RuntimeError("Odometry must contain at least two distinct timestamps")
    if not np.all(np.isfinite(poses)):
        raise ValueError("Odometry contains non-finite pose values")

    poses[:, 3:7] = make_quaternions_continuous(poses[:, 3:7])
    return pose_times, poses


def interpolate_positions(sample_times, source_times, source_positions):
    return np.column_stack(
        [
            np.interp(sample_times, source_times, source_positions[:, axis])
            for axis in range(source_positions.shape[1])
        ]
    )


def interpolate_quaternions_slerp(sample_times, source_times, source_quaternions):
    """Vectorized shortest-arc SLERP for samples inside source_times."""
    sample_times = np.asarray(sample_times, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    quaternions = make_quaternions_continuous(source_quaternions)

    right = np.searchsorted(source_times, sample_times, side="right")
    right = np.clip(right, 1, len(source_times) - 1)
    left = right - 1
    intervals = source_times[right] - source_times[left]
    alpha = ((sample_times - source_times[left]) / intervals).reshape(-1, 1)

    q0 = quaternions[left]
    q1 = quaternions[right]
    dots = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dots < 0.0, -q1, q1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    nearly_equal = dots[:, 0] > 0.9995
    result = np.empty_like(q0)
    if np.any(nearly_equal):
        linear = q0[nearly_equal] + alpha[nearly_equal] * (
            q1[nearly_equal] - q0[nearly_equal]
        )
        result[nearly_equal] = normalize_quaternions(linear)

    spherical = ~nearly_equal
    if np.any(spherical):
        theta = np.arccos(dots[spherical])
        sin_theta = np.sin(theta)
        spherical_alpha = alpha[spherical]
        weight0 = np.sin((1.0 - spherical_alpha) * theta) / sin_theta
        weight1 = np.sin(spherical_alpha * theta) / sin_theta
        result[spherical] = weight0 * q0[spherical] + weight1 * q1[spherical]

    return normalize_quaternions(result)


def decode_compressed_image(encoded_bytes):
    img = cv2.imdecode(np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode compressed image frame")
    return img


def rotation_x(angle_rad):
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float64,
    )


def project_points_camera(
    points,
    intrinsic_params,
    distortion_coeffs,
    camera_height,
    camera_x_offset,
    pitch_deg,
):
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    fx, fy, cx, cy = intrinsic_params
    k1, k2, k3 = distortion_coeffs
    ego_points = np.zeros((points.shape[0], 3), dtype=np.float64)
    ego_points[:, :2] = points

    cam_translation = np.array([camera_x_offset, 0.0, camera_height], dtype=np.float64)
    base_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pitch_rotation = rotation_x(math.radians(-pitch_deg))
    rotation = pitch_rotation @ base_rotation

    cam_points = (rotation @ (ego_points - cam_translation).T).T
    pixels = np.full((len(cam_points), 2), np.nan, dtype=np.float64)
    visible = cam_points[:, 2] > 1e-6
    if not np.any(visible):
        return pixels

    x = cam_points[visible, 0] / cam_points[visible, 2]
    y = cam_points[visible, 1] / cam_points[visible, 2]
    r2 = x ** 2 + y ** 2
    radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
    x_distorted = x * radial
    y_distorted = y * radial

    pixels[visible, 0] = fx * x_distorted + cx
    pixels[visible, 1] = fy * y_distorted + cy
    return pixels


def build_offset_curves(waypoints, path_width):
    if waypoints.shape[0] < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))

    tangents = np.empty_like(waypoints, dtype=np.float64)
    tangents[0] = waypoints[1] - waypoints[0]
    tangents[-1] = waypoints[-1] - waypoints[-2]
    if len(waypoints) > 2:
        tangents[1:-1] = waypoints[2:] - waypoints[:-2]

    norms = np.linalg.norm(tangents, axis=1)
    valid = norms >= 1e-6
    if not np.any(valid):
        return np.zeros((0, 2)), np.zeros((0, 2))

    # Fill stationary samples from their nearest valid tangent so output arrays
    # remain index-aligned with the center curve and can be sliced per horizon.
    valid_indices = np.flatnonzero(valid)
    for index in np.flatnonzero(~valid):
        nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
        tangents[index] = tangents[nearest]
        norms[index] = norms[nearest]

    unit = tangents / norms[:, None]
    normals = np.column_stack((-unit[:, 1], unit[:, 0]))
    return waypoints + path_width * normals, waypoints - path_width * normals


def finite_points(points):
    if points.size == 0:
        return points
    return points[np.all(np.isfinite(points), axis=1)]


def bounded_points(points, width, height, margin_scale=8.0):
    points = finite_points(points)
    if points.size == 0:
        return points.reshape(0, 2)

    x_min = -width * margin_scale
    x_max = width * (1.0 + margin_scale)
    y_min = -height * margin_scale
    y_max = height * (1.0 + margin_scale)
    keep = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    return points[keep]


def draw_legend(img, horizons, colors, fps):
    legend_font = cv2.FONT_HERSHEY_SIMPLEX
    legend_font_scale = 0.6
    legend_font_thickness = 2
    legend_text_color = (255, 255, 255)
    legend_bg_color = (0, 0, 0)

    item_height = 25
    padding = 10
    bar_width = 15
    bar_height = 15

    height, width = img.shape[:2]
    bg_width = 190
    total_height = len(horizons) * item_height + padding * 2
    start_x = width - bg_width - 20
    start_y = max(10, height - total_height - 10)

    cv2.rectangle(
        img,
        (start_x - padding, start_y - padding),
        (start_x + bg_width, start_y + total_height),
        legend_bg_color,
        -1,
    )
    cv2.rectangle(
        img,
        (start_x - padding, start_y - padding),
        (start_x + bg_width, start_y + total_height),
        (255, 255, 255),
        2,
    )

    for idx, horizon in enumerate(horizons):
        y_pos = start_y + idx * item_height + padding
        color = colors[idx % len(colors)]

        cv2.rectangle(
            img,
            (start_x, y_pos),
            (start_x + bar_width, y_pos + bar_height),
            color,
            -1,
        )
        cv2.rectangle(
            img,
            (start_x, y_pos),
            (start_x + bar_width, y_pos + bar_height),
            (255, 255, 255),
            1,
        )

        seconds = horizon / fps
        text = f"{horizon}f ({seconds:.1f}s)"
        cv2.putText(
            img,
            text,
            (start_x + bar_width + 8, y_pos + bar_height - 2),
            legend_font,
            legend_font_scale,
            legend_text_color,
            legend_font_thickness,
            cv2.LINE_AA,
        )


def project_path_curves(
    waypoints,
    intrinsic_params,
    distortion_coeffs,
    camera_height,
    camera_x_offset,
    pitch_deg,
    path_width,
):
    offsets_left, offsets_right = build_offset_curves(waypoints, path_width)
    if len(offsets_left) == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, empty

    projection_args = (
        intrinsic_params,
        distortion_coeffs,
        camera_height,
        camera_x_offset,
        pitch_deg,
    )
    return (
        project_points_camera(waypoints, *projection_args),
        project_points_camera(offsets_left, *projection_args),
        project_points_camera(offsets_right, *projection_args),
    )


def overlay_projected_path(
    img,
    waypoints,
    points_center,
    points_left,
    points_right,
    color,
    base_alpha=0.15,
    min_path_length_m=0.05,
):
    if len(waypoints) < 2:
        return img
    path_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
    if path_length < min_path_length_m:
        return img

    height, width = img.shape[:2]
    points_center = bounded_points(points_center, width, height)
    points_left = bounded_points(points_left, width, height)
    points_right = bounded_points(points_right, width, height)

    if len(points_left) > 1 and len(points_right) > 1:
        polygon = np.vstack([points_left, points_right[::-1]]).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            alpha = (
                base_alpha
                + 0.45 * ((y_coords / max(1, height - 1)) ** 1.5)
            ).astype(np.float32)[:, None]
            source = img[y_coords, x_coords].astype(np.float32)
            target = np.asarray(color, dtype=np.float32)[None, :]
            img[y_coords, x_coords] = source * (1.0 - alpha) + target * alpha

        darker = tuple(int(component * 0.7) for component in color)
        for boundary in (points_left, points_right):
            visible = boundary[
                (boundary[:, 0] >= 0)
                & (boundary[:, 0] < width)
                & (boundary[:, 1] >= 0)
                & (boundary[:, 1] < height)
            ].astype(np.int32)
            if len(visible) > 1:
                cv2.polylines(img, [visible], False, darker, 3, cv2.LINE_AA)

    visible_center = points_center[
        (points_center[:, 0] >= 0)
        & (points_center[:, 0] < width)
        & (points_center[:, 1] >= 0)
        & (points_center[:, 1] < height)
    ]
    if len(visible_center) > 0:
        visible_center = np.vstack(
            [np.array([[width // 2, height - 1]], dtype=np.float64), visible_center]
        ).astype(np.int32)
        center_color = tuple(int(component * 0.8) for component in color)
        cv2.polylines(img, [visible_center], False, center_color, 2, cv2.LINE_AA)

    return img


def choose_yaw_offset_deg(odom_positions_xy, odom_yaws, fallback_deg):
    candidate_degs = [fallback_deg, -fallback_deg]
    if abs(fallback_deg) < 1e-6:
        return 0.0, "fallback_zero"

    gap = min(40, max(5, len(odom_positions_xy) // 30))
    valid_scores = []

    for candidate_deg in candidate_degs:
        candidate_rad = math.radians(candidate_deg)
        scores = []
        for index in range(0, len(odom_positions_xy) - gap, max(2, gap // 4)):
            delta = odom_positions_xy[index + gap] - odom_positions_xy[index]
            if float(np.linalg.norm(delta)) < 1e-4:
                continue
            rotated = rotate_points_xy(delta.reshape(1, 2), -(odom_yaws[index] + candidate_rad))[0]
            scores.append(float(rotated[0] - 0.25 * abs(rotated[1])))
        if scores:
            valid_scores.append((candidate_deg, float(np.mean(scores)), len(scores)))

    if not valid_scores:
        return fallback_deg, "fallback_low_motion"

    valid_scores.sort(key=lambda item: item[1], reverse=True)
    best_deg, best_score, _ = valid_scores[0]
    if len(valid_scores) > 1 and abs(valid_scores[0][1] - valid_scores[1][1]) < 1e-5:
        return fallback_deg, "fallback_ambiguous"
    return best_deg, f"auto_score={best_score:.6e}"


def import_mcap_runtime():
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing ROS2 MCAP dependency. Install `mcap` and `mcap-ros2-support`."
        ) from exc
    return make_reader, DecoderFactory


def message_stamp_ns(decoded_msg, fallback_ns):
    stamp = decoded_msg.header.stamp
    timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return timestamp_ns if timestamp_ns > 0 else int(fallback_ns)


def get_aligned_data(
    input_path,
    image_topic,
    odom_topic,
    offset_ms,
    yaw_offset_deg,
    yaw_offset_mode,
    max_duration_s=0.0,
):
    make_reader, DecoderFactory = import_mcap_runtime()
    frame_times_raw = []
    poses = []
    pose_times = []
    max_duration_ns = int(max_duration_s * 1e9) if max_duration_s > 0 else 0
    first_image_time_ns = None

    # Do not retain compressed image payloads. This pass only builds the small
    # timestamp/pose cache; images are decoded one at a time during rendering.
    with open(input_path, "rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded_msg in reader.iter_decoded_messages(
            topics=[image_topic, odom_topic]
        ):
            timestamp_ns = message_stamp_ns(decoded_msg, message.log_time)
            if channel.topic == image_topic:
                if first_image_time_ns is None:
                    first_image_time_ns = timestamp_ns
                if max_duration_ns and timestamp_ns - first_image_time_ns >= max_duration_ns:
                    continue
                frame_times_raw.append(timestamp_ns)
            elif channel.topic == odom_topic:
                pose_times.append(timestamp_ns)
                pose = decoded_msg.pose.pose
                poses.append(
                    [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                )

    if not frame_times_raw:
        raise RuntimeError(f"No image frames found on topic {image_topic}")
    if not poses:
        raise RuntimeError(f"No odometry messages found on topic {odom_topic}")

    frame_times_raw = np.asarray(frame_times_raw, dtype=np.int64)
    if np.any(np.diff(frame_times_raw) < 0):
        raise RuntimeError("Image header timestamps are not monotonic in MCAP order")

    pose_times, poses = sort_and_deduplicate_poses(pose_times, poses)
    query_times = frame_times_raw.astype(np.float64) - float(offset_ms) * 1e6
    covered = (query_times >= pose_times[0]) & (query_times <= pose_times[-1])
    if not np.any(covered):
        raise RuntimeError(
            "Image and odometry timestamps do not overlap after applying --offset-ms"
        )

    source_indices = np.flatnonzero(covered).astype(np.int64)
    dropped = int(len(frame_times_raw) - len(source_indices))
    if dropped:
        print(f"[align] dropped {dropped} image frame(s) outside odometry coverage")
    query_times = query_times[covered]

    frame_positions = interpolate_positions(query_times, pose_times, poses[:, :3])
    frame_orientations = interpolate_quaternions_slerp(
        query_times, pose_times, poses[:, 3:7]
    )

    odom_yaws = np.unwrap(
        np.asarray(
            [quaternion_xyzw_to_yaw(quaternion) for quaternion in poses[:, 3:7]],
            dtype=np.float64,
        )
    )
    if yaw_offset_mode == "auto":
        chosen_offset_deg, yaw_reason = choose_yaw_offset_deg(
            odom_positions_xy=poses[:, :2],
            odom_yaws=odom_yaws,
            fallback_deg=yaw_offset_deg,
        )
    else:
        chosen_offset_deg = yaw_offset_deg
        yaw_reason = "fixed"

    return {
        "frame_source_indices": source_indices,
        "frame_positions": frame_positions.astype(np.float32),
        "frame_orientations": frame_orientations.astype(np.float32),
        "yaw_offset_deg": float(chosen_offset_deg),
        "yaw_offset_reason": yaw_reason,
        "dropped_frame_count": dropped,
    }


def open_video_writer(output_path, width, height, fps):
    codecs = ["mp4v", "avc1", "MJPG"]
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
    raise RuntimeError(f"Failed to open video writer for {output_path}")


def write_json_atomic(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError):
        return None


def input_signature(input_path):
    resolved = input_path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cache_key_for(
    input_path,
    image_topic,
    odom_topic,
    offset_ms,
    yaw_offset_deg,
    yaw_offset_mode,
    max_duration_s,
):
    return {
        "version": CACHE_VERSION,
        "input": input_signature(input_path),
        "alignment": {
            "image_topic": image_topic,
            "odom_topic": odom_topic,
            "offset_ms": float(offset_ms),
            "yaw_offset_deg": float(yaw_offset_deg),
            "yaw_offset_mode": yaw_offset_mode,
            "max_duration_s": float(max_duration_s),
        },
    }


def cache_directory(output_root, input_path):
    path_hash = hashlib.sha256(str(input_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return output_root / ".visualize_go_zju_mcap_gt_cache" / f"{input_path.stem}_{path_hash}"


def cache_arrays_are_valid(cache_dir, expected_count):
    try:
        source_indices = np.load(
            cache_dir / "frame_source_indices.npy", mmap_mode="r"
        )
        positions = np.load(cache_dir / "frame_positions.npy", mmap_mode="r")
        orientations = np.load(cache_dir / "frame_orientations.npy", mmap_mode="r")
    except (OSError, ValueError):
        return False
    return (
        source_indices.ndim == 1
        and positions.ndim == 2
        and positions.shape[1] == 3
        and orientations.ndim == 2
        and orientations.shape[1] == 4
        and len(source_indices) == len(positions) == len(orientations) == expected_count
    )


def prepare_cache(
    input_path,
    output_root,
    image_topic,
    odom_topic,
    offset_ms,
    yaw_offset_deg,
    yaw_offset_mode,
    force,
    max_duration_s,
):
    scenario_name = input_path.stem
    scenario_dir = cache_directory(output_root, input_path)
    manifest_path = scenario_dir / "manifest.json"
    required_names = (
        "frame_source_indices.npy",
        "frame_positions.npy",
        "frame_orientations.npy",
    )
    cache_key = cache_key_for(
        input_path=input_path,
        image_topic=image_topic,
        odom_topic=odom_topic,
        offset_ms=offset_ms,
        yaw_offset_deg=yaw_offset_deg,
        yaw_offset_mode=yaw_offset_mode,
        max_duration_s=max_duration_s,
    )
    manifest = read_json(manifest_path)
    cache_valid = (
        not force
        and manifest is not None
        and manifest.get("cache_key") == cache_key
        and isinstance(manifest.get("result"), dict)
        and "yaw_offset_deg" in manifest["result"]
        and all((scenario_dir / name).is_file() for name in required_names)
        and cache_arrays_are_valid(
            scenario_dir, int(manifest["result"].get("frame_count", -1))
        )
    )
    if cache_valid:
        print(f"[reuse] {scenario_name}: alignment cache matches input and parameters")
        return scenario_dir

    # The manifest is the commit marker. Remove it before rebuilding so an
    # interrupted --force run can never make partially written arrays reusable.
    manifest_path.unlink(missing_ok=True)
    print(f"[extract] {input_path}")
    aligned = get_aligned_data(
        input_path=input_path,
        image_topic=image_topic,
        odom_topic=odom_topic,
        offset_ms=offset_ms,
        yaw_offset_deg=yaw_offset_deg,
        yaw_offset_mode=yaw_offset_mode,
        max_duration_s=max_duration_s,
    )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        scenario_dir / "frame_source_indices.npy", aligned["frame_source_indices"]
    )
    np.save(scenario_dir / "frame_positions.npy", aligned["frame_positions"])
    np.save(scenario_dir / "frame_orientations.npy", aligned["frame_orientations"])
    write_json_atomic(
        manifest_path,
        {
            "cache_key": cache_key,
            "result": {
                "frame_count": int(len(aligned["frame_positions"])),
                "dropped_frame_count": int(aligned["dropped_frame_count"]),
                "yaw_offset_deg": aligned["yaw_offset_deg"],
                "yaw_offset_reason": aligned["yaw_offset_reason"],
            },
        },
    )
    return scenario_dir


def iter_selected_image_frames(
    input_path,
    image_topic,
    source_indices,
    max_duration_s,
):
    make_reader, DecoderFactory = import_mcap_runtime()
    source_indices = np.asarray(source_indices, dtype=np.int64)
    wanted_position = 0
    source_index = 0
    first_image_time_ns = None
    max_duration_ns = int(max_duration_s * 1e9) if max_duration_s > 0 else 0

    with open(input_path, "rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _, message, decoded_msg in reader.iter_decoded_messages(
            topics=[image_topic]
        ):
            timestamp_ns = message_stamp_ns(decoded_msg, message.log_time)
            if first_image_time_ns is None:
                first_image_time_ns = timestamp_ns
            if max_duration_ns and timestamp_ns - first_image_time_ns >= max_duration_ns:
                break

            if (
                wanted_position < len(source_indices)
                and source_index == source_indices[wanted_position]
            ):
                yield decoded_msg.data
                wanted_position += 1
                if wanted_position == len(source_indices):
                    break
            source_index += 1

    if wanted_position != len(source_indices):
        raise RuntimeError(
            f"MCAP ended after {wanted_position}/{len(source_indices)} cached image frames; "
            "rerun with --force"
        )


def render_gt_video(
    input_path,
    cache_dir,
    output_root,
    output_name,
    image_topic,
    max_duration_s,
    horizons,
    colors,
    intrinsic_params,
    distortion_coeffs,
    camera_height,
    camera_x_offset,
    pitch_deg,
    path_width,
    fps,
    odom_lateral_convention,
):
    positions = np.load(cache_dir / "frame_positions.npy", mmap_mode="r")
    orientations = np.load(cache_dir / "frame_orientations.npy", mmap_mode="r")
    source_indices = np.load(cache_dir / "frame_source_indices.npy", mmap_mode="r")
    manifest = read_json(cache_dir / "manifest.json")
    if manifest is None:
        raise RuntimeError("Alignment cache manifest is missing or invalid")
    yaw_offset_deg = float(manifest["result"]["yaw_offset_deg"])
    total_frames = len(positions)
    if (
        total_frames == 0
        or len(orientations) != total_frames
        or len(source_indices) != total_frames
    ):
        raise RuntimeError("Alignment cache arrays have inconsistent lengths")

    output_mp4 = output_root / f"{output_name}_gt.mp4"
    temporary_mp4 = output_root / f".{output_name}_gt.tmp.mp4"
    temporary_mp4.unlink(missing_ok=True)
    writer = None
    frame_size = None
    codec = None
    max_horizon = max(horizons)
    print(f"[render] {output_name}: frames={total_frames}, fps={fps:.3f}")

    try:
        frame_iterator = iter_selected_image_frames(
            input_path=input_path,
            image_topic=image_topic,
            source_indices=source_indices,
            max_duration_s=max_duration_s,
        )
        rendered_count = 0
        for frame_index, encoded in enumerate(frame_iterator):
            frame = decode_compressed_image(encoded)
            if writer is None:
                height, width = frame.shape[:2]
                frame_size = (width, height)
                writer, codec = open_video_writer(temporary_mp4, width, height, fps)
            elif (frame.shape[1], frame.shape[0]) != frame_size:
                raise RuntimeError("Image resolution changed within the MCAP")

            yaw = quaternion_xyzw_to_yaw(orientations[frame_index]) + math.radians(
                yaw_offset_deg
            )
            max_end_index = min(frame_index + max_horizon, total_frames - 1)
            if max_end_index > frame_index:
                waypoints_world = positions[frame_index : max_end_index + 1, :2]
                waypoints_ego = rotate_points_xy(
                    waypoints_world - waypoints_world[0], -yaw
                )
                if odom_lateral_convention == "right_positive":
                    waypoints_ego[:, 1] *= -1.0

                projected_center, projected_left, projected_right = project_path_curves(
                    waypoints_ego,
                    intrinsic_params=intrinsic_params,
                    distortion_coeffs=distortion_coeffs,
                    camera_height=camera_height,
                    camera_x_offset=camera_x_offset,
                    pitch_deg=pitch_deg,
                    path_width=path_width,
                )
                for horizon_index, horizon in enumerate(horizons):
                    point_count = min(horizon, max_end_index - frame_index) + 1
                    if point_count < 2:
                        continue
                    overlay_projected_path(
                        frame,
                        waypoints=waypoints_ego[:point_count],
                        points_center=projected_center[:point_count],
                        points_left=projected_left[:point_count],
                        points_right=projected_right[:point_count],
                        color=colors[horizon_index % len(colors)],
                    )

            draw_legend(frame, horizons, colors, fps)
            writer.write(frame)
            rendered_count += 1
            if frame_index % 100 == 0 or rendered_count == total_frames:
                print(f"[render] {output_name}: {rendered_count}/{total_frames}")

        if rendered_count != total_frames:
            raise RuntimeError(
                f"Rendered {rendered_count}/{total_frames} expected image frames"
            )
    except Exception:
        if writer is not None:
            writer.release()
        temporary_mp4.unlink(missing_ok=True)
        raise

    writer.release()
    temporary_mp4.replace(output_mp4)
    write_json_atomic(
        output_root / f"{output_name}_gt.json",
        {
            "input": input_signature(input_path),
            "alignment": manifest["cache_key"]["alignment"],
            "alignment_result": manifest["result"],
            "render": {
                "camera_height": float(camera_height),
                "camera_x_offset": float(camera_x_offset),
                "codec": codec,
                "colors_bgr": [list(color) for color in colors],
                "distortion": list(distortion_coeffs),
                "fps": float(fps),
                "frame_count": int(total_frames),
                "horizons": list(horizons),
                "intrinsics": list(intrinsic_params),
                "odom_lateral_convention": odom_lateral_convention,
                "path_half_width": float(path_width),
                "pitch_deg": float(pitch_deg),
            },
        },
    )
    print(f"[done] {output_mp4} (codec={codec})")
    return output_mp4


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_duration_s < 0:
        raise ValueError("--max-duration-s cannot be negative")
    if args.path_width <= 0:
        raise ValueError("--path-width must be positive")
    if args.camera_height <= 0:
        raise ValueError("--camera-height must be positive")
    if not args.horizons or any(horizon <= 0 for horizon in args.horizons):
        raise ValueError("--horizons must contain positive frame counts")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(mcap).expanduser().resolve() for mcap in args.mcaps]
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Input is not a file: {input_path}")
        if input_path.suffix.lower() != ".mcap":
            raise ValueError(f"Only ROS2 .mcap inputs are supported: {input_path}")

    stem_counts = {
        stem: sum(path.stem == stem for path in input_paths)
        for stem in {path.stem for path in input_paths}
    }
    scenarios = []
    for input_path in input_paths:
        cache_dir = prepare_cache(
            input_path=input_path,
            output_root=output_root,
            image_topic=args.image_topic,
            odom_topic=args.odom_topic,
            offset_ms=args.offset_ms,
            yaw_offset_deg=args.yaw_offset_deg,
            yaw_offset_mode=args.yaw_offset_mode,
            force=args.force,
            max_duration_s=args.max_duration_s,
        )
        output_name = input_path.stem
        if stem_counts[input_path.stem] > 1:
            output_name += "_" + hashlib.sha256(
                str(input_path).encode("utf-8")
            ).hexdigest()[:12]
        scenarios.append((input_path, cache_dir, output_name))

    results = []
    for input_path, cache_dir, output_name in scenarios:
        results.append(
            render_gt_video(
                input_path=input_path,
                cache_dir=cache_dir,
                output_root=output_root,
                output_name=output_name,
                image_topic=args.image_topic,
                max_duration_s=args.max_duration_s,
                horizons=args.horizons,
                colors=DEFAULT_COLORS,
                intrinsic_params=tuple(args.intrinsics),
                distortion_coeffs=tuple(args.distortion),
                camera_height=args.camera_height,
                camera_x_offset=args.camera_x_offset,
                pitch_deg=args.pitch_deg,
                path_width=args.path_width,
                fps=args.fps,
                odom_lateral_convention=args.odom_lateral_convention,
            )
        )

    print("\nGenerated videos:")
    for result in results:
        print(f"  {result}")


if __name__ == "__main__":
    main()
