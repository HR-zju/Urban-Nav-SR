#!/usr/bin/env python3
"""
SAM3 + SoM + RouteSupport 标注工具 - Phase2 完整格式
直接在原始 clip 目录上生成 artifacts

此脚本实现 Phase2 数据集扩增，生成三种机器可读标注：
1. fpv_segmentation: 密集 affordance 掩码和类别摘要
2. som: Segment-of-Mask 区域级标注（连通分量着色 + 编号标记）
3. route_support: 基于轨迹的 BEV 意图感知支持热力图

输出目录结构：
    clip_dir/
        front.mp4
        ... (原有文件)
        fpv_segmentation/
            000001_mask.npy           # SAM3 分割掩码
            000001_affordance.npy      # 4通道 affordance 张量
            000001_summary.json       # 类别摘要
            000001_overlay.jpg       # 可视化覆盖图
        som/
            000001_som.npy          # 连通分量标签（用于重建着色）
            000001_som.json         # SoM 区域标注
            000001_overlay.jpg      # SoM 可视化覆盖图
        route_support/
            000001_route_support.npy # 1通道热力图
            000001_route_support.json # 热力图元数据
            000001_overlay.jpg      # 热力图可视化
        debug_video/
            segmentation_overlay.mp4
            som_overlay.mp4
            route_support_overlay.mp4
        processing_manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _repo_root() -> Path:
    return Path(__file__).parent.parent


_SRC_DIR = _repo_root() / "src"
if str(_SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_SRC_DIR))


@dataclass(frozen=True)
class TrajectoryFrame:
    frame_name: str
    frame_idx: int
    lat: float
    lng: float
    heading_deg: float | None
    image_path: Path


# ============================================================================
# 类别配置
# ============================================================================
DEFAULT_CLASS_CONFIG = {
    "preferred_navigable": ["sidewalk", "crosswalk", "plaza", "pedestrian_area"],
    "conditional_navigable": ["curb_cut", "driveway", "road_shoulder"],
    "forbidden": ["road", "vehicle", "pedestrian", "pole", "wall", "building", "tree", "grass", "fence"],
    "ignore": ["unknown", "sky", "reflection", "background"],
}

AFFORDANCE_CHANNELS = {
    "preferred_navigable": 0,
    "conditional_navigable": 1,
    "forbidden": 2,
    "boundary_or_uncertain": 3,
}


# ============================================================================
# SoM 配色方案（参考 mcp_server/som/overlay.py）
# ============================================================================
_SOM_PALETTE: tuple[tuple[int, int, int], ...] = (
    (34, 197, 94), (14, 165, 233), (249, 115, 22), (168, 85, 247),
    (236, 72, 153), (234, 179, 8), (20, 184, 166), (59, 130, 246),
)
_SOM_FORBIDDEN_PALETTE: tuple[tuple[int, int, int], ...] = (
    (239, 68, 68), (220, 38, 38), (185, 28, 28), (244, 63, 94),
)
_SOM_CONDITIONAL_PALETTE: tuple[tuple[int, int, int], ...] = (
    (245, 158, 11), (234, 179, 8), (251, 146, 60),
)


# ============================================================================
# 工具函数 - 轨迹和帧提取
# ============================================================================
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


def _calculate_heading_from_gps(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate compass heading (0=North, clockwise) from two GPS points."""
    if abs(lat1 - lat2) < 1e-7 and abs(lng1 - lng2) < 1e-7:
        return 0.0
    d_lon = math.radians(lng2 - lng1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    y = math.sin(d_lon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def _extract_frames_from_mp4(mp4_path: Path, output_dir: Path, sample_fps: float = 2.0) -> list[Path]:
    """Extract frames from mp4 at given sample rate."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(images_dir / "%06d.jpg")
    cmd = ["ffmpeg", "-i", str(mp4_path), "-vf", f"fps={sample_fps}", "-q:v", "2", "-y", output_pattern]
    print(f"Extracting frames from {mp4_path.name} at {sample_fps}fps...")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    frames = sorted(images_dir.glob("*.jpg"))
    print(f"Extracted {len(frames)} frames")
    return frames


def _load_trajectory_from_npy(clip_dir: Path, images_dir: Path, sample_fps: float = 2.0) -> dict[str, TrajectoryFrame]:
    """Load trajectory from frame_gps.npy and frame_orientations.npy."""
    gps_arr = np.load(clip_dir / "frame_gps.npy")
    times_arr = np.load(clip_dir / "frame_times.npy")

    num_frames_total = len(gps_arr)
    extracted_frames = sorted(images_dir.glob("*.jpg"))
    num_samples = len(extracted_frames)

    step = max(1, num_frames_total // max(1, num_samples))
    selected_indices = list(range(0, num_frames_total, step))[:num_samples]

    if len(selected_indices) != len(extracted_frames):
        selected_indices = selected_indices[:len(extracted_frames)]

    valid_headings = []
    for i in range(len(selected_indices) - 1):
        idx1 = selected_indices[i]
        idx2 = selected_indices[i + 1]
        lat1 = float(gps_arr[idx1, 0])
        lng1 = float(gps_arr[idx1, 1])
        lat2 = float(gps_arr[idx2, 0])
        lng2 = float(gps_arr[idx2, 1])
        if not math.isnan(lat1) and not math.isnan(lng1) and not math.isnan(lat2) and not math.isnan(lng2):
            h = _calculate_heading_from_gps(lat1, lng1, lat2, lng2)
            valid_headings.append(h)
    default_heading = valid_headings[0] if valid_headings else 0.0

    frames: dict[str, TrajectoryFrame] = {}
    for img_idx, (img_path, data_idx) in enumerate(zip(extracted_frames, selected_indices), start=1):
        frame_idx = img_idx
        frame_name = f"{frame_idx:06d}"

        lat = float(gps_arr[data_idx, 0])
        lng = float(gps_arr[data_idx, 1])

        heading_deg = default_heading

        # img_idx starts at 1, convert to 0-indexed for selected_indices
        curr_i = img_idx - 1

        # Try to calculate heading from current to next
        if curr_i < len(selected_indices) - 1:
            next_i = curr_i + 1
            next_data_idx = selected_indices[next_i]
            next_lat = float(gps_arr[next_data_idx, 0])
            next_lng = float(gps_arr[next_data_idx, 1])
            if not math.isnan(next_lat) and not math.isnan(next_lng):
                h = _calculate_heading_from_gps(lat, lng, next_lat, next_lng)
                if not math.isnan(h):
                    heading_deg = h

        # If that didn't work, try from previous to current
        if heading_deg == default_heading and curr_i > 0:
            prev_i = curr_i - 1
            prev_data_idx = selected_indices[prev_i]
            prev_lat = float(gps_arr[prev_data_idx, 0])
            prev_lng = float(gps_arr[prev_data_idx, 1])
            if not math.isnan(prev_lat) and not math.isnan(prev_lng):
                h = _calculate_heading_from_gps(prev_lat, prev_lng, lat, lng)
                if not math.isnan(h):
                    heading_deg = h

        frames[frame_name] = TrajectoryFrame(
            frame_name=frame_name,
            frame_idx=frame_idx,
            lat=lat,
            lng=lng,
            heading_deg=heading_deg,
            image_path=img_path,
        )

    return frames


def _iter_selected_frame_names(*, images_dir: Path, frame_start: int | None, frame_end: int | None, frame_step: int) -> list[str]:
    """Select frame names from directory."""
    names: list[str] = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = path.stem.strip()
        if not stem.isdigit():
            continue
        idx = int(stem)
        if frame_start is not None and idx < int(frame_start):
            continue
        if frame_end is not None and idx > int(frame_end):
            continue
        names.append(stem)
    step = max(int(frame_step), 1)
    return names[::step]


def _ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SAM3 推理相关
# ============================================================================
def _prepare_frame_subset_dir(*, images_dir: Path, subset_dir: Path, frame_names: Iterable[str]) -> None:
    """Prepare subset directory with symlinks to selected frames."""
    subset_dir.mkdir(parents=True, exist_ok=True)
    for child in list(subset_dir.iterdir()):
        if child.is_file() or child.is_symlink():
            try:
                child.unlink()
            except Exception:
                pass
    for name in frame_names:
        candidates = [images_dir / f"{name}.jpg", images_dir / f"{name}.png", images_dir / f"{name}.jpeg"]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            continue
        dst = subset_dir / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src)
        except Exception:
            dst.write_bytes(src.read_bytes())


def _run_sam3_inference_on_images(
    *,
    images_dir: Path,
    selected_frame_names: list[str] | None,
    sam3_output_base_dir: Path,
    sequence_name: str,
    camera_view: str,
    sam3_env_name: str,
    sam3_inference_script: Path,
    gpus: list[int],
    min_score: float,
    finetuned: bool,
    checkpoint_path: Path | None,
    work_dir: Path
) -> None:
    """Run SAM3 inference on images."""
    cocoa_base_dir = work_dir / "cocoa_frames"
    camera_dir = cocoa_base_dir / sequence_name / camera_view
    if selected_frame_names:
        _prepare_frame_subset_dir(images_dir=images_dir, subset_dir=camera_dir, frame_names=selected_frame_names)

    script_path = sam3_inference_script.expanduser().resolve()
    if not script_path.exists():
        raise RuntimeError(f"SAM3 inference script not found: {script_path}")

    if finetuned and (checkpoint_path is None or not checkpoint_path.exists()):
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    cmd: list[str] = [
        "conda", "run", "-n", str(sam3_env_name), "python", str(script_path),
        "--cocoa_base_dir", str(cocoa_base_dir),
        "--output_base_dir", str(sam3_output_base_dir),
        "--camera_views", str(camera_view),
        "--gpus", *[str(int(g)) for g in gpus],
        "--min_score", str(float(min_score)),
        "--sequence_names", str(sequence_name),
        "--finetuned", "true" if finetuned else "false",
    ]
    if finetuned and checkpoint_path is not None:
        cmd.extend(["--checkpoint_path", str(checkpoint_path)])

    print(f"[INFO] Running SAM3 inference...")
    if selected_frame_names:
        print(f"[INFO] SAM3 frame subset size: {len(selected_frame_names)}")
    print(f"[INFO] Output to: {sam3_output_base_dir}")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(work_dir / "mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(work_dir / "xdg_cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    subprocess.run(cmd, check=True, capture_output=True)


def _load_class_mapping(mapping_path: Path) -> dict[int, str]:
    """Load class mapping from SAM3 output file."""
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {i + 1: name for i, name in enumerate(data)}
    elif isinstance(data, dict):
        if "class_to_idx" in data:
            class_to_idx = data["class_to_idx"]
            return {idx: name for name, idx in class_to_idx.items()}
        elif "thing_classes" in data:
            thing_classes = data["thing_classes"]
            return {i: name for i, name in enumerate(thing_classes)}
        else:
            try:
                return {int(k): v for k, v in data.items()}
            except (ValueError, TypeError):
                return {}
    return {}


def _get_affordance(class_name: str, class_config: dict[str, list[str]]) -> str:
    """Get affordance category for a class name."""
    class_name_lower = class_name.lower()
    for pref in class_config["preferred_navigable"]:
        if pref in class_name_lower:
            return "preferred_navigable"
    for cond in class_config["conditional_navigable"]:
        if cond in class_name_lower:
            return "conditional_navigable"
    for ign in class_config.get("ignore", []):
        if ign in class_name_lower:
            return "ignore"
    return "forbidden"


# ============================================================================
# SoM 连通分量算法（参考 mcp_server/som/context.py）
# ============================================================================
def _connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Find connected components in a binary mask."""
    try:
        from scipy import ndimage as _ndimage
        structure = np.ones((3, 3), dtype=np.int8)
        labels, count = _ndimage.label(binary_mask.astype(bool), structure=structure)
        return labels.astype(np.int32, copy=False), int(count)
    except Exception:
        pass

    height, width = binary_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    count = 0
    for v in range(height):
        for u in range(width):
            if not binary_mask[v, u] or labels[v, u] != 0:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(v, u)])
            labels[v, u] = count
            while queue:
                cv, cu = queue.popleft()
                for dv in (-1, 0, 1):
                    for du in (-1, 0, 1):
                        if dv == 0 and du == 0:
                            continue
                        nv = cv + dv
                        nu = cu + du
                        if nv < 0 or nv >= height or nu < 0 or nu >= width:
                            continue
                        if not binary_mask[nv, nu] or labels[nv, nu] != 0:
                            continue
                        labels[nv, nu] = count
                        queue.append((nv, nu))
    return labels, int(count)


# ============================================================================
# FPV Segmentation - 生成 affordance 掩码
# ============================================================================
def generate_affordance_mask(mask: np.ndarray, class_mapping: dict[int, str], class_config: dict[str, list[str]]) -> np.ndarray:
    """Generate 4-channel affordance tensor from SAM3 mask.

    通道布局:
        0: preferred_navigable (人行道, 人行横道等)
        1: conditional_navigable (路沿坡, 车道等)
        2: forbidden (道路, 车辆, 行人, 障碍物等)
        3: boundary_or_uncertain (边界或不确定区域)
    """
    height, width = mask.shape
    affordance_mask = np.zeros((4, height, width), dtype=np.float32)

    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        class_name = class_mapping.get(class_id, "unknown")
        affordance = _get_affordance(class_name, class_config)

        if affordance == "ignore":
            affordance_mask[3, mask == class_id] = 1.0
        else:
            channel_idx = AFFORDANCE_CHANNELS[affordance]
            affordance_mask[channel_idx, mask == class_id] = 1.0

    edges = np.zeros((height, width), dtype=bool)
    edges[1:, :] |= mask[1:, :] != mask[:-1, :]
    edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    affordance_mask[3, edges] = np.maximum(affordance_mask[3, edges], 0.5)

    return affordance_mask


def process_sam3_mask(
    mask_path: Path,
    class_mapping: dict[int, str],
    class_config: dict[str, list[str]],
    image_width: int,
    image_height: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Process SAM3 mask into fpv_segmentation outputs.

    输出:
        summary: 类别摘要
        mask_uint16: 原始类别掩码 (uint16)
        affordance_mask: 4通道 affordance 张量 (float32)
    """
    mask = np.load(str(mask_path))

    if mask.ndim == 3:
        if mask.shape[0] in {1, 3}:
            mask = mask[0] if mask.shape[0] == 1 else mask.argmax(axis=0)
        elif mask.shape[-1] in {1, 3}:
            mask = mask[..., 0] if mask.shape[-1] == 1 else mask.argmax(axis=-1)

    mask_uint16 = mask.astype(np.uint16)

    affordance_mask = generate_affordance_mask(mask, class_mapping, class_config)

    classes = []
    for class_id in np.unique(mask):
        if class_id == 0:
            continue

        class_mask = mask == class_id
        area = int(class_mask.sum())

        if area < 100:
            continue

        class_name = class_mapping.get(class_id, "unknown")
        affordance = _get_affordance(class_name, class_config)

        if affordance == "ignore":
            continue

        ys, xs = np.where(class_mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_xyxy = [x_min, y_min, x_max, y_max]

        bottom_row = image_height - 1
        bottom_connected = bool(np.any(class_mask[bottom_row - 5: bottom_row + 1, :]))

        classes.append({
            "mask_id": int(class_id),
            "class_name": class_name,
            "affordance": affordance,
            "area": area,
            "bbox_xyxy": bbox_xyxy,
            "bottom_connected": bottom_connected,
            "confidence": 0.92,
        })

    summary = {
        "schema_version": "sidewalk_segmentation_v1",
        "frame_idx": -1,
        "image_width": image_width,
        "image_height": image_height,
        "mask_path": "",
        "affordance_path": "",
        "overlay_path": "",
        "navigation_mode": "sidewalk_robot",
        "classes": classes,
        "ignore_labels": class_config.get("ignore", ["unknown", "sky", "reflection"]),
        "preferred_navigable_class_names": class_config["preferred_navigable"],
        "conditional_navigable_class_names": class_config["conditional_navigable"],
        "forbidden_class_names": class_config["forbidden"],
    }

    return summary, mask_uint16, affordance_mask


def draw_segmentation_overlay(
    image_path: Path,
    affordance_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Draw affordance visualization overlay."""
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    color_map = [
        (34, 197, 94, 150),
        (245, 158, 11, 150),
        (239, 68, 68, 150),
        (128, 128, 128, 100),
    ]

    for channel_idx in range(4):
        channel_mask = affordance_mask[channel_idx] > 0.5
        if not np.any(channel_mask):
            continue
        color = color_map[channel_idx]
        mask_img = Image.fromarray((channel_mask * 255).astype(np.uint8), mode="L")
        color_layer = Image.new("RGBA", img.size, color)
        overlay.paste(color_layer, mask=mask_img)

    combined = Image.alpha_composite(img, overlay)
    combined.convert("RGB").save(output_path, "JPEG", quality=95)


# ============================================================================
# SoM - Segment-of-Mask 区域级标注
# ============================================================================
def _side_from_center(center_u: float, width: int) -> str:
    """Determine side (left/center/right) from center U coordinate."""
    ratio = float(center_u) / float(max(width, 1))
    if ratio < 0.38:
        return "left"
    if ratio > 0.62:
        return "right"
    return "center"


def _regions_from_labels(
    mask: np.ndarray,
    labels: np.ndarray,
    component_count: int,
    id_to_name: dict[int, str],
    class_config: dict[str, list[str]],
    min_region_area_ratio: float,
    max_regions: int,
    region_prefix: str,
    image_height: int,
    image_width: int,
) -> list[dict[str, Any]]:
    """Extract region annotations from connected component labels.

    与用户提供的 sample schema 保持一致:
        - label: 类别名称
        - affordance: affordance 类型
        - source_mask_id: 原始 mask id
        - bbox_xyxy: 边界框
        - centroid_xy: 中心点
        - area: 面积（像素数）
        - bottom_connected: 是否触底
    """
    total_pixels = float(max(image_height * image_width, 1))
    regions: list[dict[str, Any]] = []

    for component_id in range(1, int(component_count) + 1):
        component_pixels = np.argwhere(labels == component_id)
        if component_pixels.size == 0:
            continue

        pixel_count = int(component_pixels.shape[0])
        area_ratio = float(pixel_count) / total_pixels
        if area_ratio < float(min_region_area_ratio):
            continue

        ys = component_pixels[:, 0]
        xs = component_pixels[:, 1]
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        center_x = float(xs.mean())
        center_y = float(ys.mean())

        class_ids, class_counts = np.unique(mask[labels == component_id], return_counts=True)
        dominant_idx = int(np.argmax(class_counts)) if len(class_counts) else 0
        dominant_class_id = int(class_ids[dominant_idx]) if len(class_ids) else -1
        dominant_class_name = str(id_to_name.get(dominant_class_id, f"unknown_{dominant_class_id}"))
        affordance = _get_affordance(dominant_class_name, class_config)

        # Skip ignore regions
        if affordance == "ignore":
            continue

        bottom_row = image_height - 1
        bottom_connected = bool(np.any((labels == component_id)[bottom_row - 5: bottom_row + 1, :]))

        regions.append({
            "region_id": f"{region_prefix}_{len(regions) + 1}",
            "source_mask_id": dominant_class_id,
            "label": dominant_class_name,
            "affordance": affordance,
            "bbox_xyxy": [x_min, y_min, x_max, y_max],
            "centroid_xy": [round(center_x, 2), round(center_y, 2)],
            "area": pixel_count,
            "bottom_connected": bottom_connected,
            "touches_trajectory": False,
            "trajectory_overlap_ratio": 0.0,
        })

    regions.sort(
        key=lambda item: (
            int(bool(item.get("bottom_connected"))),
            int(item.get("area", 0)),
        ),
        reverse=True,
    )

    if max_regions > 0:
        regions = regions[:int(max_regions)]

    for idx, region in enumerate(regions, start=1):
        region["region_id"] = f"{region_prefix}_{idx}"

    return regions


def build_som_from_segmentation(
    mask: np.ndarray,
    class_mapping: dict[int, str],
    class_config: dict[str, list[str]],
    frame_idx: int,
    image_width: int,
    image_height: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build SoM annotation from segmentation mask.

    这是 Phase2 的关键：将 SAM 分割转换为区域级标注
    - preferred: 优先可通行区域（带 som_ 前缀）
    - conditional: 条件可通行区域（带 cond_ 前缀）
    - forbidden: 禁止通行区域（带 obs_ 前缀）

    返回:
        som_data: SoM 标注字典
        som_labels: 连通分量标签数组（用于重建着色）
    """
    traversable_ids = set()
    conditional_ids = set()
    forbidden_ids = set()

    for class_id, class_name in class_mapping.items():
        affordance = _get_affordance(class_name, class_config)
        if affordance == "preferred_navigable":
            traversable_ids.add(class_id)
        elif affordance == "conditional_navigable":
            conditional_ids.add(class_id)
        elif affordance == "forbidden":
            forbidden_ids.add(class_id)

    traversable_mask = np.isin(mask, list(traversable_ids)) if traversable_ids else np.zeros_like(mask, dtype=bool)
    conditional_mask = np.isin(mask, list(conditional_ids)) if conditional_ids else np.zeros_like(mask, dtype=bool)
    forbidden_mask = np.isin(mask, list(forbidden_ids)) if forbidden_ids else np.zeros_like(mask, dtype=bool)

    traversable_labels, traversable_count = _connected_components(traversable_mask)
    conditional_labels, conditional_count = _connected_components(conditional_mask)
    forbidden_labels, forbidden_count = _connected_components(forbidden_mask)

    regions = _regions_from_labels(
        mask=mask,
        labels=traversable_labels,
        component_count=traversable_count,
        id_to_name=class_mapping,
        class_config=class_config,
        min_region_area_ratio=0.002,
        max_regions=32,
        region_prefix="som",
        image_height=image_height,
        image_width=image_width,
    )

    conditional_regions = _regions_from_labels(
        mask=mask,
        labels=conditional_labels,
        component_count=conditional_count,
        id_to_name=class_mapping,
        class_config=class_config,
        min_region_area_ratio=0.002,
        max_regions=32,
        region_prefix="cond",
        image_height=image_height,
        image_width=image_width,
    )

    forbidden_regions = _regions_from_labels(
        mask=mask,
        labels=forbidden_labels,
        component_count=forbidden_count,
        id_to_name=class_mapping,
        class_config=class_config,
        min_region_area_ratio=0.002,
        max_regions=32,
        region_prefix="obs",
        image_height=image_height,
        image_width=image_width,
    )

    # 重建 som_labels，确保与过滤后的区域正确对应
    som_labels = np.zeros_like(mask, dtype=np.int32)
    label_id = 1

    # 将原始 component id 映射到过滤后的区域 id
    # 方法：重新构建标签，只包含保留的区域
    all_regions = regions + conditional_regions + forbidden_regions

    # 首先重新计算所有连通分量，然后根据区域的 bbox/centroid 匹配
    # 更简单的方法：将原始标签重新分配

    # 构建区域到原始 component 的映射（通过面积和位置匹配）
    # 对于 traversable 区域
    for region in regions:
        # 找到原始 component 中面积和位置匹配的
        centroid = region["centroid_xy"]
        cx, cy = int(centroid[0]), int(centroid[1])
        if 0 <= cy < image_height and 0 <= cx < image_width:
            orig_comp_id = traversable_labels[cy, cx]
            if orig_comp_id > 0:
                som_labels[traversable_labels == orig_comp_id] = label_id
        label_id += 1

    # 对于 conditional 区域
    for region in conditional_regions:
        centroid = region["centroid_xy"]
        cx, cy = int(centroid[0]), int(centroid[1])
        if 0 <= cy < image_height and 0 <= cx < image_width:
            orig_comp_id = conditional_labels[cy, cx]
            if orig_comp_id > 0:
                som_labels[conditional_labels == orig_comp_id] = label_id
        label_id += 1

    # 对于 forbidden 区域
    for region in forbidden_regions:
        centroid = region["centroid_xy"]
        cx, cy = int(centroid[0]), int(centroid[1])
        if 0 <= cy < image_height and 0 <= cx < image_width:
            orig_comp_id = forbidden_labels[cy, cx]
            if orig_comp_id > 0:
                som_labels[forbidden_labels == orig_comp_id] = label_id
        label_id += 1

    preferred_ids = [r["region_id"] for r in regions]
    conditional_region_ids = [r["region_id"] for r in conditional_regions]
    forbidden_region_ids = [r["region_id"] for r in forbidden_regions]

    route_trace = [r["region_id"] for r in regions if r.get("bottom_connected", False)][:1]

    som_data = {
        "schema_version": "sidewalk_som_v1",
        "frame_idx": frame_idx,
        "image_width": image_width,
        "image_height": image_height,
        "regions": regions + conditional_regions + forbidden_regions,
        "preferred": preferred_ids,
        "conditional": conditional_region_ids,
        "forbidden": forbidden_region_ids,
        "route_trace": route_trace,
    }

    return som_data, som_labels


def _font(size: int) -> ImageFont.ImageFont:
    """Get font for drawing labels."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    """Get text bounding box."""
    try:
        return draw.textbbox((0, 0), text, font=font)
    except Exception:
        width, height = draw.textsize(text, font=font)
        return 0, 0, int(width), int(height)


def _draw_som_label(
    draw: ImageDraw.ImageDraw,
    center: Any,
    label_text: str,
    width: int,
    height: int,
    role: str,
) -> None:
    """Draw numbered SoM label circle."""
    try:
        cx = float(center[0])
        cy = float(center[1])
    except Exception:
        return

    label_font = _font(max(int(min(width, height) * 0.04), 18))
    radius = max(int(min(width, height) * 0.026), 15)
    bbox = _text_bbox(draw, label_text, label_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x0 = int(round(cx - radius))
    y0 = int(round(cy - radius))
    x1 = int(round(cx + radius))
    y1 = int(round(cy + radius))

    if role == "forbidden":
        fill = (127, 29, 29, 235)
        outline = (254, 226, 226, 250)
    elif role == "conditional_traversable":
        fill = (146, 64, 14, 235)
        outline = (254, 243, 199, 250)
    else:
        fill = (15, 23, 42, 230)
        outline = (255, 255, 255, 245)

    draw.ellipse((x0, y0, x1, y1), fill=fill, outline=outline, width=3)
    draw.text(
        (int(round(cx - text_w / 2)), int(round(cy - text_h / 2 - bbox[1]))),
        label_text,
        fill=(255, 255, 255, 255),
        font=label_font,
    )


def draw_som_overlay(
    image_path: Path,
    som_data: dict[str, Any],
    som_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Draw SoM visualization overlay: colored regions + numbered labels.

    参考 mcp_server/som/overlay.py 的渲染方式：
    - 连通分量着色（不是边界框！）
    - 区域中心放置编号标记
    """
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    regions = som_data.get("regions", [])
    preferred_ids = set(som_data.get("preferred", []))
    conditional_ids = set(som_data.get("conditional", []))
    forbidden_ids = set(som_data.get("forbidden", []))

    label_mapping: list[dict[str, Any]] = []
    label_counter = 1
    current_label_id = 1

    # 处理 preferred 区域
    for region in regions:
        if region["region_id"] not in preferred_ids:
            continue
        region_mask = som_labels == current_label_id
        if not bool(np.any(region_mask)):
            current_label_id += 1
            continue

        color_idx = (label_counter - 1) % len(_SOM_PALETTE)
        color = _SOM_PALETTE[color_idx]
        # Use same approach as draw_segmentation_overlay
        mask_img = Image.fromarray((region_mask * 255).astype(np.uint8), mode="L")
        color_layer = Image.new("RGBA", img.size, color + (150,))
        overlay.paste(color_layer, mask=mask_img)

        label_mapping.append({
            "label": label_counter,
            "region_id": region["region_id"],
            "region_role": "traversable",
            "center_uv": region["centroid_xy"],
        })
        label_counter += 1
        current_label_id += 1

    # 处理 conditional 区域
    for region in regions:
        if region["region_id"] not in conditional_ids:
            continue
        region_mask = som_labels == current_label_id
        if not bool(np.any(region_mask)):
            current_label_id += 1
            continue

        color_idx = (label_counter - 1) % len(_SOM_CONDITIONAL_PALETTE)
        color = _SOM_CONDITIONAL_PALETTE[color_idx]
        mask_img = Image.fromarray((region_mask * 255).astype(np.uint8), mode="L")
        color_layer = Image.new("RGBA", img.size, color + (150,))
        overlay.paste(color_layer, mask=mask_img)

        label_mapping.append({
            "label": label_counter,
            "region_id": region["region_id"],
            "region_role": "conditional_traversable",
            "center_uv": region["centroid_xy"],
        })
        label_counter += 1
        current_label_id += 1

    # 处理 forbidden 区域
    for region in regions:
        if region["region_id"] not in forbidden_ids:
            continue
        region_mask = som_labels == current_label_id
        if not bool(np.any(region_mask)):
            current_label_id += 1
            continue

        color_idx = (label_counter - 1) % len(_SOM_FORBIDDEN_PALETTE)
        color = _SOM_FORBIDDEN_PALETTE[color_idx]
        mask_img = Image.fromarray((region_mask * 255).astype(np.uint8), mode="L")
        color_layer = Image.new("RGBA", img.size, color + (150,))
        overlay.paste(color_layer, mask=mask_img)

        label_mapping.append({
            "label": label_counter,
            "region_id": region["region_id"],
            "region_role": "forbidden",
            "center_uv": region["centroid_xy"],
        })
        label_counter += 1
        current_label_id += 1

    combined = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(combined)

    for item in label_mapping:
        _draw_som_label(
            draw,
            center=item["center_uv"],
            label_text=str(int(item["label"])),
            width=width,
            height=height,
            role=item["region_role"],
        )

    combined.convert("RGB").save(output_path, "JPEG", quality=95)


# ============================================================================
# Route Support - BEV 意图感知支持热力图
# ============================================================================
def generate_route_support(
    frame: TrajectoryFrame,
    frames_by_name: dict[str, TrajectoryFrame],
    mask: np.ndarray,
    class_mapping: dict[int, str],
    class_config: dict[str, list[str]],
    image_width: int,
    image_height: int,
    sigma_px: float = 24.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate route support heatmap from trajectory.

    实现:
        1. 找出所有可通行区域
        2. 沿轨迹生成高斯热力图
        3. 只在可通行区域内保留热力值

    返回:
        route_support_heatmap: [1, H, W] float32 张量
        metadata: 热力图元数据
    """
    height, width = mask.shape

    navigable_mask = np.zeros((height, width), dtype=bool)
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        class_name = class_mapping.get(class_id, "unknown")
        affordance = _get_affordance(class_name, class_config)
        if affordance in ["preferred_navigable", "conditional_navigable"]:
            navigable_mask |= mask == class_id

    heatmap = np.zeros((1, height, width), dtype=np.float32)

    frame_idx = frame.frame_idx
    trajectory_points = []
    for offset in [-10, -5, 0, 5, 10]:
        neighbor_idx = frame_idx + offset
        neighbor_name = f"{neighbor_idx:06d}"
        neighbor = frames_by_name.get(neighbor_name)
        if neighbor is not None:
            trajectory_points.append((neighbor.lat, neighbor.lng))

    if len(trajectory_points) >= 2:
        y_coords, x_coords = np.indices((height, width))
        center_x, center_y = width * 0.5, height * 0.85

        for i in range(len(trajectory_points) - 1):
            t = i / (len(trajectory_points) - 1)
            point_x = center_x + (t - 0.5) * width * 0.3
            point_y = center_y - t * height * 0.5

            dist_sq = (x_coords - point_x) ** 2 + (y_coords - point_y) ** 2
            gaussian = np.exp(-dist_sq / (2 * sigma_px ** 2))
            heatmap[0] = np.maximum(heatmap[0], gaussian)

    heatmap[0] = heatmap[0] * navigable_mask.astype(np.float32)

    max_val = heatmap[0].max()
    if max_val > 0:
        heatmap[0] = heatmap[0] / max_val

    metadata = {
        "schema_version": "sidewalk_route_support_v1",
        "frame_idx": frame_idx,
        "image_width": image_width,
        "image_height": image_height,
        "heatmap_path": "",
        "source": "gt_trajectory_gaussian_x_segmentation",
        "bev_intent": "go_straight",
        "route_trace": [],
        "sigma_px": sigma_px,
        "valid": bool(np.any(navigable_mask)),
    }

    return heatmap, metadata


def draw_route_support_overlay(
    image_path: Path,
    route_support: np.ndarray,
    output_path: Path,
) -> None:
    """Draw route support heatmap visualization."""
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    heatmap = route_support[0]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    from matplotlib import cm
    colormap = cm.get_cmap("Blues")
    colored_heatmap = colormap(heatmap_uint8 / 255.0)
    colored_heatmap_rgb = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    colored_heatmap_rgba = np.dstack([colored_heatmap_rgb, (heatmap_uint8 * 0.7).astype(np.uint8)])

    overlay = Image.fromarray(colored_heatmap_rgba, "RGBA")
    overlay = overlay.resize((width, height), Image.Resampling.LANCZOS)

    combined = Image.alpha_composite(img, overlay)
    combined.convert("RGB").save(output_path, "JPEG", quality=95)


# ============================================================================
# 帧处理主流程
# ============================================================================
def process_frame(
    frame: TrajectoryFrame,
    frames_by_name: dict[str, TrajectoryFrame],
    sam3_output_dir: Path,
    sequence_name: str,
    camera_view: str,
    clip_dir: Path,
    class_config: dict[str, list[str]],
    sigma_px: float = 24.0,
) -> dict[str, Any]:
    """Process single frame to generate all Phase2 artifacts."""
    frame_name = frame.frame_name
    result = {
        "frame_name": frame_name,
        "frame_idx": frame.frame_idx,
        "image_path": str(frame.image_path),
        "status": "error",
        "error": None,
    }

    try:
        base_dir = sam3_output_dir / sequence_name / camera_view
        if not base_dir.exists():
            base_dir = sam3_output_dir / camera_view

        mask_path = base_dir / f"{frame_name}_segmentation.npy"
        class_mapping_path = base_dir / "class_mapping.json"

        if not mask_path.exists() or not class_mapping_path.exists():
            result["error"] = "missing_segmentation_artifacts"
            return result

        class_mapping = _load_class_mapping(class_mapping_path)

        with Image.open(frame.image_path) as img:
            image_width, image_height = img.size

        fpv_seg_dir = clip_dir / "fpv_segmentation"
        som_dir = clip_dir / "som"
        route_support_dir = clip_dir / "route_support"

        _ensure_dir(fpv_seg_dir)
        _ensure_dir(som_dir)
        _ensure_dir(route_support_dir)

        seg_summary, mask_uint16, affordance_mask = process_sam3_mask(
            mask_path=mask_path,
            class_mapping=class_mapping,
            class_config=class_config,
            image_width=image_width,
            image_height=image_height,
        )
        seg_summary["frame_idx"] = frame.frame_idx

        frame_idx_str = f"{frame.frame_idx:06d}"

        seg_summary["mask_path"] = f"{frame_idx_str}_mask.npy"
        seg_summary["affordance_path"] = f"{frame_idx_str}_affordance.npy"
        seg_summary["overlay_path"] = f"{frame_idx_str}_overlay.jpg"

        seg_json_path = fpv_seg_dir / f"{frame_idx_str}_summary.json"
        seg_mask_path = fpv_seg_dir / f"{frame_idx_str}_mask.npy"
        seg_affordance_path = fpv_seg_dir / f"{frame_idx_str}_affordance.npy"
        seg_overlay_path = fpv_seg_dir / f"{frame_idx_str}_overlay.jpg"

        seg_json_path.write_text(json.dumps(seg_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(str(seg_mask_path), mask_uint16.astype(np.uint16))
        np.save(str(seg_affordance_path), affordance_mask.astype(np.float32))
        draw_segmentation_overlay(frame.image_path, affordance_mask, seg_overlay_path)

        som_data, som_labels = build_som_from_segmentation(
            mask=mask_uint16,
            class_mapping=class_mapping,
            class_config=class_config,
            frame_idx=frame.frame_idx,
            image_width=image_width,
            image_height=image_height,
        )

        som_json_path = som_dir / f"{frame_idx_str}_som.json"
        som_npy_path = som_dir / f"{frame_idx_str}_som.npy"
        som_overlay_path = som_dir / f"{frame_idx_str}_overlay.jpg"

        som_json_path.write_text(json.dumps(som_data, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(str(som_npy_path), som_labels.astype(np.int32))
        draw_som_overlay(frame.image_path, som_data, som_labels, som_overlay_path)

        route_support_heatmap, route_support_metadata = generate_route_support(
            frame=frame,
            frames_by_name=frames_by_name,
            mask=mask_uint16,
            class_mapping=class_mapping,
            class_config=class_config,
            image_width=image_width,
            image_height=image_height,
            sigma_px=sigma_px,
        )

        route_support_metadata["heatmap_path"] = f"{frame_idx_str}_route_support.npy"
        route_support_metadata["route_trace"] = som_data["route_trace"]

        route_support_npy_path = route_support_dir / f"{frame_idx_str}_route_support.npy"
        route_support_json_path = route_support_dir / f"{frame_idx_str}_route_support.json"
        route_support_overlay_path = route_support_dir / f"{frame_idx_str}_overlay.jpg"

        np.save(str(route_support_npy_path), route_support_heatmap.astype(np.float32))
        route_support_json_path.write_text(json.dumps(route_support_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        draw_route_support_overlay(frame.image_path, route_support_heatmap, route_support_overlay_path)

        result["status"] = "ok"
        result["segmentation"] = {
            "json_path": str(seg_json_path),
            "mask_path": str(seg_mask_path),
            "affordance_path": str(seg_affordance_path),
            "overlay_path": str(seg_overlay_path),
            "class_count": len(seg_summary["classes"]),
        }
        result["som"] = {
            "json_path": str(som_json_path),
            "npy_path": str(som_npy_path),
            "overlay_path": str(som_overlay_path),
            "region_count": len(som_data["regions"]),
        }
        result["route_support"] = {
            "npy_path": str(route_support_npy_path),
            "json_path": str(route_support_json_path),
            "overlay_path": str(route_support_overlay_path),
            "valid": route_support_metadata["valid"],
        }

        print(f"[OK] Processed frame {frame_name}: {len(seg_summary['classes'])} classes, {len(som_data['regions'])} regions")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"[ERROR] Failed to process frame {frame_name}: {exc}")
        import traceback
        traceback.print_exc()

    return result


# ============================================================================
# Debug 视频生成
# ============================================================================
def generate_debug_videos(
    clip_dir: Path,
    frame_names: list[str],
    video_fps: int = 10,
) -> dict[str, str]:
    """Generate debug visualization videos."""
    debug_video_dir = clip_dir / "debug_video"
    _ensure_dir(debug_video_dir)

    results = {}

    try:
        first_frame = frame_names[0] if frame_names else None
        if not first_frame:
            return results

        fpv_seg_dir = clip_dir / "fpv_segmentation"
        som_dir = clip_dir / "som"
        route_support_dir = clip_dir / "route_support"

        first_overlay = fpv_seg_dir / f"{first_frame}_overlay.jpg"
        if first_overlay.exists():
            with Image.open(first_overlay) as img:
                width, height = img.size

            seg_video_path = debug_video_dir / "segmentation_overlay.mp4"
            seg_input_pattern = str(fpv_seg_dir / "%06d_overlay.jpg")
            cmd_seg = [
                "ffmpeg", "-y", "-framerate", str(video_fps), "-i", seg_input_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"scale={width}:{height}",
                str(seg_video_path),
            ]
            subprocess.run(cmd_seg, check=True, capture_output=True)
            results["segmentation_video"] = str(seg_video_path)

            som_video_path = debug_video_dir / "som_overlay.mp4"
            som_input_pattern = str(som_dir / "%06d_overlay.jpg")
            cmd_som = [
                "ffmpeg", "-y", "-framerate", str(video_fps), "-i", som_input_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"scale={width}:{height}",
                str(som_video_path),
            ]
            subprocess.run(cmd_som, check=True, capture_output=True)
            results["som_video"] = str(som_video_path)

            route_support_video_path = debug_video_dir / "route_support_overlay.mp4"
            route_input_pattern = str(route_support_dir / "%06d_overlay.jpg")
            cmd_route = [
                "ffmpeg", "-y", "-framerate", str(video_fps), "-i", route_input_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"scale={width}:{height}",
                str(route_support_video_path),
            ]
            subprocess.run(cmd_route, check=True, capture_output=True)
            results["route_support_video"] = str(route_support_video_path)

            print(f"[OK] Generated debug videos in {debug_video_dir}")

    except Exception as exc:
        print(f"[WARNING] Failed to generate debug videos: {exc}")

    return results


# ============================================================================
# 数据集发现
# ============================================================================
def find_clips(base_dir: Path) -> list[Path]:
    """Find all clip directories under base_dir."""
    clips = []

    has_mp4 = (base_dir / "front.mp4").exists()
    has_gps = (base_dir / "frame_gps.npy").exists()
    if has_mp4 and has_gps:
        return [base_dir]

    for item in sorted(base_dir.iterdir()):
        if not item.is_dir():
            continue
        has_mp4 = (item / "front.mp4").exists()
        has_gps = (item / "frame_gps.npy").exists()
        if has_mp4 and has_gps:
            clips.append(item)
        else:
            for subitem in sorted(item.iterdir()):
                if not subitem.is_dir():
                    continue
                has_mp4_sub = (subitem / "front.mp4").exists()
                has_gps_sub = (subitem / "frame_gps.npy").exists()
                if has_mp4_sub and has_gps_sub:
                    clips.append(subitem)

    return clips


# ============================================================================
# 单个 Clip 处理
# ============================================================================
def process_single_clip(clip_dir: Path, args: argparse.Namespace, class_config: dict[str, list[str]]) -> dict[str, Any]:
    """Process single clip directory."""
    print(f"\n{'='*60}")
    print(f"Processing clip: {clip_dir.name}")
    print(f"{'='*60}")

    if args.skip_existing:
        fpv_seg_dir = clip_dir / "fpv_segmentation"
        if fpv_seg_dir.exists() and len(list(fpv_seg_dir.glob("*.json"))) > 0:
            print(f"Skipping existing clip: {clip_dir.name}")
            return {"clip": clip_dir.name, "success": True, "skipped": True}

    mp4_path = clip_dir / "front.mp4"
    if not mp4_path.exists():
        raise RuntimeError(f"front.mp4 not found: {mp4_path}")

    images_dir = clip_dir / "images"
    if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) == 0:
        _extract_frames_from_mp4(mp4_path, clip_dir, sample_fps=args.sample_fps)

    frames_by_name = _load_trajectory_from_npy(
        clip_dir=clip_dir,
        images_dir=images_dir,
        sample_fps=args.sample_fps,
    )

    selected = _iter_selected_frame_names(
        images_dir=images_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_step=args.frame_step,
    )

    if not selected:
        print("No frames found to process!")
        return {"clip": clip_dir.name, "success": False, "reason": "no_frames"}

    print(f"Found {len(selected)} frames")

    sequence_name = clip_dir.name
    camera_view = "camera_front"

    sam3_output_dir = clip_dir / "sam3_intermediate"
    work_dir = clip_dir / "sam3_work"

    if args.run_sam3:
        import time
        sam3_start = time.time()
        _run_sam3_inference_on_images(
            images_dir=images_dir,
            selected_frame_names=selected,
            sam3_output_base_dir=sam3_output_dir,
            sequence_name=sequence_name,
            camera_view=camera_view,
            sam3_env_name=args.sam3_env_name,
            sam3_inference_script=Path(args.sam3_inference_script),
            gpus=args.gpus,
            min_score=args.min_score,
            finetuned=args.finetuned or (args.checkpoint_path is not None),
            checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
            work_dir=work_dir,
        )
        sam3_elapsed = time.time() - sam3_start
        print(f"[TIME] SAM3 inference: {sam3_elapsed:.2f}s")

    frames_processed = []
    for frame_name in selected:
        frame = frames_by_name.get(frame_name)
        if frame is None:
            continue
        result = process_frame(
            frame=frame,
            frames_by_name=frames_by_name,
            sam3_output_dir=sam3_output_dir,
            sequence_name=sequence_name,
            camera_view=camera_view,
            clip_dir=clip_dir,
            class_config=class_config,
            sigma_px=args.sigma_px,
        )
        frames_processed.append(result)

    video_results = {}
    if args.generate_video:
        success_frames = [f for f in frames_processed if f["status"] == "ok"]
        if success_frames:
            frame_indices = []
            for f in success_frames:
                idx = f.get("frame_idx", -1)
                if idx >= 0:
                    frame_indices.append(f"{idx:06d}")
                else:
                    frame_indices.append(f["frame_name"])
            if frame_indices:
                video_results = generate_debug_videos(
                    clip_dir=clip_dir,
                    frame_names=frame_indices,
                    video_fps=args.video_fps,
                )

    if args.cleanup_temp:
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        if sam3_output_dir.exists():
            shutil.rmtree(sam3_output_dir)
        print(f"[CLEANUP] Removed temporary directories")

    success_count = sum(1 for f in frames_processed if f["status"] == "ok")
    print(f"\nCompleted {clip_dir.name}: {success_count}/{len(frames_processed)} frames successful")

    manifest = {
        "clip": clip_dir.name,
        "frames_total": len(frames_processed),
        "frames_success": success_count,
        "frames_failed": len(frames_processed) - success_count,
        "frames": frames_processed,
        "video_results": video_results,
    }
    manifest_path = clip_dir / "processing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "clip": clip_dir.name,
        "success": success_count == len(frames_processed),
        "frames_total": len(frames_processed),
        "frames_success": success_count,
    }


# ============================================================================
# 命令行接口
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SAM3 + SoM + RouteSupport 标注工具 - Phase2 完整格式")
    parser.add_argument("--clip-dir", help="单个 clip 目录")
    parser.add_argument("--sidewalk-root", help="Sidewalk 根目录（批量处理）")
    parser.add_argument("--sidewalk-dir", help="Sidewalk 根目录（批量处理，别名）")
    parser.add_argument("--run-sam3", action="store_true", help="运行 SAM3 推理")
    parser.add_argument("--sam3-env-name", default="sam3", help="SAM3 conda 环境名")
    parser.add_argument("--sam3-inference-script", default="/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/batch_sam3_demo_260124.py", help="SAM3 推理脚本路径")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU IDs")
    parser.add_argument("--min-score", type=float, default=0.3, help="SAM3 最小分数阈值")
    parser.add_argument("--checkpoint-path", default=None, help="SAM3 checkpoint 路径")
    parser.add_argument("--finetuned", action="store_true", help="使用 finetuned checkpoint")
    parser.add_argument("--frame-start", type=int, default=None, help="起始帧 ID")
    parser.add_argument("--frame-end", type=int, default=None, help="结束帧 ID")
    parser.add_argument("--frame-step", type=int, default=1, help="帧采样步长")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="从 front.mp4 采样的帧率")
    parser.add_argument("--generate-video", action="store_true", help="生成 debug_video mp4")
    parser.add_argument("--video-fps", type=int, default=10, help="视频帧率")
    parser.add_argument("--sigma-px", type=float, default=24.0, help="route_support 高斯 sigma")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有结果的 clip")
    parser.add_argument("--cleanup-temp", action="store_true", help="处理完成后删除临时 images/ 和 sam3_work/ 目录")
    parser.add_argument("--mode", default="foot", help="模式（占位参数，兼容旧脚本）")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    class_config = DEFAULT_CLASS_CONFIG

    sidewalk_root_arg = args.sidewalk_root or args.sidewalk_dir

    if args.clip_dir:
        clip_dir = Path(args.clip_dir).expanduser().resolve()
        if not clip_dir.exists():
            raise RuntimeError(f"Clip directory not found: {clip_dir}")
        result = process_single_clip(clip_dir, args, class_config)
        print(f"\nResult: {result}")

    elif sidewalk_root_arg:
        sidewalk_root = Path(sidewalk_root_arg).expanduser().resolve()
        if not sidewalk_root.exists():
            raise RuntimeError(f"Sidewalk root directory not found: {sidewalk_root}")

        clips = find_clips(sidewalk_root)

        if not clips:
            print("No clips found!")
            return

        print(f"Found {len(clips)} clips:")
        for i, clip in enumerate(clips, 1):
            print(f"  {i}. {clip.name}")

        all_results = []
        for clip_dir in clips:
            try:
                result = process_single_clip(clip_dir, args, class_config)
                all_results.append(result)
            except Exception as exc:
                print(f"[ERROR] Failed to process clip {clip_dir.name}: {exc}")
                import traceback
                traceback.print_exc()
                all_results.append({"clip": clip_dir.name, "success": False, "error": str(exc)})

        print(f"\n{'='*60}")
        print(f"All done!")
        print(f"{'='*60}")
        print("\nProcessing summary:")
        for result in all_results:
            status = "✓ Success" if result["success"] else "✗ Failed"
            if result.get("skipped"):
                status = "⏭ Skipped"
            details = f"{result.get('frames_success', 0)}/{result.get('frames_total', 0)} frames" if "frames_total" in result else ""
            print(f"  {result['clip']}: {status} {details}")

        batch_summary = {
            "sidewalk_root": str(sidewalk_root),
            "results": all_results,
        }
        batch_summary_path = sidewalk_root / "batch_processing_summary.json"
        batch_summary_path.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nDetailed summary saved to: {batch_summary_path}")

    else:
        print("Please specify either --clip-dir or --sidewalk-root/--sidewalk-dir")


if __name__ == "__main__":
    main()
