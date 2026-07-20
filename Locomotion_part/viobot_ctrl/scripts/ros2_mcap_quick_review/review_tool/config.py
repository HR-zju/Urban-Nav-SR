from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {"data_root": "/mnt/nas/rosbags", "work_dir": "./work"},
    "scan": {"include_standalone_mcap": True, "include_standalone_db3": False},
    "reader": {},
    "topics": {
        "camera": {
            "names": [],
            "supported_types": ["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"],
            "name_hints": ["front", "camera", "image"],
        },
        "gps": {
            "names": [],
            "supported_types": [
                "sensor_msgs/msg/NavSatFix",
                "nav_msgs/msg/Odometry",
                "geometry_msgs/msg/PoseStamped",
            ],
            "name_hints": ["gps", "gnss", "fused"],
            "coordinate_mode": "auto",
            "fields": {"latitude": "", "longitude": "", "x": "", "y": "", "z": ""},
        },
        "vio": {
            "names": [],
            "supported_types": ["nav_msgs/msg/Odometry", "geometry_msgs/msg/PoseStamped"],
            "name_hints": ["vio", "odom", "odometry"],
            "heading_yaw_offset_deg": 0.0,
            "fields": {
                "x": "", "y": "", "z": "", "qx": "", "qy": "", "qz": "", "qw": ""
            },
        },
    },
    "timestamps": {
        "source": "auto",
        "minimum_header_valid_ratio": 0.80,
        "minimum_header_overlap_ratio": 0.50,
    },
    "preprocess": {
        "output_width": 1280,
        "output_height": 720,
        "output_fps": 10,
        "playback_speed": 2.0,
        "max_source_camera_fps": 15,
        "max_gps_fps": 10,
        "max_vio_fps": 20,
        "jpeg_quality": 85,
        "max_review_duration_sec": 180,
        "camera_max_gap_sec": 0.25,
        "gps_max_gap_sec": 1.0,
        "vio_max_gap_sec": 0.2,
        "ffmpeg_crf": 23,
        "ffmpeg_preset": "veryfast",
    },
    "review": {
        "bad_reasons": [
            "图像异常", "GPS异常", "VIO异常", "时间不同步",
            "数据缺失", "场景无效", "文件损坏", "其他",
        ]
    },
}

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, user_config)
    base_dir = config_path.parent
    for key in ("data_root", "work_dir"):
        p = Path(cfg["paths"][key]).expanduser()
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        cfg["paths"][key] = str(p)
    source = str(cfg["timestamps"]["source"]).lower()
    if source not in {"auto", "header", "record"}:
        raise ValueError("timestamps.source must be auto, header, or record")
    cfg["timestamps"]["source"] = source
    work_dir = Path(cfg["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "cache").mkdir(parents=True, exist_ok=True)
    return cfg
