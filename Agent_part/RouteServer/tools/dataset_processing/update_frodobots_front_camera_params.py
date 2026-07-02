#!/usr/bin/env python3
"""Update FrodoBots prepared clips with a public approximate front-camera K.

The source calibration is reported for Earth Rover Zero front camera images at
1024x576.  Prepared Berkeley-FrodoBots clips in this workspace are 224x128, so
this script scales fx/fy/cx/cy for pure resize and records the OpenCV distortion
coefficients as metadata.  The current Phase1 projection code consumes K only;
D is retained for downstream undistortion experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


K_1024X576 = np.asarray(
    [
        [407.860, 0.0, 533.301],
        [0.0, 407.866, 278.699],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

D_OPENCV = [-0.2172, 0.0537, 0.001853, -0.002105, -0.006000]


def scaled_k(width: int, height: int) -> list[list[float]]:
    sx = float(width) / 1024.0
    sy = float(height) / 576.0
    k = K_1024X576.copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return [[round(float(v), 6) for v in row] for row in k.tolist()]


def load_clip_dirs(root: Path) -> list[Path]:
    if (root / "structured_windows_report.json").is_file():
        report = json.loads((root / "structured_windows_report.json").read_text(encoding="utf-8"))
        dirs = []
        for row in report.get("selected", []):
            raw = row.get("original_clip_dir") or row.get("clip_dir")
            if raw:
                dirs.append(Path(str(raw)).expanduser().resolve())
        return sorted(set(path for path in dirs if path.is_dir()))
    if (root / "clips_structured").is_dir():
        root = root / "clips_structured"
    return sorted(set(path.resolve() for path in root.iterdir() if path.is_dir()))


def read_existing(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-root", required=True)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--camera-height-m", type=float, default=0.6)
    parser.add_argument("--backup-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    root = Path(args.clips_root).expanduser()
    clip_dirs = load_clip_dirs(root)
    payload = {
        "schema_version": "frodobots_public_front_camera_params_v1",
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "camera_height_meters": float(args.camera_height_m),
        "K": scaled_k(int(args.image_width), int(args.image_height)),
        "distortion_model": "opencv_pinhole_radial_tangential",
        "D": D_OPENCV,
        "source": "zhang2024frodobots_public_front_camera_calibration_scaled_from_1024x576",
        "source_resolution": [1024, 576],
        "target_resolution": [int(args.image_width), int(args.image_height)],
        "note": (
            "Approximate public Earth Rover Zero front-camera calibration. "
            "Not an official Berkeley-FrodoBots-7K per-robot calibration. "
            "Current Phase1 projection consumes K only; D is recorded for optional OpenCV undistort."
        ),
    }

    updated = 0
    backed_up = 0
    skipped = 0
    for clip_dir in clip_dirs:
        target = clip_dir / "camera_params.json"
        existing = read_existing(target)
        if existing.get("source") == payload["source"] and existing.get("K") == payload["K"]:
            skipped += 1
            continue
        if target.is_file() and bool(args.backup_existing):
            backup = clip_dir / "camera_params.approx_hfov90_backup.json"
            if not backup.exists():
                backup.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
                backed_up += 1
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1

    report = {
        "schema_version": "frodobots_camera_params_update_report_v1",
        "clips_root": str(root),
        "clip_dirs": len(clip_dirs),
        "updated": updated,
        "skipped_already_updated": skipped,
        "backed_up": backed_up,
        "camera_params": payload,
    }
    report_path = Path(args.report_json).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
