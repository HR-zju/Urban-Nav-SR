"""Frame and artifact IO retained for offline historical dataset review."""
from __future__ import annotations
import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

@dataclass(frozen=True)
class TrajectoryFrame:
    frame_name: str
    lat: float
    lng: float
    heading_deg: float | None
    image_path: Path
    roll_deg: float | None = None
    pitch_deg: float | None = None
    attitude_source: str | None = None


def build_navigation_path_gps(
    *,
    selected_frame_names: list[str],
    frames_by_name: dict[str, TrajectoryFrame],
) -> list[dict[str, float]]:
    """Collect the offline dataset trajectory for BEV map visualization."""
    path: list[dict[str, float]] = []
    for frame_name in selected_frame_names:
        frame = frames_by_name.get(frame_name)
        if frame is None:
            continue
        path.append({"lat": float(frame.lat), "lng": float(frame.lng)})
    return path


def _load_trajectory_frames(*, trajectory_path: Path, images_dir: Path) -> dict[str, TrajectoryFrame]:
    raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"trajectory.json must be a list, got: {type(raw)}")

    frames: dict[str, TrajectoryFrame] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        image_path_raw = str(item.get("image_path") or "").strip()
        if not image_path_raw:
            continue
        frame_name = Path(image_path_raw).stem
        if not frame_name:
            continue
        lat = float(item.get("lat"))
        lng = float(item.get("lng"))
        heading = item.get("heading_deg")
        heading_deg = float(heading) if heading is not None else None
        roll = item.get("roll_deg")
        pitch = item.get("pitch_deg")
        attitude_source = item.get("attitude_source")
        resolved_image = images_dir / f"{frame_name}.jpg"
        if not resolved_image.exists():
            # Try original extension if it's not jpg.
            resolved_image = images_dir / Path(image_path_raw).name
        frames[frame_name] = TrajectoryFrame(
            frame_name=str(frame_name),
            lat=lat,
            lng=lng,
            heading_deg=heading_deg,
            image_path=resolved_image,
            roll_deg=float(roll) if roll is not None else None,
            pitch_deg=float(pitch) if pitch is not None else None,
            attitude_source=str(attitude_source) if attitude_source else None,
        )
    if not frames:
        raise RuntimeError(f"No usable frames found in trajectory: {trajectory_path}")
    return frames


def _iter_selected_frame_names(
    *,
    images_dir: Path,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
) -> list[str]:
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


def _write_data_url_image(data_url: str, output_path_without_suffix: Path) -> Path | None:
    raw = str(data_url or "").strip()
    if not raw.startswith("data:image/") or "," not in raw:
        return None
    header, encoded = raw.split(",", 1)
    mime = header.split(";", 1)[0].replace("data:", "")
    ext = ".jpg" if mime in {"image/jpeg", "image/jpg"} else ".png" if mime == "image/png" else ".img"
    output_path = Path(str(output_path_without_suffix) + ext)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(encoded))
    return output_path
