from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _normalize_timestamp_ms(value: Any) -> int:
    raw = float(value)
    # Front camera timestamps are in seconds, GPS usually in milliseconds.
    if raw < 100_000_000_000:
        raw *= 1000.0
    return int(round(raw))


def _find_single_csv(ride_dir: Path, pattern: str) -> Path:
    matches = sorted(ride_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched pattern '{pattern}' under {ride_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files matched pattern '{pattern}' under {ride_dir}: {matches}")
    return matches[0]


def _nearest_index(sorted_pairs: list[tuple[int, Any]], target_ms: int) -> int:
    if not sorted_pairs:
        raise RuntimeError("No data points available for nearest-neighbor lookup")
    lo = 0
    hi = len(sorted_pairs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_pairs[mid][0] < target_ms:
            lo = mid + 1
        else:
            hi = mid
    right = lo
    left = max(right - 1, 0)
    if abs(sorted_pairs[left][0] - target_ms) <= abs(sorted_pairs[right][0] - target_ms):
        return left
    return right


def _parse_compass_heading_deg(compass_payload: str) -> float | None:
    text = str(compass_payload or "").strip()
    if not text:
        return None
    payload: Any | None = None
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if payload is None:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return None
    try:
        mx = float(first[0])
        my = float(first[1])
    except Exception:
        return None
    # Convert magnetometer x/y to heading, normalized to [0, 360).
    return (math.degrees(math.atan2(my, mx)) + 360.0) % 360.0


@dataclass(frozen=True)
class FramePose:
    frame_id: int
    frame_timestamp_ms: int
    latitude: float
    longitude: float
    gps_timestamp_ms: int
    heading_deg: float | None
    heading_timestamp_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "frame_timestamp_ms": int(self.frame_timestamp_ms),
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "gps_timestamp_ms": int(self.gps_timestamp_ms),
            "heading_deg": None if self.heading_deg is None else float(self.heading_deg),
            "heading_timestamp_ms": (
                None if self.heading_timestamp_ms is None else int(self.heading_timestamp_ms)
            ),
        }


class RidePoseReader:
    """Read per-frame pose from UrbanNav ride exports.

    Expected files under a ride directory:
    - front_camera_timestamps_*.csv (frame_id,timestamp)
    - gps_data_*.csv (latitude,longitude,timestamp)
    - imu_data_*.csv (optional, heading from compass)
    """

    def __init__(
        self,
        *,
        frame_timestamps: dict[int, int],
        gps_samples: list[tuple[int, float, float]],
        heading_samples: list[tuple[int, float]],
    ) -> None:
        self._frame_timestamps = dict(frame_timestamps)
        self._gps_samples = sorted(gps_samples, key=lambda x: x[0])
        self._heading_samples = sorted(heading_samples, key=lambda x: x[0])
        if not self._frame_timestamps:
            raise RuntimeError("No frame timestamps were loaded")
        if not self._gps_samples:
            raise RuntimeError("No GPS samples were loaded")

    @classmethod
    def from_ride_directory(cls, ride_dir: str | Path, *, camera: str = "front") -> RidePoseReader:
        base = Path(ride_dir).expanduser()
        if not base.exists():
            raise FileNotFoundError(f"Ride directory does not exist: {base}")
        if not base.is_dir():
            raise NotADirectoryError(f"Ride path is not a directory: {base}")

        frame_file = _find_single_csv(base, f"{camera}_camera_timestamps_*.csv")
        gps_file = _find_single_csv(base, "gps_data_*.csv")
        imu_matches = sorted(base.glob("imu_data_*.csv"))
        imu_file = imu_matches[0] if imu_matches else None

        frame_timestamps: dict[int, int] = {}
        with frame_file.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                frame_id = int(str(row.get("frame_id", "")).strip())
                timestamp_ms = _normalize_timestamp_ms(row.get("timestamp"))
                frame_timestamps[frame_id] = timestamp_ms

        gps_samples: list[tuple[int, float, float]] = []
        with gps_file.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                timestamp_ms = _normalize_timestamp_ms(row.get("timestamp"))
                latitude = float(row.get("latitude"))
                longitude = float(row.get("longitude"))
                gps_samples.append((timestamp_ms, latitude, longitude))

        heading_samples: list[tuple[int, float]] = []
        if imu_file is not None:
            with imu_file.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    heading_deg = _parse_compass_heading_deg(str(row.get("compass", "")))
                    if heading_deg is None:
                        continue
                    try:
                        timestamp_ms = _normalize_timestamp_ms(row.get("timestamp"))
                    except Exception:
                        continue
                    heading_samples.append((timestamp_ms, heading_deg))

        return cls(
            frame_timestamps=frame_timestamps,
            gps_samples=gps_samples,
            heading_samples=heading_samples,
        )

    def frame_timestamp_ms(self, frame_id: int) -> int:
        if frame_id not in self._frame_timestamps:
            raise KeyError(f"frame_id {frame_id} not found in frame timestamps")
        return self._frame_timestamps[frame_id]

    def get_frame_pose(self, frame_id: int) -> FramePose:
        target_ms = self.frame_timestamp_ms(frame_id)

        gps_pairs = [(ts, (lat, lon)) for ts, lat, lon in self._gps_samples]
        gps_index = _nearest_index(gps_pairs, target_ms)
        gps_timestamp_ms, (latitude, longitude) = gps_pairs[gps_index]

        heading_deg: float | None = None
        heading_timestamp_ms: int | None = None
        if self._heading_samples:
            heading_index = _nearest_index(self._heading_samples, target_ms)
            heading_timestamp_ms, heading_deg = self._heading_samples[heading_index]

        return FramePose(
            frame_id=int(frame_id),
            frame_timestamp_ms=int(target_ms),
            latitude=float(latitude),
            longitude=float(longitude),
            gps_timestamp_ms=int(gps_timestamp_ms),
            heading_deg=None if heading_deg is None else float(heading_deg),
            heading_timestamp_ms=None if heading_timestamp_ms is None else int(heading_timestamp_ms),
        )


def read_frame_pose(ride_dir: str | Path, frame_id: int, *, camera: str = "front") -> dict[str, Any]:
    """Convenience interface for one-shot frame pose lookup."""
    reader = RidePoseReader.from_ride_directory(ride_dir, camera=camera)
    return reader.get_frame_pose(frame_id).to_dict()
