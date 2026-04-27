from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "services" / "ride_pose_service.py"
_SPEC = importlib.util.spec_from_file_location("ride_pose_service", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

RidePoseReader = _MODULE.RidePoseReader
read_frame_pose = _MODULE.read_frame_pose


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_read_frame_pose_matches_nearest_gps_and_heading(tmp_path: Path) -> None:
    ride_dir = tmp_path / "ride_foo"
    ride_dir.mkdir(parents=True, exist_ok=True)

    _write(
        ride_dir / "front_camera_timestamps_1.csv",
        "frame_id,timestamp\n0,1705292584.214\n5,1705292584.464\n",
    )
    _write(
        ride_dir / "gps_data_1.csv",
        "latitude,longitude,timestamp\n35.1,139.1,1705292584200\n35.2,139.2,1705292584500\n",
    )
    _write(
        ride_dir / "imu_data_1.csv",
        'compass,accelerometer,gyroscope,timestamp\n"[[1, 0, 0, 1705292584.200]]","[]","[]",1705292584200\n"[[0, 1, 0, 1705292584.500]]","[]","[]",1705292584500\n',
    )

    pose = read_frame_pose(ride_dir, frame_id=5)
    assert pose["frame_id"] == 5
    assert pose["frame_timestamp_ms"] == 1705292584464
    assert abs(pose["latitude"] - 35.2) < 1e-9
    assert abs(pose["longitude"] - 139.2) < 1e-9
    assert pose["gps_timestamp_ms"] == 1705292584500
    assert pose["heading_timestamp_ms"] == 1705292584500
    # atan2(1,0) -> 90 degrees
    assert abs(float(pose["heading_deg"]) - 90.0) < 1e-6


def test_reader_without_imu_returns_none_heading(tmp_path: Path) -> None:
    ride_dir = tmp_path / "ride_bar"
    ride_dir.mkdir(parents=True, exist_ok=True)

    _write(
        ride_dir / "front_camera_timestamps_2.csv",
        "frame_id,timestamp\n7,1705292584.700\n",
    )
    _write(
        ride_dir / "gps_data_2.csv",
        "latitude,longitude,timestamp\n35.4,139.4,1705292584700\n",
    )

    reader = RidePoseReader.from_ride_directory(ride_dir)
    pose = reader.get_frame_pose(7).to_dict()
    assert pose["heading_deg"] is None
    assert pose["heading_timestamp_ms"] is None
