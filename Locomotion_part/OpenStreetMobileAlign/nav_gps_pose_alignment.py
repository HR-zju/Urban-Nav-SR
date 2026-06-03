#!/usr/bin/env python3
"""Align NAV REC local pose to GPS-derived local ENU positions.

OpenStreetMobile records ARKit camera poses in ``pose0.csv``. With
``ARWorldTrackingConfiguration.worldAlignment = .gravity``, ARKit uses ``y`` as
the gravity-aligned vertical axis, so this tool defaults to the horizontal
``x,-z`` pose plane when fitting the GPS ENU trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import math
import sys
from typing import Iterable


EARTH_RADIUS_M = 6378137.0
DEFAULT_POSE_AXES = "x,-z"


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass(frozen=True)
class GpsSample:
    timestamp_ns: int
    latitude: float
    longitude: float
    altitude: float
    h_accuracy: float
    v_accuracy: float


@dataclass(frozen=True)
class EnuSample:
    timestamp_ns: int
    east: float
    north: float
    up: float


@dataclass(frozen=True)
class PoseAxes:
    horizontal_a: str
    horizontal_b: str

    def project(self, coordinates: dict[str, float]) -> tuple[float, float]:
        return (
            _project_axis(self.horizontal_a, coordinates),
            _project_axis(self.horizontal_b, coordinates),
        )


@dataclass(frozen=True)
class Alignment2D:
    yaw_rad: float
    translation_east: float
    translation_north: float
    rmse_m: float
    sample_count: int


def parse_pose_axes(spec: str) -> PoseAxes:
    """Parse a two-axis pose projection, for example ``x,-z``."""
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 2:
        raise ValueError("pose axes must contain exactly two comma-separated axes")
    normalized = []
    unsigned = []
    for part in parts:
        if not part:
            raise ValueError("pose axis cannot be empty")
        signless = part[1:] if part.startswith("-") else part
        if signless not in {"x", "y", "z"} or part.count("-") > 1:
            raise ValueError(f"invalid pose axis {part!r}; use x, y, z, or a negated axis")
        normalized.append(part)
        unsigned.append(signless)
    if unsigned[0] == unsigned[1]:
        raise ValueError("pose axes must refer to two different source axes")
    return PoseAxes(normalized[0], normalized[1])


def geodetic_to_enu(
    lat: float,
    lon: float,
    alt: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
) -> tuple[float, float, float]:
    """Convert nearby WGS84 coordinates to local tangent ENU meters."""
    lat_rad = math.radians(lat)
    origin_lat_rad = math.radians(origin_lat)
    d_lat = lat_rad - origin_lat_rad
    d_lon = math.radians(lon - origin_lon)
    east = EARTH_RADIUS_M * math.cos(origin_lat_rad) * d_lon
    north = EARTH_RADIUS_M * d_lat
    up = alt - origin_alt
    return east, north, up


def estimate_yaw_translation_2d(
    pose_xy: list[tuple[float, float]],
    enu_xy: list[tuple[float, float]],
) -> Alignment2D:
    if len(pose_xy) != len(enu_xy):
        raise ValueError("pose_xy and enu_xy must have the same length")
    if len(pose_xy) < 2:
        raise ValueError("at least two paired samples are required")

    pose_cx = sum(x for x, _ in pose_xy) / len(pose_xy)
    pose_cy = sum(y for _, y in pose_xy) / len(pose_xy)
    enu_cx = sum(x for x, _ in enu_xy) / len(enu_xy)
    enu_cy = sum(y for _, y in enu_xy) / len(enu_xy)

    cross = 0.0
    dot = 0.0
    for (px, py), (ex, ey) in zip(pose_xy, enu_xy):
        px -= pose_cx
        py -= pose_cy
        ex -= enu_cx
        ey -= enu_cy
        dot += px * ex + py * ey
        cross += px * ey - py * ex

    yaw = math.atan2(cross, dot)
    c = math.cos(yaw)
    s = math.sin(yaw)
    tx = enu_cx - (c * pose_cx - s * pose_cy)
    ty = enu_cy - (s * pose_cx + c * pose_cy)

    squared_error = 0.0
    for (px, py), (ex, ey) in zip(pose_xy, enu_xy):
        predicted_east = c * px - s * py + tx
        predicted_north = s * px + c * py + ty
        squared_error += (predicted_east - ex) ** 2 + (predicted_north - ey) ** 2
    rmse = math.sqrt(squared_error / len(pose_xy))

    return Alignment2D(
        yaw_rad=yaw,
        translation_east=tx,
        translation_north=ty,
        rmse_m=rmse,
        sample_count=len(pose_xy),
    )


def run_alignment(
    dataset_path: str | Path,
    max_h_accuracy: float = 20.0,
    pose_axes: PoseAxes = parse_pose_axes(DEFAULT_POSE_AXES),
) -> tuple[Path, Path]:
    dataset = Path(dataset_path)
    poses = _read_pose_csv(dataset / "pose0.csv")
    gps = _read_gps_csv(dataset / "gps0.csv", max_h_accuracy=max_h_accuracy)
    if not poses:
        raise ValueError(f"no ARKit poses found in {dataset / 'pose0.csv'}")
    if len(gps) < 2:
        raise ValueError(f"at least two GPS samples are required in {dataset / 'gps0.csv'}")

    origin = gps[0]
    gps_enu = []
    for sample in gps:
        east, north, up = geodetic_to_enu(
            sample.latitude,
            sample.longitude,
            sample.altitude,
            origin.latitude,
            origin.longitude,
            origin.altitude,
        )
        gps_enu.append(EnuSample(sample.timestamp_ns, east, north, up))

    paired_poses: list[PoseSample] = []
    paired_pose_horizontal: list[tuple[float, float]] = []
    paired_enu: list[EnuSample] = []
    for pose in poses:
        enu = _interpolate_enu(gps_enu, pose.timestamp_ns)
        if enu is None:
            continue
        paired_poses.append(pose)
        paired_pose_horizontal.append(_project_pose(pose, pose_axes))
        paired_enu.append(enu)

    if len(paired_poses) < 2:
        raise ValueError("at least two pose samples must overlap filtered GPS timestamps")

    alignment = estimate_yaw_translation_2d(
        paired_pose_horizontal,
        [(enu.east, enu.north) for enu in paired_enu],
    )

    output_pose = dataset / "aligned_pose_enu.csv"
    output_meta = dataset / "alignment_metadata.txt"
    _write_aligned_pose_csv(output_pose, paired_poses, paired_pose_horizontal, paired_enu, alignment)
    _write_metadata(output_meta, dataset, origin, alignment, max_h_accuracy, pose_axes)
    return output_pose, output_meta


def _project_axis(axis: str, coordinates: dict[str, float]) -> float:
    if axis.startswith("-"):
        return -coordinates[axis[1:]]
    return coordinates[axis]


def _project_pose(pose: PoseSample, pose_axes: PoseAxes) -> tuple[float, float]:
    return pose_axes.project({"x": pose.x, "y": pose.y, "z": pose.z})


def _read_pose_csv(path: Path) -> list[PoseSample]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(_without_comment_prefix(handle))
        return [
            PoseSample(
                timestamp_ns=int(row["timestamp [ns]"]),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                qw=float(row["qw"]),
                qx=float(row["qx"]),
                qy=float(row["qy"]),
                qz=float(row["qz"]),
            )
            for row in reader
        ]


def _read_gps_csv(path: Path, max_h_accuracy: float) -> list[GpsSample]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(_without_comment_prefix(handle))
        samples = []
        for row in reader:
            h_accuracy = float(row["h_accuracy"])
            if h_accuracy < 0 or h_accuracy > max_h_accuracy:
                continue
            samples.append(
                GpsSample(
                    timestamp_ns=int(row["timestamp [ns]"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    altitude=float(row["altitude"]),
                    h_accuracy=h_accuracy,
                    v_accuracy=float(row["v_accuracy"]),
                )
            )
        return sorted(samples, key=lambda sample: sample.timestamp_ns)


def _without_comment_prefix(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        if line.startswith("#timestamp"):
            yield line[1:]
        elif line.startswith("#"):
            continue
        else:
            yield line


def _interpolate_enu(samples: list[EnuSample], timestamp_ns: int) -> EnuSample | None:
    if timestamp_ns < samples[0].timestamp_ns or timestamp_ns > samples[-1].timestamp_ns:
        return None
    for index, sample in enumerate(samples):
        if sample.timestamp_ns == timestamp_ns:
            return sample
        if sample.timestamp_ns > timestamp_ns and index > 0:
            prev = samples[index - 1]
            span = sample.timestamp_ns - prev.timestamp_ns
            ratio = (timestamp_ns - prev.timestamp_ns) / span
            return EnuSample(
                timestamp_ns=timestamp_ns,
                east=prev.east + ratio * (sample.east - prev.east),
                north=prev.north + ratio * (sample.north - prev.north),
                up=prev.up + ratio * (sample.up - prev.up),
            )
    return None


def _write_aligned_pose_csv(
    path: Path,
    poses: list[PoseSample],
    pose_horizontal: list[tuple[float, float]],
    gps_enu: list[EnuSample],
    alignment: Alignment2D,
) -> None:
    c = math.cos(alignment.yaw_rad)
    s = math.sin(alignment.yaw_rad)
    with path.open("w", newline="") as handle:
        fieldnames = [
            "timestamp [ns]",
            "east",
            "north",
            "up",
            "gps_east",
            "gps_north",
            "gps_up",
            "pose_horizontal_a",
            "pose_horizontal_b",
            "pose_x",
            "pose_y",
            "pose_z",
            "qw",
            "qx",
            "qy",
            "qz",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pose, (pose_a, pose_b), gps in zip(poses, pose_horizontal, gps_enu):
            east = c * pose_a - s * pose_b + alignment.translation_east
            north = s * pose_a + c * pose_b + alignment.translation_north
            writer.writerow({
                "timestamp [ns]": pose.timestamp_ns,
                "east": f"{east:.9f}",
                "north": f"{north:.9f}",
                "up": f"{gps.up:.9f}",
                "gps_east": f"{gps.east:.9f}",
                "gps_north": f"{gps.north:.9f}",
                "gps_up": f"{gps.up:.9f}",
                "pose_horizontal_a": f"{pose_a:.9f}",
                "pose_horizontal_b": f"{pose_b:.9f}",
                "pose_x": f"{pose.x:.9f}",
                "pose_y": f"{pose.y:.9f}",
                "pose_z": f"{pose.z:.9f}",
                "qw": f"{pose.qw:.9f}",
                "qx": f"{pose.qx:.9f}",
                "qy": f"{pose.qy:.9f}",
                "qz": f"{pose.qz:.9f}",
            })


def _write_metadata(
    path: Path,
    dataset: Path,
    origin: GpsSample,
    alignment: Alignment2D,
    max_h_accuracy: float,
    pose_axes: PoseAxes,
) -> None:
    path.write_text(
        "\n".join([
            "alignment_type=arkit_yaw_translation_2d",
            f"dataset_path={dataset}",
            f"aligned_samples={alignment.sample_count}",
            f"pose_horizontal_axes={pose_axes.horizontal_a},{pose_axes.horizontal_b}",
            "pose_vertical_axis=y",
            f"yaw_rad={alignment.yaw_rad:.12f}",
            f"yaw_deg={math.degrees(alignment.yaw_rad):.9f}",
            f"translation_east={alignment.translation_east:.9f}",
            f"translation_north={alignment.translation_north:.9f}",
            f"rmse_m={alignment.rmse_m:.9f}",
            f"gps_origin_timestamp_ns={origin.timestamp_ns}",
            f"gps_origin_latitude={origin.latitude:.12f}",
            f"gps_origin_longitude={origin.longitude:.12f}",
            f"gps_origin_altitude={origin.altitude:.9f}",
            f"max_h_accuracy={max_h_accuracy:.3f}",
            "",
        ])
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Align NAV REC ARKit pose0.csv to GPS ENU coordinates.",
    )
    parser.add_argument("dataset", type=Path, help="NAV REC dataset directory")
    parser.add_argument(
        "--max-h-accuracy",
        type=float,
        default=20.0,
        help="discard GPS samples with horizontal accuracy above this many meters",
    )
    parser.add_argument(
        "--pose-axes",
        default=DEFAULT_POSE_AXES,
        help="two ARKit pose axes used as the horizontal plane, default: x,-z",
    )
    args = parser.parse_args(argv)

    try:
        pose_axes = parse_pose_axes(args.pose_axes)
        output_pose, output_meta = run_alignment(
            args.dataset,
            max_h_accuracy=args.max_h_accuracy,
            pose_axes=pose_axes,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"wrote {output_pose}")
    print(f"wrote {output_meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
