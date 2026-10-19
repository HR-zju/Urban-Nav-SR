#!/usr/bin/env python3
"""Project trusted ego trajectories onto a canonical V4.6 direction profile.

This preprocessing step is intentionally separate from the direction stamper.
It reads trusted metric ``trajectory_ego_t`` geometry and records exactly how
the canonical direction was derived. It never reads images, BEV, GPS, route
state, language, or an LVLM answer.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.direction_profile import (
    EGO_DEGREE10_TERMINAL_CENTROID_V2,
    canonical_direction_derivation_version,
    resolve_stage_a_direction_profile,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from urbannav_contracts.stage_ii_cfm import StageIITrajectoryTargetV2


_PROJECTION_SCHEMA_VERSION = "reasoning_v4_ego_direction_projection_v1"
_DEFAULT_ALLOWED_TRAJECTORY_SOURCES = ("egowalk_odometry",)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse ego-direction projection arguments.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--dataset-source", default="egowalk")
    parser.add_argument(
        "--target-profile-id",
        default=EGO_DEGREE10_TERMINAL_CENTROID_V2,
    )
    parser.add_argument(
        "--allowed-trajectory-source",
        action="append",
        dest="allowed_trajectory_sources",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--history-package", action="store_true",
                        help="Project source_row in observations.jsonl; preserve observation and Stage II sidecar")
    parser.add_argument("--direction-from-sidecar", action="store_true",
                        help="Use the package's metric FLU Stage II target for direction only")
    return parser.parse_args(argv)


def project_reasoning_v4_ego_direction_row(
    row: Mapping[str, Any],
    *,
    dataset_source: str = "egowalk",
    target_profile_id: str = EGO_DEGREE10_TERMINAL_CENTROID_V2,
    allowed_trajectory_sources: Sequence[str] = (
        _DEFAULT_ALLOWED_TRAJECTORY_SOURCES
    ),
) -> dict[str, Any]:
    r"""Derive one canonical direction stamp from trusted ego geometry.

    The trajectory frame is ``x_right/y_front`` in meters. For a trajectory
    with :math:`N` points and profile lookahead ratio :math:`r`, the reference
    point is

    \[
    p_{ref}=\operatorname{mean}(p_k,\ldots,p_{N-1}),\qquad
    k=\operatorname{round}(r(N-1)).
    \]

    Displacement below the profile threshold is ``stop``. Otherwise the raw
    angle is :math:`\operatorname{atan2}(x_{right},y_{front})` and is mapped to
    the nearest registered output angle.

    Args:
        row: Source row containing trusted ``trajectory_ego_t`` geometry.
        dataset_source: Dataset identity written to the projected row.
        target_profile_id: Registered ego terminal-centroid profile.
        allowed_trajectory_sources: Exact trusted trajectory provenance codes.

    Returns:
        Copied row with canonical flat and nested direction supervision.

    Raises:
        ValueError: If provenance, trajectory geometry, or profile is invalid.
    """

    profile = resolve_stage_a_direction_profile(target_profile_id)
    if profile.get("derivation_strategy") != "ego_terminal_centroid":
        raise ValueError(
            "target profile must use the ego terminal-centroid producer"
        )
    normalized_dataset = str(dataset_source).strip().lower()
    existing_dataset = str(
        row.get("source_dataset") or row.get("dataset_source") or ""
    ).strip().lower()
    if existing_dataset and existing_dataset != normalized_dataset:
        raise ValueError(
            "source dataset does not match requested dataset: "
            f"{existing_dataset!r}!={normalized_dataset!r}"
        )

    trajectory_policy = row.get("trajectory_policy")
    if not isinstance(trajectory_policy, Mapping):
        raise ValueError("trajectory_policy must be an object")
    trajectory_source = str(trajectory_policy.get("source") or "").strip()
    allowed = {str(value).strip() for value in allowed_trajectory_sources}
    if trajectory_source not in allowed:
        raise ValueError(
            "trajectory source is not approved for ego direction projection: "
            f"{trajectory_source!r}"
        )

    points = _trajectory_xy(row.get("trajectory_ego_t"))
    lookahead_ratio = float(profile["lookahead_ratio"])
    reference_start = int(
        round(lookahead_ratio * max(len(points) - 1, 0))
    )
    reference_start = min(max(reference_start, 0), len(points) - 1)
    suffix = points[reference_start:]
    x_right = sum(point[0] for point in suffix) / len(suffix)
    y_front = sum(point[1] for point in suffix) / len(suffix)
    displacement = math.hypot(x_right, y_front)
    stop_threshold = float(profile["stop_threshold"])
    labels = tuple(str(value) for value in profile["labels"])
    derivation_version = canonical_direction_derivation_version(
        target_profile_id
    )

    if displacement < stop_threshold:
        label_name = "stop"
        raw_angle = None
    else:
        raw_angle = math.degrees(
            math.atan2(x_right, max(y_front, 1e-6))
        )
        moving = tuple(
            (index, float(angle))
            for index, angle in enumerate(profile["angles"])
            if angle is not None
        )
        label_index, _ = min(
            moving,
            key=lambda candidate: abs(candidate[1] - raw_angle),
        )
        label_name = labels[label_index]

    label_index = labels.index(label_name)
    choice = str(profile["choices"][label_index])
    angle = profile["angles"][label_index]
    supervision = {
        "profile_id": target_profile_id,
        "profile_version": str(profile["profile_version"]),
        "label_index": label_index,
        "label_name": label_name,
        "choice": choice,
        "angle_deg": angle,
        "source": derivation_version,
        "bin_mode": str(profile["bin_mode"]),
        "derivation_version": derivation_version,
    }

    projected = dict(row)
    legacy = {
        "label_index": row.get("direction_choice_label"),
        "label_name": row.get("direction_name"),
        "choice": row.get("direction_choice"),
        "source": row.get("direction_source"),
        "bin_mode": row.get("direction_bin_mode"),
        "raw_angle_deg": row.get("direction_raw_angle_deg"),
    }
    projected.update(
        {
            "dataset_source": normalized_dataset,
            "direction_supervision": supervision,
            "direction_profile": target_profile_id,
            "direction_profile_id": target_profile_id,
            "direction_profile_version": str(profile["profile_version"]),
            "direction_choice_label": label_index,
            "direction_label_index": label_index,
            "direction_name": label_name,
            "direction_label_name": label_name,
            "direction_choice": choice,
            "direction_angle_deg": angle,
            "direction_raw_angle_deg": raw_angle,
            "direction_source": derivation_version,
            "direction_bin_mode": str(profile["bin_mode"]),
            "direction_derivation_version": derivation_version,
            "direction_reference_method": "terminal_centroid",
            "direction_reference_start_index": reference_start,
            "direction_reference_end_index": len(points) - 1,
            "direction_reference_point_count": len(suffix),
            "direction_displacement": displacement,
            "direction_candidates": [
                {
                    "index": index,
                    "choice": str(profile["choices"][index]),
                    "name": name,
                    "angle_deg": profile["angles"][index],
                }
                for index, name in enumerate(labels)
            ],
            "direction_projection": {
                "schema_version": _PROJECTION_SCHEMA_VERSION,
                "trajectory_field": "trajectory_ego_t",
                "trajectory_source": trajectory_source,
                "coordinate_frame": str(profile["coordinate_frame"]),
                "coordinate_unit": str(profile["coordinate_unit"]),
                "lookahead_ratio": lookahead_ratio,
                "stop_threshold": stop_threshold,
                "reference_start_index": reference_start,
                "reference_point_count": len(suffix),
                "reference_x_right_m": x_right,
                "reference_y_front_m": y_front,
                "displacement_m": displacement,
                "target_profile_id": target_profile_id,
                "target_label_name": label_name,
                "legacy_direction": legacy,
            },
        }
    )
    return projected


def project_history_sidecar_direction(record: Mapping[str, Any], *, dataset_source: str,
                                      target_profile_id: str) -> dict[str, Any]:
    """Project a linked FLU target into direction fields without rewriting action GT.

    Args:
        record: Materialized observation record with its original trajectory sidecar.
        dataset_source: Expected dataset identity.
        target_profile_id: Explicit registered ego terminal-centroid profile.
    Returns:
        Source row carrying newly computed direction supervision and provenance.
    Raises:
        ValueError: If target identity or the metric FLU contract is inconsistent.
    """
    sidecar = record["trajectory_sidecar"]
    if sidecar is None:
        raise ValueError(f"{record['source_parent_id']}: direction requires a linked trajectory sidecar")
    if sidecar["sample_id"] != record["source_parent_id"] or sidecar["dataset_source"] != dataset_source:
        raise ValueError("Direction sidecar must match the observation parent and dataset")
    target = StageIITrajectoryTargetV2.model_validate(sidecar["trajectory_target"])
    policy = target.policy
    # 仅在方向计算临时视图中换轴：(x_right,y_front)=(-y_left,x_forward)。
    # mask 决定可用点，原 [32,3]、yaw 和 action_mask 全部保留在 sidecar 中。
    points = [[-point[1], point[0]] for point, valid in zip(target.trajectory_ego_t, target.action_mask) if valid]
    original = record["source_row"]
    temporary = {**original, "source_dataset": dataset_source, "dataset_source": dataset_source,
                 "trajectory_ego_t": points, "trajectory_policy": {"source": policy.trajectory_source}}
    projected = project_reasoning_v4_ego_direction_row(
        temporary, dataset_source=dataset_source, target_profile_id=target_profile_id,
        allowed_trajectory_sources=(policy.trajectory_source,))
    projected["direction_projection"].update({
        "trajectory_field": "trajectory_sidecar.trajectory_target.trajectory_ego_t",
        "target_digest": target.target_digest,
        "input_coordinate_frame": policy.coordinate_frame,
        "valid_target_indices": [i for i, valid in enumerate(target.action_mask) if valid],
    })
    return {**original, **{key: value for key, value in projected.items() if key.startswith("direction_")}}


def project_reasoning_v4_ego_direction_manifest(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path,
    dataset_source: str = "egowalk",
    target_profile_id: str = EGO_DEGREE10_TERMINAL_CENTROID_V2,
    allowed_trajectory_sources: Sequence[str] = (
        _DEFAULT_ALLOWED_TRAJECTORY_SOURCES
    ),
    limit: int | None = None,
    skip_errors: bool = False,
    history_package: bool = False,
    direction_from_sidecar: bool = False,
) -> dict[str, Any]:
    """Project a JSONL manifest and write a deterministic audit report.

    Args:
        input_jsonl: Legacy source JSONL.
        output_jsonl: Atomically replaced projected JSONL.
        report_json: Projection report path.
        dataset_source: Exact dataset identity.
        target_profile_id: Registered ego direction profile.
        allowed_trajectory_sources: Trusted trajectory provenance allowlist.
        limit: Optional positive input-row limit.
        skip_errors: Continue after invalid rows when true.
        history_package: Read nested source_row and preserve the rest of each package record.
        direction_from_sidecar: Derive direction from linked metric FLU targets instead of legacy geometry.

    Returns:
        Projection counts, bounded errors, and profile identity.

    Raises:
        ValueError: If arguments or a strict input row are invalid.
    """

    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive")
    if direction_from_sidecar and not history_package:
        raise ValueError("direction_from_sidecar requires history_package")
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    def rows() -> Any:
        for index, row in enumerate(iter_jsonl(input_jsonl)):
            if limit is not None and index >= int(limit):
                break
            counts["input_rows"] += 1
            if direction_from_sidecar and row["trajectory_sidecar"] is None:
                # 正常的监督缺失独立记录；坏 schema、身份错配与 IO 错误仍按原严格路径报出。
                counts["missing_linked_sidecar"] += 1
                exclusions.append({"source_parent_id": row["source_parent_id"],
                                   "reason": "missing_linked_sidecar"})
                continue
            source_row = row["source_row"] if history_package else row
            try:
                projected = (project_history_sidecar_direction(
                    row, dataset_source=dataset_source, target_profile_id=target_profile_id)
                    if direction_from_sidecar else project_reasoning_v4_ego_direction_row(
                    source_row,
                    dataset_source=dataset_source,
                    target_profile_id=target_profile_id,
                    allowed_trajectory_sources=allowed_trajectory_sources,
                ))
            except (TypeError, ValueError) as exc:
                counts["invalid_rows"] += 1
                if len(errors) < 100:
                    errors.append(
                        {
                            "index": index,
                            "id": row.get("id"),
                            "error": str(exc),
                        }
                    )
                if not skip_errors:
                    raise
                continue
            counts["output_rows"] += 1
            name = str(projected["direction_name"])
            counts[f"direction:{name}"] += 1
            if name != str(source_row.get("direction_name") or ""):
                counts["legacy_label_changed_rows"] += 1
            # 方向来自原 x_right/y_front odometry；包中的 FLU action sidecar 保持原样。
            yield {**row, "source_row": projected} if history_package else projected

    written = write_jsonl_atomic(output_jsonl, rows())
    if written != counts["output_rows"]:
        raise RuntimeError("atomic writer count does not match projection count")
    report = {
        "schema_version": _PROJECTION_SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "dataset_source": str(dataset_source).strip().lower(),
        "target_profile_id": target_profile_id,
        "allowed_trajectory_sources": list(allowed_trajectory_sources) if not direction_from_sidecar else None,
        "trajectory_input_contract": "StageIITrajectoryTargetV2:current_ego_flu" if direction_from_sidecar
                                     else "legacy_x_right_y_front",
        "trajectory_read": True,
        "direction_rederived": True,
        "history_package": history_package,
        "direction_from_sidecar": direction_from_sidecar,
        "counts": dict(sorted(counts.items())),
        "errors": errors,
        "passed": counts.get("invalid_rows", 0) == 0,
    }
    report_path = Path(report_json)
    if direction_from_sidecar:
        drop_path = report_path.with_name(report_path.stem + "-drops.jsonl")
        write_jsonl_atomic(drop_path, exclusions)
        report["exclusion_jsonl"] = str(drop_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _trajectory_xy(value: Any) -> list[tuple[float, float]]:
    """Validate and return finite ego ``x_right/y_front`` points."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("trajectory_ego_t must be a non-empty list")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(
                f"trajectory_ego_t[{index}] must contain x_right/y_front"
            )
        try:
            x_right = float(point[0])
            y_front = float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"trajectory_ego_t[{index}] is not numeric"
            ) from exc
        if not math.isfinite(x_right) or not math.isfinite(y_front):
            raise ValueError(
                f"trajectory_ego_t[{index}] contains a non-finite value"
            )
        points.append((x_right, y_front))
    return points


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line ego direction projector.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        ``None``.
    """

    args = parse_args(argv)
    report = project_reasoning_v4_ego_direction_manifest(
        args.input_jsonl,
        args.output_jsonl,
        report_json=args.report_json,
        dataset_source=args.dataset_source,
        target_profile_id=args.target_profile_id,
        allowed_trajectory_sources=(
            tuple(args.allowed_trajectory_sources)
            if args.allowed_trajectory_sources
            else _DEFAULT_ALLOWED_TRAJECTORY_SOURCES
        ),
        limit=args.limit,
        skip_errors=args.skip_errors,
        history_package=args.history_package,
        direction_from_sidecar=args.direction_from_sidecar,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
