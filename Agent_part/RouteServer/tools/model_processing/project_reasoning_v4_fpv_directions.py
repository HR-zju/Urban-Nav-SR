#!/usr/bin/env python3
"""Project legacy FPV direction rows onto the frozen V4.6 20-way profile.

The projection consumes the already recorded continuous
``direction_raw_angle_deg`` produced by the canonical terminal-centroid rule.
It does not infer direction from images, GPS, route state, or language.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    canonical_direction_derivation_version,
    resolve_stage_a_direction_profile,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic


_PROJECTION_SCHEMA_VERSION = "reasoning_v4_fpv_direction_projection_v1"
_SOURCE_BIN_MODE = "degree10_12way_with_stop"
_SOURCE_LABELS = (
    "stop",
    "straight",
    "left_10",
    "right_10",
    "left_20",
    "right_20",
    "left_30",
    "right_30",
    "left_40",
    "right_40",
    "left_50plus",
    "right_50plus",
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse manifest projection arguments.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--dataset-source", default="sidewalk")
    parser.add_argument(
        "--target-profile-id",
        default=FPV_DEGREE10_TERMINAL_CENTROID_V2,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def project_reasoning_v4_fpv_direction_row(
    row: Mapping[str, Any],
    *,
    dataset_source: str = "sidewalk",
    target_profile_id: str = FPV_DEGREE10_TERMINAL_CENTROID_V2,
) -> dict[str, Any]:
    """Project one validated legacy FPV direction row.

    Args:
        row: Legacy row containing a consistent 12-way direction label and
            canonical continuous raw angle.
        dataset_source: Dataset identity written to the projected row.
        target_profile_id: Exact frozen target profile.

    Returns:
        Copied row with a complete target-profile flat direction contract.

    Raises:
        ValueError: If the source contract is inconsistent, a moving row lacks
            a finite raw angle, or the target is not the frozen FPV profile.
    """

    profile = resolve_stage_a_direction_profile(target_profile_id)
    if target_profile_id != FPV_DEGREE10_TERMINAL_CENTROID_V2:
        raise ValueError(
            "only the frozen FPV degree10 profile is supported"
        )
    source_bin_mode = str(row.get("direction_bin_mode") or "").strip()
    if source_bin_mode != _SOURCE_BIN_MODE:
        raise ValueError(
            "source direction_bin_mode must be "
            f"{_SOURCE_BIN_MODE!r}: {source_bin_mode!r}"
        )
    source_index = _required_int(row, "direction_choice_label")
    if source_index < 0 or source_index >= len(_SOURCE_LABELS):
        raise ValueError(
            f"source direction index is outside 12-way profile: {source_index}"
        )
    source_name = str(row.get("direction_name") or "").strip()
    expected_source_name = _SOURCE_LABELS[source_index]
    if source_name != expected_source_name:
        raise ValueError(
            "source direction name/index mismatch: "
            f"expected={expected_source_name!r}, actual={source_name!r}"
        )
    source_choice = str(row.get("direction_choice") or "").strip().upper()
    expected_source_choice = chr(ord("A") + source_index)
    if source_choice != expected_source_choice:
        raise ValueError(
            "source direction choice/index mismatch: "
            f"expected={expected_source_choice!r}, actual={source_choice!r}"
        )

    labels = tuple(str(value) for value in profile["labels"])
    derivation_version = canonical_direction_derivation_version(
        target_profile_id
    )
    expected_source = (
        "finish_stop_rule"
        if source_name == "stop"
        else derivation_version
    )
    actual_source = str(row.get("direction_source") or "").strip()
    if actual_source != expected_source:
        raise ValueError(
            "source direction derivation is not canonical: "
            f"expected={expected_source!r}, actual={actual_source!r}"
        )
    if source_name == "stop":
        target_name = "stop"
        target_angle = None
        target_source = "finish_stop_rule"
        raw_angle = None
    else:
        raw_angle = _required_finite_float(
            row,
            "direction_raw_angle_deg",
        )
        candidates = [
            (index, float((index - 9) * 10))
            for index, name in enumerate(labels)
            if name != "stop"
        ]
        target_index, target_angle = min(
            candidates,
            key=lambda item: abs(item[1] - raw_angle),
        )
        target_name = labels[target_index]
        target_source = derivation_version
    target_index = labels.index(target_name)
    target_choice = chr(ord("A") + target_index)

    projected = dict(row)
    projected.pop("direction_supervision", None)
    projected["dataset_source"] = str(dataset_source).strip().lower()
    projected["direction_choice_label"] = target_index
    projected["direction_name"] = target_name
    projected["direction_choice"] = target_choice
    projected["direction_angle_deg"] = target_angle
    projected["direction_raw_angle_deg"] = raw_angle
    projected["direction_source"] = target_source
    projected["direction_bin_mode"] = str(profile["bin_mode"])
    projected["direction_profile"] = target_profile_id
    projected["direction_profile_id"] = target_profile_id
    projected["direction_profile_version"] = str(
        profile["profile_version"]
    )
    # V4 source normalization switches to its canonical flat-field mapping
    # as soon as ``direction_profile_id`` is present.  Keep the legacy aliases
    # above for existing data tools, but always stamp the canonical V4 names
    # as well so the projected row is directly consumable by the source
    # builder instead of becoming a half-V4 record.
    projected["direction_label_index"] = target_index
    projected["direction_label_name"] = target_name
    projected["direction_derivation_version"] = derivation_version
    projected["direction_candidates"] = [
        {
            "index": index,
            "choice": chr(ord("A") + index),
            "name": name,
            "angle_deg": (
                None if name == "stop" else float((index - 9) * 10)
            ),
        }
        for index, name in enumerate(labels)
    ]
    projected["direction_projection"] = {
        "schema_version": _PROJECTION_SCHEMA_VERSION,
        "source_bin_mode": source_bin_mode,
        "source_label_index": source_index,
        "source_label_name": source_name,
        "source_choice": source_choice,
        "source_raw_angle_deg": raw_angle,
        "target_profile_id": target_profile_id,
        "target_label_index": target_index,
        "target_label_name": target_name,
        "target_choice": target_choice,
    }
    return projected


def project_reasoning_v4_fpv_direction_manifest(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path,
    dataset_source: str = "sidewalk",
    target_profile_id: str = FPV_DEGREE10_TERMINAL_CENTROID_V2,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Project a JSONL manifest and write a deterministic audit report.

    Args:
        input_jsonl: Legacy source JSONL.
        output_jsonl: Atomically replaced projected JSONL.
        report_json: Atomically replaced projection report.
        dataset_source: Dataset identity written to each row.
        target_profile_id: Exact frozen target profile.
        limit: Optional input-row limit.
        skip_errors: Continue after invalid rows when true.

    Returns:
        Projection counts, bounded errors, and target-profile identity.

    Raises:
        ValueError: If ``limit`` is invalid or a row is invalid while
            ``skip_errors`` is false.
    """

    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive")
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []

    def rows() -> Any:
        for index, row in enumerate(iter_jsonl(input_jsonl)):
            if limit is not None and index >= int(limit):
                break
            counts["input_rows"] += 1
            try:
                projected = project_reasoning_v4_fpv_direction_row(
                    row,
                    dataset_source=dataset_source,
                    target_profile_id=target_profile_id,
                )
            except (TypeError, ValueError) as exc:
                counts["invalid_rows"] += 1
                if len(errors) < 100:
                    errors.append(
                        {"index": index, "error": str(exc)}
                    )
                if not skip_errors:
                    raise
                continue
            counts["output_rows"] += 1
            counts[
                f"direction:{projected['direction_name']}"
            ] += 1
            yield projected

    write_jsonl_atomic(output_jsonl, rows())
    report = {
        "schema_version": _PROJECTION_SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "dataset_source": str(dataset_source).strip().lower(),
        "target_profile_id": target_profile_id,
        "counts": dict(counts),
        "errors": errors,
        "passed": counts["invalid_rows"] == 0,
    }
    report_path = Path(report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _required_int(row: Mapping[str, Any], field: str) -> int:
    """Return one required integer field."""

    value = row.get(field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not an integer: {value!r}") from exc


def _required_finite_float(
    row: Mapping[str, Any],
    field: str,
) -> float:
    """Return one required finite floating-point field."""

    value = row.get(field)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the projection CLI.

    Args:
        argv: Optional explicit CLI arguments.
    """

    args = parse_args(argv)
    report = project_reasoning_v4_fpv_direction_manifest(
        args.input_jsonl,
        args.output_jsonl,
        report_json=args.report_json,
        dataset_source=args.dataset_source,
        target_profile_id=args.target_profile_id,
        limit=args.limit,
        skip_errors=args.skip_errors,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
