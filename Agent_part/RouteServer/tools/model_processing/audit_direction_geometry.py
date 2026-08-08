#!/usr/bin/env python3
"""Build and audit Direction Geometry V1 sidecars for bounded JSONL smoke runs.

The tool is intentionally data-only.  It never decodes media, derives a route
from GPS/BEV, or turns a discrete candidate center back into a fake continuous
angle.  Its output can be supplied to the normal Reasoning V4 source builder.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionGeometrySupervisionV1,
    DirectionSupervisionV4,
)
from mcp_server.data.canonical.direction_geometry import (
    build_direction_geometry_supervision_v1,
    direction_geometry_distribution_report,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic


_V4_FIELDS = {
    "profile_id": "direction_profile_id",
    "profile_version": "direction_profile_version",
    "label_index": "direction_label_index",
    "label_name": "direction_label_name",
    "choice": "direction_choice",
    "angle_deg": "direction_angle_deg",
    "source": "direction_source",
    "bin_mode": "direction_bin_mode",
    "derivation_version": "direction_derivation_version",
}
_LEGACY_FIELDS = {
    **_V4_FIELDS,
    "profile_id": "direction_profile",
    "label_index": "direction_choice_label",
    "label_name": "direction_name",
}


def enrich_direction_geometry_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Copy one row and attach a V1 sidecar when a strong direction exists.

    Args:
        row: Projected raw row or a V4 source-like row.

    Returns:
        Copied row plus one stable status code.

    Raises:
        ValueError: If a present strong direction or geometry sidecar violates
            the canonical contract.
    """

    output = dict(row)
    existing = row.get("direction_geometry")
    direction = _direction_from_row(row)
    if direction is None:
        return output, "missing_direction_supervision"
    if existing is not None:
        if not isinstance(existing, Mapping):
            raise ValueError("direction_geometry must be an object")
        # 已有 V4 source 不应从离散标签中心反推 raw angle；只验证其 identity。
        geometry = DirectionGeometrySupervisionV1.model_validate(existing)
        _validate_geometry_direction_identity(geometry, direction)
        output["direction_geometry"] = geometry.model_dump(mode="json")
        return output, str(geometry.confidence)
    geometry = build_direction_geometry_supervision_v1(
        row,
        direction=direction,
    )
    output["direction_geometry"] = geometry.model_dump(mode="json")
    return output, str(geometry.confidence)


def _validate_geometry_direction_identity(
    geometry: DirectionGeometrySupervisionV1,
    direction: DirectionSupervisionV4,
) -> None:
    """Require an existing sidecar to remain tied to its strong direction."""

    if (
        geometry.profile_id != direction.profile_id
        or geometry.profile_version != direction.profile_version
        or geometry.label_index != direction.label_index
        or geometry.label_name != direction.label_name
        or geometry.source != direction.source
    ):
        raise ValueError("direction_geometry identity does not match direction_supervision")


def audit_direction_geometry_manifest(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    *,
    dataset_source: str,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Write a bounded geometry-enriched manifest and deterministic report.

    Args:
        input_jsonl: Projected input manifest.
        output_jsonl: Sidecar-enriched output manifest.
        report_json: Machine-readable smoke/audit report destination.
        dataset_source: Dataset name recorded in the report only.
        limit: Optional maximum rows; ``100`` is the recommended smoke size.
        skip_errors: Continue after bad input rows while recording them.

    Returns:
        Counts, bounded errors, and a geometry distribution report.
    """

    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive")
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        if limit is not None and index >= int(limit):
            break
        counts["input_rows"] += 1
        try:
            enriched, status = enrich_direction_geometry_row(row)
        except (TypeError, ValueError) as exc:
            counts["invalid_rows"] += 1
            if len(errors) < 100:
                errors.append({"index": index, "error": str(exc)})
            if not skip_errors:
                raise
            continue
        output_rows.append(enriched)
        counts["output_rows"] += 1
        counts[f"status:{status}"] += 1
    write_jsonl_atomic(output_jsonl, output_rows)
    report = {
        "schema_version": "urbannav_direction_geometry_smoke_audit_v1",
        "validation_scope": "direction_geometry_contract_only",
        "dataset_source": str(dataset_source),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "counts": dict(sorted(counts.items())),
        "geometry_distribution": direction_geometry_distribution_report(output_rows),
        "errors": errors,
        "passed": counts["invalid_rows"] == 0,
    }
    report_path = Path(report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _direction_from_row(row: Mapping[str, Any]) -> DirectionSupervisionV4 | None:
    """Read a complete nested or flat strong direction without re-deriving it."""

    nested = row.get("direction_supervision")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("direction_supervision must be an object")
        return DirectionSupervisionV4.model_validate(nested)
    fields = _V4_FIELDS if any(key in row for key in _V4_FIELDS.values()) else _LEGACY_FIELDS
    if not any(key in row for key in fields.values()):
        return None
    return DirectionSupervisionV4.model_validate(
        {target: row.get(source) for target, source in fields.items()}
    )


def _parse_dataset_input(value: str) -> tuple[str, Path]:
    """Parse one ``dataset=path`` CLI value."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset input must use dataset=path")
    dataset, raw_path = value.split("=", 1)
    dataset = dataset.strip().lower()
    path = Path(raw_path).expanduser()
    if not dataset or not str(path):
        raise argparse.ArgumentTypeError("dataset input must use non-empty dataset and path")
    return dataset, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse multi-dataset smoke arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-input",
        action="append",
        type=_parse_dataset_input,
        required=True,
        help="repeat as dataset=/absolute/path/to/input.jsonl",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit-per-dataset", type=int, default=100)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one deterministic smoke/audit per requested dataset."""

    args = parse_args(argv)
    root = args.output_root.expanduser()
    reports: dict[str, Any] = {}
    for dataset, input_path in args.dataset_input:
        destination = root / dataset
        reports[dataset] = audit_direction_geometry_manifest(
            input_path,
            destination / "direction_geometry_smoke.jsonl",
            destination / "direction_geometry_report.json",
            dataset_source=dataset,
            limit=args.limit_per_dataset,
            skip_errors=args.skip_errors,
        )
    summary_path = root / "direction_geometry_smoke_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "urbannav_direction_geometry_multi_smoke_v1",
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
