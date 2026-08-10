"""Read-only aggregate receipts for navigation-quality annotation runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.data.io.jsonl import iter_jsonl


NAVIGATION_QUALITY_RUN_REPORT_SCHEMA_VERSION_V1 = "urbannav_navigation_quality_run_report_v1"


def summarize_navigation_quality_run_v1(
    *,
    run_root: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    """Summarize completed quality sidecars without changing their ledgers.

    A dataset directory participates only when it has a source ledger.  The
    legacy smoke layout uses ``source/smoke.jsonl`` while a full run records a
    read-only ``source/source.jsonl`` symlink to its canonical input.
    Missing optional outputs are reported as zero/empty counts so a partially
    completed smoke can be audited without manufacturing an acceptance claim.

    Args:
        run_root: Root containing one dataset subdirectory per smoke run.
        output_json: Aggregate report destination.

    Returns:
        Dataset-level and total parser, suitability, motion, and direction
        distributions.

    Raises:
        FileNotFoundError: If ``run_root`` does not exist.
        ValueError: If it contains no dataset smoke directory.
    """

    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"navigation_quality_run_root_not_found:{root}")
    datasets: dict[str, dict[str, Any]] = {}
    totals = _empty_counts_v1()
    for dataset_root in sorted(path for path in root.iterdir() if path.is_dir()):
        if _source_manifest_path_v1(dataset_root) is None:
            continue
        summary = _summarize_dataset_v1(dataset_root)
        datasets[dataset_root.name] = summary
        _merge_counts_v1(totals, summary)
    if not datasets:
        raise ValueError(f"navigation_quality_run_has_no_dataset_sources:{root}")
    report = {
        "schema_version": NAVIGATION_QUALITY_RUN_REPORT_SCHEMA_VERSION_V1,
        "run_root": str(root),
        "dataset_count": len(datasets),
        "datasets": _serialize_counts_v1(datasets),
        "totals": _serialize_counts_v1(totals),
        "source_manifests_unchanged": True,
        "legacy_language_goal_manifests_unchanged": True,
    }
    destination = Path(output_json).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _summarize_dataset_v1(dataset_root: Path) -> dict[str, Any]:
    """Read one dataset source directory into additive count structures."""

    geometry_decisions = list(_iter_optional_jsonl_v1(dataset_root / "geometry" / "decisions.jsonl"))
    geometry_rows = list(_iter_optional_jsonl_v1(dataset_root / "geometry" / "geometry.jsonl"))
    filter_rows = list(_iter_optional_jsonl_v1(dataset_root / "suitability" / "filter.jsonl"))
    view_report = _load_optional_json_v1(dataset_root / "views" / "report.json")
    source_selection = _load_optional_json_v1(dataset_root / "source" / "source_selection.json")
    if not source_selection:
        source_selection = _load_optional_json_v1(dataset_root / "source" / "smoke_selection.json")
    geometry_errors: Counter[str] = Counter()
    for decision in geometry_decisions:
        for attempt in decision.get("attempts", []):
            if isinstance(attempt, dict):
                geometry_errors.update(
                    _compact_geometry_error_code_v1(str(error))
                    for error in attempt.get("errors", [])
                    if str(error)
                )
    direction_by_status: dict[str, Counter[str]] = {}
    for status, values in (view_report.get("direction_counts_by_status") or {}).items():
        if isinstance(values, dict):
            direction_by_status[str(status)] = Counter(
                {str(label): int(count) for label, count in values.items()}
            )
    return {
        "source_selection": {
            "selected_count": int(source_selection.get("selected_count", 0)),
            "selected_primary_counts": dict(source_selection.get("selected_primary_counts") or {}),
            "shortfalls": dict(source_selection.get("shortfalls") or {}),
        },
        "geometry": {
            "decision_count": len(geometry_decisions),
            "annotation_count": len(geometry_rows),
            "decision_status_counts": Counter(
                str(row.get("status", "unknown")) for row in geometry_decisions
            ),
            "error_code_counts": geometry_errors,
            "region_role_counts": Counter(
                str(region.get("role", "unknown"))
                for row in geometry_rows
                for region in row.get("regions", [])
                if isinstance(region, dict)
            ),
            "surface_type_counts": Counter(
                str(region.get("surface_type", "unknown"))
                for row in geometry_rows
                for region in row.get("regions", [])
                if isinstance(region, dict)
            ),
        },
        "suitability": {
            "filter_count": len(filter_rows),
            "status_counts": Counter(str(row.get("status", "unknown")) for row in filter_rows),
            "reason_code_counts": Counter(
                str(reason)
                for row in filter_rows
                for reason in row.get("reason_codes", [])
            ),
            "scene_class_counts": Counter(
                str((row.get("scene_assessment") or {}).get("scene_class", "unknown"))
                for row in filter_rows
            ),
            "motion_source_kind_counts": Counter(
                str((row.get("motion_summary") or {}).get("source_kind", "unknown"))
                for row in filter_rows
            ),
            "motion_qc_flag_counts": Counter(
                str(flag)
                for row in filter_rows
                for flag in (row.get("motion_summary") or {}).get("qc_flags", [])
            ),
            "eligibility_counts": Counter(
                eligibility
                for row in filter_rows
                for eligibility, enabled in (row.get("eligibility") or {}).items()
                if enabled
            ),
        },
        "views": {
            "status_counts": Counter(view_report.get("status_counts") or {}),
            "eligibility_counts": Counter(view_report.get("eligibility_counts") or {}),
            "direction_counts_by_status": direction_by_status,
            "stop_counts_by_status": Counter(view_report.get("stop_counts_by_status") or {}),
        },
    }


def _iter_optional_jsonl_v1(path: Path):
    """Yield no rows when an optional run artifact has not been produced yet."""

    return iter_jsonl(path) if path.is_file() else iter(())


def _source_manifest_path_v1(dataset_root: Path) -> Path | None:
    """Return the legacy smoke or full-run source ledger when present."""

    for name in ("source.jsonl", "smoke.jsonl"):
        path = dataset_root / "source" / name
        if path.is_file():
            return path
    return None


def _load_optional_json_v1(path: Path) -> dict[str, Any]:
    """Load an optional object report without treating a partial run as failure."""

    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _compact_geometry_error_code_v1(error: str) -> str:
    """Collapse Pydantic value payloads into stable audit reason codes."""

    for pattern in (
        r"Value error, ([a-z][a-z0-9_]+)",
        r"^(nonaccepted_geometry_must_not_emit_regions_or_selected_corridor)$",
        r"^(accepted_geometry_requires_high_or_medium_confidence)$",
    ):
        match = re.search(pattern, error)
        if match:
            return match.group(1)
    if "literal_error" in error:
        return "ontology_literal_error"
    if error.startswith("geometry_schema:"):
        return "geometry_schema_other"
    return error.split(":", maxsplit=1)[0][:128] or "unknown_geometry_error"


def _empty_counts_v1() -> dict[str, Any]:
    """Return recursively additive counter containers for run-level totals."""

    return {
        "source_selected_count": 0,
        "geometry_decision_count": 0,
        "geometry_annotation_count": 0,
        "geometry_decision_status_counts": Counter(),
        "geometry_error_code_counts": Counter(),
        "geometry_region_role_counts": Counter(),
        "geometry_surface_type_counts": Counter(),
        "suitability_filter_count": 0,
        "suitability_status_counts": Counter(),
        "suitability_reason_code_counts": Counter(),
        "scene_class_counts": Counter(),
        "motion_source_kind_counts": Counter(),
        "motion_qc_flag_counts": Counter(),
        "eligibility_counts": Counter(),
        "direction_counts_by_status": {},
        "stop_counts_by_status": Counter(),
    }


def _merge_counts_v1(total: dict[str, Any], dataset: dict[str, Any]) -> None:
    """Merge one dataset summary into the additive run-level receipt."""

    total["source_selected_count"] += dataset["source_selection"]["selected_count"]
    total["geometry_decision_count"] += dataset["geometry"]["decision_count"]
    total["geometry_annotation_count"] += dataset["geometry"]["annotation_count"]
    total["suitability_filter_count"] += dataset["suitability"]["filter_count"]
    mappings = (
        ("geometry_decision_status_counts", dataset["geometry"]["decision_status_counts"]),
        ("geometry_error_code_counts", dataset["geometry"]["error_code_counts"]),
        ("geometry_region_role_counts", dataset["geometry"]["region_role_counts"]),
        ("geometry_surface_type_counts", dataset["geometry"]["surface_type_counts"]),
        ("suitability_status_counts", dataset["suitability"]["status_counts"]),
        ("suitability_reason_code_counts", dataset["suitability"]["reason_code_counts"]),
        ("scene_class_counts", dataset["suitability"]["scene_class_counts"]),
        ("motion_source_kind_counts", dataset["suitability"]["motion_source_kind_counts"]),
        ("motion_qc_flag_counts", dataset["suitability"]["motion_qc_flag_counts"]),
        ("eligibility_counts", dataset["suitability"]["eligibility_counts"]),
        ("stop_counts_by_status", dataset["views"]["stop_counts_by_status"]),
    )
    for name, counts in mappings:
        total[name].update(counts)
    for status, counts in dataset["views"]["direction_counts_by_status"].items():
        total["direction_counts_by_status"].setdefault(status, Counter()).update(counts)


def _serialize_counts_v1(value: Any) -> Any:
    """Recursively convert internal counters into stable JSON objects."""

    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): _serialize_counts_v1(item) for key, item in value.items()}
    return value


__all__ = [
    "NAVIGATION_QUALITY_RUN_REPORT_SCHEMA_VERSION_V1",
    "summarize_navigation_quality_run_v1",
]
