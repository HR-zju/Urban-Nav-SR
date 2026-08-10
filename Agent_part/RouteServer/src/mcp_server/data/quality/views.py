"""Read-only trainability views projected from immutable source and filter ledgers.

The filter ledger is deliberately a sidecar. This module writes new JSONL
views containing original Reasoning V4 source rows, so V4.6 readers and legacy
language-goal manifests remain compatible. Filter state is kept in a separate
association index rather than injected into the canonical source schema.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_server.contracts.navigation_trainability import NavigationTrainabilityFilterV1
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic


NAVIGATION_TRAINABILITY_VIEW_SCHEMA_VERSION_V1 = "urbannav_navigation_trainability_views_v1"


@dataclass(frozen=True)
class NavigationTrainabilityViewConfigV1:
    """Configure deterministic projection of new filter views.

    Args:
        write_eligibility_views: Also write Stage I, Stage II, and VQA source
            views. They are not compiler inputs until a later explicit step.
    """

    write_eligibility_views: bool = True


def write_navigation_trainability_views_v1(
    *,
    source_jsonl: str | Path,
    filter_jsonl: str | Path,
    output_root: str | Path,
    report_json: str | Path | None = None,
    config: NavigationTrainabilityViewConfigV1 | None = None,
) -> dict[str, Any]:
    """Project source rows into immutable status and eligibility views.

    Args:
        source_jsonl: Canonical Reasoning V4 source JSONL.
        filter_jsonl: Append-only typed trainability ledger.
        output_root: New directory receiving only derived JSONL views.
        report_json: Optional report destination; defaults under ``output_root``.
        config: Optional eligibility-view policy.

    Returns:
        Counts, distributions, source/filter digests, and derived paths.

    Raises:
        ValueError: If a ledger contains duplicate parents or conflicting source
            revisions.
    """

    cfg = config or NavigationTrainabilityViewConfigV1()
    sources = _load_sources_v1(source_jsonl)
    filters = _load_latest_filters_v1(filter_jsonl)
    filter_parent_ids = {parent_id for parent_id, _fingerprint in filters}
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_rows: dict[str, list[dict[str, Any]]] = {
        "accepted": [],
        "quarantine": [],
        "rejected": [],
        "pending_retry": [],
    }
    eligibility_rows: dict[str, list[dict[str, Any]]] = {
        "stage_i_eligible": [],
        "stage_ii_eligible": [],
        "vqa_eligible": [],
    }
    associations: list[dict[str, Any]] = []
    stale_filter_rows = 0
    direction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    stop_counts: Counter[str] = Counter()
    motion_qc_counts: Counter[str] = Counter()
    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for parent_id, source in sources.items():
        result = filters.get((parent_id, source.source_fingerprint))
        status = "pending_retry"
        reason = "missing_filter_result"
        if result is not None:
            status = result.status
            reason = result.reason_codes[0] if result.reason_codes else status
            motion_qc_counts.update(result.motion_summary.qc_flags)
        elif parent_id in filter_parent_ids:
            stale_filter_rows += 1
            reason = "filter_source_fingerprint_mismatch"
        raw = source.model_dump(mode="json")
        status_rows[status].append(raw)
        dataset_counts[status][source.observation.dataset_source] += 1
        direction = _direction_label_v1(source)
        direction_counts[status][direction] += 1
        stop_counts[status] += int(direction == "stop")
        if result is not None:
            if result.eligibility.stage_i_decision:
                eligibility_rows["stage_i_eligible"].append(raw)
            if result.eligibility.stage_ii_cfm:
                eligibility_rows["stage_ii_eligible"].append(raw)
            if result.eligibility.vqa:
                eligibility_rows["vqa_eligible"].append(raw)
        associations.append(
            {
                "parent_sample_id": parent_id,
                "source_fingerprint": source.source_fingerprint,
                "filter_status": status,
                "filter_reason": reason,
                "filter_decision_id": result.decision_id if result is not None else None,
            }
        )
    output_paths: dict[str, str] = {}
    for status, rows in status_rows.items():
        path = output / f"{status}.jsonl"
        write_jsonl_atomic(path, rows)
        output_paths[status] = str(path)
    if cfg.write_eligibility_views:
        for name, rows in eligibility_rows.items():
            path = output / f"{name}.jsonl"
            write_jsonl_atomic(path, rows)
            output_paths[name] = str(path)
    association_path = output / "source_filter_association.jsonl"
    write_jsonl_atomic(association_path, associations)
    output_paths["source_filter_association"] = str(association_path)
    report = {
        "schema_version": NAVIGATION_TRAINABILITY_VIEW_SCHEMA_VERSION_V1,
        "source_jsonl": str(Path(source_jsonl).expanduser().resolve()),
        "filter_jsonl": str(Path(filter_jsonl).expanduser().resolve()),
        "source_jsonl_sha256": file_sha256(source_jsonl),
        "filter_jsonl_sha256": file_sha256(filter_jsonl),
        "source_count": len(sources),
        "filter_latest_count": len(filters),
        "filter_orphan_count": len(filter_parent_ids - set(sources)),
        "stale_filter_rows": stale_filter_rows,
        "status_counts": {name: len(rows) for name, rows in status_rows.items()},
        "eligibility_counts": {name: len(rows) for name, rows in eligibility_rows.items()},
        "dataset_counts_by_status": {name: dict(counts) for name, counts in dataset_counts.items()},
        "direction_counts_by_status": {name: dict(counts) for name, counts in direction_counts.items()},
        "stop_counts_by_status": dict(stop_counts),
        "motion_qc_flag_counts": dict(motion_qc_counts),
        "view_paths": output_paths,
        # 原始 source / 旧 language-goal manifest 是只读输入，永远不原地修改。
        "source_manifest_unchanged": True,
        "legacy_language_goal_manifest_unchanged": True,
    }
    report_path = Path(report_json).expanduser().resolve() if report_json else output / "navigation_trainability_views_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report | {"report_json": str(report_path)}


def _load_sources_v1(path: str | Path) -> dict[str, ReasoningSourceV4]:
    """Load exactly one source revision per stable physical parent."""

    rows: dict[str, ReasoningSourceV4] = {}
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        parent_id = source.observation.source_group_id
        if parent_id in rows:
            raise ValueError(f"duplicate_source_parent_sample_id:{parent_id}")
        rows[parent_id] = source
    return rows


def _load_latest_filters_v1(path: str | Path) -> dict[tuple[str, str], NavigationTrainabilityFilterV1]:
    """Load the final append-only row for each exact source revision.

    Source contracts can be repaired while preserving the parent id. Therefore
    source fingerprint is a required cache key rather than a conflict.
    """

    rows: dict[tuple[str, str], NavigationTrainabilityFilterV1] = {}
    for payload in iter_jsonl(path):
        result = NavigationTrainabilityFilterV1.model_validate(payload)
        rows[(result.parent_sample_id, result.source_fingerprint)] = result
    return rows


def _direction_label_v1(source: ReasoningSourceV4) -> str:
    """Return an audit-only direction label without using it as filter input."""

    supervision = source.direction_supervision
    if supervision is None:
        return "unknown"
    payload = supervision.model_dump(mode="json")
    for key in ("label_name", "label", "direction_label", "choice", "class_name"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


__all__ = [
    "NAVIGATION_TRAINABILITY_VIEW_SCHEMA_VERSION_V1",
    "NavigationTrainabilityViewConfigV1",
    "write_navigation_trainability_views_v1",
]
