"""Deterministically shard and merge resumable language-goal annotation ledgers.

The helpers in this module own only run-local derived files.  They never
rewrite a canonical Reasoning V4 source, a Stage-II sidecar, or an existing
append-only annotation ledger.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from mcp_server.contracts.language_waypoint_benchmark import (
    LanguageGoalAnnotationV1,
    LanguageGoalCandidateV1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.contracts.stage_ii_cfm import StageIITrajectorySidecarV2
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic


LANGUAGE_WAYPOINT_ANNOTATION_SHARD_REPORT_SCHEMA_VERSION_V1 = (
    "urbannav_language_waypoint_annotation_shard_report_v1"
)
LANGUAGE_WAYPOINT_ANNOTATION_MERGE_REPORT_SCHEMA_VERSION_V1 = (
    "urbannav_language_waypoint_annotation_merge_report_v1"
)
LANGUAGE_WAYPOINT_SIDECAR_MATERIALIZATION_REPORT_SCHEMA_VERSION_V1 = (
    "urbannav_language_waypoint_sidecar_materialization_report_v1"
)


def materialize_language_goal_sidecars_by_dataset_v1(
    *,
    trajectory_sidecar_jsonls: Sequence[str | Path],
    output_root: str | Path,
    report_json: str | Path,
) -> dict[str, Any]:
    """Deduplicate immutable V2 sidecars into dataset-local ledgers.

    Candidate building is dataset-local.  Materializing a small set of
    dataset-local sidecar ledgers avoids parsing every 4-second sidecar for
    every source manifest, while retaining the exact input digest in the
    report.

    Args:
        trajectory_sidecar_jsonls: One or more immutable accepted V2 ledgers.
        output_root: New derived-artifact directory.
        report_json: Durable sidecar materialization receipt.

    Returns:
        Input/output hashes and accepted sidecar counts by dataset.

    Raises:
        FileNotFoundError: If an input sidecar ledger does not exist.
        ValueError: If a parent has conflicting sidecar revisions.
    """

    if not trajectory_sidecar_jsonls:
        raise ValueError("at_least_one_trajectory_sidecar_jsonl_required")
    inputs = [Path(path).expanduser().resolve() for path in trajectory_sidecar_jsonls]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows_by_dataset: dict[str, dict[str, StageIITrajectorySidecarV2]] = defaultdict(dict)
    duplicate_equivalent = 0
    for path in inputs:
        for payload in iter_jsonl(path):
            sidecar = StageIITrajectorySidecarV2.model_validate(payload)
            existing = rows_by_dataset[sidecar.dataset_source].get(sidecar.sample_id)
            if existing is not None:
                if existing.sidecar_digest != sidecar.sidecar_digest:
                    raise ValueError(f"conflicting_stage_ii_sidecar:{sidecar.sample_id}")
                duplicate_equivalent += 1
                continue
            rows_by_dataset[sidecar.dataset_source][sidecar.sample_id] = sidecar
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    counts: dict[str, int] = {}
    for dataset_source in sorted(rows_by_dataset):
        path = destination / f"{dataset_source}_trajectory_sidecars_v2.jsonl"
        rows = [
            sidecar.model_dump(mode="json")
            for _sample_id, sidecar in sorted(rows_by_dataset[dataset_source].items())
        ]
        write_jsonl_atomic(path, rows)
        output_paths[dataset_source] = str(path)
        counts[dataset_source] = len(rows)
    report = {
        "schema_version": LANGUAGE_WAYPOINT_SIDECAR_MATERIALIZATION_REPORT_SCHEMA_VERSION_V1,
        "input_sidecar_jsonls": [str(path) for path in inputs],
        "input_sidecar_sha256": {str(path): file_sha256(path) for path in inputs},
        "output_paths": output_paths,
        "output_sha256": {dataset: file_sha256(path) for dataset, path in output_paths.items()},
        "counts_by_dataset": counts,
        "total_sidecar_count": sum(counts.values()),
        "duplicate_equivalent_count": duplicate_equivalent,
    }
    _write_json_v1(report_json, report)
    return report


def partition_language_goal_annotation_inputs_v1(
    *,
    source_jsonl: str | Path,
    candidates_jsonl: str | Path,
    output_root: str | Path,
    report_json: str | Path,
    lane_count: int = 2,
) -> dict[str, Any]:
    """Create stable parent-disjoint source/candidate lanes for LVLM workers.

    The lane is ``sha256(parent_sample_id) mod lane_count``.  Re-running with
    the same input and lane count therefore preserves assignments, which makes
    append-only resumption safe after a service interruption.

    Args:
        source_jsonl: Canonical Reasoning V4 source ledger.
        candidates_jsonl: Teacher-only candidate ledger for the same parents.
        output_root: New directory receiving ``lane_XX`` source/candidate files.
        report_json: Durable partition receipt.
        lane_count: Number of independent provider lanes.

    Returns:
        Input fingerprints, matched counts, and lane-level distributions.

    Raises:
        ValueError: If parents are duplicate, missing, or candidate identities
            conflict.
    """

    if lane_count < 1:
        raise ValueError("lane_count_must_be_positive")
    source_path = Path(source_jsonl).expanduser().resolve()
    candidate_path = Path(candidates_jsonl).expanduser().resolve()
    sources = _load_unique_sources_v1(source_path)
    candidates = _load_unique_candidates_v1(candidate_path)
    candidate_parents = set(candidates)
    missing_sources = sorted(candidate_parents - set(sources))
    if missing_sources:
        raise ValueError(f"candidate_source_missing:{missing_sources[0]}")

    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lane_sources: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    lane_candidates: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    lane_dataset_counts: list[Counter[str]] = [Counter() for _ in range(lane_count)]
    for parent_id, source in sources.items():
        if parent_id not in candidate_parents:
            continue
        lane = _lane_for_parent_v1(parent_id, lane_count=lane_count)
        lane_sources[lane].append(source.model_dump(mode="json"))
    for parent_id, candidate in candidates.items():
        lane = _lane_for_parent_v1(parent_id, lane_count=lane_count)
        lane_candidates[lane].append(candidate.model_dump(mode="json"))
        lane_dataset_counts[lane][candidate.dataset_source] += 1
    lanes: list[dict[str, Any]] = []
    for lane in range(lane_count):
        lane_root = destination / f"lane_{lane:02d}"
        source_output = lane_root / "sources.jsonl"
        candidate_output = lane_root / "candidates.jsonl"
        source_count = write_jsonl_atomic(source_output, lane_sources[lane])
        candidate_count = write_jsonl_atomic(candidate_output, lane_candidates[lane])
        if source_count != candidate_count:
            raise AssertionError("language_annotation_lane_source_candidate_count_mismatch")
        lanes.append(
            {
                "lane": lane,
                "source_jsonl": str(source_output),
                "candidates_jsonl": str(candidate_output),
                "source_count": source_count,
                "candidate_count": candidate_count,
                "dataset_counts": dict(sorted(lane_dataset_counts[lane].items())),
                "source_sha256": file_sha256(source_output),
                "candidates_sha256": file_sha256(candidate_output),
            }
        )
    report = {
        "schema_version": LANGUAGE_WAYPOINT_ANNOTATION_SHARD_REPORT_SCHEMA_VERSION_V1,
        "source_jsonl": str(source_path),
        "candidates_jsonl": str(candidate_path),
        "source_jsonl_sha256": file_sha256(source_path),
        "candidates_jsonl_sha256": file_sha256(candidate_path),
        "source_count": len(sources),
        "candidate_count": len(candidates),
        "unpaired_source_count": len(sources) - len(candidates),
        "lane_count": lane_count,
        "lanes": lanes,
    }
    _write_json_v1(report_json, report)
    return report


def merge_language_goal_annotation_shards_v1(
    *,
    candidates_jsonl: str | Path,
    annotation_jsonls: Sequence[str | Path],
    decisions_jsonls: Sequence[str | Path],
    output_annotations_jsonl: str | Path,
    output_decisions_jsonl: str | Path,
    report_json: str | Path,
) -> dict[str, Any]:
    """Merge dual-service append-only ledgers in canonical candidate order.

    Annotation rows exist only after a typed parser accepts them.  Decision
    rows retain parser/provider failures.  The merged outputs preserve this
    distinction and never manufacture accepted labels for missing candidates.

    Args:
        candidates_jsonl: Canonical candidate order for one dataset.
        annotation_jsonls: Lane-local append-only typed annotation ledgers.
        decisions_jsonls: Lane-local append-only provider/audit ledgers.
        output_annotations_jsonl: New merged accepted-annotation ledger.
        output_decisions_jsonl: New merged decision/audit ledger.
        report_json: Durable merge receipt.

    Returns:
        Coverage, status, missing-decision, and immutable hash statistics.

    Raises:
        ValueError: If lanes disagree on a candidate or emit conflicting
            accepted annotations for one parent.
    """

    if not annotation_jsonls or not decisions_jsonls:
        raise ValueError("annotation_and_decision_shards_required")
    candidates = _load_unique_candidates_v1(Path(candidates_jsonl).expanduser().resolve())
    annotations = _load_latest_annotations_v1(annotation_jsonls, candidates=candidates)
    decisions = _load_latest_decisions_v1(decisions_jsonls, candidates=candidates)
    ordered_annotations: list[dict[str, Any]] = []
    ordered_decisions: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    missing_decisions: list[str] = []
    for parent_id, candidate in candidates.items():
        decision = decisions.get(candidate.candidate_id)
        if decision is None:
            missing_decisions.append(candidate.candidate_id)
            status_counts["missing_decision"] += 1
        else:
            ordered_decisions.append(decision)
            status_counts[str(decision.get("status", "unknown"))] += 1
        annotation = annotations.get(parent_id)
        if annotation is not None:
            ordered_annotations.append(annotation.model_dump(mode="json"))
    annotation_count = write_jsonl_atomic(output_annotations_jsonl, ordered_annotations)
    decision_count = write_jsonl_atomic(output_decisions_jsonl, ordered_decisions)
    report = {
        "schema_version": LANGUAGE_WAYPOINT_ANNOTATION_MERGE_REPORT_SCHEMA_VERSION_V1,
        "candidates_jsonl": str(Path(candidates_jsonl).expanduser().resolve()),
        "candidate_count": len(candidates),
        "annotation_count": annotation_count,
        "decision_count": decision_count,
        "status_counts": dict(sorted(status_counts.items())),
        "missing_decision_count": len(missing_decisions),
        "missing_decision_candidate_ids": missing_decisions[:200],
        "annotation_coverage": annotation_count / len(candidates) if candidates else 0.0,
        "decision_coverage": decision_count / len(candidates) if candidates else 0.0,
        "input_annotation_jsonls": [str(Path(path).expanduser().resolve()) for path in annotation_jsonls],
        "input_decisions_jsonls": [str(Path(path).expanduser().resolve()) for path in decisions_jsonls],
        "output_annotations_jsonl": str(Path(output_annotations_jsonl).expanduser().resolve()),
        "output_decisions_jsonl": str(Path(output_decisions_jsonl).expanduser().resolve()),
        "output_annotations_sha256": file_sha256(output_annotations_jsonl),
        "output_decisions_sha256": file_sha256(output_decisions_jsonl),
    }
    _write_json_v1(report_json, report)
    return report


def _load_unique_sources_v1(path: Path) -> dict[str, ReasoningSourceV4]:
    """Load one immutable source revision per physical parent."""

    rows: dict[str, ReasoningSourceV4] = {}
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        parent_id = source.observation.source_group_id
        if parent_id in rows:
            raise ValueError(f"duplicate_source_group_id:{parent_id}")
        rows[parent_id] = source
    return rows


def _load_unique_candidates_v1(path: Path) -> dict[str, LanguageGoalCandidateV1]:
    """Load exactly one teacher candidate per physical parent."""

    rows: dict[str, LanguageGoalCandidateV1] = {}
    candidate_ids: set[str] = set()
    for payload in iter_jsonl(path):
        candidate = LanguageGoalCandidateV1.model_validate(payload)
        if candidate.parent_sample_id in rows:
            raise ValueError(f"duplicate_candidate_parent_sample_id:{candidate.parent_sample_id}")
        if candidate.candidate_id in candidate_ids:
            raise ValueError(f"duplicate_candidate_id:{candidate.candidate_id}")
        rows[candidate.parent_sample_id] = candidate
        candidate_ids.add(candidate.candidate_id)
    return rows


def _load_latest_annotations_v1(
    paths: Sequence[str | Path],
    *,
    candidates: dict[str, LanguageGoalCandidateV1],
) -> dict[str, LanguageGoalAnnotationV1]:
    """Read compatible accepted annotations and reject cross-lane conflicts."""

    result: dict[str, LanguageGoalAnnotationV1] = {}
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_file():
            continue
        for payload in iter_jsonl(path):
            annotation = LanguageGoalAnnotationV1.model_validate(payload)
            parent_id = annotation.parent_sample_id
            if parent_id not in candidates:
                raise ValueError(f"annotation_parent_not_in_candidates:{parent_id}")
            existing = result.get(parent_id)
            if existing is not None and existing.model_dump(mode="json") != annotation.model_dump(mode="json"):
                raise ValueError(f"conflicting_language_annotation:{parent_id}")
            result[parent_id] = annotation
    return result


def _load_latest_decisions_v1(
    paths: Sequence[str | Path],
    *,
    candidates: dict[str, LanguageGoalCandidateV1],
) -> dict[str, dict[str, Any]]:
    """Load lane decision ledgers keyed by candidate id in append order."""

    expected_by_id = {candidate.candidate_id: candidate for candidate in candidates.values()}
    result: dict[str, dict[str, Any]] = {}
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_file():
            continue
        for payload in iter_jsonl(path):
            candidate_id = str(payload.get("candidate_id") or "")
            parent_id = str(payload.get("parent_sample_id") or "")
            candidate = expected_by_id.get(candidate_id)
            if candidate is None:
                raise ValueError(f"decision_candidate_not_in_candidates:{candidate_id or 'missing'}")
            if parent_id != candidate.parent_sample_id:
                raise ValueError(f"decision_parent_mismatch:{candidate_id}")
            result[candidate_id] = payload
    return result


def _lane_for_parent_v1(parent_id: str, *, lane_count: int) -> int:
    """Map one stable parent identity to a deterministic provider lane."""

    return int(hashlib.sha256(parent_id.encode("utf-8")).hexdigest(), 16) % lane_count


def _write_json_v1(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one small deterministic JSON receipt."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "LANGUAGE_WAYPOINT_ANNOTATION_MERGE_REPORT_SCHEMA_VERSION_V1",
    "LANGUAGE_WAYPOINT_ANNOTATION_SHARD_REPORT_SCHEMA_VERSION_V1",
    "LANGUAGE_WAYPOINT_SIDECAR_MATERIALIZATION_REPORT_SCHEMA_VERSION_V1",
    "materialize_language_goal_sidecars_by_dataset_v1",
    "merge_language_goal_annotation_shards_v1",
    "partition_language_goal_annotation_inputs_v1",
]
