"""Deterministic retry-source construction from immutable quality ledgers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic


NAVIGATION_QUALITY_RETRY_MANIFEST_SCHEMA_VERSION_V1 = "urbannav_navigation_quality_retry_manifest_v1"


def build_geometry_retry_source_manifest_v1(
    *,
    source_jsonl: str | Path,
    geometry_decisions_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    retry_statuses: Iterable[str] = ("error", "rejected"),
) -> dict[str, Any]:
    """Build a source-only retry view for failed direct-geometry annotations.

    The output contains canonical source rows unchanged.  It intentionally does
    not copy raw teacher answers into model-visible data; the decision ledger
    remains the audit trail used by the next repair prompt.

    Args:
        source_jsonl: Immutable canonical V4 source ledger.
        geometry_decisions_jsonl: Append-only geometry decision ledger.
        output_jsonl: New source-only retry view destination.
        report_json: Immutable selection receipt destination.
        retry_statuses: Decision statuses eligible for a repair attempt.

    Returns:
        Input fingerprints, selected count, and status/error distributions.

    Raises:
        ValueError: If canonical source parents are duplicated.
    """

    allowed = frozenset(str(status).strip() for status in retry_statuses if str(status).strip())
    if not allowed:
        raise ValueError("geometry_retry_statuses_must_not_be_empty")
    sources = _load_sources_v1(source_jsonl)
    latest = _load_latest_geometry_decisions_v1(geometry_decisions_jsonl)
    selected: list[ReasoningSourceV4] = []
    decision_status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    orphan_or_stale = 0
    for parent_sample_id in sorted(sources):
        source = sources[parent_sample_id]
        decisions = [
            payload
            for (decision_parent, fingerprint, _profile_digest), payload in latest.items()
            if decision_parent == parent_sample_id and fingerprint == source.source_fingerprint
        ]
        if not decisions:
            continue
        # Multiple prompt revisions can coexist.  One failed revision is enough
        # to make this physical parent useful for an explicitly new repair pass.
        matching = [payload for payload in decisions if str(payload.get("status", "")) in allowed]
        if not matching:
            continue
        selected.append(source)
        for decision in matching:
            decision_status_counts[str(decision.get("status", "unknown"))] += 1
            error_counts.update(_decision_errors_v1(decision))
    for (parent_sample_id, fingerprint, _profile_digest) in latest:
        source = sources.get(parent_sample_id)
        if source is None or source.source_fingerprint != fingerprint:
            orphan_or_stale += 1
    write_jsonl_atomic(output_jsonl, [source.model_dump(mode="json") for source in selected])
    report = {
        "schema_version": NAVIGATION_QUALITY_RETRY_MANIFEST_SCHEMA_VERSION_V1,
        "source_jsonl": str(Path(source_jsonl).expanduser().resolve()),
        "geometry_decisions_jsonl": str(Path(geometry_decisions_jsonl).expanduser().resolve()),
        "source_jsonl_sha256": file_sha256(source_jsonl),
        "geometry_decisions_jsonl_sha256": file_sha256(geometry_decisions_jsonl),
        "output_jsonl": str(Path(output_jsonl).expanduser().resolve()),
        "retry_statuses": sorted(allowed),
        "source_count": len(sources),
        "selected_count": len(selected),
        "decision_status_counts": dict(decision_status_counts),
        "decision_error_counts": dict(error_counts),
        "orphan_or_stale_decision_count": orphan_or_stale,
        "source_manifest_unchanged": True,
    }
    report_path = Path(report_json).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _load_sources_v1(path: str | Path) -> dict[str, ReasoningSourceV4]:
    """Load exactly one canonical source revision per physical parent."""

    sources: dict[str, ReasoningSourceV4] = {}
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        parent_sample_id = source.observation.source_group_id
        if parent_sample_id in sources:
            raise ValueError(f"duplicate_source_parent_sample_id:{parent_sample_id}")
        sources[parent_sample_id] = source
    return sources


def _load_latest_geometry_decisions_v1(path: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Retain the final append-only geometry decision for each profile revision."""

    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for payload in iter_jsonl(path):
        if payload.get("kind") != "traversability_geometry":
            continue
        parent_sample_id = payload.get("parent_sample_id")
        source_fingerprint = payload.get("source_fingerprint")
        profile_digest = payload.get("profile_digest")
        if not all(
            isinstance(value, str) and value
            for value in (parent_sample_id, source_fingerprint, profile_digest)
        ):
            continue
        latest[(parent_sample_id, source_fingerprint, profile_digest)] = payload
    return latest


def _decision_errors_v1(decision: dict[str, Any]) -> list[str]:
    """Flatten provider/parser errors without interpreting their semantics."""

    errors: list[str] = []
    for attempt in decision.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        errors.extend(str(error) for error in attempt.get("errors", []) if str(error))
    return errors


__all__ = [
    "NAVIGATION_QUALITY_RETRY_MANIFEST_SCHEMA_VERSION_V1",
    "build_geometry_retry_source_manifest_v1",
]
