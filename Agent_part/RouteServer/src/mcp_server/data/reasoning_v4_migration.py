"""Deterministic canonical-vocabulary migration for Reasoning V4.6 rows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.agent.reasoning_v4_validation import (
    sanitize_reasoning_section_evidence_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    GroundingOutputsV4,
    REASONING_ANNOTATION_V4_SCHEMA_VERSION,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.contracts.reasoning_v4 import (
    ReasoningTargetV4,
    ReasoningTraceV4,
)
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.reasoning_v4_reconciliation import (
    build_action_reconciliation_audit_v4,
)
from mcp_server.data.reasoning_v4_state import (
    derive_reasoning_release_state_v4,
)


REASONING_V4_CANONICAL_MIGRATION_ID = (
    "urbannav_reasoning_v4_6_canonical_vocabulary_migration_v1"
)

_GROUNDING_FIELDS = (
    ("TaskGrounding", "task_grounding"),
    ("SceneGrounding", "scene_grounding"),
    ("LocalTraversability", "local_traversability"),
    (
        "NavigationContextGrounding",
        "navigation_context_grounding",
    ),
    (
        "InteractionConstraintGrounding",
        "interaction_constraint_grounding",
    ),
)


def migrate_reasoning_annotation_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
) -> tuple[ReasoningAnnotationV4, tuple[str, ...]]:
    """Canonicalize bounded aliases and rebuild derived action audit facts.

    The source, raw stage answers, Decision proposal, and final selected action
    are preserved.  Only deterministic vocabulary normalization and fields
    derived from the normalized grounding payload are refreshed.
    """

    if (
        source.observation.source_group_id != annotation.source_group_id
        or source.source_fingerprint != annotation.source_fingerprint
    ):
        raise ValueError("reasoning migration source/annotation mismatch")

    updates: dict[str, Any] = {}
    warnings: list[str] = []
    for section_name, field_name in _GROUNDING_FIELDS:
        section = getattr(annotation.grounding_outputs, field_name)
        if section is None:
            updates[field_name] = None
            continue
        normalized, section_warnings = sanitize_reasoning_section_evidence_v4(
            section_name,
            section,
            observation=source.observation,
        )
        updates[field_name] = normalized
        warnings.extend(section_warnings)
    groundings = GroundingOutputsV4(**updates)

    candidate_target = annotation.candidate_target
    if candidate_target is not None and groundings.complete:
        trace = candidate_target.reasoning
        # Candidate grounding sections are projections of the same accepted
        # stage outputs; rebuild them from the normalized canonical payload.
        candidate_target = ReasoningTargetV4(
            reasoning=ReasoningTraceV4(
                task_grounding=groundings.task_grounding,
                scene_grounding=groundings.scene_grounding,
                local_traversability=groundings.local_traversability,
                navigation_context_grounding=(
                    groundings.navigation_context_grounding
                ),
                interaction_constraint_grounding=(
                    groundings.interaction_constraint_grounding
                ),
                decision=trace.decision,
            )
        )

    audit = annotation.action_reconciliation
    if annotation.decision_proposal is not None and groundings.complete:
        audit = build_action_reconciliation_audit_v4(
            source,
            groundings,
            annotation.decision_proposal,
        )
    elif groundings != annotation.grounding_outputs:
        # An audit cannot remain attached to a changed incomplete grounding
        # payload, and it cannot be rebuilt until all five sections exist.
        audit = None

    metadata = dict(annotation.metadata)
    history = list(metadata.get("deterministic_migrations") or [])
    if REASONING_V4_CANONICAL_MIGRATION_ID not in history:
        history.append(REASONING_V4_CANONICAL_MIGRATION_ID)
    metadata["deterministic_migrations"] = history
    annotation_warnings = list(dict.fromkeys(
        [*annotation.warnings, *warnings]
    ))
    migrated = ReasoningAnnotationV4.model_validate(
        annotation.model_copy(
            update={
                "grounding_outputs": groundings,
                "candidate_target": candidate_target,
                "action_reconciliation": audit,
                "warnings": annotation_warnings,
                "metadata": metadata,
            }
        ).model_dump(mode="json", exclude_none=True)
    )
    return migrated, tuple(dict.fromkeys(warnings))


def migrate_reasoning_annotation_manifest_v4(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Write a new migrated manifest and a release-gate audit report."""

    source_path = Path(source_jsonl).expanduser()
    annotation_path = Path(annotation_jsonl).expanduser()
    output_path = Path(output_jsonl).expanduser()
    sources = {
        source.observation.source_group_id: source
        for source in (
            ReasoningSourceV4.model_validate(row)
            for row in iter_jsonl(source_path)
        )
    }
    counts: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for raw in iter_jsonl(annotation_path):
        if raw.get("schema_version") != REASONING_ANNOTATION_V4_SCHEMA_VERSION:
            raise ValueError(
                "canonical vocabulary migration requires a structurally "
                "upgraded Reasoning V4.6 annotation manifest"
            )
        annotation = ReasoningAnnotationV4.model_validate(raw)
        try:
            source = sources[annotation.source_group_id]
        except KeyError as exc:
            raise ValueError(
                "reasoning migration source row is missing: "
                f"{annotation.source_group_id}"
            ) from exc
        migrated, warnings = migrate_reasoning_annotation_v4(
            source,
            annotation,
        )
        counts["input_rows"] += 1
        counts["changed_rows"] += int(migrated != annotation)
        counts["audit_refreshed"] += int(
            migrated.action_reconciliation
            != annotation.action_reconciliation
        )
        for warning in warnings:
            counts[f"normalization:{warning}"] += 1
        release = derive_reasoning_release_state_v4(source, migrated)
        counts["formal_training_eligible"] += int(
            release.formal_training_eligible
        )
        counts["pending_human"] += int(
            release.review_status == "pending_human"
        )
        for blocker in release.blockers:
            counts[f"release_blocker:{blocker}"] += 1
        output_rows.append(
            migrated.model_dump(mode="json", exclude_none=True)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in output_rows
    )
    output_path.write_text(content, encoding="utf-8")
    report = {
        "schema_version": REASONING_V4_CANONICAL_MIGRATION_ID,
        "source_artifact": _artifact_record(source_path),
        "input_annotation_artifact": _artifact_record(annotation_path),
        "output_annotation_artifact": {
            "path": str(output_path),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "rows": len(output_rows),
        },
        "counts": dict(sorted(counts.items())),
        "semantic_fields_preserved": [
            "stage_records.raw_answer",
            "decision_proposal",
            "composer_candidate",
            "candidate_target.reasoning.decision",
        ],
        "human_review_transferred": False,
    }
    if report_json is not None:
        report_path = Path(report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return report


def _artifact_record(path: Path) -> dict[str, Any]:
    """Return immutable input evidence for one migration report."""

    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


__all__ = [
    "REASONING_V4_CANONICAL_MIGRATION_ID",
    "migrate_reasoning_annotation_manifest_v4",
    "migrate_reasoning_annotation_v4",
]
