"""Export Reasoning V4.5 action-audit and human-review projections."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.contracts.direction_profile import (
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_v4_reconciliation import (
    action_reconciliation_review_priority_v4,
    action_reconciliation_review_required_v4,
    validate_action_reconciliation_audit_v4,
)
from mcp_server.data.reasoning_v4_state import (
    derive_reasoning_release_state_v4,
    load_latest_action_resolutions_v4,
)


RECONCILIATION_REVIEW_REPORT_SCHEMA_V4 = (
    "urbannav_action_reconciliation_review_report_v2"
)


def export_action_reconciliation_review_v4(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    audit_jsonl: str | Path,
    queue_jsonl: str | Path,
    report_json: str | Path,
    resolution_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Export immutable audits and unresolved human-review rows.

    Args:
        source_jsonl: Frozen V4.5 source manifest.
        annotation_jsonl: Append-only annotation manifest.
        audit_jsonl: Destination joined audit projection.
        queue_jsonl: Destination pending-review projection.
        report_json: Destination deterministic summary.
        resolution_jsonl: Optional append-only human-resolution sidecar.

    Returns:
        Exact report payload written to disk.

    Raises:
        ValueError: If joined identities or deterministic audits drift.
    """

    sources = _source_index_v4(source_jsonl)
    annotations = _latest_annotation_index_v4(
        annotation_jsonl
    )
    resolutions = load_latest_action_resolutions_v4(
        resolution_jsonl
    )
    audit_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    delta_bins: Counter[str] = Counter()
    errors: list[str] = []
    seen_audit_ids: set[str] = set()

    for source_id, source in sources.items():
        counts["sources"] += 1
        annotation = annotations.get(source_id)
        if annotation is None:
            counts["missing_annotations"] += 1
            errors.append(f"missing_annotation:{source_id}")
            continue
        counts[
            f"record_status:{annotation.record_status}"
        ] += 1
        if (
            annotation.source_fingerprint
            != source.source_fingerprint
        ):
            errors.append(
                f"source_fingerprint_mismatch:{source_id}"
            )
            continue
        counts[
            f"composition:{annotation.composition_record.status}"
        ] += 1
        audit = annotation.action_reconciliation
        resolution = (
            resolutions.get(audit.audit_id)
            if audit is not None
            else None
        )
        state = derive_reasoning_release_state_v4(
            source,
            annotation,
            resolution=resolution,
        )
        counts[
            f"review_status:{state.review_status}"
        ] += 1
        counts[
            f"training_status:{state.training_status}"
        ] += 1
        counts["all_sections_accepted"] += int(
            state.all_sections_accepted
        )
        counts["training_eligible"] += int(
            state.training_eligible
        )
        if audit is None:
            if source.direction_supervision is not None:
                errors.append(
                    "strong_direction_without_action_audit:"
                    f"{source_id}"
                )
            continue
        if annotation.decision_proposal is None:
            errors.append(
                "action_audit_without_decision_proposal:"
                f"{source_id}"
            )
            continue
        validate_action_reconciliation_audit_v4(
            source,
            annotation.grounding_outputs,
            annotation.decision_proposal,
            audit,
        )
        if audit.audit_id in seen_audit_ids:
            errors.append(
                f"duplicate_action_audit_id:{audit.audit_id}"
            )
            continue
        seen_audit_ids.add(audit.audit_id)
        counts["audit_rows"] += 1
        required = action_reconciliation_review_required_v4(
            audit
        )
        priority = action_reconciliation_review_priority_v4(
            audit
        )
        proposal_label = str(
            annotation.decision_proposal.selected_action_label
        )
        assert source.direction_supervision is not None
        strong_label = str(
            source.direction_supervision.label_name
        )
        confusion[(proposal_label, strong_label)] += 1
        tolerance = float(
            resolve_stage_a_direction_profile(
                source.observation.direction_context.action_profile_id
            )["intent_compatibility_tolerance_deg"]
        )
        delta_bins[
            _angular_delta_bin_v4(
                audit.comparison.angular_delta_deg,
                tolerance_deg=tolerance,
            )
        ] += 1
        counts[f"review_priority:{priority}"] += 1
        for code in audit.review_trigger_codes:
            counts[f"review_trigger:{code}"] += 1

        audit_rows.append(
            {
                "source_group_id": source_id,
                "audit": audit.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "review_status": state.review_status,
                "training_status": state.training_status,
                "training_eligible": state.training_eligible,
                "blockers": list(state.blockers),
            }
        )
        if required and state.review_status == "pending_human":
            queue_rows.append(
                _queue_projection_v4(
                    source,
                    annotation,
                    priority=priority,
                )
            )

    orphan_ids = sorted(set(annotations) - set(sources))
    if orphan_ids:
        errors.append(
            f"annotations_without_source:{len(orphan_ids)}"
        )
    orphan_resolutions = sorted(
        set(resolutions) - seen_audit_ids
    )
    if orphan_resolutions:
        errors.append(
            "resolutions_without_audit:"
            f"{len(orphan_resolutions)}"
        )
    write_jsonl_atomic(audit_jsonl, audit_rows)
    write_jsonl_atomic(queue_jsonl, queue_rows)
    report = {
        "schema_version": (
            RECONCILIATION_REVIEW_REPORT_SCHEMA_V4
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "resolution_jsonl": (
            str(resolution_jsonl)
            if resolution_jsonl is not None
            else None
        ),
        "audit_jsonl": str(audit_jsonl),
        "queue_jsonl": str(queue_jsonl),
        "release_track": "review_preview",
        "formal_release_ready": (
            not errors
            and counts["training_eligible"] > 0
            and counts["review_status:pending_human"] == 0
        ),
        "counts": dict(counts),
        "proposal_strong_confusion": {
            f"{proposal}->{strong}": count
            for (proposal, strong), count in sorted(
                confusion.items()
            )
        },
        "angular_delta_bins": dict(sorted(delta_bins.items())),
        "errors": errors[:200],
        "passed": not errors,
    }
    destination = Path(report_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _queue_projection_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    *,
    priority: str,
) -> dict[str, Any]:
    """Expand an audit into one reviewer-facing queue row."""

    audit = annotation.action_reconciliation
    proposal = annotation.decision_proposal
    direction = source.direction_supervision
    assert audit is not None
    assert proposal is not None
    assert direction is not None
    profile_id = (
        source.observation.direction_context.action_profile_id
    )
    tolerance = float(
        resolve_stage_a_direction_profile(profile_id)[
            "intent_compatibility_tolerance_deg"
        ]
    )
    return {
        "queue_key": audit.audit_id,
        "audit_id": audit.audit_id,
        "source_group_id": (
            source.observation.source_group_id
        ),
        "review_status": "pending_human",
        "review_priority": priority,
        "action_profile_id": profile_id,
        "action_profile_version": (
            source.observation.direction_context
            .action_profile_version
        ),
        "profile_tolerance_deg": tolerance,
        "teacher_proposal_action": str(
            proposal.selected_action_label
        ),
        "strong_direction_action": str(
            direction.label_name
        ),
        "composer_candidate_action": (
            str(
                annotation.composer_candidate
                .selected_action_label
            )
            if annotation.composer_candidate is not None
            else None
        ),
        "comparison": audit.comparison.model_dump(
            mode="json"
        ),
        "probe_projection": (
            audit.probe_projection.model_dump(
                mode="json",
                exclude_none=True,
            )
        ),
        "checks": audit.checks.model_dump(mode="json"),
        "review_trigger_codes": list(
            audit.review_trigger_codes
        ),
        "composition_record": (
            annotation.composition_record.model_dump(
                mode="json",
                exclude_none=True,
            )
        ),
    }


def _source_index_v4(
    path: str | Path,
) -> dict[str, ReasoningSourceV4]:
    """Return unique sources keyed by source group."""

    result: dict[str, ReasoningSourceV4] = {}
    for row in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(row)
        source_id = source.observation.source_group_id
        if source_id in result:
            raise ValueError(
                f"duplicate source_group_id: {source_id}"
            )
        result[source_id] = source
    return result


def _angular_delta_bin_v4(
    value: float | None,
    *,
    tolerance_deg: float,
) -> str:
    """Bucket angular delta relative to profile tolerance."""

    if value is None:
        return "not_applicable"
    delta = float(value)
    tolerance = float(tolerance_deg)
    if delta == 0.0:
        return "exact"
    if delta <= tolerance:
        return "within_profile_tolerance"
    if delta <= 2.0 * tolerance:
        return "above_tolerance_to_2x"
    return "above_2x_tolerance"


def _latest_annotation_index_v4(
    path: str | Path,
) -> dict[str, ReasoningAnnotationV4]:
    """Return latest append-only annotation per source group."""

    result: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV4.model_validate(
            row
        )
        result[annotation.source_group_id] = annotation
    return result


__all__ = [
    "RECONCILIATION_REVIEW_REPORT_SCHEMA_V4",
    "export_action_reconciliation_review_v4",
]
