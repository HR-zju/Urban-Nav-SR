"""Derived review and formal-training state for Reasoning V4.6 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    ActionAdjudicationV4,
    ActionReviewDiagnosisRecordV4,
    ProjectedReviewStatusV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.data.jsonl_io import append_jsonl_row, iter_jsonl
from mcp_server.data.reasoning_v4_reconciliation import (
    action_reconciliation_review_required_v4,
)


@dataclass(frozen=True)
class ReasoningReleaseStateV4:
    """Carry derived, non-persisted release axes.

    Args:
        record_complete: Annotation pipeline completed.
        all_sections_accepted: All six independent stages accepted.
        candidate_structurally_valid: Candidate exists and structurally validates.
        audit_id: Parent action-audit identity.
        review_status: Derived review state.
        formal_training_eligible: Exact formal eligibility bit.
        blockers: Stable fail-closed blocker codes.
    """

    record_complete: bool
    all_sections_accepted: bool
    candidate_structurally_valid: bool
    audit_id: str | None
    review_status: ProjectedReviewStatusV4
    formal_training_eligible: bool
    blockers: tuple[str, ...]


def derive_reasoning_release_state_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    *,
    diagnosis: ActionReviewDiagnosisRecordV4 | None = None,
    adjudication: ActionAdjudicationV4 | None = None,
) -> ReasoningReleaseStateV4:
    """Derive review and formal-training eligibility fail closed.

    Training is eligible only when the record completed, all six stages were
    accepted, the candidate is structurally valid, an action audit exists,
    review is either not required or resolved, and the candidate revalidates
    against the source.  LVLM filtering never changes eligibility.  A human
    adjudication resolves the current audit only through
    ``keep_without_change``; every repair or quarantine action rejects the old
    lineage until a new source/annotation/audit is produced.

    Args:
        source: Frozen source row.
        annotation: Immutable annotation row.
        diagnosis: Latest two-stage LVLM diagnosis for this audit.
        adjudication: Latest append-only human adjudication for this audit.

    Returns:
        Derived release state and blockers.

    Raises:
        ValueError: If joined source, annotation, or adjudication identities
            disagree.
    """

    if (
        source.observation.source_group_id
        != annotation.source_group_id
        or source.source_fingerprint
        != annotation.source_fingerprint
    ):
        raise ValueError(
            "source and annotation identity mismatch"
        )
    audit = annotation.action_reconciliation
    audit_id = audit.audit_id if audit is not None else None
    if diagnosis is not None and (
        diagnosis.audit_id != audit_id
        or diagnosis.source_group_id
        != source.observation.source_group_id
        or diagnosis.source_fingerprint != source.source_fingerprint
    ):
        raise ValueError(
            "diagnosis identity does not match source/annotation"
        )
    if (
        adjudication is not None
        and (
            adjudication.audit_id != audit_id
            or adjudication.source_group_id
            != source.observation.source_group_id
            or adjudication.source_fingerprint
            != source.source_fingerprint
        )
    ):
        raise ValueError(
            "adjudication identity does not match source/annotation"
        )

    blockers: list[str] = []
    record_complete = annotation.record_status == "complete"
    if not record_complete:
        blockers.append("record_not_complete")
    all_sections_accepted = annotation.all_sections_accepted
    if not all_sections_accepted:
        blockers.append("sections_not_all_accepted")
    candidate_structurally_valid = (
        annotation.candidate_structure_record.status == "valid"
        and annotation.candidate_target is not None
        and annotation.composer_candidate is not None
    )
    if not candidate_structurally_valid:
        blockers.append("candidate_structure_not_valid")
    if audit is None:
        review_status: ProjectedReviewStatusV4 = (
            "not_required"
        )
        blockers.append("action_audit_missing")
    elif not action_reconciliation_review_required_v4(audit):
        review_status = "not_required"
        if diagnosis is not None:
            blockers.append("unexpected_diagnosis_without_trigger")
        if adjudication is not None:
            blockers.append("unexpected_adjudication_without_trigger")
    elif adjudication is None:
        review_status = "pending_human"
        blockers.append("human_review_pending")
    elif (
        diagnosis is None
        or diagnosis.record_status != "complete"
        or diagnosis.diagnosis is None
        or diagnosis.diagnosis.diagnosis_id
        != adjudication.diagnosis_id
    ):
        review_status = "rejected"
        blockers.append("adjudication_diagnosis_lineage_invalid")
    elif adjudication.repair_action != "keep_without_change":
        review_status = "rejected"
        if adjudication.repair_action == "quarantine":
            blockers.append("human_review_quarantined")
        else:
            blockers.append(
                "human_review_repair_required:"
                f"{adjudication.repair_action}"
            )
    else:
        review_status = "resolved"

    if annotation.candidate_target is not None:
        validation = validate_reasoning_target_v4(
            annotation.candidate_target,
            observation=source.observation,
            direction_supervision=source.direction_supervision,
        )
        blockers.extend(
            f"candidate_revalidation:{error}"
            for error in validation.errors
        )

    formal_training_eligible = not blockers and review_status in {
        "not_required",
        "resolved",
    }
    return ReasoningReleaseStateV4(
        record_complete=record_complete,
        all_sections_accepted=all_sections_accepted,
        candidate_structurally_valid=candidate_structurally_valid,
        audit_id=audit_id,
        review_status=review_status,
        formal_training_eligible=formal_training_eligible,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def latest_action_adjudications_v4(
    rows: Iterable[dict],
) -> dict[str, ActionAdjudicationV4]:
    """Project append-only adjudication rows to their latest audit state.

    Args:
        rows: JSON-compatible adjudication rows.

    Returns:
        Latest adjudication per audit by timezone-aware ``reviewed_at``.

    Raises:
        ValueError: If duplicate audit/timestamp entries disagree.
    """

    result: dict[str, ActionAdjudicationV4] = {}
    seen: dict[tuple[str, str], ActionAdjudicationV4] = {}
    for row in rows:
        adjudication = ActionAdjudicationV4.model_validate(row)
        key = (
            adjudication.audit_id,
            adjudication.reviewed_at.isoformat(),
        )
        previous_same = seen.get(key)
        if (
            previous_same is not None
            and previous_same != adjudication
        ):
            raise ValueError(
                "conflicting append-only adjudication identity: "
                f"{key}"
            )
        seen[key] = adjudication
        previous = result.get(adjudication.audit_id)
        if (
            previous is None
            or adjudication.reviewed_at > previous.reviewed_at
        ):
            result[adjudication.audit_id] = adjudication
    return result


def load_latest_action_adjudications_v4(
    path: str | Path | None,
) -> dict[str, ActionAdjudicationV4]:
    """Load the latest human-adjudication projection from JSONL.

    Args:
        path: Optional append-only adjudication JSONL.

    Returns:
        Empty mapping when omitted, otherwise latest adjudication per audit.
    """

    if path is None:
        return {}
    return latest_action_adjudications_v4(iter_jsonl(path))


def append_action_adjudication_v4(
    path: str | Path,
    adjudication: ActionAdjudicationV4,
) -> None:
    """Append one validated human adjudication without rewriting history.

    Args:
        path: Append-only adjudication JSONL.
        adjudication: Typed adjudication row.

    Raises:
        ValueError: If the exact audit/timestamp identity already exists.
    """

    destination = Path(path).expanduser()
    if destination.exists():
        for row in iter_jsonl(destination):
            current = ActionAdjudicationV4.model_validate(row)
            if (
                current.adjudication_id
                == adjudication.adjudication_id
                or (
                    current.audit_id == adjudication.audit_id
                    and current.reviewed_at == adjudication.reviewed_at
                )
            ):
                raise ValueError(
                    "adjudication identity or audit/timestamp already exists"
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as handle:
        append_jsonl_row(
            handle,
            adjudication.model_dump(
                mode="json",
                exclude_none=True,
            ),
        )


__all__ = [
    "ReasoningReleaseStateV4",
    "append_action_adjudication_v4",
    "derive_reasoning_release_state_v4",
    "latest_action_adjudications_v4",
    "load_latest_action_adjudications_v4",
]
