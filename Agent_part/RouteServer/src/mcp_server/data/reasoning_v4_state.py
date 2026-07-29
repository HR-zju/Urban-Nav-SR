"""Derived review and formal-training state for Reasoning V4.6 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    ActionReviewResolutionV4,
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
    resolution: ActionReviewResolutionV4 | None = None,
) -> ReasoningReleaseStateV4:
    """Derive review and formal-training eligibility fail closed.

    Training is eligible only when the record completed, all six stages were
    accepted, the candidate is structurally valid, an action audit exists,
    review is either not required or resolved, and the candidate revalidates
    against the source.
    A human-selected action differing from the frozen strong stamp is blocked;
    that case requires upstream source correction and re-annotation.

    Args:
        source: Frozen source row.
        annotation: Immutable annotation row.
        resolution: Latest append-only resolution for this audit.

    Returns:
        Derived release state and blockers.

    Raises:
        ValueError: If joined source, annotation, or resolution identities
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
    if (
        resolution is not None
        and resolution.audit_id != audit_id
    ):
        raise ValueError(
            "resolution audit_id does not match annotation"
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
        if resolution is not None:
            blockers.append("unexpected_resolution_without_trigger")
    elif resolution is None:
        review_status = "pending_human"
        blockers.append("human_review_pending")
    elif resolution.review_status == "rejected":
        review_status = "rejected"
        blockers.append("human_review_rejected")
    else:
        review_status = "resolved"
        assert resolution.resolved_action_label is not None
        strong = source.direction_supervision
        if strong is None:
            blockers.append("strong_direction_missing")
        elif (
            str(resolution.resolved_action_label)
            != str(strong.label_name)
        ):
            blockers.append(
                "resolution_requires_source_direction_correction"
            )
        if (
            annotation.candidate_target is None
            or str(
                annotation.candidate_target.reasoning.decision
                .selected_action_label
            )
            != str(resolution.resolved_action_label)
        ):
            blockers.append(
                "resolved_action_candidate_mismatch"
            )

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


def latest_action_resolutions_v4(
    rows: Iterable[dict],
) -> dict[str, ActionReviewResolutionV4]:
    """Project append-only resolution rows to their latest audit state.

    Args:
        rows: JSON-compatible resolution rows.

    Returns:
        Latest resolution per audit by timezone-aware ``resolved_at``.

    Raises:
        ValueError: If duplicate audit/timestamp entries disagree.
    """

    result: dict[str, ActionReviewResolutionV4] = {}
    seen: dict[tuple[str, str], ActionReviewResolutionV4] = {}
    for row in rows:
        resolution = ActionReviewResolutionV4.model_validate(row)
        key = (
            resolution.audit_id,
            resolution.resolved_at.isoformat(),
        )
        previous_same = seen.get(key)
        if (
            previous_same is not None
            and previous_same != resolution
        ):
            raise ValueError(
                "conflicting append-only resolution identity: "
                f"{key}"
            )
        seen[key] = resolution
        previous = result.get(resolution.audit_id)
        if (
            previous is None
            or resolution.resolved_at > previous.resolved_at
        ):
            result[resolution.audit_id] = resolution
    return result


def load_latest_action_resolutions_v4(
    path: str | Path | None,
) -> dict[str, ActionReviewResolutionV4]:
    """Load the latest resolution projection from JSONL.

    Args:
        path: Optional append-only resolution JSONL.

    Returns:
        Empty mapping when omitted, otherwise latest resolution per audit.
    """

    if path is None:
        return {}
    return latest_action_resolutions_v4(iter_jsonl(path))


def append_action_review_resolution_v4(
    path: str | Path,
    resolution: ActionReviewResolutionV4,
) -> None:
    """Append one validated human resolution without rewriting history.

    Args:
        path: Append-only resolution JSONL.
        resolution: Typed resolution row.

    Raises:
        ValueError: If the exact audit/timestamp identity already exists.
    """

    destination = Path(path).expanduser()
    if destination.exists():
        for row in iter_jsonl(destination):
            current = ActionReviewResolutionV4.model_validate(row)
            if (
                current.audit_id == resolution.audit_id
                and current.resolved_at == resolution.resolved_at
            ):
                raise ValueError(
                    "resolution audit/timestamp identity already exists"
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as handle:
        append_jsonl_row(
            handle,
            resolution.model_dump(
                mode="json",
                exclude_none=True,
            ),
        )


__all__ = [
    "ReasoningReleaseStateV4",
    "append_action_review_resolution_v4",
    "derive_reasoning_release_state_v4",
    "latest_action_resolutions_v4",
    "load_latest_action_resolutions_v4",
]
