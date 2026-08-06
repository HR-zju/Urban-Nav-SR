"""Immutable LVLM diagnosis and human-adjudication helpers for V4.6."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from mcp_server.contracts.reasoning_dataset_v4 import (
    ACTION_ADJUDICATION_V4_SCHEMA_VERSION,
    ACTION_REVIEW_FILTER_VERSION_V4,
    ACTION_REVIEW_DIAGNOSIS_V4_SCHEMA_VERSION,
    REASONING_REPAIR_REQUEST_V4_SCHEMA_VERSION,
    ActionAdjudicationPayloadV4,
    ActionAdjudicationV4,
    ActionReviewDiagnosisPayloadV4,
    ActionReviewDiagnosisRecordV4,
    ActionReviewDiagnosisV4,
    ReasoningAnnotationV4,
    ReasoningRepairRequestV4,
    ReasoningSourceV4,
    action_adjudication_id_v4,
    action_review_diagnosis_id_v4,
    reasoning_repair_request_id_v4,
)
from mcp_server.data.io.jsonl import append_jsonl_row, iter_jsonl


def bind_action_review_diagnosis_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    payload: ActionReviewDiagnosisPayloadV4,
) -> ActionReviewDiagnosisV4:
    """Bind an LVLM semantic payload to the current immutable audit.

    Args:
        source: Exact source revision under review.
        annotation: Exact annotation containing the deterministic audit.
        payload: Parsed four-way LVLM output without identity fields.

    Returns:
        Content-addressed diagnosis.

    Raises:
        ValueError: If source, annotation, or audit identities disagree.
    """

    if (
        source.observation.source_group_id != annotation.source_group_id
        or source.source_fingerprint != annotation.source_fingerprint
    ):
        raise ValueError("diagnosis source and annotation identity mismatch")
    audit = annotation.action_reconciliation
    if audit is None:
        raise ValueError("diagnosis requires an action audit")
    values = {
        "schema_version": ACTION_REVIEW_DIAGNOSIS_V4_SCHEMA_VERSION,
        "filter_version": ACTION_REVIEW_FILTER_VERSION_V4,
        "source_group_id": source.observation.source_group_id,
        "audit_id": audit.audit_id,
        **payload.model_dump(mode="json", exclude_none=True),
    }
    values["diagnosis_id"] = action_review_diagnosis_id_v4(values)
    return ActionReviewDiagnosisV4.model_validate(values)


def bind_action_adjudication_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    diagnosis: ActionReviewDiagnosisRecordV4,
    payload: ActionAdjudicationPayloadV4,
    *,
    reviewer: str,
    reviewed_at: datetime,
) -> ActionAdjudicationV4:
    """Bind one human payload to the exact diagnosis and audit lineage.

    Args:
        source: Exact source revision.
        annotation: Exact annotation and action audit.
        diagnosis: Completed LVLM diagnosis record reviewed by the human.
        payload: Human semantic decision exported by the review UI.
        reviewer: Stable reviewer identity.
        reviewed_at: Timezone-aware review timestamp.

    Returns:
        Content-addressed append-only adjudication.

    Raises:
        ValueError: If any parent identity or diagnosis state disagrees.
    """

    audit = annotation.action_reconciliation
    if (
        audit is None
        or diagnosis.record_status != "complete"
        or diagnosis.diagnosis is None
        or source.observation.source_group_id != annotation.source_group_id
        or source.source_fingerprint != annotation.source_fingerprint
        or diagnosis.source_group_id != annotation.source_group_id
        or diagnosis.source_fingerprint != source.source_fingerprint
        or diagnosis.audit_id != audit.audit_id
    ):
        raise ValueError(
            "adjudication requires one completed matching diagnosis lineage"
        )
    lvlm = diagnosis.diagnosis
    compared_fields = (
        payload.final_verdict == lvlm.verdict,
        payload.error_subtype == lvlm.error_subtype,
        payload.affected_components == lvlm.affected_components,
        payload.repair_action == lvlm.repair_action,
    )
    if (
        payload.diagnosis_disposition == "confirmed"
        and not all(compared_fields)
    ):
        raise ValueError(
            "confirmed adjudication must preserve the LVLM verdict, subtype, "
            "components, and repair action"
        )
    if (
        payload.diagnosis_disposition == "overridden"
        and all(compared_fields)
    ):
        raise ValueError(
            "overridden adjudication must change at least one diagnosed field"
        )
    reviewed_at_value = reviewed_at.isoformat()
    if reviewed_at.utcoffset() is not None and (
        reviewed_at.utcoffset().total_seconds() == 0
    ):
        reviewed_at_value = reviewed_at_value.replace("+00:00", "Z")
    values = {
        "schema_version": ACTION_ADJUDICATION_V4_SCHEMA_VERSION,
        "source_group_id": source.observation.source_group_id,
        "source_fingerprint": source.source_fingerprint,
        "audit_id": audit.audit_id,
        "diagnosis_id": diagnosis.diagnosis.diagnosis_id,
        "reviewer": str(reviewer),
        "reviewed_at": reviewed_at_value,
        **payload.model_dump(mode="json", exclude_none=True),
    }
    values["adjudication_id"] = action_adjudication_id_v4(values)
    return ActionAdjudicationV4.model_validate(values)


def build_reasoning_repair_request_v4(
    adjudication: ActionAdjudicationV4,
) -> ReasoningRepairRequestV4 | None:
    """Derive an upstream repair request from one human adjudication.

    Args:
        adjudication: Valid human decision.

    Returns:
        Content-addressed request for repair actions, otherwise ``None`` for
        accepted or quarantined samples.
    """

    if adjudication.repair_action in {
        "keep_without_change",
        "quarantine",
    }:
        return None
    values = {
        "schema_version": REASONING_REPAIR_REQUEST_V4_SCHEMA_VERSION,
        "source_group_id": adjudication.source_group_id,
        "source_fingerprint": adjudication.source_fingerprint,
        "audit_id": adjudication.audit_id,
        "adjudication_id": adjudication.adjudication_id,
        "repair_action": adjudication.repair_action,
        "affected_components": list(
            adjudication.affected_components
        ),
        "fields_to_inspect": list(adjudication.fields_to_inspect),
    }
    values["repair_request_id"] = reasoning_repair_request_id_v4(values)
    return ReasoningRepairRequestV4.model_validate(values)


def latest_action_review_diagnoses_v4(
    rows: Iterable[dict[str, Any]],
) -> dict[str, ActionReviewDiagnosisRecordV4]:
    """Project append-only diagnosis records to the latest row per audit.

    Args:
        rows: JSON-compatible diagnosis-record rows in append order.

    Returns:
        Latest record per audit ID.
    """

    result: dict[str, ActionReviewDiagnosisRecordV4] = {}
    for row in rows:
        record = ActionReviewDiagnosisRecordV4.model_validate(row)
        result[record.audit_id] = record
    return result


def load_latest_action_review_diagnoses_v4(
    path: str | Path | None,
) -> dict[str, ActionReviewDiagnosisRecordV4]:
    """Load latest diagnosis records from optional JSONL.

    Args:
        path: Optional append-only diagnosis JSONL.

    Returns:
        Empty mapping when omitted, otherwise latest records per audit.
    """

    if path is None:
        return {}
    return latest_action_review_diagnoses_v4(iter_jsonl(path))


def append_action_review_diagnosis_v4(
    path: str | Path,
    record: ActionReviewDiagnosisRecordV4,
) -> None:
    """Append one diagnosis record while rejecting duplicate final identity.

    Args:
        path: Append-only diagnosis JSONL.
        record: Valid two-stage diagnosis record.

    Raises:
        ValueError: If the same completed diagnosis identity already exists.
    """

    destination = Path(path).expanduser()
    if destination.exists() and record.diagnosis is not None:
        for row in iter_jsonl(destination):
            current = ActionReviewDiagnosisRecordV4.model_validate(row)
            if (
                current.diagnosis is not None
                and current.diagnosis.diagnosis_id
                == record.diagnosis.diagnosis_id
            ):
                raise ValueError(
                    "diagnosis identity already exists"
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as handle:
        append_jsonl_row(
            handle,
            record.model_dump(mode="json", exclude_none=True),
        )


__all__ = [
    "append_action_review_diagnosis_v4",
    "bind_action_adjudication_v4",
    "bind_action_review_diagnosis_v4",
    "build_reasoning_repair_request_v4",
    "latest_action_review_diagnoses_v4",
    "load_latest_action_review_diagnoses_v4",
]
