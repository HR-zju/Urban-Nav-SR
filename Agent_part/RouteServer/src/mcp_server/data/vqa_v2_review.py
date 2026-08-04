"""Human-review decisions and deterministic VQA V2 artifact filtering."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQALabelSourceV2,
    VQATrainingViewV2,
)
from mcp_server.contracts.vqa_v2 import VQATaskIdV2
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic


VQA_REVIEW_DECISION_V2_SCHEMA_VERSION = (
    "urbannav_navigation_vqa_review_decision_v2_0"
)
VQA_REVIEW_FILTER_REPORT_V2_SCHEMA_VERSION = (
    "urbannav_navigation_vqa_review_filter_report_v2_0"
)

VQAReviewVerdictV2 = Literal["keep", "reject", "needs_fix"]
VQAReviewIssueCodeV2 = Literal[
    "question_not_grounded",
    "answer_incorrect",
    "answer_unsupported",
    "modality_mismatch",
    "evidence_reference_error",
    "task_not_applicable",
    "unsafe_navigation_judgment",
    "schema_or_format_error",
    "duplicate_or_leakage",
    "other",
]
VQAUnreviewedPolicyV2 = Literal["error", "keep", "drop"]


class VQAReviewDecisionV2(BaseModel):
    """Bind one human verdict to an immutable VQA annotation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        VQA_REVIEW_DECISION_V2_SCHEMA_VERSION
    ] = VQA_REVIEW_DECISION_V2_SCHEMA_VERSION
    review_set_id: str = Field(min_length=1, max_length=160)
    annotation_id: str = Field(min_length=1, max_length=400)
    source_group_id: str = Field(min_length=1, max_length=240)
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    task_id: VQATaskIdV2
    label_source: VQALabelSourceV2
    reviewer: str = Field(min_length=1, max_length=128)
    reviewed_at: datetime
    verdict: VQAReviewVerdictV2
    issue_codes: list[VQAReviewIssueCodeV2] = Field(
        default_factory=list,
        max_length=10,
    )
    note: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_decision(self) -> "VQAReviewDecisionV2":
        """Require actionable issues and timezone-aware review time."""

        if self.reviewed_at.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")
        if len(set(self.issue_codes)) != len(self.issue_codes):
            raise ValueError("review issue codes must be unique")
        if self.verdict != "keep" and not self.issue_codes:
            raise ValueError(
                "reject and needs_fix decisions require issue_codes"
            )
        return self


def load_latest_vqa_review_decisions_v2(
    path: str | Path,
) -> dict[str, VQAReviewDecisionV2]:
    """Load the latest valid decision per annotation.

    Args:
        path: Append-order review decision JSONL.

    Returns:
        Latest decision keyed by annotation ID.

    Raises:
        ValueError: If revisions for one annotation change immutable identity.
    """

    return latest_vqa_review_decisions_v2(iter_jsonl(path))


def latest_vqa_review_decisions_v2(
    rows: Iterable[dict[str, Any]],
) -> dict[str, VQAReviewDecisionV2]:
    """Project append-order rows to the latest decision per annotation.

    Args:
        rows: JSON-compatible decision rows in append order.

    Returns:
        Latest decision keyed by annotation ID.

    Raises:
        ValueError: If revisions for one annotation change immutable identity.
    """

    result: dict[str, VQAReviewDecisionV2] = {}
    for row in rows:
        decision = VQAReviewDecisionV2.model_validate(row)
        previous = result.get(decision.annotation_id)
        if previous is not None and _decision_identity(previous) != (
            _decision_identity(decision)
        ):
            raise ValueError(
                "review revisions changed immutable annotation identity: "
                f"{decision.annotation_id}"
            )
        result[decision.annotation_id] = decision
    return result


def filter_vqa_artifacts_by_review_v2(
    *,
    annotation_jsonl: str | Path,
    decision_jsonl: str | Path,
    output_annotation_jsonl: str | Path,
    rejected_annotation_jsonl: str | Path | None = None,
    needs_fix_annotation_jsonl: str | Path | None = None,
    training_view_jsonl: str | Path | None = None,
    output_training_view_jsonl: str | Path | None = None,
    unreviewed_policy: VQAUnreviewedPolicyV2 = "error",
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Apply human decisions to annotations and optional training views.

    Args:
        annotation_jsonl: Canonical accepted VQA annotation artifact.
        decision_jsonl: Append-order human review decisions.
        output_annotation_jsonl: Filtered annotation destination.
        rejected_annotation_jsonl: Optional rejected-row destination.
        needs_fix_annotation_jsonl: Optional repair-queue destination.
        training_view_jsonl: Optional matching VQA training views.
        output_training_view_jsonl: Filtered view destination when views are
            supplied.
        unreviewed_policy: Explicit handling of annotations without decisions.
        report_json: Optional machine-readable filter report.

    Returns:
        Exact decision, annotation, and training-view counts.

    Raises:
        ValueError: If identities disagree, decisions are orphaned, or the
            unreviewed policy is ``error`` and any annotation lacks a decision.
    """

    if (training_view_jsonl is None) != (
        output_training_view_jsonl is None
    ):
        raise ValueError(
            "training view input and output must be supplied together"
        )
    decisions = load_latest_vqa_review_decisions_v2(decision_jsonl)
    annotation_counts, _seen_decisions = _partition_annotations(
        annotation_jsonl=annotation_jsonl,
        decisions=decisions,
        output_annotation_jsonl=output_annotation_jsonl,
        rejected_annotation_jsonl=rejected_annotation_jsonl,
        needs_fix_annotation_jsonl=needs_fix_annotation_jsonl,
        unreviewed_policy=unreviewed_policy,
    )
    view_counts: Counter[str] = Counter()
    if training_view_jsonl is not None:
        view_counts = _filter_training_views(
            training_view_jsonl=training_view_jsonl,
            output_training_view_jsonl=output_training_view_jsonl,
            decisions=decisions,
            unreviewed_policy=unreviewed_policy,
        )

    decision_counts = Counter(
        decision.verdict for decision in decisions.values()
    )
    report: dict[str, Any] = {
        "schema_version": VQA_REVIEW_FILTER_REPORT_V2_SCHEMA_VERSION,
        "annotation_jsonl": str(Path(annotation_jsonl).expanduser()),
        "decision_jsonl": str(Path(decision_jsonl).expanduser()),
        "output_annotation_jsonl": str(
            Path(output_annotation_jsonl).expanduser()
        ),
        "output_training_view_jsonl": (
            str(Path(output_training_view_jsonl).expanduser())
            if output_training_view_jsonl is not None
            else None
        ),
        "unreviewed_policy": unreviewed_policy,
        "decision_counts": dict(sorted(decision_counts.items())),
        "annotation_counts": dict(sorted(annotation_counts.items())),
        "training_view_counts": dict(sorted(view_counts.items())),
    }
    if report_json is not None:
        destination = Path(report_json).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _partition_annotations(
    *,
    annotation_jsonl: str | Path,
    decisions: dict[str, VQAReviewDecisionV2],
    output_annotation_jsonl: str | Path,
    rejected_annotation_jsonl: str | Path | None,
    needs_fix_annotation_jsonl: str | Path | None,
    unreviewed_policy: VQAUnreviewedPolicyV2,
) -> tuple[Counter[str], set[str]]:
    """Stream annotations into transactional verdict partitions."""

    destinations = {
        "kept": Path(output_annotation_jsonl).expanduser(),
    }
    if rejected_annotation_jsonl is not None:
        destinations["rejected"] = Path(
            rejected_annotation_jsonl
        ).expanduser()
    if needs_fix_annotation_jsonl is not None:
        destinations["needs_fix"] = Path(
            needs_fix_annotation_jsonl
        ).expanduser()
    writers = _open_atomic_writers(destinations)
    counts: Counter[str] = Counter()
    seen_decisions: set[str] = set()
    try:
        for row in iter_jsonl(annotation_jsonl):
            annotation = VQAAnnotationV2.model_validate(row)
            decision = decisions.get(annotation.annotation_id)
            disposition = _annotation_disposition(
                annotation.annotation_id,
                decision,
                unreviewed_policy,
            )
            if decision is not None:
                _validate_decision_annotation_identity(
                    decision,
                    annotation,
                )
                if annotation.annotation_id in seen_decisions:
                    raise ValueError(
                        "reviewed annotation appears more than once: "
                        f"{annotation.annotation_id}"
                    )
                seen_decisions.add(annotation.annotation_id)
            counts[disposition] += 1
            target = {
                "keep": "kept",
                "reject": "rejected",
                "needs_fix": "needs_fix",
                "unreviewed_keep": "kept",
            }.get(disposition)
            if target is not None and target in writers:
                _write_row(writers[target][2], row)
        orphaned = sorted(set(decisions) - seen_decisions)
        if orphaned:
            raise ValueError(
                "review decisions reference absent annotations: "
                f"{orphaned[:5]}"
            )
        _commit_atomic_writers(writers)
    except Exception:
        _abort_atomic_writers(writers)
        raise
    return counts, seen_decisions


def _filter_training_views(
    *,
    training_view_jsonl: str | Path,
    output_training_view_jsonl: str | Path,
    decisions: dict[str, VQAReviewDecisionV2],
    unreviewed_policy: VQAUnreviewedPolicyV2,
) -> Counter[str]:
    """Stream matching training views through the same review policy."""

    counts: Counter[str] = Counter()

    def kept_rows() -> Iterator[dict[str, Any]]:
        for row in iter_jsonl(training_view_jsonl):
            view = VQATrainingViewV2.model_validate(row)
            decision = decisions.get(view.annotation_id)
            disposition = _annotation_disposition(
                view.annotation_id,
                decision,
                unreviewed_policy,
            )
            if decision is not None:
                if (
                    decision.source_group_id != view.source_group_id
                    or decision.task_id != view.question.task_id
                    or decision.label_source != view.label_source
                ):
                    raise ValueError(
                        "review decision/training-view identity mismatch: "
                        f"{view.annotation_id}"
                    )
            counts[disposition] += 1
            if disposition in {"keep", "unreviewed_keep"}:
                yield row

    counts["written"] = write_jsonl_atomic(
        output_training_view_jsonl,
        kept_rows(),
    )
    return counts


def _annotation_disposition(
    annotation_id: str,
    decision: VQAReviewDecisionV2 | None,
    unreviewed_policy: VQAUnreviewedPolicyV2,
) -> str:
    """Resolve one annotation to a deterministic output partition."""

    if decision is not None:
        return decision.verdict
    if unreviewed_policy == "error":
        raise ValueError(
            "annotation has no human decision under error policy: "
            f"{annotation_id}"
        )
    return f"unreviewed_{unreviewed_policy}"


def _validate_decision_annotation_identity(
    decision: VQAReviewDecisionV2,
    annotation: VQAAnnotationV2,
) -> None:
    """Require review decisions to bind the exact canonical annotation."""

    if (
        decision.source_group_id != annotation.source_group_id
        or decision.source_fingerprint != annotation.source_fingerprint
        or decision.task_id != annotation.question.task_id
        or decision.label_source != annotation.label_source
    ):
        raise ValueError(
            "review decision/annotation identity mismatch: "
            f"{annotation.annotation_id}"
        )


def _decision_identity(decision: VQAReviewDecisionV2) -> tuple[str, ...]:
    """Return immutable identity fields for revision checks."""

    return (
        decision.annotation_id,
        decision.source_group_id,
        decision.source_fingerprint,
        decision.task_id,
        decision.label_source,
    )


def _open_atomic_writers(
    destinations: dict[str, Path],
) -> dict[str, tuple[Path, Path, TextIO]]:
    """Open temporary writers beside all requested output files."""

    result: dict[str, tuple[Path, Path, TextIO]] = {}
    try:
        for key, destination in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=str(destination.parent),
                text=True,
            )
            result[key] = (
                destination,
                Path(temporary_name),
                os.fdopen(descriptor, "w", encoding="utf-8"),
            )
    except Exception:
        _abort_atomic_writers(result)
        raise
    return result


def _write_row(handle: TextIO, row: dict[str, Any]) -> None:
    """Write one compact JSONL row to an open temporary artifact."""

    handle.write(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )


def _commit_atomic_writers(
    writers: dict[str, tuple[Path, Path, TextIO]],
) -> None:
    """Flush and atomically publish all annotation partitions."""

    for _destination, _temporary, handle in writers.values():
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    for destination, temporary, _handle in writers.values():
        # 临时文件与目标位于同一目录，rename 不会产生半写入文件。
        os.replace(temporary, destination)


def _abort_atomic_writers(
    writers: dict[str, tuple[Path, Path, TextIO]],
) -> None:
    """Close and remove every unpublished temporary artifact."""

    for _destination, temporary, handle in writers.values():
        if not handle.closed:
            handle.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "VQA_REVIEW_DECISION_V2_SCHEMA_VERSION",
    "VQA_REVIEW_FILTER_REPORT_V2_SCHEMA_VERSION",
    "VQAReviewDecisionV2",
    "filter_vqa_artifacts_by_review_v2",
    "latest_vqa_review_decisions_v2",
    "load_latest_vqa_review_decisions_v2",
]
