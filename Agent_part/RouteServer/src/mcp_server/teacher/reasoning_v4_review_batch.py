"""Resumable two-stage LVLM filtering for Reasoning V4.6 audits."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel

from mcp_server.contracts.reasoning_dataset_v4 import (
    ActionBlindReviewV4,
    ActionReviewDiagnosisPayloadV4,
    ActionReviewDiagnosisRecordV4,
    AnnotationStageRecordV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.data.jsonl_io import append_jsonl_row, iter_jsonl
from mcp_server.data.reasoning_v4_reconciliation import (
    action_reconciliation_review_required_v4,
)
from mcp_server.data.reasoning_v4_review import (
    bind_action_review_diagnosis_v4,
    load_latest_action_review_diagnoses_v4,
)
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration
from mcp_server.teacher.reasoning_v4_batch import (
    ReasoningV4TeacherBackend,
    VisionReasoningV4TeacherBackend,
)
from mcp_server.teacher.reasoning_v4_review import (
    build_reasoning_v4_blind_review_question,
    build_reasoning_v4_comparative_review_question,
    parse_reasoning_v4_review_answer,
)
from mcp_server.teacher.reasoning_v4_review_profiles import (
    REASONING_V4_REVIEW_PROFILES,
    ReasoningV4ReviewProfile,
)


@dataclass(frozen=True)
class ReasoningV4ReviewBatchConfig:
    """Configure bounded and resumable LVLM review filtering.

    Args:
        max_retries: Repair calls after each initial stage request.
        image_detail: OpenAI-compatible detail for all reviewer-visible images.
        concurrency: Maximum concurrently reviewed audit rows.
        retry_failed_on_resume: Reprocess cached error records.
        raw_answer_max_chars: Maximum raw provider text retained per stage.
        progress_interval: Completed-row console interval.
    """

    max_retries: int = 2
    image_detail: str = "high"
    concurrency: int = 1
    retry_failed_on_resume: bool = False
    raw_answer_max_chars: int = 50000
    progress_interval: int = 50


@dataclass(frozen=True)
class _ReviewStageOutcome:
    """Store one independently retried review-stage result."""

    value: BaseModel | None
    record: AnnotationStageRecordV4
    generations: tuple[TeacherGeneration, ...]


async def diagnose_reasoning_action_review_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    *,
    backend: ReasoningV4TeacherBackend,
    config: ReasoningV4ReviewBatchConfig | None = None,
) -> ActionReviewDiagnosisRecordV4:
    """Run blind visual assessment followed by comparative diagnosis.

    Args:
        source: Exact source revision under review.
        annotation: Exact annotation containing a triggered action audit.
        backend: Multimodal review provider.
        config: Retry, image-detail, and raw-answer limits.

    Returns:
        Immutable completed or error diagnosis record.

    Raises:
        ValueError: If parent identities or review preconditions disagree.
    """

    cfg = config or ReasoningV4ReviewBatchConfig()
    if cfg.image_detail not in {"auto", "low", "high"}:
        raise ValueError("review image_detail must be auto, low, or high")
    if (
        source.observation.source_group_id != annotation.source_group_id
        or source.source_fingerprint != annotation.source_fingerprint
    ):
        raise ValueError("review source and annotation identity mismatch")
    audit = annotation.action_reconciliation
    if (
        audit is None
        or not action_reconciliation_review_required_v4(audit)
    ):
        raise ValueError(
            "LVLM filtering requires a deterministically triggered audit"
        )

    blind_profile = REASONING_V4_REVIEW_PROFILES["blind_visual"]
    blind = await _run_review_stage_v4(
        source,
        profile=blind_profile,
        backend=backend,
        config=cfg,
        question_builder=lambda previous_answer, previous_errors: (
            build_reasoning_v4_blind_review_question(
                source.observation,
                blind_profile,
                previous_answer=previous_answer,
                previous_errors=previous_errors,
            )
        ),
    )
    blind_value = (
        blind.value
        if isinstance(blind.value, ActionBlindReviewV4)
        else None
    )
    if blind_value is None:
        return ActionReviewDiagnosisRecordV4(
            source_group_id=source.observation.source_group_id,
            source_fingerprint=source.source_fingerprint,
            audit_id=audit.audit_id,
            record_status="error",
            blind_assessment=None,
            diagnosis=None,
            stage_records={"blind_visual": blind.record},
            errors=[
                f"blind_visual:{error}"
                for error in blind.record.errors
            ]
            or ["blind_visual_failed"],
        )

    comparative_profile = REASONING_V4_REVIEW_PROFILES[
        "comparative_diagnosis"
    ]
    comparative = await _run_review_stage_v4(
        source,
        profile=comparative_profile,
        backend=backend,
        config=cfg,
        question_builder=lambda previous_answer, previous_errors: (
            build_reasoning_v4_comparative_review_question(
                source,
                annotation,
                blind_value,
                comparative_profile,
                previous_answer=previous_answer,
                previous_errors=previous_errors,
            )
        ),
    )
    diagnosis_payload = (
        comparative.value
        if isinstance(
            comparative.value,
            ActionReviewDiagnosisPayloadV4,
        )
        else None
    )
    if diagnosis_payload is None:
        return ActionReviewDiagnosisRecordV4(
            source_group_id=source.observation.source_group_id,
            source_fingerprint=source.source_fingerprint,
            audit_id=audit.audit_id,
            record_status="error",
            blind_assessment=blind_value,
            diagnosis=None,
            stage_records={
                "blind_visual": blind.record,
                "comparative_diagnosis": comparative.record,
            },
            errors=[
                f"comparative_diagnosis:{error}"
                for error in comparative.record.errors
            ]
            or ["comparative_diagnosis_failed"],
        )
    diagnosis = bind_action_review_diagnosis_v4(
        source,
        annotation,
        diagnosis_payload,
    )
    return ActionReviewDiagnosisRecordV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        audit_id=audit.audit_id,
        record_status="complete",
        blind_assessment=blind_value,
        diagnosis=diagnosis,
        stage_records={
            "blind_visual": blind.record,
            "comparative_diagnosis": comparative.record,
        },
    )


async def label_reasoning_review_queue_v4(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    diagnosis_jsonl: str | Path,
    *,
    api_keys: Iterable[str] = (),
    config: ReasoningV4ReviewBatchConfig | None = None,
    limit: int | None = None,
    report_json: str | Path | None = None,
    backend_factory: Callable[
        [int, str | None],
        ReasoningV4TeacherBackend,
    ]
    | None = None,
) -> dict[str, Any]:
    """Diagnose all triggered audits with bounded concurrency and resume.

    Args:
        source_jsonl: Frozen unique source manifest.
        annotation_jsonl: Append-only annotation manifest.
        diagnosis_jsonl: Append-only diagnosis-record destination.
        api_keys: Optional worker-local provider keys.
        config: Review runner configuration.
        limit: Optional maximum pending audit rows.
        report_json: Optional atomic progress report.
        backend_factory: Optional custom or test backend factory.

    Returns:
        Machine-readable filtering report.
    """

    cfg = config or ReasoningV4ReviewBatchConfig()
    sources = _source_index_v4(source_jsonl)
    annotations = _latest_annotation_index_v4(annotation_jsonl)
    cached = load_latest_action_review_diagnoses_v4(
        diagnosis_jsonl
        if Path(diagnosis_jsonl).expanduser().exists()
        else None
    )
    pending: list[tuple[ReasoningSourceV4, ReasoningAnnotationV4]] = []
    counts: Counter[str] = Counter()
    for source_id, source in sources.items():
        annotation = annotations.get(source_id)
        if annotation is None:
            counts["missing_annotation"] += 1
            continue
        audit = annotation.action_reconciliation
        if (
            audit is None
            or not action_reconciliation_review_required_v4(audit)
        ):
            counts["not_triggered"] += 1
            continue
        counts["triggered"] += 1
        existing = cached.get(audit.audit_id)
        retryable = (
            existing is not None
            and existing.record_status == "error"
            and cfg.retry_failed_on_resume
        )
        if existing is None or retryable:
            pending.append((source, annotation))
        else:
            counts[
                f"cached:{existing.record_status}"
            ] += 1
    if limit is not None:
        pending = pending[: max(0, int(limit))]
    keys = tuple(
        str(key).strip()
        for key in api_keys
        if str(key).strip()
    )
    worker_count = min(
        max(1, int(cfg.concurrency)),
        max(1, len(pending)),
    )
    if keys:
        worker_count = min(worker_count, len(keys))
    report: dict[str, Any] = {
        "schema_version": (
            "urbannav_action_review_diagnosis_run_report_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "diagnosis_jsonl": str(diagnosis_jsonl),
        "status": "running" if pending else "done",
        "pending_rows": len(pending),
        "concurrency": worker_count,
        "counts": dict(counts),
        "verdict_counts": {},
        "errors": [],
    }
    _write_report(report_json, report)
    if not pending:
        report["passed"] = not report["errors"]
        _write_report(report_json, report)
        return report

    queue: asyncio.Queue[
        tuple[ReasoningSourceV4, ReasoningAnnotationV4] | None
    ] = asyncio.Queue()
    results: asyncio.Queue[ActionReviewDiagnosisRecordV4] = (
        asyncio.Queue()
    )
    for item in pending:
        queue.put_nowait(item)
    for _ in range(worker_count):
        queue.put_nowait(None)

    def make_backend(worker_id: int) -> ReasoningV4TeacherBackend:
        """Build one worker-local review backend."""

        key = (
            keys[(worker_id - 1) % len(keys)]
            if keys
            else None
        )
        if backend_factory is not None:
            return backend_factory(worker_id, key)
        return VisionReasoningV4TeacherBackend(api_key=key)

    async def worker(worker_id: int) -> None:
        """Consume audit rows until the queue sentinel."""

        backend = make_backend(worker_id)
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                source, annotation = item
                try:
                    record = await diagnose_reasoning_action_review_v4(
                        source,
                        annotation,
                        backend=backend,
                        config=cfg,
                    )
                except Exception as exc:
                    audit = annotation.action_reconciliation
                    assert audit is not None
                    record = ActionReviewDiagnosisRecordV4(
                        source_group_id=(
                            source.observation.source_group_id
                        ),
                        source_fingerprint=source.source_fingerprint,
                        audit_id=audit.audit_id,
                        record_status="error",
                        errors=[
                            "worker_exception:"
                            f"{type(exc).__name__}:{str(exc)[:500]}"
                        ],
                    )
                await results.put(record)
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker(index))
        for index in range(1, worker_count + 1)
    ]
    destination = Path(diagnosis_jsonl).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open(
            "a",
            encoding="utf-8",
            buffering=1,
        ) as handle:
            for completed in range(1, len(pending) + 1):
                record = await results.get()
                append_jsonl_row(
                    handle,
                    record.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                )
                counts[
                    f"record_status:{record.record_status}"
                ] += 1
                if record.diagnosis is not None:
                    counts[
                        f"verdict:{record.diagnosis.verdict}"
                    ] += 1
                else:
                    report["errors"].append(
                        {
                            "audit_id": record.audit_id,
                            "errors": list(record.errors),
                        }
                    )
                report["counts"] = dict(counts)
                report["verdict_counts"] = {
                    key.removeprefix("verdict:"): value
                    for key, value in counts.items()
                    if key.startswith("verdict:")
                }
                if (
                    cfg.progress_interval > 0
                    and completed % cfg.progress_interval == 0
                ):
                    _write_report(report_json, report)
    finally:
        await queue.join()
        await asyncio.gather(*workers)
    report["status"] = "done"
    report["passed"] = not report["errors"]
    _write_report(report_json, report)
    return report


async def _run_review_stage_v4(
    source: ReasoningSourceV4,
    *,
    profile: ReasoningV4ReviewProfile,
    backend: ReasoningV4TeacherBackend,
    config: ReasoningV4ReviewBatchConfig,
    question_builder: Callable[[str | None, list[str]], str],
) -> _ReviewStageOutcome:
    """Generate, parse, and repair one review stage."""

    max_attempts = min(
        max(0, int(config.max_retries)) + 1,
        16,
    )
    answer: str | None = None
    errors: list[str] = []
    generations: list[TeacherGeneration] = []
    provider_failures = 0
    value: BaseModel | None = None
    for _ in range(max_attempts):
        question = question_builder(answer, errors)
        try:
            generation = await backend.generate(
                source.observation,
                question=question,
                system_prompt=profile.system_prompt,
                image_detail=config.image_detail,
            )
        except Exception as exc:
            provider_failures += 1
            errors = [
                "provider_call_failed:"
                f"{type(exc).__name__}:{str(exc)[:500]}"
            ]
            continue
        generations.append(generation)
        answer = str(generation.answer or "").strip()
        parsed = parse_reasoning_v4_review_answer(
            answer,
            profile,
        )
        errors = list(parsed.errors)
        value = parsed.value
        if parsed.valid:
            break
        value = None
    status = (
        "accepted"
        if value is not None and not errors
        else "error"
        if provider_failures == max_attempts
        else "rejected"
    )
    return _ReviewStageOutcome(
        value=value,
        record=AnnotationStageRecordV4(
            status=status,
            attempts=provider_failures + len(generations),
            raw_answer=(
                answer[
                    : max(0, int(config.raw_answer_max_chars))
                ]
                if answer is not None
                else None
            ),
            errors=[] if status == "accepted" else errors,
            telemetry={
                "image_policy_id": profile.image_policy_id,
                "image_detail": config.image_detail,
                "selected_image_roles": list(
                    source.observation.image_roles
                ),
                "selected_image_count": len(
                    source.observation.images
                ),
                "calls": [
                    generation.telemetry or {}
                    for generation in generations
                ],
            },
            producer_kind="lvlm",
            producer_profile_id=profile.profile_id,
            derivation_version=profile.derivation_version,
            prompt_policy_digest=profile.policy_digest,
        ),
        generations=tuple(generations),
    )


def _source_index_v4(
    path: str | Path,
) -> dict[str, ReasoningSourceV4]:
    """Load one unique source row per group."""

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


def _latest_annotation_index_v4(
    path: str | Path,
) -> dict[str, ReasoningAnnotationV4]:
    """Load the latest append-only annotation per source group."""

    result: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV4.model_validate(row)
        result[annotation.source_group_id] = annotation
    return result


def _write_report(
    path: str | Path | None,
    report: dict[str, Any],
) -> None:
    """Write an optional progress report atomically."""

    if path is None:
        return
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


__all__ = [
    "ReasoningV4ReviewBatchConfig",
    "diagnose_reasoning_action_review_v4",
    "label_reasoning_review_queue_v4",
]
