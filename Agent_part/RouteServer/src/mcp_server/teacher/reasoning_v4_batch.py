"""Resumable six-call annotation runner for shared-input Reasoning V4."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from pydantic import BaseModel

from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_section_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    REASONING_ANNOTATION_V4_SCHEMA_VERSION,
    AnnotationStageRecordV4,
    GroundingOutputsV4,
    NavObservationV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
    TeacherActionConstraintV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
    DecisionV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    SceneGroundingV4,
    TaskGroundingV4,
)
from mcp_server.data.jsonl_io import append_jsonl_row, iter_jsonl
from mcp_server.data.reasoning_v4_composer import (
    REASONING_V4_COMPOSER_VERSION,
    compose_reasoning_target_v4,
)
from mcp_server.teacher.reasoning_v2_teacher import (
    TeacherGeneration,
    VisionReasoningTeacherBackend,
)
from mcp_server.teacher.reasoning_v4_profiles import (
    REASONING_V4_TEACHER_PROFILES,
    ReasoningV4TeacherProfile,
)
from mcp_server.teacher.reasoning_v4_teacher import (
    build_reasoning_v4_section_question,
    parse_reasoning_v4_section_answer,
)


_GROUNDING_SECTION_NAMES = (
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
)


@dataclass(frozen=True)
class ReasoningV4BatchConfig:
    """Configure bounded and resumable V4 annotation.

    Args:
        max_retries: Repair calls allowed after the initial request.
        image_detail: OpenAI-compatible image detail setting.
        concurrency: Maximum concurrently processed source rows.
        retry_failed_on_resume: Retry cached partial/rejected/error rows.
        raw_answer_max_chars: Maximum raw provider text retained per stage.
        progress_interval: Completed-row interval for console progress.
        input_cost_per_million_tokens: Optional prompt-token price for smoke
            cost estimation.
        output_cost_per_million_tokens: Optional completion-token price for
            smoke cost estimation.
    """

    max_retries: int = 2
    image_detail: str = "low"
    concurrency: int = 1
    retry_failed_on_resume: bool = False
    raw_answer_max_chars: int = 50000
    progress_interval: int = 100
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None


class ReasoningV4TeacherBackend(Protocol):
    """Provider boundary that receives only a V4 visible observation."""

    async def generate(
        self,
        observation: NavObservationV4,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Generate one response without access to private supervision."""


class VisionReasoningV4TeacherBackend:
    """Adapt the existing vision provider without exposing V4 sidecars."""

    def __init__(self, *, api_key: str | None = None):
        """Initialize the existing OpenAI-compatible vision backend."""

        self._delegate = VisionReasoningTeacherBackend(api_key=api_key)

    async def generate(
        self,
        observation: NavObservationV4,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Call the provider using only the normalized V4 observation."""

        visible_source = _VisibleSourceAdapter(
            images=observation.images,
            image_roles=observation.image_roles,
        )
        return await self._delegate.generate(
            visible_source,  # type: ignore[arg-type]
            question=question,
            system_prompt=system_prompt,
            image_detail=image_detail,
        )


@dataclass(frozen=True)
class _VisibleSourceAdapter:
    """Expose the two image attributes used by the existing backend."""

    images: list[str]
    image_roles: list[str]


@dataclass(frozen=True)
class _StageOutcome:
    """Internal result of one independently retried teacher stage."""

    section: Any
    record: AnnotationStageRecordV4
    generations: tuple[TeacherGeneration, ...]


async def annotate_reasoning_source_v4(
    source: ReasoningSourceV4,
    *,
    backend: ReasoningV4TeacherBackend,
    config: ReasoningV4BatchConfig | None = None,
) -> ReasoningAnnotationV4:
    """Annotate five independent groundings and one joint Decision.

    Args:
        source: Fingerprinted visible observation and private strong sidecars.
        backend: Multimodal section-generation provider.
        config: Retry, image-detail, and audit limits.

    Returns:
        Accepted, partial, rejected, or provider-error annotation record.
    """

    cfg = config or ReasoningV4BatchConfig()
    grounding_results = await asyncio.gather(
        *(
            _run_stage_v4(
                source,
                profile=REASONING_V4_TEACHER_PROFILES[section_name],
                backend=backend,
                config=cfg,
            )
            for section_name in _GROUNDING_SECTION_NAMES
        )
    )
    outcomes = dict(zip(_GROUNDING_SECTION_NAMES, grounding_results))
    groundings = GroundingOutputsV4(
        task_grounding=_typed_section(
            outcomes["TaskGrounding"].section,
            TaskGroundingV4,
        ),
        scene_grounding=_typed_section(
            outcomes["SceneGrounding"].section,
            SceneGroundingV4,
        ),
        local_traversability=_typed_section(
            outcomes["LocalTraversability"].section,
            LocalTraversabilityV4,
        ),
        navigation_context_grounding=_typed_section(
            outcomes["NavigationContextGrounding"].section,
            NavigationContextGroundingV4,
        ),
        interaction_constraint_grounding=_typed_section(
            outcomes["InteractionConstraintGrounding"].section,
            InteractionConstraintGroundingV4,
        ),
    )
    records = {
        name: outcome.record
        for name, outcome in outcomes.items()
    }
    generations = [
        generation
        for outcome in outcomes.values()
        for generation in outcome.generations
    ]
    errors = [
        f"{name}:{error}"
        for name, outcome in outcomes.items()
        for error in outcome.record.errors
    ]
    warnings = [
        *source.warnings,
        *(
            f"{name}:{warning}"
            for name, outcome in outcomes.items()
            for warning in outcome.record.warnings
        ),
    ]

    if not groundings.complete:
        records["Decision"] = _skipped_record_v4(
            REASONING_V4_TEACHER_PROFILES["Decision"],
            reason="incomplete_groundings",
        )
        return _annotation_v4(
            source,
            status=_overall_failure_status(records.values()),
            groundings=groundings,
            decision_proposal=None,
            decision=None,
            records=records,
            errors=errors or ["incomplete_groundings"],
            warnings=warnings,
            generations=generations,
        )

    decision_outcome = await _run_stage_v4(
        source,
        profile=REASONING_V4_TEACHER_PROFILES["Decision"],
        backend=backend,
        config=cfg,
        groundings=groundings,
        teacher_action_constraint=_teacher_action_constraint_v4(source),
    )
    records["Decision"] = decision_outcome.record
    generations.extend(decision_outcome.generations)
    errors.extend(
        f"Decision:{error}"
        for error in decision_outcome.record.errors
    )
    warnings.extend(
        f"Decision:{warning}"
        for warning in decision_outcome.record.warnings
    )
    decision_proposal = _typed_section(
        decision_outcome.section,
        DecisionProposalV4,
    )
    if decision_proposal is None:
        return _annotation_v4(
            source,
            status=_overall_failure_status(records.values()),
            groundings=groundings,
            decision_proposal=None,
            decision=None,
            records=records,
            errors=errors or ["decision_missing"],
            warnings=warnings,
            generations=generations,
        )

    try:
        composition = compose_reasoning_target_v4(
            source,
            groundings,
            decision_proposal,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"composition_failed:{exc}")
        return _annotation_v4(
            source,
            status="partial",
            groundings=groundings,
            decision_proposal=decision_proposal,
            decision=None,
            records=records,
            errors=errors,
            warnings=warnings,
            generations=generations,
        )

    return _annotation_v4(
        source,
        status="accepted",
        groundings=groundings,
        decision_proposal=decision_proposal,
        decision=composition.target.reasoning.decision,
        target=composition.target,
        records=records,
        errors=[],
        warnings=[*warnings, *composition.warnings],
        generations=generations,
    )


async def _run_stage_v4(
    source: ReasoningSourceV4,
    *,
    profile: ReasoningV4TeacherProfile,
    backend: ReasoningV4TeacherBackend,
    config: ReasoningV4BatchConfig,
    groundings: GroundingOutputsV4 | None = None,
    teacher_action_constraint: TeacherActionConstraintV4 | None = None,
) -> _StageOutcome:
    """Generate, parse, validate, and repair one V4 section."""

    max_attempts = min(max(int(config.max_retries), 0) + 1, 16)
    errors: list[str] = []
    warnings: list[str] = []
    answer: str | None = None
    generations: list[TeacherGeneration] = []
    provider_failures = 0
    section: BaseModel | None = None
    for _ in range(max_attempts):
        question = build_reasoning_v4_section_question(
            source.observation,
            profile,
            groundings=groundings,
            teacher_action_constraint=teacher_action_constraint,
            previous_answer=answer,
            previous_errors=errors,
        )
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
                f"provider_call_failed:{type(exc).__name__}:{str(exc)[:500]}"
            ]
            continue
        generations.append(generation)
        answer = str(generation.answer or "").strip()
        parsed = parse_reasoning_v4_section_answer(answer, profile)
        errors = list(parsed.errors)
        warnings = list(parsed.warnings)
        if not parsed.valid or parsed.section is None:
            continue
        checked = validate_reasoning_section_v4(
            profile.section_name,
            parsed.section,
            observation=source.observation,
        )
        errors = list(checked.errors)
        warnings = [*warnings, *checked.warnings]
        if not errors:
            section = parsed.section
            break
    status = (
        "accepted"
        if section is not None and not errors
        else "error"
        if provider_failures == max_attempts
        else "rejected"
    )
    record = AnnotationStageRecordV4(
        status=status,
        attempts=max(provider_failures + len(generations), 0),
        raw_answer=(
            answer[: max(0, int(config.raw_answer_max_chars))]
            if answer is not None
            else None
        ),
        errors=[] if status == "accepted" else errors or ["stage_failed"],
        warnings=list(dict.fromkeys(warnings)),
        telemetry={
            "successful_provider_calls": len(generations),
            "calls": [item.telemetry or {} for item in generations],
        },
        prompt_profile_id=profile.profile_id,
        prompt_policy_digest=profile.policy_digest,
    )
    return _StageOutcome(
        section=section,
        record=record,
        generations=tuple(generations),
    )


def _annotation_v4(
    source: ReasoningSourceV4,
    *,
    status: str,
    groundings: GroundingOutputsV4,
    decision_proposal: DecisionProposalV4 | None,
    decision: DecisionV4 | None,
    records: dict[str, AnnotationStageRecordV4],
    errors: list[str],
    warnings: list[str],
    generations: list[TeacherGeneration],
    target: Any = None,
) -> ReasoningAnnotationV4:
    """Construct one schema-consistent annotation audit row."""

    latest = generations[-1] if generations else None
    return ReasoningAnnotationV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status=status,
        grounding_outputs=groundings,
        decision_proposal=decision_proposal,
        decision=decision,
        target=target,
        stage_records=records,
        errors=list(dict.fromkeys(errors))[:64],
        warnings=list(dict.fromkeys(warnings))[:128],
        teacher_provider=latest.provider if latest else None,
        teacher_model=latest.model if latest else None,
        metadata={
            "composer_version": REASONING_V4_COMPOSER_VERSION,
            "annotation_flow": "five_independent_groundings_then_decision",
            "successful_provider_calls": len(generations),
        },
    )


def _teacher_action_constraint_v4(
    source: ReasoningSourceV4,
) -> TeacherActionConstraintV4 | None:
    """Expose demonstrated direction only at the private Decision boundary."""

    direction = source.direction_supervision
    if direction is None:
        return None
    return TeacherActionConstraintV4(
        demonstrated_direction_label=direction.label_name,
    )


def _typed_section(value: Any, expected: type[BaseModel]) -> Any:
    """Return a section only when it has the expected concrete type."""

    return value if isinstance(value, expected) else None


def _skipped_record_v4(
    profile: ReasoningV4TeacherProfile,
    *,
    reason: str,
) -> AnnotationStageRecordV4:
    """Create a zero-call skipped stage record."""

    return AnnotationStageRecordV4(
        status="skipped",
        attempts=0,
        errors=[reason],
        prompt_profile_id=profile.profile_id,
        prompt_policy_digest=profile.policy_digest,
    )


def _overall_failure_status(
    records: Iterable[AnnotationStageRecordV4],
) -> str:
    """Map stage availability to a valid non-accepted annotation status."""

    values = list(records)
    if any(record.status == "accepted" for record in values):
        return "partial"
    if values and all(record.status in {"error", "skipped"} for record in values):
        return "error"
    return "rejected"


async def label_reasoning_source_manifest_v4(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    rejected_jsonl: str | Path | None = None,
    report_json: str | Path | None = None,
    config: ReasoningV4BatchConfig | None = None,
    limit: int | None = None,
    api_keys: Iterable[str] = (),
    backend_factory: Callable[
        [int, str | None], ReasoningV4TeacherBackend
    ]
    | None = None,
) -> dict[str, Any]:
    """Label a frozen V4 manifest with append-only resume semantics.

    Args:
        source_jsonl: Frozen V4 source manifest.
        annotation_jsonl: Append-only annotation destination.
        rejected_jsonl: Optional mirror for non-accepted rows.
        report_json: Optional atomically replaced progress report.
        config: Retry, concurrency, and audit policy.
        limit: Optional number of source rows to process.
        api_keys: Optional worker-rotated provider keys.
        backend_factory: Optional test or custom backend factory.

    Returns:
        Final progress and status report.
    """

    cfg = config or ReasoningV4BatchConfig()
    for name, value in (
        (
            "input_cost_per_million_tokens",
            cfg.input_cost_per_million_tokens,
        ),
        (
            "output_cost_per_million_tokens",
            cfg.output_cost_per_million_tokens,
        ),
    ):
        if value is not None and float(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    sources = [
        ReasoningSourceV4.model_validate(row)
        for row in iter_jsonl(source_jsonl)
    ]
    if limit is not None:
        sources = sources[: int(limit)]
    cached = _latest_annotation_cache_v4(annotation_jsonl)
    pending: list[ReasoningSourceV4] = []
    resume_counts: Counter[str] = Counter()
    expected_digests = {
        name: profile.policy_digest
        for name, profile in REASONING_V4_TEACHER_PROFILES.items()
    }
    for source in sources:
        source_id = source.observation.source_group_id
        existing = cached.get(source_id)
        stale = (
            existing is None
            or existing.source_fingerprint != source.source_fingerprint
            or any(
                existing.stage_records.get(name) is None
                or existing.stage_records[name].prompt_policy_digest != digest
                for name, digest in expected_digests.items()
            )
            or existing.metadata.get("composer_version")
            != REASONING_V4_COMPOSER_VERSION
        )
        if stale:
            pending.append(source)
            resume_counts["cache_miss_or_stale"] += 1
        elif existing.status == "accepted":
            resume_counts["cached_accepted"] += 1
        elif cfg.retry_failed_on_resume:
            pending.append(source)
            resume_counts[f"retried_cached_{existing.status}"] += 1
        else:
            resume_counts[f"cached_{existing.status}"] += 1

    keys = tuple(str(key).strip() for key in api_keys if str(key).strip())
    worker_count = min(max(1, int(cfg.concurrency)), max(1, len(pending)))
    if keys:
        worker_count = min(worker_count, len(keys))
    report: dict[str, Any] = {
        "schema_version": f"{REASONING_ANNOTATION_V4_SCHEMA_VERSION}_report",
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "input_rows": len(sources),
        "pending_rows": len(pending),
        "processed_rows": 0,
        "accepted_rows": 0,
        "partial_rows": 0,
        "rejected_rows": 0,
        "error_rows": 0,
        "concurrency": worker_count,
        "resume_counts": dict(resume_counts),
        "prompt_policy_digests": expected_digests,
        "composer_version": REASONING_V4_COMPOSER_VERSION,
        "status": "running",
        "error_counts": {},
        "warning_counts": {},
        "dataset_counts": {},
        "stage_metrics": {},
        "telemetry_totals": {
            "provider_calls": 0,
            "roundtrip_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
        },
        "cost_estimate": {
            "currency": "USD",
            "input_cost_per_million_tokens": (
                cfg.input_cost_per_million_tokens
            ),
            "output_cost_per_million_tokens": (
                cfg.output_cost_per_million_tokens
            ),
            "estimated_total": None,
            "status": (
                "configured"
                if cfg.input_cost_per_million_tokens is not None
                and cfg.output_cost_per_million_tokens is not None
                else "unavailable_no_pricing_contract"
            ),
        },
    }
    if not pending:
        report["status"] = "done"
        _write_report_v4(report_json, report)
        return report

    output_path = Path(annotation_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = (
        Path(rejected_jsonl).expanduser()
        if rejected_jsonl is not None
        else None
    )
    if rejected_path is not None:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[ReasoningSourceV4 | None] = asyncio.Queue()
    results: asyncio.Queue[
        tuple[ReasoningSourceV4, ReasoningAnnotationV4]
    ] = asyncio.Queue()
    for source in pending:
        queue.put_nowait(source)
    for _ in range(worker_count):
        queue.put_nowait(None)

    def make_backend(worker_id: int) -> ReasoningV4TeacherBackend:
        """Build one worker-local provider adapter."""

        key = keys[(worker_id - 1) % len(keys)] if keys else None
        if backend_factory is not None:
            return backend_factory(worker_id, key)
        return VisionReasoningV4TeacherBackend(api_key=key)

    async def worker(worker_id: int) -> None:
        """Consume frozen sources until the queue sentinel is reached."""

        backend = make_backend(worker_id)
        while True:
            source = await queue.get()
            try:
                if source is None:
                    return
                try:
                    annotation = await annotate_reasoning_source_v4(
                        source,
                        backend=backend,
                        config=cfg,
                    )
                except Exception as exc:
                    annotation = ReasoningAnnotationV4(
                        source_group_id=source.observation.source_group_id,
                        source_fingerprint=source.source_fingerprint,
                        status="error",
                        grounding_outputs=GroundingOutputsV4(),
                        errors=[
                            f"worker_exception:{type(exc).__name__}:"
                            f"{str(exc)[:500]}"
                        ],
                        metadata={
                            "worker_id": worker_id,
                            "composer_version": REASONING_V4_COMPOSER_VERSION,
                        },
                    )
                await results.put((source, annotation))
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker(index))
        for index in range(1, worker_count + 1)
    ]
    started_at = time.monotonic()
    reject_handle = (
        rejected_path.open("a", encoding="utf-8", buffering=1)
        if rejected_path is not None
        else None
    )
    try:
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for _ in range(len(pending)):
                source, annotation = await results.get()
                row = annotation.model_dump(mode="json", exclude_none=True)
                append_jsonl_row(handle, row)
                if annotation.status != "accepted" and reject_handle is not None:
                    append_jsonl_row(reject_handle, row)
                report["processed_rows"] += 1
                report[f"{annotation.status}_rows"] += 1
                dataset = source.observation.dataset_source
                report["dataset_counts"][dataset] = (
                    int(report["dataset_counts"].get(dataset, 0)) + 1
                )
                for error in annotation.errors:
                    key = error.split(":", 1)[0]
                    report["error_counts"][key] = (
                        int(report["error_counts"].get(key, 0)) + 1
                    )
                _accumulate_annotation_metrics_v4(
                    report,
                    annotation,
                )
                _refresh_annotation_aggregates_v4(report, cfg)
                elapsed = max(time.monotonic() - started_at, 1e-6)
                report["rate_rows_per_hour"] = (
                    report["processed_rows"] / elapsed * 3600.0
                )
                report["pending_rows"] = (
                    len(pending) - report["processed_rows"]
                )
                _write_report_v4(report_json, report)
                if (
                    cfg.progress_interval > 0
                    and report["processed_rows"] % cfg.progress_interval == 0
                ):
                    print(
                        json.dumps(
                            {
                                "reasoning_v4_progress": (
                                    f"{report['processed_rows']}/"
                                    f"{len(pending)}"
                                ),
                                "accepted": report["accepted_rows"],
                                "partial": report["partial_rows"],
                                "rejected": report["rejected_rows"],
                                "error": report["error_rows"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        if reject_handle is not None:
            reject_handle.close()
    await queue.join()
    await asyncio.gather(*workers)
    report["status"] = "done"
    _write_report_v4(report_json, report)
    return report


def _latest_annotation_cache_v4(
    path: str | Path,
) -> dict[str, ReasoningAnnotationV4]:
    """Load the latest append-only V4 annotation for each source id."""

    value = Path(path).expanduser()
    if not value.exists():
        return {}
    cache: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(value):
        annotation = ReasoningAnnotationV4.model_validate(row)
        cache[annotation.source_group_id] = annotation
    return cache


def _accumulate_annotation_metrics_v4(
    report: dict[str, Any],
    annotation: ReasoningAnnotationV4,
) -> None:
    """Accumulate stage acceptance, warning, timing, and token metrics."""

    for warning in annotation.warnings:
        code = warning.split(":", 1)[0]
        report["warning_counts"][code] = (
            int(report["warning_counts"].get(code, 0)) + 1
        )
    totals = report["telemetry_totals"]
    for section_name, record in annotation.stage_records.items():
        metric = report["stage_metrics"].setdefault(
            section_name,
            {
                "accepted": 0,
                "rejected": 0,
                "error": 0,
                "skipped": 0,
                "attempts": 0,
                "provider_calls": 0,
                "roundtrip_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
                "error_counts": {},
                "warning_counts": {},
            },
        )
        metric[record.status] = int(metric[record.status]) + 1
        metric["attempts"] = int(metric["attempts"]) + int(
            record.attempts
        )
        for error in record.errors:
            code = error.split(":", 1)[0]
            metric["error_counts"][code] = (
                int(metric["error_counts"].get(code, 0)) + 1
            )
        for warning in record.warnings:
            code = warning.split(":", 1)[0]
            metric["warning_counts"][code] = (
                int(metric["warning_counts"].get(code, 0)) + 1
            )
        calls = record.telemetry.get("calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            metric["provider_calls"] = int(
                metric["provider_calls"]
            ) + 1
            totals["provider_calls"] = int(totals["provider_calls"]) + 1
            roundtrip_ms = _finite_number(call.get("roundtrip_ms"))
            metric["roundtrip_ms"] = (
                float(metric["roundtrip_ms"]) + roundtrip_ms
            )
            totals["roundtrip_ms"] = (
                float(totals["roundtrip_ms"]) + roundtrip_ms
            )
            response = call.get("response")
            usage = (
                response.get("usage")
                if isinstance(response, dict)
                and isinstance(response.get("usage"), dict)
                else {}
            )
            prompt_tokens = _nonnegative_int(
                usage.get("prompt_tokens")
            )
            completion_tokens = _nonnegative_int(
                usage.get("completion_tokens")
            )
            total_tokens = _nonnegative_int(
                usage.get("total_tokens")
            )
            prompt_details = usage.get("prompt_tokens_details")
            cached_tokens = _nonnegative_int(
                prompt_details.get("cached_tokens")
                if isinstance(prompt_details, dict)
                else call.get("prompt_cached_tokens")
            )
            for key, value in (
                ("prompt_tokens", prompt_tokens),
                ("completion_tokens", completion_tokens),
                ("total_tokens", total_tokens),
                ("cached_prompt_tokens", cached_tokens),
            ):
                metric[key] = int(metric[key]) + value
                totals[key] = int(totals[key]) + value


def _refresh_annotation_aggregates_v4(
    report: dict[str, Any],
    config: ReasoningV4BatchConfig,
) -> None:
    """Refresh derived acceptance, latency, and optional cost estimates."""

    for metric in report["stage_metrics"].values():
        records = sum(
            int(metric[name])
            for name in ("accepted", "rejected", "error", "skipped")
        )
        provider_calls = int(metric["provider_calls"])
        metric["acceptance_fraction"] = (
            float(metric["accepted"]) / records if records else 0.0
        )
        metric["average_attempts"] = (
            float(metric["attempts"]) / records if records else 0.0
        )
        metric["average_roundtrip_ms"] = (
            float(metric["roundtrip_ms"]) / provider_calls
            if provider_calls
            else None
        )
    totals = report["telemetry_totals"]
    provider_calls = int(totals["provider_calls"])
    totals["average_roundtrip_ms"] = (
        float(totals["roundtrip_ms"]) / provider_calls
        if provider_calls
        else None
    )
    input_rate = config.input_cost_per_million_tokens
    output_rate = config.output_cost_per_million_tokens
    if input_rate is not None and output_rate is not None:
        report["cost_estimate"]["estimated_total"] = (
            int(totals["prompt_tokens"]) * float(input_rate)
            + int(totals["completion_tokens"]) * float(output_rate)
        ) / 1_000_000.0


def _finite_number(value: Any) -> float:
    """Return one finite non-negative numeric telemetry value."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if result < 0 or result == float("inf") or result != result:
        return 0.0
    return result


def _nonnegative_int(value: Any) -> int:
    """Return one non-negative integer telemetry value."""

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _write_report_v4(
    path: str | Path | None,
    report: dict[str, Any],
) -> None:
    """Atomically replace the optional progress report."""

    if path is None:
        return
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
