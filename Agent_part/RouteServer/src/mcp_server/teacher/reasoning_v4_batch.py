"""Resumable six-section annotation runner for Reasoning V4.6."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from pydantic import BaseModel

from mcp_server.agent.reasoning_v4_validation import (
    sanitize_reasoning_section_evidence_v4,
    validate_reasoning_section_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    REASONING_ANNOTATION_V4_SCHEMA_VERSION,
    ActionReconciliationAuditV4,
    AnnotationStageRecordV4,
    CandidateStructureRecordV4,
    GroundingOutputsV4,
    NavObservationV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
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
from mcp_server.data.reasoning_v4_reconciliation import (
    build_action_reconciliation_audit_v4,
)
from mcp_server.data.reasoning_v4_task_canonicalization import (
    REASONING_V4_TASK_CANONICALIZATION_VERSION,
    canonicalize_task_instruction_v4,
    task_instruction_fingerprint_v4,
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


_LVLM_GROUNDING_SECTION_NAMES = (
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
)
_SOURCE_INPUT_DIAGNOSTIC_CODES = {
    "history_fpv_is_repeated_padding",
}


@dataclass(frozen=True)
class ReasoningV4BatchConfig:
    """Configure bounded, resumable V4.6 annotation.

    Args:
        max_retries: Repair calls after the initial request.
        image_detail: OpenAI-compatible image detail.
        navigation_context_image_detail: Higher-detail setting used only for
            NavigationContextGrounding so thin BEV route overlays survive
            provider-side image tokenization.
        concurrency: Maximum concurrently processed source rows.
        retry_failed_on_resume: Retry cached incomplete or invalid records.
        raw_answer_max_chars: Maximum raw provider text retained per stage.
        progress_interval: Completed-row console interval.
        input_cost_per_million_tokens: Optional input-token price.
        output_cost_per_million_tokens: Optional output-token price.
    """

    max_retries: int = 2
    image_detail: str = "low"
    navigation_context_image_detail: str = "high"
    concurrency: int = 1
    retry_failed_on_resume: bool = False
    raw_answer_max_chars: int = 50000
    progress_interval: int = 100
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None


class ReasoningV4TeacherBackend(Protocol):
    """Provider boundary receiving only prompt-visible input."""

    async def generate(
        self,
        observation: NavObservationV4,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Generate one section response.

        Args:
            observation: Prompt-visible observation.
            question: Section-specific user question.
            system_prompt: Exact section policy.
            image_detail: Provider image-detail setting.

        Returns:
            Provider generation and telemetry.
        """


class VisionReasoningV4TeacherBackend:
    """Adapt the existing vision backend without private sidecars."""

    def __init__(self, *, api_key: str | None = None):
        """Initialize the OpenAI-compatible vision backend.

        Args:
            api_key: Optional worker-local provider key.
        """

        self._delegate = VisionReasoningTeacherBackend(
            api_key=api_key
        )

    async def generate(
        self,
        observation: NavObservationV4,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Call the provider with visible images only.

        Args:
            observation: Prompt-visible observation.
            question: Section user prompt.
            system_prompt: Exact section system prompt.
            image_detail: Provider image-detail setting.

        Returns:
            Provider generation and telemetry.
        """

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
    """Expose image attributes required by the provider adapter."""

    images: list[str]
    image_roles: list[str]


@dataclass(frozen=True)
class _StageOutcome:
    """Store one independently retried stage result."""

    section: BaseModel | None
    record: AnnotationStageRecordV4
    generations: tuple[TeacherGeneration, ...]
    provenance: dict[str, str] = field(default_factory=dict)


class _TaskCanonicalizationCoordinator:
    """Ensure each unknown instruction fingerprint is annotated once."""

    def __init__(
        self,
        initial_cache: dict[str, TaskGroundingV4] | None = None,
    ) -> None:
        """Initialize the persistent-projection taxonomy cache and locks.

        Args:
            initial_cache: Task labels recovered from compatible existing
                annotations in the append-only output manifest.
        """

        self.cache: dict[str, TaskGroundingV4] = dict(
            initial_cache or {}
        )
        self.locks: dict[str, asyncio.Lock] = {}

    async def resolve(
        self,
        source: ReasoningSourceV4,
        *,
        backend: ReasoningV4TeacherBackend,
        config: ReasoningV4BatchConfig,
    ) -> _StageOutcome:
        """Resolve known registry entries or annotate one unknown fingerprint."""

        instruction = source.observation.instruction
        known = canonicalize_task_instruction_v4(
            instruction,
            annotation_cache=self.cache,
        )
        if known is not None:
            return _deterministic_task_outcome_v4(known.task_grounding)
        if instruction is None:
            return _run_missing_task_outcome_v4()
        fingerprint = task_instruction_fingerprint_v4(instruction)
        lock = self.locks.setdefault(
            fingerprint,
            asyncio.Lock(),
        )
        async with lock:
            cached = canonicalize_task_instruction_v4(
                instruction,
                annotation_cache=self.cache,
            )
            if cached is not None:
                return _deterministic_task_outcome_v4(
                    cached.task_grounding,
                    source="annotation_cache",
                )
            outcome = await _run_stage_v4(
                source,
                profile=REASONING_V4_TEACHER_PROFILES[
                    "TaskGrounding"
                ],
                backend=backend,
                config=config,
            )
            if isinstance(outcome.section, TaskGroundingV4):
                self.cache[fingerprint] = outcome.section
            return _StageOutcome(
                section=outcome.section,
                record=outcome.record,
                generations=outcome.generations,
                provenance={
                    "task_canonicalization_source": "lvlm_teacher",
                },
            )


async def annotate_reasoning_source_v4(
    source: ReasoningSourceV4,
    *,
    backend: ReasoningV4TeacherBackend,
    config: ReasoningV4BatchConfig | None = None,
    task_coordinator: _TaskCanonicalizationCoordinator | None = None,
) -> ReasoningAnnotationV4:
    """Annotate five shared-input groundings and one blind Decision.

    Args:
        source: Fingerprinted visible input and private sidecars.
        backend: Multimodal section teacher.
        config: Retry, image-detail, and audit limits.
        task_coordinator: Optional run-wide task deduplication coordinator.

    Returns:
        Immutable completed annotation with independent section, candidate
        structure, and reconciliation state.
    """

    cfg = config or ReasoningV4BatchConfig()
    for name, value in (
        ("image_detail", cfg.image_detail),
        (
            "navigation_context_image_detail",
            cfg.navigation_context_image_detail,
        ),
    ):
        if value not in {"auto", "low", "high"}:
            raise ValueError(
                f"{name} must be auto, low, or high: {value!r}"
            )
    coordinator = (
        task_coordinator or _TaskCanonicalizationCoordinator()
    )
    task_future = coordinator.resolve(
        source,
        backend=backend,
        config=cfg,
    )
    lvlm_futures = [
        _run_stage_v4(
            source,
            profile=REASONING_V4_TEACHER_PROFILES[name],
            backend=backend,
            config=cfg,
        )
        for name in _LVLM_GROUNDING_SECTION_NAMES
    ]
    results = await asyncio.gather(
        task_future,
        *lvlm_futures,
    )
    outcomes = {
        "TaskGrounding": results[0],
        **dict(
            zip(
                _LVLM_GROUNDING_SECTION_NAMES,
                results[1:],
            )
        ),
    }
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
        *(
            warning
            for warning in source.warnings
            if warning not in _SOURCE_INPUT_DIAGNOSTIC_CODES
        ),
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
            record_status="complete",
            groundings=groundings,
            decision_proposal=None,
            action_reconciliation=None,
            composer_candidate=None,
            candidate_target=None,
            candidate_structure_record=CandidateStructureRecordV4(
                composer_version=REASONING_V4_COMPOSER_VERSION,
                status="not_available",
                errors=["incomplete_groundings"],
            ),
            records=records,
            errors=errors or ["incomplete_groundings"],
            warnings=warnings,
            generations=generations,
            provenance=outcomes["TaskGrounding"].provenance,
        )

    decision_outcome = await _run_stage_v4(
        source,
        profile=REASONING_V4_TEACHER_PROFILES["Decision"],
        backend=backend,
        config=cfg,
        groundings=groundings,
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
            record_status="complete",
            groundings=groundings,
            decision_proposal=None,
            action_reconciliation=None,
            composer_candidate=None,
            candidate_target=None,
            candidate_structure_record=CandidateStructureRecordV4(
                composer_version=REASONING_V4_COMPOSER_VERSION,
                status="not_available",
                errors=["decision_missing"],
            ),
            records=records,
            errors=errors or ["decision_missing"],
            warnings=warnings,
            generations=generations,
            provenance=outcomes["TaskGrounding"].provenance,
        )

    try:
        audit = build_action_reconciliation_audit_v4(
            source,
            groundings,
            decision_proposal,
        )
    except (TypeError, ValueError) as exc:
        audit = None
        errors.append(
            f"action_reconciliation_failed:{exc}"
        )

    try:
        composition = compose_reasoning_target_v4(
            source,
            groundings,
            decision_proposal,
        )
    except (TypeError, ValueError) as exc:
        candidate_error = f"candidate_construction_failed:{exc}"
        errors.append(candidate_error)
        unavailable = (
            source.direction_supervision is None
        )
        return _annotation_v4(
            source,
            record_status="complete",
            groundings=groundings,
            decision_proposal=decision_proposal,
            action_reconciliation=audit,
            composer_candidate=None,
            candidate_target=None,
            candidate_structure_record=CandidateStructureRecordV4(
                composer_version=REASONING_V4_COMPOSER_VERSION,
                status=(
                    "not_available"
                    if unavailable
                    else "invalid"
                ),
                errors=[candidate_error],
            ),
            records=records,
            errors=errors,
            warnings=warnings,
            generations=generations,
            provenance=outcomes["TaskGrounding"].provenance,
        )

    return _annotation_v4(
        source,
        record_status="complete",
        groundings=groundings,
        decision_proposal=decision_proposal,
        action_reconciliation=audit,
        composer_candidate=(
            composition.candidate_target.reasoning.decision
        ),
        candidate_target=composition.candidate_target,
        candidate_structure_record=CandidateStructureRecordV4(
            composer_version=REASONING_V4_COMPOSER_VERSION,
            status="valid",
            warnings=composition.warnings,
        ),
        records=records,
        errors=errors,
        warnings=[*warnings, *composition.warnings],
        generations=generations,
        provenance=outcomes["TaskGrounding"].provenance,
    )


async def _run_stage_v4(
    source: ReasoningSourceV4,
    *,
    profile: ReasoningV4TeacherProfile,
    backend: ReasoningV4TeacherBackend,
    config: ReasoningV4BatchConfig,
    groundings: GroundingOutputsV4 | None = None,
) -> _StageOutcome:
    """Generate, parse, validate, and repair one LVLM section."""

    max_attempts = min(
        max(int(config.max_retries), 0) + 1,
        16,
    )
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
            previous_answer=answer,
            previous_errors=errors,
        )
        try:
            generation = await backend.generate(
                source.observation,
                question=question,
                system_prompt=profile.system_prompt,
                image_detail=(
                    config.navigation_context_image_detail
                    if profile.section_name
                    == "NavigationContextGrounding"
                    else config.image_detail
                ),
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
        parsed = parse_reasoning_v4_section_answer(
            answer,
            profile,
        )
        errors = list(parsed.errors)
        warnings = list(parsed.warnings)
        if not parsed.valid or parsed.section is None:
            continue
        try:
            section, sanitizer_warnings = (
                sanitize_reasoning_section_evidence_v4(
                    profile.section_name,
                    parsed.section,
                    observation=source.observation,
                )
            )
        except (TypeError, ValueError) as exc:
            errors = [f"section_sanitization_failed:{exc}"]
            section = None
            continue
        warnings.extend(sanitizer_warnings)
        checked = validate_reasoning_section_v4(
            profile.section_name,
            section,
            observation=source.observation,
        )
        errors = list(checked.errors)
        warnings = [*warnings, *checked.warnings]
        if not errors:
            break
        section = None
    status = (
        "accepted"
        if section is not None and not errors
        else "error"
        if provider_failures == max_attempts
        else "rejected"
    )
    record = AnnotationStageRecordV4(
        status=status,
        attempts=provider_failures + len(generations),
        raw_answer=(
            answer[: max(0, int(config.raw_answer_max_chars))]
            if answer is not None
            else None
        ),
        errors=(
            []
            if status == "accepted"
            else errors or ["stage_failed"]
        ),
        warnings=list(dict.fromkeys(warnings)),
        telemetry={
            "image_detail": (
                config.navigation_context_image_detail
                if profile.section_name
                == "NavigationContextGrounding"
                else config.image_detail
            ),
            "successful_provider_calls": len(generations),
            "calls": [
                item.telemetry or {} for item in generations
            ],
        },
        producer_kind="lvlm",
        producer_profile_id=profile.profile_id,
        derivation_version=profile.derivation_version,
        prompt_policy_digest=profile.policy_digest,
    )
    return _StageOutcome(
        section=section,
        record=record,
        generations=tuple(generations),
    )


def _deterministic_task_outcome_v4(
    task: TaskGroundingV4,
    *,
    source: str = "versioned_registry",
) -> _StageOutcome:
    """Build a zero-call deterministic Task outcome."""

    return _StageOutcome(
        section=task,
        record=AnnotationStageRecordV4(
            status="accepted",
            attempts=0,
            producer_kind="deterministic",
            producer_profile_id=(
                "reasoning_v4_task_canonicalizer"
            ),
            derivation_version=(
                REASONING_V4_TASK_CANONICALIZATION_VERSION
            ),
        ),
        generations=(),
        provenance={
            "task_canonicalization_source": source,
        },
    )


def _run_missing_task_outcome_v4() -> _StageOutcome:
    """Build a deterministic rejected Task outcome without instruction."""

    return _StageOutcome(
        section=None,
        record=AnnotationStageRecordV4(
            status="rejected",
            attempts=0,
            errors=["instruction_missing"],
            producer_kind="deterministic",
            producer_profile_id=(
                "reasoning_v4_task_canonicalizer"
            ),
            derivation_version=(
                REASONING_V4_TASK_CANONICALIZATION_VERSION
            ),
        ),
        generations=(),
    )


def _skipped_record_v4(
    profile: ReasoningV4TeacherProfile,
    *,
    reason: str,
) -> AnnotationStageRecordV4:
    """Build one explicit skipped LVLM stage record."""

    return AnnotationStageRecordV4(
        status="skipped",
        attempts=0,
        errors=[reason],
        producer_kind="lvlm",
        producer_profile_id=profile.profile_id,
        derivation_version=profile.derivation_version,
        prompt_policy_digest=profile.policy_digest,
    )


def _typed_section(
    value: BaseModel | None,
    expected: type[BaseModel],
) -> Any:
    """Return value only when it has the expected model type."""

    return value if isinstance(value, expected) else None


def _annotation_v4(
    source: ReasoningSourceV4,
    *,
    record_status: str,
    groundings: GroundingOutputsV4,
    decision_proposal: DecisionProposalV4 | None,
    action_reconciliation: ActionReconciliationAuditV4 | None,
    composer_candidate: DecisionV4 | None,
    candidate_target: Any,
    candidate_structure_record: CandidateStructureRecordV4,
    records: dict[str, AnnotationStageRecordV4],
    errors: list[str],
    warnings: list[str],
    generations: list[TeacherGeneration],
    provenance: dict[str, str],
) -> ReasoningAnnotationV4:
    """Build one annotation and aggregate provider identity."""

    providers = {
        str(item.telemetry.get("provider"))
        for item in generations
        if isinstance(item.telemetry, dict)
        and item.telemetry.get("provider")
    }
    models = {
        str(item.telemetry.get("model"))
        for item in generations
        if isinstance(item.telemetry, dict)
        and item.telemetry.get("model")
    }
    return ReasoningAnnotationV4(
        source_group_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        record_status=record_status,
        grounding_outputs=groundings,
        decision_proposal=decision_proposal,
        action_reconciliation=action_reconciliation,
        composer_candidate=composer_candidate,
        candidate_target=candidate_target,
        candidate_structure_record=candidate_structure_record,
        stage_records=records,
        errors=list(dict.fromkeys(errors)),
        warnings=list(dict.fromkeys(warnings)),
        teacher_provider=(
            next(iter(providers)) if len(providers) == 1 else None
        ),
        teacher_model=(
            next(iter(models)) if len(models) == 1 else None
        ),
        metadata={
            "composer_version": REASONING_V4_COMPOSER_VERSION,
            "task_canonicalization_version": (
                REASONING_V4_TASK_CANONICALIZATION_VERSION
            ),
            **provenance,
            "input_diagnostics": [
                warning
                for warning in source.warnings
                if warning in _SOURCE_INPUT_DIAGNOSTIC_CODES
            ],
        },
    )


async def label_reasoning_source_manifest_v4(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    api_keys: Iterable[str] = (),
    config: ReasoningV4BatchConfig | None = None,
    limit: int | None = None,
    rejected_jsonl: str | Path | None = None,
    report_json: str | Path | None = None,
    backend_factory: Callable[
        [int, str | None],
        ReasoningV4TeacherBackend,
    ]
    | None = None,
) -> dict[str, Any]:
    """Annotate a source manifest with bounded concurrency and resume.

    Args:
        source_jsonl: Frozen source manifest.
        annotation_jsonl: Append-only annotation destination.
        api_keys: Optional worker-local provider keys.
        config: Runner configuration.
        limit: Optional source-row limit.
        rejected_jsonl: Optional non-formal record mirror.
        report_json: Optional atomic progress report.
        backend_factory: Optional custom/test backend factory.

    Returns:
        Final machine-readable annotation report.
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
    for source in sources:
        existing = cached.get(
            source.observation.source_group_id
        )
        stale = _annotation_is_stale_v4(source, existing)
        retryable = (
            existing is not None
            and (
                not existing.all_sections_accepted
                or existing.candidate_structure_record.status
                != "valid"
            )
        )
        if stale or (cfg.retry_failed_on_resume and retryable):
            pending.append(source)
            resume_counts[
                "cache_miss_or_stale"
                if stale
                else "retry_requested"
            ] += 1
        else:
            resume_counts["cached_complete"] += 1

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
    report = _initial_report_v4(
        source_jsonl,
        annotation_jsonl,
        input_rows=len(sources),
        pending_rows=len(pending),
        concurrency=worker_count,
        resume_counts=resume_counts,
        config=cfg,
    )
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
        rejected_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    queue: asyncio.Queue[ReasoningSourceV4 | None] = (
        asyncio.Queue()
    )
    results: asyncio.Queue[
        tuple[ReasoningSourceV4, ReasoningAnnotationV4]
    ] = asyncio.Queue()
    for source in pending:
        queue.put_nowait(source)
    for _ in range(worker_count):
        queue.put_nowait(None)
    task_coordinator = _TaskCanonicalizationCoordinator(
        _task_cache_from_annotations_v4(
            sources,
            cached,
        )
    )

    def make_backend(worker_id: int) -> ReasoningV4TeacherBackend:
        """Build one worker-local provider adapter."""

        key = (
            keys[(worker_id - 1) % len(keys)]
            if keys
            else None
        )
        if backend_factory is not None:
            return backend_factory(worker_id, key)
        return VisionReasoningV4TeacherBackend(api_key=key)

    async def worker(worker_id: int) -> None:
        """Consume frozen sources until the queue sentinel."""

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
                        task_coordinator=task_coordinator,
                    )
                except Exception as exc:
                    annotation = ReasoningAnnotationV4(
                        source_group_id=(
                            source.observation.source_group_id
                        ),
                        source_fingerprint=(
                            source.source_fingerprint
                        ),
                        record_status="error",
                        grounding_outputs=GroundingOutputsV4(),
                        candidate_structure_record=(
                            CandidateStructureRecordV4(
                                composer_version=(
                                    REASONING_V4_COMPOSER_VERSION
                                ),
                                status="not_available",
                                errors=[
                                    "worker_exception:"
                                    f"{type(exc).__name__}:"
                                    f"{str(exc)[:500]}"
                                ],
                            )
                        ),
                        errors=[
                            "worker_exception:"
                            f"{type(exc).__name__}:"
                            f"{str(exc)[:500]}"
                        ],
                        metadata={"worker_id": worker_id},
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
        rejected_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        if rejected_path is not None
        else None
    )
    try:
        with output_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        ) as handle:
            for _ in range(len(pending)):
                source, annotation = await results.get()
                row = annotation.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                append_jsonl_row(handle, row)
                non_formal = (
                    annotation.record_status != "complete"
                    or not annotation.all_sections_accepted
                    or annotation.candidate_structure_record.status
                    != "valid"
                )
                if non_formal and reject_handle is not None:
                    append_jsonl_row(reject_handle, row)
                _accumulate_annotation_metrics_v4(
                    report,
                    source,
                    annotation,
                )
                _refresh_annotation_aggregates_v4(
                    report,
                    cfg,
                )
                elapsed = max(
                    time.monotonic() - started_at,
                    1e-6,
                )
                report["rate_rows_per_hour"] = (
                    report["processed_rows"]
                    / elapsed
                    * 3600.0
                )
                report["pending_rows"] = (
                    len(pending) - report["processed_rows"]
                )
                _write_report_v4(report_json, report)
                if (
                    cfg.progress_interval > 0
                    and report["processed_rows"]
                    % cfg.progress_interval
                    == 0
                ):
                    print(
                        json.dumps(
                            {
                                "reasoning_v4_6_progress": (
                                    f"{report['processed_rows']}/"
                                    f"{len(pending)}"
                                ),
                                "record_complete": report[
                                    "record_complete_rows"
                                ],
                                "all_sections_accepted": report[
                                    "all_sections_accepted_rows"
                                ],
                                "candidate_structurally_valid": report[
                                    "candidate_structurally_valid_rows"
                                ],
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


def _task_cache_from_annotations_v4(
    sources: Iterable[ReasoningSourceV4],
    annotations: dict[str, ReasoningAnnotationV4],
) -> dict[str, TaskGroundingV4]:
    """Recover stable unknown-task labels from compatible annotations.

    Args:
        sources: Current frozen source rows.
        annotations: Latest append-only annotations by source group.

    Returns:
        Instruction-fingerprint cache reused by resumed annotation.

    Raises:
        ValueError: If one normalized instruction has conflicting labels.
    """

    result: dict[str, TaskGroundingV4] = {}
    for source in sources:
        instruction = source.observation.instruction
        if instruction is None:
            continue
        if canonicalize_task_instruction_v4(instruction) is not None:
            # Built-in registry ownership always wins.  Never recover an old
            # LVLM/cache label for an instruction added by a newer registry.
            continue
        annotation = annotations.get(
            source.observation.source_group_id
        )
        if (
            annotation is None
            or annotation.source_fingerprint
            != source.source_fingerprint
        ):
            continue
        task = annotation.grounding_outputs.task_grounding
        stage = annotation.stage_records.get("TaskGrounding")
        if (
            task is None
            or stage is None
            or stage.status != "accepted"
        ):
            continue
        fingerprint = task_instruction_fingerprint_v4(
            instruction
        )
        previous = result.get(fingerprint)
        if previous is not None and previous != task:
            raise ValueError(
                "conflicting cached task taxonomy for instruction "
                f"fingerprint: {fingerprint}"
            )
        result[fingerprint] = task
    return result


def _annotation_is_stale_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4 | None,
) -> bool:
    """Return whether a cached row cannot be reused."""

    if annotation is None:
        return True
    if (
        annotation.source_fingerprint
        != source.source_fingerprint
        or annotation.schema_version
        != REASONING_ANNOTATION_V4_SCHEMA_VERSION
        or annotation.metadata.get("composer_version")
        != REASONING_V4_COMPOSER_VERSION
    ):
        return True
    registered_task = canonicalize_task_instruction_v4(
        source.observation.instruction
    )
    if registered_task is not None:
        task_record = annotation.stage_records.get("TaskGrounding")
        if (
            annotation.grounding_outputs.task_grounding
            != registered_task.task_grounding
            or task_record is None
            or task_record.status != "accepted"
            or task_record.producer_kind != "deterministic"
            or task_record.derivation_version
            != REASONING_V4_TASK_CANONICALIZATION_VERSION
        ):
            return True
    for name, profile in REASONING_V4_TEACHER_PROFILES.items():
        record = annotation.stage_records.get(name)
        if record is None:
            return True
        if record.producer_kind == "lvlm" and (
            record.producer_profile_id != profile.profile_id
            or record.derivation_version
            != profile.derivation_version
            or record.prompt_policy_digest
            != profile.policy_digest
        ):
            return True
        if (
            record.producer_kind == "deterministic"
            and (
                name != "TaskGrounding"
                or record.derivation_version
                != REASONING_V4_TASK_CANONICALIZATION_VERSION
            )
        ):
            return True
    return False


def _latest_annotation_cache_v4(
    path: str | Path,
) -> dict[str, ReasoningAnnotationV4]:
    """Load latest append-only annotation per source group."""

    value = Path(path).expanduser()
    if not value.exists():
        return {}
    cache: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(value):
        if (
            row.get("schema_version")
            != REASONING_ANNOTATION_V4_SCHEMA_VERSION
        ):
            continue
        annotation = ReasoningAnnotationV4.model_validate(
            row
        )
        cache[annotation.source_group_id] = annotation
    return cache


def _initial_report_v4(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    input_rows: int,
    pending_rows: int,
    concurrency: int,
    resume_counts: Counter[str],
    config: ReasoningV4BatchConfig,
) -> dict[str, Any]:
    """Build a stable machine-readable progress report."""

    return {
        "schema_version": (
            f"{REASONING_ANNOTATION_V4_SCHEMA_VERSION}_report_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "input_rows": input_rows,
        "pending_rows": pending_rows,
        "processed_rows": 0,
        "record_complete_rows": 0,
        "record_error_rows": 0,
        "all_sections_accepted_rows": 0,
        "candidate_structurally_valid_rows": 0,
        "pending_human_rows": 0,
        "concurrency": concurrency,
        "resume_counts": dict(resume_counts),
        "composer_version": REASONING_V4_COMPOSER_VERSION,
        "task_canonicalization_version": (
            REASONING_V4_TASK_CANONICALIZATION_VERSION
        ),
        "status": "running",
        "error_counts": {},
        "warning_counts": {},
        "dataset_counts": {},
        "stage_metrics": {},
        "candidate_structure_counts": {},
        "image_detail": config.image_detail,
        "navigation_context_image_detail": (
            config.navigation_context_image_detail
        ),
        "review_trigger_counts": {},
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
                config.input_cost_per_million_tokens
            ),
            "output_cost_per_million_tokens": (
                config.output_cost_per_million_tokens
            ),
            "estimated_total": None,
            "status": (
                "configured"
                if config.input_cost_per_million_tokens is not None
                and config.output_cost_per_million_tokens is not None
                else "unavailable_no_pricing_contract"
            ),
        },
    }


def _accumulate_annotation_metrics_v4(
    report: dict[str, Any],
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
) -> None:
    """Accumulate status, stage, warning, and telemetry metrics."""

    report["processed_rows"] += 1
    report[
        f"record_{annotation.record_status}_rows"
    ] += 1
    report["all_sections_accepted_rows"] += int(
        annotation.all_sections_accepted
    )
    structure_status = annotation.candidate_structure_record.status
    report["candidate_structurally_valid_rows"] += int(
        structure_status == "valid"
    )
    report["candidate_structure_counts"][structure_status] = (
        int(
            report["candidate_structure_counts"].get(
                structure_status,
                0,
            )
        )
        + 1
    )
    dataset = source.observation.dataset_source
    report["dataset_counts"][dataset] = (
        int(report["dataset_counts"].get(dataset, 0)) + 1
    )
    audit = annotation.action_reconciliation
    if audit is not None and audit.review_trigger_codes:
        report["pending_human_rows"] += 1
        for code in audit.review_trigger_codes:
            value = str(code)
            report["review_trigger_counts"][value] = (
                int(
                    report["review_trigger_counts"].get(
                        value,
                        0,
                    )
                )
                + 1
            )
    for error in annotation.errors:
        code = error.split(":", 1)[0]
        report["error_counts"][code] = (
            int(report["error_counts"].get(code, 0)) + 1
        )
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
                "producer_counts": {},
                "provider_calls": 0,
                "roundtrip_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
            },
        )
        metric[record.status] += 1
        metric["attempts"] += int(record.attempts)
        producer = record.producer_kind
        metric["producer_counts"][producer] = (
            int(metric["producer_counts"].get(producer, 0))
            + 1
        )
        calls = record.telemetry.get("calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            metric["provider_calls"] += 1
            totals["provider_calls"] += 1
            roundtrip_ms = _finite_number(
                call.get("roundtrip_ms")
            )
            metric["roundtrip_ms"] += roundtrip_ms
            totals["roundtrip_ms"] += roundtrip_ms
            response = call.get("response")
            usage = (
                response.get("usage")
                if isinstance(response, dict)
                and isinstance(response.get("usage"), dict)
                else {}
            )
            prompt_details = usage.get(
                "prompt_tokens_details"
            )
            values = {
                "prompt_tokens": _nonnegative_int(
                    usage.get("prompt_tokens")
                ),
                "completion_tokens": _nonnegative_int(
                    usage.get("completion_tokens")
                ),
                "total_tokens": _nonnegative_int(
                    usage.get("total_tokens")
                ),
                "cached_prompt_tokens": _nonnegative_int(
                    prompt_details.get("cached_tokens")
                    if isinstance(prompt_details, dict)
                    else call.get("prompt_cached_tokens")
                ),
            }
            for key, value in values.items():
                metric[key] += value
                totals[key] += value


def _refresh_annotation_aggregates_v4(
    report: dict[str, Any],
    config: ReasoningV4BatchConfig,
) -> None:
    """Refresh acceptance, latency, and optional cost estimates."""

    for metric in report["stage_metrics"].values():
        records = sum(
            int(metric[name])
            for name in (
                "accepted",
                "rejected",
                "error",
                "skipped",
            )
        )
        calls = int(metric["provider_calls"])
        metric["acceptance_fraction"] = (
            float(metric["accepted"]) / records
            if records
            else 0.0
        )
        metric["average_attempts"] = (
            float(metric["attempts"]) / records
            if records
            else 0.0
        )
        metric["average_roundtrip_ms"] = (
            float(metric["roundtrip_ms"]) / calls
            if calls
            else None
        )
    totals = report["telemetry_totals"]
    calls = int(totals["provider_calls"])
    totals["average_roundtrip_ms"] = (
        float(totals["roundtrip_ms"]) / calls
        if calls
        else None
    )
    input_rate = config.input_cost_per_million_tokens
    output_rate = config.output_cost_per_million_tokens
    if input_rate is not None and output_rate is not None:
        report["cost_estimate"]["estimated_total"] = (
            int(totals["prompt_tokens"]) * float(input_rate)
            + int(totals["completion_tokens"])
            * float(output_rate)
        ) / 1_000_000.0


def _finite_number(value: Any) -> float:
    """Return one finite non-negative telemetry number."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if (
        result < 0
        or result == float("inf")
        or result != result
    ):
        return 0.0
    return result


def _nonnegative_int(value: Any) -> int:
    """Return one non-negative telemetry integer."""

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
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )
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
    "ReasoningV4BatchConfig",
    "ReasoningV4TeacherBackend",
    "VisionReasoningV4TeacherBackend",
    "annotate_reasoning_source_v4",
    "label_reasoning_source_manifest_v4",
]
