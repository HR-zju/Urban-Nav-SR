"""Resumable multimodal annotation runner for the minimal Reasoning V3.4 contract."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from mcp_server.agent.reasoning_v3_text import render_reasoning_target_v3
from mcp_server.agent.reasoning_v3_validation import validate_reasoning_target_v3
from mcp_server.contracts.reasoning_dataset_v3 import (
    REASONING_ANNOTATION_V3_SCHEMA_VERSION,
    ReasoningAnnotationV3,
    ReasoningSourceV3,
)
from mcp_server.data.jsonl_io import append_jsonl_row, iter_jsonl
from mcp_server.data.reasoning_v3_composer import (
    REASONING_V3_COMPOSER_VERSION,
    compose_reasoning_target_v3,
    derive_decision_outcome_v3,
)
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3
from mcp_server.teacher.reasoning_v2_teacher import (
    TeacherGeneration,
    VisionReasoningTeacherBackend,
)
from mcp_server.teacher.reasoning_v3_profiles import (
    TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3,
    TEACHER_OBSERVATION_RUNTIME_PROFILE_V3,
    validate_observation_with_runtime_v3,
    validate_rationale_with_runtime_v3,
)
from mcp_server.teacher.reasoning_v3_fewshot import REASONING_V3_FEWSHOT_VERSION


_OBSERVATION_RUNTIME = TEACHER_OBSERVATION_RUNTIME_PROFILE_V3
_RATIONALE_RUNTIME = TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3
REASONING_V3_TEACHER_PROMPT_VERSION = _OBSERVATION_RUNTIME.profile.profile_id
REASONING_V3_RATIONALE_PROMPT_VERSION = _RATIONALE_RUNTIME.profile.profile_id


@dataclass(frozen=True)
class ReasoningV3BatchConfig:
    """Configure resumable V3.4 teacher annotation.

    Args:
        max_retries: Maximum repair calls after the initial request.
        image_detail: OpenAI-compatible image detail setting.
        concurrency: Maximum concurrent annotation workers.
        retry_failed_on_resume: Retry cached rejected and error records.
        raw_answer_max_chars: Maximum provider text retained for audit.
        progress_interval: Completed-row interval for progress output.
    """

    max_retries: int = 2
    image_detail: str = "low"
    concurrency: int = 1
    retry_failed_on_resume: bool = False
    raw_answer_max_chars: int = 50000
    progress_interval: int = 100


class ReasoningV3TeacherBackend(Protocol):
    """Protocol shared by semantic-observation and rationale teachers."""

    async def generate(
        self,
        source: ReasoningSourceV3,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Generate one raw V3.4 teacher response.

        Args:
            source: Frozen source whose images are sent to the backend.
            question: Stage-specific user prompt.
            system_prompt: Versioned teacher policy.
            image_detail: Provider image-detail setting.

        Returns:
            Raw provider generation and telemetry.
        """


class VisionReasoningV3TeacherBackend:
    """Adapt the existing OpenAI-compatible vision backend to wrapped V3 sources."""

    def __init__(self, *, api_key: str | None = None):
        """Initialize the existing vision backend adapter.

        Args:
            api_key: Optional worker-specific API key.
        """

        self._delegate = VisionReasoningTeacherBackend(api_key=api_key)

    async def generate(
        self,
        source: ReasoningSourceV3,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Generate a teacher response while exposing only normalized images.

        Args:
            source: Frozen V3.4 source wrapper.
            question: Stage-specific user prompt.
            system_prompt: Versioned teacher policy.
            image_detail: Provider image-detail setting.

        Returns:
            Raw provider generation and telemetry.
        """

        return await self._delegate.generate(
            source.model_input,
            question=question,
            system_prompt=system_prompt,
            image_detail=image_detail,
        )


async def annotate_reasoning_source_v3(
    source: ReasoningSourceV3,
    *,
    backend: ReasoningV3TeacherBackend,
    config: ReasoningV3BatchConfig | None = None,
) -> ReasoningAnnotationV3:
    """Generate, repair, compose, and validate one frozen V3.4 source.

    Args:
        source: Fingerprinted source with isolated private supervision.
        backend: Multimodal provider shared by both bounded teacher stages.
        config: Retry, image, and audit limits.

    Returns:
        Full-CoT, observation-only, or rejected/error annotation record.
    """

    cfg = config or ReasoningV3BatchConfig()
    max_attempts = min(max(0, int(cfg.max_retries)) + 1, 16)
    errors: list[str] = []
    warnings: list[str] = []
    history: list[dict[str, Any]] = []
    raw_answers: dict[str, str] = {}
    generations: list[TeacherGeneration] = []
    provider_errors = 0
    total_calls = 0
    observation = None

    # Pass A+B: task semantics and visual facts see model-visible input only.
    for attempt in range(max_attempts):
        question = _OBSERVATION_RUNTIME.user_prompt_builder(
            source,
            previous_errors=errors if attempt else (),
            previous_answer=raw_answers.get("semantic_observation") if attempt else None,
        )
        total_calls += 1
        try:
            generation = await backend.generate(
                source,
                question=question,
                system_prompt=_OBSERVATION_RUNTIME.profile.system_prompt,
                image_detail=cfg.image_detail,
            )
        except Exception as exc:
            provider_errors += 1
            errors = [f"semantic_observation_call_failed:{type(exc).__name__}:{str(exc)[:500]}"]
            history.append({"stage": "semantic_observation", "attempt": attempt + 1, "errors": list(errors)})
            continue
        generations.append(generation)
        raw_answers["semantic_observation"] = str(generation.answer or "").strip()
        parsed = _OBSERVATION_RUNTIME.parser(raw_answers["semantic_observation"])
        errors = list(parsed.errors)
        if not parsed.valid or parsed.observation is None:
            history.append({"stage": "semantic_observation", "attempt": attempt + 1, "errors": list(errors)})
            continue
        observation = parsed.observation
        context_sources = build_navigation_context_sources_v3(source)
        checked = validate_observation_with_runtime_v3(
            _OBSERVATION_RUNTIME,
            observation,
            instruction=source.model_input.instruction,
            evidence_catalog=source.model_input.evidence_catalog,
            available_modalities=source.model_input.available_modalities,
            navigation_context_source_refs=[item.source_ref for item in context_sources],
        )
        errors = list(checked.errors)
        warnings = list(checked.warnings)
        if errors:
            history.append({"stage": "semantic_observation", "attempt": attempt + 1, "errors": list(errors)})
            continue
        break

    if observation is None or errors:
        return _failed_annotation_v3(
            source,
            status="error" if provider_errors == total_calls else "rejected",
            observation=observation,
            decision_rationale=None,
            raw_answers=raw_answers,
            errors=errors or ["semantic_observation_validation_failed"],
            warnings=warnings,
            attempts=total_calls,
            generations=generations,
            history=history,
            raw_answer_max_chars=cfg.raw_answer_max_chars,
        )

    observation_only_reason = _observation_only_reason_v3(source)
    if observation_only_reason is not None:
        latest = generations[-1]
        return ReasoningAnnotationV3(
            source_group_id=source.model_input.source_group_id,
            source_fingerprint=source.source_fingerprint,
            status="observation_only",
            observation=observation,
            raw_answer=_serialize_raw_answers(raw_answers, cfg.raw_answer_max_chars),
            errors=[],
            warnings=list(dict.fromkeys([*warnings, observation_only_reason])),
            label_provenance=source.label_provenance,
            attempts=total_calls,
            teacher_provider=latest.provider,
            teacher_model=latest.model,
            telemetry=_aggregate_telemetry(generations),
            metadata=_annotation_metadata_v3(
                history=history,
                annotation_mode="observation_only",
                observation_only_reason=observation_only_reason,
            ),
        )

    try:
        outcome = derive_decision_outcome_v3(source, observation)
    except (KeyError, TypeError, ValueError) as exc:
        errors = [f"outcome_composition_failed:{exc}"]
        history.append({"stage": "outcome_composer", "attempt": 1, "errors": list(errors)})
        # Outcome errors may expose private GT, so never send them back to the observation teacher.
        return _failed_annotation_v3(
            source,
            status="rejected",
            observation=observation,
            decision_rationale=None,
            raw_answers=raw_answers,
            errors=errors,
            warnings=warnings,
            attempts=total_calls,
            generations=generations,
            history=history,
            raw_answer_max_chars=cfg.raw_answer_max_chars,
        )

    rationale = None
    composition = None
    rationale_provider_errors = 0
    # Pass C: GT constrains the outcome, while reasons can only cite frozen Pass A/B facts.
    for attempt in range(max_attempts):
        question = _RATIONALE_RUNTIME.user_prompt_builder(
            source,
            observation=observation,
            expected_outcome=outcome.model_dump(),
            previous_errors=errors if attempt else (),
            previous_answer=raw_answers.get("decision_rationale") if attempt else None,
        )
        total_calls += 1
        try:
            generation = await backend.generate(
                source,
                question=question,
                system_prompt=_RATIONALE_RUNTIME.profile.system_prompt,
                image_detail=cfg.image_detail,
            )
        except Exception as exc:
            provider_errors += 1
            rationale_provider_errors += 1
            errors = [f"decision_rationale_call_failed:{type(exc).__name__}:{str(exc)[:500]}"]
            history.append({"stage": "decision_rationale", "attempt": attempt + 1, "errors": list(errors)})
            continue
        generations.append(generation)
        raw_answers["decision_rationale"] = str(generation.answer or "").strip()
        parsed = _RATIONALE_RUNTIME.parser(raw_answers["decision_rationale"])
        errors = list(parsed.errors)
        if not parsed.valid or parsed.rationale is None:
            history.append({"stage": "decision_rationale", "attempt": attempt + 1, "errors": list(errors)})
            continue
        rationale = parsed.rationale
        checked = validate_rationale_with_runtime_v3(
            _RATIONALE_RUNTIME,
            rationale,
            expected_outcome=outcome.model_dump(),
            observation=observation,
            evidence_catalog=source.model_input.evidence_catalog,
            fixed_labels=source.teacher_supervision_context.get("fixed_labels") or {},
        )
        errors = list(checked.errors)
        warnings = [*warnings, *checked.warnings]
        if errors:
            history.append({"stage": "decision_rationale", "attempt": attempt + 1, "errors": list(errors)})
            continue
        try:
            composition = compose_reasoning_target_v3(source, observation, rationale)
        except (TypeError, ValueError) as exc:
            errors = [f"composition_failed:{exc}"]
            history.append({"stage": "decision_rationale", "attempt": attempt + 1, "errors": list(errors)})
            continue
        target = composition.target
        validation = validate_reasoning_target_v3(
            target,
            instruction=source.model_input.instruction,
            evidence_catalog=source.model_input.evidence_catalog,
            available_modalities=source.model_input.available_modalities,
            label_provenance=composition.label_provenance,
            section_loss_mask=composition.section_loss_mask,
            source_section_loss_mask=source.section_loss_mask,
            fixed_labels=source.teacher_supervision_context.get("fixed_labels") or {},
            dataset_source=source.model_input.dataset_source,
            assistant_text=render_reasoning_target_v3(target),
        )
        errors = list(validation.errors)
        warnings = [*warnings, *composition.warnings, *validation.warnings]
        if not errors:
            latest = generations[-1]
            return ReasoningAnnotationV3(
                source_group_id=source.model_input.source_group_id,
                source_fingerprint=source.source_fingerprint,
                status="accepted",
                observation=observation,
                decision_rationale=rationale,
                target=target,
                section_loss_mask=composition.section_loss_mask,
                raw_answer=_serialize_raw_answers(raw_answers, cfg.raw_answer_max_chars),
                errors=[],
                warnings=list(dict.fromkeys(warnings)),
                label_provenance=composition.label_provenance,
                attempts=total_calls,
                teacher_provider=latest.provider,
                teacher_model=latest.model,
                telemetry=_aggregate_telemetry(generations),
                metadata=_annotation_metadata_v3(
                    history=history,
                    target=target,
                    annotation_mode="full_cot",
                ),
            )
        history.append({"stage": "decision_rationale", "attempt": attempt + 1, "errors": list(errors)})

    return _failed_annotation_v3(
        source,
        status="error" if rationale_provider_errors == max_attempts else "rejected",
        observation=observation,
        decision_rationale=rationale,
        raw_answers=raw_answers,
        errors=errors or ["decision_rationale_validation_failed"],
        warnings=warnings,
        attempts=total_calls,
        generations=generations,
        history=history,
        raw_answer_max_chars=cfg.raw_answer_max_chars,
    )


def _failed_annotation_v3(
    source: ReasoningSourceV3,
    *,
    status: str,
    observation: Any,
    decision_rationale: Any,
    raw_answers: dict[str, str],
    errors: list[str],
    warnings: list[str],
    attempts: int,
    generations: list[TeacherGeneration],
    history: list[dict[str, Any]],
    raw_answer_max_chars: int,
) -> ReasoningAnnotationV3:
    """Build a rejected or provider-error audit row.

    Args:
        source: Frozen source bound to the failed attempt.
        status: ``rejected`` for semantic failure or ``error`` for provider failure.
        observation: Last schema-valid first-stage observation, if any.
        decision_rationale: Last schema-valid rationale, if any.
        raw_answers: Latest raw response for each stage.
        errors: Final hard errors.
        warnings: Non-fatal findings accumulated before failure.
        attempts: Total provider calls across both stages.
        generations: Successful provider responses used for audit metadata.
        history: Per-stage repair history.
        raw_answer_max_chars: Maximum serialized raw-answer length.

    Returns:
        Valid V3.4 failed annotation without a trainable target.
    """

    latest = generations[-1] if generations else None
    return ReasoningAnnotationV3(
        source_group_id=source.model_input.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="error" if status == "error" else "rejected",
        observation=observation,
        decision_rationale=decision_rationale,
        raw_answer=_serialize_raw_answers(raw_answers, raw_answer_max_chars),
        errors=errors or ["teacher_validation_failed"],
        warnings=list(dict.fromkeys(warnings)),
        attempts=attempts,
        teacher_provider=latest.provider if latest else None,
        teacher_model=latest.model if latest else None,
        telemetry=_aggregate_telemetry(generations),
        metadata=_annotation_metadata_v3(history=history),
    )


def _serialize_raw_answers(values: dict[str, str], max_chars: int) -> str | None:
    """Serialize both stage responses into one bounded audit field."""

    if not values:
        return None
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))[: max(0, int(max_chars))]


def _aggregate_telemetry(generations: list[TeacherGeneration]) -> dict[str, Any]:
    """Preserve per-call provider telemetry without assuming token-key names."""

    return {
        "successful_provider_calls": len(generations),
        "calls": [item.telemetry or {} for item in generations],
    }


def _annotation_metadata_v3(
    *,
    history: list[dict[str, Any]],
    target: Any = None,
    annotation_mode: str | None = None,
    observation_only_reason: str | None = None,
) -> dict[str, Any]:
    """Build version-complete metadata used for cache invalidation."""

    metadata: dict[str, Any] = {
        "teacher_prompt_version": REASONING_V3_TEACHER_PROMPT_VERSION,
        "teacher_prompt_policy_digest": _OBSERVATION_RUNTIME.profile.policy_digest,
        "rationale_prompt_version": REASONING_V3_RATIONALE_PROMPT_VERSION,
        "rationale_prompt_policy_digest": _RATIONALE_RUNTIME.profile.policy_digest,
        "fewshot_registry_version": REASONING_V3_FEWSHOT_VERSION,
        "composer_version": REASONING_V3_COMPOSER_VERSION,
        "repair_history": history,
    }
    if annotation_mode is not None:
        metadata["annotation_mode"] = annotation_mode
    if observation_only_reason is not None:
        metadata["observation_only_reason"] = observation_only_reason
    if target is not None:
        metadata["decision_rationale_status"] = target.reasoning.decision.rationale_status
    return metadata


def _observation_only_reason_v3(source: ReasoningSourceV3) -> str | None:
    """Return why a valid Pass1 result cannot become a Full-CoT target."""

    dataset_source = source.model_input.dataset_source.strip().lower()
    if dataset_source.startswith("frodobots"):
        return "direction_supervision_disabled_by_dataset_policy"
    fixed_labels = source.teacher_supervision_context.get("fixed_labels") or {}
    if not isinstance(fixed_labels.get("canonical_direction"), dict):
        return "canonical_direction_missing"
    return None


async def label_reasoning_source_manifest_v3(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    rejected_jsonl: str | Path | None = None,
    report_json: str | Path | None = None,
    config: ReasoningV3BatchConfig | None = None,
    limit: int | None = None,
    api_keys: Iterable[str] = (),
    backend_factory: Callable[[int, str | None], ReasoningV3TeacherBackend] | None = None,
) -> dict[str, Any]:
    """Run bounded concurrent V3.4 labeling with append-only cache.

    Args:
        source_jsonl: Frozen source manifest.
        annotation_jsonl: Append-only destination annotation manifest.
        rejected_jsonl: Optional rejected/error mirror.
        report_json: Optional atomically refreshed progress report.
        config: Retry, concurrency, and audit limits.
        limit: Optional maximum source rows.
        api_keys: Optional worker-rotated provider keys.
        backend_factory: Optional backend factory for tests or custom serving.

    Returns:
        Final labeling report with cache, status, error, and rate counts.

    Raises:
        ValueError: If source or cached annotation rows violate V3.4 schemas.
    """

    cfg = config or ReasoningV3BatchConfig()
    sources = [ReasoningSourceV3.model_validate(row) for row in iter_jsonl(source_jsonl)]
    if limit is not None:
        sources = sources[: int(limit)]
    completed = _latest_annotation_cache(annotation_jsonl)
    pending: list[ReasoningSourceV3] = []
    resume_counts: Counter[str] = Counter()
    for source in sources:
        source_id = source.model_input.source_group_id
        cached = completed.get(source_id)
        if cached is None or cached.source_fingerprint != source.source_fingerprint:
            pending.append(source)
            resume_counts["cache_miss_or_stale"] += 1
        elif cached.metadata.get("teacher_prompt_version") != REASONING_V3_TEACHER_PROMPT_VERSION:
            pending.append(source)
            resume_counts["cache_stale_prompt_version"] += 1
        elif cached.metadata.get("teacher_prompt_policy_digest") != _OBSERVATION_RUNTIME.profile.policy_digest:
            pending.append(source)
            resume_counts["cache_stale_prompt_policy_digest"] += 1
        elif cached.metadata.get("fewshot_registry_version") != REASONING_V3_FEWSHOT_VERSION:
            pending.append(source)
            resume_counts["cache_stale_fewshot_registry"] += 1
        elif cached.status == "observation_only":
            resume_counts["cached_observation_only"] += 1
        elif cached.metadata.get("rationale_prompt_version") != REASONING_V3_RATIONALE_PROMPT_VERSION:
            pending.append(source)
            resume_counts["cache_stale_rationale_prompt_version"] += 1
        elif cached.metadata.get("rationale_prompt_policy_digest") != _RATIONALE_RUNTIME.profile.policy_digest:
            pending.append(source)
            resume_counts["cache_stale_rationale_prompt_policy_digest"] += 1
        elif cached.metadata.get("composer_version") != REASONING_V3_COMPOSER_VERSION:
            pending.append(source)
            resume_counts["cache_stale_composer_version"] += 1
        elif cached.status == "accepted":
            resume_counts["cached_accepted"] += 1
        elif cfg.retry_failed_on_resume:
            pending.append(source)
            resume_counts[f"retried_cached_{cached.status}"] += 1
        else:
            resume_counts[f"cached_{cached.status}"] += 1

    keys = tuple(str(key).strip() for key in api_keys if str(key).strip())
    worker_count = min(max(1, int(cfg.concurrency)), max(1, len(pending)))
    if keys:
        worker_count = min(worker_count, len(keys))
    output_path = Path(annotation_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = Path(rejected_jsonl).expanduser() if rejected_jsonl is not None else None
    if rejected_path is not None:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": f"{REASONING_ANNOTATION_V3_SCHEMA_VERSION}_report",
        "teacher_prompt_version": REASONING_V3_TEACHER_PROMPT_VERSION,
        "teacher_prompt_policy_digest": _OBSERVATION_RUNTIME.profile.policy_digest,
        "rationale_prompt_version": REASONING_V3_RATIONALE_PROMPT_VERSION,
        "rationale_prompt_policy_digest": _RATIONALE_RUNTIME.profile.policy_digest,
        "fewshot_registry_version": REASONING_V3_FEWSHOT_VERSION,
        "composer_version": REASONING_V3_COMPOSER_VERSION,
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "rejected_jsonl": str(rejected_jsonl) if rejected_jsonl is not None else None,
        "input_rows": len(sources),
        "pending_rows": len(pending),
        "processed_rows": 0,
        "accepted_rows": 0,
        "observation_only_rows": 0,
        "rejected_rows": 0,
        "error_rows": 0,
        "unsupported_rationale_rows": 0,
        "partial_rationale_rows": 0,
        "concurrency": worker_count,
        "resume_counts": dict(resume_counts),
        "status": "running",
        "error_counts": {},
        "dataset_counts": {},
    }
    if not pending:
        report["status"] = "done"
        _write_report(report_json, report)
        return report

    queue: asyncio.Queue[ReasoningSourceV3 | None] = asyncio.Queue()
    results: asyncio.Queue[tuple[ReasoningSourceV3, ReasoningAnnotationV3]] = asyncio.Queue()
    for source in pending:
        queue.put_nowait(source)
    for _ in range(worker_count):
        queue.put_nowait(None)

    def make_backend(worker_id: int) -> ReasoningV3TeacherBackend:
        key = keys[(worker_id - 1) % len(keys)] if keys else None
        if backend_factory is not None:
            return backend_factory(worker_id, key)
        return VisionReasoningV3TeacherBackend(api_key=key)

    async def worker(worker_id: int) -> None:
        backend = make_backend(worker_id)
        while True:
            source = await queue.get()
            try:
                if source is None:
                    return
                try:
                    annotation = await annotate_reasoning_source_v3(source, backend=backend, config=cfg)
                except Exception as exc:
                    annotation = ReasoningAnnotationV3(
                        source_group_id=source.model_input.source_group_id,
                        source_fingerprint=source.source_fingerprint,
                        status="error",
                        errors=[f"worker_exception:{type(exc).__name__}:{str(exc)[:500]}"],
                        attempts=0,
                        metadata={
                            "worker_id": worker_id,
                            **_annotation_metadata_v3(history=[]),
                        },
                    )
                await results.put((source, annotation))
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker(index)) for index in range(1, worker_count + 1)]
    started_at = time.monotonic()
    reject_handle = rejected_path.open("a", encoding="utf-8", buffering=1) if rejected_path else None
    try:
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for _ in range(len(pending)):
                source, annotation = await results.get()
                serialized = annotation.model_dump(mode="json", exclude_none=True)
                append_jsonl_row(handle, serialized)
                if annotation.status in {"rejected", "error"} and reject_handle is not None:
                    append_jsonl_row(reject_handle, serialized)
                report["processed_rows"] += 1
                report[f"{annotation.status}_rows"] += 1
                dataset = source.model_input.dataset_source
                report["dataset_counts"][dataset] = int(report["dataset_counts"].get(dataset, 0)) + 1
                if annotation.target is not None:
                    rationale_status = annotation.target.reasoning.decision.rationale_status
                    if rationale_status == "unsupported":
                        report["unsupported_rationale_rows"] += 1
                    elif rationale_status == "partial":
                        report["partial_rationale_rows"] += 1
                for error in annotation.errors:
                    key = error.split(":", 1)[0]
                    report["error_counts"][key] = int(report["error_counts"].get(key, 0)) + 1
                elapsed = max(time.monotonic() - started_at, 1e-6)
                report["rate_rows_per_hour"] = report["processed_rows"] / elapsed * 3600.0
                report["pending_rows"] = len(pending) - report["processed_rows"]
                _write_report(report_json, report)
                if cfg.progress_interval > 0 and report["processed_rows"] % cfg.progress_interval == 0:
                    print(
                        json.dumps(
                            {
                                "reasoning_v3_progress": f"{report['processed_rows']}/{len(pending)}",
                                "accepted": report["accepted_rows"],
                                "observation_only": report["observation_only_rows"],
                                "rejected": report["rejected_rows"],
                                "error": report["error_rows"],
                                "rate_rows_per_hour": round(report["rate_rows_per_hour"], 1),
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
    _write_report(report_json, report)
    return report


def _latest_annotation_cache(path: str | Path) -> dict[str, ReasoningAnnotationV3]:
    value = Path(path).expanduser()
    if not value.is_file():
        return {}
    result: dict[str, ReasoningAnnotationV3] = {}
    for row in iter_jsonl(value):
        annotation = ReasoningAnnotationV3.model_validate(row)
        result[annotation.source_group_id] = annotation
    return result


def _write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


__all__ = [
    "REASONING_V3_RATIONALE_PROMPT_VERSION",
    "REASONING_V3_TEACHER_PROMPT_VERSION",
    "ReasoningV3BatchConfig",
    "ReasoningV3TeacherBackend",
    "VisionReasoningV3TeacherBackend",
    "annotate_reasoning_source_v3",
    "label_reasoning_source_manifest_v3",
]
