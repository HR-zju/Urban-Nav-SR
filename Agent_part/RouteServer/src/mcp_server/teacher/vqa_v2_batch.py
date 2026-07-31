"""Resumable deterministic/weak-label annotation runner for VQA V2."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQACapabilityConfigV2,
    VQASourceV2,
    annotation_context_digest_v2,
)
from mcp_server.contracts.vqa_v2 import VQAQuestionV2
from mcp_server.data.jsonl_io import append_jsonl_row, iter_jsonl
from mcp_server.data.vqa_v2_producers import (
    CanonicalDemonstratedDirectionProducerV2,
    VQAProducerResultV2,
)
from mcp_server.data.vqa_v2_eligibility import vqa_task_eligibility_v2
from mcp_server.data.vqa_v2_questions import build_vqa_questions_v2
from mcp_server.teacher.reasoning_v2_teacher import (
    TeacherGeneration,
    VisionReasoningTeacherBackend,
)
from mcp_server.teacher.vqa_v2_profiles import (
    LVLM_VQA_PRODUCER_ID,
    LVLM_VQA_PRODUCER_VERSION,
    VQA_V2_TEACHER_PROFILES,
    VQATeacherProfileV2,
)
from mcp_server.teacher.vqa_v2_teacher import (
    build_vqa_teacher_question_v2,
    parse_vqa_teacher_answer_v2,
)


@dataclass(frozen=True)
class VQABatchConfigV2:
    """Configure bounded VQA annotation.

    Args:
        max_retries: Repair attempts after the initial request.
        image_detail: Provider image detail.
        concurrency: Maximum in-flight source rows.
        retry_failed_on_resume: Reserved explicit retry policy.
        progress_interval: Completed-source reporting interval.
        raw_answer_max_chars: Maximum provider text retained in metadata.
    """

    max_retries: int = 2
    image_detail: str = "low"
    concurrency: int = 1
    retry_failed_on_resume: bool = False
    progress_interval: int = 100
    raw_answer_max_chars: int = 20000


class VQATeacherBackendV2(Protocol):
    """Backend boundary that receives only visible VQA observation."""

    async def generate(
        self,
        source: Any,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Generate one raw answer from visible inputs."""


class VisionVQATeacherBackendV2:
    """Adapt the existing OpenAI-compatible vision backend."""

    def __init__(self, *, api_key: str | None = None):
        """Initialize the backend without serializing its key.

        Args:
            api_key: Optional in-memory per-worker key.
        """

        self._delegate = VisionReasoningTeacherBackend(api_key=api_key)

    async def generate(
        self,
        source: Any,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Call the provider with only image paths from visible observation.

        Args:
            source: Frozen VQA source.
            question: Canonical teacher question.
            system_prompt: Task teacher policy.
            image_detail: Provider detail level.

        Returns:
            Raw generation and sanitized telemetry.
        """

        adapter = _VisibleVQASource(
            images=source.observation.images,
            image_roles=source.observation.image_roles,
        )
        return await self._delegate.generate(
            adapter,  # type: ignore[arg-type]
            question=question,
            system_prompt=system_prompt,
            image_detail=image_detail,
        )


@dataclass(frozen=True)
class _VisibleVQASource:
    """Expose only media fields used by the shared vision backend."""

    images: list[str]
    image_roles: list[str]


@dataclass(frozen=True)
class _QuestionScopedObservation:
    """Carry only question-required media into the provider boundary."""

    images: list[str]
    image_roles: list[str]


@dataclass(frozen=True)
class _QuestionScopedTeacherSource:
    """Expose the observation shape expected by VQA teacher backends."""

    observation: _QuestionScopedObservation


class LVLMWeakLabelProducerV2:
    """Adapt the visible-input teacher to the async producer protocol."""

    producer_id = LVLM_VQA_PRODUCER_ID
    producer_version = LVLM_VQA_PRODUCER_VERSION
    supported_task_ids = tuple(VQA_V2_TEACHER_PROFILES)

    def __init__(
        self,
        backend: VQATeacherBackendV2,
        *,
        config: VQABatchConfigV2 | None = None,
    ):
        """Initialize one provider-bound weak-label producer.

        Args:
            backend: Visible-only LVLM backend.
            config: Retry and image policy.
        """

        self._backend = backend
        self._config = config or VQABatchConfigV2()

    async def produce(
        self,
        source: VQASourceV2,
        question: VQAQuestionV2,
        capabilities: VQACapabilityConfigV2,
    ) -> VQAProducerResultV2:
        """Produce an accepted weak label or explicit unavailable/rejected.

        Args:
            source: Frozen visible/private-partitioned source.
            question: Answer-independent active question.
            capabilities: Versioned task activation policy.

        Returns:
            Producer result without placeholder annotations.
        """

        if question.task_id not in self.supported_task_ids:
            return VQAProducerResultV2(
                status="unavailable",
                reason="producer_does_not_support_task",
            )
        eligibility = vqa_task_eligibility_v2(
            source,
            question.task_id,
            capabilities,
        )
        if not eligibility.eligible:
            return VQAProducerResultV2(
                status="unavailable",
                reason=eligibility.reason,
                errors=list(eligibility.missing_requirements),
            )
        annotation, errors = await annotate_vqa_question_v2(
            source,
            question,
            backend=self._backend,
            config=self._config,
        )
        if annotation is None:
            return VQAProducerResultV2(
                status="rejected",
                reason="lvlm_teacher_validation_failed",
                errors=errors,
            )
        return VQAProducerResultV2(
            status="produced",
            annotation=annotation,
        )


async def annotate_vqa_question_v2(
    source: VQASourceV2,
    question: VQAQuestionV2,
    *,
    backend: VQATeacherBackendV2,
    config: VQABatchConfigV2 | None = None,
) -> tuple[VQAAnnotationV2 | None, list[str]]:
    """Generate and validate one weak-label annotation.

    Args:
        source: Frozen VQA source.
        question: Active LVLM question.
        backend: Visible-input provider boundary.
        config: Retry and image policy.

    Returns:
        Accepted annotation or errors. No fallback label is generated.

    Raises:
        ValueError: If the task has no weak-label profile.
    """

    cfg = config or VQABatchConfigV2()
    try:
        profile = VQA_V2_TEACHER_PROFILES[question.task_id]
    except KeyError as exc:
        raise ValueError(
            f"task has no weak-label producer: {question.task_id}"
        ) from exc
    raw_answer = ""
    errors: list[str] = []
    generation: TeacherGeneration | None = None
    attempts: list[dict[str, Any]] = []
    for attempt in range(max(0, cfg.max_retries) + 1):
        provider_question = build_vqa_teacher_question_v2(
            source,
            question,
            profile,
            previous_answer=raw_answer if attempt else None,
            previous_errors=errors if attempt else None,
        )
        try:
            generation = await backend.generate(
                _question_scoped_teacher_source(source, question),
                question=provider_question,
                system_prompt=profile.system_prompt,
                image_detail=cfg.image_detail,
            )
        except Exception as exc:
            errors = [
                "teacher_call_failed:"
                f"{type(exc).__name__}:{str(exc)[:500]}"
            ]
            attempts.append(
                {"attempt": attempt + 1, "errors": list(errors)}
            )
            continue
        raw_answer = generation.answer
        answer, errors = parse_vqa_teacher_answer_v2(
            raw_answer,
            question=question,
        )
        if answer is None:
            attempts.append(
                {"attempt": attempt + 1, "errors": list(errors)}
            )
            continue
        context_digest = annotation_context_digest_v2(
            source=source,
            question=question,
            producer_id=LVLM_VQA_PRODUCER_ID,
            producer_version=LVLM_VQA_PRODUCER_VERSION,
        )
        annotation = VQAAnnotationV2(
            annotation_id=_annotation_id(
                source,
                question,
                profile,
            ),
            source_group_id=source.observation.source_group_id,
            source_fingerprint=source.source_fingerprint,
            question=question,
            answer=answer,
            label_source="lvlm_weak",
            source_kind=_weak_source_kind(question.task_id),
            evidence_refs=list(question.evidence_refs),
            source_field_refs=[],
            producer_id=LVLM_VQA_PRODUCER_ID,
            producer_version=LVLM_VQA_PRODUCER_VERSION,
            input_fingerprint=source.source_fingerprint,
            annotation_context_digest=context_digest,
            prompt_profile_id=profile.profile_id,
            prompt_policy_digest=profile.policy_digest,
            teacher_provider=generation.provider,
            teacher_model=(
                generation.model or "unknown_teacher_model"
            ),
            metadata={
                "attempts": attempt + 1,
                "repair_history": attempts,
                "raw_answer": raw_answer[
                    : cfg.raw_answer_max_chars
                ],
                "telemetry": generation.telemetry or {},
                "teacher_private_context_used": False,
            },
        )
        return annotation, []
    return None, errors or ["teacher_validation_failed"]


def _question_scoped_teacher_source(
    source: VQASourceV2,
    question: VQAQuestionV2,
) -> _QuestionScopedTeacherSource:
    """Drop all images not declared as necessary question evidence."""

    allowed = set(question.evidence_refs)
    pairs = [
        (image, role)
        for image, role in zip(
            source.observation.images,
            source.observation.image_roles,
        )
        if role in allowed
    ]
    if not pairs:
        raise ValueError(
            "weak VQA question has no attached image evidence"
        )
    return _QuestionScopedTeacherSource(
        observation=_QuestionScopedObservation(
            images=[image for image, _role in pairs],
            image_roles=[role for _image, role in pairs],
        )
    )


async def label_vqa_source_manifest_v2(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    capabilities: VQACapabilityConfigV2,
    report_json: str | Path | None = None,
    config: VQABatchConfigV2 | None = None,
    limit: int | None = None,
    api_keys: Iterable[str] = (),
    backend_factory: (
        Callable[[int, str | None], VQATeacherBackendV2] | None
    ) = None,
) -> dict[str, Any]:
    """Label active deterministic and weak VQA tasks with resume support.

    Args:
        source_jsonl: Frozen VQA source rows.
        annotation_jsonl: Append-only accepted annotations.
        capabilities: Versioned task activation policy.
        report_json: Optional durable report path.
        config: Retry/concurrency policy.
        limit: Optional source-row limit.
        api_keys: Optional in-memory worker key rotation.
        backend_factory: Optional fake/custom backend for tests.

    Returns:
        Final counts, bounded errors, and capability digest.
    """

    cfg = config or VQABatchConfigV2()
    sources = [
        VQASourceV2.model_validate(row)
        for row in iter_jsonl(source_jsonl)
    ]
    if limit is not None:
        sources = sources[: int(limit)]
    completed = _existing_annotation_ids(annotation_jsonl)
    keys = [str(value) for value in api_keys if str(value)]
    backend_count = max(1, min(cfg.concurrency, len(sources) or 1))
    backends = [
        (
            backend_factory(
                index,
                keys[index % len(keys)] if keys else None,
            )
            if backend_factory is not None
            else VisionVQATeacherBackendV2(
                api_key=(
                    keys[index % len(keys)] if keys else None
                )
            )
        )
        for index in range(backend_count)
    ]
    output_path = Path(annotation_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, cfg.concurrency))
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    direction_producer = CanonicalDemonstratedDirectionProducerV2()
    weak_producers = [
        LVLMWeakLabelProducerV2(backend, config=cfg)
        for backend in backends
    ]

    async def process(
        source_index: int,
        source: VQASourceV2,
    ) -> list[VQAAnnotationV2]:
        """Annotate all eligible tasks for one source row."""

        async with semaphore:
            build = build_vqa_questions_v2(source, capabilities)
            annotations: list[VQAAnnotationV2] = []
            for task_id, unavailable in build.unavailable.items():
                counts[f"unavailable:{task_id}:{unavailable.reason}"] += 1
            for question in build.questions:
                provisional_id = _provisional_annotation_id(
                    source,
                    question,
                )
                if provisional_id in completed:
                    counts["resumed_annotations"] += 1
                    continue
                if question.task_id == (
                    "e0_demonstrated_direction_selection"
                ):
                    result = direction_producer.produce(
                        source,
                        question,
                        capabilities,
                    )
                    if result.annotation is not None:
                        annotations.append(result.annotation)
                    else:
                        errors.append(
                            {
                                "source_group_id": (
                                    source.observation.source_group_id
                                ),
                                "task_id": question.task_id,
                                "errors": result.errors
                                or [str(result.reason)],
                            }
                        )
                    continue
                weak_producer = weak_producers[
                    source_index % len(weak_producers)
                ]
                result = await weak_producer.produce(
                    source,
                    question,
                    capabilities,
                )
                if result.annotation is None:
                    errors.append(
                        {
                            "source_group_id": (
                                source.observation.source_group_id
                            ),
                            "task_id": question.task_id,
                            "errors": result.errors
                            or [str(result.reason)],
                        }
                    )
                else:
                    annotations.append(result.annotation)
            return annotations

    with output_path.open("a", encoding="utf-8") as handle:
        tasks = [
            asyncio.create_task(process(index, source))
            for index, source in enumerate(sources)
        ]
        for future in asyncio.as_completed(tasks):
            annotations = await future
            async with lock:
                for annotation in annotations:
                    if annotation.annotation_id in completed:
                        continue
                    append_jsonl_row(
                        handle,
                        annotation.model_dump(
                            mode="json",
                            exclude_none=True,
                        ),
                    )
                    completed.add(annotation.annotation_id)
                    counts["accepted_annotations"] += 1
                    counts[
                        f"task:{annotation.question.task_id}"
                    ] += 1
                    counts[
                        f"label_source:{annotation.label_source}"
                    ] += 1
                counts["completed_sources"] += 1
                if (
                    cfg.progress_interval > 0
                    and counts["completed_sources"]
                    % cfg.progress_interval
                    == 0
                ):
                    _write_report(
                        report_json,
                        capabilities,
                        counts,
                        errors,
                    )
    return _write_report(
        report_json,
        capabilities,
        counts,
        errors,
    )


def _weak_source_kind(task_id: str) -> str:
    """Return an honest weak-label provenance code."""

    if task_id.startswith("a"):
        return "lvlm_task_grounding_weak"
    if task_id.startswith("b"):
        return "lvlm_direction_affordance_weak"
    if task_id.startswith("c"):
        return "lvlm_goal_direction_weak"
    if task_id.startswith("d"):
        return "lvlm_social_direction_weak"
    return "lvlm_alternative_direction_weak"


def _annotation_id(
    source: VQASourceV2,
    question: VQAQuestionV2,
    profile: VQATeacherProfileV2,
) -> str:
    """Return stable weak annotation identity."""

    digest = hashlib.sha256(
        (
            source.source_fingerprint
            + ":"
            + question.task_id
            + ":"
            + profile.policy_digest
        ).encode("utf-8")
    ).hexdigest()[:20]
    return (
        f"{source.observation.source_group_id}:vqa_v2:"
        f"{question.task_id}:{digest}"
    )


def _provisional_annotation_id(
    source: VQASourceV2,
    question: VQAQuestionV2,
) -> str:
    """Return the id expected from the active producer."""

    if question.task_id == "e0_demonstrated_direction_selection":
        digest = hashlib.sha256(
            (
                source.source_fingerprint
                + ":"
                + question.task_id
                + ":"
                + question.template_id
                + ":canonical_demonstrated_direction_v2"
            ).encode("utf-8")
        ).hexdigest()[:20]
        return (
            f"{source.observation.source_group_id}:vqa_v2:"
            f"{question.task_id}:{digest}"
        )
    return _annotation_id(
        source,
        question,
        VQA_V2_TEACHER_PROFILES[question.task_id],
    )


def _existing_annotation_ids(path: str | Path) -> set[str]:
    """Load accepted annotation ids for append-only resume."""

    input_path = Path(path).expanduser()
    if not input_path.is_file():
        return set()
    return {
        VQAAnnotationV2.model_validate(row).annotation_id
        for row in iter_jsonl(input_path)
    }


def _write_report(
    path: str | Path | None,
    capabilities: VQACapabilityConfigV2,
    counts: Counter[str],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build and optionally persist one annotation report."""

    report = {
        "schema_version": "urbannav_navigation_vqa_label_report_v2",
        "generated_at_unix": time.time(),
        "capability_config_digest": capabilities.digest(),
        "counts": dict(counts),
        "errors": errors[:200],
        "error_count": len(errors),
    }
    if path is not None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return report


__all__ = [
    "LVLMWeakLabelProducerV2",
    "VQABatchConfigV2",
    "VQATeacherBackendV2",
    "VisionVQATeacherBackendV2",
    "annotate_vqa_question_v2",
    "label_vqa_source_manifest_v2",
]
