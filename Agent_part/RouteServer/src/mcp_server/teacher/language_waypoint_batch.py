"""Resumable LVLM runner for visible-FPV language-goal annotations."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from mcp_server.contracts.language_waypoint_benchmark import (
    LanguageGoalAnnotationV1,
    LanguageGoalCandidateV1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.data.io.jsonl import append_jsonl_row, iter_jsonl
from mcp_server.teacher.language_waypoint_teacher import (
    LanguageGoalTeacherProfileV1,
    build_language_goal_teacher_question_v1,
    language_goal_teacher_profile_v1,
    parse_language_goal_teacher_answer_v1,
)
from mcp_server.teacher.reasoning_v2_teacher import (
    TeacherGeneration,
    VisionReasoningTeacherBackend,
)


@dataclass(frozen=True)
class LanguageGoalAnnotationConfigV1:
    """Configure a bounded visible-FPV language annotation run.

    Args:
        max_retries: Repair attempts after the first teacher response.
        image_detail: OpenAI-compatible current-FPV detail setting.
        concurrency: Maximum in-flight candidate annotations.
        retry_nonaccepted_on_resume: Retry rejected/pending rows on resume.
        progress_interval: Completed candidate reporting interval.
        raw_answer_max_chars: Maximum provider answer stored in decisions.
    """

    max_retries: int = 1
    image_detail: str = "high"
    concurrency: int = 1
    retry_nonaccepted_on_resume: bool = False
    progress_interval: int = 25
    raw_answer_max_chars: int = 12000


class LanguageGoalTeacherBackendV1(Protocol):
    """Provider boundary that receives only a current FPV image."""

    async def generate(
        self,
        source: Any,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Generate one raw annotation response."""


class VisionLanguageGoalTeacherBackendV1:
    """Adapt the shared OpenAI-compatible vision teacher to one FPV image."""

    def __init__(self, *, api_key: str | None = None):
        """Initialize the provider adapter.

        Args:
            api_key: Optional in-memory key; never written to artifacts.
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
        """Call the provider with only ``fpv.current``.

        Args:
            source: Minimal source exposing one current image.
            question: User-side schema contract.
            system_prompt: Frozen visible-evidence policy.
            image_detail: Provider detail policy.

        Returns:
            Raw teacher generation and sanitized telemetry.
        """

        return await self._delegate.generate(
            source,
            question=question,
            system_prompt=system_prompt,
            image_detail=image_detail,
        )


@dataclass(frozen=True)
class _VisibleCurrentFPVSourceV1:
    """Expose one current FPV path in the shared vision-backend shape."""

    images: list[str]
    image_roles: list[str]


async def annotate_language_goal_candidate_v1(
    source: ReasoningSourceV4,
    candidate: LanguageGoalCandidateV1,
    *,
    backend: LanguageGoalTeacherBackendV1,
    config: LanguageGoalAnnotationConfigV1 | None = None,
    profile: LanguageGoalTeacherProfileV1 | None = None,
) -> tuple[LanguageGoalAnnotationV1 | None, dict[str, Any]]:
    """Generate and validate one language-goal annotation.

    Args:
        source: Frozen source row containing the anchor FPV frame.
        candidate: Teacher-only future target candidate for this source.
        backend: Visible-only LVLM provider adapter.
        config: Bounded retry and image policies.
        profile: Optional frozen prompt profile.

    Returns:
        Typed annotation when parsing succeeds plus a decision/audit payload.
    """

    cfg = config or LanguageGoalAnnotationConfigV1()
    effective_profile = profile or language_goal_teacher_profile_v1()
    current = _current_fpv_source_v1(source)
    raw_answer = ""
    errors: list[str] = []
    attempts: list[dict[str, Any]] = []
    for attempt in range(max(0, cfg.max_retries) + 1):
        question = build_language_goal_teacher_question_v1(
            source,
            candidate,
            effective_profile,
            previous_answer=raw_answer if attempt else None,
            previous_errors=errors if attempt else (),
        )
        try:
            generation = await backend.generate(
                current,
                question=question,
                system_prompt=effective_profile.system_prompt,
                image_detail=cfg.image_detail,
            )
        except Exception as exc:
            errors = [f"teacher_call_failed:{type(exc).__name__}:{str(exc)[:500]}"]
            attempts.append({"attempt": attempt + 1, "errors": list(errors)})
            continue
        raw_answer = generation.answer
        annotation, errors = parse_language_goal_teacher_answer_v1(
            raw_answer,
            source=source,
            candidate=candidate,
            teacher_model=generation.model,
            profile=effective_profile,
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "errors": list(errors),
                "telemetry": generation.telemetry or {},
            }
        )
        if annotation is not None:
            return annotation, {
                "parent_sample_id": candidate.parent_sample_id,
                "candidate_id": candidate.candidate_id,
                "status": annotation.validator_status,
                "attempts": attempts,
                "raw_answer": raw_answer[: cfg.raw_answer_max_chars],
                "profile_id": effective_profile.profile_id,
                "profile_digest": effective_profile.policy_digest,
                "source_fingerprint": source.source_fingerprint,
            }
    return None, {
        "parent_sample_id": candidate.parent_sample_id,
        "candidate_id": candidate.candidate_id,
        "status": "error",
        "attempts": attempts,
        "errors": list(errors) or ["teacher_validation_failed"],
        "raw_answer": raw_answer[: cfg.raw_answer_max_chars],
        "profile_id": effective_profile.profile_id,
        "profile_digest": effective_profile.policy_digest,
        "source_fingerprint": source.source_fingerprint,
    }


async def label_language_goal_candidates_v1(
    *,
    source_jsonl: str | Path,
    candidates_jsonl: str | Path,
    annotation_jsonl: str | Path,
    decisions_jsonl: str | Path,
    report_json: str | Path | None = None,
    config: LanguageGoalAnnotationConfigV1 | None = None,
    limit: int | None = None,
    api_keys: Iterable[str] = (),
    backend_factory: Callable[[int, str | None], LanguageGoalTeacherBackendV1] | None = None,
) -> dict[str, Any]:
    """Run a resumable visible-FPV annotation batch.

    Args:
        source_jsonl: Canonical Reasoning V4 sources.
        candidates_jsonl: Teacher-only language-goal candidates.
        annotation_jsonl: Append-only typed annotation output.
        decisions_jsonl: Append-only provider/validator audit output.
        report_json: Optional durable aggregate report.
        config: Bounded retry/concurrency policy.
        limit: Optional candidate limit after deterministic file order.
        api_keys: Optional in-memory key rotation for external providers.
        backend_factory: Optional fake/custom backend for tests.

    Returns:
        Counts, prompt identity, and bounded validation diagnostics.
    """

    cfg = config or LanguageGoalAnnotationConfigV1()
    profile = language_goal_teacher_profile_v1()
    sources = _load_sources_v1(source_jsonl)
    candidates = [LanguageGoalCandidateV1.model_validate(row) for row in iter_jsonl(candidates_jsonl)]
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]
    completed = _completed_parent_statuses_v1(annotation_jsonl)
    keys = [str(value) for value in api_keys if str(value)]
    backend_count = max(1, min(cfg.concurrency, len(candidates) or 1))
    backends = [
        (
            backend_factory(index, keys[index % len(keys)] if keys else None)
            if backend_factory is not None
            else VisionLanguageGoalTeacherBackendV1(
                api_key=keys[index % len(keys)] if keys else None
            )
        )
        for index in range(backend_count)
    ]
    annotation_path = Path(annotation_jsonl).expanduser()
    decisions_path = Path(decisions_jsonl).expanduser()
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, cfg.concurrency))
    write_lock = asyncio.Lock()
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()

    async def process(index: int, candidate: LanguageGoalCandidateV1) -> None:
        """Annotate one candidate and durably append its outcome."""

        prior_status = completed.get(candidate.parent_sample_id)
        if prior_status is not None and (
            prior_status == "accepted" or not cfg.retry_nonaccepted_on_resume
        ):
            counts["resumed"] += 1
            return
        source = sources.get(candidate.parent_sample_id)
        if source is None:
            counts["source_missing"] += 1
            errors.append({"parent_sample_id": candidate.parent_sample_id, "error": "source_missing"})
            return
        async with semaphore:
            annotation, decision = await annotate_language_goal_candidate_v1(
                source,
                candidate,
                backend=backends[index % len(backends)],
                config=cfg,
                profile=profile,
            )
        async with write_lock:
            with decisions_path.open("a", encoding="utf-8") as decision_handle:
                append_jsonl_row(decision_handle, decision)
            if annotation is not None:
                with annotation_path.open("a", encoding="utf-8") as annotation_handle:
                    append_jsonl_row(annotation_handle, annotation.model_dump(mode="json"))
                counts[annotation.validator_status] += 1
            else:
                counts["error"] += 1
                errors.append(
                    {
                        "parent_sample_id": candidate.parent_sample_id,
                        "candidate_id": candidate.candidate_id,
                        "errors": decision.get("errors", []),
                    }
                )
            completed_count = sum(counts.values())
            if cfg.progress_interval and completed_count % cfg.progress_interval == 0:
                elapsed_s = max(1e-6, time.perf_counter() - started)
                print(
                    json.dumps(
                        {
                            "language_goal_progress": f"{completed_count}/{len(candidates)}",
                            "accepted": counts["accepted"],
                            "rejected": counts["rejected"],
                            "pending_human": counts["pending_human"],
                            "error": counts["error"],
                            "rows_per_hour": round(completed_count * 3600.0 / elapsed_s, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    await asyncio.gather(*(process(index, candidate) for index, candidate in enumerate(candidates)))
    report = {
        "schema_version": "urbannav_language_goal_annotation_report_v1",
        "generated_at_unix": time.time(),
        "candidate_count": len(candidates),
        "source_count": len(sources),
        "counts": dict(sorted(counts.items())),
        "error_count": len(errors),
        "errors": errors[:200],
        "profile_id": profile.profile_id,
        "profile_digest": profile.policy_digest,
        "producer_version": profile.producer_version,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    if report_json is not None:
        target = Path(report_json).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _current_fpv_source_v1(source: ReasoningSourceV4) -> _VisibleCurrentFPVSourceV1:
    """Select exactly the current FPV image for LVLM annotation."""

    for image, role in zip(source.observation.images, source.observation.image_roles):
        if role == "fpv.current":
            return _VisibleCurrentFPVSourceV1(images=[image], image_roles=[role])
    raise ValueError("source has no fpv.current image")


def _load_sources_v1(path: str | Path) -> dict[str, ReasoningSourceV4]:
    """Load source rows keyed by their stable physical parent id."""

    rows: dict[str, ReasoningSourceV4] = {}
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        parent_id = source.observation.source_group_id
        if parent_id in rows:
            raise ValueError(f"duplicate source_group_id: {parent_id}")
        rows[parent_id] = source
    return rows


def _completed_parent_statuses_v1(path: str | Path) -> dict[str, str]:
    """Load the latest status for each parent to support safe resume."""

    input_path = Path(path).expanduser()
    if not input_path.is_file():
        return {}
    statuses: dict[str, str] = {}
    for payload in iter_jsonl(input_path):
        annotation = LanguageGoalAnnotationV1.model_validate(payload)
        statuses[annotation.parent_sample_id] = annotation.validator_status
    return statuses


__all__ = [
    "LanguageGoalAnnotationConfigV1",
    "LanguageGoalTeacherBackendV1",
    "VisionLanguageGoalTeacherBackendV1",
    "annotate_language_goal_candidate_v1",
    "label_language_goal_candidates_v1",
]
