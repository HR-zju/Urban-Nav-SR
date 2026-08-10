"""Resumable direct-LVLM geometry annotation and dataset-suitability filtering."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from mcp_server.contracts.local_traversability_geometry import LocalTraversabilityGeometryV1
from mcp_server.contracts.navigation_trainability import (
    MotionEpisodeSummaryV1,
    NavigationTrainabilityFilterV1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.contracts.stage_ii_cfm import StageIITrajectorySidecarV2
from mcp_server.data.io.jsonl import append_jsonl_row, iter_jsonl
from mcp_server.data.quality.trainability import (
    NavigationTrainabilityPolicyV1,
    derive_navigation_trainability_v1,
    summarize_reasoning_source_motion_v1,
    summarize_stage_ii_sidecar_motion_v1,
)
from mcp_server.teacher.navigation_trainability_teacher import (
    NavigationTrainabilityTeacherProfileV1,
    build_motion_scene_question_v1,
    build_scene_suitability_question_v1,
    motion_scene_teacher_profile_v1,
    parse_motion_scene_answer_v1,
    parse_scene_suitability_answer_v1,
    scene_suitability_teacher_profile_v1,
)
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration, VisionReasoningTeacherBackend
from mcp_server.teacher.traversability_geometry_teacher import (
    TraversabilityGeometryTeacherProfileV1,
    build_traversability_geometry_question_v1,
    parse_traversability_geometry_answer_v1,
    traversability_geometry_teacher_profile_v1,
)


@dataclass(frozen=True)
class NavigationQualityBatchConfigV1:
    """Configure bounded, append-only direct-LVLM quality processing.

    Args:
        max_retries: Repair attempts after each initial provider response.
        image_detail: OpenAI-compatible image detail policy.
        concurrency: Maximum in-flight source rows.
        max_temporal_fpv_frames: Maximum ordered FPV frames for consistency.
        retry_nonaccepted_on_resume: Reprocess geometry rejected/pending rows.
        retry_pending_on_resume: Reprocess filter pending-retry rows.
        progress_interval: Completed-row console report interval.
        raw_answer_max_chars: Provider answer truncation in audit ledgers.
    """

    max_retries: int = 1
    image_detail: str = "high"
    concurrency: int = 1
    max_temporal_fpv_frames: int = 5
    retry_nonaccepted_on_resume: bool = False
    retry_pending_on_resume: bool = False
    progress_interval: int = 25
    raw_answer_max_chars: int = 12000


class NavigationQualityTeacherBackendV1(Protocol):
    """Provider boundary receiving only explicitly selected images."""

    async def generate(
        self,
        source: Any,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Generate one raw JSON response for a profile-visible source."""


class VisionNavigationQualityTeacherBackendV1:
    """Reuse the shared OpenAI-compatible multimodal provider adapter."""

    def __init__(self, *, api_key: str | None = None):
        """Initialize a worker-local provider adapter.

        Args:
            api_key: Optional in-memory key override; never serialized.
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
        """Call the shared provider with the preselected image sequence."""

        return await self._delegate.generate(
            source,
            question=question,
            system_prompt=system_prompt,
            image_detail=image_detail,
        )


@dataclass(frozen=True)
class _VisibleImageSourceV1:
    """Minimal source shape required by the shared vision provider backend."""

    images: list[str]
    image_roles: list[str]


async def annotate_traversability_geometry_source_v1(
    source: ReasoningSourceV4,
    *,
    backend: NavigationQualityTeacherBackendV1,
    config: NavigationQualityBatchConfigV1 | None = None,
    profile: TraversabilityGeometryTeacherProfileV1 | None = None,
) -> tuple[LocalTraversabilityGeometryV1 | None, dict[str, Any]]:
    """Directly annotate one current FPV as traversability quadrilaterals.

    Args:
        source: Frozen canonical source row.
        backend: Worker-local direct-LVLM provider adapter.
        config: Retry and image-detail policy.
        profile: Optional frozen geometry profile.

    Returns:
        Parsed geometry sidecar or ``None`` with append-only audit data.
    """

    cfg = config or NavigationQualityBatchConfigV1()
    effective_profile = profile or traversability_geometry_teacher_profile_v1()
    current = _current_fpv_source_v1(source)
    raw_answer = ""
    errors: list[str] = []
    attempts: list[dict[str, Any]] = []
    for attempt in range(max(0, cfg.max_retries) + 1):
        question = build_traversability_geometry_question_v1(
            source,
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
        annotation, errors = parse_traversability_geometry_answer_v1(
            raw_answer,
            source=source,
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
            return annotation, _geometry_decision_v1(
                source=source,
                annotation=annotation,
                attempts=attempts,
                raw_answer=raw_answer,
                raw_answer_max_chars=cfg.raw_answer_max_chars,
            )
    return None, {
        "kind": "traversability_geometry",
        "parent_sample_id": source.observation.source_group_id,
        "source_fingerprint": source.source_fingerprint,
        "status": "error",
        "errors": list(errors) or ["teacher_validation_failed"],
        "attempts": attempts,
        "raw_answer": raw_answer[: cfg.raw_answer_max_chars],
        "profile_id": effective_profile.profile_id,
        "profile_digest": effective_profile.policy_digest,
    }


async def filter_navigation_trainability_source_v1(
    source: ReasoningSourceV4,
    *,
    backend: NavigationQualityTeacherBackendV1,
    config: NavigationQualityBatchConfigV1 | None = None,
    scene_profile: NavigationTrainabilityTeacherProfileV1 | None = None,
    motion_profile: NavigationTrainabilityTeacherProfileV1 | None = None,
    policy: NavigationTrainabilityPolicyV1 | None = None,
    motion_summary: MotionEpisodeSummaryV1 | None = None,
) -> tuple[NavigationTrainabilityFilterV1, dict[str, Any]]:
    """Filter one source with separate scene and motion-consistency LVLM passes.

    Args:
        source: Frozen canonical source row.
        backend: Worker-local multimodal provider adapter.
        config: Bounded retry and image-sequence policy.
        scene_profile: Optional frozen single-FPV suitability profile.
        motion_profile: Optional frozen temporal consistency profile.
        policy: Deterministic downstream eligibility projection policy.
        motion_summary: Optional external-sidecar motion summary.  When
            omitted, only trajectory fields already present in the V4 source
            are used.

    Returns:
        Typed post-annotation filter row and provider/validator audit payload.
    """

    cfg = config or NavigationQualityBatchConfigV1()
    effective_scene = scene_profile or scene_suitability_teacher_profile_v1()
    effective_motion = motion_profile or motion_scene_teacher_profile_v1()
    effective_policy = policy or NavigationTrainabilityPolicyV1()
    effective_motion_summary = motion_summary or summarize_reasoning_source_motion_v1(
        source,
        policy=effective_policy,
    )
    current = _current_fpv_source_v1(source)
    scene_value, scene_audit = await _run_profile_v1(
        backend=backend,
        visible_source=current,
        question_builder=lambda previous_answer, previous_errors: build_scene_suitability_question_v1(
            source,
            effective_scene,
            previous_answer=previous_answer,
            previous_errors=previous_errors,
        ),
        parser=lambda answer, model: parse_scene_suitability_answer_v1(
            answer,
            source=source,
            teacher_model=model,
            profile=effective_scene,
        ),
        system_prompt=effective_scene.system_prompt,
        config=cfg,
    )
    if scene_value is None:
        result = derive_navigation_trainability_v1(
            source=source,
            motion_summary=effective_motion_summary,
            scene_assessment=None,
            motion_scene_assessment=None,
            scene_profile_digest=effective_scene.policy_digest,
            motion_profile_digest=effective_motion.policy_digest,
            policy=effective_policy,
            retry_required=True,
        )
        return result, _filter_decision_v1(
            source=source,
            result=result,
            scene_audit=scene_audit,
            motion_audit=None,
            motion_summary=effective_motion_summary,
        )

    temporal = _temporal_fpv_source_v1(source, max_frames=cfg.max_temporal_fpv_frames)
    motion_value = None
    motion_audit: dict[str, Any] | None = None
    if effective_motion_summary.available:
        motion_value, motion_audit = await _run_profile_v1(
            backend=backend,
            visible_source=temporal,
            question_builder=lambda previous_answer, previous_errors: build_motion_scene_question_v1(
                source,
                effective_motion_summary,
                effective_motion,
                temporal_image_roles=tuple(temporal.image_roles),
                previous_answer=previous_answer,
                previous_errors=previous_errors,
            ),
            parser=lambda answer, model: parse_motion_scene_answer_v1(
                answer,
                source=source,
                teacher_model=model,
                profile=effective_motion,
                temporal_image_roles=tuple(temporal.image_roles),
            ),
            system_prompt=effective_motion.system_prompt,
            config=cfg,
        )
    result = derive_navigation_trainability_v1(
        source=source,
        motion_summary=effective_motion_summary,
        scene_assessment=scene_value,
        motion_scene_assessment=motion_value,
        scene_profile_digest=effective_scene.policy_digest,
        motion_profile_digest=effective_motion.policy_digest,
        policy=effective_policy,
        retry_required=effective_motion_summary.available and motion_value is None,
    )
    return result, _filter_decision_v1(
        source=source,
        result=result,
        scene_audit=scene_audit,
        motion_audit=motion_audit,
        motion_summary=effective_motion_summary,
    )


async def label_traversability_geometry_sources_v1(
    *,
    source_jsonl: str | Path,
    geometry_jsonl: str | Path,
    decisions_jsonl: str | Path,
    report_json: str | Path | None = None,
    config: NavigationQualityBatchConfigV1 | None = None,
    limit: int | None = None,
    api_keys: Iterable[str] = (),
    backend_factory: Callable[[int, str | None], NavigationQualityTeacherBackendV1] | None = None,
) -> dict[str, Any]:
    """Run a resumable direct-LVLM quadrilateral annotation batch.

    Args:
        source_jsonl: Canonical Reasoning V4 source manifest.
        geometry_jsonl: Append-only typed geometry output ledger.
        decisions_jsonl: Append-only raw provider and validator audit ledger.
        report_json: Optional durable aggregate report path.
        config: Bounded worker and retry settings.
        limit: Optional deterministic maximum number of source rows.
        api_keys: Optional in-memory worker key rotation.
        backend_factory: Optional fake/custom backend for focused tests.

    Returns:
        Machine-readable run report.
    """

    cfg = config or NavigationQualityBatchConfigV1()
    profile = traversability_geometry_teacher_profile_v1()
    sources = _load_sources_v1(source_jsonl, limit=limit)
    completed = _latest_geometry_v1(geometry_jsonl)

    def should_run(source: ReasoningSourceV4) -> bool:
        """Return whether source cache misses this exact geometry profile."""

        existing = completed.get((source.observation.source_group_id, source.source_fingerprint))
        return bool(
            existing is None
            or existing.prompt_profile_digest != profile.policy_digest
            or (
                cfg.retry_nonaccepted_on_resume
                and existing.validator_status in {"rejected", "pending_human"}
            )
        )

    pending = [source for source in sources if should_run(source)]
    return await _run_geometry_batch_v1(
        sources=sources,
        pending=pending,
        geometry_jsonl=geometry_jsonl,
        decisions_jsonl=decisions_jsonl,
        report_json=report_json,
        config=cfg,
        profile=profile,
        api_keys=api_keys,
        backend_factory=backend_factory,
    )


async def filter_navigation_trainability_sources_v1(
    *,
    source_jsonl: str | Path,
    filter_jsonl: str | Path,
    decisions_jsonl: str | Path,
    report_json: str | Path | None = None,
    config: NavigationQualityBatchConfigV1 | None = None,
    policy: NavigationTrainabilityPolicyV1 | None = None,
    limit: int | None = None,
    trajectory_sidecar_jsonl: str | Path | None = None,
    api_keys: Iterable[str] = (),
    backend_factory: Callable[[int, str | None], NavigationQualityTeacherBackendV1] | None = None,
) -> dict[str, Any]:
    """Run a resumable post-annotation navigation suitability filtering batch.

    Args:
        source_jsonl: Canonical Reasoning V4 source manifest.
        filter_jsonl: Append-only typed suitability decision ledger.
        decisions_jsonl: Append-only provider and validation audit ledger.
        report_json: Optional durable aggregate report path.
        config: Bounded worker and retry settings.
        policy: Deterministic eligibility projection policy.
        limit: Optional deterministic maximum number of source rows.
        trajectory_sidecar_jsonl: Optional accepted Stage-II V2 sidecar ledger
            used only for teacher-side motion QC.
        api_keys: Optional in-memory worker key rotation.
        backend_factory: Optional fake/custom backend for focused tests.

    Returns:
        Machine-readable run report.
    """

    cfg = config or NavigationQualityBatchConfigV1()
    effective_policy = policy or NavigationTrainabilityPolicyV1()
    scene_profile = scene_suitability_teacher_profile_v1()
    motion_profile = motion_scene_teacher_profile_v1()
    sources = _load_sources_v1(source_jsonl, limit=limit)
    motion_sidecars = _load_stage_ii_sidecars_v2(trajectory_sidecar_jsonl)
    motion_summaries = {
        source.observation.source_group_id: _motion_summary_for_source_v1(
            source,
            sidecar=motion_sidecars.get(source.observation.source_group_id),
            policy=effective_policy,
        )
        for source in sources
    }
    completed = _latest_trainability_filter_v1(filter_jsonl)

    def should_run(source: ReasoningSourceV4) -> bool:
        """Return whether source cache misses either independent profile."""

        existing = completed.get((source.observation.source_group_id, source.source_fingerprint))
        expected_motion = motion_summaries[source.observation.source_group_id]
        if existing is None:
            return True
        if existing.status == "pending_retry":
            return cfg.retry_pending_on_resume
        scene = existing.scene_assessment
        motion = existing.motion_scene_assessment
        if scene is None or scene.prompt_profile_digest != scene_profile.policy_digest:
            return True
        if existing.motion_summary.available and (
            motion is None or motion.prompt_profile_digest != motion_profile.policy_digest
        ):
            return True
        if existing.motion_summary.model_dump(mode="json") != expected_motion.model_dump(mode="json"):
            return True
        return False

    pending = [source for source in sources if should_run(source)]
    return await _run_trainability_batch_v1(
        sources=sources,
        pending=pending,
        filter_jsonl=filter_jsonl,
        decisions_jsonl=decisions_jsonl,
        report_json=report_json,
        config=cfg,
        policy=effective_policy,
        scene_profile=scene_profile,
        motion_profile=motion_profile,
        motion_summaries=motion_summaries,
        motion_sidecar_input_count=len(motion_sidecars),
        api_keys=api_keys,
        backend_factory=backend_factory,
    )


async def _run_geometry_batch_v1(
    *,
    sources: list[ReasoningSourceV4],
    pending: list[ReasoningSourceV4],
    geometry_jsonl: str | Path,
    decisions_jsonl: str | Path,
    report_json: str | Path | None,
    config: NavigationQualityBatchConfigV1,
    profile: TraversabilityGeometryTeacherProfileV1,
    api_keys: Iterable[str],
    backend_factory: Callable[[int, str | None], NavigationQualityTeacherBackendV1] | None,
) -> dict[str, Any]:
    """Execute and append one geometry queue with bounded concurrency."""

    keys = _clean_keys_v1(api_keys)
    backends = _make_backends_v1(
        count=min(max(1, config.concurrency), max(1, len(pending))),
        keys=keys,
        backend_factory=backend_factory,
    )
    geometry_path = Path(geometry_jsonl).expanduser()
    decision_path = Path(decisions_jsonl).expanduser()
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, config.concurrency))
    write_lock = asyncio.Lock()
    counts: Counter[str] = Counter({"source_rows": len(sources), "pending_rows": len(pending)})
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()

    async def process(index: int, source: ReasoningSourceV4) -> None:
        """Annotate and durably append one source result."""

        async with semaphore:
            annotation, decision = await annotate_traversability_geometry_source_v1(
                source,
                backend=backends[index % len(backends)],
                config=config,
                profile=profile,
            )
        async with write_lock:
            with decision_path.open("a", encoding="utf-8") as handle:
                append_jsonl_row(handle, decision)
            if annotation is None:
                counts["error"] += 1
                errors.append({"parent_sample_id": source.observation.source_group_id, "errors": decision.get("errors", [])})
            else:
                with geometry_path.open("a", encoding="utf-8") as handle:
                    append_jsonl_row(handle, annotation.model_dump(mode="json"))
                counts[annotation.validator_status] += 1
            _maybe_progress_v1(
                label="traversability_geometry_progress",
                completed=sum(value for key, value in counts.items() if key in {"accepted", "rejected", "pending_human", "error"}),
                total=len(pending),
                counts=counts,
                started=started,
                interval=config.progress_interval,
            )

    await asyncio.gather(*(process(index, source) for index, source in enumerate(pending)))
    report = _report_v1(
        schema_version="urbannav_local_traversability_geometry_run_report_v1",
        counts=counts,
        errors=errors,
        elapsed_s=time.perf_counter() - started,
        profiles={"traversability_geometry": {"id": profile.profile_id, "digest": profile.policy_digest}},
    )
    _write_report_v1(report_json, report)
    return report


async def _run_trainability_batch_v1(
    *,
    sources: list[ReasoningSourceV4],
    pending: list[ReasoningSourceV4],
    filter_jsonl: str | Path,
    decisions_jsonl: str | Path,
    report_json: str | Path | None,
    config: NavigationQualityBatchConfigV1,
    policy: NavigationTrainabilityPolicyV1,
    scene_profile: NavigationTrainabilityTeacherProfileV1,
    motion_profile: NavigationTrainabilityTeacherProfileV1,
    motion_summaries: Mapping[str, MotionEpisodeSummaryV1],
    motion_sidecar_input_count: int,
    api_keys: Iterable[str],
    backend_factory: Callable[[int, str | None], NavigationQualityTeacherBackendV1] | None,
) -> dict[str, Any]:
    """Execute and append one suitability queue with bounded concurrency."""

    keys = _clean_keys_v1(api_keys)
    backends = _make_backends_v1(
        count=min(max(1, config.concurrency), max(1, len(pending))),
        keys=keys,
        backend_factory=backend_factory,
    )
    filter_path = Path(filter_jsonl).expanduser()
    decision_path = Path(decisions_jsonl).expanduser()
    filter_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, config.concurrency))
    write_lock = asyncio.Lock()
    counts: Counter[str] = Counter({"source_rows": len(sources), "pending_rows": len(pending)})
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()

    async def process(index: int, source: ReasoningSourceV4) -> None:
        """Filter one source and persist its sidecar and provider audit."""

        async with semaphore:
            result, decision = await filter_navigation_trainability_source_v1(
                source,
                backend=backends[index % len(backends)],
                config=config,
                scene_profile=scene_profile,
                motion_profile=motion_profile,
                policy=policy,
                motion_summary=motion_summaries[source.observation.source_group_id],
            )
        async with write_lock:
            with decision_path.open("a", encoding="utf-8") as handle:
                append_jsonl_row(handle, decision)
            with filter_path.open("a", encoding="utf-8") as handle:
                append_jsonl_row(handle, result.model_dump(mode="json"))
            counts[f"status:{result.status}"] += 1
            counts[f"stage_i:{str(result.eligibility.stage_i_decision).lower()}"] += 1
            counts[f"stage_ii:{str(result.eligibility.stage_ii_cfm).lower()}"] += 1
            counts[f"vqa:{str(result.eligibility.vqa).lower()}"] += 1
            if result.status == "pending_retry":
                errors.append({"parent_sample_id": result.parent_sample_id, "errors": decision.get("errors", [])})
            _maybe_progress_v1(
                label="navigation_trainability_progress",
                completed=sum(value for key, value in counts.items() if key.startswith("status:")),
                total=len(pending),
                counts=counts,
                started=started,
                interval=config.progress_interval,
            )

    await asyncio.gather(*(process(index, source) for index, source in enumerate(pending)))
    report = _report_v1(
        schema_version="urbannav_navigation_trainability_run_report_v1",
        counts=counts,
        errors=errors,
        elapsed_s=time.perf_counter() - started,
        profiles={
            "scene_suitability": {"id": scene_profile.profile_id, "digest": scene_profile.policy_digest},
            "motion_scene_consistency": {"id": motion_profile.profile_id, "digest": motion_profile.policy_digest},
        },
    )
    report["motion_sidecar_input_count"] = motion_sidecar_input_count
    report["external_motion_summary_count"] = sum(
        summary.source_kind == "external_sidecar" for summary in motion_summaries.values()
    )
    _write_report_v1(report_json, report)
    return report


async def _run_profile_v1(
    *,
    backend: NavigationQualityTeacherBackendV1,
    visible_source: _VisibleImageSourceV1,
    question_builder: Callable[[str | None, Iterable[str]], str],
    parser: Callable[[str, str | None], tuple[Any | None, list[str]]],
    system_prompt: str,
    config: NavigationQualityBatchConfigV1,
) -> tuple[Any | None, dict[str, Any]]:
    """Run one independently retried profile and retain sanitized audit data."""

    raw_answer = ""
    errors: list[str] = []
    attempts: list[dict[str, Any]] = []
    for attempt in range(max(0, config.max_retries) + 1):
        question = question_builder(raw_answer if attempt else None, errors if attempt else ())
        try:
            generation = await backend.generate(
                visible_source,
                question=question,
                system_prompt=system_prompt,
                image_detail=config.image_detail,
            )
        except Exception as exc:
            errors = [f"teacher_call_failed:{type(exc).__name__}:{str(exc)[:500]}"]
            attempts.append({"attempt": attempt + 1, "errors": list(errors)})
            continue
        raw_answer = generation.answer
        value, errors = parser(raw_answer, generation.model)
        attempts.append({"attempt": attempt + 1, "errors": list(errors), "telemetry": generation.telemetry or {}})
        if value is not None:
            return value, {
                "status": "complete",
                "attempts": attempts,
                "raw_answer": raw_answer[: config.raw_answer_max_chars],
            }
    return None, {
        "status": "error",
        "errors": list(errors) or ["teacher_validation_failed"],
        "attempts": attempts,
        "raw_answer": raw_answer[: config.raw_answer_max_chars],
    }


def _current_fpv_source_v1(source: ReasoningSourceV4) -> _VisibleImageSourceV1:
    """Expose exactly one current FPV image to local geometry/scene profiles."""

    for image, role in zip(source.observation.images, source.observation.image_roles):
        if role == "fpv.current":
            return _VisibleImageSourceV1(images=[image], image_roles=[role])
    raise ValueError("source_has_no_fpv_current")


def _temporal_fpv_source_v1(source: ReasoningSourceV4, *, max_frames: int) -> _VisibleImageSourceV1:
    """Expose ordered history/current FPV frames for temporal consistency only."""

    if max_frames < 1:
        raise ValueError("max_temporal_fpv_frames_must_be_positive")
    selected = [
        (image, role)
        for image, role in zip(source.observation.images, source.observation.image_roles)
        if role == "fpv.current" or role.startswith("fpv.history_")
    ]
    if not selected or selected[-1][1] != "fpv.current":
        selected = sorted(selected, key=lambda item: (item[1] == "fpv.current", item[1]))
    selected = selected[-max_frames:]
    if not any(role == "fpv.current" for _image, role in selected):
        raise ValueError("source_has_no_fpv_current")
    return _VisibleImageSourceV1(images=[item[0] for item in selected], image_roles=[item[1] for item in selected])


def _load_sources_v1(path: str | Path, *, limit: int | None) -> list[ReasoningSourceV4]:
    """Load unique source rows in deterministic manifest order."""

    rows: list[ReasoningSourceV4] = []
    seen: set[str] = set()
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        parent = source.observation.source_group_id
        if parent in seen:
            raise ValueError(f"duplicate_source_group_id:{parent}")
        seen.add(parent)
        rows.append(source)
        if limit is not None and len(rows) >= max(0, int(limit)):
            break
    return rows


def _latest_geometry_v1(path: str | Path) -> dict[tuple[str, str], LocalTraversabilityGeometryV1]:
    """Load latest geometry sidecar per exact source revision for resume."""

    source = Path(path).expanduser()
    if not source.is_file():
        return {}
    rows: dict[tuple[str, str], LocalTraversabilityGeometryV1] = {}
    for payload in iter_jsonl(source):
        annotation = LocalTraversabilityGeometryV1.model_validate(payload)
        rows[(annotation.parent_sample_id, annotation.source_fingerprint)] = annotation
    return rows


def _latest_trainability_filter_v1(
    path: str | Path,
) -> dict[tuple[str, str], NavigationTrainabilityFilterV1]:
    """Load latest suitability result per exact source revision for resume."""

    source = Path(path).expanduser()
    if not source.is_file():
        return {}
    rows: dict[tuple[str, str], NavigationTrainabilityFilterV1] = {}
    for payload in iter_jsonl(source):
        decision = NavigationTrainabilityFilterV1.model_validate(payload)
        rows[(decision.parent_sample_id, decision.source_fingerprint)] = decision
    return rows


def _load_stage_ii_sidecars_v2(
    path: str | Path | None,
) -> dict[str, StageIITrajectorySidecarV2]:
    """Load one immutable accepted Stage-II V2 sidecar per physical parent.

    Args:
        path: Optional V2 sidecar JSONL.  ``None`` keeps source-only motion
            behavior for legacy-compatible scene/geometry filtering.

    Returns:
        Sidecars keyed by ``sample_id``.

    Raises:
        FileNotFoundError: If an explicitly requested sidecar ledger is absent.
        ValueError: If one parent has conflicting accepted sidecars.
    """

    if path is None:
        return {}
    sidecar_path = Path(path).expanduser()
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"trajectory_sidecar_jsonl_not_found:{sidecar_path}")
    result: dict[str, StageIITrajectorySidecarV2] = {}
    for payload in iter_jsonl(sidecar_path):
        sidecar = StageIITrajectorySidecarV2.model_validate(payload)
        existing = result.get(sidecar.sample_id)
        if existing is not None and existing.sidecar_digest != sidecar.sidecar_digest:
            raise ValueError(f"conflicting_stage_ii_sidecar:{sidecar.sample_id}")
        result[sidecar.sample_id] = sidecar
    return result


def _motion_summary_for_source_v1(
    source: ReasoningSourceV4,
    *,
    sidecar: StageIITrajectorySidecarV2 | None,
    policy: NavigationTrainabilityPolicyV1,
) -> MotionEpisodeSummaryV1:
    """Prefer a lineage-compatible V2 sidecar over optional source trajectory.

    The join is intentionally narrow: physical parent, dataset and split-group
    must match before a teacher-only sidecar can influence Stage-II eligibility.
    """

    if sidecar is None:
        return summarize_reasoning_source_motion_v1(source, policy=policy)
    if (
        sidecar.dataset_source != source.observation.dataset_source
        or sidecar.source_group_id != source.observation.source_group_id
        or _dataset_qualified_split_group_v1(
            sidecar.dataset_source,
            sidecar.split_group_id,
        )
        != _dataset_qualified_split_group_v1(
            source.observation.dataset_source,
            source.observation.split_group_id,
        )
    ):
        raise ValueError(f"trajectory_sidecar_identity_mismatch:{sidecar.sample_id}")
    return summarize_stage_ii_sidecar_motion_v1(sidecar, policy=policy)


def _dataset_qualified_split_group_v1(dataset_source: str, split_group_id: str) -> str:
    """Normalize legacy bare and dataset-qualified split-group identities.

    Reasoning V4 sources created before the Stage-II V2 sidecar schema may
    carry ``clip_001`` while the immutable sidecar carries
    ``sidewalk:clip_001``.  They denote the same split unit only when the
    dataset source is identical, which is checked by the caller first.
    """

    prefix = f"{dataset_source}:"
    return split_group_id if split_group_id.startswith(prefix) else f"{prefix}{split_group_id}"


def _make_backends_v1(
    *,
    count: int,
    keys: tuple[str, ...],
    backend_factory: Callable[[int, str | None], NavigationQualityTeacherBackendV1] | None,
) -> list[NavigationQualityTeacherBackendV1]:
    """Create one provider adapter per bounded worker slot."""

    return [
        backend_factory(index, keys[index % len(keys)] if keys else None)
        if backend_factory is not None
        else VisionNavigationQualityTeacherBackendV1(api_key=keys[index % len(keys)] if keys else None)
        for index in range(max(1, count))
    ]


def _clean_keys_v1(values: Iterable[str]) -> tuple[str, ...]:
    """Normalize non-empty in-memory API keys without persistence."""

    return tuple(str(value).strip() for value in values if str(value).strip())


def _geometry_decision_v1(
    *,
    source: ReasoningSourceV4,
    annotation: LocalTraversabilityGeometryV1,
    attempts: list[dict[str, Any]],
    raw_answer: str,
    raw_answer_max_chars: int,
) -> dict[str, Any]:
    """Build an append-only geometry audit payload with no image bytes."""

    return {
        "kind": "traversability_geometry",
        "parent_sample_id": source.observation.source_group_id,
        "source_fingerprint": source.source_fingerprint,
        "status": annotation.validator_status,
        "geometry_id": annotation.geometry_id,
        "profile_id": annotation.prompt_profile_id,
        "profile_digest": annotation.prompt_profile_digest,
        "attempts": attempts,
        "raw_answer": raw_answer[:raw_answer_max_chars],
    }


def _filter_decision_v1(
    *,
    source: ReasoningSourceV4,
    result: NavigationTrainabilityFilterV1,
    scene_audit: dict[str, Any],
    motion_audit: dict[str, Any] | None,
    motion_summary: Any,
) -> dict[str, Any]:
    """Build an append-only suitability audit without embedding media bytes."""

    return {
        "kind": "navigation_trainability",
        "parent_sample_id": source.observation.source_group_id,
        "source_fingerprint": source.source_fingerprint,
        "decision_id": result.decision_id,
        "status": result.status,
        "reason_codes": list(result.reason_codes),
        "eligibility": result.eligibility.model_dump(mode="json"),
        "motion_summary": motion_summary.model_dump(mode="json"),
        "scene": scene_audit,
        "motion_scene": motion_audit,
    }


def _maybe_progress_v1(
    *,
    label: str,
    completed: int,
    total: int,
    counts: Counter[str],
    started: float,
    interval: int,
) -> None:
    """Emit a compact periodic progress event without writing mutable state."""

    if interval <= 0 or completed == 0 or completed % interval != 0:
        return
    elapsed = max(time.perf_counter() - started, 1e-6)
    print(
        json.dumps(
            {
                label: f"{completed}/{total}",
                "rows_per_hour": round(completed * 3600.0 / elapsed, 2),
                "counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _report_v1(
    *,
    schema_version: str,
    counts: Counter[str],
    errors: list[dict[str, Any]],
    elapsed_s: float,
    profiles: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Assemble one durable report with bounded error examples."""

    return {
        "schema_version": schema_version,
        "counts": dict(sorted(counts.items())),
        "error_count": len(errors),
        "errors": errors[:200],
        "profiles": profiles,
        "elapsed_s": round(elapsed_s, 3),
    }


def _write_report_v1(path: str | Path | None, report: dict[str, Any]) -> None:
    """Atomically replace the optional aggregate report after a batch completes."""

    if path is None:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "NavigationQualityBatchConfigV1",
    "NavigationQualityTeacherBackendV1",
    "VisionNavigationQualityTeacherBackendV1",
    "annotate_traversability_geometry_source_v1",
    "filter_navigation_trainability_source_v1",
    "filter_navigation_trainability_sources_v1",
    "label_traversability_geometry_sources_v1",
]
