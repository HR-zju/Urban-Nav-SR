"""Validated multimodal teacher pipeline for ReasoningTraceV2.

This module calls a provider only during offline annotation.  Evaluation and
training consume frozen source/annotation/view artifacts and never call a live
LLM.  The current evidence-conditioned E2E phase labels navigate decisions
only; tool choice, user interaction, and terminal decisions belong to later
agent-session datasets.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from mcp_server.agent.reasoning_text import parse_reasoning_target_v2, render_reasoning_target_v2
from mcp_server.agent.reasoning_validation import validate_reasoning_target_v2
from mcp_server.contracts.reasoning import ReasoningTargetV2
from mcp_server.contracts.reasoning_dataset import ReasoningAnnotationV2, ReasoningSourceV2
from mcp_server.data.jsonl_io import append_jsonl_row, iter_jsonl
from mcp_server.llm import LLMSettings, VisionAnalyzer, local_image_path_to_data_url


REASONING_V2_TEACHER_SYSTEM_PROMPT = (
    "You are an offline annotation teacher for the System2 reasoning policy of an urban assistive "
    "navigation robot. The robot supports blind or low-vision users, people with mobility impairments, "
    "and other pedestrians who need accessible, predictable, socially considerate guidance.\n\n"
    "Use only the provided model-visible instruction, context, evidence catalog, and ordered images. "
    "FPV is authoritative for immediate safety. Route, BEV, waypoint, and goal evidence provide task "
    "direction but cannot establish unseen free space or override a visible hazard. Do not infer agent "
    "motion from one image. Apply a traffic or social constraint only when its entity or controlled "
    "region, interaction with the selected path, and current travel mode are supported.\n\n"
    "This annotation stage is evidence-conditioned local navigation, not agent tool-policy training. "
    "Always emit decision_type=navigate. If immediate motion is unsafe or evidence is insufficient, "
    "emit a conservative navigate stop with trajectory_type=stop_hold and low confidence. Do not emit "
    "ask_user, complete_task, abort_task, tool calls, continuous trajectories, pixel coordinates, "
    "segmentation labels, or private teacher judgments.\n\n"
    "Return one strict JSON object and no markdown or prose outside it. Preserve only evidence refs "
    "listed in evidence_catalog; unsupported facts must be unknown or absent."
)


@dataclass(frozen=True)
class ReasoningTeacherConfig:
    """Configure validated V2 teacher calls and retry behavior.

    Args:
        max_retries: Repair attempts after the first provider response.
        image_detail: Provider image detail level.
        concurrency: Maximum in-flight source rows.
        retry_failed_on_resume: Reprocess prior rejected/error annotations.
        raw_answer_max_chars: Audit truncation limit for provider text.
    """

    max_retries: int = 2
    image_detail: str = "auto"
    concurrency: int = 1
    retry_failed_on_resume: bool = False
    raw_answer_max_chars: int = 50000


@dataclass(frozen=True)
class TeacherGeneration:
    """Raw backend generation and sanitized provider metadata.

    Args:
        answer: Raw generated text.
        provider: Provider name.
        model: Model identifier.
        telemetry: Timing and usage fields without prompts, images, or keys.
    """

    answer: str
    provider: str | None = None
    model: str | None = None
    telemetry: dict[str, Any] | None = None


class ReasoningTeacherBackend(Protocol):
    """Backend protocol used by the offline annotation runner."""

    async def generate(
        self,
        source: ReasoningSourceV2,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Generate one raw target for a frozen source.

        Args:
            source: Frozen multimodal teacher input.
            question: User-side annotation instruction.
            system_prompt: Teacher policy.
            image_detail: Provider image detail.

        Returns:
            Raw generation and sanitized metadata.
        """


class VisionReasoningTeacherBackend:
    """OpenAI-compatible multimodal backend implemented by ``VisionAnalyzer``.

    Args:
        settings: Optional explicit provider settings.  Environment settings
            are used when omitted.
        api_key: Optional per-worker key override.  It is never serialized.
    """

    def __init__(self, settings: LLMSettings | None = None, *, api_key: str | None = None):
        effective = settings or LLMSettings.from_env()
        if api_key:
            effective = replace(effective, api_key=str(api_key).strip())
        self.settings = effective

    async def generate(
        self,
        source: ReasoningSourceV2,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        """Call the configured multimodal endpoint for one source.

        Args:
            source: Frozen source with ordered image paths/URLs.
            question: Annotation question.
            system_prompt: Teacher policy.
            image_detail: ``auto``, ``low``, or ``high``.

        Returns:
            Raw answer plus sanitized telemetry.

        Raises:
            RuntimeError: If no provider key/model is configured.
            FileNotFoundError: If a local source image is missing.
        """

        analyzer = VisionAnalyzer(settings=self.settings)
        if not analyzer.is_available():
            raise RuntimeError("V2 reasoning teacher is not configured; set ARK_API_KEY or LLM_API_KEY")
        image_urls = [_image_reference_to_url(value) for value in source.images]
        result = await analyzer.analyze(
            image_url=image_urls[0],
            extra_image_urls=image_urls[1:],
            question=question,
            system_prompt=system_prompt,
            detail=image_detail,
        )
        return TeacherGeneration(
            answer=str(result.get("answer") or "").strip(),
            provider=str(result.get("provider") or self.settings.provider or "") or None,
            model=str(result.get("model") or self.settings.model or "") or None,
            telemetry=_sanitize_telemetry(result.get("llm_telemetry")),
        )


def build_reasoning_teacher_question(
    source: ReasoningSourceV2,
    *,
    previous_answer: str | None = None,
    previous_errors: Iterable[str] = (),
) -> str:
    """Build the exact JSON-only V2 annotation instruction.

    Args:
        source: Frozen model-visible input.
        previous_answer: Optional invalid response used for repair.
        previous_errors: Deterministic validator errors from that response.

    Returns:
        Provider user prompt.  ``teacher_private_context`` and continuous
        supervision are intentionally excluded.
    """

    visible = {
        "source_group_id": source.source_group_id,
        "travel_mode": source.travel_mode,
        "instruction": source.instruction,
        "image_order": [
            {"ordinal": index + 1, "role": role}
            for index, role in enumerate(source.image_roles)
        ],
        "available_modalities": source.available_modalities,
        "model_visible_context": source.model_visible_context,
        "evidence_catalog": [item.model_dump(mode="json", exclude_none=True) for item in source.evidence_catalog],
    }
    has_bev = "bev" in source.available_modalities
    skeleton = {
        "reasoning": {
            "schema_version": "urbannav_reasoning_trace_v2",
            "task_grounding": {
                "goal_ref": None,
                "route_ref": None,
                "local_objective": "unknown",
                "instruction_constraints": [],
            },
            "evidence_grounding": {
                "task_evidence_refs": [],
                "route_evidence_refs": [],
                "fpv_evidence_refs": [],
                "tool_evidence_refs": [],
                "relevant_facts": [],
            },
            "cross_modal_alignment": {
                "available": has_bev,
                "desired_direction": "unknown",
                "route_to_fpv": "unknown",
                "selected_corridor_ref": None,
                "status": "insufficient_evidence",
                "evidence_refs": [],
            },
            "constraint_applicability": {
                "applicable": [],
                "not_applicable": [],
            },
            "decision": {
                "motion": "go_straight",
                "speed": "normal",
                "social_action": "none",
                "selected_corridor_ref": None,
                "fallback": "slow",
                "confidence": "low",
                "reason_codes": [],
            },
        },
        "finish": {
            "schema_version": "urbannav_finish_v2",
            "decision_type": "navigate",
            "motion": "go_straight",
            "speed": "normal",
            "social_action": "none",
            "confidence_level": "low",
            "trajectory_type": "local_navigation",
            "tool_used": False,
        },
    }
    parts = [
        "Annotate this frozen input. Return one JSON object matching the skeleton exactly.",
        "ENUMS=" + json.dumps(_teacher_enum_contract(), ensure_ascii=False, separators=(",", ":")),
        (
            "For evidence_grounding.relevant_facts, every item MUST have exactly "
            "{\"fact_code\":\"lower_snake_case\",\"evidence_refs\":[\"catalog.ref\"]}. "
            "Do not emit ref, fact, text, description, or any other keys. fact_code is a compact "
            "controlled-style identifier such as center_corridor_visible, route_turns_left, or "
            "crossing_agent_on_path. Every evidence_refs value must be copied from FROZEN_INPUT "
            "evidence_catalog; use [] when no catalog reference supports the fact."
        ),
        (
            "For constraint_applicability.applicable, every item MUST have exactly "
            "{\"constraint\":\"enum_value\",\"trigger_refs\":[\"catalog.ref\"],\"effect\":\"enum_value\"}. "
            "Do not emit path_ref, controlled_path_refs, explanation, or any other keys. Every applicable "
            "item must include at least one explicit path ref with prefix corridor., route., waypoint., "
            "crosswalk., or bev. The wait_for_pedestrian_signal and use_designated_crossing constraints "
            "additionally require a control.* ref in the same trigger_refs list. If either required ref is "
            "absent from evidence_catalog, do not mark that constraint applicable. For "
            "constraint_applicability.not_applicable, every item MUST have exactly "
            "{\"constraint\":\"enum_value\",\"trigger_refs\":[\"catalog.ref\"],\"reason\":\"enum_value\"}."
            " The yield_to_crossing_agent, follow_agent, pass_left, and pass_right constraints additionally "
            "require a track.* ref and an explicit path ref in the same trigger_refs list. Seeing a person "
            "in a single FPV image does not create a track ref. If evidence_catalog has no track.* entry, "
            "do not mark any of those four constraints applicable; use a less specific visual-clearance "
            "constraint only when its own path evidence is present. Emit at most 3 applicable items and "
            "at most 3 not_applicable items; these are sparse examples, not exhaustive rule lists."
        ),
        (
            "The FPV images are ordered oldest to current according to image_order. Use temporal change across "
            "all FPV frames for scene evolution, but do not claim a tracked agent unless evidence_catalog has "
            "a track.* ref. If available_modalities does not contain bev, cross_modal_alignment MUST remain "
            "exactly unavailable: available=false, desired_direction=unknown, route_to_fpv=unknown, "
            "selected_corridor_ref=null, status=insufficient_evidence, evidence_refs=[]. If bev is present and "
            "cross_modal_alignment.available=true, evidence_refs MUST include bev.current and fpv.current or "
            "a corridor.* ref."
        ),
        "Keep applicable constraints sparse. Every applicable constraint must cite an explicit corridor/route/waypoint/crosswalk/BEV path ref. A single image may support visible occupancy or clearance but not a tracked motion claim.",
        "tool_evidence_refs must be [] in this stage. BEV and route refs belong in route_evidence_refs, not tool_evidence_refs.",
        "FROZEN_INPUT=" + json.dumps(visible, ensure_ascii=False, separators=(",", ":")),
        "OUTPUT_SKELETON=" + json.dumps(skeleton, ensure_ascii=False, separators=(",", ":")),
    ]
    errors = [str(item) for item in previous_errors if str(item)]
    if previous_answer is not None:
        parts.extend(
            [
                "The previous response was invalid. Correct it without adding unsupported evidence.",
                "VALIDATION_ERRORS=" + json.dumps(errors, ensure_ascii=False, separators=(",", ":")),
                "PREVIOUS_RESPONSE=" + str(previous_answer)[:16000],
            ]
        )
    return "\n\n".join(parts)


def _teacher_enum_contract() -> dict[str, list[str]]:
    """Return controlled values required by the JSON-only teacher prompt."""

    return {
        "local_objective": [
            "continue_route",
            "approach_goal",
            "enter_crosswalk",
            "exit_crosswalk",
            "follow_instruction",
            "stop_at_goal",
            "unknown",
        ],
        "desired_direction": ["left", "straight", "right", "stop", "unknown"],
        "route_to_fpv": ["left", "center", "right", "not_visible", "unknown"],
        "alignment_status": [
            "aligned",
            "locally_blocked",
            "conflict",
            "not_visible",
            "insufficient_evidence",
        ],
        "constraint": [
            "maintain_clearance",
            "slow_for_agent",
            "yield_to_crossing_agent",
            "wait_for_gap",
            "follow_agent",
            "pass_left",
            "pass_right",
            "detour_around_agent",
            "wait_for_pedestrian_signal",
            "use_designated_crossing",
            "avoid_vehicle_lane",
        ],
        "constraint_effect": ["go", "slow", "stop", "wait", "detour"],
        "not_applicable_reason": [
            "wrong_travel_mode",
            "path_does_not_interact",
            "path_does_not_enter_controlled_region",
            "entity_not_on_selected_corridor",
            "insufficient_evidence",
        ],
        "motion": ["go_straight", "turn_left", "turn_right", "stop"],
        "speed": ["normal", "slow", "stop"],
        "social_action": ["none", "yield", "wait", "follow", "pass_left", "pass_right", "detour", "unknown"],
        "fallback": ["continue", "slow", "stop", "request_replan", "unknown"],
        "confidence": ["high", "medium", "low", "unknown"],
    }


def parse_reasoning_teacher_answer(answer: str) -> tuple[ReasoningTargetV2 | None, list[str]]:
    """Parse raw provider JSON or already-rendered V2 assistant text.

    Args:
        answer: Raw provider response.

    Returns:
        ``(target, errors)``.
    """

    text = str(answer or "").strip()
    if "<TaskGrounding>" in text:
        parsed = parse_reasoning_target_v2(text)
        return parsed.target, list(parsed.errors)
    text = _strip_markdown_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"json_decode:{exc.msg}"]
    if not isinstance(payload, dict):
        return None, ["teacher_output_not_object"]
    if isinstance(payload.get("target"), dict):
        payload = payload["target"]
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict):
        applicability = reasoning.get("constraint_applicability")
        if isinstance(applicability, dict):
            for key in ("applicable", "not_applicable"):
                items = applicability.get(key)
                if isinstance(items, list) and len(items) > 3:
                    applicability[key] = items[:3]
    try:
        return ReasoningTargetV2.model_validate(payload), []
    except (TypeError, ValueError) as exc:
        return None, [f"reasoning_schema:{exc}"]


async def annotate_reasoning_source(
    source: ReasoningSourceV2,
    *,
    backend: ReasoningTeacherBackend,
    config: ReasoningTeacherConfig | None = None,
) -> ReasoningAnnotationV2:
    """Generate, repair, and validate one V2 annotation.

    Args:
        source: Frozen source row.
        backend: Multimodal teacher backend.
        config: Retry and image policy.

    Returns:
        Accepted, rejected, or provider-error annotation.  No template fallback
        is generated for invalid responses.
    """

    cfg = config or ReasoningTeacherConfig()
    raw_answer = ""
    errors: list[str] = []
    warnings: list[str] = []
    attempt_history: list[dict[str, Any]] = []
    generation: TeacherGeneration | None = None
    provider_errors = 0
    for attempt in range(max(0, int(cfg.max_retries)) + 1):
        question = build_reasoning_teacher_question(
            source,
            previous_answer=raw_answer if attempt else None,
            previous_errors=errors if attempt else (),
        )
        try:
            generation = await backend.generate(
                source,
                question=question,
                system_prompt=REASONING_V2_TEACHER_SYSTEM_PROMPT,
                image_detail=cfg.image_detail,
            )
        except Exception as exc:
            provider_errors += 1
            errors = [f"teacher_call_failed:{type(exc).__name__}:{str(exc)[:500]}"]
            attempt_history.append({"attempt": attempt + 1, "errors": list(errors)})
            continue
        raw_answer = generation.answer
        target, errors = parse_reasoning_teacher_answer(raw_answer)
        if target is None:
            attempt_history.append({"attempt": attempt + 1, "errors": list(errors)})
            continue
        if target.finish.decision_type != "navigate":
            errors = ["e2e_teacher_requires_navigate"]
            attempt_history.append({"attempt": attempt + 1, "errors": list(errors)})
            continue
        rendered = render_reasoning_target_v2(target)
        validation = validate_reasoning_target_v2(
            target,
            evidence_catalog=[item.model_dump(mode="json") for item in source.evidence_catalog],
            available_modalities=source.available_modalities,
            travel_mode=source.travel_mode,
            assistant_text=rendered,
            require_navigation_readout=False,
        )
        errors = list(validation.errors)
        warnings = list(validation.warnings)
        if not errors:
            return ReasoningAnnotationV2(
                source_group_id=source.source_group_id,
                source_fingerprint=source.source_fingerprint,
                status="accepted",
                target=target,
                raw_answer=raw_answer[: cfg.raw_answer_max_chars],
                errors=[],
                warnings=warnings,
                attempts=attempt + 1,
                teacher_provider=generation.provider,
                teacher_model=generation.model,
                telemetry=generation.telemetry or {},
                metadata={
                    "fallback_used": False,
                    "teacher_private_context_used": False,
                    "repair_history": attempt_history,
                },
            )
        attempt_history.append({"attempt": attempt + 1, "errors": list(errors)})
    status = "error" if provider_errors == max(0, int(cfg.max_retries)) + 1 else "rejected"
    return ReasoningAnnotationV2(
        source_group_id=source.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status=status,
        target=None,
        raw_answer=raw_answer[: cfg.raw_answer_max_chars] or None,
        errors=errors or ["teacher_validation_failed"],
        warnings=warnings,
        attempts=max(0, int(cfg.max_retries)) + 1,
        teacher_provider=generation.provider if generation else None,
        teacher_model=generation.model if generation else None,
        telemetry=generation.telemetry if generation and generation.telemetry else {},
        metadata={
            "fallback_used": False,
            "teacher_private_context_used": False,
            "repair_history": attempt_history,
        },
    )


async def label_reasoning_source_manifest(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
    config: ReasoningTeacherConfig | None = None,
    limit: int | None = None,
    api_keys: Iterable[str] = (),
    backend_factory: Callable[[int, str | None], ReasoningTeacherBackend] | None = None,
) -> dict[str, Any]:
    """Label a source manifest with bounded concurrency and append-only resume.

    Args:
        source_jsonl: Frozen ``ReasoningSourceV2`` JSONL.
        annotation_jsonl: Append-only output annotations.
        report_json: Optional periodically updated report.
        config: Retry/concurrency policy.
        limit: Optional maximum source rows.
        api_keys: Optional in-memory worker key rotation. Keys are never logged.
        backend_factory: Optional fake/custom backend factory for tests.

    Returns:
        Final annotation report.
    """

    cfg = config or ReasoningTeacherConfig()
    sources = [ReasoningSourceV2.model_validate(row) for row in iter_jsonl(source_jsonl)]
    if limit is not None:
        sources = sources[: int(limit)]
    completed = _existing_annotation_statuses(annotation_jsonl)
    pending = [
        source
        for source in sources
        if source.source_group_id not in completed
        or (cfg.retry_failed_on_resume and completed[source.source_group_id] != "accepted")
    ]
    keys = tuple(str(key).strip() for key in api_keys if str(key).strip())
    worker_count = min(max(1, int(cfg.concurrency)), max(1, len(pending)))
    if keys:
        worker_count = min(worker_count, len(keys))
    output_path = Path(annotation_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "urbannav_reasoning_annotation_v2_report",
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "input_rows": len(sources),
        "resumed_rows": len(sources) - len(pending),
        "pending_rows": len(pending),
        "processed_rows": 0,
        "accepted_rows": 0,
        "rejected_rows": 0,
        "error_rows": 0,
        "concurrency": worker_count,
        "status": "running",
        "error_counts": {},
    }
    if not pending:
        report["status"] = "done"
        _write_report(report_json, report)
        return report

    queue: asyncio.Queue[ReasoningSourceV2 | None] = asyncio.Queue()
    results: asyncio.Queue[ReasoningAnnotationV2] = asyncio.Queue()
    for source in pending:
        queue.put_nowait(source)
    for _ in range(worker_count):
        queue.put_nowait(None)

    def make_backend(worker_id: int) -> ReasoningTeacherBackend:
        key = keys[(worker_id - 1) % len(keys)] if keys else None
        if backend_factory is not None:
            return backend_factory(worker_id, key)
        return VisionReasoningTeacherBackend(api_key=key)

    async def worker(worker_id: int) -> None:
        backend = make_backend(worker_id)
        while True:
            source = await queue.get()
            try:
                if source is None:
                    return
                try:
                    annotation = await annotate_reasoning_source(source, backend=backend, config=cfg)
                except Exception as exc:
                    # 单行内部异常也必须形成可恢复 annotation，不能让 writer 永久等待。
                    annotation = ReasoningAnnotationV2(
                        source_group_id=source.source_group_id,
                        source_fingerprint=source.source_fingerprint,
                        status="error",
                        errors=[f"worker_exception:{type(exc).__name__}:{str(exc)[:500]}"],
                        attempts=0,
                        metadata={"worker_id": worker_id, "fallback_used": False},
                    )
                await results.put(annotation)
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker(index)) for index in range(1, worker_count + 1)]
    started_at = time.monotonic()
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for _ in range(len(pending)):
            annotation = await results.get()
            append_jsonl_row(handle, annotation.model_dump(mode="json", exclude_none=True))
            report["processed_rows"] += 1
            report[f"{annotation.status}_rows"] += 1
            for error in annotation.errors:
                key = error.split(":", 1)[0]
                report["error_counts"][key] = int(report["error_counts"].get(key, 0)) + 1
            elapsed = max(time.monotonic() - started_at, 1e-6)
            report["rate_rows_per_hour"] = float(report["processed_rows"]) / elapsed * 3600.0
            report["pending_rows"] = len(pending) - int(report["processed_rows"])
            _write_report(report_json, report)
    await queue.join()
    await asyncio.gather(*workers)
    report["status"] = "done"
    _write_report(report_json, report)
    return report


def _existing_annotation_statuses(path: str | Path) -> dict[str, str]:
    """Read latest per-source status from an existing annotation JSONL."""

    value = Path(path).expanduser()
    if not value.is_file():
        return {}
    statuses: dict[str, str] = {}
    for row in iter_jsonl(value):
        source_id = str(row.get("source_group_id") or "").strip()
        status = str(row.get("status") or "").strip()
        if source_id and status:
            statuses[source_id] = status
    return statuses


def _image_reference_to_url(value: str) -> str:
    """Convert a local path to a data URL while preserving remote/data URLs."""

    text = str(value).strip()
    if re.match(r"^(?:https?://|data:image/)", text, flags=re.IGNORECASE):
        return text
    return local_image_path_to_data_url(text)


def _strip_markdown_fence(text: str) -> str:
    """Strip one optional JSON markdown fence from a provider response."""

    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _sanitize_telemetry(value: Any) -> dict[str, Any]:
    """Keep only provider timing/usage fields safe for annotation artifacts."""

    if not isinstance(value, dict):
        return {}
    allowed = {
        "api_style",
        "provider",
        "model",
        "roundtrip_ms",
        "request_text_chars",
        "request_payload_chars",
        "response_text_chars",
        "temperature",
        "reasoning_effort",
        "timeout_s",
        "max_retries",
        "response",
        "prompt_cached_tokens",
        "stream_timing",
        "server_timing_available",
    }
    return {key: value[key] for key in allowed if key in value}


def _write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    """Write one human-readable progress report when configured."""

    if path is None:
        return
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


__all__ = [
    "REASONING_V2_TEACHER_SYSTEM_PROMPT",
    "ReasoningTeacherBackend",
    "ReasoningTeacherConfig",
    "TeacherGeneration",
    "VisionReasoningTeacherBackend",
    "annotate_reasoning_source",
    "build_reasoning_teacher_question",
    "label_reasoning_source_manifest",
    "parse_reasoning_teacher_answer",
]
