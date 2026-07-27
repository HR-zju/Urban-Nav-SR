"""Offline acceptance audit for Reasoning V4 source, annotation, and views."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_prompt_policy import (
    get_prompt_profile_v4,
)
from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_target_v4,
)
from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    AnnotationStageRecordV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
    V4_FORBIDDEN_VISIBLE_CONTEXT_KEY_PARTS,
)
from mcp_server.data.jsonl_io import iter_jsonl


def audit_reasoning_v4_source_coverage(
    source_jsonls: Sequence[str | Path],
    *,
    required_dataset_families: tuple[str, ...] = (),
    min_source_rows: int = 0,
    check_media: bool = False,
) -> dict[str, Any]:
    """Audit one or more V4 source manifests before teacher annotation.

    Args:
        source_jsonls: Frozen V4 source manifests to validate jointly.
        required_dataset_families: Dataset prefixes that must occur.
        min_source_rows: Minimum aggregate row count.
        check_media: Require every local image path to exist.

    Returns:
        Coverage, supervision, warning, leakage, and media findings.
    """

    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[str] = Counter()
    source_ids: set[str] = set()
    paths = [Path(value).expanduser() for value in source_jsonls]
    if not paths:
        errors.append("no_source_manifests")
    for path in paths:
        if not path.is_file():
            errors.append(f"source_manifest_missing:{path}")
            continue
        counts["source_manifests"] += 1
        for row in iter_jsonl(path):
            raw_observation = row.get("observation")
            raw_context = (
                raw_observation.get("model_visible_context")
                if isinstance(raw_observation, Mapping)
                else {}
            )
            private_paths = _forbidden_visible_key_paths(raw_context)
            source_id_hint = (
                str(raw_observation.get("source_group_id") or "unknown")
                if isinstance(raw_observation, Mapping)
                else "unknown"
            )
            if private_paths:
                errors.append(
                    f"private_visible_context_keys:{source_id_hint}:"
                    f"{private_paths}"
                )
            try:
                source = ReasoningSourceV4.model_validate(row)
            except (TypeError, ValueError, ValidationError) as exc:
                counts["invalid_source_rows"] += 1
                errors.append(
                    f"invalid_source_row:{source_id_hint}:{str(exc)[:500]}"
                )
                continue
            observation = source.observation
            source_id = observation.source_group_id
            if source_id in source_ids:
                errors.append(f"duplicate_source_group_id:{source_id}")
            source_ids.add(source_id)
            counts["source_rows"] += 1
            counts[f"dataset:{observation.dataset_source}"] += 1
            counts["instruction_rows"] += int(
                observation.input_modalities.instruction
            )
            counts["temporal_eligible_rows"] += int(
                observation.temporal_evidence_eligible
            )
            counts["direction_source_rows"] += int(
                source.direction_supervision is not None
            )
            counts["system1_source_rows"] += int(
                source.trajectory_ego_t is not None
            )
            if source.direction_supervision is not None:
                counts[
                    "direction_profile:"
                    f"{source.direction_supervision.profile_id}"
                ] += 1
            for name, present in observation.input_modalities.model_dump(
                mode="python"
            ).items():
                counts[f"modality:{name}"] += int(bool(present))
            for warning in source.warnings:
                warning_code = warning.split(":", 1)[0]
                counts[f"source_warning:{warning_code}"] += 1
            if check_media:
                for image in observation.images:
                    value = str(image)
                    if value.startswith(
                        ("http://", "https://", "data:image/")
                    ):
                        continue
                    if not Path(value).expanduser().is_file():
                        errors.append(
                            f"source_image_missing:{source_id}:{value}"
                        )

    if counts["source_rows"] < max(0, int(min_source_rows)):
        errors.append(
            "insufficient_source_rows:"
            f"{counts['source_rows']}<{max(0, int(min_source_rows))}"
        )
    for family in required_dataset_families:
        if not any(
            key.startswith("dataset:")
            and key.removeprefix("dataset:").lower().startswith(
                family.lower()
            )
            and value > 0
            for key, value in counts.items()
        ):
            errors.append(f"required_dataset_missing:{family}")
    if not counts["direction_source_rows"]:
        warnings.append("no_direction_supervision_in_coverage_set")
    if not counts["system1_source_rows"]:
        warnings.append("no_system1_supervision_in_coverage_set")
    return {
        "schema_version": (
            "urbannav_reasoning_v4_2_source_coverage_report_v1"
        ),
        "source_jsonls": [str(path) for path in paths],
        "passed": not errors,
        "counts": dict(counts),
        "errors": errors[:500],
        "warnings": warnings[:100],
        "requirements": {
            "required_dataset_families": list(
                required_dataset_families
            ),
            "min_source_rows": max(0, int(min_source_rows)),
            "check_media": bool(check_media),
        },
    }


def audit_reasoning_v4_artifacts(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    view_jsonl: str | Path,
    *,
    required_dataset_families: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Audit V4 artifacts without calling mutable or external services.

    Args:
        source_jsonl: Frozen source manifest.
        annotation_jsonl: Append-only annotation manifest.
        view_jsonl: Exported training-view manifest.
        required_dataset_families: Optional dataset prefixes that must occur.

    Returns:
        Deterministic counts, bounded findings, and an overall pass flag.
    """

    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[str] = Counter()
    stage_metrics: dict[str, dict[str, Any]] = {}
    telemetry_totals: dict[str, Any] = {
        "provider_calls": 0,
        "roundtrip_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
    }
    sources: dict[str, ReasoningSourceV4] = {}
    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV4.model_validate(row)
        source_id = source.observation.source_group_id
        if source_id in sources:
            errors.append(f"duplicate_source_group_id:{source_id}")
        sources[source_id] = source
        dataset = source.observation.dataset_source
        counts["source_rows"] += 1
        counts[f"dataset:{dataset}"] += 1
        counts["direction_source_rows"] += int(
            source.direction_supervision is not None
        )
        counts["system1_source_rows"] += int(
            source.trajectory_ego_t is not None
        )
        counts["temporal_eligible_rows"] += int(
            source.observation.temporal_evidence_eligible
        )
        for warning in source.warnings:
            counts[f"source_warning:{warning.split(':', 1)[0]}"] += 1

    annotations: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(annotation_jsonl):
        annotation = ReasoningAnnotationV4.model_validate(row)
        annotations[annotation.source_group_id] = annotation
    for source_id, annotation in annotations.items():
        counts["annotation_rows"] += 1
        counts[f"annotation:{annotation.status}"] += 1
        source = sources.get(source_id)
        if source is None:
            errors.append(f"annotation_without_source:{source_id}")
            continue
        if annotation.source_fingerprint != source.source_fingerprint:
            errors.append(f"annotation_source_fingerprint_mismatch:{source_id}")
            continue
        for error in annotation.errors:
            counts[f"annotation_error:{error.split(':', 1)[0]}"] += 1
            if "selected_action_decision_type_conflict" in error:
                counts["selected_action_decision_type_conflict_rows"] += 1
            if "stop_action_lacks_visible_support" in error:
                counts["unsupported_stop_rows"] += 1
        for warning in annotation.warnings:
            counts[
                f"annotation_warning:{warning.split(':', 1)[0]}"
            ] += 1
        for name, record in annotation.stage_records.items():
            counts[f"stage:{name}:{record.status}"] += 1
            if record.raw_answer:
                counts["teacher_answer_rows"] += 1
                leakage = _reasoning_v4_leakage_categories(
                    record.raw_answer
                )
                for category in leakage:
                    counts[
                        f"teacher_answer_leakage:{category}"
                    ] += 1
                if record.status == "accepted":
                    counts["accepted_teacher_answer_rows"] += 1
                    for category in leakage:
                        counts[
                            "accepted_teacher_answer_leakage:"
                            f"{category}"
                        ] += 1
            _accumulate_audit_stage_metrics(
                stage_metrics,
                telemetry_totals,
                section_name=name,
                record=record,
            )
            profile = record.prompt_profile_id
            if not profile.startswith("reasoning_v4_teacher_"):
                errors.append(
                    f"invalid_teacher_prompt_profile:{source_id}:{name}:{profile}"
                )
        if annotation.target is not None:
            result = validate_reasoning_target_v4(
                annotation.target,
                observation=source.observation,
            )
            if not result.valid:
                errors.extend(
                    f"annotation_target:{source_id}:{item}"
                    for item in result.errors
                )
            warnings.extend(
                f"annotation_target:{source_id}:{item}"
                for item in result.warnings
            )
            trace = annotation.target.reasoning
            counts["scene_entity_count"] += len(
                trace.scene_grounding.entities
            )
            counts["direction_affordance_count"] += len(
                trace.local_traversability.direction_affordances
            )
            counts["constraint_count"] += len(
                trace.interaction_constraint_grounding.constraints
            )
            counts[
                f"decision_type:{trace.decision.decision_type}"
            ] += 1

    sample_ids: set[str] = set()
    full_view_sources: set[str] = set()
    for row in iter_jsonl(view_jsonl):
        view = ReasoningTrainingViewV4.model_validate(row)
        counts["view_rows"] += 1
        counts[f"view:{view.view_type}"] += 1
        if view.section_name is not None:
            counts[f"view_section:{view.section_name}"] += 1
        if view.sample_id in sample_ids:
            errors.append(f"duplicate_view_sample_id:{view.sample_id}")
        sample_ids.add(view.sample_id)
        source = sources.get(view.source_group_id)
        if source is None:
            errors.append(f"view_without_source:{view.sample_id}")
            continue
        profile = get_prompt_profile_v4(view.prompt_profile_id)
        if (
            profile.policy_digest != view.prompt_policy_digest
            or profile.view_type != view.view_type
            or profile.section_name != view.section_name
        ):
            errors.append(f"view_prompt_profile_mismatch:{view.sample_id}")
        if view.view_type == "full_cot_sft":
            full_view_sources.add(view.source_group_id)
            parsed = parse_reasoning_target_v4(view.assistant_text)
            if not parsed.valid or parsed.target is None:
                errors.append(
                    f"invalid_full_view_text:{view.sample_id}:{parsed.errors}"
                )
            elif parsed.target != view.reasoning_target:
                errors.append(
                    f"full_view_target_mismatch:{view.sample_id}"
                )
            if source.direction_supervision != view.direction_supervision:
                errors.append(
                    f"full_view_direction_sidecar_mismatch:{view.sample_id}"
                )
            if (
                view.reasoning_target is not None
                and view.reasoning_target.finish.decision_type == "navigate"
            ):
                if source.trajectory_ego_t != view.trajectory_ego_t:
                    errors.append(
                        "full_view_system1_sidecar_mismatch:"
                        f"{view.sample_id}"
                    )
            elif view.trajectory_ego_t is not None:
                errors.append(
                    "non_navigation_full_view_contains_system1_sidecar:"
                    f"{view.sample_id}"
                )
            elif (
                source.trajectory_ego_t is not None
                and view.metadata.get("system1_omitted_reason")
                != "decision_not_navigate"
            ):
                errors.append(
                    "non_navigation_full_view_missing_system1_omission_audit:"
                    f"{view.sample_id}"
                )
        elif (
            view.direction_supervision is not None
            or view.trajectory_ego_t is not None
        ):
            errors.append(
                f"section_view_contains_private_sidecar:{view.sample_id}"
            )

    for family in required_dataset_families:
        if not any(
            source.observation.dataset_source.lower().startswith(
                family.lower()
            )
            for source in sources.values()
        ):
            errors.append(f"required_dataset_missing:{family}")
    accepted_source_ids = {
        source_id
        for source_id, annotation in annotations.items()
        if annotation.status == "accepted"
    }
    missing_full = sorted(accepted_source_ids - full_view_sources)
    if missing_full:
        errors.append(f"accepted_sources_without_full_view:{missing_full[:20]}")
    for warning in warnings:
        counts[f"validation_warning:{warning.split(':', 2)[-1]}"] += 1
    annotation_rows = counts["annotation_rows"]
    accepted_rows = counts["annotation:accepted"]
    annotation_acceptance_fraction = (
        accepted_rows / annotation_rows if annotation_rows else 0.0
    )
    _finalize_audit_stage_metrics(stage_metrics)
    accepted_answer_rows = int(
        counts["accepted_teacher_answer_rows"]
    )
    leakage_metrics: dict[str, dict[str, Any]] = {}
    for category in (
        "semantic_region",
        "numeric_confidence_or_cost",
        "disabled_geometry",
    ):
        leakage_rows = int(
            counts[
                f"accepted_teacher_answer_leakage:{category}"
            ]
        )
        leakage_metrics[category] = {
            "rows": leakage_rows,
            "fraction": (
                leakage_rows / accepted_answer_rows
                if accepted_answer_rows
                else 0.0
            ),
        }
        if leakage_rows:
            errors.append(
                "accepted_teacher_answer_leakage:"
                f"{category}:{leakage_rows}"
            )
    provider_calls = int(telemetry_totals["provider_calls"])
    telemetry_totals["average_roundtrip_ms"] = (
        float(telemetry_totals["roundtrip_ms"]) / provider_calls
        if provider_calls
        else None
    )
    return {
        "schema_version": "urbannav_reasoning_v4_2_audit_report_v1",
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "view_jsonl": str(view_jsonl),
        "passed": not errors,
        "counts": dict(counts),
        "metrics": {
            "annotation_acceptance_fraction": (
                annotation_acceptance_fraction
            ),
            "stage_metrics": stage_metrics,
            "telemetry_totals": telemetry_totals,
            "accepted_teacher_answer_leakage": leakage_metrics,
        },
        "errors": errors[:200],
        "warnings": warnings[:500],
    }


def _accumulate_audit_stage_metrics(
    stage_metrics: dict[str, dict[str, Any]],
    telemetry_totals: dict[str, Any],
    *,
    section_name: str,
    record: AnnotationStageRecordV4,
) -> None:
    """Accumulate one annotation stage into stable audit counters."""

    metric = stage_metrics.setdefault(
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
        },
    )
    metric[record.status] = int(metric[record.status]) + 1
    metric["attempts"] = int(metric["attempts"]) + int(record.attempts)
    calls = record.telemetry.get("calls")
    if not isinstance(calls, list):
        return
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        metric["provider_calls"] = int(metric["provider_calls"]) + 1
        telemetry_totals["provider_calls"] = (
            int(telemetry_totals["provider_calls"]) + 1
        )
        roundtrip_ms = _audit_number(call.get("roundtrip_ms"))
        metric["roundtrip_ms"] = (
            float(metric["roundtrip_ms"]) + roundtrip_ms
        )
        telemetry_totals["roundtrip_ms"] = (
            float(telemetry_totals["roundtrip_ms"]) + roundtrip_ms
        )
        response = call.get("response")
        usage = (
            response.get("usage")
            if isinstance(response, Mapping)
            and isinstance(response.get("usage"), Mapping)
            else {}
        )
        prompt_details = usage.get("prompt_tokens_details")
        values = {
            "prompt_tokens": _audit_int(usage.get("prompt_tokens")),
            "completion_tokens": _audit_int(
                usage.get("completion_tokens")
            ),
            "total_tokens": _audit_int(usage.get("total_tokens")),
            "cached_prompt_tokens": _audit_int(
                prompt_details.get("cached_tokens")
                if isinstance(prompt_details, Mapping)
                else call.get("prompt_cached_tokens")
            ),
        }
        for key, value in values.items():
            metric[key] = int(metric[key]) + value
            telemetry_totals[key] = (
                int(telemetry_totals[key]) + value
            )


def _finalize_audit_stage_metrics(
    stage_metrics: dict[str, dict[str, Any]],
) -> None:
    """Add derived acceptance and average-latency values in place."""

    for metric in stage_metrics.values():
        records = sum(
            int(metric[name])
            for name in ("accepted", "rejected", "error", "skipped")
        )
        provider_calls = int(metric["provider_calls"])
        metric["acceptance_fraction"] = (
            int(metric["accepted"]) / records if records else 0.0
        )
        metric["average_attempts"] = (
            int(metric["attempts"]) / records if records else 0.0
        )
    metric["average_roundtrip_ms"] = (
        float(metric["roundtrip_ms"]) / provider_calls
        if provider_calls
        else None
    )


def _reasoning_v4_leakage_categories(text: str) -> set[str]:
    """Classify unsupported structured claims in one raw teacher answer."""

    value = str(text or "")
    findings: set[str] = set()
    if re.search(
        r'(?i)(?:"(?:region_ref|regions|selected_region_ref)"\s*:'
        r"|\bregion\.[a-z0-9_]+)",
        value,
    ):
        findings.add("semantic_region")
    if re.search(
        r'(?i)"(?:confidence|confidence_score|cost|cost_score)"'
        r"\s*:\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        value,
    ):
        findings.add("numeric_confidence_or_cost")
    if re.search(
        r"(?i)(?:route_theta|trajectory_(?:ego|fpv)_t|"
        r"\b(?:gps|gnss|ttc|track_id|bbox_ref|mask_ref|"
        r"segmentation_mask|instance_mask|projected_gt)\b)",
        value,
    ):
        findings.add("disabled_geometry")
    return findings


def _audit_number(value: Any) -> float:
    """Coerce finite non-negative telemetry to float."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if result < 0 or result != result or result == float("inf"):
        return 0.0
    return result


def _audit_int(value: Any) -> int:
    """Coerce telemetry to a non-negative integer."""

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _forbidden_visible_key_paths(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return prompt-visible mapping paths with forbidden private names."""

    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            normalized = key.lower()
            if any(
                part in normalized
                for part in V4_FORBIDDEN_VISIBLE_CONTEXT_KEY_PARTS
            ):
                findings.append(path)
            findings.extend(
                _forbidden_visible_key_paths(item, prefix=path)
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            findings.extend(
                _forbidden_visible_key_paths(item, prefix=path)
            )
    return findings
