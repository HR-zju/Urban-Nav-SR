"""Deterministic offline audits for Reasoning V4.5 artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_prompt_policy import (
    get_prompt_profile_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    ReasoningAnnotationV4,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
    V4_FORBIDDEN_VISIBLE_CONTEXT_KEY_PARTS,
)
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.reasoning_v4_compact import (
    validate_compact_full_cot_projection_v4,
)
from mcp_server.data.reasoning_v4_reconciliation import (
    action_reconciliation_review_required_v4,
    validate_action_reconciliation_audit_v4,
)
from mcp_server.data.reasoning_v4_state import (
    derive_reasoning_release_state_v4,
    load_latest_action_resolutions_v4,
)


def audit_reasoning_v4_annotation_completeness(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Audit source coverage and completed annotation records.

    This audit intentionally does not collapse section acceptance, composition,
    review, or training eligibility into record completeness.

    Args:
        source_jsonl: Frozen V4.5 source manifest.
        annotation_jsonl: Append-only annotation manifest.
        expected_rows: Exact expected unique source count.

    Returns:
        Coverage metrics, bounded errors, and pass flag.
    """

    expected = int(expected_rows)
    if expected <= 0:
        raise ValueError("expected_rows must be positive")
    errors: list[str] = []
    sources = _source_index_v4(source_jsonl, errors=errors)
    annotations, file_rows = _annotation_index_v4(
        annotation_jsonl,
        errors=errors,
    )
    source_ids = set(sources)
    annotation_ids = set(annotations)
    missing = sorted(source_ids - annotation_ids)
    orphan = sorted(annotation_ids - source_ids)
    counts: Counter[str] = Counter()
    for source_id in sorted(source_ids & annotation_ids):
        source = sources[source_id]
        annotation = annotations[source_id]
        counts[
            f"record_status:{annotation.record_status}"
        ] += 1
        counts["all_sections_accepted"] += int(
            annotation.all_sections_accepted
        )
        counts["composition_valid"] += int(
            annotation.composition_record.status == "valid"
        )
        audit = annotation.action_reconciliation
        counts["pending_human"] += int(
            audit is not None
            and action_reconciliation_review_required_v4(audit)
        )
        if (
            annotation.source_fingerprint
            != source.source_fingerprint
        ):
            errors.append(
                f"source_fingerprint_mismatch:{source_id}"
            )
    if len(sources) != expected:
        errors.append(
            f"source_row_count_mismatch:{len(sources)}!={expected}"
        )
    if missing:
        errors.append(f"missing_annotations:{len(missing)}")
    if orphan:
        errors.append(
            f"annotations_without_source:{len(orphan)}"
        )
    if counts["record_status:complete"] != expected:
        errors.append(
            "record_complete_count_mismatch:"
            f"{counts['record_status:complete']}!={expected}"
        )
    return {
        "schema_version": (
            "urbannav_reasoning_v4_5_annotation_completeness_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "expected_rows": expected,
        "source_rows": len(sources),
        "annotation_file_rows": file_rows,
        "latest_annotation_rows": len(
            source_ids & annotation_ids
        ),
        "metrics": {
            "sources": len(sources),
            "record_complete": int(
                counts["record_status:complete"]
            ),
            "all_sections_accepted": int(
                counts["all_sections_accepted"]
            ),
            "composition_valid": int(
                counts["composition_valid"]
            ),
            "pending_human": int(
                counts["pending_human"]
            ),
            "training_eligible": 0,
        },
        "record_status_counts": {
            key.removeprefix("record_status:"): value
            for key, value in counts.items()
            if key.startswith("record_status:")
        },
        "missing_rows": len(missing),
        "missing_examples": missing[:50],
        "orphan_rows": len(orphan),
        "orphan_examples": orphan[:50],
        "errors": errors[:200],
        "passed": not errors,
    }


def audit_reasoning_v4_source_coverage(
    source_jsonls: Sequence[str | Path],
    *,
    required_dataset_families: tuple[str, ...] = (),
    min_source_rows: int = 0,
    check_media: bool = False,
) -> dict[str, Any]:
    """Audit one or more source manifests before annotation.

    Args:
        source_jsonls: Frozen source manifests.
        required_dataset_families: Required dataset prefixes.
        min_source_rows: Minimum aggregate source count.
        check_media: Require every referenced local image to exist.

    Returns:
        Coverage, direction, modality, leakage, and media findings.
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
            source_id_hint = (
                str(
                    raw_observation.get("source_group_id")
                    or "unknown"
                )
                if isinstance(raw_observation, Mapping)
                else "unknown"
            )
            private_paths = _forbidden_visible_key_paths(
                raw_context
            )
            if private_paths:
                errors.append(
                    "private_visible_context_keys:"
                    f"{source_id_hint}:{private_paths}"
                )
            try:
                source = ReasoningSourceV4.model_validate(row)
            except (TypeError, ValueError, ValidationError) as exc:
                counts["invalid_source_rows"] += 1
                errors.append(
                    "invalid_source_row:"
                    f"{source_id_hint}:{str(exc)[:500]}"
                )
                continue
            source_id = source.observation.source_group_id
            if source_id in source_ids:
                errors.append(
                    f"duplicate_source_group_id:{source_id}"
                )
            source_ids.add(source_id)
            counts["source_rows"] += 1
            dataset = source.observation.dataset_source
            family = dataset.split(":", 1)[0].lower()
            counts[f"dataset:{dataset}"] += 1
            counts[f"dataset_family:{family}"] += 1
            counts["strong_direction_rows"] += int(
                source.direction_supervision is not None
            )
            counts["system1_rows"] += int(
                source.trajectory_ego_t is not None
            )
            for name, present in (
                source.observation.input_modalities.model_dump(
                    mode="python"
                ).items()
            ):
                counts[f"modality:{name}"] += int(bool(present))
            for warning in source.warnings:
                counts[
                    "source_warning:"
                    f"{warning.split(':', 1)[0]}"
                ] += 1
            if check_media:
                for image in source.observation.images:
                    if not Path(image).expanduser().is_file():
                        errors.append(
                            f"missing_media:{source_id}:{image}"
                        )
    if counts["source_rows"] < int(min_source_rows):
        errors.append(
            "source_rows_below_minimum:"
            f"{counts['source_rows']}<{int(min_source_rows)}"
        )
    present_families = {
        key.removeprefix("dataset_family:")
        for key, value in counts.items()
        if key.startswith("dataset_family:") and value
    }
    for family in required_dataset_families:
        if family.lower() not in present_families:
            errors.append(
                f"required_dataset_family_missing:{family}"
            )
    if counts["source_rows"] and not counts[
        "strong_direction_rows"
    ]:
        warnings.append("no_strong_direction_supervision")
    return {
        "schema_version": (
            "urbannav_reasoning_v4_5_source_coverage_v1"
        ),
        "source_jsonls": [str(path) for path in paths],
        "counts": dict(counts),
        "errors": errors[:200],
        "warnings": warnings[:200],
        "passed": not errors,
    }


def audit_reasoning_v4_artifacts(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    view_jsonl: str | Path,
    *,
    resolution_jsonl: str | Path | None = None,
    required_dataset_families: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Audit source, annotation, and view artifacts offline.

    Args:
        source_jsonl: Frozen source manifest.
        annotation_jsonl: Append-only annotation manifest.
        view_jsonl: Review-preview or formal-training views.
        resolution_jsonl: Optional append-only human resolutions.
        required_dataset_families: Optional required dataset prefixes.

    Returns:
        Exact six-axis metrics, findings, and overall pass flag.
    """

    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[str] = Counter()
    sources = _source_index_v4(source_jsonl, errors=errors)
    annotations, _ = _annotation_index_v4(
        annotation_jsonl,
        errors=errors,
    )
    resolutions = load_latest_action_resolutions_v4(
        resolution_jsonl
    )
    for source_id, source in sources.items():
        counts["sources"] += 1
        dataset = source.observation.dataset_source
        family = dataset.split(":", 1)[0].lower()
        counts[f"dataset_family:{family}"] += 1
        annotation = annotations.get(source_id)
        if annotation is None:
            errors.append(f"missing_annotation:{source_id}")
            continue
        if (
            annotation.source_fingerprint
            != source.source_fingerprint
        ):
            errors.append(
                "annotation_source_fingerprint_mismatch:"
                f"{source_id}"
            )
            continue
        counts["record_complete"] += int(
            annotation.record_status == "complete"
        )
        counts["all_sections_accepted"] += int(
            annotation.all_sections_accepted
        )
        counts["composition_valid"] += int(
            annotation.composition_record.status == "valid"
        )
        audit = annotation.action_reconciliation
        resolution = (
            resolutions.get(audit.audit_id)
            if audit is not None
            else None
        )
        try:
            state = derive_reasoning_release_state_v4(
                source,
                annotation,
                resolution=resolution,
            )
        except ValueError as exc:
            errors.append(
                f"release_state:{source_id}:{exc}"
            )
            continue
        counts[
            f"review_status:{state.review_status}"
        ] += 1
        counts["pending_human"] += int(
            state.review_status == "pending_human"
        )
        counts["training_eligible"] += int(
            state.training_eligible
        )
        if (
            audit is not None
            and annotation.decision_proposal is not None
        ):
            try:
                validate_action_reconciliation_audit_v4(
                    source,
                    annotation.grounding_outputs,
                    annotation.decision_proposal,
                    audit,
                )
            except ValueError as exc:
                errors.append(
                    f"audit_reconstruction:{source_id}:{exc}"
                )

    view_ids: set[str] = set()
    for row in iter_jsonl(view_jsonl):
        try:
            view = ReasoningTrainingViewV4.model_validate(row)
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(f"invalid_view:{str(exc)[:500]}")
            continue
        if view.sample_id in view_ids:
            errors.append(
                f"duplicate_view_sample_id:{view.sample_id}"
            )
        view_ids.add(view.sample_id)
        counts["views"] += 1
        counts[
            f"release_track:{view.release_track}"
        ] += 1
        counts[
            f"review_status:{view.review_status}"
        ] += 1
        counts["view_training_eligible"] += int(
            view.training_eligible
        )
        try:
            profile = get_prompt_profile_v4(
                view.prompt_profile_id
            )
            if (
                profile.policy_digest
                != view.prompt_policy_digest
            ):
                raise ValueError("prompt_policy_digest_mismatch")
            validate_compact_full_cot_projection_v4(
                view.assistant_text,
                view.reasoning_target,
                action_profile_id=(
                    view.direction_context.action_profile_id
                ),
            )
        except ValueError as exc:
            errors.append(
                f"view_contract:{view.sample_id}:{exc}"
            )
        forbidden = _forbidden_assistant_tokens_v4(
            view.assistant_text
        )
        if forbidden:
            errors.append(
                "forbidden_assistant_fields:"
                f"{view.sample_id}:{forbidden}"
            )
    for source_id in sorted(set(annotations) - set(sources)):
        errors.append(
            f"annotation_without_source:{source_id}"
        )
    present_families = {
        key.removeprefix("dataset_family:")
        for key, value in counts.items()
        if key.startswith("dataset_family:") and value
    }
    for family in required_dataset_families:
        if family.lower() not in present_families:
            errors.append(
                f"required_dataset_family_missing:{family}"
            )
    return {
        "schema_version": (
            "urbannav_reasoning_v4_5_artifact_audit_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "resolution_jsonl": (
            str(resolution_jsonl)
            if resolution_jsonl is not None
            else None
        ),
        "view_jsonl": str(view_jsonl),
        "metrics": {
            "sources": int(counts["sources"]),
            "record_complete": int(
                counts["record_complete"]
            ),
            "all_sections_accepted": int(
                counts["all_sections_accepted"]
            ),
            "composition_valid": int(
                counts["composition_valid"]
            ),
            "pending_human": int(
                counts["pending_human"]
            ),
            "training_eligible": int(
                counts["training_eligible"]
            ),
        },
        "counts": dict(counts),
        "errors": errors[:200],
        "warnings": warnings[:200],
        "passed": not errors,
    }


def _source_index_v4(
    path: str | Path,
    *,
    errors: list[str],
) -> dict[str, ReasoningSourceV4]:
    """Load unique sources and append validation errors."""

    result: dict[str, ReasoningSourceV4] = {}
    for row in iter_jsonl(path):
        try:
            source = ReasoningSourceV4.model_validate(row)
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(
                f"invalid_source:{str(exc)[:500]}"
            )
            continue
        source_id = source.observation.source_group_id
        if source_id in result:
            errors.append(
                f"duplicate_source_group_id:{source_id}"
            )
        result[source_id] = source
    return result


def _annotation_index_v4(
    path: str | Path,
    *,
    errors: list[str],
) -> tuple[dict[str, ReasoningAnnotationV4], int]:
    """Load latest append-only annotations and file-row count."""

    result: dict[str, ReasoningAnnotationV4] = {}
    file_rows = 0
    for row in iter_jsonl(path):
        file_rows += 1
        try:
            annotation = ReasoningAnnotationV4.model_validate(
                row
            )
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(
                f"invalid_annotation:{str(exc)[:500]}"
            )
            continue
        result[annotation.source_group_id] = annotation
    return result, file_rows


def _forbidden_assistant_tokens_v4(
    text: str,
) -> list[str]:
    """Return removed V4 concepts found in assistant text."""

    forbidden = (
        "<finish>",
        "<readout>",
        "evidence_refs",
        "source_alignments",
        "route_progress",
        "goal_alignment",
        "decision_type",
        "social_action",
        "fallback",
        "support_refs",
        "critical_uncertainties",
        "temporal",
    )
    return [token for token in forbidden if token in text]


def _forbidden_visible_key_paths(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return visible-context paths containing private names."""

    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if any(
                part in key.lower()
                for part in V4_FORBIDDEN_VISIBLE_CONTEXT_KEY_PARTS
            ):
                findings.append(path)
            findings.extend(
                _forbidden_visible_key_paths(
                    item,
                    prefix=path,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(
                _forbidden_visible_key_paths(
                    item,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return findings


__all__ = [
    "audit_reasoning_v4_annotation_completeness",
    "audit_reasoning_v4_artifacts",
    "audit_reasoning_v4_source_coverage",
]
