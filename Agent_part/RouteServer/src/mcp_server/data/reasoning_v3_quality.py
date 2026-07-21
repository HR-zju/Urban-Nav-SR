"""Quality distributions and smoke gates for Reasoning V3.3 artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3, ReasoningSourceV3
from mcp_server.data.jsonl_io import iter_jsonl


REASONING_V3_QUALITY_REPORT_VERSION = "urbannav_reasoning_v3_3_quality_report_v1"


@dataclass(frozen=True)
class ReasoningV3QualityThresholds:
    """Configure provisional V3.3 smoke acceptance thresholds.

    Fractions use source rows for acceptance/rejection, accepted rows for
    conflict/unknown/insufficient, optional-context rows for context accounting,
    and projection candidates for projection QC.

    Args:
        min_acceptance_fraction: Minimum accepted/source fraction.
        max_reject_fraction: Maximum failed/source fraction.
        max_conflict_fraction: Maximum context-conflict/accepted fraction.
        max_unknown_fraction: Maximum critical-unknown/accepted fraction.
        max_insufficient_fraction: Maximum insufficient-context/accepted fraction.
        max_task_unknown_fraction: Maximum unresolved active-task fraction.
        max_unsupported_rationale_fraction: Maximum unsupported-decision fraction.
        min_projection_qc_coverage: Minimum passed/candidate projection fraction.
        min_context_accounting_fraction: Minimum explicit-accounted/context-row fraction.
    """

    min_acceptance_fraction: float = 0.75
    max_reject_fraction: float = 0.20
    max_conflict_fraction: float = 0.10
    max_unknown_fraction: float = 0.10
    max_insufficient_fraction: float = 0.10
    max_task_unknown_fraction: float = 0.20
    max_unsupported_rationale_fraction: float = 0.20
    min_projection_qc_coverage: float = 0.95
    min_context_accounting_fraction: float = 1.0

    def __post_init__(self) -> None:
        """Validate every configured fraction.

        Raises:
            ValueError: If any threshold lies outside ``[0, 1]``.
        """

        for name, value in self.__dict__.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def evaluate_reasoning_v3_quality(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    thresholds: ReasoningV3QualityThresholds | None = None,
    required_dataset_families: Iterable[str] = (),
) -> dict[str, Any]:
    """Measure latest V3.3 annotations and apply per-dataset smoke gates.

    The latest append-only annotation is authoritative for each source id.
    Accepted rows are sliced by traversability state, optional-context usage,
    per-source navigation grounding, social interaction, and traffic rules.

    Args:
        source_jsonl: Frozen V3.3 source manifest.
        annotation_jsonl: Append-only V3.3 annotation manifest.
        thresholds: Optional smoke gate configuration.
        required_dataset_families: Dataset families required in the source.

    Returns:
        Global and per-dataset counts, fractions, violations, and pass state.

    Raises:
        ValueError: If an artifact row violates its typed V3.3 schema.
    """

    limits = thresholds or ReasoningV3QualityThresholds()
    sources: dict[str, ReasoningSourceV3] = {}
    family_by_id: dict[str, str] = {}
    counters: dict[str, Counter[str]] = {}
    overall_counts: Counter[str] = Counter()
    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV3.model_validate(row)
        source_id = source.model_input.source_group_id
        family = _dataset_family(source.model_input.dataset_source)
        family_counts = counters.setdefault(family, Counter())
        family_counts["source_rows"] += 1
        overall_counts["source_rows"] += 1
        if source_id in sources:
            family_counts["duplicate_source_ids"] += 1
            overall_counts["duplicate_source_ids"] += 1
        sources[source_id] = source
        family_by_id[source_id] = family
        for counts in (family_counts, overall_counts):
            _count_source_capabilities(source, counts)
            _count_projection_qc(source, counts)

    latest: dict[str, ReasoningAnnotationV3] = {}
    for row in iter_jsonl(annotation_jsonl):
        annotation = ReasoningAnnotationV3.model_validate(row)
        latest[annotation.source_group_id] = annotation
        overall_counts["annotation_records"] += 1

    for source_id, source in sources.items():
        family_counts = counters[family_by_id[source_id]]
        annotation = latest.get(source_id)
        if annotation is None:
            family_counts["missing_annotation_rows"] += 1
            overall_counts["missing_annotation_rows"] += 1
            continue
        for counts in (family_counts, overall_counts):
            counts[f"status:{annotation.status}"] += 1
            if annotation.status != "accepted" or annotation.target is None:
                _count_failed_annotation(annotation, counts)
            else:
                _count_accepted_annotation(source, annotation, counts)

    orphan_ids = set(latest) - set(sources)
    overall_counts["orphan_latest_annotations"] = len(orphan_ids)
    required = {_dataset_family(value) for value in required_dataset_families}
    missing_families = sorted(required - set(counters))
    per_dataset: dict[str, Any] = {}
    violations: list[str] = []
    for family, counts in sorted(counters.items()):
        summary = _summarize_counts(counts, limits=limits, scope=family)
        per_dataset[family] = summary
        violations.extend(summary["violations"])
    overall = _summarize_counts(overall_counts, limits=limits, scope="overall")
    violations.extend(overall["violations"])
    violations.extend(f"required_dataset_missing:{family}" for family in missing_families)
    if overall_counts["duplicate_source_ids"]:
        violations.append("overall:duplicate_source_ids")
    if overall_counts["orphan_latest_annotations"]:
        violations.append("overall:orphan_latest_annotations")
    return {
        "schema_version": REASONING_V3_QUALITY_REPORT_VERSION,
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "thresholds": limits.__dict__,
        "overall": overall,
        "per_dataset": per_dataset,
        "missing_required_datasets": missing_families,
        "violation_count": len(violations),
        "violations": violations,
        "passed": not violations,
    }


def _count_source_capabilities(source: ReasoningSourceV3, counts: Counter[str]) -> None:
    """Count source-side modality and trainable-section capabilities."""

    modalities = set(source.model_input.available_modalities)
    for modality in sorted(modalities):
        counts[f"source_modality:{modality}"] += 1
    optional_refs = [
        entry.ref
        for entry in source.model_input.evidence_catalog
        if entry.modality in {"bev", "goal", "goal_image", "waypoint", "route", "tool", "tool_route"}
    ]
    counts["optional_context_source_items"] += len(optional_refs)
    counts["rows_with_optional_context"] += int(bool(optional_refs))
    for name, enabled in source.section_loss_mask.model_dump(mode="python").items():
        counts[f"source_section_enabled:{name}"] += int(bool(enabled))


def _count_projection_qc(source: ReasoningSourceV3, counts: Counter[str]) -> None:
    """Count explicit FPV projection candidates and QC outcomes."""

    provenance = source.label_provenance.get("demonstrated_corridor_ref")
    warnings = tuple(source.warnings)
    passed = provenance is not None and provenance.source == "validated_fpv_projection"
    candidate = passed or any(
        warning == "fpv_projection_ignored_without_explicit_qc"
        or warning.startswith("fpv_projection_rejected:")
        for warning in warnings
    )
    counts["projection_candidate_rows"] += int(candidate)
    counts["projection_qc_passed_rows"] += int(passed)
    counts["projection_qc_missing_rows"] += int("fpv_projection_ignored_without_explicit_qc" in warnings)
    counts["projection_qc_rejected_rows"] += int(
        any(warning.startswith("fpv_projection_rejected:") for warning in warnings)
    )


def _count_failed_annotation(annotation: ReasoningAnnotationV3, counts: Counter[str]) -> None:
    """Classify conflict, unknown, and omission errors in a failed annotation."""

    errors = " ".join(annotation.errors).lower()
    if "conflict" in errors:
        counts["conflict_rows"] += 1
        counts["failed_conflict_rows"] += 1
    if "unknown" in errors:
        counts["unknown_rows"] += 1
        counts["failed_unknown_rows"] += 1
    if "omitted" in errors or "coverage_mismatch" in errors:
        counts["context_accounting_failure_rows"] += 1


def _count_accepted_annotation(
    source: ReasoningSourceV3,
    annotation: ReasoningAnnotationV3,
    counts: Counter[str],
) -> None:
    """Count six-section distributions for one accepted annotation."""

    target = annotation.target
    if target is None:
        return
    trace = target.reasoning
    scene = trace.scene_grounding
    local = trace.local_traversability
    context = trace.navigation_context_grounding
    interaction = trace.interaction_and_rule_grounding
    task = trace.task_grounding
    decision = trace.decision
    counts["accepted_rows"] += 1

    counts[f"task_applicable:{str(task.applicable).lower()}"] += 1
    counts[f"task_instruction_type:{task.instruction_type}"] += 1
    counts[f"task_caption_role:{task.caption_role}"] += 1
    counts[f"task_local_objective:{task.local_objective}"] += 1
    counts["task_requirement_items"] += len(task.requirements)
    counts["task_referent_items"] += len(task.referents)
    counts["task_ambiguity_items"] += len(task.ambiguities)
    counts["active_task_unknown_rows"] += int(
        task.applicable
        and (task.instruction_type == "unknown" or task.local_objective == "unknown")
    )
    counts[f"decision_rationale_status:{decision.rationale_status}"] += 1
    counts["unsupported_rationale_rows"] += int(decision.rationale_status == "unsupported")
    counts["partial_rationale_rows"] += int(decision.rationale_status == "partial")
    counts["supported_rationale_rows"] += int(decision.rationale_status == "supported")
    counts["decision_support_items"] += len(decision.supporting_reasons)

    region_states = Counter(item.state for item in local.regions)
    for state in ("direct", "conditional", "blocked", "forbidden"):
        counts[f"region_state:{state}:items"] += region_states[state]
        counts[f"rows_with_region_state:{state}"] += int(region_states[state] > 0)

    expected_context_refs = {
        entry.ref
        for entry in source.model_input.evidence_catalog
        if entry.modality in {"bev", "goal", "goal_image", "waypoint", "route", "tool", "tool_route"}
    }
    observed_context = {item.source_ref: item for item in scene.context_observations}
    if expected_context_refs:
        counts["accepted_rows_with_optional_context"] += 1
        accounted = set(observed_context) == expected_context_refs
        counts["context_accounted_rows"] += int(accounted)
    for item in scene.context_observations:
        counts[f"context_usage:{item.usage}"] += 1
        counts[f"context_role:{item.role}"] += 1
        modality = _catalog_modality(source, item.source_ref)
        counts[f"context_modality:{modality}:{item.usage}"] += 1
    for item in scene.cross_modal_relations:
        counts[f"cross_modal_status:{item.status}"] += 1
        counts[f"cross_modal_relation:{item.relation}"] += 1

    if context.source_consistency == "conflict" or context.fused_execution_status == "conflict":
        counts["conflict_rows"] += 1
        counts["accepted_conflict_rows"] += 1
    if context.fused_execution_status == "insufficient_evidence":
        counts["insufficient_evidence_rows"] += 1
    for item in context.source_groundings:
        counts[f"navigation_source_type:{item.source_type}"] += 1
        counts[f"navigation_source_status:{item.source_type}:{item.status}"] += 1

    critical_unknown = local.applicable and (
        local.visibility == "unknown"
        or local.local_risk == "unknown"
        or local.confidence == "unknown"
    )
    if critical_unknown:
        counts["unknown_rows"] += 1
        counts["accepted_critical_unknown_rows"] += 1
    if (
        context.fused_target_corridor_ref is not None
        and context.fused_target_corridor_ref != trace.decision.selected_corridor_ref
    ):
        counts["fallback_rows"] += 1
    status = context.fused_execution_status if context.applicable else "not_applicable"
    counts[f"navigation_context_status:{status}"] += 1

    counts["interaction_items"] += len(interaction.interactions)
    counts["traffic_rule_items"] += len(interaction.traffic_rules)
    counts["rows_with_interaction"] += int(bool(interaction.interactions))
    counts["rows_with_traffic_rule"] += int(bool(interaction.traffic_rules))
    for item in interaction.interactions:
        counts[f"interaction_relation:{item.relation}"] += 1
        counts[f"required_behavior:{item.required_behavior}"] += 1
    for item in interaction.traffic_rules:
        counts[f"traffic_rule:{item.rule}"] += 1
        counts[f"required_behavior:{item.required_behavior}"] += 1


def _catalog_modality(source: ReasoningSourceV3, ref: str) -> str:
    """Return the frozen input modality for one evidence reference."""

    for entry in source.model_input.evidence_catalog:
        if entry.ref == ref:
            return entry.modality
    return "unknown"


def _summarize_counts(
    counts: Counter[str],
    *,
    limits: ReasoningV3QualityThresholds,
    scope: str,
) -> dict[str, Any]:
    """Compute configured fractions and threshold violations."""

    source_rows = counts["source_rows"]
    accepted = counts["accepted_rows"]
    failed = counts["status:rejected"] + counts["status:error"] + counts["missing_annotation_rows"]
    projection_candidates = counts["projection_candidate_rows"]
    context_rows = counts["accepted_rows_with_optional_context"]
    fractions = {
        "acceptance_fraction": _fraction(accepted, source_rows),
        "reject_fraction": _fraction(failed, source_rows),
        "conflict_fraction_among_accepted": _fraction(counts["accepted_conflict_rows"], accepted),
        "critical_unknown_fraction_among_accepted": _fraction(
            counts["accepted_critical_unknown_rows"], accepted
        ),
        "insufficient_evidence_fraction_among_accepted": _fraction(
            counts["insufficient_evidence_rows"], accepted
        ),
        "active_task_unknown_fraction_among_accepted": _fraction(
            counts["active_task_unknown_rows"], accepted
        ),
        "unsupported_rationale_fraction_among_accepted": _fraction(
            counts["unsupported_rationale_rows"], accepted
        ),
        "optional_context_accounting_fraction": (
            _fraction(counts["context_accounted_rows"], context_rows) if context_rows else None
        ),
        "projection_qc_coverage": (
            _fraction(counts["projection_qc_passed_rows"], projection_candidates)
            if projection_candidates
            else None
        ),
    }
    checks = (
        ("acceptance_fraction", fractions["acceptance_fraction"], limits.min_acceptance_fraction, "min"),
        ("reject_fraction", fractions["reject_fraction"], limits.max_reject_fraction, "max"),
        ("conflict_fraction", fractions["conflict_fraction_among_accepted"], limits.max_conflict_fraction, "max"),
        ("critical_unknown_fraction", fractions["critical_unknown_fraction_among_accepted"], limits.max_unknown_fraction, "max"),
        ("insufficient_evidence_fraction", fractions["insufficient_evidence_fraction_among_accepted"], limits.max_insufficient_fraction, "max"),
        ("active_task_unknown_fraction", fractions["active_task_unknown_fraction_among_accepted"], limits.max_task_unknown_fraction, "max"),
        ("unsupported_rationale_fraction", fractions["unsupported_rationale_fraction_among_accepted"], limits.max_unsupported_rationale_fraction, "max"),
    )
    violations = [
        f"{scope}:{name}_{'below_min' if mode == 'min' else 'above_max'}:{value:.6f}:{limit:.6f}"
        for name, value, limit, mode in checks
        if (mode == "min" and value < limit) or (mode == "max" and value > limit)
    ]
    context_coverage = fractions["optional_context_accounting_fraction"]
    if context_coverage is not None and context_coverage < limits.min_context_accounting_fraction:
        violations.append(
            f"{scope}:optional_context_accounting_below_min:"
            f"{context_coverage:.6f}:{limits.min_context_accounting_fraction:.6f}"
        )
    projection_coverage = fractions["projection_qc_coverage"]
    if projection_coverage is not None and projection_coverage < limits.min_projection_qc_coverage:
        violations.append(
            f"{scope}:projection_qc_coverage_below_min:"
            f"{projection_coverage:.6f}:{limits.min_projection_qc_coverage:.6f}"
        )
    return {
        "counts": dict(sorted(counts.items())),
        "fractions": fractions,
        "violations": violations,
        "passed": not violations,
    }


def _fraction(numerator: int, denominator: int) -> float:
    """Return a stable zero-safe fraction."""

    return float(numerator) / float(denominator) if denominator else 0.0


def _dataset_family(value: str) -> str:
    """Normalize one dataset alias to its canonical family."""

    normalized = str(value).strip().lower().replace("_", "-")
    for family in ("sidewalk", "foxglove", "frodobots", "egowalk"):
        if normalized.startswith(family):
            return family
    return normalized or "unknown"


__all__ = [
    "REASONING_V3_QUALITY_REPORT_VERSION",
    "ReasoningV3QualityThresholds",
    "evaluate_reasoning_v3_quality",
]
