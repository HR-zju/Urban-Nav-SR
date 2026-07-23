"""Availability and provenance quality report for minimal Reasoning V3.4."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3, ReasoningSourceV3
from mcp_server.contracts.reasoning_v3 import LabelProvenanceEntryV3
from mcp_server.data.jsonl_io import iter_jsonl


REASONING_V3_QUALITY_REPORT_VERSION = "urbannav_reasoning_v3_4_quality_report_v1"
_REMOVED_THRESHOLD_NAMES = {
    "max_conflict_fraction",
    "max_unknown_fraction",
    "max_insufficient_fraction",
    "max_task_unknown_fraction",
    "max_unsupported_rationale_fraction",
    "min_projection_qc_coverage",
    "min_context_accounting_fraction",
}


@dataclass(frozen=True, init=False)
class ReasoningV3QualityThresholds:
    """Configure only artifact-completion gates.

    ``min_acceptance_fraction`` is retained as the public compatibility name,
    but V3.4 applies it to the union of accepted-target and observation-only
    annotations. Removed V3.3 calibration, fusion, projection, and subjective
    ontology thresholds are accepted as constructor keywords and ignored so
    older runners do not silently reactivate those claims.
    """

    min_acceptance_fraction: float
    max_reject_fraction: float

    def __init__(
        self,
        min_acceptance_fraction: float = 0.75,
        max_reject_fraction: float = 0.20,
        **removed_thresholds: Any,
    ) -> None:
        """Validate active thresholds and explicitly absorb removed keywords."""

        unknown = sorted(set(removed_thresholds) - _REMOVED_THRESHOLD_NAMES)
        if unknown:
            raise TypeError(f"unexpected Reasoning V3.4 quality thresholds: {unknown}")
        for name, value in {
            "min_acceptance_fraction": min_acceptance_fraction,
            "max_reject_fraction": max_reject_fraction,
            **removed_thresholds,
        }.items():
            if value is None:
                continue
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        object.__setattr__(self, "min_acceptance_fraction", float(min_acceptance_fraction))
        object.__setattr__(self, "max_reject_fraction", float(max_reject_fraction))

    def report_thresholds(self) -> dict[str, float]:
        """Return only thresholds that still affect V3.4 pass/fail."""

        return {
            "min_valid_annotation_fraction": self.min_acceptance_fraction,
            "max_failed_annotation_fraction": self.max_reject_fraction,
        }


def evaluate_reasoning_v3_quality(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    thresholds: ReasoningV3QualityThresholds | None = None,
    required_dataset_families: Iterable[str] = (),
) -> dict[str, Any]:
    """Report schema, evidence, provenance, and supervision availability.

    This report measures artifact coverage rather than semantic correctness.
    In particular, it makes no calibration, route-fusion, projection, tracking,
    or geometric-accuracy claim.
    """

    limits = thresholds or ReasoningV3QualityThresholds()
    sources: dict[str, ReasoningSourceV3] = {}
    family_by_id: dict[str, str] = {}
    counters: dict[str, Counter[str]] = {}
    overall_counts: Counter[str] = Counter()
    hard_violations: list[str] = []

    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV3.model_validate(row)
        source_id = source.model_input.source_group_id
        family = _dataset_family(source.model_input.dataset_source)
        family_counts = counters.setdefault(family, Counter())
        if source_id in sources:
            family_counts["duplicate_source_ids"] += 1
            overall_counts["duplicate_source_ids"] += 1
        sources[source_id] = source
        family_by_id[source_id] = family
        for counts in (family_counts, overall_counts):
            _count_source(source, counts)
        if _weak_lm_fields(source.label_provenance):
            hard_violations.append(f"source_weak_lm_provenance:{source_id}")

    latest: dict[str, ReasoningAnnotationV3] = {}
    for row in iter_jsonl(annotation_jsonl):
        annotation = ReasoningAnnotationV3.model_validate(row)
        latest[annotation.source_group_id] = annotation
        overall_counts["annotation_records"] += 1
        overall_counts[f"annotation_schema:{annotation.schema_version}"] += 1

    for source_id, source in sources.items():
        family_counts = counters[family_by_id[source_id]]
        annotation = latest.get(source_id)
        if annotation is None:
            for counts in (family_counts, overall_counts):
                counts["missing_annotation_rows"] += 1
            continue
        for counts in (family_counts, overall_counts):
            _count_latest_annotation(source, annotation, counts)
        if annotation.source_fingerprint != source.source_fingerprint:
            hard_violations.append(f"annotation_source_fingerprint_mismatch:{source_id}")
        if _weak_lm_fields(annotation.label_provenance):
            hard_violations.append(f"annotation_weak_lm_provenance:{source_id}")

    orphan_ids = set(latest) - set(sources)
    overall_counts["orphan_latest_annotations"] = len(orphan_ids)
    required = {_dataset_family(value) for value in required_dataset_families}
    missing_families = sorted(required - set(counters))

    per_dataset: dict[str, Any] = {}
    violations: list[str] = list(hard_violations)
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
        "thresholds": limits.report_thresholds(),
        "report_semantics": (
            "availability_only:no_calibration_projection_fusion_tracking_or_geometry_claim"
        ),
        "overall": overall,
        "per_dataset": per_dataset,
        "missing_required_datasets": missing_families,
        "violation_count": len(violations),
        "violations": violations,
        "passed": not violations,
    }


def _count_source(source: ReasoningSourceV3, counts: Counter[str]) -> None:
    """Count typed source, evidence, provenance, and sidecar availability."""

    model_input = source.model_input
    counts["source_rows"] += 1
    counts[f"source_schema:{source.schema_version}"] += 1
    counts["evidence_catalog_items"] += len(model_input.evidence_catalog)
    counts["rows_with_evidence_catalog"] += int(bool(model_input.evidence_catalog))
    for entry in model_input.evidence_catalog:
        counts[f"evidence_modality:{entry.modality}"] += 1
    for modality in sorted(set(model_input.available_modalities)):
        counts[f"source_modality:{modality}"] += 1

    direction_available = _direction_available(source)
    system1_available = _system1_available(source)
    counts["direction_available_rows"] += int(direction_available)
    counts["direction_unavailable_rows"] += int(not direction_available)
    counts["system1_available_rows"] += int(system1_available)
    counts["system1_unavailable_rows"] += int(not system1_available)
    counts["direction_and_system1_available_rows"] += int(
        direction_available and system1_available
    )
    _count_provenance(source.label_provenance, counts, prefix="source")
    counts["source_warning_items"] += len(source.warnings)
    counts["source_rows_with_warnings"] += int(bool(source.warnings))


def _count_latest_annotation(
    source: ReasoningSourceV3,
    annotation: ReasoningAnnotationV3,
    counts: Counter[str],
) -> None:
    """Count latest status, weak observations, and typed annotation provenance."""

    counts[f"status:{annotation.status}"] += 1
    counts[f"latest_annotation_schema:{annotation.schema_version}"] += 1
    counts["annotation_fingerprint_match_rows"] += int(
        annotation.source_fingerprint == source.source_fingerprint
    )
    if annotation.status == "accepted":
        counts["accepted_target_rows"] += 1
        _count_provenance(annotation.label_provenance, counts, prefix="annotation")
        if annotation.section_loss_mask is not None:
            for name, available in annotation.section_loss_mask.model_dump(
                mode="python"
            ).items():
                counts[f"weak_section_available:{name}"] += int(bool(available))
    elif annotation.status == "observation_only":
        counts["observation_only_rows"] += 1
    elif annotation.status in {"rejected", "error"}:
        counts["failed_annotation_rows"] += 1

    if (
        annotation.status in {"accepted", "observation_only"}
        and annotation.observation is not None
    ):
        _count_weak_observation(annotation, counts)


def _count_weak_observation(
    annotation: ReasoningAnnotationV3,
    counts: Counter[str],
) -> None:
    """Count weak-label availability without treating it as accuracy."""

    observation = annotation.observation
    if observation is None:
        return
    counts["weak_observation_rows"] += 1
    task = observation.task_semantic
    scene = observation.scene_grounding
    local = observation.local_traversability
    context = observation.navigation_context_grounding
    for name, applicable in {
        "task_grounding": task.applicable,
        "scene_grounding": scene.applicable,
        "local_traversability": local.applicable,
        "navigation_context_grounding": context.applicable,
    }.items():
        counts[f"weak_field_applicable:{name}"] += int(bool(applicable))
    counts["task_requirement_items"] += len(task.requirements)
    counts["task_referent_items"] += len(task.referents)
    counts["task_ambiguity_items"] += len(task.ambiguities)
    counts["scene_entity_items"] += len(scene.entities)
    counts["traffic_control_items"] += len(scene.traffic_controls)
    counts["traversability_region_items"] += len(local.regions)
    counts["navigation_source_grounding_items"] += len(context.source_groundings)

    rationale = annotation.decision_rationale
    if rationale is not None:
        counts["weak_rationale_rows"] += 1
        counts[f"weak_rationale_status:{rationale.rationale_status}"] += 1
        counts["interaction_items"] += len(
            rationale.interaction_and_rule_grounding.interactions
        )
        counts["traffic_rule_items"] += len(
            rationale.interaction_and_rule_grounding.traffic_rules
        )
        counts["decision_support_items"] += len(rationale.supporting_reasons)


def _count_provenance(
    provenance: Mapping[str, LabelProvenanceEntryV3],
    counts: Counter[str],
    *,
    prefix: str,
) -> None:
    """Count typed provenance categories and weak coverage."""

    if provenance:
        counts[f"{prefix}_rows_with_provenance"] += 1
    weak_present = False
    for field_path, entry in provenance.items():
        counts[f"{prefix}_provenance_items"] += 1
        counts[f"{prefix}_provenance_source_kind:{entry.source_kind}"] += 1
        counts[f"{prefix}_provenance_supervision:{entry.supervision}"] += 1
        counts[f"{prefix}_provenance_loss_channel:{entry.loss_channel}"] += 1
        if entry.supervision == "weak":
            weak_present = True
            counts[f"weak_provenance_field:{field_path}"] += 1
    counts[f"{prefix}_rows_with_weak_provenance"] += int(weak_present)


def _direction_available(source: ReasoningSourceV3) -> bool:
    """Return whether the source carries one strong canonical direction label."""

    fixed = source.teacher_supervision_context.get("fixed_labels")
    canonical = fixed.get("canonical_direction") if isinstance(fixed, Mapping) else None
    provenance = source.label_provenance.get("decision.direction")
    return bool(
        isinstance(canonical, Mapping)
        and provenance is not None
        and provenance.source_kind == "deterministic_adapter"
        and provenance.supervision == "strong"
        and provenance.student_inferable
        and provenance.loss_channel == "direction_choice"
    )


def _system1_available(source: ReasoningSourceV3) -> bool:
    """Return whether System1 has an eligible strong dataset sidecar."""

    provenance = source.label_provenance.get("system1.trajectory")
    return bool(
        source.model_input.system1_enabled
        and provenance is not None
        and provenance.source_kind == "dataset_gt"
        and provenance.supervision == "strong"
        and provenance.student_inferable
        and provenance.loss_channel == "system1_action"
    )


def _weak_lm_fields(
    provenance: Mapping[str, LabelProvenanceEntryV3],
) -> list[str]:
    """Return weak fields that still advertise removed language loss."""

    return sorted(
        field_path
        for field_path, entry in provenance.items()
        if entry.supervision == "weak" and entry.loss_channel != "none"
    )


def _summarize_counts(
    counts: Counter[str],
    *,
    limits: ReasoningV3QualityThresholds,
    scope: str,
) -> dict[str, Any]:
    """Compute availability fractions and the two completion gates."""

    source_rows = counts["source_rows"]
    accepted = counts["accepted_target_rows"]
    observation_only = counts["observation_only_rows"]
    valid = accepted + observation_only
    failed = counts["failed_annotation_rows"] + counts["missing_annotation_rows"]
    fractions = {
        "valid_annotation_fraction": _fraction(valid, source_rows),
        "accepted_target_fraction": _fraction(accepted, source_rows),
        "observation_only_fraction": _fraction(observation_only, source_rows),
        "failed_annotation_fraction": _fraction(failed, source_rows),
        "direction_available_fraction": _fraction(
            counts["direction_available_rows"], source_rows
        ),
        "system1_available_fraction": _fraction(
            counts["system1_available_rows"], source_rows
        ),
        "weak_observation_fraction_among_valid_annotations": _fraction(
            counts["weak_observation_rows"], valid
        ),
        "weak_provenance_fraction_among_accepted_targets": _fraction(
            counts["annotation_rows_with_weak_provenance"], accepted
        ),
    }
    violations: list[str] = []
    if fractions["valid_annotation_fraction"] < limits.min_acceptance_fraction:
        violations.append(
            f"{scope}:valid_annotation_fraction_below_min:"
            f"{fractions['valid_annotation_fraction']:.6f}:"
            f"{limits.min_acceptance_fraction:.6f}"
        )
    if fractions["failed_annotation_fraction"] > limits.max_reject_fraction:
        violations.append(
            f"{scope}:failed_annotation_fraction_above_max:"
            f"{fractions['failed_annotation_fraction']:.6f}:"
            f"{limits.max_reject_fraction:.6f}"
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
