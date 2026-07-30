"""Deterministic offline audits for Reasoning V4.6 artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_section_v4,
)
from mcp_server.agent.reasoning_v4_prompt_policy import (
    get_prompt_profile_v4,
)
from mcp_server.contracts.direction_profile import (
    affordance_probe_candidates_for_action,
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
    load_latest_action_adjudications_v4,
)
from mcp_server.data.reasoning_v4_review import (
    load_latest_action_review_diagnoses_v4,
)
from mcp_server.data.reasoning_v4_task_canonicalization import (
    canonicalize_task_instruction_v4,
)


def audit_reasoning_v4_annotation_completeness(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Audit source coverage and completed annotation records.

    This audit intentionally does not collapse section acceptance, candidate
    structure, review, or formal eligibility into record completeness.

    Args:
        source_jsonl: Frozen V4 source manifest.
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
        counts["candidate_structurally_valid"] += int(
            annotation.candidate_structure_record.status == "valid"
        )
        _accumulate_p0_annotation_gates_v4(
            counts,
            errors,
            source_id=source_id,
            source=source,
            annotation=annotation,
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
            "urbannav_reasoning_v4_6_annotation_completeness_v1"
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
            "candidate_structurally_valid": int(
                counts["candidate_structurally_valid"]
            ),
            "pending_human": int(
                counts["pending_human"]
            ),
            "formal_training_eligible": 0,
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
    diagnosis_jsonl: str | Path | None = None,
    adjudication_jsonl: str | Path | None = None,
    required_dataset_families: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Audit source, annotation, and view artifacts offline.

    Args:
        source_jsonl: Frozen source manifest.
        annotation_jsonl: Append-only annotation manifest.
        view_jsonl: Review-preview or formal-training views.
        diagnosis_jsonl: Optional append-only LVLM diagnoses.
        adjudication_jsonl: Optional append-only human adjudications.
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
    diagnoses = load_latest_action_review_diagnoses_v4(
        diagnosis_jsonl
    )
    adjudications = load_latest_action_adjudications_v4(
        adjudication_jsonl
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
        counts["candidate_structurally_valid"] += int(
            annotation.candidate_structure_record.status == "valid"
        )
        _accumulate_p0_annotation_gates_v4(
            counts,
            errors,
            source_id=source_id,
            source=source,
            annotation=annotation,
        )
        audit = annotation.action_reconciliation
        diagnosis = (
            diagnoses.get(audit.audit_id)
            if audit is not None
            else None
        )
        adjudication = (
            adjudications.get(audit.audit_id)
            if audit is not None
            else None
        )
        try:
            state = derive_reasoning_release_state_v4(
                source,
                annotation,
                diagnosis=diagnosis,
                adjudication=adjudication,
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
        counts["formal_training_eligible"] += int(
            state.formal_training_eligible
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
        counts["view_artifact_training_eligible"] += int(
            view.artifact_training_eligible
        )
        if (
            view.release_track == "review_preview"
            and view.artifact_training_eligible
        ):
            counts["preview_training_leak"] += 1
            errors.append(
                f"preview_training_leak:{view.sample_id}"
            )
        if (
            view.release_track == "formal_training"
            and (
                not view.artifact_training_eligible
                or not view.formal_training_eligible
            )
        ):
            counts["formal_ineligible_artifact_leak"] += 1
            errors.append(
                f"formal_ineligible_artifact_leak:{view.sample_id}"
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
            "urbannav_reasoning_v4_6_artifact_audit_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "diagnosis_jsonl": (
            str(diagnosis_jsonl)
            if diagnosis_jsonl is not None
            else None
        ),
        "adjudication_jsonl": (
            str(adjudication_jsonl)
            if adjudication_jsonl is not None
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
            "candidate_structurally_valid": int(
                counts["candidate_structurally_valid"]
            ),
            "pending_human": int(
                counts["pending_human"]
            ),
            "formal_training_eligible": int(
                counts["formal_training_eligible"]
            ),
            "task_canonicalization_drift": int(
                counts["task_canonicalization_drift"]
            ),
            "probe_tie_loss": int(counts["probe_tie_loss"]),
            "preview_training_leak": int(
                counts["preview_training_leak"]
            ),
            "formal_ineligible_artifact_leak": int(
                counts["formal_ineligible_artifact_leak"]
            ),
        },
        "counts": dict(counts),
        "errors": errors[:200],
        "warnings": warnings[:200],
        "passed": not errors,
    }


def audit_reasoning_v4_pilot(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    formal_view_jsonl: str | Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Audit a natural-distribution V4.6 pilot before full annotation.

    The audit reports review prevalence instead of requiring zero review
    rows.  A strong-moving/global-stop collision is an explicitly supported
    fail-closed review-only outcome: it may carry no candidate, but must stay
    pending and outside the formal manifest.  Every other invalid candidate
    remains a hard error.  Other hard gates cover completed records, tuple
    consistency, canonical vocabularies, exact seven-probe output, and
    lossless filtering into the formal-training manifest.

    Args:
        source_jsonl: Frozen natural-random pilot sources.
        annotation_jsonl: Completed pilot annotations.
        formal_view_jsonl: Formal-training exporter dry-run output.
        expected_rows: Exact pilot source count.

    Returns:
        Pilot metrics, trigger distributions, bounded errors, and pass flag.

    Raises:
        ValueError: If ``expected_rows`` is not positive.
    """

    expected = int(expected_rows)
    if expected <= 0:
        raise ValueError("expected_rows must be positive")
    errors: list[str] = []
    counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    normalization_counts: Counter[str] = Counter()
    probe_affordance_counts: Counter[str] = Counter()
    sources = _source_index_v4(source_jsonl, errors=errors)
    annotations, annotation_file_rows = _annotation_index_v4(
        annotation_jsonl,
        errors=errors,
    )
    source_ids = set(sources)
    annotation_ids = set(annotations)
    missing_annotations = sorted(source_ids - annotation_ids)
    orphan_annotations = sorted(annotation_ids - source_ids)
    formal_eligible_ids: set[str] = set()

    if len(sources) != expected:
        errors.append(
            f"source_row_count_mismatch:{len(sources)}!={expected}"
        )
    if missing_annotations:
        errors.append(
            f"missing_annotations:{len(missing_annotations)}"
        )
    if orphan_annotations:
        errors.append(
            f"annotations_without_source:{len(orphan_annotations)}"
        )

    for source_id in sorted(source_ids & annotation_ids):
        source = sources[source_id]
        annotation = annotations[source_id]
        counts["joined_rows"] += 1
        if annotation.source_fingerprint != source.source_fingerprint:
            errors.append(
                f"source_fingerprint_mismatch:{source_id}"
            )
            continue
        is_complete = annotation.record_status == "complete"
        all_sections_accepted = annotation.all_sections_accepted
        candidate_valid = (
            annotation.candidate_structure_record.status == "valid"
        )
        audit = annotation.action_reconciliation
        structure_errors = set(
            annotation.candidate_structure_record.errors
        )
        fail_closed_stop_conflict = (
            annotation.candidate_structure_record.status == "invalid"
            and structure_errors
            == {
                "candidate_construction_failed:"
                "strong_moving_action_conflicts_with_stop_constraint"
            }
            and audit is not None
            and "strong_action_constraint_conflict"
            in audit.review_trigger_codes
        )
        counts["record_complete"] += int(is_complete)
        counts["all_sections_accepted"] += int(
            all_sections_accepted
        )
        counts["candidate_structurally_valid"] += int(
            candidate_valid
        )
        counts["candidate_fail_closed_review_only"] += int(
            fail_closed_stop_conflict
        )
        if not is_complete:
            errors.append(f"record_not_complete:{source_id}")
        if not all_sections_accepted:
            errors.append(
                f"sections_not_all_accepted:{source_id}"
            )
        if not candidate_valid and not fail_closed_stop_conflict:
            counts["candidate_unexpected_invalid_rows"] += 1
            errors.append(
                f"candidate_not_structurally_valid:{source_id}"
            )

        if audit is not None:
            for code in audit.review_trigger_codes:
                trigger_counts[str(code)] += 1
            if audit.checks.action_tuple_consistency == "conflict":
                counts["action_tuple_conflict_rows"] += 1
                errors.append(
                    f"action_tuple_conflict:{source_id}"
                )
            if (
                audit.checks.constraint_effect_consistency
                == "conflict"
            ):
                counts["constraint_effect_conflict_rows"] += 1

        try:
            state = derive_reasoning_release_state_v4(
                source,
                annotation,
            )
        except ValueError as exc:
            errors.append(f"release_state:{source_id}:{exc}")
        else:
            counts["pending_human"] += int(
                state.review_status == "pending_human"
            )
            counts["formal_training_eligible"] += int(
                state.formal_training_eligible
            )
            if fail_closed_stop_conflict and (
                state.review_status != "pending_human"
                or state.formal_training_eligible
            ):
                errors.append(
                    "fail_closed_candidate_not_quarantined:"
                    f"{source_id}:review={state.review_status}:"
                    f"formal={state.formal_training_eligible}"
                )
            if state.formal_training_eligible:
                formal_eligible_ids.add(source_id)

        row_vocabulary_normalized = False
        for stage in annotation.stage_records.values():
            for warning in stage.warnings:
                if warning in {
                    "scene_vocabulary_normalized",
                    "constraint_vocabulary_normalized",
                }:
                    normalization_counts[warning] += 1
                    row_vocabulary_normalized = True
                    break
        counts["vocabulary_normalization_applied_rows"] += int(
            row_vocabulary_normalized
        )

        vocabulary_errors: list[str] = []
        scene = annotation.grounding_outputs.scene_grounding
        if scene is not None:
            scene_result = validate_reasoning_section_v4(
                "SceneGrounding",
                scene,
                observation=source.observation,
            )
            vocabulary_errors.extend(
                item
                for item in scene_result.errors
                if item.startswith(
                    (
                        "noncanonical_scene_entity_labels:",
                        "noncanonical_traffic_context:",
                    )
                )
            )
        constraints = (
            annotation.grounding_outputs
            .interaction_constraint_grounding
        )
        if constraints is not None:
            constraint_result = validate_reasoning_section_v4(
                "InteractionConstraintGrounding",
                constraints,
                observation=source.observation,
            )
            vocabulary_errors.extend(
                item
                for item in constraint_result.errors
                if item.startswith(
                    "noncanonical_constraint_kind:"
                )
            )
        if vocabulary_errors:
            counts[
                "post_normalization_vocab_violation_rows"
            ] += 1
            errors.append(
                "post_normalization_vocab_violation:"
                f"{source_id}:{vocabulary_errors}"
            )

        traversability = (
            annotation.grounding_outputs.local_traversability
        )
        probes = (
            list(traversability.direction_affordances)
            if traversability is not None
            else []
        )
        if len(probes) != 7:
            counts["probe_rows_not_exactly_seven"] += 1
            errors.append(
                f"probe_count_mismatch:{source_id}:{len(probes)}"
            )
            continue
        counts["probe_rows_exactly_seven"] += 1
        signatures = {
            (
                item.affordance,
                item.clearance,
                tuple(str(value) for value in item.limiting_factors),
            )
            for item in probes
        }
        affordances = [item.affordance for item in probes]
        counts["all_seven_probes_identical"] += int(
            len(signatures) == 1
        )
        counts["all_seven_probes_blocked"] += int(
            all(value == "blocked" for value in affordances)
        )
        counts["all_seven_probes_conditional"] += int(
            all(value == "conditional" for value in affordances)
        )
        for value in affordances:
            probe_affordance_counts[value] += 1

    formal_view_ids: set[str] = set()
    formal_view_sample_ids: set[str] = set()
    formal_contract_leak_ids: set[str] = set()
    for row in iter_jsonl(formal_view_jsonl):
        try:
            view = ReasoningTrainingViewV4.model_validate(row)
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(f"invalid_formal_view:{str(exc)[:500]}")
            continue
        if view.sample_id in formal_view_sample_ids:
            errors.append(
                f"duplicate_formal_view_sample_id:{view.sample_id}"
            )
            continue
        formal_view_sample_ids.add(view.sample_id)
        if view.source_group_id in formal_view_ids:
            errors.append(
                "duplicate_formal_view_source_group_id:"
                f"{view.source_group_id}"
            )
            continue
        formal_view_ids.add(view.source_group_id)
        counts["formal_view_rows"] += 1
        if (
            view.release_track != "formal_training"
            or not view.formal_training_eligible
            or not view.artifact_training_eligible
        ):
            formal_contract_leak_ids.add(view.source_group_id)
            errors.append(
                f"formal_ineligible_artifact_leak:{view.sample_id}"
            )

    leaked_ids = sorted(
        (formal_view_ids - formal_eligible_ids)
        | formal_contract_leak_ids
    )
    missing_eligible_ids = sorted(
        formal_eligible_ids - formal_view_ids
    )
    counts["formal_ineligible_artifact_leak"] = len(leaked_ids)
    counts["formal_eligible_missing_from_manifest"] = len(
        missing_eligible_ids
    )
    if leaked_ids:
        errors.append(
            f"formal_manifest_leaked_ids:{leaked_ids[:20]}"
        )
    if missing_eligible_ids:
        errors.append(
            "formal_manifest_missing_eligible_ids:"
            f"{missing_eligible_ids[:20]}"
        )

    denominator = max(1, expected)
    formal_denominator = max(1, counts["formal_view_rows"])
    return {
        "schema_version": (
            "urbannav_reasoning_v4_6_natural_pilot_audit_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "formal_view_jsonl": str(formal_view_jsonl),
        "expected_rows": expected,
        "annotation_file_rows": annotation_file_rows,
        "metrics": {
            "sources": len(sources),
            "record_complete": int(counts["record_complete"]),
            "all_sections_accepted": int(
                counts["all_sections_accepted"]
            ),
            "candidate_structurally_valid": int(
                counts["candidate_structurally_valid"]
            ),
            "candidate_fail_closed_review_only": int(
                counts["candidate_fail_closed_review_only"]
            ),
            "candidate_fail_closed_review_only_rate": (
                counts["candidate_fail_closed_review_only"]
                / denominator
            ),
            "candidate_unexpected_invalid_rows": int(
                counts["candidate_unexpected_invalid_rows"]
            ),
            "pending_human": int(counts["pending_human"]),
            "pending_human_rate": (
                counts["pending_human"] / denominator
            ),
            "formal_training_eligible": int(
                counts["formal_training_eligible"]
            ),
            "formal_view_rows": int(counts["formal_view_rows"]),
            "formal_ineligible_artifact_leak": int(
                counts["formal_ineligible_artifact_leak"]
            ),
            "formal_eligible_missing_from_manifest": int(
                counts[
                    "formal_eligible_missing_from_manifest"
                ]
            ),
            "formal_manifest_leakage_rate": (
                counts["formal_ineligible_artifact_leak"]
                / formal_denominator
            ),
            "action_tuple_conflict_rows": int(
                counts["action_tuple_conflict_rows"]
            ),
            "action_tuple_conflict_rate": (
                counts["action_tuple_conflict_rows"]
                / denominator
            ),
            "constraint_effect_conflict_rows": int(
                counts["constraint_effect_conflict_rows"]
            ),
            "constraint_effect_conflict_rate": (
                counts["constraint_effect_conflict_rows"]
                / denominator
            ),
            "vocabulary_normalization_applied_rows": int(
                counts["vocabulary_normalization_applied_rows"]
            ),
            "vocabulary_normalization_applied_rate": (
                counts["vocabulary_normalization_applied_rows"]
                / denominator
            ),
            "post_normalization_vocab_violation_rows": int(
                counts[
                    "post_normalization_vocab_violation_rows"
                ]
            ),
            "post_normalization_vocab_violation_rate": (
                counts[
                    "post_normalization_vocab_violation_rows"
                ]
                / denominator
            ),
            "probe_rows_exactly_seven": int(
                counts["probe_rows_exactly_seven"]
            ),
            "probe_rows_not_exactly_seven": int(
                counts["probe_rows_not_exactly_seven"]
            ),
            "all_seven_probes_identical_rate": (
                counts["all_seven_probes_identical"]
                / denominator
            ),
            "all_seven_probes_blocked_rate": (
                counts["all_seven_probes_blocked"]
                / denominator
            ),
            "all_seven_probes_conditional_rate": (
                counts["all_seven_probes_conditional"]
                / denominator
            ),
        },
        "review_trigger_counts": dict(
            sorted(trigger_counts.items())
        ),
        "normalization_warning_counts": dict(
            sorted(normalization_counts.items())
        ),
        "probe_affordance_counts": dict(
            sorted(probe_affordance_counts.items())
        ),
        "missing_annotation_examples": missing_annotations[:50],
        "orphan_annotation_examples": orphan_annotations[:50],
        "errors": errors[:200],
        "passed": not errors,
    }


def _accumulate_p0_annotation_gates_v4(
    counts: Counter[str],
    errors: list[str],
    *,
    source_id: str,
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
) -> None:
    """Count task-registry drift and nearest-probe tie loss.

    Args:
        counts: Mutable audit counters.
        errors: Mutable bounded error collection.
        source_id: Stable source identity.
        source: Frozen source row.
        annotation: Joined annotation row.

    Returns:
        None.
    """

    canonical_task = canonicalize_task_instruction_v4(
        source.observation.instruction
    )
    if (
        canonical_task is not None
        and annotation.grounding_outputs.task_grounding
        != canonical_task.task_grounding
    ):
        counts["task_canonicalization_drift"] += 1
        errors.append(
            f"task_canonicalization_drift:{source_id}"
        )

    audit = annotation.action_reconciliation
    direction = source.direction_supervision
    if audit is None or direction is None:
        return
    strong_label = str(direction.label_name)
    expected_labels = (
        ()
        if strong_label == "stop"
        else tuple(
            sorted(
                candidate.label
                for candidate in affordance_probe_candidates_for_action(
                    source.observation.direction_context.action_profile_id,
                    strong_label,
                )
            )
        )
    )
    actual_labels = tuple(
        str(item.direction_label)
        for item in audit.probe_projections
    )
    if actual_labels != expected_labels:
        counts["probe_tie_loss"] += 1
        errors.append(
            "probe_tie_loss:"
            f"{source_id}:{actual_labels}!={expected_labels}"
        )


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
