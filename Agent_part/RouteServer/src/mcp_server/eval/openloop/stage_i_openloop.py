"""Formal Stage I open-loop evaluation for FullCoT, VQA, and Decision."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_section_v4,
    parse_reasoning_target_v4,
)
from mcp_server.agent.reasoning_v4_validation import (
    decision_constraint_errors_v4,
    validate_reasoning_target_v4,
)
from mcp_server.agent.vqa_v2_text import parse_vqa_answer_v2
from mcp_server.contracts.direction_profile import (
    direction_comparison_facts,
    direction_label_names,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.openloop_benchmark import OpenLoopReferenceV1
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionSupervisionV4,
    NavObservationV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    ReasoningTargetV4,
)
from mcp_server.contracts.vqa_v2 import (
    VQAQuestionV2,
    build_vqa_answer_v2,
)
from mcp_server.eval.openloop.openloop_common import (
    OpenLoopJoinedRowV1,
    coverage_metrics_v1,
    macro_mean_v1,
    parent_balanced_mean_v1,
)


STAGE_I_OPENLOOP_EVALUATOR_VERSION = (
    "urbannav_stage_i_openloop_evaluator_v2_0"
)

_COMMON_12_LABELS = (
    "stop",
    "straight",
    "left_10",
    "right_10",
    "left_20",
    "right_20",
    "left_30",
    "right_30",
    "left_40",
    "right_40",
    "left_50plus",
    "right_50plus",
)

_PROVENANCE_SCOPES = (
    "deterministic_gt",
    "human_gold",
    "lvlm_weak",
    "reference_annotation",
)
_UNORDERED_REASONING_PATHS = {
    "task_grounding.requirements",
    "task_grounding.ambiguities",
    "scene_grounding.key_entities",
}
@dataclass
class _FieldStats:
    """Accumulate full-denominator scalar and collection agreement."""

    rows: int = 0
    exact: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add_scalar(self, expected: Any, predicted: Any) -> None:
        """Add one normalized scalar comparison."""

        self.rows += 1
        self.exact += _normalize_scalar(expected) == _normalize_scalar(
            predicted
        )

    def add_multiset(self, expected: Any, predicted: Any) -> None:
        """Add one order-invariant, duplicate-preserving comparison."""

        expected_counter = _value_counter(expected)
        predicted_counter = _value_counter(predicted)
        self.rows += 1
        self.exact += expected_counter == predicted_counter
        self.true_positive += sum(
            (expected_counter & predicted_counter).values()
        )
        self.false_positive += sum(
            (predicted_counter - expected_counter).values()
        )
        self.false_negative += sum(
            (expected_counter - predicted_counter).values()
        )

    def as_dict(self) -> dict[str, Any]:
        """Return counts, exact accuracy, and collection P/R/F1."""

        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = (
            self.true_positive / precision_denominator
            if precision_denominator
            else None
        )
        recall = (
            self.true_positive / recall_denominator
            if recall_denominator
            else None
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        )
        return {
            "reference_count": self.rows,
            "exact_accuracy": self.exact / self.rows if self.rows else None,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def evaluate_stage_i_openloop_v1(
    rows: Iterable[OpenLoopJoinedRowV1],
) -> dict[str, Any]:
    """Evaluate a mixed Stage I artifact on independent scorecards.

    Args:
        rows: Full-reference joined rows containing only Stage I tasks.

    Returns:
        FullCoT, VQA, and Decision scorecards plus shared coverage.

    Raises:
        ValueError: If Stage II or unsupported task rows are present.
    """

    materialized = list(rows)
    invalid = [row.reference.sample_id for row in materialized if row.reference.stage != "stage_i"]
    if invalid:
        raise ValueError(f"Stage I evaluator received Stage II rows: {invalid[:8]}")
    by_task: dict[str, list[OpenLoopJoinedRowV1]] = defaultdict(list)
    for row in materialized:
        by_task[row.reference.task_type].append(row)
    unsupported = set(by_task) - {"full_cot", "vqa", "decision", "reasoning_dropout"}
    if unsupported:
        raise ValueError(f"unsupported Stage I task types: {sorted(unsupported)}")
    direct_task_types = set(by_task) & {"decision", "reasoning_dropout"}
    if len(direct_task_types) != 1:
        raise ValueError(
            "Stage I aggregate evaluator requires exactly one direct task type: "
            f"{sorted(direct_task_types)}"
        )
    direct_task_type = next(iter(direct_task_types))
    return {
        "evaluator_version": STAGE_I_OPENLOOP_EVALUATOR_VERSION,
        "coverage": coverage_metrics_v1(materialized),
        "full_cot": evaluate_full_cot_openloop_v1(by_task["full_cot"]),
        "vqa": evaluate_vqa_openloop_v1(by_task["vqa"]),
        "decision": evaluate_decision_openloop_v1(by_task[direct_task_type]),
        "direct_task_type": direct_task_type,
        "modality_robustness": {
            "status": "N/A",
            "reason": "V2.0 has no approved same-parent paired target reprojection",
        },
    }


def evaluate_stage_i_decision_only_v1(
    rows: Iterable[OpenLoopJoinedRowV1],
) -> dict[str, Any]:
    """Evaluate only the Stage I short-horizon Decision contract.

    FullCoT and VQA are training views and auxiliary generation gates for the
    current experiment.  They are intentionally excluded from this primary
    test scorecard.
    """

    materialized = list(rows)
    accepted_task_types = {"decision", "reasoning_dropout"}
    invalid = [
        row.reference.sample_id
        for row in materialized
        if row.reference.stage != "stage_i"
        or row.reference.task_type not in accepted_task_types
    ]
    if invalid:
        raise ValueError(
            "Direction-only evaluator received rows outside Stage I direct Decision views: "
            f"{invalid[:8]}"
        )
    task_types = {row.reference.task_type for row in materialized}
    if len(task_types) != 1:
        raise ValueError(f"Direction-only evaluator cannot mix direct task types: {sorted(task_types)}")
    return {
        "evaluator_version": STAGE_I_OPENLOOP_EVALUATOR_VERSION,
        "evaluation_scope": "stage_i_direction_only",
        "direct_task_type": next(iter(task_types)),
        "coverage": coverage_metrics_v1(materialized),
        "decision": evaluate_decision_openloop_v1(materialized),
    }


def evaluate_full_cot_openloop_v1(
    rows: Iterable[OpenLoopJoinedRowV1],
) -> dict[str, Any]:
    """Evaluate strict V4.6 FullCoT reliability and typed agreement.

    Args:
        rows: FullCoT reference rows with optional outputs/failures.

    Returns:
        Mechanical, typed-field, provenance, and diagnostic metrics.
    """

    materialized = list(rows)
    mechanical = Counter()
    fields: dict[str, _FieldStats] = defaultdict(_FieldStats)
    provenance_values: dict[str, list[float]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    constraint_consistent = 0
    observation_validator_rows = 0
    observation_validator_pass = 0
    embedded_direction_rows = 0
    embedded_direction_correct = 0

    for joined in materialized:
        reference = joined.reference
        expected = _reference_reasoning_target(reference)
        mechanical["reference"] += 1
        response = joined.prediction.response if joined.prediction else ""
        if joined.prediction is not None:
            mechanical["output"] += 1
        parsed = parse_reasoning_target_v4(response)
        predicted = parsed.target if parsed.valid else None
        if predicted is not None:
            mechanical["strict_parse"] += 1
            mechanical["six_section_complete"] += 1
            mechanical["schema_valid"] += 1
            mechanical["no_extra_text"] += 1
            if not decision_constraint_errors_v4(
                predicted.reasoning.decision,
                predicted.reasoning.interaction_constraint_grounding,
            ):
                constraint_consistent += 1
        elif len(errors) < 100:
            errors.append(
                {
                    "sample_id": reference.sample_id,
                    "errors": list(parsed.errors) if response else ["missing_output"],
                }
            )

        observation_payload = reference.target.get("observation")
        if observation_payload is not None and predicted is not None:
            observation_validator_rows += 1
            observation = NavObservationV4.model_validate(observation_payload)
            supervision = (
                DirectionSupervisionV4.model_validate(
                    reference.direction_stamp.model_dump(mode="json")
                )
                if reference.direction_stamp is not None
                else None
            )
            validation = validate_reasoning_target_v4(
                predicted,
                observation=observation,
                direction_supervision=supervision,
            )
            observation_validator_pass += validation.valid

        expected_flat = _flatten_reasoning_target(expected)
        predicted_flat = (
            _flatten_reasoning_target(predicted) if predicted is not None else {}
        )
        row_field_scores: dict[str, list[float]] = defaultdict(list)
        for path, expected_value in expected_flat.items():
            predicted_value = predicted_flat.get(path)
            if path in _UNORDERED_REASONING_PATHS or path.endswith(
                ".limiting_factors"
            ) or path.endswith(".affected_direction_labels") or path == (
                "interaction_constraint_grounding.constraints"
            ):
                fields[path].add_multiset(expected_value, predicted_value or [])
                correct = _value_counter(expected_value) == _value_counter(
                    predicted_value or []
                )
            else:
                fields[path].add_scalar(expected_value, predicted_value)
                correct = _normalize_scalar(expected_value) == _normalize_scalar(
                    predicted_value
                )
            provenance = reference.field_provenance.get(
                path,
                reference.label_provenance,
            )
            provenance_values[provenance].append(float(correct))
            row_field_scores[provenance].append(float(correct))

        if reference.direction_stamp is not None:
            embedded_direction_rows += 1
            if predicted is not None:
                embedded_direction_correct += (
                    str(predicted.reasoning.decision.selected_action_label)
                    == str(reference.direction_stamp.label_name)
                )

    denominator = len(materialized) or 1
    return {
        "reference_count": len(materialized),
        "mechanical": {
            "output_coverage": mechanical["output"] / denominator,
            "strict_parse_rate": mechanical["strict_parse"] / denominator,
            "six_section_completeness": mechanical["six_section_complete"] / denominator,
            "schema_valid_rate": mechanical["schema_valid"] / denominator,
            "no_extra_text_rate": mechanical["no_extra_text"] / denominator,
            "observation_validator_pass_rate": (
                observation_validator_pass / observation_validator_rows
                if observation_validator_rows
                else None
            ),
            "observation_validator_coverage": observation_validator_rows / denominator,
            "constraint_decision_consistency": constraint_consistent / denominator,
        },
        "field_metrics": {
            path: stats.as_dict() for path, stats in sorted(fields.items())
        },
        "semantic_scopes": {
            scope: {
                "field_instances": len(provenance_values[scope]),
                "field_macro_agreement": (
                    sum(provenance_values[scope]) / len(provenance_values[scope])
                    if provenance_values[scope]
                    else None
                ),
                "claim": _provenance_claim(scope),
            }
            for scope in _PROVENANCE_SCOPES
        },
        "embedded_direction_diagnostic": {
            "headline": False,
            "reason": "selected_action_label LM span is masked in FullCoT",
            "reference_count": embedded_direction_rows,
            "accuracy": (
                embedded_direction_correct / embedded_direction_rows
                if embedded_direction_rows
                else None
            ),
        },
        "errors": errors,
    }


def evaluate_vqa_openloop_v1(
    rows: Iterable[OpenLoopJoinedRowV1],
) -> dict[str, Any]:
    """Evaluate VQA on the full reference set with provenance separation.

    Args:
        rows: VQA reference rows with optional outputs/failures.

    Returns:
        Per-task typed metrics and provenance-scoped parent-balanced macros.
    """

    materialized = list(rows)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_fields: dict[str, dict[str, _FieldStats]] = defaultdict(
        lambda: defaultdict(_FieldStats)
    )
    scoped_task_parent: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    errors: list[dict[str, Any]] = []
    for joined in materialized:
        reference = joined.reference
        question, expected = _reference_vqa(reference)
        task_id = question.task_id
        provenance = reference.label_provenance
        counts = task_counts[task_id]
        counts["reference"] += 1
        if joined.prediction is not None:
            counts["prediction"] += 1
        response = joined.prediction.response if joined.prediction else ""
        parsed = parse_vqa_answer_v2(
            response,
            expected_task_id=task_id,
            question=question,
        )
        predicted = parsed.answer
        if predicted is not None:
            counts["parsed"] += 1
        elif len(errors) < 100:
            errors.append(
                {
                    "sample_id": reference.sample_id,
                    "task_id": task_id,
                    "errors": list(parsed.errors) if response else ["missing_output"],
                }
            )
        expected_payload = expected.payload.model_dump(mode="json")
        predicted_payload = (
            predicted.payload.model_dump(mode="json")
            if predicted is not None
            else {}
        )
        exact = _normalize_vqa_payload(expected_payload) == _normalize_vqa_payload(
            predicted_payload
        )
        counts["exact"] += exact
        counts[f"provenance:{provenance}"] += 1
        scoped_task_parent[provenance][task_id].append(
            (reference.parent_sample_id, float(exact))
        )
        for field_name, expected_value in expected_payload.items():
            predicted_value = predicted_payload.get(field_name)
            if isinstance(expected_value, list):
                task_fields[task_id][field_name].add_multiset(
                    expected_value,
                    predicted_value or [],
                )
            else:
                task_fields[task_id][field_name].add_scalar(
                    expected_value,
                    predicted_value,
                )

    task_metrics: dict[str, Any] = {}
    for task_id, counts in sorted(task_counts.items()):
        denominator = counts["reference"] or 1
        task_metrics[task_id] = {
            "reference_count": counts["reference"],
            "prediction_count": counts["prediction"],
            "parse_count": counts["parsed"],
            "parse_rate_full_reference": counts["parsed"] / denominator,
            "strict_exact_full_reference": counts["exact"] / denominator,
            "provenance_counts": {
                scope: counts[f"provenance:{scope}"]
                for scope in _PROVENANCE_SCOPES
            },
            "fields": {
                field_name: values.as_dict()
                for field_name, values in sorted(task_fields[task_id].items())
            },
        }

    scope_metrics: dict[str, Any] = {}
    for scope in _PROVENANCE_SCOPES:
        by_task = scoped_task_parent[scope]
        task_parent_scores = {
            task_id: parent_balanced_mean_v1(values)
            for task_id, values in sorted(by_task.items())
        }
        scope_metrics[scope] = {
            "claim": _provenance_claim(scope),
            "task_count": len(task_parent_scores),
            "task_parent_balanced": task_parent_scores,
            "primary_task_macro": macro_mean_v1(task_parent_scores.values()),
        }
    return {
        "reference_count": len(materialized),
        "task_metrics": task_metrics,
        "provenance_scorecards": scope_metrics,
        "unscoped_overall_accuracy": {
            "status": "N/A",
            "reason": "strong and weak VQA labels must not be pooled",
        },
        "errors": errors,
    }


def evaluate_decision_openloop_v1(
    rows: Iterable[OpenLoopJoinedRowV1],
) -> dict[str, Any]:
    """Evaluate standalone Decision outputs in each native profile.

    Args:
        rows: Decision-only reference rows with optional outputs/failures.

    Returns:
        Profile-native classification, stop, and angular diagnostics.

    Raises:
        ValueError: If a reference lacks a canonical direction stamp.
    """

    grouped: dict[str, list[OpenLoopJoinedRowV1]] = defaultdict(list)
    for row in rows:
        stamp = row.reference.direction_stamp
        if stamp is None:
            raise ValueError(
                f"Decision reference requires direction stamp: {row.reference.sample_id}"
            )
        grouped[stamp.profile_id].append(row)
    profiles = {
        profile_id: _evaluate_decision_profile(profile_id, profile_rows)
        for profile_id, profile_rows in sorted(grouped.items())
    }
    return {
        "reference_count": sum(len(values) for values in grouped.values()),
        "primary_aggregation": "native_profile_macro",
        "profiles": profiles,
        "native_profile_macro_accuracy": macro_mean_v1(
            value["strict_direction_accuracy"] for value in profiles.values()
        ),
    }


def _evaluate_decision_profile(
    profile_id: str,
    rows: list[OpenLoopJoinedRowV1],
) -> dict[str, Any]:
    """Compute one profile-native Decision scorecard."""

    profile = resolve_stage_a_direction_profile(profile_id)
    labels = direction_label_names(profile_id)
    confusion: dict[str, Counter[str]] = {label: Counter() for label in labels}
    parsed_count = 0
    correct = 0
    moving_rows = 0
    moving_correct = 0
    stop_tp = stop_fp = stop_fn = 0
    exact_angle_reference = 0
    exact_angle_defined = 0
    angular_errors: list[float] = []
    range_rows = 0
    range_correct = 0
    within_one_bin = 0
    within_two_bins = 0
    common12_correct = 0
    common12_confusion: dict[str, Counter[str]] = {
        label: Counter() for label in _COMMON_12_LABELS
    }
    speed_correct = 0
    response_correct = 0
    tuple_correct = 0
    tuple_valid = 0
    constraint_consistent = 0
    errors: list[dict[str, Any]] = []

    for joined in rows:
        reference = joined.reference
        stamp = reference.direction_stamp
        assert stamp is not None
        expected_label = str(stamp.label_name)
        expected_decision = _reference_decision(reference)
        response = joined.prediction.response if joined.prediction else ""
        parsed = parse_reasoning_section_v4(response, "Decision")
        predicted = parsed.section if isinstance(parsed.section, DecisionV4) else None
        predicted_label = str(predicted.selected_action_label) if predicted else "__invalid__"
        if predicted is not None and predicted_label in labels:
            parsed_count += 1
            tuple_valid += 1
        else:
            predicted = None
            if len(errors) < 100:
                errors.append(
                    {
                        "sample_id": reference.sample_id,
                        "errors": list(parsed.errors) if response else ["missing_output"],
                    }
                )
        confusion[expected_label][predicted_label] += 1
        is_correct = predicted_label == expected_label
        correct += is_correct
        expected_common12 = _common12_direction_label(expected_label)
        predicted_common12 = (
            _common12_direction_label(predicted_label)
            if predicted is not None
            else "__invalid__"
        )
        common12_confusion[expected_common12][predicted_common12] += 1
        common12_correct += predicted_common12 == expected_common12
        if expected_label != "stop":
            moving_rows += 1
            moving_correct += is_correct
        if expected_label == "stop" and predicted_label == "stop":
            stop_tp += 1
        elif expected_label != "stop" and predicted_label == "stop":
            stop_fp += 1
        elif expected_label == "stop" and predicted_label != "stop":
            stop_fn += 1

        expected_index = labels.index(expected_label)
        expected_angle = profile["angles"][expected_index]
        is_range_reference = expected_label.endswith("50plus")
        if is_range_reference:
            range_rows += 1
            range_correct += is_correct
        elif expected_angle is not None:
            exact_angle_reference += 1
            if predicted is not None and not predicted_label.endswith("50plus") and predicted_label != "stop":
                predicted_index = labels.index(predicted_label)
                predicted_angle = profile["angles"][predicted_index]
                if predicted_angle is not None:
                    exact_angle_defined += 1
                    angular_errors.append(abs(float(predicted_angle) - float(expected_angle)))
        if predicted is not None:
            try:
                facts = direction_comparison_facts(
                    profile_id,
                    expected_label,
                    predicted_label,
                )
                within_one_bin += (
                    facts.angular_delta_deg is not None
                    and facts.angular_delta_deg <= 10.0
                ) or is_correct
                within_two_bins += (
                    facts.angular_delta_deg is not None
                    and facts.angular_delta_deg <= 20.0
                ) or is_correct
            except ValueError:
                pass
            speed_correct += predicted.speed == expected_decision.speed
            response_correct += predicted.response == expected_decision.response
            tuple_correct += predicted == expected_decision
            constraints_payload = reference.target.get("constraints")
            if constraints_payload is None:
                constraint_consistent += 1
            else:
                from mcp_server.contracts.reasoning_v4 import InteractionConstraintGroundingV4

                constraints = InteractionConstraintGroundingV4.model_validate(constraints_payload)
                constraint_consistent += not decision_constraint_errors_v4(predicted, constraints)

    denominator = len(rows) or 1
    stop_precision = stop_tp / (stop_tp + stop_fp) if stop_tp + stop_fp else None
    stop_recall = stop_tp / (stop_tp + stop_fn) if stop_tp + stop_fn else None
    stop_f1 = (
        2.0 * stop_precision * stop_recall / (stop_precision + stop_recall)
        if stop_precision is not None
        and stop_recall is not None
        and stop_precision + stop_recall
        else None
    )
    per_class = _classification_metrics(labels, confusion)
    common12_per_class = _classification_metrics(
        _COMMON_12_LABELS,
        common12_confusion,
    )
    return {
        "reference_count": len(rows),
        "strict_parse_rate": parsed_count / denominator,
        "strict_direction_accuracy": correct / denominator,
        "macro_f1": macro_mean_v1(value["f1"] for value in per_class.values()),
        "balanced_accuracy": macro_mean_v1(value["recall"] for value in per_class.values()),
        "per_class": per_class,
        "confusion_matrix": {
            label: dict(sorted(values.items())) for label, values in confusion.items()
        },
        "stop": {
            "precision": stop_precision,
            "recall": stop_recall,
            "f1": stop_f1,
            "accuracy": stop_recall,
        },
        "moving_accuracy": moving_correct / moving_rows if moving_rows else None,
        "non_stop_accuracy": (
            moving_correct / moving_rows if moving_rows else None
        ),
        "exact_center_angular": {
            "reference_count": exact_angle_reference,
            "defined_count": exact_angle_defined,
            "coverage": exact_angle_defined / exact_angle_reference if exact_angle_reference else None,
            "mae_deg": sum(angular_errors) / len(angular_errors) if angular_errors else None,
            "median_deg": _percentile(angular_errors, 0.5),
            "p90_deg": _percentile(angular_errors, 0.9),
        },
        "range_50plus": {
            "reference_count": range_rows,
            "accuracy": range_correct / range_rows if range_rows else None,
        },
        "within_one_bin_accuracy": within_one_bin / denominator,
        "within_10_accuracy": within_one_bin / denominator,
        "within_20_accuracy": within_two_bins / denominator,
        "common_12way": {
            "strict_direction_accuracy": common12_correct / denominator,
            "macro_f1": macro_mean_v1(
                value["f1"] for value in common12_per_class.values()
            ),
            "balanced_accuracy": macro_mean_v1(
                value["recall"] for value in common12_per_class.values()
            ),
            "per_class": common12_per_class,
            "confusion_matrix": {
                label: dict(sorted(values.items()))
                for label, values in common12_confusion.items()
                if sum(values.values())
            },
        },
        "speed_agreement": speed_correct / denominator,
        "response_agreement": response_correct / denominator,
        "exact_tuple_agreement": tuple_correct / denominator,
        "tuple_validity": tuple_valid / denominator,
        "constraint_consistency": constraint_consistent / denominator,
        "errors": errors,
    }


def _common12_direction_label(label: str) -> str:
    """Project a native degree-10 label into the shared 12-way space."""

    value = str(label)
    if value in {"stop", "straight"}:
        return value
    side, magnitude = value.split("_", 1)
    if magnitude == "50plus":
        return value
    angle = int(magnitude)
    return f"{side}_{angle}" if angle <= 40 else f"{side}_50plus"


def _classification_metrics(
    labels: tuple[str, ...],
    confusion: dict[str, Counter[str]],
) -> dict[str, dict[str, Any]]:
    """Compute supported per-class precision, recall, and F1."""

    result: dict[str, dict[str, Any]] = {}
    for label in labels:
        support = sum(confusion[label].values())
        if not support:
            continue
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        result[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def _reference_reasoning_target(reference: OpenLoopReferenceV1) -> ReasoningTargetV4:
    """Load one canonical Reasoning target from a hidden reference."""

    payload = reference.target.get("reasoning_target", reference.target)
    return ReasoningTargetV4.model_validate(payload)


def _reference_decision(reference: OpenLoopReferenceV1) -> DecisionV4:
    """Load one canonical standalone Decision target."""

    payload = reference.target.get("decision", reference.target)
    return DecisionV4.model_validate(payload)


def _reference_vqa(reference: OpenLoopReferenceV1) -> tuple[VQAQuestionV2, Any]:
    """Load one VQA question and typed answer from a hidden reference."""

    question = VQAQuestionV2.model_validate(reference.target["question"])
    answer_payload = reference.target["answer"]
    if isinstance(answer_payload, dict) and "payload" in answer_payload:
        task_id = str(answer_payload.get("task_id") or question.task_id)
        payload = answer_payload["payload"]
    else:
        task_id = question.task_id
        payload = answer_payload
    return question, build_vqa_answer_v2(task_id, payload)


def _flatten_reasoning_target(target: ReasoningTargetV4) -> dict[str, Any]:
    """Flatten typed V4.6 fields using stable semantic keys."""

    trace = target.reasoning.model_dump(mode="json")
    result: dict[str, Any] = {}
    for section_name in (
        "task_grounding",
        "scene_grounding",
        "navigation_context_grounding",
        "decision",
    ):
        _flatten_mapping(section_name, trace[section_name], result)
    affordances = trace["local_traversability"]["direction_affordances"]
    for item in affordances:
        label = str(item["direction_label"])
        for field_name in ("affordance", "clearance", "limiting_factors"):
            result[f"local_traversability.{label}.{field_name}"] = item[field_name]
    constraints = trace["interaction_constraint_grounding"]["constraints"]
    result["interaction_constraint_grounding.constraints"] = constraints
    return result


def _flatten_mapping(prefix: str, value: dict[str, Any], output: dict[str, Any]) -> None:
    """Flatten nested mappings while preserving list values."""

    for key, item in value.items():
        path = f"{prefix}.{key}"
        if isinstance(item, dict):
            _flatten_mapping(path, item, output)
        else:
            output[path] = item


def _normalize_scalar(value: Any) -> Any:
    """Normalize open text/codes without changing categorical meaning."""

    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return value


def _canonical_json(value: Any) -> str:
    """Return stable JSON for one collection element."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _value_counter(value: Any) -> Counter[str]:
    """Convert a list or scalar to an order-invariant multiset."""

    values = value if isinstance(value, list) else ([] if value is None else [value])
    return Counter(_canonical_json(item) for item in values)


def _normalize_vqa_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize VQA lists as duplicate-preserving unordered collections."""

    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            normalized[key] = sorted(_canonical_json(item) for item in value)
        elif isinstance(value, str):
            normalized[key] = _normalize_scalar(value)
        else:
            normalized[key] = value
    return normalized


def _provenance_claim(scope: str) -> str:
    """Return the permitted scientific claim for one reference provenance."""

    return {
        "deterministic_gt": "objective_accuracy",
        "human_gold": "human_gold_accuracy",
        "lvlm_weak": "teacher_agreement",
        "reference_annotation": "reference_agreement",
    }[scope]


def _percentile(values: list[float], quantile: float) -> float | None:
    """Compute a deterministic linearly interpolated percentile."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = [
    "STAGE_I_OPENLOOP_EVALUATOR_VERSION",
    "evaluate_decision_openloop_v1",
    "evaluate_full_cot_openloop_v1",
    "evaluate_stage_i_decision_only_v1",
    "evaluate_stage_i_openloop_v1",
    "evaluate_vqa_openloop_v1",
]
