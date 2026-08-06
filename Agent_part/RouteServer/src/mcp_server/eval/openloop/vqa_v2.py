"""Deterministic offline evaluation for Navigation VQA V2 predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcp_server.agent.vqa_v2_text import parse_vqa_answer_v2
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQACapabilityConfigV2,
)
from mcp_server.data.io.jsonl import iter_jsonl


def evaluate_vqa_predictions_v2(
    annotation_jsonl: str | Path,
    prediction_jsonl: str | Path,
    *,
    capabilities: VQACapabilityConfigV2,
) -> dict[str, Any]:
    """Evaluate predictions by stable annotation id.

    Args:
        annotation_jsonl: Frozen accepted reference annotations.
        prediction_jsonl: Rows with annotation_id and response.
        capabilities: Versioned evaluation activation policy.

    Returns:
        Parse, exact-match, direction, field, and per-task metrics.

    Raises:
        ValueError: If ids are duplicated or predictions target dormant tasks.
    """

    references = _annotation_index(annotation_jsonl)
    predictions = _prediction_index(prediction_jsonl)
    counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, Any]] = []
    for annotation_id, reference in references.items():
        counts["reference_rows"] += 1
        task_id = reference.question.task_id
        capability = capabilities.capability(task_id)
        if not capability.evaluation_enabled:
            raise ValueError(
                f"dormant task has a reference annotation: {task_id}"
            )
        prediction = predictions.get(annotation_id)
        if prediction is None:
            counts["missing_prediction_rows"] += 1
            task_counts[task_id]["missing"] += 1
            continue
        counts["prediction_rows"] += 1
        task_counts[task_id]["rows"] += 1
        parsed = parse_vqa_answer_v2(
            str(prediction.get("response") or ""),
            expected_task_id=task_id,
            question=reference.question,
        )
        if not parsed.valid:
            counts["invalid_parse_rows"] += 1
            task_counts[task_id]["invalid_parse"] += 1
            if len(errors) < 200:
                errors.append(
                    {
                        "annotation_id": annotation_id,
                        "task_id": task_id,
                        "errors": parsed.errors,
                    }
                )
            continue
        counts["canonical_parse_rows"] += 1
        task_counts[task_id]["canonical_parse"] += 1
        predicted_payload = parsed.answer.payload.model_dump(
            mode="json"
        )
        reference_payload = reference.answer.payload.model_dump(
            mode="json"
        )
        if predicted_payload == reference_payload:
            counts["exact_match_rows"] += 1
            task_counts[task_id]["exact"] += 1
        for field_name, reference_value in reference_payload.items():
            field_key = f"{task_id}.{field_name}"
            field_counts[field_key]["rows"] += 1
            if predicted_payload.get(field_name) == reference_value:
                field_counts[field_key]["correct"] += 1
        direction_field = _direction_field(reference_payload)
        if direction_field is not None:
            counts["direction_rows"] += 1
            task_counts[task_id]["direction_rows"] += 1
            if (
                predicted_payload.get(direction_field)
                == reference_payload.get(direction_field)
            ):
                counts["direction_correct_rows"] += 1
                task_counts[task_id]["direction_correct"] += 1
        scope = (
            "teacher_agreement"
            if reference.label_source == "lvlm_weak"
            else "derived_or_external_gt"
        )
        counts[f"scope:{scope}:rows"] += 1
        if predicted_payload == reference_payload:
            counts[f"scope:{scope}:exact"] += 1
    total = int(counts["reference_rows"])
    parsed_total = int(counts["canonical_parse_rows"])
    task_metrics = {}
    for capability in capabilities.tasks:
        if not capability.evaluation_enabled:
            task_metrics[capability.task_id] = {
                "status": "N/A",
                "reason": str(capability.disabled_reason),
            }
            continue
        task = task_counts[capability.task_id]
        rows = int(task["rows"])
        task_metrics[capability.task_id] = {
            "status": "available",
            "reference_rows": rows + int(task["missing"]),
            "prediction_rows": rows,
            "canonical_parse_fraction": (
                task["canonical_parse"] / rows if rows else None
            ),
            "exact_match_fraction": (
                task["exact"] / rows if rows else None
            ),
            "direction_accuracy": (
                task["direction_correct"]
                / task["direction_rows"]
                if task["direction_rows"]
                else None
            ),
        }
    return {
        "schema_version": "urbannav_navigation_vqa_evaluation_v2",
        "capability_config_digest": capabilities.digest(),
        "counts": dict(counts),
        "coverage": (
            counts["prediction_rows"] / total if total else 0.0
        ),
        "canonical_parse_fraction": (
            counts["canonical_parse_rows"]
            / counts["prediction_rows"]
            if counts["prediction_rows"]
            else 0.0
        ),
        "exact_match_fraction": (
            counts["exact_match_rows"] / parsed_total
            if parsed_total
            else 0.0
        ),
        "direction_accuracy": (
            counts["direction_correct_rows"]
            / counts["direction_rows"]
            if counts["direction_rows"]
            else None
        ),
        "task_metrics": task_metrics,
        "field_metrics": {
            field: {
                "rows": int(values["rows"]),
                "accuracy": (
                    values["correct"] / values["rows"]
                    if values["rows"]
                    else None
                ),
            }
            for field, values in sorted(field_counts.items())
        },
        "dense_metrics": {
            "miou": "N/A",
            "sam_som_grounding_accuracy": "N/A",
            "tracking_accuracy": "N/A",
            "ttc_error": "N/A",
            "route_projection_error": "N/A",
        },
        "errors": errors,
    }


def _annotation_index(
    path: str | Path,
) -> dict[str, VQAAnnotationV2]:
    """Load unique accepted annotations by stable id."""

    result: dict[str, VQAAnnotationV2] = {}
    for row in iter_jsonl(path):
        annotation = VQAAnnotationV2.model_validate(row)
        if annotation.annotation_id in result:
            raise ValueError(
                f"duplicate annotation id: {annotation.annotation_id}"
            )
        result[annotation.annotation_id] = annotation
    return result


def _prediction_index(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load unique prediction rows by annotation/sample id."""

    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        annotation_id = str(
            row.get("annotation_id")
            or row.get("sample_id")
            or ""
        )
        if not annotation_id:
            raise ValueError("prediction requires annotation_id")
        if annotation_id in result:
            raise ValueError(
                f"duplicate prediction id: {annotation_id}"
            )
        result[annotation_id] = row
    return result


def _direction_field(payload: dict[str, Any]) -> str | None:
    """Return the primary selected-direction field when present."""

    for field_name in (
        "direction_label",
        "preferred_direction_label",
        "alternative_direction_label",
        "future_direction_label",
    ):
        if payload.get(field_name) is not None:
            return field_name
    return None


__all__ = ["evaluate_vqa_predictions_v2"]
