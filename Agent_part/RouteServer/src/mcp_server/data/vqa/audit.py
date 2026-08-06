"""Deterministic offline audits for Navigation VQA V2 artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.agent.vqa_v2_text import parse_vqa_answer_v2
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQACapabilityConfigV2,
    VQASourceV2,
    VQATrainingViewV2,
)
from mcp_server.data.io.jsonl import iter_jsonl


def audit_vqa_artifacts_v2(
    *,
    capabilities: VQACapabilityConfigV2,
    source_jsonl: str | Path | None = None,
    annotation_jsonl: str | Path | None = None,
    training_view_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Audit schema, task activation, parsing, and unavailable metrics.

    Args:
        capabilities: Versioned activation policy.
        source_jsonl: Optional frozen source artifact.
        annotation_jsonl: Optional annotation artifact.
        training_view_jsonl: Optional training-view artifact.

    Returns:
        Offline deterministic audit report.
    """

    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    if source_jsonl is not None:
        for index, row in enumerate(iter_jsonl(source_jsonl)):
            try:
                source = VQASourceV2.model_validate(row)
            except (TypeError, ValueError) as exc:
                errors.append(
                    {
                        "artifact": "source",
                        "index": index,
                        "error": str(exc),
                    }
                )
                continue
            counts["source_rows"] += 1
            counts[
                f"source_dataset:{source.observation.dataset_source}"
            ] += 1
    if annotation_jsonl is not None:
        for index, row in enumerate(iter_jsonl(annotation_jsonl)):
            try:
                annotation = VQAAnnotationV2.model_validate(row)
                capability = capabilities.capability(
                    annotation.question.task_id
                )
                if not capability.training_export_enabled:
                    raise ValueError(
                        "dormant task produced an annotation"
                    )
            except (TypeError, ValueError) as exc:
                errors.append(
                    {
                        "artifact": "annotation",
                        "index": index,
                        "error": str(exc),
                    }
                )
                continue
            counts["annotation_rows"] += 1
            counts[
                f"annotation_task:{annotation.question.task_id}"
            ] += 1
    if training_view_jsonl is not None:
        for index, row in enumerate(iter_jsonl(training_view_jsonl)):
            try:
                view = VQATrainingViewV2.model_validate(row)
                capability = capabilities.capability(
                    view.question.task_id
                )
                if not capability.training_export_enabled:
                    raise ValueError(
                        "dormant task produced a training view"
                    )
                parsed = parse_vqa_answer_v2(
                    view.assistant_text,
                    expected_task_id=view.question.task_id,
                    question=view.question,
                )
                if not parsed.valid:
                    raise ValueError(
                        "assistant target is not canonical: "
                        f"{parsed.errors}"
                    )
            except (TypeError, ValueError) as exc:
                errors.append(
                    {
                        "artifact": "training_view",
                        "index": index,
                        "error": str(exc),
                    }
                )
                continue
            counts["training_view_rows"] += 1
            counts[
                f"training_task:{view.question.task_id}"
            ] += 1
    task_metrics = {
        item.task_id: (
            {"status": "available"}
            if item.evaluation_enabled
            else {
                "status": "N/A",
                "reason": str(item.disabled_reason),
            }
        )
        for item in capabilities.tasks
    }
    return {
        "schema_version": "urbannav_navigation_vqa_audit_v2",
        "capability_config_digest": capabilities.digest(),
        "passed": not errors,
        "counts": dict(counts),
        "task_metrics": task_metrics,
        "dense_metrics": {
            "miou": "N/A",
            "sam_som_grounding_accuracy": "N/A",
            "tracking_accuracy": "N/A",
            "ttc_error": "N/A",
            "route_projection_error": "N/A",
        },
        "errors": errors[:200],
    }


__all__ = ["audit_vqa_artifacts_v2"]
