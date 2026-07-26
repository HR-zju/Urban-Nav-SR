"""Build frozen Navigation VQA V2 sources from canonical Stage-A inputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.contracts.vqa_dataset_v2 import (
    VQASourceInterfacesV2,
    VQASourceV2,
    VQA_SOURCE_V2_SCHEMA_VERSION,
    vqa_source_fingerprint_v2,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic


_LEGACY_VQA_V1_SCHEMA_VERSIONS = {
    "sidewalk_navigation_vqa_annotation_v1",
    "sidewalk_navigation_vqa_training_view_v1",
}


def build_vqa_source_v2(
    reasoning_source: ReasoningSourceV4 | Mapping[str, Any],
    *,
    interfaces: VQASourceInterfacesV2 | None = None,
) -> VQASourceV2:
    """Project one frozen Reasoning V4 source into a VQA source.

    Args:
        reasoning_source: Canonical Stage-A source or its JSON payload.
        interfaces: Optional accepted external-GT interface records.

    Returns:
        Fingerprinted VQA source with visible input and private labels split.

    Raises:
        ValueError: If a legacy VQA row or invalid source is supplied.
    """

    if isinstance(reasoning_source, Mapping):
        schema_version = str(
            reasoning_source.get("schema_version") or ""
        )
        if schema_version in _LEGACY_VQA_V1_SCHEMA_VERSIONS:
            raise ValueError("legacy_vqa_v1_not_supported")
        source = ReasoningSourceV4.model_validate(reasoning_source)
    else:
        source = reasoning_source
    payload: dict[str, Any] = {
        "schema_version": VQA_SOURCE_V2_SCHEMA_VERSION,
        "observation": source.observation.model_dump(mode="json"),
        "direction_supervision": (
            source.direction_supervision.model_dump(mode="json")
            if source.direction_supervision is not None
            else None
        ),
        "interfaces": (
            interfaces or VQASourceInterfacesV2()
        ).model_dump(mode="json"),
        "warnings": list(source.warnings),
    }
    payload["source_fingerprint"] = vqa_source_fingerprint_v2(
        payload
    )
    return VQASourceV2.model_validate(payload)


def build_vqa_source_manifest_v2(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Build VQA sources from a Reasoning V4 source JSONL.

    Args:
        input_jsonl: Frozen Reasoning V4 source rows.
        output_jsonl: Destination VQA source JSONL.
        limit: Optional maximum input rows.
        skip_errors: Continue after invalid input when true.

    Returns:
        Deterministic conversion report.

    Raises:
        ValueError: If an invalid row is found and skipping is disabled.
    """

    outputs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        if limit is not None and index >= int(limit):
            break
        counts["input_rows"] += 1
        try:
            source = build_vqa_source_v2(row)
        except (TypeError, ValueError) as exc:
            counts["invalid_rows"] += 1
            if len(errors) < 100:
                errors.append(
                    {"index": index, "error": str(exc)}
                )
            if not skip_errors:
                raise
            continue
        outputs.append(
            source.model_dump(mode="json", exclude_none=True)
        )
        counts["output_rows"] += 1
        counts[
            f"dataset:{source.observation.dataset_source}"
        ] += 1
        counts["direction_rows"] += int(
            source.direction_supervision is not None
        )
        counts["temporal_rows"] += int(
            source.observation.temporal_evidence_eligible
        )
    write_jsonl_atomic(output_jsonl, outputs)
    return {
        "schema_version": (
            f"{VQA_SOURCE_V2_SCHEMA_VERSION}_build_report"
        ),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "counts": dict(counts),
        "errors": errors,
    }


__all__ = [
    "build_vqa_source_manifest_v2",
    "build_vqa_source_v2",
]
