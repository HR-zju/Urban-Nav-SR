"""Build V3 frozen sources by extending the validated V2 model-input view."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from mcp_server.contracts.reasoning_dataset import reasoning_source_fingerprint
from mcp_server.contracts.reasoning_dataset_v3 import REASONING_SOURCE_V3_SCHEMA_VERSION, ReasoningSourceV3
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig, build_reasoning_source_v2
from mcp_server.data.reasoning_v3_labels import build_deterministic_supervision_v3


def build_reasoning_source_v3(
    row: Mapping[str, Any],
    *,
    index: int,
    config: ReasoningSourceBuildConfig,
) -> ReasoningSourceV3:
    """Freeze model input and teacher-only truth in separate V3.3 partitions.

    Args:
        row: One heterogeneous source-manifest row.
        index: Stable fallback row index used by the V2 normalizer.
        config: Shared source normalization and dataset trust configuration.

    Returns:
        Fingerprinted V3.3 source with model input isolated from supervision.

    Raises:
        ValueError: If source normalization, trust-policy derivation, or schema
            validation fails.
    """

    model_input = build_reasoning_source_v2(row, index=index, config=config)
    label_row = dict(row)
    if model_input.trajectory_ego_t is not None:
        label_row["trajectory_ego_t"] = model_input.trajectory_ego_t
        label_row["trajectory_policy"] = model_input.trajectory_policy
    supervision = build_deterministic_supervision_v3(
        label_row,
        dataset_source=model_input.dataset_source,
    )
    warnings = list(supervision.warnings)
    if supervision.system1_enabled != model_input.system1_enabled:
        warnings.append(
            "deterministic_system1_eligibility_differs_from_validated_source:"
            f"labels={supervision.system1_enabled},source={model_input.system1_enabled}"
        )
    teacher_context = dict(supervision.teacher_supervision_context)
    teacher_context["fixed_labels"]["system1_enabled"] = model_input.system1_enabled
    section_loss_mask = supervision.section_loss_mask
    # 原始 manifest 的 images 可能交错存放 FPV/BEV 且没有 image_roles。
    # section 授权必须依据规范化后的模型可见证据，不能再次猜测原始字段形态。
    if _model_input_has_navigation_context(model_input):
        section_loss_mask = section_loss_mask.model_copy(
            update={"navigation_context_grounding": True}
        )
    payload = {
        "schema_version": REASONING_SOURCE_V3_SCHEMA_VERSION,
        "model_input": model_input.model_dump(mode="json"),
        "teacher_supervision_context": teacher_context,
        "label_provenance": {
            key: value.model_dump(mode="json")
            for key, value in supervision.label_provenance.items()
        },
        "section_loss_mask": section_loss_mask.model_dump(mode="json"),
        "warnings": warnings,
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint(payload)
    return ReasoningSourceV3.model_validate(payload)


def _model_input_has_navigation_context(model_input: Any) -> bool:
    """Return whether normalized model-visible evidence can ground navigation context.

    Args:
        model_input: Validated V2 model-input object produced by source normalization.

    Returns:
        ``True`` for a route-specific instruction or any optional navigation
        context modality visible to both teacher and student.
    """

    # Instruction/caption semantics are resolved by the LVLM teacher. Exposing
    # the source here does not assert that it is route-specific.
    if any(entry.modality == "instruction" for entry in model_input.evidence_catalog):
        return True
    navigation_modalities = {
        "bev",
        "goal",
        "goal_image",
        "waypoint",
        "route",
        "tool",
        "tool_route",
    }
    return any(entry.modality in navigation_modalities for entry in model_input.evidence_catalog)


def build_reasoning_source_manifest_v3(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    config: ReasoningSourceBuildConfig,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Build frozen V3 sources and report supervision eligibility.

    The output keeps model-visible input and teacher-only deterministic labels
    in separate validated objects. No LVLM call is performed here.

    Args:
        input_jsonl: Original heterogeneous source manifest.
        output_jsonl: Destination frozen V3.3 source manifest.
        config: Shared source normalization configuration.
        limit: Optional maximum input rows.
        skip_errors: Continue after per-row conversion failures.

    Returns:
        Deterministic conversion report and bounded error samples.

    Raises:
        ValueError: If a row is invalid and ``skip_errors`` is false.
    """

    counters: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []

    def normalized_rows() -> Iterable[dict[str, Any]]:
        for index, row in enumerate(iter_jsonl(input_jsonl)):
            if limit is not None and index >= int(limit):
                break
            counters["input_rows"] += 1
            try:
                source = build_reasoning_source_v3(row, index=index, config=config)
            except (TypeError, ValueError) as exc:
                counters["rejected_rows"] += 1
                if len(errors) < 100:
                    errors.append({"index": index, "id": row.get("id"), "error": str(exc)})
                if not skip_errors:
                    raise
                continue

            model_input = source.model_input
            counters["written_rows"] += 1
            counters[f"dataset:{model_input.dataset_source}"] += 1
            counters["system1_rows" if model_input.system1_enabled else "lm_only_rows"] += 1
            counters["fpv_bev_rows" if "bev" in model_input.available_modalities else "fpv_only_rows"] += 1
            fpv_frame_count = sum(role.startswith("fpv.") for role in model_input.image_roles)
            counters[f"fpv_frames:{fpv_frame_count}"] += 1
            for section_name, enabled in source.section_loss_mask.model_dump(mode="python").items():
                if enabled:
                    counters[f"section_enabled:{section_name}"] += 1
            for label_name, provenance in source.label_provenance.items():
                counters[f"label:{label_name}:{provenance.strength}"] += 1
            yield source.model_dump(mode="json", exclude_none=True)

    write_jsonl_atomic(output_jsonl, normalized_rows())
    return {
        "schema_version": f"{REASONING_SOURCE_V3_SCHEMA_VERSION}_build_report_v1",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "counts": dict(counters),
        "errors": errors[:100],
    }


__all__ = ["build_reasoning_source_manifest_v3", "build_reasoning_source_v3"]
