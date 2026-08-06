"""First-class fail-closed batch projection for UrbanNav Stage I."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.reasoning_dataset_v4 import ReasoningTrainingViewV4
from mcp_server.contracts.stage_i_sft import StageIProjectionDropV1
from mcp_server.contracts.vqa_dataset_v2 import VQAAnnotationV2, VQASourceV2
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.stage_i_provenance import (
    load_stage_i_instruction_provenance_v1,
    resolve_stage_i_instruction_role_v1,
)
from mcp_server.data.stage_i_views import (
    STAGE_I_MODALITY_PROFILES_V1,
    project_stage_i_reasoning_view_with_audit_v1,
    project_stage_i_reasoning_view_with_audit_v2,
    project_stage_i_vqa_view_with_audit_v1,
    project_stage_i_vqa_view_with_audit_v2,
)
from mcp_server.data.vqa_v2_eligibility import load_vqa_capabilities_v2

STAGE_I_BATCH_PROJECTOR_VERSION = "urbannav_stage_i_batch_projector_v1"
STAGE_I_BATCH_PROJECTOR_V2_VERSION = "urbannav_stage_i_batch_projector_v2"


def project_stage_i_batch_v1(
    *,
    output_view_jsonl: str | Path,
    output_drop_jsonl: str | Path,
    instruction_provenance_jsonl: str | Path,
    reasoning_view_jsonls: Sequence[str | Path] = (),
    vqa_source_jsonl: str | Path | None = None,
    vqa_annotation_jsonl: str | Path | None = None,
    modality_profile_ids: Sequence[str] | None = None,
    allow_fpv_only_decision: bool = False,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Project canonical parents into exchange views plus structured drops."""

    profiles = tuple(modality_profile_ids
                     or sorted(STAGE_I_MODALITY_PROFILES_V1))
    unknown_profiles = sorted(
        set(profiles) - set(STAGE_I_MODALITY_PROFILES_V1))
    if unknown_profiles:
        raise ValueError(
            f"unknown Stage I modality profiles: {unknown_profiles}")
    registry = load_stage_i_instruction_provenance_v1(
        instruction_provenance_jsonl)
    views: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for path_value in reasoning_view_jsonls:
        path = Path(path_value).expanduser()
        for raw in iter_jsonl(path):
            parent = ReasoningTrainingViewV4.model_validate(raw)
            source_fingerprint = str(
                parent.metadata.get("source_fingerprint") or "")
            role = resolve_stage_i_instruction_role_v1(
                source_group_id=parent.source_group_id,
                source_fingerprint=source_fingerprint,
                instruction=parent.instruction,
                registry=registry,
            )
            for task_type in ("full_cot", "decision"):
                for profile_id in profiles:
                    result = project_stage_i_reasoning_view_with_audit_v1(
                        parent,
                        task_type=task_type,
                        modality_profile_id=profile_id,
                        instruction_training_role=role,
                        allow_fpv_only_decision=allow_fpv_only_decision,
                    )
                    _record_projection(result, views, drops, counts)

    if (vqa_source_jsonl is None) != (vqa_annotation_jsonl is None):
        raise ValueError(
            "VQA source and annotation manifests must be supplied together")
    if vqa_source_jsonl is not None and vqa_annotation_jsonl is not None:
        sources: dict[str, VQASourceV2] = {}
        for raw in iter_jsonl(Path(vqa_source_jsonl).expanduser()):
            source = VQASourceV2.model_validate(raw)
            group = source.observation.source_group_id
            if group in sources:
                raise ValueError(f"duplicate VQA source group: {group}")
            sources[group] = source
        capabilities = load_vqa_capabilities_v2()
        for raw in iter_jsonl(Path(vqa_annotation_jsonl).expanduser()):
            annotation = VQAAnnotationV2.model_validate(raw)
            try:
                source = sources[annotation.source_group_id]
            except KeyError as exc:
                raise ValueError("VQA annotation source row is missing: "
                                 f"{annotation.source_group_id}") from exc
            role = resolve_stage_i_instruction_role_v1(
                source_group_id=source.observation.source_group_id,
                source_fingerprint=source.source_fingerprint,
                instruction=source.observation.instruction,
                registry=registry,
            )
            for profile_id in profiles:
                result = project_stage_i_vqa_view_with_audit_v1(
                    source,
                    annotation,
                    capabilities=capabilities,
                    modality_profile_id=profile_id,
                    instruction_training_role=role,
                )
                _record_projection(result, views, drops, counts)

    view_path = Path(output_view_jsonl).expanduser()
    drop_path = Path(output_drop_jsonl).expanduser()
    view_text = _write_jsonl(view_path, views)
    drop_text = _write_jsonl(drop_path, drops)
    report = {
        "schema_version":
        STAGE_I_BATCH_PROJECTOR_VERSION,
        "modality_profile_ids":
        list(profiles),
        "instruction_provenance_artifact":
        _input_record(Path(instruction_provenance_jsonl).expanduser()),
        "counts":
        dict(sorted(counts.items())),
        "unique_parent_samples":
        len({str(row["parent_sample_id"])
             for row in views}),
        "unique_source_groups":
        len({str(row["source_group_id"])
             for row in views}),
        "view_artifact":
        _output_record(view_path, view_text),
        "drop_artifact":
        _output_record(drop_path, drop_text),
    }
    if report_json is not None:
        path = Path(report_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) +
            "\n",
            encoding="utf-8",
        )
    return report


def project_stage_i_batch_v2(
    *,
    output_view_jsonl: str | Path,
    output_drop_jsonl: str | Path,
    instruction_provenance_jsonl: str | Path,
    reasoning_view_jsonls: Sequence[str | Path] = (),
    vqa_source_jsonl: str | Path | None = None,
    vqa_annotation_jsonl: str | Path | None = None,
    modality_profile_ids: Sequence[str] | None = None,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Project canonical parents into the Stage I V2 exchange pool.

    Only V2 FullCoT and VQA parents are materialized.  Direct Decision rows
    are intentionally absent: the ms-swift epoch compiler must first route an
    accepted FullCoT appearance to no-CoT and then derive its signed
    ``reasoning_dropout`` mirror.  This keeps the two routes mutually
    exclusive and prevents independent or failed generations from becoming a
    fake dropout source.

    Args:
        output_view_jsonl: Destination for accepted V2 parent views.
        output_drop_jsonl: Destination for structured projection failures.
        instruction_provenance_jsonl: Exact instruction-role registry.
        reasoning_view_jsonls: Formal Reasoning V4.6 parent artifacts.
        vqa_source_jsonl: Optional frozen VQA source manifest.
        vqa_annotation_jsonl: Optional accepted VQA annotation manifest.
        modality_profile_ids: Registered input profiles to project.
        report_json: Optional deterministic report destination.

    Returns:
        Projection counts, immutable input identity, and output fingerprints.
    """

    profiles = tuple(modality_profile_ids
                     or sorted(STAGE_I_MODALITY_PROFILES_V1))
    unknown_profiles = sorted(
        set(profiles) - set(STAGE_I_MODALITY_PROFILES_V1))
    if unknown_profiles:
        raise ValueError(
            f"unknown Stage I modality profiles: {unknown_profiles}")
    registry = load_stage_i_instruction_provenance_v1(
        instruction_provenance_jsonl)
    views: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for path_value in reasoning_view_jsonls:
        path = Path(path_value).expanduser()
        for raw in iter_jsonl(path):
            parent = ReasoningTrainingViewV4.model_validate(raw)
            role = resolve_stage_i_instruction_role_v1(
                source_group_id=parent.source_group_id,
                source_fingerprint=str(
                    parent.metadata.get("source_fingerprint") or ""),
                instruction=parent.instruction,
                registry=registry,
            )
            for profile_id in profiles:
                result = project_stage_i_reasoning_view_with_audit_v2(
                    parent,
                    modality_profile_id=profile_id,
                    instruction_training_role=role,
                )
                _record_projection(result, views, drops, counts)

    if (vqa_source_jsonl is None) != (vqa_annotation_jsonl is None):
        raise ValueError(
            "VQA source and annotation manifests must be supplied together")
    if vqa_source_jsonl is not None and vqa_annotation_jsonl is not None:
        sources: dict[str, VQASourceV2] = {}
        for raw in iter_jsonl(Path(vqa_source_jsonl).expanduser()):
            source = VQASourceV2.model_validate(raw)
            group = source.observation.source_group_id
            if group in sources:
                raise ValueError(f"duplicate VQA source group: {group}")
            sources[group] = source
        capabilities = load_vqa_capabilities_v2()
        for raw in iter_jsonl(Path(vqa_annotation_jsonl).expanduser()):
            annotation = VQAAnnotationV2.model_validate(raw)
            try:
                source = sources[annotation.source_group_id]
            except KeyError as exc:
                raise ValueError(
                    f"VQA annotation source row is missing: {annotation.source_group_id}"
                ) from exc
            role = resolve_stage_i_instruction_role_v1(
                source_group_id=source.observation.source_group_id,
                source_fingerprint=source.source_fingerprint,
                instruction=source.observation.instruction,
                registry=registry,
            )
            for profile_id in profiles:
                result = project_stage_i_vqa_view_with_audit_v2(
                    source,
                    annotation,
                    capabilities=capabilities,
                    modality_profile_id=profile_id,
                    instruction_training_role=role,
                )
                _record_projection(result, views, drops, counts)

    view_path = Path(output_view_jsonl).expanduser()
    drop_path = Path(output_drop_jsonl).expanduser()
    view_text = _write_jsonl(view_path, views)
    drop_text = _write_jsonl(drop_path, drops)
    report = {
        "schema_version":
        STAGE_I_BATCH_PROJECTOR_V2_VERSION,
        "training_contract":
        "v2",
        "modality_profile_ids":
        list(profiles),
        "reasoning_dropout_policy": {
            "materialized_by_batch_projector": False,
            "source": "deterministic_projection_after_epoch_route",
            "eligible_parent_count": int(counts.get("task:full_cot", 0)),
        },
        "instruction_provenance_artifact":
        _input_record(Path(instruction_provenance_jsonl).expanduser()),
        "counts":
        dict(sorted(counts.items())),
        "unique_parent_samples":
        len({str(row["parent_sample_id"])
             for row in views}),
        "unique_source_groups":
        len({str(row["source_group_id"])
             for row in views}),
        "view_artifact":
        _output_record(view_path, view_text),
        "drop_artifact":
        _output_record(drop_path, drop_text),
    }
    if report_json is not None:
        path = Path(report_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) +
            "\n",
            encoding="utf-8",
        )
    return report


def _record_projection(
    result: Any,
    views: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    """Separate trainable views from annotation-only drop audits."""

    payload = result.model_dump(mode="json", exclude_none=True)
    if isinstance(result, StageIProjectionDropV1):
        drops.append(payload)
        counts["drops"] += 1
        counts[f"drop:{result.reason_code}"] += 1
        counts[f"drop_task:{result.task_type}"] += 1
        return
    views.append(payload)
    counts["views"] += 1
    counts[f"task:{result.task_type}"] += 1
    counts[f"mode:{result.modality_profile_id}"] += 1


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write deterministic compact JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows)
    path.write_text(content, encoding="utf-8")
    return content


def _input_record(path: Path) -> dict[str, Any]:
    """Fingerprint one immutable input artifact."""

    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }


def _output_record(path: Path, content: str) -> dict[str, Any]:
    """Fingerprint one newly written output artifact."""

    return {
        "path": str(path),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "rows": content.count("\n"),
    }


__all__ = [
    "STAGE_I_BATCH_PROJECTOR_VERSION",
    "STAGE_I_BATCH_PROJECTOR_V2_VERSION",
    "project_stage_i_batch_v1",
    "project_stage_i_batch_v2",
]
