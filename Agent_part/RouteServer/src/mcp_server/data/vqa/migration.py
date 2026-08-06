"""Versioned deterministic migrations for legacy VQA V2 artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.contracts.direction_profile import (
    affordance_probe_profile_for_action,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionSupervisionV4,
    NavObservationV4,
)
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQASourceInterfacesV2,
    VQASourceV2,
    annotation_context_digest_v2,
    vqa_source_fingerprint_v2,
)
from mcp_server.contracts.vqa_v2 import VQAQuestionV2
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.data.vqa.eligibility import load_vqa_capabilities_v2
from mcp_server.data.vqa.producers import (
    CanonicalDemonstratedDirectionProducerV2,
)
from mcp_server.data.vqa.questions import build_vqa_questions_v2


VQA_V2_DIRECTION_CONTEXT_MIGRATION_ID = (
    "urbannav_vqa_v2_direction_context_migration_v1"
)


def migrate_vqa_source_payload_v2(
    raw: dict[str, Any],
) -> tuple[VQASourceV2, bool]:
    """Replace the retired direction-output alias and refresh the fingerprint."""

    payload = json.loads(json.dumps(raw))
    observation = payload["observation"]
    legacy = observation.pop("direction_output_space", None)
    changed = legacy is not None
    if legacy is not None:
        action = resolve_stage_a_direction_profile(str(legacy["profile_id"]))
        if (
            str(legacy["profile_version"])
            != str(action["profile_version"])
            or list(legacy["allowed_labels"]) != list(action["labels"])
        ):
            raise ValueError("legacy VQA direction output space is stale")
        probe_id = affordance_probe_profile_for_action(str(legacy["profile_id"]))
        probe = resolve_affordance_probe_profile(probe_id)
        observation["direction_context"] = {
            "action_profile_id": legacy["profile_id"],
            "action_profile_version": legacy["profile_version"],
            "affordance_probe_profile_id": probe_id,
            "affordance_probe_profile_version": probe["profile_version"],
        }
    payload["observation"] = NavObservationV4.model_validate(
        observation
    ).model_dump(mode="json")
    if payload.get("direction_supervision") is not None:
        payload["direction_supervision"] = (
            DirectionSupervisionV4.model_validate(
                payload["direction_supervision"]
            ).model_dump(mode="json")
        )
    payload["interfaces"] = VQASourceInterfacesV2.model_validate(
        payload.get("interfaces") or {}
    ).model_dump(mode="json")
    warnings = list(payload.get("warnings") or [])
    if changed and VQA_V2_DIRECTION_CONTEXT_MIGRATION_ID not in warnings:
        warnings.append(VQA_V2_DIRECTION_CONTEXT_MIGRATION_ID)
    payload["warnings"] = warnings
    payload["source_fingerprint"] = vqa_source_fingerprint_v2({
        key: value
        for key, value in payload.items()
        if key != "source_fingerprint"
    })
    return VQASourceV2.model_validate(payload), changed


def migrate_vqa_annotation_payload_v2(
    raw: dict[str, Any],
    source: VQASourceV2,
) -> VQAAnnotationV2:
    """Rebind an unchanged Q/A pair to the migrated visible source identity."""

    payload = json.loads(json.dumps(raw))
    if str(payload["source_group_id"]) != source.observation.source_group_id:
        raise ValueError("VQA migration source/annotation group mismatch")
    question = VQAQuestionV2.model_validate(payload["question"])
    old_annotation_id = str(payload["annotation_id"])
    payload["source_fingerprint"] = source.source_fingerprint
    payload["input_fingerprint"] = source.source_fingerprint
    payload["annotation_context_digest"] = annotation_context_digest_v2(
        source=source,
        question=question,
        producer_id=str(payload["producer_id"]),
        producer_version=str(payload["producer_version"]),
    )
    identity_digest = hashlib.sha256(
        (
            old_annotation_id
            + ":"
            + source.source_fingerprint
            + ":"
            + VQA_V2_DIRECTION_CONTEXT_MIGRATION_ID
        ).encode("utf-8")
    ).hexdigest()[:20]
    payload["annotation_id"] = (
        f"{source.observation.source_group_id}:vqa_v2:"
        f"{question.task_id}:migrated:{identity_digest}"
    )
    metadata = dict(payload.get("metadata") or {})
    metadata["migration_id"] = VQA_V2_DIRECTION_CONTEXT_MIGRATION_ID
    metadata["migrated_from_annotation_id"] = old_annotation_id
    metadata["question_answer_changed"] = False
    payload["metadata"] = metadata
    return VQAAnnotationV2.model_validate(payload)


def migrate_vqa_artifacts_v2(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_source_jsonl: str | Path,
    output_annotation_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Create new canonical source/annotation files and immutable audit data."""

    source_path = Path(source_jsonl).expanduser()
    annotation_path = Path(annotation_jsonl).expanduser()
    migrated_sources: dict[str, VQASourceV2] = {}
    source_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw in iter_jsonl(source_path):
        source, changed = migrate_vqa_source_payload_v2(raw)
        group = source.observation.source_group_id
        if group in migrated_sources:
            raise ValueError(f"duplicate VQA source group: {group}")
        migrated_sources[group] = source
        source_rows.append(source.model_dump(mode="json", exclude_none=True))
        counts["source_rows"] += 1
        counts["source_alias_migrated"] += int(changed)

    annotation_rows: list[dict[str, Any]] = []
    capabilities = load_vqa_capabilities_v2()
    direction_producer = CanonicalDemonstratedDirectionProducerV2()
    for raw in iter_jsonl(annotation_path):
        group = str(raw["source_group_id"])
        try:
            source = migrated_sources[group]
        except KeyError as exc:
            raise ValueError(f"VQA annotation source is missing: {group}") from exc
        annotation = migrate_vqa_annotation_payload_v2(raw, source)
        if (
            annotation.label_source == "derived_gt"
            and annotation.question.task_id
            == "e0_demonstrated_direction_selection"
        ):
            questions = {
                item.task_id: item
                for item in build_vqa_questions_v2(
                    source,
                    capabilities,
                ).questions
            }
            result = direction_producer.produce(
                source,
                questions[annotation.question.task_id],
                capabilities,
            )
            if result.annotation is None:
                raise ValueError(
                    "deterministic VQA minimal-evidence rebuild failed: "
                    f"{result.reason}"
                )
            if result.annotation.answer != annotation.answer:
                raise ValueError(
                    "deterministic VQA migration changed the GT answer"
                )
            rebuilt_metadata = dict(result.annotation.metadata)
            rebuilt_metadata.update({
                "migration_id": VQA_V2_DIRECTION_CONTEXT_MIGRATION_ID,
                "migrated_from_annotation_id": str(raw["annotation_id"]),
                "question_rebuilt_with_minimal_evidence": True,
                "answer_changed": False,
            })
            annotation = VQAAnnotationV2.model_validate(
                result.annotation.model_copy(
                    update={"metadata": rebuilt_metadata}
                ).model_dump(mode="json", exclude_none=True)
            )
            counts["derived_gt_minimal_evidence_rebuilt"] += 1
        elif annotation.label_source == "lvlm_weak":
            # Weak answers remain tied to the complete evidence they saw.
            # New minimal-evidence questions require a fresh teacher call.
            counts["weak_label_reannotation_required"] += 1
        annotation_rows.append(
            annotation.model_dump(mode="json", exclude_none=True)
        )
        counts["annotation_rows"] += 1
        counts[f"label_source:{annotation.label_source}"] += 1

    source_output = Path(output_source_jsonl).expanduser()
    annotation_output = Path(output_annotation_jsonl).expanduser()
    source_text = _write_jsonl(source_output, source_rows)
    annotation_text = _write_jsonl(annotation_output, annotation_rows)
    report = {
        "schema_version": VQA_V2_DIRECTION_CONTEXT_MIGRATION_ID,
        "input_source_artifact": _artifact_record(source_path),
        "input_annotation_artifact": _artifact_record(annotation_path),
        "output_source_artifact": _output_record(source_output, source_text),
        "output_annotation_artifact": _output_record(
            annotation_output,
            annotation_text,
        ),
        "counts": dict(sorted(counts.items())),
        "answer_changed": False,
        "derived_gt_question_rebuilt": True,
        "weak_label_evidence_reduced": False,
    }
    if report_json is not None:
        report_path = Path(report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write compact deterministic JSONL and return its exact bytes as text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(content, encoding="utf-8")
    return content


def _artifact_record(path: Path) -> dict[str, Any]:
    """Return an input fingerprint record."""

    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _output_record(path: Path, content: str) -> dict[str, Any]:
    """Return an output fingerprint record."""

    return {
        "path": str(path),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "rows": content.count("\n"),
    }


__all__ = [
    "VQA_V2_DIRECTION_CONTEXT_MIGRATION_ID",
    "migrate_vqa_annotation_payload_v2",
    "migrate_vqa_artifacts_v2",
    "migrate_vqa_source_payload_v2",
]
