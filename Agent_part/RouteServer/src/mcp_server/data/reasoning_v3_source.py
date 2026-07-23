"""Build leakage-filtered Reasoning V3.4 frozen sources.

The shared V2 normalizer remains unchanged.  This module applies the stricter
V3.4 boundary afterwards: only images and text context remain model-visible;
tracking, GPS/pose, route geometry, and projection fields are removed before
the source fingerprint is frozen.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from mcp_server.contracts.reasoning_dataset import ReasoningSourceV2, reasoning_source_fingerprint
from mcp_server.contracts.reasoning_dataset_v3 import REASONING_SOURCE_V3_SCHEMA_VERSION, ReasoningSourceV3
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig, build_reasoning_source_v2
from mcp_server.data.reasoning_v3_labels import build_deterministic_supervision_v3


_V3_SOURCE_BOUNDARY_FILTER_VERSION = "reasoning_v3_4_visible_text_boundary_v1"
_FORBIDDEN_CONTEXT_KEYS = {
    "active_route_state",
    "active_waypoint",
    "active_waypoint_index",
    "active_waypoint_ref",
    "agent_tracks",
    "arrived",
    "camera_pose",
    "coarse_waypoints",
    "direction",
    "goal_distance_m",
    "goal_geometry",
    "goal_gps",
    "goal_reached",
    "independent_route_evidence",
    "is_goal_reached",
    "odometry",
    "pose",
    "route_direction",
    "route_geometry",
    "route_points",
    "route_points_ego_t",
    "route_state",
    "route_theta_deg",
    "runtime_task_phase",
    "selected_path",
    "task_phase",
    "tracked_agents",
    "tracking",
    "traffic_controls",
    "waypoints",
    "waypoints_ego_t",
}
_FORBIDDEN_CONTEXT_KEY_PARTS = (
    "calibrat",
    "extrinsic",
    "gnss",
    "gps",
    "homography",
    "intrinsic",
    "latitude",
    "longitude",
    "pixel_trajectory",
    "projection",
    "route_polyline",
    "track_id",
    "trajectory_fpv",
)
_FORBIDDEN_EVIDENCE_MODALITIES = {
    "control",
    "geometry",
    "gnss",
    "gps",
    "pose",
    "projection",
    "selected_path",
    "track",
    "waypoint",
}
_FORBIDDEN_EVIDENCE_REF_PARTS = (
    "direction",
    "gnss",
    "gps",
    "pose",
    "projection",
    "selected_path",
    "track.",
    "trajectory",
    "waypoint.",
)


def build_reasoning_source_v3(
    row: Mapping[str, Any],
    *,
    index: int,
    config: ReasoningSourceBuildConfig,
) -> ReasoningSourceV3:
    """Freeze model input and teacher-only truth in separate V3.4 partitions.

    Args:
        row: One heterogeneous source-manifest row.
        index: Stable fallback row index used by the V2 normalizer.
        config: Shared source normalization and dataset trust configuration.

    Returns:
        Fingerprinted V3.4 source with model input isolated from supervision.

    Raises:
        ValueError: If source normalization, trust-policy derivation, or schema
            validation fails.
    """

    legacy_model_input = build_reasoning_source_v2(row, index=index, config=config)
    model_input, boundary_audit = _filter_model_input_v3(legacy_model_input)
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
        raise ValueError(
            "V3.4 System1 policy disagrees with the normalized source: "
            f"labels={supervision.system1_enabled},source={model_input.system1_enabled}"
        )
    teacher_context = dict(supervision.teacher_supervision_context)
    teacher_context["source_boundary_filter"] = boundary_audit
    p0_flags = _p0_supervision_flags(row)
    teacher_context["p0_supervision_flags"] = list(p0_flags)
    warnings.extend(f"p0_supervision_flag:{flag}" for flag in p0_flags)
    for category, count in boundary_audit["removed_counts"].items():
        if count:
            warnings.append(f"v3_4_source_boundary_filtered:{category}:{count}")
    payload = {
        "schema_version": REASONING_SOURCE_V3_SCHEMA_VERSION,
        "model_input": model_input.model_dump(mode="json"),
        "teacher_supervision_context": teacher_context,
        "label_provenance": {
            key: value.model_dump(mode="json")
            for key, value in supervision.label_provenance.items()
        },
        "section_loss_mask": supervision.section_loss_mask.model_dump(mode="json"),
        "warnings": warnings,
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint(payload)
    return ReasoningSourceV3.model_validate(payload)


def _filter_model_input_v3(
    model_input: ReasoningSourceV2,
) -> tuple[ReasoningSourceV2, dict[str, Any]]:
    """Apply the V3.4 visible-text boundary to a normalized V2 source."""

    removed_context_paths: list[str] = []
    context = _sanitize_visible_text_value(
        model_input.model_visible_context,
        path="model_visible_context",
        removed_paths=removed_context_paths,
    )
    if not isinstance(context, dict):
        context = {}
    for context_key in ("goal", "route", "tool_response_context"):
        if context_key in context and not _has_text_payload(context[context_key]):
            removed_context_paths.append(f"model_visible_context.{context_key}")
            context.pop(context_key, None)

    removed_evidence_refs: list[str] = []
    evidence_catalog: list[dict[str, Any]] = []
    for entry in model_input.evidence_catalog:
        if _evidence_entry_is_forbidden(entry.ref, entry.modality):
            removed_evidence_refs.append(entry.ref)
            continue
        if entry.modality == "goal" and not _nonempty_visible_value(context.get("goal")):
            removed_evidence_refs.append(entry.ref)
            continue
        if entry.modality == "route" and not _nonempty_visible_value(context.get("route")):
            removed_evidence_refs.append(entry.ref)
            continue
        if entry.modality in {"tool", "tool_route"} and not _nonempty_visible_value(
            context.get("tool_response_context")
        ):
            removed_evidence_refs.append(entry.ref)
            continue
        attributes = _sanitize_visible_text_value(
            entry.attributes,
            path=f"evidence_catalog.{entry.ref}.attributes",
            removed_paths=removed_context_paths,
        )
        evidence_catalog.append(
            {
                "ref": entry.ref,
                "modality": entry.modality,
                "summary": entry.summary,
                "attributes": attributes if isinstance(attributes, dict) else {},
            }
        )

    allowed_modalities = {item["modality"] for item in evidence_catalog}
    available_modalities = [
        modality
        for modality in model_input.available_modalities
        if modality not in _FORBIDDEN_EVIDENCE_MODALITIES
        and (
            modality in allowed_modalities
            or modality in {"fpv", "temporal_fpv", "bev", "goal_image", "instruction"}
        )
    ]

    payload = model_input.model_dump(mode="json")
    payload.update(
        {
            "model_visible_context": context,
            "evidence_catalog": evidence_catalog,
            "available_modalities": sorted(set(available_modalities)),
            "goal_geometry": None,
            "route_points_ego_t": None,
            "reasoning_qa_label_validity": {},
            # V3.4 never exposes or carries forward the shared V2 private
            # dictionary.  Approved strong labels live in the separate V3
            # supervision partition.
            "teacher_private_context": {},
        }
    )
    legacy_metadata = dict(payload.get("metadata") or {})
    metadata = {
        key: legacy_metadata[key]
        for key in (
            "source_index",
            "image_selection_policy",
            "system1_disabled_reason",
            "source_path",
            "source_namespace",
            "media_root",
        )
        if key in legacy_metadata
    }
    metadata["v3_source_boundary_filter_version"] = _V3_SOURCE_BOUNDARY_FILTER_VERSION
    payload["metadata"] = metadata
    payload.pop("source_fingerprint", None)
    payload["source_fingerprint"] = reasoning_source_fingerprint(payload)
    filtered = ReasoningSourceV2.model_validate(payload)

    cleared_sidecars = sum(
        value is not None
        for value in (model_input.goal_geometry, model_input.route_points_ego_t)
    )
    cleared_private = int(bool(model_input.teacher_private_context))
    cleared_legacy_qa = int(bool(model_input.reasoning_qa_label_validity))
    audit = {
        "version": _V3_SOURCE_BOUNDARY_FILTER_VERSION,
        "removed_counts": {
            "context_paths": len(set(removed_context_paths)),
            "evidence_refs": len(set(removed_evidence_refs)),
            "geometry_sidecars": cleared_sidecars,
            "private_contexts": cleared_private,
            "legacy_qa_contexts": cleared_legacy_qa,
        },
        "removed_context_paths": sorted(set(removed_context_paths)),
        "removed_evidence_refs": sorted(set(removed_evidence_refs)),
    }
    return filtered, audit


def _sanitize_visible_text_value(
    value: Any,
    *,
    path: str,
    removed_paths: list[str],
) -> Any:
    """Retain strings/booleans while removing structured geometry and tracks."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            item_path = f"{path}.{key}"
            if _context_key_is_forbidden(key):
                removed_paths.append(item_path)
                continue
            sanitized = _sanitize_visible_text_value(
                item,
                path=item_path,
                removed_paths=removed_paths,
            )
            if _nonempty_visible_value(sanitized):
                result[key] = sanitized
        return result
    if isinstance(value, (list, tuple)):
        # Numeric arrays are route/pose geometry, not text context.
        if any(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            removed_paths.append(path)
            return None
        result = [
            _sanitize_visible_text_value(
                item,
                path=f"{path}[{index}]",
                removed_paths=removed_paths,
            )
            for index, item in enumerate(value)
        ]
        return [item for item in result if _nonempty_visible_value(item)]
    if isinstance(value, str) or isinstance(value, bool):
        return value
    if value is not None:
        removed_paths.append(path)
    return None


def _context_key_is_forbidden(value: str) -> bool:
    """Return whether a context key denotes private state or geometry."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in _FORBIDDEN_CONTEXT_KEYS:
        return True
    if normalized in {"lat", "lng", "lon", "yaw", "heading", "quaternion"}:
        return True
    if normalized.startswith("direction_") or normalized.endswith("_pose"):
        return True
    return any(part in normalized for part in _FORBIDDEN_CONTEXT_KEY_PARTS)


def _evidence_entry_is_forbidden(ref: str, modality: str) -> bool:
    """Return whether an evidence entry exposes disallowed structured state."""

    normalized_modality = str(modality).strip().lower()
    normalized_ref = str(ref).strip().lower().replace("-", "_")
    return (
        normalized_modality in _FORBIDDEN_EVIDENCE_MODALITIES
        or any(part in normalized_ref for part in _FORBIDDEN_EVIDENCE_REF_PARTS)
    )


def _nonempty_visible_value(value: Any) -> bool:
    """Return whether a sanitized context value still carries visible content."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple)):
        return bool(value)
    return True


def _has_text_payload(value: Any) -> bool:
    """Return whether context contains text beyond reference bookkeeping."""

    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(
            key not in {"id", "ref", "source_ref"}
            and _has_text_payload(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_text_payload(item) for item in value)
    return False


def _p0_supervision_flags(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Identify source patterns that cannot be primary Full-CoT supervision.

    The legacy instruction-goal builder may derive an instruction from the
    same motion target later predicted by the student. Likewise, the legacy
    direction-choice manifest may synthesize a structured ``<think>`` block
    directly from its answer label. V3.4 does not consume that old response as
    its canonical target, but recording these flags prevents a future export
    from silently treating such rows as official primary reasoning labels.
    """

    flags: set[str] = set()
    instruction_meta = row.get("instruction_meta")
    if isinstance(instruction_meta, Mapping):
        source = str(instruction_meta.get("source") or "").strip().lower()
        rule_version = str(instruction_meta.get("rule_version") or "").strip().lower()
        motion_source = str(instruction_meta.get("motion_source") or "").strip().lower()
        if source in {"rule", "egowalk_caption_rule"} or (
            rule_version == "instruction_rule_v1" and motion_source
        ):
            flags.add("target_derived_instruction_candidate")

    sidewalk_meta = row.get("sidewalk_meta")
    if isinstance(sidewalk_meta, Mapping) and sidewalk_meta.get("direction_choice_mode"):
        assistant_text = _source_assistant_text(row)
        if "<think>" in assistant_text and "</think>" in assistant_text:
            flags.add("answer_projected_direction_choice_cot_candidate")
    return tuple(sorted(flags))


def _source_assistant_text(row: Mapping[str, Any]) -> str:
    """Return a pre-existing assistant response for conservative source audit."""

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    return str(row.get("response") or "")


def build_reasoning_source_manifest_v3(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    config: ReasoningSourceBuildConfig,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Build frozen V3.4 sources and report supervision eligibility.

    The output keeps model-visible input and teacher-only deterministic labels
    in separate validated objects. No LVLM call is performed here.

    Args:
        input_jsonl: Original heterogeneous source manifest.
        output_jsonl: Destination frozen V3.4 source manifest.
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
            counters["system1_rows" if model_input.system1_enabled else "non_system1_rows"] += 1
            counters["fpv_bev_rows" if "bev" in model_input.available_modalities else "fpv_only_rows"] += 1
            fpv_frame_count = sum(role.startswith("fpv.") for role in model_input.image_roles)
            counters[f"fpv_frames:{fpv_frame_count}"] += 1
            for section_name, enabled in source.section_loss_mask.model_dump(mode="python").items():
                if enabled:
                    counters[f"section_enabled:{section_name}"] += 1
            for label_name, provenance in source.label_provenance.items():
                counters[f"label:{label_name}:{provenance.supervision}"] += 1
            canonical_direction = (
                source.teacher_supervision_context.get("fixed_labels") or {}
            ).get("canonical_direction")
            if isinstance(canonical_direction, Mapping):
                counters["canonical_direction_rows"] += 1
                counters[f"canonical_direction_profile:{canonical_direction['profile_id']}"] += 1
            else:
                counters["observation_only_rows"] += 1
            boundary_counts = (
                source.teacher_supervision_context.get("source_boundary_filter") or {}
            ).get("removed_counts") or {}
            for category, count in boundary_counts.items():
                counters[f"source_boundary_filtered:{category}"] += int(count)
            for flag in source.teacher_supervision_context.get("p0_supervision_flags") or []:
                counters[f"p0_supervision_flag:{flag}"] += 1
            for warning in source.warnings:
                counters[f"warning:{str(warning).split(':', 1)[0]}"] += 1
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
