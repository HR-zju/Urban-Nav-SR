"""Build leakage-bounded single-turn Reasoning V4 sources."""

from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from urbannav_contracts.history_observation import history_image_roles, validate_history_observation

from mcp_server.contracts.direction_profile import (
    affordance_probe_profile_for_action,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionGeometrySupervisionV1,
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    V4_FORBIDDEN_VISIBLE_CONTEXT_KEY_PARTS,
    BevRouteEvidenceV4,
    DirectionContextV4,
    DirectionSupervisionV4,
    EvidenceCatalogEntryV4,
    InputModalitiesV4,
    NavObservationV4,
    ReasoningSourceV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.data.canonical.direction_geometry import (
    build_direction_geometry_supervision_v1,
    direction_geometry_distribution_report,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from mcp_server.data.legacy.reasoning_v2.source import (
    ReasoningSourceBuildConfig,
    _dataset_source,
    _ordered_images,
    _resolve_image_path,
    _source_record_id,
    _split_group_id,
)
from mcp_server.runtime.system1_contract import (
    canonical_trajectory_policy,
    canonicalize_trajectory_ego_t,
)


_DIRECTION_FLAT_FIELDS = {
    "profile_id": "direction_profile",
    "profile_version": "direction_profile_version",
    "label_index": "direction_choice_label",
    "label_name": "direction_name",
    "choice": "direction_choice",
    "angle_deg": "direction_angle_deg",
    "source": "direction_source",
    "bin_mode": "direction_bin_mode",
    "derivation_version": "direction_derivation_version",
}
_DIRECTION_V4_FLAT_FIELDS = {
    "profile_id": "direction_profile_id",
    "profile_version": "direction_profile_version",
    "label_index": "direction_label_index",
    "label_name": "direction_label_name",
    "choice": "direction_choice",
    "angle_deg": "direction_angle_deg",
    "source": "direction_source",
    "bin_mode": "direction_bin_mode",
    "derivation_version": "direction_derivation_version",
}
_SUPPORTED_DATASET_FAMILIES = (
    "sidewalk",
    "foxglove",
    "egowalk",
    "frodobots",
)
_DIRECTION_GT_DISABLED_DATASET_FAMILIES = frozenset({"frodobots"})


def build_reasoning_source_v4(
    row: Mapping[str, Any],
    *,
    index: int,
    config: ReasoningSourceBuildConfig,
) -> ReasoningSourceV4:
    """Convert one heterogeneous dataset row into a frozen V4 source.

    Args:
        row: Source dataset record.
        index: Stable fallback index used only when the record has no id.
        config: Image selection, media-root, and System1 policy.

    Returns:
        Fingerprinted V4 source with visible and private fields separated.

    Raises:
        ValueError: If image roles, direction stamps, or System1 sidecars are
            inconsistent.
    """

    if row.get("schema_version") == REASONING_SOURCE_V4_SCHEMA_VERSION:
        return ReasoningSourceV4.model_validate(row)

    dataset_source = _dataset_source(row, fallback=config.dataset_source)
    source_record_id = _source_record_id(row, index=index)
    source_group_id = str(
        row.get("source_group_id")
        or f"{dataset_source}:{source_record_id}"
    )
    split_group_id = _split_group_id(
        row,
        source_record_id=source_record_id,
    )
    history = row.get("history_observation")
    if history is None:
        images, image_roles = _ordered_images(row, config=config)
    else:
        validate_history_observation(history, max_lag_ns=row["history_max_lag_ns"])
        if history["source_parent_id"] != source_group_id:
            raise ValueError("History Reasoning source parent must match source_group_id")
        history = copy.deepcopy(history)
        history["fpv_paths"] = [_resolve_image_path(path, media_root=config.media_root)
                                for path in history["fpv_paths"]]
        images = list(history["fpv_paths"])
        if len(images) != 9 or len(set(images)) != 9:
            raise ValueError("History Reasoning source requires nine unique FPV paths")
        if history["bev"] is not None:
            bev = history["bev"]
            if bev["input_profile"] not in ("causal_map_v1", "recorded_bev_v1"):
                raise ValueError("Unsupported history Reasoning BEV profile")
            if bev["input_profile"] == "causal_map_v1" and bev["draw_trajectory"]:
                raise ValueError("Map-only BEV must preserve its route-free content contract")
            age = history["source_ns"][-1] - bev["timestamp"]
            if age < 0 or age != bev["age_ns"]:
                raise ValueError("History Reasoning BEV requires a causal timestamp and matching age")
            bev["bev_image_path"] = _resolve_image_path(bev["bev_image_path"], media_root=config.media_root)
            images.append(bev["bev_image_path"])
        # 新 profile 的角色和帧数由时间合同决定，保持旧五帧 selector 的行为不变。
        image_roles = history_image_roles(include_bev=history["bev"] is not None)
    instruction = _optional_instruction(row)
    visible_context, context_warnings = _visible_context_v4(row)
    evidence_row = row
    if history is not None:
        bev = history["bev"]
        route = BevRouteEvidenceV4()
        if bev is not None and bev["input_profile"] == "recorded_bev_v1":
            # 时间 predecessor 不改变旧图中的 route；公开证据按原绘制来源重建。
            route = _bev_route_evidence_v4({
                "bev_route_scope": bev["bev_route_scope"],
                "bev_route_overlay": bev.get("bev_route_overlay", {}),
                "bev_image_alignment": bev.get("bev_alignment", "unknown"),
            }, role="bev.current", dataset_source=dataset_source)
        evidence_row = {**row, "bev_route_evidence": route.model_dump(mode="json")}
    evidence_catalog = _evidence_catalog_v4(
        row=evidence_row,
        dataset_source=dataset_source,
        image_roles=image_roles,
        instruction=instruction,
        visible_context=visible_context,
    )
    if history is not None:
        current = next(entry for entry in evidence_catalog if entry.ref == "fpv.current")
        current.attributes.update({"history_observation": history,
                                   "history_max_lag_ns": row["history_max_lag_ns"]})
    modalities = InputModalitiesV4(
        history_fpv=any(role.startswith("fpv.history_") for role in image_roles),
        instruction=instruction is not None,
        bev=any(role.startswith("bev.") for role in image_roles),
        goal_image=any(role.startswith("goal.image_") for role in image_roles),
        pixel_goal=any(
            entry.modality == "pixel_goal" for entry in evidence_catalog
        ),
        waypoint=any(
            entry.modality == "waypoint" for entry in evidence_catalog
        ),
    )
    fpv_paths = [
        image
        for image, role in zip(images, image_roles)
        if role.startswith("fpv.")
    ]
    has_distinct_history = len(set(fpv_paths)) >= 2
    direction, direction_warnings = _direction_supervision_v4(
        row,
        dataset_source=dataset_source,
    )
    direction_geometry, geometry_warnings = _direction_geometry_v1(
        row,
        direction=direction,
    )
    direction_context = _direction_context_v4(
        row,
        dataset_source=dataset_source,
        direction=direction,
    )
    observation = NavObservationV4(
        source_group_id=source_group_id,
        dataset_source=dataset_source,
        source_record_id=source_record_id,
        split_group_id=split_group_id,
        images=images,
        image_roles=image_roles,
        instruction=instruction,
        travel_mode=str(
            row.get("travel_mode")
            or row.get("mode")
            or config.travel_mode
        ),
        model_visible_context=visible_context,
        evidence_catalog=evidence_catalog,
        input_modalities=modalities,
        direction_context=direction_context,
        temporal_evidence_eligible=False,
    )

    trajectory, trajectory_policy, action_mask, trajectory_warning = (
        _system1_sidecars_v4(
            row,
            dataset_source=dataset_source,
            config=config,
        )
    )
    warnings = [
        *context_warnings,
        *direction_warnings,
        *geometry_warnings,
        *([trajectory_warning] if trajectory_warning else []),
    ]
    if modalities.history_fpv and not has_distinct_history:
        warnings.append("history_fpv_is_repeated_padding")

    payload: dict[str, Any] = {
        "schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION,
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": (
            direction.model_dump(mode="json")
            if direction is not None
            else None
        ),
        "direction_geometry": (
            direction_geometry.model_dump(mode="json")
            if direction_geometry is not None
            else None
        ),
        "trajectory_ego_t": trajectory,
        "trajectory_policy": trajectory_policy,
        "action_mask": action_mask,
        "warnings": warnings,
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return ReasoningSourceV4.model_validate(payload)


def build_history_reasoning_source_v4(
    row: Mapping[str, Any],
    *,
    index: int,
    config: ReasoningSourceBuildConfig,
    max_lag_ns: int,
) -> ReasoningSourceV4:
    """Build a canonical source from one direction-projected history row.

    Args:
        row: Packaged observation with its original source fields.
        index: Source row position.
        config: Build policy; media_root is this row's clip package directory.
        max_lag_ns: Sampling budget recorded by the package compiler.

    Returns:
        Canonical source retaining the new physical-time observation.
    """
    original = row["source_row"]
    dataset_source = _dataset_source(original, fallback=config.dataset_source)
    direction, _ = _direction_supervision_v4(original, dataset_source=dataset_source)
    geometry, _ = _direction_geometry_v1(original, direction=direction)
    # 旧 assistant 留在原包；新 source 只接收公开指令、物理时间与私有方向监督。
    projected = {
        "id": row["source_parent_id"],
        "source_group_id": row["source_parent_id"],
        "split_group_id": row["split_group_id"],
        "dataset_source": dataset_source,
        "instruction": _optional_instruction(original),
        "direction_supervision": _raw_direction_record(original),
        "direction_geometry": geometry.model_dump(mode="json") if geometry is not None else None,
        "history_observation": row["observation"],
        "history_max_lag_ns": max_lag_ns,
    }
    return build_reasoning_source_v4(projected, index=index, config=config)


def build_reasoning_source_manifest_v4(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    config: ReasoningSourceBuildConfig,
    limit: int | None = None,
    skip_errors: bool = False,
    history_max_lag_ns: int | None = None,
) -> dict[str, Any]:
    """Build a V4 source manifest and deterministic conversion report.

    Args:
        input_jsonl: Heterogeneous dataset manifest.
        output_jsonl: Destination frozen V4 source manifest.
        config: Source build policy.
        limit: Optional maximum number of input rows.
        skip_errors: Continue after invalid rows when true.
        history_max_lag_ns: Explicit budget for materialized history-package rows.

    Returns:
        Conversion counts and bounded errors.

    Raises:
        ValueError: If one row is invalid and ``skip_errors`` is false.
    """

    outputs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        if limit is not None and index >= int(limit):
            break
        counts["input_rows"] += 1
        try:
            if history_max_lag_ns is not None:
                source = build_history_reasoning_source_v4(
                    row, index=index, config=config, max_lag_ns=history_max_lag_ns)
            else:
                source = build_reasoning_source_v4(row, index=index, config=config)
        except (TypeError, ValueError) as exc:
            counts["invalid_rows"] += 1
            if len(errors) < 100:
                errors.append({"index": index, "error": str(exc)})
            if not skip_errors:
                raise
            continue
        outputs.append(source.model_dump(mode="json", exclude_none=True))
        counts["output_rows"] += 1
        counts[
            f"dataset:{source.observation.dataset_source}"
        ] += 1
        counts["direction_rows"] += int(source.direction_supervision is not None)
        counts["direction_geometry_rows"] += int(source.direction_geometry is not None)
        if source.direction_geometry is not None:
            counts[f"direction_geometry:{source.direction_geometry.confidence}"] += 1
        counts["system1_rows"] += int(source.trajectory_ego_t is not None)
        for name, present in source.observation.input_modalities.model_dump(
            mode="python"
        ).items():
            counts[f"modality:{name}"] += int(bool(present))
    write_jsonl_atomic(output_jsonl, outputs)
    return {
        "schema_version": f"{REASONING_SOURCE_V4_SCHEMA_VERSION}_report",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "counts": dict(counts),
        "direction_geometry": direction_geometry_distribution_report(outputs),
        "errors": errors,
    }


def _optional_instruction(row: Mapping[str, Any]) -> str | None:
    """Return only an instruction actually present in the source record."""

    for key in ("instruction", "navigation_instruction", "user_request"):
        text = str(row.get(key) or "").strip()
        if text:
            return text
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                not isinstance(message, Mapping)
                or str(message.get("role") or "").lower() != "user"
            ):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            matches = re.findall(
                r"(?im)^\s*(?:instruction|navigation instruction)\s*:"
                r"\s*(?P<instruction>[^\r\n]+?)\s*$",
                content,
            )
            if matches:
                return str(matches[-1]).strip()
    return None


def _visible_context_v4(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Keep only prompt-visible Stage-A goal and waypoint context."""

    warnings: list[str] = []
    context: dict[str, Any] = {}
    existing = row.get("model_visible_context") or row.get(
        "reasoning_input_context"
    )
    if isinstance(existing, Mapping):
        sanitized = _sanitize_visible_value(existing)
        if isinstance(sanitized, dict):
            for key in ("goal", "pixel_goal", "waypoint", "waypoints"):
                if key in sanitized:
                    context[key] = sanitized[key]

    goal = _first_present(
        row,
        "goal_spec",
        "goal",
        "goal_text",
    )
    if goal is None and row.get("destination") is not None:
        destination = row["destination"]
        if _destination_is_prompt_visible(row, destination):
            goal = destination
        else:
            warnings.append(
                "unverified_or_coordinate_destination_dropped"
            )
    if goal is not None and "goal" not in context:
        context["goal"] = _sanitize_visible_value(goal)

    pixel_goal = _first_present(
        row,
        "pixel_goal",
        "goal_pixel",
        "point_prompt",
    )
    if pixel_goal is not None and "pixel_goal" not in context:
        context["pixel_goal"] = _sanitize_visible_value(pixel_goal)

    waypoint = _first_present(row, "waypoint", "waypoints")
    if waypoint is not None and "waypoint" not in context and "waypoints" not in context:
        if _waypoint_is_prompt_visible(row):
            context["waypoint"] = _sanitize_visible_value(waypoint)
        else:
            warnings.append("unverified_or_gt_waypoint_dropped")
    return context, warnings


def _evidence_catalog_v4(
    *,
    row: Mapping[str, Any],
    dataset_source: str,
    image_roles: list[str],
    instruction: str | None,
    visible_context: Mapping[str, Any],
) -> list[EvidenceCatalogEntryV4]:
    """Build raw evidence addresses without synthetic corridor references."""

    entries: list[EvidenceCatalogEntryV4] = []
    if instruction is not None:
        entries.append(
            EvidenceCatalogEntryV4(
                ref="instruction.active",
                modality="instruction",
                summary="active navigation instruction",
            )
        )
    for role in image_roles:
        if role.startswith("fpv."):
            entries.append(
                EvidenceCatalogEntryV4(
                    ref=role,
                    modality="fpv",
                    summary="prompt-visible first-person observation",
                )
            )
        elif role.startswith("bev."):
            route_evidence = _bev_route_evidence_v4(
                row,
                role=role,
                dataset_source=dataset_source,
            )
            entries.append(
                EvidenceCatalogEntryV4(
                    ref=role,
                    modality="bev",
                    summary="optional route/topology context without local safety authority",
                    bev_route_evidence=route_evidence,
                )
            )
        elif role.startswith("goal.image_"):
            entries.append(
                EvidenceCatalogEntryV4(
                    ref=role,
                    modality="goal_image",
                    summary="visual goal reference",
                )
            )
    if visible_context.get("goal") is not None:
        entries.append(
            EvidenceCatalogEntryV4(
                ref="goal.active",
                modality="goal",
                summary="explicit prompt-visible goal specification",
            )
        )
    if visible_context.get("pixel_goal") is not None:
        entries.append(
            EvidenceCatalogEntryV4(
                ref="goal.pixel_01",
                modality="pixel_goal",
                summary="explicit prompt-visible pixel goal",
            )
        )
    waypoint_value = visible_context.get(
        "waypoint",
        visible_context.get("waypoints"),
    )
    if waypoint_value is not None:
        values = (
            waypoint_value
            if isinstance(waypoint_value, list)
            else [waypoint_value]
        )
        for index, _ in enumerate(values[:16], start=1):
            entries.append(
                EvidenceCatalogEntryV4(
                    ref=f"waypoint.next_{index:02d}",
                    modality="waypoint",
                    summary="explicit prompt-visible waypoint condition",
                )
            )
    return entries


def _bev_route_evidence_v4(
    row: Mapping[str, Any],
    *,
    role: str,
    dataset_source: str,
) -> BevRouteEvidenceV4:
    """Build fail-closed BEV route capability metadata.

    Args:
        row: Heterogeneous source row.
        role: Exact BEV evidence ref.
        dataset_source: Normalized source dataset identifier.

    Returns:
        Typed route evidence. Existing recorded-clip overlays never gain route
        progress authority.
    """

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    raw = row.get("bev_route_evidence") or metadata.get(
        "bev_route_evidence"
    )
    if isinstance(raw, Mapping):
        selected = raw.get(role)
        if isinstance(selected, Mapping):
            return BevRouteEvidenceV4.model_validate(selected)
        if any(
            key in raw
            for key in (
                "route_overlay_visible",
                "route_endpoint_semantics",
                "producer_contract_id",
            )
        ):
            return BevRouteEvidenceV4.model_validate(raw)

    overlay_meta = row.get("bev_route_overlay") or metadata.get(
        "bev_route_overlay"
    )
    if not isinstance(overlay_meta, Mapping):
        overlay_meta = {}
    overlay_scope = str(
        overlay_meta.get("scope")
        or row.get("bev_route_scope")
        or "none"
    )
    overlay_point_count = int(
        overlay_meta.get("point_count") or 0
    )
    overlay_visible = bool(
        row.get("bev_patch_meta_path")
        or row.get("route_context")
        or row.get("route_overlay_meta")
        or overlay_scope != "none"
        or overlay_point_count > 0
    )
    family = _dataset_family(dataset_source)
    alignment = str(
        row.get("bev_image_alignment")
        or metadata.get("bev_alignment")
        or "unknown"
    )
    if (
        overlay_visible
        and family in {"sidewalk", "foxglove"}
        and alignment == "ego_aligned"
    ):
        endpoint_semantics = str(
            overlay_meta.get("route_endpoint_semantics")
            or (
                "crop_boundary"
                if overlay_scope == "local_future"
                else "recorded_clip_endpoint"
            )
        )
        return BevRouteEvidenceV4(
            frame="ego",
            ego_forward="up",
            route_overlay_visible=True,
            route_direction_authority="visual_conditional",
            route_progress_authority="none",
            route_terminal_marker_visible=True,
            route_endpoint_semantics=endpoint_semantics,
            producer_contract_id=(
                "legacy_bev_overlay_explicit_ego_alignment"
            ),
            producer_contract_version="1.0",
        )
    return BevRouteEvidenceV4(
        route_overlay_visible=overlay_visible,
        route_direction_authority="none",
        route_progress_authority="none",
        route_terminal_marker_visible=False,
        route_endpoint_semantics="unknown",
        producer_contract_id="legacy_bev_unknown",
        producer_contract_version="1.0",
    )


def _direction_supervision_v4(
    row: Mapping[str, Any],
    *,
    dataset_source: str,
) -> tuple[DirectionSupervisionV4 | None, list[str]]:
    """Validate one precomputed direction record without deriving it in V4."""

    raw = _raw_direction_record(row)
    family = _dataset_family(dataset_source)
    if family in _DIRECTION_GT_DISABLED_DATASET_FAMILIES:
        raise ValueError(
            "dataset has no approved Reasoning V4 direction GT: "
            f"{family}"
        )
    if raw is None:
        return None, ["canonical_direction_missing"]
    core_fields = (
        "profile_id",
        "profile_version",
        "label_index",
        "label_name",
        "choice",
        "source",
        "bin_mode",
        "derivation_version",
    )
    missing = [
        field
        for field in core_fields
        if raw.get(field) is None or raw.get(field) == ""
    ]
    if missing and not _has_explicit_v4_direction_stamp(row):
        return None, [
            "incomplete_legacy_direction_dropped:"
            + ",".join(missing)
        ]
    direction = DirectionSupervisionV4.model_validate(raw)
    return direction, []


def _direction_context_v4(
    row: Mapping[str, Any],
    *,
    dataset_source: str,
    direction: DirectionSupervisionV4 | None,
) -> DirectionContextV4:
    """Build public action/probe identities without repeating label arrays."""

    family = _dataset_family(dataset_source)
    if family in _DIRECTION_GT_DISABLED_DATASET_FAMILIES:
        raise ValueError(
            "dataset has no approved Reasoning V4 direction GT: "
            f"{family}"
        )
    raw = _raw_direction_record(row)
    profile_id = (
        direction.profile_id
        if direction is not None
        else str((raw or {}).get("profile_id") or "").strip()
    )
    if not profile_id:
        raise ValueError(
            "Reasoning V4 source requires explicit direction_profile_id"
        )
    profile = resolve_stage_a_direction_profile(profile_id)
    probe_profile_id = affordance_probe_profile_for_action(profile_id)
    probe = resolve_affordance_probe_profile(probe_profile_id)
    return DirectionContextV4(
        action_profile_id=profile_id,
        action_profile_version=str(profile["profile_version"]),
        affordance_probe_profile_id=probe_profile_id,
        affordance_probe_profile_version=str(probe["profile_version"]),
    )


def _direction_geometry_v1(
    row: Mapping[str, Any],
    *,
    direction: DirectionSupervisionV4 | None,
) -> tuple[DirectionGeometrySupervisionV1 | None, list[str]]:
    """Reuse or construct the private raw-angle geometry sidecar.

    A missing raw angle remains explicit ``unavailable`` geometry for a valid
    moving strong label.  The builder never substitutes a discrete candidate
    center for missing continuous teacher geometry.
    """

    if direction is None:
        return None, ["direction_geometry_missing_strong_direction"]
    existing = row.get("direction_geometry")
    if existing is not None:
        if not isinstance(existing, Mapping):
            raise ValueError("direction_geometry must be an object")
        typed = DirectionGeometrySupervisionV1.model_validate(existing)
        return typed, []
    geometry = build_direction_geometry_supervision_v1(
        row,
        direction=direction,
    )
    warnings: list[str] = []
    if geometry.confidence == "unavailable":
        warnings.append("direction_geometry_raw_angle_missing")
    if geometry.is_stop and not geometry.stop_rule_verified:
        warnings.append("direction_geometry_stop_displacement_unverified")
    return geometry, warnings


def _has_explicit_v4_direction_stamp(row: Mapping[str, Any]) -> bool:
    """Return whether a row explicitly claims a canonical V4 direction."""

    return (
        row.get("direction_supervision") is not None
        or row.get("canonical_direction") is not None
        or any(
            field in row
            for field in (
                "direction_profile_id",
                "direction_label_index",
                "direction_label_name",
            )
        )
    )


def _raw_direction_record(
    row: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Read a nested or canonical flat direction record."""

    nested = row.get("direction_supervision") or row.get("canonical_direction")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("direction supervision must be an object")
        return {
            "profile_id": nested.get("profile_id"),
            "profile_version": nested.get(
                "profile_version",
                nested.get("version"),
            ),
            "label_index": nested.get(
                "label_index",
                nested.get("index"),
            ),
            "label_name": nested.get(
                "label_name",
                nested.get("name"),
            ),
            "choice": nested.get("choice"),
            "angle_deg": nested.get(
                "angle_deg",
                nested.get("angle"),
            ),
            "source": nested.get("source"),
            "bin_mode": nested.get("bin_mode"),
            "derivation_version": nested.get("derivation_version"),
        }
    flat_fields = (
        _DIRECTION_V4_FLAT_FIELDS
        if any(
            field in row
            for field in (
                "direction_profile_id",
                "direction_label_index",
                "direction_label_name",
            )
        )
        else _DIRECTION_FLAT_FIELDS
    )
    if not any(field in row for field in flat_fields.values()):
        return None
    return {
        target: row.get(source)
        for target, source in flat_fields.items()
    }


def _system1_sidecars_v4(
    row: Mapping[str, Any],
    *,
    dataset_source: str,
    config: ReasoningSourceBuildConfig,
) -> tuple[
    list[list[float]] | None,
    dict[str, Any] | None,
    list[bool] | list[int] | None,
    str | None,
]:
    """Copy one trusted System1 target or return a disable reason."""

    if not config.enable_system1:
        return None, None, None, "system1_disabled_by_config"
    family = _dataset_family(dataset_source)
    if family == "frodobots":
        return None, None, None, "frodobots_system1_disabled"
    raw = row.get("trajectory_ego_t")
    if raw is None:
        return None, None, None, "trajectory_ego_t_missing"
    policy = (
        row.get("trajectory_policy")
        if isinstance(row.get("trajectory_policy"), Mapping)
        else {}
    )
    source_format = str(
        policy.get("system1_target_format")
        or row.get("system1_target_format")
        or ""
    ).strip() or None
    try:
        trajectory = canonicalize_trajectory_ego_t(
            raw,
            source_format=source_format,
        )
        canonical_policy = canonical_trajectory_policy(
            policy,
            horizon_steps=len(trajectory),
            yaw_source=str(policy.get("yaw_source") or "path_tangent"),
        )
    except (TypeError, ValueError) as exc:
        return None, None, None, f"trajectory_contract_error:{exc}"
    action_mask = row.get("action_mask")
    if not isinstance(action_mask, list) or len(action_mask) != len(trajectory):
        action_mask = [True] * len(trajectory)
    return trajectory, canonical_policy, action_mask, None


def _waypoint_is_prompt_visible(row: Mapping[str, Any]) -> bool:
    """Reject waypoints derived from future demonstration or unknown provenance."""

    explicit = row.get("waypoint_prompt_visible")
    if explicit is not None:
        return bool(explicit)
    source = str(
        row.get("waypoint_source")
        or row.get("route_points_source")
        or ""
    ).strip().lower()
    if any(
        marker in source
        for marker in (
            "ground_truth",
            "ground-truth",
            "trajectory",
            "demonstration",
            "future_pose",
            "gt_",
            "_gt",
        )
    ):
        return False
    return source in {
        "prompt",
        "human_annotation",
        "dataset_input",
        "goal_encoder",
        "visual_prompt",
    }


def _destination_is_prompt_visible(
    row: Mapping[str, Any],
    destination: Any,
) -> bool:
    """Allow textual destinations or explicitly visible goal conditions."""

    explicit = row.get("goal_prompt_visible")
    if explicit is not None:
        return bool(explicit)
    if isinstance(destination, str):
        return bool(destination.strip())
    source = str(
        row.get("goal_source")
        or row.get("destination_source")
        or ""
    ).strip().lower()
    return source in {
        "prompt",
        "instruction",
        "human_annotation",
        "dataset_input",
        "goal_encoder",
        "visual_prompt",
    }


def _sanitize_visible_value(value: Any) -> Any:
    """Remove private geometry, tracking, and supervision keys recursively."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.lower()
            if any(
                part in normalized
                for part in V4_FORBIDDEN_VISIBLE_CONTEXT_KEY_PARTS
            ):
                continue
            result[key] = _sanitize_visible_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_visible_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    """Return the first non-null source value among candidate keys."""

    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def _dataset_family(value: str) -> str:
    """Normalize one dataset id to a V4 supervision family."""

    normalized = str(value).strip().lower().replace("_", "-")
    for family in _SUPPORTED_DATASET_FAMILIES:
        if normalized.startswith(family):
            return family
    raise ValueError(
        f"unsupported Reasoning V4 dataset family: {value!r}"
    )
