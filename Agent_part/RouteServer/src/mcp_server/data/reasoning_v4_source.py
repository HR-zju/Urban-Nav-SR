"""Build leakage-bounded single-turn Reasoning V4 sources."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from mcp_server.contracts.direction_profile import (
    affordance_probe_profile_for_action,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
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
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_source import (
    ReasoningSourceBuildConfig,
    _dataset_source,
    _ordered_images,
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
_DIRECTION_DATASET_ALLOWLIST = {
    "sidewalk": frozenset({"fpv_degree10_terminal_centroid_v2"}),
    "foxglove": frozenset({"fpv_degree10_terminal_centroid_v2"}),
    "egowalk": frozenset({"ego_degree10_12way_terminal_centroid_v2"}),
    "frodobots": frozenset(),
}


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
    images, image_roles = _ordered_images(row, config=config)
    instruction = _optional_instruction(row)
    visible_context, context_warnings = _visible_context_v4(row)
    evidence_catalog = _evidence_catalog_v4(
        row=row,
        dataset_source=dataset_source,
        image_roles=image_roles,
        instruction=instruction,
        visible_context=visible_context,
    )
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
        "trajectory_ego_t": trajectory,
        "trajectory_policy": trajectory_policy,
        "action_mask": action_mask,
        "warnings": warnings,
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return ReasoningSourceV4.model_validate(payload)


def build_reasoning_source_manifest_v4(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    config: ReasoningSourceBuildConfig,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Build a V4 source manifest and deterministic conversion report.

    Args:
        input_jsonl: Heterogeneous dataset manifest.
        output_jsonl: Destination frozen V4 source manifest.
        config: Source build policy.
        limit: Optional maximum number of input rows.
        skip_errors: Continue after invalid rows when true.

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
            source = build_reasoning_source_v4(
                row,
                index=index,
                config=config,
            )
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
    if overlay_visible and family in {"sidewalk", "foxglove"}:
        endpoint_semantics = str(
            overlay_meta.get("route_endpoint_semantics")
            or (
                "crop_boundary"
                if overlay_scope == "local_future"
                else "recorded_clip_endpoint"
            )
        )
        alignment = str(
            row.get("bev_image_alignment")
            or metadata.get("bev_alignment")
            or "ego_aligned"
        )
        ego_aligned = alignment == "ego_aligned"
        return BevRouteEvidenceV4(
            frame="ego" if ego_aligned else "north_up",
            ego_forward="up" if ego_aligned else "unknown",
            route_overlay_visible=True,
            route_direction_authority=(
                "visual_conditional" if ego_aligned else "none"
            ),
            route_progress_authority="none",
            route_terminal_marker_visible=True,
            route_endpoint_semantics=endpoint_semantics,
            producer_contract_id="sidewalk_bev_route_renderer_legacy",
            producer_contract_version="1.0",
        )
    return BevRouteEvidenceV4(
        route_overlay_visible=overlay_visible,
        route_direction_authority=(
            "visual_conditional" if overlay_visible else "none"
        ),
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
    approved = _DIRECTION_DATASET_ALLOWLIST.get(family)
    if approved is None:
        raise ValueError(
            f"unsupported V4 dataset direction policy: {dataset_source}"
        )
    if raw is None:
        return None, ["canonical_direction_missing"]
    if not approved:
        return None, [f"{family}_direction_ignored_by_policy"]
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
    if direction.profile_id not in approved:
        raise ValueError(
            "direction profile is not approved for dataset: "
            f"dataset={family}, profile={direction.profile_id}"
        )
    return direction, []


def _direction_context_v4(
    row: Mapping[str, Any],
    *,
    dataset_source: str,
    direction: DirectionSupervisionV4 | None,
) -> DirectionContextV4:
    """Build public action/probe identities without repeating label arrays."""

    family = _dataset_family(dataset_source)
    approved = _DIRECTION_DATASET_ALLOWLIST.get(family)
    if approved is None:
        raise ValueError(
            f"unsupported V4.5 direction policy: {dataset_source}"
        )
    if not approved:
        raise ValueError(
            f"no approved V4.5 direction profile for dataset: {family}"
        )
    raw = _raw_direction_record(row)
    profile_id = (
        direction.profile_id
        if direction is not None
        else str((raw or {}).get("profile_id") or "").strip()
    )
    if not profile_id:
        if len(approved) != 1:
            raise ValueError(
                "direction output profile is ambiguous for dataset"
            )
        profile_id = next(iter(approved))
    if profile_id not in approved:
        raise ValueError(
            "direction output profile is not approved for dataset: "
            f"dataset={family}, profile={profile_id}"
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
    for family in _DIRECTION_DATASET_ALLOWLIST:
        if normalized.startswith(family):
            return family
    return normalized
