"""Normalize heterogeneous navigation assets into frozen ReasoningSourceV2 rows."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from mcp_server.contracts.reasoning_dataset import (
    REASONING_SOURCE_V2_SCHEMA_VERSION,
    EvidenceCatalogEntryV2,
    ReasoningSourceV2,
    reasoning_source_fingerprint,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.runtime.system1_contract import (
    SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1,
    canonical_trajectory_policy,
    canonicalize_trajectory_ego_t,
)


_FORBIDDEN_VISIBLE_KEY_PARTS = (
    "ground_truth",
    "gt_",
    "_gt",
    "pixel_trajectory",
    "trajectory_fpv",
    "trajectory_ego_t",
    "action_ego_t",
    "segmentation_label",
    "teacher_private",
    "private_judgment",
)
_PSEUDO_GOAL_MARKERS = ("trajectory", "demonstration", "future_pose", "ground_truth")
_DEFAULT_SYSTEM1_DISABLED_DATASETS = frozenset({"frodobots", "frodobots-2k", "frodobots-7k"})


@dataclass(frozen=True)
class ReasoningSourceBuildConfig:
    """Configure conservative legacy-row normalization.

    Args:
        dataset_source: Explicit source-family name written to every output.
        default_instruction: Instruction used only when a legacy local-nav row
            has no instruction field.
        travel_mode: Default travel mode.
        max_fpv_frames: Maximum temporal FPV frames, including current frame.
        include_current_bev: Whether to keep the newest explicit BEV frame.
        enable_system1: Whether valid trajectory rows may carry CFM targets.
        disabled_system1_datasets: Dataset names whose motion labels are not
            trusted for continuous supervision.
        allow_pseudo_goal_condition: Whether route/goal encoder inputs derived
            from demonstration trajectory may remain model-visible.
        strict_image_roles: Reject ambiguous top-level image lists instead of
            guessing roles from image count.
        media_root: Optional root used to resolve and validate relative image
            paths before the source fingerprint is frozen.
    """

    dataset_source: str
    default_instruction: str = "Continue safe pedestrian navigation."
    travel_mode: str = "pedestrian_assist"
    max_fpv_frames: int = 5
    include_current_bev: bool = True
    enable_system1: bool = True
    disabled_system1_datasets: frozenset[str] = _DEFAULT_SYSTEM1_DISABLED_DATASETS
    allow_pseudo_goal_condition: bool = False
    strict_image_roles: bool = True
    media_root: Path | None = None


def build_reasoning_source_v2(
    row: Mapping[str, Any],
    *,
    index: int,
    config: ReasoningSourceBuildConfig,
) -> ReasoningSourceV2:
    """Convert one heterogeneous row into a frozen V2 teacher input.

    Args:
        row: Legacy or partially normalized source row.
        index: Stable input-file index used only when no source id exists.
        config: Source build policy.

    Returns:
        Validated ``ReasoningSourceV2``.

    Raises:
        ValueError: If image roles are ambiguous, no image exists, or a hard
            source contract cannot be constructed.
    """

    if row.get("schema_version") == REASONING_SOURCE_V2_SCHEMA_VERSION:
        return ReasoningSourceV2.model_validate(row)

    dataset_source = _dataset_source(row, fallback=config.dataset_source)
    source_record_id = _source_record_id(row, index=index)
    source_namespace = _source_namespace(config.dataset_source, dataset_source=dataset_source)
    source_group_id = str(row.get("source_group_id") or f"{source_namespace}:{source_record_id}")
    split_group_id = _split_group_id(row, source_record_id=source_record_id)
    images, image_roles = _ordered_images(row, config=config)
    instruction = str(
        row.get("instruction")
        or row.get("navigation_instruction")
        or row.get("user_request")
        or config.default_instruction
    ).strip()
    if not instruction:
        raise ValueError("reasoning source requires a non-empty instruction")

    context = _build_model_visible_context(row, instruction=instruction)
    pseudo_goal = _is_pseudo_goal(row)
    goal_geometry = _first_present(row, "goal_geometry", "goal_ego_t", "goal_gps")
    route_points = _first_present(row, "route_points_ego_t", "waypoints_ego_t")
    dropped_pseudo_goal = False
    if pseudo_goal and not config.allow_pseudo_goal_condition:
        goal_geometry = None
        route_points = None
        context.pop("goal", None)
        context.pop("waypoints", None)
        dropped_pseudo_goal = True

    independent_route = _has_independent_route_evidence(row)
    dropped_unverified_route = False
    if not independent_route:
        # route/waypoint 必须来自地图、工具或人工标注；不能把 GT 派生方向伪装成输入证据。
        dropped_unverified_route = any(
            value is not None
            for value in (context.pop("route", None), context.pop("waypoints", None), route_points)
        )
        route_points = None
    catalog = _build_evidence_catalog(
        row,
        image_roles=image_roles,
        context=context,
        independent_route=independent_route,
    )
    modalities = _available_modalities(
        image_roles=image_roles,
        context=context,
        catalog=catalog,
        independent_route=independent_route,
    )
    legacy_qa_audit = _legacy_qa_audit(row)

    trajectory, trajectory_policy, action_mask, system1_reason = _system1_sidecars(
        row,
        dataset_source=dataset_source,
        config=config,
    )
    metadata = {
        "source_index": int(index),
        "source_fields": sorted(str(key) for key in row.keys()),
        "image_selection_policy": "temporal_fpv_then_current_bev_v2",
        "independent_route_evidence": independent_route,
        "unverified_route_context_dropped": dropped_unverified_route,
        "pseudo_goal_detected": pseudo_goal,
        "pseudo_goal_condition_dropped": dropped_pseudo_goal,
        "system1_disabled_reason": system1_reason,
        "source_path": row.get("source_path") or row.get("manifest_path"),
        "source_namespace": source_namespace,
        "media_root": str(config.media_root) if config.media_root is not None else None,
    }
    _drop_none(metadata)
    payload: dict[str, Any] = {
        "schema_version": REASONING_SOURCE_V2_SCHEMA_VERSION,
        "source_group_id": source_group_id,
        "dataset_source": dataset_source,
        "source_record_id": source_record_id,
        "split_group_id": split_group_id,
        "images": images,
        "image_roles": image_roles,
        "available_modalities": modalities,
        "instruction": instruction,
        "travel_mode": str(row.get("travel_mode") or row.get("mode") or config.travel_mode),
        "model_visible_context": context,
        "evidence_catalog": [item.model_dump(mode="json", exclude_none=True) for item in catalog],
        "goal_geometry": goal_geometry,
        "route_points_ego_t": route_points,
        "trajectory_ego_t": trajectory,
        "trajectory_policy": trajectory_policy,
        "action_mask": action_mask,
        "system1_enabled": trajectory is not None,
        "reasoning_qa_label_validity": legacy_qa_audit,
        # 私有字段只用于离线审计；teacher prompt builder 默认完全忽略该字段。
        "teacher_private_context": _teacher_private_context(row),
        "metadata": metadata,
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint(payload)
    return ReasoningSourceV2.model_validate(payload)


def build_reasoning_source_manifest(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    config: ReasoningSourceBuildConfig,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Build a deterministic source manifest and conversion report.

    Args:
        input_jsonl: Heterogeneous source manifest.
        output_jsonl: Frozen V2 output manifest.
        config: Normalization policy.
        limit: Optional maximum input rows.
        skip_errors: Continue after invalid source rows.

    Returns:
        Conversion counts and a bounded error list.

    Raises:
        ValueError: On the first invalid row when ``skip_errors`` is false.
    """

    errors: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    def normalized_rows() -> Iterable[dict[str, Any]]:
        for index, row in enumerate(iter_jsonl(input_jsonl)):
            if limit is not None and index >= int(limit):
                break
            counters["input_rows"] += 1
            try:
                source = build_reasoning_source_v2(row, index=index, config=config)
            except (TypeError, ValueError) as exc:
                counters["rejected_rows"] += 1
                if len(errors) < 100:
                    errors.append({"index": index, "id": row.get("id"), "error": str(exc)})
                if not skip_errors:
                    raise
                continue
            counters["written_rows"] += 1
            counters["system1_rows" if source.system1_enabled else "lm_only_rows"] += 1
            fpv_frame_count = sum(role.startswith("fpv.") for role in source.image_roles)
            counters[f"fpv_frames:{fpv_frame_count}"] += 1
            if fpv_frame_count > 1:
                counters["temporal_fpv_rows"] += 1
            if "bev" in source.available_modalities:
                counters["fpv_bev_rows"] += 1
            else:
                counters["fpv_only_rows"] += 1
            yield source.model_dump(mode="json", exclude_none=True)

    write_jsonl_atomic(output_jsonl, normalized_rows())
    return {
        "schema_version": "urbannav_reasoning_source_v2_build_report",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "counts": dict(counters),
        "errors": errors[:100],
    }


def _ordered_images(
    row: Mapping[str, Any],
    *,
    config: ReasoningSourceBuildConfig,
) -> tuple[list[str], list[str]]:
    """Resolve explicit image roles and apply the temporal-view policy."""

    images = _string_list(row.get("images"))
    roles = _string_list(row.get("image_roles"))
    if images and roles:
        if len(images) != len(roles):
            raise ValueError("legacy images/image_roles length mismatch")
        pairs = [(image, _canonical_image_role(role)) for image, role in zip(images, roles)]
        return _select_image_pairs(pairs, config=config)

    inferred_pairs = _paired_images_from_metadata(row, images=images)
    if inferred_pairs:
        return _select_image_pairs(inferred_pairs, config=config)

    fpv = _string_list(row.get("fpv_image_paths"))
    if not fpv:
        fpv = _string_list(row.get("history_fpv_image_paths"))
        current = _first_present(row, "fpv_image_path", "front_image_path", "image_path")
        if current is not None:
            # Repeated paths are valid temporal padding and still require a current slot.
            fpv.append(str(current))
    bev = _string_list(row.get("bev_image_paths"))
    if not bev:
        current_bev = _first_present(row, "bev_image_path", "map_image_path")
        if current_bev is not None:
            bev = [str(current_bev)]
    goal_images = _string_list(row.get("goal_images") or row.get("goal_image_paths"))
    if row.get("goal_image_path") and str(row["goal_image_path"]) not in goal_images:
        goal_images.append(str(row["goal_image_path"]))
    if fpv or bev or goal_images:
        pairs: list[tuple[str, str]] = []
        for index, image in enumerate(fpv):
            role = "fpv.current" if index == len(fpv) - 1 else f"fpv.history_{index:02d}"
            pairs.append((image, role))
        pairs.extend((image, f"bev.history_{index:02d}") for index, image in enumerate(bev[:-1]))
        if bev:
            pairs.append((bev[-1], "bev.current"))
        pairs.extend((image, f"goal.image_{index:02d}") for index, image in enumerate(goal_images))
        return _select_image_pairs(pairs, config=config)

    if images and len(images) == 1 and not config.strict_image_roles:
        return images, ["fpv.current"]
    if images:
        raise ValueError("ambiguous top-level images require explicit image_roles")
    raise ValueError("reasoning source has no image input")


def _dataset_source(row: Mapping[str, Any], *, fallback: str) -> str:
    """Resolve dataset provenance without relying on a merged-file default."""

    sidewalk_meta = row.get("sidewalk_meta")
    if not isinstance(sidewalk_meta, Mapping):
        sidewalk_meta = {}
    instruction_meta = row.get("instruction_meta")
    if not isinstance(instruction_meta, Mapping):
        instruction_meta = {}
    value = (
        row.get("dataset_source")
        or row.get("source_dataset")
        or sidewalk_meta.get("source_dataset_resolved")
        or instruction_meta.get("source_dataset")
        or fallback
    )
    normalized = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "sidewalk-main": "sidewalk",
        "sidewalk-extra": "sidewalk",
        "foxglove-yaw-p90": "foxglove",
        "egowalk-fpv-only": "egowalk",
        "frodobots-qc": "frodobots",
    }
    return aliases.get(normalized, normalized)


def _source_namespace(configured_source: str, *, dataset_source: str) -> str:
    """Keep manifest-part identity separate from the canonical dataset family."""

    configured = str(configured_source or "").strip().lower().replace("_", "-")
    if configured and configured not in {"auto", "merged", "unknown"}:
        return configured
    return dataset_source


def _paired_images_from_metadata(
    row: Mapping[str, Any],
    *,
    images: list[str],
) -> list[tuple[str, str]]:
    """Recover image roles only from an explicit interleaved-pair contract."""

    if not images:
        return []
    sidewalk_meta = row.get("sidewalk_meta")
    if not isinstance(sidewalk_meta, Mapping):
        return []
    pair_order = str(
        sidewalk_meta.get("multiframe_pair_order")
        or sidewalk_meta.get("multiframe_order")
        or ""
    ).strip().lower()
    if pair_order not in {
        "fpv_bev_pairs",
        "fpv_bev_pairs_oldest_to_newest",
    }:
        return []
    if len(images) % 2:
        raise ValueError("explicit fpv/bev pair order requires an even image count")
    pairs: list[tuple[str, str]] = []
    pair_count = len(images) // 2
    for index in range(pair_count):
        fpv = images[index * 2]
        bev = images[index * 2 + 1]
        fpv_role = "fpv.current" if index == pair_count - 1 else f"fpv.history_{index:02d}"
        bev_role = "bev.current" if index == pair_count - 1 else f"bev.history_{index:02d}"
        pairs.extend(((fpv, fpv_role), (bev, bev_role)))
    return pairs


def _select_image_pairs(
    pairs: list[tuple[str, str]],
    *,
    config: ReasoningSourceBuildConfig,
) -> tuple[list[str], list[str]]:
    """Keep the newest FPV window, current BEV, and explicit goal images."""

    fpv_pairs = [pair for pair in pairs if pair[1].startswith("fpv.")]
    bev_pairs = [pair for pair in pairs if pair[1].startswith("bev.")]
    other_pairs = [pair for pair in pairs if not pair[1].startswith(("fpv.", "bev."))]
    fpv_pairs = fpv_pairs[-max(1, int(config.max_fpv_frames)) :]
    if fpv_pairs:
        fpv_pairs = [
            (image, "fpv.current" if index == len(fpv_pairs) - 1 else f"fpv.history_{index:02d}")
            for index, (image, _) in enumerate(fpv_pairs)
        ]
    bev_pairs = bev_pairs[-1:] if config.include_current_bev else []
    if bev_pairs:
        bev_pairs = [(bev_pairs[-1][0], "bev.current")]
    selected = [*fpv_pairs, *bev_pairs, *other_pairs]
    if not any(role == "fpv.current" for _, role in selected):
        raise ValueError("reasoning source requires an explicit current FPV image")
    images = [_resolve_image_path(image, media_root=config.media_root) for image, _ in selected]
    roles = [role for _, role in selected]
    if len(set(roles)) != len(roles):
        raise ValueError(f"duplicate canonical image roles: {roles}")
    return images, roles


def _resolve_image_path(value: str, *, media_root: Path | None) -> str:
    """Resolve one image path against an explicit mounted dataset root.

    Args:
        value: Absolute or manifest-relative image path.
        media_root: Optional mounted dataset root.

    Returns:
        Original path when no root is configured, otherwise an absolute path.

    Raises:
        ValueError: If the root-backed path does not resolve to a file.
    """

    path = Path(str(value)).expanduser()
    if media_root is None:
        return str(path)
    candidate = path if path.is_absolute() else Path(media_root).expanduser() / path
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"reasoning source image does not exist under media_root: {resolved}")
    return str(resolved)


def _build_model_visible_context(row: Mapping[str, Any], *, instruction: str) -> dict[str, Any]:
    """Build a leakage-filtered context dictionary shown to the teacher/model."""

    context_value = row.get("model_visible_context") or row.get("reasoning_input_context") or {}
    context = _sanitize_visible_value(context_value) if isinstance(context_value, Mapping) else {}
    context = dict(context or {})
    context["instruction"] = {"ref": "instruction.active", "text": instruction}

    goal_value = _first_present(row, "goal_spec", "goal", "destination", "goal_text")
    if goal_value is not None and "goal" not in context:
        context["goal"] = {"ref": "goal.active", "value": _sanitize_visible_value(goal_value)}
    route_value = _first_present(row, "route_context", "route_summary", "route_instruction", "route_state")
    if route_value is not None and "route" not in context:
        context["route"] = {"ref": "route.active", "value": _sanitize_visible_value(route_value)}
    waypoints = _first_present(row, "waypoints", "coarse_waypoints")
    if waypoints is not None and "waypoints" not in context:
        context["waypoints"] = _sanitize_visible_value(waypoints)
    for key in ("tracked_agents", "traffic_controls", "selected_path", "tool_response_context"):
        if row.get(key) is not None and key not in context:
            context[key] = _sanitize_visible_value(row[key])
    return context


def _build_evidence_catalog(
    row: Mapping[str, Any],
    *,
    image_roles: list[str],
    context: Mapping[str, Any],
    independent_route: bool,
) -> list[EvidenceCatalogEntryV2]:
    """Construct the complete set of model-visible evidence addresses."""

    entries: dict[str, EvidenceCatalogEntryV2] = {}

    def add(ref: str, modality: str, summary: str | None = None, **attributes: Any) -> None:
        if ref in entries:
            return
        entries[ref] = EvidenceCatalogEntryV2(
            ref=ref,
            modality=modality,
            summary=summary,
            attributes={key: value for key, value in attributes.items() if value is not None},
        )

    for value in row.get("evidence_catalog") or ():
        if isinstance(value, str):
            add(value, _modality_for_ref(value))
        elif isinstance(value, Mapping) and value.get("ref"):
            entry = EvidenceCatalogEntryV2.model_validate(value)
            entries[entry.ref] = entry
    add("instruction.active", "instruction", "active navigation instruction")
    for role in image_roles:
        if role == "fpv.current":
            add("fpv.current", "fpv", "current first-person observation")
            # Corridor refs address image regions; they are not passability labels.
            for side in ("left", "center", "right"):
                add(f"corridor.{side}", "fpv", f"current FPV {side} corridor region")
        elif role.startswith("fpv.history_"):
            add(role, "fpv", "historical first-person observation")
        elif role == "bev.current":
            add(
                "bev.current",
                "bev",
                "optional bird's-eye route and topology context",
                **_bev_evidence_attributes(row),
            )
        elif role.startswith("goal.image_"):
            add(role, "goal_image", "visual goal reference")
    if context.get("goal") is not None:
        add("goal.active", "goal", "active goal specification")
    if independent_route:
        add("route.active", "route", "active route or map-derived local direction")
    waypoints = context.get("waypoints")
    if independent_route and isinstance(waypoints, list):
        for index, _ in enumerate(waypoints[:32]):
            add(f"waypoint.wp_{index:02d}", "waypoint", "active route waypoint")
    for index, agent in enumerate(_mapping_list(context.get("tracked_agents"))):
        identifier = _safe_ref_segment(agent.get("track_id") or agent.get("id") or index)
        add(f"track.{identifier}", "track", "temporally tracked agent")
    for index, control in enumerate(_mapping_list(context.get("traffic_controls"))):
        identifier = _safe_ref_segment(control.get("control_id") or control.get("id") or index)
        add(f"control.{identifier}", "control", "visible traffic control")
    if context.get("selected_path") is not None:
        add("selected_path.active", "selected_path", "model-visible selected local path")
    tool_context = context.get("tool_response_context")
    if tool_context is not None:
        add("tool.latest", "tool", "latest frozen tool response")
    return [entries[key] for key in sorted(entries)]


def _bev_evidence_attributes(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build model-visible BEV semantics without leaking private GT labels.

    Args:
        row: Original heterogeneous manifest row.

    Returns:
        Stable BEV-role metadata and optional synchronization metadata already
        present in the source row.
    """

    result: dict[str, Any] = {
        "semantic_role": "optional_route_topology_context",
        "local_safety_authority": False,
        "requires_fpv_safety_check": True,
        "visual_convention": (
            "Interpret visible route overlays, ego/goal markers, and local topology only; "
            "do not infer unseen FPV clearance from BEV."
        ),
    }
    for key in ("fps", "frame_dt_sec", "bev_frame_delta", "bev_time_delta_sec"):
        value = row.get(key)
        if value is not None:
            result[key] = _sanitize_visible_value(value)
    return result


def _available_modalities(
    *,
    image_roles: list[str],
    context: Mapping[str, Any],
    catalog: Iterable[EvidenceCatalogEntryV2],
    independent_route: bool,
) -> list[str]:
    """Derive prompt-visible modality flags without reading supervision fields."""

    modalities = {"instruction"}
    if any(role.startswith("fpv.") for role in image_roles):
        modalities.add("fpv")
    if sum(role.startswith("fpv.") for role in image_roles) > 1:
        modalities.add("temporal_fpv")
    if any(role.startswith("bev.") for role in image_roles):
        modalities.add("bev")
    if any(role.startswith("goal.image_") for role in image_roles):
        modalities.add("goal_image")
    if context.get("goal") is not None:
        modalities.add("goal")
    if independent_route:
        modalities.add("route")
    if context.get("waypoints") is not None and independent_route:
        modalities.add("waypoint")
    modalities.update(item.modality for item in catalog if item.modality in {"track", "control", "selected_path", "tool"})
    return sorted(modalities)


def _legacy_qa_audit(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Preserve frozen legacy QA metadata without deriving training labels."""

    value = row.get("reasoning_qa_label_validity") or {}
    return {str(key): dict(item) for key, item in value.items() if isinstance(item, Mapping)}


def _system1_sidecars(
    row: Mapping[str, Any],
    *,
    dataset_source: str,
    config: ReasoningSourceBuildConfig,
) -> tuple[list[list[float]] | None, dict[str, Any] | None, list[bool] | list[int] | None, str | None]:
    """Canonicalize a trusted trajectory or return an explicit disable reason."""

    if not config.enable_system1:
        return None, None, None, "disabled_by_config"
    normalized_dataset = dataset_source.strip().lower()
    if normalized_dataset in {item.lower() for item in config.disabled_system1_datasets}:
        return None, None, None, "dataset_motion_labels_disabled"
    raw = row.get("trajectory_ego_t")
    if raw is None:
        return None, None, None, "trajectory_ego_t_missing"
    policy = row.get("trajectory_policy") if isinstance(row.get("trajectory_policy"), Mapping) else {}
    source_format = str(policy.get("system1_target_format") or row.get("system1_target_format") or "").strip() or None
    try:
        trajectory = canonicalize_trajectory_ego_t(raw, source_format=source_format)
        yaw_source = str(policy.get("yaw_source") or "path_tangent")
        canonical_policy = canonical_trajectory_policy(policy, horizon_steps=len(trajectory), yaw_source=yaw_source)
    except (TypeError, ValueError) as exc:
        return None, None, None, f"trajectory_contract_error:{exc}"
    action_mask = row.get("action_mask")
    if not isinstance(action_mask, list) or len(action_mask) != len(trajectory):
        action_mask = [True] * len(trajectory)
    return trajectory, canonical_policy, action_mask, None


def _teacher_private_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded audit-only provenance never rendered into prompts."""

    value = row.get("teacher_private_context")
    context = dict(value) if isinstance(value, Mapping) else {}
    context.setdefault("has_pixel_trajectory", row.get("pixel_trajectory") is not None)
    context.setdefault("has_segmentation_label", row.get("semantic_segmentation_path") is not None)
    context.setdefault("goal_source", row.get("goal_source"))
    _drop_none(context)
    return context


def _has_independent_route_evidence(
    row: Mapping[str, Any],
) -> bool:
    """Return whether route context is independent from future demonstration GT."""

    if row.get("independent_route_evidence") is not None:
        return bool(row.get("independent_route_evidence"))

    provenance_values: list[str] = []
    for key in (
        "route_source",
        "route_provenance",
        "waypoint_source",
        "route_points_source",
        "map_provider",
        "route_provider",
    ):
        value = row.get(key)
        if value is not None:
            provenance_values.append(str(value).strip().lower())
    route_context = row.get("route_context")
    if isinstance(route_context, Mapping):
        for key in ("source", "provenance", "provider"):
            value = route_context.get(key)
            if value is not None:
                provenance_values.append(str(value).strip().lower())

    joined = " ".join(provenance_values)
    derived_markers = (
        "ground_truth",
        "ground-truth",
        "trajectory",
        "demonstration",
        "future_pose",
        "future pose",
        "gt_",
        "_gt",
    )
    if any(marker in joined for marker in derived_markers) or any(
        value in {"gt", "groundtruth"} for value in provenance_values
    ):
        return False
    independent_markers = (
        "map_provider",
        "route_session",
        "tool_response",
        "human_annotation",
        "frozen_route",
        "route_geometry",
        "routing_api",
        "router",
        "osm",
        "amap",
        "baidu",
        "google",
    )
    if any(marker in joined for marker in independent_markers):
        return True
    if row.get("route_session_id") is not None or row.get("route_provider_cache_key") is not None:
        return True
    # route_context/route_points 单独出现时来源不明，必须由上游显式声明 provenance。
    return False


def _is_pseudo_goal(row: Mapping[str, Any]) -> bool:
    """Identify goal/waypoint conditions derived from future demonstration labels."""

    source = str(row.get("goal_source") or row.get("route_points_source") or "").lower()
    return any(marker in source for marker in _PSEUDO_GOAL_MARKERS)


def _sanitize_visible_value(value: Any) -> Any:
    """Recursively remove known supervision/private keys from visible context."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if any(part in normalized_key for part in _FORBIDDEN_VISIBLE_KEY_PARTS):
                continue
            result[str(key)] = _sanitize_visible_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_visible_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _source_record_id(row: Mapping[str, Any], *, index: int) -> str:
    """Resolve a stable source record id."""

    for key in ("source_record_id", "id", "sample_id", "case_id", "row_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    metadata = row.get("sidewalk_meta")
    if not isinstance(metadata, Mapping):
        metadata = {}
    clip = str(
        row.get("clip_name")
        or row.get("clip_id")
        or metadata.get("clip_id")
        or "row"
    ).strip()
    frame = row.get("fpv_frame_idx")
    if frame is None:
        frame = row.get("frame_idx")
    if frame is None:
        frame = metadata.get("frame_name")
    if frame is None:
        frame = metadata.get("frame_idx")
    return f"{clip}:{frame if frame is not None else index}"


def _split_group_id(row: Mapping[str, Any], *, source_record_id: str) -> str:
    """Resolve the clip/session grouping key used before train/eval splitting."""

    for key in ("split_group_id", "session_id", "route_session_id", "clip_name", "clip_id", "sequence_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    metadata = row.get("sidewalk_meta")
    if isinstance(metadata, Mapping):
        clip_id = str(metadata.get("clip_id") or "").strip()
        if clip_id:
            return clip_id
    return source_record_id


def _canonical_image_role(value: str) -> str:
    """Normalize common legacy image-role spellings."""

    text = str(value).strip().lower().replace("_", ".")
    aliases = {
        "fpv": "fpv.current",
        "front": "fpv.current",
        "front.current": "fpv.current",
        "fpv.current": "fpv.current",
        "bev": "bev.current",
        "map": "bev.current",
        "bev.current": "bev.current",
        "goal": "goal.image_00",
        "goal.image": "goal.image_00",
    }
    if text in aliases:
        return aliases[text]
    history = re.fullmatch(r"(fpv|bev)\.history\.?([0-9]+)", text)
    if history:
        return f"{history.group(1)}.history_{int(history.group(2)):02d}"
    goal = re.fullmatch(r"goal\.image\.?([0-9]+)", text)
    if goal:
        return f"goal.image_{int(goal.group(1)):02d}"
    return str(value).strip()


def _modality_for_ref(ref: str) -> str:
    """Infer an evidence modality only from an explicit reference prefix."""

    prefix = str(ref).split(".", 1)[0].lower()
    return {
        "fpv": "fpv",
        "corridor": "fpv",
        "bev": "bev",
        "instruction": "instruction",
        "goal": "goal",
        "route": "route",
        "waypoint": "waypoint",
        "track": "track",
        "control": "control",
        "selected_path": "selected_path",
        "tool": "tool",
    }.get(prefix, "context")


def _safe_ref_segment(value: Any) -> str:
    """Convert an external id to one EvidenceRef-safe segment."""

    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "unknown"
    return text[:80]


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    """Return mapping elements from a possible list."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    """Normalize one string or sequence into non-empty strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    """Return the first non-None source field."""

    for key in keys:
        if row.get(key) is not None:
            return row.get(key)
    return None


def _drop_none(payload: dict[str, Any]) -> None:
    """Remove top-level ``None`` values in place."""

    for key in [key for key, value in payload.items() if value is None]:
        payload.pop(key, None)


__all__ = [
    "ReasoningSourceBuildConfig",
    "build_reasoning_source_manifest",
    "build_reasoning_source_v2",
]
