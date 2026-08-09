"""Build leakage-safe language and planner-guided benchmark packages.

The builder consumes canonical Reasoning V4 observations and Stage-II V2
future sidecars.  It creates benchmark inputs only from accepted language
annotations and independently produced planner routes; future odometry stays
in the hidden reference ledger.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkMessageV1,
    BenchmarkProfileV1,
    BenchmarkReferenceV1,
    BenchmarkTimePolicyV1,
)
from mcp_server.contracts.language_waypoint_benchmark import (
    LANGUAGE_GOAL_ANNOTATION_SCHEMA_VERSION,
    BenchmarkArmV1,
    LanguageGoalAnnotationV1,
    LanguageGoalCandidateV1,
    LanguageWaypointBenchmarkBuildConfigV1,
    PlannerRouteRecordWgs84V1,
    WaypointGuidanceV1,
    build_waypoint_guidance_from_record_v1,
    normalize_waypoint_guidance_for_model_v1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.contracts.stage_ii_cfm import StageIITrajectorySidecarV2
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkDatasetPluginV1
from mcp_server.eval.openloop.benchmark_plugin_package import (
    artifact_sha256_v1,
    build_benchmark_package_manifest_v1,
    write_benchmark_rows_v1,
)


LANGUAGE_WAYPOINT_DATASET_PLUGIN_ID_V1 = "urbannav_language_waypoint_v1"
LANGUAGE_WAYPOINT_PREDICTION_GEOMETRY_ID_V1 = "urbannav_current_ego_flu_xy_v1"
LANGUAGE_WAYPOINT_COORDINATE_FRAME_V1 = "current_ego_x_forward_y_left_v1"


class LanguageWaypointDatasetPluginV1(BenchmarkDatasetPluginV1):
    """Materialize paired language/waypoint arms without modifying source data.

    The plugin accepts exactly the typed arguments of
    :func:`build_language_waypoint_package_v1`.
    """

    plugin_id = LANGUAGE_WAYPOINT_DATASET_PLUGIN_ID_V1

    def build_package(self, **kwargs: Any) -> dict[str, Any]:
        """Build one frozen arm.

        Args:
            **kwargs: Arguments accepted by
                :func:`build_language_waypoint_package_v1`.

        Returns:
            Immutable package report.
        """

        return build_language_waypoint_package_v1(**kwargs)


def language_waypoint_default_profile_v1(
    *,
    arm: BenchmarkArmV1,
    include_bev: bool = False,
    horizons_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> BenchmarkProfileV1:
    """Return a generic profile for one frozen public-condition arm.

    Args:
        arm: Selected public language/route condition combination.
        include_bev: Whether every input must include ``bev.current``.
        horizons_s: Common physical evaluation horizons.

    Returns:
        Profile bound to the language-waypoint dataset plugin.
    """

    modality = _modality_profile_id_v1(arm=arm, include_bev=include_bev)
    return BenchmarkProfileV1(
        profile_id=f"urbannav_{modality}_openloop_v1",
        dataset_plugin_id=LANGUAGE_WAYPOINT_DATASET_PLUGIN_ID_V1,
        model_input_profile_id=modality,
        prediction_geometry_id=LANGUAGE_WAYPOINT_PREDICTION_GEOMETRY_ID_V1,
        metric_suite_ids=("trajectory_geometry_core_v1", "urbannav_openloop_metrics_v1"),
        time_policy=BenchmarkTimePolicyV1(
            policy_id="urbannav_stage_ii_v2_fixed_time_v1",
            horizons_s=horizons_s,
            min_angle_reference_distance_m=1e-3,
        ),
    )


def build_language_goal_candidates_v1(
    *,
    source_jsonl: str | Path,
    trajectory_sidecar_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    candidate_time_s: float = 4.0,
) -> dict[str, Any]:
    """Create teacher-only language-goal candidates from trusted future targets.

    This performs no language generation.  An LVLM or human annotation stage
    may consume these candidates later and must emit ``LanguageGoalAnnotationV1``.

    Args:
        source_jsonl: Canonical Reasoning V4 source manifest.
        trajectory_sidecar_jsonl: Accepted trusted Stage-II V2 sidecar ledger.
        output_jsonl: Teacher-only candidate destination.
        report_json: Deterministic report destination.
        candidate_time_s: Requested point time within the sidecar horizon.

    Returns:
        Candidate counts, reject counts, and immutable input fingerprints.

    Raises:
        ValueError: If candidate time is invalid or input identities drift.
    """

    if candidate_time_s <= 0.0:
        raise ValueError("candidate_time_s must be positive")
    sources = _load_reasoning_sources_v1(source_jsonl)
    sidecars = _load_unique_v1(trajectory_sidecar_jsonl, StageIITrajectorySidecarV2, "sample_id")
    candidates: list[LanguageGoalCandidateV1] = []
    rejects: Counter[str] = Counter()
    for parent_id, source in sorted(sources.items()):
        # 未来轨迹仅用于 teacher 选择候选目标，绝不进入模型可见 package。
        sidecar = sidecars.get(parent_id)
        if sidecar is None:
            rejects["trajectory_sidecar_missing"] += 1
            continue
        target = sidecar.trajectory_target
        dt = float(target.policy.target_dt_sec)
        target_index = max(0, min(len(target.trajectory_ego_t) - 1, round(candidate_time_s / dt) - 1))
        point = target.trajectory_ego_t[target_index]
        actual_time_s = dt * (target_index + 1)
        candidates.append(
            LanguageGoalCandidateV1(
                candidate_id=_stable_id_v1("language_goal_candidate", parent_id, str(target_index)),
                parent_sample_id=parent_id,
                source_group_id=source.observation.source_group_id,
                split_group_id=source.observation.split_group_id,
                dataset_source=source.observation.dataset_source,
                candidate_time_s=actual_time_s,
                candidate_xy_flu_m={"x_m": float(point[0]), "y_m": float(point[1])},
                candidate_yaw_rel_rad=float(point[2]),
                source_trajectory_sidecar_digest=sidecar.sidecar_digest,
            )
        )
    write_jsonl_atomic(output_jsonl, (row.model_dump(mode="json") for row in candidates))
    report = {
        "schema_version": "urbannav_language_goal_candidate_report_v1",
        "source_count": len(sources),
        "sidecar_count": len(sidecars),
        "candidate_count": len(candidates),
        "candidate_time_requested_s": candidate_time_s,
        "reject_counts": dict(sorted(rejects.items())),
        "source_jsonl_sha256": artifact_sha256_v1(source_jsonl),
        "trajectory_sidecar_jsonl_sha256": artifact_sha256_v1(trajectory_sidecar_jsonl),
    }
    _write_json_v1(report_json, report)
    return report


def build_waypoint_guidance_manifest_v1(
    *,
    planner_route_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    route_step_m: float = 2.0,
    route_horizon_m: float = 30.0,
    route_max_points: int = 8,
    local_goal_distance_m: float = 10.0,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Normalize typed independent planner records into reusable guidance rows.

    The input is intentionally a strict ``PlannerRouteRecordWgs84V1`` ledger.
    Each dataset needs a small explicit native adapter before this point rather
    than an unsafe collection of field-name heuristics.

    Args:
        planner_route_jsonl: Typed ``PlannerRouteRecordWgs84V1`` input ledger.
        output_jsonl: Normalized guidance ledger destination.
        report_json: Deterministic report destination.
        route_step_m: Public spatial resampling step.
        route_horizon_m: Public route horizon.
        route_max_points: Maximum exported public route points.
        local_goal_distance_m: Local goal distance along the route.
        skip_errors: Emit a report rather than fail on malformed source rows.

    Returns:
        Guidance/reject counts and immutable source fingerprint.

    Raises:
        ValueError: If ``skip_errors`` is false and a source record is invalid.
    """

    guidance_rows: list[WaypointGuidanceV1] = []
    reject_counts: Counter[str] = Counter()
    parent_ids: set[str] = set()
    for index, payload in enumerate(iter_jsonl(planner_route_jsonl), start=1):
        try:
            record = PlannerRouteRecordWgs84V1.model_validate(payload)
            if record.parent_sample_id in parent_ids:
                raise ValueError("duplicate_parent_sample_id")
            guidance = build_waypoint_guidance_from_record_v1(
                record,
                route_step_m=route_step_m,
                route_horizon_m=route_horizon_m,
                route_max_points=route_max_points,
                local_goal_distance_m=local_goal_distance_m,
            )
        except (TypeError, ValueError) as exc:
            if not skip_errors:
                raise ValueError(f"planner route row {index}: {exc}") from exc
            reject_counts[str(exc).split(":", 1)[0]] += 1
            continue
        parent_ids.add(record.parent_sample_id)
        guidance_rows.append(guidance)
    write_jsonl_atomic(output_jsonl, (row.model_dump(mode="json") for row in guidance_rows))
    report = {
        "schema_version": "urbannav_waypoint_guidance_report_v1",
        "planner_route_count": len(parent_ids) + sum(reject_counts.values()),
        "guidance_count": len(guidance_rows),
        "reject_counts": dict(sorted(reject_counts.items())),
        "route_step_m": route_step_m,
        "route_horizon_m": route_horizon_m,
        "route_max_points": route_max_points,
        "local_goal_distance_m": local_goal_distance_m,
        "planner_route_jsonl_sha256": artifact_sha256_v1(planner_route_jsonl),
    }
    _write_json_v1(report_json, report)
    return report


def build_language_waypoint_package_v1(
    *,
    source_jsonl: str | Path,
    trajectory_sidecar_jsonl: str | Path,
    output_root: str | Path,
    split: str,
    package_id: str,
    profile: BenchmarkProfileV1,
    config: LanguageWaypointBenchmarkBuildConfigV1,
    language_annotation_jsonl: str | Path | None = None,
    waypoint_guidance_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize one frozen arm from source, accepted labels, and planner data.

    ``fpv_only`` and ``fpv_language`` can be built before planner routes exist.
    Waypoint arms fail closed per row when the independent guidance ledger does
    not contain a valid, non-stale route for the same physical parent.

    Args:
        source_jsonl: Canonical Reasoning V4 source manifest.
        trajectory_sidecar_jsonl: Accepted hidden V2 trajectory ledger.
        output_root: Destination for immutable benchmark artifacts.
        split: Frozen split selected from the sidecar ledger.
        package_id: Immutable human-readable package identity.
        profile: Generic benchmark profile for the requested arm.
        config: Parameterized public image/condition policy.
        language_annotation_jsonl: Optional accepted annotation ledger.
        waypoint_guidance_jsonl: Optional independent planner guidance ledger.

    Returns:
        Package paths, denominator counts, rejects, config, and fingerprints.

    Raises:
        ValueError: If the profile/config drift or no parent is eligible.
    """

    _validate_package_args_v1(split=split, profile=profile, config=config)
    sources = _load_reasoning_sources_v1(source_jsonl)
    sidecars = _load_unique_v1(trajectory_sidecar_jsonl, StageIITrajectorySidecarV2, "sample_id")
    annotations = _load_accepted_annotations_v1(language_annotation_jsonl)
    guidance = _load_unique_v1(waypoint_guidance_jsonl, WaypointGuidanceV1, "parent_sample_id") if waypoint_guidance_jsonl else {}
    inputs: list[BenchmarkInputV1] = []
    references: list[BenchmarkReferenceV1] = []
    rejects: list[dict[str, str]] = []
    for parent_id, source in sorted(sources.items()):
        sidecar = sidecars.get(parent_id)
        if sidecar is None or sidecar.split != split:
            continue
        annotation = annotations.get(parent_id)
        route = guidance.get(parent_id)
        # 同一 parent 的各 arm 共用 hidden reference，缺失条件时 fail closed。
        reason = _eligibility_reject_v1(config=config, annotation=annotation, route=route)
        if reason is not None:
            rejects.append({"parent_sample_id": parent_id, "reason": reason})
            continue
        try:
            images, roles = _select_benchmark_images_v1(source, config=config)
        except ValueError as exc:
            rejects.append({"parent_sample_id": parent_id, "reason": str(exc)})
            continue
        missing = [image for image in images if not Path(image).is_file()]
        if missing and not config.allow_missing_images:
            rejects.append({"parent_sample_id": parent_id, "reason": "prompt_image_missing"})
            continue
        sample_id = _stable_id_v1("language_waypoint", config.arm, profile.profile_id, parent_id)
        public_context = _public_context_v1(config=config, guidance=route)
        inputs.append(
            BenchmarkInputV1(
                sample_id=sample_id,
                parent_sample_id=parent_id,
                source_group_id=source.observation.source_group_id,
                split_group_id=source.observation.split_group_id,
                split=split,
                dataset_plugin_id=LANGUAGE_WAYPOINT_DATASET_PLUGIN_ID_V1,
                benchmark_profile_id=profile.profile_id,
                dataset_source=source.observation.dataset_source,
                messages=_messages_v1(config=config, annotation=annotation),
                images=images,
                image_roles=roles,
                public_context=public_context,
                modality_profile_id=profile.model_input_profile_id,
            )
        )
        target = sidecar.trajectory_target
        references.append(
            BenchmarkReferenceV1(
                sample_id=sample_id,
                parent_sample_id=parent_id,
                source_group_id=source.observation.source_group_id,
                split_group_id=source.observation.split_group_id,
                split=split,
                dataset_plugin_id=LANGUAGE_WAYPOINT_DATASET_PLUGIN_ID_V1,
                benchmark_profile_id=profile.profile_id,
                dataset_source=source.observation.dataset_source,
                trajectory=[[float(value) for value in point] for point in target.trajectory_ego_t],
                trajectory_timestamps_s=[float(target.policy.target_dt_sec) * (index + 1) for index in range(len(target.trajectory_ego_t))],
                trajectory_valid_mask=[bool(item) for item in target.action_mask],
                coordinate_frame=LANGUAGE_WAYPOINT_COORDINATE_FRAME_V1,
                label_provenance="deterministic_gt",
                subgroup_metadata={
                    "benchmark_arm": config.arm,
                    "language_annotation_id": annotation.annotation_id if annotation else None,
                    "waypoint_guidance_id": route.guidance_id if route else None,
                    "direction_label": source.direction_supervision.label_name if source.direction_supervision else None,
                    "trajectory_sidecar_digest": sidecar.sidecar_digest,
                },
            )
        )
    if not inputs:
        raise ValueError("no eligible rows for language-waypoint benchmark package")
    output = Path(output_root).expanduser().resolve()
    inputs_path = output / "inputs.jsonl"
    references_path = output / "references.jsonl"
    rejects_path = output / "rejects.jsonl"
    package_path = output / "package.json"
    write_benchmark_rows_v1(inputs_path, inputs)
    write_benchmark_rows_v1(references_path, references)
    write_jsonl_atomic(rejects_path, rejects)
    fingerprints = {
        "source_jsonl": artifact_sha256_v1(source_jsonl),
        "trajectory_sidecar_jsonl": artifact_sha256_v1(trajectory_sidecar_jsonl),
    }
    if language_annotation_jsonl:
        fingerprints["language_annotation_jsonl"] = artifact_sha256_v1(language_annotation_jsonl)
    if waypoint_guidance_jsonl:
        fingerprints["waypoint_guidance_jsonl"] = artifact_sha256_v1(waypoint_guidance_jsonl)
    manifest = build_benchmark_package_manifest_v1(
        package_id=package_id,
        profile=profile,
        split=split,
        input_jsonl=inputs_path,
        reference_jsonl=references_path,
        output_json=package_path,
        source_fingerprints=fingerprints,
    )
    report = {
        "schema_version": "urbannav_language_waypoint_package_report_v1",
        "dataset_plugin_id": LANGUAGE_WAYPOINT_DATASET_PLUGIN_ID_V1,
        "package_id": package_id,
        "package_path": str(package_path),
        "input_count": manifest.input_count,
        "reference_count": manifest.reference_count,
        "reject_count": len(rejects),
        "reject_counts": dict(Counter(item["reason"] for item in rejects)),
        "config": config.model_dump(mode="json"),
        "source_fingerprints": fingerprints,
    }
    _write_json_v1(output / "report.json", report)
    return report


def _validate_package_args_v1(*, split: str, profile: BenchmarkProfileV1, config: LanguageWaypointBenchmarkBuildConfigV1) -> None:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if profile.dataset_plugin_id != LANGUAGE_WAYPOINT_DATASET_PLUGIN_ID_V1:
        raise ValueError("profile dataset plugin differs from language-waypoint builder")
    expected = _modality_profile_id_v1(arm=config.arm, include_bev=config.include_bev)
    if profile.model_input_profile_id != expected:
        raise ValueError("profile input identity differs from build config")
    if profile.prediction_geometry_id != LANGUAGE_WAYPOINT_PREDICTION_GEOMETRY_ID_V1:
        raise ValueError("profile prediction geometry differs from builder")


def _load_reasoning_sources_v1(path: str | Path) -> dict[str, ReasoningSourceV4]:
    rows: dict[str, ReasoningSourceV4] = {}
    for payload in iter_jsonl(path):
        row = ReasoningSourceV4.model_validate(payload)
        parent_id = row.observation.source_group_id
        if parent_id in rows:
            raise ValueError(f"duplicate source_group_id: {parent_id}")
        rows[parent_id] = row
    return rows


def _load_unique_v1(path: str | Path | None, model_type: type[Any], key: str) -> dict[str, Any]:
    if path is None:
        return {}
    rows: dict[str, Any] = {}
    for payload in iter_jsonl(path):
        row = model_type.model_validate(payload)
        identifier = str(getattr(row, key))
        if identifier in rows:
            raise ValueError(f"duplicate {key}: {identifier}")
        rows[identifier] = row
    return rows


def _load_accepted_annotations_v1(path: str | Path | None) -> dict[str, LanguageGoalAnnotationV1]:
    if path is None:
        return {}
    rows: dict[str, LanguageGoalAnnotationV1] = {}
    for payload in iter_jsonl(path):
        row = LanguageGoalAnnotationV1.model_validate(payload)
        if row.validator_status != "accepted" or row.human_review == "rejected":
            continue
        existing = rows.get(row.parent_sample_id)
        if existing is not None:
            raise ValueError(f"multiple accepted language annotations: {row.parent_sample_id}")
        rows[row.parent_sample_id] = row
    return rows


def _eligibility_reject_v1(
    *,
    config: LanguageWaypointBenchmarkBuildConfigV1,
    annotation: LanguageGoalAnnotationV1 | None,
    route: WaypointGuidanceV1 | None,
) -> str | None:
    """Return the first fail-closed eligibility reason for one paired parent."""

    needs_language = config.arm in {"fpv_language", "fpv_language_waypoint"}
    needs_waypoint = config.arm in {"fpv_waypoint", "fpv_language_waypoint"}
    if config.require_language_annotation_for_controls and annotation is None:
        return "accepted_language_annotation_missing"
    if needs_language and annotation is None:
        return "accepted_language_annotation_missing"
    if needs_waypoint and route is None:
        return "independent_waypoint_guidance_missing"
    if route is not None and (
        abs(route.route_step_m - config.public_route_step_m) > 1e-9
        or abs(route.route_horizon_m - config.public_route_horizon_m) > 1e-9
        or route.route_max_points != config.public_route_max_points
        or abs(route.local_goal_distance_m - config.public_local_goal_distance_m) > 1e-9
    ):
        return "waypoint_sampling_policy_mismatch"
    return None


def _select_benchmark_images_v1(
    source: ReasoningSourceV4,
    *,
    config: LanguageWaypointBenchmarkBuildConfigV1,
) -> tuple[list[str], list[str]]:
    """Select ordered FPV context and an optional current BEV image."""

    fpv = [
        (image, role)
        for image, role in zip(source.observation.images, source.observation.image_roles)
        if role.startswith("fpv.")
    ]
    if len(fpv) != config.required_fpv_frame_count:
        raise ValueError("required_fpv_frame_count_mismatch")
    selected = list(fpv)
    if config.include_bev:
        bev = next(
            (
                (image, role)
                for image, role in zip(source.observation.images, source.observation.image_roles)
                if role == "bev.current"
            ),
            None,
        )
        if bev is None:
            raise ValueError("current_bev_missing")
        selected.append(bev)
    return [image for image, _ in selected], [role for _, role in selected]


def _messages_v1(
    *,
    config: LanguageWaypointBenchmarkBuildConfigV1,
    annotation: LanguageGoalAnnotationV1 | None,
) -> list[BenchmarkMessageV1]:
    """Render only the public condition selected by the paired arm."""

    condition = "Use the ordered FPV observations to predict a safe local motion plan."
    if config.arm in {"fpv_language", "fpv_language_waypoint"}:
        if annotation is None or annotation.instruction_text is None:
            raise ValueError("language arm requires accepted instruction")
        condition = f"Navigation instruction: {annotation.instruction_text.strip()}"
    if config.arm in {"fpv_waypoint", "fpv_language_waypoint"}:
        condition += " A normalized planner route is available in public_context.route_guidance."
    return [
        BenchmarkMessageV1(role="system", content=config.system_prompt),
        BenchmarkMessageV1(role="user", content=condition),
    ]


def _public_context_v1(
    *,
    config: LanguageWaypointBenchmarkBuildConfigV1,
    guidance: WaypointGuidanceV1 | None,
) -> dict[str, Any]:
    """Project only normalized planner input to the model-visible envelope."""

    if config.arm not in {"fpv_waypoint", "fpv_language_waypoint"}:
        return {"condition_arm": config.arm}
    if guidance is None:
        raise ValueError("waypoint arm requires guidance")
    return {
        "condition_arm": config.arm,
        "route_guidance": {
            "schema_version": guidance.schema_version,
            "feature_layout": [
                "forward_over_horizon",
                "left_over_horizon",
                "sin_heading_rel",
                "cos_heading_rel",
                "arc_over_horizon",
                "terminal",
            ],
            "distance_scale_m": config.public_route_horizon_m,
            "points": normalize_waypoint_guidance_for_model_v1(
                guidance,
                distance_scale_m=config.public_route_horizon_m,
            ),
            "local_goal": [
                guidance.local_goal_ego_flu_m.x_m / config.public_route_horizon_m,
                guidance.local_goal_ego_flu_m.y_m / config.public_route_horizon_m,
            ],
            "planner_state": {
                "reroute": guidance.reroute,
                "stale": guidance.stale,
                "valid": guidance.valid,
            },
        },
    }


def _modality_profile_id_v1(*, arm: BenchmarkArmV1, include_bev: bool) -> str:
    """Return a stable, parameter-derived public modality identity."""

    suffix = "bev" if include_bev else "no_bev"
    return f"urbannav_{arm}_{suffix}_5fpv_v1"


def _stable_id_v1(*parts: str) -> str:
    """Return a bounded stable identifier without truncating parent provenance."""

    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{parts[0]}:{digest}"


def _write_json_v1(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one small deterministic JSON report."""

    import json

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "LANGUAGE_WAYPOINT_COORDINATE_FRAME_V1",
    "LANGUAGE_WAYPOINT_DATASET_PLUGIN_ID_V1",
    "LANGUAGE_WAYPOINT_PREDICTION_GEOMETRY_ID_V1",
    "LanguageWaypointDatasetPluginV1",
    "build_language_goal_candidates_v1",
    "build_waypoint_guidance_manifest_v1",
    "build_language_waypoint_package_v1",
    "language_waypoint_default_profile_v1",
]
