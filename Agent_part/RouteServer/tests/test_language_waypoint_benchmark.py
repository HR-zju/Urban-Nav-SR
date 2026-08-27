"""Focused tests for language/planner benchmark preprocessing V1."""

from __future__ import annotations

import json
import math
from pathlib import Path

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_AFFORDANCE7_V1,
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.language_waypoint_benchmark import (
    LanguageGoalAnnotationV1,
    LanguageWaypointBenchmarkBuildConfigV1,
    PlannerRouteRecordWgs84V1,
    WaypointGuidanceProvenanceV1,
    Wgs84CoordinateV1,
    build_waypoint_guidance_from_record_v1,
    normalize_waypoint_guidance_for_model_v1,
    wgs84_to_current_ego_flu_v1,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    DirectionContextV4,
    InputModalitiesV4,
    NavObservationV4,
    ReasoningSourceV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.contracts.stage_ii_cfm import (
    StageIITrajectoryPolicyV2,
    StageIITrajectorySidecarV2,
    StageIITrajectoryTargetV2,
    stage_ii_digest_v1,
)
from mcp_server.data.benchmark.language_waypoint_dataset_plugin import (
    build_language_goal_candidates_v1,
    build_language_waypoint_package_v1,
    build_waypoint_guidance_manifest_v1,
    language_waypoint_default_profile_v1,
)
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.data.stage_ii_trajectory import build_stage_ii_trajectory_qc_v2
from mcp_server.eval.openloop.benchmark_plugin_package import load_benchmark_package_v1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source(tmp_path: Path) -> ReasoningSourceV4:
    action = resolve_stage_a_direction_profile(FPV_DEGREE10_TERMINAL_CENTROID_V2)
    probe = resolve_affordance_probe_profile(FPV_DEGREE10_AFFORDANCE7_V1)
    paths = []
    roles = []
    for index in range(4):
        path = tmp_path / f"history_{index}.jpg"
        path.write_bytes(b"not-decoded-in-schema-smoke")
        paths.append(str(path))
        roles.append(f"fpv.history_{index}")
    current = tmp_path / "current.jpg"
    current.write_bytes(b"not-decoded-in-schema-smoke")
    paths.append(str(current))
    roles.append("fpv.current")
    bev = tmp_path / "bev.png"
    bev.write_bytes(b"not-decoded-in-schema-smoke")
    paths.append(str(bev))
    roles.append("bev.current")
    observation = NavObservationV4(
        source_group_id="sidewalk:language-waypoint-case",
        dataset_source="sidewalk",
        source_record_id="language-waypoint-case",
        split_group_id="sidewalk:clip-1",
        images=paths,
        image_roles=roles,
        input_modalities=InputModalitiesV4(history_fpv=True, bev=True),
        evidence_catalog=[
            {"ref": role, "modality": "fpv" if role.startswith("fpv") else "bev"}
            for role in roles
        ],
        direction_context=DirectionContextV4(
            action_profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
            action_profile_version=action["profile_version"],
            affordance_probe_profile_id=FPV_DEGREE10_AFFORDANCE7_V1,
            affordance_probe_profile_version=probe["profile_version"],
        ),
    )
    payload = {
        "schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION,
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": None,
        "direction_geometry": None,
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": [],
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return ReasoningSourceV4.model_validate(payload)


def _sidecar(source: ReasoningSourceV4) -> StageIITrajectorySidecarV2:
    policy = StageIITrajectoryPolicyV2(
        source_adapter_id="sidewalk_raw_pose_v2",
        trajectory_source="odometry",
        raw_timestamp_unit="millisecond",
        yaw_offset_deg=0.0,
    )
    target_payload = {
        "schema_version": "urbannav_stage_ii_trajectory_target_v2_1",
        "trajectory_ego_t": [[0.125 * (index + 1), 0.01 * index, 0.0] for index in range(32)],
        "action_mask": [True] * 32,
        "valid_step_count": 32,
        "policy": policy.model_dump(mode="json"),
        "source_target_digest": "a" * 64,
    }
    target_payload["target_digest"] = stage_ii_digest_v1(target_payload)
    target = StageIITrajectoryTargetV2.model_validate(target_payload)
    qc = build_stage_ii_trajectory_qc_v2(
        sample_id=source.observation.source_group_id,
        dataset_source=source.observation.dataset_source,
        source_adapter_id=policy.source_adapter_id,
        trajectory_source=policy.trajectory_source,
        clip_id="clip-1",
        frame_idx=1,
        status="accepted",
        reason_codes=("trajectory_build_accepted",),
        raw_pose_count=33,
        source_valid_step_count=33,
        source_horizon_seconds=4.0,
        max_source_gap_sec=0.125,
        max_source_step_m=0.2,
        max_source_speed_mps=1.0,
        max_source_yaw_rate_rad_s=0.1,
        target_valid_step_count=32,
        source_evidence_digest=target.source_target_digest,
    )
    payload = {
        "schema_version": "urbannav_stage_ii_trajectory_sidecar_v2_1",
        "sample_id": source.observation.source_group_id,
        "source_group_id": source.observation.source_group_id,
        "split_group_id": source.observation.split_group_id,
        "dataset_source": source.observation.dataset_source,
        "split": "test",
        "source_manifest_name": "synthetic/source.jsonl",
        "source_manifest_sha256": "b" * 64,
        "source_line": 1,
        "trajectory_target": target.model_dump(mode="json"),
        "trajectory_qc": qc.model_dump(mode="json"),
    }
    payload["sidecar_digest"] = stage_ii_digest_v1(payload)
    return StageIITrajectorySidecarV2.model_validate(payload)


def _annotation(parent_id: str) -> LanguageGoalAnnotationV1:
    return LanguageGoalAnnotationV1(
        annotation_id="language-1",
        parent_sample_id=parent_id,
        instruction_text="Move toward the visible building on the right.",
        instruction_type="landmark_object_goal",
        target_ref="fpv.current",
        visibility_at_anchor=True,
        evidence_refs=("fpv.current",),
        teacher_model="reviewed-lvlm",
        prompt_version="language_goal_v1",
        confidence="high",
        validator_status="accepted",
        human_review="accepted",
        provenance="lvlm_annotation",
    )


def _planner_record(parent_id: str) -> PlannerRouteRecordWgs84V1:
    return PlannerRouteRecordWgs84V1(
        record_id="planner-1",
        parent_sample_id=parent_id,
        current_wgs84={"latitude_deg": 30.0, "longitude_deg": 120.0},
        current_heading_rad=0.0,
        goal_wgs84={"latitude_deg": 30.0004, "longitude_deg": 120.0},
        planner_polyline_wgs84=[
            {"latitude_deg": 30.0, "longitude_deg": 120.0},
            {"latitude_deg": 30.0002, "longitude_deg": 120.0},
            {"latitude_deg": 30.0004, "longitude_deg": 120.0},
        ],
        provenance=WaypointGuidanceProvenanceV1(
            planner_id="test_map_planner",
            route_version="v1",
            plan_timestamp_s=1.0,
            datum="WGS84",
            source_kind="map_planner",
            independent_of_future_target=True,
        ),
    )


def test_wgs84_planner_guidance_uses_public_fixed_normalization() -> None:
    origin = Wgs84CoordinateV1(latitude_deg=30.0, longitude_deg=120.0)
    north = Wgs84CoordinateV1(latitude_deg=30.0001, longitude_deg=120.0)
    point = wgs84_to_current_ego_flu_v1(origin=origin, heading_rad=0.0, point=north)
    assert point.x_m > 11.0
    assert abs(point.y_m) < 1e-6
    guidance = build_waypoint_guidance_from_record_v1(_planner_record("case-1"), route_horizon_m=30.0)
    features = normalize_waypoint_guidance_for_model_v1(guidance, distance_scale_m=30.0)
    assert len(features) == 8
    assert all(len(row) == 6 for row in features)
    assert features[-1][-1] == 0.0
    assert all(math.isfinite(value) for row in features for value in row)


def test_candidate_guidance_and_paired_packages_keep_future_target_hidden(tmp_path: Path) -> None:
    source = _source(tmp_path)
    sidecar = _sidecar(source)
    source_path = tmp_path / "source.jsonl"
    sidecar_path = tmp_path / "sidecar.jsonl"
    annotation_path = tmp_path / "annotation.jsonl"
    planner_path = tmp_path / "planner.jsonl"
    guidance_path = tmp_path / "guidance.jsonl"
    _write_jsonl(source_path, [source.model_dump(mode="json")])
    _write_jsonl(sidecar_path, [sidecar.model_dump(mode="json")])
    _write_jsonl(annotation_path, [_annotation(source.observation.source_group_id).model_dump(mode="json")])
    _write_jsonl(planner_path, [_planner_record(source.observation.source_group_id).model_dump(mode="json")])
    candidate_report = build_language_goal_candidates_v1(
        source_jsonl=source_path,
        trajectory_sidecar_jsonl=sidecar_path,
        output_jsonl=tmp_path / "candidates.jsonl",
        report_json=tmp_path / "candidate_report.json",
    )
    assert candidate_report["candidate_count"] == 1
    guidance_report = build_waypoint_guidance_manifest_v1(
        planner_route_jsonl=planner_path,
        output_jsonl=guidance_path,
        report_json=tmp_path / "guidance_report.json",
    )
    assert guidance_report["guidance_count"] == 1
    for arm in ("fpv_only", "fpv_language", "fpv_waypoint", "fpv_language_waypoint"):
        config = LanguageWaypointBenchmarkBuildConfigV1(arm=arm)
        profile = language_waypoint_default_profile_v1(arm=arm)
        root = tmp_path / arm
        report = build_language_waypoint_package_v1(
            source_jsonl=source_path,
            trajectory_sidecar_jsonl=sidecar_path,
            language_annotation_jsonl=annotation_path,
            waypoint_guidance_jsonl=guidance_path,
            output_root=root,
            split="test",
            package_id=f"test-{arm}",
            profile=profile,
            config=config,
        )
        assert report["input_count"] == 1
        _, inputs, references = load_benchmark_package_v1(root / "package.json")
        input_row = next(iter(inputs.values()))
        reference_row = next(iter(references.values()))
        assert len(input_row.images) == 5
        assert len(reference_row.trajectory) == 32
        rendered_public = json.dumps(input_row.model_dump(mode="json"), sort_keys=True)
        assert "trajectory_ego_t" not in rendered_public
        assert "route_guidance" in rendered_public if "waypoint" in arm else "route_guidance" not in rendered_public
        assert "Navigation instruction:" in rendered_public if "language" in arm else "Navigation instruction:" not in rendered_public
    assert len(list(iter_jsonl(tmp_path / "candidates.jsonl"))) == 1
