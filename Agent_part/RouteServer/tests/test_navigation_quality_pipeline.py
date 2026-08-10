"""Focused contract, batch, view, and HTML tests for navigation quality V1."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_AFFORDANCE7_V1,
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    DirectionContextV4,
    InputModalitiesV4,
    NavObservationV4,
    ReasoningSourceV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from mcp_server.data.quality.review import (
    NavigationQualityReviewConfigV1,
    write_navigation_trainability_review_html_v1,
    write_traversability_geometry_review_html_v1,
)
from mcp_server.data.quality.retry import build_geometry_retry_source_manifest_v1
from mcp_server.data.quality.reports import summarize_navigation_quality_run_v1
from mcp_server.data.quality.smoke import (
    NavigationQualitySmokeConfigV1,
    _source_strata_hints_v1,
    build_navigation_quality_smoke_manifest_v1,
)
from mcp_server.data.quality.trainability import (
    NavigationTrainabilityPolicyV1,
    derive_navigation_trainability_v1,
    summarize_reasoning_source_motion_v1,
    summarize_stage_ii_sidecar_motion_v1,
)
from mcp_server.data.quality.views import write_navigation_trainability_views_v1
from mcp_server.teacher.navigation_quality_batch import (
    NavigationQualityBatchConfigV1,
    _motion_summary_for_source_v1,
    filter_navigation_trainability_source_v1,
    filter_navigation_trainability_sources_v1,
    label_traversability_geometry_sources_v1,
)
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration
from mcp_server.teacher.traversability_geometry_teacher import (
    parse_traversability_geometry_answer_v1,
    traversability_geometry_teacher_profile_v1,
)


class _FakeQualityBackend:
    """Return profile-specific strict JSON without opening an LVLM provider."""

    async def generate(self, source, *, question, system_prompt, image_detail):
        """Return valid evidence-grounded geometry or suitability JSON."""

        del system_prompt, image_detail
        if "traversability quadrilaterals" in question:
            assert source.image_roles == ["fpv.current"]
            payload = {
                "regions": [
                    {
                        "region_id": "sidewalk_center",
                        "role": "direct_traversable",
                        "surface_type": "sidewalk",
                        "quadrilateral_uv": {
                            "near_left": {"u": 0.24, "v": 0.94},
                            "near_right": {"u": 0.78, "v": 0.94},
                            "far_right": {"u": 0.61, "v": 0.46},
                            "far_left": {"u": 0.41, "v": 0.46},
                        },
                        "limiting_factors": [],
                        "evidence_refs": ["fpv.current"],
                    }
                ],
                "selected_corridor_region_id": "sidewalk_center",
                "confidence": "high",
                "validator_status": "accepted",
            }
        elif "Classify the current scene" in question:
            assert source.image_roles == ["fpv.current"]
            payload = {
                "scene_class": "outdoor_pedestrian_supported",
                "confidence": "high",
                "evidence_refs": ["fpv.current"],
                "reason_codes": ["visible_sidewalk"],
            }
        else:
            assert source.image_roles[-1] == "fpv.current"
            payload = {
                "consistency": "consistent",
                "crowd_density_near_path": "sparse",
                "corridor_occupancy": "clear",
                "visible_stop_cue": "none",
                "confidence": "high",
                "evidence_refs": ["fpv.current"],
                "reason_codes": ["motion_matches_clear_corridor"],
            }
        return TeacherGeneration(
            answer=json.dumps(payload),
            provider="local",
            model="fake-quality-vlm",
            telemetry={"roundtrip_ms": 1},
        )


def _source(tmp_path: Path, *, parent_id: str = "sidewalk:quality-case") -> ReasoningSourceV4:
    """Create one five-FPV source with a valid low-speed trajectory sidecar."""

    action = resolve_stage_a_direction_profile(FPV_DEGREE10_TERMINAL_CENTROID_V2)
    probe = resolve_affordance_probe_profile(FPV_DEGREE10_AFFORDANCE7_V1)
    images: list[str] = []
    roles: list[str] = []
    for index in range(4):
        path = tmp_path / f"history_{index}.jpg"
        Image.new("RGB", (96, 64), color=(20 + index * 15, 70, 110)).save(path, format="JPEG")
        images.append(str(path))
        roles.append(f"fpv.history_{index}")
    current = tmp_path / "current.jpg"
    Image.new("RGB", (96, 64), color=(72, 128, 96)).save(current, format="JPEG")
    images.append(str(current))
    roles.append("fpv.current")
    observation = NavObservationV4(
        source_group_id=parent_id,
        dataset_source="sidewalk",
        source_record_id=parent_id.split(":", maxsplit=1)[-1],
        split_group_id="sidewalk:quality-clip",
        images=images,
        image_roles=roles,
        instruction="Proceed along the accessible sidewalk ahead.",
        input_modalities=InputModalitiesV4(history_fpv=True, instruction=True),
        evidence_catalog=[{"ref": role, "modality": "fpv"} for role in roles],
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
        "trajectory_ego_t": [[0.0, 0.1 * index, 0.0] for index in range(5)],
        "trajectory_policy": {"dt_s": 0.2, "coordinate_frame": "ego_flu"},
        "action_mask": [True] * 5,
        "warnings": [],
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return ReasoningSourceV4.model_validate(payload)


def test_geometry_parser_rejects_a_generic_full_image_corridor(tmp_path: Path) -> None:
    """The polygon validator rejects near-full-image geometry before review."""

    source = _source(tmp_path)
    profile = traversability_geometry_teacher_profile_v1()
    answer = json.dumps(
        {
            "regions": [
                {
                    "region_id": "too_large",
                    "role": "direct_traversable",
                    "surface_type": "sidewalk",
                    "quadrilateral_uv": {
                        "near_left": {"u": 0.02, "v": 0.99},
                        "near_right": {"u": 0.98, "v": 0.99},
                        "far_right": {"u": 0.98, "v": 0.05},
                        "far_left": {"u": 0.02, "v": 0.05},
                    },
                    "limiting_factors": [],
                    "evidence_refs": ["fpv.current"],
                }
            ],
            "selected_corridor_region_id": "too_large",
            "confidence": "high",
            "validator_status": "accepted",
        }
    )
    annotation, errors = parse_traversability_geometry_answer_v1(
        answer,
        source=source,
        teacher_model="test",
        profile=profile,
    )
    assert annotation is None
    assert any("quadrilateral_area_too_large" in error for error in errors)


def test_geometry_parser_rejects_repeated_quad_vertex(tmp_path: Path) -> None:
    """A repeated vertex cannot silently become a traversability polygon."""

    source = _source(tmp_path)
    profile = traversability_geometry_teacher_profile_v1()
    answer = json.dumps(
        {
            "regions": [
                {
                    "region_id": "degenerate_vehicle",
                    "role": "blocked",
                    "surface_type": "vehicle",
                    "quadrilateral_uv": {
                        "near_left": {"u": 0.10, "v": 0.72},
                        "near_right": {"u": 0.34, "v": 0.88},
                        "far_right": {"u": 0.38, "v": 0.46},
                        "far_left": {"u": 0.10, "v": 0.72},
                    },
                    "limiting_factors": ["solid_obstacle"],
                    "evidence_refs": ["fpv.current"],
                }
            ],
            "selected_corridor_region_id": None,
            "confidence": "low",
            "validator_status": "pending_human",
        }
    )
    annotation, errors = parse_traversability_geometry_answer_v1(
        answer,
        source=source,
        teacher_model="test",
        profile=profile,
    )
    assert annotation is None
    assert errors


def test_geometry_parser_records_lossless_road_alias_normalization(tmp_path: Path) -> None:
    """The controlled road alias is auditable without weakening geometry QC."""

    source = _source(tmp_path)
    profile = traversability_geometry_teacher_profile_v1()
    answer = json.dumps(
        {
            "regions": [
                {
                    "region_id": "road_constraint",
                    "role": "blocked",
                    "surface_type": "road",
                    "quadrilateral_uv": {
                        "near_left": {"u": 0.15, "v": 0.95},
                        "near_right": {"u": 0.85, "v": 0.95},
                        "far_right": {"u": 0.70, "v": 0.55},
                        "far_left": {"u": 0.30, "v": 0.55},
                    },
                    "limiting_factors": ["vehicle_lane"],
                    "evidence_refs": ["fpv.current"],
                },
                {
                    "region_id": "sidewalk_center",
                    "role": "direct_traversable",
                    "surface_type": "sidewalk",
                    "quadrilateral_uv": {
                        "near_left": {"u": 0.25, "v": 0.92},
                        "near_right": {"u": 0.60, "v": 0.92},
                        "far_right": {"u": 0.54, "v": 0.52},
                        "far_left": {"u": 0.35, "v": 0.52},
                    },
                    "limiting_factors": [],
                    "evidence_refs": ["fpv.current"],
                },
            ],
            "selected_corridor_region_id": "sidewalk_center",
            "confidence": "high",
            "validator_status": "accepted",
        }
    )
    annotation, errors = parse_traversability_geometry_answer_v1(
        answer,
        source=source,
        teacher_model="test",
        profile=profile,
    )
    assert errors == []
    assert annotation is not None
    assert annotation.regions[0].surface_type == "road_edge"
    assert "surface_type_alias_normalized:0:road->road_edge" in annotation.warnings


def test_quality_batch_views_and_reviews_are_sidecar_only(tmp_path: Path) -> None:
    """Run synthetic annotation without mutating the V4 source or media."""

    source = _source(tmp_path)
    source_path = tmp_path / "source.jsonl"
    geometry_path = tmp_path / "geometry.jsonl"
    geometry_decisions = tmp_path / "geometry.decisions.jsonl"
    filter_path = tmp_path / "filter.jsonl"
    filter_decisions = tmp_path / "filter.decisions.jsonl"
    write_jsonl_atomic(source_path, [source.model_dump(mode="json")])
    config = NavigationQualityBatchConfigV1(concurrency=1, max_retries=0, progress_interval=0)
    geometry_report = asyncio.run(
        label_traversability_geometry_sources_v1(
            source_jsonl=source_path,
            geometry_jsonl=geometry_path,
            decisions_jsonl=geometry_decisions,
            config=config,
            backend_factory=lambda _index, _key: _FakeQualityBackend(),
        )
    )
    assert geometry_report["counts"]["accepted"] == 1
    # 缓存命中后不重复调用或追加同一几何 sidecar。
    geometry_resumed = asyncio.run(
        label_traversability_geometry_sources_v1(
            source_jsonl=source_path,
            geometry_jsonl=geometry_path,
            decisions_jsonl=geometry_decisions,
            config=config,
            backend_factory=lambda _index, _key: _FakeQualityBackend(),
        )
    )
    assert geometry_resumed["counts"]["pending_rows"] == 0
    filter_report = asyncio.run(
        filter_navigation_trainability_sources_v1(
            source_jsonl=source_path,
            filter_jsonl=filter_path,
            decisions_jsonl=filter_decisions,
            config=config,
            backend_factory=lambda _index, _key: _FakeQualityBackend(),
        )
    )
    assert filter_report["counts"]["status:accepted"] == 1
    views = write_navigation_trainability_views_v1(
        source_jsonl=source_path,
        filter_jsonl=filter_path,
        output_root=tmp_path / "views",
    )
    assert views["status_counts"]["accepted"] == 1
    assert views["eligibility_counts"]["stage_ii_eligible"] == 1
    accepted_rows = list(iter_jsonl(views["view_paths"]["accepted"]))
    assert accepted_rows[0]["source_fingerprint"] == source.source_fingerprint
    assert "navigation_trainability" not in accepted_rows[0]
    geometry_html = tmp_path / "geometry.html"
    suitability_html = tmp_path / "suitability.html"
    geometry_review = write_traversability_geometry_review_html_v1(
        source_jsonl=source_path,
        geometry_jsonl=geometry_path,
        geometry_decisions_jsonl=geometry_decisions,
        output_html=geometry_html,
        config=NavigationQualityReviewConfigV1(max_samples=10),
    )
    suitability_review = write_navigation_trainability_review_html_v1(
        source_jsonl=source_path,
        filter_jsonl=filter_path,
        output_html=suitability_html,
        config=NavigationQualityReviewConfigV1(max_samples=10),
    )
    assert geometry_review["rendered_count"] == 1
    assert geometry_review["geometry_decision_count"] == 1
    assert suitability_review["rendered_count"] == 1
    assert "<svg class=\"overlay\"" in geometry_html.read_text(encoding="utf-8")
    assert "teacher-only" in suitability_html.read_text(encoding="utf-8")


def test_geometry_review_includes_parser_failures_without_polygon_overlay(tmp_path: Path) -> None:
    """Failed LVLM geometry attempts remain inspectable instead of disappearing."""

    source = _source(tmp_path)
    source_path = tmp_path / "source.jsonl"
    geometry_path = tmp_path / "geometry.jsonl"
    decisions_path = tmp_path / "geometry.decisions.jsonl"
    output_html = tmp_path / "geometry-errors.html"
    write_jsonl_atomic(source_path, [source.model_dump(mode="json")])
    write_jsonl_atomic(geometry_path, [])
    write_jsonl_atomic(
        decisions_path,
        [
            {
                "kind": "traversability_geometry",
                "parent_sample_id": source.observation.source_group_id,
                "source_fingerprint": source.source_fingerprint,
                "status": "error",
                "profile_digest": "b" * 64,
                "attempts": [{"attempt": 1, "errors": ["response_json_invalid"]}],
                "raw_answer": "not valid json",
            }
        ],
    )
    report = write_traversability_geometry_review_html_v1(
        source_jsonl=source_path,
        geometry_jsonl=geometry_path,
        geometry_decisions_jsonl=decisions_path,
        output_html=output_html,
        config=NavigationQualityReviewConfigV1(max_samples=10),
    )
    rendered = output_html.read_text(encoding="utf-8")
    assert report["geometry_status_counts"] == {"error": 1}
    assert "response_json_invalid" in rendered
    assert "not valid json" in rendered


def test_review_offset_selects_a_deterministic_page(tmp_path: Path) -> None:
    """Pagination changes browser payload size without changing review order."""

    first = _source(tmp_path, parent_id="sidewalk:page-first")
    second = _source(tmp_path, parent_id="sidewalk:page-second")
    source_path = tmp_path / "source.jsonl"
    filter_path = tmp_path / "filter.jsonl"
    decisions_path = tmp_path / "filter.decisions.jsonl"
    output_html = tmp_path / "suitability-page.html"
    write_jsonl_atomic(source_path, [first.model_dump(mode="json"), second.model_dump(mode="json")])
    asyncio.run(
        filter_navigation_trainability_sources_v1(
            source_jsonl=source_path,
            filter_jsonl=filter_path,
            decisions_jsonl=decisions_path,
            config=NavigationQualityBatchConfigV1(concurrency=1, max_retries=0, progress_interval=0),
            backend_factory=lambda _index, _key: _FakeQualityBackend(),
        )
    )
    report = write_navigation_trainability_review_html_v1(
        source_jsonl=source_path,
        filter_jsonl=filter_path,
        output_html=output_html,
        config=NavigationQualityReviewConfigV1(sample_offset=1, max_samples=1),
    )
    rendered = output_html.read_text(encoding="utf-8")
    assert report["rendered_count"] == 1
    assert report["sample_offset"] == 1
    assert "sidewalk:page-second" in rendered
    assert "sidewalk:page-first" not in rendered


def test_geometry_retry_manifest_selects_only_current_failed_parent(tmp_path: Path) -> None:
    """Retry construction preserves source rows and excludes accepted decisions."""

    failed = _source(tmp_path, parent_id="sidewalk:failed")
    accepted = _source(tmp_path, parent_id="sidewalk:accepted")
    source_path = tmp_path / "source.jsonl"
    decisions_path = tmp_path / "geometry.decisions.jsonl"
    output_path = tmp_path / "retry.jsonl"
    report_path = tmp_path / "retry.report.json"
    write_jsonl_atomic(source_path, [failed.model_dump(mode="json"), accepted.model_dump(mode="json")])
    write_jsonl_atomic(
        decisions_path,
        [
            {
                "kind": "traversability_geometry",
                "parent_sample_id": failed.observation.source_group_id,
                "source_fingerprint": failed.source_fingerprint,
                "status": "error",
                "profile_digest": "c" * 64,
                "attempts": [{"errors": ["response_json_invalid"]}],
            },
            {
                "kind": "traversability_geometry",
                "parent_sample_id": accepted.observation.source_group_id,
                "source_fingerprint": accepted.source_fingerprint,
                "status": "accepted",
                "profile_digest": "c" * 64,
                "attempts": [],
            },
        ],
    )
    report = build_geometry_retry_source_manifest_v1(
        source_jsonl=source_path,
        geometry_decisions_jsonl=decisions_path,
        output_jsonl=output_path,
        report_json=report_path,
    )
    rows = list(iter_jsonl(output_path))
    assert report["selected_count"] == 1
    assert rows[0]["observation"]["source_group_id"] == failed.observation.source_group_id
    assert rows[0] == failed.model_dump(mode="json")


def test_quality_run_report_aggregates_read_only_ledgers(tmp_path: Path) -> None:
    """Run-level QC combines sidecar receipts without mutating a source row."""

    source = _source(tmp_path)
    dataset_root = tmp_path / "run" / "sidewalk_main"
    (dataset_root / "source").mkdir(parents=True)
    (dataset_root / "geometry").mkdir()
    (dataset_root / "suitability").mkdir()
    (dataset_root / "views").mkdir()
    write_jsonl_atomic(dataset_root / "source" / "smoke.jsonl", [source.model_dump(mode="json")])
    (dataset_root / "source" / "smoke_selection.json").write_text(
        json.dumps({"selected_count": 1, "selected_primary_counts": {"turn": 1}, "shortfalls": {}}),
        encoding="utf-8",
    )
    write_jsonl_atomic(
        dataset_root / "geometry" / "decisions.jsonl",
        [
            {
                "kind": "traversability_geometry",
                "status": "error",
                "attempts": [{"errors": ["geometry_schema: Value error, quadrilateral_area_too_small"]}],
            }
        ],
    )
    write_jsonl_atomic(
        dataset_root / "geometry" / "geometry.jsonl",
        [{"regions": [{"role": "direct_traversable", "surface_type": "sidewalk"}]}],
    )
    write_jsonl_atomic(
        dataset_root / "suitability" / "filter.jsonl",
        [
            {
                "status": "quarantine",
                "reason_codes": ["trajectory_missing_or_dt_unavailable"],
                "scene_assessment": {"scene_class": "outdoor_pedestrian_supported"},
                "motion_summary": {"source_kind": "unavailable", "qc_flags": ["trajectory_missing_or_dt_unavailable"]},
                "eligibility": {"stage_i_decision": True, "stage_ii_cfm": False, "vqa": True},
            }
        ],
    )
    (dataset_root / "views" / "report.json").write_text(
        json.dumps(
            {
                "status_counts": {"quarantine": 1},
                "eligibility_counts": {"stage_i_eligible": 1, "vqa_eligible": 1},
                "direction_counts_by_status": {"quarantine": {"straight": 1}},
                "stop_counts_by_status": {"quarantine": 0},
            }
        ),
        encoding="utf-8",
    )
    report = summarize_navigation_quality_run_v1(
        run_root=tmp_path / "run",
        output_json=tmp_path / "run" / "quality_report.json",
    )
    assert report["totals"]["source_selected_count"] == 1
    assert report["totals"]["geometry_error_code_counts"]["quadrilateral_area_too_small"] == 1
    assert report["totals"]["suitability_status_counts"]["quarantine"] == 1
    assert source.model_dump(mode="json") == list(iter_jsonl(dataset_root / "source" / "smoke.jsonl"))[0]


def test_quality_run_report_accepts_full_source_layout(tmp_path: Path) -> None:
    """Full runs use a neutral source name instead of a misleading smoke name."""

    source = _source(tmp_path)
    dataset_root = tmp_path / "run" / "foxglove"
    (dataset_root / "source").mkdir(parents=True)
    (dataset_root / "geometry").mkdir()
    (dataset_root / "suitability").mkdir()
    (dataset_root / "views").mkdir()
    write_jsonl_atomic(dataset_root / "source" / "source.jsonl", [source.model_dump(mode="json")])
    (dataset_root / "source" / "source_selection.json").write_text(
        json.dumps({"selected_count": 1, "selection_kind": "full_source_manifest"}),
        encoding="utf-8",
    )
    report = summarize_navigation_quality_run_v1(
        run_root=tmp_path / "run",
        output_json=tmp_path / "run" / "quality_report.json",
    )
    assert report["dataset_count"] == 1
    assert report["totals"]["source_selected_count"] == 1


def test_hard_kinematic_failure_is_rejected_before_training_views(tmp_path: Path) -> None:
    """Deterministic extreme-speed QC overrides otherwise positive LVLM facts."""

    source = _source(tmp_path)
    raw = source.model_dump(mode="json")
    raw["trajectory_ego_t"] = [[0.0, 0.0, 0.0], [0.0, 8.5, 0.0]]
    raw["trajectory_policy"] = {"dt_s": 0.2}
    raw["action_mask"] = [True, True]
    raw["source_fingerprint"] = reasoning_source_fingerprint_v4(
        {key: value for key, value in raw.items() if key != "source_fingerprint"}
    )
    corrupted = ReasoningSourceV4.model_validate(raw)
    summary = summarize_reasoning_source_motion_v1(corrupted)
    assert "kinematic_speed_extreme" in summary.qc_flags
    result, _decision = asyncio.run(
        filter_navigation_trainability_source_v1(
            corrupted,
            backend=_FakeQualityBackend(),
            config=NavigationQualityBatchConfigV1(max_retries=0),
        )
    )
    assert result.status == "rejected"
    assert not result.eligibility.stage_i_decision
    assert not result.eligibility.stage_ii_cfm


def test_external_stage_ii_motion_sidecar_enables_temporal_filtering(tmp_path: Path) -> None:
    """An immutable V2 sidecar supplies motion without rewriting the source."""

    source = _source(tmp_path)
    sidecar = SimpleNamespace(
        dataset_source="sidewalk",
        source_group_id=source.observation.source_group_id,
        # Stage-II V2 writers may include this dataset prefix while older V4
        # source rows store the same split unit without it.
        split_group_id="quality-clip",
        sample_id=source.observation.source_group_id,
        sidecar_digest="a" * 64,
        trajectory_target=SimpleNamespace(
            trajectory_ego_t=[[0.1 * index, 0.0, 0.0] for index in range(1, 33)],
            policy=SimpleNamespace(target_dt_sec=0.125),
        ),
    )
    summary = summarize_stage_ii_sidecar_motion_v1(sidecar)
    assert summary.available
    assert summary.source_kind == "external_sidecar"
    assert summary.source_artifact_digest == "a" * 64
    joined_summary = _motion_summary_for_source_v1(
        source,
        sidecar=sidecar,
        policy=NavigationTrainabilityPolicyV1(),
    )
    assert joined_summary.source_artifact_digest == "a" * 64
    result, _decision = asyncio.run(
        filter_navigation_trainability_source_v1(
            source,
            backend=_FakeQualityBackend(),
            config=NavigationQualityBatchConfigV1(max_retries=0),
            motion_summary=summary,
        )
    )
    assert result.status == "accepted"
    assert result.eligibility.stage_ii_cfm


def test_smoke_sampler_reports_stratum_shortfalls_without_copying_rows(tmp_path: Path) -> None:
    """A sparse source pool remains auditable instead of duplicating examples."""

    source = _source(tmp_path)
    source_path = tmp_path / "source.jsonl"
    smoke_path = tmp_path / "smoke.jsonl"
    report_path = tmp_path / "smoke.report.json"
    write_jsonl_atomic(source_path, [source.model_dump(mode="json")])
    report = build_navigation_quality_smoke_manifest_v1(
        source_jsonl=source_path,
        output_jsonl=smoke_path,
        report_json=report_path,
        dataset_source="sidewalk",
        config=NavigationQualitySmokeConfigV1(target_per_stratum=2, max_rows=10),
    )
    assert report["selected_count"] == 1
    assert report["selected_all_hint_counts"]["ordinary_sidewalk"] == 1
    assert report["shortfalls"]["stop"] == 2
    assert len(list(iter_jsonl(smoke_path))) == 1


def test_smoke_hints_remain_compatible_with_pre_geometry_v46_source(tmp_path: Path) -> None:
    """A historical V4.6 source without the optional field remains selectable."""

    source = _source(tmp_path)
    legacy_source = SimpleNamespace(
        observation=source.observation,
        direction_supervision=source.direction_supervision,
        trajectory_ego_t=source.trajectory_ego_t,
        trajectory_policy=source.trajectory_policy,
        warnings=source.warnings,
    )
    hints = _source_strata_hints_v1(legacy_source, high_speed_mps=3.0)
    assert "ordinary_sidewalk" in hints


def test_stale_filter_revision_never_enters_current_source_view(tmp_path: Path) -> None:
    """A repaired source cannot reuse a prior parent-level filter decision."""

    current = _source(tmp_path)
    old_raw = current.model_dump(mode="json")
    old_raw["observation"]["instruction"] = "Legacy instruction before source repair."
    old_raw["source_fingerprint"] = reasoning_source_fingerprint_v4(
        {key: value for key, value in old_raw.items() if key != "source_fingerprint"}
    )
    old = ReasoningSourceV4.model_validate(old_raw)
    old_filter, _decision = asyncio.run(
        filter_navigation_trainability_source_v1(
            old,
            backend=_FakeQualityBackend(),
            config=NavigationQualityBatchConfigV1(max_retries=0),
        )
    )
    source_path = tmp_path / "current-source.jsonl"
    filter_path = tmp_path / "old-filter.jsonl"
    write_jsonl_atomic(source_path, [current.model_dump(mode="json")])
    write_jsonl_atomic(filter_path, [old_filter.model_dump(mode="json")])
    report = write_navigation_trainability_views_v1(
        source_jsonl=source_path,
        filter_jsonl=filter_path,
        output_root=tmp_path / "stale-views",
    )
    assert report["stale_filter_rows"] == 1
    assert report["status_counts"]["pending_retry"] == 1
    assert report["eligibility_counts"]["stage_i_eligible"] == 0
