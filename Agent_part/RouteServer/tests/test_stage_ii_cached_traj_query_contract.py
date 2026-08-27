import hashlib
import json
import sqlite3
from collections import Counter

import pytest

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE,
    STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2,
    STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION,
    StageIICachedTrajQueryBindingV1,
    StageIICachedTrajQueryObservationV1,
    StageIICachedTrajQuerySourceViewV2,
    stage_ii_digest_v1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningTrainingViewV4
from mcp_server.data.stage_ii.views import (
    build_stage_ii_cached_traj_query_observation_v1,
    build_stage_ii_cached_traj_query_source_view_v2,
    build_stage_ii_cached_traj_query_training_view_v1,
    build_stage_ii_cached_traj_query_training_view_v2,
)
from mcp_server.data.stage_ii.cached_pairs import (
    _create_index,
    _insert_source_timing,
    _raw_timestamp,
    _resolve_source_role_timestamps,
    build_stage_ii_cached_traj_query_pairs_v1,
    build_stage_ii_cached_traj_query_pairs_v2,
    build_stage_ii_cached_traj_query_source_views_v2,
)
from mcp_server.data.stage_ii_views import project_stage_ii_cfm_view_v2
from mcp_server.data.stage_i_views import project_stage_i_reasoning_view_v2

from test_stage_ii_cfm_v2 import (
    _cache,
    _decision_text,
    _stage_i_full_v2,
    _trajectory_sidecar_v2,
    _trajectory_v2,
)
from test_stage_i_sft import _formal_reasoning_view


def test_cached_traj_query_binding_requires_causal_consistent_timestamps():
    binding = StageIICachedTrajQueryBindingV1(
        query_source_timestamp_sec=10.0,
        action_observation_timestamp_sec=10.25,
        query_age_sec=0.25,
        modality_profile_id="fpv_temporal_instruction_bev_v1",
    )
    assert binding.condition_mode == STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE


def test_raw_timestamp_preserves_nanosecond_integer_precision():
    value = 1780829124292947867

    assert _raw_timestamp(value, "test timestamp") == value


def test_duplicate_source_timing_merges_manifest_fingerprints():
    connection = sqlite3.connect(":memory:")
    try:
        _create_index(connection)
        counters = Counter()
        base = {
            "kind": "multiframe",
            "source_group_id": "sidewalk:clip:0001",
            "split_group_id": "sidewalk:clip",
            "dataset_source": "sidewalk",
        }
        _insert_source_timing(
            connection,
            {**base, "source_manifest_sha256s": ["a" * 64]},
            counters,
        )
        _insert_source_timing(
            connection,
            {**base, "source_manifest_sha256s": ["b" * 64]},
            counters,
        )
        payload = connection.execute(
            "SELECT payload FROM source_timing WHERE source_group_id = ?",
            (base["source_group_id"],),
        ).fetchone()[0]
        assert json.loads(payload)["source_manifest_sha256s"] == ["a" * 64, "b" * 64]
        assert counters["equivalent_duplicate_sources"] == 1
    finally:
        connection.close()


def test_source_timing_rejects_future_fpv_timestamp():
    connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError) as exc_info:
            _resolve_source_role_timestamps(
                connection,
                image_roles=["fpv.history_00", "fpv.current"],
                raw_timestamp_unit="millisecond",
                timing={
                    "kind": "multiframe",
                    "observation_raw": 10000.0,
                    "fpv_timestamps_raw": [9000.0, 11000.0],
                    "bev_timestamp_raw": None,
                },
                sample_id="sidewalk:clip:0001",
            )
        assert exc_info.value.code == "future_fpv_timestamp"
    finally:
        connection.close()


def test_cached_traj_query_binding_rejects_future_query():
    with pytest.raises(ValueError, match="cannot come from the future"):
        StageIICachedTrajQueryBindingV1(
            query_source_timestamp_sec=11.0,
            action_observation_timestamp_sec=10.25,
            query_age_sec=0.0,
            modality_profile_id="fpv_temporal_instruction_bev_v1",
        )


def test_cached_pair_keeps_query_observation_target_free():
    stage_i_view = _stage_i_full_v2()
    decision = _cache(stage_i_view, "decision", _decision_text(stage_i_view))
    action_view = project_stage_ii_cfm_view_v2(
        stage_i_view,
        trajectory_target=_trajectory_v2(),
        reasoning_context=_cache(
            stage_i_view,
            "reasoning_grounding",
            stage_i_view.assistant_text,
        ),
        decision_context=decision,
        review_status="not_required",
    )
    pair = build_stage_ii_cached_traj_query_training_view_v1(
        action_view,
        action_view,
        query_source_timestamp_sec=1.0,
        action_observation_timestamp_sec=1.2,
        query_source_frame_timestamps_sec=[
            1.0 - 0.2 * (len(action_view.images) - index - 1)
            for index in range(len(action_view.images))
        ],
        action_frame_timestamps_sec=[
            1.2 - 0.2 * (len(action_view.images) - index - 1)
            for index in range(len(action_view.images))
        ],
    )

    assert pair.cached_traj_query.query_age_sec == pytest.approx(0.2)
    assert "trajectory_target" not in pair.query_source.model_dump(mode="json")


def test_cached_pair_v2_bypasses_reasoning_and_decision_context():
    stage_i_view = _stage_i_full_v2()
    action_view = project_stage_ii_cfm_view_v2(
        stage_i_view,
        trajectory_target=_trajectory_v2(),
        reasoning_context=_cache(
            stage_i_view,
            "reasoning_grounding",
            stage_i_view.assistant_text,
        ),
        decision_context=_cache(
            stage_i_view,
            "decision",
            _decision_text(stage_i_view),
        ),
        review_status="not_required",
    )
    query_observation = build_stage_ii_cached_traj_query_observation_v1(
        action_view,
        observation_timestamp_sec=1.0,
        frame_timestamps_sec=[
            1.0 - 0.2 * (len(action_view.images) - index - 1)
            for index in range(len(action_view.images))
        ],
    )
    action_observation = build_stage_ii_cached_traj_query_observation_v1(
        action_view,
        observation_timestamp_sec=1.2,
        frame_timestamps_sec=[
            1.2 - 0.2 * (len(action_view.images) - index - 1)
            for index in range(len(action_view.images))
        ],
    )
    pair = build_stage_ii_cached_traj_query_training_view_v2(
        query_observation,
        action_observation,
        trajectory_target=_trajectory_v2(),
        condition_digest="a" * 64,
        provenance={"trajectory_source_group_id": action_view.source_group_id},
    )

    assert pair.condition_mode == STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2
    assert pair.temporal_binding.query_age_sec == pytest.approx(0.2)
    assert "action_view" not in pair.model_dump(mode="json")
    assert "reasoning_context" not in pair.model_dump(mode="json")
    assert "decision_context" not in pair.model_dump(mode="json")


def test_cached_pair_v2_accepts_waypoint_stage_i_profile():
    payload = _formal_reasoning_view().model_dump(mode="json")
    payload["input_modalities"]["waypoint"] = True
    payload["evidence_catalog"].append({
        "ref": "waypoint.active",
        "modality": "waypoint",
        "summary": "independent planner route",
    })
    payload["model_visible_context"] = {
        "waypoint": {
            "independent_of_future_target": True,
            "source": "independent_gps_route",
            "waypoint_input": {
                "available": True,
                "independent_of_future_target": True,
                "source": "independent_gps_route",
            },
        }
    }
    stage_i_view = project_stage_i_reasoning_view_v2(
        ReasoningTrainingViewV4.model_validate(payload),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_waypoint_v2",
        instruction_training_role="primary",
    )
    observation = build_stage_ii_cached_traj_query_observation_v1(
        stage_i_view,
        observation_timestamp_sec=1.0,
        frame_timestamps_sec=[1.0],
    )

    assert observation.modality_profile_id == "fpv_instruction_waypoint_v2"


def test_cached_source_v2_is_built_from_visible_stage_i_and_sidecar_only():
    stage_i_view = _stage_i_full_v2()
    source = build_stage_ii_cached_traj_query_source_view_v2(
        stage_i_view,
        trajectory_target=_trajectory_v2(),
        observation_timestamp_sec=1.2,
        frame_timestamps_sec=[
            1.2 - 0.2 * (len(stage_i_view.images) - index - 1)
            for index in range(len(stage_i_view.images))
        ],
        bev_timestamp_sec=None,
    )

    assert isinstance(source, StageIICachedTrajQuerySourceViewV2)
    assert source.observation.sample_id == stage_i_view.sample_id
    assert source.provenance["trajectory_source_group_id"] == source.source_group_id
    assert "assistant_text" not in source.model_dump(mode="json")
    assert "reasoning_context" not in source.model_dump(mode="json")
    assert "decision_context" not in source.model_dump(mode="json")


def test_cached_pairs_v2_materializer_uses_source_views_without_r_d(tmp_path):
    from test_stage_ii_cfm import _source_registry_payload

    stage_i_view = _stage_i_full_v2()
    source = build_stage_ii_cached_traj_query_source_view_v2(
        stage_i_view,
        trajectory_target=_trajectory_v2(),
        observation_timestamp_sec=1.2,
        frame_timestamps_sec=[1.2],
    )
    source_views_path = tmp_path / "source_views.jsonl"
    source_views_path.write_text(
        json.dumps(source.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    registry = _source_registry_payload()
    sidewalk = next(
        item for item in registry["datasets"] if item["dataset_source"] == "sidewalk"
    )
    sidewalk["sample_identity_fields"] = ["id"]
    registry["registry_digest"] = stage_ii_digest_v1(
        {key: value for key, value in registry.items() if key != "registry_digest"}
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    source_path = tmp_path / "sidewalk.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "id": "case-straight",
                "trajectory_policy": {
                    "schema_version": "sidewalk_time_conditioned_policy_v1",
                    "time_unit": "seconds",
                    "trajectory_source": "calibrated_pose_projection",
                },
                "sidewalk_meta": {
                    "clip_id": "clip-1",
                    "timestamp": 1200,
                    "multiframe_frames": [{"timestamp": 1200}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_stage_ii_cached_traj_query_pairs_v2(
        source_views_jsonl=source_views_path,
        dataset_manifests={"sidewalk": [source_path]},
        trajectory_source_registry_json=registry_path,
        output_pairs_jsonl=tmp_path / "pairs.jsonl",
        output_reject_jsonl=tmp_path / "rejects.jsonl",
        output_report_json=tmp_path / "report.json",
        query_period_sec=0.5,
        max_query_age_sec=0.5,
    )

    assert report["complete"] is True
    assert report["counts"]["accepted_pairs"] == 1
    pair = json.loads((tmp_path / "pairs.jsonl").read_text())
    assert pair["condition_mode"] == STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2
    assert "reasoning_context" not in pair
    assert "decision_context" not in pair


def test_cached_source_view_materializer_binds_stage_i_and_sidecar(tmp_path):
    from test_stage_ii_cfm import _source_registry_payload

    stage_i_view = _stage_i_full_v2()
    sidecar = _trajectory_sidecar_v2(stage_i_view)
    registry = _source_registry_payload()
    sidewalk = next(
        item for item in registry["datasets"] if item["dataset_source"] == "sidewalk"
    )
    sidewalk["sample_identity_fields"] = ["id"]
    registry["registry_digest"] = stage_ii_digest_v1(
        {key: value for key, value in registry.items() if key != "registry_digest"}
    )

    stage_i_path = tmp_path / "stage_i.jsonl"
    sidecar_path = tmp_path / "trajectory_sidecars.jsonl"
    source_path = tmp_path / "sidewalk.jsonl"
    registry_path = tmp_path / "registry.json"
    output_source_path = tmp_path / "source_views.jsonl"
    output_reject_path = tmp_path / "rejects.jsonl"
    output_report_path = tmp_path / "report.json"
    stage_i_path.write_text(
        json.dumps(stage_i_view.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    source_path.write_text(
        json.dumps(
            {
                "id": "case-straight",
                "trajectory_policy": {
                    "schema_version": "sidewalk_time_conditioned_policy_v1",
                    "time_unit": "seconds",
                },
                "sidewalk_meta": {
                    "clip_id": "clip-1",
                    "timestamp": 1200,
                    "multiframe_frames": [{"timestamp": 1200}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar_payload = sidecar.model_dump(mode="json")
    sidecar_payload["source_manifest_sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    sidecar_payload["sidecar_digest"] = stage_ii_digest_v1(
        {key: value for key, value in sidecar_payload.items() if key != "sidecar_digest"}
    )
    sidecar_path.write_text(
        json.dumps(sidecar_payload) + "\n",
        encoding="utf-8",
    )
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = build_stage_ii_cached_traj_query_source_views_v2(
        stage_i_views_jsonl=stage_i_path,
        trajectory_sidecars_jsonl=[sidecar_path],
        dataset_manifests={"sidewalk": [source_path]},
        trajectory_source_registry_json=registry_path,
        output_source_views_jsonl=output_source_path,
        output_reject_jsonl=output_reject_path,
        output_report_json=output_report_path,
    )

    assert report["counts"]["accepted_source_views"] == 1
    assert report["counts"].get("rejected_source_views", 0) == 0
    payload = json.loads(output_source_path.read_text(encoding="utf-8"))
    assert payload["trajectory_target"]["valid_step_count"] == 32
    assert payload["observation"]["observation_timestamp_sec"] == 1.2
    assert output_reject_path.read_text(encoding="utf-8") == ""


def _stage_ii_view(label: str = "straight", *, instruction: str | None = None):
    """Build one valid Stage II route over a distinct physical Stage I sample."""

    stage_i_view = _stage_i_full_v2() if label == "straight" else None
    if stage_i_view is None:
        from mcp_server.data.stage_i_views import project_stage_i_reasoning_view_v2
        from test_stage_i_sft import _formal_reasoning_view

        reasoning_view = _formal_reasoning_view(label)
        if instruction is not None:
            payload = reasoning_view.model_dump(mode="json")
            payload["instruction"] = instruction
            reasoning_view = type(reasoning_view).model_validate(payload)
        stage_i_view = project_stage_i_reasoning_view_v2(
            reasoning_view,
            task_type="full_cot",
            modality_profile_id="fpv_instruction_v1",
            instruction_training_role="primary",
        )
    decision = _cache(stage_i_view, "decision", _decision_text(stage_i_view))
    return project_stage_ii_cfm_view_v2(
        stage_i_view,
        trajectory_target=_trajectory_v2(),
        reasoning_context=_cache(
            stage_i_view,
            "reasoning_grounding",
            stage_i_view.assistant_text,
        ),
        decision_context=decision,
        review_status="not_required",
    )


def test_cached_pair_accepts_different_physical_samples_in_one_split():
    query_view = _stage_ii_view("left_10")
    action_view = _stage_ii_view("straight")

    pair = build_stage_ii_cached_traj_query_training_view_v1(
        action_view,
        query_view,
        query_source_timestamp_sec=1.0,
        action_observation_timestamp_sec=1.4,
        query_source_frame_timestamps_sec=[1.0],
        action_frame_timestamps_sec=[1.4],
    )

    assert pair.query_source.source_group_id != pair.action_observation.source_group_id
    assert pair.action_observation.source_group_id == pair.action_view.source_group_id
    assert pair.cached_traj_query.query_age_sec == pytest.approx(0.4)


def test_cached_pair_rejects_visible_condition_change():
    query_view = _stage_ii_view("left_10", instruction="Turn left at the next path.")
    action_view = _stage_ii_view("straight")

    with pytest.raises(ValueError, match="visible inputs disagree"):
        build_stage_ii_cached_traj_query_training_view_v1(
            action_view,
            query_view,
            query_source_timestamp_sec=1.0,
            action_observation_timestamp_sec=1.4,
            query_source_frame_timestamps_sec=[1.0],
            action_frame_timestamps_sec=[1.4],
        )


def test_cached_pair_accepts_older_bev_after_current_fpv_slot():
    action_view = _stage_ii_view("straight")
    payload = {
        "schema_version": STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION,
        "sample_id": action_view.sample_id,
        "source_group_id": action_view.source_group_id,
        "split_group_id": action_view.split_group_id,
        "dataset_source": action_view.dataset_source,
        "modality_profile_id": action_view.modality_profile_id,
        "images": ["/data/current.jpg", "/data/current_bev.jpg"],
        "image_roles": ["fpv.current", "bev.current"],
        "instruction": action_view.instruction,
        "model_visible_context": action_view.model_visible_context,
        "evidence_catalog": [
            item.model_dump(mode="json") for item in action_view.evidence_catalog
        ],
        "input_modalities": action_view.input_modalities.model_dump(mode="json"),
        "frame_timestamps_sec": [1.0, 0.8],
        "observation_timestamp_sec": 1.0,
        "bev_timestamp_sec": 0.8,
    }
    payload["view_digest"] = stage_ii_digest_v1(payload)

    observation = StageIICachedTrajQueryObservationV1.model_validate(payload)

    assert observation.frame_timestamps_sec == [1.0, 0.8]


def test_cached_pair_materializer_uses_later_action_sample(tmp_path):
    from test_stage_ii_cfm import _source_registry_payload

    registry = _source_registry_payload()
    sidewalk = next(item for item in registry["datasets"] if item["dataset_source"] == "sidewalk")
    sidewalk["sample_identity_fields"] = ["id"]
    registry_path = tmp_path / "registry.json"
    registry["registry_digest"] = stage_ii_digest_v1(
        {key: value for key, value in registry.items() if key != "registry_digest"}
    )
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    query_view = _stage_ii_view("left_10")
    action_view = _stage_ii_view("straight")
    views_path = tmp_path / "stage_ii_views.jsonl"
    views_path.write_text(
        json.dumps(query_view.model_dump(mode="json"))
        + "\n"
        + json.dumps(action_view.model_dump(mode="json"))
        + "\n",
        encoding="utf-8",
    )

    def source_row(source_id: str, timestamp: int) -> dict:
        return {
            "id": source_id,
            "trajectory_policy": {
                "schema_version": "sidewalk_time_conditioned_policy_v1",
                "time_unit": "seconds",
                "trajectory_source": "calibrated_pose_projection",
            },
            "sidewalk_meta": {
                "clip_id": "clip-1",
                "timestamp": timestamp,
                "bev_timestamp": timestamp,
                "multiframe_frames": [{"timestamp": timestamp}],
            },
        }

    source_path = tmp_path / "sidewalk.jsonl"
    source_path.write_text(
        json.dumps(source_row("case-left_10", 1000))
        + "\n"
        + json.dumps(source_row("case-straight", 1200))
        + "\n",
        encoding="utf-8",
    )
    pairs_path = tmp_path / "pairs.jsonl"
    rejects_path = tmp_path / "rejects.jsonl"
    report_path = tmp_path / "report.json"

    report = build_stage_ii_cached_traj_query_pairs_v1(
        stage_ii_views_jsonl=views_path,
        dataset_manifests={"sidewalk": [source_path]},
        trajectory_source_registry_json=registry_path,
        output_pairs_jsonl=pairs_path,
        output_reject_jsonl=rejects_path,
        output_report_json=report_path,
        query_period_sec=0.5,
        max_query_age_sec=0.5,
    )

    rows = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    assert report["counts"]["accepted_pairs"] == 2
    assert report["counts"]["query_refreshes"] == 1
    assert rows[1]["cached_traj_query"]["query_age_sec"] == pytest.approx(0.2)
    assert rows[1]["query_source"]["source_group_id"] != rows[1]["action_observation"]["source_group_id"]
