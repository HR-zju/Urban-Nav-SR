import pytest

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE,
    StageIICachedTrajQueryBindingV1,
)
from mcp_server.data.stage_ii.views import (
    build_stage_ii_cached_traj_query_training_view_v1,
)
from mcp_server.data.stage_ii_views import project_stage_ii_cfm_view_v2

from test_stage_ii_cfm_v2 import (
    _cache,
    _decision_text,
    _stage_i_full_v2,
    _trajectory_v2,
)


def test_cached_traj_query_binding_requires_causal_consistent_timestamps():
    binding = StageIICachedTrajQueryBindingV1(
        query_source_timestamp_sec=10.0,
        action_observation_timestamp_sec=10.25,
        query_age_sec=0.25,
        modality_profile_id="fpv_temporal_instruction_bev_v1",
    )
    assert binding.condition_mode == STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE


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
