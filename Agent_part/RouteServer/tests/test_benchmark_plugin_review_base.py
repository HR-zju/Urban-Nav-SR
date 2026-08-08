"""Focused tests for public native-adapter review sidecars and time splits."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkMessageV1,
    BenchmarkPredictionV1,
    BenchmarkReferenceV1,
    benchmark_plugin_digest_v1,
)
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkJoinedRowV1
from mcp_server.eval.openloop.benchmark_plugin_review_base import (
    BenchmarkNativeAdapterTraceV1,
    BenchmarkReviewRendererBaseV1,
    load_benchmark_native_adapter_traces_v1,
    split_benchmark_review_curve_at_horizon_v1,
    summarize_benchmark_native_adapter_funnel_v1,
)


def _input() -> BenchmarkInputV1:
    """Return one public two-frame synthetic benchmark input."""

    return BenchmarkInputV1(
        sample_id="urban:synthetic:01",
        parent_sample_id="parent:01",
        source_group_id="source:01",
        split_group_id="split:01",
        split="validation",
        dataset_plugin_id="urbannav_plugin_v1",
        benchmark_profile_id="urbannav_profile_v1",
        dataset_source="urban_synthetic",
        messages=[BenchmarkMessageV1(role="user", content="public prompt")],
        images=["/public/0.jpg", "/public/1.jpg"],
        image_roles=["fpv_history", "fpv_current"],
        public_context={"history_xy_m": [[0.0, -1.0], [0.0, 0.0]]},
        modality_profile_id="urban_fpv_v1",
    )


def _row() -> BenchmarkJoinedRowV1:
    """Return a three-second explicit trajectory ledger for review tests."""

    reference = BenchmarkReferenceV1(
        sample_id="urban:synthetic:01",
        parent_sample_id="parent:01",
        source_group_id="source:01",
        split_group_id="split:01",
        split="validation",
        dataset_plugin_id="urbannav_plugin_v1",
        benchmark_profile_id="urbannav_profile_v1",
        dataset_source="urban_synthetic",
        trajectory=[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        trajectory_timestamps_s=[1.0, 2.0, 3.0],
        trajectory_valid_mask=[True, True, True],
        coordinate_frame="urbannav_current_ego_flu_xy_v1",
    )
    prediction = BenchmarkPredictionV1(
        sample_id="urban:synthetic:01",
        parent_sample_id="parent:01",
        benchmark_profile_id="urbannav_profile_v1",
        model_adapter_id="native_citywalker_adapter_v1",
        adaptation_mode="reference",
        prediction_geometry_id="urbannav_current_ego_flu_xy_v1",
        trajectory=[[0.2, 1.0], [0.2, 2.0], [0.2, 3.0]],
        time_basis="explicit",
        trajectory_timestamps_s=[1.0, 2.0, 3.0],
        input_digest="0" * 64,
    )
    return BenchmarkJoinedRowV1(reference=reference,
                                prediction=prediction,
                                failure=None)


def _eligible_trace(
        input_row: BenchmarkInputV1) -> BenchmarkNativeAdapterTraceV1:
    """Return a strict public-only native adapter trace for the synthetic input."""

    return BenchmarkNativeAdapterTraceV1(
        sample_id=input_row.sample_id,
        benchmark_input_digest=benchmark_plugin_digest_v1(input_row),
        native_adapter_id="citywalker_to_urbannav_v1",
        native_adapter_version="1.0",
        native_checkpoint_digest="a" * 64,
        selected_image_indices=(0, 1),
        selected_image_roles=("fpv_history", "fpv_current"),
        public_history_transform_id="urban_public_history_to_citywalker_v1",
        public_scale_estimator_id="public_history_displacement_v1",
        public_scale_value=2.5,
        input_coordinate_projection_id="urbannav_flu_to_citywalker_xy_v1",
        output_coordinate_projection_id="citywalker_xy_to_urbannav_flu_v1",
        native_output_schedule_id="citywalker_five_waypoint_public_schedule_v1",
        native_output_count=5,
        eligibility_status="eligible",
    )


def test_time_split_keeps_score_prefix_and_future_terminal_separate():
    """The score marker at 2s must not be rendered as a 3s terminal."""

    split = split_benchmark_review_curve_at_horizon_v1(
        [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        [1.0, 2.0, 3.0],
        horizon_s=2.0,
    )
    assert split.scored_prefix == ((0.0, 1.0), (0.0, 2.0))
    assert split.future_suffix == ((0.0, 2.0), (0.0, 3.0))
    assert split.at_horizon == (0.0, 2.0)
    assert split.terminal == (0.0, 3.0)
    geometry = BenchmarkReviewRendererBaseV1(horizon_s=2.0).prepare_case(
        _row())
    assert geometry.l2_at_horizon_m == pytest.approx(0.2)


def test_trace_loader_binds_digest_roles_and_rejects_evaluator_fields(
        tmp_path):
    """Trace JSONL must bind public input and forbid hidden evaluator fields."""

    input_row = _input()
    trace = _eligible_trace(input_row)
    path = tmp_path / "native_adapter_trace.jsonl"
    path.write_text(json.dumps(trace.model_dump(mode="json")) + "\n",
                    encoding="utf-8")
    loaded = load_benchmark_native_adapter_traces_v1(
        path,
        inputs={input_row.sample_id: input_row},
        require_complete=True,
    )
    assert loaded == {input_row.sample_id: trace}

    leaked = trace.model_dump(mode="json")
    leaked["reference_timestamps_s"] = [1.0, 2.0]
    with pytest.raises(ValidationError, match="reference_timestamps_s"):
        BenchmarkNativeAdapterTraceV1.model_validate(leaked)

    wrong_role = trace.model_copy(
        update={"selected_image_roles": ("fpv_current", "fpv_history")})
    path.write_text(json.dumps(wrong_role.model_dump(mode="json")) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="selected image roles drift"):
        load_benchmark_native_adapter_traces_v1(
            path, inputs={input_row.sample_id: input_row})


def test_adapter_funnel_keeps_trace_and_metric_coverage_separate():
    """Adapter admission cannot shrink the evaluator review denominator."""

    input_row = _input()
    trace = _eligible_trace(input_row)
    funnel = summarize_benchmark_native_adapter_funnel_v1(
        rows=[_row()],
        inputs={input_row.sample_id: input_row},
        traces={input_row.sample_id: trace},
        horizons_s=(1.0, 2.0),
    )
    assert funnel["denominator_count"] == 1
    assert funnel["trace_present_count"] == 1
    assert funnel["adapter_status_counts"]["eligible"] == 1
    assert funnel["public_input_gate_counts"][
        "public_fpv_available_count"] == 1
    assert funnel["adapter_reject_reason_counts"]["invalid_scale"] == 0
    assert funnel["horizon_coverage"]["1s"]["joint_covered_count"] == 1
    assert funnel["horizon_coverage"]["2s"]["joint_covered_count"] == 1


def test_trace_reject_reason_must_match_its_admission_state():
    """A native model failure cannot masquerade as a public scale rejection."""

    input_row = _input()
    payload = _eligible_trace(input_row).model_dump(mode="json")
    payload.update({
        "selected_image_indices": (),
        "selected_image_roles": (),
        "public_history_transform_id": None,
        "public_scale_estimator_id": None,
        "public_scale_value": None,
        "input_coordinate_projection_id": None,
        "output_coordinate_projection_id": None,
        "native_output_schedule_id": None,
        "native_output_count": 0,
        "eligibility_status": "native_model_error",
        "reject_reason": "invalid_scale",
    })
    with pytest.raises(ValidationError,
                       match="native_model_error reject_reason"):
        BenchmarkNativeAdapterTraceV1.model_validate(payload)
