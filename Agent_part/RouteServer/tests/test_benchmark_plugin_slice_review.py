"""Focused deterministic tests for post-hoc plugin-benchmark slice review."""

from __future__ import annotations

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkMessageV1,
    BenchmarkPredictionV1,
    BenchmarkReferenceV1,
    benchmark_plugin_digest_v1,
)
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkJoinedRowV1
from mcp_server.eval.openloop.benchmark_plugin_review import write_benchmark_plugin_review_html_v1
from mcp_server.eval.openloop.benchmark_plugin_review_base import (
    BenchmarkNativeAdapterTraceV1, )
from mcp_server.eval.openloop.benchmark_plugin_slice_review import (
    build_benchmark_slice_review_diagnosis_template_v1,
    build_benchmark_slice_review_selection_v1,
    evaluate_benchmark_coordinate_diagnostic_v1,
    evaluate_benchmark_reference_motion_diagnostic_v1,
    select_benchmark_slice_review_rows_v1,
)


def _row(index: int, *, labels: tuple[str, ...],
         prediction_xy: tuple[float, float]) -> BenchmarkJoinedRowV1:
    """Return one two-second synthetic row with a controlled final L2 error."""

    sample_id = f"citywalker:synthetic:{index:02d}"
    target = [float(index), 2.0]
    reference = BenchmarkReferenceV1(
        sample_id=sample_id,
        parent_sample_id=f"parent:{index:02d}",
        source_group_id="synthetic_group",
        split_group_id="synthetic_group",
        split="validation",
        dataset_plugin_id="citywalker_teleop_v1",
        benchmark_profile_id="test_citywalker_v1",
        dataset_source="citywalker_teleop",
        trajectory=[[target[0] / 2.0, 1.0], target],
        trajectory_timestamps_s=[1.0, 2.0],
        trajectory_valid_mask=[True, True],
        coordinate_frame="citywalker_current_pose_x_right_y_front_v1",
        scenario_labels=labels,
        label_provenance="reference_annotation",
    )
    prediction = BenchmarkPredictionV1(
        sample_id=sample_id,
        parent_sample_id=f"parent:{index:02d}",
        benchmark_profile_id="test_citywalker_v1",
        model_adapter_id="synthetic",
        adaptation_mode="zero_shot",
        prediction_geometry_id="citywalker_current_pose_xy_v1",
        trajectory=[[prediction_xy[0] / 2.0, prediction_xy[1] / 2.0],
                    list(prediction_xy)],
        time_basis="explicit",
        trajectory_timestamps_s=[1.0, 2.0],
        input_digest="0" * 64,
        metadata={"coordinate_projection_id": "flu_to_citywalker_xy_v1"},
    )
    return BenchmarkJoinedRowV1(reference=reference,
                                prediction=prediction,
                                failure=None)


def test_slice_selection_is_stable_and_preserves_overlap_reasons():
    """One physical case may retain multiple slice-review reasons exactly once."""

    rows = [
        _row(index,
             labels=("turn", "crossing") if index % 2 == 0 else ("turn", ),
             prediction_xy=(index, 2.0 + index)) for index in range(1, 10)
    ]
    first = build_benchmark_slice_review_selection_v1(
        rows,
        slice_labels=("turn", "crossing"),
        horizon_s=2.0,
        per_slice=4,
        seed=7,
    )
    second = build_benchmark_slice_review_selection_v1(
        rows,
        slice_labels=("turn", "crossing"),
        horizon_s=2.0,
        per_slice=4,
        seed=7,
    )
    assert first == second
    assert first["slice_summaries"]["turn"]["selected_count"] == 4
    assert first["slice_summaries"]["crossing"]["selected_count"] == 4
    assert len(first["entries"]) == len(
        {entry["sample_id"]
         for entry in first["entries"]})
    assert any(
        len(entry["selection_reasons"]) == 2 for entry in first["entries"])
    resolved = select_benchmark_slice_review_rows_v1(rows, selection=first)
    assert [row.reference.sample_id
            for row in resolved] == first["selected_sample_ids"]
    template = build_benchmark_slice_review_diagnosis_template_v1(first)
    assert len(template) == len(resolved)
    assert all(item["review_status"] == "pending" for item in template)


def test_coordinate_diagnostic_detects_global_x_sign_candidate():
    """A synthetic sign inversion should make only its audit transform optimal."""

    rows = [
        _row(index, labels=("turn", ), prediction_xy=(-float(index), 2.0))
        for index in range(1, 6)
    ]
    diagnostic = evaluate_benchmark_coordinate_diagnostic_v1(rows,
                                                             horizon_s=2.0)
    assert diagnostic["comparable_count"] == 5
    assert diagnostic["best_transform_id"] == "x_sign_flip"
    assert diagnostic["identity_minus_best_mean_l2_m"] > 0.0
    assert diagnostic[
        "interpretation"] == "global_axis_or_sign_transform_improves_l2_candidate"


def test_reference_motion_diagnostic_keeps_non_comparable_rows_out_of_bucket_denominator(
):
    """Reference-motion grouping must expose only prediction/reference pairs."""

    rows = [
        _row(index, labels=("other", ), prediction_xy=(float(index), 2.0))
        for index in range(1, 5)
    ]
    diagnostic = evaluate_benchmark_reference_motion_diagnostic_v1(
        rows, horizon_s=2.0)
    assert diagnostic["comparable_count"] == 4
    assert sum(item["count"] for item in diagnostic["bins"].values()) == 4
    assert diagnostic["bins"]["reference_displacement_1_to_3m"]["count"] == 2
    assert diagnostic["bins"]["reference_displacement_ge_3m"]["count"] == 2


def test_portable_review_can_embed_resolved_media_and_selection_annotation(
        tmp_path):
    """The human-review bundle must retain copied-media and audit context."""

    row = _row(1, labels=("turn", ), prediction_xy=(1.0, 2.0))
    input_row = BenchmarkInputV1(
        sample_id=row.reference.sample_id,
        parent_sample_id=row.reference.parent_sample_id,
        source_group_id=row.reference.source_group_id,
        split_group_id=row.reference.split_group_id,
        split="validation",
        dataset_plugin_id="citywalker_teleop_v1",
        benchmark_profile_id="test_citywalker_v1",
        dataset_source="citywalker_teleop",
        messages=[BenchmarkMessageV1(role="user", content="visible prompt")],
        images=["/source/frame.jpg"],
        image_roles=["fpv_history"],
        public_context={
            "coordinate_frame": "citywalker_current_pose_x_right_y_front_v1",
            "history_xy_m": [[-0.2, -0.7], [0.0, 0.0]],
            "target_xy_m": [50.0, 0.0],
        },
        modality_profile_id="citywalker_fpv_pose_goal_prompt_v1",
    )
    output = tmp_path / "review.html"
    write_benchmark_plugin_review_html_v1(
        path=output,
        rows=[row],
        inputs={input_row.sample_id: input_row},
        horizon_s=2.0,
        title="test",
        media_href_resolver=lambda _image: "media/frame.jpg",
        review_annotations={input_row.sample_id: "turn / high_l2 / L2=2.000m"},
        embed_media=True,
    )
    text = output.read_text(encoding="utf-8")
    assert 'src="media/frame.jpg"' in text
    assert "audit selection:" in text
    assert 'class="fpv current"' in text
    assert "x_right (m)" in text
    assert "y_front (m)" in text
    assert "#5771DB" in text
    assert "#A157DB" in text
    assert "#4F9A26" in text
    assert "#C6423B" in text
    assert 'text-anchor="end" class="goal-label">goal (clipped; 50.0 m)' in text


def test_native_adapter_review_adds_three_panels_without_hiding_terminals(
        tmp_path):
    """Trace-backed cards must reveal native choices and split 2s from terminal."""

    sample_id = "citywalker:synthetic:trace"
    reference = BenchmarkReferenceV1(
        sample_id=sample_id,
        parent_sample_id="parent:trace",
        source_group_id="synthetic_group",
        split_group_id="synthetic_group",
        split="validation",
        dataset_plugin_id="citywalker_teleop_v1",
        benchmark_profile_id="test_citywalker_v1",
        dataset_source="citywalker_teleop",
        trajectory=[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        trajectory_timestamps_s=[1.0, 2.0, 3.0],
        trajectory_valid_mask=[True, True, True],
        coordinate_frame="citywalker_current_pose_x_right_y_front_v1",
        label_provenance="reference_annotation",
    )
    prediction = BenchmarkPredictionV1(
        sample_id=sample_id,
        parent_sample_id="parent:trace",
        benchmark_profile_id="test_citywalker_v1",
        model_adapter_id="citywalker_native_synthetic",
        adaptation_mode="reference",
        prediction_geometry_id="citywalker_current_pose_xy_v1",
        trajectory=[[0.2, 1.0], [0.2, 2.0], [0.2, 3.0]],
        time_basis="explicit",
        trajectory_timestamps_s=[1.0, 2.0, 3.0],
        input_digest="0" * 64,
    )
    row = BenchmarkJoinedRowV1(reference=reference,
                               prediction=prediction,
                               failure=None)
    input_row = BenchmarkInputV1(
        sample_id=sample_id,
        parent_sample_id="parent:trace",
        source_group_id="synthetic_group",
        split_group_id="synthetic_group",
        split="validation",
        dataset_plugin_id="citywalker_teleop_v1",
        benchmark_profile_id="test_citywalker_v1",
        dataset_source="citywalker_teleop",
        messages=[BenchmarkMessageV1(role="user", content="visible prompt")],
        images=["/source/history.jpg", "/source/current.jpg"],
        image_roles=["fpv_history", "fpv_current"],
        public_context={
            "coordinate_frame": "citywalker_current_pose_x_right_y_front_v1",
            "history_xy_m": [[0.0, -1.0], [0.0, 0.0]],
            "target_xy_m": [5.0, 0.0],
        },
        modality_profile_id="citywalker_fpv_pose_goal_prompt_v1",
    )
    trace = BenchmarkNativeAdapterTraceV1(
        sample_id=sample_id,
        benchmark_input_digest=benchmark_plugin_digest_v1(input_row),
        native_adapter_id="citywalker_to_urbannav_v1",
        native_adapter_version="1.0",
        native_checkpoint_digest="b" * 64,
        selected_image_indices=(0, ),
        selected_image_roles=("fpv_history", ),
        public_history_transform_id="public_history_to_native_v1",
        public_scale_estimator_id="public_history_scale_v1",
        public_scale_value=1.5,
        input_coordinate_projection_id="urbannav_to_citywalker_v1",
        output_coordinate_projection_id="citywalker_to_urbannav_v1",
        native_output_schedule_id="citywalker_five_waypoint_schedule_v1",
        native_output_count=5,
        eligibility_status="eligible",
    )
    output = tmp_path / "native_review.html"
    write_benchmark_plugin_review_html_v1(
        path=output,
        rows=[row],
        inputs={sample_id: input_row},
        horizon_s=2.0,
        title="native review",
        media_href_resolver=lambda _image: "media/frame.jpg",
        embed_media=True,
        adapter_traces={sample_id: trace},
        coverage_horizons_s=(1.0, 2.0),
    )
    text = output.read_text(encoding="utf-8")
    assert "1. Model-visible UrbanNav input" in text
    assert "2. Native-adapter audit" in text
    assert "3. Evaluator-only geometry" in text
    assert "Adapter coverage funnel (review denominator)" in text
    assert "public scale estimator" in text
    assert 'class="fpv native-selected"' in text
    assert 'class="fpv current native-unused"' in text
    assert 'stroke-dasharray="8 6"' in text
    assert "GT @ 2s" in text
    assert "Pred @ 2s" in text
    assert "GT terminal @ 3s" in text
    assert "Pred terminal @ 3s" in text
