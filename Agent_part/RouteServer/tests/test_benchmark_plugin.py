"""Focused contract, package, and metric tests for benchmark-plugin V1."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkMediaManifestEntryV1,
    BenchmarkMessageV1,
    BenchmarkPredictionV1,
    BenchmarkReferenceV1,
    BenchmarkTimePolicyV1,
)
from mcp_server.data.benchmark.citywalker_dataset_plugin import citywalker_default_profile_v1
from mcp_server.data.benchmark.registry import registered_benchmark_dataset_plugin_ids_v1
from mcp_server.eval.openloop.benchmark_plugin_package import (
    build_benchmark_package_manifest_v1,
    input_digest_v1,
    load_benchmark_package_v1,
    write_benchmark_rows_v1,
)
from mcp_server.eval.openloop.benchmark_plugin_media import build_benchmark_media_manifest_v1
from mcp_server.eval.openloop.benchmark_plugin_runner import (
    benchmark_evaluation_admission_v1,
    evaluate_benchmark_plugin_v1,
)


def _profile():
    """Return a short synthetic CityWalker profile with two time horizons."""

    return citywalker_default_profile_v1(profile_id='test_citywalker_v1', horizons_s=(1.0, 2.0))


def _input() -> BenchmarkInputV1:
    """Return one strict model-visible synthetic CityWalker input."""

    return BenchmarkInputV1(
        sample_id='citywalker:10:frame:0001:target:010:seed:1',
        parent_sample_id='citywalker:10:frame:0001',
        source_group_id='citywalker_sequence_10',
        split_group_id='citywalker_sequence_10',
        split='test',
        dataset_plugin_id='citywalker_teleop_v1',
        benchmark_profile_id='test_citywalker_v1',
        dataset_source='citywalker_teleop',
        messages=[BenchmarkMessageV1(role='user', content='visible prompt')],
        images=['/tmp/citywalker.jpg'],
        image_roles=['fpv_history'],
        public_context={'target_xy_m': [0.0, 2.0]},
        modality_profile_id='citywalker_fpv_pose_goal_prompt_v1',
    )


def _reference() -> BenchmarkReferenceV1:
    """Return one hidden two-point synthetic CityWalker reference."""

    return BenchmarkReferenceV1(
        sample_id='citywalker:10:frame:0001:target:010:seed:1',
        parent_sample_id='citywalker:10:frame:0001',
        source_group_id='citywalker_sequence_10',
        split_group_id='citywalker_sequence_10',
        split='test',
        dataset_plugin_id='citywalker_teleop_v1',
        benchmark_profile_id='test_citywalker_v1',
        dataset_source='citywalker_teleop',
        trajectory=[[0.0, 1.0], [0.0, 2.0]],
        trajectory_timestamps_s=[1.0, 2.0],
        trajectory_valid_mask=[True, True],
        coordinate_frame='citywalker_current_pose_x_right_y_front_v1',
        arrival_label=True,
        scenario_labels=('turn', 'crossing'),
        label_provenance='reference_annotation',
    )


def test_plugin_input_rejects_hidden_target_key():
    """The generic public boundary must reject a nested hidden trajectory key."""

    payload = _input().model_dump(mode='json')
    payload['public_context'] = {'nested': {'trajectory_ego_t': [[0.0, 1.0]]}}
    with pytest.raises(ValueError, match='private supervision'):
        BenchmarkInputV1(**payload)


def test_dataset_plugin_registry_has_citywalker_and_urbannav():
    """The registry must expose every legal RouteServer-owned plugin explicitly."""

    assert registered_benchmark_dataset_plugin_ids_v1() == (
        'citywalker_teleop_v1',
        'urbannav_language_waypoint_v1',
        'urbannav_stage_ii_projection_v1',
    )


def test_plugin_package_runner_reports_geometry_angle_and_arrival(tmp_path: Path):
    """A perfect explicit prediction must yield exact common/CityWalker scores."""

    profile = _profile()
    input_row = _input()
    reference = _reference()
    prediction = BenchmarkPredictionV1(
        sample_id=input_row.sample_id,
        parent_sample_id=input_row.parent_sample_id,
        benchmark_profile_id=profile.profile_id,
        model_adapter_id='synthetic',
        adaptation_mode='zero_shot',
        prediction_geometry_id='citywalker_current_pose_xy_v1',
        trajectory=[[0.0, 1.0], [0.0, 2.0]],
        time_basis='explicit',
        trajectory_timestamps_s=[1.0, 2.0],
        arrival_score=0.9,
        input_digest=input_digest_v1(input_row),
    )
    inputs = tmp_path / 'inputs.jsonl'
    references = tmp_path / 'references.jsonl'
    manifest = tmp_path / 'package.json'
    write_benchmark_rows_v1(inputs, [input_row])
    write_benchmark_rows_v1(references, [reference])
    build_benchmark_package_manifest_v1(
        package_id='synthetic-citywalker',
        profile=profile,
        split='test',
        input_jsonl=inputs,
        reference_jsonl=references,
        output_json=manifest,
    )
    package, loaded_inputs, loaded_references = load_benchmark_package_v1(manifest)
    report = evaluate_benchmark_plugin_v1(
        profile=package.benchmark_profile,
        inputs=loaded_inputs,
        references=loaded_references,
        predictions={prediction.sample_id: prediction},
    )
    geometry = report['metric_suites']['trajectory_geometry_core_v1']
    citywalker = report['metric_suites']['citywalker_openloop_metrics_v1']
    assert geometry['amde_m'] == 0.0
    assert geometry['horizons']['1s']['geometry_coverage'] == 1.0
    assert citywalker['angle']['maoe_deg'] == 0.0
    assert citywalker['arrival']['accuracy'] == 1.0
    assert citywalker['overlap_permitted_slices']['turn']['reference_count'] == 1
    assert citywalker['overlap_permitted_slices']['crossing']['reference_count'] == 1


def test_plugin_runner_reports_missing_prediction_as_coverage_drop():
    """A missing output must preserve the full hidden reference denominator."""

    profile = _profile()
    input_row = _input()
    report = evaluate_benchmark_plugin_v1(
        profile=profile,
        inputs={input_row.sample_id: input_row},
        references={input_row.sample_id: _reference()},
        predictions={},
    )
    geometry = report['metric_suites']['trajectory_geometry_core_v1']
    assert geometry['reference_count'] == 1
    assert geometry['success_coverage'] == 0.0
    assert geometry['horizons']['1s']['valid_prediction_count'] == 0


def test_media_manifest_binds_portable_bytes_without_rewriting_input_digest(
        tmp_path: Path):
    """A package must bind logical paths to exact portable image bytes."""

    media_root = tmp_path / 'media'
    image = media_root / 'traj_10' / 'forward_0001.jpg'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'public-citywalker-frame')
    payload = _input().model_dump(mode='json')
    payload['images'] = [str(image)]
    input_row = BenchmarkInputV1.model_validate(payload)
    reference = _reference()
    inputs = tmp_path / 'inputs.jsonl'
    references = tmp_path / 'references.jsonl'
    media_manifest = tmp_path / 'media_manifest.jsonl'
    package_json = tmp_path / 'package.json'
    write_benchmark_rows_v1(inputs, [input_row])
    write_benchmark_rows_v1(references, [reference])
    media_report = build_benchmark_media_manifest_v1(
        inputs=[input_row],
        source_root=media_root,
        output_jsonl=media_manifest,
    )
    build_benchmark_package_manifest_v1(
        package_id='portable-media',
        profile=_profile(),
        split='test',
        input_jsonl=inputs,
        reference_jsonl=references,
        output_json=package_json,
        media_manifest_jsonl=media_manifest,
    )
    package, loaded_inputs, _ = load_benchmark_package_v1(package_json)
    assert package.media_manifest_path == 'media_manifest.jsonl'
    assert package.media_manifest_digest == media_report['media_manifest_digest']
    assert package.media_manifest_count == 1
    assert input_digest_v1(loaded_inputs[input_row.sample_id]) == input_digest_v1(input_row)


def test_media_manifest_rejects_path_traversal():
    """Runtime media entries must never escape an execution mirror root."""

    with pytest.raises(ValueError, match='safe non-empty relative'):
        BenchmarkMediaManifestEntryV1(
            logical_path='/logical/frame.jpg',
            relative_path='../escape.jpg',
            byte_size=1,
            sha256='a' * 64,
        )


def test_evaluation_admission_rejects_all_failure_ledger():
    """An evaluator scorecard may exist, but a zero-success run cannot pass."""

    profile = _profile()
    failed = benchmark_evaluation_admission_v1(
        profile=profile,
        reference_count=8,
        prediction_count=0,
    )
    passed = benchmark_evaluation_admission_v1(
        profile=profile,
        reference_count=8,
        prediction_count=8,
    )
    assert failed['status'] == 'failed_inference_coverage'
    assert failed['success_coverage'] == 0.0
    assert passed['status'] == 'passed'
