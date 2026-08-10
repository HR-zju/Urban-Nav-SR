"""Offline deterministic runner for model-independent benchmark packages."""

from __future__ import annotations

from typing import Any

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkFailureV1,
    BenchmarkInputV1,
    BenchmarkPredictionV1,
    BenchmarkProfileV1,
    BenchmarkReferenceV1,
    benchmark_plugin_digest_v1,
)
from mcp_server.eval.openloop.benchmark_plugin_package import join_benchmark_artifacts_v1
from mcp_server.eval.openloop.benchmark_plugin_registry import get_metric_plugin_v1


def evaluate_benchmark_plugin_v1(
    *,
    profile: BenchmarkProfileV1,
    inputs: dict[str, BenchmarkInputV1],
    references: dict[str, BenchmarkReferenceV1],
    predictions: dict[str, BenchmarkPredictionV1],
    failures: dict[str, BenchmarkFailureV1] | None = None,
) -> dict[str, Any]:
    """Validate ledgers and evaluate every metric suite selected by a profile.

    Args:
        profile: Immutable legal plugin/profile/time contract.
        inputs: Exact model-visible cases keyed by sample id.
        references: Exact hidden references keyed by sample id.
        predictions: Successful output ledger keyed by sample id.
        failures: Optional explicit failure ledger keyed by sample id.

    Returns:
        Deterministic JSON-compatible report with per-suite scorecards.

    Raises:
        ValueError: If package/profile/input-digest invariants drift.
    """

    _validate_runner_ledgers_v1(profile, inputs, references, predictions, failures or {})
    joined = join_benchmark_artifacts_v1(references, predictions, failures)
    suites = {
        suite_id: get_metric_plugin_v1(suite_id).evaluate(joined, profile=profile)
        for suite_id in profile.metric_suite_ids
    }
    return {
        "schema_version": "urbannav_benchmark_evaluation_v1",
        "benchmark_profile": profile.model_dump(mode="json"),
        "profile_digest": benchmark_plugin_digest_v1(profile),
        "reference_count": len(references),
        "prediction_count": len(predictions),
        "failure_count": len(failures or {}),
        "metric_suites": suites,
    }


def benchmark_evaluation_admission_v1(
    *,
    profile: BenchmarkProfileV1,
    reference_count: int,
    prediction_count: int,
    minimum_successful_predictions: int = 1,
) -> dict[str, object]:
    """Decide whether a formal benchmark runner may report ``passed``.

    The evaluator still writes a complete scorecard for all explicit failures,
    but an all-failure ledger is an operational failure rather than a valid
    model result.  Let ``N`` be the frozen denominator and ``P`` successful
    predictions.  Admission requires ``P >= minimum_successful_predictions``
    and ``P / N >= profile.minimum_success_coverage``.

    Args:
        profile: Frozen benchmark profile declaring minimum success coverage.
        reference_count: Full frozen denominator ``N``.
        prediction_count: Successful prediction count ``P``.
        minimum_successful_predictions: Absolute nonzero/smoke guard.

    Returns:
        JSON-compatible passed or failed-inference-coverage gate report.

    Raises:
        ValueError: If counts or the requested minimum are invalid.
    """

    if reference_count <= 0:
        raise ValueError('benchmark reference_count must be positive')
    if prediction_count < 0 or prediction_count > reference_count:
        raise ValueError('benchmark prediction_count must be within frozen denominator')
    if minimum_successful_predictions < 1:
        raise ValueError('minimum_successful_predictions must be positive')
    success_coverage = prediction_count / reference_count
    count_passed = prediction_count >= minimum_successful_predictions
    coverage_passed = success_coverage >= profile.minimum_success_coverage
    passed = count_passed and coverage_passed
    return {
        'schema_version': 'urbannav_benchmark_evaluation_admission_v1',
        'status': 'passed' if passed else 'failed_inference_coverage',
        'reference_count': reference_count,
        'successful_prediction_count': prediction_count,
        'success_coverage': success_coverage,
        'minimum_successful_predictions': minimum_successful_predictions,
        'minimum_success_coverage': profile.minimum_success_coverage,
        'count_passed': count_passed,
        'coverage_passed': coverage_passed,
    }


def _validate_runner_ledgers_v1(
    profile: BenchmarkProfileV1,
    inputs: dict[str, BenchmarkInputV1],
    references: dict[str, BenchmarkReferenceV1],
    predictions: dict[str, BenchmarkPredictionV1],
    failures: dict[str, BenchmarkFailureV1],
) -> None:
    """Fail closed on profile/input digest drift before any metric calculation."""

    if set(inputs) != set(references):
        raise ValueError("runner input/reference sample_id mismatch")
    if set(predictions) & set(failures):
        raise ValueError("prediction/failure ledgers overlap")
    for sample_id, input_row in inputs.items():
        reference = references[sample_id]
        if input_row.benchmark_profile_id != profile.profile_id or reference.benchmark_profile_id != profile.profile_id:
            raise ValueError(f"profile id mismatch: {sample_id}")
        if (
            input_row.dataset_plugin_id != profile.dataset_plugin_id
            or reference.dataset_plugin_id != profile.dataset_plugin_id
        ):
            raise ValueError(f"dataset plugin mismatch: {sample_id}")
        digest = benchmark_plugin_digest_v1(input_row)
        output = predictions.get(sample_id)
        failure = failures.get(sample_id)
        if output is not None:
            if output.input_digest != digest:
                raise ValueError(f"prediction input digest mismatch: {sample_id}")
            if output.prediction_geometry_id != profile.prediction_geometry_id:
                raise ValueError(f"prediction geometry mismatch: {sample_id}")
        if failure is not None and failure.input_digest != digest:
            raise ValueError(f"failure input digest mismatch: {sample_id}")


__all__ = ["benchmark_evaluation_admission_v1", "evaluate_benchmark_plugin_v1"]
