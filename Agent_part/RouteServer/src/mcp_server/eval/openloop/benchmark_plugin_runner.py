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


__all__ = ["evaluate_benchmark_plugin_v1"]
