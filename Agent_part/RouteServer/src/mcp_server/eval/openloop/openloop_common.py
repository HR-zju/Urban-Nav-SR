"""Shared deterministic utilities for UrbanNav open-loop evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from pydantic import BaseModel

from mcp_server.contracts.openloop_benchmark import (
    CheckpointExposureRegistryV1,
    OpenLoopFailureV1,
    OpenLoopInputV1,
    OpenLoopPackageManifestV1,
    OpenLoopPredictionV1,
    OpenLoopReferenceV1,
    OpenLoopRunManifestV1,
)
from mcp_server.data.io.jsonl import iter_jsonl


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class OpenLoopJoinedRowV1:
    """Join one reference with at most one prediction or failure.

    Args:
        reference: Hidden reference defining the denominator.
        prediction: Successful model output, if present.
        failure: Explicit failed attempt, if present.
    """

    reference: OpenLoopReferenceV1
    prediction: OpenLoopPredictionV1 | None
    failure: OpenLoopFailureV1 | None

    @property
    def attempted(self) -> bool:
        """Return whether a prediction or explicit failure exists."""

        return self.prediction is not None or self.failure is not None


@dataclass(frozen=True)
class OpenLoopPackageArtifactsV1:
    """Carry a validated package, run, and typed row indexes.

    Args:
        package: Frozen benchmark package manifest.
        run: Frozen model run manifest.
        inputs: Model-visible rows keyed by sample id.
        references: Hidden rows keyed by sample id.
        input_path: Resolved model-input artifact path.
        reference_path: Resolved hidden-reference artifact path.
    """

    package: OpenLoopPackageManifestV1
    run: OpenLoopRunManifestV1
    inputs: dict[str, OpenLoopInputV1]
    references: dict[str, OpenLoopReferenceV1]
    input_path: Path
    reference_path: Path


def load_unique_jsonl_models_v1(
    path: str | Path,
    model_type: type[T],
    *,
    id_field: str = "sample_id",
) -> dict[str, T]:
    """Load strict typed JSONL rows keyed by one stable identity.

    Args:
        path: JSONL artifact path.
        model_type: Pydantic model used for strict validation.
        id_field: Model field containing the unique identity.

    Returns:
        Mapping preserving JSONL order.

    Raises:
        ValueError: If a row has no identity or duplicates a prior row.
    """

    rows: dict[str, T] = {}
    for payload in iter_jsonl(path):
        model = model_type.model_validate(payload)
        value = str(getattr(model, id_field, ""))
        if not value:
            raise ValueError(f"{model_type.__name__} requires {id_field}")
        if value in rows:
            raise ValueError(f"duplicate {id_field}: {value}")
        rows[value] = model
    return rows


def join_openloop_artifacts_v1(
    references: dict[str, OpenLoopReferenceV1],
    predictions: dict[str, OpenLoopPredictionV1],
    failures: dict[str, OpenLoopFailureV1] | None = None,
) -> list[OpenLoopJoinedRowV1]:
    """Join outputs to the complete frozen reference denominator.

    Missing rows remain in the returned list as unattempted failures.  Extra
    ids, duplicate success/failure ids, and identity drift fail closed.

    Args:
        references: Unique hidden reference rows.
        predictions: Unique successful prediction rows.
        failures: Unique explicit failure rows.

    Returns:
        Joined rows in reference-artifact order.

    Raises:
        ValueError: If artifact ids or row identities are inconsistent.
    """

    failures = failures or {}
    reference_ids = set(references)
    output_ids = set(predictions) | set(failures)
    overlap = set(predictions) & set(failures)
    extra = output_ids - reference_ids
    if overlap:
        raise ValueError(
            "sample ids appear in both predictions and failures: "
            + ",".join(sorted(overlap)[:8])
        )
    if extra:
        raise ValueError(
            "outputs contain unknown sample ids: "
            + ",".join(sorted(extra)[:8])
        )

    result: list[OpenLoopJoinedRowV1] = []
    for sample_id, reference in references.items():
        prediction = predictions.get(sample_id)
        failure = failures.get(sample_id)
        output = prediction or failure
        if output is not None:
            _validate_join_identity(reference, output)
        result.append(
            OpenLoopJoinedRowV1(
                reference=reference,
                prediction=prediction,
                failure=failure,
            )
        )
    return result


def load_and_join_openloop_artifacts_v1(
    reference_jsonl: str | Path,
    prediction_jsonl: str | Path,
    failure_jsonl: str | Path | None = None,
) -> list[OpenLoopJoinedRowV1]:
    """Load and join one reference/prediction/failure artifact set.

    Args:
        reference_jsonl: Hidden reference JSONL.
        prediction_jsonl: Successful prediction JSONL.
        failure_jsonl: Optional explicit failure JSONL.

    Returns:
        Full-reference joined rows.
    """

    references = load_unique_jsonl_models_v1(
        reference_jsonl,
        OpenLoopReferenceV1,
    )
    predictions = load_unique_jsonl_models_v1(
        prediction_jsonl,
        OpenLoopPredictionV1,
    )
    failures = (
        load_unique_jsonl_models_v1(failure_jsonl, OpenLoopFailureV1)
        if failure_jsonl is not None
        else {}
    )
    return join_openloop_artifacts_v1(references, predictions, failures)


def validate_openloop_package_v1(
    package_manifest_path: str | Path,
    run_manifest_path: str | Path,
) -> OpenLoopPackageArtifactsV1:
    """Validate package/run digests and the model-visible/hidden boundary.

    Args:
        package_manifest_path: Frozen benchmark package manifest JSON.
        run_manifest_path: Frozen model run manifest JSON.

    Returns:
        Validated manifests, paths, and typed row indexes.

    Raises:
        ValueError: If any digest, count, renderer, metric, split, or identity
            gate fails.
    """

    package_path = Path(package_manifest_path).expanduser().resolve()
    run_path = Path(run_manifest_path).expanduser().resolve()
    package = OpenLoopPackageManifestV1.model_validate(
        json.loads(package_path.read_text(encoding="utf-8"))
    )
    run = OpenLoopRunManifestV1.model_validate(
        json.loads(run_path.read_text(encoding="utf-8"))
    )
    input_path = _resolve_artifact_path(package_path.parent, package.input_path)
    reference_path = _resolve_artifact_path(
        package_path.parent,
        package.reference_path,
    )
    if artifact_sha256_v1(input_path) != package.input_digest:
        raise ValueError("benchmark input artifact digest mismatch")
    if artifact_sha256_v1(reference_path) != package.reference_digest:
        raise ValueError("benchmark reference artifact digest mismatch")
    package_digest = artifact_sha256_v1(package_path)
    if run.benchmark_id != package.benchmark_id:
        raise ValueError("run benchmark_id differs from package")
    if run.benchmark_digest != package_digest:
        raise ValueError("run benchmark digest differs from package artifact")
    expected_metric_digest = canonical_row_digest_v1(package.metric_policy)
    if run.metric_policy_digest != expected_metric_digest:
        raise ValueError("run metric-policy digest differs from package")
    if run.renderer_policy != package.renderer_policy:
        raise ValueError("run renderer policy differs from package")
    if run.context_identity.input_manifest_digest != package.input_digest:
        raise ValueError(
            "run context input manifest digest differs from package"
        )
    expected_renderer_digest = canonical_row_digest_v1(package.renderer_policy)

    inputs = load_unique_jsonl_models_v1(input_path, OpenLoopInputV1)
    references = load_unique_jsonl_models_v1(
        reference_path,
        OpenLoopReferenceV1,
    )
    if len(references) != package.reference_count:
        raise ValueError("package reference_count differs from artifact")
    audit_input_reference_identity_v1(inputs, references)
    for row in inputs.values():
        if row.split != package.split:
            raise ValueError(
                f"input split differs from package: {row.sample_id}"
            )
        if row.renderer_policy_digest != expected_renderer_digest:
            raise ValueError(
                "input renderer policy digest differs from package: "
                f"{row.sample_id}"
            )
        if row.stage != run.stage:
            raise ValueError(
                f"input stage differs from run: {row.sample_id}"
            )
    return OpenLoopPackageArtifactsV1(
        package=package,
        run=run,
        inputs=inputs,
        references=references,
        input_path=input_path,
        reference_path=reference_path,
    )


def audit_input_reference_identity_v1(
    inputs: dict[str, OpenLoopInputV1],
    references: dict[str, OpenLoopReferenceV1],
) -> None:
    """Require one-to-one input/reference identity and split metadata.

    Args:
        inputs: Model-visible rows keyed by sample id.
        references: Hidden rows keyed by sample id.

    Raises:
        ValueError: If ids or stable grouping metadata drift.
    """

    if set(inputs) != set(references):
        missing_input = sorted(set(references) - set(inputs))
        missing_reference = sorted(set(inputs) - set(references))
        raise ValueError(
            "input/reference id mismatch: "
            f"missing_input={missing_input[:8]}, "
            f"missing_reference={missing_reference[:8]}"
        )
    for sample_id, input_row in inputs.items():
        reference = references[sample_id]
        pairs = {
            "parent_sample_id": (
                input_row.parent_sample_id,
                reference.parent_sample_id,
            ),
            "source_group_id": (
                input_row.source_group_id,
                reference.source_group_id,
            ),
            "split_group_id": (
                input_row.split_group_id,
                reference.split_group_id,
            ),
            "stage": (input_row.stage, reference.stage),
            "task_type": (input_row.task_type, reference.task_type),
            "dataset_source": (
                input_row.dataset_source,
                reference.dataset_source,
            ),
        }
        drift = {
            field: values
            for field, values in pairs.items()
            if values[0] != values[1]
        }
        if drift:
            raise ValueError(
                f"input/reference identity drift for {sample_id}: {drift}"
            )


def audit_output_input_digests_v1(
    inputs: dict[str, OpenLoopInputV1],
    predictions: dict[str, OpenLoopPredictionV1],
    failures: dict[str, OpenLoopFailureV1] | None = None,
) -> None:
    """Require every attempted output to bind to its exact frozen input.

    Args:
        inputs: Canonical model-visible rows keyed by sample id.
        predictions: Successful outputs keyed by sample id.
        failures: Explicit failures keyed by sample id.

    Raises:
        ValueError: If an output id is unknown or its input digest differs.
    """

    failures = failures or {}
    for sample_id, output in {**predictions, **failures}.items():
        input_row = inputs.get(sample_id)
        if input_row is None:
            raise ValueError(f"output has no frozen input row: {sample_id}")
        expected = canonical_row_digest_v1(input_row)
        if output.input_digest != expected:
            raise ValueError(
                f"output input digest differs from package: {sample_id}"
            )


def audit_checkpoint_exposure_v1(
    inputs: Iterable[OpenLoopInputV1],
    registry: CheckpointExposureRegistryV1,
    *,
    checkpoint_digest: str,
) -> None:
    """Fail when any evaluation parent or group was exposed in training.

    Args:
        inputs: Frozen validation/test input rows.
        registry: Checkpoint-lineage exposure registry.
        checkpoint_digest: Evaluated checkpoint digest.

    Raises:
        ValueError: If registry identity differs or exposure overlaps.
    """

    if registry.checkpoint_digest != checkpoint_digest:
        raise ValueError("exposure registry checkpoint digest mismatch")
    exposed_parents = {row.parent_sample_id for row in registry.entries}
    exposed_groups = {row.split_group_id for row in registry.entries}
    parent_overlap: set[str] = set()
    group_overlap: set[str] = set()
    for row in inputs:
        if row.parent_sample_id in exposed_parents:
            parent_overlap.add(row.parent_sample_id)
        if row.split_group_id in exposed_groups:
            group_overlap.add(row.split_group_id)
    if parent_overlap or group_overlap:
        raise ValueError(
            "checkpoint exposure overlap: "
            f"parents={sorted(parent_overlap)[:8]}, "
            f"groups={sorted(group_overlap)[:8]}"
        )


def artifact_sha256_v1(path: str | Path) -> str:
    """Return the streaming SHA256 digest of one artifact.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_row_digest_v1(value: BaseModel | dict[str, Any]) -> str:
    """Return a stable digest for one JSON-compatible record.

    Args:
        value: Pydantic model or JSON-compatible mapping.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """

    payload = (
        value.model_dump(mode="json")
        if isinstance(value, BaseModel)
        else value
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coverage_metrics_v1(
    rows: Iterable[OpenLoopJoinedRowV1],
) -> dict[str, int | float]:
    """Compute attempt and output coverage on the full reference set.

    Args:
        rows: Joined reference rows.

    Returns:
        Counts and fractions with full-reference denominators.
    """

    materialized = list(rows)
    reference_count = len(materialized)
    attempted = sum(row.attempted for row in materialized)
    predicted = sum(row.prediction is not None for row in materialized)
    explicit_failures = sum(row.failure is not None for row in materialized)
    missing = reference_count - attempted
    denominator = reference_count or 1
    return {
        "reference_count": reference_count,
        "attempted_count": attempted,
        "prediction_count": predicted,
        "explicit_failure_count": explicit_failures,
        "unattempted_count": missing,
        "attempt_coverage": attempted / denominator,
        "output_coverage": predicted / denominator,
    }


def parent_balanced_mean_v1(
    values: Iterable[tuple[str, float]],
) -> float | None:
    """Average rows within parent, then average equally across parents.

    Args:
        values: ``(parent_sample_id, score)`` pairs.

    Returns:
        Parent-balanced mean, or ``None`` for no values.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    for parent_sample_id, value in values:
        if not math.isfinite(float(value)):
            raise ValueError("parent-balanced values must be finite")
        grouped[str(parent_sample_id)].append(float(value))
    if not grouped:
        return None
    parent_means = [sum(items) / len(items) for items in grouped.values()]
    return sum(parent_means) / len(parent_means)


def macro_mean_v1(values: Iterable[float | None]) -> float | None:
    """Return the unweighted mean of defined finite values.

    Args:
        values: Optional scalar metric values.

    Returns:
        Macro mean or ``None`` when every value is undefined.

    Raises:
        ValueError: If a defined value is non-finite.
    """

    materialized = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in materialized):
        raise ValueError("macro values must be finite")
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def grouped_scores_v1(
    rows: Iterable[OpenLoopJoinedRowV1],
    score: Callable[[OpenLoopJoinedRowV1], float],
    key: Callable[[OpenLoopReferenceV1], str],
) -> dict[str, float]:
    """Compute natural-micro scores for deterministic reference groups.

    Args:
        rows: Joined rows.
        score: Per-reference score, including failures as zero.
        key: Group key derived only from the hidden reference.

    Returns:
        Sorted mapping from group identity to mean score.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(score(row))
        if not math.isfinite(value):
            raise ValueError("grouped score must be finite")
        grouped[str(key(row.reference))].append(value)
    return {
        name: sum(values) / len(values)
        for name, values in sorted(grouped.items())
    }


def _validate_join_identity(
    reference: OpenLoopReferenceV1,
    output: OpenLoopPredictionV1 | OpenLoopFailureV1,
) -> None:
    """Reject output rows that drift from their hidden reference identity."""

    fields = ("sample_id", "parent_sample_id", "stage", "task_type")
    drift = {
        field: (getattr(reference, field), getattr(output, field))
        for field in fields
        if getattr(reference, field) != getattr(output, field)
    }
    if drift:
        raise ValueError(
            f"output/reference identity drift for {reference.sample_id}: "
            f"{drift}"
        )


def _resolve_artifact_path(root: Path, value: str) -> Path:
    """Resolve a package-relative path without changing its contents."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


__all__ = [
    "OpenLoopJoinedRowV1",
    "OpenLoopPackageArtifactsV1",
    "artifact_sha256_v1",
    "audit_checkpoint_exposure_v1",
    "audit_input_reference_identity_v1",
    "audit_output_input_digests_v1",
    "canonical_row_digest_v1",
    "coverage_metrics_v1",
    "grouped_scores_v1",
    "join_openloop_artifacts_v1",
    "load_and_join_openloop_artifacts_v1",
    "load_unique_jsonl_models_v1",
    "macro_mean_v1",
    "parent_balanced_mean_v1",
    "validate_openloop_package_v1",
]
