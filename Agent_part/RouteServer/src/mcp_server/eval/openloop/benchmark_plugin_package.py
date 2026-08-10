"""Immutable package IO and full-denominator joins for benchmark plugins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkFailureV1,
    BenchmarkInputV1,
    BenchmarkPackageManifestV1,
    BenchmarkPredictionV1,
    BenchmarkProfileV1,
    BenchmarkReferenceV1,
    benchmark_plugin_digest_v1,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkJoinedRowV1
from mcp_server.eval.openloop.benchmark_plugin_media import (
    audit_benchmark_media_manifest_coverage_v1,
    load_benchmark_media_manifest_v1,
)


T = TypeVar("T", bound=BaseModel)


def artifact_sha256_v1(path: str | Path) -> str:
    """Return SHA256 for one immutable benchmark artifact.

    Args:
        path: Existing file path.

    Returns:
        Lowercase SHA256 digest.
    """

    import hashlib

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_benchmark_package_manifest_v1(
    *,
    package_id: str,
    profile: BenchmarkProfileV1,
    split: str,
    input_jsonl: str | Path,
    reference_jsonl: str | Path,
    output_json: str | Path,
    media_manifest_jsonl: str | Path | None = None,
    source_fingerprints: dict[str, str] | None = None,
) -> BenchmarkPackageManifestV1:
    """Validate a visible/hidden pair and freeze one package manifest.

    Args:
        package_id: Human-readable immutable package identity.
        profile: Legal dataset/input/metric profile.
        split: Expected package split.
        input_jsonl: Model-visible JSONL artifact.
        reference_jsonl: Hidden reference JSONL artifact.
        output_json: Manifest destination.
        media_manifest_jsonl: Optional strict logical-path to media-byte
            manifest.  When supplied it is bound into the package and must
            cover every input image exactly.
        source_fingerprints: Optional source artifact evidence.

    Returns:
        Validated package manifest.

    Raises:
        ValueError: If identities, profile, split, or package boundary drift.
    """

    input_path = Path(input_jsonl).expanduser().resolve()
    reference_path = Path(reference_jsonl).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    inputs = load_unique_models_v1(input_path, BenchmarkInputV1)
    references = load_unique_models_v1(reference_path, BenchmarkReferenceV1)
    _audit_input_reference_identity(inputs, references)
    for input_row in inputs.values():
        if input_row.split != split:
            raise ValueError(f"input split differs from package: {input_row.sample_id}")
        if input_row.benchmark_profile_id != profile.profile_id:
            raise ValueError(f"input profile differs from package: {input_row.sample_id}")
        if input_row.dataset_plugin_id != profile.dataset_plugin_id:
            raise ValueError(f"input dataset plugin differs from profile: {input_row.sample_id}")
    for reference in references.values():
        if reference.split != split:
            raise ValueError(f"reference split differs from package: {reference.sample_id}")
        if reference.benchmark_profile_id != profile.profile_id:
            raise ValueError(f"reference profile differs from package: {reference.sample_id}")
        if reference.dataset_plugin_id != profile.dataset_plugin_id:
            raise ValueError(f"reference dataset plugin differs from profile: {reference.sample_id}")
    media_manifest_path: Path | None = None
    media_manifest_digest: str | None = None
    media_manifest_count: int | None = None
    if media_manifest_jsonl is not None:
        media_manifest_path = Path(media_manifest_jsonl).expanduser().resolve()
        entries = load_benchmark_media_manifest_v1(media_manifest_path)
        audit_benchmark_media_manifest_coverage_v1(
            inputs=inputs.values(),
            entries=entries,
        )
        media_manifest_digest = artifact_sha256_v1(media_manifest_path)
        media_manifest_count = len(entries)
    payload = BenchmarkPackageManifestV1(
        package_id=package_id,
        benchmark_profile=profile,
        split=split,
        input_path=_relative_path(output_path.parent, input_path),
        reference_path=_relative_path(output_path.parent, reference_path),
        input_digest=artifact_sha256_v1(input_path),
        reference_digest=artifact_sha256_v1(reference_path),
        input_count=len(inputs),
        reference_count=len(references),
        media_manifest_path=(None if media_manifest_path is None else
                             _relative_path(output_path.parent,
                                            media_manifest_path)),
        media_manifest_digest=media_manifest_digest,
        media_manifest_count=media_manifest_count,
        source_fingerprints=dict(source_fingerprints or {}),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_benchmark_package_v1(
    package_json: str | Path,
) -> tuple[BenchmarkPackageManifestV1, dict[str, BenchmarkInputV1], dict[str, BenchmarkReferenceV1]]:
    """Load and verify one immutable benchmark package.

    Args:
        package_json: Package manifest path.

    Returns:
        Validated manifest, visible inputs, and hidden references.

    Raises:
        ValueError: If artifact digest, count, or input/reference identity drifts.
    """

    package_path = Path(package_json).expanduser().resolve()
    package = BenchmarkPackageManifestV1.model_validate_json(package_path.read_text(encoding="utf-8"))
    input_path = _resolve_relative(package_path.parent, package.input_path)
    reference_path = _resolve_relative(package_path.parent, package.reference_path)
    if artifact_sha256_v1(input_path) != package.input_digest:
        raise ValueError("benchmark plugin input digest mismatch")
    if artifact_sha256_v1(reference_path) != package.reference_digest:
        raise ValueError("benchmark plugin reference digest mismatch")
    inputs = load_unique_models_v1(input_path, BenchmarkInputV1)
    references = load_unique_models_v1(reference_path, BenchmarkReferenceV1)
    if len(inputs) != package.input_count or len(references) != package.reference_count:
        raise ValueError("benchmark plugin package count differs from artifacts")
    _audit_input_reference_identity(inputs, references)
    if package.media_manifest_path is not None:
        assert package.media_manifest_digest is not None
        assert package.media_manifest_count is not None
        media_manifest_path = _resolve_relative(package_path.parent,
                                                package.media_manifest_path)
        if artifact_sha256_v1(media_manifest_path) != package.media_manifest_digest:
            raise ValueError("benchmark plugin media manifest digest mismatch")
        entries = load_benchmark_media_manifest_v1(media_manifest_path)
        if len(entries) != package.media_manifest_count:
            raise ValueError("benchmark plugin media manifest count differs from artifact")
        audit_benchmark_media_manifest_coverage_v1(
            inputs=inputs.values(),
            entries=entries,
        )
    return package, inputs, references


def load_unique_models_v1(path: str | Path, model_type: type[T]) -> dict[str, T]:
    """Load strict JSONL records keyed by their unique ``sample_id``.

    Args:
        path: JSONL artifact path.
        model_type: Pydantic model used for strict validation.

    Returns:
        Sample-id mapping preserving artifact order.
    """

    rows: dict[str, T] = {}
    for payload in iter_jsonl(path):
        row = model_type.model_validate(payload)
        sample_id = str(getattr(row, "sample_id"))
        if sample_id in rows:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        rows[sample_id] = row
    return rows


def write_benchmark_rows_v1(path: str | Path, rows: list[BaseModel]) -> None:
    """Write typed benchmark rows as deterministic JSONL.

    Args:
        path: Output JSONL destination.
        rows: Typed records in deterministic order.
    """

    write_jsonl_atomic(path, (row.model_dump(mode="json", exclude_none=True) for row in rows))


def join_benchmark_artifacts_v1(
    references: dict[str, BenchmarkReferenceV1],
    predictions: dict[str, BenchmarkPredictionV1],
    failures: dict[str, BenchmarkFailureV1] | None = None,
) -> list[BenchmarkJoinedRowV1]:
    """Join successful/failure output ledgers to the full reference denominator.

    Args:
        references: Hidden reference rows keyed by sample id.
        predictions: Successful predictions keyed by sample id.
        failures: Optional explicit failures keyed by sample id.

    Returns:
        Rows in frozen reference order, including missing attempts.
    """

    failures = failures or {}
    output_ids = set(predictions) | set(failures)
    overlap = set(predictions) & set(failures)
    extra = output_ids - set(references)
    if overlap:
        raise ValueError("sample ids appear in both predictions and failures: " + ",".join(sorted(overlap)[:8]))
    if extra:
        raise ValueError("output contains unknown sample ids: " + ",".join(sorted(extra)[:8]))
    joined: list[BenchmarkJoinedRowV1] = []
    for sample_id, reference in references.items():
        prediction = predictions.get(sample_id)
        failure = failures.get(sample_id)
        if prediction is not None:
            if prediction.parent_sample_id != reference.parent_sample_id:
                raise ValueError(f"prediction parent drift: {sample_id}")
            if prediction.benchmark_profile_id != reference.benchmark_profile_id:
                raise ValueError(f"prediction profile drift: {sample_id}")
        if failure is not None:
            if failure.parent_sample_id != reference.parent_sample_id:
                raise ValueError(f"failure parent drift: {sample_id}")
            if failure.benchmark_profile_id != reference.benchmark_profile_id:
                raise ValueError(f"failure profile drift: {sample_id}")
        joined.append(
            BenchmarkJoinedRowV1(
                reference=reference,
                prediction=prediction,
                failure=failure.model_dump(mode="json") if failure is not None else None,
            )
        )
    return joined


def input_digest_v1(input_row: BenchmarkInputV1) -> str:
    """Return the canonical digest of one exact model-visible input.

    Args:
        input_row: Visible frozen benchmark case.

    Returns:
        SHA256 digest used by prediction/failure ledgers.
    """

    return benchmark_plugin_digest_v1(input_row)


def _audit_input_reference_identity(
    inputs: dict[str, BenchmarkInputV1],
    references: dict[str, BenchmarkReferenceV1],
) -> None:
    """Require exact visible/hidden row identity equality."""

    if set(inputs) != set(references):
        raise ValueError("benchmark plugin input/reference sample_id mismatch")
    for sample_id, input_row in inputs.items():
        reference = references[sample_id]
        fields = (
            "parent_sample_id",
            "source_group_id",
            "split_group_id",
            "split",
            "dataset_plugin_id",
            "benchmark_profile_id",
            "dataset_source",
        )
        drift = [name for name in fields if getattr(input_row, name) != getattr(reference, name)]
        if drift:
            raise ValueError(f"benchmark input/reference identity drift for {sample_id}: {drift}")


def _relative_path(parent: Path, path: Path) -> str:
    """Use a portable relative path where possible."""

    try:
        return path.relative_to(parent).as_posix()
    except ValueError:
        return str(path)


def _resolve_relative(parent: Path, value: str) -> Path:
    """Resolve a manifest-owned path without escaping through a relative link."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


__all__ = [
    "artifact_sha256_v1",
    "build_benchmark_package_manifest_v1",
    "input_digest_v1",
    "join_benchmark_artifacts_v1",
    "load_benchmark_package_v1",
    "load_unique_models_v1",
    "write_benchmark_rows_v1",
]
