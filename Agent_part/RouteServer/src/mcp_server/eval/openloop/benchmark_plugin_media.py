"""Portable verified media manifests for frozen benchmark-plugin inputs.

The benchmark input keeps its original logical image string so its canonical
digest remains stable.  This module binds that string to a safe relative media
path and exact file bytes, allowing a model runner to relocate public media
without rewriting its prompt/input contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkMediaManifestEntryV1,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic


def build_benchmark_media_manifest_v1(
    *,
    inputs: Iterable[BenchmarkInputV1],
    source_root: str | Path,
    output_jsonl: str | Path,
) -> dict[str, object]:
    """Write one byte-verified media manifest for frozen visible inputs.

    Args:
        inputs: Frozen benchmark inputs whose ordered logical paths are bound.
        source_root: Existing source directory below which every image resides.
        output_jsonl: Destination for deterministic unique manifest entries.

    Returns:
        Manifest path, digest, entry count, and byte total.

    Raises:
        FileNotFoundError: If a visible image is absent or not a regular file.
        ValueError: If an image escapes ``source_root`` or no images are found.
    """

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'benchmark media source root is missing: {root}')
    logical_paths = sorted({image for input_row in inputs for image in input_row.images})
    if not logical_paths:
        raise ValueError('benchmark media manifest requires at least one visible image')
    entries = [_build_media_entry_v1(logical_path, source_root=root) for logical_path in logical_paths]
    destination = Path(output_jsonl).expanduser().resolve()
    write_jsonl_atomic(destination, (entry.model_dump(mode='json') for entry in entries))
    return {
        'schema_version': 'urbannav_benchmark_media_manifest_build_report_v1',
        'media_manifest_path': str(destination),
        'media_manifest_digest': artifact_sha256_v1(destination),
        'media_manifest_count': len(entries),
        'total_byte_size': sum(entry.byte_size for entry in entries),
    }


def load_benchmark_media_manifest_v1(
    path: str | Path,
) -> dict[str, BenchmarkMediaManifestEntryV1]:
    """Load one strict manifest keyed by its frozen logical image path.

    Args:
        path: Existing manifest JSONL path.

    Returns:
        Unique logical-path to media-entry mapping.

    Raises:
        ValueError: If logical or relative paths duplicate.
    """

    entries: dict[str, BenchmarkMediaManifestEntryV1] = {}
    relative_paths: set[str] = set()
    for payload in iter_jsonl(path):
        entry = BenchmarkMediaManifestEntryV1.model_validate(payload)
        if entry.logical_path in entries:
            raise ValueError(f'duplicate media logical_path: {entry.logical_path}')
        if entry.relative_path in relative_paths:
            raise ValueError(f'duplicate media relative_path: {entry.relative_path}')
        entries[entry.logical_path] = entry
        relative_paths.add(entry.relative_path)
    if not entries:
        raise ValueError('benchmark media manifest is empty')
    return entries


def audit_benchmark_media_manifest_coverage_v1(
    *,
    inputs: Iterable[BenchmarkInputV1],
    entries: Mapping[str, BenchmarkMediaManifestEntryV1],
) -> None:
    """Require an exact manifest binding for every visible image reference.

    Args:
        inputs: Frozen benchmark inputs.
        entries: Logical-path keyed media entries.

    Returns:
        ``None`` after exact coverage validation.

    Raises:
        ValueError: If entries are missing or not referenced by frozen inputs.
    """

    expected = {image for input_row in inputs for image in input_row.images}
    observed = set(entries)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            'benchmark media manifest coverage mismatch: '
            f'missing={len(missing)} {missing[:3]}, unexpected={len(unexpected)} {unexpected[:3]}')


def resolve_benchmark_media_path_v1(
    entry: BenchmarkMediaManifestEntryV1,
    *,
    media_root: str | Path,
) -> Path:
    """Resolve one safe manifest relative path below a supplied media root.

    Args:
        entry: Strict logical-to-relative media binding.
        media_root: Read-only root containing the relocated media tree.

    Returns:
        Resolved path below ``media_root``.  The path may be checked for bytes
        separately so callers can choose an appropriate preflight scope.

    Raises:
        ValueError: If a path escapes the resolved runtime media root.
    """

    root = Path(media_root).expanduser().resolve()
    candidate = (root / entry.relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'media path escapes runtime root: {entry.relative_path}') from exc
    return candidate


def artifact_sha256_v1(path: str | Path) -> str:
    """Return SHA256 of one media or manifest artifact without loading it all.

    Args:
        path: Existing regular file.

    Returns:
        Lowercase SHA256 hex digest.
    """

    digest = hashlib.sha256()
    with Path(path).expanduser().open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _build_media_entry_v1(
    logical_path: str,
    *,
    source_root: Path,
) -> BenchmarkMediaManifestEntryV1:
    """Bind one existing logical source image to its portable relative path."""

    source = Path(logical_path).expanduser().resolve()
    try:
        relative_path = source.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise ValueError(f'benchmark media source escapes root: {logical_path}') from exc
    if not source.is_file():
        raise FileNotFoundError(f'benchmark media file is missing: {source}')
    return BenchmarkMediaManifestEntryV1(
        logical_path=logical_path,
        relative_path=relative_path,
        byte_size=source.stat().st_size,
        sha256=artifact_sha256_v1(source),
    )


__all__ = [
    'artifact_sha256_v1',
    'audit_benchmark_media_manifest_coverage_v1',
    'build_benchmark_media_manifest_v1',
    'load_benchmark_media_manifest_v1',
    'resolve_benchmark_media_path_v1',
]
