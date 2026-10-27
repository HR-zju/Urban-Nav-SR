#!/usr/bin/env python3
"""Build a portable ModelScope view of the completed history corpus.

The generated media stays in the existing corpus directory.  The upload view
uses hard links for package files on the same filesystem, rewrites training
manifest media paths relative to the upload root, and excludes interrupted or
quarantined artifacts.  No source clip is regenerated.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import urllib.parse
from typing import Any, Iterator


DATASETS = ("egowalk", "sidewalk", "foxglove")
RESIDUAL_SUFFIXES = (".partial", ".corrupt")
HISTORY_PROFILE = "history_4s_2hz_9f_v1"
DEFAULT_QUERY_COUNT = 16


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the input and output roots for one immutable packaging run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and preserve parse errors with the source path."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows while keeping file and line information on errors."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def _write_json(path: Path, value: Any) -> None:
    """Write a deterministic UTF-8 JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    """Write rewritten rows and return their count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _is_residual(path: Path) -> bool:
    """Return whether a path is an interrupted or quarantined artifact."""

    return any(part.endswith(RESIDUAL_SUFFIXES) for part in path.parts)


def _hardlink_tree(source: Path, destination: Path) -> dict[str, int]:
    """Hard-link a package tree while excluding residual artifacts.

    Args:
        source: Completed dataset ``packages`` directory.
        destination: Portable dataset ``packages`` directory.

    Returns:
        File count and byte count for linked files.

    Raises:
        FileExistsError: If the destination already contains an entry.
        OSError: If a source file cannot be linked on the same filesystem.
    """

    files = 0
    bytes_total = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if _is_residual(path):
            continue
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise ValueError(f"unsupported package entry: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        files += 1
        bytes_total += path.stat().st_size
    return {"files": files, "bytes": bytes_total}


def _load_clip_map(dataset_root: Path) -> dict[str, str]:
    """Map materialized clip IDs to their portable package directories."""

    path = dataset_root / "packages/clips.jsonl"
    mapping: dict[str, str] = {}
    for row in _iter_jsonl(path):
        if row.get("status") == "materialized":
            mapping[row["clip_id"]] = row["package_directory"]
    return mapping


def _clip_id(row: dict[str, Any]) -> str | None:
    """Extract the stable source clip ID from a direction row."""

    source_row = row.get("source_row")
    if isinstance(source_row, dict) and source_row.get("clip_id"):
        return str(source_row["clip_id"])
    parent = str(row.get("source_parent_id", ""))
    if ":" in parent:
        parent = parent.split(":", 1)[1]
    return parent.rsplit(":", 1)[0] if ":" in parent else None


def _external_source_uri(dataset: str, path: Path) -> str:
    """Represent an unavailable source path without exposing a local pathname."""

    encoded = urllib.parse.quote(path.as_posix().lstrip("/"), safe="")
    return f"external-source://{dataset}/{encoded}"


def _package_media_path(
    path: Path,
    *,
    dataset: str,
    package_directory: str | None,
    source_dataset_root: Path,
) -> str | None:
    """Resolve an original FPV/BEV path to a packaged media path.

    Only media whose basename is present in the corresponding materialized
    package is mapped.  Sidecars and depth files remain explicit external-source
    references instead of being silently presented as uploaded data.
    """

    if package_directory is None:
        return None
    parts = {part.lower() for part in path.parts}
    if "fpv" in parts:
        media_kind = "fpv"
    elif "bev" in parts or "bev_images" in parts:
        media_kind = "bev"
    else:
        return None
    package = source_dataset_root / "packages" / package_directory
    candidate_names = [path.name]
    if path.name.endswith("_bev.jpg"):
        candidate_names.append(path.name.removesuffix("_bev.jpg") + ".jpg")
    for name in candidate_names:
        candidate = package / "media" / media_kind / name
        if candidate.is_file():
            return (
                Path(dataset) / "packages" / package_directory / "media" / media_kind / name
            ).as_posix()
    return None


def _rewrite_path(
    value: str,
    *,
    dataset: str,
    source_root: Path,
    upload_root: Path,
    package_directory: str | None,
) -> str:
    """Rewrite one absolute path to a portable package path or source URI."""

    if not value.startswith("/"):
        return value
    path = Path(value)
    if path.is_relative_to(source_root):
        return path.relative_to(source_root).as_posix()
    package_path = _package_media_path(
        path,
        dataset=dataset,
        package_directory=package_directory,
        source_dataset_root=source_root / dataset,
    )
    if package_path is not None:
        return package_path
    return _external_source_uri(dataset, path)


def _rewrite_value(
    value: Any,
    *,
    dataset: str,
    source_root: Path,
    upload_root: Path,
    package_directory: str | None,
) -> Any:
    """Recursively rewrite filesystem paths in a manifest or receipt."""

    if isinstance(value, dict):
        return {
            key: _rewrite_value(
                item,
                dataset=dataset,
                source_root=source_root,
                upload_root=upload_root,
                package_directory=package_directory,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_value(
                item,
                dataset=dataset,
                source_root=source_root,
                upload_root=upload_root,
                package_directory=package_directory,
            )
            for item in value
        ]
    if isinstance(value, str):
        return _rewrite_path(
            value,
            dataset=dataset,
            source_root=source_root,
            upload_root=upload_root,
            package_directory=package_directory,
        )
    return value


def _rewrite_manifest(
    source: Path,
    destination: Path,
    *,
    dataset: str,
    source_root: Path,
    upload_root: Path,
    clip_map: dict[str, str],
) -> tuple[int, Counter[str]]:
    """Rewrite one JSONL manifest and count external provenance references."""

    external = Counter()

    def rows() -> Iterator[dict[str, Any]]:
        for row in _iter_jsonl(source):
            clip_id = _clip_id(row)
            package_directory = clip_map.get(clip_id) if clip_id else None
            rewritten = _rewrite_value(
                row,
                dataset=dataset,
                source_root=source_root,
                upload_root=upload_root,
                package_directory=package_directory,
            )
            for value in _walk_strings(rewritten):
                if value.startswith("external-source://"):
                    external["external_source_values"] += 1
            yield rewritten

    return _write_jsonl(destination, rows()), external


def _walk_strings(value: Any) -> Iterator[str]:
    """Yield all string leaves from a JSON-compatible value."""

    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _residual_receipt_entries(root: Path) -> list[dict[str, Any]]:
    """Record every excluded residual path and its byte size."""

    entries = []
    for path in sorted(root.rglob("*")):
        if not _is_residual(path):
            continue
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "kind": "directory" if path.is_dir() else "file",
            "bytes": path.stat().st_size if path.is_file() else 0,
        })
    return entries


def _excluded_clip_rows(dataset_root: Path) -> list[dict[str, Any]]:
    """Read explicit excluded clip rows from the materialization receipt."""

    entries = []
    for row in _iter_jsonl(dataset_root / "packages/clips.jsonl"):
        if row.get("status") == "excluded":
            entries.append({
                key: row[key]
                for key in ("clip_id", "package_directory", "status", "reason")
                if key in row
            })
    return entries


def _history_contract(
    dataset_receipts: dict[str, dict[str, Any]], *, query_count: int
) -> dict[str, Any]:
    """Build the explicit consumer contract from completed generation receipts."""

    return {
        "schema_version": "urbannav_history_modelscope_contract_v1",
        "observation_profile": HISTORY_PROFILE,
        "history_window_seconds": 4.0,
        "history_sample_rate_hz": 2.0,
        "history_slots": 9,
        "history_target_offsets_seconds": [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0],
        "history_selection": "latest source frame at or before each target timestamp",
        "history_padding": False,
        "history_max_lag_ns": sorted({
            lag
            for receipt in dataset_receipts.values()
            for lag in receipt.get("history_max_lag_ns", [])
        }),
        "bev_selection": "latest causal BEV at or before the observation anchor",
        "direction_contract": "ego_degree10_12way_terminal_centroid_v2",
        "stage_ii_query_count": query_count,
        "stage_ii_trajectory_shape": [32, 3],
        "stage_ii_trajectory_frame": "metric FLU cumulative pose",
        "velocity_target": False,
        "path_contract": {
            "manifest_paths": "relative to the ModelScope dataset root",
            "stage_ii_image_root": ".",
            "external_source_paths": "external-source URI; not uploaded media",
        },
        "datasets": list(dataset_receipts),
    }


def _portable_receipt(
    receipt: dict[str, Any],
    *,
    dataset: str,
    source_root: Path,
    upload_root: Path,
) -> dict[str, Any]:
    """Preserve a generation receipt while rewriting its filesystem paths."""

    return _rewrite_value(
        receipt,
        dataset=dataset,
        source_root=source_root,
        upload_root=upload_root,
        package_directory=None,
    )


def package(input_root: Path, output_root: Path, *, query_count: int) -> dict[str, Any]:
    """Create and locally validate the ModelScope upload view.

    Args:
        input_root: Completed ``urbannav_history_4s_2hz_v1`` directory.
        output_root: New portable view directory on the same filesystem.
        query_count: Stage II query count encoded in the aggregate manifest name.

    Returns:
        A machine-readable packaging receipt.

    Raises:
        FileExistsError: If the output directory already exists.
        ValueError: If generation receipts or manifest paths are inconsistent.
        OSError: If a package file cannot be hard-linked.
    """

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if query_count not in (4, 8, 16):
        raise ValueError("query_count must be one of 4, 8, or 16")

    root_receipt = _read_json(input_root / "generation_receipt.json")
    if root_receipt.get("status") != "generated":
        raise ValueError("input root receipt is not generated")
    output_root.mkdir(parents=True)
    receipts: dict[str, dict[str, Any]] = {}
    dataset_results: dict[str, Any] = {}
    manifest_counts: dict[str, dict[str, int]] = {}
    external_counts = Counter()
    try:
        for dataset in DATASETS:
            source_dataset = input_root / dataset
            destination_dataset = output_root / dataset
            receipt = _read_json(source_dataset / "generation_receipt.json")
            if receipt.get("dataset_source") != dataset or receipt.get("status") != "generated":
                raise ValueError(f"{source_dataset}: generation receipt does not match dataset")
            receipts[dataset] = receipt
            package_stats = _hardlink_tree(
                source_dataset / "packages",
                destination_dataset / "packages",
            )
            clip_map = _load_clip_map(source_dataset)
            aggregate_stage_ii = source_dataset / (
                f"manifests/{dataset}_stage_ii_direct_action_q{query_count}.jsonl"
            )
            if not aggregate_stage_ii.is_file():
                raise FileNotFoundError(aggregate_stage_ii)
            stage_ii_destination = destination_dataset / "manifests" / aggregate_stage_ii.name
            stage_ii_rows, counts = _rewrite_manifest(
                aggregate_stage_ii,
                stage_ii_destination,
                dataset=dataset,
                source_root=input_root,
                upload_root=output_root,
                clip_map=clip_map,
            )
            external_counts.update(counts)

            direction_source = source_dataset / "manifests" / (
                f"{dataset}_observations_direction12.jsonl"
            )
            if not direction_source.is_file():
                direction_candidates = sorted(
                    source_dataset.glob("manifests/*_observations_direction12.jsonl")
                )
                if len(direction_candidates) != 1:
                    raise FileNotFoundError(
                        f"{source_dataset}: expected one direction manifest, "
                        f"found {direction_candidates}"
                    )
                direction_source = direction_candidates[0]
            direction_destination = destination_dataset / "manifests" / (
                f"{dataset}_observations_direction12.jsonl"
            )
            direction_rows, counts = _rewrite_manifest(
                direction_source,
                direction_destination,
                dataset=dataset,
                source_root=input_root,
                upload_root=output_root,
                clip_map=clip_map,
            )
            external_counts.update(counts)
            manifest_counts[dataset] = {
                "stage_ii_rows": stage_ii_rows,
                "direction_rows": direction_rows,
            }
            dataset_results[dataset] = {
                "package_files": package_stats["files"],
                "package_bytes": package_stats["bytes"],
                "stage_ii_manifest": stage_ii_destination.relative_to(output_root).as_posix(),
                "direction_manifest": direction_destination.relative_to(output_root).as_posix(),
                **manifest_counts[dataset],
            }

        portable_receipts = {
            dataset: _portable_receipt(
                receipts[dataset],
                dataset=dataset,
                source_root=input_root,
                upload_root=output_root,
            )
            for dataset in DATASETS
        }
        _write_json(output_root / "metadata/generation_receipt.json", {
            "status": "generated",
            "dataset_order": list(DATASETS),
            "datasets": portable_receipts,
            "path_mode": "portable_relative_manifest_paths",
        })
        _write_json(
            output_root / "metadata/history_contract.json",
            _history_contract(receipts, query_count=query_count),
        )
        _write_json(output_root / "metadata/exclusions.json", {
            "schema_version": "urbannav_history_modelscope_exclusions_v1",
            "residual_rule": "exclude every path containing .partial or .corrupt",
            "residual_paths": _residual_receipt_entries(input_root),
            "excluded_clips": {
                dataset: _excluded_clip_rows(input_root / dataset)
                for dataset in DATASETS
            },
        })
        packaging_receipt = {
            "schema_version": "urbannav_history_modelscope_package_v1",
            "status": "portable_view_ready",
            "source_generation_status": root_receipt.get("status"),
            "path_mode": "relative_to_dataset_root",
            "input_root_name": input_root.name,
            "output_root_name": output_root.name,
            "datasets": dataset_results,
            "manifest_counts": manifest_counts,
            "external_source_uri_count": external_counts["external_source_values"],
            "excluded_residual_count": len(_residual_receipt_entries(input_root)),
        }
        _write_json(output_root / "metadata/portable_manifest_receipt.json", packaging_receipt)
        (output_root / "README.md").write_text(
            "# UrbanNav history 4s/2Hz portable dataset\n\n"
            "This view is the ModelScope upload root. Training manifest `images` values "
            "are relative to this directory. The package media are hard links to the "
            "completed local corpus and `.partial`/`.corrupt` artifacts are excluded.\n\n"
            "Read `metadata/history_contract.json`, `metadata/generation_receipt.json`, "
            "and `metadata/exclusions.json` before starting a consumer run.\n",
            encoding="utf-8",
        )
        _validate_portable_view(output_root, query_count=query_count)
        return packaging_receipt
    except BaseException:
        # An incomplete view must never look uploadable to a later run.
        import shutil

        shutil.rmtree(output_root)
        raise


def _validate_portable_view(output_root: Path, *, query_count: int) -> None:
    """Run the local portable-manifest smoke and fail on unresolved media."""

    for dataset in DATASETS:
        manifest = output_root / dataset / "manifests" / (
            f"{dataset}_stage_ii_direct_action_q{query_count}.jsonl"
        )
        for row in _iter_jsonl(manifest):
            images = row.get("images")
            if not isinstance(images, list) or not images:
                raise ValueError(f"{manifest}: row has no images")
            for image in images:
                path = Path(image)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"{manifest}: non-portable image path {image}")
                resolved = output_root / path
                if not resolved.is_file():
                    raise FileNotFoundError(f"{manifest}: missing packaged image {resolved}")
        for manifest in sorted((output_root / dataset / "manifests").glob("*.jsonl")):
            for row in _iter_jsonl(manifest):
                for value in _walk_strings(row):
                    if value.startswith("/"):
                        raise ValueError(f"{manifest}: absolute path remains: {value}")
    for path in output_root.rglob("*"):
        if _is_residual(path):
            raise ValueError(f"residual artifact entered portable view: {path}")


def main(argv: list[str] | None = None) -> None:
    """Build and validate the portable view, then print its receipt."""

    args = _parse_args(argv)
    result = package(args.input_root, args.output_root, query_count=args.query_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
