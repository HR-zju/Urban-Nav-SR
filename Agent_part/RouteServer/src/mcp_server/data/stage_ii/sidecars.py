"""Build exact Stage II trajectory sidecars for frozen Stage I splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV1,
    parse_stage_i_training_view_v1,
)
from mcp_server.data.stage_ii.sources import (
    StageIISourceAdapterErrorV1,
    adapt_stage_ii_trajectory_source_row_v1,
    load_stage_ii_trajectory_source_registry_v1,
    stage_ii_source_row_identity_v1,
)


STAGE_II_SIDECAR_BUILD_REPORT_VERSION = (
    "urbannav_stage_ii_trajectory_sidecar_build_report_v1"
)


def build_stage_ii_trajectory_sidecars_v1(
    *,
    stage_i_jsonl: str | Path,
    split_manifests: Mapping[str, str | Path],
    dataset_manifests: Mapping[str, Sequence[str | Path]],
    registry_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Join real trajectory rows to all FullCoT parents in explicit splits.

    Split membership comes only from the supplied lineage-safe Stage I
    manifests.  The source registry controls dataset identity and trajectory
    semantics, but it is not allowed to re-split samples during this join.

    Args:
        stage_i_jsonl: Canonical Stage I exchange views containing FullCoT rows.
        split_manifests: Explicit ``train`` and ``validation`` physical views.
        dataset_manifests: Dataset family to one or more raw source manifests.
        registry_json: Frozen Stage II trajectory source registry.
        output_dir: Destination for per-split sidecars, drops, and report.

    Returns:
        Deterministic build report with exact coverage and source evidence.

    Raises:
        ValueError: If splits overlap, identities drift, sources duplicate, or
            required dataset manifests are missing.
        FileNotFoundError: If an input artifact does not exist.
    """

    if set(split_manifests) != {"train", "validation"}:
        raise ValueError("split_manifests must contain exactly train and validation")
    registry = load_stage_ii_trajectory_source_registry_v1(registry_json)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    split_ids, split_counts = _load_split_full_cot_ids(split_manifests)
    selected_views, canonical_counts = _load_selected_full_cot_views(
        stage_i_jsonl,
        split_ids=split_ids,
    )
    source_to_stage_i: dict[tuple[str, str], tuple[str, str]] = {}
    for split, views in selected_views.items():
        for sample_id, view in views.items():
            key = (str(view.dataset_source), str(view.source_group_id))
            if key in source_to_stage_i:
                raise ValueError(f"duplicate FullCoT source identity: {key}")
            source_to_stage_i[key] = (split, sample_id)

    required_datasets = {dataset for dataset, _source_id in source_to_stage_i}
    supplied_datasets = set(dataset_manifests)
    if required_datasets - supplied_datasets:
        raise ValueError(
            "missing raw manifests for required datasets: "
            f"{sorted(required_datasets - supplied_datasets)}"
        )
    unknown_datasets = supplied_datasets - {
        str(spec.dataset_source) for spec in registry.datasets
    }
    if unknown_datasets:
        raise ValueError(f"unknown dataset manifest families: {sorted(unknown_datasets)}")

    sidecars: dict[str, dict[str, dict[str, Any]]] = {
        split: {} for split in split_ids
    }
    drops: list[dict[str, Any]] = []
    scan_reports: dict[str, Any] = {}
    for dataset in sorted(required_datasets):
        spec = registry.dataset(dataset)
        if spec.status != "enabled":
            raise ValueError(f"required dataset is disabled by registry: {dataset}")
        manifests = [Path(path).expanduser() for path in dataset_manifests[dataset]]
        if not manifests:
            raise ValueError(f"dataset={dataset} has no source manifests")
        scan_reports[dataset] = _scan_dataset_manifests(
            manifests,
            spec=spec,
            source_to_stage_i=source_to_stage_i,
            sidecars=sidecars,
            drops=drops,
        )

    missing: list[dict[str, Any]] = []
    for (dataset, source_group_id), (split, sample_id) in source_to_stage_i.items():
        if sample_id in sidecars[split]:
            continue
        missing.append({
            "stage_i_sample_id": sample_id,
            "source_group_id": source_group_id,
            "dataset_source": dataset,
            "split": split,
            "reason_code": "trajectory_source_row_missing",
        })
    drops.extend(missing)

    outputs: dict[str, str] = {}
    for split in ("train", "validation"):
        stage_i_path = destination / f"stage_i_full_cot_{split}.jsonl"
        _write_jsonl(
            stage_i_path,
            [
                selected_views[split][sample_id].model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for sample_id in sorted(selected_views[split])
            ],
        )
        path = destination / f"trajectory_sidecars_{split}.jsonl"
        rows = [sidecars[split][sample_id] for sample_id in sorted(sidecars[split])]
        _write_jsonl(path, rows)
        outputs[f"{split}_stage_i_jsonl"] = str(stage_i_path)
        outputs[f"{split}_sidecars_jsonl"] = str(path)
    drop_path = destination / "trajectory_sidecar_drops.jsonl"
    _write_jsonl(drop_path, drops)
    outputs["drops_jsonl"] = str(drop_path)
    report_path = destination / "trajectory_sidecar_build_report.json"
    outputs["report_json"] = str(report_path)

    expected = sum(len(values) for values in split_ids.values())
    accepted = sum(len(values) for values in sidecars.values())
    report = {
        "schema_version": STAGE_II_SIDECAR_BUILD_REPORT_VERSION,
        "stage_i_jsonl": str(Path(stage_i_jsonl).expanduser()),
        "registry": {
            "path": str(Path(registry_json).expanduser()),
            "registry_id": registry.registry_id,
            "registry_version": registry.registry_version,
            "registry_digest": registry.registry_digest,
        },
        "split_authority": "explicit_stage_i_lineage_safe_manifests",
        "split_counts": split_counts,
        "canonical_counts": canonical_counts,
        "dataset_scan": scan_reports,
        "expected_full_cot_parents": expected,
        "accepted_sidecars": accepted,
        "drop_count": len(drops),
        "complete": accepted == expected and not drops,
        "outputs": outputs,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_split_full_cot_ids(
    split_manifests: Mapping[str, str | Path],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Load exact FullCoT IDs and reject cross-split overlap."""

    split_ids: dict[str, set[str]] = {}
    counts: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for split, raw_path in split_manifests.items():
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        ids: set[str] = set()
        rows = 0
        non_full_cot = 0
        for row in _iter_jsonl(path):
            rows += 1
            if str(row.get("task_type")) != "full_cot":
                non_full_cot += 1
                continue
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in ids:
                raise ValueError(f"missing or duplicate {split} FullCoT sample_id: {sample_id!r}")
            if sample_id in owner:
                raise ValueError(
                    f"FullCoT sample appears in both {owner[sample_id]} and {split}: {sample_id}"
                )
            owner[sample_id] = split
            ids.add(sample_id)
        split_ids[split] = ids
        counts[split] = {
            "physical_rows": rows,
            "full_cot_rows": len(ids),
            "non_full_cot_rows": non_full_cot,
            "manifest": str(path),
        }
    return split_ids, counts


def _load_selected_full_cot_views(
    stage_i_jsonl: str | Path,
    *,
    split_ids: Mapping[str, set[str]],
) -> tuple[dict[str, dict[str, StageIFullCoTViewV1]], dict[str, int]]:
    """Read canonical views once and retain exact selected FullCoT parents."""

    path = Path(stage_i_jsonl).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    sample_to_split = {
        sample_id: split
        for split, values in split_ids.items()
        for sample_id in values
    }
    selected: dict[str, dict[str, StageIFullCoTViewV1]] = {
        split: {} for split in split_ids
    }
    counts: Counter[str] = Counter()
    for payload in _iter_jsonl(path):
        counts["input_rows"] += 1
        sample_id = str(payload.get("sample_id") or "")
        split = sample_to_split.get(sample_id)
        if split is None:
            continue
        typed = parse_stage_i_training_view_v1(payload)
        if not isinstance(typed, StageIFullCoTViewV1):
            raise ValueError(f"selected split ID is not a FullCoT view: {sample_id}")
        if sample_id in selected[split]:
            raise ValueError(f"duplicate canonical FullCoT sample: {sample_id}")
        selected[split][sample_id] = typed
        counts[f"selected:{split}"] += 1
    missing = sorted(set(sample_to_split) - {
        sample_id for values in selected.values() for sample_id in values
    })
    if missing:
        raise ValueError(f"split manifests reference missing canonical FullCoT views: {missing[:8]}")
    return selected, dict(counts)


def _scan_dataset_manifests(
    manifests: Sequence[Path],
    *,
    spec,
    source_to_stage_i: Mapping[tuple[str, str], tuple[str, str]],
    sidecars: Mapping[str, dict[str, dict[str, Any]]],
    drops: list[dict[str, Any]],
) -> dict[str, Any]:
    """Scan one dataset family and retain only requested exact identities."""

    counts: Counter[str] = Counter()
    evidence: list[dict[str, Any]] = []
    duplicate_examples: list[dict[str, Any]] = []
    for path in manifests:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line.encode("utf-8"))
                if not line.strip():
                    continue
                counts["scanned_rows"] += 1
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise StageIISourceAdapterErrorV1(
                            "trajectory_row_not_object",
                            "source JSONL row must be an object",
                        )
                    source_group_id, _split_group_id = stage_ii_source_row_identity_v1(raw, spec)
                except (json.JSONDecodeError, StageIISourceAdapterErrorV1) as exc:
                    counts[f"ignored_invalid:{getattr(exc, 'code', 'trajectory_json_invalid')}"] += 1
                    continue
                target = source_to_stage_i.get((str(spec.dataset_source), source_group_id))
                if target is None:
                    continue
                split, stage_i_sample_id = target
                try:
                    adapted = adapt_stage_ii_trajectory_source_row_v1(raw, spec)
                except StageIISourceAdapterErrorV1 as exc:
                    if stage_i_sample_id in sidecars[split]:
                        raise ValueError(
                            "duplicate matched trajectory source row is invalid: "
                            f"{source_group_id}: {exc.code}"
                        ) from exc
                    drops.append({
                        "stage_i_sample_id": stage_i_sample_id,
                        "source_group_id": source_group_id,
                        "dataset_source": str(spec.dataset_source),
                        "split": split,
                        "reason_code": exc.code,
                        "detail": str(exc)[:1000],
                        "source_manifest": str(path),
                        "source_line": line_number,
                    })
                    counts[f"matched_drop:{exc.code}"] += 1
                    continue
                if adapted.sample_id != source_group_id:
                    raise ValueError(
                        f"adapted source identity drift: {adapted.sample_id} != {source_group_id}"
                    )
                candidate = {
                    "stage_i_sample_id": stage_i_sample_id,
                    "trajectory_ego_t": [list(row) for row in adapted.raw_trajectory],
                    "action_mask": list(adapted.action_mask),
                    "source_format": adapted.source_format,
                    "trajectory_source": adapted.trajectory_source,
                    "yaw_source": adapted.yaw_source,
                    "review_status": "not_required",
                    "source_manifest": str(path),
                    "source_line": line_number,
                    "source_parent_id": adapted.sample_id,
                    "mask_provenance": adapted.mask_provenance,
                    "origin_provenance": adapted.origin_provenance,
                }
                existing = sidecars[split].get(stage_i_sample_id)
                if existing is not None:
                    if _trajectory_sidecar_semantics(existing) != _trajectory_sidecar_semantics(candidate):
                        raise ValueError(
                            "conflicting duplicate matched trajectory source row: "
                            f"{source_group_id}"
                        )
                    counts["matched_duplicate_equivalent"] += 1
                    if len(duplicate_examples) < 20:
                        duplicate_examples.append({
                            "source_group_id": source_group_id,
                            "kept_manifest": existing["source_manifest"],
                            "kept_line": existing["source_line"],
                            "duplicate_manifest": str(path),
                            "duplicate_line": line_number,
                        })
                    continue
                sidecars[split][stage_i_sample_id] = candidate
                counts[f"matched:{split}"] += 1
        evidence.append({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return {
        "dataset_source": str(spec.dataset_source),
        "counts": dict(counts),
        "manifests": evidence,
        "equivalent_duplicate_examples": duplicate_examples,
    }


def _trajectory_sidecar_semantics(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must agree across duplicate source manifests."""

    return {
        key: sidecar[key]
        for key in (
            "stage_i_sample_id",
            "trajectory_ego_t",
            "action_mask",
            "source_format",
            "trajectory_source",
            "yaw_source",
            "review_status",
            "source_parent_id",
            "mask_provenance",
            "origin_provenance",
        )
    }


def _iter_jsonl(path: str | Path):
    """Yield non-empty JSON objects from one JSONL artifact."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic compact JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


__all__ = [
    "STAGE_II_SIDECAR_BUILD_REPORT_VERSION",
    "build_stage_ii_trajectory_sidecars_v1",
]
