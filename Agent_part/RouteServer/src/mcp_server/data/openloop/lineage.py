"""Build immutable split lineage and freeze canonical benchmark views."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mcp_server.contracts.openloop_benchmark import (
    OpenLoopSplitLineageEntryV1,
)
from mcp_server.contracts.stage_i_sft import parse_stage_i_training_view_v1
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from mcp_server.eval.openloop.openloop_common import artifact_sha256_v1


OPENLOOP_SPLIT_SOURCE_SPEC_VERSION = (
    "urbannav_openloop_split_source_spec_v1"
)
OPENLOOP_SPLIT_LINEAGE_REPORT_VERSION = (
    "urbannav_openloop_split_lineage_report_v2_0"
)
OPENLOOP_FROZEN_VIEW_REPORT_VERSION = (
    "urbannav_openloop_frozen_view_report_v2_0"
)


def build_openloop_split_lineage_v1(
    spec_json: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    *,
    quarantine_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Build canonical lineage from explicitly mapped split manifests.

    The source specification must list every manifest and explicitly name its
    split, parent, split-group, and dataset fields. Nested selectors use dotted
    paths. Dataset aliases must also be explicit; no dataset identity is
    inferred from paths or row IDs.

    Args:
        spec_json: Versioned mapping specification.
        output_jsonl: Canonical lineage-row destination.
        report_json: Audit-report destination.
        quarantine_jsonl: Required quarantine destination when the explicit
            overlap policy permits conservative removal.

    Returns:
        Deterministic lineage report.

    Raises:
        ValueError: If the mapping is incomplete, identities conflict, or one
            canonical group occurs in more than one split.
    """

    spec_path = Path(spec_json).expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != OPENLOOP_SPLIT_SOURCE_SPEC_VERSION:
        raise ValueError("unsupported split source specification version")
    manifests = spec.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("split source specification requires manifests")

    entries: dict[tuple[str, str, str, str], OpenLoopSplitLineageEntryV1] = {}
    group_splits: dict[tuple[str, str], set[str]] = {}
    manifest_reports: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    for item in manifests:
        if not isinstance(item, dict):
            raise ValueError("each split manifest specification must be an object")
        split = str(item.get("split") or "")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported explicit split: {split!r}")
        manifest_id = _required_spec_string(item, "manifest_id")
        source_path = _resolve_spec_path(
            spec_path.parent,
            _required_spec_string(item, "path"),
        )
        source_digest = artifact_sha256_v1(source_path)
        parent_field = _required_spec_string(item, "parent_field")
        split_group_field = _required_spec_string(item, "split_group_field")
        dataset_field = _required_spec_string(item, "dataset_field")
        aliases = item.get("dataset_aliases", {})
        if not isinstance(aliases, dict):
            raise ValueError("dataset_aliases must be an object")
        aliases = {str(key): str(value) for key, value in aliases.items()}
        manifest_rows = 0
        for row_number, row in enumerate(iter_jsonl(source_path), start=1):
            manifest_rows += 1
            raw_dataset = resolve_explicit_field_v1(row, dataset_field)
            if raw_dataset not in aliases:
                raise ValueError(
                    f"manifest {manifest_id} has unmapped dataset value: "
                    f"{raw_dataset!r}"
                )
            dataset_source = aliases[raw_dataset]
            parent_sample_id = resolve_explicit_field_v1(row, parent_field)
            split_group_id = resolve_explicit_field_v1(row, split_group_field)
            entry = OpenLoopSplitLineageEntryV1(
                split=split,
                dataset_source=dataset_source,
                parent_sample_id=parent_sample_id,
                split_group_id=split_group_id,
                source_manifest_id=manifest_id,
                source_manifest_digest=source_digest,
                source_row_number=row_number,
            )
            key = (split, dataset_source, parent_sample_id, split_group_id)
            prior = entries.get(key)
            if prior is None:
                entries[key] = entry
            elif (
                prior.source_manifest_id != manifest_id
                or prior.source_manifest_digest != source_digest
            ):
                raise ValueError(
                    "one parent/group is attributed to conflicting manifests: "
                    f"{parent_sample_id}/{split_group_id}"
                )
            group_splits.setdefault(
                (dataset_source, split_group_id), set()
            ).add(split)
            raw_counts[f"split:{split}"] += 1
            raw_counts[f"dataset:{dataset_source}"] += 1
        manifest_reports.append(
            {
                "manifest_id": manifest_id,
                "split": split,
                "path": str(source_path),
                "sha256": source_digest,
                "raw_rows": manifest_rows,
            }
        )

    overlaps = {
        f"{dataset}:{group}": sorted(splits)
        for (dataset, group), splits in sorted(group_splits.items())
        if len(splits) > 1
    }
    overlap_policy = spec.get("overlap_policy", "fail_closed")
    quarantine_rows: list[dict[str, Any]] = []
    if overlaps and overlap_policy == "fail_closed":
        examples = dict(list(overlaps.items())[:20])
        raise ValueError(f"split-group leakage detected: {examples}")
    if overlap_policy not in {
        "fail_closed",
        "quarantine_evaluation_conflicts_v1",
    }:
        raise ValueError(f"unsupported split overlap policy: {overlap_policy}")
    if overlap_policy == "quarantine_evaluation_conflicts_v1":
        if quarantine_jsonl is None:
            raise ValueError("quarantine overlap policy requires quarantine_jsonl")
        overlap_keys = {
            (dataset, group): splits
            for (dataset, group), splits in group_splits.items()
            if len(splits) > 1
        }
        filtered: dict[
            tuple[str, str, str, str], OpenLoopSplitLineageEntryV1
        ] = {}
        for key, entry in entries.items():
            group_key = (entry.dataset_source, entry.split_group_id)
            splits = overlap_keys.get(group_key)
            if splits is None:
                filtered[key] = entry
                continue
            keep_train = "train" in splits and entry.split == "train"
            if keep_train:
                filtered[key] = entry
                continue
            quarantine_rows.append(
                {
                    "schema_version": (
                        "urbannav_openloop_split_lineage_quarantine_v2_0"
                    ),
                    "reason": (
                        "evaluation_group_exposed_to_training"
                        if "train" in splits
                        else "validation_test_group_conflict"
                    ),
                    "conflicting_splits": sorted(splits),
                    "entry": entry.model_dump(mode="json"),
                }
            )
        entries = filtered
        write_jsonl_atomic(quarantine_jsonl, quarantine_rows)
        group_splits = {}
        for entry in entries.values():
            group_splits.setdefault(
                (entry.dataset_source, entry.split_group_id), set()
            ).add(entry.split)
        residual = {
            key: splits
            for key, splits in group_splits.items()
            if len(splits) > 1
        }
        if residual:
            raise AssertionError(f"overlap quarantine left residuals: {residual}")

    ordered = [
        entries[key].model_dump(mode="json") for key in sorted(entries)
    ]
    write_jsonl_atomic(output_jsonl, ordered)
    emitted_counts = Counter(
        f"split:{row['split']}" for row in ordered
    )
    emitted_counts.update(
        f"dataset:{row['dataset_source']}" for row in ordered
    )
    report = {
        "schema_version": OPENLOOP_SPLIT_LINEAGE_REPORT_VERSION,
        "source_spec": str(spec_path),
        "source_spec_sha256": artifact_sha256_v1(spec_path),
        "output_jsonl": str(Path(output_jsonl).expanduser().resolve()),
        "output_sha256": artifact_sha256_v1(output_jsonl),
        "manifests": manifest_reports,
        "raw_counts": dict(sorted(raw_counts.items())),
        "emitted_counts": dict(sorted(emitted_counts.items())),
        "emitted_rows": len(ordered),
        "unique_groups": len(group_splits),
        "cross_split_group_overlap": 0,
        "overlap_policy": overlap_policy,
        "source_overlap_group_count": len(overlaps),
        "quarantined_rows": len(quarantine_rows),
        "quarantine_jsonl": (
            str(Path(quarantine_jsonl).expanduser().resolve())
            if quarantine_jsonl is not None
            else None
        ),
        "quarantine_sha256": (
            artifact_sha256_v1(quarantine_jsonl)
            if quarantine_jsonl is not None
            else None
        ),
    }
    _write_json(report_json, report)
    return report


def freeze_openloop_stage_i_views_v1(
    lineage_jsonl: str | Path,
    view_jsonls: Iterable[str | Path],
    output_jsonl: str | Path,
    report_json: str | Path,
    *,
    split: str,
    task_types: set[str] | None = None,
) -> dict[str, Any]:
    """Select canonical Stage I views by frozen dataset/group lineage.

    Args:
        lineage_jsonl: Canonical split lineage built by this module.
        view_jsonls: One or more canonical Stage I exchange-view artifacts.
        output_jsonl: Frozen selected-view destination.
        report_json: Selection audit destination.
        split: Validation or test split to freeze.
        task_types: Optional explicit task-family allowlist.

    Returns:
        Deterministic selection report.

    Raises:
        ValueError: If lineage overlaps, view IDs duplicate, or unsupported
            task types are requested.
    """

    if split not in {"validation", "test"}:
        raise ValueError("frozen benchmark split must be validation or test")
    allowed_tasks = task_types or {"full_cot", "vqa", "decision"}
    invalid_tasks = sorted(allowed_tasks - {"full_cot", "vqa", "decision"})
    if invalid_tasks:
        raise ValueError(f"unsupported Stage I task types: {invalid_tasks}")

    membership: dict[tuple[str, str], str] = {}
    eligible_groups: set[tuple[str, str]] = set()
    for payload in iter_jsonl(lineage_jsonl):
        entry = OpenLoopSplitLineageEntryV1.model_validate(payload)
        key = (entry.dataset_source, entry.split_group_id)
        prior = membership.get(key)
        if prior is not None and prior != entry.split:
            raise ValueError(
                f"lineage group belongs to multiple splits: {key}"
            )
        membership[key] = entry.split
        if entry.split == split:
            eligible_groups.add(key)

    selected: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    selected_groups: set[tuple[str, str]] = set()
    input_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    source_reports: list[dict[str, Any]] = []
    for source in view_jsonls:
        source_path = Path(source).expanduser().resolve()
        source_rows = 0
        source_selected = 0
        for payload in iter_jsonl(source_path):
            source_rows += 1
            view = parse_stage_i_training_view_v1(payload)
            input_counts[f"task:{view.task_type}"] += 1
            if view.task_type not in allowed_tasks:
                continue
            key = (view.dataset_source, view.split_group_id)
            if membership.get(key) != split:
                continue
            if view.sample_id in sample_ids:
                raise ValueError(
                    f"duplicate selected Stage I sample_id: {view.sample_id}"
                )
            sample_ids.add(view.sample_id)
            selected_groups.add(key)
            selected.append(view.model_dump(mode="json", exclude_none=True))
            selected_counts[f"task:{view.task_type}"] += 1
            selected_counts[f"dataset:{view.dataset_source}"] += 1
            source_selected += 1
        source_reports.append(
            {
                "path": str(source_path),
                "sha256": artifact_sha256_v1(source_path),
                "input_rows": source_rows,
                "selected_rows": source_selected,
            }
        )
    if not selected:
        raise ValueError(f"no Stage I views matched frozen {split} lineage")
    selected.sort(key=lambda row: row["sample_id"])
    write_jsonl_atomic(output_jsonl, selected)
    report = {
        "schema_version": OPENLOOP_FROZEN_VIEW_REPORT_VERSION,
        "split": split,
        "lineage_jsonl": str(Path(lineage_jsonl).expanduser().resolve()),
        "lineage_sha256": artifact_sha256_v1(lineage_jsonl),
        "source_views": source_reports,
        "output_jsonl": str(Path(output_jsonl).expanduser().resolve()),
        "output_sha256": artifact_sha256_v1(output_jsonl),
        "input_counts": dict(sorted(input_counts.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "selected_rows": len(selected),
        "eligible_group_count": len(eligible_groups),
        "selected_group_count": len(selected_groups),
        "group_coverage": len(selected_groups) / len(eligible_groups),
    }
    _write_json(report_json, report)
    return report


def resolve_explicit_field_v1(row: dict[str, Any], selector: str) -> str:
    """Resolve a required dotted field selector without inference.

    Args:
        row: Source row.
        selector: Explicit dot-delimited mapping path.

    Returns:
        Non-empty scalar value rendered as text.

    Raises:
        ValueError: If the selector is invalid, missing, or non-scalar.
    """

    if not selector or selector.startswith(".") or selector.endswith("."):
        raise ValueError(f"invalid explicit field selector: {selector!r}")
    value: Any = row
    for part in selector.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"source row is missing explicit field: {selector}")
        value = value[part]
    if value is None or isinstance(value, (dict, list)) or str(value) == "":
        raise ValueError(
            f"explicit field must resolve to a non-empty scalar: {selector}"
        )
    return str(value)


def _required_spec_string(spec: dict[str, Any], name: str) -> str:
    """Read one required non-empty string from a mapping specification."""

    value = spec.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"split manifest specification requires {name}")
    return value


def _resolve_spec_path(root: Path, value: str) -> Path:
    """Resolve one source artifact relative to its mapping specification."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one stable JSON audit artifact."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "OPENLOOP_SPLIT_SOURCE_SPEC_VERSION",
    "build_openloop_split_lineage_v1",
    "freeze_openloop_stage_i_views_v1",
    "resolve_explicit_field_v1",
]
