#!/usr/bin/env python3
"""Audit frozen Stage II trajectory sources on unique real train parents."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.stage_ii_cfm import stage_ii_digest_v1
from mcp_server.data.stage_ii.sources import StageIISourceAdapterErrorV1, adapt_stage_ii_trajectory_source_row_v1, load_stage_ii_trajectory_source_registry_v1, stage_ii_assign_split_v1, stage_ii_source_row_identity_v1
from mcp_server.data.stage_ii.trajectory import build_stage_ii_trajectory_target_v1


AUDIT_REPORT_SCHEMA_VERSION = "urbannav_stage_ii_target_audit_report_v1"
AUDIT_ROW_SCHEMA_VERSION = "urbannav_stage_ii_target_audit_row_v1"


def audit_stage_ii_trajectory_targets(
    *,
    registry_json: str | Path,
    dataset_manifests: Mapping[str, str | Path],
    output_dir: str | Path,
    parents_per_dataset: int = 200,
    max_parents_per_group: int = 8,
    scan_limit_per_dataset: int = 25000,
) -> dict[str, Any]:
    """Run a deterministic target-only audit on real train-split parents.

    Args:
        registry_json: Frozen canonical source registry.
        dataset_manifests: Explicit dataset-family to source-JSONL paths.
        output_dir: Durable audit artifact directory.
        parents_per_dataset: Selected unique train parents per enabled dataset.
        max_parents_per_group: Maximum selected candidates retained per clip.
        scan_limit_per_dataset: Maximum real source rows scanned per dataset.

    Returns:
        Aggregate report with accepted/drop, horizon, mask, and yaw-source
        distributions.

    Raises:
        ValueError: If sources do not match the registry or the final scope is
            outside 100--1000 unique parents.
    """

    if parents_per_dataset < 1:
        raise ValueError("parents_per_dataset must be positive")
    if max_parents_per_group < 1:
        raise ValueError("max_parents_per_group must be positive")
    if scan_limit_per_dataset < parents_per_dataset:
        raise ValueError("scan_limit_per_dataset must cover parents_per_dataset")
    registry = load_stage_ii_trajectory_source_registry_v1(registry_json)
    supplied = {str(name): Path(path).expanduser() for name, path in dataset_manifests.items()}
    enabled = {
        str(spec.dataset_source)
        for spec in registry.datasets
        if spec.status == "enabled"
    }
    if set(supplied) != enabled:
        raise ValueError(
            "dataset manifests must exactly cover enabled registry datasets: "
            f"expected={sorted(enabled)}, got={sorted(supplied)}"
        )
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    dataset_scan_reports: dict[str, Any] = {}
    manifest_evidence: dict[str, Any] = {}
    for spec in registry.datasets:
        dataset = str(spec.dataset_source)
        if spec.status == "disabled":
            dataset_scan_reports[dataset] = {
                "admission_status": "disabled",
                "disabled_reason": str(spec.disabled_reason),
                "scanned_rows": 0,
                "selected_unique_parents": 0,
            }
            continue
        path = supplied[dataset]
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        manifest_evidence[dataset] = {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        selected, scan_report = _audit_dataset(
            path,
            spec=spec,
            registry=registry,
            parents_per_dataset=parents_per_dataset,
            max_parents_per_group=max_parents_per_group,
            scan_limit=scan_limit_per_dataset,
        )
        if len(selected) < parents_per_dataset:
            raise ValueError(
                f"dataset={dataset} yielded only {len(selected)} train parents; "
                f"requested {parents_per_dataset}"
            )
        selected_rows.extend(selected)
        dataset_scan_reports[dataset] = scan_report
    unique_ids = {str(row["sample_id"]) for row in selected_rows}
    if len(unique_ids) != len(selected_rows):
        raise ValueError("target audit selected duplicate parent IDs across datasets")
    if not 100 <= len(selected_rows) <= 1000:
        raise ValueError("target audit scope must contain 100--1000 unique parents")

    accepted = [row for row in selected_rows if row["status"] == "accepted"]
    dropped = [row for row in selected_rows if row["status"] == "dropped"]
    qc_rows = [row["qc"] for row in selected_rows if row.get("qc") is not None]
    _write_jsonl(destination / "accepted_targets.jsonl", accepted)
    _write_jsonl(destination / "dropped_targets.jsonl", dropped)
    _write_jsonl(destination / "target_qc.jsonl", qc_rows)
    report = {
        "schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "audit_scope": "real_train_split_unique_parent_target_only",
        "training_admission_scope": "target_only_not_full_stage_ii_view",
        "registry": {
            "path": str(Path(registry_json).expanduser()),
            "registry_id": registry.registry_id,
            "registry_version": registry.registry_version,
            "registry_digest": registry.registry_digest,
        },
        "selection_policy": {
            "split_policy": registry.split_policy.model_dump(mode="json"),
            "parents_per_dataset": parents_per_dataset,
            "max_parents_per_group": max_parents_per_group,
            "scan_limit_per_dataset": scan_limit_per_dataset,
            "candidate_rank": "sha256(seed:sample_id)",
        },
        "manifest_evidence": manifest_evidence,
        "dataset_scan": dataset_scan_reports,
        "unique_parent_count": len(selected_rows),
        "accepted_count": len(accepted),
        "drop_count": len(dropped),
        "selection_digest": stage_ii_digest_v1(sorted(unique_ids)),
        "distributions": _distribution_report(selected_rows),
        "per_dataset": {
            dataset: _distribution_report(
                [row for row in selected_rows if row["dataset_source"] == dataset]
            )
            for dataset in sorted(enabled)
        },
        "outputs": {
            "accepted_targets_jsonl": str(destination / "accepted_targets.jsonl"),
            "dropped_targets_jsonl": str(destination / "dropped_targets.jsonl"),
            "target_qc_jsonl": str(destination / "target_qc.jsonl"),
            "report_json": str(destination / "report.json"),
        },
    }
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _audit_dataset(
    path: Path,
    *,
    spec,
    registry,
    parents_per_dataset: int,
    max_parents_per_group: int,
    scan_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan one real manifest and retain deterministic diverse train parents."""

    group_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if counts["scanned_rows"] >= scan_limit:
                break
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
                sample_id, split_group_id = stage_ii_source_row_identity_v1(raw, spec)
            except (json.JSONDecodeError, StageIISourceAdapterErrorV1) as exc:
                code = getattr(exc, "code", "trajectory_json_invalid")
                counts[f"preselection_drop:{code}"] += 1
                continue
            split = stage_ii_assign_split_v1(split_group_id, registry)
            counts[f"split:{split}"] += 1
            if split != "train":
                continue
            if sample_id in seen_ids:
                counts["duplicate_parent_rows"] += 1
                continue
            seen_ids.add(sample_id)
            audit_row = _build_audit_row(
                raw,
                spec=spec,
                registry=registry,
                source_path=path,
                source_line=line_number,
                sample_id=sample_id,
                split_group_id=split_group_id,
            )
            candidates = group_candidates[split_group_id]
            candidates.append(audit_row)
            candidates.sort(key=lambda item: item["selection_hash"])
            if len(candidates) > max_parents_per_group:
                candidates.pop()
    flattened = [item for group in group_candidates.values() for item in group]
    flattened.sort(key=lambda item: item["selection_hash"])
    selected = flattened[:parents_per_dataset]
    counts["unique_train_parents_seen"] = len(seen_ids)
    counts["eligible_groups_seen"] = len(group_candidates)
    counts["selected_unique_parents"] = len(selected)
    return selected, {
        "admission_status": "enabled",
        "manifest": str(path),
        "counts": dict(counts),
        "selected_group_count": len({row["split_group_id"] for row in selected}),
    }


def _build_audit_row(
    row: Mapping[str, Any],
    *,
    spec,
    registry,
    source_path: Path,
    source_line: int,
    sample_id: str,
    split_group_id: str,
) -> dict[str, Any]:
    """Adapt, build, and serialize one compact target-audit result."""

    selection_hash = hashlib.sha256(
        f"{registry.split_policy.seed}:{sample_id}".encode("utf-8")
    ).hexdigest()
    base = {
        "schema_version": AUDIT_ROW_SCHEMA_VERSION,
        "sample_id": sample_id,
        "split_group_id": split_group_id,
        "split": "train",
        "dataset_source": str(spec.dataset_source),
        "source_manifest": str(source_path),
        "source_line": source_line,
        "registry_digest": registry.registry_digest,
        "selection_hash": selection_hash,
    }
    try:
        adapted = adapt_stage_ii_trajectory_source_row_v1(row, spec)
    except StageIISourceAdapterErrorV1 as exc:
        return {
            **base,
            "status": "dropped",
            "drop_reason": exc.code,
            "detail": str(exc)[:1000],
        }
    build = build_stage_ii_trajectory_target_v1(
        sample_id=adapted.sample_id,
        dataset_source=adapted.dataset_source,
        raw_trajectory=adapted.raw_trajectory,
        action_mask=adapted.action_mask,
        source_format=adapted.source_format,
        trajectory_source=adapted.trajectory_source,
        allowed_trajectory_sources=spec.allowed_trajectory_sources,
        yaw_source=adapted.yaw_source,
    )
    source_audit = {
        "source_format": adapted.source_format,
        "trajectory_source": adapted.trajectory_source,
        "yaw_source": adapted.yaw_source,
        "mask_provenance": adapted.mask_provenance,
        "origin_provenance": adapted.origin_provenance,
        "source_step_count": len(adapted.raw_trajectory),
        "source_valid_step_count": sum(adapted.action_mask),
        "source_horizon_seconds": adapted.source_horizon_seconds,
    }
    if build.target is None:
        return {
            **base,
            "status": "dropped",
            "drop_reason": str(build.qc.reason_codes[0]),
            "source_audit": source_audit,
            "qc": build.qc.model_dump(mode="json"),
        }
    return {
        **base,
        "status": "accepted",
        "source_audit": source_audit,
        "target_valid_horizon_seconds": (
            build.target.valid_step_count
            * build.target.policy.horizon_seconds
            / build.target.policy.horizon_steps
        ),
        "target_mask_pattern": (
            f"{build.target.valid_step_count}/{build.target.policy.horizon_steps}"
        ),
        "trajectory_target": build.target.model_dump(mode="json"),
        "qc": build.qc.model_dump(mode="json"),
    }


def _distribution_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the exact target-audit distributions requested by the gate."""

    counts: Counter[str] = Counter()
    for row in rows:
        status = str(row["status"])
        counts[f"status:{status}"] += 1
        if status == "dropped":
            counts[f"drop_reason:{row.get('drop_reason', 'unknown')}"] += 1
            continue
        source = row["source_audit"]
        counts[f"source_format:{source['source_format']}"] += 1
        counts[f"trajectory_source:{source['trajectory_source']}"] += 1
        counts[f"yaw_source:{source['yaw_source']}"] += 1
        counts[f"mask_provenance:{source['mask_provenance']}"] += 1
        counts[f"origin_provenance:{source['origin_provenance']}"] += 1
        counts[f"source_steps:{source['source_step_count']}"] += 1
        counts[f"source_valid_steps:{source['source_valid_step_count']}"] += 1
        counts[f"target_mask:{row['target_mask_pattern']}"] += 1
        counts[f"target_horizon_sec:{row['target_valid_horizon_seconds']:.3f}"] += 1
        counts[
            f"source_horizon_bucket:{_horizon_bucket(source['source_horizon_seconds'])}"
        ] += 1
    return {
        "parent_count": len(rows),
        "accepted": counts.pop("status:accepted", 0),
        "dropped": counts.pop("status:dropped", 0),
        "drop_reasons": _counter_prefix(counts, "drop_reason:"),
        "target_horizon_seconds": _counter_prefix(counts, "target_horizon_sec:"),
        "target_mask_patterns": _counter_prefix(counts, "target_mask:"),
        "source_horizon_buckets": _counter_prefix(counts, "source_horizon_bucket:"),
        "source_step_counts": _counter_prefix(counts, "source_steps:"),
        "source_valid_step_counts": _counter_prefix(counts, "source_valid_steps:"),
        "mask_provenance": _counter_prefix(counts, "mask_provenance:"),
        "origin_provenance": _counter_prefix(counts, "origin_provenance:"),
        "yaw_source": _counter_prefix(counts, "yaw_source:"),
        "source_format": _counter_prefix(counts, "source_format:"),
        "trajectory_source": _counter_prefix(counts, "trajectory_source:"),
    }


def _horizon_bucket(value: float) -> str:
    """Bucket source temporal coverage without hiding fixed-grid precision."""

    if value < 0.5:
        return "lt_0p5"
    if value < 1.0:
        return "0p5_to_lt_1p0"
    if value < 2.0:
        return "1p0_to_lt_2p0"
    return "ge_2p0"


def _counter_prefix(counter: Counter[str], prefix: str) -> dict[str, int]:
    """Return sorted counter entries after removing one namespace prefix."""

    return {
        key[len(prefix):]: value
        for key, value in sorted(counter.items())
        if key.startswith(prefix)
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write compact deterministic JSONL audit artifacts."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _dataset_manifest_args(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``dataset=/path/source.jsonl`` CLI values."""

    result: dict[str, Path] = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or not dataset.strip() or not path.strip():
            raise ValueError(f"invalid --dataset-manifest value: {value!r}")
        dataset = dataset.strip()
        if dataset in result:
            raise ValueError(f"duplicate dataset manifest: {dataset}")
        result[dataset] = Path(path).expanduser()
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse target-audit command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-json", required=True, type=Path)
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        default=[],
        help="Repeat dataset=/absolute/or/relative/source.jsonl",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parents-per-dataset", type=int, default=200)
    parser.add_argument("--max-parents-per-group", type=int, default=8)
    parser.add_argument("--scan-limit-per-dataset", type=int, default=25000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the real-train target audit and print its durable report."""

    args = parse_args(argv)
    report = audit_stage_ii_trajectory_targets(
        registry_json=args.registry_json,
        dataset_manifests=_dataset_manifest_args(args.dataset_manifest),
        output_dir=args.output_dir,
        parents_per_dataset=args.parents_per_dataset,
        max_parents_per_group=args.max_parents_per_group,
        scan_limit_per_dataset=args.scan_limit_per_dataset,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
