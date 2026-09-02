"""Materialize portable 4-second/32-point Stage II trajectory sidecars."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_HORIZON_STEPS_V2,
    STAGE_II_TRAJECTORY_POLICY_V2_ID,
    STAGE_II_TRAJECTORY_POLICY_V2_VERSION,
    STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION,
    StageIITrajectorySidecarV2,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_ii.raw_trajectory_v2 import (
    EGOWALK_RAW_ADAPTER_V2,
    FOXGLOVE_RAW_ADAPTER_V2,
    SIDEWALK_RAW_ADAPTER_V2,
    StageIIRawTrajectoryErrorV2,
    StageIIRawTrajectoryReaderV2,
)
from mcp_server.data.stage_ii.sources import (
    StageIISourceAdapterErrorV1,
    load_stage_ii_trajectory_source_registry_v1,
    stage_ii_assign_split_v1,
    stage_ii_source_row_identity_v1,
)
from mcp_server.data.stage_ii.trajectory import (
    build_stage_ii_trajectory_qc_v2,
    build_stage_ii_trajectory_target_v2,
)


STAGE_II_TRAJECTORY_CORPUS_REPORT_V2 = (
    "urbannav_stage_ii_trajectory_corpus_report_v2"
)


def build_stage_ii_trajectory_corpus_v2(
    *,
    dataset_manifests: Mapping[str, Sequence[str | Path]],
    raw_roots: Mapping[str, str | Path],
    identity_registry_json: str | Path,
    output_dir: str | Path,
    limit_per_dataset: int | None = None,
    limit_per_manifest: int | None = None,
    raw_cache_size: int = 8,
    progress_interval: int = 0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    full_horizon_required: bool = True,
) -> dict[str, Any]:
    """Build source-parent V2 sidecars, QC, rejects, and a coverage report.

    Args:
        dataset_manifests: Dataset family to one or more prepared manifests.
        raw_roots: Matching raw SideWalk/Foxglove/EgoWalk roots.
        identity_registry_json: Existing frozen parent/split identity registry.
        output_dir: Versioned output directory.
        limit_per_dataset: Optional non-empty-row smoke limit per dataset.
        limit_per_manifest: Optional non-empty-row smoke limit for each input
            manifest.  This keeps SideWalk main and extra represented together.
        raw_cache_size: Maximum loaded raw clips retained by the reader.
        progress_interval: Emit a callback every N globally scanned rows; zero
            disables progress events.
        progress_callback: Optional consumer for deterministic progress maps.
        full_horizon_required: Select full-horizon admission or the explicit
            prefix-mask capability for the materialized target rows.

    Returns:
        Deterministic report covering every scanned non-empty input row.

    Raises:
        ValueError: If input datasets, duplicate parents, or split identities
            conflict.
        FileNotFoundError: If an input manifest or registry is missing.

    Notes:
        This is a source-pool artifact.  It does not read Stage I targets or
        alter any current 16-query model view.  Later Stage II assembly joins
        ``source_group_id`` to an explicit lineage-safe Stage I split.
    """

    if limit_per_dataset is not None and int(limit_per_dataset) < 1:
        raise ValueError("limit_per_dataset must be positive")
    if limit_per_manifest is not None and int(limit_per_manifest) < 1:
        raise ValueError("limit_per_manifest must be positive")
    if int(progress_interval) < 0:
        raise ValueError("progress_interval cannot be negative")
    manifests = {
        str(dataset): [Path(path).expanduser() for path in paths]
        for dataset, paths in dataset_manifests.items()
    }
    if not manifests:
        raise ValueError("at least one dataset manifest is required")
    if set(manifests) != set(raw_roots):
        raise ValueError("dataset manifests and raw roots must cover identical datasets")
    registry = load_stage_ii_trajectory_source_registry_v1(identity_registry_json)
    for dataset, paths in manifests.items():
        spec = registry.dataset(dataset)
        if spec.status != "enabled":
            raise ValueError(f"dataset={dataset} is disabled by the identity registry")
        if not paths:
            raise ValueError(f"dataset={dataset} has no manifests")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": destination / "trajectory_sidecars_v2_train.jsonl",
        "validation": destination / "trajectory_sidecars_v2_validation.jsonl",
        "test": destination / "trajectory_sidecars_v2_test.jsonl",
        "qc": destination / "trajectory_qc_v2.jsonl",
        "rejected": destination / "trajectory_rejected_v2.jsonl",
    }
    temp_paths = {name: path.with_suffix(path.suffix + ".tmp") for name, path in output_paths.items()}
    handles = {
        name: path.open("w", encoding="utf-8")
        for name, path in temp_paths.items()
    }
    reader = StageIIRawTrajectoryReaderV2(raw_roots, cache_size=raw_cache_size)
    counts: Counter[str] = Counter()
    per_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    per_manifest: dict[str, Counter[str]] = defaultdict(Counter)
    metric_values: dict[str, list[float]] = defaultdict(list)
    seen_targets: dict[str, str] = {}
    manifest_evidence: list[dict[str, Any]] = []
    try:
        for dataset in sorted(manifests):
            scanned_for_dataset = 0
            spec = registry.dataset(dataset)
            for manifest in manifests[dataset]:
                scanned_for_manifest = 0
                manifest_key = f"{dataset}/{manifest.name}"
                manifest_sha = _file_sha256(manifest)
                manifest_evidence.append({
                    "dataset_source": dataset,
                    "path": str(manifest),
                    "name": f"{dataset}/{manifest.name}",
                    "size_bytes": manifest.stat().st_size,
                    "sha256": manifest_sha,
                })
                with manifest.open("r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        if (
                            limit_per_dataset is not None
                            and scanned_for_dataset >= int(limit_per_dataset)
                        ):
                            break
                        if (
                            limit_per_manifest is not None
                            and scanned_for_manifest >= int(limit_per_manifest)
                        ):
                            break
                        scanned_for_dataset += 1
                        scanned_for_manifest += 1
                        counts["input_rows"] += 1
                        per_dataset[dataset]["input_rows"] += 1
                        per_manifest[manifest_key]["input_rows"] += 1
                        try:
                            row = json.loads(line)
                            if not isinstance(row, dict):
                                raise StageIISourceAdapterErrorV1(
                                    "trajectory_row_not_object",
                                    "source JSONL row must be an object",
                                )
                            sample_id, split_group_id = stage_ii_source_row_identity_v1(
                                row,
                                spec,
                            )
                            split = stage_ii_assign_split_v1(split_group_id, registry)
                        except (json.JSONDecodeError, StageIISourceAdapterErrorV1) as exc:
                            reason = getattr(exc, "code", "trajectory_json_invalid")
                            _write_unresolved_rejection(
                                handles["rejected"],
                                dataset=dataset,
                                manifest=manifest,
                                manifest_sha=manifest_sha,
                                line_number=line_number,
                                reason=reason,
                                detail=str(exc),
                            )
                            counts[f"rejected:{reason}"] += 1
                            per_dataset[dataset][f"rejected:{reason}"] += 1
                            per_manifest[manifest_key][f"rejected:{reason}"] += 1
                            _emit_progress(
                                callback=progress_callback,
                                interval=progress_interval,
                                counts=counts,
                                dataset=dataset,
                                manifest=manifest,
                                line_number=line_number,
                            )
                            continue
                        try:
                            raw_window = reader.extract(
                                dataset,
                                row,
                                full_horizon_required=bool(full_horizon_required),
                            )
                            result = build_stage_ii_trajectory_target_v2(
                                sample_id=sample_id,
                                dataset_source=dataset,
                                clip_id=raw_window.clip_id,
                                frame_idx=raw_window.frame_idx,
                                raw_trajectory=raw_window.trajectory_ego_timed,
                                action_mask=raw_window.action_mask,
                                source_adapter_id=raw_window.source_adapter_id,
                                trajectory_source=raw_window.trajectory_source,
                                allowed_trajectory_sources={raw_window.trajectory_source},
                                raw_timestamp_unit=raw_window.raw_timestamp_unit,
                                yaw_offset_deg=raw_window.yaw_offset_deg,
                                full_horizon_required=bool(full_horizon_required),
                            )
                            qc = result.qc
                        except StageIIRawTrajectoryErrorV2 as exc:
                            context = _rejection_context(dataset, row)
                            qc = build_stage_ii_trajectory_qc_v2(
                                sample_id=sample_id,
                                dataset_source=dataset,
                                source_adapter_id=context["source_adapter_id"],
                                trajectory_source=context["trajectory_source"],
                                clip_id=context["clip_id"],
                                frame_idx=context["frame_idx"],
                                status="rejected",
                                reason_codes=(exc.code,),
                                raw_pose_count=0,
                                source_valid_step_count=0,
                                source_horizon_seconds=None,
                                max_source_gap_sec=None,
                                max_source_step_m=None,
                                max_source_speed_mps=None,
                                max_source_yaw_rate_rad_s=None,
                                target_valid_step_count=0,
                                source_evidence_digest=None,
                            )
                            result = None

                        if sample_id in seen_targets:
                            current_digest = (
                                result.target.target_digest
                                if result is not None and result.target is not None
                                else f"rejected:{qc.reason_codes[0]}"
                            )
                            if seen_targets[sample_id] != current_digest:
                                raise ValueError(
                                    f"conflicting duplicate V2 trajectory parent: {sample_id}"
                                )
                            counts["duplicate_equivalent"] += 1
                            per_dataset[dataset]["duplicate_equivalent"] += 1
                            per_manifest[manifest_key]["duplicate_equivalent"] += 1
                            _emit_progress(
                                callback=progress_callback,
                                interval=progress_interval,
                                counts=counts,
                                dataset=dataset,
                                manifest=manifest,
                                line_number=line_number,
                            )
                            continue
                        seen_targets[sample_id] = (
                            result.target.target_digest
                            if result is not None and result.target is not None
                            else f"rejected:{qc.reason_codes[0]}"
                        )
                        _write_json_line(handles["qc"], qc.model_dump(mode="json", exclude_none=True))
                        if result is None or not result.accepted or result.target is None:
                            reason = str(qc.reason_codes[0])
                            _write_json_line(
                                handles["rejected"],
                                {
                                    "sample_id": sample_id,
                                    "source_group_id": sample_id,
                                    "split_group_id": split_group_id,
                                    "dataset_source": dataset,
                                    "split": split,
                                    "source_manifest_name": f"{dataset}/{manifest.name}",
                                    "source_manifest_sha256": manifest_sha,
                                    "source_line": line_number,
                                    "trajectory_qc": qc.model_dump(
                                        mode="json",
                                        exclude_none=True,
                                    ),
                                },
                            )
                            counts[f"rejected:{reason}"] += 1
                            per_dataset[dataset][f"rejected:{reason}"] += 1
                            per_manifest[manifest_key][f"rejected:{reason}"] += 1
                            _emit_progress(
                                callback=progress_callback,
                                interval=progress_interval,
                                counts=counts,
                                dataset=dataset,
                                manifest=manifest,
                                line_number=line_number,
                            )
                            continue
                        sidecar_payload = {
                            "schema_version": STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION,
                            "sample_id": sample_id,
                            "source_group_id": sample_id,
                            "split_group_id": split_group_id,
                            "dataset_source": dataset,
                            "split": split,
                            "source_manifest_name": f"{dataset}/{manifest.name}",
                            "source_manifest_sha256": manifest_sha,
                            "source_line": line_number,
                            "trajectory_target": result.target.model_dump(mode="json"),
                            "trajectory_qc": qc.model_dump(mode="json", exclude_none=True),
                        }
                        sidecar_payload["sidecar_digest"] = stage_ii_digest_v1(sidecar_payload)
                        sidecar = StageIITrajectorySidecarV2.model_validate(sidecar_payload)
                        _write_json_line(handles[split], sidecar.model_dump(mode="json"))
                        counts["accepted"] += 1
                        counts[f"accepted:{split}"] += 1
                        per_dataset[dataset]["accepted"] += 1
                        per_dataset[dataset][f"accepted:{split}"] += 1
                        per_manifest[manifest_key]["accepted"] += 1
                        per_manifest[manifest_key][f"accepted:{split}"] += 1
                        for audit_code in qc.reason_codes[1:]:
                            counts[f"audit:{audit_code}"] += 1
                            per_dataset[dataset][f"audit:{audit_code}"] += 1
                            per_manifest[manifest_key][f"audit:{audit_code}"] += 1
                        terminal = result.target.trajectory_ego_t[-1]
                        metric_values["terminal_distance_m"].append(
                            math.hypot(float(terminal[0]), float(terminal[1]))
                        )
                        metric_values["terminal_abs_lateral_m"].append(abs(float(terminal[1])))
                        metric_values["terminal_abs_yaw_rad"].append(abs(float(terminal[2])))
                        for name in (
                            "max_source_gap_sec",
                            "max_source_step_m",
                            "max_source_speed_mps",
                            "max_source_yaw_rate_rad_s",
                        ):
                            value = getattr(qc, name)
                            if value is not None:
                                metric_values[name].append(float(value))
                        _emit_progress(
                            callback=progress_callback,
                            interval=progress_interval,
                            counts=counts,
                            dataset=dataset,
                            manifest=manifest,
                            line_number=line_number,
                        )
                    if (
                        limit_per_dataset is not None
                        and scanned_for_dataset >= int(limit_per_dataset)
                    ):
                        break
    finally:
        for handle in handles.values():
            handle.close()

    for name, path in output_paths.items():
        temp_paths[name].replace(path)
    output_evidence = {
        name: {
            "path": str(path),
            "rows": _line_count(path),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for name, path in output_paths.items()
    }
    rejected_count = sum(
        count for name, count in counts.items() if name.startswith("rejected:")
    )
    report = {
        "schema_version": STAGE_II_TRAJECTORY_CORPUS_REPORT_V2,
        "trajectory_policy_id": STAGE_II_TRAJECTORY_POLICY_V2_ID,
        "trajectory_policy_version": STAGE_II_TRAJECTORY_POLICY_V2_VERSION,
        "target_shape": [STAGE_II_HORIZON_STEPS_V2, 3],
        "selection_scope": "prepared_source_pool",
        "split_authority": {
            "registry_path": str(Path(identity_registry_json).expanduser()),
            "registry_id": registry.registry_id,
            "registry_version": registry.registry_version,
            "registry_digest": registry.registry_digest,
            "split_policy": registry.split_policy.model_dump(mode="json"),
        },
        "raw_roots": {
            dataset: str(Path(path).expanduser())
            for dataset, path in raw_roots.items()
        },
        "limit_per_dataset": limit_per_dataset,
        "limit_per_manifest": limit_per_manifest,
        "full_horizon_required": bool(full_horizon_required),
        "counts": {
            **dict(counts),
            "rejected_total": rejected_count,
            "accounted_unique_parents": counts["accepted"] + rejected_count,
        },
        "per_dataset": {
            dataset: dict(values)
            for dataset, values in sorted(per_dataset.items())
        },
        "per_manifest": {
            manifest: dict(values)
            for manifest, values in sorted(per_manifest.items())
        },
        "metric_distributions": {
            name: _distribution(values)
            for name, values in sorted(metric_values.items())
        },
        "manifest_evidence": manifest_evidence,
        "outputs": output_evidence,
        "complete": (
            counts["input_rows"]
            == counts["accepted"] + rejected_count + counts["duplicate_equivalent"]
        ),
    }
    report_path = destination / "trajectory_build_report_v2.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _emit_progress(
    *,
    callback: Callable[[dict[str, Any]], None] | None,
    interval: int,
    counts: Counter[str],
    dataset: str,
    manifest: Path,
    line_number: int,
) -> None:
    """Emit one bounded progress event after an accounted input row."""

    if callback is None or int(interval) <= 0:
        return
    input_rows = int(counts["input_rows"])
    if input_rows % int(interval) != 0:
        return
    rejected = sum(
        int(value)
        for name, value in counts.items()
        if name.startswith("rejected:")
    )
    callback({
        "input_rows": input_rows,
        "accepted": int(counts["accepted"]),
        "rejected": rejected,
        "duplicate_equivalent": int(counts["duplicate_equivalent"]),
        "dataset_source": dataset,
        "manifest_name": manifest.name,
        "source_line": int(line_number),
    })


def _rejection_context(dataset: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Recover portable source identity for a pre-builder rejection."""

    if dataset == "egowalk":
        return {
            "source_adapter_id": EGOWALK_RAW_ADAPTER_V2,
            "trajectory_source": "egowalk_odometry",
            "clip_id": str(row.get("clip_id") or "unknown"),
            "frame_idx": _safe_nonnegative_int(row.get("frame_idx")),
        }
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), Mapping) else {}
    return {
        "source_adapter_id": (
            SIDEWALK_RAW_ADAPTER_V2 if dataset == "sidewalk" else FOXGLOVE_RAW_ADAPTER_V2
        ),
        "trajectory_source": f"{dataset}_frame_pose_odometry",
        "clip_id": str(meta.get("clip_id") or row.get("clip_id") or "unknown"),
        "frame_idx": _safe_nonnegative_int(
            meta.get("frame_idx", row.get("frame_idx", 0))
        ),
    }


def _safe_nonnegative_int(value: Any) -> int:
    """Render malformed frame identity as zero only inside rejection audit."""

    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _write_unresolved_rejection(
    handle: TextIO,
    *,
    dataset: str,
    manifest: Path,
    manifest_sha: str,
    line_number: int,
    reason: str,
    detail: str,
) -> None:
    """Write a bounded rejection when canonical parent identity is unavailable."""

    _write_json_line(
        handle,
        {
            "sample_id": f"unresolved:{dataset}:{manifest_sha[:12]}:{line_number}",
            "dataset_source": dataset,
            "source_manifest_name": f"{dataset}/{manifest.name}",
            "source_manifest_sha256": manifest_sha,
            "source_line": line_number,
            "reason_code": reason,
            "detail": detail[:1000],
        },
    )


def _write_json_line(handle: TextIO, payload: Mapping[str, Any]) -> None:
    """Append one compact deterministic JSON object."""

    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    """Hash one input/output file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    """Count physical lines in one generated JSONL artifact."""

    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return exact bounded quantiles for one scalar audit distribution."""

    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        left = int(math.floor(position))
        right = int(math.ceil(position))
        alpha = position - left
        return ordered[left] + alpha * (ordered[right] - ordered[left])

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }


__all__ = [
    "STAGE_II_TRAJECTORY_CORPUS_REPORT_V2",
    "build_stage_ii_trajectory_corpus_v2",
]
