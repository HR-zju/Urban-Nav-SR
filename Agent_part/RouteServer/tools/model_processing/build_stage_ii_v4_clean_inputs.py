#!/usr/bin/env python3
"""Materialize a Stage-I V2 -> clean CFM-V1 denominator from immutable QC truth.

The historical clean target ledger is an annotation-only proof that a parent
passed the 3 m source-step gate.  This tool does not reuse its legacy prompt
profile: it selects matching current Stage-I V2 FullCoT parents, rechecks the
underlying raw sidecar against the same bound, and emits inputs for the
unchanged 16-point CFM-V1 trajectory contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV2,
    parse_stage_i_training_view_v2,
)
from mcp_server.contracts.stage_ii_cfm import (
    StageIILegacyDirectTargetV1,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_ii.sources import (
    load_stage_ii_trajectory_source_registry_v1,
)
from mcp_server.data.stage_ii.trajectory import (
    build_stage_ii_trajectory_target_v1,
)


REPORT_SCHEMA_VERSION = "urbannav_stage_ii_v4_clean_input_build_report_v1"
PROFILE_ID = "reasoning_v4_6_stage_i_v2_to_cfm_v1_clean_v1"
PROFILE_VERSION = "1.0"
DEFAULT_MAX_SOURCE_STEP_M = 3.0


def build_stage_ii_v4_clean_inputs(
    *,
    stage_i_jsonl: str | Path,
    clean_train_targets_jsonl: str | Path,
    clean_validation_targets_jsonl: str | Path,
    prebuilt_train_sidecars_jsonl: str | Path,
    prebuilt_validation_sidecars_jsonl: str | Path,
    trajectory_source_registry_json: str | Path,
    output_dir: str | Path,
    max_source_step_m: float = DEFAULT_MAX_SOURCE_STEP_M,
) -> dict[str, Any]:
    """Build a source-compatible clean V4 Stage II denominator.

    Args:
        stage_i_jsonl: Current canonical V4 FullCoT source artifact.
        clean_train_targets_jsonl: Immutable accepted historical train targets.
        clean_validation_targets_jsonl: Immutable accepted validation targets.
        prebuilt_train_sidecars_jsonl: Raw-timed V4 train sidecars covering the
            historical Stage II population.
        prebuilt_validation_sidecars_jsonl: Matching validation sidecars.
        trajectory_source_registry_json: Frozen dataset/source allowlist.
        output_dir: Destination for V4-selected inputs and lineage audits.
        max_source_step_m: Strict bound independently rechecked against raw
            sidecars before the target is admitted.

    Returns:
        A deterministic report with source coverage, verified target counts,
        and SHA-256 fingerprints for every output.

    Raises:
        ValueError: If a selected identity, target, source registry declaration,
            or rechecked trajectory semantics drifts.
        FileNotFoundError: If a supplied immutable artifact is absent.
    """

    if max_source_step_m <= 0.0:
        raise ValueError("max_source_step_m must be positive")
    registry = load_stage_ii_trajectory_source_registry_v1(
        trajectory_source_registry_json
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    stage_i = _load_v4_full_cot(stage_i_jsonl)
    targets = {
        "train": _load_clean_targets(clean_train_targets_jsonl, split="train"),
        "validation": _load_clean_targets(
            clean_validation_targets_jsonl,
            split="validation",
        ),
    }
    _validate_target_split_disjointness(targets)
    sidecars = {
        "train": _index_unique(prebuilt_train_sidecars_jsonl, "stage_i_sample_id"),
        "validation": _index_unique(
            prebuilt_validation_sidecars_jsonl,
            "stage_i_sample_id",
        ),
    }

    selected_views: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    selected_sidecars: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    coverage_drops: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for split in ("train", "validation"):
        for parent_id, clean_target in sorted(targets[split].items()):
            counts[f"clean_target:{split}"] += 1
            view = stage_i.get(parent_id)
            if view is None:
                coverage_drops.append(_coverage_drop(clean_target, split=split))
                counts[f"drop:stage_i_v4_source_missing:{split}"] += 1
                continue
            raw_sidecar = sidecars[split].get(parent_id)
            if raw_sidecar is None:
                raise ValueError(
                    f"selected V4 parent has no prebuilt {split} sidecar: {parent_id}"
                )
            _validate_selected_lineage(
                view,
                clean_target=clean_target,
                raw_sidecar=raw_sidecar,
                registry=registry,
                max_source_step_m=max_source_step_m,
            )
            selected_views[split].append(
                view.model_dump(mode="json", exclude_none=True)
            )
            selected_sidecars[split].append(
                _attach_clean_lineage(
                    raw_sidecar,
                    clean_target=clean_target,
                    max_source_step_m=max_source_step_m,
                )
            )
            counts[f"selected:{split}"] += 1
            counts[f"selected_dataset:{split}:{view.dataset_source}"] += 1

    paths = {
        "train_stage_i_jsonl": destination / "stage_i_full_cot_train.jsonl",
        "validation_stage_i_jsonl": destination / "stage_i_full_cot_validation.jsonl",
        "train_sidecars_jsonl": destination / "trajectory_sidecars_train.jsonl",
        "validation_sidecars_jsonl": destination / "trajectory_sidecars_validation.jsonl",
        "coverage_drops_jsonl": destination / "v4_source_coverage_drops.jsonl",
        "report_json": destination / "build_report.json",
    }
    for split in ("train", "validation"):
        _write_jsonl(
            paths[f"{split}_stage_i_jsonl"],
            sorted(selected_views[split], key=lambda item: str(item["sample_id"])),
        )
        _write_jsonl(
            paths[f"{split}_sidecars_jsonl"],
            sorted(
                selected_sidecars[split],
                key=lambda item: str(item["stage_i_sample_id"]),
            ),
        )
    _write_jsonl(paths["coverage_drops_jsonl"], coverage_drops)
    output_sha256 = {
        key: _file_sha256(path)
        for key, path in paths.items()
        if key != "report_json"
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "selection_authority": {
            "target_filter": "immutable_clean_target_ledger",
            "target_recheck": "raw_sidecar_3m_source_step_bound",
            "stage_i_source": "current_v4_full_cot_only",
            "split_authority": "immutable_clean_target_split",
        },
        "max_source_step_m": max_source_step_m,
        "inputs": {
            "stage_i_jsonl": str(Path(stage_i_jsonl).expanduser().resolve()),
            "clean_train_targets_jsonl": str(
                Path(clean_train_targets_jsonl).expanduser().resolve()
            ),
            "clean_validation_targets_jsonl": str(
                Path(clean_validation_targets_jsonl).expanduser().resolve()
            ),
            "prebuilt_train_sidecars_jsonl": str(
                Path(prebuilt_train_sidecars_jsonl).expanduser().resolve()
            ),
            "prebuilt_validation_sidecars_jsonl": str(
                Path(prebuilt_validation_sidecars_jsonl).expanduser().resolve()
            ),
            "trajectory_source_registry_json": str(
                Path(trajectory_source_registry_json).expanduser().resolve()
            ),
        },
        "registry": {
            "registry_id": registry.registry_id,
            "registry_version": registry.registry_version,
            "registry_digest": registry.registry_digest,
        },
        "counts": dict(sorted(counts.items())),
        "stage_i_v2_source_rows": len(stage_i),
        "full_clean_target_coverage": {
            split: {
                "clean_target_rows": len(targets[split]),
                "selected_rows": len(selected_views[split]),
                "missing_current_v4_source_rows": sum(
                    1 for row in coverage_drops if row["split"] == split
                ),
            }
            for split in ("train", "validation")
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "output_sha256": output_sha256,
        "complete_for_v4_clean_denominator": all(
            selected_sidecars[split]
            and len(selected_sidecars[split]) == len(selected_views[split])
            for split in ("train", "validation")
        ),
        "full_clean_target_coverage_complete": not coverage_drops,
    }
    paths["report_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_v4_full_cot(path: str | Path) -> dict[str, StageIFullCoTViewV2]:
    """Index formal current V2-envelope FullCoT rows by exact sample ID."""

    result: dict[str, StageIFullCoTViewV2] = {}
    for payload in _iter_jsonl(path):
        typed = parse_stage_i_training_view_v2(payload)
        if not isinstance(typed, StageIFullCoTViewV2):
            continue
        if typed.sample_id in result:
            raise ValueError(f"duplicate V2-envelope FullCoT sample_id: {typed.sample_id}")
        if typed.parent_audit_id is None:
            raise ValueError(
                f"V4 FullCoT sample has no resolved action audit: {typed.sample_id}"
            )
        result[typed.sample_id] = typed
    if not result:
        raise ValueError("stage_i_jsonl contains no formal V2-envelope FullCoT rows")
    return result


def _load_clean_targets(
    path: str | Path,
    *,
    split: str,
) -> dict[str, StageIILegacyDirectTargetV1]:
    """Validate immutable clean targets and index their parent identities."""

    result: dict[str, StageIILegacyDirectTargetV1] = {}
    for payload in _iter_jsonl(path):
        target = StageIILegacyDirectTargetV1.model_validate(payload)
        if target.split != split:
            raise ValueError(
                f"clean target split drift: expected={split} got={target.split}"
            )
        if target.legacy_parent_id in result:
            raise ValueError(
                f"duplicate clean target parent: {target.legacy_parent_id}"
            )
        result[target.legacy_parent_id] = target
    if not result:
        raise ValueError(f"clean target ledger is empty: {path}")
    return result


def _validate_target_split_disjointness(
    targets: Mapping[str, Mapping[str, StageIILegacyDirectTargetV1]],
) -> None:
    """Reject any parent or split-group leakage across clean target splits."""

    train = targets["train"]
    validation = targets["validation"]
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(f"clean target parent crosses train/validation: {sorted(overlap)[:3]}")
    train_groups = {target.split_group_id for target in train.values()}
    validation_groups = {target.split_group_id for target in validation.values()}
    group_overlap = train_groups & validation_groups
    if group_overlap:
        raise ValueError(
            "clean target split_group crosses train/validation: "
            f"{sorted(group_overlap)[:3]}"
        )


def _index_unique(path: str | Path, key: str) -> dict[str, dict[str, Any]]:
    """Read one raw-sidecar JSONL artifact into a unique identity index."""

    result: dict[str, dict[str, Any]] = {}
    for payload in _iter_jsonl(path):
        identity = str(payload.get(key) or "")
        if not identity or identity in result:
            raise ValueError(f"missing or duplicate {key}: {identity!r}")
        result[identity] = payload
    return result


def _validate_selected_lineage(
    view: StageIFullCoTViewV2,
    *,
    clean_target: StageIILegacyDirectTargetV1,
    raw_sidecar: Mapping[str, Any],
    registry,
    max_source_step_m: float,
) -> None:
    """Prove the V4 source, raw sidecar, and clean target are identical truth."""

    if view.sample_id != clean_target.legacy_parent_id:
        raise ValueError("V4 clean target parent identity mismatch")
    for field in ("source_group_id", "split_group_id", "dataset_source"):
        if str(getattr(view, field)) != str(getattr(clean_target, field)):
            raise ValueError(f"V4 clean target {field} mismatch")
        if str(raw_sidecar.get(field) or "") not in {"", str(getattr(view, field))}:
            raise ValueError(f"prebuilt sidecar {field} mismatch")
    if str(raw_sidecar.get("stage_i_sample_id") or "") != view.sample_id:
        raise ValueError("prebuilt sidecar Stage I identity mismatch")
    spec = registry.dataset(view.dataset_source)
    if spec.status != "enabled":
        raise ValueError("selected V4 dataset is disabled by trajectory registry")
    source_format = str(raw_sidecar.get("source_format") or "")
    trajectory_source = str(raw_sidecar.get("trajectory_source") or "")
    yaw_source = str(raw_sidecar.get("yaw_source") or "")
    target_policy = clean_target.trajectory_target.policy
    if source_format != target_policy.source_format:
        raise ValueError("clean target/source sidecar format mismatch")
    if trajectory_source != target_policy.trajectory_source:
        raise ValueError("clean target/source sidecar trajectory source mismatch")
    if yaw_source != target_policy.yaw_source:
        raise ValueError("clean target/source sidecar yaw source mismatch")
    build = build_stage_ii_trajectory_target_v1(
        sample_id=view.sample_id,
        dataset_source=view.dataset_source,
        raw_trajectory=raw_sidecar.get("trajectory_ego_t") or [],
        action_mask=raw_sidecar.get("action_mask"),
        source_format=source_format,
        trajectory_source=trajectory_source,
        allowed_trajectory_sources=spec.allowed_trajectory_sources,
        yaw_source=yaw_source,
        max_source_step_m=max_source_step_m,
    )
    if not build.accepted or build.target is None:
        reason = build.qc.reason_codes[0] if build.qc.reason_codes else "unknown"
        raise ValueError(f"clean target failed independent source QC: {reason}")
    if build.target.target_digest != clean_target.trajectory_target.target_digest:
        raise ValueError("clean target digest differs from independently rebuilt raw target")


def _attach_clean_lineage(
    raw_sidecar: Mapping[str, Any],
    *,
    clean_target: StageIILegacyDirectTargetV1,
    max_source_step_m: float,
) -> dict[str, Any]:
    """Attach only audit stamps to an otherwise raw target sidecar."""

    result = dict(raw_sidecar)
    result.update({
        "clean_target_profile_id": PROFILE_ID,
        "clean_target_profile_version": PROFILE_VERSION,
        "clean_target_record_digest": clean_target.record_digest,
        "clean_target_trajectory_qc_digest": clean_target.trajectory_qc_digest,
        "clean_target_trajectory_digest": clean_target.trajectory_target.target_digest,
        "clean_target_max_source_step_m": max_source_step_m,
    })
    return result


def _coverage_drop(
    target: StageIILegacyDirectTargetV1,
    *,
    split: str,
) -> dict[str, Any]:
    """Record a target accepted historically but absent from the V4 source."""

    payload = {
        "schema_version": "urbannav_stage_ii_v4_clean_coverage_drop_v1",
        "split": split,
        "parent_sample_id": target.legacy_parent_id,
        "source_group_id": target.source_group_id,
        "split_group_id": target.split_group_id,
        "dataset_source": target.dataset_source,
        "reason_code": "stage_i_v4_source_missing",
        "clean_target_record_digest": target.record_digest,
    }
    payload["drop_digest"] = stage_ii_digest_v1(payload)
    return payload


def _iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects from one immutable JSONL input."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row must be an object: {source}")
                yield payload


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one deterministic compact JSONL artifact."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 for one frozen artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse V4 clean-input materialization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-jsonl", required=True, type=Path)
    parser.add_argument("--clean-train-targets-jsonl", required=True, type=Path)
    parser.add_argument(
        "--clean-validation-targets-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument("--prebuilt-train-sidecars-jsonl", required=True, type=Path)
    parser.add_argument(
        "--prebuilt-validation-sidecars-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument("--trajectory-source-registry-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-source-step-m",
        type=float,
        default=DEFAULT_MAX_SOURCE_STEP_M,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Materialize V4 clean inputs and print their deterministic report."""

    args = parse_args(argv)
    report = build_stage_ii_v4_clean_inputs(
        stage_i_jsonl=args.stage_i_jsonl,
        clean_train_targets_jsonl=args.clean_train_targets_jsonl,
        clean_validation_targets_jsonl=args.clean_validation_targets_jsonl,
        prebuilt_train_sidecars_jsonl=args.prebuilt_train_sidecars_jsonl,
        prebuilt_validation_sidecars_jsonl=args.prebuilt_validation_sidecars_jsonl,
        trajectory_source_registry_json=args.trajectory_source_registry_json,
        output_dir=args.output_dir,
        max_source_step_m=args.max_source_step_m,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
