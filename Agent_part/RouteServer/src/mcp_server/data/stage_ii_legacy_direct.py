"""Build canonical trajectory targets for the legacy direct-latent profile."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV1,
    parse_stage_i_training_view_v1,
)

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_LEGACY_DIRECT_PROFILE_ID,
    STAGE_II_LEGACY_DIRECT_PROFILE_VERSION,
    STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION,
    StageIILegacyDirectTargetV1,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_ii_sources import (
    StageIISourceAdapterErrorV1,
    adapt_stage_ii_trajectory_source_row_v1,
    load_stage_ii_trajectory_source_registry_v1,
    stage_ii_source_row_identity_v1,
)
from mcp_server.data.stage_ii_trajectory import (
    build_stage_ii_trajectory_target_v1,
)


STAGE_II_LEGACY_DIRECT_BUILD_REPORT_VERSION = (
    "urbannav_stage_ii_legacy_direct_build_report_v2"
)
STAGE_II_LEGACY_DIRECT_SELECTION_REPORT_VERSION = (
    "urbannav_stage_ii_legacy_direct_selection_report_v1"
)
LEGACY_FULL_COT_REASONING_MODE = "legacy_full_cot"
LEGACY_THINK_FINISH_RESPONSE_CONTRACT = "legacy_think_finish"
STAGE_II_LEGACY_DIRECT_TRAJECTORY_ADMISSION_POLICY_ID = (
    "urbannav_stage_ii_legacy_direct_source_qc_v2"
)
STAGE_II_LEGACY_DIRECT_TRAJECTORY_ADMISSION_POLICY_VERSION = "2.0"
STAGE_II_LEGACY_DIRECT_MAX_SOURCE_STEP_M = 3.0


def build_stage_ii_legacy_direct_selection_v1(
    *,
    source_jsonl: str | Path,
    stage_i_split_manifests: Mapping[str, str | Path],
    source_dataset_map: Mapping[str, str],
    registry_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Recover original legacy prompts for the exact Stage I split allowlist.

    The current Stage I train/validation manifests remain the sole split
    authority.  This function scans the frozen legacy combined source only to
    recover the original model-visible system/user/images contract.  It never
    reads the historical legacy train/validation/test split files.
    """

    if set(stage_i_split_manifests) != {"train", "validation"}:
        raise ValueError(
            "stage_i_split_manifests must contain exactly train and validation"
        )
    registry = load_stage_ii_trajectory_source_registry_v1(registry_json)
    registered = {str(item.dataset_source) for item in registry.datasets}
    invalid_targets = set(source_dataset_map.values()) - registered
    if invalid_targets:
        raise ValueError(
            f"source_dataset_map has unknown datasets: {sorted(invalid_targets)}"
        )
    source_path = Path(source_jsonl).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    requested, _ordered_keys, split_counts = _load_stage_i_parent_selection(
        stage_i_split_manifests
    )
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    selected_semantics: dict[tuple[str, str], dict[str, Any]] = {}
    selected_priority: dict[tuple[str, str], tuple[int, int]] = {}
    counts: Counter[str] = Counter()
    duplicate_examples: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as source_handle:
        for line_no, raw_line in enumerate(source_handle, start=1):
            if not raw_line.strip():
                continue
            raw_row = json.loads(raw_line)
            source_name = _legacy_source_name(raw_row)
            dataset_source = source_dataset_map.get(source_name)
            if dataset_source is None:
                continue
            spec = registry.dataset(dataset_source)
            try:
                source_group_id, _source_split_group = (
                    stage_ii_source_row_identity_v1(raw_row, spec)
                )
                adapted = adapt_stage_ii_trajectory_source_row_v1(raw_row, spec)
            except StageIISourceAdapterErrorV1:
                continue
            key = (dataset_source, source_group_id)
            request = requested.get(key)
            if request is None:
                continue
            candidate = _normalize_selected_legacy_row(
                raw_row,
                request=request,
                source_name=source_name,
                source_line_no=line_no,
            )
            semantics = _legacy_selection_semantics(
                candidate,
                adapted_trajectory={
                    "raw_trajectory": adapted.raw_trajectory,
                    "action_mask": adapted.action_mask,
                    "source_format": adapted.source_format,
                    "trajectory_source": adapted.trajectory_source,
                    "yaw_source": adapted.yaw_source,
                    "mask_provenance": adapted.mask_provenance,
                    "origin_provenance": adapted.origin_provenance,
                    "source_horizon_seconds": adapted.source_horizon_seconds,
                },
            )
            priority = (_legacy_source_priority(source_name), line_no)
            existing = selected.get(key)
            if existing is not None:
                if selected_semantics[key] != semantics:
                    raise ValueError(
                        "conflicting duplicate legacy prompt source: "
                        f"dataset={dataset_source} source_group_id={source_group_id}"
                    )
                counts["matched_duplicate_equivalent"] += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append({
                        "dataset_source": dataset_source,
                        "source_group_id": source_group_id,
                        "kept_source": existing["source_dataset"],
                        "kept_line": existing["sidewalk_meta"]["source_record"]["line_no"],
                        "duplicate_source": source_name,
                        "duplicate_line": line_no,
                    })
                if priority >= selected_priority[key]:
                    continue
            selected[key] = candidate
            selected_semantics[key] = semantics
            selected_priority[key] = priority

    missing = [
        {
            "dataset_source": key[0],
            "source_group_id": key[1],
            "split": requested[key]["split"],
            "stage_i_parent_sample_id": requested[key]["parent_sample_id"],
        }
        for key in requested
        if key not in selected
    ]
    output_paths: dict[str, Path] = {}
    output_digests: dict[str, str] = {}
    for split in ("train", "validation"):
        path = destination / f"legacy_direct_selected_{split}.jsonl"
        rows = sorted(
            (
                row
                for key, row in selected.items()
                if requested[key]["split"] == split
            ),
            key=lambda row: int(
                row["sidewalk_meta"]["source_record"]["line_no"]
            ),
        )
        _write_jsonl(path, rows)
        output_paths[split] = path
        output_digests[split] = _file_sha256(path)
        counts[f"selected:{split}"] = len(rows)

    expected = sum(split_counts.values())
    selected_count = len(selected)
    report = {
        "schema_version": STAGE_II_LEGACY_DIRECT_SELECTION_REPORT_VERSION,
        "profile_id": STAGE_II_LEGACY_DIRECT_PROFILE_ID,
        "profile_version": STAGE_II_LEGACY_DIRECT_PROFILE_VERSION,
        "split_authority": "explicit_stage_i_train_validation_manifests",
        "source_jsonl": str(source_path),
        "stage_i_split_manifests": {
            split: str(Path(path).expanduser().resolve())
            for split, path in stage_i_split_manifests.items()
        },
        "split_counts": split_counts,
        "expected_rows": expected,
        "selected_rows": selected_count,
        "missing_rows": len(missing),
        "missing_examples": missing[:20],
        "counts": dict(sorted(counts.items())),
        "equivalent_duplicate_examples": duplicate_examples,
        "outputs": {
            split: {
                "path": str(output_paths[split]),
                "sha256": output_digests[split],
                "rows": int(counts[f"selected:{split}"]),
            }
            for split in ("train", "validation")
        },
        "historical_legacy_split_manifests_read": False,
        "hidden_test_manifest_read": False,
        "output_ordering": "combined_source_line_ascending",
        "complete": selected_count == expected and not missing,
    }
    report_path = destination / "legacy_direct_selection_report.json"
    report["outputs"]["report"] = {"path": str(report_path)}
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_stage_ii_legacy_direct_targets_v1(
    *,
    source_jsonl: str | Path,
    split_manifests: Mapping[str, str | Path],
    source_dataset_map: Mapping[str, str],
    registry_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build train/validation targets without reading the held-out test split.

    The legacy split manifests contain the exact one-based line number in the
    frozen combined source JSONL.  This builder uses that lineage pointer to
    recover dataset trajectory truth, validates the original visible input,
    and delegates all geometry conversion to the existing Stage II registry
    and target builder.
    """

    if set(split_manifests) != {"train", "validation"}:
        raise ValueError("split_manifests must contain exactly train and validation")
    registry = load_stage_ii_trajectory_source_registry_v1(registry_json)
    registered = {str(item.dataset_source) for item in registry.datasets}
    invalid_targets = set(source_dataset_map.values()) - registered
    if invalid_targets:
        raise ValueError(f"source_dataset_map has unknown datasets: {sorted(invalid_targets)}")

    source_path = Path(source_jsonl).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    requested, split_counts = _load_requested_rows(
        split_manifests,
        source_dataset_map=source_dataset_map,
    )
    target_paths = {
        split: destination / f"legacy_direct_targets_{split}.jsonl"
        for split in ("train", "validation")
    }
    target_handles = {
        split: path.open("w", encoding="utf-8")
        for split, path in target_paths.items()
    }
    qc_path = destination / "legacy_direct_trajectory_qc.jsonl"
    drops_path = destination / "legacy_direct_drops.jsonl"
    qc_handle = qc_path.open("w", encoding="utf-8")
    drops_handle = drops_path.open("w", encoding="utf-8")
    counts: Counter[str] = Counter()
    seen_lines: set[int] = set()
    try:
        with source_path.open("r", encoding="utf-8") as source_handle:
            for line_no, raw_line in enumerate(source_handle, start=1):
                request = requested.get(line_no)
                if request is None:
                    continue
                if line_no in seen_lines:
                    raise ValueError(f"duplicate requested source line: {line_no}")
                seen_lines.add(line_no)
                raw_row = json.loads(raw_line)
                _validate_source_join(raw_row, request)
                dataset_source = str(request["dataset_source"])
                spec = registry.dataset(dataset_source)
                try:
                    adapted = adapt_stage_ii_trajectory_source_row_v1(raw_row, spec)
                    result = build_stage_ii_trajectory_target_v1(
                        sample_id=str(request["legacy_parent_id"]),
                        dataset_source=dataset_source,
                        raw_trajectory=adapted.raw_trajectory,
                        action_mask=adapted.action_mask,
                        source_format=adapted.source_format,
                        trajectory_source=adapted.trajectory_source,
                        allowed_trajectory_sources=spec.allowed_trajectory_sources,
                        yaw_source=adapted.yaw_source,
                        max_source_step_m=STAGE_II_LEGACY_DIRECT_MAX_SOURCE_STEP_M,
                    )
                except StageIISourceAdapterErrorV1 as exc:
                    _write_jsonl_row(qc_handle, {
                        "sample_id": request["legacy_parent_id"],
                        "status": "rejected",
                        "reason_codes": [exc.code],
                        "detail": str(exc),
                    })
                    _write_drop(drops_handle, request, exc.code, str(exc))
                    counts[f"drop:{exc.code}"] += 1
                    continue

                _write_jsonl_row(qc_handle, result.qc.model_dump(mode="json"))
                if not result.accepted or result.target is None:
                    reason = str(result.qc.reason_codes[0])
                    _write_drop(drops_handle, request, reason, "trajectory target rejected by deterministic QC")
                    counts[f"drop:{reason}"] += 1
                    continue

                payload = {
                    "schema_version": STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION,
                    "profile_id": STAGE_II_LEGACY_DIRECT_PROFILE_ID,
                    "profile_version": STAGE_II_LEGACY_DIRECT_PROFILE_VERSION,
                    "sample_id": f"stage-ii-legacy-direct:{request['legacy_parent_id']}",
                    "legacy_parent_id": request["legacy_parent_id"],
                    "split": request["split"],
                    "source_group_id": adapted.source_group_id,
                    "split_group_id": request["split_group_id"],
                    "dataset_source": dataset_source,
                    "legacy_source_name": request["legacy_source_name"],
                    "legacy_source_line_no": line_no,
                    "legacy_split_row_digest": request["legacy_split_row_digest"],
                    "legacy_source_row_digest": stage_ii_digest_v1(raw_row),
                    "legacy_visible_input_digest": request["legacy_visible_input_digest"],
                    "trajectory_target": result.target.model_dump(mode="json"),
                    "trajectory_qc_digest": result.qc.qc_digest,
                }
                payload["record_digest"] = stage_ii_digest_v1(payload)
                typed = StageIILegacyDirectTargetV1.model_validate(payload)
                _write_jsonl_row(
                    target_handles[str(request["split"])],
                    typed.model_dump(mode="json"),
                )
                counts[f"accepted:{request['split']}"] += 1
                counts[f"dataset:{dataset_source}"] += 1
    finally:
        for handle in target_handles.values():
            handle.close()
        qc_handle.close()
        drops_handle.close()

    missing_lines = sorted(set(requested) - seen_lines)
    if missing_lines:
        raise ValueError(
            f"combined source is missing {len(missing_lines)} requested rows; "
            f"first={missing_lines[:5]}"
        )
    expected = len(requested)
    accepted = counts["accepted:train"] + counts["accepted:validation"]
    rejected = expected - accepted
    complete = expected > 0 and len(seen_lines) == expected
    report = {
        "schema_version": STAGE_II_LEGACY_DIRECT_BUILD_REPORT_VERSION,
        "profile_id": STAGE_II_LEGACY_DIRECT_PROFILE_ID,
        "profile_version": STAGE_II_LEGACY_DIRECT_PROFILE_VERSION,
        "source_jsonl": str(source_path),
        "registry_json": str(Path(registry_json).expanduser().resolve()),
        "source_dataset_map": dict(sorted(source_dataset_map.items())),
        "split_counts": split_counts,
        "requested_rows": expected,
        "accepted_rows": accepted,
        "rejected_by_trajectory_qc": rejected,
        "trajectory_admission_policy": {
            "policy_id": STAGE_II_LEGACY_DIRECT_TRAJECTORY_ADMISSION_POLICY_ID,
            "policy_version": STAGE_II_LEGACY_DIRECT_TRAJECTORY_ADMISSION_POLICY_VERSION,
            "max_source_step_m": STAGE_II_LEGACY_DIRECT_MAX_SOURCE_STEP_M,
        },
        "counts": dict(sorted(counts.items())),
        "outputs": {
            "train_targets": str(target_paths["train"]),
            "validation_targets": str(target_paths["validation"]),
            "trajectory_qc": str(qc_path),
            "drops": str(drops_path),
        },
        "hidden_test_read": False,
        "complete": complete,
        "passed": complete and accepted > 0,
    }
    report_path = destination / "legacy_direct_build_report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def legacy_visible_input_digest_v1(row: Mapping[str, Any]) -> str:
    """Digest only original model-visible legacy system/user input."""

    messages = row.get("messages")
    images = row.get("images")
    if not isinstance(messages, list) or not isinstance(images, list) or not images:
        raise ValueError("legacy row requires messages and images")
    visible: list[dict[str, str]] = []
    assistant_seen = False
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("legacy message must be a mapping")
        role = str(message.get("role") or "")
        content = message.get("content")
        if role == "assistant":
            assistant_seen = True
            continue
        if assistant_seen:
            raise ValueError("legacy visible message cannot follow assistant output")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError("legacy visible messages must be system/user text")
        visible.append({"role": role, "content": content})
    if [message["role"] for message in visible] != ["system", "user"]:
        raise ValueError("legacy direct profile requires one system and one user turn")
    if not assistant_seen:
        raise ValueError("legacy source row must retain its auditable assistant answer")
    return stage_ii_digest_v1({"messages": visible, "images": list(images)})


def _load_stage_i_parent_selection(
    split_manifests: Mapping[str, str | Path],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, list[tuple[str, str]]],
    dict[str, int],
]:
    """Load exact FullCoT parents and preserve split-manifest row order."""

    requested: dict[tuple[str, str], dict[str, Any]] = {}
    ordered: dict[str, list[tuple[str, str]]] = {
        "train": [],
        "validation": [],
    }
    split_counts: dict[str, int] = {}
    parent_owner: dict[str, str] = {}
    split_group_owner: dict[str, str] = {}
    for split in ("train", "validation"):
        path = Path(split_manifests[split]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                typed = parse_stage_i_training_view_v1(json.loads(line))
                if not isinstance(typed, StageIFullCoTViewV1):
                    raise ValueError(
                        f"{path}:{line_no} is not a Stage I FullCoT parent"
                    )
                parent_id = str(typed.sample_id)
                if parent_id in parent_owner:
                    raise ValueError(
                        f"Stage I parent appears in {parent_owner[parent_id]} and {split}: "
                        f"{parent_id}"
                    )
                parent_owner[parent_id] = split
                split_group = str(typed.split_group_id)
                previous_split = split_group_owner.setdefault(split_group, split)
                if previous_split != split:
                    raise ValueError(
                        f"Stage I split group appears in {previous_split} and {split}: "
                        f"{split_group}"
                    )
                key = (str(typed.dataset_source), str(typed.source_group_id))
                if key in requested:
                    raise ValueError(f"duplicate Stage I source identity: {key}")
                requested[key] = {
                    "split": split,
                    "parent_sample_id": parent_id,
                    "source_group_id": str(typed.source_group_id),
                    "split_group_id": split_group,
                    "dataset_source": str(typed.dataset_source),
                    "parent_source_fingerprint": str(
                        typed.parent_source_fingerprint
                    ),
                }
                ordered[split].append(key)
                count += 1
        split_counts[split] = count
    return requested, ordered, split_counts


def _normalize_selected_legacy_row(
    raw_row: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    source_name: str,
    source_line_no: int,
) -> dict[str, Any]:
    """Build one original-prompt row under the current Stage I split authority."""

    messages = raw_row.get("messages")
    images = raw_row.get("images")
    if not isinstance(messages, list) or not isinstance(images, list):
        raise ValueError("selected legacy source requires messages and images")
    normalized_messages = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("selected legacy message must be an object")
        role = str(message.get("role") or "")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(
            content, str
        ):
            raise ValueError("selected legacy message role/content drift")
        normalized_messages.append({"role": role, "content": content})
    normalized = {
        "id": str(request["parent_sample_id"]),
        "messages": normalized_messages,
        "images": list(images),
        "direction_name": str(raw_row.get("direction_name") or ""),
        "direction_choice": str(raw_row.get("direction_choice") or ""),
        "direction_choice_label": int(raw_row.get("direction_choice_label")),
        "direction_bin_mode": str(raw_row.get("direction_bin_mode") or ""),
        "direction_profile_id": raw_row.get("direction_profile_id"),
        "direction_profile_version": raw_row.get("direction_profile_version"),
        "response_contract": LEGACY_THINK_FINISH_RESPONSE_CONTRACT,
        "reasoning_mode": LEGACY_FULL_COT_REASONING_MODE,
        "required_loss_scale_strategy": "ignore_empty_think",
        "action_supervision_location": "finish.direction",
        "source_dataset": source_name,
        "sidewalk_meta": {
            "experiment_group": {
                "schema_version": "stage_ii_legacy_direct_selection_v1",
                "source": source_name,
                "clip_id": str(request["split_group_id"]),
                "group_key": str(request["split_group_id"]),
                "split": (
                    "train_pool"
                    if request["split"] == "train"
                    else "val"
                ),
                "split_authority": "explicit_stage_i_manifest",
            },
            "image_roles": {
                "schema_version": "sidewalk_multiframe_image_roles_v1",
                "roles": _resolve_legacy_image_roles(images),
            },
            "source_record": {
                "schema_version": "stage_ii_legacy_source_pointer_v1",
                "line_no": int(source_line_no),
                "original_id": str(
                    raw_row.get("id")
                    or raw_row.get("sample_id")
                    or request["source_group_id"]
                ),
            },
            "stage_i_parent": {
                "sample_id": str(request["parent_sample_id"]),
                "source_group_id": str(request["source_group_id"]),
                "source_fingerprint": str(
                    request["parent_source_fingerprint"]
                ),
            },
        },
    }
    if not normalized["direction_name"] or not normalized["direction_choice"]:
        raise ValueError("selected legacy source has no direction label")
    legacy_visible_input_digest_v1(normalized)
    return normalized


def _legacy_selection_semantics(
    row: Mapping[str, Any],
    *,
    adapted_trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Return fields that must agree across main/extra source duplicates."""

    meta = row["sidewalk_meta"]
    return {
        "visible_input_digest": legacy_visible_input_digest_v1(row),
        "direction_name": row["direction_name"],
        "direction_choice": row["direction_choice"],
        "direction_choice_label": row["direction_choice_label"],
        "direction_bin_mode": row["direction_bin_mode"],
        "direction_profile_id": row.get("direction_profile_id"),
        "direction_profile_version": row.get("direction_profile_version"),
        "image_roles": meta["image_roles"],
        "adapted_trajectory": dict(adapted_trajectory),
    }


def _legacy_source_name(row: Mapping[str, Any]) -> str:
    """Resolve the frozen combined-source family without dataset guessing."""

    meta = row.get("sidewalk_meta")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    if not isinstance(meta, Mapping):
        meta = {}
    for value in (
        meta.get("direction_sample_source"),
        meta.get("source_dataset"),
        meta.get("combined_dataset_part"),
        row.get("source_dataset"),
    ):
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("selected legacy source family is missing")


def _legacy_source_priority(source_name: str) -> int:
    """Prefer the historical main source when main/extra are equivalent."""

    priorities = {
        "foxglove": 0,
        "sidewalk_main_multiframe": 0,
        "sidewalk_extra_multiframe": 1,
    }
    return priorities.get(source_name, 100)


def _resolve_legacy_image_roles(images: Sequence[Any]) -> list[dict[str, Any]]:
    """Reproduce the frozen 10-image SideWalk or 6-image Foxglove layout."""

    modalities = [_legacy_image_modality(value) for value in images]
    roles: list[dict[str, Any]] = []
    if len(images) == 10:
        if modalities != ["fpv", "bev"] * 5:
            raise ValueError("legacy SideWalk image order drift")
        for index in range(5):
            temporal_role = "current" if index == 4 else "history"
            roles.extend([
                {
                    "modality": "fpv",
                    "temporal_role": temporal_role,
                    "temporal_index": index,
                },
                {
                    "modality": "bev",
                    "temporal_role": temporal_role,
                    "temporal_index": index,
                },
            ])
    elif len(images) == 6:
        if modalities != ["fpv", "fpv", "fpv", "fpv", "fpv", "bev"]:
            raise ValueError("legacy Foxglove image order drift")
        roles.extend([
            {
                "modality": "fpv",
                "temporal_role": "current" if index == 4 else "history",
                "temporal_index": index,
            }
            for index in range(5)
        ])
        roles.append({
            "modality": "bev",
            "temporal_role": "current",
            "temporal_index": 4,
        })
    else:
        raise ValueError(f"unsupported legacy image layout: {len(images)}")
    return roles


def _legacy_image_modality(value: Any) -> str:
    """Classify one frozen image path as FPV or BEV."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("legacy image path is invalid")
    normalized = value.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    if "/bev/" in normalized or "_bev." in name:
        return "bev"
    if "/fpv/" in normalized:
        return "fpv"
    raise ValueError(f"legacy image modality is unknown: {value}")


def _load_requested_rows(
    split_manifests: Mapping[str, str | Path],
    *,
    source_dataset_map: Mapping[str, str],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Load compact lineage metadata and reject overlap or test leakage."""

    requested: dict[int, dict[str, Any]] = {}
    group_splits: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    for split in ("train", "validation"):
        path = Path(split_manifests[split]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_legacy_split = "train_pool" if split == "train" else "val"
        with path.open("r", encoding="utf-8") as handle:
            for manifest_line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("reasoning_mode") != LEGACY_FULL_COT_REASONING_MODE:
                    raise ValueError(f"{path}:{manifest_line_no} reasoning_mode drift")
                if row.get("response_contract") != LEGACY_THINK_FINISH_RESPONSE_CONTRACT:
                    raise ValueError(f"{path}:{manifest_line_no} response_contract drift")
                meta = row.get("sidewalk_meta")
                if not isinstance(meta, Mapping):
                    raise ValueError(f"{path}:{manifest_line_no} missing sidewalk_meta")
                source_record = meta.get("source_record")
                group = meta.get("experiment_group")
                if not isinstance(source_record, Mapping) or not isinstance(group, Mapping):
                    raise ValueError(f"{path}:{manifest_line_no} missing lineage metadata")
                source_line_no = int(source_record.get("line_no") or 0)
                if source_line_no < 1 or source_line_no in requested:
                    raise ValueError(f"duplicate/invalid combined source line: {source_line_no}")
                if str(group.get("split")) != expected_legacy_split:
                    raise ValueError(f"{path}:{manifest_line_no} split stamp mismatch")
                source_name = str(group.get("source") or "")
                dataset_source = source_dataset_map.get(source_name)
                if dataset_source is None:
                    raise ValueError(f"no canonical dataset mapping for legacy source={source_name!r}")
                group_key = str(group.get("group_key") or "")
                if not group_key:
                    raise ValueError(f"{path}:{manifest_line_no} missing group_key")
                previous_split = group_splits.setdefault(group_key, split)
                if previous_split != split:
                    raise ValueError(f"legacy split group overlap: {group_key}")
                parent_id = str(row.get("id") or "")
                if not parent_id:
                    raise ValueError(f"{path}:{manifest_line_no} missing id")
                requested[source_line_no] = {
                    "split": split,
                    "legacy_parent_id": parent_id,
                    "legacy_source_name": source_name,
                    "dataset_source": dataset_source,
                    "split_group_id": group_key,
                    "legacy_split_row_digest": stage_ii_digest_v1(row),
                    "legacy_visible_input_digest": legacy_visible_input_digest_v1(row),
                    "source_contract_digest": _legacy_source_contract_digest(row),
                }
                split_counts[split] += 1
    return requested, dict(split_counts)


def _validate_source_join(raw_row: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    """Require the split row to be an exact projection of its source line."""

    if _legacy_source_contract_digest(raw_row) != request["source_contract_digest"]:
        raise ValueError(
            f"legacy source/split content mismatch at line {request.get('legacy_source_line_no')}"
        )
    if legacy_visible_input_digest_v1(raw_row) != request["legacy_visible_input_digest"]:
        raise ValueError("legacy visible input digest mismatch")


def _legacy_source_contract_digest(row: Mapping[str, Any]) -> str:
    """Digest fields copied verbatim into the legacy experiment manifest."""

    return stage_ii_digest_v1({
        "messages": row.get("messages"),
        "images": row.get("images"),
        "direction_name": row.get("direction_name"),
        "direction_choice": row.get("direction_choice"),
        "direction_choice_label": row.get("direction_choice_label"),
    })


def _write_drop(
    handle: Any,
    request: Mapping[str, Any],
    reason_code: str,
    detail: str,
) -> None:
    """Write one bounded deterministic QC rejection."""

    _write_jsonl_row(handle, {
        "legacy_parent_id": request["legacy_parent_id"],
        "split": request["split"],
        "dataset_source": request["dataset_source"],
        "reason_code": reason_code,
        "detail": detail[:1000],
    })


def _write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    """Write compact deterministic JSONL."""

    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a complete deterministic JSONL artifact."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            _write_jsonl_row(handle, row)


def _file_sha256(path: Path) -> str:
    """Return the byte-level SHA-256 of one immutable artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "STAGE_II_LEGACY_DIRECT_BUILD_REPORT_VERSION",
    "STAGE_II_LEGACY_DIRECT_SELECTION_REPORT_VERSION",
    "build_stage_ii_legacy_direct_selection_v1",
    "build_stage_ii_legacy_direct_targets_v1",
    "legacy_visible_input_digest_v1",
]
