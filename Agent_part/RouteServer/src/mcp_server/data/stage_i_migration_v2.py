"""Migrate validated Stage I V1 FullCoT parents into the V2 envelope."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV1,
    parse_stage_i_training_view_v1,
    stage_i_projection_digest_v1,
)
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.stage_i_views import upgrade_stage_i_full_cot_view_v2

STAGE_I_FULL_COT_MIGRATION_V2_REPORT_VERSION = (
    "urbannav_stage_i_full_cot_migration_v1_to_v2_report_v1")


STAGE_I_FULL_COT_SELECTION_V1_REPORT_VERSION = (
    "urbannav_stage_i_full_cot_selection_v1_report_v1")


def extract_stage_i_full_cot_manifest_v1(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one mixed V1 source and select only canonical FullCoT rows.

    The selector does not reinterpret Decision rows or mutate accepted FullCoT
    payloads. Every source row must parse as a V1 Stage I view; non-FullCoT
    tasks are counted and skipped before the strict V2 migration step.

    Args:
        input_jsonl: Mixed canonical Stage I V1 manifest.
        output_jsonl: Fresh pure V1 FullCoT destination.
        report_json: Optional deterministic selection report destination.

    Returns:
        Input/output fingerprints and per-task source counts.

    Raises:
        ValueError: If paths alias, FullCoT IDs repeat, or no FullCoT exists.
    """

    source = Path(input_jsonl).expanduser().resolve()
    destination = Path(output_jsonl).expanduser().resolve()
    if source == destination:
        raise ValueError("Stage I FullCoT selection requires a fresh output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for raw in iter_jsonl(source):
                parsed = parse_stage_i_training_view_v1(raw)
                counts["input_rows"] += 1
                counts[f"task:{parsed.task_type}"] += 1
                if not isinstance(parsed, StageIFullCoTViewV1):
                    counts["skipped_non_full_cot"] += 1
                    continue
                if parsed.sample_id in seen_ids:
                    raise ValueError(
                        f"duplicate V1 FullCoT sample_id: {parsed.sample_id}")
                seen_ids.add(parsed.sample_id)
                handle.write(
                    json.dumps(
                        raw,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ) + "\n")
                counts["output_full_cot_rows"] += 1
        if not counts["output_full_cot_rows"]:
            raise ValueError("Stage I mixed source contains no V1 FullCoT rows")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": STAGE_I_FULL_COT_SELECTION_V1_REPORT_VERSION,
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
            "schema_version": "urbannav_stage_i_training_view_v1",
        },
        "output": {
            "path": str(destination),
            "sha256": _sha256(destination),
            "schema_version": "urbannav_stage_i_training_view_v1",
            "task_type": "full_cot",
            "rows": counts["output_full_cot_rows"],
        },
        "counts": dict(sorted(counts.items())),
        "selection_contract": {
            "all_input_rows_validated": True,
            "accepted_task": "full_cot",
            "payload_mutation": False,
            "reasoning_dropout_materialized": False,
        },
    }
    if report_json is not None:
        report_path = Path(report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def upgrade_stage_i_full_cot_manifest_v2(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
    image_prefix_maps: Sequence[tuple[str | Path, str | Path]] = (),
) -> dict[str, Any]:
    """Upgrade a V1 FullCoT manifest without creating Decision mirrors.

    Every input row is first validated against the immutable V1 contract.  A
    Decision or VQA row therefore fails closed instead of being reinterpreted
    as a V2 parent.  The output remains one row per physical FullCoT parent;
    reasoning dropout is still derived only after epoch routing.

    Args:
        input_jsonl: Canonical V1 FullCoT manifest.
        output_jsonl: Fresh V2 FullCoT manifest destination.
        report_json: Optional deterministic migration report destination.
        image_prefix_maps: Explicit anchored old-to-new image-path prefixes.

    Returns:
        Input/output fingerprints and migrated parent count.

    Raises:
        ValueError: If the input is empty, duplicated, non-FullCoT, or aliases
            the output path.
    """

    source = Path(input_jsonl).expanduser().resolve()
    destination = Path(output_jsonl).expanduser().resolve()
    if source == destination:
        raise ValueError("Stage I V2 migration requires a fresh output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    prefix_maps = _normalize_prefix_maps(image_prefix_maps)
    seen_ids: set[str] = set()
    rows = 0
    rewritten_image_references = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for line_number, raw in enumerate(iter_jsonl(source), start=1):
                parsed = parse_stage_i_training_view_v1(raw)
                if not isinstance(parsed, StageIFullCoTViewV1):
                    raise ValueError(
                        "Stage I V2 migration accepts only V1 FullCoT rows; "
                        f"found {parsed.task_type!r} at line {line_number}")
                if parsed.sample_id in seen_ids:
                    raise ValueError(
                        f"duplicate V1 FullCoT sample_id: {parsed.sample_id}")
                seen_ids.add(parsed.sample_id)
                parsed, rewritten = _rewrite_image_paths(parsed, prefix_maps)
                rewritten_image_references += rewritten
                upgraded = upgrade_stage_i_full_cot_view_v2(parsed)
                handle.write(
                    json.dumps(
                        upgraded.model_dump(mode="json", exclude_none=True),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ) + "\n")
                rows += 1
        if rows == 0:
            raise ValueError("Stage I V2 migration input is empty")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": STAGE_I_FULL_COT_MIGRATION_V2_REPORT_VERSION,
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
            "schema_version": "urbannav_stage_i_training_view_v1",
        },
        "output": {
            "path": str(destination),
            "sha256": _sha256(destination),
            "schema_version": "urbannav_stage_i_training_view_v2",
            "rows": rows,
        },
        "migration_contract": {
            "accepted_input_task":
            "full_cot",
            "visible_input_changes":
            False,
            "reasoning_dropout_materialized":
            False,
            "decision_loss_policy":
            "all_canonical_tokens",
            "image_prefix_maps": [{
                "old": str(old),
                "new": str(new)
            } for old, new in prefix_maps],
            "rewritten_image_references":
            rewritten_image_references,
        },
    }
    if report_json is not None:
        report_path = Path(report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _normalize_prefix_maps(
    mappings: Sequence[tuple[str | Path, str | Path]],
) -> tuple[tuple[Path, Path], ...]:
    """Normalize explicit anchored image-prefix rewrites."""

    normalized: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for old_value, new_value in mappings:
        old = Path(old_value).expanduser()
        new = Path(new_value).expanduser().resolve()
        if not old.is_absolute() or not new.is_absolute():
            raise ValueError("Stage I image prefix maps must be absolute")
        if old in seen:
            raise ValueError(f"duplicate Stage I image prefix: {old}")
        seen.add(old)
        normalized.append((old, new))
    return tuple(
        sorted(normalized, key=lambda item: len(item[0].parts), reverse=True))


def _rewrite_image_paths(
    parsed: StageIFullCoTViewV1,
    prefix_maps: Sequence[tuple[Path, Path]],
) -> tuple[StageIFullCoTViewV1, int]:
    """Apply explicit path maps and revalidate the V1 projection digest."""

    if not prefix_maps:
        return parsed, 0
    rewritten: list[str] = []
    changed = 0
    for image_value in parsed.images:
        recorded = Path(image_value).expanduser()
        resolved = recorded
        if not recorded.is_file():
            for old, new in prefix_maps:
                try:
                    suffix = recorded.relative_to(old)
                except ValueError:
                    continue
                resolved = new / suffix
                changed += 1
                break
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Stage I mapped image does not exist: {image_value!r} -> {resolved}"
            )
        rewritten.append(str(resolved.resolve()))
    payload = parsed.model_dump(mode="json", exclude_none=True)
    payload["images"] = rewritten
    payload.pop("projection_digest", None)
    payload["projection_digest"] = stage_i_projection_digest_v1(payload)
    migrated = parse_stage_i_training_view_v1(payload)
    if not isinstance(migrated, StageIFullCoTViewV1):
        raise RuntimeError(
            "Stage I image rewrite changed the FullCoT task type")
    return migrated, changed


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "STAGE_I_FULL_COT_MIGRATION_V2_REPORT_VERSION",
    "STAGE_I_FULL_COT_SELECTION_V1_REPORT_VERSION",
    "extract_stage_i_full_cot_manifest_v1",
    "upgrade_stage_i_full_cot_manifest_v2",
]
