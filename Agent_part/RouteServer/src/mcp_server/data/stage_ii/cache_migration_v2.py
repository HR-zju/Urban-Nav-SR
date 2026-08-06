"""Rebind Stage II context caches after a transport-only Stage I migration."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV1,
    StageIFullCoTViewV2,
    parse_stage_i_training_view_v1,
    parse_stage_i_training_view_v2,
    stage_i_projection_digest_v1,
)
from mcp_server.contracts.stage_ii_cfm import StageIIContextCacheV1
from mcp_server.data.stage_i.migration_v2 import (
    STAGE_I_FULL_COT_MIGRATION_V2_REPORT_VERSION, )
from mcp_server.data.stage_ii.context import (
    build_stage_ii_context_cache_v2,
    stage_ii_projected_input_digest_v1,
)

STAGE_II_CACHE_TRANSPORT_REBIND_V2_REPORT_VERSION = (
    "urbannav_stage_ii_cache_transport_rebind_v2_report_v1")


def rebind_stage_ii_context_cache_manifest_v2(
    *,
    stage_i_v1_jsonl: str | Path,
    stage_i_v2_jsonl: str | Path,
    migration_report_json: str | Path,
    input_cache_jsonl: str | Path,
    output_cache_jsonl: str | Path,
    cache_input_case_jsonl: str | Path | None = None,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Rebind cache identity after an explicitly audited media-path rewrite.

    The operation does not regenerate model output.  It first proves that the
    Stage I migration changed only transport paths, that every old cache is
    bound to the corresponding V1 observation, and that reparsing the same raw
    generation produces byte-identical canonical content.  Only then is the
    cache resealed with the V2 projected-input digest.

    Args:
        stage_i_v1_jsonl: Original validated V1 FullCoT parents.
        stage_i_v2_jsonl: Migrated validated V2 FullCoT parents.
        migration_report_json: Report emitted by the Stage I V2 migrator.
        input_cache_jsonl: Existing Pass-R or Pass-D cache records.
        output_cache_jsonl: Fresh destination for rebound cache records.
        cache_input_case_jsonl: Optional historical open-loop cases whose
            content-equivalent media were used to create the old caches.
        report_json: Optional deterministic audit-report destination.

    Returns:
        Artifact fingerprints and per-kind rebind counts.

    Raises:
        ValueError: If transport equivalence, cache identity, parser output, or
            artifact lineage cannot be proven exactly.
    """

    v1_path = Path(stage_i_v1_jsonl).expanduser().resolve()
    v2_path = Path(stage_i_v2_jsonl).expanduser().resolve()
    migration_path = Path(migration_report_json).expanduser().resolve()
    cache_path = Path(input_cache_jsonl).expanduser().resolve()
    destination = Path(output_cache_jsonl).expanduser().resolve()
    case_path = (Path(cache_input_case_jsonl).expanduser().resolve()
                 if cache_input_case_jsonl is not None else None)
    if cache_path == destination:
        raise ValueError("Stage II cache rebind requires a fresh output path")

    migration = _load_migration_report(migration_path,
                                       v1_path=v1_path,
                                       v2_path=v2_path)
    prefix_maps = _prefix_maps(migration)
    v1_views = _load_full_cot_v1(v1_path)
    v2_views = _load_full_cot_v2(v2_path)
    cache_cases = (_load_cache_input_cases(case_path)
                   if case_path is not None else {})
    if set(v1_views) != set(v2_views):
        raise ValueError("Stage I V1/V2 FullCoT parent sets differ")

    changed_images = 0
    for sample_id in sorted(v1_views):
        changed_images += _validate_transport_only_parent(
            v1_views[sample_id],
            v2_views[sample_id],
            prefix_maps=prefix_maps,
        )
    expected_changed = int(migration["migration_contract"].get(
        "rewritten_image_references", 0))
    if changed_images != expected_changed:
        raise ValueError(
            "Stage I migration rewritten-image count does not match manifests")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    counts: Counter[str] = Counter()
    seen_cache_keys: set[tuple[str, str]] = set()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for raw in _iter_jsonl(cache_path):
                old_cache = StageIIContextCacheV1.model_validate(raw)
                key = (old_cache.sample_id, old_cache.context_kind)
                if key in seen_cache_keys:
                    raise ValueError(
                        f"duplicate Stage II context cache: {key!r}")
                seen_cache_keys.add(key)
                v1_view = v1_views.get(old_cache.sample_id)
                v2_view = v2_views.get(old_cache.sample_id)
                if v1_view is None or v2_view is None:
                    raise ValueError(
                        f"cache parent is absent from Stage I migration: {old_cache.sample_id}"
                    )
                binding_mode, media_pairs = _validate_old_cache_identity(
                    old_cache,
                    v1_view,
                    v2_view,
                    cache_case=cache_cases.get(old_cache.sample_id),
                    prefix_maps=prefix_maps,
                )
                rebound = build_stage_ii_context_cache_v2(
                    v2_view,
                    context_kind=old_cache.context_kind,
                    producer=old_cache.producer,
                    raw_generation=old_cache.raw_generation,
                )
                _validate_parser_stability(old_cache, rebound)
                handle.write(
                    json.dumps(
                        rebound.model_dump(mode="json", exclude_none=True),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ) + "\n")
                counts["input_caches"] += 1
                counts["output_caches"] += 1
                counts[f"context_kind:{old_cache.context_kind}"] += 1
                counts[f"parse_status:{old_cache.parse_status}"] += 1
                counts[f"old_cache_binding:{binding_mode}"] += 1
                counts["content_sha256_verified_media_pairs"] += media_pairs
                if old_cache.projected_input_digest != rebound.projected_input_digest:
                    counts["projected_input_digest_changed"] += 1
                if old_cache.cache_digest != rebound.cache_digest:
                    counts["cache_digest_changed"] += 1
        if not counts["input_caches"]:
            raise ValueError("Stage II cache rebind input is empty")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": STAGE_II_CACHE_TRANSPORT_REBIND_V2_REPORT_VERSION,
        "inputs": {
            "stage_i_v1":
            _artifact(v1_path),
            "stage_i_v2":
            _artifact(v2_path),
            "stage_i_migration_report":
            _artifact(migration_path),
            "context_cache":
            _artifact(cache_path),
            "cache_input_cases":
            (_artifact(case_path) if case_path is not None else None),
        },
        "output": _artifact(destination),
        "contract": {
            "transport_only":
            True,
            "raw_generation_reused":
            True,
            "canonical_parser_output_must_match":
            True,
            "image_prefix_maps": [{
                "old": str(old),
                "new": str(new)
            } for old, new in prefix_maps],
            "verified_parent_count":
            len(v1_views),
            "verified_rewritten_image_references":
            changed_images,
        },
        "counts": dict(sorted(counts.items())),
    }
    if report_json is not None:
        report_path = Path(report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _load_migration_report(
    path: Path,
    *,
    v1_path: Path,
    v2_path: Path,
) -> dict[str, Any]:
    """Load one exact Stage I migration report and bind its artifacts."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get(
            "schema_version") != STAGE_I_FULL_COT_MIGRATION_V2_REPORT_VERSION:
        raise ValueError("unsupported Stage I V2 migration report")
    if value.get("migration_contract",
                 {}).get("visible_input_changes") is not False:
        raise ValueError(
            "cache rebind requires a transport-only Stage I migration")
    if value.get("migration_contract",
                 {}).get("reasoning_dropout_materialized") is not False:
        raise ValueError(
            "cache rebind forbids a materialized dropout migration")
    expected = (
        (value.get("input", {}), v1_path),
        (value.get("output", {}), v2_path),
    )
    for artifact, actual_path in expected:
        if Path(str(artifact.get("path")
                    or "")).expanduser().resolve() != actual_path:
            raise ValueError(
                "Stage I migration report path does not match input")
        if artifact.get("sha256") != _sha256(actual_path):
            raise ValueError(
                "Stage I migration report SHA-256 does not match input")
    return value


def _prefix_maps(report: Mapping[str, Any]) -> tuple[tuple[Path, Path], ...]:
    """Parse absolute anchored path maps from the migration report."""

    result: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    raw_maps = report.get("migration_contract",
                          {}).get("image_prefix_maps", [])
    if not isinstance(raw_maps, list):
        raise ValueError("Stage I migration image_prefix_maps is malformed")
    for item in raw_maps:
        if not isinstance(item, Mapping):
            raise ValueError(
                "Stage I migration image prefix entry is malformed")
        old = Path(str(item.get("old") or "")).expanduser()
        new = Path(str(item.get("new") or "")).expanduser().resolve()
        if not old.is_absolute() or not new.is_absolute() or old in seen:
            raise ValueError("Stage I migration image prefix map is invalid")
        seen.add(old)
        result.append((old, new))
    return tuple(
        sorted(result, key=lambda item: len(item[0].parts), reverse=True))


def _load_full_cot_v1(path: Path) -> dict[str, StageIFullCoTViewV1]:
    """Index unique V1 FullCoT parents."""

    result: dict[str, StageIFullCoTViewV1] = {}
    for raw in _iter_jsonl(path):
        parsed = parse_stage_i_training_view_v1(raw)
        if not isinstance(parsed, StageIFullCoTViewV1):
            raise ValueError("cache rebind accepts only V1 FullCoT parents")
        if parsed.sample_id in result:
            raise ValueError(
                f"duplicate V1 FullCoT parent: {parsed.sample_id}")
        result[parsed.sample_id] = parsed
    if not result:
        raise ValueError("Stage I V1 FullCoT manifest is empty")
    return result


def _load_full_cot_v2(path: Path) -> dict[str, StageIFullCoTViewV2]:
    """Index unique V2 FullCoT parents."""

    result: dict[str, StageIFullCoTViewV2] = {}
    for raw in _iter_jsonl(path):
        parsed = parse_stage_i_training_view_v2(raw)
        if not isinstance(parsed, StageIFullCoTViewV2):
            raise ValueError("cache rebind accepts only V2 FullCoT parents")
        if parsed.sample_id in result:
            raise ValueError(
                f"duplicate V2 FullCoT parent: {parsed.sample_id}")
        result[parsed.sample_id] = parsed
    if not result:
        raise ValueError("Stage I V2 FullCoT manifest is empty")
    return result


def _validate_transport_only_parent(
    v1_view: StageIFullCoTViewV1,
    v2_view: StageIFullCoTViewV2,
    *,
    prefix_maps: Sequence[tuple[Path, Path]],
) -> int:
    """Prove that one parent differs only by version envelope and media path."""

    common_fields = (
        "sample_id",
        "source_group_id",
        "split_group_id",
        "dataset_source",
        "modality_profile_id",
        "native_availability",
        "training_selection",
        "modality_state",
        "instruction_training_role",
        "image_roles",
        "instruction",
        "model_visible_context",
        "evidence_catalog",
        "input_modalities",
        "assistant_text",
        "parent_artifact_kind",
        "parent_sample_id",
        "parent_source_fingerprint",
        "parent_audit_id",
        "reasoning_target",
        "direction_context",
        "direction_supervision",
        "prompt_profile_id",
        "prompt_policy_digest",
    )
    for name in common_fields:
        left = getattr(v1_view, name)
        right = getattr(v2_view, name)
        left_value = left.model_dump(mode="json",
                                     exclude_none=True) if hasattr(
                                         left, "model_dump") else left
        right_value = right.model_dump(mode="json",
                                       exclude_none=True) if hasattr(
                                           right, "model_dump") else right
        if left_value != right_value:
            raise ValueError(
                f"Stage I migration changed semantic field: {name}")

    if len(v1_view.images) != len(v2_view.images):
        raise ValueError("Stage I migration changed image count")
    changed = 0
    for old_value, new_value in zip(v1_view.images, v2_view.images):
        old_resolved = Path(old_value).expanduser().resolve()
        new_resolved = Path(new_value).expanduser().resolve()
        mapped = _mapped_transport_path(old_value, prefix_maps)
        if new_resolved == old_resolved:
            was_rewritten = False
        elif mapped is not None and new_resolved == mapped:
            was_rewritten = True
        else:
            raise ValueError(
                "Stage I migration image path is not covered by prefix map")
        changed += int(was_rewritten)
    source_digest = v2_view.metadata.get("source_stage_i_projection_digest_v1")
    rewritten_v1 = v1_view.model_dump(mode="json", exclude_none=True)
    rewritten_v1["images"] = list(v2_view.images)
    rewritten_v1.pop("projection_digest", None)
    if source_digest != stage_i_projection_digest_v1(rewritten_v1):
        raise ValueError(
            "Stage I V2 source projection digest is not transport-bound")
    return changed


def _mapped_transport_path(
    value: str,
    prefix_maps: Sequence[tuple[Path, Path]],
) -> Path | None:
    """Return the anchored mapped path when one explicit prefix matches."""

    source = Path(value).expanduser()
    for old, new in prefix_maps:
        try:
            suffix = source.relative_to(old)
        except ValueError:
            continue
        return (new / suffix).resolve()
    return None


def _validate_old_cache_identity(
    cache: StageIIContextCacheV1,
    v1_view: StageIFullCoTViewV1,
    v2_view: StageIFullCoTViewV2,
    *,
    cache_case: Mapping[str, Any] | None,
    prefix_maps: Sequence[tuple[Path, Path]],
) -> tuple[str, int]:
    """Require a direct V1 digest or a content-equivalent historical case."""

    expected_input_digest = stage_ii_projected_input_digest_v1(
        images=list(v1_view.images),
        image_roles=list(v1_view.image_roles),
        instruction=v1_view.instruction,
        model_visible_context=v1_view.model_visible_context,
        evidence_catalog=[
            item.model_dump(mode="json", exclude_none=True)
            for item in v1_view.evidence_catalog
        ],
        input_modalities=v1_view.input_modalities.model_dump(mode="json"),
    )
    identities = (
        cache.sample_id == v1_view.sample_id,
        cache.source_group_id == v1_view.source_group_id,
        cache.split_group_id == v1_view.split_group_id,
        cache.dataset_source == v1_view.dataset_source,
        cache.modality_profile_id == v1_view.modality_profile_id,
        cache.parent_source_fingerprint == v1_view.parent_source_fingerprint,
    )
    if not all(identities):
        raise ValueError(
            "Stage II context cache is not bound to the V1 parent")
    if cache.projected_input_digest == expected_input_digest:
        return "stage_i_v1_direct", 0
    if cache_case is None:
        raise ValueError(
            "Stage II context cache is not bound to the V1 parent")
    media_pairs = _validate_cache_case_identity(
        cache,
        v1_view,
        v2_view,
        cache_case,
        prefix_maps=prefix_maps,
    )
    return "historical_case_content_equivalent", media_pairs


def _load_cache_input_cases(path: Path) -> dict[str, dict[str, Any]]:
    """Index historical open-loop cases by their Stage I parent identity."""

    result: dict[str, dict[str, Any]] = {}
    for raw in _iter_jsonl(path):
        public = raw.get("public_context")
        if not isinstance(public, Mapping):
            raise ValueError("historical cache case lacks public_context")
        parent_id = str(public.get("parent_stage_i_sample_id") or "")
        if not parent_id:
            raise ValueError(
                "historical cache case lacks parent_stage_i_sample_id")
        if parent_id in result:
            raise ValueError(
                f"duplicate historical cache case parent: {parent_id}")
        result[parent_id] = raw
    if not result:
        raise ValueError("historical cache input case manifest is empty")
    return result


def _validate_cache_case_identity(
    cache: StageIIContextCacheV1,
    v1_view: StageIFullCoTViewV1,
    v2_view: StageIFullCoTViewV2,
    case: Mapping[str, Any],
    *,
    prefix_maps: Sequence[tuple[Path, Path]],
) -> int:
    """Prove old case media and canonical V2 media are byte-identical."""

    public = case.get("public_context")
    if not isinstance(public, Mapping):
        raise ValueError("historical cache case lacks public_context")
    case_images = case.get("images")
    case_roles = case.get("image_roles")
    expected_modalities = v1_view.input_modalities.model_dump(mode="json")
    identities = (
        str(public.get("parent_stage_i_sample_id") or "") == cache.sample_id,
        str(public.get("parent_source_fingerprint")
            or "") == cache.parent_source_fingerprint,
        str(public.get("projected_input_digest")
            or "") == cache.projected_input_digest,
        case.get("source_group_id") == cache.source_group_id,
        case.get("split_group_id") == cache.split_group_id,
        case.get("dataset_source") == cache.dataset_source,
        case.get("modality_profile_id") == str(cache.modality_profile_id),
        public.get("input_modalities") == expected_modalities,
        case_roles == list(v1_view.image_roles),
        case_roles == list(v2_view.image_roles),
        isinstance(case_images, list),
        len(case_images) == len(v2_view.images) if isinstance(
            case_images, list) else False,
    )
    if not all(identities):
        raise ValueError(
            "historical cache case identity does not match Stage I parent")

    assert isinstance(case_images, list)
    verified = 0
    for case_value, canonical_value in zip(case_images, v2_view.images):
        case_path = _resolve_existing_transport_path(str(case_value),
                                                     prefix_maps)
        canonical_path = Path(canonical_value).expanduser().resolve()
        if not canonical_path.is_file():
            raise FileNotFoundError(
                f"canonical Stage I V2 image does not exist: {canonical_path}")
        if _sha256(case_path) != _sha256(canonical_path):
            raise ValueError(
                "historical cache case media content does not match V2 parent")
        verified += 1
    return verified


def _resolve_existing_transport_path(
    value: str,
    prefix_maps: Sequence[tuple[Path, Path]],
) -> Path:
    """Resolve one historical path directly or through an explicit map."""

    source = Path(value).expanduser()
    if source.is_file():
        return source.resolve()
    mapped = _mapped_transport_path(value, prefix_maps)
    if mapped is None or not mapped.is_file():
        raise FileNotFoundError(
            f"historical cache case image does not exist: {value}")
    return mapped


def _validate_parser_stability(
    old_cache: StageIIContextCacheV1,
    rebound: StageIIContextCacheV1,
) -> None:
    """Reject parser drift while resealing the same raw generation."""

    stable_fields = (
        "context_kind",
        "section_names",
        "producer",
        "raw_generation",
        "canonical_text",
        "parse_status",
        "errors",
    )
    for name in stable_fields:
        left = getattr(old_cache, name)
        right = getattr(rebound, name)
        left_value = left.model_dump(mode="json",
                                     exclude_none=True) if hasattr(
                                         left, "model_dump") else left
        right_value = right.model_dump(mode="json",
                                       exclude_none=True) if hasattr(
                                           right, "model_dump") else right
        if left_value != right_value:
            raise ValueError(
                f"Stage II cache parser output changed during rebind: {name}")


def _artifact(path: Path) -> dict[str, Any]:
    """Describe one immutable file artifact."""

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size
    }


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from one non-empty JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row {line_number} in {path} is not an object")
            yield value


__all__ = [
    "STAGE_II_CACHE_TRANSPORT_REBIND_V2_REPORT_VERSION",
    "rebind_stage_ii_context_cache_manifest_v2",
]
