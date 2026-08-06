"""Recover explicit SideWalk BEV route contracts from producer indices.

The recovery is intentionally split into two deterministic phases.  A trusted
machine with access to the raw SideWalk producer tree builds a compact registry
from ``bev_index.json`` files.  The annotation server then applies that
registry to the original manifest, verifies the archived BEV bytes, and emits
only the failed rows that require a new Reasoning V4 source fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from mcp_server.contracts.reasoning_dataset_v4 import BevRouteEvidenceV4
from mcp_server.contracts.reasoning_dataset_v4 import (
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from mcp_server.data.legacy.reasoning_v2.source import (
    _dataset_source,
    _source_record_id,
)


SIDEWALK_BEV_CONTRACT_REGISTRY_SCHEMA_VERSION = (
    "urbannav_sidewalk_bev_contract_registry_v1"
)
SIDEWALK_BEV_CONTRACT_RECOVERY_VERSION = (
    "urbannav_sidewalk_bev_contract_recovery_v1"
)


class SidewalkBevProducerEvidence(BaseModel):
    """Store the producer facts supporting one recovered BEV contract.

    Args:
        index_schema_version: Exact raw producer index schema.
        index_relative_path: Producer index path relative to its frozen root.
        image_relative_path: Raw BEV image path relative to that root.
        image_sha256: SHA-256 digest of the raw producer JPEG.
        image_size_bytes: Raw producer JPEG size.
        alignment: Declared BEV presentation alignment.
        route_scope: Declared rendered route scope.
        route_point_count: Number of rendered route points.
        draw_trajectory: Whether the producer rendered a trajectory overlay.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index_schema_version: Literal["sidewalk_bev_index_v1"]
    index_relative_path: str = Field(min_length=1, max_length=512)
    image_relative_path: str = Field(min_length=1, max_length=512)
    image_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    image_size_bytes: int = Field(gt=0)
    alignment: Literal["ego_aligned"]
    route_scope: Literal["global_full"]
    route_point_count: int = Field(gt=0)
    draw_trajectory: Literal[True]


class SidewalkBevContractRegistryRow(BaseModel):
    """Bind one failed annotation row to an authoritative producer contract.

    Args:
        schema_version: Exact registry row schema.
        source_group_id: Stable Reasoning source identity.
        source_record_id: Dataset-local source identity.
        source_index: Zero-based row index in the original SideWalk manifest.
        failure_category: Original failure slice.
        clip_id: Raw SideWalk clip identity.
        sample_frame_name: Current FPV sample frame.
        bev_frame_name: Current BEV producer frame.
        archived_bev_relative_path: Dataset-relative archived BEV path.
        bev_route_evidence: Explicit Reasoning V4 route capability contract.
        producer: Auditable raw producer facts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        SIDEWALK_BEV_CONTRACT_REGISTRY_SCHEMA_VERSION
    ] = SIDEWALK_BEV_CONTRACT_REGISTRY_SCHEMA_VERSION
    source_group_id: str = Field(min_length=1, max_length=240)
    source_record_id: str = Field(min_length=1, max_length=220)
    source_index: int = Field(ge=0)
    failure_category: Literal["instruction", "bev", "decode"]
    clip_id: str = Field(min_length=1, max_length=200)
    sample_frame_name: str = Field(min_length=1, max_length=64)
    bev_frame_name: str = Field(min_length=1, max_length=64)
    archived_bev_relative_path: str = Field(min_length=1, max_length=512)
    bev_route_evidence: BevRouteEvidenceV4
    producer: SidewalkBevProducerEvidence


def build_sidewalk_bev_contract_registry(
    failure_rows_jsonl: str | Path,
    producer_root: str | Path,
    output_jsonl: str | Path,
    *,
    media_repair_root: str | Path | None = None,
) -> dict[str, Any]:
    r"""Build a compact registry from audited failures and raw producer indices.

    Each selected BEV must satisfy the conjunction

    \[
    A = E \land T \land G \land (N_{route} > 0),
    \]

    where ``E`` is ego alignment, ``T`` is a rendered trajectory, ``G`` is
    ``global_full`` scope, and ``N_route`` is the producer route-point count.
    Only rows satisfying ``A`` receive direction authority.

    Args:
        failure_rows_jsonl: Audited NavigationContext failure rows.
        producer_root: Raw SideWalk root containing ``chunk_*/*/bev_index.json``.
        output_jsonl: Destination compact registry.
        media_repair_root: Optional root receiving raw images for ``decode``
            failures at their dataset-relative archive paths.

    Returns:
        Deterministic registry-build report.

    Raises:
        ValueError: If a failure row is duplicated or lacks a valid producer
            contract.
    """

    root = Path(producer_root).expanduser().resolve()
    index_by_clip = _producer_index_paths(root)
    repair_root = (
        Path(media_repair_root).expanduser()
        if media_repair_root is not None
        else None
    )
    outputs: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for failure in iter_jsonl(failure_rows_jsonl):
        counts["input_rows"] += 1
        source_group_id = str(failure.get("source_group_id") or "").strip()
        source_record_id = str(failure.get("source_record_id") or "").strip()
        if not source_group_id or not source_record_id:
            raise ValueError("failure row lacks source identity")
        if source_group_id in seen_source_ids:
            raise ValueError(f"duplicate source_group_id: {source_group_id}")
        seen_source_ids.add(source_group_id)
        category = str(failure.get("category") or "").strip()
        if category not in {"instruction", "bev", "decode"}:
            raise ValueError(f"unsupported failure category: {category}")
        clip_id, sample_frame_name = _split_source_record_id(source_record_id)
        archived_relative = _dataset_relative_media_path(
            str(failure.get("bev_path") or "")
        )
        bev_frame_name = _bev_frame_name(archived_relative)
        index_path = index_by_clip.get(clip_id)
        if index_path is None:
            raise ValueError(f"producer index missing for clip: {clip_id}")
        index_payload = _load_json_object(index_path)
        entry = _validated_producer_entry(
            index_payload,
            clip_id=clip_id,
            bev_frame_name=bev_frame_name,
        )
        raw_image = _resolve_raw_bev_image(
            index_path,
            entry,
            bev_frame_name=bev_frame_name,
        )
        image_sha256 = _sha256_file(raw_image)
        route_overlay = entry["bev_route_overlay"]
        evidence = BevRouteEvidenceV4(
            frame="ego",
            ego_forward="up",
            route_overlay_visible=True,
            route_direction_authority="visual_conditional",
            route_progress_authority="none",
            route_terminal_marker_visible=False,
            route_endpoint_semantics="recorded_clip_endpoint",
            local_safety_authority=False,
            producer_contract_id="sidewalk_bev_index_v1",
            producer_contract_version="1.0",
        )
        registry_row = SidewalkBevContractRegistryRow(
            source_group_id=source_group_id,
            source_record_id=source_record_id,
            source_index=int(failure["index"]),
            failure_category=category,
            clip_id=clip_id,
            sample_frame_name=sample_frame_name,
            bev_frame_name=bev_frame_name,
            archived_bev_relative_path=archived_relative,
            bev_route_evidence=evidence,
            producer=SidewalkBevProducerEvidence(
                index_schema_version="sidewalk_bev_index_v1",
                index_relative_path=index_path.relative_to(root).as_posix(),
                image_relative_path=raw_image.relative_to(root).as_posix(),
                image_sha256=image_sha256,
                image_size_bytes=raw_image.stat().st_size,
                alignment="ego_aligned",
                route_scope="global_full",
                route_point_count=int(route_overlay["point_count"]),
                draw_trajectory=True,
            ),
        )
        outputs.append(registry_row.model_dump(mode="json"))
        counts[f"category:{category}"] += 1
        if category == "decode" and repair_root is not None:
            destination = repair_root / archived_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_image, destination)
            if _sha256_file(destination) != image_sha256:
                raise ValueError(
                    f"media repair copy digest mismatch: {destination}"
                )
            counts["media_repair_rows"] += 1
    write_jsonl_atomic(output_jsonl, outputs)
    counts["output_rows"] = len(outputs)
    return {
        "schema_version": (
            f"{SIDEWALK_BEV_CONTRACT_REGISTRY_SCHEMA_VERSION}_report"
        ),
        "failure_rows_jsonl": str(failure_rows_jsonl),
        "producer_root": str(root),
        "output_jsonl": str(output_jsonl),
        "media_repair_root": (
            str(repair_root) if repair_root is not None else None
        ),
        "counts": dict(counts),
        "passed": counts["input_rows"] == counts["output_rows"],
    }


def build_sidewalk_navigation_repair_subset(
    input_jsonl: str | Path,
    registry_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    media_root: str | Path | None = None,
) -> dict[str, Any]:
    """Apply recovered contracts and emit only selected repair rows.

    Args:
        input_jsonl: Original full SideWalk manifest in audited row order.
        registry_jsonl: Compact producer-backed contract registry.
        output_jsonl: Repaired subset consumed by a new annotation run.
        media_root: Optional dataset root.  When supplied, every archived BEV
            is required to match the raw producer SHA-256 digest.

    Returns:
        Deterministic subset-build report with category counts.

    Raises:
        ValueError: If identities, media bytes, or registry coverage differ
            from the audited failure set.
    """

    registry = _load_registry(registry_jsonl)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    dataset_root = (
        Path(media_root).expanduser()
        if media_root is not None
        else None
    )
    seen_indices: set[int] = set()
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        contract = registry.get(index)
        if contract is None:
            continue
        seen_indices.add(index)
        counts["selected_input_rows"] += 1
        dataset_source = _dataset_source(row, fallback="sidewalk")
        source_record_id = _source_record_id(row, index=index)
        source_group_id = str(
            row.get("source_group_id")
            or f"{dataset_source}:{source_record_id}"
        )
        if source_group_id != contract.source_group_id:
            raise ValueError(
                "source_group_id mismatch at index "
                f"{index}: {source_group_id} != {contract.source_group_id}"
            )
        current_bev = _current_bev_path(row)
        if _media_identity_path(current_bev) != _media_identity_path(
            contract.archived_bev_relative_path
        ):
            raise ValueError(
                "current BEV path mismatch at index "
                f"{index}: {current_bev} != "
                f"{contract.archived_bev_relative_path}"
            )
        if dataset_root is not None:
            raw_archived_path = Path(current_bev).expanduser()
            archived_path = (
                raw_archived_path
                if raw_archived_path.is_absolute()
                else dataset_root / raw_archived_path
            )
            if not archived_path.is_file() or archived_path.stat().st_size <= 0:
                raise ValueError(f"archived BEV is missing or empty: {archived_path}")
            archived_sha256 = _sha256_file(archived_path)
            if archived_sha256 != contract.producer.image_sha256:
                raise ValueError(
                    "archived BEV digest differs from producer: "
                    f"{archived_path}"
                )
            counts["media_digest_verified_rows"] += 1
        enriched = dict(row)
        enriched["bev_route_evidence"] = {
            "bev.current": contract.bev_route_evidence.model_dump(
                mode="json"
            )
        }
        sidewalk_meta = dict(enriched.get("sidewalk_meta") or {})
        sidewalk_meta["bev_route_contract_recovery"] = {
            "recovery_version": SIDEWALK_BEV_CONTRACT_RECOVERY_VERSION,
            "registry_schema_version": contract.schema_version,
            "producer_index_relative_path": (
                contract.producer.index_relative_path
            ),
            "producer_image_sha256": contract.producer.image_sha256,
            "bev_frame_name": contract.bev_frame_name,
        }
        enriched["sidewalk_meta"] = sidewalk_meta
        selected.append(enriched)
        counts[f"category:{contract.failure_category}"] += 1
    missing_indices = sorted(set(registry) - seen_indices)
    if missing_indices:
        raise ValueError(
            "registry indices missing from input manifest: "
            f"{missing_indices[:20]}"
        )
    write_jsonl_atomic(output_jsonl, selected)
    counts["output_rows"] = len(selected)
    return {
        "schema_version": (
            f"{SIDEWALK_BEV_CONTRACT_RECOVERY_VERSION}_report"
        ),
        "input_jsonl": str(input_jsonl),
        "registry_jsonl": str(registry_jsonl),
        "output_jsonl": str(output_jsonl),
        "media_root": str(dataset_root) if dataset_root is not None else None,
        "counts": dict(counts),
        "passed": len(selected) == len(registry),
    }


def audit_sidewalk_navigation_repair(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Audit repaired source authority and accepted dependent sections.

    Args:
        source_jsonl: Rebuilt V4 source subset.
        annotation_jsonl: Latest annotation rows for that subset.
        expected_rows: Exact audited failure-set size.

    Returns:
        Fail-closed repair report.  Candidate structural validity is reported
        but is not a pass condition because strong-action conflicts remain a
        separate human-adjudication axis.
    """

    sources: dict[str, ReasoningSourceV4] = {}
    annotations: dict[str, ReasoningAnnotationV4] = {}
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for raw in iter_jsonl(source_jsonl):
        source = ReasoningSourceV4.model_validate(raw)
        source_id = source.observation.source_group_id
        if source_id in sources:
            raise ValueError(f"duplicate repair source: {source_id}")
        sources[source_id] = source
        route_entries = [
            entry
            for entry in source.observation.evidence_catalog
            if str(entry.ref) == "bev.current"
            and entry.bev_route_evidence is not None
        ]
        if len(route_entries) == 1:
            evidence = route_entries[0].bev_route_evidence
            assert evidence is not None
            if (
                evidence.producer_contract_id == "sidewalk_bev_index_v1"
                and evidence.frame == "ego"
                and evidence.ego_forward == "up"
                and evidence.route_direction_authority
                == "visual_conditional"
            ):
                counts["source_authority_valid_rows"] += 1
    for raw in iter_jsonl(annotation_jsonl):
        annotation = ReasoningAnnotationV4.model_validate(raw)
        annotations[annotation.source_group_id] = annotation
    for source_id, source in sources.items():
        annotation = annotations.get(source_id)
        if annotation is None:
            if len(errors) < 100:
                errors.append(
                    {"source_group_id": source_id, "error": "annotation_missing"}
                )
            continue
        counts["annotation_rows"] += 1
        if annotation.source_fingerprint != source.source_fingerprint:
            if len(errors) < 100:
                errors.append(
                    {
                        "source_group_id": source_id,
                        "error": "source_fingerprint_mismatch",
                    }
                )
            continue
        counts["record_complete_rows"] += int(
            annotation.record_status == "complete"
        )
        counts["all_sections_accepted_rows"] += int(
            annotation.all_sections_accepted
        )
        counts["candidate_structurally_valid_rows"] += int(
            annotation.candidate_structure_record.status == "valid"
        )
        for section_name in (
            "TaskGrounding",
            "NavigationContextGrounding",
            "Decision",
        ):
            record = annotation.stage_records.get(section_name)
            counts[f"{section_name}:accepted"] += int(
                record is not None and record.status == "accepted"
            )
        task_record = annotation.stage_records.get("TaskGrounding")
        counts["task_registry_rows"] += int(
            task_record is not None
            and task_record.producer_kind == "deterministic"
            and task_record.derivation_version
            == "reasoning_v4_task_canonicalization_v3"
        )
    source_ids = set(sources)
    annotation_ids = set(annotations)
    extra_annotations = sorted(annotation_ids - source_ids)
    if extra_annotations:
        errors.append(
            {
                "source_group_id": extra_annotations[0],
                "error": "annotation_without_repair_source",
            }
        )
    required_equal = (
        len(sources),
        counts["source_authority_valid_rows"],
        counts["annotation_rows"],
        counts["record_complete_rows"],
        counts["all_sections_accepted_rows"],
        counts["TaskGrounding:accepted"],
        counts["NavigationContextGrounding:accepted"],
        counts["Decision:accepted"],
        counts["task_registry_rows"],
    )
    passed = (
        not errors
        and len(sources) == expected_rows
        and all(value == expected_rows for value in required_equal)
    )
    return {
        "schema_version": (
            "urbannav_sidewalk_navigation_context_repair_audit_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "expected_rows": expected_rows,
        "source_rows": len(sources),
        "unique_annotation_rows": len(annotations),
        "counts": dict(counts),
        "errors": errors[:100],
        "passed": passed,
    }


def _producer_index_paths(root: Path) -> dict[str, Path]:
    """Index exactly one top-level producer manifest per clip."""

    result: dict[str, Path] = {}
    for path in sorted(root.glob("chunk_*/*/bev_index.json")):
        clip_id = path.parent.name
        if clip_id in result:
            raise ValueError(f"duplicate producer index for clip: {clip_id}")
        result[clip_id] = path
    if not result:
        raise ValueError(f"no SideWalk producer indices found under: {root}")
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object with a bounded error contract."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return value


def _validated_producer_entry(
    payload: Mapping[str, Any],
    *,
    clip_id: str,
    bev_frame_name: str,
) -> dict[str, Any]:
    """Return one entry only after every direction-authority fact passes."""

    if payload.get("schema_version") != "sidewalk_bev_index_v1":
        raise ValueError(f"unexpected producer schema for clip: {clip_id}")
    if str(payload.get("clip_id") or "") != clip_id:
        raise ValueError(f"producer clip identity mismatch: {clip_id}")
    if payload.get("bev_alignment") != "ego_aligned":
        raise ValueError(f"producer BEV is not ego aligned: {clip_id}")
    if payload.get("draw_trajectory") is not True:
        raise ValueError(f"producer trajectory overlay is disabled: {clip_id}")
    if payload.get("bev_route_scope") != "global_full":
        raise ValueError(f"producer route scope is not global_full: {clip_id}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"producer entries missing: {clip_id}")
    matches = [
        value
        for value in entries
        if isinstance(value, dict)
        and _normalized_frame_name(value.get("frame_name"))
        == _normalized_frame_name(bev_frame_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one producer entry for {clip_id}:{bev_frame_name}, "
            f"found {len(matches)}"
        )
    entry = matches[0]
    overlay = entry.get("bev_route_overlay")
    if not isinstance(overlay, dict):
        raise ValueError(f"route overlay metadata missing: {clip_id}:{bev_frame_name}")
    if entry.get("bev_alignment") != "ego_aligned":
        raise ValueError(f"entry alignment mismatch: {clip_id}:{bev_frame_name}")
    if entry.get("draw_trajectory") is not True:
        raise ValueError(f"entry trajectory missing: {clip_id}:{bev_frame_name}")
    if overlay.get("scope") != "global_full":
        raise ValueError(f"entry route scope mismatch: {clip_id}:{bev_frame_name}")
    if int(overlay.get("point_count") or 0) <= 0:
        raise ValueError(f"entry route points missing: {clip_id}:{bev_frame_name}")
    return entry


def _resolve_raw_bev_image(
    index_path: Path,
    entry: Mapping[str, Any],
    *,
    bev_frame_name: str,
) -> Path:
    """Resolve a producer image despite stale absolute paths in old indices."""

    recorded = Path(str(entry.get("bev_image_path") or "")).expanduser()
    filename = recorded.name or f"{bev_frame_name}_bev.jpg"
    candidates = (
        recorded,
        index_path.parent / "bev_images" / filename,
        index_path.parent / "bev_images" / f"{bev_frame_name}_bev.jpg",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise ValueError(
        "raw producer BEV is missing or empty: "
        f"{index_path.parent.name}:{bev_frame_name}"
    )


def _load_registry(
    path: str | Path,
) -> dict[int, SidewalkBevContractRegistryRow]:
    """Load registry rows keyed by unique source index."""

    result: dict[int, SidewalkBevContractRegistryRow] = {}
    source_ids: set[str] = set()
    for raw in iter_jsonl(path):
        row = SidewalkBevContractRegistryRow.model_validate(raw)
        if row.source_index in result:
            raise ValueError(f"duplicate registry index: {row.source_index}")
        if row.source_group_id in source_ids:
            raise ValueError(
                f"duplicate registry source_group_id: {row.source_group_id}"
            )
        result[row.source_index] = row
        source_ids.add(row.source_group_id)
    if not result:
        raise ValueError("BEV contract registry is empty")
    return result


def _split_source_record_id(value: str) -> tuple[str, str]:
    """Split ``clip_id:sample_frame`` from the right."""

    clip_id, separator, frame_name = value.rpartition(":")
    if not separator or not clip_id or not frame_name:
        raise ValueError(f"invalid SideWalk source_record_id: {value}")
    return clip_id, _normalized_frame_name(frame_name)


def _dataset_relative_media_path(value: str) -> str:
    """Normalize an absolute or relative archive path below ``media/``."""

    path = PurePosixPath(value)
    parts = path.parts
    try:
        media_index = parts.index("media")
    except ValueError as exc:
        raise ValueError(f"BEV path is not below a media root: {value}") from exc
    return PurePosixPath(*parts[media_index:]).as_posix()


def _current_bev_path(row: Mapping[str, Any]) -> str:
    """Return the current BEV selected by the multiframe source contract."""

    sidewalk_meta = row.get("sidewalk_meta")
    if isinstance(sidewalk_meta, Mapping):
        frames = sidewalk_meta.get("multiframe_frames")
        if isinstance(frames, list) and frames:
            current = frames[-1]
            if isinstance(current, Mapping) and current.get("bev_image"):
                return str(current["bev_image"])
    images = row.get("images")
    if isinstance(images, list) and images:
        for value in reversed(images):
            if "_bev." in str(value).lower():
                return str(value)
    raise ValueError("SideWalk row has no current BEV image")


def _media_identity_path(value: str) -> str:
    """Return the stable ``clips/<clip>/bev/<frame>`` media suffix."""

    path = PurePosixPath(value)
    parts = path.parts
    try:
        clips_index = parts.index("clips")
    except ValueError as exc:
        raise ValueError(f"BEV path lacks a clips component: {value}") from exc
    return PurePosixPath(*parts[clips_index:]).as_posix()


def _bev_frame_name(relative_path: str) -> str:
    """Extract the producer frame name from ``NNNNNN_bev.jpg``."""

    stem = PurePosixPath(relative_path).stem
    if stem.endswith("_bev"):
        stem = stem[:-4]
    return _normalized_frame_name(stem)


def _normalized_frame_name(value: Any) -> str:
    """Normalize numeric frame identifiers to six digits."""

    text = str(value or "").strip()
    if not text:
        raise ValueError("frame name is empty")
    return f"{int(text):06d}" if text.isdigit() else text


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 file digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SIDEWALK_BEV_CONTRACT_RECOVERY_VERSION",
    "SIDEWALK_BEV_CONTRACT_REGISTRY_SCHEMA_VERSION",
    "SidewalkBevContractRegistryRow",
    "SidewalkBevProducerEvidence",
    "build_sidewalk_bev_contract_registry",
    "build_sidewalk_navigation_repair_subset",
    "audit_sidewalk_navigation_repair",
]
