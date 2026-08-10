"""Project CityWalker-compatible scene-event tags from quality sidecars.

The projector is deterministic and sidecar-only.  It supplies auditable slice
memberships for analysis without exposing trajectory labels to student inputs
or treating UrbanNav-derived tags as original CityWalker human annotation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcp_server.contracts.local_traversability_geometry import LocalTraversabilityGeometryV1
from mcp_server.contracts.navigation_trainability import NavigationTrainabilityFilterV1
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.contracts.scene_event_tags import (
    CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1,
    SCENE_EVENT_TAGS_SCHEMA_VERSION_V1,
    SceneEventTagV1,
    SceneEventTagsV1,
    scene_event_projection_id_v1,
)
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic


SCENE_EVENT_TAGS_REPORT_SCHEMA_VERSION_V1 = "urbannav_scene_event_tags_report_v1"


def project_citywalker_scene_event_tags_v1(
    *,
    source: ReasoningSourceV4,
    trainability_filter: NavigationTrainabilityFilterV1 | None,
    geometry: LocalTraversabilityGeometryV1 | None,
    warnings: tuple[str, ...] = (),
) -> SceneEventTagsV1:
    """Project one source into a complete CityWalker-compatible tag vector.

    The mapping is intentionally conservative:

    - ``crowd`` requires moderate/dense near-path crowd evidence or a visible
      pedestrian-cluster geometry region;
    - ``person_close_by`` requires shared/blocked near-corridor occupancy and
      positive nearby-person evidence;
    - ``crossing`` requires a crosswalk region or a red-light/crossing cue;
    - ``turn`` uses only the teacher-only direction label or recorded turning
      state as an audit slice;
    - ``action_target_mismatch`` remains unavailable without an independent
      action/target audit; and
    - ``other`` captures an explicit non-core scene-suitability class.

    Args:
        source: Canonical source row.  It remains unchanged.
        trainability_filter: Exact current suitability sidecar, if available.
        geometry: Exact current traversability sidecar, if available.
        warnings: Precomputed missing/stale-sidecar audit findings.

    Returns:
        A full, ordered multi-label event sidecar.
    """

    _validate_sidecar_identity_v1(source, trainability_filter, geometry)
    assessment = trainability_filter.motion_scene_assessment if trainability_filter else None
    scene = trainability_filter.scene_assessment if trainability_filter else None
    accepted_geometry = geometry if geometry and geometry.validator_status == "accepted" else None
    motion_refs = assessment.evidence_refs if assessment else ()
    scene_refs = scene.evidence_refs if scene else ()
    pedestrian_refs = _region_refs_v1(accepted_geometry, surface_type="pedestrian_cluster")
    crosswalk_refs = _region_refs_v1(accepted_geometry, surface_type="crosswalk")
    density = assessment.crowd_density_near_path if assessment else "unknown"
    occupancy = assessment.corridor_occupancy if assessment else "unknown"
    stop_cue = assessment.visible_stop_cue if assessment else "unknown"

    tags = (
        _crowd_tag_v1(density=density, motion_refs=motion_refs, pedestrian_refs=pedestrian_refs),
        _person_close_by_tag_v1(
            density=density,
            occupancy=occupancy,
            motion_refs=motion_refs,
            pedestrian_refs=pedestrian_refs,
        ),
        _turn_tag_v1(source=source, trainability_filter=trainability_filter),
        SceneEventTagV1(
            name="action_target_mismatch",
            state="unavailable",
            provenance=("independent_action_target_audit_unavailable",),
            reason_codes=("independent_action_target_audit_not_available",),
        ),
        _crossing_tag_v1(stop_cue=stop_cue, motion_refs=motion_refs, crosswalk_refs=crosswalk_refs),
        _other_tag_v1(scene_class=scene.scene_class if scene else "unknown", scene_refs=scene_refs),
    )
    return SceneEventTagsV1(
        projection_id=scene_event_projection_id_v1(
            parent_sample_id=source.observation.source_group_id,
            source_fingerprint=source.source_fingerprint,
            filter_decision_id=trainability_filter.decision_id if trainability_filter else None,
            geometry_id=geometry.geometry_id if geometry else None,
        ),
        parent_sample_id=source.observation.source_group_id,
        source_fingerprint=source.source_fingerprint,
        dataset_source=source.observation.dataset_source,
        filter_decision_id=trainability_filter.decision_id if trainability_filter else None,
        geometry_id=geometry.geometry_id if geometry else None,
        tags=tags,
        warnings=warnings,
    )


def write_citywalker_scene_event_tags_v1(
    *,
    source_jsonl: str | Path,
    filter_jsonl: str | Path,
    geometry_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Write event sidecars for every current source revision.

    Args:
        source_jsonl: Canonical Reasoning V4 source JSONL.
        filter_jsonl: Append-only trainability-filter ledger.
        geometry_jsonl: Append-only local-traversability geometry ledger.
        output_jsonl: Replaced derived event-sidecar JSONL destination.
        report_json: Optional receipt path; defaults next to ``output_jsonl``.

    Returns:
        Deterministic write counts, sidecar coverage, state distributions, and
        input/output artifact digests.

    Raises:
        ValueError: If the source ledger repeats a physical parent.
    """

    sources = _load_sources_v1(source_jsonl)
    filters = _load_latest_filters_v1(filter_jsonl)
    geometries = _load_latest_geometry_v1(geometry_jsonl)
    filter_parents = {parent_id for parent_id, _fingerprint in filters}
    geometry_parents = {parent_id for parent_id, _fingerprint in geometries}
    state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    present_counts_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    filter_missing = 0
    filter_stale = 0
    geometry_missing = 0
    geometry_stale = 0
    rows: list[dict[str, Any]] = []
    for parent_id, source in sources.items():
        key = (parent_id, source.source_fingerprint)
        trainability_filter = filters.get(key)
        geometry = geometries.get(key)
        warnings: list[str] = []
        if trainability_filter is None:
            if parent_id in filter_parents:
                filter_stale += 1
                warnings.append("trainability_filter_source_fingerprint_mismatch")
            else:
                filter_missing += 1
                warnings.append("missing_trainability_filter")
        if geometry is None:
            if parent_id in geometry_parents:
                geometry_stale += 1
                warnings.append("geometry_source_fingerprint_mismatch")
            else:
                geometry_missing += 1
                warnings.append("missing_local_traversability_geometry")
        sidecar = project_citywalker_scene_event_tags_v1(
            source=source,
            trainability_filter=trainability_filter,
            geometry=geometry,
            warnings=tuple(warnings),
        )
        rows.append(sidecar.model_dump(mode="json"))
        for tag in sidecar.tags:
            state_counts[tag.name][tag.state] += 1
            if tag.state == "present":
                present_counts_by_dataset[source.observation.dataset_source][tag.name] += 1
    write_count = write_jsonl_atomic(output_jsonl, rows)
    destination = Path(output_jsonl).expanduser().resolve()
    report_path = (
        Path(report_json).expanduser().resolve()
        if report_json is not None
        else destination.with_name("scene_event_tags_report.json")
    )
    report = {
        "schema_version": SCENE_EVENT_TAGS_REPORT_SCHEMA_VERSION_V1,
        "event_sidecar_schema_version": SCENE_EVENT_TAGS_SCHEMA_VERSION_V1,
        "taxonomy_order": list(CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1),
        "source_jsonl": str(Path(source_jsonl).expanduser().resolve()),
        "filter_jsonl": str(Path(filter_jsonl).expanduser().resolve()),
        "geometry_jsonl": str(Path(geometry_jsonl).expanduser().resolve()),
        "source_jsonl_sha256": file_sha256(source_jsonl),
        "filter_jsonl_sha256": file_sha256(filter_jsonl),
        "geometry_jsonl_sha256": file_sha256(geometry_jsonl),
        "output_jsonl": str(destination),
        "output_count": write_count,
        "output_jsonl_sha256": file_sha256(destination),
        "filter_missing_count": filter_missing,
        "filter_stale_count": filter_stale,
        "geometry_missing_count": geometry_missing,
        "geometry_stale_count": geometry_stale,
        "tag_state_counts": {name: dict(state_counts[name]) for name in CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1},
        "present_counts_by_dataset": {dataset: dict(counts) for dataset, counts in present_counts_by_dataset.items()},
        # 只读 projection，绝不把事件标签回写到 canonical source。
        "source_manifest_unchanged": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report | {"report_json": str(report_path)}


def _crowd_tag_v1(
    *, density: str,
    motion_refs: tuple[str, ...],
    pedestrian_refs: tuple[str, ...],
) -> SceneEventTagV1:
    """Map local density and pedestrian-cluster geometry to ``crowd``."""

    if density == "unknown" and not pedestrian_refs:
        return _unavailable_tag_v1("crowd", ("motion_scene_assessment", "local_traversability_geometry"), "crowd_evidence_unavailable")
    if density in {"moderate", "dense"} or pedestrian_refs:
        refs = _ordered_unique_refs_v1(motion_refs + pedestrian_refs)
        provenance = ("motion_scene_assessment", "local_traversability_geometry") if pedestrian_refs else ("motion_scene_assessment",)
        return _available_tag_v1("crowd", "present", provenance, refs, "near_path_crowd_detected")
    return _available_tag_v1("crowd", "absent", ("motion_scene_assessment",), motion_refs, "near_path_crowd_not_detected")


def _person_close_by_tag_v1(
    *, density: str,
    occupancy: str,
    motion_refs: tuple[str, ...],
    pedestrian_refs: tuple[str, ...],
) -> SceneEventTagV1:
    """Map close shared-corridor evidence to ``person_close_by``."""

    if occupancy == "unknown" and density == "unknown" and not pedestrian_refs:
        return _unavailable_tag_v1("person_close_by", ("motion_scene_assessment", "local_traversability_geometry"), "nearby_person_evidence_unavailable")
    present = bool(pedestrian_refs) or (occupancy in {"shared", "blocked"} and density in {"sparse", "moderate", "dense"})
    refs = _ordered_unique_refs_v1(motion_refs + pedestrian_refs)
    provenance = ("motion_scene_assessment", "local_traversability_geometry") if pedestrian_refs else ("motion_scene_assessment",)
    return _available_tag_v1(
        "person_close_by",
        "present" if present else "absent",
        provenance,
        refs,
        "nearby_person_or_shared_corridor" if present else "no_nearby_person_or_shared_corridor",
    )


def _turn_tag_v1(
    *,
    source: ReasoningSourceV4,
    trainability_filter: NavigationTrainabilityFilterV1 | None,
) -> SceneEventTagV1:
    """Map teacher-only direction/motion state to an audit-only turn slice."""

    label = _direction_label_v1(source).lower()
    recorded_motion_state = (
        trainability_filter.motion_summary.recorded_motion_state if trainability_filter else "unknown"
    )
    if label != "unknown":
        present = label.startswith(("left", "right", "turn_left", "turn_right"))
        return _available_tag_v1(
            "turn",
            "present" if present else "absent",
            ("direction_supervision",),
            (),
            "direction_label_turn" if present else "direction_label_not_turn",
        )
    if recorded_motion_state != "unknown":
        return _available_tag_v1(
            "turn",
            "present" if recorded_motion_state == "turning" else "absent",
            ("motion_episode_summary",),
            (),
            "recorded_motion_turning" if recorded_motion_state == "turning" else "recorded_motion_not_turning",
        )
    return _unavailable_tag_v1("turn", ("direction_supervision", "motion_scene_assessment"), "turn_evidence_unavailable")


def _crossing_tag_v1(
    *,
    stop_cue: str,
    motion_refs: tuple[str, ...],
    crosswalk_refs: tuple[str, ...],
) -> SceneEventTagV1:
    """Map crossing cues and crosswalk geometry to ``crossing``."""

    if stop_cue == "unknown" and not crosswalk_refs:
        return _unavailable_tag_v1("crossing", ("motion_scene_assessment", "local_traversability_geometry"), "crossing_evidence_unavailable")
    present = stop_cue in {"red_light", "crossing_wait"} or bool(crosswalk_refs)
    refs = _ordered_unique_refs_v1(motion_refs + crosswalk_refs)
    provenance = ("motion_scene_assessment", "local_traversability_geometry") if crosswalk_refs else ("motion_scene_assessment",)
    return _available_tag_v1(
        "crossing",
        "present" if present else "absent",
        provenance,
        refs,
        "visible_crossing_or_crosswalk" if present else "no_visible_crossing_or_crosswalk",
    )


def _other_tag_v1(*, scene_class: str, scene_refs: tuple[str, ...]) -> SceneEventTagV1:
    """Reserve ``other`` for explicit non-core scene-suitability classes."""

    if scene_class == "unknown":
        return _unavailable_tag_v1("other", ("scene_suitability_assessment",), "scene_class_unavailable")
    present = scene_class != "outdoor_pedestrian_supported"
    return _available_tag_v1(
        "other",
        "present" if present else "absent",
        ("scene_suitability_assessment",),
        scene_refs,
        f"scene_class_{scene_class}" if present else "scene_class_core_outdoor_pedestrian",
    )


def _available_tag_v1(
    name: str,
    state: str,
    provenance: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    reason_code: str,
) -> SceneEventTagV1:
    """Build one evidence-available tag through the typed schema."""

    return SceneEventTagV1(
        name=name,
        state=state,
        provenance=provenance,
        evidence_refs=_ordered_unique_refs_v1(evidence_refs),
        reason_codes=(reason_code,),
    )


def _unavailable_tag_v1(name: str, provenance: tuple[str, ...], reason_code: str) -> SceneEventTagV1:
    """Build one explicitly unavailable tag without fabricated evidence."""

    return SceneEventTagV1(
        name=name,
        state="unavailable",
        provenance=provenance,
        reason_codes=(reason_code,),
    )


def _region_refs_v1(
    geometry: LocalTraversabilityGeometryV1 | None,
    *,
    surface_type: str,
) -> tuple[str, ...]:
    """Return current-FPV refs for matching accepted geometry surfaces."""

    if geometry is None:
        return ()
    return _ordered_unique_refs_v1(
        tuple(
            ref
            for region in geometry.regions
            if region.surface_type == surface_type
            for ref in region.evidence_refs
        )
    )


def _ordered_unique_refs_v1(refs: tuple[str, ...]) -> tuple[str, ...]:
    """Deduplicate evidence refs without changing their visible order."""

    return tuple(dict.fromkeys(refs))


def _direction_label_v1(source: ReasoningSourceV4) -> str:
    """Return a private audit label without exposing it to a student prompt."""

    supervision = source.direction_supervision
    if supervision is None:
        return "unknown"
    return str(supervision.label_name or "unknown")


def _validate_sidecar_identity_v1(
    source: ReasoningSourceV4,
    trainability_filter: NavigationTrainabilityFilterV1 | None,
    geometry: LocalTraversabilityGeometryV1 | None,
) -> None:
    """Reject direct calls that accidentally combine different source revisions."""

    parent_id = source.observation.source_group_id
    fingerprint = source.source_fingerprint
    for label, sidecar in (("trainability_filter", trainability_filter), ("geometry", geometry)):
        if sidecar is not None and (
            sidecar.parent_sample_id != parent_id or sidecar.source_fingerprint != fingerprint
        ):
            raise ValueError(f"scene_event_projection_{label}_source_identity_mismatch")


def _load_sources_v1(path: str | Path) -> dict[str, ReasoningSourceV4]:
    """Load exactly one current source revision per physical parent."""

    rows: dict[str, ReasoningSourceV4] = {}
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        parent_id = source.observation.source_group_id
        if parent_id in rows:
            raise ValueError(f"duplicate_source_parent_sample_id:{parent_id}")
        rows[parent_id] = source
    return rows


def _load_latest_filters_v1(path: str | Path) -> dict[tuple[str, str], NavigationTrainabilityFilterV1]:
    """Load the latest exact-source filter result from an append-only ledger."""

    rows: dict[tuple[str, str], NavigationTrainabilityFilterV1] = {}
    for payload in iter_jsonl(path):
        result = NavigationTrainabilityFilterV1.model_validate(payload)
        rows[(result.parent_sample_id, result.source_fingerprint)] = result
    return rows


def _load_latest_geometry_v1(path: str | Path) -> dict[tuple[str, str], LocalTraversabilityGeometryV1]:
    """Load the latest exact-source geometry result from an append-only ledger."""

    rows: dict[tuple[str, str], LocalTraversabilityGeometryV1] = {}
    for payload in iter_jsonl(path):
        result = LocalTraversabilityGeometryV1.model_validate(payload)
        rows[(result.parent_sample_id, result.source_fingerprint)] = result
    return rows


__all__ = [
    "SCENE_EVENT_TAGS_REPORT_SCHEMA_VERSION_V1",
    "project_citywalker_scene_event_tags_v1",
    "write_citywalker_scene_event_tags_v1",
]
