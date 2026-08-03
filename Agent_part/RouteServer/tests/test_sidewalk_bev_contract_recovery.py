"""Focused tests for SideWalk producer-backed BEV contract recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_v4_source import _bev_route_evidence_v4
from mcp_server.data.sidewalk_bev_contract_recovery import (
    SidewalkBevContractRegistryRow,
    build_sidewalk_bev_contract_registry,
    build_sidewalk_navigation_repair_subset,
)


def _write_producer_fixture(root: Path) -> tuple[str, Path]:
    """Create one raw producer index and BEV image."""

    clip_id = "2025-07-13__C10003__fixture__15"
    clip_root = root / "chunk_0" / clip_id
    image_path = clip_root / "bev_images" / "000101_bev.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"valid-jpeg-fixture")
    payload = {
        "schema_version": "sidewalk_bev_index_v1",
        "clip_id": clip_id,
        "bev_sample_fps": 2.0,
        "bev_alignment": "ego_aligned",
        "draw_trajectory": True,
        "bev_route_scope": "global_full",
        "entries": [
            {
                "frame_idx": 101,
                "frame_name": "000101",
                "bev_image_path": "/stale/raw/path/000101_bev.jpg",
                "bev_alignment": "ego_aligned",
                "draw_trajectory": True,
                "bev_route_overlay": {
                    "scope": "global_full",
                    "point_count": 17,
                },
            }
        ],
    }
    (clip_root / "bev_index.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return clip_id, image_path


def _failure_row(clip_id: str) -> dict[str, object]:
    """Build one audited failure row selecting producer frame 101."""

    source_record_id = f"{clip_id}:000100"
    return {
        "index": 0,
        "source_group_id": f"sidewalk:{source_record_id}",
        "source_record_id": source_record_id,
        "category": "decode",
        "bev_path": (
            "/dataset/media/sidewalk/archive/clips/"
            f"{clip_id}/bev/000101_bev.jpg"
        ),
    }


def _input_row(clip_id: str) -> dict[str, object]:
    """Build one multiframe row with current FPV 100 and BEV 101."""

    current_bev = (
        "media/sidewalk/archive/clips/"
        f"{clip_id}/bev/000101_bev.jpg"
    )
    return {
        "dataset_source": "sidewalk",
        "images": [
            f"media/sidewalk/archive/clips/{clip_id}/fpv/000100.jpg",
            current_bev,
        ],
        "sidewalk_meta": {
            "clip_id": clip_id,
            "frame_idx": 100,
            "frame_name": "000100",
            "multiframe_frames": [
                {
                    "frame_idx": 100,
                    "fpv_image": (
                        "media/sidewalk/archive/clips/"
                        f"{clip_id}/fpv/000100.jpg"
                    ),
                    "bev_image": current_bev,
                }
            ],
        },
    }


def test_registry_recovers_authority_and_decode_media(
    tmp_path: Path,
) -> None:
    """Producer facts yield one explicit contract and one repair image."""

    producer_root = tmp_path / "producer"
    clip_id, raw_image = _write_producer_fixture(producer_root)
    failures = tmp_path / "failures.jsonl"
    registry = tmp_path / "registry.jsonl"
    repair_root = tmp_path / "repair"
    write_jsonl_atomic(failures, [_failure_row(clip_id)])

    report = build_sidewalk_bev_contract_registry(
        failures,
        producer_root,
        registry,
        media_repair_root=repair_root,
    )
    record = SidewalkBevContractRegistryRow.model_validate(
        next(iter_jsonl(registry))
    )

    assert report["passed"] is True
    assert report["counts"] == {
        "input_rows": 1,
        "category:decode": 1,
        "media_repair_rows": 1,
        "output_rows": 1,
    }
    assert record.bev_route_evidence.frame == "ego"
    assert record.bev_route_evidence.ego_forward == "up"
    assert (
        record.bev_route_evidence.route_direction_authority
        == "visual_conditional"
    )
    copied = repair_root / record.archived_bev_relative_path
    assert copied.read_bytes() == raw_image.read_bytes()


def test_apply_registry_verifies_media_and_emits_selected_subset(
    tmp_path: Path,
) -> None:
    """Application binds row identity, image bytes, and explicit evidence."""

    producer_root = tmp_path / "producer"
    clip_id, raw_image = _write_producer_fixture(producer_root)
    failures = tmp_path / "failures.jsonl"
    registry = tmp_path / "registry.jsonl"
    source = tmp_path / "source.jsonl"
    output = tmp_path / "repair.jsonl"
    media_root = tmp_path / "dataset"
    write_jsonl_atomic(failures, [_failure_row(clip_id)])
    write_jsonl_atomic(source, [_input_row(clip_id)])
    build_sidewalk_bev_contract_registry(
        failures,
        producer_root,
        registry,
    )
    archived = (
        media_root
        / "media/sidewalk/archive/clips"
        / clip_id
        / "bev/000101_bev.jpg"
    )
    archived.parent.mkdir(parents=True)
    archived.write_bytes(raw_image.read_bytes())

    report = build_sidewalk_navigation_repair_subset(
        source,
        registry,
        output,
        media_root=media_root,
    )
    repaired = next(iter_jsonl(output))
    route_evidence = _bev_route_evidence_v4(
        repaired,
        role="bev.current",
        dataset_source="sidewalk",
    )

    assert report["passed"] is True
    assert report["counts"]["output_rows"] == 1
    assert report["counts"]["media_digest_verified_rows"] == 1
    assert route_evidence.producer_contract_id == "sidewalk_bev_index_v1"
    assert route_evidence.route_direction_authority == "visual_conditional"
    assert repaired["sidewalk_meta"]["bev_route_contract_recovery"][
        "bev_frame_name"
    ] == "000101"


def test_apply_registry_rejects_archive_digest_mismatch(
    tmp_path: Path,
) -> None:
    """Corrupt archived media cannot enter the repair subset."""

    producer_root = tmp_path / "producer"
    clip_id, _ = _write_producer_fixture(producer_root)
    failures = tmp_path / "failures.jsonl"
    registry = tmp_path / "registry.jsonl"
    source = tmp_path / "source.jsonl"
    media_root = tmp_path / "dataset"
    write_jsonl_atomic(failures, [_failure_row(clip_id)])
    write_jsonl_atomic(source, [_input_row(clip_id)])
    build_sidewalk_bev_contract_registry(
        failures,
        producer_root,
        registry,
    )
    archived = (
        media_root
        / "media/sidewalk/archive/clips"
        / clip_id
        / "bev/000101_bev.jpg"
    )
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="differs from producer"):
        build_sidewalk_navigation_repair_subset(
            source,
            registry,
            tmp_path / "repair.jsonl",
            media_root=media_root,
        )


def test_legacy_overlay_without_alignment_has_no_direction_authority() -> None:
    """A visible legacy line does not prove an ego-frame direction contract."""

    evidence = _bev_route_evidence_v4(
        {
            "bev_patch_meta_path": "/legacy/patch.json",
            "bev_route_overlay": {
                "scope": "global_full",
                "point_count": 8,
            },
        },
        role="bev.current",
        dataset_source="sidewalk",
    )

    assert evidence.route_overlay_visible is True
    assert evidence.frame == "unknown"
    assert evidence.ego_forward == "unknown"
    assert evidence.route_direction_authority == "none"
