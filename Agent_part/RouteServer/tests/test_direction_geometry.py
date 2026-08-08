from __future__ import annotations

import json

import pytest

from mcp_server.contracts.direction_profile import (
    EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    quantize_direction_angle,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionSupervisionV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.data.canonical.direction_geometry import (
    build_direction_geometry_supervision_v1,
)
from tools.model_processing.audit_direction_geometry import (
    audit_direction_geometry_manifest,
)


def _direction_for_angle(raw_angle_deg: float) -> DirectionSupervisionV4:
    """Build a valid strong stamp from the canonical quantizer only."""

    profile = resolve_stage_a_direction_profile(
        EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
    )
    quantized = quantize_direction_angle(
        EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
        raw_angle_deg,
    )
    return DirectionSupervisionV4(
        profile_id=EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
        profile_version=str(profile["profile_version"]),
        label_index=quantized.label_index,
        label_name=quantized.label_name,
        choice=str(profile["choices"][quantized.label_index]),
        angle_deg=quantized.label_angle_deg,
        source=str(profile["allowed_sources"][0]),
        bin_mode=str(profile["bin_mode"]),
        derivation_version=str(profile["derivation_version"]),
    )


def test_quantizer_and_sidecar_keep_raw_boundary_geometry() -> None:
    direction = _direction_for_angle(15.1)
    geometry = build_direction_geometry_supervision_v1(
        {"direction_raw_angle_deg": 15.1, "direction_displacement": 1.0},
        direction=direction,
    )

    assert direction.label_name == "right_20"
    assert geometry.raw_angle_deg == 15.1
    assert geometry.nearest_boundary_deg == 15.0
    assert geometry.boundary_distance_deg == pytest.approx(0.1)
    assert geometry.confidence == "boundary_ambiguous"
    assert geometry.confidence_weight == 0.25


def test_outer_bin_keeps_raw_angle_instead_of_fake_center() -> None:
    direction = _direction_for_angle(70.0)
    geometry = build_direction_geometry_supervision_v1(
        {"direction_raw_angle_deg": 70.0, "direction_displacement": 1.0},
        direction=direction,
    )

    assert direction.label_name == "right_50plus"
    assert geometry.raw_angle_deg == 70.0
    assert geometry.angle_valid is True
    assert geometry.confidence == "high_confidence"


def test_nonfinite_displacement_becomes_unverified_audit_state() -> None:
    direction = _direction_for_angle(20.0)
    geometry = build_direction_geometry_supervision_v1(
        {"direction_raw_angle_deg": 20.0, "direction_displacement": float("nan")},
        direction=direction,
    )

    assert geometry.raw_angle_deg == 20.0
    assert geometry.displacement is None
    assert geometry.stop_rule_verified is False


def test_sidecar_rejects_raw_angle_with_different_strong_label() -> None:
    direction = _direction_for_angle(10.0)

    with pytest.raises(ValueError, match="different strong label"):
        build_direction_geometry_supervision_v1(
            {"direction_raw_angle_deg": 25.0},
            direction=direction,
        )


def test_audit_reuses_existing_sidecar_without_reconstruction(tmp_path) -> None:
    direction = _direction_for_angle(70.0)
    geometry = build_direction_geometry_supervision_v1(
        {"direction_raw_angle_deg": 70.0, "direction_displacement": 1.0},
        direction=direction,
    )
    row = {
        "direction_supervision": direction.model_dump(mode="json"),
        "direction_geometry": geometry.model_dump(mode="json"),
    }
    input_jsonl = tmp_path / "source.jsonl"
    output_jsonl = tmp_path / "out.jsonl"
    report_json = tmp_path / "report.json"
    input_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = audit_direction_geometry_manifest(
        input_jsonl,
        output_jsonl,
        report_json,
        dataset_source="egowalk",
    )

    assert report["passed"] is True
    assert report["counts"]["status:high_confidence"] == 1
    output = json.loads(output_jsonl.read_text(encoding="utf-8"))
    assert output["direction_geometry"]["raw_angle_deg"] == 70.0


def test_source_fingerprint_normalizes_absent_and_null_geometry() -> None:
    """Preserve the frozen identity of sources written before the sidecar."""

    legacy = {"schema_version": "v4", "direction_supervision": None}
    explicit_null = {**legacy, "direction_geometry": None}
    populated = {**legacy, "direction_geometry": {"raw_angle_deg": 10.0}}

    assert reasoning_source_fingerprint_v4(legacy) == reasoning_source_fingerprint_v4(
        explicit_null,
    )
    assert reasoning_source_fingerprint_v4(legacy) != reasoning_source_fingerprint_v4(
        populated,
    )
