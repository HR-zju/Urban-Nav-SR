"""Tests for paired Reasoning V4 image-policy comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_server.teacher.reasoning_v4_image_policy import (
    REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
    REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID,
)
from tools.model_processing.compare_reasoning_v4_image_policy_runs import (
    compare_reasoning_v4_image_policy_runs,
)


_SECTIONS = (
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
    "Decision",
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write compact JSONL rows."""

    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source(source_id: str, label: str) -> dict[str, object]:
    """Build one comparison source."""

    return {
        "observation": {
            "source_group_id": source_id,
            "input_modalities": {"bev": True},
        },
        "direction_supervision": {"label_name": label},
    }


def _annotation(
    source_id: str,
    *,
    policy_id: str,
    action: str,
    roles_by_section: dict[str, list[str]],
) -> dict[str, object]:
    """Build one accepted comparison annotation."""

    stages = {
        section: {
            "status": "accepted",
            "attempts": 1,
            "telemetry": {
                "selected_image_roles": roles_by_section[section],
            },
        }
        for section in _SECTIONS
    }
    return {
        "source_group_id": source_id,
        "record_status": "complete",
        "metadata": {
            "section_image_policy_id": policy_id,
            "section_image_policy_digest": f"digest:{policy_id}",
        },
        "stage_records": stages,
        "candidate_structure_record": {"status": "valid"},
        "decision_proposal": {"selected_action_label": action},
        "action_reconciliation": {"review_trigger_codes": []},
        "grounding_outputs": {
            "navigation_context_grounding": {
                "task_direction": {
                    "status": "aligned",
                    "source": "bev_route",
                    "direction": action,
                }
            },
            "interaction_constraint_grounding": {
                "constraints": [],
            },
        },
    }


def test_paired_comparison_reports_quality_and_throughput_gate(
    tmp_path: Path,
) -> None:
    """Identical labels plus faster routed throughput recommend review."""

    source_path = tmp_path / "source.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    routed_path = tmp_path / "routed.jsonl"
    baseline_report = tmp_path / "baseline_report.json"
    routed_report = tmp_path / "routed_report.json"
    sources = [_source("a", "straight"), _source("b", "left_10")]
    all_roles = [
        "fpv.history_4",
        "fpv.history_3",
        "fpv.history_2",
        "fpv.history_1",
        "fpv.current",
        "bev.current",
    ]
    baseline_roles = {
        section: all_roles for section in _SECTIONS
    }
    routed_roles = {
        "TaskGrounding": ["fpv.current", "bev.current"],
        "SceneGrounding": all_roles[:-1],
        "LocalTraversability": ["fpv.current"],
        "NavigationContextGrounding": [
            "fpv.current",
            "bev.current",
        ],
        "InteractionConstraintGrounding": all_roles[:-1],
        "Decision": ["fpv.current", "bev.current"],
    }
    _write_jsonl(source_path, sources)
    _write_jsonl(
        baseline_path,
        [
            _annotation(
                "a",
                policy_id=REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
                action="straight",
                roles_by_section=baseline_roles,
            ),
            _annotation(
                "b",
                policy_id=REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
                action="left_10",
                roles_by_section=baseline_roles,
            ),
        ],
    )
    _write_jsonl(
        routed_path,
        [
            _annotation(
                "a",
                policy_id=REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID,
                action="straight",
                roles_by_section=routed_roles,
            ),
            _annotation(
                "b",
                policy_id=REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID,
                action="left_10",
                roles_by_section=routed_roles,
            ),
        ],
    )
    baseline_report.write_text(
        json.dumps({"status": "done", "rate_rows_per_hour": 100.0}),
        encoding="utf-8",
    )
    routed_report.write_text(
        json.dumps({"status": "done", "rate_rows_per_hour": 125.0}),
        encoding="utf-8",
    )

    report = compare_reasoning_v4_image_policy_runs(
        source_jsonl=source_path,
        baseline_annotation_jsonl=baseline_path,
        routed_annotation_jsonl=routed_path,
        baseline_report_json=baseline_report,
        routed_report_json=routed_report,
        expected_rows=2,
    )

    assert report["paired_source_rows"] == 2
    assert report["quality_gates_passed"] is True
    assert report["replacement_recommended"] is True
    assert report["throughput"]["relative_gain"] == pytest.approx(0.25)
    assert report["paired"]["decision_exact_agreement_fraction"] == 1.0
    assert report["routed"]["selected_image_role_patterns"][
        "Decision"
    ] == {"fpv.current+bev.current": 2}
    assert report["baseline"]["annotation_version_counts"] == {1: 2}


def test_paired_comparison_rejects_nonidentical_source_sets(
    tmp_path: Path,
) -> None:
    """An A/B run is invalid when either arm omits a source ID."""

    source_path = tmp_path / "source.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    routed_path = tmp_path / "routed.jsonl"
    report_path = tmp_path / "report.json"
    roles = {section: ["fpv.current"] for section in _SECTIONS}
    _write_jsonl(source_path, [_source("a", "straight")])
    _write_jsonl(
        baseline_path,
        [
            _annotation(
                "a",
                policy_id=REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
                action="straight",
                roles_by_section=roles,
            )
        ],
    )
    _write_jsonl(
        routed_path,
        [
            _annotation(
                "other",
                policy_id=REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID,
                action="straight",
                roles_by_section=roles,
            )
        ],
    )
    report_path.write_text(
        json.dumps({"status": "done", "rate_rows_per_hour": 1.0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source IDs do not match"):
        compare_reasoning_v4_image_policy_runs(
            source_jsonl=source_path,
            baseline_annotation_jsonl=baseline_path,
            routed_annotation_jsonl=routed_path,
            baseline_report_json=report_path,
            routed_report_json=report_path,
            expected_rows=1,
        )
