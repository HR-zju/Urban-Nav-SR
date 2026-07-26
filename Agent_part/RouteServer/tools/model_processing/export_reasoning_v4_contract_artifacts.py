#!/usr/bin/env python3
"""Export deterministic Reasoning V4 schemas and golden parity fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.agent.reasoning_v4_prompt_policy import (
    PROMPT_PROFILE_REGISTRY_V4,
)
from mcp_server.agent.reasoning_v4_text import (
    render_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    ReasoningTrainingViewV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    FinishV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    NavigationSourceAlignmentV4,
    REASONING_V4_CONTRACT_SCHEMA_DIGEST,
    REASONING_V4_EVIDENCE_POLICY_VERSION,
    REASONING_V4_PARSER_ID,
    REASONING_V4_RENDERER_ID,
    REASONING_V4_SECTION_DEFINITION_VERSION,
    REASONING_V4_VALIDATOR_ID,
    REASONING_V4_WIRE_FORMAT_VERSION,
    ReasoningTargetV4,
    ReasoningTraceV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGoalV4,
    TaskGroundingV4,
    TraversabilityRegionV4,
)


ARTIFACT_SCHEMA_VERSION = "urbannav_reasoning_v4_contract_artifacts_v1"
ARTIFACT_FILENAMES = (
    "reasoning_v4.schema.json",
    "reasoning_training_view_v4.schema.json",
    "reasoning_v4_golden_valid.json",
    "reasoning_v4_golden_invalid.json",
    "contract_digest.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse contract-artifact export arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed files differ instead of replacing them.",
    )
    return parser.parse_args(argv)


def build_reasoning_v4_contract_artifacts() -> dict[str, Any]:
    """Build every canonical V4 schema and golden fixture in memory.

    Returns:
        Mapping from stable artifact filename to JSON-compatible payload.
    """

    target = _valid_target()
    assistant_text = render_reasoning_target_v4(target)
    valid_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "cases": [
            {
                "case_id": "minimal_shared_input_navigation",
                "entrypoint": "reasoning_target",
                "value": target.model_dump(mode="json", exclude_none=True),
                "canonical_assistant_text": assistant_text,
            }
        ],
    }
    target_payload = target.model_dump(mode="json", exclude_none=True)
    invalid_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "cases": _invalid_cases(
            target_payload=target_payload,
            assistant_text=assistant_text,
        ),
    }
    artifacts: dict[str, Any] = {
        "reasoning_v4.schema.json": ReasoningTargetV4.model_json_schema(),
        "reasoning_training_view_v4.schema.json": (
            ReasoningTrainingViewV4.model_json_schema()
        ),
        "reasoning_v4_golden_valid.json": valid_payload,
        "reasoning_v4_golden_invalid.json": invalid_payload,
    }
    artifacts["contract_digest.json"] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "reasoning_contract_schema_digest": (
            REASONING_V4_CONTRACT_SCHEMA_DIGEST
        ),
        "protocol_ids": {
            "wire_format_version": REASONING_V4_WIRE_FORMAT_VERSION,
            "section_definition_version": (
                REASONING_V4_SECTION_DEFINITION_VERSION
            ),
            "evidence_policy_version": (
                REASONING_V4_EVIDENCE_POLICY_VERSION
            ),
            "renderer_id": REASONING_V4_RENDERER_ID,
            "parser_id": REASONING_V4_PARSER_ID,
            "validator_id": REASONING_V4_VALIDATOR_ID,
        },
        "canonical_source_sha256": _canonical_source_hashes(),
        "prompt_policy_digests": {
            profile_id: profile.policy_digest
            for profile_id, profile in sorted(
                PROMPT_PROFILE_REGISTRY_V4.items()
            )
        },
        "artifact_sha256": {
            filename: _payload_sha256(payload)
            for filename, payload in artifacts.items()
        },
    }
    return artifacts


def _canonical_source_hashes() -> dict[str, str]:
    """Hash behavior-bearing canonical sources for freshness CI."""

    route_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "src/mcp_server/contracts/direction_profile.py",
        "src/mcp_server/contracts/reasoning_v4.py",
        "src/mcp_server/contracts/reasoning_dataset_v4.py",
        "src/mcp_server/agent/reasoning_v4_prompt_policy.py",
        "src/mcp_server/agent/reasoning_v4_text.py",
        "src/mcp_server/agent/reasoning_v4_validation.py",
        "src/mcp_server/data/reasoning_v4_source.py",
        "src/mcp_server/data/reasoning_v4_composer.py",
        "src/mcp_server/data/reasoning_v4_views.py",
        "src/mcp_server/data/reasoning_v4_audit.py",
        "src/mcp_server/teacher/reasoning_v4_profiles.py",
        "src/mcp_server/teacher/reasoning_v4_teacher.py",
        "src/mcp_server/teacher/reasoning_v4_batch.py",
    )
    return {
        relative: hashlib.sha256(
            (route_root / relative).read_bytes()
        ).hexdigest()
        for relative in relative_paths
    }


def export_reasoning_v4_contract_artifacts(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Write or verify deterministic V4 contract artifacts.

    Args:
        output_dir: Directory containing the five stable artifact filenames.
        check: Compare against committed files without modifying them.

    Returns:
        Machine-readable file digests and check status.

    Raises:
        ValueError: If ``check`` finds a missing or stale artifact.
    """

    destination = Path(output_dir).expanduser()
    artifacts = build_reasoning_v4_contract_artifacts()
    stale: list[str] = []
    written: list[str] = []
    for filename in ARTIFACT_FILENAMES:
        payload = artifacts[filename]
        text = _json_text(payload)
        path = destination / filename
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(filename)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(filename)
    if stale:
        raise ValueError(
            "Reasoning V4 contract artifacts are stale or missing: "
            f"{stale}"
        )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "output_dir": str(destination),
        "checked": bool(check),
        "written": written,
        "stale": stale,
        "artifact_sha256": {
            filename: _payload_sha256(artifacts[filename])
            for filename in ARTIFACT_FILENAMES
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Export or verify the requested artifact directory."""

    args = parse_args(argv)
    report = export_reasoning_v4_contract_artifacts(
        args.output_dir,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _valid_target() -> ReasoningTargetV4:
    """Build one minimal canonical navigation target."""

    task = TaskGroundingV4(
        objective="follow_instruction",
        goal=TaskGoalV4(
            goal_type="none",
            specification="absent",
        ),
    )
    traversability = LocalTraversabilityV4(
        regions=[
            TraversabilityRegionV4(
                region_ref="region.center",
                surface="sidewalk",
                lateral_position="center",
                state="direct",
                clearance="sufficient",
                evidence_refs=["fpv.current"],
            )
        ]
    )
    navigation = NavigationContextGroundingV4(
        source_alignments=[
            NavigationSourceAlignmentV4(
                alignment_ref="alignment.instruction",
                source_ref="instruction.active",
                local_effect="straight",
                aligned_region_refs=["region.center"],
                status="supports",
                evidence_refs=[
                    "instruction.active",
                    "fpv.current",
                ],
            )
        ],
        task_direction=TaskDirectionV4(
            desired_motion="straight",
            candidate_region_refs=["region.center"],
            status="aligned",
            evidence_refs=["instruction.active", "fpv.current"],
        ),
    )
    decision = DecisionV4(
        decision_type="navigate",
        motion="go_straight",
        speed="normal",
        social_action="none",
        selected_region_ref="region.center",
        fallback="stop",
        support_refs=["nav.task_direction", "region.center"],
        evidence_refs=["instruction.active", "fpv.current"],
    )
    trace = ReasoningTraceV4(
        task_grounding=task,
        scene_grounding=SceneGroundingV4(),
        local_traversability=traversability,
        navigation_context_grounding=navigation,
        interaction_constraint_grounding=(
            InteractionConstraintGroundingV4()
        ),
        decision=decision,
    )
    return ReasoningTargetV4(
        reasoning=trace,
        finish=FinishV4(
            decision_type=decision.decision_type,
            motion=decision.motion,
            speed=decision.speed,
            social_action=decision.social_action,
        ),
    )


def _invalid_cases(
    *,
    target_payload: dict[str, Any],
    assistant_text: str,
) -> list[dict[str, Any]]:
    """Build bounded invalid fixtures for model and wire parity."""

    cases: list[dict[str, Any]] = []

    finish_mismatch = _clone(target_payload)
    finish_mismatch["finish"]["motion"] = "turn_left"
    cases.append(
        _invalid_target_case(
            "finish_projection_mismatch",
            finish_mismatch,
            "finish must be an exact semantic projection",
        )
    )

    navigate_without_region = _clone(target_payload)
    navigate_without_region["reasoning"]["decision"].pop(
        "selected_region_ref",
        None,
    )
    cases.append(
        _invalid_target_case(
            "navigate_without_selected_region",
            navigate_without_region,
            "navigate decision requires selected_region_ref",
        )
    )

    scene_without_fpv = _clone(target_payload)
    scene_without_fpv["reasoning"]["scene_grounding"]["entities"] = [
        {
            "entity_ref": "entity.person",
            "category": "pedestrian",
            "lateral_position": "center",
            "depth_position": "near",
            "evidence_refs": ["instruction.active"],
        }
    ]
    cases.append(
        _invalid_target_case(
            "visual_entity_without_fpv_evidence",
            scene_without_fpv,
            "scene entity requires at least one FPV evidence ref",
        )
    )

    conditional_without_reason = _clone(target_payload)
    region = conditional_without_reason["reasoning"][
        "local_traversability"
    ]["regions"][0]
    region["state"] = "conditional"
    region["limiting_factor_codes"] = []
    cases.append(
        _invalid_target_case(
            "conditional_region_without_limiting_factor",
            conditional_without_reason,
            "conditional region requires a limiting factor code",
        )
    )

    dynamic_without_history = _clone(target_payload)
    dynamic_without_history["reasoning"][
        "interaction_constraint_grounding"
    ]["constraints"] = [
        {
            "constraint_ref": "constraint.crossing",
            "constraint_code": "pedestrian_crossing",
            "kind": "social",
            "source_refs": ["fpv.current"],
            "applies_to_region_refs": ["region.center"],
            "relation": "crosses_candidate_path",
            "temporal_basis": "single_frame",
            "status": "applicable",
            "effect": "yield",
            "evidence_refs": ["fpv.current"],
        }
    ]
    cases.append(
        _invalid_target_case(
            "dynamic_relation_without_two_frames",
            dynamic_without_history,
            "dynamic interaction requires temporal_basis=multi_frame",
        )
    )

    cases.extend(
        [
            {
                "case_id": "free_text_after_document",
                "entrypoint": "assistant_text",
                "value": assistant_text + "\nfree text",
                "expected_error_contains": "unexpected_text_outside_document",
            },
            {
                "case_id": "section_order_mismatch",
                "entrypoint": "assistant_text",
                "value": _swap_task_scene_tags(assistant_text),
                "expected_error_contains": "invalid_reasoning_section_order",
            },
        ]
    )
    return cases


def _invalid_target_case(
    case_id: str,
    value: dict[str, Any],
    expected_error: str,
) -> dict[str, Any]:
    """Create one typed-target invalid fixture row."""

    return {
        "case_id": case_id,
        "entrypoint": "reasoning_target",
        "value": value,
        "expected_error_contains": expected_error,
    }


def _swap_task_scene_tags(value: str) -> str:
    """Swap only Task/Scene tag names to produce invalid wire order."""

    replacements = (
        ("TaskGrounding", "__TASK__"),
        ("SceneGrounding", "TaskGrounding"),
        ("__TASK__", "SceneGrounding"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def _clone(value: Any) -> Any:
    """Deep-copy one JSON-compatible value deterministically."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def _json_text(payload: Any) -> str:
    """Serialize an artifact with stable human-readable formatting."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _payload_sha256(payload: Any) -> str:
    """Hash one JSON payload using its canonical compact form."""

    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
