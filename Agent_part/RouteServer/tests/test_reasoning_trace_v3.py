from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v3_text import (
    parse_reasoning_target_v3,
    render_reasoning_target_v3,
)
from mcp_server.contracts.reasoning_v3 import (
    REASONING_V3_CONTRACT_SCHEMA_DIGEST,
    REASONING_V3_SECTION_NAMES,
    ReasoningTargetV3,
    ReasoningTraceV3,
)
from reasoning_v3_test_utils import accepted_annotation, build_test_source


EXPECTED_V3_4_SCHEMA_DIGEST = (
    "abd18f659fa989f1ba1e767b72aeed46061286b3268e327858d5d4fe6a55d2d0"
)


def test_reasoning_v3_4_wire_roundtrip_has_exact_six_sections() -> None:
    target = accepted_annotation(build_test_source()).target
    assert target is not None

    text = render_reasoning_target_v3(target)
    parsed = parse_reasoning_target_v3(text)

    assert parsed.valid, parsed.errors
    assert parsed.target == target
    assert REASONING_V3_SECTION_NAMES == (
        "TaskGrounding",
        "SceneGrounding",
        "LocalTraversability",
        "NavigationContextGrounding",
        "InteractionAndRuleGrounding",
        "Decision",
    )
    assert [text.index(f"<{name}>") for name in REASONING_V3_SECTION_NAMES] == sorted(
        text.index(f"<{name}>") for name in REASONING_V3_SECTION_NAMES
    )
    assert text.count("<think>") == text.count("</think>") == 1
    assert text.count("<finish>") == text.count("</finish>") == 1


def test_reasoning_v3_4_renderer_uses_spaced_readout_and_strict_document_grammar() -> None:
    target = accepted_annotation(build_test_source()).target
    assert target is not None
    rendered = render_reasoning_target_v3(
        target,
        readout_tokens=["[SYS1_0]", "[SYS1_1]"],
    )

    assert rendered.endswith("<readout>[SYS1_0] [SYS1_1]</readout>")
    assert parse_reasoning_target_v3(rendered).valid
    assert "unexpected_text_outside_document" in parse_reasoning_target_v3(
        "prefix\n" + rendered
    ).errors
    assert "unexpected_text_inside_think" in parse_reasoning_target_v3(
        rendered.replace("<SceneGrounding>", "free text\n<SceneGrounding>", 1)
    ).errors
    with pytest.raises(ValueError, match="ToolResultUse"):
        render_reasoning_target_v3(target, tool_result_use={"tool_used": True})


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("task_grounding", "instruction_type", "directional"),
        ("task_grounding", "local_objective", "follow_instruction"),
        ("scene_grounding", "confidence", "high"),
        ("scene_grounding", "mask", "rle"),
        ("scene_grounding", "bbox", [0, 0, 10, 10]),
        ("scene_grounding", "track", "track.01"),
        ("interaction_and_rule_grounding", "ttc", 1.5),
        ("navigation_context_grounding", "route_theta", 30.0),
    ],
)
def test_reasoning_v3_4_contract_explicitly_rejects_removed_fields(
    section: str,
    field: str,
    value: Any,
) -> None:
    target = accepted_annotation(build_test_source()).target
    assert target is not None
    payload = target.reasoning.model_dump(mode="python")
    payload[section][field] = value

    with pytest.raises(ValidationError) as exc_info:
        ReasoningTraceV3.model_validate(payload)

    assert "extra_forbidden" in str(exc_info.value)
    assert field in str(exc_info.value)


def test_reasoning_v3_4_direction_and_finish_copy_complete_profile_stamp() -> None:
    target = accepted_annotation(build_test_source()).target
    assert target is not None

    decision = target.reasoning.decision
    finish = target.finish
    expected = {
        "profile_id": "fpv_degree10_terminal_centroid_v2",
        "profile_version": "2.0",
        "label_index": 12,
        "label_name": "right_30",
        "choice": "M",
        "angle_deg": 30.0,
        "source": "direction_choice_adapter",
    }
    assert {
        key: getattr(decision, key)
        for key in expected
    } == expected
    assert {
        key: getattr(finish, key)
        for key in expected
    } == expected

    mismatched = target.model_dump(mode="python")
    mismatched["finish"]["label_index"] = 11
    with pytest.raises(ValidationError, match="exact projection"):
        ReasoningTargetV3.model_validate(mismatched)


def test_ms_swift_v3_4_schema_and_digest_match_routeserver() -> None:
    consumer = _load_ms_swift_reasoning_v3_contract()
    if consumer is None:
        pytest.skip("sibling ms-swift checkout is not available")

    assert _semantic_schema(ReasoningTargetV3.model_json_schema()) == _semantic_schema(
        consumer.ReasoningTargetV3.model_json_schema()
    )
    assert REASONING_V3_CONTRACT_SCHEMA_DIGEST == EXPECTED_V3_4_SCHEMA_DIGEST
    assert consumer.REASONING_V3_CONTRACT_SCHEMA_DIGEST == REASONING_V3_CONTRACT_SCHEMA_DIGEST


def _load_ms_swift_reasoning_v3_contract() -> Any | None:
    code_root = Path(__file__).resolve().parents[4]
    package_dir = code_root / "ms-swift/swift/sidewalk_nav"
    reasoning_path = package_dir / "reasoning.py"
    contract_path = package_dir / "reasoning_v3.py"
    if not reasoning_path.is_file() or not contract_path.is_file():
        return None

    package_name = "_ms_swift_sidewalk_nav_contract_v34"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package
    for module_name, module_path in (
        ("reasoning", reasoning_path),
        ("reasoning_v3", contract_path),
    ):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified_name, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.reasoning_v3"]


def _semantic_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_schema(item)
            for key, item in sorted(value.items())
            if key not in {"title", "description"}
        }
    if isinstance(value, list):
        return [_semantic_schema(item) for item in value]
    return value
