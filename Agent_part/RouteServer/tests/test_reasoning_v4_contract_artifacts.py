from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_text import parse_reasoning_target_v4
from mcp_server.contracts.reasoning_v4 import ReasoningTargetV4
from tools.model_processing.export_reasoning_v4_contract_artifacts import (
    ARTIFACT_FILENAMES,
    export_reasoning_v4_contract_artifacts,
)
from tools.model_processing.sync_reasoning_v4_cross_repo import (
    sync_reasoning_v4_cross_repo,
)


ROUTE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[4]
ROUTE_ARTIFACTS = ROUTE_ROOT / "schemas" / "reasoning_v4"
CONTRACT_ROOT = CODE_ROOT / "UrbanNav-Contracts"
CONTRACT_PACKAGE = CONTRACT_ROOT / "src" / "urbannav_contracts"
CONTRACT_ARTIFACTS = CONTRACT_PACKAGE / "schemas" / "reasoning_v4"


def test_committed_contract_artifacts_are_current(tmp_path: Path) -> None:
    generated = tmp_path / "reasoning_v4"
    export_reasoning_v4_contract_artifacts(generated)

    for filename in ARTIFACT_FILENAMES:
        assert (ROUTE_ARTIFACTS / filename).read_bytes() == (
            generated / filename
        ).read_bytes()
    export_reasoning_v4_contract_artifacts(
        ROUTE_ARTIFACTS,
        check=True,
    )


def test_contract_bundle_artifact_is_byte_identical() -> None:
    assert CONTRACT_ARTIFACTS.is_dir()

    for filename in ARTIFACT_FILENAMES:
        assert (CONTRACT_ARTIFACTS / filename).read_bytes() == (
            ROUTE_ARTIFACTS / filename
        ).read_bytes()


def test_generated_contract_python_modules_are_current() -> None:
    assert CONTRACT_ROOT.is_dir()

    report = sync_reasoning_v4_cross_repo(CONTRACT_ROOT, check=True)

    assert report["passed"] is True
    assert report["stale"] == []


def test_golden_valid_and_invalid_fixtures_have_parser_parity() -> None:
    assert CONTRACT_ROOT.is_dir()
    sys.path.insert(0, str(CONTRACT_ROOT / "src"))
    try:
        contract_module = importlib.import_module(
            "urbannav_contracts.reasoning_v4_text"
        )
        valid = _load_json("reasoning_v4_golden_valid.json")
        invalid = _load_json("reasoning_v4_golden_invalid.json")

        for case in valid["cases"]:
            target = ReasoningTargetV4.model_validate(case["value"])
            package_target = contract_module.ReasoningTargetV4.model_validate(
                case["value"]
            )
            assert target.model_dump(mode="json") == package_target.model_dump(
                mode="json"
            )
            route_parsed = parse_reasoning_target_v4(
                case["canonical_assistant_text"]
            )
            package_parsed = contract_module.parse_reasoning_target_v4(
                case["canonical_assistant_text"]
            )
            assert route_parsed.valid and package_parsed.valid

        for case in invalid["cases"]:
            route_error = _route_error(case)
            package_error = _package_error(case, contract_module)
            expected = str(case["expected_error_contains"])
            assert expected in route_error
            assert expected in package_error
    finally:
        sys.path.remove(str(CONTRACT_ROOT / "src"))


def _route_error(case: dict[str, Any]) -> str:
    """Return the RouteServer failure text for one invalid fixture."""

    if case["entrypoint"] == "reasoning_target":
        with pytest.raises((ValidationError, ValueError)) as captured:
            ReasoningTargetV4.model_validate(case["value"])
        return str(captured.value)
    parsed = parse_reasoning_target_v4(str(case["value"]))
    assert not parsed.valid
    return "\n".join(parsed.errors)


def _package_error(case: dict[str, Any], module: Any) -> str:
    """Return the standalone package failure text for one invalid fixture."""

    if case["entrypoint"] == "reasoning_target":
        with pytest.raises((ValidationError, ValueError)) as captured:
            module.ReasoningTargetV4.model_validate(case["value"])
        return str(captured.value)
    parsed = module.parse_reasoning_target_v4(str(case["value"]))
    assert not parsed.valid
    return "\n".join(parsed.errors)


def _load_json(filename: str) -> dict[str, Any]:
    """Load one canonical RouteServer contract artifact."""

    return json.loads(
        (ROUTE_ARTIFACTS / filename).read_text(encoding="utf-8")
    )
