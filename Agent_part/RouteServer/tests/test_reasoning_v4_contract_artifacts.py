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
MS_ROOT = CODE_ROOT / "ms-swift"
MS_ARTIFACTS = (
    MS_ROOT / "swift" / "sidewalk_nav" / "schemas" / "reasoning_v4"
)


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


def test_ms_swift_contract_artifact_mirror_is_byte_identical() -> None:
    if not MS_ARTIFACTS.is_dir():
        pytest.skip("sibling ms-swift artifact mirror is unavailable")

    for filename in ARTIFACT_FILENAMES:
        assert (MS_ARTIFACTS / filename).read_bytes() == (
            ROUTE_ARTIFACTS / filename
        ).read_bytes()


def test_generated_ms_swift_python_mirrors_are_current() -> None:
    if not MS_ROOT.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")

    report = sync_reasoning_v4_cross_repo(MS_ROOT, check=True)

    assert report["passed"] is True
    assert report["stale"] == []


def test_golden_valid_and_invalid_fixtures_have_parser_parity() -> None:
    if not MS_ROOT.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(MS_ROOT))
    try:
        ms_contract = importlib.import_module(
            "swift.sidewalk_nav.reasoning_v4"
        )
        valid = _load_json("reasoning_v4_golden_valid.json")
        invalid = _load_json("reasoning_v4_golden_invalid.json")

        for case in valid["cases"]:
            target = ReasoningTargetV4.model_validate(case["value"])
            ms_target = ms_contract.ReasoningTargetV4.model_validate(
                case["value"]
            )
            assert target.model_dump(mode="json") == ms_target.model_dump(
                mode="json"
            )
            route_parsed = parse_reasoning_target_v4(
                case["canonical_assistant_text"]
            )
            ms_parsed = ms_contract.parse_reasoning_target_v4(
                case["canonical_assistant_text"]
            )
            assert route_parsed.valid and ms_parsed.valid

        for case in invalid["cases"]:
            route_error = _route_error(case)
            ms_error = _ms_error(case, ms_contract)
            expected = str(case["expected_error_contains"])
            assert expected in route_error
            assert expected in ms_error
    finally:
        sys.path.remove(str(MS_ROOT))


def _route_error(case: dict[str, Any]) -> str:
    """Return the RouteServer failure text for one invalid fixture."""

    if case["entrypoint"] == "reasoning_target":
        with pytest.raises((ValidationError, ValueError)) as captured:
            ReasoningTargetV4.model_validate(case["value"])
        return str(captured.value)
    parsed = parse_reasoning_target_v4(str(case["value"]))
    assert not parsed.valid
    return "\n".join(parsed.errors)


def _ms_error(case: dict[str, Any], module: Any) -> str:
    """Return the ms-swift failure text for one invalid fixture."""

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
