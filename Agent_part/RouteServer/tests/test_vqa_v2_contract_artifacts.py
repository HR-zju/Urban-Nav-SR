from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.agent.vqa_v2_text import parse_vqa_answer_v2
import mcp_server.contracts.vqa_dataset_v2 as route_dataset_contract
from mcp_server.contracts.vqa_v2 import (
    VQAQuestionV2,
    VQA_V2_TASK_IDS,
)
from tools.model_processing.export_vqa_v2_contract_artifacts import (
    ARTIFACT_FILENAMES,
    export_vqa_v2_contract_artifacts,
)
from tools.model_processing.sync_vqa_v2_cross_repo import (
    sync_vqa_v2_cross_repo,
)


ROUTE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[4]
ROUTE_ARTIFACTS = ROUTE_ROOT / "schemas" / "vqa_v2"
MS_ROOT = CODE_ROOT / "ms-swift"
MS_ARTIFACTS = (
    MS_ROOT / "swift" / "sidewalk_nav" / "schemas" / "vqa_v2"
)


def test_committed_vqa_artifacts_are_current(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "vqa_v2"
    export_vqa_v2_contract_artifacts(generated)
    for filename in ARTIFACT_FILENAMES:
        assert (ROUTE_ARTIFACTS / filename).read_bytes() == (
            generated / filename
        ).read_bytes()
    export_vqa_v2_contract_artifacts(
        ROUTE_ARTIFACTS,
        check=True,
    )


def test_ms_swift_vqa_artifacts_are_byte_identical() -> None:
    if not MS_ARTIFACTS.is_dir():
        pytest.skip("sibling ms-swift mirror is unavailable")
    for filename in ARTIFACT_FILENAMES:
        assert (MS_ARTIFACTS / filename).read_bytes() == (
            ROUTE_ARTIFACTS / filename
        ).read_bytes()


def test_generated_vqa_python_mirrors_are_current() -> None:
    if not MS_ROOT.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    report = sync_vqa_v2_cross_repo(MS_ROOT, check=True)
    assert report["passed"] is True
    assert report["stale"] == []
    assert "_generated_direction_profile.py" not in report["sha256"]


def test_golden_vqa_parser_parity() -> None:
    if not MS_ROOT.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(MS_ROOT))
    try:
        ms_codec = importlib.import_module(
                "swift.sidewalk_nav.prompts.vqa_v2_codec"
        )
        valid = _load("vqa_v2_golden_valid.json")
        invalid = _load("vqa_v2_golden_invalid.json")
        assert {
            case["task_id"] for case in valid["cases"]
        } == set(VQA_V2_TASK_IDS)
        assert set(VQA_V2_TASK_IDS).issubset(
            {
                case["task_id"]
                for case in invalid["cases"]
                if case["entrypoint"] == "assistant_text"
            }
        )
        for case in valid["cases"]:
            question = VQAQuestionV2.model_validate(
                case["question"]
            )
            route = parse_vqa_answer_v2(
                case["canonical_assistant_text"],
                expected_task_id=case["task_id"],
                question=question,
            )
            ms_question_type = importlib.import_module(
                "swift.sidewalk_nav._generated_vqa_v2_contract"
            ).VQAQuestionV2
            ms_question = ms_question_type.model_validate(
                case["question"]
            )
            training = ms_codec.parse_vqa_answer_v2(
                case["canonical_assistant_text"],
                expected_task_id=case["task_id"],
                question=ms_question,
            )
            assert route.valid and training.valid
        for case in invalid["cases"]:
            route_error = _route_error(case)
            ms_error = _ms_error(case, ms_codec)
            assert case["expected_error_contains"] in route_error
            assert case["expected_error_contains"] in ms_error
    finally:
        sys.path.remove(str(MS_ROOT))


def test_dormant_interface_fixture_parity() -> None:
    if not MS_ROOT.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(MS_ROOT))
    try:
        ms_dataset_contract = importlib.import_module(
            "swift.sidewalk_nav._generated_vqa_dataset_v2"
        )
        valid = _load("vqa_v2_interface_golden_valid.json")
        invalid = _load(
            "vqa_v2_interface_golden_invalid.json"
        )
        expected_tasks = {
            "c2_pixel_goal_to_direction",
            "c2_waypoint_to_direction",
            "c3_bev_route_alignment",
            "c4_bev_object_anchoring",
            "c5_outside_view_direction",
            "c6_bev_fpv_executability",
            "e1_candidate_trajectory_preference",
            "e2_candidate_direction_classification",
            "e3_trajectory_compatibility",
        }
        assert {
            case["task_id"] for case in valid["cases"]
        } == expected_tasks
        assert expected_tasks.issubset(
            {case["task_id"] for case in invalid["cases"]}
        )
        for case in valid["cases"]:
            route_model = getattr(
                route_dataset_contract,
                case["interface_model"],
            )
            ms_model = getattr(
                ms_dataset_contract,
                case["interface_model"],
            )
            route_value = route_model.model_validate(case["value"])
            ms_value = ms_model.model_validate(case["value"])
            assert route_value.model_dump(mode="json") == (
                ms_value.model_dump(mode="json")
            )
        for case in invalid["cases"]:
            route_model = getattr(
                route_dataset_contract,
                case["interface_model"],
            )
            ms_model = getattr(
                ms_dataset_contract,
                case["interface_model"],
            )
            with pytest.raises(
                (ValidationError, ValueError)
            ) as route_error:
                route_model.model_validate(case["value"])
            with pytest.raises(
                (ValidationError, ValueError)
            ) as ms_error:
                ms_model.model_validate(case["value"])
            expected = case["expected_error_contains"]
            assert expected in str(route_error.value)
            assert expected in str(ms_error.value)
    finally:
        sys.path.remove(str(MS_ROOT))


def test_vqa_v1_production_modules_and_active_links_are_absent() -> None:
    if not MS_ROOT.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    removed = (
        MS_ROOT / "swift/sidewalk_nav/vqa_v1.py",
        MS_ROOT / "swift/sidewalk_nav/data/vqa_v1_manifest.py",
        MS_ROOT / "tests/test_sidewalk_vqa_v1.py",
        MS_ROOT
        / "docs/Task/sidewalk_navigation_vqa_v1_design.md",
    )
    assert all(not path.exists() for path in removed)
    for path in (MS_ROOT / "swift/sidewalk_nav").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "vqa_v1" not in str(node.module or "")
            elif isinstance(node, ast.Import):
                assert all(
                    "vqa_v1" not in alias.name
                    for alias in node.names
                )
    for path in (MS_ROOT / "docs").rglob("*.md"):
        assert "sidewalk_navigation_vqa_v1_design.md" not in (
            path.read_text(encoding="utf-8")
        )


def _route_error(case: dict) -> str:
    if case["entrypoint"] == "question":
        with pytest.raises((ValidationError, ValueError)) as captured:
            VQAQuestionV2.model_validate(case["value"])
        return str(captured.value)
    question = VQAQuestionV2.model_validate(case["question"])
    result = parse_vqa_answer_v2(
        case["value"],
        expected_task_id=case["task_id"],
        question=question,
    )
    assert not result.valid
    return "\n".join(result.errors)


def _ms_error(case: dict, codec) -> str:
    contract = importlib.import_module(
        "swift.sidewalk_nav._generated_vqa_v2_contract"
    )
    if case["entrypoint"] == "question":
        with pytest.raises((ValidationError, ValueError)) as captured:
            contract.VQAQuestionV2.model_validate(case["value"])
        return str(captured.value)
    question = contract.VQAQuestionV2.model_validate(
        case["question"]
    )
    result = codec.parse_vqa_answer_v2(
        case["value"],
        expected_task_id=case["task_id"],
        question=question,
    )
    assert not result.valid
    return "\n".join(result.errors)


def _load(filename: str) -> dict:
    return json.loads(
        (ROUTE_ARTIFACTS / filename).read_text(encoding="utf-8")
    )
