"""Verify the data package split keeps legacy imports identity-compatible."""

from mcp_server.data.canonical.reasoning_v4_source import build_reasoning_source_v4
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.data.openloop.benchmark import build_stage_i_openloop_reference_v1
from mcp_server.data.stage_i.views import project_stage_i_reasoning_view_v1
from mcp_server.data.stage_ii.trajectory import build_stage_ii_trajectory_target_v1
from mcp_server.data.vqa.questions import build_vqa_questions_v2

from mcp_server.data.jsonl_io import iter_jsonl as legacy_iter_jsonl
from mcp_server.data.openloop_benchmark import (
    build_stage_i_openloop_reference_v1 as legacy_openloop_reference,
)
from mcp_server.data.reasoning_v4_source import (
    build_reasoning_source_v4 as legacy_reasoning_source,
)
from mcp_server.data.stage_i_views import (
    project_stage_i_reasoning_view_v1 as legacy_stage_i_view,
)
from mcp_server.data.stage_ii_trajectory import (
    build_stage_ii_trajectory_target_v1 as legacy_trajectory_target,
)
from mcp_server.data.vqa_v2_questions import (
    build_vqa_questions_v2 as legacy_vqa_questions,
)


def test_new_and_legacy_data_imports_are_identity_compatible():
    assert iter_jsonl is legacy_iter_jsonl
    assert build_stage_i_openloop_reference_v1 is legacy_openloop_reference
    assert build_reasoning_source_v4 is legacy_reasoning_source
    assert project_stage_i_reasoning_view_v1 is legacy_stage_i_view
    assert build_stage_ii_trajectory_target_v1 is legacy_trajectory_target
    assert build_vqa_questions_v2 is legacy_vqa_questions
