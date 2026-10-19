"""Focused contracts for visual-goal provenance and section-only repair."""
import asyncio

from test_reasoning_v4 import _source, _FakeBackend
from mcp_server.agent.reasoning_v4_validation import validate_reasoning_section_v4
from mcp_server.contracts.reasoning_v4 import NavigationContextGroundingV4
from mcp_server.teacher.reasoning_v4_batch import annotate_reasoning_source_v4, ReasoningV4BatchConfig
from tools.model_processing.repair_reasoning_v4_sections import repair_sections


def test_visual_goal_provenance_keeps_instruction_rule():
    observation = _source().observation.model_copy(update={'instruction': 'Go to the yellow flower bed.'})
    for provenance, expected in [('visual_goal', []), ('instruction', ['instruction_task_direction_without_explicit_direction'])]:
        section = NavigationContextGroundingV4.model_validate({'task_direction': {
            'status': 'aligned', 'source': provenance, 'direction': 'left_10'}})
        assert validate_reasoning_section_v4('NavigationContextGrounding', section,
            observation=observation).errors == expected


def test_repair_reuses_four_sections_and_preserves_original():
    source = _source()
    config = ReasoningV4BatchConfig(max_retries=0)
    old = asyncio.run(annotate_reasoning_source_v4(source, backend=_FakeBackend(), config=config))
    before = old.model_dump_json()
    backend = _FakeBackend()
    reused = {name: old for name in ('TaskGrounding', 'SceneGrounding', 'LocalTraversability', 'InteractionConstraintGrounding')}
    new = asyncio.run(annotate_reasoning_source_v4(source, backend=backend, config=config,
        reused_groundings=reused))
    assert set(backend.requested_sections) == {'NavigationContextGrounding', 'Decision'}
    for name in reused:
        assert new.stage_records[name] == old.stage_records[name]
    assert old.model_dump_json() == before
    assert repair_sections(old) == set()
