"""Decision-only repair preserves previous answers and keeps GT out of prompts."""
import asyncio
import json

from tools.model_processing import fill_history_decisions as fill


def test_rejected_raw_decision_is_preserved():
    assert fill._has_answer({'stage_records': {'Decision': {'status': 'rejected', 'raw_answer': 'old answer'}}})
    assert not fill._has_answer({'stage_records': {'Decision': {'status': 'skipped', 'raw_answer': None}}})


def test_decision_only_input_and_resume(tmp_path, monkeypatch):
    row = {'source_group_id': 'sidewalk:clip:anchor', 'instruction': '',
           'images': [f'{i}.jpg' for i in range(10)],
           'image_roles': [f'fpv.history_{i}' for i in range(8)] + ['fpv.current', 'bev.current'],
           'relative_times_sec': [-4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, 0, -.2],
           'grounding_outputs': {'scene_grounding': {'description': 'old scene'}},
           'raw_context': {'NavigationContextGrounding': 'old rejected context'},
           'original_annotation': 'original.jsonl', 'trajectory_target': 'PRIVATE_GT'}
    (tmp_path / 'inputs.jsonl').write_text(json.dumps(row)+'\n')
    calls = []

    class Backend:
        def __init__(self, **kwargs):
            pass

        async def generate(self, observation, **kwargs):
            calls.append(kwargs)
            assert len(observation.images) == 10
            assert 'PRIVATE_GT' not in kwargs['question']
            payload = json.loads(kwargs['question'])
            assert payload['image_order'][8]['relative_time_sec'] == 0
            return fill.SimpleNamespace(answer=json.dumps({'decision': {'selected_action_label': 'straight',
                'speed': 'slow', 'response': 'yield'}, 'context_sections_to_review': [],
                'evidence_note': 'current crossing'}), telemetry={}, model='fixture')

    monkeypatch.setattr(fill, 'VisionReasoningV4TeacherBackend', Backend)
    asyncio.run(fill.annotate(tmp_path, tmp_path, 1, None))
    asyncio.run(fill.annotate(tmp_path, tmp_path, 1, None))
    assert len(calls) == 1
    result = json.loads((tmp_path / 'decisions.jsonl').read_text())
    assert result['status'] == 'generated'
    assert result['groundings_modified'] is False
