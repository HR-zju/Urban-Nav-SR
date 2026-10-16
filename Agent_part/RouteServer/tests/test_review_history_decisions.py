"""Bounded context review keeps pending evidence separate from resolved edits."""
import json
import pytest
from tools.model_processing.review_history_decisions import select_cases, validate_review


def test_select_only_exceptions():
    inputs = [{'source_group_id': str(i)} for i in range(3)]
    proposals = [{'source_group_id': str(i), 'status': 'generated' if i < 2 else 'parse_error',
                  'context_sections_to_review': ['SceneGrounding'] if i == 1 else [],
                  'raw_answer': json.dumps({'context_sections_to_review': [] if i != 1 else ['SceneGrounding']})}
                 for i in range(3)]
    assert len(select_cases(inputs, proposals)) == 2


def test_pending_context_and_invalid_direction():
    case = {'requested_sections': ['SceneGrounding'], 'input': {'image_roles': ['fpv.current']}}
    result = {'decision': {'selected_action_label': 'straight', 'speed': 'normal', 'response': 'proceed'},
              'decision_reason': 'Clear current path', 'decision_evidence_frames': ['fpv.current'],
              'sections': {'SceneGrounding': {'verdict': 'insufficient_evidence',
                  'reason': 'Occlusion', 'evidence_frames': ['fpv.current'], 'replacement': None}}}
    assert validate_review(json.dumps(result), case)[1] == 'pending'
    result['decision']['selected_action_label'] = 'left_90'
    with pytest.raises(ValueError, match='12-way'):
        validate_review(json.dumps(result), case)
