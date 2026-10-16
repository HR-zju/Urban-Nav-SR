"""Teacher exchange preserves raw rejected sections and explicit action lineage."""
import json

from tools.model_processing.export_teacher_history_views import export, recover_groundings, teacher_action


def _annotation():
    fields = {
        'task_grounding': {'objective': 'navigate_sidewalk', 'requirements': [], 'ambiguities': []},
        'scene_grounding': {'scene_type': 'urban_sidewalk', 'walkable_support': 'paved_sidewalk',
                            'key_entities': [], 'traffic_context': 'none'},
        'local_traversability': {'direction_affordances': []},
        'interaction_constraint_grounding': {'constraints': []},
    }
    return {'source_group_id': 'sidewalk:clip:old', 'grounding_outputs': fields,
            'decision_proposal': {'selected_action_label': 'left_90', 'speed': 'slow', 'response': 'proceed'},
            'stage_records': {'NavigationContextGrounding': {'status': 'rejected', 'raw_answer': json.dumps(
                {'task_direction': {'status': 'not_visible', 'source': 'none', 'direction': None}})}}}


def test_raw_section_recovery_and_teacher_projection():
    a = _annotation()
    assert recover_groundings(a)['navigation_context_grounding']['task_direction']['status'] == 'not_visible'
    result, transformation = teacher_action(a['decision_proposal'])
    assert result['selected_action_label'] == 'left_50plus'
    assert transformation == 'left_90->left_50plus'
    assert a['decision_proposal']['selected_action_label'] == 'left_90'


def test_portable_teacher_export_without_gt(tmp_path):
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    migration = tmp_path/'UrbanNavigation/annotation_migrations/history_time_only_20261010_r1/data'
    migration.mkdir(parents=True)
    (migration/'config.json').write_text(json.dumps({'files': [
        {'dataset': 'sidewalk', 'kind': 'reasoning', 'path': 'original/annotations.jsonl'}]}))
    sid = 'sidewalk:clip:new'
    write(migration/'file-00/candidates.jsonl', [{'new_anchor': {'source_group_id': sid}, 'annotation': _annotation()}])
    fill, review = tmp_path/'fill', tmp_path/'review'
    fill.mkdir(); review.mkdir()
    write(fill/'decisions.jsonl', [])
    (fill/'selection.json').write_text(json.dumps({'merged_files': []}))
    (review/'plan.json').write_text(json.dumps({'ids': []}))
    write(review/'reviews.jsonl', [])
    record = {'source_parent_id': sid, 'split_group_id': 'clip', 'split': 'train',
              'source_row': {'instruction': ''}, 'observation': {
                  'fpv_paths': [f'media/{i}.jpg' for i in range(9)],
                  'source_ns': list(range(0, 4_000_000_001, 500_000_000)), 'bev': None},
              'trajectory_sidecar': {'private_gt': 'KEEP_OUT'}}
    write(tmp_path/'dataset/urbannav_history_4s_2hz_v1/sidewalk/packages/clips/000000/manifests/observations.jsonl', [record])
    output = tmp_path/'export'
    export(tmp_path, fill, review, output)
    text = (output/'teacher_full_cot.jsonl').read_text()
    assert 'KEEP_OUT' not in text
    row = json.loads(text)
    assert row['source_group_id'] == sid
    assert row['reasoning_target']['reasoning']['decision']['selected_action_label'] == 'left_50plus'
    assert not row['images'][0].startswith('/')
    assert not (output/'pending.jsonl').read_text()
