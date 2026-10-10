"""Small contract check for streaming migration and file-level resume."""

import json
from pathlib import Path

from mcp_server.data.history_candidate_batch import _key, run_batch


def test_extra_identity_retains_frame_number():
    assert _key('sidewalk', 'sidewalk:sidewalk:clip:151') == 'sidewalk:clip:000151'


def test_batch_keeps_rejected_answers_without_sidecars_and_resumes(tmp_path):
    media = tmp_path / 'media'
    package = media / 'sidewalk/packages/clips/000000'
    (package / 'manifests').mkdir(parents=True)
    (media / 'sidewalk/packages/clips.jsonl').write_text(json.dumps({
        'status': 'materialized', 'clip_id': 'clip', 'package_directory': 'clips/000000'}) + '\n')
    row = dict(source_parent_id='sidewalk:clip:001', split_group_id='sidewalk:clip', split='train',
               observation={'anchor_row': 1, 'source_ns': [1000000000]},
               source_row={'sidewalk_meta': {'clip_id': 'clip', 'frame_idx': 1, 'timestamp': 1000000000}})
    (package / 'manifests/observations.jsonl').write_text(json.dumps(row) + '\n')
    annotation = dict(source_group_id='sidewalk:clip:001', grounding_outputs={},
                      stage_records={'SceneGrounding': {'status': 'rejected', 'raw_answer': 'preserve me'}})
    (tmp_path / 'old.jsonl').write_text(json.dumps(annotation) + '\n')
    config = dict(dataset_root=str(media), annotation_root=str(tmp_path), datasets=['sidewalk'],
                  files=[dict(dataset='sidewalk', kind='reasoning', path='old.jsonl')])
    output = tmp_path / 'output'
    report = run_batch(config, output)
    migrated = json.loads((output / 'file-00/candidates.jsonl').read_text())
    assert migrated['annotation'] == annotation
    assert report['coverage']['sidewalk']['with_reasoning'] == 1
    assert report['formal_training_rows'] == 0
    assert run_batch(config, output) == report
