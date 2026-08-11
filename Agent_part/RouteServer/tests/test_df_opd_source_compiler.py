"""CPU tests for the Stage-I/II to canonical DF-OPD source boundary."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from mcp_server.contracts.reasoning_v4 import DecisionV4
from mcp_server.data.df_opd.oracle_renderer import render_df_opd_exact_future_oracle_cue
from mcp_server.data.df_opd.source_compiler import build_df_opd_canonical_from_stage_views_v1
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic

_PROFILE = 'fpv_degree10_terminal_centroid_v2'


def _stage_i_row(label: str, index: int) -> dict:
    return {
        'sample_id': f'stage-i-{index}',
        'source_group_id': f'group-{index}',
        'split_group_id': 'train',
        'dataset_source': 'sidewalk',
        'images': [f'current-{index}.jpg'],
        'image_roles': ['fpv.current'],
        'assistant_text': f'<FullCoT>{label}</FullCoT>',
    }


def _stage_ii_row(label: str, index: int) -> dict:
    return {
        'sample_id':
        f'stage-ii-{index}',
        'source_group_id':
        f'group-{index}',
        'split_group_id':
        'train',
        'dataset_source':
        'sidewalk',
        'images': [f'current-{index}.jpg'],
        'image_roles': ['fpv.current'],
        'messages': [
            {
                'role': 'system',
                'content': 'system'
            },
            {
                'role': 'user',
                'content': 'user'
            },
        ],
        'trajectory_ego_t':
        [[0.125 * (step + 1), 0.0, 0.0] for step in range(16)],
        'action_mask': [True] * 16,
        'direction_profile_id':
        _PROFILE,
        'direction_label_name':
        label,
    }


def _fake_stage_i(label: str) -> SimpleNamespace:
    return SimpleNamespace(
        assistant_text=f'<FullCoT>{label}</FullCoT>',
        reasoning_target=SimpleNamespace(reasoning=SimpleNamespace(
            decision=DecisionV4(
                selected_action_label=label,
                speed='normal',
                response='proceed',
            ))),
        direction_supervision=SimpleNamespace(
            profile_id=_PROFILE,
            label_name=label,
        ),
    )


def _patch_stage_i_parser(monkeypatch) -> None:
    import mcp_server.data.df_opd.source_compiler as source_compiler

    monkeypatch.setattr(source_compiler, 'StageIFullCoTViewV2',
                        SimpleNamespace)
    monkeypatch.setattr(
        source_compiler,
        'parse_stage_i_training_view_v2',
        lambda payload: _fake_stage_i('straight' if payload['sample_id'].
                                      endswith('0') else 'left_20'),
    )


def test_source_compiler_builds_four_private_views_and_student_only_messages(
        tmp_path, monkeypatch):
    _patch_stage_i_parser(monkeypatch)
    stage_i = tmp_path / 'stage_i.jsonl'
    stage_ii = tmp_path / 'stage_ii.jsonl'
    canonical = tmp_path / 'canonical.jsonl'
    messages = tmp_path / 'messages.jsonl'
    report = tmp_path / 'report.json'
    cue_root = tmp_path / 'cues'
    write_jsonl_atomic(
        stage_i, [_stage_i_row('straight', 0),
                  _stage_i_row('left_20', 1)])
    write_jsonl_atomic(
        stage_ii, [_stage_ii_row('straight', 0),
                   _stage_ii_row('left_20', 1)])

    result = build_df_opd_canonical_from_stage_views_v1(
        stage_i,
        stage_ii,
        canonical,
        messages,
        cue_root,
        report,
        profile_id='df_opd_2s16_q16_v1',
    )

    rows = list(iter_jsonl(canonical))
    message_rows = list(iter_jsonl(messages))
    assert result['emitted_rows'] == 2
    assert result['stage_i_input_rows'] == 2
    assert result['exact_join_rows'] == 2
    assert result['teacher_view_count'] == 8
    assert result['rendered_cue_count'] == 6
    assert result['student_message_count'] == 2
    assert len(rows) == 2
    assert all(len(row['teacher_views']) == 4 for row in rows)
    assert all(row['teacher_views'][1]['cue_class'] != row['teacher_views'][2]
               ['cue_class'] for row in rows)
    assert all(
        (cue_root /
         f"{hashlib.sha256(row['sample_id'].encode()).hexdigest()}.to.png"
         ).is_file() for row in rows)
    assert all([message['role']
                for message in row['messages']] == ['system', 'user']
               for row in message_rows)
    assert 'messages' not in rows[0]
    assert json.loads(
        report.read_text())['report_digest'] == result['report_digest']


def test_source_compiler_rejects_assistant_or_private_message_data(
        tmp_path, monkeypatch):
    _patch_stage_i_parser(monkeypatch)
    stage_i = tmp_path / 'stage_i.jsonl'
    stage_ii = tmp_path / 'stage_ii.jsonl'
    row = _stage_ii_row('straight', 0)
    row['messages'].append({'role': 'assistant', 'content': 'target'})
    write_jsonl_atomic(stage_i, [_stage_i_row('straight', 0)])
    write_jsonl_atomic(stage_ii, [row])

    with pytest.raises(ValueError, match='not Student-visible input'):
        build_df_opd_canonical_from_stage_views_v1(
            stage_i,
            stage_ii,
            tmp_path / 'canonical.jsonl',
            tmp_path / 'messages.jsonl',
            tmp_path / 'cues',
            tmp_path / 'report.json',
            profile_id='df_opd_2s16_q16_v1',
        )


def test_oracle_renderer_is_private_and_requires_a_valid_prefix(tmp_path):
    cue = render_df_opd_exact_future_oracle_cue(
        tmp_path / 'oracle.png',
        trajectory_ego_t=[(0.1, 0.0, 0.0)] * 16,
        action_mask=[True] * 8 + [False] * 8,
    )

    assert cue.cue_class == 'exact_future_oracle'
    assert (tmp_path / 'oracle.png').is_file()
    with pytest.raises(ValueError, match='valid-prefix'):
        render_df_opd_exact_future_oracle_cue(
            tmp_path / 'invalid.png',
            trajectory_ego_t=[(0.1, 0.0, 0.0)] * 16,
            action_mask=[True, False, True] + [False] * 13,
        )
