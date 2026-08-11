"""Focused tests for RouteServer's canonical DF-OPD exchange boundary."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from mcp_server.contracts.df_opd import (
    DFOPDCanonicalSampleV1,
    DFOPDProfileV1,
    DFOPDTeacherViewV1,
    DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
    DF_OPD_ORACLE_CUE_ROLE,
    DF_OPD_PROFILE_2S16,
    DF_OPD_TEACHER_CUE_ROLE,
    choose_ts_donor_v1,
    df_opd_digest_v1,
)
from mcp_server.data.df_opd.export import build_df_opd_dtot_manifest_v1
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic


_POLICY_DIGEST = hashlib.sha256(b'df-opd-policy').hexdigest()
_CUE_DIGEST = hashlib.sha256(b'df-opd-cue').hexdigest()


def _sample() -> DFOPDCanonicalSampleV1:
    profile = DFOPDProfileV1(profile_id=DF_OPD_PROFILE_2S16, target_horizon_steps=16)
    images = ('current.jpg', 'history.jpg')
    roles = ('fpv.current', 'fpv.history_1')
    teacher_views = (
        DFOPDTeacherViewV1(
            condition='same_input_t0',
            images=images,
            image_roles=roles,
            privilege_policy_digest=_POLICY_DIGEST,
        ),
        DFOPDTeacherViewV1(
            condition='correct_privilege_tc',
            images=(*images, 'cue.png'),
            image_roles=(*roles, DF_OPD_TEACHER_CUE_ROLE),
            cue_class='straight',
            privilege_policy_digest=_POLICY_DIGEST,
            cue_artifact_sha256=_CUE_DIGEST,
        ),
        DFOPDTeacherViewV1(
            condition='shuffled_privilege_ts',
            images=(*images, 'ts-cue.png'),
            image_roles=(*roles, DF_OPD_TEACHER_CUE_ROLE),
            cue_class='turn_left',
            privilege_policy_digest=_POLICY_DIGEST,
            cue_artifact_sha256=_CUE_DIGEST,
        ),
        DFOPDTeacherViewV1(
            condition='exact_future_oracle_to',
            images=(*images, 'oracle.png'),
            image_roles=(*roles, DF_OPD_ORACLE_CUE_ROLE),
            cue_class='oracle',
            privilege_policy_digest=_POLICY_DIGEST,
            cue_artifact_sha256=_CUE_DIGEST,
        ),
    )
    payload = {
        'schema_version': DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
        'sample_id': 'sample-1',
        'source_group_id': 'clip-1',
        'split_group_id': 'clip-1',
        'dataset_source': 'sidewalk',
        'profile': profile,
        'images': images,
        'image_roles': roles,
        'decision_target': (
            '<Decision>{"selected_action_label":"straight","speed":"normal",'
            '"response":"proceed"}</Decision>'
        ),
        'trajectory_ego_t': tuple((float(index), 0.0, 0.0) for index in range(16)),
        'action_mask': (True,) * 16,
        'teacher_views': teacher_views,
    }
    serialized = {
        key: value.model_dump(mode='json') if hasattr(value, 'model_dump') else value
        for key, value in payload.items()
    }
    serialized['teacher_views'] = [view.model_dump(mode='json') for view in teacher_views]
    return DFOPDCanonicalSampleV1(**payload, canonical_digest=df_opd_digest_v1(serialized))


def test_canonical_export_keeps_student_and_teacher_media_separate():
    sample = _sample()
    row = sample.to_trainable_physical_row(
        messages=[{'role': 'user', 'content': 'go'}],
        teacher_condition='correct_privilege_tc',
    )

    assert row['images'] == ['current.jpg', 'history.jpg']
    assert row['teacher_images'] == ['current.jpg', 'history.jpg', 'cue.png']
    assert row['teacher_image_roles'][-1] == DF_OPD_TEACHER_CUE_ROLE
    assert row['df_opd_profile_digest'] == sample.profile.profile_digest
    assert row['df_opd_teacher_cue_sha256'] == _CUE_DIGEST
    assert row['teacher_messages'] == row['messages']
    t0_row = sample.to_trainable_physical_row(
        messages=[{'role': 'user', 'content': 'go'}],
        teacher_condition='same_input_t0',
    )
    assert t0_row['teacher_images'] == t0_row['images']
    assert t0_row['df_opd_teacher_cue_sha256'] is None


def test_oracle_ts_and_private_message_fields_are_rejected_from_trainable_export():
    sample = _sample()
    with pytest.raises(ValueError, match='evaluation-only'):
        sample.to_trainable_physical_row(
            messages=[{'role': 'user', 'content': 'go'}],
            teacher_condition='exact_future_oracle_to',
        )
    with pytest.raises(ValueError, match='evaluation-only'):
        sample.to_trainable_physical_row(
            messages=[{'role': 'user', 'content': 'go'}],
            teacher_condition='shuffled_privilege_ts',
        )

    with pytest.raises(ValueError, match='private DF-OPD field'):
        sample.to_trainable_physical_row(
            messages=[{'role': 'user', 'exact_future_trajectory': [1.0, 2.0, 3.0]}],
            teacher_condition='correct_privilege_tc',
        )


def test_canonical_sample_requires_complete_distinct_t0_tc_ts_to_views():
    sample = _sample()
    payload = sample.model_dump(mode='json', exclude={'canonical_digest'})
    payload['teacher_views'] = payload['teacher_views'][:-1]
    payload['canonical_digest'] = df_opd_digest_v1(payload)
    with pytest.raises(ValueError, match='at least 4 items'):
        DFOPDCanonicalSampleV1.model_validate(payload)

    payload = sample.model_dump(mode='json', exclude={'canonical_digest'})
    payload['teacher_views'][2]['cue_class'] = payload['teacher_views'][1]['cue_class']
    payload['canonical_digest'] = df_opd_digest_v1(payload)
    with pytest.raises(ValueError, match='TS cue_class must differ'):
        DFOPDCanonicalSampleV1.model_validate(payload)


def test_canonical_sample_rejects_noncanonical_or_semantically_invalid_decision_target():
    sample = _sample()
    payload = sample.model_dump(mode='json', exclude={'canonical_digest'})
    payload['decision_target'] = (
        '<Decision>{"selected_action_label":"straight","speed":"normal","response":"go"}</Decision>'
    )
    payload['canonical_digest'] = df_opd_digest_v1(payload)
    with pytest.raises(ValueError, match='speed or response'):
        DFOPDCanonicalSampleV1.model_validate(payload)

    payload = sample.model_dump(mode='json', exclude={'canonical_digest'})
    payload['decision_target'] = (
        '<Decision>{"speed":"normal","selected_action_label":"straight","response":"proceed"}</Decision>'
    )
    payload['canonical_digest'] = df_opd_digest_v1(payload)
    with pytest.raises(ValueError, match='fields must be canonical and ordered'):
        DFOPDCanonicalSampleV1.model_validate(payload)


def test_ts_derangement_is_deterministic_and_requires_different_class():
    candidates = {'sample-1': 'straight', 'sample-2': 'left', 'sample-3': 'right'}
    first = choose_ts_donor_v1(
        sample_id='sample-1', own_cue_class='straight', candidates=candidates, seed=7
    )
    second = choose_ts_donor_v1(
        sample_id='sample-1', own_cue_class='straight', candidates=candidates, seed=7
    )
    assert first == second
    assert candidates[first] != 'straight'
    with pytest.raises(ValueError, match='different coarse'):
        choose_ts_donor_v1(
            sample_id='sample-1', own_cue_class='straight', candidates={'sample-1': 'straight'}, seed=7
        )


def test_ms_swift_profile_transport_mirror_matches_canonical_contract():
    code_root = Path(__file__).resolve().parents[4]
    ms_root = code_root / 'ms-swift'
    sys.path.insert(0, str(ms_root))
    try:
        from swift.sidewalk_nav.contracts.df_opd import resolve_df_opd_profile

        canonical = DFOPDProfileV1(profile_id=DF_OPD_PROFILE_2S16, target_horizon_steps=16)
        mirrored = resolve_df_opd_profile(DF_OPD_PROFILE_2S16)
        assert mirrored.to_dict() == canonical.model_dump(mode='json')
        assert mirrored.profile_digest == canonical.profile_digest
    finally:
        sys.path.remove(str(ms_root))


def test_student_only_dtot_export_compiles_in_ms_swift_without_private_transport(tmp_path):
    canonical = _sample()
    canonical_path = tmp_path / 'canonical.jsonl'
    messages_path = tmp_path / 'messages.jsonl'
    output_path = tmp_path / 'dtot.jsonl'
    report_path = tmp_path / 'dtot.report.json'
    write_jsonl_atomic(canonical_path, [canonical.model_dump(mode='json')])
    write_jsonl_atomic(
        messages_path,
        [{'sample_id': canonical.sample_id, 'messages': [{'role': 'user', 'content': 'go'}]}],
    )
    build_df_opd_dtot_manifest_v1(canonical_path, messages_path, output_path, report_path)
    row = next(iter_jsonl(output_path))

    code_root = Path(__file__).resolve().parents[4]
    ms_root = code_root / 'ms-swift'
    sys.path.insert(0, str(ms_root))
    try:
        from swift.sidewalk_nav.prompts.df_opd_dtot import build_df_opd_dtot_student_row

        compiled = build_df_opd_dtot_student_row(row, profile=DF_OPD_PROFILE_2S16)
    finally:
        sys.path.remove(str(ms_root))

    assert compiled['messages'][-1]['role'] == 'assistant'
    assert compiled['messages'][-1]['content'].endswith('[TRAJ_15]')
    assert not any('teacher' in key or 'oracle' in key or 'privilege' in key for key in compiled)
