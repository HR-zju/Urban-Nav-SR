"""Compile paired Stage-I/II source views into canonical DF-OPD artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mcp_server.agent.reasoning_v4_text import render_reasoning_section_v4
from mcp_server.contracts.df_opd import (
    DFOPDCanonicalSampleV1,
    DFOPDProfileV1,
    DFOPDTeacherViewV1,
    DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
    DF_OPD_ORACLE_CUE_ROLE,
    DF_OPD_PROFILES_V1,
    DF_OPD_TEACHER_CUE_ROLE,
    df_opd_digest_v1,
)
from mcp_server.contracts.direction_profile import coarse_motion_for_direction_label
from mcp_server.contracts.stage_i_sft import StageIFullCoTViewV2, parse_stage_i_training_view_v2
from mcp_server.data.df_opd.cue_renderer import (
    DF_OPD_CUE_RENDER_POLICY_DIGEST_V1,
    render_df_opd_coarse_motion_cue,
)
from mcp_server.data.df_opd.oracle_renderer import render_df_opd_exact_future_oracle_cue
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic

DF_OPD_CANONICAL_SOURCE_REPORT_SCHEMA_V1 = 'urbannav_df_opd_canonical_source_report_v1'
_PRIVATE_MESSAGE_KEY_PARTS = ('teacher', 'oracle', 'trajectory', 'action_mask',
                              'future_motion_cue')
_JoinKey = tuple[str, str, str, tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class _StageICandidate:
    """Only Stage-I fields permitted to cross into canonical DF-OPD data."""

    decision_target: str
    direction_profile_id: str
    direction_label_name: str


def build_df_opd_canonical_from_stage_views_v1(
    stage_i_full_cot_jsonl: str | Path,
    stage_ii_manifest_jsonl: str | Path,
    output_canonical_jsonl: str | Path,
    output_messages_jsonl: str | Path,
    cue_artifact_root: str | Path,
    report_json: str | Path,
    *,
    profile_id: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Join frozen Stage-I FullCoT and Stage-II targets into DF-OPD canonical rows.

    The join key deliberately includes group/split/source identity and ordered
    media, so a similar prompt cannot silently inherit another trajectory.
    Student messages come exclusively from Stage-II's system/user input; Stage-I
    contributes only its canonical Decision and FullCoT lineage authority.

    Args:
        stage_i_full_cot_jsonl: Frozen V2 FullCoT source JSONL.
        stage_ii_manifest_jsonl: Frozen 16/32-step physical Stage-II JSONL.
        output_canonical_jsonl: New private canonical DF-OPD destination.
        output_messages_jsonl: New Student-visible message mapping destination.
        cue_artifact_root: New private TC/TS/TO cue root.
        report_json: New metadata-only compilation report.
        profile_id: Exact DF-OPD profile that must match Stage-II target length.
        max_rows: Optional deterministic prefix limit for a review-first smoke.

    Returns:
        Metadata-only report binding all source and output fingerprints.

    Raises:
        ValueError: If source contracts, joins, messages, or profile geometry
            do not satisfy the fail-closed canonical boundary.
    """

    profile = _resolve_profile(profile_id)
    if max_rows is not None and max_rows <= 0:
        raise ValueError('max_rows must be positive when provided')
    stage_i_path = Path(stage_i_full_cot_jsonl).expanduser().resolve()
    stage_ii_path = Path(stage_ii_manifest_jsonl).expanduser().resolve()
    canonical_path = Path(output_canonical_jsonl).expanduser().resolve()
    messages_path = Path(output_messages_jsonl).expanduser().resolve()
    cue_root = Path(cue_artifact_root).expanduser().resolve()
    report_path = Path(report_json).expanduser().resolve()
    _require_distinct_outputs(stage_i_path, stage_ii_path, canonical_path,
                              messages_path, report_path)

    stage_ii_keys = _read_stage_ii_keys(stage_ii_path)
    candidates, stage_i_input_rows = _read_stage_i_candidates(
        stage_i_path, stage_ii_keys)
    missing = set(stage_ii_keys).difference(candidates)
    if missing:
        raise ValueError(
            f'{len(missing)} Stage-II rows have no exact Stage-I FullCoT join')
    # Validate all selected physical rows before generating a single cue byte.
    for line_number, row in enumerate(iter_jsonl(stage_ii_path), start=1):
        _validate_stage_ii_row(
            row, candidates[_stage_join_key(row,
                                            source_name='Stage-II',
                                            line_number=line_number)], profile,
            line_number)
    ts_sources = _build_ts_sources(candidates)

    canonical_rows: list[dict[str, Any]] = []
    message_rows: list[dict[str, Any]] = []
    selected = 0
    for line_number, row in enumerate(iter_jsonl(stage_ii_path), start=1):
        if max_rows is not None and selected >= max_rows:
            break
        key = _stage_join_key(row,
                              source_name='Stage-II',
                              line_number=line_number)
        candidate = candidates[key]
        _validate_stage_ii_row(row, candidate, profile, line_number)
        sample = _build_canonical_sample(
            row,
            candidate,
            ts_source=ts_sources[key],
            profile=profile,
            cue_root=cue_root,
        )
        canonical_rows.append(sample.model_dump(mode='json'))
        message_rows.append({
            'sample_id': sample.sample_id,
            'messages': _student_messages(row, line_number)
        })
        selected += 1

    write_jsonl_atomic(canonical_path, canonical_rows)
    write_jsonl_atomic(messages_path, message_rows)
    report = {
        'schema_version': DF_OPD_CANONICAL_SOURCE_REPORT_SCHEMA_V1,
        'private_artifact': True,
        'stage_i_full_cot_jsonl': str(stage_i_path),
        'stage_i_full_cot_jsonl_sha256': file_sha256(stage_i_path),
        'stage_ii_manifest_jsonl': str(stage_ii_path),
        'stage_ii_manifest_jsonl_sha256': file_sha256(stage_ii_path),
        'output_canonical_jsonl': str(canonical_path),
        'output_canonical_jsonl_sha256': file_sha256(canonical_path),
        'output_messages_jsonl': str(messages_path),
        'output_messages_jsonl_sha256': file_sha256(messages_path),
        'cue_artifact_root': str(cue_root),
        'profile_id': profile.profile_id,
        'profile_digest': profile.profile_digest,
        'stage_i_input_rows': stage_i_input_rows,
        'stage_i_matching_rows': len(candidates),
        'stage_ii_input_rows': len(stage_ii_keys),
        'exact_join_rows': len(candidates),
        'emitted_rows': selected,
        'teacher_view_count': selected * 4,
        'rendered_cue_count': selected * 3,
        'student_message_count': len(message_rows),
        'max_rows': max_rows,
        'report_digest': '',
    }
    report['report_digest'] = df_opd_digest_v1({
        key: value
        for key, value in report.items() if key != 'report_digest'
    })
    _write_json_atomic(report_path, report)
    return report


def _resolve_profile(profile_id: str) -> DFOPDProfileV1:
    try:
        return DF_OPD_PROFILES_V1[profile_id]
    except KeyError as exc:
        raise ValueError(f'unknown DF-OPD profile_id: {profile_id!r}') from exc


def _read_stage_ii_keys(path: Path) -> dict[_JoinKey, int]:
    """Read unique Stage-II physical identities without retaining private rows."""

    keys: dict[_JoinKey, int] = {}
    for line_number, row in enumerate(iter_jsonl(path), start=1):
        key = _stage_join_key(row,
                              source_name='Stage-II',
                              line_number=line_number)
        if key in keys:
            raise ValueError(
                f'duplicate Stage-II join key at lines {keys[key]} and {line_number}'
            )
        keys[key] = line_number
    if not keys:
        raise ValueError('Stage-II manifest is empty')
    return keys


def _read_stage_i_candidates(
    path: Path, requested: Mapping[_JoinKey, int]
) -> tuple[dict[_JoinKey, _StageICandidate], int]:
    """Parse the required V2 FullCoT rows and retain only canonical authority."""

    candidates: dict[_JoinKey, _StageICandidate] = {}
    seen: dict[_JoinKey, int] = {}
    input_rows = 0
    for line_number, row in enumerate(iter_jsonl(path), start=1):
        input_rows += 1
        key = _stage_join_key(row,
                              source_name='Stage-I',
                              line_number=line_number)
        if key in seen:
            raise ValueError(
                f'duplicate Stage-I join key at lines {seen[key]} and {line_number}'
            )
        seen[key] = line_number
        if key not in requested:
            continue
        parsed = parse_stage_i_training_view_v2(row)
        if not isinstance(parsed, StageIFullCoTViewV2):
            raise ValueError(
                f'Stage-I row {line_number} is not a V2 FullCoT view')
        decision = render_reasoning_section_v4(
            'Decision', parsed.reasoning_target.reasoning.decision)
        candidates[key] = _StageICandidate(
            decision_target=decision,
            direction_profile_id=parsed.direction_supervision.profile_id,
            direction_label_name=parsed.direction_supervision.label_name,
        )
    return candidates, input_rows


def _build_ts_sources(
    candidates: Mapping[_JoinKey, _StageICandidate]
) -> dict[_JoinKey, _StageICandidate]:
    """Derange within dataset source so every TS coarse class differs from TC."""

    grouped: dict[str, list[tuple[_JoinKey, _StageICandidate, str]]] = {}
    for key, candidate in candidates.items():
        coarse = coarse_motion_for_direction_label(
            candidate.direction_profile_id, candidate.direction_label_name)
        grouped.setdefault(key[2], []).append((key, candidate, coarse))
    resolved: dict[_JoinKey, _StageICandidate] = {}
    for dataset_source, entries in grouped.items():
        ordered = sorted(entries,
                         key=lambda item:
                         (item[0][0], item[0][1], item[0][3], item[0][4]))
        if len({item[2] for item in ordered}) < 2:
            raise ValueError(
                f'cannot construct TS derangement: dataset {dataset_source!r} has one coarse class'
            )
        for index, (key, _candidate, coarse) in enumerate(ordered):
            source = next(
                (item[1] for item in ordered[index + 1:] + ordered[:index + 1]
                 if item[2] != coarse), None)
            if source is None:
                raise ValueError(
                    f'cannot construct TS derangement for source group {key[0]!r}'
                )
            resolved[key] = source
    return resolved


def _build_canonical_sample(
    stage_ii: Mapping[str, Any],
    candidate: _StageICandidate,
    *,
    ts_source: _StageICandidate,
    profile: DFOPDProfileV1,
    cue_root: Path,
) -> DFOPDCanonicalSampleV1:
    """Build one validated four-view canonical sample and its private cues."""

    sample_id = _required_text(stage_ii, 'sample_id')
    suffix = hashlib.sha256(sample_id.encode('utf-8')).hexdigest()
    tc = render_df_opd_coarse_motion_cue(
        cue_root / f'{suffix}.tc.png',
        direction_profile_id=candidate.direction_profile_id,
        direction_label_name=candidate.direction_label_name,
    )
    ts = render_df_opd_coarse_motion_cue(
        cue_root / f'{suffix}.ts.png',
        direction_profile_id=ts_source.direction_profile_id,
        direction_label_name=ts_source.direction_label_name,
    )
    trajectory = tuple(
        tuple(float(value) for value in point)
        for point in stage_ii['trajectory_ego_t'])
    action_mask = tuple(bool(value) for value in stage_ii['action_mask'])
    oracle = render_df_opd_exact_future_oracle_cue(cue_root /
                                                   f'{suffix}.to.png',
                                                   trajectory_ego_t=trajectory,
                                                   action_mask=action_mask)
    images = tuple(str(value) for value in stage_ii['images'])
    roles = tuple(str(value) for value in stage_ii['image_roles'])
    views = (
        DFOPDTeacherViewV1(
            condition='same_input_t0',
            images=images,
            image_roles=roles,
            privilege_policy_digest=DF_OPD_CUE_RENDER_POLICY_DIGEST_V1,
        ),
        DFOPDTeacherViewV1(
            condition='correct_privilege_tc',
            images=(*images, tc.path),
            image_roles=(*roles, DF_OPD_TEACHER_CUE_ROLE),
            cue_class=tc.cue_class,
            privilege_policy_digest=tc.render_policy_digest,
            cue_artifact_sha256=tc.file_sha256,
        ),
        DFOPDTeacherViewV1(
            condition='shuffled_privilege_ts',
            images=(*images, ts.path),
            image_roles=(*roles, DF_OPD_TEACHER_CUE_ROLE),
            cue_class=ts.cue_class,
            privilege_policy_digest=ts.render_policy_digest,
            cue_artifact_sha256=ts.file_sha256,
        ),
        DFOPDTeacherViewV1(
            condition='exact_future_oracle_to',
            images=(*images, oracle.path),
            image_roles=(*roles, DF_OPD_ORACLE_CUE_ROLE),
            cue_class=oracle.cue_class,
            privilege_policy_digest=oracle.render_policy_digest,
            cue_artifact_sha256=oracle.file_sha256,
        ),
    )
    payload = {
        'schema_version': DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
        'sample_id': sample_id,
        'source_group_id': _required_text(stage_ii, 'source_group_id'),
        'split_group_id': _required_text(stage_ii, 'split_group_id'),
        'dataset_source': _required_text(stage_ii, 'dataset_source'),
        'profile': profile,
        'images': images,
        'image_roles': roles,
        'decision_target': candidate.decision_target,
        'trajectory_ego_t': trajectory,
        'action_mask': action_mask,
        'teacher_views': views,
    }
    serialized = {
        key:
        value.model_dump(
            mode='json') if hasattr(value, 'model_dump') else value
        for key, value in payload.items()
    }
    serialized['teacher_views'] = [
        view.model_dump(mode='json') for view in views
    ]
    return DFOPDCanonicalSampleV1(**payload,
                                  canonical_digest=df_opd_digest_v1(
                                      serialized))


def _validate_stage_ii_row(
    row: Mapping[str, Any],
    candidate: _StageICandidate,
    profile: DFOPDProfileV1,
    line_number: int,
) -> None:
    """Reject Stage-II drift before any private cue bytes are written."""

    trajectory = row.get('trajectory_ego_t')
    action_mask = row.get('action_mask')
    if not isinstance(trajectory, list) or not isinstance(action_mask, list):
        raise ValueError(
            f'Stage-II row {line_number} lacks trajectory_ego_t/action_mask lists'
        )
    if len(trajectory) != profile.target_horizon_steps or len(
            action_mask) != profile.target_horizon_steps:
        raise ValueError(
            f'Stage-II row {line_number} target length does not match {profile.profile_id}'
        )
    if any(not isinstance(value, bool) for value in action_mask):
        raise ValueError(
            f'Stage-II row {line_number} action_mask must contain booleans')
    if row.get('direction_profile_id') != candidate.direction_profile_id:
        raise ValueError(
            f'Stage-II row {line_number} direction profile drifts from Stage-I FullCoT'
        )
    if row.get('direction_label_name') != candidate.direction_label_name:
        raise ValueError(
            f'Stage-II row {line_number} direction label drifts from Stage-I FullCoT'
        )
    _student_messages(row, line_number)


def _student_messages(row: Mapping[str, Any],
                      line_number: int) -> list[dict[str, Any]]:
    """Copy only structured system/user input after recursive private-key audit."""

    messages = row.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f'Stage-II row {line_number} requires non-empty messages')
    safe: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(
                f'Stage-II row {line_number} contains a non-object message')
        role = message.get('role')
        if role not in {'system', 'user'}:
            raise ValueError(
                f'Stage-II row {line_number} message role {role!r} is not Student-visible input'
            )
        copied = dict(message)
        _reject_private_message_keys(copied, line_number)
        safe.append(copied)
    return safe


def _reject_private_message_keys(value: Any, line_number: int) -> None:
    """Reject structured target/Teacher data before writing a message artifact."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in _PRIVATE_MESSAGE_KEY_PARTS):
                raise ValueError(
                    f'Stage-II row {line_number} messages contain private key {key!r}'
                )
            _reject_private_message_keys(nested, line_number)
    elif isinstance(value, list):
        for nested in value:
            _reject_private_message_keys(nested, line_number)


def _stage_join_key(row: Mapping[str, Any], *, source_name: str,
                    line_number: int) -> _JoinKey:
    """Return the exact identity contract shared by Stage-I and Stage-II."""

    fields = tuple(
        _required_text(row, field)
        for field in ('source_group_id', 'split_group_id', 'dataset_source'))
    images = row.get('images')
    roles = row.get('image_roles')
    if not isinstance(images, list) or not images or not isinstance(
            roles, list) or len(images) != len(roles):
        raise ValueError(
            f'{source_name} row {line_number} has invalid ordered images/image_roles'
        )
    return (*fields, tuple(str(value) for value in images),
            tuple(str(value) for value in roles))


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f'missing non-empty {field!r}')
    return value


def _require_distinct_outputs(*paths: Path) -> None:
    """Prevent source or sibling output replacement through a mistaken path."""

    if len(set(paths)) != len(paths):
        raise ValueError(
            'DF-OPD canonical compiler source and output paths must be distinct'
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one metadata object atomically, without a transient report gap."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, 'w', encoding='utf-8') as handle:
            json.dump(dict(payload),
                      handle,
                      ensure_ascii=False,
                      sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    'DF_OPD_CANONICAL_SOURCE_REPORT_SCHEMA_V1',
    'build_df_opd_canonical_from_stage_views_v1',
]
