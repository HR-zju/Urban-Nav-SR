#!/usr/bin/env python3
"""Restore audited clip-tail GT into new portable Stage II manifests.

Original media, observations and semantic labels remain unchanged. The output
uses paths relative to --dataset-root, including in the merged training views.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION,
    StageIITrajectorySidecarV2, stage_ii_digest_v1,
)
from mcp_server.data.stage_ii.raw_trajectory_v2 import (
    StageIIRawTrajectoryErrorV2, StageIIRawTrajectoryReaderV2,
)
from mcp_server.data.stage_ii.trajectory import build_stage_ii_trajectory_target_v2
from swift.sidewalk_nav.data.stage_ii.stage_ii_manifest_v2 import build_direct_action_manifest_row_v2


def _rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def _portable(value, root):
    """Relocate media references without copying media or changing sample IDs."""
    if isinstance(value, dict):
        return {k: _portable(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v, root) for v in value]
    if isinstance(value, str) and value.startswith(str(root) + '/'):
        return str(Path(value).relative_to(root))
    return value


def _write(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def _mark_supervision(row):
    """Describe GT coverage, not recording duration or model predictions.

    有效监督终点为 K * dt；mask=false 的补齐点不延长真实监督区间。
    标签属于数据管理元信息，不进入模型提示词或改变loss权重。
    """
    valid_steps = sum(row['action_mask'])
    partial = valid_steps < len(row['action_mask'])
    row['trajectory_supervision'] = {
        'schema_version': 'urbannav_trajectory_supervision_v1',
        'kind': 'partial_horizon' if partial else 'full_horizon',
        'valid_steps': valid_steps,
        'supervised_horizon_sec': valid_steps * row['trajectory_policy']['target_dt_sec'],
        'target_horizon_sec': row['trajectory_policy']['horizon_seconds'],
        'padding': 'last_valid_pose' if partial else 'none',
        'full_4s_eval_eligible': not partial,
        'evaluation_group': 'partial_horizon' if partial else 'full_4s',
    }
    return row


def _split_views(output):
    """Preserve frozen splits so the combined inventory is not used as train."""
    counts = Counter()
    for dataset in ('sidewalk', 'foxglove'):
        with ExitStack() as stack:
            streams = {split: stack.enter_context((output / f'{dataset}_{split}.jsonl').open('x'))
                       for split in ('train', 'validation', 'test')}
            evaluation = {(split, group): stack.enter_context(
                (output / f'{dataset}_{split}_{group}.jsonl').open('x'))
                for split in ('validation', 'test') for group in ('full_4s', 'partial_horizon')}
            for row in _rows(output / f'{dataset}_stage_ii_direct_action_q16.jsonl'):
                _write(streams[row['split']], row)
                counts[f'{dataset}:{row["split"]}'] += 1
                if row['split'] in ('validation', 'test'):
                    group = row['trajectory_supervision']['evaluation_group']
                    _write(evaluation[(row['split'], group)], row)
                    counts[f'{dataset}:{row["split"]}:{group}'] += 1
    return dict(counts)


def main():
    """Build new sidecars and filtered manifests from explicit audited inputs.

    Args:
        None; paths are supplied by CLI arguments.
    Returns:
        None; writes new manifests, exclusions and a completion receipt.
    Raises:
        ValueError: On mismatched sample identity or malformed contracts.
        OSError: On unavailable inputs or an existing output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--audit-jsonl', required=True, type=Path)
    parser.add_argument('--old-sidecar-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = {r['sample_id']: r for r in _rows(args.audit_jsonl)}
    rejected = {r['sample_id']: r for r in _rows(args.old_sidecar_root / 'trajectory_rejected_v2.jsonl')
                if r['sample_id'] in audit}
    raw_roots = json.loads((args.old_sidecar_root / 'trajectory_build_report_v2.json').read_text())['raw_roots']
    reader = StageIIRawTrajectoryReaderV2({d: raw_roots[d] for d in ('sidewalk', 'foxglove')})
    # 审计已逐clip定位原始数组；复用这些明确路径，避免再次扫描全部媒体。
    reader._clip_indexes = {d: {r['clip_id']: Path(r['video']).parent for r in audit.values()
                                if r['dataset'] == d} for d in ('sidewalk', 'foxglove')}
    counts = Counter()
    blocked = set(audit)
    restored = set()
    with (output / 'exclusions.jsonl').open('x') as drops, (output / 'restored_sidecars.jsonl').open('x') as sidecars:
        for sid, item in audit.items():
            if item['status'] != 'partial_future' or item['alignment_review_required'] or item.get('available_125ms_steps', 0) == 0:
                _write(drops, {'sample_id': sid, 'status': item['status'],
                               'reason': 'alignment_review' if item['alignment_review_required'] else item.get('eligibility_reason', item['historical_reason'])})
        for dataset in ('sidewalk', 'foxglove'):
            plan = Path(json.loads((root / dataset / 'generation_receipt.json').read_text())['plan'])
            wanted = {sid for sid, r in audit.items() if r['dataset'] == dataset and r['status'] == 'partial_future'
                      and not r['alignment_review_required'] and r.get('available_125ms_steps', 0) > 0}
            visited = set()
            with (output / f'{dataset}_stage_ii_direct_action_q16.jsonl').open('x') as merged:
                existing = root / dataset / 'manifests' / f'{dataset}_stage_ii_direct_action_q16.jsonl'
                for row in _rows(existing):
                    if row['source_group_id'] in blocked:
                        counts[f'{dataset}:removed_existing'] += 1
                        continue
                    _write(merged, _portable(_mark_supervision(row), root))
                    counts[f'{dataset}:retained'] += 1
                for clip in _rows(root / dataset / 'packages/clips.jsonl'):
                    if clip.get('status') != 'materialized':
                        continue
                    if not any(audit[sid]['clip_id'] == clip['clip_id'] for sid in wanted):
                        continue
                    package = root / dataset / 'packages' / clip['package_directory']
                    sampling = json.loads((plan / clip['package_directory'] / 'summary.json').read_text())
                    for record in _rows(package / 'manifests/observations.jsonl'):
                        sid = record['source_parent_id']
                        if sid not in wanted:
                            continue
                        visited.add(sid)
                        source = record['source_row']
                        # 旧包缺省的预算从原采样计划恢复，保持原值而非新设过滤阈值。
                        record['observation']['bev']['bev_tolerance_ns'] = sampling['max_bev_age_ns']
                        try:
                            raw = reader.extract(dataset, source, full_horizon_required=False)
                        except StageIIRawTrajectoryErrorV2 as error:
                            _write(drops, {'sample_id': sid, 'reason': error.code})
                            counts[f'{dataset}:rejected:{error.code}'] += 1
                            continue
                        result = build_stage_ii_trajectory_target_v2(
                            sample_id=sid, dataset_source=dataset, clip_id=raw.clip_id, frame_idx=raw.frame_idx,
                            raw_trajectory=raw.trajectory_ego_timed, action_mask=raw.action_mask,
                            source_adapter_id=raw.source_adapter_id, trajectory_source=raw.trajectory_source,
                            allowed_trajectory_sources={raw.trajectory_source}, raw_timestamp_unit=raw.raw_timestamp_unit,
                            yaw_offset_deg=raw.yaw_offset_deg, full_horizon_required=False)
                        if not result.accepted:
                            reason = result.qc.reason_codes[0]
                            _write(drops, {'sample_id': sid, 'reason': reason,
                                           'status': 'invalid' if reason == 'trajectory_source_step_exceeded' else 'needs_review',
                                           'trajectory_qc': result.qc.model_dump(mode='json', exclude_none=True)})
                            counts[f'{dataset}:rejected:{reason}'] += 1
                            continue
                        old = rejected[sid]
                        payload = {k: old[k] for k in ('sample_id', 'source_group_id', 'split_group_id', 'dataset_source',
                                                       'split', 'source_manifest_name', 'source_manifest_sha256', 'source_line')}
                        payload.update(schema_version=STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION,
                                       trajectory_target=result.target.model_dump(mode='json'),
                                       trajectory_qc=result.qc.model_dump(mode='json', exclude_none=True))
                        payload['sidecar_digest'] = stage_ii_digest_v1(payload)
                        typed = StageIITrajectorySidecarV2.model_validate(payload)
                        row = build_direct_action_manifest_row_v2(source, typed.model_dump(mode='json'), query_count=16,
                            media_root=package, history_observation=record['observation'], history_max_lag_ns=sampling['max_lag_ns'])
                        # 部分未来不提供完整4秒方向监督；已有语义标签仍在原始记录中。
                        for key in ('direction_choice', 'direction_name', 'direction_choice_label', 'direction_source'):
                            row[key] = None
                        _write(sidecars, typed.model_dump(mode='json'))
                        _write(merged, _portable(_mark_supervision(row), root))
                        restored.add(sid)
                        counts[f'{dataset}:restored'] += 1
            if visited != wanted:
                raise ValueError(f'{dataset}: {len(wanted - visited)} audited candidates were not found')
            print(dataset, dict(counts), flush=True)
    receipt = {'complete': True, 'counts': dict(counts), 'dataset_root': str(root),
               'image_path_base': 'dataset_root', 'padding': 'last_valid_pose_with_false_mask',
               'invalid_excluded': sum(r['status'] == 'invalid' for r in audit.values()) + sum(
                   n for key, n in counts.items() if key.endswith(':rejected:trajectory_source_step_exceeded')),
               'restored': len(restored), 'server_sync': 'not_performed', 'semantic_labels': 'unchanged'}
    receipt['split_counts'] = _split_views(output)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
