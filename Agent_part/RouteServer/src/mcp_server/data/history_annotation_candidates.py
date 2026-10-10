"""Time-only candidate migration; semantic admission belongs to later calibration."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from mcp_server.data.history_label_reuse import history_observation
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4


def migrate_annotation_candidates(new_sources: Path, old_records: Path, old_anchors: Path,
                                  output: Path, *, tolerance_ns: int = 100_000_000,
                                  packaged_observations: bool = False) -> dict:
    """Attach every old annotation version to its nearest eligible new anchor.

    Args:
        new_sources: Canonical sources for the selected clip.
        old_records: JSONL wrappers with kind, annotation and origin fields.
        old_anchors: Prepared sampling records with source_group_id, dataset,
            clip_id, timestamp_ns and provenance; optional physical frame time.
        output: Fresh candidate-only output directory.
        tolerance_ns: Inclusive bound |t_old - t_new| <= tolerance_ns. A known
            physical frame timestamp takes priority over the nominal old time.
        packaged_observations: Read original observation rows, independently of
            trajectory sidecars and direction-label availability.

    Returns:
        Counts of matched versions, conflicts, unresolved rows and new anchors.

    Raises:
        ValueError: For negative tolerance or conflicting source-clock records.
        FileExistsError: When output already exists.

    Equal-distance ties retain all nearest destinations and mark ambiguity.
    Versions with different answers remain separate; exact duplicate records
    share one candidate while retaining every original file/line occurrence.
    """
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be nonnegative")
    anchors = {}
    for row in iter_jsonl(old_anchors):
        key = row['source_group_id']
        if key in anchors and row != anchors[key]:
            raise ValueError(f"conflicting old clock record: {key}")
        anchors[key] = row
    destinations = defaultdict(list)
    for index, raw in enumerate(iter_jsonl(new_sources)):
        if packaged_observations:
            dataset, clip = raw['split_group_id'].split(':', 1)
            # 语义候选只依赖实际 observation；轨迹 sidecar 缺失不删除迁移目标。
            destinations[(dataset, clip)].append({
                'source_group_id': raw['source_parent_id'], 'observation_record_index': index,
                'timestamp_ns': int(raw['observation']['source_ns'][-1]),
            })
            continue
        source = ReasoningSourceV4.model_validate(raw)
        obs = source.observation
        clip = obs.split_group_id.removeprefix(f'{obs.dataset_source}:')
        destinations[(obs.dataset_source, clip)].append({
            'source_group_id': obs.source_group_id, 'source_fingerprint': source.source_fingerprint,
            'timestamp_ns': int(history_observation(obs)['source_ns'][-1]),
        })
    versions = {}
    for wrapped in iter_jsonl(old_records):
        signature = (wrapped['kind'], json.dumps(wrapped['annotation'], sort_keys=True, ensure_ascii=False))
        if signature not in versions:
            versions[signature] = dict(kind=wrapped['kind'], annotation=wrapped['annotation'], origins=[])
        versions[signature]['origins'].append(wrapped['origin'])
    output.mkdir(parents=True, exist_ok=False)
    candidates, unmatched = [], []
    counts = Counter()
    for index, version in enumerate(versions.values()):
        key = version['annotation']['source_group_id']
        clock = anchors.get(key)
        reason = 'original_timestamp_unresolved'
        nearest = []
        if clock is not None:
            verified = clock.get('frame_times_by_source_fingerprint', {})
            time = int(verified.get(version['annotation'].get('source_fingerprint'),
                                   clock.get('current_frame_timestamp_ns', clock['timestamp_ns'])))
            eligible = destinations.get((clock['dataset'], clock['clip_id']), [])
            if eligible:
                distance = min(abs(time - target['timestamp_ns']) for target in eligible)
                if distance <= tolerance_ns:
                    nearest = [target for target in eligible if abs(time - target['timestamp_ns']) == distance]
                reason = 'outside_100ms_window' if tolerance_ns == 100_000_000 else 'outside_time_window'
            else:
                reason = 'dataset_or_clip_unmatched'
        if not nearest:
            unmatched.append(dict(**version, reason=reason, old_anchor=clock))
            counts['unmatched_versions'] += 1
            continue
        counts['matched_versions'] += 1
        for target in nearest:
            candidates.append(dict(
                **version, candidate_id=f'candidate-{index:06d}-{len(candidates):06d}',
                status='pending_calibration', training_eligible=False,
                old_anchor=clock, new_anchor=target, time_delta_ns=time - target['timestamp_ns'],
                tolerance_ns=tolerance_ns, ambiguous_time_match=len(nearest) > 1,
            ))
            counts[f"candidate_kind:{version['kind']}"] += 1
    # 分字段暴露答案冲突；prompt、方向合同与旧验收状态均不充当迁移筛选门禁。
    slots = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        raw = candidate['annotation']
        if candidate['kind'] == 'vqa':
            question = raw['question']
            slot = 'VQA:' + question['task_id'] + ':' + str(question.get('conditioned_direction_label'))
            values = {slot: {'question': question, 'answer': raw['answer']}}
        elif candidate['kind'] == 'reasoning':
            values = dict(raw.get('grounding_outputs') or {})
            for field in ['decision_proposal', 'action_reconciliation', 'composer_candidate',
                          'candidate_target', 'candidate_structure_record']:
                if field in raw:
                    values[field] = raw[field]
            decision = raw.get('stage_records', {}).get('Decision')
            if decision is not None:
                values['Decision_raw_answer'] = decision.get('raw_answer')
        else:
            values = {'reconciliation_audit': raw}
        for field, value in values.items():
            if value is not None:
                signature = json.dumps(value, sort_keys=True, ensure_ascii=False)
                slots[(candidate['new_anchor']['source_group_id'], field)][signature].append(candidate['candidate_id'])
    conflicts = [dict(source_group_id=key, field=field,
                      variants=[dict(value=json.loads(value), candidate_ids=ids) for value, ids in variants.items()])
                 for (key, field), variants in slots.items() if len(variants) > 1]
    covered = {row['new_anchor']['source_group_id'] for row in candidates}
    empty = [target for group in destinations.values() for target in group
             if target['source_group_id'] not in covered]
    write_jsonl_atomic(output / 'candidates.jsonl', candidates)
    write_jsonl_atomic(output / 'unmatched_old.jsonl', unmatched)
    write_jsonl_atomic(output / 'unmatched_new.jsonl', empty)
    write_jsonl_atomic(output / 'conflicts.jsonl', conflicts)
    report = dict(counts, old_occurrences=sum(len(v['origins']) for v in versions.values()),
                  unique_old_versions=len(versions), candidate_rows=len(candidates),
                  matched_new_anchors=len(covered), unmatched_new_anchors=len(empty),
                  conflicting_fields=len(conflicts), tolerance_ns=tolerance_ns,
                  teacher_calls=0, formal_training_rows=0, status='pending_calibration')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report
