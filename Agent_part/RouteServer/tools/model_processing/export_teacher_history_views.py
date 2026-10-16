"""Merge migrated labels, Decision fill and bounded review into teacher views.

The original migration and annotation files remain immutable. Incomplete or
unresolved rows are written to pending.jsonl with their exact reason.
"""
import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from urbannav_contracts.teacher_stage_i import TeacherStageIViewV1
from mcp_server.teacher.reasoning_v4_teacher import parse_reasoning_v4_section_answer
from mcp_server.teacher.reasoning_v4_profiles import get_reasoning_v4_teacher_profiles
from tools.model_processing.review_history_decisions import SECTIONS, read_rows

FIELDS = {'TaskGrounding': 'task_grounding', **{s: field for s, (field, _) in SECTIONS.items()}}
PROFILES = get_reasoning_v4_teacher_profiles('reasoning_v4_section_images_routed_visual_v1',
    direction_registry_version='urbannav_direction_profile_registry_v2')


def recover_groundings(annotation):
    """Recover original typed sections irrespective of historical QC status.

    Args:
        annotation: Original reasoning record.
    Returns:
        Parsed section fields; original raw answers remain in source artifact.
    Raises:
        ValueError: A required original section is absent or unparsable.
    """
    groundings = copy.deepcopy(annotation['grounding_outputs'])
    for section, field in FIELDS.items():
        if groundings.get(field) is None:
            raw = annotation['stage_records'][section].get('raw_answer')
            if not raw:
                raise ValueError(f'missing section: {section}')
            result = parse_reasoning_v4_section_answer(raw, PROFILES[section])
            if not result.valid:
                raise ValueError(f'unparsed section {section}: {result.errors}')
            groundings[field] = result.section.model_dump(mode='json')
    return groundings


def teacher_action(decision):
    """Project legacy teacher angular actions into the saturated 12-way bins.

    Args:
        decision: Teacher tuple, never a GT-assembled candidate_target.
    Returns:
        Copied action tuple and transformation description.
    """
    result = dict(decision)
    label = result['selected_action_label']
    for side in ('left', 'right'):
        if label in {f'{side}_{angle}' for angle in range(50, 91, 10)}:
            result['selected_action_label'] = f'{side}_50plus'
    return result, f'{label}->{result["selected_action_label"]}' if label != result['selected_action_label'] else 'identity'


def export(base, fill, review, output):
    """Build teacher exchange views with frozen splits and pending receipts.

    Args:
        base: Shared suh root.
        fill: Completed Decision fill run.
        review: Completed bounded-review run; every selected ID must be present.
        output: New export directory; existing exports are never overwritten.
    Raises:
        ValueError: Missing review records or unresolved ambiguous lineage.
        OSError: Missing source or existing destination.
    """
    runs = base/'UrbanNavigation/annotation_runs'
    migration = base/'UrbanNavigation/annotation_migrations/history_time_only_20261010_r1/data'
    media = base/'dataset/urbannav_history_4s_2hz_v1'
    reviewed = {r['source_group_id']: r for r in read_rows(review/'reviews.jsonl')}
    plan = json.loads((review/'plan.json').read_text())
    if set(reviewed) != set(plan['ids']):
        raise ValueError('bounded review must finish before merging training views')
    filled = {r['source_group_id']: r for r in read_rows(fill/'decisions.jsonl')}
    labels, aliases, origins, lineage = {}, defaultdict(set), {}, {}
    config = json.loads((migration/'config.json').read_text())
    for i, entry in enumerate(config['files']):
        if entry['dataset'] not in ('sidewalk', 'foxglove') or entry['kind'] != 'reasoning':
            continue
        for row in read_rows(migration/f'file-{i:02d}/candidates.jsonl'):
            sid = row['new_anchor']['source_group_id']
            if sid in labels:
                raise ValueError(f'ambiguous own-source candidate: {sid}')
            labels[sid] = row['annotation']
            origins[sid] = entry['path']
            lineage[sid] = {'original_source_group_id': row['annotation']['source_group_id'],
                            'time_delta_ns': str(row.get('time_delta_ns', 'not_recorded')),
                            'source_chain': entry['path']}
            aliases[row['annotation']['source_group_id']].add(sid)
    # 使用已冻结的源文件集合；新增补标按自身新ID对齐，旧ID通过已验证映射关联。
    selection = json.loads((fill/'selection.json').read_text())
    for path in sorted(map(Path, selection['merged_files']), key=lambda p: (p.stat().st_mtime_ns, str(p))):
        for row in read_rows(path):
            own = row['source_group_id']
            targets = aliases.get(own, {own})
            for sid in targets:
                old = labels.get(sid)
                merged = copy.deepcopy(row)
                if old:
                    for field, value in old.get('grounding_outputs', {}).items():
                        if not merged['grounding_outputs'].get(field):
                            merged['grounding_outputs'][field] = value
                    for section, value in old['stage_records'].items():
                        if not merged['stage_records'].get(section, {}).get('raw_answer'):
                            merged['stage_records'][section] = value
                    if not merged.get('decision_proposal'):
                        merged['decision_proposal'] = old.get('decision_proposal')
                labels[sid], origins[sid] = merged, str(path.relative_to(runs))
                chain = lineage.setdefault(sid, {'original_source_group_id': own})
                chain['source_chain'] = chain.get('source_chain', '') + '\n' + origins[sid]
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    groups = {}
    with (output/'teacher_full_cot.jsonl').open('x') as views, (output/'pending.jsonl').open('x') as pending:
        for family in ('sidewalk', 'foxglove'):
            for manifest in sorted((media/family/'packages/clips').glob('*/manifests/observations.jsonl')):
                package = manifest.parent.parent.relative_to(media)
                for row in read_rows(manifest):
                    sid = row['source_parent_id']
                    counts[f'{family}:observations'] += 1
                    previous_split = groups.setdefault(row['split_group_id'], row['split'])
                    if previous_split != row['split']:
                        raise ValueError('source split group crosses partitions')
                    try:
                        if sid not in labels:
                            raise ValueError('missing own reasoning annotation')
                        a = labels[sid]
                        a = copy.deepcopy(a)
                        proposal = a.get('decision_proposal')
                        provenance = {'original': origins[sid], **lineage[sid]}
                        status = 'migrated_teacher'
                        if sid in filled:
                            proposal = filled[sid].get('decision_proposal')
                            provenance['decision_fill'] = fill.name+'/decisions.jsonl'
                        if sid in reviewed:
                            r = reviewed[sid]
                            if r['status'] != 'resolved':
                                raise ValueError('bounded review '+r['status'])
                            proposal = r['review']['decision']
                            for section, result in r['review']['sections'].items():
                                if result['verdict'] == 'corrected':
                                    a['grounding_outputs'][FIELDS[section]] = result['replacement']
                            provenance['section_review'] = review.name+'/reviews.jsonl'
                            status = 'section_reviewed_teacher'
                        if proposal is None:
                            raw = a['stage_records']['Decision'].get('raw_answer')
                            if not raw:
                                raise ValueError('missing teacher Decision')
                            parsed = parse_reasoning_v4_section_answer(raw, PROFILES['Decision'])
                            if not parsed.valid:
                                raise ValueError('unparsed teacher Decision: '+str(parsed.errors))
                            proposal = parsed.section.model_dump(mode='json')
                        groundings = recover_groundings(a)
                        proposal, provenance['decision_projection'] = teacher_action(proposal)
                        obs = row['observation']
                        images = [str(package/p) for p in obs['fpv_paths']]
                        roles = [f'fpv.history_{i}' for i in range(8)] + ['fpv.current']
                        times = [(int(t)-int(obs['source_ns'][-1]))/1e9 for t in obs['source_ns']]
                        timestamps = list(obs['source_ns'])
                        if obs['bev']:
                            images.append(str(package/obs['bev']['bev_image_path']))
                            roles.append('bev.current')
                            times.append(-obs['bev']['age_ns']/1e9)
                            timestamps.append(obs['bev']['timestamp'])
                        context_text = json.dumps(groundings)
                        context_profile = 'legacy_degree10' if any(
                            f'"{side}_{angle}"' in context_text for side in ('left', 'right')
                            for angle in range(50, 91, 10)) else 'ego_12way'
                        view = TeacherStageIViewV1(source_group_id=sid, split_group_id=row['split_group_id'],
                            split=row['split'], dataset_source=family, images=images, image_roles=roles,
                            relative_times_sec=times, instruction=row['source_row'].get('instruction') or '',
                            frame_timestamps_ns=timestamps,
                            reasoning_target={'reasoning': {**groundings, 'decision': proposal}},
                            context_direction_profile=context_profile, label_status=status, provenance=provenance)
                        views.write(view.model_dump_json()+'\n')
                        counts[f'{family}:exported:{row["split"]}'] += 1
                    except (ValueError, KeyError, TypeError) as error:
                        # 待处理记录独立保存，保留来源；不把历史QC诊断当作删除依据。
                        pending.write(json.dumps({'source_group_id': sid, 'reason': str(error),
                            'original': origins.get(sid), 'review': review.name if sid in reviewed else None})+'\n')
                        counts[f'{family}:pending'] += 1
    report = {'schema_version': 'urbannav_teacher_history_export_v1', 'counts': dict(counts),
              'decision_authority': 'teacher', 'gt_in_stage_i': False,
              'context_probe_policy': 'preserve original angular probes with explicit vocabulary',
              'review_scope': len(reviewed), 'full_corpus_visually_calibrated': False}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--fill', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.base, args.fill, args.review, args.output)
