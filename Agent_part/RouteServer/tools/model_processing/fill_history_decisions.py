"""Fill missing migrated Decisions from ordered images and preserved context.

This revision produces teacher proposals only. It keeps historical QC advisory,
does not consume GT trajectories, and never overwrites previous annotations.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from mcp_server.contracts.reasoning_v4 import DecisionProposalV4
from mcp_server.teacher.reasoning_v4_batch import VisionReasoningV4TeacherBackend
from mcp_server.teacher.reasoning_v4_profiles import get_reasoning_v4_teacher_profiles

LABELS = ['stop', 'straight', 'left_10', 'right_10', 'left_20', 'right_20',
          'left_30', 'right_30', 'left_40', 'right_40', 'left_50plus', 'right_50plus']
SYSTEM = """You annotate only the missing Decision for pedestrian/robot navigation.
Use the attached FPV images in chronological order and the optional current BEV.
Relative timestamps identify history and the current anchor. Decide the immediate
action at the current anchor, not an average action over the history.
The old grounding sections are reference context, NOT validated current truth.
Check them against the images. Historical QC rejection alone is not evidence
that a label is wrong. Do not rewrite Task, Scene, LocalTraversability,
NavigationContext or Interaction. If an unsupported or conflicting contextual
claim affects your decision, name that section in context_sections_to_review.
Use only the supplied ego-metric 12-way action_labels: negative angular bins
are left, positive bins right; 50plus represents angles of 50 degrees or more.
Distinguish pedestrian, cycle and vehicle signals and their applicable paths.
An applicable current stop requires stop/stop/stop. Moving actions cannot use
stop speed or response. Yield versus proceed must match visible interaction.
Do not infer stopping from the recording ending. Future GT is unavailable.
If evidence cannot support a decision, return decision=null and explain why.
Return JSON only: {"decision": {"selected_action_label":..., "speed":...,
"response":...} or null, "context_sections_to_review":[], "evidence_note":"..."}.
Use the provided Decision schema. Context checks are diagnostics, not edits.
"""


def _rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def _write(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def _has_answer(annotation):
    return bool(annotation.get('decision_proposal') or
                annotation.get('stage_records', {}).get('Decision', {}).get('raw_answer'))


def prepare(base, output):
    """Merge current own-ID answers before freezing the missing-Decision input.

    Args:
        base: Shared suh directory containing dataset and UrbanNavigation.
        output: New run directory.
    Returns:
        Selection counts and historical NavigationContext rejection reasons.
    Raises:
        ValueError: Duplicate migrated candidate or unlocated input.
        OSError: Missing required input or existing frozen sources.
    """
    runs = base / 'UrbanNavigation/annotation_runs'
    migration = base / 'UrbanNavigation/annotation_migrations/history_time_only_20261010_r1/data'
    media = base / 'dataset/urbannav_history_4s_2hz_v1'
    output.mkdir(parents=True, exist_ok=True)
    labels, origins, aliases = {}, {}, defaultdict(set)
    config = json.loads((migration / 'config.json').read_text())
    authoritative = []
    for i, entry in enumerate(config['files']):
        if entry['dataset'] not in ('sidewalk', 'foxglove') or entry['kind'] != 'reasoning':
            continue
        authoritative.append(runs / entry['path'])
        for row in _rows(migration / f'file-{i:02d}/candidates.jsonl'):
            sid = row['new_anchor']['source_group_id']
            if sid in labels:
                raise ValueError(f'multiple migrated candidates: {sid}')
            labels[sid] = row['annotation']
            origins[sid] = str(runs / entry['path'])
            aliases[row['annotation']['source_group_id']].add(sid)
    baseline_missing = {sid for sid, a in labels.items() if not _has_answer(a)}
    # Only current full sources and named later supplementation/repair outputs.
    files = set(authoritative)
    for run in runs.iterdir():
        if not run.is_dir() or run.resolve() == output.resolve():
            continue
        if run.name.startswith('vla_') or 'navigation_contract_repair' in run.name:
            files.update(run.glob('annotations.jsonl'))
            files.update(run.glob('*/annotations.jsonl'))
            files.update(run.glob('*/repair.jsonl'))
            files.update(run.glob('annotations/reasoning_v4_annotations.jsonl'))
    preserved_answers = set()
    for path in sorted(files, key=lambda p: (p.stat().st_mtime_ns, str(p))):
        for a in _rows(path):
            for sid in aliases.get(a['source_group_id'], ()):
                if _has_answer(a):
                    preserved_answers.add(sid)
                labels[sid] = a
                origins[sid] = str(path)
    missing = {sid: a for sid, a in labels.items() if not _has_answer(a) and sid not in preserved_answers}
    targets = {r['source_group_id']: r for r in _rows(migration / 'observations.jsonl') if r['source_group_id'] in missing}
    packages = defaultdict(set)
    for sid, target in targets.items():
        packages[target['observation_ref']['package']].add(sid)
    counts, reasons, seen = Counter(), Counter(), set()
    with (output / 'inputs.jsonl').open('x') as stream:
        for package, ids in packages.items():
            for record in _rows(media / package / 'manifests/observations.jsonl'):
                sid = record['source_parent_id']
                if sid not in ids:
                    continue
                annotation = missing[sid]
                obs = record['observation']
                images = [str(Path(package) / p) for p in obs['fpv_paths']]
                times = [(int(t)-int(obs['source_ns'][-1])) / 1e9 for t in obs['source_ns']]
                roles = [f'fpv.history_{i}' for i in range(8)] + ['fpv.current']
                if obs['bev']:
                    images.append(str(Path(package) / obs['bev']['bev_image_path']))
                    roles.append('bev.current')
                    times.append(-obs['bev']['age_ns']/1e9)
                context = {k: v.get('raw_answer') for k, v in annotation['stage_records'].items() if k != 'Decision'}
                nav = annotation['stage_records']['NavigationContextGrounding']
                reasons.update(nav.get('errors', []))
                source = record['source_row']
                _write(stream, {'source_group_id': sid, 'dataset': targets[sid]['dataset'],
                    'images': images, 'image_roles': roles, 'relative_times_sec': times,
                    'instruction': source.get('instruction', ''),
                    'grounding_outputs': annotation['grounding_outputs'], 'raw_context': context,
                    'original_annotation': origins[sid], 'original_source_group_id': annotation['source_group_id'],
                    'historical_navigation_diagnostics': nav.get('errors', [])})
                seen.add(sid)
                counts[targets[sid]['dataset']] += 1
    if seen != set(missing):
        raise ValueError(f'{len(set(missing)-seen)} missing inputs')
    report = {'baseline_missing': len(baseline_missing), 'remaining_missing': dict(counts),
              'existing_answers_preserved': len(preserved_answers), 'navigation_errors': dict(reasons),
              'merged_files': [str(p) for p in sorted(files)], 'no_gt_in_prompt': True}
    (output / 'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    profile = get_reasoning_v4_teacher_profiles('reasoning_v4_section_images_routed_visual_v1',
        direction_registry_version='urbannav_direction_profile_registry_v2')['Decision']
    (output / 'prompt.json').write_text(json.dumps({'system': SYSTEM, 'action_labels': LABELS,
        'decision_schema': DecisionProposalV4.model_json_schema(), 'reference_profile': asdict(profile),
        'change': 'all nine frames; unvalidated old context; allow insufficient evidence; decision only'},
        ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False), flush=True)


async def annotate(base, output, workers, limit):
    """Append resumable Decision proposals with raw responses and provenance.

    Args:
        base: Shared data root.
        output: Prepared run directory.
        workers: Concurrent HTTP requests to one TP4 service.
        limit: Optional one-sample startup check.
    Returns:
        None; append-only output and completion report.
    Raises:
        Exception: Provider and filesystem failures propagate with traceback.
    """
    destination = output / 'decisions.jsonl'
    done = {r['source_group_id'] for r in _rows(destination)} if destination.exists() else set()
    pending = [r for r in _rows(output / 'inputs.jsonl') if r['source_group_id'] not in done]
    if limit:
        pending = pending[:limit]
    queue = asyncio.Queue()
    for row in pending:
        queue.put_nowait(row)
    for _ in range(workers):
        queue.put_nowait(None)
    schema = DecisionProposalV4.model_json_schema()
    with destination.open('a', buffering=1) as stream:
        async def worker():
            backend = VisionReasoningV4TeacherBackend(api_key='local-vllm', media_root=base / 'dataset/urbannav_history_4s_2hz_v1')
            while (row := await queue.get()) is not None:
                visible = {k: row[k] for k in ('source_group_id', 'instruction', 'grounding_outputs', 'raw_context')}
                visible['image_order'] = [{'ordinal': i+1, 'role': role, 'relative_time_sec': t}
                    for i, (role, t) in enumerate(zip(row['image_roles'], row['relative_times_sec']))]
                visible.update(action_labels=LABELS, decision_schema=schema)
                generation = await backend.generate(SimpleNamespace(images=row['images'], image_roles=row['image_roles']),
                    question=json.dumps(visible, ensure_ascii=False), system_prompt=SYSTEM, image_detail='low')
                raw = generation.answer
                result = {'source_group_id': row['source_group_id'], 'original_annotation': row['original_annotation'],
                    'raw_answer': raw, 'telemetry': generation.telemetry, 'model': generation.model,
                    'prompt_revision': 'vla_history_decision_only_12way_v1', 'groundings_modified': False}
                # 内容解析失败留存原文供核查；不自动当成已通过正式监督。
                try:
                    parsed = json.loads(raw)
                    decision = DecisionProposalV4.model_validate(parsed['decision']) if parsed['decision'] is not None else None
                    if decision and decision.selected_action_label not in LABELS:
                        raise ValueError('selected_action_label outside ego-metric 12-way')
                    result.update(decision_proposal=decision.model_dump(mode='json') if decision else None,
                        context_sections_to_review=parsed['context_sections_to_review'], evidence_note=parsed['evidence_note'],
                        status='generated' if decision else 'insufficient_evidence')
                except (ValueError, KeyError, TypeError) as error:
                    result.update(status='parse_error', error=str(error))
                _write(stream, result)
                done.add(row['source_group_id'])
                print(f'[decision] {len(done)} {row["source_group_id"]} {result["status"]}', flush=True)
        async with asyncio.TaskGroup() as group:
            for _ in range(workers):
                group.create_task(worker())
    report = dict(Counter(r['status'] for r in _rows(destination)))
    (output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.base, args.output)
    else:
        asyncio.run(annotate(args.base, args.output, args.workers, args.limit))
