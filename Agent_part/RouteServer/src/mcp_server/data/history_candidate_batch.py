"""Stream full-corpus, time-only annotation migration with disk-backed indexes.

This module uses only the standard library so the exact same code can run next
to the server's data without changing its training environment.
"""

import argparse
import hashlib
import html
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _rows(path):
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def _write(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def _save(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    partial.replace(path)


def _key(dataset, value):
    # SideWalk extra 使用双前缀和未补零帧号；新清单保留同一 frame_idx，统一身份表示。
    if dataset == 'sidewalk' and value.startswith('sidewalk:sidewalk:'):
        prefix, frame = value.removeprefix('sidewalk:').rsplit(':', 1)
        return prefix + ':' + frame.zfill(6)
    return value


def run_batch(config: dict, output: Path) -> dict:
    """Migrate all packaged observations against explicitly selected old files.

    Args:
        config: dataset_root, annotation_root and ordered files (dataset, kind,
            path, optional canonicalization_version). The caller freezes sources.
        output: New or interrupted run directory; completed files resume by receipt.

    Returns:
        Actual per-dataset coverage, missing-label counts and source limits.

    Raises:
        ValueError: If a frozen input changes or supplied configuration changes.
        JSONDecodeError: If an input JSONL is malformed; errors remain visible.

    An original prepared timestamp comes from retained source_row metadata.
    Matching chooses argmin |t_old - t_new| in the same dataset and clip, with
    |t_old - t_new| <= 100000000 ns; ties stay explicit. Stable IDs locate the
    original sampling evidence, not semantic eligibility. An unindexed old
    anchor is reported separately, never assigned an invented timestamp.
    """
    output.mkdir(parents=True, exist_ok=True)
    saved = output / 'config.json'
    if saved.exists() and json.loads(saved.read_text()) != config:
        raise ValueError('resume configuration changed')
    _save(saved, config)
    _save(output / 'visualization_manifest.json', {'review': 'review.html', 'scope': 'all_packaged_observations'})
    media = Path(config['dataset_root'])
    annotations = Path(config['annotation_root'])
    inventory = []
    for spec in config['files']:
        stat = (annotations / spec['path']).stat()
        inventory.append(dict(path=spec['path'], bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    frozen_inputs = output / 'input_inventory.json'
    if frozen_inputs.exists() and json.loads(frozen_inputs.read_text()) != inventory:
        raise ValueError('annotation input changed since run started')
    _save(frozen_inputs, inventory)
    db = sqlite3.connect(output / 'index.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS targets (
        key TEXT PRIMARY KEY, dataset TEXT, clip TEXT, old_ns INTEGER, new_ns INTEGER, ref TEXT);
      CREATE INDEX IF NOT EXISTS target_time ON targets(dataset,clip,new_ns);
      CREATE TABLE IF NOT EXISTS coverage (key TEXT, kind TEXT, PRIMARY KEY(key,kind));
      CREATE TABLE IF NOT EXISTS versions (file_id INTEGER, identity TEXT, line INTEGER,
        offset INTEGER, payload TEXT, PRIMARY KEY(file_id,identity));
      CREATE TABLE IF NOT EXISTS variants (key TEXT, slot TEXT, signature TEXT, origin TEXT,
        PRIMARY KEY(key,slot,signature,origin));
      CREATE INDEX IF NOT EXISTS variant_key ON variants(key,slot);
    ''')
    if not (output / 'target_report.json').exists():
        db.execute('DELETE FROM targets')
        counts = Counter()
        with (output / 'observations.jsonl.partial').open('w') as stream:
            for dataset in config['datasets']:
                for _, package in _rows(media / dataset / 'packages/clips.jsonl'):
                    if package['status'] != 'materialized':
                        counts[f'{dataset}:excluded_clips'] += 1
                        continue
                    package_path = Path(dataset) / 'packages' / package['package_directory']
                    manifest = package_path / 'manifests/observations.jsonl'
                    for line, row in _rows(media / manifest):
                        key = row['source_parent_id']; clip = package['clip_id']
                        if row['split_group_id'] != f'{dataset}:{clip}':
                            raise ValueError(f'package clip mismatch: {key}')
                        original = row['source_row']
                        meta = original if dataset == 'egowalk' else original['sidewalk_meta']
                        if meta['clip_id'] != clip or int(meta['frame_idx']) != row['observation']['anchor_row']:
                            raise ValueError(f'original sampling identity mismatch: {key}')
                        old_ns = int(meta['timestamp']) * (1000000 if dataset == 'egowalk' else 1)
                        new_ns = int(row['observation']['source_ns'][-1])
                        ref = dict(manifest=manifest.as_posix(), line=line, package=package_path.as_posix())
                        db.execute('INSERT INTO targets VALUES (?,?,?,?,?,?)',
                                   (key, dataset, clip, old_ns, new_ns, json.dumps(ref)))
                        _write(stream, dict(source_group_id=key, dataset=dataset, clip_id=clip,
                                            split=row['split'], original_timestamp_ns=old_ns,
                                            timestamp_ns=new_ns, observation_ref=ref))
                        counts[dataset] += 1
                        if sum(counts.values()) % 10000 == 0:
                            db.commit()
                            _save(output / 'progress.json', dict(stage='indexing_observations', counts=dict(counts)))
                            print('indexed', dict(counts), flush=True)
        db.commit()
        (output / 'observations.jsonl.partial').replace(output / 'observations.jsonl')
        _save(output / 'target_report.json', dict(counts))

    reports = []
    for file_id, spec in enumerate(config['files']):
        folder = output / f'file-{file_id:02d}'
        folder.mkdir(exist_ok=True)
        source = annotations / spec['path']
        stat = source.stat()
        identity = dict(path=spec['path'], bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        receipt = folder / 'report.json'
        if receipt.exists():
            report = json.loads(receipt.read_text())
            if report['input'] != identity:
                raise ValueError(f'completed input changed: {source}')
            reports.append(report)
            continue
        # 文件内按样本/问题取最后的同合同记录；不按 accepted 或旧 QC 选择。
        db.execute('DELETE FROM versions WHERE file_id=?', (file_id,))
        counts = Counter()
        with (folder / 'prior_versions.jsonl.partial').open('w') as prior:
            for line, raw in _rows(source):
                counts['input_rows'] += 1
                expected = spec.get('canonicalization_version')
                if expected and raw.get('metadata', {}).get('task_canonicalization_version') != expected:
                    _write(prior, dict(reason='different_run_contract', line=line, annotation=raw))
                    counts['different_run_contract'] += 1
                    continue
                key = _key(spec['dataset'], raw['source_group_id'])
                record_id = key
                if spec['kind'] == 'vqa':
                    q = raw['question']
                    record_id += ':' + q['task_id'] + ':' + str(q.get('conditioned_direction_label'))
                previous = db.execute('SELECT line,payload FROM versions WHERE file_id=? AND identity=?',
                                      (file_id, record_id)).fetchone()
                if previous:
                    _write(prior, dict(reason='earlier_append_only_revision', line=previous[0],
                                       annotation=json.loads(previous[1])))
                    counts['prior_versions_preserved'] += 1
                db.execute('INSERT OR REPLACE INTO versions VALUES (?,?,?,?,?)',
                           (file_id, record_id, line, 0, json.dumps(raw, ensure_ascii=False)))
                if counts['input_rows'] % 10000 == 0:
                    db.commit()
                    _save(output / 'progress.json', dict(stage='reading_annotations', file_id=file_id, counts=dict(counts)))
                    print('read', file_id, counts['input_rows'], flush=True)
        db.commit()
        with (folder / 'candidates.jsonl.partial').open('w') as matched, \
                (folder / 'unmatched_old.jsonl.partial').open('w') as unmatched:
            for line, payload in db.execute('SELECT line,payload FROM versions WHERE file_id=? ORDER BY line', (file_id,)):
                raw = json.loads(payload)
                key = _key(spec['dataset'], raw['source_group_id'])
                original = db.execute('SELECT clip,old_ns FROM targets WHERE key=? AND dataset=?',
                                      (key, spec['dataset'])).fetchone()
                origin = dict(file=spec['path'], line=line)
                if original is None:
                    _write(unmatched, dict(origin=origin, annotation=raw,
                                           reason='original_sampling_time_not_indexed'))
                    counts['old_time_unresolved'] += 1
                    continue
                clip, time = original
                override = config.get('verified_frame_times', {}).get(raw.get('source_fingerprint'))
                if override is not None:
                    time = int(override)
                nearby = db.execute('SELECT key,new_ns,ref FROM targets WHERE dataset=? AND clip=? '
                                    'AND new_ns BETWEEN ? AND ? ORDER BY ABS(new_ns-?)',
                                    (spec['dataset'], clip, time-100000000, time+100000000, time)).fetchall()
                if not nearby:
                    _write(unmatched, dict(origin=origin, annotation=raw, old_timestamp_ns=time,
                                           reason='outside_100ms_window'))
                    counts['outside_tolerance'] += 1
                    continue
                best = abs(nearby[0][1]-time)
                nearest = [r for r in nearby if abs(r[1]-time) == best]
                for target, new_ns, ref in nearest:
                    candidate = dict(kind=spec['kind'], status='pending_calibration', training_eligible=False,
                                     dataset=spec['dataset'], clip_id=clip, annotation=raw, origin=origin,
                                     old_timestamp_ns=time, time_delta_ns=time-new_ns, tolerance_ns=100000000,
                                     alignment_level='prepared_sampling_timestamps',
                                     new_anchor=dict(source_group_id=target, timestamp_ns=new_ns, observation_ref=json.loads(ref)),
                                     historical_diagnostics_used_for_filtering=False, ambiguous_time_match=len(nearest)>1)
                    _write(matched, candidate)
                    db.execute('INSERT OR IGNORE INTO coverage VALUES (?,?)', (target, spec['kind']))
                    values = {}
                    if spec['kind'] == 'reasoning':
                        values.update(raw.get('grounding_outputs') or {})
                        for name, record in raw.get('stage_records', {}).items():
                            values[name + ':raw_answer'] = record.get('raw_answer')
                    elif spec['kind'] == 'vqa':
                        question = raw['question']
                        values['VQA:' + question['task_id'] + ':' + str(question.get('conditioned_direction_label'))] = raw['answer']
                    for slot, value in values.items():
                        if value is not None:
                            signature = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                            db.execute('INSERT OR IGNORE INTO variants VALUES (?,?,?,?)',
                                       (target, slot, signature, json.dumps(origin)))
                    counts['matched_rows'] += 1
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError(f'input changed during migration: {source}')
        db.commit()
        for name in ['prior_versions.jsonl', 'candidates.jsonl', 'unmatched_old.jsonl']:
            (folder / (name + '.partial')).replace(folder / name)
        report = dict(input=identity, dataset=spec['dataset'], kind=spec['kind'], counts=dict(counts))
        _save(receipt, report)
        reports.append(report)
        print('completed_file', file_id, json.dumps(report), flush=True)

    summary = {}
    with (output / 'coverage.jsonl').open('w') as stream:
        for dataset in config['datasets']:
            counts = Counter()
            for key, clip, ref in db.execute('SELECT key,clip,ref FROM targets WHERE dataset=?', (dataset,)):
                kinds = {r[0] for r in db.execute('SELECT kind FROM coverage WHERE key=?', (key,))}
                counts['observations'] += 1
                counts['with_any_label' if kinds & {'reasoning', 'vqa'} else 'no_existing_label'] += 1
                for kind in kinds:
                    counts['with_' + kind] += 1
                _write(stream, dict(source_group_id=key, dataset=dataset, clip_id=clip,
                                    existing_label_kinds=sorted(kinds), observation_ref=json.loads(ref)))
            summary[dataset] = dict(counts)
    conflicts = 0
    with (output / 'conflicts.jsonl').open('w') as stream:
        for key, slot in db.execute('SELECT key,slot FROM variants GROUP BY key,slot HAVING COUNT(DISTINCT signature)>1'):
            variants = [dict(answer_identity=r[0], origin=json.loads(r[1])) for r in
                        db.execute('SELECT signature,origin FROM variants WHERE key=? AND slot=?', (key, slot))]
            _write(stream, dict(source_group_id=key, field=slot, variants=variants))
            conflicts += 1
    report = dict(status='available_source_migration_complete', coverage=summary, files=reports,
                  conflicting_fields=conflicts,
                  teacher_calls=0, formal_training_rows=0, source_limits=config.get('source_limits', []))
    _save(output / 'report.json', report)
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>全量时间关联迁移</title>'
        '<h1>全量时间关联迁移</h1><p>旧审核状态仅作历史诊断；未调用 LVLM，未导出正式训练集。</p>'
        '<p>逐 observation 覆盖与原始媒体位置：<a href="coverage.jsonl">coverage.jsonl</a>。'
        '完整候选位于每个 file-* 目录，source_group_id 和 observation_ref 定位新媒体。</p>'
        '<pre>' + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + '</pre>')
    _save(output / 'visualization_index.json', dict(review='review.html', coverage='coverage.jsonl'))
    _save(output / 'progress.json', dict(stage='complete', coverage=summary))
    db.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_batch(json.loads(args.config.read_text()), args.output), ensure_ascii=False, indent=2))
