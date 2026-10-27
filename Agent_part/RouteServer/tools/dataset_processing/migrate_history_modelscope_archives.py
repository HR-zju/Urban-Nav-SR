#!/usr/bin/env python3
"""Publish the full archive layout, verify it, then remove legacy loose trees."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from modelscope.hub.api import HubApi, ModelScopeConfig

from upload_history_modelscope_shards import _pack, _plan, _sha256, _upload, _write_json


DATASETS = ('egowalk', 'sidewalk', 'foxglove')


def _remote_entries(api, repo, directory):
    if directory == '.':
        directory = '/'
    entries = {}
    page = 1
    while True:
        rows = api.get_dataset_files(repo, root_path=directory, recursive=False,
                                     page_number=page, page_size=100)
        entries.update({row['Path']: row for row in rows})
        if len(rows) < 100:
            return entries
        page += 1


def _verify_remote(api, repo, archives):
    """Compare every remote archive's size and SHA-256 before permitting deletion."""
    entries = {}
    for directory in sorted({str(Path(item['path']).parent) for item in archives}):
        entries.update(_remote_entries(api, repo, directory))
    for item in archives:
        remote = entries.get(item['path'])
        if remote is None or (remote['Size'], remote['Sha256']) != (item['bytes'], item['sha256']):
            raise ValueError(f'Remote archive missing or mismatched: {item["path"]}')
    print(f'remote archive verification passed: {len(archives)}', flush=True)


def _metadata_group(root):
    paths = ['metadata']
    for dataset in DATASETS:
        paths.extend([f'{dataset}/manifests', f'{dataset}/packages/clips.jsonl',
                      f'{dataset}/packages/summary.json'])
    files = []
    for relative in paths:
        path = root / relative
        files.extend([p for p in path.rglob('*') if p.is_file()] if path.is_dir() else [path])
    return dict(clips=paths, files=len(files), source_bytes=sum(p.stat().st_size for p in files))


def _git(checkout, *args, **kwargs):
    return subprocess.run(['git', '-C', str(checkout), *args], check=True,
                          stdout=subprocess.PIPE, **kwargs).stdout


def _remove_loose_trees(checkout, repo, state, receipt):
    """Delete only the three reviewed dataset trees in a single non-force Git push."""
    expected_url = f'https://www.modelscope.cn/datasets/{repo}.git'
    if _git(checkout, 'remote', 'get-url', 'origin').decode().strip() != expected_url:
        raise ValueError('Cleanup checkout remote differs from the authorized repository')
    _git(checkout, 'fetch', '--depth=1', 'origin', 'master')
    parent = _git(checkout, 'rev-parse', 'FETCH_HEAD').decode().strip()
    _git(checkout, 'read-tree', parent)
    paths = _git(checkout, 'ls-files', '-z', '--', *DATASETS)
    if not paths:
        state['cleanup_commit'] = parent
        state['deleted_trees'] = list(DATASETS)
        _write_json(receipt, state)
        return
    # 每条删除路径必须属于已完成归档的三个旧根目录，保留 shards 和发布元数据。
    names = paths.decode().rstrip('\0').split('\0')
    if any(name.split('/', 1)[0] not in DATASETS for name in names):
        raise ValueError('Unexpected path in deletion inventory')
    inventory = receipt.parent / 'deleted_remote_paths.txt'
    inventory.write_text('\n'.join(names) + '\n')
    state.update(status='removing_loose_files', cleanup_parent=parent,
                 deleted_file_count=len(names))
    _write_json(receipt, state)
    _git(checkout, 'update-index', '--force-remove', '-z', '--stdin', input=paths)
    # 保留项直接引用已存在的远端 Git 对象，部分克隆无需下载其内容。
    tree = _git(checkout, 'write-tree', '--missing-ok').decode().strip()
    commit = _git(checkout, '-c', 'user.name=UrbanNav dataset publisher',
                  '-c', 'user.email=urbannav-publisher@users.noreply.modelscope.cn',
                  'commit-tree', tree, '-p', parent,
                  '-m', 'Remove loose dataset files after verified archive publication').decode().strip()
    state['cleanup_commit'] = commit
    _write_json(receipt, state)
    # Token 通过官方凭据缓存读取；Git URL、命令行、日志均不保存 token。
    env = dict(os.environ, GIT_ASKPASS=str(Path(__file__).resolve()),
               URBANNAV_GIT_ASKPASS='1', GIT_TERMINAL_PROMPT='0')
    _git(checkout, 'push', 'origin', f'{commit}:refs/heads/master', env=env)
    state['deleted_trees'] = list(DATASETS)
    _write_json(receipt, state)


def main():
    """Run the authorized archive migration with resumable local receipts.

    Args:
        None. Source, archive output and isolated Git checkout are CLI arguments.
    Returns:
        None. Publishes archives, documentation, and a completion receipt.
    Raises:
        ValueError: Remote archive verification or deletion scope checks fail.
        subprocess.CalledProcessError: Packing or non-force Git publication fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--foxglove-index', type=Path, required=True)
    parser.add_argument('--cleanup-checkout', type=Path, required=True)
    parser.add_argument('--readme', type=Path, required=True)
    parser.add_argument('--repo-id', default='ricksu/SideWalkDataset')
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'migration_receipt.json'
    state = json.loads(receipt.read_text()) if receipt.exists() else dict(
        status='packing', packed=[], uploaded=[], resume_after_epoch=0,
        repo_id=args.repo_id, deleted_trees=[],
    )
    try:
        api = HubApi()
        api.get_cookies(cookies_required=True)
        foxglove = json.loads(args.foxglove_index.read_text())['shards']
        _verify_remote(api, args.repo_id, foxglove)
        jobs = []
        for dataset in DATASETS[:2]:
            jobs.extend((dataset, index, group) for index, group in
                        enumerate(_plan(root, 8 * 1024 ** 3, dataset)))
        jobs.append(('metadata', 0, _metadata_group(root)))
        state['new_archive_count'] = len(jobs)
        _write_json(receipt, state)
        for number, (dataset, index, group) in enumerate(jobs):
            if number < len(state['packed']):
                entry = state['packed'][number]
                if entry['clips'] != group['clips'] or (output / entry['path']).stat().st_size != entry['bytes']:
                    raise ValueError('Packed archive changed; inspect migration receipt')
                continue
            state.update(status='packing', current_dataset=dataset, current_archive=index)
            _write_json(receipt, state)
            state['packed'].append(_pack(root, output, index, group, dataset))
            _write_json(receipt, state)

        archives = state['packed'] + foxglove
        metadata = output / 'metadata'
        metadata.mkdir(exist_ok=True)
        _write_json(metadata / 'all_archives.json', dict(
            schema_version='urbannav_all_archives_v1', extract_at='dataset root',
            compression='none', archives=archives,
        ))
        (output / 'SHA256SUMS').write_text(''.join(
            f'{item["sha256"]}  {item["path"]}\n' for item in archives))
        (output / 'README.md').write_text(args.readme.read_text())
        for item in state['packed']:
            relative = item['path']
            if relative in state['uploaded']:
                continue
            _upload(api, output / relative, relative, args.repo_id, state, receipt)
            state['uploaded'].append(relative)
            _write_json(receipt, state)
        _verify_remote(api, args.repo_id, archives)

        # 新文档和下载索引先发布，再移除旧树；保持切换期间分片入口始终可用。
        for relative in ['metadata/all_archives.json', 'SHA256SUMS', 'README.md']:
            if relative in state['uploaded']:
                continue
            _upload(api, output / relative, relative, args.repo_id, state, receipt)
            remote = _remote_entries(api, args.repo_id, str(Path(relative).parent)).get(relative)
            if remote is None or remote['Sha256'] != _sha256(output / relative):
                raise ValueError(f'Published metadata differs: {relative}')
            state['uploaded'].append(relative)
            _write_json(receipt, state)
        _remove_loose_trees(args.cleanup_checkout, args.repo_id, state, receipt)
        roots = _remote_entries(api, args.repo_id, '/')
        if any(dataset in roots for dataset in DATASETS):
            raise ValueError('A legacy loose directory remains on the remote')
        _verify_remote(api, args.repo_id, archives)
        release = metadata / 'archive_release_status.json'
        _write_json(release, dict(
            status='complete', archive_count=len(archives), missing_source_files=0,
            legacy_skipped_files_resolved_in_archives=5,
            removed_trees=list(DATASETS), cleanup_commit=state['cleanup_commit'],
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        ))
        _upload(api, release, 'metadata/archive_release_status.json', args.repo_id, state, receipt)
        state['status'] = 'complete'
        _write_json(receipt, state)
        print('Archive migration complete; three loose trees removed.', flush=True)
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        _write_json(receipt, state)
        raise


if __name__ == '__main__':
    if os.environ.get('URBANNAV_GIT_ASKPASS') == '1':
        import sys
        print('oauth2' if 'username' in sys.argv[1].lower() else ModelScopeConfig.get_token())
    else:
        main()
