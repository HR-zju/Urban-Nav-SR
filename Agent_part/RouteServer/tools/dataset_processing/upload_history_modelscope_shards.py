#!/usr/bin/env python3
"""Pack Foxglove into clip-aligned tar shards and publish the mixed-layout dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

import requests
from modelscope.hub.api import HubApi


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _plan(root, limit, dataset='foxglove'):
    """Group complete clips by logical bytes; each shard extracts at dataset root."""
    groups, clips, size, count = [], [], 0, 0
    index = root / dataset / 'packages/clips.jsonl'
    for line in index.read_text().splitlines():
        row = json.loads(line)
        if row['status'] != 'materialized':
            continue
        relative = Path(dataset) / 'packages' / row['package_directory']
        files = [p for p in (root / relative).rglob('*') if p.is_file()]
        clip_size = sum(p.stat().st_size for p in files)
        if clips and size + clip_size > limit:
            groups.append(dict(clips=clips, source_bytes=size, files=count))
            clips, size, count = [], 0, 0
        clips.append(relative.as_posix())
        size += clip_size
        count += len(files)
    if clips:
        groups.append(dict(clips=clips, source_bytes=size, files=count))
    return groups


def _pack(root, output, index, group, dataset='foxglove'):
    """Build, inspect and hash one shard before its atomic publication locally."""
    relative = f'shards/{dataset}/{dataset}-{index:05d}.tar'
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.tar.partial')
    print(f'packing {relative}: {len(group["clips"])} clips', flush=True)
    # --hard-dereference 使每片独立可解包；不依赖本机硬链接或其他分片。
    subprocess.run(
        ['tar', '--hard-dereference', '-C', str(root), '-cf', str(partial),
         '--null', '-T', '-'],
        input=b''.join(p.encode() + b'\0' for p in group['clips']), check=True,
    )
    count, size = 0, 0
    with tarfile.open(partial) as archive:
        for member in archive:
            if member.isfile():
                count += 1
                size += member.size
    if (count, size) != (group['files'], group['source_bytes']):
        raise ValueError(f'{relative}: tar member counts/bytes differ from source')
    digest = _sha256(partial)
    os.replace(partial, target)
    return dict(group, path=relative, bytes=target.stat().st_size, sha256=digest)


def _wait_until(deadline):
    while time.time() < deadline:
        time.sleep(min(60, deadline - time.time()))


def _upload(api, source, remote, repo, state, receipt):
    """Wait on repository quota; bound transient retries and propagate other failures."""
    transient_attempts = 0
    while True:
        _wait_until(state.get('resume_after_epoch', 0))
        state.update(status='uploading', current_file=remote)
        _write_json(receipt, state)
        try:
            api.upload_file(
                path_or_fileobj=str(source), path_in_repo=remote,
                repo_id=repo, repo_type='dataset',
                commit_message=f'Publish {remote}', tqdm_desc=remote,
            )
            return
        except (ValueError, requests.exceptions.RequestException) as exc:
            message = str(exc)
            delay = re.search(r'try again after (\d+) seconds', message, re.I)
            if '429' in message and delay:
                deadline = time.time() + int(delay.group(1)) + 5
                state.update(status='waiting_for_quota', resume_after_epoch=deadline,
                             resume_after_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat())
                _write_json(receipt, state)
                print(f'quota wait until {state["resume_after_utc"]}: {remote}', flush=True)
                continue
            response = getattr(exc, 'response', None)
            retryable = isinstance(exc, (requests.ConnectionError, requests.Timeout)) or (
                response is not None and response.status_code >= 500)
            if retryable and transient_attempts < 3:
                transient_attempts += 1
                print(f'transient retry {transient_attempts}: {remote}: {exc}', flush=True)
                time.sleep(10 * transient_attempts)
                continue
            raise


def main():
    """Package and upload using local SDK credentials and persistent progress.

    Args:
        None. Paths, repository and quota deadline are supplied through the CLI.
    Returns:
        None. Writes archive indexes and progress receipts.
    Raises:
        OSError: Local data cannot be read or written.
        ValueError: Archive verification or remote publication fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--previous-receipt', type=Path, required=True)
    parser.add_argument('--readme', type=Path, required=True)
    parser.add_argument('--repo-id', default='ricksu/SideWalkDataset')
    parser.add_argument('--shard-gib', type=float, default=8)
    parser.add_argument('--not-before', type=float, default=0)
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output_root.resolve()
    receipt = output / 'upload_receipt.json'
    state = json.loads(receipt.read_text()) if receipt.exists() else dict(
        status='packing', packed=[], uploaded=[], repo_id=args.repo_id,
        resume_after_epoch=args.not_before,
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        plan = _plan(root, int(args.shard_gib * 1024 ** 3))
        state['shard_count'] = len(plan)
        _write_json(receipt, state)
        for index, group in enumerate(plan):
            if index < len(state['packed']):
                entry = state['packed'][index]
                if entry['clips'] != group['clips'] or (output / entry['path']).stat().st_size != entry['bytes']:
                    raise ValueError('Shard plan or packed file changed; inspect the receipt')
                continue
            state['current_shard'] = index
            _write_json(receipt, state)
            state['packed'].append(_pack(root, output, index, group))
            _write_json(receipt, state)

        metadata = output / 'metadata'
        metadata.mkdir(exist_ok=True)
        _write_json(metadata / 'foxglove_shards.json', dict(
            schema_version='urbannav_tar_shards_v1', dataset='foxglove',
            archive_format='tar', compression='none', extract_at='dataset root',
            shards=state['packed'],
        ))
        (output / 'SHA256SUMS').write_text(''.join(
            f'{entry["sha256"]}  {entry["path"]}\n' for entry in state['packed']))
        previous = json.loads(args.previous_receipt.read_text())
        _write_json(metadata / 'upload_skipped_files.json', dict(
            scope='legacy loose-file upload', files=previous.get('skipped_files', [])))
        shutil.copyfile(args.readme, output / 'README.md')
        state.update(status='waiting_for_quota' if time.time() < state['resume_after_epoch'] else 'uploading')
        _write_json(receipt, state)
        print(f'packing complete: {len(plan)} shards; quota deadline {state["resume_after_epoch"]}', flush=True)
        api = HubApi()
        api.get_cookies(cookies_required=True)
        # 清单先公开；release_status 只有全部分片提交成功后才标记完成。
        files = ['README.md', 'metadata/foxglove_shards.json', 'SHA256SUMS',
                 'metadata/upload_skipped_files.json'] + [entry['path'] for entry in state['packed']]
        for relative in files:
            if relative in state['uploaded']:
                continue
            _upload(api, output / relative, relative, args.repo_id, state, receipt)
            state['uploaded'].append(relative)
            _write_json(receipt, state)
        release = metadata / 'shard_release_status.json'
        _write_json(release, dict(
            status='uploaded_with_legacy_skips' if previous.get('skipped_files') else 'uploaded',
            foxglove_shards=len(plan), legacy_skipped_files=len(previous.get('skipped_files', [])),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        ))
        _upload(api, release, 'metadata/shard_release_status.json', args.repo_id, state, receipt)
        state['status'] = 'uploaded_with_legacy_skips' if previous.get('skipped_files') else 'uploaded'
        _write_json(receipt, state)
        print(state['status'], flush=True)
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        _write_json(receipt, state)
        raise


if __name__ == '__main__':
    main()
