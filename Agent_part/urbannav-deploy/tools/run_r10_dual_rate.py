"""One new recorded-input dual-rate run using frozen model code and engine.

Run inside the existing maintained server environment. The caller creates a new
artifact directory, archives this client and supplies the baseline environment.
No inference retries; inspect durable requests after any interrupted transport.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def write(path, value):
    """Persist an experiment receipt as readable JSON."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    """Prepare rows, launch one service, run one replay and preserve all evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    root, old = args.root, args.baseline
    base = old.parent.parent
    media = base / 'runs/r10_full_clip_20261015/media'
    from urbannav_deploy.replay import plan_clip
    from urbannav_deploy.telemetry import summarize
    from swift.sidewalk_nav.runtime.service_request_adapter import build_service_row

    manifest_path = media / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    previous = json.loads((old / 'replay/requests.jsonl').read_text().splitlines()[0])
    template = previous['row']['cached_traj_query_training']['query_source']
    streams = {}
    for kind in ('fpv', 'bev'):
        frames = {event['frame']['timestamp_ns']: event['frame'] for event in manifest['events']
                  if event['kind'] == kind and event['status'] == 'delivered'}
        streams[kind] = [frames[t] for t in sorted(frames)]
    planned = plan_clip(streams['fpv'], streams['bev'], start_ns=manifest['source_start_ns'],
                        end_ns=manifest['source_end_ns'], prediction_period_ns=200_000_000)
    requests = []
    for event in planned:
        if event['status'] != 'eligible':
            continue
        row = build_service_row(event, media_root=media, clip_id=manifest['clip_id'],
                                instruction=manifest['instruction'], observation_template=template)
        requests.append({'request_id': f'dual-rate-{len(requests):04d}',
                         'offset_sec': event['offset_sec'], 'row': row})
    with (root / 'requests.jsonl').open('x') as stream:
        for request in requests:
            stream.write(json.dumps(request) + '\n')
    write(root / 'input_identity.json', {
        'manifest': str(manifest_path),
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'planned_requests': len(requests), 'source_duration_sec': manifest['source_duration_sec'],
        'fpv_hz': 5, 'recorded_bev_hz': 2, 'action_request_hz': 5,
        'scope': 'recorded_clip_latest_only_dual_rate_not_physical'})
    visual = root / 'visualization'
    visual.mkdir()
    write(visual / 'visualization_manifest.json', {
        'events': '../events.jsonl', 'scorecard': '../scorecard.json', 'review': 'review.html',
        'refresh': 'events flushed throughout run; metrics and review after terminal worker',
        'access': 'private filesystem via authenticated SSH', 'training': False})

    uuid = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'], text=True)
    write(root / 'resources_before.json', {'gpu_uuid': uuid, 'processes': processes})
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise RuntimeError('chosen GPU occupied; no service started')
    config = json.loads((old / 'config.json').read_text())
    config.update(run_root=str(root / 'service'), port=18716, run_id=root.name,
                  query_cache_max_age_sec=.5, query_refresh_period_sec=.5)
    write(root / 'model_config.json', config)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    command = [sys.executable, '-u', '-m', 'swift.sidewalk_nav.cli.serve_policy',
               '--config', str(root / 'model_config.json')]
    with (root / 'service.log').open('x') as log:
        service = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    write(root / 'service_process.json', {'pid': service.pid, 'command': command})
    url = 'http://127.0.0.1:18716'
    for _ in range(90):
        if service.poll() is not None:
            raise RuntimeError('service exited: ' + (root / 'service.log').read_text()[-3000:])
        try:
            with urlopen(url + '/healthz', timeout=2) as response:
                health = json.load(response)
            if health['ready']:
                break
        except URLError as exc:
            # 启动时端口尚未监听；此处只轮询健康状态，从不重发推理请求。
            write(root / 'readiness_observation.json', {'error': repr(exc), 'attempt': _})
        time.sleep(2)
    else:
        raise TimeoutError('readiness observation expired; inspect original service PID')
    write(root / 'health_before.json', health)
    if health['pid'] != service.pid or health['completed_requests'] != 0:
        raise RuntimeError('endpoint identity differs from the newly launched service')
    client = {
        'mode': 'shadow',
        'source': {'factory': 'urbannav_deploy.replay_source:create_source', 'options': {
            'requests_jsonl': str(root / 'requests.jsonl'),
            'source_start_sec': manifest['source_start_ns'] / 1e9,
            'end_offset_sec': manifest['source_duration_sec']}},
        'inference': {'factory': 'urbannav_deploy.configuration:http_client_factory',
                      'options': {'url': url, 'timeout': 120}},
        'control_hz': 20, 'max_plan_age_sec': .75, 'close_timeout_sec': 5}
    write(root / 'client_config.json', client)
    with (root / 'events.jsonl').open('x') as out, (root / 'client.stderr').open('x') as err:
        result = subprocess.run([sys.executable, '-m', 'urbannav_deploy', '--config',
                                 str(root / 'client_config.json')], stdout=out, stderr=err, env=env)
    write(root / 'client_exit.json', {'returncode': result.returncode})
    if result.returncode:
        raise RuntimeError('client failed; inspect events and stderr, no retry')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    score = summarize(events)
    predictions = [e['prediction'] for e in events if e['event'] == 'prediction']
    valid = sum(len(p['trajectory_ego_t']) == 32 and
                all(len(row) == 3 and all(math.isfinite(v) for v in row)
                    for row in p['trajectory_ego_t']) for p in predictions)
    score.update(planned_requests=len(requests), finite_32x3_trajectories=valid,
                 skipped_source_slots=sum(e['skipped_slots'] for e in events
                                          if e['event'] == 'replay_delivery'),
                 query_refresh_reasons=dict(Counter(p['metadata']['query_schedule']['reason']
                                                    for p in predictions)),
                 telemetry_writer=json.loads((root / 'client.stderr').read_text().splitlines()[-1]),
                 automatic_retries=0, semantic_acceptance=False, physical_verified=False)
    with urlopen(url + '/healthz', timeout=5) as response:
        score['health_after'] = json.load(response)
    score['request_count_matches'] = score['health_after']['completed_requests'] == len(predictions)
    write(root / 'scorecard.json', score)
    cards = []
    for index in sorted({0, len(predictions) // 2, len(predictions) - 1}):
        prediction = predictions[index]
        points = ' '.join(f'{150 - y * 15:.2f},{280 - x * 15:.2f}'
                          for x, y, _ in prediction['trajectory_ego_t'])
        cards.append(f'<h2>{html.escape(prediction["request_id"])}</h2>'
                     f'<svg width="300" height="300" viewBox="0 0 300 300">'
                     f'<path d="M150 0V300M0 280H300" stroke="gray"/>'
                     f'<polyline points="{points}" fill="none" stroke="blue"/></svg>'
                     f'<pre>{html.escape(json.dumps(prediction, indent=2))}</pre>')
    (visual / 'review.html').write_text('<!doctype html><meta charset="utf-8">'
        '<h1>Recorded dual-rate trajectory review</h1><p>15 px/m; x forward upward, y left (FLU). '
        'Display convention only; physical frame alignment unverified.</p>' + ''.join(cards))
    print(json.dumps(score, indent=2), flush=True)


if __name__ == '__main__':
    main()
