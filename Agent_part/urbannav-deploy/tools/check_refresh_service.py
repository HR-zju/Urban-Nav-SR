"""Execute the five new online refresh cases once; archive before asserting."""
import argparse
import html
import json
import math
from pathlib import Path
import time

from urbannav_deploy.transport import HttpInferenceClient


def main():
    """Check prepared cases against the warmed service without retrying failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18716')
    args = parser.parse_args()
    root = args.root
    client = HttpInferenceClient(args.url)
    before = client.health()
    rows = [json.loads(line) for line in (root / 'requests.jsonl').read_text().splitlines()]
    results, cards = [], []
    for case in rows:
        started = time.monotonic()
        output = client.predict(case['request_id'], case['row'])
        elapsed = time.monotonic() - started
        (root / (case['request_id'] + '.json')).write_text(json.dumps(output, indent=2) + '\n')
        meta = output['metadata']
        age = output['timestamp_sec'] - meta['query_source_timestamp_sec']
        finite = len(output['trajectory_ego_t']) == 32 and all(
            len(point) == 3 and all(math.isfinite(x) for x in point) for point in output['trajectory_ego_t'])
        checks = {
            'identity': output['request_id'] == case['request_id'],
            'reason': meta['query_schedule']['reason'] == case['expected_reason'],
            'reuse': meta['query_profile']['reused_query'] == case['expected_reused'],
            'age': 0 <= age <= .5 and abs(age - meta['query_schedule']['query_age_sec']) < 1e-6,
            'finite_32x3': finite,
        }
        results.append({'request_id': case['request_id'], 'seconds': elapsed,
                        'schedule': meta['query_schedule'], 'checks': checks})
        (root / 'checks.json').write_text(json.dumps(results, indent=2) + '\n')
        if not all(checks.values()):
            raise RuntimeError(f'case failed without retry: {case["request_id"]}: {checks}')
        cards.append('<h2>' + html.escape(case['request_id']) + '</h2><pre>' +
                     html.escape(json.dumps(output, indent=2)) + '</pre>')
    after = client.health()
    score = {'cases': results, 'health_before': before, 'health_after': after,
             'service_requests_added': after['completed_requests'] - before['completed_requests'],
             'same_loaded_instance': all(before[k] == after[k] for k in ('pid', 'model_instance_id')),
             'scope': 'controlled_refresh_triggers_not_navigation_quality',
             'automatic_retries': 0}
    (root / 'scorecard.json').write_text(json.dumps(score, indent=2) + '\n')
    (root / 'visualization/review.html').write_text('<!doctype html><meta charset="utf-8">'
        '<h1>Online refresh cases</h1><p>Controlled metadata/instruction changes; no physical acceptance.</p>'
        + ''.join(cards))
    if score['service_requests_added'] != len(rows) or not score['same_loaded_instance']:
        raise RuntimeError('unexpected request count or changed model instance')
    print(json.dumps(score, indent=2), flush=True)


if __name__ == '__main__':
    main()
