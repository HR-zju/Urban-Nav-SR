"""Pair existing outputs by current input; report differences, never GT errors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def identity(row):
    """Hash the exact action observation excluding its metadata-only sample ID."""
    observation = dict(row['cached_traj_query_training']['action_observation'])
    observation.pop('sample_id')
    return hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()


def main():
    """Read saved outputs, export paired differences and FLU review figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root, out = args.artifacts, args.output
    out.mkdir(exist_ok=False)
    fresh = root / 'continuous_cache/replay'
    dual = root / 'dual_rate_shadow'
    original = [json.loads(line) for line in (fresh / 'requests.jsonl').read_text().splitlines()]
    lookup = {identity(request['row']): request for request in original}
    current = {request['request_id']: request['row'] for request in
               map(json.loads, (dual / 'requests.jsonl').read_text().splitlines())}
    predictions = [event['prediction'] for event in
                   map(json.loads, (dual / 'events.jsonl').read_text().splitlines())
                   if event['event'] == 'prediction']
    pairs, plot_data = [], {}
    for retained in predictions:
        row = current[retained['request_id']]
        previous = lookup.get(identity(row))
        if previous is None:
            continue
        baseline = json.loads((fresh / 'responses' / (previous['request_id'] + '.json')).read_text())
        # 仅比较同一当前视觉/指令输入。模型、归一化、时间网格、掩码、FLU 坐标也必须相同。
        for field in ('checkpoint_digest', 'normalization_digest', 'coordinate_frame',
                      'trajectory_format', 'trajectory_timestamps_sec', 'valid_mask'):
            if baseline[field] != retained[field]:
                raise ValueError(f'paired output contract mismatch: {field}')
        if previous['row']['messages'] != row['messages']:
            raise ValueError('paired current observation prompts differ')
        a, b = baseline['trajectory_ego_t'], retained['trajectory_ego_t']
        if len(a) != 32 or len(b) != 32:
            raise ValueError('expected [32,3] outputs')
        xy = [math.hypot(x[0] - y[0], x[1] - y[1]) for x, y in zip(a, b)]
        yaw = [abs(math.atan2(math.sin(x[2] - y[2]), math.cos(x[2] - y[2]))) for x, y in zip(a, b)]
        age = retained['metadata']['query_schedule']['query_age_sec']
        pairs.append({'fresh_id': previous['request_id'], 'retained_id': retained['request_id'],
                      'age_sec': age, 'mean_xy_difference_m': statistics.mean(xy),
                      'final_xy_difference_m': xy[-1], 'max_xy_difference_m': max(xy),
                      'mean_yaw_difference_rad': statistics.mean(yaw),
                      'complete_input_row_equal': previous['row'] == row})
        plot_data[retained['request_id']] = (a, b, row)
    summaries = {}
    for name, items in [('fresh_again', [p for p in pairs if p['age_sec'] == 0]),
                        ('retained_query', [p for p in pairs if p['age_sec'] > 0])]:
        summaries[name] = {'count': len(items), **{
            key: statistics.mean(item[key] for item in items) if items else None
            for key in ('mean_xy_difference_m', 'final_xy_difference_m', 'mean_yaw_difference_rad')},
            'max_xy_difference_m': max((item['max_xy_difference_m'] for item in items), default=None)}
    report = {'scope': 'output_differences_not_ground_truth_errors', 'model_calls': 0,
              'fresh_outputs': len(original), 'dual_outputs': len(predictions), 'matched': len(pairs),
              'matching': 'same action observation except metadata sample_id; current messages equal',
              'summary': summaries, 'pairs': pairs,
              'evidence_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (fresh / 'requests.jsonl', dual / 'requests.jsonl',
                                               dual / 'events.jsonl')},
              'navigation_quality_accepted': False}
    (out / 'paired_differences.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, color, predicate in [('Fresh again', 'tab:blue', lambda p: p['age_sec'] == 0),
                                    ('Retained query', 'tab:orange', lambda p: p['age_sec'] > 0)]:
        items = [p for p in pairs if predicate(p)]
        for ax, metric in zip(axes, ('mean_xy_difference_m', 'final_xy_difference_m')):
            ax.scatter([p['age_sec'] for p in items], [p[metric] for p in items],
                       s=22, alpha=.7, label=name, color=color)
            ax.set_xlabel('Query source age [s]')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Same current input: output difference, not ground-truth error')
    fig.savefig(out / 'differences.png', dpi=160)
    plt.close(fig)
    chosen = sorted(pairs, key=lambda p: p['mean_xy_difference_m'], reverse=True)[:3]
    chosen += [max((p for p in pairs if p['age_sec'] == 0), key=lambda p: p['max_xy_difference_m'])]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 3.4 * len(chosen)), constrained_layout=True)
    for axes_row, pair in zip(axes, chosen):
        a, b, row = plot_data[pair['retained_id']]
        for trajectory, label in ((a, 'Fresh baseline'), (b, 'Dual-rate output')):
            axes_row[0].plot([p[0] for p in trajectory], [p[1] for p in trajectory], label=label)
        axes_row[0].set(xlabel='x forward [m]', ylabel='y left [m]', aspect='equal',
                        title=f"{pair['retained_id']}, query age {pair['age_sec']:.3f}s")
        span = max(point[0] for point in a + b) - min(point[0] for point in a + b)
        axes_row[0].set_ylim(-max(1., span * .4), max(1., span * .4))
        axes_row[0].legend(loc='upper left', fontsize=8)
        axes_row[0].grid(alpha=.2)
        for ax, index, folder in ((axes_row[1], 4, 'fpv'), (axes_row[2], 5, 'bev')):
            path = root / 'input' / folder / Path(row['images'][index]).name
            ax.imshow(plt.imread(path))
            ax.set_title('Current FPV' if index == 4 else 'Recorded BEV (not Amap GUI)')
            ax.axis('off')
    fig.savefig(out / 'matched_cases.png', dpi=140)
    plt.close(fig)
    (out / 'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Saved output audit</h1>'
        '<p>All matched observations; no new inference. Blue is another model output, not ground truth.</p>'
        '<img src="differences.png" style="max-width:100%"><img src="matched_cases.png" style="max-width:100%">')
    print(json.dumps({'matched': len(pairs), 'summary': summaries}, indent=2))


if __name__ == '__main__':
    main()
