"""Compare saved predictions to declared clip odometry without model calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    """Export all matched full-horizon errors and selected diagnostic plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--clip', required=True, type=Path)
    parser.add_argument('--reader', required=True, type=Path)
    args = parser.parse_args()
    root = args.artifacts
    out = root / 'odometry_review_20261016'
    out.mkdir(exist_ok=False)
    (out / 'visualization_manifest.json').write_text(json.dumps({
        'scope': 'saved_output_odometry_reference_not_navigation_acceptance',
        'scorecard': 'scorecard.json', 'review': 'review.html', 'plot': 'review.png'}))
    spec = importlib.util.spec_from_file_location('diagnostic_raw_reader', args.reader)
    reader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reader
    spec.loader.exec_module(reader)
    provenance = json.loads((root / 'pose_provenance_20261016/audit.json').read_text())
    if not provenance['timestamp_all_equal'] or not provenance['replay_frame_times_hash_equal']:
        raise ValueError('Source identity audit must pass first')
    for path, digest in provenance['source_sha256'].items():
        if path == '/tmp/r10-clip-calibrated-rows.jsonl':
            path = root / 'pose_provenance_20261016/selected_rows.jsonl'
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source changed after provenance audit: {path}')
    # 来源声明在 clip 级审计绑定；只读原始位姿，不能伪造某个输入行的训练标签声明。
    series = reader._load_npy_series(args.clip, dataset='foxglove')
    if series.globally_static:
        raise ValueError('Globally static pose cannot establish motion reference')
    paired_path = root / 'predeployment_review_20261016/paired_differences.json'
    pairs = json.loads(paired_path.read_text())['pairs']
    requests_path = root / 'dual_rate_shadow/requests.jsonl'
    events_path = root / 'dual_rate_shadow/events.jsonl'
    requests = {r['request_id']: r['row'] for r in map(json.loads, requests_path.read_text().splitlines())}
    predictions = {r['prediction']['request_id']: r['prediction'] for r in
                   map(json.loads, events_path.read_text().splitlines()) if r['event'] == 'prediction'}
    records, pictures = [], []
    for pair in pairs:
        item = dict(pair)
        row = requests[pair['retained_id']]
        obs = row['cached_traj_query_training']['action_observation']
        index = obs['image_roles'].index('fpv.current')
        frame = int(Path(obs['images'][index]).stem)
        if abs(float(series.timestamps[frame]) / 1e9 - obs['observation_timestamp_sec']) > 1e-6:
            raise ValueError('Current FPV and reference clock mismatch')
        try:
            timed = np.asarray(reader._relative_window(series, current_index=frame,
                               horizon_seconds=4., yaw_offset_deg=90., full_horizon_required=True))
        except reader.StageIIRawTrajectoryErrorV2 as exc:
            item.update(status='rejected', reason=exc.code)
            records.append(item)
            continue
        fresh_path = root / 'continuous_cache/replay/responses' / (pair['fresh_id'] + '.json')
        fresh = json.loads(fresh_path.read_text())
        dual = predictions[pair['retained_id']]
        if fresh['trajectory_timestamps_sec'] != dual['trajectory_timestamps_sec']:
            raise ValueError('Prediction time grids differ')
        times = np.asarray(dual['trajectory_timestamps_sec'])
        if times.min() < timed[0, 3] or times.max() > timed[-1, 3]:
            raise ValueError('Reference does not cover prediction time grid')
        reference = np.stack([np.interp(times, timed[:, 3], timed[:, 0]),
                              np.interp(times, timed[:, 3], timed[:, 1]),
                              np.interp(times, timed[:, 3], np.unwrap(timed[:, 2]))], axis=1)
        for name, prediction in [('fresh', fresh), ('dual', dual)]:
            if not all(prediction['valid_mask']) or prediction['coordinate_frame'] != 'current_ego_flu':
                raise ValueError('Expected full FLU prediction mask and coordinate frame')
            points = np.asarray(prediction['trajectory_ego_t'])
            delta = points - reference
            xy = np.linalg.norm(delta[:, :2], axis=1)
            item[name] = {'mean_xy_error_m': float(xy.mean()), 'final_xy_error_m': float(xy[-1]),
                          'mean_yaw_error_rad': float(np.abs(np.arctan2(np.sin(delta[:, 2]),
                                                                       np.cos(delta[:, 2]))).mean())}
        item.update(status='accepted', frame_idx=frame)
        records.append(item)
        pictures.append((item, reference, fresh, dual, obs['images'][index]))
    summary = {}
    for name, keep in [('fresh_again', lambda r: r['age_sec'] == 0),
                       ('retained_query', lambda r: r['age_sec'] > 0)]:
        group = [r for r in records if r['status'] == 'accepted' and keep(r)]
        summary[name] = {'count': len(group), **{arm: {
            metric: float(np.mean([r[arm][metric] for r in group])) if group else None
            for metric in ('mean_xy_error_m', 'final_xy_error_m', 'mean_yaw_error_rad')}
            for arm in ('fresh', 'dual')}}
    score = {'scope': 'single_clip_odometry_reference_with_historical_cache_confound',
             'model_calls': 0, 'navigation_quality_accepted': False, 'pairs': records,
             'summary': summary, 'planned_pairs': len(pairs),
             'accepted_pairs': len(pictures), 'rejected_pairs': len(pairs) - len(pictures),
             'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (args.reader, paired_path, requests_path, events_path)}}
    (out / 'scorecard.json').write_text(json.dumps(score, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = sorted(pictures, key=lambda r: r[0]['mean_xy_difference_m'], reverse=True)[:3]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 3.6 * len(chosen)), constrained_layout=True)
    for ax, (item, reference, fresh, dual, image_path) in zip(axes, chosen):
        for points, label in [(reference, 'Raw odometry reference'),
                              (np.asarray(fresh['trajectory_ego_t']), 'Fresh prediction'),
                              (np.asarray(dual['trajectory_ego_t']), 'Dual prediction')]:
            ax[0].plot(points[:, 0], points[:, 1], label=label)
        ax[0].set(xlabel='x forward [m]', ylabel='y left [m]',
                  title=f"{item['retained_id']}, age={item['age_sec']:.3f}s")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=.3)
        ax[1].imshow(plt.imread(root / 'input/fpv' / Path(image_path).name))
        ax[1].axis('off')
    fig.savefig(out / 'review.png', dpi=140)
    plt.close(fig)
    (out / 'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Odometry reference review</h1>'
        '<p>Historical cache confound remains. Not physical navigation acceptance.</p>'
        '<img src="review.png" style="max-width:100%">')
    print(json.dumps({k: score[k] for k in ('summary', 'accepted_pairs', 'rejected_pairs')}, indent=2))


if __name__ == '__main__':
    main()
