"""Localize one fresh-input discrepancy by changing only visual cache history."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess


def write(path, value):
    """Write a diagnostic receipt, retaining failed and negative findings."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    """Run four full forwards and one TRT-only replay on one fixed observation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    root, baseline = args.root, args.baseline
    import torch
    from swift.sidewalk_nav.runtime.inference_service import R10InferenceService

    records = [json.loads(line) for line in (baseline / 'requests.jsonl').read_text().splitlines()]
    row = next(r['row'] for r in records if r['request_id'] == 'dual-rate-0046')
    write(root / 'input.json', row)
    config = json.loads((baseline / 'model_config.json').read_text())
    config.update(run_root=str(root / 'service'), run_id=root.name,
                  save_intermediate_tensors=True, query_refresh_period_sec=None, query_cache_max_age_sec=0.)
    write(root / 'config.json', config)
    visual = root / 'visualization'
    visual.mkdir()
    write(visual / 'visualization_manifest.json', {
        'scope': 'one_fixed_input_cache_batch_diagnosis', 'scorecard': '../scorecard.json',
        'review': 'review.html', 'intermediates': '../service/requests/*/condition.pt',
        'access': 'private filesystem via authenticated SSH'})
    uuid = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
                                    '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'], text=True)
    write(root / 'resources_before.json', {'gpu_uuid': uuid, 'processes': processes})
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise RuntimeError('diagnostic GPU occupied; no model loaded')
    write(root / 'started.json', {'pid': os.getpid(), 'input_sha256':
                                 hashlib.sha256((root / 'input.json').read_bytes()).hexdigest(),
                                 'planned_full_forwards': 4, 'planned_trt_replays': 1})
    service = None
    try:
        service = R10InferenceService(config)
        cache = service.vision_cache
        capture = {}

        def capture_pixels(module, inputs, kwargs):
            capture['pixels'] = inputs[0].detach().clone()
            capture['grid'] = kwargs['grid_thw'].detach().clone()
            capture['identities'] = module.images

        hook = cache.register_forward_pre_hook(capture_pixels, with_kwargs=True)
        predictions = {}

        def forward(name):
            # 每个分支均重算语言 query；只操纵图像特征缓存，不混入 query 年龄或语言缓存变量。
            service.query_cache.clear()
            service.scheduled_query = None
            prediction = service.predict({'request_id': name, 'row': row})
            predictions[name] = prediction
            return prediction

        forward('all_six_cold')
        hook.remove()
        torch.save({'pixels': capture['pixels'].cpu(), 'grid': capture['grid'].cpu()},
                   root / 'visual_input.pt')
        original_entries = {key: value.detach().cpu().clone() for key, value in cache.entries.items()}
        forward('all_six_warm')

        cache.entries.clear()
        cache.used_bytes = cache.hits = cache.misses = 0
        grids = capture['grid']
        sizes = [int(t * h * w) for t, h, w in grids.tolist()]
        patches = capture['pixels'].split(sizes)
        # 图像独立的逻辑保持相同；唯一变化是前三张先以三图批次进入视觉塔。
        cache.images = capture['identities'][:3]
        with torch.inference_mode():
            cache(torch.cat(patches[:3]), grid_thw=grids[:3])
        cache.images = capture['identities']
        forward('three_plus_three')
        visual_diffs = []
        for key, expected in original_entries.items():
            actual = cache.entries[key].detach().cpu()
            visual_diffs.append({'image': str(key[0]), 'equal': torch.equal(expected, actual),
                                 'max_abs': float((expected.float() - actual.float()).abs().max())})
        forward('three_plus_three_warm')

        conditions = {name: torch.load(root / 'service/requests' / name / 'condition.pt',
                                      map_location='cpu', weights_only=True) for name in predictions}
        reference = conditions['all_six_cold']
        diffs = {}
        for name, tensors in conditions.items():
            diffs[name] = {key: {'equal': torch.equal(value, reference[key]),
                                'max_abs': float((value.float() - reference[key].float()).abs().max())}
                           for key, value in tensors.items()}
        with torch.inference_mode():
            repeated = service.backend.sample_actions(
                **{key: value.to(service.device).contiguous() for key, value in reference.items()}, flow_steps=8)[0]
        normalized = torch.load(root / 'service/requests/all_six_cold/normalized_trajectory.pt',
                                map_location='cpu', weights_only=True)
        repeated = repeated.detach().cpu().float()
        torch.save(repeated, root / 'trt_replay.pt')
        trajectories = {name: torch.tensor(p['trajectory_ego_t']) for name, p in predictions.items()}
        xy = {name: float(torch.linalg.vector_norm((value - trajectories['all_six_cold'])[:, :2], dim=1).max())
              for name, value in trajectories.items()}
        score = {'scope': 'single_input_cache_batch_diagnosis_not_quality_acceptance',
                 'condition_diffs_vs_all_six_cold': diffs, 'visual_feature_diffs': visual_diffs,
                 'trajectory_max_xy_difference_m': xy,
                 'trt_repeat_equal': torch.equal(repeated, normalized),
                 'trt_repeat_max_abs': float((repeated - normalized).abs().max()),
                 'split_repeat_equal': torch.equal(trajectories['three_plus_three'],
                                                  trajectories['three_plus_three_warm']),
                 'full_forward_count': service.completed, 'trt_only_replays': 1,
                 'torch_version': torch.__version__, 'torch_cuda': torch.version.cuda,
                 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                 'model_training': service.bundle.model.training,
                 'condition_capture': True, 'automatic_retries': 0}
        write(root / 'scorecard.json', score)
        (visual / 'review.html').write_text('<!doctype html><meta charset="utf-8">'
            '<h1>Fixed input, changed visual batch history</h1><pre>' +
            html.escape(json.dumps(score, indent=2)) + '</pre>')
        print(json.dumps(score, indent=2), flush=True)
    finally:
        if service is not None:
            service.close()


if __name__ == '__main__':
    main()
