"""Run two fixed initial-noise actions on saved conditions, without an LVLM."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess


def main():
    """Keep conditions and solver fixed; compare saved zero to two normal seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--previous', required=True, type=Path)
    args = parser.parse_args()
    root, previous = args.root, args.previous
    import torch
    from swift.sidewalk_nav.runtime.action_subgraph import TensorRTFlowActionBackend
    config = json.loads((previous / 'config.json').read_text())
    source = previous / 'service/requests/cold_same_process'
    paths = [source / 'condition.pt', source / 'response.json', root / 'reference.json',
             Path(config['engine_path']), Path(config['spec_path'])]
    uuid = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
                                   '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'], text=True)
    (root / 'resources_before.json').write_text(json.dumps({'gpu_uuid': uuid, 'processes': processes}))
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise RuntimeError('GPU occupied; engine not loaded')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if hashes[config['engine_path']] != config['engine_digest']:
        raise ValueError('Frozen engine identity mismatch')
    (root / 'visualization_manifest.json').write_text(json.dumps({
        'scope': 'one_condition_two_fixed_seeds_no_encoder_forward',
        'scorecard': 'scorecard.json', 'review': 'review.html'}))
    (root / 'started.json').write_text(json.dumps({'seeds': [0, 1], 'flow_steps': 8,
        'zero_baseline': 'saved_response_not_recomputed', 'source_sha256': hashes}, indent=2))
    tensors = torch.load(source / 'condition.pt', map_location='cuda', weights_only=True)
    reference = torch.tensor(json.loads((root / 'reference.json').read_text())['trajectory_ego_t'])
    old = json.loads((source / 'response.json').read_text())
    profile = json.loads((Path(config['model_dir']) / 'config.json').read_text())[
        'urban_nav_policy']['contracts']['normalization_profile']
    scale = torch.tensor(profile['scales'])
    results = [{'name': 'saved_zero', 'trajectory': old['trajectory_ego_t']}]
    backend = TensorRTFlowActionBackend.from_files(config['engine_path'], config['spec_path'])
    try:
        with torch.inference_mode():
            for seed in (0, 1):
                initial = torch.randn((1, 32, 3), generator=torch.Generator().manual_seed(seed))
                normalized = backend.sample_actions(**tensors, flow_steps=8, initial_action=initial)[0]
                torch.save(initial, root / f'initial_seed_{seed}.pt')
                metric = normalized.detach().cpu().float() * scale
                results.append({'name': f'normal_seed_{seed}', 'trajectory': metric.tolist()})
        for result in results:
            delta = torch.tensor(result['trajectory']) - reference
            xy = torch.linalg.vector_norm(delta[:, :2], dim=1)
            result.update(mean_xy_error_m=float(xy.mean()), final_xy_error_m=float(xy[-1]),
                          mean_yaw_error_rad=float(torch.atan2(delta[:, 2].sin(), delta[:, 2].cos()).abs().mean()))
        score = {'scope': 'single_condition_initial_noise_diagnosis', 'full_model_forwards': 0,
                 'new_action_rollouts': 2, 'flow_steps': 8, 'automatic_retries': 0,
                 'navigation_quality_accepted': False, 'reference': reference.tolist(), 'results': results,
                 'backend_timing': backend.timing_report()}
        (root / 'scorecard.json').write_text(json.dumps(score, indent=2, allow_nan=False))
        (root / 'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Initial noise diagnosis</h1>'
            '<p>One fixed condition, saved zero baseline, two predetermined normal seeds.</p><pre>'
            + html.escape(json.dumps(score, indent=2)) + '</pre>')
        print(json.dumps({r['name']: {k:v for k,v in r.items() if k != 'trajectory'} for r in results}), flush=True)
    finally:
        backend.close()


if __name__ == '__main__':
    main()
