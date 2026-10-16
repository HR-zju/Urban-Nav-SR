"""Replay exact visual birth batches for one historical input discrepancy."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess


def write(path, value):
    """Persist diagnosis state without removing earlier branches."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    """Replay vision only first; run language/action only if features change."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--per-image', action='store_true',
                        help='Verify fixed per-image encoding, including both full history branches')
    args = parser.parse_args()
    if args.per_image:
        from urbannav_deploy.per_image_vision import PerImageVision
    root, old = args.root, args.previous
    import torch
    from swift.sidewalk_nav.runtime.inference_service import R10InferenceService
    from swift.sidewalk_nav.cli.infer import row_to_infer_request
    from swift.sidewalk_nav.contracts.tokens import DEFAULT_TRAJ_TOKEN

    config = json.loads((old / 'config.json').read_text())
    config.update(run_root=str(root / 'service'), run_id=root.name)
    write(root / 'config.json', config)
    row = json.loads((old / 'input.json').read_text())
    birth = json.loads((root / 'birth_batches.json').read_text())
    uuid = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
                                    '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'], text=True)
    write(root / 'resources_before.json', {'gpu_uuid': uuid, 'processes': processes})
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise RuntimeError('GPU occupied, no model loaded')
    visual = root / 'visualization'
    visual.mkdir()
    write(visual / 'visualization_manifest.json', {'scorecard': '../scorecard.json', 'review': 'review.html',
                                                 'scope': 'historical_visual_birth_batches'})
    write(root / 'started.json', {'pid': os.getpid(), 'max_full_forwards': 3,
                                  'input_sha256': hashlib.sha256((old / 'input.json').read_bytes()).hexdigest()})
    service = None
    try:
        service = R10InferenceService(config)
        cache = service.vision_cache
        if args.per_image:
            cache.visual = PerImageVision(cache.visual).eval()
        cold = service.predict({'request_id': 'cold_same_process', 'row': row})
        original = {key: value.detach().cpu().clone() for key, value in cache.entries.items()}
        cold_condition = torch.load(root / 'service/requests/cold_same_process/condition.pt',
                                    map_location='cpu', weights_only=True)
        prior_condition = torch.load(old / 'service/requests/all_six_cold/condition.pt',
                                     map_location='cpu', weights_only=True)
        score = {'cross_process_cold_condition_diffs': {
            key: {'equal': torch.equal(value, prior_condition[key]),
                  'max_abs': float((value.float() - prior_condition[key].float()).abs().max())}
            for key, value in cold_condition.items()}, 'branches': {}}
        # 只做一次额外模板预处理取得旧地图的原生像素，不调用语言模型。
        previous_rows = [json.loads(line) for line in
                         (args.baseline / 'replay/requests.jsonl').read_text().splitlines()]
        predecessor = next(r['row'] for r in previous_rows
                           if r['request_id'] == 'r10-cache-continuous-20261015-0017')
        service.bundle.template.encode(row_to_infer_request(
            predecessor['cached_traj_query_training']['query_row'],
            add_traj_token=True, traj_token=DEFAULT_TRAJ_TOKEN, traj_num_query_tokens=16))
        pixels = {key[0]: entry['processed'] for key, entry in service.image_cache.entries.items()
                  if entry['processed'] is not None}
        fingerprints = {image: hashlib.sha256(Path(image).read_bytes()).hexdigest()
                        for info in birth.values() for item in info.values() for image in item['batch']}
        write(root / 'media_sha256.json', fingerprints)
        for name, images in birth.items():
            cache.entries.clear()
            cache.used_bytes = cache.hits = cache.misses = 0
            batches = list(dict.fromkeys(tuple(item['batch']) for item in images.values()))
            for batch in batches:
                cache.set_images(batch, json.dumps(row.get('chat_template_kwargs', {}), sort_keys=True))
                with torch.inference_mode():
                    cache(torch.cat([pixels[p]['pixel_values'] for p in batch]).to(service.device),
                          grid_thw=torch.cat([pixels[p]['image_grid_thw'] for p in batch]).to(service.device))
            differences = [{'image': str(key[0]), 'equal': torch.equal(value, cache.entries[key].cpu()),
                            'max_abs': float((value.float() - cache.entries[key].cpu().float()).abs().max())}
                           for key, value in original.items()]
            result = {'visual_differences': differences, 'birth_batches': batches,
                      'full_forward_needed': args.per_image or any(not item['equal'] for item in differences)}
            if result['full_forward_needed']:
                service.query_cache.clear()
                service.scheduled_query = None
                prediction = service.predict({'request_id': name, 'row': row})
                condition = torch.load(root / 'service/requests' / name / 'condition.pt',
                                       map_location='cpu', weights_only=True)
                result['condition_diffs'] = {key: {'equal': torch.equal(value, cold_condition[key]),
                    'max_abs': float((value.float() - cold_condition[key].float()).abs().max())}
                    for key, value in condition.items()}
                delta = torch.tensor(prediction['trajectory_ego_t']) - torch.tensor(cold['trajectory_ego_t'])
                result['trajectory_max_xy_difference_m'] = float(torch.linalg.vector_norm(delta[:, :2], dim=1).max())
            score['branches'][name] = result
            write(root / 'scorecard.json', score)
        score.update(full_forward_count=service.completed, automatic_retries=0,
                     navigation_quality_accepted=False, per_image=args.per_image)
        if args.per_image:
            score['history_independence_passed'] = all(
                all(item['equal'] for item in branch['visual_differences'])
                and all(item['equal'] for item in branch['condition_diffs'].values())
                and branch['trajectory_max_xy_difference_m'] == 0
                for branch in score['branches'].values())
        write(root / 'scorecard.json', score)
        (visual / 'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Historical cache batches</h1>'
                                          '<pre>' + html.escape(json.dumps(score, indent=2)) + '</pre>')
        print(json.dumps(score, indent=2), flush=True)
    finally:
        if service is not None:
            service.close()


if __name__ == '__main__':
    main()
