"""Audited source-media materialization for causal full-clip service acceptance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from .replay import plan_clip


def digest(path):
    """Hash a source artifact without altering it.

    Args:
        path: Existing source artifact.
    Returns:
        Lowercase SHA256 hex digest.
    """
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, mapping, output):
    """Decode each selected frame once and preserve actual source timestamps.

    Args:
        source: Audited clip directory containing original video and source indices.
        mapping: Previously established ordinal-map JSON for this exact clip.
        output: New persistent output directory; never overwritten.
    Raises:
        FileExistsError: Output already exists, including an interrupted attempt.
        ValueError: Mapping identity or source clocks are invalid.
        OSError: Required source image cannot be decoded or written.
    """
    import cv2
    import numpy as np
    source, mapping, output = Path(source), Path(mapping), Path(output)
    if output.exists():
        raise FileExistsError(output)
    stamps = np.load(source / 'frame_times.npy').tolist()
    audit = json.loads(mapping.read_text())
    ordinals = audit['video_ordinals']
    if len(stamps) != len(ordinals) or audit['clip_id'] != source.name:
        raise ValueError('mapping does not cover the source clip')
    fpv = [{'timestamp_ns': int(stamp), 'path': f'fpv/{ordinal:06d}.jpg', 'video_ordinal': ordinal}
           for stamp, ordinal in zip(stamps, ordinals)]
    index = json.loads((source / 'bev_index.json').read_text())
    bev = [{'timestamp_ns': int(row['timestamp']), 'path': f"bev/{Path(row['bev_image_path']).name}"}
           for row in index['entries']]
    events = plan_clip(fpv, bev, start_ns=stamps[0], end_ns=stamps[-1])
    selected = {e['frame']['path'] for e in events if e['kind'] != 'predict' and e['status'] == 'delivered'}
    output.mkdir(parents=True)
    (output / 'fpv').mkdir()
    (output / 'bev').mkdir()
    wanted = {x['video_ordinal']: x['path'] for x in fpv if x['path'] in selected}
    cap = cv2.VideoCapture(str(source / 'front.mp4'))
    try:
        if not cap.isOpened():
            raise OSError('unable to open source video')
        for ordinal in range(max(wanted) + 1):
            ok, image = cap.read()
            if not ok:
                raise OSError(f'video ended before required ordinal {ordinal}')
            if ordinal in wanted and not cv2.imwrite(str(output / wanted[ordinal]), image,
                                                     [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f'image write failed: {ordinal}')
    finally:
        cap.release()
    for row in index['entries']:
        relative = f"bev/{Path(row['bev_image_path']).name}"
        if relative in selected:
            original = source / 'bev_images' / Path(row['bev_image_path']).name
            shutil.copyfile(original, output / relative)
    payload = {
        'schema_version': 'urbannav_causal_clip_replay_v1', 'clip_id': source.name,
        'source_start_ns': stamps[0], 'source_end_ns': stamps[-1],
        'source_duration_sec': (stamps[-1] - stamps[0]) / 1e9,
        'source_fpv_count': len(fpv), 'source_bev_count': len(bev),
        'fpv_delivery_hz': 5, 'bev_delivery_hz': 2, 'prediction_hz': 2,
        'instruction': 'Navigate safely on the sidewalk and follow the visible route.',
        'instruction_provenance': 'original clip labeling manifest; unchanged generic navigation instruction',
        'max_fpv_lag_sec': .1, 'max_bev_age_sec': 1.1,
        'bev_provenance': 'recorded_clip_global_full_route_overlay_not_live_GUI',
        'events': events,
        'source_identity': {str(p): digest(p) for p in [mapping, source / 'frame_times.npy',
                                                       source / 'bev_index.json', source / 'front.mp4']},
        'media_sha256': {p: digest(output / p) for p in sorted(selected)},
    }
    (output / 'manifest.json').write_text(json.dumps(payload, indent=2) + '\n')
    summary = {k: v for k, v in payload.items() if k not in ('events', 'media_sha256', 'source_identity')}
    summary['materialized_images'] = len(selected)
    summary['prediction_slots'] = sum(e['kind'] == 'predict' for e in events)
    summary['eligible_predictions'] = sum(e['status'] == 'eligible' for e in events)
    summary['rejected_slots'] = [{k: e[k] for k in ('kind', 'offset_sec', 'reason')}
                                 for e in events if e['status'] == 'rejected']
    (output / 'preparation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
