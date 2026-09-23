"""Offline, parent-addressable review of materialized history observations."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


def _gt_svg(sidecar: dict | None) -> str:
    """Project valid GT ego-FLU points with image-right=-left, image-up=forward."""
    if sidecar is None:
        return '<p>该小包未关联 trajectory sidecar。</p>'
    target = sidecar['trajectory_target']
    points = [(0., 0.)] + [(float(point[0]), float(point[1]))
                          for point, valid in zip(target['trajectory_ego_t'], target['action_mask']) if valid]
    radius = max(1., max(max(abs(x), abs(y)) for x, y in points))
    scale = 110 / radius
    # FLU: x 前、y 左；仅画有效 mask 的轨迹点，避免把补零尾段画成回到原点。
    polyline = ' '.join(f'{140-y*scale:.2f},{140-x*scale:.2f}' for x, y in points)
    return (
        '<svg viewBox="0 0 280 280" role="img" aria-label="GT ego-FLU trajectory">'
        '<path d="M140 20V260 M20 140H260" stroke="#a8adb8" fill="none"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#245dcc" stroke-width="3"/>'
        '<circle cx="140" cy="140" r="4" fill="#202a3a"/>'
        '<text x="146" y="24">x 前</text><text x="20" y="133">y 左</text>'
        f'<text x="12" y="274">轴半径 {radius:.2f} m；有效 GT 点 {len(points)-1}</text></svg>'
    )


def write_history_review(rows: list[dict], output: Path) -> Path:
    """Write a local HTML review inside one clip shard, with no external assets.

    Args:
        rows: Materialized source/observation/sidecar records in stable source order.
        output: Shard root containing the observation's relative media paths.
    Returns:
        Path to review/index.html; each parent has a URL fragment.
    Raises:
        OSError: If review output creation fails.
    """
    review = output / 'review'
    review.mkdir(exist_ok=False)
    sections, links = [], []
    for row in rows:
        parent = row['source_parent_id']
        anchor_id = quote(parent, safe='')
        links.append(f'<li><a href="#{anchor_id}">{escape(parent)}</a></li>')
        observation = row['observation']
        anchor_ns = observation['source_ns'][-1]
        figures = []
        for i, (path, source_ns, target_ns, ordinal) in enumerate(zip(
                observation['fpv_paths'], observation['source_ns'],
                observation['target_ns'], observation['video_ordinals'])):
            url = quote('../' + path, safe='/')
            figures.append(
                f'<figure><img loading="lazy" src="{url}" alt="FPV {i}">'
                f'<figcaption>FPV {i} · ordinal {ordinal}<br>'
                f'实际 {(source_ns-anchor_ns)/1e9:.3f}s · 目标 {(target_ns-anchor_ns)/1e9:.3f}s<br>'
                f'lag {(target_ns-source_ns)/1e6:.3f} ms</figcaption></figure>')
        bev = observation['bev']
        if bev is not None:
            figures.append(
                f'<figure><img loading="lazy" src="{quote("../" + bev["bev_image_path"], safe="/")}" alt="BEV">'
                f'<figcaption>BEV age {bev["age_ns"]/1e6:.3f} ms<br>'
                f'传感器同步：{escape(bev["sensor_sync_causality"])}</figcaption></figure>')
        source = row['source_row']
        label_fields = {key: source[key] for key in (
            'instruction', 'response', 'messages', 'direction_choice', 'direction_name',
        ) if key in source}
        labels = escape(json.dumps(label_fields, ensure_ascii=False, indent=2))
        if row.get('source_variants'):
            candidates = [{
                'source_manifest': variant['source_manifest'], 'source_line': variant['source_line'],
                'labels': {key: variant['source_row'][key] for key in (
                    'instruction', 'response', 'messages', 'direction_choice', 'direction_name',
                ) if key in variant['source_row']},
            } for variant in row['source_variants']]
            labels = escape(json.dumps({'semantic_label_selection': 'pending', 'variants': candidates},
                                       ensure_ascii=False, indent=2))
        sections.append(
            f'<section id="{escape(parent, quote=True)}"><h2>{escape(parent)}</h2>'
            f'<p>split: {escape(row["split"])} · 标签适配：待审查</p>'
            f'<div class="frames">{"".join(figures)}</div>'
            f'<div class="gt">{_gt_svg(row["trajectory_sidecar"])}</div>'
            f'<details><summary>原标签 / 文本（原样保留）</summary><pre>{labels}</pre></details></section>')
    html = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>4s / 2Hz observation review</title><style>'
        'body{font:15px/1.5 sans-serif;background:#f5f6fa;color:#202a3a;margin:24px}'
        'section{background:white;padding:18px;margin:20px 0;border:1px solid #ddd}'
        'h2{font-size:17px;overflow-wrap:anywhere}.frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}'
        'figure{margin:0}img{width:100%;max-height:280px;object-fit:contain;background:#eee}'
        'figcaption{font-size:13px}.gt{max-width:320px}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        '</style><h1>4 秒 / 2 Hz · 9 FPV</h1>'
        '<p>这是原标签与新观测的审核页。图像可读、时间满足合同，不代表旧标签已通过语义兼容验收。'
        'GT 为独立 ego-FLU 图，未投影到 FPV。</p>'
        f'<details><summary>按 parent 定位（{len(rows)} 条）</summary><ul>{"".join(links)}</ul></details>'
        + ''.join(sections) + '</html>'
    )
    path = review / 'index.html'
    path.write_text(html, encoding='utf-8')
    return path
