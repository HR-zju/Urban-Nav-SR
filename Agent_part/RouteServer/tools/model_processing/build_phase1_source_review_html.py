#!/usr/bin/env python3
"""Build a static HTML review page for SideWalk Phase1 source JSONL."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_intent(row: dict[str, Any]) -> str:
    route_context = row.get("route_context")
    if isinstance(route_context, dict) and route_context.get("intent_gt"):
        return str(route_context["intent_gt"])
    sidewalk_meta = row.get("sidewalk_meta")
    if isinstance(sidewalk_meta, dict) and sidewalk_meta.get("intent_gt"):
        return str(sidewalk_meta["intent_gt"])
    return str(row.get("intent_gt") or "unknown")


def file_uri(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        try:
            return path.resolve().as_uri()
        except Exception:
            return str(path)
    return str(value)


def pick_balanced(rows: list[dict[str, Any]], *, max_samples: int, per_intent: int) -> list[dict[str, Any]]:
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intent[row_intent(row)].append(row)

    selected: list[dict[str, Any]] = []
    for intent in sorted(by_intent):
        items = by_intent[intent]
        if not items:
            continue
        limit = min(per_intent, len(items))
        if limit <= 0:
            continue
        if limit == len(items):
            selected.extend(items)
            continue
        step = max(len(items) / float(limit), 1.0)
        used: set[int] = set()
        for i in range(limit):
            idx = min(len(items) - 1, int(round(i * step)))
            while idx in used and idx + 1 < len(items):
                idx += 1
            used.add(idx)
            selected.append(items[idx])

    selected.sort(key=lambda row: (str(row.get("clip_id") or ""), int(row.get("frame_idx") or 0), row_intent(row)))
    return selected[:max_samples]


def compact_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    route = row.get("route_context") if isinstance(row.get("route_context"), dict) else {}
    policy = row.get("trajectory_policy") if isinstance(row.get("trajectory_policy"), dict) else {}
    return {
        "index": index,
        "id": row.get("id"),
        "clip_id": row.get("clip_id"),
        "frame_idx": row.get("frame_idx"),
        "frame_name": row.get("frame_name"),
        "intent": row_intent(row),
        "route_theta_deg": route.get("route_theta_deg", meta.get("route_theta_deg")),
        "anchor": route.get("anchor"),
        "speed_policy": policy.get("speed_policy"),
        "trajectory_source": meta.get("trajectory_source", policy.get("trajectory_source")),
        "timestamp": meta.get("timestamp", row.get("timestamp")),
        "bev_frame_idx": meta.get("bev_frame_idx"),
        "bev_time_delta_ms": meta.get("bev_time_delta_ms"),
        "fpv": file_uri(row.get("fpv_image_path")),
        "bev": file_uri(row.get("bev_image_path")),
        "trajectory_label_path": str(row.get("trajectory_label_path") or ""),
        "bev_patch_meta_path": str(row.get("bev_patch_meta_path") or ""),
        "pixel_trajectory": row.get("pixel_trajectory") or [],
        "trajectory_fpv_t": row.get("trajectory_fpv_t") or [],
        "trajectory_ego_t": row.get("trajectory_ego_t") or [],
    }


def render_html(samples: list[dict[str, Any]], source_jsonl: Path) -> str:
    data = json.dumps(samples, ensure_ascii=False)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SideWalk Phase1 Source Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ position: sticky; top: 0; z-index: 3; background: #ffffff; border-bottom: 1px solid #d9dee7; padding: 10px 16px; }}
    .bar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    button, select, input {{ height: 32px; border: 1px solid #b7c0ce; border-radius: 6px; background: #fff; padding: 0 10px; }}
    button {{ cursor: pointer; }}
    main {{ padding: 14px 16px 24px; }}
    .layout {{ display: grid; grid-template-columns: minmax(420px, 1fr) minmax(360px, 0.8fr); gap: 14px; align-items: start; }}
    .panel {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }}
    .panel h2 {{ font-size: 14px; margin: 0; padding: 9px 12px; border-bottom: 1px solid #e5e9f0; }}
    .media {{ padding: 10px; }}
    .fpv-wrap {{ position: relative; width: 100%; background: #111; align-self: start; line-height: 0; }}
    .fpv-wrap img, .bev img {{ display: block; width: 100%; height: auto; }}
    .fpv-wrap canvas {{ position: absolute; left: 0; top: 0; pointer-events: none; }}
    .bev {{ background: #111; }}
    .meta {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 10px; font-size: 13px; }}
    .kv {{ border: 1px solid #e4e8ef; border-radius: 6px; padding: 7px 8px; min-width: 0; }}
    .kv b {{ display: block; font-size: 11px; color: #5b6676; margin-bottom: 3px; }}
    .kv span {{ overflow-wrap: anywhere; }}
    pre {{ margin: 0; padding: 10px; overflow: auto; max-height: 310px; background: #0f1720; color: #e8edf3; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e9f0; padding: 6px 8px; text-align: left; }}
    tr {{ cursor: pointer; }}
    tr.active {{ background: #e9f2ff; }}
    .list {{ max-height: calc(100vh - 430px); overflow: auto; }}
    .intent {{ font-weight: 700; }}
    .intent.turn_left, .intent.turn_right {{ color: #0b7285; }}
    .intent.go_straight {{ color: #2f6f3e; }}
    .intent.stop {{ color: #a14600; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} .list {{ max-height: 360px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <strong>SideWalk Phase1 Source Review</strong>
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <label>Intent <select id="intentFilter"><option value="">all</option></select></label>
      <label>Index <input id="indexInput" type="number" min="0" value="0" style="width:88px"></label>
      <span id="counter"></span>
    </div>
    <div style="font-size:12px;color:#5b6676;margin-top:6px">source: {source_jsonl}</div>
  </header>
  <main>
    <div class="layout">
      <section class="panel">
        <h2>FPV + GT Trajectory</h2>
        <div class="media">
          <div class="fpv-wrap">
            <img id="fpvImg" alt="FPV">
            <canvas id="trajCanvas"></canvas>
          </div>
        </div>
      </section>
      <section class="panel">
        <h2>BEV</h2>
        <div class="media bev"><img id="bevImg" alt="BEV"></div>
      </section>
      <section class="panel">
        <h2>Metadata</h2>
        <div id="meta" class="meta"></div>
        <pre id="jsonBox"></pre>
      </section>
      <section class="panel">
        <h2>Samples</h2>
        <div class="list">
          <table>
            <thead><tr><th>#</th><th>intent</th><th>clip</th><th>frame</th></tr></thead>
            <tbody id="sampleRows"></tbody>
          </table>
        </div>
      </section>
    </div>
  </main>
  <script>
    const samples = {data};
    let filtered = samples.slice();
    let current = 0;

    const fpvImg = document.getElementById('fpvImg');
    const bevImg = document.getElementById('bevImg');
    const canvas = document.getElementById('trajCanvas');
    const meta = document.getElementById('meta');
    const jsonBox = document.getElementById('jsonBox');
    const rows = document.getElementById('sampleRows');
    const counter = document.getElementById('counter');
    const intentFilter = document.getElementById('intentFilter');
    const indexInput = document.getElementById('indexInput');

    function cls(intent) {{ return String(intent || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_'); }}
    function esc(text) {{ return String(text ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}

    function drawTrajectory() {{
      const s = filtered[current];
      const rect = fpvImg.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = `${{rect.width}}px`;
      canvas.style.height = `${{rect.height}}px`;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const pts = (s && s.pixel_trajectory) || [];
      if (!pts.length) return;
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#00e676';
      ctx.fillStyle = '#00e676';
      ctx.beginPath();
      pts.forEach((p, i) => {{
        const x = Number(p[0]) * rect.width;
        const y = Number(p[1]) * rect.height;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      pts.forEach((p, i) => {{
        const x = Number(p[0]) * rect.width;
        const y = Number(p[1]) * rect.height;
        ctx.beginPath();
        ctx.arc(x, y, i === pts.length - 1 ? 4 : 2.7, 0, Math.PI * 2);
        ctx.fill();
      }});
    }}

    function renderRows() {{
      rows.innerHTML = filtered.map((s, i) => `
        <tr data-i="${{i}}" class="${{i === current ? 'active' : ''}}">
          <td>${{i}}</td>
          <td><span class="intent ${{cls(s.intent)}}">${{esc(s.intent)}}</span></td>
          <td>${{esc(s.clip_id)}}</td>
          <td>${{esc(s.frame_name || s.frame_idx)}}</td>
        </tr>`).join('');
      rows.querySelectorAll('tr').forEach(tr => tr.addEventListener('click', () => show(Number(tr.dataset.i))));
    }}

    function renderIntentOptions() {{
      const intents = Array.from(new Set(samples.map(s => s.intent))).sort();
      intentFilter.innerHTML = '<option value="">all</option>' + intents.map(x => `<option value="${{esc(x)}}">${{esc(x)}}</option>`).join('');
    }}

    function renderMeta(s) {{
      const items = [
        ['id', s.id], ['intent', s.intent], ['clip', s.clip_id], ['frame', s.frame_name || s.frame_idx],
        ['theta_deg', s.route_theta_deg], ['anchor', s.anchor], ['speed_policy', s.speed_policy], ['trajectory_source', s.trajectory_source],
        ['bev_frame_idx', s.bev_frame_idx], ['bev_delta_ms', s.bev_time_delta_ms], ['label', s.trajectory_label_path]
      ];
      meta.innerHTML = items.map(([k, v]) => `<div class="kv"><b>${{esc(k)}}</b><span>${{esc(v)}}</span></div>`).join('');
    }}

    function show(i) {{
      if (!filtered.length) return;
      current = Math.max(0, Math.min(i, filtered.length - 1));
      const s = filtered[current];
      fpvImg.src = s.fpv;
      bevImg.src = s.bev;
      renderMeta(s);
      jsonBox.textContent = JSON.stringify({{
        pixel_trajectory: s.pixel_trajectory,
        trajectory_fpv_t: s.trajectory_fpv_t,
        trajectory_ego_t: s.trajectory_ego_t
      }}, null, 2);
      counter.textContent = `${{current + 1}} / ${{filtered.length}}`;
      indexInput.value = current;
      renderRows();
      fpvImg.onload = drawTrajectory;
      setTimeout(drawTrajectory, 50);
    }}

    function applyFilter() {{
      const intent = intentFilter.value;
      filtered = intent ? samples.filter(s => s.intent === intent) : samples.slice();
      current = 0;
      show(0);
    }}

    document.getElementById('prevBtn').onclick = () => show(current - 1);
    document.getElementById('nextBtn').onclick = () => show(current + 1);
    intentFilter.onchange = applyFilter;
    indexInput.onchange = () => show(Number(indexInput.value || 0));
    window.addEventListener('resize', drawTrajectory);
    document.addEventListener('keydown', e => {{
      if (e.key === 'ArrowLeft') show(current - 1);
      if (e.key === 'ArrowRight') show(current + 1);
    }});

    renderIntentOptions();
    renderRows();
    show(0);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static Phase1 source review HTML")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--per-intent", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_jsonl = Path(args.source_jsonl)
    output_html = Path(args.output_html)
    rows = read_jsonl(source_jsonl)
    selected = pick_balanced(rows, max_samples=int(args.max_samples), per_intent=int(args.per_intent))
    samples = [compact_row(row, index=i) for i, row in enumerate(selected)]
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(render_html(samples, source_jsonl), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_jsonl": str(source_jsonl),
                "output_html": str(output_html),
                "input_rows": len(rows),
                "review_samples": len(samples),
                "intent_counts": {intent: sum(1 for row in samples if row["intent"] == intent) for intent in sorted({row["intent"] for row in samples})},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
