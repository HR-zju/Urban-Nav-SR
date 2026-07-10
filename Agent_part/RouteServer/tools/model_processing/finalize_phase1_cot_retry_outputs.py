#!/usr/bin/env python3
"""Finalize SideWalk Phase1 CoT outputs after retrying dropped rows.

The script merges original and retry CoT rows by sample id, preserving source
JSONL order, and builds a compact review HTML for samples that are still
missing after the retry.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def row_intent(row: dict[str, Any]) -> str:
    route = row.get("route_context")
    if isinstance(route, dict) and route.get("intent_gt"):
        return str(route["intent_gt"])
    meta = row.get("sidewalk_meta")
    if isinstance(meta, dict) and meta.get("intent_gt"):
        return str(meta["intent_gt"])
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


def error_key(error: str) -> str:
    for sep in (":", "="):
        if sep in error:
            return error.split(sep, 1)[0]
    return error


def load_cot_rows(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], int, int]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    total_rows = 0
    duplicate_rows = 0
    for path in paths:
        for row in read_jsonl(path):
            total_rows += 1
            sample_id = str(row.get("id") or "")
            if not sample_id:
                continue
            if sample_id in rows_by_id:
                duplicate_rows += 1
                continue
            rows_by_id[sample_id] = row
    return rows_by_id, total_rows, duplicate_rows


def load_state_by_id(path: Path) -> dict[str, dict[str, Any]]:
    state_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("id") or row.get("row_key") or "")
        if sample_id:
            state_by_id[sample_id] = row
    return state_by_id


def compact_failure_row(row: dict[str, Any], *, dataset: str, state: dict[str, Any], index: int) -> dict[str, Any]:
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    route = row.get("route_context") if isinstance(row.get("route_context"), dict) else {}
    policy = row.get("trajectory_policy") if isinstance(row.get("trajectory_policy"), dict) else {}
    errors = [str(item) for item in state.get("errors") or []]
    return {
        "index": index,
        "dataset": dataset,
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
        "fpv": file_uri(row.get("fpv_image_path")),
        "bev": file_uri(row.get("bev_image_path")),
        "pixel_trajectory": row.get("pixel_trajectory") or [],
        "trajectory_fpv_t": row.get("trajectory_fpv_t") or [],
        "trajectory_ego_t": row.get("trajectory_ego_t") or [],
        "retry_status": state.get("status", "missing_state"),
        "retry_errors": errors,
        "primary_error": error_key(errors[0]) if errors else "missing_state",
    }


def render_failure_html(samples: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    data = json.dumps(samples, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SideWalk Phase1 Final Dropped Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #17202a; background: #f5f7fa; }}
    header {{ position: sticky; top: 0; z-index: 3; background: #fff; border-bottom: 1px solid #d9dee7; padding: 10px 14px; }}
    .bar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    button, select, input {{ height: 32px; border: 1px solid #b7c0ce; border-radius: 6px; background: #fff; padding: 0 10px; }}
    button {{ cursor: pointer; }}
    main {{ padding: 14px; }}
    .layout {{ display: grid; grid-template-columns: minmax(420px, 1fr) minmax(360px, 0.82fr); gap: 14px; align-items: start; }}
    .panel {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }}
    .panel h2 {{ margin: 0; padding: 9px 12px; font-size: 14px; border-bottom: 1px solid #e5e9f0; }}
    .media {{ padding: 10px; }}
    .fpv-wrap {{ position: relative; width: 100%; background: #111; line-height: 0; }}
    .fpv-wrap img, .bev img {{ display: block; width: 100%; height: auto; }}
    .fpv-wrap canvas {{ position: absolute; left: 0; top: 0; pointer-events: none; }}
    .bev {{ background: #111; }}
    .meta {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 10px; font-size: 13px; }}
    .kv {{ border: 1px solid #e4e8ef; border-radius: 6px; padding: 7px 8px; min-width: 0; }}
    .kv b {{ display: block; font-size: 11px; color: #5b6676; margin-bottom: 3px; }}
    .kv span {{ overflow-wrap: anywhere; }}
    pre {{ margin: 0; padding: 10px; overflow: auto; max-height: 330px; background: #0f1720; color: #e8edf3; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e9f0; padding: 6px 8px; text-align: left; vertical-align: top; }}
    tr {{ cursor: pointer; }}
    tr.active {{ background: #e9f2ff; }}
    .list {{ max-height: calc(100vh - 460px); overflow: auto; }}
    .tag {{ display: inline-block; border-radius: 999px; padding: 2px 7px; background: #eef2f7; font-weight: 650; }}
    .err {{ color: #9b1c1c; font-weight: 650; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} .list {{ max-height: 380px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <strong>SideWalk Phase1 Final Dropped Review</strong>
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <label>Dataset <select id="datasetFilter"><option value="">all</option></select></label>
      <label>Intent <select id="intentFilter"><option value="">all</option></select></label>
      <label>Error <select id="errorFilter"><option value="">all</option></select></label>
      <label>Index <input id="indexInput" type="number" min="0" value="0" style="width:88px"></label>
      <span id="counter"></span>
    </div>
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
        <h2>Failure Metadata</h2>
        <div id="meta" class="meta"></div>
        <pre id="jsonBox"></pre>
      </section>
      <section class="panel">
        <h2>Samples</h2>
        <div class="list">
          <table>
            <thead><tr><th>#</th><th>dataset</th><th>intent</th><th>error</th><th>clip/frame</th></tr></thead>
            <tbody id="sampleRows"></tbody>
          </table>
        </div>
      </section>
      <section class="panel" style="grid-column: 1 / -1">
        <h2>Summary</h2>
        <pre>{html.escape(summary_json)}</pre>
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
    const datasetFilter = document.getElementById('datasetFilter');
    const intentFilter = document.getElementById('intentFilter');
    const errorFilter = document.getElementById('errorFilter');
    const indexInput = document.getElementById('indexInput');

    function esc(text) {{ return String(text ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
    function unique(key) {{ return Array.from(new Set(samples.map(s => s[key]).filter(Boolean))).sort(); }}

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
          <td><span class="tag">${{esc(s.dataset)}}</span></td>
          <td>${{esc(s.intent)}}</td>
          <td class="err">${{esc(s.primary_error)}}</td>
          <td>${{esc(s.clip_id)}} / ${{esc(s.frame_name || s.frame_idx)}}</td>
        </tr>`).join('');
      rows.querySelectorAll('tr').forEach(tr => tr.addEventListener('click', () => show(Number(tr.dataset.i))));
    }}

    function renderOptions() {{
      for (const [select, key] of [[datasetFilter, 'dataset'], [intentFilter, 'intent'], [errorFilter, 'primary_error']]) {{
        select.innerHTML = '<option value="">all</option>' + unique(key).map(x => `<option value="${{esc(x)}}">${{esc(x)}}</option>`).join('');
      }}
    }}

    function renderMeta(s) {{
      const items = [
        ['id', s.id], ['dataset', s.dataset], ['intent', s.intent], ['clip', s.clip_id],
        ['frame', s.frame_name || s.frame_idx], ['theta_deg', s.route_theta_deg], ['anchor', s.anchor],
        ['speed_policy', s.speed_policy], ['trajectory_source', s.trajectory_source], ['retry_status', s.retry_status],
        ['primary_error', s.primary_error], ['errors', (s.retry_errors || []).join(' | ')]
      ];
      meta.innerHTML = items.map(([k, v]) => `<div class="kv"><b>${{esc(k)}}</b><span>${{esc(v)}}</span></div>`).join('');
    }}

    function show(i) {{
      if (!filtered.length) {{
        counter.textContent = '0 / 0';
        return;
      }}
      current = Math.max(0, Math.min(i, filtered.length - 1));
      const s = filtered[current];
      fpvImg.src = s.fpv;
      bevImg.src = s.bev;
      renderMeta(s);
      jsonBox.textContent = JSON.stringify({{
        retry_errors: s.retry_errors,
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
      const dataset = datasetFilter.value;
      const intent = intentFilter.value;
      const error = errorFilter.value;
      filtered = samples.filter(s => (!dataset || s.dataset === dataset) && (!intent || s.intent === intent) && (!error || s.primary_error === error));
      current = 0;
      renderRows();
      show(0);
    }}

    document.getElementById('prevBtn').onclick = () => show(current - 1);
    document.getElementById('nextBtn').onclick = () => show(current + 1);
    datasetFilter.onchange = applyFilter;
    intentFilter.onchange = applyFilter;
    errorFilter.onchange = applyFilter;
    indexInput.onchange = () => show(Number(indexInput.value || 0));
    window.addEventListener('resize', drawTrajectory);
    document.addEventListener('keydown', e => {{
      if (e.key === 'ArrowLeft') show(current - 1);
      if (e.key === 'ArrowRight') show(current + 1);
    }});
    renderOptions();
    renderRows();
    show(0);
  </script>
</body>
</html>
"""


def finalize_dataset(
    *,
    name: str,
    source_jsonl: Path,
    cot_paths: list[Path],
    retry_state_jsonl: Path,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    cot_by_id, cot_total_rows, duplicate_cot_rows = load_cot_rows(cot_paths)
    retry_state_by_id = load_state_by_id(retry_state_jsonl)

    output_jsonl = output_dir / "manifests" / f"{output_prefix}.jsonl"
    source_clean_jsonl = output_dir / "manifests" / f"{output_prefix.replace('phase1_cot_', 'phase1_source_', 1)}.jsonl"
    missing_source_jsonl = output_dir / "manifests" / f"{output_prefix}.final_dropped_source.jsonl"
    missing_state_jsonl = output_dir / "manifests" / f"{output_prefix}.final_dropped_state.jsonl"

    source_rows = 0
    written_rows = 0
    missing_rows: list[dict[str, Any]] = []
    missing_states: list[dict[str, Any]] = []
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as out_f, source_clean_jsonl.open("w", encoding="utf-8") as source_f:
        for source_row in read_jsonl(source_jsonl):
            source_rows += 1
            sample_id = str(source_row.get("id") or "")
            cot_row = cot_by_id.get(sample_id)
            if cot_row is None:
                state = retry_state_by_id.get(sample_id, {})
                enriched_source = dict(source_row)
                enriched_source["final_cot_status"] = "dropped"
                enriched_source["final_cot_errors"] = list(state.get("errors") or ["missing_cot_row"])
                missing_rows.append(enriched_source)
                missing_states.append(
                    {
                        "dataset": name,
                        "id": sample_id,
                        "status": state.get("status", "missing_cot_row"),
                        "errors": list(state.get("errors") or ["missing_cot_row"]),
                        "retry_state": state,
                    }
                )
                continue
            out_f.write(json.dumps(cot_row, ensure_ascii=False) + "\n")
            source_f.write(json.dumps(source_row, ensure_ascii=False) + "\n")
            written_rows += 1

    write_jsonl(missing_source_jsonl, missing_rows)
    write_jsonl(missing_state_jsonl, missing_states)

    error_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    for row in missing_rows:
        intent_counts[row_intent(row)] += 1
    for state in missing_states:
        for error in state.get("errors") or []:
            error_counts[error_key(str(error))] += 1

    return {
        "dataset": name,
        "source_jsonl": str(source_jsonl),
        "source_rows": source_rows,
        "cot_files": [str(path) for path in cot_paths],
        "cot_total_rows": cot_total_rows,
        "cot_unique_rows": len(cot_by_id),
        "duplicate_cot_rows": duplicate_cot_rows,
        "written_rows": written_rows,
        "missing_rows": len(missing_rows),
        "usable_rate": (written_rows / source_rows) if source_rows else 0.0,
        "output_jsonl": str(output_jsonl),
        "source_clean_jsonl": str(source_clean_jsonl),
        "missing_source_jsonl": str(missing_source_jsonl),
        "missing_state_jsonl": str(missing_state_jsonl),
        "missing_error_counts": dict(error_counts.most_common()),
        "missing_intent_counts": dict(intent_counts.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frodobots-source-jsonl", required=True)
    parser.add_argument("--foxglove-source-jsonl", required=True)
    parser.add_argument("--frodobots-original-cot-jsonl", required=True)
    parser.add_argument("--frodobots-retry-cot-jsonl", required=True)
    parser.add_argument("--frodobots-retry-state-jsonl", required=True)
    parser.add_argument("--foxglove-original-cot-jsonl", required=True)
    parser.add_argument("--foxglove-retry-cot-jsonl", required=True)
    parser.add_argument("--foxglove-retry-state-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--review-html", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    datasets = [
        (
            "frodobots_qc",
            Path(args.frodobots_source_jsonl),
            [Path(args.frodobots_original_cot_jsonl), Path(args.frodobots_retry_cot_jsonl)],
            Path(args.frodobots_retry_state_jsonl),
            output_root / "frodobots_qc",
            "phase1_cot_frodobots_qc_lite_cot_final",
        ),
        (
            "foxglove_yaw_p90_clean",
            Path(args.foxglove_source_jsonl),
            [Path(args.foxglove_original_cot_jsonl), Path(args.foxglove_retry_cot_jsonl)],
            Path(args.foxglove_retry_state_jsonl),
            output_root / "foxglove_yaw_p90_clean",
            "phase1_cot_foxglove_yaw_p90_clean_lite_cot_final",
        ),
    ]

    reports: list[dict[str, Any]] = []
    for dataset in datasets:
        reports.append(
            finalize_dataset(
                name=dataset[0],
                source_jsonl=dataset[1],
                cot_paths=dataset[2],
                retry_state_jsonl=dataset[3],
                output_dir=dataset[4],
                output_prefix=dataset[5],
            )
        )

    combined_manifest = output_root / "manifests" / "phase1_cot_frodobots_foxglove_lite_cot_final.jsonl"
    combined_source_clean = output_root / "manifests" / "phase1_source_frodobots_foxglove_lite_cot_final_clean.jsonl"
    combined_manifest.parent.mkdir(parents=True, exist_ok=True)
    combined_rows = 0
    combined_source_rows = 0
    with combined_manifest.open("w", encoding="utf-8") as out_f:
        for report in reports:
            for row in read_jsonl(Path(str(report["output_jsonl"]))):
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                combined_rows += 1
    with combined_source_clean.open("w", encoding="utf-8") as out_f:
        for report in reports:
            for row in read_jsonl(Path(str(report["source_clean_jsonl"]))):
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                combined_source_rows += 1

    failed_source_jsonl = output_root / "manifests" / "phase1_cot_frodobots_foxglove_lite_cot_final_dropped_source.jsonl"
    failed_state_jsonl = output_root / "manifests" / "phase1_cot_frodobots_foxglove_lite_cot_final_dropped_state.jsonl"
    failed_rows: list[dict[str, Any]] = []
    failed_states: list[dict[str, Any]] = []
    review_samples: list[dict[str, Any]] = []
    for report in reports:
        dataset_name = str(report["dataset"])
        states_by_id = {
            str(state.get("id") or ""): state
            for state in read_jsonl(Path(str(report["missing_state_jsonl"])))
        }
        for row in read_jsonl(Path(str(report["missing_source_jsonl"]))):
            failed_rows.append(row)
            state = states_by_id.get(str(row.get("id") or ""), {})
            failed_states.append(state)
            review_samples.append(
                compact_failure_row(
                    row,
                    dataset=dataset_name,
                    state=state,
                    index=len(review_samples),
                )
            )
    write_jsonl(failed_source_jsonl, failed_rows)
    write_jsonl(failed_state_jsonl, failed_states)

    combined_error_counts: Counter[str] = Counter()
    for state in failed_states:
        for error in state.get("errors") or []:
            combined_error_counts[error_key(str(error))] += 1

    summary = {
        "schema_version": "sidewalk_phase1_cot_final_retry_merge_v1",
        "output_root": str(output_root),
        "datasets": reports,
        "combined": {
            "source_rows": sum(int(report["source_rows"]) for report in reports),
            "written_rows": combined_rows,
            "missing_rows": len(failed_rows),
            "usable_rate": (combined_rows / sum(int(report["source_rows"]) for report in reports)) if reports else 0.0,
            "output_jsonl": str(combined_manifest),
            "source_clean_jsonl": str(combined_source_clean),
            "source_clean_rows": combined_source_rows,
            "missing_source_jsonl": str(failed_source_jsonl),
            "missing_state_jsonl": str(failed_state_jsonl),
            "missing_error_counts": dict(combined_error_counts.most_common()),
        },
    }

    review_html = Path(args.review_html) if args.review_html else output_root / "review" / "phase1_cot_final_dropped_review.html"
    review_html.parent.mkdir(parents=True, exist_ok=True)
    review_html.write_text(render_failure_html(review_samples, summary), encoding="utf-8")
    summary["combined"]["review_html"] = str(review_html)

    summary_json = output_root / "manifests" / "phase1_cot_frodobots_foxglove_lite_cot_final_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
