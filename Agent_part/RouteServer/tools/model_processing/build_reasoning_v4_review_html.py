#!/usr/bin/env python3
"""Build a self-contained human-review page for Reasoning V4 artifacts."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mcp_server.data.jsonl_io import iter_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Reasoning V4 review-builder arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--views-jsonl", type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--title", default="UrbanNav Reasoning V4.2 Review")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Link local files instead of embedding image bytes.",
    )
    return parser.parse_args(argv)


def build_reasoning_v4_review_html(
    *,
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_html: str | Path,
    views_jsonl: str | Path | None = None,
    report_json: str | Path | None = None,
    title: str = "UrbanNav Reasoning V4.2 Review",
    max_samples: int = 100,
    embed_images: bool = True,
) -> dict[str, Any]:
    """Join V4 sources, latest annotations, and optional exported views.

    The page deliberately displays rejected and provider-error samples. This
    makes an annotation smoke review useful even when no trainable view can be
    exported.
    """

    sources = _index_unique(
        iter_jsonl(source_jsonl),
        key_path=("observation", "source_group_id"),
    )
    annotations = _index_latest(
        iter_jsonl(annotation_jsonl),
        key_path=("source_group_id",),
    )
    views_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if views_jsonl is not None and Path(views_jsonl).expanduser().is_file():
        for view in iter_jsonl(views_jsonl):
            source_id = str(view.get("source_group_id") or "")
            if source_id:
                views_by_source[source_id].append(view)

    selected_ids = list(sources)[: max(0, int(max_samples))]
    status_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    image_count = 0
    cards: list[str] = []
    for source_id in selected_ids:
        source = sources[source_id]
        annotation = annotations.get(source_id)
        status = str((annotation or {}).get("status") or "missing")
        status_counts[status] += 1
        observation = source.get("observation") or {}
        image_count += len(observation.get("images") or [])
        for name, record in ((annotation or {}).get("stage_records") or {}).items():
            stage_counts[f"{name}:{record.get('status', 'missing')}"] += 1
        cards.append(
            _render_sample(
                source=source,
                annotation=annotation,
                views=views_by_source.get(source_id, []),
                embed_images=embed_images,
            )
        )

    view_count = sum(len(views_by_source[item]) for item in selected_ids)
    summary: dict[str, Any] = {
        "schema_version": "urbannav_reasoning_v4_2_review_report_v1",
        "output_html": str(Path(output_html).expanduser()),
        "source_rows": len(selected_ids),
        "annotation_status_counts": dict(status_counts),
        "stage_status_counts": dict(stage_counts),
        "rendered_images": image_count,
        "rendered_views": view_count,
        "embedded_images": bool(embed_images),
    }
    page = _render_page(
        title=title,
        cards=cards,
        summary=summary,
        provenance={
            "source_jsonl": str(source_jsonl),
            "annotation_jsonl": str(annotation_jsonl),
            "views_jsonl": str(views_jsonl) if views_jsonl else None,
        },
    )
    destination = Path(output_html).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    if report_json is not None:
        report = Path(report_json).expanduser()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Build a standalone V4 review page and print its summary."""

    args = parse_args(argv)
    result = build_reasoning_v4_review_html(
        source_jsonl=args.source_jsonl,
        annotation_jsonl=args.annotation_jsonl,
        views_jsonl=args.views_jsonl,
        output_html=args.output_html,
        report_json=args.report_json,
        title=args.title,
        max_samples=args.max_samples,
        embed_images=not args.link_images,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _index_unique(
    rows: Iterable[dict[str, Any]],
    *,
    key_path: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Index rows by a required nested key and reject duplicates."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(_nested(row, key_path) or "").strip()
        if not key:
            raise ValueError(f"row is missing key path {key_path}")
        if key in result:
            raise ValueError(f"duplicate source id: {key}")
        result[key] = row
    return result


def _index_latest(
    rows: Iterable[dict[str, Any]],
    *,
    key_path: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Index append-only rows, keeping the latest value per key."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(_nested(row, key_path) or "").strip()
        if key:
            result[key] = row
    return result


def _nested(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Read a nested mapping path without raising on absent fields."""

    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _render_page(
    *,
    title: str,
    cards: list[str],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    """Render the complete self-contained HTML document."""

    counts = summary["annotation_status_counts"]
    metrics = (
        ("Sources", summary["source_rows"]),
        ("Accepted", counts.get("accepted", 0)),
        ("Failed/partial", sum(value for key, value in counts.items() if key != "accepted")),
        ("Images", summary["rendered_images"]),
        ("Views", summary["rendered_views"]),
    )
    metric_html = "".join(
        f"<div class='metric'><strong>{_e(value)}</strong><span>{_e(label)}</span></div>"
        for label, value in metrics
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title>
<style>
:root{{--ink:#17212b;--muted:#64717e;--line:#d6dde5;--soft:#f3f6f8;--ok:#18794e;--bad:#b42318;--warn:#9a6700}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf1f4;color:var(--ink);font:14px/1.5 Inter,system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:4;padding:14px 22px;background:#fffffff5;border-bottom:1px solid var(--line)}}
.head{{max-width:1540px;margin:auto;display:flex;gap:20px;justify-content:space-between;align-items:center}}h1{{font-size:20px;margin:0}}
.muted{{color:var(--muted);font-size:12px;overflow-wrap:anywhere}}input,select{{height:34px;border:1px solid #aab6c2;background:white;padding:0 10px}}
main{{max-width:1540px;margin:18px auto;padding:0 18px 60px}}.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:14px}}
.metric,.sample{{background:white;border:1px solid var(--line);border-radius:7px}}.metric{{padding:10px 13px}}.metric strong{{display:block;font-size:21px}}.metric span{{font-size:12px;color:var(--muted)}}
.sample{{margin-bottom:18px;overflow:hidden}}.sample-head{{padding:13px 16px;background:#f8fafb;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:18px}}
h2{{font-size:16px;margin:0 0 3px}}.badge{{display:inline-block;border:1px solid;padding:2px 7px;font-size:11px;font-weight:700;background:#fff}}
.accepted{{color:var(--ok);border-color:#8bc9ad;background:#edf8f2}}.error,.rejected,.partial{{color:var(--bad);border-color:#e2a4a4;background:#fff1f1}}
.missing{{color:var(--warn);border-color:#dbc27b;background:#fff8df}}.body{{padding:15px 16px}}.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:10px}}
figure{{margin:0;background:#10161c;border:1px solid var(--line)}}figure img{{width:100%;height:310px;display:block;object-fit:contain}}figcaption{{padding:6px 8px;color:#fff;font-size:11px;overflow-wrap:anywhere}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(380px,.7fr);gap:14px;margin-top:14px}}section{{border-top:1px solid var(--line);padding:12px 0}}section:first-child{{border-top:0}}
h3{{margin:0 0 7px;font-size:12px;text-transform:uppercase;color:#3d4a57}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--soft);border:1px solid #e0e6eb;padding:9px 11px;margin:0;font:12px/1.45 ui-monospace,monospace}}
table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{text-align:left;vertical-align:top;padding:7px;border-bottom:1px solid var(--line)}}th{{background:var(--soft)}}details{{margin-top:8px}}summary{{cursor:pointer;font-weight:700;color:#1769aa}}.hidden{{display:none!important}}
@media(max-width:900px){{header{{position:static}}.head,.grid{{display:block}}.metrics{{grid-template-columns:repeat(2,1fr)}}figure img{{height:auto;max-height:420px}}}}
</style></head><body>
<header><div class="head"><div><h1>{_e(title)}</h1><div class="muted">{_e(json.dumps(provenance, ensure_ascii=False))}</div></div>
<div><input id="q" type="search" placeholder="Filter id, output, error"><select id="s"><option value="">All status</option><option>accepted</option><option>partial</option><option>rejected</option><option>error</option><option>missing</option></select></div></div></header>
<main><div class="metrics">{metric_html}</div>{''.join(cards) or '<p>No samples.</p>'}</main>
<script>const q=document.querySelector('#q'),s=document.querySelector('#s');function f(){{const a=q.value.toLowerCase(),b=s.value;document.querySelectorAll('.sample').forEach(x=>x.classList.toggle('hidden',(a&&!x.dataset.search.includes(a))||(b&&x.dataset.status!==b)))}}q.addEventListener('input',f);s.addEventListener('change',f);</script>
</body></html>"""


def _render_sample(
    *,
    source: Mapping[str, Any],
    annotation: Mapping[str, Any] | None,
    views: list[dict[str, Any]],
    embed_images: bool,
) -> str:
    """Render one source group with every independent stage record."""

    observation = source.get("observation") or {}
    source_id = str(observation.get("source_group_id") or "unknown")
    status = str((annotation or {}).get("status") or "missing")
    figures = _render_images(observation, embed_images=embed_images)
    stage_rows = []
    for name, record in ((annotation or {}).get("stage_records") or {}).items():
        raw = record.get("raw_answer")
        details = (
            f"<details><summary>raw answer</summary>{_json_pre(raw)}</details>"
            if raw is not None
            else ""
        )
        stage_rows.append(
            "<tr>"
            f"<td>{_e(name)}</td><td>{_e(record.get('status'))}</td>"
            f"<td>{_e(record.get('attempts'))}</td>"
            f"<td>{_json_pre({'errors': record.get('errors') or [], 'warnings': record.get('warnings') or []})}{details}</td>"
            "</tr>"
        )
    stage_table = (
        "<table><thead><tr><th>Stage</th><th>Status</th><th>Calls</th><th>Diagnostics / raw</th></tr></thead>"
        f"<tbody>{''.join(stage_rows)}</tbody></table>"
        if stage_rows
        else "<p class='muted'>No annotation stage records.</p>"
    )
    annotation_payload = annotation or {"status": "missing"}
    target_or_partial = {
        "grounding_outputs": annotation_payload.get("grounding_outputs"),
        "decision": annotation_payload.get("decision"),
        "target": annotation_payload.get("target"),
    }
    private_sidecars = {
        "direction_supervision": source.get("direction_supervision"),
        "trajectory_ego_t": source.get("trajectory_ego_t"),
        "trajectory_policy": source.get("trajectory_policy"),
        "action_mask": source.get("action_mask"),
    }
    visible = {
        "instruction": observation.get("instruction"),
        "input_modalities": observation.get("input_modalities"),
        "model_visible_context": observation.get("model_visible_context"),
        "evidence_catalog": observation.get("evidence_catalog"),
        "temporal_evidence_eligible": observation.get("temporal_evidence_eligible"),
    }
    searchable = json.dumps(
        {
            "id": source_id,
            "status": status,
            "annotation": annotation_payload,
        },
        ensure_ascii=False,
    ).lower()
    return f"""<article class="sample" data-status="{_e(status)}" data-search="{_e(searchable)}">
<div class="sample-head"><div><h2>{_e(source_id)}</h2><div class="muted">{_e(observation.get('dataset_source'))} · {_e(observation.get('source_record_id'))} · {_e(str(source.get('source_fingerprint') or '')[:12])}</div></div>
<div><span class="badge {_e(status)}">{_e(status)}</span> <span class="badge">{_e((annotation or {}).get('teacher_model') or 'no model')}</span> <span class="badge">{_e(len(views))} views</span></div></div>
<div class="body">{figures}<div class="grid"><div><section><h3>Visible input</h3>{_json_pre(visible)}</section><section><h3>Groundings, Decision, assembled target</h3>{_json_pre(target_or_partial)}</section></div>
<aside><section><h3>Independent stage audit</h3>{stage_table}</section><section><h3>Global errors / warnings</h3>{_json_pre({'errors': annotation_payload.get('errors') or [], 'warnings': annotation_payload.get('warnings') or []})}</section>
<section><h3>Private strong sidecars — never sent to teacher</h3>{_json_pre(private_sidecars)}</section><section><h3>Exported view identities</h3>{_json_pre([{'sample_id': v.get('sample_id'), 'view_type': v.get('view_type'), 'section_name': v.get('section_name')} for v in views])}</section></aside></div>
<details><summary>Raw joined artifacts</summary>{_json_pre({'source': source, 'annotation': annotation, 'views': views})}</details></div></article>"""


def _render_images(
    observation: Mapping[str, Any],
    *,
    embed_images: bool,
) -> str:
    """Render all ordered observation images with explicit evidence roles."""

    images = observation.get("images") or []
    roles = observation.get("image_roles") or []
    figures: list[str] = []
    for index, value in enumerate(images):
        role = roles[index] if index < len(roles) else f"image.{index}"
        source = _image_source(value, embed=embed_images)
        body = (
            f"<img src='{_e(source)}' alt='{_e(role)}'>"
            if source
            else f"<pre>Unavailable image: {_e(value)}</pre>"
        )
        figures.append(
            f"<figure>{body}<figcaption>{_e(role)} · {_e(value)}</figcaption></figure>"
        )
    return f"<div class='images'>{''.join(figures)}</div>"


def _image_source(value: Any, *, embed: bool) -> str:
    """Resolve an image reference as a data URL or local file URI."""

    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(("https://", "http://", "data:image/")):
        return text
    path = Path(text).expanduser()
    if not path.is_file():
        return ""
    if not embed:
        return path.resolve().as_uri()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _json_pre(value: Any) -> str:
    """Serialize and escape one JSON-compatible value."""

    return f"<pre>{_e(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"


def _e(value: Any) -> str:
    """Escape HTML text and attribute values."""

    return html.escape(str(value if value is not None else ""), quote=True)


if __name__ == "__main__":
    main()
